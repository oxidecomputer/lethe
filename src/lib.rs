#![cfg_attr(not(test), no_std)]

pub mod low_level;

use core::borrow::BorrowMut;
use crate::low_level::{Flash, Space};
use core::cell::RefCell;

pub struct Store<'b, F: Flash> {
    flash: F,
    buffers: RefCell<&'b mut StoreBuffers<F>>,
    current: Space,
    next_free: u32,
    incomplete_write: bool,
    tail_erased: bool,
    other_erased: bool,
}

/// Basic read-only store access.
impl<F: Flash> Store<'_, F> {
    /// Searches for the most recent data entry matching `key` and reads its
    /// value into `value_out`.
    ///
    /// At most `value_out.len()` bytes will be read; the actual value in the
    /// entry may be longer than this. (TODO: not great.)
    ///
    /// On success, returns `Ok(Some(bytes_read))`. If no matching entry is
    /// found, or if the key has been deleted, returns `Ok(None)`.
    ///
    /// Returns `Err` only if an error occurs accessing the flash dveice, or if
    /// the log is found to be corrupt.
    pub fn read_kv(
        &self,
        key: &[u8],
        value_out: &mut [u8],
    ) -> Result<Option<usize>, low_level::ReadError<F::Error>> {
        let mut b = self.buffers.borrow_mut();
        let loc = low_level::seek_kv_backwards(
            &self.flash,
            &mut b.b0,
            self.current,
            self.next_free,
            key,
        )?;

        if let Some(head_sector) = loc {
            let n = low_level::read_contents(
                &self.flash,
                &mut b.b0,
                self.current,
                head_sector,
                key.len() as u32,
                value_out,
            )?;
            Ok(Some(n))
        } else {
            Ok(None)
        }
    }

    /// Searches for the most recent data entry matching `key` and returns its
    /// location in the current space. This is mostly useful for interacting
    /// with lower level APIs.
    ///
    /// If found, returns `Ok(Some(sector_number))`. 
    ///
    /// If there is no entry for `key` or it has been deleted, returns
    /// `Ok(None)`.
    ///
    /// Returns `Err` only if an error occurs accessing the flash device, or if
    /// the log is found to be corrupt.
    pub fn locate_kv(
        &self,
        key: &[u8],
    ) -> Result<Option<u32>, low_level::ReadError<F::Error>> {
        let mut b = self.buffers.borrow_mut();
        low_level::seek_kv_backwards(
            &self.flash,
            &mut b.b0,
            self.current,
            self.next_free,
            key,
        )
    }
}

/// Accessing check results, repair, mounting writable.
impl<'b, F: Flash> Store<'b, F> {
    /// Checks whether the log check said we can mount this writable without
    /// further effort.
    pub fn can_mount_writable(&self) -> bool {
        self.incomplete_write == false
            && self.tail_erased == true
            && self.other_erased == true
    }

    /// Attempts to mount this writable without doing any repair actions. This
    /// will fail if the check found problems, which you can detect in advance
    /// by calling `can_mount_writable`.
    pub fn mount_writable(self) -> Result<WritableStore<'b, F>, Self> {
        if !self.can_mount_writable() {
            return Err(self);
        }
        
        Ok(WritableStore(self))
    }

    pub fn repair(&mut self) -> Result<(), RepairError<F::Error>> {
        if self.incomplete_write {
            let r = low_level::read_entry_from_head(
                &mut self.flash,
                &mut self.buffers.get_mut().b0,
                self.current,
                self.next_free,
            );
            use low_level::ReadError;
            let entry = match r {
                Err(ReadError::Flash(e)) => return Err(e.into()),
                Err(_) => return Err(RepairError::Corrupt),
                Ok(entry) => entry,
            };

            let mut meta = *entry.meta;
            meta.subtype = low_level::KnownSubtypes::Aborted as u8;
            meta.sub_bytes = 0;

            let trailer = entry.next_sector - 1;

            let buffer = &mut self.buffers.get_mut().b0;
            let b = (*buffer).borrow_mut();
            b.fill(0);
            *low_level::cast_suffix_mut(b).1 = meta;
            self.flash.program_sector(self.current, trailer, buffer)?;

            self.next_free = trailer + 1;
            self.incomplete_write = false;
        }

        if !self.other_erased {
            panic!("requires erase");
        }

        if !self.tail_erased {
            panic!("requires evacuation and erase");
        }

        Ok(())
    }
}

pub enum RepairError<E> {
    Corrupt,
    Flash(E),
}

impl<E> From<E> for RepairError<E> {
    fn from(e: E) -> Self {
        Self::Flash(e)
    }
}


pub struct WritableStore<'b, F: Flash>(Store<'b, F>);

impl<'b, F: Flash> core::ops::Deref for WritableStore<'b, F> {
    type Target = Store<'b, F>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<F: Flash> WritableStore<'_, F> {
    pub fn write_kv(
        &mut self,
        key: &[u8],
        value: &[u8],
    ) -> Result<(), low_level::WriteError<F::Error>> {
        low_level::write_kv(
            &mut self.0.flash,
            &mut self.0.buffers.get_mut().b0,
            self.0.current,
            self.0.next_free,
            key,
            value,
        )?;
        Ok(())
    }
}

#[derive(Default)]
pub struct StoreBuffers<F: Flash> {
    pub b0: F::Sector,
    pub b1: F::Sector,
}

pub fn mount<F: Flash>(mut flash: F, buffers: &mut StoreBuffers<F>) -> Result<Store<'_, F>, MountError<F>> {
    match mount_inner(&mut flash, buffers) {
        Err(cause) => Err(MountError { flash, cause }),
        Ok((current, next_free, incomplete_write, tail_erased, other_erased)) => {
            Ok(Store {
                flash,
                buffers: buffers.into(),
                current,
                next_free,
                incomplete_write,
                tail_erased,
                other_erased,
            })
        }
    }
}

fn mount_inner<F: Flash>(flash: &mut F, buffers: &mut StoreBuffers<F>) -> Result<(Space, u32, bool, bool, bool), MountErrorCause<F::Error>> {
    // Run log checks on both spaces.
    use low_level::CheckResult;
    let c0 = low_level::check(flash, &mut buffers.b0, Space::Zero)?;
    let c1 = low_level::check(flash, &mut buffers.b0, Space::One)?;

    let (current, other_erased) = match (&c0, &c1) {
        // If both spaces contain valid data, choose the one with the greater
        // generation number, using sequence number arithmetic.
        (CheckResult::ValidLog { generation: g0, .. }, CheckResult::ValidLog { generation: g1, .. }) => {
            use core::cmp::Ordering;
            match sequence_compare(*g0, *g1) {
                Ordering::Greater => (Space::Zero, false),
                Ordering::Less => (Space::One, false),
                Ordering::Equal => return Err(MountErrorCause::GenerationsAmbiguous),
            }
        }

        // If only one space is valid, choose it, obvs.
        (CheckResult::ValidLog { .. }, CheckResult::Bad(e)) => (
            Space::Zero,
            *e == low_level::CheckError::Erased,
        ),
        (CheckResult::Bad(e), CheckResult::ValidLog { .. }) => (
            Space::One,
            *e == low_level::CheckError::Erased,
        ),

        // If both spaces are bad... we can't proceed.
        (CheckResult::Bad(bad0), CheckResult::Bad(bad1)) => return Err(MountErrorCause::NoValidLogs(
            *bad0,
            *bad1,
        )),
    };

    let current_result = match current {
        Space::Zero => c0,
        Space::One => c1,
    };

    let (end, incomplete_write, tail_erased) = match current_result {
        CheckResult::ValidLog { end, incomplete_write, tail_erased, .. } => {
            (end, incomplete_write, tail_erased)
        }
        _ => unreachable!(),
    };

    Ok((current, end, incomplete_write, tail_erased, other_erased))
}

pub struct MountError<F: Flash> {
    flash: F,
    cause: MountErrorCause<F::Error>,
}

impl<F: Flash> MountError<F> {
    pub fn into_inner(self) -> F {
        self.flash
    }

    pub fn cause(&self) -> &MountErrorCause<F::Error> {
        &self.cause
    }
}

#[derive(Copy, Clone, Debug)]
pub enum MountErrorCause<E> {
    GenerationsAmbiguous,
    NoValidLogs(low_level::CheckError, low_level::CheckError),
    Flash(E),
}

impl<E> From<E> for MountErrorCause<E> {
    fn from(e: E) -> Self {
        Self::Flash(e)
    }
}

fn sequence_compare(a: u32, b: u32) -> core::cmp::Ordering {
    (a.wrapping_sub(b) as i32).cmp(&0)
}
