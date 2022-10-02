#![cfg_attr(not(test), no_std)]

pub mod low_level;

use core::mem::size_of;
use core::borrow::{Borrow, BorrowMut};
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

impl<'b, F: Flash> Store<'b, F> {
    pub fn can_mount_writable(&self) -> bool {
        self.incomplete_write == false
            && self.tail_erased == true
            && self.other_erased == true
    }

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

impl<F: Flash> Store<'_, F> {
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

    // TODO this should not be in the high level interface since it allows
    // arbitrary sector reads
    pub fn read_contents(
        &self,
        head_sector: u32,
        offset: u32,
        out: &mut [u8],
    ) -> Result<usize, low_level::ReadError<F::Error>> {
        // Read the head sector. (TODO: in the case where we're being called by
        // read_kv this is technically unnecessary; provide a path that can
        // reuse the trailer metadata that function has already read.)
        let mut b = self.buffers.borrow_mut();
        let entry = low_level::read_entry_from_head(
            &self.flash,
            &mut b.b0,
            self.current,
            head_sector,
        )?;

        // Work out the actual shape of the read.
        let overall_len = entry.meta.contents_length.get();
        // Allow the offset to be anywhere up to the end; reject offsets beyond
        // that as errors. Compute the remaining length past our offset.
        let offset_len = overall_len.checked_sub(offset)
            .ok_or(low_level::ReadError::End(overall_len))?;
        // We'll read all of the remaining contents, or as much as will fit into
        // `out`, whichever is shorter.
        let read_size = usize::min(offset_len as usize, out.len());
        let mut out = &mut out[..read_size];

        // Adjust the offset to account for the entry header and submeta.
        let mut offset = size_of::<low_level::EntryHeader>()
            + entry.meta.sub_bytes as usize
            + offset as usize;
        // Work out our sector address given that offset.
        let mut sector = head_sector as usize;

        // TODO given that sector size is compile-time constant this should
        // probably become div/mod.
        while offset >= size_of::<F::Sector>() {
            offset -= size_of::<F::Sector>();
            sector += 1;
        }

        // Transfer the data!
        while !out.is_empty() {
            // Each time into this loop we need to read `sector`. For the first
            // sector, `offset` will give a number of bytes to ignore; on
            // subsequent sectors `offset` is zero.

            self.flash.read_sector(
                self.current,
                sector as u32,
                &mut b.b0,
            )?;

            // Copy out the sector (less our offset) but allow for shorter
            // copies on the final sector.
            let n = usize::min(size_of::<F::Sector>() - offset, out.len());
            out[..n].copy_from_slice(&b.b0.borrow()[offset..offset + n]);

            offset = 0;
            sector += 1;
            out = &mut out[n..];
        }

        Ok(read_size)
    }

    pub fn read_kv(
        &self,
        key: &[u8],
        value_out: &mut [u8],
    ) -> Result<Option<usize>, low_level::ReadError<F::Error>> {
        let loc = {
            // limit scope of borrow so that read_contents below can work
            let mut b = self.buffers.borrow_mut();
            low_level::seek_kv_backwards(
                &self.flash,
                &mut b.b0,
                self.current,
                self.next_free,
                key,
            )?
        };

        if let Some(head_sector) = loc {
            Ok(Some(self.read_contents(head_sector, key.len() as u32, value_out)?))
        } else {
            Ok(None)
        }
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
    let c0 = low_level::check(flash, &mut buffers.b0, &mut buffers.b1, Space::Zero)?;
    let c1 = low_level::check(flash, &mut buffers.b0, &mut buffers.b1, Space::One)?;

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
