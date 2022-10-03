// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use core::mem::size_of;
use core::marker::PhantomData;
use core::borrow::{Borrow, BorrowMut};
use zerocopy::{AsBytes, FromBytes, Unaligned};
use num_traits::FromPrimitive;

//////////////////////////////////////////////////////////////////////////////
// Convenience wrappers for zerocopy.

pub fn cast_prefix<T>(bytes: &[u8]) -> (&T, &[u8])
    where T: FromBytes + Unaligned,
{
    let (lv, rest) = zerocopy::LayoutVerified::<_, T>::new_unaligned_from_prefix(bytes)
        .expect("type does not fit in sector");
    (lv.into_ref(), rest)
}

fn cast_prefix_mut<T>(bytes: &mut [u8]) -> (&mut T, &mut [u8])
    where T: AsBytes + FromBytes + Unaligned,
{
    let (lv, rest) = zerocopy::LayoutVerified::<_, T>::new_unaligned_from_prefix(bytes)
        .expect("type does not fit in sector");
    (lv.into_mut(), rest)
}

fn cast_suffix<T>(bytes: &[u8]) -> (&[u8], &T)
    where T: FromBytes + Unaligned,
{
    let (rest, lv) = zerocopy::LayoutVerified::<_, T>::new_unaligned_from_suffix(bytes)
        .expect("type does not fit in sector");
    (rest, lv.into_ref())
}

pub(crate) fn cast_suffix_mut<T>(bytes: &mut [u8]) -> (&mut [u8], &mut T)
    where T: AsBytes + FromBytes + Unaligned,
{
    let (rest, lv) = zerocopy::LayoutVerified::<_, T>::new_unaligned_from_suffix(bytes)
        .expect("type does not fit in sector");
    (rest, lv.into_mut())
}

//////////////////////////////////////////////////////////////////////////////
// At-rest layout.

/// Shorthand for a `u16` in little-endian representation.
type U16LE = zerocopy::U16<byteorder::LittleEndian>;
/// Shorthand for a `u32` in little-endian representation.
type U32LE = zerocopy::U32<byteorder::LittleEndian>;

/// Header written to the start of a formatted space, to mark it as such.
///
/// This will appear at the start of the first sector in a space. We take care
/// to write this sector _last_ so that this header only appears on fully
/// initialized spaces.
#[derive(Copy, Clone, Debug, FromBytes, AsBytes, Unaligned)]
#[repr(C)]
pub struct SpaceHeader {
    /// Magic number (`EXPECTED_MAGIC`) distinguishing this from arbitrary data.
    pub magic: U32LE,
    /// Generation number (sequence number) of this space. This is used to
    /// tie-break if we find both spaces initialized.
    pub generation: U32LE,
    /// Base 2 logarithm of the sector size. This is used as a check to ensure
    /// that the implementation has been correctly configured for the space.
    pub l2_sector_size: u8,
    /// Reserved padding bytes, must be zero.
    pub pad: [u8; 3],
    /// CRC32 of the above data, in order. Used to be doubly sure this isn't
    /// arbitrary data.
    pub crc: U32LE,
}

impl SpaceHeader {
    /// Bits we expect to find in the `magic` field. (This is a random number.)
    pub const EXPECTED_MAGIC: u32 = 0x53_be_88_9f;
    /// Smallest practical sector size (16 bytes).
    pub const MIN_L2_SECTOR_SIZE: u8 = 4;
    /// Largest halfway-reasonable sector size (1 GiB).
    pub const MAX_L2_SECTOR_SIZE: u8 = 30;

    /// Basic internal integrity check of the header. Checks fields against
    /// static ranges and verifies the magic and CRC. This doesn't know the
    /// sector size you're expecting, so you'll need to check that separately.
    pub fn check(&self) -> bool {
        self.magic.get() == Self::EXPECTED_MAGIC
            && self.l2_sector_size >= Self::MIN_L2_SECTOR_SIZE
            && self.l2_sector_size <= Self::MAX_L2_SECTOR_SIZE
            && self.pad == [0; 3]
            && self.crc_valid()
    }

    /// Compute the _expected_ CRC given all the other contents of `self`.
    pub fn expected_crc(&self) -> u32 {
        let algo = crc::Crc::<u32>::new(&crc::CRC_32_ISO_HDLC);
        let mut digest = algo.digest();
        digest.update(self.magic.as_bytes());
        digest.update(self.generation.as_bytes());
        digest.update(self.l2_sector_size.as_bytes());
        digest.update(&self.pad);

        digest.finalize()
    }

    /// Checks if the CRC field correctly describes the other fields in `self`.
    pub fn crc_valid(&self) -> bool {
        self.crc.get() == self.expected_crc()
    }
}

/// Metadata record used to identify entries.
///
/// The metadata is written at both offset 0 in the first sector of the entry,
/// and in the final bytes of the last sector, to ensure that the entry can be
/// processed from either direction.
#[derive(Copy, Clone, Debug, FromBytes, AsBytes, Unaligned, Eq, PartialEq)]
#[repr(C)]
pub struct EntryMeta {
    /// Marker to designate an entry and help distinguish it from
    /// unprogrammed or random data.
    pub magic: U16LE,
    /// Type of entry, and thus sub-meta. See the `KnownSubtypes` enum for
    /// defined values.
    pub subtype: u8,
    /// Length of submeta in bytes. The length must be correct for the `subtype`
    /// -- this is _not_ intended to support variable-length submeta.
    pub sub_bytes: u8,
    /// Length of contents separating the header from the trailer. This length
    /// is in bytes; the actual contents will be followed by enough padding to
    /// justify the subtrailer/trailer to the end of the sector.
    pub contents_length: U32LE,
}

impl EntryMeta {
    /// Bits we expect to find in the `magic` field.
    pub const EXPECTED_MAGIC: u16 = 0xCB_F5;

    /// Number of bytes in the full header/trailer including both this and the
    /// associated sub-meta.
    pub fn meta_length(&self) -> usize {
        size_of::<Self>() + usize::from(self.sub_bytes)
    }

    /// Number of bytes in the entire entry this record describes, without
    /// padding for sector alignment.
    pub fn unpadded_entry_length(&self) -> usize {
        2 * self.meta_length() + self.contents_length.get() as usize
    }

    /// Number of sectors in the entire entry this record describes.
    pub fn entry_sectors<F: Flash>(&self) -> u32 {
        bytes_to_sectors::<F>(self.unpadded_entry_length() as u32)
    }
}

/// Defined values for the `EntryMeta::subtype` field.
#[derive(Copy, Clone, Debug, Eq, PartialEq, num_derive::FromPrimitive)]
pub enum KnownSubtypes {
    // Note: 0 is reserved.

    /// An entry containing stored data. The sub-header/trailer will use the
    /// `DataSubMeta` format.
    Data = 0x01,
    /// An entry that serves to delete a previous entry. Uses the
    /// `DeleteSubMeta` format which happens to match data.
    Delete = 0x02,

    /// An entry that was incompletely written before a restart (and then
    /// repaired). Has no sub-header/trailer (length 0).
    ///
    /// `Aborted` is the only subtype that can appear in the trailer of an entry
    /// whose header reports a different type.
    Aborted = 0xFE,

    // Note: 0xFF is reserved.
}

/// Sub-metadata used for Data entries, representing key-value pairs.
#[derive(Copy, Clone, Debug, FromBytes, AsBytes, Unaligned)]
#[repr(C)]
pub struct DataSubMeta {
    /// Number of bytes in the key.
    pub key_length: U32LE,
    /// Hash of the key bytes using FNV-1, to assist in key lookup.
    pub key_hash: U32LE,
}

impl DataSubMeta {
    /// Size of the sub-metadata, in bytes.
    pub const SIZE: usize = size_of::<Self>();
    /// Same, but converted to `u8` for convenient use in the `sub_bytes` field.
    pub const SUB_BYTES: u8 = Self::SIZE as u8;
}

/// Computes the hash value corresponding to a particular key.
pub fn hash_key(key: &[u8]) -> u32 {
    const KEY_HASH_KEY: u64 = 0;

    use core::hash::{Hash, Hasher};

    let mut hasher = fnv::FnvHasher::with_key(KEY_HASH_KEY);
    key.hash(&mut hasher);
    let h = hasher.finish();
    h as u32 ^ (h >> 32) as u32
}

/// Sub-metadata used for Delete entries.
pub type DeleteSubMeta = DataSubMeta;


//////////////////////////////////////////////////////////////////////////////
// Flash device interface.

/// Designates one of the two spaces in a flash device. This is like a ranged
/// integer, or a bool with application-specific names.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Space {
    Zero = 0,
    One = 1,
}

impl Space {
    /// Convenient array of all spaces.
    pub const ALL: [Self; 2] = [Self::Zero, Self::One];

    /// Given a space, get the _other_ one.
    pub fn other(self) -> Self {
        match self {
            Self::Zero => Self::One,
            Self::One => Self::Zero,
        }
    }
}

impl From<Space> for usize {
    fn from(s: Space) -> Self {
        match s {
            Space::Zero => 0,
            Space::One => 1,
        }
    }
}

/// Trait describing flash memory for the purposes of our datastore.
pub trait Flash {
    /// Type of sector, which is typically a `[u8; N]` for some sector size `N`;
    /// this is a type alias rather than a `const` because of restrictions on
    /// the use of associated constants on type parameters in array sizes; see
    /// `rust-lang/rust#43408`.
    type Sector: Sized + BorrowMut<[u8]> + Borrow<[u8]>;

    /// Error type that can be produced during flash accesses.
    type Error;

    /// Returns the number of sectors per space (of two spaces) in this device.
    /// Note that this operation cannot fail; it's expected to be relatively
    /// quick and always return the same result.
    fn sectors_per_space(&self) -> u32;

    /// Reads sector `index` from `space` in the flash device, writing the data
    /// into `dest`.
    ///
    /// On success, `dest` should be fully overwritten with data from the
    /// device. On failure, you can leave `dest` partially or completely
    /// untouched.
    ///
    /// Depending on the underlying device, `read_sector` may return an error
    /// when applied to an erased sector, or may produce data.
    fn read_sector(
        &self,
        space: Space,
        index: u32,
        dest: &mut Self::Sector,
    ) -> Result<(), Self::Error>;

    /// Checks whether sector `index` in `space` in the flash device can be
    /// read. This is intended to signal that the sector has been programmed
    /// since last being erased. However, some devices will freely read
    /// erased-but-not-programmed sectors; for such devices, this function can
    /// simply return `true`.
    fn can_read_sector(&self, space: Space, index: u32) -> Result<bool, Self::Error>;

    /// Checks whether sector `index` in `space` in the flash device can be
    /// programmed (using `program_sector`). This is intended to check whether
    /// the sector has been erased.
    ///
    /// For devices that support multiple overlapping writes per sector, this
    /// should indicate whether the sector can be programmed _and then
    /// faithfully read back the intended data,_ rather than (say)
    /// bitwise-NANDing it with previous contents.
    fn can_program_sector(
        &self,
        space: Space,
        index: u32,
    ) -> Result<bool, Self::Error>;

    /// Writes `data` into sector `index` in `space` in the flash device.
    ///
    /// On success, the flash device _should_ report the same data from
    /// `read_sector` until the sector is erased -- modulo hardware failures,
    /// etc. On error, the sector _may or may not_ contain the data. It's
    /// probably best to assume that it needs to be erased.
    fn program_sector(
        &mut self,
        space: Space,
        index: u32,
        data: &Self::Sector,
    ) -> Result<(), Self::Error>;

    /// Erases the contents of `space`.
    fn erase_space(
        &mut self,
        space: Space,
    ) -> Result<(), Self::Error>;

    /// Compares `data.len()` bytes starting at `offset` from the start of
    /// sector `index` for equality. `offset` may be larger than a sector, for
    /// convenience.
    ///
    /// This is a "pushed compare" operation to take advantage of situations
    /// where we can do the compare without reading out every sector into RAM,
    /// such as directly-addressable flash.
    ///
    /// The `buffer` argument is loaned to the driver, which may arbitrarily
    /// scribble over its contents while doing the compare.
    ///
    /// The default implementation uses `read_sector`.
    fn compare_contents(
        &self,
        space: Space,
        buffer: &mut Self::Sector,
        mut index: u32,
        offset: u32,
        mut data: &[u8],
    ) -> Result<bool, Self::Error> {
        let sector_size = size_of::<Self::Sector>();

        let mut offset = offset as usize;

        while offset >= sector_size {
            index += 1;
            offset -= sector_size;
        }

        while !data.is_empty() {
            self.read_sector(space, index, buffer)?;
            let n = sector_size - offset;
            let n = usize::min(n, data.len());
            let (this_data, next_data) = data.split_at(n);
            if (*buffer).borrow()[offset..offset+n] != *this_data {
                return Ok(false);
            }

            index += 1;
            offset = 0;
            data = next_data;
        }

        Ok(true)
    }

    /// Compares two sequences of bytes in the flash device for equality. The
    /// sequences start at a sector boundary but can be any byte length, and do
    /// not need to be in the same space.
    ///
    /// This is a "pushed compare" operation for cases where the flash device
    /// can compare sequences of bytes with fewer copies.
    ///
    /// The default implementation uses `read_sector`.
    fn compare_internal(
        &self,
        space0: Space,
        mut sector0: u32,
        space1: Space,
        mut sector1: u32,
        mut length: u32,
        buffer0: &mut Self::Sector,
        buffer1: &mut Self::Sector,
    ) -> Result<bool, Self::Error> {
        while length > 0 {
            self.read_sector(space0, sector0, buffer0)?;
            self.read_sector(space1, sector1, buffer1)?;
            let n = length.min(size_of::<Self::Sector>() as u32);
            if (*buffer0).borrow()[..n as usize] != (*buffer1).borrow()[..n as usize] {
                return Ok(false);
            }
            sector0 += 1;
            sector1 += 1;
            length -= n;
        }
        Ok(true)
    }

    /// Copies a sequence of sectors from `from_space`/`from_sector` to
    /// `to_sector` in the other space.
    ///
    /// Drivers that can do a flash-to-flash copy without needing to copy all
    /// the data through RAM can implement this to do so. The default
    /// implementation uses `read_sector`/`program_sector`.
    fn copy_across(
        &mut self,
        from_space: Space,
        from_sector: u32,
        to_sector: u32,
        count: u32,
        buffer: &mut Self::Sector,
    ) -> Result<(), Self::Error> {
        let to_space = from_space.other();
        let src = from_sector..from_sector + count;
        let dst = to_sector..to_sector + count;

        for (fs, ts) in src.zip(dst) {
            self.read_sector(from_space, fs, buffer)?;
            self.program_sector(to_space, ts, buffer)?;
        }
        Ok(())
    }
}

/// Handy routine for converting a byte length to a sector count (rounded up) on
/// a given flash device.
///
/// Note: this is not `const` merely because of limitations on the use of trait
/// bounds in `const fn` at the time of this writing.
pub fn bytes_to_sectors<F: Flash>(x: u32) -> u32 {
    let sector_size = size_of::<F::Sector>() as u32;
    (x + sector_size - 1) / sector_size
}

/// Provides a way to hang constants off an implementation of the Flash trait
/// without them being overrideable to incorrect values by an implementation.
pub struct Constants<F>(PhantomData<F>);

impl<F: Flash> Constants<F> {
    pub const HEADER_SECTORS: u32 = {
        let sector = size_of::<F::Sector>();
        ((size_of::<SpaceHeader>() + sector - 1) / sector) as u32
    };
}


//////////////////////////////////////////////////////////////////////////////
// Filesystem operations/algorithms: general entry structure, reading, and
// search.

/// Reads entry metadata given the address of its first (head) sector.
///
/// This will fail if the entry metadata is corrupt: bad magic number,
/// sub-header size out of range, or nonsensical content length.
///
/// On success, the first sector of the entry is loaded into `buffer`, and an
/// `EntryInfo` struct referencing `buffer` provides the parse results. The
/// `EntryInfo::next_sector` field points to the sector _just past_ the tail
/// sector of this entry -- which is where you'd need to apply
/// `read_entry_from_head` again to continue forward through the log.
pub fn read_entry_from_head<'b, F: Flash>(
    flash: &F,
    buffer: &'b mut F::Sector,
    current: Space,
    sector: u32,
) -> Result<EntryInfo<'b>, ReadError<F::Error>> {
    flash.read_sector(current, sector, buffer)?;
    let data = (*buffer).borrow();
    let (meta, data) = cast_prefix::<EntryMeta>(data);

    if meta.magic.get() != EntryMeta::EXPECTED_MAGIC {
        return Err(ReadError::BadMagic(sector));
    }
    let submeta = data.get(..usize::from(meta.sub_bytes))
        .ok_or(ReadError::BadSubBytes(sector))?;

    let next_sector = sector.checked_add(meta.entry_sectors::<F>())
        .ok_or(ReadError::BadLength(sector))?;

    Ok(EntryInfo {
        next_sector,
        meta,
        submeta,
    })
}

/// Parsed information about an entry.
#[derive(Copy, Clone, Debug)]
pub struct EntryInfo<'a> {
    /// Sector number of the other end of this entry. Which end depends on which
    /// direction you were reading in.
    pub next_sector: u32,
    /// Entry metadata in sector buffer.
    pub meta: &'a EntryMeta,
    /// Entry sub-metadata in sector buffer.
    pub submeta: &'a [u8],
}

/// Things that can go wrong while reading low-level entries.
#[derive(Copy, Clone, Debug)]
pub enum ReadError<E> {
    /// Magic number (given) was wrong.
    BadMagic(u32),
    /// Length of sub-metadata (given) was too large.
    BadSubBytes(u32),
    /// Content length of entry (given) was too large for device.
    BadLength(u32),

    // TODO: probably doesn't belong in low_level
    End(u32),

    Flash(E),
}

impl<E> From<E> for ReadError<E> {
    fn from(e: E) -> Self {
        Self::Flash(e)
    }
}

/// Reads entry metadata given the address of the first sector past its final
/// sector (tail).
///
/// This will fail if the entry metadata is corrupt: bad magic number,
/// sub-header size out of range, or nonsensical content length.
///
/// On success, the last sector of the entry is loaded into `buffer`, and an
/// `EntryInfo` struct referencing `buffer` provides the parse results. The
/// `EntryInfo::next_sector` field points to the entry's head (first) sector --
/// which is where you'd need to apply `read_entry_from_tail` again to continue
/// backward through the log.
pub(crate) fn read_entry_from_tail<'b, F: Flash>(
    flash: &F,
    buffer: &'b mut F::Sector,
    current: Space,
    sector: u32,
) -> Result<EntryInfo<'b>, ReadError<F::Error>> {
    // Adjust sector number to point at the tail instead of just past it.
    let sector = sector - 1;

    flash.read_sector(current, sector, buffer)?;
    let data = (*buffer).borrow();
    let (data, meta) = cast_suffix::<EntryMeta>(data);

    if meta.magic.get() != EntryMeta::EXPECTED_MAGIC {
        return Err(ReadError::BadMagic(sector));
    }
    let submeta_start = data.len().checked_sub(usize::from(meta.sub_bytes))
        .ok_or(ReadError::BadSubBytes(sector))?;
    let submeta = &data[submeta_start..];

    let next_trailer = sector
        .checked_sub(meta.entry_sectors::<F>())
        .ok_or(ReadError::BadLength(sector))?;
    let next_sector = next_trailer + 1;

    Ok(EntryInfo {
        next_sector,
        meta,
        submeta,
    })
}

/// Processes entries in the log in space `current` starting from the end
/// (`start_sector`) and working backwards. Each entry is presented to `filter`,
/// which controls the traversal.
///
/// If `filter` returns `Ignore` for an entry, traversal continues. If traversal
/// hits the start of the log, this function returns `Ok(None)`.
///
/// If it returns `Accept`, traversal stops, and the head sector number for the
/// entry is returned.
///
/// If it returns `Abort`, traversal stops, and the function returns `Ok(None)`
/// as if no entry were found. (This is helpful for implementing Delete
/// entries.)
pub(crate) fn seek_backwards<'b, F: Flash>(
    flash: &F,
    buffer: &'b mut F::Sector,
    current: Space,
    start_sector: u32,
    mut filter: impl FnMut(&F, &mut F::Sector, u32, KnownSubMetas) -> Result<EntryDecision, F::Error>,
) -> Result<Option<u32>, ReadError<F::Error>> {
    let header_sectors = Constants::<F>::HEADER_SECTORS;

    assert!(start_sector >= header_sectors);

    let mut sector = start_sector;

    while sector > header_sectors {
        let entry = read_entry_from_tail(flash, buffer, current, sector)?;

        let head_sector = entry.next_sector;
        let ksh = KnownSubMetas::new(entry.meta.subtype, entry.submeta);

        let r = filter(flash, buffer, head_sector, ksh)?;
        match r {
            EntryDecision::Ignore => (),
            EntryDecision::Accept => return Ok(Some(head_sector)),
            EntryDecision::Abort => return Ok(None),
        }

        sector = head_sector;
    }

    Ok(None)
}

/// Metadata passed by-value to an entry seek predicate.
#[derive(Copy, Clone, Debug)]
pub enum KnownSubMetas {
    Data(DataSubMeta),
    Delete(DeleteSubMeta),
    Aborted,
    Other(u8),
}

impl KnownSubMetas {
    pub fn new(subtype: u8, submeta: &[u8]) -> Self {
        match KnownSubtypes::from_u8(subtype) {
            Some(KnownSubtypes::Data) => Self::Data(*cast_prefix(submeta).0),
            Some(KnownSubtypes::Delete) => Self::Delete(*cast_prefix(submeta).0),
            Some(KnownSubtypes::Aborted) => Self::Aborted,
            None => Self::Other(subtype),
        }
    }
}

/// Decisions a predicate may make about an entry during log traversal.
#[derive(Copy, Clone, Debug)]
pub(crate) enum EntryDecision {
    /// Keep going.
    Ignore,
    /// This is the entry we were looking for.
    Accept,
    /// Something about this entry means that the log doesn't _contain_ the
    /// entry we were looking for.
    Abort,
}

/// Reads the contents bytes of an entry -- the stuff between the sub-header and
/// sub-trailer. This is a raw access function used in the implementation of
/// higher level access functions, and is exposed for use in tooling.
///
/// When applied to a Data entry, this will read out the raw data record. Other
/// entries do not currently have contents.
///
/// The entry will be found starting at `head_sector` in `flash` from the
/// `current` space. Data will be copied starting `offset` bytes into the
/// contents, into `out`. `buffer` will be scribbled upon in the process as
/// scratch.
///
/// This works somewhat like POSIX read:
///
/// - If there are `out.len()` bytes available to read starting at `offset`,
///   this returns `Ok(out.len())` to indicate that it has filled `out`.
///
/// - If it would overlap the end of the entry's contents, it only fills in as
///   much of `out` as there are bytes available, and returns
///   `Ok(bytes_filled)`.
///
/// - If `offset` is exactly the length of the contents, returns `Ok(0)`.
///
/// - If `offset` is _outside_ the length of the contents, returns `Err(End)`.
pub fn read_contents<F: Flash>(
    flash: &F,
    buffer: &mut F::Sector,
    current: Space,
    head_sector: u32,
    offset: u32,
    out: &mut [u8],
) -> Result<usize, ReadError<F::Error>> {
    let entry = read_entry_from_head(
        flash,
        buffer,
        current,
        head_sector,
    )?;
    // Reject offsets that are totally outside the contents.
    let len_after_offset = entry.meta.contents_length.get()
        .checked_sub(offset)
        .ok_or(ReadError::End(entry.meta.contents_length.get()))?;
    // Compute the sector-relative offset by adding the length of the headers.
    let abs_offset = size_of::<EntryMeta>()
        + usize::from(entry.meta.sub_bytes)
        + offset as usize;
    // Limit the length of the read to the size of the out buffer.
    let xfer_len = usize::min(len_after_offset as usize, out.len());

    // Set up our loop variables by computing our starting sector, offset within
    // the sector, and length of output buffer.
    let mut sector = head_sector as usize
        + abs_offset / size_of::<F::Sector>();
    let mut offset = abs_offset % size_of::<F::Sector>();
    let mut out = &mut out[..xfer_len];

    while !out.is_empty() {
        flash.read_sector(
            current,
            sector as u32,
            buffer,
        )?;

        let n = usize::min(size_of::<F::Sector>() - offset, xfer_len);
        out[..n].copy_from_slice(&(*buffer).borrow()[offset..offset + n]);

        // To prepare for the next iteration, we zero offset (so that it's
        // non-zero only on the first iteration) and advance the others.
        offset = 0;
        sector += 1;
        out = &mut out[n..];
    }

    Ok(xfer_len)
}

//////////////////////////////////////////////////////////////////////////////
// Filesystem operations/algorithms: general entry writing.

/// Writes a new entry.
///
/// The entry will be written starting at `start_sector` in `flash`, in the
/// space `current`.
///
/// The entry will be constructed from:
///
/// - the slices in `pieces`, in order without padding,
/// - enough padding to right-align the remainder,
/// - `subtrailer` and `trailer` in that order.
///
/// This means that, in practice, `pieces[0]` should be the header metadata and
/// `pieces[1]` the submetadata.
///
/// On success, this function returns the sector number of the next free sector
/// after the entry is written.
///
/// Sectors are written strictly in-order, so on a write failure, you'll be left
/// in one of three possible states:
///
/// 1. The failure prevented the header from being written. The contents of the
///    log are unchanged.
/// 2. The failure happened somewhere before the trailer was finished and the
///    log now ends in an incomplete entry.
/// 3. The failure happened during write of the final sector, but did not
///    prevent the sector from being written; the log now ends in your complete
///    entry as though no error occurred. (We allow for this case to make flash
///    device implementation slightly easier.)
///
/// You can use `check_entry` if you need to distinguish, or treat any error as
/// potentially incomplete and attempt to repair the log.
pub(crate) fn write_entry<F: Flash>(
    flash: &mut F,
    buffer: &mut F::Sector,
    current: Space,
    start_sector: u32,
    pieces: &[&[u8]],
    subtrailer: &[u8],
    trailer: &[u8],
) -> Result<u32, WriteError<F::Error>> {
    // Work out the length without padding.
    let total_length = pieces.iter().map(|p| p.len()).sum::<usize>()
        + subtrailer.len()
        + trailer.len();
    let total_length = u32::try_from(total_length).unwrap();
    // Convert to sectors, which has the effect of including the padding.
    let total_sectors = bytes_to_sectors::<F>(total_length);
    // Detect out-of-space.
    if flash.sectors_per_space() - start_sector < total_sectors {
        return Err(WriteError::NoSpace);
    }

    // Set up loop variables. Management of these variables is a little subtle
    // since we don't require `pieces` to be sector-aligned.
    let mut sector = start_sector;
    let mut data = buffer.borrow_mut();

    // Gather all the `pieces` and write them contiguously.
    for mut piece in pieces.iter().cloned() {
        // Loop because a piece may be larger than a sector.
        while !piece.is_empty() {
            // Break both the piece and the remaining sector buffer into the
            // largest common chunk (piece0/data0) and the remainder
            // (piece1/data1).
            let n = usize::min(piece.len(), data.len());
            let (piece0, piece1) = piece.split_at(n);
            let (data0, data1) = data.split_at_mut(n);

            data0.copy_from_slice(piece0);

            piece = piece1;
            data = data1;

            // Check for entirely filled sector buffer; write it out and reset
            // it.
            if data.is_empty() {
                drop(data);

                flash.program_sector(current, sector, buffer)?;
                sector += 1;

                data = buffer.borrow_mut();
            }
        }
    }

    // Check how much space we have left in the final sector.
    if data.len() < subtrailer.len() + trailer.len() {
        // We can't fit the full trailer into the final sector, so we have to
        // burn a sector on the trailer by flushing this sector and starting a
        // new one.
        //
        // Zero-fill the remainder of the flushed sector to avoid writing
        // arbitrary goo.
        data.fill(0);
        drop(data);

        flash.program_sector(current, sector, buffer)?;
        sector += 1;

        data = buffer.borrow_mut();
    }

    // Fill in the trailer.
    {
        // Split remaining sector tail into unused area, which will be filled
        // with padding, and the trailer part.
        let dl = data.len();
        let unused = dl - trailer.len() - subtrailer.len();
        let (pad, tail) = data.split_at_mut(unused);
        pad.fill(0);

        // Now split the tail into subtrailer and trailer regions and fill them
        // in.
        let (sub, tail) = tail.split_at_mut(subtrailer.len());
        sub.copy_from_slice(subtrailer);
        tail.copy_from_slice(trailer);
    }

    // Write the final sector.
    drop(data);
    flash.program_sector(current, sector, buffer)?;

    Ok(sector + 1)
}

/// Things that can go wrong while writing an entry.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum WriteError<E> {
    /// There is not enough space on the flash device to add this entry to the
    /// log.
    NoSpace,
    /// An underlying flash access error occurred.
    Flash(E),
}

impl<E> From<E> for WriteError<E> {
    fn from(e: E) -> Self {
        Self::Flash(e)
    }
}


//////////////////////////////////////////////////////////////////////////////
// Data entry implementation.

/// Searches backwards through the log for a data entry matching `key`.
///
/// `start_sector` is the index of the sector _just past_ the valid data you
/// wish to search. The search will start at `start_sector - 1` and run back to
/// the space header, stopping prematurely if a matching entry is found.
///
/// Return values:
///
/// - `Ok(Some(n))` indicates that a matching entry was found, and its header
///   sector is at index `n`.
/// - `Ok(None)` indicates that there is no matching entry in the log, either
///   because it does not exist, or because it has been deleted.
/// - `Err(e)` if the search could not be completed because a corrupt entry was
///   discovered, or if the underlying flash layer returns an error.
pub(crate) fn seek_kv_backwards<F: Flash>(
    flash: &F,
    buffer: &mut F::Sector,
    current: Space,
    start_sector: u32,
    key: &[u8],
) -> Result<Option<u32>, ReadError<F::Error>> {
    let key_len = u32::try_from(key.len())
        .expect("key too long");
    let key_hash = hash_key(key);

    seek_backwards(
        flash,
        buffer,
        current,
        start_sector,
        |flash, buffer, index, ksub| {
            match ksub {
                KnownSubMetas::Data(sub) | KnownSubMetas::Delete(sub) => {
                    // For these types, we want to check the key.
                    if sub.key_hash.get() == key_hash
                        && sub.key_length.get() == key_len
                    {
                        // A potential match! We need to compare the first
                        // `key.len()` bytes after the header.
                        let meta_bytes = size_of::<EntryMeta>()
                            + size_of::<DataSubMeta>();
                        let key_eq = flash.compare_contents(
                            current,
                            buffer,
                            index,
                            meta_bytes as u32,
                            key,
                        )?;
                        if key_eq {
                            // Now, the difference between Data and Delete comes
                            // into play.
                            if matches!(ksub, KnownSubMetas::Data(_)) {
                                return Ok(EntryDecision::Accept)
                            } else {
                                // A delete entry causes us to early-abort with no
                                // match:
                                return Ok(EntryDecision::Abort)
                            };
                        }
                    }
                }
                _ => {
                    // Everything else, we'll just skip.
                }
            }
            Ok(EntryDecision::Ignore)
        },
    )
}

/// Writes a new Data entry for `key` storing `value` and superceding any
/// previous entry for `key`.
///
/// The entry will be written starting at `start_sector` (inclusive) into
/// `flash` in space `current`. `buffer` will be scribbled on by the
/// implementation.
pub(crate) fn write_kv<F: Flash>(
    flash: &mut F,
    buffer: &mut F::Sector,
    current: Space,
    start_sector: u32,
    key: &[u8],
    value: &[u8],
) -> Result<u32, WriteError<F::Error>> {
    // Paranoid parameter validation: are any of these too large for a u32? On a
    // 32-bit platform this can't happen.
    let key_len = u32::try_from(key.len())
        .expect("key too long");
    let value_len = u32::try_from(value.len())
        .expect("value too long");
    let contents_length = key_len.checked_add(value_len)
        .expect("key+value too long");

    // Construct header/trailer. The same bits do double-duty in both places.
    let meta = EntryMeta {
        magic: EntryMeta::EXPECTED_MAGIC.into(),
        subtype: KnownSubtypes::Data as u8,
        sub_bytes: DataSubMeta::SUB_BYTES,
        contents_length: contents_length.into(),
    };
    let submeta = DataSubMeta {
        key_length: key_len.into(),
        key_hash: hash_key(key).into(),
    };

    // Go!
    write_entry(
        flash,
        buffer,
        current,
        start_sector,
        &[
            meta.as_bytes(),
            submeta.as_bytes(),
            key,
            value,
        ],
        // Note that order is reversed here:
        submeta.as_bytes(),
        meta.as_bytes(),
    )
}

//////////////////////////////////////////////////////////////////////////////
// Space formatting and checking.

/// Creates a new empty log in device `flash` and space `current`.
///
/// This requires that the space has been erased. If it has not been erased,
/// this will fail with `FormatError::NeedsErase`.
///
/// Given an erased space, formatting is simply a matter of writing a valid
/// space header, so this only winds up needing one sector write.
pub fn format<F: Flash>(
    flash: &mut F,
    buffer: &mut F::Sector,
    current: Space,
    initial_generation: u32,
) -> Result<(), FormatError<F::Error>> {
    // Check that the target space is entirely erased.
    for s in 0..flash.sectors_per_space() {
        if !flash.can_program_sector(current, s)? {
            return Err(FormatError::NeedsErase);
        }
    }

    // Write the header.
    let (header, tail) = cast_prefix_mut::<SpaceHeader>((*buffer).borrow_mut());
    *header = SpaceHeader {
        magic: SpaceHeader::EXPECTED_MAGIC.into(),
        generation: initial_generation.into(),
        l2_sector_size: size_of::<F::Sector>().trailing_zeros() as u8,
        pad: [0; 3],
        crc: 0.into(),
    };
    header.crc = header.expected_crc().into();
    tail.fill(0);

    flash.program_sector(current, 0, buffer)?;

    Ok(())
}

/// Things that can go wrong with `format`.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum FormatError<E> {
    NeedsErase,
    Flash(E),
}

impl<E> From<E> for FormatError<E> {
    fn from(e: E) -> Self {
        Self::Flash(e)
    }
}

/// Scans the contents of `flash` to determine the state of space `current`.
///
/// `buffer` will be scribbled on by the implementation as scratch space.
///
/// This function will fail (return `Err`) only if it is unable to scan the
/// space. An `Ok` return means that the check finished, _not_ that the space
/// contains a valid log. See the `CheckResult` type for more details.
pub fn check<F: Flash>(
    flash: &mut F,
    buffer: &mut F::Sector,
    current: Space,
) -> Result<CheckResult, F::Error> {
    let sector_count = flash.sectors_per_space();

    // Check if the first sector is programmable and use that as a proxy for
    // "erased." Our space header contents are (deliberately) distinguishable
    // from typical erased flash, even for devices that allow us to read back
    // erased sectors, so this should be universal.
    if flash.can_program_sector(current, 0)? {
        // Well, there's no valid store here... let's distinguish between full
        // and partial erase to help our caller out.
        for sector in 1..sector_count {
            if !flash.can_program_sector(current, sector)? {
                // There's at least one un-erased sector; we're going to have to
                // erase this space before we can use it.
                return Ok(CheckResult::Bad(CheckError::PartiallyErased));
            }
        }

        // This space is empty and could be used as an idle space.
        return Ok(CheckResult::Bad(CheckError::Erased));
    }

    // Check the space header!
    flash.read_sector(current, 0, buffer)?;
    let (space_header, _) = cast_prefix::<SpaceHeader>((*buffer).borrow());

    if !space_header.check() {
        return Ok(CheckResult::Bad(CheckError::BadSpaceHeader));
    }
    if 1 << space_header.l2_sector_size != size_of::<F::Sector>() {
        return Ok(CheckResult::Bad(CheckError::WrongSectorSize));
    }

    // Copy the generation out so that we can let go of buffer0.
    let generation = space_header.generation.get();
    drop(space_header);

    let mut sector = Constants::<F>::HEADER_SECTORS;
    let mut incomplete_write_end = None;

    // Work over the remaining sectors in the (alleged) log, treating them as
    // log entries.
    while sector < sector_count {
        let r = check_entry(flash, buffer, current, sector)?;

        match r {
            CheckEntryResult::ChecksPassed(next) => {
                // Advance the sector pointer and _keep going!_
                sector = next;
            }

            CheckEntryResult::HeadErased => {
                // This is probably the end of the log in this space. Don't
                // advance the sector pointer.
                break;
            }
            CheckEntryResult::IncompleteWrite(next) => {
                // Note that we need repair. DO NOT advance the sector pointer!
                // We will leave the incomplete write just past the end of the
                // valid log so that reads don't have to think about it.
                incomplete_write_end = Some(next);
                break;
            }

            CheckEntryResult::HeadCorrupt
                | CheckEntryResult::HeadTailMismatch
                | CheckEntryResult::TailCorrupt => {
                return Ok(CheckResult::Bad(CheckError::Corrupt(sector)));
            }

            CheckEntryResult::PartiallyErased => {
                return Ok(CheckResult::Bad(CheckError::UnprogrammedData(sector)));
            }
        }
    }
    
    // Now, scan the rest of the space to see if we've really found the end of
    // the log, i.e. whether the rest of the space is erased.
    //
    // We want to start the scan at the end of the log, _or_ just past an
    // incomplete write, if one was found.
    let mut tail_erased = true;
    let scan_start = incomplete_write_end.unwrap_or(sector);
    for s in scan_start..sector_count {
        if !flash.can_program_sector(current, s)? {
            // Welp, there's at least one unerased turd in the space past the
            // end of the log, which means we can't mount this writable yet.
            tail_erased = false;
            break;
        }
    }

    // Return our findings.
    Ok(CheckResult::ValidLog {
        generation,
        end: sector,
        incomplete_write: incomplete_write_end.is_some(),
        tail_erased,
    })
}

/// Result of completing the `check` process.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum CheckResult {
    /// There is no valid log in this space, for the reason documented in
    /// `CheckError`.
    Bad(CheckError),

    /// A valid data store was found, with a valid header and some number of
    /// valid entries running up to sector index `end` (which is one past the
    /// end of valid data). This indicates that the store can be mounted at
    /// least read-only.
    ///
    /// To mount read-write, `tail_erased` must be true, and `incomplete_write`
    /// must be false. Or, repair action must be taken.
    ValidLog {
        /// Generation number found in the space header.
        generation: u32,
        /// Index of first non-log sector in the space.
        end: u32,
        /// Whether all sectors in the space starting with `end` have been
        /// erased (`true`) or if some haven't (`false`).
        tail_erased: bool,
        incomplete_write: bool,
    }
}

/// Errors reported by the `check` process.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum CheckError {
    /// While there is no log in the space, the space is entirely erased. It can
    /// be used as the idle space with no further work.
    Erased,
    /// Not all of the sectors in the space have been erased, but enough have
    /// been erased to render the data structure useless. It must be fully
    /// erased before it can be used.
    PartiallyErased,
    /// The initial sector of the space is programmed, but can't be validated as
    /// a space header.
    BadSpaceHeader,
    /// The space begins with what appears to be a valid space header, but the
    /// recorded sector size is wrong for this flash device.
    WrongSectorSize,

    /// The store is corrupt starting with the "entry" at the given sector
    /// index. You may be able to successfully read a prior version of the store
    /// by treating this as the end of the log.
    Corrupt(u32),

    /// An entry at the given sector number contains at least one unprogrammed
    /// sector, making it unsafe to read. You may be able to successfully read a
    /// prior version of the store by treating this as the end of the log.
    UnprogrammedData(u32),
}

/// Reads the entry starting at `head_sector` and checks that it's valid.
///
/// Note that success (`Ok`) of this function just means that the check
/// completed without flash access errors. It does _not_ mean the entry is fine.
///
/// There are three classes of `Ok` results from this operation.
///
/// Results that indicate everything is fine:
/// - `ChecksPassed(next)` indicates a valid and matching head/tail, that all
///   data sectors between are programmed with _something,_ and the next entry
///   should begin at `next`.
///
/// Results that indicate a problem with _this_ entry but don't necessarily
/// imply corruption of the overall structure:
///
/// - `HeadErased` probably just indicates the end of the log.
/// - `IncompleteWrite` probably means the end of the log plus a power loss.
///
/// Finally, the rest of the `CheckEntryResult` variants indicate entry
/// corruption.
pub(crate) fn check_entry<F: Flash>(
    flash: &mut F,
    buffer: &mut F::Sector,
    current: Space,
    head_sector: u32,
) -> Result<CheckEntryResult, F::Error> {
    // Check if the first sector is written.
    if flash.can_program_sector(current, head_sector)? {
        return Ok(CheckEntryResult::HeadErased);
    }
    // Attempt to read and parse the header.
    let head_info = match read_entry_from_head(flash, buffer, current, head_sector) {
        Err(ReadError::Flash(e)) => return Err(e),
        Err(_) => return Ok(CheckEntryResult::HeadCorrupt),
        Ok(entry) => entry,
    };

    // Using the length information from the header, locate the tail sector and
    // see if it's programmed. Unprogrammed tail sector indicates an incomplete
    // write.
    if flash.can_program_sector(current, head_info.next_sector - 1)? {
        return Ok(CheckEntryResult::IncompleteWrite(head_info.next_sector));
    }

    // Get ready to read the tail sector by freeing the buffer.
    let next_entry = head_info.next_sector;
    let head_meta = *head_info.meta;
    drop(head_info);

    // Attempt to read and parse the trailer.
    let tail_info = match read_entry_from_tail(flash, buffer, current, next_entry) {
        Err(ReadError::Flash(e)) => return Err(e),
        Err(_) => return Ok(CheckEntryResult::TailCorrupt),
        Ok(entry) => entry,
    };
   
    // Make sure the basics match the header.
    if tail_info.next_sector != head_sector
        || tail_info.meta.contents_length != head_meta.contents_length
    {
        return Ok(CheckEntryResult::HeadTailMismatch);
    }

    // We permit only one kind of mismatch between header and trailer: the
    // trailer may be marked Aborted, regardless of how the header is marked. In
    // this case we still require the other fields to match.
    if tail_info.meta.subtype != KnownSubtypes::Aborted as u8
        && (tail_info.meta.subtype != head_meta.subtype 
            || tail_info.meta.sub_bytes != head_meta.sub_bytes)
    {
        return Ok(CheckEntryResult::HeadTailMismatch);
    }

    if tail_info.meta.subtype != KnownSubtypes::Aborted as u8 {
        // For any other subtype we require the entry's sectors to be totally
        // readable, which typically means programmed -- though some flashes
        // will expose unprogrammed sectors as all FF, which we tolerate because
        // it won't introduce flash read errors into log operations.
        for s in head_sector + 1..next_entry {
            if !flash.can_read_sector(current, s)? {
                return Ok(CheckEntryResult::PartiallyErased);
            }
        }
    }

    Ok(CheckEntryResult::ChecksPassed(next_entry))
}

/// Possible successful results of `check_entry`.
#[derive(Copy, Clone, Debug)]
pub enum CheckEntryResult {
    /// The head sector for this entry is blank. This probably means you've
    /// reached the end of the log.
    HeadErased,

    /// The head sector for this entry is programmed and looks reasonable, but
    /// the tail sector is blank. This suggests an entry that was being written
    /// when we lost power or crashed. This condition is recoverable with
    /// effort.
    ///
    /// The given sector index is the first sector _after_ the unprogrammed tial
    /// section.
    IncompleteWrite(u32),

    /// The head sector for this entry is corrupt or contains arbitrary data.
    HeadCorrupt,
    /// The head and tail sectors for this entry are both programmed, but do not
    /// match one another, suggesting data corruption.
    HeadTailMismatch,
    /// The tail sector is programmed with invalid data, suggesting data
    /// corruption.
    TailCorrupt,
    /// The head and tail sectors of this entry are valid, but at least one of
    /// the data sectors between is blank.
    PartiallyErased,

    /// The entry appears valid, and the next entry will be at the given sector
    /// index.
    ChecksPassed(u32),
}

//////////////////////////////////////////////////////////////////////////////
// Space evacuation / garbage collection.

/// Processes the log in `from_space`, identifies the subset of its entries that
/// have not been superceded by later entries, and copies them into the opposite
/// space on the same flash device.
///
/// The relative order of the entries is _not_ preserved, to simplify the
/// implementation.
///
/// `watermark` is the number of sectors in `from_space` that have been written,
/// i.e. the length of the log including the space header.
///
/// `buffer0` and `buffer1` are temporary storage space used by the
/// implementation, and their contents will be scribbled upon.
///
/// On success, the space opposite to `from_space` contains a log that is
/// semantically equivalent to the one in `from_space`, but with all redundant
/// entries removed, and the generation counter advanced by 1. This function
/// returns the watermark for that new log.
///
/// Failures may be `ReadError` indicating a problem processing the `from_space`
/// log, or errors from the underlying flash device indicating problems writing
/// the new log.
///
/// This function takes care to write the new space header _last,_ ensuring that
/// the destination space will only `check` as a valid log if all writes have
/// completed. Any failure should leave the destination space in the `Erased` or
/// `PartiallyErased` state where it can be erased and attempted again if
/// necessary.
///
/// TODO: this implementation is currently O(n^2) as it doesn't use any kind of
/// cache to track which entries have been evacuated, requiring a scan of
/// to-space any time a data entry is copied. This could be fixed by giving it
/// more RAM to track copies.
pub fn evacuate<F: Flash>(
    flash: &mut F,
    buffer0: &mut F::Sector,
    buffer1: &mut F::Sector,
    from_space: Space,
    watermark: u32,
) -> Result<u32, ReadError<F::Error>> {
    flash.read_sector(from_space, 0, buffer0)?;
    let (space_header, _) = cast_prefix::<SpaceHeader>((*buffer0).borrow());
    let from_generation = space_header.generation.get();
    let to_space = from_space.other();

    // Read every entry in from-space starting at the most recent and working
    // back, to ensure that we see any superceding entries before their
    // predecessors.
    let header_sectors = Constants::<F>::HEADER_SECTORS;
    let mut from_sector = watermark;
    let mut to_sector = header_sectors;
    while from_sector > header_sectors {
        let entry = read_entry_from_tail(flash, buffer0, from_space, from_sector)?;

        let meta_bytes = entry.meta.meta_length();
        let entry_sectors = entry.meta.entry_sectors::<F>();

        let head_sector = entry.next_sector;

        let subtype = KnownSubtypes::from_u8(entry.meta.subtype);
        let should_copy = match subtype {
            Some(KnownSubtypes::Data) | Some(KnownSubtypes::Delete) => {
                // These get copied over but only if they're not superceded.
                // This means we need to determine whether an entry for this key
                // has already been evacuated into to-space.
                //
                // We will probably want to add some sort of cache for this in
                // the future, but since RAM usage will be limited, we will
                // probably always need to be able to fall back to scanning the
                // in-progress to-space data structure for a match.

                let (_, subtrailer) = cast_suffix::<DataSubMeta>(entry.submeta);
                let key_hash = subtrailer.key_hash.get();
                let key_len = subtrailer.key_length.get();

                let pred = seek_backwards(
                    flash,
                    buffer0,
                    to_space,
                    to_sector,
                    |flash, buf, s, sub| {
                        match sub {
                            KnownSubMetas::Data(sub) | KnownSubMetas::Delete(sub) => {
                                if sub.key_hash.get() == key_hash
                                    && sub.key_length.get() == key_len
                                {
                                    let key_eq = flash.compare_internal(
                                        from_space,
                                        head_sector,
                                        to_space,
                                        s,
                                        key_len + meta_bytes as u32,
                                        buf,
                                        buffer1,
                                    )?;

                                    if key_eq {
                                        // Match!
                                        return Ok(EntryDecision::Accept);
                                    }
                                }
                            }
                            _ => (),
                        }
                        Ok(EntryDecision::Ignore)
                    }
                )?;

                if pred.is_some() {
                    false
                } else {
                    true
                }
            }
            Some(KnownSubtypes::Aborted) => {
                // Not copied, no effect on store.
                false
            }
            None => {
                // We can't safely copy entries we don't understand, because
                // they may contain internal references to other parts of the
                // data structure.
                panic!()
            }
        };

        if should_copy {
            flash.copy_across(
                from_space,
                head_sector,
                to_sector,
                entry_sectors,
                buffer0,
            )?;
            to_sector += entry_sectors;
        }

        from_sector = head_sector;
    }

    let b0 = buffer0.borrow_mut();
    let (to_header, btail) = cast_prefix_mut(b0);
    *to_header = SpaceHeader {
        magic: SpaceHeader::EXPECTED_MAGIC.into(),
        generation: from_generation.wrapping_add(1).into(),
        l2_sector_size: size_of::<F::Sector>().trailing_zeros() as u8,
        pad: [0; 3],
        crc: 0.into(),
    };
    to_header.crc = to_header.expected_crc().into();
    btail.fill(0);

    flash.program_sector(to_space, 0, buffer0)?;

    Ok(to_sector)
}

#[cfg(test)]
mod tests {
    use super::*;

    pub struct FakeFlash<const N: usize> {
        sectors: [Vec<Option<[u8; N]>>; 2],
    }

    impl<const N: usize> FakeFlash<N> {
        pub fn new(sector_count: usize) -> Self {
            Self {
                sectors: [
                    vec![None; sector_count],
                    vec![None; sector_count],
                ],
            }
        }

        pub fn sectors(&self, space: Space) -> &[Option<[u8; N]>] {
            match space {
                Space::Zero => &self.sectors[0],
                Space::One => &self.sectors[1],
            }
        }

        pub fn sectors_mut(&mut self, space: Space) -> &mut [Option<[u8; N]>] {
            match space {
                Space::Zero => &mut self.sectors[0],
                Space::One => &mut self.sectors[1],
            }
        }

        pub fn erase_sector(&mut self, space: Space, index: u32) {
            self.sectors_mut(space)[index as usize] = None;
        }
    }

    #[derive(Copy, Clone, Debug, Eq, PartialEq)]
    pub enum FakeFlashError {}

    impl<const N: usize> Flash for FakeFlash<N> {
        type Sector = [u8; N];
        type Error = FakeFlashError;

        fn sectors_per_space(&self) -> u32 {
            u32::try_from(self.sectors[0].len()).unwrap()
        }

        fn read_sector(&self, space: Space, index: u32, dest: &mut Self::Sector) -> Result<(), Self::Error> {
            let sectors = self.sectors(space);
            *dest = sectors[index as usize]
                .expect("read of unprogrammed sector");
            Ok(())
        }

        fn can_program_sector(&self, space: Space, index: u32) -> Result<bool, Self::Error> {
            Ok(self.sectors(space)[index as usize].is_none())
        }

        fn can_read_sector(&self, space: Space, index: u32) -> Result<bool, Self::Error> {
            Ok(self.sectors(space)[index as usize].is_some())
        }

        fn program_sector(&mut self, space: Space, index: u32, data: &Self::Sector) -> Result<(), Self::Error> {
            let s = &mut self.sectors_mut(space)[index as usize];
            if s.is_some() {
                panic!("attempt to double-program sector {index}");
            }
            *s = Some(*data);
            Ok(())
        }

        fn erase_space(&mut self, space: Space) -> Result<(), Self::Error> {
            self.sectors_mut(space).fill(None);
            Ok(())
        }
    }

    #[test]
    fn test_write_entry_single_sector_no_pad() {
        let mut flash = FakeFlash::<16>::new(64);
        let mut buffer = [0u8; 16];

        let next_free = write_entry(
            &mut flash,
            &mut buffer,
            Space::Zero,
            1,
            &[
                &[0, 1],
                &[2, 3],
                &[4, 5, 6, 7],
            ],
            &[8, 9, 10, 11],
            &[12, 13, 14, 15],
        ).map_err(|_| ()).unwrap();

        assert_eq!(next_free, 2);

        let mut expected = [0; 16];
        for (i, byte) in expected.iter_mut().enumerate() {
            *byte = i as u8;
        }

        assert_eq!(&flash.sectors(Space::Zero)[1], &Some(expected));
    }

    #[test]
    fn test_write_entry_single_sector_pad() {
        let mut flash = FakeFlash::<16>::new(64);
        let mut buffer = [0u8; 16];

        let next_free = write_entry(
            &mut flash,
            &mut buffer,
            Space::Zero,
            1,
            &[
                &[0, 1],
                &[2, 3],
                &[4, 5],
            ],
            &[8, 9, 10, 11],
            &[12, 13, 14, 15],
        ).map_err(|_| ()).unwrap();

        assert_eq!(next_free, 2);

        let mut expected = [0; 16];
        for (i, byte) in expected.iter_mut().enumerate() {
            *byte = i as u8;
        }
        // fill in padding
        expected[6] = 0;
        expected[7] = 0;

        assert_eq!(&flash.sectors(Space::Zero)[1], &Some(expected));
    }

    #[test]
    fn test_empty_entry_zero_fill() {
        let mut flash = FakeFlash::<16>::new(64);
        let mut buffer = [0u8; 16];

        let next_free = write_entry(
            &mut flash,
            &mut buffer,
            Space::Zero,
            1,
            &[],
            &[],
            &[],
        ).map_err(|_| ()).unwrap();

        // Should still burn a sector.
        assert_eq!(next_free, 2);

        assert_eq!(&flash.sectors(Space::Zero)[1], &Some([0; 16]));
    }

    #[test]
    fn test_write_entry_two_sector_short_data() {
        let mut flash = FakeFlash::<16>::new(64);
        let mut buffer = [0u8; 16];

        let header = [1; 4];
        let subheader = [2; 4];
        // Just enough data to evict the subtrailer/trailer from the first
        // sector, but not enough to fill the sector. This tests trailing
        // padding of sectors at the end of data.
        let data = [0xAA; 4];
        let subtrailer = [3; 4];
        let trailer = [4; 4];

        let next_free = write_entry(
            &mut flash,
            &mut buffer,
            Space::Zero,
            1,
            &[
                &header,
                &subheader,
                &data,
            ],
            &subtrailer,
            &trailer,
        ).map_err(|_| ()).unwrap();

        // Should burn two sectors
        assert_eq!(next_free, 3);

        // Header plus data with trailing padding:
        assert_eq!(&flash.sectors(Space::Zero)[1], &Some([
            1, 1, 1, 1, 2, 2, 2, 2,
            0xAA, 0xAA, 0xAA, 0xAA, 0, 0, 0, 0,
        ]));
        // Second sector is all padding plus trailer:
        assert_eq!(&flash.sectors(Space::Zero)[2], &Some([
            0, 0, 0, 0, 0, 0, 0, 0,
            3, 3, 3, 3, 4, 4, 4, 4,
        ]));
    }

    #[test]
    fn test_write_entry_two_sector_longer_data() {
        let mut flash = FakeFlash::<16>::new(64);
        let mut buffer = [0u8; 16];

        let header = [1; 4];
        let subheader = [2; 4];
        let data = [0xAA; 14];
        let subtrailer = [3; 4];
        let trailer = [4; 4];

        let next_free = write_entry(
            &mut flash,
            &mut buffer,
            Space::Zero,
            1,
            &[
                &header,
                &subheader,
                &data,
            ],
            &subtrailer,
            &trailer,
        ).map_err(|_| ()).unwrap();

        // Should burn two sectors
        assert_eq!(next_free, 3);

        assert_eq!(&flash.sectors(Space::Zero)[1], &Some([
            1, 1, 1, 1, 2, 2, 2, 2,
            0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA,
        ]));
        // Test second sector with embedded padding:
        assert_eq!(&flash.sectors(Space::Zero)[2], &Some([
            0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA,
            0, 0,
            3, 3, 3, 3, 4, 4, 4, 4,
        ]));
    }

    #[test]
    fn test_read_entry_from_tail_single() {
        let mut flash = FakeFlash::<32>::new(64);
        // sector 0 is the space header.
        // sector 1 will be both header and trailer:
        //  00  magic, subtype, sub_bytes
        //  04  contents_length
        //  08  fake subheader
        //
        //  0c  data
        //  10  data
        //
        //  14  fake subheader
        //  18  magic, subtype, sub_bytes
        //  1C  contents_length
        let mut sec1 = [0; 32];
        sec1[0..=1].copy_from_slice(&EntryMeta::EXPECTED_MAGIC.to_le_bytes());
        sec1[2] = 0xAA;
        sec1[3] = 4;
        sec1[4..=7].copy_from_slice(&8_u32.to_le_bytes());

        sec1[8..=0xB].copy_from_slice(&0xDEAD_BEEF_u32.to_le_bytes());

        sec1[0x0C..=0x13].fill(0xFF);

        sec1[0x14..=0x17].copy_from_slice(&0xDEAD_BEEF_u32.to_le_bytes());

        sec1[0x18..=0x19].copy_from_slice(&EntryMeta::EXPECTED_MAGIC.to_le_bytes());
        sec1[0x1A] = 0xAA;
        sec1[0x1B] = 4;
        sec1[0x1C..=0x1F].copy_from_slice(&8_u32.to_le_bytes());
        flash.program_sector(Space::Zero, 1, &sec1).unwrap();

        let mut buffer = [0u8; 32];

        // We request to read backwards starting at sector 2, not sector 1,
        // because that's how the API crumbles
        let entry_info = read_entry_from_tail(
            &mut flash,
            &mut buffer,
            Space::Zero,
            2,
        ).expect("entry should pass read validation");

        assert_eq!(entry_info.meta, &EntryMeta {
            magic: EntryMeta::EXPECTED_MAGIC.into(),
            subtype: 0xAA,
            sub_bytes: 4,
            contents_length: 8.into(),
        });
        assert_eq!(entry_info.next_sector, 1);
        assert_eq!(entry_info.submeta, &0xDEAD_BEEF_u32.to_le_bytes());
    }

    #[test]
    fn kv_entry_round_trip() {
        let mut flash = FakeFlash::<32>::new(64);
        let mut buffer = [0u8; 32];

        let end_of_entry = write_kv(
            &mut flash,
            &mut buffer,
            Space::Zero,
            1,
            b"hello",
            b"world",
        ).expect("entry should write");

        let result = seek_kv_backwards(
            &mut flash,
            &mut buffer,
            Space::Zero,
            end_of_entry,
            b"hello",
        ).expect("entry should read back");

        assert_eq!(result, Some(1));

        let neg_result = seek_kv_backwards(
            &mut flash,
            &mut buffer,
            Space::Zero,
            end_of_entry,
            b"no",
        );
        match neg_result {
            Ok(None) => (),
            Ok(Some(s)) => panic!("nonexistent entry 'found' at sector {s}"),
            Err(e) => panic!("entry read should succeed, and yet, {:?}", e),
        }
    }

    #[test]
    fn format_without_erase_fails() {
        let mut flash = FakeFlash::<16>::new(64);
        // Program an arbitrary sector.
        flash.program_sector(Space::Zero, 7, &[0; 16]).unwrap();
        
        const G: u32 = 0xBAAD_F00D;
        let mut buffer = [0; 16];

        assert_eq!(
            format(&mut flash, &mut buffer, Space::Zero, G),
            Err(FormatError::NeedsErase),
        );
    }

    #[test]
    fn format_then_check() {
        let mut flash = FakeFlash::<16>::new(64);
        const G: u32 = 0xBAAD_F00D;
        let mut buffer = [0; 16];
        format(&mut flash, &mut buffer, Space::Zero, G)
            .expect("format should succeed");

        let result = check(&mut flash, &mut buffer, Space::Zero)
            .expect("check should not fail");

        assert_eq!(result, CheckResult::ValidLog {
            generation: G,
            end: 1,
            tail_erased: true,
            incomplete_write: false,
        });
    }

    #[test]
    fn format_write1_check() {
        let mut flash = FakeFlash::<16>::new(64);
        const G: u32 = 0xBAAD_F00D;
        let mut buffer = [0; 16];
        format(&mut flash, &mut buffer, Space::Zero, G)
            .expect("format should succeed");

        let end_of_entry =
            write_kv(&mut flash, &mut buffer, Space::Zero, 1, b"hi", b"there")
            .expect("write should succeed");
        assert_eq!(end_of_entry, 1 + 3);

        let result = check(&mut flash, &mut buffer, Space::Zero)
            .expect("check should not fail");

        assert_eq!(result, CheckResult::ValidLog {
            generation: G,
            end: 1 + 3,
            tail_erased: true,
            incomplete_write: false,
        });
    }

    #[test]
    fn format_write3_check() {
        let mut flash = FakeFlash::<16>::new(64);
        const G: u32 = 0xBAAD_F00D;
        let mut buffer = [0; 16];
        format(&mut flash, &mut buffer, Space::Zero, G)
            .expect("format should succeed");

        let mut p = 1;
        for _ in 0..3 {
            p =
                write_kv(&mut flash, &mut buffer, Space::Zero, p, b"hi", b"there")
                .expect("write should succeed");
        }

        let result = check(&mut flash, &mut buffer, Space::Zero)
            .expect("check should not fail");

        assert_eq!(result, CheckResult::ValidLog {
            generation: G,
            end: 1 + 3 * 3,
            tail_erased: true,
            incomplete_write: false,
        });
    }

    #[test]
    fn write_out_of_space() {
        let mut flash = FakeFlash::<16>::new(16);
        const G: u32 = 0xBAAD_F00D;
        let mut buffer = [0; 16];
        format(&mut flash, &mut buffer, Space::Zero, G)
            .expect("format should succeed");

        let mut p = 1;
        for _ in 0..5 {
            p =
                write_kv(&mut flash, &mut buffer, Space::Zero, p, b"hi", b"there")
                .expect("write should succeed");
        }

        let final_result = write_kv(&mut flash, &mut buffer, Space::Zero, p, b"hi", b"there");
        assert_eq!(Err(WriteError::NoSpace), final_result);

        let result = check(&mut flash, &mut buffer, Space::Zero)
            .expect("check should not fail");

        assert_eq!(result, CheckResult::ValidLog {
            generation: G,
            end: 1 + 5 * 3, // final one should not be reflected
            tail_erased: true,
            incomplete_write: false,
        });
    }

    #[test]
    fn check_incomplete() {
        let mut flash = FakeFlash::<16>::new(64);
        const G: u32 = 0xBAAD_F00D;
        let mut buffer = [0; 16];
        format(&mut flash, &mut buffer, Space::Zero, G)
            .expect("format should succeed");

        let end_of_entry =
            write_kv(&mut flash, &mut buffer, Space::Zero, 1, b"hi", b"there")
            .expect("write should succeed");
        // Sneakily erase all sectors but the first.
        for s in 2..end_of_entry {
            flash.erase_sector(Space::Zero, s);
        }

        let result = check(&mut flash, &mut buffer, Space::Zero)
            .expect("check should not fail");

        assert_eq!(result, CheckResult::ValidLog {
            generation: G,
            end: 1, // does not include incomplete entry
            tail_erased: true,
            incomplete_write: true, // <-- the point
        });
    }

    #[test]
    fn check_tail_not_erased() {
        let mut flash = FakeFlash::<16>::new(64);
        const G: u32 = 0xBAAD_F00D;
        let mut buffer = [0; 16];
        format(&mut flash, &mut buffer, Space::Zero, G)
            .expect("format should succeed");

        let end_of_entry =
            write_kv(&mut flash, &mut buffer, Space::Zero, 1, b"hi", b"there")
            .expect("write should succeed");
        // Sneakily program a sector after the entry.
        flash.program_sector(Space::Zero, end_of_entry + 2, &[0; 16]).unwrap();

        let result = check(&mut flash, &mut buffer, Space::Zero)
            .expect("check should not fail");

        assert_eq!(result, CheckResult::ValidLog {
            generation: G,
            end: 1 + 3,
            tail_erased: false, // <--- the point
            incomplete_write: false,
        });
    }

    #[test]
    fn check_data_erasure() {
        let mut flash = FakeFlash::<16>::new(64);
        const G: u32 = 0xBAAD_F00D;
        let mut buffer = [0; 16];
        format(&mut flash, &mut buffer, Space::Zero, G)
            .expect("format should succeed");

        let end_of_entry =
            write_kv(&mut flash, &mut buffer, Space::Zero, 1, b"hi", b"there")
            .expect("write should succeed");
        assert_eq!(end_of_entry, 4);
        // Nuke the central data sector.
        flash.erase_sector(Space::Zero, 2);

        let result = check(&mut flash, &mut buffer, Space::Zero)
            .expect("check should not fail");

        assert_eq!(result, CheckResult::Bad(CheckError::UnprogrammedData(1)));
    }

    #[test]
    fn evacuate_basic() {
        let mut flash = FakeFlash::<32>::new(64);
        let mut buffer = [0u8; 32];
        const G: u32 = 1;
        format(&mut flash, &mut buffer, Space::Zero, G)
            .expect("format should succeed");

        let end_of_entry = write_kv(
            &mut flash,
            &mut buffer,
            Space::Zero,
            1,
            b"hello",
            b"world",
        ).expect("entry should write");

        let mut buffer1 = [0u8; 32];
        evacuate(
            &mut flash,
            &mut buffer,
            &mut buffer1,
            Space::Zero,
            end_of_entry,
        ).expect("evacuate should succeed");

        let result = seek_kv_backwards(
            &mut flash,
            &mut buffer,
            Space::One,
            end_of_entry,
            b"hello",
        ).expect("entry should read back");

        assert_eq!(result, Some(1));
    }

    #[test]
    fn evacuate_overwrite() {
        let mut flash = FakeFlash::<32>::new(64);
        let mut buffer = [0u8; 32];
        const G: u32 = 1;
        format(&mut flash, &mut buffer, Space::Zero, G)
            .expect("format should succeed");

        let end_of_entry = write_kv(
            &mut flash,
            &mut buffer,
            Space::Zero,
            1,
            b"hello",
            b"world",
        ).expect("entry should write");
        let end_of_entry = write_kv(
            &mut flash,
            &mut buffer,
            Space::Zero,
            end_of_entry,
            b"hello",
            b"there",
        ).expect("entry should write");

        let mut buffer1 = [0u8; 32];
        evacuate(
            &mut flash,
            &mut buffer,
            &mut buffer1,
            Space::Zero,
            end_of_entry,
        ).expect("evacuate should succeed");

        /*
        let result = seek_kv_backwards(
            &mut flash,
            &mut buffer,
            Space::One,
            end_of_entry,
            b"hello",
        ).expect("entry should read back");

        assert_eq!(result, Some(1));
        */
    }
}
