use core::mem::size_of;
use core::marker::PhantomData;
use core::borrow::{Borrow, BorrowMut};
use zerocopy::{AsBytes, FromBytes, Unaligned};
use num_traits::FromPrimitive;

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

#[derive(Copy, Clone, Debug, FromBytes, AsBytes, Unaligned, Eq, PartialEq)]
#[repr(C)]
pub struct EntryHeader {
    /// Marker to designate an entry header and help distinguish it from
    /// unprogrammed or random data.
    pub magic: U16LE,
    /// Type of subheader.
    pub subtype: u8,
    /// Length of subheader (or subtrailer, as their lengths must match) in
    /// bytes.
    pub sub_bytes: u8,
    /// Length of contents separating the subheader from subtrailer. This length
    /// is in bytes; the actual contents will be followed by enough padding to
    /// justify the subtrailer/trailer to the end of the sector.
    pub contents_length: U32LE,
}

impl EntryHeader {
    /// Bits we expect to find in the `magic` field.
    pub const EXPECTED_MAGIC: u16 = 0xCB_F5;
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, num_derive::FromPrimitive)]
pub enum KnownSubtypes {
    // Note: 0 is reserved.

    Data = 0x01,
    Delete = 0x02,

    Aborted = 0xFE,

    // Note: 0xFF is reserved.
}

#[derive(Copy, Clone, Debug, FromBytes, AsBytes, Unaligned)]
#[repr(C)]
pub struct DataSubHeader {
    /// Number of bytes in the key.
    pub key_length: U32LE,
    /// Hash of the key bytes using FNV-1, to assist in key lookup.
    pub key_hash: U32LE,
}

impl DataSubHeader {
    pub const SIZE: usize = size_of::<Self>();
    pub const SUB_BYTES: u8 = Self::SIZE as u8;
}

const KEY_HASH_KEY: u64 = 0;

fn hash_key(key: &[u8]) -> u32 {
    use core::hash::{Hash, Hasher};

    let mut hasher = fnv::FnvHasher::with_key(KEY_HASH_KEY);
    key.hash(&mut hasher);
    let h = hasher.finish();
    h as u32 ^ (h >> 32) as u32
}

pub type DeleteSubHeader = DataSubHeader;

pub trait Flash {
    type Sector: Sized + BorrowMut<[u8]> + Borrow<[u8]>;
    type Error;

    fn sectors_per_space(&self) -> u32;

    fn read_sector(&self, space: Space, index: u32, dest: &mut Self::Sector) -> Result<(), Self::Error>;
    fn can_program_sector(&self, space: Space, index: u32) -> Result<bool, Self::Error>;
    fn can_read_sector(&self, space: Space, index: u32) -> Result<bool, Self::Error>;
    fn program_sector(&mut self, space: Space, index: u32, data: &Self::Sector) -> Result<(), Self::Error>;

    /// Compares `data.len()` bytes starting at `offset` from the start of
    /// sector `index` for equality. `offset` may be larger than a sector, for
    /// convenience.
    ///
    /// This is a "pushed compare" operation to take advantage of situations
    /// where we can do the compare without reading out every sector into RAM,
    /// such as directly-addressable flash.
    fn compare_contents(&self, space: Space, buffer: &mut Self::Sector, mut index: u32, offset: u32, mut data: &[u8]) -> Result<bool, Self::Error> {
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
            #[cfg(test)]
            println!("comparing:\n{:x?}\n{:x?}",
                &(*buffer0).borrow()[..n as usize],
                &(*buffer1).borrow()[..n as usize],
            );
            if (*buffer0).borrow()[..n as usize] != (*buffer1).borrow()[..n as usize] {
                return Ok(false);
            }
            #[cfg(test)]
            println!("equal");
            sector0 += 1;
            sector1 += 1;
            length -= n;
        }
        Ok(true)
    }

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

pub struct Constants<F>(PhantomData<F>);

impl<F: Flash> Constants<F> {
    pub const HEADER_SECTORS: u32 = {
        let sector = size_of::<F::Sector>();
        ((size_of::<SpaceHeader>() + sector - 1) / sector) as u32
    };
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Space {
    Zero = 0,
    One = 1,
}

impl Space {
    pub const ALL: [Self; 2] = [Self::Zero, Self::One];

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

pub fn bytes_to_sectors<F: Flash>(x: u32) -> u32 {
    let sector_size = size_of::<F::Sector>() as u32;
    (x + sector_size - 1) / sector_size
}


/// Creates a new empty log in device `flash` and space `current`.
///
/// This requires that the space has been erased. If it has not been erased,
/// this will fail with `FormatError::NeedsErase`.
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

pub fn check<F: Flash>(
    flash: &mut F,
    buffer0: &mut F::Sector,
    buffer1: &mut F::Sector,
    current: Space,
) -> Result<CheckResult, F::Error> {
    let sector_count = flash.sectors_per_space();

    if flash.can_program_sector(current, 0)? {
        // Well, there's no valid store here... let's distinguish between full
        // and partial erase.
        for sector in 0..sector_count {
            if !flash.can_program_sector(current, sector)? {
                return Ok(CheckResult::Bad(CheckError::PartiallyErased));
            }
        }

        return Ok(CheckResult::Bad(CheckError::Erased));
    }
    flash.read_sector(current, 0, buffer0)?;
    let (space_header, _) = cast_prefix::<SpaceHeader>((*buffer0).borrow());
    let generation = space_header.generation.get();

    if !space_header.check() {
        return Ok(CheckResult::Bad(CheckError::BadSpaceHeader));
    }
    if 1 << space_header.l2_sector_size != size_of::<F::Sector>() {
        return Ok(CheckResult::Bad(CheckError::WrongSectorSize));
    }

    let mut sector = Constants::<F>::HEADER_SECTORS;
    let mut incomplete_write = false;

    while sector < sector_count {
        let r = check_entry(flash, buffer0, buffer1, current, sector)?;
        #[cfg(test)]
        println!("check_entry result: {:?}", r);

        match r {
            CheckEntryResult::HeadErased => {
                // This is probably the end of the log in this space.
                break;
            }
            CheckEntryResult::IncompleteWrite(next) => {
                sector = next;
                incomplete_write = true;
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

            CheckEntryResult::ChecksPassed(next) => sector = next,
        }
    }

    
    let mut tail_erased = true;
    for s in sector+1..sector_count {
        if !flash.can_program_sector(current, s)? {
            tail_erased = false;
            break;
        }
    }
    Ok(CheckResult::ValidLog {
        generation,
        end: sector,
        incomplete_write,
        tail_erased,
    })
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum CheckResult {
    Bad(CheckError),

    /// A valid data store was found, with a valid header and some number of
    /// valid entries running up to sector index `end` (which is one past the
    /// end of valid data). This indicates that the store can be mounted at
    /// least read-only.
    ///
    /// To mount read-write, `tail_erased` must be true, and `incomplete_write`
    /// must be false. Or, repair action must be taken.
    ValidLog {
        generation: u32,
        end: u32,
        tail_erased: bool,
        incomplete_write: bool,
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum CheckError {
    /// The space is entirely erased. It can be used as the idle space with no
    /// further work.
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
    /// sector, making it unsafe to read. 
    UnprogrammedData(u32),
}

pub enum LogTailStatus {
    /// All sectors after the log are ready to be programmed.
    Erased,
    /// At least one sector after the log is already programmed. This means
    /// erasing is required before we can write.
    PartiallyErased,
    /// 
    IncompleteWrite,
}

/// Reads the entry starting at `head_sector` and checks that it's valid.
///
/// There are three classes of results from this operation.
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
    buffer0: &mut F::Sector,
    buffer1: &mut F::Sector,
    current: Space,
    head_sector: u32,
) -> Result<CheckEntryResult, F::Error> {
    #[cfg(test)]
    println!("checking entry at {head_sector}");

    if flash.can_program_sector(current, head_sector)? {
        return Ok(CheckEntryResult::HeadErased);
    }
    let head_info = match read_entry_from_head(flash, buffer0, current, head_sector) {
        Err(ReadError::Flash(e)) => return Err(e),
        Err(_) => return Ok(CheckEntryResult::HeadCorrupt),
        Ok(entry) => entry,
    };

    #[cfg(test)]
    println!("{head_info:#?}");

    if flash.can_program_sector(current, head_info.next_sector - 1)? {
        return Ok(CheckEntryResult::IncompleteWrite(head_info.next_sector));
    }

    let next_entry = head_info.next_sector;

    let tail_info = match read_entry_from_tail(flash, buffer1, current, next_entry) {
        Err(ReadError::Flash(e)) => return Err(e),
        Err(_) => return Ok(CheckEntryResult::TailCorrupt),
        Ok(entry) => entry,
    };
    
    if tail_info.next_sector != head_sector
        || tail_info.meta.contents_length != head_info.meta.contents_length
    {
        return Ok(CheckEntryResult::HeadTailMismatch);
    }

    // We permit only one kind of mismatch between header and trailer: the
    // trailer may be marked Aborted, regardless of how the header is marked. In
    // this case we still require the other fields to match.
    if tail_info.meta.subtype != KnownSubtypes::Aborted as u8
        && (tail_info.meta.subtype != head_info.meta.subtype 
            || tail_info.meta.sub_bytes != head_info.meta.sub_bytes)
    {
        return Ok(CheckEntryResult::HeadTailMismatch);
    }

    if tail_info.meta.subtype != KnownSubtypes::Aborted as u8 {
        // For any other subtype we require the entry's sectors to be totally
        // readable, which typically means programmed -- though some flashes
        // will expose unprogrammed sectors as all FF, which is ok here.
        for s in head_sector + 1..next_entry {
            if !flash.can_read_sector(current, s)? {
                return Ok(CheckEntryResult::PartiallyErased);
            }
        }
    }

    Ok(CheckEntryResult::ChecksPassed(next_entry))
}

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

#[derive(Copy, Clone, Debug)]
pub enum ReadError<E> {
    BadMagic(u32),
    BadSubBytes(u32),
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

pub(crate) fn seek_backwards<'b, F: Flash>(
    flash: &F,
    buffer: &'b mut F::Sector,
    current: Space,
    start_sector: u32,
    mut filter: impl FnMut(&F, &mut F::Sector, u32, KnownSubHeaders) -> Result<EntryDecision, F::Error>,
) -> Result<Option<u32>, ReadError<F::Error>> {
    let header_sectors = Constants::<F>::HEADER_SECTORS;

    assert!(start_sector >= header_sectors);

    let mut sector = start_sector;

    while sector > header_sectors {
        let entry = read_entry_from_tail(flash, buffer, current, sector)?;

        let head_sector = entry.next_sector;
        let ksh = KnownSubHeaders::new(entry.meta.subtype, entry.submeta);

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

#[derive(Copy, Clone, Debug)]
pub enum EntryDecision {
    Ignore,
    Accept,
    Abort,
}

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
                KnownSubHeaders::Data(sub) | KnownSubHeaders::Delete(sub) => {
                    // For these types, we want to check the key.
                    if sub.key_hash.get() == key_hash
                        && sub.key_length.get() == key_len
                    {
                        // A potential match!
                        let meta_bytes = size_of::<EntryHeader>()
                            + size_of::<DataSubHeader>();
                        let key_eq = flash.compare_contents(current, buffer, index, meta_bytes as u32, key)?;
                        if key_eq {
                            // Now, the difference between Data and Delete comes
                            // into play.
                            if matches!(ksub, KnownSubHeaders::Data(_)) {
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

#[derive(Copy, Clone, Debug)]
pub enum KnownSubHeaders {
    Data(DataSubHeader),
    Delete(DeleteSubHeader),
    Aborted,
    Other(u8),
}

impl KnownSubHeaders {
    pub fn new(subtype: u8, submeta: &[u8]) -> Self {
        match KnownSubtypes::from_u8(subtype) {
            Some(KnownSubtypes::Data) => Self::Data(*cast_prefix(submeta).0),
            Some(KnownSubtypes::Delete) => Self::Delete(*cast_prefix(submeta).0),
            Some(KnownSubtypes::Aborted) => Self::Aborted,
            None => Self::Other(subtype),
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct EntryInfo<'a> {
    pub next_sector: u32,
    pub meta: &'a EntryHeader,
    pub submeta: &'a [u8],
}

/// Reads an entry from the end; `start_sector` is the index of the sector _one
/// past the end_ of the entry.
///
/// If this returns `Ok`, `buffer` will also contain a copy of the entry's
/// trailer. This is a deliberate side effect.
pub(crate) fn read_entry_from_tail<'b, F: Flash>(
    flash: &F,
    buffer: &'b mut F::Sector,
    current: Space,
    start_sector: u32,
) -> Result<EntryInfo<'b>, ReadError<F::Error>> {
    let sector = start_sector - 1;

    flash.read_sector(current, sector, buffer)?;
    let data = (*buffer).borrow();
    let (data, meta) = cast_suffix::<EntryHeader>(data);

    if meta.magic.get() != EntryHeader::EXPECTED_MAGIC {
        return Err(ReadError::BadMagic(sector));
    }
    let submeta_start = data.len().checked_sub(usize::from(meta.sub_bytes))
        .ok_or(ReadError::BadSubBytes(sector))?;
    let submeta = &data[submeta_start..];

    let meta_bytes = size_of::<EntryHeader>() as u32
        + u32::from(meta.sub_bytes);

    let entry_length = 2 * meta_bytes + meta.contents_length.get();
    let next_trailer = sector
        .checked_sub(bytes_to_sectors::<F>(entry_length))
        .ok_or(ReadError::BadLength(sector))?;
    let next_sector = next_trailer + 1;

    Ok(EntryInfo {
        next_sector,
        meta,
        submeta,
    })
}

pub fn read_contents<F: Flash>(
    flash: &mut F,
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
    let value_offset = size_of::<EntryHeader>()
        + size_of::<DataSubHeader>()
        + offset as usize;
    assert!(offset <= entry.meta.contents_length.get(),
        "can't read at offset {offset} into data of size {}",
        entry.meta.contents_length.get());
    let value_len = entry.meta.contents_length.get() as usize
        - offset as usize;
    let value_len = value_len.min(out.len());

    let mut sector = value_offset / size_of::<F::Sector>()
        + head_sector as usize;
    let mut offset = value_offset % size_of::<F::Sector>();
    let mut out = &mut out[..value_len];

    while offset >= size_of::<F::Sector>() {
        offset -= size_of::<F::Sector>();
        sector += 1;
    }

    while !out.is_empty() {
        flash.read_sector(
            current,
            sector as u32,
            buffer,
        )?;

        let n = (size_of::<F::Sector>() - offset).min(value_len);
        out[..n].copy_from_slice(&(*buffer).borrow()[offset..offset + n]);
        offset = 0;
        sector += 1;
        out = &mut out[n..];
    }

    Ok(value_len)
}

pub fn read_entry_from_head<'b, F: Flash>(
    flash: &F,
    buffer: &'b mut F::Sector,
    current: Space,
    sector: u32,
) -> Result<EntryInfo<'b>, ReadError<F::Error>> {
    flash.read_sector(current, sector, buffer)?;
    let data = (*buffer).borrow();
    let (meta, data) = cast_prefix::<EntryHeader>(data);

    if meta.magic.get() != EntryHeader::EXPECTED_MAGIC {
        return Err(ReadError::BadMagic(sector));
    }
    let submeta = data.get(..usize::from(meta.sub_bytes))
        .ok_or(ReadError::BadSubBytes(sector))?;

    let meta_bytes = size_of::<EntryHeader>() as u32
        + u32::from(meta.sub_bytes);

    let entry_length = 2 * meta_bytes + meta.contents_length.get();
    let next_sector = sector.checked_add(bytes_to_sectors::<F>(entry_length))
        .ok_or(ReadError::BadLength(sector))?;

    Ok(EntryInfo {
        next_sector,
        meta,
        submeta,
    })
}

pub(crate) fn write_kv<F: Flash>(
    flash: &mut F,
    buffer: &mut F::Sector,
    current: Space,
    start_sector: u32,
    key: &[u8],
    value: &[u8],
) -> Result<u32, WriteError<F::Error>> {
    let key_len = u32::try_from(key.len())
        .expect("key too long");
    let value_len = u32::try_from(value.len())
        .expect("value too long");
    let contents_length = key_len.checked_add(value_len)
        .expect("key+value too long");

    let header = EntryHeader {
        magic: EntryHeader::EXPECTED_MAGIC.into(),
        subtype: KnownSubtypes::Data as u8,
        sub_bytes: DataSubHeader::SUB_BYTES,
        contents_length: contents_length.into(),
    };
    let subheader = DataSubHeader {
        key_length: key_len.into(),
        key_hash: hash_key(key).into(),
    };

    write_entry(
        flash,
        buffer,
        current,
        start_sector,
        &[
            header.as_bytes(),
            subheader.as_bytes(),
            key,
            value,
        ],
        subheader.as_bytes(),
        header.as_bytes(),
    )
}

pub(crate) fn write_entry<F: Flash>(
    flash: &mut F,
    buffer: &mut F::Sector,
    current: Space,
    start_sector: u32,
    pieces: &[&[u8]],
    subtrailer: &[u8],
    trailer: &[u8],
) -> Result<u32, WriteError<F::Error>> {
    let total_length = pieces.iter().map(|p| p.len()).sum::<usize>()
        + subtrailer.len()
        + trailer.len();
    let total_length = u32::try_from(total_length).unwrap();
    let total_sectors = bytes_to_sectors::<F>(total_length);
    if flash.sectors_per_space() - start_sector < total_sectors {
        return Err(WriteError::NoSpace);
    }

    let mut sector = start_sector;
    let mut data = buffer.borrow_mut();
    for mut piece in pieces.iter().cloned() {
        while !piece.is_empty() {
            let n = usize::min(piece.len(), data.len());
            let (piece0, piece1) = piece.split_at(n);
            let (data0, data1) = data.split_at_mut(n);
            data0.copy_from_slice(piece0);
            piece = piece1;
            data = data1;

            if data.is_empty() {
                drop(data);

                flash.program_sector(current, sector, buffer)?;
                sector += 1;

                data = buffer.borrow_mut();
            }
        }
    }

    if data.len() < subtrailer.len() + trailer.len() {
        // We have to burn a sector on the trailer. Flush the data, but
        // zero-fill the end of the buffer first.
        data.fill(0);
        drop(data);

        flash.program_sector(current, sector, buffer)?;
        sector += 1;

        data = buffer.borrow_mut();
    }

    {
        // Split remaining sector tail into unused area, which will be filled
        // with padding, and the trailer part.
        let dl = data.len();
        let unused = dl - trailer.len() - subtrailer.len();
        let (pad, tail) = data.split_at_mut(unused);
        pad.fill(0);

        let (sub, tail) = tail.split_at_mut(subtrailer.len());
        sub.copy_from_slice(subtrailer);
        tail.copy_from_slice(trailer);
    }
    drop(data);

    flash.program_sector(current, sector, buffer)?;

    Ok(sector + 1)
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum WriteError<E> {
    NoSpace,
    Flash(E),
}

impl<E> From<E> for WriteError<E> {
    fn from(e: E) -> Self {
        Self::Flash(e)
    }
}

pub fn evacuate<F: Flash>(
    flash: &mut F,
    buffer0: &mut F::Sector,
    buffer1: &mut F::Sector,
    from_space: Space,
    watermark: u32,
) -> Result<(), ReadError<F::Error>> {
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

        let meta_bytes = size_of::<EntryHeader>() as u32
            + u32::from(entry.meta.sub_bytes);
        let entry_bytes = 2 * meta_bytes
            + entry.meta.contents_length.get();
        let entry_sectors = bytes_to_sectors::<F>(entry_bytes);

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

                let (_, subtrailer) = cast_suffix::<DataSubHeader>(entry.submeta);
                let key_hash = subtrailer.key_hash.get();
                let key_len = subtrailer.key_length.get();

                let pred = seek_backwards(
                    flash,
                    buffer0,
                    to_space,
                    to_sector,
                    |flash, buf, s, sub| {
                        match sub {
                            KnownSubHeaders::Data(sub) | KnownSubHeaders::Delete(sub) => {
                                if sub.key_hash.get() == key_hash
                                    && sub.key_length.get() == key_len
                                {
                                    let key_eq = flash.compare_internal(
                                        from_space,
                                        head_sector,
                                        to_space,
                                        s,
                                        key_len + meta_bytes,
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

    Ok(())
}

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
        sec1[0..=1].copy_from_slice(&EntryHeader::EXPECTED_MAGIC.to_le_bytes());
        sec1[2] = 0xAA;
        sec1[3] = 4;
        sec1[4..=7].copy_from_slice(&8_u32.to_le_bytes());

        sec1[8..=0xB].copy_from_slice(&0xDEAD_BEEF_u32.to_le_bytes());

        sec1[0x0C..=0x13].fill(0xFF);

        sec1[0x14..=0x17].copy_from_slice(&0xDEAD_BEEF_u32.to_le_bytes());

        sec1[0x18..=0x19].copy_from_slice(&EntryHeader::EXPECTED_MAGIC.to_le_bytes());
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

        assert_eq!(entry_info.meta, &EntryHeader {
            magic: EntryHeader::EXPECTED_MAGIC.into(),
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

        let mut buffer1 = [0; 16];
        let result = check(&mut flash, &mut buffer, &mut buffer1, Space::Zero)
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

        let mut buffer1 = [0; 16];
        let result = check(&mut flash, &mut buffer, &mut buffer1, Space::Zero)
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

        let mut buffer1 = [0; 16];
        let result = check(&mut flash, &mut buffer, &mut buffer1, Space::Zero)
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

        let mut buffer1 = [0; 16];
        let result = check(&mut flash, &mut buffer, &mut buffer1, Space::Zero)
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

        let mut buffer1 = [0; 16];
        let result = check(&mut flash, &mut buffer, &mut buffer1, Space::Zero)
            .expect("check should not fail");

        assert_eq!(result, CheckResult::ValidLog {
            generation: G,
            end: 1 + 3,
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

        let mut buffer1 = [0; 16];
        let result = check(&mut flash, &mut buffer, &mut buffer1, Space::Zero)
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

        let mut buffer1 = [0; 16];
        let result = check(&mut flash, &mut buffer, &mut buffer1, Space::Zero)
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
