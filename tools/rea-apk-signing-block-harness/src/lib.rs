//! Bounded structural inventory of the Android APK Signing Block.
//!
//! This module does not verify signatures or establish signer identity, trust, authorization,
//! installability, runtime behavior, or software safety. It preserves raw ID/value occurrences
//! as observations and recognizes only documented scheme-presence IDs.

use sha2::{Digest, Sha256};
use std::fmt;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

const EOCD_SIGNATURE: u32 = 0x0605_4b50;
const CENTRAL_DIRECTORY_SIGNATURE: u32 = 0x0201_4b50;
const EOCD_FIXED_BYTES: usize = 22;
const EOCD_MAX_COMMENT_BYTES: usize = u16::MAX as usize;
const SIGNING_BLOCK_MAGIC: &[u8; 16] = b"APK Sig Block 42";

const APK_V2_BLOCK_ID: u32 = 0x7109_871a;
const APK_V3_BLOCK_ID: u32 = 0xf053_68c0;
const APK_V3_2_BLOCK_ID: u32 = 0x70e1_c89f;

const HARD_MAX_ARCHIVE_BYTES: u64 = 4 * 1024 * 1024 * 1024;
const HARD_MAX_CENTRAL_DIRECTORY_BYTES: u64 = 64 * 1024 * 1024;
const HARD_MAX_SIGNING_BLOCK_BYTES: u64 = 128 * 1024 * 1024;
const HARD_MAX_PAIRS: usize = 4096;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScanLimits {
    pub max_archive_bytes: u64,
    pub max_central_directory_bytes: u64,
    pub max_entries: u64,
    pub max_signing_block_bytes: u64,
    pub max_pairs: usize,
    pub max_total_pair_value_bytes: u64,
}

impl Default for ScanLimits {
    fn default() -> Self {
        Self {
            max_archive_bytes: HARD_MAX_ARCHIVE_BYTES,
            max_central_directory_bytes: HARD_MAX_CENTRAL_DIRECTORY_BYTES,
            max_entries: 100_000,
            max_signing_block_bytes: HARD_MAX_SIGNING_BLOCK_BYTES,
            max_pairs: HARD_MAX_PAIRS,
            max_total_pair_value_bytes: HARD_MAX_SIGNING_BLOCK_BYTES,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScanError {
    Io(String),
    Rejected(String),
    Unsupported(String),
    SourceChanged,
}

impl fmt::Display for ScanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(detail) => write!(f, "APK scan I/O error: {detail}"),
            Self::Rejected(detail) => write!(f, "APK structure rejected: {detail}"),
            Self::Unsupported(detail) => write!(f, "APK layout unsupported: {detail}"),
            Self::SourceChanged => write!(f, "APK source bytes changed during scan"),
        }
    }
}

impl std::error::Error for ScanError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchemeKind {
    V2,
    V3,
    V3_2HybridPqc,
}

impl SchemeKind {
    pub fn from_block_id(id: u32) -> Option<Self> {
        match id {
            APK_V2_BLOCK_ID => Some(Self::V2),
            APK_V3_BLOCK_ID => Some(Self::V3),
            APK_V3_2_BLOCK_ID => Some(Self::V3_2HybridPqc),
            _ => None,
        }
    }

    pub fn block_id(self) -> u32 {
        match self {
            Self::V2 => APK_V2_BLOCK_ID,
            Self::V3 => APK_V3_BLOCK_ID,
            Self::V3_2HybridPqc => APK_V3_2_BLOCK_ID,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PairOccurrence {
    pub occurrence_index: usize,
    pub id: u32,
    pub pair_offset: u64,
    pub value_offset: u64,
    pub value_length: u64,
    pub value_sha256: String,
    pub recognized_scheme: Option<SchemeKind>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SigningBlock {
    pub start_offset: u64,
    pub end_offset: u64,
    pub encoded_size_excluding_leading_field: u64,
    pub block_sha256: String,
    pub pairs: Vec<PairOccurrence>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApkSigningBlockReport {
    pub source_size_bytes: u64,
    pub source_sha256: String,
    pub central_directory_offset: u64,
    pub central_directory_size_bytes: u64,
    pub central_directory_entries: u64,
    /// None means no valid footer magic was observed immediately before the central directory.
    /// It is not equivalent to a failed or valid APK signature.
    pub signing_block: Option<SigningBlock>,
}

#[derive(Debug, Clone, Copy)]
struct Eocd {
    absolute_offset: u64,
    central_directory_offset: u64,
    central_directory_size: u64,
    entries: u64,
}

/// Open the source once; hash, inspect and re-hash this same opened file object.
pub fn scan_apk(
    path: impl AsRef<Path>,
    limits: ScanLimits,
) -> Result<ApkSigningBlockReport, ScanError> {
    validate_limits(limits)?;
    let mut file = File::open(path.as_ref()).map_err(io_error)?;
    scan_open_file(&mut file, limits)
}

/// Public entry point for already-open descriptors and deterministic in-memory test cursors.
pub fn scan_open_file<R: Read + Seek>(
    file: &mut R,
    limits: ScanLimits,
) -> Result<ApkSigningBlockReport, ScanError> {
    validate_limits(limits)?;
    let source_length = file.seek(SeekFrom::End(0)).map_err(io_error)?;
    if source_length > limits.max_archive_bytes {
        return Err(ScanError::Rejected(
            "archive exceeds configured byte limit".into(),
        ));
    }

    let (source_sha256, observed_length) = hash_source(file, limits.max_archive_bytes)?;
    if observed_length != source_length {
        return Err(ScanError::SourceChanged);
    }

    let eocd = locate_eocd(file, source_length)?;
    validate_central_directory(file, eocd, limits)?;

    let signing_block = locate_signing_block(file, eocd, limits)?;

    let (source_sha256_after, observed_length_after) = hash_source(file, limits.max_archive_bytes)?;
    if observed_length_after != source_length || source_sha256_after != source_sha256 {
        return Err(ScanError::SourceChanged);
    }

    Ok(ApkSigningBlockReport {
        source_size_bytes: source_length,
        source_sha256,
        central_directory_offset: eocd.central_directory_offset,
        central_directory_size_bytes: eocd.central_directory_size,
        central_directory_entries: eocd.entries,
        signing_block,
    })
}

fn validate_limits(limits: ScanLimits) -> Result<(), ScanError> {
    if limits.max_archive_bytes == 0
        || limits.max_archive_bytes > HARD_MAX_ARCHIVE_BYTES
        || limits.max_central_directory_bytes == 0
        || limits.max_central_directory_bytes > HARD_MAX_CENTRAL_DIRECTORY_BYTES
        || limits.max_signing_block_bytes == 0
        || limits.max_signing_block_bytes > HARD_MAX_SIGNING_BLOCK_BYTES
        || limits.max_pairs == 0
        || limits.max_pairs > HARD_MAX_PAIRS
        || limits.max_total_pair_value_bytes > limits.max_signing_block_bytes
    {
        return Err(ScanError::Rejected(
            "configured scan limits exceed hard safety bounds".into(),
        ));
    }
    Ok(())
}

fn locate_eocd<R: Read + Seek>(file: &mut R, source_length: u64) -> Result<Eocd, ScanError> {
    let tail_length = source_length.min((EOCD_FIXED_BYTES + EOCD_MAX_COMMENT_BYTES) as u64);
    let tail_start = source_length
        .checked_sub(tail_length)
        .ok_or_else(|| ScanError::Rejected("EOCD search window underflow".into()))?;
    let mut tail = vec![
        0u8;
        usize::try_from(tail_length).map_err(|_| {
            ScanError::Rejected("EOCD search window does not fit address space".into())
        })?
    ];
    read_exact_at(file, tail_start, &mut tail)?;

    let mut candidates = Vec::new();
    if tail.len() >= EOCD_FIXED_BYTES {
        for offset in (0..=tail.len() - EOCD_FIXED_BYTES).rev() {
            if read_u32(&tail, offset) != EOCD_SIGNATURE {
                continue;
            }
            let comment_length = read_u16(&tail, offset + 20) as usize;
            if offset + EOCD_FIXED_BYTES + comment_length == tail.len() {
                candidates.push(offset);
            }
        }
    }
    if candidates.is_empty() {
        return Err(ScanError::Rejected(
            "no EOCD candidate terminates at source EOF".into(),
        ));
    }
    if candidates.len() != 1 {
        return Err(ScanError::Rejected("EOCD candidate is ambiguous".into()));
    }

    let offset = candidates[0];
    let disk_number = read_u16(&tail, offset + 4);
    let central_disk = read_u16(&tail, offset + 6);
    let entries_on_disk = read_u16(&tail, offset + 8);
    let entries_total = read_u16(&tail, offset + 10);
    let central_directory_size = u64::from(read_u32(&tail, offset + 12));
    let central_directory_offset = u64::from(read_u32(&tail, offset + 16));

    if disk_number != 0 || central_disk != 0 || entries_on_disk != entries_total {
        return Err(ScanError::Unsupported(
            "multi-disk/split ZIP is outside the APK scanner subset".into(),
        ));
    }
    if entries_total == u16::MAX
        || central_directory_size == u64::from(u32::MAX)
        || central_directory_offset == u64::from(u32::MAX)
    {
        return Err(ScanError::Unsupported(
            "ZIP64 APK layout is outside this first scanner subset".into(),
        ));
    }

    Ok(Eocd {
        absolute_offset: tail_start + offset as u64,
        central_directory_offset,
        central_directory_size,
        entries: u64::from(entries_total),
    })
}

fn validate_central_directory<R: Read + Seek>(
    file: &mut R,
    eocd: Eocd,
    limits: ScanLimits,
) -> Result<(), ScanError> {
    if eocd.central_directory_size > limits.max_central_directory_bytes {
        return Err(ScanError::Rejected(
            "central directory exceeds configured byte limit".into(),
        ));
    }
    if eocd.entries > limits.max_entries {
        return Err(ScanError::Rejected(
            "central directory exceeds configured entry-count limit".into(),
        ));
    }
    let central_end = eocd
        .central_directory_offset
        .checked_add(eocd.central_directory_size)
        .ok_or_else(|| ScanError::Rejected("central directory range overflows".into()))?;
    if central_end != eocd.absolute_offset {
        return Err(ScanError::Rejected(
            "central directory is not immediately adjacent to EOCD or EOCD has trailing records"
                .into(),
        ));
    }

    let mut cursor = eocd.central_directory_offset;
    for _ in 0..eocd.entries {
        let fixed_end = cursor.checked_add(46).ok_or_else(|| {
            ScanError::Rejected("central-directory record offset overflows".into())
        })?;
        if fixed_end > central_end {
            return Err(ScanError::Rejected(
                "truncated central-directory file header".into(),
            ));
        }
        let mut fixed = [0u8; 46];
        read_exact_at(file, cursor, &mut fixed)?;
        if read_u32(&fixed, 0) != CENTRAL_DIRECTORY_SIGNATURE {
            return Err(ScanError::Rejected(
                "invalid central-directory file-header signature".into(),
            ));
        }
        let name_length = u64::from(read_u16(&fixed, 28));
        let extra_length = u64::from(read_u16(&fixed, 30));
        let comment_length = u64::from(read_u16(&fixed, 32));
        let disk_start = read_u16(&fixed, 34);
        if disk_start != 0 {
            return Err(ScanError::Unsupported(
                "central-directory record references another disk".into(),
            ));
        }
        let variable_length = name_length
            .checked_add(extra_length)
            .and_then(|v| v.checked_add(comment_length))
            .ok_or_else(|| {
                ScanError::Rejected("central-directory variable lengths overflow".into())
            })?;
        cursor = fixed_end.checked_add(variable_length).ok_or_else(|| {
            ScanError::Rejected("central-directory record range overflows".into())
        })?;
        if cursor > central_end {
            return Err(ScanError::Rejected(
                "central-directory variable fields exceed declared directory".into(),
            ));
        }
    }
    if cursor != central_end {
        return Err(ScanError::Rejected(
            "central-directory byte count does not match the EOCD entry count".into(),
        ));
    }
    Ok(())
}

fn locate_signing_block<R: Read + Seek>(
    file: &mut R,
    eocd: Eocd,
    limits: ScanLimits,
) -> Result<Option<SigningBlock>, ScanError> {
    if eocd.central_directory_offset < 24 {
        return Ok(None);
    }
    let magic_offset = eocd.central_directory_offset - 16;
    let mut magic = [0u8; 16];
    read_exact_at(file, magic_offset, &mut magic)?;
    if &magic != SIGNING_BLOCK_MAGIC {
        return Ok(None);
    }

    let trailing_size_offset = magic_offset
        .checked_sub(8)
        .ok_or_else(|| ScanError::Rejected("signing-block footer underflows source".into()))?;
    let mut trailing_size_bytes = [0u8; 8];
    read_exact_at(file, trailing_size_offset, &mut trailing_size_bytes)?;
    let encoded_size = u64::from_le_bytes(trailing_size_bytes);
    if encoded_size < 24 {
        return Err(ScanError::Rejected(
            "APK Signing Block size is smaller than its footer".into(),
        ));
    }
    if encoded_size > limits.max_signing_block_bytes {
        return Err(ScanError::Rejected(
            "APK Signing Block exceeds configured byte limit".into(),
        ));
    }
    let total_length = encoded_size
        .checked_add(8)
        .ok_or_else(|| ScanError::Rejected("APK Signing Block total length overflows".into()))?;
    let block_start = eocd
        .central_directory_offset
        .checked_sub(total_length)
        .ok_or_else(|| ScanError::Rejected("APK Signing Block start underflows source".into()))?;
    let block_end = block_start
        .checked_add(total_length)
        .ok_or_else(|| ScanError::Rejected("APK Signing Block end overflows".into()))?;
    if block_end != eocd.central_directory_offset {
        return Err(ScanError::Rejected(
            "APK Signing Block does not end at central-directory start".into(),
        ));
    }

    let mut leading_size_bytes = [0u8; 8];
    read_exact_at(file, block_start, &mut leading_size_bytes)?;
    let leading_size = u64::from_le_bytes(leading_size_bytes);
    if leading_size != encoded_size {
        return Err(ScanError::Rejected(
            "APK Signing Block leading/trailing size fields differ".into(),
        ));
    }

    let pairs_start = block_start
        .checked_add(8)
        .ok_or_else(|| ScanError::Rejected("APK Signing Block pair start overflows".into()))?;
    let pairs_end = trailing_size_offset;
    if pairs_start > pairs_end {
        return Err(ScanError::Rejected(
            "APK Signing Block has an inverted pair region".into(),
        ));
    }

    let mut cursor = pairs_start;
    let mut total_value_bytes = 0u64;
    let mut pairs = Vec::new();
    while cursor < pairs_end {
        if pairs.len() >= limits.max_pairs {
            return Err(ScanError::Rejected(
                "APK Signing Block pair count exceeds configured limit".into(),
            ));
        }
        let length_end = cursor
            .checked_add(8)
            .ok_or_else(|| ScanError::Rejected("APK pair length field overflows".into()))?;
        if length_end > pairs_end {
            return Err(ScanError::Rejected(
                "truncated APK Signing Block pair length".into(),
            ));
        }
        let mut pair_length_bytes = [0u8; 8];
        read_exact_at(file, cursor, &mut pair_length_bytes)?;
        let pair_length = u64::from_le_bytes(pair_length_bytes);
        if pair_length < 4 {
            return Err(ScanError::Rejected(
                "APK Signing Block pair length is smaller than its ID".into(),
            ));
        }
        let pair_end = length_end
            .checked_add(pair_length)
            .ok_or_else(|| ScanError::Rejected("APK Signing Block pair end overflows".into()))?;
        if pair_end > pairs_end {
            return Err(ScanError::Rejected(
                "APK Signing Block pair extends beyond pair region".into(),
            ));
        }

        let mut id_bytes = [0u8; 4];
        read_exact_at(file, length_end, &mut id_bytes)?;
        let id = u32::from_le_bytes(id_bytes);
        let value_offset = length_end + 4;
        let value_length = pair_length - 4;
        total_value_bytes = total_value_bytes.checked_add(value_length).ok_or_else(|| {
            ScanError::Rejected("APK Signing Block aggregate value length overflows".into())
        })?;
        if total_value_bytes > limits.max_total_pair_value_bytes {
            return Err(ScanError::Rejected(
                "APK Signing Block aggregate pair values exceed configured limit".into(),
            ));
        }

        let value_sha256 = hash_range(file, value_offset, value_length)?;
        pairs.push(PairOccurrence {
            occurrence_index: pairs.len(),
            id,
            pair_offset: cursor,
            value_offset,
            value_length,
            value_sha256,
            recognized_scheme: SchemeKind::from_block_id(id),
        });
        cursor = pair_end;
    }
    if cursor != pairs_end {
        return Err(ScanError::Rejected(
            "APK Signing Block pair parser did not end at footer".into(),
        ));
    }
    if pairs.is_empty() {
        return Err(ScanError::Rejected(
            "APK Signing Block contains no ID/value pairs".into(),
        ));
    }

    let block_sha256 = hash_range(file, block_start, total_length)?;
    Ok(Some(SigningBlock {
        start_offset: block_start,
        end_offset: block_end,
        encoded_size_excluding_leading_field: encoded_size,
        block_sha256,
        pairs,
    }))
}

fn hash_source<R: Read + Seek>(file: &mut R, limit: u64) -> Result<(String, u64), ScanError> {
    let initial_length = file.seek(SeekFrom::End(0)).map_err(io_error)?;
    if initial_length > limit {
        return Err(ScanError::Rejected("archive exceeds byte limit".into()));
    }
    file.seek(SeekFrom::Start(0)).map_err(io_error)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    let mut total = 0u64;
    loop {
        let read = file.read(&mut buffer).map_err(io_error)?;
        if read == 0 {
            break;
        }
        total = total
            .checked_add(read as u64)
            .ok_or_else(|| ScanError::Rejected("source byte count overflows".into()))?;
        if total > limit {
            return Err(ScanError::Rejected(
                "source read exceeded byte limit".into(),
            ));
        }
        hasher.update(&buffer[..read]);
    }
    let final_length = file.seek(SeekFrom::End(0)).map_err(io_error)?;
    if total != initial_length || total != final_length {
        return Err(ScanError::SourceChanged);
    }
    Ok((format!("{:x}", hasher.finalize()), total))
}

fn hash_range<R: Read + Seek>(file: &mut R, offset: u64, length: u64) -> Result<String, ScanError> {
    let end = offset
        .checked_add(length)
        .ok_or_else(|| ScanError::Rejected("hash range overflows".into()))?;
    let source_length = file.seek(SeekFrom::End(0)).map_err(io_error)?;
    if end > source_length {
        return Err(ScanError::Rejected(
            "hash range extends beyond source".into(),
        ));
    }
    file.seek(SeekFrom::Start(offset)).map_err(io_error)?;
    let mut remaining = length;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    while remaining > 0 {
        let amount = usize::try_from(remaining.min(buffer.len() as u64)).unwrap_or(buffer.len());
        file.read_exact(&mut buffer[..amount]).map_err(io_error)?;
        hasher.update(&buffer[..amount]);
        remaining -= amount as u64;
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn read_exact_at<R: Read + Seek>(
    file: &mut R,
    offset: u64,
    buffer: &mut [u8],
) -> Result<(), ScanError> {
    let end = offset
        .checked_add(buffer.len() as u64)
        .ok_or_else(|| ScanError::Rejected("read range overflows".into()))?;
    let length = file.seek(SeekFrom::End(0)).map_err(io_error)?;
    if end > length {
        return Err(ScanError::Rejected("read extends beyond source".into()));
    }
    file.seek(SeekFrom::Start(offset)).map_err(io_error)?;
    file.read_exact(buffer).map_err(io_error)
}

fn read_u16(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([bytes[offset], bytes[offset + 1]])
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
}

fn io_error(error: std::io::Error) -> ScanError {
    ScanError::Io(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn put_u16(out: &mut Vec<u8>, value: u16) {
        out.extend_from_slice(&value.to_le_bytes());
    }
    fn put_u32(out: &mut Vec<u8>, value: u32) {
        out.extend_from_slice(&value.to_le_bytes());
    }
    fn put_u64(out: &mut Vec<u8>, value: u64) {
        out.extend_from_slice(&value.to_le_bytes());
    }

    fn central_directory_record() -> Vec<u8> {
        let mut out = vec![0u8; 46];
        out[0..4].copy_from_slice(&CENTRAL_DIRECTORY_SIGNATURE.to_le_bytes());
        out
    }

    fn eocd(entries: u16, cd_size: u32, cd_offset: u32, comment: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        put_u32(&mut out, EOCD_SIGNATURE);
        put_u16(&mut out, 0);
        put_u16(&mut out, 0);
        put_u16(&mut out, entries);
        put_u16(&mut out, entries);
        put_u32(&mut out, cd_size);
        put_u32(&mut out, cd_offset);
        put_u16(&mut out, comment.len() as u16);
        out.extend_from_slice(comment);
        out
    }

    fn signing_block(pairs: &[(u32, &[u8])]) -> Vec<u8> {
        let mut pair_bytes = Vec::new();
        for (id, value) in pairs {
            put_u64(&mut pair_bytes, (4 + value.len()) as u64);
            put_u32(&mut pair_bytes, *id);
            pair_bytes.extend_from_slice(value);
        }
        let encoded_size = pair_bytes.len() as u64 + 24;
        let mut out = Vec::new();
        put_u64(&mut out, encoded_size);
        out.extend_from_slice(&pair_bytes);
        put_u64(&mut out, encoded_size);
        out.extend_from_slice(SIGNING_BLOCK_MAGIC);
        out
    }

    fn apk_with_pairs(pairs: &[(u32, &[u8])]) -> Vec<u8> {
        let mut out = vec![0x41u8; 64];
        let block = signing_block(pairs);
        out.extend_from_slice(&block);
        let cd = central_directory_record();
        let cd_offset = out.len() as u32;
        out.extend_from_slice(&cd);
        let end = eocd(1, cd.len() as u32, cd_offset, &[]);
        out.extend_from_slice(&end);
        out
    }

    fn apk_without_block() -> Vec<u8> {
        let mut out = vec![0x41u8; 64];
        let cd = central_directory_record();
        let cd_offset = out.len() as u32;
        out.extend_from_slice(&cd);
        out.extend_from_slice(&eocd(1, cd.len() as u32, cd_offset, &[]));
        out
    }

    fn scan_bytes(bytes: Vec<u8>) -> Result<ApkSigningBlockReport, ScanError> {
        scan_open_file(&mut Cursor::new(bytes), ScanLimits::default())
    }

    #[test]
    fn inventories_v2_v3_v32_and_unknown_pairs_in_order() {
        let bytes = apk_with_pairs(&[
            (APK_V2_BLOCK_ID, b"v2 opaque"),
            (0xdead_beef, b"unknown one"),
            (APK_V3_BLOCK_ID, b"v3 opaque"),
            (APK_V3_2_BLOCK_ID, b"v3.2 hybrid opaque"),
            (0xdead_beef, b"unknown duplicate"),
        ]);
        let report = scan_bytes(bytes).unwrap();
        let block = report.signing_block.unwrap();
        assert_eq!(block.pairs.len(), 5);
        assert_eq!(block.pairs[0].recognized_scheme, Some(SchemeKind::V2));
        assert_eq!(block.pairs[1].recognized_scheme, None);
        assert_eq!(block.pairs[2].recognized_scheme, Some(SchemeKind::V3));
        assert_eq!(
            block.pairs[3].recognized_scheme,
            Some(SchemeKind::V3_2HybridPqc)
        );
        assert_eq!(block.pairs[4].id, 0xdead_beef);
        assert_eq!(block.pairs[4].occurrence_index, 4);
        assert_ne!(block.pairs[1].value_sha256, block.pairs[4].value_sha256);
    }

    #[test]
    fn ordinary_zip_returns_no_signing_block() {
        let report = scan_bytes(apk_without_block()).unwrap();
        assert!(report.signing_block.is_none());
    }

    #[test]
    fn signing_block_magic_inside_entry_payload_is_not_a_scheme_observation() {
        let mut bytes = apk_without_block();
        let cd_offset = 64usize;
        let position = 32usize;
        bytes[position..position + 16].copy_from_slice(SIGNING_BLOCK_MAGIC);
        // Keep the magic well away from the exact 16 bytes preceding the central directory.
        assert_eq!(cd_offset, 64);
        let report = scan_bytes(bytes).unwrap();
        assert!(report.signing_block.is_none());
    }

    #[test]
    fn leading_and_trailing_sizes_must_match() {
        let mut bytes = apk_with_pairs(&[(APK_V2_BLOCK_ID, b"v2")]);
        let cd_offset =
            u32::from_le_bytes(bytes[bytes.len() - 6..bytes.len() - 2].try_into().unwrap())
                as usize;
        bytes[cd_offset - 24..cd_offset - 16].copy_from_slice(&0u64.to_le_bytes());
        assert!(matches!(scan_bytes(bytes), Err(ScanError::Rejected(_))));
    }

    #[test]
    fn pair_length_underflow_overflow_and_trailing_fragment_are_rejected() {
        let mut short_pair = Vec::new();
        put_u64(&mut short_pair, 3);
        put_u32(&mut short_pair, APK_V2_BLOCK_ID);
        let encoded = 24u64 + short_pair.len() as u64;
        let mut block = Vec::new();
        put_u64(&mut block, encoded);
        block.extend_from_slice(&short_pair);
        put_u64(&mut block, encoded);
        block.extend_from_slice(SIGNING_BLOCK_MAGIC);
        assert!(matches!(
            scan_bytes(assemble_with_block(block)),
            Err(ScanError::Rejected(_))
        ));

        let mut overflow_pair = Vec::new();
        put_u64(&mut overflow_pair, u64::MAX);
        let encoded = 24u64 + overflow_pair.len() as u64;
        let mut block = Vec::new();
        put_u64(&mut block, encoded);
        block.extend_from_slice(&overflow_pair);
        put_u64(&mut block, encoded);
        block.extend_from_slice(SIGNING_BLOCK_MAGIC);
        assert!(matches!(
            scan_bytes(assemble_with_block(block)),
            Err(ScanError::Rejected(_))
        ));
    }

    fn assemble_with_block(block: Vec<u8>) -> Vec<u8> {
        let mut out = vec![0x41u8; 64];
        out.extend_from_slice(&block);
        let cd = central_directory_record();
        let cd_offset = out.len() as u32;
        out.extend_from_slice(&cd);
        out.extend_from_slice(&eocd(1, cd.len() as u32, cd_offset, &[]));
        out
    }

    #[test]
    fn ambiguous_eocd_candidate_is_rejected() {
        let fake = eocd(0, 0, 0, &[]);
        let bytes = apk_without_block();
        let mut comment_wrapped = bytes[..bytes.len() - 22].to_vec();
        comment_wrapped.extend_from_slice(&eocd(1, 46, 64, &fake));
        assert!(matches!(
            scan_bytes(comment_wrapped),
            Err(ScanError::Rejected(_))
        ));
    }

    #[test]
    fn zip64_sentinel_is_explicitly_unsupported() {
        let mut bytes = apk_without_block();
        let offset = bytes.len() - 22;
        bytes[offset + 8..offset + 10].copy_from_slice(&u16::MAX.to_le_bytes());
        bytes[offset + 10..offset + 12].copy_from_slice(&u16::MAX.to_le_bytes());
        assert!(matches!(scan_bytes(bytes), Err(ScanError::Unsupported(_))));
    }

    #[test]
    fn pair_and_block_byte_budgets_fail_closed() {
        let bytes = apk_with_pairs(&[(APK_V2_BLOCK_ID, b"large")]);
        let limits = ScanLimits {
            max_pairs: 1,
            max_total_pair_value_bytes: 2,
            ..ScanLimits::default()
        };
        assert!(matches!(
            scan_open_file(&mut Cursor::new(bytes), limits),
            Err(ScanError::Rejected(_))
        ));

        let bytes = apk_with_pairs(&[(APK_V2_BLOCK_ID, b"v2"), (APK_V3_BLOCK_ID, b"v3")]);
        let limits = ScanLimits {
            max_pairs: 1,
            ..ScanLimits::default()
        };
        assert!(matches!(
            scan_open_file(&mut Cursor::new(bytes), limits),
            Err(ScanError::Rejected(_))
        ));
    }

    #[test]
    fn unknown_pair_id_is_retained_without_scheme_inference() {
        let report = scan_bytes(apk_with_pairs(&[(0x1234_5678, b"not a known scheme")])).unwrap();
        let pairs = report.signing_block.unwrap().pairs;
        assert_eq!(pairs[0].id, 0x1234_5678);
        assert_eq!(pairs[0].recognized_scheme, None);
    }
}
