//! Synthetic [6,4,3] bootstrap metadata codec reference.
//!
//! Dependency-free reference for GF(2^8), polynomial 0x11d, systematic data
//! rows, and parity rows [1,1,1,1] and [1,2,3,4]. Not a production media format.
//!
//! Trust boundary: this module reconstructs bytes and carries expected SHA-256
//! values, but does not compute hashes, authenticate provenance, validate
//! calibration, or establish physical durability. Verify externally before use.

const GENERATOR: [[u8; 4]; 6] = [
    [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
    [1, 1, 1, 1], [1, 2, 3, 4],
];
const MANIFEST_MAGIC: &[u8; 4] = b"BSM1";
const BUNDLE_MAGIC: &[u8; 4] = b"BSP1";
const FORMAT_VERSION: u16 = 1;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BootstrapCodecError {
    UnequalShardLengths,
    NotEnoughShards { available: usize },
    SingularSurvivorSet,
    InconsistentSurvivor { shard_index: usize },
    InvalidMagic,
    UnsupportedVersion,
    InvalidUtf8,
    Truncated,
    NonCanonicalRecordOrder,
    EmptyObjectId,
    EmptyVersion,
    LengthOverflow,
    InvalidPadding,
    TrailingBytes,
    InvalidBundleLength,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BootstrapObject {
    pub object_id: String,
    /// Registry-defined type code; unknown values remain opaque to this codec.
    pub semantic_type: u8,
    pub version: String,
    /// Expected SHA-256 of content, supplied by an independent manifest.
    /// This codec carries but DOES NOT verify this value.
    pub expected_content_sha256: [u8; 32],
    pub content: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BootstrapRecovery {
    pub data_shards: [Vec<u8>; 4],
    pub all_shards: [Vec<u8>; 6],
    /// Sorted indices of shards reconstructed from erasures.
    pub recovered_indices: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BootstrapBundle {
    /// Shard order: data 0..3, then parity 4..5.
    pub shards: [Vec<u8>; 6],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedBootstrapBundle {
    pub objects: Vec<BootstrapObject>,
    pub recovered_shard_indices: Vec<usize>,
}

fn gf_mul(mut a: u8, mut b: u8) -> u8 {
    let mut product = 0u8;
    for _ in 0..8 {
        if b & 1 != 0 {
            product ^= a;
        }
        let high_bit = a & 0x80;
        a <<= 1;
        if high_bit != 0 {
            // 0x11d reduced by x^8 is 0x1d.
            a ^= 0x1d;
        }
        b >>= 1;
    }
    product
}

fn gf_pow(mut base: u8, mut exponent: u16) -> u8 {
    let mut result = 1u8;
    while exponent != 0 {
        if exponent & 1 != 0 {
            result = gf_mul(result, base);
        }
        base = gf_mul(base, base);
        exponent >>= 1;
    }
    result
}

fn gf_inv(value: u8) -> Option<u8> {
    if value == 0 { None } else { Some(gf_pow(value, 254)) }
}

fn invert_4x4(matrix: [[u8; 4]; 4]) -> Result<[[u8; 4]; 4], BootstrapCodecError> {
    let mut augmented = [[0u8; 8]; 4];
    for row in 0..4 {
        for column in 0..4 {
            augmented[row][column] = matrix[row][column];
        }
        augmented[row][4 + row] = 1;
    }

    for column in 0..4 {
        let pivot = (column..4)
            .find(|&row| augmented[row][column] != 0)
            .ok_or(BootstrapCodecError::SingularSurvivorSet)?;
        if pivot != column {
            augmented.swap(pivot, column);
        }
        let inverse = gf_inv(augmented[column][column])
            .ok_or(BootstrapCodecError::SingularSurvivorSet)?;
        for item in 0..8 {
            augmented[column][item] = gf_mul(augmented[column][item], inverse);
        }
        let normalized_pivot = augmented[column];
        for row in 0..4 {
            if row == column {
                continue;
            }
            let factor = augmented[row][column];
            if factor == 0 {
                continue;
            }
            for item in 0..8 {
                augmented[row][item] ^= gf_mul(factor, normalized_pivot[item]);
            }
        }
    }

    let mut inverse = [[0u8; 4]; 4];
    for row in 0..4 {
        inverse[row].copy_from_slice(&augmented[row][4..8]);
    }
    Ok(inverse)
}

/// Encode four equally-sized data streams into the fixture's six-shard code.
pub fn encode_bootstrap_shards(data: [&[u8]; 4]) -> Result<[Vec<u8>; 6], BootstrapCodecError> {
    let shard_len = data[0].len();
    if data.iter().any(|shard| shard.len() != shard_len) {
        return Err(BootstrapCodecError::UnequalShardLengths);
    }
    let mut shards: [Vec<u8>; 6] = std::array::from_fn(|index| {
        if index < 4 { data[index].to_vec() } else { vec![0; shard_len] }
    });
    for parity_index in 4..6 {
        for data_index in 0..4 {
            let coefficient = GENERATOR[parity_index][data_index];
            for offset in 0..shard_len {
                shards[parity_index][offset] ^=
                    gf_mul(coefficient, data[data_index][offset]);
            }
        }
    }
    Ok(shards)
}

/// Recover four data shards from any four survivors (known erasures only).
///
/// With exactly four survivors, an undetected error in one cannot in general
/// be distinguished from another valid codeword. Extra survivors are checked,
/// but external digests remain mandatory for acceptance.
pub fn recover_bootstrap_shards(
    shards: [Option<Vec<u8>>; 6],
) -> Result<BootstrapRecovery, BootstrapCodecError> {
    let available_indices: Vec<usize> = shards.iter().enumerate()
        .filter_map(|(index, shard)| shard.as_ref().map(|_| index))
        .collect();
    if available_indices.len() < 4 {
        return Err(BootstrapCodecError::NotEnoughShards {
            available: available_indices.len(),
        });
    }
    let shard_len = shards[available_indices[0]].as_ref().unwrap().len();
    for index in &available_indices {
        if shards[*index].as_ref().unwrap().len() != shard_len {
            return Err(BootstrapCodecError::UnequalShardLengths);
        }
    }

    let selected = [
        available_indices[0], available_indices[1],
        available_indices[2], available_indices[3],
    ];
    let selected_matrix: [[u8; 4]; 4] =
        std::array::from_fn(|row| GENERATOR[selected[row]]);
    let inverse = invert_4x4(selected_matrix)?;

    let mut data_shards: [Vec<u8>; 4] =
        std::array::from_fn(|_| vec![0u8; shard_len]);
    for data_index in 0..4 {
        for equation in 0..4 {
            let coefficient = inverse[data_index][equation];
            let observed = shards[selected[equation]].as_ref().unwrap();
            for offset in 0..shard_len {
                data_shards[data_index][offset] ^=
                    gf_mul(coefficient, observed[offset]);
            }
        }
    }

    let data_refs: [&[u8]; 4] = [
        &data_shards[0], &data_shards[1], &data_shards[2], &data_shards[3],
    ];
    let all_shards = encode_bootstrap_shards(data_refs)?;
    for index in &available_indices {
        if shards[*index].as_ref().unwrap() != &all_shards[*index] {
            return Err(BootstrapCodecError::InconsistentSurvivor {
                shard_index: *index,
            });
        }
    }
    let recovered_indices = (0..6).filter(|index| shards[*index].is_none()).collect();
    Ok(BootstrapRecovery { data_shards, all_shards, recovered_indices })
}

/// Canonical wire format (big-endian):
/// BSM1 | u16 format_version | u32 record_count | records...
/// record = u16 id_len | u8 type | u16 version_len | u64 content_len |
///          32-byte expected SHA-256 | UTF-8 id | UTF-8 version | content
///
/// Expected content digests are carried, not calculated or verified.
pub fn encode_bootstrap_manifest(
    objects: &[BootstrapObject],
) -> Result<Vec<u8>, BootstrapCodecError> {
    let count = u32::try_from(objects.len()).map_err(|_| BootstrapCodecError::LengthOverflow)?;
    let mut ordered: Vec<&BootstrapObject> = objects.iter().collect();
    ordered.sort_by(|left, right| left.object_id.as_bytes().cmp(right.object_id.as_bytes()));

    let mut output = Vec::new();
    output.extend_from_slice(MANIFEST_MAGIC);
    output.extend_from_slice(&FORMAT_VERSION.to_be_bytes());
    output.extend_from_slice(&count.to_be_bytes());

    let mut previous_id: Option<&[u8]> = None;
    for object in ordered {
        let id = object.object_id.as_bytes();
        let version = object.version.as_bytes();
        if id.is_empty() || id.iter().all(u8::is_ascii_whitespace) {
            return Err(BootstrapCodecError::EmptyObjectId);
        }
        if version.is_empty() {
            return Err(BootstrapCodecError::EmptyVersion);
        }
        if previous_id.is_some_and(|previous| previous >= id) {
            return Err(BootstrapCodecError::NonCanonicalRecordOrder);
        }
        previous_id = Some(id);

        let id_len = u16::try_from(id.len()).map_err(|_| BootstrapCodecError::LengthOverflow)?;
        let version_len =
            u16::try_from(version.len()).map_err(|_| BootstrapCodecError::LengthOverflow)?;
        let content_len =
            u64::try_from(object.content.len()).map_err(|_| BootstrapCodecError::LengthOverflow)?;
        output.extend_from_slice(&id_len.to_be_bytes());
        output.push(object.semantic_type);
        output.extend_from_slice(&version_len.to_be_bytes());
        output.extend_from_slice(&content_len.to_be_bytes());
        output.extend_from_slice(&object.expected_content_sha256);
        output.extend_from_slice(id);
        output.extend_from_slice(version);
        output.extend_from_slice(&object.content);
    }
    Ok(output)
}

fn take<'a>(input: &'a [u8], cursor: &mut usize, length: usize)
    -> Result<&'a [u8], BootstrapCodecError>
{
    let end = cursor.checked_add(length).ok_or(BootstrapCodecError::LengthOverflow)?;
    let value = input.get(*cursor..end).ok_or(BootstrapCodecError::Truncated)?;
    *cursor = end;
    Ok(value)
}
fn take_u16(input: &[u8], cursor: &mut usize) -> Result<u16, BootstrapCodecError> {
    let bytes: [u8; 2] = take(input, cursor, 2)?.try_into()
        .map_err(|_| BootstrapCodecError::Truncated)?;
    Ok(u16::from_be_bytes(bytes))
}
fn take_u32(input: &[u8], cursor: &mut usize) -> Result<u32, BootstrapCodecError> {
    let bytes: [u8; 4] = take(input, cursor, 4)?.try_into()
        .map_err(|_| BootstrapCodecError::Truncated)?;
    Ok(u32::from_be_bytes(bytes))
}
fn take_u64(input: &[u8], cursor: &mut usize) -> Result<u64, BootstrapCodecError> {
    let bytes: [u8; 8] = take(input, cursor, 8)?.try_into()
        .map_err(|_| BootstrapCodecError::Truncated)?;
    Ok(u64::from_be_bytes(bytes))
}

/// Decode record framing. Content digests remain untrusted claims until checked.
pub fn decode_bootstrap_manifest(
    input: &[u8],
) -> Result<Vec<BootstrapObject>, BootstrapCodecError> {
    let mut cursor = 0usize;
    if take(input, &mut cursor, 4)? != MANIFEST_MAGIC {
        return Err(BootstrapCodecError::InvalidMagic);
    }
    if take_u16(input, &mut cursor)? != FORMAT_VERSION {
        return Err(BootstrapCodecError::UnsupportedVersion);
    }
    let count = usize::try_from(take_u32(input, &mut cursor)?)
        .map_err(|_| BootstrapCodecError::LengthOverflow)?;
    let mut objects = Vec::with_capacity(count);
    let mut previous_id: Option<Vec<u8>> = None;

    for _ in 0..count {
        let id_len = usize::from(take_u16(input, &mut cursor)?);
        let semantic_type = take(input, &mut cursor, 1)?[0];
        let version_len = usize::from(take_u16(input, &mut cursor)?);
        let content_len = usize::try_from(take_u64(input, &mut cursor)?)
            .map_err(|_| BootstrapCodecError::LengthOverflow)?;
        let expected_content_sha256: [u8; 32] = take(input, &mut cursor, 32)?
            .try_into().map_err(|_| BootstrapCodecError::Truncated)?;
        let id_bytes = take(input, &mut cursor, id_len)?.to_vec();
        let version_bytes = take(input, &mut cursor, version_len)?.to_vec();
        let content = take(input, &mut cursor, content_len)?.to_vec();

        if id_bytes.is_empty() || id_bytes.iter().all(u8::is_ascii_whitespace) {
            return Err(BootstrapCodecError::EmptyObjectId);
        }
        if version_bytes.is_empty() {
            return Err(BootstrapCodecError::EmptyVersion);
        }
        if previous_id.as_ref().is_some_and(
            |previous| previous.as_slice() >= id_bytes.as_slice()
        ) {
            return Err(BootstrapCodecError::NonCanonicalRecordOrder);
        }
        let object_id = String::from_utf8(id_bytes.clone())
            .map_err(|_| BootstrapCodecError::InvalidUtf8)?;
        let version = String::from_utf8(version_bytes)
            .map_err(|_| BootstrapCodecError::InvalidUtf8)?;
        previous_id = Some(id_bytes);
        objects.push(BootstrapObject {
            object_id, semantic_type, version, expected_content_sha256, content,
        });
    }
    if cursor != input.len() {
        return Err(BootstrapCodecError::TrailingBytes);
    }
    Ok(objects)
}

/// Pack a manifest into four equal data shards, add two parity shards, and
/// embed the exact manifest length before zero padding to a four-way boundary.
pub fn encode_bootstrap_bundle(
    objects: &[BootstrapObject],
) -> Result<BootstrapBundle, BootstrapCodecError> {
    let manifest = encode_bootstrap_manifest(objects)?;
    let manifest_len =
        u64::try_from(manifest.len()).map_err(|_| BootstrapCodecError::LengthOverflow)?;
    let mut framed = Vec::with_capacity(12 + manifest.len() + 3);
    framed.extend_from_slice(BUNDLE_MAGIC);
    framed.extend_from_slice(&manifest_len.to_be_bytes());
    framed.extend_from_slice(&manifest);
    while framed.len() % 4 != 0 {
        framed.push(0);
    }
    let shard_len = framed.len() / 4;
    let data = [
        &framed[0..shard_len],
        &framed[shard_len..2 * shard_len],
        &framed[2 * shard_len..3 * shard_len],
        &framed[3 * shard_len..4 * shard_len],
    ];
    Ok(BootstrapBundle { shards: encode_bootstrap_shards(data)? })
}

/// Decode a bundle candidate. The caller must independently verify object
/// digests, authenticity, calibration scope, and provenance before acceptance.
pub fn decode_bootstrap_bundle(
    shards: [Option<Vec<u8>>; 6],
) -> Result<DecodedBootstrapBundle, BootstrapCodecError> {
    let recovery = recover_bootstrap_shards(shards)?;
    let total_len: usize = recovery.data_shards.iter().map(Vec::len).sum();
    if total_len % 4 != 0 {
        return Err(BootstrapCodecError::InvalidBundleLength);
    }
    let mut framed = Vec::with_capacity(total_len);
    for shard in &recovery.data_shards {
        framed.extend_from_slice(shard);
    }
    let mut cursor = 0usize;
    if take(&framed, &mut cursor, 4)? != BUNDLE_MAGIC {
        return Err(BootstrapCodecError::InvalidMagic);
    }
    let manifest_len = usize::try_from(take_u64(&framed, &mut cursor)?)
        .map_err(|_| BootstrapCodecError::LengthOverflow)?;
    let manifest = take(&framed, &mut cursor, manifest_len)?;
    if framed[cursor..].iter().any(|byte| *byte != 0) {
        return Err(BootstrapCodecError::InvalidPadding);
    }
    let objects = decode_bootstrap_manifest(manifest)?;
    Ok(DecodedBootstrapBundle {
        objects,
        recovered_shard_indices: recovery.recovered_indices,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn from_hex(value: &str) -> Vec<u8> {
        assert_eq!(value.len() % 2, 0);
        (0..value.len()).step_by(2)
            .map(|index| u8::from_str_radix(&value[index..index + 2], 16).unwrap())
            .collect()
    }
    fn sample_objects() -> Vec<BootstrapObject> {
        vec![BootstrapObject {
            object_id: "FMT-V1".into(),
            semantic_type: 1,
            version: "physical-format-v1".into(),
            expected_content_sha256: from_hex(
                "8a2cbf9b800e7e46748d574a556a3d1aba5b4c3ec959a75255aac5183fbfba69",
            ).try_into().unwrap(),
            content: b"PHYSFMT1".to_vec(),
        }]
    }
    fn full_slots(shards: &[Vec<u8>; 6]) -> [Option<Vec<u8>>; 6] {
        std::array::from_fn(|index| Some(shards[index].clone()))
    }

    #[test]
    fn published_4_plus_2_parity_known_answer() {
        let encoded = encode_bootstrap_shards([
            b"PHYSFMT1".as_slice(), b"CODEV001".as_slice(),
            b"LAYOUT01".as_slice(), b"CALIB001".as_slice(),
        ]).unwrap();
        assert_eq!(encoded[4], from_hex("1c07081007196400"));
        assert_eq!(encoded[5], from_hex("130c17310011a4c4"));
    }

    #[test]
    fn all_fifteen_two_erasure_patterns_recover_exact_data() {
        let encoded = encode_bootstrap_bundle(&sample_objects()).unwrap();
        for first in 0..6 {
            for second in (first + 1)..6 {
                let mut slots = full_slots(&encoded.shards);
                slots[first] = None;
                slots[second] = None;
                let decoded = decode_bootstrap_bundle(slots).unwrap();
                assert_eq!(decoded.objects, sample_objects());
                assert_eq!(decoded.recovered_shard_indices, vec![first, second]);
            }
        }
    }

    #[test]
    fn three_erasures_are_not_misreported_as_recoverable() {
        let encoded = encode_bootstrap_bundle(&sample_objects()).unwrap();
        let mut slots = full_slots(&encoded.shards);
        slots[0] = None; slots[1] = None; slots[2] = None;
        assert_eq!(
            decode_bootstrap_bundle(slots),
            Err(BootstrapCodecError::NotEnoughShards { available: 3 })
        );
    }

    #[test]
    fn extra_survivors_are_cross_checked_for_inconsistency() {
        let encoded = encode_bootstrap_bundle(&sample_objects()).unwrap();
        let mut slots = full_slots(&encoded.shards);
        slots[0].as_mut().unwrap()[0] ^= 0x01;
        slots[5] = None;
        assert_eq!(
            decode_bootstrap_bundle(slots),
            Err(BootstrapCodecError::InconsistentSurvivor { shard_index: 4 })
        );
    }

    #[test]
    fn nonzero_canonical_padding_is_rejected() {
        let encoded = encode_bootstrap_bundle(&sample_objects()).unwrap();
        let mut slots = full_slots(&encoded.shards);
        slots[3].as_mut().unwrap()[24] ^= 0x01;
        slots[4] = None; slots[5] = None;
        assert_eq!(
            decode_bootstrap_bundle(slots),
            Err(BootstrapCodecError::InvalidPadding)
        );
    }

    #[test]
    fn object_order_does_not_change_canonical_bytes() {
        let mut objects = sample_objects();
        objects.push(BootstrapObject {
            object_id: "CODE-V1".into(),
            semantic_type: 2,
            version: "code-v1".into(),
            expected_content_sha256: [0x22; 32],
            content: b"code".to_vec(),
        });
        let mut reversed = objects.clone();
        reversed.reverse();
        assert_eq!(
            encode_bootstrap_manifest(&objects).unwrap(),
            encode_bootstrap_manifest(&reversed).unwrap()
        );
    }

    #[test]
    fn decoder_rejects_trailing_bytes_and_truncated_records() {
        let mut encoded = encode_bootstrap_manifest(&sample_objects()).unwrap();
        encoded.push(0);
        assert_eq!(decode_bootstrap_manifest(&encoded), Err(BootstrapCodecError::TrailingBytes));
        encoded.pop();
        encoded[6..10].copy_from_slice(&2u32.to_be_bytes());
        assert_eq!(decode_bootstrap_manifest(&encoded), Err(BootstrapCodecError::Truncated));
    }
}
