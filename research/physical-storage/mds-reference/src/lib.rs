//! Deterministic physical-storage reference with spatially interleaved MDS(6,4) stripes.
//!
//! This is an algorithmic channel fixture, not a physical optical, magnetic, NAND,
//! or tape model. Erasures are known to the decoder; unflagged substitutions are
//! detected by the final SHA-256 integrity check but are not corrected here.

use std::fmt;

pub const TILE_COUNT: usize = 48;
pub const STRIPE_COUNT: usize = 7;
pub const DATA_SHARDS: usize = 4;
pub const PARITY_SHARDS: usize = 2;
pub const SHARDS_PER_STRIPE: usize = DATA_SHARDS + PARITY_SHARDS;
pub const BLOCK_BYTES: usize = 8;
pub const PAYLOAD_BYTES: usize = STRIPE_COUNT * DATA_SHARDS * BLOCK_BYTES;
pub const EXPECTED_PAYLOAD_SHA256: &str =
    "7e47dde9a2e52a0067f80a149abf606ea4ec25690637632d34561432c0738877";

const HEADER: [u8; BLOCK_BYTES] = *b"PSH0v001";
const CALIBRATION_A: [u8; BLOCK_BYTES] = *b"CAL0v001";
const CALIBRATION_B: [u8; BLOCK_BYTES] = *b"CAL1v001";
const CONTROL_TILES: [usize; 4] = [0, 15, 16, 31];
const SPARE_TILES: [usize; 2] = [32, 47];

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StorageError {
    InvalidPayloadLength { expected: usize, actual: usize },
    TileOutOfRange(usize),
    TileUnavailable(usize),
    TooManyErasures,
    SingularGeneratorMatrix,
}

impl fmt::Display for StorageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPayloadLength { expected, actual } => {
                write!(f, "invalid payload length: expected {expected}, got {actual}")
            }
            Self::TileOutOfRange(tile) => write!(f, "tile index out of range: {tile}"),
            Self::TileUnavailable(tile) => write!(f, "tile is unavailable: {tile}"),
            Self::TooManyErasures => write!(f, "more than two shards erased in a stripe"),
            Self::SingularGeneratorMatrix => write!(f, "singular recovery matrix"),
        }
    }
}

impl std::error::Error for StorageError {}

/// 4 × 4 × 3 media grid. Control cells and two spare cells are excluded from
/// the data/parity pool. A deterministic spatial interleaver distributes each
/// six-shard stripe across the remaining 42 locations.
#[derive(Debug, Clone)]
pub struct PhysicalLayout {
    available_tiles: [usize; 42],
}

impl Default for PhysicalLayout {
    fn default() -> Self {
        Self::new()
    }
}

impl PhysicalLayout {
    pub fn new() -> Self {
        let mut available_tiles = [0usize; 42];
        let mut count = 0usize;

        for tile in 0..TILE_COUNT {
            let is_control = CONTROL_TILES.contains(&tile);
            let is_spare = SPARE_TILES.contains(&tile);
            if !is_control && !is_spare {
                available_tiles[count] = tile;
                count += 1;
            }
        }

        assert_eq!(count, available_tiles.len());
        Self { available_tiles }
    }

    /// Return the physical tile for a logical stripe and shard.
    ///
    /// The available index is stripe + STRIPE_COUNT * shard, spreading each
    /// stripe across the available medium instead of placing shards contiguously.
    pub fn shard_tile(&self, stripe: usize, shard: usize) -> Option<usize> {
        if stripe >= STRIPE_COUNT || shard >= SHARDS_PER_STRIPE {
            return None;
        }
        Some(self.available_tiles[stripe + STRIPE_COUNT * shard])
    }

    pub fn control_tiles(&self) -> [usize; 4] {
        CONTROL_TILES
    }

    pub fn spare_tiles(&self) -> [usize; 2] {
        SPARE_TILES
    }

    pub fn active_tiles(&self) -> &[usize; 42] {
        &self.available_tiles
    }
}

/// A physical image stores an optional 8-byte observation in each tile.
/// None means an independently identified erasure.
#[derive(Debug, Clone)]
pub struct MediaImage {
    cells: [Option<[u8; BLOCK_BYTES]>; TILE_COUNT],
    /// Expected digest retained outside the simulated physical tile array.
    /// This models an independently preserved archive manifest entry.
    expected_payload_sha256: [u8; 32],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecoveryReceipt {
    pub exact_recovery: bool,
    pub digest_matches: bool,
    pub control_metadata_available: bool,
    pub recovered_payload_bytes: usize,
    pub recovered_stripes: usize,
    pub uncorrectable_stripes: Vec<usize>,
    pub payload_sha256: String,
    pub expected_sha256: String,
}

impl RecoveryReceipt {
    pub fn to_json(&self) -> String {
        let stripes = format!(
            "[{}]",
            self.uncorrectable_stripes
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(",")
        );
        format!(
            "{{\"exact_recovery\":{},\"digest_matches\":{},\"control_metadata_available\":{},\"recovered_payload_bytes\":{},\"recovered_stripes\":{},\"uncorrectable_stripes\":{},\"payload_sha256\":\"{}\",\"expected_sha256\":\"{}\"}}",
            self.exact_recovery,
            self.digest_matches,
            self.control_metadata_available,
            self.recovered_payload_bytes,
            self.recovered_stripes,
            stripes,
            self.payload_sha256,
            self.expected_sha256
        )
    }
}

impl MediaImage {
    pub fn write_reference(payload: &[u8], layout: &PhysicalLayout) -> Result<Self, StorageError> {
        if payload.len() != PAYLOAD_BYTES {
            return Err(StorageError::InvalidPayloadLength {
                expected: PAYLOAD_BYTES,
                actual: payload.len(),
            });
        }

        let mut cells = [None; TILE_COUNT];
        cells[0] = Some(HEADER);
        cells[15] = Some(CALIBRATION_A);
        cells[16] = Some(CALIBRATION_B);
        cells[31] = Some(HEADER);

        for stripe in 0..STRIPE_COUNT {
            let mut data = [[0u8; BLOCK_BYTES]; DATA_SHARDS];
            for data_shard in 0..DATA_SHARDS {
                let offset = (stripe * DATA_SHARDS + data_shard) * BLOCK_BYTES;
                data[data_shard].copy_from_slice(&payload[offset..offset + BLOCK_BYTES]);
            }

            let encoded = mds_encode(data);
            for shard in 0..SHARDS_PER_STRIPE {
                let tile = layout.shard_tile(stripe, shard).ok_or(
                    StorageError::TileOutOfRange(shard),
                )?;
                cells[tile] = Some(encoded[shard]);
            }
        }

        Ok(Self {
            cells,
            expected_payload_sha256: sha256(payload),
        })
    }

    /// Erase physical tiles. This models known erasures, not arbitrary bit
    /// errors or a measured physical defect distribution.
    pub fn erase_tiles(&mut self, tiles: &[usize]) -> Result<(), StorageError> {
        for &tile in tiles {
            if tile >= TILE_COUNT {
                return Err(StorageError::TileOutOfRange(tile));
            }
            self.cells[tile] = None;
        }
        Ok(())
    }

    /// Apply an unflagged byte substitution. The current reference decoder
    /// does not correct unknown errors; payload hashing must reject the result.
    pub fn corrupt_byte(
        &mut self,
        tile: usize,
        byte_offset: usize,
        xor_mask: u8,
    ) -> Result<(), StorageError> {
        if tile >= TILE_COUNT {
            return Err(StorageError::TileOutOfRange(tile));
        }
        let block = self.cells[tile]
            .as_mut()
            .ok_or(StorageError::TileUnavailable(tile))?;
        if byte_offset >= BLOCK_BYTES {
            return Err(StorageError::TileOutOfRange(byte_offset));
        }
        block[byte_offset] ^= xor_mask;
        Ok(())
    }

    pub fn read_and_recover(&self, layout: &PhysicalLayout) -> RecoveryReceipt {
        let header_available =
            self.cells[0] == Some(HEADER) || self.cells[31] == Some(HEADER);
        let calibration_available =
            self.cells[15] == Some(CALIBRATION_A) || self.cells[16] == Some(CALIBRATION_B);
        let control_metadata_available = header_available && calibration_available;

        let mut payload = vec![0u8; PAYLOAD_BYTES];
        let mut recovered_payload_bytes = 0usize;
        let mut recovered_stripes = 0usize;
        let mut uncorrectable_stripes = Vec::new();

        for stripe in 0..STRIPE_COUNT {
            let shards: [Option<[u8; BLOCK_BYTES]>; SHARDS_PER_STRIPE] =
                std::array::from_fn(|shard| {
                    let tile = layout.shard_tile(stripe, shard).expect("bounded shard index");
                    self.cells[tile]
                });

            match mds_recover(shards) {
                Ok(recovered) => {
                    for data_shard in 0..DATA_SHARDS {
                        let offset = (stripe * DATA_SHARDS + data_shard) * BLOCK_BYTES;
                        payload[offset..offset + BLOCK_BYTES]
                            .copy_from_slice(&recovered[data_shard]);
                        recovered_payload_bytes += BLOCK_BYTES;
                    }
                    recovered_stripes += 1;
                }
                Err(_) => uncorrectable_stripes.push(stripe),
            }
        }

        let payload_sha256 = hex_digest(&sha256(&payload));
        let expected_sha256 = hex_digest(&self.expected_payload_sha256);
        let digest_matches = payload_sha256 == expected_sha256;
        let exact_recovery = control_metadata_available
            && uncorrectable_stripes.is_empty()
            && digest_matches;

        RecoveryReceipt {
            exact_recovery,
            digest_matches,
            control_metadata_available,
            recovered_payload_bytes,
            recovered_stripes,
            uncorrectable_stripes,
            payload_sha256,
            expected_sha256,
        }
    }
}

/// Systematic [6,4,3] MDS array code over GF(256), using polynomial 0x11d.
/// Each shard is eight bytes. Any four of six shard rows recover the data.
/// This is an erasure-only decoder: unknown substitutions are rejected by
/// the end-to-end digest and are not corrected here.
pub fn mds_encode(data: [[u8; BLOCK_BYTES]; DATA_SHARDS])
    -> [[u8; BLOCK_BYTES]; SHARDS_PER_STRIPE]
{
    let mut encoded = [[0u8; BLOCK_BYTES]; SHARDS_PER_STRIPE];
    encoded[..DATA_SHARDS].copy_from_slice(&data);

    for byte in 0..BLOCK_BYTES {
        let mut parity0 = 0u8;
        let mut parity1 = 0u8;
        for data_shard in 0..DATA_SHARDS {
            parity0 ^= data[data_shard][byte];
            parity1 ^= gf_mul((data_shard + 1) as u8, data[data_shard][byte]);
        }
        encoded[4][byte] = parity0;
        encoded[5][byte] = parity1;
    }
    encoded
}

pub fn mds_recover(
    shards: [Option<[u8; BLOCK_BYTES]>; SHARDS_PER_STRIPE],
) -> Result<[[u8; BLOCK_BYTES]; SHARDS_PER_STRIPE], StorageError> {
    let selected: Vec<usize> = (0..SHARDS_PER_STRIPE)
        .filter(|&index| shards[index].is_some())
        .take(DATA_SHARDS)
        .collect();

    if selected.len() != DATA_SHARDS {
        return Err(StorageError::TooManyErasures);
    }

    let mut augmented = [[0u8; DATA_SHARDS * 2]; DATA_SHARDS];
    for row in 0..DATA_SHARDS {
        let generator = generator_row(selected[row]);
        augmented[row][..DATA_SHARDS].copy_from_slice(&generator);
        augmented[row][DATA_SHARDS + row] = 1;
    }
    invert_matrix(&mut augmented)?;

    let mut data = [[0u8; BLOCK_BYTES]; DATA_SHARDS];
    for data_shard in 0..DATA_SHARDS {
        for byte in 0..BLOCK_BYTES {
            let mut value = 0u8;
            for equation in 0..DATA_SHARDS {
                let observed = shards[selected[equation]].expect("selected shard is present")[byte];
                value ^= gf_mul(augmented[data_shard][DATA_SHARDS + equation], observed);
            }
            data[data_shard][byte] = value;
        }
    }

    Ok(mds_encode(data))
}

fn generator_row(shard: usize) -> [u8; DATA_SHARDS] {
    match shard {
        0..=3 => {
            let mut row = [0u8; DATA_SHARDS];
            row[shard] = 1;
            row
        }
        4 => [1, 1, 1, 1],
        5 => [1, 2, 3, 4],
        _ => unreachable!("shard index is bounded"),
    }
}

fn invert_matrix(
    augmented: &mut [[u8; DATA_SHARDS * 2]; DATA_SHARDS],
) -> Result<(), StorageError> {
    for column in 0..DATA_SHARDS {
        let pivot = (column..DATA_SHARDS)
            .find(|&row| augmented[row][column] != 0)
            .ok_or(StorageError::SingularGeneratorMatrix)?;
        augmented.swap(column, pivot);

        let scale =
            gf_inv(augmented[column][column]).ok_or(StorageError::SingularGeneratorMatrix)?;
        for entry in 0..DATA_SHARDS * 2 {
            augmented[column][entry] = gf_mul(augmented[column][entry], scale);
        }

        for row in 0..DATA_SHARDS {
            if row == column {
                continue;
            }
            let factor = augmented[row][column];
            if factor == 0 {
                continue;
            }
            for entry in 0..DATA_SHARDS * 2 {
                augmented[row][entry] ^= gf_mul(factor, augmented[column][entry]);
            }
        }
    }
    Ok(())
}

fn gf_mul(mut a: u8, mut b: u8) -> u8 {
    let mut product = 0u8;
    for _ in 0..8 {
        if b & 1 != 0 {
            product ^= a;
        }
        let carry = a & 0x80;
        a <<= 1;
        if carry != 0 {
            // 0x11d reduced by x^8 gives 0x1d.
            a ^= 0x1d;
        }
        b >>= 1;
    }
    product
}

fn gf_pow(mut value: u8, mut exponent: u16) -> u8 {
    let mut result = 1u8;
    while exponent > 0 {
        if exponent & 1 != 0 {
            result = gf_mul(result, value);
        }
        value = gf_mul(value, value);
        exponent >>= 1;
    }
    result
}

fn gf_inv(value: u8) -> Option<u8> {
    if value == 0 {
        None
    } else {
        Some(gf_pow(value, 254))
    }
}

/// Self-contained SHA-256: no third-party dependency. This is an integrity
/// digest, not a signature or proof of source authenticity.
pub fn sha256(input: &[u8]) -> [u8; 32] {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
        0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
        0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
        0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
        0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
        0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
        0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
        0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    ];

    let mut state = [
        0x6a09e667u32, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    ];
    let bit_length = (input.len() as u64).wrapping_mul(8);
    let mut padded = input.to_vec();
    padded.push(0x80);
    while padded.len() % 64 != 56 {
        padded.push(0);
    }
    padded.extend_from_slice(&bit_length.to_be_bytes());

    for chunk in padded.chunks_exact(64) {
        let mut words = [0u32; 64];
        for index in 0..16 {
            let offset = index * 4;
            words[index] = u32::from_be_bytes([
                chunk[offset], chunk[offset + 1], chunk[offset + 2], chunk[offset + 3],
            ]);
        }
        for index in 16..64 {
            let s0 = words[index - 15].rotate_right(7)
                ^ words[index - 15].rotate_right(18)
                ^ (words[index - 15] >> 3);
            let s1 = words[index - 2].rotate_right(17)
                ^ words[index - 2].rotate_right(19)
                ^ (words[index - 2] >> 10);
            words[index] = words[index - 16]
                .wrapping_add(s0)
                .wrapping_add(words[index - 7])
                .wrapping_add(s1);
        }

        let mut work = state;
        for index in 0..64 {
            let sum1 = work[4].rotate_right(6)
                ^ work[4].rotate_right(11)
                ^ work[4].rotate_right(25);
            let choose = (work[4] & work[5]) ^ ((!work[4]) & work[6]);
            let temp1 = work[7]
                .wrapping_add(sum1)
                .wrapping_add(choose)
                .wrapping_add(K[index])
                .wrapping_add(words[index]);
            let sum0 = work[0].rotate_right(2)
                ^ work[0].rotate_right(13)
                ^ work[0].rotate_right(22);
            let majority = (work[0] & work[1])
                ^ (work[0] & work[2])
                ^ (work[1] & work[2]);
            let temp2 = sum0.wrapping_add(majority);

            work = [
                temp1.wrapping_add(temp2),
                work[0],
                work[1],
                work[2],
                work[3].wrapping_add(temp1),
                work[4],
                work[5],
                work[6],
            ];
        }

        for index in 0..8 {
            state[index] = state[index].wrapping_add(work[index]);
        }
    }

    let mut digest = [0u8; 32];
    for (index, word) in state.iter().enumerate() {
        digest[index * 4..index * 4 + 4].copy_from_slice(&word.to_be_bytes());
    }
    digest
}

pub fn hex_digest(digest: &[u8; 32]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(64);
    for byte in digest {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}

pub fn reference_payload() -> Vec<u8> {
    (0u8..=223u8).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_known_answers() {
        assert_eq!(
            hex_digest(&sha256(b"abc")),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            hex_digest(&sha256(&reference_payload())),
            EXPECTED_PAYLOAD_SHA256
        );
    }

    #[test]
    fn layout_partitions_all_tiles_without_overlap() {
        let layout = PhysicalLayout::new();
        let mut seen = [false; TILE_COUNT];
        for tile in layout.active_tiles() {
            assert!(!seen[*tile]);
            seen[*tile] = true;
        }
        for tile in CONTROL_TILES {
            assert!(!seen[tile]);
            seen[tile] = true;
        }
        for tile in SPARE_TILES {
            assert!(!seen[tile]);
            seen[tile] = true;
        }
        assert!(seen.into_iter().all(|present| present));
        assert_eq!(layout.active_tiles().len(), 42);
    }

    #[test]
    fn every_two_shard_erasure_pattern_recovers_one_stripe() {
        let data = [
            [0x10; BLOCK_BYTES],
            [0x21; BLOCK_BYTES],
            [0x32; BLOCK_BYTES],
            [0x43; BLOCK_BYTES],
        ];
        let encoded = mds_encode(data);

        for first in 0..SHARDS_PER_STRIPE {
            for second in first + 1..SHARDS_PER_STRIPE {
                let mut damaged = encoded.map(Some);
                damaged[first] = None;
                damaged[second] = None;
                assert_eq!(mds_recover(damaged).unwrap(), encoded);
            }
        }
    }

    #[test]
    fn one_complete_physical_layer_is_recoverable() {
        let layout = PhysicalLayout::new();
        let payload = reference_payload();

        // Each layer contains exactly two shards from every stripe. Two parity
        // shards therefore recover a complete one-layer dropout.
        for z in 0..3 {
            let mut image = MediaImage::write_reference(&payload, &layout).unwrap();
            let layer_tiles: Vec<usize> = (z * 16..(z + 1) * 16).collect();
            image.erase_tiles(&layer_tiles).unwrap();
            let receipt = image.read_and_recover(&layout);
            assert!(receipt.exact_recovery, "layer {z}: {}", receipt.to_json());
            assert_eq!(receipt.recovered_payload_bytes, PAYLOAD_BYTES);
        }
    }

    #[test]
    fn localized_two_by_two_cluster_is_recoverable() {
        let layout = PhysicalLayout::new();
        let payload = reference_payload();
        let mut image = MediaImage::write_reference(&payload, &layout).unwrap();
        image.erase_tiles(&[5, 6, 9, 10]).unwrap();
        let receipt = image.read_and_recover(&layout);
        assert!(receipt.exact_recovery, "{}", receipt.to_json());
    }

    #[test]
    fn two_erasures_in_each_stripe_are_recoverable() {
        let layout = PhysicalLayout::new();
        let payload = reference_payload();
        let mut image = MediaImage::write_reference(&payload, &layout).unwrap();
        let erasures: Vec<usize> = (0..STRIPE_COUNT)
            .flat_map(|stripe| {
                [
                    layout.shard_tile(stripe, 0).unwrap(),
                    layout.shard_tile(stripe, 1).unwrap(),
                ]
            })
            .collect();
        image.erase_tiles(&erasures).unwrap();
        let receipt = image.read_and_recover(&layout);
        assert!(receipt.exact_recovery, "{}", receipt.to_json());
    }

    #[test]
    fn more_than_two_erasures_in_a_stripe_fails_closed() {
        let layout = PhysicalLayout::new();
        let payload = reference_payload();
        let mut image = MediaImage::write_reference(&payload, &layout).unwrap();
        let erasures = [
            layout.shard_tile(0, 0).unwrap(),
            layout.shard_tile(0, 1).unwrap(),
            layout.shard_tile(0, 2).unwrap(),
        ];
        image.erase_tiles(&erasures).unwrap();
        let receipt = image.read_and_recover(&layout);
        assert!(!receipt.exact_recovery);
        assert!(!receipt.uncorrectable_stripes.is_empty());
    }

    #[test]
    fn unflagged_byte_substitution_is_detected_not_silently_accepted() {
        let layout = PhysicalLayout::new();
        let payload = reference_payload();
        let mut image = MediaImage::write_reference(&payload, &layout).unwrap();
        let tile = layout.shard_tile(0, 0).unwrap();
        image.corrupt_byte(tile, 3, 0x80).unwrap();
        let receipt = image.read_and_recover(&layout);
        assert!(!receipt.digest_matches);
        assert!(!receipt.exact_recovery);
    }

    #[test]
    fn one_copy_of_each_control_type_is_sufficient() {
        let layout = PhysicalLayout::new();
        let payload = reference_payload();
        let mut image = MediaImage::write_reference(&payload, &layout).unwrap();
        image.erase_tiles(&[0, 16]).unwrap();
        let receipt = image.read_and_recover(&layout);
        assert!(receipt.control_metadata_available);
        assert!(receipt.exact_recovery);
    }

    #[test]
    fn loss_of_both_headers_or_both_calibration_references_fails_closed() {
        let layout = PhysicalLayout::new();
        let payload = reference_payload();

        let mut no_headers = MediaImage::write_reference(&payload, &layout).unwrap();
        no_headers.erase_tiles(&[0, 31]).unwrap();
        assert!(!no_headers.read_and_recover(&layout).exact_recovery);

        let mut no_calibration = MediaImage::write_reference(&payload, &layout).unwrap();
        no_calibration.erase_tiles(&[15, 16]).unwrap();
        assert!(!no_calibration.read_and_recover(&layout).exact_recovery);
    }

    fn tile_for(layout: &PhysicalLayout, stripe: usize, shard: usize, row_major: bool) -> usize {
        if row_major {
            layout.active_tiles()[stripe * SHARDS_PER_STRIPE + shard]
        } else {
            layout.shard_tile(stripe, shard).unwrap()
        }
    }

    fn worst_erasures(
        layout: &PhysicalLayout,
        mask: &[bool; TILE_COUNT],
        row_major: bool,
    ) -> usize {
        (0..STRIPE_COUNT)
            .map(|stripe| {
                (0..SHARDS_PER_STRIPE)
                    .filter(|&shard| mask[tile_for(layout, stripe, shard, row_major)])
                    .count()
            })
            .max()
            .unwrap_or(0)
    }

    fn rectangular_fault_tournament(
        layout: &PhysicalLayout,
        dimensions: (usize, usize, usize),
        row_major: bool,
    ) -> (usize, usize, usize) {
        let (dx, dy, dz) = dimensions;
        let mut worst = 0usize;
        let mut failures = 0usize;
        let mut scenarios = 0usize;

        for z in 0..=3 - dz {
            for y in 0..=4 - dy {
                for x in 0..=4 - dx {
                    let mut mask = [false; TILE_COUNT];
                    for zz in z..z + dz {
                        for yy in y..y + dy {
                            for xx in x..x + dx {
                                mask[zz * 16 + yy * 4 + xx] = true;
                            }
                        }
                    }
                    let score = worst_erasures(layout, &mask, row_major);
                    worst = worst.max(score);
                    if score > PARITY_SHARDS {
                        failures += 1;
                    }
                    scenarios += 1;
                }
            }
        }

        (worst, failures, scenarios)
    }

    #[test]
    fn spatial_interleaving_outperforms_row_major_on_local_fault_masks() {
        let layout = PhysicalLayout::new();

        // Exhaustively score all axis-aligned 2x2x1 boxes in this 4x4x3
        // fixture. The spatial-stride map keeps every box within one erasure
        // per stripe; row-major placement exceeds the two-erasure budget in
        // 13 of 27 positions.
        assert_eq!(
            rectangular_fault_tournament(&layout, (2, 2, 1), false),
            (1, 0, 27)
        );
        assert_eq!(
            rectangular_fault_tournament(&layout, (2, 2, 1), true),
            (4, 13, 27)
        );

        // 3x3x1 boxes remain recoverable spatially; row-major fails every
        // placement under the same two-erasure budget.
        assert_eq!(
            rectangular_fault_tournament(&layout, (3, 3, 1), false),
            (2, 0, 12)
        );
        assert_eq!(
            rectangular_fault_tournament(&layout, (3, 3, 1), true),
            (5, 12, 12)
        );

        // The full x-y layer slice is also recoverable spatially; row-major
        // exceeds the erasure budget in all three z positions.
        assert_eq!(
            rectangular_fault_tournament(&layout, (4, 4, 1), false),
            (2, 0, 3)
        );
        assert_eq!(
            rectangular_fault_tournament(&layout, (4, 4, 1), true),
            (6, 3, 3)
        );

        // 2x2x2 boxes remain recoverable under the spatial mapping; row-major
        // fails 14 of 18 positions.
        assert_eq!(
            rectangular_fault_tournament(&layout, (2, 2, 2), false),
            (2, 0, 18)
        );
        assert_eq!(
            rectangular_fault_tournament(&layout, (2, 2, 2), true),
            (4, 14, 18)
        );

        // The synthetic design envelope has a visible boundary: 3x3x2 boxes
        // exceed two erasures in all eight positions.
        assert_eq!(
            rectangular_fault_tournament(&layout, (3, 3, 2), false),
            (4, 8, 8)
        );
        assert_eq!(
            rectangular_fault_tournament(&layout, (3, 3, 2), true),
            (5, 8, 8)
        );
    }

    #[test]
    fn row_major_mapping_cannot_survive_any_single_full_layer_dropout() {
        let layout = PhysicalLayout::new();
        for z in 0..3 {
            let mut mask = [false; TILE_COUNT];
            for tile in z * 16..(z + 1) * 16 {
                mask[tile] = true;
            }
            assert_eq!(worst_erasures(&layout, &mask, false), 2);
            assert_eq!(worst_erasures(&layout, &mask, true), 6);
        }
    }

    #[test]
    fn arbitrary_payload_round_trips_against_its_out_of_band_manifest_digest() {
        let layout = PhysicalLayout::new();
        let payload: Vec<u8> = (0..PAYLOAD_BYTES)
            .map(|index| (index as u8).wrapping_mul(37).wrapping_add(11))
            .collect();
        let mut image = MediaImage::write_reference(&payload, &layout).unwrap();

        let erasures: Vec<usize> = (0..STRIPE_COUNT)
            .map(|stripe| layout.shard_tile(stripe, 4).unwrap())
            .collect();
        image.erase_tiles(&erasures).unwrap();

        let receipt = image.read_and_recover(&layout);
        assert!(receipt.exact_recovery, "{}", receipt.to_json());
        assert_eq!(receipt.payload_sha256, hex_digest(&sha256(&payload)));
    }

    #[test]
    fn wrong_payload_length_is_rejected() {
        let layout = PhysicalLayout::new();
        assert!(matches!(
            MediaImage::write_reference(&[0u8; 223], &layout),
            Err(StorageError::InvalidPayloadLength { .. })
        ));
    }
}
