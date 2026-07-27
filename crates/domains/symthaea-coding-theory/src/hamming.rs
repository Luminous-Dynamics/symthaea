// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Hamming distance/weight, the single-error-correcting **Hamming(7,4)** code,
//! and the extended **Hamming(8,4) SECDED** code.

use std::fmt;

/// The Hamming weight (number of 1-bits) of a byte slice.
#[must_use]
pub fn weight(a: &[u8]) -> usize {
    a.iter().map(|byte| byte.count_ones() as usize).sum()
}

/// The Hamming distance between equal-length byte slices.
///
/// Returns `None` on a length mismatch.
#[must_use]
pub fn distance(a: &[u8], b: &[u8]) -> Option<usize> {
    if a.len() != b.len() {
        return None;
    }
    Some(
        a.iter()
            .zip(b)
            .map(|(left, right)| (left ^ right).count_ones() as usize)
            .sum(),
    )
}

/// A symbol outside the binary alphabet `{0, 1}`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InvalidBit {
    /// Index of the invalid symbol.
    pub index: usize,
    /// Supplied byte value.
    pub value: u8,
}

impl fmt::Display for InvalidBit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "expected bit 0 or 1 at index {}, got {}",
            self.index, self.value
        )
    }
}

impl std::error::Error for InvalidBit {}

/// Invalid packed Hamming input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackedHammingError {
    /// A source nibble used bits above bit three.
    NibbleOutOfRange { value: u8 },
    /// A packed Hamming(7,4) word used the reserved eighth bit.
    Hamming74WordOutOfRange { value: u8 },
}

impl fmt::Display for PackedHammingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NibbleOutOfRange { value } => {
                write!(
                    f,
                    "packed Hamming payload must fit in four bits, got 0x{value:02x}"
                )
            }
            Self::Hamming74WordOutOfRange { value } => write!(
                f,
                "packed Hamming(7,4) word must fit in seven bits, got 0x{value:02x}"
            ),
        }
    }
}

impl std::error::Error for PackedHammingError {}

/// Outcome of a Hamming(7,4) decode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Hamming74Status {
    /// The received word already satisfied all parity checks.
    Clean,
    /// A non-zero syndrome was interpreted as a single-bit error.
    Corrected { position: usize },
}

/// Auditable Hamming(7,4) decode result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Hamming74DecodeResult {
    /// Recovered four-bit payload.
    pub data: [u8; 4],
    /// Codeword after applying the single-error correction rule.
    pub corrected_codeword: [u8; 7],
    /// Whether a correction was applied.
    pub status: Hamming74Status,
}

/// Encode four validated data bits into a Hamming(7,4) codeword.
pub fn hamming74_encode_checked(data: [u8; 4]) -> Result<[u8; 7], InvalidBit> {
    validate_bits(&data)?;
    Ok(hamming74_encode_bits(data))
}

/// Decode a validated Hamming(7,4) word and report any correction.
///
/// Hamming(7,4) cannot distinguish one error from many multi-bit error
/// patterns. Use [`hamming84_decode`] when double-error detection is required.
pub fn hamming74_decode_report(codeword: [u8; 7]) -> Result<Hamming74DecodeResult, InvalidBit> {
    validate_bits(&codeword)?;
    Ok(hamming74_decode_bits(codeword))
}

/// Compatibility encoder that reduces each byte modulo two.
#[must_use]
pub fn hamming74_encode(data: [u8; 4]) -> [u8; 7] {
    hamming74_encode_bits(data.map(|bit| bit & 1))
}

/// Compatibility decoder that reduces each byte modulo two and returns only
/// the payload.
#[must_use]
pub fn hamming74_decode(codeword: [u8; 7]) -> [u8; 4] {
    hamming74_decode_bits(codeword.map(|bit| bit & 1)).data
}

/// Outcome of an extended Hamming(8,4) SECDED decode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Hamming84Status {
    /// No parity check indicated corruption.
    Clean,
    /// One of the seven Hamming positions was corrected.
    Corrected { position: usize },
    /// Only the eighth, overall-parity bit was corrected.
    CorrectedOverallParity,
    /// A two-bit error was detected and no payload is returned.
    DetectedDoubleError,
}

/// Auditable extended Hamming(8,4) SECDED decode result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Hamming84DecodeResult {
    /// Recovered payload, absent when a double error is detected.
    pub data: Option<[u8; 4]>,
    /// Word after any safe single-bit correction.
    pub corrected_codeword: [u8; 8],
    /// Error classification.
    pub status: Hamming84Status,
}

/// Encode four validated bits as extended Hamming(8,4) SECDED.
pub fn hamming84_encode_checked(data: [u8; 4]) -> Result<[u8; 8], InvalidBit> {
    validate_bits(&data)?;
    Ok(hamming84_encode_bits(data))
}

/// Compatibility SECDED encoder that reduces each byte modulo two.
#[must_use]
pub fn hamming84_encode(data: [u8; 4]) -> [u8; 8] {
    hamming84_encode_bits(data.map(|bit| bit & 1))
}

/// Decode and classify a validated extended Hamming(8,4) word.
pub fn hamming84_decode(codeword: [u8; 8]) -> Result<Hamming84DecodeResult, InvalidBit> {
    validate_bits(&codeword)?;
    Ok(hamming84_decode_bits(codeword))
}

/// Packed Hamming(7,4) decode result.
///
/// Bit `i` of `corrected_codeword` stores codeword position `i`; the payload is
/// returned in the low nibble of `data`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PackedHamming74DecodeResult {
    pub data: u8,
    pub corrected_codeword: u8,
    pub status: Hamming74Status,
}

/// Packed Hamming(8,4) SECDED decode result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PackedHamming84DecodeResult {
    pub data: Option<u8>,
    pub corrected_codeword: u8,
    pub status: Hamming84Status,
}

/// Encode a low-nibble payload into bits 0 through 6 of a byte.
pub fn hamming74_encode_packed(nibble: u8) -> Result<u8, PackedHammingError> {
    validate_nibble(nibble)?;
    Ok(pack_bits(&hamming74_encode_bits(nibble_bits(nibble))))
}

/// Decode a seven-bit packed Hamming(7,4) word.
pub fn hamming74_decode_packed(
    codeword: u8,
) -> Result<PackedHamming74DecodeResult, PackedHammingError> {
    if codeword & 0x80 != 0 {
        return Err(PackedHammingError::Hamming74WordOutOfRange { value: codeword });
    }
    let report = hamming74_decode_bits(unpack_bits::<7>(codeword));
    Ok(PackedHamming74DecodeResult {
        data: bits_nibble(report.data),
        corrected_codeword: pack_bits(&report.corrected_codeword),
        status: report.status,
    })
}

/// Encode a low-nibble payload into an eight-bit SECDED codeword.
pub fn hamming84_encode_packed(nibble: u8) -> Result<u8, PackedHammingError> {
    validate_nibble(nibble)?;
    Ok(pack_bits(&hamming84_encode_bits(nibble_bits(nibble))))
}

/// Decode any packed eight-bit SECDED word.
#[must_use]
pub fn hamming84_decode_packed(codeword: u8) -> PackedHamming84DecodeResult {
    let report = hamming84_decode_bits(unpack_bits::<8>(codeword));
    PackedHamming84DecodeResult {
        data: report.data.map(bits_nibble),
        corrected_codeword: pack_bits(&report.corrected_codeword),
        status: report.status,
    }
}

fn validate_nibble(value: u8) -> Result<(), PackedHammingError> {
    if value & 0xF0 != 0 {
        return Err(PackedHammingError::NibbleOutOfRange { value });
    }
    Ok(())
}

fn nibble_bits(nibble: u8) -> [u8; 4] {
    [
        nibble & 1,
        (nibble >> 1) & 1,
        (nibble >> 2) & 1,
        (nibble >> 3) & 1,
    ]
}

fn bits_nibble(bits: [u8; 4]) -> u8 {
    bits[0] | (bits[1] << 1) | (bits[2] << 2) | (bits[3] << 3)
}

fn pack_bits(bits: &[u8]) -> u8 {
    bits.iter()
        .enumerate()
        .fold(0, |packed, (index, &bit)| packed | (bit << index))
}

fn unpack_bits<const N: usize>(packed: u8) -> [u8; N] {
    std::array::from_fn(|index| (packed >> index) & 1)
}

fn hamming74_encode_bits(data: [u8; 4]) -> [u8; 7] {
    let [d1, d2, d3, d4] = data;
    let p1 = d1 ^ d2 ^ d4;
    let p2 = d1 ^ d3 ^ d4;
    let p3 = d2 ^ d3 ^ d4;
    [p1, p2, d1, p3, d2, d3, d4]
}

fn hamming74_decode_bits(mut codeword: [u8; 7]) -> Hamming74DecodeResult {
    let syndrome = syndrome74(&codeword);
    let status = if syndrome == 0 {
        Hamming74Status::Clean
    } else {
        let position = usize::from(syndrome - 1);
        codeword[position] ^= 1;
        Hamming74Status::Corrected { position }
    };

    Hamming74DecodeResult {
        data: extract_data74(&codeword),
        corrected_codeword: codeword,
        status,
    }
}

fn hamming84_encode_bits(data: [u8; 4]) -> [u8; 8] {
    let hamming = hamming74_encode_bits(data);
    let overall_parity = hamming.iter().fold(0, |parity, bit| parity ^ bit);
    [
        hamming[0],
        hamming[1],
        hamming[2],
        hamming[3],
        hamming[4],
        hamming[5],
        hamming[6],
        overall_parity,
    ]
}

fn hamming84_decode_bits(mut codeword: [u8; 8]) -> Hamming84DecodeResult {
    let first_seven = [
        codeword[0],
        codeword[1],
        codeword[2],
        codeword[3],
        codeword[4],
        codeword[5],
        codeword[6],
    ];
    let syndrome = syndrome74(&first_seven);
    let overall_parity = codeword.iter().fold(0, |parity, bit| parity ^ bit);

    let status = match (syndrome, overall_parity) {
        (0, 0) => Hamming84Status::Clean,
        (0, 1) => {
            codeword[7] ^= 1;
            Hamming84Status::CorrectedOverallParity
        }
        (non_zero, 1) => {
            let position = usize::from(non_zero - 1);
            codeword[position] ^= 1;
            Hamming84Status::Corrected { position }
        }
        (_, 0) => Hamming84Status::DetectedDoubleError,
        _ => unreachable!("syndrome and parity are bounded binary values"),
    };

    let data = if status == Hamming84Status::DetectedDoubleError {
        None
    } else {
        Some(extract_data74(&[
            codeword[0],
            codeword[1],
            codeword[2],
            codeword[3],
            codeword[4],
            codeword[5],
            codeword[6],
        ]))
    };

    Hamming84DecodeResult {
        data,
        corrected_codeword: codeword,
        status,
    }
}

fn syndrome74(codeword: &[u8; 7]) -> u8 {
    let s1 = codeword[0] ^ codeword[2] ^ codeword[4] ^ codeword[6];
    let s2 = codeword[1] ^ codeword[2] ^ codeword[5] ^ codeword[6];
    let s3 = codeword[3] ^ codeword[4] ^ codeword[5] ^ codeword[6];
    s1 | (s2 << 1) | (s3 << 2)
}

fn extract_data74(codeword: &[u8; 7]) -> [u8; 4] {
    [codeword[2], codeword[4], codeword[5], codeword[6]]
}

fn validate_bits(bits: &[u8]) -> Result<(), InvalidBit> {
    for (index, &value) in bits.iter().enumerate() {
        if value > 1 {
            return Err(InvalidBit { index, value });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data_bits(message: u8) -> [u8; 4] {
        [
            message & 1,
            (message >> 1) & 1,
            (message >> 2) & 1,
            (message >> 3) & 1,
        ]
    }

    #[test]
    fn weight_and_distance_work() {
        assert_eq!(weight(&[0b1011]), 3);
        assert_eq!(distance(&[0b1010], &[0b0011]), Some(2));
        assert_eq!(distance(&[0], &[0, 0]), None);
    }

    #[test]
    fn packed_hamming_round_trips_every_nibble() {
        for nibble in 0u8..16 {
            let encoded74 = hamming74_encode_packed(nibble).unwrap();
            assert_eq!(hamming74_decode_packed(encoded74).unwrap().data, nibble);

            let encoded84 = hamming84_encode_packed(nibble).unwrap();
            assert_eq!(hamming84_decode_packed(encoded84).data, Some(nibble));
        }
    }

    #[test]
    fn packed_hamming_classifies_every_single_and_double_error() {
        for nibble in 0u8..16 {
            let clean = hamming84_encode_packed(nibble).unwrap();
            for first in 0..8 {
                let corrupted = clean ^ (1 << first);
                let report = hamming84_decode_packed(corrupted);
                assert_eq!(report.data, Some(nibble));
                assert_eq!(report.corrected_codeword, clean);
            }
            for first in 0..8 {
                for second in first + 1..8 {
                    let corrupted = clean ^ (1 << first) ^ (1 << second);
                    let report = hamming84_decode_packed(corrupted);
                    assert_eq!(report.data, None);
                    assert_eq!(report.status, Hamming84Status::DetectedDoubleError);
                }
            }
        }
    }

    #[test]
    fn packed_hamming_rejects_noncanonical_widths() {
        assert_eq!(
            hamming74_encode_packed(0x10),
            Err(PackedHammingError::NibbleOutOfRange { value: 0x10 })
        );
        assert_eq!(
            hamming74_decode_packed(0x80),
            Err(PackedHammingError::Hamming74WordOutOfRange { value: 0x80 })
        );
        assert_eq!(
            hamming84_encode_packed(0xA5),
            Err(PackedHammingError::NibbleOutOfRange { value: 0xA5 })
        );
    }

    #[test]
    fn strict_apis_reject_non_bits() {
        assert_eq!(
            hamming74_encode_checked([0, 1, 2, 0]),
            Err(InvalidBit { index: 2, value: 2 })
        );
        assert_eq!(
            hamming84_decode([0, 0, 0, 0, 0, 0, 0, 9]),
            Err(InvalidBit { index: 7, value: 9 })
        );
    }

    #[test]
    fn hamming74_corrects_every_single_error_and_reports_it() {
        for message in 0u8..16 {
            let data = data_bits(message);
            let codeword = hamming74_encode_checked(data).unwrap();
            let clean = hamming74_decode_report(codeword).unwrap();
            assert_eq!(clean.data, data);
            assert_eq!(clean.status, Hamming74Status::Clean);

            for position in 0..7 {
                let mut corrupted = codeword;
                corrupted[position] ^= 1;
                let decoded = hamming74_decode_report(corrupted).unwrap();
                assert_eq!(decoded.data, data, "message={message}, position={position}");
                assert_eq!(decoded.corrected_codeword, codeword);
                assert_eq!(decoded.status, Hamming74Status::Corrected { position });
            }
        }
    }

    #[test]
    fn secded_corrects_every_single_error() {
        for message in 0u8..16 {
            let data = data_bits(message);
            let codeword = hamming84_encode_checked(data).unwrap();
            let clean = hamming84_decode(codeword).unwrap();
            assert_eq!(clean.data, Some(data));
            assert_eq!(clean.status, Hamming84Status::Clean);

            for position in 0..8 {
                let mut corrupted = codeword;
                corrupted[position] ^= 1;
                let decoded = hamming84_decode(corrupted).unwrap();
                assert_eq!(decoded.data, Some(data));
                assert_eq!(decoded.corrected_codeword, codeword);
                let expected = if position == 7 {
                    Hamming84Status::CorrectedOverallParity
                } else {
                    Hamming84Status::Corrected { position }
                };
                assert_eq!(decoded.status, expected);
            }
        }
    }

    #[test]
    fn secded_detects_every_double_error() {
        for message in 0u8..16 {
            let codeword = hamming84_encode(data_bits(message));
            for first in 0..8 {
                for second in (first + 1)..8 {
                    let mut corrupted = codeword;
                    corrupted[first] ^= 1;
                    corrupted[second] ^= 1;
                    let decoded = hamming84_decode(corrupted).unwrap();
                    assert_eq!(decoded.status, Hamming84Status::DetectedDoubleError);
                    assert_eq!(decoded.data, None);
                }
            }
        }
    }
}
