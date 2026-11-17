//! Explicitly declared raw-bit decoder attempts.
//!
//! This module never scores semantic plausibility or chooses a "best message".
//! It returns raw bytes and printable-byte diagnostics only.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BitOrder {
    MostSignificantFirst,
    LeastSignificantFirst,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PayloadEncoding {
    Ascii7,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecoderSpec {
    pub id: String,
    pub bits_per_symbol: u8,
    pub bit_order: BitOrder,
    pub encoding: PayloadEncoding,
}

impl DecoderSpec {
    pub fn ascii7(id: impl Into<String>, bit_order: BitOrder) -> Self {
        Self {
            id: id.into(),
            bits_per_symbol: 7,
            bit_order,
            encoding: PayloadEncoding::Ascii7,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedPayload {
    pub bytes: Vec<u8>,
    /// Hex is unambiguous even when decoded bytes are not printable text.
    pub hex: String,
    pub printable_ascii_fraction: f64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecodeError {
    EmptyDecoderId,
    UnsupportedWidth { requested: u8, expected: u8 },
    IncompleteSymbol { bit_count: usize, width: usize },
    NonAsciiInput { byte_index: usize },
}

impl std::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyDecoderId => write!(f, "decoder id must not be empty"),
            Self::UnsupportedWidth { requested, expected } => write!(
                f,
                "ASCII-7 requires {expected} bits per symbol; received {requested}"
            ),
            Self::IncompleteSymbol { bit_count, width } => write!(
                f,
                "{bit_count} bits do not form complete {width}-bit symbols"
            ),
            Self::NonAsciiInput { byte_index } => {
                write!(f, "input byte {byte_index} is not 7-bit ASCII")
            }
        }
    }
}

impl std::error::Error for DecodeError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodeOutcome {
    pub decoder: DecoderSpec,
    pub result: Result<DecodedPayload, DecodeError>,
}

/// Decode exactly the declared encoding. No orientation, dimensions, codebook,
/// language model or error-correction search is performed here.
pub fn decode(bits: &[bool], spec: &DecoderSpec) -> Result<DecodedPayload, DecodeError> {
    if spec.id.trim().is_empty() {
        return Err(DecodeError::EmptyDecoderId);
    }
    if spec.bits_per_symbol != 7 {
        return Err(DecodeError::UnsupportedWidth {
            requested: spec.bits_per_symbol,
            expected: 7,
        });
    }
    if bits.is_empty() || bits.len() % 7 != 0 {
        return Err(DecodeError::IncompleteSymbol {
            bit_count: bits.len(),
            width: 7,
        });
    }

    let mut bytes = Vec::with_capacity(bits.len() / 7);
    for symbol in bits.chunks_exact(7) {
        let mut value = 0_u8;
        match spec.bit_order {
            BitOrder::MostSignificantFirst => {
                for bit in symbol {
                    value = (value << 1) | u8::from(*bit);
                }
            }
            BitOrder::LeastSignificantFirst => {
                for (index, bit) in symbol.iter().enumerate() {
                    value |= u8::from(*bit) << index;
                }
            }
        }
        bytes.push(value);
    }

    let printable_count = bytes
        .iter()
        .filter(|byte| (0x20..=0x7e).contains(byte) || matches!(**byte, b'\n' | b'\r' | b'\t'))
        .count();
    let printable_ascii_fraction = printable_count as f64 / bytes.len() as f64;
    let hex = bytes.iter().map(|byte| format!("{byte:02x}")).collect();

    Ok(DecodedPayload {
        bytes,
        hex,
        printable_ascii_fraction,
    })
}

/// Encode a known 7-bit ASCII positive control using a declared bit order.
pub fn encode_ascii7(text: &str, order: BitOrder) -> Result<Vec<bool>, DecodeError> {
    let mut bits = Vec::with_capacity(text.len() * 7);
    for (byte_index, byte) in text.bytes().enumerate() {
        if byte > 0x7f {
            return Err(DecodeError::NonAsciiInput { byte_index });
        }
        match order {
            BitOrder::MostSignificantFirst => {
                for shift in (0..7).rev() {
                    bits.push(((byte >> shift) & 1) == 1);
                }
            }
            BitOrder::LeastSignificantFirst => {
                for shift in 0..7 {
                    bits.push(((byte >> shift) & 1) == 1);
                }
            }
        }
    }
    Ok(bits)
}

/// Run the complete supplied decoder list and retain every success/failure.
/// The caller is responsible for ensuring this list was frozen in advance.
pub fn run_tournament(bits: &[bool], specs: &[DecoderSpec]) -> Vec<DecodeOutcome> {
    specs
        .iter()
        .cloned()
        .map(|decoder| DecodeOutcome {
            result: decode(bits, &decoder),
            decoder,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_ascii_positive_control_round_trips() {
        let bits = encode_ascii7("HELLO", BitOrder::MostSignificantFirst)
            .expect("ASCII positive control");
        let result = decode(
            &bits,
            &DecoderSpec::ascii7("positive-control-msb", BitOrder::MostSignificantFirst),
        )
        .expect("valid bit sequence");

        assert_eq!(result.bytes, b"HELLO");
        assert_eq!(result.hex, "48454c4c4f");
        assert_eq!(result.printable_ascii_fraction, 1.0);
    }

    #[test]
    fn least_significant_first_is_explicit_and_round_trips() {
        let bits = encode_ascii7("A", BitOrder::LeastSignificantFirst)
            .expect("ASCII positive control");
        let result = decode(
            &bits,
            &DecoderSpec::ascii7("positive-control-lsb", BitOrder::LeastSignificantFirst),
        )
        .expect("valid bit sequence");

        assert_eq!(result.bytes, b"A");
    }

    #[test]
    fn tournament_preserves_all_attempts() {
        let bits = encode_ascii7("A", BitOrder::MostSignificantFirst).expect("valid ASCII");
        let specs = [
            DecoderSpec::ascii7("msb", BitOrder::MostSignificantFirst),
            DecoderSpec::ascii7("lsb", BitOrder::LeastSignificantFirst),
        ];
        let outcomes = run_tournament(&bits, &specs);

        assert_eq!(outcomes.len(), 2);
        assert_eq!(outcomes[0].decoder.id, "msb");
        assert_eq!(outcomes[1].decoder.id, "lsb");
        assert!(outcomes.iter().all(|outcome| outcome.result.is_ok()));
    }

    #[test]
    fn incomplete_or_unsupported_inputs_fail_closed() {
        let decoder = DecoderSpec::ascii7("seven-bit", BitOrder::MostSignificantFirst);
        assert_eq!(
            decode(&[true, false], &decoder),
            Err(DecodeError::IncompleteSymbol {
                bit_count: 2,
                width: 7
            })
        );

        let unsupported = DecoderSpec {
            id: "eight-bit".to_string(),
            bits_per_symbol: 8,
            bit_order: BitOrder::MostSignificantFirst,
            encoding: PayloadEncoding::Ascii7,
        };
        assert_eq!(
            decode(&[false; 8], &unsupported),
            Err(DecodeError::UnsupportedWidth {
                requested: 8,
                expected: 7
            })
        );
    }
}
