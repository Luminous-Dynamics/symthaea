use serde::{Deserialize, Serialize};

use crate::{blake3_digest, valid_blake3_digest, COUNTER_BYTES, Tpm2AdapterError};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Tpm2NvPublicEvidence {
    pub nv_index: u32,
    pub attributes_friendly: String,
    pub data_size: usize,
    pub raw_public_output_blake3: String,
}

impl Tpm2NvPublicEvidence {
    pub fn validate(&self, expected_index: u32) -> bool {
        self.nv_index == expected_index
            && self.data_size == COUNTER_BYTES
            && attributes_declare_counter(&self.attributes_friendly)
            && valid_blake3_digest(&self.raw_public_output_blake3)
    }
}

pub fn parse_nv_public(
    stdout: &[u8],
    expected_index: u32,
) -> Result<Tpm2NvPublicEvidence, Tpm2AdapterError> {
    let text = std::str::from_utf8(stdout).map_err(|_| Tpm2AdapterError::InvalidPublicOutput)?;
    let mut found = false;
    let mut in_expected = false;
    let mut in_attributes = false;
    let mut attributes = None;
    let mut size = None;

    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("0x") && trimmed.ends_with(':') {
            let raw = trimmed.trim_end_matches(':').trim_start_matches("0x");
            let parsed = u32::from_str_radix(raw, 16).ok();
            in_expected = parsed == Some(expected_index);
            found |= in_expected;
            in_attributes = false;
            continue;
        }
        if !in_expected {
            continue;
        }
        if trimmed == "attributes:" {
            in_attributes = true;
            continue;
        }
        if let Some(value) = trimmed.strip_prefix("size:") {
            size = value.trim().parse::<usize>().ok();
            continue;
        }
        if in_attributes {
            if let Some(value) = trimmed.strip_prefix("friendly:") {
                attributes = Some(value.trim().to_ascii_lowercase());
                in_attributes = false;
            }
        }
    }

    if !found {
        return Err(Tpm2AdapterError::NvIndexNotPresent);
    }
    let attributes = attributes.ok_or(Tpm2AdapterError::MissingAttributes)?;
    if !attributes_declare_counter(&attributes) {
        return Err(Tpm2AdapterError::NvIndexIsNotCounter);
    }
    let size = size.ok_or(Tpm2AdapterError::MissingDataSize)?;
    if size != COUNTER_BYTES {
        return Err(Tpm2AdapterError::UnexpectedDataSize(size));
    }

    Ok(Tpm2NvPublicEvidence {
        nv_index: expected_index,
        attributes_friendly: attributes,
        data_size: size,
        raw_public_output_blake3: blake3_digest(stdout),
    })
}

fn attributes_declare_counter(attributes: &str) -> bool {
    attributes
        .split('|')
        .map(str::trim)
        .any(|token| matches!(token, "counter" | "nt=counter"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn yaml_with_handle(handle: &str, attributes: &str, size: usize) -> Vec<u8> {
        format!(
            "{handle}:\n  name: 000bdeadbeef\n  hash algorithm:\n    friendly: sha256\n    value: 0xb\n  attributes:\n    friendly: {attributes}\n    value: 0x20040004\n  size: {size}\n  authorization policy:\n"
        )
        .into_bytes()
    }

    fn yaml(attributes: &str, size: usize) -> Vec<u8> {
        yaml_with_handle("0x1500016", attributes, size)
    }

    #[test]
    fn exact_counter_metadata_parses() {
        let parsed = parse_nv_public(&yaml("ownerread|ownerwrite|nt=counter", 8), 0x0150_0016)
            .unwrap();
        assert_eq!(parsed.data_size, 8);
        assert!(parsed.validate(0x0150_0016));
    }

    #[test]
    fn leading_zero_handle_format_is_equivalent() {
        let parsed = parse_nv_public(
            &yaml_with_handle("0x01500016", "ownerread|nt=counter", 8),
            0x0150_0016,
        )
        .unwrap();
        assert_eq!(parsed.nv_index, 0x0150_0016);
    }

    #[test]
    fn ordinary_index_is_not_accepted_as_counter() {
        assert_eq!(
            parse_nv_public(&yaml("ownerread|ownerwrite|ordinary", 8), 0x0150_0016),
            Err(Tpm2AdapterError::NvIndexIsNotCounter)
        );
    }

    #[test]
    fn wrong_sized_counter_is_rejected() {
        assert_eq!(
            parse_nv_public(&yaml("ownerread|nt=counter", 32), 0x0150_0016),
            Err(Tpm2AdapterError::UnexpectedDataSize(32))
        );
    }

    #[test]
    fn different_handle_is_not_substitutable() {
        assert_eq!(
            parse_nv_public(&yaml("ownerread|nt=counter", 8), 0x0150_0017),
            Err(Tpm2AdapterError::NvIndexNotPresent)
        );
    }
}
