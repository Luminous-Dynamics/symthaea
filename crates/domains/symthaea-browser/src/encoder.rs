// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Bounded HDC encoder: [`PageObservation`] -> `ContinuousHV(16_384)`.
//!
//! Encoding strategy:
//! 1. Genesis-seeded role vectors provide stable semantic channels.
//! 2. Bounded text is projected deterministically with a dense whole-text seed
//!    plus a lightweight token feature sketch.
//! 3. Role/name/value representations are bound and accumulated in one mutable
//!    buffer rather than allocating a new 16,384D vector for every addition.
//! 4. Focus and early-page salience affect weight; URL data is redacted before
//!    encoding.

use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::{ContinuousHV, HDC_DIMENSION};

use crate::observation::{MAX_OBSERVATION_TEXT_CHARS, PageObservation};

const ROLE_CODEBOOK_SIZE: usize = 16;
const MAX_ENCODER_ELEMENTS: usize = 256;
const MAX_FEATURE_TOKENS: usize = 32;

const KNOWN_ROLES: [&str; ROLE_CODEBOOK_SIZE] = [
    "button",
    "link",
    "textbox",
    "heading",
    "checkbox",
    "radio",
    "combobox",
    "menuitem",
    "tab",
    "slider",
    "img",
    "list",
    "listitem",
    "navigation",
    "main",
    "dialog",
];

/// Browser-specific HDC encoder.
pub struct BrowserHdcEncoder {
    role_codebook: Vec<(String, ContinuousHV)>,
    url_base: ContinuousHV,
    title_base: ContinuousHV,
    unknown_role_base: ContinuousHV,
}

impl BrowserHdcEncoder {
    pub fn new(genesis: &GenesisSeed) -> Self {
        let role_codebook = KNOWN_ROLES
            .iter()
            .map(|role| {
                let label = format!("browser_role_{role}");
                let hv = ContinuousHV::from_genesis(genesis, &label, HDC_DIMENSION);
                (role.to_string(), hv)
            })
            .collect();

        Self {
            role_codebook,
            url_base: ContinuousHV::from_genesis(genesis, "browser_url", HDC_DIMENSION),
            title_base: ContinuousHV::from_genesis(genesis, "browser_title", HDC_DIMENSION),
            unknown_role_base: ContinuousHV::from_genesis(
                genesis,
                "browser_unknown_role",
                HDC_DIMENSION,
            ),
        }
    }

    /// Encode a page observation into a normalized hypervector.
    pub fn encode(&self, observation: &PageObservation) -> ContinuousHV {
        let mut accumulator = vec![0.0_f32; HDC_DIMENSION];

        let url = observation.redacted_url();
        let url_hv = self.encode_text(&url, &self.url_base);
        let title_hv = self.encode_text(&observation.title, &self.title_base);
        accumulate(&mut accumulator, &url_hv, 0.5);
        accumulate(&mut accumulator, &title_hv, 0.5);

        for (index, element) in observation
            .elements
            .iter()
            .take(MAX_ENCODER_ELEMENTS)
            .enumerate()
        {
            let derived_role;
            let role_base = if let Some(known) = self.known_role_vector(&element.role) {
                known
            } else {
                derived_role = self.encode_text(&element.role, &self.unknown_role_base);
                &derived_role
            };

            let name_hv = self.encode_text(&element.name, role_base);
            let mut weight = if index < 10 {
                1.0 + 0.5 * (1.0 - index as f32 / 10.0)
            } else {
                1.0
            };
            if observation.focused_element == Some(index) || element.focused {
                weight = 3.0;
            }
            if element.disabled {
                weight *= 0.5;
            }

            if let Some(value) = element.value.as_deref() {
                let value_hv = self.encode_text(value, role_base);
                accumulate_bound(&mut accumulator, &name_hv, &value_hv, weight);
            } else {
                accumulate(&mut accumulator, &name_hv, weight);
            }

            if let Some(description) = element.description.as_deref() {
                let description_hv = self.encode_text(description, role_base);
                accumulate(&mut accumulator, &description_hv, weight * 0.25);
            }
        }

        normalize_values(&mut accumulator);
        ContinuousHV::from_vec(accumulator)
    }

    fn known_role_vector(&self, role: &str) -> Option<&ContinuousHV> {
        self.role_codebook
            .iter()
            .find_map(|(known_role, vector)| (known_role == role).then_some(vector))
    }

    fn encode_text(&self, text: &str, base: &ContinuousHV) -> ContinuousHV {
        if text.is_empty() {
            return base.clone();
        }
        let text_hv = text_to_hv(text);
        bind(base, &text_hv)
    }
}

/// Project bounded text into a deterministic dense vector.
///
/// Unlike the previous implementation, this allocates one vector per text field
/// rather than one 16,384D random vector per character.
fn text_to_hv(text: &str) -> ContinuousHV {
    let normalized: String = text
        .chars()
        .filter(|character| !character.is_control())
        .flat_map(char::to_lowercase)
        .take(MAX_OBSERVATION_TEXT_CHARS)
        .collect();

    if normalized.is_empty() {
        return ContinuousHV::zero(HDC_DIMENSION);
    }

    let mut state = stable_hash(normalized.as_bytes());
    let mut values = vec![0.0_f32; HDC_DIMENSION];
    for value in &mut values {
        state = splitmix64(state);
        *value = if state & 1 == 0 { -1.0 } else { 1.0 };
    }

    // Shared tokens leave a sparse similarity trace on top of the dense text
    // identity vector without requiring another dense allocation per token.
    for token in normalized.split_whitespace().take(MAX_FEATURE_TOKENS) {
        let token_hash = stable_hash(token.as_bytes());
        for lane in 0..8_u64 {
            let mixed = splitmix64(token_hash.wrapping_add(lane));
            let index = (mixed as usize) % HDC_DIMENSION;
            let sign = if mixed & (1 << 63) == 0 { -1.0 } else { 1.0 };
            values[index] += sign * 8.0;
        }
    }

    normalize_values(&mut values);
    ContinuousHV::from_vec(values)
}

fn stable_hash(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
    value ^ (value >> 31)
}

fn bind(left: &ContinuousHV, right: &ContinuousHV) -> ContinuousHV {
    ContinuousHV::from_vec(
        left.as_slice()
            .iter()
            .zip(right.as_slice())
            .map(|(left, right)| left * right)
            .collect(),
    )
}

fn accumulate(target: &mut [f32], vector: &ContinuousHV, weight: f32) {
    for (target, value) in target.iter_mut().zip(vector.as_slice()) {
        *target += value * weight;
    }
}

fn accumulate_bound(target: &mut [f32], left: &ContinuousHV, right: &ContinuousHV, weight: f32) {
    for ((target, left), right) in target.iter_mut().zip(left.as_slice()).zip(right.as_slice()) {
        *target += left * right * weight;
    }
}

fn normalize_values(values: &mut [f32]) {
    let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > 1e-10 {
        let inverse = 1.0 / norm;
        for value in values {
            *value *= inverse;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observation::AccessibleElement;

    #[test]
    fn encoder_output_is_normalized_and_dimensioned() {
        let genesis = GenesisSeed::from_phrase("test-browser");
        let encoder = BrowserHdcEncoder::new(&genesis);
        let observation = PageObservation {
            url: "https://example.com".into(),
            title: "Example".into(),
            elements: vec![AccessibleElement {
                backend_node_id: 1,
                role: "button".into(),
                name: "Click me".into(),
                value: None,
                description: None,
                focused: true,
                disabled: false,
            }],
            focused_element: Some(0),
        };
        let vector = encoder.encode(&observation);
        assert_eq!(vector.dim(), HDC_DIMENSION);
        let norm = vector
            .as_slice()
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt();
        assert!((norm - 1.0).abs() < 0.01, "norm = {norm}");
    }

    #[test]
    fn text_projection_is_deterministic_and_bounded() {
        let long = "browser observation ".repeat(10_000);
        let first = text_to_hv(&long);
        let second = text_to_hv(&long);
        assert!(first.similarity(&second) > 0.9999);
        assert_eq!(first.dim(), HDC_DIMENSION);
    }

    #[test]
    fn different_pages_produce_different_vectors() {
        let genesis = GenesisSeed::from_phrase("test-browser");
        let encoder = BrowserHdcEncoder::new(&genesis);
        let first = PageObservation {
            url: "https://example.com".into(),
            title: "Example".into(),
            elements: Vec::new(),
            focused_element: None,
        };
        let second = PageObservation {
            url: "https://other.example".into(),
            title: "Other".into(),
            elements: Vec::new(),
            focused_element: None,
        };
        assert!(encoder.encode(&first).similarity(&encoder.encode(&second)) < 0.99);
    }
}
