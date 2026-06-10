// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! HDC encoder: PageObservation -> ContinuousHV (16,384D).
//!
//! Encoding strategy:
//! 1. **Role codebook**: Each ARIA role gets a genesis-seeded base vector.
//!    Roles not in the codebook use a hash-derived vector.
//! 2. **Text hashing**: Element names/values are hashed (SHAKE-256) to produce
//!    deterministic HVs, then bound (element-wise multiply) with the role vector.
//! 3. **Superposition**: All element HVs are summed with positional weighting
//!    (focused element gets 3x weight, first 10 elements get distance-decayed boost).
//! 4. **Page context**: URL and title are encoded as separate HVs and added at 0.5x weight.

use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::{ContinuousHV, HDC_DIMENSION};

use crate::observation::PageObservation;

/// Number of pre-built role codebook entries.
const ROLE_CODEBOOK_SIZE: usize = 16;

/// ARIA roles with pre-built codebook vectors.
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
///
/// Transforms a `PageObservation` into a `ContinuousHV` suitable for the
/// cognitive loop. Uses genesis-seeded codebook vectors for deterministic
/// encoding across runs.
pub struct BrowserHdcEncoder {
    /// Pre-built role codebook: role -> base HV.
    role_codebook: Vec<(String, ContinuousHV)>,
    /// Base HV for URL encoding.
    url_base: ContinuousHV,
    /// Base HV for title encoding.
    title_base: ContinuousHV,
    /// Fallback base HV for unknown roles.
    unknown_role_base: ContinuousHV,
}

impl BrowserHdcEncoder {
    /// Create a new encoder with genesis-seeded codebook.
    pub fn new(genesis: &GenesisSeed) -> Self {
        let role_codebook = KNOWN_ROLES
            .iter()
            .map(|role| {
                let label = format!("browser_role_{}", role);
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

    /// Encode a page observation into a 16,384D hypervector.
    pub fn encode(&self, observation: &PageObservation) -> ContinuousHV {
        let mut result = ContinuousHV::zero(HDC_DIMENSION);

        // Page context at 0.5x weight
        let url_hv = self.encode_text(&observation.url, &self.url_base);
        let title_hv = self.encode_text(&observation.title, &self.title_base);
        result = add_scaled(&result, &url_hv, 0.5);
        result = add_scaled(&result, &title_hv, 0.5);

        // Elements with positional weighting
        for (i, elem) in observation.elements.iter().enumerate() {
            let role_base = self.role_vector(&elem.role);
            let name_hv = self.encode_text(&elem.name, &role_base);

            // Focused element gets 3x, first 10 get distance-decayed boost
            let mut weight = 1.0_f32;
            if Some(i) == observation.focused_element {
                weight = 3.0;
            } else if i < 10 {
                weight = 1.0 + 0.5 * (1.0 - i as f32 / 10.0);
            }

            // Bind value if present
            let elem_hv = if let Some(ref val) = elem.value {
                let val_hv = self.encode_text(val, &role_base);
                bind(&name_hv, &val_hv)
            } else {
                name_hv
            };

            result = add_scaled(&result, &elem_hv, weight);
        }

        // Normalize to unit sphere
        normalize(&mut result);
        result
    }

    /// Look up or derive a role base vector.
    fn role_vector(&self, role: &str) -> ContinuousHV {
        for (r, hv) in &self.role_codebook {
            if r == role {
                return hv.clone();
            }
        }
        // Hash the unknown role name into the unknown base
        self.encode_text(role, &self.unknown_role_base)
    }

    /// Encode a text string by hashing each character and binding with the base.
    fn encode_text(&self, text: &str, base: &ContinuousHV) -> ContinuousHV {
        if text.is_empty() {
            return base.clone();
        }
        let text_hv = text_to_hv(text);
        bind(base, &text_hv)
    }
}

/// Hash a text string into a deterministic ContinuousHV.
///
/// Uses a simple hash-based approach: each character contributes a rotated
/// random vector seeded by its codepoint and position.
fn text_to_hv(text: &str) -> ContinuousHV {
    let mut result = ContinuousHV::zero(HDC_DIMENSION);
    for (pos, ch) in text.chars().enumerate() {
        let seed = (ch as u64)
            .wrapping_mul(0x517CC1B727220A95)
            .wrapping_add(pos as u64);
        let char_hv = ContinuousHV::random(HDC_DIMENSION, seed);
        result = add_scaled(&result, &char_hv, 1.0);
    }
    normalize(&mut result);
    result
}

/// Element-wise multiply (binding in HDC).
fn bind(a: &ContinuousHV, b: &ContinuousHV) -> ContinuousHV {
    let vals: Vec<f32> = a
        .as_slice()
        .iter()
        .zip(b.as_slice().iter())
        .map(|(x, y)| x * y)
        .collect();
    ContinuousHV::from_vec(vals)
}

/// Add `b` scaled by `weight` to `a`, returning a new HV.
fn add_scaled(a: &ContinuousHV, b: &ContinuousHV, weight: f32) -> ContinuousHV {
    let vals: Vec<f32> = a
        .as_slice()
        .iter()
        .zip(b.as_slice().iter())
        .map(|(x, y)| x + y * weight)
        .collect();
    ContinuousHV::from_vec(vals)
}

/// In-place L2 normalization to unit sphere.
fn normalize(hv: &mut ContinuousHV) {
    let norm: f32 = hv.as_slice().iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-10 {
        let inv = 1.0 / norm;
        let vals: Vec<f32> = hv.as_slice().iter().map(|x| x * inv).collect();
        *hv = ContinuousHV::from_vec(vals);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoder_dimensionality() {
        let genesis = GenesisSeed::from_phrase("test-browser");
        let encoder = BrowserHdcEncoder::new(&genesis);
        let obs = PageObservation {
            url: "https://example.com".into(),
            title: "Example".into(),
            elements: vec![],
            focused_element: None,
        };
        let hv = encoder.encode(&obs);
        assert_eq!(hv.dim(), HDC_DIMENSION);
    }

    #[test]
    fn test_different_pages_produce_different_hvs() {
        let genesis = GenesisSeed::from_phrase("test-browser");
        let encoder = BrowserHdcEncoder::new(&genesis);

        let obs1 = PageObservation {
            url: "https://example.com".into(),
            title: "Example".into(),
            elements: vec![],
            focused_element: None,
        };
        let obs2 = PageObservation {
            url: "https://other.com".into(),
            title: "Other".into(),
            elements: vec![],
            focused_element: None,
        };

        let hv1 = encoder.encode(&obs1);
        let hv2 = encoder.encode(&obs2);
        let sim = hv1.similarity(&hv2);
        // Different pages should not be identical
        assert!(sim < 0.99, "similarity too high: {}", sim);
    }

    #[test]
    fn test_normalized_output() {
        let genesis = GenesisSeed::from_phrase("test-browser");
        let encoder = BrowserHdcEncoder::new(&genesis);

        let obs = PageObservation {
            url: "https://example.com".into(),
            title: "Test".into(),
            elements: vec![crate::observation::AccessibleElement {
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

        let hv = encoder.encode(&obs);
        let norm: f32 = hv.as_slice().iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 0.01,
            "output not normalized: norm = {}",
            norm
        );
    }
}
