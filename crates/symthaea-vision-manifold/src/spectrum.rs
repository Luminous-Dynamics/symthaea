// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Electromagnetic spectrum extension for the vision manifold.
//!
//! Extends the pipeline from visible-light-only to multi-band sensing, where
//! each spectral band captures different physical phenomena:
//!
//! | Band | Physics captured |
//! |------|-----------------|
//! | Visible | Surface reflectance (400–700 nm) |
//! | NearIR | Vegetation, materials (700–2500 nm) |
//! | ThermalIR | Heat signature (8–14 µm) |
//! | UV | Fluorescence, pathogens (10–400 nm) |
//! | Radio | Structural penetration (mm–m) |
//!
//! ## HDC encoding strategy
//!
//! Each band's pixel data is encoded via the standard patch-HDC pipeline and
//! then **bound with a band-identity vector** unique to that band:
//!
//! ```text
//! band_frame_hv = band_id_hv ⊗ encode(band_pixels)
//! ```
//!
//! This ensures that the same spatial feature (e.g., a person) at the same
//! location produces *different* but *related* HVs across bands — the system
//! can learn cross-band correlations while preserving spectral identity.
//!
//! Final multi-spectral HV bundles all bands with equal weight:
//!
//! ```text
//! multi_hv = normalize(Σ_b  band_id_hv ⊗ encode(band_b_pixels))
//! ```
//!
//! References:
//! - Kanerva (2009) — Hyperdimensional Computing
//! - Elachi & Van Zyl (2006) — Introduction to the Physics and Techniques of Remote Sensing

use symthaea_core::hdc::ContinuousHV;

use crate::encoder::PatchHdcEncoder;
use crate::types::VisionConfig;

/// Electromagnetic spectrum bands supported by the multi-spectral pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SpectrumBand {
    /// Visible light (400–700 nm). Standard camera channels (RGB or grayscale).
    Visible,
    /// Near-infrared (700–2500 nm). Reveals vegetation health, material properties.
    NearIR,
    /// Thermal infrared (8–14 µm). Heat signature, temperature mapping.
    ThermalIR,
    /// Ultraviolet (10–400 nm). Fluorescence detection, pathogen imaging.
    UV,
    /// Radio / microwave (mm–m). Structural penetration, radar returns.
    Radio,
}

impl SpectrumBand {
    /// All supported bands in a fixed canonical order.
    pub const ALL: [Self; 5] = [
        Self::Visible,
        Self::NearIR,
        Self::ThermalIR,
        Self::UV,
        Self::Radio,
    ];

    /// Human-readable label.
    pub fn label(self) -> &'static str {
        match self {
            Self::Visible => "visible",
            Self::NearIR => "near_ir",
            Self::ThermalIR => "thermal_ir",
            Self::UV => "uv",
            Self::Radio => "radio",
        }
    }

    /// Seed offset used to generate this band's identity HV.
    ///
    /// Each band gets a unique seed so `band_id_hv(Visible) ≠ band_id_hv(NearIR)`.
    /// The offsets are large and sparse to avoid accidental similarity with
    /// the encoder's row/col/feature basis vectors.
    fn seed_offset(self) -> u64 {
        match self {
            Self::Visible => 1_000_000,
            Self::NearIR => 1_000_001,
            Self::ThermalIR => 1_000_002,
            Self::UV => 1_000_003,
            Self::Radio => 1_000_004,
        }
    }
}

/// One data layer for a single spectral band.
///
/// The `data` field contains raw 8-bit pixel intensities in row-major order,
/// normalized by the caller to [0, 255]. For thermal IR, values should be
/// mapped from physical temperature (e.g., 20–40°C → 0–255).
#[derive(Debug, Clone)]
pub struct SpectralLayer {
    pub band: SpectrumBand,
    /// Raw grayscale pixel data (row-major, 8-bit, length = width × height).
    pub data: Vec<u8>,
}

/// A multi-spectral frame containing one or more spectral layers at the same
/// spatial resolution.
///
/// Layers need not be exhaustive — only bands that are physically sensed need
/// to be present. Missing bands are simply not included in the bundled HV.
#[derive(Debug, Clone)]
pub struct MultiSpectralFrame {
    pub width: u32,
    pub height: u32,
    /// Spectral layers. May be empty (produces a zero HV).
    pub layers: Vec<SpectralLayer>,
}

impl MultiSpectralFrame {
    /// Create a frame from a single visible-light grayscale image.
    ///
    /// Convenience constructor for the common single-band case.
    pub fn from_visible(pixels: Vec<u8>, width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            layers: vec![SpectralLayer {
                band: SpectrumBand::Visible,
                data: pixels,
            }],
        }
    }

    /// Add a spectral layer to this frame.
    pub fn with_layer(mut self, band: SpectrumBand, data: Vec<u8>) -> Self {
        self.layers.push(SpectralLayer { band, data });
        self
    }
}

/// Multi-spectral HDC encoder.
///
/// Maintains one per-band identity HV and delegates per-band pixel encoding
/// to the standard [`PatchHdcEncoder`].
pub struct MultiSpectralEncoder {
    /// Shared encoder for converting pixel patches → HVs.
    encoder: PatchHdcEncoder,
    /// Pre-generated identity HVs for each supported band.
    band_hvs: Vec<(SpectrumBand, ContinuousHV)>,
    /// HDC dimension (cached for convenience).
    hdc_dim: usize,
}

impl MultiSpectralEncoder {
    /// Create a new multi-spectral encoder.
    ///
    /// # Parameters
    /// - `config` — Vision config (shared with the main pipeline).
    /// - `max_width`, `max_height` — Maximum frame dimensions.
    pub fn new(config: &VisionConfig, max_width: u32, max_height: u32) -> Self {
        let encoder = PatchHdcEncoder::new(config, max_width, max_height);
        let hdc_dim = config.hdc_dim;

        // Generate orthogonal identity HVs for each band.
        // Using the band's seed_offset ensures:
        //   1. Reproducibility (same seed → same HV)
        //   2. Near-orthogonality (all bands ~cos_sim 0 to each other)
        //   3. No collision with encoder basis vectors (large offset gap)
        let band_hvs = SpectrumBand::ALL
            .iter()
            .map(|&band| {
                let seed = config.seed + band.seed_offset();
                let hv = ContinuousHV::random(hdc_dim, seed);
                (band, hv)
            })
            .collect();

        Self {
            encoder,
            band_hvs,
            hdc_dim,
        }
    }

    /// Encode a multi-spectral frame into a single holographic hypervector.
    ///
    /// Each band's pixel data is:
    /// 1. Encoded by the patch-HDC encoder → `frame_hv`
    /// 2. Bound with the band's identity HV: `band_id_hv ⊗ frame_hv`
    ///
    /// All band contributions are bundled with equal weight and normalized.
    ///
    /// Returns a zero-filled HV if no layers are present.
    pub fn encode(&mut self, frame: &MultiSpectralFrame) -> ContinuousHV {
        if frame.layers.is_empty() {
            return ContinuousHV::zero(self.hdc_dim);
        }

        let mut band_encoded: Vec<ContinuousHV> = Vec::with_capacity(frame.layers.len());

        for layer in &frame.layers {
            // Encode the band's pixels using the shared patch encoder
            // encode_frame returns (frame_hv, patch_hvs); we only need the frame HV here.
            let (frame_hv, _patch_hvs) = self.encoder.encode_frame(
                &layer.data,
                frame.width,
                frame.height,
                1, // channels (grayscale)
            );

            // Bind with band-identity HV (tags the encoding with spectral origin)
            let band_id_hv = self.band_id_hv(layer.band);
            let tagged_hv = band_id_hv.bind(&frame_hv);
            band_encoded.push(tagged_hv);
        }

        // Bundle all band HVs with equal weight
        let refs: Vec<&ContinuousHV> = band_encoded.iter().collect();
        let weights = vec![1.0f32; refs.len()];
        ContinuousHV::weighted_bundle(&refs, &weights).normalize()
    }

    /// Decode which band a bound HV came from by probing band identity vectors.
    ///
    /// Returns the band with the highest cosine similarity to `hv ⊗ band_id_hv^{-1}`.
    /// Useful for attention probing: "which spectral region is this thought about?"
    pub fn probe_band(&self, hv: &ContinuousHV) -> SpectrumBand {
        self.band_hvs
            .iter()
            .map(|(band, band_hv)| {
                // Unbind: HRR inverse is the same vector for ContinuousHV (self-inverse bind)
                let unbound = band_hv.bind(hv);
                let sim = unbound.similarity(hv);
                (*band, sim)
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(band, _)| band)
            .unwrap_or(SpectrumBand::Visible)
    }

    /// Get the band identity HV for `band`, or a zero HV if not found.
    pub fn band_id_hv(&self, band: SpectrumBand) -> &ContinuousHV {
        self.band_hvs
            .iter()
            .find(|(b, _)| *b == band)
            .map(|(_, hv)| hv)
            .expect("all bands pre-generated in new()")
    }

    /// Check that all five band identity HVs are mutually near-orthogonal.
    ///
    /// In 16,384D, expected cos_sim ≈ 0 ± 1/√d ≈ ±0.008. This method
    /// returns `true` if all pairwise similarities are within ±0.05.
    pub fn bands_are_orthogonal(&self) -> bool {
        for i in 0..self.band_hvs.len() {
            for j in (i + 1)..self.band_hvs.len() {
                let sim = self.band_hvs[i].1.similarity(&self.band_hvs[j].1);
                if sim.abs() > 0.05 {
                    return false;
                }
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> VisionConfig {
        VisionConfig::default()
    }

    fn solid_frame(width: u32, height: u32, value: u8) -> Vec<u8> {
        vec![value; (width * height) as usize]
    }

    // === SpectrumBand ===

    #[test]
    fn test_all_bands_have_distinct_seed_offsets() {
        let offsets: std::collections::HashSet<u64> =
            SpectrumBand::ALL.iter().map(|b| b.seed_offset()).collect();
        assert_eq!(offsets.len(), 5, "All bands must have distinct seed offsets");
    }

    #[test]
    fn test_band_labels_are_distinct() {
        let labels: std::collections::HashSet<&str> =
            SpectrumBand::ALL.iter().map(|b| b.label()).collect();
        assert_eq!(labels.len(), 5, "All bands must have distinct labels");
    }

    // === MultiSpectralFrame ===

    #[test]
    fn test_from_visible_has_one_layer() {
        let frame = MultiSpectralFrame::from_visible(solid_frame(64, 64, 128), 64, 64);
        assert_eq!(frame.layers.len(), 1);
        assert_eq!(frame.layers[0].band, SpectrumBand::Visible);
    }

    #[test]
    fn test_with_layer_adds_band() {
        let frame = MultiSpectralFrame::from_visible(solid_frame(64, 64, 128), 64, 64)
            .with_layer(SpectrumBand::ThermalIR, solid_frame(64, 64, 200));
        assert_eq!(frame.layers.len(), 2);
        assert_eq!(frame.layers[1].band, SpectrumBand::ThermalIR);
    }

    // === MultiSpectralEncoder ===

    #[test]
    fn test_encoder_creates_orthogonal_band_hvs() {
        let config = default_config();
        let enc = MultiSpectralEncoder::new(&config, 64, 64);
        assert!(
            enc.bands_are_orthogonal(),
            "Band identity HVs should be mutually near-orthogonal in 16,384D"
        );
    }

    #[test]
    fn test_single_band_encode_returns_valid_hv() {
        let config = default_config();
        let mut enc = MultiSpectralEncoder::new(&config, 64, 64);
        let frame = MultiSpectralFrame::from_visible(solid_frame(64, 64, 100), 64, 64);
        let hv = enc.encode(&frame);
        // HV should have finite, non-NaN values
        assert!(
            hv.as_slice().iter().all(|x| x.is_finite()),
            "Encoded HV contains non-finite values"
        );
    }

    #[test]
    fn test_empty_frame_returns_zero_hv() {
        let config = default_config();
        let mut enc = MultiSpectralEncoder::new(&config, 64, 64);
        let frame = MultiSpectralFrame {
            width: 64,
            height: 64,
            layers: vec![],
        };
        let hv = enc.encode(&frame);
        let norm: f32 = hv.as_slice().iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(norm < 1e-6, "Empty frame should produce near-zero HV, norm={norm}");
    }

    #[test]
    fn test_different_bands_same_image_produce_different_hvs() {
        let config = default_config();
        let mut enc = MultiSpectralEncoder::new(&config, 64, 64);
        let pixels = solid_frame(64, 64, 128);

        let vis_frame = MultiSpectralFrame::from_visible(pixels.clone(), 64, 64);
        let ir_frame = MultiSpectralFrame {
            width: 64,
            height: 64,
            layers: vec![SpectralLayer {
                band: SpectrumBand::ThermalIR,
                data: pixels,
            }],
        };

        let vis_hv = enc.encode(&vis_frame);
        let ir_hv = enc.encode(&ir_frame);

        // Same image, different bands: HVs should be distinct (band-id binding)
        let sim = vis_hv.similarity(&ir_hv);
        assert!(
            sim < 0.95,
            "Same image in different bands should produce different HVs, sim={sim}"
        );
    }

    #[test]
    fn test_multi_band_frame_differs_from_single_band() {
        let config = default_config();
        let mut enc = MultiSpectralEncoder::new(&config, 64, 64);
        let pixels = solid_frame(64, 64, 100);

        let single = MultiSpectralFrame::from_visible(pixels.clone(), 64, 64);
        let multi = MultiSpectralFrame::from_visible(pixels.clone(), 64, 64)
            .with_layer(SpectrumBand::NearIR, solid_frame(64, 64, 150));

        let single_hv = enc.encode(&single);
        let multi_hv = enc.encode(&multi);

        // Different number of bands → different HV
        let sim = single_hv.similarity(&multi_hv);
        assert!(
            sim < 0.99,
            "Multi-band HV should differ from single-band HV, sim={sim}"
        );
    }

    #[test]
    fn test_same_frame_same_band_is_deterministic() {
        let config = default_config();
        let pixels = solid_frame(64, 64, 77);
        let frame = MultiSpectralFrame::from_visible(pixels, 64, 64);

        let mut enc1 = MultiSpectralEncoder::new(&config, 64, 64);
        let mut enc2 = MultiSpectralEncoder::new(&config, 64, 64);

        let hv1 = enc1.encode(&frame.clone());
        let hv2 = enc2.encode(&frame);

        let sim = hv1.similarity(&hv2);
        assert!(
            sim > 0.999,
            "Encoding should be deterministic across encoder instances, sim={sim}"
        );
    }

    #[test]
    fn test_visible_different_content_produces_different_hvs() {
        let config = default_config();
        let mut enc = MultiSpectralEncoder::new(&config, 64, 64);

        let frame_dark = MultiSpectralFrame::from_visible(solid_frame(64, 64, 10), 64, 64);
        let frame_bright = MultiSpectralFrame::from_visible(solid_frame(64, 64, 245), 64, 64);

        let hv_dark = enc.encode(&frame_dark);
        let hv_bright = enc.encode(&frame_bright);

        let sim = hv_dark.similarity(&hv_bright);
        assert!(
            sim < 0.95,
            "Different image content should produce different HVs, sim={sim}"
        );
    }

    #[test]
    fn test_band_hvs_are_not_zero() {
        let config = default_config();
        let enc = MultiSpectralEncoder::new(&config, 64, 64);
        for band in SpectrumBand::ALL {
            let hv = enc.band_id_hv(band);
            let norm: f32 = hv.as_slice().iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(norm > 0.1, "Band HV for {:?} should be non-zero, norm={norm}", band);
        }
    }

    #[test]
    fn test_all_five_bands_encode_without_panic() {
        let config = default_config();
        let mut enc = MultiSpectralEncoder::new(&config, 64, 64);
        let pixels = solid_frame(64, 64, 128);

        for band in SpectrumBand::ALL {
            let frame = MultiSpectralFrame {
                width: 64,
                height: 64,
                layers: vec![SpectralLayer {
                    band,
                    data: pixels.clone(),
                }],
            };
            let hv = enc.encode(&frame);
            assert!(
                hv.as_slice().iter().all(|x| x.is_finite()),
                "Band {:?} encoding produced non-finite values",
                band
            );
        }
    }

    #[test]
    fn test_thermal_ir_different_temperature_produces_different_hvs() {
        let config = default_config();
        let mut enc = MultiSpectralEncoder::new(&config, 64, 64);

        // Cold scene (20°C → ~0) vs hot scene (40°C → ~255)
        let cold = MultiSpectralFrame {
            width: 64,
            height: 64,
            layers: vec![SpectralLayer {
                band: SpectrumBand::ThermalIR,
                data: solid_frame(64, 64, 10),
            }],
        };
        let hot = MultiSpectralFrame {
            width: 64,
            height: 64,
            layers: vec![SpectralLayer {
                band: SpectrumBand::ThermalIR,
                data: solid_frame(64, 64, 245),
            }],
        };

        let cold_hv = enc.encode(&cold);
        let hot_hv = enc.encode(&hot);
        let sim = cold_hv.similarity(&hot_hv);
        assert!(
            sim < 0.95,
            "Hot and cold thermal frames should produce different HVs, sim={sim}"
        );
    }
}
