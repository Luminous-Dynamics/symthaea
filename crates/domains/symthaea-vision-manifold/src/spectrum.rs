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

use serde::{Deserialize, Serialize};
use symthaea_core::hdc::ContinuousHV;

use crate::encoder::PatchHdcEncoder;
use crate::types::VisionConfig;

/// Electromagnetic spectrum bands supported by the multi-spectral pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
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

/// One ranked spectral-band explanation for an encoded observation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BandProbeScore {
    pub band: SpectrumBand,
    pub score: f32,
}

/// Ambiguity-aware evidence returned by spectral probing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BandProbeEvidence {
    /// Ranked explanations, best first.
    pub rankings: Vec<BandProbeScore>,
    /// Best band only when evidence is available.
    pub best_band: Option<SpectrumBand>,
    /// Similarity of the best explanation.
    pub best_score: f32,
    /// Difference between the best and runner-up explanations.
    pub margin: f32,
    /// Whether score and margin both satisfy the caller's thresholds.
    pub confident: bool,
}

/// Serializable temporal and cached evidence for one spectral encoder.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpectralBandEncoderState {
    pub band: SpectrumBand,
    pub prev_patch_lum: Vec<f32>,
    pub last_frame_hv: Option<Vec<f32>>,
}

/// Serializable state for a [`MultiSpectralEncoder`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MultiSpectralEncoderState {
    pub schema_version: u32,
    pub hdc_dim: usize,
    pub bands: Vec<SpectralBandEncoderState>,
}

const MULTISPECTRAL_STATE_SCHEMA_VERSION: u32 = 1;

impl MultiSpectralFrame {
    /// Create an empty frame at the given resolution, with no layers yet.
    ///
    /// Layers are added via [`Self::with_layer`].
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            layers: Vec::new(),
        }
    }

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
    /// Independent temporal encoder per spectral band. Sharing one encoder
    /// would misinterpret cross-band differences as frame-to-frame motion.
    encoders: Vec<(SpectrumBand, PatchHdcEncoder)>,
    /// Pre-generated identity HVs for each supported band.
    band_hvs: Vec<(SpectrumBand, ContinuousHV)>,
    /// Most recent untagged frame encoding for each observed band.
    last_band_frames: Vec<(SpectrumBand, ContinuousHV)>,
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
        let encoders = SpectrumBand::ALL
            .iter()
            .map(|&band| (band, PatchHdcEncoder::new(config, max_width, max_height)))
            .collect();
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
            encoders,
            band_hvs,
            last_band_frames: Vec::new(),
            hdc_dim,
        }
    }

    /// Clear per-band temporal history and cached probe evidence while
    /// preserving band identities and learned feature weights.
    pub fn reset_runtime(&mut self) {
        for (_, encoder) in &mut self.encoders {
            encoder.prev_patch_lum.clear();
        }
        self.last_band_frames.clear();
    }

    /// Current HDC dimension of all band encoders and identities.
    pub fn hdc_dim(&self) -> usize {
        self.hdc_dim
    }

    /// Perform 'Holographic Dilation' - scale all internal encoders and band HVs.
    pub fn dilate(&mut self, target_dim: usize) {
        if self.hdc_dim == target_dim {
            return;
        }

        for (_, encoder) in &mut self.encoders {
            encoder.dilate(target_dim);
        }
        for (_, hv) in &mut self.band_hvs {
            *hv = hv.dilate(target_dim);
        }
        for (_, frame_hv) in &mut self.last_band_frames {
            *frame_hv = frame_hv.dilate(target_dim);
        }
        self.hdc_dim = target_dim;
    }

    /// Encode a multi-spectral frame into a single holographic hypervector.
    ///
    /// This compatibility wrapper fails closed to a zero HV on malformed
    /// sensor input. New integrations should prefer [`Self::encode_checked`]
    /// so geometry or duplicate-band errors remain visible to the caller.
    pub fn encode(&mut self, frame: &MultiSpectralFrame) -> ContinuousHV {
        match self.encode_checked(frame) {
            Ok(hv) => hv,
            Err(error) => {
                tracing::warn!(%error, "rejected malformed multispectral frame");
                ContinuousHV::zero(self.hdc_dim)
            }
        }
    }

    /// Validate and encode a multi-spectral frame.
    ///
    /// Every layer must match `width × height`, dimensions must be non-zero,
    /// and a physical band may appear at most once. Rejecting duplicate bands
    /// prevents accidental re-weighting of one sensor by repetition.
    pub fn encode_checked(&mut self, frame: &MultiSpectralFrame) -> Result<ContinuousHV, String> {
        if frame.layers.is_empty() {
            return Ok(ContinuousHV::zero(self.hdc_dim));
        }
        if frame.width == 0 || frame.height == 0 {
            return Err(format!(
                "multispectral dimensions must be non-zero, got {}x{}",
                frame.width, frame.height
            ));
        }

        let expected_len = (frame.width as usize)
            .checked_mul(frame.height as usize)
            .ok_or_else(|| "multispectral frame geometry overflow".to_string())?;
        let mut seen = std::collections::HashSet::new();
        for layer in &frame.layers {
            if !seen.insert(layer.band) {
                return Err(format!(
                    "duplicate multispectral band: {}",
                    layer.band.label()
                ));
            }
            if layer.data.len() != expected_len {
                return Err(format!(
                    "{} layer length mismatch: got {}, expected {} for {}x{}",
                    layer.band.label(),
                    layer.data.len(),
                    expected_len,
                    frame.width,
                    frame.height
                ));
            }
        }

        // Capacity validation must also happen before any per-band encoder
        // advances its temporal history. Otherwise an oversized later layer can
        // leave earlier bands committed even though the frame is rejected.
        for layer in &frame.layers {
            let encoder = self
                .encoders
                .iter()
                .find(|(band, _)| *band == layer.band)
                .map(|(_, encoder)| encoder)
                .expect("all bands have an independent encoder");
            let grid = encoder.grid_for(frame.width, frame.height);
            if grid.rows > encoder.max_rows() || grid.cols > encoder.max_cols() {
                return Err(format!(
                    "{} layer {}x{} exceeds encoder capacity of {}x{} patches",
                    layer.band.label(),
                    frame.width,
                    frame.height,
                    encoder.max_cols(),
                    encoder.max_rows()
                ));
            }
        }

        let mut band_encoded: Vec<ContinuousHV> = Vec::with_capacity(frame.layers.len());

        for layer in &frame.layers {
            let frame_hv = {
                let encoder = self
                    .encoders
                    .iter_mut()
                    .find(|(band, _)| *band == layer.band)
                    .map(|(_, encoder)| encoder)
                    .expect("all bands have an independent encoder");
                let (frame_hv, _patch_hvs) =
                    encoder.encode_frame(&layer.data, frame.width, frame.height, 1);
                frame_hv
            };

            if let Some((_, cached)) = self
                .last_band_frames
                .iter_mut()
                .find(|(band, _)| *band == layer.band)
            {
                *cached = frame_hv.clone();
            } else {
                self.last_band_frames.push((layer.band, frame_hv.clone()));
            }

            let tagged_hv = self.band_id_hv(layer.band).bind(&frame_hv);
            band_encoded.push(tagged_hv);
        }

        let refs: Vec<&ContinuousHV> = band_encoded.iter().collect();
        let weights = vec![1.0f32; refs.len()];
        Ok(ContinuousHV::weighted_bundle(&refs, &weights).normalize())
    }

    /// Snapshot independent per-band temporal histories and cached evidence.
    pub fn save_state(&self) -> MultiSpectralEncoderState {
        MultiSpectralEncoderState {
            schema_version: MULTISPECTRAL_STATE_SCHEMA_VERSION,
            hdc_dim: self.hdc_dim,
            bands: self
                .encoders
                .iter()
                .map(|(band, encoder)| SpectralBandEncoderState {
                    band: *band,
                    prev_patch_lum: encoder.prev_patch_lum.clone(),
                    last_frame_hv: self
                        .last_band_frames
                        .iter()
                        .find(|(cached_band, _)| cached_band == band)
                        .map(|(_, hv)| hv.as_slice().to_vec()),
                })
                .collect(),
        }
    }

    /// Validate a saved state without mutating the encoder.
    pub fn validate_state(&self, state: &MultiSpectralEncoderState) -> Result<(), String> {
        if state.schema_version > MULTISPECTRAL_STATE_SCHEMA_VERSION {
            return Err(format!(
                "unsupported multispectral checkpoint schema: saved={}, supported={}",
                state.schema_version, MULTISPECTRAL_STATE_SCHEMA_VERSION
            ));
        }
        if state.schema_version == 0 {
            return Err("multispectral checkpoint schema must be non-zero".to_string());
        }
        if state.hdc_dim != self.hdc_dim {
            return Err(format!(
                "multispectral HDC dimension mismatch: saved={}, current={}",
                state.hdc_dim, self.hdc_dim
            ));
        }
        if state.bands.len() != SpectrumBand::ALL.len() {
            return Err(format!(
                "multispectral band-state count mismatch: saved={}, expected={}",
                state.bands.len(),
                SpectrumBand::ALL.len()
            ));
        }

        let mut seen = std::collections::HashSet::new();
        for band_state in &state.bands {
            if !seen.insert(band_state.band) {
                return Err(format!(
                    "duplicate multispectral checkpoint band: {}",
                    band_state.band.label()
                ));
            }
            if band_state
                .prev_patch_lum
                .iter()
                .any(|value| !value.is_finite())
            {
                return Err(format!(
                    "{} temporal history contains non-finite values",
                    band_state.band.label()
                ));
            }
            if let Some(values) = &band_state.last_frame_hv {
                if values.len() != self.hdc_dim {
                    return Err(format!(
                        "{} cached frame dimension mismatch: saved={}, expected={}",
                        band_state.band.label(),
                        values.len(),
                        self.hdc_dim
                    ));
                }
                if values.iter().any(|value| !value.is_finite()) {
                    return Err(format!(
                        "{} cached frame contains non-finite values",
                        band_state.band.label()
                    ));
                }
            }
        }
        for band in SpectrumBand::ALL {
            if !seen.contains(&band) {
                return Err(format!(
                    "multispectral checkpoint is missing band: {}",
                    band.label()
                ));
            }
        }
        Ok(())
    }

    /// Restore per-band temporal histories and probe references atomically.
    pub fn load_state(&mut self, state: &MultiSpectralEncoderState) -> Result<(), String> {
        self.validate_state(state)?;

        for band_state in &state.bands {
            let encoder = self
                .encoders
                .iter_mut()
                .find(|(band, _)| *band == band_state.band)
                .map(|(_, encoder)| encoder)
                .expect("validated checkpoints contain every supported band");
            encoder.prev_patch_lum = band_state.prev_patch_lum.clone();
        }
        self.last_band_frames = state
            .bands
            .iter()
            .filter_map(|band_state| {
                band_state
                    .last_frame_hv
                    .as_ref()
                    .map(|values| (band_state.band, ContinuousHV::from_vec(values.clone())))
            })
            .collect();
        Ok(())
    }

    /// Rank recently observed bands by how well they explain a tagged or
    /// bundled HV. Only bands with cached, untagged evidence are returned.
    pub fn probe_bands(&self, hv: &ContinuousHV) -> Vec<(SpectrumBand, f32)> {
        let mut scores: Vec<(SpectrumBand, f32)> = self
            .band_hvs
            .iter()
            .filter_map(|(band, band_hv)| {
                let reference = self
                    .last_band_frames
                    .iter()
                    .find(|(cached_band, _)| cached_band == band)
                    .map(|(_, frame_hv)| frame_hv)?;
                if hv.dim() != band_hv.dim() || reference.dim() != hv.dim() {
                    return None;
                }
                let unbound = band_hv.bind(hv);
                let score = unbound.similarity(reference);
                score.is_finite().then_some((*band, score))
            })
            .collect();
        scores.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.seed_offset().cmp(&b.0.seed_offset()))
        });
        scores
    }

    /// Evaluate spectral identity without forcing a label under ambiguity.
    pub fn probe_evidence(
        &self,
        hv: &ContinuousHV,
        min_score: f32,
        min_margin: f32,
    ) -> Result<BandProbeEvidence, String> {
        if !min_score.is_finite() || !min_margin.is_finite() {
            return Err("spectral probe thresholds must be finite".to_string());
        }
        if min_margin < 0.0 {
            return Err("spectral probe margin must be non-negative".to_string());
        }
        if hv.dim() != self.hdc_dim {
            return Err(format!(
                "spectral probe dimension mismatch: got {}, expected {}",
                hv.dim(),
                self.hdc_dim
            ));
        }

        let rankings: Vec<BandProbeScore> = self
            .probe_bands(hv)
            .into_iter()
            .map(|(band, score)| BandProbeScore { band, score })
            .collect();
        let best_band = rankings.first().map(|entry| entry.band);
        let best_score = rankings.first().map_or(0.0, |entry| entry.score);
        let margin = match rankings.get(1) {
            Some(runner_up) => best_score - runner_up.score,
            None if best_score > 0.0 => best_score,
            None => 0.0,
        };
        let confident = best_band.is_some() && best_score >= min_score && margin >= min_margin;

        Ok(BandProbeEvidence {
            rankings,
            best_band,
            best_score,
            margin,
            confident,
        })
    }

    /// Return a band only when score and runner-up margin are sufficient.
    pub fn probe_band_confident(
        &self,
        hv: &ContinuousHV,
        min_score: f32,
        min_margin: f32,
    ) -> Result<Option<SpectrumBand>, String> {
        let evidence = self.probe_evidence(hv, min_score, min_margin)?;
        Ok(if evidence.confident {
            evidence.best_band
        } else {
            None
        })
    }

    /// Decode which recently observed band best explains a tagged or bundled
    /// HV. Returns `Visible` only when no band evidence is available.
    /// Prefer [`Self::probe_band_confident`] for decisions that must fail closed.
    pub fn probe_band(&self, hv: &ContinuousHV) -> SpectrumBand {
        self.probe_bands(hv)
            .first()
            .map(|(band, _)| *band)
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
        assert_eq!(
            offsets.len(),
            5,
            "All bands must have distinct seed offsets"
        );
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
        assert!(
            norm < 1e-6,
            "Empty frame should produce near-zero HV, norm={norm}"
        );
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
            assert!(
                norm > 0.1,
                "Band HV for {:?} should be non-zero, norm={norm}",
                band
            );
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

    #[test]
    fn test_layer_order_does_not_create_cross_band_motion() {
        let config = default_config();
        let visible = solid_frame(64, 64, 40);
        let thermal = solid_frame(64, 64, 220);
        let frame_a = MultiSpectralFrame {
            width: 64,
            height: 64,
            layers: vec![
                SpectralLayer {
                    band: SpectrumBand::Visible,
                    data: visible.clone(),
                },
                SpectralLayer {
                    band: SpectrumBand::ThermalIR,
                    data: thermal.clone(),
                },
            ],
        };
        let frame_b = MultiSpectralFrame {
            width: 64,
            height: 64,
            layers: vec![
                SpectralLayer {
                    band: SpectrumBand::ThermalIR,
                    data: thermal,
                },
                SpectralLayer {
                    band: SpectrumBand::Visible,
                    data: visible,
                },
            ],
        };

        let mut enc_a = MultiSpectralEncoder::new(&config, 64, 64);
        let mut enc_b = MultiSpectralEncoder::new(&config, 64, 64);
        let hv_a = enc_a.encode(&frame_a);
        let hv_b = enc_b.encode(&frame_b);
        assert!(
            hv_a.similarity(&hv_b) > 0.999,
            "band order must not alter temporal features"
        );
    }

    #[test]
    fn test_probe_band_uses_cached_untagged_reference() {
        let config = default_config();
        let mut encoder = MultiSpectralEncoder::new(&config, 64, 64);
        let frame = MultiSpectralFrame {
            width: 64,
            height: 64,
            layers: vec![SpectralLayer {
                band: SpectrumBand::ThermalIR,
                data: solid_frame(64, 64, 180),
            }],
        };
        let tagged = encoder.encode(&frame);
        assert_eq!(encoder.probe_band(&tagged), SpectrumBand::ThermalIR);
    }
    #[test]
    fn test_checked_encode_rejects_malformed_layer() {
        let config = default_config();
        let mut encoder = MultiSpectralEncoder::new(&config, 8, 8);
        let frame = MultiSpectralFrame::from_visible(vec![0; 63], 8, 8);
        let error = encoder.encode_checked(&frame).unwrap_err();
        assert!(error.contains("length mismatch"));
    }

    #[test]
    fn test_checked_encode_rejects_duplicate_band() {
        let config = default_config();
        let mut encoder = MultiSpectralEncoder::new(&config, 8, 8);
        let frame = MultiSpectralFrame::from_visible(vec![10; 64], 8, 8)
            .with_layer(SpectrumBand::Visible, vec![20; 64]);
        let error = encoder.encode_checked(&frame).unwrap_err();
        assert!(error.contains("duplicate"));
    }

    #[test]
    fn test_checked_encode_rejects_over_capacity_before_mutation() {
        let mut config = default_config();
        config.hdc_dim = 256;
        config.patch_size = 4;
        let mut encoder = MultiSpectralEncoder::new(&config, 8, 8);
        encoder
            .encode_checked(&MultiSpectralFrame::from_visible(vec![80; 64], 8, 8))
            .unwrap();
        let before = encoder.save_state();

        let oversized = MultiSpectralFrame::from_visible(vec![90; 16 * 8], 16, 8);
        let error = encoder.encode_checked(&oversized).unwrap_err();

        assert!(error.contains("exceeds encoder capacity"));
        assert_eq!(encoder.save_state(), before);
    }

    #[test]
    fn test_multispectral_state_roundtrip_preserves_band_histories() {
        let mut config = default_config();
        config.hdc_dim = 256;
        config.patch_size = 4;
        let mut source = MultiSpectralEncoder::new(&config, 8, 8);

        let visible = MultiSpectralFrame::from_visible(vec![30; 64], 8, 8);
        source.encode_checked(&visible).unwrap();
        let thermal = MultiSpectralFrame {
            width: 8,
            height: 8,
            layers: vec![SpectralLayer {
                band: SpectrumBand::ThermalIR,
                data: vec![210; 64],
            }],
        };
        let tagged_thermal = source.encode_checked(&thermal).unwrap();
        let saved = source.save_state();

        let visible_state = saved
            .bands
            .iter()
            .find(|state| state.band == SpectrumBand::Visible)
            .unwrap();
        let thermal_state = saved
            .bands
            .iter()
            .find(|state| state.band == SpectrumBand::ThermalIR)
            .unwrap();
        assert!(!visible_state.prev_patch_lum.is_empty());
        assert!(!thermal_state.prev_patch_lum.is_empty());
        assert_ne!(visible_state.prev_patch_lum, thermal_state.prev_patch_lum);

        let mut restored = MultiSpectralEncoder::new(&config, 8, 8);
        restored.load_state(&saved).unwrap();
        assert_eq!(restored.save_state(), saved);
        assert_eq!(
            restored
                .probe_band_confident(&tagged_thermal, 0.5, 0.2)
                .unwrap(),
            Some(SpectrumBand::ThermalIR)
        );
    }

    #[test]
    fn test_spectral_probe_refuses_ambiguous_bundle() {
        let mut config = default_config();
        config.hdc_dim = 1024;
        config.patch_size = 4;
        let mut encoder = MultiSpectralEncoder::new(&config, 8, 8);
        let frame = MultiSpectralFrame::from_visible(vec![40; 64], 8, 8)
            .with_layer(SpectrumBand::ThermalIR, vec![220; 64]);
        encoder.encode_checked(&frame).unwrap();

        let visible_frame = &encoder
            .last_band_frames
            .iter()
            .find(|(band, _)| *band == SpectrumBand::Visible)
            .unwrap()
            .1;
        let thermal_frame = &encoder
            .last_band_frames
            .iter()
            .find(|(band, _)| *band == SpectrumBand::ThermalIR)
            .unwrap()
            .1;
        let tagged_visible = encoder
            .band_id_hv(SpectrumBand::Visible)
            .bind(visible_frame);
        let tagged_thermal = encoder
            .band_id_hv(SpectrumBand::ThermalIR)
            .bind(thermal_frame);
        let ambiguous =
            ContinuousHV::weighted_bundle(&[&tagged_visible, &tagged_thermal], &[1.0, 1.0])
                .normalize();

        let evidence = encoder.probe_evidence(&ambiguous, 0.2, 0.2).unwrap();
        assert_eq!(evidence.rankings.len(), 2);
        assert!(!evidence.confident);
        assert!(evidence.margin < 0.2);
        assert_eq!(
            encoder.probe_band_confident(&ambiguous, 0.2, 0.2).unwrap(),
            None
        );
    }

    #[test]
    fn test_multispectral_future_schema_rejected_before_mutation() {
        let mut config = default_config();
        config.hdc_dim = 256;
        config.patch_size = 4;
        let mut encoder = MultiSpectralEncoder::new(&config, 8, 8);
        encoder
            .encode_checked(&MultiSpectralFrame::from_visible(vec![80; 64], 8, 8))
            .unwrap();
        let before = encoder.save_state();
        let mut future = before.clone();
        future.schema_version = MULTISPECTRAL_STATE_SCHEMA_VERSION + 1;
        future.bands[0].prev_patch_lum.clear();

        let error = encoder.load_state(&future).unwrap_err();
        assert!(error.contains("unsupported multispectral checkpoint schema"));
        assert_eq!(encoder.save_state(), before);
    }

    #[test]
    fn test_probe_bands_exposes_ranked_evidence() {
        let config = default_config();
        let mut encoder = MultiSpectralEncoder::new(&config, 8, 8);
        let thermal = MultiSpectralFrame {
            width: 8,
            height: 8,
            layers: vec![SpectralLayer {
                band: SpectrumBand::ThermalIR,
                data: vec![180; 64],
            }],
        };
        let tagged = encoder.encode_checked(&thermal).unwrap();
        let scores = encoder.probe_bands(&tagged);
        assert_eq!(
            scores.first().map(|entry| entry.0),
            Some(SpectrumBand::ThermalIR)
        );
        assert!(scores.first().unwrap().1.is_finite());
    }
}
