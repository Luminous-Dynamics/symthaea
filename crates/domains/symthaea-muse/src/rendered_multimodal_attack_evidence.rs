// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Bind verified time-domain and frequency-domain evidence to one exact
//! performed attack without pretending the two feature views are independent
//! observations.

use crate::evidence_digest::{
    canonical_json_sha256,
    rendered_attack_crosscheck::{
        RenderedAttackCrosscheckErrorV1, RenderedAttackCrosscheckV1,
        verify_rendered_attack_crosscheck,
    },
    rendered_spectral_window_evidence::{
        RenderedSpectralWindowEvidenceErrorV1, RenderedSpectralWindowEvidenceV1,
        verify_rendered_spectral_window_evidence,
    },
};
use serde::{Deserialize, Serialize};

pub const RENDERED_MULTIMODAL_ATTACK_EVIDENCE_VERSION: &str =
    "rendered-multimodal-attack-evidence-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RenderedMultimodalWindowOverlapV1 {
    pub time_domain_start_sample: usize,
    pub time_domain_end_sample: usize,
    pub spectral_start_sample: usize,
    pub spectral_end_sample: usize,
    /// Number of raw waveform samples reused by both feature views.
    pub shared_sample_count: usize,
    /// Samples used only by the C5 time-domain window.
    pub time_domain_only_sample_count: usize,
    /// Samples used only by the C6 spectral window.
    pub spectral_only_sample_count: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RenderedMultimodalAttackEvidenceV1 {
    pub evidence_version: String,
    pub sample_rate: u32,
    pub attack_time_seconds: f32,
    pub center_sample: usize,
    /// Canonical identity of the complete C5 cross-check record.
    pub time_domain_crosscheck_sha256: String,
    /// Canonical identity of the C5 matched-window evidence retained by C5B.
    pub time_domain_source_evidence_sha256: String,
    /// Canonical identity of the complete C6A spectral record.
    pub spectral_evidence_sha256: String,
    pub time_domain: RenderedAttackCrosscheckV1,
    pub spectral: RenderedSpectralWindowEvidenceV1,
    pub raw_sample_overlap: RenderedMultimodalWindowOverlapV1,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RenderedMultimodalAttackEvidenceErrorV1 {
    UnsupportedEvidenceVersion,
    InvalidTimeDomain(RenderedAttackCrosscheckErrorV1),
    InvalidSpectral(RenderedSpectralWindowEvidenceErrorV1),
    SampleRateMismatch,
    AttackTimeMismatch,
    CenterSampleMismatch,
    InvalidWindowGeometry,
    DigestFailed,
    InconsistentStoredBinding,
}

/// Bind two verified acoustic feature views to one exact performed attack.
///
/// The views intentionally keep their native window geometry. Their raw sample
/// overlap is measured and retained so downstream code cannot interpret
/// agreement between time/frequency descriptors as independent replication.
pub fn bind_rendered_multimodal_attack_evidence(
    time_domain: &RenderedAttackCrosscheckV1,
    spectral: &RenderedSpectralWindowEvidenceV1,
) -> Result<RenderedMultimodalAttackEvidenceV1, RenderedMultimodalAttackEvidenceErrorV1> {
    verify_rendered_attack_crosscheck(time_domain)
        .map_err(RenderedMultimodalAttackEvidenceErrorV1::InvalidTimeDomain)?;
    verify_rendered_spectral_window_evidence(spectral)
        .map_err(RenderedMultimodalAttackEvidenceErrorV1::InvalidSpectral)?;

    validate_shared_attack_identity(time_domain, spectral)?;
    let raw_sample_overlap = derive_window_overlap(time_domain, spectral)?;
    let time_domain_crosscheck_sha256 = canonical_json_sha256(time_domain)
        .map_err(|_| RenderedMultimodalAttackEvidenceErrorV1::DigestFailed)?;
    let spectral_evidence_sha256 = canonical_json_sha256(spectral)
        .map_err(|_| RenderedMultimodalAttackEvidenceErrorV1::DigestFailed)?;

    Ok(RenderedMultimodalAttackEvidenceV1 {
        evidence_version: RENDERED_MULTIMODAL_ATTACK_EVIDENCE_VERSION.into(),
        sample_rate: time_domain.sample_rate,
        attack_time_seconds: time_domain.attack_time_seconds,
        center_sample: time_domain.center_sample,
        time_domain_crosscheck_sha256,
        time_domain_source_evidence_sha256: time_domain.source_evidence_sha256.clone(),
        spectral_evidence_sha256,
        time_domain: time_domain.clone(),
        spectral: spectral.clone(),
        raw_sample_overlap,
    })
}

/// Verify serialized multimodal evidence without trusting its stored digests,
/// attack binding, or overlap accounting.
pub fn verify_rendered_multimodal_attack_evidence(
    evidence: &RenderedMultimodalAttackEvidenceV1,
) -> Result<(), RenderedMultimodalAttackEvidenceErrorV1> {
    if evidence.evidence_version != RENDERED_MULTIMODAL_ATTACK_EVIDENCE_VERSION {
        return Err(RenderedMultimodalAttackEvidenceErrorV1::UnsupportedEvidenceVersion);
    }
    verify_rendered_attack_crosscheck(&evidence.time_domain)
        .map_err(RenderedMultimodalAttackEvidenceErrorV1::InvalidTimeDomain)?;
    verify_rendered_spectral_window_evidence(&evidence.spectral)
        .map_err(RenderedMultimodalAttackEvidenceErrorV1::InvalidSpectral)?;
    validate_shared_attack_identity(&evidence.time_domain, &evidence.spectral)?;

    let expected_overlap = derive_window_overlap(&evidence.time_domain, &evidence.spectral)?;
    let expected_time_digest = canonical_json_sha256(&evidence.time_domain)
        .map_err(|_| RenderedMultimodalAttackEvidenceErrorV1::DigestFailed)?;
    let expected_spectral_digest = canonical_json_sha256(&evidence.spectral)
        .map_err(|_| RenderedMultimodalAttackEvidenceErrorV1::DigestFailed)?;

    if evidence.sample_rate != evidence.time_domain.sample_rate
        || evidence.attack_time_seconds.to_bits()
            != evidence.time_domain.attack_time_seconds.to_bits()
        || evidence.center_sample != evidence.time_domain.center_sample
        || evidence.time_domain_crosscheck_sha256 != expected_time_digest
        || evidence.time_domain_source_evidence_sha256
            != evidence.time_domain.source_evidence_sha256
        || evidence.spectral_evidence_sha256 != expected_spectral_digest
        || evidence.raw_sample_overlap != expected_overlap
    {
        return Err(RenderedMultimodalAttackEvidenceErrorV1::InconsistentStoredBinding);
    }

    Ok(())
}

fn validate_shared_attack_identity(
    time_domain: &RenderedAttackCrosscheckV1,
    spectral: &RenderedSpectralWindowEvidenceV1,
) -> Result<(), RenderedMultimodalAttackEvidenceErrorV1> {
    if time_domain.sample_rate != spectral.sample_rate {
        return Err(RenderedMultimodalAttackEvidenceErrorV1::SampleRateMismatch);
    }
    if time_domain.attack_time_seconds.to_bits() != spectral.attack_time_seconds.to_bits() {
        return Err(RenderedMultimodalAttackEvidenceErrorV1::AttackTimeMismatch);
    }
    if time_domain.center_sample != spectral.center_sample {
        return Err(RenderedMultimodalAttackEvidenceErrorV1::CenterSampleMismatch);
    }
    Ok(())
}

fn derive_window_overlap(
    time_domain: &RenderedAttackCrosscheckV1,
    spectral: &RenderedSpectralWindowEvidenceV1,
) -> Result<RenderedMultimodalWindowOverlapV1, RenderedMultimodalAttackEvidenceErrorV1> {
    let Some((time_start, time_end)) = time_domain.sample_interval() else {
        return Err(RenderedMultimodalAttackEvidenceErrorV1::InvalidWindowGeometry);
    };
    let spectral_start = spectral.pre_start_sample;
    let spectral_end = spectral.post_end_sample;
    if time_start >= time_end || spectral_start >= spectral_end {
        return Err(RenderedMultimodalAttackEvidenceErrorV1::InvalidWindowGeometry);
    }

    let shared_start = time_start.max(spectral_start);
    let shared_end = time_end.min(spectral_end);
    let shared_sample_count = shared_end.saturating_sub(shared_start);
    let time_domain_len = time_end - time_start;
    let spectral_len = spectral_end - spectral_start;

    Ok(RenderedMultimodalWindowOverlapV1 {
        time_domain_start_sample: time_start,
        time_domain_end_sample: time_end,
        spectral_start_sample: spectral_start,
        spectral_end_sample: spectral_end,
        shared_sample_count,
        time_domain_only_sample_count: time_domain_len - shared_sample_count,
        spectral_only_sample_count: spectral_len - shared_sample_count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evidence_digest::{
        rendered_attack_crosscheck::{
            RenderedAttackCrosscheckClassV1, evaluate_rendered_attack_crosscheck,
        },
        rendered_attack_evidence::measure_rendered_attack,
        rendered_attack_gate::RenderedAttackGateConfigV1,
        rendered_spectral_window_evidence::{
            RenderedSpectralWindowConfigV1, measure_rendered_spectral_window,
        },
        rendered_transient_contrast::RenderedTransientContrastConfigV1,
    };

    const SAMPLE_RATE: u32 = 8_192;
    const ATTACK_TIME: f32 = 0.5;

    fn audio() -> Vec<[f32; 2]> {
        vec![[0.0, 0.0]; SAMPLE_RATE as usize]
    }

    fn spectral_config() -> RenderedSpectralWindowConfigV1 {
        RenderedSpectralWindowConfigV1 {
            fft_size: 256,
            mel_bands: 24,
            f_min_hz: 20.0,
            f_max_hz: 3_500.0,
        }
    }

    fn pair(
        attack_time: f32,
    ) -> (RenderedAttackCrosscheckV1, RenderedSpectralWindowEvidenceV1) {
        let baseline = audio();
        let candidate = baseline.clone();
        let time = measure_rendered_attack(
            &baseline,
            &candidate,
            SAMPLE_RATE,
            ATTACK_TIME,
            0.010,
            0.020,
        )
        .unwrap();
        let crosscheck = evaluate_rendered_attack_crosscheck(
            &time,
            RenderedAttackGateConfigV1::default(),
            RenderedTransientContrastConfigV1::default(),
        )
        .unwrap();
        let spectral = measure_rendered_spectral_window(
            &baseline,
            &candidate,
            SAMPLE_RATE,
            attack_time,
            spectral_config(),
        )
        .unwrap();
        (crosscheck, spectral)
    }

    #[test]
    fn binds_same_attack_but_retains_shared_raw_sample_dependence() {
        let (time_domain, spectral) = pair(ATTACK_TIME);
        let evidence = bind_rendered_multimodal_attack_evidence(&time_domain, &spectral).unwrap();
        verify_rendered_multimodal_attack_evidence(&evidence).unwrap();

        assert_eq!(evidence.center_sample, time_domain.center_sample);
        assert_eq!(evidence.center_sample, spectral.center_sample);
        assert_eq!(evidence.time_domain_crosscheck_sha256.len(), 64);
        assert_eq!(evidence.spectral_evidence_sha256.len(), 64);
        assert!(evidence.raw_sample_overlap.shared_sample_count > 0);
        assert_eq!(evidence.raw_sample_overlap.time_domain_only_sample_count, 0);
        assert!(evidence.raw_sample_overlap.spectral_only_sample_count > 0);
    }

    #[test]
    fn different_marked_attack_time_is_rejected() {
        let (time_domain, spectral) = pair(0.6);
        assert_eq!(
            bind_rendered_multimodal_attack_evidence(&time_domain, &spectral),
            Err(RenderedMultimodalAttackEvidenceErrorV1::AttackTimeMismatch)
        );
    }

    #[test]
    fn forged_nested_or_overlap_state_fails_closed() {
        let (mut time_domain, spectral) = pair(ATTACK_TIME);
        time_domain.class = RenderedAttackCrosscheckClassV1::LocalizedDifferenceOnly;
        assert!(matches!(
            bind_rendered_multimodal_attack_evidence(&time_domain, &spectral),
            Err(RenderedMultimodalAttackEvidenceErrorV1::InvalidTimeDomain(_))
        ));

        let (time_domain, spectral) = pair(ATTACK_TIME);
        let mut evidence =
            bind_rendered_multimodal_attack_evidence(&time_domain, &spectral).unwrap();
        evidence.raw_sample_overlap.shared_sample_count += 1;
        assert_eq!(
            verify_rendered_multimodal_attack_evidence(&evidence),
            Err(RenderedMultimodalAttackEvidenceErrorV1::InconsistentStoredBinding)
        );
    }
}
