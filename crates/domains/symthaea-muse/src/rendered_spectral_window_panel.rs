// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Homogeneous replication panels for descriptive C6A frequency-domain evidence.
//!
//! The panel retains every attack-level spectral observation, summarizes the
//! observed metric distributions, and separately records direct waveform-sample
//! reuse under C6A's wider spectral windows. It emits no aggregate success or
//! onset/audibility verdict.

use crate::evidence_digest::{
    canonical_json_sha256,
    rendered_spectral_window_evidence::{
        RENDERED_SPECTRAL_REPRESENTATION_ID, RENDERED_SPECTRAL_WINDOW_EVIDENCE_VERSION,
        RenderedSpectralWindowConfigV1, RenderedSpectralWindowEvidenceErrorV1,
        RenderedSpectralWindowEvidenceV1, verify_rendered_spectral_window_evidence,
    },
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const RENDERED_SPECTRAL_WINDOW_PANEL_VERSION: &str =
    "rendered-spectral-window-panel-v1";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RenderedSpectralPanelSampleV1 {
    pub subject_id: String,
    pub attack_ordinal: usize,
    pub evidence: RenderedSpectralWindowEvidenceV1,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RenderedSpectralMetricSummaryV1 {
    pub minimum: f64,
    pub maximum: f64,
    pub mean: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RenderedSpectralWindowClusterMemberV1 {
    pub attack_ordinal: usize,
    pub source_evidence_sha256: String,
    pub center_sample: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RenderedSpectralWindowClusterV1 {
    pub subject_id: String,
    pub cluster_ordinal: usize,
    pub start_sample: usize,
    pub end_sample: usize,
    pub member_count: usize,
    pub members: Vec<RenderedSpectralWindowClusterMemberV1>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RenderedSpectralSubjectSummaryV1 {
    pub subject_id: String,
    pub attack_count: usize,
    pub window_overlap_cluster_count: usize,
    pub singleton_cluster_count: usize,
    pub multi_member_cluster_count: usize,
    pub attacks_in_overlapping_clusters: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RenderedSpectralWindowPanelV1 {
    pub panel_version: String,
    pub source_evidence_version: String,
    pub representation_id: String,
    pub sample_rate: u32,
    pub config: RenderedSpectralWindowConfigV1,
    pub subject_count: usize,
    pub attack_count: usize,
    pub window_overlap_cluster_count: usize,
    pub singleton_cluster_count: usize,
    pub multi_member_cluster_count: usize,
    pub attacks_in_overlapping_clusters: usize,
    pub baseline_positive_logmel_flux: RenderedSpectralMetricSummaryV1,
    pub candidate_positive_logmel_flux: RenderedSpectralMetricSummaryV1,
    pub candidate_flux_excess: RenderedSpectralMetricSummaryV1,
    pub pre_between_arm_rms_distance: RenderedSpectralMetricSummaryV1,
    pub post_between_arm_rms_distance: RenderedSpectralMetricSummaryV1,
    pub subjects: Vec<RenderedSpectralSubjectSummaryV1>,
    pub clusters: Vec<RenderedSpectralWindowClusterV1>,
    /// Every attack-level observation remains available. Panel summaries and
    /// overlap clusters never replace or erase the underlying evidence.
    pub samples: Vec<RenderedSpectralPanelSampleV1>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RenderedSpectralWindowPanelErrorV1 {
    EmptyPanel,
    EmptySubjectId,
    DuplicateAttackIdentity,
    InvalidEvidence(RenderedSpectralWindowEvidenceErrorV1),
    MixedSampleRate,
    MixedConfig,
    EvidenceDigestFailed,
}

/// Summarize one homogeneous set of C6A spectral observations.
///
/// Half-open intervals `[pre_start_sample, post_end_sample)` are clustered
/// independently per subject. Touching intervals share no samples and remain
/// separate. Non-overlapping clusters are not claimed statistically independent.
pub fn summarize_rendered_spectral_window_panel(
    samples: &[RenderedSpectralPanelSampleV1],
) -> Result<RenderedSpectralWindowPanelV1, RenderedSpectralWindowPanelErrorV1> {
    let Some(first) = samples.first() else {
        return Err(RenderedSpectralWindowPanelErrorV1::EmptyPanel);
    };
    verify_rendered_spectral_window_evidence(&first.evidence)
        .map_err(RenderedSpectralWindowPanelErrorV1::InvalidEvidence)?;

    let sample_rate = first.evidence.sample_rate;
    let config = first.evidence.config;
    let mut identities = BTreeSet::new();
    let mut by_subject: BTreeMap<String, Vec<&RenderedSpectralPanelSampleV1>> = BTreeMap::new();
    let mut baseline_flux = Vec::with_capacity(samples.len());
    let mut candidate_flux = Vec::with_capacity(samples.len());
    let mut flux_excess = Vec::with_capacity(samples.len());
    let mut pre_distance = Vec::with_capacity(samples.len());
    let mut post_distance = Vec::with_capacity(samples.len());

    for sample in samples {
        if sample.subject_id.trim().is_empty() {
            return Err(RenderedSpectralWindowPanelErrorV1::EmptySubjectId);
        }
        if !identities.insert((sample.subject_id.clone(), sample.attack_ordinal)) {
            return Err(RenderedSpectralWindowPanelErrorV1::DuplicateAttackIdentity);
        }
        verify_rendered_spectral_window_evidence(&sample.evidence)
            .map_err(RenderedSpectralWindowPanelErrorV1::InvalidEvidence)?;
        if sample.evidence.sample_rate != sample_rate {
            return Err(RenderedSpectralWindowPanelErrorV1::MixedSampleRate);
        }
        if sample.evidence.config != config {
            return Err(RenderedSpectralWindowPanelErrorV1::MixedConfig);
        }

        baseline_flux.push(sample.evidence.baseline_positive_logmel_flux);
        candidate_flux.push(sample.evidence.candidate_positive_logmel_flux);
        flux_excess.push(sample.evidence.candidate_flux_excess);
        pre_distance.push(sample.evidence.pre_between_arm_rms_distance);
        post_distance.push(sample.evidence.post_between_arm_rms_distance);
        by_subject
            .entry(sample.subject_id.clone())
            .or_default()
            .push(sample);
    }

    let mut subjects = Vec::with_capacity(by_subject.len());
    let mut clusters = Vec::new();

    for (subject_id, mut subject_samples) in by_subject {
        subject_samples.sort_by(|left, right| {
            left.evidence
                .pre_start_sample
                .cmp(&right.evidence.pre_start_sample)
                .then_with(|| left.evidence.post_end_sample.cmp(&right.evidence.post_end_sample))
                .then_with(|| left.attack_ordinal.cmp(&right.attack_ordinal))
        });

        let subject_cluster_start = clusters.len();
        let mut cluster_ordinal = 0usize;
        let mut current_start = 0usize;
        let mut current_end = 0usize;
        let mut current_members: Vec<RenderedSpectralWindowClusterMemberV1> = Vec::new();

        let flush = |clusters: &mut Vec<RenderedSpectralWindowClusterV1>,
                     current_start: usize,
                     current_end: usize,
                     current_members: &mut Vec<RenderedSpectralWindowClusterMemberV1>,
                     cluster_ordinal: usize| {
            if current_members.is_empty() {
                return;
            }
            clusters.push(RenderedSpectralWindowClusterV1 {
                subject_id: subject_id.clone(),
                cluster_ordinal,
                start_sample: current_start,
                end_sample: current_end,
                member_count: current_members.len(),
                members: std::mem::take(current_members),
            });
        };

        for sample in subject_samples {
            let member = RenderedSpectralWindowClusterMemberV1 {
                attack_ordinal: sample.attack_ordinal,
                source_evidence_sha256: canonical_json_sha256(&sample.evidence)
                    .map_err(|_| RenderedSpectralWindowPanelErrorV1::EvidenceDigestFailed)?,
                center_sample: sample.evidence.center_sample,
            };
            let start = sample.evidence.pre_start_sample;
            let end = sample.evidence.post_end_sample;

            if current_members.is_empty() {
                current_start = start;
                current_end = end;
                current_members.push(member);
                continue;
            }

            if start < current_end {
                current_end = current_end.max(end);
                current_members.push(member);
            } else {
                flush(
                    &mut clusters,
                    current_start,
                    current_end,
                    &mut current_members,
                    cluster_ordinal,
                );
                cluster_ordinal += 1;
                current_start = start;
                current_end = end;
                current_members.push(member);
            }
        }
        flush(
            &mut clusters,
            current_start,
            current_end,
            &mut current_members,
            cluster_ordinal,
        );

        let subject_clusters = &clusters[subject_cluster_start..];
        let singleton_cluster_count = subject_clusters
            .iter()
            .filter(|cluster| cluster.member_count == 1)
            .count();
        let multi_member_cluster_count = subject_clusters.len() - singleton_cluster_count;
        let attacks_in_overlapping_clusters = subject_clusters
            .iter()
            .filter(|cluster| cluster.member_count > 1)
            .map(|cluster| cluster.member_count)
            .sum();
        let attack_count = subject_clusters
            .iter()
            .map(|cluster| cluster.member_count)
            .sum();

        subjects.push(RenderedSpectralSubjectSummaryV1 {
            subject_id,
            attack_count,
            window_overlap_cluster_count: subject_clusters.len(),
            singleton_cluster_count,
            multi_member_cluster_count,
            attacks_in_overlapping_clusters,
        });
    }

    let singleton_cluster_count = clusters
        .iter()
        .filter(|cluster| cluster.member_count == 1)
        .count();
    let multi_member_cluster_count = clusters.len() - singleton_cluster_count;
    let attacks_in_overlapping_clusters = clusters
        .iter()
        .filter(|cluster| cluster.member_count > 1)
        .map(|cluster| cluster.member_count)
        .sum();

    Ok(RenderedSpectralWindowPanelV1 {
        panel_version: RENDERED_SPECTRAL_WINDOW_PANEL_VERSION.into(),
        source_evidence_version: RENDERED_SPECTRAL_WINDOW_EVIDENCE_VERSION.into(),
        representation_id: RENDERED_SPECTRAL_REPRESENTATION_ID.into(),
        sample_rate,
        config,
        subject_count: subjects.len(),
        attack_count: samples.len(),
        window_overlap_cluster_count: clusters.len(),
        singleton_cluster_count,
        multi_member_cluster_count,
        attacks_in_overlapping_clusters,
        baseline_positive_logmel_flux: summarize_metric(&baseline_flux),
        candidate_positive_logmel_flux: summarize_metric(&candidate_flux),
        candidate_flux_excess: summarize_metric(&flux_excess),
        pre_between_arm_rms_distance: summarize_metric(&pre_distance),
        post_between_arm_rms_distance: summarize_metric(&post_distance),
        subjects,
        clusters,
        samples: samples.to_vec(),
    })
}

fn summarize_metric(values: &[f64]) -> RenderedSpectralMetricSummaryV1 {
    RenderedSpectralMetricSummaryV1 {
        minimum: values.iter().copied().fold(f64::INFINITY, f64::min),
        maximum: values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        mean: values.iter().sum::<f64>() / values.len() as f64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evidence_digest::rendered_spectral_window_evidence::measure_rendered_spectral_window;

    const SAMPLE_RATE: u32 = 8_192;

    fn config(fft_size: usize) -> RenderedSpectralWindowConfigV1 {
        RenderedSpectralWindowConfigV1 {
            fft_size,
            mel_bands: 24,
            f_min_hz: 20.0,
            f_max_hz: 3_500.0,
        }
    }

    fn sample(
        subject: &str,
        ordinal: usize,
        attack_time: f32,
        config: RenderedSpectralWindowConfigV1,
    ) -> RenderedSpectralPanelSampleV1 {
        let audio = vec![[0.0, 0.0]; SAMPLE_RATE as usize * 2];
        let evidence = measure_rendered_spectral_window(
            &audio,
            &audio,
            SAMPLE_RATE,
            attack_time,
            config,
        )
        .unwrap();
        RenderedSpectralPanelSampleV1 {
            subject_id: subject.into(),
            attack_ordinal: ordinal,
            evidence,
        }
    }

    #[test]
    fn wider_spectral_windows_cluster_direct_sample_reuse_transitively() {
        let cfg = config(256);
        let panel = summarize_rendered_spectral_window_panel(&[
            sample("seed-1", 0, 0.50, cfg),
            sample("seed-1", 1, 0.52, cfg),
            sample("seed-1", 2, 0.54, cfg),
            sample("seed-1", 3, 0.80, cfg),
            sample("seed-2", 0, 0.50, cfg),
        ])
        .unwrap();

        assert_eq!(panel.subject_count, 2);
        assert_eq!(panel.attack_count, 5);
        assert_eq!(panel.window_overlap_cluster_count, 3);
        assert_eq!(panel.multi_member_cluster_count, 1);
        assert_eq!(panel.attacks_in_overlapping_clusters, 3);
        assert_eq!(panel.samples.len(), 5);
        assert_eq!(panel.baseline_positive_logmel_flux.mean, 0.0);
    }

    #[test]
    fn touching_half_open_spectral_windows_remain_separate() {
        let cfg = config(256);
        // Two windows have total width 512 samples. A 512-sample center
        // separation makes the half-open intervals touch but not overlap.
        let first_center = 4_096usize;
        let second_center = first_center + 512;
        let panel = summarize_rendered_spectral_window_panel(&[
            sample(
                "seed-1",
                0,
                first_center as f32 / SAMPLE_RATE as f32,
                cfg,
            ),
            sample(
                "seed-1",
                1,
                second_center as f32 / SAMPLE_RATE as f32,
                cfg,
            ),
        ])
        .unwrap();
        assert_eq!(panel.window_overlap_cluster_count, 2);
        assert_eq!(panel.multi_member_cluster_count, 0);
    }

    #[test]
    fn mixed_valid_spectral_configuration_is_rejected() {
        let result = summarize_rendered_spectral_window_panel(&[
            sample("seed-1", 0, 0.50, config(256)),
            sample("seed-2", 0, 0.50, config(512)),
        ]);
        assert_eq!(result, Err(RenderedSpectralWindowPanelErrorV1::MixedConfig));
    }

    #[test]
    fn forged_evidence_and_duplicate_identity_fail_closed() {
        let cfg = config(256);
        let one = sample("seed-1", 0, 0.50, cfg);
        assert_eq!(
            summarize_rendered_spectral_window_panel(&[one.clone(), one]),
            Err(RenderedSpectralWindowPanelErrorV1::DuplicateAttackIdentity)
        );

        let mut forged = sample("seed-1", 0, 0.50, cfg);
        forged.evidence.candidate_flux_excess = 1.0;
        assert_eq!(
            summarize_rendered_spectral_window_panel(&[forged]),
            Err(RenderedSpectralWindowPanelErrorV1::InvalidEvidence(
                RenderedSpectralWindowEvidenceErrorV1::InconsistentStoredSummary
            ))
        );
    }
}
