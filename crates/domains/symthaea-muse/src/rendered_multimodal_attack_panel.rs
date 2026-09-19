// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Homogeneous panels for exact C5+C6 multimodal attack evidence.
//!
//! The panel keeps time-domain class outcomes, spectral observations, raw-sample
//! dependence, and each representation's direct window-overlap topology
//! separate. It emits no aggregate musical-quality or perceptual verdict.

use crate::evidence_digest::{
    rendered_attack_crosscheck::RenderedAttackCrosscheckClassV1,
    rendered_multimodal_attack_evidence::{
        RenderedMultimodalAttackEvidenceErrorV1, RenderedMultimodalAttackEvidenceV1,
        verify_rendered_multimodal_attack_evidence,
    },
    rendered_spectral_window_evidence::RenderedSpectralWindowConfigV1,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const RENDERED_MULTIMODAL_ATTACK_PANEL_VERSION: &str =
    "rendered-multimodal-attack-panel-v1";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RenderedMultimodalPanelSampleV1 {
    pub subject_id: String,
    pub attack_ordinal: usize,
    pub evidence: RenderedMultimodalAttackEvidenceV1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RenderedMultimodalWindowViewV1 {
    TimeDomain,
    Spectral,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RenderedMultimodalWindowClusterV1 {
    pub view: RenderedMultimodalWindowViewV1,
    pub subject_id: String,
    pub cluster_ordinal: usize,
    pub start_sample: usize,
    pub end_sample: usize,
    pub member_attack_ordinals: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RenderedMultimodalClassCountsV1 {
    pub localized_difference_and_candidate_transient: usize,
    pub localized_difference_only: usize,
    pub candidate_transient_only: usize,
    pub neither: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RenderedMultimodalSampleCountSummaryV1 {
    pub minimum: usize,
    pub maximum: usize,
    pub mean: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RenderedMultimodalSubjectSummaryV1 {
    pub subject_id: String,
    pub attack_count: usize,
    pub time_domain_window_cluster_count: usize,
    pub spectral_window_cluster_count: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RenderedMultimodalAttackPanelV1 {
    pub panel_version: String,
    pub sample_rate: u32,
    pub time_domain_pre_samples: usize,
    pub time_domain_post_samples: usize,
    pub spectral_config: RenderedSpectralWindowConfigV1,
    pub subject_count: usize,
    pub attack_count: usize,
    pub time_domain_window_cluster_count: usize,
    pub spectral_window_cluster_count: usize,
    pub c5_class_counts: RenderedMultimodalClassCountsV1,
    pub shared_sample_count: RenderedMultimodalSampleCountSummaryV1,
    pub time_domain_only_sample_count: RenderedMultimodalSampleCountSummaryV1,
    pub spectral_only_sample_count: RenderedMultimodalSampleCountSummaryV1,
    pub subjects: Vec<RenderedMultimodalSubjectSummaryV1>,
    pub clusters: Vec<RenderedMultimodalWindowClusterV1>,
    pub samples: Vec<RenderedMultimodalPanelSampleV1>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RenderedMultimodalAttackPanelErrorV1 {
    EmptyPanel,
    EmptySubjectId,
    DuplicateAttackIdentity,
    InvalidEvidence(RenderedMultimodalAttackEvidenceErrorV1),
    MixedSampleRate,
    MixedTimeDomainWindowGeometry,
    MixedTimeDomainGateConfig,
    MixedTransientConfig,
    MixedSpectralConfig,
}

pub fn summarize_rendered_multimodal_attack_panel(
    samples: &[RenderedMultimodalPanelSampleV1],
) -> Result<RenderedMultimodalAttackPanelV1, RenderedMultimodalAttackPanelErrorV1> {
    let Some(first) = samples.first() else {
        return Err(RenderedMultimodalAttackPanelErrorV1::EmptyPanel);
    };
    verify_rendered_multimodal_attack_evidence(&first.evidence)
        .map_err(RenderedMultimodalAttackPanelErrorV1::InvalidEvidence)?;

    let sample_rate = first.evidence.sample_rate;
    let time_domain_pre_samples = first.evidence.time_domain.pre_samples;
    let time_domain_post_samples = first.evidence.time_domain.post_samples;
    let localized_config = first.evidence.time_domain.localized_difference.config;
    let transient_config = first.evidence.time_domain.candidate_transient.config;
    let spectral_config = first.evidence.spectral.config;

    let mut identities = BTreeSet::new();
    let mut by_subject: BTreeMap<String, Vec<&RenderedMultimodalPanelSampleV1>> = BTreeMap::new();
    let mut class_counts = RenderedMultimodalClassCountsV1 {
        localized_difference_and_candidate_transient: 0,
        localized_difference_only: 0,
        candidate_transient_only: 0,
        neither: 0,
    };
    let mut shared_counts = Vec::with_capacity(samples.len());
    let mut time_only_counts = Vec::with_capacity(samples.len());
    let mut spectral_only_counts = Vec::with_capacity(samples.len());

    for sample in samples {
        if sample.subject_id.trim().is_empty() {
            return Err(RenderedMultimodalAttackPanelErrorV1::EmptySubjectId);
        }
        if !identities.insert((sample.subject_id.clone(), sample.attack_ordinal)) {
            return Err(RenderedMultimodalAttackPanelErrorV1::DuplicateAttackIdentity);
        }
        verify_rendered_multimodal_attack_evidence(&sample.evidence)
            .map_err(RenderedMultimodalAttackPanelErrorV1::InvalidEvidence)?;
        if sample.evidence.sample_rate != sample_rate {
            return Err(RenderedMultimodalAttackPanelErrorV1::MixedSampleRate);
        }
        if sample.evidence.time_domain.pre_samples != time_domain_pre_samples
            || sample.evidence.time_domain.post_samples != time_domain_post_samples
        {
            return Err(RenderedMultimodalAttackPanelErrorV1::MixedTimeDomainWindowGeometry);
        }
        if sample.evidence.time_domain.localized_difference.config != localized_config {
            return Err(RenderedMultimodalAttackPanelErrorV1::MixedTimeDomainGateConfig);
        }
        if sample.evidence.time_domain.candidate_transient.config != transient_config {
            return Err(RenderedMultimodalAttackPanelErrorV1::MixedTransientConfig);
        }
        if sample.evidence.spectral.config != spectral_config {
            return Err(RenderedMultimodalAttackPanelErrorV1::MixedSpectralConfig);
        }

        match sample.evidence.time_domain.class {
            RenderedAttackCrosscheckClassV1::LocalizedDifferenceAndCandidateTransient => {
                class_counts.localized_difference_and_candidate_transient += 1
            }
            RenderedAttackCrosscheckClassV1::LocalizedDifferenceOnly => {
                class_counts.localized_difference_only += 1
            }
            RenderedAttackCrosscheckClassV1::CandidateTransientOnly => {
                class_counts.candidate_transient_only += 1
            }
            RenderedAttackCrosscheckClassV1::Neither => class_counts.neither += 1,
        }
        shared_counts.push(sample.evidence.raw_sample_overlap.shared_sample_count);
        time_only_counts.push(
            sample
                .evidence
                .raw_sample_overlap
                .time_domain_only_sample_count,
        );
        spectral_only_counts.push(
            sample
                .evidence
                .raw_sample_overlap
                .spectral_only_sample_count,
        );
        by_subject
            .entry(sample.subject_id.clone())
            .or_default()
            .push(sample);
    }

    let mut subjects = Vec::with_capacity(by_subject.len());
    let mut clusters = Vec::new();

    for (subject_id, subject_samples) in &by_subject {
        let mut time_intervals = Vec::with_capacity(subject_samples.len());
        let mut spectral_intervals = Vec::with_capacity(subject_samples.len());
        for sample in subject_samples {
            let overlap = sample.evidence.raw_sample_overlap;
            time_intervals.push((
                sample.attack_ordinal,
                overlap.time_domain_start_sample,
                overlap.time_domain_end_sample,
            ));
            spectral_intervals.push((
                sample.attack_ordinal,
                overlap.spectral_start_sample,
                overlap.spectral_end_sample,
            ));
        }

        let time_clusters = build_clusters(
            subject_id,
            RenderedMultimodalWindowViewV1::TimeDomain,
            &mut time_intervals,
        );
        let spectral_clusters = build_clusters(
            subject_id,
            RenderedMultimodalWindowViewV1::Spectral,
            &mut spectral_intervals,
        );
        subjects.push(RenderedMultimodalSubjectSummaryV1 {
            subject_id: subject_id.clone(),
            attack_count: subject_samples.len(),
            time_domain_window_cluster_count: time_clusters.len(),
            spectral_window_cluster_count: spectral_clusters.len(),
        });
        clusters.extend(time_clusters);
        clusters.extend(spectral_clusters);
    }

    let time_domain_window_cluster_count = clusters
        .iter()
        .filter(|cluster| cluster.view == RenderedMultimodalWindowViewV1::TimeDomain)
        .count();
    let spectral_window_cluster_count = clusters
        .iter()
        .filter(|cluster| cluster.view == RenderedMultimodalWindowViewV1::Spectral)
        .count();

    Ok(RenderedMultimodalAttackPanelV1 {
        panel_version: RENDERED_MULTIMODAL_ATTACK_PANEL_VERSION.into(),
        sample_rate,
        time_domain_pre_samples,
        time_domain_post_samples,
        spectral_config,
        subject_count: subjects.len(),
        attack_count: samples.len(),
        time_domain_window_cluster_count,
        spectral_window_cluster_count,
        c5_class_counts: class_counts,
        shared_sample_count: summarize_counts(&shared_counts),
        time_domain_only_sample_count: summarize_counts(&time_only_counts),
        spectral_only_sample_count: summarize_counts(&spectral_only_counts),
        subjects,
        clusters,
        samples: samples.to_vec(),
    })
}

fn build_clusters(
    subject_id: &str,
    view: RenderedMultimodalWindowViewV1,
    intervals: &mut [(usize, usize, usize)],
) -> Vec<RenderedMultimodalWindowClusterV1> {
    intervals.sort_by(|left, right| {
        left.1
            .cmp(&right.1)
            .then_with(|| left.2.cmp(&right.2))
            .then_with(|| left.0.cmp(&right.0))
    });
    let mut clusters = Vec::new();
    let mut current_start = 0usize;
    let mut current_end = 0usize;
    let mut members = Vec::new();

    for &(attack_ordinal, start, end) in intervals.iter() {
        if members.is_empty() {
            current_start = start;
            current_end = end;
            members.push(attack_ordinal);
            continue;
        }
        if start < current_end {
            current_end = current_end.max(end);
            members.push(attack_ordinal);
        } else {
            let cluster_ordinal = clusters.len();
            clusters.push(RenderedMultimodalWindowClusterV1 {
                view,
                subject_id: subject_id.into(),
                cluster_ordinal,
                start_sample: current_start,
                end_sample: current_end,
                member_attack_ordinals: std::mem::take(&mut members),
            });
            current_start = start;
            current_end = end;
            members.push(attack_ordinal);
        }
    }

    if !members.is_empty() {
        let cluster_ordinal = clusters.len();
        clusters.push(RenderedMultimodalWindowClusterV1 {
            view,
            subject_id: subject_id.into(),
            cluster_ordinal,
            start_sample: current_start,
            end_sample: current_end,
            member_attack_ordinals: members,
        });
    }
    clusters
}

fn summarize_counts(values: &[usize]) -> RenderedMultimodalSampleCountSummaryV1 {
    RenderedMultimodalSampleCountSummaryV1 {
        minimum: values.iter().copied().min().unwrap_or(0),
        maximum: values.iter().copied().max().unwrap_or(0),
        mean: values.iter().sum::<usize>() as f64 / values.len() as f64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evidence_digest::{
        rendered_attack_crosscheck::evaluate_rendered_attack_crosscheck,
        rendered_attack_evidence::measure_rendered_attack,
        rendered_attack_gate::RenderedAttackGateConfigV1,
        rendered_multimodal_attack_evidence::bind_rendered_multimodal_attack_evidence,
        rendered_spectral_window_evidence::{
            RenderedSpectralWindowConfigV1, measure_rendered_spectral_window,
        },
        rendered_transient_contrast::RenderedTransientContrastConfigV1,
    };

    const SAMPLE_RATE: u32 = 8_192;

    fn spectral_config() -> RenderedSpectralWindowConfigV1 {
        RenderedSpectralWindowConfigV1 {
            fft_size: 256,
            mel_bands: 24,
            f_min_hz: 20.0,
            f_max_hz: 3_500.0,
        }
    }

    fn sample(subject: &str, ordinal: usize, center_sample: usize) -> RenderedMultimodalPanelSampleV1 {
        let audio = vec![[0.0, 0.0]; SAMPLE_RATE as usize];
        let attack_time = center_sample as f32 / SAMPLE_RATE as f32;
        let time_evidence = measure_rendered_attack(
            &audio,
            &audio,
            SAMPLE_RATE,
            attack_time,
            0.010,
            0.020,
        )
        .unwrap();
        let time_domain = evaluate_rendered_attack_crosscheck(
            &time_evidence,
            RenderedAttackGateConfigV1::default(),
            RenderedTransientContrastConfigV1::default(),
        )
        .unwrap();
        let spectral = measure_rendered_spectral_window(
            &audio,
            &audio,
            SAMPLE_RATE,
            attack_time,
            spectral_config(),
        )
        .unwrap();
        let evidence = bind_rendered_multimodal_attack_evidence(&time_domain, &spectral).unwrap();
        RenderedMultimodalPanelSampleV1 {
            subject_id: subject.into(),
            attack_ordinal: ordinal,
            evidence,
        }
    }

    #[test]
    fn time_and_spectral_views_keep_distinct_overlap_topologies() {
        let panel = summarize_rendered_multimodal_attack_panel(&[
            sample("seed-1", 0, 4_096),
            // 256 samples later: C5's 246-sample window is disjoint, while
            // C6's 512-sample spectral window still overlaps directly.
            sample("seed-1", 1, 4_352),
        ])
        .unwrap();

        assert_eq!(panel.subject_count, 1);
        assert_eq!(panel.attack_count, 2);
        assert_eq!(panel.time_domain_window_cluster_count, 2);
        assert_eq!(panel.spectral_window_cluster_count, 1);
        assert_eq!(panel.c5_class_counts.neither, 2);
        assert!(panel.shared_sample_count.minimum > 0);
    }

    #[test]
    fn duplicate_identity_and_forged_nested_evidence_fail_closed() {
        let one = sample("seed-1", 0, 4_096);
        assert_eq!(
            summarize_rendered_multimodal_attack_panel(&[one.clone(), one]),
            Err(RenderedMultimodalAttackPanelErrorV1::DuplicateAttackIdentity)
        );

        let mut forged = sample("seed-1", 0, 4_096);
        forged.evidence.raw_sample_overlap.shared_sample_count += 1;
        assert!(matches!(
            summarize_rendered_multimodal_attack_panel(&[forged]),
            Err(RenderedMultimodalAttackPanelErrorV1::InvalidEvidence(_))
        ));
    }
}
