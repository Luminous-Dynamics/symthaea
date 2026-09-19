// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Detect direct matched-window sample reuse among rendered-attack crosschecks.
//!
//! A performed-note attack is not automatically an independent acoustic
//! observation. Rolled chord tones or nearby events may have overlapping
//! pre/post measurement windows and therefore reuse many of the same audio
//! samples. This module groups such observations into exact window-overlap
//! dependency clusters while preserving every note-level result.
//!
//! Non-overlapping clusters are **not** claimed statistically independent:
//! reverb, phrase context, renderer state, and other long-range dependencies
//! may still correlate them. V1 establishes only direct sample-window overlap.

use crate::evidence_digest::rendered_attack_crosscheck::{
    RenderedAttackCrosscheckClassV1, RenderedAttackCrosscheckErrorV1,
    RenderedAttackCrosscheckV1, verify_rendered_attack_crosscheck,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const ACOUSTIC_WINDOW_DEPENDENCY_VERSION: &str = "acoustic-window-dependency-v1";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AcousticWindowDependencySampleV1 {
    pub subject_id: String,
    pub attack_ordinal: usize,
    pub crosscheck: RenderedAttackCrosscheckV1,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AcousticWindowDependencyMemberV1 {
    pub attack_ordinal: usize,
    pub source_evidence_sha256: String,
    pub center_sample: usize,
    pub class: RenderedAttackCrosscheckClassV1,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AcousticWindowDependencyClusterV1 {
    pub subject_id: String,
    pub cluster_ordinal: usize,
    pub sample_rate: u32,
    /// Half-open union of all overlapping source evidence windows.
    pub start_sample: usize,
    pub end_sample: usize,
    pub member_count: usize,
    pub members: Vec<AcousticWindowDependencyMemberV1>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AcousticWindowDependencySubjectV1 {
    pub subject_id: String,
    pub sample_rate: u32,
    pub attack_count: usize,
    pub window_overlap_cluster_count: usize,
    pub singleton_cluster_count: usize,
    pub multi_member_cluster_count: usize,
    /// Number of note-level observations participating in a cluster of size > 1.
    pub attacks_in_overlapping_clusters: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AcousticWindowDependencyPanelV1 {
    pub dependency_version: String,
    pub subject_count: usize,
    pub attack_count: usize,
    pub window_overlap_cluster_count: usize,
    pub singleton_cluster_count: usize,
    pub multi_member_cluster_count: usize,
    pub attacks_in_overlapping_clusters: usize,
    pub subjects: Vec<AcousticWindowDependencySubjectV1>,
    pub clusters: Vec<AcousticWindowDependencyClusterV1>,
    /// Original note-level observations are retained; clustering does not erase
    /// or promote any acoustic verdict.
    pub samples: Vec<AcousticWindowDependencySampleV1>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AcousticWindowDependencyErrorV1 {
    EmptyPanel,
    EmptySubjectId,
    DuplicateAttackIdentity,
    InvalidCrosscheck(RenderedAttackCrosscheckErrorV1),
    InvalidWindowGeometry,
    MixedSampleRateWithinSubject,
}

/// Group exact matched-window intervals that reuse at least one audio sample.
///
/// Clustering is performed independently within each subject. Half-open
/// intervals `[start, end)` that merely touch (`left.end == right.start`) do
/// not overlap and therefore remain separate clusters.
pub fn summarize_acoustic_window_dependencies(
    samples: &[AcousticWindowDependencySampleV1],
) -> Result<AcousticWindowDependencyPanelV1, AcousticWindowDependencyErrorV1> {
    if samples.is_empty() {
        return Err(AcousticWindowDependencyErrorV1::EmptyPanel);
    }

    let mut identities = BTreeSet::new();
    let mut by_subject: BTreeMap<String, Vec<&AcousticWindowDependencySampleV1>> = BTreeMap::new();

    for sample in samples {
        if sample.subject_id.trim().is_empty() {
            return Err(AcousticWindowDependencyErrorV1::EmptySubjectId);
        }
        if !identities.insert((sample.subject_id.clone(), sample.attack_ordinal)) {
            return Err(AcousticWindowDependencyErrorV1::DuplicateAttackIdentity);
        }
        verify_rendered_attack_crosscheck(&sample.crosscheck)
            .map_err(AcousticWindowDependencyErrorV1::InvalidCrosscheck)?;
        let Some((start, end)) = sample.crosscheck.sample_interval() else {
            return Err(AcousticWindowDependencyErrorV1::InvalidWindowGeometry);
        };
        if start >= end {
            return Err(AcousticWindowDependencyErrorV1::InvalidWindowGeometry);
        }
        by_subject
            .entry(sample.subject_id.clone())
            .or_default()
            .push(sample);
    }

    let mut subjects = Vec::with_capacity(by_subject.len());
    let mut clusters = Vec::new();

    for (subject_id, mut subject_samples) in by_subject {
        let sample_rate = subject_samples[0].crosscheck.sample_rate;
        if subject_samples
            .iter()
            .any(|sample| sample.crosscheck.sample_rate != sample_rate)
        {
            return Err(AcousticWindowDependencyErrorV1::MixedSampleRateWithinSubject);
        }

        subject_samples.sort_by(|left, right| {
            let (left_start, left_end) = left.crosscheck.sample_interval().unwrap();
            let (right_start, right_end) = right.crosscheck.sample_interval().unwrap();
            left_start
                .cmp(&right_start)
                .then_with(|| left_end.cmp(&right_end))
                .then_with(|| left.attack_ordinal.cmp(&right.attack_ordinal))
        });

        let cluster_start_index = clusters.len();
        let mut current_start = 0usize;
        let mut current_end = 0usize;
        let mut current_members: Vec<AcousticWindowDependencyMemberV1> = Vec::new();
        let mut cluster_ordinal = 0usize;

        for sample in subject_samples {
            let (start, end) = sample.crosscheck.sample_interval().unwrap();
            let member = AcousticWindowDependencyMemberV1 {
                attack_ordinal: sample.attack_ordinal,
                source_evidence_sha256: sample.crosscheck.source_evidence_sha256.clone(),
                center_sample: sample.crosscheck.center_sample,
                class: sample.crosscheck.class,
            };

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
                clusters.push(AcousticWindowDependencyClusterV1 {
                    subject_id: subject_id.clone(),
                    cluster_ordinal,
                    sample_rate,
                    start_sample: current_start,
                    end_sample: current_end,
                    member_count: current_members.len(),
                    members: std::mem::take(&mut current_members),
                });
                cluster_ordinal += 1;
                current_start = start;
                current_end = end;
                current_members.push(member);
            }
        }

        if !current_members.is_empty() {
            clusters.push(AcousticWindowDependencyClusterV1 {
                subject_id: subject_id.clone(),
                cluster_ordinal,
                sample_rate,
                start_sample: current_start,
                end_sample: current_end,
                member_count: current_members.len(),
                members: current_members,
            });
        }

        let subject_clusters = &clusters[cluster_start_index..];
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

        subjects.push(AcousticWindowDependencySubjectV1 {
            subject_id,
            sample_rate,
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

    Ok(AcousticWindowDependencyPanelV1 {
        dependency_version: ACOUSTIC_WINDOW_DEPENDENCY_VERSION.into(),
        subject_count: subjects.len(),
        attack_count: samples.len(),
        window_overlap_cluster_count: clusters.len(),
        singleton_cluster_count,
        multi_member_cluster_count,
        attacks_in_overlapping_clusters,
        subjects,
        clusters,
        samples: samples.to_vec(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evidence_digest::{
        rendered_attack_crosscheck::evaluate_rendered_attack_crosscheck,
        rendered_attack_evidence::measure_rendered_attack,
        rendered_attack_gate::RenderedAttackGateConfigV1,
        rendered_transient_contrast::RenderedTransientContrastConfigV1,
    };

    fn sample(
        subject: &str,
        ordinal: usize,
        sample_rate: u32,
        attack_time: f32,
        pre_window: f32,
        post_window: f32,
    ) -> AcousticWindowDependencySampleV1 {
        let baseline = vec![[0.0, 0.0]; sample_rate as usize * 4];
        let evidence = measure_rendered_attack(
            &baseline,
            &baseline,
            sample_rate,
            attack_time,
            pre_window,
            post_window,
        )
        .unwrap();
        let crosscheck = evaluate_rendered_attack_crosscheck(
            &evidence,
            RenderedAttackGateConfigV1::default(),
            RenderedTransientContrastConfigV1::default(),
        )
        .unwrap();
        AcousticWindowDependencySampleV1 {
            subject_id: subject.into(),
            attack_ordinal: ordinal,
            crosscheck,
        }
    }

    #[test]
    fn overlapping_windows_cluster_transitively_without_erasing_members() {
        let panel = summarize_acoustic_window_dependencies(&[
            sample("seed-1", 0, 1_000, 1.0, 0.1, 0.15), // [900, 1150)
            sample("seed-1", 1, 1_000, 1.15, 0.1, 0.15), // [1050, 1300)
            sample("seed-1", 2, 1_000, 1.35, 0.1, 0.15), // [1250, 1500)
            sample("seed-1", 3, 1_000, 2.0, 0.1, 0.1), // [1900, 2100)
            sample("seed-2", 0, 1_000, 1.0, 0.1, 0.15),
        ])
        .unwrap();

        assert_eq!(panel.subject_count, 2);
        assert_eq!(panel.attack_count, 5);
        assert_eq!(panel.window_overlap_cluster_count, 3);
        assert_eq!(panel.multi_member_cluster_count, 1);
        assert_eq!(panel.attacks_in_overlapping_clusters, 3);
        let cluster = panel
            .clusters
            .iter()
            .find(|cluster| cluster.subject_id == "seed-1" && cluster.member_count == 3)
            .unwrap();
        assert_eq!(cluster.start_sample, 900);
        assert_eq!(cluster.end_sample, 1_500);
        assert_eq!(cluster.members.len(), 3);
    }

    #[test]
    fn touching_half_open_windows_are_not_marked_as_overlapping() {
        let panel = summarize_acoustic_window_dependencies(&[
            sample("seed-1", 0, 1_000, 1.0, 0.1, 0.1), // [900, 1100)
            sample("seed-1", 1, 1_000, 1.2, 0.1, 0.1), // [1100, 1300)
        ])
        .unwrap();
        assert_eq!(panel.window_overlap_cluster_count, 2);
        assert_eq!(panel.multi_member_cluster_count, 0);
    }

    #[test]
    fn duplicate_note_level_identity_and_mixed_sample_rate_fail_closed() {
        let one = sample("seed-1", 0, 1_000, 1.0, 0.1, 0.1);
        assert_eq!(
            summarize_acoustic_window_dependencies(&[one.clone(), one]),
            Err(AcousticWindowDependencyErrorV1::DuplicateAttackIdentity)
        );

        assert_eq!(
            summarize_acoustic_window_dependencies(&[
                sample("seed-1", 0, 1_000, 1.0, 0.1, 0.1),
                sample("seed-1", 1, 2_000, 1.5, 0.1, 0.1),
            ]),
            Err(AcousticWindowDependencyErrorV1::MixedSampleRateWithinSubject)
        );
    }

    #[test]
    fn forged_crosscheck_is_rejected_before_clustering() {
        let mut forged = sample("seed-1", 0, 1_000, 1.0, 0.1, 0.1);
        forged.crosscheck.class = RenderedAttackCrosscheckClassV1::LocalizedDifferenceOnly;
        assert_eq!(
            summarize_acoustic_window_dependencies(&[forged]),
            Err(AcousticWindowDependencyErrorV1::InvalidCrosscheck(
                RenderedAttackCrosscheckErrorV1::InconsistentDerivedEvidence
            ))
        );
    }
}
