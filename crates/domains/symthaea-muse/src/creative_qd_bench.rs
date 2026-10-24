// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic, fixture-backed contract and descriptor-diversity benchmark.
//!
//! This is a benchmark harness, not production candidate selection. It separates
//! contract acceptance from descriptor-space coverage and never converts either
//! into a universal artistic-quality score.

use crate::{
    Note,
    creative_contract::{
        CreativeCandidateAuditV1, CreativeIntentContractV1, audit_candidate,
    },
    creative_temporal_support::{
        CreativeTemporalSupportError, CreativeTemporalSupportV1, measure_temporal_support,
    },
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt,
};

/// Schema version of the frozen input brief/candidate corpus.
pub const CREATIVE_QD_CORPUS_SCHEMA_VERSION: u32 = 1;
/// Schema version of the emitted benchmark receipt.
pub const CREATIVE_QD_BENCH_SCHEMA_VERSION: u32 = 2;
pub const CREATIVE_QD_DESCRIPTOR_GRID_CELLS: usize = 6 * 5 * 4 * 5;

/// Frozen input corpus for the first contract/descriptor benchmark.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CreativeIntentBenchmarkCorpusV1 {
    pub schema_version: u32,
    pub corpus_id: String,
    pub intents: Vec<CreativeIntentBenchmarkIntentV1>,
}

/// A task family with one declared mechanical contract and a fixed candidate budget.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CreativeIntentBenchmarkIntentV1 {
    pub intent_id: String,
    pub family: String,
    pub contract: CreativeIntentContractV1,
    pub candidates: Vec<CreativeIntentBenchmarkCandidateV1>,
}

/// One synthetic candidate. A note is the four-number tuple [frequency, onset, duration, velocity].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CreativeIntentBenchmarkCandidateV1 {
    pub candidate_id: String,
    pub expected_accepted: bool,
    pub duration_secs: f32,
    pub notes: Vec<[f32; 4]>,
}

/// Stable, inspectable feature bins for the simple QD archive baseline.
///
/// Bins are engineering descriptors, not quality judgments or validated perceptual axes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize,
)]
#[serde(deny_unknown_fields)]
pub struct CreativeArchiveDescriptorCellV1 {
    /// 0 = no valid pitches; 1–4 = exact count; 5 = five or more.
    pub pitch_diversity_bin: usize,
    /// Normalized active-window coverage ratio, rounded into bins 0–4.
    pub active_window_coverage_bin: usize,
    /// Distinct onset events per second: <=0.5, <=1.5, <=3, >3.
    pub onset_density_bin: usize,
    /// Inter-onset interval CV: <=0.1, <=0.4, <=0.8, <=1.5, >1.5.
    pub onset_interval_cv_bin: usize,
}

/// An occupied descriptor cell and its deterministic representative candidate.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CreativeArchiveEntryV1 {
    pub cell: CreativeArchiveDescriptorCellV1,
    pub representative_candidate_id: String,
}

/// One candidate's full contract receipt.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CreativeIntentBenchmarkCandidateReceiptV1 {
    pub candidate_id: String,
    pub expected_accepted: bool,
    pub observed_accepted: bool,
    pub descriptor_cell: Option<CreativeArchiveDescriptorCellV1>,
    pub retained_in_archive: bool,
    pub audit: CreativeCandidateAuditV1,
    /// Descriptive temporal evidence. Not currently used for acceptance or archive bins.
    pub temporal_support: CreativeTemporalSupportOutcomeV1,
}

/// Temporal support measurement outcome emitted with every candidate receipt.
///
/// Malformed note input stays visible as an explicit measurement error rather than
/// being omitted or presented as zero coverage.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum CreativeTemporalSupportOutcomeV1 {
    Measured {
        metrics: CreativeTemporalSupportV1,
    },
    Unavailable {
        error: CreativeTemporalSupportError,
    },
}

/// Report for one intent family; hard acceptance and archive coverage stay separate.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CreativeIntentBenchmarkIntentReceiptV1 {
    pub intent_id: String,
    pub family: String,
    pub candidate_budget: usize,
    pub accepted_candidate_count: usize,
    pub archive_entry_count: usize,
    pub accepted_candidates_collapsed_by_descriptor_bins: usize,
    pub descriptor_grid_occupancy_ratio: f64,
    pub archive: Vec<CreativeArchiveEntryV1>,
    pub candidates: Vec<CreativeIntentBenchmarkCandidateReceiptV1>,
}

/// Deterministic receipt bound to the exact corpus bytes and optional source commit.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CreativeIntentBenchmarkReceiptV1 {
    pub schema_version: u32,
    pub corpus_id: String,
    pub corpus_sha256: String,
    pub source_commit: Option<String>,
    pub candidate_budget_per_intent: Vec<usize>,
    pub total_candidate_count: usize,
    pub accepted_candidate_count: usize,
    pub rejected_candidate_count: usize,
    pub archive_retained_candidate_count: usize,
    pub accepted_candidates_collapsed_by_descriptor_bins: usize,
    pub expected_outcome_mismatches: Vec<String>,
    pub intents: Vec<CreativeIntentBenchmarkIntentReceiptV1>,
}

/// Corpus schema or integrity failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CreativeBenchmarkError(String);

impl fmt::Display for CreativeBenchmarkError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for CreativeBenchmarkError {}

impl CreativeBenchmarkError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

/// Run the frozen corpus without RNG, model inference, network access, or wall-clock time.
///
/// Invalid candidate notes are evaluated by the same fail-closed contract audit as callers
/// of the library. Corpus-level schema faults return an error; expected-result mismatches are
/// retained in the receipt so the CLI can print evidence before exiting non-zero.
pub fn run_corpus_bytes(
    corpus_bytes: &[u8],
    source_commit: Option<&str>,
) -> Result<CreativeIntentBenchmarkReceiptV1, CreativeBenchmarkError> {
    let corpus: CreativeIntentBenchmarkCorpusV1 = serde_json::from_slice(corpus_bytes)
        .map_err(|error| CreativeBenchmarkError::new(format!("invalid corpus JSON: {error}")))?;

    if corpus.schema_version != CREATIVE_QD_CORPUS_SCHEMA_VERSION {
        return Err(CreativeBenchmarkError::new(format!(
            "unsupported corpus schema version {}",
            corpus.schema_version
        )));
    }
    if corpus.corpus_id.trim().is_empty() || corpus.intents.is_empty() {
        return Err(CreativeBenchmarkError::new(
            "corpus ID and at least one intent are required",
        ));
    }

    let mut seen_intents = BTreeSet::new();
    let mut mismatches = Vec::new();
    let mut intent_reports = Vec::with_capacity(corpus.intents.len());
    let mut per_intent_budgets = Vec::with_capacity(corpus.intents.len());
    let mut total_candidate_count = 0usize;
    let mut accepted_candidate_count = 0usize;
    let mut archive_retained_candidate_count = 0usize;
    let mut descriptor_collisions = 0usize;

    for intent in &corpus.intents {
        if intent.intent_id.trim().is_empty()
            || intent.family.trim().is_empty()
            || !seen_intents.insert(intent.intent_id.as_str())
            || intent.candidates.is_empty()
        {
            return Err(CreativeBenchmarkError::new(format!(
                "invalid or duplicate intent identity: {:?}",
                intent.intent_id
            )));
        }
        if !intent.contract.is_valid() {
            return Err(CreativeBenchmarkError::new(format!(
                "intent {} contains an invalid contract",
                intent.intent_id
            )));
        }

        let mut seen_candidates = BTreeSet::new();
        let mut candidate_audits = Vec::with_capacity(intent.candidates.len());
        let mut archive: BTreeMap<CreativeArchiveDescriptorCellV1, String> = BTreeMap::new();

        for candidate in &intent.candidates {
            if candidate.candidate_id.trim().is_empty()
                || !seen_candidates.insert(candidate.candidate_id.as_str())
            {
                return Err(CreativeBenchmarkError::new(format!(
                    "intent {} has an empty or duplicate candidate ID: {:?}",
                    intent.intent_id, candidate.candidate_id
                )));
            }

            let notes: Vec<Note> = candidate
                .notes
                .iter()
                .map(|note| Note {
                    frequency: note[0],
                    start_time: note[1],
                    duration: note[2],
                    velocity: note[3],
                })
                .collect();
            let audit = audit_candidate(&notes, candidate.duration_secs, &intent.contract);
            let observed_accepted = audit.accepted;
            let temporal_support = match measure_temporal_support(
                &notes,
                candidate.duration_secs,
                intent.contract.window_count,
            ) {
                Ok(metrics) => CreativeTemporalSupportOutcomeV1::Measured { metrics },
                Err(error) => CreativeTemporalSupportOutcomeV1::Unavailable { error },
            };

            if observed_accepted != candidate.expected_accepted {
                mismatches.push(format!(
                    "{}/{} expected accepted={} but observed accepted={}",
                    intent.intent_id,
                    candidate.candidate_id,
                    candidate.expected_accepted,
                    observed_accepted
                ));
            }

            let cell = if observed_accepted {
                Some(descriptor_cell(&audit, candidate.duration_secs))
            } else {
                None
            };
            if let Some(cell) = cell {
                match archive.get_mut(&cell) {
                    Some(representative) if candidate.candidate_id < *representative => {
                        *representative = candidate.candidate_id.clone();
                    }
                    Some(_) => {}
                    None => {
                        archive.insert(cell, candidate.candidate_id.clone());
                    }
                }
            }

            candidate_audits.push((
                candidate.candidate_id.clone(),
                candidate.expected_accepted,
                observed_accepted,
                cell,
                audit,
                temporal_support,
            ));
        }

        let retained_ids: BTreeSet<&str> = archive.values().map(String::as_str).collect();
        let archive_entries: Vec<CreativeArchiveEntryV1> = archive
            .iter()
            .map(|(cell, candidate_id)| CreativeArchiveEntryV1 {
                cell: *cell,
                representative_candidate_id: candidate_id.clone(),
            })
            .collect();
        let candidate_receipts: Vec<CreativeIntentBenchmarkCandidateReceiptV1> =
            candidate_audits
                .into_iter()
                .map(
                    |(
                        candidate_id,
                        expected_accepted,
                        observed_accepted,
                        descriptor_cell,
                        audit,
                        temporal_support,
                    )| {
                        CreativeIntentBenchmarkCandidateReceiptV1 {
                            retained_in_archive: retained_ids.contains(candidate_id.as_str()),
                            candidate_id,
                            expected_accepted,
                            observed_accepted,
                            descriptor_cell,
                            audit,
                            temporal_support,
                        }
                    },
                )
                .collect();

        let accepted = candidate_receipts
            .iter()
            .filter(|candidate| candidate.observed_accepted)
            .count();
        let archive_count = archive_entries.len();
        let collisions = accepted.saturating_sub(archive_count);
        total_candidate_count += candidate_receipts.len();
        accepted_candidate_count += accepted;
        archive_retained_candidate_count += archive_count;
        descriptor_collisions += collisions;
        per_intent_budgets.push(candidate_receipts.len());

        intent_reports.push(CreativeIntentBenchmarkIntentReceiptV1 {
            intent_id: intent.intent_id.clone(),
            family: intent.family.clone(),
            candidate_budget: candidate_receipts.len(),
            accepted_candidate_count: accepted,
            archive_entry_count: archive_count,
            accepted_candidates_collapsed_by_descriptor_bins: collisions,
            descriptor_grid_occupancy_ratio: archive_count as f64
                / CREATIVE_QD_DESCRIPTOR_GRID_CELLS as f64,
            archive: archive_entries,
            candidates: candidate_receipts,
        });
    }

    let corpus_sha256 = sha256_hex(corpus_bytes);
    Ok(CreativeIntentBenchmarkReceiptV1 {
        schema_version: CREATIVE_QD_BENCH_SCHEMA_VERSION,
        corpus_id: corpus.corpus_id,
        corpus_sha256,
        source_commit: source_commit.map(str::to_owned),
        candidate_budget_per_intent: per_intent_budgets,
        total_candidate_count,
        accepted_candidate_count,
        rejected_candidate_count: total_candidate_count - accepted_candidate_count,
        archive_retained_candidate_count,
        accepted_candidates_collapsed_by_descriptor_bins: descriptor_collisions,
        expected_outcome_mismatches: mismatches,
        intents: intent_reports,
    })
}

fn descriptor_cell(
    audit: &CreativeCandidateAuditV1,
    duration_secs: f32,
) -> CreativeArchiveDescriptorCellV1 {
    let density_per_second = audit.distinct_onset_count as f64 / f64::from(duration_secs);
    let onset_density_bin = if density_per_second <= 0.5 {
        0
    } else if density_per_second <= 1.5 {
        1
    } else if density_per_second <= 3.0 {
        2
    } else {
        3
    };
    let onset_interval_cv_bin = if audit.onset_interval_cv <= 0.1 {
        0
    } else if audit.onset_interval_cv <= 0.4 {
        1
    } else if audit.onset_interval_cv <= 0.8 {
        2
    } else if audit.onset_interval_cv <= 1.5 {
        3
    } else {
        4
    };

    CreativeArchiveDescriptorCellV1 {
        pitch_diversity_bin: audit.distinct_pitch_count.min(5),
        active_window_coverage_bin: (audit.active_window_ratio * 4.0)
            .round()
            .clamp(0.0, 4.0) as usize,
        onset_density_bin,
        onset_interval_cv_bin,
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const FROZEN_CORPUS: &[u8] = include_bytes!("../fixtures/creative_intent_contract_v1.json");

    #[test]
    fn frozen_corpus_matches_all_preregistered_outcomes() {
        let report = run_corpus_bytes(FROZEN_CORPUS, Some("test-source-sha"))
            .expect("frozen creative intent corpus should parse and validate");
        assert_eq!(report.schema_version, CREATIVE_QD_BENCH_SCHEMA_VERSION);
        assert_eq!(report.total_candidate_count, 17);
        assert_eq!(report.accepted_candidate_count, 8);
        assert_eq!(report.rejected_candidate_count, 9);
        assert!(
            report.expected_outcome_mismatches.is_empty(),
            "{:?}",
            report.expected_outcome_mismatches
        );
        assert_eq!(report.corpus_sha256.len(), 64);
        assert_eq!(report.source_commit.as_deref(), Some("test-source-sha"));
    }

    #[test]
    fn archive_preserves_density_and_interval_variants_without_scalar_ranking() {
        let report = run_corpus_bytes(FROZEN_CORPUS, None).expect("corpus should be valid");
        let ambient = report
            .intents
            .iter()
            .find(|intent| intent.intent_id == "ambient-drone")
            .unwrap();
        assert_eq!(
            ambient.archive_entry_count, 2,
            "steady and dense textures occupy different cells"
        );
        let groove = report
            .intents
            .iter()
            .find(|intent| intent.intent_id == "rhythmic-groove")
            .unwrap();
        assert_eq!(
            groove.archive_entry_count, 2,
            "regular and irregular onset intervals occupy different cells"
        );
        assert_eq!(groove.accepted_candidates_collapsed_by_descriptor_bins, 0);
        assert_eq!(
            report.accepted_candidates_collapsed_by_descriptor_bins,
            1,
            "the duplicate ambient candidate should collide with one archive cell"
        );

        let steady_a = ambient
            .candidates
            .iter()
            .find(|candidate| candidate.candidate_id == "steady-drone-a")
            .unwrap();
        let steady_z = ambient
            .candidates
            .iter()
            .find(|candidate| candidate.candidate_id == "steady-drone-z")
            .unwrap();
        assert!(steady_a.retained_in_archive);
        assert!(!steady_z.retained_in_archive);
        assert!(matches!(
            &steady_a.temporal_support,
            CreativeTemporalSupportOutcomeV1::Measured { .. }
        ));
        if let CreativeTemporalSupportOutcomeV1::Measured { metrics } =
            &steady_a.temporal_support
        {
            assert!((metrics.occupied_time_ratio - 0.4).abs() < 1.0e-6);
            assert_eq!(metrics.onset_window_coverage_ratio, 1.0);
            assert_eq!(metrics.sustain_window_coverage_ratio, 1.0);
        } else {
            panic!("valid ambient fixture should have temporal support metrics");
        }

        let dense_drone = ambient
            .candidates
            .iter()
            .find(|candidate| candidate.candidate_id == "dense-drone")
            .unwrap();
        if let CreativeTemporalSupportOutcomeV1::Measured { metrics } =
            &dense_drone.temporal_support
        {
            assert_eq!(metrics.onset_window_coverage_ratio, 1.0);
            assert_eq!(metrics.sustain_window_coverage_ratio, 1.0);
            assert!((metrics.occupied_time_ratio - 0.8).abs() < 1.0e-6);
            assert!((metrics.maximum_silent_gap_secs - 0.05).abs() < 1.0e-6);
        } else {
            panic!("valid dense-drone fixture should have temporal support metrics");
        }
    }

    #[test]
    fn invalid_note_has_an_explicit_temporal_measurement_failure() {
        let report = run_corpus_bytes(FROZEN_CORPUS, None).expect("corpus should be valid");
        let melody = report
            .intents
            .iter()
            .find(|intent| intent.intent_id == "lyrical-melody")
            .unwrap();
        let invalid = melody
            .candidates
            .iter()
            .find(|candidate| candidate.candidate_id == "invalid-note-in-melody")
            .unwrap();
        assert!(matches!(
            &invalid.temporal_support,
            CreativeTemporalSupportOutcomeV1::Unavailable {
                error: CreativeTemporalSupportError::InvalidNote { index: 3 }
            }
        ));
    }

    #[test]
    fn zero_velocity_candidate_fails_closed_and_has_no_archive_cell() {
        let report = run_corpus_bytes(FROZEN_CORPUS, None).expect("corpus should be valid");
        let melody = report
            .intents
            .iter()
            .find(|intent| intent.intent_id == "lyrical-melody")
            .unwrap();
        let silent = melody
            .candidates
            .iter()
            .find(|candidate| candidate.candidate_id == "silent-note-count-trap")
            .unwrap();

        assert!(!silent.observed_accepted);
        assert!(silent.descriptor_cell.is_none());
        assert!(matches!(
            &silent.temporal_support,
            CreativeTemporalSupportOutcomeV1::Unavailable {
                error: CreativeTemporalSupportError::InvalidNote { index: 0 }
            }
        ));
    }

    #[test]
    fn archive_representative_is_independent_of_candidate_order() {
        let mut corpus: CreativeIntentBenchmarkCorpusV1 =
            serde_json::from_slice(FROZEN_CORPUS).expect("fixture parses");
        let before = run_corpus_bytes(FROZEN_CORPUS, None).expect("fixture validates");
        for intent in &mut corpus.intents {
            intent.candidates.reverse();
        }
        let reversed_bytes = serde_json::to_vec(&corpus).expect("reversed corpus serializes");
        let after = run_corpus_bytes(&reversed_bytes, None).expect("reversed fixture validates");
        for before_intent in &before.intents {
            let after_intent = after
                .intents
                .iter()
                .find(|item| item.intent_id == before_intent.intent_id)
                .unwrap();
            assert_eq!(before_intent.archive, after_intent.archive);
            let before_support: BTreeMap<&str, &CreativeTemporalSupportOutcomeV1> = before_intent
                .candidates
                .iter()
                .map(|candidate| (candidate.candidate_id.as_str(), &candidate.temporal_support))
                .collect();
            let after_support: BTreeMap<&str, &CreativeTemporalSupportOutcomeV1> = after_intent
                .candidates
                .iter()
                .map(|candidate| (candidate.candidate_id.as_str(), &candidate.temporal_support))
                .collect();
            assert_eq!(before_support, after_support);
        }
    }

    #[test]
    fn unknown_corpus_fields_fail_closed() {
        let altered = String::from_utf8_lossy(FROZEN_CORPUS)
            .replacen(
                "\"schema_version\": 1,",
                "\"schema_version\": 1,\n  \"unexpected\": true,",
                1,
            );
        assert!(run_corpus_bytes(altered.as_bytes(), None).is_err());
    }
}
