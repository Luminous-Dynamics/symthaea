//! Complete pairwise split-closure audit for the science benchmark refresh lane.
//!
//! The lower-level policy-input and lineage auditors each compare two splits.
//! This module makes the required three-role composition an invariant: one
//! development, one anchor, and one refresh partition, with all three pairs
//! audited. A clean result is limited to the supplied bytes and caller-declared
//! lineage commitments; it is not proof of semantic independence or corpus
//! non-contamination.

use crate::{
    leakage_audit::{OverlapAuditReportV1, audit_policy_input_overlap, validate_split_label},
    lineage_audit::{LineageAuditReportV1, audit_lineage_overlap},
};
use serde::Serialize;
use std::collections::BTreeMap;

const REPORT_SCHEMA: &str = "symthaea.science_bench_refresh_split_matrix_report.v1";
const REQUIRED_ROLES: [&str; 3] = ["anchor", "development", "refresh"];
const EXPECTED_PAIR_COUNT: u64 = 3;

/// Borrowed input artifact bundle. Labels are checked against each embedded
/// lineage ledger rather than trusted as caller-only report decoration.
#[derive(Debug, Clone, Copy)]
pub struct SplitArtifactRefV1<'a> {
    pub split_label: &'a str,
    pub task_set_bytes: &'a [u8],
    pub lineage_ledger_bytes: &'a [u8],
}

#[derive(Debug, Clone, Serialize)]
pub struct SplitPairAuditV1 {
    pub left_split_label: String,
    pub right_split_label: String,
    pub policy_input: OverlapAuditReportV1,
    pub lineage: LineageAuditReportV1,
    pub hard_blocker: bool,
    pub manual_review_required: bool,
    pub admission: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct SplitMatrixAuditReportV1 {
    pub schema: String,
    /// True means the supplied three-way matrix is structurally complete and
    /// every required pair was audited. It does not mean the splits are proven
    /// semantically independent.
    pub valid: bool,
    pub complete_matrix: bool,
    pub required_split_roles: Vec<String>,
    pub observed_split_labels: Vec<String>,
    pub expected_pair_count: u64,
    pub observed_pair_count: u64,
    pub hard_blocking_pair_count: u64,
    pub manual_review_pair_count: u64,
    pub shared_task_id_occurrences: u64,
    pub exact_policy_overlap_pair_count: u64,
    pub numeric_near_duplicate_candidate_pair_count: u64,
    pub shared_lineage_root_occurrences: u64,
    pub shared_lineage_task_pair_occurrences: u64,
    pub pairs: Vec<SplitPairAuditV1>,
    /// Deliberately never says "certified independent" or "contamination-free".
    pub admission: String,
    pub evidence_scope: String,
    pub lineage_attestation: String,
    pub semantic_independence: String,
    pub model_training_contamination: String,
}

fn role_of(label: &str) -> &str {
    label.split(':').next().unwrap_or_default()
}

fn add_checked(total: &mut u64, value: u64, name: &str) -> Result<(), String> {
    *total = total
        .checked_add(value)
        .ok_or_else(|| format!("{name} overflows u64"))?;
    Ok(())
}

/// Audit the complete refresh-lane split matrix.
///
/// Exactly one each of development, anchor, and refresh is required.
/// Every unordered pair is checked by both existing auditors. Any malformed,
/// duplicate, missing, unknown, or internally mislabelled partition rejects
/// the whole operation; callers cannot receive a complete-matrix report after
/// checking only a convenient subset.
pub fn audit_required_split_matrix(
    splits: &[SplitArtifactRefV1<'_>],
) -> Result<SplitMatrixAuditReportV1, String> {
    if splits.len() != REQUIRED_ROLES.len() {
        return Err(format!(
            "split matrix requires exactly {} partitions (development, anchor, refresh); observed {}",
            REQUIRED_ROLES.len(),
            splits.len()
        ));
    }

    let mut by_role = BTreeMap::<&str, SplitArtifactRefV1<'_>>::new();
    for split in splits {
        validate_split_label(split.split_label)?;
        let role = role_of(split.split_label);
        if !REQUIRED_ROLES.contains(&role) {
            return Err(format!(
                "split matrix contains unsupported role {role} in label {}",
                split.split_label
            ));
        }
        if by_role.insert(role, *split).is_some() {
            return Err(format!(
                "split matrix contains more than one partition for role {role}"
            ));
        }
    }

    for role in REQUIRED_ROLES {
        if !by_role.contains_key(role) {
            return Err(format!("split matrix is missing required role {role}"));
        }
    }

    let ordered: Vec<SplitArtifactRefV1<'_>> = REQUIRED_ROLES
        .iter()
        .map(|role| {
            by_role
                .get(role)
                .copied()
                .ok_or_else(|| format!("split matrix is missing required role {role}"))
        })
        .collect::<Result<_, _>>()?;

    let mut pairs = Vec::with_capacity(EXPECTED_PAIR_COUNT as usize);
    let mut hard_blocking_pair_count = 0_u64;
    let mut manual_review_pair_count = 0_u64;
    let mut shared_task_id_occurrences = 0_u64;
    let mut exact_policy_overlap_pair_count = 0_u64;
    let mut numeric_near_duplicate_candidate_pair_count = 0_u64;
    let mut shared_lineage_root_occurrences = 0_u64;
    let mut shared_lineage_task_pair_occurrences = 0_u64;

    for left_index in 0..ordered.len() {
        for right_index in (left_index + 1)..ordered.len() {
            let left = ordered[left_index];
            let right = ordered[right_index];

            let policy_input = audit_policy_input_overlap(
                left.split_label,
                left.task_set_bytes,
                right.split_label,
                right.task_set_bytes,
            )?;
            let lineage = audit_lineage_overlap(
                left.task_set_bytes,
                left.lineage_ledger_bytes,
                right.task_set_bytes,
                right.lineage_ledger_bytes,
            )?;

            if lineage.left_split_label != left.split_label
                || lineage.right_split_label != right.split_label
            {
                return Err(format!(
                    "lineage ledger labels do not match caller labels for {} ↔ {}",
                    left.split_label, right.split_label
                ));
            }

            let exact_overlap = policy_input.shared_task_id_count > 0
                || policy_input.exact_overlap_pair_count > 0;
            let lineage_overlap = lineage.shared_lineage_root_count > 0;
            let exposed_refresh = lineage.left_exposed_refresh_task_count > 0
                || lineage.right_exposed_refresh_task_count > 0;
            let hard_blocker = exact_overlap || lineage_overlap || exposed_refresh;
            let manual_review_required =
                policy_input.near_duplicate_candidate_pair_count > 0 && !hard_blocker;

            if hard_blocker {
                add_checked(
                    &mut hard_blocking_pair_count,
                    1,
                    "hard-blocking pair count",
                )?;
            } else if manual_review_required {
                add_checked(
                    &mut manual_review_pair_count,
                    1,
                    "manual-review pair count",
                )?;
            }

            add_checked(
                &mut shared_task_id_occurrences,
                policy_input.shared_task_id_count,
                "shared task-ID occurrences",
            )?;
            add_checked(
                &mut exact_policy_overlap_pair_count,
                policy_input.exact_overlap_pair_count,
                "exact policy-overlap pair count",
            )?;
            add_checked(
                &mut numeric_near_duplicate_candidate_pair_count,
                policy_input.near_duplicate_candidate_pair_count,
                "numeric near-duplicate candidate pair count",
            )?;
            add_checked(
                &mut shared_lineage_root_occurrences,
                lineage.shared_lineage_root_count,
                "shared-lineage root occurrences",
            )?;
            add_checked(
                &mut shared_lineage_task_pair_occurrences,
                lineage.shared_lineage_task_pair_count,
                "shared-lineage task-pair occurrences",
            )?;

            let admission = if hard_blocker {
                "blocked_exact_overlap_shared_lineage_or_exposure"
            } else if manual_review_required {
                "manual_review_required_numeric_near_duplicate_candidates"
            } else {
                "no_detected_overlap_within_this_declared_pair"
            };

            pairs.push(SplitPairAuditV1 {
                left_split_label: left.split_label.to_owned(),
                right_split_label: right.split_label.to_owned(),
                policy_input,
                lineage,
                hard_blocker,
                manual_review_required,
                admission: admission.to_owned(),
            });
        }
    }

    let observed_pair_count =
        u64::try_from(pairs.len()).map_err(|_| "observed pair count exceeds u64")?;
    if observed_pair_count != EXPECTED_PAIR_COUNT {
        return Err(format!(
            "internal split-matrix closure error: expected {EXPECTED_PAIR_COUNT} pairs, observed {observed_pair_count}"
        ));
    }

    let admission = if hard_blocking_pair_count > 0 {
        "blocked_complete_split_matrix"
    } else if manual_review_pair_count > 0 {
        "manual_review_required_complete_split_matrix"
    } else {
        "no_detected_overlap_across_complete_declared_matrix_only"
    };

    Ok(SplitMatrixAuditReportV1 {
        schema: REPORT_SCHEMA.to_owned(),
        valid: true,
        complete_matrix: true,
        required_split_roles: REQUIRED_ROLES.iter().map(|role| (*role).to_owned()).collect(),
        observed_split_labels: ordered
            .iter()
            .map(|split| split.split_label.to_owned())
            .collect(),
        expected_pair_count: EXPECTED_PAIR_COUNT,
        observed_pair_count,
        hard_blocking_pair_count,
        manual_review_pair_count,
        shared_task_id_occurrences,
        exact_policy_overlap_pair_count,
        numeric_near_duplicate_candidate_pair_count,
        shared_lineage_root_occurrences,
        shared_lineage_task_pair_occurrences,
        pairs,
        admission: admission.to_owned(),
        evidence_scope: "complete_pairwise_audit_of_the_three_supplied_task_sets_and_caller_declared_lineage_commitments_only".to_owned(),
        lineage_attestation: "caller_declared_not_independently_authenticated".to_owned(),
        semantic_independence: "not_assessed".to_owned(),
        model_training_contamination: "not_assessed".to_owned(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        TASK_SET_SCHEMA, TaskSetArtifactV1, leakage_audit::canonical_policy_input_sha256,
        sha256_hex,
    };
    use serde_json::{Value, json};

    fn task_set(
        generation: &str,
        task_id: &str,
        instruction: &str,
        x: f64,
        y: f64,
    ) -> Vec<u8> {
        serde_json::to_vec(&json!({
            "schema": TASK_SET_SCHEMA,
            "generation_id": generation,
            "tasks": [{
                "task_id": task_id,
                "family_id": "fixture-family",
                "policy_input": {
                    "instruction": instruction,
                    "observations": [{"x": x, "y": y}, {"x": x + 1.0, "y": y + 2.0}]
                },
                "evaluator_only": {"expected_law": "private-fixture-answer"}
            }]
        }))
        .expect("task set serializes")
    }

    fn ledger(
        task_set_bytes: &[u8],
        split_label: &str,
        lineage_root: &str,
        exposure_state: &str,
    ) -> Vec<u8> {
        let task_set: TaskSetArtifactV1 = serde_json::from_slice(task_set_bytes).unwrap();
        let task = &task_set.tasks[0];
        let policy_digest = canonical_policy_input_sha256(&task.policy_input).unwrap();
        serde_json::to_vec(&json!({
            "schema": "symthaea.science_bench_refresh_lineage_ledger.v1",
            "generation_id": task_set.generation_id.clone(),
            "split_label": split_label,
            "task_set_sha256": sha256_hex(task_set_bytes),
            "records": [{
                "task_id": task.task_id.clone(),
                "policy_input_sha256": policy_digest,
                "exposure_state": exposure_state,
                "lineage_commitments": [{
                    "kind": "parent_world",
                    "commitment_sha256": sha256_hex(lineage_root.as_bytes())
                }]
            }]
        }))
        .unwrap()
    }

    struct Fixture {
        label: String,
        task_set: Vec<u8>,
        ledger: Vec<u8>,
    }

    impl Fixture {
        fn borrowed(&self) -> SplitArtifactRefV1<'_> {
            SplitArtifactRefV1 {
                split_label: &self.label,
                task_set_bytes: &self.task_set,
                lineage_ledger_bytes: &self.ledger,
            }
        }
    }

    fn fixture(
        role: &str,
        task_id: &str,
        instruction: &str,
        x: f64,
        y: f64,
        root: &str,
        exposure: &str,
    ) -> Fixture {
        let label = format!("{role}:v1");
        let set = task_set(
            &format!("{role}-generation"),
            task_id,
            instruction,
            x,
            y,
        );
        let ledger = ledger(&set, &label, root, exposure);
        Fixture {
            label,
            task_set: set,
            ledger,
        }
    }

    fn clean_fixtures() -> [Fixture; 3] {
        [
            fixture(
                "anchor",
                "a-1",
                "Recover the relation.",
                1.0,
                2.0,
                "anchor-world",
                "never_exposed",
            ),
            fixture(
                "development",
                "d-1",
                "Infer a relationship.",
                4.0,
                8.0,
                "development-world",
                "public_development",
            ),
            fixture(
                "refresh",
                "r-1",
                "Estimate the mapping.",
                9.0,
                18.0,
                "refresh-world",
                "never_exposed",
            ),
        ]
    }

    #[test]
    fn complete_matrix_audits_exactly_three_pairs_deterministically() {
        let fixtures = clean_fixtures();
        let refs: Vec<_> = fixtures.iter().map(Fixture::borrowed).collect();
        let first = audit_required_split_matrix(&refs).unwrap();
        let second = audit_required_split_matrix(&refs).unwrap();

        assert!(first.valid);
        assert!(first.complete_matrix);
        assert_eq!(first.expected_pair_count, 3);
        assert_eq!(first.observed_pair_count, 3);
        assert_eq!(first.pairs.len(), 3);
        assert_eq!(
            first.admission,
            "no_detected_overlap_across_complete_declared_matrix_only"
        );
        assert_eq!(first.semantic_independence, "not_assessed");
        assert_eq!(first.model_training_contamination, "not_assessed");
        assert_eq!(
            serde_json::to_vec(&first).unwrap(),
            serde_json::to_vec(&second).unwrap()
        );
    }

    #[test]
    fn omission_of_any_required_role_fails_closed() {
        let fixtures = clean_fixtures();
        for omitted in 0..fixtures.len() {
            let refs: Vec<_> = fixtures
                .iter()
                .enumerate()
                .filter(|(index, _)| *index != omitted)
                .map(|(_, fixture)| fixture.borrowed())
                .collect();
            assert!(audit_required_split_matrix(&refs).is_err());
        }
    }

    #[test]
    fn duplicate_role_with_a_different_generation_suffix_fails_closed() {
        let mut fixtures = clean_fixtures();
        fixtures[1].label = "anchor:v2".to_owned();
        fixtures[1].ledger = ledger(
            &fixtures[1].task_set,
            "anchor:v2",
            "development-world",
            "never_exposed",
        );
        let refs: Vec<_> = fixtures.iter().map(Fixture::borrowed).collect();
        let error = audit_required_split_matrix(&refs).unwrap_err();
        assert!(error.contains("more than one partition for role anchor"));
    }

    #[test]
    fn caller_label_must_match_embedded_lineage_label() {
        let mut fixtures = clean_fixtures();
        fixtures[0].ledger = ledger(
            &fixtures[0].task_set,
            "refresh:v1",
            "anchor-world",
            "never_exposed",
        );
        let refs: Vec<_> = fixtures.iter().map(Fixture::borrowed).collect();
        let error = audit_required_split_matrix(&refs).unwrap_err();
        assert!(error.contains("lineage ledger labels do not match caller labels"));
    }

    #[test]
    fn overlap_in_anchor_refresh_pair_is_found_even_when_other_pairs_are_clean() {
        let mut fixtures = clean_fixtures();
        fixtures[2].task_set = task_set(
            "refresh-generation",
            "r-1",
            "Recover the relation.",
            1.0,
            2.0,
        );
        fixtures[2].ledger = ledger(
            &fixtures[2].task_set,
            "refresh:v1",
            "refresh-world",
            "never_exposed",
        );
        let refs: Vec<_> = fixtures.iter().map(Fixture::borrowed).collect();
        let report = audit_required_split_matrix(&refs).unwrap();

        assert_eq!(report.hard_blocking_pair_count, 1);
        let anchor_refresh = report
            .pairs
            .iter()
            .find(|pair| {
                pair.left_split_label.starts_with("anchor:")
                    && pair.right_split_label.starts_with("refresh:")
            })
            .unwrap();
        assert!(anchor_refresh.hard_blocker);
        assert_eq!(anchor_refresh.policy_input.exact_overlap_pair_count, 1);
    }

    #[test]
    fn reused_task_id_blocks_even_when_policy_inputs_differ() {
        let mut fixtures = clean_fixtures();
        fixtures[2] = fixture(
            "refresh",
            "a-1",
            "A different instruction and different observations.",
            91.0,
            182.0,
            "unique-refresh-world",
            "never_exposed",
        );
        let refs: Vec<_> = fixtures.iter().map(Fixture::borrowed).collect();
        let report = audit_required_split_matrix(&refs).unwrap();

        assert_eq!(report.shared_task_id_occurrences, 1);
        assert_eq!(report.hard_blocking_pair_count, 1);
        let anchor_refresh = report
            .pairs
            .iter()
            .find(|pair| {
                pair.left_split_label.starts_with("anchor:")
                    && pair.right_split_label.starts_with("refresh:")
            })
            .unwrap();
        assert_eq!(anchor_refresh.policy_input.shared_task_id_count, 1);
        assert_eq!(anchor_refresh.policy_input.exact_overlap_pair_count, 0);
        assert!(anchor_refresh.hard_blocker);
    }

    #[test]
    fn numeric_near_duplicate_is_review_not_a_confirmed_block_or_clean_result() {
        let mut fixtures = clean_fixtures();
        fixtures[2].task_set = task_set(
            "refresh-generation",
            "r-1",
            "Recover the relation.",
            1.23452,
            2.34562,
        );
        fixtures[2].ledger = ledger(
            &fixtures[2].task_set,
            "refresh:v1",
            "refresh-world",
            "never_exposed",
        );
        fixtures[0].task_set = task_set(
            "anchor-generation",
            "a-1",
            "Recover the relation.",
            1.23451,
            2.34561,
        );
        fixtures[0].ledger = ledger(
            &fixtures[0].task_set,
            "anchor:v1",
            "anchor-world",
            "never_exposed",
        );
        let refs: Vec<_> = fixtures.iter().map(Fixture::borrowed).collect();
        let report = audit_required_split_matrix(&refs).unwrap();

        assert_eq!(report.hard_blocking_pair_count, 0);
        assert_eq!(report.manual_review_pair_count, 1);
        assert_eq!(
            report.admission,
            "manual_review_required_complete_split_matrix"
        );
    }

    #[test]
    fn exposed_refresh_task_blocks_even_when_lineage_roots_are_distinct() {
        let mut fixtures = clean_fixtures();
        fixtures[2] = fixture(
            "refresh",
            "r-1",
            "Estimate the mapping.",
            9.0,
            18.0,
            "unique-refresh-world",
            "prior_scored_refresh",
        );
        let refs: Vec<_> = fixtures.iter().map(Fixture::borrowed).collect();
        let report = audit_required_split_matrix(&refs).unwrap();
        assert!(report.hard_blocking_pair_count > 0);
        assert_eq!(report.admission, "blocked_complete_split_matrix");
    }

    #[test]
    fn incomplete_lineage_ledger_fails_closed() {
        let mut fixtures = clean_fixtures();
        let mut value: Value = serde_json::from_slice(&fixtures[0].ledger).unwrap();
        value["records"] = json!([]);
        fixtures[0].ledger = serde_json::to_vec(&value).unwrap();
        let refs: Vec<_> = fixtures.iter().map(Fixture::borrowed).collect();
        assert!(audit_required_split_matrix(&refs).is_err());
    }

    #[test]
    fn report_never_discloses_evaluator_only_targets() {
        let fixtures = clean_fixtures();
        let refs: Vec<_> = fixtures.iter().map(Fixture::borrowed).collect();
        let report = audit_required_split_matrix(&refs).unwrap();
        let encoded = serde_json::to_string(&report).unwrap();
        assert!(!encoded.contains("private-fixture-answer"));
    }

    #[test]
    fn every_required_pair_contains_both_independent_audit_reports() {
        let fixtures = clean_fixtures();
        let refs: Vec<_> = fixtures.iter().map(Fixture::borrowed).collect();
        let report = audit_required_split_matrix(&refs).unwrap();
        for pair in report.pairs {
            assert!(pair.policy_input.valid);
            assert!(pair.lineage.valid);
            assert!(!pair.left_split_label.is_empty());
            assert!(!pair.right_split_label.is_empty());
        }
    }

    #[test]
    fn checked_aggregate_counters_reject_overflow() {
        let mut total = u64::MAX;
        assert!(add_checked(&mut total, 1, "test counter").is_err());
    }
}
