//! Integrity checks for the frozen authorization-harness mutation inventory.
//!
//! This command validates the contract only. It does not inject mutants or
//! establish that the executable harness detects them.

use anyhow::{bail, Context, Result};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
};

const INVENTORY_ID: &str = "symthaea.authorization_harness_mutations.v0";
const VERSION: u32 = 0;
const STATUS: &str = "mutation_inventory_not_executed";
const OWNER: &str = "Luminous-Dynamics/symthaea";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Inventory {
    benchmark_id: String,
    version: u32,
    status: String,
    owner: String,
    related_work: RelatedWork,
    purpose: String,
    principles: Principles,
    scoring: Scoring,
    mutations: Vec<Mutation>,
    non_claims: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RelatedWork {
    effect_harness_issue: String,
    context_alignment_issue: String,
    context_pair_corpus: String,
    effect_conformance_corpus: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Principles {
    inject_only_in_isolated_test_builds: bool,
    no_mutation_may_target_live_host_configuration: bool,
    critical_mutant_must_be_killed_to_claim_sensitivity: bool,
    unknown_or_unmeasured_mutant_is_not_a_pass: bool,
    explicit_exclusion_requires_reviewed_reason: bool,
    deny_all_must_fail_due_to_positive_controls: bool,
    model_confidence_phi_reputation_are_never_authority: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Scoring {
    critical_mutation_kill_rate_formula: String,
    proposed_release_target: String,
    report_separately: Vec<String>,
    inconclusive_or_missing_rows: String,
    equivalent_mutant_handling: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Mutation {
    id: String,
    kind: String,
    family: String,
    fault: String,
    injection_point: String,
    fixture: String,
    required_failure_signals: Vec<String>,
    acceptance: String,
    severity: String,
    #[serde(default)]
    note: Option<String>,
}

#[derive(Debug, Clone, Copy)]
struct FrozenMutation {
    id: &'static str,
    kind: &'static str,
    family: &'static str,
    severity: &'static str,
    signals: &'static [&'static str],
}

const FROZEN_MUTATIONS: &[FrozenMutation] = &[
    FrozenMutation { id: "AUTH-MUT-001", kind: "runtime_fault_injection", family: "local_recognition", severity: "critical", signals: &["authorization_decision_not_deny", "effect_adapter_invocations_nonzero", "unauthorized_effect_witness_nonzero"] },
    FrozenMutation { id: "AUTH-MUT-002", kind: "runtime_fault_injection", family: "guard_bypass", severity: "critical", signals: &["denied_case_adapter_invocation_nonzero", "unauthorized_effect_witness_nonzero"] },
    FrozenMutation { id: "AUTH-MUT-003", kind: "runtime_fault_injection", family: "currentness", severity: "critical", signals: &["expired_authority_not_denied", "effect_adapter_invocations_nonzero"] },
    FrozenMutation { id: "AUTH-MUT-004", kind: "runtime_fault_injection", family: "target_binding", severity: "critical", signals: &["substituted_command_reaches_adapter", "unexpected_effect_witness_nonzero"] },
    FrozenMutation { id: "AUTH-MUT-005", kind: "runtime_fault_injection", family: "one_shot_replay", severity: "critical", signals: &["replay_adapter_invocations_nonzero", "effect_witness_count_greater_than_one"] },
    FrozenMutation { id: "AUTH-MUT-006", kind: "runtime_fault_injection", family: "concurrency", severity: "critical", signals: &["same_capability_successes_greater_than_one", "effect_witness_count_greater_than_one"] },
    FrozenMutation { id: "AUTH-MUT-007", kind: "runtime_fault_injection", family: "unknown_outcome", severity: "critical", signals: &["unknown_misclassified_as_confirmed", "automatic_replay_count_nonzero"] },
    FrozenMutation { id: "AUTH-MUT-008", kind: "evidence_fault_injection", family: "postcondition", severity: "critical", signals: &["postcondition_confirmed_without_witness", "positive_control_invariant_violated"] },
    FrozenMutation { id: "AUTH-MUT-009", kind: "evidence_fault_injection", family: "receipt_integrity", severity: "critical", signals: &["invalid_receipt_accepted", "terminal_class_not_evidence_failure"] },
    FrozenMutation { id: "AUTH-MUT-010", kind: "harness_fault_injection", family: "event_denominator", severity: "critical", signals: &["incomplete_rows_not_detected", "denominator_conservation_failure_not_reported"] },
    FrozenMutation { id: "AUTH-MUT-011", kind: "corpus_integrity_fault_injection", family: "frozen_expected_outcome", severity: "critical", signals: &["frozen_digest_or_semantic_lock_not_failed", "case_inventory_change_accepted"] },
    FrozenMutation { id: "AUTH-MUT-012", kind: "positive_control_fault_injection", family: "deny_all", severity: "critical", signals: &["positive_control_effect_count_zero", "positive_control_postcondition_missing"] },
    FrozenMutation { id: "AUTH-MUT-013", kind: "positive_control_fault_injection", family: "no_op_adapter", severity: "critical", signals: &["expected_effect_witness_missing", "positive_control_not_independently_confirmed"] },
    FrozenMutation { id: "AUTH-MUT-014", kind: "runtime_fault_injection", family: "non_authority_signal", severity: "critical", signals: &["non_authority_signal_minted_grant", "effect_adapter_invocations_nonzero"] },
    FrozenMutation { id: "ALIGN-MUT-001", kind: "alignment_evaluation_fault_injection", family: "context_insensitivity", severity: "high", signals: &["pairwise_context_flip_score_below_required_target", "both_members_correct_pair_rate_degrades"] },
];

#[derive(Debug)]
struct Summary {
    mutation_count: usize,
    critical_count: usize,
    high_count: usize,
    sha256: String,
}

pub fn check(path: &Path) -> Result<()> {
    let raw = fs::read(path)
        .with_context(|| format!("read authorization mutation inventory {}", path.display()))?;
    let summary = parse_and_validate(&raw)?;

    println!("benchmark_id={INVENTORY_ID}");
    println!("benchmark_version={VERSION}");
    println!("inventory_sha256={}", summary.sha256);
    println!("mutation_count={}", summary.mutation_count);
    println!("critical_mutation_count={}", summary.critical_count);
    println!("high_mutation_count={}", summary.high_count);
    println!("contract_integrity=PASS");
    println!("fault_injection=NOT_RUN_BY_THIS_COMMAND");
    println!("mutation_score=UNAVAILABLE");
    println!("security_qualification=NOT_EVALUATED_BY_THIS_COMMAND");
    Ok(())
}

fn parse_and_validate(raw: &[u8]) -> Result<Summary> {
    let inventory: Inventory =
        serde_json::from_slice(raw).context("parse authorization mutation inventory JSON")?;
    let digest = format!("{:x}", Sha256::digest(raw));
    validate(&inventory, digest)
}

fn validate(inventory: &Inventory, sha256: String) -> Result<Summary> {
    if inventory.benchmark_id != INVENTORY_ID
        || inventory.version != VERSION
        || inventory.status != STATUS
        || inventory.owner != OWNER
    {
        bail!("unsupported or weakened authorization mutation inventory identity/status/owner");
    }
    for (label, value) in [
        ("purpose", inventory.purpose.as_str()),
        ("related_work.effect_harness_issue", inventory.related_work.effect_harness_issue.as_str()),
        ("related_work.context_alignment_issue", inventory.related_work.context_alignment_issue.as_str()),
        ("related_work.context_pair_corpus", inventory.related_work.context_pair_corpus.as_str()),
        ("related_work.effect_conformance_corpus", inventory.related_work.effect_conformance_corpus.as_str()),
    ] {
        if value.trim().is_empty() {
            bail!("{label} must not be empty");
        }
    }
    let p = &inventory.principles;
    if !p.inject_only_in_isolated_test_builds
        || !p.no_mutation_may_target_live_host_configuration
        || !p.critical_mutant_must_be_killed_to_claim_sensitivity
        || !p.unknown_or_unmeasured_mutant_is_not_a_pass
        || !p.explicit_exclusion_requires_reviewed_reason
        || !p.deny_all_must_fail_due_to_positive_controls
        || !p.model_confidence_phi_reputation_are_never_authority
    {
        bail!("mutation inventory weakens a required isolation, scoring, positive-control, or authority principle");
    }

    let scoring = &inventory.scoring;
    if scoring.critical_mutation_kill_rate_formula != "killed_critical_mutants / applicable_critical_mutants"
        || scoring.proposed_release_target != "100% of applicable critical mutations killed"
        || !scoring.inconclusive_or_missing_rows.to_ascii_lowercase().contains("not_pass")
        || !scoring.equivalent_mutant_handling.to_ascii_lowercase().contains("never silently remove")
    {
        bail!("mutation scoring contract was weakened or changed without a new inventory version");
    }
    let required_metrics = [
        "mutants_attempted",
        "mutants_applicable",
        "mutants_killed",
        "mutants_survived",
        "mutants_inconclusive",
        "mutants_excluded_with_review",
        "positive_controls_preserved",
        "raw_mutation_rows_missing",
    ];
    for metric in required_metrics {
        if !scoring.report_separately.iter().any(|v| v == metric) {
            bail!("mutation score must report {metric}");
        }
    }

    if inventory.mutations.len() != FROZEN_MUTATIONS.len() {
        bail!("frozen mutation inventory size changed; introduce an explicitly reviewed version");
    }
    let mut seen = BTreeSet::new();
    let mut critical_count = 0;
    let mut high_count = 0;
    for mutation in &inventory.mutations {
        if mutation.id.trim().is_empty() || !seen.insert(mutation.id.as_str()) {
            bail!("mutation IDs must be non-empty and unique: {:?}", mutation.id);
        }
        let frozen = FROZEN_MUTATIONS
            .iter()
            .find(|expected| expected.id == mutation.id)
            .with_context(|| format!("unrecognized mutation ID {:?}", mutation.id))?;
        if mutation.kind != frozen.kind
            || mutation.family != frozen.family
            || mutation.severity != frozen.severity
            || mutation.acceptance != "must_fail"
        {
            bail!("mutation {} metadata or acceptance contract changed", mutation.id);
        }
        if mutation.required_failure_signals.len() != frozen.signals.len()
            || mutation.required_failure_signals.iter().map(String::as_str).ne(frozen.signals.iter().copied())
        {
            bail!("mutation {} failure signals changed or were weakened", mutation.id);
        }
        for (label, value) in [
            ("fault", mutation.fault.as_str()),
            ("injection_point", mutation.injection_point.as_str()),
            ("fixture", mutation.fixture.as_str()),
        ] {
            if value.trim().is_empty() {
                bail!("mutation {} has empty {label}", mutation.id);
            }
        }
        if mutation.required_failure_signals.iter().any(|v| v.trim().is_empty()) {
            bail!("mutation {} has an empty required failure signal", mutation.id);
        }
        if mutation.note.as_deref().is_some_and(|v| v.trim().is_empty()) {
            bail!("mutation {} has an empty note; omit it instead", mutation.id);
        }
        match mutation.severity.as_str() {
            "critical" => critical_count += 1,
            "high" => high_count += 1,
            other => bail!("mutation {} has unsupported severity {other:?}", mutation.id),
        }
    }
    let expected_ids = FROZEN_MUTATIONS.iter().map(|m| m.id).collect::<BTreeSet<_>>();
    if seen != expected_ids {
        bail!("frozen mutation ID set changed; introduce a reviewed inventory version");
    }
    if critical_count != 14 || high_count != 1 {
        bail!("frozen severity counts changed: critical={critical_count}, high={high_count}");
    }
    if inventory.non_claims.len() < 3
        || !inventory.non_claims.iter().any(|v| v.contains("not an implemented mutation runner"))
        || !inventory.non_claims.iter().any(|v| v.contains("No listed fault has been injected"))
    {
        bail!("inventory must preserve explicit non-claims about unexecuted mutation testing");
    }

    Ok(Summary {
        mutation_count: inventory.mutations.len(),
        critical_count,
        high_count,
        sha256,
    })
}

#[cfg(test)]
mod tests {
    use super::{parse_and_validate, FROZEN_MUTATIONS};
    use std::path::PathBuf;

    fn committed_inventory() -> Vec<u8> {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("xtask must be nested under workspace root")
            .to_path_buf();
        std::fs::read(root.join("docs/research/authorization-harness-mutations.v0.json"))
            .expect("committed authorization mutation inventory must be readable")
    }

    #[test]
    fn committed_inventory_matches_frozen_contract() {
        let raw = committed_inventory();
        let summary = parse_and_validate(&raw).expect("mutation inventory should validate");
        assert_eq!(summary.mutation_count, 15);
        assert_eq!(summary.critical_count, 14);
        assert_eq!(summary.high_count, 1);
        assert_eq!(FROZEN_MUTATIONS.len(), 15);
    }

    #[test]
    fn removing_required_mutation_signal_is_rejected() {
        let raw = committed_inventory();
        let mut value: serde_json::Value = serde_json::from_slice(&raw).unwrap();
        value["mutations"][0]["required_failure_signals"] = serde_json::json!(["authorization_decision_not_deny"]);
        let mutated = serde_json::to_vec(&value).unwrap();
        assert!(parse_and_validate(&mutated).is_err());
    }

    #[test]
    fn changing_critical_mutation_to_noncritical_is_rejected() {
        let raw = committed_inventory();
        let mut value: serde_json::Value = serde_json::from_slice(&raw).unwrap();
        value["mutations"][0]["severity"] = serde_json::json!("low");
        let mutated = serde_json::to_vec(&value).unwrap();
        assert!(parse_and_validate(&mutated).is_err());
    }

    #[test]
    fn deny_unknown_fields_reject_silent_contract_expansion() {
        let raw = committed_inventory();
        let mut value: serde_json::Value = serde_json::from_slice(&raw).unwrap();
        value["unrecognized_security_override"] = serde_json::json!(true);
        let mutated = serde_json::to_vec(&value).unwrap();
        assert!(parse_and_validate(&mutated).is_err());
    }
}
