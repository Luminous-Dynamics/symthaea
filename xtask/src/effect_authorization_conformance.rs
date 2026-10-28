//! Structural integrity checks for the frozen effect-authorization conformance corpus.
//!
//! This validates the benchmark contract only. It does not execute cases, verify
//! credentials, instrument side effects, or qualify any effect surface.

use anyhow::{bail, Context, Result};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{collections::BTreeSet, fs, path::Path};

const BENCHMARK_ID: &str = "symthaea.effect_authorization_conformance.v1";
const VERSION: u32 = 1;
const FROZEN_STATUS: &str = "frozen_design_contract_not_executed";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    benchmark_id: String,
    version: u32,
    status: String,
    owner: String,
    related_work: RelatedWork,
    purpose: String,
    fixture_contract: FixtureContract,
    test_profile: TestProfile,
    event_denominators: Vec<String>,
    case_schema: CaseSchema,
    gates: Gates,
    performance_policy: PerformancePolicy,
    cases: Vec<Case>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RelatedWork {
    registry: String,
    federated_identity_issue: String,
    effect_surface_issue: String,
    mycelix_claim_recognition_issue: String,
    mycelix_typed_receipt_issue: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FixtureContract {
    baseline_fixture: String,
    isolation: String,
    mutation_semantics: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TestProfile {
    id: String,
    currentness_required_at_effect_entry: bool,
    deny_effect_on_denial_or_indeterminate: bool,
    foreign_claim_can_mint_local_grant: bool,
    reputation_or_model_confidence_can_mint_authority: bool,
    one_shot_consumption_must_be_atomic: bool,
    consumed_permit_survives_declared_restart_model: bool,
    successful_dispatch_requires_independent_postcondition: bool,
    uncertain_outcome_may_auto_retry: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CaseSchema {
    expected_effect_adapter_invocations_for_case: String,
    effect_witness: String,
    terminal_class: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Gates {
    frozen_negative_controls_correct: String,
    unauthorized_or_indeterminate_effect_adapter_invocations: u64,
    unauthorized_side_effects: u64,
    positive_controls: String,
    same_capability_concurrent_duplicate_effects: u64,
    consumed_capability_reuse_after_restart: u64,
    unknown_outcome_automatic_replays: u64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PerformancePolicy {
    report_separately: Vec<String>,
    percentiles: Vec<String>,
    occupancy_points: Vec<u64>,
    requirements: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    id: String,
    category: String,
    description: String,
    fixture: String,
    setup: Setup,
    mutations: Vec<Mutation>,
    expected: Expected,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Setup {
    successful_effects_before_case: u64,
    #[serde(default)]
    consumed_capability_persisted: Option<bool>,
    #[serde(default)]
    one_shared_capability: Option<bool>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Mutation {
    path: String,
    value: serde_json::Value,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Expected {
    authorization_decision: String,
    effect_adapter_invocations_for_case: u64,
    effect_witness: String,
    terminal_class: String,
}

#[derive(Debug)]
struct Summary {
    cases: usize,
    positive: usize,
    negative: usize,
    indeterminate: usize,
    lifecycle: usize,
    concurrent: usize,
    corpus_sha256: String,
}

pub fn check(path: &Path) -> Result<()> {
    let raw = fs::read(path)
        .with_context(|| format!("read conformance corpus {}", path.display()))?;
    let manifest: Manifest =
        serde_json::from_slice(&raw).context("parse effect-authorization conformance JSON")?;
    let digest = format!("{:x}", Sha256::digest(&raw));
    let summary = validate(&manifest, digest)?;

    println!("benchmark_id={}", manifest.benchmark_id);
    println!("benchmark_version={}", manifest.version);
    println!("corpus_sha256={}", summary.corpus_sha256);
    println!("case_count={}", summary.cases);
    println!("positive_control_count={}", summary.positive);
    println!("negative_control_count={}", summary.negative);
    println!("indeterminate_control_count={}", summary.indeterminate);
    println!("lifecycle_case_count={}", summary.lifecycle);
    println!("concurrent_single_use_case_count={}", summary.concurrent);
    println!("contract_integrity=PASS");
    println!("execution_status=NOT_EXECUTED_BY_THIS_COMMAND");
    println!("security_qualification=NOT_EVALUATED_BY_THIS_COMMAND");
    println!("effect_surface_coverage=NOT_EVALUATED_BY_THIS_COMMAND");
    Ok(())
}

fn validate(manifest: &Manifest, corpus_sha256: String) -> Result<Summary> {
    if manifest.benchmark_id != BENCHMARK_ID || manifest.version != VERSION {
        bail!(
            "unsupported benchmark identity/version: {:?} v{}",
            manifest.benchmark_id,
            manifest.version
        );
    }
    if manifest.status != FROZEN_STATUS {
        bail!(
            "v1 contract status must remain {:?}; got {:?}",
            FROZEN_STATUS,
            manifest.status
        );
    }
    require_nonempty("owner", &manifest.owner)?;
    require_nonempty("purpose", &manifest.purpose)?;
    require_nonempty("related_work.registry", &manifest.related_work.registry)?;
    require_nonempty("related_work.federated_identity_issue", &manifest.related_work.federated_identity_issue)?;
    require_nonempty("related_work.effect_surface_issue", &manifest.related_work.effect_surface_issue)?;
    require_nonempty("related_work.mycelix_claim_recognition_issue", &manifest.related_work.mycelix_claim_recognition_issue)?;
    require_nonempty("related_work.mycelix_typed_receipt_issue", &manifest.related_work.mycelix_typed_receipt_issue)?;
    require_nonempty("fixture_contract.baseline_fixture", &manifest.fixture_contract.baseline_fixture)?;
    require_nonempty("fixture_contract.isolation", &manifest.fixture_contract.isolation)?;
    require_nonempty("fixture_contract.mutation_semantics", &manifest.fixture_contract.mutation_semantics)?;
    require_nonempty("test_profile.id", &manifest.test_profile.id)?;
    require_nonempty("case_schema.expected_effect_adapter_invocations_for_case", &manifest.case_schema.expected_effect_adapter_invocations_for_case)?;
    require_nonempty("case_schema.effect_witness", &manifest.case_schema.effect_witness)?;
    require_nonempty("case_schema.terminal_class", &manifest.case_schema.terminal_class)?;
    if !manifest.case_schema.expected_effect_adapter_invocations_for_case.contains("point-of-effect adapter") {
        bail!("case schema must count invocations at the instrumented point-of-effect adapter");
    }

    ensure_nonempty_list("event_denominators", &manifest.event_denominators)?;
    ensure_nonempty_list("performance_policy.report_separately", &manifest.performance_policy.report_separately)?;
    ensure_nonempty_list("performance_policy.percentiles", &manifest.performance_policy.percentiles)?;
    ensure_nonempty_list("performance_policy.occupancy_points", &manifest.performance_policy.occupancy_points)?;
    ensure_nonempty_list("performance_policy.requirements", &manifest.performance_policy.requirements)?;
    const REQUIRED_EVENTS: &[&str] = &[
        "effect_adapter_invocations", "observed_side_effects", "postconditions_confirmed",
        "postconditions_failed", "outcome_unknown", "receipt_verification_failed",
        "timeouts", "missing_or_incomplete_rows",
    ];
    for event in REQUIRED_EVENTS {
        if !manifest.event_denominators.iter().any(|item| item == event) {
            bail!("event denominators are missing required outcome stage {event}");
        }
    }
    const REQUIRED_STAGES: &[&str] = &[
        "credential_verification", "local_recognition", "policy_decision",
        "durable_reservation", "dispatch_admission", "effect_adapter",
        "postcondition_verification", "end_to_end",
    ];
    for stage in REQUIRED_STAGES {
        if !manifest.performance_policy.report_separately.iter().any(|item| item == stage) {
            bail!("performance report is missing comparable stage {stage}");
        }
    }
    if !manifest.gates.positive_controls.to_ascii_lowercase().contains("exactly once")
        || !manifest.gates.positive_controls.to_ascii_lowercase().contains("deny-all")
    {
        bail!("positive-control gate must require exactly-once legitimate execution and explicitly fail deny-all systems");
    }

    let profile = &manifest.test_profile;
    if !profile.currentness_required_at_effect_entry
        || !profile.deny_effect_on_denial_or_indeterminate
        || profile.foreign_claim_can_mint_local_grant
        || profile.reputation_or_model_confidence_can_mint_authority
        || !profile.one_shot_consumption_must_be_atomic
        || !profile.consumed_permit_survives_declared_restart_model
        || !profile.successful_dispatch_requires_independent_postcondition
        || profile.uncertain_outcome_may_auto_retry
    {
        bail!("strict_current_authority_v1 contains a weakened or contradictory security invariant");
    }

    if !manifest.gates.frozen_negative_controls_correct.starts_with("100%")
        || manifest.gates.unauthorized_or_indeterminate_effect_adapter_invocations != 0
        || manifest.gates.unauthorized_side_effects != 0
        || manifest.gates.same_capability_concurrent_duplicate_effects != 1
        || manifest.gates.consumed_capability_reuse_after_restart != 0
        || manifest.gates.unknown_outcome_automatic_replays != 0
    {
        bail!("benchmark correctness gates are weakened or contradictory");
    }
    require_nonempty("gates.positive_controls", &manifest.gates.positive_controls)?;

    const REQUIRED_PERCENTILES: &[&str] = &["p50", "p95", "p99", "p99.9"];
    for percentile in REQUIRED_PERCENTILES {
        if !manifest.performance_policy.percentiles.iter().any(|item| item == percentile) {
            bail!("performance policy is missing percentile {percentile}");
        }
    }
    const REQUIRED_OCCUPANCY: &[u64] = &[1, 1_000, 10_000, 100_000];
    for expected in REQUIRED_OCCUPANCY {
        if !manifest.performance_policy.occupancy_points.contains(expected) {
            bail!("performance policy is missing occupancy point {expected}");
        }
    }

    if manifest.cases.is_empty() {
        bail!("conformance corpus must not be empty");
    }
    let mut ids = BTreeSet::new();
    let mut positive = 0usize;
    let mut negative = 0usize;
    let mut indeterminate = 0usize;
    let mut lifecycle = 0usize;
    let mut concurrent = 0usize;

    for case in &manifest.cases {
        if case.id.trim().is_empty()
            || !case.id.bytes().all(|byte| byte.is_ascii_uppercase() || byte.is_ascii_digit() || byte == b'-')
            || !ids.insert(case.id.clone())
        {
            bail!("invalid or duplicate conformance case id: {:?}", case.id);
        }
        require_nonempty("case.category", &case.category)?;
        require_nonempty("case.description", &case.description)?;
        require_nonempty("case.fixture", &case.fixture)?;
        require_nonempty("case.expected.authorization_decision", &case.expected.authorization_decision)?;
        require_nonempty("case.expected.effect_witness", &case.expected.effect_witness)?;
        require_nonempty("case.expected.terminal_class", &case.expected.terminal_class)?;
        if case.fixture != manifest.fixture_contract.baseline_fixture {
            bail!("case {} uses an unknown baseline fixture {:?}", case.id, case.fixture);
        }
        validate_case(case)?;
        match case.category.as_str() {
            "positive_control" => positive += 1,
            "lifecycle" => lifecycle += 1,
            "concurrent_replay" => concurrent += 1,
            _ => {}
        }
        if case.id.starts_with("POS-") {
            positive += 0; // ID/category consistency is checked in validate_case.
        } else if case.id.starts_with("NEG-") {
            negative += 1;
        } else if case.id.starts_with("IND-") {
            indeterminate += 1;
        } else if case.id.starts_with("LIFE-") {
            lifecycle += 0; // Avoid double-counting; category was counted above.
        } else {
            bail!("case {} has an unsupported ID class", case.id);
        }
    }
    let expected_ids = (1..=4)
        .map(|number| format!("POS-{number:03}"))
        .chain((1..=32).map(|number| format!("NEG-{number:03}")))
        .chain(std::iter::once("IND-001".to_string()))
        .chain((1..=4).map(|number| format!("LIFE-{number:03}")))
        .collect::<BTreeSet<_>>();
    let actual_ids = manifest.cases.iter().map(|case| case.id.clone()).collect::<BTreeSet<_>>();
    if actual_ids != expected_ids {
        let missing = expected_ids.difference(&actual_ids).cloned().collect::<Vec<_>>();
        let unexpected = actual_ids.difference(&expected_ids).cloned().collect::<Vec<_>>();
        bail!("frozen v1 case inventory changed; missing={missing:?}, unexpected={unexpected:?}; create a reviewed corpus version instead of silently changing denominators");
    }
    if positive != 4 || negative != 32 || indeterminate != 1 || lifecycle != 4 || concurrent != 1 {
        bail!(
            "frozen v1 class counts changed (got +{positive}, -{negative}, indeterminate {indeterminate}, lifecycle {lifecycle}, concurrent {concurrent})"
        );
    }

    Ok(Summary {
        cases: manifest.cases.len(),
        positive,
        negative,
        indeterminate,
        lifecycle,
        concurrent,
        corpus_sha256,
    })
}

#[derive(Debug)]
struct FrozenCase {
    category: &'static str,
    successful_effects_before_case: u64,
    consumed_capability_persisted: Option<bool>,
    one_shared_capability: Option<bool>,
    // Each expected value is represented by its canonical JSON scalar text.
    mutations: &'static [(&'static str, &'static str)],
    authorization_decision: &'static str,
    effect_adapter_invocations: u64,
    effect_witness: &'static str,
    terminal_class: &'static str,
}

fn frozen_case(id: &str) -> Option<FrozenCase> {
    match id {
        "POS-001" => Some(FrozenCase {
            category: "positive_control",
            successful_effects_before_case: 0,
            consumed_capability_persisted: None,
            one_shared_capability: None,
            mutations: &[],
            authorization_decision: "allow",
            effect_adapter_invocations: 1,
            effect_witness: "exactly_one",
            terminal_class: "postcondition_confirmed",
        }),
        "POS-002" => Some(FrozenCase {
            category: "positive_control",
            successful_effects_before_case: 0,
            consumed_capability_persisted: None,
            one_shared_capability: None,
            mutations: &[("delegation.scope_relation", "\"strict_subset\"")],
            authorization_decision: "allow",
            effect_adapter_invocations: 1,
            effect_witness: "exactly_one",
            terminal_class: "postcondition_confirmed",
        }),
        "POS-003" => Some(FrozenCase {
            category: "positive_control",
            successful_effects_before_case: 0,
            consumed_capability_persisted: None,
            one_shared_capability: None,
            mutations: &[("principal_binding.on_behalf_of_match", "true")],
            authorization_decision: "allow",
            effect_adapter_invocations: 1,
            effect_witness: "exactly_one",
            terminal_class: "postcondition_confirmed",
        }),
        "NEG-001" => Some(FrozenCase {
            category: "credential_integrity",
            successful_effects_before_case: 0,
            consumed_capability_persisted: None,
            one_shared_capability: None,
            mutations: &[("credential.signature_valid", "false")],
            authorization_decision: "deny",
            effect_adapter_invocations: 0,
            effect_witness: "none",
            terminal_class: "denied_before_effect",
        }),
        "NEG-002" => Some(FrozenCase {
            category: "issuer_trust",
            successful_effects_before_case: 0,
            consumed_capability_persisted: None,
            one_shared_capability: None,
            mutations: &[("issuer.trust_status", "\"untrusted\"")],
            authorization_decision: "deny",
            effect_adapter_invocations: 0,
            effect_witness: "none",
            terminal_class: "denied_before_effect",
        }),
        "NEG-003" => Some(FrozenCase {
            category: "issuer_currentness",
            successful_effects_before_case: 0,
            consumed_capability_persisted: None,
            one_shared_capability: None,
            mutations: &[("issuer.current_status", "\"revoked\"")],
            authorization_decision: "deny",
            effect_adapter_invocations: 0,
            effect_witness: "none",
            terminal_class: "denied_before_effect",
        }),
        "NEG-004" => Some(FrozenCase {
            category: "expiry",
            successful_effects_before_case: 0,
            consumed_capability_persisted: None,
            one_shared_capability: None,
            mutations: &[("authority.expired_at_dispatch", "true")],
            authorization_decision: "deny",
            effect_adapter_invocations: 0,
            effect_witness: "none",
            terminal_class: "denied_before_effect",
        }),
        "NEG-005" => Some(FrozenCase {
            category: "audience",
            successful_effects_before_case: 0,
            consumed_capability_persisted: None,
            one_shared_capability: None,
            mutations: &[("credential.audience_matches", "false")],
            authorization_decision: "deny",
            effect_adapter_invocations: 0,
            effect_witness: "none",
            terminal_class: "denied_before_effect",
        }),
        "NEG-006" => Some(FrozenCase {
            category: "purpose",
            successful_effects_before_case: 0,
            consumed_capability_persisted: None,
            one_shared_capability: None,
            mutations: &[("recognition.purpose_matches", "false")],
            authorization_decision: "deny",
            effect_adapter_invocations: 0,
            effect_witness: "none",
            terminal_class: "denied_before_effect",
        }),
        "NEG-007" => Some(FrozenCase {
            category: "target_binding",
            successful_effects_before_case: 0,
            consumed_capability_persisted: None,
            one_shared_capability: None,
            mutations: &[("action.target_matches_grant", "false")],
            authorization_decision: "deny",
            effect_adapter_invocations: 0,
            effect_witness: "none",
            terminal_class: "denied_before_effect",
        }),
        "NEG-008" => Some(FrozenCase {
            category: "resource_scope",
            successful_effects_before_case: 0,
            consumed_capability_persisted: None,
            one_shared_capability: None,
            mutations: &[("action.resource_matches_grant", "false")],
            authorization_decision: "deny",
            effect_adapter_invocations: 0,
            effect_witness: "none",
            terminal_class: "denied_before_effect",
        }),
        "NEG-009" => Some(FrozenCase {
            category: "operation_scope",
            successful_effects_before_case: 0,
            consumed_capability_persisted: None,
            one_shared_capability: None,
            mutations: &[("action.operation_digest_matches", "false")],
            authorization_decision: "deny",
            effect_adapter_invocations: 0,
            effect_witness: "none",
            terminal_class: "denied_before_effect",
        }),
        "NEG-010" => Some(FrozenCase {
            category: "local_authority",
            successful_effects_before_case: 0,
            consumed_capability_persisted: None,
            one_shared_capability: None,
            mutations: &[("authority.local_grant_present", "false")],
            authorization_decision: "deny",
            effect_adapter_invocations: 0,
            effect_witness: "none",
            terminal_class: "denied_before_effect",
        }),
        "NEG-011" => Some(FrozenCase {
            category: "delegation_revocation",
            successful_effects_before_case: 0,
            consumed_capability_persisted: None,
            one_shared_capability: None,
            mutations: &[("delegation.parent_revoked_after_issue", "true")],
            authorization_decision: "deny",
            effect_adapter_invocations: 0,
            effect_witness: "none",
            terminal_class: "denied_before_effect",
        }),
        "NEG-012" => Some(FrozenCase {
            category: "delegation_attenuation",
            successful_effects_before_case: 0,
            consumed_capability_persisted: None,
            one_shared_capability: None,
            mutations: &[("delegation.child_widens_parent", "true")],
            authorization_decision: "deny",
            effect_adapter_invocations: 0,
            effect_witness: "none",
            terminal_class: "denied_before_effect",
        }),
        "NEG-013" => Some(FrozenCase {
            category: "policy_freshness",
            successful_effects_before_case: 0,
            consumed_capability_persisted: None,
            one_shared_capability: None,
            mutations: &[("recognition.policy_generation_current", "false")],
            authorization_decision: "deny",
            effect_adapter_invocations: 0,
            effect_witness: "none",
            terminal_class: "denied_before_effect",
        }),
        "NEG-014" => Some(FrozenCase {
            category: "translation_loss",
            successful_effects_before_case: 0,
            consumed_capability_persisted: None,
            one_shared_capability: None,
            mutations: &[("translation.prohibited_semantic_loss", "true")],
            authorization_decision: "deny",
            effect_adapter_invocations: 0,
            effect_witness: "none",
            terminal_class: "quarantined_before_effect",
        }),
        "NEG-015" => Some(FrozenCase {
            category: "claim_conflict",
            successful_effects_before_case: 0,
            consumed_capability_persisted: None,
            one_shared_capability: None,
            mutations: &[("translation.authoritative_conflict", "true")],
            authorization_decision: "indeterminate",
            effect_adapter_invocations: 0,
            effect_witness: "none",
            terminal_class: "indeterminate_before_effect",
        }),
        "NEG-016" => Some(FrozenCase {
            category: "actor_binding",
            successful_effects_before_case: 0,
            consumed_capability_persisted: None,
            one_shared_capability: None,
            mutations: &[("principal_binding.actor_matches", "false")],
            authorization_decision: "deny",
            effect_adapter_invocations: 0,
            effect_witness: "none",
            terminal_class: "denied_before_effect",
        }),
        "NEG-017" => Some(FrozenCase {
            category: "on_behalf_of_binding",
            successful_effects_before_case: 0,
            consumed_capability_persisted: None,
            one_shared_capability: None,
            mutations: &[("principal_binding.on_behalf_of_matches_grant", "false")],
            authorization_decision: "deny",
            effect_adapter_invocations: 0,
            effect_witness: "none",
            terminal_class: "denied_before_effect",
        }),
        "NEG-018" => Some(FrozenCase {
            category: "executor_composition",
            successful_effects_before_case: 0,
            consumed_capability_persisted: None,
            one_shared_capability: None,
            mutations: &[("executor.software_binding_matches", "false")],
            authorization_decision: "deny",
            effect_adapter_invocations: 0,
            effect_witness: "none",
            terminal_class: "denied_before_effect",
        }),
        "NEG-019" => Some(FrozenCase {
            category: "runtime_incarnation",
            successful_effects_before_case: 0,
            consumed_capability_persisted: None,
            one_shared_capability: None,
            mutations: &[("executor.runtime_incarnation_matches", "false")],
            authorization_decision: "deny",
            effect_adapter_invocations: 0,
            effect_witness: "none",
            terminal_class: "denied_before_effect",
        }),
        "NEG-020" => Some(FrozenCase {
            category: "same_subject_join",
            successful_effects_before_case: 0,
            consumed_capability_persisted: None,
            one_shared_capability: None,
            mutations: &[("executor.verifier_owned_subject_join", "false")],
            authorization_decision: "deny",
            effect_adapter_invocations: 0,
            effect_witness: "none",
            terminal_class: "denied_before_effect",
        }),
        "NEG-021" => Some(FrozenCase {
            category: "reputation_non_authority",
            successful_effects_before_case: 0,
            consumed_capability_persisted: None,
            one_shared_capability: None,
            mutations: &[("authority.local_grant_present", "false"), ("signals.reputation", "100")],
            authorization_decision: "deny",
            effect_adapter_invocations: 0,
            effect_witness: "none",
            terminal_class: "denied_before_effect",
        }),
        "NEG-022" => Some(FrozenCase {
            category: "model_signal_non_authority",
            successful_effects_before_case: 0,
            consumed_capability_persisted: None,
            one_shared_capability: None,
            mutations: &[("authority.local_grant_present", "false"), ("signals.phi", "0.99")],
            authorization_decision: "deny",
            effect_adapter_invocations: 0,
            effect_witness: "none",
            terminal_class: "denied_before_effect",
        }),
        "NEG-023" => Some(FrozenCase {
            category: "receipt_laundering",
            successful_effects_before_case: 0,
            consumed_capability_persisted: None,
            one_shared_capability: None,
            mutations: &[("recognition.is_historical_only", "true")],
            authorization_decision: "deny",
            effect_adapter_invocations: 0,
            effect_witness: "none",
            terminal_class: "denied_before_effect",
        }),
        "NEG-024" => Some(FrozenCase {
            category: "one_shot_replay",
            successful_effects_before_case: 1,
            consumed_capability_persisted: Some(true),
            one_shared_capability: None,
            mutations: &[("attempt.prior_consumption", "\"consumed\"")],
            authorization_decision: "deny",
            effect_adapter_invocations: 0,
            effect_witness: "unchanged_from_setup",
            terminal_class: "consumed_capability_rejected",
        }),
        "NEG-025" => Some(FrozenCase {
            category: "concurrent_replay",
            successful_effects_before_case: 0,
            consumed_capability_persisted: None,
            one_shared_capability: Some(true),
            mutations: &[("attempt.concurrent_submissions", "32")],
            authorization_decision: "allow_then_deny",
            effect_adapter_invocations: 1,
            effect_witness: "exactly_one_total",
            terminal_class: "single_effect_under_race",
        }),
        "NEG-026" => Some(FrozenCase {
            category: "restart_replay",
            successful_effects_before_case: 1,
            consumed_capability_persisted: Some(true),
            one_shared_capability: None,
            mutations: &[("attempt.restart_after_consumption", "true"), ("attempt.prior_consumption", "\"consumed\"")],
            authorization_decision: "deny",
            effect_adapter_invocations: 0,
            effect_witness: "unchanged_from_setup",
            terminal_class: "restart_replay_rejected",
        }),
        "IND-001" => Some(FrozenCase {
            category: "currentness_unavailable",
            successful_effects_before_case: 0,
            consumed_capability_persisted: None,
            one_shared_capability: None,
            mutations: &[("issuer.currentness_available", "false")],
            authorization_decision: "indeterminate",
            effect_adapter_invocations: 0,
            effect_witness: "none",
            terminal_class: "indeterminate_before_effect",
        }),
        "NEG-027" => Some(FrozenCase {
            category: "surface_bypass",
            successful_effects_before_case: 0,
            consumed_capability_persisted: None,
            one_shared_capability: None,
            mutations: &[("effect.entry_path", "\"unregistered_bypass\"")],
            authorization_decision: "deny",
            effect_adapter_invocations: 0,
            effect_witness: "none",
            terminal_class: "bypass_blocked_and_witnessed",
        }),
        "NEG-028" => Some(FrozenCase {
            category: "prestate_binding",
            successful_effects_before_case: 0,
            consumed_capability_persisted: None,
            one_shared_capability: None,
            mutations: &[("action.prestate_digest_matches", "false")],
            authorization_decision: "deny",
            effect_adapter_invocations: 0,
            effect_witness: "none",
            terminal_class: "denied_before_effect",
        }),
        "NEG-029" => Some(FrozenCase {
            category: "configuration_drift",
            successful_effects_before_case: 0,
            consumed_capability_persisted: None,
            one_shared_capability: None,
            mutations: &[("action.target_generation_current", "false")],
            authorization_decision: "deny",
            effect_adapter_invocations: 0,
            effect_witness: "none",
            terminal_class: "denied_before_effect",
        }),
        "LIFE-001" => Some(FrozenCase {
            category: "lifecycle",
            successful_effects_before_case: 0,
            consumed_capability_persisted: None,
            one_shared_capability: None,
            mutations: &[("effect.result", "\"timeout_after_possible_commit\"")],
            authorization_decision: "allow",
            effect_adapter_invocations: 1,
            effect_witness: "unknown_until_reconciled",
            terminal_class: "outcome_unknown_no_automatic_replay",
        }),
        "LIFE-002" => Some(FrozenCase {
            category: "lifecycle",
            successful_effects_before_case: 0,
            consumed_capability_persisted: None,
            one_shared_capability: None,
            mutations: &[("effect.exit_code", "0"), ("postcondition.verified", "false")],
            authorization_decision: "allow",
            effect_adapter_invocations: 1,
            effect_witness: "observed_but_postcondition_false",
            terminal_class: "postcondition_mismatch_quarantined",
        }),
        "LIFE-003" => Some(FrozenCase {
            category: "lifecycle",
            successful_effects_before_case: 0,
            consumed_capability_persisted: None,
            one_shared_capability: None,
            mutations: &[("effect.witness_present", "true"), ("receipt.independent_outcome_present", "false")],
            authorization_decision: "allow",
            effect_adapter_invocations: 1,
            effect_witness: "observed_effect_receipt_missing",
            terminal_class: "evidence_incomplete_not_success",
        }),
        "LIFE-004" => Some(FrozenCase {
            category: "lifecycle",
            successful_effects_before_case: 0,
            consumed_capability_persisted: None,
            one_shared_capability: None,
            mutations: &[("receipt.integrity_valid", "false")],
            authorization_decision: "allow",
            effect_adapter_invocations: 1,
            effect_witness: "effect_may_have_occurred",
            terminal_class: "receipt_tamper_detected_no_success_claim",
        }),
        "NEG-030" => Some(FrozenCase {
            category: "security_domain",
            successful_effects_before_case: 0,
            consumed_capability_persisted: None,
            one_shared_capability: None,
            mutations: &[("action.security_domain_matches", "false")],
            authorization_decision: "deny",
            effect_adapter_invocations: 0,
            effect_witness: "none",
            terminal_class: "denied_before_effect",
        }),
        "NEG-031" => Some(FrozenCase {
            category: "resource_path_alias",
            successful_effects_before_case: 0,
            consumed_capability_persisted: None,
            one_shared_capability: None,
            mutations: &[("action.canonical_resource_within_grant", "false")],
            authorization_decision: "deny",
            effect_adapter_invocations: 0,
            effect_witness: "none",
            terminal_class: "denied_before_effect",
        }),
        "NEG-032" => Some(FrozenCase {
            category: "parameter_mutation",
            successful_effects_before_case: 0,
            consumed_capability_persisted: None,
            one_shared_capability: None,
            mutations: &[("action.execution_manifest_digest_matches", "false")],
            authorization_decision: "deny",
            effect_adapter_invocations: 0,
            effect_witness: "none",
            terminal_class: "denied_before_effect",
        }),
        "POS-004" => Some(FrozenCase {
            category: "positive_control",
            successful_effects_before_case: 0,
            consumed_capability_persisted: None,
            one_shared_capability: None,
            mutations: &[],
            authorization_decision: "allow",
            effect_adapter_invocations: 1,
            effect_witness: "exactly_one",
            terminal_class: "postcondition_confirmed",
        }),
        _ => None,
    }
}

fn validate_case(case: &Case) -> Result<()> {
    let mut mutation_paths = BTreeSet::new();
    for mutation in &case.mutations {
        if !is_field_path(&mutation.path) {
            bail!("case {} has invalid mutation path {:?}", case.id, mutation.path);
        }
        if !mutation_paths.insert(mutation.path.as_str()) {
            bail!("case {} mutates the same field more than once: {}", case.id, mutation.path);
        }
    }

    let frozen = frozen_case(&case.id)
        .with_context(|| format!("unknown v1 conformance case {}; update the versioned contract deliberately", case.id))?;
    if case.category != frozen.category {
        bail!("case {} category changed from frozen v1 value {:?} to {:?}", case.id, frozen.category, case.category);
    }
    if case.setup.successful_effects_before_case != frozen.successful_effects_before_case
        || case.setup.consumed_capability_persisted != frozen.consumed_capability_persisted
        || case.setup.one_shared_capability != frozen.one_shared_capability
    {
        bail!("case {} setup semantics differ from frozen v1 fixture", case.id);
    }
    if case.expected.authorization_decision != frozen.authorization_decision
        || case.expected.effect_adapter_invocations_for_case != frozen.effect_adapter_invocations
        || case.expected.effect_witness != frozen.effect_witness
        || case.expected.terminal_class != frozen.terminal_class
    {
        bail!("case {} expected outcome differs from frozen v1 contract", case.id);
    }
    if case.mutations.len() != frozen.mutations.len() {
        bail!("case {} mutation count changed from frozen v1 contract", case.id);
    }
    for (path, value_json) in frozen.mutations {
        let actual = case.mutations.iter().find(|mutation| mutation.path == *path)
            .with_context(|| format!("case {} is missing frozen mutation {}", case.id, path))?;
        let expected_value: serde_json::Value = serde_json::from_str(value_json)
            .with_context(|| format!("invalid embedded frozen JSON scalar for {}", path))?;
        if actual.value != expected_value {
            bail!("case {} mutation {} value differs from frozen v1 contract", case.id, path);
        }
    }

    match case.id.split_once('-').map(|pair| pair.0) {
        Some("POS") => {
            if case.category != "positive_control"
                || case.expected.authorization_decision != "allow"
                || case.expected.effect_adapter_invocations_for_case != 1
                || case.expected.effect_witness != "exactly_one"
                || case.expected.terminal_class != "postcondition_confirmed"
                || case.setup.successful_effects_before_case != 0
            {
                bail!("positive control {} has a weakened or contradictory expected outcome", case.id);
            }
        }
        Some("NEG") if case.id == "NEG-025" => {
            if case.category != "concurrent_replay"
                || case.expected.authorization_decision != "allow_then_deny"
                || case.expected.effect_adapter_invocations_for_case != 1
                || case.expected.effect_witness != "exactly_one_total"
                || case.expected.terminal_class != "single_effect_under_race"
                || case.setup.one_shared_capability != Some(true)
                || !case.mutations.iter().any(|mutation| mutation.path == "attempt.concurrent_submissions" && mutation.value.as_u64().is_some_and(|value| value > 1))
            {
                bail!("concurrent single-use case {} must expect exactly one aggregate effect under a shared-capability race", case.id);
            }
        }
        Some("NEG") => {
            if case.mutations.is_empty() || case.expected.effect_adapter_invocations_for_case != 0 {
                bail!("negative control {} must mutate the baseline and expect zero effect-adapter calls", case.id);
            }
            match case.id.as_str() {
                "NEG-024" | "NEG-026" => {
                    let expected_terminal = if case.id == "NEG-024" {
                        "consumed_capability_rejected"
                    } else {
                        "restart_replay_rejected"
                    };
                    if case.expected.authorization_decision != "deny"
                        || case.expected.effect_witness != "unchanged_from_setup"
                        || case.expected.terminal_class != expected_terminal
                        || case.setup.successful_effects_before_case != 1
                        || case.setup.consumed_capability_persisted != Some(true)
                    {
                        bail!("replay case {} must preserve prior-effect state and reject the persisted consumed capability", case.id);
                    }
                }
                "NEG-027" => {
                    if case.expected.authorization_decision != "deny"
                        || case.expected.effect_witness != "none"
                        || case.expected.terminal_class != "bypass_blocked_and_witnessed"
                        || case.setup.successful_effects_before_case != 0
                    {
                        bail!("effect-surface bypass case must be blocked without executing the requested effect");
                    }
                }
                _ => {
                    if !matches!(case.expected.authorization_decision.as_str(), "deny" | "indeterminate")
                        || case.expected.effect_witness != "none"
                        || case.setup.successful_effects_before_case != 0
                        || !case.expected.terminal_class.ends_with("_before_effect")
                    {
                        bail!("negative control {} must fail before effect entry without changing the baseline world", case.id);
                    }
                }
            }
        }
        Some("IND") => {
            if case.category != "currentness_unavailable"
                || case.expected.authorization_decision != "indeterminate"
                || case.expected.effect_adapter_invocations_for_case != 0
                || case.expected.effect_witness != "none"
                || case.setup.successful_effects_before_case != 0
            {
                bail!("indeterminate control {} must fail closed before effect entry", case.id);
            }
        }
        Some("LIFE") => {
            let expected_terminal = match case.id.as_str() {
                "LIFE-001" => "outcome_unknown_no_automatic_replay",
                "LIFE-002" => "postcondition_mismatch_quarantined",
                "LIFE-003" => "evidence_incomplete_not_success",
                "LIFE-004" => "receipt_tamper_detected_no_success_claim",
                _ => bail!("unknown lifecycle vector {}; add an explicit expected terminal disposition", case.id),
            };
            if case.category != "lifecycle"
                || case.expected.authorization_decision != "allow"
                || case.expected.effect_adapter_invocations_for_case != 1
                || case.setup.successful_effects_before_case != 0
                || case.expected.terminal_class != expected_terminal
                || case.mutations.is_empty()
            {
                bail!("lifecycle case {} must distinguish valid authorization from its exact later effect/outcome/evidence failure", case.id);
            }
        }
        _ => bail!("case {} must use a POS-, NEG-, IND-, or LIFE- prefix", case.id),
    }

    if case.id.starts_with("LIFE-") && case.mutations.is_empty() {
        bail!("lifecycle case {} must explicitly model its failure condition", case.id);
    }
    if case.id.starts_with("NEG-")
        && case.id != "NEG-025"
        && !matches!(case.id.as_str(), "NEG-024" | "NEG-026" | "NEG-027")
        && !case.expected.terminal_class.ends_with("_before_effect")
    {
        bail!("negative control {} uses a terminal class inconsistent with pre-effect denial", case.id);
    }
    Ok(())
}

fn is_field_path(value: &str) -> bool {
    !value.is_empty()
        && value.split('.').all(|segment| {
            !segment.is_empty()
                && segment.bytes().all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
        })
}

fn require_nonempty(field: &str, value: &str) -> Result<()> {
    if value.trim().is_empty() {
        bail!("{field} must not be empty");
    }
    Ok(())
}

fn ensure_nonempty_list<T>(field: &str, values: &[T]) -> Result<()> {
    if values.is_empty() {
        bail!("{field} must not be empty");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{Case, Expected, Mutation, Setup, validate_case};
    use serde_json::json;
    use sha2::Digest;

    fn sample_case(id: &str, category: &str, decision: &str, calls: u64, witness: &str, terminal: &str) -> Case {
        Case {
            id: id.to_string(),
            category: category.to_string(),
            description: "test case".to_string(),
            fixture: "baseline".to_string(),
            setup: Setup {
                successful_effects_before_case: 0,
                consumed_capability_persisted: None,
                one_shared_capability: None,
            },
            mutations: Vec::new(),
            expected: Expected {
                authorization_decision: decision.to_string(),
                effect_adapter_invocations_for_case: calls,
                effect_witness: witness.to_string(),
                terminal_class: terminal.to_string(),
            },
        }
    }

    #[test]
    fn committed_v1_corpus_matches_the_frozen_contract() {
        let raw = include_bytes!("../../docs/security/effect-authorization-conformance.v1.json");
        let manifest: super::Manifest = serde_json::from_slice(raw).unwrap();
        let digest = format!("{:x}", sha2::Sha256::digest(raw));
        let result = super::validate(&manifest, digest);
        assert!(result.is_ok(), "committed v1 corpus failed structural validation: {result:?}");
    }

    #[test]
    fn positive_control_cannot_pass_with_no_effect_or_missing_postcondition() {
        let good = sample_case("POS-001", "positive_control", "allow", 1, "exactly_one", "postcondition_confirmed");
        assert!(validate_case(&good).is_ok());

        let no_effect = sample_case("POS-001", "positive_control", "allow", 0, "none", "postcondition_confirmed");
        assert!(validate_case(&no_effect).is_err());

        let no_postcondition = sample_case("POS-001", "positive_control", "allow", 1, "exactly_one", "exit_code_zero");
        assert!(validate_case(&no_postcondition).is_err());
    }

    #[test]
    fn mutation_value_cannot_be_changed_while_case_id_stays_fixed() {
        let mut case = sample_case("NEG-001", "credential_integrity", "deny", 0, "none", "denied_before_effect");
        case.mutations.push(Mutation { path: "credential.signature_valid".to_string(), value: json!(true) });
        assert!(validate_case(&case).is_err());
    }

    #[test]
    fn frozen_case_category_cannot_be_relabelled() {
        let mut case = sample_case("NEG-001", "credential_integrity", "deny", 0, "none", "denied_before_effect");
        case.mutations.push(Mutation { path: "credential.signature_valid".to_string(), value: json!(false) });
        assert!(validate_case(&case).is_ok());
        case.category = "benign".to_string();
        assert!(validate_case(&case).is_err());
    }

    #[test]
    fn negative_case_cannot_silently_ignore_a_mutation() {
        let mut case = sample_case("NEG-001", "credential_integrity", "deny", 0, "none", "denied_before_effect");
        case.mutations.push(Mutation { path: "credential.signature_valid".to_string(), value: json!(false) });
        assert!(validate_case(&case).is_ok());
        case.mutations.clear();
        assert!(validate_case(&case).is_err());
    }

    #[test]
    fn duplicate_or_unparseable_mutation_paths_are_rejected() {
        let mut case = sample_case("NEG-001", "credential_integrity", "deny", 0, "none", "denied_before_effect");
        case.mutations = vec![
            Mutation { path: "credential.signature_valid".to_string(), value: json!(false) },
            Mutation { path: "credential.signature_valid".to_string(), value: json!(true) },
        ];
        assert!(validate_case(&case).is_err());
        case.mutations = vec![Mutation { path: "credential..valid".to_string(), value: json!(false) }];
        assert!(validate_case(&case).is_err());
    }

    #[test]
    fn denied_case_cannot_expect_an_effect() {
        let mut case = sample_case("NEG-001", "credential_integrity", "deny", 1, "exactly_one", "denied_before_effect");
        case.mutations.push(Mutation { path: "credential.signature_valid".to_string(), value: json!(false) });
        assert!(validate_case(&case).is_err());
    }

    #[test]
    fn lifecycle_case_does_not_claim_success_just_from_authorization() {
        let mut case = sample_case("LIFE-001", "lifecycle", "allow", 1, "unknown_until_reconciled", "outcome_unknown_no_automatic_replay");
        case.mutations.push(Mutation { path: "effect.result".to_string(), value: json!("timeout_after_possible_commit") });
        assert!(validate_case(&case).is_ok());
        case.expected.terminal_class = "postcondition_confirmed".to_string();
        assert!(validate_case(&case).is_err());
    }
}
