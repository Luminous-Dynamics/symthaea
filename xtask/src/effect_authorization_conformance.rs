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
