use jsonschema::{Validator, draft202012};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

const SCHEMA_ID: &str =
    "https://luminousdynamics.io/schemas/transport/ev_soh_adaptation_portfolio/0.1";
const CORPUS_VERSION: &str = "ev-soh-adaptation-portfolio-corpus/0.1";
const PROTOCOL_VERSION: &str = "ev-soh-adaptation-portfolio/0.1";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Corpus {
    corpus_version: String,
    schema_id: String,
    cases: Vec<CorpusCase>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CorpusCase {
    case_id: String,
    expectation: String,
    manifest: Value,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    protocol_version: String,
    portfolio_id: String,
    parent_benchmark_manifest_digest: String,
    dataset_digest: String,
    split_manifest_digest: String,
    evaluation_mode: String,
    split_frozen_at_ms: u64,
    test_accessed_at_ms: u64,
    source_domain_group_identity_digests: Vec<String>,
    target_domain: TargetDomain,
    router: Router,
    experts: Vec<Expert>,
    selected_expert_id: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TargetDomain {
    domain_id: String,
    adaptation_group_identity_digests: Vec<String>,
    evaluation_group_identity_digests: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Router {
    router_ref: String,
    router_digest: String,
    feature_schema_digest: String,
    training_group_identity_digests: Vec<String>,
    fit_scope: String,
    uses_target_evaluation_labels: bool,
    selection_time_ms: u64,
    selection_receipt_digest: String,
    features: Vec<RouterFeature>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RouterFeature {
    feature_id: String,
    feature_scope: String,
    available_at_ms: u64,
    requires_target_labels: bool,
    derived_from_target_labels: bool,
    evidence_digest: String,
    evidence_class: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Expert {
    expert_id: String,
    strategy: String,
    model_digest: String,
    target_adaptation_group_identity_digests: Vec<String>,
    unlabeled_target_group_budget: usize,
    labeled_target_group_budget: usize,
    labeled_target_evidence_digest: Option<String>,
    minimum_support_group_count: usize,
    max_updates: u64,
    adaptation_completed_at_ms: u64,
    wall_time_ms: u64,
    peak_memory_bytes: u64,
    energy_wh: Option<f64>,
    evaluation_group_identity_digests: Vec<String>,
    selection_status: String,
    result_digest: Option<String>,
}

#[derive(Debug, Serialize)]
struct CaseAudit {
    case_id: String,
    expectation: String,
    accepted: bool,
    errors: Vec<String>,
    portfolio_id: Option<String>,
    selected_expert_id: Option<String>,
    selected_strategy: Option<String>,
    selected_unlabeled_budget: usize,
    selected_labeled_budget: usize,
    selected_adaptation_completion_ms: Option<u64>,
    candidate_count: usize,
    evaluation_group_count: usize,
    claim_ceiling: String,
}

#[derive(Debug, Serialize)]
struct ReportPayload {
    corpus_version: String,
    schema_id: String,
    schema_sha256: String,
    canonical_corpus_sha256: String,
    case_count: usize,
    accepted_count: usize,
    rejected_count: usize,
    selected_strategy_counts: BTreeMap<String, usize>,
    cases: Vec<CaseAudit>,
}

#[derive(Debug, Serialize)]
pub struct CorpusReport {
    #[serde(flatten)]
    payload: ReportPayload,
    report_sha256: String,
}

fn sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn is_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn valid_evidence_class(value: &str) -> bool {
    matches!(
        value,
        "physical_observation"
            | "derived_from_physical"
            | "model_prediction"
            | "simulation"
            | "synthetic_fixture"
            | "vendor_claim"
    )
}

fn schema_validator(bytes: &[u8]) -> Result<Validator, String> {
    let schema: Value =
        serde_json::from_slice(bytes).map_err(|error| format!("schema JSON: {error}"))?;
    if schema.get("$schema").and_then(Value::as_str)
        != Some("https://json-schema.org/draft/2020-12/schema")
    {
        return Err("schema must declare JSON Schema Draft 2020-12".to_owned());
    }
    if schema.get("$id").and_then(Value::as_str) != Some(SCHEMA_ID) {
        return Err("schema ID does not match SOH adaptation portfolio v0.1".to_owned());
    }
    if !draft202012::meta::is_valid(&schema) {
        return Err("schema fails Draft 2020-12 meta-schema validation".to_owned());
    }
    draft202012::options()
        .should_validate_formats(true)
        .build(&schema)
        .map_err(|error| format!("schema compilation: {error}"))
}

fn digest_set(values: &[String], label: &str, errors: &mut Vec<String>) -> BTreeSet<String> {
    let mut output = BTreeSet::new();
    for value in values {
        if !is_digest(value) || !output.insert(value.clone()) {
            errors.push(format!("{label} contains an invalid or duplicate physical identity digest"));
        }
    }
    output
}

fn validate_manifest(manifest: &Manifest) -> Vec<String> {
    let mut errors = Vec::new();
    if manifest.protocol_version != PROTOCOL_VERSION || !nonempty(&manifest.portfolio_id) {
        errors.push("portfolio protocol version or identity is invalid".to_owned());
    }
    if !is_digest(&manifest.parent_benchmark_manifest_digest)
        || !is_digest(&manifest.dataset_digest)
        || !is_digest(&manifest.split_manifest_digest)
    {
        errors.push("parent benchmark, dataset or split digest is invalid".to_owned());
    }
    if !matches!(
        manifest.evaluation_mode.as_str(),
        "unseen_cell" | "unseen_pack" | "unseen_vehicle" | "unseen_lab" | "unseen_domain"
    ) {
        errors.push("adaptation portfolio requires an explicitly group-disjoint unseen-asset/domain claim".to_owned());
    }
    if manifest.split_frozen_at_ms == 0
        || manifest.test_accessed_at_ms <= manifest.split_frozen_at_ms
    {
        errors.push("split-freeze and first-test-access chronology is invalid".to_owned());
    }

    let source = digest_set(
        &manifest.source_domain_group_identity_digests,
        "source-domain groups",
        &mut errors,
    );
    let adaptation_pool = digest_set(
        &manifest.target_domain.adaptation_group_identity_digests,
        "target adaptation groups",
        &mut errors,
    );
    let evaluation = digest_set(
        &manifest.target_domain.evaluation_group_identity_digests,
        "target evaluation groups",
        &mut errors,
    );
    if source.len() < 2 || !nonempty(&manifest.target_domain.domain_id) {
        errors.push("source domains and target-domain identity must be explicit".to_owned());
    }
    if evaluation.len() < 2 {
        errors.push("final target evaluation requires at least two physical groups".to_owned());
    }
    if !source.is_disjoint(&adaptation_pool)
        || !source.is_disjoint(&evaluation)
        || !adaptation_pool.is_disjoint(&evaluation)
    {
        errors.push("source, target adaptation and final evaluation physical groups must be pairwise disjoint".to_owned());
    }

    let router = &manifest.router;
    let router_training = digest_set(
        &router.training_group_identity_digests,
        "router training groups",
        &mut errors,
    );
    if !nonempty(&router.router_ref)
        || !is_digest(&router.router_digest)
        || !is_digest(&router.feature_schema_digest)
        || router.fit_scope != "source_domains_only"
        || router.uses_target_evaluation_labels
        || !is_digest(&router.selection_receipt_digest)
        || router.selection_time_ms <= manifest.split_frozen_at_ms
        || router.selection_time_ms >= manifest.test_accessed_at_ms
        || router_training.len() < 2
        || !router_training.is_subset(&source)
        || !router_training.is_disjoint(&adaptation_pool)
        || !router_training.is_disjoint(&evaluation)
    {
        errors.push("router must be source-domain fitted, target-evaluation-label blind, and chronologically separated from test access".to_owned());
    }
    if router.features.is_empty() {
        errors.push("router requires at least one declared selection feature".to_owned());
    }
    let mut feature_ids = BTreeSet::new();
    for feature in &router.features {
        if !nonempty(&feature.feature_id)
            || !feature_ids.insert(feature.feature_id.clone())
            || !matches!(
                feature.feature_scope.as_str(),
                "fleet_metadata"
                    | "sensor_prefix"
                    | "model_internal_uncertainty"
                    | "reference_test_measurement"
            )
            || feature.available_at_ms == 0
            || feature.available_at_ms > router.selection_time_ms
            || feature.requires_target_labels
            || feature.derived_from_target_labels
            || !is_digest(&feature.evidence_digest)
            || !valid_evidence_class(&feature.evidence_class)
        {
            errors.push(format!(
                "router feature {} is duplicated, unavailable at selection, label-dependent or unprovenanced",
                feature.feature_id
            ));
        }
    }

    let mut expert_ids = BTreeSet::new();
    let mut selected = Vec::new();
    let mut abstentions = Vec::new();
    for expert in &manifest.experts {
        if !nonempty(&expert.expert_id) || !expert_ids.insert(expert.expert_id.clone()) {
            errors.push("expert IDs must be non-empty and unique".to_owned());
        }
        if !is_digest(&expert.model_digest) {
            errors.push(format!("expert {} model digest is invalid", expert.expert_id));
        }
        let adaptation_ids = digest_set(
            &expert.target_adaptation_group_identity_digests,
            &format!("expert {} adaptation groups", expert.expert_id),
            &mut errors,
        );
        let expert_evaluation = digest_set(
            &expert.evaluation_group_identity_digests,
            &format!("expert {} evaluation groups", expert.expert_id),
            &mut errors,
        );
        if expert_evaluation != evaluation {
            errors.push(format!(
                "expert {} does not use the exact common final evaluation group set",
                expert.expert_id
            ));
        }
        if !adaptation_ids.is_subset(&adaptation_pool)
            || !adaptation_ids.is_disjoint(&source)
            || !adaptation_ids.is_disjoint(&evaluation)
        {
            errors.push(format!(
                "expert {} target adaptation groups must come from the declared target support pool and remain disjoint from source/test groups",
                expert.expert_id
            ));
        }
        if expert.unlabeled_target_group_budget + expert.labeled_target_group_budget
            != adaptation_ids.len()
        {
            errors.push(format!(
                "expert {} labeled/unlabeled budgets must account for every identified adaptation group exactly once",
                expert.expert_id
            ));
        }
        if expert.adaptation_completed_at_ms < router.selection_time_ms
            || expert.adaptation_completed_at_ms >= manifest.test_accessed_at_ms
        {
            errors.push(format!(
                "expert {} must be ready after router selection and before final test access",
                expert.expert_id
            ));
        }

        match expert.strategy.as_str() {
            "zero_shot" | "physical_statistical" => {
                if !adaptation_ids.is_empty()
                    || expert.unlabeled_target_group_budget != 0
                    || expert.labeled_target_group_budget != 0
                    || expert.labeled_target_evidence_digest.is_some()
                    || expert.minimum_support_group_count != 0
                    || expert.max_updates != 0
                    || expert.adaptation_completed_at_ms != router.selection_time_ms
                {
                    errors.push(format!(
                        "{} expert {} must declare zero target adaptation and zero updates",
                        expert.strategy, expert.expert_id
                    ));
                }
            }
            "unlabeled_tta" => {
                if expert.unlabeled_target_group_budget == 0
                    || expert.labeled_target_group_budget != 0
                    || expert.labeled_target_evidence_digest.is_some()
                    || adaptation_ids.len() != expert.unlabeled_target_group_budget
                    || expert.max_updates == 0
                    || expert.minimum_support_group_count != 0
                {
                    errors.push(format!(
                        "unlabeled TTA expert {} requires identified unlabeled target groups, zero target labels and a nonzero update budget",
                        expert.expert_id
                    ));
                }
            }
            "supervised_finetune" => {
                if expert.labeled_target_group_budget == 0
                    || expert.unlabeled_target_group_budget != 0
                    || adaptation_ids.len() != expert.labeled_target_group_budget
                    || expert
                        .labeled_target_evidence_digest
                        .as_deref()
                        .is_none_or(|digest| !is_digest(digest))
                    || expert.max_updates == 0
                    || expert.minimum_support_group_count != 0
                {
                    errors.push(format!(
                        "supervised fine-tuning expert {} requires labeled target group IDs, label evidence and a nonzero update budget",
                        expert.expert_id
                    ));
                }
            }
            "few_shot_maml" => {
                if expert.minimum_support_group_count < 2
                    || expert.labeled_target_group_budget < expert.minimum_support_group_count
                    || adaptation_ids.len() != expert.labeled_target_group_budget
                    || expert.unlabeled_target_group_budget != 0
                    || expert
                        .labeled_target_evidence_digest
                        .as_deref()
                        .is_none_or(|digest| !is_digest(digest))
                    || expert.max_updates == 0
                {
                    errors.push(format!(
                        "few-shot MAML expert {} does not meet its declared labeled support minimum",
                        expert.expert_id
                    ));
                }
            }
            "abstain" => {
                if !adaptation_ids.is_empty()
                    || expert.unlabeled_target_group_budget != 0
                    || expert.labeled_target_group_budget != 0
                    || expert.labeled_target_evidence_digest.is_some()
                    || expert.minimum_support_group_count != 0
                    || expert.max_updates != 0
                    || expert.result_digest.is_some()
                    || expert.selection_status != "abstain"
                    || expert.adaptation_completed_at_ms != router.selection_time_ms
                {
                    errors.push("abstain expert must not claim adaptation, updates or a model result".to_owned());
                }
                abstentions.push(expert);
            }
            _ => errors.push(format!("expert {} has an unknown strategy", expert.expert_id)),
        }

        if expert.wall_time_ms == 0 && expert.max_updates > 0 {
            errors.push(format!("expert {} has updates but reports zero wall time", expert.expert_id));
        }
        if expert.peak_memory_bytes == 0 && expert.max_updates > 0 {
            errors.push(format!("expert {} has updates but reports zero peak memory", expert.expert_id));
        }
        if expert.strategy != "abstain"
            && expert
                .result_digest
                .as_deref()
                .is_none_or(|digest| !is_digest(digest))
        {
            errors.push(format!("expert {} is missing its matched evaluation result digest", expert.expert_id));
        }
        match expert.selection_status.as_str() {
            "selected" => selected.push(expert),
            "not_selected" => {}
            "abstain" if expert.strategy == "abstain" => {}
            _ => errors.push(format!("expert {} selection status is inconsistent with its strategy", expert.expert_id)),
        }
    }

    if selected.len() == 1 {
        if selected[0].expert_id != manifest.selected_expert_id {
            errors.push("selected_expert_id does not identify the sole selected expert".to_owned());
        }
        if !abstentions.is_empty() {
            errors.push("an abstain candidate cannot coexist with a selected predictive expert in v0.1".to_owned());
        }
    } else if selected.is_empty() {
        if abstentions.len() != 1
            || abstentions[0].expert_id != manifest.selected_expert_id
        {
            errors.push("portfolio must select exactly one predictive expert or one explicit abstain result".to_owned());
        }
    } else {
        errors.push("portfolio must select exactly one expert".to_owned());
    }

    errors
}

fn canonicalize_corpus(value: &mut Value) {
    let Some(cases) = value.get_mut("cases").and_then(Value::as_array_mut) else {
        return;
    };
    cases.sort_by(|left, right| {
        left.get("case_id")
            .and_then(Value::as_str)
            .cmp(&right.get("case_id").and_then(Value::as_str))
    });
    for case in cases {
        if let Some(manifest) = case.get_mut("manifest") {
            for field in [
                "source_domain_group_identity_digests",
                "target_domain.adaptation_group_identity_digests",
                "target_domain.evaluation_group_identity_digests",
                "router.training_group_identity_digests",
            ] {
                let (parent, key) = match field.rsplit_once('.') {
                    Some((parent, key)) => (
                        manifest
                            .get_mut(parent)
                            .and_then(Value::as_object_mut),
                        key,
                    ),
                    None => (manifest.as_object_mut(), field),
                };
                if let Some(rows) = parent
                    .and_then(|object| object.get_mut(key))
                    .and_then(Value::as_array_mut)
                {
                    rows.sort_by(|left, right| left.as_str().cmp(&right.as_str()));
                }
            }
            if let Some(features) = manifest
                .get_mut("router")
                .and_then(|router| router.get_mut("features"))
                .and_then(Value::as_array_mut)
            {
                features.sort_by(|left, right| {
                    left.get("feature_id")
                        .and_then(Value::as_str)
                        .cmp(&right.get("feature_id").and_then(Value::as_str))
                });
            }
            if let Some(experts) = manifest.get_mut("experts").and_then(Value::as_array_mut) {
                experts.sort_by(|left, right| {
                    left.get("expert_id")
                        .and_then(Value::as_str)
                        .cmp(&right.get("expert_id").and_then(Value::as_str))
                });
                for expert in experts {
                    for field in [
                        "target_adaptation_group_identity_digests",
                        "evaluation_group_identity_digests",
                    ] {
                        if let Some(rows) = expert.get_mut(field).and_then(Value::as_array_mut) {
                            rows.sort_by(|left, right| left.as_str().cmp(&right.as_str()));
                        }
                    }
                }
            }
        }
    }
}

fn selected_summary(
    manifest_value: &Value,
) -> (
    Option<String>,
    Option<String>,
    usize,
    usize,
    Option<u64>,
    usize,
    usize,
    String,
) {
    let Some(manifest) = manifest_value.as_object() else {
        return (None, None, 0, 0, None, 0, 0, "invalid_manifest".to_owned());
    };
    let selected_id = manifest
        .get("selected_expert_id")
        .and_then(Value::as_str)
        .map(str::to_owned);
    let experts = manifest.get("experts").and_then(Value::as_array);
    let candidate_count = experts.map_or(0, Vec::len);
    let evaluation_count = manifest
        .get("target_domain")
        .and_then(|target| target.get("evaluation_group_identity_digests"))
        .and_then(Value::as_array)
        .map_or(0, Vec::len);
    if let (Some(selected_id), Some(experts)) = (selected_id.as_ref(), experts) {
        if let Some(expert) = experts.iter().find(|expert| {
            expert.get("expert_id").and_then(Value::as_str) == Some(selected_id)
        }) {
            let strategy = expert
                .get("strategy")
                .and_then(Value::as_str)
                .map(str::to_owned);
            let unlabeled =
                expert.get("unlabeled_target_group_budget").and_then(Value::as_u64).unwrap_or(0)
                    as usize;
            let labeled =
                expert.get("labeled_target_group_budget").and_then(Value::as_u64).unwrap_or(0)
                    as usize;
            let completion = expert
                .get("adaptation_completed_at_ms")
                .and_then(Value::as_u64);
            let ceiling = if strategy.as_deref() == Some("abstain") {
                "no eligible expert; explicit abstention"
            } else if labeled > 0 {
                "target-labeled adaptation; not a zero-shot result"
            } else if unlabeled > 0 {
                "unlabeled target adaptation; not a zero-shot result"
            } else {
                "zero target adaptation"
            };
            return (
                Some(selected_id.clone()),
                strategy,
                unlabeled,
                labeled,
                completion,
                candidate_count,
                evaluation_count,
                format!("{ceiling}; {evaluation_count} common evaluation groups"),
            );
        }
    }
    (
        selected_id,
        None,
        0,
        0,
        None,
        candidate_count,
        evaluation_count,
        format!("unresolved selection; {evaluation_count} evaluation groups"),
    )
}

pub fn audit_corpus(schema_bytes: &[u8], corpus_bytes: &[u8]) -> Result<CorpusReport, String> {
    let validator = schema_validator(schema_bytes)?;
    let mut corpus_value: Value =
        serde_json::from_slice(corpus_bytes).map_err(|error| format!("corpus JSON: {error}"))?;
    canonicalize_corpus(&mut corpus_value);
    let canonical = serde_json::to_vec(&corpus_value)
        .map_err(|error| format!("canonical corpus serialization: {error}"))?;
    let corpus: Corpus =
        serde_json::from_slice(&canonical).map_err(|error| format!("typed corpus: {error}"))?;
    if corpus.corpus_version != CORPUS_VERSION || corpus.schema_id != SCHEMA_ID {
        return Err("corpus version or schema identity mismatch".to_owned());
    }

    let mut seen = BTreeSet::new();
    let mut cases = Vec::new();
    let mut accepted_count = 0usize;
    let mut rejected_count = 0usize;
    let mut selected_strategy_counts = BTreeMap::new();

    for case in corpus.cases {
        if !nonempty(&case.case_id) || !seen.insert(case.case_id.clone()) {
            return Err(format!("case identity is empty or duplicated: {}", case.case_id));
        }
        if !matches!(case.expectation.as_str(), "accept" | "reject") {
            return Err(format!("case {} has unknown expectation", case.case_id));
        }

        let schema_errors: Vec<String> = validator
            .iter_errors(&case.manifest)
            .map(|error| error.to_string())
            .collect();
        let outcome = if schema_errors.is_empty() {
            match serde_json::from_value::<Manifest>(case.manifest.clone()) {
                Ok(manifest) => {
                    let errors = validate_manifest(&manifest);
                    if errors.is_empty() {
                        Ok(())
                    } else {
                        Err(errors)
                    }
                }
                Err(error) => Err(vec![format!("typed manifest parse: {error}")]),
            }
        } else {
            Err(schema_errors)
        };

        let accepted = outcome.is_ok();
        if accepted != (case.expectation == "accept") {
            let detail = match &outcome {
                Ok(()) => "unexpected acceptance".to_owned(),
                Err(errors) => errors.join("; "),
            };
            return Err(format!("case {} expectation mismatch: {}", case.case_id, detail));
        }

        let (
            errors,
            selected_expert_id,
            selected_strategy,
            unlabeled,
            labeled,
            completion,
            candidate_count,
            evaluation_group_count,
            claim_ceiling,
        ) = match outcome {
            Ok(()) => {
                accepted_count += 1;
                let (id, strategy, unlabeled, labeled, completion, candidates, evals, ceiling) =
                    selected_summary(&case.manifest);
                if let Some(strategy) = strategy.as_ref() {
                    *selected_strategy_counts.entry(strategy.clone()).or_insert(0) += 1;
                }
                (Vec::new(), id, strategy, unlabeled, labeled, completion, candidates, evals, ceiling)
            }
            Err(errors) => {
                rejected_count += 1;
                let (id, strategy, unlabeled, labeled, completion, candidates, evals, ceiling) =
                    selected_summary(&case.manifest);
                (errors, id, strategy, unlabeled, labeled, completion, candidates, evals, ceiling)
            }
        };

        cases.push(CaseAudit {
            case_id: case.case_id,
            expectation: case.expectation,
            accepted,
            errors,
            portfolio_id: case.manifest.get("portfolio_id").and_then(Value::as_str).map(str::to_owned),
            selected_expert_id,
            selected_strategy,
            selected_unlabeled_budget: unlabeled,
            selected_labeled_budget: labeled,
            selected_adaptation_completion_ms: completion,
            candidate_count,
            evaluation_group_count,
            claim_ceiling,
        });
    }

    let payload = ReportPayload {
        corpus_version: CORPUS_VERSION.to_owned(),
        schema_id: SCHEMA_ID.to_owned(),
        schema_sha256: sha256(schema_bytes),
        canonical_corpus_sha256: sha256(&canonical),
        case_count: cases.len(),
        accepted_count,
        rejected_count,
        selected_strategy_counts,
        cases,
    };
    let bytes =
        serde_json::to_vec(&payload).map_err(|error| format!("report serialization: {error}"))?;
    Ok(CorpusReport {
        payload,
        report_sha256: sha256(&bytes),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    const SCHEMA: &[u8] =
        include_bytes!("../../../schemas/transport/ev_soh_adaptation_portfolio_v0_1.schema.json");
    const CORPUS: &[u8] =
        include_bytes!("../../../fixtures/ev_battery/soh_adaptation_portfolios_v0_1.json");

    fn report() -> CorpusReport {
        audit_corpus(SCHEMA, CORPUS).expect("frozen SOH adaptation portfolio corpus")
    }

    #[test]
    fn frozen_corpus_accepts_five_and_rejects_eight_controls() {
        let report = report();
        assert_eq!(report.payload.case_count, 16);
        assert_eq!(report.payload.accepted_count, 5);
        assert_eq!(report.payload.rejected_count, 11);
    }

    #[test]
    fn adaptation_strategies_have_separate_explicit_target_budgets() {
        let report = report();
        for (case_id, strategy, unlabeled, labeled) in [
            ("zero-shot-selected", "zero_shot", 0, 0),
            ("unlabeled-tta-selected", "unlabeled_tta", 2, 0),
            ("supervised-finetune-selected", "supervised_finetune", 0, 2),
            ("few-shot-maml-selected", "few_shot_maml", 0, 2),
        ] {
            let case = report.payload.cases.iter().find(|case| case.case_id == case_id).unwrap();
            assert!(case.accepted);
            assert_eq!(case.selected_strategy.as_deref(), Some(strategy));
            assert_eq!(case.selected_unlabeled_budget, unlabeled);
            assert_eq!(case.selected_labeled_budget, labeled);
        }
    }

    #[test]
    fn adaptation_completion_precedes_final_test_access() {
        let report = report();
        let case = report.payload.cases.iter()
            .find(|case| case.case_id == "unlabeled-tta-selected")
            .expect("TTA selected case");
        assert_eq!(case.selected_adaptation_completion_ms, Some(1100));
        assert!(case.accepted);
    }

    #[test]
    fn router_features_must_be_available_before_selection() {
        let report = report();
        let case = report.payload.cases.iter()
            .find(|case| case.case_id == "feature-after-selection-time")
            .expect("late feature control");
        assert!(!case.accepted);
    }

    #[test]
    fn every_candidate_uses_the_same_unseen_domain_evaluation_groups() {
        let report = report();
        let case = report.payload.cases.iter()
            .find(|case| case.case_id == "expert-evaluation-splits-differ")
            .expect("evaluation mismatch control");
        assert!(!case.accepted);
    }

    #[test]
    fn labeled_and_unlabeled_budget_misrepresentation_is_rejected() {
        let report = report();
        for case_id in [
            "zero-shot-has-target-budget",
            "unlabeled-tta-claims-labels",
            "fine-tuning-missing-label-evidence",
        ] {
            let case = report.payload.cases.iter().find(|case| case.case_id == case_id).unwrap();
            assert!(!case.accepted);
        }
    }

    #[test]
    fn adaptation_groups_cannot_overlap_final_evaluation_groups() {
        let report = report();
        let case = report.payload.cases.iter()
            .find(|case| case.case_id == "target-adaptation-overlaps-evaluation")
            .expect("adaptation/test overlap control");
        assert!(!case.accepted);
    }

    #[test]
    fn maml_requires_its_declared_minimum_labeled_support() {
        let report = report();
        let case = report.payload.cases.iter()
            .find(|case| case.case_id == "maml-below-minimum-support")
            .expect("MAML support control");
        assert!(!case.accepted);
    }

    #[test]
    fn router_must_be_source_fitted_and_test_label_blind() {
        let report = report();
        let case = report.payload.cases.iter()
            .find(|case| case.case_id == "router-uses-test-labels")
            .expect("router test-label leakage control");
        assert!(!case.accepted);
    }

    #[test]
    fn no_eligible_expert_requires_explicit_abstention() {
        let report = report();
        let case = report.payload.cases.iter()
            .find(|case| case.case_id == "no-eligible-expert-abstains")
            .expect("explicit abstention");
        assert!(case.accepted);
        assert_eq!(case.selected_strategy.as_deref(), Some("abstain"));
        assert_eq!(case.selected_expert_id.as_deref(), Some("abstain"));
    }

    #[test]
    fn router_features_must_not_be_derived_from_target_labels() {
        let report = report();
        let case = report.payload.cases.iter()
            .find(|case| case.case_id == "router-feature-derived-from-target-labels")
            .expect("target-derived router feature control");
        assert!(!case.accepted);
    }

    #[test]
    fn expert_adaptation_must_finish_before_final_test_access() {
        let report = report();
        let case = report.payload.cases.iter()
            .find(|case| case.case_id == "expert-adaptation-completes-after-test-access")
            .expect("late expert readiness control");
        assert!(!case.accepted);
    }

    #[test]
    fn explicit_abstention_cannot_claim_a_prediction_result() {
        let report = report();
        let case = report.payload.cases.iter()
            .find(|case| case.case_id == "abstain-expert-claims-prediction-result")
            .expect("abstain result digest control");
        assert!(!case.accepted);
    }

    #[test]
    fn canonical_report_is_invariant_to_case_expert_and_identity_order() {
        let original = report();
        let mut value: Value = serde_json::from_slice(CORPUS).expect("corpus JSON");
        value["cases"].as_array_mut().unwrap().reverse();
        for case in value["cases"].as_array_mut().unwrap() {
            let manifest = &mut case["manifest"];
            manifest["source_domain_group_identity_digests"].as_array_mut().unwrap().reverse();
            manifest["target_domain"]["adaptation_group_identity_digests"].as_array_mut().unwrap().reverse();
            manifest["target_domain"]["evaluation_group_identity_digests"].as_array_mut().unwrap().reverse();
            manifest["router"]["training_group_identity_digests"].as_array_mut().unwrap().reverse();
            manifest["router"]["features"].as_array_mut().unwrap().reverse();
            manifest["experts"].as_array_mut().unwrap().reverse();
            for expert in manifest["experts"].as_array_mut().unwrap() {
                expert["target_adaptation_group_identity_digests"].as_array_mut().unwrap().reverse();
                expert["evaluation_group_identity_digests"].as_array_mut().unwrap().reverse();
            }
        }
        let permuted = serde_json::to_vec(&value).expect("permuted corpus");
        let replay = audit_corpus(SCHEMA, &permuted).expect("permuted adaptation audit");
        assert_eq!(original.report_sha256, replay.report_sha256);
        assert_eq!(
            original.payload.canonical_corpus_sha256,
            replay.payload.canonical_corpus_sha256
        );
    }
}
