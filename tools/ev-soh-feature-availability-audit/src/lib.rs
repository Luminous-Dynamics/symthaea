use jsonschema::{Validator, draft202012};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

const SCHEMA_ID: &str =
    "https://luminousdynamics.io/schemas/transport/ev-soh-feature-availability/0.1";
const CORPUS_VERSION: &str = "ev-soh-feature-availability-corpus/0.1";
const PROTOCOL_VERSION: &str = "ev-soh-feature-availability/0.1";

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
    expected_claim_ceiling: Option<String>,
    manifest: Value,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    protocol_version: String,
    manifest_id: String,
    evaluation_mode: String,
    dataset_digest: String,
    model_digest: String,
    split_manifest_digest: String,
    split_frozen_at_ms: u64,
    test_accessed_at_ms: u64,
    train_group_identity_digests: Vec<String>,
    test_group_identity_digests: Vec<String>,
    feature_definitions: Vec<FeatureDefinition>,
    prediction_rows: Vec<PredictionRow>,
    reference_randomization: ReferenceRandomization,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FeatureDefinition {
    feature_id: String,
    feature_schema_digest: String,
    implementation_digest: String,
    transform_ref: String,
    transform_digest: String,
    fit_scope: String,
    fit_at_ms: u64,
    fit_group_identity_digests: Vec<String>,
    predeclared_at_ms: Option<u64>,
    predeclaration_digest: Option<String>,
    reference_kind: String,
    reference_policy_digest: String,
    uses_test_labels: bool,
    uses_future_observations: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PredictionRow {
    sample_digest: String,
    asset_identity_digest: String,
    decision_at_ms: u64,
    target_value: f64,
    target_evidence_digest: String,
    feature_uses: Vec<FeatureUse>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FeatureUse {
    feature_id: String,
    value_digest: String,
    latest_measurement_at_ms: u64,
    latest_received_at_ms: u64,
    derived_at_ms: u64,
    reference_kind: String,
    reference_identity_digest: Option<String>,
    reference_population_digest: Option<String>,
    reference_measurement_at_ms: Option<u64>,
    reference_received_at_ms: Option<u64>,
    used_target_label: bool,
    used_future_observation: bool,
    availability_evidence_digest: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReferenceRandomization {
    status: String,
    method_ref: String,
    method_digest: String,
    assignment_digest: Option<String>,
    seed: Option<u64>,
    derangement_count: usize,
    schedule_frozen_at_ms: Option<u64>,
    arms: Vec<String>,
}

#[derive(Debug, Serialize)]
struct CaseAudit {
    case_id: String,
    expectation: String,
    accepted: bool,
    errors: Vec<String>,
    claim_ceiling: String,
    prediction_count: usize,
    feature_definition_count: usize,
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
    strict_online_eligible_count: usize,
    offline_oracle_only_count: usize,
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

fn schema_validator(bytes: &[u8]) -> Result<Validator, String> {
    let schema: Value =
        serde_json::from_slice(bytes).map_err(|error| format!("schema JSON: {error}"))?;
    if schema.get("$schema").and_then(Value::as_str)
        != Some("https://json-schema.org/draft/2020-12/schema")
    {
        return Err("schema must declare JSON Schema Draft 2020-12".to_owned());
    }
    if schema.get("$id").and_then(Value::as_str) != Some(SCHEMA_ID) {
        return Err("schema identity does not match feature availability v0.1".to_owned());
    }
    if !draft202012::meta::is_valid(&schema) {
        return Err("schema fails Draft 2020-12 meta-schema validation".to_owned());
    }
    draft202012::options()
        .should_validate_formats(true)
        .build(&schema)
        .map_err(|error| format!("schema compilation: {error}"))
}

fn unique_digest_set(
    values: &[String],
    label: &str,
    errors: &mut Vec<String>,
) -> BTreeSet<String> {
    let mut output = BTreeSet::new();
    for value in values {
        if !is_digest(value) || !output.insert(value.clone()) {
            errors.push(format!("{label} contains an invalid or duplicate digest"));
        }
    }
    output
}

fn validate_manifest(manifest: &Manifest) -> (Vec<String>, String) {
    let mut errors = Vec::new();
    let mut offline_only = manifest.evaluation_mode == "offline_oracle";

    if manifest.protocol_version != PROTOCOL_VERSION || !nonempty(&manifest.manifest_id) {
        errors.push("protocol version or manifest identity is invalid".to_owned());
    }
    if !matches!(manifest.evaluation_mode.as_str(), "strict_online" | "offline_oracle") {
        errors.push("unknown evaluation mode".to_owned());
    }
    if !is_digest(&manifest.dataset_digest)
        || !is_digest(&manifest.model_digest)
        || !is_digest(&manifest.split_manifest_digest)
    {
        errors.push("dataset, model or split manifest digest is invalid".to_owned());
    }
    if manifest.split_frozen_at_ms == 0
        || manifest.test_accessed_at_ms <= manifest.split_frozen_at_ms
    {
        errors.push("test access must follow a non-zero frozen split timestamp".to_owned());
    }

    let train = unique_digest_set(
        &manifest.train_group_identity_digests,
        "train identities",
        &mut errors,
    );
    let test = unique_digest_set(
        &manifest.test_group_identity_digests,
        "test identities",
        &mut errors,
    );
    if train.is_empty() || test.is_empty() {
        errors.push("train and test physical identity sets must be non-empty".to_owned());
    }
    if !train.is_disjoint(&test) {
        errors.push("train and test physical identity sets overlap".to_owned());
    }

    let mut definitions = BTreeMap::new();
    for definition in &manifest.feature_definitions {
        if !nonempty(&definition.feature_id)
            || definitions.insert(definition.feature_id.clone(), definition).is_some()
            || !is_digest(&definition.feature_schema_digest)
            || !is_digest(&definition.implementation_digest)
            || !nonempty(&definition.transform_ref)
            || !is_digest(&definition.transform_digest)
            || !is_digest(&definition.reference_policy_digest)
            || definition.fit_at_ms == 0
        {
            errors.push(format!(
                "feature {} has an invalid/duplicate identity or implementation receipt",
                definition.feature_id
            ));
        }

        let fit_groups = unique_digest_set(
            &definition.fit_group_identity_digests,
            "feature transform fit identities",
            &mut errors,
        );
        match definition.fit_scope.as_str() {
            "train_only" => {
                if fit_groups.is_empty()
                    || !fit_groups.is_subset(&train)
                    || !fit_groups.is_disjoint(&test)
                    || definition.fit_at_ms >= manifest.test_accessed_at_ms
                    || definition.predeclared_at_ms.is_some()
                    || definition.predeclaration_digest.is_some()
                {
                    errors.push(format!(
                        "feature {} train-only transform is not bound to development-only assets and time",
                        definition.feature_id
                    ));
                }
            }
            "online_prefix" => {
                if definition.fit_at_ms == 0
                    || !fit_groups.is_empty()
                    || definition.predeclared_at_ms.is_some()
                    || definition.predeclaration_digest.is_some()
                {
                    errors.push(format!(
                        "feature {} online-prefix fit scope must not carry an offline fit population",
                        definition.feature_id
                    ));
                }
            }
            "fixed_predeclared" => {
                if definition.predeclared_at_ms.is_none_or(|time| {
                    time == 0
                        || time >= manifest.split_frozen_at_ms
                        || definition.fit_at_ms > time
                }) || definition.fit_at_ms >= manifest.split_frozen_at_ms
                    || definition
                        .predeclaration_digest
                        .as_deref()
                        .is_none_or(|digest| !is_digest(digest))
                    || !fit_groups.is_empty()
                {
                    errors.push(format!(
                        "feature {} fixed transform was not predeclared before split freeze",
                        definition.feature_id
                    ));
                }
            }
            "full_trajectory" | "test_or_future" => {
                if manifest.evaluation_mode == "strict_online" {
                    errors.push(format!(
                        "feature {} uses a non-deployable fit scope in strict-online mode",
                        definition.feature_id
                    ));
                } else {
                    offline_only = true;
                }
            }
            _ => errors.push(format!("feature {} has an unknown fit scope", definition.feature_id)),
        }

        if definition.uses_test_labels || definition.uses_future_observations {
            if manifest.evaluation_mode == "strict_online" {
                errors.push(format!(
                    "feature {} declares test-label or future-observation dependence",
                    definition.feature_id
                ));
            } else {
                offline_only = true;
            }
        }
        if definition.reference_kind == "full_lifetime_extrema" {
            if manifest.evaluation_mode == "strict_online" {
                errors.push(format!(
                    "feature {} uses future full-lifetime extrema in strict-online mode",
                    definition.feature_id
                ));
            } else {
                offline_only = true;
            }
        }
        if !matches!(
            definition.reference_kind.as_str(),
            "none"
                | "same_asset_prefix"
                | "training_population"
                | "peer_asset"
                | "external_fixed"
                | "full_lifetime_extrema"
        ) {
            errors.push(format!(
                "feature {} has an unknown reference kind",
                definition.feature_id
            ));
        }
    }

    let mut sample_ids = BTreeSet::new();
    let mut peer_reference_used = false;
    for row in &manifest.prediction_rows {
        if !is_digest(&row.sample_digest)
            || !sample_ids.insert(row.sample_digest.clone())
            || !is_digest(&row.asset_identity_digest)
            || !test.contains(&row.asset_identity_digest)
            || row.decision_at_ms <= manifest.split_frozen_at_ms
            || row.decision_at_ms >= manifest.test_accessed_at_ms
            || !row.target_value.is_finite()
            || !(0.0..=1.0).contains(&row.target_value)
            || !is_digest(&row.target_evidence_digest)
        {
            errors.push("prediction row identity, split scope, target or decision-time ordering is invalid".to_owned());
        }

        let mut used = BTreeSet::new();
        for feature_use in &row.feature_uses {
            let Some(definition) = definitions.get(&feature_use.feature_id) else {
                errors.push(format!(
                    "prediction {} uses undefined feature {}",
                    row.sample_digest, feature_use.feature_id
                ));
                continue;
            };
            if !used.insert(feature_use.feature_id.clone()) {
                errors.push(format!(
                    "prediction {} repeats feature {}",
                    row.sample_digest, feature_use.feature_id
                ));
            }
            if !is_digest(&feature_use.value_digest)
                || !is_digest(&feature_use.availability_evidence_digest)
                || feature_use.reference_kind != definition.reference_kind
                || feature_use.latest_measurement_at_ms == 0
                || feature_use.latest_received_at_ms < feature_use.latest_measurement_at_ms
                || feature_use.derived_at_ms < feature_use.latest_received_at_ms
            {
                errors.push(format!(
                    "prediction {} feature {} has invalid value/availability evidence",
                    row.sample_digest, feature_use.feature_id
                ));
            }

            if manifest.evaluation_mode == "strict_online" {
                if feature_use.latest_measurement_at_ms > row.decision_at_ms
                    || feature_use.latest_received_at_ms > row.decision_at_ms
                    || feature_use.derived_at_ms > row.decision_at_ms
                    || feature_use.used_target_label
                    || feature_use.used_future_observation
                    || definition.fit_at_ms > row.decision_at_ms
                {
                    errors.push(format!(
                        "prediction {} feature {} depends on data or computation unavailable at decision time",
                        row.sample_digest, feature_use.feature_id
                    ));
                }
            } else if feature_use.used_target_label
                || feature_use.used_future_observation
                || feature_use.latest_received_at_ms > row.decision_at_ms
                || feature_use.derived_at_ms > row.decision_at_ms
            {
                offline_only = true;
            }

            match feature_use.reference_kind.as_str() {
                "none" => {
                    if feature_use.reference_identity_digest.is_some()
                        || feature_use.reference_population_digest.is_some()
                        || feature_use.reference_measurement_at_ms.is_some()
                        || feature_use.reference_received_at_ms.is_some()
                    {
                        errors.push(format!(
                            "prediction {} feature {} declares reference data for a reference-free feature",
                            row.sample_digest, feature_use.feature_id
                        ));
                    }
                }
                "same_asset_prefix" => {
                    if feature_use.reference_identity_digest.as_deref()
                        != Some(row.asset_identity_digest.as_str())
                        || feature_use.reference_measurement_at_ms.is_none()
                        || feature_use.reference_received_at_ms.is_none()
                        || feature_use.reference_population_digest.is_some()
                        || feature_use.reference_received_at_ms
                            .zip(feature_use.reference_measurement_at_ms)
                            .is_some_and(|(received, measured)| received < measured)
                    {
                        errors.push(format!(
                            "prediction {} feature {} does not bind its same-asset prefix reference",
                            row.sample_digest, feature_use.feature_id
                        ));
                    }
                    if manifest.evaluation_mode == "strict_online"
                        && (feature_use.reference_measurement_at_ms
                            .is_none_or(|time| time > row.decision_at_ms)
                            || feature_use.reference_received_at_ms
                                .is_none_or(|time| time > row.decision_at_ms))
                    {
                        errors.push(format!(
                            "prediction {} feature {} uses a same-asset reference unavailable at decision time",
                            row.sample_digest, feature_use.feature_id
                        ));
                    }
                }
                "training_population" => {
                    if feature_use.reference_population_digest.as_deref().is_none_or(|digest| !is_digest(digest))
                        || feature_use.reference_identity_digest.is_some()
                        || feature_use.reference_measurement_at_ms.is_some()
                        || feature_use.reference_received_at_ms.is_some()
                        || definition.fit_group_identity_digests.is_empty()
                        || !definition
                            .fit_group_identity_digests
                            .iter()
                            .all(|identity| train.contains(identity))
                    {
                        errors.push(format!(
                            "prediction {} feature {} has invalid training-population reference lineage",
                            row.sample_digest, feature_use.feature_id
                        ));
                    }
                }
                "peer_asset" => {
                    peer_reference_used = true;
                    if feature_use.reference_identity_digest.as_deref().is_none_or(|identity| {
                        !train.contains(identity) || test.contains(identity)
                    }) || feature_use.reference_measurement_at_ms.is_none()
                        || feature_use.reference_received_at_ms.is_none()
                        || feature_use.reference_population_digest.is_some()
                        || feature_use.reference_received_at_ms
                            .zip(feature_use.reference_measurement_at_ms)
                            .is_some_and(|(received, measured)| received < measured)
                    {
                        errors.push(format!(
                            "prediction {} feature {} peer reference is not a distinct training asset",
                            row.sample_digest, feature_use.feature_id
                        ));
                    }
                    if manifest.evaluation_mode == "strict_online"
                        && (feature_use.reference_measurement_at_ms
                            .is_none_or(|time| time > row.decision_at_ms)
                            || feature_use.reference_received_at_ms
                                .is_none_or(|time| time > row.decision_at_ms))
                    {
                        errors.push(format!(
                            "prediction {} feature {} peer reference was unavailable at decision time",
                            row.sample_digest, feature_use.feature_id
                        ));
                    }
                }
                "external_fixed" => {
                    if feature_use.reference_population_digest.as_deref().is_none_or(|digest| !is_digest(digest))
                        || feature_use.reference_identity_digest.is_some()
                        || definition.fit_scope != "fixed_predeclared"
                    {
                        errors.push(format!(
                            "prediction {} feature {} external reference is not fixed/predeclared",
                            row.sample_digest, feature_use.feature_id
                        ));
                    }
                }
                "full_lifetime_extrema" => {
                    if feature_use.reference_measurement_at_ms.is_none()
                        || feature_use.reference_received_at_ms.is_none()
                        || feature_use.reference_received_at_ms
                            .zip(feature_use.reference_measurement_at_ms)
                            .is_some_and(|(received, measured)| received < measured)
                    {
                        errors.push(format!(
                            "prediction {} feature {} lacks a chronologically valid full-lifetime reference receipt",
                            row.sample_digest, feature_use.feature_id
                        ));
                    }
                    if manifest.evaluation_mode == "strict_online" {
                        errors.push(format!(
                            "prediction {} feature {} uses a full-lifetime reference in strict-online mode",
                            row.sample_digest, feature_use.feature_id
                        ));
                    } else {
                        offline_only = true;
                    }
                }
                _ => errors.push(format!(
                    "prediction {} feature {} has an unknown reference kind",
                    row.sample_digest, feature_use.feature_id
                )),
            }
        }
        if used.len() != definitions.len()
            || definitions.keys().any(|feature_id| !used.contains(feature_id))
        {
            errors.push(format!(
                "prediction {} does not contain exactly one receipt for every declared feature",
                row.sample_digest
            ));
        }
    }

    if manifest.prediction_rows.is_empty() {
        errors.push("manifest contains no prediction rows".to_owned());
    }

    let randomization = &manifest.reference_randomization;
    if !nonempty(&randomization.method_ref) || !is_digest(&randomization.method_digest) {
        errors.push("reference randomization method identity is invalid".to_owned());
    }
    if peer_reference_used {
        let required_arms = [
            "true_pair",
            "randomized_derangement",
            "affine_negative_control",
        ];
        if randomization.status != "pre_registered"
            || randomization.assignment_digest.as_deref().is_none_or(|digest| !is_digest(digest))
            || randomization.seed.is_none()
            || randomization.derangement_count < 100
            || randomization.schedule_frozen_at_ms.is_none_or(|time| {
                time == 0 || time >= manifest.split_frozen_at_ms
            })
            || required_arms.iter().any(|arm| !randomization.arms.iter().any(|item| item == arm))
        {
            errors.push(
                "peer-asset references require a preregistered assignment and at least 100 derangements plus negative controls"
                    .to_owned(),
            );
        }
    } else if randomization.status != "not_applicable"
        || randomization.assignment_digest.is_some()
        || randomization.seed.is_some()
        || randomization.derangement_count != 0
        || randomization.schedule_frozen_at_ms.is_some()
        || !randomization.arms.is_empty()
    {
        errors.push("reference randomization must be not_applicable when no peer reference is used".to_owned());
    }

    let claim_ceiling = if offline_only {
        "OFFLINE_ORACLE_ONLY"
    } else {
        "STRICT_ONLINE_ELIGIBLE"
    };
    (errors, claim_ceiling.to_owned())
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
            for field in ["train_group_identity_digests", "test_group_identity_digests"] {
                if let Some(rows) = manifest.get_mut(field).and_then(Value::as_array_mut) {
                    rows.sort_by(|left, right| left.as_str().cmp(&right.as_str()));
                }
            }
            if let Some(rows) = manifest.get_mut("feature_definitions").and_then(Value::as_array_mut) {
                rows.sort_by(|left, right| {
                    left.get("feature_id")
                        .and_then(Value::as_str)
                        .cmp(&right.get("feature_id").and_then(Value::as_str))
                });
                for feature in rows {
                    if let Some(groups) = feature
                        .get_mut("fit_group_identity_digests")
                        .and_then(Value::as_array_mut)
                    {
                        groups.sort_by(|left, right| left.as_str().cmp(&right.as_str()));
                    }
                }
            }
            if let Some(rows) = manifest.get_mut("prediction_rows").and_then(Value::as_array_mut) {
                rows.sort_by(|left, right| {
                    left.get("sample_digest")
                        .and_then(Value::as_str)
                        .cmp(&right.get("sample_digest").and_then(Value::as_str))
                });
                for row in rows {
                    if let Some(uses) = row.get_mut("feature_uses").and_then(Value::as_array_mut) {
                        uses.sort_by(|left, right| {
                            left.get("feature_id")
                                .and_then(Value::as_str)
                                .cmp(&right.get("feature_id").and_then(Value::as_str))
                        });
                    }
                }
            }
            if let Some(arms) = manifest
                .get_mut("reference_randomization")
                .and_then(|randomization| randomization.get_mut("arms"))
                .and_then(Value::as_array_mut)
            {
                arms.sort_by(|left, right| left.as_str().cmp(&right.as_str()));
            }
        }
    }
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
    let mut strict_online_eligible_count = 0usize;
    let mut offline_oracle_only_count = 0usize;
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
            match serde_json::from_value::<Manifest>(case.manifest) {
                Ok(manifest) => {
                    let (errors, claim_ceiling) = validate_manifest(&manifest);
                    if errors.is_empty() {
                        Ok((claim_ceiling, manifest.prediction_rows.len(), manifest.feature_definitions.len()))
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
                Ok((claim, _, _)) => format!("unexpected acceptance with claim ceiling {claim}"),
                Err(errors) => errors.join("; "),
            };
            return Err(format!("case {} expectation mismatch: {}", case.case_id, detail));
        }

        let (claim_ceiling, prediction_count, feature_definition_count, errors) = match outcome {
            Ok((claim, predictions, features)) => {
                if case.expected_claim_ceiling.as_deref() != Some(claim.as_str()) {
                    return Err(format!(
                        "case {} claim ceiling differs from known-truth expectation: {claim}",
                        case.case_id
                    ));
                }
                if claim == "STRICT_ONLINE_ELIGIBLE" {
                    strict_online_eligible_count += 1;
                } else {
                    offline_oracle_only_count += 1;
                }
                accepted_count += 1;
                (claim, predictions, features, Vec::new())
            }
            Err(errors) => {
                if case.expected_claim_ceiling.is_some() {
                    return Err(format!(
                        "rejected case {} must not declare a claim ceiling",
                        case.case_id
                    ));
                }
                rejected_count += 1;
                ("REJECTED".to_owned(), 0, 0, errors)
            }
        };
        cases.push(CaseAudit {
            case_id: case.case_id,
            expectation: case.expectation,
            accepted,
            errors,
            claim_ceiling,
            prediction_count,
            feature_definition_count,
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
        strict_online_eligible_count,
        offline_oracle_only_count,
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
        include_bytes!("../../../schemas/transport/ev_soh_feature_availability_v0_1.schema.json");
    const CORPUS: &[u8] =
        include_bytes!("../../../fixtures/ev_battery/soh_feature_availability_scenarios_v0_1.json");

    fn report() -> CorpusReport {
        audit_corpus(SCHEMA, CORPUS).expect("frozen feature availability corpus")
    }

    #[test]
    fn frozen_corpus_accepts_three_and_rejects_eleven_controls() {
        let report = report();
        assert_eq!(report.payload.case_count, 14);
        assert_eq!(report.payload.accepted_count, 3);
        assert_eq!(report.payload.rejected_count, 11);
        assert_eq!(report.payload.strict_online_eligible_count, 2);
        assert_eq!(report.payload.offline_oracle_only_count, 1);
    }

    #[test]
    fn offline_oracle_is_preserved_but_never_marked_strict_online_eligible() {
        let report = report();
        let case = report.payload.cases.iter()
            .find(|case| case.case_id == "full-history-offline-oracle")
            .expect("offline oracle fixture");
        assert!(case.accepted);
        assert_eq!(case.claim_ceiling, "OFFLINE_ORACLE_ONLY");
    }

    #[test]
    fn strict_online_feature_inputs_must_arrive_and_be_derived_before_decision() {
        let report = report();
        for case_id in [
            "measurement-arrives-after-decision-rejected",
            "feature-derived-after-decision-rejected",
        ] {
            let case = report.payload.cases.iter()
                .find(|case| case.case_id == case_id)
                .expect("feature availability control");
            assert!(!case.accepted);
        }
    }

    #[test]
    fn same_asset_prefix_reference_is_physically_bound_to_the_prediction_asset() {
        let report = report();
        let case = report.payload.cases.iter()
            .find(|case| case.case_id == "same-cell-anchor-identity-mismatch-rejected")
            .expect("same-cell reference identity control");
        assert!(!case.accepted);
    }

    #[test]
    fn peer_references_require_training_asset_lineage_and_preregistered_derangements() {
        let report = report();
        let valid = report.payload.cases.iter()
            .find(|case| case.case_id == "peer-reference-randomization-preregistered")
            .expect("valid peer reference fixture");
        assert!(valid.accepted);
        assert_eq!(valid.claim_ceiling, "STRICT_ONLINE_ELIGIBLE");
        for case_id in [
            "peer-reference-heldout-cell-rejected",
            "peer-reference-without-randomization-controls-rejected",
        ] {
            let case = report.payload.cases.iter()
                .find(|case| case.case_id == case_id)
                .expect("peer reference negative control");
            assert!(!case.accepted);
        }
    }

    #[test]
    fn target_label_future_transform_and_full_lifetime_leakage_are_rejected_online() {
        let report = report();
        for case_id in [
            "full-lifetime-normalization-rejected",
            "test-label-feature-leakage-rejected",
            "same-cell-reference-after-decision-rejected",
            "train-transform-fitted-on-test-cell-rejected",
            "fixed-transform-predeclared-after-split-rejected",
        ] {
            let case = report.payload.cases.iter()
                .find(|case| case.case_id == case_id)
                .expect("feature leakage negative control");
            assert!(!case.accepted);
        }
    }

    #[test]
    fn impossible_source_and_feature_computation_timestamp_orderings_are_rejected() {
        let report = report();
        for case_id in [
            "receive-timestamp-precedes-measurement-rejected",
            "feature-derived-before-input-arrival-rejected",
            "peer-reference-received-before-measurement-rejected",
        ] {
            let case = report.payload.cases.iter()
                .find(|case| case.case_id == case_id)
                .expect("timestamp chronology negative control");
            assert!(!case.accepted);
        }
    }

    #[test]
    fn fixed_fit_must_precede_its_predeclaration_and_full_lifetime_refs_need_time_receipts() {
        let report = report();
        for case_id in [
            "fixed-transform-fit-after-split-freeze-rejected",
            "offline-full-life-reference-without-time-receipt-rejected",
        ] {
            let case = report.payload.cases.iter()
                .find(|case| case.case_id == case_id)
                .expect("reference/fit chronology control");
            assert!(!case.accepted);
        }
    }

    #[test]
    fn every_prediction_requires_exactly_one_receipt_for_every_feature() {
        let report = report();
        let case = report.payload.cases.iter()
            .find(|case| case.case_id == "missing-feature-receipt-rejected")
            .expect("missing feature receipt control");
        assert!(!case.accepted);
    }

    #[test]
    fn canonical_report_is_invariant_to_case_feature_row_and_identity_order() {
        let original = report();
        let mut value: Value = serde_json::from_slice(CORPUS).expect("corpus JSON");
        value["cases"].as_array_mut().unwrap().reverse();
        for case in value["cases"].as_array_mut().unwrap() {
            case["manifest"]["train_group_identity_digests"].as_array_mut().unwrap().reverse();
            case["manifest"]["test_group_identity_digests"].as_array_mut().unwrap().reverse();
            case["manifest"]["feature_definitions"].as_array_mut().unwrap().reverse();
            for feature in case["manifest"]["feature_definitions"].as_array_mut().unwrap() {
                feature["fit_group_identity_digests"].as_array_mut().unwrap().reverse();
            }
            case["manifest"]["prediction_rows"].as_array_mut().unwrap().reverse();
            for row in case["manifest"]["prediction_rows"].as_array_mut().unwrap() {
                row["feature_uses"].as_array_mut().unwrap().reverse();
            }
            case["manifest"]["reference_randomization"]["arms"].as_array_mut().unwrap().reverse();
        }
        let bytes = serde_json::to_vec(&value).expect("permuted feature corpus");
        let replay = audit_corpus(SCHEMA, &bytes).expect("permuted feature availability audit");
        assert_eq!(original.report_sha256, replay.report_sha256);
        assert_eq!(
            original.payload.canonical_corpus_sha256,
            replay.payload.canonical_corpus_sha256
        );
    }
}
