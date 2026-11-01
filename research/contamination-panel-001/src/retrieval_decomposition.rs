//! Strict evidence-graph and paired-arm contract auditor for retrieval decomposition.
//!
//! This module qualifies synthetic corpus/oracle/experiment artifacts. It does
//! not execute a model, search engine, citation judge or randomized experiment.
//! A valid artifact audit is not a benchmark result or capability claim.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

pub const CORPUS_SCHEMA: &str = "symthaea.retrieval_decomposition.corpus.v1";
pub const ORACLE_SCHEMA: &str = "symthaea.retrieval_decomposition.sealed_oracle.v1";
pub const EXPERIMENT_SCHEMA: &str = "symthaea.retrieval_decomposition.experiment.v1";
pub const AUDIT_SCHEMA: &str = "symthaea.retrieval_decomposition.audit.v1";
pub const REQUIRED_ARMS: [&str; 6] = [
    "no_evidence",
    "native_search",
    "gold_evidence_injection",
    "gold_plus_hard_distractors",
    "evidence_complete_stop",
    "plain_qa_repackaging",
];

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RetrievalResponseContractV1 {
    pub format: String,
    pub fields: Vec<String>,
    pub units: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RetrievalTaskV1 {
    pub task_id: String,
    pub task_family: String,
    pub task_text: String,
    pub response_contract: RetrievalResponseContractV1,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RetrievalDocumentV1 {
    pub document_id: String,
    pub source_family_id: String,
    pub title: String,
    pub body: String,
    pub content_sha256: String,
    pub body_utf8_bytes: u64,
    pub provenance: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RetrievalCorpusV1 {
    pub schema: String,
    pub benchmark_id: String,
    pub benchmark_version: String,
    pub task: RetrievalTaskV1,
    pub documents: Vec<RetrievalDocumentV1>,
    pub corpus_sha256: String,
    pub claim_boundary: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RetrievalExpectedAnswerV1 {
    pub amount: i64,
    pub units: String,
    pub canonicalization: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RetrievalCalculationV1 {
    pub labor_rate_per_day: i64,
    pub billable_days: i64,
    pub overnight_rate: i64,
    pub overnight_stays: i64,
    pub expected_total: i64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RetrievalEvidenceSpanV1 {
    pub document_id: String,
    pub start_byte: u64,
    pub end_byte: u64,
    pub span_sha256: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RetrievalRequiredClaimV1 {
    pub claim_id: String,
    pub claim_text: String,
    pub supporting_spans: Vec<RetrievalEvidenceSpanV1>,
    pub derived_from_claim_ids: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RetrievalDistractorScenarioV1 {
    pub document_id: String,
    pub adjacent_query: String,
    pub distractor_answer: i64,
    pub billable_days: i64,
    pub overnight_stays: i64,
    pub confusion_mechanism: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RetrievalSealedOracleV1 {
    pub schema: String,
    pub task_id: String,
    pub expected_answer: RetrievalExpectedAnswerV1,
    pub calculation: RetrievalCalculationV1,
    pub required_claims: Vec<RetrievalRequiredClaimV1>,
    pub distractor_scenarios: Vec<RetrievalDistractorScenarioV1>,
    pub allowed_alternative_evidence_paths: Vec<Vec<String>>,
    pub answer_sha256: String,
    pub oracle_sha256: String,
    pub claim_boundary: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RetrievalExperimentReplicateV1 {
    pub task_id: String,
    pub replicate_id: String,
    pub paired_seed_commitment_sha256: String,
    pub arm_order: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RetrievalExperimentV1 {
    pub schema: String,
    pub experiment_id: String,
    pub benchmark_id: String,
    pub benchmark_version: String,
    pub model_checkpoint_sha256: String,
    pub corpus_sha256: String,
    pub oracle_sha256: String,
    pub inference_harness_sha256: String,
    pub prompt_and_decoding_sha256: String,
    pub budget_policy_sha256: String,
    pub sample_size_plan_sha256: String,
    pub arms: Vec<String>,
    pub task_ids: Vec<String>,
    pub replicates: Vec<RetrievalExperimentReplicateV1>,
    pub randomization_seed_commitment_sha256: String,
    pub experiment_status: String,
    pub claim_boundary: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RetrievalDecompositionBundleV1 {
    pub corpus: RetrievalCorpusV1,
    pub oracle: RetrievalSealedOracleV1,
    pub experiment: RetrievalExperimentV1,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RetrievalDecompositionAuditV1 {
    pub schema: String,
    pub valid: bool,
    pub corpus_digest_matches: bool,
    pub oracle_digest_matches: bool,
    pub document_hashes_and_lengths_match: bool,
    pub evidence_spans_match: bool,
    pub claim_graph_is_acyclic: bool,
    pub distractor_arithmetic_matches: bool,
    pub expected_answer_matches_calculation: bool,
    pub experiment_bindings_match: bool,
    pub all_six_arms_present: bool,
    pub replicate_arm_orders_are_permutations: bool,
    pub unique_task_and_replicate_bindings: bool,
    pub candidate_oracle_separation_declared: bool,
    pub document_count: u64,
    pub claim_count: u64,
    pub distractor_count: u64,
    pub replicate_count: u64,
    pub corpus_sha256: String,
    pub oracle_sha256: String,
    pub experiment_id: String,
    pub experiment_status: String,
    pub issues: Vec<String>,
    pub claim_boundary: String,
}

fn sha256(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(output, "{byte:02x}");
    }
    output
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn digest_normalized<T: Serialize + Clone>(
    value: &T,
    clear_digest: impl FnOnce(&mut T),
) -> Result<String, String> {
    let mut normalized = value.clone();
    clear_digest(&mut normalized);
    let bytes = serde_json::to_vec(&normalized)
        .map_err(|error| format!("cannot serialize normalized artifact: {error}"))?;
    Ok(sha256(&bytes))
}

fn exact_span<'a>(
    documents: &'a BTreeMap<&str, &RetrievalDocumentV1>,
    span: &RetrievalEvidenceSpanV1,
) -> Result<&'a str, String> {
    let document = documents
        .get(span.document_id.as_str())
        .ok_or_else(|| format!("evidence references unknown document {}", span.document_id))?;
    let start = usize::try_from(span.start_byte)
        .map_err(|_| "evidence start byte exceeds usize")?;
    let end = usize::try_from(span.end_byte)
        .map_err(|_| "evidence end byte exceeds usize")?;
    if start >= end || end > document.body.len() {
        return Err(format!("evidence span bounds are invalid for {}", span.document_id));
    }
    let slice = document.body.get(start..end)
        .ok_or_else(|| format!("evidence span for {} is not on UTF-8 boundaries", span.document_id))?;
    if sha256(slice.as_bytes()) != span.span_sha256 {
        return Err(format!("evidence span digest mismatch for {}", span.document_id));
    }
    Ok(slice)
}

fn visit_claim(
    claim_id: &str,
    claims: &BTreeMap<&str, &RetrievalRequiredClaimV1>,
    visiting: &mut BTreeSet<String>,
    visited: &mut BTreeSet<String>,
) -> Result<(), String> {
    if visited.contains(claim_id) {
        return Ok(());
    }
    if !visiting.insert(claim_id.to_owned()) {
        return Err(format!("claim dependency cycle includes {claim_id}"));
    }
    let claim = claims
        .get(claim_id)
        .ok_or_else(|| format!("unknown claim dependency {claim_id}"))?;
    for parent in &claim.derived_from_claim_ids {
        if !claims.contains_key(parent.as_str()) {
            return Err(format!("claim {} depends on unknown claim {parent}", claim.claim_id));
        }
        visit_claim(parent, claims, visiting, visited)?;
    }
    visiting.remove(claim_id);
    visited.insert(claim_id.to_owned());
    Ok(())
}

fn audit_corpus_and_oracle(
    corpus: &RetrievalCorpusV1,
    oracle: &RetrievalSealedOracleV1,
) -> Result<(bool, bool, bool, bool, u64, u64), String> {
    if corpus.schema != CORPUS_SCHEMA || oracle.schema != ORACLE_SCHEMA {
        return Err("corpus or oracle schema identifier is invalid".to_owned());
    }
    if corpus.benchmark_id.trim().is_empty()
        || corpus.benchmark_version.trim().is_empty()
        || corpus.task.task_id.trim().is_empty()
        || corpus.task.task_family.trim().is_empty()
        || corpus.task.task_text.trim().is_empty()
    {
        return Err("corpus task and benchmark identities must be nonempty".to_owned());
    }
    if corpus.task.response_contract.format != "json"
        || corpus.task.response_contract.units.trim().is_empty()
        || corpus.task.response_contract.fields.is_empty()
    {
        return Err("candidate response contract is incomplete or unsupported".to_owned());
    }
    let corpus_digest = digest_normalized(corpus, |value| value.corpus_sha256.clear())?;
    let oracle_digest = digest_normalized(oracle, |value| value.oracle_sha256.clear())?;
    if !valid_sha256(&corpus.corpus_sha256)
        || corpus_digest != corpus.corpus_sha256
        || !valid_sha256(&oracle.oracle_sha256)
        || oracle_digest != oracle.oracle_sha256
    {
        return Err("corpus or oracle canonical digest does not match contents".to_owned());
    }
    let mut document_map = BTreeMap::new();
    let mut document_families = BTreeSet::new();
    for document in &corpus.documents {
        if document.document_id.trim().is_empty()
            || document.source_family_id.trim().is_empty()
            || document.title.trim().is_empty()
            || document.provenance.trim().is_empty()
        {
            return Err("document identities, title and provenance must be nonempty".to_owned());
        }
        if document_map.insert(document.document_id.as_str(), document).is_some() {
            return Err(format!("duplicate document ID {}", document.document_id));
        }
        let actual_len = u64::try_from(document.body.as_bytes().len())
            .map_err(|_| "document body length exceeds u64")?;
        if document.body_utf8_bytes != actual_len
            || !valid_sha256(&document.content_sha256)
            || sha256(document.body.as_bytes()) != document.content_sha256
        {
            return Err(format!("document {} content hash or byte length mismatch", document.document_id));
        }
        document_families.insert(document.source_family_id.as_str());
    }
    if oracle.task_id != corpus.task.task_id {
        return Err("oracle task ID differs from candidate corpus task".to_owned());
    }
    if oracle.expected_answer.canonicalization != "exact_integer"
        || oracle.expected_answer.units != corpus.task.response_contract.units
    {
        return Err("oracle answer canonicalization/units differ from candidate response contract".to_owned());
    }
    let calc = &oracle.calculation;
    if calc.labor_rate_per_day < 0
        || calc.billable_days < 0
        || calc.overnight_rate < 0
        || calc.overnight_stays < 0
        || calc.labor_rate_per_day.checked_mul(calc.billable_days)
            .and_then(|labor| calc.overnight_rate.checked_mul(calc.overnight_stays)
                .and_then(|lodging| labor.checked_add(lodging)))
            != Some(calc.expected_total)
        || calc.expected_total != oracle.expected_answer.amount
    {
        return Err("primary answer does not match checked exact arithmetic".to_owned());
    }
    let mut claim_map = BTreeMap::new();
    let mut evidence_spans_match = true;
    for claim in &oracle.required_claims {
        if claim.claim_id.trim().is_empty() || claim.claim_text.trim().is_empty() {
            return Err("required claim ID and text must be nonempty".to_owned());
        }
        if claim_map.insert(claim.claim_id.as_str(), claim).is_some() {
            return Err(format!("duplicate required claim ID {}", claim.claim_id));
        }
        if claim.supporting_spans.is_empty() && claim.derived_from_claim_ids.is_empty() {
            return Err(format!("claim {} has neither source support nor derivation", claim.claim_id));
        }
        for span in &claim.supporting_spans {
            if exact_span(&document_map, span).is_err() {
                evidence_spans_match = false;
            }
        }
    }
    let mut visiting = BTreeSet::new();
    let mut visited = BTreeSet::new();
    for claim_id in claim_map.keys() {
        visit_claim(claim_id, &claim_map, &mut visiting, &mut visited)?;
    }
    let mut distractor_math = true;
    let mut distractor_docs = BTreeSet::new();
    for distractor in &oracle.distractor_scenarios {
        if !document_map.contains_key(distractor.document_id.as_str())
            || distractor.document_id.trim().is_empty()
            || distractor.adjacent_query.trim().is_empty()
            || distractor.confusion_mechanism.trim().is_empty()
            || !distractor_docs.insert(distractor.document_id.as_str())
        {
            return Err("distractor references an invalid or duplicate document".to_owned());
        }
        let calculated = calc.labor_rate_per_day
            .checked_mul(distractor.billable_days)
            .and_then(|labor| calc.overnight_rate.checked_mul(distractor.overnight_stays)
                .and_then(|lodging| labor.checked_add(lodging)));
        distractor_math &= calculated == Some(distractor.distractor_answer)
            && distractor.distractor_answer != oracle.expected_answer.amount;
    }
    if oracle.allowed_alternative_evidence_paths.is_empty() {
        return Err("oracle must declare at least one allowed evidence path".to_owned());
    }
    for path in &oracle.allowed_alternative_evidence_paths {
        let mut unique = BTreeSet::new();
        if path.is_empty()
            || path.iter().any(|id| !unique.insert(id.as_str()) || !document_map.contains_key(id.as_str()))
        {
            return Err("alternative evidence path contains an empty, duplicate or unknown document".to_owned());
        }
    }
    let answer_bytes = oracle.expected_answer.amount.to_string();
    if sha256(answer_bytes.as_bytes()) != oracle.answer_sha256 {
        return Err("oracle answer digest does not match canonical integer answer".to_owned());
    }
    Ok((
        evidence_spans_match,
        true,
        distractor_math,
        oracle.expected_answer.amount == calc.expected_total,
        u64::try_from(corpus.documents.len()).map_err(|_| "document count exceeds u64")?,
        u64::try_from(oracle.required_claims.len()).map_err(|_| "claim count exceeds u64")?,
    ))
}

/// Audit the corpus, sealed oracle and six-arm paired experiment contracts.
/// This does not execute a model or establish external benchmark validity.
pub fn audit_retrieval_decomposition_bundle(
    bundle: &RetrievalDecompositionBundleV1,
) -> Result<RetrievalDecompositionAuditV1, String> {
    let (spans_match, graph_acyclic, distractor_math, answer_matches, document_count, claim_count) =
        audit_corpus_and_oracle(&bundle.corpus, &bundle.oracle)?;
    let experiment = &bundle.experiment;
    if experiment.schema != EXPERIMENT_SCHEMA {
        return Err("experiment schema identifier is invalid".to_owned());
    }
    if experiment.experiment_id.trim().is_empty()
        || experiment.claim_boundary.trim().is_empty()
        || experiment.task_ids.is_empty()
        || experiment.replicates.is_empty()
    {
        return Err("experiment identity, tasks and replicates must be nonempty".to_owned());
    }
    let required_arms: BTreeSet<&str> = REQUIRED_ARMS.into_iter().collect();
    let arms: BTreeSet<&str> = experiment.arms.iter().map(String::as_str).collect();
    let all_six_arms_present = arms == required_arms && experiment.arms.len() == REQUIRED_ARMS.len();
    let unique_task_ids: BTreeSet<&str> = experiment.task_ids.iter().map(String::as_str).collect();
    if unique_task_ids.len() != experiment.task_ids.len()
        || !unique_task_ids.contains(bundle.corpus.task.task_id.as_str())
    {
        return Err("experiment task IDs must be unique and include the corpus task".to_owned());
    }
    if experiment.benchmark_id != bundle.corpus.benchmark_id
        || experiment.benchmark_version != bundle.corpus.benchmark_version
        || experiment.corpus_sha256 != bundle.corpus.corpus_sha256
        || experiment.oracle_sha256 != bundle.oracle.oracle_sha256
    {
        return Err("experiment benchmark/corpus/oracle bindings do not match the bundle".to_owned());
    }
    for (digest, label) in [
        (&experiment.model_checkpoint_sha256, "model_checkpoint_sha256"),
        (&experiment.inference_harness_sha256, "inference_harness_sha256"),
        (&experiment.prompt_and_decoding_sha256, "prompt_and_decoding_sha256"),
        (&experiment.budget_policy_sha256, "budget_policy_sha256"),
        (&experiment.sample_size_plan_sha256, "sample_size_plan_sha256"),
        (&experiment.randomization_seed_commitment_sha256, "randomization_seed_commitment_sha256"),
    ] {
        if !valid_sha256(digest) {
            return Err(format!("{label} is not a lowercase SHA-256 digest"));
        }
    }
    let mut replicate_ids = BTreeSet::new();
    let mut pair_seed_commitments = BTreeSet::new();
    let mut replicate_arm_orders_are_permutations = true;
    for replicate in &experiment.replicates {
        if !unique_task_ids.contains(replicate.task_id.as_str())
            || replicate.replicate_id.trim().is_empty()
            || !replicate_ids.insert(replicate.replicate_id.as_str())
            || !valid_sha256(&replicate.paired_seed_commitment_sha256)
            || !pair_seed_commitments.insert(replicate.paired_seed_commitment_sha256.as_str())
        {
            return Err("replicate task/identity/seed bindings must be unique and valid".to_owned());
        }
        let order: BTreeSet<&str> = replicate.arm_order.iter().map(String::as_str).collect();
        replicate_arm_orders_are_permutations &=
            order == required_arms && replicate.arm_order.len() == REQUIRED_ARMS.len();
    }
    let experiment_status_allowed = matches!(
        experiment.experiment_status.as_str(),
        "protocol_only" | "exploratory" | "confirmatory_preregistered"
    );
    if !experiment_status_allowed {
        return Err("unknown experiment_status".to_owned());
    }
    let candidate_oracle_separation_declared = bundle.corpus.claim_boundary.contains("candidate")
        && bundle.oracle.claim_boundary.contains("sealed");
    let valid = spans_match
        && graph_acyclic
        && distractor_math
        && answer_matches
        && all_six_arms_present
        && replicate_arm_orders_are_permutations
        && candidate_oracle_separation_declared;
    Ok(RetrievalDecompositionAuditV1 {
        schema: AUDIT_SCHEMA.to_owned(),
        valid,
        corpus_digest_matches: true,
        oracle_digest_matches: true,
        document_hashes_and_lengths_match: true,
        evidence_spans_match: spans_match,
        claim_graph_is_acyclic: graph_acyclic,
        distractor_arithmetic_matches: distractor_math,
        expected_answer_matches_calculation: answer_matches,
        experiment_bindings_match: true,
        all_six_arms_present,
        replicate_arm_orders_are_permutations,
        unique_task_and_replicate_bindings: true,
        candidate_oracle_separation_declared,
        document_count,
        claim_count,
        distractor_count: u64::try_from(bundle.oracle.distractor_scenarios.len())
            .map_err(|_| "distractor count exceeds u64")?,
        replicate_count: u64::try_from(experiment.replicates.len())
            .map_err(|_| "replicate count exceeds u64")?,
        corpus_sha256: bundle.corpus.corpus_sha256.clone(),
        oracle_sha256: bundle.oracle.oracle_sha256.clone(),
        experiment_id: experiment.experiment_id.clone(),
        experiment_status: experiment.experiment_status.clone(),
        issues: Vec::new(),
        claim_boundary: "artifact_contract_and_synthetic_arithmetic_checked;no_model_runs_or_search_quality_claims;confirmatory_status_requires_independent_preregistration_and_trusted_execution_evidence".to_owned(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn fixture() -> RetrievalDecompositionBundleV1 {
        RetrievalDecompositionBundleV1 {
            corpus: serde_json::from_str(include_str!(
                "../fixtures/retrieval-decomposition-pilot-v1.corpus.json"
            )).unwrap(),
            oracle: serde_json::from_str(include_str!(
                "../fixtures/retrieval-decomposition-pilot-v1.oracle.json"
            )).unwrap(),
            experiment: serde_json::from_str(include_str!(
                "../fixtures/retrieval-decomposition-pilot-v1.experiment.json"
            )).unwrap(),
        }
    }

    #[test]
    fn strict_schemas_match_serialized_rust_types() {
        let bundle = fixture();
        let audit = audit_retrieval_decomposition_bundle(&bundle).unwrap();
        let schemas = [
            serde_json::from_str::<Value>(include_str!(
                "../schema/retrieval-decomposition-corpus.schema.json"
            )).unwrap(),
            serde_json::from_str::<Value>(include_str!(
                "../schema/retrieval-decomposition-oracle.schema.json"
            )).unwrap(),
            serde_json::from_str::<Value>(include_str!(
                "../schema/retrieval-decomposition-experiment.schema.json"
            )).unwrap(),
        ];
        let values = [
            serde_json::to_value(&bundle.corpus).unwrap(),
            serde_json::to_value(&bundle.oracle).unwrap(),
            serde_json::to_value(&bundle.experiment).unwrap(),
        ];
        for (schema, value) in schemas.iter().zip(values.iter()) {
            let required: BTreeSet<String> = schema["required"].as_array().unwrap()
                .iter().map(|field| field.as_str().unwrap().to_owned()).collect();
            let properties: BTreeSet<String> = schema["properties"].as_object().unwrap()
                .keys().cloned().collect();
            let actual: BTreeSet<String> = value.as_object().unwrap()
                .keys().cloned().collect();
            assert_eq!(required, properties);
            assert_eq!(actual, properties);
        }
        let audit_value = serde_json::to_value(&audit).unwrap();
        assert_eq!(audit_value["schema"], AUDIT_SCHEMA);
    }

    #[test]
    fn pilot_arithmetic_and_distractor_are_distinct_and_exact() {
        let bundle = fixture();
        let audit = audit_retrieval_decomposition_bundle(&bundle).unwrap();
        assert!(audit.valid);
        assert!(audit.expected_answer_matches_calculation);
        assert!(audit.distractor_arithmetic_matches);
        assert_eq!(bundle.oracle.expected_answer.amount, 690);
        assert_eq!(bundle.oracle.distractor_scenarios[0].distractor_answer, 765);
    }

    #[test]
    fn tampered_document_body_fails_digest_and_length_audit() {
        let mut bundle = fixture();
        bundle.corpus.documents[0].body.push('!');
        assert!(audit_retrieval_decomposition_bundle(&bundle).is_err());
    }

    #[test]
    fn evidence_span_rejects_rebound_or_out_of_range_bytes() {
        let mut bundle = fixture();
        bundle.oracle.required_claims[0].supporting_spans[0].end_byte += 1;
        let audit = audit_retrieval_decomposition_bundle(&bundle).unwrap();
        assert!(!audit.valid);
        assert!(!audit.evidence_spans_match);
    }

    #[test]
    fn claim_dependency_cycles_fail_closed() {
        let mut bundle = fixture();
        bundle.oracle.required_claims[0].derived_from_claim_ids.push("total-cost".to_owned());
        assert!(audit_retrieval_decomposition_bundle(&bundle).is_err());
    }

    #[test]
    fn changed_gold_evidence_identity_breaks_experiment_binding() {
        let mut bundle = fixture();
        bundle.experiment.corpus_sha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_owned();
        assert!(audit_retrieval_decomposition_bundle(&bundle).is_err());
    }

    #[test]
    fn every_arm_order_is_a_permutation_and_replicate_seeds_are_unique() {
        let mut bundle = fixture();
        bundle.experiment.replicates[0].arm_order.pop();
        let audit = audit_retrieval_decomposition_bundle(&bundle).unwrap();
        assert!(!audit.valid);
        assert!(!audit.replicate_arm_orders_are_permutations);
    }

    #[test]
    fn answer_digest_must_match_exact_canonical_integer() {
        let mut bundle = fixture();
        bundle.oracle.answer_sha256 = "0000000000000000000000000000000000000000000000000000000000000000".to_owned();
        assert!(audit_retrieval_decomposition_bundle(&bundle).is_err());
    }

    #[test]
    fn unknown_fields_are_rejected_at_every_artifact_boundary() {
        let mut value: Value = serde_json::to_value(fixture().corpus).unwrap();
        value["leaked_answer"] = serde_json::json!(690);
        assert!(serde_json::from_value::<RetrievalCorpusV1>(value).is_err());
    }
}
