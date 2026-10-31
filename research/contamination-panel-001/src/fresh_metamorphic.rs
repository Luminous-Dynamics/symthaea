//! Metamorphic generation and scoring for secret-seeded fresh linear-system tasks.
//!
//! The generator creates deterministic, mathematically equivalent presentation
//! variants of every base task. Exact correctness and cross-form agreement are
//! reported separately: a consistently wrong answer is never counted correct.
//! The base oracle and all variant parameters remain evaluator-only.

use crate::{
    fresh_instance::{
        task_text, normalized_determinant_margin_permille, ExactLinearSolutionV1,
        FreshChallengeBundleV1, LinearSystemParametersV1,
    },
    fresh_scoring::{validate_bundle, FreshItemScoreStatusV1},
    hash_domain,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const FRESH_METAMORPHIC_PACKET_SCHEMA: &str =
    "symthaea.fresh_metamorphic_public_packet.v1";
pub const FRESH_METAMORPHIC_ORACLE_SCHEMA: &str =
    "symthaea.fresh_metamorphic_sealed_oracle.v1";
pub const FRESH_METAMORPHIC_MANIFEST_SCHEMA: &str =
    "symthaea.fresh_metamorphic_manifest.v1";
pub const FRESH_METAMORPHIC_AUDIT_SCHEMA: &str =
    "symthaea.fresh_metamorphic_audit.v1";
pub const FRESH_METAMORPHIC_SUBMISSION_SCHEMA: &str =
    "symthaea.fresh_metamorphic_submission.v1";
pub const FRESH_METAMORPHIC_SCORE_SCHEMA: &str =
    "symthaea.fresh_metamorphic_score_report.v1";
pub const FRESH_METAMORPHIC_METRIC_ID: &str =
    "fresh-linear-system-metamorphic.v1";
pub const FRESH_METAMORPHIC_FORMS_PER_GROUP: usize = 5;
pub const FRESH_METAMORPHIC_MAX_ITEMS: usize = 50_000;
pub const FRESH_METAMORPHIC_MAX_SUBMISSION_BYTES: usize = 32 * 1024 * 1024;
pub const FRESH_METAMORPHIC_MAX_RESPONSE_BYTES: usize = 1_048_576;
pub const FRESH_METAMORPHIC_MAX_TOTAL_RESPONSE_BYTES: usize = 16 * 1024 * 1024;

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum FreshMetamorphicFormV1 {
    Base,
    SwapRows,
    ScaleFirstRowByTwo,
    ScaleSecondRowByNegativeTwo,
    SwapAndScaleRows,
}

pub const FRESH_METAMORPHIC_FORMS: [FreshMetamorphicFormV1; FRESH_METAMORPHIC_FORMS_PER_GROUP] = [
    FreshMetamorphicFormV1::Base,
    FreshMetamorphicFormV1::SwapRows,
    FreshMetamorphicFormV1::ScaleFirstRowByTwo,
    FreshMetamorphicFormV1::ScaleSecondRowByNegativeTwo,
    FreshMetamorphicFormV1::SwapAndScaleRows,
];

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct FreshMetamorphicTaskV1 {
    pub item_id: String,
    pub group_id: String,
    pub parent_item_id: String,
    pub form: FreshMetamorphicFormV1,
    pub task_text: String,
    pub task_text_sha256: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FreshMetamorphicPublicPacketV1 {
    pub schema: String,
    pub batch_id: String,
    pub benchmark_id: String,
    pub benchmark_version: String,
    pub task_family: String,
    pub model_checkpoint_sha256: String,
    pub base_public_packet_sha256: String,
    pub base_manifest_sha256: String,
    pub item_count: u64,
    pub group_count: u64,
    pub forms_per_group: u64,
    pub tasks: Vec<FreshMetamorphicTaskV1>,
    pub public_packet_sha256: String,
    pub claim_boundary: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FreshMetamorphicOracleEntryV1 {
    pub item_id: String,
    pub group_id: String,
    pub parent_item_id: String,
    pub form: FreshMetamorphicFormV1,
    pub parameters: LinearSystemParametersV1,
    pub solution: ExactLinearSolutionV1,
    pub normalized_determinant_margin_permille: u16,
    pub parent_answer_sha256: String,
    pub parameter_sha256: String,
    pub answer_sha256: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FreshMetamorphicSealedOracleV1 {
    pub schema: String,
    pub batch_id: String,
    pub base_sealed_oracle_sha256: String,
    pub generation_seed_commitment_sha256: String,
    pub item_count: u64,
    pub group_count: u64,
    pub entries: Vec<FreshMetamorphicOracleEntryV1>,
    pub sealed_oracle_sha256: String,
    pub claim_boundary: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FreshMetamorphicManifestV1 {
    pub schema: String,
    pub batch_id: String,
    pub benchmark_id: String,
    pub benchmark_version: String,
    pub task_family: String,
    pub model_checkpoint_sha256: String,
    pub base_public_packet_sha256: String,
    pub base_manifest_sha256: String,
    pub base_sealed_oracle_sha256: String,
    pub public_packet_sha256: String,
    pub sealed_oracle_sha256: String,
    pub transformation_rules_sha256: String,
    pub item_set_sha256: String,
    pub group_count: u64,
    pub item_count: u64,
    pub forms_per_group: u64,
    pub manifest_sha256: String,
    pub claim_boundary: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FreshMetamorphicBundleV1 {
    pub public_packet: FreshMetamorphicPublicPacketV1,
    pub sealed_oracle: FreshMetamorphicSealedOracleV1,
    pub manifest: FreshMetamorphicManifestV1,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FreshMetamorphicAuditV1 {
    pub schema: String,
    pub valid: bool,
    pub deterministic_regeneration_matches: bool,
    pub base_bundle_validated: bool,
    pub all_variants_preserve_exact_solution: bool,
    pub all_variants_preserve_stability_margin: bool,
    pub all_groups_have_exact_form_universe: bool,
    pub public_packet_contains_no_oracle_fields: bool,
    pub group_count: u64,
    pub item_count: u64,
    pub forms_per_group: u64,
    pub manifest_sha256: String,
    pub independently_verified: bool,
    pub claim_boundary: String,
    pub issues: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FreshMetamorphicResponseV1 {
    pub item_id: String,
    pub response_text: String,
    pub response_sha256: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FreshMetamorphicSubmissionV1 {
    pub schema: String,
    pub run_id: String,
    pub model_provider: String,
    pub model_name: String,
    pub model_version: String,
    pub model_access_mode: String,
    pub model_checkpoint_sha256: String,
    pub base_public_packet_sha256: String,
    pub base_manifest_sha256: String,
    pub metamorphic_public_packet_sha256: String,
    pub metamorphic_manifest_sha256: String,
    pub inference_harness_sha256: String,
    pub prompt_and_decoding_sha256: String,
    pub metric_definition_sha256: String,
    pub run_execution_receipt_sha256: String,
    pub model_call_count: u64,
    pub input_token_count: u64,
    pub output_token_count: u64,
    pub wall_clock_millis: u64,
    pub tool_call_count: u64,
    pub human_assistance_minutes: u64,
    pub compute_cost_microusd: u64,
    pub responses: Vec<FreshMetamorphicResponseV1>,
    pub submission_sha256: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FreshMetamorphicItemScoreV1 {
    pub item_id: String,
    pub group_id: String,
    pub parent_item_id: String,
    pub form: FreshMetamorphicFormV1,
    pub response_sha256: String,
    pub status: FreshItemScoreStatusV1,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FreshMetamorphicGroupScoreV1 {
    pub group_id: String,
    pub item_count: u64,
    pub valid_response_count: u64,
    pub correct_count: u64,
    pub complete_valid_group: bool,
    pub outputs_agree: Option<bool>,
    pub fully_correct_and_consistent: bool,
    pub consistent_but_wrong: bool,
    pub base_correct_variant_inconsistent: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FreshMetamorphicScoreReportV1 {
    pub schema: String,
    pub metric_id: String,
    pub run_id: String,
    pub model_provider: String,
    pub model_name: String,
    pub model_version: String,
    pub model_access_mode: String,
    pub model_checkpoint_sha256: String,
    pub batch_id: String,
    pub benchmark_id: String,
    pub benchmark_version: String,
    pub task_family: String,
    pub base_public_packet_sha256: String,
    pub base_manifest_sha256: String,
    pub base_sealed_oracle_sha256: String,
    pub metamorphic_public_packet_sha256: String,
    pub metamorphic_manifest_sha256: String,
    pub metamorphic_sealed_oracle_sha256: String,
    pub submission_sha256: String,
    pub inference_harness_sha256: String,
    pub prompt_and_decoding_sha256: String,
    pub metric_definition_sha256: String,
    pub run_execution_receipt_sha256: String,
    pub item_count: u64,
    pub base_item_count: u64,
    pub transformed_item_count: u64,
    pub correct_count: u64,
    pub incorrect_count: u64,
    pub invalid_format_count: u64,
    pub abstained_count: u64,
    pub valid_response_count: u64,
    pub base_correct_count: u64,
    pub transformed_correct_count: u64,
    pub accuracy_all_forms_permille: f64,
    pub accuracy_base_form_permille: f64,
    pub accuracy_transformed_forms_permille: f64,
    pub coverage_permille: f64,
    pub group_count: u64,
    pub complete_valid_group_count: u64,
    pub incomplete_group_count: u64,
    pub consistent_group_count: u64,
    pub disagreement_group_count: u64,
    pub consistent_but_wrong_group_count: u64,
    pub fully_correct_and_consistent_group_count: u64,
    pub base_correct_variant_inconsistent_group_count: u64,
    pub metamorphic_consistency_permille: Option<f64>,
    pub model_call_count: u64,
    pub input_token_count: u64,
    pub output_token_count: u64,
    pub wall_clock_millis: u64,
    pub tool_call_count: u64,
    pub human_assistance_minutes: u64,
    pub compute_cost_microusd: u64,
    pub independently_verified: bool,
    pub items: Vec<FreshMetamorphicItemScoreV1>,
    pub groups: Vec<FreshMetamorphicGroupScoreV1>,
    pub report_sha256: String,
    pub claim_boundary: String,
}

fn is_digest(value: &str) -> bool {
    value.len() == 64
        && value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn validate_digest(value: &str, label: &str) -> Result<(), String> {
    if is_digest(value) {
        Ok(())
    } else {
        Err(format!("{label} must be lowercase SHA-256 hex"))
    }
}

fn determinant(parameters: &LinearSystemParametersV1) -> i64 {
    i64::from(parameters.a11) * i64::from(parameters.a22)
        - i64::from(parameters.a12) * i64::from(parameters.a21)
}

fn compute_rhs(
    parameters: &LinearSystemParametersV1,
    solution: &ExactLinearSolutionV1,
) -> Result<(i64, i64), String> {
    let b1 = i64::from(parameters.a11)
        .checked_mul(i64::from(solution.x))
        .and_then(|value| {
            i64::from(parameters.a12)
                .checked_mul(i64::from(solution.y))
                .and_then(|right| value.checked_add(right))
        })
        .ok_or_else(|| "metamorphic rhs b1 overflowed".to_owned())?;
    let b2 = i64::from(parameters.a21)
        .checked_mul(i64::from(solution.x))
        .and_then(|value| {
            i64::from(parameters.a22)
                .checked_mul(i64::from(solution.y))
                .and_then(|right| value.checked_add(right))
        })
        .ok_or_else(|| "metamorphic rhs b2 overflowed".to_owned())?;
    Ok((b1, b2))
}

fn scale_i32(value: i32, factor: i32, label: &str) -> Result<i32, String> {
    if factor == 0 {
        return Err("row scaling factor must be nonzero".to_owned());
    }
    value.checked_mul(factor).ok_or_else(|| format!("{label} overflows i32 under row scaling"))
}

fn scale_i64(value: i64, factor: i64, label: &str) -> Result<i64, String> {
    if factor == 0 {
        return Err("row scaling factor must be nonzero".to_owned());
    }
    value.checked_mul(factor).ok_or_else(|| format!("{label} overflows i64 under row scaling"))
}

fn apply_form(
    base: &LinearSystemParametersV1,
    form: FreshMetamorphicFormV1,
) -> Result<LinearSystemParametersV1, String> {
    let row1 = (base.a11, base.a12, base.b1);
    let row2 = (base.a21, base.a22, base.b2);
    let (new_row1, new_row2) = match form {
        FreshMetamorphicFormV1::Base => (row1, row2),
        FreshMetamorphicFormV1::SwapRows => (row2, row1),
        FreshMetamorphicFormV1::ScaleFirstRowByTwo => (
            (
                scale_i32(row1.0, 2, "a11")?,
                scale_i32(row1.1, 2, "a12")?,
                scale_i64(row1.2, 2, "b1")?,
            ),
            row2,
        ),
        FreshMetamorphicFormV1::ScaleSecondRowByNegativeTwo => (
            row1,
            (
                scale_i32(row2.0, -2, "a21")?,
                scale_i32(row2.1, -2, "a22")?,
                scale_i64(row2.2, -2, "b2")?,
            ),
        ),
        FreshMetamorphicFormV1::SwapAndScaleRows => (
            (
                scale_i32(row2.0, 3, "a11")?,
                scale_i32(row2.1, 3, "a12")?,
                scale_i64(row2.2, 3, "b1")?,
            ),
            (
                scale_i32(row1.0, -2, "a21")?,
                scale_i32(row1.1, -2, "a22")?,
                scale_i64(row1.2, -2, "b2")?,
            ),
        ),
    };
    Ok(LinearSystemParametersV1 {
        a11: new_row1.0,
        a12: new_row1.1,
        a21: new_row2.0,
        a22: new_row2.1,
        b1: new_row1.2,
        b2: new_row2.2,
    })
}

fn variant_id(parent_item_id: &str, form: FreshMetamorphicFormV1) -> String {
    match form {
        FreshMetamorphicFormV1::Base => parent_item_id.to_owned(),
        FreshMetamorphicFormV1::SwapRows => format!("{parent_item_id}__swap_rows"),
        FreshMetamorphicFormV1::ScaleFirstRowByTwo => format!("{parent_item_id}__scale_row1_x2"),
        FreshMetamorphicFormV1::ScaleSecondRowByNegativeTwo => {
            format!("{parent_item_id}__scale_row2_xminus2")
        }
        FreshMetamorphicFormV1::SwapAndScaleRows => {
            format!("{parent_item_id}__swap_scale_rows")
        }
    }
}

fn public_packet_digest(packet: &FreshMetamorphicPublicPacketV1) -> Result<String, String> {
    let mut normalized = packet.clone();
    normalized.public_packet_sha256.clear();
    let bytes = serde_json::to_vec(&normalized)
        .map_err(|error| format!("cannot serialize metamorphic public packet: {error}"))?;
    Ok(hash_domain(b"fresh-metamorphic-public-packet.v1", &bytes))
}

fn oracle_digest(oracle: &FreshMetamorphicSealedOracleV1) -> Result<String, String> {
    let mut normalized = oracle.clone();
    normalized.sealed_oracle_sha256.clear();
    let bytes = serde_json::to_vec(&normalized)
        .map_err(|error| format!("cannot serialize metamorphic oracle: {error}"))?;
    Ok(hash_domain(b"fresh-metamorphic-sealed-oracle.v1", &bytes))
}

fn item_set_digest(tasks: &[FreshMetamorphicTaskV1]) -> Result<String, String> {
    let items: Vec<(&str, &str, &str)> = tasks.iter()
        .map(|task| (
            task.item_id.as_str(),
            task.group_id.as_str(),
            task.task_text_sha256.as_str(),
        ))
        .collect();
    let bytes = serde_json::to_vec(&items)
        .map_err(|error| format!("cannot serialize metamorphic item set: {error}"))?;
    Ok(hash_domain(b"fresh-metamorphic-item-set.v1", &bytes))
}

fn manifest_digest(manifest: &FreshMetamorphicManifestV1) -> Result<String, String> {
    let mut normalized = manifest.clone();
    normalized.manifest_sha256.clear();
    let bytes = serde_json::to_vec(&normalized)
        .map_err(|error| format!("cannot serialize metamorphic manifest: {error}"))?;
    Ok(hash_domain(b"fresh-metamorphic-manifest.v1", &bytes))
}

fn task_parameters_digest(parameters: &LinearSystemParametersV1) -> Result<String, String> {
    let bytes = serde_json::to_vec(parameters)
        .map_err(|error| format!("cannot serialize metamorphic parameters: {error}"))?;
    Ok(hash_domain(b"fresh-task-parameters.v1", &bytes))
}

fn task_answer_digest(solution: &ExactLinearSolutionV1) -> Result<String, String> {
    let bytes = serde_json::to_vec(solution)
        .map_err(|error| format!("cannot serialize metamorphic solution: {error}"))?;
    Ok(hash_domain(b"fresh-task-answer.v1", &bytes))
}

fn task_text_digest(text: &str) -> String {
    hash_domain(b"fresh-task-text.v1", text.as_bytes())
}

fn transformation_rules_digest() -> Result<String, String> {
    let rules = [
        FreshMetamorphicFormV1::Base,
        FreshMetamorphicFormV1::SwapRows,
        FreshMetamorphicFormV1::ScaleFirstRowByTwo,
        FreshMetamorphicFormV1::ScaleSecondRowByNegativeTwo,
        FreshMetamorphicFormV1::SwapAndScaleRows,
    ];
    let bytes = serde_json::to_vec(&rules)
        .map_err(|error| format!("cannot serialize transformation rules: {error}"))?;
    Ok(hash_domain(b"fresh-metamorphic-transformation-rules.v1", &bytes))
}

pub fn generate_fresh_metamorphic_bundle(
    base: &FreshChallengeBundleV1,
) -> Result<FreshMetamorphicBundleV1, String> {
    let base_answers = validate_bundle(base)?;
    let base_oracle_by_id: BTreeMap<&str, _> = base.sealed_oracle_packet.entries.iter()
        .map(|entry| (entry.item_id.as_str(), entry))
        .collect();
    if base.public_packet.tasks.len() > FRESH_METAMORPHIC_MAX_ITEMS
            .checked_div(FRESH_METAMORPHIC_FORMS_PER_GROUP)
            .unwrap_or(0)
    {
        return Err("base item count exceeds metamorphic expansion safety bound".to_owned());
    }
    let mut tasks = Vec::with_capacity(
        base.public_packet.tasks.len() * FRESH_METAMORPHIC_FORMS_PER_GROUP
    );
    let mut oracle_entries = Vec::with_capacity(tasks.capacity());
    let mut item_ids = BTreeSet::new();

    for base_task in &base.public_packet.tasks {
        let base_entry = base_oracle_by_id.get(base_task.item_id.as_str())
            .ok_or_else(|| format!("base oracle entry missing for {}", base_task.item_id))?;
        let expected_solution = base_answers.get(&base_task.item_id)
            .ok_or_else(|| format!("base expected solution missing for {}", base_task.item_id))?;
        if base_entry.solution != *expected_solution {
            return Err(format!("base answer map mismatch for {}", base_task.item_id));
        }
        for form in FRESH_METAMORPHIC_FORMS {
            let item_id = variant_id(&base_task.item_id, form);
            if !item_ids.insert(item_id.clone()) {
                return Err("metamorphic generator produced duplicate variant IDs".to_owned());
            }
            let parameters = apply_form(&base_entry.parameters, form)?;
            let transformed_determinant = determinant(&parameters);
            if transformed_determinant == 0 {
                return Err(format!("metamorphic transform produced a singular matrix for {item_id}"));
            }
            let transformed_margin = normalized_determinant_margin_permille(&parameters);
            if transformed_margin != base_entry.normalized_determinant_margin_permille {
                return Err(format!("metamorphic transform changed normalized stability margin for {item_id}"));
            }
            let (b1, b2) = compute_rhs(&parameters, expected_solution)?;
            if b1 != parameters.b1 || b2 != parameters.b2 {
                return Err(format!("metamorphic transform failed exact solution preservation for {item_id}"));
            }
            let text = task_text(&parameters);
            let text_sha256 = task_text_digest(&text);
            if form == FreshMetamorphicFormV1::Base
                && (text != base_task.task_text || text_sha256 != base_task.task_text_sha256)
            {
                return Err(format!("base form changed while constructing metamorphic group {}", base_task.item_id));
            }
            let parameter_sha256 = task_parameters_digest(&parameters)?;
            let answer_sha256 = task_answer_digest(expected_solution)?;
            tasks.push(FreshMetamorphicTaskV1 {
                item_id: item_id.clone(),
                group_id: base_task.item_id.clone(),
                parent_item_id: base_task.item_id.clone(),
                form,
                task_text: text,
                task_text_sha256: text_sha256,
            });
            oracle_entries.push(FreshMetamorphicOracleEntryV1 {
                item_id,
                group_id: base_task.item_id.clone(),
                parent_item_id: base_task.item_id.clone(),
                form,
                parameters,
                solution: expected_solution.clone(),
                normalized_determinant_margin_permille: transformed_margin,
                parent_answer_sha256: base_entry.answer_sha256.clone(),
                parameter_sha256,
                answer_sha256,
            });
        }
    }
    let group_count = u64::try_from(base.public_packet.tasks.len())
        .map_err(|_| "metamorphic group count exceeds u64")?;
    let item_count = u64::try_from(tasks.len())
        .map_err(|_| "metamorphic item count exceeds u64")?;
    let mut public_packet = FreshMetamorphicPublicPacketV1 {
        schema: FRESH_METAMORPHIC_PACKET_SCHEMA.to_owned(),
        batch_id: base.public_packet.batch_id.clone(),
        benchmark_id: base.public_packet.benchmark_id.clone(),
        benchmark_version: base.public_packet.benchmark_version.clone(),
        task_family: base.public_packet.task_family.clone(),
        model_checkpoint_sha256: base.public_packet.model_checkpoint_sha256.clone(),
        base_public_packet_sha256: base.public_packet.public_packet_sha256.clone(),
        base_manifest_sha256: base.manifest.manifest_sha256.clone(),
        item_count,
        group_count,
        forms_per_group: FRESH_METAMORPHIC_FORMS_PER_GROUP as u64,
        tasks,
        public_packet_sha256: String::new(),
        claim_boundary: "metamorphic_public_variants_preserve_registered_linear_system_solution;no_claim_of_general_reasoning_or_task_recipe_novelty".to_owned(),
    };
    public_packet.public_packet_sha256 = public_packet_digest(&public_packet)?;

    let mut sealed_oracle = FreshMetamorphicSealedOracleV1 {
        schema: FRESH_METAMORPHIC_ORACLE_SCHEMA.to_owned(),
        batch_id: base.public_packet.batch_id.clone(),
        base_sealed_oracle_sha256: base.sealed_oracle_packet.sealed_oracle_sha256.clone(),
        generation_seed_commitment_sha256: base.sealed_oracle_packet.generation_seed_commitment_sha256.clone(),
        item_count,
        group_count,
        entries: oracle_entries,
        sealed_oracle_sha256: String::new(),
        claim_boundary: "evaluator_only_exact_parameters_and_answers_for_equivalent_forms".to_owned(),
    };
    sealed_oracle.sealed_oracle_sha256 = oracle_digest(&sealed_oracle)?;
    let item_set_sha256 = item_set_digest(&public_packet.tasks)?;
    let mut manifest = FreshMetamorphicManifestV1 {
        schema: FRESH_METAMORPHIC_MANIFEST_SCHEMA.to_owned(),
        batch_id: base.public_packet.batch_id.clone(),
        benchmark_id: base.public_packet.benchmark_id.clone(),
        benchmark_version: base.public_packet.benchmark_version.clone(),
        task_family: base.public_packet.task_family.clone(),
        model_checkpoint_sha256: base.public_packet.model_checkpoint_sha256.clone(),
        base_public_packet_sha256: base.public_packet.public_packet_sha256.clone(),
        base_manifest_sha256: base.manifest.manifest_sha256.clone(),
        base_sealed_oracle_sha256: base.sealed_oracle_packet.sealed_oracle_sha256.clone(),
        public_packet_sha256: public_packet.public_packet_sha256.clone(),
        sealed_oracle_sha256: sealed_oracle.sealed_oracle_sha256.clone(),
        transformation_rules_sha256: transformation_rules_digest()?,
        item_set_sha256,
        group_count,
        item_count,
        forms_per_group: FRESH_METAMORPHIC_FORMS_PER_GROUP as u64,
        manifest_sha256: String::new(),
        claim_boundary: "base_fresh_bundle_and_registered_equivalence_rules_are_explicitly_bound;no_external_semantic_claim".to_owned(),
    };
    manifest.manifest_sha256 = manifest_digest(&manifest)?;
    Ok(FreshMetamorphicBundleV1 {
        public_packet,
        sealed_oracle,
        manifest,
    })
}

pub fn audit_fresh_metamorphic_bundle(
    base: &FreshChallengeBundleV1,
    supplied: &FreshMetamorphicBundleV1,
) -> Result<FreshMetamorphicAuditV1, String> {
    let expected = generate_fresh_metamorphic_bundle(base)?;
    let supplied_bytes = serde_json::to_vec(supplied)
        .map_err(|error| format!("cannot serialize supplied metamorphic bundle: {error}"))?;
    let expected_bytes = serde_json::to_vec(&expected)
        .map_err(|error| format!("cannot serialize expected metamorphic bundle: {error}"))?;
    if supplied_bytes != expected_bytes {
        return Err("metamorphic bundle differs from deterministic regeneration".to_owned());
    }
    if public_packet_digest(&supplied.public_packet)? != supplied.public_packet.public_packet_sha256
        || oracle_digest(&supplied.sealed_oracle)? != supplied.sealed_oracle.sealed_oracle_sha256
        || manifest_digest(&supplied.manifest)? != supplied.manifest.manifest_sha256
        || item_set_digest(&supplied.public_packet.tasks)? != supplied.manifest.item_set_sha256
    {
        return Err("metamorphic artifact digest mismatch".to_owned());
    }
    let mut exact_preservation = true;
    let mut margin_preservation = true;
    let mut groups: BTreeMap<&str, Vec<&FreshMetamorphicOracleEntryV1>> = BTreeMap::new();
    for entry in &supplied.sealed_oracle.entries {
        groups.entry(entry.group_id.as_str()).or_default().push(entry);
        let (b1, b2) = compute_rhs(&entry.parameters, &entry.solution)?;
        exact_preservation &= b1 == entry.parameters.b1 && b2 == entry.parameters.b2;
        let observed_margin = normalized_determinant_margin_permille(&entry.parameters);
        margin_preservation &= observed_margin == entry.normalized_determinant_margin_permille;
        validate_digest(&entry.parent_answer_sha256, "parent_answer_sha256")?;
        validate_digest(&entry.parameter_sha256, "parameter_sha256")?;
        validate_digest(&entry.answer_sha256, "answer_sha256")?;
        exact_preservation &= task_parameters_digest(&entry.parameters)? == entry.parameter_sha256
            && task_answer_digest(&entry.solution)? == entry.answer_sha256;
    }
    let forms: BTreeSet<FreshMetamorphicFormV1> = FRESH_METAMORPHIC_FORMS.into_iter().collect();
    let mut all_groups_exact = true;
    for group in groups.values() {
        let found: BTreeSet<FreshMetamorphicFormV1> = group.iter().map(|entry| entry.form).collect();
        let same_answer = group.iter().all(|entry| entry.solution == group[0].solution);
        all_groups_exact &= found == forms
            && group.len() == FRESH_METAMORPHIC_FORMS_PER_GROUP
            && same_answer;
    }
    if !exact_preservation || !margin_preservation || !all_groups_exact {
        return Err("metamorphic oracle violates exact solution, stability margin or form-universe invariants".to_owned());
    }
    let public_value = serde_json::to_value(&supplied.public_packet)
        .map_err(|error| format!("cannot serialize metamorphic public packet: {error}"))?;
    let public_keys: BTreeSet<String> = public_value.as_object()
        .ok_or_else(|| "metamorphic public packet must be a JSON object".to_owned())?
        .keys().cloned().collect();
    let expected_public_keys: BTreeSet<String> = [
        "schema", "batch_id", "benchmark_id", "benchmark_version", "task_family",
        "model_checkpoint_sha256", "base_public_packet_sha256", "base_manifest_sha256",
        "item_count", "group_count", "forms_per_group", "tasks",
        "public_packet_sha256", "claim_boundary",
    ].iter().map(|field| (*field).to_owned()).collect();
    let task_fields_valid = supplied.public_packet.tasks.iter().all(|task| {
        serde_json::to_value(task).ok().and_then(|value| {
            let actual: BTreeSet<String> = value.as_object()?.keys().cloned().collect();
            let expected: BTreeSet<String> = [
                "item_id", "group_id", "parent_item_id", "form", "task_text", "task_text_sha256",
            ].iter().map(|field| (*field).to_owned()).collect();
            Some(actual == expected)
        }).unwrap_or(false)
    });
    if public_keys != expected_public_keys
        || !task_fields_valid
        || public_value.to_string().contains(&supplied.sealed_oracle.generation_seed_commitment_sha256)
    {
        return Err("metamorphic public packet contains evaluator-only or unexpected fields".to_owned());
    }
    Ok(FreshMetamorphicAuditV1 {
        schema: FRESH_METAMORPHIC_AUDIT_SCHEMA.to_owned(),
        valid: true,
        deterministic_regeneration_matches: true,
        base_bundle_validated: true,
        all_variants_preserve_exact_solution: exact_preservation,
        all_variants_preserve_stability_margin: margin_preservation,
        all_groups_have_exact_form_universe: all_groups_exact,
        public_packet_contains_no_oracle_fields: true,
        group_count: supplied.manifest.group_count,
        item_count: supplied.manifest.item_count,
        forms_per_group: supplied.manifest.forms_per_group,
        manifest_sha256: supplied.manifest.manifest_sha256.clone(),
        independently_verified: false,
        claim_boundary: "deterministic_registered_linear_system_equivalence_checked;checkpoint_authenticity_and_general_reasoning_not_verified".to_owned(),
        issues: vec![
            "metamorphic relations cover row permutation and nonzero row scaling only".to_owned(),
            "psychometric item difficulty is not calibrated by this generator".to_owned(),
        ],
    })
}

fn response_digest(text: &str) -> String {
    hash_domain(b"fresh-metamorphic-response.v1", text.as_bytes())
}

fn submission_digest(submission: &FreshMetamorphicSubmissionV1) -> Result<String, String> {
    let mut normalized = submission.clone();
    normalized.submission_sha256.clear();
    let bytes = serde_json::to_vec(&normalized)
        .map_err(|error| format!("cannot serialize metamorphic submission: {error}"))?;
    Ok(hash_domain(b"fresh-metamorphic-submission.v1", &bytes))
}

fn report_digest(report: &FreshMetamorphicScoreReportV1) -> Result<String, String> {
    let mut normalized = report.clone();
    normalized.report_sha256.clear();
    let bytes = serde_json::to_vec(&normalized)
        .map_err(|error| format!("cannot serialize metamorphic score report: {error}"))?;
    Ok(hash_domain(b"fresh-metamorphic-score-report.v1", &bytes))
}

fn rate(numerator: u64, denominator: u64) -> Option<f64> {
    if denominator == 0 {
        None
    } else {
        Some(numerator as f64 * 1_000.0 / denominator as f64)
    }
}

pub fn score_fresh_metamorphic_submission(
    base: &FreshChallengeBundleV1,
    bundle: &FreshMetamorphicBundleV1,
    submission_bytes: &[u8],
) -> Result<FreshMetamorphicScoreReportV1, String> {
    audit_fresh_metamorphic_bundle(base, bundle)?;
    if submission_bytes.len() > FRESH_METAMORPHIC_MAX_SUBMISSION_BYTES {
        return Err("metamorphic submission exceeds 32 MiB safety bound".to_owned());
    }
    let submission: FreshMetamorphicSubmissionV1 = serde_json::from_slice(submission_bytes)
        .map_err(|error| format!("metamorphic submission does not match strict v1 JSON: {error}"))?;
    if submission.schema != FRESH_METAMORPHIC_SUBMISSION_SCHEMA {
        return Err(format!("metamorphic submission schema must be {FRESH_METAMORPHIC_SUBMISSION_SCHEMA}"));
    }
    for (value, label) in [
        (&submission.model_checkpoint_sha256, "model_checkpoint_sha256"),
        (&submission.base_public_packet_sha256, "base_public_packet_sha256"),
        (&submission.metamorphic_public_packet_sha256, "metamorphic_public_packet_sha256"),
        (&submission.inference_harness_sha256, "inference_harness_sha256"),
        (&submission.prompt_and_decoding_sha256, "prompt_and_decoding_sha256"),
        (&submission.metric_definition_sha256, "metric_definition_sha256"),
        (&submission.run_execution_receipt_sha256, "run_execution_receipt_sha256"),
        (&submission.submission_sha256, "submission_sha256"),
    ] {
        validate_digest(value, label)?;
    }
    if submission.run_id.trim().is_empty()
        || submission.model_provider.trim().is_empty()
        || submission.model_name.trim().is_empty()
        || submission.model_version.trim().is_empty()
        || submission.model_access_mode.trim().is_empty()
    {
        return Err("metamorphic run and model identity fields must be nonempty".to_owned());
    }
    if submission.model_checkpoint_sha256 != bundle.public_packet.model_checkpoint_sha256
        || submission.base_public_packet_sha256 != bundle.public_packet.base_public_packet_sha256
        || submission.base_manifest_sha256 != bundle.public_packet.base_manifest_sha256
        || submission.metamorphic_public_packet_sha256 != bundle.public_packet.public_packet_sha256
        || submission.metamorphic_manifest_sha256 != bundle.manifest.manifest_sha256
    {
        return Err("metamorphic submission checkpoint or packet binding mismatch".to_owned());
    }
    if submission.metric_definition_sha256
        != hash_domain(b"fresh-metric-definition.v1", FRESH_METAMORPHIC_METRIC_ID.as_bytes())
    {
        return Err("metamorphic metric definition digest mismatch".to_owned());
    }
    if submission.model_call_count == 0 || submission.wall_clock_millis == 0 {
        return Err("metamorphic model_call_count and wall_clock_millis must be positive".to_owned());
    }
    if submission.responses.len() != bundle.public_packet.tasks.len() {
        return Err("metamorphic submission must contain exactly one response per public variant".to_owned());
    }
    let response_bytes = submission.responses.iter().try_fold(0_usize, |total, response| {
        if response.response_text.len() > FRESH_METAMORPHIC_MAX_RESPONSE_BYTES {
            return Err("metamorphic individual response exceeds 1 MiB safety bound".to_owned());
        }
        total.checked_add(response.response_text.len())
            .ok_or_else(|| "metamorphic response byte count overflowed".to_owned())
    })?;
    if response_bytes > FRESH_METAMORPHIC_MAX_TOTAL_RESPONSE_BYTES {
        return Err("metamorphic aggregate responses exceed 16 MiB safety bound".to_owned());
    }
    if submission_digest(&submission)? != submission.submission_sha256 {
        return Err("metamorphic submission digest does not match contents".to_owned());
    }

    let task_by_id: BTreeMap<&str, &FreshMetamorphicTaskV1> = bundle.public_packet.tasks.iter()
        .map(|task| (task.item_id.as_str(), task))
        .collect();
    let oracle_by_id: BTreeMap<&str, &FreshMetamorphicOracleEntryV1> =
        bundle.sealed_oracle.entries.iter().map(|entry| (entry.item_id.as_str(), entry)).collect();
    let mut response_by_id = BTreeMap::new();
    for response in &submission.responses {
        if response.item_id.trim().is_empty()
            || response_by_id.insert(response.item_id.as_str(), response).is_some()
        {
            return Err("metamorphic response item IDs must be unique and nonempty".to_owned());
        }
        let task = task_by_id.get(response.item_id.as_str())
            .ok_or_else(|| format!("unknown metamorphic item {}", response.item_id))?;
        validate_digest(&response.response_sha256, "response_sha256")?;
        if response_digest(&response.response_text) != response.response_sha256 {
            return Err(format!("metamorphic response digest mismatch for {}", response.item_id));
        }
        if task.parent_item_id != task.group_id {
            return Err(format!("metamorphic task {} has invalid parent/group binding", task.item_id));
        }
    }
    if response_by_id.len() != task_by_id.len() {
        return Err("metamorphic response item set is incomplete".to_owned());
    }

    let mut item_scores = Vec::with_capacity(submission.responses.len());
    let mut status_by_id = BTreeMap::new();
    let mut parsed_by_id: BTreeMap<&str, ExactLinearSolutionV1> = BTreeMap::new();
    let mut correct_count = 0_u64;
    let mut incorrect_count = 0_u64;
    let mut invalid_count = 0_u64;
    let mut abstained_count = 0_u64;
    let mut base_correct_count = 0_u64;
    let mut transformed_correct_count = 0_u64;
    let base_ids: BTreeSet<&str> =
        base.public_packet.tasks.iter().map(|task| task.item_id.as_str()).collect();
    for response in &submission.responses {
        let task = task_by_id.get(response.item_id.as_str()).ok_or("unknown task")?;
        let oracle = oracle_by_id.get(response.item_id.as_str()).ok_or("missing metamorphic oracle")?;
        let status = if response.response_text.trim().is_empty() {
            abstained_count += 1;
            FreshItemScoreStatusV1::Abstained
        } else {
            match serde_json::from_str::<ExactLinearSolutionV1>(&response.response_text) {
                Ok(parsed) if parsed == oracle.solution => {
                    correct_count += 1;
                    if task.form == FreshMetamorphicFormV1::Base {
                        base_correct_count += 1;
                    } else {
                        transformed_correct_count += 1;
                    }
                    parsed_by_id.insert(response.item_id.as_str(), parsed);
                    FreshItemScoreStatusV1::Correct
                }
                Ok(parsed) => {
                    incorrect_count += 1;
                    if task.form == FreshMetamorphicFormV1::Base {
                        parsed_by_id.insert(response.item_id.as_str(), parsed);
                    } else {
                        parsed_by_id.insert(response.item_id.as_str(), parsed);
                    }
                    FreshItemScoreStatusV1::Incorrect
                }
                Err(_) => {
                    invalid_count += 1;
                    FreshItemScoreStatusV1::InvalidFormat
                }
            }
        };
        status_by_id.insert(response.item_id.as_str(), status);
        item_scores.push(FreshMetamorphicItemScoreV1 {
            item_id: task.item_id.clone(),
            group_id: task.group_id.clone(),
            parent_item_id: task.parent_item_id.clone(),
            form: task.form,
            response_sha256: response.response_sha256.clone(),
            status,
        });
    }

    let mut group_scores = Vec::with_capacity(bundle.public_packet.group_count as usize);
    let mut complete_valid_groups = 0_u64;
    let mut incomplete_groups = 0_u64;
    let mut consistent_groups = 0_u64;
    let mut disagreement_groups = 0_u64;
    let mut consistent_but_wrong_groups = 0_u64;
    let mut fully_correct_and_consistent_groups = 0_u64;
    let mut base_correct_variant_inconsistent_groups = 0_u64;
    let groups: BTreeMap<&str, Vec<&FreshMetamorphicTaskV1>> =
        bundle.public_packet.tasks.iter().fold(BTreeMap::new(), |mut map, task| {
            map.entry(task.group_id.as_str()).or_default().push(task);
            map
        });
    for (group_id, tasks) in &groups {
        if tasks.len() != FRESH_METAMORPHIC_FORMS_PER_GROUP {
            return Err(format!("metamorphic group {group_id} does not contain five forms"));
        }
        let valid_response_count = tasks.iter()
            .filter(|task| matches!(
                status_by_id.get(task.item_id.as_str()),
                Some(&FreshItemScoreStatusV1::Correct) | Some(&FreshItemScoreStatusV1::Incorrect)
            ))
            .count() as u64;
        let correct_in_group = tasks.iter()
            .filter(|task| status_by_id.get(task.item_id.as_str())
                == Some(&FreshItemScoreStatusV1::Correct))
            .count() as u64;
        let complete = valid_response_count == tasks.len() as u64;
        let outputs: Vec<&ExactLinearSolutionV1> = tasks.iter()
            .filter_map(|task| parsed_by_id.get(task.item_id.as_str()))
            .collect();
        let outputs_agree = if complete {
            Some(outputs.windows(2).all(|pair| pair[0] == pair[1]))
        } else {
            None
        };
        if complete {
            complete_valid_groups += 1;
            if outputs_agree == Some(true) {
                consistent_groups += 1;
                if correct_in_group == tasks.len() as u64 {
                    fully_correct_and_consistent_groups += 1;
                } else if correct_in_group == 0 {
                    consistent_but_wrong_groups += 1;
                }
            } else {
                disagreement_groups += 1;
            }
        } else {
            incomplete_groups += 1;
        }
        let base_task = tasks.iter().find(|task| task.form == FreshMetamorphicFormV1::Base)
            .ok_or_else(|| format!("metamorphic group {group_id} has no base form"))?;
        let base_correct = status_by_id.get(base_task.item_id.as_str())
            == Some(&FreshItemScoreStatusV1::Correct);
        let base_output = parsed_by_id.get(base_task.item_id.as_str());
        let variant_disagrees = tasks.iter()
            .filter(|task| task.form != FreshMetamorphicFormV1::Base)
            .any(|task| parsed_by_id.get(task.item_id.as_str())
                .is_some_and(|output| base_output.is_some_and(|base| output != base)));
        if complete && base_correct && variant_disagrees {
            base_correct_variant_inconsistent_groups += 1;
        }
        group_scores.push(FreshMetamorphicGroupScoreV1 {
            group_id: (*group_id).to_owned(),
            item_count: tasks.len() as u64,
            valid_response_count,
            correct_count: correct_in_group,
            complete_valid_group: complete,
            outputs_agree,
            fully_correct_and_consistent: complete
                && outputs_agree == Some(true)
                && correct_in_group == tasks.len() as u64,
            consistent_but_wrong: complete
                && outputs_agree == Some(true)
                && correct_in_group == 0,
            base_correct_variant_inconsistent: complete && base_correct && variant_disagrees,
        });
    }

    let item_count = u64::try_from(bundle.public_packet.tasks.len())
        .map_err(|_| "metamorphic item count exceeds u64")?;
    let base_item_count = u64::try_from(base_ids.len()).map_err(|_| "base item count exceeds u64")?;
    let transformed_item_count = item_count.checked_sub(base_item_count)
        .ok_or_else(|| "metamorphic transformed item count underflowed".to_owned())?;
    let valid_response_count = correct_count.checked_add(incorrect_count)
        .ok_or_else(|| "metamorphic valid response count overflowed".to_owned())?;
    let answered_count = item_count.checked_sub(abstained_count)
        .ok_or_else(|| "metamorphic answered count underflowed".to_owned())?;
    let base_accuracy_denom = base_item_count as f64;
    let transformed_accuracy_denom = transformed_item_count as f64;
    let mut report = FreshMetamorphicScoreReportV1 {
        schema: FRESH_METAMORPHIC_SCORE_SCHEMA.to_owned(),
        metric_id: FRESH_METAMORPHIC_METRIC_ID.to_owned(),
        run_id: submission.run_id.clone(),
        model_provider: submission.model_provider.clone(),
        model_name: submission.model_name.clone(),
        model_version: submission.model_version.clone(),
        model_access_mode: submission.model_access_mode.clone(),
        model_checkpoint_sha256: submission.model_checkpoint_sha256.clone(),
        batch_id: bundle.public_packet.batch_id.clone(),
        benchmark_id: bundle.public_packet.benchmark_id.clone(),
        benchmark_version: bundle.public_packet.benchmark_version.clone(),
        task_family: bundle.public_packet.task_family.clone(),
        base_public_packet_sha256: bundle.public_packet.base_public_packet_sha256.clone(),
        base_manifest_sha256: bundle.public_packet.base_manifest_sha256.clone(),
        base_sealed_oracle_sha256: bundle.manifest.base_sealed_oracle_sha256.clone(),
        metamorphic_public_packet_sha256: bundle.public_packet.public_packet_sha256.clone(),
        metamorphic_manifest_sha256: bundle.manifest.manifest_sha256.clone(),
        metamorphic_sealed_oracle_sha256: bundle.sealed_oracle.sealed_oracle_sha256.clone(),
        submission_sha256: submission.submission_sha256.clone(),
        inference_harness_sha256: submission.inference_harness_sha256.clone(),
        prompt_and_decoding_sha256: submission.prompt_and_decoding_sha256.clone(),
        metric_definition_sha256: submission.metric_definition_sha256.clone(),
        run_execution_receipt_sha256: submission.run_execution_receipt_sha256.clone(),
        item_count,
        base_item_count,
        transformed_item_count,
        correct_count,
        incorrect_count,
        invalid_format_count: invalid_count,
        abstained_count,
        valid_response_count,
        base_correct_count,
        transformed_correct_count,
        accuracy_all_forms_permille: correct_count as f64 * 1_000.0 / item_count as f64,
        accuracy_base_form_permille: base_correct_count as f64 * 1_000.0 / base_accuracy_denom,
        accuracy_transformed_forms_permille: transformed_correct_count as f64 * 1_000.0 / transformed_accuracy_denom,
        coverage_permille: answered_count as f64 * 1_000.0 / item_count as f64,
        group_count: u64::try_from(groups.len()).map_err(|_| "metamorphic group count exceeds u64")?,
        complete_valid_group_count: complete_valid_groups,
        incomplete_group_count: incomplete_groups,
        consistent_group_count: consistent_groups,
        disagreement_group_count: disagreement_groups,
        consistent_but_wrong_group_count: consistent_but_wrong_groups,
        fully_correct_and_consistent_group_count: fully_correct_and_consistent_groups,
        base_correct_variant_inconsistent_group_count: base_correct_variant_inconsistent_groups,
        metamorphic_consistency_permille: rate(consistent_groups, complete_valid_groups),
        model_call_count: submission.model_call_count,
        input_token_count: submission.input_token_count,
        output_token_count: submission.output_token_count,
        wall_clock_millis: submission.wall_clock_millis,
        tool_call_count: submission.tool_call_count,
        human_assistance_minutes: submission.human_assistance_minutes,
        compute_cost_microusd: submission.compute_cost_microusd,
        independently_verified: false,
        items: item_scores,
        groups: group_scores,
        report_sha256: String::new(),
        claim_boundary: "exact_solution_and_cross_form_agreement_under_registered_row_operations_only;run_receipt_not_authenticated;no_general_reasoning_claim".to_owned(),
    };
    report.report_sha256 = report_digest(&report)?;
    Ok(report)
}

pub fn verify_fresh_metamorphic_score_report_integrity(
    report: &FreshMetamorphicScoreReportV1,
) -> Result<(), String> {
    if report.schema != FRESH_METAMORPHIC_SCORE_SCHEMA
        || report.metric_id != FRESH_METAMORPHIC_METRIC_ID
        || report.independently_verified
    {
        return Err("metamorphic score report schema, metric or trust boundary is invalid".to_owned());
    }
    for (digest, label) in [
        (&report.model_checkpoint_sha256, "model_checkpoint_sha256"),
        (&report.base_public_packet_sha256, "base_public_packet_sha256"),
        (&report.base_manifest_sha256, "base_manifest_sha256"),
        (&report.base_sealed_oracle_sha256, "base_sealed_oracle_sha256"),
        (&report.metamorphic_public_packet_sha256, "metamorphic_public_packet_sha256"),
        (&report.metamorphic_manifest_sha256, "metamorphic_manifest_sha256"),
        (&report.metamorphic_sealed_oracle_sha256, "metamorphic_sealed_oracle_sha256"),
        (&report.submission_sha256, "submission_sha256"),
        (&report.inference_harness_sha256, "inference_harness_sha256"),
        (&report.prompt_and_decoding_sha256, "prompt_and_decoding_sha256"),
        (&report.metric_definition_sha256, "metric_definition_sha256"),
        (&report.run_execution_receipt_sha256, "run_execution_receipt_sha256"),
        (&report.report_sha256, "report_sha256"),
    ] {
        validate_digest(digest, label)?;
    }
    if report_digest(report)? != report.report_sha256 {
        return Err("metamorphic score report digest mismatch".to_owned());
    }
    let mut ids = BTreeSet::new();
    let mut correct = 0_u64;
    let mut incorrect = 0_u64;
    let mut invalid = 0_u64;
    let mut abstained = 0_u64;
    let mut base_count = 0_u64;
    let mut transformed_count = 0_u64;
    let mut base_correct = 0_u64;
    let mut transformed_correct = 0_u64;
    for item in &report.items {
        if item.item_id.trim().is_empty() || !ids.insert(item.item_id.as_str()) {
            return Err("metamorphic score report item IDs must be unique and nonempty".to_owned());
        }
        validate_digest(&item.response_sha256, "item.response_sha256")?;
        match item.form {
            FreshMetamorphicFormV1::Base => {
                base_count += 1;
                if item.status == FreshItemScoreStatusV1::Correct {
                    base_correct += 1;
                }
            }
            _ => {
                transformed_count += 1;
                if item.status == FreshItemScoreStatusV1::Correct {
                    transformed_correct += 1;
                }
            }
        }
        match item.status {
            FreshItemScoreStatusV1::Correct => correct += 1,
            FreshItemScoreStatusV1::Incorrect => incorrect += 1,
            FreshItemScoreStatusV1::InvalidFormat => invalid += 1,
            FreshItemScoreStatusV1::Abstained => abstained += 1,
        }
    }
    let partitioned = correct.checked_add(incorrect)
        .and_then(|value| value.checked_add(invalid))
        .and_then(|value| value.checked_add(abstained))
        .ok_or_else(|| "metamorphic score item count overflowed".to_owned())?;
    let valid_responses = correct.checked_add(incorrect)
        .ok_or_else(|| "metamorphic valid response count overflowed".to_owned())?;
    if partitioned != report.item_count
        || report.items.len() as u64 != report.item_count
        || base_count != report.base_item_count
        || transformed_count != report.transformed_item_count
        || correct != report.correct_count
        || incorrect != report.incorrect_count
        || invalid != report.invalid_format_count
        || abstained != report.abstained_count
        || valid_responses != report.valid_response_count
        || base_correct != report.base_correct_count
        || transformed_correct != report.transformed_correct_count
        || base_count == 0
        || transformed_count == 0
    {
        return Err("metamorphic score item statuses and summary counts disagree".to_owned());
    }

    let mut group_ids = BTreeSet::new();
    let mut items_by_group: BTreeMap<&str, Vec<&FreshMetamorphicItemScoreV1>> = BTreeMap::new();
    for item in &report.items {
        items_by_group.entry(item.group_id.as_str()).or_default().push(item);
    }
    let mut group_item_total = 0_u64;
    let mut complete = 0_u64;
    let mut incomplete = 0_u64;
    let mut consistent = 0_u64;
    let mut disagreement = 0_u64;
    let mut consistent_wrong = 0_u64;
    let mut fully_correct_consistent = 0_u64;
    let mut base_correct_variant_inconsistent = 0_u64;
    for group in &report.groups {
        if group.group_id.trim().is_empty() || !group_ids.insert(group.group_id.as_str()) {
            return Err("metamorphic score group IDs must be unique and nonempty".to_owned());
        }
        if group.item_count != FRESH_METAMORPHIC_FORMS_PER_GROUP as u64
            || group.valid_response_count > group.item_count
            || group.correct_count > group.valid_response_count
            || group.complete_valid_group != (group.valid_response_count == group.item_count)
            || group.outputs_agree.is_some() != group.complete_valid_group
            || (group.fully_correct_and_consistent
                && !(group.complete_valid_group
                    && group.outputs_agree == Some(true)
                    && group.correct_count == group.item_count))
            || (group.consistent_but_wrong
                && !(group.complete_valid_group
                    && group.outputs_agree == Some(true)
                    && group.correct_count == 0))
            || (group.outputs_agree == Some(true)
                && group.correct_count != 0
                && group.correct_count != group.item_count)
            || (group.base_correct_variant_inconsistent
                && !(group.complete_valid_group && group.outputs_agree == Some(false)))
        {
            return Err(format!("metamorphic group {} has inconsistent summary fields", group.group_id));
        }
        let group_items = items_by_group.get(group.group_id.as_str())
            .ok_or_else(|| format!("metamorphic group {} has no item records", group.group_id))?;
        if group_items.len() as u64 != group.item_count {
            return Err(format!("metamorphic group {} item membership count mismatch", group.group_id));
        }
        let group_correct = group_items.iter()
            .filter(|item| item.status == FreshItemScoreStatusV1::Correct).count() as u64;
        let group_valid = group_items.iter()
            .filter(|item| matches!(
                item.status,
                FreshItemScoreStatusV1::Correct | FreshItemScoreStatusV1::Incorrect
            )).count() as u64;
        let base_items: Vec<_> = group_items.iter()
            .filter(|item| item.form == FreshMetamorphicFormV1::Base).collect();
        if group_correct != group.correct_count
            || group_valid != group.valid_response_count
            || base_items.len() != 1
        {
            return Err(format!("metamorphic group {} summaries do not match item records", group.group_id));
        }
        if group.base_correct_variant_inconsistent
            && base_items[0].status != FreshItemScoreStatusV1::Correct
        {
            return Err(format!("metamorphic group {} claims base-correct inconsistency without a correct base form", group.group_id));
        }
        group_item_total = group_item_total.checked_add(group.item_count)
            .ok_or_else(|| "metamorphic group item total overflowed".to_owned())?;
        if group.complete_valid_group {
            complete += 1;
            if group.outputs_agree == Some(true) {
                consistent += 1;
            } else {
                disagreement += 1;
            }
        } else {
            incomplete += 1;
        }
        consistent_wrong += if group.consistent_but_wrong { 1 } else { 0 };
        fully_correct_consistent += if group.fully_correct_and_consistent { 1 } else { 0 };
        base_correct_variant_inconsistent +=
            if group.base_correct_variant_inconsistent { 1 } else { 0 };
    }
    if group_item_total != report.item_count
        || group_ids.len() as u64 != report.group_count
        || complete != report.complete_valid_group_count
        || incomplete != report.incomplete_group_count
        || consistent != report.consistent_group_count
        || disagreement != report.disagreement_group_count
        || consistent_wrong != report.consistent_but_wrong_group_count
        || fully_correct_consistent != report.fully_correct_and_consistent_group_count
        || base_correct_variant_inconsistent != report.base_correct_variant_inconsistent_group_count
    {
        return Err("metamorphic score group totals disagree with report summaries".to_owned());
    }
    let rate_matches = |actual: f64, expected: f64| {
        actual.is_finite() && (actual - expected).abs() < 1.0e-9
    };
    let expected_all = correct as f64 * 1_000.0 / report.item_count as f64;
    let expected_base = base_correct as f64 * 1_000.0 / base_count as f64;
    let expected_transformed = transformed_correct as f64 * 1_000.0 / transformed_count as f64;
    let answered = report.item_count.checked_sub(abstained)
        .ok_or_else(|| "metamorphic abstention count underflowed".to_owned())?;
    let expected_coverage = answered as f64 * 1_000.0 / report.item_count as f64;
    if !rate_matches(report.accuracy_all_forms_permille, expected_all)
        || !rate_matches(report.accuracy_base_form_permille, expected_base)
        || !rate_matches(report.accuracy_transformed_forms_permille, expected_transformed)
        || !rate_matches(report.coverage_permille, expected_coverage)
    {
        return Err("metamorphic score accuracy/coverage rates disagree with item counts".to_owned());
    }
    match (report.metamorphic_consistency_permille, complete) {
        (None, 0) => {}
        (Some(value), denominator) if denominator > 0
            && rate_matches(value, consistent as f64 * 1_000.0 / denominator as f64) => {}
        _ => return Err("metamorphic consistency rate does not match complete-group denominator".to_owned()),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fresh_instance::{
        generate_fresh_challenge_bundle, FreshChallengeConfigV1, FRESH_CONFIG_SCHEMA,
    };
    use serde_json::Value;

    fn digest(label: &str) -> String {
        hash_domain(b"fresh-metamorphic-test", label.as_bytes())
    }

    fn base_config() -> FreshChallengeConfigV1 {
        let mut config = FreshChallengeConfigV1 {
            schema: FRESH_CONFIG_SCHEMA.to_owned(),
            batch_id: "metamorphic-fixture".to_owned(),
            benchmark_id: "fresh-algebra".to_owned(),
            benchmark_version: "v1".to_owned(),
            task_family: "linear-system-2x2".to_owned(),
            requested_item_count: 3,
            matrix_coefficient_min: -9,
            matrix_coefficient_max: 9,
            solution_component_min: -5,
            solution_component_max: 5,
            minimum_absolute_determinant: 1,
            minimum_normalized_determinant_margin_permille: 50,
            model_checkpoint_sha256: digest("checkpoint"),
            checkpoint_freeze_receipt_sha256: digest("freeze"),
            checkpoint_frozen_at_utc: "2026-10-31T18:00:00Z".to_owned(),
            challenge_generated_at_utc: "2026-10-31T18:01:00Z".to_owned(),
            generator_source_sha256: digest("source"),
            generator_binary_sha256: digest("binary"),
            runtime_image_sha256: digest("runtime"),
            generator_config_sha256: String::new(),
        };
        config.generator_config_sha256 =
            crate::fresh_instance::canonical_config_digest(&config).unwrap();
        config
    }

    fn base_bundle() -> FreshChallengeBundleV1 {
        generate_fresh_challenge_bundle(&base_config(), &[0xD5; 32]).unwrap()
    }

    fn make_submission(
        bundle: &FreshMetamorphicBundleV1,
        responses: Vec<FreshMetamorphicResponseV1>,
    ) -> FreshMetamorphicSubmissionV1 {
        let mut submission = FreshMetamorphicSubmissionV1 {
            schema: FRESH_METAMORPHIC_SUBMISSION_SCHEMA.to_owned(),
            run_id: "metamorphic-run-001".to_owned(),
            model_provider: "fixture".to_owned(),
            model_name: "fixture-model".to_owned(),
            model_version: "v1".to_owned(),
            model_access_mode: "offline-fixture".to_owned(),
            model_checkpoint_sha256: bundle.public_packet.model_checkpoint_sha256.clone(),
            base_public_packet_sha256: bundle.public_packet.base_public_packet_sha256.clone(),
            base_manifest_sha256: bundle.public_packet.base_manifest_sha256.clone(),
            metamorphic_public_packet_sha256: bundle.public_packet.public_packet_sha256.clone(),
            metamorphic_manifest_sha256: bundle.manifest.manifest_sha256.clone(),
            inference_harness_sha256: digest("harness"),
            prompt_and_decoding_sha256: digest("prompt"),
            metric_definition_sha256: hash_domain(
                b"fresh-metric-definition.v1",
                FRESH_METAMORPHIC_METRIC_ID.as_bytes(),
            ),
            run_execution_receipt_sha256: digest("receipt"),
            model_call_count: bundle.public_packet.item_count,
            input_token_count: 1000,
            output_token_count: 100,
            wall_clock_millis: 10000,
            tool_call_count: 0,
            human_assistance_minutes: 0,
            compute_cost_microusd: 0,
            responses,
            submission_sha256: String::new(),
        };
        submission.submission_sha256 = submission_digest(&submission).unwrap();
        submission
    }

    fn exact_responses(bundle: &FreshMetamorphicBundleV1) -> Vec<FreshMetamorphicResponseV1> {
        bundle.sealed_oracle.entries.iter().map(|entry| {
            let response_text = serde_json::to_string(&entry.solution).unwrap();
            FreshMetamorphicResponseV1 {
                item_id: entry.item_id.clone(),
                response_sha256: response_digest(&response_text),
                response_text,
            }
        }).collect()
    }

    fn score_all(
        base: &FreshChallengeBundleV1,
        bundle: &FreshMetamorphicBundleV1,
        responses: Vec<FreshMetamorphicResponseV1>,
    ) -> FreshMetamorphicScoreReportV1 {
        let submission = make_submission(bundle, responses);
        score_fresh_metamorphic_submission(base, bundle, &serde_json::to_vec(&submission).unwrap()).unwrap()
    }

    #[test]
    fn metamorphic_score_report_verifier_checks_group_and_accuracy_partitions() {
        let base = base_bundle();
        let bundle = generate_fresh_metamorphic_bundle(&base).unwrap();
        let report = score_all(&base, &bundle, exact_responses(&bundle));
        assert!(verify_fresh_metamorphic_score_report_integrity(&report).is_ok());

        let mut tampered = report.clone();
        tampered.base_correct_count -= 1;
        tampered.report_sha256.clear();
        tampered.report_sha256 = report_digest(&tampered).unwrap();
        assert!(verify_fresh_metamorphic_score_report_integrity(&tampered)
            .unwrap_err().contains("item statuses and summary counts"));

        let mut tampered = report;
        tampered.consistent_group_count -= 1;
        tampered.report_sha256.clear();
        tampered.report_sha256 = report_digest(&tampered).unwrap();
        assert!(verify_fresh_metamorphic_score_report_integrity(&tampered)
            .unwrap_err().contains("group totals disagree"));
    }

    #[test]
    fn generated_five_form_groups_preserve_exact_solution_and_margin() {
        let base = base_bundle();
        let bundle = generate_fresh_metamorphic_bundle(&base).unwrap();
        let audit = audit_fresh_metamorphic_bundle(&base, &bundle).unwrap();
        assert!(audit.valid);
        assert_eq!(audit.group_count, 3);
        assert_eq!(audit.item_count, 15);
        assert_eq!(audit.forms_per_group, 5);
        assert!(audit.all_variants_preserve_exact_solution);
        assert!(audit.all_variants_preserve_stability_margin);
        assert!(!audit.independently_verified);
    }

    #[test]
    fn row_swap_and_scaling_are_reversible_or_equivalent() {
        let base = base_bundle();
        for entry in &base.sealed_oracle_packet.entries {
            let swapped_twice = apply_form(
                &apply_form(&entry.parameters, FreshMetamorphicFormV1::SwapRows).unwrap(),
                FreshMetamorphicFormV1::SwapRows,
            ).unwrap();
            assert_eq!(swapped_twice.a11, entry.parameters.a11);
            assert_eq!(swapped_twice.a12, entry.parameters.a12);
            assert_eq!(swapped_twice.a21, entry.parameters.a21);
            assert_eq!(swapped_twice.a22, entry.parameters.a22);
            assert_eq!(swapped_twice.b1, entry.parameters.b1);
            assert_eq!(swapped_twice.b2, entry.parameters.b2);
            for form in FRESH_METAMORPHIC_FORMS {
                let transformed = apply_form(&entry.parameters, form).unwrap();
                let (b1, b2) = compute_rhs(&transformed, &entry.solution).unwrap();
                assert_eq!(b1, transformed.b1);
                assert_eq!(b2, transformed.b2);
            }
        }
    }

    #[test]
    fn zero_scale_and_overflow_are_rejected() {
        assert!(scale_i32(7, 0, "test").is_err());
        assert!(scale_i32(i32::MAX, 2, "test").is_err());
        assert!(scale_i64(i64::MAX, 2, "test").is_err());
    }

    #[test]
    fn deterministic_regeneration_and_tamper_detection() {
        let base = base_bundle();
        let first = generate_fresh_metamorphic_bundle(&base).unwrap();
        let second = generate_fresh_metamorphic_bundle(&base).unwrap();
        assert_eq!(
            serde_json::to_vec(&first).unwrap(),
            serde_json::to_vec(&second).unwrap()
        );
        let mut tampered = first;
        tampered.public_packet.tasks[1].task_text.push_str(" altered");
        assert!(audit_fresh_metamorphic_bundle(&base, &tampered).unwrap_err()
            .contains("differs from deterministic regeneration"));
    }

    #[test]
    fn perfect_answers_are_correct_and_consistent() {
        let base = base_bundle();
        let bundle = generate_fresh_metamorphic_bundle(&base).unwrap();
        let report = score_all(&base, &bundle, exact_responses(&bundle));
        assert_eq!(report.correct_count, 15);
        assert_eq!(report.accuracy_all_forms_permille, 1000.0);
        assert_eq!(report.complete_valid_group_count, 3);
        assert_eq!(report.consistent_group_count, 3);
        assert_eq!(report.fully_correct_and_consistent_group_count, 3);
        assert_eq!(report.consistent_but_wrong_group_count, 0);
    }

    #[test]
    fn identical_wrong_answers_are_consistent_but_never_correct() {
        let base = base_bundle();
        let bundle = generate_fresh_metamorphic_bundle(&base).unwrap();
        let responses = bundle.public_packet.tasks.iter().map(|task| {
            let response_text = r#"{"x":0,"y":0}"#.to_owned();
            FreshMetamorphicResponseV1 {
                item_id: task.item_id.clone(),
                response_sha256: response_digest(&response_text),
                response_text,
            }
        }).collect();
        let report = score_all(&base, &bundle, responses);
        assert_eq!(report.correct_count, 0);
        assert_eq!(report.consistent_group_count, 3);
        assert_eq!(report.consistent_but_wrong_group_count, 3);
        assert_eq!(report.fully_correct_and_consistent_group_count, 0);
    }

    #[test]
    fn correct_base_answer_with_variant_disagreement_is_distinct() {
        let base = base_bundle();
        let bundle = generate_fresh_metamorphic_bundle(&base).unwrap();
        let responses = bundle.sealed_oracle.entries.iter().map(|entry| {
            let solution = if entry.form == FreshMetamorphicFormV1::Base {
                entry.solution.clone()
            } else {
                ExactLinearSolutionV1 {
                    x: entry.solution.x.saturating_add(1),
                    y: entry.solution.y,
                }
            };
            let response_text = serde_json::to_string(&solution).unwrap();
            FreshMetamorphicResponseV1 {
                item_id: entry.item_id.clone(),
                response_sha256: response_digest(&response_text),
                response_text,
            }
        }).collect();
        let report = score_all(&base, &bundle, responses);
        assert_eq!(report.base_correct_count, 3);
        assert_eq!(report.transformed_correct_count, 0);
        assert_eq!(report.disagreement_group_count, 3);
        assert_eq!(report.base_correct_variant_inconsistent_group_count, 3);
    }

    #[test]
    fn invalid_and_abstained_variants_make_group_incomplete_not_consistent() {
        let base = base_bundle();
        let bundle = generate_fresh_metamorphic_bundle(&base).unwrap();
        let mut responses = exact_responses(&bundle);
        let one = &bundle.public_packet.tasks[1].item_id;
        let index = responses.iter().position(|response| &response.item_id == one).unwrap();
        responses[index].response_text.clear();
        responses[index].response_sha256 = response_digest("");
        let report = score_all(&base, &bundle, responses);
        assert_eq!(report.incomplete_group_count, 1);
        assert_eq!(report.complete_valid_group_count, 2);
        assert_eq!(report.consistent_group_count, 2);
        assert_eq!(report.abstained_count, 1);
    }

    #[test]
    fn missing_duplicate_and_unknown_variant_responses_fail_closed() {
        let base = base_bundle();
        let bundle = generate_fresh_metamorphic_bundle(&base).unwrap();
        let mut responses = exact_responses(&bundle);
        responses.pop();
        let submission = make_submission(&bundle, responses);
        assert!(score_fresh_metamorphic_submission(&base, &bundle, &serde_json::to_vec(&submission).unwrap())
            .unwrap_err().contains("exactly one response"));

        let mut responses = exact_responses(&bundle);
        responses[1].item_id = responses[0].item_id.clone();
        let submission = make_submission(&bundle, responses);
        assert!(score_fresh_metamorphic_submission(&base, &bundle, &serde_json::to_vec(&submission).unwrap()).is_err());

        let mut responses = exact_responses(&bundle);
        responses[0].item_id = "unknown-form".to_owned();
        let submission = make_submission(&bundle, responses);
        assert!(score_fresh_metamorphic_submission(&base, &bundle, &serde_json::to_vec(&submission).unwrap()).is_err());
    }

    #[test]
    fn stale_response_or_submission_digest_fails_closed() {
        let base = base_bundle();
        let bundle = generate_fresh_metamorphic_bundle(&base).unwrap();
        let mut submission = make_submission(&bundle, exact_responses(&bundle));
        submission.responses[0].response_text.push(' ');
        assert!(score_fresh_metamorphic_submission(&base, &bundle, &serde_json::to_vec(&submission).unwrap())
            .unwrap_err().contains("response digest"));
        let mut submission = make_submission(&bundle, exact_responses(&bundle));
        submission.input_token_count += 1;
        assert!(score_fresh_metamorphic_submission(&base, &bundle, &serde_json::to_vec(&submission).unwrap())
            .unwrap_err().contains("submission digest"));
    }
}
