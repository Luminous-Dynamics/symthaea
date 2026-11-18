//! Resident-intent compilation and candidate lineage contracts.
//!
//! This crate does not generate geometry or evaluate building physics. It
//! preserves source-qualified intent, gates mappings, derives deterministic
//! identities and admits candidates to a physical Pareto set only after
//! owner-qualified geometry/evaluation receipts exist.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::{BTreeMap, BTreeSet};

pub const DESIGN_PROBLEM_SCHEMA_VERSION: &str = "0.1.0";
pub const CANDIDATE_SET_SCHEMA_VERSION: &str = "0.1.0";

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum AxisState {
    Undecided,
    IntentSelected,
    IntentOnlyNotEvaluated,
    ConflictUnresolved,
    OptionalIdeasOnly,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Preference {
    Preferred,
    Considered,
    Avoided,
    Undecided,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ExpressionRole {
    Primary,
    Secondary,
    Accent,
    ReferenceOnly,
    Avoid,
    Acceptable,
    Undecided,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct IntentEntry {
    pub option_id: String,
    pub preference: Preference,
    #[serde(default)]
    pub expression_role: Option<ExpressionRole>,
    #[serde(default)]
    pub resident_selected: Option<bool>,
    #[serde(default)]
    pub blend_weight_pct: Option<u8>,
    #[serde(default)]
    pub source_refs: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct AxisIntent {
    pub state: AxisState,
    #[serde(default)]
    pub entries: Vec<IntentEntry>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ResidentIntent {
    pub grammar_id: String,
    pub design_revision_ref: String,
    pub resident_statement: String,
    pub context_state: String,
    pub spatial_typology: AxisIntent,
    pub architectural_expression: AxisIntent,
    pub material_tectonic_intent: AxisIntent,
    pub environmental_strategy_intent: AxisIntent,
    pub experience_programme: AxisIntent,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct SourceBindings {
    pub sol_atlas_pr: String,
    pub sol_atlas_pr_head_sha: String,
    pub catalogue_revision: String,
    pub catalogue_blob_sha: String,
    pub grammar_schema_blob_sha: String,
    pub conformance_corpus_blob_sha: String,
    pub conformance_corpus_schema_blob_sha: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum RequirementSourceClass {
    ResidentNonNegotiable,
    JurisdictionalRule,
    ProfessionalReviewRequirement,
    SystemInvariant,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum VerificationState {
    ExplicitUnverified,
    SourceBoundUnreviewed,
    OwnerValidated,
    Unknown,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Requirement {
    pub requirement_id: String,
    pub statement: String,
    pub source_class: RequirementSourceClass,
    pub source_ref: Option<String>,
    pub verification_state: VerificationState,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum DesignAxis {
    SpatialTypology,
    ArchitecturalExpression,
    MaterialTectonicIntent,
    EnvironmentalStrategyIntent,
    ExperienceProgramme,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum MappingReviewState {
    Draft,
    OwnerReview,
    Approved,
    Rejected,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct MappingRule {
    pub mapping_rule_id: String,
    pub source_axis: DesignAxis,
    pub source_option_id: String,
    pub target_parameter: String,
    pub mapping_version: String,
    pub scope: String,
    pub source_ref: Option<String>,
    pub owner_reviewer: Option<String>,
    pub review_state: MappingReviewState,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum MappingState {
    NoOwnerReviewedMapping,
    Partial,
    Qualified,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum DecisionState {
    NoResidentChoiceOrDesignApprovalRecorded,
    PartialResidentIntentRecorded,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct DesignProblem {
    pub schema_version: String,
    pub problem_id: String,
    pub maturity: String,
    pub source_bindings: SourceBindings,
    pub input: ResidentIntent,
    pub hard_requirements: Vec<Requirement>,
    pub authority_constraints: Vec<Requirement>,
    pub mapping_state: MappingState,
    pub mapping_rules: Vec<MappingRule>,
    pub unresolved_conflicts: Vec<String>,
    pub decision_state: DecisionState,
    pub claim_ceiling: String,
}

#[derive(Clone, Debug, Default)]
pub struct CatalogueVocabulary {
    pub catalogue_revision: String,
    pub style_sources: BTreeMap<String, BTreeSet<String>>,
    pub spatial_typology_ids: BTreeSet<String>,
    pub material_tectonic_ids: BTreeSet<String>,
    pub environmental_strategy_ids: BTreeSet<String>,
    pub experience_package_ids: BTreeSet<String>,
}

impl CatalogueVocabulary {
    /// Load identifiers from the versioned Sol-Atlas catalogue. No option lists
    /// are duplicated in this crate; the caller supplies the exact catalogue
    /// bytes whose Git blob SHA is bound in SourceBindings.
    pub fn from_catalogue_json(source: &str) -> Result<Self, String> {
        let value: Value = serde_json::from_str(source)
            .map_err(|error| format!("catalogue JSON is invalid: {error}"))?;
        let revision = value
            .get("catalog_revision")
            .and_then(Value::as_str)
            .ok_or_else(|| "catalogue_revision is missing".to_owned())?
            .to_owned();

        let mut vocabulary = Self {
            catalogue_revision: revision,
            ..Self::default()
        };

        let styles = value
            .get("style_catalog")
            .and_then(Value::as_array)
            .ok_or_else(|| "style_catalog is missing or not an array".to_owned())?;
        for style in styles {
            let id = required_string(style, "style_id")?;
            let sources = style
                .get("reference_sources")
                .and_then(Value::as_array)
                .ok_or_else(|| format!("style {id} has no reference_sources array"))?
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_owned)
                .collect();
            if vocabulary.style_sources.insert(id.clone(), sources).is_some() {
                return Err(format!("duplicate style identifier: {id}"));
            }
        }

        vocabulary.spatial_typology_ids =
            collect_ids(&value, "spatial_typology_catalog", "option_id")?;
        vocabulary.material_tectonic_ids =
            collect_ids(&value, "material_tectonic_intent_catalog", "option_id")?;
        vocabulary.environmental_strategy_ids =
            collect_ids(&value, "environmental_strategy_intent_catalog", "option_id")?;
        vocabulary.experience_package_ids =
            collect_ids(&value, "experience_catalog", "package_id")?;
        Ok(vocabulary)
    }

    pub fn style_ids(&self) -> BTreeSet<String> {
        self.style_sources.keys().cloned().collect()
    }
}

fn required_string(value: &Value, key: &str) -> Result<String, String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .filter(|text| !text.trim().is_empty())
        .map(str::to_owned)
        .ok_or_else(|| format!("required non-empty string {key} is missing"))
}

fn collect_ids(value: &Value, catalogue_key: &str, id_key: &str) -> Result<BTreeSet<String>, String> {
    let items = value
        .get(catalogue_key)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{catalogue_key} is missing or not an array"))?;
    let mut ids = BTreeSet::new();
    for item in items {
        let id = required_string(item, id_key)?;
        if !ids.insert(id.clone()) {
            return Err(format!("duplicate identifier {id} in {catalogue_key}"));
        }
    }
    Ok(ids)
}

fn is_sha1(value: &str) -> bool {
    value.len() == 40 && value.bytes().all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

fn validate_axis(
    name: &str,
    axis: &AxisIntent,
    allowed_ids: &BTreeSet<String>,
    expression_axis: bool,
    style_sources: Option<&BTreeMap<String, BTreeSet<String>>>,
    errors: &mut Vec<String>,
) {
    if axis.entries.is_empty() && axis.state != AxisState::Undecided {
        errors.push(format!("{name}: empty axis must remain UNDECIDED"));
    }
    if !axis.entries.is_empty() && axis.state == AxisState::Undecided {
        errors.push(format!("{name}: non-empty axis cannot be UNDECIDED"));
    }

    let mut seen = BTreeSet::new();
    let mut primary_count = 0usize;
    let mut preferred_count = 0usize;
    for entry in &axis.entries {
        if entry.option_id.trim().is_empty() || !allowed_ids.contains(&entry.option_id) {
            errors.push(format!("{name}: unknown option ID {}", entry.option_id));
        }
        if !seen.insert(entry.option_id.clone()) {
            errors.push(format!("{name}: duplicate option ID {}", entry.option_id));
        }
        if entry.preference == Preference::Preferred {
            preferred_count += 1;
        }

        if expression_axis {
            match entry.expression_role.as_ref() {
                Some(ExpressionRole::Primary) => {
                    primary_count += 1;
                    if entry.resident_selected != Some(true) {
                        errors.push(format!("{name}: PRIMARY {} is not explicitly resident-selected", entry.option_id));
                    }
                }
                Some(ExpressionRole::ReferenceOnly) => {
                    if entry.resident_selected != Some(true) {
                        errors.push(format!("{name}: REFERENCE_ONLY {} is not explicitly selected", entry.option_id));
                    }
                    let allowed = style_sources
                        .and_then(|styles| styles.get(&entry.option_id));
                    if entry.source_refs.is_empty() {
                        errors.push(format!("{name}: REFERENCE_ONLY {} has no source", entry.option_id));
                    } else if let Some(allowed_sources) = allowed {
                        for source in &entry.source_refs {
                            if !allowed_sources.contains(source) {
                                errors.push(format!("{name}: source {source} is not catalogue-bound for {}", entry.option_id));
                            }
                        }
                    }
                }
                Some(_) => {
                    if entry.resident_selected != Some(true) {
                        errors.push(format!("{name}: selected role {} has no resident selection flag", entry.option_id));
                    }
                }
                None => errors.push(format!("{name}: expression entry {} has no role", entry.option_id)),
            }
            if entry.blend_weight_pct.is_some_and(|weight| weight > 100) {
                errors.push(format!("{name}: blend weight for {} exceeds 100", entry.option_id));
            }
        } else if entry.expression_role.is_some() || entry.blend_weight_pct.is_some() {
            errors.push(format!("{name}: non-expression axis {} contains expression-only fields", entry.option_id));
        }
    }

    if expression_axis && primary_count > 1 {
        errors.push("architectural_expression: more than one PRIMARY role".to_owned());
    }
    if name == "spatial_typology" && preferred_count > 1 && axis.state != AxisState::ConflictUnresolved {
        errors.push("spatial_typology: multiple preferred entries must remain CONFLICT_UNRESOLVED".to_owned());
    }
    if name == "spatial_typology" && axis.state == AxisState::ConflictUnresolved && preferred_count < 2 {
        errors.push("spatial_typology: conflict state requires multiple preferred entries".to_owned());
    }
}

pub fn validate_design_problem(
    problem: &DesignProblem,
    vocabulary: &CatalogueVocabulary,
) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();
    if problem.schema_version != DESIGN_PROBLEM_SCHEMA_VERSION {
        errors.push("unsupported design-problem schema version".to_owned());
    }
    if problem.problem_id.trim().is_empty() {
        errors.push("problem_id is required".to_owned());
    }
    if problem.source_bindings.catalogue_revision != vocabulary.catalogue_revision {
        errors.push("catalogue revision does not match loaded vocabulary".to_owned());
    }
    for (name, digest) in [
        ("sol_atlas_pr_head_sha", problem.source_bindings.sol_atlas_pr_head_sha.as_str()),
        ("catalogue_blob_sha", problem.source_bindings.catalogue_blob_sha.as_str()),
        ("grammar_schema_blob_sha", problem.source_bindings.grammar_schema_blob_sha.as_str()),
        ("conformance_corpus_blob_sha", problem.source_bindings.conformance_corpus_blob_sha.as_str()),
        ("conformance_corpus_schema_blob_sha", problem.source_bindings.conformance_corpus_schema_blob_sha.as_str()),
    ] {
        if !is_sha1(digest) {
            errors.push(format!("{name} is not a lowercase 40-character Git object SHA"));
        }
    }

    let input = &problem.input;
    if input.grammar_id.trim().is_empty() || input.design_revision_ref.trim().is_empty() {
        errors.push("grammar_id and design_revision_ref are required".to_owned());
    }
    if !["UNKNOWN", "PARTIAL", "SOURCE_BOUND_NOT_QUALIFIED"].contains(&input.context_state.as_str()) {
        errors.push("context_state is unsupported or over-claimed".to_owned());
    }

    validate_axis(
        "spatial_typology",
        &input.spatial_typology,
        &vocabulary.spatial_typology_ids,
        false,
        None,
        &mut errors,
    );
    validate_axis(
        "architectural_expression",
        &input.architectural_expression,
        &vocabulary.style_ids(),
        true,
        Some(&vocabulary.style_sources),
        &mut errors,
    );
    validate_axis(
        "material_tectonic_intent",
        &input.material_tectonic_intent,
        &vocabulary.material_tectonic_ids,
        false,
        None,
        &mut errors,
    );
    validate_axis(
        "environmental_strategy_intent",
        &input.environmental_strategy_intent,
        &vocabulary.environmental_strategy_ids,
        false,
        None,
        &mut errors,
    );
    validate_axis(
        "experience_programme",
        &input.experience_programme,
        &vocabulary.experience_package_ids,
        false,
        None,
        &mut errors,
    );

    for requirement in problem.hard_requirements.iter().chain(problem.authority_constraints.iter()) {
        if requirement.requirement_id.trim().is_empty() || requirement.statement.trim().is_empty() {
            errors.push("requirements need a non-empty ID and statement".to_owned());
        }
        if matches!(
            requirement.source_class,
            RequirementSourceClass::JurisdictionalRule
                | RequirementSourceClass::ProfessionalReviewRequirement
        ) && requirement.source_ref.as_deref().is_none_or(str::is_empty)
        {
            errors.push(format!(
                "authority requirement {} has no source reference",
                requirement.requirement_id
            ));
        }
    }

    let mut mapping_ids = BTreeSet::new();
    for rule in &problem.mapping_rules {
        if rule.mapping_rule_id.trim().is_empty() || !mapping_ids.insert(rule.mapping_rule_id.clone()) {
            errors.push(format!("missing or duplicate mapping rule ID: {}", rule.mapping_rule_id));
        }
        if rule.target_parameter.trim().is_empty() || rule.mapping_version.trim().is_empty() || rule.scope.trim().is_empty() {
            errors.push(format!("mapping rule {} lacks target, version or scope", rule.mapping_rule_id));
        }
        if rule.review_state == MappingReviewState::Approved {
            if rule.source_ref.as_deref().is_none_or(str::is_empty) || rule.owner_reviewer.as_deref().is_none_or(str::is_empty) {
                errors.push(format!("approved mapping rule {} lacks source/reviewer", rule.mapping_rule_id));
            }
        }
    }
    match problem.mapping_state {
        MappingState::NoOwnerReviewedMapping if !problem.mapping_rules.is_empty() => {
            errors.push("NO_OWNER_REVIEWED_MAPPING cannot carry mapping rules".to_owned());
        }
        MappingState::Partial if problem.mapping_rules.is_empty() => {
            errors.push("PARTIAL mapping state requires at least one mapping rule".to_owned());
        }
        MappingState::Qualified
            if problem.mapping_rules.is_empty()
                || problem.mapping_rules.iter().any(|rule| rule.review_state != MappingReviewState::Approved) =>
        {
            errors.push("QUALIFIED mapping state requires only owner-approved mapping rules".to_owned());
        }
        _ => {}
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

pub fn symbolic_candidate_generation_allowed(problem: &DesignProblem) -> bool {
    problem.mapping_state == MappingState::Qualified
        && !problem.mapping_rules.is_empty()
        && problem.mapping_rules.iter().all(|rule| {
            rule.review_state == MappingReviewState::Approved
                && rule.source_ref.as_deref().is_some_and(|source| !source.is_empty())
                && rule.owner_reviewer.as_deref().is_some_and(|reviewer| !reviewer.is_empty())
        })
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct GeometryProviderQualification {
    pub provider_id: String,
    pub provider_revision: String,
    pub qualified: bool,
    pub exact_geometry_digest: Option<String>,
}

pub fn geometry_handoff_eligible(
    problem: &DesignProblem,
    provider: &GeometryProviderQualification,
) -> bool {
    symbolic_candidate_generation_allowed(problem)
        && provider.qualified
        && !provider.provider_id.is_empty()
        && !provider.provider_revision.is_empty()
        && provider
            .exact_geometry_digest
            .as_deref()
            .is_some_and(is_sha1)
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ObjectiveValue {
    pub objective_id: String,
    pub value: f64,
    pub unit: String,
    pub source_receipt_digest: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct PhysicalEvaluationReceipt {
    pub owner_system: String,
    pub solver_version: String,
    pub exact_input_digest: String,
    pub qualified: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Candidate {
    pub candidate_id: String,
    pub parent_candidate_ids: Vec<String>,
    pub source_intent_refs: Vec<String>,
    pub parameter_deltas: BTreeMap<String, Value>,
    pub unresolved_requirements: Vec<String>,
    pub hard_gate_state: GateState,
    pub geometry_ref: Option<String>,
    pub physical_evaluations: Vec<PhysicalEvaluationReceipt>,
    pub objectives: Vec<ObjectiveValue>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum GateState {
    Pass,
    Fail,
    Unknown,
    Conditional,
}

pub fn candidate_eligible_for_physical_pareto(candidate: &Candidate) -> bool {
    candidate.hard_gate_state == GateState::Pass
        && candidate.geometry_ref.as_deref().is_some_and(|reference| !reference.is_empty())
        && !candidate.physical_evaluations.is_empty()
        && candidate
            .physical_evaluations
            .iter()
            .all(|receipt| receipt.qualified && is_sha1(&receipt.exact_input_digest))
        && !candidate.objectives.is_empty()
        && candidate.objectives.iter().all(|objective| {
            objective.value.is_finite()
                && !objective.objective_id.is_empty()
                && !objective.unit.is_empty()
                && is_sha1(&objective.source_receipt_digest)
        })
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReplayReceipt {
    pub problem_digest: String,
    pub generator_build_digest: String,
    pub seed: u64,
    pub ordered_candidate_ids: Vec<String>,
}

pub fn design_problem_digest(problem: &DesignProblem) -> Result<String, serde_json::Error> {
    let bytes = serde_json::to_vec(problem)?;
    Ok(blake3::hash(&bytes).to_hex().to_string())
}

/// Derive an ID from all inputs that are allowed to change the generated
/// candidate. The caller supplies a canonical parameter-delta map.
pub fn derive_candidate_id(
    problem_digest: &str,
    generator_build_digest: &str,
    seed: u64,
    ordinal: u32,
    parameter_deltas: &BTreeMap<String, Value>,
) -> Result<String, serde_json::Error> {
    let payload = serde_json::to_vec(&(
        problem_digest,
        generator_build_digest,
        seed,
        ordinal,
        parameter_deltas,
    ))?;
    Ok(format!("cand-{}", blake3::hash(&payload).to_hex()))
}

pub fn replay_matches(expected: &ReplayReceipt, actual: &ReplayReceipt) -> bool {
    expected == actual
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_catalogue() -> Value {
        serde_json::json!({
            "catalog_revision":"HOME-EXPRESSION-CATALOG-0.2.0",
            "style_catalog":[
                {"style_id":"HIGHVELD_STONE_TIMBER_CONTEMPORARY","reference_sources":[]},
                {"style_id":"CAPE_GEORGIAN_INFLUENCE","reference_sources":["https://example.org/cape-source"]}
            ],
            "spatial_typology_catalog":[{"option_id":"COURTYARD"},{"option_id":"ROW_TERRACE"}],
            "material_tectonic_intent_catalog":[{"option_id":"STONE_MASONRY"},{"option_id":"TIMBER"}],
            "environmental_strategy_intent_catalog":[{"option_id":"EXTERNAL_SHADING"},{"option_id":"CROSS_VENTILATION"}],
            "experience_catalog":[{"package_id":"PRIVATE_OBSERVATORY"}]
        })
    }

    fn valid_problem() -> DesignProblem {
        let source_bindings = SourceBindings {
            sol_atlas_pr:"https://github.com/Luminous-Dynamics/sol-atlas-leptos/pull/935".to_owned(),
            sol_atlas_pr_head_sha:"42caacb8b3e00f87fcf46c0dd8ceae87763aa1d6".to_owned(),
            catalogue_revision:"HOME-EXPRESSION-CATALOG-0.2.0".to_owned(),
            catalogue_blob_sha:"d784b285a706068aeba7bf7a88bdb857ed2053cc".to_owned(),
            grammar_schema_blob_sha:"661c2c5a96e13c7678b80714bdb9e59d787c5a7f".to_owned(),
            conformance_corpus_blob_sha:"61ff73fca82af4905bb9d5dd2a020bfaf438e94f".to_owned(),
            conformance_corpus_schema_blob_sha:"918ab3b936b45528360a705975f3f50328aea293".to_owned(),
        };
        let undecided = AxisIntent { state:AxisState::Undecided, entries:vec![] };
        DesignProblem {
            schema_version:DESIGN_PROBLEM_SCHEMA_VERSION.to_owned(),
            problem_id:"SYNTH-HOME-EXPRESSION-DESIGN-PROBLEM-001".to_owned(),
            maturity:"SYNTHETIC_NOT_EVALUATED".to_owned(),
            source_bindings,
            input:ResidentIntent {
                grammar_id:"SYNTH-HIGHVELD-GRAMMAR-001".to_owned(),
                design_revision_ref:"SYNTH-HV-GRAMMAR-REV-0".to_owned(),
                resident_statement:"Synthetic intent only".to_owned(),
                context_state:"UNKNOWN".to_owned(),
                spatial_typology:undecided,
                architectural_expression:AxisIntent {
                    state:AxisState::IntentSelected,
                    entries:vec![
                        IntentEntry {option_id:"HIGHVELD_STONE_TIMBER_CONTEMPORARY".to_owned(),preference:Preference::Preferred,expression_role:Some(ExpressionRole::Primary),resident_selected:Some(true),blend_weight_pct:Some(100),source_refs:vec![]},
                        IntentEntry {option_id:"CAPE_GEORGIAN_INFLUENCE".to_owned(),preference:Preference::Considered,expression_role:Some(ExpressionRole::ReferenceOnly),resident_selected:Some(true),blend_weight_pct:None,source_refs:vec!["https://example.org/cape-source".to_owned()]}
                    ]
                },
                material_tectonic_intent:AxisIntent {state:AxisState::IntentOnlyNotEvaluated,entries:vec![IntentEntry {option_id:"STONE_MASONRY".to_owned(),preference:Preference::Preferred,expression_role:None,resident_selected:None,blend_weight_pct:None,source_refs:vec![]}]},
                environmental_strategy_intent:AxisIntent {state:AxisState::IntentOnlyNotEvaluated,entries:vec![IntentEntry {option_id:"EXTERNAL_SHADING".to_owned(),preference:Preference::Preferred,expression_role:None,resident_selected:None,blend_weight_pct:None,source_refs:vec![]}]},
                experience_programme:AxisIntent {state:AxisState::OptionalIdeasOnly,entries:vec![IntentEntry {option_id:"PRIVATE_OBSERVATORY".to_owned(),preference:Preference::Considered,expression_role:None,resident_selected:None,blend_weight_pct:None,source_refs:vec![]}]}
            },
            hard_requirements:vec![],
            authority_constraints:vec![],
            mapping_state:MappingState::NoOwnerReviewedMapping,
            mapping_rules:vec![],
            unresolved_conflicts:vec![],
            decision_state:DecisionState::NoResidentChoiceOrDesignApprovalRecorded,
            claim_ceiling:"INTENT_ONLY_NO_GEOMETRY_COST_PERFORMANCE_CODE_OR_PROFESSIONAL_APPROVAL".to_owned()
        }
    }

    #[test]
    fn catalogue_vocabulary_is_loaded_not_duplicated() {
        let vocabulary=CatalogueVocabulary::from_catalogue_json(&synthetic_catalogue().to_string()).unwrap();
        assert!(vocabulary.style_sources.contains_key("HIGHVELD_STONE_TIMBER_CONTEMPORARY"));
        assert!(vocabulary.spatial_typology_ids.contains("COURTYARD"));
    }

    #[test]
    fn valid_synthetic_problem_passes_and_keeps_geometry_blocked() {
        let vocabulary=CatalogueVocabulary::from_catalogue_json(&synthetic_catalogue().to_string()).unwrap();
        let problem=valid_problem();
        assert!(validate_design_problem(&problem,&vocabulary).is_ok());
        assert!(!symbolic_candidate_generation_allowed(&problem));
    }

    #[test]
    fn unknown_style_or_option_id_is_rejected() {
        let vocabulary=CatalogueVocabulary::from_catalogue_json(&synthetic_catalogue().to_string()).unwrap();
        let mut problem=valid_problem();
        problem.input.material_tectonic_intent.entries[0].option_id="UNLISTED_COMPOSITE".to_owned();
        assert!(validate_design_problem(&problem,&vocabulary).is_err());
    }

    #[test]
    fn reference_only_style_requires_a_catalogue_bound_source() {
        let vocabulary=CatalogueVocabulary::from_catalogue_json(&synthetic_catalogue().to_string()).unwrap();
        let mut problem=valid_problem();
        problem.input.architectural_expression.entries[1].source_refs=vec!["https://example.org/unbound".to_owned()];
        assert!(validate_design_problem(&problem,&vocabulary).is_err());
    }

    #[test]
    fn preferred_typology_conflict_is_not_collapsed() {
        let vocabulary=CatalogueVocabulary::from_catalogue_json(&synthetic_catalogue().to_string()).unwrap();
        let mut problem=valid_problem();
        problem.input.spatial_typology=AxisIntent {
            state:AxisState::ConflictUnresolved,
            entries:vec![
                IntentEntry {option_id:"COURTYARD".to_owned(),preference:Preference::Preferred,expression_role:None,resident_selected:None,blend_weight_pct:None,source_refs:vec![]},
                IntentEntry {option_id:"ROW_TERRACE".to_owned(),preference:Preference::Preferred,expression_role:None,resident_selected:None,blend_weight_pct:None,source_refs:vec![]}
            ]
        };
        assert!(validate_design_problem(&problem,&vocabulary).is_ok());
        problem.input.spatial_typology.state=AxisState::IntentOnlyNotEvaluated;
        assert!(validate_design_problem(&problem,&vocabulary).is_err());
    }

    #[test]
    fn no_mapping_or_partial_mapping_cannot_start_candidate_generation() {
        let mut problem=valid_problem();
        assert!(!symbolic_candidate_generation_allowed(&problem));
        problem.mapping_state=MappingState::Partial;
        problem.mapping_rules.push(MappingRule {
            mapping_rule_id:"MAP-001".to_owned(),
            source_axis:DesignAxis::MaterialTectonicIntent,
            source_option_id:"STONE_MASONRY".to_owned(),
            target_parameter:"assembly.material_family".to_owned(),
            mapping_version:"0.1.0".to_owned(),
            scope:"PAVILION_ONLY".to_owned(),
            source_ref:Some("https://example.org/material-spec".to_owned()),
            owner_reviewer:Some("qualified-owner".to_owned()),
            review_state:MappingReviewState::OwnerReview
        });
        assert!(!symbolic_candidate_generation_allowed(&problem));
    }

    #[test]
    fn only_owner_approved_mappings_open_symbolic_generation() {
        let mut problem=valid_problem();
        problem.mapping_state=MappingState::Qualified;
        problem.mapping_rules.push(MappingRule {
            mapping_rule_id:"MAP-001".to_owned(),
            source_axis:DesignAxis::MaterialTectonicIntent,
            source_option_id:"STONE_MASONRY".to_owned(),
            target_parameter:"assembly.material_family".to_owned(),
            mapping_version:"0.1.0".to_owned(),
            scope:"PAVILION_ONLY".to_owned(),
            source_ref:Some("https://example.org/material-spec".to_owned()),
            owner_reviewer:Some("qualified-owner".to_owned()),
            review_state:MappingReviewState::Approved
        });
        assert!(symbolic_candidate_generation_allowed(&problem));
    }

    #[test]
    fn unknown_or_failed_hard_gates_never_enter_physical_pareto() {
        let mut candidate=Candidate {
            candidate_id:"cand-1".to_owned(),parent_candidate_ids:vec![],source_intent_refs:vec![],
            parameter_deltas:BTreeMap::new(),unresolved_requirements:vec![],
            hard_gate_state:GateState::Unknown,geometry_ref:Some("geometry://exact".to_owned()),
            physical_evaluations:vec![PhysicalEvaluationReceipt {owner_system:"symtropy".to_owned(),solver_version:"v1".to_owned(),exact_input_digest:"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_owned(),qualified:true}],
            objectives:vec![ObjectiveValue {objective_id:"energy".to_owned(),value:10.0,unit:"kWh".to_owned(),source_receipt_digest:"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".to_owned()}]
        };
        assert!(!candidate_eligible_for_physical_pareto(&candidate));
        candidate.hard_gate_state=GateState::Pass;
        assert!(candidate_eligible_for_physical_pareto(&candidate));
    }

    #[test]
    fn candidate_identity_and_problem_digest_are_deterministic() {
        let problem=valid_problem();
        let first=design_problem_digest(&problem).unwrap();
        let second=design_problem_digest(&problem).unwrap();
        assert_eq!(first,second);
        let deltas=BTreeMap::from([("layout.module".to_owned(),Value::String("A".to_owned()))]);
        assert_eq!(
            derive_candidate_id(&first,"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",42,0,&deltas).unwrap(),
            derive_candidate_id(&second,"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",42,0,&deltas).unwrap()
        );
    }

    #[test]
    fn replay_receipt_detects_order_or_digest_drift() {
        let receipt=ReplayReceipt {problem_digest:"a".repeat(64),generator_build_digest:"b".repeat(64),seed:7,ordered_candidate_ids:vec!["cand-1".to_owned(),"cand-2".to_owned()]};
        let mut changed=receipt.clone();
        changed.ordered_candidate_ids.reverse();
        assert!(replay_matches(&receipt,&receipt));
        assert!(!replay_matches(&receipt,&changed));
    }

    #[test]
    fn geometry_handoff_needs_qualified_provider_and_mapping() {
        let problem=valid_problem();
        let provider=GeometryProviderQualification {provider_id:"symtropy-geometry".to_owned(),provider_revision:"0.1".to_owned(),qualified:true,exact_geometry_digest:Some("c".repeat(40))};
        assert!(!geometry_handoff_eligible(&problem,&provider));
    }
}
