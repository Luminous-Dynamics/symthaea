//! Source-bound resident-intent and candidate-search primitives.
//!
//! This crate validates a five-axis design problem and provides deterministic
//! finite-domain search utilities. It does not generate building geometry,
//! simulate physical performance, establish legal compliance or certify work.
//! Approval and physical-result verification remain blocked until their
//! Mycelix/Symtropy trust-boundary adapters are integrated.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

pub const DESIGN_PROBLEM_SCHEMA_VERSION: &str = "0.1.0";
pub const CANDIDATE_SET_SCHEMA_VERSION: &str = "0.1.0";
pub const INTENT_ONLY_CLAIM_CEILING: &str =
    "INTENT_ONLY_NO_GEOMETRY_COST_PERFORMANCE_CODE_OR_PROFESSIONAL_APPROVAL";

const SYNTHETIC_PROBLEM: &str = include_str!(
    "../../../../docs/research/built-environment/HOME_EXPRESSION_DESIGN_PROBLEM_SYNTHETIC_V0.1.json"
);

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum AxisState {
    Undecided,
    IntentSelected,
    IntentOnlyNotEvaluated,
    ConflictUnresolved,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ExperienceState {
    Undecided,
    OptionalIdeasOnly,
    ConflictUnresolved,
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
pub struct OptionPreferenceEntry {
    pub option_id: String,
    pub preference: Preference,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct IntentAxis {
    pub state: AxisState,
    #[serde(default)]
    pub entries: Vec<OptionPreferenceEntry>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ExpressionEntry {
    pub option_id: String,
    pub role: ExpressionRole,
    pub resident_selected: bool,
    #[serde(default)]
    pub blend_weight_pct: Option<u8>,
    #[serde(default)]
    pub source_refs: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ExpressionAxis {
    pub state: AxisState,
    #[serde(default)]
    pub entries: Vec<ExpressionEntry>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ExperienceAxis {
    pub state: ExperienceState,
    #[serde(default)]
    pub entries: Vec<OptionPreferenceEntry>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct FiveAxisIntent {
    pub spatial_typology: IntentAxis,
    pub architectural_expression: ExpressionAxis,
    pub material_tectonic_intent: IntentAxis,
    pub environmental_strategy_intent: IntentAxis,
    pub experience_programme: ExperienceAxis,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ContextRefs {
    pub site: Option<String>,
    pub climate: Option<String>,
    pub jurisdiction: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ResidentIntent {
    pub grammar_id: String,
    pub design_revision_ref: String,
    pub source_fixture_ref: String,
    pub resident_statement: String,
    pub context_state: String,
    pub context_refs: ContextRefs,
    pub axis_intent: FiveAxisIntent,
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
pub enum DesignAxis {
    #[serde(rename = "SPATIAL_TYPOLOGY")]
    SpatialTypology,
    #[serde(rename = "ARCHITECTURAL_EXPRESSION")]
    ArchitecturalExpression,
    #[serde(rename = "MATERIAL_AND_TECTONIC_INTENT")]
    MaterialTectonicIntent,
    #[serde(rename = "ENVIRONMENTAL_STRATEGY_INTENT")]
    EnvironmentalStrategyIntent,
    #[serde(rename = "EXPERIENCE_AND_PROGRAMME")]
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
pub enum CompilationState {
    IntentParsedNotCompiledToGeometry,
    CompiledToSymbolicDesignQuestion,
    ReadyForCandidateGeneration,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Compilation {
    pub compiler_version: String,
    pub state: CompilationState,
    pub hard_requirements: Vec<Requirement>,
    pub authority_constraints: Vec<Requirement>,
    pub mapping_state: MappingState,
    pub mapping_rule_ids: Vec<String>,
    pub mapping_rules: Vec<MappingRule>,
    pub unresolved_conflicts: Vec<String>,
    pub preserved_undecided_axes: Vec<String>,
    pub no_inference_rule: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum SymbolicGenerationPermission {
    RequiresReviewedMappingRules,
    AllowedWithReviewedMappings,
    Blocked,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum GeometryGenerationPermission {
    BlockedPendingSymtropyGeometryProviderAndOwnerReview,
    AllowedWithQualifiedGeometryProvider,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum PhysicalEvaluationPermission {
    BlockedPendingExactRealizedGeometryAndQualifiedSolverInputs,
    AllowedWithQualifiedSolverInputs,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum CertificationPermission {
    Prohibited,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct GenerationPermissions {
    pub symbolic_candidate_generation: SymbolicGenerationPermission,
    pub geometry_generation: GeometryGenerationPermission,
    pub physical_evaluation: PhysicalEvaluationPermission,
    pub certification: CertificationPermission,
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
    pub compilation: Compilation,
    pub generation_permissions: GenerationPermissions,
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
    /// Parse shared option IDs from the exact caller-pinned catalogue bytes.
    /// The list itself is never copied into this crate or another UI module.
    pub fn from_catalogue_json(source: &str) -> Result<Self, String> {
        let value: Value = serde_json::from_str(source)
            .map_err(|error| format!("catalogue JSON is invalid: {error}"))?;
        let mut vocabulary = Self {
            catalogue_revision: required_string(&value, "catalog_revision")?,
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
                return Err(format!("duplicate style ID {id}"));
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

    fn contains_source_option(&self, axis: &DesignAxis, option_id: &str) -> bool {
        match axis {
            DesignAxis::SpatialTypology => self.spatial_typology_ids.contains(option_id),
            DesignAxis::ArchitecturalExpression => self.style_sources.contains_key(option_id),
            DesignAxis::MaterialTectonicIntent => self.material_tectonic_ids.contains(option_id),
            DesignAxis::EnvironmentalStrategyIntent => {
                self.environmental_strategy_ids.contains(option_id)
            }
            DesignAxis::ExperienceProgramme => self.experience_package_ids.contains(option_id),
        }
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

fn is_lowercase_sha1(value: &str) -> bool {
    value.len() == 40
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

fn validate_intent_axis(
    name: &str,
    axis: &IntentAxis,
    allowed_ids: &BTreeSet<String>,
    errors: &mut Vec<String>,
) {
    if axis.entries.is_empty() && axis.state != AxisState::Undecided {
        errors.push(format!("{name}: empty axis must remain UNDECIDED"));
    }
    if !axis.entries.is_empty() && axis.state == AxisState::Undecided {
        errors.push(format!("{name}: non-empty axis cannot be UNDECIDED"));
    }
    let mut seen = BTreeSet::new();
    let mut preferred_count = 0;
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
    }
    if name == "spatial_typology"
        && preferred_count > 1
        && axis.state != AxisState::ConflictUnresolved
    {
        errors.push("spatial_typology: multiple preferred options must remain CONFLICT_UNRESOLVED".to_owned());
    }
    if name == "spatial_typology"
        && axis.state == AxisState::ConflictUnresolved
        && preferred_count < 2
    {
        errors.push("spatial_typology: conflict state requires multiple preferred options".to_owned());
    }
}

fn validate_expression_axis(
    axis: &ExpressionAxis,
    vocabulary: &CatalogueVocabulary,
    errors: &mut Vec<String>,
) {
    if axis.entries.is_empty() && axis.state != AxisState::Undecided {
        errors.push("architectural_expression: empty axis must remain UNDECIDED".to_owned());
    }
    if !axis.entries.is_empty() && axis.state == AxisState::Undecided {
        errors.push("architectural_expression: non-empty axis cannot be UNDECIDED".to_owned());
    }
    let mut seen = BTreeSet::new();
    let mut primary_count = 0;
    for entry in &axis.entries {
        if !vocabulary.style_sources.contains_key(&entry.option_id) {
            errors.push(format!("architectural_expression: unknown style ID {}", entry.option_id));
        }
        if !seen.insert(entry.option_id.clone()) {
            errors.push(format!("architectural_expression: duplicate style ID {}", entry.option_id));
        }
        if !entry.resident_selected {
            errors.push(format!("architectural_expression: {} lacks resident selection", entry.option_id));
        }
        if entry.role == ExpressionRole::Primary {
            primary_count += 1;
        }
        if entry.role == ExpressionRole::ReferenceOnly {
            match vocabulary.style_sources.get(&entry.option_id) {
                None => errors.push(format!("reference-only style {} is unknown", entry.option_id)),
                Some(_allowed) if entry.source_refs.is_empty() => {
                    errors.push(format!("reference-only style {} has no source", entry.option_id));
                }
                Some(allowed) => {
                    for source in &entry.source_refs {
                        if !allowed.contains(source) {
                            errors.push(format!("unbound source {source} for style {}", entry.option_id));
                        }
                    }
                }
            }
        }
        if entry.blend_weight_pct.is_some_and(|weight| weight > 100) {
            errors.push(format!("architectural_expression: blend weight for {} exceeds 100", entry.option_id));
        }
    }
    if primary_count > 1 {
        errors.push("architectural_expression: multiple PRIMARY roles are unresolved".to_owned());
    }
}

fn validate_experience_axis(
    axis: &ExperienceAxis,
    allowed_ids: &BTreeSet<String>,
    errors: &mut Vec<String>,
) {
    if axis.entries.is_empty() && axis.state != ExperienceState::Undecided {
        errors.push("experience_programme: empty axis must remain UNDECIDED".to_owned());
    }
    if !axis.entries.is_empty() && axis.state == ExperienceState::Undecided {
        errors.push("experience_programme: non-empty axis cannot be UNDECIDED".to_owned());
    }
    let mut seen = BTreeSet::new();
    for entry in &axis.entries {
        if !allowed_ids.contains(&entry.option_id) {
            errors.push(format!("experience_programme: unknown package ID {}", entry.option_id));
        }
        if !seen.insert(entry.option_id.clone()) {
            errors.push(format!("experience_programme: duplicate package ID {}", entry.option_id));
        }
    }
}

fn validate_requirements(
    requirements: &[Requirement],
    seen: &mut BTreeSet<String>,
    errors: &mut Vec<String>,
) {
    for requirement in requirements {
        if requirement.requirement_id.trim().is_empty()
            || requirement.statement.trim().is_empty()
            || !seen.insert(requirement.requirement_id.clone())
        {
            errors.push(format!("empty or duplicate requirement ID/statement: {}", requirement.requirement_id));
        }
        if matches!(
            requirement.source_class,
            RequirementSourceClass::JurisdictionalRule
                | RequirementSourceClass::ProfessionalReviewRequirement
        ) && requirement.source_ref.as_deref().is_none_or(str::is_empty)
        {
            errors.push(format!("authority requirement {} lacks a source reference", requirement.requirement_id));
        }
        if requirement.verification_state == VerificationState::OwnerValidated
            && requirement.source_ref.as_deref().is_none_or(str::is_empty)
        {
            errors.push(format!("owner-validated requirement {} lacks an evidence source", requirement.requirement_id));
        }
    }
}

fn validate_compilation(
    compilation: &Compilation,
    input: &ResidentIntent,
    vocabulary: &CatalogueVocabulary,
    errors: &mut Vec<String>,
) {
    let mut requirement_ids = BTreeSet::new();
    validate_requirements(&compilation.hard_requirements, &mut requirement_ids, errors);
    validate_requirements(&compilation.authority_constraints, &mut requirement_ids, errors);

    let mut mapping_ids = BTreeSet::new();
    let mut detailed_ids = Vec::new();
    let mut approved_sources = BTreeSet::new();
    for rule in &compilation.mapping_rules {
        if rule.mapping_rule_id.trim().is_empty()
            || !mapping_ids.insert(rule.mapping_rule_id.clone())
        {
            errors.push(format!("missing or duplicate mapping ID {}", rule.mapping_rule_id));
        }
        detailed_ids.push(rule.mapping_rule_id.clone());
        if !vocabulary.contains_source_option(&rule.source_axis, &rule.source_option_id) {
            errors.push(format!("mapping {} refers to unknown source option {}", rule.mapping_rule_id, rule.source_option_id));
        }
        if rule.target_parameter.trim().is_empty()
            || rule.mapping_version.trim().is_empty()
            || rule.scope.trim().is_empty()
        {
            errors.push(format!("mapping {} lacks target, version or scope", rule.mapping_rule_id));
        }
        if rule.review_state == MappingReviewState::Approved {
            if rule.source_ref.as_deref().is_none_or(str::is_empty)
                || rule.owner_reviewer.as_deref().is_none_or(str::is_empty)
            {
                errors.push(format!("approved mapping {} lacks source/reviewer", rule.mapping_rule_id));
            } else {
                approved_sources.insert((rule.source_axis.clone(), rule.source_option_id.clone()));
            }
        }
    }
    detailed_ids.sort();
    let mut declared_ids = compilation.mapping_rule_ids.clone();
    declared_ids.sort();
    if detailed_ids != declared_ids {
        errors.push("mapping_rule_ids differs from detailed mapping_rules".to_owned());
    }

    match compilation.mapping_state {
        MappingState::NoOwnerReviewedMapping
            if compilation.mapping_rules.iter().any(|rule| rule.review_state == MappingReviewState::Approved) =>
        {
            errors.push("NO_OWNER_REVIEWED_MAPPING cannot contain approved mappings".to_owned());
        }
        MappingState::Partial
            if compilation.mapping_rules.is_empty()
                || compilation.mapping_rules.iter().all(|rule| rule.review_state == MappingReviewState::Approved) =>
        {
            errors.push("PARTIAL requires both covered and uncovered mapping scope".to_owned());
        }
        MappingState::Qualified => {
            if compilation.mapping_rules.is_empty()
                || compilation.mapping_rules.iter().any(|rule| rule.review_state != MappingReviewState::Approved)
            {
                errors.push("QUALIFIED requires only approved mapping rules".to_owned());
            }
            let axes = [
                (DesignAxis::SpatialTypology, input.axis_intent.spatial_typology.entries.iter().map(|e| e.option_id.as_str()).collect::<Vec<_>>()),
                (DesignAxis::ArchitecturalExpression, input.axis_intent.architectural_expression.entries.iter().map(|e| e.option_id.as_str()).collect::<Vec<_>>()),
                (DesignAxis::MaterialTectonicIntent, input.axis_intent.material_tectonic_intent.entries.iter().map(|e| e.option_id.as_str()).collect::<Vec<_>>()),
                (DesignAxis::EnvironmentalStrategyIntent, input.axis_intent.environmental_strategy_intent.entries.iter().map(|e| e.option_id.as_str()).collect::<Vec<_>>()),
                (DesignAxis::ExperienceProgramme, input.axis_intent.experience_programme.entries.iter().map(|e| e.option_id.as_str()).collect::<Vec<_>>()),
            ];
            for (axis, ids) in axes {
                for id in ids {
                    if !approved_sources.contains(&(axis.clone(), id.to_owned())) {
                        errors.push(format!("QUALIFIED lacks approved mapping for {axis:?}:{id}"));
                    }
                }
            }
            if !compilation.unresolved_conflicts.is_empty() {
                errors.push("QUALIFIED cannot erase unresolved conflicts".to_owned());
            }
        }
        _ => {}
    }
}

pub fn validate_design_problem(
    problem: &DesignProblem,
    vocabulary: &CatalogueVocabulary,
) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();
    if problem.schema_version != DESIGN_PROBLEM_SCHEMA_VERSION {
        errors.push("unsupported schema_version".to_owned());
    }
    if problem.problem_id.trim().is_empty() {
        errors.push("problem_id is required".to_owned());
    }
    if problem.source_bindings.catalogue_revision != vocabulary.catalogue_revision {
        errors.push("catalogue revision does not match parsed vocabulary".to_owned());
    }
    if !problem.source_bindings.sol_atlas_pr.starts_with("https://github.com/Luminous-Dynamics/sol-atlas-leptos/pull/") {
        errors.push("Sol-Atlas source must be an explicit PR URL".to_owned());
    }
    for (name, digest) in [
        ("sol_atlas_pr_head_sha", problem.source_bindings.sol_atlas_pr_head_sha.as_str()),
        ("catalogue_blob_sha", problem.source_bindings.catalogue_blob_sha.as_str()),
        ("grammar_schema_blob_sha", problem.source_bindings.grammar_schema_blob_sha.as_str()),
        ("conformance_corpus_blob_sha", problem.source_bindings.conformance_corpus_blob_sha.as_str()),
        ("conformance_corpus_schema_blob_sha", problem.source_bindings.conformance_corpus_schema_blob_sha.as_str()),
    ] {
        if !is_lowercase_sha1(digest) {
            errors.push(format!("{name} is not a lowercase 40-character Git object SHA"));
        }
    }

    let input = &problem.input;
    if input.grammar_id.trim().is_empty()
        || input.design_revision_ref.trim().is_empty()
        || input.source_fixture_ref.trim().is_empty()
    {
        errors.push("grammar_id/design_revision_ref/source_fixture_ref are required".to_owned());
    }
    if input.resident_statement.chars().count() > 3000 {
        errors.push("resident_statement exceeds upstream maximum length".to_owned());
    }
    if !["UNKNOWN", "PARTIAL", "SOURCE_BOUND_NOT_QUALIFIED"].contains(&input.context_state.as_str()) {
        errors.push("context state is unsupported or over-claimed".to_owned());
    }

    let axes = &input.axis_intent;
    validate_intent_axis("spatial_typology", &axes.spatial_typology, &vocabulary.spatial_typology_ids, &mut errors);
    validate_expression_axis(&axes.architectural_expression, vocabulary, &mut errors);
    validate_intent_axis("material_tectonic_intent", &axes.material_tectonic_intent, &vocabulary.material_tectonic_ids, &mut errors);
    validate_intent_axis("environmental_strategy_intent", &axes.environmental_strategy_intent, &vocabulary.environmental_strategy_ids, &mut errors);
    validate_experience_axis(&axes.experience_programme, &vocabulary.experience_package_ids, &mut errors);
    validate_compilation(&problem.compilation, input, vocabulary, &mut errors);

    if problem.compilation.state == CompilationState::ReadyForCandidateGeneration
        && problem.compilation.mapping_state != MappingState::Qualified
    {
        errors.push("READY_FOR_CANDIDATE_GENERATION requires QUALIFIED mapping state".to_owned());
    }
    if problem.generation_permissions.certification != CertificationPermission::Prohibited {
        errors.push("certification must remain PROHIBITED".to_owned());
    }
    if problem.generation_permissions.geometry_generation
        == GeometryGenerationPermission::AllowedWithQualifiedGeometryProvider
        && problem.compilation.mapping_state != MappingState::Qualified
    {
        errors.push("geometry permission requires qualified mapping".to_owned());
    }
    if problem.generation_permissions.physical_evaluation
        == PhysicalEvaluationPermission::AllowedWithQualifiedSolverInputs
        && problem.generation_permissions.geometry_generation
            != GeometryGenerationPermission::AllowedWithQualifiedGeometryProvider
    {
        errors.push("physical evaluation permission requires geometry permission".to_owned());
    }
    if problem.claim_ceiling != INTENT_ONLY_CLAIM_CEILING {
        errors.push("claim_ceiling exceeds the intent-only contract".to_owned());
    }

    if errors.is_empty() { Ok(()) } else { Err(errors) }
}

/// The v0.1 crate intentionally does not activate candidate generation.
/// A future Mycelix adapter must cryptographically verify authority receipts;
/// a text field naming an owner/reviewer is not proof of approval.
pub fn symbolic_candidate_generation_allowed(_problem: &DesignProblem) -> bool {
    false
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReplayReceipt {
    pub problem_digest: String,
    pub generator_build_digest: String,
    pub seed: u64,
    pub ordered_candidate_ids: Vec<String>,
}

pub fn design_problem_digest(problem: &DesignProblem) -> Result<String, serde_json::Error> {
    Ok(blake3::hash(&serde_json::to_vec(problem)?).to_hex().to_string())
}

pub fn derive_candidate_id(
    problem_digest: &str,
    generator_build_digest: &str,
    seed: u64,
    ordinal: u32,
    parameter_deltas: &BTreeMap<String, Value>,
) -> Result<String, serde_json::Error> {
    let bytes = serde_json::to_vec(&(
        problem_digest,
        generator_build_digest,
        seed,
        ordinal,
        parameter_deltas,
    ))?;
    Ok(format!("cand-{}", blake3::hash(&bytes).to_hex()))
}

pub fn replay_matches(expected: &ReplayReceipt, actual: &ReplayReceipt) -> bool {
    expected == actual
}

/// A finite categorical variable used by the exact-enumeration oracle.
/// Values are sorted before expansion so results do not depend on source order.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct DiscreteVariable {
    pub variable_id: String,
    pub domain: Vec<String>,
}

pub type Assignment = BTreeMap<String, String>;

pub fn enumerate_assignments(
    variables: &[DiscreteVariable],
    max_assignments: usize,
) -> Result<Vec<Assignment>, String> {
    if variables.is_empty() {
        return Err("search space must contain at least one variable".to_owned());
    }
    let mut variable_ids = BTreeSet::new();
    let mut domains = Vec::with_capacity(variables.len());
    let mut cardinality = 1usize;

    for variable in variables {
        if variable.variable_id.trim().is_empty() || !variable_ids.insert(variable.variable_id.clone()) {
            return Err(format!("empty or duplicate variable ID: {}", variable.variable_id));
        }
        if variable.domain.is_empty() {
            return Err(format!("{} has an empty domain", variable.variable_id));
        }
        let mut domain = variable.domain.clone();
        domain.sort();
        domain.dedup();
        if domain.len() != variable.domain.len() {
            return Err(format!("{} has duplicate domain values", variable.variable_id));
        }
        cardinality = cardinality
            .checked_mul(domain.len())
            .ok_or_else(|| "search-space cardinality overflow".to_owned())?;
        if cardinality > max_assignments {
            return Err(format!(
                "search space has {cardinality} assignments, cap is {max_assignments}"
            ));
        }
        domains.push(domain);
    }

    let mut assignments = vec![Assignment::new()];
    for (variable, domain) in variables.iter().zip(domains) {
        let mut next = Vec::with_capacity(assignments.len() * domain.len());
        for partial in &assignments {
            for value in &domain {
                let mut assignment = partial.clone();
                assignment.insert(variable.variable_id.clone(), value.clone());
                next.push(assignment);
            }
        }
        assignments = next;
    }
    Ok(assignments)
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ObjectiveDirection {
    Minimize,
    Maximize,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct OracleObjective {
    pub objective_id: String,
    pub direction: ObjectiveDirection,
    pub value: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct OracleEvaluation {
    pub candidate_id: String,
    pub gate_state: GateState,
    pub objectives: Vec<OracleObjective>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum GateState {
    Pass,
    Fail,
    Unknown,
    Conditional,
}

fn dominates(left: &OracleEvaluation, right: &OracleEvaluation) -> bool {
    if left.objectives.len() != right.objectives.len() || left.objectives.is_empty() {
        return false;
    }
    let mut strictly_better = false;
    for (l, r) in left.objectives.iter().zip(&right.objectives) {
        if l.objective_id != r.objective_id || l.direction != r.direction {
            return false;
        }
        let (no_worse, better) = match l.direction {
            ObjectiveDirection::Minimize => (l.value <= r.value, l.value < r.value),
            ObjectiveDirection::Maximize => (l.value >= r.value, l.value > r.value),
        };
        if !no_worse { return false; }
        strictly_better |= better;
    }
    strictly_better
}

/// Compute an exact front over a finite, already evaluated set.
/// Only PASS candidates with finite and structurally identical objective vectors
/// are considered. FAIL/UNKNOWN/CONDITIONAL entries are never admitted.
pub fn exact_pareto_front(evaluations: &[OracleEvaluation]) -> Result<Vec<String>, String> {
    let mut ids = BTreeSet::new();
    for evaluation in evaluations {
        if evaluation.candidate_id.trim().is_empty() || !ids.insert(evaluation.candidate_id.clone()) {
            return Err("candidate IDs must be non-empty and unique".to_owned());
        }
    }
    let eligible: Vec<&OracleEvaluation> = evaluations
        .iter()
        .filter(|evaluation| {
            evaluation.gate_state == GateState::Pass
                && !evaluation.objectives.is_empty()
                && evaluation.objectives.iter().all(|objective| {
                    !objective.objective_id.trim().is_empty() && objective.value.is_finite()
                })
        })
        .collect();
    if let Some(first) = eligible.first() {
        for candidate in &eligible[1..] {
            if candidate.objectives.len() != first.objectives.len()
                || candidate.objectives.iter().zip(&first.objectives).any(|(a, b)| {
                    a.objective_id != b.objective_id || a.direction != b.direction
                })
            {
                return Err("eligible candidates have mismatched objective schemas".to_owned());
            }
        }
    }
    let mut front = Vec::new();
    for candidate in &eligible {
        let dominated = eligible.iter().any(|other| {
            other.candidate_id != candidate.candidate_id && dominates(other, candidate)
        });
        if !dominated {
            front.push(candidate.candidate_id.clone());
        }
    }
    front.sort();
    Ok(front)
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ScenarioEvaluation {
    pub scenario_id: String,
    pub gate_state: GateState,
    pub objectives: Vec<OracleObjective>,
}

/// Conservative worst-case aggregation over an explicitly supplied scenario set.
/// Missing, unknown or failing scenarios abstain instead of being pooled away.
pub fn worst_case_objectives(
    scenarios: &[ScenarioEvaluation],
) -> Result<Vec<OracleObjective>, String> {
    if scenarios.is_empty() {
        return Err("at least one named scenario is required".to_owned());
    }
    let mut scenario_ids = BTreeSet::new();
    let first = &scenarios[0];
    if first.scenario_id.trim().is_empty() || !scenario_ids.insert(first.scenario_id.clone()) {
        return Err("scenario IDs must be non-empty and unique".to_owned());
    }
    if first.gate_state != GateState::Pass || first.objectives.is_empty() {
        return Err("every scenario must PASS and contain objective values".to_owned());
    }
    let mut result = first.objectives.clone();
    if result.iter().any(|objective| !objective.value.is_finite() || objective.objective_id.trim().is_empty()) {
        return Err("scenario objective values must be finite and named".to_owned());
    }

    for scenario in &scenarios[1..] {
        if scenario.scenario_id.trim().is_empty() || !scenario_ids.insert(scenario.scenario_id.clone()) {
            return Err("scenario IDs must be non-empty and unique".to_owned());
        }
        if scenario.gate_state != GateState::Pass {
            return Err(format!("scenario {} is not PASS; robust aggregate abstains", scenario.scenario_id));
        }
        if scenario.objectives.len() != result.len() {
            return Err("scenario objective count differs".to_owned());
        }
        for (aggregate, observation) in result.iter_mut().zip(&scenario.objectives) {
            if aggregate.objective_id != observation.objective_id
                || aggregate.direction != observation.direction
                || !observation.value.is_finite()
            {
                return Err(format!("scenario {} has incompatible objective schema/value", scenario.scenario_id));
            }
            match aggregate.direction {
                ObjectiveDirection::Minimize => aggregate.value = aggregate.value.max(observation.value),
                ObjectiveDirection::Maximize => aggregate.value = aggregate.value.min(observation.value),
            }
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_catalogue() -> Value {
        serde_json::json!({
            "catalog_revision":"HOME-EXPRESSION-CATALOG-0.2.0",
            "style_catalog":[
                {"style_id":"HIGHVELD_STONE_TIMBER_CONTEMPORARY","reference_sources":[]},
                {"style_id":"CAPE_GEORGIAN_INFLUENCE","reference_sources":["https://repository.up.ac.za/server/api/core/bitstreams/fc087af1-69b5-4f2f-9e74-9ed9fe58716e/content"]}
            ],
            "spatial_typology_catalog":[{"option_id":"COURTYARD"},{"option_id":"ROW_TERRACE"}],
            "material_tectonic_intent_catalog":[{"option_id":"STONE_MASONRY"},{"option_id":"TIMBER"}],
            "environmental_strategy_intent_catalog":[{"option_id":"EXTERNAL_SHADING"},{"option_id":"CROSS_VENTILATION"},{"option_id":"THERMAL_MASS"}],
            "experience_catalog":[{"package_id":"PRIVATE_OBSERVATORY"}]
        })
    }

    fn parsed_problem() -> DesignProblem {
        serde_json::from_str(SYNTHETIC_PROBLEM).expect("synthetic problem must deserialize")
    }

    fn vocabulary() -> CatalogueVocabulary {
        CatalogueVocabulary::from_catalogue_json(&synthetic_catalogue().to_string()).unwrap()
    }

    #[test]
    fn upstream_synthetic_problem_deserializes_and_validates() {
        let problem = parsed_problem();
        assert!(validate_design_problem(&problem, &vocabulary()).is_ok());
        assert_eq!(problem.input.axis_intent.spatial_typology.state, AxisState::Undecided);
        assert_eq!(problem.claim_ceiling, INTENT_ONLY_CLAIM_CEILING);
    }

    #[test]
    fn unknown_or_duplicate_option_ids_are_rejected() {
        let mut problem = parsed_problem();
        problem.input.axis_intent.material_tectonic_intent.entries[0].option_id =
            "UNLISTED_COMPOSITE".to_owned();
        assert!(validate_design_problem(&problem, &vocabulary()).is_err());

        let mut duplicate = parsed_problem();
        duplicate.input.axis_intent.material_tectonic_intent.entries.push(
            duplicate.input.axis_intent.material_tectonic_intent.entries[0].clone()
        );
        assert!(validate_design_problem(&duplicate, &vocabulary()).is_err());
    }

    #[test]
    fn reference_only_style_requires_catalogue_bound_source() {
        let mut problem = parsed_problem();
        problem.input.axis_intent.architectural_expression.entries[1].source_refs =
            vec!["https://example.invalid/not-in-catalogue".to_owned()];
        assert!(validate_design_problem(&problem, &vocabulary()).is_err());
    }

    #[test]
    fn unresolved_typology_conflict_is_preserved_not_resolved() {
        let mut problem = parsed_problem();
        problem.input.axis_intent.spatial_typology = IntentAxis {
            state: AxisState::ConflictUnresolved,
            entries: vec![
                OptionPreferenceEntry { option_id:"COURTYARD".to_owned(), preference:Preference::Preferred },
                OptionPreferenceEntry { option_id:"ROW_TERRACE".to_owned(), preference:Preference::Preferred },
            ],
        };
        assert!(validate_design_problem(&problem, &vocabulary()).is_ok());
        problem.input.axis_intent.spatial_typology.state = AxisState::IntentOnlyNotEvaluated;
        assert!(validate_design_problem(&problem, &vocabulary()).is_err());
    }

    #[test]
    fn mapping_name_alone_never_authorizes_generation() {
        let problem = parsed_problem();
        assert!(!symbolic_candidate_generation_allowed(&problem));
        assert_eq!(problem.compilation.mapping_state, MappingState::NoOwnerReviewedMapping);
    }

    #[test]
    fn exact_source_bindings_require_lowercase_git_sha_shapes() {
        let mut problem = parsed_problem();
        problem.source_bindings.catalogue_blob_sha = "NOT_A_GIT_SHA".to_owned();
        assert!(validate_design_problem(&problem, &vocabulary()).is_err());
    }

    #[test]
    fn finite_domain_enumeration_is_exact_sorted_and_bounded() {
        let variables = vec![
            DiscreteVariable { variable_id:"topology".to_owned(), domain:vec!["courtyard".to_owned(),"row".to_owned()] },
            DiscreteVariable { variable_id:"shade".to_owned(), domain:vec!["external".to_owned(),"internal".to_owned()] },
        ];
        let assignments = enumerate_assignments(&variables, 4).unwrap();
        assert_eq!(assignments.len(), 4);
        assert_eq!(assignments[0]["shade"], "external");
        assert_eq!(assignments[3]["topology"], "row");
        assert!(enumerate_assignments(&variables, 3).is_err());
    }

    #[test]
    fn exact_oracle_excludes_invalid_and_returns_known_pareto_front() {
        let make = |id: &str, gate_state: GateState, first: f64, second: f64| OracleEvaluation {
            candidate_id:id.to_owned(),
            gate_state,
            objectives:vec![
                OracleObjective {objective_id:"energy".to_owned(),direction:ObjectiveDirection::Minimize,value:first},
                OracleObjective {objective_id:"water".to_owned(),direction:ObjectiveDirection::Minimize,value:second},
            ],
        };
        let evaluations = vec![
            make("a",GateState::Pass,5.0,5.0),
            make("b",GateState::Pass,4.0,7.0),
            make("c",GateState::Pass,7.0,4.0),
            make("dominated",GateState::Pass,6.0,6.0),
            make("invalid",GateState::Fail,1.0,1.0),
            make("unknown",GateState::Unknown,0.5,0.5),
        ];
        assert_eq!(exact_pareto_front(&evaluations).unwrap(), vec!["a".to_owned(),"b".to_owned(),"c".to_owned()]);
    }

    #[test]
    fn robust_worst_case_preserves_scenario_adversity_and_abstains_on_unknown() {
        let scenario = |id: &str, gate: GateState, energy: f64, comfort: f64| ScenarioEvaluation {
            scenario_id:id.to_owned(),
            gate_state:gate,
            objectives:vec![
                OracleObjective {objective_id:"energy".to_owned(),direction:ObjectiveDirection::Minimize,value:energy},
                OracleObjective {objective_id:"comfort_hours".to_owned(),direction:ObjectiveDirection::Maximize,value:comfort},
            ],
        };
        let result = worst_case_objectives(&[
            scenario("dry-hot",GateState::Pass,10.0,120.0),
            scenario("cool-wet",GateState::Pass,12.0,150.0),
        ]).unwrap();
        assert_eq!(result[0].value, 12.0);
        assert_eq!(result[1].value, 120.0);
        assert!(worst_case_objectives(&[
            scenario("known",GateState::Pass,10.0,120.0),
            scenario("unknown",GateState::Unknown,9.0,160.0),
        ]).is_err());
    }

    #[test]
    fn deterministic_problem_digest_and_candidate_ids_replay() {
        let problem = parsed_problem();
        let first = design_problem_digest(&problem).unwrap();
        assert_eq!(first, design_problem_digest(&problem).unwrap());
        let deltas = BTreeMap::from([("intent.shade".to_owned(),Value::String("external".to_owned()))]);
        let id = derive_candidate_id(&first,"generator-build",7,0,&deltas).unwrap();
        assert_eq!(id,derive_candidate_id(&first,"generator-build",7,0,&deltas).unwrap());
    }
}
