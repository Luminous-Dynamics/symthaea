//! Source-bound resident-intent and candidate-search primitives.
//!
//! This crate validates a five-axis design problem and provides deterministic
//! finite-domain search utilities. It does not generate building geometry,
//! simulate physical performance, establish legal compliance or certify work.
//! Approval and physical-result verification remain blocked until their
//! Mycelix/Symtropy trust-boundary adapters are integrated.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha1::{Digest, Sha1};
use std::collections::{BTreeMap, BTreeSet};

pub const DESIGN_PROBLEM_SCHEMA_VERSION: &str = "0.1.0";
pub const CANDIDATE_SET_SCHEMA_VERSION: &str = "0.1.0";
pub const INTENT_ONLY_CLAIM_CEILING: &str =
    "INTENT_ONLY_NO_GEOMETRY_COST_PERFORMANCE_CODE_OR_PROFESSIONAL_APPROVAL";

const SYNTHETIC_PROBLEM: &str = include_str!(
    "../../../../docs/research/built-environment/HOME_EXPRESSION_DESIGN_PROBLEM_SYNTHETIC_V0.1.json"
);
const SYNTHETIC_CANDIDATE_SET: &str = include_str!(
    "../../../../docs/research/built-environment/HOME_EXPRESSION_CANDIDATE_SET_SYNTHETIC_V0.1.json"
);

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Copy)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum AxisState {
    Undecided,
    IntentSelected,
    IntentOnlyNotEvaluated,
    ConflictUnresolved,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Copy)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ExperienceState {
    Undecided,
    OptionalIdeasOnly,
    ConflictUnresolved,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Copy)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Preference {
    Preferred,
    Considered,
    Avoided,
    Undecided,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Copy)]
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
    pub catalogue_schema_blob_sha: String,
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
    pub catalogue_blob_sha: String,
    pub catalogue_schema_blob_sha: String,
    pub style_sources: BTreeMap<String, BTreeSet<String>>,
    pub spatial_typology_ids: BTreeSet<String>,
    pub material_tectonic_ids: BTreeSet<String>,
    pub environmental_strategy_ids: BTreeSet<String>,
    pub experience_package_ids: BTreeSet<String>,
}

impl CatalogueVocabulary {
    /// Parse shared option IDs from the exact caller-pinned catalogue bytes.
    /// The list itself is never copied into this crate or another UI module.
    fn from_catalogue_json(source: &str) -> Result<Self, String> {
        let value: Value = serde_json::from_str(source)
            .map_err(|error| format!("catalogue JSON is invalid: {error}"))?;
        let mut vocabulary = Self {
            catalogue_revision: required_string(&value, "catalog_revision")?,
            catalogue_blob_sha: git_blob_sha1(source.as_bytes()),
            ..Self::default()
        };
        let styles = value
            .get("style_catalog")
            .and_then(Value::as_array)
            .ok_or_else(|| "style_catalog is missing or not an array".to_owned())?;
        for style in styles {
            let id = required_string(style, "style_id")?;
            let source_values = style
                .get("reference_sources")
                .and_then(Value::as_array)
                .ok_or_else(|| format!("style {id} has no reference_sources array"))?;
            let mut sources = BTreeSet::new();
            for source in source_values {
                let source = source
                    .as_str()
                    .ok_or_else(|| format!("style {id} contains a non-string source reference"))?;
                if !source.starts_with("https://") {
                    return Err(format!("style {id} contains a non-HTTPS source reference"));
                }
                if !sources.insert(source.to_owned()) {
                    return Err(format!("style {id} contains a duplicate source reference"));
                }
            }
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

    /// Load and bind both the catalogue and its interpretation schema from the
    /// exact caller-supplied bytes. A problem validated with this vocabulary
    /// must carry matching Git blob SHAs for both objects.
    pub fn from_catalogue_sources(
        catalogue_source: &str,
        catalogue_schema_source: &str,
    ) -> Result<Self, String> {
        let schema: Value = serde_json::from_str(catalogue_schema_source)
            .map_err(|error| format!("catalogue schema JSON is invalid: {error}"))?;
        if schema.get("$schema").and_then(Value::as_str)
            != Some("https://json-schema.org/draft/2020-12/schema")
        {
            return Err("catalogue schema must use JSON Schema Draft 2020-12".to_owned());
        }
        let mut vocabulary = Self::from_catalogue_json(catalogue_source)?;
        vocabulary.catalogue_schema_blob_sha =
            git_blob_sha1(catalogue_schema_source.as_bytes());
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

fn git_blob_sha1(bytes: &[u8]) -> String {
    let header = format!("blob {}\0", bytes.len());
    let mut hasher = Sha1::new();
    hasher.update(header.as_bytes());
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
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
    if problem.source_bindings.catalogue_blob_sha != vocabulary.catalogue_blob_sha {
        errors.push("catalogue blob SHA does not match the exact loaded catalogue bytes".to_owned());
    }
    if problem.source_bindings.catalogue_schema_blob_sha
        != vocabulary.catalogue_schema_blob_sha
    {
        errors.push("catalogue schema blob SHA does not match the exact loaded schema bytes".to_owned());
    }
    if !problem.source_bindings.sol_atlas_pr.starts_with("https://github.com/Luminous-Dynamics/sol-atlas-leptos/pull/") {
        errors.push("Sol-Atlas source must be an explicit PR URL".to_owned());
    }
    for (name, digest) in [
        ("sol_atlas_pr_head_sha", problem.source_bindings.sol_atlas_pr_head_sha.as_str()),
        ("catalogue_blob_sha", problem.source_bindings.catalogue_blob_sha.as_str()),
        ("catalogue_schema_blob_sha", problem.source_bindings.catalogue_schema_blob_sha.as_str()),
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

    if problem.compilation.mapping_state == MappingState::Qualified {
        errors.push("mapping QUALIFIED is unsupported until Mycelix authority receipts are cryptographically verified".to_owned());
    }
    if problem.compilation.state == CompilationState::ReadyForCandidateGeneration {
        errors.push("candidate generation is disabled in v0.1 until reviewed mappings and an authority verifier exist".to_owned());
    }
    if problem.generation_permissions.symbolic_candidate_generation
        != SymbolicGenerationPermission::RequiresReviewedMappingRules
    {
        errors.push("symbolic candidate generation must remain closed in v0.1".to_owned());
    }
    if problem.generation_permissions.geometry_generation
        != GeometryGenerationPermission::BlockedPendingSymtropyGeometryProviderAndOwnerReview
    {
        errors.push("geometry generation must remain blocked until the Symtropy provider contract is qualified".to_owned());
    }
    if problem.generation_permissions.physical_evaluation
        != PhysicalEvaluationPermission::BlockedPendingExactRealizedGeometryAndQualifiedSolverInputs
    {
        errors.push("physical evaluation must remain blocked until exact geometry and qualified solver inputs exist".to_owned());
    }
    if problem.generation_permissions.certification != CertificationPermission::Prohibited {
        errors.push("certification must remain PROHIBITED".to_owned());
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

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct CandidateSetGenerator {
    pub name: String,
    pub version: Option<String>,
    pub build_digest: Option<String>,
    pub method: String,
    pub seed: Option<u64>,
    pub replay_status: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct CandidateSourceIntent {
    pub axis: String,
    pub option_id: String,
    pub preference: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ConceptCandidateReceipt {
    pub candidate_id: String,
    pub candidate_state: String,
    pub interpretation: String,
    pub source_intents: Vec<CandidateSourceIntent>,
    pub unresolved_intent_axes: Vec<String>,
    pub parameter_deltas: Vec<Value>,
    pub geometry_ref: Option<String>,
    pub hard_gate_state: String,
    pub physical_evaluation_state: String,
    pub pareto_state: String,
    pub resident_choice_state: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct CandidateSetReceipt {
    pub schema_version: String,
    pub candidate_set_id: String,
    pub maturity: String,
    pub source_problem_id: String,
    pub source_problem_revision: String,
    pub source_catalogue_revision: String,
    pub generation_status: String,
    pub generator: CandidateSetGenerator,
    pub candidate_count: usize,
    pub candidates: Vec<ConceptCandidateReceipt>,
    pub claim_ceiling: String,
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

/// Validate the illustrative candidate-set receipt without promoting it to a
/// generated set. Executable generation requires a later, reviewed authority
/// bridge and a separately qualified design/geometry provider.
fn expected_resident_source_preference(
    problem: &DesignProblem,
    axis: &str,
    option_id: &str,
) -> Option<&'static str> {
    let axes = &problem.input.axis_intent;
    match axis {
        "spatial_typology" => axes
            .spatial_typology
            .entries
            .iter()
            .find(|entry| entry.option_id == option_id)
            .map(|entry| match entry.preference {
                Preference::Preferred => "PREFERRED",
                Preference::Considered => "CONSIDERED",
                Preference::Avoided => "AVOIDED",
                Preference::Undecided => "UNDECIDED",
            }),
        "architectural_expression" => axes
            .architectural_expression
            .entries
            .iter()
            .find(|entry| entry.option_id == option_id)
            .map(|entry| match entry.role {
                ExpressionRole::Primary => "PRIMARY",
                ExpressionRole::Secondary => "SECONDARY",
                ExpressionRole::Accent => "ACCENT",
                ExpressionRole::ReferenceOnly => "REFERENCE_ONLY",
                ExpressionRole::Avoid => "AVOID",
                ExpressionRole::Acceptable => "ACCEPTABLE",
                ExpressionRole::Undecided => "UNDECIDED",
            }),
        "material_tectonic_intent" => axes
            .material_tectonic_intent
            .entries
            .iter()
            .find(|entry| entry.option_id == option_id)
            .map(|entry| match entry.preference {
                Preference::Preferred => "PREFERRED",
                Preference::Considered => "CONSIDERED",
                Preference::Avoided => "AVOIDED",
                Preference::Undecided => "UNDECIDED",
            }),
        "environmental_strategy_intent" => axes
            .environmental_strategy_intent
            .entries
            .iter()
            .find(|entry| entry.option_id == option_id)
            .map(|entry| match entry.preference {
                Preference::Preferred => "PREFERRED",
                Preference::Considered => "CONSIDERED",
                Preference::Avoided => "AVOIDED",
                Preference::Undecided => "UNDECIDED",
            }),
        "experience_programme" => axes
            .experience_programme
            .entries
            .iter()
            .find(|entry| entry.option_id == option_id)
            .map(|entry| match entry.preference {
                Preference::Preferred => "PREFERRED",
                Preference::Considered => "CONSIDERED",
                Preference::Avoided => "AVOIDED",
                Preference::Undecided => "UNDECIDED",
            }),
        _ => None,
    }
}

fn explicitly_unresolved_intent_axes(problem: &DesignProblem) -> BTreeSet<String> {
    let axes = &problem.input.axis_intent;
    let mut unresolved = BTreeSet::new();
    if matches!(axes.spatial_typology.state, AxisState::Undecided | AxisState::ConflictUnresolved) {
        unresolved.insert("SPATIAL_TYPOLOGY".to_owned());
    }
    if matches!(axes.architectural_expression.state, AxisState::Undecided | AxisState::ConflictUnresolved) {
        unresolved.insert("ARCHITECTURAL_EXPRESSION".to_owned());
    }
    if matches!(axes.material_tectonic_intent.state, AxisState::Undecided | AxisState::ConflictUnresolved) {
        unresolved.insert("MATERIAL_AND_TECTONIC_INTENT".to_owned());
    }
    if matches!(axes.environmental_strategy_intent.state, AxisState::Undecided | AxisState::ConflictUnresolved) {
        unresolved.insert("ENVIRONMENTAL_STRATEGY_INTENT".to_owned());
    }
    if matches!(axes.experience_programme.state, ExperienceState::Undecided | ExperienceState::ConflictUnresolved) {
        unresolved.insert("EXPERIENCE_AND_PROGRAMME".to_owned());
    }
    unresolved
}

pub fn validate_illustrative_candidate_set(
    receipt: &CandidateSetReceipt,
    problem: &DesignProblem,
    vocabulary: &CatalogueVocabulary,
) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();
    if receipt.schema_version != CANDIDATE_SET_SCHEMA_VERSION {
        errors.push("unsupported candidate-set schema version".to_owned());
    }
    if receipt.candidate_set_id.trim().is_empty() {
        errors.push("candidate_set_id is required".to_owned());
    }
    if receipt.source_problem_id != problem.problem_id
        || receipt.source_problem_revision != problem.input.design_revision_ref
    {
        errors.push("candidate set is not bound to this exact design problem revision".to_owned());
    }
    if receipt.source_catalogue_revision != problem.source_bindings.catalogue_revision
        || receipt.source_catalogue_revision != vocabulary.catalogue_revision
    {
        errors.push("candidate set catalogue revision differs from the source problem".to_owned());
    }
    if receipt.maturity != "ILLUSTRATIVE_NOT_EXECUTED"
        || receipt.generation_status != "EXAMPLE_ONLY_NOT_EXECUTED"
        || receipt.claim_ceiling
            != "CONCEPTUAL_INTENT_DERIVATIVES_ONLY_NO_GEOMETRY_COST_PERFORMANCE_OR_FEASIBILITY"
    {
        errors.push("candidate set claims exceed the illustrative, non-executed envelope".to_owned());
    }
    if receipt.generator.name != "NOT_IMPLEMENTED"
        || receipt.generator.version.is_some()
        || receipt.generator.build_digest.is_some()
        || receipt.generator.seed.is_some()
        || receipt.generator.method != "NOT_SELECTED"
        || receipt.generator.replay_status != "NOT_RUN"
    {
        errors.push("illustrative candidate set must not invent an executed generator identity".to_owned());
    }
    if receipt.candidate_count != receipt.candidates.len() || receipt.candidate_count != 3 {
        errors.push("candidate_count must exactly match the three illustrative records".to_owned());
    }

    let mut candidate_ids = BTreeSet::new();
    for candidate in &receipt.candidates {
        if candidate.candidate_id.trim().is_empty()
            || !candidate_ids.insert(candidate.candidate_id.clone())
        {
            errors.push("candidate IDs must be non-empty and unique".to_owned());
        }
        if candidate.candidate_state != "ILLUSTRATIVE_NOT_GENERATED"
            || candidate.interpretation.trim().is_empty()
            || !candidate.parameter_deltas.is_empty()
            || candidate.geometry_ref.is_some()
            || candidate.hard_gate_state != "UNKNOWN_NOT_EVALUATED"
            || candidate.physical_evaluation_state != "NOT_EVALUATED"
            || candidate.pareto_state != "NOT_EVALUATED"
            || candidate.resident_choice_state != "NOT_RECORDED"
        {
            errors.push(format!(
                "candidate {} attempts to promote an illustrative intent into a generated or evaluated design",
                candidate.candidate_id
            ));
        }

        let expected_unresolved = explicitly_unresolved_intent_axes(problem);
        let actual_unresolved: BTreeSet<String> =
            candidate.unresolved_intent_axes.iter().cloned().collect();
        if actual_unresolved.len() != candidate.unresolved_intent_axes.len()
            || actual_unresolved != expected_unresolved
        {
            errors.push(format!(
                "candidate {} does not preserve the parent's explicitly unresolved intent axes",
                candidate.candidate_id
            ));
        }

        for source in &candidate.source_intents {
            let known = match source.axis.as_str() {
                "spatial_typology" => vocabulary.spatial_typology_ids.contains(&source.option_id),
                "architectural_expression" => vocabulary.style_sources.contains_key(&source.option_id),
                "material_tectonic_intent" => vocabulary.material_tectonic_ids.contains(&source.option_id),
                "environmental_strategy_intent" => vocabulary.environmental_strategy_ids.contains(&source.option_id),
                "experience_programme" => vocabulary.experience_package_ids.contains(&source.option_id),
                _ => false,
            };
            if !known {
                errors.push(format!(
                    "candidate {} references unknown source intent {}:{}",
                    candidate.candidate_id, source.axis, source.option_id
                ));
            }
            if ![
                "PRIMARY", "SECONDARY", "ACCENT", "REFERENCE_ONLY", "AVOID",
                "ACCEPTABLE", "UNDECIDED", "PREFERRED", "CONSIDERED", "AVOIDED",
            ]
            .contains(&source.preference.as_str())
            {
                errors.push(format!(
                    "candidate {} contains unsupported source preference {}",
                    candidate.candidate_id, source.preference
                ));
            }
            match expected_resident_source_preference(problem, &source.axis, &source.option_id) {
                None => errors.push(format!(
                    "candidate {} invents source intent {}:{} that is absent or undecided in the parent problem",
                    candidate.candidate_id, source.axis, source.option_id
                )),
                Some(expected) if expected != source.preference => errors.push(format!(
                    "candidate {} changes resident source preference {}:{} from {} to {}",
                    candidate.candidate_id, source.axis, source.option_id, expected, source.preference
                )),
                Some(_) => {}
            }
        }
    }

    if errors.is_empty() { Ok(()) } else { Err(errors) }
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
        if evaluation.gate_state == GateState::Pass
            && (evaluation.objectives.is_empty()
                || evaluation.objectives.iter().any(|objective| {
                    objective.objective_id.trim().is_empty() || !objective.value.is_finite()
                }))
        {
            return Err(format!(
                "candidate {} is marked PASS but has an empty or non-finite objective vector",
                evaluation.candidate_id
            ));
        }
        if evaluation.gate_state == GateState::Pass {
            let mut objective_ids = BTreeSet::new();
            if evaluation.objectives.iter().any(|objective| {
                !objective_ids.insert(objective.objective_id.clone())
            }) {
                return Err(format!(
                    "candidate {} has duplicate objective identifiers",
                    evaluation.candidate_id
                ));
            }
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

/// Exact minimization-oriented hypervolume of a union of reference-point boxes.
/// Coordinates must already share a declared normalization/unit system.
fn dominated_hypervolume_recursive(points: &[Vec<f64>], reference: &[f64]) -> f64 {
    if points.is_empty() || reference.is_empty() {
        return 0.0;
    }
    if reference.len() == 1 {
        let best = points.iter().map(|point| point[0]).fold(f64::INFINITY, f64::min);
        return (reference[0] - best).max(0.0);
    }

    let mut boundaries: Vec<f64> = points
        .iter()
        .map(|point| point[0])
        .filter(|coordinate| *coordinate < reference[0])
        .collect();
    boundaries.sort_by(f64::total_cmp);
    boundaries.dedup_by(|left, right| *left == *right);

    let mut volume = 0.0;
    for (index, boundary) in boundaries.iter().enumerate() {
        let next = boundaries.get(index + 1).copied().unwrap_or(reference[0]);
        let width = next - boundary;
        if width <= 0.0 {
            continue;
        }
        let active: Vec<Vec<f64>> = points
            .iter()
            .filter(|point| point[0] <= *boundary)
            .map(|point| point[1..].to_vec())
            .collect();
        volume += width * dominated_hypervolume_recursive(&active, &reference[1..]);
    }
    volume
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ObjectiveNormalization {
    pub objective_id: String,
    pub direction: ObjectiveDirection,
    pub lower_bound: f64,
    pub upper_bound: f64,
}

/// Normalize declared raw objectives into a minimization vector.
///
/// Bounds must be pre-registered for the exact task and expressed in the
/// objective's native units. Out-of-range values fail closed; they are never
/// silently clamped because clipping changes the Pareto front and hypervolume.
pub fn normalize_evaluations(
    evaluations: &[OracleEvaluation],
    normalizations: &[ObjectiveNormalization],
) -> Result<Vec<OracleEvaluation>, String> {
    if normalizations.is_empty() {
        return Err("at least one objective normalization is required".to_owned());
    }
    let mut seen_ids = BTreeSet::new();
    for rule in normalizations {
        if rule.objective_id.trim().is_empty() || !seen_ids.insert(rule.objective_id.clone()) {
            return Err("normalization objective IDs must be non-empty and unique".to_owned());
        }
        if !rule.lower_bound.is_finite()
            || !rule.upper_bound.is_finite()
            || rule.upper_bound <= rule.lower_bound
        {
            return Err(format!("objective {} has invalid normalization bounds", rule.objective_id));
        }
    }

    let mut candidate_ids = BTreeSet::new();
    let mut normalized = Vec::with_capacity(evaluations.len());
    for evaluation in evaluations {
        if evaluation.candidate_id.trim().is_empty()
            || !candidate_ids.insert(evaluation.candidate_id.clone())
        {
            return Err("candidate IDs must be non-empty and unique".to_owned());
        }
        if evaluation.objectives.is_empty() {
            if evaluation.gate_state == GateState::Pass {
                return Err(format!("PASS candidate {} has no objective values", evaluation.candidate_id));
            }
            normalized.push(evaluation.clone());
            continue;
        }
        if evaluation.objectives.len() != normalizations.len() {
            return Err(format!("candidate {} objective count differs from normalization schema", evaluation.candidate_id));
        }

        let mut values = Vec::with_capacity(normalizations.len());
        for (objective, rule) in evaluation.objectives.iter().zip(normalizations) {
            if objective.objective_id != rule.objective_id || objective.direction != rule.direction {
                return Err(format!("candidate {} objective order/direction differs at {}", evaluation.candidate_id, rule.objective_id));
            }
            if !objective.value.is_finite()
                || objective.value < rule.lower_bound
                || objective.value > rule.upper_bound
            {
                return Err(format!("candidate {} objective {} is outside its pre-registered bounds", evaluation.candidate_id, rule.objective_id));
            }
            let width = rule.upper_bound - rule.lower_bound;
            let value = match rule.direction {
                ObjectiveDirection::Minimize => (objective.value - rule.lower_bound) / width,
                ObjectiveDirection::Maximize => (rule.upper_bound - objective.value) / width,
            };
            values.push(OracleObjective {
                objective_id: rule.objective_id.clone(),
                direction: ObjectiveDirection::Minimize,
                value,
            });
        }
        normalized.push(OracleEvaluation {
            candidate_id: evaluation.candidate_id.clone(),
            gate_state: evaluation.gate_state.clone(),
            objectives: values,
        });
    }
    Ok(normalized)
}

/// Compute exact hypervolume for PASS candidates only.
///
/// The reference point is expressed in each objective's native direction and
/// units. It must be worse than the solutions being measured: higher for a
/// minimized objective and lower for a maximized objective. Objectives should
/// be normalized before cross-case comparison; this function does not invent
/// normalization bounds. FAIL/UNKNOWN/CONDITIONAL candidates contribute zero.
/// A malformed PASS receipt or mismatched objective schema is an error.
pub fn exact_hypervolume(
    evaluations: &[OracleEvaluation],
    reference_point: &[f64],
) -> Result<f64, String> {
    if reference_point.is_empty() || reference_point.iter().any(|value| !value.is_finite()) {
        return Err("hypervolume reference point must be non-empty and finite".to_owned());
    }

    let front_ids = exact_pareto_front(evaluations)?;
    if front_ids.is_empty() {
        return Ok(0.0);
    }

    let first = evaluations
        .iter()
        .find(|evaluation| evaluation.candidate_id == front_ids[0])
        .ok_or_else(|| "exact Pareto front referenced a missing candidate".to_owned())?;
    if first.objectives.len() != reference_point.len() {
        return Err("hypervolume reference-point dimension differs from objective schema".to_owned());
    }

    let mut normalized_points = Vec::new();
    for candidate_id in front_ids {
        let candidate = evaluations
            .iter()
            .find(|evaluation| evaluation.candidate_id == candidate_id)
            .ok_or_else(|| format!("Pareto candidate {candidate_id} is missing"))?;

        let mut point = Vec::with_capacity(candidate.objectives.len());
        for (objective, reference) in candidate.objectives.iter().zip(reference_point) {
            let (value, transformed_reference) = match objective.direction {
                ObjectiveDirection::Minimize => (objective.value, *reference),
                ObjectiveDirection::Maximize => (-objective.value, -*reference),
            };
            if value > transformed_reference {
                return Err(format!(
                    "hypervolume reference point does not bound PASS candidate {}",
                    candidate.candidate_id
                ));
            }
            point.push(value);
        }
        normalized_points.push(point);
    }

    let transformed_reference: Vec<f64> = first
        .objectives
        .iter()
        .zip(reference_point)
        .map(|(objective, value)| match objective.direction {
            ObjectiveDirection::Minimize => *value,
            ObjectiveDirection::Maximize => -*value,
        })
        .collect();

    let volume = dominated_hypervolume_recursive(&normalized_points, &transformed_reference);
    if volume.is_finite() && volume >= 0.0 {
        Ok(volume)
    } else {
        Err("hypervolume calculation produced a non-finite or negative result".to_owned())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ScenarioEvaluation {
    pub scenario_id: String,
    pub gate_state: GateState,
    pub objectives: Vec<OracleObjective>,
}

/// Conservative worst-case aggregation over an explicitly supplied scenario set.
/// Missing, unknown or failing scenarios abstain instead of being pooled away.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct EvidenceDependencyNode {
    pub node_id: String,
    pub content_digest: String,
    #[serde(default)]
    pub depends_on: Vec<String>,
    #[serde(default)]
    pub evidence_receipt_digest: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct EvidenceDependencyGraph {
    pub graph_id: String,
    pub graph_revision: String,
    pub nodes: Vec<EvidenceDependencyNode>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct EvidenceReceiptReference {
    pub node_id: String,
    pub receipt_digest: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct InvalidationReceipt {
    pub graph_id: String,
    pub prior_revision: String,
    pub new_revision: String,
    pub graph_digest: String,
    pub changed_node_ids: Vec<String>,
    pub invalidated_node_ids: Vec<String>,
    pub invalidated_evidence: Vec<EvidenceReceiptReference>,
    pub preserved_evidence: Vec<EvidenceReceiptReference>,
    pub receipt_digest: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum NodeChangeReason {
    Added,
    Removed,
    ContentChanged,
    DependenciesChanged,
    EvidenceReceiptChanged,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct NodeRevisionChange {
    pub node_id: String,
    pub reasons: Vec<NodeChangeReason>,
    pub prior_content_digest: Option<String>,
    pub current_content_digest: Option<String>,
    pub prior_dependencies: Vec<String>,
    pub current_dependencies: Vec<String>,
    pub prior_evidence_receipt_digest: Option<String>,
    pub current_evidence_receipt_digest: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct EvidenceReceiptTransition {
    pub node_id: String,
    pub prior_receipt_digest: Option<String>,
    pub current_receipt_digest: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct RevisionInvalidationReceipt {
    pub graph_id: String,
    pub prior_revision: String,
    pub new_revision: String,
    pub prior_graph_digest: String,
    pub current_graph_digest: String,
    pub changes: Vec<NodeRevisionChange>,
    pub changed_node_ids: Vec<String>,
    pub invalidated_node_ids: Vec<String>,
    pub invalidated_evidence: Vec<EvidenceReceiptTransition>,
    pub preserved_evidence: Vec<EvidenceReceiptReference>,
    pub receipt_digest: String,
}

fn is_blake3_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

fn canonical_dependency_graph(graph: &EvidenceDependencyGraph) -> EvidenceDependencyGraph {
    let mut canonical = graph.clone();
    canonical.nodes.sort_by(|left, right| left.node_id.cmp(&right.node_id));
    for node in &mut canonical.nodes {
        node.depends_on.sort();
    }
    canonical
}

fn validate_dependency_graph(
    graph: &EvidenceDependencyGraph,
) -> Result<(EvidenceDependencyGraph, BTreeMap<String, BTreeSet<String>>), Vec<String>> {
    let mut errors = Vec::new();
    if graph.graph_id.trim().is_empty() || graph.graph_revision.trim().is_empty() {
        errors.push("dependency graph ID and revision are required".to_owned());
    }
    if graph.nodes.is_empty() {
        errors.push("dependency graph must contain at least one node".to_owned());
    }

    let mut nodes_by_id = BTreeMap::new();
    for node in &graph.nodes {
        if node.node_id.trim().is_empty() {
            errors.push("dependency node ID cannot be empty".to_owned());
        }
        if nodes_by_id.insert(node.node_id.clone(), node).is_some() {
            errors.push(format!("duplicate dependency node ID: {}", node.node_id));
        }
        if !is_blake3_digest(&node.content_digest) {
            errors.push(format!("node {} has an invalid content digest", node.node_id));
        }
        if node
            .evidence_receipt_digest
            .as_deref()
            .is_some_and(|digest| !is_blake3_digest(digest))
        {
            errors.push(format!("node {} has an invalid evidence receipt digest", node.node_id));
        }
        let mut seen_dependencies = BTreeSet::new();
        for dependency in &node.depends_on {
            if dependency == &node.node_id {
                errors.push(format!("node {} cannot depend on itself", node.node_id));
            }
            if !seen_dependencies.insert(dependency.clone()) {
                errors.push(format!("node {} has duplicate dependency {dependency}", node.node_id));
            }
        }
    }

    let mut dependents: BTreeMap<String, BTreeSet<String>> = nodes_by_id
        .keys()
        .map(|node_id| (node_id.clone(), BTreeSet::new()))
        .collect();
    let mut indegree: BTreeMap<String, usize> = nodes_by_id
        .keys()
        .map(|node_id| (node_id.clone(), 0))
        .collect();

    for node in &graph.nodes {
        for dependency in &node.depends_on {
            if !nodes_by_id.contains_key(dependency) {
                errors.push(format!("node {} references missing dependency {dependency}", node.node_id));
                continue;
            }
            dependents
                .entry(dependency.clone())
                .or_default()
                .insert(node.node_id.clone());
            *indegree.entry(node.node_id.clone()).or_default() += 1;
        }
    }
    if !errors.is_empty() {
        return Err(errors);
    }

    // Kahn's algorithm detects cycles without recursion-depth dependence.
    let mut ready: BTreeSet<String> = indegree
        .iter()
        .filter(|(_, degree)| **degree == 0)
        .map(|(node_id, _)| node_id.clone())
        .collect();
    let mut visited = 0usize;
    while let Some(node_id) = ready.pop_first() {
        visited += 1;
        if let Some(children) = dependents.get(&node_id) {
            for child in children {
                if let Some(degree) = indegree.get_mut(child) {
                    *degree -= 1;
                    if *degree == 0 {
                        ready.insert(child.clone());
                    }
                }
            }
        }
    }
    if visited != nodes_by_id.len() {
        return Err(vec!["dependency graph contains a directed cycle".to_owned()]);
    }

    Ok((canonical_dependency_graph(graph), dependents))
}

/// Compute the transitive invalidation closure for an explicit set of changed
/// source nodes. Every downstream dependent is stale. This declared graph is
/// not an oracle for hidden causal dependencies.
pub fn compute_invalidation_receipt(
    graph: &EvidenceDependencyGraph,
    prior_revision: &str,
    new_revision: &str,
    changed_node_ids: &[String],
) -> Result<InvalidationReceipt, Vec<String>> {
    let (canonical, dependents) = validate_dependency_graph(graph)?;
    let mut errors = Vec::new();
    if prior_revision.trim().is_empty() || new_revision.trim().is_empty() {
        errors.push("prior and new revisions are required".to_owned());
    }
    if graph.graph_revision != prior_revision {
        errors.push("dependency graph revision does not match prior revision".to_owned());
    }
    if prior_revision == new_revision {
        errors.push("new revision must differ from prior revision".to_owned());
    }

    let known_nodes: BTreeSet<String> = canonical
        .nodes
        .iter()
        .map(|node| node.node_id.clone())
        .collect();
    let changed: BTreeSet<String> = changed_node_ids.iter().cloned().collect();
    if changed.is_empty() {
        errors.push("at least one changed node is required".to_owned());
    }
    if changed.len() != changed_node_ids.len() {
        errors.push("changed node IDs must be unique".to_owned());
    }
    for node_id in &changed {
        if !known_nodes.contains(node_id) {
            errors.push(format!("changed node does not exist: {node_id}"));
        }
    }
    if !errors.is_empty() {
        return Err(errors);
    }

    let mut invalidated = changed.clone();
    let mut frontier = changed.clone();
    while let Some(node_id) = frontier.pop_first() {
        if let Some(children) = dependents.get(&node_id) {
            for child in children {
                if invalidated.insert(child.clone()) {
                    frontier.insert(child.clone());
                }
            }
        }
    }

    let mut invalidated_evidence = Vec::new();
    let mut preserved_evidence = Vec::new();
    for node in &canonical.nodes {
        if let Some(receipt_digest) = &node.evidence_receipt_digest {
            let reference = EvidenceReceiptReference {
                node_id: node.node_id.clone(),
                receipt_digest: receipt_digest.clone(),
            };
            if invalidated.contains(&node.node_id) {
                invalidated_evidence.push(reference);
            } else {
                preserved_evidence.push(reference);
            }
        }
    }

    let graph_bytes = serde_json::to_vec(&canonical)
        .map_err(|error| vec![format!("cannot serialize dependency graph: {error}")])?;
    let graph_digest = blake3::hash(&graph_bytes).to_hex().to_string();
    let mut receipt = InvalidationReceipt {
        graph_id: canonical.graph_id.clone(),
        prior_revision: prior_revision.to_owned(),
        new_revision: new_revision.to_owned(),
        graph_digest,
        changed_node_ids: changed.into_iter().collect(),
        invalidated_node_ids: invalidated.into_iter().collect(),
        invalidated_evidence,
        preserved_evidence,
        receipt_digest: String::new(),
    };
    let payload = serde_json::to_vec(&(
        &receipt.graph_id,
        &receipt.prior_revision,
        &receipt.new_revision,
        &receipt.graph_digest,
        &receipt.changed_node_ids,
        &receipt.invalidated_node_ids,
        &receipt.invalidated_evidence,
        &receipt.preserved_evidence,
    ))
    .map_err(|error| vec![format!("cannot serialize invalidation receipt: {error}")])?;
    receipt.receipt_digest = blake3::hash(&payload).to_hex().to_string();
    Ok(receipt)
}

/// Verify that a receipt still describes the current declared graph and change set.
pub fn verify_invalidation_receipt(
    graph: &EvidenceDependencyGraph,
    expected: &InvalidationReceipt,
    changed_node_ids: &[String],
) -> Result<(), Vec<String>> {
    let actual = compute_invalidation_receipt(
        graph,
        &expected.prior_revision,
        &expected.new_revision,
        changed_node_ids,
    )?;
    if actual != *expected {
        return Err(vec![
            "invalidation receipt does not match the current graph/change set".to_owned(),
        ]);
    }
    Ok(())
}

fn dependency_graph_digest(graph: &EvidenceDependencyGraph) -> Result<String, Vec<String>> {
    let canonical = canonical_dependency_graph(graph);
    let bytes = serde_json::to_vec(&canonical)
        .map_err(|error| vec![format!("cannot serialize dependency graph: {error}")])?;
    Ok(blake3::hash(&bytes).to_hex().to_string())
}

/// Derive changed nodes directly from two adjacent graph revisions. Changes
/// include additions, removals, content changes, dependency rewires and
/// evidence-receipt changes. The invalidation closure follows the union of old
/// and new dependency edges so removed prerequisites cannot leave stale
/// downstream evidence active.
pub fn compute_revision_invalidation_receipt(
    prior_graph: &EvidenceDependencyGraph,
    current_graph: &EvidenceDependencyGraph,
) -> Result<RevisionInvalidationReceipt, Vec<String>> {
    let (prior_canonical, prior_dependents) = validate_dependency_graph(prior_graph)?;
    let (current_canonical, current_dependents) = validate_dependency_graph(current_graph)?;
    let mut errors = Vec::new();
    if prior_canonical.graph_id != current_canonical.graph_id {
        errors.push("graph ID cannot change across a revision diff".to_owned());
    }
    if prior_canonical.graph_revision.trim().is_empty()
        || current_canonical.graph_revision.trim().is_empty()
    {
        errors.push("both graph revisions are required".to_owned());
    }
    if prior_canonical.graph_revision == current_canonical.graph_revision {
        errors.push("current graph revision must differ from prior graph revision".to_owned());
    }
    if !errors.is_empty() {
        return Err(errors);
    }

    let prior_nodes: BTreeMap<String, &EvidenceDependencyNode> = prior_canonical
        .nodes
        .iter()
        .map(|node| (node.node_id.clone(), node))
        .collect();
    let current_nodes: BTreeMap<String, &EvidenceDependencyNode> = current_canonical
        .nodes
        .iter()
        .map(|node| (node.node_id.clone(), node))
        .collect();
    let all_node_ids: BTreeSet<String> = prior_nodes
        .keys()
        .chain(current_nodes.keys())
        .cloned()
        .collect();

    let mut changes = Vec::new();
    for node_id in &all_node_ids {
        let prior = prior_nodes.get(node_id).copied();
        let current = current_nodes.get(node_id).copied();
        let mut reasons = Vec::new();
        match (prior, current) {
            (None, Some(_)) => reasons.push(NodeChangeReason::Added),
            (Some(_), None) => reasons.push(NodeChangeReason::Removed),
            (Some(before), Some(after)) => {
                if before.content_digest != after.content_digest {
                    reasons.push(NodeChangeReason::ContentChanged);
                }
                let before_dependencies: BTreeSet<String> =
                    before.depends_on.iter().cloned().collect();
                let after_dependencies: BTreeSet<String> =
                    after.depends_on.iter().cloned().collect();
                if before_dependencies != after_dependencies {
                    reasons.push(NodeChangeReason::DependenciesChanged);
                }
                if before.evidence_receipt_digest != after.evidence_receipt_digest {
                    reasons.push(NodeChangeReason::EvidenceReceiptChanged);
                }
            }
            (None, None) => unreachable!("node ID came from the union of graph node sets"),
        }

        if !reasons.is_empty() {
            let mut prior_dependencies = prior
                .map(|node| node.depends_on.clone())
                .unwrap_or_default();
            let mut current_dependencies = current
                .map(|node| node.depends_on.clone())
                .unwrap_or_default();
            prior_dependencies.sort();
            current_dependencies.sort();
            changes.push(NodeRevisionChange {
                node_id: node_id.clone(),
                reasons,
                prior_content_digest: prior.map(|node| node.content_digest.clone()),
                current_content_digest: current.map(|node| node.content_digest.clone()),
                prior_dependencies,
                current_dependencies,
                prior_evidence_receipt_digest: prior
                    .and_then(|node| node.evidence_receipt_digest.clone()),
                current_evidence_receipt_digest: current
                    .and_then(|node| node.evidence_receipt_digest.clone()),
            });
        }
    }

    if changes.is_empty() {
        return Err(vec![
            "graph revision changed without node-level content, dependency or receipt changes"
                .to_owned(),
        ]);
    }
    changes.sort_by(|left, right| left.node_id.cmp(&right.node_id));
    let changed_node_ids: Vec<String> =
        changes.iter().map(|change| change.node_id.clone()).collect();
    let changed: BTreeSet<String> = changed_node_ids.iter().cloned().collect();

    // A consumer may rely on an edge that exists only in the prior graph or
    // only in the current graph. Both relationships matter for invalidation.
    let mut union_dependents: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for node_id in prior_nodes.keys().chain(current_nodes.keys()) {
        union_dependents.entry(node_id.clone()).or_default();
    }
    for dependent_map in [&prior_dependents, &current_dependents] {
        for (source, children) in dependent_map {
            union_dependents
                .entry(source.clone())
                .or_default()
                .extend(children.iter().cloned());
        }
    }

    let mut invalidated = changed.clone();
    let mut frontier = changed.clone();
    while let Some(node_id) = frontier.pop_first() {
        if let Some(children) = union_dependents.get(&node_id) {
            for child in children {
                if invalidated.insert(child.clone()) {
                    frontier.insert(child.clone());
                }
            }
        }
    }

    let mut invalidated_evidence = Vec::new();
    for node_id in &invalidated {
        let prior_receipt_digest = prior_nodes
            .get(node_id)
            .and_then(|node| node.evidence_receipt_digest.clone());
        let current_receipt_digest = current_nodes
            .get(node_id)
            .and_then(|node| node.evidence_receipt_digest.clone());
        if prior_receipt_digest.is_some() || current_receipt_digest.is_some() {
            invalidated_evidence.push(EvidenceReceiptTransition {
                node_id: node_id.clone(),
                prior_receipt_digest,
                current_receipt_digest,
            });
        }
    }
    invalidated_evidence.sort_by(|left, right| left.node_id.cmp(&right.node_id));

    let mut preserved_evidence = Vec::new();
    for (node_id, current) in &current_nodes {
        if !invalidated.contains(node_id) {
            if let Some(receipt_digest) = &current.evidence_receipt_digest {
                preserved_evidence.push(EvidenceReceiptReference {
                    node_id: node_id.clone(),
                    receipt_digest: receipt_digest.clone(),
                });
            }
        }
    }
    preserved_evidence.sort_by(|left, right| left.node_id.cmp(&right.node_id));

    let prior_graph_digest = dependency_graph_digest(&prior_canonical)?;
    let current_graph_digest = dependency_graph_digest(&current_canonical)?;
    let mut receipt = RevisionInvalidationReceipt {
        graph_id: current_canonical.graph_id.clone(),
        prior_revision: prior_canonical.graph_revision.clone(),
        new_revision: current_canonical.graph_revision.clone(),
        prior_graph_digest,
        current_graph_digest,
        changes,
        changed_node_ids,
        invalidated_node_ids: invalidated.into_iter().collect(),
        invalidated_evidence,
        preserved_evidence,
        receipt_digest: String::new(),
    };
    let payload = serde_json::to_vec(&(
        &receipt.graph_id,
        &receipt.prior_revision,
        &receipt.new_revision,
        &receipt.prior_graph_digest,
        &receipt.current_graph_digest,
        &receipt.changes,
        &receipt.changed_node_ids,
        &receipt.invalidated_node_ids,
        &receipt.invalidated_evidence,
        &receipt.preserved_evidence,
    ))
    .map_err(|error| vec![format!("cannot serialize revision invalidation receipt: {error}")])?;
    receipt.receipt_digest = blake3::hash(&payload).to_hex().to_string();
    Ok(receipt)
}

pub fn verify_revision_invalidation_receipt(
    prior_graph: &EvidenceDependencyGraph,
    current_graph: &EvidenceDependencyGraph,
    expected: &RevisionInvalidationReceipt,
) -> Result<(), Vec<String>> {
    let actual = compute_revision_invalidation_receipt(prior_graph, current_graph)?;
    if actual != *expected {
        return Err(vec![
            "revision invalidation receipt differs from the two supplied graph revisions"
                .to_owned(),
        ]);
    }
    Ok(())
}

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

    const SYNTHETIC_CATALOGUE_SCHEMA: &str = r#"{
        "$schema":"https://json-schema.org/draft/2020-12/schema",
        "type":"object",
        "properties":{"catalog_revision":{"type":"string"}}
    }"#;

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
        let mut problem: DesignProblem =
            serde_json::from_str(SYNTHETIC_PROBLEM).expect("synthetic problem must deserialize");
        // Unit tests use a deliberately minimal vocabulary fixture. Bind its
        // exact Git blob digest rather than pretending it is the full upstream
        // Sol-Atlas catalogue.
        let vocabulary = vocabulary();
        problem.source_bindings.catalogue_blob_sha = vocabulary.catalogue_blob_sha;
        problem.source_bindings.catalogue_schema_blob_sha =
            vocabulary.catalogue_schema_blob_sha;
        problem
    }

    fn vocabulary() -> CatalogueVocabulary {
        CatalogueVocabulary::from_catalogue_sources(
            &synthetic_catalogue().to_string(),
            SYNTHETIC_CATALOGUE_SCHEMA,
        )
        .unwrap()
    }

    #[derive(Deserialize)]
    struct EvidenceDependencyNodeFixture {
        node_id: String,
        content_payload: String,
        depends_on: Vec<String>,
        evidence_receipt_payload: Option<String>,
    }

    #[derive(Deserialize)]
    struct EvidenceDependencyGraphFixture {
        graph_id: String,
        graph_revision: String,
        nodes: Vec<EvidenceDependencyNodeFixture>,
        expected_cases: Vec<EvidenceInvalidationCaseFixture>,
    }

    #[derive(Deserialize)]
    struct RevisionGraphFixture {
        graph_id: String,
        graph_revision: String,
        nodes: Vec<EvidenceDependencyNodeFixture>,
    }

    #[derive(Deserialize)]
    struct RevisionDiffFixture {
        scenario_id: String,
        prior_graph: RevisionGraphFixture,
        current_graph: RevisionGraphFixture,
        expected_changed_node_ids: Vec<String>,
        expected_invalidated_node_ids: Vec<String>,
        expected_preserved_evidence_node_ids: Vec<String>,
        expected_change_reasons: BTreeMap<String, Vec<NodeChangeReason>>,
    }

    const SYNTHETIC_REVISION_DIFF: &str = include_str!(
        "../../../../docs/research/built-environment/EVIDENCE_REVISION_DIFF_SYNTHETIC_V0.1.json"
    );

    fn graph_from_revision_fixture(fixture: RevisionGraphFixture) -> EvidenceDependencyGraph {
        EvidenceDependencyGraph {
            graph_id: fixture.graph_id,
            graph_revision: fixture.graph_revision,
            nodes: fixture
                .nodes
                .into_iter()
                .map(|node| EvidenceDependencyNode {
                    node_id: node.node_id,
                    content_digest: blake3::hash(node.content_payload.as_bytes())
                        .to_hex()
                        .to_string(),
                    depends_on: node.depends_on,
                    evidence_receipt_digest: node
                        .evidence_receipt_payload
                        .map(|payload| blake3::hash(payload.as_bytes()).to_hex().to_string()),
                })
                .collect(),
        }
    }

    #[derive(Deserialize)]
    struct EvidenceInvalidationCaseFixture {
        case_id: String,
        changed_node_ids: Vec<String>,
        invalidated_node_ids: Vec<String>,
        preserved_evidence_node_ids: Vec<String>,
    }

    const SYNTHETIC_DEPENDENCY_GRAPH: &str = include_str!(
        "../../../../docs/research/built-environment/EVIDENCE_DEPENDENCY_GRAPH_SYNTHETIC_V0.1.json"
    );

    fn dependency_graph_fixture() -> EvidenceDependencyGraphFixture {
        serde_json::from_str(SYNTHETIC_DEPENDENCY_GRAPH)
            .expect("synthetic evidence dependency graph must deserialize")
    }

    fn sample_dependency_graph() -> EvidenceDependencyGraph {
        let fixture = dependency_graph_fixture();
        EvidenceDependencyGraph {
            graph_id: fixture.graph_id,
            graph_revision: fixture.graph_revision,
            nodes: fixture
                .nodes
                .into_iter()
                .map(|node| EvidenceDependencyNode {
                    node_id: node.node_id,
                    content_digest: blake3::hash(node.content_payload.as_bytes())
                        .to_hex()
                        .to_string(),
                    depends_on: node.depends_on,
                    evidence_receipt_digest: node
                        .evidence_receipt_payload
                        .map(|payload| blake3::hash(payload.as_bytes()).to_hex().to_string()),
                })
                .collect(),
        }
    }

    #[test]
    fn frozen_dependency_graph_fixture_matches_expected_invalidation_sets() {
        let fixture = dependency_graph_fixture();
        let graph = sample_dependency_graph();
        for case in fixture.expected_cases {
            let receipt = compute_invalidation_receipt(
                &graph,
                "REV-A",
                "REV-B",
                &case.changed_node_ids,
            )
            .unwrap_or_else(|errors| panic!("{}: {errors:?}", case.case_id));
            assert_eq!(
                receipt.invalidated_node_ids,
                case.invalidated_node_ids,
                "invalidated set differs for {}",
                case.case_id
            );
            assert_eq!(
                receipt
                    .preserved_evidence
                    .iter()
                    .map(|reference| reference.node_id.clone())
                    .collect::<Vec<_>>(),
                case.preserved_evidence_node_ids,
                "preserved evidence differs for {}",
                case.case_id
            );
        }
    }

    #[test]
    fn invalidation_diamond_only_stales_changed_branch_and_descendants() {
        let graph = sample_dependency_graph();
        let receipt = compute_invalidation_receipt(
            &graph,
            "REV-A",
            "REV-B",
            &["style-intent".to_owned()],
        )
        .unwrap();
        assert_eq!(
            receipt.invalidated_node_ids,
            vec![
                "energy-evaluation".to_owned(),
                "geometry".to_owned(),
                "style-intent".to_owned(),
            ]
        );
        let invalidated: BTreeSet<_> = receipt
            .invalidated_evidence
            .iter()
            .map(|item| item.node_id.as_str())
            .collect();
        assert_eq!(
            invalidated,
            BTreeSet::from(["energy-evaluation", "geometry", "style-intent"])
        );
        let preserved: BTreeSet<_> = receipt
            .preserved_evidence
            .iter()
            .map(|item| item.node_id.as_str())
            .collect();
        assert_eq!(
            preserved,
            BTreeSet::from(["typology-intent", "water-evaluation", "water-policy"])
        );
        verify_invalidation_receipt(&graph, &receipt, &["style-intent".to_owned()]).unwrap();
    }

    #[test]
    fn common_root_change_invalidates_all_declared_descendants() {
        let graph = sample_dependency_graph();
        let receipt = compute_invalidation_receipt(
            &graph,
            "REV-A",
            "REV-B",
            &["resident-brief".to_owned()],
        )
        .unwrap();
        assert_eq!(
            receipt.invalidated_node_ids,
            vec![
                "energy-evaluation".to_owned(),
                "geometry".to_owned(),
                "resident-brief".to_owned(),
                "style-intent".to_owned(),
                "typology-intent".to_owned(),
            ]
        );
        assert_eq!(
            receipt
                .preserved_evidence
                .iter()
                .map(|item| item.node_id.as_str())
                .collect::<Vec<_>>(),
            vec!["water-evaluation", "water-policy"]
        );
    }

    #[test]
    fn invalidation_rejects_cycles_missing_nodes_and_empty_changes() {
        let mut cycle = sample_dependency_graph();
        cycle.nodes[0].depends_on = vec!["energy-evaluation".to_owned()];
        assert!(validate_dependency_graph(&cycle).is_err());

        let mut missing = sample_dependency_graph();
        missing.nodes[1].depends_on = vec!["missing-node".to_owned()];
        assert!(validate_dependency_graph(&missing).is_err());

        let graph = sample_dependency_graph();
        assert!(compute_invalidation_receipt(&graph, "REV-A", "REV-B", &[]).is_err());
        assert!(
            compute_invalidation_receipt(
                &graph,
                "REV-A",
                "REV-B",
                &["not-present".to_owned()]
            )
            .is_err()
        );
    }

    #[test]
    fn invalidation_rejects_duplicate_edges_self_edges_bad_digests_and_stale_revision() {
        let mut duplicate = sample_dependency_graph();
        duplicate.nodes[3].depends_on.push("style-intent".to_owned());
        assert!(validate_dependency_graph(&duplicate).is_err());

        let mut self_edge = sample_dependency_graph();
        self_edge.nodes[2].depends_on.push("typology-intent".to_owned());
        assert!(validate_dependency_graph(&self_edge).is_err());

        let mut bad_digest = sample_dependency_graph();
        bad_digest.nodes[0].content_digest = "not-a-blake3-digest".to_owned();
        assert!(validate_dependency_graph(&bad_digest).is_err());

        let graph = sample_dependency_graph();
        assert!(compute_invalidation_receipt(
            &graph,
            "STALE-REVISION",
            "REV-B",
            &["style-intent".to_owned()]
        )
        .is_err());
        assert!(compute_invalidation_receipt(
            &graph,
            "REV-A",
            "REV-B",
            &["style-intent".to_owned(), "style-intent".to_owned()]
        )
        .is_err());
    }

    #[test]
    fn invalidation_receipt_is_order_independent_and_detects_graph_drift() {
        let graph = sample_dependency_graph();
        let changed = vec!["typology-intent".to_owned()];
        let first =
            compute_invalidation_receipt(&graph, "REV-A", "REV-B", &changed).unwrap();

        let mut reordered = graph.clone();
        reordered.nodes.reverse();
        for node in &mut reordered.nodes {
            node.depends_on.reverse();
        }
        let second =
            compute_invalidation_receipt(&reordered, "REV-A", "REV-B", &changed).unwrap();
        assert_eq!(first, second);

        let mut drifted = graph;
        drifted.nodes[1].content_digest = blake3::hash(b"style-b").to_hex().to_string();
        assert!(verify_invalidation_receipt(&drifted, &first, &changed).is_err());
    }

    #[test]
    fn frozen_revision_diff_fixture_matches_derived_changes_and_invalidation() {
        let fixture: RevisionDiffFixture =
            serde_json::from_str(SYNTHETIC_REVISION_DIFF)
                .expect("synthetic revision-diff fixture must deserialize");
        let prior = graph_from_revision_fixture(fixture.prior_graph);
        let current = graph_from_revision_fixture(fixture.current_graph);
        let receipt = compute_revision_invalidation_receipt(&prior, &current)
            .unwrap_or_else(|errors| panic!("{}: {errors:?}", fixture.scenario_id));

        assert_eq!(receipt.changed_node_ids, fixture.expected_changed_node_ids);
        assert_eq!(receipt.invalidated_node_ids, fixture.expected_invalidated_node_ids);
        assert_eq!(
            receipt.preserved_evidence.iter().map(|item| item.node_id.clone()).collect::<Vec<_>>(),
            fixture.expected_preserved_evidence_node_ids
        );
        let reasons: BTreeMap<String, Vec<NodeChangeReason>> = receipt
            .changes
            .iter()
            .map(|change| (change.node_id.clone(), change.reasons.clone()))
            .collect();
        assert_eq!(reasons, fixture.expected_change_reasons);
        verify_revision_invalidation_receipt(&prior, &current, &receipt).unwrap();
    }

    #[test]
    fn revision_diff_derives_changed_nodes_and_uses_union_of_edges() {
        let prior = sample_dependency_graph();
        let mut current = prior.clone();
        current.graph_revision = "REV-B".to_owned();
        let style = current.nodes.iter_mut()
            .find(|node| node.node_id == "style-intent").unwrap();
        style.content_digest = blake3::hash(b"style-b").to_hex().to_string();

        let receipt = compute_revision_invalidation_receipt(&prior, &current).unwrap();
        assert_eq!(receipt.changed_node_ids, vec!["style-intent".to_owned()]);
        assert_eq!(
            receipt.invalidated_node_ids,
            vec![
                "energy-evaluation".to_owned(),
                "geometry".to_owned(),
                "style-intent".to_owned()
            ]
        );
        assert_eq!(
            receipt.preserved_evidence.iter().map(|item| item.node_id.as_str()).collect::<Vec<_>>(),
            vec!["typology-intent", "water-evaluation", "water-policy"]
        );
        verify_revision_invalidation_receipt(&prior, &current, &receipt).unwrap();
    }

    #[test]
    fn revision_diff_detects_added_removed_rewired_and_receipt_changed_nodes() {
        let prior = sample_dependency_graph();
        let mut current = prior.clone();
        current.graph_revision = "REV-B".to_owned();
        current.nodes.push(EvidenceDependencyNode {
            node_id: "material-spec".to_owned(),
            content_digest: blake3::hash(b"material-v1").to_hex().to_string(),
            depends_on: vec!["resident-brief".to_owned()],
            evidence_receipt_digest: Some(blake3::hash(b"material-receipt").to_hex().to_string()),
        });
        let geometry = current.nodes.iter_mut()
            .find(|node| node.node_id == "geometry").unwrap();
        geometry.depends_on.push("material-spec".to_owned());

        let water = current.nodes.iter_mut()
            .find(|node| node.node_id == "water-policy").unwrap();
        water.evidence_receipt_digest =
            Some(blake3::hash(b"water-policy-receipt-v2").to_hex().to_string());

        let receipt = compute_revision_invalidation_receipt(&prior, &current).unwrap();
        assert_eq!(
            receipt.changed_node_ids,
            vec![
                "geometry".to_owned(),
                "material-spec".to_owned(),
                "water-policy".to_owned()
            ]
        );
        assert!(receipt.invalidated_node_ids.contains(&"energy-evaluation".to_owned()));
        assert!(receipt.invalidated_node_ids.contains(&"water-evaluation".to_owned()));
        assert!(receipt.invalidated_node_ids.contains(&"material-spec".to_owned()));

        let mut removed = current.clone();
        removed.graph_revision = "REV-C".to_owned();
        removed.nodes.retain(|node| node.node_id != "material-spec");
        let geometry = removed.nodes.iter_mut()
            .find(|node| node.node_id == "geometry").unwrap();
        geometry.depends_on.retain(|dependency| dependency != "material-spec");
        let removed_receipt = compute_revision_invalidation_receipt(&current, &removed).unwrap();
        assert!(removed_receipt.changed_node_ids.contains(&"material-spec".to_owned()));
        assert!(removed_receipt.invalidated_node_ids.contains(&"geometry".to_owned()));
        assert!(removed_receipt.invalidated_node_ids.contains(&"energy-evaluation".to_owned()));
    }

    #[test]
    fn revision_diff_rejects_noop_revision_and_wrong_graph_identity() {
        let prior = sample_dependency_graph();
        let mut noop = prior.clone();
        noop.graph_revision = "REV-B".to_owned();
        assert!(compute_revision_invalidation_receipt(&prior, &noop).is_err());

        let mut wrong_identity = prior.clone();
        wrong_identity.graph_revision = "REV-B".to_owned();
        wrong_identity.graph_id = "DIFFERENT-GRAPH".to_owned();
        assert!(compute_revision_invalidation_receipt(&prior, &wrong_identity).is_err());
    }

    #[test]
    fn catalogue_source_references_fail_closed_on_bad_types_urls_and_duplicates() {
        let mut catalog = synthetic_catalogue();
        catalog["style_catalog"][1]["reference_sources"] =
            serde_json::json!(["https://example.org/a", 17]);
        assert!(CatalogueVocabulary::from_catalogue_json(&catalog.to_string()).is_err());

        let mut non_https = synthetic_catalogue();
        non_https["style_catalog"][1]["reference_sources"] =
            serde_json::json!(["http://example.org/a"]);
        assert!(CatalogueVocabulary::from_catalogue_json(&non_https.to_string()).is_err());

        let mut duplicate = synthetic_catalogue();
        duplicate["style_catalog"][1]["reference_sources"] =
            serde_json::json!(["https://example.org/a", "https://example.org/a"]);
        assert!(CatalogueVocabulary::from_catalogue_json(&duplicate.to_string()).is_err());
    }

    #[test]
    fn upstream_synthetic_problem_deserializes_and_validates() {
        let problem = parsed_problem();
        assert!(validate_design_problem(&problem, &vocabulary()).is_ok());
        assert_eq!(problem.input.axis_intent.spatial_typology.state, AxisState::Undecided);
        assert_eq!(problem.claim_ceiling, INTENT_ONLY_CLAIM_CEILING);
    }

    #[test]
    fn illustrative_candidate_set_is_bound_and_non_executable() {
        let problem = parsed_problem();
        let vocabulary = vocabulary();
        let receipt: CandidateSetReceipt =
            serde_json::from_str(SYNTHETIC_CANDIDATE_SET).unwrap();
        assert!(validate_illustrative_candidate_set(&receipt, &problem, &vocabulary).is_ok());

        let mut promoted = receipt.clone();
        promoted.candidates[0].geometry_ref = Some("geometry://not-qualified".to_owned());
        assert!(validate_illustrative_candidate_set(&promoted, &problem, &vocabulary).is_err());

        let mut stale = receipt;
        stale.source_problem_revision = "STALE-REVISION".to_owned();
        assert!(validate_illustrative_candidate_set(&stale, &problem, &vocabulary).is_err());
    }

    #[test]
    fn illustrative_candidates_cannot_invent_or_rewrite_resident_intent() {
        let problem = parsed_problem();
        let vocabulary = vocabulary();
        let receipt: CandidateSetReceipt =
            serde_json::from_str(SYNTHETIC_CANDIDATE_SET).unwrap();

        let mut invented = receipt.clone();
        invented.candidates[0].source_intents.push(CandidateSourceIntent {
            axis:"spatial_typology".to_owned(),
            option_id:"COURTYARD".to_owned(),
            preference:"PREFERRED".to_owned(),
        });
        assert!(validate_illustrative_candidate_set(&invented, &problem, &vocabulary).is_err());

        let mut hidden = receipt.clone();
        hidden.candidates[0].unresolved_intent_axes.clear();
        assert!(validate_illustrative_candidate_set(&hidden, &problem, &vocabulary).is_err());

        let mut rewritten = receipt;
        rewritten.candidates[0].source_intents[0].preference = "CONSIDERED".to_owned();
        assert!(validate_illustrative_candidate_set(&rewritten, &problem, &vocabulary).is_err());
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
    fn future_mapping_and_permission_flags_cannot_open_generation_in_v01() {
        let mut problem = parsed_problem();
        problem.compilation.mapping_state = MappingState::Qualified;
        assert!(validate_design_problem(&problem, &vocabulary()).is_err());
        assert!(!symbolic_candidate_generation_allowed(&problem));

        let mut geometry = parsed_problem();
        geometry.generation_permissions.geometry_generation =
            GeometryGenerationPermission::AllowedWithQualifiedGeometryProvider;
        assert!(validate_design_problem(&geometry, &vocabulary()).is_err());

        let mut physics = parsed_problem();
        physics.generation_permissions.physical_evaluation =
            PhysicalEvaluationPermission::AllowedWithQualifiedSolverInputs;
        assert!(validate_design_problem(&physics, &vocabulary()).is_err());
    }

    #[test]
    fn git_blob_digest_matches_git_object_format() {
        assert_eq!(git_blob_sha1(b"test\n"), "9daeafb9864cf43055ae93beb0afd6c7d144bfa4");
    }

    #[test]
    fn exact_source_bindings_require_lowercase_git_sha_shapes() {
        let mut problem = parsed_problem();
        problem.source_bindings.catalogue_schema_blob_sha = "NOT_A_GIT_SHA".to_owned();
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
    fn floorplanbench_five_objectives_normalize_to_minimization_reference() {
        let raw = OracleEvaluation {
            candidate_id:"floorplan-task-001".to_owned(),
            gate_state:GateState::Pass,
            objectives:vec![
                OracleObjective {objective_id:"efficiency".to_owned(),direction:ObjectiveDirection::Maximize,value:0.75},
                OracleObjective {objective_id:"circulation".to_owned(),direction:ObjectiveDirection::Minimize,value:5.0},
                OracleObjective {objective_id:"daylight".to_owned(),direction:ObjectiveDirection::Maximize,value:0.8},
                OracleObjective {objective_id:"privacy".to_owned(),direction:ObjectiveDirection::Maximize,value:0.6},
                OracleObjective {objective_id:"carbon".to_owned(),direction:ObjectiveDirection::Minimize,value:400.0},
            ],
        };
        let rules = vec![
            ObjectiveNormalization {objective_id:"efficiency".to_owned(),direction:ObjectiveDirection::Maximize,lower_bound:0.0,upper_bound:1.0},
            ObjectiveNormalization {objective_id:"circulation".to_owned(),direction:ObjectiveDirection::Minimize,lower_bound:0.0,upper_bound:10.0},
            ObjectiveNormalization {objective_id:"daylight".to_owned(),direction:ObjectiveDirection::Maximize,lower_bound:0.0,upper_bound:1.0},
            ObjectiveNormalization {objective_id:"privacy".to_owned(),direction:ObjectiveDirection::Maximize,lower_bound:0.0,upper_bound:1.0},
            ObjectiveNormalization {objective_id:"carbon".to_owned(),direction:ObjectiveDirection::Minimize,lower_bound:300.0,upper_bound:800.0},
        ];
        let values = normalize_evaluations(&[raw],&rules).unwrap();
        let vector: Vec<f64> = values[0].objectives.iter().map(|objective| objective.value).collect();
        assert_eq!(vector, vec![0.25,0.5,0.2,0.4,0.2]);
        assert!(values[0].objectives.iter().all(|objective| objective.direction == ObjectiveDirection::Minimize));
        let expected = [0.7501,0.5001,0.8001,0.6001,0.8001].iter().product::<f64>();
        assert!((exact_hypervolume(&values,&[1.0001;5]).unwrap()-expected).abs()<1e-12);
    }

    #[test]
    fn objective_normalization_rejects_bad_bounds_order_and_out_of_range_values() {
        let raw = OracleEvaluation {
            candidate_id:"candidate".to_owned(),
            gate_state:GateState::Pass,
            objectives:vec![
                OracleObjective {objective_id:"energy".to_owned(),direction:ObjectiveDirection::Minimize,value:11.0},
                OracleObjective {objective_id:"comfort".to_owned(),direction:ObjectiveDirection::Maximize,value:8.0},
            ],
        };
        let valid = vec![
            ObjectiveNormalization {objective_id:"energy".to_owned(),direction:ObjectiveDirection::Minimize,lower_bound:0.0,upper_bound:10.0},
            ObjectiveNormalization {objective_id:"comfort".to_owned(),direction:ObjectiveDirection::Maximize,lower_bound:0.0,upper_bound:10.0},
        ];
        assert!(normalize_evaluations(&[raw.clone()],&valid).is_err());
        let mut wrong_order = valid.clone();
        wrong_order.swap(0,1);
        assert!(normalize_evaluations(&[raw.clone()],&wrong_order).is_err());
        let mut invalid_bounds = valid;
        invalid_bounds[0].upper_bound=0.0;
        assert!(normalize_evaluations(&[raw],&invalid_bounds).is_err());
    }

    #[test]
    fn exact_hypervolume_matches_known_two_dimensional_union() {
        let evaluations = vec![
            OracleEvaluation {
                candidate_id:"a".to_owned(),
                gate_state:GateState::Pass,
                objectives:vec![
                    OracleObjective {objective_id:"energy".to_owned(),direction:ObjectiveDirection::Minimize,value:1.0},
                    OracleObjective {objective_id:"water".to_owned(),direction:ObjectiveDirection::Minimize,value:3.0},
                ],
            },
            OracleEvaluation {
                candidate_id:"b".to_owned(),
                gate_state:GateState::Pass,
                objectives:vec![
                    OracleObjective {objective_id:"energy".to_owned(),direction:ObjectiveDirection::Minimize,value:2.0},
                    OracleObjective {objective_id:"water".to_owned(),direction:ObjectiveDirection::Minimize,value:2.0},
                ],
            },
            OracleEvaluation {
                candidate_id:"invalid".to_owned(),
                gate_state:GateState::Unknown,
                objectives:vec![
                    OracleObjective {objective_id:"energy".to_owned(),direction:ObjectiveDirection::Minimize,value:0.0},
                    OracleObjective {objective_id:"water".to_owned(),direction:ObjectiveDirection::Minimize,value:0.0},
                ],
            },
        ];
        assert_eq!(exact_hypervolume(&evaluations,&[4.0,4.0]).unwrap(),5.0);
    }

    #[test]
    fn exact_hypervolume_supports_mixed_objective_directions() {
        let evaluations = vec![
            OracleEvaluation {
                candidate_id:"mixed".to_owned(),
                gate_state:GateState::Pass,
                objectives:vec![
                    OracleObjective {objective_id:"energy".to_owned(),direction:ObjectiveDirection::Minimize,value:2.0},
                    OracleObjective {objective_id:"comfort".to_owned(),direction:ObjectiveDirection::Maximize,value:8.0},
                ],
            }
        ];
        assert_eq!(exact_hypervolume(&evaluations,&[5.0,4.0]).unwrap(),12.0);
    }

    #[test]
    fn exact_hypervolume_supports_five_objectives_for_floorplan_benchmarks() {
        let evaluation = OracleEvaluation {
            candidate_id:"five-objective".to_owned(),
            gate_state:GateState::Pass,
            objectives:vec![
                OracleObjective {objective_id:"space".to_owned(),direction:ObjectiveDirection::Minimize,value:1.0},
                OracleObjective {objective_id:"circulation".to_owned(),direction:ObjectiveDirection::Minimize,value:2.0},
                OracleObjective {objective_id:"daylight".to_owned(),direction:ObjectiveDirection::Maximize,value:5.0},
                OracleObjective {objective_id:"privacy".to_owned(),direction:ObjectiveDirection::Maximize,value:6.0},
                OracleObjective {objective_id:"carbon".to_owned(),direction:ObjectiveDirection::Minimize,value:5.0},
            ],
        };
        assert_eq!(
            exact_hypervolume(&[evaluation], &[3.0,4.0,3.0,4.0,7.0]).unwrap(),
            32.0
        );
    }

    #[test]
    fn exact_hypervolume_rejects_bad_reference_points_and_dimensions() {
        let evaluation = OracleEvaluation {
            candidate_id:"single".to_owned(),
            gate_state:GateState::Pass,
            objectives:vec![
                OracleObjective {objective_id:"energy".to_owned(),direction:ObjectiveDirection::Minimize,value:2.0},
                OracleObjective {objective_id:"water".to_owned(),direction:ObjectiveDirection::Minimize,value:3.0},
            ],
        };
        assert!(exact_hypervolume(&[evaluation.clone()],&[]).is_err());
        assert!(exact_hypervolume(&[evaluation.clone()],&[5.0]).is_err());
        assert!(exact_hypervolume(&[evaluation.clone()],&[1.0,4.0]).is_err());
        assert!(exact_hypervolume(&[evaluation],&[f64::INFINITY,4.0]).is_err());
    }

    #[test]
    fn exact_oracle_rejects_pass_receipt_with_non_finite_objective() {
        let malformed = OracleEvaluation {
            candidate_id:"bad-pass".to_owned(),
            gate_state:GateState::Pass,
            objectives:vec![
                OracleObjective {
                    objective_id:"energy".to_owned(),
                    direction:ObjectiveDirection::Minimize,
                    value:f64::NAN,
                }
            ],
        };
        assert!(exact_pareto_front(&[malformed]).is_err());
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
