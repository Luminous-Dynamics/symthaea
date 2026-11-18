//! Typed resident-intent and candidate-lineage contracts for built-environment design.
//!
//! This crate does not generate geometry or evaluate building physics. It
//! validates source-bound five-axis intent, separates authority from
//! preferences, gates mappings, derives deterministic IDs and prevents
//! unknown or unqualified candidates entering a physical Pareto set.

use serde::{Deserialize, Serialize};
use serde_json::{Value};
use std::collections::{BTreeMap, BTreeSet};

pub const DESIGN_PROBLEM_SCHEMA_VERSION: &str = "0.1.0";
pub const CANDIDATE_SET_SCHEMA_VERSION: &str = "0.1.0";

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum AxisState { Undecided, IntentSelected, IntentOnlyNotEvaluated, ConflictUnresolved }

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ExperienceState { Undecided, OptionalIdeasOnly, ConflictUnresolved }

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Preference { Preferred, Considered, Avoided, Undecided }

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ExpressionRole { Primary, Secondary, Accent, ReferenceOnly, Avoid, Acceptable, Undecided }

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
    #[serde(rename = "role")]
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
    ResidentNonNegotiable, JurisdictionalRule, ProfessionalReviewRequirement, SystemInvariant,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum VerificationState { ExplicitUnverified, SourceBoundUnreviewed, OwnerValidated, Unknown }

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
    #[serde(rename = "SPATIAL_TYPOLOGY")] SpatialTypology,
    #[serde(rename = "ARCHITECTURAL_EXPRESSION")] ArchitecturalExpression,
    #[serde(rename = "MATERIAL_AND_TECTONIC_INTENT")] MaterialTectonicIntent,
    #[serde(rename = "ENVIRONMENTAL_STRATEGY_INTENT")] EnvironmentalStrategyIntent,
    #[serde(rename = "EXPERIENCE_AND_PROGRAMME")] ExperienceProgramme,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum MappingReviewState { Draft, OwnerReview, Approved, Rejected }

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
pub enum MappingState { NoOwnerReviewedMapping, Partial, Qualified }

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum CompilationState {
    IntentParsedNotCompiledToGeometry, CompiledToSymbolicDesignQuestion, ReadyForCandidateGeneration,
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
pub enum SymbolicGenerationPermission { RequiresReviewedMappingRules, AllowedWithReviewedMappings, Blocked }

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum GeometryGenerationPermission {
    BlockedPendingSymtropyGeometryProviderAndOwnerReview, AllowedWithQualifiedGeometryProvider,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum PhysicalEvaluationPermission {
    BlockedPendingExactRealizedGeometryAndQualifiedSolverInputs, AllowedWithQualifiedSolverInputs,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum CertificationPermission { Prohibited }

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct GenerationPermissions {
    pub symbolic_candidate_generation: SymbolicGenerationPermission,
    pub geometry_generation: GeometryGenerationPermission,
    pub physical_evaluation: PhysicalEvaluationPermission,
    pub certification: CertificationPermission,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum DecisionState { NoResidentChoiceOrDesignApprovalRecorded, PartialResidentIntentRecorded }

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
    /// Load the shared v0.2 identifiers from caller-pinned catalogue bytes.
    /// This crate intentionally contains no second copy of the UI vocabulary.
    pub fn from_catalogue_json(source: &str) -> Result<Self, String> {
        let value: Value = serde_json::from_str(source)
            .map_err(|error| format!("catalogue JSON is invalid: {error}"))?;
        let mut vocabulary = Self {
            catalogue_revision: required_string(&value, "catalog_revision")?,
            ..Self::default()
        };
        let styles = value.get("style_catalog").and_then(Value::as_array)
            .ok_or_else(|| "style_catalog is missing or not an array".to_owned())?;
        for style in styles {
            let id = required_string(style, "style_id")?;
            let sources = style.get("reference_sources").and_then(Value::as_array)
                .ok_or_else(|| format!("style {id} has no reference_sources array"))?
                .iter().filter_map(Value::as_str).map(str::to_owned).collect();
            if vocabulary.style_sources.insert(id.clone(), sources).is_some() {
                return Err(format!("duplicate style identifier: {id}"));
            }
        }
        vocabulary.spatial_typology_ids = collect_ids(&value, "spatial_typology_catalog", "option_id")?;
        vocabulary.material_tectonic_ids = collect_ids(&value, "material_tectonic_intent_catalog", "option_id")?;
        vocabulary.environmental_strategy_ids = collect_ids(&value, "environmental_strategy_intent_catalog", "option_id")?;
        vocabulary.experience_package_ids = collect_ids(&value, "experience_catalog", "package_id")?;
        Ok(vocabulary)
    }

    fn contains_source_option(&self, axis: &DesignAxis, option_id: &str) -> bool {
        match axis {
            DesignAxis::SpatialTypology => self.spatial_typology_ids.contains(option_id),
            DesignAxis::ArchitecturalExpression => self.style_sources.contains_key(option_id),
            DesignAxis::MaterialTectonicIntent => self.material_tectonic_ids.contains(option_id),
            DesignAxis::EnvironmentalStrategyIntent => self.environmental_strategy_ids.contains(option_id),
            DesignAxis::ExperienceProgramme => self.experience_package_ids.contains(option_id),
        }
    }
}

fn required_string(value: &Value, key: &str) -> Result<String, String> {
    value.get(key).and_then(Value::as_str).filter(|s| !s.trim().is_empty())
        .map(str::to_owned).ok_or_else(|| format!("required non-empty string {key} is missing"))
}

fn collect_ids(value: &Value, catalogue_key: &str, id_key: &str) -> Result<BTreeSet<String>, String> {
    let items = value.get(catalogue_key).and_then(Value::as_array)
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
    value.len() == 40 && value.bytes().all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
}

fn validate_intent_axis(name: &str, axis: &IntentAxis, allowed: &BTreeSet<String>, errors: &mut Vec<String>) {
    if axis.entries.is_empty() && axis.state != AxisState::Undecided {
        errors.push(format!("{name}: empty axis must remain UNDECIDED"));
    }
    if !axis.entries.is_empty() && axis.state == AxisState::Undecided {
        errors.push(format!("{name}: non-empty axis cannot be UNDECIDED"));
    }
    let mut seen = BTreeSet::new();
    let mut preferred = 0usize;
    for entry in &axis.entries {
        if entry.option_id.trim().is_empty() || !allowed.contains(&entry.option_id) {
            errors.push(format!("{name}: unknown option ID {}", entry.option_id));
        }
        if !seen.insert(entry.option_id.clone()) {
            errors.push(format!("{name}: duplicate option ID {}", entry.option_id));
        }
        if entry.preference == Preference::Preferred { preferred += 1; }
    }
    if name == "spatial_typology" && preferred > 1 && axis.state != AxisState::ConflictUnresolved {
        errors.push("spatial_typology: multiple preferred entries must remain CONFLICT_UNRESOLVED".to_owned());
    }
    if name == "spatial_typology" && axis.state == AxisState::ConflictUnresolved && preferred < 2 {
        errors.push("spatial_typology: conflict state requires multiple preferred entries".to_owned());
    }
}

fn validate_expression_axis(axis: &ExpressionAxis, vocabulary: &CatalogueVocabulary, errors: &mut Vec<String>) {
    if axis.entries.is_empty() && axis.state != AxisState::Undecided {
        errors.push("architectural_expression: empty axis must remain UNDECIDED".to_owned());
    }
    if !axis.entries.is_empty() && axis.state == AxisState::Undecided {
        errors.push("architectural_expression: non-empty axis cannot be UNDECIDED".to_owned());
    }
    let mut seen = BTreeSet::new();
    let mut primary = 0usize;
    for entry in &axis.entries {
        if !vocabulary.style_sources.contains_key(&entry.option_id) {
            errors.push(format!("architectural_expression: unknown style ID {}", entry.option_id));
        }
        if !seen.insert(entry.option_id.clone()) {
            errors.push(format!("architectural_expression: duplicate style ID {}", entry.option_id));
        }
        if !entry.resident_selected {
            errors.push(format!("architectural_expression: {} has no resident selection", entry.option_id));
        }
        if entry.role == ExpressionRole::Primary { primary += 1; }
        if entry.role == ExpressionRole::ReferenceOnly {
            let allowed = vocabulary.style_sources.get(&entry.option_id);
            if entry.source_refs.is_empty() {
                errors.push(format!("architectural_expression: reference {} has no source", entry.option_id));
            } else if let Some(allowed_sources) = allowed {
                for source in &entry.source_refs {
                    if !allowed_sources.contains(source) {
                        errors.push(format!("architectural_expression: source {source} is not catalogue-bound for {}", entry.option_id));
                    }
                }
            }
        }
        if entry.blend_weight_pct.is_some_and(|weight| weight > 100) {
            errors.push(format!("architectural_expression: blend weight for {} exceeds 100", entry.option_id));
        }
    }
    if primary > 1 {
        errors.push("architectural_expression: more than one PRIMARY role".to_owned());
    }
}

fn validate_experience_axis(axis: &ExperienceAxis, allowed: &BTreeSet<String>, errors: &mut Vec<String>) {
    if axis.entries.is_empty() && axis.state != ExperienceState::Undecided {
        errors.push("experience_programme: empty axis must remain UNDECIDED".to_owned());
    }
    if !axis.entries.is_empty() && axis.state == ExperienceState::Undecided {
        errors.push("experience_programme: non-empty axis cannot be UNDECIDED".to_owned());
    }
    let mut seen = BTreeSet::new();
    for entry in &axis.entries {
        if !allowed.contains(&entry.option_id) {
            errors.push(format!("experience_programme: unknown package ID {}", entry.option_id));
        }
        if !seen.insert(entry.option_id.clone()) {
            errors.push(format!("experience_programme: duplicate package ID {}", entry.option_id));
        }
    }
}

fn validate_requirements(requirements: &[Requirement], seen: &mut BTreeSet<String>, errors: &mut Vec<String>) {
    for requirement in requirements {
        if requirement.requirement_id.trim().is_empty()
            || requirement.statement.trim().is_empty()
            || !seen.insert(requirement.requirement_id.clone())
        {
            errors.push(format!("requirement has an empty or duplicate ID/statement: {}", requirement.requirement_id));
        }
        if matches!(requirement.source_class, RequirementSourceClass::JurisdictionalRule | RequirementSourceClass::ProfessionalReviewRequirement)
            && requirement.source_ref.as_deref().is_none_or(str::is_empty)
        {
            errors.push(format!("authority requirement {} has no source reference", requirement.requirement_id));
        }
        if requirement.verification_state == VerificationState::OwnerValidated
            && requirement.source_ref.as_deref().is_none_or(str::is_empty)
        {
            errors.push(format!("owner-validated requirement {} lacks an evidence source", requirement.requirement_id));
        }
    }
}

fn validate_compilation(compilation: &Compilation, input: &ResidentIntent, vocabulary: &CatalogueVocabulary, errors: &mut Vec<String>) {
    let mut requirement_ids = BTreeSet::new();
    validate_requirements(&compilation.hard_requirements, &mut requirement_ids, errors);
    validate_requirements(&compilation.authority_constraints, &mut requirement_ids, errors);

    let mut mapping_ids = BTreeSet::new();
    let mut detailed_ids = Vec::new();
    let mut approved_pairs = BTreeSet::new();
    for rule in &compilation.mapping_rules {
        if rule.mapping_rule_id.trim().is_empty() || !mapping_ids.insert(rule.mapping_rule_id.clone()) {
            errors.push(format!("missing or duplicate mapping rule ID: {}", rule.mapping_rule_id));
        }
        detailed_ids.push(rule.mapping_rule_id.clone());
        if !vocabulary.contains_source_option(&rule.source_axis, &rule.source_option_id) {
            errors.push(format!("mapping {} references unknown source option {}", rule.mapping_rule_id, rule.source_option_id));
        }
        if rule.target_parameter.trim().is_empty() || rule.mapping_version.trim().is_empty() || rule.scope.trim().is_empty() {
            errors.push(format!("mapping rule {} lacks target, version or scope", rule.mapping_rule_id));
        }
        if rule.review_state == MappingReviewState::Approved {
            if rule.source_ref.as_deref().is_none_or(str::is_empty) || rule.owner_reviewer.as_deref().is_none_or(str::is_empty) {
                errors.push(format!("approved mapping rule {} lacks source/reviewer", rule.mapping_rule_id));
            } else {
                approved_pairs.insert((rule.source_axis.clone(), rule.source_option_id.clone()));
            }
        }
    }
    detailed_ids.sort();
    let mut declared_ids = compilation.mapping_rule_ids.clone();
    declared_ids.sort();
    if detailed_ids != declared_ids {
        errors.push("mapping_rule_ids does not match detailed mapping_rules".to_owned());
    }
    match compilation.mapping_state {
        MappingState::NoOwnerReviewedMapping if compilation.mapping_rules.iter().any(|r| r.review_state == MappingReviewState::Approved) => {
            errors.push("NO_OWNER_REVIEWED_MAPPING cannot contain approved mapping rules".to_owned());
        }
        MappingState::Partial if compilation.mapping_rules.is_empty() || compilation.mapping_rules.iter().all(|r| r.review_state == MappingReviewState::Approved) => {
            errors.push("PARTIAL mapping state needs covered and uncovered mapping scope".to_owned());
        }
        MappingState::Qualified => {
            if compilation.mapping_rules.is_empty() || compilation.mapping_rules.iter().any(|r| r.review_state != MappingReviewState::Approved) {
                errors.push("QUALIFIED mapping state requires only owner-approved mapping rules".to_owned());
            }
            let axes: Vec<(DesignAxis, Vec<String>)> = vec![
                (DesignAxis::SpatialTypology, input.axis_intent.spatial_typology.entries.iter().map(|e| e.option_id.clone()).collect()),
                (DesignAxis::ArchitecturalExpression, input.axis_intent.architectural_expression.entries.iter().map(|e| e.option_id.clone()).collect()),
                (DesignAxis::MaterialTectonicIntent, input.axis_intent.material_tectonic_intent.entries.iter().map(|e| e.option_id.clone()).collect()),
                (DesignAxis::EnvironmentalStrategyIntent, input.axis_intent.environmental_strategy_intent.entries.iter().map(|e| e.option_id.clone()).collect()),
                (DesignAxis::ExperienceProgramme, input.axis_intent.experience_programme.entries.iter().map(|e| e.option_id.clone()).collect()),
            ];
            for (axis, ids) in axes {
                for id in ids {
                    if !approved_pairs.contains(&(axis.clone(), id.clone())) {
                        errors.push(format!("QUALIFIED mapping state lacks approved mapping for {axis:?}:{id}"));
                    }
                }
            }
            if !compilation.unresolved_conflicts.is_empty() {
                errors.push("QUALIFIED mapping state cannot retain unresolved conflicts".to_owned());
            }
        }
        _ => {}
    }
}

pub fn validate_design_problem(problem: &DesignProblem, vocabulary: &CatalogueVocabulary) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();
    if problem.schema_version != DESIGN_PROBLEM_SCHEMA_VERSION { errors.push("unsupported design-problem schema version".to_owned()); }
    if problem.problem_id.trim().is_empty() { errors.push("problem_id is required".to_owned()); }
    if problem.source_bindings.catalogue_revision != vocabulary.catalogue_revision { errors.push("catalogue revision does not match loaded vocabulary".to_owned()); }
    if !problem.source_bindings.sol_atlas_pr.starts_with("https://github.com/") { errors.push("Sol-Atlas source must be an HTTPS GitHub PR URL".to_owned()); }
    for (name, digest) in [
        ("sol_atlas_pr_head_sha", problem.source_bindings.sol_atlas_pr_head_sha.as_str()),
        ("catalogue_blob_sha", problem.source_bindings.catalogue_blob_sha.as_str()),
        ("grammar_schema_blob_sha", problem.source_bindings.grammar_schema_blob_sha.as_str()),
        ("conformance_corpus_blob_sha", problem.source_bindings.conformance_corpus_blob_sha.as_str()),
        ("conformance_corpus_schema_blob_sha", problem.source_bindings.conformance_corpus_schema_blob_sha.as_str()),
    ] {
        if !is_sha1(digest) { errors.push(format!("{name} is not a lowercase 40-character Git object SHA")); }
    }
    let input = &problem.input;
    if input.grammar_id.trim().is_empty() || input.design_revision_ref.trim().is_empty() || input.source_fixture_ref.trim().is_empty() {
        errors.push("grammar_id, design_revision_ref and source_fixture_ref are required".to_owned());
    }
    if input.resident_statement.len() > 3000 { errors.push("resident_statement exceeds the v0.2 contract limit".to_owned()); }
    if !["UNKNOWN", "PARTIAL", "SOURCE_BOUND_NOT_QUALIFIED"].contains(&input.context_state.as_str()) {
        errors.push("context_state is unsupported or over-claimed".to_owned());
    }
    let axes = &input.axis_intent;
    validate_intent_axis("spatial_typology", &axes.spatial_typology, &vocabulary.spatial_typology_ids, &mut errors);
    validate_expression_axis(&axes.architectural_expression, vocabulary, &mut errors);
    validate_intent_axis("material_tectonic_intent", &axes.material_tectonic_intent, &vocabulary.material_tectonic_ids, &mut errors);
    validate_intent_axis("environmental_strategy_intent", &axes.environmental_strategy_intent, &vocabulary.environmental_strategy_ids, &mut errors);
    validate_experience_axis(&axes.experience_programme, &vocabulary.experience_package_ids, &mut errors);
    validate_compilation(&problem.compilation, input, vocabulary, &mut errors);

    if problem.compilation.state == CompilationState::ReadyForCandidateGeneration && problem.compilation.mapping_state != MappingState::Qualified {
        errors.push("READY_FOR_CANDIDATE_GENERATION requires QUALIFIED mapping state".to_owned());
    }
    if problem.generation_permissions.certification != CertificationPermission::Prohibited {
        errors.push("this adapter must never issue certification".to_owned());
    }
    if problem.generation_permissions.geometry_generation == GeometryGenerationPermission::AllowedWithQualifiedGeometryProvider
        && problem.compilation.mapping_state != MappingState::Qualified
    {
        errors.push("geometry generation permission requires qualified intent mappings".to_owned());
    }
    if problem.generation_permissions.physical_evaluation == PhysicalEvaluationPermission::AllowedWithQualifiedSolverInputs
        && problem.generation_permissions.geometry_generation != GeometryGenerationPermission::AllowedWithQualifiedGeometryProvider
    {
        errors.push("physical evaluation permission requires qualified geometry permission".to_owned());
    }
    if errors.is_empty() { Ok(()) } else { Err(errors) }
}

pub fn symbolic_candidate_generation_allowed(problem: &DesignProblem) -> bool {
    problem.compilation.mapping_state == MappingState::Qualified
        && problem.compilation.state == CompilationState::ReadyForCandidateGeneration
        && problem.generation_permissions.symbolic_candidate_generation == SymbolicGenerationPermission::AllowedWithReviewedMappings
        && problem.compilation.unresolved_conflicts.is_empty()
        && problem.compilation.mapping_rules.iter().all(|rule| {
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

pub fn geometry_handoff_eligible(problem: &DesignProblem, provider: &GeometryProviderQualification) -> bool {
    symbolic_candidate_generation_allowed(problem)
        && problem.generation_permissions.geometry_generation == GeometryGenerationPermission::AllowedWithQualifiedGeometryProvider
        && provider.qualified && !provider.provider_id.is_empty() && !provider.provider_revision.is_empty()
        && provider.exact_geometry_digest.as_deref().is_some_and(is_sha1)
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ObjectiveValue { pub objective_id: String, pub value: f64, pub unit: String, pub source_receipt_digest: String }

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct PhysicalEvaluationReceipt { pub owner_system: String, pub solver_version: String, pub exact_input_digest: String, pub qualified: bool }

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
pub enum GateState { Pass, Fail, Unknown, Conditional }

pub fn candidate_eligible_for_physical_pareto(candidate: &Candidate) -> bool {
    candidate.hard_gate_state == GateState::Pass
        && candidate.geometry_ref.as_deref().is_some_and(|r| !r.is_empty())
        && !candidate.physical_evaluations.is_empty()
        && candidate.physical_evaluations.iter().all(|r| r.qualified && !r.owner_system.is_empty() && !r.solver_version.is_empty() && is_sha1(&r.exact_input_digest))
        && !candidate.objectives.is_empty()
        && candidate.objectives.iter().all(|o| o.value.is_finite() && !o.objective_id.is_empty() && !o.unit.is_empty() && is_sha1(&o.source_receipt_digest))
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReplayReceipt { pub problem_digest: String, pub generator_build_digest: String, pub seed: u64, pub ordered_candidate_ids: Vec<String> }

pub fn design_problem_digest(problem: &DesignProblem) -> Result<String, serde_json::Error> {
    Ok(blake3::hash(&serde_json::to_vec(problem)?).to_hex().to_string())
}

pub fn derive_candidate_id(problem_digest: &str, generator_build_digest: &str, seed: u64, ordinal: u32, parameter_deltas: &BTreeMap<String, Value>) -> Result<String, serde_json::Error> {
    let bytes = serde_json::to_vec(&(problem_digest, generator_build_digest, seed, ordinal, parameter_deltas))?;
    Ok(format!("cand-{}", blake3::hash(&bytes).to_hex()))
}

pub fn replay_matches(expected: &ReplayReceipt, actual: &ReplayReceipt) -> bool { expected == actual }

#[cfg(test)]
mod tests {
    use super::*;
    const SYNTHETIC_PROBLEM: &str = include_str!("../../../docs/research/built-environment/HOME_EXPRESSION_DESIGN_PROBLEM_SYNTHETIC_V0.1.json");

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

    #[test]
    fn upstream_synthetic_problem_deserializes_and_validates() {
        let vocabulary = CatalogueVocabulary::from_catalogue_json(&synthetic_catalogue().to_string()).unwrap();
        let problem: DesignProblem = serde_json::from_str(SYNTHETIC_PROBLEM).unwrap();
        assert!(validate_design_problem(&problem, &vocabulary).is_ok());
        assert_eq!(problem.input.axis_intent.spatial_typology.state, AxisState::Undecided);
        assert!(!symbolic_candidate_generation_allowed(&problem));
    }

    #[test]
    fn unknown_source_option_is_rejected() {
        let vocabulary = CatalogueVocabulary::from_catalogue_json(&synthetic_catalogue().to_string()).unwrap();
        let mut problem: DesignProblem = serde_json::from_str(SYNTHETIC_PROBLEM).unwrap();
        problem.input.axis_intent.material_tectonic_intent.entries[0].option_id = "UNLISTED_COMPOSITE".to_owned();
        assert!(validate_design_problem(&problem, &vocabulary).is_err());
    }

    #[test]
    fn reference_only_style_requires_catalogue_bound_source() {
        let vocabulary = CatalogueVocabulary::from_catalogue_json(&synthetic_catalogue().to_string()).unwrap();
        let mut problem: DesignProblem = serde_json::from_str(SYNTHETIC_PROBLEM).unwrap();
        problem.input.axis_intent.architectural_expression.entries[1].source_refs = vec!["https://example.org/unbound".to_owned()];
        assert!(validate_design_problem(&problem, &vocabulary).is_err());
    }

    #[test]
    fn preferred_typology_conflict_is_not_collapsed() {
        let vocabulary = CatalogueVocabulary::from_catalogue_json(&synthetic_catalogue().to_string()).unwrap();
        let mut problem: DesignProblem = serde_json::from_str(SYNTHETIC_PROBLEM).unwrap();
        problem.input.axis_intent.spatial_typology = IntentAxis {
            state: AxisState::ConflictUnresolved,
            entries: vec![
                OptionPreferenceEntry { option_id:"COURTYARD".to_owned(), preference:Preference::Preferred },
                OptionPreferenceEntry { option_id:"ROW_TERRACE".to_owned(), preference:Preference::Preferred }
            ]
        };
        assert!(validate_design_problem(&problem, &vocabulary).is_ok());
        problem.input.axis_intent.spatial_typology.state = AxisState::IntentOnlyNotEvaluated;
        assert!(validate_design_problem(&problem, &vocabulary).is_err());
    }

    #[test]
    fn no_owner_reviewed_mapping_blocks_generation() {
        let vocabulary = CatalogueVocabulary::from_catalogue_json(&synthetic_catalogue().to_string()).unwrap();
        let problem: DesignProblem = serde_json::from_str(SYNTHETIC_PROBLEM).unwrap();
        assert!(!symbolic_candidate_generation_allowed(&problem));
        assert!(problem.compilation.mapping_rules.is_empty());
        assert_eq!(problem.generation_permissions.certification, CertificationPermission::Prohibited);
    }

    #[test]
    fn mapping_rules_must_match_catalogue_and_owner_approval() {
        let vocabulary = CatalogueVocabulary::from_catalogue_json(&synthetic_catalogue().to_string()).unwrap();
        let mut problem: DesignProblem = serde_json::from_str(SYNTHETIC_PROBLEM).unwrap();
        let rule = MappingRule {
            mapping_rule_id:"MAP-001".to_owned(),
            source_axis:DesignAxis::MaterialTectonicIntent,
            source_option_id:"STONE_MASONRY".to_owned(),
            target_parameter:"assembly.material_family".to_owned(),
            mapping_version:"0.1.0".to_owned(),
            scope:"PAVILION_ONLY".to_owned(),
            source_ref:Some("https://example.org/reviewed-material-spec".to_owned()),
            owner_reviewer:Some("domain-owner".to_owned()),
            review_state:MappingReviewState::OwnerReview
        };
        problem.compilation.mapping_state = MappingState::Partial;
        problem.compilation.mapping_rule_ids = vec!["MAP-001".to_owned()];
        problem.compilation.mapping_rules = vec![rule.clone()];
        assert!(validate_design_problem(&problem, &vocabulary).is_ok());
        assert!(!symbolic_candidate_generation_allowed(&problem));
        let mut invalid = rule;
        invalid.source_option_id = "UNLISTED_MATERIAL".to_owned();
        problem.compilation.mapping_rules = vec![invalid];
        assert!(validate_design_problem(&problem, &vocabulary).is_err());
    }

    #[test]
    fn unknown_or_failed_hard_gates_never_enter_physical_pareto() {
        let mut candidate = Candidate {
            candidate_id:"cand-1".to_owned(), parent_candidate_ids:vec![], source_intent_refs:vec![],
            parameter_deltas:BTreeMap::new(), unresolved_requirements:vec![],
            hard_gate_state:GateState::Unknown, geometry_ref:Some("geometry://exact".to_owned()),
            physical_evaluations:vec![PhysicalEvaluationReceipt {owner_system:"symtropy".to_owned(),solver_version:"v1".to_owned(),exact_input_digest:"a".repeat(40),qualified:true}],
            objectives:vec![ObjectiveValue {objective_id:"energy".to_owned(),value:10.0,unit:"kWh".to_owned(),source_receipt_digest:"b".repeat(40)}]
        };
        assert!(!candidate_eligible_for_physical_pareto(&candidate));
        candidate.hard_gate_state = GateState::Pass;
        assert!(candidate_eligible_for_physical_pareto(&candidate));
        candidate.objectives[0].value = f64::NAN;
        assert!(!candidate_eligible_for_physical_pareto(&candidate));
    }

    #[test]
    fn problem_digest_candidate_id_and_replay_are_deterministic() {
        let vocabulary = CatalogueVocabulary::from_catalogue_json(&synthetic_catalogue().to_string()).unwrap();
        let problem: DesignProblem = serde_json::from_str(SYNTHETIC_PROBLEM).unwrap();
        assert!(validate_design_problem(&problem, &vocabulary).is_ok());
        let digest = design_problem_digest(&problem).unwrap();
        assert_eq!(digest, design_problem_digest(&problem).unwrap());
        let deltas = BTreeMap::from([("intent.shading".to_owned(), Value::String("EXTERNAL".to_owned()))]);
        assert_eq!(derive_candidate_id(&digest,"generator-build",7,0,&deltas).unwrap(), derive_candidate_id(&digest,"generator-build",7,0,&deltas).unwrap());
        let receipt = ReplayReceipt {problem_digest:digest,generator_build_digest:"b".repeat(64),seed:7,ordered_candidate_ids:vec!["cand-1".to_owned(),"cand-2".to_owned()]};
        let mut changed = receipt.clone();
        changed.ordered_candidate_ids.reverse();
        assert!(replay_matches(&receipt,&receipt));
        assert!(!replay_matches(&receipt,&changed));
    }
}
