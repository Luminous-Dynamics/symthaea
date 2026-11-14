use anyhow::{bail, Context, Result};
use serde::Deserialize;
use std::collections::HashSet;
use std::path::Path;

use crate::plasma_claim_scope::{decide, Disposition as ClaimDisposition, ScopeEvidence};

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Category {
    id: String,
    label: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CategoryFraction {
    category_id: String,
    fraction: f64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SpecimenClassification {
    specimen_id: String,
    cluster_id: String,
    image_sha256: String,
    observation_mode: String,
    observer_refs: Vec<String>,
    blinding_status: String,
    fractions: Vec<CategoryFraction>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CohortReceipt {
    cohort_id: String,
    category_ontology_id: String,
    category_ontology_version: String,
    independent_unit: String,
    grouping_scheme: String,
    measurement_basis: String,
    fraction_protocol_id: String,
    fraction_protocol_sha256: String,
    observer_reliability_sha256: String,
    unresolved_disagreement_count: usize,
    specimens: Vec<SpecimenClassification>,
    unclassifiable_specimen_ids: Vec<String>,
    classification_receipt: ScopeEvidence,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CategoryMargin {
    category_id: String,
    lower: f64,
    upper: f64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AnalysisPlan {
    plan_id: String,
    plan_sha256: String,
    analysis_profile: String,
    analysis_method_ref: String,
    frozen_at: i64,
    first_outcome_available_at: i64,
    changed_after_outcome: bool,
    alpha: f64,
    confidence_level: f64,
    zero_policy: String,
    pseudocount: Option<f64>,
    joint_distance_margin: f64,
    category_margins: Vec<CategoryMargin>,
    minimum_baseline_clusters: usize,
    minimum_candidate_clusters: usize,
    minimum_observer_agreement: f64,
    agreement_policy_ref: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CategoryEffect {
    category_id: String,
    lower: f64,
    upper: f64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AnalysisResult {
    result_id: String,
    result_sha256: String,
    plan_sha256: String,
    vector_manifest_sha256: String,
    analysis_profile: String,
    baseline_specimen_count: usize,
    candidate_specimen_count: usize,
    baseline_cluster_count: usize,
    candidate_cluster_count: usize,
    observer_agreement: f64,
    alpha: f64,
    confidence_level: f64,
    category_effects: Vec<CategoryEffect>,
    joint_distance_lower: f64,
    joint_distance_upper: f64,
    outcome: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FailureCompositionReceipt {
    receipt_id: String,
    category_ontology_id: String,
    category_ontology_version: String,
    categories: Vec<Category>,
    measurement_tolerance: f64,
    baseline: CohortReceipt,
    candidate: CohortReceipt,
    analysis_plan: AnalysisPlan,
    analysis_result: AnalysisResult,
    vector_manifest_sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    id: String,
    receipt: FailureCompositionReceipt,
    expected_disposition: String,
    expected_outcome: String,
    rationale: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Fixture {
    schema_version: String,
    cases: Vec<Case>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Disposition {
    Pass,
    Blocked,
    Rejected,
}

impl Disposition {
    fn as_str(self) -> &'static str {
        match self {
            Self::Pass => "PASS",
            Self::Blocked => "BLOCKED",
            Self::Rejected => "REJECTED",
        }
    }
}

struct Decision {
    disposition: Disposition,
    outcome: &'static str,
    reason: &'static str,
}

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn sha256_shape(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

fn unique_strings(values: &[String]) -> bool {
    let mut seen = HashSet::new();
    values.iter().all(|value| nonempty(value) && seen.insert(value.as_str()))
}

fn specimen_ids(cohort: &CohortReceipt) -> Vec<String> {
    cohort.specimens.iter().map(|sample| sample.specimen_id.clone()).collect()
}

fn cluster_ids(cohort: &CohortReceipt) -> HashSet<String> {
    cohort.specimens.iter().map(|sample| sample.cluster_id.clone()).collect()
}

fn audit_cohort(
    cohort: &CohortReceipt,
    categories: &[Category],
    ontology_id: &str,
    ontology_version: &str,
    tolerance: f64,
) -> std::result::Result<(), Disposition> {
    if !nonempty(&cohort.cohort_id)
        || !nonempty(&cohort.category_ontology_id)
        || !nonempty(&cohort.category_ontology_version)
        || !nonempty(&cohort.fraction_protocol_id)
        || !sha256_shape(&cohort.fraction_protocol_sha256)
        || !sha256_shape(&cohort.observer_reliability_sha256)
    {
        return Err(Disposition::Blocked);
    }
    if !matches!(cohort.independent_unit.as_str(), "material_lot" | "adhesive_batch" | "plasma_head" | "production_day" | "production_line" | "manufacturing_site") {
        return Err(Disposition::Blocked);
    }
    if !matches!(cohort.grouping_scheme.as_str(), "independent_arms" | "paired_by_cluster") {
        return Err(Disposition::Rejected);
    }
    if !matches!(cohort.measurement_basis.as_str(), "visual_area_fraction" | "specimen_count_fraction") {
        return Err(Disposition::Rejected);
    }
    if cohort.category_ontology_id != ontology_id
        || cohort.category_ontology_version != ontology_version
    {
        return Err(Disposition::Rejected);
    }
    if cohort.unresolved_disagreement_count > 0
        || cohort.specimens.is_empty()
        || !cohort.unclassifiable_specimen_ids.is_empty()
    {
        return Err(Disposition::Blocked);
    }

    let method_observation_mode = match cohort.classification_receipt.method_id.as_str() {
        "ASTM_D5573" => "direct_unaided_visual",
        "ISO_10365" => "method_declared_visual",
        _ => return Err(Disposition::Blocked),
    };

    let mut specimen_set = HashSet::new();
    let mut cluster_set = HashSet::new();
    for sample in &cohort.specimens {
        if !nonempty(&sample.specimen_id) || !nonempty(&sample.cluster_id) {
            return Err(Disposition::Blocked);
        }
        if !specimen_set.insert(sample.specimen_id.as_str()) {
            return Err(Disposition::Rejected);
        }
        cluster_set.insert(sample.cluster_id.as_str());
        if !sha256_shape(&sample.image_sha256) {
            return Err(Disposition::Blocked);
        }
        if sample.observation_mode != method_observation_mode {
            return Err(Disposition::Rejected);
        }
        if sample.observer_refs.len() < 2 || !unique_strings(&sample.observer_refs) {
            return Err(Disposition::Blocked);
        }
        if !matches!(sample.blinding_status.as_str(), "blinded" | "unblinded_declared") {
            return Err(Disposition::Blocked);
        }
        if sample.fractions.len() != categories.len() {
            return Err(Disposition::Blocked);
        }

        let mut sum = 0.0;
        for (index, fraction) in sample.fractions.iter().enumerate() {
            if fraction.category_id != categories[index].id {
                return Err(Disposition::Rejected);
            }
            if !fraction.fraction.is_finite() || !(0.0..=1.0).contains(&fraction.fraction) {
                return Err(Disposition::Rejected);
            }
            sum += fraction.fraction;
        }
        if !sum.is_finite() || (sum - 1.0).abs() > tolerance {
            return Err(Disposition::Rejected);
        }
        if cohort.measurement_basis == "specimen_count_fraction"
            && sample.fractions.iter().filter(|fraction| fraction.fraction == 1.0).count() != 1
        {
            return Err(Disposition::Rejected);
        }
        if cohort.measurement_basis == "specimen_count_fraction"
            && sample.fractions.iter().any(|fraction| fraction.fraction != 0.0 && fraction.fraction != 1.0)
        {
            return Err(Disposition::Rejected);
        }
    }

    if !cohort.unclassifiable_specimen_ids.is_empty()
        && !unique_strings(&cohort.unclassifiable_specimen_ids)
    {
        return Err(Disposition::Rejected);
    }

    let receipt = &cohort.classification_receipt;
    if receipt.specimen_ids != specimen_ids(cohort) {
        return Err(Disposition::Rejected);
    }
    if !receipt.protocol_id.as_deref().is_some_and(nonempty)
        || !receipt.protocol_sha256.as_deref().is_some_and(sha256_shape)
        || !receipt.raw_data_sha256.as_deref().is_some_and(sha256_shape)
    {
        return Err(Disposition::Blocked);
    }
    let _ = (ontology_id, ontology_version, cluster_set);
    match decide(receipt).disposition {
        ClaimDisposition::Supported => Ok(()),
        ClaimDisposition::Blocked => Err(Disposition::Blocked),
        ClaimDisposition::Rejected => Err(Disposition::Rejected),
    }
}

fn audit_receipt(receipt: &FailureCompositionReceipt) -> Decision {
    let blocked = |reason| Decision {
        disposition: Disposition::Blocked,
        outcome: "NOT_EVALUATED",
        reason,
    };
    let rejected = |reason| Decision {
        disposition: Disposition::Rejected,
        outcome: "NOT_EVALUATED",
        reason,
    };

    if !nonempty(&receipt.receipt_id)
        || !nonempty(&receipt.category_ontology_id)
        || !nonempty(&receipt.category_ontology_version)
    {
        return blocked("receipt or category-ontology identity is missing");
    }
    if receipt.categories.len() < 2 {
        return blocked("failure-mode ontology must declare at least two categories");
    }
    let mut category_ids = HashSet::new();
    for category in &receipt.categories {
        if !nonempty(&category.id) || !nonempty(&category.label) {
            return blocked("failure-mode category identity or label is missing");
        }
        if !category_ids.insert(category.id.as_str()) {
            return rejected("failure-mode ontology contains duplicate category IDs");
        }
    }
    if !receipt.measurement_tolerance.is_finite()
        || receipt.measurement_tolerance < 0.0
        || receipt.measurement_tolerance > 0.05
    {
        return rejected("closed-composition tolerance is outside the declared S0 range");
    }
    if !sha256_shape(&receipt.vector_manifest_sha256) {
        return blocked("full vector-manifest digest is missing or malformed");
    }

    for cohort in [&receipt.baseline, &receipt.candidate] {
        if let Err(disposition) = audit_cohort(
            cohort,
            &receipt.categories,
            &receipt.category_ontology_id,
            &receipt.category_ontology_version,
            receipt.measurement_tolerance,
        ) {
            return match disposition {
                Disposition::Blocked => blocked("cohort specimen, observer, category vector or classification receipt is incomplete"),
                Disposition::Rejected => rejected("cohort specimen, category vector or method-scope contract is contradictory"),
                Disposition::Pass => unreachable!(),
            };
        }
    }

    if receipt.baseline.category_ontology_id != receipt.candidate.category_ontology_id
        || receipt.baseline.category_ontology_version != receipt.candidate.category_ontology_version
        || receipt.baseline.measurement_basis != receipt.candidate.measurement_basis
        || receipt.baseline.fraction_protocol_id != receipt.candidate.fraction_protocol_id
        || receipt.baseline.fraction_protocol_sha256 != receipt.candidate.fraction_protocol_sha256
    {
        return rejected("baseline and candidate cohorts use different fraction bases or measurement protocols");
    }

    let baseline_method = &receipt.baseline.classification_receipt;
    let candidate_method = &receipt.candidate.classification_receipt;
    if baseline_method.method_id != candidate_method.method_id
        || baseline_method.edition != candidate_method.edition
        || baseline_method.evidence_kind != candidate_method.evidence_kind
        || baseline_method.material_pair_class != candidate_method.material_pair_class
        || baseline_method.geometry != candidate_method.geometry
        || baseline_method.endpoint != candidate_method.endpoint
        || baseline_method.requested_claim != candidate_method.requested_claim
    {
        return rejected("baseline and candidate failure classifications use different methods or claim scopes");
    }

    let expected_category_ids: &[&str] = match (
        receipt.category_ontology_id.as_str(),
        baseline_method.method_id.as_str(),
    ) {
        ("D5573-SMC-FAILURE-V1", "ASTM_D5573") => &[
            "adhesive_failure",
            "cohesive_failure",
            "thin_layer_cohesive",
            "fiber_tear",
            "light_fiber_tear",
            "stock_break",
        ],
        ("ISO-10365-GENERAL-FAILURE-V1", "ISO_10365") => &[
            "SF", "CSF", "DF", "CF", "SCF", "SWCF", "AF", "ACFP", "COR",
        ],
        _ => return rejected("failure category ontology does not match its exact classification method"),
    };
    let actual_category_ids: Vec<&str> = receipt.categories.iter().map(|category| category.id.as_str()).collect();
    if actual_category_ids != expected_category_ids {
        return rejected("failure category IDs do not match the frozen method-specific ontology");
    }
    if receipt.baseline.independent_unit != receipt.candidate.independent_unit
        || receipt.baseline.grouping_scheme != receipt.candidate.grouping_scheme
    {
        return rejected("baseline and candidate cohorts declare different independent-unit or grouping semantics");
    }

    let baseline_specimens: HashSet<&str> = receipt.baseline.specimens.iter().map(|s| s.specimen_id.as_str()).collect();
    let candidate_specimens: HashSet<&str> = receipt.candidate.specimens.iter().map(|s| s.specimen_id.as_str()).collect();
    let baseline_clusters = cluster_ids(&receipt.baseline);
    let candidate_clusters = cluster_ids(&receipt.candidate);
    if !baseline_specimens.is_disjoint(&candidate_specimens) {
        return rejected("baseline and candidate cohorts reuse specimen identities");
    }
    match receipt.baseline.grouping_scheme.as_str() {
        "independent_arms" if !baseline_clusters.is_disjoint(&candidate_clusters) => {
            return rejected("independent-arm failure-composition cohorts share a manufacturing cluster");
        }
        "paired_by_cluster" if baseline_clusters != candidate_clusters => {
            return rejected("paired-by-cluster cohort IDs do not form the exact same independent pairs");
        }
        _ => {}
    }

    let plan = &receipt.analysis_plan;
    for value in [
        plan.plan_id.as_str(),
        plan.analysis_profile.as_str(),
        plan.analysis_method_ref.as_str(),
        plan.agreement_policy_ref.as_str(),
    ] {
        if !nonempty(value) {
            return blocked("compositional analysis plan identity or method reference is missing");
        }
    }
    if !sha256_shape(&plan.plan_sha256) {
        return blocked("compositional plan digest is missing or malformed");
    }
    if plan.changed_after_outcome || plan.frozen_at >= plan.first_outcome_available_at {
        return rejected("compositional analysis plan was changed or frozen after outcome availability");
    }
    if !plan.alpha.is_finite()
        || plan.alpha <= 0.0
        || plan.alpha >= 0.5
        || !plan.confidence_level.is_finite()
        || !(0.0..1.0).contains(&plan.confidence_level)
        || plan.confidence_level <= 0.0
        || !plan.joint_distance_margin.is_finite()
        || plan.joint_distance_margin <= 0.0
        || !plan.minimum_observer_agreement.is_finite()
        || !(0.0..=1.0).contains(&plan.minimum_observer_agreement)
    {
        return rejected("compositional alpha, confidence, distance margin or observer-agreement threshold is invalid");
    }
    if plan.minimum_baseline_clusters == 0 || plan.minimum_candidate_clusters == 0 {
        return rejected("minimum independent-cluster counts must be positive");
    }
    if !matches!(plan.zero_policy.as_str(), "none" | "multiplicative_replacement" | "model_based") {
        return rejected("zero/near-zero handling policy is not registered");
    }
    let has_zero = receipt.baseline.specimens.iter().chain(receipt.candidate.specimens.iter())
        .flat_map(|sample| sample.fractions.iter())
        .any(|fraction| fraction.fraction == 0.0);
    match plan.zero_policy.as_str() {
        "none" if plan.pseudocount.is_some() => return rejected("pseudocount supplied while zero policy is none"),
        "none" if has_zero => return blocked("zero-valued categories require an explicit preregistered structural-zero or replacement policy"),
        "multiplicative_replacement" if !plan.pseudocount.is_some_and(|value| value.is_finite() && value > 0.0) => {
            return blocked("multiplicative zero replacement requires a declared positive pseudocount")
        }
        "model_based" if plan.pseudocount.is_some() => return rejected("model-based zero handling must not also supply a pseudocount"),
        _ => {}
    }

    if baseline_clusters.len() < plan.minimum_baseline_clusters
        || candidate_clusters.len() < plan.minimum_candidate_clusters
    {
        return blocked("actual independent cluster count is below the preregistered compositional minimum");
    }
    if plan.category_margins.len() != receipt.categories.len() {
        return blocked("analysis plan must provide a margin for every declared failure-mode category");
    }
    for (index, margin) in plan.category_margins.iter().enumerate() {
        if margin.category_id != receipt.categories[index].id {
            return rejected("analysis category margins do not match the frozen ontology order");
        }
        if !margin.lower.is_finite() || !margin.upper.is_finite() || margin.lower >= margin.upper {
            return rejected("category-specific compositional margins are invalid");
        }
    }

    let result = &receipt.analysis_result;
    if !nonempty(&result.result_id)
        || !sha256_shape(&result.result_sha256)
        || !sha256_shape(&result.vector_manifest_sha256)
    {
        return blocked("compositional analysis result identity or digest is missing");
    }
    if result.plan_sha256 != plan.plan_sha256
        || result.vector_manifest_sha256 != receipt.vector_manifest_sha256
        || result.analysis_profile != plan.analysis_profile
        || result.baseline_specimen_count != receipt.baseline.specimens.len()
        || result.candidate_specimen_count != receipt.candidate.specimens.len()
        || result.baseline_cluster_count != baseline_clusters.len()
        || result.candidate_cluster_count != candidate_clusters.len()
    {
        return rejected("analysis result does not bind the frozen plan, full-vector manifest, or actual cohort counts");
    }
    if !result.observer_agreement.is_finite()
        || !(0.0..=1.0).contains(&result.observer_agreement)
        || !result.alpha.is_finite()
        || !result.confidence_level.is_finite()
    {
        return rejected("observed inter-rater agreement is outside [0,1]");
    }
    if result.observer_agreement < plan.minimum_observer_agreement {
        return blocked("observed inter-rater agreement is below the preregistered minimum");
    }
    if (result.alpha - plan.alpha).abs() > 1e-9
        || (result.confidence_level - plan.confidence_level).abs() > 1e-9
    {
        return rejected("analysis result alpha or confidence level differs from the frozen plan");
    }
    if result.category_effects.len() != receipt.categories.len() {
        return blocked("analysis result does not report a full per-category effect vector");
    }
    let mut all_categories_within_margin = true;
    for (index, effect) in result.category_effects.iter().enumerate() {
        let margin = &plan.category_margins[index];
        if effect.category_id != receipt.categories[index].id || effect.category_id != margin.category_id {
            return rejected("analysis result category effects do not match the frozen ontology order");
        }
        if !effect.lower.is_finite() || !effect.upper.is_finite() || effect.lower > effect.upper {
            return rejected("per-category effect interval is non-finite or reversed");
        }
        if effect.lower < margin.lower || effect.upper > margin.upper {
            all_categories_within_margin = false;
        }
    }
    if !result.joint_distance_lower.is_finite()
        || !result.joint_distance_upper.is_finite()
        || result.joint_distance_lower < 0.0
        || result.joint_distance_upper < result.joint_distance_lower
    {
        return rejected("joint compositional-distance interval is invalid");
    }
    let equivalent = all_categories_within_margin
        && result.joint_distance_upper <= plan.joint_distance_margin;
    let expected_outcome = if equivalent {
        "COMPOSITIONAL_EQUIVALENCE_ESTABLISHED"
    } else {
        "COMPOSITIONAL_EQUIVALENCE_NOT_ESTABLISHED"
    };
    if result.outcome != expected_outcome {
        return rejected("reported compositional-equivalence outcome does not match per-category and joint-distance intervals");
    }

    Decision {
        disposition: Disposition::Pass,
        outcome: expected_outcome,
        reason: "complete specimen-level failure-mode vectors, observer receipt, method scope and preregistered cluster-aware result linkage are structurally consistent; no acceptability decision is implied",
    }
}

fn read_fixture(path: &Path) -> Result<Fixture> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read fixture {}", path.display()))?;
    serde_json::from_str(&content)
        .with_context(|| format!("failed to parse fixture {}", path.display()))
}

pub fn run(path: &Path) -> Result<()> {
    let fixture = read_fixture(path)?;
    if fixture.schema_version != "plasma-failure-composition-fixture-v0.1" {
        bail!("unsupported fixture schema version: {}", fixture.schema_version);
    }

    let mut counts = [0usize; 3];
    for case in &fixture.cases {
        let decision = audit_receipt(&case.receipt);
        let index = match decision.disposition {
            Disposition::Pass => 0,
            Disposition::Blocked => 1,
            Disposition::Rejected => 2,
        };
        counts[index] += 1;
        println!("{}: {} / {} — {}", case.id, decision.disposition.as_str(), decision.outcome, decision.reason);
        if decision.disposition.as_str() != case.expected_disposition {
            bail!("case {}: expected {}, got {}", case.id, case.expected_disposition, decision.disposition.as_str());
        }
        if decision.outcome != case.expected_outcome {
            bail!("case {}: expected outcome {}, got {}", case.id, case.expected_outcome, decision.outcome);
        }
        if !nonempty(&case.rationale) {
            bail!("case {} has an empty rationale", case.id);
        }
    }

    println!("PLASMA-FAILURE-COMPOSITION S0 — {} cases; PASS={} BLOCKED={} REJECTED={}", fixture.cases.len(), counts[0], counts[1], counts[2]);
    println!("Claim ceiling: descriptive failure-mode distribution only; no acceptability, structural safety or product-release authority.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str =
        include_str!("../../data/research/plasma_failure_composition_fixture_v0_1.json");

    #[test]
    fn frozen_fixture_replays_all_expected_outcomes() {
        let fixture: Fixture = serde_json::from_str(FIXTURE).expect("valid failure-composition fixture");
        assert_eq!(fixture.schema_version, "plasma-failure-composition-fixture-v0.1");
        for case in &fixture.cases {
            let result = audit_receipt(&case.receipt);
            assert_eq!(result.disposition.as_str(), case.expected_disposition, "{}", case.id);
            assert_eq!(result.outcome, case.expected_outcome, "{}", case.id);
        }
    }

    #[test]
    fn incomplete_vector_is_rejected_not_renormalized() {
        let fixture: Fixture = serde_json::from_str(FIXTURE).expect("valid fixture");
        let case = fixture.cases.iter().find(|case| case.id == "specimen-vector-sums-below-one").expect("case exists");
        assert_eq!(audit_receipt(&case.receipt).disposition, Disposition::Rejected);
    }

    #[test]
    fn scalar_agreement_does_not_override_full_composition_change() {
        let fixture: Fixture = serde_json::from_str(FIXTURE).expect("valid fixture");
        let case = fixture.cases.iter().find(|case| case.id == "same-cohesive-share-different-mode-vector").expect("case exists");
        let result = audit_receipt(&case.receipt);
        assert_eq!(result.disposition, Disposition::Pass);
        assert_eq!(result.outcome, "COMPOSITIONAL_EQUIVALENCE_NOT_ESTABLISHED");
    }

    #[test]
    fn post_outcome_plan_freeze_is_rejected() {
        let fixture: Fixture = serde_json::from_str(FIXTURE).expect("valid fixture");
        let case = fixture.cases.iter().find(|case| case.id == "plan-frozen-after-outcomes").expect("case exists");
        assert_eq!(audit_receipt(&case.receipt).disposition, Disposition::Rejected);
    }

    #[test]
    fn shared_lot_does_not_count_as_multiple_independent_clusters() {
        let fixture: Fixture = serde_json::from_str(FIXTURE).expect("valid fixture");
        let case = fixture.cases.iter().find(|case| case.id == "many-specimens-one-lot").expect("case exists");
        assert_eq!(audit_receipt(&case.receipt).disposition, Disposition::Blocked);
    }

    #[test]
    fn iso_10365_uses_its_own_general_bonded_assembly_ontology() {
        let fixture: Fixture = serde_json::from_str(FIXTURE).expect("valid fixture");
        let case = fixture.cases.iter().find(|case| case.id == "iso-10365-general-bonded-assembly-pattern").expect("case exists");
        assert_eq!(audit_receipt(&case.receipt).disposition, Disposition::Pass);
    }

    #[test]
    fn iso_10365_and_d5573_ontologies_cannot_be_mixed() {
        let mut fixture: Fixture = serde_json::from_str(FIXTURE).expect("valid fixture");
        let case = fixture.cases.iter_mut().find(|case| case.id == "valid-independent-arm-failure-composition").expect("case exists");
        case.receipt.candidate.classification_receipt.method_id = "ISO_10365".to_owned();
        assert_eq!(audit_receipt(&case.receipt).disposition, Disposition::Rejected);
    }

    #[test]
    fn magnified_classification_does_not_claim_d5573_visual_scope() {
        let mut fixture: Fixture = serde_json::from_str(FIXTURE).expect("valid fixture");
        let case = fixture.cases.iter_mut().find(|case| case.id == "valid-independent-arm-failure-composition").expect("case exists");
        case.receipt.baseline.specimens[0].observation_mode = "magnified_imaging".to_owned();
        assert_eq!(audit_receipt(&case.receipt).disposition, Disposition::Rejected);
    }

    #[test]
    fn specimen_count_fraction_requires_one_hot_specimen_classification() {
        let fixture: Fixture = serde_json::from_str(FIXTURE).expect("valid fixture");
        let case = fixture.cases.iter().find(|case| case.id == "count-fraction-must-be-one-hot").expect("case exists");
        assert_eq!(audit_receipt(&case.receipt).disposition, Disposition::Rejected);
    }

    #[test]
    fn d5573_classification_never_becomes_acceptability() {
        let fixture: Fixture = serde_json::from_str(FIXTURE).expect("valid fixture");
        let case = fixture.cases.iter().find(|case| case.id == "classification-claim-is-descriptive").expect("case exists");
        assert_eq!(audit_receipt(&case.receipt).disposition, Disposition::Pass);
    }
}
