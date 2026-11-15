use anyhow::{bail, Context};
use serde::Deserialize;
use std::collections::HashSet;
use std::fs;
use std::path::Path;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Registry {
    schema_version: String,
    programme: String,
    generated_date: String,
    claim_ceiling: String,
    sources: Vec<Source>,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Source {
    id: String,
    modality: String,
    indication: String,
    qualified_scope: String,
    evidence_class: String,
    study_design: String,
    publication_status: String,
    human_participants: Option<u32>,
    study_count: Option<u32>,
    claim_ceiling: String,
    allowed_claims: Vec<String>,
    source_url: String,
    limitations: Vec<String>,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Fixture {
    schema_version: String,
    benchmark_id: String,
    generated_date: String,
    claim_ceiling: String,
    counts_expected: Counts,
    cases: Vec<Case>,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Counts {
    #[serde(rename = "SUPPORTED")] supported: usize,
    #[serde(rename = "BLOCKED")] blocked: usize,
    #[serde(rename = "REJECTED")] rejected: usize,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    id: String,
    source_id: String,
    requested_modality: String,
    requested_indication: String,
    requested_scope: String,
    requested_claim: String,
    expected_disposition: String,
    rationale: String,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Disposition { Supported, Blocked, Rejected }
impl Disposition {
    fn as_str(self) -> &'static str {
        match self { Self::Supported => "SUPPORTED", Self::Blocked => "BLOCKED", Self::Rejected => "REJECTED" }
    }
}
struct Outcome { id: String, disposition: Disposition, reason: &'static str }
fn outcome(case: &Case, disposition: Disposition, reason: &'static str) -> Outcome {
    Outcome { id: case.id.clone(), disposition, reason }
}
fn decide(case: &Case, registry: &Registry) -> Outcome {
    let blocked = |reason| outcome(case, Disposition::Blocked, reason);
    let rejected = |reason| outcome(case, Disposition::Rejected, reason);
    let supported = |reason| outcome(case, Disposition::Supported, reason);
    let Some(source) = registry.sources.iter().find(|source| source.id == case.source_id) else {
        return blocked("source identity is absent from the versioned research register");
    };
    if case.requested_modality != source.modality {
        return blocked("intervention modality differs; an explicit modality-specific evidence bridge is required");
    }
    if case.requested_indication != source.indication {
        return blocked("medical indication differs from the source population and intended-use scope");
    }
    if case.requested_scope != source.qualified_scope {
        return blocked("requested population/device/protocol scope differs from the source-specific scope");
    }
    if source.allowed_claims.iter().any(|claim| claim == &case.requested_claim) {
        supported("requested claim is within the registered source-specific evidence ceiling")
    } else {
        rejected("requested clinical or regulatory claim exceeds the registered evidence ceiling")
    }
}
fn validate_registry(registry: &Registry) -> anyhow::Result<()> {
    if registry.schema_version != "0.1.0"
        || registry.programme != "PLASMA-MEDICAL-001"
        || registry.claim_ceiling != "PUBLIC_RESEARCH_SOURCE_REGISTER_ONLY_NO_TREATMENT_DOSE_DEVICE_SAFETY_CLINICAL_EFFICACY_OR_REGULATORY_AUTHORIZATION"
        || registry.sources.is_empty() {
        bail!("medical source register identity, version, claim ceiling or source set is invalid");
    }
    if registry.generated_date.trim().is_empty() { bail!("medical source register date is empty"); }
    let mut ids = HashSet::new();
    for source in &registry.sources {
        if source.id.trim().is_empty() || !ids.insert(source.id.as_str()) {
            bail!("source identity is empty or duplicated: {}", source.id);
        }
        if source.modality.trim().is_empty() || source.indication.trim().is_empty()
            || source.qualified_scope.trim().is_empty() || source.evidence_class.trim().is_empty()
            || source.study_design.trim().is_empty() || source.publication_status.trim().is_empty()
            || source.claim_ceiling.trim().is_empty() || source.allowed_claims.is_empty()
            || source.limitations.is_empty() {
            bail!("source {} has incomplete scope/evidence/limitation declarations", source.id);
        }
        if !source.source_url.starts_with("https://") {
            bail!("source {} has a non-HTTPS source authority", source.id);
        }
        if source.human_participants == Some(0) || source.study_count == Some(0) {
            bail!("source {} contains a zero-valued population/study count", source.id);
        }
        let mut claims = HashSet::new();
        for claim in &source.allowed_claims {
            if claim.trim().is_empty() || !claims.insert(claim.as_str()) {
                bail!("source {} has an empty or duplicate allowed claim", source.id);
            }
        }
    }
    Ok(())
}
fn evaluate(registry: &Registry, fixture: &Fixture) -> anyhow::Result<Vec<Outcome>> {
    validate_registry(registry)?;
    if fixture.schema_version != "0.1.0" || fixture.benchmark_id != "PLASMA-MEDICAL-001"
        || fixture.claim_ceiling != "SYNTHETIC_EVIDENCE_CLAIM_AUDIT_ONLY_NO_CLINICAL_DECISION_OR_DEVICE_AUTHORITY"
        || fixture.generated_date.trim().is_empty() || fixture.cases.is_empty() {
        bail!("medical fixture identity, version, claim ceiling or scenario set is invalid");
    }
    let mut ids = HashSet::new();
    let mut results = Vec::with_capacity(fixture.cases.len());
    for case in &fixture.cases {
        if case.id.trim().is_empty() || !ids.insert(case.id.as_str()) {
            bail!("scenario identity is empty or duplicated: {}", case.id);
        }
        if case.requested_modality.trim().is_empty() || case.requested_indication.trim().is_empty()
            || case.requested_scope.trim().is_empty() || case.requested_claim.trim().is_empty() {
            bail!("scenario {} has an incomplete requested claim", case.id);
        }
        let result = decide(case, registry);
        if result.disposition.as_str() != case.expected_disposition {
            bail!("known-answer mismatch for {}: expected {}, got {} ({})",
                result.id, case.expected_disposition, result.disposition.as_str(), result.reason);
        }
        let _rationale_for_review = case.rationale.as_str();
        results.push(result);
    }
    let supported = results.iter().filter(|x| x.disposition == Disposition::Supported).count();
    let blocked = results.iter().filter(|x| x.disposition == Disposition::Blocked).count();
    let rejected = results.iter().filter(|x| x.disposition == Disposition::Rejected).count();
    if supported != fixture.counts_expected.supported || blocked != fixture.counts_expected.blocked
        || rejected != fixture.counts_expected.rejected {
        bail!("fixture disposition counts do not match evaluated outcomes");
    }
    Ok(results)
}
pub fn run(root: &Path) -> anyhow::Result<()> {
    let data_dir = root.join("data/research");
    let registry_path = data_dir.join("plasma_medical_source_register_v0_1.json");
    let fixture_path = data_dir.join("plasma_medical_claim_fixture_v0_1.json");
    let registry_text = fs::read_to_string(&registry_path).with_context(|| format!("reading {}", registry_path.display()))?;
    let fixture_text = fs::read_to_string(&fixture_path).with_context(|| format!("reading {}", fixture_path.display()))?;
    let registry: Registry = serde_json::from_str(&registry_text).with_context(|| format!("parsing {}", registry_path.display()))?;
    let fixture: Fixture = serde_json::from_str(&fixture_text).with_context(|| format!("parsing {}", fixture_path.display()))?;
    let results = evaluate(&registry, &fixture)?;
    let supported = results.iter().filter(|x| x.disposition == Disposition::Supported).count();
    let blocked = results.iter().filter(|x| x.disposition == Disposition::Blocked).count();
    let rejected = results.iter().filter(|x| x.disposition == Disposition::Rejected).count();
    println!("{}: {}/{} known-answer claims matched (SUPPORTED {}, BLOCKED {}, REJECTED {})",
        fixture.benchmark_id, results.len(), fixture.cases.len(), supported, blocked, rejected);
    for item in results { println!("{} {} — {}", item.disposition.as_str(), item.id, item.reason); }
    println!("Public-source claim audit only; no medical decision, dose, device or clinical authority.");
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    fn inputs() -> (Registry, Fixture) {
        let registry: Registry = serde_json::from_str(include_str!("../../data/research/plasma_medical_source_register_v0_1.json")).expect("source register parses");
        let fixture: Fixture = serde_json::from_str(include_str!("../../data/research/plasma_medical_claim_fixture_v0_1.json")).expect("claim fixture parses");
        (registry, fixture)
    }
    #[test]
    fn all_medical_claim_scenarios_replay_with_frozen_counts() {
        let (registry, fixture) = inputs();
        let result = evaluate(&registry, &fixture).expect("medical claim replay");
        assert_eq!(result.len(), 29);
        assert_eq!(fixture.counts_expected.supported, 11);
        assert_eq!(fixture.counts_expected.blocked, 7);
        assert_eq!(fixture.counts_expected.rejected, 11);
    }
    #[test]
    fn phase_one_oncology_safety_signal_does_not_support_efficacy() {
        let (registry, fixture) = inputs();
        let case = fixture.cases.iter().find(|x| x.id == "ONCOLOGY_PHASE_I_PROMOTED_TO_CANCER_EFFICACY").unwrap();
        assert_eq!(decide(case, &registry).disposition, Disposition::Rejected);
    }
    #[test]
    fn trial_protocol_does_not_become_a_result() {
        let (registry, fixture) = inputs();
        let case = fixture.cases.iter().find(|x| x.id == "NIPP_RD_III_PROTOCOL_AS_EFFICACY_RESULT").unwrap();
        assert_eq!(decide(case, &registry).disposition, Disposition::Rejected);
    }
    #[test]
    fn direct_cap_does_not_transfer_to_plasma_activated_water() {
        let (registry, fixture) = inputs();
        let case = fixture.cases.iter().find(|x| x.id == "DIRECT_CAP_TRANSFERRED_TO_PAW").unwrap();
        assert_eq!(decide(case, &registry).disposition, Disposition::Blocked);
    }
    #[test]
    fn indication_mismatch_blocks_cross_population_transfer() {
        let (registry, fixture) = inputs();
        let case = fixture.cases.iter().find(|x| x.id == "WOUND_RCT_TRANSFERRED_TO_VENOUS_ULCERS").unwrap();
        assert_eq!(decide(case, &registry).disposition, Disposition::Blocked);
    }
    #[test]
    fn marketing_authorization_is_not_inferred_from_an_ide_trial() {
        let (registry, fixture) = inputs();
        let case = fixture.cases.iter().find(|x| x.id == "ONCOLOGY_IDE_MISSTATED_AS_MARKETING_APPROVAL").unwrap();
        assert_eq!(decide(case, &registry).disposition, Disposition::Rejected);
    }

    #[test]
    fn nonsignificant_venous_ulcer_trial_is_not_promoted_to_superiority_or_equivalence() {
        let (registry, fixture) = inputs();
        let superiority = fixture.cases.iter().find(|x| x.id == "VLU_RCT_PROMOTED_TO_STATISTICAL_SUPERIORITY").unwrap();
        let equivalence = fixture.cases.iter().find(|x| x.id == "VLU_RCT_NON_SIGNIFICANCE_PROMOTED_TO_EQUIVALENCE").unwrap();
        assert_eq!(decide(superiority, &registry).disposition, Disposition::Rejected);
        assert_eq!(decide(equivalence, &registry).disposition, Disposition::Rejected);
    }

    #[test]
    fn device_catalogue_does_not_establish_medical_efficacy() {
        let (registry, fixture) = inputs();
        let case = fixture.cases.iter().find(|x| x.id == "KINPEN_MED_SPECIFICATION_PROMOTED_TO_EFFICACY").unwrap();
        assert_eq!(decide(case, &registry).disposition, Disposition::Rejected);
    }

    #[test]
    fn recruitment_status_does_not_promote_phase_three_protocol_to_results() {
        let (registry, fixture) = inputs();
        let status = fixture.cases.iter().find(|x| x.id == "NIPP_RD_III_RECRUITMENT_COMPLETE_STATUS").unwrap();
        let efficacy = fixture.cases.iter().find(|x| x.id == "NIPP_RD_III_PROTOCOL_AS_EFFICACY_RESULT").unwrap();
        assert_eq!(decide(status, &registry).disposition, Disposition::Supported);
        assert_eq!(decide(efficacy, &registry).disposition, Disposition::Rejected);
    }

    #[test]
    fn proposed_dose_framework_is_not_a_validated_universal_standard() {
        let (registry, fixture) = inputs();
        let case = fixture.cases.iter().find(|x| x.id == "DOSE_FRAMEWORK_PROMOTED_TO_UNIVERSAL_STANDARD").unwrap();
        assert_eq!(decide(case, &registry).disposition, Disposition::Rejected);
    }
}
