use anyhow::{bail, Context};
use serde::Deserialize;
use std::collections::HashSet;
use std::path::Path;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Fixture {
    schema_version: String,
    cases: Vec<Case>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Case {
    pub(super) id: String,
    pub(super) method_id: String,
    pub(super) edition: String,
    pub(super) evidence_kind: String,
    pub(super) material_pair_class: String,
    pub(super) geometry: String,
    pub(super) endpoint: String,
    pub(super) requested_claim: String,
    pub(super) protocol_id: Option<String>,
    pub(super) protocol_sha256: Option<String>,
    pub(super) raw_data_sha256: Option<String>,
    pub(super) suitability_review_sha256: Option<String>,
    pub(super) specimen_ids: Vec<String>,
    pub(super) expected_disposition: String,
    pub(super) rationale: String,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub(super) struct ScopeEvidence {
    pub(super) id: String,
    pub(super) method_id: String,
    pub(super) edition: String,
    pub(super) evidence_kind: String,
    pub(super) material_pair_class: String,
    pub(super) geometry: String,
    pub(super) endpoint: String,
    pub(super) requested_claim: String,
    pub(super) protocol_id: Option<String>,
    pub(super) protocol_sha256: Option<String>,
    pub(super) raw_data_sha256: Option<String>,
    pub(super) suitability_review_sha256: Option<String>,
    pub(super) specimen_ids: Vec<String>,
}

impl From<&Case> for ScopeEvidence {
    fn from(case: &Case) -> Self {
        Self {
            id: case.id.clone(),
            method_id: case.method_id.clone(),
            edition: case.edition.clone(),
            evidence_kind: case.evidence_kind.clone(),
            material_pair_class: case.material_pair_class.clone(),
            geometry: case.geometry.clone(),
            endpoint: case.endpoint.clone(),
            requested_claim: case.requested_claim.clone(),
            protocol_id: case.protocol_id.clone(),
            protocol_sha256: case.protocol_sha256.clone(),
            raw_data_sha256: case.raw_data_sha256.clone(),
            suitability_review_sha256: case.suitability_review_sha256.clone(),
            specimen_ids: case.specimen_ids.clone(),
        }
    }
}

#[derive(Clone, Copy)]
struct MethodRule {
    method_id: &'static str,
    edition: &'static str,
    evidence_kind: &'static str,
    material_pair_classes: &'static [&'static str],
    geometry: &'static str,
    endpoint: &'static str,
    allowed_claims: &'static [&'static str],
    claim_ceiling: &'static str,
}

const RULES: &[MethodRule] = &[
    MethodRule { method_id: "ISO_4587", edition: "ISO 4587:2003", evidence_kind: "measurement", material_pair_classes: &["rigid_to_rigid"], geometry: "single_lap", endpoint: "tensile_lap_shear_strength", allowed_claims: &["method_observation"], claim_ceiling: "tensile lap-shear measurement for the specified rigid-to-rigid bonded assembly and test conditions; not design information" },
    MethodRule { method_id: "ASTM_D1002", edition: "D1002-26", evidence_kind: "measurement", material_pair_classes: &["metal_to_metal"], geometry: "single_lap", endpoint: "apparent_shear_strength", allowed_claims: &["method_observation", "comparative_screening"], claim_ceiling: "apparent shear measurement / cautious comparative screening for the specified metal-to-metal single-lap specimen; not a structural design allowable" },
    MethodRule { method_id: "ASTM_D3163", edition: "D3163-01R23", evidence_kind: "measurement", material_pair_classes: &["rigid_plastic_pair"], geometry: "single_lap", endpoint: "apparent_shear_strength", allowed_claims: &["method_observation", "comparative_screening"], claim_ceiling: "comparative apparent shear measurement for the specified rigid-plastic single-lap joint; not a structural design allowable" },
    MethodRule { method_id: "ASTM_D5868", edition: "D5868-01R23", evidence_kind: "measurement", material_pair_classes: &["frp_to_frp", "frp_to_metal"], geometry: "single_lap", endpoint: "apparent_shear_strength", allowed_claims: &["method_observation", "comparative_screening"], claim_ceiling: "comparative apparent shear measurement for the specified FRP-to-FRP or FRP-to-metal single-lap joint; not a structural design allowable" },
    MethodRule { method_id: "ASTM_D3762", edition: "D3762-24", evidence_kind: "measurement", material_pair_classes: &["aluminum_to_aluminum"], geometry: "wedge", endpoint: "qualitative_wedge_durability", allowed_claims: &["method_observation", "qualitative_surface_durability_screen"], claim_ceiling: "qualitative wedge-test surface-durability discrimination and failure-mode observation in the stated applicable specimen scope; not quantitative fracture strength" },
    MethodRule { method_id: "WATER_CONTACT_ANGLE_PROTOCOL", edition: "protocol-defined-v1", evidence_kind: "measurement", material_pair_classes: &["material_surface"], geometry: "sessile_drop", endpoint: "water_contact_angle", allowed_claims: &["wettability_observation"], claim_ceiling: "water-contact-angle / wettability observation under the identified protocol; not adhesive bond strength" },
    MethodRule { method_id: "ASTM_D2093", edition: "D2093-03R25", evidence_kind: "preparation_practice", material_pair_classes: &["plastic_adherend"], geometry: "surface_preparation", endpoint: "plastic_surface_preparation_protocol", allowed_claims: &["preparation_protocol_reference"], claim_ceiling: "plastic-adherend surface-preparation procedure reference only; it does not establish specimen testing or bond performance" },
    MethodRule { method_id: "ASTM_D1151", edition: "D1151-00R22", evidence_kind: "conditioning_practice", material_pair_classes: &["adhesive_bonded_specimens"], geometry: "continuous_environmental_conditioning", endpoint: "moisture_temperature_exposure_protocol", allowed_claims: &["conditioning_protocol_reference"], claim_ceiling: "continuous moisture/temperature conditioning protocol reference only; retained-strength outcome requires its own qualified test evidence" },
    MethodRule { method_id: "ASTM_D1183", edition: "D1183-03R19", evidence_kind: "conditioning_practice", material_pair_classes: &["adhesive_bonded_specimens"], geometry: "cyclic_environmental_conditioning", endpoint: "cyclic_temperature_humidity_exposure_protocol", allowed_claims: &["conditioning_protocol_reference"], claim_ceiling: "cyclic temperature/humidity conditioning protocol reference only; it does not by itself establish service-life equivalence" },
    MethodRule { method_id: "ASTM_D4896", edition: "D4896-01R24", evidence_kind: "interpretation_guide", material_pair_classes: &["single_lap_bonded_joint"], geometry: "single_lap", endpoint: "single_lap_result_interpretation_guidance", allowed_claims: &["interpretation_guidance_reference"], claim_ceiling: "guidance reference for interpreting single-lap adhesive-joint strength results; not a test measurement or design allowable" },
    MethodRule { method_id: "ASTM_D5656_D5656M", edition: "D5656/D5656M-26", evidence_kind: "measurement", material_pair_classes: &["aluminum_thick_adherend_joint"], geometry: "thick_adherend_lap", endpoint: "adhesive_shear_stress_strain_response", allowed_claims: &["adhesive_constitutive_response_observation"], claim_ceiling: "adhesive shear stress-strain, proportional-limit and shear-strain-at-failure characterization for the stated aluminum thick-adherend specimen; not substrate-specific adhesion strength or a design allowable" },
    MethodRule { method_id: "ASTM_D5573", edition: "D5573-99R19", evidence_kind: "classification_practice", material_pair_classes: &["frp_smc_joint"], geometry: "failure_surface_classification", endpoint: "frp_joint_failure_mode_classification", allowed_claims: &["failure_mode_classification_reference"], claim_ceiling: "descriptive classification of failure modes for the scoped SMC FRP adhesive joint; the practice does not judge acceptability, strength, product disposition or safety" },
    MethodRule { method_id: "ISO_10365", edition: "ISO 10365:2022", evidence_kind: "classification_practice", material_pair_classes: &["bonded_assembly_any_adherends"], geometry: "failure_surface_classification", endpoint: "adhesive_joint_failure_pattern", allowed_claims: &["failure_pattern_classification_reference"], claim_ceiling: "descriptive designation of main failure patterns for mechanical tests on bonded assemblies regardless of adherend or adhesive; not failure acceptability, strength, safety or product disposition" },
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Disposition {
    Supported,
    Blocked,
    Rejected,
}

impl Disposition {
    fn as_str(self) -> &'static str {
        match self {
            Self::Supported => "SUPPORTED",
            Self::Blocked => "BLOCKED",
            Self::Rejected => "REJECTED",
        }
    }
}

pub(super) struct Decision {
    pub(super) disposition: Disposition,
    pub(super) reason: &'static str,
    pub(super) claim_ceiling: &'static str,
}

fn is_sha256(value: Option<&str>) -> bool {
    value.is_some_and(|digest| {
        digest.len() == 64
            && digest
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    })
}

fn is_known_overclaim(claim: &str) -> bool {
    matches!(
        claim,
        "design_allowable"
            | "structural_certification"
            | "service_lifetime"
            | "time_to_failure"
            | "quantitative_fracture_strength"
            | "quantitative_fracture_toughness"
            | "bond_strength"
            | "bond_strength_from_contact_angle"
            | "fatigue_life"
            | "equivalence_from_nonsignificance"
            | "substrate_adhesion_strength"
            | "interfacial_adhesion_strength"
            | "bond_strength_from_adhesive_constitutive_response"
            | "failure_mode_acceptability"
            | "acceptable_failure_mode"
            | "failure_mode_design_allowable"
            | "acceptability_from_classification"
            | "product_disposition"
    )
}

pub(super) fn decide(case: &ScopeEvidence) -> Decision {
    let blocked = |reason, ceiling| Decision {
        disposition: Disposition::Blocked,
        reason,
        claim_ceiling: ceiling,
    };
    let rejected = |reason, ceiling| Decision {
        disposition: Disposition::Rejected,
        reason,
        claim_ceiling: ceiling,
    };
    let supported = |ceiling| Decision {
        disposition: Disposition::Supported,
        reason: "the requested claim is within the versioned method, material, geometry and endpoint map; structural evidence fields are present",
        claim_ceiling: ceiling,
    };

    let Some(rule) = RULES.iter().find(|rule| rule.method_id == case.method_id) else {
        return blocked(
            "method is not in the versioned S0 registry; a name or vendor assertion cannot establish scope",
            "no claim supported",
        );
    };

    if case.edition != rule.edition {
        return blocked(
            "edition is unrecognized or not the exact edition pinned by this registry",
            "no claim supported until the applicable controlled edition is reviewed",
        );
    }

    if case.evidence_kind != rule.evidence_kind {
        return rejected(
            "evidence kind does not match the registered method/practice/guide family",
            rule.claim_ceiling,
        );
    }

    if is_known_overclaim(&case.requested_claim) {
        return rejected(
            "requested claim exceeds the method or observation type; a separate qualified evidence bridge is required",
            rule.claim_ceiling,
        );
    }

    if case.method_id == "ASTM_D3762"
        && case.material_pair_class != "aluminum_to_aluminum"
    {
        let _documented_review_digest = case.suitability_review_sha256.as_deref();
        return blocked(
            "D3762-24 defaults to aluminium-to-aluminium; use on other materials requires a separate thickness/rigidity suitability review that S0 does not authenticate or approve",
            rule.claim_ceiling,
        );
    }

    if case.method_id == "ASTM_D5656_D5656M"
        && case.material_pair_class != "aluminum_thick_adherend_joint"
    {
        return blocked(
            "D5656/D5656M-26 was developed using bonded aluminum adherends; non-aluminum applicability is not assumed and requires a separately qualified scope bridge",
            rule.claim_ceiling,
        );
    }

    if case.method_id == "ASTM_D5573"
        && case.material_pair_class != "frp_smc_joint"
    {
        return blocked(
            "D5573-99R19 was developed for the stated SMC FRP joint family; generalized CFRP or other FRP applicability requires a separately reviewed scope bridge",
            rule.claim_ceiling,
        );
    }

    if !rule.material_pair_classes.contains(&case.material_pair_class.as_str()) {
        return rejected(
            "material-pair class is outside the encoded scope of the exact method edition",
            rule.claim_ceiling,
        );
    }

    if case.geometry != rule.geometry {
        return rejected(
            "specimen/joint geometry does not match the encoded method scope",
            rule.claim_ceiling,
        );
    }

    if case.endpoint != rule.endpoint {
        return rejected(
            "reported endpoint does not match the endpoint defined by the method scope",
            rule.claim_ceiling,
        );
    }

    if !rule.allowed_claims.contains(&case.requested_claim.as_str()) {
        return blocked(
            "claim mapping is not explicitly admitted by the S0 method registry",
            rule.claim_ceiling,
        );
    }

    if case.protocol_id.as_deref().is_none_or(str::is_empty) {
        return blocked("protocol identity is missing", rule.claim_ceiling);
    }

    if !is_sha256(case.protocol_sha256.as_deref()) {
        return blocked(
            "protocol digest is missing or malformed",
            rule.claim_ceiling,
        );
    }

    if !is_sha256(case.raw_data_sha256.as_deref()) {
        return blocked(
            "raw-data artifact digest is missing or malformed",
            rule.claim_ceiling,
        );
    }

    if case.specimen_ids.is_empty() {
        return blocked("specimen identity list is empty", rule.claim_ceiling);
    }

    let mut unique_ids = HashSet::new();
    if case
        .specimen_ids
        .iter()
        .any(|specimen_id| specimen_id.trim().is_empty() || !unique_ids.insert(specimen_id))
    {
        return blocked(
            "specimen identities are blank or duplicated",
            rule.claim_ceiling,
        );
    }

    supported(rule.claim_ceiling)
}

fn read_fixture(path: &Path) -> anyhow::Result<Fixture> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read fixture {}", path.display()))?;
    serde_json::from_str(&content)
        .with_context(|| format!("failed to parse fixture {}", path.display()))
}

pub fn run(path: &Path) -> anyhow::Result<()> {
    let fixture = read_fixture(path)?;
    if fixture.schema_version != "plasma-claim-scope-fixture-v0.3" {
        bail!("unsupported fixture schema version: {}", fixture.schema_version);
    }

    let mut counts = [0usize; 3];
    println!("PLASMA-CLAIM-TYPE S0 — structural claim-scope audit");
    println!("Claim ceiling: method-scoped measurement only; no product qualification, design allowable, service-life, or standards-compliance claim.");

    for case in &fixture.cases {
        let evidence = ScopeEvidence::from(case);
        let decision = decide(&evidence);
        let index = match decision.disposition {
            Disposition::Supported => 0,
            Disposition::Blocked => 1,
            Disposition::Rejected => 2,
        };
        counts[index] += 1;
        println!(
            "{}: {} — {}",
            case.id,
            decision.disposition.as_str(),
            decision.reason
        );

        if decision.disposition.as_str() != case.expected_disposition {
            bail!(
                "fixture mismatch for {}: expected {}, got {}",
                case.id,
                case.expected_disposition,
                decision.disposition.as_str()
            );
        }

        if decision.disposition == Disposition::Supported {
            println!("  evidence kind: {}", case.evidence_kind);
            println!("  exact ceiling: {}", decision.claim_ceiling);
        }

        if case.rationale.trim().is_empty() {
            bail!("fixture case {} has an empty rationale", case.id);
        }
    }

    println!(
        "Fixture replay: {}/{} expected dispositions matched",
        fixture.cases.len(),
        fixture.cases.len()
    );
    println!(
        "SUPPORTED {} | BLOCKED {} | REJECTED {}",
        counts[0], counts[1], counts[2]
    );
    println!("Structural result only. No laboratory procedure, physical property, product or design is qualified.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str =
        include_str!("../../data/research/plasma_claim_scope_fixture_v0_3.json");

    #[test]
    fn frozen_fixture_matches_all_expected_dispositions() {
        let fixture: Fixture = serde_json::from_str(FIXTURE).expect("valid fixture JSON");
        assert_eq!(fixture.schema_version, "plasma-claim-scope-fixture-v0.3");
        assert!(!fixture.cases.is_empty());

        for case in &fixture.cases {
            let actual = decide(&ScopeEvidence::from(case)).disposition.as_str();
            assert_eq!(
                actual, case.expected_disposition,
                "disposition mismatch for {}",
                case.id
            );
        }
    }

    #[test]
    fn sha256_validation_is_strict() {
        assert!(is_sha256(Some(
            "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
        )));
        assert!(!is_sha256(Some(
            "0123456789ABCDEF0123456789abcdef0123456789abcdef0123456789abcdef"
        )));
        assert!(!is_sha256(Some("not-a-digest")));
        assert!(!is_sha256(None));
    }

    #[test]
    fn plastic_and_frp_method_families_are_registered() {
        let fixture: Fixture = serde_json::from_str(FIXTURE).expect("valid fixture JSON");
        for id in ["astm-d3163-rigid-plastic-comparative", "astm-d5868-frp-to-frp-comparative", "astm-d5868-frp-to-metal-comparative"] {
            let case = fixture.cases.iter().find(|case| case.id == id).expect("case exists");
            assert_eq!(decide(&ScopeEvidence::from(case)).disposition, Disposition::Supported, "{id}");
        }
    }

    #[test]
    fn preparation_and_conditioning_practices_are_not_bond_tests() {
        let fixture: Fixture = serde_json::from_str(FIXTURE).expect("valid fixture JSON");
        for id in ["astm-d2093-preparation-reference", "astm-d1151-conditioning-reference", "astm-d1183-conditioning-reference", "astm-d4896-interpretation-guide-reference"] {
            let case = fixture.cases.iter().find(|case| case.id == id).expect("case exists");
            assert_eq!(decide(&ScopeEvidence::from(case)).disposition, Disposition::Supported, "{id}");
            assert_ne!(case.evidence_kind, "measurement");
        }
    }

    #[test]
    fn thick_adherend_response_does_not_establish_substrate_adhesion() {
        let fixture: Fixture = serde_json::from_str(FIXTURE).expect("valid fixture JSON");
        let valid = fixture.cases.iter().find(|case| case.id == "astm-d5656-thick-adherend-response").expect("case exists");
        let overclaim = fixture.cases.iter().find(|case| case.id == "astm-d5656-substrate-adhesion-overclaim").expect("case exists");
        assert_eq!(decide(&ScopeEvidence::from(valid)).disposition, Disposition::Supported);
        assert_eq!(decide(&ScopeEvidence::from(overclaim)).disposition, Disposition::Rejected);
    }

    #[test]
    fn failure_mode_classification_is_not_failure_acceptance() {
        let fixture: Fixture = serde_json::from_str(FIXTURE).expect("valid fixture JSON");
        let valid = fixture.cases.iter().find(|case| case.id == "astm-d5573-frp-smc-failure-classification").expect("case exists");
        let overclaim = fixture.cases.iter().find(|case| case.id == "astm-d5573-failure-acceptability-overclaim").expect("case exists");
        assert_eq!(decide(&ScopeEvidence::from(valid)).disposition, Disposition::Supported);
        assert_eq!(decide(&ScopeEvidence::from(overclaim)).disposition, Disposition::Rejected);
    }

    #[test]
    fn evidence_kind_mismatch_is_rejected() {
        let fixture: Fixture = serde_json::from_str(FIXTURE).expect("valid fixture JSON");
        let case = fixture.cases.iter().find(|case| case.id == "astm-d3163-evidence-kind-mismatch").expect("case exists");
        assert_eq!(decide(&ScopeEvidence::from(case)).disposition, Disposition::Rejected);
    }

    #[test]
    fn explicit_design_and_lifetime_escalations_are_rejected() {
        assert!(is_known_overclaim("design_allowable"));
        assert!(is_known_overclaim("service_lifetime"));
        assert!(is_known_overclaim("quantitative_fracture_strength"));
        assert!(is_known_overclaim("bond_strength_from_contact_angle"));
    }
}
