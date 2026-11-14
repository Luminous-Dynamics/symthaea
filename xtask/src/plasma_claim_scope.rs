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
struct Case {
    id: String,
    method_id: String,
    edition: String,
    material_pair_class: String,
    geometry: String,
    endpoint: String,
    requested_claim: String,
    protocol_id: Option<String>,
    protocol_sha256: Option<String>,
    raw_data_sha256: Option<String>,
    suitability_review_sha256: Option<String>,
    specimen_ids: Vec<String>,
    expected_disposition: String,
    rationale: String,
}

#[derive(Clone, Copy)]
struct MethodRule {
    method_id: &'static str,
    edition: &'static str,
    material_pair_class: &'static str,
    geometry: &'static str,
    endpoint: &'static str,
    allowed_claims: &'static [&'static str],
    claim_ceiling: &'static str,
}

const RULES: &[MethodRule] = &[
    MethodRule {
        method_id: "ISO_4587",
        edition: "2003",
        material_pair_class: "rigid_to_rigid",
        geometry: "single_lap",
        endpoint: "tensile_lap_shear_strength",
        allowed_claims: &["method_observation"],
        claim_ceiling: "tensile lap-shear measurement for the specified rigid-to-rigid bonded assembly and test conditions; not design information",
    },
    MethodRule {
        method_id: "ASTM_D1002",
        edition: "2026",
        material_pair_class: "metal_to_metal",
        geometry: "single_lap",
        endpoint: "apparent_shear_strength",
        allowed_claims: &["method_observation", "comparative_screening"],
        claim_ceiling: "apparent shear measurement / cautious comparative screening for the specified metal-to-metal single-lap specimen; not a structural design allowable",
    },
    MethodRule {
        method_id: "ASTM_D3762",
        edition: "2024",
        material_pair_class: "aluminum_to_aluminum",
        geometry: "wedge",
        endpoint: "qualitative_wedge_durability",
        allowed_claims: &["method_observation", "qualitative_surface_durability_screen"],
        claim_ceiling: "qualitative wedge-test surface-durability discrimination and failure-mode observation in the stated applicable specimen scope; not quantitative fracture strength",
    },
    MethodRule {
        method_id: "WATER_CONTACT_ANGLE_PROTOCOL",
        edition: "protocol-defined-v1",
        material_pair_class: "material_surface",
        geometry: "sessile_drop",
        endpoint: "water_contact_angle",
        allowed_claims: &["wettability_observation"],
        claim_ceiling: "water-contact-angle / wettability observation under the identified protocol; not adhesive bond strength",
    },
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Disposition {
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

struct Decision {
    disposition: Disposition,
    reason: &'static str,
    claim_ceiling: &'static str,
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
    )
}

fn decide(case: &Case) -> Decision {
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

    if case.material_pair_class != rule.material_pair_class {
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
    if fixture.schema_version != "plasma-claim-scope-fixture-v0.1" {
        bail!("unsupported fixture schema version: {}", fixture.schema_version);
    }

    let mut counts = [0usize; 3];
    println!("PLASMA-CLAIM-TYPE S0 — structural claim-scope audit");
    println!("Claim ceiling: method-scoped measurement only; no product qualification, design allowable, service-life, or standards-compliance claim.");

    for case in &fixture.cases {
        let decision = decide(case);
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
        include_str!("../../data/research/plasma_claim_scope_fixture_v0_1.json");

    #[test]
    fn frozen_fixture_matches_all_expected_dispositions() {
        let fixture: Fixture = serde_json::from_str(FIXTURE).expect("valid fixture JSON");
        assert_eq!(fixture.schema_version, "plasma-claim-scope-fixture-v0.1");
        assert!(!fixture.cases.is_empty());

        for case in &fixture.cases {
            let actual = decide(case).disposition.as_str();
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
    fn explicit_design_and_lifetime_escalations_are_rejected() {
        assert!(is_known_overclaim("design_allowable"));
        assert!(is_known_overclaim("service_lifetime"));
        assert!(is_known_overclaim("quantitative_fracture_strength"));
        assert!(is_known_overclaim("bond_strength_from_contact_angle"));
    }
}
