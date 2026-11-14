use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::path::Path;

use crate::plasma_claim_scope::{decide, Disposition as ScopeDisposition, ScopeEvidence};

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ProfileContent {
    profile_id: String,
    material_pair_class: String,
    joint_geometry: String,
    intended_use: String,
    claim_requirements: Vec<ClaimRequirement>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ClaimRequirement {
    claim_id: String,
    claim_family: String,
    required: bool,
    roles: Vec<RoleRequirement>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct RoleRequirement {
    role_id: String,
    required: bool,
    method_id: String,
    edition: String,
    evidence_kind: String,
    material_pair_class: String,
    geometry: String,
    endpoint: String,
    requested_claim: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Profile {
    content: ProfileContent,
    profile_sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct EvidenceBinding {
    claim_id: String,
    role_id: String,
    evidence: ScopeEvidence,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    id: String,
    profile: Profile,
    evidence: Vec<EvidenceBinding>,
    expected_disposition: String,
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

fn same_scope(actual: &ScopeEvidence, required: &RoleRequirement) -> bool {
    actual.method_id == required.method_id
        && actual.edition == required.edition
        && actual.evidence_kind == required.evidence_kind
        && actual.material_pair_class == required.material_pair_class
        && actual.geometry == required.geometry
        && actual.endpoint == required.endpoint
        && actual.requested_claim == required.requested_claim
}

fn unique_nonempty(values: &[String]) -> bool {
    let mut seen = HashSet::new();
    values.iter().all(|value| nonempty(value) && seen.insert(value.as_str()))
}

fn audit_profile(case: &Case) -> Decision {
    let blocked = |reason| Decision { disposition: Disposition::Blocked, reason };
    let rejected = |reason| Decision { disposition: Disposition::Rejected, reason };

    let profile = &case.profile;
    let content = &profile.content;
    if !nonempty(&case.id)
        || !nonempty(&content.profile_id)
        || !nonempty(&content.material_pair_class)
        || !nonempty(&content.joint_geometry)
        || !nonempty(&content.intended_use)
    {
        return blocked("profile or use-case identity is missing");
    }
    if !sha256_shape(&profile.profile_sha256) {
        return blocked("profile digest is missing or malformed");
    }
    let canonical = match serde_json::to_vec(content) {
        Ok(bytes) => bytes,
        Err(_) => return blocked("profile cannot be serialized for canonical digest verification"),
    };
    let actual_digest = format!("{:x}", Sha256::digest(canonical));
    if actual_digest != profile.profile_sha256 {
        return rejected("profile digest does not match the exact frozen claim contract");
    }
    if content.claim_requirements.is_empty() {
        return blocked("profile declares no qualified claims");
    }

    let mut claims = HashMap::<&str, &ClaimRequirement>::new();
    for claim in &content.claim_requirements {
        if !nonempty(&claim.claim_id) || !nonempty(&claim.claim_family) || claim.roles.is_empty() {
            return blocked("claim identity, family or method-role list is incomplete");
        }
        if claims.insert(claim.claim_id.as_str(), claim).is_some() {
            return rejected("profile repeats a claim ID");
        }
        let family_valid = match claim.claim_family.as_str() {
            "joint_strength" => claim.roles.iter().all(|role| {
                matches!(role.method_id.as_str(), "ISO_4587" | "ASTM_D1002" | "ASTM_D3163" | "ASTM_D5868")
                    && role.evidence_kind == "measurement"
                    && role.material_pair_class == content.material_pair_class
                    && role.geometry == content.joint_geometry
            }),
            "failure_pattern" => claim.roles.iter().all(|role| {
                matches!(role.method_id.as_str(), "ASTM_D5573" | "ISO_10365")
                    && role.evidence_kind == "classification_practice"
                    && role.geometry == "failure_surface_classification"
                    && (role.method_id != "ASTM_D5573" || content.material_pair_class == "frp_smc_joint")
            }),
            "adhesive_constitutive_response" => claim.roles.iter().all(|role| {
                role.method_id == "ASTM_D5656_D5656M"
                    && role.evidence_kind == "measurement"
                    && content.material_pair_class == "aluminum_thick_adherend_joint"
                    && content.joint_geometry == "thick_adherend_lap"
            }),
            "surface_state" => claim.roles.iter().all(|role| {
                role.method_id == "WATER_CONTACT_ANGLE_PROTOCOL"
                    && role.evidence_kind == "measurement"
                    && role.endpoint == "water_contact_angle"
            }),
            "conditioned_measurement" => {
                claim.roles.iter().any(|role| role.evidence_kind == "conditioning_practice")
                    && claim.roles.iter().any(|role| {
                        role.evidence_kind == "measurement"
                            && role.material_pair_class == content.material_pair_class
                            && role.geometry == content.joint_geometry
                    })
            }
            _ => false,
        };
        if !family_valid {
            return rejected("claim family is incompatible with its declared method/evidence roles or profile scope");
        }

        let mut roles = HashSet::new();
        for role in &claim.roles {
            if !nonempty(&role.role_id)
                || !nonempty(&role.method_id)
                || !nonempty(&role.edition)
                || !nonempty(&role.evidence_kind)
                || !nonempty(&role.material_pair_class)
                || !nonempty(&role.geometry)
                || !nonempty(&role.endpoint)
                || !nonempty(&role.requested_claim)
            {
                return blocked("method-role requirement is incomplete");
            }
            if !roles.insert(role.role_id.as_str()) {
                return rejected("claim repeats a method-role ID");
            }
        }
    }

    let mut bindings = HashMap::<(&str, &str), &EvidenceBinding>::new();
    let mut evidence_ids = HashSet::new();
    let mut raw_digests = HashSet::new();
    for binding in &case.evidence {
        let Some(claim) = claims.get(binding.claim_id.as_str()) else {
            return rejected("evidence is supplied for a claim not declared by the frozen profile");
        };
        let Some(role) = claim.roles.iter().find(|role| role.role_id == binding.role_id) else {
            return rejected("evidence is supplied for an undeclared method role");
        };
        if bindings.insert((binding.claim_id.as_str(), binding.role_id.as_str()), binding).is_some() {
            return rejected("duplicate evidence binding for the same claim and method role");
        }

        if !evidence_ids.insert(binding.evidence.id.as_str()) {
            return rejected("one method receipt identity is reused across profile claims");
        }
        let Some(raw_digest) = binding.evidence.raw_data_sha256.as_deref() else {
            return blocked("method receipt has no raw-data digest");
        };
        if !sha256_shape(raw_digest) {
            return blocked("method receipt raw-data digest is malformed");
        }
        if !raw_digests.insert(raw_digest) {
            return rejected("one raw-data artifact is reused to satisfy multiple claim roles");
        }
        if !same_scope(&binding.evidence, role) {
            return rejected("method receipt does not match the exact method/edition/evidence/material/geometry/endpoint claim role");
        }

        match decide(&binding.evidence).disposition {
            ScopeDisposition::Supported => {}
            ScopeDisposition::Blocked => return blocked("shared method-to-claim registry blocks the evidence role"),
            ScopeDisposition::Rejected => return rejected("shared method-to-claim registry rejects the evidence role"),
        }

        if claim.claim_family == "joint_strength"
            && (binding.evidence.material_pair_class != content.material_pair_class
                || binding.evidence.geometry != content.joint_geometry)
        {
            return rejected("joint-strength method scope does not match the profile material pair and joint geometry");
        }
        if binding.evidence.method_id == "ASTM_D5573"
            && content.material_pair_class != "frp_smc_joint"
        {
            return rejected("D5573 SMC-FRP classification cannot be attached to a non-SMC profile");
        }
        if binding.evidence.method_id == "ASTM_D5656_D5656M"
            && claim.claim_family != "adhesive_constitutive_response"
        {
            return rejected("D5656 constitutive response cannot satisfy a joint-strength or adhesion claim");
        }
    }

    for claim in &content.claim_requirements {
        let present = claim.roles.iter().any(|role| bindings.contains_key(&(claim.claim_id.as_str(), role.role_id.as_str())));
        if !present && !claim.required {
            continue;
        }
        if !present {
            return blocked("a required profile claim has no evidence roles");
        }
        for role in &claim.roles {
            if role.required && !bindings.contains_key(&(claim.claim_id.as_str(), role.role_id.as_str())) {
                return blocked("a required method role for an active claim is missing");
            }
        }
        if claim.claim_family == "conditioned_measurement" {
            let linked: Vec<&EvidenceBinding> = claim.roles.iter()
                .filter_map(|role| bindings.get(&(claim.claim_id.as_str(), role.role_id.as_str())).copied())
                .collect();
            if linked.len() < 2 {
                return blocked("conditioned outcome requires both conditioning and measurement evidence");
            }
            let first = &linked[0].evidence.specimen_ids;
            if first.is_empty() || !unique_nonempty(first) {
                return blocked("conditioned outcome specimen lineage is incomplete");
            }
            for binding in linked.iter().skip(1) {
                if binding.evidence.specimen_ids != *first {
                    return rejected("conditioning and post-conditioning measurement do not bind the exact same specimen cohort");
                }
            }
        }
    }

    Decision {
        disposition: Disposition::Pass,
        reason: "profile-specific claims and required method roles match the exact shared method registry; no irrelevant global endpoint list is imposed",
    }
}

impl Disposition {
    fn into_decision(self, reason: &'static str) -> Decision {
        Decision { disposition: self, reason }
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
    if fixture.schema_version != "plasma-profile-contract-fixture-v0.1" {
        bail!("unsupported fixture schema version: {}", fixture.schema_version);
    }
    let mut counts = [0usize; 3];
    for case in &fixture.cases {
        let result = audit_profile(case);
        let index = match result.disposition {
            Disposition::Pass => 0,
            Disposition::Blocked => 1,
            Disposition::Rejected => 2,
        };
        counts[index] += 1;
        println!("{}: {} — {}", case.id, result.disposition.as_str(), result.reason);
        if result.disposition.as_str() != case.expected_disposition {
            bail!("case {}: expected {}, got {}", case.id, case.expected_disposition, result.disposition.as_str());
        }
        if !nonempty(&case.rationale) {
            bail!("case {} has an empty rationale", case.id);
        }
    }
    println!("PLASMA-PROFILE-CONTRACT S0 — {} cases; PASS={} BLOCKED={} REJECTED={}", fixture.cases.len(), counts[0], counts[1], counts[2]);
    println!("Claim ceiling: profile-specific method/claim consistency only; no physical qualification or production release is implied.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str =
        include_str!("../../data/research/plasma_profile_contract_fixture_v0_1.json");

    #[test]
    fn frozen_profile_fixture_matches_all_expected_dispositions() {
        let fixture: Fixture = serde_json::from_str(FIXTURE).expect("valid profile fixture");
        assert_eq!(fixture.schema_version, "plasma-profile-contract-fixture-v0.1");
        for case in &fixture.cases {
            assert_eq!(audit_profile(case).disposition.as_str(), case.expected_disposition, "{}", case.id);
        }
    }

    #[test]
    fn optional_failure_classification_is_not_forced_on_polymer_profile() {
        let fixture: Fixture = serde_json::from_str(FIXTURE).expect("valid profile fixture");
        let case = fixture.cases.iter().find(|case| case.id == "valid-pp-profile-without-classification").expect("case exists");
        assert_eq!(audit_profile(case).disposition, Disposition::Pass);
    }

    #[test]
    fn constitutive_response_cannot_fill_joint_strength_claim() {
        let fixture: Fixture = serde_json::from_str(FIXTURE).expect("valid profile fixture");
        let case = fixture.cases.iter().find(|case| case.id == "d5656-cannot-fill-joint-strength").expect("case exists");
        assert_eq!(audit_profile(case).disposition, Disposition::Rejected);
    }

    fn rehash(profile: &mut Profile) {
        let bytes = serde_json::to_vec(&profile.content).expect("profile serializes");
        profile.profile_sha256 = format!("{:x}", Sha256::digest(bytes));
    }

    #[test]
    fn lap_shear_method_cannot_be_declared_as_failure_classification() {
        let fixture: Fixture = serde_json::from_str(FIXTURE).expect("valid profile fixture");
        let mut case = fixture.cases.into_iter().find(|case| case.id == "valid-pp-profile-without-classification").expect("case exists");
        let role = &mut case.profile.content.claim_requirements[1].roles[0];
        role.method_id = "ASTM_D3163".to_owned();
        role.edition = "D3163-01R23".to_owned();
        role.evidence_kind = "measurement".to_owned();
        role.material_pair_class = "rigid_plastic_pair".to_owned();
        role.geometry = "single_lap".to_owned();
        role.endpoint = "apparent_shear_strength".to_owned();
        role.requested_claim = "comparative_screening".to_owned();
        rehash(&mut case.profile);
        assert_eq!(audit_profile(&case).disposition, Disposition::Rejected);
    }

    #[test]
    fn d5573_cannot_be_declared_for_non_smc_profile_even_before_evidence_arrives() {
        let fixture: Fixture = serde_json::from_str(FIXTURE).expect("valid profile fixture");
        let mut case = fixture.cases.into_iter().find(|case| case.id == "valid-pp-profile-without-classification").expect("case exists");
        let role = &mut case.profile.content.claim_requirements[1].roles[0];
        role.method_id = "ASTM_D5573".to_owned();
        role.edition = "D5573-99R19".to_owned();
        role.evidence_kind = "classification_practice".to_owned();
        role.material_pair_class = "frp_smc_joint".to_owned();
        role.geometry = "failure_surface_classification".to_owned();
        role.endpoint = "frp_joint_failure_mode_classification".to_owned();
        role.requested_claim = "failure_mode_classification_reference".to_owned();
        rehash(&mut case.profile);
        assert_eq!(audit_profile(&case).disposition, Disposition::Rejected);
    }

    #[test]
    fn unknown_claim_family_is_rejected() {
        let fixture: Fixture = serde_json::from_str(FIXTURE).expect("valid profile fixture");
        let mut case = fixture.cases.into_iter().find(|case| case.id == "valid-pp-profile-without-classification").expect("case exists");
        case.profile.content.claim_requirements[0].claim_family = "quality_score".to_owned();
        rehash(&mut case.profile);
        assert_eq!(audit_profile(&case).disposition, Disposition::Rejected);
    }

    #[test]
    fn aged_strength_requires_same_specimen_cohort_for_conditioning_and_measurement() {
        let fixture: Fixture = serde_json::from_str(FIXTURE).expect("valid profile fixture");
        let case = fixture.cases.iter().find(|case| case.id == "durability-roles-require-same-specimens").expect("case exists");
        assert_eq!(audit_profile(case).disposition, Disposition::Pass);
    }
}
