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
    baseline_evidence: Vec<EvidenceBinding>,
    candidate_evidence: Vec<EvidenceBinding>,
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
pub(super) enum Disposition {
    Pass,
    Blocked,
    Rejected,
}

impl Disposition {
    pub(super) fn as_str(self) -> &'static str {
        match self {
            Self::Pass => "PASS",
            Self::Blocked => "BLOCKED",
            Self::Rejected => "REJECTED",
        }
    }

    fn into_decision(self, reason: &'static str) -> Decision {
        Decision { disposition: self, reason }
    }
}

pub(super) struct Decision {
    pub(super) disposition: Disposition,
    pub(super) reason: &'static str,
}

type BindingMap<'a> = HashMap<(String, String), &'a EvidenceBinding>;

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

fn family_matches_profile(
    claim: &ClaimRequirement,
    content: &ProfileContent,
) -> bool {
    match claim.claim_family.as_str() {
        "joint_strength" => claim.roles.iter().all(|role| {
            matches!(
                role.method_id.as_str(),
                "ISO_4587" | "ASTM_D1002" | "ASTM_D3163" | "ASTM_D5868"
            ) && role.evidence_kind == "measurement"
                && role.material_pair_class == content.material_pair_class
                && role.geometry == content.joint_geometry
        }),
        "failure_pattern" => claim.roles.iter().all(|role| {
            matches!(role.method_id.as_str(), "ASTM_D5573" | "ISO_10365")
                && role.evidence_kind == "classification_practice"
                && role.geometry == "failure_surface_classification"
                && (role.method_id != "ASTM_D5573"
                    || content.material_pair_class == "frp_smc_joint")
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
    }
}

fn validate_profile_definition(content: &ProfileContent) -> std::result::Result<(), Disposition> {
    if !nonempty(&content.profile_id)
        || !nonempty(&content.material_pair_class)
        || !nonempty(&content.joint_geometry)
        || !nonempty(&content.intended_use)
    {
        return Err(Disposition::Blocked);
    }
    if content.claim_requirements.is_empty() {
        return Err(Disposition::Blocked);
    }

    let mut claim_ids = HashSet::new();
    for claim in &content.claim_requirements {
        if !nonempty(&claim.claim_id) || !nonempty(&claim.claim_family) || claim.roles.is_empty() {
            return Err(Disposition::Blocked);
        }
        if !claim_ids.insert(claim.claim_id.as_str()) {
            return Err(Disposition::Rejected);
        }
        if !family_matches_profile(claim, content) {
            return Err(Disposition::Rejected);
        }

        let mut role_ids = HashSet::new();
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
                return Err(Disposition::Blocked);
            }
            if !role_ids.insert(role.role_id.as_str()) {
                return Err(Disposition::Rejected);
            }
        }
    }
    Ok(())
}

fn audit_evidence_side<'a>(
    content: &ProfileContent,
    evidence: &'a [EvidenceBinding],
) -> std::result::Result<BindingMap<'a>, Disposition> {
    let claims: HashMap<&str, &ClaimRequirement> = content
        .claim_requirements
        .iter()
        .map(|claim| (claim.claim_id.as_str(), claim))
        .collect();
    let mut bindings = BindingMap::new();
    let mut evidence_ids = HashSet::new();
    let mut raw_digests = HashSet::new();

    for binding in evidence {
        let Some(claim) = claims.get(binding.claim_id.as_str()) else {
            return Err(Disposition::Rejected);
        };
        let Some(role) = claim.roles.iter().find(|role| role.role_id == binding.role_id) else {
            return Err(Disposition::Rejected);
        };
        let key = (binding.claim_id.clone(), binding.role_id.clone());
        if bindings.insert(key, binding).is_some() {
            return Err(Disposition::Rejected);
        }

        if !evidence_ids.insert(binding.evidence.id.as_str()) {
            return Err(Disposition::Rejected);
        }
        let Some(raw_digest) = binding.evidence.raw_data_sha256.as_deref() else {
            return Err(Disposition::Blocked);
        };
        if !sha256_shape(raw_digest) {
            return Err(Disposition::Blocked);
        }
        if !raw_digests.insert(raw_digest) {
            return Err(Disposition::Rejected);
        }
        if !same_scope(&binding.evidence, role) {
            return Err(Disposition::Rejected);
        }

        match decide(&binding.evidence).disposition {
            ScopeDisposition::Supported => {}
            ScopeDisposition::Blocked => return Err(Disposition::Blocked),
            ScopeDisposition::Rejected => return Err(Disposition::Rejected),
        }

        if claim.claim_family == "joint_strength"
            && (binding.evidence.material_pair_class != content.material_pair_class
                || binding.evidence.geometry != content.joint_geometry)
        {
            return Err(Disposition::Rejected);
        }
        if binding.evidence.method_id == "ASTM_D5573"
            && content.material_pair_class != "frp_smc_joint"
        {
            return Err(Disposition::Rejected);
        }
        if binding.evidence.method_id == "ASTM_D5656_D5656M"
            && claim.claim_family != "adhesive_constitutive_response"
        {
            return Err(Disposition::Rejected);
        }
    }

    for claim in &content.claim_requirements {
        let present = claim.roles.iter().any(|role| {
            bindings.contains_key(&(claim.claim_id.clone(), role.role_id.clone()))
        });
        if !present && !claim.required {
            continue;
        }
        if !present {
            return Err(Disposition::Blocked);
        }
        for role in &claim.roles {
            if role.required
                && !bindings.contains_key(&(claim.claim_id.clone(), role.role_id.clone()))
            {
                return Err(Disposition::Blocked);
            }
        }
        if claim.claim_family == "conditioned_measurement" {
            let linked: Vec<&EvidenceBinding> = claim
                .roles
                .iter()
                .filter_map(|role| {
                    bindings
                        .get(&(claim.claim_id.clone(), role.role_id.clone()))
                        .copied()
                })
                .collect();
            let has_conditioning = linked
                .iter()
                .any(|binding| binding.evidence.evidence_kind == "conditioning_practice");
            let has_measurement = linked
                .iter()
                .any(|binding| binding.evidence.evidence_kind == "measurement");
            if linked.len() < 2 || !has_conditioning || !has_measurement {
                return Err(Disposition::Blocked);
            }
            let first = &linked[0].evidence.specimen_ids;
            if first.is_empty() || !unique_nonempty(first) {
                return Err(Disposition::Blocked);
            }
            for binding in linked.iter().skip(1) {
                if binding.evidence.specimen_ids != *first {
                    return Err(Disposition::Rejected);
                }
            }
        }
    }

    Ok(bindings)
}

fn audit_profile(case: &Case) -> Decision {
    let blocked = |reason| Decision { disposition: Disposition::Blocked, reason };
    let rejected = |reason| Decision { disposition: Disposition::Rejected, reason };

    if !nonempty(&case.id) {
        return blocked("case identity is missing");
    }
    if !sha256_shape(&case.profile.profile_sha256) {
        return blocked("profile digest is missing or malformed");
    }
    let canonical = match serde_json::to_vec(&case.profile.content) {
        Ok(bytes) => bytes,
        Err(_) => return blocked("profile cannot be serialized for canonical digest verification"),
    };
    let actual_digest = format!("{:x}", Sha256::digest(canonical));
    if actual_digest != case.profile.profile_sha256 {
        return rejected("profile digest does not match the exact frozen claim contract");
    }
    if let Err(disposition) = validate_profile_definition(&case.profile.content) {
        return match disposition {
            Disposition::Blocked => blocked("profile claim definition or method-role declaration is incomplete"),
            Disposition::Rejected => rejected("profile claim family is incompatible with its method/evidence roles or material scope"),
            Disposition::Pass => unreachable!(),
        };
    }

    let baseline = match audit_evidence_side(&case.profile.content, &case.baseline_evidence) {
        Ok(bindings) => bindings,
        Err(Disposition::Blocked) => return blocked("baseline profile evidence is incomplete"),
        Err(Disposition::Rejected) => return rejected("baseline profile evidence violates its declared method/claim scope"),
        Err(Disposition::Pass) => unreachable!(),
    };
    let candidate = match audit_evidence_side(&case.profile.content, &case.candidate_evidence) {
        Ok(bindings) => bindings,
        Err(Disposition::Blocked) => return blocked("candidate profile evidence is incomplete"),
        Err(Disposition::Rejected) => return rejected("candidate profile evidence violates its declared method/claim scope"),
        Err(Disposition::Pass) => unreachable!(),
    };

    for claim in &case.profile.content.claim_requirements {
        for role in &claim.roles {
            let key = (claim.claim_id.clone(), role.role_id.clone());
            match (baseline.get(&key), candidate.get(&key)) {
                (Some(base), Some(candidate)) => {
                    let left = &base.evidence;
                    let right = &candidate.evidence;
                    if left.method_id != right.method_id
                        || left.edition != right.edition
                        || left.evidence_kind != right.evidence_kind
                        || left.material_pair_class != right.material_pair_class
                        || left.geometry != right.geometry
                        || left.endpoint != right.endpoint
                        || left.requested_claim != right.requested_claim
                        || left.protocol_id != right.protocol_id
                        || left.protocol_sha256 != right.protocol_sha256
                    {
                        return rejected("baseline and candidate evidence do not share the same method, edition, scope and protocol");
                    }
                    if left.id == right.id || left.raw_data_sha256 == right.raw_data_sha256 {
                        return rejected("baseline and candidate reuse one evidence receipt or raw-data artifact");
                    }
                    if left.specimen_ids.is_empty()
                        || right.specimen_ids.is_empty()
                        || !unique_nonempty(&left.specimen_ids)
                        || !unique_nonempty(&right.specimen_ids)
                    {
                        return blocked("baseline or candidate specimen identity list is incomplete");
                    }
                    let candidate_ids: HashSet<&str> =
                        right.specimen_ids.iter().map(String::as_str).collect();
                    if left
                        .specimen_ids
                        .iter()
                        .any(|specimen| candidate_ids.contains(specimen.as_str()))
                    {
                        return rejected("baseline and candidate use overlapping specimen identities");
                    }
                }
                (None, None) if !claim.required || !role.required => {},
                (None, None) => return blocked("required role is absent from both transfer cohorts"),
                _ => return rejected("baseline and candidate do not supply the same profile claim/role set"),
            }
        }
    }

    Decision {
        disposition: Disposition::Pass,
        reason: "baseline and candidate both satisfy the frozen profile claim matrix with exact, non-reused method evidence",
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
    if fixture.schema_version != "plasma-profile-contract-fixture-v0.2" {
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
            bail!("case {} has an empty rationale", case.rationale);
        }
    }
    println!("PLASMA-PROFILE-CONTRACT S0 — {} cases; PASS={} BLOCKED={} REJECTED={}", fixture.cases.len(), counts[0], counts[1], counts[2]);
    println!("Claim ceiling: profile-specific baseline/candidate method-claim consistency only; no physical qualification or production release is implied.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str =
        include_str!("../../data/research/plasma_profile_contract_fixture_v0_2.json");

    #[test]
    fn frozen_profile_fixture_matches_all_expected_dispositions() {
        let fixture: Fixture = serde_json::from_str(FIXTURE).expect("valid profile fixture");
        assert_eq!(fixture.schema_version, "plasma-profile-contract-fixture-v0.2");
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

    #[test]
    fn baseline_and_candidate_both_supply_same_roles_and_distinct_data() {
        let fixture: Fixture = serde_json::from_str(FIXTURE).expect("valid profile fixture");
        let case = fixture.cases.iter().find(|case| case.id == "valid-pp-profile-without-classification").expect("case exists");
        assert_eq!(audit_profile(case).disposition, Disposition::Pass);
    }

    #[test]
    fn candidate_cannot_omit_a_required_profile_role() {
        let fixture: Fixture = serde_json::from_str(FIXTURE).expect("valid profile fixture");
        let case = fixture.cases.iter().find(|case| case.id == "candidate-missing-required-role").expect("case exists");
        assert_eq!(audit_profile(case).disposition, Disposition::Blocked);
    }

    #[test]
    fn candidate_cannot_omit_a_required_profile_role() {
        let fixture: Fixture = serde_json::from_str(FIXTURE).expect("valid profile fixture");
        let case = fixture.cases.iter().find(|case| case.id == "candidate-missing-required-role").expect("case exists");
        assert_eq!(audit_profile(case).disposition, Disposition::Blocked);
    }

    #[test]
    fn baseline_and_candidate_both_supply_same_roles_and_distinct_data() {
        let fixture: Fixture = serde_json::from_str(FIXTURE).expect("valid profile fixture");
        let case = fixture.cases.iter().find(|case| case.id == "valid-pp-profile-without-classification").expect("case exists");
        assert_eq!(audit_profile(case).disposition, Disposition::Pass);
    }

    #[test]
    fn aged_strength_requires_same_specimen_cohort_for_conditioning_and_measurement() {
        let fixture: Fixture = serde_json::from_str(FIXTURE).expect("valid profile fixture");
        let case = fixture.cases.iter().find(|case| case.id == "durability-roles-require-same-specimens").expect("case exists");
        assert_eq!(audit_profile(case).disposition, Disposition::Pass);
    }
}
