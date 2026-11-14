use anyhow::{bail, Context, Result};
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::path::Path;

use crate::plasma_claim_scope::{decide, Disposition as ClaimDisposition, ScopeEvidence};
use crate::plasma_equivalence_design::{validate_bridge, Disposition as DesignDisposition, EquivalenceDesignReceipt};

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Fixture {
    schema_version: String,
    cases: Vec<CompositionCase>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Profile {
    material_pair_class: String,
    geometry: String,
    #[serde(default)]
    profile_sha256: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct EndpointEvidence {
    endpoint: String,
    baseline_specimen_ids: Vec<String>,
    candidate_specimen_ids: Vec<String>,
    baseline_receipt: ScopeEvidence,
    candidate_receipt: ScopeEvidence,
    equivalence_design: EquivalenceDesignReceipt,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct CompositionCase {
    id: String,
    baseline_profile: Profile,
    candidate_profile: Profile,
    required_endpoints: Vec<String>,
    endpoint_evidence: Vec<EndpointEvidence>,
    expected_disposition: String,
    rationale: String,
    #[serde(default)]
    transfer_case_id: Option<String>,
    #[serde(default)]
    transfer_bridge_id: Option<String>,
    #[serde(default)]
    bridge_dimension_keys: Option<Vec<String>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Disposition {
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

pub(super) struct Decision {
    pub(super) disposition: Disposition,
    pub(super) reason: &'static str,
}

fn unique_nonempty(values: &[String]) -> bool {
    let mut seen = HashSet::new();
    values
        .iter()
        .all(|value| !value.trim().is_empty() && seen.insert(value.as_str()))
}

fn same_members(left: &[String], right: &[String]) -> bool {
    if !unique_nonempty(left) || !unique_nonempty(right) || left.len() != right.len() {
        return false;
    }
    let left_set: HashSet<&str> = left.iter().map(String::as_str).collect();
    let right_set: HashSet<&str> = right.iter().map(String::as_str).collect();
    left_set == right_set
}

fn intersects(left: &[String], right: &[String]) -> bool {
    let right_set: HashSet<&str> = right.iter().map(String::as_str).collect();
    left.iter()
        .any(|value| right_set.contains(value.as_str()))
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn audit_case(case: &CompositionCase) -> Decision {
    let blocked = |reason| Decision {
        disposition: Disposition::Blocked,
        reason,
    };
    let rejected = |reason| Decision {
        disposition: Disposition::Rejected,
        reason,
    };
    let pass = || Decision {
        disposition: Disposition::Pass,
        reason: "every required endpoint is bound to baseline and candidate observations admitted by the shared method-scope evaluator",
    };

    if case.required_endpoints.is_empty() || !unique_nonempty(&case.required_endpoints) {
        return rejected("required endpoint set is empty or contains duplicate/blank entries");
    }

    if case.endpoint_evidence.is_empty() {
        return blocked("no method-qualified endpoint receipts were supplied");
    }

    let mut by_endpoint = HashMap::<&str, &EndpointEvidence>::new();
    let mut receipt_ids = HashSet::<String>::new();
    let mut artifact_digests = HashSet::<String>::new();

    for evidence in &case.endpoint_evidence {
        if evidence.endpoint.trim().is_empty() {
            return blocked("endpoint evidence has a blank endpoint identity");
        }
        if by_endpoint.insert(evidence.endpoint.as_str(), evidence).is_some() {
            return rejected("more than one bridge record claims the same endpoint identity");
        }
        if !receipt_ids.insert(evidence.baseline_receipt.id.clone())
            || !receipt_ids.insert(evidence.candidate_receipt.id.clone())
        {
            return rejected("a method-qualified observation receipt is reused across bridge endpoints or cohorts");
        }

        for receipt in [&evidence.baseline_receipt, &evidence.candidate_receipt] {
            let Some(digest) = receipt.raw_data_sha256.as_deref() else {
                return blocked("method-qualified receipt has no raw-data artifact digest");
            };
            if !is_sha256(digest) {
                return blocked("method-qualified receipt has a malformed raw-data artifact digest");
            }
            if !artifact_digests.insert(digest.to_ascii_lowercase()) {
                return rejected("raw-data artifact is reused across method-qualified observation receipts");
            }
        }
    }

    for extra in by_endpoint.keys() {
        if !case.required_endpoints.iter().any(|required| required == extra) {
            return rejected("bridge contains an endpoint not declared in the required endpoint set");
        }
    }

    let mut saw_blocked = false;
    for endpoint in &case.required_endpoints {
        let Some(evidence) = by_endpoint.get(endpoint.as_str()).copied() else {
            return blocked("a required transfer endpoint has no method-qualified evidence bridge");
        };

        if !unique_nonempty(&evidence.baseline_specimen_ids)
            || !unique_nonempty(&evidence.candidate_specimen_ids)
        {
            return blocked("baseline or candidate specimen identity list is empty, blank or duplicated");
        }
        if intersects(
            &evidence.baseline_specimen_ids,
            &evidence.candidate_specimen_ids,
        ) {
            return rejected("baseline and candidate specimen cohorts overlap");
        }

        if !same_members(
            &evidence.baseline_specimen_ids,
            &evidence.baseline_receipt.specimen_ids,
        ) || !same_members(
            &evidence.candidate_specimen_ids,
            &evidence.candidate_receipt.specimen_ids,
        ) {
            return rejected("method-scope receipt specimen identities do not exactly match the bridge evidence cohorts");
        }

        let baseline = &evidence.baseline_receipt;
        let candidate = &evidence.candidate_receipt;

        if evidence.endpoint != baseline.endpoint || evidence.endpoint != candidate.endpoint {
            return rejected("transfer endpoint name does not exactly match the endpoint measured by both method receipts");
        }

        for (profile, receipt, side) in [
            (&case.baseline_profile, baseline, "baseline"),
            (&case.candidate_profile, candidate, "candidate"),
        ] {
            if profile.material_pair_class != receipt.material_pair_class
                || profile.geometry != receipt.geometry
            {
                return rejected(if side == "baseline" {
                    "baseline profile material/geometry does not match its method-qualified observation"
                } else {
                    "candidate profile material/geometry does not match its method-qualified observation"
                });
            }
        }

        if baseline.method_id != candidate.method_id
            || baseline.edition != candidate.edition
            || baseline.evidence_kind != candidate.evidence_kind
            || baseline.endpoint != candidate.endpoint
            || baseline.requested_claim != candidate.requested_claim
        {
            return rejected("baseline and candidate observations do not use the same method/edition/evidence/endpoint/claim contract");
        }

        if baseline.material_pair_class != candidate.material_pair_class
            || baseline.geometry != candidate.geometry
        {
            return rejected("baseline and candidate material-pair class or specimen geometry differ; same method family does not establish cross-configuration equivalence");
        }

        for receipt in [baseline, candidate] {
            match decide(receipt).disposition {
                ClaimDisposition::Supported => {}
                ClaimDisposition::Blocked => saw_blocked = true,
                ClaimDisposition::Rejected => {
                    return rejected("the shared method-to-claim evaluator rejected a required endpoint receipt")
                }
            }
        }

        match (baseline.protocol_id.as_deref(), candidate.protocol_id.as_deref()) {
            (Some(left), Some(right)) if left != right => {
                return rejected("baseline and candidate measurement protocol identities differ");
            }
            (None, _) | (_, None) => saw_blocked = true,
            _ => {}
        }
        match (
            baseline.protocol_sha256.as_deref(),
            candidate.protocol_sha256.as_deref(),
        ) {
            (Some(left), Some(right)) if left != right => {
                return rejected("baseline and candidate measurement protocol digests differ");
            }
            (None, _) | (_, None) => saw_blocked = true,
            _ => {}
        }

        if baseline.id == candidate.id
            || baseline.raw_data_sha256 == candidate.raw_data_sha256
        {
            return rejected("baseline and candidate observation identities or raw-data artifacts are reused");
        }

        match validate_bridge(
            &evidence.equivalence_design,
            endpoint,
            baseline,
            candidate,
        ).disposition {
            DesignDisposition::Pass => {}
            DesignDisposition::Blocked => {
                saw_blocked = true;
            }
            DesignDisposition::Rejected => {
                return rejected("preregistered equivalence-design receipt is not bound to the exact endpoint observations or fails the study-design contract");
            }
        }
    }

    if saw_blocked {
        blocked("one or more required endpoint receipts are incomplete or outside the known registry; transfer remains blocked")
    } else {
        pass()
    }
}


pub(super) fn validate_transfer_case(
    path: &Path,
    composition_case_id: &str,
    transfer_case_id: &str,
    transfer_bridge_id: &str,
    baseline_profile_sha256: &str,
    candidate_profile_sha256: &str,
    material_pair_class: &str,
    geometry: &str,
    bridge_dimension_keys: &[String],
) -> Result<Decision> {
    let fixture = read_fixture(path)?;
    if fixture.schema_version != "plasma-transfer-composition-fixture-v0.1" {
        return Ok(Decision {
            disposition: Disposition::Blocked,
            reason: "transfer composition fixture schema is not the required version",
        });
    }
    let matches: Vec<&CompositionCase> = fixture.cases.iter()
        .filter(|case| case.id == composition_case_id)
        .collect();
    if matches.is_empty() {
        return Ok(Decision {
            disposition: Disposition::Blocked,
            reason: "transfer bridge references a missing method-qualified composition case",
        });
    }
    if matches.len() != 1 {
        return Ok(Decision {
            disposition: Disposition::Rejected,
            reason: "composition fixture contains duplicate case identities",
        });
    }
    let case = matches[0];
    let decision = audit_case(case);
    if decision.disposition != Disposition::Pass {
        return Ok(decision);
    }

    let Some(case_transfer_id) = case.transfer_case_id.as_deref() else {
        return Ok(Decision {
            disposition: Disposition::Blocked,
            reason: "composition case lacks transfer-case identity",
        });
    };
    let Some(case_bridge_id) = case.transfer_bridge_id.as_deref() else {
        return Ok(Decision {
            disposition: Disposition::Blocked,
            reason: "composition case lacks transfer-bridge identity",
        });
    };
    let Some(case_keys) = case.bridge_dimension_keys.as_deref() else {
        return Ok(Decision {
            disposition: Disposition::Blocked,
            reason: "composition case lacks an explicit bridge dimension set",
        });
    };
    let Some(base_digest) = case.baseline_profile.profile_sha256.as_deref() else {
        return Ok(Decision {
            disposition: Disposition::Blocked,
            reason: "composition case lacks exact baseline profile digest",
        });
    };
    let Some(candidate_digest) = case.candidate_profile.profile_sha256.as_deref() else {
        return Ok(Decision {
            disposition: Disposition::Blocked,
            reason: "composition case lacks exact candidate bridge profile digest",
        });
    };

    let mut actual_keys = case_keys.to_vec();
    let mut expected_keys = bridge_dimension_keys.to_vec();
    actual_keys.sort();
    expected_keys.sort();
    if case_transfer_id != transfer_case_id
        || case_bridge_id != transfer_bridge_id
        || base_digest != baseline_profile_sha256
        || candidate_digest != candidate_profile_sha256
        || case.baseline_profile.material_pair_class != material_pair_class
        || case.candidate_profile.material_pair_class != material_pair_class
        || case.baseline_profile.geometry != geometry
        || case.candidate_profile.geometry != geometry
        || actual_keys != expected_keys
    {
        return Ok(Decision {
            disposition: Disposition::Rejected,
            reason: "composition evidence is bound to a different transfer case, bridge dimension set or exact profile digest",
        });
    }

    for evidence in &case.endpoint_evidence {
        let design = crate::plasma_equivalence_design::validate_transfer_binding(
            &evidence.equivalence_design,
            transfer_case_id,
            transfer_bridge_id,
            baseline_profile_sha256,
            candidate_profile_sha256,
            bridge_dimension_keys,
            &evidence.endpoint,
            &evidence.baseline_receipt,
            &evidence.candidate_receipt,
        );
        if design.disposition != crate::plasma_equivalence_design::Disposition::Pass {
            return Ok(Decision {
                disposition: match design.disposition {
                    crate::plasma_equivalence_design::Disposition::Pass => Disposition::Pass,
                    crate::plasma_equivalence_design::Disposition::Blocked => Disposition::Blocked,
                    crate::plasma_equivalence_design::Disposition::Rejected => Disposition::Rejected,
                },
                reason: design.reason,
            });
        }
    }

    Ok(Decision {
        disposition: Disposition::Pass,
        reason: "method-qualified endpoint observations and their study designs match the exact transfer profile pair and bridge dimension set",
    })
}

fn read_fixture(path: &Path) -> Result<Fixture> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read fixture {}", path.display()))?;
    serde_json::from_str(&content)
        .with_context(|| format!("failed to parse fixture {}", path.display()))
}

pub fn run(path: &Path) -> Result<()> {
    let fixture = read_fixture(path)?;
    if !matches!(
        fixture.schema_version.as_str(),
        "plasma-evidence-composition-fixture-v0.2" | "plasma-transfer-composition-fixture-v0.1"
    ) {
        bail!("unsupported fixture schema version: {}", fixture.schema_version);
    }

    let mut counts = [0usize; 3];
    println!("PLASMA-EVIDENCE-COMPOSITION S0 — method-qualified transfer evidence audit");
    println!("Claim ceiling: specimen/endpoint/method-scope linkage only; no equivalence estimate, physical transfer, product qualification or release decision.");

    for case in &fixture.cases {
        let decision = audit_case(case);
        let index = match decision.disposition {
            Disposition::Pass => 0,
            Disposition::Blocked => 1,
            Disposition::Rejected => 2,
        };
        counts[index] += 1;
        println!("{}: {} — {}", case.id, decision.disposition.as_str(), decision.reason);

        if decision.disposition.as_str() != case.expected_disposition {
            bail!(
                "fixture mismatch for {}: expected {}, got {}",
                case.id,
                case.expected_disposition,
                decision.disposition.as_str()
            );
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
        "PASS {} | BLOCKED {} | REJECTED {}",
        counts[0], counts[1], counts[2]
    );
    println!("Structural composition only. Existing transfer interval, interaction and physical qualification gates remain independently required.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str =
        include_str!("../../data/research/plasma_evidence_composition_fixture_v0_2.json");

    #[test]
    fn frozen_composition_fixture_matches_all_expected_dispositions() {
        let fixture: Fixture = serde_json::from_str(FIXTURE).expect("valid composition fixture JSON");
        assert_eq!(fixture.schema_version, "plasma-evidence-composition-fixture-v0.2");
        assert!(!fixture.cases.is_empty());

        for case in &fixture.cases {
            assert_eq!(
                audit_case(case).disposition.as_str(),
                case.expected_disposition,
                "composition mismatch for {}",
                case.id
            );
        }
    }

    #[test]
    fn valid_transfer_endpoint_has_a_bound_preregistered_design_receipt() {
        let fixture: Fixture = serde_json::from_str(FIXTURE).expect("valid composition fixture JSON");
        let case = fixture.cases.iter()
            .find(|case| case.id == "valid-rigid-plastic-d3163")
            .expect("valid case exists");
        let evidence = &case.endpoint_evidence[0];
        let decision = validate_bridge(
            &evidence.equivalence_design,
            &evidence.endpoint,
            &evidence.baseline_receipt,
            &evidence.candidate_receipt,
        );
        assert_eq!(decision.disposition, DesignDisposition::Pass);
        assert_eq!(decision.equivalence, "EQUIVALENCE_ESTABLISHED");
    }

    #[test]
    fn design_receipt_cannot_be_substituted_from_another_endpoint_observation() {
        let fixture: Fixture = serde_json::from_str(FIXTURE).expect("valid composition fixture JSON");
        let valid = fixture.cases.iter()
            .find(|case| case.id == "valid-rigid-plastic-d3163")
            .expect("valid rigid-plastic case exists");
        let other = fixture.cases.iter()
            .find(|case| case.id == "valid-frp-to-frp-d5868")
            .expect("valid FRP case exists");
        let evidence = &valid.endpoint_evidence[0];
        let other_baseline = &other.endpoint_evidence[0].baseline_receipt;
        let decision = validate_bridge(
            &evidence.equivalence_design,
            &evidence.endpoint,
            other_baseline,
            &evidence.candidate_receipt,
        );
        assert_eq!(decision.disposition, DesignDisposition::Rejected);
    }

    #[test]
    fn wrong_test_method_cannot_supply_a_transfer_endpoint() {
        let fixture: Fixture = serde_json::from_str(FIXTURE).expect("valid composition fixture JSON");
        let case = fixture.cases.iter()
            .find(|case| case.id == "contact-angle-cannot-supply-lap-shear")
            .expect("case exists");
        assert_eq!(audit_case(case).disposition, Disposition::Rejected);
    }

    #[test]
    fn missing_method_receipt_blocks_transfer() {
        let fixture: Fixture = serde_json::from_str(FIXTURE).expect("valid composition fixture JSON");
        let case = fixture.cases.iter()
            .find(|case| case.id == "missing-required-endpoint-receipt")
            .expect("case exists");
        assert_eq!(audit_case(case).disposition, Disposition::Blocked);
    }

    #[test]
    fn same_standard_family_does_not_authorize_cross_material_pair_transfer() {
        let fixture: Fixture = serde_json::from_str(FIXTURE).expect("valid composition fixture JSON");
        let case = fixture.cases.iter()
            .find(|case| case.id == "frp-pair-class-change-rejects-transfer")
            .expect("case exists");
        assert_eq!(audit_case(case).disposition, Disposition::Rejected);
    }

    #[test]
    fn specimen_cohort_must_match_method_receipt_exactly() {
        let fixture: Fixture = serde_json::from_str(FIXTURE).expect("valid composition fixture JSON");
        let case = fixture.cases.iter()
            .find(|case| case.id == "bridge-specimens-do-not-match-receipt")
            .expect("case exists");
        assert_eq!(audit_case(case).disposition, Disposition::Rejected);
    }
}
