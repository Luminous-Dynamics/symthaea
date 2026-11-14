use anyhow::{bail, Context, Result};
use serde::Deserialize;
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::path::Path;

use crate::plasma_claim_scope::{decide, Disposition as ClaimDisposition, ScopeEvidence};
use crate::plasma_equivalence_design::{matches_transfer_interval, matches_transfer_sample_mapping, validate_bridge, Disposition as DesignDisposition, EquivalenceDesignReceipt, SampleIdentity};

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Fixture {
    schema_version: String,
    cases: Vec<CompositionCase>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Profile {
    material_pair_class: String,
    geometry: String,
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
struct CompositionCase {
    id: String,
    baseline_profile: Profile,
    candidate_profile: Profile,
    required_endpoints: Vec<String>,
    endpoint_evidence: Vec<EndpointEvidence>,
    expected_disposition: String,
    rationale: String,
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

struct Decision {
    disposition: Disposition,
    reason: &'static str,
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

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FailurePatternShare {
    designation: String,
    percentage: f64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FailureClassificationRecord {
    specimen_id: String,
    status: String,
    patterns: Vec<FailurePatternShare>,
    oscillating: bool,
    reviewer_refs: Vec<String>,
    adjudication_ref: String,
    reason_ref: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FailureModeAnalysis {
    endpoint: String,
    target_pattern: String,
    grouping_scheme: String,
    baseline_sample_count: usize,
    candidate_sample_count: usize,
    baseline_cluster_count: usize,
    candidate_cluster_count: usize,
    baseline_fraction: f64,
    candidate_fraction: f64,
    difference_fraction: f64,
    margin: f64,
    lower_abs_difference_bound: f64,
    upper_abs_difference_bound: f64,
    confidence_level: f64,
    rounding_tolerance_percentage: f64,
    max_unclassified_rate: f64,
    baseline_unclassified_rate: f64,
    candidate_unclassified_rate: f64,
    analysis_profile: String,
    analysis_result_sha256: String,
    analysis_receipt_ref: String,
    claim_ceiling: String,
    acceptability_assessment: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct EmbeddedTransferComposition {
    transfer_bridge_artifact_sha256: String,
    composition: CompositionCase,
    failure_classification_receipt: ScopeEvidence,
    baseline_failure_classifications: Vec<FailureClassificationRecord>,
    candidate_failure_classifications: Vec<FailureClassificationRecord>,
    failure_mode_analysis: FailureModeAnalysis,
}

const ISO_10365_PATTERNS: &[&str] = &[
    "SF", "CSF", "DF", "CF", "SCF", "SWCF", "AF", "ACFP", "COR",
];

fn classification_share(
    records: &[FailureClassificationRecord],
    expected_ids: &[String],
    target_pattern: &str,
    rounding_tolerance: f64,
) -> std::result::Result<(f64, f64), Disposition> {
    if records.is_empty() || !unique_nonempty(expected_ids) {
        return Err(Disposition::Blocked);
    }
    let actual_ids: Vec<String> = records.iter().map(|record| record.specimen_id.clone()).collect();
    if !same_members(expected_ids, &actual_ids) {
        return Err(Disposition::Rejected);
    }

    let mut unclassified = 0usize;
    let mut classified_count = 0usize;
    let mut target_total = 0.0f64;
    for record in records {
        if !nonempty(&record.specimen_id)
            || !unique_nonempty(&record.reviewer_refs)
            || !nonempty(&record.adjudication_ref)
        {
            return Err(Disposition::Blocked);
        }
        match record.status.as_str() {
            "classified" => {
                if record.patterns.is_empty() || record.reason_ref.is_some() {
                    return Err(Disposition::Rejected);
                }
                if record.oscillating && record.patterns.len() < 2 {
                    return Err(Disposition::Rejected);
                }
                let mut seen = HashSet::new();
                let mut total = 0.0f64;
                let mut target = 0.0f64;
                for pattern in &record.patterns {
                    if !ISO_10365_PATTERNS.contains(&pattern.designation.as_str()) {
                        return Err(Disposition::Rejected);
                    }
                    if !seen.insert(pattern.designation.as_str())
                        || !pattern.percentage.is_finite()
                        || pattern.percentage <= 0.0
                        || pattern.percentage > 100.0
                    {
                        return Err(Disposition::Rejected);
                    }
                    total += pattern.percentage;
                    if pattern.designation == target_pattern {
                        target += pattern.percentage;
                    }
                }
                if (total - 100.0).abs() > rounding_tolerance {
                    return Err(Disposition::Rejected);
                }
                classified_count += 1;
                target_total += target;
            }
            "ambiguous" | "unclassifiable" => {
                if !record.patterns.is_empty()
                    || record.reason_ref.as_deref().is_none_or(str::is_empty)
                {
                    return Err(Disposition::Rejected);
                }
                unclassified += 1;
            }
            _ => return Err(Disposition::Rejected),
        }
    }
    if classified_count == 0 {
        return Err(Disposition::Blocked);
    }
    let fraction = target_total / (classified_count as f64 * 100.0);
    let unclassified_rate = unclassified as f64 / records.len() as f64;
    Ok((fraction, unclassified_rate))
}

fn validate_failure_mode_distribution(
    envelope: &EmbeddedTransferComposition,
    evidence: &Value,
    baseline_ids: &[String],
    candidate_ids: &[String],
    baseline_groups: &[String],
    candidate_groups: &[String],
) -> Disposition {
    let blocked = Disposition::Blocked;
    let rejected = Disposition::Rejected;
    let receipt = &envelope.failure_classification_receipt;
    if receipt.method_id != "ISO_10365"
        || receipt.edition != "ISO 10365:2022"
        || receipt.evidence_kind != "classification_practice"
        || receipt.material_pair_class != "bonded_assembly_any_adherends"
        || receipt.geometry != "failure_surface_classification"
        || receipt.endpoint != "adhesive_joint_failure_pattern"
        || receipt.requested_claim != "failure_pattern_classification_reference"
    {
        return rejected;
    }
    let mut expected_specimens = baseline_ids.to_vec();
    expected_specimens.extend(candidate_ids.iter().cloned());
    if !unique_nonempty(&expected_specimens)
        || !same_members(&expected_specimens, &receipt.specimen_ids)
    {
        return rejected;
    }
    match decide(receipt).disposition {
        ClaimDisposition::Supported => {}
        ClaimDisposition::Blocked => return blocked,
        ClaimDisposition::Rejected => return rejected,
    }

    let analysis = &envelope.failure_mode_analysis;
    if analysis.endpoint != "failure_mode_fraction"
        || analysis.grouping_scheme != "independent_arms"
        || analysis.analysis_profile != "cluster_aware_pattern_share_difference_v1"
        || analysis.claim_ceiling != "descriptive_distribution_only"
        || analysis.acceptability_assessment.is_some()
        || !is_sha256(&analysis.analysis_result_sha256)
        || !nonempty(&analysis.analysis_receipt_ref)
        || !ISO_10365_PATTERNS.contains(&analysis.target_pattern.as_str())
    {
        return rejected;
    }
    if analysis.baseline_sample_count != baseline_ids.len()
        || analysis.candidate_sample_count != candidate_ids.len()
        || analysis.baseline_cluster_count != baseline_groups.len()
        || analysis.candidate_cluster_count != candidate_groups.len()
    {
        return rejected;
    }

    let rounding = analysis.rounding_tolerance_percentage;
    if !rounding.is_finite() || !(0.0..=2.0).contains(&rounding)
        || !analysis.max_unclassified_rate.is_finite()
        || !(0.0..1.0).contains(&analysis.max_unclassified_rate)
    {
        return rejected;
    }
    let baseline = match classification_share(
        &envelope.baseline_failure_classifications,
        baseline_ids,
        &analysis.target_pattern,
        rounding,
    ) {
        Ok(value) => value,
        Err(disposition) => return disposition,
    };
    let candidate = match classification_share(
        &envelope.candidate_failure_classifications,
        candidate_ids,
        &analysis.target_pattern,
        rounding,
    ) {
        Ok(value) => value,
        Err(disposition) => return disposition,
    };

    if !analysis.baseline_fraction.is_finite()
        || !analysis.candidate_fraction.is_finite()
        || !analysis.difference_fraction.is_finite()
        || !analysis.margin.is_finite()
        || !analysis.lower_abs_difference_bound.is_finite()
        || !analysis.upper_abs_difference_bound.is_finite()
        || !analysis.confidence_level.is_finite()
        || !close(analysis.baseline_fraction, baseline.0)
        || !close(analysis.candidate_fraction, candidate.0)
        || !close(analysis.difference_fraction, (candidate.0 - baseline.0).abs())
        || !close(analysis.baseline_unclassified_rate, baseline.1)
        || !close(analysis.candidate_unclassified_rate, candidate.1)
        || baseline.1 > analysis.max_unclassified_rate
        || candidate.1 > analysis.max_unclassified_rate
    {
        return rejected;
    }
    if analysis.margin <= 0.0
        || analysis.lower_abs_difference_bound < 0.0
        || analysis.upper_abs_difference_bound < analysis.lower_abs_difference_bound
        || analysis.confidence_level < 0.0
        || analysis.confidence_level > 1.0
    {
        return rejected;
    }

    let Some(intervals) = evidence.get("intervals").and_then(Value::as_array) else {
        return blocked;
    };
    let Some(interval) = intervals.iter().find(|row| {
        row.get("endpoint").and_then(Value::as_str) == Some("failure_mode_fraction")
    }) else {
        return blocked;
    };
    let (Some(margin), Some(lower), Some(upper), Some(confidence)) = (
        interval.get("margin").and_then(Value::as_f64),
        interval.get("lower_abs_difference_bound").and_then(Value::as_f64),
        interval.get("upper_abs_difference_bound").and_then(Value::as_f64),
        interval.get("confidence_level").and_then(Value::as_f64),
    ) else {
        return blocked;
    };
    if !close(analysis.margin, margin)
        || !close(analysis.lower_abs_difference_bound, lower)
        || !close(analysis.upper_abs_difference_bound, upper)
        || !close(analysis.confidence_level, confidence)
        || analysis.difference_fraction < lower - rounding / 100.0
        || analysis.difference_fraction > upper + rounding / 100.0
    {
        return rejected;
    }
    Disposition::Pass
}

pub(super) fn audit_transfer_bridge(
    embedded: &Value,
    bridge: &Value,
    material_pair_class: &str,
    geometry: &str,
) -> Disposition {
    let blocked = Disposition::Blocked;
    let rejected = Disposition::Rejected;
    let Ok(envelope) = serde_json::from_value::<EmbeddedTransferComposition>(embedded.clone()) else {
        return blocked;
    };
    let Some(evidence) = bridge.get("evidence") else {
        return blocked;
    };
    let Some(artifact_sha) = evidence.get("artifact_sha256").and_then(Value::as_str) else {
        return blocked;
    };
    if !is_sha256(artifact_sha) || !is_sha256(&envelope.transfer_bridge_artifact_sha256) {
        return blocked;
    }
    if artifact_sha != envelope.transfer_bridge_artifact_sha256 {
        return rejected;
    }

    let Some(baseline_ids) = string_list(evidence.get("baseline_specimen_ids")) else {
        return blocked;
    };
    let Some(candidate_ids) = string_list(evidence.get("candidate_specimen_ids")) else {
        return blocked;
    };
    let Some(baseline_groups) = string_list(evidence.get("baseline_independence_groups")) else {
        return blocked;
    };
    let Some(candidate_groups) = string_list(evidence.get("candidate_independence_groups")) else {
        return blocked;
    };
    let Some(baseline_samples_value) = evidence.get("baseline_sample_clusters") else {
        return blocked;
    };
    let Some(candidate_samples_value) = evidence.get("candidate_sample_clusters") else {
        return blocked;
    };
    let Ok(baseline_samples) = serde_json::from_value::<Vec<SampleIdentity>>(baseline_samples_value.clone()) else {
        return blocked;
    };
    let Ok(candidate_samples) = serde_json::from_value::<Vec<SampleIdentity>>(candidate_samples_value.clone()) else {
        return blocked;
    };

    if !unique_nonempty(&baseline_ids)
        || !unique_nonempty(&candidate_ids)
        || !unique_nonempty(&baseline_groups)
        || !unique_nonempty(&candidate_groups)
    {
        return rejected;
    }
    let baseline_sample_ids: Vec<String> = baseline_samples.iter().map(|s| s.specimen_id.clone()).collect();
    let candidate_sample_ids: Vec<String> = candidate_samples.iter().map(|s| s.specimen_id.clone()).collect();
    let baseline_sample_groups: Vec<String> = baseline_samples.iter().map(|s| s.cluster_id.clone()).collect();
    let candidate_sample_groups: Vec<String> = candidate_samples.iter().map(|s| s.cluster_id.clone()).collect();
    if !same_members(&baseline_ids, &baseline_sample_ids)
        || !same_members(&candidate_ids, &candidate_sample_ids)
        || !same_members(&baseline_groups, &baseline_sample_groups)
        || !same_members(&candidate_groups, &candidate_sample_groups)
    {
        return rejected;
    }

    if envelope.composition.baseline_profile.material_pair_class != material_pair_class
        || envelope.composition.candidate_profile.material_pair_class != material_pair_class
        || envelope.composition.baseline_profile.geometry != geometry
        || envelope.composition.candidate_profile.geometry != geometry
    {
        return rejected;
    }

    let intervals = evidence.get("intervals").and_then(Value::as_array);
    let Some(strength_interval) = intervals.and_then(|rows| rows.iter().find(|row| {
        row.get("endpoint").and_then(Value::as_str) == Some("initial_lap_shear_mpa")
    })) else {
        return blocked;
    };

    if envelope.composition.endpoint_evidence.len() != 1 {
        return rejected;
    }
    let endpoint = &envelope.composition.endpoint_evidence[0];
    if endpoint.endpoint != "apparent_shear_strength"
        || !same_members(&baseline_ids, &endpoint.baseline_specimen_ids)
        || !same_members(&candidate_ids, &endpoint.candidate_specimen_ids)
    {
        return rejected;
    }
    if !matches_transfer_sample_mapping(
        &endpoint.equivalence_design,
        &baseline_samples,
        &candidate_samples,
    ) || !matches_transfer_interval(&endpoint.equivalence_design, strength_interval)
    {
        return rejected;
    }

    let composition = audit_case(&envelope.composition).disposition;
    if composition != Disposition::Pass {
        return composition;
    }
    validate_failure_mode_distribution(
        &envelope,
        evidence,
        &baseline_ids,
        &candidate_ids,
        &baseline_groups,
        &candidate_groups,
    )
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

fn read_fixture(path: &Path) -> Result<Fixture> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read fixture {}", path.display()))?;
    serde_json::from_str(&content)
        .with_context(|| format!("failed to parse fixture {}", path.display()))
}

pub fn run(path: &Path) -> Result<()> {
    let fixture = read_fixture(path)?;
    if fixture.schema_version != "plasma-evidence-composition-fixture-v0.2" {
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
