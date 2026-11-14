use anyhow::{bail, Context, Result};
use serde::Deserialize;
use std::collections::HashSet;
use std::path::Path;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SpecimenObservation {
    specimen_id: String,
    independent_group_id: String,
    material_pair_class: String,
    joint_geometry: String,
    truth_state: String,
    defect_class: Option<String>,
    defect_size_mm: Option<f64>,
    truth_method_id: String,
    truth_protocol_sha256: String,
    truth_artifact_sha256: String,
    truth_adjudication_ref: String,
    blinded_to_truth: bool,
    scan_coverage_fraction: f64,
    indication: bool,
    repeated_scan_count: usize,
    raw_signal_sha256: String,
    inspection_protocol_sha256: String,
    sensor_setup_sha256: String,
    decision_threshold_sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AnalysisPlan {
    plan_id: String,
    plan_sha256: String,
    profile_sha256: String,
    material_pair_class: String,
    joint_geometry: String,
    defect_class: String,
    defect_size_metric: String,
    inspection_protocol_sha256: String,
    sensor_setup_sha256: String,
    calibration_receipt_sha256: String,
    decision_threshold_sha256: String,
    scan_plan_sha256: String,
    ground_truth_method_id: String,
    ground_truth_protocol_sha256: String,
    analysis_profile: String,
    analysis_method_ref: String,
    independent_unit: String,
    minimum_defect_specimens: usize,
    minimum_sound_specimens: usize,
    minimum_defect_clusters: usize,
    minimum_sound_clusters: usize,
    minimum_scan_coverage: f64,
    critical_defect_size_mm: f64,
    target_pod: f64,
    confidence_level: f64,
    maximum_false_call_rate: f64,
    frozen_at: i64,
    first_outcome_available_at: i64,
    decision_threshold_frozen_at: i64,
    threshold_changed_after_unblind: bool,
    claim_ceiling: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AnalysisResult {
    result_id: String,
    result_sha256: String,
    plan_sha256: String,
    profile_sha256: String,
    specimen_manifest_sha256: String,
    inspection_protocol_sha256: String,
    sensor_setup_sha256: String,
    calibration_receipt_sha256: String,
    decision_threshold_sha256: String,
    scan_plan_sha256: String,
    defect_size_metric: String,
    critical_defect_size_mm: f64,
    analysis_profile: String,
    defect_class: String,
    defect_specimen_count: usize,
    sound_specimen_count: usize,
    defect_cluster_count: usize,
    sound_cluster_count: usize,
    hit_count: usize,
    miss_count: usize,
    false_call_count: usize,
    a90_95_size_mm: f64,
    false_call_rate_upper: f64,
    target_pod: f64,
    confidence_level: f64,
    outcome: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Study {
    study_id: String,
    profile_id: String,
    profile_sha256: String,
    material_pair_class: String,
    joint_geometry: String,
    adhesive_system: String,
    environment_profile_id: String,
    defect_class: String,
    defect_size_metric: String,
    ground_truth_method_id: String,
    ground_truth_protocol_sha256: String,
    inspection_method_id: String,
    inspection_protocol_id: String,
    inspection_protocol_sha256: String,
    equipment_id: String,
    sensor_setup_sha256: String,
    calibration_receipt_sha256: String,
    decision_threshold_sha256: String,
    decision_threshold: f64,
    scan_plan_sha256: String,
    analysis_plan: AnalysisPlan,
    specimens: Vec<SpecimenObservation>,
    specimen_manifest_sha256: String,
    analysis_result: AnalysisResult,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    id: String,
    study: Study,
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

fn allowed_defect_class(value: &str) -> bool {
    matches!(
        value,
        "void" | "disbond" | "weak_adhesion" | "kissing_bond" | "edge_delamination"
    )
}

fn audit_study(study: &Study) -> Decision {
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

    for value in [
        study.study_id.as_str(),
        study.profile_id.as_str(),
        study.environment_profile_id.as_str(),
        study.adhesive_system.as_str(),
        study.material_pair_class.as_str(),
        study.joint_geometry.as_str(),
        study.defect_class.as_str(),
        study.defect_size_metric.as_str(),
        study.inspection_method_id.as_str(),
        study.inspection_protocol_id.as_str(),
        study.ground_truth_method_id.as_str(),
        study.equipment_id.as_str(),
    ] {
        if !nonempty(value) {
            return blocked("study, material, defect, equipment or protocol identity is missing");
        }
    }
    if !allowed_defect_class(&study.defect_class) {
        return blocked("defect family is not registered in this S0 POD vocabulary");
    }
    for digest in [
        study.profile_sha256.as_str(),
        study.ground_truth_protocol_sha256.as_str(),
        study.inspection_protocol_sha256.as_str(),
        study.sensor_setup_sha256.as_str(),
        study.calibration_receipt_sha256.as_str(),
        study.decision_threshold_sha256.as_str(),
        study.scan_plan_sha256.as_str(),
        study.specimen_manifest_sha256.as_str(),
    ] {
        if !sha256_shape(digest) {
            return blocked("profile, protocol, sensor, calibration, threshold, scan-plan or manifest digest is missing or malformed");
        }
    }
    if !study.decision_threshold.is_finite() {
        return rejected("decision threshold is non-finite");
    }

    let plan = &study.analysis_plan;
    if ![
        plan.plan_id.as_str(),
        plan.analysis_method_ref.as_str(),
        plan.claim_ceiling.as_str(),
        plan.material_pair_class.as_str(),
        plan.joint_geometry.as_str(),
        plan.defect_class.as_str(),
        plan.defect_size_metric.as_str(),
        plan.ground_truth_method_id.as_str(),
    ]
    .iter()
    .all(|value| nonempty(value))
        || !sha256_shape(&plan.plan_sha256)
        || !sha256_shape(&plan.profile_sha256)
        || !sha256_shape(&plan.inspection_protocol_sha256)
        || !sha256_shape(&plan.sensor_setup_sha256)
        || !sha256_shape(&plan.calibration_receipt_sha256)
        || !sha256_shape(&plan.decision_threshold_sha256)
        || !sha256_shape(&plan.scan_plan_sha256)
        || !sha256_shape(&plan.ground_truth_protocol_sha256)
    {
        return blocked("analysis plan identity, profile, inspection or ground-truth digest is missing");
    }
    if plan.profile_sha256 != study.profile_sha256
        || plan.material_pair_class != study.material_pair_class
        || plan.joint_geometry != study.joint_geometry
        || plan.defect_class != study.defect_class
        || plan.defect_size_metric != study.defect_size_metric
        || plan.inspection_protocol_sha256 != study.inspection_protocol_sha256
        || plan.sensor_setup_sha256 != study.sensor_setup_sha256
        || plan.calibration_receipt_sha256 != study.calibration_receipt_sha256
        || plan.decision_threshold_sha256 != study.decision_threshold_sha256
        || plan.scan_plan_sha256 != study.scan_plan_sha256
        || plan.ground_truth_method_id != study.ground_truth_method_id
        || plan.ground_truth_protocol_sha256 != study.ground_truth_protocol_sha256
    {
        return rejected("analysis plan does not bind the exact profile, inspection, calibration, threshold, scan and ground-truth configuration");
    }
    if plan.analysis_profile != "hit_miss_pod_v1" {
        return blocked("only the registered hit/miss POD profile is supported by this audit");
    }
    if plan.independent_unit != "manufacturing_lot"
        && plan.independent_unit != "production_line"
        && plan.independent_unit != "manufacturing_site"
    {
        return blocked("independent unit must be a manufacturing lot, production line or site");
    }
    if plan.claim_ceiling != "defect_detection_only" {
        return rejected("POD receipt claim ceiling cannot include strength, acceptability or safety");
    }
    if plan.frozen_at >= plan.first_outcome_available_at
        || plan.decision_threshold_frozen_at > plan.first_outcome_available_at
        || plan.threshold_changed_after_unblind
    {
        return rejected("analysis plan or decision threshold was frozen/changed after outcome availability");
    }
    if plan.minimum_defect_specimens == 0
        || plan.minimum_sound_specimens == 0
        || plan.minimum_defect_clusters == 0
        || plan.minimum_sound_clusters == 0
    {
        return rejected("planned defect, sound-control and independent-cluster minima must be positive");
    }
    if !plan.minimum_scan_coverage.is_finite()
        || !(0.0..=1.0).contains(&plan.minimum_scan_coverage)
        || !plan.critical_defect_size_mm.is_finite()
        || plan.critical_defect_size_mm <= 0.0
        || (plan.target_pod - 0.90).abs() > 1e-9
        || (plan.confidence_level - 0.95).abs() > 1e-9
        || !plan.maximum_false_call_rate.is_finite()
        || !(0.0..=1.0).contains(&plan.maximum_false_call_rate)
    {
        return rejected("critical size, scan coverage, false-call limit or 90/95 target is invalid");
    }

    if study.specimens.is_empty() {
        return blocked("no inspection specimens are supplied");
    }
    let mut specimen_ids = HashSet::new();
    let mut signal_digests = HashSet::new();
    let mut truth_digests = HashSet::new();
    let mut defect_clusters = HashSet::new();
    let mut sound_clusters = HashSet::new();
    let mut defect_count = 0usize;
    let mut sound_count = 0usize;
    let mut hits = 0usize;
    let mut misses = 0usize;
    let mut false_calls = 0usize;
    let mut sound_no_calls = 0usize;

    for specimen in &study.specimens {
        if !nonempty(&specimen.specimen_id) || !nonempty(&specimen.independent_group_id) {
            return blocked("specimen or independent-group identity is missing");
        }
        if !specimen_ids.insert(specimen.specimen_id.as_str()) {
            return rejected("specimen appears more than once; repeated scans cannot inflate independent sample count");
        }
        if specimen.material_pair_class != study.material_pair_class
            || specimen.joint_geometry != study.joint_geometry
        {
            return rejected("specimen material pair or joint geometry differs from the qualified profile");
        }
        if !sha256_shape(&specimen.truth_artifact_sha256)
            || !sha256_shape(&specimen.raw_signal_sha256)
            || !sha256_shape(&specimen.inspection_protocol_sha256)
            || !sha256_shape(&specimen.sensor_setup_sha256)
            || !sha256_shape(&specimen.decision_threshold_sha256)
        {
            return blocked("specimen truth, waveform, protocol, sensor or threshold digest is missing");
        }
        if !truth_digests.insert(specimen.truth_artifact_sha256.as_str())
            || !signal_digests.insert(specimen.raw_signal_sha256.as_str())
        {
            return rejected("truth or raw signal artifact is reused across specimen records");
        }
        if specimen.inspection_protocol_sha256 != study.inspection_protocol_sha256
            || specimen.sensor_setup_sha256 != study.sensor_setup_sha256
            || specimen.decision_threshold_sha256 != study.decision_threshold_sha256
        {
            return rejected("specimen was inspected under a different protocol, sensor setup or decision threshold");
        }
        if !specimen.blinded_to_truth {
            return blocked("inspection decision was not blinded to known truth");
        }
        if specimen.repeated_scan_count == 0 {
            return blocked("specimen has no recorded scan");
        }
        if !specimen.scan_coverage_fraction.is_finite()
            || !(0.0..=1.0).contains(&specimen.scan_coverage_fraction)
        {
            return rejected("scan coverage fraction is invalid");
        }
        if specimen.scan_coverage_fraction < plan.minimum_scan_coverage {
            return blocked("scan coverage is below the preregistered minimum");
        }
        if specimen.truth_method_id == study.inspection_method_id {
            return rejected("known truth cannot be established by the same inspection method being evaluated");
        }
        if !nonempty(&specimen.truth_method_id)
            || !nonempty(&specimen.truth_adjudication_ref)
            || !sha256_shape(&specimen.truth_protocol_sha256)
        {
            return blocked("independent ground-truth method, protocol digest or adjudication reference is missing");
        }
        if specimen.truth_method_id != study.ground_truth_method_id
            || specimen.truth_protocol_sha256 != study.ground_truth_protocol_sha256
        {
            return rejected("specimen ground-truth method or protocol differs from the preregistered truth reference");
        }

        match specimen.truth_state.as_str() {
            "defect_confirmed" => {
                if specimen.defect_class.as_deref() != Some(study.defect_class.as_str()) {
                    return Disposition::Rejected.into_decision("specimen defect class differs from the single defect family qualified by this study");
                }
                let Some(size) = specimen.defect_size_mm else {
                    return blocked("confirmed defect has no measured size");
                };
                if !size.is_finite() || size <= 0.0 {
                    return rejected("defect size is non-finite or non-positive");
                }
                if !matches!(
                    specimen.truth_method_id.as_str(),
                    "independent_destructive_reference" | "independent_complementary_metrology"
                ) {
                    return blocked("defect truth is not independently confirmed; seeded design alone is insufficient");
                }
                defect_count += 1;
                defect_clusters.insert(specimen.independent_group_id.as_str());
                if specimen.indication {
                    hits += 1;
                } else {
                    misses += 1;
                }
            }
            "sound_confirmed" => {
                if specimen.defect_class.is_some() || specimen.defect_size_mm.is_some() {
                    return rejected("sound control contains a defect class or defect size");
                }
                if !matches!(
                    specimen.truth_method_id.as_str(),
                    "independent_destructive_reference" | "independent_complementary_metrology"
                ) {
                    return blocked("sound-control truth is not independently confirmed");
                }
                sound_count += 1;
                sound_clusters.insert(specimen.independent_group_id.as_str());
                if specimen.indication {
                    false_calls += 1;
                } else {
                    sound_no_calls += 1;
                }
            }
            "truth_uncertain" => {
                return blocked("uncertain ground truth cannot be counted as a POD hit, miss or sound control");
            }
            _ => return blocked("unknown truth state"),
        }
    }

    if defect_count < plan.minimum_defect_specimens
        || sound_count < plan.minimum_sound_specimens
        || defect_clusters.len() < plan.minimum_defect_clusters
        || sound_clusters.len() < plan.minimum_sound_clusters
    {
        return blocked("actual defect/sound specimen or independent-cluster counts are below the preregistered minimum");
    }
    if hits == 0 || misses == 0 || sound_no_calls == 0 {
        return blocked("hit/miss study needs detected and missed defects plus confirmed sound no-call controls");
    }

    let result = &study.analysis_result;
    if !nonempty(&result.result_id)
        || !sha256_shape(&result.result_sha256)
        || !sha256_shape(&result.specimen_manifest_sha256)
    {
        return blocked("POD analysis result identity or digest is missing");
    }
    if result.plan_sha256 != plan.plan_sha256
        || result.profile_sha256 != study.profile_sha256
        || result.specimen_manifest_sha256 != study.specimen_manifest_sha256
        || result.inspection_protocol_sha256 != study.inspection_protocol_sha256
        || result.sensor_setup_sha256 != study.sensor_setup_sha256
        || result.calibration_receipt_sha256 != study.calibration_receipt_sha256
        || result.decision_threshold_sha256 != study.decision_threshold_sha256
        || result.scan_plan_sha256 != study.scan_plan_sha256
        || result.defect_size_metric != study.defect_size_metric
        || (result.critical_defect_size_mm - plan.critical_defect_size_mm).abs() > 1e-9
        || result.analysis_profile != plan.analysis_profile
        || result.defect_class != study.defect_class
        || result.defect_specimen_count != defect_count
        || result.sound_specimen_count != sound_count
        || result.defect_cluster_count != defect_clusters.len()
        || result.sound_cluster_count != sound_clusters.len()
        || result.hit_count != hits
        || result.miss_count != misses
        || result.false_call_count != false_calls
        || (result.target_pod - plan.target_pod).abs() > 1e-9
        || (result.confidence_level - plan.confidence_level).abs() > 1e-9
    {
        return rejected("POD result does not bind the exact plan, defect family, specimen manifest, hit/miss counts or cluster counts");
    }
    if !result.a90_95_size_mm.is_finite()
        || result.a90_95_size_mm <= 0.0
        || !result.false_call_rate_upper.is_finite()
        || !(0.0..=1.0).contains(&result.false_call_rate_upper)
    {
        return rejected("a90/95 size or false-call upper bound is invalid");
    }

    let established = result.a90_95_size_mm <= plan.critical_defect_size_mm
        && result.false_call_rate_upper <= plan.maximum_false_call_rate;
    let outcome = if established {
        "POD_QUALIFIED_AT_CRITICAL_SIZE"
    } else {
        "POD_NOT_ESTABLISHED"
    };
    if result.outcome != outcome {
        return rejected("reported POD outcome does not match the preregistered a90/95 size and false-call limits");
    }

    Decision {
        disposition: Disposition::Pass,
        outcome,
        reason: "defect-specific hit/miss evidence, known truth, blind inspections, independent units and POD result are structurally linked; no POD calculation is performed here",
    }
}

impl Disposition {
    fn into_decision(self, reason: &'static str) -> Decision {
        Decision {
            disposition: self,
            outcome: "NOT_EVALUATED",
            reason,
        }
    }
}

fn audit_case(case: &Case) -> Decision {
    audit_study(&case.study)
}

fn read_fixture(path: &Path) -> Result<Fixture> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read fixture {}", path.display()))?;
    serde_json::from_str(&content)
        .with_context(|| format!("failed to parse fixture {}", path.display()))
}

pub fn run(path: &Path) -> Result<()> {
    let fixture = read_fixture(path)?;
    if fixture.schema_version != "plasma-ndt-pod-fixture-v0.1" {
        bail!("unsupported fixture schema version: {}", fixture.schema_version);
    }
    let mut counts = [0usize; 3];
    let mut outcomes = [0usize; 2];
    for case in &fixture.cases {
        let result = audit_case(case);
        let index = match result.disposition {
            Disposition::Pass => 0,
            Disposition::Blocked => 1,
            Disposition::Rejected => 2,
        };
        counts[index] += 1;
        match result.outcome {
            "POD_QUALIFIED_AT_CRITICAL_SIZE" => outcomes[0] += 1,
            "POD_NOT_ESTABLISHED" => outcomes[1] += 1,
            _ => {}
        }
        println!("{}: {} / {} — {}", case.id, result.disposition.as_str(), result.outcome, result.reason);
        if result.disposition.as_str() != case.expected_disposition {
            bail!("case {}: expected {}, got {}", case.id, case.expected_disposition, result.disposition.as_str());
        }
        if result.outcome != case.expected_outcome {
            bail!("case {}: expected outcome {}, got {}", case.id, case.expected_outcome, result.outcome);
        }
        if !nonempty(&case.rationale) {
            bail!("case {} has an empty rationale", case.id);
        }
    }
    println!("PLASMA-NDT-POD S0 — {} cases; PASS={} BLOCKED={} REJECTED={}; POD_QUALIFIED={} POD_NOT_ESTABLISHED={}", fixture.cases.len(), counts[0], counts[1], counts[2], outcomes[0], outcomes[1]);
    println!("Claim ceiling: defect-specific detection only; no joint strength, failure acceptability, structural safety or product release is implied.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str =
        include_str!("../../data/research/plasma_ndt_pod_fixture_v0_1.json");

    #[test]
    fn frozen_pod_fixture_matches_all_dispositions_and_outcomes() {
        let fixture: Fixture = serde_json::from_str(FIXTURE).expect("valid POD fixture");
        assert_eq!(fixture.schema_version, "plasma-ndt-pod-fixture-v0.1");
        for case in &fixture.cases {
            let result = audit_case(case);
            assert_eq!(result.disposition.as_str(), case.expected_disposition, "{}", case.id);
            assert_eq!(result.outcome, case.expected_outcome, "{}", case.id);
        }
    }

    #[test]
    fn uncertain_kissing_bond_truth_blocks_pod_claim() {
        let fixture: Fixture = serde_json::from_str(FIXTURE).expect("valid POD fixture");
        let case = fixture.cases.iter().find(|case| case.id == "kissing-bond-truth-uncertain").expect("case exists");
        assert_eq!(audit_study(&case.study).disposition, Disposition::Blocked);
    }

    #[test]
    fn incomplete_scan_coverage_blocks() {
        let fixture: Fixture = serde_json::from_str(FIXTURE).expect("valid POD fixture");
        let case = fixture.cases.iter().find(|case| case.id == "incomplete-scan-coverage").expect("case exists");
        assert_eq!(audit_study(&case.study).disposition, Disposition::Blocked);
    }

    #[test]
    fn same_specimen_cannot_be_counted_as_repeated_independent_scans() {
        let fixture: Fixture = serde_json::from_str(FIXTURE).expect("valid POD fixture");
        let case = fixture.cases.iter().find(|case| case.id == "repeated-scan-row-duplicate-specimen").expect("case exists");
        assert_eq!(audit_study(&case.study).disposition, Disposition::Rejected);
    }

    #[test]
    fn valid_study_can_fail_to_establish_pod_at_critical_size() {
        let fixture: Fixture = serde_json::from_str(FIXTURE).expect("valid POD fixture");
        let case = fixture.cases.iter().find(|case| case.id == "valid-study-pod-not-established").expect("case exists");
        let result = audit_study(&case.study);
        assert_eq!(result.disposition, Disposition::Pass);
        assert_eq!(result.outcome, "POD_NOT_ESTABLISHED");
    }
}
