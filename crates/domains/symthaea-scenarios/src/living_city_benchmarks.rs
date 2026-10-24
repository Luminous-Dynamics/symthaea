//! Reproducible stress-case registry and fail-closed coverage assessment.
//!
//! This module defines which cases a living-city evaluation must exercise; it
//! does not simulate heat, flood, water, energy, or buildings itself. Physical
//! magnitudes come from a site-specific, versioned profile and evidence manifest.
//! The registry is a test protocol, not evidence that any solver is valid.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

/// Bump when the meaning, membership, or required domains of the suite change.
pub const LIVING_CITY_BENCHMARK_SUITE_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Hazard {
    ExtremeHeat,
    Drought,
    GridOutage72Hours,
    Flood,
    CommunicationsFailure,
    EnergyShortfall,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BenchmarkClass {
    Baseline,
    SingleHazard,
    CompoundHazard,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum EngineeringDomain {
    BuildingThermal,
    CoolingAccess,
    Energy,
    Water,
    CriticalServices,
    Communications,
    Stormwater,
    EmergencyAccess,
    Mobility,
    HousingAndAccess,
    EcologyAndMaterials,
}

/// A stable test-case definition. Hazard severity is intentionally not encoded
/// here: every profile must be specific to the declared site, jurisdiction,
/// climate basis, time horizon, and evidence snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BenchmarkDefinition {
    pub id: &'static str,
    pub class: BenchmarkClass,
    pub hazards: &'static [Hazard],
    pub required_domains: &'static [EngineeringDomain],
    /// Key for the locally defined and independently versioned model profile.
    pub local_profile_key: &'static str,
    pub rationale: &'static str,
}

/// Initial reference suite: one baseline, three single hazards, and three
/// compound hazards. The local pilot may extend this suite, but must not quietly
/// remove a required case to make a candidate pass.
pub const LIVING_CITY_BENCHMARKS: &[BenchmarkDefinition] = &[
    BenchmarkDefinition {
        id: "baseline-operating-envelope",
        class: BenchmarkClass::Baseline,
        hazards: &[],
        required_domains: &[
            EngineeringDomain::BuildingThermal,
            EngineeringDomain::Energy,
            EngineeringDomain::Water,
            EngineeringDomain::Mobility,
            EngineeringDomain::CriticalServices,
            EngineeringDomain::HousingAndAccess,
            EngineeringDomain::EcologyAndMaterials,
        ],
        local_profile_key: "site.baseline",
        rationale: "Common reference for comparison; record present-day and business-as-usual assumptions separately.",
    },
    BenchmarkDefinition {
        id: "heat-design-event",
        class: BenchmarkClass::SingleHazard,
        hazards: &[Hazard::ExtremeHeat],
        required_domains: &[
            EngineeringDomain::BuildingThermal,
            EngineeringDomain::CoolingAccess,
            EngineeringDomain::Energy,
            EngineeringDomain::CriticalServices,
        ],
        local_profile_key: "site.hazard.heat",
        rationale: "Evaluate thermal safety, cooling/refuge access, peak demand, and critical loads for a locally approved event.",
    },
    BenchmarkDefinition {
        id: "drought-design-case",
        class: BenchmarkClass::SingleHazard,
        hazards: &[Hazard::Drought],
        required_domains: &[
            EngineeringDomain::Water,
            EngineeringDomain::Energy,
            EngineeringDomain::EcologyAndMaterials,
        ],
        local_profile_key: "site.hazard.drought",
        rationale: "Evaluate supply reliability, demand reduction, reuse constraints, and cross-sector consequences under local drought assumptions.",
    },
    BenchmarkDefinition {
        id: "critical-services-72h-grid-outage",
        class: BenchmarkClass::SingleHazard,
        hazards: &[Hazard::GridOutage72Hours],
        required_domains: &[
            EngineeringDomain::Energy,
            EngineeringDomain::Water,
            EngineeringDomain::Communications,
            EngineeringDomain::CriticalServices,
            EngineeringDomain::EmergencyAccess,
        ],
        local_profile_key: "site.hazard.grid-outage-72h",
        rationale: "Test named critical assets during a 72-hour outage; 72 hours is a suite test condition, not a claim that all assets can survive it.",
    },
    BenchmarkDefinition {
        id: "heat-plus-72h-grid-outage",
        class: BenchmarkClass::CompoundHazard,
        hazards: &[Hazard::ExtremeHeat, Hazard::GridOutage72Hours],
        required_domains: &[
            EngineeringDomain::BuildingThermal,
            EngineeringDomain::CoolingAccess,
            EngineeringDomain::Energy,
            EngineeringDomain::Communications,
            EngineeringDomain::CriticalServices,
        ],
        local_profile_key: "site.compound.heat-grid-outage",
        rationale: "Test correlated heat and power failure, including cooling loss, backup fuel/storage duration, and the accessibility of refuges.",
    },
    BenchmarkDefinition {
        id: "drought-plus-energy-shortfall",
        class: BenchmarkClass::CompoundHazard,
        hazards: &[Hazard::Drought, Hazard::EnergyShortfall],
        required_domains: &[
            EngineeringDomain::Water,
            EngineeringDomain::Energy,
            EngineeringDomain::CriticalServices,
            EngineeringDomain::EcologyAndMaterials,
        ],
        local_profile_key: "site.compound.drought-energy-shortfall",
        rationale: "Test water supply/treatment/pumping dependencies when energy availability is constrained.",
    },
    BenchmarkDefinition {
        id: "flood-plus-communications-failure",
        class: BenchmarkClass::CompoundHazard,
        hazards: &[Hazard::Flood, Hazard::CommunicationsFailure],
        required_domains: &[
            EngineeringDomain::Stormwater,
            EngineeringDomain::Communications,
            EngineeringDomain::EmergencyAccess,
            EngineeringDomain::CriticalServices,
            EngineeringDomain::Mobility,
        ],
        local_profile_key: "site.compound.flood-comms-failure",
        rationale: "Test inundation, loss of telemetry/communications, emergency reachability, and safe operating fallbacks.",
    },
];

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BenchmarkProfileBinding {
    pub benchmark_id: String,
    pub profile_id: String,
    /// Digest of the retained profile plus its source/evidence references.
    pub profile_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BenchmarkContractError {
    pub field: &'static str,
    pub reason: &'static str,
}

impl std::fmt::Display for BenchmarkContractError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "invalid living-city benchmark {}: {}", self.field, self.reason)
    }
}

impl std::error::Error for BenchmarkContractError {}

fn invalid(field: &'static str, reason: &'static str) -> BenchmarkContractError {
    BenchmarkContractError { field, reason }
}

fn non_empty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

/// Check the registry's internal semantics before trusting or extending it.
pub fn validate_benchmark_registry(
    definitions: &[BenchmarkDefinition],
) -> Result<(), BenchmarkContractError> {
    if definitions.is_empty() {
        return Err(invalid("definitions", "registry cannot be empty"));
    }
    let mut ids = BTreeSet::new();
    let mut baselines = 0usize;
    for definition in definitions {
        if !non_empty(definition.id) {
            return Err(invalid("id", "must be non-empty"));
        }
        if !ids.insert(definition.id) {
            return Err(invalid("id", "benchmark IDs must be unique"));
        }
        if !non_empty(definition.local_profile_key) {
            return Err(invalid("local_profile_key", "must be non-empty"));
        }
        if !non_empty(definition.rationale) {
            return Err(invalid("rationale", "must be non-empty"));
        }
        if definition.required_domains.is_empty() {
            return Err(invalid("required_domains", "at least one domain must be declared"));
        }
        let mut hazards = BTreeSet::new();
        for hazard in definition.hazards {
            if !hazards.insert(*hazard) {
                return Err(invalid("hazards", "hazards within one case must be unique"));
            }
        }
        match definition.class {
            BenchmarkClass::Baseline => {
                baselines += 1;
                if !definition.hazards.is_empty() {
                    return Err(invalid("class", "baseline cases cannot declare hazard stressors"));
                }
            }
            BenchmarkClass::SingleHazard if definition.hazards.len() != 1 => {
                return Err(invalid("class", "single-hazard cases must declare exactly one hazard"));
            }
            BenchmarkClass::CompoundHazard if definition.hazards.len() < 2 => {
                return Err(invalid("class", "compound-hazard cases must declare at least two hazards"));
            }
            BenchmarkClass::SingleHazard | BenchmarkClass::CompoundHazard => {}
        }
    }
    if baselines != 1 {
        return Err(invalid("class", "registry must define exactly one baseline"));
    }
    Ok(())
}

/// Require exactly one traceable, local profile for every required benchmark.
pub fn validate_profile_bindings(
    definitions: &[BenchmarkDefinition],
    bindings: &[BenchmarkProfileBinding],
) -> Result<(), BenchmarkContractError> {
    validate_benchmark_registry(definitions)?;
    let required: BTreeSet<&str> = definitions.iter().map(|item| item.id).collect();
    let mut bound = BTreeSet::new();
    for binding in bindings {
        if !required.contains(binding.benchmark_id.as_str()) {
            return Err(invalid("benchmark_id", "profile references an unknown benchmark"));
        }
        if !bound.insert(binding.benchmark_id.as_str()) {
            return Err(invalid("benchmark_id", "each benchmark must have exactly one profile binding"));
        }
        if !non_empty(&binding.profile_id) {
            return Err(invalid("profile_id", "must identify the selected local profile revision"));
        }
        if !is_sha256(&binding.profile_sha256) {
            return Err(invalid("profile_sha256", "must be a 64-character lowercase SHA-256 digest"));
        }
    }
    if bound.len() != required.len() {
        return Err(invalid("bindings", "every required benchmark needs a profile binding"));
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BenchmarkStatus {
    Pass,
    Fail,
    Blocked,
    NotRun,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BenchmarkExecutionRecord {
    pub benchmark_id: String,
    pub status: BenchmarkStatus,
    /// Identity/version of the verifier that checked this exact result artifact.
    pub verifier_id: Option<String>,
    pub result_sha256: Option<String>,
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BenchmarkSuiteState {
    EligibleForNextStage,
    Failed,
    Blocked,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BenchmarkSuiteAssessment {
    pub state: BenchmarkSuiteState,
    pub missing: Vec<String>,
    pub failed: Vec<String>,
    pub blocked: Vec<String>,
}

fn validate_execution_record(
    record: &BenchmarkExecutionRecord,
) -> Result<(), BenchmarkContractError> {
    if !non_empty(&record.benchmark_id) {
        return Err(invalid("benchmark_id", "must be non-empty"));
    }
    match record.status {
        BenchmarkStatus::Pass | BenchmarkStatus::Fail => {
            if !record.verifier_id.as_deref().is_some_and(non_empty) {
                return Err(invalid("verifier_id", "pass/fail results require verifier identity"));
            }
            if !record.result_sha256.as_deref().is_some_and(is_sha256) {
                return Err(invalid("result_sha256", "pass/fail results require a SHA-256 result artifact digest"));
            }
            if record.status == BenchmarkStatus::Fail
                && !record.reason.as_deref().is_some_and(non_empty)
            {
                return Err(invalid("reason", "failed result must state the failing condition"));
            }
        }
        BenchmarkStatus::Blocked | BenchmarkStatus::NotRun => {
            if !record.reason.as_deref().is_some_and(non_empty) {
                return Err(invalid("reason", "blocked/not-run result must explain why"));
            }
        }
    }
    Ok(())
}

/// Assess required-case coverage without trusting a generator-supplied summary.
/// Missing or not-run required cases block promotion; any failed required case
/// fails the suite. A complete suite means only eligible for the next workflow
/// stage, not that the physical model is valid or a design is construction-ready.
pub fn assess_benchmark_suite(
    definitions: &[BenchmarkDefinition],
    records: &[BenchmarkExecutionRecord],
) -> Result<BenchmarkSuiteAssessment, BenchmarkContractError> {
    validate_benchmark_registry(definitions)?;
    let expected: BTreeSet<&str> = definitions.iter().map(|item| item.id).collect();
    let mut seen = BTreeSet::new();
    for record in records {
        validate_execution_record(record)?;
        if !expected.contains(record.benchmark_id.as_str()) {
            return Err(invalid("benchmark_id", "execution record references an unknown benchmark"));
        }
        if !seen.insert(record.benchmark_id.as_str()) {
            return Err(invalid("benchmark_id", "execution records must be unique per benchmark"));
        }
    }

    let by_id: std::collections::BTreeMap<&str, &BenchmarkExecutionRecord> = records
        .iter()
        .map(|record| (record.benchmark_id.as_str(), record))
        .collect();
    let mut assessment = BenchmarkSuiteAssessment {
        state: BenchmarkSuiteState::EligibleForNextStage,
        missing: Vec::new(),
        failed: Vec::new(),
        blocked: Vec::new(),
    };
    for definition in definitions {
        match by_id.get(definition.id) {
            None => assessment.missing.push(definition.id.to_owned()),
            Some(record) => match record.status {
                BenchmarkStatus::Pass => {}
                BenchmarkStatus::Fail => assessment.failed.push(definition.id.to_owned()),
                BenchmarkStatus::Blocked | BenchmarkStatus::NotRun => {
                    assessment.blocked.push(definition.id.to_owned())
                }
            },
        }
    }
    if !assessment.failed.is_empty() {
        assessment.state = BenchmarkSuiteState::Failed;
    } else if !assessment.missing.is_empty() || !assessment.blocked.is_empty() {
        assessment.state = BenchmarkSuiteState::Blocked;
    }
    Ok(assessment)
}

#[cfg(test)]
mod tests {
    use super::*;

    const HASH: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    fn all_pass_records() -> Vec<BenchmarkExecutionRecord> {
        LIVING_CITY_BENCHMARKS
            .iter()
            .map(|definition| BenchmarkExecutionRecord {
                benchmark_id: definition.id.to_owned(),
                status: BenchmarkStatus::Pass,
                verifier_id: Some("independent-benchmark-verifier/0.1".into()),
                result_sha256: Some(HASH.into()),
                reason: None,
            })
            .collect()
    }

    fn all_profile_bindings() -> Vec<BenchmarkProfileBinding> {
        LIVING_CITY_BENCHMARKS
            .iter()
            .map(|definition| BenchmarkProfileBinding {
                benchmark_id: definition.id.to_owned(),
                profile_id: format!("fixture-profile/{}", definition.id),
                profile_sha256: HASH.into(),
            })
            .collect()
    }

    #[test]
    fn registry_has_baseline_singles_and_compound_hazards() {
        validate_benchmark_registry(LIVING_CITY_BENCHMARKS).unwrap();
        assert_eq!(
            LIVING_CITY_BENCHMARKS
                .iter()
                .filter(|case| case.class == BenchmarkClass::Baseline)
                .count(),
            1
        );
        assert!(LIVING_CITY_BENCHMARKS.iter().any(|case| case.class == BenchmarkClass::SingleHazard));
        assert!(LIVING_CITY_BENCHMARKS.iter().filter(|case| case.class == BenchmarkClass::CompoundHazard).count() >= 3);
    }

    #[test]
    fn profile_bindings_cover_every_case() {
        validate_profile_bindings(LIVING_CITY_BENCHMARKS, &all_profile_bindings()).unwrap();
    }

    #[test]
    fn missing_profile_blocks_configuration() {
        let mut bindings = all_profile_bindings();
        bindings.pop();
        assert_eq!(
            validate_profile_bindings(LIVING_CITY_BENCHMARKS, &bindings)
                .unwrap_err()
                .field,
            "bindings"
        );
    }

    #[test]
    fn duplicate_profile_binding_is_rejected() {
        let mut bindings = all_profile_bindings();
        bindings.push(bindings[0].clone());
        assert_eq!(
            validate_profile_bindings(LIVING_CITY_BENCHMARKS, &bindings)
                .unwrap_err()
                .field,
            "benchmark_id"
        );
    }

    #[test]
    fn all_verified_cases_make_suite_eligible_for_next_stage() {
        let assessment = assess_benchmark_suite(LIVING_CITY_BENCHMARKS, &all_pass_records()).unwrap();
        assert_eq!(assessment.state, BenchmarkSuiteState::EligibleForNextStage);
        assert!(assessment.missing.is_empty());
        assert!(assessment.failed.is_empty());
        assert!(assessment.blocked.is_empty());
    }

    #[test]
    fn missing_case_blocks_suite_even_if_all_returned_cases_pass() {
        let mut records = all_pass_records();
        records.pop();
        let assessment = assess_benchmark_suite(LIVING_CITY_BENCHMARKS, &records).unwrap();
        assert_eq!(assessment.state, BenchmarkSuiteState::Blocked);
        assert_eq!(assessment.missing.len(), 1);
    }

    #[test]
    fn failed_compound_hazard_fails_suite() {
        let mut records = all_pass_records();
        let record = records
            .iter_mut()
            .find(|record| record.benchmark_id == "heat-plus-72h-grid-outage")
            .unwrap();
        record.status = BenchmarkStatus::Fail;
        record.reason = Some("synthetic negative-control failure".into());
        let assessment = assess_benchmark_suite(LIVING_CITY_BENCHMARKS, &records).unwrap();
        assert_eq!(assessment.state, BenchmarkSuiteState::Failed);
        assert_eq!(assessment.failed, vec!["heat-plus-72h-grid-outage"]);
    }

    #[test]
    fn not_run_case_blocks_suite() {
        let mut records = all_pass_records();
        let record = records
            .iter_mut()
            .find(|record| record.benchmark_id == "drought-design-case")
            .unwrap();
        record.status = BenchmarkStatus::NotRun;
        record.verifier_id = None;
        record.result_sha256 = None;
        record.reason = Some("solver unavailable in this run".into());
        let assessment = assess_benchmark_suite(LIVING_CITY_BENCHMARKS, &records).unwrap();
        assert_eq!(assessment.state, BenchmarkSuiteState::Blocked);
        assert_eq!(assessment.blocked, vec!["drought-design-case"]);
    }

    #[test]
    fn pass_without_verifier_evidence_is_rejected() {
        let mut records = all_pass_records();
        records[0].result_sha256 = None;
        assert_eq!(
            assess_benchmark_suite(LIVING_CITY_BENCHMARKS, &records)
                .unwrap_err()
                .field,
            "result_sha256"
        );
    }

    #[test]
    fn duplicate_or_unknown_execution_records_are_rejected() {
        let mut records = all_pass_records();
        records.push(records[0].clone());
        assert_eq!(
            assess_benchmark_suite(LIVING_CITY_BENCHMARKS, &records)
                .unwrap_err()
                .field,
            "benchmark_id"
        );

        let unknown = vec![BenchmarkExecutionRecord {
            benchmark_id: "unregistered-case".into(),
            status: BenchmarkStatus::NotRun,
            verifier_id: None,
            result_sha256: None,
            reason: Some("fixture".into()),
        }];
        assert_eq!(
            assess_benchmark_suite(LIVING_CITY_BENCHMARKS, &unknown)
                .unwrap_err()
                .field,
            "benchmark_id"
        );
    }
}
