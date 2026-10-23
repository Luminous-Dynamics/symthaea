// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Evidence-carrying cookware comparison contracts.
//!
//! This module defines an auditable benchmark protocol for comparing cookware
//! under matched, declared conditions. It keeps instrument measurements,
//! simulation, manufacturer claims, and practitioner experience distinct.
//! Passing validation means the benchmark manifest is structurally consistent;
//! it is not a food-contact certification or a claim of superior performance.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const COOKWARE_BENCHMARK_SCHEMA_VERSION: u32 = 1;

/// Outcomes with units fixed by the metric variant.
///
/// Values are signed integers in the declared subunit to avoid hidden
/// float/rounding differences in the evidence contract.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CookwareMetric {
    SurfaceTemperatureSpreadMilliCelsius,
    HeatUpEnergyMilliJoule,
    RecoveryTimeMillis,
    HandleMaximumTemperatureMilliCelsius,
    FoodReleasePeakForceMilliNewton,
    CleanupTimeMillis,
    FoodMassLossMilligram,
    PanWarpMicrometre,
    EnergyPerEdibleServingMilliJoule,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CookwareEvidenceClass {
    ObservedInstrumental,
    ObservedPractitioner,
    Simulated,
    ManufacturerDeclared,
    SyntheticFixture,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CookwareRole {
    Candidate,
    Reference,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FoodContactEvidenceStatus {
    Unverified,
    QualifiedForDeclaredUse,
    NotApplicableForDeclaredTest,
}

/// One exact candidate design revision. Material and qualification references
/// point to external evidence; this schema does not authenticate those records.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct CookwareCandidateV1 {
    pub candidate_id: String,
    pub design_revision_id: String,
    pub role: CookwareRole,
    pub construction_family_id: String,
    pub intended_task_profile_id: String,
    pub material_dossier_ref: String,
    pub food_contact_status: FoodContactEvidenceStatus,
    pub food_contact_qualification_ref: Option<String>,
    pub food_contact_jurisdiction: Option<String>,
}

/// Frozen comparison conditions. Candidate results are comparable only within
/// the same declared test protocol and block definition.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct CookwareProtocolV1 {
    pub protocol_id: String,
    pub task_profile_id: String,
    pub geometry_profile_id: String,
    pub heat_source_profile_id: String,
    pub load_profile_id: String,
    pub ambient_profile_id: String,
    pub jurisdiction: String,
    pub randomization_method_ref: String,
    pub minimum_blocks_per_candidate: u32,
    pub minimum_independent_units_per_candidate: u32,
    pub required_metrics: Vec<CookwareMetric>,
    /// When true, every required primary measurement must be instrument-observed.
    pub empirical_comparison: bool,
    /// If true, all candidates must carry current qualification evidence for
    /// the declared use/jurisdiction before this comparison can be qualified
    /// to make food-contact suitability claims.
    pub claims_food_contact_suitability: bool,
}

/// Instrumental or declared value associated with one run and one metric.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct CookwareObservationV1 {
    pub metric: CookwareMetric,
    pub value: i64,
    pub evidence_class: CookwareEvidenceClass,
    pub evidence_ref: String,
    pub measurement_method_ref: String,
    pub calibration_ref: Option<String>,
    /// Absolute uncertainty in the metric's fixed unit; null means not supplied.
    pub uncertainty_abs: Option<u64>,
}

/// A single run under a frozen protocol. Repeated measurements of one physical
/// unit and measurements across independent units remain distinguishable.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct CookwareRunV1 {
    pub run_id: String,
    pub candidate_id: String,
    pub cookware_unit_id: String,
    pub block_id: String,
    /// Actual run order inside the block; the randomization method is protocol-pinned.
    pub order_index: u32,
    pub protocol_id: String,
    pub task_profile_id: String,
    pub geometry_profile_id: String,
    pub heat_source_profile_id: String,
    pub load_profile_id: String,
    pub ambient_profile_id: String,
    pub observations: Vec<CookwareObservationV1>,
}

/// Craft outcomes are separate from thermal or cooking performance metrics.
/// Do not derive a universal craft score from these dimensions.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CraftExperienceDimension {
    PerceivedControl,
    TactileFeedback,
    CreativeAgency,
    LearningSupport,
    Enjoyment,
    PhysicalStrain,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AssistanceMode {
    Manual,
    OptionalAssistance,
    AutomatedAssistance,
}

/// A practitioner feedback record. Participant IDs must be pseudonymous opaque
/// identifiers; do not place names, contact details, audio/video or free-text
/// personal data in this contract.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct CraftExperienceRecordV1 {
    pub record_id: String,
    pub participant_pseudonym: String,
    pub consent_receipt_ref: String,
    pub instrument_ref: String,
    pub candidate_id: String,
    pub assistance_mode: AssistanceMode,
    /// Separate 1..=7 ratings; report distributions per dimension, not a sum.
    pub ratings: Vec<CraftExperienceRatingV1>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct CraftExperienceRatingV1 {
    pub dimension: CraftExperienceDimension,
    pub rating_1_to_7: u8,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct CookwareBenchmarkManifestV1 {
    pub schema_version: u32,
    pub benchmark_id: String,
    pub protocol: CookwareProtocolV1,
    pub candidates: Vec<CookwareCandidateV1>,
    pub runs: Vec<CookwareRunV1>,
    pub craft_experience: Vec<CraftExperienceRecordV1>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CookwareViolationCode {
    UnsupportedSchemaVersion,
    MissingRequiredValue,
    DuplicateIdentifier,
    MissingReferenceCandidate,
    MissingCandidate,
    UnknownCandidate,
    InvalidProtocol,
    DuplicateRequiredMetric,
    ContextMismatch,
    MissingBlock,
    IncompleteBlock,
    DuplicateRunInBlock,
    DuplicateRunOrder,
    InsufficientBlocks,
    InsufficientIndependentUnits,
    CookwareUnitIdentityConflict,
    DuplicateObservation,
    MissingRequiredMetric,
    MissingEvidenceReference,
    MissingCalibrationReference,
    WrongEvidenceClassForEmpiricalClaim,
    InvalidMetricValue,
    MissingFoodContactQualification,
    MissingCraftConsent,
    MissingCraftInstrument,
    InvalidCraftRating,
    DuplicateCraftDimension,
}

/// Machine-readable reason why a benchmark cannot be qualified.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct CookwareViolation {
    pub code: CookwareViolationCode,
    pub candidate_id: Option<String>,
    pub run_id: Option<String>,
    pub detail: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct CookwareValidationReport {
    pub benchmark_id: String,
    pub candidate_count: usize,
    pub run_count: usize,
    pub block_count: usize,
    pub craft_record_count: usize,
    pub violations: Vec<CookwareViolation>,
}

impl CookwareValidationReport {
    pub fn is_valid(&self) -> bool {
        self.violations.is_empty()
    }
}

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn violation(
    violations: &mut Vec<CookwareViolation>,
    code: CookwareViolationCode,
    candidate_id: Option<&str>,
    run_id: Option<&str>,
    detail: impl Into<String>,
) {
    violations.push(CookwareViolation {
        code,
        candidate_id: candidate_id.map(str::to_owned),
        run_id: run_id.map(str::to_owned),
        detail: detail.into(),
    });
}

fn metric_requires_nonnegative_value(metric: CookwareMetric) -> bool {
    !matches!(
        metric,
        CookwareMetric::HandleMaximumTemperatureMilliCelsius
    )
}

/// Validate manifest structure, matched comparison conditions, replication,
/// evidence sufficiency, food-contact claim ceilings and craft-data separation.
///
/// This function authenticates neither evidence references nor laboratory
/// qualification. A valid report means only that the declared contract is
/// internally consistent enough to proceed to independent evidence review.
pub fn validate_cookware_benchmark(
    manifest: &CookwareBenchmarkManifestV1,
) -> CookwareValidationReport {
    let mut violations = Vec::new();
    if manifest.schema_version != COOKWARE_BENCHMARK_SCHEMA_VERSION {
        violation(
            &mut violations,
            CookwareViolationCode::UnsupportedSchemaVersion,
            None,
            None,
            format!(
                "schema version {} is unsupported; expected {}",
                manifest.schema_version, COOKWARE_BENCHMARK_SCHEMA_VERSION
            ),
        );
    }
    if !nonempty(&manifest.benchmark_id) {
        violation(
            &mut violations,
            CookwareViolationCode::MissingRequiredValue,
            None,
            None,
            "benchmark_id must be non-empty",
        );
    }

    let protocol = &manifest.protocol;
    for (name, value) in [
        ("protocol_id", protocol.protocol_id.as_str()),
        ("task_profile_id", protocol.task_profile_id.as_str()),
        ("geometry_profile_id", protocol.geometry_profile_id.as_str()),
        ("heat_source_profile_id", protocol.heat_source_profile_id.as_str()),
        ("load_profile_id", protocol.load_profile_id.as_str()),
        ("ambient_profile_id", protocol.ambient_profile_id.as_str()),
        ("jurisdiction", protocol.jurisdiction.as_str()),
        ("randomization_method_ref", protocol.randomization_method_ref.as_str()),
    ] {
        if !nonempty(value) {
            violation(
                &mut violations,
                CookwareViolationCode::InvalidProtocol,
                None,
                None,
                format!("{name} must be non-empty"),
            );
        }
    }
    if protocol.minimum_blocks_per_candidate == 0
        || protocol.minimum_independent_units_per_candidate == 0
        || protocol.required_metrics.is_empty()
    {
        violation(
            &mut violations,
            CookwareViolationCode::InvalidProtocol,
            None,
            None,
            "minimum block/unit counts and required metrics must be non-zero",
        );
    }

    let mut required_metrics = BTreeSet::new();
    for metric in &protocol.required_metrics {
        if !required_metrics.insert(*metric) {
            violation(
                &mut violations,
                CookwareViolationCode::DuplicateRequiredMetric,
                None,
                None,
                format!("required metric {metric:?} is declared more than once"),
            );
        }
    }

    let mut candidates: BTreeMap<&str, &CookwareCandidateV1> = BTreeMap::new();
    let mut has_reference = false;
    for candidate in &manifest.candidates {
        if !nonempty(&candidate.candidate_id)
            || !nonempty(&candidate.design_revision_id)
            || !nonempty(&candidate.construction_family_id)
            || !nonempty(&candidate.intended_task_profile_id)
            || !nonempty(&candidate.material_dossier_ref)
        {
            violation(
                &mut violations,
                CookwareViolationCode::MissingRequiredValue,
                Some(&candidate.candidate_id),
                None,
                "candidate identity, design revision, construction family, task profile and material dossier are required",
            );
        }
        if candidate.role == CookwareRole::Reference {
            has_reference = true;
        }
        if !candidate.candidate_id.trim().is_empty()
            && candidates
                .insert(candidate.candidate_id.as_str(), candidate)
                .is_some()
        {
            violation(
                &mut violations,
                CookwareViolationCode::DuplicateIdentifier,
                Some(&candidate.candidate_id),
                None,
                "candidate_id is duplicated",
            );
        }
        if candidate.intended_task_profile_id != protocol.task_profile_id {
            violation(
                &mut violations,
                CookwareViolationCode::ContextMismatch,
                Some(&candidate.candidate_id),
                None,
                "candidate intended task profile differs from the frozen protocol",
            );
        }
        if protocol.claims_food_contact_suitability {
            let qualified = candidate.food_contact_status
                == FoodContactEvidenceStatus::QualifiedForDeclaredUse;
            let has_ref = candidate
                .food_contact_qualification_ref
                .as_deref()
                .is_some_and(nonempty);
            let matching_jurisdiction = candidate
                .food_contact_jurisdiction
                .as_deref()
                .is_some_and(|jurisdiction| jurisdiction == protocol.jurisdiction);
            if !qualified || !has_ref || !matching_jurisdiction {
                violation(
                    &mut violations,
                    CookwareViolationCode::MissingFoodContactQualification,
                    Some(&candidate.candidate_id),
                    None,
                    concat!(
                        "food-contact suitability claim requires a qualification reference ",
                        "for this declared use and jurisdiction"
                    ),
                );
            }
        }
    }
    if manifest.candidates.len() < 2 {
        violation(
            &mut violations,
            CookwareViolationCode::MissingCandidate,
            None,
            None,
            "at least two candidates are required for comparison",
        );
    }
    if !has_reference {
        violation(
            &mut violations,
            CookwareViolationCode::MissingReferenceCandidate,
            None,
            None,
            "at least one established reference candidate is required",
        );
    }

    let mut run_ids = BTreeSet::new();
    let mut blocks: BTreeMap<&str, BTreeMap<&str, &CookwareRunV1>> = BTreeMap::new();
    let mut run_counts: BTreeMap<&str, u32> = BTreeMap::new();
    let mut unit_sets: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
    let mut unit_owners: BTreeMap<&str, &str> = BTreeMap::new();

    for run in &manifest.runs {
        if !nonempty(&run.run_id)
            || !nonempty(&run.candidate_id)
            || !nonempty(&run.cookware_unit_id)
            || !nonempty(&run.block_id)
        {
            violation(
                &mut violations,
                CookwareViolationCode::MissingRequiredValue,
                Some(&run.candidate_id),
                Some(&run.run_id),
                "run, candidate, cookware unit and block IDs must be non-empty",
            );
        }
        if !run_ids.insert(run.run_id.as_str()) {
            violation(
                &mut violations,
                CookwareViolationCode::DuplicateIdentifier,
                Some(&run.candidate_id),
                Some(&run.run_id),
                "run_id is duplicated",
            );
        }
        if !candidates.contains_key(run.candidate_id.as_str()) {
            violation(
                &mut violations,
                CookwareViolationCode::UnknownCandidate,
                Some(&run.candidate_id),
                Some(&run.run_id),
                "run references an undeclared candidate",
            );
        }
        for (name, actual, expected) in [
            (
                "protocol_id",
                run.protocol_id.as_str(),
                protocol.protocol_id.as_str(),
            ),
            (
                "task_profile_id",
                run.task_profile_id.as_str(),
                protocol.task_profile_id.as_str(),
            ),
            (
                "geometry_profile_id",
                run.geometry_profile_id.as_str(),
                protocol.geometry_profile_id.as_str(),
            ),
            (
                "heat_source_profile_id",
                run.heat_source_profile_id.as_str(),
                protocol.heat_source_profile_id.as_str(),
            ),
            (
                "load_profile_id",
                run.load_profile_id.as_str(),
                protocol.load_profile_id.as_str(),
            ),
            (
                "ambient_profile_id",
                run.ambient_profile_id.as_str(),
                protocol.ambient_profile_id.as_str(),
            ),
        ] {
            if actual != expected {
                violation(
                    &mut violations,
                    CookwareViolationCode::ContextMismatch,
                    Some(&run.candidate_id),
                    Some(&run.run_id),
                    format!("{name} differs from the frozen protocol"),
                );
            }
        }
        if run.order_index == 0 {
            violation(
                &mut violations,
                CookwareViolationCode::InvalidProtocol,
                Some(&run.candidate_id),
                Some(&run.run_id),
                "order_index is 1-based and must be greater than zero",
            );
        }

        let block = blocks.entry(run.block_id.as_str()).or_default();
        if block.insert(run.candidate_id.as_str(), run).is_some() {
            violation(
                &mut violations,
                CookwareViolationCode::DuplicateRunInBlock,
                Some(&run.candidate_id),
                Some(&run.run_id),
                "candidate has more than one run in this block",
            );
        }
        let current_order_conflict = manifest.runs.iter().any(|other| {
            other.run_id != run.run_id
                && other.block_id == run.block_id
                && other.order_index == run.order_index
        });
        if current_order_conflict {
            violation(
                &mut violations,
                CookwareViolationCode::DuplicateRunOrder,
                Some(&run.candidate_id),
                Some(&run.run_id),
                "order_index is not unique within its block",
            );
        }

        *run_counts.entry(run.candidate_id.as_str()).or_default() += 1;
        unit_sets
            .entry(run.candidate_id.as_str())
            .or_default()
            .insert(run.cookware_unit_id.as_str());
        if let Some(owner) = unit_owners.insert(
            run.cookware_unit_id.as_str(),
            run.candidate_id.as_str(),
        ) {
            if owner != run.candidate_id {
                violation(
                    &mut violations,
                    CookwareViolationCode::CookwareUnitIdentityConflict,
                    Some(&run.candidate_id),
                    Some(&run.run_id),
                    format!(
                        "physical cookware unit {} is also assigned to candidate {}",
                        run.cookware_unit_id, owner
                    ),
                );
            }
        }

        let mut observed_metrics = BTreeSet::new();
        for observation in &run.observations {
            if !observed_metrics.insert(observation.metric) {
                violation(
                    &mut violations,
                    CookwareViolationCode::DuplicateObservation,
                    Some(&run.candidate_id),
                    Some(&run.run_id),
                    format!("metric {:?} appears more than once in one run", observation.metric),
                );
            }
            if !nonempty(&observation.evidence_ref)
                || !nonempty(&observation.measurement_method_ref)
            {
                violation(
                    &mut violations,
                    CookwareViolationCode::MissingEvidenceReference,
                    Some(&run.candidate_id),
                    Some(&run.run_id),
                    format!(
                        "metric {:?} requires evidence and method references",
                        observation.metric
                    ),
                );
            }
            if observation.evidence_class == CookwareEvidenceClass::ObservedInstrumental
                && observation
                    .calibration_ref
                    .as_deref()
                    .is_none_or(|reference| !nonempty(reference))
            {
                violation(
                    &mut violations,
                    CookwareViolationCode::MissingCalibrationReference,
                    Some(&run.candidate_id),
                    Some(&run.run_id),
                    format!(
                        "instrument-observed metric {:?} requires a calibration reference",
                        observation.metric
                    ),
                );
            }
            if metric_requires_nonnegative_value(observation.metric) && observation.value < 0 {
                violation(
                    &mut violations,
                    CookwareViolationCode::InvalidMetricValue,
                    Some(&run.candidate_id),
                    Some(&run.run_id),
                    format!("metric {:?} cannot be negative", observation.metric),
                );
            }
            if protocol.empirical_comparison
                && required_metrics.contains(&observation.metric)
                && observation.evidence_class != CookwareEvidenceClass::ObservedInstrumental
            {
                violation(
                    &mut violations,
                    CookwareViolationCode::WrongEvidenceClassForEmpiricalClaim,
                    Some(&run.candidate_id),
                    Some(&run.run_id),
                    format!(
                        concat!(
                            "required metric {:?} must be instrument-observed ",
                            "for an empirical comparison"
                        ),
                        observation.metric
                    ),
                );
            }
        }
        for metric in &protocol.required_metrics {
            if !observed_metrics.contains(metric) {
                violation(
                    &mut violations,
                    CookwareViolationCode::MissingRequiredMetric,
                    Some(&run.candidate_id),
                    Some(&run.run_id),
                    format!("required metric {metric:?} is missing"),
                );
            }
        }
    }

    for (block_id, candidates_in_block) in &blocks {
        if candidates_in_block.len() != candidates.len() {
            violation(
                &mut violations,
                CookwareViolationCode::IncompleteBlock,
                None,
                None,
                format!(
                    "block {block_id} contains {} candidates; expected {}",
                    candidates_in_block.len(),
                    candidates.len()
                ),
            );
        }
    }

    for candidate_id in candidates.keys() {
        let count = run_counts.get(candidate_id).copied().unwrap_or(0);
        let units = unit_sets.get(candidate_id).map_or(0, |set| set.len()) as u32;
        if count < protocol.minimum_blocks_per_candidate {
            violation(
                &mut violations,
                CookwareViolationCode::InsufficientBlocks,
                Some(candidate_id),
                None,
                format!(
                    "candidate has {count} runs; minimum is {}",
                    protocol.minimum_blocks_per_candidate
                ),
            );
        }
        if units < protocol.minimum_independent_units_per_candidate {
            violation(
                &mut violations,
                CookwareViolationCode::InsufficientIndependentUnits,
                Some(candidate_id),
                None,
                format!(
                    "candidate has {units} independent cookware units; minimum is {}",
                    protocol.minimum_independent_units_per_candidate
                ),
            );
        }
    }

    let mut craft_record_ids = BTreeSet::new();
    for record in &manifest.craft_experience {
        if !craft_record_ids.insert(record.record_id.as_str()) {
            violation(
                &mut violations,
                CookwareViolationCode::DuplicateIdentifier,
                Some(&record.candidate_id),
                None,
                format!("craft record ID {} is duplicated", record.record_id),
            );
        }
        if !candidates.contains_key(record.candidate_id.as_str()) {
            violation(
                &mut violations,
                CookwareViolationCode::UnknownCandidate,
                Some(&record.candidate_id),
                None,
                "craft record references an undeclared candidate",
            );
        }
        if !nonempty(&record.record_id) || !nonempty(&record.participant_pseudonym) {
            violation(
                &mut violations,
                CookwareViolationCode::MissingRequiredValue,
                Some(&record.candidate_id),
                None,
                "craft record ID and pseudonymous participant ID must be non-empty",
            );
        }
        if !nonempty(&record.consent_receipt_ref) {
            violation(
                &mut violations,
                CookwareViolationCode::MissingCraftConsent,
                Some(&record.candidate_id),
                None,
                "practitioner feedback requires an explicit consent receipt reference",
            );
        }
        if !nonempty(&record.instrument_ref) {
            violation(
                &mut violations,
                CookwareViolationCode::MissingCraftInstrument,
                Some(&record.candidate_id),
                None,
                "craft ratings require a declared survey/instrument version",
            );
        }
        let mut dimensions = BTreeSet::new();
        if record.ratings.is_empty() {
            violation(
                &mut violations,
                CookwareViolationCode::MissingRequiredValue,
                Some(&record.candidate_id),
                None,
                "craft experience record must contain at least one declared dimension",
            );
        }
        for rating in &record.ratings {
            if !(1..=7).contains(&rating.rating_1_to_7) {
                violation(
                    &mut violations,
                    CookwareViolationCode::InvalidCraftRating,
                    Some(&record.candidate_id),
                    None,
                    "craft ratings must be on the declared 1..=7 response scale",
                );
            }
            if !dimensions.insert(rating.dimension) {
                violation(
                    &mut violations,
                    CookwareViolationCode::DuplicateCraftDimension,
                    Some(&record.candidate_id),
                    None,
                    format!("craft dimension {:?} is duplicated", rating.dimension),
                );
            }
        }
    }

    violations.sort_by(|a, b| {
        (
            a.code,
            a.candidate_id.as_deref().unwrap_or(""),
            a.run_id.as_deref().unwrap_or(""),
            a.detail.as_str(),
        )
            .cmp(&(
                b.code,
                b.candidate_id.as_deref().unwrap_or(""),
                b.run_id.as_deref().unwrap_or(""),
                b.detail.as_str(),
            ))
    });

    CookwareValidationReport {
        benchmark_id: manifest.benchmark_id.clone(),
        candidate_count: manifest.candidates.len(),
        run_count: manifest.runs.len(),
        block_count: blocks.len(),
        craft_record_count: manifest.craft_experience.len(),
        violations,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(id: &str, role: CookwareRole) -> CookwareCandidateV1 {
        CookwareCandidateV1 {
            candidate_id: id.to_owned(),
            design_revision_id: format!("{id}-rev-1"),
            role,
            construction_family_id: format!("{id}-construction"),
            intended_task_profile_id: "saute-v1".to_owned(),
            material_dossier_ref: format!("evidence://materials/{id}"),
            food_contact_status: FoodContactEvidenceStatus::Unverified,
            food_contact_qualification_ref: None,
            food_contact_jurisdiction: None,
        }
    }

    fn protocol() -> CookwareProtocolV1 {
        CookwareProtocolV1 {
            protocol_id: "thermal-grid-v1".to_owned(),
            task_profile_id: "saute-v1".to_owned(),
            geometry_profile_id: "matched-28cm-v1".to_owned(),
            heat_source_profile_id: "induction-profile-v1".to_owned(),
            load_profile_id: "water-load-1kg-v1".to_owned(),
            ambient_profile_id: "lab-ambient-v1".to_owned(),
            jurisdiction: "ZA".to_owned(),
            randomization_method_ref: "protocol://randomize-blocks-v1".to_owned(),
            minimum_blocks_per_candidate: 2,
            minimum_independent_units_per_candidate: 2,
            required_metrics: vec![
                CookwareMetric::SurfaceTemperatureSpreadMilliCelsius,
                CookwareMetric::HeatUpEnergyMilliJoule,
                CookwareMetric::RecoveryTimeMillis,
            ],
            empirical_comparison: true,
            claims_food_contact_suitability: false,
        }
    }

    fn observation(metric: CookwareMetric, value: i64) -> CookwareObservationV1 {
        CookwareObservationV1 {
            metric,
            value,
            evidence_class: CookwareEvidenceClass::ObservedInstrumental,
            evidence_ref: format!("evidence://measurement/{metric:?}/{value}"),
            measurement_method_ref: "method://instrumented-cookware-v1".to_owned(),
            calibration_ref: Some("calibration://thermocouple-set-v2".to_owned()),
            uncertainty_abs: Some(100),
        }
    }

    fn run(candidate_id: &str, block_id: &str, unit_id: &str, order_index: u32) -> CookwareRunV1 {
        CookwareRunV1 {
            run_id: format!("{candidate_id}-{block_id}"),
            candidate_id: candidate_id.to_owned(),
            cookware_unit_id: unit_id.to_owned(),
            block_id: block_id.to_owned(),
            order_index,
            protocol_id: "thermal-grid-v1".to_owned(),
            task_profile_id: "saute-v1".to_owned(),
            geometry_profile_id: "matched-28cm-v1".to_owned(),
            heat_source_profile_id: "induction-profile-v1".to_owned(),
            load_profile_id: "water-load-1kg-v1".to_owned(),
            ambient_profile_id: "lab-ambient-v1".to_owned(),
            observations: vec![
                observation(CookwareMetric::SurfaceTemperatureSpreadMilliCelsius, 8000),
                observation(CookwareMetric::HeatUpEnergyMilliJoule, 420000),
                observation(CookwareMetric::RecoveryTimeMillis, 12000),
            ],
        }
    }

    fn manifest() -> CookwareBenchmarkManifestV1 {
        CookwareBenchmarkManifestV1 {
            schema_version: COOKWARE_BENCHMARK_SCHEMA_VERSION,
            benchmark_id: "cookware-matched-comparison-v1".to_owned(),
            protocol: protocol(),
            candidates: vec![
                candidate("reference-pan", CookwareRole::Reference),
                candidate("candidate-pan", CookwareRole::Candidate),
            ],
            runs: vec![
                run("reference-pan", "block-1", "ref-unit-1", 1),
                run("candidate-pan", "block-1", "cand-unit-1", 2),
                run("candidate-pan", "block-2", "cand-unit-2", 1),
                run("reference-pan", "block-2", "ref-unit-2", 2),
            ],
            craft_experience: Vec::new(),
        }
    }

    #[test]
    fn matched_replicated_empirical_comparison_is_accepted() {
        let report = validate_cookware_benchmark(&manifest());
        assert!(report.is_valid(), "{:?}", report.violations);
        assert_eq!(report.candidate_count, 2);
        assert_eq!(report.block_count, 2);
    }

    #[test]
    fn incomplete_matched_block_is_rejected() {
        let mut input = manifest();
        input.runs.pop();
        let report = validate_cookware_benchmark(&input);
        assert!(report
            .violations
            .iter()
            .any(|item| item.code == CookwareViolationCode::IncompleteBlock));
    }

    #[test]
    fn simulation_cannot_satisfy_empirical_primary_metric() {
        let mut input = manifest();
        input.runs[0].observations[0].evidence_class = CookwareEvidenceClass::Simulated;
        let report = validate_cookware_benchmark(&input);
        assert!(report.violations.iter().any(|item| {
            item.code == CookwareViolationCode::WrongEvidenceClassForEmpiricalClaim
                && item.run_id.as_deref() == Some("reference-pan-block-1")
        }));
    }

    #[test]
    fn uncalibrated_instrument_observation_is_rejected() {
        let mut input = manifest();
        input.runs[0].observations[0].calibration_ref = None;
        let report = validate_cookware_benchmark(&input);
        assert!(report
            .violations
            .iter()
            .any(|item| item.code == CookwareViolationCode::MissingCalibrationReference));
    }

    #[test]
    fn food_contact_claim_requires_scoped_qualification_evidence() {
        let mut input = manifest();
        input.protocol.claims_food_contact_suitability = true;
        let report = validate_cookware_benchmark(&input);
        assert_eq!(
            report
                .violations
                .iter()
                .filter(|item| item.code == CookwareViolationCode::MissingFoodContactQualification)
                .count(),
            2
        );
    }

    #[test]
    fn required_metric_cannot_be_silently_omitted() {
        let mut input = manifest();
        input.runs[0].observations.retain(|item| {
            item.metric != CookwareMetric::RecoveryTimeMillis
        });
        let report = validate_cookware_benchmark(&input);
        assert!(report
            .violations
            .iter()
            .any(|item| item.code == CookwareViolationCode::MissingRequiredMetric));
    }

    #[test]
    fn practitioner_ratings_require_consent_and_do_not_collapse_dimensions() {
        let mut input = manifest();
        input.craft_experience.push(CraftExperienceRecordV1 {
            record_id: "craft-1".to_owned(),
            participant_pseudonym: "pseudonym-7".to_owned(),
            consent_receipt_ref: "consent://study-v1/pseudonym-7".to_owned(),
            instrument_ref: "survey://craft-experience-v1".to_owned(),
            candidate_id: "candidate-pan".to_owned(),
            assistance_mode: AssistanceMode::Manual,
            ratings: vec![
                CraftExperienceRatingV1 {
                    dimension: CraftExperienceDimension::PerceivedControl,
                    rating_1_to_7: 6,
                },
                CraftExperienceRatingV1 {
                    dimension: CraftExperienceDimension::CreativeAgency,
                    rating_1_to_7: 7,
                },
            ],
        });
        assert!(validate_cookware_benchmark(&input).is_valid());
        assert_eq!(input.craft_experience[0].ratings.len(), 2);
    }

    #[test]
    fn missing_consent_and_invalid_craft_rating_are_rejected() {
        let mut input = manifest();
        input.craft_experience.push(CraftExperienceRecordV1 {
            record_id: "craft-2".to_owned(),
            participant_pseudonym: "pseudonym-8".to_owned(),
            consent_receipt_ref: String::new(),
            instrument_ref: "survey://craft-experience-v1".to_owned(),
            candidate_id: "candidate-pan".to_owned(),
            assistance_mode: AssistanceMode::OptionalAssistance,
            ratings: vec![CraftExperienceRatingV1 {
                dimension: CraftExperienceDimension::Enjoyment,
                rating_1_to_7: 9,
            }],
        });
        let report = validate_cookware_benchmark(&input);
        assert!(report
            .violations
            .iter()
            .any(|item| item.code == CookwareViolationCode::MissingCraftConsent));
        assert!(report
            .violations
            .iter()
            .any(|item| item.code == CookwareViolationCode::InvalidCraftRating));
    }

    #[test]
    fn mismatched_protocol_context_is_rejected() {
        let mut input = manifest();
        input.runs[1].load_profile_id = "different-load".to_owned();
        let report = validate_cookware_benchmark(&input);
        assert!(report
            .violations
            .iter()
            .any(|item| item.code == CookwareViolationCode::ContextMismatch));
    }

    #[test]
    fn no_reference_candidate_is_rejected() {
        let mut input = manifest();
        input.candidates[0].role = CookwareRole::Candidate;
        let report = validate_cookware_benchmark(&input);
        assert!(report
            .violations
            .iter()
            .any(|item| item.code == CookwareViolationCode::MissingReferenceCandidate));
    }

    #[test]
    fn physical_unit_cannot_be_claimed_by_two_design_candidates() {
        let mut input = manifest();
        input.runs[1].cookware_unit_id = "ref-unit-1".to_owned();
        let report = validate_cookware_benchmark(&input);
        assert!(report.violations.iter().any(|item| {
            item.code == CookwareViolationCode::CookwareUnitIdentityConflict
                && item.run_id.as_deref() == Some("candidate-pan-block-1")
        }));
    }

}
