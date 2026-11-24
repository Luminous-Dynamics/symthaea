// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Preregistered, source-bound fungal microscopy observer-effect trial contracts.
//!
//! This module treats acquisition policy as a possible biological treatment.
//! It validates treatment allocation, experimental-unit and interference
//! boundaries, exposure linkage, endpoint definitions and observation coverage.
//! It does not prove that randomization was executed correctly, estimate causal
//! effects, establish biological neutrality or control microscope hardware.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::mycelial_photobiological_exposure::{
    MycelialAmbientLightState, MycelialIlluminationPurpose, MycelialIlluminationScheduleKind,
    MycelialPhotobiologicalExposureRecord, MycelialPhotobiologicalMeasurement,
    MycelialPhotobiologicalObservationWindow, MycelialPhotobiologicalResponseObservation,
};

pub const MYCELIAL_OBSERVER_EFFECT_TRIAL_SCHEMA_VERSION: u16 = 1;
pub const MYCELIAL_OBSERVER_EFFECT_REPORT_SCHEMA_VERSION: u16 = 1;
pub const MAX_MYCELIAL_OBSERVER_EFFECT_ARMS: usize = 16;
pub const MAX_MYCELIAL_OBSERVER_EFFECT_ENDPOINTS: usize = 64;
pub const MAX_MYCELIAL_OBSERVER_EFFECT_UNITS: usize = 4_096;
pub const MAX_MYCELIAL_OBSERVER_EFFECT_OBSERVATIONS: usize = 1_000_000;
pub const MAX_MYCELIAL_OBSERVER_EFFECT_ID_BYTES: usize = 128;
pub const MAX_MYCELIAL_OBSERVER_EFFECT_DESCRIPTION_BYTES: usize = 512;
const DIGEST_HEX_BYTES: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MycelialObserverEffectArmKind {
    DarkControl,
    AmbientControl,
    HandlingSham,
    LowExposureImaging,
    ContinuousImaging,
    FractionatedImaging,
    DeliberatePhotoperiod,
    Photostimulation,
    Photoinactivation,
    Other,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MycelialObserverEffectArm {
    pub arm_id: String,
    pub arm_kind: MycelialObserverEffectArmKind,
    pub is_reference_comparator: bool,
    pub description: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MycelialObserverEffectRandomizationStrategy {
    Simple,
    Blocked,
    Stratified,
    ClusterRandomized,
    BlockedClusterRandomized,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MycelialObserverEffectEndpointKind {
    HyphalTipExtension,
    BranchingDensity,
    CordAreaFraction,
    ConidiationCount,
    ViabilityFraction,
    ReactiveOxygenSpeciesProxy,
    SecondaryMetaboliteConcentration,
    FruitingInitiationLatency,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MycelialObserverEffectEndpointDefinition {
    pub endpoint_id: String,
    pub endpoint_kind: MycelialObserverEffectEndpointKind,
    /// Analysis unit after the preregistered endpoint transformation, e.g. um/hour.
    pub analysis_unit_id: String,
    /// Smallest biologically important effect, in analysis_unit_id.
    pub smallest_effect_size_of_interest: f64,
    /// Prespecified equivalence margin, in analysis_unit_id.
    pub equivalence_margin: f64,
    pub primary: bool,
    /// Required only for analyte-specific metabolite endpoints.
    pub analyte_id: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MycelialObserverEffectEstimandKind {
    MeanDifference,
    MedianDifference,
    RiskDifference,
    RateDifference,
    LogRateRatio,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MycelialObserverEffectEstimand {
    pub endpoint_id: String,
    pub treatment_arm_id: String,
    pub comparator_arm_id: String,
    pub estimand_kind: MycelialObserverEffectEstimandKind,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MycelialObserverEffectAssignment {
    pub assignment_id: String,
    pub sample_id: String,
    pub biological_replicate_id: String,
    pub cultivation_unit_id: String,
    /// Shared chamber, enclosure, plate region or other potential interference unit.
    /// Every cultivation unit in one cluster must receive the same arm.
    pub interference_cluster_id: String,
    pub randomization_block_id: String,
    /// Required for stratified and blocked-cluster randomization.
    pub stratum_evidence_digest: Option<String>,
    pub treatment_arm_id: String,
    pub assigned_at_ns: u64,
    /// Canonical digest of this immutable assignment record.
    pub assignment_digest: String,
    /// Receipt from the allocation system; a digest alone does not prove randomness.
    pub assignment_receipt_digest: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MycelialObserverEffectTrial {
    pub schema_version: u16,
    pub trial_id: String,
    pub experiment_id: String,
    pub protocol_digest: String,
    pub power_analysis_digest: String,
    pub timebase_id: String,
    pub randomization_algorithm_id: String,
    pub randomization_algorithm_version: String,
    pub randomization_strategy: MycelialObserverEffectRandomizationStrategy,
    /// External allocation receipt that permits independent audit of the randomization.
    pub allocation_receipt_digest: String,
    pub allocation_committed_at_ns: u64,
    pub assessor_blinded: bool,
    /// Required when outcome assessors were not blinded.
    pub unblinding_or_unblinded_assessment_receipt_digest: Option<String>,
    /// Power/sample-size plan counts independent interference clusters, not nested vessels.
    pub minimum_independent_clusters_per_arm: u32,
    pub planned_independent_clusters_per_arm: u32,
    pub arms: Vec<MycelialObserverEffectArm>,
    pub endpoints: Vec<MycelialObserverEffectEndpointDefinition>,
    pub estimands: Vec<MycelialObserverEffectEstimand>,
    pub assignments: Vec<MycelialObserverEffectAssignment>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MycelialObserverEffectEndpointObservation {
    pub endpoint_id: String,
    pub response: MycelialPhotobiologicalResponseObservation,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MycelialObserverEffectEndpointCoverage {
    pub endpoint_id: String,
    pub cultivation_unit_id: String,
    pub has_baseline: bool,
    pub has_post_exposure: bool,
    pub explicit_missing_observation_count: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MycelialObserverEffectTrialReport {
    pub schema_version: u16,
    pub trial_digest: String,
    pub assignment_inventory_digest: String,
    pub exposure_inventory_digest: String,
    pub observation_inventory_digest: String,
    pub assigned_cultivation_unit_count: u32,
    pub interference_cluster_count: u32,
    pub exposure_record_count: u32,
    pub endpoint_observation_count: u32,
    pub explicit_missing_observation_count: u32,
    pub primary_endpoint_coverage: Vec<MycelialObserverEffectEndpointCoverage>,
    /// This registry validates design and provenance; it does not estimate a causal effect.
    pub causal_effect_estimated: bool,
    pub canonical_digest: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MycelialObserverEffectTrialError {
    InvalidSchemaVersion(u16),
    InvalidIdentifier(&'static str),
    InvalidDigest(&'static str),
    InvalidTrial(&'static str),
    InvalidArm(&'static str),
    InvalidEndpoint(&'static str),
    InvalidAssignment(&'static str),
    InvalidEstimand(&'static str),
    DuplicateId(&'static str),
    UnknownArm,
    UnknownEndpoint,
    AssignmentExposureMismatch(&'static str),
    InterferenceClusterSplit,
    ExposureAssignmentCountMismatch,
    EndpointMeasurementMismatch,
    EndpointCoverageIncomplete,
    ReportMismatch,
    ResourceLimitExceeded(&'static str),
    ExposureValidationFailed,
    ResponseValidationFailed,
}

impl fmt::Display for MycelialObserverEffectTrialError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSchemaVersion(version) => {
                write!(
                    f,
                    "unsupported Mycelial observer-effect schema version {version}"
                )
            }
            Self::InvalidIdentifier(field) => {
                write!(f, "invalid observer-effect identifier: {field}")
            }
            Self::InvalidDigest(field) => {
                write!(f, "invalid observer-effect evidence digest: {field}")
            }
            Self::InvalidTrial(field) => {
                write!(f, "invalid observer-effect trial: {field}")
            }
            Self::InvalidArm(field) => {
                write!(f, "invalid observer-effect treatment arm: {field}")
            }
            Self::InvalidEndpoint(field) => {
                write!(f, "invalid observer-effect endpoint definition: {field}")
            }
            Self::InvalidAssignment(field) => {
                write!(f, "invalid observer-effect assignment: {field}")
            }
            Self::InvalidEstimand(field) => {
                write!(f, "invalid observer-effect estimand: {field}")
            }
            Self::DuplicateId(field) => {
                write!(f, "duplicate observer-effect identity: {field}")
            }
            Self::UnknownArm => write!(f, "assignment references an unregistered treatment arm"),
            Self::UnknownEndpoint => write!(
                f,
                "estimand or observation references an unregistered endpoint"
            ),
            Self::AssignmentExposureMismatch(field) => {
                write!(
                    f,
                    "exposure disagrees with preregistered assignment: {field}"
                )
            }
            Self::InterferenceClusterSplit => {
                write!(
                    f,
                    "one illumination/interference cluster is split across treatment arms"
                )
            }
            Self::ExposureAssignmentCountMismatch => {
                write!(
                    f,
                    "each randomized cultivation unit must have exactly one registered treatment exposure"
                )
            }
            Self::EndpointMeasurementMismatch => {
                write!(
                    f,
                    "typed response measurement does not match its preregistered endpoint"
                )
            }
            Self::EndpointCoverageIncomplete => {
                write!(
                    f,
                    "a primary endpoint lacks a baseline or post-exposure observation for an assigned unit"
                )
            }
            Self::ReportMismatch => {
                write!(
                    f,
                    "observer-effect report differs from independent evidence replay"
                )
            }
            Self::ResourceLimitExceeded(field) => {
                write!(f, "observer-effect resource limit exceeded: {field}")
            }
            Self::ExposureValidationFailed => {
                write!(f, "source-bound photobiological exposure validation failed")
            }
            Self::ResponseValidationFailed => {
                write!(f, "source-bound photobiological response validation failed")
            }
        }
    }
}

impl Error for MycelialObserverEffectTrialError {}

fn valid_id(value: &str, maximum_bytes: usize) -> bool {
    !value.trim().is_empty() && value.len() <= maximum_bytes && !value.chars().any(char::is_control)
}

fn valid_digest(value: &str) -> bool {
    value.len() == DIGEST_HEX_BYTES
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn validate_id(value: &str, field: &'static str) -> Result<(), MycelialObserverEffectTrialError> {
    if !valid_id(value, MAX_MYCELIAL_OBSERVER_EFFECT_ID_BYTES) {
        return Err(MycelialObserverEffectTrialError::InvalidIdentifier(field));
    }
    Ok(())
}

fn validate_digest(
    value: &str,
    field: &'static str,
) -> Result<(), MycelialObserverEffectTrialError> {
    if !valid_digest(value) {
        return Err(MycelialObserverEffectTrialError::InvalidDigest(field));
    }
    Ok(())
}

fn hash_string(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

fn hash_f64(hasher: &mut blake3::Hasher, value: f64) {
    hasher.update(&value.to_bits().to_le_bytes());
}

fn finish_digest(hasher: blake3::Hasher) -> String {
    hasher.finalize().to_hex().to_string()
}

fn arm_kind_tag(kind: MycelialObserverEffectArmKind) -> u8 {
    match kind {
        MycelialObserverEffectArmKind::DarkControl => 0,
        MycelialObserverEffectArmKind::AmbientControl => 1,
        MycelialObserverEffectArmKind::HandlingSham => 2,
        MycelialObserverEffectArmKind::LowExposureImaging => 3,
        MycelialObserverEffectArmKind::ContinuousImaging => 4,
        MycelialObserverEffectArmKind::FractionatedImaging => 5,
        MycelialObserverEffectArmKind::DeliberatePhotoperiod => 6,
        MycelialObserverEffectArmKind::Photostimulation => 7,
        MycelialObserverEffectArmKind::Photoinactivation => 8,
        MycelialObserverEffectArmKind::Other => 9,
    }
}

fn endpoint_kind_tag(kind: MycelialObserverEffectEndpointKind) -> u8 {
    match kind {
        MycelialObserverEffectEndpointKind::HyphalTipExtension => 0,
        MycelialObserverEffectEndpointKind::BranchingDensity => 1,
        MycelialObserverEffectEndpointKind::CordAreaFraction => 2,
        MycelialObserverEffectEndpointKind::ConidiationCount => 3,
        MycelialObserverEffectEndpointKind::ViabilityFraction => 4,
        MycelialObserverEffectEndpointKind::ReactiveOxygenSpeciesProxy => 5,
        MycelialObserverEffectEndpointKind::SecondaryMetaboliteConcentration => 6,
        MycelialObserverEffectEndpointKind::FruitingInitiationLatency => 7,
    }
}

fn randomization_strategy_tag(strategy: MycelialObserverEffectRandomizationStrategy) -> u8 {
    match strategy {
        MycelialObserverEffectRandomizationStrategy::Simple => 0,
        MycelialObserverEffectRandomizationStrategy::Blocked => 1,
        MycelialObserverEffectRandomizationStrategy::Stratified => 2,
        MycelialObserverEffectRandomizationStrategy::ClusterRandomized => 3,
        MycelialObserverEffectRandomizationStrategy::BlockedClusterRandomized => 4,
    }
}

fn estimand_kind_tag(kind: MycelialObserverEffectEstimandKind) -> u8 {
    match kind {
        MycelialObserverEffectEstimandKind::MeanDifference => 0,
        MycelialObserverEffectEstimandKind::MedianDifference => 1,
        MycelialObserverEffectEstimandKind::RiskDifference => 2,
        MycelialObserverEffectEstimandKind::RateDifference => 3,
        MycelialObserverEffectEstimandKind::LogRateRatio => 4,
    }
}

fn measurement_matches_endpoint(
    measurement: &MycelialPhotobiologicalMeasurement,
    endpoint: &MycelialObserverEffectEndpointDefinition,
) -> bool {
    match (measurement, endpoint.endpoint_kind) {
        (
            MycelialPhotobiologicalMeasurement::HyphalTipExtension { .. },
            MycelialObserverEffectEndpointKind::HyphalTipExtension,
        )
        | (
            MycelialPhotobiologicalMeasurement::BranchingDensity { .. },
            MycelialObserverEffectEndpointKind::BranchingDensity,
        )
        | (
            MycelialPhotobiologicalMeasurement::CordAreaFraction { .. },
            MycelialObserverEffectEndpointKind::CordAreaFraction,
        )
        | (
            MycelialPhotobiologicalMeasurement::ConidiationCount { .. },
            MycelialObserverEffectEndpointKind::ConidiationCount,
        )
        | (
            MycelialPhotobiologicalMeasurement::ViabilityFraction { .. },
            MycelialObserverEffectEndpointKind::ViabilityFraction,
        )
        | (
            MycelialPhotobiologicalMeasurement::ReactiveOxygenSpeciesProxy { .. },
            MycelialObserverEffectEndpointKind::ReactiveOxygenSpeciesProxy,
        )
        | (
            MycelialPhotobiologicalMeasurement::FruitingInitiationLatency { .. },
            MycelialObserverEffectEndpointKind::FruitingInitiationLatency,
        ) => true,
        (
            MycelialPhotobiologicalMeasurement::SecondaryMetaboliteConcentration {
                analyte_id, ..
            },
            MycelialObserverEffectEndpointKind::SecondaryMetaboliteConcentration,
        ) => endpoint.analyte_id.as_deref() == Some(analyte_id.as_str()),
        (MycelialPhotobiologicalMeasurement::MeasurementUnavailable { .. }, _) => true,
        _ => false,
    }
}

impl MycelialObserverEffectTrial {
    pub fn validate(&self) -> Result<(), MycelialObserverEffectTrialError> {
        if self.schema_version != MYCELIAL_OBSERVER_EFFECT_TRIAL_SCHEMA_VERSION {
            return Err(MycelialObserverEffectTrialError::InvalidSchemaVersion(
                self.schema_version,
            ));
        }
        for (value, field) in [
            (self.trial_id.as_str(), "trial_id"),
            (self.experiment_id.as_str(), "experiment_id"),
            (self.timebase_id.as_str(), "timebase_id"),
            (
                self.randomization_algorithm_id.as_str(),
                "randomization_algorithm_id",
            ),
            (
                self.randomization_algorithm_version.as_str(),
                "randomization_algorithm_version",
            ),
        ] {
            validate_id(value, field)?;
        }
        for (value, field) in [
            (self.protocol_digest.as_str(), "protocol_digest"),
            (self.power_analysis_digest.as_str(), "power_analysis_digest"),
            (
                self.allocation_receipt_digest.as_str(),
                "allocation_receipt_digest",
            ),
        ] {
            validate_digest(value, field)?;
        }
        if self.allocation_committed_at_ns == 0
            || self.minimum_independent_clusters_per_arm < 2
            || self.planned_independent_clusters_per_arm < self.minimum_independent_clusters_per_arm
            || self.arms.len() < 2
            || self.arms.len() > MAX_MYCELIAL_OBSERVER_EFFECT_ARMS
            || self.endpoints.is_empty()
            || self.endpoints.len() > MAX_MYCELIAL_OBSERVER_EFFECT_ENDPOINTS
            || self.assignments.is_empty()
            || self.assignments.len() > MAX_MYCELIAL_OBSERVER_EFFECT_UNITS
        {
            return Err(MycelialObserverEffectTrialError::InvalidTrial(
                "bounded arms, endpoints, assignments and a power-derived unit plan are required",
            ));
        }
        if self.assessor_blinded {
            if self
                .unblinding_or_unblinded_assessment_receipt_digest
                .is_some()
            {
                return Err(MycelialObserverEffectTrialError::InvalidTrial(
                    "blinded assessment must not include an unblinding receipt",
                ));
            }
        } else {
            let Some(receipt) = self
                .unblinding_or_unblinded_assessment_receipt_digest
                .as_deref()
            else {
                return Err(MycelialObserverEffectTrialError::InvalidTrial(
                    "unblinded assessment requires an explicit receipt",
                ));
            };
            validate_digest(receipt, "unblinding_or_unblinded_assessment_receipt_digest")?;
        }

        let mut arm_ids = BTreeSet::new();
        let mut comparator_count = 0usize;
        for arm in &self.arms {
            validate_id(arm.arm_id.as_str(), "arm.arm_id")?;
            if arm.description.trim().is_empty()
                || arm.description.len() > MAX_MYCELIAL_OBSERVER_EFFECT_DESCRIPTION_BYTES
                || arm.description.chars().any(char::is_control)
            {
                return Err(MycelialObserverEffectTrialError::InvalidArm(
                    "arm description must be bounded and non-empty",
                ));
            }
            if !arm_ids.insert(arm.arm_id.as_str()) {
                return Err(MycelialObserverEffectTrialError::DuplicateId("arm_id"));
            }
            if arm.is_reference_comparator {
                comparator_count += 1;
            }
        }
        if comparator_count == 0 {
            return Err(MycelialObserverEffectTrialError::InvalidArm(
                "at least one explicit reference comparator is required",
            ));
        }

        let mut endpoint_ids = BTreeSet::new();
        let mut primary_endpoint_ids = BTreeSet::new();
        for endpoint in &self.endpoints {
            validate_id(endpoint.endpoint_id.as_str(), "endpoint.endpoint_id")?;
            validate_id(
                endpoint.analysis_unit_id.as_str(),
                "endpoint.analysis_unit_id",
            )?;
            if !endpoint.smallest_effect_size_of_interest.is_finite()
                || endpoint.smallest_effect_size_of_interest <= 0.0
                || !endpoint.equivalence_margin.is_finite()
                || endpoint.equivalence_margin <= 0.0
            {
                return Err(MycelialObserverEffectTrialError::InvalidEndpoint(
                    "endpoint requires positive finite SESOI and equivalence margin",
                ));
            }
            match (endpoint.endpoint_kind, endpoint.analyte_id.as_deref()) {
                (
                    MycelialObserverEffectEndpointKind::SecondaryMetaboliteConcentration,
                    Some(analyte_id),
                ) => validate_id(analyte_id, "endpoint.analyte_id")?,
                (MycelialObserverEffectEndpointKind::SecondaryMetaboliteConcentration, None) => {
                    return Err(MycelialObserverEffectTrialError::InvalidEndpoint(
                        "secondary-metabolite endpoint requires an analyte identity",
                    ));
                }
                (_, Some(_)) => {
                    return Err(MycelialObserverEffectTrialError::InvalidEndpoint(
                        "analyte identity is valid only for a metabolite endpoint",
                    ));
                }
                (_, None) => {}
            }
            if !endpoint_ids.insert(endpoint.endpoint_id.as_str()) {
                return Err(MycelialObserverEffectTrialError::DuplicateId("endpoint_id"));
            }
            if endpoint.primary {
                primary_endpoint_ids.insert(endpoint.endpoint_id.as_str());
            }
        }
        if primary_endpoint_ids.is_empty() {
            return Err(MycelialObserverEffectTrialError::InvalidEndpoint(
                "at least one primary endpoint must be preregistered",
            ));
        }

        let arm_by_id = self
            .arms
            .iter()
            .map(|arm| (arm.arm_id.as_str(), arm))
            .collect::<BTreeMap<_, _>>();
        let endpoint_by_id = self
            .endpoints
            .iter()
            .map(|endpoint| (endpoint.endpoint_id.as_str(), endpoint))
            .collect::<BTreeMap<_, _>>();

        let mut assignment_ids = BTreeSet::new();
        let mut cultivation_units = BTreeSet::new();
        let mut biological_replicates = BTreeSet::new();
        let mut cluster_arm = BTreeMap::<&str, &str>::new();
        let mut arm_interference_clusters = BTreeMap::<&str, BTreeSet<&str>>::new();

        for assignment in &self.assignments {
            for (value, field) in [
                (
                    assignment.assignment_id.as_str(),
                    "assignment.assignment_id",
                ),
                (assignment.sample_id.as_str(), "assignment.sample_id"),
                (
                    assignment.biological_replicate_id.as_str(),
                    "assignment.biological_replicate_id",
                ),
                (
                    assignment.cultivation_unit_id.as_str(),
                    "assignment.cultivation_unit_id",
                ),
                (
                    assignment.interference_cluster_id.as_str(),
                    "assignment.interference_cluster_id",
                ),
                (
                    assignment.randomization_block_id.as_str(),
                    "assignment.randomization_block_id",
                ),
                (
                    assignment.treatment_arm_id.as_str(),
                    "assignment.treatment_arm_id",
                ),
            ] {
                validate_id(value, field)?;
            }
            validate_digest(
                assignment.assignment_digest.as_str(),
                "assignment.assignment_digest",
            )?;
            validate_digest(
                assignment.assignment_receipt_digest.as_str(),
                "assignment.assignment_receipt_digest",
            )?;
            if assignment.assignment_digest
                != mycelial_observer_effect_assignment_digest(assignment)
            {
                return Err(MycelialObserverEffectTrialError::InvalidAssignment(
                    "assignment fields do not match their canonical digest",
                ));
            }
            if assignment.assigned_at_ns == 0
                || assignment.assigned_at_ns > self.allocation_committed_at_ns
            {
                return Err(MycelialObserverEffectTrialError::InvalidAssignment(
                    "assignment timestamp must precede or equal the allocation commit",
                ));
            }
            if let Some(digest) = assignment.stratum_evidence_digest.as_deref() {
                validate_digest(digest, "assignment.stratum_evidence_digest")?;
            }
            if matches!(
                self.randomization_strategy,
                MycelialObserverEffectRandomizationStrategy::Stratified
                    | MycelialObserverEffectRandomizationStrategy::BlockedClusterRandomized
            ) && assignment.stratum_evidence_digest.is_none()
            {
                return Err(MycelialObserverEffectTrialError::InvalidAssignment(
                    "stratified allocation requires a stratum evidence digest",
                ));
            }
            if !assignment_ids.insert(assignment.assignment_id.as_str()) {
                return Err(MycelialObserverEffectTrialError::DuplicateId(
                    "assignment_id",
                ));
            }
            if !cultivation_units.insert(assignment.cultivation_unit_id.as_str()) {
                return Err(MycelialObserverEffectTrialError::DuplicateId(
                    "cultivation_unit_id",
                ));
            }
            if !biological_replicates.insert(assignment.biological_replicate_id.as_str()) {
                return Err(MycelialObserverEffectTrialError::DuplicateId(
                    "biological_replicate_id",
                ));
            }
            if !arm_by_id.contains_key(assignment.treatment_arm_id.as_str()) {
                return Err(MycelialObserverEffectTrialError::UnknownArm);
            }
            if let Some(previous_arm) = cluster_arm.insert(
                assignment.interference_cluster_id.as_str(),
                assignment.treatment_arm_id.as_str(),
            ) && previous_arm != assignment.treatment_arm_id
            {
                return Err(MycelialObserverEffectTrialError::InterferenceClusterSplit);
            }
            arm_interference_clusters
                .entry(assignment.treatment_arm_id.as_str())
                .or_default()
                .insert(assignment.interference_cluster_id.as_str());
        }

        for arm in &self.arms {
            let count = arm_interference_clusters
                .get(arm.arm_id.as_str())
                .map(BTreeSet::len)
                .unwrap_or(0) as u32;
            if count < self.minimum_independent_clusters_per_arm
                || count > self.planned_independent_clusters_per_arm
            {
                return Err(MycelialObserverEffectTrialError::InvalidTrial(
                    "each arm's independent interference-cluster count must remain within the preregistered power plan",
                ));
            }
        }

        let mut estimand_keys = BTreeSet::new();
        for estimand in &self.estimands {
            validate_id(estimand.endpoint_id.as_str(), "estimand.endpoint_id")?;
            validate_id(
                estimand.treatment_arm_id.as_str(),
                "estimand.treatment_arm_id",
            )?;
            validate_id(
                estimand.comparator_arm_id.as_str(),
                "estimand.comparator_arm_id",
            )?;
            let Some(endpoint) = endpoint_by_id.get(estimand.endpoint_id.as_str()) else {
                return Err(MycelialObserverEffectTrialError::UnknownEndpoint);
            };
            let Some(treatment) = arm_by_id.get(estimand.treatment_arm_id.as_str()) else {
                return Err(MycelialObserverEffectTrialError::UnknownArm);
            };
            let Some(comparator) = arm_by_id.get(estimand.comparator_arm_id.as_str()) else {
                return Err(MycelialObserverEffectTrialError::UnknownArm);
            };
            if estimand.treatment_arm_id == estimand.comparator_arm_id
                || treatment.is_reference_comparator
                || !comparator.is_reference_comparator
                || !endpoint.primary
            {
                return Err(MycelialObserverEffectTrialError::InvalidEstimand(
                    "estimand must compare a non-reference treatment against a reference arm for a primary endpoint",
                ));
            }
            if !estimand_keys.insert((
                estimand.endpoint_id.as_str(),
                estimand.treatment_arm_id.as_str(),
                estimand.comparator_arm_id.as_str(),
            )) {
                return Err(MycelialObserverEffectTrialError::DuplicateId("estimand"));
            }
        }
        for endpoint_id in primary_endpoint_ids {
            for arm in &self.arms {
                if arm.is_reference_comparator {
                    continue;
                }
                if !self.estimands.iter().any(|estimand| {
                    estimand.endpoint_id == endpoint_id && estimand.treatment_arm_id == arm.arm_id
                }) {
                    return Err(MycelialObserverEffectTrialError::InvalidEstimand(
                        "every primary endpoint and non-reference arm needs a preregistered comparator estimand",
                    ));
                }
            }
        }
        Ok(())
    }

    pub fn validate_against_evidence(
        &self,
        exposures: &[MycelialPhotobiologicalExposureRecord],
        observations: &[MycelialObserverEffectEndpointObservation],
    ) -> Result<MycelialObserverEffectTrialReport, MycelialObserverEffectTrialError> {
        self.validate()?;
        if exposures.len() != self.assignments.len() {
            return Err(MycelialObserverEffectTrialError::ExposureAssignmentCountMismatch);
        }
        if exposures.len() > MAX_MYCELIAL_OBSERVER_EFFECT_UNITS
            || observations.len() > MAX_MYCELIAL_OBSERVER_EFFECT_OBSERVATIONS
        {
            return Err(MycelialObserverEffectTrialError::ResourceLimitExceeded(
                "exposure or endpoint observation inventory",
            ));
        }

        let assignments_by_unit = self
            .assignments
            .iter()
            .map(|assignment| (assignment.cultivation_unit_id.as_str(), assignment))
            .collect::<BTreeMap<_, _>>();
        let arms_by_id = self
            .arms
            .iter()
            .map(|arm| (arm.arm_id.as_str(), arm))
            .collect::<BTreeMap<_, _>>();
        let endpoints_by_id = self
            .endpoints
            .iter()
            .map(|endpoint| (endpoint.endpoint_id.as_str(), endpoint))
            .collect::<BTreeMap<_, _>>();

        let mut exposure_by_id = BTreeMap::<&str, &MycelialPhotobiologicalExposureRecord>::new();
        let mut exposure_by_unit = BTreeMap::<&str, &MycelialPhotobiologicalExposureRecord>::new();
        for exposure in exposures {
            exposure
                .validate()
                .map_err(|_| MycelialObserverEffectTrialError::ExposureValidationFailed)?;
            if exposure.experiment_id != self.experiment_id
                || exposure.timebase_id != self.timebase_id
                || exposure.exposure_start_ns <= self.allocation_committed_at_ns
            {
                return Err(
                    MycelialObserverEffectTrialError::AssignmentExposureMismatch(
                        "experiment, timebase or allocation-before-exposure ordering",
                    ),
                );
            }
            let Some(assignment) = assignments_by_unit.get(exposure.cultivation_unit_id.as_str())
            else {
                return Err(
                    MycelialObserverEffectTrialError::AssignmentExposureMismatch(
                        "exposure cultivation unit is not randomized",
                    ),
                );
            };
            if exposure.sample_id != assignment.sample_id
                || exposure.biological_replicate_id != assignment.biological_replicate_id
                || exposure.treatment_arm_id != assignment.treatment_arm_id
            {
                return Err(
                    MycelialObserverEffectTrialError::AssignmentExposureMismatch(
                        "sample, biological replicate or treatment arm",
                    ),
                );
            }
            let Some(arm) = arms_by_id.get(assignment.treatment_arm_id.as_str()) else {
                return Err(MycelialObserverEffectTrialError::UnknownArm);
            };
            if !exposure_purpose_matches_arm(
                arm.arm_kind,
                exposure.illumination_purpose,
                exposure.schedule_kind,
                exposure.ambient_light.state,
            ) {
                return Err(
                    MycelialObserverEffectTrialError::AssignmentExposureMismatch(
                        "illumination purpose or schedule differs from the preregistered arm",
                    ),
                );
            }
            if exposure_by_id
                .insert(exposure.exposure_id.as_str(), exposure)
                .is_some()
            {
                return Err(MycelialObserverEffectTrialError::DuplicateId("exposure_id"));
            }
            if exposure_by_unit
                .insert(exposure.cultivation_unit_id.as_str(), exposure)
                .is_some()
            {
                return Err(MycelialObserverEffectTrialError::DuplicateId(
                    "exposure per cultivation unit",
                ));
            }
        }

        let mut observation_ids = BTreeSet::new();
        let mut coverage = BTreeMap::<(String, String), (bool, bool, u32)>::new();
        let mut observation_count = 0u32;
        let mut missing_count = 0u32;
        for item in observations {
            validate_id(item.endpoint_id.as_str(), "observation.endpoint_id")?;
            let Some(endpoint) = endpoints_by_id.get(item.endpoint_id.as_str()) else {
                return Err(MycelialObserverEffectTrialError::UnknownEndpoint);
            };
            item.response
                .validate()
                .map_err(|_| MycelialObserverEffectTrialError::ResponseValidationFailed)?;
            if !measurement_matches_endpoint(&item.response.measurement, endpoint) {
                return Err(MycelialObserverEffectTrialError::EndpointMeasurementMismatch);
            }
            let Some(exposure) = exposure_by_id.get(item.response.exposure_id.as_str()) else {
                return Err(
                    MycelialObserverEffectTrialError::AssignmentExposureMismatch(
                        "response references an unregistered exposure",
                    ),
                );
            };
            item.response
                .validate_against_exposure(exposure)
                .map_err(|_| MycelialObserverEffectTrialError::ResponseValidationFailed)?;
            if item.response.experiment_id != self.experiment_id {
                return Err(
                    MycelialObserverEffectTrialError::AssignmentExposureMismatch(
                        "response experiment identity",
                    ),
                );
            }
            if !observation_ids.insert(item.response.observation_id.as_str()) {
                return Err(MycelialObserverEffectTrialError::DuplicateId(
                    "observation_id",
                ));
            }
            let key = (
                item.endpoint_id.clone(),
                exposure.cultivation_unit_id.clone(),
            );
            let status = coverage.entry(key).or_insert((false, false, 0));
            match item.response.observation_window {
                MycelialPhotobiologicalObservationWindow::BaselineBeforeExposure => {
                    status.0 = true;
                }
                MycelialPhotobiologicalObservationWindow::ImmediatePostExposure
                | MycelialPhotobiologicalObservationWindow::RecoveryFollowUp
                | MycelialPhotobiologicalObservationWindow::LongitudinalFollowUp => {
                    status.1 = true;
                }
                MycelialPhotobiologicalObservationWindow::DuringExposure => {}
            }
            if matches!(
                &item.response.measurement,
                MycelialPhotobiologicalMeasurement::MeasurementUnavailable { .. }
            ) {
                status.2 = status.2.saturating_add(1);
                missing_count = missing_count.saturating_add(1);
            }
            observation_count = observation_count.saturating_add(1);
        }

        let mut primary_endpoint_coverage = Vec::new();
        for endpoint in self.endpoints.iter().filter(|endpoint| endpoint.primary) {
            for assignment in &self.assignments {
                let key = (
                    endpoint.endpoint_id.clone(),
                    assignment.cultivation_unit_id.clone(),
                );
                let (has_baseline, has_post_exposure, missing) =
                    coverage.get(&key).copied().unwrap_or((false, false, 0));
                primary_endpoint_coverage.push(MycelialObserverEffectEndpointCoverage {
                    endpoint_id: endpoint.endpoint_id.clone(),
                    cultivation_unit_id: assignment.cultivation_unit_id.clone(),
                    has_baseline,
                    has_post_exposure,
                    explicit_missing_observation_count: missing,
                });
                if !has_baseline || !has_post_exposure {
                    return Err(MycelialObserverEffectTrialError::EndpointCoverageIncomplete);
                }
            }
        }
        primary_endpoint_coverage.sort_by(|left, right| {
            (&left.endpoint_id, &left.cultivation_unit_id)
                .cmp(&(&right.endpoint_id, &right.cultivation_unit_id))
        });

        let trial_digest = mycelial_observer_effect_trial_digest(self);
        let assignment_inventory_digest = assignment_inventory_digest(&self.assignments);
        let exposure_inventory_digest = exposure_inventory_digest(exposures)?;
        let observation_inventory_digest = observation_inventory_digest(observations)?;

        let mut report = MycelialObserverEffectTrialReport {
            schema_version: MYCELIAL_OBSERVER_EFFECT_REPORT_SCHEMA_VERSION,
            trial_digest,
            assignment_inventory_digest,
            exposure_inventory_digest,
            observation_inventory_digest,
            assigned_cultivation_unit_count: self.assignments.len() as u32,
            interference_cluster_count: self
                .assignments
                .iter()
                .map(|assignment| assignment.interference_cluster_id.as_str())
                .collect::<BTreeSet<_>>()
                .len() as u32,
            exposure_record_count: exposures.len() as u32,
            endpoint_observation_count: observation_count,
            explicit_missing_observation_count: missing_count,
            primary_endpoint_coverage,
            causal_effect_estimated: false,
            canonical_digest: String::new(),
        };
        report.canonical_digest = mycelial_observer_effect_trial_report_digest(&report);
        Ok(report)
    }
}

impl MycelialObserverEffectTrialReport {
    /// Recomputes the trial receipt from the exact preregistration and source evidence.
    pub fn validate_against_evidence(
        &self,
        trial: &MycelialObserverEffectTrial,
        exposures: &[MycelialPhotobiologicalExposureRecord],
        observations: &[MycelialObserverEffectEndpointObservation],
    ) -> Result<(), MycelialObserverEffectTrialError> {
        let expected = trial.validate_against_evidence(exposures, observations)?;
        if self != &expected {
            return Err(MycelialObserverEffectTrialError::ReportMismatch);
        }
        Ok(())
    }
}

fn exposure_purpose_matches_arm(
    arm_kind: MycelialObserverEffectArmKind,
    purpose: MycelialIlluminationPurpose,
    schedule: MycelialIlluminationScheduleKind,
    ambient_state: MycelialAmbientLightState,
) -> bool {
    match arm_kind {
        MycelialObserverEffectArmKind::DarkControl => {
            purpose == MycelialIlluminationPurpose::DarkAmbientControl
                && ambient_state == MycelialAmbientLightState::VerifiedDark
        }
        MycelialObserverEffectArmKind::AmbientControl => {
            purpose == MycelialIlluminationPurpose::DarkAmbientControl
                && ambient_state == MycelialAmbientLightState::AmbientMeasured
        }
        MycelialObserverEffectArmKind::HandlingSham => {
            purpose == MycelialIlluminationPurpose::DarkAmbientControl
                && ambient_state != MycelialAmbientLightState::Unknown
        }
        MycelialObserverEffectArmKind::LowExposureImaging => {
            purpose == MycelialIlluminationPurpose::ImagingIllumination
        }
        MycelialObserverEffectArmKind::ContinuousImaging => {
            purpose == MycelialIlluminationPurpose::ImagingIllumination
                && schedule == MycelialIlluminationScheduleKind::Continuous
        }
        MycelialObserverEffectArmKind::FractionatedImaging => {
            purpose == MycelialIlluminationPurpose::ImagingIllumination
                && matches!(
                    schedule,
                    MycelialIlluminationScheduleKind::Pulsed
                        | MycelialIlluminationScheduleKind::Fractionated
                )
        }
        MycelialObserverEffectArmKind::DeliberatePhotoperiod => {
            purpose == MycelialIlluminationPurpose::DeliberatePhotoperiod
        }
        MycelialObserverEffectArmKind::Photostimulation => {
            purpose == MycelialIlluminationPurpose::Photostimulation
        }
        MycelialObserverEffectArmKind::Photoinactivation => {
            purpose == MycelialIlluminationPurpose::Photoinactivation
        }
        MycelialObserverEffectArmKind::Other => true,
    }
}

pub fn mycelial_observer_effect_trial_digest(trial: &MycelialObserverEffectTrial) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-mycelial-observer-effect-trial-v1");
    hasher.update(&trial.schema_version.to_le_bytes());
    for value in [
        trial.trial_id.as_str(),
        trial.experiment_id.as_str(),
        trial.protocol_digest.as_str(),
        trial.power_analysis_digest.as_str(),
        trial.timebase_id.as_str(),
        trial.randomization_algorithm_id.as_str(),
        trial.randomization_algorithm_version.as_str(),
        trial.allocation_receipt_digest.as_str(),
    ] {
        hash_string(&mut hasher, value);
    }
    hasher.update(&[randomization_strategy_tag(trial.randomization_strategy)]);
    hasher.update(&trial.allocation_committed_at_ns.to_le_bytes());
    hasher.update(&[u8::from(trial.assessor_blinded)]);
    match trial
        .unblinding_or_unblinded_assessment_receipt_digest
        .as_deref()
    {
        Some(value) => {
            hasher.update(&[1]);
            hash_string(&mut hasher, value);
        }
        None => {
            hasher.update(&[0]);
        }
    }
    hasher.update(&trial.minimum_independent_clusters_per_arm.to_le_bytes());
    hasher.update(&trial.planned_independent_clusters_per_arm.to_le_bytes());

    let mut arms: Vec<_> = trial.arms.iter().collect();
    arms.sort_by(|left, right| left.arm_id.cmp(&right.arm_id));
    hasher.update(&(arms.len() as u64).to_le_bytes());
    for arm in arms {
        hash_string(&mut hasher, &arm.arm_id);
        hasher.update(&[
            arm_kind_tag(arm.arm_kind),
            u8::from(arm.is_reference_comparator),
        ]);
        hash_string(&mut hasher, &arm.description);
    }

    let mut endpoints: Vec<_> = trial.endpoints.iter().collect();
    endpoints.sort_by(|left, right| left.endpoint_id.cmp(&right.endpoint_id));
    hasher.update(&(endpoints.len() as u64).to_le_bytes());
    for endpoint in endpoints {
        hash_string(&mut hasher, &endpoint.endpoint_id);
        hasher.update(&[
            endpoint_kind_tag(endpoint.endpoint_kind),
            u8::from(endpoint.primary),
        ]);
        hash_string(&mut hasher, &endpoint.analysis_unit_id);
        hash_f64(&mut hasher, endpoint.smallest_effect_size_of_interest);
        hash_f64(&mut hasher, endpoint.equivalence_margin);
        match endpoint.analyte_id.as_deref() {
            Some(value) => {
                hasher.update(&[1]);
                hash_string(&mut hasher, value);
            }
            None => {
                hasher.update(&[0]);
            }
        }
    }

    let mut estimands: Vec<_> = trial.estimands.iter().collect();
    estimands.sort_by(|left, right| {
        (
            &left.endpoint_id,
            &left.treatment_arm_id,
            &left.comparator_arm_id,
        )
            .cmp(&(
                &right.endpoint_id,
                &right.treatment_arm_id,
                &right.comparator_arm_id,
            ))
    });
    hasher.update(&(estimands.len() as u64).to_le_bytes());
    for estimand in estimands {
        hash_string(&mut hasher, &estimand.endpoint_id);
        hash_string(&mut hasher, &estimand.treatment_arm_id);
        hash_string(&mut hasher, &estimand.comparator_arm_id);
        hasher.update(&[estimand_kind_tag(estimand.estimand_kind)]);
    }

    hash_assignment_inventory_into(&mut hasher, &trial.assignments);
    finish_digest(hasher)
}

/// Canonical commitment to one immutable treatment allocation.
pub fn mycelial_observer_effect_assignment_digest(
    assignment: &MycelialObserverEffectAssignment,
) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-mycelial-observer-effect-assignment-v1");
    for value in [
        assignment.assignment_id.as_str(),
        assignment.sample_id.as_str(),
        assignment.biological_replicate_id.as_str(),
        assignment.cultivation_unit_id.as_str(),
        assignment.interference_cluster_id.as_str(),
        assignment.randomization_block_id.as_str(),
        assignment.treatment_arm_id.as_str(),
        assignment.assignment_receipt_digest.as_str(),
    ] {
        hash_string(&mut hasher, value);
    }
    hasher.update(&assignment.assigned_at_ns.to_le_bytes());
    match assignment.stratum_evidence_digest.as_deref() {
        Some(value) => {
            hasher.update(&[1]);
            hash_string(&mut hasher, value);
        }
        None => {
            hasher.update(&[0]);
        }
    }
    finish_digest(hasher)
}

fn hash_assignment_inventory_into(
    hasher: &mut blake3::Hasher,
    assignments: &[MycelialObserverEffectAssignment],
) {
    let mut ordered: Vec<_> = assignments.iter().collect();
    ordered.sort_by(|left, right| left.assignment_id.cmp(&right.assignment_id));
    hasher.update(&(ordered.len() as u64).to_le_bytes());
    for assignment in ordered {
        for value in [
            assignment.assignment_id.as_str(),
            assignment.sample_id.as_str(),
            assignment.biological_replicate_id.as_str(),
            assignment.cultivation_unit_id.as_str(),
            assignment.interference_cluster_id.as_str(),
            assignment.randomization_block_id.as_str(),
            assignment.treatment_arm_id.as_str(),
            assignment.assignment_digest.as_str(),
            assignment.assignment_receipt_digest.as_str(),
        ] {
            hash_string(hasher, value);
        }
        hasher.update(&assignment.assigned_at_ns.to_le_bytes());
        match assignment.stratum_evidence_digest.as_deref() {
            Some(value) => {
                hasher.update(&[1]);
                hash_string(hasher, value);
            }
            None => {
                hasher.update(&[0]);
            }
        }
    }
}

fn assignment_inventory_digest(assignments: &[MycelialObserverEffectAssignment]) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-mycelial-observer-effect-assignment-inventory-v1");
    hash_assignment_inventory_into(&mut hasher, assignments);
    finish_digest(hasher)
}

fn exposure_inventory_digest(
    exposures: &[MycelialPhotobiologicalExposureRecord],
) -> Result<String, MycelialObserverEffectTrialError> {
    let mut ordered = Vec::with_capacity(exposures.len());
    for exposure in exposures {
        let digest = exposure
            .canonical_digest_hex()
            .map_err(|_| MycelialObserverEffectTrialError::ExposureValidationFailed)?;
        ordered.push((exposure.exposure_id.as_str(), digest));
    }
    ordered.sort_by(|left, right| left.0.cmp(right.0));
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-mycelial-observer-effect-exposure-inventory-v1");
    hasher.update(&(ordered.len() as u64).to_le_bytes());
    for (exposure_id, digest) in ordered {
        hash_string(&mut hasher, exposure_id);
        hash_string(&mut hasher, &digest);
    }
    Ok(finish_digest(hasher))
}

fn observation_inventory_digest(
    observations: &[MycelialObserverEffectEndpointObservation],
) -> Result<String, MycelialObserverEffectTrialError> {
    let mut ordered = Vec::with_capacity(observations.len());
    for item in observations {
        let digest = item
            .response
            .canonical_digest_hex()
            .map_err(|_| MycelialObserverEffectTrialError::ResponseValidationFailed)?;
        ordered.push((
            item.endpoint_id.as_str(),
            item.response.observation_id.as_str(),
            digest,
        ));
    }
    ordered.sort_by(|left, right| (left.0, left.1).cmp(&(right.0, right.1)));
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-mycelial-observer-effect-observation-inventory-v1");
    hasher.update(&(ordered.len() as u64).to_le_bytes());
    for (endpoint_id, observation_id, digest) in ordered {
        hash_string(&mut hasher, endpoint_id);
        hash_string(&mut hasher, observation_id);
        hash_string(&mut hasher, &digest);
    }
    Ok(finish_digest(hasher))
}

pub fn mycelial_observer_effect_trial_report_digest(
    report: &MycelialObserverEffectTrialReport,
) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-mycelial-observer-effect-trial-report-v1");
    hasher.update(&report.schema_version.to_le_bytes());
    for value in [
        report.trial_digest.as_str(),
        report.assignment_inventory_digest.as_str(),
        report.exposure_inventory_digest.as_str(),
        report.observation_inventory_digest.as_str(),
    ] {
        hash_string(&mut hasher, value);
    }
    hasher.update(&report.assigned_cultivation_unit_count.to_le_bytes());
    hasher.update(&report.interference_cluster_count.to_le_bytes());
    hasher.update(&report.exposure_record_count.to_le_bytes());
    hasher.update(&report.endpoint_observation_count.to_le_bytes());
    hasher.update(&report.explicit_missing_observation_count.to_le_bytes());
    hasher.update(&[u8::from(report.causal_effect_estimated)]);
    hasher.update(&(report.primary_endpoint_coverage.len() as u64).to_le_bytes());
    for item in &report.primary_endpoint_coverage {
        hash_string(&mut hasher, &item.endpoint_id);
        hash_string(&mut hasher, &item.cultivation_unit_id);
        hasher.update(&[
            u8::from(item.has_baseline),
            u8::from(item.has_post_exposure),
        ]);
        hasher.update(&item.explicit_missing_observation_count.to_le_bytes());
    }
    finish_digest(hasher)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mycelial_photobiological_exposure::{
        MycelialIlluminationInterval, MycelialPhotobiologicalChannelExposure,
        MycelialPhotobiologicalStage, MycelialPhotoperiodPhase,
        mycelial_illumination_schedule_digest, mycelial_photobiological_footprint_digest,
    };

    fn digest(character: char) -> String {
        format!("{:064x}", character as u32)
    }

    fn arms() -> Vec<MycelialObserverEffectArm> {
        vec![
            MycelialObserverEffectArm {
                arm_id: "dark".to_string(),
                arm_kind: MycelialObserverEffectArmKind::DarkControl,
                is_reference_comparator: true,
                description: "Ambient sensor verified dark within preregistered threshold"
                    .to_string(),
            },
            MycelialObserverEffectArm {
                arm_id: "low-imaging".to_string(),
                arm_kind: MycelialObserverEffectArmKind::LowExposureImaging,
                is_reference_comparator: false,
                description: "Qualified low-exposure imaging condition".to_string(),
            },
        ]
    }

    fn trial() -> MycelialObserverEffectTrial {
        let mut assignments = Vec::new();
        for arm_id in ["dark", "dark", "low-imaging", "low-imaging"] {
            let index = assignments.len();
            let mut assignment = MycelialObserverEffectAssignment {
                assignment_id: format!("assignment-{index}"),
                sample_id: format!("sample-{index}"),
                biological_replicate_id: format!("replicate-{index}"),
                cultivation_unit_id: format!("unit-{index}"),
                interference_cluster_id: format!("cluster-{index}"),
                randomization_block_id: format!("block-{}", index / 2),
                stratum_evidence_digest: None,
                treatment_arm_id: arm_id.to_string(),
                assigned_at_ns: 100 + index as u64,
                assignment_digest: String::new(),
                assignment_receipt_digest: digest('a'),
            };
            assignment.assignment_digest = mycelial_observer_effect_assignment_digest(&assignment);
            assignments.push(assignment);
        }
        MycelialObserverEffectTrial {
            schema_version: MYCELIAL_OBSERVER_EFFECT_TRIAL_SCHEMA_VERSION,
            trial_id: "observer-trial-001".to_string(),
            experiment_id: "fungal-light-study-001".to_string(),
            protocol_digest: digest('b'),
            power_analysis_digest: digest('c'),
            timebase_id: "monotonic-ns-v1".to_string(),
            randomization_algorithm_id: "blocked-permutation".to_string(),
            randomization_algorithm_version: "1.0.0".to_string(),
            randomization_strategy: MycelialObserverEffectRandomizationStrategy::Blocked,
            allocation_receipt_digest: digest('d'),
            allocation_committed_at_ns: 500,
            assessor_blinded: true,
            unblinding_or_unblinded_assessment_receipt_digest: None,
            minimum_independent_clusters_per_arm: 2,
            planned_independent_clusters_per_arm: 2,
            arms: arms(),
            endpoints: vec![MycelialObserverEffectEndpointDefinition {
                endpoint_id: "extension-rate".to_string(),
                endpoint_kind: MycelialObserverEffectEndpointKind::HyphalTipExtension,
                analysis_unit_id: "micrometers-per-hour".to_string(),
                smallest_effect_size_of_interest: 1.0,
                equivalence_margin: 0.5,
                primary: true,
                analyte_id: None,
            }],
            estimands: vec![MycelialObserverEffectEstimand {
                endpoint_id: "extension-rate".to_string(),
                treatment_arm_id: "low-imaging".to_string(),
                comparator_arm_id: "dark".to_string(),
                estimand_kind: MycelialObserverEffectEstimandKind::MeanDifference,
            }],
            assignments,
        }
    }

    fn exposure_for(
        assignment: &MycelialObserverEffectAssignment,
    ) -> MycelialPhotobiologicalExposureRecord {
        let illuminated = assignment.treatment_arm_id == "low-imaging";
        let intervals = if illuminated {
            vec![MycelialIlluminationInterval {
                channel_id: "blue".to_string(),
                start_offset_ns: 0,
                end_offset_ns: 1_000,
            }]
        } else {
            Vec::new()
        };
        let channels = if illuminated {
            vec![MycelialPhotobiologicalChannelExposure {
                channel_id: "blue".to_string(),
                minimum_wavelength_nm: 440,
                maximum_wavelength_nm: 480,
                spectral_distribution_digest: digest('e'),
                calibration_digest: digest('f'),
                irradiance_watts_per_cm2_upper: 0.001,
                fluence_joules_per_cm2_upper: 0.000_000_001,
                photon_dose_per_um2_upper: 100.0,
                continuous_illumination_duration_ns: 1_000,
                duty_cycle_parts_per_million: 1_000_000,
            }]
        } else {
            Vec::new()
        };
        let schedule_kind = if illuminated {
            MycelialIlluminationScheduleKind::Continuous
        } else {
            MycelialIlluminationScheduleKind::Unknown
        };
        let schedule_digest = mycelial_illumination_schedule_digest(1_000, &intervals).unwrap();
        let plan_digest = digest('g');
        let device_digest = digest('h');
        let cells = vec!["cell-a".to_string(), "cell-b".to_string()];
        let coordinate_frame_id = "specimen-frame-v1".to_string();
        let footprint_digest = mycelial_photobiological_footprint_digest(
            &coordinate_frame_id,
            &plan_digest,
            &device_digest,
            &cells,
        )
        .unwrap();

        MycelialPhotobiologicalExposureRecord {
            schema_version: crate::mycelial_photobiological_exposure::MYCELIAL_PHOTOBIOLOGICAL_EXPOSURE_SCHEMA_VERSION,
            exposure_id: format!("exposure-{}", assignment.cultivation_unit_id),
            experiment_id: "fungal-light-study-001".to_string(),
            treatment_arm_id: assignment.treatment_arm_id.clone(),
            sample_id: assignment.sample_id.clone(),
            species_id: "agaricus-bisporus".to_string(),
            strain_or_isolate_id: "isolate-001".to_string(),
            biological_replicate_id: assignment.biological_replicate_id.clone(),
            cultivation_unit_id: assignment.cultivation_unit_id.clone(),
            developmental_stage: MycelialPhotobiologicalStage::VegetativeColonization,
            substrate_or_medium_id: "compost-001".to_string(),
            substrate_condition_digest: digest('i'),
            water_activity_parts_per_million: Some(950_000),
            hydration_evidence_digest: Some(digest('j')),
            specimen_temperature_millikelvin: Some(298_150),
            temperature_evidence_digest: Some(digest('k')),
            illumination_purpose: if illuminated {
                MycelialIlluminationPurpose::ImagingIllumination
            } else {
                MycelialIlluminationPurpose::DarkAmbientControl
            },
            schedule_kind,
            ambient_light: crate::mycelial_photobiological_exposure::MycelialAmbientLightObservation {
                state: MycelialAmbientLightState::VerifiedDark,
                measurement_start_ns: 1_000,
                measurement_end_ns: 2_000,
                specimen_plane_irradiance_watts_per_cm2_upper: Some(0.000_000_1),
                spectral_distribution_digest: Some(digest('a')),
                calibration_digest: Some(digest('b')),
                dark_classification_threshold_watts_per_cm2: Some(0.000_001),
                classification_policy_digest: Some(digest('c')),
                unknown_reason: None,
                evidence_receipt_digest: digest('d'),
            },
            illumination_schedule_digest: schedule_digest,
            illumination_intervals: intervals,
            photoperiod_phase: MycelialPhotoperiodPhase::Unknown,
            photoperiod_cycle_id: "cycle-001".to_string(),
            elapsed_since_photoperiod_transition_ns: None,
            photoperiod_evidence_digest: digest('l'),
            acquisition_session_id: format!("session-{}", assignment.cultivation_unit_id),
            timebase_id: "monotonic-ns-v1".to_string(),
            timebase_evidence_digest: digest('m'),
            exposure_start_ns: 1_000,
            exposure_end_ns: 2_000,
            coordinate_frame_id,
            acquisition_plan_digest: plan_digest,
            device_capability_manifest_digest: device_digest,
            spatial_footprint_digest: footprint_digest,
            spatial_footprint_cell_ids: cells,
            execution_receipt_digest: digest('n'),
            output_artifact_inventory_digest: Some(digest('o')),
            channels,
        }
    }

    fn response(
        exposure: &MycelialPhotobiologicalExposureRecord,
        observation_id: &str,
        baseline: bool,
    ) -> MycelialObserverEffectEndpointObservation {
        let response = MycelialPhotobiologicalResponseObservation {
            schema_version: crate::mycelial_photobiological_exposure::MYCELIAL_PHOTOBIOLOGICAL_RESPONSE_SCHEMA_VERSION,
            observation_id: observation_id.to_string(),
            exposure_id: exposure.exposure_id.clone(),
            exposure_digest: exposure.canonical_digest_hex().unwrap(),
            experiment_id: exposure.experiment_id.clone(),
            treatment_arm_id: exposure.treatment_arm_id.clone(),
            sample_id: exposure.sample_id.clone(),
            biological_replicate_id: exposure.biological_replicate_id.clone(),
            cultivation_unit_id: exposure.cultivation_unit_id.clone(),
            timebase_id: exposure.timebase_id.clone(),
            observation_timestamp_ns: if baseline { 750 } else { 2_500 },
            observation_window: if baseline {
                MycelialPhotobiologicalObservationWindow::BaselineBeforeExposure
            } else {
                MycelialPhotobiologicalObservationWindow::RecoveryFollowUp
            },
            source_artifact_digest: digest('p'),
            measurement_method_id: "hyphal-front-method-v1".to_string(),
            measurement_config_digest: digest('q'),
            measurement_calibration_digest: digest('r'),
            measurement: MycelialPhotobiologicalMeasurement::HyphalTipExtension {
                distance_micrometers: if baseline { 10.0 } else { 12.0 },
                elapsed_ns: 3_600_000_000_000,
                measurement_method_id: "hyphal-front-method-v1".to_string(),
            },
        };
        MycelialObserverEffectEndpointObservation {
            endpoint_id: "extension-rate".to_string(),
            response,
        }
    }

    fn evidence() -> (
        MycelialObserverEffectTrial,
        Vec<MycelialPhotobiologicalExposureRecord>,
        Vec<MycelialObserverEffectEndpointObservation>,
    ) {
        let trial = trial();
        let exposures = trial
            .assignments
            .iter()
            .map(exposure_for)
            .collect::<Vec<_>>();
        let mut observations = Vec::new();
        for exposure in &exposures {
            observations.push(response(
                exposure,
                &format!("baseline-{}", exposure.cultivation_unit_id),
                true,
            ));
            observations.push(response(
                exposure,
                &format!("followup-{}", exposure.cultivation_unit_id),
                false,
            ));
        }
        (trial, exposures, observations)
    }

    #[test]
    fn randomized_observer_effect_trial_replays_with_unit_level_coverage() {
        let (trial, exposures, observations) = evidence();
        let report = trial
            .validate_against_evidence(&exposures, &observations)
            .unwrap();
        assert_eq!(report.assigned_cultivation_unit_count, 4);
        assert_eq!(report.interference_cluster_count, 4);
        assert_eq!(report.exposure_record_count, 4);
        assert_eq!(report.endpoint_observation_count, 8);
        assert_eq!(report.primary_endpoint_coverage.len(), 4);
        assert!(!report.causal_effect_estimated);
        report
            .validate_against_evidence(&trial, &exposures, &observations)
            .unwrap();
    }

    #[test]
    fn shared_illumination_cluster_cannot_be_split_across_treatment_arms() {
        let (mut trial, exposures, observations) = evidence();
        trial.assignments[0].interference_cluster_id =
            trial.assignments[2].interference_cluster_id.clone();
        trial.assignments[0].assignment_digest =
            mycelial_observer_effect_assignment_digest(&trial.assignments[0]);
        assert_eq!(
            trial.validate_against_evidence(&exposures, &observations),
            Err(MycelialObserverEffectTrialError::InterferenceClusterSplit)
        );
    }

    #[test]
    fn shared_chamber_vessels_do_not_inflate_independent_sample_size() {
        let (mut trial, _, _) = evidence();
        trial.assignments[1].interference_cluster_id =
            trial.assignments[0].interference_cluster_id.clone();
        trial.assignments[1].assignment_digest =
            mycelial_observer_effect_assignment_digest(&trial.assignments[1]);
        assert!(matches!(
            trial.validate(),
            Err(MycelialObserverEffectTrialError::InvalidTrial(_))
        ));
    }

    #[test]
    fn assignment_tampering_is_detected_by_canonical_commitment() {
        let (mut trial, _, _) = evidence();
        trial.assignments[0].treatment_arm_id = "low-imaging".to_string();
        assert!(matches!(
            trial.validate(),
            Err(MycelialObserverEffectTrialError::InvalidAssignment(_))
        ));
    }

    #[test]
    fn measured_ambient_condition_is_a_distinct_reference_arm() {
        let (mut trial, mut exposures, observations) = evidence();
        trial.arms[0].arm_kind = MycelialObserverEffectArmKind::AmbientControl;
        for exposure in exposures.iter_mut().filter(|exposure| exposure.treatment_arm_id == "dark") {
            exposure.ambient_light.state = MycelialAmbientLightState::AmbientMeasured;
            exposure.ambient_light.specimen_plane_irradiance_watts_per_cm2_upper =
                Some(0.000_01);
        }
        let report = trial
            .validate_against_evidence(&exposures, &observations)
            .unwrap();
        assert_eq!(report.assigned_cultivation_unit_count, 4);
    }

    #[test]
    fn unknown_ambient_light_cannot_qualify_as_dark_control() {
        let (trial, mut exposures, observations) = evidence();
        exposures[0].ambient_light.state = MycelialAmbientLightState::Unknown;
        exposures[0]
            .ambient_light
            .specimen_plane_irradiance_watts_per_cm2_upper = None;
        exposures[0].ambient_light.spectral_distribution_digest = None;
        exposures[0].ambient_light.calibration_digest = None;
        exposures[0]
            .ambient_light
            .dark_classification_threshold_watts_per_cm2 = None;
        exposures[0].ambient_light.classification_policy_digest = None;
        exposures[0].ambient_light.unknown_reason = Some("ambient sensor unavailable".to_string());
        assert!(matches!(
            trial.validate_against_evidence(&exposures, &observations),
            Err(MycelialObserverEffectTrialError::AssignmentExposureMismatch(_))
        ));
    }

    #[test]
    fn measured_ambient_exposure_cannot_be_assigned_to_dark_control() {
        let (trial, mut exposures, observations) = evidence();
        exposures[0].ambient_light.state = MycelialAmbientLightState::AmbientMeasured;
        exposures[0]
            .ambient_light
            .specimen_plane_irradiance_watts_per_cm2_upper = Some(0.000_01);
        assert!(matches!(
            trial.validate_against_evidence(&exposures, &observations),
            Err(MycelialObserverEffectTrialError::AssignmentExposureMismatch(_))
        ));
    }

    #[test]
    fn primary_endpoint_requires_baseline_and_post_exposure_observation() {
        let (trial, exposures, mut observations) = evidence();
        observations.pop();
        assert_eq!(
            trial.validate_against_evidence(&exposures, &observations),
            Err(MycelialObserverEffectTrialError::EndpointCoverageIncomplete)
        );
    }

    #[test]
    fn assignment_treatment_arm_must_match_exact_exposure() {
        let (trial, mut exposures, observations) = evidence();
        exposures[0].treatment_arm_id = "low-imaging".to_string();
        exposures[0].illumination_purpose = MycelialIlluminationPurpose::ImagingIllumination;
        exposures[0].schedule_kind = MycelialIlluminationScheduleKind::Continuous;
        exposures[0].channels = exposures[2].channels.clone();
        exposures[0].illumination_intervals = exposures[2].illumination_intervals.clone();
        exposures[0].illumination_schedule_digest =
            exposures[2].illumination_schedule_digest.clone();
        assert!(matches!(
            trial.validate_against_evidence(&exposures, &observations),
            Err(MycelialObserverEffectTrialError::AssignmentExposureMismatch(_))
        ));
    }

    #[test]
    fn explicit_missing_endpoint_values_are_preserved_and_counted() {
        let (trial, exposures, mut observations) = evidence();
        observations[1].response.measurement =
            MycelialPhotobiologicalMeasurement::MeasurementUnavailable {
                status: crate::mycelial_photobiological_exposure::MycelialPhotobiologicalMissingStatus::BelowDetection,
                reason: "below assay detection threshold".to_string(),
            };
        let report = trial
            .validate_against_evidence(&exposures, &observations)
            .unwrap();
        assert_eq!(report.explicit_missing_observation_count, 1);
        assert!(report.primary_endpoint_coverage[0].has_post_exposure);
        assert_eq!(
            report
                .primary_endpoint_coverage
                .iter()
                .map(|item| item.explicit_missing_observation_count)
                .sum::<u32>(),
            1
        );
    }

    #[test]
    fn unblinded_assessment_requires_a_provenance_receipt() {
        let (mut trial, _, _) = evidence();
        trial.assessor_blinded = false;
        trial.unblinding_or_unblinded_assessment_receipt_digest = None;
        assert!(matches!(
            trial.validate(),
            Err(MycelialObserverEffectTrialError::InvalidTrial(_))
        ));
    }

    #[test]
    fn endpoint_measurement_kind_must_match_preregistered_endpoint() {
        let (trial, exposures, mut observations) = evidence();
        observations[0].response.measurement =
            MycelialPhotobiologicalMeasurement::ViabilityFraction {
                fraction: 0.9,
                assay_id: "viability-assay".to_string(),
            };
        assert_eq!(
            trial.validate_against_evidence(&exposures, &observations),
            Err(MycelialObserverEffectTrialError::EndpointMeasurementMismatch)
        );
    }

    #[test]
    fn reassignment_after_allocation_commit_is_rejected() {
        let (mut trial, _, _) = evidence();
        trial.assignments[0].assigned_at_ns = trial.allocation_committed_at_ns + 1;
        assert!(matches!(
            trial.validate(),
            Err(MycelialObserverEffectTrialError::InvalidAssignment(_))
        ));
    }

    #[test]
    fn tampered_trial_receipt_is_rejected_by_independent_replay() {
        let (trial, exposures, observations) = evidence();
        let mut report = trial
            .validate_against_evidence(&exposures, &observations)
            .unwrap();
        report.assigned_cultivation_unit_count += 1;
        assert_eq!(
            report.validate_against_evidence(&trial, &exposures, &observations),
            Err(MycelialObserverEffectTrialError::ReportMismatch)
        );
    }

    #[test]
    fn metabolite_endpoint_requires_analyte_specific_identity() {
        let (mut trial, _, _) = evidence();
        trial.endpoints[0].endpoint_kind =
            MycelialObserverEffectEndpointKind::SecondaryMetaboliteConcentration;
        trial.endpoints[0].analyte_id = None;
        assert!(matches!(
            trial.validate(),
            Err(MycelialObserverEffectTrialError::InvalidEndpoint(_))
        ));
    }
}
