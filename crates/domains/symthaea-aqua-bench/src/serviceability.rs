//! Evidence-bound repair and serviceability observations for Luminous Aqua.
//!
//! This module classifies recorded service trials. It does not calculate lifecycle cost,
//! environmental impact, water/energy resource closure, hygiene, metrology, or safety.
//! Those remain with the named owner systems. Unknown access or incomplete return-to-service
//! evidence is never counted as a successful non-destructive repair.

use std::collections::HashSet;
use std::error::Error;
use std::fmt::{Display, Formatter};

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn require_text(field: &'static str, value: &str) -> Result<(), ServiceabilityError> {
    if value.trim().is_empty() {
        return Err(ServiceabilityError::MissingField { field });
    }
    Ok(())
}

fn require_digest(field: &'static str, value: &str) -> Result<(), ServiceabilityError> {
    if !is_sha256(value) {
        return Err(ServiceabilityError::InvalidDigest { field });
    }
    Ok(())
}

/// How the priority component was accessed during the exact service job.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServiceAccessOutcome {
    DrySideWithoutDestructiveFinish,
    WetSideWithoutDestructiveFinish,
    DestructiveFinishRemovalRequired,
    NotRepairable,
    Unknown,
}

impl ServiceAccessOutcome {
    pub fn is_non_destructive(self) -> bool {
        matches!(
            self,
            Self::DrySideWithoutDestructiveFinish | Self::WetSideWithoutDestructiveFinish
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TechnicianSkillClass {
    UserServiceable,
    GeneralPlumber,
    QualifiedSpecialist,
    ManufacturerSpecialist,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SparePartAvailability {
    AvailableBeforeJob,
    NotAvailableBeforeJob,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PostRepairFunction {
    Pass,
    Fail,
    NotAssessed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReturnToServiceStatus {
    QualifiedReady,
    Pending,
    Failed,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ServiceEvidenceOwner {
    IndependentDeviceAuthority,
    MycelixWaterQuality,
    MycelixMetrology,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServiceEvidenceRef {
    pub owner: ServiceEvidenceOwner,
    pub record_id: String,
    pub sha256: String,
}

impl ServiceEvidenceRef {
    fn validate(&self) -> Result<(), ServiceabilityError> {
        require_text("service_evidence.record_id", &self.record_id)?;
        require_digest("service_evidence.sha256", &self.sha256)
    }
}

/// One measured repair/maintenance trial. Resource fields are retained as observations,
/// but Aqua does not aggregate them: Symtropy owns canonical resource closure.
#[derive(Debug, Clone, PartialEq)]
pub struct ServiceJobObservation {
    pub run_id: String,
    pub replicate_id: String,
    pub technician_block_id: String,
    pub specimen_id: String,
    pub market_variant_id: String,
    pub hardware_configuration_sha256: String,
    pub fault_manifest_sha256: String,
    pub job_profile_id: String,
    pub protocol_sha256: String,
    pub routine_job: bool,
    pub access_outcome: ServiceAccessOutcome,
    /// Count of ordered disassembly actions from initial access to target component.
    /// This is defined by the frozen job instruction, not by parts removed.
    pub disassembly_depth: u32,
    /// Number of distinct components physically removed during the job.
    pub components_removed_count: u32,
    /// Number of fastener / connector separations required by the job.
    pub fasteners_removed_count: u32,
    pub tools_used: Vec<String>,
    pub spare_part_id: Option<String>,
    pub spare_part_availability: SparePartAvailability,
    pub technician_skill: TechnicianSkillClass,
    pub technician_minutes: f64,
    pub downtime_minutes: f64,
    pub water_used_litres: Option<f64>,
    pub electricity_used_kwh: Option<f64>,
    pub waste_mass_grams: Option<f64>,
    pub post_repair_function: PostRepairFunction,
    pub return_to_service: ReturnToServiceStatus,
    pub water_quality_requalification_required: bool,
    pub metrology_requalification_required: bool,
    pub evidence: Vec<ServiceEvidenceRef>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ServiceabilityError {
    MissingField { field: &'static str },
    InvalidDigest { field: &'static str },
    InvalidMeasurement { field: &'static str },
    SparePartIdentityRequired,
    DuplicateTool { tool_id: String },
    DuplicateEvidenceOwner { owner: ServiceEvidenceOwner },
    MissingDeviceAuthorityEvidence,
    MissingWaterQualityEvidence,
    MissingMetrologyEvidence,
    FunctionCheckNotPassed,
    SameRunCompared,
    MatchedJobContextMismatch { field: &'static str },
    EmptyCohort,
    DuplicateRunId { run_id: String },
}

impl Display for ServiceabilityError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingField { field } => write!(f, "missing serviceability field: {field}"),
            Self::InvalidDigest { field } => write!(f, "invalid serviceability SHA-256: {field}"),
            Self::InvalidMeasurement { field } => {
                write!(f, "invalid serviceability measurement: {field}")
            }
            Self::SparePartIdentityRequired => {
                write!(f, "known spare-part availability requires an exact part identifier")
            }
            Self::DuplicateTool { tool_id } => {
                write!(f, "duplicate tool identifier in service job: {tool_id}")
            }
            Self::DuplicateEvidenceOwner { owner } => {
                write!(f, "duplicate service evidence owner: {owner:?}")
            }
            Self::MissingDeviceAuthorityEvidence => {
                write!(f, "qualified return-to-service lacks device-authority evidence")
            }
            Self::MissingWaterQualityEvidence => {
                write!(f, "required water-quality requalification evidence is missing")
            }
            Self::MissingMetrologyEvidence => {
                write!(f, "required metrology requalification evidence is missing")
            }
            Self::FunctionCheckNotPassed => {
                write!(f, "qualified return-to-service requires a passed function check")
            }
            Self::SameRunCompared => write!(f, "a service run cannot be compared with itself"),
            Self::MatchedJobContextMismatch { field } => {
                write!(f, "matched service-job context differs at: {field}")
            }
            Self::EmptyCohort => write!(f, "serviceability cohort is empty"),
            Self::DuplicateRunId { run_id } => {
                write!(f, "duplicate serviceability run identifier: {run_id}")
            }
        }
    }
}

impl Error for ServiceabilityError {}

impl ServiceJobObservation {
    pub fn validate(&self) -> Result<(), ServiceabilityError> {
        for (field, value) in [
            ("run_id", self.run_id.as_str()),
            ("replicate_id", self.replicate_id.as_str()),
            ("technician_block_id", self.technician_block_id.as_str()),
            ("specimen_id", self.specimen_id.as_str()),
            ("market_variant_id", self.market_variant_id.as_str()),
            ("job_profile_id", self.job_profile_id.as_str()),
        ] {
            require_text(field, value)?;
        }

        for (field, digest) in [
            ("hardware_configuration_sha256", self.hardware_configuration_sha256.as_str()),
            ("fault_manifest_sha256", self.fault_manifest_sha256.as_str()),
            ("protocol_sha256", self.protocol_sha256.as_str()),
        ] {
            require_digest(field, digest)?;
        }

        if !self.technician_minutes.is_finite() || self.technician_minutes <= 0.0 {
            return Err(ServiceabilityError::InvalidMeasurement {
                field: "technician_minutes",
            });
        }
        if !self.downtime_minutes.is_finite() || self.downtime_minutes < 0.0 {
            return Err(ServiceabilityError::InvalidMeasurement {
                field: "downtime_minutes",
            });
        }
        for (field, value) in [
            ("water_used_litres", self.water_used_litres),
            ("electricity_used_kwh", self.electricity_used_kwh),
            ("waste_mass_grams", self.waste_mass_grams),
        ] {
            if value.is_some_and(|number| !number.is_finite() || number < 0.0) {
                return Err(ServiceabilityError::InvalidMeasurement { field });
            }
        }

        if let Some(part_id) = &self.spare_part_id {
            require_text("spare_part_id", part_id)?;
        }
        if matches!(
            self.spare_part_availability,
            SparePartAvailability::AvailableBeforeJob
                | SparePartAvailability::NotAvailableBeforeJob
        ) && self.spare_part_id.is_none()
        {
            return Err(ServiceabilityError::SparePartIdentityRequired);
        }

        let mut tools = HashSet::with_capacity(self.tools_used.len());
        for tool in &self.tools_used {
            require_text("tools_used", tool)?;
            if !tools.insert(tool.as_str()) {
                return Err(ServiceabilityError::DuplicateTool {
                    tool_id: tool.clone(),
                });
            }
        }

        let mut owners = HashSet::with_capacity(self.evidence.len());
        for evidence in &self.evidence {
            evidence.validate()?;
            if !owners.insert(evidence.owner) {
                return Err(ServiceabilityError::DuplicateEvidenceOwner {
                    owner: evidence.owner,
                });
            }
        }

        if self.return_to_service == ReturnToServiceStatus::QualifiedReady {
            if self.post_repair_function != PostRepairFunction::Pass {
                return Err(ServiceabilityError::FunctionCheckNotPassed);
            }
            if !owners.contains(&ServiceEvidenceOwner::IndependentDeviceAuthority) {
                return Err(ServiceabilityError::MissingDeviceAuthorityEvidence);
            }
            if self.water_quality_requalification_required
                && !owners.contains(&ServiceEvidenceOwner::MycelixWaterQuality)
            {
                return Err(ServiceabilityError::MissingWaterQualityEvidence);
            }
            if self.metrology_requalification_required
                && !owners.contains(&ServiceEvidenceOwner::MycelixMetrology)
            {
                return Err(ServiceabilityError::MissingMetrologyEvidence);
            }
        }

        Ok(())
    }

    fn has_required_return_evidence(&self) -> bool {
        let has_owner = |owner| {
            self.evidence.iter().any(|evidence| {
                evidence.owner == owner
                    && !evidence.record_id.trim().is_empty()
                    && is_sha256(&evidence.sha256)
            })
        };

        has_owner(ServiceEvidenceOwner::IndependentDeviceAuthority)
            && (!self.water_quality_requalification_required
                || has_owner(ServiceEvidenceOwner::MycelixWaterQuality))
            && (!self.metrology_requalification_required
                || has_owner(ServiceEvidenceOwner::MycelixMetrology))
    }

    pub fn counts_as_qualified_return(&self) -> bool {
        self.validate().is_ok()
            && self.return_to_service == ReturnToServiceStatus::QualifiedReady
            && self.post_repair_function == PostRepairFunction::Pass
            && self.has_required_return_evidence()
    }

    pub fn counts_as_non_destructive_repair(&self) -> bool {
        self.validate().is_ok()
            && self.routine_job
            && self.access_outcome.is_non_destructive()
            && self.counts_as_qualified_return()
    }
}

/// Descriptive cohort summary. No lifecycle-cost or resource calculator is included.
#[derive(Debug, Clone, PartialEq)]
pub struct ServiceabilityCohortSummary {
    pub observation_count: usize,
    pub routine_job_count: usize,
    pub routine_non_destructive_qualified_count: usize,
    pub routine_non_destructive_qualified_fraction: Option<f64>,
    pub routine_unknown_access_count: usize,
    pub routine_destructive_access_count: usize,
    pub routine_spare_available_before_job_count: usize,
    pub routine_spare_unavailable_before_job_count: usize,
    pub routine_spare_availability_unknown_count: usize,
    pub routine_spare_available_fraction_among_known: Option<f64>,
    pub unique_spare_part_count: usize,
    pub median_disassembly_depth: f64,
    pub median_components_removed_count: f64,
    pub median_fasteners_removed_count: f64,
    pub qualified_return_count: usize,
    pub post_repair_function_pass_count: usize,
    pub unique_tool_count: usize,
    pub median_technician_minutes: f64,
    pub median_downtime_minutes: f64,
}

/// Unknown, failed and pending jobs remain in the routine-job denominator.
pub fn summarize_serviceability_cohort(
    observations: &[ServiceJobObservation],
) -> Result<ServiceabilityCohortSummary, ServiceabilityError> {
    if observations.is_empty() {
        return Err(ServiceabilityError::EmptyCohort);
    }

    let mut run_ids = HashSet::with_capacity(observations.len());
    let mut tools = HashSet::new();
    let mut routine_job_count = 0;
    let mut routine_non_destructive_qualified_count = 0;
    let mut routine_unknown_access_count = 0;
    let mut routine_destructive_access_count = 0;
    let mut routine_spare_available_before_job_count = 0;
    let mut routine_spare_unavailable_before_job_count = 0;
    let mut routine_spare_availability_unknown_count = 0;
    let mut spare_parts = HashSet::new();
    let mut disassembly_depths = Vec::with_capacity(observations.len());
    let mut components_removed = Vec::with_capacity(observations.len());
    let mut fasteners_removed = Vec::with_capacity(observations.len());
    let mut qualified_return_count = 0;
    let mut post_repair_function_pass_count = 0;
    let mut technician_times = Vec::with_capacity(observations.len());
    let mut downtime_times = Vec::with_capacity(observations.len());

    for observation in observations {
        observation.validate()?;
        if !run_ids.insert(observation.run_id.as_str()) {
            return Err(ServiceabilityError::DuplicateRunId {
                run_id: observation.run_id.clone(),
            });
        }

        tools.extend(observation.tools_used.iter().map(String::as_str));
        if let Some(part_id) = &observation.spare_part_id {
            spare_parts.insert(part_id.as_str());
        }
        technician_times.push(observation.technician_minutes);
        downtime_times.push(observation.downtime_minutes);
        disassembly_depths.push(observation.disassembly_depth as f64);
        components_removed.push(observation.components_removed_count as f64);
        fasteners_removed.push(observation.fasteners_removed_count as f64);

        if observation.routine_job {
            routine_job_count += 1;
            if observation.counts_as_non_destructive_repair() {
                routine_non_destructive_qualified_count += 1;
            }
            match observation.access_outcome {
                ServiceAccessOutcome::Unknown => routine_unknown_access_count += 1,
                ServiceAccessOutcome::DestructiveFinishRemovalRequired
                | ServiceAccessOutcome::NotRepairable => {
                    routine_destructive_access_count += 1;
                }
                ServiceAccessOutcome::DrySideWithoutDestructiveFinish
                | ServiceAccessOutcome::WetSideWithoutDestructiveFinish => {}
            }
            match observation.spare_part_availability {
                SparePartAvailability::AvailableBeforeJob => {
                    routine_spare_available_before_job_count += 1;
                }
                SparePartAvailability::NotAvailableBeforeJob => {
                    routine_spare_unavailable_before_job_count += 1;
                }
                SparePartAvailability::Unknown => {
                    routine_spare_availability_unknown_count += 1;
                }
            }
        }

        if observation.counts_as_qualified_return() {
            qualified_return_count += 1;
        }
        if observation.post_repair_function == PostRepairFunction::Pass {
            post_repair_function_pass_count += 1;
        }
    }

    technician_times.sort_by(f64::total_cmp);
    downtime_times.sort_by(f64::total_cmp);
    disassembly_depths.sort_by(f64::total_cmp);
    components_removed.sort_by(f64::total_cmp);
    fasteners_removed.sort_by(f64::total_cmp);
    let median = |values: &[f64]| {
        let middle = values.len() / 2;
        if values.len() % 2 == 0 {
            (values[middle - 1] + values[middle]) / 2.0
        } else {
            values[middle]
        }
    };

    let routine_non_destructive_qualified_fraction = if routine_job_count == 0 {
        None
    } else {
        Some(routine_non_destructive_qualified_count as f64 / routine_job_count as f64)
    };
    let known_spare_availability_count = routine_spare_available_before_job_count
        + routine_spare_unavailable_before_job_count;
    let routine_spare_available_fraction_among_known =
        if known_spare_availability_count == 0 {
            None
        } else {
            Some(
                routine_spare_available_before_job_count as f64
                    / known_spare_availability_count as f64,
            )
        };

    Ok(ServiceabilityCohortSummary {
        observation_count: observations.len(),
        routine_job_count,
        routine_non_destructive_qualified_count,
        routine_non_destructive_qualified_fraction,
        routine_unknown_access_count,
        routine_destructive_access_count,
        routine_spare_available_before_job_count,
        routine_spare_unavailable_before_job_count,
        routine_spare_availability_unknown_count,
        routine_spare_available_fraction_among_known,
        unique_spare_part_count: spare_parts.len(),
        median_disassembly_depth: median(&disassembly_depths),
        median_components_removed_count: median(&components_removed),
        median_fasteners_removed_count: median(&fasteners_removed),
        qualified_return_count,
        post_repair_function_pass_count,
        unique_tool_count: tools.len(),
        median_technician_minutes: median(&technician_times),
        median_downtime_minutes: median(&downtime_times),
    })
}

/// Pair incumbent and candidate observations on the same repair job, replicate,
/// technician block, fault manifest and protocol. Product/specimen identity may differ.
pub fn validate_matched_service_job(
    incumbent: &ServiceJobObservation,
    candidate: &ServiceJobObservation,
) -> Result<(), ServiceabilityError> {
    incumbent.validate()?;
    candidate.validate()?;

    if incumbent.run_id == candidate.run_id {
        return Err(ServiceabilityError::SameRunCompared);
    }

    for (field, left, right) in [
        ("replicate_id", incumbent.replicate_id.as_str(), candidate.replicate_id.as_str()),
        (
            "technician_block_id",
            incumbent.technician_block_id.as_str(),
            candidate.technician_block_id.as_str(),
        ),
        (
            "fault_manifest_sha256",
            incumbent.fault_manifest_sha256.as_str(),
            candidate.fault_manifest_sha256.as_str(),
        ),
        ("job_profile_id", incumbent.job_profile_id.as_str(), candidate.job_profile_id.as_str()),
        ("protocol_sha256", incumbent.protocol_sha256.as_str(), candidate.protocol_sha256.as_str()),
    ] {
        if left != right {
            return Err(ServiceabilityError::MatchedJobContextMismatch { field });
        }
    }
    if incumbent.technician_skill != candidate.technician_skill {
        return Err(ServiceabilityError::MatchedJobContextMismatch {
            field: "technician_skill",
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(byte: char) -> String {
        std::iter::repeat_n(byte, 64).collect()
    }

    fn evidence(owner: ServiceEvidenceOwner, id: &str, byte: char) -> ServiceEvidenceRef {
        ServiceEvidenceRef {
            owner,
            record_id: id.to_owned(),
            sha256: digest(byte),
        }
    }

    fn observation(run_id: &str, access_outcome: ServiceAccessOutcome) -> ServiceJobObservation {
        ServiceJobObservation {
            run_id: run_id.to_owned(),
            replicate_id: "replicate-01".to_owned(),
            technician_block_id: "technician-block-01".to_owned(),
            specimen_id: "specimen-01".to_owned(),
            market_variant_id: "AQS1-TEST".to_owned(),
            hardware_configuration_sha256: digest('a'),
            fault_manifest_sha256: digest('b'),
            job_profile_id: "NOZZLE_BLOCKAGE".to_owned(),
            protocol_sha256: digest('c'),
            routine_job: true,
            access_outcome,
            disassembly_depth: 2,
            components_removed_count: 3,
            fasteners_removed_count: 4,
            tools_used: vec!["hex-key".to_owned(), "spanner".to_owned()],
            spare_part_id: Some("nozzle-cassette-01".to_owned()),
            spare_part_availability: SparePartAvailability::AvailableBeforeJob,
            technician_skill: TechnicianSkillClass::GeneralPlumber,
            technician_minutes: 12.0,
            downtime_minutes: 20.0,
            water_used_litres: Some(1.5),
            electricity_used_kwh: Some(0.02),
            waste_mass_grams: Some(4.0),
            post_repair_function: PostRepairFunction::Pass,
            return_to_service: ReturnToServiceStatus::QualifiedReady,
            water_quality_requalification_required: false,
            metrology_requalification_required: false,
            evidence: vec![evidence(
                ServiceEvidenceOwner::IndependentDeviceAuthority,
                "device-requal-01",
                'd',
            )],
        }
    }

    #[test]
    fn counts_non_destructive_repair_only_after_qualified_return() {
        let record = observation("run-01", ServiceAccessOutcome::DrySideWithoutDestructiveFinish);
        assert!(record.counts_as_non_destructive_repair());
        let mut pending = record;
        pending.return_to_service = ReturnToServiceStatus::Pending;
        assert!(!pending.counts_as_non_destructive_repair());
    }

    #[test]
    fn unknown_access_is_not_counted_as_non_destructive_success() {
        let record = observation("run-01", ServiceAccessOutcome::Unknown);
        assert!(!record.counts_as_non_destructive_repair());
    }

    #[test]
    fn destructive_finish_removal_is_not_a_non_destructive_success() {
        let record = observation("run-01", ServiceAccessOutcome::DestructiveFinishRemovalRequired);
        assert!(!record.counts_as_non_destructive_repair());
    }

    #[test]
    fn qualified_return_requires_device_authority_and_passed_function_check() {
        let mut record = observation(
            "run-01",
            ServiceAccessOutcome::DrySideWithoutDestructiveFinish,
        );
        record.evidence.clear();
        assert_eq!(record.validate(), Err(ServiceabilityError::MissingDeviceAuthorityEvidence));
        record.evidence.push(evidence(
            ServiceEvidenceOwner::IndependentDeviceAuthority,
            "device",
            'd',
        ));
        record.post_repair_function = PostRepairFunction::Fail;
        assert_eq!(record.validate(), Err(ServiceabilityError::FunctionCheckNotPassed));
    }

    #[test]
    fn water_quality_requalification_requires_myclx_receipt() {
        let mut record = observation(
            "run-01",
            ServiceAccessOutcome::WetSideWithoutDestructiveFinish,
        );
        record.water_quality_requalification_required = true;
        assert_eq!(record.validate(), Err(ServiceabilityError::MissingWaterQualityEvidence));
        record.evidence.push(evidence(ServiceEvidenceOwner::MycelixWaterQuality, "quality", 'e'));
        assert_eq!(record.validate(), Ok(()));
    }

    #[test]
    fn metrology_requalification_requires_mycelix_receipt() {
        let mut record = observation(
            "run-01",
            ServiceAccessOutcome::WetSideWithoutDestructiveFinish,
        );
        record.metrology_requalification_required = true;
        assert_eq!(record.validate(), Err(ServiceabilityError::MissingMetrologyEvidence));
        record.evidence.push(evidence(ServiceEvidenceOwner::MycelixMetrology, "metrology", 'f'));
        assert_eq!(record.validate(), Ok(()));
    }

    #[test]
    fn rejects_non_finite_technician_time_and_negative_resource_values() {
        let mut record = observation(
            "run-01",
            ServiceAccessOutcome::DrySideWithoutDestructiveFinish,
        );
        record.technician_minutes = f64::NAN;
        assert_eq!(
            record.validate(),
            Err(ServiceabilityError::InvalidMeasurement { field: "technician_minutes" })
        );
        record.technician_minutes = 12.0;
        record.water_used_litres = Some(-0.1);
        assert_eq!(
            record.validate(),
            Err(ServiceabilityError::InvalidMeasurement { field: "water_used_litres" })
        );
    }

    #[test]
    fn rejects_duplicate_tools_and_duplicate_evidence_owners() {
        let mut record = observation(
            "run-01",
            ServiceAccessOutcome::DrySideWithoutDestructiveFinish,
        );
        record.tools_used.push("hex-key".to_owned());
        assert_eq!(
            record.validate(),
            Err(ServiceabilityError::DuplicateTool { tool_id: "hex-key".to_owned() })
        );
        record.tools_used.pop();
        record.evidence.push(evidence(
            ServiceEvidenceOwner::IndependentDeviceAuthority,
            "device-duplicate",
            'e',
        ));
        assert_eq!(
            record.validate(),
            Err(ServiceabilityError::DuplicateEvidenceOwner {
                owner: ServiceEvidenceOwner::IndependentDeviceAuthority
            })
        );
    }

    #[test]
    fn cohort_keeps_unknown_jobs_in_the_routine_denominator() {
        let qualified = observation(
            "run-01",
            ServiceAccessOutcome::DrySideWithoutDestructiveFinish,
        );
        let unknown = observation("run-02", ServiceAccessOutcome::Unknown);
        let summary = summarize_serviceability_cohort(&[qualified, unknown]).expect("cohort");
        assert_eq!(summary.routine_job_count, 2);
        assert_eq!(summary.routine_non_destructive_qualified_count, 1);
        assert_eq!(summary.routine_unknown_access_count, 1);
        let fraction = summary
            .routine_non_destructive_qualified_fraction
            .expect("routine denominator");
        assert!((fraction - 0.5).abs() < 1e-12);
    }

    #[test]
    fn cohort_summary_reports_median_times_without_costing() {
        let first = observation("run-01", ServiceAccessOutcome::DrySideWithoutDestructiveFinish);
        let mut second = observation(
            "run-02",
            ServiceAccessOutcome::WetSideWithoutDestructiveFinish,
        );
        second.technician_minutes = 20.0;
        second.downtime_minutes = 40.0;
        let summary = summarize_serviceability_cohort(&[first, second]).expect("cohort");
        assert_eq!(summary.unique_tool_count, 2);
        assert_eq!(summary.median_technician_minutes, 16.0);
        assert_eq!(summary.median_downtime_minutes, 30.0);
    }

    #[test]
    fn matched_jobs_require_same_fault_manifest_profile_replicate_and_technician_block() {
        let incumbent = observation(
            "incumbent-run",
            ServiceAccessOutcome::DestructiveFinishRemovalRequired,
        );
        let mut candidate = observation(
            "candidate-run",
            ServiceAccessOutcome::DrySideWithoutDestructiveFinish,
        );
        candidate.market_variant_id = "INCUMBENT-SKU".to_owned();
        assert_eq!(validate_matched_service_job(&incumbent, &candidate), Ok(()));
        candidate.job_profile_id = "DRAIN_CLEANOUT".to_owned();
        assert_eq!(
            validate_matched_service_job(&incumbent, &candidate),
            Err(ServiceabilityError::MatchedJobContextMismatch { field: "job_profile_id" })
        );
    }

    #[test]
    fn routine_job_denominator_excludes_non_routine_maintenance_jobs() {
        let routine = observation("run-01", ServiceAccessOutcome::DrySideWithoutDestructiveFinish);
        let mut non_routine = observation("run-02", ServiceAccessOutcome::NotRepairable);
        non_routine.routine_job = false;
        let summary = summarize_serviceability_cohort(&[routine, non_routine]).expect("cohort");
        assert_eq!(summary.routine_job_count, 1);
        assert_eq!(summary.routine_non_destructive_qualified_count, 1);
    }

    #[test]
    fn rejects_missing_metadata_and_invalid_digests() {
        let mut record = observation(
            "run-01",
            ServiceAccessOutcome::DrySideWithoutDestructiveFinish,
        );
        record.job_profile_id.clear();
        assert_eq!(
            record.validate(),
            Err(ServiceabilityError::MissingField { field: "job_profile_id" })
        );
        record.job_profile_id = "NOZZLE_BLOCKAGE".to_owned();
        record.hardware_configuration_sha256 = "bad".to_owned();
        assert_eq!(
            record.validate(),
            Err(ServiceabilityError::InvalidDigest { field: "hardware_configuration_sha256" })
        );
    }

    #[test]
    fn cohort_reports_no_fraction_when_no_routine_jobs_exist() {
        let mut non_routine = observation("run-01", ServiceAccessOutcome::NotRepairable);
        non_routine.routine_job = false;
        let summary = summarize_serviceability_cohort(&[non_routine]).expect("cohort");
        assert_eq!(summary.routine_job_count, 0);
        assert_eq!(summary.routine_non_destructive_qualified_fraction, None);
    }

    #[test]
    fn cohort_rejects_duplicate_run_identifiers() {
        let first = observation("same-run", ServiceAccessOutcome::DrySideWithoutDestructiveFinish);
        let second = first.clone();
        assert_eq!(
            summarize_serviceability_cohort(&[first, second]),
            Err(ServiceabilityError::DuplicateRunId { run_id: "same-run".to_owned() })
        );
    }

    #[test]
    fn qualified_return_with_unknown_or_failed_status_is_not_counted() {
        let mut record = observation(
            "run-01",
            ServiceAccessOutcome::DrySideWithoutDestructiveFinish,
        );
        record.return_to_service = ReturnToServiceStatus::Unknown;
        assert!(!record.counts_as_qualified_return());
        record.return_to_service = ReturnToServiceStatus::Failed;
        assert!(!record.counts_as_qualified_return());
    }

    #[test]
    fn keeps_spare_unavailable_and_unknown_states_separate() {
        let available = observation(
            "run-01",
            ServiceAccessOutcome::DrySideWithoutDestructiveFinish,
        );
        let mut unavailable = observation(
            "run-02",
            ServiceAccessOutcome::DrySideWithoutDestructiveFinish,
        );
        unavailable.spare_part_availability = SparePartAvailability::NotAvailableBeforeJob;
        let mut unknown = observation(
            "run-03",
            ServiceAccessOutcome::Unknown,
        );
        unknown.spare_part_availability = SparePartAvailability::Unknown;
        unknown.spare_part_id = None;

        let summary =
            summarize_serviceability_cohort(&[available, unavailable, unknown]).expect("cohort");
        assert_eq!(summary.routine_spare_available_before_job_count, 1);
        assert_eq!(summary.routine_spare_unavailable_before_job_count, 1);
        assert_eq!(summary.routine_spare_availability_unknown_count, 1);
        assert_eq!(
            summary.routine_spare_available_fraction_among_known,
            Some(0.5)
        );
        assert_eq!(summary.unique_spare_part_count, 1);
    }

    #[test]
    fn zero_known_spare_availability_keeps_fraction_undefined() {
        let mut unknown = observation(
            "run-01",
            ServiceAccessOutcome::Unknown,
        );
        unknown.spare_part_availability = SparePartAvailability::Unknown;
        unknown.spare_part_id = None;
        let summary = summarize_serviceability_cohort(&[unknown]).expect("cohort");
        assert_eq!(summary.routine_spare_available_fraction_among_known, None);
    }

    #[test]
    fn reports_disassembly_depth_as_a_separate_median() {
        let mut first = observation(
            "run-01",
            ServiceAccessOutcome::DrySideWithoutDestructiveFinish,
        );
        let mut second = observation(
            "run-02",
            ServiceAccessOutcome::WetSideWithoutDestructiveFinish,
        );
        first.disassembly_depth = 1;
        second.disassembly_depth = 4;
        first.components_removed_count = 2;
        second.components_removed_count = 6;
        first.fasteners_removed_count = 1;
        second.fasteners_removed_count = 5;
        let summary = summarize_serviceability_cohort(&[first, second]).expect("cohort");
        assert_eq!(summary.median_disassembly_depth, 2.5);
        assert_eq!(summary.median_components_removed_count, 4.0);
        assert_eq!(summary.median_fasteners_removed_count, 3.0);
    }

    #[test]
    fn available_spare_requires_exact_part_identity() {
        let mut record = observation(
            "run-01",
            ServiceAccessOutcome::DrySideWithoutDestructiveFinish,
        );
        record.spare_part_id = None;
        record.spare_part_availability = SparePartAvailability::AvailableBeforeJob;
        assert_eq!(
            record.validate(),
            Err(ServiceabilityError::SparePartIdentityRequired)
        );
    }


    #[test]
    fn pending_repair_can_retain_partial_requalification_evidence_without_counting_as_returned() {
        let mut record = observation(
            "run-01",
            ServiceAccessOutcome::WetSideWithoutDestructiveFinish,
        );
        record.return_to_service = ReturnToServiceStatus::Pending;
        record.water_quality_requalification_required = true;
        record.evidence = vec![evidence(
            ServiceEvidenceOwner::MycelixWaterQuality,
            "partial-water-quality",
            'e',
        )];

        assert_eq!(record.validate(), Ok(()));
        assert!(!record.counts_as_qualified_return());
        assert!(!record.counts_as_non_destructive_repair());
    }

    #[test]
    fn public_return_predicate_rejects_missing_required_requalification_receipt() {
        let mut record = observation(
            "run-01",
            ServiceAccessOutcome::WetSideWithoutDestructiveFinish,
        );
        record.water_quality_requalification_required = true;

        assert!(!record.counts_as_qualified_return());
        assert_eq!(
            record.validate(),
            Err(ServiceabilityError::MissingWaterQualityEvidence)
        );
    }

    #[test]
    fn known_unavailable_spare_also_requires_exact_part_identity() {
        let mut record = observation(
            "run-01",
            ServiceAccessOutcome::DrySideWithoutDestructiveFinish,
        );
        record.spare_part_id = None;
        record.spare_part_availability = SparePartAvailability::NotAvailableBeforeJob;

        assert_eq!(
            record.validate(),
            Err(ServiceabilityError::SparePartIdentityRequired)
        );
    }

}
