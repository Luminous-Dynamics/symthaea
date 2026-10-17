// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Evidence-bounded task-card and maintenance-record validation.
//!
//! This crate validates that a proposed maintenance task is bound to an exact
//! aircraft configuration, controlled source revision, stated jurisdiction,
//! task-specific eligibility evidence, prerequisites, tools and parts. It does
//! not grant maintenance privileges, replace approved maintenance data, perform
//! aircraft maintenance, or independently establish airworthiness.
//!
//! A valid record means only that the supplied evidence is internally
//! consistent according to this schema. References and credentials must be
//! verified by the responsible organization and competent authority.

#![forbid(unsafe_code)]

use std::collections::BTreeSet;

/// Current schema revision for canonical encoders maintained by callers.
pub const TASK_CARD_SCHEMA_VERSION: u16 = 1;

/// SHA-256 digest syntax helper. This does not compute or resolve a digest.
pub fn is_sha256_digest(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

/// A revision-bound source reference. A digest identifies bytes; it does not
/// prove that the source is genuine, applicable, or approved.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ControlledDocumentRef {
    pub document_id: String,
    pub revision: String,
    pub sha256: String,
    pub source_section: String,
}

/// Product-design task class. This is deliberately distinct from legal
/// authorization and must never be used alone to decide who may do a task.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskClass {
    P0OwnerObservation,
    P1OwnerWorkOnlyWherePermitted,
    P2TrainedMaintainer,
    P3ApprovedSpecialist,
}

/// Roles recorded by the organization responsible for the applicable
/// jurisdiction/certification basis. A role assertion is not self-proving.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaintenanceRole {
    OwnerPilot,
    Repairman,
    CertificatedMechanic,
    ApprovedRepairStation,
    ManufacturerApprovedSpecialist,
}

/// Workflow state for a controlled task card.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskState {
    Draft,
    SourceVerified,
    EligibilityChecked,
    ReadyToPerform,
    WorkRecorded,
    ReviewPending,
    Released,
    Rejected,
    Superseded,
}

/// Human review of the jurisdiction/certification/task-permission basis.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuthorityReview {
    Unresolved,
    Verified {
        jurisdiction: String,
        certification_basis: String,
        permission_source: ControlledDocumentRef,
        reviewer_id: String,
        reviewed_on: String,
        scope_statement: String,
    },
}

/// Required tool; calibration must be in-date when explicitly required.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolRequirement {
    pub tool_id: String,
    pub calibration_required: bool,
}

/// Required serialized/controlled part revision; substitutes are never
/// assumed equivalent by this schema.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PartRequirement {
    pub part_number: String,
    pub revision: String,
    pub source: ControlledDocumentRef,
}

/// Procedure content is referenced, not silently copied from the controlling
/// maintenance source. Each step specifies evidence fields that must be
/// recorded after the approved procedure is performed by an eligible person.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ControlledStep {
    pub step_id: String,
    pub source_document_id: String,
    pub source_section: String,
    pub required_evidence_fields: Vec<String>,
}

/// A source-bound acceptance check. A machine-readable numerical range is
/// allowed only when the range is explicitly tied to the identified controlled
/// source. SourceControlledReview means a qualified human must evaluate the
/// result against the current source; this crate cannot infer the limit.
#[derive(Debug, Clone, PartialEq)]
pub enum AcceptanceRule {
    NumericRange {
        minimum: Option<f64>,
        maximum: Option<f64>,
    },
    SourceControlledReview,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AcceptanceCheck {
    pub criterion_id: String,
    pub source_document_id: String,
    pub source_section: String,
    pub units: String,
    pub rule: AcceptanceRule,
}

/// Verified-by-responsible-organization permission metadata. A state of
/// Verified is an auditable assertion, not an external legal determination.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TaskAuthority {
    pub jurisdiction: String,
    pub certification_basis: String,
    pub review: AuthorityReview,
    pub required_role: MaintenanceRole,
    pub required_training_ids: Vec<String>,
    pub ownership_interest_required: bool,
    pub return_to_service_role: MaintenanceRole,
}

/// Exact serial-number effectivity for the current schema. Ranges can be added
/// only with a separately specified, model-aware comparison rule; lexical
/// string comparison is intentionally not used for aircraft serials.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AircraftApplicability {
    pub family_id: String,
    pub model_id: String,
    pub serial_numbers: Vec<String>,
    pub configuration_sha256: String,
}

/// One task card revision.
#[derive(Debug, Clone, PartialEq)]
pub struct MaintenanceTaskCard {
    pub schema_version: u16,
    /// Digest of the canonical, revision-bound task-card artifact.
    pub card_sha256: String,
    pub task_id: String,
    pub revision: u32,
    pub state: TaskState,
    pub applicability: AircraftApplicability,
    pub authority: TaskAuthority,
    pub controlling_manual: ControlledDocumentRef,
    pub supporting_sources: Vec<ControlledDocumentRef>,
    pub task_class: TaskClass,
    pub prerequisites: Vec<String>,
    pub tools: Vec<ToolRequirement>,
    pub parts: Vec<PartRequirement>,
    pub steps: Vec<ControlledStep>,
    pub acceptance_checks: Vec<AcceptanceCheck>,
}

/// Identity and current configuration of the actual aircraft being serviced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AircraftIdentity {
    pub family_id: String,
    pub model_id: String,
    pub serial_number: String,
    pub configuration_sha256: String,
    pub jurisdiction: String,
    pub certification_basis: String,
    pub owner_ids: Vec<String>,
}

/// An individual tool and its calibration status, if applicable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolAvailability {
    pub tool_id: String,
    pub calibration_valid_until: Option<String>,
    pub calibration_record: Option<ControlledDocumentRef>,
}

/// Exact stock identity. No implicit substitutions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PartAvailability {
    pub part_number: String,
    pub revision: String,
    pub source: ControlledDocumentRef,
}

/// Explicit information required before a task may be considered ready.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PerformerEvidence {
    pub person_id: String,
    pub role: MaintenanceRole,
    pub jurisdiction: String,
    pub certification_basis: String,
    pub training_ids: Vec<String>,
    pub aircraft_ownership_ids: Vec<String>,
}

/// Current source snapshot from the organization's controlled documentation
/// register. The register itself must be authoritative and kept current.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CurrentDocument {
    pub reference: ControlledDocumentRef,
}

/// All inputs required to validate readiness. Dates use canonical YYYY-MM-DD
/// format so ordering is stable; the caller is responsible for a trusted clock.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionContext {
    pub aircraft: AircraftIdentity,
    pub performer: PerformerEvidence,
    pub current_date: String,
    pub current_documents: Vec<CurrentDocument>,
    pub available_tools: Vec<ToolAvailability>,
    pub available_parts: Vec<PartAvailability>,
    pub completed_prerequisites: Vec<String>,
}

/// Result attached to each source-referenced task criterion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReviewOutcome {
    Accepted,
    Rejected,
    Indeterminate,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CriterionReview {
    pub criterion_id: String,
    pub outcome: ReviewOutcome,
    pub reviewer_id: String,
    pub authority_reference: ControlledDocumentRef,
}

/// Numeric observations are evidence, not independent permission to accept a
/// task; they are checked against the source-bound criterion in the card.
#[derive(Debug, Clone, PartialEq)]
pub struct NumericMeasurement {
    pub criterion_id: String,
    pub value: f64,
    pub units: String,
    pub recorded_by: String,
}

/// Evidence value can refer to an image, instrument output, signed report or
/// another controlled artifact. The record must retain the artifact digest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecordedEvidence {
    pub field_id: String,
    pub artifact_locator: String,
    pub sha256: String,
    pub recorded_by: String,
}

/// Completion record for one task-card revision and one aircraft configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct MaintenanceWorkRecord {
    pub task_id: String,
    pub task_revision: u32,
    pub task_card_sha256: String,
    pub aircraft_serial_number: String,
    pub aircraft_configuration_sha256: String,
    pub controlling_manual_sha256: String,
    pub completed_step_ids: Vec<String>,
    pub evidence: Vec<RecordedEvidence>,
    pub measurements: Vec<NumericMeasurement>,
    pub human_reviews: Vec<CriterionReview>,
    pub recorded_on: String,
}

/// Actors are typed so a software checker cannot be represented as a human
/// return-to-service signer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActorKind {
    HumanSourceReviewer,
    HumanEligibilityReviewer,
    HumanPerformer,
    HumanInspector,
    HumanReturnToServiceSigner,
    SoftwareChecker,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActorEvidence {
    pub actor_id: String,
    pub kind: ActorKind,
    pub role: Option<MaintenanceRole>,
    pub jurisdiction: String,
    pub certification_basis: String,
    pub credential_reference: Option<ControlledDocumentRef>,
}

/// State-transition request. Transitions are workflow governance, not a legal
/// determination. Credential references still require trusted verification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskTransition {
    MarkSourceVerified,
    MarkEligibilityChecked,
    MarkReadyToPerform,
    RecordWork,
    RequestReview,
    ReleaseToService,
    Reject,
    Supersede,
}

/// Fail-closed validation errors. They are intentionally explicit enough to
/// drive a task-card UI without guessing a corrective action.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TaskCardError {
    InvalidSchemaVersion,
    EmptyField { field: String },
    InvalidDigest { field: String },
    InvalidDate { field: String },
    WrongTaskState { expected: TaskState, actual: TaskState },
    WrongAircraftFamily,
    WrongAircraftModel,
    SerialNotApplicable,
    ConfigurationMismatch,
    JurisdictionMismatch,
    CertificationBasisMismatch,
    AuthorityReviewUnresolved,
    AuthorityReviewMismatch,
    MissingTraining { training_id: String },
    WrongPerformerRole,
    OwnershipRequirementNotMet,
    MissingPrerequisite { task_id: String },
    MissingTool { tool_id: String },
    ToolCalibrationInvalid { tool_id: String },
    MissingPart { part_number: String, revision: String },
    StaleOrMissingSource { document_id: String },
    DuplicateIdentifier { field: String, id: String },
    MissingProcedureStep { step_id: String },
    UnknownProcedureStep { step_id: String },
    MissingEvidence { field_id: String },
    MissingMeasurement { criterion_id: String },
    DuplicateMeasurement { criterion_id: String },
    InvalidMeasurement { criterion_id: String },
    MeasurementOutOfRange { criterion_id: String },
    MissingHumanReview { criterion_id: String },
    HumanReviewRejected { criterion_id: String },
    HumanReviewIndeterminate { criterion_id: String },
    UnauthorizedTransition,
    SoftwareCannotReleaseToService,
    MissingReleaseCredential,
    RecordIdentityMismatch { field: String },
}

/// Validate the card's static shape and source/effectivity metadata. Draft
/// cards may be authored and stored, but can never pass readiness validation.
pub fn validate_task_card_static(
    card: &MaintenanceTaskCard,
) -> Result<(), Vec<TaskCardError>> {
    let mut errors = Vec::new();
    if card.schema_version != TASK_CARD_SCHEMA_VERSION {
        errors.push(TaskCardError::InvalidSchemaVersion);
    }
    nonempty("task_id", &card.task_id, &mut errors);
    digest("card_sha256", &card.card_sha256, &mut errors);
    if card.revision == 0 {
        errors.push(TaskCardError::EmptyField {
            field: "revision".into(),
        });
    }
    nonempty("applicability.family_id", &card.applicability.family_id, &mut errors);
    nonempty("applicability.model_id", &card.applicability.model_id, &mut errors);
    if card.applicability.serial_numbers.is_empty() {
        errors.push(TaskCardError::EmptyField {
            field: "applicability.serial_numbers".into(),
        });
    }
    unique_strings(
        "applicability.serial_numbers",
        &card.applicability.serial_numbers,
        &mut errors,
    );
    digest("applicability.configuration_sha256", &card.applicability.configuration_sha256, &mut errors);
    nonempty("authority.jurisdiction", &card.authority.jurisdiction, &mut errors);
    nonempty("authority.certification_basis", &card.authority.certification_basis, &mut errors);
    nonempty("controlling_manual.document_id", &card.controlling_manual.document_id, &mut errors);
    validate_document("controlling_manual", &card.controlling_manual, &mut errors);

    if card.supporting_sources.is_empty() {
        errors.push(TaskCardError::EmptyField {
            field: "supporting_sources".into(),
        });
    }
    let mut source_ids = BTreeSet::new();
    for (index, source) in card.supporting_sources.iter().enumerate() {
        validate_document(&format!("supporting_sources[{index}]"), source, &mut errors);
        if !source_ids.insert(source.document_id.clone()) {
            errors.push(TaskCardError::DuplicateIdentifier {
                field: "supporting_sources.document_id".into(),
                id: source.document_id.clone(),
            });
        }
    }

    for training in &card.authority.required_training_ids {
        nonempty("authority.required_training_ids", training, &mut errors);
    }
    unique_strings(
        "authority.required_training_ids",
        &card.authority.required_training_ids,
        &mut errors,
    );
    for prerequisite in &card.prerequisites {
        nonempty("prerequisites", prerequisite, &mut errors);
    }
    unique_strings("prerequisites", &card.prerequisites, &mut errors);

    for (index, tool) in card.tools.iter().enumerate() {
        nonempty(&format!("tools[{index}].tool_id"), &tool.tool_id, &mut errors);
    }
    unique_strings(
        "tools.tool_id",
        &card.tools.iter().map(|tool| tool.tool_id.clone()).collect::<Vec<_>>(),
        &mut errors,
    );

    for (index, part) in card.parts.iter().enumerate() {
        nonempty(&format!("parts[{index}].part_number"), &part.part_number, &mut errors);
        nonempty(&format!("parts[{index}].revision"), &part.revision, &mut errors);
        validate_document(&format!("parts[{index}].source"), &part.source, &mut errors);
    }

    if card.steps.is_empty() {
        errors.push(TaskCardError::EmptyField {
            field: "steps".into(),
        });
    }
    let mut step_ids = BTreeSet::new();
    for (index, step) in card.steps.iter().enumerate() {
        nonempty(&format!("steps[{index}].step_id"), &step.step_id, &mut errors);
        nonempty(
            &format!("steps[{index}].source_document_id"),
            &step.source_document_id,
            &mut errors,
        );
        nonempty(&format!("steps[{index}].source_section"), &step.source_section, &mut errors);
        if !step_ids.insert(step.step_id.clone()) {
            errors.push(TaskCardError::DuplicateIdentifier {
                field: "steps.step_id".into(),
                id: step.step_id.clone(),
            });
        }
        for field in &step.required_evidence_fields {
            nonempty("steps.required_evidence_fields", field, &mut errors);
        }
        unique_strings(
            "steps.required_evidence_fields",
            &step.required_evidence_fields,
            &mut errors,
        );
        let step_source_matches = if card.controlling_manual.document_id == step.source_document_id {
            card.controlling_manual.source_section == step.source_section
        } else {
            card.supporting_sources.iter().any(|source| {
                source.document_id == step.source_document_id
                    && source.source_section == step.source_section
            })
        };
        if !step_source_matches {
            errors.push(TaskCardError::StaleOrMissingSource {
                document_id: step.source_document_id.clone(),
            });
        }
    }

    if card.acceptance_checks.is_empty() {
        errors.push(TaskCardError::EmptyField {
            field: "acceptance_checks".into(),
        });
    }
    let mut criterion_ids = BTreeSet::new();
    for (index, check) in card.acceptance_checks.iter().enumerate() {
        nonempty(&format!("acceptance_checks[{index}].criterion_id"), &check.criterion_id, &mut errors);
        nonempty(
            &format!("acceptance_checks[{index}].source_document_id"),
            &check.source_document_id,
            &mut errors,
        );
        nonempty(&format!("acceptance_checks[{index}].source_section"), &check.source_section, &mut errors);
        nonempty(&format!("acceptance_checks[{index}].units"), &check.units, &mut errors);
        if !criterion_ids.insert(check.criterion_id.clone()) {
            errors.push(TaskCardError::DuplicateIdentifier {
                field: "acceptance_checks.criterion_id".into(),
                id: check.criterion_id.clone(),
            });
        }
        let criterion_source_matches = if card.controlling_manual.document_id == check.source_document_id {
            card.controlling_manual.source_section == check.source_section
        } else {
            card.supporting_sources.iter().any(|source| {
                source.document_id == check.source_document_id
                    && source.source_section == check.source_section
            })
        };
        if !criterion_source_matches {
            errors.push(TaskCardError::StaleOrMissingSource {
                document_id: check.source_document_id.clone(),
            });
        }
        if let AcceptanceRule::NumericRange { minimum, maximum } = &check.rule {
            match (*minimum, *maximum) {
                (None, None) => errors.push(TaskCardError::EmptyField {
                    field: format!("acceptance_checks[{}].numeric_range", check.criterion_id),
                }),
                (Some(min), Some(max)) if !min.is_finite() || !max.is_finite() || min > max => {
                    errors.push(TaskCardError::InvalidMeasurement {
                        criterion_id: check.criterion_id.clone(),
                    });
                }
                (Some(min), None) if !min.is_finite() => errors.push(TaskCardError::InvalidMeasurement {
                    criterion_id: check.criterion_id.clone(),
                }),
                (None, Some(max)) if !max.is_finite() => errors.push(TaskCardError::InvalidMeasurement {
                    criterion_id: check.criterion_id.clone(),
                }),
                _ => {}
            }
        }
    }

    if !errors.is_empty() {
        Err(errors)
    } else {
        Ok(())
    }
}

/// Validate that a task is ready to begin for one exact aircraft, person, date,
/// source register, tool/part inventory and prerequisite set.
pub fn validate_task_ready(
    card: &MaintenanceTaskCard,
    context: &ExecutionContext,
) -> Result<(), Vec<TaskCardError>> {
    let mut errors = Vec::new();
    if let Err(static_errors) = validate_task_card_static(card) {
        errors.extend(static_errors);
    }
    if card.state != TaskState::ReadyToPerform {
        errors.push(TaskCardError::WrongTaskState {
            expected: TaskState::ReadyToPerform,
            actual: card.state,
        });
    }
    validate_date("context.current_date", &context.current_date, &mut errors);
    let aircraft = &context.aircraft;
    if aircraft.family_id != card.applicability.family_id {
        errors.push(TaskCardError::WrongAircraftFamily);
    }
    if aircraft.model_id != card.applicability.model_id {
        errors.push(TaskCardError::WrongAircraftModel);
    }
    if !card.applicability.serial_numbers.iter().any(|serial| serial == &aircraft.serial_number) {
        errors.push(TaskCardError::SerialNotApplicable);
    }
    if !same_digest(
        &aircraft.configuration_sha256,
        &card.applicability.configuration_sha256,
    ) {
        errors.push(TaskCardError::ConfigurationMismatch);
    }
    if aircraft.jurisdiction != card.authority.jurisdiction
        || context.performer.jurisdiction != card.authority.jurisdiction
    {
        errors.push(TaskCardError::JurisdictionMismatch);
    }
    if aircraft.certification_basis != card.authority.certification_basis
        || context.performer.certification_basis != card.authority.certification_basis
    {
        errors.push(TaskCardError::CertificationBasisMismatch);
    }
    if context.performer.role != card.authority.required_role {
        errors.push(TaskCardError::WrongPerformerRole);
    }
    if card.authority.ownership_interest_required
        && !context.performer.aircraft_ownership_ids.iter().any(|id| id == &aircraft.serial_number)
    {
        errors.push(TaskCardError::OwnershipRequirementNotMet);
    }
    for training in &card.authority.required_training_ids {
        if !context.performer.training_ids.iter().any(|id| id == training) {
            errors.push(TaskCardError::MissingTraining {
                training_id: training.clone(),
            });
        }
    }
    match &card.authority.review {
        AuthorityReview::Unresolved => errors.push(TaskCardError::AuthorityReviewUnresolved),
        AuthorityReview::Verified {
            jurisdiction,
            certification_basis,
            permission_source,
            reviewer_id,
            reviewed_on,
            scope_statement,
        } => {
            if jurisdiction != &card.authority.jurisdiction
                || certification_basis != &card.authority.certification_basis
                || reviewer_id.trim().is_empty()
                || scope_statement.trim().is_empty()
                || !valid_iso_date(reviewed_on)
            {
                errors.push(TaskCardError::AuthorityReviewMismatch);
            }
            validate_document("authority.review.permission_source", permission_source, &mut errors);
            if !document_is_current(permission_source, &context.current_documents) {
                errors.push(TaskCardError::StaleOrMissingSource {
                    document_id: permission_source.document_id.clone(),
                });
            }
        }
    }

    for source in std::iter::once(&card.controlling_manual).chain(card.supporting_sources.iter()) {
        if !document_is_current(source, &context.current_documents) {
            errors.push(TaskCardError::StaleOrMissingSource {
                document_id: source.document_id.clone(),
            });
        }
    }
    for part in &card.parts {
        if !context.available_parts.iter().any(|available| {
            available.part_number == part.part_number
                && available.revision == part.revision
                && available.source == part.source
        }) {
            errors.push(TaskCardError::MissingPart {
                part_number: part.part_number.clone(),
                revision: part.revision.clone(),
            });
        }
        if !document_is_current(&part.source, &context.current_documents) {
            errors.push(TaskCardError::StaleOrMissingSource {
                document_id: part.source.document_id.clone(),
            });
        }
    }
    for tool in &card.tools {
        match context.available_tools.iter().find(|available| available.tool_id == tool.tool_id) {
            None => errors.push(TaskCardError::MissingTool {
                tool_id: tool.tool_id.clone(),
            }),
            Some(available) => {
                if tool.calibration_required {
                    match (
                        available.calibration_valid_until.as_deref(),
                        available.calibration_record.as_ref(),
                    ) {
                        (Some(valid_until), Some(record))
                            if valid_iso_date(valid_until)
                                && valid_until.as_str() >= context.current_date.as_str()
                                && document_is_current(record, &context.current_documents) => {}
                        _ => errors.push(TaskCardError::ToolCalibrationInvalid {
                            tool_id: tool.tool_id.clone(),
                        }),
                    }
                }
            }
        }
    }
    for prerequisite in &card.prerequisites {
        if !context.completed_prerequisites.iter().any(|done| done == prerequisite) {
            errors.push(TaskCardError::MissingPrerequisite {
                task_id: prerequisite.clone(),
            });
        }
    }

    if !errors.is_empty() {
        Err(errors)
    } else {
        Ok(())
    }
}

/// Validate a completed work record against the exact card revision and actual
/// aircraft configuration. It checks evidence coverage and numeric criteria
/// whose source-bound rule is encoded in the card; qualitative checks remain
/// pending until their required human review is recorded.
pub fn validate_work_record(
    card: &MaintenanceTaskCard,
    aircraft: &AircraftIdentity,
    record: &MaintenanceWorkRecord,
) -> Result<(), Vec<TaskCardError>> {
    let mut errors = Vec::new();
    if !matches!(card.state, TaskState::WorkRecorded | TaskState::ReviewPending) {
        errors.push(TaskCardError::WrongTaskState {
            expected: TaskState::WorkRecorded,
            actual: card.state,
        });
    }
    if record.task_id != card.task_id {
        errors.push(TaskCardError::RecordIdentityMismatch { field: "task_id".into() });
    }
    if record.task_revision != card.revision {
        errors.push(TaskCardError::RecordIdentityMismatch { field: "task_revision".into() });
    }
    if !same_digest(&record.task_card_sha256, &card.card_sha256) {
        errors.push(TaskCardError::RecordIdentityMismatch { field: "task_card_sha256".into() });
    }
    if record.aircraft_serial_number != aircraft.serial_number
        || !card.applicability.serial_numbers.iter().any(|serial| serial == &record.aircraft_serial_number)
    {
        errors.push(TaskCardError::RecordIdentityMismatch { field: "aircraft_serial_number".into() });
    }
    if !same_digest(&record.aircraft_configuration_sha256, &aircraft.configuration_sha256)
        || !same_digest(
            &record.aircraft_configuration_sha256,
            &card.applicability.configuration_sha256,
        )
    {
        errors.push(TaskCardError::RecordIdentityMismatch { field: "aircraft_configuration_sha256".into() });
    }
    if !same_digest(&record.controlling_manual_sha256, &card.controlling_manual.sha256) {
        errors.push(TaskCardError::RecordIdentityMismatch { field: "controlling_manual_sha256".into() });
    }
    validate_date("record.recorded_on", &record.recorded_on, &mut errors);

    let defined_step_ids: BTreeSet<&str> = card.steps.iter().map(|step| step.step_id.as_str()).collect();
    let mut completed = BTreeSet::new();
    for step_id in &record.completed_step_ids {
        if !defined_step_ids.contains(step_id.as_str()) {
            errors.push(TaskCardError::UnknownProcedureStep { step_id: step_id.clone() });
        }
        if !completed.insert(step_id.as_str()) {
            errors.push(TaskCardError::DuplicateIdentifier {
                field: "record.completed_step_ids".into(),
                id: step_id.clone(),
            });
        }
    }
    for step in &card.steps {
        if !completed.contains(step.step_id.as_str()) {
            errors.push(TaskCardError::MissingProcedureStep { step_id: step.step_id.clone() });
        }
    }

    let mut evidence_fields = BTreeSet::new();
    for item in &record.evidence {
        nonempty("record.evidence.field_id", &item.field_id, &mut errors);
        nonempty("record.evidence.artifact_locator", &item.artifact_locator, &mut errors);
        nonempty("record.evidence.recorded_by", &item.recorded_by, &mut errors);
        digest("record.evidence.sha256", &item.sha256, &mut errors);
        if !evidence_fields.insert(item.field_id.as_str()) {
            errors.push(TaskCardError::DuplicateIdentifier {
                field: "record.evidence.field_id".into(),
                id: item.field_id.clone(),
            });
        }
    }
    for step in &card.steps {
        for required in &step.required_evidence_fields {
            if !evidence_fields.contains(required.as_str()) {
                errors.push(TaskCardError::MissingEvidence { field_id: required.clone() });
            }
        }
    }

    let mut measurements = BTreeSet::new();
    for measurement in &record.measurements {
        if !measurements.insert(measurement.criterion_id.as_str()) {
            errors.push(TaskCardError::DuplicateMeasurement {
                criterion_id: measurement.criterion_id.clone(),
            });
        }
        if !measurement.value.is_finite()
            || measurement.units.trim().is_empty()
            || measurement.recorded_by.trim().is_empty()
        {
            errors.push(TaskCardError::InvalidMeasurement {
                criterion_id: measurement.criterion_id.clone(),
            });
        }
    }

    let mut reviews_by_id = BTreeSet::new();
    for review in &record.human_reviews {
        if review.reviewer_id.trim().is_empty()
            || !valid_document_shape(&review.authority_reference)
        {
            errors.push(TaskCardError::InvalidMeasurement {
                criterion_id: review.criterion_id.clone(),
            });
        }
        if !reviews_by_id.insert(review.criterion_id.as_str()) {
            errors.push(TaskCardError::DuplicateIdentifier {
                field: "record.human_reviews.criterion_id".into(),
                id: review.criterion_id.clone(),
            });
        }
    }

    for check in &card.acceptance_checks {
        match &check.rule {
            AcceptanceRule::NumericRange { minimum, maximum } => {
                let Some(measurement) = record.measurements.iter().find(|m| m.criterion_id == check.criterion_id) else {
                    errors.push(TaskCardError::MissingMeasurement { criterion_id: check.criterion_id.clone() });
                    continue;
                };
                if measurement.units != check.units {
                    errors.push(TaskCardError::InvalidMeasurement { criterion_id: check.criterion_id.clone() });
                    continue;
                }
                if (*minimum).is_some_and(|min| measurement.value < min)
                    || (*maximum).is_some_and(|max| measurement.value > max)
                {
                    errors.push(TaskCardError::MeasurementOutOfRange {
                        criterion_id: check.criterion_id.clone(),
                    });
                }
            }
            AcceptanceRule::SourceControlledReview => {
                match record.human_reviews.iter().find(|review| review.criterion_id == check.criterion_id) {
                    None => errors.push(TaskCardError::MissingHumanReview { criterion_id: check.criterion_id.clone() }),
                    Some(review) if review.outcome == ReviewOutcome::Rejected => {
                        errors.push(TaskCardError::HumanReviewRejected { criterion_id: check.criterion_id.clone() });
                    }
                    Some(review) if review.outcome == ReviewOutcome::Indeterminate => {
                        errors.push(TaskCardError::HumanReviewIndeterminate { criterion_id: check.criterion_id.clone() });
                    }
                    Some(_) => {}
                }
            }
        }
    }
    if !errors.is_empty() {
        Err(errors)
    } else {
        Ok(())
    }
}

/// Move an eligibility-reviewed card to ReadyToPerform only after the full
/// readiness validator succeeds for this exact aircraft, source register,
/// person, tool/part inventory and prerequisite set.
pub fn prepare_task_for_execution(
    card: &MaintenanceTaskCard,
    context: &ExecutionContext,
    reviewer: &ActorEvidence,
) -> Result<TaskState, Vec<TaskCardError>> {
    let mut errors = Vec::new();
    if card.state != TaskState::EligibilityChecked {
        errors.push(TaskCardError::WrongTaskState {
            expected: TaskState::EligibilityChecked,
            actual: card.state,
        });
    }
    if reviewer.kind != ActorKind::HumanEligibilityReviewer
        || reviewer.actor_id.trim().is_empty()
        || reviewer.actor_id != context.performer.person_id
        || reviewer.jurisdiction != card.authority.jurisdiction
        || reviewer.certification_basis != card.authority.certification_basis
    {
        errors.push(TaskCardError::UnauthorizedTransition);
    }

    let mut ready_card = card.clone();
    ready_card.state = TaskState::ReadyToPerform;
    if let Err(readiness_errors) = validate_task_ready(&ready_card, context) {
        errors.extend(readiness_errors);
    }

    if errors.is_empty() {
        Ok(TaskState::ReadyToPerform)
    } else {
        Err(errors)
    }
}

/// Apply a constrained workflow transition. This function cannot mark a
/// task ReadyToPerform: use prepare_task_for_execution, which runs the complete
/// readiness validation. It also cannot release a task; use
/// release_task_to_service with a valid work record and current source register.

pub fn transition_task(
    card: &MaintenanceTaskCard,
    transition: TaskTransition,
    actor: &ActorEvidence,
) -> Result<TaskState, TaskCardError> {
    if actor.actor_id.trim().is_empty()
        || actor.jurisdiction != card.authority.jurisdiction
        || actor.certification_basis != card.authority.certification_basis
    {
        return Err(TaskCardError::UnauthorizedTransition);
    }
    if actor.kind == ActorKind::SoftwareChecker && transition == TaskTransition::ReleaseToService {
        return Err(TaskCardError::SoftwareCannotReleaseToService);
    }
    let next = match (card.state, transition) {
        (TaskState::Draft, TaskTransition::MarkSourceVerified)
            if actor.kind == ActorKind::HumanSourceReviewer => TaskState::SourceVerified,
        (TaskState::SourceVerified, TaskTransition::MarkEligibilityChecked)
            if actor.kind == ActorKind::HumanEligibilityReviewer => TaskState::EligibilityChecked,
        (TaskState::ReadyToPerform, TaskTransition::RecordWork)
            if actor.kind == ActorKind::HumanPerformer
                && actor.role == Some(card.authority.required_role) => TaskState::WorkRecorded,
        (TaskState::WorkRecorded, TaskTransition::RequestReview)
            if matches!(actor.kind, ActorKind::SoftwareChecker | ActorKind::HumanPerformer) => TaskState::ReviewPending,
        (state, TaskTransition::Reject)
            if !matches!(state, TaskState::Released | TaskState::Superseded)
                && matches!(actor.kind, ActorKind::SoftwareChecker | ActorKind::HumanSourceReviewer | ActorKind::HumanEligibilityReviewer | ActorKind::HumanInspector) => TaskState::Rejected,
        (state, TaskTransition::Supersede)
            if !matches!(state, TaskState::Released | TaskState::Superseded)
                && actor.kind == ActorKind::HumanSourceReviewer => TaskState::Superseded,
        _ => return Err(TaskCardError::UnauthorizedTransition),
    };
    Ok(next)
}

/// The only API that can produce Released. It requires an exact, valid work
/// record, review-pending state, a matching aircraft configuration, a verified
/// task authority review, and an explicitly typed human release signer with a
/// traceable credential reference. The credential itself must still be
/// authenticated by the responsible organization outside this crate.
pub fn release_task_to_service(
    card: &MaintenanceTaskCard,
    aircraft: &AircraftIdentity,
    record: &MaintenanceWorkRecord,
    current_documents: &[CurrentDocument],
    actor: &ActorEvidence,
) -> Result<TaskState, Vec<TaskCardError>> {
    if actor.kind == ActorKind::SoftwareChecker {
        return Err(vec![TaskCardError::SoftwareCannotReleaseToService]);
    }
    let mut errors = Vec::new();
    if let Err(static_errors) = validate_task_card_static(card) {
        errors.extend(static_errors);
    }
    if card.state != TaskState::ReviewPending {
        errors.push(TaskCardError::WrongTaskState {
            expected: TaskState::ReviewPending,
            actual: card.state,
        });
    }
    if actor.kind != ActorKind::HumanReturnToServiceSigner
        || actor.role != Some(card.authority.return_to_service_role)
        || actor.actor_id.trim().is_empty()
        || actor.jurisdiction != card.authority.jurisdiction
        || actor.certification_basis != card.authority.certification_basis
    {
        errors.push(TaskCardError::UnauthorizedTransition);
    }
    if !actor.credential_reference.as_ref().is_some_and(valid_document_shape) {
        errors.push(TaskCardError::MissingReleaseCredential);
    }
    match &card.authority.review {
        AuthorityReview::Unresolved => errors.push(TaskCardError::AuthorityReviewUnresolved),
        AuthorityReview::Verified {
            jurisdiction,
            certification_basis,
            permission_source,
            reviewer_id,
            reviewed_on,
            scope_statement,
        } => {
            if jurisdiction != &card.authority.jurisdiction
                || certification_basis != &card.authority.certification_basis
                || reviewer_id.trim().is_empty()
                || scope_statement.trim().is_empty()
                || !valid_iso_date(reviewed_on)
            {
                errors.push(TaskCardError::AuthorityReviewMismatch);
            }
            validate_document("authority.review.permission_source", permission_source, &mut errors);
            if !document_is_current(permission_source, current_documents) {
                errors.push(TaskCardError::StaleOrMissingSource {
                    document_id: permission_source.document_id.clone(),
                });
            }
        }
    }
    for source in std::iter::once(&card.controlling_manual)
        .chain(card.supporting_sources.iter())
        .chain(card.parts.iter().map(|part| &part.source))
    {
        if !document_is_current(source, current_documents) {
            errors.push(TaskCardError::StaleOrMissingSource {
                document_id: source.document_id.clone(),
            });
        }
    }
    if let AuthorityReview::Verified { permission_source, .. } = &card.authority.review {
        if !document_is_current(permission_source, current_documents) {
            errors.push(TaskCardError::StaleOrMissingSource {
                document_id: permission_source.document_id.clone(),
            });
        }
    }
    if let Err(record_errors) = validate_work_record(card, aircraft, record) {
        errors.extend(record_errors);
    }
    if errors.is_empty() {
        Ok(TaskState::Released)
    } else {
        Err(errors)
    }
}

impl TaskAuthority {
    fn review_is_verified(&self) -> bool {
        matches!(&self.review, AuthorityReview::Verified { .. })
    }
}

fn document_is_current(source: &ControlledDocumentRef, current: &[CurrentDocument]) -> bool {
    current.iter().any(|document| document.reference == *source)
}

fn valid_document_shape(source: &ControlledDocumentRef) -> bool {
    !source.document_id.trim().is_empty()
        && !source.revision.trim().is_empty()
        && !source.source_section.trim().is_empty()
        && is_sha256_digest(&source.sha256)
}

fn validate_document(field: &str, source: &ControlledDocumentRef, errors: &mut Vec<TaskCardError>) {
    nonempty(&format!("{field}.document_id"), &source.document_id, errors);
    nonempty(&format!("{field}.revision"), &source.revision, errors);
    nonempty(&format!("{field}.source_section"), &source.source_section, errors);
    digest(&format!("{field}.sha256"), &source.sha256, errors);
}

fn nonempty(field: &str, value: &str, errors: &mut Vec<TaskCardError>) {
    if value.trim().is_empty() || value.trim() != value {
        errors.push(TaskCardError::EmptyField { field: field.into() });
    }
}

fn digest(field: &str, value: &str, errors: &mut Vec<TaskCardError>) {
    if !is_sha256_digest(value) {
        errors.push(TaskCardError::InvalidDigest { field: field.into() });
    }
}

fn unique_strings(field: &str, values: &[String], errors: &mut Vec<TaskCardError>) {
    let mut seen = BTreeSet::new();
    for value in values {
        if !seen.insert(value.as_str()) {
            errors.push(TaskCardError::DuplicateIdentifier {
                field: field.into(),
                id: value.clone(),
            });
        }
    }
}

fn same_digest(left: &str, right: &str) -> bool {
    is_sha256_digest(left) && is_sha256_digest(right) && left.eq_ignore_ascii_case(right)
}

fn validate_date(field: &str, value: &str, errors: &mut Vec<TaskCardError>) {
    if !valid_iso_date(value) {
        errors.push(TaskCardError::InvalidDate { field: field.into() });
    }
}

fn valid_iso_date(value: &str) -> bool {
    let bytes = value.as_bytes();
    bytes.len() == 10
        && bytes[4] == b'-'
        && bytes[7] == b'-'
        && bytes.iter().enumerate().all(|(index, byte)| {
            matches!(index, 4 | 7) || byte.is_ascii_digit()
        })
        && value[5..7].parse::<u8>().is_ok_and(|month| (1..=12).contains(&month))
        && value[8..10].parse::<u8>().is_ok_and(|day| (1..=31).contains(&day))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(c: char) -> String {
        c.to_string().repeat(64)
    }

    fn doc(id: &str, revision: &str, c: char) -> ControlledDocumentRef {
        ControlledDocumentRef {
            document_id: id.into(),
            revision: revision.into(),
            sha256: digest(c),
            source_section: "synthetic-fixture-only".into(),
        }
    }

    fn card() -> MaintenanceTaskCard {
        let manual = doc("manual-main", "rev-1", 'a');
        let permission = doc("rule-1", "rev-1", 'b');
        let supporting = doc("test-data", "rev-1", 'c');
        MaintenanceTaskCard {
            schema_version: TASK_CARD_SCHEMA_VERSION,
            card_sha256: digest('8'),
            task_id: "SYNTHETIC-NO-REAL-PROCEDURE".into(),
            revision: 1,
            state: TaskState::ReadyToPerform,
            applicability: AircraftApplicability {
                family_id: "example-family".into(),
                model_id: "example-model".into(),
                serial_numbers: vec!["SERIAL-001".into()],
                configuration_sha256: digest('d'),
            },
            authority: TaskAuthority {
                jurisdiction: "TEST-ONLY".into(),
                certification_basis: "SYNTHETIC-NO-AIRCRAFT".into(),
                review: AuthorityReview::Verified {
                    jurisdiction: "TEST-ONLY".into(),
                    certification_basis: "SYNTHETIC-NO-AIRCRAFT".into(),
                    permission_source: permission.clone(),
                    reviewer_id: "reviewer-fixture".into(),
                    reviewed_on: "2026-10-17".into(),
                    scope_statement: "Synthetic schema tests only.".into(),
                },
                required_role: MaintenanceRole::CertificatedMechanic,
                required_training_ids: vec!["training-fixture".into()],
                ownership_interest_required: false,
                return_to_service_role: MaintenanceRole::CertificatedMechanic,
            },
            controlling_manual: manual.clone(),
            supporting_sources: vec![supporting.clone()],
            task_class: TaskClass::P2TrainedMaintainer,
            prerequisites: vec!["prerequisite-fixture".into()],
            tools: vec![ToolRequirement {
                tool_id: "tool-fixture".into(),
                calibration_required: true,
            }],
            parts: vec![PartRequirement {
                part_number: "part-fixture".into(),
                revision: "rev-1".into(),
                source: supporting.clone(),
            }],
            steps: vec![
                ControlledStep {
                    step_id: "step-1".into(),
                    source_document_id: manual.document_id.clone(),
                    source_section: "synthetic-section-1".into(),
                    required_evidence_fields: vec!["photo-fixture".into()],
                },
                ControlledStep {
                    step_id: "step-2".into(),
                    source_document_id: manual.document_id.clone(),
                    source_section: "synthetic-section-2".into(),
                    required_evidence_fields: vec!["test-result-fixture".into()],
                },
            ],
            acceptance_checks: vec![
                AcceptanceCheck {
                    criterion_id: "numeric-fixture".into(),
                    source_document_id: supporting.document_id.clone(),
                    source_section: "synthetic-criterion".into(),
                    units: "arbitrary-test-units".into(),
                    rule: AcceptanceRule::NumericRange {
                        minimum: Some(1.0),
                        maximum: Some(5.0),
                    },
                },
                AcceptanceCheck {
                    criterion_id: "human-review-fixture".into(),
                    source_document_id: manual.document_id.clone(),
                    source_section: "synthetic-human-check".into(),
                    units: "not-applicable: human review".into(),
                    rule: AcceptanceRule::SourceControlledReview,
                },
            ],
        }
    }

    fn context(card: &MaintenanceTaskCard) -> ExecutionContext {
        let aircraft = AircraftIdentity {
            family_id: card.applicability.family_id.clone(),
            model_id: card.applicability.model_id.clone(),
            serial_number: "SERIAL-001".into(),
            configuration_sha256: card.applicability.configuration_sha256.clone(),
            jurisdiction: card.authority.jurisdiction.clone(),
            certification_basis: card.authority.certification_basis.clone(),
            owner_ids: vec!["owner-fixture".into()],
        };
        let performer = PerformerEvidence {
            person_id: "maintainer-fixture".into(),
            role: card.authority.required_role,
            jurisdiction: card.authority.jurisdiction.clone(),
            certification_basis: card.authority.certification_basis.clone(),
            training_ids: vec!["training-fixture".into()],
            aircraft_ownership_ids: vec![],
        };
        let mut docs = vec![CurrentDocument { reference: card.controlling_manual.clone() }];
        docs.extend(card.supporting_sources.iter().cloned().map(|reference| CurrentDocument { reference }));
        if let AuthorityReview::Verified { permission_source, .. } = &card.authority.review {
            docs.push(CurrentDocument { reference: permission_source.clone() });
        }
        let tool = ToolAvailability {
            tool_id: "tool-fixture".into(),
            calibration_valid_until: Some("2027-10-17".into()),
            calibration_record: Some(doc("calibration-tool-fixture", "rev-1", 'e')),
        };
        docs.push(CurrentDocument { reference: tool.calibration_record.clone().unwrap() });
        let part = card.parts[0].clone();
        ExecutionContext {
            aircraft,
            performer,
            current_date: "2026-10-17".into(),
            current_documents: docs,
            available_tools: vec![tool],
            available_parts: vec![PartAvailability {
                part_number: part.part_number,
                revision: part.revision,
                source: part.source,
            }],
            completed_prerequisites: card.prerequisites.clone(),
        }
    }

    fn work_record(card: &MaintenanceTaskCard) -> MaintenanceWorkRecord {
        MaintenanceWorkRecord {
            task_id: card.task_id.clone(),
            task_revision: card.revision,
            task_card_sha256: card.card_sha256.clone(),
            aircraft_serial_number: "SERIAL-001".into(),
            aircraft_configuration_sha256: card.applicability.configuration_sha256.clone(),
            controlling_manual_sha256: card.controlling_manual.sha256.clone(),
            completed_step_ids: card.steps.iter().map(|step| step.step_id.clone()).collect(),
            evidence: vec![
                RecordedEvidence {
                    field_id: "photo-fixture".into(),
                    artifact_locator: "fixture://photo".into(),
                    sha256: digest('f'),
                    recorded_by: "maintainer-fixture".into(),
                },
                RecordedEvidence {
                    field_id: "test-result-fixture".into(),
                    artifact_locator: "fixture://measurement".into(),
                    sha256: digest('1'),
                    recorded_by: "maintainer-fixture".into(),
                },
            ],
            measurements: vec![NumericMeasurement {
                criterion_id: "numeric-fixture".into(),
                value: 3.0,
                units: "arbitrary-test-units".into(),
                recorded_by: "maintainer-fixture".into(),
            }],
            human_reviews: vec![CriterionReview {
                criterion_id: "human-review-fixture".into(),
                outcome: ReviewOutcome::Accepted,
                reviewer_id: "qualified-reviewer-fixture".into(),
                authority_reference: doc("review-authority", "rev-1", '2'),
            }],
            recorded_on: "2026-10-17".into(),
        }
    }

    #[test]
    fn synthetically_valid_ready_task_passes_contract_validation() {
        let card = card();
        validate_task_ready(&card, &context(&card)).unwrap();
    }

    #[test]
    fn wrong_serial_and_configuration_fail_closed() {
        let card = card();
        let mut input = context(&card);
        input.aircraft.serial_number = "SERIAL-999".into();
        input.aircraft.configuration_sha256 = digest('0');
        let result = validate_task_ready(&card, &input).unwrap_err();
        assert!(result.contains(&TaskCardError::SerialNotApplicable));
        assert!(result.contains(&TaskCardError::ConfigurationMismatch));
    }

    #[test]
    fn task_cannot_be_marked_ready_without_complete_prechecks() {
        let mut card = card();
        card.state = TaskState::EligibilityChecked;
        let mut context = context(&card);
        context.available_tools.clear();
        let reviewer = ActorEvidence {
            actor_id: context.performer.person_id.clone(),
            kind: ActorKind::HumanEligibilityReviewer,
            role: Some(MaintenanceRole::CertificatedMechanic),
            jurisdiction: card.authority.jurisdiction.clone(),
            certification_basis: card.authority.certification_basis.clone(),
            credential_reference: Some(doc("review-credential", "rev-1", '4')),
        };
        let result = prepare_task_for_execution(&card, &context, &reviewer).unwrap_err();
        assert!(result.contains(&TaskCardError::MissingTool { tool_id: "tool-fixture".into() }));
    }

    #[test]
    fn stale_manual_and_missing_training_fail_closed() {
        let card = card();
        let mut input = context(&card);
        input.current_documents.retain(|doc| doc.reference.document_id != card.controlling_manual.document_id);
        input.performer.training_ids.clear();
        let result = validate_task_ready(&card, &input).unwrap_err();
        assert!(result.contains(&TaskCardError::StaleOrMissingSource {
            document_id: card.controlling_manual.document_id.clone()
        }));
        assert!(result.contains(&TaskCardError::MissingTraining {
            training_id: "training-fixture".into()
        }));
    }

    #[test]
    fn missing_part_tool_and_prerequisite_fail_closed() {
        let card = card();
        let mut input = context(&card);
        input.available_parts.clear();
        input.available_tools.clear();
        input.completed_prerequisites.clear();
        let result = validate_task_ready(&card, &input).unwrap_err();
        assert!(result.contains(&TaskCardError::MissingPart {
            part_number: "part-fixture".into(),
            revision: "rev-1".into()
        }));
        assert!(result.contains(&TaskCardError::MissingTool {
            tool_id: "tool-fixture".into()
        }));
        assert!(result.contains(&TaskCardError::MissingPrerequisite {
            task_id: "prerequisite-fixture".into()
        }));
    }

    #[test]
    fn software_checker_cannot_release_a_task() {
        let card = card();
        let actor = ActorEvidence {
            actor_id: "checker-fixture".into(),
            kind: ActorKind::SoftwareChecker,
            role: None,
            jurisdiction: card.authority.jurisdiction.clone(),
            certification_basis: card.authority.certification_basis.clone(),
            credential_reference: None,
        };
        assert_eq!(
            transition_task(&card, TaskTransition::ReleaseToService, &actor),
            Err(TaskCardError::SoftwareCannotReleaseToService)
        );
    }

    #[test]
    fn work_record_detects_out_of_range_values_and_missing_steps() {
        let mut card = card();
        card.state = TaskState::ReviewPending;
        let aircraft = context(&card).aircraft;
        let mut record = work_record(&card);
        record.measurements[0].value = 9.0;
        record.completed_step_ids.pop();
        let result = validate_work_record(&card, &aircraft, &record).unwrap_err();
        assert!(result.contains(&TaskCardError::MeasurementOutOfRange {
            criterion_id: "numeric-fixture".into()
        }));
        assert!(result.contains(&TaskCardError::MissingProcedureStep {
            step_id: "step-2".into()
        }));
    }

    #[test]
    fn qualitative_check_requires_human_review() {
        let mut card = card();
        card.state = TaskState::ReviewPending;
        let aircraft = context(&card).aircraft;
        let mut record = work_record(&card);
        record.human_reviews.clear();
        let result = validate_work_record(&card, &aircraft, &record).unwrap_err();
        assert!(result.contains(&TaskCardError::MissingHumanReview {
            criterion_id: "human-review-fixture".into()
        }));
    }

    #[test]
    fn stale_or_invalid_digest_in_task_source_is_rejected() {
        let mut card = card();
        card.controlling_manual.sha256 = "unresolved".into();
        assert!(matches!(
            validate_task_card_static(&card),
            Err(errors) if errors.iter().any(|error| matches!(error, TaskCardError::InvalidDigest { field } if field == "controlling_manual.sha256"))
        ));
    }

    #[test]
    fn software_cannot_skip_directly_from_work_recorded_to_released() {
        let mut card = card();
        card.state = TaskState::WorkRecorded;
        let actor = ActorEvidence {
            actor_id: "checker-fixture".into(),
            kind: ActorKind::SoftwareChecker,
            role: None,
            jurisdiction: card.authority.jurisdiction.clone(),
            certification_basis: card.authority.certification_basis.clone(),
            credential_reference: None,
        };
        assert_eq!(
            transition_task(&card, TaskTransition::ReleaseToService, &actor),
            Err(TaskCardError::SoftwareCannotReleaseToService)
        );
    }

    #[test]
    fn valid_work_record_still_requires_human_release_signer() {
        let mut card = card();
        card.state = TaskState::ReviewPending;
        let context = context(&card);
        let record = work_record(&card);
        let signer = ActorEvidence {
            actor_id: "authorized-signer-fixture".into(),
            kind: ActorKind::HumanReturnToServiceSigner,
            role: Some(card.authority.return_to_service_role),
            jurisdiction: card.authority.jurisdiction.clone(),
            certification_basis: card.authority.certification_basis.clone(),
            credential_reference: Some(doc("credential-fixture", "rev-1", '3')),
        };
        assert_eq!(
            release_task_to_service(&card, &context.aircraft, &record, &context.current_documents, &signer),
            Ok(TaskState::Released)
        );
    }

    #[test]
    fn software_checker_cannot_release_even_with_a_valid_record() {
        let mut card = card();
        card.state = TaskState::ReviewPending;
        let context = context(&card);
        let record = work_record(&card);
        let checker = ActorEvidence {
            actor_id: "software-checker-fixture".into(),
            kind: ActorKind::SoftwareChecker,
            role: None,
            jurisdiction: card.authority.jurisdiction.clone(),
            certification_basis: card.authority.certification_basis.clone(),
            credential_reference: None,
        };
        assert_eq!(
            release_task_to_service(&card, &context.aircraft, &record, &context.current_documents, &checker),
            Err(vec![TaskCardError::SoftwareCannotReleaseToService])
        );
    }

    #[test]
    fn superseded_source_blocks_return_to_service() {
        let mut card = card();
        card.state = TaskState::ReviewPending;
        let context = context(&card);
        let record = work_record(&card);
        let signer = ActorEvidence {
            actor_id: "authorized-signer-fixture".into(),
            kind: ActorKind::HumanReturnToServiceSigner,
            role: Some(card.authority.return_to_service_role),
            jurisdiction: card.authority.jurisdiction.clone(),
            certification_basis: card.authority.certification_basis.clone(),
            credential_reference: Some(doc("credential-fixture", "rev-1", '3')),
        };
        let current_without_manual: Vec<CurrentDocument> = context.current_documents
            .iter()
            .filter(|source| source.reference.document_id != card.controlling_manual.document_id)
            .cloned()
            .collect();
        let result = release_task_to_service(
            &card,
            &context.aircraft,
            &record,
            &current_without_manual,
            &signer,
        ).unwrap_err();
        assert!(result.contains(&TaskCardError::StaleOrMissingSource {
            document_id: card.controlling_manual.document_id.clone()
        }));
    }

    #[test]
    fn invalid_work_record_cannot_be_released_to_service() {
        let mut card = card();
        card.state = TaskState::ReviewPending;
        let context = context(&card);
        let mut record = work_record(&card);
        record.measurements[0].value = 99.0;
        let signer = ActorEvidence {
            actor_id: "authorized-signer-fixture".into(),
            kind: ActorKind::HumanReturnToServiceSigner,
            role: Some(card.authority.return_to_service_role),
            jurisdiction: card.authority.jurisdiction.clone(),
            certification_basis: card.authority.certification_basis.clone(),
            credential_reference: Some(doc("credential-fixture", "rev-1", '3')),
        };
        let result = release_task_to_service(&card, &context.aircraft, &record, &context.current_documents, &signer).unwrap_err();
        assert!(result.contains(&TaskCardError::MeasurementOutOfRange {
            criterion_id: "numeric-fixture".into()
        }));
    }

    #[test]
    fn source_references_must_be_unique_and_steps_bound_to_sources() {
        let mut card = card();
        card.supporting_sources.push(card.supporting_sources[0].clone());
        card.steps[0].source_document_id = "unknown-manual".into();
        let result = validate_task_card_static(&card).unwrap_err();
        assert!(result.iter().any(|error| matches!(error, TaskCardError::DuplicateIdentifier { field, .. } if field == "supporting_sources.document_id")));
        assert!(result.contains(&TaskCardError::StaleOrMissingSource { document_id: "unknown-manual".into() }));
    }
}
