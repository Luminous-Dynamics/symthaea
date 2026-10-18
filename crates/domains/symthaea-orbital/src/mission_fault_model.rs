//! Typed, design-derived fault-campaign obligations.
//!
//! This is a small intermediate representation, not a SysML v2 parser or a
//! complete FMEA/FMECA. A generated simulator schedule proves only that a
//! scenario was described; critical coverage remains open while any required
//! trace, observable, monitor, mitigation, invariant, or evidence link is gapful.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use crate::perturbations::{
    OrbitalPerturbation, PerturbationError, PerturbationSchedule,
};

pub const MISSION_FAULT_MODEL_SCHEMA_VERSION: &str = "0.1";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MissionFaultModel {
    pub schema_version: String,
    pub model_id: String,
    pub model_revision: String,
    pub source_design: SourceDesignRef,
    pub failure_modes: Vec<FailureMode>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SourceDesignRef {
    pub system_id: String,
    pub revision: String,
    /// SHA-256 of the exact source-design bytes, encoded as 64 hex characters.
    pub sha256: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FailureCriticality {
    Critical,
    Major,
    Minor,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FailureMode {
    pub id: String,
    pub component_id: String,
    pub interface_id: String,
    pub description: String,
    pub criticality: FailureCriticality,
    pub requirement_refs: Vec<String>,
    pub assumption_refs: Vec<String>,
    pub telemetry_label_semantics: Option<String>,
    pub injection: Option<InjectionSpec>,
    pub observable_signals: Vec<ObservableSignal>,
    pub monitor: Option<MonitorContract>,
    pub mitigation: Option<MitigationConstraint>,
    pub safety_invariant_refs: Vec<String>,
    pub evidence_refs: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum InjectionSpec {
    CommunicationBlackout {
        at_step: usize,
        duration_steps: usize,
    },
    /// Synthetic instantaneous external delta-v proxy in inertial-frame m/s.
    /// This does not model debris contact, fragmentation, or impact mechanics.
    VelocityImpulse {
        at_step: usize,
        delta_v_m_s: [f64; 3],
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObservableSignal {
    pub id: String,
    pub unit: String,
    pub expected_fault_effect: String,
    /// Canonical categorical value used only for structural signature comparison.
    /// None means unknown; unknown never counts as a distinguishing observation.
    pub signature_value: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MonitorContract {
    pub id: String,
    pub required_alarm: String,
    /// Maximum permitted simulator steps between injection and required alarm.
    pub response_deadline_steps: usize,
    /// Nominal or confounding conditions that must not trigger this monitor.
    pub false_alarm_controls: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MitigationConstraint {
    pub id: String,
    pub allowed_actions: Vec<String>,
    pub protected_invariant_refs: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CoverageGap {
    MissingRequirementTrace,
    MissingEventAssumptionRef,
    MissingTelemetryLabelSemantics,
    MissingInjection,
    MissingObservableSignal,
    MissingCanonicalSignalOutcome,
    MissingMonitorContract,
    MissingMonitorNegativeControl,
    MissingMitigationContract,
    MissingPermittedMitigation,
    MissingSafetyInvariant,
    MissingMitigationInvariantLink,
    MissingEvidenceReference,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FaultScenario {
    pub id: String,
    pub failure_mode_id: String,
    pub source_design_revision: String,
    pub source_design_sha256: String,
    pub component_id: String,
    pub interface_id: String,
    pub event_assumption_refs: Vec<String>,
    pub telemetry_label_semantics: Option<String>,
    pub injection: InjectionSpec,
    pub expected_signal_ids: Vec<String>,
    pub monitor_id: String,
    pub response_deadline_steps: usize,
    pub allowed_mitigation_actions: Vec<String>,
    pub protected_invariant_refs: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FaultObligation {
    pub failure_mode_id: String,
    pub criticality: FailureCriticality,
    /// Present means a mechanically representable scenario was generated.
    /// It does not mean the associated failure mode is fully covered.
    pub scenario: Option<FaultScenario>,
    pub gaps: Vec<CoverageGap>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FaultPairIndeterminacy {
    MissingCanonicalOutcome,
    NoComparableSignals,
    SignalSetsNotEquivalent,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "disposition", rename_all = "snake_case")]
pub enum FaultPairDisposition {
    Distinguishable { separating_signal_ids: Vec<String> },
    Indistinguishable { identical_signal_ids: Vec<String> },
    NotDemonstrated {
        reason: FaultPairIndeterminacy,
        comparable_signal_ids: Vec<String>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FaultPairFinding {
    pub left_failure_mode_id: String,
    pub right_failure_mode_id: String,
    pub disposition: FaultPairDisposition,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FaultDiagnosabilityReport {
    pub model_id: String,
    pub model_revision: String,
    pub source_design_revision: String,
    pub source_design_sha256: String,
    pub critical_failure_mode_ids: Vec<String>,
    /// One result for each unordered pair among declared critical modes.
    pub findings: Vec<FaultPairFinding>,
}

impl FaultDiagnosabilityReport {
    pub fn critical_pair_denominator(&self) -> usize {
        self.findings.len()
    }

    pub fn unresolved_or_indistinguishable_pair_count(&self) -> usize {
        self.findings
            .iter()
            .filter(|finding| {
                !matches!(finding.disposition, FaultPairDisposition::Distinguishable { .. })
            })
            .count()
    }

    /// Structural only: needs >= 2 critical modes and every pair separated by
    /// at least one common signal with two explicit, different canonical values.
    pub fn all_critical_pairs_distinguishable(&self) -> bool {
        self.critical_failure_mode_ids.len() >= 2
            && !self.findings.is_empty()
            && self.findings.iter().all(|finding| {
                matches!(finding.disposition, FaultPairDisposition::Distinguishable { .. })
            })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FaultCampaignManifest {
    pub schema_version: String,
    pub model_id: String,
    pub model_revision: String,
    pub source_design_revision: String,
    pub source_design_sha256: String,
    pub obligations: Vec<FaultObligation>,
    /// Contains representable injections even when other evidence obligations
    /// remain open; consumers must inspect obligations before claiming coverage.
    pub schedule: PerturbationSchedule,
}

impl FaultCampaignManifest {
    /// A non-empty critical denominator is mandatory; an empty campaign is not
    /// accepted by vacuous truth.
    pub fn critical_coverage_closed(&self) -> bool {
        let critical: Vec<_> = self
            .obligations
            .iter()
            .filter(|obligation| obligation.criticality == FailureCriticality::Critical)
            .collect();

        !critical.is_empty()
            && critical
                .iter()
                .all(|obligation| obligation.scenario.is_some() && obligation.gaps.is_empty())
    }

    pub fn critical_mode_count(&self) -> usize {
        self.obligations
            .iter()
            .filter(|obligation| obligation.criticality == FailureCriticality::Critical)
            .count()
    }

    pub fn critical_modes_with_gaps(&self) -> Vec<&FaultObligation> {
        self.obligations
            .iter()
            .filter(|obligation| {
                obligation.criticality == FailureCriticality::Critical
                    && (obligation.scenario.is_none() || !obligation.gaps.is_empty())
            })
            .collect()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MissionFaultModelError {
    UnsupportedSchemaVersion(String),
    EmptyField(&'static str),
    EmptyFailureModes,
    InvalidSourceSha256,
    DuplicateFailureModeId(String),
    DuplicateSignalId {
        failure_mode_id: String,
        signal_id: String,
    },
    InvalidInjectionParameters {
        failure_mode_id: String,
        reason: String,
    },
    InvalidGeneratedSchedule(PerturbationError),
}

impl std::fmt::Display for MissionFaultModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for MissionFaultModelError {}

impl MissionFaultModel {
    /// Validate identity and executable event assumptions. Missing engineering
    /// links are deliberately not rejected here; they become explicit coverage
    /// gaps in the emitted manifest.
    pub fn validate(&self) -> Result<(), MissionFaultModelError> {
        if self.schema_version != MISSION_FAULT_MODEL_SCHEMA_VERSION {
            return Err(MissionFaultModelError::UnsupportedSchemaVersion(
                self.schema_version.clone(),
            ));
        }

        require_nonblank(&self.model_id, "model_id")?;
        require_nonblank(&self.model_revision, "model_revision")?;
        require_nonblank(&self.source_design.system_id, "source_design.system_id")?;
        require_nonblank(&self.source_design.revision, "source_design.revision")?;
        if self.source_design.sha256.len() != 64
            || !self
                .source_design
                .sha256
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
        {
            return Err(MissionFaultModelError::InvalidSourceSha256);
        }
        if self.failure_modes.is_empty() {
            return Err(MissionFaultModelError::EmptyFailureModes);
        }

        let mut mode_ids = HashSet::new();
        for mode in &self.failure_modes {
            require_nonblank(&mode.id, "failure_mode.id")?;
            require_nonblank(&mode.component_id, "failure_mode.component_id")?;
            require_nonblank(&mode.interface_id, "failure_mode.interface_id")?;
            require_nonblank(&mode.description, "failure_mode.description")?;
            if let Some(monitor) = &mode.monitor {
                require_nonblank(&monitor.id, "monitor.id")?;
                require_nonblank(&monitor.required_alarm, "monitor.required_alarm")?;
            }
            if let Some(mitigation) = &mode.mitigation {
                require_nonblank(&mitigation.id, "mitigation.id")?;
            }

            if !mode_ids.insert(mode.id.as_str()) {
                return Err(MissionFaultModelError::DuplicateFailureModeId(
                    mode.id.clone(),
                ));
            }

            let mut signal_ids = HashSet::new();
            for signal in &mode.observable_signals {
                require_nonblank(&signal.id, "observable_signal.id")?;
                require_nonblank(&signal.unit, "observable_signal.unit")?;
                require_nonblank(
                    &signal.expected_fault_effect,
                    "observable_signal.expected_fault_effect",
                )?;
                if let Some(value) = &signal.signature_value {
                    require_nonblank(value, "observable_signal.signature_value")?;
                }
                if !signal_ids.insert(signal.id.as_str()) {
                    return Err(MissionFaultModelError::DuplicateSignalId {
                        failure_mode_id: mode.id.clone(),
                        signal_id: signal.id.clone(),
                    });
                }
            }

            if let Some(injection) = &mode.injection {
                match injection {
                    InjectionSpec::CommunicationBlackout {
                        at_step,
                        duration_steps,
                    } => {
                        if *duration_steps == 0 {
                            return Err(MissionFaultModelError::InvalidInjectionParameters {
                                failure_mode_id: mode.id.clone(),
                                reason: "blackout duration must be positive".to_owned(),
                            });
                        }
                        if at_step.checked_add(*duration_steps).is_none() {
                            return Err(MissionFaultModelError::InvalidInjectionParameters {
                                failure_mode_id: mode.id.clone(),
                                reason: "blackout end step overflows usize".to_owned(),
                            });
                        }
                    }
                    InjectionSpec::VelocityImpulse { delta_v_m_s, .. } => {
                        if delta_v_m_s.iter().any(|value| !value.is_finite()) {
                            return Err(MissionFaultModelError::InvalidInjectionParameters {
                                failure_mode_id: mode.id.clone(),
                                reason: "delta-v components must be finite".to_owned(),
                            });
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Compare declared critical-mode observation signatures. This is a
    /// conservative design-time lint, not a physical diagnosability proof or
    /// a minimum-sensor-set/cut-set oracle. Unknown values never distinguish.
    pub fn analyze_critical_fault_signatures(
        &self,
    ) -> Result<FaultDiagnosabilityReport, MissionFaultModelError> {
        self.validate()?;
        let critical: Vec<&FailureMode> = self
            .failure_modes
            .iter()
            .filter(|mode| mode.criticality == FailureCriticality::Critical)
            .collect();

        let mut findings = Vec::new();
        for left_index in 0..critical.len() {
            for right_index in (left_index + 1)..critical.len() {
                let left = critical[left_index];
                let right = critical[right_index];
                findings.push(FaultPairFinding {
                    left_failure_mode_id: left.id.clone(),
                    right_failure_mode_id: right.id.clone(),
                    disposition: classify_fault_pair(left, right),
                });
            }
        }

        Ok(FaultDiagnosabilityReport {
            model_id: self.model_id.clone(),
            model_revision: self.model_revision.clone(),
            source_design_revision: self.source_design.revision.clone(),
            source_design_sha256: self.source_design.sha256.clone(),
            critical_failure_mode_ids: critical.iter().map(|mode| mode.id.clone()).collect(),
            findings,
        })
    }

    /// Project the design IR into a deterministic scenario manifest and the
    /// currently supported orbital perturbation interface. Coverage gaps stay
    /// attached to each failure mode; generating an event never clears them.
    pub fn compile_campaign(
        &self,
    ) -> Result<FaultCampaignManifest, MissionFaultModelError> {
        self.validate()?;

        let mut obligations = Vec::with_capacity(self.failure_modes.len());
        let mut perturbations = Vec::new();

        for mode in &self.failure_modes {
            let gaps = coverage_gaps(mode);
            let scenario = mode.injection.as_ref().map(|injection| {
                let monitor = mode.monitor.as_ref();
                let mitigation = mode.mitigation.as_ref();

                let mut protected_invariant_refs = mode.safety_invariant_refs.clone();
                if let Some(mitigation) = mitigation {
                    for reference in &mitigation.protected_invariant_refs {
                        if !protected_invariant_refs.contains(reference) {
                            protected_invariant_refs.push(reference.clone());
                        }
                    }
                }

                FaultScenario {
                    id: format!("{}::{}", self.model_id, mode.id),
                    failure_mode_id: mode.id.clone(),
                    source_design_revision: self.source_design.revision.clone(),
                    source_design_sha256: self.source_design.sha256.clone(),
                    component_id: mode.component_id.clone(),
                    interface_id: mode.interface_id.clone(),
                    event_assumption_refs: mode.assumption_refs.clone(),
                    telemetry_label_semantics: mode.telemetry_label_semantics.clone(),
                    injection: injection.clone(),
                    expected_signal_ids: mode
                        .observable_signals
                        .iter()
                        .map(|signal| signal.id.clone())
                        .collect(),
                    monitor_id: monitor.map(|value| value.id.clone()).unwrap_or_default(),
                    response_deadline_steps: monitor
                        .map(|value| value.response_deadline_steps)
                        .unwrap_or_default(),
                    allowed_mitigation_actions: mitigation
                        .map(|value| value.allowed_actions.clone())
                        .unwrap_or_default(),
                    protected_invariant_refs,
                }
            });

            if let Some(injection) = &mode.injection {
                let event = match injection {
                    InjectionSpec::CommunicationBlackout {
                        at_step,
                        duration_steps,
                    } => OrbitalPerturbation::CommBlackout {
                        at_step: *at_step,
                        duration: *duration_steps,
                    },
                    InjectionSpec::VelocityImpulse {
                        at_step,
                        delta_v_m_s,
                    } => OrbitalPerturbation::DebrisImpact {
                        at_step: *at_step,
                        delta_v_m_s: *delta_v_m_s,
                    },
                };
                perturbations.push(event);
            }

            obligations.push(FaultObligation {
                failure_mode_id: mode.id.clone(),
                criticality: mode.criticality,
                scenario,
                gaps,
            });
        }

        let schedule = PerturbationSchedule { perturbations };
        schedule
            .validate()
            .map_err(MissionFaultModelError::InvalidGeneratedSchedule)?;

        Ok(FaultCampaignManifest {
            schema_version: MISSION_FAULT_MODEL_SCHEMA_VERSION.to_owned(),
            model_id: self.model_id.clone(),
            model_revision: self.model_revision.clone(),
            source_design_revision: self.source_design.revision.clone(),
            source_design_sha256: self.source_design.sha256.clone(),
            obligations,
            schedule,
        })
    }
}

fn require_nonblank(
    value: &str,
    field: &'static str,
) -> Result<(), MissionFaultModelError> {
    if value.trim().is_empty() {
        return Err(MissionFaultModelError::EmptyField(field));
    }
    Ok(())
}

fn has_reference(references: &[String]) -> bool {
    references.iter().any(|reference| !reference.trim().is_empty())
}

fn classify_fault_pair(left: &FailureMode, right: &FailureMode) -> FaultPairDisposition {
    let mut separating_signal_ids = Vec::new();
    let mut comparable_signal_ids = Vec::new();

    for left_signal in &left.observable_signals {
        let Some(right_signal) = right
            .observable_signals
            .iter()
            .find(|signal| signal.id == left_signal.id && signal.unit == left_signal.unit)
        else {
            continue;
        };
        comparable_signal_ids.push(left_signal.id.clone());

        if let (Some(left_value), Some(right_value)) =
            (&left_signal.signature_value, &right_signal.signature_value)
        {
            if !left_value.trim().is_empty()
                && !right_value.trim().is_empty()
                && left_value != right_value
            {
                separating_signal_ids.push(left_signal.id.clone());
            }
        }
    }

    separating_signal_ids.sort();
    separating_signal_ids.dedup();
    comparable_signal_ids.sort();
    comparable_signal_ids.dedup();

    if !separating_signal_ids.is_empty() {
        return FaultPairDisposition::Distinguishable {
            separating_signal_ids,
        };
    }

    let left_complete = !left.observable_signals.is_empty()
        && left.observable_signals.iter().all(|signal| {
            signal
                .signature_value
                .as_ref()
                .is_some_and(|value| !value.trim().is_empty())
        });
    let right_complete = !right.observable_signals.is_empty()
        && right.observable_signals.iter().all(|signal| {
            signal
                .signature_value
                .as_ref()
                .is_some_and(|value| !value.trim().is_empty())
        });
    let identical_complete_signatures = left_complete
        && right_complete
        && left.observable_signals.len() == right.observable_signals.len()
        && left.observable_signals.iter().all(|left_signal| {
            right.observable_signals.iter().any(|right_signal| {
                right_signal.id == left_signal.id
                    && right_signal.unit == left_signal.unit
                    && right_signal.signature_value == left_signal.signature_value
            })
        });

    if identical_complete_signatures {
        return FaultPairDisposition::Indistinguishable {
            identical_signal_ids: left
                .observable_signals
                .iter()
                .map(|signal| signal.id.clone())
                .collect(),
        };
    }

    let reason = if !left_complete || !right_complete {
        FaultPairIndeterminacy::MissingCanonicalOutcome
    } else if comparable_signal_ids.is_empty() {
        FaultPairIndeterminacy::NoComparableSignals
    } else {
        FaultPairIndeterminacy::SignalSetsNotEquivalent
    };

    FaultPairDisposition::NotDemonstrated {
        reason,
        comparable_signal_ids,
    }
}

fn coverage_gaps(mode: &FailureMode) -> Vec<CoverageGap> {
    let mut gaps = Vec::new();

    if !has_reference(&mode.requirement_refs) {
        gaps.push(CoverageGap::MissingRequirementTrace);
    }
    if !has_reference(&mode.assumption_refs) {
        gaps.push(CoverageGap::MissingEventAssumptionRef);
    }
    if mode
        .telemetry_label_semantics
        .as_ref()
        .is_none_or(|label| label.trim().is_empty())
    {
        gaps.push(CoverageGap::MissingTelemetryLabelSemantics);
    }
    if mode.injection.is_none() {
        gaps.push(CoverageGap::MissingInjection);
    }
    if mode.observable_signals.is_empty() {
        gaps.push(CoverageGap::MissingObservableSignal);
    } else if mode.observable_signals.iter().any(|signal| {
        signal
            .signature_value
            .as_ref()
            .is_none_or(|value| value.trim().is_empty())
    }) {
        gaps.push(CoverageGap::MissingCanonicalSignalOutcome);
    }
    match &mode.monitor {
        None => gaps.push(CoverageGap::MissingMonitorContract),
        Some(monitor) if !has_reference(&monitor.false_alarm_controls) => {
            gaps.push(CoverageGap::MissingMonitorNegativeControl);
        }
        Some(_) => {}
    }
    match &mode.mitigation {
        None => gaps.push(CoverageGap::MissingMitigationContract),
        Some(mitigation) => {
            if !has_reference(&mitigation.allowed_actions) {
                gaps.push(CoverageGap::MissingPermittedMitigation);
            }
            if !has_reference(&mitigation.protected_invariant_refs) {
                gaps.push(CoverageGap::MissingMitigationInvariantLink);
            }
        }
    }
    if !has_reference(&mode.safety_invariant_refs) {
        gaps.push(CoverageGap::MissingSafetyInvariant);
    }
    if !has_reference(&mode.evidence_refs) {
        gaps.push(CoverageGap::MissingEvidenceReference);
    }

    gaps
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_model() -> MissionFaultModel {
        MissionFaultModel {
            schema_version: MISSION_FAULT_MODEL_SCHEMA_VERSION.to_owned(),
            model_id: "synthetic-communications-resilience".to_owned(),
            model_revision: "0.1.0".to_owned(),
            source_design: SourceDesignRef {
                system_id: "synthetic-cubesat-communications".to_owned(),
                revision: "v0.1".to_owned(),
                sha256: "748da4fc495778c2b46ef7397ea4a3f435637f837b2aa8e61d10ef4b4d996656"
                    .to_owned(),
            },
            failure_modes: vec![FailureMode {
                id: "FM-COMM-001".to_owned(),
                component_id: "comms_transceiver".to_owned(),
                interface_id: "ground_comm_link".to_owned(),
                description: "Synthetic total radio-link blackout".to_owned(),
                criticality: FailureCriticality::Critical,
                requirement_refs: vec!["REQ-COMM-001".to_owned()],
                assumption_refs: vec!["ASSUMP-COMM-UPLINK-AVAILABLE-AT-START".to_owned()],
                telemetry_label_semantics: Some("communication_link_unavailable".to_owned()),
                injection: Some(InjectionSpec::CommunicationBlackout {
                    at_step: 4,
                    duration_steps: 3,
                }),
                observable_signals: vec![ObservableSignal {
                    id: "ground_link_available".to_owned(),
                    unit: "boolean".to_owned(),
                    expected_fault_effect:
                        "false for steps 4 through 6; geometry remains unchanged".to_owned(),
                    signature_value: Some("unavailable".to_owned()),
                }],
                monitor: Some(MonitorContract {
                    id: "MON-COMM-001".to_owned(),
                    required_alarm: "COMM_LINK_UNAVAILABLE".to_owned(),
                    response_deadline_steps: 1,
                    false_alarm_controls: vec![
                        "nominal_link_available".to_owned(),
                        "line_of_sight_loss_without_radio_fault".to_owned(),
                    ],
                }),
                mitigation: Some(MitigationConstraint {
                    id: "MIT-COMM-001".to_owned(),
                    allowed_actions: vec![
                        "buffer_noncritical_telemetry".to_owned(),
                        "retry_after_link_recovery".to_owned(),
                    ],
                    protected_invariant_refs: vec!["INV-COMMAND-AUTHORITY".to_owned()],
                }),
                safety_invariant_refs: vec![
                    "INV-COMMAND-AUTHORITY".to_owned(),
                    "INV-POWER-BUDGET".to_owned(),
                ],
                evidence_refs: vec!["EVID-SYNTHETIC-SOURCE-001".to_owned()],
            }],
        }
    }

    #[test]
    fn complete_critical_mode_generates_schedule_and_closes_synthetic_coverage() {
        let manifest = valid_model().compile_campaign().expect("valid synthetic model");
        assert_eq!(manifest.critical_mode_count(), 1);
        assert!(manifest.critical_coverage_closed());
        assert!(manifest.critical_modes_with_gaps().is_empty());
        assert_eq!(manifest.schedule.perturbations.len(), 1);
        assert!(matches!(
            &manifest.schedule.perturbations[0],
            OrbitalPerturbation::CommBlackout {
                at_step: 4,
                duration: 3
            }
        ));
    }

    #[test]
    fn missing_monitor_is_a_gap_even_when_injection_is_generated() {
        let mut model = valid_model();
        model.failure_modes[0].monitor = None;

        let manifest = model.compile_campaign().expect("missing links are coverage gaps");
        assert!(manifest.obligations[0].scenario.is_some());
        assert!(manifest.obligations[0]
            .gaps
            .contains(&CoverageGap::MissingMonitorContract));
        assert!(!manifest.critical_coverage_closed());
    }

    #[test]
    fn missing_observable_is_not_laundered_by_a_simulator_schedule() {
        let mut model = valid_model();
        model.failure_modes[0].observable_signals.clear();

        let manifest = model.compile_campaign().expect("model remains structurally valid");
        assert_eq!(manifest.schedule.perturbations.len(), 1);
        assert!(manifest.obligations[0]
            .gaps
            .contains(&CoverageGap::MissingObservableSignal));
        assert!(!manifest.critical_coverage_closed());
    }

    #[test]
    fn duplicate_failure_modes_and_bad_source_digests_fail_closed() {
        let mut duplicate = valid_model();
        duplicate.failure_modes.push(duplicate.failure_modes[0].clone());
        assert!(matches!(
            duplicate.validate(),
            Err(MissionFaultModelError::DuplicateFailureModeId(_))
        ));

        let mut bad_digest = valid_model();
        bad_digest.source_design.sha256 = "not-a-sha256".to_owned();
        assert_eq!(
            bad_digest.validate(),
            Err(MissionFaultModelError::InvalidSourceSha256)
        );
    }

    #[test]
    fn invalid_injection_parameters_fail_before_manifest_generation() {
        let mut zero_duration = valid_model();
        zero_duration.failure_modes[0].injection =
            Some(InjectionSpec::CommunicationBlackout {
                at_step: 4,
                duration_steps: 0,
            });
        assert!(matches!(
            zero_duration.compile_campaign(),
            Err(MissionFaultModelError::InvalidInjectionParameters { .. })
        ));

        let mut non_finite = valid_model();
        non_finite.failure_modes[0].injection = Some(InjectionSpec::VelocityImpulse {
            at_step: 4,
            delta_v_m_s: [0.0, f64::NAN, 0.0],
        });
        assert!(matches!(
            non_finite.validate(),
            Err(MissionFaultModelError::InvalidInjectionParameters { .. })
        ));
    }

    #[test]
    fn checked_in_fixture_deserializes_and_compiles_with_source_binding() {
        use sha2::{Digest, Sha256};

        let source_design = include_str!("../fixtures/space_fdir_mbse_001/source_design.txt");
        let model: MissionFaultModel = serde_json::from_str(include_str!(
            "../fixtures/space_fdir_mbse_001/model.json"
        ))
        .expect("versioned JSON fixture must deserialize into the typed IR");

        let actual_sha256 = format!("{:x}", Sha256::digest(source_design.as_bytes()));
        assert_eq!(actual_sha256, model.source_design.sha256);
        assert!(source_design.starts_with("Synthetic spacecraft communications design"));
        let manifest = model.compile_campaign().expect("fixture model compiles");
        assert!(manifest.critical_coverage_closed());
        assert_eq!(manifest.source_design_sha256, model.source_design.sha256);
    }

    #[test]
    fn identical_source_model_compiles_deterministically() {
        let model = valid_model();
        let first = model.compile_campaign().expect("first compilation");
        let second = model.compile_campaign().expect("second compilation");
        assert_eq!(first, second);
    }
}
