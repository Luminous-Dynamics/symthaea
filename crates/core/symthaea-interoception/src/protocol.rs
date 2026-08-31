use std::collections::BTreeSet;
use std::fmt::Write as _;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{
    AllostaticConfig, ForecastBasisId, InteroceptiveDrive, InteroceptiveDynamicsConfig,
    InteroceptiveIntervention, NativeInteroceptiveState, ViabilityChannel,
    INTEROCEPTIVE_MODEL_SEMANTICS_VERSION, INTEROCEPTIVE_SNAPSHOT_SCHEMA_VERSION,
};

pub const PREREGISTRATION_SCHEMA_VERSION: u16 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DrivePhase {
    pub steps: u32,
    pub drive: InteroceptiveDrive,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ScheduledIntervention {
    /// Intervention is applied immediately before this zero-based model step.
    pub before_step: u64,
    pub intervention: InteroceptiveIntervention,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExperimentArmSpec {
    pub arm_id: String,
    /// Opaque label intended for blinded primary analysis exports.
    pub blind_code: String,
    pub initial_state: NativeInteroceptiveState,
    pub dynamics_config: InteroceptiveDynamicsConfig,
    pub phases: Vec<DrivePhase>,
    /// Vector order is execution order when multiple interventions share a step.
    pub interventions: Vec<ScheduledIntervention>,
}

impl ExperimentArmSpec {
    pub fn total_steps(&self) -> Option<u64> {
        self.phases.iter().try_fold(0_u64, |total, phase| {
            total.checked_add(u64::from(phase.steps))
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ProtocolForecastSpec {
    Kinematic {
        config: AllostaticConfig,
    },
    DynamicsAwareConstantDrive {
        config: AllostaticConfig,
        drive: InteroceptiveDrive,
    },
}

impl ProtocolForecastSpec {
    pub fn basis(self) -> ForecastBasisId {
        match self {
            Self::Kinematic { .. } => ForecastBasisId::Kinematic,
            Self::DynamicsAwareConstantDrive { .. } => ForecastBasisId::DynamicsAwareConstantDrive,
        }
    }

    fn config(self) -> AllostaticConfig {
        match self {
            Self::Kinematic { config } | Self::DynamicsAwareConstantDrive { config, .. } => config,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum RegisteredMeasure {
    TerminalHomeostaticWeightedDeviation,
    TerminalHomeostaticPeakDeviation,
    TerminalChannelDeviation { channel: ViabilityChannel },
    TerminalForecastDiscountedDebt { forecast: ProtocolForecastSpec },
    TerminalForecastTerminalDeviation { forecast: ProtocolForecastSpec },
    TerminalForecastBreachExposures { forecast: ProtocolForecastSpec },
    TerminalForecastUniqueBreachedChannels { forecast: ProtocolForecastSpec },
}

impl RegisteredMeasure {
    fn forecast(self) -> Option<ProtocolForecastSpec> {
        match self {
            Self::TerminalForecastDiscountedDebt { forecast }
            | Self::TerminalForecastTerminalDeviation { forecast }
            | Self::TerminalForecastBreachExposures { forecast }
            | Self::TerminalForecastUniqueBreachedChannels { forecast } => Some(forecast),
            Self::TerminalHomeostaticWeightedDeviation
            | Self::TerminalHomeostaticPeakDeviation
            | Self::TerminalChannelDeviation { .. } => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RegisteredMetricSpec {
    pub metric_id: String,
    pub measure: RegisteredMeasure,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OutcomeRef {
    pub arm_id: String,
    pub metric_id: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ExpectedRelation {
    LessThan,
    LessThanOrEqual,
    GreaterThan,
    GreaterThanOrEqual,
    EqualWithin { absolute_tolerance: f32 },
}

impl ExpectedRelation {
    fn validation_error(self) -> Option<String> {
        match self {
            Self::EqualWithin { absolute_tolerance }
                if !absolute_tolerance.is_finite() || absolute_tolerance < 0.0 =>
            {
                Some("equal-within tolerance must be finite and non-negative".into())
            }
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HypothesisSpec {
    pub hypothesis_id: String,
    pub primary: bool,
    pub left: OutcomeRef,
    pub relation: ExpectedRelation,
    pub right: OutcomeRef,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExclusionCriterion {
    pub criterion_id: String,
    pub description: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExperimentPreregistration {
    pub schema_version: u16,
    pub model_semantics_version: u16,
    pub snapshot_schema_version: u16,
    pub protocol_id: String,
    pub analysis_version: String,
    pub blind_arm_identity_during_primary_analysis: bool,
    pub arms: Vec<ExperimentArmSpec>,
    pub metrics: Vec<RegisteredMetricSpec>,
    pub hypotheses: Vec<HypothesisSpec>,
    pub exclusions: Vec<ExclusionCriterion>,
}

impl ExperimentPreregistration {
    pub fn validation_errors(&self) -> Vec<String> {
        let mut errors = Vec::new();

        if self.schema_version != PREREGISTRATION_SCHEMA_VERSION {
            errors.push(format!(
                "unsupported preregistration schema version: {}",
                self.schema_version
            ));
        }
        if self.model_semantics_version != INTEROCEPTIVE_MODEL_SEMANTICS_VERSION {
            errors.push(format!(
                "model semantics version mismatch: {}",
                self.model_semantics_version
            ));
        }
        if self.snapshot_schema_version != INTEROCEPTIVE_SNAPSHOT_SCHEMA_VERSION {
            errors.push(format!(
                "snapshot schema version mismatch: {}",
                self.snapshot_schema_version
            ));
        }
        if self.protocol_id.trim().is_empty() {
            errors.push("protocol_id must not be empty".into());
        }
        if self.analysis_version.trim().is_empty() {
            errors.push("analysis_version must not be empty".into());
        }
        if self.arms.is_empty() {
            errors.push("at least one experimental arm is required".into());
        }
        if self.metrics.is_empty() {
            errors.push("at least one registered metric is required".into());
        }
        if self.hypotheses.is_empty() {
            errors.push("at least one preregistered hypothesis is required".into());
        }
        if !self.hypotheses.iter().any(|hypothesis| hypothesis.primary) {
            errors.push("at least one hypothesis must be marked primary".into());
        }

        let mut arm_ids = BTreeSet::new();
        let mut blind_codes = BTreeSet::new();
        for arm in &self.arms {
            if arm.arm_id.trim().is_empty() {
                errors.push("arm_id must not be empty".into());
            } else if !arm_ids.insert(arm.arm_id.as_str()) {
                errors.push(format!("duplicate arm_id: {}", arm.arm_id));
            }
            if arm.blind_code.trim().is_empty() {
                errors.push(format!("arm {} must have a non-empty blind_code", arm.arm_id));
            } else if !blind_codes.insert(arm.blind_code.as_str()) {
                errors.push(format!("duplicate blind_code: {}", arm.blind_code));
            }
            if let Err(error) = arm.dynamics_config.try_validate() {
                errors.push(format!("arm {} has invalid dynamics config: {error}", arm.arm_id));
            }
            if arm.phases.is_empty() {
                errors.push(format!("arm {} must have at least one drive phase", arm.arm_id));
            }
            if arm.phases.iter().any(|phase| phase.steps == 0) {
                errors.push(format!("arm {} contains a zero-step drive phase", arm.arm_id));
            }

            let Some(total_steps) = arm.total_steps() else {
                errors.push(format!("arm {} total step count overflows u64", arm.arm_id));
                continue;
            };
            if total_steps == 0 {
                errors.push(format!("arm {} total step count must be positive", arm.arm_id));
            }

            let mut previous_step = None;
            for scheduled in &arm.interventions {
                if scheduled.before_step >= total_steps {
                    errors.push(format!(
                        "arm {} intervention at step {} is outside the {}-step run",
                        arm.arm_id, scheduled.before_step, total_steps
                    ));
                }
                if previous_step.is_some_and(|previous| scheduled.before_step < previous) {
                    errors.push(format!(
                        "arm {} interventions must be ordered by before_step",
                        arm.arm_id
                    ));
                }
                previous_step = Some(scheduled.before_step);
            }
        }

        let mut metric_ids = BTreeSet::new();
        for metric in &self.metrics {
            if metric.metric_id.trim().is_empty() {
                errors.push("metric_id must not be empty".into());
            } else if !metric_ids.insert(metric.metric_id.as_str()) {
                errors.push(format!("duplicate metric_id: {}", metric.metric_id));
            }
            if let Some(forecast) = metric.measure.forecast() {
                if let Err(error) = forecast.config().try_validate() {
                    errors.push(format!(
                        "metric {} has invalid forecast config: {error}",
                        metric.metric_id
                    ));
                }
            }
        }

        let mut hypothesis_ids = BTreeSet::new();
        for hypothesis in &self.hypotheses {
            if hypothesis.hypothesis_id.trim().is_empty() {
                errors.push("hypothesis_id must not be empty".into());
            } else if !hypothesis_ids.insert(hypothesis.hypothesis_id.as_str()) {
                errors.push(format!(
                    "duplicate hypothesis_id: {}",
                    hypothesis.hypothesis_id
                ));
            }
            if let Some(error) = hypothesis.relation.validation_error() {
                errors.push(format!("hypothesis {}: {error}", hypothesis.hypothesis_id));
            }
            if hypothesis.left == hypothesis.right {
                errors.push(format!(
                    "hypothesis {} compares an outcome with itself",
                    hypothesis.hypothesis_id
                ));
            }
            for outcome in [&hypothesis.left, &hypothesis.right] {
                let Some(arm) = self.arms.iter().find(|arm| arm.arm_id == outcome.arm_id) else {
                    errors.push(format!(
                        "hypothesis {} references unknown arm {}",
                        hypothesis.hypothesis_id, outcome.arm_id
                    ));
                    continue;
                };
                let Some(metric) = self
                    .metrics
                    .iter()
                    .find(|metric| metric.metric_id == outcome.metric_id)
                else {
                    errors.push(format!(
                        "hypothesis {} references unknown metric {}",
                        hypothesis.hypothesis_id, outcome.metric_id
                    ));
                    continue;
                };
                if let Some(ProtocolForecastSpec::DynamicsAwareConstantDrive { config, .. }) =
                    metric.measure.forecast()
                {
                    if (config.dt - arm.dynamics_config.step_dt).abs() > f32::EPSILON {
                        errors.push(format!(
                            "hypothesis {} uses dynamics-aware metric {} with dt incompatible with arm {}",
                            hypothesis.hypothesis_id, metric.metric_id, arm.arm_id
                        ));
                    }
                }
            }
        }

        let mut exclusion_ids = BTreeSet::new();
        for exclusion in &self.exclusions {
            if exclusion.criterion_id.trim().is_empty() {
                errors.push("exclusion criterion_id must not be empty".into());
            } else if !exclusion_ids.insert(exclusion.criterion_id.as_str()) {
                errors.push(format!(
                    "duplicate exclusion criterion_id: {}",
                    exclusion.criterion_id
                ));
            }
            if exclusion.description.trim().is_empty() {
                errors.push(format!(
                    "exclusion {} must have a non-empty description",
                    exclusion.criterion_id
                ));
            }
        }

        errors
    }

    pub fn validate(&self) -> Result<(), Vec<String>> {
        let errors = self.validation_errors();
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    /// Canonical representation for protocol locking under the pinned dependency set.
    ///
    /// The protocol contains only structs, enums, vectors, fixed arrays, and strings;
    /// there are no unordered maps. With the captured `Cargo.lock`, this byte sequence
    /// is a reproducible prospective-plan identity.
    pub fn canonical_json(&self) -> Result<Vec<u8>, Vec<String>> {
        self.validate()?;
        serde_json::to_vec(self)
            .map_err(|error| vec![format!("failed to serialize preregistration: {error}")])
    }

    pub fn sha256(&self) -> Result<String, Vec<String>> {
        let bytes = self.canonical_json()?;
        let digest = Sha256::digest(&bytes);
        let mut encoded = String::with_capacity(64);
        for byte in digest {
            write!(&mut encoded, "{byte:02x}").expect("writing to a String cannot fail");
        }
        Ok(encoded)
    }
}
