use serde::{Deserialize, Serialize};

use crate::{
    assess_allostasis, assess_allostasis_with_drive, assess_homeostasis, AllostaticConfig,
    AllostaticReport, HomeostaticReport, InteroceptiveDrive, InteroceptiveDynamicsConfig,
    NativeInteroceptiveModel, NativeInteroceptiveState,
};

/// Scientific semantics version for the native interoceptive model contract.
/// Increment when a change alters the meaning of state, regulation, or forecast behavior.
pub const INTEROCEPTIVE_MODEL_SEMANTICS_VERSION: u16 = 1;
pub const INTEROCEPTIVE_SNAPSHOT_SCHEMA_VERSION: u16 = 2;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AllostaticForecastSnapshot {
    /// Linear projection from the state's measured velocity.
    Kinematic {
        config: AllostaticConfig,
        report: AllostaticReport,
    },
    /// Rollout of the native transition law under an explicit constant drive.
    DynamicsAwareConstantDrive {
        config: AllostaticConfig,
        drive: InteroceptiveDrive,
        report: AllostaticReport,
    },
}

impl AllostaticForecastSnapshot {
    pub fn report(&self) -> &AllostaticReport {
        match self {
            Self::Kinematic { report, .. } | Self::DynamicsAwareConstantDrive { report, .. } => {
                report
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InteroceptiveSnapshot {
    pub schema_version: u16,
    pub model_semantics_version: u16,
    pub cycle: u64,
    pub dynamics_config: InteroceptiveDynamicsConfig,
    pub state: NativeInteroceptiveState,
    pub homeostasis: HomeostaticReport,
    pub forecast: AllostaticForecastSnapshot,
}

impl InteroceptiveSnapshot {
    /// Backward-compatible alias for a kinematic snapshot.
    pub fn capture(model: &NativeInteroceptiveModel, allostatic_config: AllostaticConfig) -> Self {
        Self::capture_kinematic(model, allostatic_config)
    }

    pub fn capture_kinematic(
        model: &NativeInteroceptiveModel,
        allostatic_config: AllostaticConfig,
    ) -> Self {
        let state = model.state().clone();
        let report = assess_allostasis(&state, allostatic_config);
        Self {
            schema_version: INTEROCEPTIVE_SNAPSHOT_SCHEMA_VERSION,
            model_semantics_version: INTEROCEPTIVE_MODEL_SEMANTICS_VERSION,
            cycle: model.cycle(),
            dynamics_config: model.config(),
            homeostasis: assess_homeostasis(&state),
            forecast: AllostaticForecastSnapshot::Kinematic {
                config: allostatic_config,
                report,
            },
            state,
        }
    }

    pub fn capture_with_drive(
        model: &NativeInteroceptiveModel,
        drive: InteroceptiveDrive,
        allostatic_config: AllostaticConfig,
    ) -> Self {
        let state = model.state().clone();
        let report = assess_allostasis_with_drive(model, drive, allostatic_config);
        Self {
            schema_version: INTEROCEPTIVE_SNAPSHOT_SCHEMA_VERSION,
            model_semantics_version: INTEROCEPTIVE_MODEL_SEMANTICS_VERSION,
            cycle: model.cycle(),
            dynamics_config: model.config(),
            homeostasis: assess_homeostasis(&state),
            forecast: AllostaticForecastSnapshot::DynamicsAwareConstantDrive {
                config: allostatic_config,
                drive,
                report,
            },
            state,
        }
    }
}
