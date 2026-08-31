use serde::{de, Deserialize, Deserializer, Serialize};

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

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct InteroceptiveSnapshot {
    pub schema_version: u16,
    pub model_semantics_version: u16,
    pub cycle: u64,
    pub dynamics_config: InteroceptiveDynamicsConfig,
    pub state: NativeInteroceptiveState,
    pub homeostasis: HomeostaticReport,
    pub forecast: AllostaticForecastSnapshot,
}

#[derive(Deserialize)]
struct InteroceptiveSnapshotWire {
    schema_version: u16,
    model_semantics_version: u16,
    cycle: u64,
    dynamics_config: InteroceptiveDynamicsConfig,
    state: NativeInteroceptiveState,
    homeostasis: HomeostaticReport,
    forecast: AllostaticForecastSnapshot,
}

impl<'de> Deserialize<'de> for InteroceptiveSnapshot {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = InteroceptiveSnapshotWire::deserialize(deserializer)?;
        let snapshot = Self {
            schema_version: wire.schema_version,
            model_semantics_version: wire.model_semantics_version,
            cycle: wire.cycle,
            dynamics_config: wire.dynamics_config,
            state: wire.state,
            homeostasis: wire.homeostasis,
            forecast: wire.forecast,
        };
        snapshot.validate().map_err(|errors| {
            de::Error::custom(format!(
                "invalid interoceptive snapshot: {}",
                errors.join("; ")
            ))
        })?;
        Ok(snapshot)
    }
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

    pub fn validation_errors(&self) -> Vec<String> {
        let mut errors = Vec::new();
        if self.schema_version != INTEROCEPTIVE_SNAPSHOT_SCHEMA_VERSION {
            errors.push(format!(
                "snapshot schema version mismatch: {}",
                self.schema_version
            ));
        }
        if self.model_semantics_version != INTEROCEPTIVE_MODEL_SEMANTICS_VERSION {
            errors.push(format!(
                "model semantics version mismatch: {}",
                self.model_semantics_version
            ));
        }
        if let Err(error) = self.dynamics_config.try_validate() {
            errors.push(format!("invalid dynamics config: {error}"));
        }

        let expected_homeostasis = assess_homeostasis(&self.state);
        if self.homeostasis != expected_homeostasis {
            errors.push("homeostatic report does not match serialized state".into());
        }

        match &self.forecast {
            AllostaticForecastSnapshot::Kinematic { config, report } => {
                if let Err(error) = config.try_validate() {
                    errors.push(format!("invalid kinematic forecast config: {error}"));
                } else {
                    let expected = assess_allostasis(&self.state, *config);
                    if report != &expected {
                        errors.push("kinematic forecast report does not match serialized state".into());
                    }
                }
            }
            AllostaticForecastSnapshot::DynamicsAwareConstantDrive {
                config,
                drive,
                report,
            } => {
                if let Err(error) = config.try_validate() {
                    errors.push(format!("invalid dynamics-aware forecast config: {error}"));
                } else if (config.dt - self.dynamics_config.step_dt).abs() > f32::EPSILON {
                    errors.push("dynamics-aware forecast dt does not match model step_dt".into());
                } else if self.dynamics_config.try_validate().is_ok() {
                    let model = NativeInteroceptiveModel::new(
                        self.state.clone(),
                        self.dynamics_config,
                    );
                    let expected = assess_allostasis_with_drive(&model, *drive, *config);
                    if report != &expected {
                        errors.push(
                            "dynamics-aware forecast report does not match serialized state and drive"
                                .into(),
                        );
                    }
                }
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
}
