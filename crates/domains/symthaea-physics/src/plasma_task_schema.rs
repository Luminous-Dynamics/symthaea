//! Versioned heterogeneous plasma task schemas.
//!
//! Task schemas describe the physical meaning, native representation, time
//! support, and evidence policy of context and target signals. They provide a
//! common contract for schema-conditioned encoders/decoders without assuming
//! that every diagnostic is a scalar sampled at the same rate.
//!
//! Validation is structural. It does not prove physical observability,
//! identifiability, prediction quality, or permission to control a machine.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const PLASMA_TASK_SCHEMA_VERSION: &str =
    "luminous.plasma-task-schema.v0.1";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlasmaTaskSchemaError {
    pub message: String,
}

impl PlasmaTaskSchemaError {
    fn new(message: impl Into<String>) -> Self {
        Self { message: message.into() }
    }
}

impl std::fmt::Display for PlasmaTaskSchemaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for PlasmaTaskSchemaError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlasmaSignalModality {
    Scalar,
    TimeSeries,
    Profile,
    Field,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlasmaSignalRole {
    DiagnosticContext,
    PlasmaStateContext,
    ActuatorContext,
    ObservedResponseContext,
    PlasmaResponseTarget,
    ActuatorInversionTarget,
    DisruptionRiskTarget,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceAdmissionPolicy {
    DirectMeasuredOnly,
    MeasuredOrSyntheticForResearch,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskAnchorSemantics {
    ShotRelativeNanoseconds,
    ContextWindowEnd,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlasmaTemporalSupport {
    /// Signed offset from the task anchor. Negative values precede the anchor.
    pub start_offset_ns: i64,
    /// Signed offset from the task anchor. Context must end at or before zero.
    pub end_offset_ns: i64,
    /// Native sampling interval for regularly sampled time series.
    pub sample_interval_ns: Option<u64>,
}

impl PlasmaTemporalSupport {
    fn validate(
        &self,
        modality: PlasmaSignalModality,
        is_context: bool,
        descriptor_id: &str,
    ) -> Result<(), PlasmaTaskSchemaError> {
        if self.start_offset_ns > self.end_offset_ns {
            return Err(PlasmaTaskSchemaError::new(format!(
                "signal descriptor {descriptor_id} has reversed temporal support"
            )));
        }
        if is_context && self.end_offset_ns > 0 {
            return Err(PlasmaTaskSchemaError::new(format!(
                "context signal {descriptor_id} extends beyond the task anchor"
            )));
        }
        if !is_context && self.start_offset_ns < 0 {
            return Err(PlasmaTaskSchemaError::new(format!(
                "target signal {descriptor_id} begins before the task anchor"
            )));
        }
        match (modality, self.sample_interval_ns) {
            (PlasmaSignalModality::TimeSeries, None) => {
                return Err(PlasmaTaskSchemaError::new(format!(
                    "time-series signal {descriptor_id} requires a native sample interval"
                )));
            }
            (_, Some(0)) => {
                return Err(PlasmaTaskSchemaError::new(format!(
                    "signal {descriptor_id} sample interval must be positive"
                )));
            }
            _ => {}
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlasmaSignalDescriptor {
    /// Unique descriptor identity. A physical signal may appear in both
    /// context and target with different descriptor IDs and time support.
    pub descriptor_id: String,
    pub signal_id: String,
    pub diagnostic_system_id: String,
    pub modality: PlasmaSignalModality,
    pub role: PlasmaSignalRole,
    pub native_units: String,
    /// Empty for scalar values; otherwise dimensions in native signal order.
    pub native_shape: Vec<usize>,
    pub temporal_support: PlasmaTemporalSupport,
    pub required: bool,
    /// Present only for context inputs. Targets are generated quantities.
    pub evidence_admission: Option<EvidenceAdmissionPolicy>,
}

impl PlasmaSignalDescriptor {
    fn validate(
        &self,
        is_context: bool,
        descriptor_ids: &mut BTreeSet<String>,
    ) -> Result<(), PlasmaTaskSchemaError> {
        for (value, field) in [
            (self.descriptor_id.as_str(), "descriptor_id"),
            (self.signal_id.as_str(), "signal_id"),
            (self.diagnostic_system_id.as_str(), "diagnostic_system_id"),
            (self.native_units.as_str(), "native_units"),
        ] {
            if value.trim().is_empty() {
                return Err(PlasmaTaskSchemaError::new(format!(
                    "signal descriptor has an empty {field}"
                )));
            }
        }
        if !descriptor_ids.insert(self.descriptor_id.clone()) {
            return Err(PlasmaTaskSchemaError::new(format!(
                "duplicate signal descriptor ID: {}",
                self.descriptor_id
            )));
        }

        match self.modality {
            PlasmaSignalModality::Scalar if !self.native_shape.is_empty() => {
                return Err(PlasmaTaskSchemaError::new(format!(
                    "scalar signal {} must have an empty native_shape",
                    self.descriptor_id
                )));
            }
            PlasmaSignalModality::TimeSeries
            | PlasmaSignalModality::Profile
            | PlasmaSignalModality::Field
                if self.native_shape.is_empty()
                    || self.native_shape.contains(&0) =>
            {
                return Err(PlasmaTaskSchemaError::new(format!(
                    "non-scalar signal {} requires positive native dimensions",
                    self.descriptor_id
                )));
            }
            _ => {}
        }

        self.temporal_support.validate(
            self.modality,
            is_context,
            &self.descriptor_id,
        )?;

        if is_context {
            let valid_role = matches!(
                self.role,
                PlasmaSignalRole::DiagnosticContext
                    | PlasmaSignalRole::PlasmaStateContext
                    | PlasmaSignalRole::ActuatorContext
                    | PlasmaSignalRole::ObservedResponseContext
            );
            if !valid_role {
                return Err(PlasmaTaskSchemaError::new(format!(
                    "context signal {} uses a target-only role",
                    self.descriptor_id
                )));
            }
            if self.evidence_admission.is_none() {
                return Err(PlasmaTaskSchemaError::new(format!(
                    "context signal {} must declare an evidence admission policy",
                    self.descriptor_id
                )));
            }
        } else {
            let valid_role = matches!(
                self.role,
                PlasmaSignalRole::PlasmaResponseTarget
                    | PlasmaSignalRole::ActuatorInversionTarget
                    | PlasmaSignalRole::DisruptionRiskTarget
            );
            if !valid_role {
                return Err(PlasmaTaskSchemaError::new(format!(
                    "target signal {} uses a context-only role",
                    self.descriptor_id
                )));
            }
            if self.evidence_admission.is_some() {
                return Err(PlasmaTaskSchemaError::new(format!(
                    "target signal {} must not declare an input evidence admission policy",
                    self.descriptor_id
                )));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlasmaTaskSchema {
    pub schema_version: String,
    pub task_id: String,
    pub anchor_semantics: TaskAnchorSemantics,
    pub clock_domain: String,
    pub context_signals: Vec<PlasmaSignalDescriptor>,
    pub target_signals: Vec<PlasmaSignalDescriptor>,
}

impl PlasmaTaskSchema {
    pub fn validate(&self) -> Result<(), PlasmaTaskSchemaError> {
        if self.schema_version != PLASMA_TASK_SCHEMA_VERSION {
            return Err(PlasmaTaskSchemaError::new(
                "unsupported plasma task schema version",
            ));
        }
        if self.task_id.trim().is_empty() || self.clock_domain.trim().is_empty() {
            return Err(PlasmaTaskSchemaError::new(
                "task_id and clock_domain must be non-empty",
            ));
        }
        if self.context_signals.is_empty() || self.target_signals.is_empty() {
            return Err(PlasmaTaskSchemaError::new(
                "task schema requires at least one context and one target signal",
            ));
        }

        let mut descriptor_ids = BTreeSet::new();
        for descriptor in &self.context_signals {
            descriptor.validate(true, &mut descriptor_ids)?;
        }
        for descriptor in &self.target_signals {
            descriptor.validate(false, &mut descriptor_ids)?;
        }
        Ok(())
    }

    /// True when each context signal may have its own sampling interval. No
    /// global resampling or zero-fill requirement is implied by the schema.
    pub fn is_multi_rate(&self) -> bool {
        let mut intervals = BTreeSet::new();
        for descriptor in &self.context_signals {
            if let Some(interval) = descriptor.temporal_support.sample_interval_ns {
                intervals.insert(interval);
            }
        }
        intervals.len() > 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn context(
        descriptor_id: &str,
        signal_id: &str,
        modality: PlasmaSignalModality,
        shape: Vec<usize>,
        sample_interval_ns: Option<u64>,
        end_offset_ns: i64,
    ) -> PlasmaSignalDescriptor {
        PlasmaSignalDescriptor {
            descriptor_id: descriptor_id.to_owned(),
            signal_id: signal_id.to_owned(),
            diagnostic_system_id: format!("system-{signal_id}"),
            modality,
            role: PlasmaSignalRole::DiagnosticContext,
            native_units: "test-unit".to_owned(),
            native_shape: shape,
            temporal_support: PlasmaTemporalSupport {
                start_offset_ns: -100_000_000,
                end_offset_ns,
                sample_interval_ns,
            },
            required: true,
            evidence_admission: Some(EvidenceAdmissionPolicy::DirectMeasuredOnly),
        }
    }

    fn risk_target() -> PlasmaSignalDescriptor {
        PlasmaSignalDescriptor {
            descriptor_id: "risk-target".to_owned(),
            signal_id: "soft-ttd-risk".to_owned(),
            diagnostic_system_id: "derived-risk-target".to_owned(),
            modality: PlasmaSignalModality::Scalar,
            role: PlasmaSignalRole::DisruptionRiskTarget,
            native_units: "dimensionless".to_owned(),
            native_shape: Vec::new(),
            temporal_support: PlasmaTemporalSupport {
                start_offset_ns: 0,
                end_offset_ns: 0,
                sample_interval_ns: None,
            },
            required: true,
            evidence_admission: None,
        }
    }

    fn group6_schema() -> PlasmaTaskSchema {
        PlasmaTaskSchema {
            schema_version: PLASMA_TASK_SCHEMA_VERSION.to_owned(),
            task_id: "tokamark-group6-3-risk".to_owned(),
            anchor_semantics: TaskAnchorSemantics::ContextWindowEnd,
            clock_domain: "shot-relative-ns".to_owned(),
            context_signals: vec![context(
                "ip-context",
                "summary-ip",
                PlasmaSignalModality::TimeSeries,
                vec![400],
                Some(250_000),
                0,
            )],
            target_signals: vec![risk_target()],
        }
    }

    #[test]
    fn single_channel_group6_schema_validates_with_native_time_support() {
        let schema = group6_schema();
        schema.validate().expect("valid Group 6 schema");
        assert!(!schema.is_multi_rate());
    }

    #[test]
    fn heterogeneous_multi_rate_context_is_allowed_without_forced_alignment() {
        let mut schema = group6_schema();
        let mut temperature = context(
            "temperature-context",
            "electron-temperature",
            PlasmaSignalModality::TimeSeries,
            vec![20],
            Some(5_000_000),
            0,
        );
        temperature.role = PlasmaSignalRole::PlasmaStateContext;
        temperature.evidence_admission = Some(EvidenceAdmissionPolicy::MeasuredOrSyntheticForResearch);
        schema.context_signals.push(temperature);
        schema.validate().expect("multi-rate schema should validate");
        assert!(schema.is_multi_rate());
    }

    #[test]
    fn future_context_is_rejected_before_model_admission() {
        let mut schema = group6_schema();
        schema.context_signals[0].temporal_support.end_offset_ns = 1;
        assert!(schema.validate().is_err());
    }

    #[test]
    fn time_series_requires_native_interval_and_nonzero_shape() {
        let mut schema = group6_schema();
        schema.context_signals[0].temporal_support.sample_interval_ns = None;
        assert!(schema.validate().is_err());

        let mut zero_shape = group6_schema();
        zero_shape.context_signals[0].native_shape = vec![0];
        assert!(zero_shape.validate().is_err());
    }

    #[test]
    fn duplicate_descriptor_ids_are_rejected_but_signal_role_reuse_is_explicit() {
        let mut schema = group6_schema();
        let mut future_ip_target = schema.context_signals[0].clone();
        future_ip_target.descriptor_id = "future-ip-target".to_owned();
        future_ip_target.role = PlasmaSignalRole::PlasmaResponseTarget;
        future_ip_target.temporal_support.start_offset_ns = 0;
        future_ip_target.temporal_support.end_offset_ns = 25_000_000;
        future_ip_target.evidence_admission = None;
        schema.target_signals.push(future_ip_target);
        schema.validate().expect("same signal ID may appear as context and target");

        schema.target_signals[1].descriptor_id = schema.target_signals[0].descriptor_id.clone();
        assert!(schema.validate().is_err());
    }

    #[test]
    fn frozen_group6_single_channel_schema_deserializes_and_validates() {
        let manifest: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../benchmarks/plasma/task-schema-v0.json"
        ))
        .expect("task schema manifest should parse");
        let value = manifest["example_schemas"]["group6_task_6_3_single_plasma_current"].clone();
        let schema: PlasmaTaskSchema =
            serde_json::from_value(value).expect("Group 6 schema should deserialize");
        schema.validate().expect("Group 6 schema should validate");
        assert_eq!(schema.task_id, "task_6-3");
        assert_eq!(schema.context_signals[0].native_shape, vec![400]);
        assert_eq!(
            schema.context_signals[0].temporal_support.sample_interval_ns,
            Some(250_000)
        );
        assert_eq!(
            schema.context_signals[0].temporal_support.start_offset_ns,
            -100_000_000
        );
        assert_eq!(schema.target_signals[0].role, PlasmaSignalRole::DisruptionRiskTarget);
    }

    #[test]
    fn context_and_target_roles_and_evidence_fields_are_disjoint() {
        let mut bad_context = group6_schema();
        bad_context.context_signals[0].role = PlasmaSignalRole::DisruptionRiskTarget;
        assert!(bad_context.validate().is_err());

        let mut bad_target = group6_schema();
        bad_target.target_signals[0].evidence_admission =
            Some(EvidenceAdmissionPolicy::DirectMeasuredOnly);
        assert!(bad_target.validate().is_err());
    }

    #[test]
    fn manifest_pins_schema_contract_and_temporal_causality() {
        let manifest: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../benchmarks/plasma/task-schema-v0.json"
        ))
        .expect("event evaluation manifest");
        let contract = &manifest["schema_conditioned_task_contract"];
        assert_eq!(
            contract["schema_version"],
            PLASMA_TASK_SCHEMA_VERSION
        );
        assert_eq!(
            contract["context_temporal_rule"],
            "context_support_end_offset_ns_must_be_lte_zero"
        );
        assert_eq!(
            contract["multi_rate_rule"],
            "per-signal native sample intervals are retained; no global zero-fill or forced rate conversion"
        );
        assert_eq!(
            contract["group6_task_anchor"],
            "end of the 100ms causal history window"
        );
    }
}
