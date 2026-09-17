// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Preregistered comparison contracts for heterogeneous physical backends.
//!
//! The comparison layer prevents a substrate experiment from selecting or
//! rescaling readouts after seeing results. A plan is bound to an exact
//! PHYS-002 manifest commitment and adds raw-observation projections,
//! structural budgets, a budget-match policy, and direction/role-bearing
//! metric specifications before execution.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use std::collections::BTreeSet;

use symthaea_physical_cognition::ObservationFrame;
use symthaea_physical_experiment::{Digest32, PhysicalExperimentManifest};

const PLAN_DOMAIN: &[u8] = b"symthaea:physical-comparison:plan:v1\0";

/// One affine projection from a raw observation scalar into a comparison axis.
#[derive(Debug, Clone, PartialEq)]
pub struct AxisProjection {
    /// Raw observation key to consume.
    pub source_key: String,
    /// Positive multiplicative normalization.
    pub scale: f64,
    /// Additive normalization applied after scaling.
    pub offset: f64,
}

impl AxisProjection {
    /// Validate projection parameters.
    pub fn validate(&self) -> Result<(), ComparisonError> {
        validate_token("source key", &self.source_key)?;
        if !self.scale.is_finite() || self.scale <= 0.0 {
            return Err(ComparisonError::InvalidScale(self.source_key.clone()));
        }
        if !self.offset.is_finite() {
            return Err(ComparisonError::InvalidOffset(self.source_key.clone()));
        }
        Ok(())
    }

    fn apply(&self, observation: &ObservationFrame) -> Result<f64, ComparisonError> {
        let raw = observation
            .scalars
            .get(&self.source_key)
            .copied()
            .ok_or_else(|| ComparisonError::MissingSource(self.source_key.clone()))?;
        if !raw.is_finite() {
            return Err(ComparisonError::NonFiniteSource(self.source_key.clone()));
        }
        let projected = raw * self.scale + self.offset;
        if !projected.is_finite() {
            return Err(ComparisonError::NonFiniteProjection(self.source_key.clone()));
        }
        Ok(projected)
    }
}

/// Fixed four-axis projection used for fair substrate comparisons.
#[derive(Debug, Clone, PartialEq)]
pub struct ObservationProjection {
    /// Projected mean-state axis.
    pub state_mean: AxisProjection,
    /// Projected state-variance axis.
    pub state_variance: AxisProjection,
    /// Projected activity axis.
    pub activity_fraction: AxisProjection,
    /// Projected primary-output axis.
    pub output: AxisProjection,
}

impl ObservationProjection {
    /// Validate all axes and reject source-key reuse.
    pub fn validate(&self) -> Result<(), ComparisonError> {
        let axes = [
            &self.state_mean,
            &self.state_variance,
            &self.activity_fraction,
            &self.output,
        ];
        for axis in axes {
            axis.validate()?;
        }
        let unique: BTreeSet<&str> = axes.iter().map(|axis| axis.source_key.as_str()).collect();
        if unique.len() != axes.len() {
            return Err(ComparisonError::DuplicateProjectionSource);
        }
        Ok(())
    }

    /// Project a raw backend observation into the shared comparison vector.
    pub fn project(
        &self,
        observation: &ObservationFrame,
    ) -> Result<ComparisonVector, ComparisonError> {
        self.validate()?;
        Ok(ComparisonVector {
            state_mean: self.state_mean.apply(observation)?,
            state_variance: self.state_variance.apply(observation)?,
            activity_fraction: self.activity_fraction.apply(observation)?,
            output: self.output.apply(observation)?,
        })
    }

    /// Identity projection for PHYS-003 standard readouts.
    pub fn standard_identity() -> Self {
        Self {
            state_mean: axis("state_mean"),
            state_variance: axis("state_variance"),
            activity_fraction: axis("activity_fraction"),
            output: axis("output"),
        }
    }
}

/// Shared comparison vector produced after preregistered projection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ComparisonVector {
    /// Normalized mean-state feature.
    pub state_mean: f64,
    /// Normalized state-variance feature.
    pub state_variance: f64,
    /// Normalized activity feature.
    pub activity_fraction: f64,
    /// Normalized primary output.
    pub output: f64,
}

/// Declared structural budget for one backend arm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StructuralBudget {
    /// Logical processing units.
    pub units: u64,
    /// Persistent scalar state variables retained across frames.
    pub persistent_state_scalars: u64,
    /// Recurrent graph edges.
    pub recurrent_edges: u64,
    /// Trainable parameters internal to the backend.
    pub trainable_parameters: u64,
    /// Scalar readouts exposed to the comparison layer per frame.
    pub readout_scalars: u64,
}

impl StructuralBudget {
    /// Validate non-zero unit/readout budgets.
    pub fn validate(&self) -> Result<(), ComparisonError> {
        if self.units == 0 {
            return Err(ComparisonError::ZeroUnits);
        }
        if self.readout_scalars == 0 {
            return Err(ComparisonError::ZeroReadouts);
        }
        Ok(())
    }
}

/// Which structural dimensions must match exactly before execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BudgetMatchPolicy {
    /// Require equal logical unit count.
    pub units: bool,
    /// Require equal persistent-state scalar count.
    pub persistent_state_scalars: bool,
    /// Require equal recurrent edge count.
    pub recurrent_edges: bool,
    /// Require equal internal trainable-parameter count.
    pub trainable_parameters: bool,
    /// Require equal scalar readout count.
    pub readout_scalars: bool,
}

impl BudgetMatchPolicy {
    /// Strictly match every declared structural dimension.
    pub const fn strict() -> Self {
        Self {
            units: true,
            persistent_state_scalars: true,
            recurrent_edges: true,
            trainable_parameters: true,
            readout_scalars: true,
        }
    }

    /// Match state capacity and readout bandwidth while allowing topology edges
    /// to differ as an explicitly studied variable.
    pub const fn state_and_readout_matched() -> Self {
        Self {
            units: true,
            persistent_state_scalars: true,
            recurrent_edges: false,
            trainable_parameters: true,
            readout_scalars: true,
        }
    }

    /// Validate a subject/comparator budget pair.
    pub fn validate_pair(
        &self,
        subject: StructuralBudget,
        comparator: StructuralBudget,
    ) -> Result<(), ComparisonError> {
        subject.validate()?;
        comparator.validate()?;
        check_budget(self.units, "units", subject.units, comparator.units)?;
        check_budget(
            self.persistent_state_scalars,
            "persistent_state_scalars",
            subject.persistent_state_scalars,
            comparator.persistent_state_scalars,
        )?;
        check_budget(
            self.recurrent_edges,
            "recurrent_edges",
            subject.recurrent_edges,
            comparator.recurrent_edges,
        )?;
        check_budget(
            self.trainable_parameters,
            "trainable_parameters",
            subject.trainable_parameters,
            comparator.trainable_parameters,
        )?;
        check_budget(
            self.readout_scalars,
            "readout_scalars",
            subject.readout_scalars,
            comparator.readout_scalars,
        )?;
        Ok(())
    }
}

/// Optimization direction preregistered for a metric.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetricDirection {
    /// Lower values are preferable for this metric.
    Minimize,
    /// Higher values are preferable for this metric.
    Maximize,
}

/// Claim role preregistered for a metric.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetricRole {
    /// Metric participates in the primary preregistered analysis.
    Primary,
    /// Metric is reported as secondary characterization.
    Secondary,
    /// Metric is diagnostic only and cannot support the primary claim.
    Diagnostic,
}

/// One preregistered metric with semantic direction and claim role.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetricSpec {
    /// Stable metric identifier.
    pub id: String,
    /// Whether lower or higher values are preferred.
    pub direction: MetricDirection,
    /// Primary/secondary/diagnostic role.
    pub role: MetricRole,
}

impl MetricSpec {
    /// Validate the stable metric identifier.
    pub fn validate(&self) -> Result<(), ComparisonError> {
        validate_token("metric", &self.id)
    }
}

/// Preregistered comparison details independent of any observed result.
///
/// Backend identity is intentionally *not* duplicated here. The exact subject
/// and comparator implementation/configuration bindings already live in the
/// PHYS-002 manifest whose commitment is embedded by `BoundComparisonPlan`.
#[derive(Debug, Clone, PartialEq)]
pub struct ComparisonProtocol {
    /// Protocol schema version; V1 is currently the only accepted value.
    pub schema_version: u16,
    /// Subject raw-observation projection.
    pub subject_projection: ObservationProjection,
    /// Comparator raw-observation projection.
    pub comparator_projection: ObservationProjection,
    /// Declared subject structural budget.
    pub subject_budget: StructuralBudget,
    /// Declared comparator structural budget.
    pub comparator_budget: StructuralBudget,
    /// Budget dimensions that must match before execution.
    pub budget_policy: BudgetMatchPolicy,
    /// Predeclared metrics. Canonical encoding sorts by ID, so insertion order
    /// does not change protocol identity.
    pub metrics: Vec<MetricSpec>,
}

impl ComparisonProtocol {
    /// Validate the protocol without looking at results.
    pub fn validate(&self) -> Result<(), ComparisonError> {
        if self.schema_version != 1 {
            return Err(ComparisonError::UnsupportedSchema(self.schema_version));
        }
        self.subject_projection.validate()?;
        self.comparator_projection.validate()?;
        self.budget_policy
            .validate_pair(self.subject_budget, self.comparator_budget)?;
        if self.metrics.is_empty() {
            return Err(ComparisonError::EmptyMetrics);
        }
        if self.metrics.len() > 64 {
            return Err(ComparisonError::TooManyMetrics(self.metrics.len()));
        }
        let mut unique = BTreeSet::new();
        for metric in &self.metrics {
            metric.validate()?;
            if !unique.insert(metric.id.as_str()) {
                return Err(ComparisonError::DuplicateMetric(metric.id.clone()));
            }
        }
        Ok(())
    }
}

/// Exact comparison plan bound to a PHYS-002 experiment manifest.
#[derive(Debug, Clone, PartialEq)]
pub struct BoundComparisonPlan {
    /// Commitment of the exact PHYS-002 experiment manifest.
    pub manifest_commitment: Digest32,
    /// Preregistered comparison protocol.
    pub protocol: ComparisonProtocol,
}

impl BoundComparisonPlan {
    /// Validate that this plan is bound to an exact manifest with a comparator.
    pub fn validate_against(
        &self,
        manifest: &PhysicalExperimentManifest,
    ) -> Result<(), ComparisonError> {
        self.protocol.validate()?;
        let actual_commitment = manifest
            .commitment()
            .map_err(|error| ComparisonError::Manifest(error.to_string()))?;
        if self.manifest_commitment != actual_commitment {
            return Err(ComparisonError::ManifestCommitmentMismatch);
        }
        if manifest.comparator.is_none() {
            return Err(ComparisonError::ManifestMissingComparator);
        }
        Ok(())
    }

    /// Deterministic V1 canonical bytes for the bound plan.
    pub fn canonical_bytes(
        &self,
        manifest: &PhysicalExperimentManifest,
    ) -> Result<Vec<u8>, ComparisonError> {
        self.validate_against(manifest)?;
        let mut out = Vec::new();
        out.extend_from_slice(PLAN_DOMAIN);
        out.extend_from_slice(self.manifest_commitment.as_bytes());
        push_u16(&mut out, self.protocol.schema_version);
        push_projection(&mut out, &self.protocol.subject_projection)?;
        push_projection(&mut out, &self.protocol.comparator_projection)?;
        push_budget(&mut out, self.protocol.subject_budget);
        push_budget(&mut out, self.protocol.comparator_budget);
        push_policy(&mut out, self.protocol.budget_policy);
        let mut metrics = self.protocol.metrics.clone();
        metrics.sort_by(|a, b| a.id.cmp(&b.id));
        push_len(&mut out, metrics.len())?;
        for metric in metrics {
            push_metric(&mut out, &metric)?;
        }
        Ok(out)
    }

    /// Domain-separated BLAKE3 commitment to the exact comparison plan.
    pub fn commitment(
        &self,
        manifest: &PhysicalExperimentManifest,
    ) -> Result<Digest32, ComparisonError> {
        Ok(Digest32::blake3(&self.canonical_bytes(manifest)?))
    }
}

/// Fail-closed errors from comparison preregistration or projection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ComparisonError {
    /// Unsupported protocol schema.
    UnsupportedSchema(u16),
    /// Source key or metric identifier was invalid.
    InvalidToken { field: &'static str, value: String },
    /// Affine scale was zero, negative, NaN, or infinite.
    InvalidScale(String),
    /// Affine offset was NaN or infinite.
    InvalidOffset(String),
    /// Two comparison axes reused the same source key.
    DuplicateProjectionSource,
    /// Required raw observation scalar was absent.
    MissingSource(String),
    /// Required raw source value was NaN or infinite.
    NonFiniteSource(String),
    /// Projected value overflowed or became non-finite.
    NonFiniteProjection(String),
    /// Structural budget declared zero units.
    ZeroUnits,
    /// Structural budget declared zero readouts.
    ZeroReadouts,
    /// A required structural dimension did not match.
    BudgetMismatch {
        /// Mismatched budget field.
        field: &'static str,
        /// Subject value.
        subject: u64,
        /// Comparator value.
        comparator: u64,
    },
    /// No metrics were preregistered.
    EmptyMetrics,
    /// Metric list exceeded the protocol bound.
    TooManyMetrics(usize),
    /// Duplicate metric identifier.
    DuplicateMetric(String),
    /// PHYS-002 manifest could not be validated/committed.
    Manifest(String),
    /// Stored manifest commitment did not match the supplied manifest.
    ManifestCommitmentMismatch,
    /// PHYS-002 manifest did not declare a comparator.
    ManifestMissingComparator,
    /// Canonical encoding exceeded a bounded length field.
    LengthOverflow,
}

impl std::fmt::Display for ComparisonError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedSchema(schema) => write!(f, "unsupported comparison schema {schema}"),
            Self::InvalidToken { field, value } => write!(f, "invalid {field}: {value:?}"),
            Self::InvalidScale(key) => write!(f, "invalid positive scale for {key}"),
            Self::InvalidOffset(key) => write!(f, "invalid offset for {key}"),
            Self::DuplicateProjectionSource => write!(f, "projection source keys must be distinct"),
            Self::MissingSource(key) => write!(f, "missing projection source {key}"),
            Self::NonFiniteSource(key) => write!(f, "non-finite projection source {key}"),
            Self::NonFiniteProjection(key) => write!(f, "non-finite projected value from {key}"),
            Self::ZeroUnits => write!(f, "structural budget must declare at least one unit"),
            Self::ZeroReadouts => write!(f, "structural budget must declare at least one readout"),
            Self::BudgetMismatch {
                field,
                subject,
                comparator,
            } => write!(
                f,
                "budget mismatch for {field}: subject={subject}, comparator={comparator}"
            ),
            Self::EmptyMetrics => write!(f, "comparison must preregister at least one metric"),
            Self::TooManyMetrics(count) => write!(f, "too many comparison metrics: {count}"),
            Self::DuplicateMetric(metric) => write!(f, "duplicate comparison metric {metric}"),
            Self::Manifest(error) => write!(f, "invalid experiment manifest: {error}"),
            Self::ManifestCommitmentMismatch => write!(f, "manifest commitment mismatch"),
            Self::ManifestMissingComparator => write!(f, "manifest has no comparator"),
            Self::LengthOverflow => write!(f, "canonical comparison encoding length overflow"),
        }
    }
}

impl std::error::Error for ComparisonError {}

fn axis(key: &str) -> AxisProjection {
    AxisProjection {
        source_key: key.to_string(),
        scale: 1.0,
        offset: 0.0,
    }
}

fn validate_token(field: &'static str, value: &str) -> Result<(), ComparisonError> {
    let valid = !value.is_empty()
        && value.len() <= 96
        && value.bytes().all(|byte| {
            byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b':')
        });
    if valid {
        Ok(())
    } else {
        Err(ComparisonError::InvalidToken {
            field,
            value: value.to_string(),
        })
    }
}

fn check_budget(
    required: bool,
    field: &'static str,
    subject: u64,
    comparator: u64,
) -> Result<(), ComparisonError> {
    if required && subject != comparator {
        Err(ComparisonError::BudgetMismatch {
            field,
            subject,
            comparator,
        })
    } else {
        Ok(())
    }
}

fn push_u16(out: &mut Vec<u8>, value: u16) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_len(out: &mut Vec<u8>, len: usize) -> Result<(), ComparisonError> {
    let len = u32::try_from(len).map_err(|_| ComparisonError::LengthOverflow)?;
    out.extend_from_slice(&len.to_le_bytes());
    Ok(())
}

fn push_str(out: &mut Vec<u8>, value: &str) -> Result<(), ComparisonError> {
    push_len(out, value.len())?;
    out.extend_from_slice(value.as_bytes());
    Ok(())
}

fn push_f64(out: &mut Vec<u8>, value: f64) {
    out.extend_from_slice(&value.to_bits().to_le_bytes());
}

fn push_axis(out: &mut Vec<u8>, axis: &AxisProjection) -> Result<(), ComparisonError> {
    push_str(out, &axis.source_key)?;
    push_f64(out, axis.scale);
    push_f64(out, axis.offset);
    Ok(())
}

fn push_projection(
    out: &mut Vec<u8>,
    projection: &ObservationProjection,
) -> Result<(), ComparisonError> {
    push_axis(out, &projection.state_mean)?;
    push_axis(out, &projection.state_variance)?;
    push_axis(out, &projection.activity_fraction)?;
    push_axis(out, &projection.output)?;
    Ok(())
}

fn push_budget(out: &mut Vec<u8>, budget: StructuralBudget) {
    push_u64(out, budget.units);
    push_u64(out, budget.persistent_state_scalars);
    push_u64(out, budget.recurrent_edges);
    push_u64(out, budget.trainable_parameters);
    push_u64(out, budget.readout_scalars);
}

fn push_policy(out: &mut Vec<u8>, policy: BudgetMatchPolicy) {
    for enabled in [
        policy.units,
        policy.persistent_state_scalars,
        policy.recurrent_edges,
        policy.trainable_parameters,
        policy.readout_scalars,
    ] {
        out.push(u8::from(enabled));
    }
}

fn push_metric(out: &mut Vec<u8>, metric: &MetricSpec) -> Result<(), ComparisonError> {
    metric.validate()?;
    push_str(out, &metric.id)?;
    out.push(match metric.direction {
        MetricDirection::Minimize => 0,
        MetricDirection::Maximize => 1,
    });
    out.push(match metric.role {
        MetricRole::Primary => 0,
        MetricRole::Secondary => 1,
        MetricRole::Diagnostic => 2,
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use symthaea_physical_cognition::{BackendIdentity, ExecutionBoundary, ObservationFrame};
    use symthaea_physical_experiment::{
        BackendBinding, EnergyPolicy, SamplingPlan, SeedPlan, WorkloadIdentity,
    };

    fn binding(family: &str, config: &[u8]) -> BackendBinding {
        BackendBinding {
            backend: BackendIdentity::new(family, "v1", "test-backend").unwrap(),
            implementation_digest: Digest32::blake3(b"test-backend-source"),
            configuration_digest: Digest32::blake3(config),
        }
    }

    fn manifest() -> PhysicalExperimentManifest {
        PhysicalExperimentManifest {
            schema_version: 1,
            experiment_id: "physical:comparison-smoke".to_string(),
            subject: binding("physical:subject", b"subject-config"),
            comparator: Some(binding("control:comparator", b"comparator-config")),
            workload: WorkloadIdentity {
                family: "temporal:smoke".to_string(),
                version: "v1".to_string(),
                fixture_digest: Digest32::blake3(b"fixture"),
            },
            seed_plan: SeedPlan { seeds: vec![1, 2] },
            sampling: SamplingPlan {
                warmup_frames: 2,
                measured_frames: 8,
                samples_per_frame: 1,
            },
            expected_execution: ExecutionBoundary::Simulation,
            energy_policy: EnergyPolicy::UnmeasuredAllowed,
            notes: vec![],
        }
    }

    fn budget() -> StructuralBudget {
        StructuralBudget {
            units: 128,
            persistent_state_scalars: 128,
            recurrent_edges: 256,
            trainable_parameters: 0,
            readout_scalars: 4,
        }
    }

    fn protocol() -> ComparisonProtocol {
        ComparisonProtocol {
            schema_version: 1,
            subject_projection: ObservationProjection {
                state_mean: axis("mean_conductance"),
                state_variance: axis("conductance_variance"),
                activity_fraction: axis("switched_fraction"),
                output: axis("output_current"),
            },
            comparator_projection: ObservationProjection::standard_identity(),
            subject_budget: budget(),
            comparator_budget: budget(),
            budget_policy: BudgetMatchPolicy::strict(),
            metrics: vec![
                MetricSpec {
                    id: "nrmse".to_string(),
                    direction: MetricDirection::Minimize,
                    role: MetricRole::Primary,
                },
                MetricSpec {
                    id: "memory_capacity".to_string(),
                    direction: MetricDirection::Maximize,
                    role: MetricRole::Primary,
                },
            ],
        }
    }

    #[test]
    fn projection_is_explicit_and_fail_closed() {
        let projection = protocol().subject_projection;
        let observation = ObservationFrame {
            sequence: 0,
            scalars: BTreeMap::from([
                ("mean_conductance".to_string(), 0.4),
                ("conductance_variance".to_string(), 0.03),
                ("switched_fraction".to_string(), 0.25),
                ("output_current".to_string(), -0.2),
            ]),
            samples: Some(1),
        };
        let vector = projection.project(&observation).unwrap();
        assert_eq!(vector.state_mean, 0.4);
        assert_eq!(vector.output, -0.2);

        let missing = ObservationFrame {
            scalars: BTreeMap::new(),
            ..observation
        };
        assert!(matches!(
            projection.project(&missing),
            Err(ComparisonError::MissingSource(_))
        ));
    }

    #[test]
    fn projection_rejects_source_reuse() {
        let duplicate = ObservationProjection {
            state_mean: axis("same"),
            state_variance: axis("same"),
            activity_fraction: axis("activity"),
            output: axis("output"),
        };
        assert_eq!(
            duplicate.validate(),
            Err(ComparisonError::DuplicateProjectionSource)
        );
    }

    #[test]
    fn strict_budget_policy_rejects_hidden_capacity() {
        let subject = budget();
        let comparator = StructuralBudget {
            recurrent_edges: subject.recurrent_edges + 1,
            ..subject
        };
        assert!(matches!(
            BudgetMatchPolicy::strict().validate_pair(subject, comparator),
            Err(ComparisonError::BudgetMismatch {
                field: "recurrent_edges",
                ..
            })
        ));
        BudgetMatchPolicy::state_and_readout_matched()
            .validate_pair(subject, comparator)
            .unwrap();
    }

    #[test]
    fn bound_plan_rejects_manifest_drift() {
        let manifest = manifest();
        let plan = BoundComparisonPlan {
            manifest_commitment: manifest.commitment().unwrap(),
            protocol: protocol(),
        };
        plan.validate_against(&manifest).unwrap();

        let mut drifted = manifest.clone();
        drifted.sampling.measured_frames += 1;
        assert_eq!(
            plan.validate_against(&drifted),
            Err(ComparisonError::ManifestCommitmentMismatch)
        );
    }

    #[test]
    fn bound_plan_rejects_missing_comparator() {
        let mut manifest = manifest();
        manifest.comparator = None;
        let plan = BoundComparisonPlan {
            manifest_commitment: manifest.commitment().unwrap(),
            protocol: protocol(),
        };
        assert_eq!(
            plan.validate_against(&manifest),
            Err(ComparisonError::ManifestMissingComparator)
        );
    }

    #[test]
    fn same_backend_identity_different_config_can_be_compared() {
        let mut manifest = manifest();
        let mut comparator = manifest.subject.clone();
        comparator.configuration_digest = Digest32::blake3(b"ablation-config");
        manifest.comparator = Some(comparator);
        let plan = BoundComparisonPlan {
            manifest_commitment: manifest.commitment().unwrap(),
            protocol: protocol(),
        };
        plan.validate_against(&manifest).unwrap();
    }

    #[test]
    fn metric_order_does_not_change_commitment() {
        let manifest = manifest();
        let a = protocol();
        let mut b = a.clone();
        b.metrics.reverse();
        let a = BoundComparisonPlan {
            manifest_commitment: manifest.commitment().unwrap(),
            protocol: a,
        };
        let b = BoundComparisonPlan {
            manifest_commitment: manifest.commitment().unwrap(),
            protocol: b,
        };
        assert_eq!(a.commitment(&manifest).unwrap(), b.commitment(&manifest).unwrap());
    }

    #[test]
    fn metric_direction_and_role_change_commitment() {
        let manifest = manifest();
        let a_protocol = protocol();
        let mut direction_protocol = a_protocol.clone();
        direction_protocol.metrics[0].direction = MetricDirection::Maximize;
        let mut role_protocol = a_protocol.clone();
        role_protocol.metrics[0].role = MetricRole::Secondary;

        let make_plan = |protocol| BoundComparisonPlan {
            manifest_commitment: manifest.commitment().unwrap(),
            protocol,
        };
        let a = make_plan(a_protocol);
        let direction = make_plan(direction_protocol);
        let role = make_plan(role_protocol);
        assert_ne!(a.commitment(&manifest).unwrap(), direction.commitment(&manifest).unwrap());
        assert_ne!(a.commitment(&manifest).unwrap(), role.commitment(&manifest).unwrap());
    }

    #[test]
    fn normalization_change_changes_commitment() {
        let manifest = manifest();
        let a_protocol = protocol();
        let mut b_protocol = a_protocol.clone();
        b_protocol.subject_projection.output.scale = 2.0;
        let a = BoundComparisonPlan {
            manifest_commitment: manifest.commitment().unwrap(),
            protocol: a_protocol,
        };
        let b = BoundComparisonPlan {
            manifest_commitment: manifest.commitment().unwrap(),
            protocol: b_protocol,
        };
        assert_ne!(a.commitment(&manifest).unwrap(), b.commitment(&manifest).unwrap());
    }
}
