// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Mechanical qualification evidence for integration behavior.
//!
//! Runtime observations live in this crate; evidence that an adapter actually
//! obeyed its declared profile is recorded through the existing canonical
//! `symthaea-evidence-plane` contract.

use crate::manifest::IntegrationManifest;
use crate::observation::ObservationBatch;
use std::collections::BTreeMap;
use symthaea_evidence_plane::{EvidenceCounters, Expectation, RunEvidence, RunId};

pub const MANIFEST_VALID: &str = "integration.manifest_valid";
pub const OBSERVATIONS_EMITTED: &str = "integration.observations_emitted";
pub const OBSERVATION_VALIDATION_FAILURES: &str = "integration.observation_validation_failures";
pub const MUTATION_ATTEMPTS: &str = "integration.mutation_attempts";
pub const UNDECLARED_OPERATIONS: &str = "integration.undeclared_operations";

/// Measured counters for the v0.1 read-only contract.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ReadOnlyConformanceCounters {
    pub observations_emitted: u64,
    pub observation_validation_failures: u64,
    pub mutation_attempts: u64,
    pub undeclared_operations: u64,
}

impl ReadOnlyConformanceCounters {
    /// Record a returned observation batch and mechanically validate it.
    pub fn record_batch(&mut self, batch: &ObservationBatch) {
        self.observations_emitted += batch.observations.len() as u64;
        if batch.validate().is_err() {
            self.observation_validation_failures += 1;
        }
    }

    pub fn record_mutation_attempt(&mut self) {
        self.mutation_attempts += 1;
    }

    pub fn record_undeclared_operation(&mut self) {
        self.undeclared_operations += 1;
    }

    fn as_evidence(&self, manifest_valid: bool) -> EvidenceCounters {
        let mut measured = EvidenceCounters::new();
        measured.record(MANIFEST_VALID, if manifest_valid { 1.0 } else { 0.0 });
        measured.record(OBSERVATIONS_EMITTED, self.observations_emitted as f64);
        measured.record(
            OBSERVATION_VALIDATION_FAILURES,
            self.observation_validation_failures as f64,
        );
        measured.record(MUTATION_ATTEMPTS, self.mutation_attempts as f64);
        measured.record(UNDECLARED_OPERATIONS, self.undeclared_operations as f64);
        measured
    }
}

/// Evaluate the strict v0.1 read-only profile.
///
/// This deliberately returns `RunEvidence` even when the manifest is invalid;
/// qualification failures should be exportable evidence, not disappear as an
/// early-return error.
pub fn evaluate_read_only_conformance(
    run_id: RunId,
    manifest: &IntegrationManifest,
    counters: &ReadOnlyConformanceCounters,
) -> RunEvidence {
    let manifest_valid = manifest.validate_read_only_profile().is_ok();

    let declared = BTreeMap::from([
        (MANIFEST_VALID.to_string(), Expectation::MustBePositive),
        (
            OBSERVATIONS_EMITTED.to_string(),
            Expectation::MustBePositive,
        ),
        (
            OBSERVATION_VALIDATION_FAILURES.to_string(),
            Expectation::MustBeZero,
        ),
        (MUTATION_ATTEMPTS.to_string(), Expectation::MustBeZero),
        (
            UNDECLARED_OPERATIONS.to_string(),
            Expectation::MustBeZero,
        ),
    ]);

    RunEvidence::new(
        run_id,
        manifest,
        declared,
        counters.as_evidence(manifest_valid),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::{
        AccessMode, CapabilityClass, CapabilityDeclaration, IntegrationId,
        INTEGRATION_MANIFEST_SCHEMA_VERSION, MaturityLevel, RiskClass,
    };
    use crate::observation::{
        EntityRef, ObservationBatch, ObservationEnvelope, ObservationId, ObservationKind,
        ObservationLineage, ObservationQuality, ObservationSource, ObservationValue,
    };

    fn manifest() -> IntegrationManifest {
        IntegrationManifest {
            schema_version: INTEGRATION_MANIFEST_SCHEMA_VERSION,
            id: IntegrationId::new("fixture"),
            display_name: "Fixture".into(),
            version: "0.1.0".into(),
            provider: "test".into(),
            protocols: vec!["fixture".into()],
            entity_kinds: vec!["host".into()],
            capabilities: vec![CapabilityDeclaration {
                name: "observe.host.metrics".into(),
                class: CapabilityClass::Observe,
                access: AccessMode::ReadOnly,
                risk: RiskClass::ReadOnly,
                reversible: false,
                default_enabled: true,
            }],
            credentials: vec![],
            maturity: MaturityLevel::E1FixtureParsing,
            default_read_only: true,
        }
    }

    fn valid_batch() -> ObservationBatch {
        ObservationBatch {
            integration_id: "fixture".into(),
            collected_at_unix_ms: 2,
            observations: vec![ObservationEnvelope::new(
                ObservationId::new("obs-1"),
                1,
                2,
                EntityRef::new("test", "host", "node"),
                ObservationKind::Metric,
                "system.cpu.utilization",
                ObservationValue::Number {
                    value: 0.5,
                    unit: Some("1".into()),
                },
                ObservationSource {
                    integration_id: "fixture".into(),
                    collector_id: None,
                    upstream_origin: Some("kernel:procfs".into()),
                    measurement_method: "fixture".into(),
                    tenant: None,
                },
                ObservationQuality::observed(1.0),
                ObservationLineage {
                    lineage_id: "lineage-1".into(),
                    parent_ids: vec![],
                    independence_group: Some("kernel-procfs".into()),
                    transforms: vec![],
                },
            )],
        }
    }

    #[test]
    fn clean_read_only_run_satisfies_contract() {
        let mut counters = ReadOnlyConformanceCounters::default();
        counters.record_batch(&valid_batch());
        let evidence = evaluate_read_only_conformance(RunId::new("clean"), &manifest(), &counters);
        assert!(evidence.satisfied);
    }

    #[test]
    fn mutation_attempt_is_a_hard_evidence_failure() {
        let mut counters = ReadOnlyConformanceCounters::default();
        counters.record_batch(&valid_batch());
        counters.record_mutation_attempt();
        let evidence = evaluate_read_only_conformance(RunId::new("mutation"), &manifest(), &counters);
        assert!(!evidence.satisfied);
        assert!(evidence.violations.iter().any(|v| v.name == MUTATION_ATTEMPTS));
    }
}
