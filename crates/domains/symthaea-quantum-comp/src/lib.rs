//! Experimental quantum and quantum-inspired substrate probes for Symthaea.
//!
//! This crate is intentionally conservative:
//! - it does not claim quantum consciousness;
//! - it does not claim quantum advantage;
//! - it does not replace Qiskit, Cirq, Braket, CUDA-Q, or physical backend tooling;
//! - it provides reproducible research primitives for HDC binding, similarity,
//!   noise, topology-oriented substrate probes, and future circuit export.
//!
//! See `docs/RESEARCH_NOTES.md` for what has actually been measured so far:
//! noise-robustness and bundling-capacity comparisons found no advantage for
//! phase/quantum-inspired encoding over classical binary HDC. Continuous-value
//! storage (`continuous_value_comparison`) found a real effect, but not a
//! flat win for either side — phase wins at zero noise and at higher noise,
//! classical wins at low-to-moderate noise once given a decoder that knows
//! its own channel's error rate (a real crossover around bit-error-rate
//! ≈ 0.10). A follow-up shrinkage probe tested whether an even smarter
//! (partially bias-corrected) classical decoder could close the high-noise
//! gap too — it mostly can't; phase still wins there. Read that document
//! before citing any result from this crate.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod api_inventory;
pub mod audit;
pub mod benchmark;
pub mod beta_readiness;
pub mod bundle;
pub mod calibrated_comparison;
pub mod capacity_comparison;
pub mod classical_hdc;
pub mod comparative;
pub mod continuous_value_comparison;
pub mod controls;
pub mod correlation_hdc;
pub mod entanglement_proxy;
pub mod errors;
pub mod experiment;
pub mod fixtures;
pub mod interop;
pub mod matrix;
pub mod migration;
pub mod noise_sweep;
pub mod phase_hdc;
pub mod preflight;
pub mod presets;
pub mod probe;
pub mod provenance;
#[cfg(feature = "qasm-export")]
pub mod qasm;
pub mod receipts;
pub mod release_gate;
pub mod release_manifest;
pub mod replay;
pub mod reporting;
pub mod rng;
pub mod robustness;
pub mod schema;
pub mod significance;
pub mod stability;
pub mod statistics;
pub mod substrate;
pub mod topology;
pub mod validation_snapshot;
pub mod verification_matrix;

pub use api_inventory::{ApiInventory, current_api_inventory};
pub use audit::{
    AuditFinding, AuditStatus, ClaimAuditReport, audit_binding_probe, audit_negative_control,
    audit_robustness,
};
pub use benchmark::{BenchmarkManifest, BenchmarkResult, BindingProbeReport};
pub use beta_readiness::{
    BetaReadinessFinding, BetaReadinessReport, BetaReadinessStatus, current_beta_readiness,
};
pub use bundle::ResearchBundle;
pub use calibrated_comparison::{
    CalibratedComparisonConfig, CalibratedComparisonReport, CalibratedComparisonRunner,
    CalibratedSweepConfig, CalibratedSweepReport, CalibratedSweepRunner,
    calibrate_phase_sigma_for_ber, classical_channel_ber, measure_classical_channel_ber,
    measure_phase_channel_ber,
};
pub use capacity_comparison::{
    CapacityPoint, CapacitySweepConfig, CapacitySweepReport, CapacitySweepRunner,
};
pub use classical_hdc::BinaryHypervector;
pub use comparative::{
    ComparativeBindingConfig, ComparativeBindingReport, ComparativeBindingRunner,
    MethodComparisonSummary,
};
pub use continuous_value_comparison::{
    ContinuousValuePoint, ContinuousValueSweepConfig, ContinuousValueSweepReport,
    ContinuousValueSweepRunner, ShrinkagePoint, ShrinkageProbeConfig, ShrinkageProbeReport,
    ShrinkageProbeRunner,
};
pub use controls::{NegativeControlConfig, NegativeControlReport, NegativeControlRunner};
pub use correlation_hdc::CorrelationBindingSketch;
pub use entanglement_proxy::{
    EntanglementProxyConfig, EntanglementProxyReport, EntanglementProxyRunner,
    EntanglementProxySketch,
};
pub use errors::{QuantumCompError, Result};
pub use experiment::{ClaimBoundary, ExperimentManifest, ExperimentProtocol};
pub use fixtures::{
    FixtureExpectation, FixtureIntent, FixtureSpec, fixture_catalog, fixture_names, named_fixture,
};
pub use interop::{AdapterAuthority, IntegrationDeclaration, IntegrationTarget};
pub use matrix::{
    ExperimentMatrixCell, ExperimentMatrixConfig, ExperimentMatrixReport, ExperimentMatrixRunner,
};
pub use migration::{MigrationGuide, MigrationRisk, MigrationStep, alpha9_to_alpha10_migration};
pub use noise_sweep::{NoiseSweepConfig, NoiseSweepPoint, NoiseSweepReport, NoiseSweepRunner};
pub use phase_hdc::PhaseHypervector;
pub use preflight::{
    PreflightFinding, PreflightReport, PreflightSeverity, preflight_binding_config,
    preflight_comparative_config, preflight_matrix_config, preflight_noise_sweep_config,
};
pub use presets::{RunPreset, supported_preset_names};
pub use probe::{BindingProbeConfig, BindingProbeRunner};
pub use provenance::{ReproducibilityRecord, RunEnvironment, fnv1a64};
pub use receipts::ResearchArtifactReceipt;
pub use release_gate::{
    ReleaseGateFinding, ReleaseGateReport, ReleaseGateStatus, gate_local_artifact,
};
pub use release_manifest::{AlphaReleaseManifest, ReleaseChannel, current_release_manifest};
pub use replay::{ReplayCommand, ReplayPlan, ReplayScope, supported_replay_scopes};
pub use reporting::{ReportTable, robustness_to_markdown};
pub use robustness::{MethodRobustness, NoiseRobustnessSummary};
pub use schema::{
    BINDING_PROBE_SCHEMA, CRATE_VERSION, EXPERIMENT_MATRIX_SCHEMA, FIXTURE_CATALOG_SCHEMA,
    INTEGRATION_DECLARATION_SCHEMA, NOISE_SWEEP_SCHEMA, RELEASE_GATE_SCHEMA, REPLAY_PLAN_SCHEMA,
    RESEARCH_BUNDLE_SCHEMA, RESEARCH_RECEIPT_SCHEMA, known_schema_labels,
};
pub use significance::{PairedDifferenceSummary, exact_two_sided_sign_test_p_value};
pub use stability::{
    AlphaStability, StabilityRecord, catalog_has_no_deprecated_surfaces, stability_catalog,
};
pub use statistics::{
    SampleSummary, first_threshold_crossing, linear_slope, non_increasing_violations,
    paired_effect_size, trapezoid_auc,
};
pub use substrate::{BackendKind, ConfidenceLevel, SubstrateProfile};
pub use topology::TopologySummary;
pub use validation_snapshot::{ValidationSnapshot, current_validation_snapshot};
pub use verification_matrix::{
    VerificationMatrix, VerificationMatrixRow, VerificationStage, current_verification_matrix,
};
