// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Frozen pure-SU(3) campaign manifest and execution lineage.
//!
//! This module moves the lattice program from reusable numerical/statistical
//! components toward one reproducible scientific campaign subject. It freezes
//! sampler, RNG, chain, measurement, flow, benchmark and qualification-policy
//! choices before production chains run, then records execution separately.
//!
//! **Plan != execution != qualification != physical result.**
//!
//! A valid manifest does not prove equilibrium. A valid execution record does
//! not prove the retained ensemble is scientifically adequate. Those judgments
//! remain downstream evidence gates.

use std::collections::BTreeSet;

pub const PURE_SU3_CAMPAIGN_MANIFEST_ID: &str = "pure_su3_campaign_manifest_v1";
pub const PURE_SU3_CAMPAIGN_RUN_ID: &str = "pure_su3_campaign_run_v1";
pub const WILSON_PURE_GAUGE_ACTION_ID: &str = "wilson_pure_gauge_v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CampaignInitialCondition {
    ColdIdentity,
    QualifiedDisordered,
    ExternalQualified,
}

impl CampaignInitialCondition {
    fn stable_id(self) -> &'static str {
        match self {
            Self::ColdIdentity => "cold_identity",
            Self::QualifiedDisordered => "qualified_disordered",
            Self::ExternalQualified => "external_qualified",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CampaignChainPlan {
    pub chain_id: String,
    pub initial_condition: CampaignInitialCondition,
    /// Required for non-cold starts; identifies the initialization theorem/data.
    pub initialization_evidence_id: Option<String>,
    /// Initialization and production streams are separate by construction.
    pub initialization_stream_id: String,
    pub production_stream_id: String,
    /// Optional tuning stream. Tuning must finish before production and may not
    /// share either initialization or production randomness.
    pub tuning_stream_id: Option<String>,
    /// Commitment to unrevealed seed material before execution.
    pub seed_commitment: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct WilsonRectangleSpec {
    pub spatial_direction: usize,
    pub temporal_direction: usize,
    pub spatial_extent: usize,
    pub temporal_extent: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CampaignFlowPlan {
    pub flow_implementation_id: String,
    pub flow_implementation_revision: String,
    pub flow_step_size: f64,
    /// Strictly increasing measurement times; every value must be on the exact
    /// integer `flow_step_size` grid.
    pub measurement_flow_times: Vec<f64>,
    pub energy_operator_id: String,
    pub topology_operator_id: String,
    pub flow_oracle_evidence_id: String,
    pub flow_exact_head_ci_evidence_id: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CampaignBenchmarkPlan {
    /// Stable name for the external validation target.
    pub benchmark_id: String,
    /// Immutable digest of the exact external/reference data or extraction.
    pub benchmark_source_digest: String,
    /// Immutable comparison/acceptance contract. The manifest intentionally does
    /// not encode a universal tolerance here.
    pub benchmark_contract_digest: String,
    pub required_observable_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PureSu3CampaignManifest {
    pub campaign_id: String,
    pub dims: [usize; 4],
    pub beta: f64,
    pub action_id: String,
    /// The first campaign is intentionally periodic in all four directions.
    pub periodic_boundaries: bool,

    pub sampler_id: String,
    pub sampler_config_digest: String,
    pub sampler_implementation_revision: String,
    pub sampler_transition_evidence_id: String,
    pub sampler_action_parity_evidence_id: String,
    pub sampler_exact_head_ci_evidence_id: String,

    pub rng_algorithm_id: String,
    pub rng_implementation_revision: String,
    pub rng_qualification_evidence_id: String,

    pub chains: Vec<CampaignChainPlan>,
    pub thermalization_cycles: u64,
    pub measurement_stride_cycles: u64,
    pub planned_measurements_per_chain: usize,

    pub measure_plaquette: bool,
    pub measure_polyakov_loop: bool,
    pub wilson_rectangles: Vec<WilsonRectangleSpec>,
    pub flow: CampaignFlowPlan,

    /// Frozen diagnostic policy from LQCD-018H.
    pub qualification_policy_artifact_digest: String,
    pub qualification_policy_freeze_evidence_id: String,
    pub qualification_policy_frozen_at_unix_ns: u128,

    pub benchmark: CampaignBenchmarkPlan,

    /// Exact orchestration/measurement code subject, independent of the sampler
    /// implementation revision above.
    pub campaign_code_revision: String,
    pub campaign_configuration_digest: String,
    pub campaign_freeze_evidence_id: String,
    pub campaign_frozen_at_unix_ns: u128,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CampaignChainRunRecord {
    pub chain_id: String,
    pub completed_cycles: u64,
    /// Cycle numbers at which retained measurements were emitted.
    pub retained_measurement_cycles: Vec<u64>,
    pub raw_trajectory_artifact_digest: String,
    pub sampler_trace_artifact_digest: String,
    /// Digest of disclosed seed material after completion.
    pub seed_reveal_digest: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PureSu3CampaignRunRecord {
    pub run_schema_id: &'static str,
    /// Digest of the externally frozen canonical campaign manifest bytes.
    pub campaign_manifest_artifact_digest: String,
    pub campaign_id: String,
    pub campaign_code_revision: String,
    pub started_at_unix_ns: u128,
    pub completed_at_unix_ns: u128,
    pub chains: Vec<CampaignChainRunRecord>,
    pub combined_measurement_artifact_digest: String,
    pub run_receipt_artifact_digest: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CampaignManifestError {
    EmptyField(&'static str),
    InvalidSha256Digest { field: &'static str, value: String },
    InvalidDims([usize; 4]),
    InvalidBeta(f64),
    UnsupportedAction(String),
    NonPeriodicCampaign,
    TooFewChains(usize),
    EmptyChainId { index: usize },
    DuplicateChainId { chain_id: String },
    MissingInitializationEvidence { index: usize },
    UnexpectedColdInitializationEvidence { index: usize },
    EmptyStreamId { index: usize, field: &'static str },
    ReusedRngStream { stream_id: String },
    MissingColdStart,
    MissingDisorderedStart,
    InvalidThermalizationCycles(u64),
    InvalidMeasurementStride(u64),
    InvalidMeasurementCount(usize),
    MeasurementScheduleOverflow,
    EmptyMeasurementProgram,
    InvalidWilsonRectangle { index: usize },
    DuplicateWilsonRectangle { index: usize },
    InvalidFlowStepSize(f64),
    TooFewFlowTimes(usize),
    NonFiniteFlowTime { index: usize, value: f64 },
    NonIncreasingFlowTime { previous: f64, current: f64 },
    FlowTimeOffStepGrid { index: usize, value: f64 },
    InvalidQualificationFreezeTimestamp,
    InvalidCampaignFreezeTimestamp,
    PolicyNotFrozenBeforeCampaign,
    EmptyBenchmarkObservableSet,
    EmptyBenchmarkObservableId { index: usize },
    DuplicateBenchmarkObservableId { observable_id: String },
    MissingScientificExecutionEvidence(&'static str),
    WrongRunSchema,
    RunCampaignMismatch,
    RunRevisionMismatch,
    InvalidRunChronology,
    ChainRunCountMismatch { planned: usize, actual: usize },
    MissingChainRun { chain_id: String },
    DuplicateChainRun { chain_id: String },
    ChainExecutionIncomplete { chain_id: String, required: u64, actual: u64 },
    ChainMeasurementScheduleMismatch { chain_id: String },
}

fn require_nonempty(value: &str, field: &'static str) -> Result<(), CampaignManifestError> {
    if value.trim().is_empty() {
        Err(CampaignManifestError::EmptyField(field))
    } else {
        Ok(())
    }
}

fn require_sha256(value: &str, field: &'static str) -> Result<(), CampaignManifestError> {
    let Some(hex) = value.strip_prefix("sha256:") else {
        return Err(CampaignManifestError::InvalidSha256Digest {
            field,
            value: value.to_string(),
        });
    };
    if hex.len() != 64 || !hex.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(CampaignManifestError::InvalidSha256Digest {
            field,
            value: value.to_string(),
        });
    }
    Ok(())
}

fn encode_string(value: &str) -> String {
    format!("{}:{}", value.len(), value)
}

impl PureSu3CampaignManifest {
    pub fn retained_measurement_cycles(&self) -> Result<Vec<u64>, CampaignManifestError> {
        let mut out = Vec::with_capacity(self.planned_measurements_per_chain);
        for index in 1..=self.planned_measurements_per_chain {
            let offset = self
                .measurement_stride_cycles
                .checked_mul(index as u64)
                .ok_or(CampaignManifestError::MeasurementScheduleOverflow)?;
            let cycle = self
                .thermalization_cycles
                .checked_add(offset)
                .ok_or(CampaignManifestError::MeasurementScheduleOverflow)?;
            out.push(cycle);
        }
        Ok(out)
    }

    pub fn required_completed_cycles(&self) -> Result<u64, CampaignManifestError> {
        self.retained_measurement_cycles()?
            .last()
            .copied()
            .ok_or(CampaignManifestError::InvalidMeasurementCount(0))
    }

    pub fn validate(&self) -> Result<(), CampaignManifestError> {
        require_nonempty(&self.campaign_id, "campaign_id")?;
        require_nonempty(&self.action_id, "action_id")?;
        if self.dims.iter().any(|extent| *extent < 2) {
            return Err(CampaignManifestError::InvalidDims(self.dims));
        }
        if !self.beta.is_finite() || self.beta <= 0.0 {
            return Err(CampaignManifestError::InvalidBeta(self.beta));
        }
        if self.action_id != WILSON_PURE_GAUGE_ACTION_ID {
            return Err(CampaignManifestError::UnsupportedAction(self.action_id.clone()));
        }
        if !self.periodic_boundaries {
            return Err(CampaignManifestError::NonPeriodicCampaign);
        }

        require_nonempty(&self.sampler_id, "sampler_id")?;
        require_sha256(&self.sampler_config_digest, "sampler_config_digest")?;
        require_nonempty(
            &self.sampler_implementation_revision,
            "sampler_implementation_revision",
        )?;
        require_nonempty(
            &self.sampler_transition_evidence_id,
            "sampler_transition_evidence_id",
        )?;
        require_nonempty(
            &self.sampler_action_parity_evidence_id,
            "sampler_action_parity_evidence_id",
        )?;

        require_nonempty(&self.rng_algorithm_id, "rng_algorithm_id")?;
        require_nonempty(
            &self.rng_implementation_revision,
            "rng_implementation_revision",
        )?;
        require_nonempty(
            &self.rng_qualification_evidence_id,
            "rng_qualification_evidence_id",
        )?;

        if self.chains.len() < 2 {
            return Err(CampaignManifestError::TooFewChains(self.chains.len()));
        }
        let mut chain_ids = BTreeSet::new();
        let mut stream_ids = BTreeSet::new();
        let mut has_cold = false;
        let mut has_disordered = false;
        for (index, chain) in self.chains.iter().enumerate() {
            let chain_id = chain.chain_id.trim();
            if chain_id.is_empty() {
                return Err(CampaignManifestError::EmptyChainId { index });
            }
            if !chain_ids.insert(chain_id) {
                return Err(CampaignManifestError::DuplicateChainId {
                    chain_id: chain_id.to_string(),
                });
            }
            match chain.initial_condition {
                CampaignInitialCondition::ColdIdentity => {
                    has_cold = true;
                    if chain
                        .initialization_evidence_id
                        .as_deref()
                        .is_some_and(|value| !value.trim().is_empty())
                    {
                        return Err(
                            CampaignManifestError::UnexpectedColdInitializationEvidence { index },
                        );
                    }
                }
                CampaignInitialCondition::QualifiedDisordered => {
                    has_disordered = true;
                    if chain
                        .initialization_evidence_id
                        .as_deref()
                        .is_none_or(|value| value.trim().is_empty())
                    {
                        return Err(CampaignManifestError::MissingInitializationEvidence {
                            index,
                        });
                    }
                }
                CampaignInitialCondition::ExternalQualified => {
                    if chain
                        .initialization_evidence_id
                        .as_deref()
                        .is_none_or(|value| value.trim().is_empty())
                    {
                        return Err(CampaignManifestError::MissingInitializationEvidence {
                            index,
                        });
                    }
                }
            }
            for (field, stream) in [
                ("initialization_stream_id", chain.initialization_stream_id.as_str()),
                ("production_stream_id", chain.production_stream_id.as_str()),
            ] {
                if stream.trim().is_empty() {
                    return Err(CampaignManifestError::EmptyStreamId { index, field });
                }
                if !stream_ids.insert(stream) {
                    return Err(CampaignManifestError::ReusedRngStream {
                        stream_id: stream.to_string(),
                    });
                }
            }
            if let Some(stream) = chain.tuning_stream_id.as_deref() {
                if stream.trim().is_empty() {
                    return Err(CampaignManifestError::EmptyStreamId {
                        index,
                        field: "tuning_stream_id",
                    });
                }
                if !stream_ids.insert(stream) {
                    return Err(CampaignManifestError::ReusedRngStream {
                        stream_id: stream.to_string(),
                    });
                }
            }
            require_sha256(&chain.seed_commitment, "chain.seed_commitment")?;
        }
        if !has_cold {
            return Err(CampaignManifestError::MissingColdStart);
        }
        if !has_disordered {
            return Err(CampaignManifestError::MissingDisorderedStart);
        }

        if self.thermalization_cycles == 0 {
            return Err(CampaignManifestError::InvalidThermalizationCycles(0));
        }
        if self.measurement_stride_cycles == 0 {
            return Err(CampaignManifestError::InvalidMeasurementStride(0));
        }
        if self.planned_measurements_per_chain == 0 {
            return Err(CampaignManifestError::InvalidMeasurementCount(0));
        }
        self.retained_measurement_cycles()?;

        if !self.measure_plaquette
            && !self.measure_polyakov_loop
            && self.wilson_rectangles.is_empty()
            && self.flow.measurement_flow_times.is_empty()
        {
            return Err(CampaignManifestError::EmptyMeasurementProgram);
        }
        let mut rectangles = BTreeSet::new();
        for (index, rectangle) in self.wilson_rectangles.iter().copied().enumerate() {
            let valid = rectangle.spatial_direction < 4
                && rectangle.temporal_direction < 4
                && rectangle.spatial_direction != rectangle.temporal_direction
                && rectangle.spatial_extent > 0
                && rectangle.temporal_extent > 0
                && rectangle.spatial_extent < self.dims[rectangle.spatial_direction]
                && rectangle.temporal_extent < self.dims[rectangle.temporal_direction];
            if !valid {
                return Err(CampaignManifestError::InvalidWilsonRectangle { index });
            }
            if !rectangles.insert(rectangle) {
                return Err(CampaignManifestError::DuplicateWilsonRectangle { index });
            }
        }

        require_nonempty(&self.flow.flow_implementation_id, "flow.flow_implementation_id")?;
        require_nonempty(
            &self.flow.flow_implementation_revision,
            "flow.flow_implementation_revision",
        )?;
        require_nonempty(&self.flow.energy_operator_id, "flow.energy_operator_id")?;
        require_nonempty(&self.flow.topology_operator_id, "flow.topology_operator_id")?;
        require_nonempty(
            &self.flow.flow_oracle_evidence_id,
            "flow.flow_oracle_evidence_id",
        )?;
        if !self.flow.flow_step_size.is_finite() || self.flow.flow_step_size <= 0.0 {
            return Err(CampaignManifestError::InvalidFlowStepSize(
                self.flow.flow_step_size,
            ));
        }
        if self.flow.measurement_flow_times.len() < 2 {
            return Err(CampaignManifestError::TooFewFlowTimes(
                self.flow.measurement_flow_times.len(),
            ));
        }
        for (index, flow_time) in self.flow.measurement_flow_times.iter().copied().enumerate() {
            if !flow_time.is_finite() || flow_time <= 0.0 {
                return Err(CampaignManifestError::NonFiniteFlowTime {
                    index,
                    value: flow_time,
                });
            }
            if index > 0 && flow_time <= self.flow.measurement_flow_times[index - 1] {
                return Err(CampaignManifestError::NonIncreasingFlowTime {
                    previous: self.flow.measurement_flow_times[index - 1],
                    current: flow_time,
                });
            }
            let step_count = flow_time / self.flow.flow_step_size;
            let nearest = step_count.round();
            let tolerance = 1.0e-10 * (1.0 + step_count.abs());
            if nearest < 1.0 || (step_count - nearest).abs() > tolerance {
                return Err(CampaignManifestError::FlowTimeOffStepGrid {
                    index,
                    value: flow_time,
                });
            }
        }

        require_sha256(
            &self.qualification_policy_artifact_digest,
            "qualification_policy_artifact_digest",
        )?;
        require_nonempty(
            &self.qualification_policy_freeze_evidence_id,
            "qualification_policy_freeze_evidence_id",
        )?;
        if self.qualification_policy_frozen_at_unix_ns == 0 {
            return Err(CampaignManifestError::InvalidQualificationFreezeTimestamp);
        }

        require_nonempty(&self.benchmark.benchmark_id, "benchmark.benchmark_id")?;
        require_sha256(
            &self.benchmark.benchmark_source_digest,
            "benchmark.benchmark_source_digest",
        )?;
        require_sha256(
            &self.benchmark.benchmark_contract_digest,
            "benchmark.benchmark_contract_digest",
        )?;
        if self.benchmark.required_observable_ids.is_empty() {
            return Err(CampaignManifestError::EmptyBenchmarkObservableSet);
        }
        let mut benchmark_observables = BTreeSet::new();
        for (index, observable) in self.benchmark.required_observable_ids.iter().enumerate() {
            let observable = observable.trim();
            if observable.is_empty() {
                return Err(CampaignManifestError::EmptyBenchmarkObservableId { index });
            }
            if !benchmark_observables.insert(observable) {
                return Err(CampaignManifestError::DuplicateBenchmarkObservableId {
                    observable_id: observable.to_string(),
                });
            }
        }

        require_nonempty(&self.campaign_code_revision, "campaign_code_revision")?;
        require_sha256(
            &self.campaign_configuration_digest,
            "campaign_configuration_digest",
        )?;
        require_nonempty(
            &self.campaign_freeze_evidence_id,
            "campaign_freeze_evidence_id",
        )?;
        if self.campaign_frozen_at_unix_ns == 0 {
            return Err(CampaignManifestError::InvalidCampaignFreezeTimestamp);
        }
        if self.campaign_frozen_at_unix_ns < self.qualification_policy_frozen_at_unix_ns {
            return Err(CampaignManifestError::PolicyNotFrozenBeforeCampaign);
        }
        Ok(())
    }

    /// Stronger pre-run gate for a campaign intended to generate promotable
    /// scientific evidence. This requires exact-head evidence for the sampler
    /// and flow implementation rather than allowing those fields to remain empty
    /// while a plan is still being drafted.
    pub fn validate_for_scientific_execution(&self) -> Result<(), CampaignManifestError> {
        self.validate()?;
        for (field, value) in [
            (
                "sampler_exact_head_ci_evidence_id",
                self.sampler_exact_head_ci_evidence_id.as_str(),
            ),
            (
                "flow.flow_exact_head_ci_evidence_id",
                self.flow.flow_exact_head_ci_evidence_id.as_str(),
            ),
        ] {
            if value.trim().is_empty() {
                return Err(CampaignManifestError::MissingScientificExecutionEvidence(field));
            }
        }
        Ok(())
    }

    /// Canonical material intended for external hashing/signing. Floats use bit
    /// patterns and list order is semantically meaningful except where explicitly
    /// sorted below (Wilson rectangles and benchmark observable sets).
    pub fn canonical_material(&self) -> Result<String, CampaignManifestError> {
        self.validate()?;
        let mut rectangles = self.wilson_rectangles.clone();
        rectangles.sort();
        let rectangle_material = rectangles
            .iter()
            .map(|rectangle| {
                format!(
                    "{},{},{},{}",
                    rectangle.spatial_direction,
                    rectangle.temporal_direction,
                    rectangle.spatial_extent,
                    rectangle.temporal_extent
                )
            })
            .collect::<Vec<_>>()
            .join("|");
        let mut benchmark_observables = self.benchmark.required_observable_ids.clone();
        benchmark_observables.sort();
        let benchmark_material = benchmark_observables
            .iter()
            .map(|value| encode_string(value))
            .collect::<Vec<_>>()
            .join("|");
        let chain_material = self
            .chains
            .iter()
            .map(|chain| {
                format!(
                    "{};{};{};{};{};{};{}",
                    encode_string(&chain.chain_id),
                    chain.initial_condition.stable_id(),
                    encode_string(chain.initialization_evidence_id.as_deref().unwrap_or("")),
                    encode_string(&chain.initialization_stream_id),
                    encode_string(&chain.production_stream_id),
                    encode_string(chain.tuning_stream_id.as_deref().unwrap_or("")),
                    encode_string(&chain.seed_commitment)
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        let flow_times = self
            .flow
            .measurement_flow_times
            .iter()
            .map(|value| format!("{:016x}", value.to_bits()))
            .collect::<Vec<_>>()
            .join(",");
        Ok([
            format!("schema={PURE_SU3_CAMPAIGN_MANIFEST_ID}"),
            format!("campaign_id={}", encode_string(&self.campaign_id)),
            format!("dims={},{},{},{}", self.dims[0], self.dims[1], self.dims[2], self.dims[3]),
            format!("beta_bits={:016x}", self.beta.to_bits()),
            format!("action_id={}", encode_string(&self.action_id)),
            format!("periodic={}", self.periodic_boundaries),
            format!("sampler_id={}", encode_string(&self.sampler_id)),
            format!("sampler_config={}", encode_string(&self.sampler_config_digest)),
            format!("sampler_revision={}", encode_string(&self.sampler_implementation_revision)),
            format!("sampler_transition_evidence={}", encode_string(&self.sampler_transition_evidence_id)),
            format!("sampler_action_parity={}", encode_string(&self.sampler_action_parity_evidence_id)),
            format!("sampler_ci={}", encode_string(&self.sampler_exact_head_ci_evidence_id)),
            format!("rng_algorithm={}", encode_string(&self.rng_algorithm_id)),
            format!("rng_revision={}", encode_string(&self.rng_implementation_revision)),
            format!("rng_evidence={}", encode_string(&self.rng_qualification_evidence_id)),
            format!("chains=\n{chain_material}"),
            format!("thermalization_cycles={}", self.thermalization_cycles),
            format!("measurement_stride={}", self.measurement_stride_cycles),
            format!("measurement_count={}", self.planned_measurements_per_chain),
            format!("measure_plaquette={}", self.measure_plaquette),
            format!("measure_polyakov={}", self.measure_polyakov_loop),
            format!("wilson_rectangles={rectangle_material}"),
            format!("flow_id={}", encode_string(&self.flow.flow_implementation_id)),
            format!("flow_revision={}", encode_string(&self.flow.flow_implementation_revision)),
            format!("flow_dt_bits={:016x}", self.flow.flow_step_size.to_bits()),
            format!("flow_times_bits={flow_times}"),
            format!("energy_operator={}", encode_string(&self.flow.energy_operator_id)),
            format!("topology_operator={}", encode_string(&self.flow.topology_operator_id)),
            format!("flow_oracle={}", encode_string(&self.flow.flow_oracle_evidence_id)),
            format!("flow_ci={}", encode_string(&self.flow.flow_exact_head_ci_evidence_id)),
            format!("qualification_policy={}", encode_string(&self.qualification_policy_artifact_digest)),
            format!("qualification_freeze={}", encode_string(&self.qualification_policy_freeze_evidence_id)),
            format!("qualification_frozen_at={}", self.qualification_policy_frozen_at_unix_ns),
            format!("benchmark_id={}", encode_string(&self.benchmark.benchmark_id)),
            format!("benchmark_source={}", encode_string(&self.benchmark.benchmark_source_digest)),
            format!("benchmark_contract={}", encode_string(&self.benchmark.benchmark_contract_digest)),
            format!("benchmark_observables={benchmark_material}"),
            format!("campaign_revision={}", encode_string(&self.campaign_code_revision)),
            format!("campaign_config={}", encode_string(&self.campaign_configuration_digest)),
            format!("campaign_freeze={}", encode_string(&self.campaign_freeze_evidence_id)),
            format!("campaign_frozen_at={}", self.campaign_frozen_at_unix_ns),
        ]
        .join("\n"))
    }
}

impl PureSu3CampaignRunRecord {
    pub fn validate_against(
        &self,
        manifest: &PureSu3CampaignManifest,
    ) -> Result<(), CampaignManifestError> {
        manifest.validate_for_scientific_execution()?;
        if self.run_schema_id != PURE_SU3_CAMPAIGN_RUN_ID {
            return Err(CampaignManifestError::WrongRunSchema);
        }
        require_sha256(
            &self.campaign_manifest_artifact_digest,
            "campaign_manifest_artifact_digest",
        )?;
        require_sha256(
            &self.combined_measurement_artifact_digest,
            "combined_measurement_artifact_digest",
        )?;
        require_sha256(
            &self.run_receipt_artifact_digest,
            "run_receipt_artifact_digest",
        )?;
        if self.campaign_id != manifest.campaign_id {
            return Err(CampaignManifestError::RunCampaignMismatch);
        }
        if self.campaign_code_revision != manifest.campaign_code_revision {
            return Err(CampaignManifestError::RunRevisionMismatch);
        }
        if self.started_at_unix_ns == 0
            || self.completed_at_unix_ns <= self.started_at_unix_ns
            || self.started_at_unix_ns <= manifest.campaign_frozen_at_unix_ns
        {
            return Err(CampaignManifestError::InvalidRunChronology);
        }
        if self.chains.len() != manifest.chains.len() {
            return Err(CampaignManifestError::ChainRunCountMismatch {
                planned: manifest.chains.len(),
                actual: self.chains.len(),
            });
        }
        let expected_cycles = manifest.retained_measurement_cycles()?;
        let required_completed = manifest.required_completed_cycles()?;
        let mut seen = BTreeSet::new();
        for record in &self.chains {
            if !seen.insert(record.chain_id.as_str()) {
                return Err(CampaignManifestError::DuplicateChainRun {
                    chain_id: record.chain_id.clone(),
                });
            }
            if !manifest.chains.iter().any(|chain| chain.chain_id == record.chain_id) {
                return Err(CampaignManifestError::MissingChainRun {
                    chain_id: record.chain_id.clone(),
                });
            }
            if record.completed_cycles < required_completed {
                return Err(CampaignManifestError::ChainExecutionIncomplete {
                    chain_id: record.chain_id.clone(),
                    required: required_completed,
                    actual: record.completed_cycles,
                });
            }
            if record.retained_measurement_cycles != expected_cycles {
                return Err(CampaignManifestError::ChainMeasurementScheduleMismatch {
                    chain_id: record.chain_id.clone(),
                });
            }
            require_sha256(
                &record.raw_trajectory_artifact_digest,
                "chain.raw_trajectory_artifact_digest",
            )?;
            require_sha256(
                &record.sampler_trace_artifact_digest,
                "chain.sampler_trace_artifact_digest",
            )?;
            require_sha256(&record.seed_reveal_digest, "chain.seed_reveal_digest")?;
        }
        for chain in &manifest.chains {
            if !seen.contains(chain.chain_id.as_str()) {
                return Err(CampaignManifestError::MissingChainRun {
                    chain_id: chain.chain_id.clone(),
                });
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(ch: char) -> String {
        format!("sha256:{}", ch.to_string().repeat(64))
    }

    fn manifest() -> PureSu3CampaignManifest {
        PureSu3CampaignManifest {
            campaign_id: "pure-su3-qualification-001".into(),
            dims: [4, 4, 4, 8],
            beta: 6.0,
            action_id: WILSON_PURE_GAUGE_ACTION_ID.into(),
            periodic_boundaries: true,
            sampler_id: "cm_heatbath_or_v1;force=staple;or=1;max_attempts=64".into(),
            sampler_config_digest: digest('a'),
            sampler_implementation_revision: "sampler-head".into(),
            sampler_transition_evidence_id: "LQCD-016H".into(),
            sampler_action_parity_evidence_id: "LQCD-011/012".into(),
            sampler_exact_head_ci_evidence_id: "sampler-ci".into(),
            rng_algorithm_id: "chacha8_counter_stream_v1".into(),
            rng_implementation_revision: "rng-head".into(),
            rng_qualification_evidence_id: "LQCD-015".into(),
            chains: vec![
                CampaignChainPlan {
                    chain_id: "chain-cold".into(),
                    initial_condition: CampaignInitialCondition::ColdIdentity,
                    initialization_evidence_id: None,
                    initialization_stream_id: "init-0".into(),
                    production_stream_id: "prod-0".into(),
                    tuning_stream_id: None,
                    seed_commitment: digest('b'),
                },
                CampaignChainPlan {
                    chain_id: "chain-disordered".into(),
                    initial_condition: CampaignInitialCondition::QualifiedDisordered,
                    initialization_evidence_id: Some("disordered-start-v1".into()),
                    initialization_stream_id: "init-1".into(),
                    production_stream_id: "prod-1".into(),
                    tuning_stream_id: None,
                    seed_commitment: digest('c'),
                },
            ],
            thermalization_cycles: 1_000,
            measurement_stride_cycles: 20,
            planned_measurements_per_chain: 100,
            measure_plaquette: true,
            measure_polyakov_loop: true,
            wilson_rectangles: vec![
                WilsonRectangleSpec { spatial_direction: 0, temporal_direction: 3, spatial_extent: 1, temporal_extent: 1 },
                WilsonRectangleSpec { spatial_direction: 0, temporal_direction: 3, spatial_extent: 2, temporal_extent: 1 },
                WilsonRectangleSpec { spatial_direction: 0, temporal_direction: 3, spatial_extent: 1, temporal_extent: 2 },
                WilsonRectangleSpec { spatial_direction: 0, temporal_direction: 3, spatial_extent: 2, temporal_extent: 2 },
            ],
            flow: CampaignFlowPlan {
                flow_implementation_id: "wilson_action_staple_rk3_v1".into(),
                flow_implementation_revision: "flow-head".into(),
                flow_step_size: 0.001,
                measurement_flow_times: vec![0.01, 0.02, 0.04, 0.08],
                energy_operator_id: "symthaea_clover_energy_v1".into(),
                topology_operator_id: "symthaea_clover_topology_v1".into(),
                flow_oracle_evidence_id: "LQCD-017D/RK3".into(),
                flow_exact_head_ci_evidence_id: "flow-ci".into(),
            },
            qualification_policy_artifact_digest: digest('d'),
            qualification_policy_freeze_evidence_id: "policy-freeze".into(),
            qualification_policy_frozen_at_unix_ns: 1_000,
            benchmark: CampaignBenchmarkPlan {
                benchmark_id: "pure-su3-reference-v1".into(),
                benchmark_source_digest: digest('e'),
                benchmark_contract_digest: digest('f'),
                required_observable_ids: vec!["plaquette".into(), "wilson-loop-1x1".into()],
            },
            campaign_code_revision: "campaign-head".into(),
            campaign_configuration_digest: digest('1'),
            campaign_freeze_evidence_id: "campaign-freeze".into(),
            campaign_frozen_at_unix_ns: 2_000,
        }
    }

    #[test]
    fn scientific_campaign_manifest_closes_all_pre_run_boundaries() {
        let manifest = manifest();
        manifest.validate_for_scientific_execution().unwrap();
        assert_eq!(manifest.retained_measurement_cycles().unwrap()[0], 1_020);
        assert_eq!(manifest.required_completed_cycles().unwrap(), 3_000);
        assert!(manifest.canonical_material().unwrap().contains("beta_bits="));
    }

    #[test]
    fn rng_stream_reuse_across_chains_fails_closed() {
        let mut manifest = manifest();
        manifest.chains[1].production_stream_id = "prod-0".into();
        assert!(matches!(
            manifest.validate(),
            Err(CampaignManifestError::ReusedRngStream { .. })
        ));
    }

    #[test]
    fn cold_and_disordered_starts_are_both_required() {
        let mut manifest = manifest();
        manifest.chains[1].initial_condition = CampaignInitialCondition::ExternalQualified;
        assert!(matches!(
            manifest.validate(),
            Err(CampaignManifestError::MissingDisorderedStart)
        ));
    }

    #[test]
    fn winding_rectangle_cannot_masquerade_as_static_rectangle() {
        let mut manifest = manifest();
        manifest.wilson_rectangles.push(WilsonRectangleSpec {
            spatial_direction: 0,
            temporal_direction: 3,
            spatial_extent: 4,
            temporal_extent: 1,
        });
        assert!(matches!(
            manifest.validate(),
            Err(CampaignManifestError::InvalidWilsonRectangle { .. })
        ));
    }

    #[test]
    fn scientific_execution_requires_exact_head_sampler_and_flow_evidence() {
        let mut manifest = manifest();
        manifest.sampler_exact_head_ci_evidence_id.clear();
        assert!(matches!(
            manifest.validate_for_scientific_execution(),
            Err(CampaignManifestError::MissingScientificExecutionEvidence(
                "sampler_exact_head_ci_evidence_id"
            ))
        ));
    }

    #[test]
    fn execution_must_match_predeclared_schedule() {
        let manifest = manifest();
        let expected = manifest.retained_measurement_cycles().unwrap();
        let run = PureSu3CampaignRunRecord {
            run_schema_id: PURE_SU3_CAMPAIGN_RUN_ID,
            campaign_manifest_artifact_digest: digest('2'),
            campaign_id: manifest.campaign_id.clone(),
            campaign_code_revision: manifest.campaign_code_revision.clone(),
            started_at_unix_ns: 3_000,
            completed_at_unix_ns: 4_000,
            chains: manifest.chains.iter().map(|chain| CampaignChainRunRecord {
                chain_id: chain.chain_id.clone(),
                completed_cycles: 3_000,
                retained_measurement_cycles: expected.clone(),
                raw_trajectory_artifact_digest: digest('3'),
                sampler_trace_artifact_digest: digest('4'),
                seed_reveal_digest: digest('5'),
            }).collect(),
            combined_measurement_artifact_digest: digest('6'),
            run_receipt_artifact_digest: digest('7'),
        };
        run.validate_against(&manifest).unwrap();
    }
}
