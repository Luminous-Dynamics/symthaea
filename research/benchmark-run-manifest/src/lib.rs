//! Canonical provenance and resource-boundary contract for one benchmark run.
//!
//! This manifest is not a signature. Its domain-separated SHA-256 is a stable
//! identity which can later be signed by the independent evaluator receipt
//! service. A complete manifest does not make two unlike workloads comparable;
//! it makes their differences explicit and machine-checkable.

use std::collections::BTreeSet;
use std::fmt;

use sha2::{Digest, Sha256};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProtocolMode {
    OfficialReproduction,
    LeakageControlled,
    SyntheticControl,
}

impl ProtocolMode {
    fn code(self) -> u8 {
        match self {
            Self::OfficialReproduction => 1,
            Self::LeakageControlled => 2,
            Self::SyntheticControl => 3,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DatasetAccessClass {
    Public,
    Credentialed,
    RestrictedDerived,
    Synthetic,
}

impl DatasetAccessClass {
    fn code(self) -> u8 {
        match self {
            Self::Public => 1,
            Self::Credentialed => 2,
            Self::RestrictedDerived => 3,
            Self::Synthetic => 4,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScalerFitScope {
    TrainOnly,
    FullDataset,
    ExternalPretrained,
    NotFitted,
}

impl ScalerFitScope {
    fn code(self) -> u8 {
        match self {
            Self::TrainOnly => 1,
            Self::FullDataset => 2,
            Self::ExternalPretrained => 3,
            Self::NotFitted => 4,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MissingnessPolicy {
    PreserveMask,
    DropMissing,
    Impute { imputer_sha256: [u8; 32] },
}

impl MissingnessPolicy {
    fn code(self) -> u8 {
        match self {
            Self::PreserveMask => 1,
            Self::DropMissing => 2,
            Self::Impute { .. } => 3,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PipelineStage {
    InputRead,
    Parse,
    Preprocess,
    FeatureEncode,
    HostDeviceTransfer,
    ModelInference,
    ModelTraining,
    OutputDecode,
    ModelPostprocess,
    Validation,
    CheckpointWrite,
    OutputWrite,
}

impl PipelineStage {
    fn code(self) -> u8 {
        match self {
            Self::InputRead => 1,
            Self::Parse => 2,
            Self::Preprocess => 3,
            Self::FeatureEncode => 4,
            Self::HostDeviceTransfer => 5,
            Self::ModelInference => 6,
            Self::ModelTraining => 7,
            Self::OutputDecode => 8,
            Self::ModelPostprocess => 9,
            Self::Validation => 10,
            Self::CheckpointWrite => 11,
            Self::OutputWrite => 12,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MeasurementScope {
    ModelOnly,
    EndToEndInference,
    TrainingPipeline,
}

impl MeasurementScope {
    fn code(self) -> u8 {
        match self {
            Self::ModelOnly => 1,
            Self::EndToEndInference => 2,
            Self::TrainingPipeline => 3,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResourceBoundary {
    pub scope: MeasurementScope,
    /// Set-like stage collection; canonical digest sorts stage codes.
    pub stages: Vec<PipelineStage>,
}

impl ResourceBoundary {
    pub fn validate(&self) -> Result<(), ManifestError> {
        if self.stages.is_empty() {
            return Err(ManifestError::EmptyResourceBoundary);
        }
        let mut seen = BTreeSet::new();
        for stage in &self.stages {
            if !seen.insert(stage.code()) {
                return Err(ManifestError::DuplicatePipelineStage);
            }
        }
        let has = |required: PipelineStage| self.stages.contains(&required);
        match self.scope {
            MeasurementScope::ModelOnly => {
                if self.stages.len() != 1 || !has(PipelineStage::ModelInference) {
                    return Err(ManifestError::InvalidResourceBoundary);
                }
            }
            MeasurementScope::EndToEndInference => {
                let required = [
                    PipelineStage::InputRead,
                    PipelineStage::Parse,
                    PipelineStage::Preprocess,
                    PipelineStage::FeatureEncode,
                    PipelineStage::ModelInference,
                    PipelineStage::OutputDecode,
                    PipelineStage::OutputWrite,
                ];
                if required.iter().any(|stage| !has(*stage)) {
                    return Err(ManifestError::InvalidResourceBoundary);
                }
            }
            MeasurementScope::TrainingPipeline => {
                let required = [
                    PipelineStage::InputRead,
                    PipelineStage::Parse,
                    PipelineStage::Preprocess,
                    PipelineStage::FeatureEncode,
                    PipelineStage::ModelTraining,
                    PipelineStage::Validation,
                    PipelineStage::CheckpointWrite,
                ];
                if required.iter().any(|stage| !has(*stage)) {
                    return Err(ManifestError::InvalidResourceBoundary);
                }
            }
        }
        Ok(())
    }

    pub fn sha256(&self) -> Result<[u8; 32], ManifestError> {
        self.validate()?;
        let mut h = Sha256::new();
        h.update(b"SYMTHAEA-BENCHMARK-RESOURCE-BOUNDARY-V1\0");
        h.update([self.scope.code()]);
        let mut stages: Vec<u8> = self.stages.iter().map(|stage| stage.code()).collect();
        stages.sort_unstable();
        h.update(u32::try_from(stages.len()).map_err(|_| ManifestError::ArithmeticOverflow)?.to_le_bytes());
        h.update(stages);
        Ok(h.finalize().into())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DatasetIdentity {
    pub repository_url: String,
    pub repository_commit_sha1: [u8; 20],
    pub benchmark_dataset_id: String,
    pub dataset_revision: String,
    pub entity_id: String,
    pub license_or_access_policy: String,
    pub access_class: DatasetAccessClass,
    /// Digest of the exact source file or deterministic source-file bundle.
    pub source_artifact_sha256: [u8; 32],
    pub record_count: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SplitIdentity {
    pub split_policy_sha256: [u8; 32],
    pub train_membership_sha256: [u8; 32],
    pub validation_membership_sha256: [u8; 32],
    pub test_membership_sha256: [u8; 32],
    /// Digest of an empty set is still nonzero and must be recorded explicitly.
    pub purged_membership_sha256: [u8; 32],
    /// Present when reproducing an externally defined official split.
    pub official_split_source_sha256: Option<[u8; 32]>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreprocessingIdentity {
    pub adapter_config_sha256: [u8; 32],
    pub preprocessing_pipeline_sha256: [u8; 32],
    pub scaler_fit_scope: ScalerFitScope,
    pub scaler_state_sha256: Option<[u8; 32]>,
    /// Must equal train_membership_sha256 for TrainOnly.
    /// For FullDataset or ExternalPretrained this binds the actual fitting data.
    pub scaler_fit_data_sha256: Option<[u8; 32]>,
    pub missingness_policy: MissingnessPolicy,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelIdentity {
    pub repository_url: String,
    pub repository_commit_sha1: [u8; 20],
    pub model_id: String,
    pub architecture_sha256: [u8; 32],
    pub model_config_sha256: [u8; 32],
    pub weights_or_checkpoint_sha256: [u8; 32],
    pub tuning_budget_sha256: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExecutionEnvironment {
    pub operating_system: String,
    pub kernel_or_runtime: String,
    pub compiler_or_interpreter: String,
    pub target_triple: String,
    pub cpu_model: String,
    pub accelerator_backend: String,
    pub gpu_model: Option<String>,
    pub accelerator_driver: Option<String>,
    pub environment_sha256: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LatencyMeasurement {
    pub boundary: ResourceBoundary,
    pub warmup_count: u64,
    pub measured_sample_count: u64,
    pub p50_us: u64,
    pub p95_us: u64,
    pub p99_us: u64,
    /// Fixed point: requests per second multiplied by 1,000.
    pub throughput_milli_per_second: u64,
    pub deadline_us: Option<u64>,
    pub deadline_misses: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EnergyMethod {
    CalibratedExternalMeter,
    DeviceTelemetry,
    Estimate,
}

impl EnergyMethod {
    fn code(self) -> u8 {
        match self {
            Self::CalibratedExternalMeter => 1,
            Self::DeviceTelemetry => 2,
            Self::Estimate => 3,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EnergyMeasurement {
    pub boundary: ResourceBoundary,
    pub method: EnergyMethod,
    /// Calibration certificate, telemetry schema, or estimator identity,
    /// depending on method. It is not interchangeable across methods.
    pub evidence_sha256: [u8; 32],
    pub calibration_sha256: Option<[u8; 32]>,
    pub measured_sample_count: u64,
    pub gross_energy_nj: u64,
    pub idle_baseline_energy_nj: u64,
    pub net_energy_nj: u64,
}

impl EnergyMeasurement {
    pub fn is_calibrated_external_meter(&self) -> bool {
        self.method == EnergyMethod::CalibratedExternalMeter
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResultArtifacts {
    pub predictions_sha256: [u8; 32],
    pub metric_report_sha256: [u8; 32],
    pub run_report_sha256: [u8; 32],
    pub stdout_stderr_sha256: Option<[u8; 32]>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BenchmarkRunManifest {
    pub manifest_version: u16,
    pub run_id: u64,
    pub benchmark_id: String,
    pub benchmark_version: String,
    pub track_id: String,
    pub task_id: String,
    pub horizon_id: String,
    pub protocol_mode: ProtocolMode,
    pub random_seed: u64,
    pub dataset: DatasetIdentity,
    pub split: SplitIdentity,
    pub preprocessing: PreprocessingIdentity,
    pub model: ModelIdentity,
    pub environment: ExecutionEnvironment,
    pub latency: LatencyMeasurement,
    /// None means energy was not measured for this run.
    pub energy: Option<EnergyMeasurement>,
    pub artifacts: ResultArtifacts,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ManifestError {
    UnsupportedManifestVersion,
    ZeroRunId,
    EmptyIdentity,
    ZeroCommit,
    ZeroDigest,
    EmptyDataset,
    MissingOfficialSplitSource,
    LeakageProtocolUsesFullDatasetScaler,
    InvalidScalerBinding,
    InvalidMissingnessPolicy,
    EmptyResourceBoundary,
    DuplicatePipelineStage,
    InvalidResourceBoundary,
    InvalidLatencyStatistics,
    InvalidEnergyStatistics,
    InvalidEnergyEvidence,
    ArithmeticOverflow,
}

impl fmt::Display for ManifestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedManifestVersion => write!(f, "unsupported benchmark manifest version"),
            Self::ZeroRunId => write!(f, "run ID must be nonzero"),
            Self::EmptyIdentity => write!(f, "manifest identity strings must be nonempty and trimmed"),
            Self::ZeroCommit => write!(f, "repository commit SHA-1 must be nonzero"),
            Self::ZeroDigest => write!(f, "required SHA-256 digest must be nonzero"),
            Self::EmptyDataset => write!(f, "dataset record count must be nonzero"),
            Self::MissingOfficialSplitSource => write!(f, "official reproduction must bind the official split source"),
            Self::LeakageProtocolUsesFullDatasetScaler => write!(f, "leakage-controlled protocol cannot use a full-dataset fitted scaler"),
            Self::InvalidScalerBinding => write!(f, "scaler state and its fitting-data membership digest are inconsistent"),
            Self::InvalidMissingnessPolicy => write!(f, "imputation requires a nonzero imputer identity digest"),
            Self::EmptyResourceBoundary => write!(f, "resource boundary must contain at least one stage"),
            Self::DuplicatePipelineStage => write!(f, "resource boundary contains a duplicate stage"),
            Self::InvalidResourceBoundary => write!(f, "resource scope is inconsistent with its declared pipeline stages"),
            Self::InvalidLatencyStatistics => write!(f, "latency samples, quantiles, throughput or deadline counts are invalid"),
            Self::InvalidEnergyStatistics => write!(f, "energy sample count or gross/idle/net arithmetic is invalid"),
            Self::InvalidEnergyEvidence => write!(f, "energy method evidence/calibration fields are inconsistent"),
            Self::ArithmeticOverflow => write!(f, "manifest canonical encoding overflow"),
        }
    }
}
impl std::error::Error for ManifestError {}

fn valid_text(value: &str) -> bool {
    !value.is_empty() && value.trim() == value
}

fn nonzero<const N: usize>(value: &[u8; N]) -> bool {
    *value != [0; N]
}

fn require_text(value: &str) -> Result<(), ManifestError> {
    if valid_text(value) { Ok(()) } else { Err(ManifestError::EmptyIdentity) }
}

fn require_digest(value: &[u8; 32]) -> Result<(), ManifestError> {
    if nonzero(value) { Ok(()) } else { Err(ManifestError::ZeroDigest) }
}

fn require_commit(value: &[u8; 20]) -> Result<(), ManifestError> {
    if nonzero(value) { Ok(()) } else { Err(ManifestError::ZeroCommit) }
}

impl BenchmarkRunManifest {
    pub fn validate(&self) -> Result<(), ManifestError> {
        if self.manifest_version != 1 {
            return Err(ManifestError::UnsupportedManifestVersion);
        }
        if self.run_id == 0 {
            return Err(ManifestError::ZeroRunId);
        }
        for value in [
            self.benchmark_id.as_str(),
            self.benchmark_version.as_str(),
            self.track_id.as_str(),
            self.task_id.as_str(),
            self.horizon_id.as_str(),
            self.dataset.repository_url.as_str(),
            self.dataset.benchmark_dataset_id.as_str(),
            self.dataset.dataset_revision.as_str(),
            self.dataset.entity_id.as_str(),
            self.dataset.license_or_access_policy.as_str(),
            self.model.repository_url.as_str(),
            self.model.model_id.as_str(),
            self.environment.operating_system.as_str(),
            self.environment.kernel_or_runtime.as_str(),
            self.environment.compiler_or_interpreter.as_str(),
            self.environment.target_triple.as_str(),
            self.environment.cpu_model.as_str(),
            self.environment.accelerator_backend.as_str(),
        ] {
            require_text(value)?;
        }
        for optional in [
            self.environment.gpu_model.as_deref(),
            self.environment.accelerator_driver.as_deref(),
        ].into_iter().flatten() {
            require_text(optional)?;
        }
        require_commit(&self.dataset.repository_commit_sha1)?;
        require_commit(&self.model.repository_commit_sha1)?;
        if self.dataset.record_count == 0 {
            return Err(ManifestError::EmptyDataset);
        }
        require_digest(&self.dataset.source_artifact_sha256)?;
        for digest in [
            &self.split.split_policy_sha256,
            &self.split.train_membership_sha256,
            &self.split.validation_membership_sha256,
            &self.split.test_membership_sha256,
            &self.split.purged_membership_sha256,
            &self.preprocessing.adapter_config_sha256,
            &self.preprocessing.preprocessing_pipeline_sha256,
            &self.model.architecture_sha256,
            &self.model.model_config_sha256,
            &self.model.weights_or_checkpoint_sha256,
            &self.model.tuning_budget_sha256,
            &self.environment.environment_sha256,
            &self.artifacts.predictions_sha256,
            &self.artifacts.metric_report_sha256,
            &self.artifacts.run_report_sha256,
        ] {
            require_digest(digest)?;
        }
        if let Some(official) = &self.split.official_split_source_sha256 {
            require_digest(official)?;
        }
        if self.protocol_mode == ProtocolMode::OfficialReproduction
            && self.split.official_split_source_sha256.is_none()
        {
            return Err(ManifestError::MissingOfficialSplitSource);
        }
        match self.preprocessing.scaler_fit_scope {
            ScalerFitScope::NotFitted => {
                if self.preprocessing.scaler_state_sha256.is_some()
                    || self.preprocessing.scaler_fit_data_sha256.is_some()
                {
                    return Err(ManifestError::InvalidScalerBinding);
                }
            }
            ScalerFitScope::TrainOnly => {
                let state = self.preprocessing.scaler_state_sha256
                    .as_ref().ok_or(ManifestError::InvalidScalerBinding)?;
                let fit_data = self.preprocessing.scaler_fit_data_sha256
                    .as_ref().ok_or(ManifestError::InvalidScalerBinding)?;
                require_digest(state)?;
                require_digest(fit_data)?;
                if fit_data != &self.split.train_membership_sha256 {
                    return Err(ManifestError::InvalidScalerBinding);
                }
            }
            ScalerFitScope::FullDataset | ScalerFitScope::ExternalPretrained => {
                let state = self.preprocessing.scaler_state_sha256
                    .as_ref().ok_or(ManifestError::InvalidScalerBinding)?;
                let fit_data = self.preprocessing.scaler_fit_data_sha256
                    .as_ref().ok_or(ManifestError::InvalidScalerBinding)?;
                require_digest(state)?;
                require_digest(fit_data)?;
            }
        }
        if self.protocol_mode == ProtocolMode::LeakageControlled
            && self.preprocessing.scaler_fit_scope == ScalerFitScope::FullDataset
        {
            return Err(ManifestError::LeakageProtocolUsesFullDatasetScaler);
        }
        if let MissingnessPolicy::Impute { imputer_sha256 } = self.preprocessing.missingness_policy {
            require_digest(&imputer_sha256)?;
        }
        self.latency.boundary.validate()?;
        if self.latency.boundary.scope == MeasurementScope::TrainingPipeline
            || self.latency.measured_sample_count == 0
            || self.latency.p50_us == 0
            || self.latency.p50_us > self.latency.p95_us
            || self.latency.p95_us > self.latency.p99_us
            || self.latency.throughput_milli_per_second == 0
            || self.latency.deadline_misses > self.latency.measured_sample_count
            || (self.latency.deadline_us.is_none() && self.latency.deadline_misses != 0)
        {
            return Err(ManifestError::InvalidLatencyStatistics);
        }
        if let Some(energy) = &self.energy {
            energy.boundary.validate()?;
            require_digest(&energy.evidence_sha256)?;
            if energy.measured_sample_count == 0
                || energy.gross_energy_nj == 0
                || energy.idle_baseline_energy_nj > energy.gross_energy_nj
                || energy.net_energy_nj
                    != energy.gross_energy_nj - energy.idle_baseline_energy_nj
            {
                return Err(ManifestError::InvalidEnergyStatistics);
            }
            match energy.method {
                EnergyMethod::CalibratedExternalMeter => {
                    let calibration = energy.calibration_sha256
                        .as_ref().ok_or(ManifestError::InvalidEnergyEvidence)?;
                    require_digest(calibration)?;
                }
                EnergyMethod::DeviceTelemetry | EnergyMethod::Estimate => {
                    if energy.calibration_sha256.is_some() {
                        return Err(ManifestError::InvalidEnergyEvidence);
                    }
                }
            }
        }
        if let Some(logs) = &self.artifacts.stdout_stderr_sha256 {
            require_digest(logs)?;
        }
        Ok(())
    }

    /// Canonical digest with explicit field order and domain separation.
    /// This is an identity hash, not a digital signature or evaluator attestation.
    pub fn canonical_sha256(&self) -> Result<[u8; 32], ManifestError> {
        self.validate()?;
        let mut h = Sha256::new();
        h.update(b"SYMTHAEA-BENCHMARK-RUN-MANIFEST-V1\0");

        h.update(self.manifest_version.to_le_bytes());
        h.update(self.run_id.to_le_bytes());
        hash_text(&mut h, &self.benchmark_id)?;
        hash_text(&mut h, &self.benchmark_version)?;
        hash_text(&mut h, &self.track_id)?;
        hash_text(&mut h, &self.task_id)?;
        hash_text(&mut h, &self.horizon_id)?;
        h.update([self.protocol_mode.code()]);
        h.update(self.random_seed.to_le_bytes());

        hash_text(&mut h, &self.dataset.repository_url)?;
        h.update(self.dataset.repository_commit_sha1);
        hash_text(&mut h, &self.dataset.benchmark_dataset_id)?;
        hash_text(&mut h, &self.dataset.dataset_revision)?;
        hash_text(&mut h, &self.dataset.entity_id)?;
        hash_text(&mut h, &self.dataset.license_or_access_policy)?;
        h.update([self.dataset.access_class.code()]);
        h.update(self.dataset.source_artifact_sha256);
        h.update(self.dataset.record_count.to_le_bytes());

        h.update(self.split.split_policy_sha256);
        h.update(self.split.train_membership_sha256);
        h.update(self.split.validation_membership_sha256);
        h.update(self.split.test_membership_sha256);
        h.update(self.split.purged_membership_sha256);
        hash_optional_digest(&mut h, self.split.official_split_source_sha256);

        h.update(self.preprocessing.adapter_config_sha256);
        h.update(self.preprocessing.preprocessing_pipeline_sha256);
        h.update([self.preprocessing.scaler_fit_scope.code()]);
        hash_optional_digest(&mut h, self.preprocessing.scaler_state_sha256);
        hash_optional_digest(&mut h, self.preprocessing.scaler_fit_data_sha256);
        match self.preprocessing.missingness_policy {
            MissingnessPolicy::PreserveMask => { h.update([1]); }
            MissingnessPolicy::DropMissing => { h.update([2]); }
            MissingnessPolicy::Impute { imputer_sha256 } => {
                h.update([3]);
                h.update(imputer_sha256);
            }
        }

        hash_text(&mut h, &self.model.repository_url)?;
        h.update(self.model.repository_commit_sha1);
        hash_text(&mut h, &self.model.model_id)?;
        h.update(self.model.architecture_sha256);
        h.update(self.model.model_config_sha256);
        h.update(self.model.weights_or_checkpoint_sha256);
        h.update(self.model.tuning_budget_sha256);

        hash_text(&mut h, &self.environment.operating_system)?;
        hash_text(&mut h, &self.environment.kernel_or_runtime)?;
        hash_text(&mut h, &self.environment.compiler_or_interpreter)?;
        hash_text(&mut h, &self.environment.target_triple)?;
        hash_text(&mut h, &self.environment.cpu_model)?;
        hash_text(&mut h, &self.environment.accelerator_backend)?;
        hash_optional_text(&mut h, self.environment.gpu_model.as_deref())?;
        hash_optional_text(&mut h, self.environment.accelerator_driver.as_deref())?;
        h.update(self.environment.environment_sha256);

        hash_latency(&mut h, &self.latency)?;
        match &self.energy {
            None => { h.update([0]); }
            Some(energy) => {
                h.update([1]);
                hash_boundary(&mut h, &energy.boundary)?;
                h.update([energy.method.code()]);
                h.update(energy.evidence_sha256);
                hash_optional_digest(&mut h, energy.calibration_sha256);
                h.update(energy.measured_sample_count.to_le_bytes());
                h.update(energy.gross_energy_nj.to_le_bytes());
                h.update(energy.idle_baseline_energy_nj.to_le_bytes());
                h.update(energy.net_energy_nj.to_le_bytes());
            }
        }

        h.update(self.artifacts.predictions_sha256);
        h.update(self.artifacts.metric_report_sha256);
        h.update(self.artifacts.run_report_sha256);
        hash_optional_digest(&mut h, self.artifacts.stdout_stderr_sha256);
        Ok(h.finalize().into())
    }
}

fn hash_text(hasher: &mut Sha256, value: &str) -> Result<(), ManifestError> {
    hasher.update(u64::try_from(value.len()).map_err(|_| ManifestError::ArithmeticOverflow)?.to_le_bytes());
    hasher.update(value.as_bytes());
    Ok(())
}

fn hash_optional_text(hasher: &mut Sha256, value: Option<&str>) -> Result<(), ManifestError> {
    match value {
        None => { hasher.update([0]); }
        Some(value) => {
            hasher.update([1]);
            hash_text(hasher, value)?;
        }
    }
    Ok(())
}

fn hash_optional_digest(hasher: &mut Sha256, value: Option<[u8; 32]>) {
    match value {
        None => { hasher.update([0]); }
        Some(digest) => {
            hasher.update([1]);
            hasher.update(digest);
        }
    }
}

fn hash_boundary(hasher: &mut Sha256, boundary: &ResourceBoundary) -> Result<(), ManifestError> {
    boundary.validate()?;
    hasher.update([boundary.scope.code()]);
    let mut stages: Vec<u8> = boundary.stages.iter().map(|stage| stage.code()).collect();
    stages.sort_unstable();
    hasher.update(u32::try_from(stages.len()).map_err(|_| ManifestError::ArithmeticOverflow)?.to_le_bytes());
    hasher.update(stages);
    Ok(())
}

fn hash_latency(hasher: &mut Sha256, latency: &LatencyMeasurement) -> Result<(), ManifestError> {
    hash_boundary(hasher, &latency.boundary)?;
    hasher.update(latency.warmup_count.to_le_bytes());
    hasher.update(latency.measured_sample_count.to_le_bytes());
    hasher.update(latency.p50_us.to_le_bytes());
    hasher.update(latency.p95_us.to_le_bytes());
    hasher.update(latency.p99_us.to_le_bytes());
    hasher.update(latency.throughput_milli_per_second.to_le_bytes());
    match latency.deadline_us {
        None => hasher.update([0]),
        Some(deadline) => {
            hasher.update([1]);
            hasher.update(deadline.to_le_bytes());
        }
    }
    hasher.update(latency.deadline_misses.to_le_bytes());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(value: u8) -> [u8; 32] {
        [value; 32]
    }

    fn commit(value: u8) -> [u8; 20] {
        [value; 20]
    }

    fn boundary(scope: MeasurementScope) -> ResourceBoundary {
        let stages = match scope {
            MeasurementScope::ModelOnly => vec![PipelineStage::ModelInference],
            MeasurementScope::EndToEndInference => vec![
                PipelineStage::InputRead,
                PipelineStage::Parse,
                PipelineStage::Preprocess,
                PipelineStage::FeatureEncode,
                PipelineStage::HostDeviceTransfer,
                PipelineStage::ModelInference,
                PipelineStage::OutputDecode,
                PipelineStage::OutputWrite,
            ],
            MeasurementScope::TrainingPipeline => vec![
                PipelineStage::InputRead,
                PipelineStage::Parse,
                PipelineStage::Preprocess,
                PipelineStage::FeatureEncode,
                PipelineStage::ModelTraining,
                PipelineStage::Validation,
                PipelineStage::CheckpointWrite,
            ],
        };
        ResourceBoundary { scope, stages }
    }

    fn valid_manifest() -> BenchmarkRunManifest {
        BenchmarkRunManifest {
            manifest_version: 1,
            run_id: 1,
            benchmark_id: "Time-IMM".into(),
            benchmark_version: "2025.1".into(),
            track_id: "irregular-forecasting".into(),
            task_id: "EPA-Air".into(),
            horizon_id: "24h".into(),
            protocol_mode: ProtocolMode::LeakageControlled,
            random_seed: 2026,
            dataset: DatasetIdentity {
                repository_url: "https://github.com/blacksnail789521/Time-IMM".into(),
                repository_commit_sha1: commit(1),
                benchmark_dataset_id: "EPA-Air".into(),
                dataset_revision: "Bexar".into(),
                entity_id: "Bexar".into(),
                license_or_access_policy: "repository-license-and-data-card".into(),
                access_class: DatasetAccessClass::Public,
                source_artifact_sha256: digest(2),
                record_count: 4_370,
            },
            split: SplitIdentity {
                split_policy_sha256: digest(3),
                train_membership_sha256: digest(4),
                validation_membership_sha256: digest(5),
                test_membership_sha256: digest(6),
                purged_membership_sha256: digest(7),
                official_split_source_sha256: Some(digest(8)),
            },
            preprocessing: PreprocessingIdentity {
                adapter_config_sha256: digest(9),
                preprocessing_pipeline_sha256: digest(10),
                scaler_fit_scope: ScalerFitScope::TrainOnly,
                scaler_state_sha256: Some(digest(11)),
                scaler_fit_data_sha256: Some(digest(4)),
                missingness_policy: MissingnessPolicy::PreserveMask,
            },
            model: ModelIdentity {
                repository_url: "https://github.com/Luminous-Dynamics/symthaea".into(),
                repository_commit_sha1: commit(12),
                model_id: "Symthaea-CfC-HDC".into(),
                architecture_sha256: digest(13),
                model_config_sha256: digest(14),
                weights_or_checkpoint_sha256: digest(15),
                tuning_budget_sha256: digest(16),
            },
            environment: ExecutionEnvironment {
                operating_system: "NixOS".into(),
                kernel_or_runtime: "Linux 6.x".into(),
                compiler_or_interpreter: "rustc 1.x".into(),
                target_triple: "x86_64-unknown-linux-gnu".into(),
                cpu_model: "CPU model pinned by run record".into(),
                accelerator_backend: "CPU".into(),
                gpu_model: None,
                accelerator_driver: None,
                environment_sha256: digest(17),
            },
            latency: LatencyMeasurement {
                boundary: boundary(MeasurementScope::EndToEndInference),
                warmup_count: 20,
                measured_sample_count: 100,
                p50_us: 1_000,
                p95_us: 1_500,
                p99_us: 2_000,
                throughput_milli_per_second: 50_000,
                deadline_us: Some(3_000),
                deadline_misses: 0,
            },
            energy: None,
            artifacts: ResultArtifacts {
                predictions_sha256: digest(18),
                metric_report_sha256: digest(19),
                run_report_sha256: digest(20),
                stdout_stderr_sha256: Some(digest(21)),
            },
        }
    }

    #[test]
    fn valid_manifest_has_replay_deterministic_digest() {
        let manifest = valid_manifest();
        assert_eq!(manifest.canonical_sha256().unwrap(), manifest.canonical_sha256().unwrap());
    }

    #[test]
    fn canonical_boundary_digest_is_invariant_to_stage_order() {
        let mut left = boundary(MeasurementScope::EndToEndInference);
        let mut right = left.clone();
        right.stages.reverse();
        assert_eq!(left.sha256().unwrap(), right.sha256().unwrap());
        left.stages.push(PipelineStage::ModelInference);
        assert_eq!(left.sha256(), Err(ManifestError::DuplicatePipelineStage));
    }

    #[test]
    fn changing_data_split_or_model_identity_changes_manifest_digest() {
        let base = valid_manifest().canonical_sha256().unwrap();
        let mut changed = valid_manifest();
        changed.dataset.source_artifact_sha256 = digest(22);
        assert_ne!(base, changed.canonical_sha256().unwrap());
        let mut changed = valid_manifest();
        changed.split.test_membership_sha256 = digest(23);
        assert_ne!(base, changed.canonical_sha256().unwrap());
        let mut changed = valid_manifest();
        changed.model.repository_commit_sha1 = commit(24);
        assert_ne!(base, changed.canonical_sha256().unwrap());
    }

    #[test]
    fn leakage_controlled_protocol_rejects_full_dataset_scaler() {
        let mut manifest = valid_manifest();
        manifest.preprocessing.scaler_fit_scope = ScalerFitScope::FullDataset;
        manifest.preprocessing.scaler_fit_data_sha256 = Some(manifest.dataset.source_artifact_sha256);
        assert_eq!(
            manifest.canonical_sha256(),
            Err(ManifestError::LeakageProtocolUsesFullDatasetScaler)
        );
    }

    #[test]
    fn train_only_scaler_must_bind_exact_train_membership() {
        let mut manifest = valid_manifest();
        manifest.preprocessing.scaler_fit_data_sha256 = Some(digest(42));
        assert_eq!(manifest.validate(), Err(ManifestError::InvalidScalerBinding));
    }

    #[test]
    fn unfitted_scaler_must_not_carry_fitted_state() {
        let mut manifest = valid_manifest();
        manifest.preprocessing.scaler_fit_scope = ScalerFitScope::NotFitted;
        manifest.preprocessing.scaler_state_sha256 = None;
        manifest.preprocessing.scaler_fit_data_sha256 = None;
        assert_eq!(manifest.validate(), Ok(()));
        manifest.preprocessing.scaler_state_sha256 = Some(digest(9));
        assert_eq!(manifest.validate(), Err(ManifestError::InvalidScalerBinding));
    }

    #[test]
    fn official_reproduction_requires_official_split_provenance() {
        let mut manifest = valid_manifest();
        manifest.protocol_mode = ProtocolMode::OfficialReproduction;
        manifest.split.official_split_source_sha256 = None;
        assert_eq!(manifest.validate(), Err(ManifestError::MissingOfficialSplitSource));
    }

    #[test]
    fn model_only_measurement_cannot_claim_end_to_end_stages() {
        let mut invalid = ResourceBoundary {
            scope: MeasurementScope::ModelOnly,
            stages: vec![PipelineStage::InputRead, PipelineStage::ModelInference],
        };
        assert_eq!(invalid.validate(), Err(ManifestError::InvalidResourceBoundary));
        invalid.stages = vec![PipelineStage::ModelInference];
        assert_eq!(invalid.validate(), Ok(()));
    }

    #[test]
    fn invalid_latency_order_and_unaccounted_deadline_misses_fail() {
        let mut manifest = valid_manifest();
        manifest.latency.p95_us = 900;
        assert_eq!(manifest.validate(), Err(ManifestError::InvalidLatencyStatistics));
        let mut manifest = valid_manifest();
        manifest.latency.deadline_us = None;
        manifest.latency.deadline_misses = 1;
        assert_eq!(manifest.validate(), Err(ManifestError::InvalidLatencyStatistics));
    }

    #[test]
    fn calibrated_meter_requires_calibration_and_net_energy_arithmetic() {
        let mut manifest = valid_manifest();
        manifest.energy = Some(EnergyMeasurement {
            boundary: boundary(MeasurementScope::EndToEndInference),
            method: EnergyMethod::CalibratedExternalMeter,
            evidence_sha256: digest(25),
            calibration_sha256: Some(digest(26)),
            measured_sample_count: 100,
            gross_energy_nj: 10_000,
            idle_baseline_energy_nj: 2_000,
            net_energy_nj: 8_000,
        });
        assert_eq!(manifest.validate(), Ok(()));
        manifest.energy.as_mut().unwrap().calibration_sha256 = None;
        assert_eq!(manifest.validate(), Err(ManifestError::InvalidEnergyEvidence));
        manifest.energy.as_mut().unwrap().calibration_sha256 = Some(digest(26));
        manifest.energy.as_mut().unwrap().net_energy_nj = 7_999;
        assert_eq!(manifest.validate(), Err(ManifestError::InvalidEnergyStatistics));
    }

    #[test]
    fn energy_estimate_is_labeled_and_not_a_metered_result() {
        let mut manifest = valid_manifest();
        manifest.energy = Some(EnergyMeasurement {
            boundary: boundary(MeasurementScope::ModelOnly),
            method: EnergyMethod::Estimate,
            evidence_sha256: digest(27),
            calibration_sha256: None,
            measured_sample_count: 100,
            gross_energy_nj: 1_000,
            idle_baseline_energy_nj: 0,
            net_energy_nj: 1_000,
        });
        assert_eq!(manifest.validate(), Ok(()));
        assert!(!manifest.energy.as_ref().unwrap().is_calibrated_external_meter());
    }

    #[test]
    fn energy_and_latency_boundaries_are_independently_digest_bound() {
        let mut manifest = valid_manifest();
        let first = manifest.canonical_sha256().unwrap();
        manifest.energy = Some(EnergyMeasurement {
            boundary: boundary(MeasurementScope::ModelOnly),
            method: EnergyMethod::DeviceTelemetry,
            evidence_sha256: digest(28),
            calibration_sha256: None,
            measured_sample_count: 100,
            gross_energy_nj: 1_000,
            idle_baseline_energy_nj: 100,
            net_energy_nj: 900,
        });
        let second = manifest.canonical_sha256().unwrap();
        assert_ne!(first, second);
        assert_ne!(
            manifest.latency.boundary.sha256().unwrap(),
            manifest.energy.as_ref().unwrap().boundary.sha256().unwrap()
        );
    }

    #[test]
    fn imputation_policy_binds_a_nonzero_imputer_digest() {
        let mut manifest = valid_manifest();
        manifest.preprocessing.missingness_policy = MissingnessPolicy::Impute { imputer_sha256: [0; 32] };
        assert_eq!(manifest.validate(), Err(ManifestError::ZeroDigest));
        manifest.preprocessing.missingness_policy = MissingnessPolicy::Impute { imputer_sha256: digest(29) };
        let base = manifest.canonical_sha256().unwrap();
        manifest.preprocessing.missingness_policy = MissingnessPolicy::PreserveMask;
        assert_ne!(base, manifest.canonical_sha256().unwrap());
    }

    #[test]
    fn zero_identity_digests_and_untrimmed_labels_fail_closed() {
        let mut manifest = valid_manifest();
        manifest.dataset.repository_commit_sha1 = [0; 20];
        assert_eq!(manifest.validate(), Err(ManifestError::ZeroCommit));
        let mut manifest = valid_manifest();
        manifest.benchmark_id = "Time-IMM ".into();
        assert_eq!(manifest.validate(), Err(ManifestError::EmptyIdentity));
        let mut manifest = valid_manifest();
        manifest.artifacts.metric_report_sha256 = [0; 32];
        assert_eq!(manifest.validate(), Err(ManifestError::ZeroDigest));
    }
}
