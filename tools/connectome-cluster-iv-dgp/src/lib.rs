//! Deterministic clustered instrumental-variable data generator.
//!
//! Structural DGP:
//!   X_ig = Z_ig' * pi_g + u_g + u_ig
//!   Y_ig = beta * X_ig + Z_ig' * gamma_g + alpha_g + epsilon_ig
//!
//! beta is the known structural coefficient. pi_g controls cluster-specific
//! first-stage strength. gamma_g is the direct instrument-to-outcome path.
//! Cluster and unit disturbances can each correlate the first-stage error
//! with the structural outcome error. Instrument columns may be equicorrelated.
//!
//! SplitMix64 plus a Box-Muller transform are implemented locally: no
//! third-party dependencies. CSV values use 17 significant decimal digits.

use core::fmt;

pub const GENERATOR_ALGORITHM: &str = "splitmix64-box-muller-cos-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InstrumentAssignmentLevel {
    /// Each row/assignment unit receives its own instrument vector.
    WithinCluster,
    /// Instrument vector is assigned once per independent cluster and copied
    /// to all rows in that cluster.
    ClusterLevel,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ClusterIvDgpConfig {
    pub scenario_id: String,
    pub seed: u64,
    pub structural_beta: f64,
    pub first_stage_coefficients: Vec<f64>,
    pub direct_effect_coefficients: Vec<f64>,
    /// Optional preregistered sample-RMS direct-effect support g. The generator
    /// reports whether the realized DGP direct-effect RMS lies inside this bound.
    pub registered_direct_effect_rms_bound: Option<f64>,
    pub cluster_sizes: Vec<usize>,
    pub instrument_assignment_level: InstrumentAssignmentLevel,
    /// Empty means one for each cluster; otherwise one multiplier per cluster.
    pub first_stage_cluster_multipliers: Vec<f64>,
    /// Empty means one for each cluster; otherwise one multiplier per cluster.
    pub direct_effect_cluster_multipliers: Vec<f64>,
    /// Empty means one for each cluster; otherwise one multiplier per cluster.
    pub cluster_noise_multipliers: Vec<f64>,
    /// Equicorrelation of instrument columns in [0, 1).
    pub instrument_equicorrelation: f64,
    pub cluster_shock_sd: f64,
    pub idiosyncratic_shock_sd: f64,
    /// Correlation of cluster-level treatment and outcome shocks in [-1, 1].
    pub cluster_endogeneity_correlation: f64,
    /// Correlation of unit-level treatment and outcome shocks in [-1, 1].
    pub idiosyncratic_endogeneity_correlation: f64,
}

impl ClusterIvDgpConfig {
    /// Balanced 12-cluster example with 3 correlated instruments and nonzero
    /// direct effects. Cluster multipliers deliberately vary in strength.
    pub fn example() -> Self {
        Self {
            scenario_id: "cluster-iv-baseline-correlated-instruments-direct-effects-v1".to_owned(),
            seed: 20_261_101,
            structural_beta: 0.7,
            first_stage_coefficients: vec![0.65, 0.30, 0.15],
            direct_effect_coefficients: vec![0.08, -0.05, 0.03],
            registered_direct_effect_rms_bound: Some(0.25),
            cluster_sizes: vec![8; 12],
            instrument_assignment_level: InstrumentAssignmentLevel::WithinCluster,
            first_stage_cluster_multipliers: vec![
                1.0, 1.1, 0.9, 0.75, 1.2, 0.8, 1.05, 0.65, 1.15, 0.95, 0.7, 1.25,
            ],
            direct_effect_cluster_multipliers: vec![
                1.0, 1.0, 1.1, 0.9, 1.0, 1.15, 0.85, 1.0, 0.95, 1.05, 1.0, 0.9,
            ],
            cluster_noise_multipliers: vec![
                1.0, 0.9, 1.1, 1.2, 0.8, 1.0, 1.15, 0.85, 1.05, 0.95, 1.25, 0.9,
            ],
            instrument_equicorrelation: 0.35,
            cluster_shock_sd: 0.75,
            idiosyncratic_shock_sd: 1.0,
            cluster_endogeneity_correlation: 0.55,
            idiosyncratic_endogeneity_correlation: 0.45,
        }
    }

    pub fn instrument_count(&self) -> usize {
        self.first_stage_coefficients.len()
    }

    pub fn cluster_count(&self) -> usize {
        self.cluster_sizes.len()
    }

    pub fn observation_count(&self) -> usize {
        self.cluster_sizes.iter().sum()
    }

    pub fn validate(&self) -> Result<(), DgpError> {
        if self.scenario_id.trim().is_empty() {
            return Err(DgpError::EmptyScenarioId);
        }
        if !self.structural_beta.is_finite() {
            return Err(DgpError::NonFiniteParameter { field: "structural_beta" });
        }
        if self.first_stage_coefficients.is_empty() {
            return Err(DgpError::EmptyInstrumentSet);
        }
        if let Some(bound) = self.registered_direct_effect_rms_bound {
            if !bound.is_finite() || bound < 0.0 {
                return Err(DgpError::InvalidRegisteredDirectEffectBound { value: bound });
            }
        }
        if self.first_stage_coefficients.len() != self.direct_effect_coefficients.len() {
            return Err(DgpError::ParameterDimensionMismatch {
                field: "direct_effect_coefficients",
                expected: self.first_stage_coefficients.len(),
                actual: self.direct_effect_coefficients.len(),
            });
        }
        for (index, value) in self.first_stage_coefficients.iter().enumerate() {
            if !value.is_finite() {
                return Err(DgpError::NonFiniteVectorParameter { field: "first_stage_coefficients", index });
            }
        }
        for (index, value) in self.direct_effect_coefficients.iter().enumerate() {
            if !value.is_finite() {
                return Err(DgpError::NonFiniteVectorParameter { field: "direct_effect_coefficients", index });
            }
        }
        if self.cluster_sizes.len() < 2 {
            return Err(DgpError::TooFewClusters { minimum: 2, actual: self.cluster_sizes.len() });
        }
        for (index, size) in self.cluster_sizes.iter().enumerate() {
            if *size == 0 {
                return Err(DgpError::EmptyCluster { cluster_index: index });
            }
        }
        for (field, values) in [
            ("first_stage_cluster_multipliers", &self.first_stage_cluster_multipliers),
            ("direct_effect_cluster_multipliers", &self.direct_effect_cluster_multipliers),
            ("cluster_noise_multipliers", &self.cluster_noise_multipliers),
        ] {
            if !values.is_empty() && values.len() != self.cluster_count() {
                return Err(DgpError::ClusterMultiplierDimensionMismatch {
                    field, expected: self.cluster_count(), actual: values.len(),
                });
            }
            for (index, value) in values.iter().enumerate() {
                if !value.is_finite() || *value < 0.0 {
                    return Err(DgpError::InvalidClusterMultiplier {
                        field, cluster_index: index, value: *value,
                    });
                }
            }
        }
        if !self.instrument_equicorrelation.is_finite()
            || self.instrument_equicorrelation < 0.0
            || self.instrument_equicorrelation >= 1.0
        {
            return Err(DgpError::InvalidCorrelation {
                field: "instrument_equicorrelation",
                value: self.instrument_equicorrelation,
            });
        }
        for (field, value) in [
            ("cluster_shock_sd", self.cluster_shock_sd),
            ("idiosyncratic_shock_sd", self.idiosyncratic_shock_sd),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(DgpError::InvalidScale { field, value });
            }
        }
        for (field, value) in [
            ("cluster_endogeneity_correlation", self.cluster_endogeneity_correlation),
            ("idiosyncratic_endogeneity_correlation", self.idiosyncratic_endogeneity_correlation),
        ] {
            if !value.is_finite() || !(-1.0..=1.0).contains(&value) {
                return Err(DgpError::InvalidCorrelation { field, value });
            }
        }
        if self.observation_count() < self.instrument_count() + 2 {
            return Err(DgpError::TooFewObservations {
                minimum: self.instrument_count() + 2,
                actual: self.observation_count(),
            });
        }
        Ok(())
    }

    fn multiplier(values: &[f64], cluster_index: usize) -> f64 {
        if values.is_empty() { 1.0 } else { values[cluster_index] }
    }

    fn cluster_pi(&self, cluster_index: usize) -> Vec<f64> {
        let m = Self::multiplier(&self.first_stage_cluster_multipliers, cluster_index);
        self.first_stage_coefficients.iter().map(|value| value * m).collect()
    }

    fn cluster_gamma(&self, cluster_index: usize) -> Vec<f64> {
        let m = Self::multiplier(&self.direct_effect_cluster_multipliers, cluster_index);
        self.direct_effect_coefficients.iter().map(|value| value * m).collect()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct GeneratedObservation {
    pub unit_id: String,
    pub cluster_id: String,
    pub instruments: Vec<f64>,
    pub treatment: f64,
    pub outcome: f64,
    pub first_stage_instrument_component: f64,
    pub direct_effect_component: f64,
    pub treatment_disturbance: f64,
    pub structural_error: f64,
    pub first_stage_cluster_multiplier: f64,
    pub direct_effect_cluster_multiplier: f64,
    pub cluster_noise_multiplier: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ClusterTruth {
    pub cluster_id: String,
    pub row_count: usize,
    pub first_stage_coefficients: Vec<f64>,
    pub direct_effect_coefficients: Vec<f64>,
    pub treatment_cluster_shock: f64,
    pub outcome_cluster_shock: f64,
    pub cluster_noise_multiplier: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ClusterIvDgpDataset {
    pub config: ClusterIvDgpConfig,
    pub rows: Vec<GeneratedObservation>,
    pub cluster_truth: Vec<ClusterTruth>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DgpSummary {
    pub observation_count: usize,
    pub cluster_count: usize,
    pub instrument_count: usize,
    pub structural_beta: f64,
    /// sqrt(mean_i((Z_i' gamma_cluster(i))^2)).
    pub direct_effect_sample_rms: f64,
    pub registered_direct_effect_rms_bound: Option<f64>,
    pub direct_effect_support_status: &'static str,
    pub instrument_second_moment: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum DgpError {
    EmptyScenarioId,
    EmptyInstrumentSet,
    TooFewClusters { minimum: usize, actual: usize },
    EmptyCluster { cluster_index: usize },
    TooFewObservations { minimum: usize, actual: usize },
    ParameterDimensionMismatch { field: &'static str, expected: usize, actual: usize },
    ClusterMultiplierDimensionMismatch { field: &'static str, expected: usize, actual: usize },
    NonFiniteParameter { field: &'static str },
    NonFiniteVectorParameter { field: &'static str, index: usize },
    InvalidClusterMultiplier { field: &'static str, cluster_index: usize, value: f64 },
    InvalidScale { field: &'static str, value: f64 },
    InvalidCorrelation { field: &'static str, value: f64 },
    InvalidRegisteredDirectEffectBound { value: f64 },
    NumericalOverflow { field: &'static str },
}

impl fmt::Display for DgpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyScenarioId => write!(f, "scenario id must not be empty"),
            Self::EmptyInstrumentSet => write!(f, "at least one instrument coefficient is required"),
            Self::TooFewClusters { minimum, actual } => write!(f, "at least {minimum} clusters are required; received {actual}"),
            Self::EmptyCluster { cluster_index } => write!(f, "cluster {cluster_index} has zero rows"),
            Self::TooFewObservations { minimum, actual } => write!(f, "at least {minimum} observations are required; received {actual}"),
            Self::ParameterDimensionMismatch { field, expected, actual } => write!(f, "{field} has dimension {actual}; expected {expected}"),
            Self::ClusterMultiplierDimensionMismatch { field, expected, actual } => write!(f, "{field} has length {actual}; expected {expected} or zero for default ones"),
            Self::NonFiniteParameter { field } => write!(f, "{field} is non-finite"),
            Self::NonFiniteVectorParameter { field, index } => write!(f, "{field}[{index}] is non-finite"),
            Self::InvalidClusterMultiplier { field, cluster_index, value } => write!(f, "{field}[{cluster_index}] is invalid: {value}"),
            Self::InvalidScale { field, value } => write!(f, "{field} must be finite and non-negative: {value}"),
            Self::InvalidCorrelation { field, value } => write!(f, "{field} is outside its supported correlation range: {value}"),
            Self::InvalidRegisteredDirectEffectBound { value } => write!(f, "registered direct-effect RMS bound must be finite and non-negative: {value}"),
            Self::NumericalOverflow { field } => write!(f, "numerical overflow while generating {field}"),
        }
    }
}
impl std::error::Error for DgpError {}

#[derive(Debug, Clone)]
struct SplitMix64 {
    state: u64,
}
impl SplitMix64 {
    fn new(seed: u64) -> Self { Self { state: seed } }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }
    fn uniform_open01(&mut self) -> f64 {
        let mantissa = self.next_u64() >> 11;
        (mantissa as f64 + 0.5) / ((1_u64 << 53) as f64)
    }
    fn standard_normal(&mut self) -> f64 {
        let u1 = self.uniform_open01();
        let u2 = self.uniform_open01();
        (-2.0 * u1.ln()).sqrt() * (core::f64::consts::TAU * u2).cos()
    }
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

fn json_string(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 2);
    out.push('"');
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}
fn json_number(value: f64) -> String {
    if value == 0.0 { "0.0".to_owned() } else { format!("{:.17e}", value) }
}
fn json_optional_number(value: Option<f64>) -> String {
    value.map(json_number).unwrap_or_else(|| "null".to_owned())
}
fn json_float_array(values: &[f64]) -> String {
    format!("[{}]", values.iter().map(|x| json_number(*x)).collect::<Vec<_>>().join(","))
}
fn csv_number(value: f64) -> String {
    format!("{:.17e}", value)
}

impl ClusterIvDgpDataset {
    pub fn summary(&self) -> DgpSummary {
        let n = self.rows.len();
        let r = self.config.instrument_count();
        let direct_sq = self.rows.iter()
            .map(|row| row.direct_effect_component * row.direct_effect_component)
            .sum::<f64>();
        let z_sq = self.rows.iter()
            .flat_map(|row| row.instruments.iter().map(|z| z * z))
            .sum::<f64>();
        let direct_effect_sample_rms = (direct_sq / n as f64).sqrt();
        let direct_effect_support_status = match self.config.registered_direct_effect_rms_bound {
            None => "UNREGISTERED",
            Some(bound) if direct_effect_sample_rms <= bound => "IN_BOUND",
            Some(_) => "OUT_OF_BOUND",
        };
        DgpSummary {
            observation_count: n,
            cluster_count: self.config.cluster_count(),
            instrument_count: r,
            structural_beta: self.config.structural_beta,
            direct_effect_sample_rms,
            registered_direct_effect_rms_bound: self.config.registered_direct_effect_rms_bound,
            direct_effect_support_status,
            instrument_second_moment: z_sq / (n * r) as f64,
        }
    }

    /// Stable CSV, one row per assignment/observation unit; cluster ID remains
    /// explicit for one-way cluster estimators.
    pub fn to_csv(&self) -> String {
        let mut out = String::new();
        out.push_str("unit_id,cluster_id");
        for j in 0..self.config.instrument_count() { out.push_str(&format!(",z{j}")); }
        out.push_str(",treatment,outcome,first_stage_instrument_component,direct_effect_component,treatment_disturbance,structural_error,first_stage_cluster_multiplier,direct_effect_cluster_multiplier,cluster_noise_multiplier\n");
        for row in &self.rows {
            out.push_str(&row.unit_id);
            out.push(',');
            out.push_str(&row.cluster_id);
            for z in &row.instruments {
                out.push(',');
                out.push_str(&csv_number(*z));
            }
            for value in [
                row.treatment,
                row.outcome,
                row.first_stage_instrument_component,
                row.direct_effect_component,
                row.treatment_disturbance,
                row.structural_error,
                row.first_stage_cluster_multiplier,
                row.direct_effect_cluster_multiplier,
                row.cluster_noise_multiplier,
            ] {
                out.push(',');
                out.push_str(&csv_number(value));
            }
            out.push('\n');
        }
        out
    }

    /// Machine-readable manifest; consumers SHA-256 hash the CSV and manifest
    /// separately before unblinding. The manifest contains declared and
    /// realized cluster truth, not inference results.
    pub fn manifest_json(&self) -> String {
        let c = &self.config;
        let s = self.summary();
        let truth = self.cluster_truth.iter().map(|cluster| {
            format!(
                "{{\"cluster_id\":{},\"row_count\":{},\"first_stage_coefficients\":{},\"direct_effect_coefficients\":{},\"treatment_cluster_shock\":{},\"outcome_cluster_shock\":{},\"cluster_noise_multiplier\":{}}}",
                json_string(&cluster.cluster_id),
                cluster.row_count,
                json_float_array(&cluster.first_stage_coefficients),
                json_float_array(&cluster.direct_effect_coefficients),
                json_number(cluster.treatment_cluster_shock),
                json_number(cluster.outcome_cluster_shock),
                json_number(cluster.cluster_noise_multiplier),
            )
        }).collect::<Vec<_>>().join(",");
        format!(
            concat!(
                "{{\n",
                "  \"schema_version\":\"1.0.0\",\n",
                "  \"scenario_id\":{},\n",
                "  \"generator_algorithm\":{},\n",
                "  \"seed\":{},\n",
                "  \"structural_beta\":{},\n",
                "  \"first_stage_coefficients\":{},\n",
                "  \"direct_effect_coefficients\":{},\n",
                "  \"registered_direct_effect_rms_bound\":{},\n",
                "  \"cluster_sizes\":{},\n",
                "  \"instrument_assignment_level\":{},\n",
                "  \"instrument_assignment_level\":{},\n",
                "  \"first_stage_cluster_multipliers\":{},\n",
                "  \"direct_effect_cluster_multipliers\":{},\n",
                "  \"cluster_noise_multipliers\":{},\n",
                "  \"instrument_equicorrelation\":{},\n",
                "  \"cluster_shock_sd\":{},\n",
                "  \"idiosyncratic_shock_sd\":{},\n",
                "  \"cluster_endogeneity_correlation\":{},\n",
                "  \"idiosyncratic_endogeneity_correlation\":{},\n",
                "  \"summary\":{{\"observation_count\":{},\"cluster_count\":{},\"instrument_count\":{},\"direct_effect_sample_rms\":{},\"registered_direct_effect_rms_bound\":{},\"direct_effect_support_status\":{},\"instrument_second_moment\":{}}},\n",
                "  \"cluster_truth\":[{}],\n",
                "  \"row_order\":\"cluster-major then ascending unit index\",\n",
                "  \"hashing_note\":\"Consumer must SHA-256 hash CSV and manifest before unblinding.\"\n",
                "}}\n"
            ),
            json_string(&c.scenario_id),
            json_string(GENERATOR_ALGORITHM),
            c.seed,
            json_number(c.structural_beta),
            json_float_array(&c.first_stage_coefficients),
            json_float_array(&c.direct_effect_coefficients),
            json_optional_number(c.registered_direct_effect_rms_bound),
            format!("[{}]", c.cluster_sizes.iter().map(usize::to_string).collect::<Vec<_>>().join(",")),
            json_string(match c.instrument_assignment_level { InstrumentAssignmentLevel::WithinCluster => "within_cluster", InstrumentAssignmentLevel::ClusterLevel => "cluster_level" }),
            json_float_array(&c.first_stage_cluster_multipliers),
            json_float_array(&c.direct_effect_cluster_multipliers),
            json_float_array(&c.cluster_noise_multipliers),
            json_number(c.instrument_equicorrelation),
            json_number(c.cluster_shock_sd),
            json_number(c.idiosyncratic_shock_sd),
            json_number(c.cluster_endogeneity_correlation),
            json_number(c.idiosyncratic_endogeneity_correlation),
            s.observation_count,
            s.cluster_count,
            s.instrument_count,
            json_number(s.direct_effect_sample_rms),
            json_optional_number(s.registered_direct_effect_rms_bound),
            json_string(s.direct_effect_support_status),
            json_number(s.instrument_second_moment),
            truth,
        )
    }
}


#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClusterIvScenarioPreset {
    StrongExactExclusion,
    WeakExactExclusion,
    ZeroRelevance,
    DirectEffectInsideSupport,
    DirectEffectOutsideSupport,
    StrengthConcentratedInFewClusters,
    ClusterLevelInstruments,
    UnequalHeteroskedasticClusters,
}

impl ClusterIvScenarioPreset {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::StrongExactExclusion => "strong-exact-exclusion",
            Self::WeakExactExclusion => "weak-exact-exclusion",
            Self::ZeroRelevance => "zero-relevance",
            Self::DirectEffectInsideSupport => "direct-effect-inside-support",
            Self::DirectEffectOutsideSupport => "direct-effect-outside-support",
            Self::StrengthConcentratedInFewClusters => "strength-concentrated-in-few-clusters",
            Self::ClusterLevelInstruments => "cluster-level-instruments",
            Self::UnequalHeteroskedasticClusters => "unequal-heteroskedastic-clusters",
        }
    }
}

/// Return a preregisterable synthetic design configuration. Each preset differs
/// from the common example along an explicit causal/identification axis.
pub fn scenario_preset(preset: ClusterIvScenarioPreset, seed: u64) -> ClusterIvDgpConfig {
    let mut config = ClusterIvDgpConfig::example();
    config.seed = seed;
    config.scenario_id = format!("{}-v1", preset.as_str());
    match preset {
        ClusterIvScenarioPreset::StrongExactExclusion => {
            config.direct_effect_coefficients.fill(0.0);
            config.direct_effect_cluster_multipliers.fill(1.0);
            config.first_stage_cluster_multipliers.fill(1.0);
            config.registered_direct_effect_rms_bound = Some(0.0);
        }
        ClusterIvScenarioPreset::WeakExactExclusion => {
            config.direct_effect_coefficients.fill(0.0);
            config.first_stage_cluster_multipliers.fill(0.08);
            config.registered_direct_effect_rms_bound = Some(0.0);
        }
        ClusterIvScenarioPreset::ZeroRelevance => {
            config.first_stage_coefficients.fill(0.0);
            config.direct_effect_cluster_multipliers.fill(1.0);
            config.registered_direct_effect_rms_bound = Some(0.25);
        }
        ClusterIvScenarioPreset::DirectEffectInsideSupport => {
            config.direct_effect_cluster_multipliers.fill(0.5);
            config.registered_direct_effect_rms_bound = Some(0.25);
        }
        ClusterIvScenarioPreset::DirectEffectOutsideSupport => {
            config.direct_effect_cluster_multipliers.fill(4.0);
            config.registered_direct_effect_rms_bound = Some(0.05);
        }
        ClusterIvScenarioPreset::StrengthConcentratedInFewClusters => {
            config.first_stage_cluster_multipliers = vec![
                3.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ];
        }
        ClusterIvScenarioPreset::ClusterLevelInstruments => {
            config.instrument_assignment_level = InstrumentAssignmentLevel::ClusterLevel;
        }
        ClusterIvScenarioPreset::UnequalHeteroskedasticClusters => {
            config.cluster_sizes = vec![4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26];
            config.cluster_noise_multipliers = vec![
                0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6,
            ];
            config.registered_direct_effect_rms_bound = Some(0.25);
        }
    }
    config
}

/// Generate one deterministic clustered IV dataset with known beta, pi and
/// gamma. Unit-level endogeneity and cluster-level dependence are controlled
/// independently. Instrument columns are equicorrelated with the configured
/// non-negative pairwise correlation.
pub fn generate_cluster_iv_dataset(
    config: ClusterIvDgpConfig,
) -> Result<ClusterIvDgpDataset, DgpError> {
    config.validate()?;
    let mut rng = SplitMix64::new(config.seed);
    let r = config.instrument_count();
    let mut rows = Vec::with_capacity(config.observation_count());
    let mut cluster_truth = Vec::with_capacity(config.cluster_count());
    let corr_z = config.instrument_equicorrelation.sqrt();
    let idio_z = (1.0 - config.instrument_equicorrelation).sqrt();
    let cluster_rho = config.cluster_endogeneity_correlation;
    let cluster_orthogonal = (1.0 - cluster_rho * cluster_rho).max(0.0).sqrt();
    let unit_rho = config.idiosyncratic_endogeneity_correlation;
    let unit_orthogonal = (1.0 - unit_rho * unit_rho).max(0.0).sqrt();
    let mut global_row = 0usize;

    for cluster_index in 0..config.cluster_count() {
        let cluster_id = format!("c{:04}", cluster_index);
        let first_m = ClusterIvDgpConfig::multiplier(&config.first_stage_cluster_multipliers, cluster_index);
        let direct_m = ClusterIvDgpConfig::multiplier(&config.direct_effect_cluster_multipliers, cluster_index);
        let noise_m = ClusterIvDgpConfig::multiplier(&config.cluster_noise_multipliers, cluster_index);

        let cluster_x_standard = rng.standard_normal();
        let cluster_y_independent = rng.standard_normal();
        let cluster_y_standard =
            cluster_rho * cluster_x_standard + cluster_orthogonal * cluster_y_independent;
        let cluster_x_shock = config.cluster_shock_sd * noise_m * cluster_x_standard;
        let cluster_y_shock = config.cluster_shock_sd * noise_m * cluster_y_standard;
        let pi = config.cluster_pi(cluster_index);
        let gamma = config.cluster_gamma(cluster_index);
        let cluster_level_z = if config.instrument_assignment_level == InstrumentAssignmentLevel::ClusterLevel {
            let common_z = rng.standard_normal();
            Some((0..r).map(|_| corr_z * common_z + idio_z * rng.standard_normal()).collect::<Vec<_>>())
        } else {
            None
        };

        cluster_truth.push(ClusterTruth {
            cluster_id: cluster_id.clone(),
            row_count: config.cluster_sizes[cluster_index],
            first_stage_coefficients: pi.clone(),
            direct_effect_coefficients: gamma.clone(),
            treatment_cluster_shock: cluster_x_shock,
            outcome_cluster_shock: cluster_y_shock,
            cluster_noise_multiplier: noise_m,
        });

        for _unit_index in 0..config.cluster_sizes[cluster_index] {
            let instruments: Vec<f64> = match &cluster_level_z {
                Some(shared) => shared.clone(),
                None => {
                    let common_z = rng.standard_normal();
                    (0..r).map(|_| corr_z * common_z + idio_z * rng.standard_normal()).collect()
                }
            };
            let unit_x_standard = rng.standard_normal();
            let unit_y_independent = rng.standard_normal();
            let unit_y_standard = unit_rho * unit_x_standard + unit_orthogonal * unit_y_independent;
            let unit_x_shock = config.idiosyncratic_shock_sd * unit_x_standard;
            let unit_y_shock = config.idiosyncratic_shock_sd * unit_y_standard;

            let first_stage_component = dot(&instruments, &pi);
            let direct_component = dot(&instruments, &gamma);
            let treatment_disturbance = cluster_x_shock + unit_x_shock;
            let structural_error = cluster_y_shock + unit_y_shock;
            let treatment = first_stage_component + treatment_disturbance;
            let outcome = config.structural_beta * treatment + direct_component + structural_error;

            for (field, value) in [
                ("treatment", treatment),
                ("outcome", outcome),
                ("first-stage instrument component", first_stage_component),
                ("direct effect component", direct_component),
                ("treatment disturbance", treatment_disturbance),
                ("structural error", structural_error),
            ] {
                if !value.is_finite() {
                    return Err(DgpError::NumericalOverflow { field });
                }
            }

            rows.push(GeneratedObservation {
                unit_id: format!("u{:08}", global_row),
                cluster_id: cluster_id.clone(),
                instruments,
                treatment,
                outcome,
                first_stage_instrument_component: first_stage_component,
                direct_effect_component: direct_component,
                treatment_disturbance,
                structural_error,
                first_stage_cluster_multiplier: first_m,
                direct_effect_cluster_multiplier: direct_m,
                cluster_noise_multiplier: noise_m,
            });
            global_row += 1;
        }
    }

    Ok(ClusterIvDgpDataset { config, rows, cluster_truth })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tolerance: f64) {
        assert!((a - b).abs() <= tolerance, "actual {a:.16e}, expected {b:.16e}");
    }


    fn small_known_answer_config() -> ClusterIvDgpConfig {
        ClusterIvDgpConfig {
            scenario_id: "small-cross-language-fixture-v1".to_owned(),
            seed: 42,
            structural_beta: 0.75,
            first_stage_coefficients: vec![0.8, 0.35],
            direct_effect_coefficients: vec![0.1, -0.04],
            registered_direct_effect_rms_bound: Some(0.2),
            cluster_sizes: vec![2, 2],
            instrument_assignment_level: InstrumentAssignmentLevel::WithinCluster,
            first_stage_cluster_multipliers: vec![1.0, 0.5],
            direct_effect_cluster_multipliers: vec![1.0, 1.2],
            cluster_noise_multipliers: vec![1.0, 0.8],
            instrument_equicorrelation: 0.25,
            cluster_shock_sd: 0.7,
            idiosyncratic_shock_sd: 0.9,
            cluster_endogeneity_correlation: 0.4,
            idiosyncratic_endogeneity_correlation: 0.5,
        }
    }

    fn base() -> ClusterIvDgpConfig {
        ClusterIvDgpConfig {
            scenario_id: "test-scenario".to_owned(),
            seed: 42,
            structural_beta: 0.75,
            first_stage_coefficients: vec![0.8, 0.35],
            direct_effect_coefficients: vec![0.1, -0.04],
            registered_direct_effect_rms_bound: Some(0.2),
            cluster_sizes: vec![5, 6, 7, 8],
            instrument_assignment_level: InstrumentAssignmentLevel::WithinCluster,
            first_stage_cluster_multipliers: vec![1.0, 0.5, 1.5, 0.0],
            direct_effect_cluster_multipliers: vec![1.0, 0.8, 1.2, 1.0],
            cluster_noise_multipliers: vec![1.0, 0.8, 1.2, 1.5],
            instrument_equicorrelation: 0.25,
            cluster_shock_sd: 0.7,
            idiosyncratic_shock_sd: 0.9,
            cluster_endogeneity_correlation: 0.4,
            idiosyncratic_endogeneity_correlation: 0.5,
        }
    }

    #[test]
    fn same_seed_and_config_reproduce_rows_csv_and_manifest() {
        let config = base();
        let a = generate_cluster_iv_dataset(config.clone()).unwrap();
        let b = generate_cluster_iv_dataset(config).unwrap();
        assert_eq!(a.rows, b.rows);
        assert_eq!(a.to_csv(), b.to_csv());
        assert_eq!(a.manifest_json(), b.manifest_json());
    }

    #[test]
    fn changing_seed_changes_shocks_not_structural_truth() {
        let config = base();
        let a = generate_cluster_iv_dataset(config.clone()).unwrap();
        let mut other = config;
        other.seed += 1;
        let b = generate_cluster_iv_dataset(other).unwrap();
        assert_ne!(a.rows, b.rows);
        assert_eq!(a.config.structural_beta, b.config.structural_beta);
        assert_eq!(a.config.first_stage_coefficients, b.config.first_stage_coefficients);
        assert_eq!(a.config.direct_effect_coefficients, b.config.direct_effect_coefficients);
    }

    #[test]
    fn row_count_and_cluster_sizes_are_preserved() {
        let data = generate_cluster_iv_dataset(base()).unwrap();
        assert_eq!(data.rows.len(), 26);
        assert_eq!(data.cluster_truth.len(), 4);
        assert_eq!(data.cluster_truth.iter().map(|c| c.row_count).collect::<Vec<_>>(), vec![5, 6, 7, 8]);
    }

    #[test]
    fn every_row_satisfies_the_structural_equation() {
        let data = generate_cluster_iv_dataset(base()).unwrap();
        for row in &data.rows {
            close(
                row.outcome,
                data.config.structural_beta * row.treatment
                    + row.direct_effect_component + row.structural_error,
                1.0e-12,
            );
            close(
                row.treatment,
                row.first_stage_instrument_component + row.treatment_disturbance,
                1.0e-12,
            );
        }
    }

    #[test]
    fn first_stage_and_direct_components_match_cluster_truth() {
        let data = generate_cluster_iv_dataset(base()).unwrap();
        for row in &data.rows {
            let index = row.cluster_id[1..].parse::<usize>().unwrap();
            close(row.first_stage_instrument_component, dot(&row.instruments, &data.cluster_truth[index].first_stage_coefficients), 1.0e-12);
            close(row.direct_effect_component, dot(&row.instruments, &data.cluster_truth[index].direct_effect_coefficients), 1.0e-12);
        }
    }

    #[test]
    fn zero_first_stage_multiplier_removes_relevance_in_selected_cluster() {
        let data = generate_cluster_iv_dataset(base()).unwrap();
        for row in data.rows.iter().filter(|row| row.cluster_id == "c0003") {
            close(row.first_stage_instrument_component, 0.0, 0.0);
        }
    }

    #[test]
    fn zero_direct_effect_truth_gives_zero_direct_contribution() {
        let mut config = base();
        config.direct_effect_coefficients.fill(0.0);
        let data = generate_cluster_iv_dataset(config).unwrap();
        assert!(data.rows.iter().all(|row| row.direct_effect_component == 0.0));
    }

    #[test]
    fn direct_effect_cluster_multiplier_scales_cluster_truth() {
        let data = generate_cluster_iv_dataset(base()).unwrap();
        let a = &data.cluster_truth[0].direct_effect_coefficients;
        let c = &data.cluster_truth[2].direct_effect_coefficients;
        close(c[0], a[0] * 1.2, 1.0e-12);
        close(c[1], a[1] * 1.2, 1.0e-12);
    }

    #[test]
    fn endogeneity_correlations_are_registered_and_shocks_are_present() {
        let data = generate_cluster_iv_dataset(base()).unwrap();
        assert_eq!(data.config.cluster_endogeneity_correlation, 0.4);
        assert_eq!(data.config.idiosyncratic_endogeneity_correlation, 0.5);
        assert!(data.rows.iter().any(|row| row.treatment_disturbance != 0.0));
        assert!(data.rows.iter().any(|row| row.structural_error != 0.0));
    }

    #[test]
    fn zero_cluster_and_unit_shock_scales_remove_endogeneity() {
        let mut config = base();
        config.cluster_shock_sd = 0.0;
        config.idiosyncratic_shock_sd = 0.0;
        let data = generate_cluster_iv_dataset(config).unwrap();
        assert!(data.rows.iter().all(|row| row.treatment_disturbance == 0.0));
        assert!(data.rows.iter().all(|row| row.structural_error == 0.0));
    }

    #[test]
    fn zero_instrument_correlation_is_valid() {
        let mut config = base();
        config.instrument_equicorrelation = 0.0;
        assert!(generate_cluster_iv_dataset(config).is_ok());
    }

    #[test]
    fn invalid_instrument_correlation_fails_closed() {
        let mut config = base();
        config.instrument_equicorrelation = 1.0;
        assert!(matches!(generate_cluster_iv_dataset(config), Err(DgpError::InvalidCorrelation { field: "instrument_equicorrelation", .. })));
    }

    #[test]
    fn endogeneity_correlation_endpoints_are_valid() {
        let mut config = base();
        config.cluster_endogeneity_correlation = -1.0;
        config.idiosyncratic_endogeneity_correlation = 1.0;
        assert!(generate_cluster_iv_dataset(config).is_ok());
    }

    #[test]
    fn invalid_endogeneity_correlation_fails_closed() {
        let mut config = base();
        config.idiosyncratic_endogeneity_correlation = 1.01;
        assert!(matches!(generate_cluster_iv_dataset(config), Err(DgpError::InvalidCorrelation { field: "idiosyncratic_endogeneity_correlation", .. })));
    }

    #[test]
    fn multiplier_vectors_must_match_cluster_count() {
        let mut config = base();
        config.first_stage_cluster_multipliers.pop();
        assert!(matches!(generate_cluster_iv_dataset(config), Err(DgpError::ClusterMultiplierDimensionMismatch { field: "first_stage_cluster_multipliers", .. })));
    }

    #[test]
    fn multiplier_values_must_be_finite_and_nonnegative() {
        let mut config = base();
        config.direct_effect_cluster_multipliers[1] = f64::NAN;
        assert!(matches!(generate_cluster_iv_dataset(config), Err(DgpError::InvalidClusterMultiplier { field: "direct_effect_cluster_multipliers", .. })));
    }

    #[test]
    fn empty_cluster_and_too_few_clusters_fail_closed() {
        let mut config = base();
        config.cluster_sizes[1] = 0;
        assert!(matches!(generate_cluster_iv_dataset(config), Err(DgpError::EmptyCluster { cluster_index: 1 })));
        let mut config = base();
        config.cluster_sizes = vec![5];
        config.first_stage_cluster_multipliers = vec![1.0];
        config.direct_effect_cluster_multipliers = vec![1.0];
        config.cluster_noise_multipliers = vec![1.0];
        assert!(matches!(generate_cluster_iv_dataset(config), Err(DgpError::TooFewClusters { .. })));
    }

    #[test]
    fn empty_instrument_and_truth_dimension_mismatch_fail_closed() {
        let mut config = base();
        config.first_stage_coefficients.clear();
        config.direct_effect_coefficients.clear();
        assert_eq!(config.validate(), Err(DgpError::EmptyInstrumentSet));
        let mut config = base();
        config.direct_effect_coefficients.pop();
        assert!(matches!(config.validate(), Err(DgpError::ParameterDimensionMismatch { .. })));
    }

    #[test]
    fn non_finite_truth_and_negative_scale_fail_closed() {
        let mut config = base();
        config.structural_beta = f64::NAN;
        assert_eq!(config.validate(), Err(DgpError::NonFiniteParameter { field: "structural_beta" }));
        let mut config = base();
        config.cluster_shock_sd = -0.1;
        assert!(matches!(config.validate(), Err(DgpError::InvalidScale { field: "cluster_shock_sd", .. })));
    }

    #[test]
    fn csv_has_cluster_instrument_and_truth_columns() {
        let data = generate_cluster_iv_dataset(base()).unwrap();
        let csv = data.to_csv();
        assert!(csv.starts_with("unit_id,cluster_id,z0,z1,treatment,outcome"));
        assert!(csv.contains("direct_effect_component"));
        assert_eq!(csv.lines().count(), data.rows.len() + 1);
    }

    #[test]
    fn manifest_contains_seed_algorithm_truth_and_cluster_details() {
        let data = generate_cluster_iv_dataset(base()).unwrap();
        let json = data.manifest_json();
        assert!(json.contains("\"seed\":42"));
        assert!(json.contains(GENERATOR_ALGORITHM));
        assert!(json.contains("\"structural_beta\":"));
        assert!(json.contains("\"direct_effect_coefficients\":"));
        assert!(json.contains("\"cluster_truth\":["));
        assert!(json.contains("\"row_order\":"));
    }

    #[test]
    fn summary_separates_observations_clusters_and_instruments() {
        let data = generate_cluster_iv_dataset(base()).unwrap();
        let summary = data.summary();
        assert_eq!(summary.observation_count, 26);
        assert_eq!(summary.cluster_count, 4);
        assert_eq!(summary.instrument_count, 2);
        assert_eq!(summary.structural_beta, 0.75);
        assert!(summary.direct_effect_sample_rms > 0.0);
    }

    #[test]
    fn empty_multiplier_vectors_default_to_one() {
        let mut config = base();
        config.first_stage_cluster_multipliers.clear();
        config.direct_effect_cluster_multipliers.clear();
        config.cluster_noise_multipliers.clear();
        let data = generate_cluster_iv_dataset(config).unwrap();
        assert!(data.rows.iter().all(|row| row.first_stage_cluster_multiplier == 1.0));
        assert!(data.rows.iter().all(|row| row.direct_effect_cluster_multiplier == 1.0));
        assert!(data.rows.iter().all(|row| row.cluster_noise_multiplier == 1.0));
    }

    #[test]
    fn ids_follow_cluster_major_row_order() {
        let data = generate_cluster_iv_dataset(base()).unwrap();
        assert_eq!(data.rows[0].unit_id, "u00000000");
        assert_eq!(data.rows[4].cluster_id, "c0000");
        assert_eq!(data.rows[5].unit_id, "u00000005");
        assert_eq!(data.rows[5].cluster_id, "c0001");
    }

    #[test]
    fn cluster_level_instruments_are_constant_within_cluster() {
        let mut config = base();
        config.instrument_assignment_level = InstrumentAssignmentLevel::ClusterLevel;
        let data = generate_cluster_iv_dataset(config).unwrap();
        for cluster in 0..4 {
            let cluster_id = format!("c{:04}", cluster);
            let rows: Vec<&GeneratedObservation> =
                data.rows.iter().filter(|row| row.cluster_id == cluster_id).collect();
            assert!(!rows.is_empty());
            for row in rows.iter().skip(1) {
                assert_eq!(row.instruments, rows[0].instruments);
            }
        }
        assert_eq!(data.config.instrument_assignment_level, InstrumentAssignmentLevel::ClusterLevel);
    }

    #[test]
    fn within_cluster_instruments_vary_by_assignment_unit() {
        let mut config = base();
        config.instrument_assignment_level = InstrumentAssignmentLevel::WithinCluster;
        let data = generate_cluster_iv_dataset(config).unwrap();
        let rows: Vec<&GeneratedObservation> =
            data.rows.iter().filter(|row| row.cluster_id == "c0000").collect();
        assert!(rows.windows(2).any(|pair| pair[0].instruments != pair[1].instruments));
    }


    #[test]
    fn small_fixture_matches_pinned_cross_language_known_answers() {
        let data = generate_cluster_iv_dataset(small_known_answer_config()).unwrap();
        assert_eq!(data.rows.len(), 4);
        close(data.rows[0].instruments[0], 1.33731770252701, 1.0e-12);
        close(data.rows[0].instruments[1], -0.07086852162399182, 1.0e-12);
        close(data.rows[0].treatment, -0.2656092771974454, 1.0e-12);
        close(data.rows[0].outcome, -2.212120806447358, 1.0e-12);
        close(data.rows[0].first_stage_instrument_component, 1.0450501794532108, 1.0e-12);
        close(data.rows[0].direct_effect_component, 0.13656651111766066, 1.0e-12);
        close(data.rows[3].outcome, 0.8752110905326778, 1.0e-12);
        close(data.summary().direct_effect_sample_rms, 0.10839223216462172, 1.0e-12);
        close(data.summary().instrument_second_moment, 0.7644481460876965, 1.0e-12);
        assert_eq!(data.config.instrument_assignment_level, InstrumentAssignmentLevel::WithinCluster);
    }


    #[test]
    fn every_registered_scenario_preset_generates_a_valid_dataset() {
        let presets = [
            ClusterIvScenarioPreset::StrongExactExclusion,
            ClusterIvScenarioPreset::WeakExactExclusion,
            ClusterIvScenarioPreset::ZeroRelevance,
            ClusterIvScenarioPreset::DirectEffectInsideSupport,
            ClusterIvScenarioPreset::DirectEffectOutsideSupport,
            ClusterIvScenarioPreset::StrengthConcentratedInFewClusters,
            ClusterIvScenarioPreset::ClusterLevelInstruments,
            ClusterIvScenarioPreset::UnequalHeteroskedasticClusters,
        ];
        for preset in presets {
            let config = scenario_preset(preset, 20261101);
            assert_eq!(config.scenario_id, format!("{}-v1", preset.as_str()));
            let data = generate_cluster_iv_dataset(config).unwrap();
            assert_eq!(data.config.cluster_count(), 12);
            assert_eq!(data.rows.len(), data.config.observation_count());
        }
    }

    #[test]
    fn strong_exact_exclusion_preset_has_zero_direct_effect() {
        let data = generate_cluster_iv_dataset(
            scenario_preset(ClusterIvScenarioPreset::StrongExactExclusion, 9)
        ).unwrap();
        assert!(data.config.first_stage_cluster_multipliers.iter().all(|x| *x == 1.0));
        assert!(data.rows.iter().all(|row| row.direct_effect_component == 0.0));
        assert_eq!(data.summary().direct_effect_support_status, "IN_BOUND");
        assert_eq!(data.config.registered_direct_effect_rms_bound, Some(0.0));
    }

    #[test]
    fn weak_exact_exclusion_preset_records_weak_pi_and_zero_gamma() {
        let data = generate_cluster_iv_dataset(
            scenario_preset(ClusterIvScenarioPreset::WeakExactExclusion, 9)
        ).unwrap();
        assert!(data.config.first_stage_cluster_multipliers.iter().all(|x| *x == 0.08));
        assert!(data.config.direct_effect_coefficients.iter().all(|x| *x == 0.0));
        assert_eq!(data.summary().direct_effect_support_status, "IN_BOUND");
    }

    #[test]
    fn zero_relevance_preset_removes_all_first_stage_instrument_components() {
        let data = generate_cluster_iv_dataset(
            scenario_preset(ClusterIvScenarioPreset::ZeroRelevance, 9)
        ).unwrap();
        assert!(data.config.first_stage_coefficients.iter().all(|x| *x == 0.0));
        assert!(data.rows.iter().all(|row| row.first_stage_instrument_component == 0.0));
    }

    #[test]
    fn direct_effect_presets_are_classified_against_registered_g() {
        let inside = generate_cluster_iv_dataset(
            scenario_preset(ClusterIvScenarioPreset::DirectEffectInsideSupport, 9)
        ).unwrap();
        let outside = generate_cluster_iv_dataset(
            scenario_preset(ClusterIvScenarioPreset::DirectEffectOutsideSupport, 9)
        ).unwrap();
        assert_eq!(inside.summary().direct_effect_support_status, "IN_BOUND");
        assert_eq!(outside.summary().direct_effect_support_status, "OUT_OF_BOUND");
        assert!(inside.summary().direct_effect_sample_rms <= inside.config.registered_direct_effect_rms_bound.unwrap());
        assert!(outside.summary().direct_effect_sample_rms > outside.config.registered_direct_effect_rms_bound.unwrap());
    }

    #[test]
    fn concentration_preset_places_relevance_in_only_two_clusters() {
        let data = generate_cluster_iv_dataset(
            scenario_preset(ClusterIvScenarioPreset::StrengthConcentratedInFewClusters, 9)
        ).unwrap();
        assert_eq!(data.config.first_stage_cluster_multipliers[0], 3.0);
        assert_eq!(data.config.first_stage_cluster_multipliers[1], 2.0);
        assert!(data.config.first_stage_cluster_multipliers[2..].iter().all(|x| *x == 0.0));
        for row in data.rows.iter().filter(|row| row.cluster_id >= "c0002") {
            assert_eq!(row.first_stage_instrument_component, 0.0);
        }
    }

    #[test]
    fn cluster_level_preset_reuses_instruments_within_each_cluster() {
        let data = generate_cluster_iv_dataset(
            scenario_preset(ClusterIvScenarioPreset::ClusterLevelInstruments, 9)
        ).unwrap();
        assert_eq!(data.config.instrument_assignment_level, InstrumentAssignmentLevel::ClusterLevel);
        for cluster in 0..data.config.cluster_count() {
            let id = format!("c{:04}", cluster);
            let rows: Vec<&GeneratedObservation> = data.rows.iter().filter(|row| row.cluster_id == id).collect();
            assert!(rows.iter().all(|row| row.instruments == rows[0].instruments));
        }
    }

    #[test]
    fn unequal_heteroskedastic_preset_records_cluster_size_and_noise_variation() {
        let data = generate_cluster_iv_dataset(
            scenario_preset(ClusterIvScenarioPreset::UnequalHeteroskedasticClusters, 9)
        ).unwrap();
        assert_eq!(data.config.cluster_sizes[0], 4);
        assert_eq!(data.config.cluster_sizes[11], 26);
        assert!(data.config.cluster_noise_multipliers.windows(2).all(|pair| pair[0] < pair[1]));
        assert!(data.config.cluster_sizes.windows(2).all(|pair| pair[0] < pair[1]));
    }

    #[test]
    fn registered_direct_effect_bound_must_be_finite_and_nonnegative() {
        let mut config = base();
        config.registered_direct_effect_rms_bound = Some(-0.1);
        assert!(matches!(
            generate_cluster_iv_dataset(config),
            Err(DgpError::InvalidRegisteredDirectEffectBound { .. })
        ));
        let mut config = base();
        config.registered_direct_effect_rms_bound = Some(f64::NAN);
        assert!(matches!(
            generate_cluster_iv_dataset(config),
            Err(DgpError::InvalidRegisteredDirectEffectBound { .. })
        ));
    }

    #[test]
    fn absent_registered_direct_effect_bound_is_explicitly_unregistered() {
        let mut config = base();
        config.registered_direct_effect_rms_bound = None;
        let data = generate_cluster_iv_dataset(config).unwrap();
        assert_eq!(data.summary().direct_effect_support_status, "UNREGISTERED");
        assert_eq!(data.summary().registered_direct_effect_rms_bound, None);
    }

    #[test]
    fn manifest_contains_registered_direct_effect_bound_and_status() {
        let data = generate_cluster_iv_dataset(
            scenario_preset(ClusterIvScenarioPreset::DirectEffectInsideSupport, 9)
        ).unwrap();
        let json = data.manifest_json();
        assert!(json.contains("\"registered_direct_effect_rms_bound\":"));
        assert!(json.contains("\"direct_effect_support_status\":\"IN_BOUND\""));
    }

}
