//! Spatially structured observation-only nuisance generators for RCAF.
//!
//! This module is separate from the existing common-mode calibration kernel. It operates on
//! an explicit weighted ROI graph and generates deterministic additive fields. It does not
//! load neuroimaging data, estimate connectivity, denoise observations, or infer causes.

use std::fmt;

/// Validated weighted undirected ROI graph. Matrix row/column order is roi_labels.
#[derive(Debug, Clone, PartialEq)]
pub struct SpatialGraph {
    pub roi_labels: Vec<String>,
    pub adjacency: Vec<Vec<f64>>,
}

impl SpatialGraph {
    pub fn new(
        roi_labels: Vec<String>,
        adjacency: Vec<Vec<f64>>,
    ) -> Result<Self, SpatialNuisanceError> {
        let graph = Self {
            roi_labels,
            adjacency,
        };
        graph.validate()?;
        Ok(graph)
    }

    pub fn validate(&self) -> Result<(), SpatialNuisanceError> {
        let n = self.roi_labels.len();
        if n < 2 {
            return Err(SpatialNuisanceError::InvalidGeometry(
                "at least two ROI labels are required",
            ));
        }
        if self.adjacency.len() != n || self.adjacency.iter().any(|row| row.len() != n) {
            return Err(SpatialNuisanceError::InvalidGeometry(
                "adjacency dimensions must match ROI labels",
            ));
        }
        let mut labels = std::collections::HashSet::with_capacity(n);
        for label in &self.roi_labels {
            if label.trim().is_empty() {
                return Err(SpatialNuisanceError::InvalidGeometry(
                    "ROI labels must be non-empty",
                ));
            }
            if !labels.insert(label) {
                return Err(SpatialNuisanceError::InvalidGeometry(
                    "ROI labels must be unique",
                ));
            }
        }
        for i in 0..n {
            let degree: f64 = self.adjacency[i].iter().sum();
            if !degree.is_finite() {
                return Err(SpatialNuisanceError::InvalidGeometry(
                    "weighted node degree must remain finite",
                ));
            }
            for j in 0..n {
                let weight = self.adjacency[i][j];
                if !weight.is_finite() || weight < 0.0 {
                    return Err(SpatialNuisanceError::InvalidGeometry(
                        "graph weights must be finite and non-negative",
                    ));
                }
                if i == j && weight.abs() > 1e-12 {
                    return Err(SpatialNuisanceError::InvalidGeometry(
                        "self-loop weights must be zero",
                    ));
                }
                if (weight - self.adjacency[j][i]).abs() > 1e-10 {
                    return Err(SpatialNuisanceError::InvalidGeometry(
                        "adjacency must be symmetric",
                    ));
                }
            }
        }
        Ok(())
    }

    fn laplacian(&self) -> Vec<Vec<f64>> {
        let n = self.roi_labels.len();
        let mut laplacian = vec![vec![0.0; n]; n];
        for i in 0..n {
            laplacian[i][i] = self.adjacency[i].iter().sum();
            for j in 0..n {
                if i != j {
                    laplacian[i][j] = -self.adjacency[i][j];
                }
            }
        }
        laplacian
    }
}

/// The first two spatial nuisance families in the RCAF spatial challenge.
#[derive(Debug, Clone, PartialEq)]
pub enum SpatialNuisanceFamily {
    /// Low-frequency non-constant graph Laplacian modes.
    GraphSmoothLowRank { rank: usize, diffusion_time: f64 },
    /// Heat-kernel fields seeded at explicit ROI indices.
    HeatKernelLocalized {
        seed_roi_indices: Vec<usize>,
        diffusion_time: f64,
    },
}

/// Parameters for one deterministic spatial nuisance field.
#[derive(Debug, Clone, PartialEq)]
pub struct SpatialNuisanceConfig {
    pub family: SpatialNuisanceFamily,
    pub time_points: usize,
    /// Additive nuisance loading (not a variance fraction).
    pub dose: f64,
    /// AR(1) coefficient for the independent component of each nuisance factor.
    pub temporal_ar1: f64,
    /// Target-process mixture coefficient in [0, 1].
    pub temporal_coherence: f64,
    /// Root seed; one deterministic substream is derived per spatial mode.
    pub seed: u64,
}

/// Generated field and its factors. Matrix layout: basis ROI x mode; factors mode x time;
/// values time x ROI.
#[derive(Debug, Clone, PartialEq)]
pub struct SpatialNuisanceField {
    pub roi_labels: Vec<String>,
    pub spatial_basis: Vec<Vec<f64>>,
    pub factor_series: Vec<Vec<f64>>,
    pub values: Vec<Vec<f64>>,
    pub realized_temporal_coherence: Vec<Option<f64>>,
    pub realized_temporal_ar1: Vec<f64>,
}

impl SpatialNuisanceField {
    /// Add the same generated field to a clean counterfactual. Reuse this field unchanged for
    /// nuisance-only and signal-plus-nuisance cells to guarantee matched nuisance innovations.
    pub fn add_to(
        &self,
        clean_observations: &[Vec<f64>],
    ) -> Result<Vec<Vec<f64>>, SpatialNuisanceError> {
        if self.values.iter().any(|row| {
            row.len() != self.roi_labels.len() || row.iter().any(|value| !value.is_finite())
        }) {
            return Err(SpatialNuisanceError::InvalidObservation(
                "stored nuisance field has invalid shape or non-finite values",
            ));
        }
        if clean_observations.len() != self.values.len()
            || clean_observations
                .iter()
                .any(|row| row.len() != self.roi_labels.len())
        {
            return Err(SpatialNuisanceError::DimensionMismatch(
                "clean observations must be time x ROI and match the nuisance field",
            ));
        }
        let mut combined = Vec::with_capacity(clean_observations.len());
        for (time_index, row) in clean_observations.iter().enumerate() {
            let mut output_row = Vec::with_capacity(row.len());
            for (roi_index, value) in row.iter().enumerate() {
                if !value.is_finite() {
                    return Err(SpatialNuisanceError::InvalidObservation(
                        "clean observations must be finite",
                    ));
                }
                let combined_value = *value + self.values[time_index][roi_index];
                if !combined_value.is_finite() {
                    return Err(SpatialNuisanceError::InvalidObservation(
                        "combined observations must remain finite",
                    ));
                }
                output_row.push(combined_value);
            }
            combined.push(output_row);
        }
        Ok(combined)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpatialNuisanceError {
    InvalidGeometry(&'static str),
    InvalidConfig(&'static str),
    DimensionMismatch(&'static str),
    InvalidObservation(&'static str),
    NoNonConstantGraphModes,
    RankExceedsAvailableModes { requested: usize, available: usize },
    RankSplitsDegenerateEigenvalue,
    DegenerateSpatialBasis,
    EigenSolverDidNotConverge,
}

impl fmt::Display for SpatialNuisanceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidGeometry(reason) => write!(f, "invalid spatial graph: {reason}"),
            Self::InvalidConfig(reason) => write!(f, "invalid spatial nuisance config: {reason}"),
            Self::DimensionMismatch(reason) => write!(f, "spatial nuisance dimension mismatch: {reason}"),
            Self::InvalidObservation(reason) => write!(f, "invalid spatial nuisance observation: {reason}"),
            Self::NoNonConstantGraphModes => write!(f, "graph has no usable non-constant spatial modes"),
            Self::RankExceedsAvailableModes { requested, available } => write!(
                f,
                "requested spatial rank {requested} exceeds {available} available non-constant modes"
            ),
            Self::RankSplitsDegenerateEigenvalue => write!(
                f,
                "requested rank splits a repeated Laplacian eigenspace; include the full eigenspace"
            ),
            Self::DegenerateSpatialBasis => write!(f, "spatial basis is rank deficient after graph projection"),
            Self::EigenSolverDidNotConverge => write!(f, "symmetric graph eigensolver did not converge"),
        }
    }
}

impl std::error::Error for SpatialNuisanceError {}

/// Generate a deterministic additive spatial nuisance field.
///
/// Positive coherence requires one target time series per spatial mode. Each series is
/// standardized before mixing with an independent stationary AR(1) process. The parameter is
/// a construction coefficient; realized finite-sample correlation is returned separately.
/// The same returned values must be reused across matched nuisance-only / combined cells.
pub fn generate_spatial_nuisance_field(
    graph: &SpatialGraph,
    config: &SpatialNuisanceConfig,
    target_processes: Option<&[Vec<f64>]>,
) -> Result<SpatialNuisanceField, SpatialNuisanceError> {
    graph.validate()?;
    validate_config(config)?;

    let basis = match &config.family {
        SpatialNuisanceFamily::GraphSmoothLowRank {
            rank,
            diffusion_time,
        } => graph_smooth_basis(graph, *rank, *diffusion_time)?,
        SpatialNuisanceFamily::HeatKernelLocalized {
            seed_roi_indices,
            diffusion_time,
        } => heat_kernel_basis(graph, seed_roi_indices, *diffusion_time)?,
    };
    let roi_count = graph.roi_labels.len();
    let mode_count = basis.first().map_or(0, Vec::len);
    if mode_count == 0 {
        return Err(SpatialNuisanceError::DegenerateSpatialBasis);
    }

    let standardized_targets = if config.temporal_coherence > 0.0 {
        let targets = target_processes.ok_or(SpatialNuisanceError::InvalidConfig(
            "positive temporal coherence requires target processes",
        ))?;
        if targets.len() != mode_count || targets.iter().any(|row| row.len() != config.time_points)
        {
            return Err(SpatialNuisanceError::DimensionMismatch(
                "one target time series per spatial mode is required",
            ));
        }
        let mut output = Vec::with_capacity(mode_count);
        for row in targets {
            output.push(standardize(row)?);
        }
        Some(output)
    } else {
        None
    };

    let independent_weight = (1.0 - config.temporal_coherence.powi(2)).sqrt();
    let mut factor_series = Vec::with_capacity(mode_count);
    let mut realized_temporal_coherence = Vec::with_capacity(mode_count);
    let mut realized_temporal_ar1 = Vec::with_capacity(mode_count);

    for mode in 0..mode_count {
        let mut rng = GaussianRng::new(derive_substream_seed(config.seed, mode as u64));
        let independent = ar1_series(&mut rng, config.time_points, config.temporal_ar1);
        let factor = if let Some(targets) = &standardized_targets {
            targets[mode]
                .iter()
                .zip(independent.iter())
                .map(|(target, noise)| {
                    config.temporal_coherence * target + independent_weight * noise
                })
                .collect::<Vec<_>>()
        } else {
            independent
        };
        let coherence = standardized_targets
            .as_ref()
            .map(|targets| correlation(&factor, &targets[mode]).unwrap_or(0.0));
        realized_temporal_ar1.push(lag_one_autocorrelation(&factor).unwrap_or(0.0));
        realized_temporal_coherence.push(coherence);
        factor_series.push(factor);
    }

    let mut values = vec![vec![0.0; roi_count]; config.time_points];
    for (roi, basis_row) in basis.iter().enumerate() {
        for (mode, loading) in basis_row.iter().enumerate() {
            for time in 0..config.time_points {
                values[time][roi] += config.dose * loading * factor_series[mode][time];
            }
        }
    }

    if values.iter().flatten().any(|value| !value.is_finite()) {
        return Err(SpatialNuisanceError::InvalidConfig(
            "dose and factor loadings produced a non-finite nuisance field",
        ));
    }

    Ok(SpatialNuisanceField {
        roi_labels: graph.roi_labels.clone(),
        spatial_basis: basis,
        factor_series,
        values,
        realized_temporal_coherence,
        realized_temporal_ar1,
    })
}

fn validate_config(config: &SpatialNuisanceConfig) -> Result<(), SpatialNuisanceError> {
    if config.time_points < 3 {
        return Err(SpatialNuisanceError::InvalidConfig(
            "at least three time points are required",
        ));
    }
    if !config.dose.is_finite() || config.dose < 0.0 {
        return Err(SpatialNuisanceError::InvalidConfig(
            "dose must be finite and non-negative",
        ));
    }
    if !config.temporal_ar1.is_finite() || config.temporal_ar1.abs() >= 1.0 {
        return Err(SpatialNuisanceError::InvalidConfig(
            "temporal AR(1) must be finite and strictly between -1 and 1",
        ));
    }
    if !config.temporal_coherence.is_finite() || !(0.0..=1.0).contains(&config.temporal_coherence) {
        return Err(SpatialNuisanceError::InvalidConfig(
            "temporal coherence must be finite and in [0, 1]",
        ));
    }
    match &config.family {
        SpatialNuisanceFamily::GraphSmoothLowRank {
            rank,
            diffusion_time,
        } => {
            if *rank == 0 {
                return Err(SpatialNuisanceError::InvalidConfig(
                    "graph-smooth rank must be positive",
                ));
            }
            validate_diffusion_time(*diffusion_time)?;
        }
        SpatialNuisanceFamily::HeatKernelLocalized {
            seed_roi_indices,
            diffusion_time,
        } => {
            if seed_roi_indices.is_empty() {
                return Err(SpatialNuisanceError::InvalidConfig(
                    "at least one heat-kernel seed ROI is required",
                ));
            }
            validate_diffusion_time(*diffusion_time)?;
        }
    }
    Ok(())
}

fn validate_diffusion_time(value: f64) -> Result<(), SpatialNuisanceError> {
    if !value.is_finite() || value <= 0.0 {
        return Err(SpatialNuisanceError::InvalidConfig(
            "diffusion time must be finite and positive",
        ));
    }
    Ok(())
}

#[derive(Debug, Clone)]
struct EigenSystem {
    values: Vec<f64>,
    /// Eigenvectors stored as ROI x mode.
    vectors: Vec<Vec<f64>>,
    tolerance: f64,
}

/// Jacobi eigensolver for a small symmetric Laplacian. Repeated eigenspaces are handled as
/// projectors for heat-kernel fields; low-rank truncation may not split a repeated eigenvalue.
fn symmetric_eigendecomposition(matrix: &[Vec<f64>]) -> Result<EigenSystem, SpatialNuisanceError> {
    let n = matrix.len();
    let mut diagonalized = matrix.to_vec();
    let mut vectors = vec![vec![0.0; n]; n];
    for (index, row) in vectors.iter_mut().enumerate() {
        row[index] = 1.0;
    }

    let max_iterations = 100usize.saturating_mul(n).saturating_mul(n).max(1);
    let mut converged = false;
    for _ in 0..max_iterations {
        let mut p = 0usize;
        let mut q = 1usize;
        let mut largest = 0.0f64;
        for i in 0..n {
            for j in (i + 1)..n {
                if diagonalized[i][j].abs() > largest {
                    largest = diagonalized[i][j].abs();
                    p = i;
                    q = j;
                }
            }
        }
        if largest <= 1e-13 {
            converged = true;
            break;
        }

        let app = diagonalized[p][p];
        let aqq = diagonalized[q][q];
        let apq = diagonalized[p][q];
        let angle = 0.5 * (2.0 * apq).atan2(aqq - app);
        let cosine = angle.cos();
        let sine = angle.sin();

        for k in 0..n {
            if k != p && k != q {
                let akp = diagonalized[k][p];
                let akq = diagonalized[k][q];
                let new_kp = cosine * akp - sine * akq;
                let new_kq = sine * akp + cosine * akq;
                diagonalized[k][p] = new_kp;
                diagonalized[p][k] = new_kp;
                diagonalized[k][q] = new_kq;
                diagonalized[q][k] = new_kq;
            }
        }
        diagonalized[p][p] = cosine * cosine * app - 2.0 * sine * cosine * apq + sine * sine * aqq;
        diagonalized[q][q] = sine * sine * app + 2.0 * sine * cosine * apq + cosine * cosine * aqq;
        diagonalized[p][q] = 0.0;
        diagonalized[q][p] = 0.0;

        for row in &mut vectors {
            let vip = row[p];
            let viq = row[q];
            row[p] = cosine * vip - sine * viq;
            row[q] = sine * vip + cosine * viq;
        }
    }
    if !converged {
        return Err(SpatialNuisanceError::EigenSolverDidNotConverge);
    }

    let mut order: Vec<usize> = (0..n).collect();
    order
        .sort_by(|left, right| diagonalized[*left][*left].total_cmp(&diagonalized[*right][*right]));
    let raw_values: Vec<f64> = order.iter().map(|&i| diagonalized[i][i]).collect();
    let spectral_scale = raw_values
        .iter()
        .map(|value| value.abs())
        .fold(0.0f64, f64::max);
    let tolerance = if spectral_scale > 0.0 {
        spectral_scale * 1e-10
    } else {
        1e-12
    };
    let mut values = Vec::with_capacity(n);
    for value in raw_values {
        if value < -tolerance {
            return Err(SpatialNuisanceError::InvalidGeometry(
                "graph Laplacian has a materially negative eigenvalue",
            ));
        }
        values.push(if value.abs() <= tolerance { 0.0 } else { value });
    }

    let mut sorted_vectors = vec![vec![0.0; n]; n];
    for (mode, &old_mode) in order.iter().enumerate() {
        for roi in 0..n {
            sorted_vectors[roi][mode] = vectors[roi][old_mode];
        }
        let pivot = (0..n)
            .max_by(|left, right| {
                sorted_vectors[*left][mode]
                    .abs()
                    .total_cmp(&sorted_vectors[*right][mode].abs())
            })
            .unwrap_or(0);
        if sorted_vectors[pivot][mode] < 0.0 {
            for row in &mut sorted_vectors {
                row[mode] = -row[mode];
            }
        }
    }

    Ok(EigenSystem {
        values,
        vectors: sorted_vectors,
        tolerance,
    })
}

fn non_constant_modes(eigen: &EigenSystem) -> Vec<usize> {
    eigen
        .values
        .iter()
        .enumerate()
        .filter_map(|(index, value)| (*value > eigen.tolerance).then_some(index))
        .collect()
}

fn graph_smooth_basis(
    graph: &SpatialGraph,
    rank: usize,
    diffusion_time: f64,
) -> Result<Vec<Vec<f64>>, SpatialNuisanceError> {
    let eigen = symmetric_eigendecomposition(&graph.laplacian())?;
    let modes = non_constant_modes(&eigen);
    if modes.is_empty() {
        return Err(SpatialNuisanceError::NoNonConstantGraphModes);
    }
    if rank > modes.len() {
        return Err(SpatialNuisanceError::RankExceedsAvailableModes {
            requested: rank,
            available: modes.len(),
        });
    }
    if rank < modes.len()
        && (eigen.values[modes[rank]] - eigen.values[modes[rank - 1]]).abs() <= eigen.tolerance
    {
        return Err(SpatialNuisanceError::RankSplitsDegenerateEigenvalue);
    }

    let mut basis = vec![vec![0.0; rank]; graph.roi_labels.len()];
    let mut squared_norm = 0.0;
    for (mode, &index) in modes.iter().take(rank).enumerate() {
        let attenuation = (-diffusion_time * eigen.values[index]).exp();
        for roi in 0..graph.roi_labels.len() {
            let value = eigen.vectors[roi][index] * attenuation;
            basis[roi][mode] = value;
            squared_norm += value * value;
        }
    }
    if !squared_norm.is_finite() || squared_norm <= 1e-20 {
        return Err(SpatialNuisanceError::DegenerateSpatialBasis);
    }
    let normalization = (rank as f64 / squared_norm).sqrt();
    for row in &mut basis {
        for value in row {
            *value *= normalization;
        }
    }
    Ok(basis)
}

fn heat_kernel_basis(
    graph: &SpatialGraph,
    seed_roi_indices: &[usize],
    diffusion_time: f64,
) -> Result<Vec<Vec<f64>>, SpatialNuisanceError> {
    let n = graph.roi_labels.len();
    let mut seen = std::collections::HashSet::with_capacity(seed_roi_indices.len());
    for &seed in seed_roi_indices {
        if seed >= n {
            return Err(SpatialNuisanceError::InvalidConfig(
                "heat-kernel seed is outside graph",
            ));
        }
        if !seen.insert(seed) {
            return Err(SpatialNuisanceError::InvalidConfig(
                "heat-kernel seeds must be unique",
            ));
        }
    }
    let eigen = symmetric_eigendecomposition(&graph.laplacian())?;
    let modes = non_constant_modes(&eigen);
    if modes.is_empty() {
        return Err(SpatialNuisanceError::NoNonConstantGraphModes);
    }

    let mut basis = vec![vec![0.0; seed_roi_indices.len()]; n];
    for (column, &seed) in seed_roi_indices.iter().enumerate() {
        for &mode in &modes {
            let attenuation = (-diffusion_time * eigen.values[mode]).exp();
            let seed_component = eigen.vectors[seed][mode];
            for roi in 0..n {
                basis[roi][column] += attenuation * eigen.vectors[roi][mode] * seed_component;
            }
        }
    }
    orthonormalize_columns(&mut basis)?;
    Ok(basis)
}

fn orthonormalize_columns(basis: &mut [Vec<f64>]) -> Result<(), SpatialNuisanceError> {
    if basis.is_empty() {
        return Err(SpatialNuisanceError::DegenerateSpatialBasis);
    }
    let rows = basis.len();
    let columns = basis[0].len();
    if columns == 0 || basis.iter().any(|row| row.len() != columns) {
        return Err(SpatialNuisanceError::DegenerateSpatialBasis);
    }
    for column in 0..columns {
        for previous in 0..column {
            let projection: f64 = (0..rows)
                .map(|row| basis[row][column] * basis[row][previous])
                .sum();
            for row in 0..rows {
                basis[row][column] -= projection * basis[row][previous];
            }
        }
        let norm = (0..rows)
            .map(|row| basis[row][column] * basis[row][column])
            .sum::<f64>()
            .sqrt();
        if !norm.is_finite() || norm <= 1e-10 {
            return Err(SpatialNuisanceError::DegenerateSpatialBasis);
        }
        for row in 0..rows {
            basis[row][column] /= norm;
        }
        let pivot = (0..rows)
            .max_by(|left, right| {
                basis[*left][column]
                    .abs()
                    .total_cmp(&basis[*right][column].abs())
            })
            .unwrap_or(0);
        if basis[pivot][column] < 0.0 {
            for row in basis.iter_mut() {
                row[column] = -row[column];
            }
        }
    }
    Ok(())
}

fn standardize(values: &[f64]) -> Result<Vec<f64>, SpatialNuisanceError> {
    if values.len() < 3 || values.iter().any(|value| !value.is_finite()) {
        return Err(SpatialNuisanceError::InvalidConfig(
            "target processes must contain at least three finite values",
        ));
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / values.len() as f64;
    if !variance.is_finite() || variance <= 1e-20 {
        return Err(SpatialNuisanceError::InvalidConfig(
            "target processes must have non-zero variance",
        ));
    }
    let scale = variance.sqrt();
    Ok(values.iter().map(|value| (value - mean) / scale).collect())
}

fn correlation(left: &[f64], right: &[f64]) -> Option<f64> {
    if left.len() != right.len() || left.len() < 3 {
        return None;
    }
    let left_mean = left.iter().sum::<f64>() / left.len() as f64;
    let right_mean = right.iter().sum::<f64>() / right.len() as f64;
    let mut covariance = 0.0;
    let mut left_variance = 0.0;
    let mut right_variance = 0.0;
    for (&a, &b) in left.iter().zip(right.iter()) {
        let da = a - left_mean;
        let db = b - right_mean;
        covariance += da * db;
        left_variance += da * da;
        right_variance += db * db;
    }
    let denominator = (left_variance * right_variance).sqrt();
    (denominator > 1e-20).then_some((covariance / denominator).clamp(-1.0, 1.0))
}

fn lag_one_autocorrelation(values: &[f64]) -> Option<f64> {
    if values.len() < 4 {
        return None;
    }
    correlation(&values[..values.len() - 1], &values[1..])
}

fn ar1_series(rng: &mut GaussianRng, length: usize, coefficient: f64) -> Vec<f64> {
    let innovation_scale = (1.0 - coefficient * coefficient).sqrt();
    let mut previous = rng.normal();
    let mut values = Vec::with_capacity(length);
    for _ in 0..length {
        previous = coefficient * previous + innovation_scale * rng.normal();
        values.push(previous);
    }
    values
}

fn derive_substream_seed(seed: u64, stream: u64) -> u64 {
    let mut value = seed ^ stream.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 {
                0x9E37_79B9_7F4A_7C15
            } else {
                seed
            },
        }
    }
    fn next(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }
    fn uniform_open(&mut self) -> f64 {
        let mantissa = self.next() >> 11;
        (mantissa as f64 + 0.5) * (1.0 / ((1u64 << 53) as f64))
    }
}

struct GaussianRng {
    uniform: XorShift64,
    spare: Option<f64>,
}

impl GaussianRng {
    fn new(seed: u64) -> Self {
        Self {
            uniform: XorShift64::new(seed),
            spare: None,
        }
    }
    fn normal(&mut self) -> f64 {
        if let Some(value) = self.spare.take() {
            return value;
        }
        let u1 = self.uniform.uniform_open();
        let u2 = self.uniform.uniform_open();
        let radius = (-2.0 * u1.ln()).sqrt();
        let angle = std::f64::consts::TAU * u2;
        self.spare = Some(radius * angle.sin());
        radius * angle.cos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn path_graph() -> SpatialGraph {
        let labels = (0..6).map(|index| format!("roi-{index}")).collect();
        let mut adjacency = vec![vec![0.0; 6]; 6];
        for index in 0..5 {
            adjacency[index][index + 1] = 1.0;
            adjacency[index + 1][index] = 1.0;
        }
        SpatialGraph::new(labels, adjacency).unwrap()
    }

    fn config() -> SpatialNuisanceConfig {
        SpatialNuisanceConfig {
            family: SpatialNuisanceFamily::GraphSmoothLowRank {
                rank: 2,
                diffusion_time: 0.35,
            },
            time_points: 128,
            dose: 0.2,
            temporal_ar1: 0.4,
            temporal_coherence: 0.0,
            seed: 0x5243_4146_5350_4154,
        }
    }

    fn targets(modes: usize, time_points: usize) -> Vec<Vec<f64>> {
        (0..modes)
            .map(|mode| {
                (0..time_points)
                    .map(|time| ((time as f64 + 1.0) * (mode as f64 + 1.0) * 0.071).sin())
                    .collect()
            })
            .collect()
    }

    #[test]
    fn graph_rejects_asymmetric_negative_and_duplicate_inputs() {
        let mut graph = path_graph();
        graph.adjacency[0][1] = 2.0;
        assert!(matches!(
            graph.validate(),
            Err(SpatialNuisanceError::InvalidGeometry(_))
        ));

        let mut graph = path_graph();
        graph.adjacency[0][1] = -1.0;
        graph.adjacency[1][0] = -1.0;
        assert!(matches!(
            graph.validate(),
            Err(SpatialNuisanceError::InvalidGeometry(_))
        ));

        let mut graph = path_graph();
        graph.roi_labels[1] = graph.roi_labels[0].clone();
        assert!(matches!(
            graph.validate(),
            Err(SpatialNuisanceError::InvalidGeometry(_))
        ));
    }

    #[test]
    fn graph_smooth_basis_excludes_constant_mode() {
        let field = generate_spatial_nuisance_field(&path_graph(), &config(), None).unwrap();
        for mode in 0..field.spatial_basis[0].len() {
            let sum: f64 = field.spatial_basis.iter().map(|row| row[mode]).sum();
            assert!(
                sum.abs() < 1e-9,
                "mode {mode} retained constant component {sum}"
            );
        }
    }

    #[test]
    fn generated_field_replays_exactly_for_same_seed_and_config() {
        let graph = path_graph();
        let config = config();
        assert_eq!(
            generate_spatial_nuisance_field(&graph, &config, None).unwrap(),
            generate_spatial_nuisance_field(&graph, &config, None).unwrap()
        );
    }

    #[test]
    fn zero_dose_is_an_exact_zero_field() {
        let graph = path_graph();
        let mut config = config();
        config.dose = 0.0;
        let field = generate_spatial_nuisance_field(&graph, &config, None).unwrap();
        assert!(field.values.iter().flatten().all(|value| *value == 0.0));
    }

    #[test]
    fn heat_kernel_field_is_finite_and_localized_at_declared_seed() {
        let graph = path_graph();
        let config = SpatialNuisanceConfig {
            family: SpatialNuisanceFamily::HeatKernelLocalized {
                seed_roi_indices: vec![2],
                diffusion_time: 0.2,
            },
            ..config()
        };
        let field = generate_spatial_nuisance_field(&graph, &config, None).unwrap();
        assert_eq!(field.spatial_basis.len(), 6);
        assert_eq!(field.spatial_basis[0].len(), 1);
        assert!(field.values.iter().flatten().all(|value| value.is_finite()));
        assert!(field.spatial_basis[2][0].abs() > field.spatial_basis[0][0].abs());
    }

    #[test]
    fn coherence_one_matches_standardized_target_exactly() {
        let graph = path_graph();
        let mut config = config();
        config.temporal_coherence = 1.0;
        let targets = targets(2, config.time_points);
        let field = generate_spatial_nuisance_field(&graph, &config, Some(&targets)).unwrap();
        for mode in 0..2 {
            let expected = standardize(&targets[mode]).unwrap();
            assert_eq!(field.factor_series[mode], expected);
            assert!((field.realized_temporal_coherence[mode].unwrap() - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn positive_coherence_requires_target_processes() {
        let graph = path_graph();
        let mut config = config();
        config.temporal_coherence = 0.5;
        assert!(matches!(
            generate_spatial_nuisance_field(&graph, &config, None),
            Err(SpatialNuisanceError::InvalidConfig(_))
        ));
    }

    #[test]
    fn one_generated_field_can_be_reused_across_matched_counterfactuals() {
        let graph = path_graph();
        let field = generate_spatial_nuisance_field(&graph, &config(), None).unwrap();
        let clean_a = vec![vec![0.1; 6]; 128];
        let clean_b = vec![vec![0.3; 6]; 128];
        let nuisance_only = field.add_to(&clean_a).unwrap();
        let signal_plus_nuisance = field.add_to(&clean_b).unwrap();
        for time in 0..128 {
            for roi in 0..6 {
                assert_eq!(
                    nuisance_only[time][roi] - clean_a[time][roi],
                    signal_plus_nuisance[time][roi] - clean_b[time][roi]
                );
            }
        }
    }

    #[test]
    fn add_to_rejects_wrong_observation_dimensions() {
        let field = generate_spatial_nuisance_field(&path_graph(), &config(), None).unwrap();
        assert!(matches!(
            field.add_to(&(0..128).map(|_| vec![0.0; 5]).collect::<Vec<_>>()),
            Err(SpatialNuisanceError::DimensionMismatch(_))
        ));
    }

    #[test]
    fn invalid_temporal_parameters_fail_closed() {
        let graph = path_graph();
        let mut invalid_ar1 = config();
        invalid_ar1.temporal_ar1 = 1.0;
        assert!(matches!(
            generate_spatial_nuisance_field(&graph, &invalid_ar1, None),
            Err(SpatialNuisanceError::InvalidConfig(_))
        ));
        let mut invalid_coherence = config();
        invalid_coherence.temporal_coherence = 1.01;
        assert!(matches!(
            generate_spatial_nuisance_field(&graph, &invalid_coherence, None),
            Err(SpatialNuisanceError::InvalidConfig(_))
        ));
    }

    #[test]
    fn low_rank_basis_rejects_splitting_degenerate_eigenspace() {
        let labels = (0..4).map(|index| format!("cycle-{index}")).collect();
        let mut adjacency = vec![vec![0.0; 4]; 4];
        for (left, right) in [(0, 1), (1, 2), (2, 3), (3, 0)] {
            adjacency[left][right] = 1.0;
            adjacency[right][left] = 1.0;
        }
        let graph = SpatialGraph::new(labels, adjacency).unwrap();
        let config = SpatialNuisanceConfig {
            family: SpatialNuisanceFamily::GraphSmoothLowRank {
                rank: 1,
                diffusion_time: 0.3,
            },
            ..config()
        };
        assert!(matches!(
            generate_spatial_nuisance_field(&graph, &config, None),
            Err(SpatialNuisanceError::RankSplitsDegenerateEigenvalue)
        ));
    }

    #[test]
    fn joint_roi_permutation_preserves_spatial_projector() {
        let graph = path_graph();
        let original = generate_spatial_nuisance_field(&graph, &config(), None).unwrap();
        let permutation = [3usize, 0, 5, 1, 4, 2];
        let labels = permutation
            .iter()
            .map(|&i| graph.roi_labels[i].clone())
            .collect();
        let adjacency = permutation
            .iter()
            .map(|&i| permutation.iter().map(|&j| graph.adjacency[i][j]).collect())
            .collect();
        let permuted_graph = SpatialGraph::new(labels, adjacency).unwrap();
        let permuted = generate_spatial_nuisance_field(&permuted_graph, &config(), None).unwrap();
        for left in 0..6 {
            for right in 0..6 {
                let original_projector: f64 = (0..2)
                    .map(|mode| {
                        original.spatial_basis[permutation[left]][mode]
                            * original.spatial_basis[permutation[right]][mode]
                    })
                    .sum();
                let permuted_projector: f64 = (0..2)
                    .map(|mode| {
                        permuted.spatial_basis[left][mode] * permuted.spatial_basis[right][mode]
                    })
                    .sum();
                assert!((original_projector - permuted_projector).abs() < 1e-8);
            }
        }
    }

    #[test]
    fn heat_kernel_rejects_duplicate_and_out_of_range_seeds() {
        for seeds in [vec![1, 1], vec![6]] {
            let config = SpatialNuisanceConfig {
                family: SpatialNuisanceFamily::HeatKernelLocalized {
                    seed_roi_indices: seeds,
                    diffusion_time: 0.3,
                },
                ..config()
            };
            assert!(generate_spatial_nuisance_field(&path_graph(), &config, None).is_err());
        }
    }
}
