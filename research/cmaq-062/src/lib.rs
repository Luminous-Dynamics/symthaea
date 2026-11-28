#![forbid(unsafe_code)]

//! Localized conditional e-processes on a frozen non-overlapping context tree.
//!
//! Each observation follows one root-to-leaf path determined by its
//! pre-observation covariate. Every active node computes a predictable local
//! paired feature factor; inactive nodes retain unit factor. A fixed,
//! normalized exact-rational prior mixture over all nodes yields one global
//! e-process and one alpha threshold.

use std::collections::{HashMap, HashSet};

use cmaq_048_fourier_kernel_approachability::{FourierFeatureMap, KernelError};
use cmaq_053_compositional_risk_budget::Rational;
use sha2::{Digest, Sha256};

const POLICY_DOMAIN: &[u8] = b"LUMINOUS-DYNAMICS/CMAQ-062/CONTEXT-TREE-PORTFOLIO/V1\0";
const TRACE_DOMAIN: &[u8] = b"LUMINOUS-DYNAMICS/CMAQ-062/CONTEXT-TREE-TRACE/V1\0";
const MAX_NODES: usize = 256;
const MAX_DEPTH: usize = 32;
const MAX_PAIR_IDS: usize = 1_000_000;
const MAX_ID_BYTES: usize = 256;
const NORM_TOLERANCE: f64 = 1e-12;
const POSTERIOR_TOLERANCE: f64 = 1e-8;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ContextError {
    EmptyInventory,
    TooManyNodes,
    InvalidNodeId,
    DuplicateNodeId,
    InvalidDepth,
    InvalidRegionDimension,
    InvalidRegionBound,
    InvalidRoot,
    InvalidParent,
    OverlappingSiblings,
    InvalidPrior,
    PriorsDoNotSumToOne,
    InputDimensionMismatch,
    NonFiniteInput,
    InvalidBettingRate,
    InvalidAlpha,
    InvalidPairId,
    PairIdTooLong,
    DuplicatePairId,
    PairIdInventoryFull,
    SequenceMismatch,
    AlreadyQuarantined,
    NodeFeatureBudgetExceeded,
    TotalFeatureBudgetExceeded,
    FeatureDimensionOverflow,
    ObservationNormExceeded,
    InvalidFactor,
    PosteriorNormalizationFailure,
    ArithmeticOverflow,
    Kernel(KernelError),
}

impl From<KernelError> for ContextError {
    fn from(value: KernelError) -> Self { Self::Kernel(value) }
}
impl std::fmt::Display for ContextError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "{self:?}") }
}
impl std::error::Error for ContextError {}

/// Axis-aligned half-open box [lower, upper). None means unbounded.
/// Lower boundaries are included; upper boundaries are excluded.
#[derive(Clone, Debug, PartialEq)]
pub struct ContextRegion {
    pub lower: Vec<Option<f64>>,
    pub upper: Vec<Option<f64>>,
}

impl ContextRegion {
    pub fn new(lower: Vec<Option<f64>>, upper: Vec<Option<f64>>) -> Result<Self, ContextError> {
        if lower.is_empty() || lower.len() != upper.len() {
            return Err(ContextError::InvalidRegionDimension);
        }
        for (lo, hi) in lower.iter().zip(&upper) {
            if lo.is_some_and(|x| !x.is_finite()) || hi.is_some_and(|x| !x.is_finite()) {
                return Err(ContextError::InvalidRegionBound);
            }
            if let (Some(l), Some(u)) = (lo, hi) {
                if l >= u { return Err(ContextError::InvalidRegionBound); }
            }
        }
        Ok(Self { lower, upper })
    }

    pub fn root(dimension: usize) -> Result<Self, ContextError> {
        if dimension == 0 { return Err(ContextError::InvalidRegionDimension); }
        Self::new(vec![None; dimension], vec![None; dimension])
    }

    pub fn contains(&self, point: &[f64]) -> bool {
        point.len() == self.lower.len()
            && point.iter().enumerate().all(|(i, x)| {
                self.lower[i].is_none_or(|lo| *x >= lo)
                    && self.upper[i].is_none_or(|hi| *x < hi)
            })
    }

    fn contains_region(&self, child: &Self) -> bool {
        if self.lower.len() != child.lower.len() { return false; }
        for i in 0..self.lower.len() {
            match (self.lower[i], child.lower[i]) {
                (Some(parent), Some(child)) if child < parent => return false,
                (Some(_), None) => return false,
                _ => {}
            }
            match (self.upper[i], child.upper[i]) {
                (Some(parent), Some(child)) if child > parent => return false,
                (Some(_), None) => return false,
                _ => {}
            }
        }
        true
    }

    fn overlaps(&self, other: &Self) -> bool {
        if self.lower.len() != other.lower.len() { return false; }
        (0..self.lower.len()).all(|i| {
            let left_lower = self.lower[i].unwrap_or(f64::NEG_INFINITY);
            let right_lower = other.lower[i].unwrap_or(f64::NEG_INFINITY);
            let left_upper = self.upper[i].unwrap_or(f64::INFINITY);
            let right_upper = other.upper[i].unwrap_or(f64::INFINITY);
            left_lower.max(right_lower) < left_upper.min(right_upper)
        })
    }

    fn encode(&self, hasher: &mut Sha256) {
        hasher.update((self.lower.len() as u64).to_be_bytes());
        for (lo, hi) in self.lower.iter().zip(&self.upper) {
            encode_bound(hasher, *lo);
            encode_bound(hasher, *hi);
        }
    }
}

#[derive(Clone, Debug)]
pub struct ContextNodeConfig {
    pub node_id: String,
    pub parent_id: Option<String>,
    pub depth: usize,
    pub region: ContextRegion,
    pub response_map: FourierFeatureMap,
    pub covariate_map: FourierFeatureMap,
    pub prior_weight: Rational,
}

#[derive(Clone, Debug)]
struct ContextNode {
    node_id: String,
    parent_index: Option<usize>,
    children: Vec<usize>,
    depth: usize,
    region: ContextRegion,
    response_map: FourierFeatureMap,
    covariate_map: FourierFeatureMap,
    prior_weight: Rational,
    log_prior_weight: f64,
    feature_sum: Vec<f64>,
    log_e_value: f64,
    selected_count: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ContextNodeUpdate {
    pub node_id: String,
    pub depth: usize,
    pub prior_weight: Rational,
    pub posterior_weight: f64,
    pub log_e_value: f64,
    pub log_factor: f64,
    pub factor: f64,
    pub feature_coordinates_evaluated: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ContextTreeUpdate {
    pub sequence: u64,
    pub pair_id: String,
    pub active_path: Vec<String>,
    pub nodes: Vec<ContextNodeUpdate>,
    pub log_mixture_e_value: f64,
    pub log_threshold: f64,
    pub quarantined: bool,
    pub trace_sha256: String,
    pub total_feature_coordinates_evaluated: usize,
}

#[derive(Clone, Debug)]
struct StagedNode {
    index: usize,
    feature_sum: Vec<f64>,
    log_e_value: f64,
    factor: f64,
    log_factor: f64,
    feature_coordinates: usize,
    next_selected_count: u64,
}

/// A finite frozen context tree with one fixed prior mixture over all nodes.
/// Siblings must be disjoint; every child region must be contained in its
/// parent. Thus each covariate selects exactly one root-to-leaf path.
#[derive(Clone, Debug)]
pub struct ContextTreePortfolio {
    nodes: Vec<ContextNode>,
    root_index: usize,
    alpha: Rational,
    betting_rate: f64,
    log_threshold: f64,
    policy_sha256: String,
    trace_sha256: String,
    mixture_leaf_base: usize,
    mixture_tree: Vec<f64>,
    observations_seen: u64,
    pair_ids: HashSet<String>,
    log_mixture_e_value: f64,
    quarantined: bool,
}

impl ContextTreePortfolio {
    pub fn new(
        configurations: &[ContextNodeConfig],
        alpha: Rational,
        betting_rate: f64,
    ) -> Result<Self, ContextError> {
        if configurations.is_empty() { return Err(ContextError::EmptyInventory); }
        if configurations.len() > MAX_NODES { return Err(ContextError::TooManyNodes); }
        if alpha.numerator() == 0 || alpha.denominator() == 0 || alpha.is_one() {
            return Err(ContextError::InvalidAlpha);
        }
        if !betting_rate.is_finite() || betting_rate <= 0.0 || betting_rate >= 1.0 {
            return Err(ContextError::InvalidBettingRate);
        }

        let mut ids = HashMap::with_capacity(configurations.len());
        let mut nodes = Vec::with_capacity(configurations.len());
        let mut prior_sum: Option<Rational> = None;
        let mut total_features = 0usize;
        let mut root_index = None;
        let covariate_dimension = configurations[0].region.lower.len();
        let response_input_dimension = configurations[0].response_map.input_dimension();

        for config in configurations {
            validate_id(&config.node_id)?;
            if ids.contains_key(&config.node_id) { return Err(ContextError::DuplicateNodeId); }
            if config.depth > MAX_DEPTH { return Err(ContextError::InvalidDepth); }
            if config.region.lower.len() != covariate_dimension
                || config.covariate_map.input_dimension() != covariate_dimension
                || config.response_map.input_dimension() != response_input_dimension
            {
                return Err(ContextError::InputDimensionMismatch);
            }
            if config.prior_weight.numerator() == 0 || config.prior_weight.denominator() == 0 {
                return Err(ContextError::InvalidPrior);
            }

            let parent_index = match (&config.parent_id, config.depth) {
                (None, 0) => {
                    if root_index.is_some() { return Err(ContextError::InvalidRoot); }
                    if config.region.lower.iter().any(Option::is_some)
                        || config.region.upper.iter().any(Option::is_some)
                    {
                        return Err(ContextError::InvalidRoot);
                    }
                    None
                }
                (Some(parent_id), depth) if depth > 0 => {
                    let parent = *ids.get(parent_id).ok_or(ContextError::InvalidParent)?;
                    let parent_node = &nodes[parent];
                    if parent_node.depth + 1 != depth
                        || !parent_node.region.contains_region(&config.region)
                    {
                        return Err(ContextError::InvalidParent);
                    }
                    Some(parent)
                }
                _ => return Err(ContextError::InvalidParent),
            };

            let feature_dimension = config.response_map.feature_dimension()
                .checked_mul(config.covariate_map.feature_dimension())
                .ok_or(ContextError::FeatureDimensionOverflow)?;
            if feature_dimension > 1_024 { return Err(ContextError::NodeFeatureBudgetExceeded); }
            total_features = total_features.checked_add(feature_dimension)
                .ok_or(ContextError::FeatureDimensionOverflow)?;
            if total_features > 32_768 { return Err(ContextError::TotalFeatureBudgetExceeded); }

            let log_prior_weight = rational_log(config.prior_weight)?;
            prior_sum = Some(match prior_sum {
                Some(previous) => previous.checked_add(config.prior_weight)
                    .map_err(|_| ContextError::InvalidPrior)?,
                None => config.prior_weight,
            });

            let index = nodes.len();
            ids.insert(config.node_id.clone(), index);
            if config.depth == 0 { root_index = Some(index); }
            nodes.push(ContextNode {
                node_id: config.node_id.clone(),
                parent_index,
                children: Vec::new(),
                depth: config.depth,
                region: config.region.clone(),
                response_map: config.response_map.clone(),
                covariate_map: config.covariate_map.clone(),
                prior_weight: config.prior_weight,
                log_prior_weight,
                feature_sum: vec![0.0; feature_dimension],
                log_e_value: 0.0,
                selected_count: 0,
            });
        }

        let root_index = root_index.ok_or(ContextError::InvalidRoot)?;
        if prior_sum != Some(Rational::one()) { return Err(ContextError::PriorsDoNotSumToOne); }

        for child in 0..nodes.len() {
            if let Some(parent) = nodes[child].parent_index {
                nodes[parent].children.push(child);
            }
        }

        // Validate sibling boxes are pairwise disjoint. A boundary belongs to
        // the right-hand child because regions are [lower, upper).
        for node in &nodes {
            for left in 0..node.children.len() {
                for right in (left + 1)..node.children.len() {
                    let a = &nodes[node.children[left]].region;
                    let b = &nodes[node.children[right]].region;
                    if a.overlaps(b) { return Err(ContextError::OverlappingSiblings); }
                }
            }
        }

        let policy_sha256 = policy_digest(&nodes, alpha, betting_rate);
        let trace_sha256 = policy_sha256.clone();
        let log_threshold = rational_log_inverse(alpha)?;
        let mixture_leaf_base = nodes.len().next_power_of_two();
        let mut mixture_tree = vec![f64::NEG_INFINITY; mixture_leaf_base * 2];
        for (index, node) in nodes.iter().enumerate() {
            mixture_tree[mixture_leaf_base + index] = node.log_prior_weight;
        }
        for index in (1..mixture_leaf_base).rev() {
            mixture_tree[index] = log_add_exp(
                mixture_tree[index * 2],
                mixture_tree[index * 2 + 1],
            );
        }
        if !mixture_tree[1].is_finite() || mixture_tree[1].abs() > 1e-10 {
            return Err(ContextError::PriorsDoNotSumToOne);
        }

        Ok(Self {
            nodes,
            root_index,
            alpha,
            betting_rate,
            log_threshold,
            policy_sha256,
            trace_sha256,
            mixture_leaf_base,
            mixture_tree,
            observations_seen: 0,
            pair_ids: HashSet::new(),
            log_mixture_e_value: 0.0,
            quarantined: false,
        })
    }

    /// Evaluates only the unique root-to-leaf path selected by the current
    /// pre-pair covariate. Every active node's direction comes from its own
    /// past local feature history; inactive nodes retain their wealth.
    pub fn observe_pair(
        &mut self,
        sequence: u64,
        pair_id: &str,
        covariate: &[f64],
        x: &[f64],
        y: &[f64],
    ) -> Result<ContextTreeUpdate, ContextError> {
        if self.quarantined { return Err(ContextError::AlreadyQuarantined); }
        if pair_id.trim().is_empty() { return Err(ContextError::InvalidPairId); }
        if pair_id.len() > MAX_ID_BYTES { return Err(ContextError::PairIdTooLong); }
        if self.pair_ids.contains(pair_id) { return Err(ContextError::DuplicatePairId); }
        if self.pair_ids.len() >= MAX_PAIR_IDS { return Err(ContextError::PairIdInventoryFull); }
        let expected = self.observations_seen.checked_add(1).ok_or(ContextError::ArithmeticOverflow)?;
        if sequence != expected { return Err(ContextError::SequenceMismatch); }
        if x.is_empty() || x.len() != y.len()
            || covariate.len() != self.nodes[self.root_index].region.lower.len()
            || x.len() != self.nodes[self.root_index].response_map.input_dimension()
        {
            return Err(ContextError::InputDimensionMismatch);
        }
        if covariate.iter().chain(x).chain(y).any(|v| !v.is_finite()) {
            return Err(ContextError::NonFiniteInput);
        }

        let mut path = vec![self.root_index];
        if !self.nodes[self.root_index].region.contains(covariate) {
            return Err(ContextError::InputDimensionMismatch);
        }
        let mut current = self.root_index;
        loop {
            let matching = self.nodes[current].children.iter()
                .copied()
                .find(|child| self.nodes[*child].region.contains(covariate));
            match matching {
                Some(child) => { path.push(child); current = child; }
                None => break,
            }
        }

        let mut staged = Vec::with_capacity(path.len());
        let mut total_coordinates = 0usize;
        for index in &path {
            let node = &self.nodes[*index];
            let direction_norm = l2_norm(&node.feature_sum);
            if !direction_norm.is_finite() { return Err(ContextError::ArithmeticOverflow); }
            let direction = if direction_norm > 0.0 {
                node.feature_sum.iter().map(|v| *v / direction_norm).collect::<Vec<_>>()
            } else {
                vec![0.0; node.feature_sum.len()]
            };

            // Every feature transform uses the frozen node maps and the same
            // map on X and Y; node membership was fixed before either response.
            let phi_x = node.response_map.transform(x)?;
            let phi_y = node.response_map.transform(y)?;
            let psi = node.covariate_map.transform(covariate)?;
            let response_dimension = node.response_map.feature_dimension();
            let mut difference = vec![0.0; node.feature_sum.len()];
            for (z_index, z_feature) in psi.iter().enumerate() {
                for (r_index, (y_feature, x_feature)) in phi_y.iter().zip(&phi_x).enumerate() {
                    difference[z_index * response_dimension + r_index] =
                        0.5 * z_feature * (y_feature - x_feature);
                }
            }
            let norm = l2_norm(&difference);
            if !norm.is_finite() { return Err(ContextError::ArithmeticOverflow); }
            if norm > 1.0 + NORM_TOLERANCE { return Err(ContextError::ObservationNormExceeded); }
            let projected = dot(&direction, &difference)?;
            if !(-1.0 - NORM_TOLERANCE..=1.0 + NORM_TOLERANCE).contains(&projected) {
                return Err(ContextError::ArithmeticOverflow);
            }
            let factor = 1.0 + self.betting_rate * projected;
            if !factor.is_finite() || factor <= 0.0 { return Err(ContextError::InvalidFactor); }
            let log_factor = factor.ln();
            let log_e_value = node.log_e_value + log_factor;
            if !log_e_value.is_finite() { return Err(ContextError::ArithmeticOverflow); }

            let mut feature_sum = node.feature_sum.clone();
            for (sum, value) in feature_sum.iter_mut().zip(&difference) {
                *sum += value;
                if !sum.is_finite() { return Err(ContextError::ArithmeticOverflow); }
            }
            let coordinates = phi_x.len() + phi_y.len() + psi.len() + difference.len();
            total_coordinates = total_coordinates.checked_add(coordinates)
                .ok_or(ContextError::ArithmeticOverflow)?;
            let next_selected_count = node.selected_count.checked_add(1)
                .ok_or(ContextError::ArithmeticOverflow)?;
            staged.push(StagedNode {
                index: *index,
                feature_sum,
                log_e_value,
                factor,
                log_factor,
                feature_coordinates: coordinates,
                next_selected_count,
            });
        }

        // Update a log-sum-exp segment tree at only the active node leaves.
        // Every fixed-prior component remains represented, whether active or
        // inactive. Dirty ancestors are recomputed deepest-first.
        let mut staged_tree_values = HashMap::new();
        let mut dirty = HashSet::new();
        for item in &staged {
            let node = &self.nodes[item.index];
            let mut tree_index = self.mixture_leaf_base + item.index;
            staged_tree_values.insert(tree_index, node.log_prior_weight + item.log_e_value);
            tree_index /= 2;
            while tree_index > 0 {
                dirty.insert(tree_index);
                tree_index /= 2;
            }
        }
        let mut dirty_indices = dirty.into_iter().collect::<Vec<_>>();
        dirty_indices.sort_unstable_by(|left, right| right.cmp(left));
        for index in dirty_indices {
            let left = staged_tree_values.get(&(index * 2))
                .copied()
                .unwrap_or(self.mixture_tree[index * 2]);
            let right = staged_tree_values.get(&(index * 2 + 1))
                .copied()
                .unwrap_or(self.mixture_tree[index * 2 + 1]);
            staged_tree_values.insert(index, log_add_exp(left, right));
        }
        let log_mixture = *staged_tree_values.get(&1).ok_or(ContextError::ArithmeticOverflow)?;
        if !log_mixture.is_finite() {
            return Err(ContextError::ArithmeticOverflow);
        }

        let mut posterior_sum = 0.0;
        let mut updates = Vec::with_capacity(staged.len());
        for item in &staged {
            let node = &self.nodes[item.index];
            let posterior = (node.log_prior_weight + item.log_e_value - log_mixture).exp();
            if !posterior.is_finite() || !(0.0..=1.0 + POSTERIOR_TOLERANCE).contains(&posterior) {
                return Err(ContextError::PosteriorNormalizationFailure);
            }
            posterior_sum += posterior;
            updates.push(ContextNodeUpdate {
                node_id: node.node_id.clone(),
                depth: node.depth,
                prior_weight: node.prior_weight,
                posterior_weight: posterior,
                log_e_value: item.log_e_value,
                factor: item.factor,
                log_factor: item.log_factor,
                feature_coordinates_evaluated: item.feature_coordinates,
            });
        }
        if !posterior_sum.is_finite() || posterior_sum > 1.0 + POSTERIOR_TOLERANCE {
            return Err(ContextError::PosteriorNormalizationFailure);
        }

        let path_ids = path.iter().map(|i| self.nodes[*i].node_id.clone()).collect::<Vec<_>>();
        let trace = extend_trace(&self.trace_sha256, sequence, pair_id, &staged, &self.nodes);
        let quarantined = log_mixture >= self.log_threshold;

        // Commit all active node states and the one global mixture together.
        for item in staged {
            let node = &mut self.nodes[item.index];
            node.feature_sum = item.feature_sum;
            node.log_e_value = item.log_e_value;
            node.selected_count = item.next_selected_count;
        }
        self.observations_seen = sequence;
        self.pair_ids.insert(pair_id.to_string());
        for (index, value) in staged_tree_values {
            self.mixture_tree[index] = value;
        }
        self.log_mixture_e_value = log_mixture;
        self.quarantined = quarantined;
        self.trace_sha256 = trace.clone();

        Ok(ContextTreeUpdate {
            sequence,
            pair_id: pair_id.to_string(),
            active_path: path_ids,
            nodes: updates,
            log_mixture_e_value: log_mixture,
            log_threshold: self.log_threshold,
            quarantined,
            trace_sha256: trace,
            total_feature_coordinates_evaluated: total_coordinates,
        })
    }

    pub fn observations_seen(&self) -> u64 { self.observations_seen }
    pub fn log_mixture_e_value(&self) -> f64 { self.log_mixture_e_value }
    pub fn quarantined(&self) -> bool { self.quarantined }
    pub fn selected_counts(&self) -> Vec<(String, u64)> {
        self.nodes.iter().map(|n| (n.node_id.clone(), n.selected_count)).collect()
    }
    pub fn policy_sha256(&self) -> &str { &self.policy_sha256 }
    pub fn trace_sha256(&self) -> &str { &self.trace_sha256 }
    pub fn alpha(&self) -> Rational { self.alpha }
}

fn validate_id(id: &str) -> Result<(), ContextError> {
    if id.trim().is_empty() || id.len() > MAX_ID_BYTES { Err(ContextError::InvalidNodeId) } else { Ok(()) }
}
fn encode_bound(hasher: &mut Sha256, value: Option<f64>) {
    match value {
        Some(v) => { hasher.update([1]); hasher.update(if v == 0.0 { 0.0f64 } else { v }.to_bits().to_be_bytes()); }
        None => hasher.update([0]),
    }
}
fn rational_log(value: Rational) -> Result<f64, ContextError> {
    let result = (value.numerator() as f64).ln() - (value.denominator() as f64).ln();
    if result.is_finite() { Ok(result) } else { Err(ContextError::ArithmeticOverflow) }
}
fn rational_log_inverse(value: Rational) -> Result<f64, ContextError> {
    let result = (value.denominator() as f64).ln() - (value.numerator() as f64).ln();
    if result.is_finite() { Ok(result) } else { Err(ContextError::ArithmeticOverflow) }
}
fn l2_norm(values: &[f64]) -> f64 { values.iter().map(|x| x * x).sum::<f64>().sqrt() }
fn dot(left: &[f64], right: &[f64]) -> Result<f64, ContextError> {
    if left.len() != right.len() { return Err(ContextError::InputDimensionMismatch); }
    let result = left.iter().zip(right).map(|(a,b)| a*b).sum::<f64>();
    if result.is_finite() { Ok(result) } else { Err(ContextError::ArithmeticOverflow) }
}
fn log_add_exp(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY { return b; }
    if b == f64::NEG_INFINITY { return a; }
    let m = a.max(b);
    m + ((a-m).exp() + (b-m).exp()).ln()
}
#[cfg(test)]
fn log_sum_exp(values: &[f64]) -> Result<f64, ContextError> {
    if values.is_empty() || values.iter().any(|x| !x.is_finite()) {
        return Err(ContextError::ArithmeticOverflow);
    }
    let mut result = f64::NEG_INFINITY;
    for value in values { result = log_add_exp(result, *value); }
    if result.is_finite() { Ok(result) } else { Err(ContextError::ArithmeticOverflow) }
}
fn policy_digest(nodes: &[ContextNode], alpha: Rational, lambda: f64) -> String {
    let mut h = Sha256::new();
    h.update(POLICY_DOMAIN);
    h.update(alpha.numerator().to_be_bytes());
    h.update(alpha.denominator().to_be_bytes());
    h.update(lambda.to_bits().to_be_bytes());
    h.update((nodes.len() as u64).to_be_bytes());
    for node in nodes {
        h.update((node.node_id.len() as u64).to_be_bytes());
        h.update(node.node_id.as_bytes());
        h.update((node.depth as u64).to_be_bytes());
        h.update((node.parent_index.map_or(u64::MAX, |x| x as u64)).to_be_bytes());
        h.update(node.prior_weight.numerator().to_be_bytes());
        h.update(node.prior_weight.denominator().to_be_bytes());
        node.region.encode(&mut h);
        h.update(node.response_map.map_sha256().as_bytes());
        encode_frequency_map(&mut h, &node.response_map);
        h.update(node.covariate_map.map_sha256().as_bytes());
        encode_frequency_map(&mut h, &node.covariate_map);
    }
    hex(&h.finalize())
}
fn encode_frequency_map(hasher: &mut Sha256, map: &FourierFeatureMap) {
    hasher.update((map.input_dimension() as u64).to_be_bytes());
    hasher.update((map.frequency_count() as u64).to_be_bytes());
    for frequency in map.frequencies() {
        hasher.update((frequency.len() as u64).to_be_bytes());
        for value in frequency {
            hasher.update((if *value == 0.0 { 0.0f64 } else { *value }).to_bits().to_be_bytes());
        }
    }
}
fn extend_trace(prior: &str, sequence: u64, pair_id: &str, staged: &[StagedNode], nodes: &[ContextNode]) -> String {
    let mut h = Sha256::new();
    h.update(TRACE_DOMAIN);
    h.update(prior.as_bytes());
    h.update(sequence.to_be_bytes());
    h.update((pair_id.len() as u64).to_be_bytes());
    h.update(pair_id.as_bytes());
    for item in staged {
        let node = &nodes[item.index];
        h.update((node.node_id.len() as u64).to_be_bytes());
        h.update(node.node_id.as_bytes());
        h.update(item.factor.to_bits().to_be_bytes());
        h.update(item.log_factor.to_bits().to_be_bytes());
    }
    hex(&h.finalize())
}
fn hex(bytes: &[u8]) -> String {
    const D: &[u8;16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len()*2);
    for b in bytes { out.push(D[(b>>4) as usize] as char); out.push(D[(b&15) as usize] as char); }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const C: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const D: &str = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
    const E: &str = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";
    const F: &str = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";

    fn node(id: &str, parent: Option<&str>, depth: usize, region: ContextRegion, prior: Rational, response: &str, covariate: &str, freq: f64) -> ContextNodeConfig {
        ContextNodeConfig {
            node_id: id.into(),
            parent_id: parent.map(str::to_string),
            depth,
            region,
            response_map: FourierFeatureMap::new(1, vec![vec![freq]], response.into()).unwrap(),
            covariate_map: FourierFeatureMap::new(1, vec![vec![0.5]], covariate.into()).unwrap(),
            prior_weight: prior,
        }
    }
    fn tree() -> ContextTreePortfolio {
        let configs = vec![
            node("root", None, 0, ContextRegion::root(1).unwrap(), Rational::new(1,2).unwrap(), A,B,0.5),
            node("left", Some("root"), 1, ContextRegion::new(vec![None],vec![Some(0.0)]).unwrap(), Rational::new(1,4).unwrap(), C,D,1.0),
            node("right",Some("root"),1,ContextRegion::new(vec![Some(0.0)],vec![None]).unwrap(),Rational::new(1,4).unwrap(),E,F,2.0),
        ];
        ContextTreePortfolio::new(&configs, Rational::probability(1,20).unwrap(), 0.5).unwrap()
    }

    #[test]
    fn root_to_leaf_routes_once_and_half_open_boundary_goes_right() {
        let mut m = tree();
        let update = m.observe_pair(1,"zero",&[0.0],&[0.0],&[0.2]).unwrap();
        assert_eq!(update.active_path, vec!["root","right"]);
        assert_eq!(update.nodes.len(),2);
        assert_eq!(update.total_feature_coordinates_evaluated,20);
    }

    #[test]
    fn equal_pairs_preserve_unit_mixture_and_fixed_prior_posteriors() {
        let mut m=tree();
        for t in 1..=32 {
            let u=m.observe_pair(t,&format!("equal-{t}"),&[0.25],&[0.5],&[0.5]).unwrap();
            assert!(u.log_mixture_e_value.abs()<1e-10);
            assert!(u.nodes.iter().map(|n|n.posterior_weight).sum::<f64>() <= 1.0 + 1e-10);
            assert_eq!(u.active_path.len(),2);
        }
        assert_eq!(m.selected_counts().iter().map(|x|x.1).sum::<u64>(),64);
    }

    #[test]
    fn segment_tree_mixture_matches_direct_fixed_prior_sum_after_each_update() {
        let mut monitor = tree();
        for sequence in 1..=96u64 {
            let z = if sequence % 3 == 0 { 0.5 } else { -0.5 };
            let y = if sequence % 4 == 0 { std::f64::consts::TAU } else { 0.0 };
            let update = monitor.observe_pair(
                sequence,
                &format!("mixture-{sequence}"),
                &[z],
                &[0.0],
                &[y],
            ).unwrap();
            let direct = monitor.nodes.iter()
                .map(|node| node.log_prior_weight + node.log_e_value)
                .collect::<Vec<_>>();
            let direct_total = log_sum_exp(&direct).unwrap();
            assert!((direct_total - update.log_mixture_e_value).abs() < 1e-10);
            assert!((monitor.mixture_tree[1] - direct_total).abs() < 1e-10);
        }
    }

    #[test]
    fn sparse_right_context_updates_only_its_leaf_history() {
        let mut m=tree();
        for t in 1..=64 {
            let z=if t%2==0 {0.5} else {-0.5};
            let y=if z>0.0 {std::f64::consts::TAU} else {0.0};
            let u=m.observe_pair(t,&format!("sparse-{t}"),&[z],&[0.0],&[y]).unwrap();
            assert!(u.total_feature_coordinates_evaluated>0);
        }
        let counts=m.selected_counts();
        assert_eq!(counts[0].1,64);
        assert_eq!(counts[1].1,32);
        assert_eq!(counts[2].1,32);
    }

    #[test]
    fn invalid_sibling_overlap_and_prior_sum_fail_closed() {
        let root=ContextRegion::root(1).unwrap();
        let configs=vec![
            node("root",None,0,root.clone(),Rational::new(1,2).unwrap(),A,B,0.5),
            node("left",Some("root"),1,ContextRegion::new(vec![None],vec![Some(0.5)]).unwrap(),Rational::new(1,4).unwrap(),C,D,1.0),
            node("overlap",Some("root"),1,ContextRegion::new(vec![Some(0.25)],vec![None]).unwrap(),Rational::new(1,4).unwrap(),E,F,2.0),
        ];
        assert_eq!(
            ContextTreePortfolio::new(&configs,Rational::probability(1,20).unwrap(),0.5).err(),
            Some(ContextError::OverlappingSiblings)
        );
    }
}
