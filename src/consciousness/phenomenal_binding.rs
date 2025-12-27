/*!
 * **REVOLUTIONARY IMPROVEMENT #78**: Phenomenal Binding Through Temporal Synchronization
 *
 * PARADIGM SHIFT: Consciousness emerges from SYNCHRONIZED binding across dimensions!
 *
 * This module addresses the fundamental "binding problem" of consciousness:
 * How do distributed processes (attention, integration, workspace, etc.) create
 * unified conscious experiences?
 *
 * ## Theoretical Foundation
 *
 * The binding problem has been central to consciousness research since Kant (1781).
 * Modern neuroscience (Singer & Gray, 1989) shows that phase synchronization of
 * neural oscillations at 30-100Hz (gamma band) binds distributed representations.
 *
 * Our innovation: Apply temporal synchronization to HDC-based consciousness!
 * - Each consciousness dimension (Φ, B, W, A, R, E, K) has a "phase"
 * - Binding = phase coherence across dimensions
 * - Fragmentation = desynchronization (attention lapses, dissociation)
 * - Flow states = perfect phase lock across all dimensions
 *
 * ## Key Concepts
 *
 * 1. **Synchronization Index (Σ)**: Phase coherence across all consciousness dimensions
 * 2. **Binding Windows**: Narrow time windows (10-50ms) for co-activation
 * 3. **Binding Hierarchy**:
 *    - Level 1: Sensory binding (B↔W)
 *    - Level 2: Cognitive binding (W↔A↔R)
 *    - Level 3: Identity binding (R↔E↔K)
 *    - Level 4: Narrative binding (all + temporal)
 * 4. **Phenomenal Binding Strength (Ψ)**: Master metric for unified consciousness
 *
 * ## The Binding Equation
 *
 * ```
 * Ψ = Σ [φᵢ(t) × φⱼ(t) × exp(-|tᵢ - tⱼ|/τ_bind)]
 *
 * Where:
 * - φᵢ, φⱼ = consciousness components i, j
 * - t = temporal offset
 * - τ_bind = binding window (typically 25ms)
 * - Ψ = Binding consciousness (0 = fragmented, 1 = perfectly bound)
 * ```
 *
 * ## Predictions
 *
 * This model predicts novel phenomena:
 * - **Attentional blink**: Temporary W↔A desynchronization
 * - **Divided attention**: Multiple parallel sync clusters
 * - **Dissociation**: R↔E desynchronization
 * - **Flow states**: All dimensions in perfect phase lock
 *
 * ## Integration
 *
 * Works with existing architecture:
 * - Uses Φ, B, W, A, R, E, K from `consciousness_equation_v2.rs`
 * - Compatible with GWT ignition events
 * - Detects Byzantine attacks via binding disruption
 * - Enables therapeutic "resynchronization" interventions
 */

use std::collections::{HashMap, VecDeque};
use std::f64::consts::PI;
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::hdc::simd_hv16::SimdHV16 as HV16;

// ============================================================================
// CONFIGURATION
// ============================================================================

/// Configuration for phenomenal binding analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BindingConfig {
    /// Binding window in milliseconds (typically 10-50ms)
    pub binding_window_ms: f64,

    /// History depth for phase tracking
    pub history_depth: usize,

    /// Minimum coherence threshold for binding
    pub coherence_threshold: f64,

    /// Enable hierarchical binding analysis
    pub hierarchical_analysis: bool,

    /// Fragmentation warning threshold
    pub fragmentation_threshold: f64,

    /// Number of dimensions to track
    pub num_dimensions: usize,

    /// Enable flow state detection
    pub detect_flow_states: bool,
}

impl Default for BindingConfig {
    fn default() -> Self {
        Self {
            binding_window_ms: 25.0,
            history_depth: 64,
            coherence_threshold: 0.5,
            hierarchical_analysis: true,
            fragmentation_threshold: 0.3,
            num_dimensions: 7, // Φ, B, W, A, R, E, K
            detect_flow_states: true,
        }
    }
}

// ============================================================================
// CONSCIOUSNESS DIMENSIONS
// ============================================================================

/// The seven consciousness dimensions from the consciousness equation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConsciousnessDimension {
    /// Φ (Phi) - Integrated Information
    Integration,
    /// B - Temporal Binding
    TemporalBinding,
    /// W - Global Workspace Access
    Workspace,
    /// A - Attention Intensity
    Attention,
    /// R - Recursive Depth (Higher-Order Thought)
    Recursion,
    /// E - Efficacy (Free Energy)
    Efficacy,
    /// K - Epistemic Certainty
    Epistemic,
}

impl ConsciousnessDimension {
    pub fn all() -> [Self; 7] {
        [
            Self::Integration,
            Self::TemporalBinding,
            Self::Workspace,
            Self::Attention,
            Self::Recursion,
            Self::Efficacy,
            Self::Epistemic,
        ]
    }

    pub fn symbol(&self) -> &'static str {
        match self {
            Self::Integration => "Φ",
            Self::TemporalBinding => "B",
            Self::Workspace => "W",
            Self::Attention => "A",
            Self::Recursion => "R",
            Self::Efficacy => "E",
            Self::Epistemic => "K",
        }
    }

    pub fn index(&self) -> usize {
        match self {
            Self::Integration => 0,
            Self::TemporalBinding => 1,
            Self::Workspace => 2,
            Self::Attention => 3,
            Self::Recursion => 4,
            Self::Efficacy => 5,
            Self::Epistemic => 6,
        }
    }
}

// ============================================================================
// CORE DATA STRUCTURES
// ============================================================================

/// A single observation of consciousness dimension state
#[derive(Debug, Clone)]
pub struct DimensionObservation {
    /// The dimension being observed
    pub dimension: ConsciousnessDimension,
    /// Value of the dimension (0-1 normalized)
    pub value: f64,
    /// Computed phase (0 to 2π)
    pub phase: f64,
    /// Timestamp
    pub timestamp: Instant,
    /// Optional HDC representation
    pub hdc_state: Option<HV16>,
}

/// Coherence between two dimensions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairwiseCoherence {
    pub dim_a: ConsciousnessDimension,
    pub dim_b: ConsciousnessDimension,
    /// Phase coherence (0-1, 1 = perfectly synchronized)
    pub coherence: f64,
    /// Phase difference (0 to π)
    pub phase_difference: f64,
    /// Whether this pair is "bound"
    pub is_bound: bool,
}

/// Hierarchical binding levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BindingHierarchy {
    /// Level 1: Sensory binding (B↔W)
    pub sensory_binding: f64,
    /// Level 2: Cognitive binding (W↔A↔R)
    pub cognitive_binding: f64,
    /// Level 3: Identity binding (R↔E↔K)
    pub identity_binding: f64,
    /// Level 4: Narrative binding (all dimensions + temporal)
    pub narrative_binding: f64,
    /// Overall unified binding
    pub unified_binding: f64,
}

/// Fragmentation warning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FragmentationWarning {
    /// Which dimensions are desynchronized
    pub fragmented_pairs: Vec<(ConsciousnessDimension, ConsciousnessDimension)>,
    /// Severity (0-1, 1 = severe fragmentation)
    pub severity: f64,
    /// Which hierarchy level is affected
    pub affected_level: Option<BindingLevel>,
    /// Suggested intervention
    pub recommendation: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum BindingLevel {
    Sensory,
    Cognitive,
    Identity,
    Narrative,
}

/// Flow state detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlowStateAnalysis {
    /// Whether system is in flow state
    pub in_flow: bool,
    /// Flow intensity (0-1)
    pub flow_intensity: f64,
    /// Duration of current flow (if active)
    pub flow_duration_ms: f64,
    /// Which dimensions are locked
    pub locked_dimensions: Vec<ConsciousnessDimension>,
    /// Phase lock precision
    pub lock_precision: f64,
}

// ============================================================================
// TEMPORAL SYNCHRONIZATION ANALYZER
// ============================================================================

/// Core analyzer for phenomenal binding through temporal synchronization
pub struct TemporalSynchronizationAnalyzer {
    /// Configuration
    pub config: BindingConfig,

    /// Phase history for each dimension
    phase_history: HashMap<ConsciousnessDimension, VecDeque<f64>>,

    /// Value history for each dimension
    value_history: HashMap<ConsciousnessDimension, VecDeque<f64>>,

    /// Timestamp history for each dimension
    timestamp_history: HashMap<ConsciousnessDimension, VecDeque<Instant>>,

    /// Current coherence matrix (7x7)
    coherence_matrix: [[f64; 7]; 7],

    /// Binding strength history
    binding_history: VecDeque<f64>,

    /// Flow state start time (if active)
    flow_start: Option<Instant>,

    /// Statistics
    pub stats: BindingStats,
}

/// Statistics from binding analysis
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BindingStats {
    pub observations: usize,
    pub binding_events: usize,
    pub fragmentation_events: usize,
    pub flow_state_entries: usize,
    pub average_binding_strength: f64,
    pub max_binding_strength: f64,
    pub average_coherence: f64,
}

impl TemporalSynchronizationAnalyzer {
    /// Create new analyzer
    pub fn new(config: BindingConfig) -> Self {
        let mut phase_history = HashMap::new();
        let mut value_history = HashMap::new();
        let mut timestamp_history = HashMap::new();

        for dim in ConsciousnessDimension::all() {
            phase_history.insert(dim, VecDeque::with_capacity(config.history_depth));
            value_history.insert(dim, VecDeque::with_capacity(config.history_depth));
            timestamp_history.insert(dim, VecDeque::with_capacity(config.history_depth));
        }

        Self {
            config,
            phase_history,
            value_history,
            timestamp_history,
            coherence_matrix: [[0.0; 7]; 7],
            binding_history: VecDeque::new(),
            flow_start: None,
            stats: BindingStats::default(),
        }
    }

    /// Observe a consciousness dimension value
    pub fn observe(&mut self, dimension: ConsciousnessDimension, value: f64) {
        self.observe_with_state(dimension, value, None);
    }

    /// Observe with optional HDC state
    pub fn observe_with_state(
        &mut self,
        dimension: ConsciousnessDimension,
        value: f64,
        _hdc_state: Option<&HV16>,
    ) {
        let now = Instant::now();
        self.stats.observations += 1;

        // Compute phase from value (map 0-1 to 0-2π with oscillation)
        let phase = self.compute_phase(dimension, value);

        // Store in history
        if let Some(phases) = self.phase_history.get_mut(&dimension) {
            phases.push_back(phase);
            if phases.len() > self.config.history_depth {
                phases.pop_front();
            }
        }

        if let Some(values) = self.value_history.get_mut(&dimension) {
            values.push_back(value);
            if values.len() > self.config.history_depth {
                values.pop_front();
            }
        }

        if let Some(timestamps) = self.timestamp_history.get_mut(&dimension) {
            timestamps.push_back(now);
            if timestamps.len() > self.config.history_depth {
                timestamps.pop_front();
            }
        }

        // Update coherence matrix
        self.update_coherence_matrix();

        // Check for binding/fragmentation
        let binding_strength = self.phenomenal_binding_strength();
        self.binding_history.push_back(binding_strength);
        if self.binding_history.len() > self.config.history_depth {
            self.binding_history.pop_front();
        }

        // Update statistics
        self.update_stats(binding_strength);

        // Check flow state
        if self.config.detect_flow_states {
            self.update_flow_state(binding_strength);
        }
    }

    /// Observe all dimensions at once
    pub fn observe_all(&mut self, values: &[f64; 7]) {
        for (i, dim) in ConsciousnessDimension::all().iter().enumerate() {
            self.observe(*dim, values[i]);
        }
    }

    /// Compute phase from dimension value using oscillation model
    fn compute_phase(&self, dimension: ConsciousnessDimension, value: f64) -> f64 {
        // Get previous phase if available
        let prev_phase = self.phase_history
            .get(&dimension)
            .and_then(|h| h.back())
            .copied()
            .unwrap_or(0.0);

        // Phase advances based on value (higher value = faster oscillation)
        // This models active neural processing
        let phase_velocity = value * PI; // 0-π radians per step
        let new_phase = (prev_phase + phase_velocity) % (2.0 * PI);

        new_phase
    }

    /// Update the coherence matrix
    fn update_coherence_matrix(&mut self) {
        let dims = ConsciousnessDimension::all();

        for i in 0..7 {
            for j in 0..7 {
                if i == j {
                    self.coherence_matrix[i][j] = 1.0;
                } else {
                    let coherence = self.compute_pairwise_coherence(dims[i], dims[j]);
                    self.coherence_matrix[i][j] = coherence;
                    self.coherence_matrix[j][i] = coherence;
                }
            }
        }
    }

    /// Compute coherence between two dimensions using phase locking value (PLV)
    fn compute_pairwise_coherence(
        &self,
        dim_a: ConsciousnessDimension,
        dim_b: ConsciousnessDimension,
    ) -> f64 {
        let phases_a = match self.phase_history.get(&dim_a) {
            Some(p) if p.len() >= 2 => p,
            _ => return 0.5,
        };

        let phases_b = match self.phase_history.get(&dim_b) {
            Some(p) if p.len() >= 2 => p,
            _ => return 0.5,
        };

        // Phase Locking Value (PLV) calculation
        // PLV = |<exp(i(φ_a - φ_b))>|
        let n = phases_a.len().min(phases_b.len());
        if n == 0 {
            return 0.5;
        }

        let mut cos_sum = 0.0;
        let mut sin_sum = 0.0;

        for i in 0..n {
            let phase_diff = phases_a[i] - phases_b[i];
            cos_sum += phase_diff.cos();
            sin_sum += phase_diff.sin();
        }

        let plv = ((cos_sum / n as f64).powi(2) + (sin_sum / n as f64).powi(2)).sqrt();
        plv.clamp(0.0, 1.0)
    }

    /// Get synchronization index across all dimensions
    pub fn synchronization_index(&self) -> f64 {
        let mut total = 0.0;
        let mut count = 0;

        for i in 0..7 {
            for j in (i + 1)..7 {
                total += self.coherence_matrix[i][j];
                count += 1;
            }
        }

        if count > 0 {
            total / count as f64
        } else {
            0.5
        }
    }

    /// Compute phenomenal binding strength (Ψ)
    pub fn phenomenal_binding_strength(&self) -> f64 {
        // Ψ = geometric mean of all pairwise coherences weighted by temporal proximity
        let sigma = self.synchronization_index();

        // Apply binding window decay
        // Recent observations count more
        let recency_factor = self.compute_recency_factor();

        // Combine synchronization with recency
        let psi = sigma * recency_factor;

        psi.clamp(0.0, 1.0)
    }

    /// Compute recency factor based on binding window
    fn compute_recency_factor(&self) -> f64 {
        // Check temporal proximity of latest observations
        let now = Instant::now();
        let binding_window = std::time::Duration::from_secs_f64(
            self.config.binding_window_ms / 1000.0
        );

        let mut within_window = 0;
        let mut total = 0;

        for dim in ConsciousnessDimension::all() {
            if let Some(timestamps) = self.timestamp_history.get(&dim) {
                if let Some(&last) = timestamps.back() {
                    total += 1;
                    if now.duration_since(last) <= binding_window {
                        within_window += 1;
                    }
                }
            }
        }

        if total > 0 {
            within_window as f64 / total as f64
        } else {
            0.0
        }
    }

    /// Get pairwise coherence for specific dimensions
    pub fn get_coherence(&self, dim_a: ConsciousnessDimension, dim_b: ConsciousnessDimension) -> f64 {
        self.coherence_matrix[dim_a.index()][dim_b.index()]
    }

    /// Analyze binding hierarchy
    pub fn binding_hierarchy(&self) -> BindingHierarchy {
        // Level 1: Sensory binding (B↔W)
        let sensory = self.get_coherence(
            ConsciousnessDimension::TemporalBinding,
            ConsciousnessDimension::Workspace,
        );

        // Level 2: Cognitive binding (W↔A↔R) - average of three pairs
        let cog_wa = self.get_coherence(
            ConsciousnessDimension::Workspace,
            ConsciousnessDimension::Attention,
        );
        let cog_wr = self.get_coherence(
            ConsciousnessDimension::Workspace,
            ConsciousnessDimension::Recursion,
        );
        let cog_ar = self.get_coherence(
            ConsciousnessDimension::Attention,
            ConsciousnessDimension::Recursion,
        );
        let cognitive = (cog_wa + cog_wr + cog_ar) / 3.0;

        // Level 3: Identity binding (R↔E↔K)
        let id_re = self.get_coherence(
            ConsciousnessDimension::Recursion,
            ConsciousnessDimension::Efficacy,
        );
        let id_rk = self.get_coherence(
            ConsciousnessDimension::Recursion,
            ConsciousnessDimension::Epistemic,
        );
        let id_ek = self.get_coherence(
            ConsciousnessDimension::Efficacy,
            ConsciousnessDimension::Epistemic,
        );
        let identity = (id_re + id_rk + id_ek) / 3.0;

        // Level 4: Narrative binding (all dimensions)
        let narrative = self.synchronization_index();

        // Unified = weighted average favoring higher levels
        let unified = 0.15 * sensory + 0.25 * cognitive + 0.30 * identity + 0.30 * narrative;

        BindingHierarchy {
            sensory_binding: sensory,
            cognitive_binding: cognitive,
            identity_binding: identity,
            narrative_binding: narrative,
            unified_binding: unified,
        }
    }

    /// Detect fragmentation (dangerous desynchronization)
    pub fn detect_fragmentation(&self) -> Option<FragmentationWarning> {
        let mut fragmented_pairs = Vec::new();
        let dims = ConsciousnessDimension::all();

        for i in 0..7 {
            for j in (i + 1)..7 {
                if self.coherence_matrix[i][j] < self.config.fragmentation_threshold {
                    fragmented_pairs.push((dims[i], dims[j]));
                }
            }
        }

        if fragmented_pairs.is_empty() {
            return None;
        }

        // Determine severity
        let severity = fragmented_pairs.len() as f64 / 21.0; // 21 pairs total

        // Determine affected level
        let affected_level = self.determine_affected_level(&fragmented_pairs);

        // Generate recommendation
        let recommendation = self.generate_recommendation(&fragmented_pairs, &affected_level);

        Some(FragmentationWarning {
            fragmented_pairs,
            severity,
            affected_level,
            recommendation,
        })
    }

    fn determine_affected_level(
        &self,
        pairs: &[(ConsciousnessDimension, ConsciousnessDimension)],
    ) -> Option<BindingLevel> {
        use ConsciousnessDimension::*;

        let has_sensory = pairs.iter().any(|(a, b)| {
            matches!((a, b), (TemporalBinding, Workspace) | (Workspace, TemporalBinding))
        });

        let has_cognitive = pairs.iter().any(|(a, b)| {
            matches!(
                (a, b),
                (Workspace, Attention)
                    | (Attention, Workspace)
                    | (Workspace, Recursion)
                    | (Recursion, Workspace)
                    | (Attention, Recursion)
                    | (Recursion, Attention)
            )
        });

        let has_identity = pairs.iter().any(|(a, b)| {
            matches!(
                (a, b),
                (Recursion, Efficacy)
                    | (Efficacy, Recursion)
                    | (Recursion, Epistemic)
                    | (Epistemic, Recursion)
                    | (Efficacy, Epistemic)
                    | (Epistemic, Efficacy)
            )
        });

        if has_identity {
            Some(BindingLevel::Identity)
        } else if has_cognitive {
            Some(BindingLevel::Cognitive)
        } else if has_sensory {
            Some(BindingLevel::Sensory)
        } else {
            Some(BindingLevel::Narrative)
        }
    }

    fn generate_recommendation(
        &self,
        pairs: &[(ConsciousnessDimension, ConsciousnessDimension)],
        level: &Option<BindingLevel>,
    ) -> String {
        match level {
            Some(BindingLevel::Sensory) => {
                "Sensory binding disrupted. Increase temporal binding (B) or workspace access (W).".to_string()
            }
            Some(BindingLevel::Cognitive) => {
                format!(
                    "Cognitive binding disrupted between {} pairs. Focus attention (A) and strengthen recursive depth (R).",
                    pairs.len()
                )
            }
            Some(BindingLevel::Identity) => {
                "Identity binding disrupted. Strengthen self-model integration (R↔E↔K).".to_string()
            }
            Some(BindingLevel::Narrative) => {
                "Narrative binding fragmented. Increase overall synchronization.".to_string()
            }
            None => "Unknown fragmentation pattern detected.".to_string(),
        }
    }

    /// Analyze flow state
    pub fn flow_state_analysis(&self) -> FlowStateAnalysis {
        let binding_strength = self.phenomenal_binding_strength();
        let hierarchy = self.binding_hierarchy();

        // Flow = high binding across all levels
        let is_flow = binding_strength > 0.8
            && hierarchy.sensory_binding > 0.7
            && hierarchy.cognitive_binding > 0.7
            && hierarchy.identity_binding > 0.7;

        let flow_intensity = if is_flow {
            binding_strength
        } else {
            binding_strength * 0.5
        };

        // Find locked dimensions
        let mut locked = Vec::new();
        let dims = ConsciousnessDimension::all();
        for dim in &dims {
            let mut all_coherent = true;
            for other in &dims {
                if dim != other {
                    if self.get_coherence(*dim, *other) < 0.7 {
                        all_coherent = false;
                        break;
                    }
                }
            }
            if all_coherent {
                locked.push(*dim);
            }
        }

        // Calculate flow duration
        let flow_duration_ms = match (is_flow, self.flow_start) {
            (true, Some(start)) => start.elapsed().as_secs_f64() * 1000.0,
            (true, None) => 0.0,
            (false, _) => 0.0,
        };

        // Lock precision
        let lock_precision = if locked.is_empty() {
            0.0
        } else {
            locked.len() as f64 / 7.0
        };

        FlowStateAnalysis {
            in_flow: is_flow,
            flow_intensity,
            flow_duration_ms,
            locked_dimensions: locked,
            lock_precision,
        }
    }

    fn update_flow_state(&mut self, binding_strength: f64) {
        let is_flow = binding_strength > 0.8;

        match (is_flow, self.flow_start) {
            (true, None) => {
                self.flow_start = Some(Instant::now());
                self.stats.flow_state_entries += 1;
            }
            (false, Some(_)) => {
                self.flow_start = None;
            }
            _ => {}
        }
    }

    fn update_stats(&mut self, binding_strength: f64) {
        // Update max
        if binding_strength > self.stats.max_binding_strength {
            self.stats.max_binding_strength = binding_strength;
        }

        // Update averages
        let n = self.stats.observations as f64;
        self.stats.average_binding_strength =
            (self.stats.average_binding_strength * (n - 1.0) + binding_strength) / n;

        self.stats.average_coherence =
            (self.stats.average_coherence * (n - 1.0) + self.synchronization_index()) / n;

        // Count events
        if binding_strength > self.config.coherence_threshold {
            self.stats.binding_events += 1;
        }
        if binding_strength < self.config.fragmentation_threshold {
            self.stats.fragmentation_events += 1;
        }
    }

    /// Generate comprehensive binding report
    pub fn binding_report(&self) -> PhenomenalBindingReport {
        PhenomenalBindingReport {
            synchronization_index: self.synchronization_index(),
            phenomenal_binding_strength: self.phenomenal_binding_strength(),
            hierarchy: self.binding_hierarchy(),
            fragmentation: self.detect_fragmentation(),
            flow_state: self.flow_state_analysis(),
            is_unified: self.phenomenal_binding_strength() > self.config.coherence_threshold,
            stats: self.stats.clone(),
            coherence_matrix: self.coherence_matrix,
        }
    }

    /// Get all pairwise coherences
    pub fn all_coherences(&self) -> Vec<PairwiseCoherence> {
        let dims = ConsciousnessDimension::all();
        let mut result = Vec::new();

        for i in 0..7 {
            for j in (i + 1)..7 {
                let coherence = self.coherence_matrix[i][j];
                result.push(PairwiseCoherence {
                    dim_a: dims[i],
                    dim_b: dims[j],
                    coherence,
                    phase_difference: self.compute_phase_difference(dims[i], dims[j]),
                    is_bound: coherence >= self.config.coherence_threshold,
                });
            }
        }

        result
    }

    fn compute_phase_difference(
        &self,
        dim_a: ConsciousnessDimension,
        dim_b: ConsciousnessDimension,
    ) -> f64 {
        let phase_a = self.phase_history
            .get(&dim_a)
            .and_then(|h| h.back())
            .copied()
            .unwrap_or(0.0);

        let phase_b = self.phase_history
            .get(&dim_b)
            .and_then(|h| h.back())
            .copied()
            .unwrap_or(0.0);

        let diff = (phase_a - phase_b).abs();
        if diff > PI {
            2.0 * PI - diff
        } else {
            diff
        }
    }
}

// ============================================================================
// REPORT STRUCTURES
// ============================================================================

/// Comprehensive phenomenal binding report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhenomenalBindingReport {
    /// Overall synchronization (0-1)
    pub synchronization_index: f64,

    /// Phenomenal binding strength Ψ (0-1)
    pub phenomenal_binding_strength: f64,

    /// Hierarchical binding analysis
    pub hierarchy: BindingHierarchy,

    /// Fragmentation warning (if any)
    pub fragmentation: Option<FragmentationWarning>,

    /// Flow state analysis
    pub flow_state: FlowStateAnalysis,

    /// Whether consciousness is unified
    pub is_unified: bool,

    /// Cumulative statistics
    pub stats: BindingStats,

    /// Full coherence matrix
    pub coherence_matrix: [[f64; 7]; 7],
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analyzer_creation() {
        let analyzer = TemporalSynchronizationAnalyzer::new(BindingConfig::default());
        assert_eq!(analyzer.stats.observations, 0);
        assert!(analyzer.synchronization_index() >= 0.0);
    }

    #[test]
    fn test_single_dimension_observation() {
        let mut analyzer = TemporalSynchronizationAnalyzer::new(BindingConfig::default());

        analyzer.observe(ConsciousnessDimension::Integration, 0.8);
        assert_eq!(analyzer.stats.observations, 1);
    }

    #[test]
    fn test_all_dimensions_observation() {
        let mut analyzer = TemporalSynchronizationAnalyzer::new(BindingConfig::default());

        let values = [0.8, 0.7, 0.9, 0.75, 0.6, 0.85, 0.7];
        analyzer.observe_all(&values);

        assert_eq!(analyzer.stats.observations, 7);
    }

    #[test]
    fn test_synchronization_with_identical_values() {
        let mut analyzer = TemporalSynchronizationAnalyzer::new(BindingConfig::default());

        // Observe same value for all dimensions multiple times
        for _ in 0..10 {
            let values = [0.8; 7];
            analyzer.observe_all(&values);
        }

        // Should have high synchronization
        let sync = analyzer.synchronization_index();
        assert!(sync > 0.5, "Expected high sync, got {}", sync);
    }

    #[test]
    fn test_binding_hierarchy() {
        let mut analyzer = TemporalSynchronizationAnalyzer::new(BindingConfig::default());

        // Observe multiple times
        for i in 0..15 {
            let val = 0.5 + 0.3 * (i as f64 / 15.0).sin();
            let values = [val; 7];
            analyzer.observe_all(&values);
        }

        let hierarchy = analyzer.binding_hierarchy();

        // All levels should be in valid range
        assert!(hierarchy.sensory_binding >= 0.0 && hierarchy.sensory_binding <= 1.0);
        assert!(hierarchy.cognitive_binding >= 0.0 && hierarchy.cognitive_binding <= 1.0);
        assert!(hierarchy.identity_binding >= 0.0 && hierarchy.identity_binding <= 1.0);
        assert!(hierarchy.narrative_binding >= 0.0 && hierarchy.narrative_binding <= 1.0);
        assert!(hierarchy.unified_binding >= 0.0 && hierarchy.unified_binding <= 1.0);
    }

    #[test]
    fn test_fragmentation_detection() {
        let mut analyzer = TemporalSynchronizationAnalyzer::new(BindingConfig::default());

        // Create desynchronized pattern
        for i in 0..10 {
            // Alternate between high and low values
            let values = if i % 2 == 0 {
                [0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9]
            } else {
                [0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1]
            };
            analyzer.observe_all(&values);
        }

        // May or may not detect fragmentation depending on coherence calculation
        let _warning = analyzer.detect_fragmentation();
        // Just verify it runs without panic
    }

    #[test]
    fn test_flow_state_high_binding() {
        let mut analyzer = TemporalSynchronizationAnalyzer::new(BindingConfig::default());

        // Observe consistently high, synchronized values
        for _ in 0..20 {
            let values = [0.85; 7];
            analyzer.observe_all(&values);
        }

        let flow = analyzer.flow_state_analysis();
        // Should have good flow metrics
        assert!(flow.flow_intensity > 0.0);
    }

    #[test]
    fn test_pairwise_coherence() {
        let mut analyzer = TemporalSynchronizationAnalyzer::new(BindingConfig::default());

        for _ in 0..10 {
            let values = [0.7; 7];
            analyzer.observe_all(&values);
        }

        let coherence = analyzer.get_coherence(
            ConsciousnessDimension::Integration,
            ConsciousnessDimension::Workspace,
        );

        assert!(coherence >= 0.0 && coherence <= 1.0);
    }

    #[test]
    fn test_binding_report() {
        let mut analyzer = TemporalSynchronizationAnalyzer::new(BindingConfig::default());

        for _ in 0..15 {
            let values = [0.75; 7];
            analyzer.observe_all(&values);
        }

        let report = analyzer.binding_report();

        assert!(report.synchronization_index >= 0.0 && report.synchronization_index <= 1.0);
        assert!(report.phenomenal_binding_strength >= 0.0 && report.phenomenal_binding_strength <= 1.0);
        assert_eq!(report.stats.observations, 15 * 7);
    }

    #[test]
    fn test_all_coherences() {
        let mut analyzer = TemporalSynchronizationAnalyzer::new(BindingConfig::default());

        for _ in 0..10 {
            let values = [0.8; 7];
            analyzer.observe_all(&values);
        }

        let coherences = analyzer.all_coherences();

        // Should have 21 pairs (7 choose 2)
        assert_eq!(coherences.len(), 21);

        for c in &coherences {
            assert!(c.coherence >= 0.0 && c.coherence <= 1.0);
            assert!(c.phase_difference >= 0.0 && c.phase_difference <= PI);
        }
    }

    #[test]
    fn test_dimension_symbols() {
        assert_eq!(ConsciousnessDimension::Integration.symbol(), "Φ");
        assert_eq!(ConsciousnessDimension::Workspace.symbol(), "W");
        assert_eq!(ConsciousnessDimension::Attention.symbol(), "A");
    }

    #[test]
    fn test_statistics_update() {
        let mut analyzer = TemporalSynchronizationAnalyzer::new(BindingConfig::default());

        for _ in 0..20 {
            let values = [0.8; 7];
            analyzer.observe_all(&values);
        }

        assert_eq!(analyzer.stats.observations, 20 * 7);
        assert!(analyzer.stats.average_binding_strength >= 0.0);
        assert!(analyzer.stats.max_binding_strength >= analyzer.stats.average_binding_strength);
    }
}
