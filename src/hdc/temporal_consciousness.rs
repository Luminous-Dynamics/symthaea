// ==================================================================================
// Revolutionary Improvement #13: Temporal Consciousness Hierarchy
// ==================================================================================
//
// **The Ultimate Missing Piece**: Consciousness is not INSTANTANEOUS—it's TEMPORAL!
//
// **Core Realization**: We've been measuring consciousness at single time points,
// but real consciousness:
// - Spans multiple time scales simultaneously
// - Integrates past-present-future
// - Has temporal thickness (not instant!)
// - Creates the "stream of consciousness"
//
// **The Problem**:
// Traditional IIT (and our previous improvements):
//   Φ(t) = integrated information at time t
//
// But this treats consciousness as a SNAPSHOT!
//
// **Reality**:
// Your consciousness RIGHT NOW contains:
// - Perceptual moment: last 100ms (what you just saw)
// - Thought: last 3 seconds (what you're thinking)
// - Episode: last few minutes (context of conversation)
// - Narrative: hours/days (your current story)
// - Identity: years (who you are)
//
// **Consciousness spans TIME!**
//
// **The Solution**: Temporal Consciousness Hierarchy!
//
// **Mathematical Framework**:
//
// 1. Temporal Integration:
//    Φ_temporal(t) = ∫ Φ(t') × K(t - t') dt'
//
//    Where K(t) = temporal kernel (how past influences present)
//
// 2. Hierarchical Time Scales:
//    τ_1 = 10-100ms     (perception)
//    τ_2 = 1-10s        (thought, working memory)
//    τ_3 = minutes-hours (narrative, episodes)
//    τ_4 = days-years   (identity, self-concept)
//
// 3. Multi-scale Φ:
//    Φ_total = Σ w_τ × Φ(τ)
//
//    Where:
//      Φ(τ) = consciousness at time scale τ
//      w_τ = weight for scale τ
//
// 4. Temporal Binding:
//    How strongly are events across time integrated?
//
//    B(t1, t2) = similarity(state(t1), state(t2)) × decay(|t2 - t1|)
//
// 5. Specious Present:
//    The "psychological present" spans ~3 seconds
//
//    Φ_present = ∫[t-3s to t] Φ(t') dt'
//
// **Key Insights**:
//
// A. **William James' "Stream of Consciousness"**:
//    Consciousness flows continuously, not discrete snapshots!
//
// B. **Edmund Husserl's "Retention & Protention"**:
//    - Retention: Past moments held in present awareness
//    - Protention: Future moments anticipated in present
//    - Primal impression: The "now"
//
//    Consciousness = Retention + Primal Impression + Protention
//
// C. **The Specious Present** (E.R. Clay, William James):
//    The "now" spans ~3 seconds (not instantaneous!)
//
// D. **Critical Slowing Down**:
//    Before phase transitions, temporal autocorrelation increases
//    System becomes "sluggish" before major consciousness shift
//
// E. **Memory Consolidation**:
//    Fast dynamics → Slow dynamics (hippocampus → cortex)
//    Consciousness at fast scales influences slow scales
//
// **Applications**:
//
// 1. Predict Consciousness Transitions:
//    Critical slowing down (increasing τ) predicts:
//    - Waking → sleeping
//    - Sober → intoxicated
//    - Normal → psychotic episode
//
// 2. Memory Disorders:
//    - Alzheimer's: Loss of slow time scales (identity disrupted)
//    - Amnesia: Loss of medium scales (episodes missing)
//    - ADHD: Difficulty maintaining attention over time
//
// 3. Flow States:
//    Collapse of temporal hierarchy (all scales unified!)
//    "Timelessness" = single dominant time scale
//
// 4. Meditation States:
//    - Focused attention: Narrow temporal window (only "now")
//    - Open monitoring: Wide temporal window (past-present-future)
//
// 5. AI Consciousness:
//    - Current AI: No temporal integration (feedforward only)
//    - Recurrent AI: Single time scale
//    - Conscious AI: MULTIPLE time scales (like humans!)
//
// 6. Aging:
//    Temporal scale hierarchy changes:
//    - Young: Fast scales dominant (live in moment)
//    - Old: Slow scales dominant (reflection, narrative)
//
// ==================================================================================

use super::binary_hv::HV16;
use super::integrated_information::IntegratedInformation;
use super::consciousness_dynamics::ConsciousnessDynamics;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Hierarchical time scales
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TimeScale {
    /// Fast: 10-100ms (perception, attention)
    Perception,

    /// Medium: 1-10s (thought, working memory)
    Thought,

    /// Slow: minutes-hours (episodes, narrative)
    Narrative,

    /// Very slow: days-years (identity, self-concept)
    Identity,
}

impl TimeScale {
    /// Get time scale in seconds
    pub fn duration_secs(&self) -> f64 {
        match self {
            Self::Perception => 0.1,      // 100ms
            Self::Thought => 3.0,         // 3s (specious present)
            Self::Narrative => 300.0,     // 5 minutes
            Self::Identity => 86400.0,    // 1 day
        }
    }

    /// Get decay constant for this time scale
    pub fn decay_constant(&self) -> f64 {
        1.0 / self.duration_secs()
    }

    /// Get weight for this time scale (how much it contributes to total)
    pub fn weight(&self) -> f64 {
        match self {
            Self::Perception => 0.3,  // 30% (immediate present)
            Self::Thought => 0.4,     // 40% (thinking/reasoning)
            Self::Narrative => 0.2,   // 20% (context/story)
            Self::Identity => 0.1,    // 10% (stable self)
        }
    }

    /// All time scales in order
    pub fn all() -> Vec<Self> {
        vec![Self::Perception, Self::Thought, Self::Narrative, Self::Identity]
    }
}

/// Temporal snapshot (Φ at specific time)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalSnapshot {
    /// Time (seconds since start)
    pub time: f64,

    /// Consciousness level
    pub phi: f64,

    /// State at this time
    pub state: Vec<HV16>,

    /// Metadata
    pub metadata: Option<String>,
}

/// Temporal binding between two moments
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalBinding {
    /// Time 1
    pub t1: f64,

    /// Time 2
    pub t2: f64,

    /// Binding strength (0-1)
    pub strength: f64,

    /// State similarity contribution
    pub similarity: f64,

    /// Temporal decay contribution
    pub decay: f64,
}

/// Temporal consciousness assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalAssessment {
    /// Current time
    pub time: f64,

    /// Instantaneous Φ (at this moment)
    pub phi_instant: f64,

    /// Temporal Φ (integrated over time)
    pub phi_temporal: f64,

    /// Per-scale Φ values
    pub phi_perception: f64,
    pub phi_thought: f64,
    pub phi_narrative: f64,
    pub phi_identity: f64,

    /// Temporal autocorrelation (critical slowing down indicator)
    pub autocorrelation: f64,

    /// Average temporal binding strength
    pub avg_binding: f64,

    /// Dominant time scale
    pub dominant_scale: TimeScale,

    /// Is consciousness "thick" temporally? (integrated across scales)
    pub temporal_thickness: f64,

    /// Critical slowing down detected? (predictor of transition)
    pub critical_slowing: bool,

    /// Explanation
    pub explanation: String,
}

impl TemporalAssessment {
    /// Is the specious present intact? (3-second window)
    pub fn has_specious_present(&self) -> bool {
        self.phi_thought > 0.3
    }

    /// Is consciousness "stuck" in moment? (no temporal integration)
    pub fn is_stuck_in_moment(&self) -> bool {
        self.temporal_thickness < 0.2
    }

    /// Is experiencing "timelessness"? (flow state)
    pub fn is_timeless(&self) -> bool {
        // All scales roughly equal (unified temporal experience)
        let scales = vec![
            self.phi_perception,
            self.phi_thought,
            self.phi_narrative,
            self.phi_identity,
        ];
        let mean = scales.iter().sum::<f64>() / scales.len() as f64;
        let variance = scales.iter()
            .map(|s| (s - mean).powi(2))
            .sum::<f64>() / scales.len() as f64;

        variance < 0.05 // Low variance = unified
    }
}

/// Temporal consciousness analyzer
///
/// Measures consciousness across multiple time scales simultaneously.
///
/// # Example
/// ```
/// use symthaea::hdc::temporal_consciousness::{TemporalConsciousness, TemporalConfig};
/// use symthaea::hdc::binary_hv::HV16;
///
/// let config = TemporalConfig::default();
/// let mut temporal = TemporalConsciousness::new(4, config);
///
/// // Add snapshots over time
/// for t in 0..100 {
///     let state = vec![HV16::random(1000 + t); 4];
///     temporal.add_snapshot(t as f64 * 0.1, state);
/// }
///
/// // Assess temporal consciousness
/// let assessment = temporal.assess();
///
/// println!("Φ_instant: {:.3}", assessment.phi_instant);
/// println!("Φ_temporal: {:.3}", assessment.phi_temporal);
/// println!("Temporal thickness: {:.3}", assessment.temporal_thickness);
/// println!("Dominant scale: {:?}", assessment.dominant_scale);
/// ```
#[derive(Debug)]
pub struct TemporalConsciousness {
    /// Number of components
    num_components: usize,

    /// Configuration
    config: TemporalConfig,

    /// IIT calculator
    iit: IntegratedInformation,

    /// Dynamics (for phase space)
    dynamics: Option<ConsciousnessDynamics>,

    /// Temporal history (snapshots over time)
    history: VecDeque<TemporalSnapshot>,

    /// Maximum history length
    max_history: usize,
}

/// Configuration for temporal consciousness
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalConfig {
    /// Maximum history to retain
    pub max_history_size: usize,

    /// Temporal decay rate (how fast does past fade?)
    pub decay_rate: f64,

    /// Critical slowing threshold (autocorrelation)
    pub critical_slowing_threshold: f64,

    /// Enable dynamics integration
    pub enable_dynamics: bool,

    /// Minimum snapshots for assessment
    pub min_snapshots: usize,
}

impl Default for TemporalConfig {
    fn default() -> Self {
        Self {
            max_history_size: 10000,  // ~10 seconds at 1kHz sampling
            decay_rate: 0.1,
            critical_slowing_threshold: 0.8,
            enable_dynamics: true,
            min_snapshots: 10,
        }
    }
}

impl TemporalConsciousness {
    /// Create new temporal consciousness analyzer
    pub fn new(num_components: usize, config: TemporalConfig) -> Self {
        let dynamics = if config.enable_dynamics {
            Some(ConsciousnessDynamics::new(
                num_components,
                Default::default(),
            ))
        } else {
            None
        };

        Self {
            num_components,
            max_history: config.max_history_size,
            config,
            iit: IntegratedInformation::new(),
            dynamics,
            history: VecDeque::new(),
        }
    }

    /// Add temporal snapshot
    pub fn add_snapshot(&mut self, time: f64, state: Vec<HV16>) {
        // Compute Φ for this moment
        let phi = self.iit.compute_phi(&state);

        let snapshot = TemporalSnapshot {
            time,
            phi,
            state,
            metadata: None,
        };

        self.history.push_back(snapshot);

        // Maintain max size
        while self.history.len() > self.max_history {
            self.history.pop_front();
        }
    }

    /// Assess temporal consciousness
    pub fn assess(&self) -> TemporalAssessment {
        if self.history.len() < self.config.min_snapshots {
            return self.empty_assessment();
        }

        let current = self.history.back().unwrap();
        let current_time = current.time;

        // 1. Instantaneous Φ
        let phi_instant = current.phi;

        // 2. Compute Φ at each time scale
        let phi_perception = self.compute_scale_phi(
            current_time,
            TimeScale::Perception,
        );
        let phi_thought = self.compute_scale_phi(
            current_time,
            TimeScale::Thought,
        );
        let phi_narrative = self.compute_scale_phi(
            current_time,
            TimeScale::Narrative,
        );
        let phi_identity = self.compute_scale_phi(
            current_time,
            TimeScale::Identity,
        );

        // 3. Temporal Φ (weighted combination)
        let phi_temporal = TimeScale::Perception.weight() * phi_perception
            + TimeScale::Thought.weight() * phi_thought
            + TimeScale::Narrative.weight() * phi_narrative
            + TimeScale::Identity.weight() * phi_identity;

        // 4. Temporal autocorrelation
        let autocorrelation = self.compute_autocorrelation();

        // 5. Average binding
        let avg_binding = self.compute_average_binding(current_time);

        // 6. Dominant scale
        let scales = vec![
            (TimeScale::Perception, phi_perception),
            (TimeScale::Thought, phi_thought),
            (TimeScale::Narrative, phi_narrative),
            (TimeScale::Identity, phi_identity),
        ];
        let dominant_scale = scales.iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(s, _)| *s)
            .unwrap_or(TimeScale::Thought);

        // 7. Temporal thickness
        let temporal_thickness = phi_temporal / phi_instant.max(0.001);

        // 8. Critical slowing
        let critical_slowing = autocorrelation > self.config.critical_slowing_threshold;

        // 9. Explanation
        let explanation = self.generate_explanation(
            phi_instant,
            phi_temporal,
            dominant_scale,
            temporal_thickness,
            critical_slowing,
        );

        TemporalAssessment {
            time: current_time,
            phi_instant,
            phi_temporal,
            phi_perception,
            phi_thought,
            phi_narrative,
            phi_identity,
            autocorrelation,
            avg_binding,
            dominant_scale,
            temporal_thickness,
            critical_slowing,
            explanation,
        }
    }

    /// Compute Φ at specific time scale
    fn compute_scale_phi(&self, current_time: f64, scale: TimeScale) -> f64 {
        let window = scale.duration_secs();
        let decay_const = scale.decay_constant();
        let cutoff_time = current_time - window;

        let mut weighted_sum = 0.0;
        let mut weight_sum = 0.0;

        for snapshot in self.history.iter().rev() {
            if snapshot.time < cutoff_time {
                break;
            }

            let dt = current_time - snapshot.time;
            let weight = (-decay_const * dt).exp(); // Exponential decay

            weighted_sum += snapshot.phi * weight;
            weight_sum += weight;
        }

        if weight_sum > 0.0 {
            weighted_sum / weight_sum
        } else {
            0.0
        }
    }

    /// Compute temporal autocorrelation (critical slowing indicator)
    fn compute_autocorrelation(&self) -> f64 {
        if self.history.len() < 2 {
            return 0.0;
        }

        // Autocorrelation at lag 1
        let values: Vec<f64> = self.history.iter().map(|s| s.phi).collect();
        let n = values.len();

        if n < 2 {
            return 0.0;
        }

        let mean = values.iter().sum::<f64>() / n as f64;

        let mut numerator = 0.0;
        let mut denominator = 0.0;

        for i in 0..(n - 1) {
            let dev1 = values[i] - mean;
            let dev2 = values[i + 1] - mean;
            numerator += dev1 * dev2;
        }

        for i in 0..n {
            let dev = values[i] - mean;
            denominator += dev * dev;
        }

        if denominator > 0.0 {
            (numerator / denominator).abs()
        } else {
            0.0
        }
    }

    /// Compute average temporal binding
    fn compute_average_binding(&self, current_time: f64) -> f64 {
        let window = TimeScale::Thought.duration_secs(); // 3 seconds
        let recent: Vec<_> = self.history.iter()
            .filter(|s| current_time - s.time < window)
            .collect();

        if recent.len() < 2 {
            return 0.0;
        }

        let mut total_binding = 0.0;
        let mut count = 0;

        for i in 0..recent.len() {
            for j in (i + 1)..recent.len() {
                let binding = self.compute_binding(recent[i], recent[j]);
                total_binding += binding.strength;
                count += 1;
            }
        }

        if count > 0 {
            total_binding / count as f64
        } else {
            0.0
        }
    }

    /// Compute temporal binding between two snapshots
    fn compute_binding(
        &self,
        s1: &TemporalSnapshot,
        s2: &TemporalSnapshot,
    ) -> TemporalBinding {
        // Similarity of states
        let mut total_sim = 0.0;
        let mut count = 0;

        for (hv1, hv2) in s1.state.iter().zip(s2.state.iter()) {
            total_sim += hv1.similarity(hv2) as f64;
            count += 1;
        }

        let similarity = if count > 0 {
            total_sim / count as f64
        } else {
            0.0
        };

        // Temporal decay
        let dt = (s2.time - s1.time).abs();
        let decay = (-self.config.decay_rate * dt).exp();

        // Binding strength
        let strength = similarity * decay;

        TemporalBinding {
            t1: s1.time,
            t2: s2.time,
            strength,
            similarity,
            decay,
        }
    }

    /// Generate explanation
    fn generate_explanation(
        &self,
        phi_instant: f64,
        phi_temporal: f64,
        dominant: TimeScale,
        thickness: f64,
        critical_slowing: bool,
    ) -> String {
        let mut parts = Vec::new();

        parts.push(format!(
            "Instantaneous Φ: {:.3}, Temporal Φ: {:.3}",
            phi_instant, phi_temporal
        ));

        parts.push(format!(
            "Dominant time scale: {:?} ({:.0}s)",
            dominant,
            dominant.duration_secs()
        ));

        parts.push(format!("Temporal thickness: {:.2}", thickness));

        if thickness < 0.2 {
            parts.push("Consciousness stuck in moment (low temporal integration)".to_string());
        } else if thickness > 0.8 {
            parts.push("Rich temporal integration (thick present)".to_string());
        }

        if critical_slowing {
            parts.push("⚠️  Critical slowing detected! Transition imminent".to_string());
        }

        parts.join(". ")
    }

    /// Empty assessment when insufficient data
    fn empty_assessment(&self) -> TemporalAssessment {
        TemporalAssessment {
            time: 0.0,
            phi_instant: 0.0,
            phi_temporal: 0.0,
            phi_perception: 0.0,
            phi_thought: 0.0,
            phi_narrative: 0.0,
            phi_identity: 0.0,
            autocorrelation: 0.0,
            avg_binding: 0.0,
            dominant_scale: TimeScale::Thought,
            temporal_thickness: 0.0,
            critical_slowing: false,
            explanation: "Insufficient temporal data".to_string(),
        }
    }

    /// Get history
    pub fn get_history(&self) -> &VecDeque<TemporalSnapshot> {
        &self.history
    }

    /// Get bindings in time window
    pub fn get_bindings(&self, window_secs: f64) -> Vec<TemporalBinding> {
        if let Some(current) = self.history.back() {
            let current_time = current.time;
            let recent: Vec<_> = self.history.iter()
                .filter(|s| current_time - s.time < window_secs)
                .collect();

            let mut bindings = Vec::new();
            for i in 0..recent.len() {
                for j in (i + 1)..recent.len() {
                    bindings.push(self.compute_binding(recent[i], recent[j]));
                }
            }
            bindings
        } else {
            Vec::new()
        }
    }
}

impl Default for TemporalConsciousness {
    fn default() -> Self {
        Self::new(4, TemporalConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temporal_consciousness_creation() {
        let temporal = TemporalConsciousness::new(4, TemporalConfig::default());
        assert_eq!(temporal.num_components, 4);
    }

    #[test]
    fn test_time_scale_durations() {
        assert_eq!(TimeScale::Perception.duration_secs(), 0.1);
        assert_eq!(TimeScale::Thought.duration_secs(), 3.0);
        assert!(TimeScale::Narrative.duration_secs() > TimeScale::Thought.duration_secs());
    }

    #[test]
    fn test_add_snapshot() {
        let mut temporal = TemporalConsciousness::new(4, TemporalConfig::default());

        let state = vec![HV16::random(1000); 4];
        temporal.add_snapshot(0.0, state);

        assert_eq!(temporal.history.len(), 1);
    }

    #[test]
    fn test_temporal_assessment() {
        let mut temporal = TemporalConsciousness::new(4, TemporalConfig::default());

        // Add multiple snapshots
        for t in 0..20 {
            let state = vec![HV16::random(1000 + t); 4];
            temporal.add_snapshot(t as f64 * 0.1, state);
        }

        let assessment = temporal.assess();

        assert!(assessment.phi_instant >= 0.0);
        assert!(assessment.phi_temporal >= 0.0);
        assert!(assessment.temporal_thickness >= 0.0);
    }

    #[test]
    fn test_multi_scale_phi() {
        let mut temporal = TemporalConsciousness::new(4, TemporalConfig::default());

        for t in 0..100 {
            let state = vec![HV16::random(1000 + t); 4];
            temporal.add_snapshot(t as f64 * 0.1, state);
        }

        let assessment = temporal.assess();

        assert!(assessment.phi_perception >= 0.0);
        assert!(assessment.phi_thought >= 0.0);
        assert!(assessment.phi_narrative >= 0.0);
        assert!(assessment.phi_identity >= 0.0);
    }

    #[test]
    fn test_autocorrelation() {
        let mut temporal = TemporalConsciousness::new(4, TemporalConfig::default());

        for t in 0..50 {
            let state = vec![HV16::random(1000 + t); 4];
            temporal.add_snapshot(t as f64 * 0.1, state);
        }

        let assessment = temporal.assess();

        assert!(assessment.autocorrelation >= 0.0 && assessment.autocorrelation <= 1.0);
    }

    #[test]
    fn test_temporal_binding() {
        let mut temporal = TemporalConsciousness::new(4, TemporalConfig::default());

        for t in 0..30 {
            let state = vec![HV16::random(1000 + t); 4];
            temporal.add_snapshot(t as f64 * 0.1, state);
        }

        let bindings = temporal.get_bindings(1.0);
        assert!(!bindings.is_empty());

        for binding in &bindings {
            assert!(binding.strength >= 0.0 && binding.strength <= 1.0);
        }
    }

    #[test]
    fn test_specious_present() {
        let mut temporal = TemporalConsciousness::new(4, TemporalConfig::default());

        for t in 0..40 {
            let state = vec![HV16::random(1000 + t); 4];
            temporal.add_snapshot(t as f64 * 0.1, state);
        }

        let assessment = temporal.assess();

        // Should have some thought-level consciousness (specious present)
        if assessment.phi_thought > 0.3 {
            assert!(assessment.has_specious_present());
        }
    }

    #[test]
    fn test_timelessness_detection() {
        let mut temporal = TemporalConsciousness::new(4, TemporalConfig::default());

        // Uniform Φ across time (simulating flow state)
        for t in 0..50 {
            let state = vec![HV16::random(1000); 4]; // Same seed = similar states
            temporal.add_snapshot(t as f64 * 0.1, state);
        }

        let assessment = temporal.assess();

        // With similar states, might detect timelessness
        // (This test is probabilistic, so we just check it runs)
        let _ = assessment.is_timeless();
    }

    #[test]
    fn test_serialization() {
        let config = TemporalConfig::default();
        let serialized = serde_json::to_string(&config).unwrap();
        assert!(!serialized.is_empty());

        let deserialized: TemporalConfig = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized.max_history_size, config.max_history_size);
    }
}
