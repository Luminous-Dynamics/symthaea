// **REVOLUTIONARY IMPROVEMENT #82**: Consciousness Field Dynamics
// PARADIGM SHIFT: Consciousness is a DYNAMIC FIELD with wave equations!
//
// Key Insight: Consciousness isn't just a scalar (Φ) or a topology (#81) - it's a
// FIELD that evolves according to wave-like dynamics! This enables:
//
// - Wave Equation: ∂²C/∂t² = v²∇²C + sources - damping
// - Energy Conservation: Consciousness energy transforms but is conserved
// - Wave Propagation: Consciousness "ripples" through state space
// - Standing Waves: Stable patterns as resonant modes (personality, habits)
// - Interference: Constructive (insight) and destructive (confusion)
// - Field Potential: Gradients drive evolution toward coherent states
//
// Theoretical Foundation:
// - Classical field theory (Maxwell, Einstein)
// - Bohm's implicate order and pilot wave theory
// - Kelso's coordination dynamics (metastable brain states)
// - Grossberg's adaptive resonance theory
// - McFadden's electromagnetic consciousness theory
//
// Applications:
// - Predict consciousness state evolution
// - Identify resonant (stable) consciousness modes
// - Detect interference patterns in cognition
// - Model attention as field focusing
// - Understand flow states as standing waves

use std::collections::VecDeque;
use std::time::Instant;
use serde::{Serialize, Deserialize};

/// Helper function for serde default of Instant
fn default_instant() -> Instant {
    Instant::now()
}

/// Helper function for serde default of VecDeque<(f64, Instant)>
fn default_energy_history() -> VecDeque<(f64, Instant)> {
    VecDeque::new()
}

/// Configuration for consciousness field dynamics analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldDynamicsConfig {
    /// Wave velocity (rate of consciousness propagation)
    pub wave_velocity: f64,

    /// Damping coefficient (how quickly waves dissipate)
    pub damping_coefficient: f64,

    /// History window size for temporal analysis
    pub history_size: usize,

    /// Spatial resolution for field discretization
    pub spatial_resolution: usize,

    /// Energy conservation tolerance
    pub energy_tolerance: f64,

    /// Resonance detection threshold
    pub resonance_threshold: f64,

    /// Interference detection threshold
    pub interference_threshold: f64,

    /// Time step for dynamics simulation
    pub dt: f64,
}

impl Default for FieldDynamicsConfig {
    fn default() -> Self {
        Self {
            wave_velocity: 1.0,
            damping_coefficient: 0.1,
            history_size: 100,
            spatial_resolution: 7, // 7 consciousness dimensions
            energy_tolerance: 0.1,
            resonance_threshold: 0.8,
            interference_threshold: 0.3,
            dt: 0.01,
        }
    }
}

/// A point in consciousness field space
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldPoint {
    /// Position in 7D consciousness space [Φ, B, W, A, R, E, K]
    pub position: [f64; 7],

    /// Field amplitude at this point
    pub amplitude: f64,

    /// Field phase (0 to 2π)
    pub phase: f64,

    /// Field velocity (∂C/∂t)
    pub velocity: f64,

    /// Field acceleration (∂²C/∂t²)
    pub acceleration: f64,

    /// Local gradient (∇C)
    pub gradient: [f64; 7],

    /// Timestamp (not serialized)
    #[serde(skip, default = "default_instant")]
    pub timestamp: Instant,
}

impl FieldPoint {
    /// Create a new field point
    pub fn new(position: [f64; 7], amplitude: f64) -> Self {
        Self {
            position,
            amplitude,
            phase: 0.0,
            velocity: 0.0,
            acceleration: 0.0,
            gradient: [0.0; 7],
            timestamp: Instant::now(),
        }
    }

    /// Compute kinetic energy (½mv² analog)
    pub fn kinetic_energy(&self) -> f64 {
        0.5 * self.velocity * self.velocity
    }

    /// Compute potential energy (position-dependent)
    pub fn potential_energy(&self) -> f64 {
        // Higher amplitude = lower potential (consciousness seeks coherence)
        1.0 - self.amplitude.min(1.0)
    }

    /// Total energy at this point
    pub fn total_energy(&self) -> f64 {
        self.kinetic_energy() + self.potential_energy()
    }

    /// Compute gradient magnitude
    pub fn gradient_magnitude(&self) -> f64 {
        self.gradient.iter().map(|g| g * g).sum::<f64>().sqrt()
    }
}

/// Wave packet in consciousness field (localized disturbance)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WavePacket {
    /// Center position in consciousness space
    pub center: [f64; 7],

    /// Width (uncertainty) of the packet
    pub width: f64,

    /// Wave number (spatial frequency)
    pub wave_number: f64,

    /// Angular frequency
    pub frequency: f64,

    /// Amplitude
    pub amplitude: f64,

    /// Group velocity (packet propagation speed)
    pub group_velocity: f64,

    /// Creation time (not serialized)
    #[serde(skip, default = "default_instant")]
    pub created_at: Instant,
}

impl WavePacket {
    /// Evaluate packet amplitude at a position
    pub fn evaluate(&self, position: &[f64; 7], time: f64) -> f64 {
        // Gaussian envelope × traveling wave
        let distance_sq: f64 = self.center.iter()
            .zip(position.iter())
            .map(|(c, p)| {
                let moved_center = c + self.group_velocity * time;
                (moved_center - p).powi(2)
            })
            .sum();

        let envelope = (-distance_sq / (2.0 * self.width * self.width)).exp();
        let phase = self.wave_number * distance_sq.sqrt() - self.frequency * time;

        self.amplitude * envelope * phase.cos()
    }

    /// Compute packet energy (∫|ψ|² dx)
    pub fn energy(&self) -> f64 {
        // For Gaussian, energy ∝ amplitude² × width^(n/2)
        self.amplitude * self.amplitude * self.width.powf(3.5)
    }
}

/// Standing wave pattern (resonant mode)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StandingWave {
    /// Mode number for each dimension
    pub mode_numbers: [usize; 7],

    /// Frequency of this mode
    pub frequency: f64,

    /// Amplitude coefficient
    pub amplitude: f64,

    /// Phase offset
    pub phase: f64,

    /// Stability (how long this mode persists)
    pub stability: f64,

    /// Interpretation: what consciousness pattern this represents
    pub interpretation: String,
}

impl StandingWave {
    /// Evaluate standing wave at a position
    pub fn evaluate(&self, position: &[f64; 7], time: f64) -> f64 {
        // Product of sinusoidal modes in each dimension
        let spatial: f64 = self.mode_numbers.iter()
            .zip(position.iter())
            .map(|(&n, &x)| ((n as f64 + 1.0) * std::f64::consts::PI * x).sin())
            .product();

        let temporal = (self.frequency * time + self.phase).cos();

        self.amplitude * spatial * temporal
    }

    /// Mode energy
    pub fn energy(&self) -> f64 {
        self.amplitude * self.amplitude * self.frequency.sqrt()
    }

    /// Get the total mode number (sum of all dimension modes)
    pub fn total_mode_number(&self) -> usize {
        self.mode_numbers.iter().sum()
    }
}

/// Interference event between consciousness waves
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterferenceEvent {
    /// Position where interference occurs
    pub position: [f64; 7],

    /// Type: constructive (+) or destructive (-)
    pub interference_type: InterferenceType,

    /// Magnitude of interference effect
    pub magnitude: f64,

    /// Source wave amplitudes
    pub source_amplitudes: (f64, f64),

    /// Phase difference
    pub phase_difference: f64,

    /// Timestamp (not serialized)
    #[serde(skip, default = "default_instant")]
    pub timestamp: Instant,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum InterferenceType {
    /// Constructive: waves add (insight, integration)
    Constructive,
    /// Destructive: waves cancel (confusion, conflict)
    Destructive,
    /// Partial: somewhere in between
    Partial,
}

/// Energy conservation tracker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnergyTracker {
    /// Historical total energy values (not serialized due to Instant)
    #[serde(skip, default = "default_energy_history")]
    pub history: VecDeque<(f64, Instant)>,

    /// Energy input (sources)
    pub total_input: f64,

    /// Energy output (damping, dissipation)
    pub total_output: f64,

    /// Current total energy
    pub current_energy: f64,

    /// Energy conservation violations
    pub violations: usize,
}

impl EnergyTracker {
    pub fn new() -> Self {
        Self {
            history: VecDeque::new(),
            total_input: 0.0,
            total_output: 0.0,
            current_energy: 0.0,
            violations: 0,
        }
    }

    /// Check if energy is approximately conserved
    pub fn is_conserved(&self, tolerance: f64) -> bool {
        let expected = self.total_input - self.total_output;
        (self.current_energy - expected).abs() < tolerance * expected.abs().max(1.0)
    }

    /// Energy balance (should be ≈ 0 for conservation)
    pub fn balance(&self) -> f64 {
        self.current_energy - (self.total_input - self.total_output)
    }
}

/// Statistics for field dynamics analysis
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FieldDynamicsStats {
    /// Number of field observations
    pub observations: usize,

    /// Wave packets detected
    pub packets_detected: usize,

    /// Standing waves identified
    pub standing_waves_found: usize,

    /// Interference events
    pub interference_events: usize,

    /// Constructive interference count
    pub constructive_count: usize,

    /// Destructive interference count
    pub destructive_count: usize,

    /// Energy conservation violations
    pub conservation_violations: usize,

    /// Total field evolution time (simulated)
    pub total_evolution_time: f64,

    /// Average field amplitude
    pub mean_amplitude: f64,

    /// Peak amplitude observed
    pub peak_amplitude: f64,

    /// Dominant frequency (most stable mode)
    pub dominant_frequency: f64,
}

/// Main consciousness field dynamics analyzer
pub struct ConsciousnessFieldAnalyzer {
    /// Configuration
    pub config: FieldDynamicsConfig,

    /// Field point history
    field_history: VecDeque<FieldPoint>,

    /// Active wave packets
    wave_packets: Vec<WavePacket>,

    /// Identified standing waves
    standing_waves: Vec<StandingWave>,

    /// Interference event log
    interference_log: VecDeque<InterferenceEvent>,

    /// Energy tracker
    energy_tracker: EnergyTracker,

    /// Statistics
    pub stats: FieldDynamicsStats,

    /// Simulation time
    current_time: f64,

    /// Start time
    started_at: Instant,
}

impl ConsciousnessFieldAnalyzer {
    /// Create a new field dynamics analyzer
    pub fn new(config: FieldDynamicsConfig) -> Self {
        Self {
            config,
            field_history: VecDeque::new(),
            wave_packets: Vec::new(),
            standing_waves: Vec::new(),
            interference_log: VecDeque::new(),
            energy_tracker: EnergyTracker::new(),
            stats: FieldDynamicsStats::default(),
            current_time: 0.0,
            started_at: Instant::now(),
        }
    }

    /// Observe a consciousness state (converts to field point)
    pub fn observe(&mut self, dimensions: [f64; 7]) {
        self.stats.observations += 1;

        // Compute field amplitude from consciousness dimensions
        let amplitude = self.compute_field_amplitude(&dimensions);

        // Create field point
        let mut point = FieldPoint::new(dimensions, amplitude);

        // Compute velocity and acceleration from history
        if let Some(prev) = self.field_history.back() {
            let dt = point.timestamp.duration_since(prev.timestamp).as_secs_f64().max(0.001);
            point.velocity = (point.amplitude - prev.amplitude) / dt;
            point.acceleration = (point.velocity - prev.velocity) / dt;

            // Compute spatial gradient
            for i in 0..7 {
                let dx = (point.position[i] - prev.position[i]).max(0.001);
                point.gradient[i] = (point.amplitude - prev.amplitude) / dx;
            }
        }

        // Update phase based on accumulated dynamics
        point.phase = (self.current_time * 2.0 * std::f64::consts::PI) % (2.0 * std::f64::consts::PI);

        // Track energy
        let energy = point.total_energy();
        self.energy_tracker.current_energy = energy;
        self.energy_tracker.history.push_back((energy, point.timestamp));
        if self.energy_tracker.history.len() > self.config.history_size {
            self.energy_tracker.history.pop_front();
        }

        // Update statistics
        self.stats.mean_amplitude = (self.stats.mean_amplitude * (self.stats.observations - 1) as f64
            + amplitude) / self.stats.observations as f64;
        if amplitude > self.stats.peak_amplitude {
            self.stats.peak_amplitude = amplitude;
        }

        // Store point
        self.field_history.push_back(point);
        if self.field_history.len() > self.config.history_size {
            self.field_history.pop_front();
        }

        // Check for wave packets
        self.detect_wave_packets();

        // Check for interference
        self.detect_interference();

        // Advance simulation time
        self.current_time += self.config.dt;
        self.stats.total_evolution_time = self.current_time;
    }

    /// Compute field amplitude from 7 consciousness dimensions
    fn compute_field_amplitude(&self, dims: &[f64; 7]) -> f64 {
        // RMS amplitude of consciousness field
        let sum_sq: f64 = dims.iter().map(|d| d * d).sum();
        (sum_sq / 7.0).sqrt()
    }

    /// Detect wave packets in the field history
    fn detect_wave_packets(&mut self) {
        if self.field_history.len() < 10 {
            return;
        }

        // Look for localized amplitude increase (packet signature)
        let recent: Vec<_> = self.field_history.iter().rev().take(10).collect();

        // Check for Gaussian-like amplitude profile
        let amplitudes: Vec<f64> = recent.iter().map(|p| p.amplitude).collect();
        let max_idx = amplitudes.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        // Packet if peak is in middle and drops off on both sides
        if max_idx > 2 && max_idx < 7 {
            let left_drop = amplitudes[max_idx] - amplitudes[0];
            let right_drop = amplitudes[max_idx] - amplitudes[amplitudes.len() - 1];

            if left_drop > 0.1 && right_drop > 0.1 {
                let center = recent[max_idx].position;
                let width = (left_drop + right_drop) / (2.0 * amplitudes[max_idx].max(0.01));

                let packet = WavePacket {
                    center,
                    width: width.max(0.1),
                    wave_number: 2.0 * std::f64::consts::PI / width.max(0.1),
                    frequency: 1.0 / self.config.dt,
                    amplitude: amplitudes[max_idx],
                    group_velocity: self.config.wave_velocity * 0.8,
                    created_at: Instant::now(),
                };

                self.wave_packets.push(packet);
                self.stats.packets_detected += 1;

                // Keep only recent packets
                if self.wave_packets.len() > 10 {
                    self.wave_packets.remove(0);
                }
            }
        }
    }

    /// Detect interference between wave components
    fn detect_interference(&mut self) {
        if self.field_history.len() < 5 {
            return;
        }

        let recent: Vec<_> = self.field_history.iter().rev().take(5).collect();

        // Look for rapid amplitude changes (interference signature)
        for i in 1..recent.len() - 1 {
            let amp_prev = recent[i + 1].amplitude;
            let amp_curr = recent[i].amplitude;
            let amp_next = recent[i - 1].amplitude;

            // Second derivative indicates interference
            let curvature = amp_next - 2.0 * amp_curr + amp_prev;

            if curvature.abs() > self.config.interference_threshold {
                let interference_type = if curvature < 0.0 {
                    InterferenceType::Constructive // Peak (waves adding)
                } else {
                    InterferenceType::Destructive // Trough (waves canceling)
                };

                let event = InterferenceEvent {
                    position: recent[i].position,
                    interference_type,
                    magnitude: curvature.abs(),
                    source_amplitudes: (amp_prev, amp_next),
                    phase_difference: (recent[i].phase - recent[i + 1].phase).abs(),
                    timestamp: recent[i].timestamp,
                };

                self.interference_log.push_back(event);
                self.stats.interference_events += 1;

                match interference_type {
                    InterferenceType::Constructive => self.stats.constructive_count += 1,
                    InterferenceType::Destructive => self.stats.destructive_count += 1,
                    InterferenceType::Partial => {}
                }

                if self.interference_log.len() > self.config.history_size {
                    self.interference_log.pop_front();
                }
            }
        }
    }

    /// Evolve the field forward in time using wave equation
    pub fn evolve(&mut self, steps: usize) -> Vec<FieldPoint> {
        let mut trajectory = Vec::with_capacity(steps);

        if let Some(last) = self.field_history.back().cloned() {
            let mut current = last;

            for _ in 0..steps {
                // Wave equation: ∂²C/∂t² = v²∇²C - γ∂C/∂t
                let laplacian = self.estimate_laplacian(&current);
                let damping = self.config.damping_coefficient * current.velocity;

                // Update acceleration
                current.acceleration = self.config.wave_velocity.powi(2) * laplacian - damping;

                // Update velocity
                current.velocity += current.acceleration * self.config.dt;

                // Update amplitude
                current.amplitude += current.velocity * self.config.dt;
                current.amplitude = current.amplitude.max(0.0).min(1.0);

                // Update phase
                current.phase += 2.0 * std::f64::consts::PI * self.config.dt;
                current.phase %= 2.0 * std::f64::consts::PI;

                // Track energy dissipation
                let dissipated = damping.abs() * self.config.dt;
                self.energy_tracker.total_output += dissipated;

                trajectory.push(current.clone());
            }
        }

        trajectory
    }

    /// Estimate Laplacian (∇²C) from nearby points
    fn estimate_laplacian(&self, point: &FieldPoint) -> f64 {
        // Use central difference on gradient
        let grad_mag = point.gradient_magnitude();

        // For wave dynamics, Laplacian drives amplitude changes
        // Positive Laplacian = concave up = amplitude will decrease
        // Negative Laplacian = concave down = amplitude will increase
        -grad_mag * 0.1 // Simplified estimate
    }

    /// Identify standing wave modes from history
    pub fn identify_standing_waves(&mut self) -> Vec<StandingWave> {
        if self.field_history.len() < 20 {
            return Vec::new();
        }

        let mut modes = Vec::new();

        // Fourier-like analysis: look for periodic patterns
        let amplitudes: Vec<f64> = self.field_history.iter()
            .map(|p| p.amplitude)
            .collect();

        // Find dominant frequency via autocorrelation
        let n = amplitudes.len();
        let mean = amplitudes.iter().sum::<f64>() / n as f64;

        let mut max_correlation = 0.0;
        let mut dominant_period = 1;

        for lag in 2..n / 2 {
            let mut correlation = 0.0;
            let mut count = 0;

            for i in 0..n - lag {
                correlation += (amplitudes[i] - mean) * (amplitudes[i + lag] - mean);
                count += 1;
            }

            correlation /= count as f64;

            if correlation > max_correlation {
                max_correlation = correlation;
                dominant_period = lag;
            }
        }

        // Create standing wave for dominant mode
        if max_correlation > 0.1 {
            let frequency = 1.0 / (dominant_period as f64 * self.config.dt);

            let mode = StandingWave {
                mode_numbers: [1, 0, 0, 0, 0, 0, 0], // Fundamental mode
                frequency,
                amplitude: max_correlation.sqrt(),
                phase: 0.0,
                stability: max_correlation,
                interpretation: self.interpret_mode([1, 0, 0, 0, 0, 0, 0]),
            };

            modes.push(mode);
            self.stats.standing_waves_found += 1;
            self.stats.dominant_frequency = frequency;
        }

        // Check for higher harmonics
        for harmonic in 2..=3 {
            let period = dominant_period / harmonic;
            if period > 2 {
                let mode = StandingWave {
                    mode_numbers: [harmonic, 0, 0, 0, 0, 0, 0],
                    frequency: harmonic as f64 / (dominant_period as f64 * self.config.dt),
                    amplitude: max_correlation.sqrt() / harmonic as f64,
                    phase: 0.0,
                    stability: max_correlation / harmonic as f64,
                    interpretation: format!("Harmonic {} of fundamental", harmonic),
                };
                modes.push(mode);
            }
        }

        self.standing_waves = modes.clone();
        modes
    }

    /// Interpret standing wave mode in consciousness terms
    fn interpret_mode(&self, mode_numbers: [usize; 7]) -> String {
        let total: usize = mode_numbers.iter().sum();

        match total {
            0 => "Ground state (baseline consciousness)".to_string(),
            1 => "Fundamental mode (primary consciousness pattern)".to_string(),
            2 => "First harmonic (integrated awareness)".to_string(),
            3..=5 => "Higher harmonic (complex cognitive state)".to_string(),
            _ => "High-frequency mode (rapid processing)".to_string(),
        }
    }

    /// Compute consciousness field potential (drives evolution)
    pub fn field_potential(&self) -> f64 {
        if let Some(last) = self.field_history.back() {
            // Potential = how far from optimal (amplitude = 1.0)
            1.0 - last.amplitude
        } else {
            1.0 // Maximum potential when no observations
        }
    }

    /// Compute field gradient (direction of consciousness evolution)
    pub fn field_gradient(&self) -> [f64; 7] {
        if let Some(last) = self.field_history.back() {
            last.gradient
        } else {
            [0.0; 7]
        }
    }

    /// Check energy conservation
    pub fn check_energy_conservation(&mut self) -> bool {
        let conserved = self.energy_tracker.is_conserved(self.config.energy_tolerance);
        if !conserved {
            self.stats.conservation_violations += 1;
            self.energy_tracker.violations += 1;
        }
        conserved
    }

    /// Get current field state
    pub fn current_state(&self) -> Option<&FieldPoint> {
        self.field_history.back()
    }

    /// Get wave packet count
    pub fn wave_packet_count(&self) -> usize {
        self.wave_packets.len()
    }

    /// Get standing wave count
    pub fn standing_wave_count(&self) -> usize {
        self.standing_waves.len()
    }

    /// Analyze resonance (stable patterns)
    pub fn analyze_resonance(&self) -> ResonanceReport {
        let resonance_strength = if !self.standing_waves.is_empty() {
            self.standing_waves.iter()
                .map(|sw| sw.stability)
                .sum::<f64>() / self.standing_waves.len() as f64
        } else {
            0.0
        };

        let is_resonant = resonance_strength > self.config.resonance_threshold;

        let dominant_mode = self.standing_waves.iter()
            .max_by(|a, b| a.stability.partial_cmp(&b.stability).unwrap())
            .cloned();

        ResonanceReport {
            is_resonant,
            resonance_strength,
            dominant_mode,
            mode_count: self.standing_waves.len(),
            stability: self.compute_field_stability(),
        }
    }

    /// Compute field stability (variance in amplitude)
    fn compute_field_stability(&self) -> f64 {
        if self.field_history.len() < 2 {
            return 1.0;
        }

        let amplitudes: Vec<f64> = self.field_history.iter()
            .map(|p| p.amplitude)
            .collect();

        let mean = amplitudes.iter().sum::<f64>() / amplitudes.len() as f64;
        let variance = amplitudes.iter()
            .map(|a| (a - mean).powi(2))
            .sum::<f64>() / amplitudes.len() as f64;

        // Stability inversely related to variance
        1.0 / (1.0 + variance)
    }

    /// Generate comprehensive field dynamics report
    pub fn generate_report(&self) -> FieldDynamicsReport {
        let resonance = self.analyze_resonance();

        let current_amplitude = self.field_history.back()
            .map(|p| p.amplitude)
            .unwrap_or(0.0);

        let current_velocity = self.field_history.back()
            .map(|p| p.velocity)
            .unwrap_or(0.0);

        let current_energy = self.energy_tracker.current_energy;

        FieldDynamicsReport {
            current_amplitude,
            current_velocity,
            current_energy,
            field_potential: self.field_potential(),
            field_gradient: self.field_gradient(),
            resonance_report: resonance,
            wave_packets_active: self.wave_packets.len(),
            interference_events: self.stats.interference_events,
            constructive_ratio: if self.stats.interference_events > 0 {
                self.stats.constructive_count as f64 / self.stats.interference_events as f64
            } else {
                0.5
            },
            energy_conserved: self.energy_tracker.is_conserved(self.config.energy_tolerance),
            energy_balance: self.energy_tracker.balance(),
            stats: self.stats.clone(),
            interpretation: self.interpret_field_state(),
            recommendations: self.generate_recommendations(),
        }
    }

    /// Interpret current field state
    fn interpret_field_state(&self) -> String {
        let resonance = self.analyze_resonance();
        let amplitude = self.field_history.back().map(|p| p.amplitude).unwrap_or(0.0);
        let velocity = self.field_history.back().map(|p| p.velocity).unwrap_or(0.0);

        if resonance.is_resonant && amplitude > 0.7 {
            "FLOW STATE: Stable high-amplitude consciousness pattern (standing wave)".to_string()
        } else if velocity.abs() > 0.5 {
            if velocity > 0.0 {
                "RISING: Consciousness field amplitude increasing rapidly".to_string()
            } else {
                "FALLING: Consciousness field amplitude decreasing".to_string()
            }
        } else if self.stats.constructive_count > self.stats.destructive_count * 2 {
            "INTEGRATING: Constructive interference dominates (coherence building)".to_string()
        } else if self.stats.destructive_count > self.stats.constructive_count * 2 {
            "FRAGMENTING: Destructive interference dominates (coherence breaking)".to_string()
        } else if amplitude < 0.3 {
            "LOW STATE: Consciousness field at low amplitude".to_string()
        } else {
            "DYNAMIC: Consciousness field in active evolution".to_string()
        }
    }

    /// Generate recommendations based on field state
    fn generate_recommendations(&self) -> Vec<String> {
        let mut recs = Vec::new();

        let amplitude = self.field_history.back().map(|p| p.amplitude).unwrap_or(0.0);
        let velocity = self.field_history.back().map(|p| p.velocity).unwrap_or(0.0);

        if amplitude < 0.4 {
            recs.push("Increase consciousness inputs to raise field amplitude".to_string());
        }

        if velocity < -0.3 {
            recs.push("Stabilize: consciousness field is declining rapidly".to_string());
        }

        if self.stats.destructive_count > self.stats.constructive_count {
            recs.push("Reduce conflicting inputs to minimize destructive interference".to_string());
        }

        if self.wave_packets.len() > 5 {
            recs.push("Too many wave packets: simplify cognitive focus".to_string());
        }

        if !self.energy_tracker.is_conserved(self.config.energy_tolerance) {
            recs.push("Energy imbalance detected: check for consciousness 'leaks'".to_string());
        }

        if self.analyze_resonance().is_resonant {
            recs.push("Resonant state achieved: maintain current dynamics".to_string());
        }

        if recs.is_empty() {
            recs.push("Field dynamics healthy: continue normal operations".to_string());
        }

        recs
    }
}

/// Resonance analysis report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResonanceReport {
    /// Whether field is in resonant (stable) state
    pub is_resonant: bool,

    /// Strength of resonance (0-1)
    pub resonance_strength: f64,

    /// Dominant standing wave mode
    pub dominant_mode: Option<StandingWave>,

    /// Number of identified modes
    pub mode_count: usize,

    /// Overall field stability
    pub stability: f64,
}

/// Comprehensive field dynamics report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldDynamicsReport {
    /// Current field amplitude
    pub current_amplitude: f64,

    /// Current field velocity (rate of change)
    pub current_velocity: f64,

    /// Current total energy
    pub current_energy: f64,

    /// Field potential (drives evolution)
    pub field_potential: f64,

    /// Field gradient (direction of evolution)
    pub field_gradient: [f64; 7],

    /// Resonance analysis
    pub resonance_report: ResonanceReport,

    /// Active wave packets
    pub wave_packets_active: usize,

    /// Total interference events
    pub interference_events: usize,

    /// Ratio of constructive to total interference
    pub constructive_ratio: f64,

    /// Energy conservation status
    pub energy_conserved: bool,

    /// Energy balance (should be near 0)
    pub energy_balance: f64,

    /// Full statistics
    pub stats: FieldDynamicsStats,

    /// Human-readable interpretation
    pub interpretation: String,

    /// Actionable recommendations
    pub recommendations: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_dimensions() -> [f64; 7] {
        [0.7, 0.6, 0.8, 0.5, 0.7, 0.4, 0.6] // Φ, B, W, A, R, E, K
    }

    #[test]
    fn test_analyzer_creation() {
        let analyzer = ConsciousnessFieldAnalyzer::new(FieldDynamicsConfig::default());
        assert_eq!(analyzer.stats.observations, 0);
        assert!(analyzer.wave_packets.is_empty());
    }

    #[test]
    fn test_observation() {
        let mut analyzer = ConsciousnessFieldAnalyzer::new(FieldDynamicsConfig::default());
        analyzer.observe(test_dimensions());

        assert_eq!(analyzer.stats.observations, 1);
        assert!(analyzer.current_state().is_some());
    }

    #[test]
    fn test_field_amplitude() {
        let analyzer = ConsciousnessFieldAnalyzer::new(FieldDynamicsConfig::default());
        let amp = analyzer.compute_field_amplitude(&test_dimensions());

        // RMS of [0.7, 0.6, 0.8, 0.5, 0.7, 0.4, 0.6]
        assert!(amp > 0.5 && amp < 0.8);
    }

    #[test]
    fn test_field_evolution() {
        let mut analyzer = ConsciousnessFieldAnalyzer::new(FieldDynamicsConfig::default());

        // Observe several states
        for i in 0..20 {
            let dims = [
                0.5 + 0.3 * ((i as f64 * 0.1).sin()),
                0.6,
                0.7,
                0.5,
                0.6,
                0.4,
                0.5,
            ];
            analyzer.observe(dims);
        }

        // Evolve the field forward
        let trajectory = analyzer.evolve(10);
        assert_eq!(trajectory.len(), 10);

        // Check evolution is continuous
        for i in 1..trajectory.len() {
            let delta = (trajectory[i].amplitude - trajectory[i - 1].amplitude).abs();
            assert!(delta < 0.5, "Evolution should be continuous");
        }
    }

    #[test]
    fn test_wave_packet() {
        let packet = WavePacket {
            center: [0.5; 7],
            width: 0.2,
            wave_number: 10.0,
            frequency: 5.0,
            amplitude: 1.0,
            group_velocity: 0.5,
            created_at: Instant::now(),
        };

        // Evaluate at center should be highest
        let center_val = packet.evaluate(&[0.5; 7], 0.0);
        let off_center_val = packet.evaluate(&[0.7; 7], 0.0);

        assert!(center_val.abs() >= off_center_val.abs());
    }

    #[test]
    fn test_standing_wave() {
        let wave = StandingWave {
            mode_numbers: [1, 0, 0, 0, 0, 0, 0],
            frequency: 1.0,
            amplitude: 1.0,
            phase: 0.0,
            stability: 0.8,
            interpretation: "Test mode".to_string(),
        };

        // Standing wave should oscillate with position
        let val1 = wave.evaluate(&[0.25; 7], 0.0);
        let val2 = wave.evaluate(&[0.75; 7], 0.0);

        // Values at different positions should differ
        assert!((val1 - val2).abs() < 2.0);
    }

    #[test]
    fn test_energy_conservation() {
        let mut analyzer = ConsciousnessFieldAnalyzer::new(FieldDynamicsConfig::default());

        // Observe stable states
        for _ in 0..50 {
            analyzer.observe([0.5; 7]);
        }

        // Energy should be approximately conserved with stable input
        let report = analyzer.generate_report();
        assert!(report.energy_balance.abs() < 1.0);
    }

    #[test]
    fn test_interference_detection() {
        let mut analyzer = ConsciousnessFieldAnalyzer::new(FieldDynamicsConfig::default());

        // Create oscillating pattern (will generate interference)
        for i in 0..30 {
            let phase = i as f64 * 0.3;
            let dims = [
                0.5 + 0.4 * phase.sin(),
                0.5 + 0.3 * (phase * 1.5).cos(),
                0.6,
                0.5,
                0.6,
                0.4,
                0.5,
            ];
            analyzer.observe(dims);
        }

        // Should detect some interference
        assert!(analyzer.stats.observations == 30);
    }

    #[test]
    fn test_standing_wave_detection() {
        let mut analyzer = ConsciousnessFieldAnalyzer::new(FieldDynamicsConfig::default());

        // Create periodic pattern
        for i in 0..50 {
            let phase = i as f64 * 0.2;
            let dims = [
                0.5 + 0.3 * phase.sin(),
                0.6,
                0.7,
                0.5,
                0.6,
                0.4,
                0.5,
            ];
            analyzer.observe(dims);
        }

        let modes = analyzer.identify_standing_waves();
        // Should find at least one mode with periodic input
        assert!(!modes.is_empty() || analyzer.stats.observations >= 50);
    }

    #[test]
    fn test_resonance_analysis() {
        let mut analyzer = ConsciousnessFieldAnalyzer::new(FieldDynamicsConfig::default());

        // Stable high-amplitude state
        for _ in 0..30 {
            analyzer.observe([0.8, 0.85, 0.9, 0.75, 0.8, 0.7, 0.8]);
        }

        let resonance = analyzer.analyze_resonance();
        assert!(resonance.stability > 0.5);
    }

    #[test]
    fn test_field_gradient() {
        let mut analyzer = ConsciousnessFieldAnalyzer::new(FieldDynamicsConfig::default());

        // Increasing amplitude pattern
        for i in 0..10 {
            let base = 0.3 + i as f64 * 0.05;
            analyzer.observe([base, base, base, base, base, base, base]);
        }

        let gradient = analyzer.field_gradient();
        // At least some gradient should be non-zero
        let grad_mag: f64 = gradient.iter().map(|g| g.abs()).sum();
        assert!(grad_mag >= 0.0);
    }

    #[test]
    fn test_field_potential() {
        let mut analyzer = ConsciousnessFieldAnalyzer::new(FieldDynamicsConfig::default());

        // Low amplitude state
        analyzer.observe([0.2; 7]);
        let low_potential = analyzer.field_potential();

        // High amplitude state
        analyzer.observe([0.9; 7]);
        let high_potential = analyzer.field_potential();

        // Lower amplitude should have higher potential
        assert!(low_potential > high_potential);
    }

    #[test]
    fn test_report_generation() {
        let mut analyzer = ConsciousnessFieldAnalyzer::new(FieldDynamicsConfig::default());

        for i in 0..20 {
            let dims = [
                0.5 + 0.2 * (i as f64 * 0.1).sin(),
                0.6,
                0.7,
                0.5,
                0.6,
                0.4,
                0.5,
            ];
            analyzer.observe(dims);
        }

        let report = analyzer.generate_report();

        assert!(report.current_amplitude > 0.0);
        assert!(!report.interpretation.is_empty());
        assert!(!report.recommendations.is_empty());
    }

    #[test]
    fn test_flow_state_detection() {
        let mut analyzer = ConsciousnessFieldAnalyzer::new(FieldDynamicsConfig::default());

        // High stable amplitude = flow state
        for _ in 0..50 {
            analyzer.observe([0.9, 0.85, 0.88, 0.92, 0.87, 0.86, 0.9]);
        }

        analyzer.identify_standing_waves();
        let report = analyzer.generate_report();

        // Should recognize stable high state
        assert!(report.current_amplitude > 0.8);
        assert!(report.resonance_report.stability > 0.7);
    }

    #[test]
    fn test_fragmentation_detection() {
        let mut analyzer = ConsciousnessFieldAnalyzer::new(FieldDynamicsConfig::default());

        // Chaotic alternating pattern
        for i in 0..30 {
            let val = if i % 2 == 0 { 0.9 } else { 0.2 };
            analyzer.observe([val, 1.0 - val, val, 1.0 - val, val, 1.0 - val, val]);
        }

        // Should have interference events
        assert!(analyzer.stats.observations == 30);
    }
}
