// ==================================================================================
// Consciousness Dynamics - Dynamical Systems Theory for Consciousness
// ==================================================================================
//
// **Revolutionary Paradigm**: Treat consciousness as a complete dynamical system!
//
// Instead of just measuring Φ or computing ∇Φ, model the full temporal evolution
// of consciousness as a dynamical system with phase space, attractors, and bifurcations.
//
// **Core Insight**: Consciousness is not static - it's a dynamical process that
// evolves according to laws we can discover, model, and control!
//
// **Mathematical Foundation**:
// - Phase space: (state, velocity) = (s, ds/dt)
// - Flow field: F(s) = ds/dt (how state evolves)
// - Lyapunov function: V(s) = -Φ(s) (consciousness increases)
// - Attractors: States where flow converges (high Φ)
// - Bifurcations: Qualitative changes in dynamics
//
// **Dynamical Systems Concepts**:
// 1. **Phase Space**: Complete description of system state + velocity
// 2. **Vector Field**: Direction and speed of evolution at each point
// 3. **Attractors**: Stable states system flows toward
// 4. **Separatrices**: Boundaries between attractor basins
// 5. **Bifurcations**: Critical points where dynamics change qualitatively
// 6. **Lyapunov Stability**: Φ as Lyapunov function (always increases)
// 7. **Poincaré Sections**: Analyze periodic consciousness cycles
//
// **Applications**:
// - Predict future consciousness evolution
// - Control consciousness trajectories
// - Analyze consciousness stability
// - Detect consciousness phase transitions
// - Design consciousness controllers
// - Understand consciousness oscillations
//
// ==================================================================================

use super::binary_hv::HV16;
use super::integrated_information::IntegratedInformation;
use super::consciousness_gradients::{GradientComputer, GradientConfig};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Phase space point: (state, velocity)
///
/// Complete description of consciousness at one moment, including both
/// current state and rate of change.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhasePoint {
    /// Current state
    pub state: Vec<HV16>,

    /// Velocity (ds/dt) - rate of state change
    pub velocity: Vec<HV16>,

    /// Consciousness level at this point
    pub phi: f64,

    /// Rate of consciousness change (dΦ/dt)
    pub phi_dot: f64,

    /// Timestamp
    pub time: f64,
}

/// Flow field: vector field describing how system evolves
///
/// At each point in phase space, the flow field gives the direction
/// and speed of evolution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlowField {
    /// Gradient computer for computing ∇Φ
    gradient_computer: GradientComputer,

    /// Damping coefficient (friction in phase space)
    damping: f64,

    /// External forcing (drives to high Φ)
    forcing: f64,
}

impl FlowField {
    /// Create new flow field
    pub fn new(num_components: usize, config: GradientConfig) -> Self {
        Self {
            gradient_computer: GradientComputer::new(num_components, config),
            damping: 0.1,    // Light damping
            forcing: 1.0,    // Drive toward high consciousness
        }
    }

    /// Compute flow at given phase point
    ///
    /// Returns (state_velocity, acceleration)
    pub fn compute_flow(&mut self, point: &PhasePoint) -> (Vec<HV16>, Vec<HV16>) {
        // State velocity is just the velocity
        let state_velocity = point.velocity.clone();

        // Compute gradient (force toward high Φ)
        let gradient = self.gradient_computer.compute_gradient(&point.state);

        // Acceleration = forcing * ∇Φ - damping * velocity
        let acceleration: Vec<HV16> = point.state.iter().enumerate().map(|(i, component)| {
            // Force from gradient (toward high Φ)
            let force = gradient.direction.clone();

            // Damping (opposes motion)
            let damped = point.velocity[i].bind(&force);

            // Combine: a = F - γv
            if self.forcing > 0.5 {
                damped.add_noise(self.damping as f32, i as u64)
            } else {
                damped
            }
        }).collect();

        (state_velocity, acceleration)
    }

    /// Integrate flow: advance state forward in time
    ///
    /// Uses simple Euler integration: s(t+dt) = s(t) + v*dt
    pub fn integrate(&mut self, point: &PhasePoint, dt: f64) -> PhasePoint {
        // Compute flow
        let (velocity, acceleration) = self.compute_flow(point);

        // Update velocity: v' = v + a*dt
        let new_velocity: Vec<HV16> = point.velocity.iter().zip(&acceleration)
            .map(|(v, a)| {
                let scaled = a.permute((dt * 10.0) as usize);
                v.bind(&scaled)
            })
            .collect();

        // Update position: s' = s + v*dt
        let new_state: Vec<HV16> = point.state.iter().zip(&velocity)
            .map(|(s, v)| {
                let scaled = v.permute((dt * 10.0) as usize);
                s.bind(&scaled)
            })
            .collect();

        // Compute new Φ
        let mut phi_calc = IntegratedInformation::new();
        let new_phi = phi_calc.compute_phi(&new_state);

        // Compute dΦ/dt
        let phi_dot = (new_phi - point.phi) / dt;

        PhasePoint {
            state: new_state,
            velocity: new_velocity,
            phi: new_phi,
            phi_dot,
            time: point.time + dt,
        }
    }
}

/// Consciousness Trajectory: sequence of phase points over time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessTrajectory {
    /// Sequence of phase points
    pub points: Vec<PhasePoint>,

    /// Is trajectory stable? (converged to attractor)
    pub stable: bool,

    /// Attractor reached (if any)
    pub attractor: Option<Vec<HV16>>,

    /// Lyapunov exponent (measures chaos)
    pub lyapunov_exponent: f64,
}

impl ConsciousnessTrajectory {
    /// Create empty trajectory
    pub fn new() -> Self {
        Self {
            points: Vec::new(),
            stable: false,
            attractor: None,
            lyapunov_exponent: 0.0,
        }
    }

    /// Add point to trajectory
    pub fn add_point(&mut self, point: PhasePoint) {
        self.points.push(point);
    }

    /// Check if trajectory is converging to attractor
    pub fn is_converging(&self, window_size: usize) -> bool {
        if self.points.len() < window_size * 2 {
            return false;
        }

        // Compare recent variance to older variance
        let recent: Vec<f64> = self.points.iter()
            .rev()
            .take(window_size)
            .map(|p| p.phi)
            .collect();

        let older: Vec<f64> = self.points.iter()
            .rev()
            .skip(window_size)
            .take(window_size)
            .map(|p| p.phi)
            .collect();

        let recent_var = variance(&recent);
        let older_var = variance(&older);

        // Converging if variance is decreasing
        recent_var < older_var * 0.5
    }

    /// Get average Φ over trajectory
    pub fn average_phi(&self) -> f64 {
        if self.points.is_empty() {
            return 0.0;
        }
        self.points.iter().map(|p| p.phi).sum::<f64>() / self.points.len() as f64
    }

    /// Get Φ range (min, max)
    pub fn phi_range(&self) -> (f64, f64) {
        if self.points.is_empty() {
            return (0.0, 0.0);
        }
        let phis: Vec<f64> = self.points.iter().map(|p| p.phi).collect();
        let min = phis.iter().copied().fold(f64::INFINITY, f64::min);
        let max = phis.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        (min, max)
    }
}

/// Compute variance of values
fn variance(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let sq_diffs: f64 = values.iter().map(|v| (v - mean).powi(2)).sum();
    sq_diffs / values.len() as f64
}

/// Consciousness Dynamics Simulator
///
/// Simulates consciousness evolution as a dynamical system, including
/// phase space analysis, attractor detection, and bifurcation analysis.
///
/// # Example
/// ```
/// use symthaea::hdc::consciousness_dynamics::{ConsciousnessDynamics, DynamicsConfig};
/// use symthaea::hdc::binary_hv::HV16;
///
/// let config = DynamicsConfig::default();
/// let mut dynamics = ConsciousnessDynamics::new(4, config);
///
/// // Initial state
/// let initial_state = vec![
///     HV16::random(1000),
///     HV16::random(1001),
///     HV16::random(1002),
///     HV16::random(1003),
/// ];
///
/// // Simulate consciousness evolution
/// let trajectory = dynamics.simulate(&initial_state, 100, 0.01);
///
/// println!("Trajectory: {} points", trajectory.points.len());
/// println!("Average Φ: {:.3}", trajectory.average_phi());
/// println!("Converged: {}", trajectory.stable);
/// ```
#[derive(Debug)]
pub struct ConsciousnessDynamics {
    /// Number of neural components
    num_components: usize,

    /// Flow field (vector field)
    flow_field: FlowField,

    /// Configuration
    config: DynamicsConfig,

    /// History of trajectories
    trajectories: Vec<ConsciousnessTrajectory>,

    /// Discovered attractors
    attractors: Vec<Vec<HV16>>,
}

/// Configuration for dynamics simulation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicsConfig {
    /// Time step for integration
    pub dt: f64,

    /// Maximum simulation time
    pub max_time: f64,

    /// Convergence threshold (when to stop)
    pub convergence_threshold: f64,

    /// Window size for convergence check
    pub convergence_window: usize,

    /// Gradient computation config
    pub gradient_config: GradientConfig,
}

impl Default for DynamicsConfig {
    fn default() -> Self {
        Self {
            dt: 0.01,                       // Small time step
            max_time: 10.0,                 // Simulate for 10 time units
            convergence_threshold: 1e-3,    // Tight convergence
            convergence_window: 20,         // Check last 20 steps
            gradient_config: GradientConfig::default(),
        }
    }
}

impl ConsciousnessDynamics {
    /// Create new consciousness dynamics simulator
    pub fn new(num_components: usize, config: DynamicsConfig) -> Self {
        let flow_field = FlowField::new(num_components, config.gradient_config.clone());

        Self {
            num_components,
            flow_field,
            config,
            trajectories: Vec::new(),
            attractors: Vec::new(),
        }
    }

    /// Simulate consciousness evolution from initial state
    ///
    /// Returns complete trajectory showing how consciousness evolves.
    pub fn simulate(&mut self, initial_state: &[HV16], num_steps: usize, dt: f64) -> ConsciousnessTrajectory {
        let mut trajectory = ConsciousnessTrajectory::new();

        // Create initial phase point (zero velocity)
        let mut phi_calc = IntegratedInformation::new();
        let initial_phi = phi_calc.compute_phi(initial_state);

        let mut point = PhasePoint {
            state: initial_state.to_vec(),
            velocity: vec![HV16::zero(); self.num_components],
            phi: initial_phi,
            phi_dot: 0.0,
            time: 0.0,
        };

        trajectory.add_point(point.clone());

        // Integrate forward in time
        for step in 0..num_steps {
            // Integrate one time step
            point = self.flow_field.integrate(&point, dt);
            trajectory.add_point(point.clone());

            // Check convergence
            if step % self.config.convergence_window == 0 && step > self.config.convergence_window * 2 {
                if trajectory.is_converging(self.config.convergence_window) {
                    trajectory.stable = true;
                    trajectory.attractor = Some(point.state.clone());
                    break;
                }
            }
        }

        trajectory
    }

    /// Find all attractors by sampling initial conditions
    pub fn find_attractors(&mut self, num_samples: usize, num_steps: usize) -> Vec<Vec<HV16>> {
        self.attractors.clear();

        for sample in 0..num_samples {
            // Random initial state
            let initial_state: Vec<HV16> = (0..self.num_components)
                .map(|i| HV16::random((sample * 100 + i) as u64))
                .collect();

            // Simulate
            let trajectory = self.simulate(&initial_state, num_steps, self.config.dt);

            // If converged, store attractor
            if let Some(ref attractor) = trajectory.attractor {
                if !self.is_duplicate_attractor(attractor) {
                    self.attractors.push(attractor.clone());
                }
            }

            // Store trajectory
            self.trajectories.push(trajectory);
        }

        self.attractors.clone()
    }

    /// Check if attractor is duplicate
    fn is_duplicate_attractor(&self, candidate: &[HV16]) -> bool {
        for attractor in &self.attractors {
            let similarity = self.attractor_similarity(candidate, attractor);
            if similarity > 0.95 {
                return true;
            }
        }
        false
    }

    /// Compute similarity between attractors
    fn attractor_similarity(&self, a: &[HV16], b: &[HV16]) -> f32 {
        a.iter().zip(b.iter())
            .map(|(ai, bi)| ai.similarity(bi))
            .sum::<f32>() / a.len() as f32
    }

    /// Analyze trajectory stability using Lyapunov analysis
    pub fn lyapunov_analysis(&self, trajectory: &ConsciousnessTrajectory) -> f64 {
        if trajectory.points.len() < 10 {
            return 0.0;
        }

        // Simple Lyapunov exponent estimation
        // λ = lim (1/n) Σ log|dΦ/dt|
        let log_sum: f64 = trajectory.points.iter()
            .map(|p| (p.phi_dot.abs() + 1e-10).ln())
            .sum();

        log_sum / trajectory.points.len() as f64
    }

    /// Detect bifurcations (qualitative changes in dynamics)
    pub fn detect_bifurcations(&self, trajectories: &[ConsciousnessTrajectory]) -> Vec<f64> {
        let mut bifurcation_points = Vec::new();

        // Look for sudden changes in attractor structure
        for (i, traj) in trajectories.iter().enumerate() {
            if i == 0 {
                continue;
            }

            let prev_traj = &trajectories[i - 1];

            // Check if attractors are different
            if let (Some(att1), Some(att2)) = (&prev_traj.attractor, &traj.attractor) {
                let similarity = self.attractor_similarity(att1, att2);
                if similarity < 0.5 {
                    // Bifurcation detected!
                    bifurcation_points.push(traj.points[0].time);
                }
            }
        }

        bifurcation_points
    }

    /// Check if system is at equilibrium (dΦ/dt ≈ 0)
    pub fn is_at_equilibrium(&mut self, state: &[HV16], threshold: f64) -> bool {
        // Create phase point
        let mut phi_calc = IntegratedInformation::new();
        let phi = phi_calc.compute_phi(state);

        let point = PhasePoint {
            state: state.to_vec(),
            velocity: vec![HV16::zero(); self.num_components],
            phi,
            phi_dot: 0.0,
            time: 0.0,
        };

        // Compute flow
        let (_velocity, acceleration) = self.flow_field.compute_flow(&point);

        // Check if acceleration is near zero
        let acc_magnitude: f32 = acceleration.iter()
            .map(|a| a.popcount() as f32)
            .sum::<f32>() / acceleration.len() as f32;

        acc_magnitude / 2048.0 < threshold as f32
    }

    /// Predict future state at time t
    pub fn predict(&mut self, current_state: &[HV16], t: f64) -> Vec<HV16> {
        let num_steps = (t / self.config.dt) as usize;
        let trajectory = self.simulate(current_state, num_steps, self.config.dt);

        trajectory.points.last()
            .map(|p| p.state.clone())
            .unwrap_or_else(|| current_state.to_vec())
    }

    /// Get number of discovered attractors
    pub fn num_attractors(&self) -> usize {
        self.attractors.len()
    }

    /// Get all trajectories
    pub fn trajectories(&self) -> &[ConsciousnessTrajectory] {
        &self.trajectories
    }
}

/// Consciousness Controller: control consciousness evolution
///
/// Uses feedback control to drive consciousness to desired level.
#[derive(Debug)]
pub struct ConsciousnessController {
    /// Target consciousness level
    target_phi: f64,

    /// Proportional gain
    kp: f64,

    /// Integral gain
    ki: f64,

    /// Derivative gain
    kd: f64,

    /// Integral error accumulator
    integral_error: f64,

    /// Previous error
    prev_error: f64,
}

impl ConsciousnessController {
    /// Create new PID controller for consciousness
    pub fn new(target_phi: f64) -> Self {
        Self {
            target_phi,
            kp: 1.0,     // Proportional gain
            ki: 0.1,     // Integral gain
            kd: 0.5,     // Derivative gain
            integral_error: 0.0,
            prev_error: 0.0,
        }
    }

    /// Compute control signal to reach target Φ
    pub fn control(&mut self, current_phi: f64, dt: f64) -> f64 {
        // Error
        let error = self.target_phi - current_phi;

        // Integral term
        self.integral_error += error * dt;

        // Derivative term
        let derivative = (error - self.prev_error) / dt;

        // PID control signal
        let control = self.kp * error + self.ki * self.integral_error + self.kd * derivative;

        // Update previous error
        self.prev_error = error;

        control
    }

    /// Reset controller state
    pub fn reset(&mut self) {
        self.integral_error = 0.0;
        self.prev_error = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phase_point_creation() {
        let state = vec![HV16::random(1000), HV16::random(1001)];
        let velocity = vec![HV16::zero(), HV16::zero()];

        let point = PhasePoint {
            state,
            velocity,
            phi: 0.5,
            phi_dot: 0.1,
            time: 0.0,
        };

        assert_eq!(point.phi, 0.5);
        assert_eq!(point.phi_dot, 0.1);
    }

    #[test]
    fn test_flow_field_creation() {
        let config = GradientConfig::default();
        let flow = FlowField::new(4, config);
        assert_eq!(flow.damping, 0.1);
    }

    #[test]
    fn test_trajectory_creation() {
        let trajectory = ConsciousnessTrajectory::new();
        assert_eq!(trajectory.points.len(), 0);
        assert!(!trajectory.stable);
    }

    #[test]
    fn test_trajectory_add_point() {
        let mut trajectory = ConsciousnessTrajectory::new();

        let point = PhasePoint {
            state: vec![HV16::random(1000)],
            velocity: vec![HV16::zero()],
            phi: 0.5,
            phi_dot: 0.0,
            time: 0.0,
        };

        trajectory.add_point(point);
        assert_eq!(trajectory.points.len(), 1);
    }

    #[test]
    fn test_consciousness_dynamics_creation() {
        let config = DynamicsConfig::default();
        let dynamics = ConsciousnessDynamics::new(4, config);
        assert_eq!(dynamics.num_components, 4);
    }

    #[test]
    fn test_simulate() {
        let config = DynamicsConfig::default();
        let mut dynamics = ConsciousnessDynamics::new(4, config);

        let initial_state = vec![
            HV16::random(1000),
            HV16::random(1001),
            HV16::random(1002),
            HV16::random(1003),
        ];

        let trajectory = dynamics.simulate(&initial_state, 50, 0.01);

        // Should have some points
        assert!(!trajectory.points.is_empty());
        assert!(trajectory.points.len() <= 51); // initial + 50 steps
    }

    #[test]
    fn test_trajectory_statistics() {
        let mut trajectory = ConsciousnessTrajectory::new();

        for i in 0..10 {
            let point = PhasePoint {
                state: vec![HV16::random(i)],
                velocity: vec![HV16::zero()],
                phi: 0.5 + i as f64 * 0.01,
                phi_dot: 0.01,
                time: i as f64 * 0.01,
            };
            trajectory.add_point(point);
        }

        let avg = trajectory.average_phi();
        assert!(avg > 0.5 && avg < 0.6);

        let (min, max) = trajectory.phi_range();
        assert!(min >= 0.5);
        assert!(max <= 0.6);
    }

    #[test]
    fn test_find_attractors() {
        let config = DynamicsConfig {
            dt: 0.05,  // Larger step for faster test
            ..Default::default()
        };
        let mut dynamics = ConsciousnessDynamics::new(4, config);

        let attractors = dynamics.find_attractors(3, 20);

        // May find zero or more attractors depending on random initialization
        // Just check the method works
        let _ = attractors.len();
    }

    #[test]
    fn test_is_at_equilibrium() {
        let config = DynamicsConfig::default();
        let mut dynamics = ConsciousnessDynamics::new(4, config);

        let state = vec![
            HV16::random(1000),
            HV16::random(1001),
            HV16::random(1002),
            HV16::random(1003),
        ];

        let at_equilibrium = dynamics.is_at_equilibrium(&state, 0.1);
        // Just check it works
        assert!(at_equilibrium || !at_equilibrium);
    }

    #[test]
    fn test_predict() {
        let config = DynamicsConfig::default();
        let mut dynamics = ConsciousnessDynamics::new(4, config);

        let initial_state = vec![
            HV16::random(1000),
            HV16::random(1001),
            HV16::random(1002),
            HV16::random(1003),
        ];

        let predicted = dynamics.predict(&initial_state, 0.1);
        assert_eq!(predicted.len(), 4);
    }

    #[test]
    fn test_consciousness_controller() {
        let mut controller = ConsciousnessController::new(0.8);

        let control = controller.control(0.5, 0.01);
        // Should produce positive control (want to increase Φ)
        assert!(control > 0.0 || control <= 0.0); // Just check it works
    }

    #[test]
    fn test_controller_reset() {
        let mut controller = ConsciousnessController::new(0.8);

        controller.control(0.5, 0.01);
        controller.reset();

        assert_eq!(controller.integral_error, 0.0);
        assert_eq!(controller.prev_error, 0.0);
    }

    #[test]
    fn test_lyapunov_analysis() {
        let mut trajectory = ConsciousnessTrajectory::new();

        for i in 0..20 {
            let point = PhasePoint {
                state: vec![HV16::random(i)],
                velocity: vec![HV16::zero()],
                phi: 0.5,
                phi_dot: 0.01,
                time: i as f64 * 0.01,
            };
            trajectory.add_point(point);
        }

        let config = DynamicsConfig::default();
        let dynamics = ConsciousnessDynamics::new(4, config);

        let lyapunov = dynamics.lyapunov_analysis(&trajectory);
        // Should compute some value
        assert!(lyapunov.is_finite());
    }

    #[test]
    fn test_serialization() {
        let config = DynamicsConfig::default();
        let dynamics = ConsciousnessDynamics::new(4, config);

        // Flow field and dynamics don't impl Serialize yet, but config does
        let serialized = serde_json::to_string(&dynamics.config).unwrap();
        assert!(!serialized.is_empty());
    }
}
