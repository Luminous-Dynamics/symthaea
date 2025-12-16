/*!
Liquid Time-Constant Network (LTC)
Continuous-time neurons with adaptive time constants

Differential equation: dx/dt = -x/τ + σ(Wx + b)
*/

use anyhow::Result;
use ndarray::{Array1, Array2};

/// Liquid network with continuous-time dynamics
pub struct LiquidNetwork {
    /// Number of neurons
    num_neurons: usize,

    /// Current neuron states
    state: Array1<f32>,

    /// Time constants (τ) per neuron
    tau: Array1<f32>,

    /// Weight matrix (sparse in production!)
    weights: Array2<f32>,

    /// Bias terms
    bias: Array1<f32>,

    /// Integration timestep
    dt: f32,

    /// Total evolution steps
    steps: usize,
}

impl LiquidNetwork {
    pub fn new(num_neurons: usize) -> Result<Self> {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        // Random initialization
        let state = Array1::zeros(num_neurons);

        // Time constants: uniform random [0.5, 2.0]
        let tau = Array1::from_iter(
            (0..num_neurons).map(|_| rng.gen_range(0.5..2.0))
        );

        // Sparse random weights
        let mut weights = Array2::zeros((num_neurons, num_neurons));
        for i in 0..num_neurons {
            for j in 0..num_neurons {
                if rng.gen::<f32>() < 0.1 {  // 10% connectivity
                    weights[[i, j]] = rng.gen_range(-1.0..1.0);
                }
            }
        }

        let bias = Array1::from_iter(
            (0..num_neurons).map(|_| rng.gen_range(-0.5..0.5))
        );

        Ok(Self {
            num_neurons,
            state,
            tau,
            weights,
            bias,
            dt: 0.01,  // 10ms timestep
            steps: 0,
        })
    }

    /// Inject external input (from HDC)
    pub fn inject(&mut self, input: &[f32]) -> Result<()> {
        // Add input to first N neurons
        let n = input.len().min(self.num_neurons);

        for i in 0..n {
            self.state[i] += input[i] * 0.1;  // Scaled input
        }

        Ok(())
    }

    /// Evolve network one timestep (continuous dynamics!)
    pub fn step(&mut self) -> Result<()> {
        // dx/dt = -x/τ + σ(Wx + b)

        // Compute weighted input: Wx + b
        let weighted_input = self.weights.dot(&self.state) + &self.bias;

        // Apply sigmoid activation
        let sigmoid_input = weighted_input.mapv(|x| 1.0 / (1.0 + (-x).exp()));

        // Continuous-time update: dx = (-x/τ + σ(Wx + b)) * dt
        let dx = (&sigmoid_input - &self.state / &self.tau) * self.dt;

        // Integrate: x += dx
        self.state += &dx;

        // Clip to [0, 1] range
        self.state.mapv_inplace(|x| x.max(0.0).min(1.0));

        self.steps += 1;

        Ok(())
    }

    /// Measure consciousness level (coherent activity)
    pub fn consciousness_level(&self) -> f32 {
        // Measure of synchronized, coherent activity

        // 1. Fraction of active neurons (> 0.5)
        let active_fraction = self.state
            .iter()
            .filter(|&&x| x > 0.5)
            .count() as f32 / self.num_neurons as f32;

        // 2. Variance (high variance = diverse, conscious)
        let mean = self.state.mean().unwrap_or(0.0);
        let variance = self.state
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>() / self.num_neurons as f32;

        // Combine: conscious if active AND diverse
        let consciousness = (active_fraction * variance.sqrt()).min(1.0);

        consciousness
    }

    /// Read current state as hypervector
    pub fn read_state(&self) -> Result<Vec<f32>> {
        Ok(self.state.to_vec())
    }

    /// Get neuron states for serialization
    pub fn neuron_states(&self) -> Vec<f32> {
        self.state.to_vec()
    }

    /// Activity summary
    pub fn activity_summary(&self) -> f32 {
        self.state.mean().unwrap_or(0.0)
    }

    /// Serialize for consciousness persistence
    pub fn serialize(&self) -> Result<Vec<u8>> {
        let data = (
            self.num_neurons,
            self.state.to_vec(),
            self.tau.to_vec(),
            self.bias.to_vec(),
            self.dt,
            self.steps,
        );

        Ok(bincode::serialize(&data)?)
    }

    /// Deserialize
    pub fn deserialize(data: &[u8]) -> Result<Self> {
        let (num_neurons, state_vec, tau_vec, bias_vec, dt, steps): (
            usize,
            Vec<f32>,
            Vec<f32>,
            Vec<f32>,
            f32,
            usize,
        ) = bincode::deserialize(data)?;

        let state = Array1::from_vec(state_vec);
        let tau = Array1::from_vec(tau_vec);
        let bias = Array1::from_vec(bias_vec);

        // Recreate weights (could be serialized too in production)
        let weights = Array2::zeros((num_neurons, num_neurons));

        Ok(Self {
            num_neurons,
            state,
            tau,
            weights,
            bias,
            dt,
            steps,
        })
    }
}
