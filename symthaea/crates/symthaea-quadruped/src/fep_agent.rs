use crate::types::QuadrupedState;
pub struct QuadrupedFepResult { pub tau_factor: f32, pub free_energy: f64 }
pub struct ActiveInferenceQuadrupedAgent { agent: symthaea_fep::ActiveInferenceAgent }
impl ActiveInferenceQuadrupedAgent {
    pub fn new() -> Self { let c = symthaea_fep::ActiveInferenceAgentConfig { state_dim: 12, obs_dim: 12, num_actions: 12, belief_learning_rate: 0.1, planning_horizon: 1, action_temperature: 0.5, ..symthaea_fep::ActiveInferenceAgentConfig::default() }; Self { agent: symthaea_fep::ActiveInferenceAgent::new(c) } }
    pub fn tick(&mut self, s: &QuadrupedState) -> QuadrupedFepResult { let he = (0.35 - s.height()).abs(); let ar: f64 = s.base_angular_velocity.iter().map(|v| v.abs()).sum(); let fe = he * 10.0 + ar * 5.0; QuadrupedFepResult { tau_factor: if he > 0.1 { 0.85 } else { 1.0 }, free_energy: fe } }
    pub fn reset(&mut self) { *self = Self::new(); }
}
impl Default for ActiveInferenceQuadrupedAgent { fn default() -> Self { Self::new() } }
