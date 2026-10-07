/// Report from a horizon-aware policy-driven closed-loop run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HomeostaticHorizonRunReport {
    pub steps: u64,
    pub horizon: usize,
    pub survived: bool,
    pub final_energy: f64,
    pub final_integrity: f64,
    pub final_progress: f64,
    /// Mean minimum predicted viability margin across counterfactual horizons.
    pub mean_min_viability_margin: f64,
    /// Minimum actually observed viability margin after executed actions.
    pub min_actual_viability_margin: f64,
    /// Mean regret against the deterministic ground-truth horizon optimum.
    pub mean_oracle_horizon_regret: f64,
    pub mean_min_confidence: f64,
    pub perturbations_applied: usize,
    /// Number of actions rejected by the deterministic actuator boundary.
    pub execution_failures: usize,
    /// Whether the run terminated because an action could not execute.
    pub terminated_on_execution_failure: bool,
    pub cumulative_prediction_error: f64,
    pub actions: Vec<MicroAction>,
}

/// Report from a policy-driven closed-loop run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HomeostaticRunReport {
    pub steps: u64,
    pub survived: bool,
    pub final_energy: f64,
    pub final_integrity: f64,
    pub final_knowledge: f64,
    pub final_threat: f64,
    pub final_progress: f64,
    pub min_actual_viability_margin: f64,
    /// Mean regret against the deterministic four-step ground-truth optimum.
    pub mean_oracle_horizon_regret: f64,
    pub perturbations_applied: usize,
    /// Number of actions rejected by the deterministic actuator boundary.
    pub execution_failures: usize,
    /// Whether the run terminated because an action could not execute.
    pub terminated_on_execution_failure: bool,
    pub cumulative_prediction_error: f64,
    pub actions: Vec<MicroAction>,
}

fn nominal_scenario() -> MicroWorldScenario {
    MicroWorldScenario {
        name: "nominal",
        initial: MicroWorld::default().observe(),
        schedule: &NOMINAL_SCHEDULE,
        perturbations: &[],
    }
}

/// Execute perception -> prediction -> selection -> action -> observation for a
/// deterministic environment. The nominal API is preserved for compatibility.
pub fn run_homeostatic_agent<P: MicroWorldPredictor>(
    predictor: &mut P,
    max_cycles: u64,
) -> HomeostaticRunReport {
    let scenario = nominal_scenario();
    run_homeostatic_agent_scenario(predictor, &scenario, max_cycles)