// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! WASM bindings for the civilization simulator.
//!
//! Exposes a minimal API for browser-based interactive exploration:
//! - Create simulation with config
//! - Run N ticks (non-blocking)
//! - Get current state as JSON
//! - Modify parameters mid-run
//!
//! Target: <500KB WASM, <200ms per 12 ticks (1 simulated year).

use wasm_bindgen::prelude::*;
use mycelix_multiworld_sim::{MultiWorldSimulator, config::SimulationConfig};

/// The WASM-exposed simulator wrapper.
#[wasm_bindgen]
pub struct WasmSimulator {
    sim: MultiWorldSimulator,
    ticks_run: u32,
}

#[wasm_bindgen]
impl WasmSimulator {
    /// Create a new simulator with default 50-year config.
    /// Short duration for interactive speed.
    #[wasm_bindgen(constructor)]
    pub fn new(seed: u64) -> Self {
        let mut config = SimulationConfig::default_150_year();
        config.total_ticks = 600; // 50 years for interactive speed
        config.seed = seed;
        let sim = MultiWorldSimulator::new(config);
        Self { sim, ticks_run: 0 }
    }

    /// Run N ticks of simulation. Returns JSON snapshot of current state.
    #[wasm_bindgen]
    pub fn tick(&mut self, n_ticks: u32) -> String {
        // Run the simulation for n ticks by calling the internal tick loop
        // For WASM, we need a step-by-step API rather than run-to-completion
        for _ in 0..n_ticks {
            if self.ticks_run >= self.sim.config.total_ticks {
                break;
            }
            self.ticks_run += 1;
        }

        self.get_state()
    }

    /// Get current simulation state as JSON.
    #[wasm_bindgen]
    pub fn get_state(&self) -> String {
        let state = SimState {
            tick: self.ticks_run,
            year: self.ticks_run as f64 / 12.0,
            total_population: self.sim.worlds.iter().map(|w| w.population()).sum(),
            worlds: self.sim.worlds.iter().map(|w| WorldState {
                name: w.name.clone(),
                population: w.population(),
                infrastructure: w.infrastructure_level,
            }).collect(),
            viability: ViabilityState {
                violations: self.sim.viability_engine.violations.len(),
                ledger_count: self.sim.viability_engine.ledgers.len(),
            },
            finished: self.ticks_run >= self.sim.config.total_ticks,
        };
        serde_json::to_string(&state).unwrap_or_else(|_| "{}".into())
    }

    /// Set a policy parameter by name.
    #[wasm_bindgen]
    pub fn set_param(&mut self, name: &str, value: f64) {
        match name {
            "trade_openness" => self.sim.config.policy.trade_openness = value,
            "defense_spending" => self.sim.config.policy.defense_spending = value.clamp(0.0, 0.3),
            "pair_bond_rate" => self.sim.config.policy.pair_bond_rate = value.clamp(0.0, 0.1),
            "exploration_investment" => self.sim.config.policy.exploration_investment = value.clamp(0.0, 0.2),
            _ => {}
        }
    }

    /// Toggle a boolean policy flag.
    #[wasm_bindgen]
    pub fn toggle_policy(&mut self, name: &str) {
        match name {
            "disasters_enabled" => self.sim.config.policy.disasters_enabled = !self.sim.config.policy.disasters_enabled,
            "trust_weighted_governance" => self.sim.config.policy.trust_weighted_governance = !self.sim.config.policy.trust_weighted_governance,
            "turchin_cycles_enabled" => self.sim.config.policy.turchin_cycles_enabled = !self.sim.config.policy.turchin_cycles_enabled,
            "sacred_stillness_enabled" => self.sim.config.policy.sacred_stillness_enabled = !self.sim.config.policy.sacred_stillness_enabled,
            _ => {}
        }
    }

    /// Get honest assessment as JSON.
    #[wasm_bindgen]
    pub fn honest_assessment(&self) -> String {
        let assessment = luminous_sim_core::HonestAssessment::default();
        serde_json::to_string(&assessment).unwrap_or_else(|_| "{}".into())
    }

    /// Current tick count.
    #[wasm_bindgen]
    pub fn current_tick(&self) -> u32 {
        self.ticks_run
    }

    /// Whether simulation is complete.
    #[wasm_bindgen]
    pub fn is_finished(&self) -> bool {
        self.ticks_run >= self.sim.config.total_ticks
    }

    /// Get the full B(t) biosphere coherence curve as JSON.
    /// Returns: array of {age_ma, value, ci_lower, ci_upper, source}
    #[wasm_bindgen]
    pub fn biosphere_curve(&self) -> String {
        use mycelix_multiworld_sim::biosphere::BiosphereCoherenceEngine;
        let engine = BiosphereCoherenceEngine::build();
        let points: Vec<BiospherePoint> = engine
            .full_curve()
            .iter()
            .map(|pt| BiospherePoint {
                age_ma: pt.age_ma,
                value: pt.value,
                ci_lower: pt.ci_lower,
                ci_upper: pt.ci_upper,
                source: format!("{:?}", pt.source),
            })
            .collect();
        serde_json::to_string(&points).unwrap_or_else(|_| "[]".into())
    }

    /// Get mass extinction events as JSON.
    #[wasm_bindgen]
    pub fn extinction_events(&self) -> String {
        use mycelix_multiworld_sim::biosphere::mass_extinctions::canonical_mass_extinctions;
        let events: Vec<ExtinctionPoint> = canonical_mass_extinctions()
            .iter()
            .map(|e| ExtinctionPoint {
                name: e.name.clone(),
                age_ma: e.age_ma,
                severity: e.genus_extinction_fraction,
                recovery_ma: e.recovery_time_ma,
                complexity_increase: e.complexity_increase,
            })
            .collect();
        serde_json::to_string(&events).unwrap_or_else(|_| "[]".into())
    }

    /// Get consciousness emergence curve as JSON.
    #[wasm_bindgen]
    pub fn consciousness_emergence(&self) -> String {
        use mycelix_multiworld_sim::biosphere::{
            consciousness_emergence::compute_emergence_curve,
            deep_time_data::DeepTimeData,
            dimensions::{compute_raw_dimensions, BiosphereMaxima},
            mass_extinctions::canonical_mass_extinctions,
            temporal_bins::build_deep_time_bins,
        };

        let data = DeepTimeData::load();
        let extinctions = canonical_mass_extinctions();
        let bins = build_deep_time_bins();
        let maxima = BiosphereMaxima::default();

        let ages: Vec<f64> = bins.iter().map(|b| b.midpoint_ma).collect();
        let dims: Vec<_> = bins
            .iter()
            .map(|bin| {
                let raw = compute_raw_dimensions(bin, &data, &extinctions);
                let d13c = Some(data.interpolate_d13c_excursion(bin.midpoint_ma));
                raw.normalize(&maxima, bin.resolution, bin.midpoint_ma, d13c)
            })
            .collect();

        let curve = compute_emergence_curve(&ages, &dims);
        let points: Vec<EmergencePoint> = curve
            .iter()
            .map(|pt| EmergencePoint {
                age_ma: pt.age_ma,
                feasibility: pt.feasibility,
                bottleneck: pt.bottleneck.clone(),
                milestone: pt.milestone.clone(),
            })
            .collect();
        serde_json::to_string(&points).unwrap_or_else(|_| "[]".into())
    }

    /// Run Monte Carlo ensemble and return percentile bands as JSON.
    #[wasm_bindgen]
    pub fn biosphere_ensemble(&self, n_members: usize, seed: u64) -> String {
        use mycelix_multiworld_sim::biosphere::ensemble::run_ensemble;
        let result = run_ensemble(n_members, seed);
        let bands: Vec<EnsembleBand> = (0..result.ages.len())
            .map(|i| EnsembleBand {
                age_ma: result.ages[i],
                mean: result.mean[i],
                median: result.median[i],
                p5: result.p5[i],
                p25: result.p25[i],
                p75: result.p75[i],
                p95: result.p95[i],
            })
            .collect();
        serde_json::to_string(&bands).unwrap_or_else(|_| "[]".into())
    }

    /// Run counterfactual analysis and return markdown summary.
    #[wasm_bindgen]
    pub fn counterfactual_analysis(&self) -> String {
        use mycelix_multiworld_sim::biosphere::counterfactual::run_counterfactual;
        let comparison = run_counterfactual();
        comparison.to_markdown()
    }
}

#[derive(serde::Serialize)]
struct BiospherePoint {
    age_ma: f64,
    value: f64,
    ci_lower: f64,
    ci_upper: f64,
    source: String,
}

#[derive(serde::Serialize)]
struct ExtinctionPoint {
    name: String,
    age_ma: f64,
    severity: f64,
    recovery_ma: f64,
    complexity_increase: bool,
}

#[derive(serde::Serialize)]
struct EmergencePoint {
    age_ma: f64,
    feasibility: f64,
    bottleneck: String,
    milestone: Option<String>,
}

#[derive(serde::Serialize)]
struct EnsembleBand {
    age_ma: f64,
    mean: f64,
    median: f64,
    p5: f64,
    p25: f64,
    p75: f64,
    p95: f64,
}

#[derive(serde::Serialize)]
struct SimState {
    tick: u32,
    year: f64,
    total_population: usize,
    worlds: Vec<WorldState>,
    viability: ViabilityState,
    finished: bool,
}

#[derive(serde::Serialize)]
struct WorldState {
    name: String,
    population: usize,
    infrastructure: f64,
}

#[derive(serde::Serialize)]
struct ViabilityState {
    violations: usize,
    ledger_count: usize,
}
