// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Module registry: manages pluggable simulation modules ordered by tick phase.
//!
//! This is the incremental migration path from the monolithic `run()` loop to
//! a fully modular architecture. Modules can be registered, ordered by phase,
//! and ticked each step. The existing tick loop continues to run alongside
//! registered modules — no big-bang rewrite needed.
//!
//! # Migration Strategy
//!
//! 1. Create `ModuleRegistry` with `register()` and `tick_all()`
//! 2. Extract ONE existing phase as a `SimulationModule` (proof of concept)
//! 3. Gradually migrate remaining phases (one at a time, each PR)
//! 4. Eventually, `run()` becomes just `registry.tick_all()` + epoch eval

use luminous_sim_core::module_trait::{
    ModuleId, ModuleOutputs, ReportSection, SimulationModule, TickPhase,
};
use std::collections::HashMap;

/// Registry of simulation modules, ordered by tick phase.
pub struct ModuleRegistry {
    /// Registered modules, sorted by phase.
    modules: Vec<Box<dyn SimulationModule>>,
    /// Whether modules have been sorted since last registration.
    sorted: bool,
    /// Accumulated feedback signals from all modules (cleared each tick).
    feedback: HashMap<String, f64>,
    /// Accumulated outputs from the current tick (for the engine to apply).
    pending_outputs: Vec<(ModuleId, ModuleOutputs)>,
}

impl ModuleRegistry {
    pub fn new() -> Self {
        Self {
            modules: Vec::new(),
            sorted: true,
            feedback: HashMap::new(),
            pending_outputs: Vec::new(),
        }
    }

    /// Register a module. Modules are sorted by phase before first tick.
    pub fn register(&mut self, module: Box<dyn SimulationModule>) {
        self.modules.push(module);
        self.sorted = false;
    }

    /// Number of registered modules.
    pub fn module_count(&self) -> usize {
        self.modules.len()
    }

    /// Sort modules by phase (ascending). Called automatically on first tick.
    fn ensure_sorted(&mut self) {
        if !self.sorted {
            self.modules.sort_by_key(|m| m.phase());
            self.sorted = true;
        }
    }

    /// Tick all registered modules in phase order.
    /// Returns accumulated outputs for the engine to apply to world state.
    pub fn tick_all(&mut self, tick: u32) -> Vec<(ModuleId, ModuleOutputs)> {
        self.ensure_sorted();
        self.feedback.clear();
        self.pending_outputs.clear();

        for module in &mut self.modules {
            match module.tick(tick, &self.feedback) {
                Ok(outputs) => {
                    // Merge feedback signals for subsequent modules
                    for (key, value) in &outputs.feedback_signals {
                        self.feedback.insert(key.clone(), *value);
                    }
                    self.pending_outputs.push((module.id(), outputs));
                }
                Err(e) => {
                    eprintln!("Module {} tick failed: {}", module.id(), e);
                }
            }
        }

        std::mem::take(&mut self.pending_outputs)
    }

    /// Collect report sections from all modules.
    pub fn collect_reports(&self) -> Vec<ReportSection> {
        self.modules
            .iter()
            .filter_map(|m| m.report_section())
            .collect()
    }

    /// List registered modules with their phases (sorts if needed).
    pub fn list_modules(&mut self) -> Vec<(ModuleId, &str, TickPhase)> {
        self.ensure_sorted();
        self.modules
            .iter()
            .map(|m| (m.id(), m.name(), m.phase()))
            .collect()
    }

    /// Get current feedback signals (read by modules during tick).
    pub fn feedback(&self) -> &HashMap<String, f64> {
        &self.feedback
    }
}

impl Default for ModuleRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use luminous_sim_core::module_trait::ModuleAssessment;

    /// A simple test module that counts ticks and emits feedback.
    struct CounterModule {
        count: u32,
        phase: TickPhase,
    }

    impl CounterModule {
        fn new(phase: TickPhase) -> Self {
            Self { count: 0, phase }
        }
    }

    impl SimulationModule for CounterModule {
        fn id(&self) -> ModuleId {
            "counter"
        }
        fn name(&self) -> &str {
            "Counter Module"
        }
        fn phase(&self) -> TickPhase {
            self.phase
        }

        fn tick(
            &mut self,
            _tick: u32,
            _feedback: &HashMap<String, f64>,
        ) -> Result<ModuleOutputs, String> {
            self.count += 1;
            let mut outputs = ModuleOutputs::default();
            outputs
                .feedback_signals
                .insert("counter_value".into(), self.count as f64);
            outputs
                .metrics
                .push(("ticks_counted".into(), self.count as f64));
            Ok(outputs)
        }

        fn report_section(&self) -> Option<ReportSection> {
            Some(ReportSection {
                module_id: "counter".into(),
                module_name: "Counter Module".into(),
                summary: format!("Counted {} ticks", self.count),
                metrics: vec![("total_ticks".into(), self.count as f64)],
                warnings: vec![],
            })
        }

        fn honest_assessment(&self) -> Option<ModuleAssessment> {
            Some(ModuleAssessment {
                module_id: "counter".into(),
                confidence: "High — it's just a counter".into(),
                limitations: vec!["Only counts, doesn't simulate anything".into()],
            })
        }
    }

    /// A module that reads feedback from the counter.
    struct ReaderModule {
        last_counter: f64,
    }

    impl SimulationModule for ReaderModule {
        fn id(&self) -> ModuleId {
            "reader"
        }
        fn name(&self) -> &str {
            "Reader Module"
        }
        fn phase(&self) -> TickPhase {
            TickPhase::Economy
        } // Later phase

        fn tick(
            &mut self,
            _tick: u32,
            feedback: &HashMap<String, f64>,
        ) -> Result<ModuleOutputs, String> {
            self.last_counter = feedback.get("counter_value").copied().unwrap_or(0.0);
            Ok(ModuleOutputs::default())
        }
    }

    #[test]
    fn test_registry_basic() {
        let mut registry = ModuleRegistry::new();
        assert_eq!(registry.module_count(), 0);

        registry.register(Box::new(CounterModule::new(TickPhase::Viability)));
        assert_eq!(registry.module_count(), 1);
    }

    #[test]
    fn test_registry_tick_produces_outputs() {
        let mut registry = ModuleRegistry::new();
        registry.register(Box::new(CounterModule::new(TickPhase::Viability)));

        let outputs = registry.tick_all(0);
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].0, "counter");
        assert_eq!(outputs[0].1.metrics.len(), 1);
        assert!((outputs[0].1.metrics[0].1 - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_registry_phase_ordering() {
        let mut registry = ModuleRegistry::new();
        // Register in wrong order — should be sorted by phase
        registry.register(Box::new(CounterModule::new(TickPhase::Disasters)));
        registry.register(Box::new(CounterModule::new(TickPhase::Viability)));
        registry.register(Box::new(CounterModule::new(TickPhase::Economy)));

        let modules = registry.list_modules();
        assert_eq!(modules[0].2, TickPhase::Viability);
        assert_eq!(modules[1].2, TickPhase::Disasters);
        assert_eq!(modules[2].2, TickPhase::Economy);
    }

    #[test]
    fn test_feedback_flows_between_modules() {
        let mut registry = ModuleRegistry::new();
        registry.register(Box::new(CounterModule::new(TickPhase::Viability)));
        registry.register(Box::new(ReaderModule { last_counter: 0.0 }));

        // After tick, reader should have received counter's feedback
        registry.tick_all(0);
        let feedback = registry.feedback();
        assert!(feedback.contains_key("counter_value"));
        assert!((feedback["counter_value"] - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_collect_reports() {
        let mut registry = ModuleRegistry::new();
        registry.register(Box::new(CounterModule::new(TickPhase::Viability)));

        // Tick a few times
        for tick in 0..5 {
            registry.tick_all(tick);
        }

        let reports = registry.collect_reports();
        assert_eq!(reports.len(), 1);
        assert_eq!(reports[0].module_id, "counter");
        assert!(reports[0].summary.contains("5 ticks"));
    }

    #[test]
    fn test_multiple_ticks_accumulate() {
        let mut registry = ModuleRegistry::new();
        registry.register(Box::new(CounterModule::new(TickPhase::Viability)));

        for tick in 0..10 {
            registry.tick_all(tick);
        }

        let reports = registry.collect_reports();
        assert!(
            (reports[0].metrics[0].1 - 10.0).abs() < 0.001,
            "Should count 10 ticks: {:?}",
            reports[0].metrics
        );
    }
}
