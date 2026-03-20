//! Coding Agent: multi-step consciousness-gated coding loop.
//!
//! Drives Symthaea through read → reason → write → test → fix cycles,
//! using FEP active inference for tool selection and consciousness gating
//! for epistemic honesty.
//!
//! ## Architecture
//!
//! The `CodingAgent` wraps a `CognitiveLoopService` (not the full `Symthaea` facade)
//! and directly drives the cognitive cycle, interpreting motor commands as coding actions.
//! This keeps the agent lightweight while retaining consciousness gating, moral algebra,
//! and all cognitive subsystems.
//!
//! ## Generation Pipeline
//!
//! During the Generating/Fixing phases, the agent:
//! 1. Builds a generation prompt from task + observations + errors + code context
//! 2. Calls `IntelligentDispatcher::generate()` to select the optimal backend
//!    (Native/LocalLLM/CloudLLM) based on consciousness state
//! 3. Writes the generated code to disk
//! 4. Requests `cargo check` via the motor bridge to validate
//!
//! ## State Machine
//!
//! ```text
//! Understanding → Planning → Generating → Testing → Fixing ──→ Done
//!      ↑              ↑           │            │         │
//!      └──────────────┴───────────┘            └─────────┘
//! ```

pub mod types;
mod generation;
mod auto_fix;
mod quality_gate;
mod phase_executor;
mod telemetry;

pub use types::*;

use crate::action::primitives::{
    Atom, DispatchTier, Molecule, MoleculeExecutor, PlanProfile, PrimitiveValue,
};
use crate::action::{ActionOutcome, PolicyBundle, SandboxRoot};
use crate::coding_experience::{CodingExperience, CodingExperienceStore};
use crate::cognitive_loop::motor_output_bridge::{
    ActionType, MotorActionRequest, MotorOutputBridge, MotorOutputResult,
};
use crate::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService, CycleResult};
use crate::consciousness::fep_active_inference::MotorCommandType;
use crate::language::intelligent_dispatcher::{BackendTier, DispatchResult, IntelligentDispatcher};
use crate::language::llm_backend::GenerationParams;
use crate::mind::structured_thought::EpistemicStatus;
use std::path::PathBuf;

/// A consciousness-gated multi-step coding agent.
///
/// Wraps a `CognitiveLoopService` and drives it through iterative coding cycles.
/// Each iteration runs a cognitive cycle, interprets the FEP motor command,
/// and dispatches the appropriate action (read file, write code, run tests).
pub struct CodingAgent {
    /// The cognitive loop driving consciousness and FEP.
    pub(crate) cognitive_loop: CognitiveLoopService,
    /// Agent configuration.
    pub(crate) config: CodingAgentConfig,
    /// Current phase of the task.
    pub(crate) phase: TaskPhase,
    /// Current iteration count.
    pub(crate) iteration: usize,
    /// Consecutive failures in current phase.
    pub(crate) phase_failures: usize,
    /// Files modified during this run.
    pub(crate) files_modified: Vec<PathBuf>,
    /// Phi trace across iterations.
    pub(crate) phi_trace: Vec<f32>,
    /// Observations accumulated during the run.
    pub(crate) observations: Vec<String>,
    /// Errors accumulated during the run.
    pub(crate) errors: Vec<String>,
    /// Last test/check output for error context.
    pub(crate) last_test_output: Option<String>,
    /// The original task description.
    pub(crate) task: String,
    /// Intelligent dispatcher for consciousness-routed code generation.
    pub(crate) dispatcher: Option<IntelligentDispatcher>,
    /// Last dispatch result for telemetry.
    pub(crate) last_dispatch: Option<DispatchResult>,
    /// Backend tiers used across generations.
    pub(crate) generation_tiers: Vec<BackendTier>,
    /// Last generated code (for writing to disk and feeding into Testing).
    pub(crate) generated_code: Option<String>,
    /// Codebase context: pre-queried relevant code snippets from CodebaseMemory.
    pub(crate) code_context: Vec<String>,
    /// Persistent experience store for error patterns and successful generations.
    pub(crate) experience_store: Option<CodingExperienceStore>,
    /// Accumulated failure patterns during this run: (error_text, count).
    pub(crate) failure_patterns: Vec<(String, usize)>,
    /// Set when native_code_template() returns None — forces LLM tier on next generation.
    pub(crate) native_exhausted: bool,
    /// Prediction error history for trend detection.
    pub(crate) prediction_error_history: Vec<f32>,
    /// Confidence velocity history for trend detection.
    pub(crate) confidence_velocity_history: Vec<f32>,
    /// Optional event channel for streaming agent progress.
    pub(crate) event_sink: Option<std::sync::mpsc::Sender<AgentEvent>>,
    /// Current retry state for differentiated fixing strategies.
    pub(crate) retry_state: RetryState,
    /// Indexed codebase memory for semantic code search (populated by `index_project()`).
    #[cfg(feature = "code_generation")]
    pub(crate) code_memory: Option<crate::hdc::code_memory::CodebaseMemory>,
    /// Cached file sources for source-level context extraction (path → source text).
    #[cfg(feature = "code_generation")]
    pub(crate) source_cache: std::collections::HashMap<PathBuf, String>,
    /// Current execution plan profile (computed during Planning phase).
    pub(crate) current_plan: Option<PlanProfile>,
    /// Remaining energy budget for this run.
    pub(crate) energy_budget: f32,
    /// Tracks attempted fixes as `"{error_sig}:{fix_type}"` keys to skip re-applying same fix.
    pub(crate) attempted_fixes: std::collections::HashSet<String>,
    /// Direct test result flag, set by Testing phase (not inferred from observations).
    pub(crate) tests_passed: Option<bool>,
    /// Counter: fix attempts skipped by deduplication.
    pub(crate) dedup_skips: usize,
    /// Counter: quality gate rejections.
    pub(crate) quality_rejections: usize,
    /// Counter: consciousness gate deferrals.
    pub(crate) consciousness_deferrals: usize,
    /// Whether stuck detection triggered during this run.
    pub(crate) stuck_detected: bool,
}

impl CodingAgent {
    /// Create a new coding agent with default cognitive loop config.
    pub fn new(config: CodingAgentConfig) -> anyhow::Result<Self> {
        let cognitive_loop = CognitiveLoopService::new(CognitiveLoopConfig::default())?;
        Ok(Self::with_cognitive_loop(cognitive_loop, config))
    }

    /// Create a coding agent wrapping an existing cognitive loop.
    pub fn with_cognitive_loop(
        mut cognitive_loop: CognitiveLoopService,
        config: CodingAgentConfig,
    ) -> Self {
        // Install motor output bridge if not already present
        if !cognitive_loop.has_motor_bridge() {
            let bridge = if config.enable_real_exec {
                let sandbox = SandboxRoot::at(config.working_dir.clone()).unwrap_or_else(|_| {
                    SandboxRoot::new(&config.sandbox_session).expect("sandbox")
                });
                let policy = PolicyBundle::restrictive();
                let mut bridge = MotorOutputBridge::new(policy, sandbox);
                bridge.enable_real_execution();
                Ok(bridge.with_min_phi(0.05).with_min_confidence(0.05))
            } else {
                MotorOutputBridge::with_defaults()
                    .map(|b| b.with_min_phi(0.05).with_min_confidence(0.05))
            };
            if let Ok(bridge) = bridge {
                cognitive_loop.set_motor_output_bridge(bridge);
            }
        }

        // Select dispatcher
        let cloud_backend: Option<std::sync::Arc<dyn crate::language::llm_backend::LLMBackend>> =
            if config.use_cloud_llm {
                crate::language::anthropic_backend::AnthropicBackend::from_env().map(|b| {
                    std::sync::Arc::new(b)
                        as std::sync::Arc<dyn crate::language::llm_backend::LLMBackend>
                })
            } else {
                None
            };
        let dispatcher = if config.use_local_llm {
            use crate::language::llm_backend::OllamaBackend;
            IntelligentDispatcher::new(std::sync::Arc::new(OllamaBackend::new()), cloud_backend)
                .with_energy_budget(100.0)
        } else {
            IntelligentDispatcher::new(
                std::sync::Arc::new(crate::language::llm_backend::SimulatedBackend),
                cloud_backend,
            )
            .with_energy_budget(100.0)
        };

        let experience_store = Self::try_init_experience_store(&config.working_dir);

        Self {
            cognitive_loop,
            config,
            phase: TaskPhase::Understanding,
            iteration: 0,
            phase_failures: 0,
            files_modified: Vec::new(),
            phi_trace: Vec::new(),
            observations: Vec::new(),
            errors: Vec::new(),
            last_test_output: None,
            task: String::new(),
            dispatcher: Some(dispatcher),
            last_dispatch: None,
            generation_tiers: Vec::new(),
            generated_code: None,
            code_context: Vec::new(),
            experience_store,
            failure_patterns: Vec::new(),
            native_exhausted: false,
            prediction_error_history: Vec::new(),
            confidence_velocity_history: Vec::new(),
            event_sink: None,
            retry_state: RetryState::default(),
            #[cfg(feature = "code_generation")]
            code_memory: None,
            #[cfg(feature = "code_generation")]
            source_cache: std::collections::HashMap::new(),
            current_plan: None,
            energy_budget: 100.0,
            attempted_fixes: std::collections::HashSet::new(),
            tests_passed: None,
            dedup_skips: 0,
            quality_rejections: 0,
            consciousness_deferrals: 0,
            stuck_detected: false,
        }
    }

    /// Attempt to create an experience store.
    fn try_init_experience_store(working_dir: &std::path::Path) -> Option<CodingExperienceStore> {
        let rt = tokio::runtime::Runtime::new().ok()?;

        let db_dir = working_dir.join(".symthaea");
        if std::fs::create_dir_all(&db_dir).is_ok() {
            let db_path = db_dir.join("experience.db");
            if let Ok(store) = rt.block_on(CodingExperienceStore::persistent(
                &db_path.to_string_lossy(),
            )) {
                tracing::info!(
                    target: "symthaea::coding_agent",
                    path = %db_path.display(),
                    "Persistent experience store initialized"
                );
                return Some(store);
            }
        }

        tracing::debug!(
            target: "symthaea::coding_agent",
            "Falling back to in-memory experience store"
        );
        rt.block_on(async { CodingExperienceStore::new().await.ok() })
    }

    /// Run the agent on a coding task.
    pub fn run(&mut self, task: &str) -> AgentResult {
        self.task = task.to_string();
        self.phase = TaskPhase::Understanding;
        self.iteration = 0;
        self.phase_failures = 0;
        self.files_modified.clear();
        self.phi_trace.clear();
        self.observations.clear();
        self.errors.clear();
        self.last_test_output = None;
        self.generated_code = None;
        self.generation_tiers.clear();
        self.failure_patterns.clear();
        self.native_exhausted = false;
        self.prediction_error_history.clear();
        self.confidence_velocity_history.clear();
        self.retry_state = RetryState::default();
        self.current_plan = None;
        self.energy_budget = 100.0;
        self.attempted_fixes.clear();
        self.tests_passed = None;
        self.dedup_skips = 0;
        self.quality_rejections = 0;
        self.consciousness_deferrals = 0;
        self.stuck_detected = false;

        #[cfg(feature = "code_generation")]
        if self.code_memory.is_none() && self.config.working_dir.exists() {
            match self.index_project(&self.config.working_dir.clone()) {
                Ok((files, funcs, types)) => {
                    if files > 0 {
                        tracing::info!(
                            target: "symthaea::coding_agent",
                            files, funcs, types,
                            "Auto-indexed project for codebase awareness"
                        );
                    }
                }
                Err(e) => {
                    tracing::debug!(
                        target: "symthaea::coding_agent",
                        error = %e,
                        "Auto-index skipped (non-fatal)"
                    );
                }
            }
        }

        #[cfg(feature = "code_generation")]
        if let Some(ref memory) = self.code_memory {
            let context = Self::build_source_context(memory, &self.source_cache, task);
            if !context.is_empty() {
                self.code_context = context;
            }
        }

        tracing::info!(
            target: "symthaea::coding_agent",
            task = task,
            max_iterations = self.config.max_iterations,
            context_entries = self.code_context.len(),
            "Starting coding agent"
        );

        self.warm_up_phi(3);

        while self.phase != TaskPhase::Done && self.iteration < self.config.max_iterations {
            if self.energy_budget <= 0.0 {
                tracing::warn!(
                    target: "symthaea::coding_agent",
                    iteration = self.iteration,
                    phase = %self.phase,
                    "Energy budget exhausted — terminating early"
                );
                self.observations
                    .push("Energy budget exhausted — terminating".into());
                self.phase = TaskPhase::Done;
                break;
            }

            self.step();
            self.iteration += 1;
        }

        if self.phase != TaskPhase::Done {
            tracing::warn!(
                target: "symthaea::coding_agent",
                iterations = self.iteration,
                phase = %self.phase,
                "Max iterations reached"
            );
        }

        self.flush_experience_store();

        let result = self.build_result();
        self.emit_event(AgentEvent::Done(result.clone()));
        result
    }

    /// Flush pending writes in the experience store to the database.
    pub(crate) fn flush_experience_store(&mut self) {
        if let Some(ref mut store) = self.experience_store {
            match tokio::runtime::Handle::try_current() {
                Ok(handle) => {
                    tokio::task::block_in_place(|| handle.block_on(store.flush()));
                }
                Err(_) => {
                    if let Ok(rt) = tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                    {
                        rt.block_on(store.flush());
                    }
                }
            }
        }
    }

    /// Warm up the CfC by running idle cognitive cycles.
    fn warm_up_phi(&mut self, cycles: usize) {
        let observation = format!("WARM_UP: preparing to work on: {}", self.task);
        for i in 0..cycles {
            let result = self.cognitive_loop.cycle(&observation);
            let phi = result.metadata.consciousness.consciousness_level;
            tracing::debug!(
                target: "symthaea::coding_agent",
                cycle = i,
                phi = phi,
                "Warm-up cycle"
            );
        }
    }

    /// Execute one step of the agent loop.
    fn step(&mut self) {
        if self.native_exhausted
            && self.generated_code.is_none()
            && self.iteration >= 5
            && !self.config.use_local_llm
        {
            tracing::info!(
                target: "symthaea::coding_agent",
                task = %self.task,
                iteration = self.iteration,
                "Fast-fail: native exhausted, no code produced after {} iterations",
                self.iteration
            );
            self.observations
                .push("Fast-fail: task beyond native capability, no usable code generated".into());
            self.phase = TaskPhase::Done;
            return;
        }

        if let Some((plan, profile)) = self.select_plan_fep() {
            let current_phi = self.phi_trace.last().copied().unwrap_or(0.0);
            let (approved, reason) = self.evaluate_plan(&plan, current_phi);

            if !approved {
                tracing::warn!(
                    target: "symthaea::coding_agent",
                    phase = %self.phase,
                    reason = %reason,
                    "Plan rejected — skipping step"
                );
                self.observations.push(format!("Plan rejected: {reason}"));
            } else {
                tracing::debug!(
                    target: "symthaea::coding_agent",
                    phase = %self.phase,
                    steps = profile.step_count,
                    energy = profile.total_energy,
                    "FEP plan approved"
                );
                self.current_plan = Some(profile.clone());
                self.deduct_energy(&profile);
            }
        }

        self.pre_cycle_action();

        let observation = self.build_observation();
        let motor_request = self.build_motor_request();
        self.cognitive_loop.set_motor_request(motor_request);

        let cycle_result = self.cognitive_loop.cycle(&observation);

        let signals = self.extract_consciousness_signals(&cycle_result);
        let phi = signals.phi;
        self.phi_trace.push(phi);
        self.prediction_error_history.push(signals.prediction_error);
        self.confidence_velocity_history
            .push(signals.confidence_velocity);
        if self.prediction_error_history.len() > 10 {
            self.prediction_error_history.remove(0);
        }
        if self.confidence_velocity_history.len() > 10 {
            self.confidence_velocity_history.remove(0);
        }
        self.emit_event(AgentEvent::ConsciousnessSnapshot {
            phi,
            prediction_error: signals.prediction_error,
            confidence_velocity: signals.confidence_velocity,
        });

        let motor_result = self.cognitive_loop.take_motor_result();

        let phase_before = self.phase.clone();
        self.process_step_result(&cycle_result, motor_result, phi);
        if self.phase != phase_before {
            self.emit_phase_transition(&phase_before, &self.phase.clone());
        }
    }

    /// Phase-specific actions performed before the cognitive cycle.
    fn pre_cycle_action(&mut self) {
        match self.phase {
            TaskPhase::Understanding => {
                self.do_understanding_molecule();
            }
            TaskPhase::Fixing => {
                if !self.try_structured_auto_fix() {
                    self.do_generation();
                }
            }
            TaskPhase::Generating => {
                self.do_generation();
            }
            _ => {}
        }
    }

    // ── Observation & Motor Request Building ────────────────────────────

    /// Build the observation text that the cognitive loop will process.
    fn build_observation(&self) -> String {
        let mut obs = format!("CODING TASK: {}\n", self.task);
        obs.push_str(&format!("PHASE: {}\n", self.phase));
        obs.push_str(&format!(
            "ITERATION: {}/{}\n",
            self.iteration, self.config.max_iterations
        ));

        if !self.observations.is_empty() {
            obs.push_str("CONTEXT:\n");
            for o in self.observations.iter().rev().take(3).rev() {
                obs.push_str(&format!("  {}\n", o));
            }
        }

        if let Some(ref test_output) = self.last_test_output {
            obs.push_str(&format!("LAST TEST OUTPUT:\n{}\n", test_output));
        }

        if !self.errors.is_empty() {
            obs.push_str("ERRORS:\n");
            for e in self.errors.iter().rev().take(2) {
                obs.push_str(&format!("  {}\n", e));
            }
        }

        obs
    }

    // ── Plan Evaluation (Typed Primitives) ──────────────────────────────

    fn build_execution_plan(&self) -> Option<Molecule> {
        let target = self.resolve_target_file();
        let working_dir = self.config.working_dir.clone();

        match self.phase {
            TaskPhase::Understanding => {
                let mut plan = Molecule::atom(Atom::list(working_dir.clone()));
                if target.exists() {
                    plan = plan.then(Molecule::atom(Atom::read(target)));
                }
                Some(plan)
            }
            TaskPhase::Planning => None,
            TaskPhase::Generating => {
                if let Some(ref code) = self.generated_code {
                    Some(crate::action::primitives::recipes::write_and_check(
                        target, code,
                    ))
                } else {
                    None
                }
            }
            TaskPhase::Testing => {
                Some(Molecule::atom(Atom::cargo_check(working_dir)))
            }
            TaskPhase::Fixing => {
                if let Some(ref code) = self.generated_code {
                    let write_check =
                        crate::action::primitives::recipes::write_and_check(target, code);
                    Some(write_check.recover(|_| {
                        Molecule::atom(Atom::Noop)
                    }))
                } else {
                    None
                }
            }
            TaskPhase::Done => None,
        }
    }

    fn select_plan_fep(&self) -> Option<(Molecule, PlanProfile)> {
        use crate::action::primitives::{
            select_best_plan, select_best_plan_with_history, PlanCandidate,
        };

        let target = self.resolve_target_file();
        let working_dir = self.config.working_dir.clone();
        let current_phi = self.phi_trace.last().copied().unwrap_or(0.0);

        let candidates: Vec<PlanCandidate> = match self.phase {
            TaskPhase::Understanding => {
                let mut plans = vec![];
                let list_only = Molecule::atom(Atom::list(working_dir.clone()));
                plans.push(PlanCandidate {
                    name: "list_only".into(),
                    profile: list_only.profile(),
                    molecule: list_only,
                });
                if target.exists() {
                    let list_and_read = Molecule::atom(Atom::list(working_dir.clone()))
                        .then(Molecule::atom(Atom::read(target.clone())));
                    plans.push(PlanCandidate {
                        name: "list_and_read".into(),
                        profile: list_and_read.profile(),
                        molecule: list_and_read,
                    });
                }
                let cargo_toml = working_dir.join("Cargo.toml");
                if cargo_toml.exists() && target.exists() {
                    let gather = Molecule::atom(Atom::read(cargo_toml))
                        .then(Molecule::atom(Atom::read(target.clone())))
                        .then(Molecule::atom(Atom::list(working_dir.clone())));
                    plans.push(PlanCandidate {
                        name: "full_context".into(),
                        profile: gather.profile(),
                        molecule: gather,
                    });
                }
                plans
            }
            TaskPhase::Testing => {
                let mut plans = vec![];
                let check = Molecule::atom(Atom::cargo_check(working_dir.clone()));
                plans.push(PlanCandidate {
                    name: "cargo_check".into(),
                    profile: check.profile(),
                    molecule: check,
                });
                let check_clippy = Molecule::atom(Atom::cargo_check(working_dir.clone()))
                    .then(Molecule::atom(Atom::cargo_clippy(working_dir.clone())));
                plans.push(PlanCandidate {
                    name: "check_and_clippy".into(),
                    profile: check_clippy.profile(),
                    molecule: check_clippy,
                });
                let test = Molecule::atom(Atom::cargo_test(working_dir.clone()));
                plans.push(PlanCandidate {
                    name: "cargo_test".into(),
                    profile: test.profile(),
                    molecule: test,
                });
                plans
            }
            TaskPhase::Generating => {
                let mut plans = vec![];
                if let Some(ref code) = self.generated_code {
                    let wc =
                        crate::action::primitives::recipes::write_and_check(target.clone(), code);
                    plans.push(PlanCandidate {
                        name: "write_and_check".into(),
                        profile: wc.profile(),
                        molecule: wc,
                    });
                    let wct = crate::action::primitives::recipes::full_coding_workflow(
                        target.clone(),
                        code.clone(),
                        working_dir.clone(),
                    );
                    plans.push(PlanCandidate {
                        name: "full_workflow".into(),
                        profile: wct.profile(),
                        molecule: wct,
                    });
                }
                let prompt = self.build_generation_prompt();
                for (name, mol) in crate::action::primitives::recipes::tiered_generation_candidates(
                    target.clone(),
                    &prompt,
                ) {
                    plans.push(PlanCandidate {
                        name,
                        profile: mol.profile(),
                        molecule: mol,
                    });
                }
                plans
            }
            TaskPhase::Fixing => {
                if let Some(ref code) = self.generated_code {
                    let mut plans = vec![];
                    let wc =
                        crate::action::primitives::recipes::write_and_check(target.clone(), code);
                    plans.push(PlanCandidate {
                        name: "fix_and_check".into(),
                        profile: wc.profile(),
                        molecule: wc,
                    });
                    let wcr =
                        crate::action::primitives::recipes::write_and_check(target.clone(), code)
                            .recover(|_| Molecule::atom(Atom::Noop));
                    plans.push(PlanCandidate {
                        name: "fix_with_recovery".into(),
                        profile: wcr.profile(),
                        molecule: wcr,
                    });
                    plans
                } else {
                    vec![]
                }
            }
            TaskPhase::Planning | TaskPhase::Done => vec![],
        };

        if candidates.is_empty() {
            return None;
        }

        let selected_idx = if let Some(ref store) = self.experience_store {
            let recipe_keys: Vec<&str> = candidates
                .iter()
                .map(|c| {
                    c.profile.atom_names.first().copied().unwrap_or("Unknown")
                })
                .collect();
            let rates = store.recipe_success_rates(&recipe_keys);
            select_best_plan_with_history(&candidates, current_phi, self.energy_budget, &rates)
        } else {
            select_best_plan(&candidates, current_phi, self.energy_budget)
        };

        let selected_idx = selected_idx?;
        let selected = &candidates[selected_idx];

        tracing::debug!(
            target: "symthaea::coding_agent",
            phase = %self.phase,
            selected = %selected.name,
            candidates = candidates.len(),
            energy = selected.profile.total_energy,
            "FEP selected plan (history-aware)"
        );

        let profile = selected.profile.clone();
        self.build_execution_plan().map(|m| (m, profile))
    }

    fn evaluate_plan(&self, plan: &Molecule, current_phi: f32) -> (bool, String) {
        let profile = plan.profile();

        if !profile.phi_sufficient(current_phi) {
            return (
                false,
                format!(
                    "Phi too low: {:.3} < {:.3} required",
                    current_phi, profile.min_phi
                ),
            );
        }

        if !profile.within_budget(self.energy_budget) {
            return (
                false,
                format!(
                    "Energy budget exceeded: plan costs {:.1}, budget remaining {:.1}",
                    profile.total_energy, self.energy_budget
                ),
            );
        }

        if profile.max_destructiveness == crate::action::DestructivenessLevel::Destructive {
            return (
                false,
                format!(
                    "Plan contains destructive action ({}) — requires confirmation",
                    profile.atom_names.join(" → ")
                ),
            );
        }

        (
            true,
            format!(
                "Plan approved: {} steps, energy {:.1}/{:.1}, phi {:.3}/{:.3}",
                profile.step_count,
                profile.total_energy,
                self.energy_budget,
                current_phi,
                profile.min_phi,
            ),
        )
    }

    fn deduct_energy(&mut self, profile: &PlanProfile) {
        self.energy_budget = (self.energy_budget - profile.total_energy).max(0.0);
    }

    fn build_motor_request(&self) -> MotorActionRequest {
        match self.phase {
            TaskPhase::Understanding => {
                MotorActionRequest {
                    target_path: Some(self.config.working_dir.clone()),
                    ..Default::default()
                }
            }
            TaskPhase::Planning => {
                MotorActionRequest::default()
            }
            TaskPhase::Generating => {
                if self.generated_code.is_some() {
                    MotorActionRequest {
                        target_path: Some(self.config.working_dir.clone()),
                        program: Some("cargo".into()),
                        args: vec!["check".into()],
                        ..Default::default()
                    }
                } else {
                    MotorActionRequest {
                        target_path: Some(self.config.working_dir.clone()),
                        ..Default::default()
                    }
                }
            }
            TaskPhase::Testing => {
                MotorActionRequest {
                    target_path: Some(self.config.working_dir.clone()),
                    program: Some("cargo".into()),
                    args: vec!["check".into()],
                    ..Default::default()
                }
            }
            TaskPhase::Fixing => {
                if self.generated_code.is_some() {
                    MotorActionRequest {
                        target_path: Some(self.config.working_dir.clone()),
                        program: Some("cargo".into()),
                        args: vec!["check".into()],
                        ..Default::default()
                    }
                } else {
                    MotorActionRequest {
                        target_path: Some(self.config.working_dir.clone()),
                        ..Default::default()
                    }
                }
            }
            TaskPhase::Done => MotorActionRequest::default(),
        }
    }

    // ── Public API ──────────────────────────────────────────────────────

    /// Get the current phase.
    pub fn phase(&self) -> &TaskPhase {
        &self.phase
    }

    /// Get the current iteration count.
    pub fn iteration(&self) -> usize {
        self.iteration
    }

    /// Access the underlying cognitive loop for direct inspection.
    pub fn cognitive_loop(&self) -> &CognitiveLoopService {
        &self.cognitive_loop
    }

    /// Access the underlying cognitive loop mutably.
    pub fn cognitive_loop_mut(&mut self) -> &mut CognitiveLoopService {
        &mut self.cognitive_loop
    }

    /// Set a custom intelligent dispatcher for LLM-routed code generation.
    pub fn with_dispatcher(mut self, dispatcher: IntelligentDispatcher) -> Self {
        self.dispatcher = Some(dispatcher);
        self
    }

    /// Set codebase context from an external CodebaseMemory query.
    pub fn set_code_context(&mut self, context: Vec<String>) {
        self.code_context = context;
    }

    /// Attach an event channel for streaming agent progress.
    pub fn with_event_channel(mut self) -> (Self, std::sync::mpsc::Receiver<AgentEvent>) {
        let (tx, rx) = std::sync::mpsc::channel();
        self.event_sink = Some(tx);
        (self, rx)
    }

    /// Attach an event sink after construction.
    pub fn subscribe_events(&mut self, tx: std::sync::mpsc::Sender<AgentEvent>) {
        self.event_sink = Some(tx);
    }

    /// Get the last dispatch result.
    pub fn last_dispatch(&self) -> Option<&DispatchResult> {
        self.last_dispatch.as_ref()
    }

    /// Get the dispatcher's total energy consumption.
    pub fn total_energy(&self) -> f64 {
        self.dispatcher.as_ref().map_or(0.0, |d| d.total_energy())
    }

    /// Get accumulated failure patterns from this run.
    pub fn failure_patterns(&self) -> &[(String, usize)] {
        &self.failure_patterns
    }

    /// Whether the agent has a persistent experience store.
    pub fn has_experience_store(&self) -> bool {
        self.experience_store.is_some()
    }

    /// Get the count of stored experiences.
    pub fn experience_count(&self) -> usize {
        if let Some(ref store) = self.experience_store {
            if let Ok(rt) = tokio::runtime::Runtime::new() {
                return rt.block_on(store.count());
            }
        }
        0
    }

    /// Get cached success patterns from the experience store.
    pub fn cached_successes(&self) -> Vec<(String, String)> {
        self.experience_store
            .as_ref()
            .map(|s| s.cached_successes().to_vec())
            .unwrap_or_default()
    }

    /// Get cached error hints from the experience store.
    pub fn cached_error_hints(&self) -> Vec<(String, String)> {
        self.experience_store
            .as_ref()
            .map(|s| s.cached_error_hints().to_vec())
            .unwrap_or_default()
    }

    /// Get the current execution plan profile (if any).
    pub fn current_plan_profile(&self) -> Option<&PlanProfile> {
        self.current_plan.as_ref()
    }

    /// Get remaining energy budget.
    pub fn remaining_energy(&self) -> f32 {
        self.energy_budget
    }

    /// Build a plan for a hypothetical action and evaluate it.
    pub fn evaluate_hypothetical_plan(&self, plan: &Molecule) -> (bool, String, PlanProfile) {
        let current_phi = self.phi_trace.last().copied().unwrap_or(0.0);
        let profile = plan.profile();
        let (approved, reason) = self.evaluate_plan(plan, current_phi);
        (approved, reason, profile)
    }
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
