//! Intelligent Dispatcher: consciousness-routed code generation backend selection.
//!
//! Routes coding tasks to the optimal backend based on consciousness state:
//!
//! | Epistemic Status | Prediction Error | Backend |
//! |-----------------|-----------------|---------|
//! | Certain | Low | Native (CodeGenerator via HDC+CfC, 0 latency) |
//! | Probable | Medium | Local LLM (qwen2.5-coder:7b via Ollama) |
//! | Uncertain/Unknown | High | Cloud LLM (Claude API) |
//!
//! Tracks per-backend success rates and respects metabolic budget constraints.
//! Native code generation costs 1.0 energy, local LLM costs 10.0, cloud costs 50.0.

use crate::cognitive_loop::routing::CodeTaskType;
use crate::language::llm_backend::{GenerationParams, LLMBackend, SimulatedBackend};
use crate::mind::structured_thought::EpistemicStatus;
use std::collections::HashMap;
use std::sync::Arc;

/// Metabolic energy cost per backend tier.
const COST_NATIVE: f64 = 1.0;
const COST_LOCAL_LLM: f64 = 10.0;
const COST_CLOUD_LLM: f64 = 50.0;

/// Which backend tier was selected for a generation request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackendTier {
    /// Native HDC+CfC code generation (zero external calls).
    Native,
    /// Local LLM (e.g., qwen2.5-coder:7b via Ollama).
    LocalLlm,
    /// Cloud LLM (e.g., Claude via Anthropic API).
    CloudLlm,
    /// Simulated fallback (no real generation).
    Simulated,
}

impl std::fmt::Display for BackendTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Native => write!(f, "Native"),
            Self::LocalLlm => write!(f, "LocalLLM"),
            Self::CloudLlm => write!(f, "CloudLLM"),
            Self::Simulated => write!(f, "Simulated"),
        }
    }
}

/// Result of an intelligent dispatch decision.
#[derive(Debug, Clone)]
pub struct DispatchResult {
    /// The generated code (or error message).
    pub output: String,
    /// Which backend tier was used.
    pub tier: BackendTier,
    /// Energy cost of this generation.
    pub energy_cost: f64,
    /// Whether the generation succeeded.
    pub success: bool,
}

/// Per-backend success tracking for adaptive routing.
#[derive(Debug, Clone, Default)]
struct BackendStats {
    attempts: u32,
    successes: u32,
}

impl BackendStats {
    fn success_rate(&self) -> f64 {
        if self.attempts == 0 {
            0.5 // prior: assume 50% for untried backends
        } else {
            self.successes as f64 / self.attempts as f64
        }
    }

    fn record(&mut self, success: bool) {
        self.attempts += 1;
        if success {
            self.successes += 1;
        }
    }
}

/// Consciousness-routed code generation dispatcher.
///
/// Selects the optimal backend tier based on epistemic status, prediction error,
/// and metabolic budget. Falls back gracefully: cloud → local → native → simulated.
///
/// ## Adaptive Routing
///
/// Tracks per-backend AND per-(task_type, backend) success rates. When a backend's
/// success rate for a specific task type drops below `MIN_VIABLE_RATE`, the dispatcher
/// downgrades to the next cheaper tier — preventing repeated expensive failures.
pub struct IntelligentDispatcher {
    /// Local LLM backend (Ollama with qwen2.5-coder:7b).
    local_llm: Arc<dyn LLMBackend>,
    /// Cloud LLM backend (Anthropic Claude, if available).
    cloud_llm: Option<Arc<dyn LLMBackend>>,
    /// Per-tier success tracking (global).
    native_stats: BackendStats,
    local_stats: BackendStats,
    cloud_stats: BackendStats,
    /// Per-(task_type, tier) success tracking for adaptive routing.
    task_tier_stats: HashMap<(CodeTaskType, BackendTier), BackendStats>,
    /// Cumulative energy spent.
    total_energy: f64,
    /// Maximum energy budget (0.0 = unlimited).
    energy_budget: f64,
}

/// Minimum viable success rate before a tier is downgraded for a task type.
const MIN_VIABLE_RATE: f64 = 0.25;

impl IntelligentDispatcher {
    /// Create a dispatcher with the given LLM backends.
    pub fn new(local_llm: Arc<dyn LLMBackend>, cloud_llm: Option<Arc<dyn LLMBackend>>) -> Self {
        Self {
            local_llm,
            cloud_llm,
            native_stats: BackendStats::default(),
            local_stats: BackendStats::default(),
            cloud_stats: BackendStats::default(),
            task_tier_stats: HashMap::new(),
            total_energy: 0.0,
            energy_budget: 0.0,
        }
    }

    /// Create a dispatcher with simulated backends (for testing).
    pub fn simulated() -> Self {
        Self::new(Arc::new(SimulatedBackend), None)
    }

    /// Set the energy budget. 0.0 means unlimited.
    pub fn with_energy_budget(mut self, budget: f64) -> Self {
        self.energy_budget = budget;
        self
    }

    /// Select the backend tier based on consciousness state.
    ///
    /// The selection logic:
    /// 1. **Certain** + low prediction error → Native (fast, free)
    /// 2. **Probable** + medium error → Local LLM (moderate cost)
    /// 3. **Uncertain/Unknown** → Cloud LLM (expensive but capable)
    /// 4. Adaptive: downgrade if success rate for (task_type, tier) < 25%
    /// 5. Budget exceeded → fall back to cheaper tier
    pub fn select_tier(
        &self,
        epistemic: EpistemicStatus,
        prediction_error: f64,
        consciousness_level: f64,
    ) -> BackendTier {
        self.select_tier_for_task(epistemic, prediction_error, consciousness_level, None)
    }

    /// Select the backend tier with task-type-aware adaptive routing.
    ///
    /// When `task_type` is provided, the dispatcher checks per-(task_type, tier)
    /// success rates and downgrades tiers that have been failing. This prevents
    /// expensive repeated failures on task types a backend can't handle.
    pub fn select_tier_for_task(
        &self,
        epistemic: EpistemicStatus,
        prediction_error: f64,
        consciousness_level: f64,
        task_type: Option<CodeTaskType>,
    ) -> BackendTier {
        // Budget check: if we'd exceed budget, force cheaper tier
        let remaining = if self.energy_budget > 0.0 {
            self.energy_budget - self.total_energy
        } else {
            f64::MAX
        };

        // Base tier from epistemic status + prediction error
        let base_tier = match epistemic {
            EpistemicStatus::Certain if prediction_error < 0.3 => BackendTier::Native,
            EpistemicStatus::Certain | EpistemicStatus::Probable if prediction_error < 0.6 => {
                BackendTier::LocalLlm
            }
            _ => {
                // Uncertain/Unknown or high prediction error
                if self.cloud_llm.is_some() {
                    BackendTier::CloudLlm
                } else {
                    BackendTier::LocalLlm
                }
            }
        };

        // Consciousness modulation: very low consciousness → don't trust expensive backends
        if consciousness_level < 0.2 {
            return BackendTier::Native;
        }

        // Adaptive routing: downgrade if this (task_type, tier) has poor success rate
        let tier_after_adaptive = if let Some(tt) = task_type {
            self.adaptive_downgrade(base_tier, tt)
        } else {
            // No task type — use global success rates
            self.global_adaptive_downgrade(base_tier)
        };

        // Budget constraints: fall back to cheaper tiers
        match tier_after_adaptive {
            BackendTier::CloudLlm if remaining < COST_CLOUD_LLM => {
                if remaining >= COST_LOCAL_LLM {
                    BackendTier::LocalLlm
                } else {
                    BackendTier::Native
                }
            }
            BackendTier::LocalLlm if remaining < COST_LOCAL_LLM => BackendTier::Native,
            other => other,
        }
    }

    /// Downgrade a tier if its task-type-specific success rate is below MIN_VIABLE_RATE.
    ///
    /// Only triggers after enough data: requires >= 3 attempts for a (task_type, tier) pair.
    fn adaptive_downgrade(&self, tier: BackendTier, task_type: CodeTaskType) -> BackendTier {
        let key = (task_type, tier);
        if let Some(stats) = self.task_tier_stats.get(&key) {
            if stats.attempts >= 3 && stats.success_rate() < MIN_VIABLE_RATE {
                tracing::info!(
                    target: "symthaea::dispatcher",
                    tier = %tier,
                    task_type = ?task_type,
                    rate = stats.success_rate(),
                    "Adaptive downgrade: {tier} failing on {task_type:?}"
                );
                return match tier {
                    BackendTier::CloudLlm => BackendTier::LocalLlm,
                    BackendTier::LocalLlm => BackendTier::Native,
                    other => other,
                };
            }
        }
        tier
    }

    /// Downgrade based on global success rates (no task type context).
    fn global_adaptive_downgrade(&self, tier: BackendTier) -> BackendTier {
        let rate = match tier {
            BackendTier::CloudLlm => self.cloud_stats.success_rate(),
            BackendTier::LocalLlm => self.local_stats.success_rate(),
            _ => return tier,
        };

        // Only downgrade if we have enough data AND rate is poor
        let attempts = match tier {
            BackendTier::CloudLlm => self.cloud_stats.attempts,
            BackendTier::LocalLlm => self.local_stats.attempts,
            _ => 0,
        };

        if attempts >= 3 && rate < MIN_VIABLE_RATE {
            match tier {
                BackendTier::CloudLlm => BackendTier::LocalLlm,
                BackendTier::LocalLlm => BackendTier::Native,
                other => other,
            }
        } else {
            tier
        }
    }

    /// Generate code using the consciousness-selected backend.
    ///
    /// For Native tier, returns a placeholder — the caller (CodingAgent) should
    /// use `CodeGenerator` directly. For LLM tiers, calls the async backend.
    pub async fn generate(
        &mut self,
        prompt: &str,
        params: &GenerationParams,
        epistemic: EpistemicStatus,
        prediction_error: f64,
        consciousness_level: f64,
    ) -> DispatchResult {
        self.generate_for_task(prompt, params, epistemic, prediction_error, consciousness_level, None)
            .await
    }

    /// Generate code with task-type-aware adaptive routing.
    pub async fn generate_for_task(
        &mut self,
        prompt: &str,
        params: &GenerationParams,
        epistemic: EpistemicStatus,
        prediction_error: f64,
        consciousness_level: f64,
        task_type: Option<CodeTaskType>,
    ) -> DispatchResult {
        let tier = self.select_tier_for_task(epistemic, prediction_error, consciousness_level, task_type);

        let (output, success, cost) = match tier {
            BackendTier::Native => {
                // Native generation is handled externally by CodeGenerator.
                // Return a signal that the caller should use native path.
                self.native_stats.record(true);
                ("[NATIVE: use CodeGenerator]".to_string(), true, COST_NATIVE)
            }
            BackendTier::LocalLlm => match self.local_llm.generate(prompt, params).await {
                Ok(output) => {
                    let success = !output.is_empty();
                    self.local_stats.record(success);
                    (output, success, COST_LOCAL_LLM)
                }
                Err(e) => {
                    self.local_stats.record(false);
                    (format!("Local LLM error: {e}"), false, COST_LOCAL_LLM)
                }
            },
            BackendTier::CloudLlm => {
                if let Some(ref cloud) = self.cloud_llm {
                    match cloud.generate(prompt, params).await {
                        Ok(output) => {
                            let success = !output.is_empty();
                            self.cloud_stats.record(success);
                            (output, success, COST_CLOUD_LLM)
                        }
                        Err(e) => {
                            self.cloud_stats.record(false);
                            // Fall back to local LLM on cloud failure
                            tracing::warn!(
                                target: "symthaea::dispatcher",
                                error = %e,
                                "Cloud LLM failed, falling back to local"
                            );
                            match self.local_llm.generate(prompt, params).await {
                                Ok(output) => {
                                    self.local_stats.record(true);
                                    (output, true, COST_LOCAL_LLM)
                                }
                                Err(e2) => {
                                    self.local_stats.record(false);
                                    (
                                        format!("All LLMs failed: cloud={e}, local={e2}"),
                                        false,
                                        COST_LOCAL_LLM,
                                    )
                                }
                            }
                        }
                    }
                } else {
                    // No cloud backend configured — use local
                    match self.local_llm.generate(prompt, params).await {
                        Ok(output) => {
                            self.local_stats.record(true);
                            (output, true, COST_LOCAL_LLM)
                        }
                        Err(e) => {
                            self.local_stats.record(false);
                            (format!("Local LLM error: {e}"), false, COST_LOCAL_LLM)
                        }
                    }
                }
            }
            BackendTier::Simulated => ("// simulated code output".to_string(), true, 0.0),
        };

        self.total_energy += cost;

        tracing::info!(
            target: "symthaea::dispatcher",
            tier = %tier,
            cost = cost,
            total_energy = self.total_energy,
            success = success,
            "Code generation dispatched"
        );

        DispatchResult {
            output,
            tier,
            energy_cost: cost,
            success,
        }
    }

    /// Get success rate for a given tier.
    pub fn success_rate(&self, tier: BackendTier) -> f64 {
        match tier {
            BackendTier::Native => self.native_stats.success_rate(),
            BackendTier::LocalLlm => self.local_stats.success_rate(),
            BackendTier::CloudLlm => self.cloud_stats.success_rate(),
            BackendTier::Simulated => 1.0,
        }
    }

    /// Get total energy consumed.
    pub fn total_energy(&self) -> f64 {
        self.total_energy
    }

    /// Record the outcome of a generation that was validated externally.
    ///
    /// This feeds the Bayesian routing: after the agent tests the generated code,
    /// it calls this with the tier that was used and whether the code passed checks.
    /// Over time, this shifts routing toward backends with higher success rates.
    pub fn record_outcome(&mut self, tier: BackendTier, success: bool) {
        self.record_outcome_for_task(tier, success, None);
    }

    /// Record outcome with task-type context for adaptive routing.
    ///
    /// When `task_type` is provided, updates both global and per-(task_type, tier)
    /// stats. This enables the dispatcher to learn that e.g. Native works well for
    /// Create tasks but poorly for Debug tasks.
    pub fn record_outcome_for_task(
        &mut self,
        tier: BackendTier,
        success: bool,
        task_type: Option<CodeTaskType>,
    ) {
        // Global stats
        match tier {
            BackendTier::Native => self.native_stats.record(success),
            BackendTier::LocalLlm => self.local_stats.record(success),
            BackendTier::CloudLlm => self.cloud_stats.record(success),
            BackendTier::Simulated => {}
        }

        // Per-task-type stats
        if let Some(tt) = task_type {
            self.task_tier_stats
                .entry((tt, tier))
                .or_default()
                .record(success);
        }

        tracing::debug!(
            target: "symthaea::dispatcher",
            tier = %tier,
            success = success,
            task_type = ?task_type,
            rate = self.success_rate(tier),
            "Recorded generation outcome"
        );
    }

    /// Get the success rate for a (task_type, tier) combination.
    ///
    /// Returns the task-specific rate if enough data exists (>= 3 attempts),
    /// otherwise falls back to the global tier rate.
    pub fn task_success_rate(&self, task_type: CodeTaskType, tier: BackendTier) -> f64 {
        if let Some(stats) = self.task_tier_stats.get(&(task_type, tier)) {
            if stats.attempts >= 3 {
                return stats.success_rate();
            }
        }
        self.success_rate(tier)
    }

    /// Get remaining energy budget (f64::MAX if unlimited).
    pub fn remaining_energy(&self) -> f64 {
        if self.energy_budget > 0.0 {
            (self.energy_budget - self.total_energy).max(0.0)
        } else {
            f64::MAX
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tier_display() {
        assert_eq!(format!("{}", BackendTier::Native), "Native");
        assert_eq!(format!("{}", BackendTier::LocalLlm), "LocalLLM");
        assert_eq!(format!("{}", BackendTier::CloudLlm), "CloudLLM");
    }

    #[test]
    fn test_select_tier_certain_low_error() {
        let dispatcher = IntelligentDispatcher::simulated();
        let tier = dispatcher.select_tier(EpistemicStatus::Certain, 0.1, 0.8);
        assert_eq!(tier, BackendTier::Native);
    }

    #[test]
    fn test_select_tier_probable_medium_error() {
        let dispatcher = IntelligentDispatcher::simulated();
        let tier = dispatcher.select_tier(EpistemicStatus::Probable, 0.4, 0.8);
        assert_eq!(tier, BackendTier::LocalLlm);
    }

    #[test]
    fn test_select_tier_uncertain_uses_local_when_no_cloud() {
        let dispatcher = IntelligentDispatcher::simulated();
        // No cloud backend → falls back to LocalLlm
        let tier = dispatcher.select_tier(EpistemicStatus::Uncertain, 0.8, 0.8);
        assert_eq!(tier, BackendTier::LocalLlm);
    }

    #[test]
    fn test_select_tier_uncertain_uses_cloud_when_available() {
        let dispatcher = IntelligentDispatcher::new(
            Arc::new(SimulatedBackend),
            Some(Arc::new(SimulatedBackend)),
        );
        let tier = dispatcher.select_tier(EpistemicStatus::Uncertain, 0.8, 0.8);
        assert_eq!(tier, BackendTier::CloudLlm);
    }

    #[test]
    fn test_low_consciousness_forces_native() {
        let dispatcher = IntelligentDispatcher::new(
            Arc::new(SimulatedBackend),
            Some(Arc::new(SimulatedBackend)),
        );
        // Even with Uncertain status, low consciousness → Native
        let tier = dispatcher.select_tier(EpistemicStatus::Uncertain, 0.9, 0.1);
        assert_eq!(tier, BackendTier::Native);
    }

    #[test]
    fn test_budget_forces_downgrade() {
        let dispatcher = IntelligentDispatcher::new(
            Arc::new(SimulatedBackend),
            Some(Arc::new(SimulatedBackend)),
        )
        .with_energy_budget(15.0); // enough for local but not cloud

        let tier = dispatcher.select_tier(EpistemicStatus::Unknown, 0.9, 0.8);
        // Would be CloudLlm but budget only allows LocalLlm
        assert_eq!(tier, BackendTier::LocalLlm);
    }

    #[test]
    fn test_budget_exhausted_forces_native() {
        let mut dispatcher = IntelligentDispatcher::simulated().with_energy_budget(5.0);
        dispatcher.total_energy = 4.5; // only 0.5 remaining

        let tier = dispatcher.select_tier(EpistemicStatus::Probable, 0.5, 0.8);
        // Would be LocalLlm (cost 10.0) but only 0.5 remaining → Native
        assert_eq!(tier, BackendTier::Native);
    }

    #[test]
    fn test_backend_stats_tracking() {
        let mut stats = BackendStats::default();
        assert_eq!(stats.success_rate(), 0.5); // prior

        stats.record(true);
        stats.record(true);
        stats.record(false);
        assert!((stats.success_rate() - 0.667).abs() < 0.01);
    }

    #[tokio::test]
    async fn test_generate_native_tier() {
        let mut dispatcher = IntelligentDispatcher::simulated();
        let params = GenerationParams::default();

        let result = dispatcher
            .generate("test", &params, EpistemicStatus::Certain, 0.1, 0.8)
            .await;

        assert_eq!(result.tier, BackendTier::Native);
        assert!(result.success);
        assert_eq!(result.energy_cost, COST_NATIVE);
        assert!(result.output.contains("NATIVE"));
    }

    #[tokio::test]
    async fn test_generate_local_llm_tier() {
        let mut dispatcher = IntelligentDispatcher::simulated();
        let params = GenerationParams::default();

        let result = dispatcher
            .generate("test", &params, EpistemicStatus::Probable, 0.5, 0.8)
            .await;

        assert_eq!(result.tier, BackendTier::LocalLlm);
        assert!(result.success);
        assert_eq!(result.energy_cost, COST_LOCAL_LLM);
    }

    #[tokio::test]
    async fn test_energy_accumulates() {
        let mut dispatcher = IntelligentDispatcher::simulated();
        let params = GenerationParams::default();

        let _ = dispatcher
            .generate("a", &params, EpistemicStatus::Certain, 0.1, 0.8)
            .await;
        let _ = dispatcher
            .generate("b", &params, EpistemicStatus::Probable, 0.5, 0.8)
            .await;

        assert_eq!(dispatcher.total_energy(), COST_NATIVE + COST_LOCAL_LLM);
    }

    #[test]
    fn test_record_outcome_updates_stats() {
        let mut dispatcher = IntelligentDispatcher::simulated();

        // Prior: 0.5 for all
        assert_eq!(dispatcher.success_rate(BackendTier::LocalLlm), 0.5);

        // Record outcomes
        dispatcher.record_outcome(BackendTier::LocalLlm, true);
        dispatcher.record_outcome(BackendTier::LocalLlm, true);
        dispatcher.record_outcome(BackendTier::LocalLlm, false);

        assert!((dispatcher.success_rate(BackendTier::LocalLlm) - 0.667).abs() < 0.01);

        // Native should still be at prior
        assert_eq!(dispatcher.success_rate(BackendTier::Native), 0.5);
    }

    #[test]
    fn test_record_outcome_simulated_is_noop() {
        let mut dispatcher = IntelligentDispatcher::simulated();
        // Should not panic
        dispatcher.record_outcome(BackendTier::Simulated, true);
        dispatcher.record_outcome(BackendTier::Simulated, false);
    }

    // ── Adaptive Routing Tests ────────────────────────────────────────────

    #[test]
    fn test_task_tier_stats_tracking() {
        let mut dispatcher = IntelligentDispatcher::simulated();

        // Record outcomes for specific task types
        dispatcher.record_outcome_for_task(
            BackendTier::Native,
            true,
            Some(CodeTaskType::Create),
        );
        dispatcher.record_outcome_for_task(
            BackendTier::Native,
            true,
            Some(CodeTaskType::Create),
        );
        dispatcher.record_outcome_for_task(
            BackendTier::Native,
            false,
            Some(CodeTaskType::Create),
        );

        // Task-specific rate (3 attempts, 2 successes)
        let rate = dispatcher.task_success_rate(CodeTaskType::Create, BackendTier::Native);
        assert!((rate - 0.667).abs() < 0.01, "Rate should be ~66%: {rate}");

        // Different task type should fall back to global
        let debug_rate = dispatcher.task_success_rate(CodeTaskType::Debug, BackendTier::Native);
        // Global native_stats has the same 3 recordings
        assert!((debug_rate - 0.667).abs() < 0.01);
    }

    #[test]
    fn test_adaptive_downgrade_on_poor_success() {
        let mut dispatcher = IntelligentDispatcher::new(
            Arc::new(SimulatedBackend),
            Some(Arc::new(SimulatedBackend)),
        );

        // Record 3 failures for CloudLlm on Debug tasks
        for _ in 0..3 {
            dispatcher.record_outcome_for_task(
                BackendTier::CloudLlm,
                false,
                Some(CodeTaskType::Debug),
            );
        }

        // CloudLlm now has 0% success on Debug → should downgrade to LocalLlm
        let tier = dispatcher.select_tier_for_task(
            EpistemicStatus::Uncertain,
            0.8,
            0.8,
            Some(CodeTaskType::Debug),
        );
        assert_eq!(
            tier,
            BackendTier::LocalLlm,
            "Should downgrade CloudLlm after 3 failures"
        );

        // Create tasks should still get CloudLlm (no data)
        let tier = dispatcher.select_tier_for_task(
            EpistemicStatus::Uncertain,
            0.8,
            0.8,
            Some(CodeTaskType::Create),
        );
        assert_eq!(
            tier,
            BackendTier::CloudLlm,
            "Create should still use CloudLlm"
        );
    }

    #[test]
    fn test_adaptive_requires_minimum_attempts() {
        let mut dispatcher = IntelligentDispatcher::new(
            Arc::new(SimulatedBackend),
            Some(Arc::new(SimulatedBackend)),
        );

        // Only 2 failures — not enough to trigger downgrade
        dispatcher.record_outcome_for_task(
            BackendTier::CloudLlm,
            false,
            Some(CodeTaskType::Debug),
        );
        dispatcher.record_outcome_for_task(
            BackendTier::CloudLlm,
            false,
            Some(CodeTaskType::Debug),
        );

        let tier = dispatcher.select_tier_for_task(
            EpistemicStatus::Uncertain,
            0.8,
            0.8,
            Some(CodeTaskType::Debug),
        );
        assert_eq!(
            tier,
            BackendTier::CloudLlm,
            "Should NOT downgrade with only 2 attempts"
        );
    }

    #[test]
    fn test_global_adaptive_downgrade() {
        let mut dispatcher = IntelligentDispatcher::simulated();

        // Record 4 global failures for LocalLlm
        for _ in 0..4 {
            dispatcher.record_outcome(BackendTier::LocalLlm, false);
        }

        // Without task type, global stats should trigger downgrade
        let tier = dispatcher.select_tier(EpistemicStatus::Probable, 0.5, 0.8);
        assert_eq!(
            tier,
            BackendTier::Native,
            "Should downgrade LocalLlm after 4 global failures"
        );
    }

    #[test]
    fn test_select_tier_for_task_with_none_matches_select_tier() {
        let dispatcher = IntelligentDispatcher::simulated();

        let tier1 = dispatcher.select_tier(EpistemicStatus::Certain, 0.1, 0.8);
        let tier2 =
            dispatcher.select_tier_for_task(EpistemicStatus::Certain, 0.1, 0.8, None);

        assert_eq!(tier1, tier2, "None task type should match base select_tier");
    }

    #[test]
    fn test_task_success_rate_falls_back_to_global() {
        let mut dispatcher = IntelligentDispatcher::simulated();

        // Only global stats, no task-specific
        dispatcher.record_outcome(BackendTier::Native, true);
        dispatcher.record_outcome(BackendTier::Native, true);
        dispatcher.record_outcome(BackendTier::Native, false);

        // Task-specific with < 3 attempts → falls back to global
        let rate = dispatcher.task_success_rate(CodeTaskType::Create, BackendTier::Native);
        assert!((rate - 0.667).abs() < 0.01);
    }
}
