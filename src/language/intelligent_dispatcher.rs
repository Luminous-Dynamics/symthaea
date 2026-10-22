// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Intelligent Dispatcher: consciousness-routed code generation backend selection.
//!
//! Routes coding tasks to the optimal backend based on consciousness state:
//!
//! | Epistemic Status | Prediction Error | Backend |
//! |-----------------|-----------------|---------|
//! | Certain | Low | Native (CodeGenerator via HDC+CfC, 0 latency) |
//! | Probable | Medium | Local LLM via Ollama (default model gemma4:e2b; override with SYMTHAEA_LLM_MODEL) |
//! | Uncertain/Unknown | High | Cloud LLM (Claude API) |
//!
//! Tracks per-backend success rates and respects metabolic budget constraints.
//! Native code generation costs 1.0 energy, local LLM costs 10.0, cloud costs 50.0.

use crate::language::llm_backend::{GenerationParams, LLMBackend, SimulatedBackend};
use crate::mind::structured_thought::EpistemicStatus;
use std::sync::Arc;
use std::time::Instant;

/// Metabolic energy cost per backend tier.
const COST_NATIVE: f64 = 1.0;
const COST_LOCAL_LLM: f64 = 10.0;
const COST_CLOUD_LLM: f64 = 50.0;

/// Which backend tier was selected for a generation request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendTier {
    /// Native HDC+CfC code generation (zero external calls).
    Native,
    /// Specialized Hardware Driver Emitter (I2C/MMIO).
    Hardware,
    /// Local LLM via Ollama (model defaults to gemma4:e2b and is configurable with SYMTHAEA_LLM_MODEL).
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
            Self::Hardware => write!(f, "Hardware"),
            Self::LocalLlm => write!(f, "LocalLLM"),
            Self::CloudLlm => write!(f, "CloudLLM"),
            Self::Simulated => write!(f, "Simulated"),
        }
    }
}

impl BackendTier {
    /// Metabolic energy estimate assigned to an effective backend tier.
    /// This is an internal budget unit, not provider billing or measured joules.
    pub fn estimated_energy_cost(self) -> f64 {
        match self {
            Self::Native | Self::Hardware => COST_NATIVE,
            Self::LocalLlm => COST_LOCAL_LLM,
            Self::CloudLlm => COST_CLOUD_LLM,
            Self::Simulated => 0.0,
        }
    }
}

/// One backend invocation observed during a dispatch, including fallback attempts.
#[derive(Debug, Clone, PartialEq)]
pub struct BackendAttempt {
    /// Effective tier inferred from the backend implementation that was invoked.
    pub tier: BackendTier,
    /// Backend-reported name (e.g. Ollama, Anthropic, Simulated).
    pub backend_name: String,
    /// Whether the backend returned a non-empty response.
    pub success: bool,
    /// Whether an internal backend verifier accepted the candidate, when applicable.
    /// `None` means this backend does not report internal verification.
    pub verification_passed: Option<bool>,
    /// Wall-clock time spent in this backend invocation, including failures.
    pub elapsed_ms: u128,
    /// Internal metabolic estimate assigned to the invocation's dispatch tier; not billing.
    pub estimated_energy_cost: f64,
}

/// Result of an intelligent dispatch decision.
#[derive(Debug, Clone)]
pub struct DispatchResult {
    /// Generated code (or error message).
    pub output: String,
    /// Selected routing tier before fallback; this can differ from backend_attempts.
    pub tier: BackendTier,
    /// Actual backend invocations, in order, including failed calls and fallbacks.
    pub backend_attempts: Vec<BackendAttempt>,
    /// Energy cost of this generation.
    pub energy_cost: f64,
    /// Whether the final generation result was successful.
    pub success: bool,
}

fn effective_backend_tier(backend_name: &str, selected_tier: BackendTier) -> BackendTier {
    if backend_name.eq_ignore_ascii_case("simulated") {
        BackendTier::Simulated
    } else if backend_name.eq_ignore_ascii_case("ollama") {
        BackendTier::LocalLlm
    } else {
        selected_tier
    }
}

fn record_backend_attempt(
    attempts: &mut Vec<BackendAttempt>,
    backend_name: &str,
    selected_tier: BackendTier,
    success: bool,
    elapsed_ms: u128,
) {
    let effective_tier = effective_backend_tier(backend_name, selected_tier);
    attempts.push(BackendAttempt {
        tier: effective_tier,
        backend_name: backend_name.to_string(),
        success,
        verification_passed: None,
        elapsed_ms,
        estimated_energy_cost: effective_tier.estimated_energy_cost(),
    });
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

/// Code task category for per-category routing (Phase 6).
///
/// The dispatcher tracks success rates per category, enabling it to learn
/// that e.g. native handles arithmetic well but needs LLM for graph algorithms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CodeTaskCategory {
    Arithmetic,
    StringOps,
    CollectionOps,
    BooleanChecks,
    MathFunctions,
    Sorting,
    Search,
    IteratorChains,
    ErrorHandling,
    StructDefinition,
    TraitDefinition,
    HardwareDriver,
    General,
}

impl CodeTaskCategory {
    /// Infer the category from a purpose string.
    pub fn from_purpose(purpose: &str) -> Self {
        let p = purpose.to_lowercase();
        if p.contains("i2c")
            || p.contains("register")
            || p.contains("driver")
            || p.contains("sensor")
            || p.contains("mmio")
            || p.contains("hardware")
        {
            Self::HardwareDriver
        } else if p.contains("sort") || p.contains("order") || p.contains("arrange") {
            Self::Sorting
        } else if p.contains("search") || p.contains("find") || p.contains("lookup") {
            Self::Search
        } else if p.contains("add")
            || p.contains("subtract")
            || p.contains("multiply")
            || p.contains("divide")
            || p.contains("sum")
            || p.contains("product")
        {
            Self::Arithmetic
        } else if p.contains("reverse")
            || p.contains("uppercase")
            || p.contains("lowercase")
            || p.contains("trim")
            || p.contains("split")
            || p.contains("join")
            || p.contains("replace")
            || p.contains("string")
        {
            Self::StringOps
        } else if p.contains("filter")
            || p.contains("map")
            || p.contains("collect")
            || p.contains("flatten")
            || p.contains("zip")
            || p.contains("chain")
        {
            Self::IteratorChains
        } else if p.contains("is even")
            || p.contains("is odd")
            || p.contains("is empty")
            || p.contains("is positive")
            || p.contains("is negative")
            || p.contains("palindrome")
        {
            Self::BooleanChecks
        } else if p.contains("factorial")
            || p.contains("fibonacci")
            || p.contains("gcd")
            || p.contains("sqrt")
            || p.contains("prime")
            || p.contains("distance")
        {
            Self::MathFunctions
        } else if p.contains("error")
            || p.contains("result")
            || p.contains("parse")
            || p.contains("validate")
        {
            Self::ErrorHandling
        } else if p.contains("struct") || p.contains("fields") {
            Self::StructDefinition
        } else if p.contains("trait") || p.contains("implement") {
            Self::TraitDefinition
        } else {
            Self::General
        }
    }
}

/// Consciousness-routed code generation dispatcher.
///
/// Selects the optimal backend tier based on epistemic status, prediction error,
/// and metabolic budget. Falls back gracefully: cloud → local → native → simulated.
///
/// Phase 6 additions: per-category success tracking, MAGI-style prediction
/// confidence, and honest "I don't know" refusal path.
pub struct IntelligentDispatcher {
    /// Local LLM backend (Ollama; defaults to gemma4:e2b unless SYMTHAEA_LLM_MODEL overrides it).
    local_llm: Arc<dyn LLMBackend>,
    /// Cloud LLM backend (Anthropic Claude, if available).
    cloud_llm: Option<Arc<dyn LLMBackend>>,
    /// Per-tier success tracking.
    native_stats: BackendStats,
    local_stats: BackendStats,
    cloud_stats: BackendStats,
    /// Phase 6: Per-category native success tracking.
    /// Enables the dispatcher to learn which categories native handles well.
    category_native_stats: std::collections::HashMap<CodeTaskCategory, BackendStats>,
    /// Phase 6: MAGI-style prediction confidence (EMA of prediction accuracy).
    /// When low, the dispatcher is less confident in its own routing decisions.
    prediction_confidence: f64,
    /// Phase 6: Total predictions made
    prediction_count: u64,
    /// Cumulative energy spent.
    total_energy: f64,
    /// Maximum energy budget (0.0 = unlimited).
    energy_budget: f64,
    /// If set, forces the next `select_tier` call to return this tier.
    forced_tier: Option<BackendTier>,
}

impl IntelligentDispatcher {
    /// Create a dispatcher with the given LLM backends.
    pub fn new(local_llm: Arc<dyn LLMBackend>, cloud_llm: Option<Arc<dyn LLMBackend>>) -> Self {
        Self {
            local_llm,
            cloud_llm,
            native_stats: BackendStats::default(),
            local_stats: BackendStats::default(),
            cloud_stats: BackendStats::default(),
            category_native_stats: std::collections::HashMap::new(),
            prediction_confidence: 0.5, // start with neutral confidence
            prediction_count: 0,
            total_energy: 0.0,
            energy_budget: 0.0,
            forced_tier: None,
        }
    }

    /// Create a dispatcher with simulated backends (for testing).
    pub fn simulated() -> Self {
        Self::new(Arc::new(SimulatedBackend), None)
    }

    /// Create a dispatcher with Ollama as the local LLM backend (defaults to gemma4:e2b unless SYMTHAEA_LLM_MODEL overrides it).
    ///
    /// Attempts to connect to Ollama at `localhost:11434`. If Ollama is not available,
    /// the dispatcher still works — generation will fail gracefully and fall back to
    /// Native tier on subsequent requests via the Bayesian stats tracker.
    pub fn with_local_llm() -> Self {
        use crate::language::llm_backend::OllamaBackend;
        Self::new(Arc::new(OllamaBackend::new()), None)
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
    /// 4. Budget exceeded → fall back to cheaper tier
    /// Force the next `select_tier` call to return the given tier.
    /// Consumed on first call (one-shot override).
    pub fn force_next_tier(&mut self, tier: BackendTier) {
        self.forced_tier = Some(tier);
    }

    pub fn select_tier(
        &mut self,
        epistemic: EpistemicStatus,
        prediction_error: f64,
        consciousness_level: f64,
    ) -> BackendTier {
        // Check for forced tier (one-shot override)
        if let Some(tier) = self.forced_tier.take() {
            return tier;
        }

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

        // Bayesian override: if we have enough data (5+ attempts) and the selected
        // tier's success rate is poor (<30%), try the next tier with better stats.
        let base_tier = self.bayesian_adjust(base_tier);

        // Consciousness modulation: very low consciousness → don't trust expensive backends
        if consciousness_level < 0.2 {
            return BackendTier::Native;
        }

        // Budget constraints: fall back to cheaper tiers
        match base_tier {
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
        // Use purpose text if available in params or prompt to detect hardware
        let category =
            if prompt.to_lowercase().contains("driver") || prompt.to_lowercase().contains("i2c") {
                CodeTaskCategory::HardwareDriver
            } else {
                CodeTaskCategory::General
            };

        let tier = if category == CodeTaskCategory::HardwareDriver {
            BackendTier::Hardware
        } else {
            self.select_tier(epistemic, prediction_error, consciousness_level)
        };

        let mut backend_attempts = Vec::new();
        let (output, success, cost) = match tier {
            BackendTier::Native => {
                // Native generation is handled externally by CodeGenerator.
                // The selected tier is tracked separately from LLM backend attempts.
                self.native_stats.record(true);
                ("[NATIVE: use CodeGenerator]".to_string(), true, COST_NATIVE)
            }
            BackendTier::Hardware => {
                // Hardware generation is handled externally by DriverEmitter.
                ("[HARDWARE: use DriverEmitter]".to_string(), true, COST_NATIVE)
            }
            BackendTier::LocalLlm => {
                let backend_name = self.local_llm.name().to_string();
                let attempt_started = Instant::now();
                let generated = self.local_llm.generate(prompt, params).await;
                let attempt_elapsed_ms = attempt_started.elapsed().as_millis();
                match generated {
                    Ok(output) => {
                        let success = !output.is_empty();
                        self.local_stats.record(success);
                        record_backend_attempt(
                            &mut backend_attempts,
                            &backend_name,
                            BackendTier::LocalLlm,
                            success,
                            attempt_elapsed_ms,
                        );
                        (output, success, COST_LOCAL_LLM)
                    }
                    Err(e) => {
                        self.local_stats.record(false);
                        record_backend_attempt(
                            &mut backend_attempts,
                            &backend_name,
                            BackendTier::LocalLlm,
                            false,
                            attempt_elapsed_ms,
                        );
                        (format!("Local LLM error: {e}"), false, COST_LOCAL_LLM)
                    }
                }
            }
            BackendTier::CloudLlm => {
                if let Some(ref cloud) = self.cloud_llm {
                    let cloud_name = cloud.name().to_string();
                    let attempt_started = Instant::now();
                    let generated = cloud.generate(prompt, params).await;
                    let attempt_elapsed_ms = attempt_started.elapsed().as_millis();
                    match generated {
                        Ok(output) => {
                            let success = !output.is_empty();
                            self.cloud_stats.record(success);
                            record_backend_attempt(
                                &mut backend_attempts,
                                &cloud_name,
                                BackendTier::CloudLlm,
                                success,
                                attempt_elapsed_ms,
                            );
                            (output, success, COST_CLOUD_LLM)
                        }
                        Err(e) => {
                            self.cloud_stats.record(false);
                            record_backend_attempt(
                                &mut backend_attempts,
                                &cloud_name,
                                BackendTier::CloudLlm,
                                false,
                                attempt_elapsed_ms,
                            );
                            tracing::warn!(
                                target: "symthaea::dispatcher",
                                error = %e,
                                "Cloud LLM failed, falling back to local"
                            );
                            let local_name = self.local_llm.name().to_string();
                            let fallback_started = Instant::now();
                            let fallback_result = self.local_llm.generate(prompt, params).await;
                            let fallback_elapsed_ms = fallback_started.elapsed().as_millis();
                            match fallback_result {
                                Ok(output) => {
                                    let success = !output.is_empty();
                                    self.local_stats.record(success);
                                    record_backend_attempt(
                                        &mut backend_attempts,
                                        &local_name,
                                        BackendTier::LocalLlm,
                                        success,
                                        fallback_elapsed_ms,
                                    );
                                    (output, success, COST_CLOUD_LLM + COST_LOCAL_LLM)
                                }
                                Err(e2) => {
                                    self.local_stats.record(false);
                                    record_backend_attempt(
                                        &mut backend_attempts,
                                        &local_name,
                                        BackendTier::LocalLlm,
                                        false,
                                        fallback_elapsed_ms,
                                    );
                                    (
                                        format!("All LLMs failed: cloud={e}, local={e2}"),
                                        false,
                                        COST_CLOUD_LLM + COST_LOCAL_LLM,
                                    )
                                }
                            }
                        }
                    }
                } else {
                    // A forced CloudLlm tier can reach this branch without a cloud
                    // backend; record only the backend that is actually invoked.
                    let local_name = self.local_llm.name().to_string();
                    let attempt_started = Instant::now();
                    let generated = self.local_llm.generate(prompt, params).await;
                    let attempt_elapsed_ms = attempt_started.elapsed().as_millis();
                    match generated {
                        Ok(output) => {
                            let success = !output.is_empty();
                            self.local_stats.record(success);
                            record_backend_attempt(
                                &mut backend_attempts,
                                &local_name,
                                BackendTier::LocalLlm,
                                success,
                                attempt_elapsed_ms,
                            );
                            (output, success, COST_LOCAL_LLM)
                        }
                        Err(e) => {
                            self.local_stats.record(false);
                            record_backend_attempt(
                                &mut backend_attempts,
                                &local_name,
                                BackendTier::LocalLlm,
                                false,
                                attempt_elapsed_ms,
                            );
                            (format!("Local LLM error: {e}"), false, COST_LOCAL_LLM)
                        }
                    }
                }
            }
            BackendTier::Simulated => {
                record_backend_attempt(
                    &mut backend_attempts,
                    "Simulated",
                    BackendTier::Simulated,
                    true,
                    0,
                );
                ("// simulated code output".to_string(), true, 0.0)
            }
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
            backend_attempts,
            energy_cost: cost,
            success,
        }
    }

    /// Get success rate for a given tier.
    pub fn success_rate(&self, tier: BackendTier) -> f64 {
        match tier {
            BackendTier::Native => self.native_stats.success_rate(),
            BackendTier::Hardware => 1.0, // specialized emitter is deterministic
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
        match tier {
            BackendTier::Native => self.native_stats.record(success),
            BackendTier::Hardware => {} // no tracking needed for deterministic emitter
            BackendTier::LocalLlm => self.local_stats.record(success),
            BackendTier::CloudLlm => self.cloud_stats.record(success),
            BackendTier::Simulated => {} // no tracking for simulated
        }
        tracing::debug!(
            target: "symthaea::dispatcher",
            tier = %tier,
            success = success,
            rate = self.success_rate(tier),
            "Recorded generation outcome"
        );
    }

    // =========================================================================
    // Phase 6: Epistemic Gating & Category-Aware Routing
    // =========================================================================

    /// Select tier with category-aware routing (Phase 6 enhancement).
    ///
    /// Uses per-category native success rates to make smarter routing decisions.
    /// If native has proven itself for this category (>60% success, 5+ attempts),
    /// routes to native even when base routing would choose LLM.
    pub fn select_tier_with_category(
        &mut self,
        epistemic: EpistemicStatus,
        prediction_error: f64,
        consciousness_level: f64,
        purpose: &str,
    ) -> BackendTier {
        let category = CodeTaskCategory::from_purpose(purpose);

        // Hardware intents are routed directly to the specialized emitter (INV-13)
        if category == CodeTaskCategory::HardwareDriver {
            return BackendTier::Hardware;
        }

        // Get base routing decision
        let base_tier = self.select_tier(epistemic, prediction_error, consciousness_level);

        // Phase 6: If base says LLM but native has proven itself for this category, use native
        if base_tier != BackendTier::Native {
            if let Some(cat_stats) = self.category_native_stats.get(&category) {
                if cat_stats.attempts >= 5 && cat_stats.success_rate() > 0.6 {
                    tracing::info!(
                        target: "symthaea::dispatcher",
                        category = ?category,
                        native_rate = cat_stats.success_rate(),
                        "Phase 6: category-aware override → Native (proven for this category)"
                    );
                    return BackendTier::Native;
                }
            }
        }

        base_tier
    }

    /// Record the outcome of a generation with category tracking (Phase 6).
    pub fn record_outcome_with_category(
        &mut self,
        tier: BackendTier,
        success: bool,
        purpose: &str,
    ) {
        self.record_outcome(tier, success);

        // Phase 6: also track per-category for native
        if tier == BackendTier::Native {
            let category = CodeTaskCategory::from_purpose(purpose);
            self.category_native_stats
                .entry(category)
                .or_default()
                .record(success);
        }
    }

    /// Update MAGI-style prediction confidence based on routing accuracy.
    ///
    /// Called after a generation with the predicted tier and whether the
    /// prediction was correct (code compiled/passed tests).
    pub fn update_prediction_confidence(&mut self, predicted_success: bool, actual_success: bool) {
        self.prediction_count += 1;
        let correct = predicted_success == actual_success;
        let alpha = 0.1; // EMA smoothing
        self.prediction_confidence =
            self.prediction_confidence * (1.0 - alpha) + if correct { 1.0 } else { 0.0 } * alpha;
    }

    /// Get the current MAGI-style prediction confidence (0.0-1.0).
    pub fn prediction_confidence(&self) -> f64 {
        self.prediction_confidence
    }

    /// Check if the system should honestly refuse to generate code.
    ///
    /// Returns `Some(reason)` if the dispatcher recommends an "I don't know" response.
    /// Conditions:
    /// - OutOfDomain epistemic status + low prediction confidence
    /// - All tiers have <20% success rate for this category
    /// - Budget exhausted
    pub fn should_refuse(&self, epistemic: EpistemicStatus, purpose: &str) -> Option<String> {
        // Condition 1: OutOfDomain + low confidence
        if epistemic == EpistemicStatus::OutOfDomain && self.prediction_confidence < 0.3 {
            return Some(format!(
                "Out of domain: '{}' is outside the system's training distribution (confidence: {:.1}%)",
                purpose,
                self.prediction_confidence * 100.0
            ));
        }

        // Condition 2: All tiers failing for this category
        let category = CodeTaskCategory::from_purpose(purpose);
        if let Some(cat_stats) = self.category_native_stats.get(&category) {
            if cat_stats.attempts >= 10 && cat_stats.success_rate() < 0.2 {
                let local_rate = self.local_stats.success_rate();
                let cloud_rate = self.cloud_stats.success_rate();
                if local_rate < 0.2 && cloud_rate < 0.2 {
                    return Some(format!(
                        "All backends consistently fail for {:?} tasks (native: {:.0}%, local: {:.0}%, cloud: {:.0}%)",
                        category,
                        cat_stats.success_rate() * 100.0,
                        local_rate * 100.0,
                        cloud_rate * 100.0
                    ));
                }
            }
        }

        // Condition 3: Budget exhausted
        if self.energy_budget > 0.0 && self.total_energy >= self.energy_budget {
            return Some(
                "Energy budget exhausted — cannot generate more code this session".to_string(),
            );
        }

        None
    }

    /// Get native success rate for a specific category.
    pub fn category_success_rate(&self, category: CodeTaskCategory) -> Option<f64> {
        self.category_native_stats
            .get(&category)
            .map(|s| s.success_rate())
    }

    /// Bayesian tier adjustment: if the selected tier has poor success rate
    /// (>= 5 attempts, < 30% success) and another tier has better stats, switch.
    fn bayesian_adjust(&self, tier: BackendTier) -> BackendTier {
        let stats = match tier {
            BackendTier::Native => &self.native_stats,
            BackendTier::LocalLlm => &self.local_stats,
            BackendTier::CloudLlm => &self.cloud_stats,
            BackendTier::Simulated | BackendTier::Hardware => return tier,
        };

        if stats.attempts < 5 || stats.success_rate() >= 0.3 {
            return tier;
        }

        let alternatives = match tier {
            BackendTier::Native => vec![BackendTier::LocalLlm, BackendTier::CloudLlm],
            BackendTier::LocalLlm => vec![BackendTier::CloudLlm, BackendTier::Native],
            BackendTier::CloudLlm => vec![BackendTier::LocalLlm, BackendTier::Native],
            BackendTier::Simulated | BackendTier::Hardware => return tier,
        };

        for alt in alternatives {
            let alt_rate = match alt {
                BackendTier::Native => self.native_stats.success_rate(),
                BackendTier::LocalLlm => self.local_stats.success_rate(),
                BackendTier::CloudLlm => {
                    if self.cloud_llm.is_none() {
                        continue;
                    }
                    self.cloud_stats.success_rate()
                }
                BackendTier::Simulated | BackendTier::Hardware => continue,
            };
            if alt_rate > stats.success_rate() {
                tracing::info!(
                    target: "symthaea::dispatcher",
                    from = %tier,
                    to = %alt,
                    from_rate = stats.success_rate(),
                    to_rate = alt_rate,
                    "Bayesian override: switching to tier with better success rate"
                );
                return alt;
            }
        }

        tier
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

    struct FailingBackend {
        backend_name: &'static str,
    }

    #[async_trait::async_trait]
    impl LLMBackend for FailingBackend {
        async fn generate(&self, _prompt: &str, _params: &GenerationParams) -> anyhow::Result<String> {
            anyhow::bail!("intentional test backend failure")
        }

        async fn is_available(&self) -> bool {
            true
        }

        fn name(&self) -> &str {
            self.backend_name
        }
    }

    #[tokio::test]
    async fn backend_attempts_report_simulated_implementation_under_local_selection() {
        let mut dispatcher = IntelligentDispatcher::new(Arc::new(SimulatedBackend), None);
        dispatcher.force_next_tier(BackendTier::LocalLlm);
        let result = dispatcher
            .generate("test", &GenerationParams::default(), EpistemicStatus::Probable, 0.4, 0.8)
            .await;

        assert_eq!(result.tier, BackendTier::LocalLlm, "selected tier remains visible");
        assert_eq!(result.backend_attempts.len(), 1);
        assert_eq!(result.backend_attempts[0].tier, BackendTier::Simulated);
        assert_eq!(result.backend_attempts[0].backend_name, "Simulated");
        assert!(result.backend_attempts[0].success);
    }

    #[tokio::test]
    async fn cloud_failure_records_cloud_then_actual_fallback_backend() {
        let mut dispatcher = IntelligentDispatcher::new(
            Arc::new(SimulatedBackend),
            Some(Arc::new(FailingBackend { backend_name: "Anthropic" })),
        );
        dispatcher.force_next_tier(BackendTier::CloudLlm);
        let result = dispatcher
            .generate("test", &GenerationParams::default(), EpistemicStatus::Uncertain, 0.8, 0.8)
            .await;

        assert_eq!(result.tier, BackendTier::CloudLlm);
        assert_eq!(result.backend_attempts.len(), 2);
        assert_eq!(result.backend_attempts[0].backend_name, "Anthropic");
        assert_eq!(result.backend_attempts[0].tier, BackendTier::CloudLlm);
        assert!(!result.backend_attempts[0].success);
        assert_eq!(result.backend_attempts[1].backend_name, "Simulated");
        assert_eq!(result.backend_attempts[1].tier, BackendTier::Simulated);
        assert!(result.backend_attempts[1].success);
        assert_eq!(result.backend_attempts[0].estimated_energy_cost, COST_CLOUD_LLM);
        assert_eq!(result.backend_attempts[1].estimated_energy_cost, 0.0);
        assert_eq!(result.energy_cost, COST_CLOUD_LLM + COST_LOCAL_LLM);
        assert_eq!(dispatcher.total_energy(), COST_CLOUD_LLM + COST_LOCAL_LLM);
    }

    #[test]
    fn test_tier_display() {
        assert_eq!(format!("{}", BackendTier::Native), "Native");
        assert_eq!(format!("{}", BackendTier::LocalLlm), "LocalLLM");
        assert_eq!(format!("{}", BackendTier::CloudLlm), "CloudLLM");
    }

    #[test]
    fn test_select_tier_certain_low_error() {
        let mut dispatcher = IntelligentDispatcher::simulated();
        let tier = dispatcher.select_tier(EpistemicStatus::Certain, 0.1, 0.8);
        assert_eq!(tier, BackendTier::Native);
    }

    #[test]
    fn test_select_tier_probable_medium_error() {
        let mut dispatcher = IntelligentDispatcher::simulated();
        let tier = dispatcher.select_tier(EpistemicStatus::Probable, 0.4, 0.8);
        assert_eq!(tier, BackendTier::LocalLlm);
    }

    #[test]
    fn test_select_tier_uncertain_uses_local_when_no_cloud() {
        let mut dispatcher = IntelligentDispatcher::simulated();
        // No cloud backend → falls back to LocalLlm
        let tier = dispatcher.select_tier(EpistemicStatus::Uncertain, 0.8, 0.8);
        assert_eq!(tier, BackendTier::LocalLlm);
    }

    #[test]
    fn test_select_tier_uncertain_uses_cloud_when_available() {
        let mut dispatcher = IntelligentDispatcher::new(
            Arc::new(SimulatedBackend),
            Some(Arc::new(SimulatedBackend)),
        );
        let tier = dispatcher.select_tier(EpistemicStatus::Uncertain, 0.8, 0.8);
        assert_eq!(tier, BackendTier::CloudLlm);
    }

    #[test]
    fn test_low_consciousness_forces_native() {
        let mut dispatcher = IntelligentDispatcher::new(
            Arc::new(SimulatedBackend),
            Some(Arc::new(SimulatedBackend)),
        );
        // Even with Uncertain status, low consciousness → Native
        let tier = dispatcher.select_tier(EpistemicStatus::Uncertain, 0.9, 0.1);
        assert_eq!(tier, BackendTier::Native);
    }

    #[test]
    fn test_budget_forces_downgrade() {
        let mut dispatcher = IntelligentDispatcher::new(
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

    #[test]
    fn test_bayesian_override_poor_tier() {
        let mut dispatcher = IntelligentDispatcher::simulated();
        for _ in 0..5 {
            dispatcher.record_outcome(BackendTier::Native, false);
        }
        for _ in 0..3 {
            dispatcher.record_outcome(BackendTier::LocalLlm, true);
        }
        let tier = dispatcher.select_tier(EpistemicStatus::Certain, 0.1, 0.8);
        assert_eq!(tier, BackendTier::LocalLlm);
    }

    #[test]
    fn test_bayesian_no_override_insufficient_data() {
        let mut dispatcher = IntelligentDispatcher::simulated();
        for _ in 0..3 {
            dispatcher.record_outcome(BackendTier::Native, false);
        }
        let tier = dispatcher.select_tier(EpistemicStatus::Certain, 0.1, 0.8);
        assert_eq!(tier, BackendTier::Native);
    }

    #[test]
    fn test_bayesian_no_override_acceptable_rate() {
        let mut dispatcher = IntelligentDispatcher::simulated();
        for _ in 0..3 {
            dispatcher.record_outcome(BackendTier::Native, false);
        }
        for _ in 0..2 {
            dispatcher.record_outcome(BackendTier::Native, true);
        }
        let tier = dispatcher.select_tier(EpistemicStatus::Certain, 0.1, 0.8);
        assert_eq!(tier, BackendTier::Native);
    }
}
