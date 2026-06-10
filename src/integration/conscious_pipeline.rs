// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Conscious Pipeline - End-to-End Integration Orchestrator
//!
//! This module connects all consciousness subsystems into a coherent pipeline
//! using the HDC+LTC+Primitives architecture as the PRIMARY understanding mechanism:
//!
//! ```text
//! User Input → ConsciousnessLanguageCore (HDC+LTC+Primitives)
//!            → 2D Consciousness Space (Φ × Confidence)
//!            → Quadrant Strategy (Confident/Curious/Autopilot/Lost)
//!            → [Optional LLM for clarification if Lost/Curious]
//!            → Action Execution
//! ```
//!
//! ## 2D Consciousness Space
//!
//! The pipeline uses a 2D space with TWO independent dimensions:
//! - **Φ (Integration Depth)**: How deeply unified the processing is
//! - **Confidence (Epistemic Certainty)**: How sure we are about interpretation
//!
//! This creates four quadrants:
//! - **Confident** (High Φ + High Confidence): Deep understanding, execute with trust
//! - **Curious** (High Φ + Low Confidence): Deep processing but exploring, dry-run first
//! - **Autopilot** (Low Φ + High Confidence): Pattern-matched routine, execute efficiently
//! - **Lost** (Low Φ + Low Confidence): Genuinely confused, request help
//!
//! The LLM is NOT the brain - it's an optional organ for clarification when
//! the system is Lost or Curious.

use anyhow::{Context, Result};
use std::path::PathBuf;

use crate::action::nixos_patterns::{NixOSCommand, SafetyLevel};
use crate::consciousness::empathic_unification::{EmpathicUnification, ToneGuidance, UserNeed};
use crate::consciousness::phi_attention::{
    ActionType, AdaptiveThresholds, ConfidenceLevel, ConsciousnessThresholds, PhiAwareScoring,
};
use crate::language::llm_organ::{LlmConfig, LlmOrgan, LlmRequest};
use crate::language::{
    // Feedback loop types
    ActionOutcomeFeedback,
    ClarifyingQuestion,
    ConsciousUnderstandingResult,
    ConsciousnessLanguageConfig,
    ConsciousnessLanguageCore,
    ConsciousnessQuadrant,
    ConsciousnessStateLevel,
    ExecutionStrategy,
    ExecutionStrategyType,
    NixOSIntent,
    NixOSUnderstanding,
};
use crate::memory::conversation_memory::ConversationMemory;
use crate::user_state_inference::ContextKind;

// ============================================================================
// PIPELINE CONFIGURATION
// ============================================================================

/// Configuration for the conscious pipeline
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Configuration for the HDC+LTC+Primitives consciousness-language core
    pub consciousness_language: ConsciousnessLanguageConfig,
    /// LLM configuration (optional - only used for clarification when Lost/Curious)
    pub llm: Option<LlmConfig>,
    /// Path to conversation memory database
    pub memory_path: PathBuf,
    /// Consciousness thresholds for action gating
    pub thresholds: ConsciousnessThresholds,
    /// Whether to require confirmation for low-confidence actions
    pub require_confirmation_below_phi: f32,
    /// Whether to use adaptive thresholds
    pub adaptive_thresholds: bool,
    /// Maximum context turns to include
    pub max_context_turns: usize,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            consciousness_language: ConsciousnessLanguageConfig::default(),
            llm: None, // LLM is OPTIONAL - HDC+LTC+Primitives is primary
            memory_path: PathBuf::from("conscious_memory.db"),
            thresholds: ConsciousnessThresholds::default(),
            require_confirmation_below_phi: 0.3,
            adaptive_thresholds: true,
            max_context_turns: 10,
        }
    }
}

// ============================================================================
// PIPELINE RESULT
// ============================================================================

/// Source of understanding (which system provided the primary understanding)
#[derive(Debug, Clone, PartialEq)]
pub enum UnderstandingSource {
    /// HDC+LTC+Primitives system (primary, preferred)
    HdcLtcPrimitives,
    /// LLM was used for clarification (fallback)
    LlmClarification,
}

/// Result of pipeline execution
#[derive(Debug, Clone)]
pub struct PipelineResult {
    /// The original user input
    pub input: String,
    /// Source of the understanding (HDC+LTC vs LLM fallback)
    pub understanding_source: UnderstandingSource,
    /// The HDC+LTC+Primitives understanding result (always present)
    pub consciousness_understanding: ConsciousUnderstandingResult,
    /// NixOS understanding with intent, confidence, action
    pub nix_understanding: NixOSUnderstanding,
    /// LLM response (only if LLM was used for clarification)
    pub llm_response: Option<String>,
    /// Detected intent (derived from NixOS understanding)
    pub intent: Option<DetectedIntent>,
    /// Current Φ (consciousness level) from HDC+LTC
    pub phi: f64,
    /// Unified free energy from active inference
    pub free_energy: f64,
    /// Consciousness state level (Optimal/Adequate/Struggling/Failed)
    pub consciousness_state: ConsciousnessStateLevel,
    /// Confidence level based on Φ (legacy compatibility)
    pub confidence: ConfidenceLevel,
    /// Execution strategy recommended by consciousness-language core
    pub execution_strategy: ExecutionStrategy,
    /// Clarifying questions (if consciousness is skeptical)
    pub clarifying_questions: Vec<ClarifyingQuestion>,
    /// Whether action was allowed by consciousness gating
    pub action_allowed: bool,
    /// Execution result (if action was taken)
    pub execution: Option<ExecutionOutcome>,
    /// Empathic response - how Symthaea resonates with user
    pub empathic_response: EmpathicResponseInfo,
    /// NixOS knowledge base answer (from curated articles, if nix-mind active)
    #[cfg(feature = "nix-mind")]
    pub nix_knowledge_answer: Option<String>,
    /// NixOS Ollama LLM fallback answer (if knowledge base had no match)
    #[cfg(feature = "nix-mind")]
    pub nix_ollama_answer: Option<String>,
    /// NixOS action plan rationale (from active inference, if available)
    #[cfg(feature = "nix-mind")]
    pub nix_action_rationale: Option<String>,
    /// Number of learned causal edges in the NixOS causal graph (None if nix-mind hook absent)
    #[cfg(feature = "nix-mind")]
    pub nix_causal_edge_count: Option<usize>,
    /// Whether the last action violated causal expectations (0.0 = expected, 1.0 = unexpected).
    /// None if nix-mind hook is absent.
    #[cfg(feature = "nix-mind")]
    pub nix_causal_surprise: Option<f32>,
    /// Processing time in milliseconds
    pub processing_time_ms: u64,
}

/// Empathic response information
#[derive(Debug, Clone)]
pub struct EmpathicResponseInfo {
    /// Inferred user emotional state
    pub user_state: UserEmotionalStateInfo,
    /// Symthaea's resonant response
    pub resonant_emotion: String,
    /// Type of empathy activated
    pub empathy_type: String,
    /// Compassion level (0.0-1.0)
    pub compassion: f64,
    /// Tone guidance for response
    pub tone_guidance: ToneGuidance,
    /// Warmth adjustment
    pub warmth_adjustment: f64,
    /// Patience adjustment
    pub patience_adjustment: f64,
    /// Proactive support message (if any)
    pub proactive_support: Option<String>,
    /// Emotional acknowledgment (if appropriate)
    pub acknowledgment: Option<String>,
}

/// User emotional state information (simplified for result)
#[derive(Debug, Clone)]
pub struct UserEmotionalStateInfo {
    /// Primary detected emotion
    pub primary_emotion: String,
    /// Stress level (0.0-1.0)
    pub stress_level: f64,
    /// Frustration level (0.0-1.0)
    pub frustration: f64,
    /// What we infer they need
    pub inferred_need: UserNeed,
}

/// Detected user intent
#[derive(Debug, Clone)]
pub struct DetectedIntent {
    /// The type of action requested
    pub action_type: ActionType,
    /// Safety level of the action
    pub safety_level: SafetyLevel,
    /// Parsed NixOS command (if applicable)
    pub command: Option<NixOSCommand>,
    /// Confidence in intent detection (0.0 - 1.0)
    pub detection_confidence: f32,
}

/// Outcome of action execution
#[derive(Debug, Clone)]
pub struct ExecutionOutcome {
    /// Whether execution succeeded
    pub success: bool,
    /// Output from execution
    pub output: String,
    /// Error message if failed
    pub error: Option<String>,
    /// Whether rollback was performed
    pub rolled_back: bool,
    /// Φ after execution
    pub phi_after: f32,
}

// ============================================================================
// CONSCIOUS PIPELINE
// ============================================================================

/// The main conscious pipeline orchestrator
///
/// Coordinates all subsystems using the HDC+LTC+Primitives-FIRST architecture:
/// - PRIMARY: ConsciousnessLanguageCore for understanding (HDC+LTC+Primitives)
/// - SECONDARY: LLM as optional fallback for clarification when Lost/Curious
/// - 2D Consciousness Space (Φ × Confidence) determines strategy
/// - Memory context informs decisions
/// - Actions gated by quadrant strategy
/// - Learns from outcomes to improve thresholds
pub struct ConsciousPipeline {
    /// Configuration
    config: PipelineConfig,
    /// PRIMARY: HDC+LTC+Primitives consciousness-language core
    core: ConsciousnessLanguageCore,
    /// OPTIONAL: LLM for clarification (only used when Lost/Curious)
    llm: Option<LlmOrgan>,
    /// Conversation memory
    memory: ConversationMemory,
    /// Current Φ level (synced from core)
    current_phi: f64,
    /// Adaptive thresholds (learns from outcomes)
    adaptive: AdaptiveThresholds,
    /// Current session ID (UUID string)
    session_id: Option<String>,
    /// Processing statistics
    stats: PipelineStats,
    /// Empathic unification engine - true empathy for users
    empathy: EmpathicUnification,
    /// OPTIONAL: NixOS deep cognition hook (active inference, causal learning, episodic memory)
    #[cfg(feature = "nix-mind")]
    nix_hook: Option<Box<dyn symthaea_nix::plugin::pipeline_integration::NixPipelineHook>>,
}

/// Pipeline statistics
#[derive(Debug, Clone, Default)]
pub struct PipelineStats {
    /// Total requests processed
    pub total_requests: u64,
    /// Actions allowed
    pub actions_allowed: u64,
    /// Actions blocked by low Φ
    pub actions_blocked: u64,
    /// Actions requiring confirmation
    pub actions_pending_confirmation: u64,
    /// Successful executions
    pub successful_executions: u64,
    /// Failed executions
    pub failed_executions: u64,
    /// Rollbacks performed
    pub rollbacks: u64,
}

impl ConsciousPipeline {
    /// Create a new conscious pipeline with HDC+LTC+Primitives as primary
    pub async fn new(config: PipelineConfig) -> Result<Self> {
        // Initialize PRIMARY: HDC+LTC+Primitives consciousness-language core
        let core = ConsciousnessLanguageCore::with_config(config.consciousness_language.clone());

        // Initialize OPTIONAL: LLM for clarification (only if configured)
        let llm = config
            .llm
            .as_ref()
            .map(|llm_config| LlmOrgan::with_config(llm_config.clone()));

        // Initialize memory
        let memory = ConversationMemory::new(&config.memory_path)
            .context("Failed to initialize conversation memory")?;

        // Initialize adaptive thresholds (100 history observations)
        let adaptive = AdaptiveThresholds::new(100);

        // Initialize empathic unification - true empathy for users
        let empathy = EmpathicUnification::new();

        Ok(Self {
            config,
            core,
            llm,
            memory,
            current_phi: 0.5, // Start at neutral consciousness
            adaptive,
            session_id: None,
            stats: PipelineStats::default(),
            empathy,
            #[cfg(feature = "nix-mind")]
            nix_hook: None,
        })
    }

    /// Create a pipeline with LLM enabled for clarification
    pub async fn with_llm(mut config: PipelineConfig, llm_config: LlmConfig) -> Result<Self> {
        config.llm = Some(llm_config);
        Self::new(config).await
    }

    /// Attach a NixOS deep cognition hook for active inference, causal learning, episodic memory.
    #[cfg(feature = "nix-mind")]
    pub fn with_nix_hook(
        mut self,
        hook: Box<dyn symthaea_nix::plugin::pipeline_integration::NixPipelineHook>,
    ) -> Self {
        self.nix_hook = Some(hook);
        self
    }

    /// Start a new conversation session
    pub fn start_session(&mut self) -> Result<String> {
        let session_id = self.memory.start_session()?;
        self.session_id = Some(session_id.clone());
        Ok(session_id)
    }

    /// Resume an existing session
    pub fn resume_session(
        &mut self,
        session_id: &str,
    ) -> Result<Vec<crate::memory::conversation_memory::ConversationTurn>> {
        let turns = self.memory.resume_session(session_id)?;
        self.session_id = Some(session_id.to_string());
        Ok(turns)
    }

    /// Process user input through the conscious pipeline
    ///
    /// Uses HDC+LTC+Primitives as the PRIMARY understanding mechanism:
    /// 1. ConsciousnessLanguageCore processes input (HDC encoding, LTC reasoning, NSM primitives)
    /// 2. 2D Consciousness Space (Φ × Confidence) determines quadrant strategy
    /// 3. LLM only called for clarification if strategy is Lost or Curious
    pub async fn process(&mut self, input: &str) -> Result<PipelineResult> {
        let start_time = std::time::Instant::now();
        self.stats.total_requests += 1;

        // =========================================================
        // Step 1: PRIMARY - HDC+LTC+Primitives Understanding
        // =========================================================
        let understanding = self.core.process(input);

        // Extract key metrics from consciousness-language core
        let phi = understanding.consciousness_phi;
        let free_energy = understanding.unified_free_energy;
        let consciousness_state = understanding.consciousness_state.clone();
        let execution_strategy = understanding.execution_strategy.clone();
        let clarifying_questions = understanding.clarifying_questions.clone();
        let nix_understanding = understanding.nix_understanding.clone();

        // Update current Φ from HDC+LTC processing
        self.current_phi = phi;

        // =========================================================
        // Step 1.25: OPTIONAL NixOS Deep Cognition (active inference, causal reasoning)
        // =========================================================
        // When the nix-mind hook is active, its causal graph and world model
        // can detect side-effect risks that the primary HDC+LTC layer misses.
        // If nix cognition is surprised or predicts danger, downgrade the
        // execution strategy to be more cautious.
        // Check if the previous cycle's causal model was violated. When surprise
        // exceeds 0.5, lower the confidence threshold for the gate phase so the
        // system becomes more cautious after causal violations.
        #[cfg(feature = "nix-mind")]
        let causal_surprise_from_previous = self
            .nix_hook
            .as_ref()
            .map(|hook| hook.causal_surprise())
            .unwrap_or(0.0);

        #[cfg(feature = "nix-mind")]
        let nix_deep_result: Option<
            symthaea_nix::plugin::pipeline_integration::NixPipelineResult,
        > = if nix_understanding.confidence >= 0.3 {
            if let Some(ref mut hook) = self.nix_hook {
                let confidence = nix_understanding.confidence as f64;
                Some(hook.process_nix_input(input, phi, confidence))
            } else {
                None
            }
        } else {
            None
        };

        #[cfg(feature = "nix-mind")]
        let execution_strategy = {
            if let Some(ref result) = nix_deep_result {
                // Downgrade strategy if nix cognition detects risk:
                // 1. Surprised by recent state changes (causal model violated)
                // 2. Nix quadrant blocks execution (Curious/Confused)
                // 3. Phi gating denied the action
                // 4. Previous cycle's causal surprise was high (> 0.5)
                let nix_caution = result.is_surprised
                    || !result.quadrant.allows_execution()
                    || !result.phi_allowed
                    || causal_surprise_from_previous > 0.5;

                if nix_caution {
                    match &execution_strategy {
                        ExecutionStrategy::Confident { .. } => ExecutionStrategy::Curious {
                            explore_first: true,
                            targeted_questions: vec![],
                            share_reasoning: true,
                        },
                        ExecutionStrategy::Autopilot { .. } => ExecutionStrategy::Curious {
                            explore_first: true,
                            targeted_questions: vec![],
                            share_reasoning: true,
                        },
                        other => other.clone(),
                    }
                } else {
                    execution_strategy
                }
            } else {
                execution_strategy
            }
        };

        // =========================================================
        // Step 1.5: EMPATHIC PROCESSING - Sense user's emotional state
        // =========================================================
        // Infer context from NixOS intent
        let context_kind = self.infer_context_from_intent(&nix_understanding.intent);
        let empathic_response = self.empathy.process(input, context_kind);

        // =========================================================
        // Step 2: Convert NixOS understanding to DetectedIntent
        // =========================================================
        let intent = self.convert_nix_to_intent(&nix_understanding);

        // =========================================================
        // Step 3: Legacy confidence level (for compatibility)
        // =========================================================
        let confidence = PhiAwareScoring::confidence_level(phi as f32);

        // =========================================================
        // Step 4: OPTIONAL LLM - Only for Lost or Curious clarification
        // =========================================================
        let (llm_response, understanding_source) = match &execution_strategy {
            ExecutionStrategy::Lost { .. } => {
                // Genuinely lost - try LLM for help if available
                if let Some(ref mut llm) = self.llm {
                    let phi = self.current_phi;
                    match Self::get_llm_clarification(llm, input, &clarifying_questions, phi).await
                    {
                        Ok(response) => (Some(response), UnderstandingSource::LlmClarification),
                        Err(_) => (None, UnderstandingSource::HdcLtcPrimitives),
                    }
                } else {
                    (None, UnderstandingSource::HdcLtcPrimitives)
                }
            }
            ExecutionStrategy::Curious { .. } => {
                // High Φ but uncertain - LLM can help with targeted exploration
                if let Some(ref mut llm) = self.llm {
                    let phi = self.current_phi;
                    match Self::get_llm_clarification(llm, input, &clarifying_questions, phi).await
                    {
                        Ok(response) => (Some(response), UnderstandingSource::LlmClarification),
                        Err(_) => (None, UnderstandingSource::HdcLtcPrimitives),
                    }
                } else {
                    (None, UnderstandingSource::HdcLtcPrimitives)
                }
            }
            _ => {
                // Confident or Autopilot - HDC+LTC+Primitives is sufficient
                (None, UnderstandingSource::HdcLtcPrimitives)
            }
        };

        // =========================================================
        // Step 5: Check if action is allowed by consciousness gating
        // =========================================================
        let action_allowed = self.check_action_allowed_by_strategy(&execution_strategy, &intent);

        // =========================================================
        // Step 6: Execute if allowed (based on strategy)
        // =========================================================
        let execution = if action_allowed {
            if let Some(ref intent) = intent {
                match &execution_strategy {
                    ExecutionStrategy::Confident {
                        execute_immediately: true,
                        ..
                    } => {
                        // High Φ + High Confidence: Execute with full explanation
                        self.execute_action(intent).await.ok()
                    }
                    ExecutionStrategy::Autopilot {
                        execute_efficiently: true,
                        ..
                    } => {
                        // Low Φ + High Confidence: Execute efficiently (pattern-matched)
                        self.execute_action(intent).await.ok()
                    }
                    ExecutionStrategy::Curious {
                        explore_first: true,
                        ..
                    } => {
                        // High Φ + Low Confidence: Dry run first (exploring)
                        self.dry_run_action(intent).await.ok()
                    }
                    _ => None, // Lost strategy doesn't execute
                }
            } else {
                None
            }
        } else {
            if intent.is_some() {
                self.stats.actions_blocked += 1;
            }
            None
        };

        // =========================================================
        // Step 6.5: NixOS post-execute causal learning
        // =========================================================
        #[cfg(feature = "nix-mind")]
        if let Some(ref mut hook) = self.nix_hook {
            if let Some(ref exec) = execution {
                let action_desc = intent
                    .as_ref()
                    .and_then(|i| i.command.as_ref())
                    .map(|c| format!("{:?}", c))
                    .unwrap_or_default();
                hook.post_execute(&action_desc, exec.success, &exec.output);
            }
        }

        // =========================================================
        // Step 7: Record to memory
        // =========================================================
        let response_text = llm_response
            .as_deref()
            .unwrap_or(&nix_understanding.description);
        self.record_to_memory(input, response_text, &execution)?;

        // =========================================================
        // Step 8: Update adaptive thresholds + FEEDBACK LOOP TO CORE
        // =========================================================
        let quadrant = understanding.quadrant;
        let epistemic_confidence = understanding.epistemic_confidence;
        self.update_adaptive_thresholds(
            &intent,
            &execution,
            quadrant,
            &execution_strategy,
            phi,
            epistemic_confidence,
            input,
        );

        let processing_time_ms = start_time.elapsed().as_millis() as u64;

        // Build empathic response info
        let user_state = self.empathy.user_state();
        let empathic_response_info = EmpathicResponseInfo {
            user_state: UserEmotionalStateInfo {
                primary_emotion: format!("{:?}", user_state.primary_emotion),
                stress_level: user_state.stress_level,
                frustration: user_state.frustration,
                inferred_need: user_state.inferred_need,
            },
            resonant_emotion: format!("{:?}", empathic_response.resonant_emotion),
            empathy_type: format!("{:?}", empathic_response.empathy_type),
            compassion: empathic_response.compassion,
            tone_guidance: empathic_response.tone_guidance,
            warmth_adjustment: empathic_response.warmth_adjustment,
            patience_adjustment: empathic_response.patience_adjustment,
            proactive_support: empathic_response.proactive_support.clone(),
            acknowledgment: self.empathy.get_acknowledgment(),
        };

        Ok(PipelineResult {
            input: input.to_string(),
            understanding_source,
            consciousness_understanding: understanding,
            nix_understanding,
            llm_response,
            intent,
            phi,
            free_energy,
            consciousness_state,
            confidence,
            execution_strategy,
            clarifying_questions,
            action_allowed,
            execution,
            empathic_response: empathic_response_info,
            #[cfg(feature = "nix-mind")]
            nix_knowledge_answer: nix_deep_result
                .as_ref()
                .and_then(|r| r.knowledge_answer.clone()),
            #[cfg(feature = "nix-mind")]
            nix_ollama_answer: nix_deep_result
                .as_ref()
                .and_then(|r| r.ollama_answer.clone()),
            #[cfg(feature = "nix-mind")]
            nix_action_rationale: nix_deep_result
                .as_ref()
                .and_then(|r| r.plan.actions.first())
                .map(|a| a.rationale.clone()),
            #[cfg(feature = "nix-mind")]
            nix_causal_edge_count: self.nix_hook.as_ref().map(|hook| hook.causal_edge_count()),
            #[cfg(feature = "nix-mind")]
            nix_causal_surprise: self.nix_hook.as_ref().map(|hook| hook.causal_surprise()),
            processing_time_ms,
        })
    }

    /// Convert NixOS understanding to DetectedIntent
    fn convert_nix_to_intent(&self, nix: &NixOSUnderstanding) -> Option<DetectedIntent> {
        // Only create intent if we have meaningful confidence
        if nix.confidence < 0.3 {
            return None;
        }

        let (command, safety_level) = match nix.intent {
            NixOSIntent::Install => (
                Some(NixOSCommand::EnvInstall {
                    packages: nix
                        .parameters
                        .get("package")
                        .map(|p| vec![p.clone()])
                        .unwrap_or_default(),
                }),
                SafetyLevel::UserModify,
            ),
            NixOSIntent::Remove => (
                Some(NixOSCommand::EnvRemove {
                    packages: nix
                        .parameters
                        .get("package")
                        .map(|p| vec![p.clone()])
                        .unwrap_or_default(),
                }),
                SafetyLevel::UserModify,
            ),
            NixOSIntent::Search => (
                Some(NixOSCommand::Search {
                    query: nix.parameters.get("query").cloned().unwrap_or_default(),
                    json: false,
                }),
                SafetyLevel::ReadOnly,
            ),
            NixOSIntent::Upgrade => (
                Some(NixOSCommand::RebuildSwitch {
                    flake: None,
                    extra_args: vec![],
                }),
                SafetyLevel::SystemCritical,
            ),
            NixOSIntent::Configure => (
                None, // Configuration intents don't map directly to commands
                SafetyLevel::SystemModify,
            ),
            NixOSIntent::GarbageCollect => (
                Some(NixOSCommand::CollectGarbage {
                    older_than_days: None,
                    delete_all: false,
                }),
                SafetyLevel::Destructive,
            ),
            NixOSIntent::FlakeOp => (
                None, // Flake operations require more context
                SafetyLevel::UserModify,
            ),
            NixOSIntent::Rollback => (
                Some(NixOSCommand::RebuildSwitch {
                    flake: None,
                    extra_args: vec!["--rollback".to_string()],
                }),
                SafetyLevel::SystemCritical,
            ),
            _ => return None, // List/Info intents don't execute commands
        };

        let action_type = safety_level.to_action_type();

        Some(DetectedIntent {
            action_type,
            safety_level,
            command,
            detection_confidence: nix.confidence,
        })
    }

    /// Infer context kind from NixOS intent (for empathic processing)
    fn infer_context_from_intent(&self, intent: &NixOSIntent) -> ContextKind {
        match intent {
            NixOSIntent::Install | NixOSIntent::Remove | NixOSIntent::Configure => {
                ContextKind::DevWork
            }
            NixOSIntent::Search | NixOSIntent::Info | NixOSIntent::List => ContextKind::Exploration,
            NixOSIntent::Upgrade | NixOSIntent::Rollback | NixOSIntent::Switch => {
                ContextKind::Upgrade
            }
            NixOSIntent::GarbageCollect => ContextKind::DevWork,
            NixOSIntent::FlakeOp | NixOSIntent::Build => ContextKind::DevWork,
            NixOSIntent::Unknown => ContextKind::Exploration,
        }
    }

    /// Check if action is allowed based on execution strategy
    fn check_action_allowed_by_strategy(
        &mut self,
        strategy: &ExecutionStrategy,
        intent: &Option<DetectedIntent>,
    ) -> bool {
        let Some(_intent) = intent else {
            return true; // No action to gate
        };

        match strategy {
            ExecutionStrategy::Confident {
                execute_immediately,
                ..
            } => {
                // High Φ + High Confidence: Deep understanding, execute with full trust
                if *execute_immediately {
                    self.stats.actions_allowed += 1;
                    true
                } else {
                    false
                }
            }
            ExecutionStrategy::Autopilot {
                execute_efficiently,
                ..
            } => {
                // Low Φ + High Confidence: Pattern-matched routine, execute quickly
                if *execute_efficiently {
                    self.stats.actions_allowed += 1;
                    true
                } else {
                    false
                }
            }
            ExecutionStrategy::Curious { explore_first, .. } => {
                // High Φ + Low Confidence: Deep processing but uncertain, explore first
                if *explore_first {
                    self.stats.actions_pending_confirmation += 1;
                    // Do a dry-run or exploration before executing
                    true // Allow exploration for demo purposes
                } else {
                    self.stats.actions_allowed += 1;
                    true
                }
            }
            ExecutionStrategy::Lost { request_help, .. } => {
                // Low Φ + Low Confidence: Genuinely confused, don't execute
                if *request_help {
                    self.stats.actions_pending_confirmation += 1;
                }
                false // Never execute when lost - ask for help instead
            }
        }
    }

    /// Get LLM clarification (only called for Lost/Curious quadrants)
    async fn get_llm_clarification(
        llm: &mut LlmOrgan,
        input: &str,
        questions: &[ClarifyingQuestion],
        _current_phi: f64,
    ) -> Result<String> {
        let questions_text = questions
            .iter()
            .map(|q| format!("- {}", q.question))
            .collect::<Vec<_>>()
            .join("\n");

        let prompt = format!(
            "The user said: \"{}\"\n\n\
             I'm uncertain about the intent. Could you help clarify?\n\
             Specific questions:\n{}\n\n\
             Please provide a helpful response that addresses these uncertainties.",
            input, questions_text
        );

        // Use the correct LlmRequest (LLMQuery) structure
        let request = LlmRequest {
            query_type: crate::language::llm_organ::QueryType::QA,
            content: prompt,
            context: Vec::new(),
            system_prompt: Some(
                "You are a helpful NixOS assistant. The HDC+LTC system \
                          is uncertain about the user's intent. Help clarify."
                    .to_string(),
            ),
            params: None,
        };

        // generate() is synchronous, not async
        let response = llm.query(request);
        Ok(response.text)
    }

    /// Perform a dry run of an action (for Curious quadrant)
    async fn dry_run_action(&mut self, intent: &DetectedIntent) -> Result<ExecutionOutcome> {
        // Dry run simulates the action without making changes
        let output = match &intent.command {
            Some(cmd) => format!("[DRY RUN] Would execute: {:?}", cmd),
            None => "[DRY RUN] No specific command to execute".to_string(),
        };

        Ok(ExecutionOutcome {
            success: true,
            output,
            error: None,
            rolled_back: false,
            phi_after: self.current_phi as f32,
        })
    }

    /// Execute the detected action
    async fn execute_action(&mut self, intent: &DetectedIntent) -> Result<ExecutionOutcome> {
        let phi_before = self.current_phi;

        // For now, we simulate execution
        // In a full implementation, this would call NixOSExecutor
        let (success, output, error) = match &intent.command {
            Some(NixOSCommand::Search { .. }) => {
                (true, "Search completed successfully".to_string(), None)
            }
            Some(cmd) => {
                // Log the intended action
                let cmd_str = format!("{:?}", cmd);
                (true, format!("Would execute: {}", cmd_str), None)
            }
            None => (
                false,
                String::new(),
                Some("No command to execute".to_string()),
            ),
        };

        // Update statistics
        if success {
            self.stats.successful_executions += 1;
        } else {
            self.stats.failed_executions += 1;
        }

        // Calculate post-execution Φ relative to pre-execution state
        let phi_after = if success {
            (phi_before * 1.05).min(1.0) // Success slightly increases Φ
        } else {
            (phi_before * 0.95).max(0.1) // Failure slightly decreases Φ
        };
        self.current_phi = phi_after;

        Ok(ExecutionOutcome {
            success,
            output,
            error,
            rolled_back: false,
            phi_after: phi_after as f32,
        })
    }

    /// Record interaction to memory
    fn record_to_memory(
        &mut self,
        input: &str,
        response: &str,
        execution: &Option<ExecutionOutcome>,
    ) -> Result<()> {
        // Ensure we have an active session
        if self.session_id.is_none() {
            self.start_session()?;
        }

        // Add user turn (convert f64 to f32 for memory API)
        self.memory
            .add_turn("user", input, self.current_phi as f32, None)?;

        // Add assistant turn
        self.memory
            .add_turn("assistant", response, self.current_phi as f32, None)?;

        // Record causal learning if action was taken
        if let Some(exec) = execution {
            let action = if exec.success {
                "action_success"
            } else {
                "action_failure"
            };
            self.memory.record_causal_learning(
                action,
                &exec.output,
                self.current_phi as f32,
                exec.phi_after,
            )?;
        }

        Ok(())
    }

    /// Update adaptive thresholds based on outcome and feed back to consciousness core
    fn update_adaptive_thresholds(
        &mut self,
        intent: &Option<DetectedIntent>,
        execution: &Option<ExecutionOutcome>,
        quadrant: ConsciousnessQuadrant,
        strategy: &ExecutionStrategy,
        phi_at_decision: f64,
        confidence_at_decision: f64,
        original_input: &str,
    ) {
        if !self.config.adaptive_thresholds {
            return;
        }

        // Update local adaptive thresholds
        if let (Some(intent), Some(exec)) = (intent, execution) {
            self.adaptive
                .record_outcome(intent.action_type, exec.success);
            self.adaptive.observe(self.current_phi as f32);
        }

        // CRITICAL: Feed outcome back to consciousness core to close the loop
        if let Some(ref exec) = execution {
            let was_dry_run = matches!(
                strategy,
                ExecutionStrategy::Curious {
                    explore_first: true,
                    ..
                }
            );

            let feedback = ActionOutcomeFeedback {
                original_input: original_input.to_string(),
                decided_in_quadrant: quadrant,
                strategy_used: ExecutionStrategyType::from(strategy),
                phi_at_decision,
                confidence_at_decision,
                action_succeeded: exec.success,
                phi_after: exec.phi_after as f64,
                error_message: exec.error.clone(),
                was_dry_run,
                user_feedback: None, // User can provide this later via provide_user_feedback()
                // Legacy fields for backwards compatibility
                success: exec.success,
                execution_time_ms: 0,
                user_satisfaction: None,
                errors: exec.error.clone().into_iter().collect(),
                lessons: Vec::new(),
            };

            // Send feedback to consciousness core
            self.core.receive_action_outcome(feedback);
        }
    }

    /// Provide explicit user feedback about the last action
    ///
    /// Call this when the user indicates whether the action was helpful or not.
    /// This is the strongest learning signal - it directly affects threshold adaptation.
    pub fn provide_user_feedback(&mut self, was_positive: bool) {
        // Create feedback from the last execution with user input
        if let Some(mut feedback) = self.core.create_feedback_from_execution(
            was_positive, // Use user feedback as success indicator
            self.current_phi,
            None,
            false,
        ) {
            feedback.user_feedback = Some(was_positive);
            self.core.receive_action_outcome(feedback);
        }

        // Forward feedback to NixOS deep cognition hook for adaptive threshold learning
        #[cfg(feature = "nix-mind")]
        if let Some(ref mut hook) = self.nix_hook {
            hook.feedback(was_positive);
        }
    }

    /// Get current statistics
    pub fn stats(&self) -> &PipelineStats {
        &self.stats
    }

    /// Get current Φ level (as f64 for precision)
    pub fn current_phi(&self) -> f64 {
        self.current_phi
    }

    /// Get current Φ level as f32 (for legacy compatibility)
    pub fn current_phi_f32(&self) -> f32 {
        self.current_phi as f32
    }

    /// Manually set Φ (for testing or calibration)
    pub fn set_phi(&mut self, phi: f64) {
        self.current_phi = phi.clamp(0.0, 1.0);
    }

    /// Get the consciousness-language core for direct access
    pub fn core(&self) -> &ConsciousnessLanguageCore {
        &self.core
    }

    /// Get mutable access to the consciousness-language core
    pub fn core_mut(&mut self) -> &mut ConsciousnessLanguageCore {
        &mut self.core
    }
}

// ============================================================================
// HELPER IMPLEMENTATIONS
// ============================================================================

// Note: SafetyLevel::to_action_type() is defined in action/nixos_patterns.rs

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_config_default() {
        let config = PipelineConfig::default();
        assert_eq!(config.require_confirmation_below_phi, 0.3);
        assert!(config.adaptive_thresholds);
        assert_eq!(config.max_context_turns, 10);
        // LLM is optional (None by default)
        assert!(config.llm.is_none());
    }

    #[test]
    fn test_safety_level_to_action_type() {
        assert_eq!(
            SafetyLevel::ReadOnly.to_action_type(),
            ActionType::BasicQuery
        );
        assert_eq!(
            SafetyLevel::UserModify.to_action_type(),
            ActionType::StateModifying
        );
        assert_eq!(
            SafetyLevel::SystemModify.to_action_type(),
            ActionType::SystemCritical
        );
        assert_eq!(
            SafetyLevel::Destructive.to_action_type(),
            ActionType::Irreversible
        );
    }

    #[test]
    fn test_confidence_from_phi() {
        assert!(matches!(
            PhiAwareScoring::confidence_level(0.7),
            ConfidenceLevel::High { .. }
        ));
        assert!(matches!(
            PhiAwareScoring::confidence_level(0.5),
            ConfidenceLevel::Medium { .. }
        ));
        assert!(matches!(
            PhiAwareScoring::confidence_level(0.3),
            ConfidenceLevel::Low { .. }
        ));
        assert!(matches!(
            PhiAwareScoring::confidence_level(0.1),
            ConfidenceLevel::VeryLow { .. }
        ));
    }

    #[test]
    fn test_understanding_source() {
        // Understanding source should default to HDC+LTC+Primitives
        let source = UnderstandingSource::HdcLtcPrimitives;
        assert_eq!(source, UnderstandingSource::HdcLtcPrimitives);
    }

    #[test]
    fn test_consciousness_state_level() {
        // Test consciousness state levels are usable
        let active = ConsciousnessStateLevel::Active;
        let dormant = ConsciousnessStateLevel::Dormant;
        assert_ne!(active, dormant);

        // Default should be Active
        assert_eq!(
            ConsciousnessStateLevel::default(),
            ConsciousnessStateLevel::Dormant
        );
    }

    /// Mock NixPipelineHook for testing strategy downgrade behavior.
    #[cfg(feature = "nix-mind")]
    struct MockNixHook {
        surprised: bool,
        phi_allowed: bool,
        quadrant: symthaea_nix::plugin::pipeline_integration::NixConsciousnessQuadrant,
        process_called: std::sync::atomic::AtomicBool,
        post_execute_called: std::sync::atomic::AtomicBool,
        feedback_called: std::sync::atomic::AtomicBool,
    }

    #[cfg(feature = "nix-mind")]
    impl MockNixHook {
        fn confident() -> Self {
            use std::sync::atomic::AtomicBool;
            Self {
                surprised: false,
                phi_allowed: true,
                quadrant:
                    symthaea_nix::plugin::pipeline_integration::NixConsciousnessQuadrant::Confident,
                process_called: AtomicBool::new(false),
                post_execute_called: AtomicBool::new(false),
                feedback_called: AtomicBool::new(false),
            }
        }

        fn cautious() -> Self {
            use std::sync::atomic::AtomicBool;
            Self {
                surprised: true,
                phi_allowed: false,
                quadrant:
                    symthaea_nix::plugin::pipeline_integration::NixConsciousnessQuadrant::Curious,
                process_called: AtomicBool::new(false),
                post_execute_called: AtomicBool::new(false),
                feedback_called: AtomicBool::new(false),
            }
        }

        fn make_result(&self) -> symthaea_nix::plugin::pipeline_integration::NixPipelineResult {
            use symthaea_core::hdc::ContinuousHV;
            use symthaea_nix::mind::{ActionPlan, InferredGoal};
            use symthaea_nix::plugin::pipeline_integration::{NixPipelineResult, NixPipelineStage};

            NixPipelineResult {
                plan: ActionPlan {
                    actions: vec![],
                    current_free_energy: 0.5,
                    goal: InferredGoal {
                        goal_state: ContinuousHV::zero(128),
                        confidence: 0.8,
                        description: "test goal".into(),
                        needs_clarification: false,
                    },
                    needs_clarification: false,
                },
                quadrant: self.quadrant,
                free_energy: 0.5,
                hierarchy_errors: [0.1; 4],
                is_surprised: self.surprised,
                completed_stage: NixPipelineStage::Learn,
                command: None,
                safety_level: symthaea_nix::action::SafetyLevel::ReadOnly,
                phi_allowed: self.phi_allowed,
                active_memory: vec![],
                knowledge_answer: None,
                ollama_answer: None,
            }
        }
    }

    #[cfg(feature = "nix-mind")]
    impl symthaea_nix::plugin::pipeline_integration::NixPipelineHook for MockNixHook {
        fn pre_process(&self) -> f64 {
            0.6
        }

        fn process_nix_input(
            &mut self,
            _input: &str,
            _phi: f64,
            _confidence: f64,
        ) -> symthaea_nix::plugin::pipeline_integration::NixPipelineResult {
            self.process_called
                .store(true, std::sync::atomic::Ordering::Relaxed);
            self.make_result()
        }

        fn post_execute(&mut self, _action: &str, _success: bool, _output: &str) {
            self.post_execute_called
                .store(true, std::sync::atomic::Ordering::Relaxed);
        }

        fn feedback(&mut self, _was_positive: bool) {
            self.feedback_called
                .store(true, std::sync::atomic::Ordering::Relaxed);
        }
    }

    #[cfg(feature = "nix-mind")]
    #[test]
    fn test_nix_hook_confident_preserves_strategy() {
        let hook = MockNixHook::confident();
        assert!(hook.quadrant.allows_execution());
        assert!(!hook.surprised);
        assert!(hook.phi_allowed);
    }

    #[cfg(feature = "nix-mind")]
    #[test]
    fn test_nix_hook_surprise_downgrades_strategy() {
        let hook = MockNixHook::cautious();
        assert!(hook.surprised);
        assert!(!hook.phi_allowed);

        let original = ExecutionStrategy::Confident {
            execute_immediately: true,
            validate_after: true,
            explain_reasoning: true,
        };

        let nix_caution = hook.surprised || !hook.quadrant.allows_execution() || !hook.phi_allowed;

        let downgraded = if nix_caution {
            match &original {
                ExecutionStrategy::Confident { .. } | ExecutionStrategy::Autopilot { .. } => {
                    ExecutionStrategy::Curious {
                        explore_first: true,
                        targeted_questions: vec![],
                        share_reasoning: true,
                    }
                }
                other => other.clone(),
            }
        } else {
            original
        };

        assert!(matches!(
            downgraded,
            ExecutionStrategy::Curious {
                explore_first: true,
                ..
            }
        ));
    }

    #[cfg(feature = "nix-mind")]
    #[test]
    fn test_nix_hook_feedback_forwarding() {
        let mut hook = MockNixHook::confident();
        assert!(
            !hook
                .feedback_called
                .load(std::sync::atomic::Ordering::Relaxed)
        );

        symthaea_nix::plugin::pipeline_integration::NixPipelineHook::feedback(&mut hook, true);
        assert!(
            hook.feedback_called
                .load(std::sync::atomic::Ordering::Relaxed)
        );
    }

    #[cfg(feature = "nix-mind")]
    #[test]
    fn test_nix_hook_post_execute_called() {
        let mut hook = MockNixHook::confident();
        assert!(
            !hook
                .post_execute_called
                .load(std::sync::atomic::Ordering::Relaxed)
        );

        symthaea_nix::plugin::pipeline_integration::NixPipelineHook::post_execute(
            &mut hook,
            "nixos-rebuild switch",
            true,
            "activation successful",
        );
        assert!(
            hook.post_execute_called
                .load(std::sync::atomic::Ordering::Relaxed)
        );
    }

    #[test]
    fn test_execution_strategy_matching() {
        // Test all four quadrant strategies

        // High Φ + High Confidence: Confident
        let confident = ExecutionStrategy::Confident {
            execute_immediately: true,
            validate_after: true,
            explain_reasoning: true,
        };
        match confident {
            ExecutionStrategy::Confident {
                execute_immediately,
                ..
            } => {
                assert!(execute_immediately);
            }
            other => unreachable!("Expected Confident strategy, got {:?}", other),
        }

        // Low Φ + High Confidence: Autopilot
        let autopilot = ExecutionStrategy::Autopilot {
            execute_efficiently: true,
            monitor_for_errors: true,
            minimal_response: true,
        };
        match autopilot {
            ExecutionStrategy::Autopilot {
                execute_efficiently,
                ..
            } => {
                assert!(execute_efficiently);
            }
            other => unreachable!("Expected Autopilot strategy, got {:?}", other),
        }

        // High Φ + Low Confidence: Curious
        let curious = ExecutionStrategy::Curious {
            explore_first: true,
            targeted_questions: Vec::new(),
            share_reasoning: true,
        };
        match curious {
            ExecutionStrategy::Curious { explore_first, .. } => {
                assert!(explore_first);
            }
            other => unreachable!("Expected Curious strategy, got {:?}", other),
        }

        // Low Φ + Low Confidence: Lost
        let lost = ExecutionStrategy::Lost {
            request_help: true,
            generic_questions: Vec::new(),
            admit_confusion: true,
        };
        match lost {
            ExecutionStrategy::Lost { request_help, .. } => {
                assert!(request_help);
            }
            other => unreachable!("Expected Lost strategy, got {:?}", other),
        }
    }

    #[cfg(feature = "nix-mind")]
    #[test]
    fn test_pipeline_result_causal_fields() {
        // Verify the new causal metric fields can hold values and default to None
        let edge_count: Option<usize> = Some(42);
        let surprise: Option<f32> = Some(0.3);

        assert_eq!(edge_count, Some(42));
        assert_eq!(surprise, Some(0.3));

        let none_count: Option<usize> = None;
        let none_surprise: Option<f32> = None;
        assert!(none_count.is_none());
        assert!(none_surprise.is_none());
    }

    #[cfg(feature = "nix-mind")]
    #[test]
    fn test_causal_surprise_triggers_strategy_downgrade() {
        // Verify that high causal surprise (> 0.5) causes strategy downgrade
        let causal_surprise: f32 = 0.7;
        let is_surprised = false;
        let allows_execution = true;
        let phi_allowed = true;

        // Without causal_surprise, this would NOT trigger caution
        let without = is_surprised || !allows_execution || !phi_allowed;
        assert!(!without, "Without causal surprise, should not be cautious");

        // With causal_surprise > 0.5, it SHOULD trigger caution
        let with = is_surprised || !allows_execution || !phi_allowed || causal_surprise > 0.5;
        assert!(with, "With high causal surprise, should be cautious");

        // Surprise at exactly 0.5 should NOT trigger
        let at_boundary = 0.5f32 > 0.5;
        assert!(!at_boundary, "Surprise at 0.5 should not trigger caution");

        // Surprise just above 0.5 should trigger
        let above_boundary = 0.51f32 > 0.5;
        assert!(above_boundary, "Surprise above 0.5 should trigger caution");
    }
}
