// Resonant Interaction - Bridge between SwarmAdvisor and ResonantSpeech
//
// This module wires together:
// - SwarmAdvisor (epistemic/trust logic)
// - UserStateInference (context awareness)
// - ResonantSpeechEngine (voice logic)
//
// Philosophy: Keep concerns separate
// - SwarmAdvisor doesn't know about "voice"
// - ResonantEngine doesn't know about swarm evaluation
// - This adapter coordinates both

use crate::resonant_speech::{
    ResonantContext, ResonantUtterance, RelationshipProfile, RelationshipMode,
    TemporalFrame, OutputChannel, SimpleResonantEngine,
};
use crate::user_state_inference::{
    UserStateInference, ContextKind, ExplorationMode, TimePressure,
};
use crate::swarm::{SwarmIntelligence, SwarmError};
use crate::kindex_client::KIndexClient;

/// Complete interaction context combining all inference sources
#[derive(Debug, Clone)]
pub struct InteractionContext {
    /// What kind of context we're in
    pub context_kind: ContextKind,

    /// User's current state (inferred or explicit)
    pub user_state_inference: UserStateInference,

    /// Relationship mode (can be set explicitly or inferred)
    pub relationship: RelationshipProfile,

    /// Temporal frame for this interaction
    pub temporal_frame: TemporalFrame,

    /// Output channel
    pub channel: OutputChannel,

    /// Optional urgency hint from user's actual input
    pub urgency_hint: Option<String>,

    /// User's locale
    pub locale: String,

    /// Flat mode override - disable narrative, just facts
    pub flat_mode: bool,

    /// Macro reflection enabled (K-Index arcs)
    pub macro_enabled: bool,
}

impl InteractionContext {
    /// Create a new context with defaults
    pub fn new(context_kind: ContextKind) -> Self {
        Self {
            context_kind,
            user_state_inference: UserStateInference::new(),
            relationship: RelationshipProfile::default(),
            temporal_frame: Self::infer_temporal_frame(context_kind),
            channel: OutputChannel::TextUI,
            urgency_hint: None,
            locale: "en-US".to_string(),
            flat_mode: false,
            macro_enabled: true,  // K-Index arcs enabled by default
        }
    }

    /// Infer temporal frame from context kind
    fn infer_temporal_frame(context_kind: ContextKind) -> TemporalFrame {
        match context_kind {
            ContextKind::ErrorHandling => TemporalFrame::Micro,
            ContextKind::DevWork | ContextKind::Writing | ContextKind::Exploration => {
                TemporalFrame::Meso
            }
            ContextKind::Review | ContextKind::Planning => TemporalFrame::Macro,
        }
    }

    /// Set relationship mode explicitly (e.g., from user command "/mode tech")
    pub fn set_mode(&mut self, mode: RelationshipMode) {
        self.relationship.primary = mode;
        self.relationship.secondary = None;
        self.relationship.weight_primary = 1.0;
    }

    /// Set mode blend (e.g., 80% CoAuthor, 20% Coach)
    pub fn set_mode_blend(&mut self, primary: RelationshipMode, secondary: RelationshipMode, weight: f32) {
        self.relationship.primary = primary;
        self.relationship.secondary = Some(secondary);
        self.relationship.weight_primary = weight;
    }

    /// Extract urgency hint from user query
    pub fn extract_urgency_hint(&mut self, query: &str) {
        let lower = query.to_lowercase();

        // Check for urgency markers
        if lower.contains("urgent") || lower.contains("now") ||
           lower.contains("asap") || lower.contains("help") {
            self.urgency_hint = Some(query.to_string());
        }

        // Check for calm/curious markers
        if lower.contains("curious") || lower.contains("explore") ||
           lower.contains("what do you think") {
            self.urgency_hint = Some(query.to_string());
        }
    }
}

impl Default for InteractionContext {
    fn default() -> Self {
        Self::new(ContextKind::DevWork)
    }
}

impl Default for RelationshipProfile {
    fn default() -> Self {
        Self {
            primary: RelationshipMode::CoAuthor,
            secondary: None,
            weight_primary: 1.0,
        }
    }
}

/// Suggestion decision from SwarmAdvisor
///
/// This is a placeholder for the actual SwarmAdvisor decision type.
/// In real integration, this would come from swarm.rs
#[derive(Debug, Clone)]
pub struct SuggestionDecision {
    pub kind: SuggestionDecisionKind,
    pub reason: String,
    pub claim_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SuggestionDecisionKind {
    AutoApply,
    AskUser,
    Reject,
}

/// Evaluated claim from SwarmAdvisor
///
/// This is a placeholder for the actual evaluation result.
/// In real integration, this would include MATL scores, E/N/M axes, etc.
#[derive(Debug, Clone)]
pub struct EvaluatedClaim {
    pub action_summary: String,
    pub reason_short: String,
    pub trust_label: String,
    pub reversible_statement: String,
    pub arc_name: Option<String>,
    pub arc_delta: Option<f32>,
    pub timeframe: Option<String>,
    pub controversy_note: Option<String>,
}

/// Enrich resonant context with K-Index deltas (v0.3)
///
/// Only populates k_deltas for Macro temporal frames and non-urgent contexts.
/// This is where the Voice Cortex asks the K-Index engine what changed.
pub fn enrich_with_kindex(
    ctx: &mut ResonantContext,
    kindex: &impl KIndexClient,
) {
    // Only for macro-ish contexts (not micro/urgent)
    if !matches!(ctx.temporal_frame, TemporalFrame::Macro) {
        return;
    }

    // Pull Knowledge + Governance deltas for the last week
    let mut deltas: Vec<crate::kindex_client::KDelta> = Vec::new();

    if let Some(k) = kindex.get_delta("Knowledge", "Past7Days") {
        deltas.push(k);
    }

    if let Some(g) = kindex.get_delta("Governance", "Past7Days") {
        deltas.push(g);
    }

    ctx.k_deltas = deltas;
}

/// Advise and speak - the main integration function
///
/// This takes a query, gets a decision from SwarmAdvisor,
/// infers user state, and generates a resonant utterance.
pub fn advise_and_speak(
    query: &str,
    context: &mut InteractionContext,
    engine: &SimpleResonantEngine,
) -> Result<ResonantUtterance, SwarmError> {
    // 1. Extract urgency from query
    context.extract_urgency_hint(query);

    // 2. Infer user state from context + recent events
    let mut user_state = context.user_state_inference.infer(
        context.context_kind,
        &context.locale,
    );

    // Override flat_mode if set explicitly
    user_state.flat_mode = context.flat_mode;

    // 3. Get decision from SwarmAdvisor (placeholder - real integration would call actual swarm)
    // For now, we'll use a mock decision
    let (eval, decision) = mock_advisor_decision(query, context.context_kind);

    // 4. Build ResonantContext
    let resonant_ctx = ResonantContext {
        user_state: user_state.clone(),
        relationship: context.relationship.clone(),
        temporal_frame: context.temporal_frame,
        suggestion_decision: crate::resonant_speech::SuggestionDecision {
            kind: match decision.kind {
                SuggestionDecisionKind::AutoApply =>
                    crate::resonant_speech::SuggestionDecisionKind::AutoApply,
                SuggestionDecisionKind::AskUser =>
                    crate::resonant_speech::SuggestionDecisionKind::AskUser,
                SuggestionDecisionKind::Reject =>
                    crate::resonant_speech::SuggestionDecisionKind::Reject,
            },
            reason: decision.reason,
            claim_id: decision.claim_id.and_then(|s| uuid::Uuid::parse_str(&s).ok()),
        },
        channel: context.channel,
        action_summary: eval.action_summary,
        reason_short: eval.reason_short,
        trust_label: eval.trust_label,
        reversible_statement: eval.reversible_statement,
        arc_name: eval.arc_name,
        arc_delta: eval.arc_delta,
        timeframe: eval.timeframe,
        controversy_note: eval.controversy_note,
        k_deltas: vec![],  // v0.3: Will be populated by enrich_with_kindex()
    };

    // 5. Generate resonant utterance
    Ok(engine.compose_utterance(&resonant_ctx))
}

/// Mock advisor decision (placeholder for real SwarmAdvisor integration)
fn mock_advisor_decision(
    query: &str,
    context_kind: ContextKind,
) -> (EvaluatedClaim, SuggestionDecision) {
    let eval = EvaluatedClaim {
        action_summary: format!("Process: {}", query),
        reason_short: "This is a reasonable next step based on your context".to_string(),
        trust_label: "Medium (0.75, E2, N1, M2)".to_string(),
        reversible_statement: "This action is reversible".to_string(),
        arc_name: None,
        arc_delta: None,
        timeframe: None,
        controversy_note: None,
    };

    let decision = SuggestionDecision {
        kind: match context_kind {
            ContextKind::ErrorHandling => SuggestionDecisionKind::AutoApply,
            _ => SuggestionDecisionKind::AskUser,
        },
        reason: "Standard decision based on context".to_string(),
        claim_id: None,
    };

    (eval, decision)
}

/// Command handler for explicit mode switching
pub fn handle_mode_command(context: &mut InteractionContext, command: &str) -> Option<String> {
    let cmd_lower = command.to_lowercase();

    if cmd_lower.starts_with("/mode ") {
        let mode_str = &cmd_lower[6..].trim();

        let mode = match *mode_str {
            "tech" | "technician" => RelationshipMode::Technician,
            "coauthor" | "co-author" => RelationshipMode::CoAuthor,
            "coach" => RelationshipMode::Coach,
            "witness" => RelationshipMode::Witness,
            "ritual" => RelationshipMode::Ritual,
            _ => return Some(format!("Unknown mode: {}. Try tech, coauthor, coach, witness, or ritual.", mode_str)),
        };

        context.set_mode(mode);
        return Some(format!("Mode set to {:?}", mode));
    }

    if cmd_lower == "/mode flat" {
        // Flat mode: just the facts, minimal resonance
        context.flat_mode = true;
        return Some("Flat mode enabled. I'll stick to minimal factual responses.".to_string());
    }

    if cmd_lower == "/mode normal" {
        // Return to normal resonant speech
        context.flat_mode = false;
        return Some("Normal mode enabled. I'll speak naturally.".to_string());
    }

    if cmd_lower == "/macro off" {
        // Disable K-Index arc reflections
        context.macro_enabled = false;
        return Some("Macro reflections disabled. I'll skip K-Index arcs.".to_string());
    }

    if cmd_lower == "/macro on" {
        // Enable K-Index arc reflections
        context.macro_enabled = true;
        return Some("Macro reflections enabled. I'll include K-Index arcs when appropriate.".to_string());
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_creation() {
        let ctx = InteractionContext::new(ContextKind::ErrorHandling);
        assert_eq!(ctx.temporal_frame, TemporalFrame::Micro);
    }

    #[test]
    fn test_mode_switching() {
        let mut ctx = InteractionContext::default();
        ctx.set_mode(RelationshipMode::Coach);

        assert_eq!(ctx.relationship.primary, RelationshipMode::Coach);
        assert_eq!(ctx.relationship.weight_primary, 1.0);
    }

    #[test]
    fn test_mode_command_handler() {
        let mut ctx = InteractionContext::default();
        let result = handle_mode_command(&mut ctx, "/mode tech");

        assert!(result.is_some());
        assert_eq!(ctx.relationship.primary, RelationshipMode::Technician);
    }

    #[test]
    fn test_urgency_extraction() {
        let mut ctx = InteractionContext::default();
        ctx.extract_urgency_hint("urgent: fix this now!");

        assert!(ctx.urgency_hint.is_some());
    }

    #[test]
    fn test_advise_and_speak() {
        let mut ctx = InteractionContext::new(ContextKind::DevWork);
        let engine = SimpleResonantEngine::new();

        let utterance = advise_and_speak("install firefox", &mut ctx, &engine).unwrap();

        assert!(!utterance.text.is_empty());
    }
}
