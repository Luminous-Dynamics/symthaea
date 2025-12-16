// Resonant Telemetry (v0.3)
//
// Observability for the Voice Cortex: capture events to understand behavior
// and tune templates/modes/thresholds.
//
// Philosophy: The Voice Cortex needs a "mirror" to see itself.

use crate::resonant_speech::{
    ResonantContext, ResonantUtterance, RelationshipMode, TemporalFrame,
    CognitiveLoad, SuggestionDecisionKind,
};
use serde::{Serialize, Deserialize};
use std::time::{SystemTime, UNIX_EPOCH};

/// Event capturing a resonant utterance composition
///
/// This struct records what the Voice Cortex did and why,
/// enabling analysis and improvement of speech patterns.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResonantEvent {
    /// Timestamp (Unix epoch milliseconds)
    pub timestamp: u64,

    /// Relationship mode used
    pub relationship_mode: String,

    /// Temporal frame used
    pub temporal_frame: String,

    /// Cognitive load level
    pub cognitive_load: String,

    /// Trust in Sophia (0.0 to 1.0)
    pub trust_in_sophia: f32,

    /// Was flat mode active?
    pub flat_mode: bool,

    /// Suggestion decision kind
    pub suggestion_decision: String,

    /// Arc name (if any)
    pub arc_name: Option<String>,

    /// Arc delta (if any)
    pub arc_delta: Option<f32>,

    /// K-Index deltas included
    pub k_deltas_count: usize,

    /// Tags applied to the utterance
    pub tags: Vec<String>,

    /// Utterance length (characters)
    pub utterance_length: usize,
}

impl ResonantEvent {
    /// Create event from ResonantContext and utterance
    pub fn from_context(ctx: &ResonantContext, utterance: &ResonantUtterance) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        let relationship_mode = format!("{:?}", ctx.relationship.primary);
        let temporal_frame = format!("{:?}", ctx.temporal_frame);
        let cognitive_load = format!("{:?}", ctx.user_state.cognitive_load);

        let suggestion_decision = match ctx.suggestion_decision.kind {
            SuggestionDecisionKind::AutoApply => "AutoApply",
            SuggestionDecisionKind::AskUser => "AskUser",
            SuggestionDecisionKind::Reject => "Reject",
        }.to_string();

        Self {
            timestamp,
            relationship_mode,
            temporal_frame,
            cognitive_load,
            trust_in_sophia: ctx.user_state.trust_in_sophia,
            flat_mode: ctx.user_state.flat_mode,
            suggestion_decision,
            arc_name: ctx.arc_name.clone(),
            arc_delta: ctx.arc_delta,
            k_deltas_count: ctx.k_deltas.len(),
            tags: utterance.tags.clone(),
            utterance_length: utterance.text.len(),
        }
    }

    /// Log event to JSON (for analysis)
    ///
    /// In production, this could send to a telemetry backend,
    /// write to a file, or store in SQLite for local analysis.
    pub fn log_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|_| "{}".to_string())
    }

    /// Human-readable summary
    pub fn summary(&self) -> String {
        format!(
            "[{}] {} + {} (load: {}, trust: {:.2}, flat: {}) → {} chars, {} K-deltas",
            self.timestamp,
            self.relationship_mode,
            self.temporal_frame,
            self.cognitive_load,
            self.trust_in_sophia,
            self.flat_mode,
            self.utterance_length,
            self.k_deltas_count
        )
    }
}

/// Enhanced utterance composition with telemetry
///
/// This wraps the normal composition to capture an event for observability.
pub fn compose_utterance_with_event(
    ctx: &ResonantContext,
    engine: &crate::resonant_speech::SimpleResonantEngine,
) -> (ResonantUtterance, ResonantEvent) {
    // Compose utterance normally
    let utterance = engine.compose_utterance(ctx);

    // Capture event
    let event = ResonantEvent::from_context(ctx, &utterance);

    // Log event (in production, this would go to telemetry backend)
    tracing::info!("Resonant event: {}", event.summary());
    tracing::debug!("Resonant event JSON: {}", event.log_json());

    (utterance, event)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resonant_speech::{
        UserState, RelationshipProfile, SuggestionDecision, OutputChannel,
        UtteranceComponents,
    };

    fn mock_context() -> ResonantContext {
        ResonantContext {
            user_state: UserState {
                cognitive_load: CognitiveLoad::Low,
                trust_in_sophia: 0.85,
                locale: "en-US".to_string(),
                flat_mode: false,
            },
            relationship: RelationshipProfile {
                primary: RelationshipMode::CoAuthor,
                secondary: None,
                weight_primary: 1.0,
            },
            temporal_frame: TemporalFrame::Meso,
            suggestion_decision: SuggestionDecision {
                kind: SuggestionDecisionKind::AskUser,
                reason: "Test".to_string(),
                claim_id: None,
            },
            channel: OutputChannel::TextUI,
            action_summary: "Test action".to_string(),
            reason_short: "Test reason".to_string(),
            trust_label: "High".to_string(),
            reversible_statement: "Reversible".to_string(),
            arc_name: Some("Test Arc".to_string()),
            arc_delta: Some(0.15),
            timeframe: Some("Past7Days".to_string()),
            controversy_note: None,
            k_deltas: vec![],
        }
    }

    fn mock_utterance() -> ResonantUtterance {
        ResonantUtterance {
            text: "Test utterance".to_string(),
            title: Some("Test".to_string()),
            components: UtteranceComponents {
                what: "What".to_string(),
                why: "Why".to_string(),
                certainty: "Certain".to_string(),
                tradeoffs: "Tradeoffs".to_string(),
            },
            tags: vec!["test".to_string()],
        }
    }

    #[test]
    fn test_event_creation() {
        let ctx = mock_context();
        let utterance = mock_utterance();

        let event = ResonantEvent::from_context(&ctx, &utterance);

        assert_eq!(event.relationship_mode, "CoAuthor");
        assert_eq!(event.temporal_frame, "Meso");
        assert_eq!(event.cognitive_load, "Low");
        assert_eq!(event.trust_in_sophia, 0.85);
        assert_eq!(event.flat_mode, false);
        assert_eq!(event.k_deltas_count, 0);
        assert_eq!(event.utterance_length, "Test utterance".len());
    }

    #[test]
    fn test_event_json() {
        let ctx = mock_context();
        let utterance = mock_utterance();
        let event = ResonantEvent::from_context(&ctx, &utterance);

        let json = event.log_json();
        assert!(json.contains("CoAuthor"));
        assert!(json.contains("Meso"));
    }

    #[test]
    fn test_event_summary() {
        let ctx = mock_context();
        let utterance = mock_utterance();
        let event = ResonantEvent::from_context(&ctx, &utterance);

        let summary = event.summary();
        assert!(summary.contains("CoAuthor"));
        assert!(summary.contains("Meso"));
        assert!(summary.contains("0.85"));
    }

    #[test]
    fn test_compose_with_event() {
        let ctx = mock_context();
        let engine = crate::resonant_speech::SimpleResonantEngine::new();

        let (utterance, event) = compose_utterance_with_event(&ctx, &engine);

        assert!(!utterance.text.is_empty());
        assert_eq!(event.relationship_mode, "CoAuthor");
    }
}
