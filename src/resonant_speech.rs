// Resonant Speech Engine - v0.1 Prototype
//
// Implements the Voice Cortex layer that transforms grounded decisions
// into resonant, contextual utterances tuned to:
// - User state (cognitive load, trust, locale)
// - Relationship mode (Technician, Co-author, Coach, Witness, Ritual)
// - Temporal frame (Micro, Meso, Macro)
//
// Design: RESONANT_SPEECH_PROTOCOL.md v0.1

use std::collections::HashMap;
use uuid::Uuid;

// ============================================================================
// Core Types
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CognitiveLoad {
    Low,
    Medium,
    High,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TemporalFrame {
    Micro, // Next action (seconds–minutes)
    Meso,  // Current arc (hours–weeks)
    Macro, // Life/project arc (months–years)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RelationshipMode {
    Technician, // "Fix the thing; minimal context."
    CoAuthor,   // "We're building something together."
    Coach,      // "I help you grow/learn."
    Witness,    // "I reflect back; no fixing."
    Ritual,     // "Ceremonial / arc-oriented reflection."
}

#[derive(Debug, Clone)]
pub struct RelationshipProfile {
    pub primary: RelationshipMode,
    pub secondary: Option<RelationshipMode>,
    pub weight_primary: f32,
}

#[derive(Debug, Clone)]
pub struct UserState {
    pub cognitive_load: CognitiveLoad,
    pub trust_in_sophia: f32,
    pub locale: String,
    pub flat_mode: bool,  // Epistemic safe mode: just facts, no narrative
}

#[derive(Debug, Clone, Copy)]
pub enum SuggestionDecisionKind {
    AutoApply,
    AskUser,
    Reject,
}

#[derive(Debug, Clone)]
pub struct SuggestionDecision {
    pub kind: SuggestionDecisionKind,
    pub reason: String,
    pub claim_id: Option<Uuid>,
}

#[derive(Debug, Clone)]
pub struct UtteranceComponents {
    pub what: String,
    pub why: String,
    pub certainty: String,
    pub tradeoffs: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputChannel {
    TextUI,
    Terminal,
    Voice,
    Notification,
}

#[derive(Debug, Clone)]
pub struct ResonantContext {
    pub user_state: UserState,
    pub relationship: RelationshipProfile,
    pub temporal_frame: TemporalFrame,
    pub suggestion_decision: SuggestionDecision,
    pub channel: OutputChannel,

    // Content fields
    pub action_summary: String,
    pub reason_short: String,
    pub trust_label: String,
    pub reversible_statement: String,
    pub arc_name: Option<String>,
    pub arc_delta: Option<f32>,
    pub timeframe: Option<String>,
    pub controversy_note: Option<String>,

    // v0.3: K-Index integration
    pub k_deltas: Vec<crate::kindex_client::KDelta>,
}

#[derive(Debug, Clone)]
pub struct ResonantUtterance {
    pub text: String,
    pub title: Option<String>,
    pub components: UtteranceComponents,
    pub tags: Vec<String>,
}

// ============================================================================
// Template System
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TemplateKey {
    pub mode: RelationshipMode,
    pub frame: TemporalFrame,
    pub load: CognitiveLoad,
}

impl TemplateKey {
    pub fn new(mode: RelationshipMode, frame: TemporalFrame, load: CognitiveLoad) -> Self {
        Self { mode, frame, load }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TemplateId {
    TechnicianMicroHigh, // urgent fix
    CoAuthorMesoMedium,  // project work
    CoachMacroLow,       // reflection
    ControversialAny,    // controversial / low trust
    GovernanceAny,       // governance / high-stakes
}

#[derive(Debug, Clone)]
pub struct Template {
    pub id: TemplateId,
}

impl Template {
    pub fn render(&self, ctx: &ResonantContext) -> ResonantUtterance {
        match self.id {
            TemplateId::TechnicianMicroHigh => render_technician_micro_high(ctx),
            TemplateId::CoAuthorMesoMedium => render_coauthor_meso_medium(ctx),
            TemplateId::CoachMacroLow => render_coach_macro_low(ctx),
            TemplateId::ControversialAny => render_controversial(ctx),
            TemplateId::GovernanceAny => render_governance(ctx),
        }
    }
}

// ============================================================================
// Template Renderers
// ============================================================================

/// Template 1: Technician + Micro + High load (urgent fix)
fn render_technician_micro_high(ctx: &ResonantContext) -> ResonantUtterance {
    let what = format!("Fix: {}", ctx.action_summary);
    let why = format!("Why: {}", ctx.reason_short);
    let certainty = format!("Trust: {}", ctx.trust_label);
    let tradeoffs = ctx.reversible_statement.clone();

    let text = format!(
        "{}\n\n{}\n\n{}\n\n{}\n[Details ↓]",
        what, why, certainty, tradeoffs
    );

    ResonantUtterance {
        text,
        title: None,
        components: UtteranceComponents {
            what,
            why,
            certainty,
            tradeoffs,
        },
        tags: vec!["urgent".into(), "technician".into()],
    }
}

/// Template 2: CoAuthor + Meso + Medium load (project work)
fn render_coauthor_meso_medium(ctx: &ResonantContext) -> ResonantUtterance {
    let arc = ctx
        .arc_name
        .clone()
        .unwrap_or_else(|| "this work".into());
    let timeframe = ctx.timeframe.clone().unwrap_or_else(|| "recently".into());

    let what = format!("Next step for {}: {}", arc, ctx.action_summary);

    let why = format!("Why this step: {}", ctx.reason_short);

    let delta_str = ctx.arc_delta.map_or("moves you forward".to_string(), |d| {
        format!("moves {} by approximately {:+.2}", arc, d)
    });

    let certainty = format!("Trust: {}", ctx.trust_label);

    let tradeoffs = "Tradeoffs: time and focus invested here are not spent on alternative tasks; we can revisit if priorities shift.".to_string();

    let text = format!(
        "Here's what I suggest next for {}:\n\n→ {}\n\n{}\n\nThis {}.\n{}\n\n{}\n\nSound good?",
        arc, ctx.action_summary, why, delta_str, certainty, tradeoffs
    );

    ResonantUtterance {
        text,
        title: Some(format!("Next step for {}", arc)),
        components: UtteranceComponents {
            what,
            why,
            certainty,
            tradeoffs,
        },
        tags: vec!["coauthor".into(), "meso".into()],
    }
}

/// Template 3: Coach + Macro + Low load (reflection)
/// v0.3: Now uses K-Index deltas for multi-dimensional awareness
fn render_coach_macro_low(ctx: &ResonantContext) -> ResonantUtterance {
    // v0.3: Use K-Index deltas if available
    let (main_line, drivers_line) = if let Some(best) = ctx.k_deltas.iter()
        .max_by(|a, b| a.delta.abs().partial_cmp(&b.delta.abs()).unwrap_or(std::cmp::Ordering::Equal))
    {
        let main = format!(
            "Over {}, your {} actualization shifted by {:+.2}.",
            best.timeframe, best.dimension, best.delta
        );

        let drivers = if !best.drivers.is_empty() {
            format!("Most of that movement came from: {}.", best.drivers.join(", "))
        } else {
            String::new()
        };

        (main, drivers)
    } else {
        // Fallback to legacy arc_name/arc_delta if no K-Index deltas
        let timeframe = ctx
            .timeframe
            .clone()
            .unwrap_or_else(|| "this period".into());
        let arc = ctx
            .arc_name
            .clone()
            .unwrap_or_else(|| "your current arc".into());
        let delta_str = ctx.arc_delta.map_or(
            "shifted in a subtle way".to_string(),
            |d| format!("shifted by approximately {:+.2}", d),
        );

        (
            format!("Over {}, your {} has {}.", timeframe, arc, delta_str),
            String::new()
        )
    };

    let what = format!("Zooming out: {}", main_line);

    let why = if !drivers_line.is_empty() {
        format!("Why this matters: {}. This reflects where your attention and effort have been flowing, not just what you planned.", drivers_line)
    } else {
        "Why this matters: it reflects where your attention and effort have been flowing, not just what you planned.".to_string()
    };

    let certainty = "Certainty: moderate; based on observed activity and recent work patterns. You can always correct me.".to_string();

    let arc_name = ctx.arc_name.clone().unwrap_or_else(|| "this dimension".into());
    let tradeoffs = format!(
        "Tradeoffs: focusing more on {} may mean less investment in other dimensions; we can rebalance if this doesn't feel right.",
        arc_name
    );

    let text = format!(
        "Zooming out for a moment.\n\n{}\n\n{}\n\n{}\n\n{}\n\nDoes this feel aligned with how you want this arc to grow right now?",
        what, why, certainty, tradeoffs
    );

    ResonantUtterance {
        text,
        title: Some(format!("Reflection on {}", arc_name)),
        components: UtteranceComponents {
            what,
            why,
            certainty,
            tradeoffs,
        },
        tags: vec!["coach".into(), "macro".into(), "reflection".into()],
    }
}

/// Template 4: Controversial / low trust (any mode)
fn render_controversial(ctx: &ResonantContext) -> ResonantUtterance {
    let controversy = ctx
        .controversy_note
        .clone()
        .unwrap_or_else(|| "The swarm has mixed evidence on this.".into());

    let what = format!("Possible suggestion: {}", ctx.action_summary);
    let why = format!("Epistemic status: {}\n{}", ctx.trust_label, controversy);

    let certainty = "Certainty: low to medium; there are both supporting and refuting patterns.".to_string();

    let tradeoffs = "Tradeoffs: applying this may help, but it could also introduce regressions. Safer alternatives or deeper inspection are recommended before you commit."
        .to_string();

    let text = format!(
        "I have a possible suggestion, but it's contentious in the swarm.\n\n→ {}\n\n{}\n\n{}\n\n{}\n\nGiven this, I don't recommend auto-applying. We can:\n- Explore safer alternatives, or\n- Inspect the conflicting evidence together.\n\nWhat would you prefer?",
        ctx.action_summary, why, certainty, tradeoffs
    );

    ResonantUtterance {
        text,
        title: Some("Contentious suggestion".into()),
        components: UtteranceComponents {
            what,
            why,
            certainty,
            tradeoffs,
        },
        tags: vec!["controversial".into(), "caution".into()],
    }
}

/// Template 5: Governance / high-stakes (any mode)
fn render_governance(ctx: &ResonantContext) -> ResonantUtterance {
    let what = format!("Governance-relevant action: {}", ctx.action_summary);

    let why = format!(
        "Grounding: {}\nI'm an Instrumental Actor; I can't decide this, but I can help you reason about it.",
        ctx.reason_short
    );

    let certainty = format!(
        "Trust: {} (this reflects swarm-level assessments, not a guarantee).",
        ctx.trust_label
    );

    let tradeoffs =
        "Tradeoffs: this may affect shared resources, other members, or long-term governance paths. It should be weighed against the relevant charters and current mandates."
            .to_string();

    let text = format!(
        "This action affects shared resources or other members.\n\n{}\n\n{}\n\n{}\n\n{}\n\nI can help you:\n- Simulate possible outcomes,\n- Draft a justification,\n- Or surface this to the relevant council.\n\nWhat support do you want from me here?",
        what, why, certainty, tradeoffs
    );

    ResonantUtterance {
        text,
        title: Some("Governance decision support".into()),
        components: UtteranceComponents {
            what,
            why,
            certainty,
            tradeoffs,
        },
        tags: vec!["governance".into(), "high-stakes".into()],
    }
}

// ============================================================================
// Resonant Speech Engine
// ============================================================================

pub struct SimpleResonantEngine {
    templates: HashMap<TemplateKey, Template>,
}

impl SimpleResonantEngine {
    pub fn new() -> Self {
        let mut templates = HashMap::new();

        // 1. Technician + Micro + High load
        templates.insert(
            TemplateKey::new(
                RelationshipMode::Technician,
                TemporalFrame::Micro,
                CognitiveLoad::High,
            ),
            Template {
                id: TemplateId::TechnicianMicroHigh,
            },
        );

        // 2. CoAuthor + Meso + Medium load
        templates.insert(
            TemplateKey::new(
                RelationshipMode::CoAuthor,
                TemporalFrame::Meso,
                CognitiveLoad::Medium,
            ),
            Template {
                id: TemplateId::CoAuthorMesoMedium,
            },
        );

        // 3. Coach + Macro + Low load
        templates.insert(
            TemplateKey::new(
                RelationshipMode::Coach,
                TemporalFrame::Macro,
                CognitiveLoad::Low,
            ),
            Template {
                id: TemplateId::CoachMacroLow,
            },
        );

        Self { templates }
    }

    fn select_template(&self, ctx: &ResonantContext) -> Template {
        // Governance and controversial suggestions override normal routing
        if matches!(ctx.suggestion_decision.kind, SuggestionDecisionKind::Reject)
            && ctx.controversy_note.is_some()
        {
            return Template {
                id: TemplateId::ControversialAny,
            };
        }

        // v0.1 heuristic: use governance template if arc_name hints at governance
        if let Some(arc) = &ctx.arc_name {
            if arc.to_lowercase().contains("governance") || arc.to_lowercase().contains("dao") {
                return Template {
                    id: TemplateId::GovernanceAny,
                };
            }
        }

        let key = TemplateKey::new(
            ctx.relationship.primary,
            ctx.temporal_frame,
            ctx.user_state.cognitive_load,
        );

        self.templates.get(&key).cloned().unwrap_or(Template {
            id: TemplateId::TechnicianMicroHigh, // fallback
        })
    }

    pub fn compose_utterance(&self, ctx: &ResonantContext) -> ResonantUtterance {
        // Flat mode override: epistemic safe mode
        if ctx.user_state.flat_mode || ctx.user_state.trust_in_sophia < 0.3 {
            return minimal_factual_response(ctx);
        }

        let template = self.select_template(ctx);
        let mut utterance = template.render(ctx);

        // Channel-specific tweaks
        if matches!(ctx.channel, OutputChannel::Voice) {
            // For voice, keep tradeoffs shorter
            utterance.components.tradeoffs =
                "There are some tradeoffs; I can explain if you'd like.".to_string();
        }

        utterance
    }
}

impl Default for SimpleResonantEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Flat Mode / Epistemic Safe Mode
// ============================================================================

/// Minimal factual response - no narrative, just structured facts
///
/// Triggered when:
/// - user_state.flat_mode == true
/// - trust_in_sophia < 0.3 (automatic safety override)
///
/// Format:
/// Action: ...
/// Why: ...
/// Confidence: ...
/// Risk / Tradeoffs: ...
fn minimal_factual_response(ctx: &ResonantContext) -> ResonantUtterance {
    let action = format!("Action: {}", ctx.action_summary);
    let why = format!("Why: {}", ctx.reason_short);
    let confidence = format!("Confidence: {}", ctx.trust_label);
    let risk = format!("Risk / Tradeoffs: {}", ctx.reversible_statement);

    let text = format!("{}\n{}\n{}\n{}", action, why, confidence, risk);

    ResonantUtterance {
        text,
        title: None,
        components: UtteranceComponents {
            what: ctx.action_summary.clone(),
            why: ctx.reason_short.clone(),
            certainty: ctx.trust_label.clone(),
            tradeoffs: ctx.reversible_statement.clone(),
        },
        tags: vec!["flat_mode".to_string(), "minimal".to_string()],
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_creation() {
        let engine = SimpleResonantEngine::new();
        assert_eq!(engine.templates.len(), 3);
    }

    #[test]
    fn test_technician_micro_high_template() {
        let engine = SimpleResonantEngine::new();

        let ctx = ResonantContext {
            user_state: UserState {
                cognitive_load: CognitiveLoad::High,
                trust_in_sophia: 0.5,
                locale: "en-US".to_string(),
                flat_mode: false,
            },
            relationship: RelationshipProfile {
                primary: RelationshipMode::Technician,
                secondary: None,
                weight_primary: 1.0,
            },
            temporal_frame: TemporalFrame::Micro,
            suggestion_decision: SuggestionDecision {
                kind: SuggestionDecisionKind::AutoApply,
                reason: "Test".to_string(),
                claim_id: None,
            },
            channel: OutputChannel::Terminal,
            action_summary: "Regenerate hardware config".to_string(),
            reason_short: "Fixes boot failures".to_string(),
            trust_label: "High (0.87)".to_string(),
            reversible_statement: "Reversible via rollback".to_string(),
            arc_name: None,
            arc_delta: None,
            timeframe: None,
            controversy_note: None,
            k_deltas: vec![],
        };

        let utterance = engine.compose_utterance(&ctx);

        assert!(utterance.text.contains("Fix:"));
        assert!(utterance.text.contains("Regenerate hardware config"));
        assert!(utterance.tags.contains(&"technician".to_string()));
    }

    #[test]
    fn test_controversial_override() {
        let engine = SimpleResonantEngine::new();

        let ctx = ResonantContext {
            user_state: UserState {
                cognitive_load: CognitiveLoad::Medium,
                trust_in_sophia: 0.5,
                locale: "en-US".to_string(),
                flat_mode: false,
            },
            relationship: RelationshipProfile {
                primary: RelationshipMode::Technician,
                secondary: None,
                weight_primary: 1.0,
            },
            temporal_frame: TemporalFrame::Micro,
            suggestion_decision: SuggestionDecision {
                kind: SuggestionDecisionKind::Reject,
                reason: "Low trust".to_string(),
                claim_id: None,
            },
            channel: OutputChannel::Terminal,
            action_summary: "Risky config change".to_string(),
            reason_short: "Mixed evidence".to_string(),
            trust_label: "Low (0.3)".to_string(),
            reversible_statement: "May be hard to revert".to_string(),
            arc_name: None,
            arc_delta: None,
            timeframe: None,
            controversy_note: Some("Swarm has conflicting claims".to_string()),
        };

        let utterance = engine.compose_utterance(&ctx);

        assert!(utterance.text.contains("contentious"));
        assert!(utterance.tags.contains(&"controversial".to_string()));
    }

    #[test]
    fn test_governance_detection() {
        let engine = SimpleResonantEngine::new();

        let ctx = ResonantContext {
            user_state: UserState {
                cognitive_load: CognitiveLoad::Medium,
                trust_in_sophia: 0.7,
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
                reason: "Governance decision".to_string(),
                claim_id: None,
            },
            channel: OutputChannel::Terminal,
            action_summary: "Modify DAO charter".to_string(),
            reason_short: "Aligns with governance goals".to_string(),
            trust_label: "Medium (0.6)".to_string(),
            reversible_statement: "Requires vote to revert".to_string(),
            arc_name: Some("Governance Reform".to_string()),
            arc_delta: Some(0.05),
            timeframe: Some("this quarter".to_string()),
            controversy_note: None,
            k_deltas: vec![],
        };

        let utterance = engine.compose_utterance(&ctx);

        assert!(utterance.text.contains("Governance"));
        assert!(utterance.tags.contains(&"governance".to_string()));
    }

    #[test]
    fn test_voice_channel_adaptation() {
        let engine = SimpleResonantEngine::new();

        let ctx = ResonantContext {
            user_state: UserState {
                cognitive_load: CognitiveLoad::High,
                trust_in_sophia: 0.5,
                locale: "en-US".to_string(),
                flat_mode: false,
            },
            relationship: RelationshipProfile {
                primary: RelationshipMode::Technician,
                secondary: None,
                weight_primary: 1.0,
            },
            temporal_frame: TemporalFrame::Micro,
            suggestion_decision: SuggestionDecision {
                kind: SuggestionDecisionKind::AutoApply,
                reason: "Test".to_string(),
                claim_id: None,
            },
            channel: OutputChannel::Voice,
            action_summary: "Fix boot".to_string(),
            reason_short: "Common pattern".to_string(),
            trust_label: "High".to_string(),
            reversible_statement: "Reversible".to_string(),
            arc_name: None,
            arc_delta: None,
            timeframe: None,
            controversy_note: None,
            k_deltas: vec![],
        };

        let utterance = engine.compose_utterance(&ctx);

        // Voice should have shorter tradeoffs
        assert!(utterance
            .components
            .tradeoffs
            .contains("I can explain if you'd like"));
    }
}
