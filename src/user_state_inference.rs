// User State Inference - v0.2
//
// Infers UserState from observable signals rather than requiring explicit input.
// Philosophy: "We don't guess your inner life, but we're not blind either."

use crate::resonant_speech::{UserState, CognitiveLoad};
use std::time::{Duration, Instant};

/// What kind of context is Sophia being invoked from?
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContextKind {
    /// Error handling, system recovery, incident response
    ErrorHandling,

    /// Active development work, writing code, editing docs
    DevWork,

    /// Writing papers, documentation, research
    Writing,

    /// Weekly reviews, K-Index dashboard, retrospectives
    Review,

    /// Long-term planning, visioning, architecture decisions
    Planning,

    /// General exploration, learning, curiosity-driven
    Exploration,
}

/// Exploration mode - what mental state is the user in?
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExplorationMode {
    /// Fixing something broken, high urgency
    Fixing,

    /// Learning, exploring, understanding
    Learning,

    /// Long-term visioning, strategic thinking
    Visioning,
}

/// Time pressure signal from urgency indicators
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimePressure {
    /// No particular rush
    Calm,

    /// Some time pressure but manageable
    Moderate,

    /// Urgent, need help now
    Urgent,
}

/// Recent events tracked in a sliding window
#[derive(Debug, Clone)]
pub struct RecentEvents {
    /// Number of errors in last 5 minutes
    pub errors_last_5m: usize,

    /// Number of undo/rollback operations in last 10 minutes
    pub undos_last_10m: usize,

    /// Current session duration
    pub session_duration: Duration,

    /// How long since last command (for detecting calm focus)
    pub time_since_last_command: Duration,

    /// Number of suggestions accepted vs rejected (for trust)
    pub suggestions_accepted: usize,
    pub suggestions_rejected: usize,

    /// Timestamp when tracking started
    pub window_start: Instant,
}

impl RecentEvents {
    pub fn new() -> Self {
        Self {
            errors_last_5m: 0,
            undos_last_10m: 0,
            session_duration: Duration::from_secs(0),
            time_since_last_command: Duration::from_secs(0),
            suggestions_accepted: 0,
            suggestions_rejected: 0,
            window_start: Instant::now(),
        }
    }

    /// Detect error burst pattern (panic mode indicator)
    pub fn error_burst(&self) -> bool {
        self.errors_last_5m >= 3
    }

    /// Detect long focused session (low cognitive load indicator)
    pub fn long_focus_session(&self) -> bool {
        self.session_duration > Duration::from_secs(30 * 60) // 30+ minutes
            && self.time_since_last_command > Duration::from_secs(60) // 1+ min since last command
            && self.errors_last_5m == 0
    }

    /// Detect thrashing pattern (many undos/rollbacks)
    pub fn thrashing(&self) -> bool {
        self.undos_last_10m >= 5
    }

    /// Estimate rolling trust based on accept/reject ratio
    pub fn rolling_trust_estimate(&self) -> f32 {
        let total = self.suggestions_accepted + self.suggestions_rejected;
        if total == 0 {
            return 0.5; // neutral prior
        }

        let ratio = self.suggestions_accepted as f32 / total as f32;

        // Map to 0.0-1.0 range with some dampening
        // 100% accept → 0.9 (leave room for growth)
        // 0% accept → 0.1 (don't go to zero, allow recovery)
        0.1 + (ratio * 0.8)
    }
}

impl Default for RecentEvents {
    fn default() -> Self {
        Self::new()
    }
}

/// Infers UserState from observable context + recent events
pub struct UserStateInference {
    /// Sliding window of recent activity
    recent_events: RecentEvents,
}

impl UserStateInference {
    pub fn new() -> Self {
        Self {
            recent_events: RecentEvents::new(),
        }
    }

    /// Create from existing event tracking
    pub fn with_events(recent_events: RecentEvents) -> Self {
        Self { recent_events }
    }

    /// Primary inference method
    pub fn infer(&self, context_kind: ContextKind, locale: &str) -> UserState {
        let cognitive_load = self.infer_cognitive_load(context_kind);
        let trust_in_sophia = self.recent_events.rolling_trust_estimate();

        UserState {
            cognitive_load,
            trust_in_sophia,
            locale: locale.to_string(),
            flat_mode: false,  // Inference doesn't set flat_mode - that's explicit user choice
        }
    }

    /// Infer cognitive load from context + recent friction
    fn infer_cognitive_load(&self, context_kind: ContextKind) -> CognitiveLoad {
        // High load conditions
        if self.recent_events.error_burst() {
            return CognitiveLoad::High;
        }

        if self.recent_events.thrashing() {
            return CognitiveLoad::High;
        }

        // Context-specific high load
        if matches!(context_kind, ContextKind::ErrorHandling)
            && self.recent_events.errors_last_5m > 0 {
            return CognitiveLoad::High;
        }

        // Low load conditions
        if self.recent_events.long_focus_session() {
            return CognitiveLoad::Low;
        }

        if matches!(
            context_kind,
            ContextKind::Review | ContextKind::Planning
        ) && self.recent_events.errors_last_5m == 0 {
            return CognitiveLoad::Low;
        }

        // Default to Medium
        CognitiveLoad::Medium
    }

    /// Infer exploration mode from context
    pub fn infer_exploration_mode(&self, context_kind: ContextKind) -> ExplorationMode {
        match context_kind {
            ContextKind::ErrorHandling => ExplorationMode::Fixing,
            ContextKind::DevWork | ContextKind::Writing | ContextKind::Exploration => {
                ExplorationMode::Learning
            }
            ContextKind::Review | ContextKind::Planning => ExplorationMode::Visioning,
        }
    }

    /// Infer time pressure from urgency signals
    pub fn infer_time_pressure(&self, urgency_hint: Option<&str>) -> TimePressure {
        // Explicit urgency from user input
        if let Some(hint) = urgency_hint {
            let hint_lower = hint.to_lowercase();
            if hint_lower.contains("urgent")
                || hint_lower.contains("now")
                || hint_lower.contains("pls just fix")
                || hint_lower.contains("help it's broken")
                || hint_lower.contains("asap")
            {
                return TimePressure::Urgent;
            }

            if hint_lower.contains("curious")
                || hint_lower.contains("could we explore")
                || hint_lower.contains("what do you think")
            {
                return TimePressure::Calm;
            }
        }

        // Infer from recent events
        if self.recent_events.error_burst() || self.recent_events.thrashing() {
            return TimePressure::Urgent;
        }

        if self.recent_events.long_focus_session() {
            return TimePressure::Calm;
        }

        TimePressure::Moderate
    }

    /// Update recent events (called by the system as activity happens)
    pub fn record_error(&mut self) {
        self.recent_events.errors_last_5m += 1;
    }

    pub fn record_undo(&mut self) {
        self.recent_events.undos_last_10m += 1;
    }

    pub fn record_suggestion_accepted(&mut self) {
        self.recent_events.suggestions_accepted += 1;
    }

    pub fn record_suggestion_rejected(&mut self) {
        self.recent_events.suggestions_rejected += 1;
    }

    pub fn update_session_duration(&mut self, duration: Duration) {
        self.recent_events.session_duration = duration;
    }

    pub fn update_time_since_last_command(&mut self, duration: Duration) {
        self.recent_events.time_since_last_command = duration;
    }

    /// Reset the sliding window (e.g., at end of session)
    pub fn reset(&mut self) {
        self.recent_events = RecentEvents::new();
    }
}

impl Default for UserStateInference {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_burst_triggers_high_load() {
        let mut events = RecentEvents::new();
        events.errors_last_5m = 5;

        let inference = UserStateInference::with_events(events);
        let state = inference.infer(ContextKind::DevWork, "en-US");

        assert_eq!(state.cognitive_load, CognitiveLoad::High);
    }

    #[test]
    fn test_calm_focus_triggers_low_load() {
        let mut events = RecentEvents::new();
        events.session_duration = Duration::from_secs(45 * 60); // 45 minutes
        events.time_since_last_command = Duration::from_secs(120); // 2 minutes
        events.errors_last_5m = 0;

        let inference = UserStateInference::with_events(events);
        let state = inference.infer(ContextKind::Writing, "en-US");

        assert_eq!(state.cognitive_load, CognitiveLoad::Low);
    }

    #[test]
    fn test_trust_estimate_from_accepts() {
        let mut events = RecentEvents::new();
        events.suggestions_accepted = 8;
        events.suggestions_rejected = 2;

        let trust = events.rolling_trust_estimate();
        assert!(trust > 0.7); // 80% accept rate should be >0.7
    }

    #[test]
    fn test_exploration_mode_inference() {
        let inference = UserStateInference::new();

        assert_eq!(
            inference.infer_exploration_mode(ContextKind::ErrorHandling),
            ExplorationMode::Fixing
        );

        assert_eq!(
            inference.infer_exploration_mode(ContextKind::DevWork),
            ExplorationMode::Learning
        );

        assert_eq!(
            inference.infer_exploration_mode(ContextKind::Planning),
            ExplorationMode::Visioning
        );
    }

    #[test]
    fn test_urgency_hint_detection() {
        let inference = UserStateInference::new();

        assert_eq!(
            inference.infer_time_pressure(Some("urgent fix needed")),
            TimePressure::Urgent
        );

        assert_eq!(
            inference.infer_time_pressure(Some("curious about this feature")),
            TimePressure::Calm
        );

        assert_eq!(
            inference.infer_time_pressure(None),
            TimePressure::Moderate
        );
    }
}
