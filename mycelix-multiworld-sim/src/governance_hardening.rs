// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! # Governance Hardening — Constitutional Anti-Tyranny Invariants
//!
//! Implements the 16 hardcoded invariants from the Anti-Tyranny Design
//! document and Constitution v0.25, mapped to monthly tick resolution.
//!
//! ## Temporal Mapping (1 tick = 1 month)
//!
//! - 14-day emergency → 1 tick max
//! - 30-day cooldown → 1 tick cooldown
//! - 7-day veto cooldown → sub-tick (max 1 veto per Guardian per tick)
//! - 3 vetoes per 12 months → 3 per 12 ticks
//! - 365-day term → 12 ticks
//!
//! ## AI Agent Ceiling
//!
//! Constitution Article XI, Section 4: AI agents capped at Steward tier.
//! Non-biological agents cannot reach Guardian regardless of MYCEL score.

use serde::{Deserialize, Serialize};

// ============================================================================
// Configuration (maps constitutional limits to monthly ticks)
// ============================================================================

/// Governance hardening parameters — all derived from the Constitution
/// and Anti-Tyranny Design, mapped to monthly tick resolution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GovernanceHardeningConfig {
    /// Veto override threshold (constitutional: ⅔ = 0.67).
    pub veto_override_threshold: f64,
    /// Maximum vetoes per Guardian per 12 ticks (constitutional: 3/year).
    pub veto_yearly_limit: u32,
    /// Primary quorum threshold (constitutional: 40%).
    pub quorum_primary: f64,
    /// Fallback quorum after 2 failures (constitutional: 30%).
    pub quorum_fallback: f64,
    /// Number of consecutive quorum failures before Expert Adjudication.
    pub quorum_failures_to_expert: u32,
    /// Maximum emergency session duration in ticks (14 days → 1 tick).
    pub emergency_max_ticks: u32,
    /// Maximum consecutive emergency sessions.
    pub emergency_max_consecutive: u32,
    /// Cooldown between emergency sessions in ticks (30 days → 1 tick).
    pub emergency_cooldown_ticks: u32,
    /// Maximum governance tier for non-biological agents (Steward = 3).
    pub ai_max_tier: u8,
    /// Self-reported (unsigned) phi cap (constitutional: 0.6).
    pub unsigned_phi_cap: f64,
    /// Council membership term in ticks (365 days → 12 ticks).
    pub term_limit_ticks: u32,
}

impl Default for GovernanceHardeningConfig {
    fn default() -> Self {
        Self {
            veto_override_threshold: 0.67,
            veto_yearly_limit: 3,
            quorum_primary: 0.40,
            quorum_fallback: 0.30,
            quorum_failures_to_expert: 3,
            emergency_max_ticks: 1,
            emergency_max_consecutive: 3,
            emergency_cooldown_ticks: 1,
            ai_max_tier: 3, // Steward
            unsigned_phi_cap: 0.6,
            term_limit_ticks: 12, // 1 year
        }
    }
}

// ============================================================================
// Governance State Tracking
// ============================================================================

/// Tracks governance state for hardening invariant enforcement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GovernanceHardeningState {
    /// Per-guardian veto history: (guardian_id, tick) for the last 12 ticks.
    pub veto_history: Vec<(u64, u32)>,
    /// Consecutive quorum failures for the current proposal.
    pub consecutive_quorum_failures: u32,
    /// Whether currently in Expert Adjudication mode.
    pub expert_adjudication_active: bool,
    /// Current emergency session tick counter.
    pub emergency_ticks_remaining: u32,
    /// Consecutive emergency sessions used.
    pub emergency_sessions_used: u32,
    /// Ticks remaining in emergency cooldown.
    pub emergency_cooldown_remaining: u32,
    /// Expert Adjudication invocations this year (constitutional: max 5).
    pub expert_invocations_this_year: u32,
    /// Tick of last year-start (for annual counters).
    pub year_start_tick: u32,
}

impl Default for GovernanceHardeningState {
    fn default() -> Self {
        Self {
            veto_history: Vec::new(),
            consecutive_quorum_failures: 0,
            expert_adjudication_active: false,
            emergency_ticks_remaining: 0,
            emergency_sessions_used: 0,
            emergency_cooldown_remaining: 0,
            expert_invocations_this_year: 0,
            year_start_tick: 0,
        }
    }
}

// ============================================================================
// Invariant Checks
// ============================================================================

/// Check if a Guardian can veto this tick (veto rate limit).
///
/// Returns false if the Guardian has exceeded the yearly limit (3 per 12 ticks).
pub fn can_veto(
    state: &GovernanceHardeningState,
    config: &GovernanceHardeningConfig,
    guardian_id: u64,
    current_tick: u32,
) -> bool {
    let window_start = current_tick.saturating_sub(12);
    let recent_count = state
        .veto_history
        .iter()
        .filter(|(id, tick)| *id == guardian_id && *tick >= window_start)
        .count();
    (recent_count as u32) < config.veto_yearly_limit
}

/// Record a veto. Call after a veto is exercised.
pub fn record_veto(state: &mut GovernanceHardeningState, guardian_id: u64, current_tick: u32) {
    state.veto_history.push((guardian_id, current_tick));
    // Prune old entries (older than 12 ticks)
    let cutoff = current_tick.saturating_sub(12);
    state.veto_history.retain(|(_, tick)| *tick >= cutoff);
}

/// Compute effective quorum threshold based on failure history.
///
/// Returns:
/// - `Ok(threshold)` if quorum is possible
/// - `Err("expert_adjudication")` if too many failures
pub fn effective_quorum(
    state: &GovernanceHardeningState,
    config: &GovernanceHardeningConfig,
) -> Result<f64, &'static str> {
    if state.consecutive_quorum_failures >= config.quorum_failures_to_expert {
        Err("expert_adjudication")
    } else if state.consecutive_quorum_failures >= 2 {
        Ok(config.quorum_fallback)
    } else {
        Ok(config.quorum_primary)
    }
}

/// Check if an emergency session can be started.
pub fn can_start_emergency(
    state: &GovernanceHardeningState,
    config: &GovernanceHardeningConfig,
) -> bool {
    state.emergency_cooldown_remaining == 0
        && state.emergency_sessions_used < config.emergency_max_consecutive
}

/// Start an emergency session. Returns false if not allowed.
pub fn start_emergency(
    state: &mut GovernanceHardeningState,
    config: &GovernanceHardeningConfig,
) -> bool {
    if !can_start_emergency(state, config) {
        return false;
    }
    state.emergency_ticks_remaining = config.emergency_max_ticks;
    state.emergency_sessions_used += 1;
    true
}

/// Tick the emergency session state. Call every tick.
pub fn tick_emergency(state: &mut GovernanceHardeningState, config: &GovernanceHardeningConfig) {
    if state.emergency_ticks_remaining > 0 {
        state.emergency_ticks_remaining -= 1;
        if state.emergency_ticks_remaining == 0 {
            // Session ended — start cooldown if max consecutive reached
            if state.emergency_sessions_used >= config.emergency_max_consecutive {
                state.emergency_cooldown_remaining = config.emergency_cooldown_ticks;
                state.emergency_sessions_used = 0; // reset after cooldown
            }
        }
    } else if state.emergency_cooldown_remaining > 0 {
        // Only decrement cooldown when NOT in an active session
        state.emergency_cooldown_remaining -= 1;
    }
}

/// Compute effective governance tier for an agent, respecting AI ceiling.
///
/// Non-biological agents are capped at `config.ai_max_tier` (Steward = 3).
pub fn effective_tier(raw_tier: u8, is_biological: bool, config: &GovernanceHardeningConfig) -> u8 {
    if !is_biological && raw_tier > config.ai_max_tier {
        config.ai_max_tier
    } else {
        raw_tier
    }
}

/// Cap a self-reported (unsigned) phi score.
///
/// Self-reported consciousness cannot exceed `unsigned_phi_cap` (0.6).
/// Only Ed25519-signed attestations can reach Guardian tier.
pub fn cap_unsigned_phi(phi: f64, is_signed: bool, config: &GovernanceHardeningConfig) -> f64 {
    if is_signed {
        phi
    } else {
        phi.min(config.unsigned_phi_cap)
    }
}

/// Reset annual counters. Call when `current_tick % 12 == 0`.
pub fn tick_annual_reset(state: &mut GovernanceHardeningState, current_tick: u32) {
    if current_tick >= state.year_start_tick + 12 {
        state.expert_invocations_this_year = 0;
        state.year_start_tick = current_tick;
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> GovernanceHardeningConfig {
        GovernanceHardeningConfig::default()
    }

    fn default_state() -> GovernanceHardeningState {
        GovernanceHardeningState::default()
    }

    // ---- Veto limits ----

    #[test]
    fn veto_allowed_initially() {
        let state = default_state();
        assert!(can_veto(&state, &default_config(), 1, 0));
    }

    #[test]
    fn veto_blocked_after_three_in_twelve_ticks() {
        let mut state = default_state();
        let config = default_config();
        record_veto(&mut state, 1, 0);
        record_veto(&mut state, 1, 4);
        record_veto(&mut state, 1, 8);
        assert!(
            !can_veto(&state, &config, 1, 10),
            "4th veto within 12 ticks should be blocked"
        );
    }

    #[test]
    fn veto_allowed_after_window_expires() {
        let mut state = default_state();
        let config = default_config();
        record_veto(&mut state, 1, 0);
        record_veto(&mut state, 1, 1);
        record_veto(&mut state, 1, 2);
        // At tick 13, the first veto (tick 0) is outside the 12-tick window
        assert!(can_veto(&state, &config, 1, 13));
    }

    #[test]
    fn veto_limit_per_guardian_not_global() {
        let mut state = default_state();
        let config = default_config();
        record_veto(&mut state, 1, 0);
        record_veto(&mut state, 1, 1);
        record_veto(&mut state, 1, 2);
        // Guardian 1 is blocked, but Guardian 2 is fine
        assert!(!can_veto(&state, &config, 1, 3));
        assert!(can_veto(&state, &config, 2, 3));
    }

    // ---- Quorum ----

    #[test]
    fn quorum_starts_at_primary() {
        let state = default_state();
        assert_eq!(effective_quorum(&state, &default_config()), Ok(0.40));
    }

    #[test]
    fn quorum_drops_after_two_failures() {
        let mut state = default_state();
        state.consecutive_quorum_failures = 2;
        assert_eq!(effective_quorum(&state, &default_config()), Ok(0.30));
    }

    #[test]
    fn quorum_triggers_expert_after_three_failures() {
        let mut state = default_state();
        state.consecutive_quorum_failures = 3;
        assert_eq!(
            effective_quorum(&state, &default_config()),
            Err("expert_adjudication")
        );
    }

    // ---- Emergency sessions ----

    #[test]
    fn emergency_can_start_initially() {
        let state = default_state();
        assert!(can_start_emergency(&state, &default_config()));
    }

    #[test]
    fn emergency_blocked_during_cooldown() {
        let mut state = default_state();
        state.emergency_cooldown_remaining = 1;
        assert!(!can_start_emergency(&state, &default_config()));
    }

    #[test]
    fn emergency_blocked_after_max_consecutive() {
        let mut state = default_state();
        state.emergency_sessions_used = 3;
        assert!(!can_start_emergency(&state, &default_config()));
    }

    #[test]
    fn emergency_session_expires_after_max_ticks() {
        let mut state = default_state();
        let config = default_config();
        start_emergency(&mut state, &config);
        assert_eq!(state.emergency_ticks_remaining, 1);

        tick_emergency(&mut state, &config);
        assert_eq!(
            state.emergency_ticks_remaining, 0,
            "Session should expire after 1 tick"
        );
    }

    #[test]
    fn emergency_cooldown_after_max_consecutive() {
        let mut state = default_state();
        let config = default_config();

        // Use all 3 consecutive sessions
        for _ in 0..3 {
            start_emergency(&mut state, &config);
            tick_emergency(&mut state, &config);
        }

        assert_eq!(
            state.emergency_cooldown_remaining, 1,
            "Cooldown should activate"
        );
        assert!(
            !can_start_emergency(&state, &config),
            "Should be blocked during cooldown"
        );

        tick_emergency(&mut state, &config);
        assert_eq!(
            state.emergency_cooldown_remaining, 0,
            "Cooldown should expire"
        );
    }

    // ---- AI ceiling ----

    #[test]
    fn biological_agent_unrestricted() {
        assert_eq!(effective_tier(4, true, &default_config()), 4); // Guardian
    }

    #[test]
    fn ai_agent_capped_at_steward() {
        assert_eq!(effective_tier(4, false, &default_config()), 3); // Capped to Steward
        assert_eq!(effective_tier(3, false, &default_config()), 3); // Already Steward → unchanged
        assert_eq!(effective_tier(2, false, &default_config()), 2); // Below cap → unchanged
    }

    // ---- Unsigned phi cap ----

    #[test]
    fn signed_phi_uncapped() {
        assert_eq!(cap_unsigned_phi(0.9, true, &default_config()), 0.9);
    }

    #[test]
    fn unsigned_phi_capped_at_0_6() {
        assert_eq!(cap_unsigned_phi(0.9, false, &default_config()), 0.6);
        assert_eq!(cap_unsigned_phi(0.5, false, &default_config()), 0.5); // below cap → unchanged
    }

    // ---- Annual reset ----

    #[test]
    fn annual_reset_clears_expert_count() {
        let mut state = default_state();
        state.expert_invocations_this_year = 4;
        state.year_start_tick = 0;
        tick_annual_reset(&mut state, 12);
        assert_eq!(state.expert_invocations_this_year, 0);
        assert_eq!(state.year_start_tick, 12);
    }

    // ---- Config defaults match constitution ----

    #[test]
    fn config_matches_constitution() {
        let config = default_config();
        assert!(
            (config.veto_override_threshold - 0.67).abs() < 0.01,
            "⅔ majority"
        );
        assert_eq!(config.veto_yearly_limit, 3, "3 vetoes per year");
        assert!((config.quorum_primary - 0.40).abs() < 0.01, "40% quorum");
        assert!((config.quorum_fallback - 0.30).abs() < 0.01, "30% fallback");
        assert_eq!(config.quorum_failures_to_expert, 3, "3 failures → expert");
        assert_eq!(config.emergency_max_ticks, 1, "1 month emergency max");
        assert_eq!(config.emergency_max_consecutive, 3, "3 consecutive max");
        assert_eq!(config.ai_max_tier, 3, "Steward ceiling for AI");
        assert!(
            (config.unsigned_phi_cap - 0.6).abs() < 0.01,
            "0.6 self-report cap"
        );
        assert_eq!(config.term_limit_ticks, 12, "12-month terms");
    }
}
