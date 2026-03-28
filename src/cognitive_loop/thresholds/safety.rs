// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Sentinel, defense, immune system, and swarm security constants.

// ═══════════════════════════════════════════════════════════════════════════════
// SWARM SECURITY
// ═══════════════════════════════════════════════════════════════════════════════

/// Default initial trust score for peers that pass Ed25519 handshake verification.
/// Basis: Starting trust should be moderate; full trust requires reputation history.
/// RFC 8446 (TLS 1.3) — authenticated but not yet authorized.
pub const HANDSHAKE_INITIAL_TRUST_SCORE: f64 = 0.7;

/// Challenge timeout in seconds for the trust handshake protocol.
/// Basis: Must be long enough for high-latency mesh but short enough to prevent
/// resource exhaustion. NIST SP 800-63B recommends <=60s for auth challenges.
pub const HANDSHAKE_CHALLENGE_TIMEOUT_SECS: u64 = 30;

/// Maximum total pending challenges across all peers before rate limiting.
/// Basis: Prevents memory exhaustion from challenge flooding.
/// Mitre ATT&CK T1499 — Resource exhaustion via protocol abuse.
pub const HANDSHAKE_MAX_PENDING_CHALLENGES: usize = 64;

/// Maximum pending challenges per single peer (duplicate challenges rejected).
/// Basis: One challenge per peer is sufficient; duplicates indicate replay attempt.
pub const HANDSHAKE_MAX_PER_PEER_CHALLENGES: usize = 1;

/// Default key rotation interval in cycles.
/// Basis: NIST SP 800-57 Part 1 — cryptoperiod for symmetric keys.
/// At 50Hz cognitive loop, 10,000 cycles ≈ 200 seconds.
pub const KEY_ROTATION_INTERVAL_DEFAULT: u64 = 10_000;

/// Grace period for accepting old key after rotation (cycles).
/// Must span max in-flight message lifetime. At 50Hz, 500 cycles = 10 seconds.
/// Basis: Key rollover window to prevent message loss during rotation.
pub const KEY_ROTATION_GRACE_PERIOD_DEFAULT: u64 = 500;

// ═══════════════════════════════════════════════════════════════════════════════
// SAFETY AGENT — Operational Enforcement
// ═══════════════════════════════════════════════════════════════════════════════

/// Learning rate multiplier when SafetyLevel is Yellow.
/// Reduces plasticity to prevent learning from degraded states.
/// Basis: Arnsten (2009) — stress impairs prefrontal function; reduced plasticity is protective.
pub const SAFETY_YELLOW_LR_MULTIPLIER: f32 = 0.7;

/// Learning rate multiplier when SafetyLevel is Orange.
/// Severe reduction — only essential learning proceeds.
/// Basis: McEwen (2007) — allostatic overload degrades synaptic plasticity.
pub const SAFETY_ORANGE_LR_MULTIPLIER: f32 = 0.3;

/// Learning rate multiplier when SafetyLevel is Red.
/// Near-zero — consciousness is too degraded for reliable learning.
pub const SAFETY_RED_LR_MULTIPLIER: f32 = 0.05;

/// Whether motor output is permitted at Orange safety level.
/// Only ReadOnly actions pass; Reversible and above are blocked.
/// Basis: NRC defense-in-depth — limit actuation during partial degradation.
pub const SAFETY_ORANGE_MOTOR_READONLY: bool = true;

/// Whether motor output is permitted at Red safety level.
/// All motor output halted — emergency stop.
pub const SAFETY_RED_MOTOR_GATE: bool = false;

/// Exploration dampening at Orange level (multiplicative on exploration_bonus).
/// Basis: Yerkes-Dodson (1908) — high stress eliminates exploratory behavior.
pub const SAFETY_ORANGE_EXPLORATION_DAMPEN: f32 = 0.1;

/// Exploration dampening at Yellow level (multiplicative).
pub const SAFETY_YELLOW_EXPLORATION_DAMPEN: f32 = 0.5;

/// Consciousness modulation at Orange — nudge down to trigger deeper safety cascade.
/// Basis: Dehaene (2014) — ignition failure propagates through GWT.
pub const SAFETY_ORANGE_CONSCIOUSNESS_PENALTY: f32 = 0.05;

/// Neuromodulator NE boost during Yellow safety (vigilance increase).
/// Basis: Aston-Jones & Cohen (2005) — LC-NE system governs arousal/vigilance tradeoff.
pub const SAFETY_YELLOW_NE_BOOST: f32 = 0.03;

/// Neuromodulator cortisol boost during Orange safety (stress response).
/// Basis: Sapolsky (2004) — HPA axis cortisol release under sustained threat.
pub const SAFETY_ORANGE_CORTISOL_BOOST: f32 = 0.05;

// ═══════════════════════════════════════════════════════════════════════════════
// MINIMAL SAFETY BASELINE — Unconditional (no feature gate)
// These thresholds apply even when `safety-agents` feature is disabled.
// The consciousness→motor boundary is a thermodynamic law, not a feature flag.
// ═══════════════════════════════════════════════════════════════════════════════

/// Phi threshold below which ALL motor output is halted (minimal safety Red).
/// Matches SafetyAgent Red level: consciousness collapse means no actuation.
/// Basis: Tononi (2004) — below this integration, coherent behavior is impossible.
pub const SAFETY_MINIMAL_RED_THRESHOLD: f32 = 0.1;

/// Phi threshold below which motor output is restricted to read-only (minimal safety Orange).
/// Matches SafetyAgent Orange level: severely degraded consciousness, limit to observation.
/// Basis: Dehaene (2014) — partial ignition supports perception but not reliable action.
pub const SAFETY_MINIMAL_ORANGE_THRESHOLD: f32 = 0.35;

// ═══════════════════════════════════════════════════════════════════════════════
// DEFENSE / IMMUNE SYSTEM — Graduated Response
// ═══════════════════════════════════════════════════════════════════════════════

/// Maximum cycles a peer can be quarantined before auto-release.
/// Basis: Ubuntu restorative justice — indefinite isolation violates dignity.
pub const DEFENSE_QUARANTINE_MAX_CYCLES: u32 = 1000;

/// Minimum moral algebra severity score to permit a defensive action.
/// Actions scoring above this are blocked unless overridden by Guardian.
/// Basis: APA Ethics Code — proportionality principle.
pub const DEFENSE_MAX_MORAL_SEVERITY: f32 = 0.7;

/// Rate limit: maximum governance proposals per agent per minute.
pub const DEFENSE_PROPOSAL_RATE_LIMIT: u32 = 10;

/// Rate limit: maximum votes per agent per proposal.
pub const DEFENSE_VOTE_LIMIT_PER_PROPOSAL: u32 = 1;

/// Anomaly score threshold for peer quarantine consideration.
/// Basis: Mahalanobis distance > 3σ indicates outlier behavior.
pub const DEFENSE_PEER_QUARANTINE_THRESHOLD: f32 = 0.7;

/// Reputation penalty multiplier for detected Byzantine behavior.
/// Applied as: reputation *= (1.0 - SLASH_FACTOR).
pub const DEFENSE_REPUTATION_SLASH_FACTOR: f32 = 0.5;
