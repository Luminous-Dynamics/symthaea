// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Single-agent memetic immune system: screen incoming ideas, track how
//! contagious the local idea-environment is.
//!
//! This is the reusable core for plan Phase 2 ("wire into the live cognitive
//! loop"). It combines Phase 1 (reject known-pathogen memes — *not* novel ones)
//! and Phase 0 (adoption/contagion math) into one stateful screener with two
//! live couplings the cognitive loop already exposes:
//!
//! - **Safety tier** — a `Red` guardian posture suppresses *all* meme uptake
//!   (defensive lockdown); `Orange` tightens the pathogen threshold. Mirrors the
//!   loop's NRC 4-tier `SafetyLevel` without depending on it (pure input).
//! - **Arousal** — emotionally charged content is more transmissible (Berger &
//!   Milkman 2012, *What Makes Online Content Viral?*), so high arousal raises
//!   the reported contagion index.
//!
//! It deliberately reports a per-contact **contagion index** (how adoptable the
//! local idea-stream is to *this* mind), not a population R₀ — a single agent
//! has no population. True cross-agent R₀ needs the mesh (Phase 3).

use crate::meme::Meme;
use crate::propagation::{adoption_probability, resonance_gain};
use symthaea_core::hdc::binary_hv::BinaryHV;

/// NRC-style guardian posture, mirrored from the cognitive loop's `SafetyLevel`
/// so this crate needn't depend on it. Higher = more defensive.
///
/// `PartialOrd`/`Ord` follow declaration order (Green < Yellow < Orange < Red),
/// which is what makes a floor-clamp (`posture.max(floor)`) meaningful — see
/// [`WardConfig`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum GuardianPosture {
    /// Normal operation.
    Green,
    /// Heightened awareness (no behavioural change here).
    Yellow,
    /// Defensive: tighten the pathogen-rejection threshold.
    Orange,
    /// Lockdown: reject *all* meme uptake.
    Red,
}

/// Guardian-configurable protective settings for a *warded node* — a node
/// (e.g. a child's) whose owner wants stricter-than-default memetic defense.
/// See `WARDED_NODE_DESIGN_2026-07-11.md` (monorepo root) for the full design;
/// this is Phase 1 of that design ("the posture floor"), deliberately minimal.
///
/// Nothing here is a network-level control — it only affects what *this* node
/// accepts for itself, which is why it needs no owner sign-off the way a
/// mesh-wide suppression primitive would (see that doc's Bridge C discussion).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct WardConfig {
    /// The screen's effective posture is never more lenient than this,
    /// regardless of the (e.g. psi-derived) posture the caller passes to
    /// [`MemeticImmuneSystem::screen`]. `None` = no floor (default/adult
    /// behavior: the caller's posture is used as-is).
    pub posture_floor: Option<GuardianPosture>,
    /// How to treat content from peers not yet trusted (Layer B of the
    /// design; see [`AllowlistMode`]). Default `Open` = today's unwarded
    /// behavior.
    pub allowlist_mode: AllowlistMode,
}

/// How a warded node treats content from peers its owner hasn't vouched for.
///
/// Deliberately a **pure data type with no trust-graph dependency** — this
/// crate stays mesh/identity-unaware by design (see `CORE_SUBSTRATE.md`
/// discipline). The caller (the main crate, which owns the local web-of-trust
/// graph) looks up a peer's trust score and passes the *mode* here as a
/// policy to apply to that score; it does not reach into any trust store
/// itself.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum AllowlistMode {
    /// Admit based on content screening alone — today's default (a peer is
    /// admitted unless flagged as a Sybil anomaly by the trust graph, if one
    /// exists). Unwarded/adult behavior.
    #[default]
    Open,
    /// Default-deny: only admit content from peers whose local trust score
    /// (from the node's own web-of-trust graph — not a network round-trip)
    /// meets `min_trust`. A peer with no trust data (score `0.0`, including
    /// the case where no trust graph exists locally at all) is denied — this
    /// is the opposite default from `Open`, intentionally: for a sovereign
    /// adult node, a stranger's silence isn't evidence of danger (so
    /// `Open`'s cold-start-friendly admit-by-default is right); for a warded
    /// node, it isn't evidence of safety either, so this fails closed.
    AllowlistOnly { min_trust: f64 },
}

impl AllowlistMode {
    /// Whether a peer with the given local trust score should be admitted
    /// under this mode. Pass `0.0` for `trust_score` when no trust data
    /// exists for the peer at all (including when no trust graph is compiled
    /// in) — see the `AllowlistOnly` variant's docs for why that fails
    /// closed, not open.
    pub fn admits(&self, trust_score: f64) -> bool {
        match self {
            AllowlistMode::Open => true,
            AllowlistMode::AllowlistOnly { min_trust } => trust_score >= *min_trust,
        }
    }
}

/// Base similarity to a known pathogen at or above which a meme is rejected.
const BASE_THREAT_THRESHOLD: f32 = 0.7;
/// Tighter threshold used under `Orange` posture.
const ORANGE_THREAT_THRESHOLD: f32 = 0.5;
/// Recent-window length for rolling telemetry (resonance / contagion means).
const WINDOW: usize = 64;

/// Minimum contagion an accepted meme must have to be worth re-propagating to
/// peers. Below this, re-broadcasting would amplify weak/noise ideas and risk
/// mesh broadcast storms, so the agent adopts silently instead of forwarding.
const PROPAGATION_MIN_CONTAGION: f32 = 0.15;

/// Bounded ring-buffer capacity for the filtered-items audit log (Warded Node
/// design, Phase 2 — see [`FilteredItem`]).
const FILTERED_LOG_CAPACITY: usize = 128;

/// One rejected screening decision, retained for guardian transparency
/// (Warded Node design, Phase 2: "not a black box"). A guardian inspects this
/// to see *what* was blocked and *why* — the precondition for trusting a
/// posture floor or pre-vaccination rather than fighting it blind.
///
/// Deliberately does **not** store the meme's payload (no `BinaryHV` here):
/// `meme_id` is enough to correlate with other logs (in the live loop it's
/// the content's `created_at` timestamp — see `cycle_phase_dynamics`), and
/// keeping the log payload-free means it stays cheap to retain and to clone
/// out to a caller.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FilteredItem {
    /// The rejected meme's id (correlates with other logs; see struct docs).
    pub meme_id: u64,
    /// Resonance with the agent's belief at the time of screening.
    pub resonance: f32,
    /// Strongest match to a vaccinated pathogen (0..1).
    pub threat_match: f32,
    /// Contagion index this meme would have carried, had it been admitted.
    pub contagion: f32,
    /// Human-readable rejection reason (mirrors [`ScreenOutcome::reason`]).
    pub reason: &'static str,
}

/// Outcome of screening one incoming meme.
#[derive(Debug, Clone, PartialEq)]
pub struct ScreenOutcome {
    /// Whether the meme is admitted into the local belief.
    pub accepted: bool,
    /// Resonance with our own current belief (0..1). Reported, never a reason to reject.
    pub resonance: f32,
    /// Strongest match to a vaccinated pathogen (0..1) — the rejection criterion.
    pub threat_match: f32,
    /// Per-contact adoptability of this meme to us, arousal-coupled (0..1).
    pub contagion: f32,
    /// Human/telemetry-readable reason.
    pub reason: &'static str,
}

/// Rolling telemetry for the memetic immune system.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct MemeticTelemetry {
    /// Total memes screened.
    pub seen: u64,
    /// Memes rejected (pathogen match or lockdown).
    pub rejected: u64,
    /// Memes admitted.
    pub accepted: u64,
    /// Fraction rejected over all time.
    pub rejection_rate: f64,
    /// Mean resonance-with-self of recently screened memes.
    pub mean_resonance: f64,
    /// Mean arousal-coupled contagion index of recently screened memes.
    pub contagion_index: f64,
    /// Pathogen signatures currently in immune memory.
    pub immune_memory: usize,
}

/// A single agent's memetic immune system.
pub struct MemeticImmuneSystem {
    /// Vaccinated pathogen signatures (Phase 1 immune memory).
    threat_signatures: Vec<BinaryHV>,
    /// This agent's current belief/consciousness state.
    self_state: BinaryHV,
    /// Baseline openness to ideas (0..1).
    susceptibility: f32,
    seen: u64,
    rejected: u64,
    accepted: u64,
    recent_resonance: std::collections::VecDeque<f32>,
    recent_contagion: std::collections::VecDeque<f32>,
    /// Warded-node protective settings; `Default` = no floor (unwarded/adult).
    ward: WardConfig,
    /// Bounded audit log of rejections (Warded Node Phase 2 transparency).
    filtered_log: std::collections::VecDeque<FilteredItem>,
}

impl MemeticImmuneSystem {
    /// New immune system anchored to an initial belief state.
    pub fn new(self_state: BinaryHV, susceptibility: f32) -> Self {
        Self {
            threat_signatures: Vec::new(),
            self_state,
            susceptibility: susceptibility.clamp(0.0, 1.0),
            seen: 0,
            rejected: 0,
            accepted: 0,
            recent_resonance: std::collections::VecDeque::with_capacity(WINDOW),
            recent_contagion: std::collections::VecDeque::with_capacity(WINDOW),
            ward: WardConfig::default(),
            filtered_log: std::collections::VecDeque::with_capacity(FILTERED_LOG_CAPACITY),
        }
    }

    /// Vaccinate against a known memetic pathogen (mutation-tolerant recognition).
    pub fn vaccinate(&mut self, pathogen: BinaryHV) {
        self.threat_signatures.push(pathogen);
    }

    /// Bulk-vaccinate against every entry in a [`crate::ruleset::Ruleset`]
    /// (Warded Node design, Phase 5a: ruleset import). Returns the number of
    /// signatures applied. Trust in the ruleset's provenance is entirely the
    /// caller's — see [`crate::ruleset::Ruleset`]'s docs for why this crate
    /// doesn't verify a publisher's signature.
    pub fn vaccinate_ruleset(&mut self, ruleset: &crate::ruleset::Ruleset) -> usize {
        for entry in &ruleset.entries {
            self.vaccinate(entry.signature.clone());
        }
        ruleset.entries.len()
    }

    /// Update the agent's belief state (called each cycle from consciousness).
    pub fn set_self_state(&mut self, state: BinaryHV) {
        self.self_state = state;
    }

    /// Set this node's warded-node protective settings (see [`WardConfig`]).
    /// Pass `WardConfig::default()` to clear (no floor).
    pub fn set_ward_config(&mut self, ward: WardConfig) {
        self.ward = ward;
    }

    /// Current warded-node protective settings.
    pub fn ward_config(&self) -> WardConfig {
        self.ward
    }

    pub fn immune_memory_size(&self) -> usize {
        self.threat_signatures.len()
    }

    /// Strongest match of `payload` to any vaccinated pathogen (0..1).
    /// All `BinaryHV`s are fixed 16,384-D, so no dimension guard is needed.
    fn threat_match(&self, payload: &BinaryHV) -> f32 {
        self.threat_signatures
            .iter()
            .map(|s| payload.similarity(s).clamp(0.0, 1.0))
            .fold(0.0f32, f32::max)
    }

    /// Screen one incoming meme under the current guardian posture and arousal.
    ///
    /// - `posture` `Red` ⇒ reject everything; `Orange` ⇒ tighter threshold.
    /// - `arousal ∈ [0, 1]` raises the reported contagion index (charged ideas
    ///   spread more readily), but does **not** relax the pathogen gate.
    /// - If [`WardConfig::posture_floor`] is set, the *effective* posture used
    ///   below is `max(posture, floor)` — a warded node is never more lenient
    ///   than its configured floor, regardless of what the caller (e.g. a
    ///   psi-derived posture upstream) passes in. The floor can only tighten,
    ///   never loosen, the caller's posture.
    pub fn screen(&mut self, meme: &Meme, posture: GuardianPosture, arousal: f32) -> ScreenOutcome {
        self.seen += 1;

        let posture = match self.ward.posture_floor {
            Some(floor) => posture.max(floor),
            None => posture,
        };

        let resonance = meme.payload.similarity(&self.self_state).clamp(0.0, 1.0);
        let threat_match = self.threat_match(&meme.payload);

        // Per-contact adoptability of this meme to us, then arousal coupling:
        // arousal in [0,1] scales contagion by [1.0, 2.0].
        let base_adopt = adoption_probability(meme, &self.self_state, self.susceptibility);
        let contagion = (base_adopt * (1.0 + arousal.clamp(0.0, 1.0))).clamp(0.0, 1.0);

        self.push_window(resonance, contagion);

        let threshold = match posture {
            GuardianPosture::Orange => ORANGE_THREAT_THRESHOLD,
            _ => BASE_THREAT_THRESHOLD,
        };

        let outcome = if posture == GuardianPosture::Red {
            ScreenOutcome {
                accepted: false,
                resonance,
                threat_match,
                contagion,
                reason: "guardian lockdown (Red): all meme uptake suppressed",
            }
        } else if threat_match >= threshold {
            ScreenOutcome {
                accepted: false,
                resonance,
                threat_match,
                contagion,
                reason: "manipulative meme: matches a known pathogen signature",
            }
        } else {
            ScreenOutcome {
                accepted: true,
                resonance,
                threat_match,
                contagion,
                reason: "admitted (novelty is not a threat)",
            }
        };

        if outcome.accepted {
            self.accepted += 1;
        } else {
            self.rejected += 1;
            self.log_filtered(FilteredItem {
                meme_id: meme.id,
                resonance: outcome.resonance,
                threat_match: outcome.threat_match,
                contagion: outcome.contagion,
                reason: outcome.reason,
            });
        }
        outcome
    }

    /// Append to the bounded filtered-items audit log, dropping the oldest
    /// entry once at capacity (same ring-buffer pattern as `push_window`).
    fn log_filtered(&mut self, item: FilteredItem) {
        if self.filtered_log.len() == FILTERED_LOG_CAPACITY {
            self.filtered_log.pop_front();
        }
        self.filtered_log.push_back(item);
    }

    /// The `limit` most recent filtered (rejected) items, newest first —
    /// what a guardian inspects to see what was blocked and why (Warded Node
    /// Phase 2). Empty for an unwarded node that has never rejected anything.
    pub fn filtered_log(&self, limit: usize) -> Vec<FilteredItem> {
        self.filtered_log
            .iter()
            .rev()
            .take(limit)
            .copied()
            .collect()
    }

    /// Count of entries currently retained in the filtered-items audit log
    /// (bounded by [`FILTERED_LOG_CAPACITY`]; NOT the same as lifetime
    /// `rejected` count in [`MemeticTelemetry`], which never wraps).
    pub fn filtered_log_len(&self) -> usize {
        self.filtered_log.len()
    }

    /// Record a rejection that happened *before* [`screen`](Self::screen) ran
    /// at all — e.g. a peer denied by an allowlist gate or flagged as a Sybil
    /// anomaly, upstream decisions the caller makes with data this crate
    /// deliberately doesn't have (a trust graph). Content-derived fields
    /// (`resonance`/`threat_match`/`contagion`) are `0.0`/N-A, since no
    /// meme was ever screened.
    ///
    /// Deliberately does **not** increment `seen`/`rejected`/`accepted` —
    /// those describe the memetic *screen* specifically; a gate denial is a
    /// distinct, earlier decision point. It *does* still land in the
    /// [`filtered_log`](Self::filtered_log), because "not a black box" (the
    /// Warded Node design's transparency principle) applies to everything a
    /// guardian's node blocked, not only what the memetic screen rejected.
    pub fn log_gate_denial(&mut self, meme_id: u64, reason: &'static str) {
        self.log_filtered(FilteredItem {
            meme_id,
            resonance: 0.0,
            threat_match: 0.0,
            contagion: 0.0,
            reason,
        });
    }

    fn push_window(&mut self, resonance: f32, contagion: f32) {
        if self.recent_resonance.len() == WINDOW {
            self.recent_resonance.pop_front();
            self.recent_contagion.pop_front();
        }
        self.recent_resonance.push_back(resonance);
        self.recent_contagion.push_back(contagion);
    }

    fn mean(dq: &std::collections::VecDeque<f32>) -> f64 {
        if dq.is_empty() {
            0.0
        } else {
            dq.iter().map(|&x| x as f64).sum::<f64>() / dq.len() as f64
        }
    }

    /// Current rolling telemetry.
    pub fn telemetry(&self) -> MemeticTelemetry {
        MemeticTelemetry {
            seen: self.seen,
            rejected: self.rejected,
            accepted: self.accepted,
            rejection_rate: if self.seen > 0 {
                self.rejected as f64 / self.seen as f64
            } else {
                0.0
            },
            mean_resonance: Self::mean(&self.recent_resonance),
            contagion_index: Self::mean(&self.recent_contagion),
            immune_memory: self.threat_signatures.len(),
        }
    }

    /// Decide whether an accepted meme is worth re-propagating to peers, and if
    /// so produce the *mutated* variant to broadcast (plan Phase 3, cross-agent
    /// propagation). Returns `None` — adopt silently, don't forward — when:
    ///
    /// - the meme was rejected (never forward a pathogen), or
    /// - the guardian posture is `Red` (lockdown suppresses outbound spread too),
    ///   or
    /// - its contagion is below [`PROPAGATION_MIN_CONTAGION`] (don't amplify weak
    ///   or noise ideas; this is the anti-broadcast-storm guard).
    ///
    /// The returned variant carries `mutation`-scale drift, so an idea mutates as
    /// it is retold across the swarm — the transmission-fidelity model made live.
    pub fn propagation_variant(
        &self,
        meme: &Meme,
        outcome: &ScreenOutcome,
        posture: GuardianPosture,
        mutation: f32,
        child_id: u64,
        seed: u64,
    ) -> Option<Meme> {
        if !outcome.accepted
            || posture == GuardianPosture::Red
            || outcome.contagion < PROPAGATION_MIN_CONTAGION
        {
            return None;
        }
        Some(meme.transmit(child_id, mutation, seed))
    }
}

/// The resonance-gain a single sample would contribute — exposed so callers can
/// reason about adoptability without a full [`MemeticImmuneSystem`].
pub fn self_resonance_gain(meme: &Meme, self_state: &BinaryHV) -> f32 {
    resonance_gain(meme.payload.similarity(self_state))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meme_from(hv: BinaryHV) -> Meme {
        Meme::seed(0, hv, 0.9)
    }

    #[test]
    fn red_posture_rejects_everything() {
        let state = BinaryHV::random(1);
        let mut immune = MemeticImmuneSystem::new(state.clone(), 1.0);
        // A perfectly resonant, non-pathogen meme — normally admitted.
        let m = meme_from(state.clone());
        let normal = immune.screen(&m, GuardianPosture::Green, 0.0);
        assert!(normal.accepted);
        let locked = immune.screen(&m, GuardianPosture::Red, 0.0);
        assert!(!locked.accepted, "Red must suppress all uptake");
    }

    #[test]
    fn rejects_pathogen_admits_novelty() {
        let state = BinaryHV::random(1);
        let pathogen = BinaryHV::random(2);
        let mut immune = MemeticImmuneSystem::new(state, 1.0);
        immune.vaccinate(pathogen.clone());

        // A near-copy of the pathogen ⇒ rejected.
        let bad = meme_from(pathogen.add_noise(0.1, 5));
        let r_bad = immune.screen(&bad, GuardianPosture::Green, 0.0);
        assert!(r_bad.threat_match >= BASE_THREAT_THRESHOLD && !r_bad.accepted);

        // A novel idea unrelated to the pathogen ⇒ admitted.
        let novel = meme_from(BinaryHV::random(99));
        let r_novel = immune.screen(&novel, GuardianPosture::Green, 0.0);
        assert!(r_novel.threat_match < BASE_THREAT_THRESHOLD && r_novel.accepted);
    }

    #[test]
    fn orange_is_stricter_than_green() {
        let state = BinaryHV::random(1);
        let pathogen = BinaryHV::random(2);
        let mut immune = MemeticImmuneSystem::new(state, 1.0);
        immune.vaccinate(pathogen.clone());

        // A partial variant whose match lands between the Orange (0.5) and
        // Green (0.7) thresholds: admitted under Green, rejected under Orange.
        // add_noise(0.35) flips ~17% of bits ⇒ similarity ~0.83 to pathogen...
        // use a bigger mutation to sit in the band.
        let variant = meme_from(pathogen.add_noise(0.6, 7));
        let tm = variant.payload.similarity(&pathogen);
        assert!(
            (ORANGE_THREAT_THRESHOLD..BASE_THREAT_THRESHOLD).contains(&tm),
            "variant threat_match {tm} must sit in the Orange..Green band for this test"
        );

        let mut immune_g = immune;
        let mut immune_o = MemeticImmuneSystem::new(BinaryHV::random(1), 1.0);
        immune_o.vaccinate(pathogen);

        assert!(
            immune_g
                .screen(&variant, GuardianPosture::Green, 0.0)
                .accepted
        );
        assert!(
            !immune_o
                .screen(&variant, GuardianPosture::Orange, 0.0)
                .accepted
        );
    }

    #[test]
    fn arousal_raises_contagion() {
        let state = BinaryHV::random(1);
        let mut immune = MemeticImmuneSystem::new(state.clone(), 1.0);
        let m = meme_from(state); // resonant ⇒ nonzero base adoptability
        let calm = immune.screen(&m, GuardianPosture::Green, 0.0).contagion;
        let excited = immune.screen(&m, GuardianPosture::Green, 1.0).contagion;
        assert!(calm > 0.0, "resonant meme should have nonzero contagion");
        assert!(
            excited > calm,
            "arousal should raise contagion: {excited} > {calm}"
        );
    }

    #[test]
    fn telemetry_tracks_rejections_and_resonance() {
        let state = BinaryHV::random(1);
        let pathogen = BinaryHV::random(2);
        let mut immune = MemeticImmuneSystem::new(state.clone(), 1.0);
        immune.vaccinate(pathogen.clone());

        immune.screen(&meme_from(state.clone()), GuardianPosture::Green, 0.0); // accept, high resonance
        immune.screen(
            &meme_from(pathogen.add_noise(0.1, 3)),
            GuardianPosture::Green,
            0.0,
        ); // reject

        let t = immune.telemetry();
        assert_eq!(t.seen, 2);
        assert_eq!(t.accepted, 1);
        assert_eq!(t.rejected, 1);
        assert!((t.rejection_rate - 0.5).abs() < 1e-9);
        assert_eq!(t.immune_memory, 1);
        assert!(t.mean_resonance > 0.0);
    }

    /// The plan's named Phase-2 repro (component level): feeding a *sequence of
    /// related ideas* — where belief tracks what's accepted — drives the
    /// contagion index up, while an interleaved stream of *unrelated* ideas does
    /// not. This is the emergent "echo-chamber warms up" behaviour the live loop
    /// exhibits once wired, exercised here without the full cognitive loop.
    #[test]
    fn related_idea_stream_raises_contagion() {
        let theme = BinaryHV::random(1234);

        // Stream A: each idea is a small variant of the running theme (related).
        let mut immune_a = MemeticImmuneSystem::new(theme.clone(), 1.0);
        immune_a.set_self_state(theme.clone());
        for i in 0..30 {
            let related = meme_from(theme.add_noise(0.1, 500 + i));
            let out = immune_a.screen(&related, GuardianPosture::Green, 0.3);
            assert!(out.accepted, "related benign idea should be admitted");
        }
        let related_contagion = immune_a.telemetry().contagion_index;

        // Stream B: unrelated random ideas (no shared theme).
        let mut immune_b = MemeticImmuneSystem::new(theme.clone(), 1.0);
        immune_b.set_self_state(theme.clone());
        for i in 0..30 {
            let unrelated = meme_from(BinaryHV::random(9000 + i));
            immune_b.screen(&unrelated, GuardianPosture::Green, 0.3);
        }
        let unrelated_contagion = immune_b.telemetry().contagion_index;

        assert!(
            related_contagion > unrelated_contagion * 2.0,
            "a related-idea stream should be far more contagious: related={related_contagion} vs unrelated={unrelated_contagion}"
        );
        assert!(
            related_contagion > 0.1,
            "related stream should register meaningful contagion, got {related_contagion}"
        );
    }

    /// Re-propagation (Phase 3): a resonant accepted meme yields a mutated
    /// variant to forward; a rejected meme and a lockdown posture yield None.
    #[test]
    fn propagation_variant_forwards_only_worthy_accepted_memes() {
        let state = BinaryHV::random(1);
        let mut immune = MemeticImmuneSystem::new(state.clone(), 1.0);

        // Resonant, non-pathogen ⇒ accepted with real contagion ⇒ forwarded.
        let good = meme_from(state.clone());
        let out = immune.screen(&good, GuardianPosture::Green, 0.5);
        assert!(out.accepted && out.contagion >= 0.15);
        let variant = immune.propagation_variant(&good, &out, GuardianPosture::Green, 0.05, 1, 7);
        let variant = variant.expect("worthy accepted meme should be forwarded");
        assert_eq!(variant.parent, Some(good.id));
        assert_eq!(variant.generation, good.generation + 1);
        // Forwarded copy is mutated (fidelity < 1) but still recognizably related.
        let fid = good.fidelity(&variant);
        assert!(
            fid < 1.0 && fid > 0.8,
            "forwarded variant drifts a little, fid={fid}"
        );

        // Lockdown ⇒ never forward, even a resonant meme.
        assert!(
            immune
                .propagation_variant(&good, &out, GuardianPosture::Red, 0.05, 2, 7)
                .is_none(),
            "Red lockdown must suppress outbound propagation"
        );
    }

    #[test]
    fn propagation_variant_drops_rejected_and_weak() {
        let state = BinaryHV::random(1);
        let pathogen = BinaryHV::random(2);
        let mut immune = MemeticImmuneSystem::new(state, 1.0);
        immune.vaccinate(pathogen.clone());

        // Rejected pathogen ⇒ never forwarded.
        let bad = meme_from(pathogen.add_noise(0.1, 3));
        let out_bad = immune.screen(&bad, GuardianPosture::Green, 0.5);
        assert!(!out_bad.accepted);
        assert!(
            immune
                .propagation_variant(&bad, &out_bad, GuardianPosture::Green, 0.05, 1, 7)
                .is_none(),
            "a rejected pathogen must never be re-propagated"
        );

        // Accepted but weak (near-chance resonance, no arousal) ⇒ not forwarded.
        let mut immune2 = MemeticImmuneSystem::new(BinaryHV::random(50), 1.0);
        let weak = meme_from(BinaryHV::random(51)); // unrelated ⇒ ~0 contagion
        let out_weak = immune2.screen(&weak, GuardianPosture::Green, 0.0);
        assert!(out_weak.accepted && out_weak.contagion < 0.15);
        assert!(
            immune2
                .propagation_variant(&weak, &out_weak, GuardianPosture::Green, 0.05, 1, 7)
                .is_none(),
            "a weak/noise idea should be adopted silently, not amplified"
        );
    }

    // ── WardConfig / posture floor (Warded Node design, Phase 1) ──

    #[test]
    fn posture_ordering_is_ascending_defensiveness() {
        assert!(GuardianPosture::Green < GuardianPosture::Yellow);
        assert!(GuardianPosture::Yellow < GuardianPosture::Orange);
        assert!(GuardianPosture::Orange < GuardianPosture::Red);
    }

    #[test]
    fn no_floor_is_unchanged_behavior() {
        // Default WardConfig (no floor) must reproduce plain screen() exactly —
        // a regression guard so adult/unwarded nodes are untouched by this change.
        let state = BinaryHV::random(1);
        let mut immune = MemeticImmuneSystem::new(state.clone(), 1.0);
        assert_eq!(immune.ward_config(), WardConfig::default());

        let m = meme_from(state);
        let out = immune.screen(&m, GuardianPosture::Green, 0.3);
        assert!(out.accepted, "no floor ⇒ Green posture behaves as Green");
    }

    #[test]
    fn floor_tightens_a_looser_incoming_posture() {
        let state = BinaryHV::random(1);
        let mut immune = MemeticImmuneSystem::new(state.clone(), 1.0);
        immune.set_ward_config(WardConfig {
            posture_floor: Some(GuardianPosture::Red),
            ..Default::default()
        });

        // Caller derives Green (e.g. from psi) — but the warded node's Red
        // floor overrides it: ALL uptake suppressed, exactly like Red.
        let m = meme_from(state);
        let out = immune.screen(&m, GuardianPosture::Green, 0.0);
        assert!(
            !out.accepted,
            "a Red floor must suppress uptake even when the caller passes Green"
        );
        assert_eq!(
            out.reason,
            "guardian lockdown (Red): all meme uptake suppressed"
        );
    }

    #[test]
    fn floor_never_loosens_a_stricter_incoming_posture() {
        let state = BinaryHV::random(1);
        let mut immune = MemeticImmuneSystem::new(state.clone(), 1.0);
        immune.set_ward_config(WardConfig {
            posture_floor: Some(GuardianPosture::Orange),
            ..Default::default()
        });

        // Caller passes Red (e.g. a real safety lockdown) — the Orange floor
        // must NOT relax it back down to Orange. max(Red, Orange) = Red.
        let m = meme_from(state);
        let out = immune.screen(&m, GuardianPosture::Red, 0.0);
        assert!(
            !out.accepted,
            "floor must never loosen an already-strict posture"
        );
        assert_eq!(
            out.reason,
            "guardian lockdown (Red): all meme uptake suppressed"
        );
    }

    #[test]
    fn orange_floor_tightens_pathogen_threshold_under_green() {
        // A variant that sits between the Orange (0.5) and Green/base (0.7)
        // threat-match thresholds: admitted under a plain Green posture,
        // rejected once an Orange floor is set — proving the floor actually
        // changes accept/reject outcomes, not just the reported posture.
        let state = BinaryHV::random(1);
        let pathogen = BinaryHV::random(2);

        let mut unwarded = MemeticImmuneSystem::new(state.clone(), 1.0);
        unwarded.vaccinate(pathogen.clone());
        let mut warded = MemeticImmuneSystem::new(state, 1.0);
        warded.vaccinate(pathogen.clone());
        warded.set_ward_config(WardConfig {
            posture_floor: Some(GuardianPosture::Orange),
            ..Default::default()
        });

        let variant = meme_from(pathogen.add_noise(0.6, 7));
        let tm = variant.payload.similarity(&pathogen);
        assert!(
            (ORANGE_THREAT_THRESHOLD..BASE_THREAT_THRESHOLD).contains(&tm),
            "variant threat_match {tm} must sit in the Orange..Green band for this test"
        );

        assert!(
            unwarded
                .screen(&variant, GuardianPosture::Green, 0.0)
                .accepted
        );
        assert!(
            !warded
                .screen(&variant, GuardianPosture::Green, 0.0)
                .accepted
        );
    }

    #[test]
    fn ward_config_getter_reflects_setter() {
        let mut immune = MemeticImmuneSystem::new(BinaryHV::random(1), 1.0);
        let cfg = WardConfig {
            posture_floor: Some(GuardianPosture::Yellow),
            ..Default::default()
        };
        immune.set_ward_config(cfg);
        assert_eq!(immune.ward_config(), cfg);
    }

    // ── Filtered-items audit log (Warded Node design, Phase 2) ──

    #[test]
    fn empty_log_for_a_node_that_never_rejected() {
        let state = BinaryHV::random(1);
        let mut immune = MemeticImmuneSystem::new(state.clone(), 1.0);
        assert_eq!(immune.filtered_log_len(), 0);
        assert!(immune.filtered_log(10).is_empty());

        // Admitting content must NOT add to the log.
        immune.screen(&meme_from(state), GuardianPosture::Green, 0.0);
        assert_eq!(immune.filtered_log_len(), 0);
    }

    #[test]
    fn rejection_is_logged_with_matching_details() {
        let state = BinaryHV::random(1);
        let pathogen = BinaryHV::random(2);
        let mut immune = MemeticImmuneSystem::new(state, 1.0);
        immune.vaccinate(pathogen.clone());

        let bad = Meme::seed(777, pathogen.add_noise(0.1, 3), 0.9);
        let outcome = immune.screen(&bad, GuardianPosture::Green, 0.0);
        assert!(!outcome.accepted);

        assert_eq!(immune.filtered_log_len(), 1);
        let entry = immune.filtered_log(10)[0];
        assert_eq!(entry.meme_id, 777, "log entry must correlate via meme_id");
        assert_eq!(entry.reason, outcome.reason);
        assert!((entry.threat_match - outcome.threat_match).abs() < 1e-6);
        assert!((entry.resonance - outcome.resonance).abs() < 1e-6);
    }

    #[test]
    fn log_is_newest_first_and_bounded() {
        let state = BinaryHV::random(1);
        let mut immune = MemeticImmuneSystem::new(state, 1.0);
        // Force every screen() call to reject: Red lockdown.
        immune.set_ward_config(WardConfig {
            posture_floor: Some(GuardianPosture::Red),
            ..Default::default()
        });

        for i in 0..(FILTERED_LOG_CAPACITY as u64 + 10) {
            let m = Meme::seed(i, BinaryHV::random(i.wrapping_add(1000)), 0.9);
            immune.screen(&m, GuardianPosture::Green, 0.0);
        }

        // Bounded: never exceeds capacity even though we rejected more.
        assert_eq!(immune.filtered_log_len(), FILTERED_LOG_CAPACITY);
        // Newest-first: the most recent id is first.
        let recent = immune.filtered_log(3);
        assert_eq!(recent[0].meme_id, FILTERED_LOG_CAPACITY as u64 + 9);
        assert_eq!(recent[1].meme_id, FILTERED_LOG_CAPACITY as u64 + 8);
        assert_eq!(recent[2].meme_id, FILTERED_LOG_CAPACITY as u64 + 7);

        // `limit` truncates without panicking, including limit > len.
        assert_eq!(immune.filtered_log(1_000_000).len(), FILTERED_LOG_CAPACITY);
        assert_eq!(immune.filtered_log(0).len(), 0);
    }

    // ── AllowlistMode (Warded Node design, Phase 3) ──

    #[test]
    fn open_mode_admits_regardless_of_trust_score() {
        let open = AllowlistMode::Open;
        assert!(
            open.admits(0.0),
            "Open must admit a total stranger (score 0.0)"
        );
        assert!(open.admits(1.0));
        assert!(
            open.admits(-1.0),
            "Open ignores the score entirely, even nonsensical ones"
        );
    }

    #[test]
    fn allowlist_only_denies_by_default_fails_closed() {
        let strict = AllowlistMode::AllowlistOnly { min_trust: 0.5 };
        // The critical safety property: an unknown peer (no trust data at
        // all, score 0.0 by convention) is DENIED, not admitted.
        assert!(
            !strict.admits(0.0),
            "AllowlistOnly must fail closed for an untrusted/unknown peer"
        );
    }

    #[test]
    fn allowlist_only_admits_at_or_above_threshold() {
        let strict = AllowlistMode::AllowlistOnly { min_trust: 0.5 };
        assert!(!strict.admits(0.49));
        assert!(strict.admits(0.5), "exactly at the threshold must admit");
        assert!(strict.admits(0.9));
    }

    #[test]
    fn allowlist_mode_default_is_open() {
        assert_eq!(AllowlistMode::default(), AllowlistMode::Open);
        assert_eq!(WardConfig::default().allowlist_mode, AllowlistMode::Open);
    }

    #[test]
    fn ward_config_allowlist_mode_roundtrips_through_immune_system() {
        let mut immune = MemeticImmuneSystem::new(BinaryHV::random(1), 1.0);
        let cfg = WardConfig {
            posture_floor: None,
            allowlist_mode: AllowlistMode::AllowlistOnly { min_trust: 0.7 },
        };
        immune.set_ward_config(cfg);
        assert_eq!(immune.ward_config(), cfg);
        assert_eq!(
            immune.ward_config().allowlist_mode,
            AllowlistMode::AllowlistOnly { min_trust: 0.7 }
        );
    }

    #[test]
    fn gate_denial_lands_in_audit_log_without_touching_screen_counters() {
        let mut immune = MemeticImmuneSystem::new(BinaryHV::random(1), 1.0);
        immune.log_gate_denial(42, "allowlist gate: peer trust below threshold");

        let t = immune.telemetry();
        assert_eq!(
            t.seen, 0,
            "a gate denial is not a screen() call — seen must be untouched"
        );
        assert_eq!(
            t.rejected, 0,
            "screen()'s own rejected counter must be untouched"
        );
        assert_eq!(t.accepted, 0);

        assert_eq!(immune.filtered_log_len(), 1);
        let entry = immune.filtered_log(1)[0];
        assert_eq!(entry.meme_id, 42);
        assert_eq!(entry.reason, "allowlist gate: peer trust below threshold");
        assert_eq!(entry.resonance, 0.0);
        assert_eq!(entry.threat_match, 0.0);
        assert_eq!(entry.contagion, 0.0);
    }

    #[test]
    fn gate_denials_and_screen_rejections_share_the_same_audit_log() {
        let state = BinaryHV::random(1);
        let pathogen = BinaryHV::random(2);
        let mut immune = MemeticImmuneSystem::new(state, 1.0);
        immune.vaccinate(pathogen.clone());

        immune.log_gate_denial(1, "allowlist gate: peer trust below threshold");
        immune.screen(
            &Meme::seed(2, pathogen.add_noise(0.1, 3), 0.9),
            GuardianPosture::Green,
            0.0,
        );

        // Both kinds of rejection are visible to a guardian in one log,
        // newest first.
        assert_eq!(immune.filtered_log_len(), 2);
        let log = immune.filtered_log(10);
        assert_eq!(log[0].meme_id, 2, "screen() rejection is newest");
        assert_eq!(log[1].meme_id, 1, "gate denial is oldest");
    }
}
