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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
        }
    }

    /// Vaccinate against a known memetic pathogen (mutation-tolerant recognition).
    pub fn vaccinate(&mut self, pathogen: BinaryHV) {
        self.threat_signatures.push(pathogen);
    }

    /// Update the agent's belief state (called each cycle from consciousness).
    pub fn set_self_state(&mut self, state: BinaryHV) {
        self.self_state = state;
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
    pub fn screen(&mut self, meme: &Meme, posture: GuardianPosture, arousal: f32) -> ScreenOutcome {
        self.seen += 1;

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
        }
        outcome
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
}
