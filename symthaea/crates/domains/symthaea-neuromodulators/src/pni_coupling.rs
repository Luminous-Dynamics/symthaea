// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Psychoneuroimmunology (PNI) coupling: bidirectional communication
//! between the immune system and the nervous/consciousness systems.
//!
//! Models the cytokine-neuromodulator interface where:
//! - Immune activation (threat detection) → pro-inflammatory cytokines →
//!   serotonin depletion, adenosine increase, fatigue behavior
//! - Chronic stress (cortisol) → immune suppression → vulnerability
//! - Sleep/recovery → anti-inflammatory cytokines → repair
//!
//! References:
//! - Besedovsky, H. O. & del Rey, A. (1996). Immune-neuro-endocrine interactions.
//! - Dantzer, R. (2009). Cytokine, sickness behavior, and depression.
//! - Slavich, G. M. & Irwin, M. R. (2014). Social stress to inflammation.
//! - Raison, C. L. & Miller, A. H. (2011). Inflammation in depression.
//! - Besedovsky et al. (2012). Sleep and immune function.

use serde::{Deserialize, Serialize};

// ── Named constants ──────────────────────────────────────────────────────

/// Rate at which cortisol suppresses anti-inflammatory IL-10 production.
/// Science: Sapolsky (2004) — glucocorticoid immunosuppression.
const CORTISOL_SUPPRESSION_RATE: f32 = 0.3;

/// Natural cytokine decay rate per tick (exponential decay toward baseline).
/// Science: Dantzer (2009) — cytokine half-life in hours; scaled to cognitive ticks.
const CYTOKINE_DECAY_RATE: f32 = 0.05;

/// Inflammation level above which sickness behavior manifests.
/// Science: Dantzer (2009) — cytokine-induced sickness behavior threshold.
const SICKNESS_THRESHOLD: f32 = 0.2;

/// Maximum consciousness cost from inflammation.
/// Even severe inflammation doesn't fully abolish consciousness.
/// Science: Raison & Miller (2011) — inflammation reduces but doesn't eliminate Phi.
const MAX_CONSCIOUSNESS_COST: f64 = 0.3;

/// Rate at which sickness behavior tracks inflammation (EMA alpha).
const SICKNESS_TRACKING_RATE: f32 = 0.1;

/// Sleep recovery multiplier for anti-inflammatory cytokines.
/// Science: Besedovsky et al. (2012) — immune maintenance during sleep.
const SLEEP_RECOVERY_BOOST: f32 = 0.15;

/// Pro-inflammatory reduction during sleep.
const SLEEP_ANTI_INFLAMMATORY: f32 = 0.08;

/// Chronic allostatic load baseline IL-6 elevation.
/// Science: Slavich & Irwin (2014) — chronic social stress → systemic inflammation.
const CHRONIC_INFLAMMATION_RATE: f32 = 0.2;

/// TNF-alpha acute rise coefficient (fast on, fast off).
const TNF_ACUTE_COEFFICIENT: f32 = 0.4;

/// TNF-alpha extra decay rate (faster than other cytokines).
const TNF_EXTRA_DECAY: f32 = 0.03;

/// IL-6 threat accumulation rate (proportional to threat × duration).
const IL6_THREAT_RATE: f32 = 0.05;

/// IFN-gamma sustained-threat threshold (cycles before IFN-gamma rises).
const IFN_GAMMA_ONSET_CYCLES: u32 = 5;

/// IFN-gamma rise rate for sustained threats.
const IFN_GAMMA_RATE: f32 = 0.03;

// ── Types ────────────────────────────────────────────────────────────────

/// Cytokine concentrations modeling the immune system's signaling molecules.
///
/// Pro-inflammatory (IL-6, TNF-alpha, IL-1beta, IFN-gamma) drive sickness
/// behavior and neuromodulator changes. Anti-inflammatory (IL-10, TGF-beta)
/// promote recovery and repair.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CytokineState {
    /// Interleukin-6: pro-inflammatory, rises with threat and chronic stress.
    /// Science: Slavich & Irwin (2014) — IL-6 as stress-inflammation mediator.
    pub il6: f32,
    /// Tumor necrosis factor alpha: acute pro-inflammatory (fast rise, fast decay).
    /// Science: Dantzer (2009) — TNF-alpha in acute sickness response.
    pub tnf_alpha: f32,
    /// Interleukin-1 beta: pro-inflammatory, drives fever and arousal.
    /// Science: Besedovsky & del Rey (1996) — IL-1beta → HPA axis activation.
    pub il1_beta: f32,
    /// Interleukin-10: anti-inflammatory, promotes recovery and repair.
    /// Science: Couper et al. (2008) — IL-10 as master immune regulator.
    pub il10: f32,
    /// Transforming growth factor beta: anti-inflammatory, tissue repair.
    pub tgf_beta: f32,
    /// Interferon gamma: sustained immune activation marker.
    /// Science: Schoenborn & Wilson (2007) — IFN-gamma in adaptive immunity.
    pub interferon_gamma: f32,
}

impl Default for CytokineState {
    fn default() -> Self {
        Self {
            il6: 0.0,
            tnf_alpha: 0.0,
            il1_beta: 0.0,
            il10: 0.2, // Baseline anti-inflammatory tone
            tgf_beta: 0.1,
            interferon_gamma: 0.0,
        }
    }
}

impl CytokineState {
    /// Clamp all cytokine values to [0.0, 1.0].
    fn clamp_all(&mut self) {
        self.il6 = self.il6.clamp(0.0, 1.0);
        self.tnf_alpha = self.tnf_alpha.clamp(0.0, 1.0);
        self.il1_beta = self.il1_beta.clamp(0.0, 1.0);
        self.il10 = self.il10.clamp(0.0, 1.0);
        self.tgf_beta = self.tgf_beta.clamp(0.0, 1.0);
        self.interferon_gamma = self.interferon_gamma.clamp(0.0, 1.0);
    }
}

/// Neuromodulator deltas produced by cytokine signaling.
///
/// These are additive adjustments to be applied to the `NeuromodulatorBath`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NeuromodDeltas {
    /// Serotonin change (typically negative from inflammation).
    /// Science: Dantzer (2009) — IL-6 activates IDO, depleting tryptophan → 5-HT.
    pub delta_serotonin: f32,
    /// Dopamine change (reduced by sickness behavior).
    /// Science: Felger & Treadway (2017) — inflammation → reduced DA signaling.
    pub delta_dopamine: f32,
    /// Noradrenaline change (increased by IL-1beta acute response).
    /// Science: Besedovsky & del Rey (1996) — cytokine → NE via vagus nerve.
    pub delta_noradrenaline: f32,
    /// Adenosine change (fatigue signal from TNF-alpha).
    pub delta_adenosine: f32,
    /// Oxytocin change (boosted by anti-inflammatory IL-10).
    /// Science: Li et al. (2017) — oxytocin's anti-inflammatory properties.
    pub delta_oxytocin: f32,
}

/// Full psychoneuroimmunology state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PniState {
    /// Current cytokine concentrations.
    pub cytokines: CytokineState,
    /// Net inflammation index: pro-inflammatory minus anti-inflammatory.
    pub inflammation_index: f32,
    /// Sickness behavior intensity: 0.0 (healthy) to 1.0 (full sickness).
    /// Tracks inflammation with delay (EMA smoothing).
    pub sickness_behavior: f32,
    /// Immune system competence: 1.0 (healthy) to 0.0 (immunosuppressed).
    /// Reduced by chronic cortisol, elevated allostatic load.
    pub immune_competence: f32,
    /// Phi reduction from inflammation.
    /// Science: Raison & Miller (2011) — inflammation disrupts information integration.
    pub consciousness_inflammation_cost: f64,
    /// Current recovery rate (modulated by sleep and anti-inflammatory tone).
    pub recovery_rate: f32,
}

impl Default for PniState {
    fn default() -> Self {
        Self {
            cytokines: CytokineState::default(),
            inflammation_index: 0.0,
            sickness_behavior: 0.0,
            immune_competence: 1.0,
            consciousness_inflammation_cost: 0.0,
            recovery_rate: 0.0,
        }
    }
}

/// Psychoneuroimmunology coupling engine.
///
/// Bridges the immune/sentinel system to the neuromodulator bath and
/// consciousness pipeline. Implements bidirectional cytokine-neural signaling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PniCoupling {
    /// Current PNI state.
    pub state: PniState,
}

impl Default for PniCoupling {
    fn default() -> Self {
        Self::new()
    }
}

impl PniCoupling {
    /// Create a new PNI coupling engine with healthy defaults.
    pub fn new() -> Self {
        Self {
            state: PniState::default(),
        }
    }

    /// Update cytokines from threat detection (immune activation).
    ///
    /// Threat level and duration drive pro-inflammatory cytokine release:
    /// - IL-6 rises proportional to threat × duration (chronic inflammation pathway)
    /// - TNF-alpha rises fast but decays fast (acute phase response)
    /// - IL-1beta rises with threat (fever/arousal pathway)
    /// - IFN-gamma activates for sustained threats (>5 cycles)
    ///
    /// Science: Slavich & Irwin (2014) — threat perception → inflammatory cascade.
    pub fn update_from_threat(&mut self, threat_level: f32, threat_duration: u32) {
        let threat = threat_level.clamp(0.0, 1.0);
        let duration = threat_duration.min(100);

        // IL-6: accumulates with threat × duration (chronic pathway)
        self.state.cytokines.il6 += IL6_THREAT_RATE * threat * (duration as f32).sqrt();

        // TNF-alpha: acute phase, proportional to current threat only
        self.state.cytokines.tnf_alpha += TNF_ACUTE_COEFFICIENT * threat;

        // IL-1beta: moderate rise, drives NE via vagus nerve
        self.state.cytokines.il1_beta += 0.03 * threat;

        // IFN-gamma: only for sustained threats (adaptive immunity)
        if duration > IFN_GAMMA_ONSET_CYCLES {
            self.state.cytokines.interferon_gamma += IFN_GAMMA_RATE * threat;
        }

        self.state.cytokines.clamp_all();
        self.recompute_derived();
    }

    /// Update from stress hormones (cortisol → immune suppression).
    ///
    /// - High cortisol → reduced IL-10 production (immunosuppression)
    /// - High allostatic load → chronic IL-6 baseline elevation
    /// - immune_competence = 1.0 - 0.4 * cortisol - 0.3 * allostatic_load
    ///
    /// Science: Sapolsky (2004) — glucocorticoid paradox (acute anti-inflammatory,
    /// chronic pro-inflammatory). McEwen (1998) — allostatic overload.
    pub fn update_from_stress(&mut self, cortisol: f32, allostatic_load: f32) {
        let cortisol = cortisol.clamp(0.0, 1.0);
        let load = allostatic_load.clamp(0.0, 1.0);

        // Cortisol suppresses anti-inflammatory IL-10 production
        self.state.cytokines.il10 -= CORTISOL_SUPPRESSION_RATE * cortisol;

        // Chronic allostatic load elevates baseline IL-6
        self.state.cytokines.il6 += CHRONIC_INFLAMMATION_RATE * load;

        // Immune competence degrades under stress
        self.state.immune_competence = (1.0 - 0.4 * cortisol - 0.3 * load).clamp(0.0, 1.0);

        self.state.cytokines.clamp_all();
        self.recompute_derived();
    }

    /// Update from sleep state (recovery pathway).
    ///
    /// Sleep boosts anti-inflammatory cytokines and reduces pro-inflammatory ones.
    /// Quality modulates the strength of recovery.
    ///
    /// Science: Besedovsky et al. (2012) — sleep promotes immune maintenance;
    /// Irwin (2015) — sleep disturbance → inflammation.
    pub fn update_from_sleep(&mut self, is_sleeping: bool, sleep_quality: f32) {
        if !is_sleeping {
            return;
        }
        let quality = sleep_quality.clamp(0.0, 1.0);

        // Boost anti-inflammatory during sleep
        self.state.cytokines.il10 += SLEEP_RECOVERY_BOOST * quality;
        self.state.cytokines.tgf_beta += 0.08 * quality;

        // Reduce pro-inflammatory during sleep
        self.state.cytokines.il6 -= SLEEP_ANTI_INFLAMMATORY * quality;
        self.state.cytokines.tnf_alpha -= SLEEP_ANTI_INFLAMMATORY * quality;
        self.state.cytokines.il1_beta -= 0.05 * quality;
        self.state.cytokines.interferon_gamma -= 0.04 * quality;

        // Recovery rate boosted
        self.state.recovery_rate = (0.1 * quality).clamp(0.0, 1.0);

        self.state.cytokines.clamp_all();
        self.recompute_derived();
    }

    /// Compute neuromodulator deltas from current cytokine state.
    ///
    /// Maps immune signaling molecules to neurotransmitter changes:
    /// - IL-6 → serotonin depletion (IDO pathway, Dantzer 2009)
    /// - TNF-alpha → adenosine/fatigue
    /// - IL-1beta → noradrenaline (vagal afferent, Besedovsky 1996)
    /// - IL-10 → oxytocin boost (anti-inflammatory social bond)
    /// - Sickness behavior → dopamine reduction (motivational anhedonia)
    ///
    /// Science: Felger & Treadway (2017) — inflammation → reduced DA signaling.
    pub fn compute_neuromod_deltas(&self) -> NeuromodDeltas {
        let c = &self.state.cytokines;
        NeuromodDeltas {
            // IL-6 activates indoleamine 2,3-dioxygenase (IDO), depleting tryptophan
            delta_serotonin: -0.15 * c.il6,
            // Sickness behavior → motivational anhedonia
            delta_dopamine: -0.1 * self.state.sickness_behavior,
            // IL-1beta → NE via vagus nerve afferents
            delta_noradrenaline: 0.08 * c.il1_beta,
            // TNF-alpha → fatigue / adenosine accumulation
            delta_adenosine: 0.1 * c.tnf_alpha,
            // IL-10 → social bonding / repair signaling
            delta_oxytocin: 0.05 * c.il10,
        }
    }

    /// Compute consciousness cost from inflammation.
    ///
    /// Inflammation disrupts information integration (Phi) proportionally,
    /// capped at `MAX_CONSCIOUSNESS_COST` (0.3).
    ///
    /// Science: Raison & Miller (2011) — inflammation disrupts neural integration.
    pub fn compute_consciousness_cost(&self) -> f64 {
        let cost = 0.15 * self.state.inflammation_index.max(0.0) as f64;
        cost.min(MAX_CONSCIOUSNESS_COST)
    }

    /// Advance cytokine dynamics by one tick.
    ///
    /// Natural exponential decay of all cytokines toward baseline.
    /// TNF-alpha decays faster than others (acute response kinetics).
    /// Sickness behavior tracks inflammation with EMA smoothing.
    pub fn tick(&mut self, _dt: f32) {
        let c = &mut self.state.cytokines;

        // Exponential decay toward baseline
        c.il6 *= 1.0 - CYTOKINE_DECAY_RATE;
        c.tnf_alpha *= 1.0 - CYTOKINE_DECAY_RATE - TNF_EXTRA_DECAY;
        c.il1_beta *= 1.0 - CYTOKINE_DECAY_RATE;
        c.interferon_gamma *= 1.0 - CYTOKINE_DECAY_RATE;

        // Anti-inflammatory decay toward baseline (slower)
        let il10_baseline = 0.2_f32;
        c.il10 += (il10_baseline - c.il10) * 0.02;
        let tgf_baseline = 0.1_f32;
        c.tgf_beta += (tgf_baseline - c.tgf_beta) * 0.02;

        c.clamp_all();
        self.recompute_derived();
    }

    /// Compute the net inflammation index.
    ///
    /// Positive = pro-inflammatory dominance; negative = anti-inflammatory.
    pub fn inflammation_index(&self) -> f32 {
        self.state.inflammation_index
    }

    /// Recompute derived state from cytokines.
    fn recompute_derived(&mut self) {
        let c = &self.state.cytokines;

        // Net inflammation: pro minus anti
        self.state.inflammation_index =
            (c.il6 + c.tnf_alpha + c.il1_beta) / 3.0 - (c.il10 + c.tgf_beta) / 2.0;

        // Sickness behavior tracks inflammation via EMA (sluggish onset/offset)
        let target_sickness = if self.state.inflammation_index > SICKNESS_THRESHOLD {
            ((self.state.inflammation_index - SICKNESS_THRESHOLD) * 2.0).clamp(0.0, 1.0)
        } else {
            0.0
        };
        self.state.sickness_behavior +=
            SICKNESS_TRACKING_RATE * (target_sickness - self.state.sickness_behavior);
        self.state.sickness_behavior = self.state.sickness_behavior.clamp(0.0, 1.0);

        // Consciousness cost
        self.state.consciousness_inflammation_cost = self.compute_consciousness_cost();
    }

    /// Access current PNI state (read-only).
    pub fn pni_state(&self) -> &PniState {
        &self.state
    }

    /// Access current cytokine state (read-only).
    pub fn cytokines(&self) -> &CytokineState {
        &self.state.cytokines
    }

    /// Whether the system is currently in sickness behavior.
    pub fn is_sick(&self) -> bool {
        self.state.sickness_behavior > 0.1
    }

    /// Immune competence level.
    pub fn immune_competence(&self) -> f32 {
        self.state.immune_competence
    }
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn healthy_state_has_zero_inflammation_cost() {
        let pni = PniCoupling::new();
        assert!(
            pni.state.consciousness_inflammation_cost.abs() < 1e-6,
            "healthy state should have ~zero consciousness cost, got {}",
            pni.state.consciousness_inflammation_cost
        );
        assert!(pni.state.inflammation_index <= 0.0);
    }

    #[test]
    fn threat_raises_il6_and_tnf_alpha() {
        let mut pni = PniCoupling::new();
        let il6_before = pni.state.cytokines.il6;
        let tnf_before = pni.state.cytokines.tnf_alpha;

        pni.update_from_threat(0.8, 10);

        assert!(
            pni.state.cytokines.il6 > il6_before,
            "IL-6 should rise on threat"
        );
        assert!(
            pni.state.cytokines.tnf_alpha > tnf_before,
            "TNF-alpha should rise on threat"
        );
    }

    #[test]
    fn cortisol_suppresses_immune_competence() {
        let mut pni = PniCoupling::new();
        assert!((pni.state.immune_competence - 1.0).abs() < 1e-6);

        pni.update_from_stress(0.8, 0.5);

        // immune_competence = 1.0 - 0.4*0.8 - 0.3*0.5 = 1.0 - 0.32 - 0.15 = 0.53
        assert!(
            pni.state.immune_competence < 0.6,
            "high cortisol + allostatic load should suppress immune competence, got {}",
            pni.state.immune_competence
        );
    }

    #[test]
    fn sleep_reduces_inflammation() {
        let mut pni = PniCoupling::new();
        // First inflame
        pni.update_from_threat(0.9, 10);
        let inflamed = pni.state.cytokines.il6;

        // Now sleep
        pni.update_from_sleep(true, 0.9);

        assert!(
            pni.state.cytokines.il6 < inflamed,
            "sleep should reduce IL-6"
        );
        assert!(
            pni.state.cytokines.il10 > 0.2,
            "sleep should boost IL-10 above baseline"
        );
    }

    #[test]
    fn high_inflammation_depletes_serotonin() {
        let mut pni = PniCoupling::new();
        pni.update_from_threat(1.0, 20);

        let deltas = pni.compute_neuromod_deltas();
        assert!(
            deltas.delta_serotonin < 0.0,
            "high IL-6 should deplete serotonin, got {}",
            deltas.delta_serotonin
        );
    }

    #[test]
    fn sickness_behavior_reduces_dopamine() {
        let mut pni = PniCoupling::new();
        // Drive high inflammation to trigger sickness behavior
        for _ in 0..50 {
            pni.update_from_threat(1.0, 30);
        }
        assert!(
            pni.state.sickness_behavior > 0.0,
            "repeated threats should induce sickness behavior"
        );

        let deltas = pni.compute_neuromod_deltas();
        assert!(
            deltas.delta_dopamine < 0.0,
            "sickness behavior should reduce dopamine"
        );
    }

    #[test]
    fn anti_inflammatory_cytokines_boost_recovery() {
        let mut pni = PniCoupling::new();
        // Sleep to boost anti-inflammatory
        pni.update_from_sleep(true, 1.0);

        assert!(
            pni.state.cytokines.il10 > 0.3,
            "sleep should boost IL-10, got {}",
            pni.state.cytokines.il10
        );
        assert!(
            pni.state.recovery_rate > 0.0,
            "should have positive recovery rate during sleep"
        );
    }

    #[test]
    fn consciousness_cost_proportional_to_inflammation() {
        let mut pni = PniCoupling::new();

        // Low inflammation
        pni.update_from_threat(0.3, 3);
        let low_cost = pni.state.consciousness_inflammation_cost;

        // Reset and high inflammation
        let mut pni2 = PniCoupling::new();
        pni2.update_from_threat(1.0, 20);
        let high_cost = pni2.state.consciousness_inflammation_cost;

        assert!(
            high_cost > low_cost,
            "higher inflammation should cost more consciousness: low={low_cost}, high={high_cost}"
        );
    }

    #[test]
    fn consciousness_cost_capped_at_max() {
        let mut pni = PniCoupling::new();
        // Extreme inflammation
        for _ in 0..100 {
            pni.update_from_threat(1.0, 50);
        }

        assert!(
            pni.state.consciousness_inflammation_cost <= MAX_CONSCIOUSNESS_COST,
            "consciousness cost should be capped at {MAX_CONSCIOUSNESS_COST}, got {}",
            pni.state.consciousness_inflammation_cost
        );
    }

    #[test]
    fn chronic_stress_elevates_baseline_il6() {
        let mut pni = PniCoupling::new();
        let il6_before = pni.state.cytokines.il6;

        // High allostatic load → chronic IL-6 elevation
        pni.update_from_stress(0.3, 0.9);

        assert!(
            pni.state.cytokines.il6 > il6_before,
            "high allostatic load should elevate IL-6"
        );
    }

    #[test]
    fn cytokines_decay_naturally_over_time() {
        let mut pni = PniCoupling::new();
        pni.update_from_threat(0.8, 10);
        let il6_peak = pni.state.cytokines.il6;
        let tnf_peak = pni.state.cytokines.tnf_alpha;

        // Let cytokines decay
        for _ in 0..50 {
            pni.tick(1.0);
        }

        assert!(
            pni.state.cytokines.il6 < il6_peak,
            "IL-6 should decay over time"
        );
        assert!(
            pni.state.cytokines.tnf_alpha < tnf_peak,
            "TNF-alpha should decay over time"
        );
        // TNF decays faster
        let il6_ratio = pni.state.cytokines.il6 / il6_peak;
        let tnf_ratio = pni.state.cytokines.tnf_alpha / tnf_peak;
        assert!(
            tnf_ratio < il6_ratio,
            "TNF-alpha should decay faster than IL-6: tnf_ratio={tnf_ratio}, il6_ratio={il6_ratio}"
        );
    }

    #[test]
    fn full_cycle_threat_inflammation_sickness_sleep_recovery() {
        let mut pni = PniCoupling::new();

        // Phase 1: Threat → inflammation
        for _ in 0..20 {
            pni.update_from_threat(0.7, 15);
            pni.tick(1.0);
        }
        assert!(
            pni.state.inflammation_index > 0.0,
            "should be inflamed after threat"
        );
        let peak_cost = pni.state.consciousness_inflammation_cost;
        assert!(peak_cost > 0.0, "should have consciousness cost");

        // Phase 2: Sickness behavior develops
        for _ in 0..30 {
            pni.tick(1.0);
        }
        // Sickness behavior may have developed or may be fading; check cost decreased
        let post_threat_cost = pni.state.consciousness_inflammation_cost;

        // Phase 3: Sleep recovery
        for _ in 0..50 {
            pni.update_from_sleep(true, 0.9);
            pni.tick(1.0);
        }
        let post_sleep_cost = pni.state.consciousness_inflammation_cost;

        assert!(
            post_sleep_cost < peak_cost,
            "sleep should reduce consciousness cost: peak={peak_cost}, after_sleep={post_sleep_cost}"
        );
        // Verify anti-inflammatory dominance after recovery
        assert!(
            pni.state.cytokines.il10 > pni.state.cytokines.il6,
            "after recovery, IL-10 should exceed IL-6"
        );
    }

    #[test]
    fn not_sleeping_has_no_effect() {
        let mut pni = PniCoupling::new();
        pni.update_from_threat(0.5, 5);
        let state_before = pni.state.clone();

        pni.update_from_sleep(false, 1.0);

        assert!(
            (pni.state.cytokines.il6 - state_before.cytokines.il6).abs() < 1e-6,
            "not sleeping should not change cytokines"
        );
    }

    #[test]
    fn neuromod_deltas_default_when_healthy() {
        let pni = PniCoupling::new();
        let deltas = pni.compute_neuromod_deltas();

        // Healthy: IL-6=0, TNF=0, IL-1b=0, sickness=0 → most deltas near zero
        assert!(deltas.delta_serotonin.abs() < 1e-6);
        assert!(deltas.delta_dopamine.abs() < 1e-6);
        assert!(deltas.delta_noradrenaline.abs() < 1e-6);
        assert!(deltas.delta_adenosine.abs() < 1e-6);
        // IL-10 baseline = 0.2 → small oxytocin boost
        assert!(
            deltas.delta_oxytocin > 0.0,
            "baseline IL-10 should give oxytocin boost"
        );
    }

    #[test]
    fn sustained_threat_activates_ifn_gamma() {
        let mut pni = PniCoupling::new();

        // Short threat — no IFN-gamma
        pni.update_from_threat(0.8, 3);
        let ifn_short = pni.state.cytokines.interferon_gamma;

        // Sustained threat — IFN-gamma activates
        let mut pni2 = PniCoupling::new();
        pni2.update_from_threat(0.8, 10);
        let ifn_sustained = pni2.state.cytokines.interferon_gamma;

        assert!(
            ifn_sustained > ifn_short,
            "sustained threat should activate IFN-gamma: short={ifn_short}, sustained={ifn_sustained}"
        );
    }
}
