// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Biology-generic relabeling of [`symthaea_fep::markov_blanket::PermeabilityInputs`].
//!
//! `PermeabilityInputs` is named after specific neuromodulators (acetylcholine, oxytocin,
//! serotonin, ...) because it was built for the consciousness-loop's neuromodulator bath. The
//! underlying mechanism — internal state modulating how open an organism's boundary is to the
//! outside world — is the actual, general Markov-blanket-of-life mechanism (Friston 2013;
//! Kirchhoff et al. 2018), so it's worth reusing directly rather than reimplementing. But
//! naming it "oxytocin" on a bacterium would be a false biological claim, not an analogy.
//!
//! `BoundaryModulators` is that same mechanism under names that map onto general physiological
//! signaling instead of specific consciousness neurochemistry. The mapping below is a
//! **modeling analogy** (per `ALIFE_PLAN_2026-07-08.md` Phase 0 §0d), not a validated claim that
//! any specific organism uses these exact channels:
//!
//! | Field        | Stands in for (real biology, loosely)             | Direction  |
//! |--------------|----------------------------------------------------|------------|
//! | `attention`   | expected-uncertainty focusing (cue narrowing)      | closes     |
//! | `vigilance`   | unexpected-uncertainty / acute stress response     | closes     |
//! | `safety`      | homeostatic comfort (resources adequate)           | opens      |
//! | `bonding`     | reserved for Phase 2 coalition affinity            | opens      |
//! | `threat`      | acute danger signal                                | closes     |
//! | `trust`       | reserved for Phase 2 coalescence readiness         | (indirect) |
//! | `engagement`  | safe, active engagement with the environment       | opens      |
//!
//! "Closes"/"opens" refer to blanket permeability: attentional narrowing under stress is a real,
//! well-established phenomenon (Easterbrook 1959, cue-utilization hypothesis) that this mapping
//! borrows the *shape* of, not a claim that any given organism implements it via these channels.
use serde::{Deserialize, Serialize};
use symthaea_fep::markov_blanket::PermeabilityInputs;

/// Internal-state-driven inputs to an organism's Markov blanket permeability.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BoundaryModulators {
    pub attention: f64,
    pub vigilance: f64,
    pub safety: f64,
    pub bonding: f64,
    pub threat: f64,
    pub trust: f64,
    pub engagement: f64,
}

impl Default for BoundaryModulators {
    fn default() -> Self {
        Self {
            attention: 0.5,
            vigilance: 0.3,
            safety: 0.5,
            bonding: 0.0,
            threat: 0.0,
            trust: 0.5,
            engagement: 0.0,
        }
    }
}

impl BoundaryModulators {
    /// Translate into the field names `MarkovBoundaryOperator::compute_permeability` expects.
    /// A pure relabeling — the field-for-field mapping and the math downstream are unchanged.
    pub fn to_permeability_inputs(self) -> PermeabilityInputs {
        PermeabilityInputs {
            acetylcholine: self.attention,
            noradrenaline: self.vigilance,
            serotonin: self.safety,
            oxytocin: self.bonding,
            threat_level: self.threat,
            peer_trust: self.trust,
            flow_state: self.engagement,
        }
    }

    /// Derive modulators from an organism's own physiological deficit.
    ///
    /// `deficit` is how far below its energy set-point the organism currently is, in `[0, 1]`
    /// (0 = at or above set-point, 1 = maximally depleted). Higher deficit reads as more acute
    /// stress: vigilance and threat rise, safety falls — narrowing the blanket (per the
    /// module-doc analogy above) rather than widening it, i.e. under acute deficit the organism
    /// leans on its internal priors more than on a possibly-noisy external signal.
    pub fn from_energy_deficit(deficit: f64) -> Self {
        let deficit = deficit.clamp(0.0, 1.0);
        Self {
            attention: 0.5,
            vigilance: deficit,
            safety: 1.0 - deficit,
            bonding: 0.0,
            threat: deficit,
            trust: 0.5,
            engagement: 0.0,
        }
    }
}
