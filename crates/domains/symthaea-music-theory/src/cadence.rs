// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Cadences: the harmonic punctuation that ends phrases.
//!
//! Cadences are what make phrase structure *felt*. An antecedent phrase ends
//! on a Half cadence (unresolved — a musical question); its consequent answers
//! with an Authentic cadence (resolved — the answer). This question/answer
//! tension-and-release is the skeleton of tonal phrasing (Caplin, *Classical
//! Form*).

use crate::harmony::Key;
use serde::{Deserialize, Serialize};

/// A cadence type, identified by the scale degrees of its final two chords.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Cadence {
    /// V → I: full resolution ("answer"). The strongest close.
    Authentic,
    /// any → V: ends on the dominant, unresolved ("question").
    Half,
    /// IV → I: the "amen" cadence, gentle resolution.
    Plagal,
    /// V → vi: the dominant resolves deceptively to vi, denying closure.
    Deceptive,
}

impl Cadence {
    /// Detect the cadence formed by two consecutive scale degrees, if any.
    pub fn detect(penultimate: i32, final_degree: i32) -> Option<Cadence> {
        let p = penultimate.rem_euclid(7);
        let f = final_degree.rem_euclid(7);
        match (p, f) {
            (5, 1) => Some(Cadence::Authentic),
            (4, 1) => Some(Cadence::Plagal),
            (5, 6) => Some(Cadence::Deceptive),
            (_, 5) => Some(Cadence::Half),
            _ => None,
        }
    }

    /// The two scale degrees that produce this cadence (penultimate, final).
    pub fn degrees(self) -> (i32, i32) {
        match self {
            Cadence::Authentic => (5, 1),
            Cadence::Half => (2, 5), // predominant → dominant
            Cadence::Plagal => (4, 1),
            Cadence::Deceptive => (5, 6),
        }
    }

    /// Whether this cadence resolves (closes the phrase) or leaves it open.
    pub fn is_conclusive(self) -> bool {
        matches!(self, Cadence::Authentic | Cadence::Plagal)
    }

    /// A relative "closure strength" in [0, 1] — how finished the phrase feels.
    pub fn closure(self) -> f32 {
        match self {
            Cadence::Authentic => 1.0,
            Cadence::Plagal => 0.8,
            Cadence::Deceptive => 0.4,
            Cadence::Half => 0.2,
        }
    }

    /// The scale degree a melody should land on to voice this cadence's final
    /// chord convincingly (the tonally strongest chord tone):
    /// Authentic/Plagal → tonic (1); Half → the leading tone or 2 over V;
    /// Deceptive → 1 (the third of vi) to soften the surprise.
    pub fn melodic_goal(self) -> i32 {
        match self {
            Cadence::Authentic | Cadence::Plagal => 1, // tonic
            Cadence::Half => 2,                        // supertone over V (scale degree 2)
            Cadence::Deceptive => 1,                   // 1 is the 3rd of vi
        }
    }

    /// The final two chords of this cadence realized in a key (as roots' scale
    /// degrees — the caller builds chords via `Key::diatonic_triad`).
    pub fn realize(self, _key: Key) -> (i32, i32) {
        self.degrees()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pitch::PitchClass;

    #[test]
    fn detect_ground_truth() {
        assert_eq!(Cadence::detect(5, 1), Some(Cadence::Authentic));
        assert_eq!(Cadence::detect(4, 1), Some(Cadence::Plagal));
        assert_eq!(Cadence::detect(5, 6), Some(Cadence::Deceptive));
        assert_eq!(Cadence::detect(2, 5), Some(Cadence::Half));
        assert_eq!(Cadence::detect(4, 5), Some(Cadence::Half)); // any → V
        assert_eq!(Cadence::detect(3, 2), None);
    }

    #[test]
    fn degrees_round_trip_through_detect() {
        for c in [
            Cadence::Authentic,
            Cadence::Half,
            Cadence::Plagal,
            Cadence::Deceptive,
        ] {
            let (p, f) = c.degrees();
            assert_eq!(Cadence::detect(p, f), Some(c), "round-trip {c:?}");
        }
    }

    #[test]
    fn conclusive_vs_open() {
        assert!(Cadence::Authentic.is_conclusive());
        assert!(Cadence::Plagal.is_conclusive());
        assert!(!Cadence::Half.is_conclusive()); // a "question"
        assert!(!Cadence::Deceptive.is_conclusive());
    }

    #[test]
    fn closure_orders_authentic_strongest() {
        assert!(Cadence::Authentic.closure() > Cadence::Plagal.closure());
        assert!(Cadence::Plagal.closure() > Cadence::Deceptive.closure());
        assert!(Cadence::Deceptive.closure() > Cadence::Half.closure());
    }

    #[test]
    fn realize_in_c_major_gives_final_degrees() {
        // Authentic in C: V→I, i.e. degrees 5 then 1 (G then C).
        let (p, f) = Cadence::Authentic.realize(Key::major(PitchClass::C));
        assert_eq!((p, f), (5, 1));
    }
}
