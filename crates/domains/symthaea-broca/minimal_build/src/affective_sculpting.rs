// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Affective Sculpting — The Symbiotic Interface
//!
//! Turns the EpistemicDashboard into a true steering wheel.
//! Humans can now manually sculpt the AI's cognitive style in real time.

use crate::encoder::ThoughtChannels;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AffectiveStyle {
    CalmRigor,          // Security-critical, precise
    AgitatedCreativity, // Exploratory, high temperature
    BalancedFlow,       // General purpose
    FormalProof,        // Maximum epistemic focus
}

pub struct AffectiveSculptor {
    current_style: AffectiveStyle,
}

impl AffectiveSculptor {
    pub fn new() -> Self {
        Self {
            current_style: AffectiveStyle::BalancedFlow,
        }
    }

    /// Human "nudge" — sculpt the system's emotional and epistemic state.
    pub fn sculpt(&mut self, channels: &mut ThoughtChannels, style: AffectiveStyle) {
        self.current_style = style;

        match style {
            AffectiveStyle::CalmRigor => {
                channels.set_valence(0.85);
                channels.set_arousal(0.15);
            }
            AffectiveStyle::AgitatedCreativity => {
                channels.set_valence(-0.35);
                channels.set_arousal(0.92);
            }
            AffectiveStyle::BalancedFlow => {
                channels.set_valence(0.25);
                channels.set_arousal(0.55);
            }
            AffectiveStyle::FormalProof => {
                channels.set_valence(0.65);
                channels.set_arousal(0.22);
            }
        }
    }

    pub fn current_style(&self) -> AffectiveStyle {
        self.current_style
    }

    pub fn render_affective_controls(&self) -> String {
        format!(
            "\n🎛️  AFFECTIVE SCULTING CONTROLS\n\
             Current Style: {:?}\n\n\
             Available Styles:\n\
             • CalmRigor         → Security patches, formal proofs\n\
             • AgitatedCreativity → Brainstorming, novel APIs\n\
             • BalancedFlow      → General development\n\
             • FormalProof       → Maximum mathematical certainty\n",
            self.current_style
        )
    }
}

impl Default for AffectiveSculptor {
    fn default() -> Self {
        Self::new()
    }
}
