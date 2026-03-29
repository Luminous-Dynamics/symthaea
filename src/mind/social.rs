// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Social coherence interface for the Continuous Mind.

use super::{ContinuousMind, SocialMessage};

impl ContinuousMind {
    /// Enable social coherence after construction.
    pub fn enable_social_coherence(&mut self) {
        if self.social_coherence.is_none() {
            self.social_coherence = Some(crate::brain::SocialCoherence::new(
                crate::brain::SocialCoherenceConfig {
                    dimension: self.config.dimension,
                    ..Default::default()
                },
            ));
        }
    }

    /// Receive a social message from a network peer.
    pub fn receive_social(&mut self, msg: SocialMessage) {
        self.social_inbox.push(msg);
    }

    /// Drain outgoing social messages (for network broadcast).
    pub fn drain_social_outbox(&mut self) -> Vec<SocialMessage> {
        std::mem::take(&mut self.social_outbox)
    }

    /// Check if social coherence is enabled.
    pub fn is_social(&self) -> bool {
        self.social_coherence.is_some()
    }

    /// Get a reference to the social coherence system (if enabled).
    pub fn social_coherence(&self) -> Option<&crate::brain::SocialCoherence> {
        self.social_coherence.as_ref()
    }
}
