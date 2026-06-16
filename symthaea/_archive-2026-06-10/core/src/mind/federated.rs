// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Federated learning interface for the Continuous Mind.

use super::ContinuousMind;

impl ContinuousMind {
    /// Enable federated learning with initial weights.
    pub fn enable_federated(&mut self, weights: Vec<f32>) {
        use crate::swarm::FederatedAggregator;
        self.federated = Some(FederatedAggregator::new(weights).with_byzantine_tolerance(0.1));
    }

    /// Receive a gradient message from a network peer.
    pub fn receive_gradient(&mut self, msg: crate::swarm::GradientMessage) {
        self.federated_inbox.push(msg);
    }

    /// Drain outgoing gradient messages (for network broadcast).
    pub fn drain_outbox(&mut self) -> Vec<crate::swarm::GradientMessage> {
        std::mem::take(&mut self.federated_outbox)
    }

    /// Check if federated learning is enabled.
    pub fn is_federated(&self) -> bool {
        self.federated.is_some()
    }
}
