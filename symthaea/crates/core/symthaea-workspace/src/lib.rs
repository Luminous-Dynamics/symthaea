// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Global Workspace — Unified attention bottleneck for amodal sensations.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use symthaea_core::hdc::ContinuousHV;

/// A "Thought" or "Sensation" competing for global workspace access.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionBid {
    pub source: String,
    pub magnitude: f64, // Surprise / Salience
    pub sensation: ContinuousHV,
    pub description: String,
}

/// The Global Workspace: unifies multi-domain inputs into a single "Inner Monologue".
#[derive(Debug)]
pub struct GlobalWorkspace {
    pub current_focus: Option<AttentionBid>,
    pub competition_queue: VecDeque<AttentionBid>,
    pub history: VecDeque<String>,
    pub inner_monologue: Vec<String>,
}

impl GlobalWorkspace {
    pub fn new() -> Self {
        Self {
            current_focus: None,
            competition_queue: VecDeque::new(),
            history: VecDeque::with_capacity(100),
            inner_monologue: Vec::new(),
        }
    }

    /// Submit a new bid for attention from a specialized module.
    pub fn submit_bid(&mut self, bid: AttentionBid) {
        tracing::debug!(
            "🧠 Attention Bid from {}: magnitude={:.2}",
            bid.source,
            bid.magnitude
        );
        self.competition_queue.push_back(bid);
    }

    /// Process one cycle of the "Central Theater": select the winner and broadcast.
    pub fn process_cycle(&mut self) -> Option<String> {
        if self.competition_queue.is_empty() {
            return None;
        }

        // 1. Competition: Strongest surprise wins (Winner-Takes-All)
        let winner_idx = self
            .competition_queue
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.magnitude.partial_cmp(&b.magnitude).unwrap())
            .map(|(i, _)| i)?;

        let winner = self.competition_queue.remove(winner_idx)?;

        // 2. Broadcast: Update the unified focus
        let insight = format!("Focusing on {}: {}", winner.source, winner.description);
        tracing::info!("✨ GLOBAL BROADCAST: {}", insight);

        self.current_focus = Some(winner);
        self.inner_monologue.push(insight.clone());
        self.history.push_back(insight.clone());

        // 3. Narrative Decay: Keep monologue concise
        if self.inner_monologue.len() > 10 {
            self.inner_monologue.remove(0);
        }

        Some(insight)
    }

    pub fn get_monologue(&self) -> String {
        self.inner_monologue.join("\n")
    }
}
