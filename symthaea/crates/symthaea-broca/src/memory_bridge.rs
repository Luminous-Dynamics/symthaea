// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Memory Bridge — HDC Store retrieval and blending.
//!
//! Allows Broca to query the long-term narrative self and blend past experiences
//! into the current thought stream.

use anyhow::Result;
use symthaea_core::hdc::{ContinuousHV, HV};
use symthaea_hdc_store::store::HdcStore;

/// Bridge between Broca and the long-term HDC store.
#[derive(Clone)]
pub struct MemoryBridge {
    pub store: HdcStore,
    /// Number of past experiences to retrieve and blend.
    pub top_k: usize,
    /// Blending factor (0.0 = only current, 1.0 = only past).
    pub blend_alpha: f32,
}

impl MemoryBridge {
    pub fn new(store: HdcStore, top_k: usize, blend_alpha: f32) -> Self {
        Self {
            store,
            top_k,
            blend_alpha,
        }
    }

    /// Retrieve similar past thoughts and blend them into the current thought.
    pub fn blend_past_experiences(&self, current_thought: &mut ContinuousHV) -> Result<usize> {
        // 1. Convert to binary for LSH search
        let binary_query = current_thought.to_binary(0.0);

        // 2. Query store for top-K similar vectors
        let similar = self.store.scan_similar(&binary_query, self.top_k);
        if similar.is_empty() {
            return Ok(0);
        }

        // 3. Fetch and blend
        let mut retrieved_hvs = Vec::with_capacity(similar.len());
        for (id, _sim) in &similar {
            if let Some(bhv) = self.store.get(*id) {
                // Convert BinaryHV back to bipolar ContinuousHV for blending
                retrieved_hvs.push(ContinuousHV::from_vec(bhv.to_bipolar()));
            }
        }

        if retrieved_hvs.is_empty() {
            return Ok(0);
        }

        // 4. Bundle past experiences into a single "memory vector"
        let refs: Vec<&ContinuousHV> = retrieved_hvs.iter().collect();
        let memory_hv = ContinuousHV::bundle(&refs);

        // 5. Blend into current thought: self = (1-alpha)*self + alpha*memory
        current_thought.lerp_in_place(&memory_hv, 1.0 - self.blend_alpha, self.blend_alpha);

        Ok(retrieved_hvs.len())
    }

    /// Append a new experience to long-term memory.
    pub fn remember(&mut self, id: u64, thought: &ContinuousHV) -> Result<()> {
        let binary = thought.to_binary(0.0);
        self.store
            .append(id, &binary)
            .map_err(|e| anyhow::anyhow!("Store append failed: {}", e))
    }
}
