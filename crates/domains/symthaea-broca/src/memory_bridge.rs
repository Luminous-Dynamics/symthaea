// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Memory Bridge — HDC Store retrieval and blending.
//!
//! Allows Broca to query the long-term narrative self and blend past experiences
//! into the current thought stream.

use anyhow::Result;
use symthaea_core::hdc::ContinuousHV;
use symthaea_hdc_store::store::HdcStore;

/// Bridge between Broca and the long-term HDC store.
///
/// Not `Clone`: `HdcStore` is a zero-copy mmap with an exclusive advisory
/// file lock ("one exclusive mutable opener") — cloning it would violate the
/// store's single-writer invariant. The old `#[derive(Clone)]` here only ever
/// compiled against a since-changed on-disk `symthaea-hdc-store` (that crate
/// is referenced by committed Cargo.tomls but has NO git history — a
/// never-committed-crate repo-integrity gap, found 2026-07-16 when a
/// workspace build first exercised this path).
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
        // IMPROVEMENT: Use permutation-based bundling to preserve temporal order.
        // The first (most similar) memory is not shifted, subsequent ones are
        // permuted increasingly to represent their relative 'distance' in search space
        // or implicitly their chronological order if they were stored sequentially.
        let mut bundled_values = vec![0.0f32; current_thought.dim()];
        let n = retrieved_hvs.len() as f32;

        for (i, hv) in retrieved_hvs.iter().enumerate() {
            // Permute to avoid collapsing into a simple average (Sequence-Aware Bundling)
            let shifted = hv.permute(i);
            for (b, &s) in bundled_values.iter_mut().zip(shifted.as_slice().iter()) {
                *b += s / n;
            }
        }
        let memory_hv = ContinuousHV::from_vec(bundled_values);

        // 5. IMPROVEMENT: Holographic Resonance Blending
        // Instead of fixed lerp, we compute the "Resonance" (inner product alignment).
        // High resonance (familiarity) triggers deeper integration of past experiences.
        let resonance = current_thought.similarity(&memory_hv).max(0.1);
        let dynamic_alpha = (self.blend_alpha * (1.0 + resonance)).clamp(0.05, 0.95);

        // Blend into current thought: self = (1-alpha)*self + alpha*memory
        current_thought.lerp_in_place(&memory_hv, 1.0 - dynamic_alpha, dynamic_alpha);

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
