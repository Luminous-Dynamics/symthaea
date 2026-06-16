// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! ForkedState Snapshot Management
//!
//! Creates and manages lightweight state forks for MCTS rollouts.
//! Key properties:
//! - Arc-shared weights (zero-copy)
//! - Cloned states for mutation isolation
//! - Deterministic RNG for reproducibility (INV-3)

use super::types::ForkedState;
use std::sync::Arc;

/// Snapshot manager: creates ForkedStates from network state.
#[derive(Debug, Clone)]
pub struct SnapshotManager {
    /// Current network weights (shared reference).
    weights: Arc<Vec<f32>>,
    /// Current state vectors.
    current_states: Vec<Vec<f32>>,
    /// Current time constants.
    current_taus: Vec<f32>,
    /// Base seed for deterministic forks (INV-3).
    base_seed: [u8; 32],
    /// Fork counter for deterministic seed derivation.
    fork_counter: u64,
}

impl SnapshotManager {
    /// Create a new snapshot manager.
    pub fn new(weights: Vec<f32>, states: Vec<Vec<f32>>, taus: Vec<f32>, seed: [u8; 32]) -> Self {
        Self {
            weights: Arc::new(weights),
            current_states: states,
            current_taus: taus,
            base_seed: seed,
            fork_counter: 0,
        }
    }

    /// Create a ForkedState from current network state.
    ///
    /// Each fork gets a deterministically derived RNG seed
    /// for reproducibility (INV-3).
    pub fn fork(&mut self) -> ForkedState {
        self.fork_counter += 1;
        let seed = self.derive_seed(self.fork_counter);
        ForkedState::new(
            Arc::clone(&self.weights),
            self.current_states.clone(),
            self.current_taus.clone(),
            seed,
        )
    }

    /// Update the current network state (called each cycle).
    pub fn update_state(&mut self, states: Vec<Vec<f32>>, taus: Vec<f32>) {
        self.current_states = states;
        self.current_taus = taus;
    }

    /// Update shared weights (only when network is retrained).
    pub fn update_weights(&mut self, weights: Vec<f32>) {
        self.weights = Arc::new(weights);
    }

    /// Derive a deterministic seed from the base seed and a counter.
    fn derive_seed(&self, counter: u64) -> [u8; 32] {
        let mut seed = self.base_seed;
        let counter_bytes = counter.to_le_bytes();
        for i in 0..8 {
            seed[i] ^= counter_bytes[i];
        }
        seed
    }

    /// Number of forks created so far.
    pub fn fork_count(&self) -> u64 {
        self.fork_counter
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fork_equivalence() {
        // TEST-FORK-1: fork(net).evolve(dt, x) produces same result
        // as cloning the full state and evolving
        let weights = vec![1.0, 2.0, 3.0];
        let states = vec![vec![0.5, -0.3]];
        let taus = vec![1.0];
        let seed = [42u8; 32];

        let mut manager = SnapshotManager::new(weights.clone(), states.clone(), taus.clone(), seed);

        let mut fork = manager.fork();
        let mut full_clone = ForkedState::new(Arc::new(weights), states, taus, seed);

        let input = [0.7, -0.2];
        fork.evolve(0.1, &input);
        full_clone.evolve(0.1, &input);

        // Should be bitwise equal
        for (a, b) in fork.states[0].iter().zip(full_clone.states[0].iter()) {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "TEST-FORK-1: fork and clone must evolve identically"
            );
        }
    }

    #[test]
    fn test_fork_independence() {
        // TEST-FORK-2: Two forks from same snapshot don't cross-contaminate
        let mut manager = SnapshotManager::new(vec![1.0], vec![vec![0.0]], vec![1.0], [0u8; 32]);

        let mut fork_a = manager.fork();
        let mut fork_b = manager.fork();

        fork_a.evolve(0.1, &[1.0]);
        fork_b.evolve(0.1, &[-1.0]);

        assert_ne!(
            fork_a.states[0][0], fork_b.states[0][0],
            "TEST-FORK-2: Forks must diverge under different inputs"
        );
    }

    #[test]
    fn test_fork_cow_correctness() {
        // TEST-FORK-3: Mutating weights in a full clone doesn't affect fork
        let weights = vec![1.0, 2.0, 3.0];
        let states = vec![vec![0.5]];
        let taus = vec![1.0];

        let mut manager = SnapshotManager::new(weights, states, taus, [0u8; 32]);
        let fork = manager.fork();

        // Mutate weights in the manager
        manager.update_weights(vec![999.0, 888.0, 777.0]);

        // Fork should NOT observe the mutation
        assert_eq!(
            fork.weights[0], 1.0,
            "TEST-FORK-3: Fork must not see weight mutation"
        );
        assert_eq!(fork.weights[1], 2.0);
        assert_eq!(fork.weights[2], 3.0);
    }

    #[test]
    fn test_deterministic_seeds() {
        // INV-3: Same base seed + same counter → same fork seed
        let mut manager1 = SnapshotManager::new(vec![1.0], vec![vec![0.0]], vec![1.0], [42u8; 32]);
        let mut manager2 = SnapshotManager::new(vec![1.0], vec![vec![0.0]], vec![1.0], [42u8; 32]);

        let fork1 = manager1.fork();
        let fork2 = manager2.fork();

        assert_eq!(
            fork1.rng_state, fork2.rng_state,
            "INV-3: Same seed must produce same fork"
        );
    }
}
