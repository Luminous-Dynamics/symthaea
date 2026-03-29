// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Redundant Computation Interface (Design Only)
//!
//! Defines traits for running critical computations on multiple hardware backends
//! and comparing results. Divergence indicates tampering or hardware fault.
//!
//! This is a future capability — actual multi-hardware execution requires
//! IPC/RPC infrastructure that doesn't exist yet. The traits are defined now
//! so the integrity framework has a stable interface to build against.

use std::fmt::Debug;

/// Trait for computations that can be verified by redundant execution.
///
/// Implementors wrap a critical computation (Phi, moral verdict, safety level)
/// so it can be dispatched to multiple backends and compared.
pub trait RedundantVerifiable: Send + Sync {
    /// Input type for the computation.
    type Input: Clone + Send;
    /// Output type for the computation.
    type Output: PartialEq + Debug + Send;

    /// Execute the computation on the local (primary) hardware.
    fn compute(&self, input: &Self::Input) -> Self::Output;

    /// Name of this computation for logging and telemetry.
    fn computation_name(&self) -> &'static str;
}

/// Result of a redundant computation across multiple backends.
#[derive(Debug)]
pub struct RedundantResult<T: Debug> {
    /// Consensus value (if majority agrees).
    pub consensus: Option<T>,
    /// Fraction of backends that agreed with consensus.
    pub agreement_ratio: f64,
    /// Names of backends that produced divergent results.
    pub divergent_backends: Vec<String>,
}

/// Coordinator that dispatches computations to multiple backends and compares.
///
/// Future implementation will use IPC/gRPC/shared memory to communicate
/// with verification backends running on separate hardware.
pub trait RedundantCoordinator: Send + Sync {
    /// Verify a computation by running it on all registered backends.
    ///
    /// Returns the consensus result and divergence information.
    fn verify<V: RedundantVerifiable>(
        &self,
        verifier: &V,
        input: &V::Input,
    ) -> RedundantResult<V::Output>;
}

/// Local-only coordinator that runs computations twice on the same hardware.
///
/// This is a minimal implementation that catches non-determinism and transient
/// hardware faults (bit flips), but NOT deliberate tampering of the local hardware.
/// It's the default when no external verification backends are configured.
pub struct LocalRedundantCoordinator;

impl RedundantCoordinator for LocalRedundantCoordinator {
    fn verify<V: RedundantVerifiable>(
        &self,
        verifier: &V,
        input: &V::Input,
    ) -> RedundantResult<V::Output> {
        let result1 = verifier.compute(input);
        let result2 = verifier.compute(input);
        if result1 == result2 {
            RedundantResult {
                consensus: Some(result1),
                agreement_ratio: 1.0,
                divergent_backends: Vec::new(),
            }
        } else {
            RedundantResult {
                consensus: None,
                agreement_ratio: 0.0,
                divergent_backends: vec![
                    format!("local_run_1: {:?}", result1),
                    format!("local_run_2: {:?}", result2),
                ],
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct AddVerifier;
    impl RedundantVerifiable for AddVerifier {
        type Input = (i32, i32);
        type Output = i32;
        fn compute(&self, input: &(i32, i32)) -> i32 {
            input.0 + input.1
        }
        fn computation_name(&self) -> &'static str {
            "addition"
        }
    }

    #[test]
    fn test_local_coordinator_deterministic() {
        let coord = LocalRedundantCoordinator;
        let result = coord.verify(&AddVerifier, &(2, 3));
        assert_eq!(result.consensus, Some(5));
        assert!((result.agreement_ratio - 1.0).abs() < f64::EPSILON);
        assert!(result.divergent_backends.is_empty());
    }
}
