// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Geodesic Code Synthesis (GCS) verification gate for the CodingAgent.
//!
//! After code generation succeeds (compilation passes), the GCS gate analyzes
//! the generated source via:
//! 1. **PDG**: Build a Program Dependence Graph from the Rust source
//! 2. **Topology**: Compute Betti numbers (connected components, loops, voids)
//! 3. **Oracle**: Predict execution dynamics and check convergence
//!
//! Violations are returned as human-readable strings that get injected into the
//! next generation prompt as hard constraints ("MUST FIX").

use super::CodingAgent;

impl CodingAgent {
    /// Verify generated source code using Geodesic Code Synthesis analysis.
    ///
    /// Builds a PDG, computes topological fingerprints, and runs the execution
    /// oracle. Returns a list of violation descriptions (empty = all checks passed).
    #[cfg(feature = "geodesic_synthesis")]
    pub(super) fn verify_with_gcs(&self, source: &str, function_name: &str) -> Vec<String> {
        use symthaea_geodesic::{
            ExecutionOracle, ProgramDependenceGraph, TopologicalFingerprint,
            execution_oracle::OperationType,
        };

        let mut violations = Vec::new();

        // Build PDG from source
        let pdg = ProgramDependenceGraph::from_rust_source(source, function_name);

        // Compute topology
        let complex = pdg.to_simplicial_complex();
        let fingerprint = TopologicalFingerprint::from_complex(&complex);

        // Basic topological checks: beta_2 > 0 indicates unreachable code regions (voids)
        if fingerprint.betti.beta_2 > 0 {
            violations.push(format!(
                "TOPOLOGY: {} unreachable code region(s) detected (beta_2={}). Remove dead code.",
                fingerprint.betti.beta_2, fingerprint.betti.beta_2
            ));
        }

        // Execution oracle: check convergence for loops (beta_1 > 0 means cycles exist)
        if fingerprint.betti.beta_1 > 0 {
            let mut oracle = ExecutionOracle::new();
            let statements: Vec<_> = source
                .lines()
                .filter(|l| !l.trim().is_empty() && !l.trim().starts_with("//"))
                .map(|l| {
                    let op = OperationType::classify(l);
                    let hv = symthaea_core::hdc::binary_hv::BinaryHV::random(
                        l.bytes()
                            .fold(0u64, |a, b| a.wrapping_mul(31).wrapping_add(b as u64)),
                    );
                    (hv, op)
                })
                .collect();

            let prediction = oracle.predict_sequence(&statements);
            if !prediction.converged {
                violations.push(
                    "ORACLE: Execution dynamics did not converge — potential infinite loop or non-terminating recursion.".to_string()
                );
            }
        }

        // Report structural info even when no violations
        if violations.is_empty() {
            tracing::info!(
                target: "symthaea::coding_agent::gcs",
                beta_0 = fingerprint.betti.beta_0,
                beta_1 = fingerprint.betti.beta_1,
                beta_2 = fingerprint.betti.beta_2,
                euler = fingerprint.euler_characteristic,
                "GCS verification passed"
            );
        }

        violations
    }

    /// No-op when geodesic_synthesis feature is disabled.
    #[cfg(not(feature = "geodesic_synthesis"))]
    pub(super) fn verify_with_gcs(&self, _source: &str, _function_name: &str) -> Vec<String> {
        Vec::new()
    }
}
