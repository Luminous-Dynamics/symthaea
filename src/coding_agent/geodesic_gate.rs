// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Geodesic Code Synthesis (GCS) advisory diagnostics for the CodingAgent.
//!
//! After code generation succeeds (compilation passes), the GCS gate analyzes
//! the generated source via:
//! 1. **PDG**: Build a Program Dependence Graph from the Rust source
//! 2. **Topology**: Compute Betti numbers (connected components, loops, voids)
//! 3. **Oracle**: Predict execution dynamics and check convergence
//!
//! Topology-derived findings are advisory hypotheses, not semantic proofs. They may
//! be shown to the agent as inspection prompts but must not veto compiler or test results.

use super::CodingAgent;

impl CodingAgent {
    /// Collect topology-derived hypotheses about generated source code.
    ///
    /// Builds a PDG, computes topological fingerprints, and runs an abstract
    /// convergence heuristic. Empty output means no heuristic advisory fired;
    /// it does not certify semantic correctness, reachability, or termination.
    #[cfg(feature = "geodesic_synthesis")]
    pub(super) fn collect_gcs_advisories(&self, source: &str, function_name: &str) -> Vec<String> {
        use symthaea_geodesic::{
            ExecutionOracle, ProgramDependenceGraph, TopologicalFingerprint,
            execution_oracle::OperationType,
        };

        let mut advisories = Vec::new();

        // Build PDG from source
        let pdg = ProgramDependenceGraph::from_rust_source(source, function_name);

        // Compute topology
        let complex = pdg.to_simplicial_complex();
        let fingerprint = TopologicalFingerprint::from_complex(&complex);

        // A nonzero beta_2 is a topological signal, not a proof of unreachable code.
        if fingerprint.betti.beta_2 > 0 {
            advisories.push(format!(
                "ADVISORY: nonzero second Betti number (beta_2={}); topology alone does not prove unreachable code. Confirm any dead-code claim with a control-flow reachability analysis.",
                fingerprint.betti.beta_2
            ));
        }

        // beta_1 suggests a cycle in the encoded complex; it does not prove a runtime loop.
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
                advisories.push(
                    "ADVISORY: abstract operation-sequence convergence heuristic did not converge; this is not a proof of nontermination. Confirm using explicit termination reasoning and behavioral tests.".to_string()
                );
            }
        }

        // Report structural info when no heuristic advisory is emitted.
        if advisories.is_empty() {
            tracing::info!(
                target: "symthaea::coding_agent::gcs",
                beta_0 = fingerprint.betti.beta_0,
                beta_1 = fingerprint.betti.beta_1,
                beta_2 = fingerprint.betti.beta_2,
                euler = fingerprint.euler_characteristic,
                "GCS topology analysis completed without heuristic findings"
            );
        }

        advisories
    }

    /// No-op when geodesic_synthesis feature is disabled.
    #[cfg(not(feature = "geodesic_synthesis"))]
    pub(super) fn collect_gcs_advisories(&self, _source: &str, _function_name: &str) -> Vec<String> {
        Vec::new()
    }
}
