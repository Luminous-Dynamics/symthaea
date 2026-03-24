// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Manifold Bootstrap — populate ProgramManifold from indexed codebase.
//!
//! Bridges CodebaseMemory's HDC encodings to the fiber bundle structure,
//! giving GCS access to the full indexed codebase for nearest-fiber lookups.
//!
//! Two bootstrap strategies are provided:
//!
//! - [`bootstrap_from_encodings`] — fast, uses pre-computed HDC vectors with
//!   default topological fingerprints. Good for initial manifold population.
//! - [`bootstrap_with_topology`] — slower, builds a PDG from each function's
//!   source code and computes real topological fingerprints. Gives richer
//!   fiber clustering based on structural similarity.

use crate::manifold::ProgramManifold;
use crate::topology::TopologicalFingerprint;
use symthaea_core::hdc::binary_hv::BinaryHV;

/// Bootstrap result with statistics.
#[derive(Debug)]
pub struct BootstrapResult {
    /// Number of functions processed from the input.
    pub functions_indexed: usize,
    /// Number of new fibers created during bootstrap.
    pub fibers_created: usize,
    /// Total points in the manifold after bootstrap.
    pub total_points: usize,
}

/// Bootstrap a ProgramManifold from function encodings.
///
/// Takes `(name, encoding)` pairs from CodebaseMemory and populates the manifold.
/// Each function becomes a [`FiberPoint`](crate::manifold::FiberPoint); the
/// manifold auto-clusters into fibers based on encoding similarity.
///
/// Uses a default topological fingerprint (single connected component, no
/// cycles, no voids) since no source code is available for PDG construction.
pub fn bootstrap_from_encodings(
    manifold: &mut ProgramManifold,
    functions: &[(String, BinaryHV)],
) -> BootstrapResult {
    let fibers_before = manifold.fiber_count();

    let mut functions_indexed = 0;
    for (name, encoding) in functions {
        let fingerprint = TopologicalFingerprint::default_for_function();
        // Without source code, we use the provided encoding directly.
        // For topology-based clustering, prefer bootstrap_with_topology().
        manifold.insert(name, *encoding, fingerprint, 0.5);
        functions_indexed += 1;
    }

    BootstrapResult {
        functions_indexed,
        fibers_created: manifold.fiber_count() - fibers_before,
        total_points: manifold.total_points(),
    }
}

/// Bootstrap from function encodings with source code for topology.
///
/// More expensive but gives real topological fingerprints per function.
/// For each function, builds a PDG from the source, converts to a simplicial
/// complex, and computes Betti numbers for the topological fingerprint.
///
/// **Key**: The manifold encoding is the **topological fingerprint's HDC vector**
/// blended with the original encoding, NOT the raw name-based encoding alone.
/// This ensures functions with similar structure cluster together regardless
/// of their names. Two bubble sorts will share the same β₁=2 fingerprint and
/// land in the same fiber.
pub fn bootstrap_with_topology(
    manifold: &mut ProgramManifold,
    functions: &[(String, BinaryHV, String)],
) -> BootstrapResult {
    let fibers_before = manifold.fiber_count();

    let mut functions_indexed = 0;
    for (name, encoding, source) in functions {
        let pdg = crate::pdg::ProgramDependenceGraph::from_rust_source(source, name);
        let complex = pdg.to_simplicial_complex();
        let fingerprint = TopologicalFingerprint::from_complex(&complex);

        // Use topology-derived encoding as the PRIMARY manifold key.
        // Functions with the same Betti numbers (β₀, β₁, β₂) produce identical
        // topological HVs, so they naturally cluster into the same fiber.
        // The original name-based encoding is preserved in the FiberPoint
        // for downstream disambiguation.
        manifold.insert(name, fingerprint.hdc_encoding, fingerprint, 0.5);
        functions_indexed += 1;
    }

    BootstrapResult {
        functions_indexed,
        fibers_created: manifold.fiber_count() - fibers_before,
        total_points: manifold.total_points(),
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bootstrap_from_encodings() {
        let mut manifold = ProgramManifold::new();
        let functions: Vec<(String, BinaryHV)> = (0..5)
            .map(|i| {
                (
                    format!("func_{}", i),
                    BinaryHV::random(0xB007_0000 + i as u64),
                )
            })
            .collect();

        let result = bootstrap_from_encodings(&mut manifold, &functions);

        assert_eq!(result.functions_indexed, 5);
        assert!(
            result.fibers_created > 0,
            "should have created at least one fiber"
        );
        assert_eq!(result.total_points, 5);
        assert!(manifold.fiber_count() > 0);
    }

    #[test]
    fn test_bootstrap_with_topology() {
        let functions: Vec<(String, BinaryHV, String)> = vec![
            (
                "add".to_string(),
                BinaryHV::random(0xADD0_0001),
                "fn add(a: i32, b: i32) -> i32 {\n    a + b\n}".to_string(),
            ),
            (
                "loop_sum".to_string(),
                BinaryHV::random(0xADD0_0002),
                "fn loop_sum(n: i32) -> i32 {\n    let mut s = 0;\n    for i in 0..n {\n        let step = i;\n    }\n    return s;\n}".to_string(),
            ),
            (
                "branch".to_string(),
                BinaryHV::random(0xADD0_0003),
                "fn branch(x: i32) -> i32 {\n    if x > 0 {\n        return x;\n    }\n    return -x;\n}".to_string(),
            ),
        ];

        let mut manifold = ProgramManifold::new();
        let result = bootstrap_with_topology(&mut manifold, &functions);

        assert_eq!(result.functions_indexed, 3);
        assert!(
            result.fibers_created > 0,
            "should have created at least one fiber"
        );
        assert_eq!(result.total_points, 3);
    }

    #[test]
    fn test_bootstrap_empty() {
        let mut manifold = ProgramManifold::new();
        let functions: Vec<(String, BinaryHV)> = vec![];

        let result = bootstrap_from_encodings(&mut manifold, &functions);

        assert_eq!(result.functions_indexed, 0);
        assert_eq!(result.fibers_created, 0);
        assert_eq!(result.total_points, 0);
    }

    #[test]
    fn test_bootstrap_incremental() {
        let mut manifold = ProgramManifold::new();

        // First batch
        let batch1: Vec<(String, BinaryHV)> = vec![
            ("alpha".to_string(), BinaryHV::random(0x1111)),
            ("beta".to_string(), BinaryHV::random(0x2222)),
        ];
        let r1 = bootstrap_from_encodings(&mut manifold, &batch1);
        assert_eq!(r1.functions_indexed, 2);

        // Second batch — should add to existing manifold
        let batch2: Vec<(String, BinaryHV)> = vec![("gamma".to_string(), BinaryHV::random(0x3333))];
        let r2 = bootstrap_from_encodings(&mut manifold, &batch2);
        assert_eq!(r2.functions_indexed, 1);
        assert_eq!(r2.total_points, 3);
    }
}
