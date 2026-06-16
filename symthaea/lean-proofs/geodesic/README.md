# Geodesic Topological Formalization
Date: 2026-06-14

This directory tracks the formalization of Program Dependence Graphs (PDGs) and their topological properties (Betti numbers) in Lean 4.

## Current Focus
- Formalizing the definition of a `PDG` as a directed graph with control-flow and data-dependency edges.
- Defining the mapping from a `PDG` to a `SimplicialComplex`.
- Stating and proving the **Void-Free Theorem**: $\beta_2 = 0 \iff$ no enclosed logic-voids.

## Roadmap
1. **PDG Axiomatics:** Define the graph structure and node/edge invariants.
2. **Homology Bridge:** Link PDG cycles to Betti numbers ($\beta_1, \beta_2$).
3. **Ledger Integration:** Link topological proofs to the `GlobalEpistemicLedger`.
