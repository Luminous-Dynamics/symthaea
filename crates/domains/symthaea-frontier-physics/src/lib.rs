// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # symthaea-frontier-physics — Frontier Physics Research
//!
//! Exploratory mathematical and physical models used for falsifiable research.
//! Modules in this crate may be speculative; their presence is not evidence that
//! a proposed physical or consciousness interpretation is correct.
//!
//! 1. **Path Integrals** — Feynman's sum-over-histories, connecting QM ↔ StatMech
//! 2. **Information Geometry** — Fisher metric on probability spaces
//! 3. **Geometric Emergence** — theory-neutral trajectory observatory for probability space
//! 4. **DMRG** — Density Matrix Renormalization Group for strongly correlated systems
//! 5. **Topological QFT** — Knot invariants, topological entanglement entropy
//! 6. **Quantum Error Correction** — Decoherence-free subspaces, biological coherence
//! 7. **Tensor Networks** — MPS/PEPS for efficient many-body quantum states
//! 8. **Stochastic QED** — Zero-point field, vacuum fluctuations, stochastic resonance
//! 9. **Non-equilibrium Thermodynamics** — Dissipative structures, entropy production
//! 10. **Holographic Principle** — ER=EPR, AdS/CFT, spacetime from entanglement
//! 11. **Quantum Darwinism** — Emergence of classicality from decoherence

pub mod dmrg;
pub mod geometric_emergence;
pub mod holographic;
pub mod information_geometry;
pub mod nonequilibrium_thermo;
pub mod path_integrals;
pub mod quantum_darwinism;
pub mod quantum_error_correction;
pub mod stochastic_qed;
pub mod tensor_networks;
pub mod topological_qft;
