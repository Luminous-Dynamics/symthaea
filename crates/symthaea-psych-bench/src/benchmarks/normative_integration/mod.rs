// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Normative Integration: Beyond Butlin Consciousness Indicators
//!
//! Tests whether moral reasoning coherence and consciousness level are
//! structurally coupled — a proposed 15th indicator (NI-1) extending the
//! Butlin et al. (2023) consciousness indicator battery.
//!
//! ## Key Claims Tested
//!
//! 1. **Drift-Consciousness Coupling**: Abrupt moral trajectory shifts produce
//!    measurable consciousness degradation (Festinger 1957, cognitive dissonance)
//! 2. **Anomaly Proportionality**: Consciousness dampening scales with anomaly
//!    severity, not binary threshold (graded response)
//! 3. **Topological Coherence**: Moral space fragmentation (high beta_0) predicts
//!    higher consciousness variance (Damasio 1994, somatic markers)
//!
//! ## Honest Limitations
//!
//! - We designed the coupling, then measured it (self-referential validation)
//! - Moral dimensions are our Eight Harmonies framework, not empirically derived
//! - Coupling parameters are tuned, not derived from neuroscience
//! - Single architecture — generalizability unknown
//!
//! ## References
//!
//! - Butlin, P. et al. (2023). Consciousness in Artificial Intelligence. arXiv:2308.08708.
//! - Damasio, A. (1994). Descartes' Error.
//! - Festinger, L. (1957). A Theory of Cognitive Dissonance.
//! - Moll, J. et al. (2005). The neural basis of human moral cognition. Nat Rev Neurosci 6(10).

pub mod anomaly_proportionality;
pub mod drift_coupling;
pub mod topological_coherence;

pub use anomaly_proportionality::AnomalyProportionalityBenchmark;
pub use drift_coupling::DriftCouplingBenchmark;
pub use topological_coherence::TopologicalCoherenceBenchmark;
