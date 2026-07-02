// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Prism Verifiable Epistemic Search — ZKP proofs for honest search.
//!
//! Proves:
//! 1. **Ranking proof**: Top results were ranked per the stated formula
//! 2. **Similarity proof**: Hamming similarity between query and result is correct
//! 3. **Epistemic proof**: Claim's E-level was honestly computed
//!
//! In Binius: similarity XOR is free, ranking is ~256 AND, encoding is ~3,648 AND.
//! Search results become *verifiable* — provably honest, not manipulated.

pub mod ranking;
pub mod similarity;

pub use ranking::{generate_ranking_proof, verify_ranking_proof, RankingProof};
pub use similarity::{generate_similarity_proof, verify_similarity_proof, SimilarityProof};
