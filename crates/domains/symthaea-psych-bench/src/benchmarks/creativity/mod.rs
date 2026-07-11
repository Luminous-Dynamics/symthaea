// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Creativity domain benchmarks.
//!
//! - **RemoteAssociates** — Find a word connecting three cue words (RAT)
//! - **AlternateUses** — Generate novel uses for common objects (AUT)
//! - **DivergentThinking** — Originality/flexibility in alternative uses (Guilford, 1967)
//! - **ConceptualBlending** — Combine concepts from different domains (Fauconnier & Turner, 2002)
//! - **InsightProblem** — Sudden restructuring for problem solving (Bowden & Jung-Beeman, 2003)
//! - **grounded** — Honestly-grounded RAT + DAT over real word embeddings (no planted links)
//!
//! # Scope honesty (2026-07-06 review)
//!
//! These benchmarks are **HDC-algebra sanity checks, not creativity
//! measurements**. Stimuli are synthetic hypervectors: the RAT adapter
//! (`adapter/semantic.rs::for_rat`) constructs shared cue↔solution link
//! vectors — planting the very signal the benchmark then recovers — and the
//! AUT/blending/insight tasks operate on random HVs with no real lexical
//! semantics. They verify that bundle/bind/unbind arithmetic behaves as
//! designed; they cannot distinguish a creative system from a random one and
//! must not be cited as evidence of creative capability.
//!
//! The first honestly-grounded counterparts live in [`grounded`]: a RAT
//! variant over real word embeddings (Bowden & Jung-Beeman 2003 norms via
//! TSV loader, no planted links) and the DAT (Olson et al. 2021). SemDis-
//! scored AUT over real generated text is still planned — see
//! `ART_CULTURE_REVIEW_AND_PLAN_2026-07-06.md` Phase 2.

pub mod alternate_uses;
pub mod alternate_uses_divergent;
pub mod conceptual_blending;
pub mod grounded;
pub mod insight_problem;
pub mod remote_associates;

pub use alternate_uses::AlternateUsesBenchmark;
pub use alternate_uses_divergent::DivergentThinkingBenchmark;
pub use conceptual_blending::ConceptualBlendingBenchmark;
pub use grounded::{
    DatScore, DeterministicHashEmbedder, GloVeFileEmbedder, RatGroundedBenchmark, RatItem,
    RatItemSet, RatReport, WordEmbedder, score_dat,
};
pub use insight_problem::InsightProblemBenchmark;
pub use remote_associates::RemoteAssociatesBenchmark;
