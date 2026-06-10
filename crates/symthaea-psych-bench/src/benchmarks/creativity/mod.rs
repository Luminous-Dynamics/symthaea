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

pub mod alternate_uses;
pub mod alternate_uses_divergent;
pub mod conceptual_blending;
pub mod insight_problem;
pub mod remote_associates;

pub use alternate_uses::AlternateUsesBenchmark;
pub use alternate_uses_divergent::DivergentThinkingBenchmark;
pub use conceptual_blending::ConceptualBlendingBenchmark;
pub use insight_problem::InsightProblemBenchmark;
pub use remote_associates::RemoteAssociatesBenchmark;
