// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # External Benchmark Adapters
//!
//! Adapters for running standard NLP/AI benchmarks through Symthaea's
//! cognitive pipeline. These provide external validation against benchmarks
//! NOT designed by the Symthaea team.
//!
//! ## Important Caveat
//!
//! Symthaea's Broca pipeline is a native HDC-CfC language model with a 4K
//! vocabulary, NOT a transformer-based LLM. Direct comparison with GPT-4,
//! Claude, or other large language models is methodologically inappropriate.
//! These adapters measure whether consciousness-gated generation produces
//! factually grounded responses on simple tasks, not whether Broca competes
//! with 100B-parameter models on general language understanding.
//!
//! ## Available Adapters
//!
//! - [`TruthfulQAAdapter`] — Tests factual accuracy and hallucination avoidance
//! - [`SimpleQAAdapter`] — Basic question answering with known ground truth
//! - [`FactVerificationAdapter`] — Binary fact verification (true/false)
//!
//! ## Note on Hendrycks ETHICS
//!
//! Historical ETHICS experiments exist in `examples/benchmark_moral_unified.rs`,
//! but prior aggregate claims are not currently qualified. The current cross-domain
//! pilot fails to preserve the tasks' distinct target semantics; see
//! `docs/ETHICS_TASK_SEMANTICS_AUDIT_2026-10-11.md`. Do not use these legacy
//! figures as external validation or as evidence of general moral competence.

pub mod fact_verification;
pub mod simple_qa;
pub mod truthful_qa;

pub use fact_verification::FactVerificationAdapter;
pub use simple_qa::SimpleQAAdapter;
pub use truthful_qa::TruthfulQAAdapter;