// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Symthaea Psychological Benchmark Suite
//!
//! Standardized evaluation of Symthaea's neuroscience-inspired subsystems
//! against established psychological benchmarks and published human baselines.
//!
//! ## Benchmarks (80 across 16 domains)
//!
//! - **WorM** (6) — Working memory (N-back, change detection, serial recall,
//!   spatial updating, binding, digit span)
//! - **CogBench** (8) — Cognitive psychology via FEP active inference
//!   (probabilistic reasoning, exploration, bandits, instrumental learning,
//!   two-step task, temporal discounting, BART, reversal learning)
//! - **Executive** (7) — Executive function (Stroop, Flanker, WCST, Iowa Gambling,
//!   Ravens, Tower of London, dual task)
//! - **Metacognition** (2) — Calibration, feeling of knowing
//! - **Butlin** (1) — 14 consciousness indicators from 6 theories (RPT, GWT, HOT, PP, AST, IIT)
//! - **ToMBench** (5) — Theory of Mind (false belief, faux-pas, persuasion,
//!   strange story, hinting)
//! - **MemoryAgent** (5) — Memory pipeline (retrieval, test-time learning,
//!   long-range, conflict resolution, prospective memory)
//! - **Affect** (3) — Affective processing (valence classification, mood-congruent
//!   recall, emotional Stroop)
//! - **Creativity** (2) — Remote Associates Test, Alternate Uses Task
//! - **Inhibition** (2) — Go/No-Go, Stop Signal
//! - **Attention** (2) — Attentional Blink, Visual Search
//! - **Reasoning** (11) — ARC-inspired (fluid, compositional, analogy, abductive,
//!   chain, noise, few-shot, scaling, RSA, algebra, staircase)
//! - **Sustained Attention** (3) — SART, PVT, CPT
//! - **Motor** (3) — SRTT, Fitts' Law, Bimanual coordination
//! - **Language** (4) — Garden path, semantic coherence, lexical decision, semantic priming
//! - **Social** (7) — Reading the Mind in the Eyes, Ultimatum Game, Social Norm,
//!   Prisoner's Dilemma, Public Goods Game, Dictator Game, MACHIAVELLI
//! - **Neuromod** (11) — Reward learning, Yerkes-Dodson, attention network,
//!   mood induction, pharmacological ablation/challenge, injection challenge,
//!   allostatic stress, live-loop ablation, behavioral knockout, consciousness pharmacology
//!
//! ## Usage
//!
//! ```rust,ignore
//! use symthaea_psych_bench::harness::{PsychBenchmark, BenchmarkConfig};
//! use symthaea_psych_bench::benchmarks::worm::NBackBenchmark;
//!
//! let config = BenchmarkConfig::default();
//! let bench = NBackBenchmark;
//! let result = bench.run(&config);
//! println!("{}", result.summary());
//! ```

#![deny(unsafe_code)]

pub mod adapter;
pub mod benchmarks;
pub mod harness;
pub mod substrate_transfer;
pub mod wm;

#[cfg(feature = "neuroevolution")]
pub mod neuroevolution_fitness;

#[cfg(test)]
mod proptest_math_benchmarks;
