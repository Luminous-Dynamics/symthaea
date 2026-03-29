// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Coding domain benchmarks.
//!
//! - **HumanEvalMini** — 20 HumanEval-style function synthesis problems solved via HDC (Chen et al., 2021)
//! - **BugDetection** — Detect and classify seeded bugs in code snippets via HDC (Beller et al., 2018)

pub mod bug_detection;
pub mod humaneval_mini;

pub use bug_detection::BugDetectionBenchmark;
pub use humaneval_mini::HumanEvalMiniBenchmark;
