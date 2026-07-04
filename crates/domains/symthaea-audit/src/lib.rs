// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Standalone LLM-agent-driven architecture/trust auditor.
//!
//! Points a tool-calling LLM agent at a target repository (or a named subsystem within
//! one) with a read-only sandboxed tool surface, and produces a structured six-section
//! report: WIRED / CLAIMED BUT DARK / SAFETY-CRITICAL / UNTESTED / SHOULD DELETE /
//! SHOULD GATE, with every claim required to carry a file:line citation.
//!
//! Deliberately independent of `symthaea-core` and the main `symthaea` crate so it
//! compiles fast and runs anywhere — see `llm_client` module docs for the one place
//! that trades code reuse for that independence.

pub mod agent_loop;
pub mod cli;
pub mod heuristics;
pub mod llm_client;
pub mod prefetch;
pub mod report_prompt;
pub mod tools;
pub mod verify;
