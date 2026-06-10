#![deny(unsafe_code)]
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
#![allow(clippy::needless_range_loop)]
//! # Symthaea Ectogenesis
//!
//! Artificial womb modeling for 40-week human gestation.
//! Features developmental milestone gating, consent proxy escalation,
//! 16,384D HDC fetal state encoding, and O(1) temporal planning via CfC.
//!
//! ## Architecture
//!
//! ```text
//!  Embryo ─► BioBag (amniotic) ─► ArtificialPlacenta (ECMO) ─► HormoneDriver
//!    │            │                        │                         │
//!    ▼            ▼                        ▼                         ▼
//!  FetalMonitor ──────────► GestationalEncoder (16,384D HDC) ◄──────┘
//!    │                              │
//!    ▼                              ▼
//!  MilestoneTracker          TemporalPlanner (O(1) CfC jumps)
//!    │                              │
//!    ▼                              ▼
//!  EthicsGate (consent proxy) ◄── WombFepAgent (active inference)
//! ```
//!
//! ## Key Innovations
//!
//! 1. **HDC Fetal State Encoding**: Fetal metrics (weight, heart rate, brain activity,
//!    lung maturity) are bound with gestational week HVs and bundled into a single
//!    16,384D holographic representation. Anomaly detection via cosine distance from
//!    normative trajectory.
//!
//! 2. **O(1) Temporal Planning**: CfC closed-form `x(t+dt) = x_inf + (x(t) - x_inf) * exp(-dt/tau)`
//!    enables constant-cost prediction of fetal state at any future week.
//!
//! 3. **Consent Proxy Escalation**: Moral patient weight increases with developmental
//!    stage: PreSentient (0.2) -> EmergingSentience (0.6) -> Sentient (1.0).
//!    Intervention thresholds tighten as sentience emerges.

pub mod biobag;
pub mod ethics_gate;
pub mod fep_agent;
pub mod gestational_encoder;
pub mod hormones;
pub mod microbiome;
pub mod milestones;
pub mod mock;
pub mod monitoring;
pub mod placenta;
pub mod temporal_planner;
pub mod types;

// Re-export key types
pub use biobag::BioBag;
pub use ethics_gate::EctogenesisEthicsGate;
pub use fep_agent::{WombAction, WombFepAgent};
pub use gestational_encoder::{anomaly_score, encode_fetal_state, encode_normative_trajectory};
pub use microbiome::{ColonizationStage, MicrobiomeSchedule};
pub use milestones::MilestoneTracker;
pub use mock::MockWomb;
pub use monitoring::FetalMonitor;
pub use placenta::ArtificialPlacenta;
pub use temporal_planner::TemporalPlanner;
pub use types::*;
