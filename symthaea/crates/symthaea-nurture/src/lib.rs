#![deny(unsafe_code)]
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
#![allow(clippy::needless_range_loop)]
//! # Symthaea Nurture
//!
//! Bowlby attachment theory, co-regulation, critical periods, and developmental
//! milestones. Extends RelationalCore and NeuromodulatorBath with biologically-
//! grounded infant development. Uses 16,384D HDC and O(1) CfC temporal jumps
//! for critical period plasticity modeling.
//!
//! ## Architecture
//!
//! The crate models four foundational developmental systems:
//!
//! 1. **Attachment** (Bowlby, 1969): Caregiver interaction history shapes internal
//!    working models (self_worthy, other_reliable, world_safe) encoded as 16,384D
//!    hypervectors. Four attachment styles emerge from interaction quality.
//!
//! 2. **Co-regulation** (Tronick, 1989): Bidirectional affect synchronization where
//!    caregiver contact dampens infant arousal via oxytocin-mediated coupling.
//!
//! 3. **Critical Periods** (Hubel & Wiesel, 1970): Age-dependent plasticity windows
//!    for attachment (0-24mo), language (0-60mo), and vision (0-36mo), modeled via
//!    CfC closed-form temporal dynamics.
//!
//! 4. **Developmental Milestones** (WHO, 2006): 30+ milestones across gross motor,
//!    fine motor, language, social, and cognitive domains with normative z-scoring.

pub mod attachment;
pub mod attachment_style_formation;
pub mod contingency_learning;
pub mod coregulation;
pub mod critical_periods;
pub mod developmental_profile;
pub mod feeding;
pub mod internal_working_model;
pub mod language_acquisition;
pub mod milestones;
pub mod motor_development;
pub mod secure_base;
pub mod separation_distress;
pub mod sleep;
pub mod types;

pub use attachment::AttachmentSystem;
pub use attachment_style_formation::{determine_style, style_probability_distribution};
pub use contingency_learning::ContingencyLearner;
pub use coregulation::CoRegulationState;
pub use critical_periods::{CriticalPeriodDomain, CriticalPeriodModel};
pub use developmental_profile::{DevelopmentalProfile, DomainScore};
pub use feeding::{FeedingProgram, FeedingStage};
pub use internal_working_model::{encode_working_model, update_working_model};
pub use language_acquisition::{InfantDirectedSpeech, LanguageMilestone, language_milestones};
pub use milestones::{DevelopmentalMilestoneDb, Milestone};
pub use motor_development::{MotorMilestone, fine_motor_milestones, gross_motor_milestones};
pub use secure_base::SecureBase;
pub use separation_distress::{advance_separation, neuromod_for_stage};
pub use sleep::SleepSchedule;
pub use types::*;
