// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Praxis shared types — mirrors praxis zome entries.

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Course {
    pub id: String,
    pub title: String,
    pub description: String,
    pub domain: String,
    pub creator_did: String,
    pub status: CourseStatus,
    pub created_at: i64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum CourseStatus { Draft, Published, Archived }

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LearnerProgress {
    pub learner_did: String,
    pub course_id: String,
    pub mastery_permille: u32,
    pub xp: u64,
    pub level: u32,
    pub streak_days: u32,
    pub last_reviewed: i64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Credential {
    pub id: String,
    pub learner_did: String,
    pub course_id: String,
    pub score: u32,
    pub grade: String,
    pub issued_at: i64,
    pub proof_type: String,
    pub verification_method: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FlRound {
    pub round_id: u64,
    pub participants: u32,
    pub accuracy: f64,
    pub byzantine_detected: u32,
    pub completed_at: i64,
}
