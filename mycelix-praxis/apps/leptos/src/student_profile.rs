// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Student profile — persisted to localStorage.
//!
//! Collected during first-use onboarding. Used for personalized greetings,
//! exam countdown, and priority algorithm.

use leptos::prelude::*;
use serde::{Deserialize, Serialize};

use crate::persistence;

fn default_framework() -> String {
    "caps".to_string()
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct StudentProfile {
    pub name: String,
    pub grade: u8,           // 9, 10, 11, 12
    pub exam_date: String,   // ISO date e.g. "2026-10-26"
    pub onboarding_complete: bool,
    /// Curriculum framework: "caps", "common_core", "ib", "self_directed"
    #[serde(default = "default_framework")]
    pub framework: String,
}

const PROFILE_KEY: &str = "praxis_student_profile";

/// Provide student profile context. Call once at app root.
pub fn provide_profile_context() -> (ReadSignal<StudentProfile>, WriteSignal<StudentProfile>) {
    let initial = persistence::load::<StudentProfile>(PROFILE_KEY)
        .unwrap_or_default();
    let (profile, set_profile) = signal(initial);

    Effect::new(move |_| {
        let p = profile.get();
        persistence::save(PROFILE_KEY, &p);
    });

    provide_context(profile);
    provide_context(set_profile);
    (profile, set_profile)
}

pub fn use_profile() -> ReadSignal<StudentProfile> {
    expect_context::<ReadSignal<StudentProfile>>()
}

pub fn use_set_profile() -> WriteSignal<StudentProfile> {
    expect_context::<WriteSignal<StudentProfile>>()
}

/// Days until exam from today. None if no exam date set.
pub fn days_until_exam(profile: &StudentProfile) -> Option<i64> {
    if profile.exam_date.is_empty() { return None; }
    let now = js_sys::Date::new_0();
    let today_ms = js_sys::Date::now();
    // Parse exam date
    let exam = js_sys::Date::new(&profile.exam_date.as_str().into());
    let exam_ms = exam.get_time();
    if exam_ms.is_nan() { return None; }
    let diff_days = ((exam_ms - today_ms) / (1000.0 * 60.0 * 60.0 * 24.0)).ceil() as i64;
    Some(diff_days)
}
