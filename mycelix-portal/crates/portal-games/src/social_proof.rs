// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Social proof — wise error messages that escalate with attempt count.

/// Get an encouraging message based on topic, attempt count, and difficulty.
pub fn wise_error_message(_topic: &str, attempts: u32, _is_hard: bool) -> String {
    match attempts {
        0..=1 => "Not quite — try adjusting the values.".into(),
        2 => "Getting closer. Remember what each parameter does.".into(),
        3 => "Take a moment to think about the relationship between the variables.".into(),
        4 => "You're building understanding through persistence. That's how learning works.".into(),
        5..=6 => "Every expert was once a beginner. The struggle IS the learning.".into(),
        _ => "You've shown incredible persistence. Consider reviewing the hints.".into(),
    }
}
