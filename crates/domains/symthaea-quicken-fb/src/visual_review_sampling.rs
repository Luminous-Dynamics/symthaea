// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Frozen semantic sampling points for exact Spore visual review.
//!
//! The purpose is methodological: control and treatment galleries must be
//! sampled at the same semantic stage progress so visual review cannot quietly
//! cherry-pick flattering frames. This module performs no rendering and carries
//! no boot authority.

/// Frozen v0.3.3 contact-sheet sample points in semantic stage progress.
pub const CONTACT_SHEET_PROGRESS_V1: [f32; 7] = [0.0, 0.15, 0.35, 0.50, 0.65, 0.85, 1.0];

/// Reduced sample set for expensive full lifecycle matrices.
pub const MATRIX_PROGRESS_V1: [f32; 3] = [0.20, 0.50, 0.80];

/// Return the frozen contact-sheet points.
pub const fn contact_sheet_progress() -> &'static [f32; 7] {
    &CONTACT_SHEET_PROGRESS_V1
}

/// Return the frozen reduced matrix points.
pub const fn matrix_progress() -> &'static [f32; 3] {
    &MATRIX_PROGRESS_V1
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_strictly_increasing(values: &[f32]) {
        assert!(!values.is_empty());
        for value in values {
            assert!(value.is_finite());
            assert!((0.0..=1.0).contains(value));
        }
        for pair in values.windows(2) {
            assert!(pair[0] < pair[1]);
        }
    }

    #[test]
    fn frozen_contact_sheet_sampling_is_ordered_and_bounded() {
        assert_strictly_increasing(contact_sheet_progress());
        assert_eq!(contact_sheet_progress().first().copied(), Some(0.0));
        assert_eq!(contact_sheet_progress().last().copied(), Some(1.0));
        assert!(contact_sheet_progress().contains(&0.50));
    }

    #[test]
    fn reduced_matrix_samples_early_middle_and_late_stage() {
        assert_strictly_increasing(matrix_progress());
        assert!(MATRIX_PROGRESS_V1[0] < 0.33);
        assert_eq!(MATRIX_PROGRESS_V1[1], 0.50);
        assert!(MATRIX_PROGRESS_V1[2] > 0.67);
    }

    #[test]
    fn review_sampling_is_literal_and_profile_independent() {
        // These exact vectors are part of the comparison protocol. Any change
        // requires an explicit protocol version bump rather than silent tuning.
        assert_eq!(
            CONTACT_SHEET_PROGRESS_V1,
            [0.0, 0.15, 0.35, 0.50, 0.65, 0.85, 1.0]
        );
        assert_eq!(MATRIX_PROGRESS_V1, [0.20, 0.50, 0.80]);
    }
}
