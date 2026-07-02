// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Loss utilities for CanaryCNN (MSE-on-logits canonical path).
//!
//! Provides fixed-point implementations of mean squared error (MSE) over
//! logits using one-hot targets. This keeps the proving circuit polynomial,
//! aligning with the Phase 1 zkVM plan.
//!
//! # Example
//! ```
//! use vsv_core::{Fixed, loss};
//!
//! let logits = [
//!     Fixed::from_f32(0.8),
//!     Fixed::from_f32(-0.1),
//!     Fixed::from_f32(0.3),
//! ];
//! let loss = loss::mse_single(&logits, 0).to_f32();
//! assert!(loss > 0.0);
//! ```

use crate::Fixed;

/// Compute MSE-on-logits for a single sample with one-hot target.
///
/// Returns the mean squared error between the logits and the one-hot vector
/// for `true_label`. Length of `logits` must be > 0.
pub fn mse_single(logits: &[Fixed], true_label: usize) -> Fixed {
    assert!(!logits.is_empty(), "logits length must be > 0");
    assert!(
        true_label < logits.len(),
        "true_label {} out of range for {} classes",
        true_label,
        logits.len()
    );

    let mut total = Fixed::ZERO;
    for (idx, &logit) in logits.iter().enumerate() {
        let target = if idx == true_label {
            Fixed::ONE
        } else {
            Fixed::ZERO
        };
        let error = logit - target;
        total = total + error * error;
    }

    total / Fixed::from_usize(logits.len())
}

/// Compute batch mean squared error over logits with one-hot targets.
///
/// `flat_logits` must contain `batch_size * num_classes` elements arranged
/// sample-major (`[sample0_class0, sample0_class1, ..., sampleN_classC-1]`).
pub fn mse_batch(flat_logits: &[Fixed], labels: &[usize], num_classes: usize) -> Fixed {
    assert!(num_classes > 0, "num_classes must be > 0");
    assert!(
        flat_logits.len() == labels.len() * num_classes,
        "logits length {} does not match labels {} × num_classes {}",
        flat_logits.len(),
        labels.len(),
        num_classes
    );

    if labels.is_empty() {
        return Fixed::ZERO;
    }

    let mut sum = Fixed::ZERO;
    for (sample_idx, &label) in labels.iter().enumerate() {
        let start = sample_idx * num_classes;
        let end = start + num_classes;
        let sample_logits = &flat_logits[start..end];
        sum = sum + mse_single(sample_logits, label);
    }

    sum / Fixed::from_usize(labels.len())
}

/// Compute loss delta: `loss_before - loss_after` using MSE-on-logits.
///
/// Positive values indicate improvement (loss decreased).
pub fn loss_delta_mse(before_logits: &[Fixed], after_logits: &[Fixed], true_label: usize) -> Fixed {
    assert_eq!(
        before_logits.len(),
        after_logits.len(),
        "logit vectors must have identical length"
    );
    let loss_before = mse_single(before_logits, true_label);
    let loss_after = mse_single(after_logits, true_label);
    loss_before - loss_after
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 0.002;

    fn assert_close(actual: Fixed, expected: f32) {
        let diff = (actual.to_f32() - expected).abs();
        assert!(
            diff <= EPS,
            "expected {expected}, got {}, diff {diff}",
            actual.to_f32()
        );
    }

    #[test]
    fn test_mse_single_matches_float_reference() {
        let logits = [
            Fixed::from_f32(2.15),
            Fixed::from_f32(-0.43),
            Fixed::from_f32(0.87),
        ];
        let true_label = 0usize;

        // Float reference as PyTorch-style computation
        let logits_f = [2.15f32, -0.43, 0.87];
        let mut total = 0.0f32;
        for (idx, &logit) in logits_f.iter().enumerate() {
            let target = if idx == true_label { 1.0 } else { 0.0 };
            let error = logit - target;
            total += error * error;
        }
        let expected = total / logits_f.len() as f32;

        let fixed_loss = mse_single(&logits, true_label);
        assert_close(fixed_loss, expected);
    }

    #[test]
    fn test_mse_batch_matches_float_reference() {
        let logits = vec![
            Fixed::from_f32(1.2),
            Fixed::from_f32(-0.3),
            Fixed::from_f32(0.1),
            Fixed::from_f32(-0.5),
            Fixed::from_f32(2.0),
            Fixed::from_f32(0.4),
        ];
        let labels = vec![0usize, 1usize];
        let num_classes = 3usize;

        let float_logits = [[1.2f32, -0.3, 0.1], [-0.5, 2.0, 0.4]];
        let mut total = 0.0f32;
        for (sample_idx, &label) in labels.iter().enumerate() {
            let mut sample_loss = 0.0f32;
            for (class_idx, &logit) in float_logits[sample_idx].iter().enumerate() {
                let target = if class_idx == label { 1.0 } else { 0.0 };
                let error = logit - target;
                sample_loss += error * error;
            }
            total += sample_loss / num_classes as f32;
        }
        let expected = total / labels.len() as f32;

        let fixed_loss = mse_batch(&logits, &labels, num_classes);
        assert_close(fixed_loss, expected);
    }

    #[test]
    fn test_loss_delta() {
        let before = [
            Fixed::from_f32(0.3),
            Fixed::from_f32(0.9),
            Fixed::from_f32(-0.1),
        ];
        let after = [
            Fixed::from_f32(0.1),
            Fixed::from_f32(1.2),
            Fixed::from_f32(0.0),
        ];
        let label = 1usize;

        let delta = loss_delta_mse(&before, &after, label);

        // Float reference
        let before_f = [0.3f32, 0.9, -0.1];
        let after_f = [0.1f32, 1.2, 0.0];
        let mse = |vals: &[f32], lbl: usize| -> f32 {
            let mut total = 0.0f32;
            for (idx, &logit) in vals.iter().enumerate() {
                let target = if idx == lbl { 1.0 } else { 0.0 };
                let error = logit - target;
                total += error * error;
            }
            total / vals.len() as f32
        };
        let expected = mse(&before_f, label) - mse(&after_f, label);
        assert_close(delta, expected);
        assert!(delta.to_f32() > -EPS); // ensure non-regression (should be positive)
    }
}
