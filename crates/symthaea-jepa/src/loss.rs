// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! JEPA loss functions: cosine similarity in latent space.
//!
//! JEPA does NOT use reconstruction loss. Instead, it measures similarity
//! between the predictor's output and the target encoder's output in the
//! learned latent space. Cosine loss naturally normalizes the latent space,
//! preventing magnitude collapse.

/// Cosine similarity between two latent vectors.
///
/// Returns value in [-1, 1]. Identical vectors → 1.0, orthogonal → 0.0.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for (&ai, &bi) in a.iter().zip(b.iter()) {
        dot += ai * bi;
        norm_a += ai * ai;
        norm_b += bi * bi;
    }
    let denom = (norm_a.sqrt() * norm_b.sqrt()).max(1e-8);
    (dot / denom).clamp(-1.0, 1.0)
}

/// Cosine loss: 1.0 - cosine_similarity.
///
/// Range: [0.0, 2.0]. Identical → 0.0, orthogonal → 1.0, opposite → 2.0.
pub fn cosine_loss(predicted: &[f32], target: &[f32]) -> f32 {
    1.0 - cosine_similarity(predicted, target)
}

/// Compute gradient of cosine loss w.r.t. `predicted`.
///
/// d(1 - cos(p,t))/dp = -d(cos(p,t))/dp
///
/// d(cos(p,t))/dp_i = (t_i / (||p|| * ||t||)) - (cos(p,t) * p_i / ||p||^2)
pub fn cosine_loss_gradient(predicted: &[f32], target: &[f32]) -> Vec<f32> {
    debug_assert_eq!(predicted.len(), target.len());

    let mut dot = 0.0f32;
    let mut norm_p = 0.0f32;
    let mut norm_t = 0.0f32;
    for (&pi, &ti) in predicted.iter().zip(target.iter()) {
        dot += pi * ti;
        norm_p += pi * pi;
        norm_t += ti * ti;
    }

    let norm_p = norm_p.sqrt().max(1e-8);
    let norm_t = norm_t.sqrt().max(1e-8);
    let cos_sim = dot / (norm_p * norm_t);
    let inv_pt = 1.0 / (norm_p * norm_t);
    let inv_pp = 1.0 / (norm_p * norm_p);

    // Gradient of cosine loss = -gradient of cosine similarity
    predicted
        .iter()
        .zip(target.iter())
        .map(|(&pi, &ti)| -(ti * inv_pt - cos_sim * pi * inv_pp))
        .collect()
}

/// Prediction error signal for FEP integration.
///
/// Same as cosine loss but clamped to [0, 1] for compatibility with
/// the FEP bridge's learning signal composition (expects PE in [0, 1]).
pub fn prediction_error(predicted: &[f32], target: &[f32]) -> f32 {
    cosine_loss(predicted, target).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identical_vectors() {
        let a = vec![1.0, 2.0, 3.0];
        assert!((cosine_similarity(&a, &a) - 1.0).abs() < 1e-6);
        assert!(cosine_loss(&a, &a) < 1e-6);
    }

    #[test]
    fn test_orthogonal_vectors() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-6);
        assert!((cosine_loss(&a, &b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_opposite_vectors() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![-1.0, -2.0, -3.0];
        assert!((cosine_similarity(&a, &b) - (-1.0)).abs() < 1e-6);
        assert!((cosine_loss(&a, &b) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_gradient_direction() {
        // Gradient should point in a direction that decreases cosine loss
        let predicted = vec![1.0, 0.5, -0.3];
        let target = vec![0.5, 1.0, 0.2];
        let grad = cosine_loss_gradient(&predicted, &target);

        // Applying a small step in negative gradient direction should reduce loss
        let lr = 0.01;
        let updated: Vec<f32> = predicted
            .iter()
            .zip(grad.iter())
            .map(|(p, g)| p - lr * g)
            .collect();
        let loss_before = cosine_loss(&predicted, &target);
        let loss_after = cosine_loss(&updated, &target);
        assert!(
            loss_after < loss_before,
            "gradient descent should reduce loss"
        );
    }

    #[test]
    fn test_prediction_error_clamped() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        // Opposite vectors have cosine_loss = 2.0, but PE clamps to 1.0
        assert!((prediction_error(&a, &b) - 1.0).abs() < 1e-6);
    }
}
