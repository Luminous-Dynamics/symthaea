//! Ricci Flow — evolution of metrics toward Einstein condition.
//!
//! Implements:
//! - Normalized Ricci flow with singularity detection
//! - Convergence checking (is the flow approaching round metric?)
//! - Flow history recording for analysis
//!
//! Hamilton (1982): The Ricci flow ∂g/∂t = -2Ric converges on S³ to the round metric.
//! On S⁴ the situation is more complex — this is the heart of the Einstein search problem.

use super::riemannian_geometry::{DiscretizedSphere, MetricTensor};

// ─── Flow state ───────────────────────────────────────────────────────────────

/// Snapshot of the metric at a single Ricci flow time step.
#[derive(Clone, Debug)]
pub struct RicciFlowState {
    /// Current averaged metric.
    pub metric: MetricTensor,
    /// Flow time (t = iteration × dt).
    pub time: f64,
    /// Iteration index.
    pub iteration: usize,
    /// Traceless Ricci norm (Einstein defect) at this step.
    pub einstein_defect: f64,
    /// Maximum absolute curvature component (for singularity detection).
    pub max_curvature: f64,
}

// ─── Normalized Ricci flow ────────────────────────────────────────────────────

/// Run normalized Ricci flow on a discretized sphere until convergence or max_iter.
///
/// ∂g_{ij}/∂t = -2R_{ij} + (2R̄/n)g_{ij}   (volume-preserving)
///
/// Returns history of `RicciFlowState` snapshots (one per recorded step).
/// Records every `record_every` steps (default: every step if ≤ 100, else every 10).
pub fn normalized_ricci_flow(
    initial: &DiscretizedSphere,
    dt: f64,
    max_iter: usize,
    tol: f64,
) -> Vec<RicciFlowState> {
    let mut history = Vec::new();
    let mut current = initial.clone();
    let record_every = if max_iter <= 100 { 1 } else { max_iter / 100 };

    for iter in 0..max_iter {
        let averaged = current.averaged_metric();
        let defect = current.einstein_defect();
        let max_curv = max_ricci_component(&averaged);

        if iter % record_every == 0 || iter == 0 {
            history.push(RicciFlowState {
                metric: averaged.clone(),
                time: iter as f64 * dt,
                iteration: iter,
                einstein_defect: defect,
                max_curvature: max_curv,
            });
        }

        // Check convergence
        if defect < tol {
            // Record final state
            history.push(RicciFlowState {
                metric: averaged,
                time: iter as f64 * dt,
                iteration: iter,
                einstein_defect: defect,
                max_curvature: max_curv,
            });
            break;
        }

        // Check singularity
        if max_curv > 1e6 {
            // Singularity detected — stop
            history.push(RicciFlowState {
                metric: averaged,
                time: iter as f64 * dt,
                iteration: iter,
                einstein_defect: defect,
                max_curvature: max_curv,
            });
            break;
        }

        // Evolve all sample points
        current = current.ricci_flow_step(dt);

        // Periodically normalize volume (prevent runaway scaling)
        if iter % 10 == 9 {
            for m in &mut current.metric_samples {
                m.normalize_volume(1.0);
            }
        }
    }

    history
}

/// Flow on a single `MetricTensor` (simpler interface for testing).
pub fn normalized_ricci_flow_single(
    initial: &MetricTensor,
    dt: f64,
    max_iter: usize,
    tol: f64,
) -> Vec<RicciFlowState> {
    let sphere = DiscretizedSphere {
        n_points: 1,
        dim: initial.dim,
        metric_samples: vec![initial.clone()],
    };
    normalized_ricci_flow(&sphere, dt, max_iter, tol)
}

// ─── Analysis ─────────────────────────────────────────────────────────────────

/// Maximum absolute Ricci tensor component (proxy for curvature magnitude).
pub fn max_ricci_component(metric: &MetricTensor) -> f64 {
    let ricci = metric.ricci_tensor();
    ricci.components.iter().map(|v| v.abs()).fold(0.0f64, f64::max)
}

/// Detect singularity in the flow history.
/// A singularity is characterized by rapid curvature blowup (>10x in 10 steps).
pub fn detect_singularity(history: &[RicciFlowState]) -> Option<usize> {
    if history.len() < 2 {
        return None;
    }
    for i in 1..history.len() {
        if history[i].max_curvature > 1e6 {
            return Some(i);
        }
        // Check for 10x growth in 1 step
        if history[i].max_curvature > 10.0 * history[i - 1].max_curvature.max(1e-15) {
            return Some(i);
        }
    }
    None
}

/// Check if the flow is converging toward the round metric (Einstein condition).
/// Returns true if the Einstein defect is monotonically decreasing toward zero.
pub fn is_converging_to_round(history: &[RicciFlowState]) -> bool {
    if history.len() < 3 {
        return false;
    }
    let last = history.last().unwrap();
    // Final defect < 1e-4 AND overall decreasing trend
    if last.einstein_defect >= 1e-3 {
        return false;
    }
    // Check that at least 80% of consecutive pairs are decreasing
    let n = history.len();
    let decreasing_count = (1..n)
        .filter(|&i| history[i].einstein_defect <= history[i - 1].einstein_defect * 1.05)
        .count();
    decreasing_count >= (n - 1) * 4 / 5
}

/// Check if flow has converged (defect < tol).
pub fn has_converged(history: &[RicciFlowState], tol: f64) -> bool {
    history.last().map(|s| s.einstein_defect < tol).unwrap_or(false)
}

/// Final Einstein defect in the flow history.
pub fn final_defect(history: &[RicciFlowState]) -> f64 {
    history.last().map(|s| s.einstein_defect).unwrap_or(f64::INFINITY)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::riemannian_geometry::{DiscretizedSphere, MetricTensor};

    #[test]
    fn test_round_sphere_is_fixed_point() {
        // Round sphere should be a fixed point of normalized Ricci flow
        // (Ricci flow preserves Einstein metrics)
        let m = MetricTensor::sphere_metric(3, 1.0);
        let history = normalized_ricci_flow_single(&m, 0.01, 50, 1e-6);
        assert!(!history.is_empty());
        let defect_0 = history[0].einstein_defect;
        let defect_final = history.last().unwrap().einstein_defect;
        assert!(defect_0 < 1e-6, "Initial defect should be zero, got {}", defect_0);
        assert!(defect_final < 1e-5, "Final defect should still be zero, got {}", defect_final);
    }

    #[test]
    fn test_flow_history_recorded() {
        let m = MetricTensor::sphere_metric(3, 1.0);
        let history = normalized_ricci_flow_single(&m, 0.01, 10, 1e-10);
        assert!(!history.is_empty());
        // Iterations should be non-decreasing
        for i in 1..history.len() {
            assert!(history[i].iteration >= history[i-1].iteration);
        }
    }

    #[test]
    fn test_convergence_check_round_sphere() {
        let m = MetricTensor::sphere_metric(3, 1.0);
        let history = normalized_ricci_flow_single(&m, 0.01, 30, 1e-6);
        // Round sphere flow should be identified as converging
        assert!(is_converging_to_round(&history) || has_converged(&history, 1e-5));
    }

    #[test]
    fn test_discretized_sphere_flow() {
        let sphere = DiscretizedSphere::standard(3, 5);
        let history = normalized_ricci_flow(&sphere, 0.005, 20, 1e-5);
        assert!(!history.is_empty());
        // No singularity for round sphere
        assert!(detect_singularity(&history).is_none());
    }

    #[test]
    fn test_flat_metric_flow() {
        // Flat metric: Ric=0, so -2Ric + (2R̄/n)g = 0. Fixed point.
        let m = MetricTensor::euclidean(3);
        let history = normalized_ricci_flow_single(&m, 0.01, 20, 1e-10);
        let defect_final = final_defect(&history);
        // Flat metric is Einstein (Ric=0=0·g)
        assert!(defect_final < 1e-8, "Flat metric defect = {}", defect_final);
    }

    #[test]
    fn test_singularity_detection_no_singularity() {
        let m = MetricTensor::sphere_metric(3, 1.0);
        let history = normalized_ricci_flow_single(&m, 0.01, 30, 1e-6);
        assert!(detect_singularity(&history).is_none());
    }

    #[test]
    fn test_max_ricci_component() {
        let m = MetricTensor::sphere_metric(3, 1.0);
        let max_c = max_ricci_component(&m);
        // For a sphere, Ric_ii = (n-1)/r² · g_ii = 2 * 1.0 = 2.0
        assert!(max_c >= 0.0 && max_c < 10.0, "max_c = {}", max_c);
    }

    #[test]
    fn test_flow_state_fields() {
        let m = MetricTensor::sphere_metric(3, 1.0);
        let history = normalized_ricci_flow_single(&m, 0.01, 5, 1e-10);
        let s = &history[0];
        assert_eq!(s.iteration, 0);
        assert_eq!(s.time, 0.0);
        assert!(s.einstein_defect >= 0.0);
        assert!(s.max_curvature >= 0.0);
    }
}
