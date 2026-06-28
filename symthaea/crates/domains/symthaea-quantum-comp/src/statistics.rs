//! Small dependency-free statistics helpers for research reports.
//!
//! These helpers are intentionally simple. They are meant for reproducible
//! crate-local summaries, not for replacing a full statistical package.

/// Summary statistics for a finite sample.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SampleSummary {
    /// Number of samples.
    pub count: usize,
    /// Arithmetic mean.
    pub mean: f32,
    /// Unbiased sample variance when `count > 1`, otherwise `0`.
    pub variance: f32,
    /// Sample standard deviation.
    pub std_dev: f32,
    /// Standard error of the mean.
    pub stderr: f32,
    /// Minimum sample value.
    pub min: f32,
    /// Maximum sample value.
    pub max: f32,
}

impl SampleSummary {
    /// Builds a summary from a nonempty sample slice.
    pub fn from_samples(samples: &[f32]) -> Option<Self> {
        if samples.is_empty() {
            return None;
        }
        let count = samples.len();
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        let mut sum = 0.0f32;
        for &x in samples {
            min = min.min(x);
            max = max.max(x);
            sum += x;
        }
        let mean = sum / count as f32;
        let variance = if count > 1 {
            samples
                .iter()
                .map(|x| {
                    let d = *x - mean;
                    d * d
                })
                .sum::<f32>()
                / (count as f32 - 1.0)
        } else {
            0.0
        };
        let std_dev = variance.sqrt();
        let stderr = if count == 0 {
            0.0
        } else {
            std_dev / (count as f32).sqrt()
        };
        Some(Self {
            count,
            mean,
            variance,
            std_dev,
            stderr,
            min,
            max,
        })
    }

    /// Approximate symmetric 95% confidence interval around the mean.
    ///
    /// This uses a normal approximation (`1.96 * stderr`) and is only a compact
    /// report convenience. Use stronger statistical tooling for publication.
    pub fn approximate_95_ci(&self) -> (f32, f32) {
        let half_width = 1.96 * self.stderr;
        (self.mean - half_width, self.mean + half_width)
    }
}

/// Computes a trapezoidal area under a curve sorted by `x`.
pub fn trapezoid_auc(points: &[(f32, f32)]) -> Option<f32> {
    if points.len() < 2 {
        return None;
    }
    let mut area = 0.0f32;
    for window in points.windows(2) {
        let (x0, y0) = window[0];
        let (x1, y1) = window[1];
        area += (x1 - x0).abs() * (y0 + y1) * 0.5;
    }
    Some(area)
}

/// Computes a simple least-squares slope for `y = a + slope*x`.
pub fn linear_slope(points: &[(f32, f32)]) -> Option<f32> {
    if points.len() < 2 {
        return None;
    }
    let n = points.len() as f32;
    let mean_x = points.iter().map(|p| p.0).sum::<f32>() / n;
    let mean_y = points.iter().map(|p| p.1).sum::<f32>() / n;
    let mut num = 0.0f32;
    let mut den = 0.0f32;
    for &(x, y) in points {
        let dx = x - mean_x;
        num += dx * (y - mean_y);
        den += dx * dx;
    }
    if den == 0.0 { None } else { Some(num / den) }
}

/// Paired effect size: mean(a-b) divided by standard deviation of paired differences.
pub fn paired_effect_size(a: &[f32], b: &[f32]) -> Option<f32> {
    if a.len() != b.len() || a.is_empty() {
        return None;
    }
    let diffs: Vec<f32> = a.iter().zip(b).map(|(x, y)| x - y).collect();
    let summary = SampleSummary::from_samples(&diffs)?;
    if summary.std_dev == 0.0 {
        None
    } else {
        Some(summary.mean / summary.std_dev)
    }
}

/// Returns the first x value where y drops below `floor`.
pub fn first_threshold_crossing(points: &[(f32, f32)], floor: f32) -> Option<f32> {
    points
        .iter()
        .find_map(|(x, y)| if *y < floor { Some(*x) } else { None })
}

/// Counts adjacent increases in a curve that is expected to be non-increasing.
pub fn non_increasing_violations(points: &[(f32, f32)], tolerance: f32) -> usize {
    points
        .windows(2)
        .filter(|pair| pair[1].1 > pair[0].1 + tolerance)
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_summary_reports_mean() {
        let s = SampleSummary::from_samples(&[1.0, 2.0, 3.0]).unwrap();
        assert_eq!(s.count, 3);
        assert!((s.mean - 2.0).abs() < 1e-6);
    }

    #[test]
    fn auc_and_slope_are_defined() {
        let points = [(0.0, 1.0), (1.0, 0.0)];
        assert!((trapezoid_auc(&points).unwrap() - 0.5).abs() < 1e-6);
        assert!(linear_slope(&points).unwrap() < 0.0);
    }
}
