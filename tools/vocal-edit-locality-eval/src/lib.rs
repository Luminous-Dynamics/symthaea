// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Measure whether a requested vocal edit changes its target interval while
//! preserving explicitly protected intervals.
//!
//! This is a time-domain locality metric, not a semantic quality metric. The
//! caller must supply dry, mono, sample-aligned before/after stems produced
//! from the same frozen source performance and fixed preprocessing. It does
//! not determine whether the corrected lyric, phone, note, or expression is
//! correct; pair it with phoneme/score checks and blinded listening.

#![forbid(unsafe_code)]

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RegionKind {
    Target,
    Protected,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EditRegion {
    pub label: String,
    pub kind: RegionKind,
    /// Half-open sample-frame interval.
    pub start_frame: usize,
    pub end_frame: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EditLocalityThresholds {
    /// Minimum target-region RMS sample delta, in the input amplitude units.
    pub minimum_target_rms_delta: f64,
    /// Maximum allowed RMS sample delta in each protected region, in input units.
    pub maximum_protected_rms_delta: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RegionMetrics {
    pub label: String,
    pub kind: RegionKind,
    pub start_frame: usize,
    pub end_frame: usize,
    pub sample_count: usize,
    pub before_rms: f64,
    pub after_rms: f64,
    /// sqrt(mean((before - after)^2)), in input amplitude units.
    pub rms_delta: f64,
    /// rms_delta / max(before_rms, 1e-12); undefined only for no samples.
    pub normalized_rms_delta: Option<f64>,
    /// Pearson correlation. None for constant/near-constant windows.
    pub correlation: Option<f64>,
    /// Fraction of frames whose f32 sample values are bit-identical.
    pub exact_sample_fraction: f64,
    pub maximum_absolute_delta: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EditLocalityReport {
    pub sample_rate_hz: u32,
    pub total_frames: usize,
    pub target: RegionMetrics,
    pub protected: Vec<RegionMetrics>,
    pub aggregate_protected_rms_delta: f64,
    /// Aggregate protected RMS delta divided by target RMS delta.
    /// None when the target did not change.
    pub collateral_to_target_ratio: Option<f64>,
    pub thresholds: EditLocalityThresholds,
    pub objective_pass: bool,
}

/// Evaluate one target edit against a complete, non-overlapping partition of
/// the before/after audio. Regions must start at frame 0 and end at the exact
/// shared frame count. There must be exactly one target and at least one
/// protected region.
///
/// The caller must align stems to the same sample clock before calling this
/// function. A timing shift, resampling mismatch, changed channel mix, or
/// changed loudness preprocessing is measured as collateral drift; it must
/// not be hidden by internal alignment.
pub fn score_edit_locality(
    sample_rate_hz: u32,
    before: &[f32],
    after: &[f32],
    regions: &[EditRegion],
    thresholds: EditLocalityThresholds,
) -> Result<EditLocalityReport, String> {
    validate_inputs(sample_rate_hz, before, after, regions, thresholds)?;

    let mut target_metrics = None;
    let mut protected_metrics = Vec::new();
    let mut protected_squared_delta = 0.0f64;
    let mut protected_sample_count = 0usize;

    for region in regions {
        let metrics = measure_region(before, after, region);
        match region.kind {
            RegionKind::Target => target_metrics = Some(metrics),
            RegionKind::Protected => {
                protected_squared_delta +=
                    metrics.rms_delta * metrics.rms_delta * metrics.sample_count as f64;
                protected_sample_count += metrics.sample_count;
                protected_metrics.push(metrics);
            }
        }
    }

    let target_metrics =
        target_metrics.ok_or_else(|| "exactly one target region is required".to_string())?;
    let aggregate_protected_rms_delta =
        (protected_squared_delta / protected_sample_count as f64).sqrt();
    let collateral_to_target_ratio = (target_metrics.rms_delta > 0.0)
        .then(|| aggregate_protected_rms_delta / target_metrics.rms_delta);
    let objective_pass = target_metrics.rms_delta >= thresholds.minimum_target_rms_delta
        && protected_metrics
            .iter()
            .all(|region| region.rms_delta <= thresholds.maximum_protected_rms_delta);

    Ok(EditLocalityReport {
        sample_rate_hz,
        total_frames: before.len(),
        target: target_metrics,
        protected: protected_metrics,
        aggregate_protected_rms_delta,
        collateral_to_target_ratio,
        thresholds,
        objective_pass,
    })
}

fn validate_inputs(
    sample_rate_hz: u32,
    before: &[f32],
    after: &[f32],
    regions: &[EditRegion],
    thresholds: EditLocalityThresholds,
) -> Result<(), String> {
    if sample_rate_hz == 0 {
        return Err("sample_rate_hz must be positive".into());
    }
    if before.is_empty() || before.len() != after.len() {
        return Err("before and after must be non-empty and have identical frame counts".into());
    }
    if regions.len() < 2 {
        return Err("at least one target and one protected region are required".into());
    }
    if !thresholds.minimum_target_rms_delta.is_finite()
        || thresholds.minimum_target_rms_delta <= 0.0
        || !thresholds.maximum_protected_rms_delta.is_finite()
        || thresholds.maximum_protected_rms_delta < 0.0
    {
        return Err(
            "minimum target RMS delta must be finite and positive; maximum protected RMS delta must be finite and non-negative".into(),
        );
    }
    for (index, sample) in before.iter().enumerate() {
        if !sample.is_finite() {
            return Err(format!("before sample {index} is not finite"));
        }
    }
    for (index, sample) in after.iter().enumerate() {
        if !sample.is_finite() {
            return Err(format!("after sample {index} is not finite"));
        }
    }

    let mut cursor = 0usize;
    let mut target_count = 0usize;
    let mut protected_count = 0usize;
    for (index, region) in regions.iter().enumerate() {
        if region.label.trim().is_empty() {
            return Err(format!("region {index} has an empty label"));
        }
        if regions[..index]
            .iter()
            .any(|previous| previous.label == region.label)
        {
            return Err(format!("region {index} has a duplicate label"));
        }
        if region.start_frame != cursor
            || region.start_frame >= region.end_frame
            || region.end_frame > before.len()
        {
            return Err(format!(
                "region {index} must be a non-empty, contiguous, in-range partition interval"
            ));
        }
        cursor = region.end_frame;
        match region.kind {
            RegionKind::Target => target_count += 1,
            RegionKind::Protected => protected_count += 1,
        }
    }
    if cursor != before.len() {
        return Err("regions must cover the full audio frame range".into());
    }
    if target_count != 1 {
        return Err(format!("exactly one target region is required; got {target_count}"));
    }
    if protected_count == 0 {
        return Err("at least one protected region is required".into());
    }
    Ok(())
}

fn measure_region(before: &[f32], after: &[f32], region: &EditRegion) -> RegionMetrics {
    let left = &before[region.start_frame..region.end_frame];
    let right = &after[region.start_frame..region.end_frame];
    let count = left.len();
    let count_f64 = count as f64;

    let before_energy = left.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>();
    let after_energy = right.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>();
    let mut delta_energy = 0.0f64;
    let mut maximum_absolute_delta = 0.0f64;
    let mut exact_matches = 0usize;

    for (&a, &b) in left.iter().zip(right) {
        let delta = (f64::from(a) - f64::from(b)).abs();
        delta_energy += delta * delta;
        maximum_absolute_delta = maximum_absolute_delta.max(delta);
        if a.to_bits() == b.to_bits() {
            exact_matches += 1;
        }
    }

    let before_mean = left.iter().map(|x| f64::from(*x)).sum::<f64>() / count_f64;
    let after_mean = right.iter().map(|x| f64::from(*x)).sum::<f64>() / count_f64;
    let mut covariance = 0.0f64;
    let mut before_variance = 0.0f64;
    let mut after_variance = 0.0f64;
    for (&a, &b) in left.iter().zip(right) {
        let a_centered = f64::from(a) - before_mean;
        let b_centered = f64::from(b) - after_mean;
        covariance += a_centered * b_centered;
        before_variance += a_centered * a_centered;
        after_variance += b_centered * b_centered;
    }

    let before_rms = (before_energy / count_f64).sqrt();
    let after_rms = (after_energy / count_f64).sqrt();
    let rms_delta = (delta_energy / count_f64).sqrt();
    let normalized_rms_delta = Some(rms_delta / before_rms.max(1e-12));
    let correlation_denominator = (before_variance * after_variance).sqrt();
    let correlation = (correlation_denominator > 1e-24)
        .then(|| (covariance / correlation_denominator).clamp(-1.0, 1.0));

    RegionMetrics {
        label: region.label.clone(),
        kind: region.kind,
        start_frame: region.start_frame,
        end_frame: region.end_frame,
        sample_count: count,
        before_rms,
        after_rms,
        rms_delta,
        normalized_rms_delta,
        correlation,
        exact_sample_fraction: exact_matches as f64 / count_f64,
        maximum_absolute_delta,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn thresholds(min_target: f64, max_protected: f64) -> EditLocalityThresholds {
        EditLocalityThresholds {
            minimum_target_rms_delta: min_target,
            maximum_protected_rms_delta: max_protected,
        }
    }

    fn regions() -> Vec<EditRegion> {
        vec![
            EditRegion {
                label: "intro".into(),
                kind: RegionKind::Protected,
                start_frame: 0,
                end_frame: 4,
            },
            EditRegion {
                label: "corrected-word".into(),
                kind: RegionKind::Target,
                start_frame: 4,
                end_frame: 8,
            },
            EditRegion {
                label: "outro".into(),
                kind: RegionKind::Protected,
                start_frame: 8,
                end_frame: 12,
            },
        ]
    }

    fn source() -> Vec<f32> {
        vec![
            0.1, 0.2, -0.1, -0.2,
            0.2, 0.4, -0.2, -0.4,
            0.15, 0.3, -0.15, -0.3,
        ]
    }

    #[test]
    fn target_only_change_passes_and_protected_samples_are_exact() {
        let before = source();
        let mut after = before.clone();
        after[4..8].iter_mut().for_each(|sample| *sample *= 0.5);

        let report =
            score_edit_locality(48_000, &before, &after, &regions(), thresholds(0.01, 0.0))
                .unwrap();

        assert!(report.objective_pass);
        assert!(report.target.rms_delta > 0.0);
        assert_eq!(report.protected.len(), 2);
        assert_eq!(report.protected[0].exact_sample_fraction, 1.0);
        assert_eq!(report.protected[1].exact_sample_fraction, 1.0);
        assert_eq!(report.aggregate_protected_rms_delta, 0.0);
        assert_eq!(report.collateral_to_target_ratio, Some(0.0));
    }

    #[test]
    fn no_target_change_fails_even_when_all_audio_is_preserved() {
        let before = source();
        let report =
            score_edit_locality(48_000, &before, &before, &regions(), thresholds(0.001, 0.0))
                .unwrap();

        assert!(!report.objective_pass);
        assert_eq!(report.target.rms_delta, 0.0);
        assert_eq!(report.collateral_to_target_ratio, None);
    }

    #[test]
    fn collateral_change_in_protected_phrase_fails() {
        let before = source();
        let mut after = before.clone();
        after[4..8].iter_mut().for_each(|sample| *sample *= 0.5);
        after[8..12].iter_mut().for_each(|sample| *sample *= 0.2);

        let report =
            score_edit_locality(48_000, &before, &after, &regions(), thresholds(0.01, 0.001))
                .unwrap();

        assert!(!report.objective_pass);
        assert!(report.protected[1].rms_delta > 0.001);
        assert!(report.collateral_to_target_ratio.unwrap() > 0.0);
    }

    #[test]
    fn gaps_overlaps_and_missing_regions_fail_closed() {
        let audio = source();
        let mut gap = regions();
        gap[1].start_frame = 5;
        assert!(score_edit_locality(48_000, &audio, &audio, &gap, thresholds(0.0, 0.0)).is_err());

        let mut overlap = regions();
        overlap[1].start_frame = 3;
        assert!(
            score_edit_locality(48_000, &audio, &audio, &overlap, thresholds(0.0, 0.0)).is_err()
        );

        assert!(
            score_edit_locality(
                48_000,
                &audio,
                &audio,
                &regions()[1..2],
                thresholds(0.0, 0.0)
            )
            .is_err()
        );
    }

    #[test]
    fn length_rate_and_non_finite_samples_fail_closed() {
        let before = source();
        let mut non_finite = before.clone();
        non_finite[2] = f32::NAN;

        assert!(score_edit_locality(0, &before, &before, &regions(), thresholds(0.0, 0.0)).is_err());
        assert!(
            score_edit_locality(
                48_000,
                &before,
                &before[..11],
                &regions(),
                thresholds(0.0, 0.0)
            )
            .is_err()
        );
        assert!(
            score_edit_locality(
                48_000,
                &before,
                &non_finite,
                &regions(),
                thresholds(0.0, 0.0)
            )
            .is_err()
        );
    }

    #[test]
    fn zero_minimum_target_change_and_duplicate_labels_are_rejected() {
        let audio = source();
        assert!(
            score_edit_locality(
                48_000,
                &audio,
                &audio,
                &regions(),
                thresholds(0.0, 0.0)
            )
            .is_err()
        );

        let mut duplicated = regions();
        duplicated[1].label = "intro".into();
        assert!(
            score_edit_locality(
                48_000,
                &audio,
                &audio,
                &duplicated,
                thresholds(0.001, 0.0)
            )
            .is_err()
        );
    }

    #[test]
    fn silent_protected_region_collateral_is_detected_without_correlation() {
        let before = vec![0.0f32; 12];
        let mut after = before.clone();
        after[0] = 0.2;
        let report =
            score_edit_locality(48_000, &before, &after, &regions(), thresholds(0.01, 0.0))
                .unwrap();

        assert!(!report.objective_pass);
        assert_eq!(report.protected[0].correlation, None);
        assert!(report.protected[0].rms_delta > 0.0);
        assert_eq!(report.protected[1].rms_delta, 0.0);
    }
}
