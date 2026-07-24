// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Fail-closed validation for telemetry entering the Canvas pipeline.

use crate::CognitiveSnapshot;

/// Resource and range limits applied before scene generation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SnapshotLimits {
    /// Maximum accepted value for each Betti number.
    pub max_betti: usize,
    /// Maximum number of pairs retained in each persistence diagram.
    pub max_persistence_pairs: usize,
    /// Maximum number of thought-vector dimensions retained.
    pub max_thought_dimensions: usize,
    /// Absolute bound for each thought-vector component.
    pub max_thought_component: f32,
}

impl Default for SnapshotLimits {
    fn default() -> Self {
        Self {
            max_betti: 64,
            max_persistence_pairs: 64,
            max_thought_dimensions: 256,
            max_thought_component: 4.0,
        }
    }
}

/// Audit information describing how an input snapshot was corrected.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SnapshotSanitization {
    pub non_finite_replacements: usize,
    pub clamped_scalars: usize,
    pub truncated_persistence_pairs: usize,
    pub truncated_thought_dimensions: usize,
    pub clamped_topology_counts: usize,
    pub reordered_persistence_pairs: usize,
}

impl SnapshotSanitization {
    pub fn changed(&self) -> bool {
        self.non_finite_replacements > 0
            || self.clamped_scalars > 0
            || self.truncated_persistence_pairs > 0
            || self.truncated_thought_dimensions > 0
            || self.clamped_topology_counts > 0
            || self.reordered_persistence_pairs > 0
    }
}

impl CognitiveSnapshot {
    /// Sanitize using conservative production defaults.
    pub fn sanitized(&self) -> Self {
        self.sanitize_with_limits(SnapshotLimits::default()).0
    }

    /// Return a bounded snapshot plus an audit report of every correction.
    pub fn sanitize_with_limits(&self, limits: SnapshotLimits) -> (Self, SnapshotSanitization) {
        let mut report = SnapshotSanitization::default();
        let max_thought_component = finite_nonnegative(
            limits.max_thought_component,
            SnapshotLimits::default().max_thought_component,
        );

        let persistence_components = sanitize_pairs(
            &self.persistence_components,
            limits.max_persistence_pairs,
            &mut report,
        );
        let persistence_cycles = sanitize_pairs(
            &self.persistence_cycles,
            limits.max_persistence_pairs,
            &mut report,
        );

        let mut thought_vector =
            Vec::with_capacity(self.thought_vector.len().min(limits.max_thought_dimensions));
        for &value in self
            .thought_vector
            .iter()
            .take(limits.max_thought_dimensions)
        {
            thought_vector.push(clamp_f32(
                value,
                -max_thought_component,
                max_thought_component,
                0.0,
                &mut report,
            ));
        }
        report.truncated_thought_dimensions = self
            .thought_vector
            .len()
            .saturating_sub(limits.max_thought_dimensions);

        let betti_0 = clamp_count(self.betti_0, limits.max_betti, &mut report);
        let betti_1 = clamp_count(self.betti_1, limits.max_betti, &mut report);
        let betti_2 = clamp_count(self.betti_2, limits.max_betti, &mut report);

        let harmony_activations = std::array::from_fn(|i| {
            clamp_f32(self.harmony_activations[i], 0.0, 1.0, 0.0, &mut report)
        });

        (
            Self {
                consciousness_level: clamp_f64(
                    self.consciousness_level,
                    0.0,
                    1.0,
                    0.0,
                    &mut report,
                ),
                prediction_error: clamp_f32(self.prediction_error, 0.0, 1.0, 0.0, &mut report),
                living_mind_vitality: clamp_f64(
                    self.living_mind_vitality,
                    0.0,
                    1.0,
                    0.0,
                    &mut report,
                ),
                living_mind_coherence: clamp_f64(
                    self.living_mind_coherence,
                    0.0,
                    1.0,
                    0.0,
                    &mut report,
                ),
                dopamine: clamp_f32(self.dopamine, 0.0, 1.0, 0.0, &mut report),
                noradrenaline: clamp_f32(self.noradrenaline, 0.0, 1.0, 0.0, &mut report),
                serotonin: clamp_f32(self.serotonin, 0.0, 1.0, 0.0, &mut report),
                acetylcholine: clamp_f32(self.acetylcholine, 0.0, 1.0, 0.0, &mut report),
                oxytocin: clamp_f32(self.oxytocin, 0.0, 1.0, 0.0, &mut report),
                gaba: clamp_f32(self.gaba, 0.0, 1.0, 0.0, &mut report),
                allostatic_load: clamp_f32(self.allostatic_load, 0.0, 1.0, 0.0, &mut report),
                betti_0,
                betti_1,
                betti_2,
                persistence_components,
                persistence_cycles,
                cantor_metacognitive_depth: clamp_f32(
                    self.cantor_metacognitive_depth,
                    0.0,
                    1.0,
                    0.0,
                    &mut report,
                ),
                cantor_last_depth: self.cantor_last_depth.min(8),
                valence: clamp_f32(self.valence, -1.0, 1.0, 0.0, &mut report),
                arousal: clamp_f32(self.arousal, 0.0, 1.0, 0.0, &mut report),
                harmony_activations,
                thought_vector,
                cycle_count: self.cycle_count,
            },
            report,
        )
    }
}

fn sanitize_pairs(
    pairs: &[[f64; 2]],
    max_pairs: usize,
    report: &mut SnapshotSanitization,
) -> Vec<[f64; 2]> {
    report.truncated_persistence_pairs = report
        .truncated_persistence_pairs
        .saturating_add(pairs.len().saturating_sub(max_pairs));

    pairs
        .iter()
        .take(max_pairs)
        .map(|pair| {
            let mut birth = clamp_f64(pair[0], 0.0, 1.0, 0.0, report);
            let mut death = clamp_f64(pair[1], 0.0, 1.0, birth, report);
            if death < birth {
                std::mem::swap(&mut birth, &mut death);
                report.reordered_persistence_pairs += 1;
            }
            [birth, death]
        })
        .collect()
}

fn clamp_count(value: usize, max: usize, report: &mut SnapshotSanitization) -> usize {
    if value > max {
        report.clamped_topology_counts += 1;
        max
    } else {
        value
    }
}

fn finite_nonnegative(value: f32, fallback: f32) -> f32 {
    if value.is_finite() && value >= 0.0 {
        value
    } else {
        fallback
    }
}

fn clamp_f32(
    value: f32,
    min: f32,
    max: f32,
    fallback: f32,
    report: &mut SnapshotSanitization,
) -> f32 {
    if !value.is_finite() {
        report.non_finite_replacements += 1;
        return fallback;
    }
    let clamped = value.clamp(min, max);
    if clamped != value {
        report.clamped_scalars += 1;
    }
    clamped
}

fn clamp_f64(
    value: f64,
    min: f64,
    max: f64,
    fallback: f64,
    report: &mut SnapshotSanitization,
) -> f64 {
    if !value.is_finite() {
        report.non_finite_replacements += 1;
        return fallback;
    }
    let clamped = value.clamp(min, max);
    if clamped != value {
        report.clamped_scalars += 1;
    }
    clamped
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn malformed_snapshot_is_bounded_and_audited() {
        let mut snap = CognitiveSnapshot::dormant();
        snap.consciousness_level = f64::NAN;
        snap.prediction_error = f32::INFINITY;
        snap.valence = -9.0;
        snap.betti_0 = usize::MAX;
        snap.persistence_components = vec![[0.9, 0.1], [f64::NAN, 2.0]];
        snap.thought_vector = vec![f32::NAN; 300];

        let (clean, report) = snap.sanitize_with_limits(SnapshotLimits::default());
        assert!(report.changed());
        assert_eq!(clean.consciousness_level, 0.0);
        assert_eq!(clean.prediction_error, 0.0);
        assert_eq!(clean.valence, -1.0);
        assert_eq!(clean.betti_0, 64);
        assert_eq!(clean.persistence_components[0], [0.1, 0.9]);
        assert!(
            clean
                .persistence_components
                .iter()
                .flatten()
                .all(|v| v.is_finite())
        );
        assert_eq!(clean.thought_vector.len(), 256);
        assert!(clean.thought_vector.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn valid_snapshot_is_unchanged() {
        let snap = CognitiveSnapshot::dormant();
        let (clean, report) = snap.sanitize_with_limits(SnapshotLimits::default());
        assert!(!report.changed());
        assert_eq!(clean.consciousness_level, snap.consciousness_level);
        assert_eq!(clean.thought_vector, snap.thought_vector);
    }
}
