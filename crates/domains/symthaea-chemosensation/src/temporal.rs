// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Temporal context for chemical percepts.
//!
//! Smell and taste are temporal signals: onset, persistence, adaptation, and
//! recovery carry information that a single static fingerprint cannot. This
//! tracker keeps modality-specific anchors and refuses out-of-order evidence or
//! cross-coordinate-space comparisons.

use std::collections::HashMap;

use symthaea_core::hdc::unified_hv::ContinuousHV;

use crate::{ChemicalEncodingSpaceId, ChemicalModality, ChemicalPercept};

#[derive(Debug, Clone, PartialEq)]
pub struct ChemicalTemporalContext {
    pub previous_timestamp_us: Option<u64>,
    pub elapsed_s: Option<f32>,
    pub similarity_to_previous: Option<f32>,
    /// Bounded change score in [0, 1]. Zero means no representational change.
    pub change: f32,
    /// Change per second when a previous anchor exists.
    pub change_rate_per_s: Option<f32>,
    /// Whether this percept became the new temporal anchor.
    pub anchor_updated: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TemporalConfigError {
    InvalidMinimumConfidence(f32),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TemporalError {
    NonMonotonicTimestamp {
        modality: ChemicalModality,
        previous_timestamp_us: u64,
        current_timestamp_us: u64,
    },
    EncodingSpaceMismatch {
        modality: ChemicalModality,
        previous: ChemicalEncodingSpaceId,
        current: ChemicalEncodingSpaceId,
    },
}

#[derive(Debug, Clone)]
struct TemporalAnchor {
    timestamp_us: u64,
    vector: ContinuousHV,
    encoding_space_id: ChemicalEncodingSpaceId,
}

#[derive(Debug, Clone)]
pub struct ChemicalTemporalTracker {
    anchors: HashMap<ChemicalModality, TemporalAnchor>,
    min_anchor_confidence: f32,
}

impl ChemicalTemporalTracker {
    pub fn new(min_anchor_confidence: f32) -> Result<Self, TemporalConfigError> {
        if !min_anchor_confidence.is_finite() || !(0.0..=1.0).contains(&min_anchor_confidence) {
            return Err(TemporalConfigError::InvalidMinimumConfidence(
                min_anchor_confidence,
            ));
        }
        Ok(Self {
            anchors: HashMap::new(),
            min_anchor_confidence,
        })
    }

    pub fn min_anchor_confidence(&self) -> f32 {
        self.min_anchor_confidence
    }

    pub fn clear(&mut self) {
        self.anchors.clear();
    }

    /// Compare a percept with the previous anchor for the same modality.
    ///
    /// Low-confidence percepts may be assessed but do not replace a stronger
    /// anchor. Validation completes before state mutation. A changed encoding
    /// space is an integrity boundary: callers must explicitly clear/migrate
    /// temporal state rather than interpreting coordinate changes as chemistry.
    pub fn observe(
        &mut self,
        percept: &ChemicalPercept,
    ) -> Result<ChemicalTemporalContext, TemporalError> {
        let modality = percept.evidence.modality;
        let timestamp_us = percept.timestamp_us();
        let current_space = percept.fingerprint.encoding_space_id;
        let previous = self.anchors.get(&modality);

        if let Some(anchor) = previous {
            if timestamp_us <= anchor.timestamp_us {
                return Err(TemporalError::NonMonotonicTimestamp {
                    modality,
                    previous_timestamp_us: anchor.timestamp_us,
                    current_timestamp_us: timestamp_us,
                });
            }
            if current_space != anchor.encoding_space_id {
                return Err(TemporalError::EncodingSpaceMismatch {
                    modality,
                    previous: anchor.encoding_space_id,
                    current: current_space,
                });
            }
        }

        let (previous_timestamp_us, elapsed_s, similarity_to_previous, change, change_rate_per_s) =
            if let Some(anchor) = previous {
                let similarity = anchor
                    .vector
                    .similarity(&percept.fingerprint.vector)
                    .clamp(-1.0, 1.0);
                let change = (1.0 - similarity.max(0.0)).clamp(0.0, 1.0);
                let elapsed_s = (timestamp_us - anchor.timestamp_us) as f32 / 1_000_000.0;
                (
                    Some(anchor.timestamp_us),
                    Some(elapsed_s),
                    Some(similarity),
                    change,
                    Some(change / elapsed_s),
                )
            } else {
                (None, None, None, 0.0, None)
            };

        let anchor_updated = percept.confidence() >= self.min_anchor_confidence;
        if anchor_updated {
            self.anchors.insert(
                modality,
                TemporalAnchor {
                    timestamp_us,
                    vector: percept.fingerprint.vector.clone(),
                    encoding_space_id: current_space,
                },
            );
        }

        Ok(ChemicalTemporalContext {
            previous_timestamp_us,
            elapsed_s,
            similarity_to_previous,
            change,
            change_rate_per_s,
            anchor_updated,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ChemicalFingerprint, ChemicalObservation, ChemicalPercept, EnvironmentReading};
    use symthaea_core::hdc::HDC_DIMENSION;

    fn percept(modality: ChemicalModality, timestamp_us: u64, seed: u64, confidence: f32) -> ChemicalPercept {
        ChemicalPercept {
            evidence: ChemicalObservation {
                timestamp_us,
                modality,
                source: "fixture".into(),
                channels: vec![],
                environment: EnvironmentReading::default(),
            },
            fingerprint: ChemicalFingerprint {
                vector: ContinuousHV::random(HDC_DIMENSION, seed),
                confidence,
                used_channels: 1,
                ignored_channels: 0,
                encoding_space_id: ChemicalEncodingSpaceId::from_bytes([7; 32]),
            },
        }
    }

    #[test]
    fn first_percept_has_no_synthetic_previous_state() {
        let mut tracker = ChemicalTemporalTracker::new(0.5).unwrap();
        let context = tracker
            .observe(&percept(ChemicalModality::Olfactory, 1_000_000, 1, 0.9))
            .unwrap();
        assert_eq!(context.previous_timestamp_us, None);
        assert_eq!(context.change, 0.0);
        assert!(context.anchor_updated);
    }

    #[test]
    fn repeated_fingerprint_has_zero_change() {
        let mut tracker = ChemicalTemporalTracker::new(0.5).unwrap();
        let first = percept(ChemicalModality::Olfactory, 1_000_000, 1, 0.9);
        let mut second = first.clone();
        second.evidence.timestamp_us = 2_000_000;
        tracker.observe(&first).unwrap();
        let context = tracker.observe(&second).unwrap();
        assert!(context.change < 1e-6);
        assert!(context.change_rate_per_s.unwrap() < 1e-6);
    }

    #[test]
    fn smell_and_taste_have_independent_temporal_anchors() {
        let mut tracker = ChemicalTemporalTracker::new(0.5).unwrap();
        tracker
            .observe(&percept(ChemicalModality::Olfactory, 1_000_000, 1, 0.9))
            .unwrap();
        let taste = tracker
            .observe(&percept(ChemicalModality::Gustatory, 1_500_000, 1, 0.9))
            .unwrap();
        assert_eq!(taste.previous_timestamp_us, None);
    }

    #[test]
    fn low_confidence_sample_does_not_replace_anchor() {
        let mut tracker = ChemicalTemporalTracker::new(0.5).unwrap();
        tracker
            .observe(&percept(ChemicalModality::Olfactory, 1_000_000, 1, 0.9))
            .unwrap();
        let low = tracker
            .observe(&percept(ChemicalModality::Olfactory, 2_000_000, 2, 0.1))
            .unwrap();
        assert!(!low.anchor_updated);
        let later = tracker
            .observe(&percept(ChemicalModality::Olfactory, 3_000_000, 1, 0.9))
            .unwrap();
        assert_eq!(later.previous_timestamp_us, Some(1_000_000));
        assert!(later.change < 1e-6);
    }

    #[test]
    fn out_of_order_sample_is_rejected_without_mutation() {
        let mut tracker = ChemicalTemporalTracker::new(0.5).unwrap();
        tracker
            .observe(&percept(ChemicalModality::Olfactory, 2_000_000, 1, 0.9))
            .unwrap();
        assert!(matches!(
            tracker.observe(&percept(ChemicalModality::Olfactory, 1_000_000, 2, 0.9)),
            Err(TemporalError::NonMonotonicTimestamp { .. })
        ));
        let next = tracker
            .observe(&percept(ChemicalModality::Olfactory, 3_000_000, 1, 0.9))
            .unwrap();
        assert_eq!(next.previous_timestamp_us, Some(2_000_000));
        assert!(next.change < 1e-6);
    }

    #[test]
    fn changed_encoding_space_is_not_misread_as_temporal_change() {
        let mut tracker = ChemicalTemporalTracker::new(0.5).unwrap();
        let first = percept(ChemicalModality::Olfactory, 1_000_000, 1, 0.9);
        tracker.observe(&first).unwrap();

        let mut changed = percept(ChemicalModality::Olfactory, 2_000_000, 1, 0.9);
        changed.fingerprint.encoding_space_id = ChemicalEncodingSpaceId::from_bytes([8; 32]);
        assert!(matches!(
            tracker.observe(&changed),
            Err(TemporalError::EncodingSpaceMismatch { .. })
        ));

        let continuation = percept(ChemicalModality::Olfactory, 3_000_000, 1, 0.9);
        let context = tracker.observe(&continuation).unwrap();
        assert_eq!(context.previous_timestamp_us, Some(1_000_000));
        assert!(context.change < 1e-6);
    }
}
