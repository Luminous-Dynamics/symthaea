// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Sparse HDC delta frames for persistent SCIP sessions.

use crate::{HdcPayload, InterchangeError};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HdcDeltaEntry {
    pub index: u32,
    pub delta: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SparseHdcDelta {
    pub base_semantic_hash: String,
    pub semantic_hash: String,
    pub profile_fingerprint: String,
    pub dimension: usize,
    pub epsilon: f32,
    pub changes: Vec<HdcDeltaEntry>,
}

impl SparseHdcDelta {
    pub fn between(
        base: &HdcPayload,
        target: &HdcPayload,
        epsilon: f32,
    ) -> Result<Self, InterchangeError> {
        if !epsilon.is_finite() || epsilon < 0.0 {
            return Err(InterchangeError::InvalidDelta(
                "epsilon must be finite and >= 0".into(),
            ));
        }
        if base.values.len() != target.values.len()
            || base.profile_fingerprint != target.profile_fingerprint
            || base.values.len() > u32::MAX as usize
        {
            return Err(InterchangeError::InvalidDelta(
                "base and target HDC profiles/dimensions differ".into(),
            ));
        }
        if base
            .values
            .iter()
            .chain(&target.values)
            .any(|value| !value.is_finite())
        {
            return Err(InterchangeError::InvalidDelta(
                "base or target contains non-finite values".into(),
            ));
        }

        let changes = base
            .values
            .iter()
            .zip(&target.values)
            .enumerate()
            .filter_map(|(index, (&before, &after))| {
                let delta = after - before;
                (delta.abs() > epsilon).then_some(HdcDeltaEntry {
                    index: index as u32,
                    delta,
                })
            })
            .collect();

        Ok(Self {
            base_semantic_hash: base.semantic_hash.clone(),
            semantic_hash: target.semantic_hash.clone(),
            profile_fingerprint: base.profile_fingerprint.clone(),
            dimension: base.values.len(),
            epsilon,
            changes,
        })
    }

    pub fn apply(&self, base: &HdcPayload) -> Result<HdcPayload, InterchangeError> {
        if base.semantic_hash != self.base_semantic_hash
            || base.profile_fingerprint != self.profile_fingerprint
            || base.values.len() != self.dimension
        {
            return Err(InterchangeError::InvalidDelta(
                "delta base hash/profile/dimension mismatch".into(),
            ));
        }

        let mut values = base.values.clone();
        let mut previous = None;
        for change in &self.changes {
            let index = change.index as usize;
            if index >= values.len() || !change.delta.is_finite() {
                return Err(InterchangeError::InvalidDelta(
                    "invalid delta component".into(),
                ));
            }
            if previous.is_some_and(|previous| index <= previous) {
                return Err(InterchangeError::InvalidDelta(
                    "delta component indices must be strictly increasing".into(),
                ));
            }
            values[index] += change.delta;
            previous = Some(index);
        }

        Ok(HdcPayload {
            values,
            semantic_hash: self.semantic_hash.clone(),
            profile_fingerprint: self.profile_fingerprint.clone(),
        })
    }

    pub fn changed_fraction(&self) -> f32 {
        if self.dimension == 0 {
            0.0
        } else {
            self.changes.len() as f32 / self.dimension as f32
        }
    }

    pub fn is_sparse_enough(&self, maximum_changed_fraction: f32) -> bool {
        maximum_changed_fraction.is_finite()
            && maximum_changed_fraction >= 0.0
            && self.changed_fraction() <= maximum_changed_fraction
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn payload(values: Vec<f32>, hash: &str) -> HdcPayload {
        HdcPayload {
            values,
            semantic_hash: hash.into(),
            profile_fingerprint: "profile".into(),
        }
    }

    #[test]
    fn sparse_delta_round_trips_exactly_at_zero_epsilon() {
        let base = payload(vec![0.0, 1.0, 2.0, 3.0], "a");
        let target = payload(vec![0.0, 1.5, 2.0, 4.0], "b");
        let delta = SparseHdcDelta::between(&base, &target, 0.0).unwrap();
        assert_eq!(delta.changes.len(), 2);
        assert_eq!(delta.apply(&base).unwrap(), target);
    }

    #[test]
    fn wrong_base_is_rejected() {
        let base = payload(vec![0.0, 1.0], "a");
        let target = payload(vec![0.0, 2.0], "b");
        let delta = SparseHdcDelta::between(&base, &target, 0.0).unwrap();
        let wrong = payload(vec![0.0, 1.0], "wrong");
        assert!(delta.apply(&wrong).is_err());
    }
}
