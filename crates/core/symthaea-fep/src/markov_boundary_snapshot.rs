// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Versioned persistence contract for exact [`crate::MarkovBoundaryOperator`] continuation.
//!
//! The live boundary operator carries EMA and ring-buffer history that changes future sensory
//! gating, trend telemetry, and coalescence readiness. Recreating an organism from partition
//! dimensions alone would therefore reset causal state even if its FEP agent were restored exactly.
//!
//! The EMA and history ring are intentionally validated as **distinct** state. The live
//! [`crate::MarkovBoundaryOperator::apply_topology_constraints`] path may adjust the EMA without
//! writing the history ring, so a validator that forced the latest history value to equal the
//! current EMA would reject legitimate production states and erase a real causal distinction.
//!
//! This module defines persistence semantics only. A later wiring tranche may teach the live
//! operator to emit/restore this snapshot through its private fields. Until that wiring and
//! split-run equivalence are qualified, this is not a claim that live Markov-boundary resume is
//! established.

use serde::{Deserialize, Serialize};

use crate::{BlanketPermeability, MarkovPartition};

/// Exact ring capacity used by the v1 live Markov-boundary operator.
pub const MARKOV_BOUNDARY_HISTORY_CAP_V1: usize = 64;

const PERMEABILITY_FLOOR_V1: f64 = 0.05;
const PERMEABILITY_CEILING_V1: f64 = 0.95;
const PRISTINE_PERMEABILITY_V1: f64 = 0.5;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarkovBoundarySnapshotV1 {
    pub(crate) partition: MarkovPartition,
    pub(crate) permeability: BlanketPermeability,
    pub(crate) permeability_ema: BlanketPermeability,
    pub(crate) alpha: f64,
    pub(crate) history: Vec<f64>,
    pub(crate) history_idx: usize,
    pub(crate) history_count: usize,
}

/// Non-serializable capability produced only after validating persisted Markov-boundary state.
#[derive(Debug, Clone)]
pub struct ValidatedMarkovBoundarySnapshotV1 {
    snapshot: MarkovBoundarySnapshotV1,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MarkovBoundarySnapshotErrorV1 {
    ZeroDimension { field: &'static str },
    NonFinite { field: &'static str },
    OutOfRange { field: &'static str },
    HistoryCapacityMismatch { observed: usize, expected: usize },
    HistoryIndexOutOfBounds { index: usize, len: usize },
    HistoryCountOutOfBounds { count: usize, len: usize },
    PartialHistoryCursorMismatch { count: usize, index: usize },
    PrehistoryStateNotCanonical { field: &'static str },
    UnwrittenHistoryNotCanonical { index: usize },
    EffectivePermeabilityMismatch { field: &'static str },
}

impl MarkovBoundarySnapshotV1 {
    pub fn validate(
        self,
    ) -> Result<ValidatedMarkovBoundarySnapshotV1, MarkovBoundarySnapshotErrorV1> {
        validate_snapshot(&self)?;
        Ok(ValidatedMarkovBoundarySnapshotV1 { snapshot: self })
    }
}

impl ValidatedMarkovBoundarySnapshotV1 {
    pub fn as_snapshot(&self) -> &MarkovBoundarySnapshotV1 {
        &self.snapshot
    }

    pub fn into_snapshot(self) -> MarkovBoundarySnapshotV1 {
        self.snapshot
    }

    pub fn history_count(&self) -> usize {
        self.snapshot.history_count
    }

    pub fn history_len(&self) -> usize {
        self.snapshot.history.len()
    }
}

fn validate_snapshot(
    snapshot: &MarkovBoundarySnapshotV1,
) -> Result<(), MarkovBoundarySnapshotErrorV1> {
    require_nonzero("partition.internal_dim", snapshot.partition.internal_dim)?;
    require_nonzero("partition.sensory_dim", snapshot.partition.sensory_dim)?;
    require_nonzero("partition.active_dim", snapshot.partition.active_dim)?;

    validate_permeability("permeability", &snapshot.permeability)?;
    validate_permeability("permeability_ema", &snapshot.permeability_ema)?;

    require_finite("alpha", snapshot.alpha)?;
    if !(0.01..=1.0).contains(&snapshot.alpha) {
        return Err(MarkovBoundarySnapshotErrorV1::OutOfRange { field: "alpha" });
    }

    if snapshot.history.len() != MARKOV_BOUNDARY_HISTORY_CAP_V1 {
        return Err(MarkovBoundarySnapshotErrorV1::HistoryCapacityMismatch {
            observed: snapshot.history.len(),
            expected: MARKOV_BOUNDARY_HISTORY_CAP_V1,
        });
    }
    if snapshot.history_idx >= snapshot.history.len() {
        return Err(MarkovBoundarySnapshotErrorV1::HistoryIndexOutOfBounds {
            index: snapshot.history_idx,
            len: snapshot.history.len(),
        });
    }
    if snapshot.history_count > snapshot.history.len() {
        return Err(MarkovBoundarySnapshotErrorV1::HistoryCountOutOfBounds {
            count: snapshot.history_count,
            len: snapshot.history.len(),
        });
    }

    for &value in &snapshot.history {
        require_live_permeability("history", value)?;
    }

    if snapshot.history_count == 0 {
        if snapshot.history_idx != 0 {
            return Err(MarkovBoundarySnapshotErrorV1::PrehistoryStateNotCanonical {
                field: "history_idx",
            });
        }
        // Before the first compute_permeability() call, the raw instantaneous field is untouched.
        // The EMA is deliberately NOT required to remain pristine: apply_topology_constraints()
        // can move it without writing the history ring.
        if !is_pristine_permeability(&snapshot.permeability) {
            return Err(MarkovBoundarySnapshotErrorV1::PrehistoryStateNotCanonical {
                field: "permeability",
            });
        }
        if let Some((index, _)) = snapshot
            .history
            .iter()
            .enumerate()
            .find(|(_, value)| value.to_bits() != PRISTINE_PERMEABILITY_V1.to_bits())
        {
            return Err(MarkovBoundarySnapshotErrorV1::UnwrittenHistoryNotCanonical { index });
        }
    } else if snapshot.history_count < snapshot.history.len() {
        if snapshot.history_idx != snapshot.history_count {
            return Err(MarkovBoundarySnapshotErrorV1::PartialHistoryCursorMismatch {
                count: snapshot.history_count,
                index: snapshot.history_idx,
            });
        }
        if let Some((offset, _)) = snapshot.history[snapshot.history_count..]
            .iter()
            .enumerate()
            .find(|(_, value)| value.to_bits() != PRISTINE_PERMEABILITY_V1.to_bits())
        {
            return Err(MarkovBoundarySnapshotErrorV1::UnwrittenHistoryNotCanonical {
                index: snapshot.history_count + offset,
            });
        }
    }

    Ok(())
}

fn is_pristine_permeability(permeability: &BlanketPermeability) -> bool {
    permeability.sensory.to_bits() == PRISTINE_PERMEABILITY_V1.to_bits()
        && permeability.active.to_bits() == PRISTINE_PERMEABILITY_V1.to_bits()
        && permeability.effective.to_bits() == PRISTINE_PERMEABILITY_V1.to_bits()
}

fn validate_permeability(
    field: &'static str,
    permeability: &BlanketPermeability,
) -> Result<(), MarkovBoundarySnapshotErrorV1> {
    require_live_permeability(field, permeability.sensory)?;
    require_live_permeability(field, permeability.active)?;
    require_live_permeability(field, permeability.effective)?;

    let expected = (permeability.sensory * permeability.active).sqrt();
    if (expected - permeability.effective).abs() > 1e-12 {
        return Err(MarkovBoundarySnapshotErrorV1::EffectivePermeabilityMismatch { field });
    }
    Ok(())
}

fn require_nonzero(
    field: &'static str,
    value: usize,
) -> Result<(), MarkovBoundarySnapshotErrorV1> {
    if value == 0 {
        return Err(MarkovBoundarySnapshotErrorV1::ZeroDimension { field });
    }
    Ok(())
}

fn require_live_permeability(
    field: &'static str,
    value: f64,
) -> Result<(), MarkovBoundarySnapshotErrorV1> {
    require_finite(field, value)?;
    if !(PERMEABILITY_FLOOR_V1..=PERMEABILITY_CEILING_V1).contains(&value) {
        return Err(MarkovBoundarySnapshotErrorV1::OutOfRange { field });
    }
    Ok(())
}

fn require_finite(
    field: &'static str,
    value: f64,
) -> Result<(), MarkovBoundarySnapshotErrorV1> {
    if !value.is_finite() {
        return Err(MarkovBoundarySnapshotErrorV1::NonFinite { field });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn partition() -> MarkovPartition {
        MarkovPartition {
            internal_dim: 2,
            sensory_dim: 1,
            active_dim: 1,
        }
    }

    fn permeability(value: f64) -> BlanketPermeability {
        BlanketPermeability {
            sensory: value,
            active: value,
            effective: value,
        }
    }

    fn fresh_snapshot() -> MarkovBoundarySnapshotV1 {
        MarkovBoundarySnapshotV1 {
            partition: partition(),
            permeability: permeability(PRISTINE_PERMEABILITY_V1),
            permeability_ema: permeability(PRISTINE_PERMEABILITY_V1),
            alpha: 0.1,
            history: vec![PRISTINE_PERMEABILITY_V1; MARKOV_BOUNDARY_HISTORY_CAP_V1],
            history_idx: 0,
            history_count: 0,
        }
    }

    #[test]
    fn fresh_boundary_snapshot_is_valid() {
        let validated = fresh_snapshot().validate().expect("valid fresh snapshot");
        assert_eq!(validated.history_count(), 0);
        assert_eq!(validated.history_len(), MARKOV_BOUNDARY_HISTORY_CAP_V1);
    }

    #[test]
    fn history_capacity_is_part_of_the_v1_contract() {
        let mut snapshot = fresh_snapshot();
        snapshot.history.pop();
        assert_eq!(
            snapshot.validate().unwrap_err(),
            MarkovBoundarySnapshotErrorV1::HistoryCapacityMismatch {
                observed: MARKOV_BOUNDARY_HISTORY_CAP_V1 - 1,
                expected: MARKOV_BOUNDARY_HISTORY_CAP_V1,
            }
        );
    }

    #[test]
    fn prehistory_raw_permeability_must_remain_default() {
        let mut snapshot = fresh_snapshot();
        snapshot.permeability = permeability(0.6);
        assert_eq!(
            snapshot.validate().unwrap_err(),
            MarkovBoundarySnapshotErrorV1::PrehistoryStateNotCanonical {
                field: "permeability"
            }
        );
    }

    #[test]
    fn topology_adjusted_ema_is_valid_before_first_history_write() {
        let mut snapshot = fresh_snapshot();
        snapshot.permeability_ema = BlanketPermeability {
            sensory: 0.45,
            active: 0.47,
            effective: (0.45_f64 * 0.47).sqrt(),
        };
        snapshot
            .validate()
            .expect("topology may adjust EMA before first history write");
    }

    #[test]
    fn partial_history_requires_exact_next_write_cursor() {
        let mut snapshot = fresh_snapshot();
        snapshot.history[0] = 0.6;
        snapshot.permeability = permeability(0.6);
        snapshot.history_count = 1;
        snapshot.history_idx = 2;
        assert_eq!(
            snapshot.validate().unwrap_err(),
            MarkovBoundarySnapshotErrorV1::PartialHistoryCursorMismatch { count: 1, index: 2 }
        );
    }

    #[test]
    fn unwritten_partial_ring_tail_must_remain_canonical() {
        let mut snapshot = fresh_snapshot();
        snapshot.history[0] = 0.6;
        snapshot.permeability = permeability(0.6);
        snapshot.history_count = 1;
        snapshot.history_idx = 1;
        snapshot.history[17] = 0.7;
        assert_eq!(
            snapshot.validate().unwrap_err(),
            MarkovBoundarySnapshotErrorV1::UnwrittenHistoryNotCanonical { index: 17 }
        );
    }

    #[test]
    fn history_and_ema_may_diverge_after_topology_constraints() {
        let mut snapshot = fresh_snapshot();
        snapshot.history[0] = 0.6;
        snapshot.permeability = permeability(0.6);
        snapshot.permeability_ema = BlanketPermeability {
            sensory: 0.55,
            active: 0.57,
            effective: (0.55_f64 * 0.57).sqrt(),
        };
        snapshot.history_count = 1;
        snapshot.history_idx = 1;
        snapshot
            .validate()
            .expect("EMA/history divergence is a legitimate topology-adjusted state");
    }

    #[test]
    fn noncanonical_empty_history_cannot_become_restore_authority() {
        let mut snapshot = fresh_snapshot();
        snapshot.history[0] = 0.4;
        assert_eq!(
            snapshot.validate().unwrap_err(),
            MarkovBoundarySnapshotErrorV1::UnwrittenHistoryNotCanonical { index: 0 }
        );
    }

    #[test]
    fn impossible_effective_permeability_fails_closed() {
        let mut snapshot = fresh_snapshot();
        snapshot.permeability.effective = 0.7;
        assert_eq!(
            snapshot.validate().unwrap_err(),
            MarkovBoundarySnapshotErrorV1::EffectivePermeabilityMismatch {
                field: "permeability"
            }
        );
    }

    #[test]
    fn permeability_outside_live_clamp_fails_closed() {
        let mut snapshot = fresh_snapshot();
        snapshot.permeability.sensory = 0.01;
        assert_eq!(
            snapshot.validate().unwrap_err(),
            MarkovBoundarySnapshotErrorV1::OutOfRange {
                field: "permeability"
            }
        );
    }

    #[test]
    fn invalid_alpha_fails_closed() {
        let mut snapshot = fresh_snapshot();
        snapshot.alpha = 0.0;
        assert_eq!(
            snapshot.validate().unwrap_err(),
            MarkovBoundarySnapshotErrorV1::OutOfRange { field: "alpha" }
        );
    }

    #[test]
    fn full_ring_allows_wrapped_cursor_and_topology_adjusted_ema() {
        let mut snapshot = fresh_snapshot();
        snapshot.history_count = MARKOV_BOUNDARY_HISTORY_CAP_V1;
        snapshot.history_idx = 7;
        snapshot.history[6] = 0.6;
        snapshot.permeability = permeability(0.6);
        snapshot.permeability_ema = BlanketPermeability {
            sensory: 0.52,
            active: 0.54,
            effective: (0.52_f64 * 0.54).sqrt(),
        };
        snapshot
            .validate()
            .expect("valid wrapped full ring with topology-adjusted EMA");
    }
}
