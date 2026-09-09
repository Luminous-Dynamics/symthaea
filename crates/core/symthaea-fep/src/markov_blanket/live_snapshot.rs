// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Live `MarkovBoundaryOperator` bridge for the validated v1 snapshot contract.
//!
//! This child module can access the operator's private EMA/history fields without widening the
//! public mutation surface. Raw persisted snapshots never become live boundary state directly;
//! restoration requires `ValidatedMarkovBoundarySnapshotV1`.

use super::MarkovBoundaryOperator;
use crate::{MarkovBoundarySnapshotV1, ValidatedMarkovBoundarySnapshotV1};

impl MarkovBoundaryOperator {
    /// Capture every operator-owned causal field needed for exact continuation.
    pub fn snapshot_v1(&self) -> MarkovBoundarySnapshotV1 {
        MarkovBoundarySnapshotV1 {
            partition: self.partition.clone(),
            permeability: self.permeability.clone(),
            permeability_ema: self.permeability_ema.clone(),
            alpha: self.alpha,
            history: self.history.clone(),
            history_idx: self.history_idx,
            history_count: self.history_count,
        }
    }

    /// Restore a live boundary only from a snapshot that passed the v1 validator.
    pub fn from_validated_snapshot_v1(snapshot: &ValidatedMarkovBoundarySnapshotV1) -> Self {
        let snapshot = snapshot.as_snapshot();
        Self {
            partition: snapshot.partition.clone(),
            permeability: snapshot.permeability.clone(),
            permeability_ema: snapshot.permeability_ema.clone(),
            alpha: snapshot.alpha,
            history: snapshot.history.clone(),
            history_idx: snapshot.history_idx,
            history_count: snapshot.history_count,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        HiddenState, MarkovPartition, Observation, PermeabilityInputs, TopologyBoundaryInputs,
    };

    fn inputs(i: usize) -> PermeabilityInputs {
        let phase = (i % 19) as f64 / 18.0;
        PermeabilityInputs {
            acetylcholine: 0.15 + 0.70 * phase,
            noradrenaline: 0.65 - 0.45 * phase,
            serotonin: 0.25 + 0.65 * (1.0 - phase),
            oxytocin: 0.2 + 0.55 * phase,
            threat_level: if i % 11 == 0 { 0.8 } else { 0.1 * phase },
            peer_trust: 0.35 + 0.5 * phase,
            flow_state: if i % 5 == 0 { 0.75 } else { 0.2 + 0.3 * phase },
        }
    }

    fn topology(i: usize) -> TopologyBoundaryInputs {
        TopologyBoundaryInputs {
            boundary_thickness: ((i * 7) % 13) as f64 / 12.0,
            fiedler_value: ((i * 5) % 17) as f64 / 8.0,
            boundary_components: 1 + (i % 5),
        }
    }

    fn drive(op: &mut MarkovBoundaryOperator, start: usize, count: usize) {
        for i in start..start + count {
            let _ = op.compute_permeability(&inputs(i));
            if i % 3 == 0 {
                op.apply_topology_constraints(&topology(i));
            }
        }
    }

    fn snapshot_json(op: &MarkovBoundaryOperator) -> String {
        serde_json::to_string(&op.snapshot_v1()).expect("serialize boundary snapshot")
    }

    fn assert_observables_match(
        a: &MarkovBoundaryOperator,
        b: &MarkovBoundaryOperator,
        i: usize,
    ) {
        let prior = HiddenState::new(3);
        let observation = Observation::new(
            vec![
                ((i * 3) % 17) as f64 / 16.0,
                ((i * 5) % 19) as f64 / 18.0,
                ((i * 7) % 23) as f64 / 22.0,
            ],
            0.87,
            "markov-split-run",
        );
        let ga = a.gate_observation(&observation, &prior);
        let gb = b.gate_observation(&observation, &prior);
        assert_eq!(
            ga.values.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            gb.values.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(ga.precision.to_bits(), gb.precision.to_bits());
        assert_eq!(a.trend().to_bits(), b.trend().to_bits());
        assert_eq!(a.coalescence_ready(0.6), b.coalescence_ready(0.6));
        assert_eq!(
            a.modulate_sensory_precision(0.73).to_bits(),
            b.modulate_sensory_precision(0.73).to_bits()
        );
        assert_eq!(
            a.modulate_learning_rate(0.11).to_bits(),
            b.modulate_learning_rate(0.11).to_bits()
        );
    }

    #[test]
    fn serialized_validated_restore_preserves_wrapped_history_future() {
        let partition = MarkovPartition {
            internal_dim: 3,
            sensory_dim: 2,
            active_dim: 2,
        };
        let mut uninterrupted = MarkovBoundaryOperator::new(partition).with_alpha(0.23);

        // Cross the 64-sample ring capacity before checkpointing so restore is tested with a
        // wrapped cursor rather than only the simpler partially-filled history state.
        drive(&mut uninterrupted, 0, 83);

        let encoded = snapshot_json(&uninterrupted);
        let raw: MarkovBoundarySnapshotV1 =
            serde_json::from_str(&encoded).expect("deserialize boundary snapshot");
        let validated = raw.validate().expect("validate boundary snapshot");
        let mut restored = MarkovBoundaryOperator::from_validated_snapshot_v1(&validated);

        assert_eq!(snapshot_json(&uninterrupted), snapshot_json(&restored));
        assert_observables_match(&uninterrupted, &restored, 83);

        for i in 83..183 {
            let _ = uninterrupted.compute_permeability(&inputs(i));
            let _ = restored.compute_permeability(&inputs(i));
            if i % 3 == 0 {
                let topo = topology(i);
                uninterrupted.apply_topology_constraints(&topo);
                restored.apply_topology_constraints(&topo);
            }
            assert_eq!(
                snapshot_json(&uninterrupted),
                snapshot_json(&restored),
                "boundary state diverged after restored continuation step {i}"
            );
            assert_observables_match(&uninterrupted, &restored, i);
        }
    }

    #[test]
    fn topology_adjusted_ema_history_divergence_survives_restore() {
        let partition = MarkovPartition {
            internal_dim: 2,
            sensory_dim: 1,
            active_dim: 1,
        };
        let mut uninterrupted = MarkovBoundaryOperator::new(partition).with_alpha(0.41);
        let _ = uninterrupted.compute_permeability(&inputs(4));
        uninterrupted.apply_topology_constraints(&TopologyBoundaryInputs {
            boundary_thickness: 0.9,
            fiedler_value: 0.05,
            boundary_components: 5,
        });

        let encoded = snapshot_json(&uninterrupted);
        let raw: MarkovBoundarySnapshotV1 = serde_json::from_str(&encoded).unwrap();
        let validated = raw.validate().expect("EMA/history divergence is valid live state");
        let restored = MarkovBoundaryOperator::from_validated_snapshot_v1(&validated);

        assert_eq!(snapshot_json(&uninterrupted), snapshot_json(&restored));
        assert_observables_match(&uninterrupted, &restored, 5);
    }
}
