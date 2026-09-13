// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Narrow verification facade for persisted GWT-1 causal observations.
//!
//! This module deliberately keeps `cognitive_loop::subsystem_trait` crate-private.
//! External evidence code supplies only the public exact-bit specialist output
//! records and receives the production collector's recomputed contributor set
//! and integrated output bits.

use std::collections::BTreeSet;

use crate::cognitive_loop::subsystem_trait::{OutputCollector, SubsystemOutput};

use super::gwt1_causal_lesion::Gwt1IntegratedOutputBitsV1;
use super::gwt1_specialist_qualification::{
    GWT1_SPECIALIST_IDS_V1, Gwt1SpecialistOutputMapV1, Gwt1SubsystemOutputBitsV1,
};

fn bits_to_output(bits: &Gwt1SubsystemOutputBitsV1) -> SubsystemOutput {
    SubsystemOutput {
        confidence_delta: f64::from_bits(bits.confidence_delta),
        lr_modulation: f64::from_bits(bits.lr_modulation),
        exploration_delta: f64::from_bits(bits.exploration_delta),
        arousal_delta: f32::from_bits(bits.arousal_delta),
        valence_delta: f32::from_bits(bits.valence_delta),
        flags: bits.flags,
        _reserved: bits.reserved,
    }
}

/// Recompute the canonical production collector result from persisted exact-bit
/// specialist outputs. `injected` is used only for the causal rescue arm and
/// occupies its canonical specialist position in the reduction order.
pub fn recompute_gwt1_collector_v1(
    outputs: &Gwt1SpecialistOutputMapV1,
    injected: Option<(&str, &Gwt1SubsystemOutputBitsV1)>,
) -> (BTreeSet<String>, Gwt1IntegratedOutputBitsV1) {
    let mut collector = OutputCollector::new();
    let mut contributors = BTreeSet::new();

    for id in GWT1_SPECIALIST_IDS_V1 {
        let bits = injected
            .filter(|(injected_id, _)| *injected_id == id)
            .map(|(_, value)| value)
            .or_else(|| outputs.get(id));
        if let Some(bits) = bits {
            let output = bits_to_output(bits);
            if !output.is_neutral() {
                contributors.insert(id.to_string());
            }
            collector.record(id, output);
        }
    }

    (contributors, collector.integrate().into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmarks::gwt1_causal_lesion::run_gwt1_causal_lesion_v1;

    #[test]
    fn facade_recomputes_all_persisted_active_arm_integrations() {
        let raw = run_gwt1_causal_lesion_v1();
        for row in &raw.rows {
            for arm in [&row.baseline, &row.lesion, &row.sham] {
                let recomputed = recompute_gwt1_collector_v1(&arm.executed_outputs, None);
                assert_eq!(recomputed.0, arm.recorded_contributors);
                assert_eq!(recomputed.1, arm.integrated);
            }
            let rescue = recompute_gwt1_collector_v1(
                &row.rescue.executed_outputs,
                Some((&row.target_specialist, &row.rescue_injected_target_output)),
            );
            assert_eq!(rescue.0, row.rescue.recorded_contributors);
            assert_eq!(rescue.1, row.rescue.integrated);
        }
    }
}
