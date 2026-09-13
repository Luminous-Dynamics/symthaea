// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Adversarial specificity checks for the GWT-1 causal-lesion kernel.

use super::gwt1_causal_lesion::{
    Gwt1CausalSignatureV1, run_gwt1_causal_lesion_v1,
};
use super::gwt1_specialist_qualification::GWT1_SPECIALIST_IDS_V1;
use crate::cognitive_loop::subsystem_trait::output_flags;

#[test]
fn preregistered_signature_channel_is_exclusive_to_target() {
    let observations = run_gwt1_causal_lesion_v1();

    for row in &observations.rows {
        for id in GWT1_SPECIALIST_IDS_V1 {
            let output = row
                .baseline
                .executed_outputs
                .get(id)
                .expect("baseline executes every specialist");
            let is_target = id == row.target_specialist;

            match row.preregistered_signature {
                Gwt1CausalSignatureV1::DriveValence => {
                    assert_eq!(
                        output.valence_delta != 0.0f32.to_bits(),
                        is_target,
                        "drive-valence signature leaked to {id}"
                    );
                }
                Gwt1CausalSignatureV1::MemoryConsolidationFlag => {
                    assert_eq!(
                        output.flags & output_flags::REQUEST_CONSOLIDATION != 0,
                        is_target,
                        "memory consolidation signature leaked to {id}"
                    );
                }
                Gwt1CausalSignatureV1::LearningRate => {
                    assert_eq!(
                        output.lr_modulation != 1.0f64.to_bits(),
                        is_target,
                        "learning-rate signature leaked to {id}"
                    );
                }
                Gwt1CausalSignatureV1::PerceptionConfidence => {
                    assert_eq!(
                        output.confidence_delta != 0.0f64.to_bits(),
                        is_target,
                        "perception-confidence signature leaked to {id}"
                    );
                }
            }
        }
    }
}

#[test]
fn rescue_restores_baseline_contributor_set_without_restoring_execution_set() {
    let observations = run_gwt1_causal_lesion_v1();

    for row in &observations.rows {
        assert_eq!(
            row.rescue.recorded_contributors,
            row.baseline.recorded_contributors,
            "collector rescue must restore the exact non-neutral contributor set for {}",
            row.target_specialist
        );
        assert_ne!(
            row.rescue.invoked_specialists,
            row.baseline.invoked_specialists,
            "output rescue must not be misreported as target execution for {}",
            row.target_specialist
        );
        assert!(!row.rescue.invoked_specialists.contains(&row.target_specialist));
    }
}
