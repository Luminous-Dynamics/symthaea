// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic one-parameter IIT 4.0 (2023) landscape qualification sweep.
//!
//! Every row is a fresh exact system search on a parameterized sparse causal
//! model. Rows are point samples only: no interval between grid points is
//! certified switch-free, and no interpolation is used.

use sha2::{Digest, Sha256};
use std::error::Error;
use std::time::Instant;
use symthaea_iit::{
    AnalysisQuantity, AnalysisRequest, FormalismProfile, Iit4SystemPhiSearchConfig,
    SparseBinaryCausalModel, SparseMechanismCpt,
};

const FORMALISM: &str = "IIT_4_0_2023";
const LANDSCAPE_ALGORITHM_ID: &str = "symthaea-iit/parameter-landscape-exact-samples/0.1.0";
const CANDIDATE: [usize; 3] = [0, 1, 2];
const STATE: [bool; 3] = [true, false, true];
const FACTOR_BUDGET: usize = 1 << 16;

fn model(theta: f64) -> Result<SparseBinaryCausalModel, Box<dyn Error>> {
    let mechanisms = vec![
        SparseMechanismCpt::new(0, vec![0, 2], vec![0.12, 0.38, 0.57, theta])?,
        SparseMechanismCpt::new(1, vec![0, 1], vec![0.18, 0.58, 0.31, 0.79])?,
        SparseMechanismCpt::new(2, vec![1, 2], vec![0.21, 0.43, 0.62, 0.83])?,
    ];
    Ok(SparseBinaryCausalModel::new(3, mechanisms)?)
}

fn bit_string(state: &[bool]) -> String {
    state.iter().map(|bit| if *bit { '1' } else { '0' }).collect()
}

fn state_family(states: &[Vec<bool>]) -> String {
    states.iter().map(|state| bit_string(state)).collect::<Vec<_>>().join(";")
}

fn optional_float(value: Option<f64>) -> String {
    value.map(|v| format!("{v:.17}")).unwrap_or_else(|| "NA".to_owned())
}

fn optional_usize(value: Option<usize>) -> String {
    value.map(|v| v.to_string()).unwrap_or_else(|| "NA".to_owned())
}

fn hex_digest(bytes: &[u8]) -> String {
    Sha256::digest(bytes).iter().map(|byte| format!("{byte:02x}")).collect()
}

fn cut_digest(cut: &[bool]) -> String {
    let mut bytes = b"SYMTHAEA-IIT-LANDSCAPE-CUT\0v0.1\0".to_vec();
    bytes.extend_from_slice(&(cut.len() as u64).to_le_bytes());
    bytes.extend(cut.iter().map(|value| u8::from(*value)));
    hex_digest(&bytes)
}

fn partition_parts(partition: &symthaea_iit::DirectedSystemPartition) -> String {
    partition.parts().iter().map(|part| {
        part.iter().map(usize::to_string).collect::<Vec<_>>().join(".")
    }).collect::<Vec<_>>().join(";")
}

fn partition_directions(partition: &symthaea_iit::DirectedSystemPartition) -> String {
    partition.directions().iter().map(|direction| format!("{direction:?}"))
        .collect::<Vec<_>>().join(";")
}

fn main() -> Result<(), Box<dyn Error>> {
    let grid = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80];
    let config = Iit4SystemPhiSearchConfig {
        decimal_precision: 12,
        max_partition_combinations: 100_000,
        max_state_pair_evaluations: 64,
        max_partition_score_evaluations: 100_000,
        include_total_cut: false,
    };

    println!(
        "landscape_algorithm_id,formalism,parameter_name,theta,grid_index,sparse_model_sha256,sparse_evaluation_sha256,sparse_request_sha256,canonical_dense_model_sha256,canonical_dense_evaluation_sha256,canonical_dense_analysis_request_sha256,complete,status,phi_s,selected_state_pair_index,tied_state_pair_indices,cause_state_family,effect_state_family,cause_selection_margin,effect_selection_margin,winning_mip_cut_sha256,winning_mip_parts,winning_mip_directions,partition_margin,partition_score_evaluations,receipt_sha256,elapsed_ns,qualification"
    );

    for (grid_index, theta) in grid.iter().copied().enumerate() {
        let started = Instant::now();
        let causal_model = model(theta)?;
        let intrinsic = causal_model.system_intrinsic_specification_iit4(
            &CANDIDATE,
            &STATE,
            config.decimal_precision,
            FACTOR_BUDGET,
        )?;
        let analysis = causal_model.analyze_iit4_2023_system_phi_s(
            &CANDIDATE,
            &STATE,
            config,
            FACTOR_BUDGET,
        )?;
        let sparse_model_sha256 = causal_model.model_sha256();
        let sparse_evaluation_sha256 = causal_model.evaluation_sha256(&STATE)?;
        // Cross-language identity uses the canonical dense TPM wire contract
        // shared with the pinned PyPhi oracle. The sparse representation keeps
        // its own distinct identity as well.
        let dense_model = causal_model.to_dense()?;
        let canonical_dense_model_sha256 = dense_model.model_sha256();
        let canonical_dense_evaluation_sha256 = dense_model.evaluation_sha256(&STATE)?;
        let dense_request = AnalysisRequest::new(
            &dense_model,
            STATE.to_vec(),
            CANDIDATE.to_vec(),
            FormalismProfile::Iit4_0_2023,
            AnalysisQuantity::Iit4SystemPhiS,
        )?;
        let canonical_dense_analysis_request_sha256 =
            dense_request.request_sha256(&dense_model);
        let receipt_sha256 = analysis.receipt_sha256()?;
        let selected_index = analysis.selected_state_pair_index;
        let selected = selected_index.and_then(|index| analysis.state_pair_results.get(index));
        let winning_cut_sha256 = selected
            .map(|result| cut_digest(result.winning_partition.cut_matrix()))
            .unwrap_or_else(|| "NA".to_owned());
        let winning_parts = selected
            .map(|result| partition_parts(&result.winning_partition))
            .unwrap_or_else(|| "NA".to_owned());
        let winning_directions = selected
            .map(|result| partition_directions(&result.winning_partition))
            .unwrap_or_else(|| "NA".to_owned());
        let partition_margin = selected
            .and_then(|result| result.partition_margin);
        let tied_indices = analysis.tied_state_pair_indices.iter()
            .map(usize::to_string).collect::<Vec<_>>().join(";");
        let elapsed_ns = started.elapsed().as_nanos();

        println!(
            "{},{},{},{:.8},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},EXACT_SAMPLE_ONLY",
            LANDSCAPE_ALGORITHM_ID,
            FORMALISM,
            "target0_parent_state_3_probability",
            theta,
            grid_index,
            sparse_model_sha256,
            sparse_evaluation_sha256,
            analysis.request_sha256,
            canonical_dense_model_sha256,
            canonical_dense_evaluation_sha256,
            canonical_dense_analysis_request_sha256,
            analysis.complete,
            format!("{:?}", analysis.status),
            optional_float(analysis.phi_s),
            optional_usize(selected_index),
            if tied_indices.is_empty() { "none" } else { &tied_indices },
            state_family(&intrinsic.cause.specified_states),
            state_family(&intrinsic.effect.specified_states),
            optional_float(intrinsic.cause.selection_margin),
            optional_float(intrinsic.effect.selection_margin),
            winning_cut_sha256,
            winning_parts,
            winning_directions,
            optional_float(partition_margin),
            analysis.partition_score_evaluations,
            receipt_sha256,
            elapsed_ns,
        );
    }

    Ok(())
}
