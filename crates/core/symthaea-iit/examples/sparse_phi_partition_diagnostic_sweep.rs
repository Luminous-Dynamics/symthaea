// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exhaustive per-cut score diagnostics for the 13-point IIT4 landscape fixture.
//!
//! This deliberately records each state-pair × directed-cut score instead of
//! only the selected MIP. It is intended for primitive ranking diagnosis
//! against the pinned PyPhi oracle.

use sha2::{Digest, Sha256};
use std::error::Error;
use symthaea_iit::{
    enumerate_iit4_system_partitions, AnalysisQuantity, AnalysisRequest, FormalismProfile,
    Iit4SystemPhiSearchConfig, SparseBinaryCausalModel, SparseMechanismCpt,
};

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

fn bits(state: &[bool]) -> String {
    state.iter().map(|value| if *value { '1' } else { '0' }).collect()
}

fn digest(bytes: &[u8]) -> String {
    Sha256::digest(bytes).iter().map(|value| format!("{value:02x}")).collect()
}

fn cut_digest(cut: &[bool]) -> String {
    let mut bytes = b"SYMTHAEA-IIT-LANDSCAPE-CUT\0v0.1\0".to_vec();
    bytes.extend_from_slice(&(cut.len() as u64).to_le_bytes());
    bytes.extend(cut.iter().map(|value| u8::from(*value)));
    digest(&bytes)
}

fn parts(partition: &symthaea_iit::DirectedSystemPartition) -> String {
    partition.parts().iter().map(|part| {
        part.iter().map(usize::to_string).collect::<Vec<_>>().join(".")
    }).collect::<Vec<_>>().join(";")
}

fn directions(partition: &symthaea_iit::DirectedSystemPartition) -> String {
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
    let partitions = enumerate_iit4_system_partitions(&CANDIDATE, 3, false, 100_000)?;

    println!(
        "landscape_algorithm_id,formalism,parameter_name,theta,grid_index,canonical_dense_model_sha256,canonical_dense_evaluation_sha256,canonical_dense_analysis_request_sha256,sparse_analysis_request_sha256,analysis_status,state_pair_index,cause_state,effect_state,partition_index,cut_sha256,cut_bits,parts,directions,severed_connections,cause_signed_phi,effect_signed_phi,signed_phi,phi,normalized_phi,signed_normalized_phi,partition_score_request_sha256,partition_score_receipt_sha256,qualification"
    );

    for (grid_index, theta) in grid.iter().copied().enumerate() {
        let causal_model = model(theta)?;
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
        let canonical_dense_analysis_request_sha256 = dense_request.request_sha256(&dense_model);
        let sparse_search = causal_model.analyze_iit4_2023_system_phi_s(
            &CANDIDATE, &STATE, config, FACTOR_BUDGET,
        )?;

        let mut row_count = 0usize;
        for (state_pair_index, state_pair) in sparse_search.state_pair_results.iter().enumerate() {
            for (partition_index, partition) in partitions.iter().enumerate() {
                let report = causal_model.score_iit4_system_partition(
                    partition,
                    &STATE,
                    &state_pair.system_state,
                    FACTOR_BUDGET,
                )?;
                let score = &report.score;
                let cut_bits = partition.cut_matrix().iter()
                    .map(|value| if *value { '1' } else { '0' }).collect::<String>();
                println!(
                    "symthaea-iit/parameter-landscape-per-cut/0.1.0,IIT_4_0_2023,target0_parent_state_3_probability,{theta:.8},{grid_index},{canonical_dense_model_sha256},{canonical_dense_evaluation_sha256},{canonical_dense_analysis_request_sha256},{},{:?},{state_pair_index},{},{},{partition_index},{},{},{},{},{},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{},{},EXACT_PARTITION_SAMPLE",
                    sparse_search.request_sha256,
                    sparse_search.status,
                    bits(&state_pair.system_state.cause_specified_state),
                    bits(&state_pair.system_state.effect_specified_state),
                    cut_digest(partition.cut_matrix()),
                    cut_bits,
                    parts(partition),
                    directions(partition),
                    score.severed_connections,
                    score.cause.signed_phi,
                    score.effect.signed_phi,
                    score.signed_phi,
                    score.phi,
                    score.normalized_phi,
                    score.signed_normalized_phi,
                    report.request_sha256,
                    report.receipt_sha256()?,
                );
                row_count += 1;
            }
        }
        if row_count == 0 {
            return Err("per-cut diagnostic emitted no state-pair/partition rows".into());
        }
    }
    Ok(())
}
