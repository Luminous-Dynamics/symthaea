// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Local CPT uncertainty audit for IIT4 (2023) selected MIPs and states.
//!
//! Around each of three parameter-landscape anchors, perturb every local CPT
//! probability independently at six signed magnitudes. Each row is a fresh
//! exact SIA and carries model/request/result identities. These are sampled
//! robustness probes, not certified continuous uncertainty bounds.

use sha2::{Digest, Sha256};
use std::error::Error;
use std::time::Instant;
use symthaea_iit::{
    AnalysisQuantity, AnalysisRequest, FormalismProfile, Iit4SystemPhiSearchConfig,
    SparseBinaryCausalModel, SparseMechanismCpt,
};

const FORMALISM: &str = "IIT_4_0_2023";
const ALGORITHM_ID: &str = "symthaea-iit/local-cpt-selection-robustness/0.1.0";
const CANDIDATE: [usize; 3] = [0, 1, 2];
const STATE: [bool; 3] = [true, false, true];
const FACTOR_BUDGET: usize = 1 << 16;
const ANCHORS: [f64; 3] = [0.375, 0.425, 0.525];
const DELTAS: [f64; 6] = [-0.01, -0.001, -0.0001, 0.0001, 0.001, 0.01];

const BASE_TABLES: [[f64; 4]; 3] = [
    [0.12, 0.38, 0.57, 0.0],
    [0.18, 0.58, 0.31, 0.79],
    [0.21, 0.43, 0.62, 0.83],
];

const PARENTS: [&[usize]; 3] = [&[0, 2], &[0, 1], &[1, 2]];

struct Evaluation {
    model_sha256: String,
    evaluation_sha256: String,
    request_sha256: String,
    receipt_sha256: String,
    complete: bool,
    status: String,
    phi_s: Option<f64>,
    signed_phi: Option<f64>,
    selected_index: Option<usize>,
    cause_state: String,
    effect_state: String,
    winning_cut_sha256: String,
    tied_mip_cut_sha256s: String,
    mip_tie_count: usize,
    partition_margin: Option<f64>,
    cause_selection_margin: Option<f64>,
    effect_selection_margin: Option<f64>,
    partition_score_evaluations: usize,
}

fn make_model(
    theta_anchor: f64,
    mechanism_index: usize,
    row_index: usize,
    delta: f64,
) -> Result<SparseBinaryCausalModel, Box<dyn Error>> {
    let mut tables = BASE_TABLES;
    tables[0][3] = theta_anchor;
    tables[mechanism_index][row_index] += delta;
    for (mechanism, table) in tables.iter().enumerate() {
        for (row, probability) in table.iter().enumerate() {
            if !probability.is_finite() || !(0.0..=1.0).contains(probability) {
                return Err(format!(
                    "invalid perturbed CPT probability m{mechanism}/row{row}: {probability}"
                ).into());
            }
        }
    }
    let mechanisms = (0..3)
        .map(|target| {
            SparseMechanismCpt::new(
                target,
                PARENTS[target].to_vec(),
                tables[target].to_vec(),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(SparseBinaryCausalModel::new(3, mechanisms)?)
}

fn bit_string(state: &[bool]) -> String {
    state.iter().map(|bit| if *bit { '1' } else { '0' }).collect()
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

fn optional_float(value: Option<f64>) -> String {
    value.map(|number| format!("{number:.17}")).unwrap_or_else(|| "NA".to_owned())
}

fn optional_index(value: Option<usize>) -> String {
    value.map(|number| number.to_string()).unwrap_or_else(|| "NA".to_owned())
}

fn evaluate(
    causal_model: &SparseBinaryCausalModel,
    config: Iit4SystemPhiSearchConfig,
) -> Result<Evaluation, Box<dyn Error>> {
    let analysis = causal_model.analyze_iit4_2023_system_phi_s(
        &CANDIDATE, &STATE, config, FACTOR_BUDGET,
    )?;
    let dense_model = causal_model.to_dense()?;
    let dense_request = AnalysisRequest::new(
        &dense_model,
        STATE.to_vec(),
        CANDIDATE.to_vec(),
        FormalismProfile::Iit4_0_2023,
        AnalysisQuantity::Iit4SystemPhiS,
    )?;
    let selected = analysis.selected_state_pair_index
        .and_then(|index| analysis.state_pair_results.get(index));
    let intrinsic = causal_model.system_intrinsic_specification_iit4(
        &CANDIDATE, &STATE, config.decimal_precision, FACTOR_BUDGET,
    )?;
    let (cause_state, effect_state, winning_cut_sha256, tied_mip_cut_sha256s, mip_tie_count,
        partition_margin, signed_phi) = if let Some(result) = selected {
        (
            bit_string(&result.system_state.cause_specified_state),
            bit_string(&result.system_state.effect_specified_state),
            cut_digest(result.winning_partition.cut_matrix()),
            result.tied_partitions.iter()
                .map(|partition| cut_digest(partition.cut_matrix()))
                .collect::<Vec<_>>()
                .join(";"),
            result.tied_partitions.len(),
            result.partition_margin,
            Some(result.score.signed_phi),
        )
    } else {
        ("NA".to_owned(), "NA".to_owned(), "NA".to_owned(), "NA".to_owned(),
         0, None, None)
    };
    Ok(Evaluation {
        model_sha256: dense_model.model_sha256(),
        evaluation_sha256: dense_model.evaluation_sha256(&STATE)?,
        request_sha256: dense_request.request_sha256(&dense_model),
        receipt_sha256: analysis.receipt_sha256()?,
        complete: analysis.complete,
        status: format!("{:?}", analysis.status),
        phi_s: analysis.phi_s,
        signed_phi,
        selected_index: analysis.selected_state_pair_index,
        cause_state,
        effect_state,
        winning_cut_sha256,
        tied_mip_cut_sha256s,
        mip_tie_count,
        partition_margin,
        cause_selection_margin: intrinsic.cause.selection_margin,
        effect_selection_margin: intrinsic.effect.selection_margin,
        partition_score_evaluations: analysis.partition_score_evaluations,
    })
}

fn main() -> Result<(), Box<dyn Error>> {
    let config = Iit4SystemPhiSearchConfig {
        decimal_precision: 12,
        max_partition_combinations: 100_000,
        max_state_pair_evaluations: 64,
        max_partition_score_evaluations: 100_000,
        include_total_cut: false,
    };
    println!(
        "algorithm_id,formalism,anchor_theta,parameter_id,mechanism_index,local_row,delta,perturbed_probability,model_sha256,evaluation_sha256,request_sha256,complete,status,phi_s,signed_phi,selected_state_pair_index,cause_state,effect_state,winning_mip_cut_sha256,tied_mip_cut_sha256s,mip_tie_count,partition_margin,cause_selection_margin,effect_selection_margin,partition_score_evaluations,baseline_phi_s,delta_phi_s,baseline_cause_state,baseline_effect_state,baseline_mip_cut_sha256,mip_changed,cause_state_changed,effect_state_changed,decision_changed,receipt_sha256,elapsed_ns,qualification"
    );

    for anchor_theta in ANCHORS {
        let baseline_model = make_model(anchor_theta, 0, 3, 0.0)?;
        let baseline = evaluate(&baseline_model, config)?;
        if !baseline.complete {
            return Err(format!("baseline theta={anchor_theta} was incomplete").into());
        }

        for mechanism_index in 0..3 {
            for row_index in 0..4 {
                let base_probability = if mechanism_index == 0 && row_index == 3 {
                    anchor_theta
                } else {
                    BASE_TABLES[mechanism_index][row_index]
                };
                for delta in DELTAS {
                    let started = Instant::now();
                    let causal_model = make_model(
                        anchor_theta, mechanism_index, row_index, delta,
                    )?;
                    let actual_probability = base_probability + delta;
                    let result = evaluate(&causal_model, config)?;
                    if !result.complete {
                        return Err(format!(
                            "perturbed search incomplete theta={anchor_theta} m={mechanism_index} row={row_index} delta={delta}"
                        ).into());
                    }
                    let mip_changed = result.winning_cut_sha256 != baseline.winning_cut_sha256;
                    let cause_changed = result.cause_state != baseline.cause_state;
                    let effect_changed = result.effect_state != baseline.effect_state;
                    let decision_changed = mip_changed || cause_changed || effect_changed;
                    let elapsed_ns = started.elapsed().as_nanos();
                    println!(
                        "{},{},{:.8},m{}_row{},{},{},{:.8},{:.8},{},{},{},{},{:?},{},{},{},{},{},{},{},{},{},{},{},{},{:.17},{:.17},{},{},{},{},{},{},{},{},{},EXACT_LOCAL_PERTURBATION",
                        ALGORITHM_ID,
                        FORMALISM,
                        anchor_theta,
                        mechanism_index,
                        row_index,
                        mechanism_index,
                        row_index,
                        delta,
                        actual_probability,
                        result.model_sha256,
                        result.evaluation_sha256,
                        result.request_sha256,
                        result.complete,
                        result.status,
                        result.phi_s,
                        optional_float(result.signed_phi),
                        optional_index(result.selected_index),
                        result.cause_state,
                        result.effect_state,
                        result.winning_cut_sha256,
                        result.tied_mip_cut_sha256s,
                        result.mip_tie_count,
                        optional_float(result.partition_margin),
                        optional_float(result.cause_selection_margin),
                        optional_float(result.effect_selection_margin),
                        result.partition_score_evaluations,
                        baseline.phi_s.unwrap_or(0.0),
                        result.phi_s.unwrap_or(0.0) - baseline.phi_s.unwrap_or(0.0),
                        baseline.cause_state,
                        baseline.effect_state,
                        baseline.winning_cut_sha256,
                        mip_changed,
                        cause_changed,
                        effect_changed,
                        decision_changed,
                        result.receipt_sha256,
                        elapsed_ns,
                    );
                }
            }
        }
    }
    Ok(())
}
