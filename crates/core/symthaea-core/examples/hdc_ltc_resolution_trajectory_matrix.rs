//! Research-only machine-readable HDC-LTC liquid-resolution trajectory matrix.
//!
//! This runner closes the gap between static projection/operator characterization
//! and the actual liquid state transition. It converts the complete serialized
//! neuron state/parameters, then compares:
//!   1. high-resolution evolve -> project (reference)
//!   2. project complete neuron -> evolve (candidate)
//!
//! No production adaptive-resize policy is inferred from this output.

use serde::Serialize;
use serde_json::Value;
use symthaea_core::hdc::{
    continuous_resolution_projection::{project, ContinuousProjectionFamily},
    ContinuousHV, HdcLtcUnifiedNeuron, UnifiedActivation, UnifiedConfig,
};

const DIMS: [usize; 7] = [1_024, 2_048, 4_096, 8_192, 16_384, 32_768, 65_536];
const SEED: u64 = 0x4844_432d_5452414a;
const INPUT_SEED: u64 = 0x5452414a_2d494e50;
const DT: [f32; 8] = [0.01, 0.017, 0.031, 0.007, 0.023, 0.041, 0.013, 0.029];

#[derive(Debug, Clone, Copy)]
enum Family {
    LegacyDilate,
    HadamardTruncateV1,
}

impl Family {
    fn name(self) -> &'static str {
        match self {
            Self::LegacyDilate => "legacy_dilate",
            Self::HadamardTruncateV1 => "hadamard_truncate_v1",
        }
    }
}

#[derive(Debug, Serialize)]
struct Row {
    schema_version: &'static str,
    source_dim: usize,
    target_dim: usize,
    family: &'static str,
    seed: u64,
    steps: usize,
    state_error_terminal: f32,
    state_error_mean: f32,
    tau_error_terminal: f32,
    tau_error_mean: f32,
    hysteresis_error: f32,
    clock_preserved: bool,
    update_count_preserved: bool,
}

fn config(dim: usize) -> UnifiedConfig {
    UnifiedConfig {
        dimension: dim,
        activation: UnifiedActivation::Tanh,
        tau_base: 0.1,
        backbone_tau: 0.5,
        gating_steepness: 1.0,
        interp_bias: 0.0,
        fourier_frequencies: Vec::new(),
        fourier_amplitude: 0.1,
        learning_rate: 0.01,
        momentum: 0.9,
        weight_decay: 0.0001,
    }
}

fn input(dim: usize, step: usize) -> ContinuousHV {
    ContinuousHV::random(dim, INPUT_SEED.wrapping_add(step as u64 * 7919))
}

fn neuron(dim: usize, seed: u64) -> HdcLtcUnifiedNeuron {
    let mut n = HdcLtcUnifiedNeuron::new(config(dim), seed);
    n.set_state(ContinuousHV::random(dim, seed ^ 0x5354415445));
    n
}

fn convert_hv(hv: &ContinuousHV, target_dim: usize, family: Family) -> ContinuousHV {
    match family {
        Family::LegacyDilate => hv.dilate(target_dim),
        Family::HadamardTruncateV1 => project(
            hv,
            target_dim,
            ContinuousProjectionFamily::HadamardTruncateV1,
        )
        .expect("strict power-of-two contraction"),
    }
}

fn project_field(
    object: &mut Value,
    field: &str,
    target_dim: usize,
    family: Family,
) {
    let source: ContinuousHV = serde_json::from_value(
        object
            .get(field)
            .cloned()
            .unwrap_or_else(|| panic!("missing neuron field {field}")),
    )
    .unwrap_or_else(|e| panic!("invalid ContinuousHV field {field}: {e}"));
    let projected = convert_hv(&source, target_dim, family);
    object[field] = serde_json::to_value(projected).expect("serialize projected HV");
}

/// Convert every dimension-bearing liquid component, not only the visible state.
///
/// This intentionally uses the existing serde representation as a research
/// boundary rather than introducing a production transition API prematurely.
fn convert_neuron(
    source: &HdcLtcUnifiedNeuron,
    target_dim: usize,
    family: Family,
) -> HdcLtcUnifiedNeuron {
    let mut value = serde_json::to_value(source).expect("serialize source neuron");

    for field in [
        "state",
        "weight_hv",
        "input_mask",
        "tau_modulator",
        "gate_weight",
        "gate_bias",
        "weight_momentum",
        "input_momentum",
    ] {
        project_field(&mut value, field, target_dim, family);
    }

    value["config"]["dimension"] = Value::from(target_dim as u64);

    serde_json::from_value(value).expect("deserialize converted neuron")
}

fn normalized_l2(a: &ContinuousHV, b: &ContinuousHV) -> f32 {
    a.subtract(b).norm() / a.norm().max(1e-12)
}

fn row(source_dim: usize, target_dim: usize, seed: u64, family: Family) -> Row {
    let mut high = neuron(source_dim, seed);
    let mut candidate = convert_neuron(&high, target_dim, family);

    let before = serde_json::to_value(&high).expect("serialize source");
    let after = serde_json::to_value(&candidate).expect("serialize converted");
    let clock_preserved = before["total_time"] == after["total_time"];
    let update_count_preserved = before["update_count"] == after["update_count"];

    let mut state_errors = Vec::with_capacity(DT.len());
    let mut tau_errors = Vec::with_capacity(DT.len());

    for (step, &dt) in DT.iter().enumerate() {
        let high_input = input(source_dim, step);
        let low_input = convert_hv(&high_input, target_dim, family);

        high.evolve_closed_form(dt, &high_input);
        candidate.evolve_closed_form(dt, &low_input);

        let projected_reference = convert_hv(high.state(), target_dim, family);
        state_errors.push(normalized_l2(candidate.state(), &projected_reference));

        let high_tau = high.effective_tau(&high_input);
        let low_tau = candidate.effective_tau(&low_input);
        tau_errors.push((high_tau - low_tau).abs());
    }

    let round_trip = convert_hv(
        &convert_hv(
            &ContinuousHV::random(source_dim, seed ^ 0x48595354),
            target_dim,
            family,
        ),
        source_dim,
        family,
    );
    let source_round_trip = ContinuousHV::random(source_dim, seed ^ 0x48595354);
    let hysteresis_error = normalized_l2(&source_round_trip, &round_trip);

    Row {
        schema_version: "hdc-ltc-resolution-trajectory-matrix.v1",
        source_dim,
        target_dim,
        family: family.name(),
        seed,
        steps: DT.len(),
        state_error_terminal: *state_errors.last().expect("trajectory has steps"),
        state_error_mean: state_errors.iter().sum::<f32>() / state_errors.len() as f32,
        tau_error_terminal: *tau_errors.last().expect("trajectory has steps"),
        tau_error_mean: tau_errors.iter().sum::<f32>() / tau_errors.len() as f32,
        hysteresis_error,
        clock_preserved,
        update_count_preserved,
    }
}

fn main() {
    let families = [Family::LegacyDilate, Family::HadamardTruncateV1];
    let mut rows = Vec::with_capacity(42);

    for (source_index, &source_dim) in DIMS.iter().enumerate().skip(1) {
        for (target_index, &target_dim) in DIMS.iter().enumerate().take(source_index) {
            let seed = SEED
                .wrapping_add((source_index as u64) << 32)
                .wrapping_add(target_index as u64);

            for family in families {
                rows.push(row(source_dim, target_dim, seed, family));
            }
        }
    }

    let payload = serde_json::json!({
        "schema_version": "hdc-ltc-resolution-trajectory-matrix.v1",
        "dimensions": DIMS,
        "transition_count": rows.len(),
        "families": ["legacy_dilate", "hadamard_truncate_v1"],
        "strict_contraction_transition_count_per_family": 21,
        "reference": "evolve_high_then_project",
        "candidate": "project_complete_liquid_state_then_evolve",
        "converted_components": [
            "state", "weight_hv", "input_mask", "tau_modulator",
            "gate_weight", "gate_bias", "weight_momentum", "input_momentum"
        ],
        "metrics": [
            "state_error_terminal", "state_error_mean",
            "tau_error_terminal", "tau_error_mean",
            "hysteresis_error", "clock_preserved", "update_count_preserved"
        ],
        "policy": {
            "purpose": "characterization",
            "unknown_is_not_zero": true,
            "no_production_quality_gate": true,
            "no_adaptive_controller_inferred": true
        },
        "rows": rows
    });

    println!("{}", serde_json::to_string_pretty(&payload).expect("serialize matrix"));
}
