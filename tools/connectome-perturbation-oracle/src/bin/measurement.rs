//! Emits explicit history/measurement-channel synthetic oracle cases.
//!
//! This binary keeps pulse-order dynamics and fluorescence observation
//! separate from anatomical graph and latent-state reports.

use std::process::ExitCode;

use connectome_perturbation_oracle::{
    apply_calcium_observation, simulate_contextual, CalciumObservationKernel,
    CircuitState, ContextualEdge, ContextualNetwork, Edge, EdgeScope, Injection,
    NodeTransfer, Trajectory,
};

fn vector_json(values: &[f64]) -> String {
    let mut out = String::from("[");
    for (index, value) in values.iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        out.push_str(&format!("{value:.12}"));
    }
    out.push(']');
    out
}

fn build_report() -> Result<String, String> {
    let network = ContextualNetwork {
        node_count: 3,
        edges: vec![
            ContextualEdge {
                edge: Edge {
                    source: 0,
                    target: 2,
                    weight: 0.8,
                    delay: 1,
                },
                scope: EdgeScope::Both,
            },
            ContextualEdge {
                edge: Edge {
                    source: 1,
                    target: 2,
                    weight: 0.4,
                    delay: 1,
                },
                scope: EdgeScope::Both,
            },
        ],
        transfers: vec![
            NodeTransfer::Identity,
            NodeTransfer::Identity,
            NodeTransfer::AdaptiveThreshold {
                threshold: 0.7,
                persistence: 0.5,
            },
        ],
    };

    let a_then_b = simulate_contextual(
        &network,
        CircuitState::Active,
        4,
        &[
            Injection {
                node: 0,
                epoch: 0,
                amplitude: 1.0,
            },
            Injection {
                node: 1,
                epoch: 1,
                amplitude: 1.0,
            },
        ],
    )
    .map_err(|error| error.to_string())?
    .node_trace(2)
    .map_err(|error| error.to_string())?;

    let b_then_a = simulate_contextual(
        &network,
        CircuitState::Active,
        4,
        &[
            Injection {
                node: 1,
                epoch: 0,
                amplitude: 1.0,
            },
            Injection {
                node: 0,
                epoch: 1,
                amplitude: 1.0,
            },
        ],
    )
    .map_err(|error| error.to_string())?
    .node_trace(2)
    .map_err(|error| error.to_string())?;

    let latent = Trajectory {
        values: vec![vec![1.0], vec![0.0], vec![0.0]],
    };
    let observed = apply_calcium_observation(
        &latent,
        CalciumObservationKernel {
            decay: 0.5,
            gain: 2.0,
            baseline: 0.0,
        },
    )
    .map_err(|error| error.to_string())?
    .node_trace(0)
    .map_err(|error| error.to_string())?;

    let pulse_ok = a_then_b == vec![0.0, 1.0, 1.0, 0.0]
        && b_then_a == vec![0.0, 0.0, 1.0, 0.0];
    let observation_ok = observed == vec![2.0, 1.0, 0.5];
    let pulse_status = if pulse_ok { "PASS" } else { "FAIL" };
    let observation_status = if observation_ok { "PASS" } else { "FAIL" };

    Ok(format!(
        concat!(
            "{{\n",
            "  \"schema_version\": 1,\n",
            "  \"evidence_class\": \"synthetic_known_answer\",\n",
            "  \"claim_ceiling\": \"SYNTHETIC_ONLY\",\n",
            "  \"cases\": [\n",
            "    {{\"id\":\"pulse_order_dependence\",\"response_a_then_b\":{},",
            "\"response_b_then_a\":{},\"status\":\"{}\"}},\n",
            "    {{\"id\":\"calcium_observation_kernel\",\"latent\":{},",
            "\"observed\":{},\"status\":\"{}\"}}\n",
            "  ]\n",
            "}}\n"
        ),
        vector_json(&a_then_b),
        vector_json(&b_then_a),
        pulse_status,
        vector_json(&[1.0, 0.0, 0.0]),
        vector_json(&observed),
        observation_status
    ))
}

fn main() -> ExitCode {
    match build_report() {
        Ok(report) => {
            print!("{report}");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("measurement oracle failed closed: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn report_matches_pinned_measurement_fixture_byte_for_byte() {
        let expected = include_str!("../../fixtures/measurement-suite.expected.json");
        assert_eq!(build_report().unwrap(), expected);
    }

    #[test]
    fn pulse_order_and_observation_vectors_are_distinct_known_answers() {
        let report = build_report().unwrap();
        assert!(report.contains("\"response_a_then_b\":[0.000000000000,1.000000000000,1.000000000000,0.000000000000]"));
        assert!(report.contains("\"response_b_then_a\":[0.000000000000,0.000000000000,1.000000000000,0.000000000000]"));
        assert!(report.contains("\"observed\":[2.000000000000,1.000000000000,0.500000000000]"));
    }
}
