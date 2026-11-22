use cetb_rcaf::motion_transient::{
    generate_motion_transient_field, LocalizedMotionEvent, MotionEvent, MotionTransientConfig,
};
use cetb_rcaf::spatial_nuisance::SpatialGraph;
use std::error::Error;
use std::fs;
use std::io::{self, Write};

fn field_rms(values: &[Vec<f64>]) -> f64 {
    let count = values.iter().map(Vec::len).sum::<usize>();
    let sum_squares = values
        .iter()
        .flatten()
        .map(|value| value * value)
        .sum::<f64>();
    (sum_squares / count as f64).sqrt()
}

fn max_abs(values: &[Vec<f64>]) -> f64 {
    values
        .iter()
        .flatten()
        .map(|value| value.abs())
        .fold(0.0, f64::max)
}

fn main() -> Result<(), Box<dyn Error>> {
    let roi_labels = (0..8)
        .map(|index| format!("roi-{index:02}"))
        .collect::<Vec<_>>();
    let mut adjacency = vec![vec![0.0; 8]; 8];
    for index in 0..7 {
        adjacency[index][index + 1] = 1.0;
        adjacency[index + 1][index] = 1.0;
    }
    let graph = SpatialGraph::new(roi_labels, adjacency)?;

    let config = MotionTransientConfig {
        time_points: 160,
        dose: 0.25,
        broad_loading: 0.15,
        localized_loading: 0.85,
        spatial_decay_length: 2.0,
        broad_events: vec![
            MotionEvent {
                onset: 24,
                amplitude: 1.0,
                decay_samples: 5.0,
            },
            MotionEvent {
                onset: 83,
                amplitude: 0.7,
                decay_samples: 3.5,
            },
        ],
        localized_events: vec![
            LocalizedMotionEvent {
                event: MotionEvent {
                    onset: 31,
                    amplitude: 0.8,
                    decay_samples: 4.0,
                },
                seed_roi_index: 2,
            },
            LocalizedMotionEvent {
                event: MotionEvent {
                    onset: 91,
                    amplitude: -0.5,
                    decay_samples: 6.0,
                },
                seed_roi_index: 6,
            },
        ],
    };
    let field = generate_motion_transient_field(&graph, &config)?;

    let rendered_labels = graph
        .roi_labels
        .iter()
        .map(|label| format!("\"{label}\""))
        .collect::<Vec<_>>()
        .join(", ");
    let mut edge_rows = Vec::new();
    for (left, row) in graph.adjacency.iter().enumerate() {
        for (right, weight) in row.iter().enumerate().skip(left + 1) {
            if *weight > 0.0 {
                edge_rows.push(format!(
                    "[\"{}\", \"{}\", {:.12}]",
                    graph.roi_labels[left], graph.roi_labels[right], weight
                ));
            }
        }
    }
    let rendered_edges = edge_rows.join(", ");

    let broad_events = config
        .broad_events
        .iter()
        .map(|event| {
            format!(
                "{{\"onset\": {}, \"amplitude\": {:.12}, \"decay_samples\": {:.12}}}",
                event.onset, event.amplitude, event.decay_samples
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    let localized_events = config
        .localized_events
        .iter()
        .map(|localized| {
            format!(
                "{{\"onset\": {}, \"amplitude\": {:.12}, \"decay_samples\": {:.12}, \"seed_roi_index\": {}}}",
                localized.event.onset,
                localized.event.amplitude,
                localized.event.decay_samples,
                localized.seed_roi_index
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    let localized_loadings = field
        .localized_spatial_loadings
        .iter()
        .map(|loading| {
            format!(
                "[{}]",
                loading
                    .iter()
                    .map(|value| format!("{value:.12}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })
        .collect::<Vec<_>>()
        .join(", ");

    let report = format!(
        concat!(
            "{{\n",
            "  \"schema_version\": \"rcaf.motion-transient/0.1.0\",\n",
            "  \"source_status\": \"synthetic_fixture\",\n",
            "  \"geometry\": {{\n",
            "    \"representation\": \"weighted_undirected_graph\",\n",
            "    \"source\": \"synthetic_path_graph\",\n",
            "    \"interpretation_scope\": \"synthetic_topology_only\",\n",
            "    \"roi_labels\": [{}],\n",
            "    \"weighted_edges\": [{}],\n",
            "    \"edge_length_rule\": \"inverse_adjacency_weight\"\n",
            "  }},\n",
            "  \"time_points\": {},\n",
            "  \"dose\": {:.12},\n",
            "  \"broad_loading\": {:.12},\n",
            "  \"localized_loading\": {:.12},\n",
            "  \"spatial_decay_length\": {:.12},\n",
            "  \"event_train_rule\": \"amplitude_exp_negative_elapsed_over_decay_samples\",\n",
            "  \"broad_events\": [{}],\n",
            "  \"localized_events\": [{}],\n",
            "  \"localized_spatial_loadings\": [{}],\n",
            "  \"realized_field_rms\": {:.12},\n",
            "  \"realized_field_max_abs\": {:.12},\n",
            "  \"claim_limits\": [\n",
            "    \"stylized synthetic transient nuisance; not rigid-body motion or scanner physics\",\n",
            "    \"synthetic path graph only; not an anatomical atlas\",\n",
            "    \"event onsets and amplitudes are declared inputs, not empirical motion estimates\",\n",
            "    \"field generator only; no empirical or biological validation\"\n",
            "  ]\n",
            "}}\n"
        ),
        rendered_labels,
        rendered_edges,
        config.time_points,
        config.dose,
        config.broad_loading,
        config.localized_loading,
        config.spatial_decay_length,
        broad_events,
        localized_events,
        localized_loadings,
        field_rms(&field.values),
        max_abs(&field.values),
    );

    if let Some(path) = std::env::args_os().nth(1) {
        fs::write(path, report)?;
    } else {
        io::stdout().write_all(report.as_bytes())?;
    }
    Ok(())
}
