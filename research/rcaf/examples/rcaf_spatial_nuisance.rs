use cetb_rcaf::spatial_nuisance::{
    generate_spatial_nuisance_field, SpatialGraph, SpatialNuisanceConfig, SpatialNuisanceFamily,
};
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

fn render_family(
    family_id: &str,
    config: SpatialNuisanceConfig,
    graph: &SpatialGraph,
) -> Result<String, Box<dyn Error>> {
    let field = generate_spatial_nuisance_field(graph, &config, None)?;
    let rank = field.spatial_basis.first().map_or(0, Vec::len);
    let (diffusion_time, seed_roi_indices) = match &config.family {
        SpatialNuisanceFamily::GraphSmoothLowRank { diffusion_time, .. } => {
            (*diffusion_time, String::from("null"))
        }
        SpatialNuisanceFamily::HeatKernelLocalized {
            seed_roi_indices,
            diffusion_time,
        } => {
            let indices = seed_roi_indices
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(", ");
            (*diffusion_time, format!("[{indices}]"))
        }
    };
    let autocorrelation = field
        .realized_temporal_ar1
        .iter()
        .map(|value| format!("{value:.12}"))
        .collect::<Vec<_>>()
        .join(", ");
    Ok(format!(
        "{{\n  \"family_id\": \"{family_id}\",\n  \"spatial_rank\": {rank},\n  \"diffusion_time\": {:.12},\n  \"seed_roi_indices\": {seed_roi_indices},\n  \"dose\": {:.12},\n  \"temporal_ar1\": {:.12},\n  \"temporal_coherence\": {:.12},\n  \"rng_seed\": \"0x{:016x}\",\n  \"rng_algorithm\": \"xorshift64-boxmuller-v1\",\n  \"realized_field_rms\": {:.12},\n  \"realized_temporal_ar1\": [{autocorrelation}]\n}}",
        diffusion_time,
        config.dose,
        config.temporal_ar1,
        config.temporal_coherence,
        config.seed,
        field_rms(&field.values),
    ))
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

    let common = SpatialNuisanceConfig {
        family: SpatialNuisanceFamily::GraphSmoothLowRank {
            rank: 2,
            diffusion_time: 0.35,
        },
        time_points: 160,
        dose: 0.20,
        temporal_ar1: 0.30,
        temporal_coherence: 0.0,
        seed: 0x5243_4146_5350_4154,
    };
    let graph_smooth = render_family("graph_smooth_low_rank", common.clone(), &graph)?;
    let localized = render_family(
        "heat_kernel_localized",
        SpatialNuisanceConfig {
            family: SpatialNuisanceFamily::HeatKernelLocalized {
                seed_roi_indices: vec![3],
                diffusion_time: 0.35,
            },
            ..common
        },
        &graph,
    )?;

    let report = format!(
        "{{\n  \"schema_version\": \"rcaf.spatial-nuisance-field/0.1.0\",\n  \"source_status\": \"synthetic_fixture\",\n  \"geometry\": {{\n    \"representation\": \"weighted_undirected_graph\",\n    \"source\": \"synthetic_path_graph\",\n    \"interpretation_scope\": \"synthetic_topology_only\",\n    \"roi_count\": 8,\n    \"edge_count\": 7,\n    \"roi_labels\": [{rendered_labels}],\n    \"weighted_edges\": [{rendered_edges}]\n  }},\n  \"time_points\": 160,\n  \"families\": [\n{graph_smooth},\n{localized}\n  ],\n  \"claim_limits\": [\n    \"synthetic graph topology only; not an anatomical atlas\",\n    \"field generator qualification only; no biological data analysed\",\n    \"dose is an additive loading, not a nuisance variance fraction\",\n    \"does not qualify preprocessing or causal interpretation\"\n  ]\n}}\n"
    );

    if let Some(path) = std::env::args_os().nth(1) {
        fs::write(path, report)?;
    } else {
        io::stdout().write_all(report.as_bytes())?;
    }
    Ok(())
}
