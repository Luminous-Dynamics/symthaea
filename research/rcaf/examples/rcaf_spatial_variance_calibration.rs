use cetb_rcaf::spatial_calibration::calibrate_spatial_nuisance_to_component_fraction;
use cetb_rcaf::spatial_nuisance::{
    generate_spatial_nuisance_field, SpatialGraph, SpatialNuisanceConfig, SpatialNuisanceFamily,
};
use std::error::Error;
use std::fs;
use std::io::{self, Write};

fn render_calibration(
    family_id: &str,
    config: SpatialNuisanceConfig,
    graph: &SpatialGraph,
    clean: &[Vec<f64>],
    target_fraction: f64,
) -> Result<String, Box<dyn Error>> {
    let field = generate_spatial_nuisance_field(graph, &config, None)?;
    let (_, report) =
        calibrate_spatial_nuisance_to_component_fraction(clean, &field, target_fraction)?;
    let (spatial_rank, diffusion_time, seed_roi_indices) = match &config.family {
        SpatialNuisanceFamily::GraphSmoothLowRank {
            rank,
            diffusion_time,
        } => (*rank, *diffusion_time, String::from("null")),
        SpatialNuisanceFamily::HeatKernelLocalized {
            seed_roi_indices,
            diffusion_time,
        } => {
            let indices = seed_roi_indices
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(", ");
            (
                seed_roi_indices.len(),
                *diffusion_time,
                format!("[{indices}]"),
            )
        }
    };
    Ok(format!(
        concat!(
            "{{\n",
            "  \"family_id\": \"{}\",\n",
            "  \"spatial_rank\": {},\n",
            "  \"diffusion_time\": {:.12},\n",
            "  \"seed_roi_indices\": {},\n",
            "  \"rng_seed\": \"0x{:016x}\",\n",
            "  \"initial_additive_dose\": {:.12},\n",
            "  \"temporal_ar1\": {:.12},\n",
            "  \"temporal_coherence\": {:.12},\n",
            "  \"target_component_variance_fraction\": {:.12},\n",
            "  \"realized_component_variance_fraction\": {:.12},\n",
            "  \"clean_component_variance\": {:.12},\n",
            "  \"nuisance_component_variance\": {:.12},\n",
            "  \"clean_nuisance_covariance\": {:.12},\n",
            "  \"covariance_contribution_fraction\": {:.12},\n",
            "  \"combined_observed_variance\": {:.12},\n",
            "  \"applied_scale_multiplier\": {:.12}\n",
            "}}"
        ),
        family_id,
        spatial_rank,
        diffusion_time,
        seed_roi_indices,
        config.seed,
        config.dose,
        config.temporal_ar1,
        config.temporal_coherence,
        report.requested_component_variance_fraction,
        report.realized_component_variance_fraction,
        report.clean_component_variance,
        report.nuisance_component_variance,
        report.clean_nuisance_covariance,
        report.covariance_contribution_fraction,
        report.combined_observed_variance,
        report.applied_scale_multiplier,
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

    let time_points = 160;
    let clean = (0..time_points)
        .map(|time| {
            (0..8)
                .map(|roi| {
                    ((time as f64 + 1.0) * (roi as f64 + 1.0) * 0.031).sin()
                        + 0.15 * ((time as f64 + roi as f64) * 0.17).cos()
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let target_fraction = 0.10;
    let base = SpatialNuisanceConfig {
        family: SpatialNuisanceFamily::GraphSmoothLowRank {
            rank: 2,
            diffusion_time: 0.35,
        },
        time_points,
        dose: 0.20,
        temporal_ar1: 0.30,
        temporal_coherence: 0.0,
        seed: 0x5243_4146_5641_5231,
    };
    let graph_smooth = render_calibration(
        "graph_smooth_low_rank",
        base.clone(),
        &graph,
        &clean,
        target_fraction,
    )?;
    let heat_kernel = render_calibration(
        "heat_kernel_localized",
        SpatialNuisanceConfig {
            family: SpatialNuisanceFamily::HeatKernelLocalized {
                seed_roi_indices: vec![3],
                diffusion_time: 0.35,
            },
            ..base
        },
        &graph,
        &clean,
        target_fraction,
    )?;

    let report = format!(
        concat!(
            "{{\n",
            "  \"schema_version\": \"rcaf.spatial-variance-calibration/0.1.0\",\n",
            "  \"source_status\": \"synthetic_fixture\",\n",
            "  \"geometry\": {{\n",
            "    \"representation\": \"weighted_undirected_graph\",\n",
            "    \"source\": \"synthetic_path_graph\",\n",
            "    \"interpretation_scope\": \"synthetic_topology_only\",\n",
            "    \"roi_labels\": [{rendered_labels}],\n",
            "    \"weighted_edges\": [{rendered_edges}]\n",
            "  }},\n",
            "  \"time_points\": {},\n",
            "  \"target_component_variance_fraction\": {:.12},\n",
            "  \"families\": [\n{},\n{}\n  ],\n",
            "  \"claim_limits\": [\n",
            "    \"component variance fraction excludes clean-nuisance covariance\",\n",
            "    \"combined observed variance includes the covariance contribution\",\n",
            "    \"synthetic path graph only; not an anatomical atlas\",\n",
            "    \"does not qualify empirical acquisition or biological validity\"\n",
            "  ]\n",
            "}}\n"
        ),
        time_points, target_fraction, graph_smooth, heat_kernel,
    );

    if let Some(path) = std::env::args_os().nth(1) {
        fs::write(path, report)?;
    } else {
        io::stdout().write_all(report.as_bytes())?;
    }
    Ok(())
}
