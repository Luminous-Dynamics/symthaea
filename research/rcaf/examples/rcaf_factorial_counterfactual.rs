use cetb_rcaf::factorial_counterfactual::{
    run_factorial_counterfactual_experiment, FactorialExperimentMetadata,
};
use cetb_rcaf::motion_transient::{
    generate_motion_transient_field, LocalizedMotionEvent, MotionEvent, MotionTransientConfig,
};
use cetb_rcaf::spatial_calibration::{
    calibrate_nuisance_matrix_to_component_fraction,
    calibrate_spatial_nuisance_to_component_fraction,
};
use cetb_rcaf::spatial_nuisance::{
    generate_spatial_nuisance_field, SpatialGraph, SpatialNuisanceConfig, SpatialNuisanceFamily,
};
use std::error::Error;
use std::fs;
use std::io::{self, Write};

fn clean_observations(time_points: usize, roi_count: usize) -> Vec<Vec<f64>> {
    (0..time_points)
        .map(|time| {
            (0..roi_count)
                .map(|roi| {
                    ((time as f64 + 1.0) * (roi as f64 + 1.0) * 0.031).sin()
                        + 0.15 * ((time as f64 + roi as f64) * 0.17).cos()
                })
                .collect()
        })
        .collect()
}

fn target_signal(time_points: usize, roi_count: usize) -> Vec<Vec<f64>> {
    (0..time_points)
        .map(|time| {
            (0..roi_count)
                .map(|roi| 0.08 * ((time as f64 + 2.0) * (roi as f64 + 1.0) * 0.073).sin())
                .collect()
        })
        .collect()
}

fn graph() -> Result<SpatialGraph, Box<dyn Error>> {
    let labels = (0..8)
        .map(|index| format!("roi-{index:02}"))
        .collect::<Vec<_>>();
    let mut adjacency = vec![vec![0.0; 8]; 8];
    for index in 0..7 {
        adjacency[index][index + 1] = 1.0;
        adjacency[index + 1][index] = 1.0;
    }
    Ok(SpatialGraph::new(labels, adjacency)?)
}

fn run_family(
    family_id: &str,
    instance_id: &str,
    clean: &[Vec<f64>],
    signal: &[Vec<f64>],
    nuisance: &[Vec<f64>],
    target_fraction: Option<f64>,
) -> Result<String, Box<dyn Error>> {
    let experiment = run_factorial_counterfactual_experiment(
        clean,
        signal,
        nuisance,
        &FactorialExperimentMetadata {
            experiment_id: format!("factorial-{family_id}"),
            signal_id: "distributed_target-v1".to_string(),
            nuisance_family_id: family_id.to_string(),
            nuisance_instance_id: instance_id.to_string(),
            target_component_variance_fraction: target_fraction,
        },
    )?;
    if !experiment.report.invariants.all_invariants_within_tolerance {
        return Err(format!("factorial invariants failed for {family_id}").into());
    }
    Ok(experiment.report.to_json_pretty())
}

fn main() -> Result<(), Box<dyn Error>> {
    let time_points = 96;
    let roi_count = 8;
    let clean = clean_observations(time_points, roi_count);
    let signal = target_signal(time_points, roi_count);
    let graph = graph()?;

    let common_mode = (0..time_points)
        .map(|time| {
            let value = 0.05 * ((time as f64 + 1.0) * 0.09).sin();
            vec![value; roi_count]
        })
        .collect::<Vec<_>>();

    let base = SpatialNuisanceConfig {
        family: SpatialNuisanceFamily::GraphSmoothLowRank {
            rank: 2,
            diffusion_time: 0.35,
        },
        time_points,
        dose: 0.20,
        temporal_ar1: 0.30,
        temporal_coherence: 0.0,
        seed: 0x5243_4146_4641_4354,
    };
    let graph_field = generate_spatial_nuisance_field(&graph, &base, None)?;
    let (calibrated_graph, _graph_calibration) =
        calibrate_spatial_nuisance_to_component_fraction(&clean, &graph_field, 0.10)?;
    let graph_smooth = run_family(
        "graph_smooth_low_rank",
        "graph-smooth-seed-5243464146414354",
        &clean,
        &signal,
        &calibrated_graph.values,
        Some(0.10),
    )?;

    let heat_config = SpatialNuisanceConfig {
        family: SpatialNuisanceFamily::HeatKernelLocalized {
            seed_roi_indices: vec![3],
            diffusion_time: 0.35,
        },
        ..base
    };
    let heat_field = generate_spatial_nuisance_field(&graph, &heat_config, None)?;
    let (calibrated_heat, _heat_calibration) =
        calibrate_spatial_nuisance_to_component_fraction(&clean, &heat_field, 0.10)?;
    let heat_kernel = run_family(
        "heat_kernel_localized",
        "heat-kernel-seed-roi-03",
        &clean,
        &signal,
        &calibrated_heat.values,
        Some(0.10),
    )?;

    let (calibrated_common_mode, _common_mode_calibration) =
        calibrate_nuisance_matrix_to_component_fraction(&clean, &common_mode, 0.10)?;
    let common_mode = run_family(
        "common_mode",
        "common-mode-sinusoid-calibrated-10pct-v1",
        &clean,
        &signal,
        &calibrated_common_mode,
        Some(0.10),
    )?;

    let transient_config = MotionTransientConfig {
        time_points,
        dose: 0.25,
        broad_loading: 0.15,
        localized_loading: 0.85,
        spatial_decay_length: 2.0,
        broad_events: vec![MotionEvent {
            onset: 24,
            amplitude: 1.0,
            decay_samples: 5.0,
        }],
        localized_events: vec![LocalizedMotionEvent {
            event: MotionEvent {
                onset: 57,
                amplitude: -0.6,
                decay_samples: 6.0,
            },
            seed_roi_index: 5,
        }],
    };
    let transient_field = generate_motion_transient_field(&graph, &transient_config)?;
    let (calibrated_transient, _transient_calibration) =
        calibrate_nuisance_matrix_to_component_fraction(
            &clean,
            &transient_field.values,
            0.10,
        )?;
    let motion_transient = run_family(
        "motion_transient",
        "transient-onset-24-57-seed-roi-05-calibrated-10pct",
        &clean,
        &signal,
        &calibrated_transient,
        Some(0.10),
    )?;

    let report = format!(
        concat!(
            "{{\n",
            "  \"schema_version\": \"rcaf.factorial-counterfactual-suite/0.1.0\",\n",
            "  \"source_status\": \"synthetic_fixture\",\n",
            "  \"time_points\": {},\n",
            "  \"roi_count\": {},\n",
            "  \"experiment_count\": 4,\n",
            "  \"nuisance_families\": [\n",
            "    \"common_mode\",\n",
            "    \"graph_smooth_low_rank\",\n",
            "    \"heat_kernel_localized\",\n",
            "    \"motion_transient\"\n",
            "  ],\n",
            "  \"experiments\": [\n{},\n{},\n{},\n{}\n  ],\n",
            "  \"claim_limits\": [\n",
            "    \"synthetic matched counterfactuals only; no empirical neuroimaging data analysed\",\n",
            "    \"all cells in an experiment reuse the same exact target and nuisance component matrices\",\n",
            "    \"component variance fraction excludes clean-nuisance covariance, which is reported separately\",\n",
            "    \"cell fingerprints are FNV-1a diagnostics; the workflow receipt provides content addressing\",\n",
            "    \"does not qualify a biological endpoint or a preprocessing package\"\n",
            "  ]\n",
            "}}\n"
        ),
        time_points,
        roi_count,
        indent_json(&common_mode),
        indent_json(&graph_smooth),
        indent_json(&heat_kernel),
        indent_json(&motion_transient),
    );

    if let Some(path) = std::env::args_os().nth(1) {
        fs::write(path, report)?;
    } else {
        io::stdout().write_all(report.as_bytes())?;
    }
    Ok(())
}

fn indent_json(value: &str) -> String {
    value
        .lines()
        .map(|line| format!("    {line}"))
        .collect::<Vec<_>>()
        .join("\n")
}
