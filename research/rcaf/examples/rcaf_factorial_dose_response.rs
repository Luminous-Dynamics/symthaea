use cetb_rcaf::factorial_counterfactual::FactorialExperimentMetadata;
use cetb_rcaf::factorial_dose_response::{
    run_factorial_dose_response_sweep, FactorialDoseResponseConfig,
};
use cetb_rcaf::motion_transient::{
    generate_motion_transient_field, LocalizedMotionEvent, MotionEvent, MotionTransientConfig,
};
use cetb_rcaf::spatial_nuisance::SpatialGraph;
use std::error::Error;
use std::fs;
use std::io::{self, Write};

fn main() -> Result<(), Box<dyn Error>> {
    let time_points = 64;
    let roi_count = 6;
    let roi_labels = (0..roi_count)
        .map(|index| format!("roi-{index:02}"))
        .collect::<Vec<_>>();
    let mut adjacency = vec![vec![0.0; roi_count]; roi_count];
    for index in 0..(roi_count - 1) {
        adjacency[index][index + 1] = 1.0;
        adjacency[index + 1][index] = 1.0;
    }
    let graph = SpatialGraph::new(roi_labels, adjacency)?;

    let clean = (0..time_points)
        .map(|time| {
            (0..roi_count)
                .map(|roi| {
                    ((time as f64 + 1.0) * (roi as f64 + 1.0) * 0.071).sin()
                        + 0.12 * ((time as f64 + roi as f64) * 0.17).cos()
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let target_signal = (0..time_points)
        .map(|time| {
            (0..roi_count)
                .map(|roi| {
                    0.06 * ((time as f64 + 2.0) * (roi as f64 + 1.0) * 0.043).sin()
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let transient_config = MotionTransientConfig {
        time_points,
        dose: 0.3,
        broad_loading: 0.15,
        localized_loading: 0.85,
        spatial_decay_length: 1.8,
        broad_events: vec![
            MotionEvent {
                onset: 12,
                amplitude: 1.0,
                decay_samples: 4.0,
            },
            MotionEvent {
                onset: 41,
                amplitude: -0.6,
                decay_samples: 5.5,
            },
        ],
        localized_events: vec![LocalizedMotionEvent {
            event: MotionEvent {
                onset: 27,
                amplitude: 0.8,
                decay_samples: 6.0,
            },
            seed_roi_index: 4,
        }],
    };
    let nuisance = generate_motion_transient_field(&graph, &transient_config)?;

    let config = FactorialDoseResponseConfig {
        metadata: FactorialExperimentMetadata {
            experiment_id: "s3-dose-response-01".to_string(),
            signal_id: "distributed-target-v1".to_string(),
            nuisance_family_id: "motion_transient".to_string(),
            nuisance_instance_id: "fixed-transient-events-12-27-41".to_string(),
            target_component_variance_fraction: None,
        },
        signal_scales: vec![0.0, 0.5, 1.0, 2.0],
        nuisance_component_variance_fractions: vec![0.0, 0.05, 0.10, 0.20],
    };
    let report = run_factorial_dose_response_sweep(
        &clean,
        &target_signal,
        &nuisance.values,
        &config,
    )?;
    if report.points.len() != 16 {
        return Err("factorial sweep did not generate the expected 4 x 4 grid".into());
    }
    let rendered = report.to_json_pretty();

    if let Some(path) = std::env::args_os().nth(1) {
        fs::write(path, rendered)?;
    } else {
        io::stdout().write_all(rendered.as_bytes())?;
    }
    Ok(())
}
