// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Run the real Symthaea vision manifold over a MOT-style sequence and evaluate
//! centroid tracking honestly as point tracking.
//!
//! This example does not claim HOTA/IDF1/MOTA and does not invent tracker boxes.

use std::collections::BTreeMap;
use std::error::Error;
use std::path::{Path, PathBuf};

use image::GenericImageView;
use symthaea_tracking_eval::{TrackingEvaluationPolicy, evaluate_sequence};
use symthaea_vision_manifold::{VisionConfig, VisionManifold};
use symthaea_vision_tracking_eval::{
    GroundTruthBox, frame_from_tracks, ground_truth_center_from_box,
};

#[derive(Debug)]
struct Args {
    sequence_dir: PathBuf,
    max_frames: usize,
    match_gate_norm: f64,
    target_size: u32,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    let image_dir = args.sequence_dir.join("img1");
    if !image_dir.is_dir() {
        return Err(format!("missing image directory: {}", image_dir.display()).into());
    }

    let ground_truth = load_ground_truth(&args.sequence_dir.join("gt/gt.txt"))?;

    let mut config = VisionConfig::default();
    config.enable_object_binding = true;
    config.enable_temporal_binding = true;
    let patch_size = config.patch_size.max(1);
    let mut manifold = VisionManifold::new(config, args.target_size, args.target_size);
    manifold.enable_object_memory(64);

    let grid_cols = (args.target_size as usize).div_ceil(patch_size);
    let grid_rows = (args.target_size as usize).div_ceil(patch_size);
    let dt = 1.0 / 30.0;
    let mut evaluation_frames = Vec::new();
    let mut processed = 0usize;

    for frame_index in 1..=args.max_frames {
        let image_path = image_dir.join(format!("{frame_index:06}.jpg"));
        if !image_path.exists() {
            break;
        }

        let image = image::open(&image_path)?;
        let (source_width, source_height) = image.dimensions();
        let resized = image.resize_exact(
            args.target_size,
            args.target_size,
            image::imageops::FilterType::Triangle,
        );
        let rgb = resized.to_rgb8();
        let pixels = rgb.into_raw();
        let _telemetry = manifold.observe_frame(
            &pixels,
            args.target_size,
            args.target_size,
            3,
            dt,
        );

        let frame_ground_truth = ground_truth
            .get(&(frame_index as u64))
            .into_iter()
            .flatten()
            .map(|bbox| {
                ground_truth_center_from_box(*bbox, source_width, source_height)
                    .map_err(|error| format!("frame {frame_index} ground truth: {error:?}"))
            })
            .collect::<Result<Vec<_>, _>>()?;

        let memory = manifold
            .object_memory()
            .ok_or("object memory unexpectedly disabled")?;
        let frame = frame_from_tracks(
            frame_index as u64,
            frame_ground_truth,
            memory.tracks(),
            grid_cols,
            grid_rows,
        )
        .map_err(|error| format!("frame {frame_index} prediction conversion: {error:?}"))?;
        evaluation_frames.push(frame);
        processed += 1;
    }

    if processed == 0 {
        return Err("no frames were processed".into());
    }

    let report = evaluate_sequence(
        &evaluation_frames,
        TrackingEvaluationPolicy {
            maximum_match_distance_norm: args.match_gate_norm,
        },
    )
    .map_err(|error| format!("tracking evaluation failed: {error:?}"))?;

    println!("=== Symthaea Vision Point-Tracking Evaluation ===");
    println!("sequence: {}", args.sequence_dir.display());
    println!("frames: {}", report.evaluated_frames);
    println!("ground-truth points: {}", report.ground_truth_points);
    println!("prediction points: {}", report.prediction_points);
    println!("TP: {}", report.true_positives);
    println!("FP: {}", report.false_positives);
    println!("FN: {}", report.false_negatives);
    println!("identity switches: {}", report.identity_switches);
    println!("false tracks: {}", report.false_track_count);
    println!("unique prediction tracks: {}", report.unique_prediction_tracks);
    println!("frames with FP: {}", report.frames_with_false_positives);
    println!("negative frames: {}", report.negative_frames);
    println!(
        "negative frames with FP: {}",
        report.negative_frames_with_false_positives
    );
    print_optional("precision", report.precision);
    print_optional("recall", report.recall);
    print_optional("F1", report.f1);
    print_optional(
        "mean normalized match distance",
        report.mean_match_distance_norm,
    );
    println!("FP/frame: {:.6}", report.false_positives_per_frame);
    print_optional(
        "clean-negative-frame fraction",
        report.clean_negative_frame_fraction,
    );
    print_optional("false-track fraction", report.false_track_fraction);
    println!("negative-only sequence: {}", report.negative_only_sequence);
    println!();
    println!("NOTE: centroid/point metrics only; this is not HOTA/IDF1/MOTA.");

    Ok(())
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let raw = std::env::args().skip(1).collect::<Vec<_>>();
    let Some(sequence_dir) = raw.first() else {
        return Err(
            "usage: mot_point_eval <sequence-dir> [--frames N] [--gate F] [--target N]".into(),
        );
    };

    let mut max_frames = usize::MAX;
    let mut match_gate_norm = 0.08;
    let mut target_size = 64u32;
    let mut index = 1usize;
    while index < raw.len() {
        let flag = &raw[index];
        let value = raw
            .get(index + 1)
            .ok_or_else(|| format!("missing value after {flag}"))?;
        match flag.as_str() {
            "--frames" => max_frames = value.parse()?,
            "--gate" => match_gate_norm = value.parse()?,
            "--target" => target_size = value.parse()?,
            _ => return Err(format!("unknown argument: {flag}").into()),
        }
        index += 2;
    }

    if max_frames == 0 {
        return Err("--frames must be > 0".into());
    }
    if target_size == 0 {
        return Err("--target must be > 0".into());
    }
    let policy = TrackingEvaluationPolicy {
        maximum_match_distance_norm: match_gate_norm,
    };
    if !policy.validate() {
        return Err("--gate must be finite and in (0, sqrt(2)]".into());
    }

    Ok(Args {
        sequence_dir: PathBuf::from(sequence_dir),
        max_frames,
        match_gate_norm,
        target_size,
    })
}

fn load_ground_truth(path: &Path) -> Result<BTreeMap<u64, Vec<GroundTruthBox>>, Box<dyn Error>> {
    if !path.exists() {
        // A sequence without GT is treated as background-only. This is useful for
        // long negative-scene false-alarm evaluation, provided the dataset is
        // externally established to contain no relevant tracked objects.
        return Ok(BTreeMap::new());
    }

    let mut frames = BTreeMap::<u64, Vec<GroundTruthBox>>::new();
    for (line_number, line) in std::fs::read_to_string(path)?.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let fields = line.split(',').collect::<Vec<_>>();
        if fields.len() < 6 {
            return Err(format!("invalid GT row at line {}", line_number + 1).into());
        }
        let frame: u64 = fields[0].trim().parse()?;
        let object_id: i64 = fields[1].trim().parse()?;
        if object_id < 0 {
            continue;
        }
        let confidence = fields
            .get(6)
            .and_then(|value| value.trim().parse::<f64>().ok())
            .unwrap_or(1.0);
        if confidence <= 0.0 {
            continue;
        }
        let bbox = GroundTruthBox {
            object_id: object_id as u64,
            x_px: fields[2].trim().parse()?,
            y_px: fields[3].trim().parse()?,
            width_px: fields[4].trim().parse()?,
            height_px: fields[5].trim().parse()?,
        };
        frames.entry(frame).or_default().push(bbox);
    }
    Ok(frames)
}

fn print_optional(label: &str, value: Option<f64>) {
    match value {
        Some(value) => println!("{label}: {value:.6}"),
        None => println!("{label}: n/a"),
    }
}
