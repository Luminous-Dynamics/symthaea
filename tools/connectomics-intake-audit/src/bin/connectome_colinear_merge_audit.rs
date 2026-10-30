//! Synthetic centreline change-point baselines for hidden co-linear merge proposals.
//!
//! The detector only consumes ordered morphology-feature profiles. Ground-truth
//! boundary labels are loaded separately and used after proposal generation for
//! evaluation. This is a 1D synthetic baseline, not an EM/mesh reconstruction model.

use clap::Parser;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Debug, Parser)]
#[command(
    name = "connectome-colinear-merge-audit",
    version,
    about = "Generate hidden co-linear merge boundary proposals from ordered morphology profiles"
)]
struct Args {
    /// Detector-only ordered profile input (must not contain ground-truth labels).
    #[arg(long)]
    input: PathBuf,
    /// Optional evaluator-only boundary truth file, never passed to detection.
    #[arg(long)]
    truth: Option<PathBuf>,
    /// Boundary matching tolerance in nanometres for evaluation.
    #[arg(long, default_value_t = 500.0)]
    tolerance_nm: f64,
    /// Optional JSON report path; stdout when omitted.
    #[arg(long)]
    output: Option<PathBuf>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Input {
    schema_version: u32,
    dataset_id: String,
    dataset_revision_commit_sha: String,
    detector: DetectorConfig,
    noise: FeatureNoise,
    tracks: Vec<Track>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct DetectorConfig {
    half_window_samples: usize,
    score_threshold: f64,
    min_separation_nm: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct FeatureNoise {
    radius_nm: f64,
    tangent_degrees: f64,
    membrane_integrity: f64,
    synapse_density: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Track {
    track_id: String,
    samples: Vec<Sample>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Sample {
    distance_nm: f64,
    radius_nm: f64,
    tangent_degrees: f64,
    membrane_integrity: f64,
    synapse_density: f64,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Truth {
    schema_version: u32,
    input_sha256: String,
    tracks: Vec<TruthTrack>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct TruthTrack {
    track_id: String,
    boundaries_nm: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct Candidate {
    track_id: String,
    boundary_nm: f64,
    score: f64,
}

#[derive(Debug, Serialize)]
struct MethodReport {
    method: &'static str,
    candidate_count: usize,
    candidates: Vec<Candidate>,
    evaluation: Option<Evaluation>,
}

#[derive(Debug, Serialize)]
struct Evaluation {
    tolerance_nm: f64,
    truth_boundary_count: usize,
    true_positive_count: usize,
    false_positive_count: usize,
    false_negative_count: usize,
    precision: Option<f64>,
    recall: Option<f64>,
    f1: Option<f64>,
    median_absolute_localization_error_nm: Option<f64>,
    p90_absolute_localization_error_nm: Option<f64>,
    false_proposals_per_cable_mm: Option<f64>,
}

#[derive(Debug, Serialize)]
struct Report {
    schema_version: u32,
    tool: &'static str,
    input_sha256: String,
    truth_sha256: Option<String>,
    dataset_id: String,
    dataset_revision_commit_sha: String,
    total_track_count: usize,
    total_sample_count: usize,
    total_cable_length_nm: f64,
    method_reports: Vec<MethodReport>,
    status: &'static str,
    information_flow: &'static str,
    scope_note: &'static str,
}

#[derive(Clone, Copy)]
enum Feature {
    Radius,
    Tangent,
    Membrane,
    Synapse,
}

impl Feature {
    fn value(self, sample: &Sample) -> f64 {
        match self {
            Self::Radius => sample.radius_nm,
            Self::Tangent => sample.tangent_degrees,
            Self::Membrane => sample.membrane_integrity,
            Self::Synapse => sample.synapse_density,
        }
    }

    fn sigma(self, noise: &FeatureNoise) -> f64 {
        match self {
            Self::Radius => noise.radius_nm,
            Self::Tangent => noise.tangent_degrees,
            Self::Membrane => noise.membrane_integrity,
            Self::Synapse => noise.synapse_density,
        }
    }
}

fn sha256(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn valid_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn validate_input(input: &Input) -> Result<(), String> {
    if input.schema_version != 1 || input.dataset_id.trim().is_empty() {
        return Err("schema_version must equal 1 and dataset_id must be non-empty".into());
    }
    if !valid_hex(&input.dataset_revision_commit_sha, 40) {
        return Err("dataset_revision_commit_sha must be 40 lowercase hexadecimal characters".into());
    }
    if input.detector.half_window_samples < 3 {
        return Err("half_window_samples must be at least 3 for piecewise-linear fitting".into());
    }
    if !input.detector.score_threshold.is_finite() || input.detector.score_threshold <= 0.0
        || !input.detector.min_separation_nm.is_finite() || input.detector.min_separation_nm <= 0.0
    {
        return Err("detector score threshold and minimum separation must be finite and positive".into());
    }
    for (name, value) in [
        ("radius_nm", input.noise.radius_nm),
        ("tangent_degrees", input.noise.tangent_degrees),
        ("membrane_integrity", input.noise.membrane_integrity),
        ("synapse_density", input.noise.synapse_density),
    ] {
        if !value.is_finite() || value <= 0.0 {
            return Err(format!("noise sigma {name} must be finite and positive"));
        }
    }
    if input.tracks.is_empty() {
        return Err("tracks must be non-empty".into());
    }
    let mut track_ids = BTreeSet::new();
    for track in &input.tracks {
        if track.track_id.trim().is_empty() || !track_ids.insert(track.track_id.clone()) {
            return Err("track IDs must be non-empty and unique".into());
        }
        let window = input.detector.half_window_samples;
        if track.samples.len() < (2 * window + 1) {
            return Err(format!("track {} needs at least {} samples", track.track_id, 2 * window + 1));
        }
        let mut previous_distance = None;
        for sample in &track.samples {
            let values = [
                sample.distance_nm,
                sample.radius_nm,
                sample.tangent_degrees,
                sample.membrane_integrity,
                sample.synapse_density,
            ];
            if values.iter().any(|value| !value.is_finite()) {
                return Err(format!("track {} contains non-finite sample features", track.track_id));
            }
            if sample.radius_nm <= 0.0 || sample.membrane_integrity < 0.0
                || sample.membrane_integrity > 1.0 || sample.synapse_density < 0.0
            {
                return Err(format!("track {} contains out-of-domain morphology features", track.track_id));
            }
            if previous_distance.is_some_and(|previous| sample.distance_nm <= previous) {
                return Err(format!("track {} distances must be strictly increasing", track.track_id));
            }
            previous_distance = Some(sample.distance_nm);
        }
    }
    Ok(())
}

fn fit_sse(samples: &[Sample], feature: Feature) -> f64 {
    if samples.len() < 2 {
        return 0.0;
    }
    let mean_x = samples.iter().map(|sample| sample.distance_nm).sum::<f64>() / samples.len() as f64;
    let mean_y = samples.iter().map(|sample| feature.value(sample)).sum::<f64>() / samples.len() as f64;
    let variance_x = samples.iter().map(|sample| (sample.distance_nm - mean_x).powi(2)).sum::<f64>();
    if variance_x <= f64::EPSILON {
        return f64::INFINITY;
    }
    let covariance = samples.iter()
        .map(|sample| (sample.distance_nm - mean_x) * (feature.value(sample) - mean_y))
        .sum::<f64>();
    let slope = covariance / variance_x;
    let intercept = mean_y - slope * mean_x;
    samples.iter()
        .map(|sample| (feature.value(sample) - (intercept + slope * sample.distance_nm)).powi(2))
        .sum()
}

fn feature_gain(samples: &[Sample], split: usize, half_window: usize, feature: Feature, sigma: f64) -> f64 {
    let start = split - half_window;
    let end = split + half_window;
    let local = &samples[start..end];
    let left = &samples[start..split];
    let right = &samples[split..end];
    let null_sse = fit_sse(local, feature);
    let split_sse = fit_sse(left, feature) + fit_sse(right, feature);
    // A two-line alternative adds two slope/intercept degrees of freedom.
    // Subtract their expected noise contribution rather than rewarding pure overfit.
    ((null_sse - split_sse).max(0.0) / sigma.powi(2) - 2.0).max(0.0)
}

fn score_at(track: &Track, split: usize, input: &Input, multi_feature: bool) -> f64 {
    let features: &[Feature] = if multi_feature {
        &[Feature::Radius, Feature::Tangent, Feature::Membrane, Feature::Synapse]
    } else {
        &[Feature::Radius]
    };
    let gains = features.iter()
        .map(|feature| feature_gain(
            &track.samples,
            split,
            input.detector.half_window_samples,
            *feature,
            feature.sigma(&input.noise),
        ))
        .sum::<f64>();
    gains.sqrt()
}

fn detect_track(track: &Track, input: &Input, multi_feature: bool) -> Vec<Candidate> {
    let half_window = input.detector.half_window_samples;
    let mut local_maxima = Vec::<Candidate>::new();
    for split in half_window..=(track.samples.len() - half_window) {
        let score = score_at(track, split, input, multi_feature);
        if score < input.detector.score_threshold {
            continue;
        }
        let left_score = if split > half_window {
            score_at(track, split - 1, input, multi_feature)
        } else {
            f64::NEG_INFINITY
        };
        let right_score = if split < track.samples.len() - half_window {
            score_at(track, split + 1, input, multi_feature)
        } else {
            f64::NEG_INFINITY
        };
        if score >= left_score && score >= right_score {
            local_maxima.push(Candidate {
                track_id: track.track_id.clone(),
                boundary_nm: track.samples[split].distance_nm,
                score,
            });
        }
    }

    // Greedy non-maximum suppression: strongest local proposals first, then
    // restore spatial order for stable report serialization.
    local_maxima.sort_by(|left, right| {
        right.score.total_cmp(&left.score)
            .then_with(|| left.boundary_nm.total_cmp(&right.boundary_nm))
    });
    let mut accepted = Vec::<Candidate>::new();
    for candidate in local_maxima {
        if accepted.iter().all(|prior| {
            (prior.boundary_nm - candidate.boundary_nm).abs() >= input.detector.min_separation_nm
        }) {
            accepted.push(candidate);
        }
    }
    accepted.sort_by(|left, right| left.boundary_nm.total_cmp(&right.boundary_nm));
    accepted
}

fn detect(input: &Input, multi_feature: bool) -> Vec<Candidate> {
    let mut proposals = input.tracks.iter()
        .flat_map(|track| detect_track(track, input, multi_feature))
        .collect::<Vec<_>>();
    proposals.sort_by(|left, right| {
        left.track_id.cmp(&right.track_id)
            .then_with(|| left.boundary_nm.total_cmp(&right.boundary_nm))
    });
    proposals
}

fn validate_truth(truth: &Truth, input_sha256: &str, input: &Input) -> Result<(), String> {
    if truth.schema_version != 1 || truth.input_sha256 != input_sha256 {
        return Err("truth must declare schema_version 1 and bind the exact detector input SHA-256".into());
    }
    let expected = input.tracks.iter().map(|track| track.track_id.as_str()).collect::<BTreeSet<_>>();
    let mut seen = BTreeSet::new();
    for track in &truth.tracks {
        if !expected.contains(track.track_id.as_str()) || !seen.insert(track.track_id.as_str()) {
            return Err(format!("truth contains unknown or duplicate track {}", track.track_id));
        }
        if track.boundaries_nm.iter().any(|value| !value.is_finite()) {
            return Err(format!("truth track {} contains non-finite boundary", track.track_id));
        }
        let mut boundaries = track.boundaries_nm.clone();
        boundaries.sort_by(f64::total_cmp);
        if boundaries.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(format!("truth track {} contains duplicate boundary positions", track.track_id));
        }
        let input_track = input.tracks.iter().find(|candidate| candidate.track_id == track.track_id)
            .ok_or_else(|| format!("missing detector track {}", track.track_id))?;
        let min = input_track.samples.first().unwrap().distance_nm;
        let max = input_track.samples.last().unwrap().distance_nm;
        if boundaries.iter().any(|value| *value < min || *value > max) {
            return Err(format!("truth track {} has a boundary outside its sample extent", track.track_id));
        }
    }
    if seen.len() != expected.len() {
        return Err("truth must include every detector track, including negative controls".into());
    }
    Ok(())
}

fn quantile(sorted_values: &[f64], quantile: f64) -> Option<f64> {
    if sorted_values.is_empty() {
        return None;
    }
    let position = quantile * (sorted_values.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    let fraction = position - lower as f64;
    Some(sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction)
}

fn evaluate(
    proposals: &[Candidate],
    truth: &Truth,
    input: &Input,
    tolerance_nm: f64,
) -> Evaluation {
    let mut truth_rows = truth.tracks.iter()
        .flat_map(|track| track.boundaries_nm.iter().map(move |boundary| (track.track_id.clone(), *boundary)))
        .collect::<Vec<_>>();
    truth_rows.sort_by(|left, right| left.0.cmp(&right.0).then_with(|| left.1.total_cmp(&right.1)));

    let mut pairs = Vec::<(f64, usize, usize)>::new();
    for (proposal_index, proposal) in proposals.iter().enumerate() {
        for (truth_index, (track_id, boundary)) in truth_rows.iter().enumerate() {
            if proposal.track_id == *track_id {
                let error = (proposal.boundary_nm - boundary).abs();
                if error <= tolerance_nm {
                    pairs.push((error, proposal_index, truth_index));
                }
            }
        }
    }
    pairs.sort_by(|left, right| left.0.total_cmp(&right.0));
    let mut matched_proposals = BTreeSet::new();
    let mut matched_truth = BTreeSet::new();
    let mut localization_errors = Vec::new();
    for (error, proposal_index, truth_index) in pairs {
        if !matched_proposals.contains(&proposal_index) && !matched_truth.contains(&truth_index) {
            matched_proposals.insert(proposal_index);
            matched_truth.insert(truth_index);
            localization_errors.push(error);
        }
    }
    localization_errors.sort_by(f64::total_cmp);
    let tp = localization_errors.len();
    let fp = proposals.len().saturating_sub(tp);
    let fn_count = truth_rows.len().saturating_sub(tp);
    let precision = if proposals.is_empty() { None } else { Some(tp as f64 / proposals.len() as f64) };
    let recall = if truth_rows.is_empty() { None } else { Some(tp as f64 / truth_rows.len() as f64) };
    let f1 = precision.zip(recall).and_then(|(p, r)| {
        if p + r == 0.0 { None } else { Some(2.0 * p * r / (p + r)) }
    });
    let length_nm = input.tracks.iter()
        .map(|track| track.samples.last().unwrap().distance_nm - track.samples.first().unwrap().distance_nm)
        .sum::<f64>();
    Evaluation {
        tolerance_nm,
        truth_boundary_count: truth_rows.len(),
        true_positive_count: tp,
        false_positive_count: fp,
        false_negative_count: fn_count,
        precision,
        recall,
        f1,
        median_absolute_localization_error_nm: quantile(&localization_errors, 0.5),
        p90_absolute_localization_error_nm: quantile(&localization_errors, 0.9),
        false_proposals_per_cable_mm: if length_nm <= 0.0 { None } else {
            Some(fp as f64 / (length_nm / 1_000_000.0))
        },
    }
}

fn run(input: &Input, input_bytes: &[u8], truth_bytes: Option<&[u8]>, tolerance_nm: f64) -> Result<Report, String> {
    validate_input(input)?;
    if !tolerance_nm.is_finite() || tolerance_nm <= 0.0 {
        return Err("tolerance_nm must be finite and positive".into());
    }
    let input_sha = sha256(input_bytes);
    // Detection is fully completed before evaluator-only truth is parsed.
    let radius_candidates = detect(input, false);
    let multi_candidates = detect(input, true);
    let truth = truth_bytes.map(|bytes| -> Result<Truth, String> {
        let value: Truth = serde_json::from_slice(bytes)
            .map_err(|error| format!("invalid evaluator-only truth JSON: {error}"))?;
        validate_truth(&value, &input_sha, input)?;
        Ok(value)
    }).transpose()?;
    let truth_sha = truth_bytes.map(sha256);
    let radius_evaluation = truth.as_ref().map(|value| evaluate(&radius_candidates, value, input, tolerance_nm));
    let multi_evaluation = truth.as_ref().map(|value| evaluate(&multi_candidates, value, input, tolerance_nm));
    let total_samples = input.tracks.iter().map(|track| track.samples.len()).sum();
    let total_length = input.tracks.iter()
        .map(|track| track.samples.last().unwrap().distance_nm - track.samples.first().unwrap().distance_nm)
        .sum::<f64>();

    Ok(Report {
        schema_version: 1,
        tool: "connectome-colinear-merge-audit",
        input_sha256: input_sha,
        truth_sha256: truth_sha,
        dataset_id: input.dataset_id.clone(),
        dataset_revision_commit_sha: input.dataset_revision_commit_sha.clone(),
        total_track_count: input.tracks.len(),
        total_sample_count: total_samples,
        total_cable_length_nm: total_length,
        method_reports: vec![
            MethodReport {
                method: "no_proposal",
                candidate_count: 0,
                candidates: Vec::new(),
                evaluation: truth.as_ref().map(|value| evaluate(&[], value, input, tolerance_nm)),
            },
            MethodReport {
                method: "radius_only_piecewise_linear",
                candidate_count: radius_candidates.len(),
                candidates: radius_candidates,
                evaluation: radius_evaluation,
            },
            MethodReport {
                method: "multifeature_piecewise_linear",
                candidate_count: multi_candidates.len(),
                candidates: multi_candidates,
                evaluation: multi_evaluation,
            },
        ],
        status: if truth.is_some() { "SYNTHETIC_OR_HELD_OUT_EVALUATED_NOT_PROMOTED" } else { "PROPOSALS_ONLY_NOT_EVALUATED" },
        information_flow: "Detector consumes only the exact profile input. Ground-truth boundary labels are parsed only after both candidate lists have been generated, are byte-bound to the exact input SHA-256, and are used only for evaluation.",
        scope_note: "Synthetic ordered 1D morphology profiles only. This does not inspect EM images, meshes, or skeleton geometry; does not identify a biological false merge; and does not justify automatic edits. A positive score is a human-review proposal only.",
    })
}

fn main() -> ExitCode {
    let args = Args::parse();
    let input_bytes = match fs::read(&args.input) {
        Ok(bytes) => bytes,
        Err(error) => {
            eprintln!("could not read {}: {error}", args.input.display());
            return ExitCode::from(2);
        }
    };
    let input = match serde_json::from_slice::<Input>(&input_bytes) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("invalid detector input JSON: {error}");
            return ExitCode::from(2);
        }
    };
    let truth_bytes = match args.truth {
        Some(path) => match fs::read(&path) {
            Ok(bytes) => Some(bytes),
            Err(error) => {
                eprintln!("could not read evaluator truth {}: {error}", path.display());
                return ExitCode::from(2);
            }
        },
        None => None,
    };
    let report = match run(&input, &input_bytes, truth_bytes.as_deref(), args.tolerance_nm) {
        Ok(report) => report,
        Err(error) => {
            eprintln!("co-linear merge audit failed: {error}");
            return ExitCode::from(1);
        }
    };
    let output = match serde_json::to_string_pretty(&report) {
        Ok(output) => output,
        Err(error) => {
            eprintln!("could not serialize report: {error}");
            return ExitCode::from(2);
        }
    };
    if let Some(path) = args.output {
        if let Err(error) = fs::write(&path, output) {
            eprintln!("could not write {}: {error}", path.display());
            return ExitCode::from(2);
        }
    } else {
        println!("{output}");
    }
    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_input(tracks: Vec<Track>) -> Input {
        Input {
            schema_version: 1,
            dataset_id: "synthetic-colinear-connectome".into(),
            dataset_revision_commit_sha: "a".repeat(40),
            detector: DetectorConfig {
                half_window_samples: 4,
                score_threshold: 2.5,
                min_separation_nm: 400.0,
            },
            noise: FeatureNoise {
                radius_nm: 5.0,
                tangent_degrees: 5.0,
                membrane_integrity: 0.1,
                synapse_density: 0.2,
            },
            tracks,
        }
    }

    fn make_track(track_id: &str, feature_fn: impl Fn(f64) -> (f64, f64, f64, f64)) -> Track {
        let samples = (0..=40).map(|index| {
            let distance_nm = index as f64 * 100.0;
            let (radius_nm, tangent_degrees, membrane_integrity, synapse_density) = feature_fn(distance_nm);
            Sample { distance_nm, radius_nm, tangent_degrees, membrane_integrity, synapse_density }
        }).collect();
        Track { track_id: track_id.into(), samples }
    }

    #[test]
    fn checked_in_fixture_demonstrates_multifeature_recall_tradeoff_and_false_positive_control() {
        let input_bytes = include_bytes!("../../../../docs/neuroscience/examples/CONNECTOME_COLINEAR_PROFILE_INPUT.example.json");
        let truth_bytes = include_bytes!("../../../../docs/neuroscience/examples/CONNECTOME_COLINEAR_BOUNDARY_TRUTH.example.json");
        let input: Input = serde_json::from_slice(input_bytes).expect("profile fixture should parse");
        let report = run(&input, input_bytes, Some(truth_bytes), 500.0).expect("boundaries should evaluate");
        assert_eq!(report.input_sha256, "e1cef74eb7d37866320e6d8ad93d1d030c4c6d99fd411e3dc8cc277ad83b82fb");
        let no_proposal = report.method_reports[0].evaluation.as_ref().unwrap();
        let radius = report.method_reports[1].evaluation.as_ref().unwrap();
        let multi = report.method_reports[2].evaluation.as_ref().unwrap();
        assert_eq!(no_proposal.true_positive_count, 0);
        assert_eq!(no_proposal.false_negative_count, 2);
        assert_eq!(radius.true_positive_count, 1);
        assert_eq!(radius.false_positive_count, 1);
        assert_eq!(radius.false_negative_count, 1);
        assert_eq!(multi.true_positive_count, 2);
        assert_eq!(multi.false_negative_count, 0);
        assert_eq!(multi.false_positive_count, 7);
    }

    #[test]
    fn radius_step_inside_one_non_branching_segment_is_localized() {
        let track = make_track("hidden-merge", |x| {
            if x < 2000.0 { (80.0, 4.0, 0.95, 0.2) } else { (48.0, 5.0, 0.80, 0.7) }
        });
        let input = base_input(vec![track]);
        let proposals = detect(&input, false);
        assert!(!proposals.is_empty());
        assert!((proposals[0].boundary_nm - 2000.0).abs() <= 100.0);
    }

    #[test]
    fn multifeature_method_detects_a_boundary_radius_only_misses() {
        let track = make_track("subtle-merge", |x| {
            if x < 2000.0 { (60.0, 4.0, 0.95, 0.1) } else { (60.0, 28.0, 0.55, 0.8) }
        });
        let input = base_input(vec![track]);
        assert!(detect(&input, false).is_empty());
        assert!(!detect(&input, true).is_empty());
    }

    #[test]
    fn exact_linear_taper_is_a_negative_control() {
        let track = make_track("natural-taper", |x| {
            (90.0 - x * 0.01, 2.0 + x * 0.001, 0.95, 0.2 + x * 0.0001)
        });
        let input = base_input(vec![track]);
        assert!(detect(&input, true).is_empty());
    }

    #[test]
    fn truth_is_bound_to_exact_detector_input_and_complete_track_frame() {
        let input = base_input(vec![
            make_track("one", |_| (60.0, 4.0, 0.9, 0.2)),
            make_track("two", |_| (50.0, 5.0, 0.8, 0.3)),
        ]);
        let bytes = serde_json::to_vec(&input).unwrap();
        let truth = Truth {
            schema_version: 1,
            input_sha256: sha256(&bytes),
            tracks: vec![TruthTrack { track_id: "one".into(), boundaries_nm: vec![] }],
        };
        assert!(validate_truth(&truth, &sha256(&bytes), &input).unwrap_err().contains("every detector track"));
    }

    #[test]
    fn evaluator_reports_recall_precision_and_cable_normalized_false_proposals() {
        let tracks = vec![make_track("one", |x| {
            if x < 2000.0 { (80.0, 4.0, 0.95, 0.2) } else { (48.0, 5.0, 0.80, 0.7) }
        })];
        let input = base_input(tracks);
        let input_bytes = serde_json::to_vec(&input).unwrap();
        let truth = Truth {
            schema_version: 1,
            input_sha256: sha256(&input_bytes),
            tracks: vec![TruthTrack { track_id: "one".into(), boundaries_nm: vec![2000.0] }],
        };
        let truth_bytes = serde_json::to_vec(&truth).unwrap();
        let report = run(&input, &input_bytes, Some(&truth_bytes), 500.0).unwrap();
        assert_eq!(report.status, "SYNTHETIC_OR_HELD_OUT_EVALUATED_NOT_PROMOTED");
        assert_eq!(report.method_reports[1].evaluation.as_ref().unwrap().true_positive_count, 1);
    }

    #[test]
    fn malformed_track_positions_and_feature_domains_fail_closed() {
        let mut input = base_input(vec![make_track("one", |_| (60.0, 0.0, 0.9, 0.2))]);
        input.tracks[0].samples[2].distance_nm = input.tracks[0].samples[1].distance_nm;
        assert!(validate_input(&input).unwrap_err().contains("strictly increasing"));
        let mut input = base_input(vec![make_track("one", |_| (60.0, 0.0, 1.2, 0.2))]);
        assert!(validate_input(&input).unwrap_err().contains("out-of-domain"));
    }

    #[test]
    fn score_is_finite_for_constant_feature_track() {
        let track = make_track("constant", |_| (60.0, 0.0, 0.9, 0.2));
        let input = base_input(vec![track]);
        assert!(detect(&input, true).iter().all(|candidate| candidate.score.is_finite()));
    }
}
