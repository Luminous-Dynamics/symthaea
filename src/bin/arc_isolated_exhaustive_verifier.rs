// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Independent exhaustive-reference verifier for the process-isolated ARC lane.
//!
//! Reads only target-stripped solver-view bytes plus the sealed policy report. It does not need
//! real expected outputs. It recomputes the complete frozen RQ-003 grammar and requires every
//! published post-seal exhaustive-reference field to match.

use serde::Serialize;
use serde_json::{Map, Value};
use std::collections::{BTreeMap, HashSet};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use symthaea::hdc::grid_encoder::GridEncoder;

type Grid = Vec<Vec<u8>>;

const SCHEMA_VERSION: u32 = 1;
const DOMAIN: &str = "symthaea/reasoning/arc-isolated-exhaustive-verifier/v1";
const GRID_DOMAIN: &[u8] = b"symthaea/reasoning/arc-grid/v1";

#[derive(Debug, Clone)]
struct GridPair {
    input: Grid,
    output: Grid,
}

#[derive(Debug, Clone)]
struct SolverTask {
    train: Vec<GridPair>,
    test_inputs: Vec<Grid>,
}

#[derive(Debug, Clone, Copy)]
enum Geometry {
    Identity,
    ReflectX,
    ReflectY,
    Rotate90,
    Rotate180,
    Rotate270,
    Translate { dx: i32, dy: i32 },
}

#[derive(Debug, Clone, Copy)]
enum ColorOp {
    None,
    Replace { from: u8, to: u8 },
}

#[derive(Debug, Clone, Copy)]
struct CandidateTransform {
    geometry: Geometry,
    color: ColorOp,
}

#[derive(Debug, Clone)]
struct ExhaustiveResult {
    prediction_digest: Option<String>,
    candidates_evaluated: usize,
    training_pair_checks: usize,
}

#[derive(Debug, Serialize)]
struct VerifiedProblem {
    problem_id: String,
    exhaustive_asserted: bool,
    exhaustive_prediction_digest: Option<String>,
    exhaustive_candidates_evaluated: usize,
    exhaustive_training_pair_checks: usize,
}

#[derive(Debug, Serialize)]
struct VerificationReport {
    schema_version: u32,
    domain: String,
    problems: Vec<VerifiedProblem>,
    commitment: String,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("ARC isolated exhaustive verification failed: {err}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let solver_root = PathBuf::from(required_env("SYMTHAEA_ARC_SOLVER_VIEW_ROOT")?);
    let policy_report_path = PathBuf::from(required_env("SYMTHAEA_ARC_POLICY_RESULTS_PATH")?);
    let output_path = PathBuf::from(required_env("SYMTHAEA_ARC_EXHAUSTIVE_VERIFY_PATH")?);

    let tasks = load_solver_tasks(&solver_root)?;
    let report_bytes = fs::read(&policy_report_path)
        .map_err(|err| format!("failed to read {}: {err}", policy_report_path.display()))?;
    let report: Value = serde_json::from_slice(&report_bytes)
        .map_err(|err| format!("invalid policy report JSON: {err}"))?;
    let report = object(&report, "policy report")?;
    let rows = require_array(report, "tasks", "policy report")?;
    let candidates = canonical_candidates();

    let mut by_problem: BTreeMap<String, (&Map<String, Value>, Option<&Map<String, Value>>)> =
        BTreeMap::new();
    for row in rows {
        let row = object(row, "policy task result")?;
        let problem_id = require_str(row, "problem_id", "policy task result")?.to_string();
        let policy = require_str(row, "policy_id", "policy task result")?;
        let entry = by_problem.entry(problem_id).or_insert((row, None));
        if std::ptr::eq(entry.0, row) {
            if policy != "canonical-order-v1" {
                return Err("first policy row for a problem must be canonical-order-v1".into());
            }
        } else {
            if entry.1.is_some() || policy != "uniform-random-without-replacement-v1" {
                return Err("policy report must contain exactly canonical then uniform-random per problem".into());
            }
            entry.1 = Some(row);
        }
    }

    let mut verified = Vec::with_capacity(by_problem.len());
    for (problem_id, (canonical, random)) in by_problem {
        let random = random.ok_or_else(|| format!("missing random policy row for {problem_id}"))?;
        let (stem, test_index) = parse_problem_id(&problem_id)?;
        let task = tasks
            .get(stem)
            .ok_or_else(|| format!("unknown solver-view task {stem}"))?;
        let input = task
            .test_inputs
            .get(test_index)
            .ok_or_else(|| format!("test index out of range for {problem_id}"))?;
        let full = exhaustive_reference(&task.train, input, &candidates);
        validate_published_exhaustive(canonical, &problem_id, &full)?;
        validate_published_exhaustive(random, &problem_id, &full)?;
        verified.push(VerifiedProblem {
            problem_id,
            exhaustive_asserted: full.prediction_digest.is_some(),
            exhaustive_prediction_digest: full.prediction_digest,
            exhaustive_candidates_evaluated: full.candidates_evaluated,
            exhaustive_training_pair_checks: full.training_pair_checks,
        });
    }

    let mut output = VerificationReport {
        schema_version: SCHEMA_VERSION,
        domain: DOMAIN.into(),
        problems: verified,
        commitment: String::new(),
    };
    output.commitment = report_commitment(&output);
    write_json(&output_path, &output)?;
    println!("verified {} exhaustive ARC decisions", output.problems.len());
    println!("commitment: {}", output.commitment);
    Ok(())
}

fn load_solver_tasks(root: &Path) -> Result<BTreeMap<String, SolverTask>, String> {
    let split = root.join("training");
    let mut paths = fs::read_dir(&split)
        .map_err(|err| format!("failed to list {}: {err}", split.display()))?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| format!("failed to enumerate solver-view tasks: {err}"))?;
    paths.sort();
    if paths.is_empty() {
        return Err("solver-view training directory is empty".into());
    }
    let mut tasks = BTreeMap::new();
    for path in paths {
        if path.extension().and_then(|value| value.to_str()) != Some("json") {
            return Err(format!("non-JSON solver-view entry: {}", path.display()));
        }
        let bytes = fs::read(&path)
            .map_err(|err| format!("failed to read {}: {err}", path.display()))?;
        let value: Value = serde_json::from_slice(&bytes)
            .map_err(|err| format!("invalid solver-view JSON {}: {err}", path.display()))?;
        let object = object(&value, "solver-view task")?;
        let train = require_array(object, "train", "solver-view task")?
            .iter()
            .map(parse_pair)
            .collect::<Result<Vec<_>, _>>()?;
        let test = require_array(object, "test", "solver-view task")?;
        let mut test_inputs = Vec::with_capacity(test.len());
        for pair in test {
            let pair = object(pair, "solver test pair")?;
            test_inputs.push(parse_grid(
                pair.get("input")
                    .ok_or_else(|| "solver test pair missing input".to_string())?,
            )?);
            let sentinel = parse_grid(
                pair.get("output")
                    .ok_or_else(|| "solver test pair missing sentinel".to_string())?,
            )?;
            if sentinel != vec![vec![0]] {
                return Err("solver-view task does not contain fixed public sentinel".into());
            }
        }
        let stem = path
            .file_stem()
            .and_then(|value| value.to_str())
            .ok_or_else(|| format!("solver-view filename is not UTF-8: {}", path.display()))?
            .to_string();
        if tasks.insert(stem.clone(), SolverTask { train, test_inputs }).is_some() {
            return Err(format!("duplicate solver-view task stem {stem}"));
        }
    }
    Ok(tasks)
}

fn exhaustive_reference(
    train: &[GridPair],
    test_input: &Grid,
    candidates: &[CandidateTransform],
) -> ExhaustiveResult {
    let mut predictions = Vec::<String>::new();
    let mut pair_checks = 0usize;
    for candidate in candidates {
        let mut consistent = true;
        for pair in train {
            pair_checks = pair_checks.saturating_add(1);
            if apply_transform(&pair.input, *candidate) != pair.output {
                consistent = false;
                break;
            }
        }
        if consistent {
            let digest = grid_digest(&apply_transform(test_input, *candidate));
            if !predictions.contains(&digest) {
                predictions.push(digest);
            }
        }
    }
    ExhaustiveResult {
        prediction_digest: (predictions.len() == 1).then(|| predictions.remove(0)),
        candidates_evaluated: candidates.len(),
        training_pair_checks: pair_checks,
    }
}

fn validate_published_exhaustive(
    row: &Map<String, Value>,
    problem_id: &str,
    full: &ExhaustiveResult,
) -> Result<(), String> {
    let asserted = require_bool(row, "exhaustive_asserted", "policy task result")?;
    let digest = optional_string(row, "exhaustive_prediction_digest", "policy task result")?;
    if asserted != full.prediction_digest.is_some() || digest != full.prediction_digest {
        return Err(format!("exhaustive decision mismatch for {problem_id}"));
    }
    if require_usize(row, "exhaustive_candidates_evaluated", "policy task result")?
        != full.candidates_evaluated
        || require_usize(row, "exhaustive_training_pair_checks", "policy task result")?
            != full.training_pair_checks
    {
        return Err(format!("exhaustive resource accounting mismatch for {problem_id}"));
    }
    Ok(())
}

fn canonical_candidates() -> Vec<CandidateTransform> {
    let mut geometry = vec![
        Geometry::Identity,
        Geometry::ReflectX,
        Geometry::ReflectY,
        Geometry::Rotate90,
        Geometry::Rotate180,
        Geometry::Rotate270,
    ];
    for dy in -3..=3 {
        for dx in -3..=3 {
            if dx != 0 || dy != 0 {
                geometry.push(Geometry::Translate { dx, dy });
            }
        }
    }
    let mut colors = vec![ColorOp::None];
    for from in 0..=9 {
        for to in 0..=9 {
            if from != to {
                colors.push(ColorOp::Replace { from, to });
            }
        }
    }
    let mut out = Vec::with_capacity(geometry.len() * colors.len());
    for geometry in geometry {
        for color in &colors {
            out.push(CandidateTransform {
                geometry,
                color: *color,
            });
        }
    }
    out
}

fn apply_transform(grid: &Grid, candidate: CandidateTransform) -> Grid {
    let geometry = match candidate.geometry {
        Geometry::Identity => grid.clone(),
        Geometry::ReflectX => GridEncoder::reflect_x(grid),
        Geometry::ReflectY => GridEncoder::reflect_y(grid),
        Geometry::Rotate90 => GridEncoder::rotate_90(grid),
        Geometry::Rotate180 => GridEncoder::rotate_90(&GridEncoder::rotate_90(grid)),
        Geometry::Rotate270 => {
            GridEncoder::rotate_90(&GridEncoder::rotate_90(&GridEncoder::rotate_90(grid)))
        }
        Geometry::Translate { dx, dy } => GridEncoder::translate_grid(grid, dx, dy, 0),
    };
    match candidate.color {
        ColorOp::None => geometry,
        ColorOp::Replace { from, to } => GridEncoder::color_replace(&geometry, from, to),
    }
}

fn grid_digest(grid: &Grid) -> String {
    let mut hasher = blake3::Hasher::new();
    hash_bytes(&mut hasher, GRID_DOMAIN);
    hash_u64(&mut hasher, grid.len() as u64);
    for row in grid {
        hash_u64(&mut hasher, row.len() as u64);
        for cell in row {
            hasher.update(&[*cell]);
        }
    }
    hasher.finalize().to_hex().to_string()
}

fn parse_pair(value: &Value) -> Result<GridPair, String> {
    let object = object(value, "training pair")?;
    Ok(GridPair {
        input: parse_grid(
            object
                .get("input")
                .ok_or_else(|| "training pair missing input".to_string())?,
        )?,
        output: parse_grid(
            object
                .get("output")
                .ok_or_else(|| "training pair missing output".to_string())?,
        )?,
    })
}

fn parse_grid(value: &Value) -> Result<Grid, String> {
    let rows = value
        .as_array()
        .ok_or_else(|| "grid must be an array".to_string())?;
    if rows.is_empty() {
        return Err("grid must not be empty".into());
    }
    let mut grid = Vec::with_capacity(rows.len());
    let mut width = None;
    for row in rows {
        let cells = row
            .as_array()
            .ok_or_else(|| "grid row must be an array".to_string())?;
        if cells.is_empty() {
            return Err("grid row must not be empty".into());
        }
        if let Some(expected) = width {
            if cells.len() != expected {
                return Err("grid must be rectangular".into());
            }
        } else {
            width = Some(cells.len());
        }
        let mut parsed = Vec::with_capacity(cells.len());
        for cell in cells {
            let value = cell
                .as_u64()
                .ok_or_else(|| "grid cell must be unsigned integer".to_string())?;
            if value > 9 {
                return Err("grid color must be within 0..=9".into());
            }
            parsed.push(value as u8);
        }
        grid.push(parsed);
    }
    Ok(grid)
}

fn parse_problem_id(value: &str) -> Result<(&str, usize), String> {
    let (stem, index) = value
        .rsplit_once("#test-")
        .ok_or_else(|| format!("invalid ARC problem id {value}"))?;
    let index = index
        .parse::<usize>()
        .map_err(|err| format!("invalid test index in {value}: {err}"))?;
    Ok((stem, index))
}

fn report_commitment(report: &VerificationReport) -> String {
    let mut hasher = blake3::Hasher::new();
    hash_str(&mut hasher, DOMAIN);
    hash_u64(&mut hasher, u64::from(report.schema_version));
    hash_u64(&mut hasher, report.problems.len() as u64);
    for problem in &report.problems {
        hash_str(&mut hasher, &problem.problem_id);
        hasher.update(&[u8::from(problem.exhaustive_asserted)]);
        match &problem.exhaustive_prediction_digest {
            Some(digest) => {
                hasher.update(&[1]);
                hash_str(&mut hasher, digest);
            }
            None => hasher.update(&[0]),
        }
        hash_u64(&mut hasher, problem.exhaustive_candidates_evaluated as u64);
        hash_u64(&mut hasher, problem.exhaustive_training_pair_checks as u64);
    }
    hasher.finalize().to_hex().to_string()
}

fn write_json(path: &Path, value: &impl Serialize) -> Result<(), String> {
    if let Some(parent) = path.parent().filter(|parent| !parent.as_os_str().is_empty()) {
        fs::create_dir_all(parent)
            .map_err(|err| format!("failed to create {}: {err}", parent.display()))?;
    }
    fs::write(
        path,
        serde_json::to_string_pretty(value)
            .map_err(|err| format!("failed to encode {}: {err}", path.display()))?,
    )
    .map_err(|err| format!("failed to write {}: {err}", path.display()))
}

fn object<'a>(value: &'a Value, label: &str) -> Result<&'a Map<String, Value>, String> {
    value
        .as_object()
        .ok_or_else(|| format!("{label} must be an object"))
}

fn require_array<'a>(object: &'a Map<String, Value>, key: &str, label: &str) -> Result<&'a Vec<Value>, String> {
    object
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{label} field {key} must be an array"))
}

fn require_str<'a>(object: &'a Map<String, Value>, key: &str, label: &str) -> Result<&'a str, String> {
    object
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{label} field {key} must be a string"))
}

fn require_usize(object: &Map<String, Value>, key: &str, label: &str) -> Result<usize, String> {
    let value = object
        .get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("{label} field {key} must be unsigned integer"))?;
    usize::try_from(value).map_err(|_| format!("{label} field {key} does not fit usize"))
}

fn require_bool(object: &Map<String, Value>, key: &str, label: &str) -> Result<bool, String> {
    object
        .get(key)
        .and_then(Value::as_bool)
        .ok_or_else(|| format!("{label} field {key} must be bool"))
}

fn optional_string(object: &Map<String, Value>, key: &str, label: &str) -> Result<Option<String>, String> {
    match object.get(key) {
        Some(Value::Null) => Ok(None),
        Some(Value::String(value)) => Ok(Some(value.to_ascii_lowercase())),
        Some(_) => Err(format!("{label} field {key} must be string or null")),
        None => Err(format!("{label} missing field {key}")),
    }
}

fn required_env(name: &str) -> Result<String, String> {
    match env::var(name) {
        Ok(value) if !value.trim().is_empty() => Ok(value.trim().to_string()),
        Ok(_) => Err(format!("required environment variable {name} is empty")),
        Err(err) => Err(format!("required environment variable {name} is missing: {err}")),
    }
}

fn hash_str(hasher: &mut blake3::Hasher, value: &str) {
    hash_bytes(hasher, value.as_bytes());
}

fn hash_bytes(hasher: &mut blake3::Hasher, value: &[u8]) {
    hash_u64(hasher, value.len() as u64);
    hasher.update(value);
}

fn hash_u64(hasher: &mut blake3::Hasher, value: u64) {
    hasher.update(&value.to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grammar_size_matches_frozen_rq003() {
        assert_eq!(canonical_candidates().len(), 4_914);
    }

    #[test]
    fn ambiguous_predictions_abstain() {
        let train = vec![GridPair {
            input: vec![vec![0, 0]],
            output: vec![vec![0, 0]],
        }];
        let input = vec![vec![1, 2]];
        let candidates = vec![
            CandidateTransform {
                geometry: Geometry::Identity,
                color: ColorOp::None,
            },
            CandidateTransform {
                geometry: Geometry::ReflectX,
                color: ColorOp::None,
            },
        ];
        let result = exhaustive_reference(&train, &input, &candidates);
        assert!(result.prediction_digest.is_none());
    }
}