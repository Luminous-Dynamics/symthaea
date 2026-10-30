//! HARDSEC-010 streaming graph edge-accounting and reference-path retention auditor.
//!
//! This utility compares a source edge stream with a represented/partitioned
//! edge stream and an explicit exclusion ledger. Every source edge must be
//! retained exactly or excluded with a reason. Reference paths are verified
//! against source edges and classified as retained or cut-and-reported.
//!
//! It deliberately does not parse Verilog, infer missing edges, or classify
//! hardware security. It keeps O(reference-path edges) metadata in memory and
//! streams the full source, represented and exclusion edge files.
//!
//! Input edge TSV (exact header):
//!   edge_id<TAB>src<TAB>dst<TAB>relation<TAB>partition_id
//! Exclusion TSV:
//!   edge_id<TAB>reason
//! Reference path TSV:
//!   path_id<TAB>step_index<TAB>edge_id<TAB>src<TAB>dst
//!
//! Example:
//!   cargo run --example hardsec_graph_path_retention_audit --features hardsec-graph-audit -- \
//!     --source-edges source.tsv --represented-edges reduced.tsv \
//!     --exclusions exclusions.tsv --reference-paths paths.tsv \
//!     --receipt /path/outside/inputs/graph-retention.json

use serde_json::json;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

const EDGE_HEADER: [&str; 5] = ["edge_id", "src", "dst", "relation", "partition_id"];
const EXCLUSION_HEADER: [&str; 2] = ["edge_id", "reason"];
const PATH_HEADER: [&str; 5] = ["path_id", "step_index", "edge_id", "src", "dst"];
const CLAIM_SCOPE: &str =
    "exact_edge_accounting_and_declared_reference_path_retention_only";

#[derive(Debug, Clone)]
struct Args {
    source_edges: PathBuf,
    represented_edges: PathBuf,
    exclusions: PathBuf,
    reference_paths: PathBuf,
    receipt: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Edge {
    edge_id: String,
    src: String,
    dst: String,
    relation: String,
    partition_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Exclusion {
    edge_id: String,
    reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PathStep {
    edge_id: String,
    src: String,
    dst: String,
    source_verified: bool,
    represented_partition: Option<String>,
    exclusion_reason: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ReferencePath {
    path_id: String,
    steps: Vec<PathStep>,
}

#[derive(Debug, Clone)]
struct PathSummary {
    path_id: String,
    source_node: String,
    observation_node: String,
    edge_count: usize,
    retained_edge_count: usize,
    excluded_edge_count: usize,
    partition_crossings: usize,
    status: &'static str,
}

#[derive(Debug, Clone)]
struct AuditSummary {
    source_edge_count: u64,
    represented_edge_count: u64,
    excluded_edge_count: u64,
    paths: Vec<PathSummary>,
    retained_path_count: usize,
    cut_path_count: usize,
    total_reference_path_edges: usize,
    retained_reference_path_edges: usize,
}

struct EdgeReader<R: BufRead> {
    reader: R,
    label: &'static str,
    line_number: usize,
    previous_id: Option<String>,
}

struct ExclusionReader<R: BufRead> {
    reader: R,
    line_number: usize,
    previous_id: Option<String>,
}

fn parse_args() -> Result<Option<Args>, String> {
    let mut source_edges = None;
    let mut represented_edges = None;
    let mut exclusions = None;
    let mut reference_paths = None;
    let mut receipt = None;
    let mut args = env::args().skip(1);

    while let Some(flag) = args.next() {
        if flag == "--help" || flag == "-h" {
            return Ok(None);
        }
        let value = args
            .next()
            .ok_or_else(|| format!("missing value for {flag}"))?;
        match flag.as_str() {
            "--source-edges" => source_edges = Some(PathBuf::from(value)),
            "--represented-edges" => represented_edges = Some(PathBuf::from(value)),
            "--exclusions" => exclusions = Some(PathBuf::from(value)),
            "--reference-paths" => reference_paths = Some(PathBuf::from(value)),
            "--receipt" => receipt = Some(PathBuf::from(value)),
            _ => return Err(format!("unknown argument: {flag}")),
        }
    }

    Ok(Some(Args {
        source_edges: source_edges.ok_or("--source-edges is required")?,
        represented_edges: represented_edges.ok_or("--represented-edges is required")?,
        exclusions: exclusions.ok_or("--exclusions is required")?,
        reference_paths: reference_paths.ok_or("--reference-paths is required")?,
        receipt: receipt.ok_or("--receipt is required")?,
    }))
}

fn nonempty(value: &str, field: &str, line: usize) -> Result<String, String> {
    if value.is_empty() || value.trim() != value || value.contains('\0') {
        return Err(format!("{field} must be non-empty and have no surrounding whitespace at line {line}"));
    }
    Ok(value.to_owned())
}

fn parse_header<R: BufRead>(
    reader: &mut R,
    expected: &[&str],
    label: &str,
) -> Result<(), String> {
    let mut line = String::new();
    let count = reader
        .read_line(&mut line)
        .map_err(|error| format!("read {label} header: {error}"))?;
    if count == 0 {
        return Err(format!("{label} file is empty"));
    }
    let header = line.trim_end_matches(['\n', '\r']);
    if header.split('\t').collect::<Vec<_>>() != expected {
        return Err(format!(
            "{label} header must exactly match {}",
            expected.join("\t")
        ));
    }
    Ok(())
}

impl<R: BufRead> EdgeReader<R> {
    fn new(mut reader: R, label: &'static str) -> Result<Self, String> {
        parse_header(&mut reader, &EDGE_HEADER, label)?;
        Ok(Self {
            reader,
            label,
            line_number: 1,
            previous_id: None,
        })
    }

    fn next(&mut self) -> Result<Option<Edge>, String> {
        let mut line = String::new();
        let count = self
            .reader
            .read_line(&mut line)
            .map_err(|error| format!("read {} edge row: {error}", self.label))?;
        if count == 0 {
            return Ok(None);
        }
        self.line_number += 1;
        let line = line.trim_end_matches(['\n', '\r']);
        let fields = line.split('\t').collect::<Vec<_>>();
        if fields.len() != EDGE_HEADER.len() {
            return Err(format!(
                "{} line {}: expected {} fields, got {}",
                self.label,
                self.line_number,
                EDGE_HEADER.len(),
                fields.len()
            ));
        }
        let edge = Edge {
            edge_id: nonempty(fields[0], "edge_id", self.line_number)?,
            src: nonempty(fields[1], "src", self.line_number)?,
            dst: nonempty(fields[2], "dst", self.line_number)?,
            relation: nonempty(fields[3], "relation", self.line_number)?,
            partition_id: nonempty(fields[4], "partition_id", self.line_number)?,
        };
        if let Some(previous) = &self.previous_id {
            if edge.edge_id <= *previous {
                return Err(format!(
                    "{} edge_id values must be strictly increasing; line {} has {} after {}",
                    self.label, self.line_number, edge.edge_id, previous
                ));
            }
        }
        self.previous_id = Some(edge.edge_id.clone());
        Ok(Some(edge))
    }
}

impl<R: BufRead> ExclusionReader<R> {
    fn new(mut reader: R) -> Result<Self, String> {
        parse_header(&mut reader, &EXCLUSION_HEADER, "exclusion")?;
        Ok(Self {
            reader,
            line_number: 1,
            previous_id: None,
        })
    }

    fn next(&mut self) -> Result<Option<Exclusion>, String> {
        let mut line = String::new();
        let count = self
            .reader
            .read_line(&mut line)
            .map_err(|error| format!("read exclusion row: {error}"))?;
        if count == 0 {
            return Ok(None);
        }
        self.line_number += 1;
        let line = line.trim_end_matches(['\n', '\r']);
        let fields = line.split('\t').collect::<Vec<_>>();
        if fields.len() != EXCLUSION_HEADER.len() {
            return Err(format!(
                "exclusion line {}: expected {} fields, got {}",
                self.line_number,
                EXCLUSION_HEADER.len(),
                fields.len()
            ));
        }
        let exclusion = Exclusion {
            edge_id: nonempty(fields[0], "edge_id", self.line_number)?,
            reason: nonempty(fields[1], "reason", self.line_number)?,
        };
        if let Some(previous) = &self.previous_id {
            if exclusion.edge_id <= *previous {
                return Err(format!(
                    "exclusion edge_id values must be strictly increasing; line {} has {} after {}",
                    self.line_number, exclusion.edge_id, previous
                ));
            }
        }
        self.previous_id = Some(exclusion.edge_id.clone());
        Ok(Some(exclusion))
    }
}

fn parse_reference_paths<R: BufRead>(mut reader: R) -> Result<Vec<ReferencePath>, String> {
    parse_header(&mut reader, &PATH_HEADER, "reference path")?;
    let mut paths: Vec<ReferencePath> = Vec::new();
    let mut line_number = 1usize;
    let mut previous_path_id: Option<String> = None;
    let mut previous_step_index = 0u64;

    loop {
        let mut line = String::new();
        let count = reader
            .read_line(&mut line)
            .map_err(|error| format!("read reference path row: {error}"))?;
        if count == 0 {
            break;
        }
        line_number += 1;
        let line = line.trim_end_matches(['\n', '\r']);
        let fields = line.split('\t').collect::<Vec<_>>();
        if fields.len() != PATH_HEADER.len() {
            return Err(format!(
                "reference path line {line_number}: expected {} fields, got {}",
                PATH_HEADER.len(),
                fields.len()
            ));
        }
        let path_id = nonempty(fields[0], "path_id", line_number)?;
        let step_index = fields[1]
            .parse::<u64>()
            .map_err(|error| format!("reference path step_index invalid at line {line_number}: {error}"))?;
        if step_index == 0 {
            return Err(format!("reference path step_index must start at 1 at line {line_number}"));
        }

        let starts_new_path = previous_path_id.as_deref() != Some(path_id.as_str());
        if starts_new_path {
            if let Some(previous) = &previous_path_id {
                if path_id <= *previous {
                    return Err(format!(
                        "reference path path_id must be strictly increasing: {path_id} after {previous}"
                    ));
                }
            }
            if step_index != 1 {
                return Err(format!(
                    "reference path {path_id} must start at step_index 1"
                ));
            }
            paths.push(ReferencePath {
                path_id: path_id.clone(),
                steps: Vec::new(),
            });
            previous_step_index = 0;
            previous_path_id = Some(path_id.clone());
        }
        if step_index != previous_step_index + 1 {
            return Err(format!(
                "reference path {path_id} step_index must be contiguous; expected {}, got {}",
                previous_step_index + 1,
                step_index
            ));
        }
        previous_step_index = step_index;

        let step = PathStep {
            edge_id: nonempty(fields[2], "edge_id", line_number)?,
            src: nonempty(fields[3], "src", line_number)?,
            dst: nonempty(fields[4], "dst", line_number)?,
            source_verified: false,
            represented_partition: None,
            exclusion_reason: None,
        };
        if let Some(previous) = paths.last().and_then(|path| path.steps.last()) {
            if previous.dst != step.src {
                return Err(format!(
                    "reference path {path_id} is discontinuous between edges {} and {}",
                    previous.edge_id, step.edge_id
                ));
            }
        }
        paths.last_mut().expect("path was created").steps.push(step);
    }

    if paths.is_empty() {
        return Err("reference path file has no path rows".to_owned());
    }
    Ok(paths)
}

fn audit_readers<S: BufRead, R: BufRead, X: BufRead, P: BufRead>(
    source_reader: S,
    represented_reader: R,
    exclusion_reader: X,
    path_reader: P,
) -> Result<AuditSummary, String> {
    let mut source = EdgeReader::new(source_reader, "source")?;
    let mut represented = EdgeReader::new(represented_reader, "represented")?;
    let mut exclusions = ExclusionReader::new(exclusion_reader)?;
    let mut paths = parse_reference_paths(path_reader)?;

    let mut path_edge_lookup: BTreeMap<String, Vec<(usize, usize)>> = BTreeMap::new();
    for (path_index, path) in paths.iter().enumerate() {
        if path.steps.is_empty() {
            return Err(format!("reference path {} is empty", path.path_id));
        }
        for (step_index, step) in path.steps.iter().enumerate() {
            path_edge_lookup
                .entry(step.edge_id.clone())
                .or_default()
                .push((path_index, step_index));
        }
    }

    let mut current_represented = represented.next()?;
    let mut current_exclusion = exclusions.next()?;
    let mut source_edge_count = 0u64;
    let mut represented_edge_count = 0u64;
    let mut excluded_edge_count = 0u64;

    while let Some(source_edge) = source.next()? {
        source_edge_count += 1;

        if let Some(represented_edge) = &current_represented {
            if represented_edge.edge_id < source_edge.edge_id {
                return Err(format!(
                    "orphan represented edge {} is absent from source stream",
                    represented_edge.edge_id
                ));
            }
        }
        if let Some(exclusion) = &current_exclusion {
            if exclusion.edge_id < source_edge.edge_id {
                return Err(format!(
                    "orphan exclusion {} is absent from source stream",
                    exclusion.edge_id
                ));
            }
        }

        let is_represented = current_represented
            .as_ref()
            .is_some_and(|edge| edge.edge_id == source_edge.edge_id);
        let is_excluded = current_exclusion
            .as_ref()
            .is_some_and(|edge| edge.edge_id == source_edge.edge_id);

        if is_represented && is_excluded {
            return Err(format!(
                "edge {} is both represented and excluded",
                source_edge.edge_id
            ));
        }
        if !is_represented && !is_excluded {
            return Err(format!(
                "silent edge loss: source edge {} is neither represented nor excluded",
                source_edge.edge_id
            ));
        }

        if let Some(references) = path_edge_lookup.get(&source_edge.edge_id) {
            for (path_index, step_index) in references {
                let step = &mut paths[*path_index].steps[*step_index];
                if step.src != source_edge.src || step.dst != source_edge.dst {
                    return Err(format!(
                        "reference path {} edge {} endpoint mismatch against source graph",
                        paths[*path_index].path_id, source_edge.edge_id
                    ));
                }
                step.source_verified = true;
            }
        }

        if is_represented {
            let represented_edge = current_represented
                .as_ref()
                .expect("represented flag implies edge");
            if represented_edge.src != source_edge.src
                || represented_edge.dst != source_edge.dst
                || represented_edge.relation != source_edge.relation
            {
                return Err(format!(
                    "represented edge {} changes source endpoints or relation",
                    source_edge.edge_id
                ));
            }
            represented_edge_count += 1;
            if let Some(references) = path_edge_lookup.get(&source_edge.edge_id) {
                for (path_index, step_index) in references {
                    paths[*path_index].steps[*step_index].represented_partition =
                        Some(represented_edge.partition_id.clone());
                }
            }
            current_represented = represented.next()?;
        } else {
            let exclusion = current_exclusion
                .as_ref()
                .expect("excluded flag implies exclusion");
            excluded_edge_count += 1;
            if let Some(references) = path_edge_lookup.get(&source_edge.edge_id) {
                for (path_index, step_index) in references {
                    paths[*path_index].steps[*step_index].exclusion_reason =
                        Some(exclusion.reason.clone());
                }
            }
            current_exclusion = exclusions.next()?;
        }
    }

    if let Some(edge) = current_represented {
        return Err(format!("orphan represented edge {} is absent from source", edge.edge_id));
    }
    if let Some(exclusion) = current_exclusion {
        return Err(format!("orphan exclusion {} is absent from source", exclusion.edge_id));
    }

    let mut path_summaries = Vec::with_capacity(paths.len());
    let mut retained_path_count = 0usize;
    let mut cut_path_count = 0usize;
    let mut total_reference_path_edges = 0usize;
    let mut retained_reference_path_edges = 0usize;

    for path in paths {
        let first = path.steps.first().expect("non-empty path validated");
        let last = path.steps.last().expect("non-empty path validated");
        let mut retained = 0usize;
        let mut excluded = 0usize;
        let mut crossings = 0usize;
        for (index, step) in path.steps.iter().enumerate() {
            if !step.source_verified {
                return Err(format!(
                    "reference path {} names edge {} absent from source graph",
                    path.path_id, step.edge_id
                ));
            }
            if step.represented_partition.is_some() {
                retained += 1;
            } else if step.exclusion_reason.is_some() {
                excluded += 1;
            } else {
                return Err(format!(
                    "reference path {} edge {} has no representation/exclusion disposition",
                    path.path_id, step.edge_id
                ));
            }
            if index > 0 {
                if let (Some(previous), Some(current)) = (
                    path.steps[index - 1].represented_partition.as_deref(),
                    step.represented_partition.as_deref(),
                ) {
                    if previous != current {
                        crossings += 1;
                    }
                }
            }
        }
        let status = if retained == path.steps.len() {
            retained_path_count += 1;
            "retained"
        } else {
            cut_path_count += 1;
            "cut_and_reported"
        };
        total_reference_path_edges += path.steps.len();
        retained_reference_path_edges += retained;
        path_summaries.push(PathSummary {
            path_id: path.path_id,
            source_node: first.src.clone(),
            observation_node: last.dst.clone(),
            edge_count: path.steps.len(),
            retained_edge_count: retained,
            excluded_edge_count: excluded,
            partition_crossings: crossings,
            status,
        });
    }

    Ok(AuditSummary {
        source_edge_count,
        represented_edge_count,
        excluded_edge_count,
        paths: path_summaries,
        retained_path_count,
        cut_path_count,
        total_reference_path_edges,
        retained_reference_path_edges,
    })
}

fn stable_sha256(path: &Path) -> Result<String, String> {
    let path_before = fs::symlink_metadata(path)
        .map_err(|error| format!("stat input {}: {error}", path.display()))?;
    if path_before.file_type().is_symlink() || !path_before.is_file() {
        return Err(format!("input must be a regular non-symlink file: {}", path.display()));
    }
    let mut file = File::open(path)
        .map_err(|error| format!("open input {}: {error}", path.display()))?;
    let before = file
        .metadata()
        .map_err(|error| format!("stat open input {}: {error}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let count = file
            .read(&mut buffer)
            .map_err(|error| format!("hash input {}: {error}", path.display()))?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    let after = file
        .metadata()
        .map_err(|error| format!("stat open input after hash: {error}"))?;
    let path_after = fs::symlink_metadata(path)
        .map_err(|error| format!("stat input after hash: {error}"))?;
    if path_after.file_type().is_symlink()
        || !path_after.is_file()
        || before.len() != after.len()
        || before.modified().ok() != after.modified().ok()
    {
        return Err(format!("input changed while hashing: {}", path.display()));
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn receipt_path_outside_inputs(args: &Args) -> Result<PathBuf, String> {
    let inputs = [
        &args.source_edges,
        &args.represented_edges,
        &args.exclusions,
        &args.reference_paths,
    ];
    let mut input_parents = Vec::new();
    for input in inputs {
        let canonical = fs::canonicalize(input)
            .map_err(|error| format!("canonicalize input {}: {error}", input.display()))?;
        let parent = canonical.parent().ok_or("input has no parent directory")?;
        input_parents.push(parent.to_path_buf());
    }
    let receipt_parent = args
        .receipt
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let filename = args
        .receipt
        .file_name()
        .ok_or("receipt path must include a filename")?;
    let parent = fs::canonicalize(receipt_parent)
        .map_err(|error| format!("canonicalize receipt parent: {error}"))?;
    let target = parent.join(filename);
    if input_parents.iter().any(|input_parent| target.starts_with(input_parent)) {
        return Err("receipt must be outside every input directory".to_owned());
    }
    Ok(target)
}

fn audit_files(args: &Args) -> Result<serde_json::Value, String> {
    let source_hash = stable_sha256(&args.source_edges)?;
    let represented_hash = stable_sha256(&args.represented_edges)?;
    let exclusion_hash = stable_sha256(&args.exclusions)?;
    let paths_hash = stable_sha256(&args.reference_paths)?;

    let source = BufReader::new(
        File::open(&args.source_edges)
            .map_err(|error| format!("open source edges: {error}"))?,
    );
    let represented = BufReader::new(
        File::open(&args.represented_edges)
            .map_err(|error| format!("open represented edges: {error}"))?,
    );
    let exclusions = BufReader::new(
        File::open(&args.exclusions).map_err(|error| format!("open exclusions: {error}"))?,
    );
    let paths = BufReader::new(
        File::open(&args.reference_paths)
            .map_err(|error| format!("open reference paths: {error}"))?,
    );

    let summary = audit_readers(source, represented, exclusions, paths)?;
    Ok(json!({
        "schema_version": "hardsec-graph-path-retention-receipt-v1",
        "result": "pass",
        "claim_scope": CLAIM_SCOPE,
        "inputs": {
            "source_edges_sha256": source_hash,
            "represented_edges_sha256": represented_hash,
            "exclusions_sha256": exclusion_hash,
            "reference_paths_sha256": paths_hash
        },
        "edge_accounting": {
            "source_edges": summary.source_edge_count,
            "represented_edges": summary.represented_edge_count,
            "excluded_edges": summary.excluded_edge_count,
            "accounted_edges": summary.represented_edge_count + summary.excluded_edge_count,
            "silent_drops": 0
        },
        "path_retention": {
            "reference_paths": summary.paths.len(),
            "retained_paths": summary.retained_path_count,
            "cut_and_reported_paths": summary.cut_path_count,
            "reference_path_edges": summary.total_reference_path_edges,
            "retained_reference_path_edges": summary.retained_reference_path_edges,
            "paths": summary.paths.iter().map(|path| json!({
                "path_id": path.path_id,
                "source_node": path.source_node,
                "observation_node": path.observation_node,
                "edge_count": path.edge_count,
                "retained_edge_count": path.retained_edge_count,
                "excluded_edge_count": path.excluded_edge_count,
                "partition_crossings": path.partition_crossings,
                "status": path.status
            })).collect::<Vec<_>>()
        },
        "limitations": [
            "This tool audits generic typed edge files; it does not parse Verilog or validate RTL semantics.",
            "It proves only exact accounting of the supplied source edge IDs and declared reference paths.",
            "The reference path list is caller-supplied; it does not establish that all security-relevant paths are represented.",
            "The utility streams edge and exclusion files but holds declared reference-path metadata in memory.",
            "A cut path is explicitly reported; this does not establish that the omitted relation is harmless.",
            "The receipt is unsigned and is not a detector or hardware-security verdict."
        ],
        "signature": null,
        "security_verdict": null
    }))
}

fn run(args: Args) -> Result<(), String> {
    let receipt_path = receipt_path_outside_inputs(&args)?;
    let receipt = audit_files(&args)?;
    let bytes = serde_json::to_vec_pretty(&receipt)
        .map_err(|error| format!("serialize graph path receipt: {error}"))?;
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&receipt_path)
        .map_err(|error| format!("create new receipt {}: {error}", receipt_path.display()))?;
    let write_result = output
        .write_all(&bytes)
        .and_then(|_| output.write_all(b"\n"))
        .and_then(|_| output.sync_all());
    drop(output);
    if let Err(error) = write_result {
        let _ = fs::remove_file(&receipt_path);
        return Err(format!("write receipt {}: {error}", receipt_path.display()));
    }
    println!("HARDSEC-010 graph path retention audit: PASS");
    println!("receipt: {}", receipt_path.display());
    println!("claim scope: {}", CLAIM_SCOPE);
    Ok(())
}

fn main() -> ExitCode {
    match parse_args() {
        Ok(None) => {
            println!("usage: hardsec_graph_path_retention_audit --source-edges FILE --represented-edges FILE --exclusions FILE --reference-paths FILE --receipt NEW_FILE");
            ExitCode::SUCCESS
        }
        Ok(Some(args)) => match run(args) {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                eprintln!("HARDSEC-010 graph path audit failed: {error}");
                ExitCode::FAILURE
            }
        },
        Err(error) => {
            eprintln!("{error}");
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    const EDGES_HEADER: &str = "edge_id\tsrc\tdst\trelation\tpartition_id\n";
    const EXCLUSIONS_HEADER: &str = "edge_id\treason\n";
    const PATHS_HEADER: &str = "path_id\tstep_index\tedge_id\tsrc\tdst\n";

    fn audit(
        source: &str,
        represented: &str,
        exclusions: &str,
        paths: &str,
    ) -> Result<AuditSummary, String> {
        audit_readers(
            Cursor::new(source.as_bytes()),
            Cursor::new(represented.as_bytes()),
            Cursor::new(exclusions.as_bytes()),
            Cursor::new(paths.as_bytes()),
        )
    }

    fn source_fixture() -> String {
        format!(
            "{EDGES_HEADER}e1\tn0\tn1\tdata\tp0\ne2\tn1\tn2\tcontrol\tp0\ne3\tn2\tn3\tdata\tp1\ne4\tn3\tn4\toutput\tp1\n"
        )
    }

    fn represented_fixture() -> String {
        format!(
            "{EDGES_HEADER}e1\tn0\tn1\tdata\tr0\ne2\tn1\tn2\tcontrol\tr0\ne3\tn2\tn3\tdata\tr1\ne4\tn3\tn4\toutput\tr1\n"
        )
    }

    fn paths_fixture() -> String {
        format!(
            "{PATHS_HEADER}trigger-to-output\t1\te1\tn0\tn1\ntrigger-to-output\t2\te2\tn1\tn2\ntrigger-to-output\t3\te3\tn2\tn3\ntrigger-to-output\t4\te4\tn3\tn4\n"
        )
    }

    #[test]
    fn accepts_fully_retained_reference_path() {
        let summary = audit(
            &source_fixture(),
            &represented_fixture(),
            EXCLUSIONS_HEADER,
            &paths_fixture(),
        )
        .unwrap();
        assert_eq!(summary.source_edge_count, 4);
        assert_eq!(summary.represented_edge_count, 4);
        assert_eq!(summary.excluded_edge_count, 0);
        assert_eq!(summary.retained_path_count, 1);
        assert_eq!(summary.cut_path_count, 0);
        assert_eq!(summary.paths[0].status, "retained");
    }

    #[test]
    fn reports_cut_path_when_missing_edge_has_exclusion_reason() {
        let represented = format!(
            "{EDGES_HEADER}e1\tn0\tn1\tdata\tr0\ne3\tn2\tn3\tdata\tr1\ne4\tn3\tn4\toutput\tr1\n"
        );
        let exclusions = format!("{EXCLUSIONS_HEADER}e2\tmemory budget partition cut\n");
        let summary = audit(
            &source_fixture(),
            &represented,
            &exclusions,
            &paths_fixture(),
        )
        .unwrap();
        assert_eq!(summary.represented_edge_count, 3);
        assert_eq!(summary.excluded_edge_count, 1);
        assert_eq!(summary.cut_path_count, 1);
        assert_eq!(summary.paths[0].status, "cut_and_reported");
        assert_eq!(summary.paths[0].excluded_edge_count, 1);
    }

    #[test]
    fn rejects_silent_edge_loss() {
        let represented = format!(
            "{EDGES_HEADER}e1\tn0\tn1\tdata\tr0\ne3\tn2\tn3\tdata\tr1\ne4\tn3\tn4\toutput\tr1\n"
        );
        assert!(audit(
            &source_fixture(),
            &represented,
            EXCLUSIONS_HEADER,
            &paths_fixture()
        )
        .unwrap_err()
        .contains("silent edge loss"));
    }

    #[test]
    fn rejects_orphan_represented_edge() {
        let represented = format!(
            "{EDGES_HEADER}e0\tnx\tny\tdata\tr0\ne1\tn0\tn1\tdata\tr0\ne2\tn1\tn2\tcontrol\tr0\ne3\tn2\tn3\tdata\tr1\ne4\tn3\tn4\toutput\tr1\n"
        );
        assert!(audit(
            &source_fixture(),
            &represented,
            EXCLUSIONS_HEADER,
            &paths_fixture()
        )
        .unwrap_err()
        .contains("orphan represented edge"));
    }

    #[test]
    fn rejects_orphan_exclusion() {
        let exclusions = format!("{EXCLUSIONS_HEADER}e0\tmissing from source\n");
        assert!(audit(
            &source_fixture(),
            &represented_fixture(),
            &exclusions,
            &paths_fixture()
        )
        .unwrap_err()
        .contains("orphan exclusion"));
    }

    #[test]
    fn rejects_edge_both_retained_and_excluded() {
        let exclusions = format!("{EXCLUSIONS_HEADER}e2\tcut\n");
        assert!(audit(
            &source_fixture(),
            &represented_fixture(),
            &exclusions,
            &paths_fixture()
        )
        .unwrap_err()
        .contains("both represented and excluded"));
    }

    #[test]
    fn rejects_duplicate_source_edge_ids() {
        let source = format!(
            "{EDGES_HEADER}e1\tn0\tn1\tdata\tp0\ne1\tn1\tn2\tdata\tp0\n"
        );
        assert!(audit(
            &source,
            &represented_fixture(),
            EXCLUSIONS_HEADER,
            &paths_fixture()
        )
        .unwrap_err()
        .contains("strictly increasing"));
    }

    #[test]
    fn rejects_unsorted_represented_edges() {
        let represented = format!(
            "{EDGES_HEADER}e2\tn1\tn2\tcontrol\tr0\ne1\tn0\tn1\tdata\tr0\n"
        );
        assert!(audit(
            &source_fixture(),
            &represented,
            EXCLUSIONS_HEADER,
            &paths_fixture()
        )
        .unwrap_err()
        .contains("strictly increasing"));
    }

    #[test]
    fn rejects_duplicate_exclusion_ids() {
        let exclusions = format!("{EXCLUSIONS_HEADER}e2\tcut one\ne2\tcut two\n");
        assert!(audit(
            &source_fixture(),
            &represented_fixture(),
            &exclusions,
            &paths_fixture()
        )
        .unwrap_err()
        .contains("strictly increasing"));
    }

    #[test]
    fn rejects_represented_endpoint_mutation() {
        let represented = represented_fixture().replace("e2\tn1\tn2", "e2\tn1\tnX");
        assert!(audit(
            &source_fixture(),
            &represented,
            EXCLUSIONS_HEADER,
            &paths_fixture()
        )
        .unwrap_err()
        .contains("changes source endpoints or relation"));
    }

    #[test]
    fn rejects_represented_relation_mutation() {
        let represented = represented_fixture().replace("e2\tn1\tn2\tcontrol", "e2\tn1\tn2\tdata");
        assert!(audit(
            &source_fixture(),
            &represented,
            EXCLUSIONS_HEADER,
            &paths_fixture()
        )
        .unwrap_err()
        .contains("changes source endpoints or relation"));
    }

    #[test]
    fn rejects_reference_path_endpoint_mismatch() {
        let paths = paths_fixture().replace("e2\tn1\tn2", "e2\tn1\tnX");
        assert!(audit(
            &source_fixture(),
            &represented_fixture(),
            EXCLUSIONS_HEADER,
            &paths
        )
        .unwrap_err()
        .contains("endpoint mismatch"));
    }

    #[test]
    fn rejects_discontinuous_reference_path() {
        let paths = paths_fixture().replace("trigger-to-output\t3\te3\tn2\tn3", "trigger-to-output\t3\te3\tnZ\tn3");
        assert!(audit(
            &source_fixture(),
            &represented_fixture(),
            EXCLUSIONS_HEADER,
            &paths
        )
        .unwrap_err()
        .contains("discontinuous"));
    }

    #[test]
    fn rejects_reference_path_edge_absent_from_source() {
        let paths = paths_fixture().replace("e4\tn3\tn4", "e5\tn3\tn4");
        assert!(audit(
            &source_fixture(),
            &represented_fixture(),
            EXCLUSIONS_HEADER,
            &paths
        )
        .unwrap_err()
        .contains("absent from source graph"));
    }

    #[test]
    fn requires_reference_path_steps_to_be_contiguous() {
        let paths = paths_fixture().replace("trigger-to-output\t2\te2", "trigger-to-output\t3\te2");
        assert!(audit(
            &source_fixture(),
            &represented_fixture(),
            EXCLUSIONS_HEADER,
            &paths
        )
        .unwrap_err()
        .contains("step_index must be contiguous"));
    }

    #[test]
    fn requires_reference_path_file_to_be_sorted_by_path_id() {
        let paths = format!(
            "{PATHS_HEADER}z-path\t1\te1\tn0\tn1\na-path\t1\te2\tn1\tn2\n"
        );
        assert!(audit(
            &source_fixture(),
            &represented_fixture(),
            EXCLUSIONS_HEADER,
            &paths
        )
        .unwrap_err()
        .contains("path_id must be strictly increasing"));
    }

    #[test]
    fn counts_partition_crossings_on_retained_path() {
        let summary = audit(
            &source_fixture(),
            &represented_fixture(),
            EXCLUSIONS_HEADER,
            &paths_fixture(),
        )
        .unwrap();
        assert_eq!(summary.paths[0].partition_crossings, 1);
    }

    #[test]
    fn accepts_multiple_reference_paths_sharing_edges() {
        let paths = format!(
            "{PATHS_HEADER}path-a\t1\te1\tn0\tn1\npath-a\t2\te2\tn1\tn2\npath-b\t1\te2\tn1\tn2\npath-b\t2\te3\tn2\tn3\n"
        );
        let summary = audit(
            &source_fixture(),
            &represented_fixture(),
            EXCLUSIONS_HEADER,
            &paths,
        )
        .unwrap();
        assert_eq!(summary.paths.len(), 2);
        assert_eq!(summary.retained_path_count, 2);
        assert_eq!(summary.total_reference_path_edges, 4);
    }

    #[test]
    fn rejects_empty_exclusion_reason() {
        let represented = format!(
            "{EDGE_HEADER}e1\tn0\tn1\tdata\tr0\n"
        );
        let exclusions = format!("{EXCLUSIONS_HEADER}e2\t\n");
        assert!(audit(
            &source_fixture(),
            &represented,
            &exclusions,
            &paths_fixture()
        )
        .unwrap_err()
        .contains("reason must be non-empty"));
    }
}
