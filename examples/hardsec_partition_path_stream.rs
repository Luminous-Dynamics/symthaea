//! HARDSEC-010 streaming graph-partition witness-path auditor.
//!
//! Original and partitioned edge lists must be sorted lexicographically by
//! edge_id. They are merge-joined with bounded edge-record memory. The node
//! partition map and supplied witness ledger remain in memory (O(V + W)); the
//! graph edge sets are not materialized. This checks supplied paths only.
//!
//! Input formats and claim ceilings are documented in
//! docs/research/HARDSEC_010_PATH_PRESERVING_SCALE_PROTOCOL.md.

use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

const CLAIM_SCOPE: &str = "input_integrity_and_supplied_witness_path_preservation_only";
const ORIGINAL_HEADER: &str = "edge_id\tsrc_node_id\tdst_node_id";
const PARTITIONED_HEADER: &str =
    "edge_id\tsrc_node_id\tdst_node_id\trepresentation";
const PARTITION_HEADER: &str = "node_id\tpartition_id";
const WITNESS_HEADER: &str =
    "path_id\tsequence_index\tedge_id\tsrc_node_id\tdst_node_id";

#[derive(Debug, Clone)]
struct Args {
    original_edges: PathBuf,
    partitioned_edges: PathBuf,
    node_partitions: PathBuf,
    witness_paths: PathBuf,
    receipt: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Edge {
    edge_id: String,
    src_node_id: String,
    dst_node_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PartitionedEdge {
    edge: Edge,
    representation: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct WitnessStep {
    sequence_index: usize,
    edge: Edge,
}

#[derive(Debug, Clone)]
struct Cursor<T> {
    reader: T,
    line_number: usize,
    previous_id: Option<String>,
}

impl<T: BufRead> Cursor<T> {
    fn with_header(mut reader: T, expected_header: &str, label: &str) -> Result<Self, String> {
        let mut header = String::new();
        let count = reader
            .read_line(&mut header)
            .map_err(|error| format!("read {label} header: {error}"))?;
        if count == 0 || header.trim_end_matches(&['\r', '\n'][..]) != expected_header {
            return Err(format!("{label} header must exactly match {expected_header}"));
        }
        Ok(Self {
            reader,
            line_number: 1,
            previous_id: None,
        })
    }

    fn next_fields(&mut self, label: &str, count: usize) -> Result<Option<Vec<String>>, String> {
        let mut line = String::new();
        let bytes = self
            .reader
            .read_line(&mut line)
            .map_err(|error| format!("read {label} line: {error}"))?;
        if bytes == 0 {
            return Ok(None);
        }
        self.line_number += 1;
        let line = line.trim_end_matches(&['\r', '\n'][..]);
        if line.is_empty() {
            return Err(format!("{label}: blank row at line {}", self.line_number));
        }
        let fields = line.split('\t').collect::<Vec<_>>();
        if fields.len() != count {
            return Err(format!(
                "{label}: line {} expected {count} fields, got {}",
                self.line_number,
                fields.len()
            ));
        }
        for (index, value) in fields.iter().enumerate() {
            if value.is_empty() || value.trim() != *value {
                return Err(format!(
                    "{label}: line {} field {} must be non-empty with no surrounding whitespace",
                    self.line_number,
                    index + 1
                ));
            }
        }
        let id = fields[0].to_owned();
        if let Some(previous) = &self.previous_id {
            if id <= *previous {
                return Err(format!(
                    "{label}: edge IDs must be strictly increasing; {} follows {} at line {}",
                    id, previous, self.line_number
                ));
            }
        }
        self.previous_id = Some(id);
        Ok(Some(fields.into_iter().map(str::to_owned).collect()))
    }
}

fn parse_args() -> Result<Option<Args>, String> {
    let mut original_edges = None;
    let mut partitioned_edges = None;
    let mut node_partitions = None;
    let mut witness_paths = None;
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
            "--original-edges" => original_edges = Some(PathBuf::from(value)),
            "--partitioned-edges" => partitioned_edges = Some(PathBuf::from(value)),
            "--node-partitions" => node_partitions = Some(PathBuf::from(value)),
            "--witness-paths" => witness_paths = Some(PathBuf::from(value)),
            "--receipt" => receipt = Some(PathBuf::from(value)),
            _ => return Err(format!("unknown argument: {flag}")),
        }
    }

    Ok(Some(Args {
        original_edges: original_edges.ok_or("--original-edges is required")?,
        partitioned_edges: partitioned_edges.ok_or("--partitioned-edges is required")?,
        node_partitions: node_partitions.ok_or("--node-partitions is required")?,
        witness_paths: witness_paths.ok_or("--witness-paths is required")?,
        receipt: receipt.ok_or("--receipt is required")?,
    }))
}

fn sha256_file(path: &Path, label: &str) -> Result<String, String> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| format!("stat {label} {}: {error}", path.display()))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(format!("{label} must be a regular non-symlink file"));
    }
    let mut file = File::open(path)
        .map_err(|error| format!("open {label} {}: {error}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = file
            .read(&mut buffer)
            .map_err(|error| format!("hash {label} {}: {error}", path.display()))?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    let after = fs::metadata(path)
        .map_err(|error| format!("restat {label} {}: {error}", path.display()))?;
    if metadata.len() != after.len() {
        return Err(format!("{label} changed while hashing"));
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn open_cursor(path: &Path, header: &str, label: &str) -> Result<Cursor<BufReader<File>>, String> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| format!("stat {label} {}: {error}", path.display()))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(format!("{label} must be a regular non-symlink file"));
    }
    let file = File::open(path)
        .map_err(|error| format!("open {label} {}: {error}", path.display()))?;
    Cursor::with_header(BufReader::new(file), header, label)
}

fn load_node_partitions(path: &Path) -> Result<BTreeMap<String, String>, String> {
    let file = File::open(path)
        .map_err(|error| format!("open node partitions {}: {error}", path.display()))?;
    let mut lines = BufReader::new(file).lines();
    let header = lines
        .next()
        .ok_or("node partitions file is empty")?
        .map_err(|error| format!("read node partitions header: {error}"))?;
    if header.trim_end_matches('\r') != PARTITION_HEADER {
        return Err(format!("node partitions header must exactly match {PARTITION_HEADER}"));
    }
    let mut assignments = BTreeMap::new();
    for (index, line) in lines.enumerate() {
        let line_number = index + 2;
        let line = line.map_err(|error| format!("read node partitions line: {error}"))?;
        if line.is_empty() {
            return Err(format!("node partitions: blank row at line {line_number}"));
        }
        let fields = line.split('\t').collect::<Vec<_>>();
        if fields.len() != 2 || fields.iter().any(|field| field.is_empty() || field.trim() != *field) {
            return Err(format!("node partitions: invalid row at line {line_number}"));
        }
        if assignments
            .insert(fields[0].to_owned(), fields[1].to_owned())
            .is_some()
        {
            return Err(format!("node partitions: duplicate node_id {}", fields[0]));
        }
    }
    if assignments.is_empty() {
        return Err("node partitions has no data rows".to_owned());
    }
    Ok(assignments)
}

fn load_witness_paths(path: &Path) -> Result<BTreeMap<String, Vec<WitnessStep>>, String> {
    let file = File::open(path)
        .map_err(|error| format!("open witness paths {}: {error}", path.display()))?;
    let mut lines = BufReader::new(file).lines();
    let header = lines
        .next()
        .ok_or("witness paths file is empty")?
        .map_err(|error| format!("read witness paths header: {error}"))?;
    if header.trim_end_matches('\r') != WITNESS_HEADER {
        return Err(format!("witness paths header must exactly match {WITNESS_HEADER}"));
    }

    let mut grouped: BTreeMap<String, BTreeMap<usize, WitnessStep>> = BTreeMap::new();
    for (index, line) in lines.enumerate() {
        let line_number = index + 2;
        let line = line.map_err(|error| format!("read witness paths line: {error}"))?;
        if line.is_empty() {
            return Err(format!("witness paths: blank row at line {line_number}"));
        }
        let fields = line.split('\t').collect::<Vec<_>>();
        if fields.len() != 5
            || fields.iter().any(|field| field.is_empty() || field.trim() != *field)
        {
            return Err(format!("witness paths: invalid row at line {line_number}"));
        }
        let sequence_index = fields[1].parse::<usize>().map_err(|error| {
            format!("witness path {} invalid sequence_index: {error}", fields[0])
        })?;
        let step = WitnessStep {
            sequence_index,
            edge: Edge {
                edge_id: fields[2].to_owned(),
                src_node_id: fields[3].to_owned(),
                dst_node_id: fields[4].to_owned(),
            },
        };
        if grouped
            .entry(fields[0].to_owned())
            .or_default()
            .insert(sequence_index, step)
            .is_some()
        {
            return Err(format!(
                "witness paths: duplicate sequence index {sequence_index} for {}",
                fields[0]
            ));
        }
    }

    let mut paths = BTreeMap::new();
    for (path_id, indexed_steps) in grouped {
        let mut steps: Vec<WitnessStep> = Vec::with_capacity(indexed_steps.len());
        for (expected_index, (actual_index, step)) in indexed_steps.into_iter().enumerate() {
            if actual_index != expected_index {
                return Err(format!(
                    "witness path {path_id} expected sequence index {expected_index}, got {actual_index}"
                ));
            }
            if let Some(previous) = steps.last() {
                if previous.edge.dst_node_id != step.edge.src_node_id {
                    return Err(format!(
                        "witness path {path_id} is disconnected between {} and {}",
                        previous.edge.edge_id, step.edge.edge_id
                    ));
                }
            }
            steps.push(step);
        }
        if steps.is_empty() {
            return Err(format!("witness path {path_id} is empty"));
        }
        paths.insert(path_id, steps);
    }
    if paths.is_empty() {
        return Err("witness paths has no path records".to_owned());
    }
    Ok(paths)
}

fn write_receipt(path: &Path, bytes: &[u8], inputs: &[&Path]) -> Result<(), String> {
    let filename = path.file_name().ok_or("receipt path must include filename")?;
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let canonical_parent = fs::canonicalize(parent)
        .map_err(|error| format!("canonicalize receipt parent {}: {error}", parent.display()))?;
    let target = canonical_parent.join(filename);

    for input in inputs {
        let canonical_input = fs::canonicalize(input)
            .map_err(|error| format!("canonicalize input {}: {error}", input.display()))?;
        if target == canonical_input {
            return Err("receipt path must not overwrite an input file".to_owned());
        }
    }

    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&target)
        .map_err(|error| format!("create receipt {}: {error}", target.display()))?;
    let write_result = file.write_all(bytes).and_then(|_| file.sync_all());
    drop(file);
    if let Err(error) = write_result {
        let _ = fs::remove_file(&target);
        return Err(format!("write receipt {}: {error}", target.display()));
    }
    Ok(())
}

fn partition_for<'a>(
    assignments: &'a BTreeMap<String, String>,
    node_id: &str,
    context: &str,
) -> Result<&'a str, String> {
    assignments
        .get(node_id)
        .map(String::as_str)
        .ok_or_else(|| format!("{context}: node {node_id} has no partition assignment"))
}

fn run(args: Args) -> Result<(), String> {
    let original_sha = sha256_file(&args.original_edges, "original edges")?;
    let partitioned_sha = sha256_file(&args.partitioned_edges, "partitioned edges")?;
    let partitions_sha = sha256_file(&args.node_partitions, "node partitions")?;
    let witnesses_sha = sha256_file(&args.witness_paths, "witness paths")?;

    let assignments = load_node_partitions(&args.node_partitions)?;
    let paths = load_witness_paths(&args.witness_paths)?;
    let mut witness_edges: BTreeMap<String, Vec<(String, WitnessStep)>> = BTreeMap::new();
    for (path_id, steps) in &paths {
        for step in steps {
            witness_edges
                .entry(step.edge.edge_id.clone())
                .or_default()
                .push((path_id.clone(), step.clone()));
        }
    }

    let mut original = open_cursor(
        &args.original_edges,
        ORIGINAL_HEADER,
        "original edges",
    )?;
    let mut partitioned = open_cursor(
        &args.partitioned_edges,
        PARTITIONED_HEADER,
        "partitioned edges",
    )?;

    let mut next_original = original.next_fields("original edges", 3)?;
    let mut next_partitioned = partitioned.next_fields("partitioned edges", 4)?;
    let mut seen_witness_edges = BTreeSet::new();
    let mut preserved_witness_edges = 0_usize;
    let mut lost_witness_edges = Vec::new();
    let mut original_edge_count = 0_usize;
    let mut partitioned_edge_count = 0_usize;
    let mut original_cross_partition_edge_count = 0_usize;
    let mut partitioned_bridge_count = 0_usize;
    let mut preserved_cross_partition_witness_edges = 0_usize;
    let mut total_cross_partition_witness_edges = 0_usize;

    while let Some(original_row) = next_original.take() {
        original_edge_count += 1;
        let original_edge = Edge {
            edge_id: original_row[0].clone(),
            src_node_id: original_row[1].clone(),
            dst_node_id: original_row[2].clone(),
        };
        if original_edge.src_node_id == original_edge.dst_node_id {
            return Err(format!("original edge {} is an unsupported self-loop", original_edge.edge_id));
        }
        let src_partition = partition_for(&assignments, &original_edge.src_node_id, "original edge")?;
        let dst_partition = partition_for(&assignments, &original_edge.dst_node_id, "original edge")?;
        let crosses = src_partition != dst_partition;
        if crosses {
            original_cross_partition_edge_count += 1;
        }

        loop {
            let Some(partition_row) = next_partitioned.as_ref() else {
                break;
            };
            if partition_row[0] >= original_edge.edge_id {
                break;
            }
            return Err(format!(
                "partitioned graph contains edge {} absent from or out of order relative to original graph",
                partition_row[0]
            ));
        }

        let mut retained_representation = None;
        if let Some(partition_row) = next_partitioned.as_ref() {
            if partition_row[0] == original_edge.edge_id {
                if partition_row[1] != original_edge.src_node_id
                    || partition_row[2] != original_edge.dst_node_id
                {
                    return Err(format!(
                        "partitioned edge {} has endpoints different from original graph",
                        original_edge.edge_id
                    ));
                }
                let expected = if crosses { "bridge" } else { "internal" };
                if partition_row[3] != expected {
                    return Err(format!(
                        "partitioned edge {} must be represented as {expected}",
                        original_edge.edge_id
                    ));
                }
                retained_representation = Some(partition_row[3].clone());
                partitioned_edge_count += 1;
                if crosses {
                    partitioned_bridge_count += 1;
                }
                next_partitioned = partitioned.next_fields("partitioned edges", 4)?;
            }
        }

        if let Some(references) = witness_edges.get(&original_edge.edge_id) {
            for (path_id, step) in references {
                if step.edge.src_node_id != original_edge.src_node_id
                    || step.edge.dst_node_id != original_edge.dst_node_id
                {
                    return Err(format!(
                        "witness path {path_id} edge {} endpoints disagree with original graph",
                        original_edge.edge_id
                    ));
                }
                if crosses {
                    total_cross_partition_witness_edges += 1;
                }
                if retained_representation.is_some() {
                    preserved_witness_edges += 1;
                    seen_witness_edges.insert(original_edge.edge_id.clone());
                    if crosses {
                        preserved_cross_partition_witness_edges += 1;
                    }
                } else {
                    lost_witness_edges.push(serde_json::json!({
                        "path_id": path_id,
                        "sequence_index": step.sequence_index,
                        "edge_id": original_edge.edge_id,
                        "src_node_id": original_edge.src_node_id,
                        "dst_node_id": original_edge.dst_node_id,
                        "src_partition": src_partition,
                        "dst_partition": dst_partition,
                        "crosses_partition": crosses,
                    }));
                }
            }
        }

        next_original = original.next_fields("original edges", 3)?;
    }

    if let Some(extra) = next_partitioned {
        return Err(format!(
            "partitioned graph contains extra edge {} absent from original graph",
            extra[0]
        ));
    }

    // Re-hash after the streaming pass. This detects persistent input mutation
    // between the initial digest pass and completion of the merge join.
    for (input_path, label, expected) in [
        (&args.original_edges, "original edges", &original_sha),
        (&args.partitioned_edges, "partitioned edges", &partitioned_sha),
        (&args.node_partitions, "node partitions", &partitions_sha),
        (&args.witness_paths, "witness paths", &witnesses_sha),
    ] {
        let actual = sha256_file(input_path, label)?;
        if &actual != expected {
            return Err(format!("{label} changed during audit; discard this run and retry"));
        }
    }

    for (path_id, steps) in &paths {
        for step in steps {
            if !witness_edges.contains_key(&step.edge.edge_id) {
                return Err(format!("internal witness index lost edge {}", step.edge.edge_id));
            }
            if !seen_witness_edges.contains(&step.edge.edge_id) {
                // The edge may have been absent from the original graph.
                if !lost_witness_edges.iter().any(|item| {
                    item["path_id"] == *path_id
                        && item["edge_id"] == step.edge.edge_id
                }) {
                    return Err(format!(
                        "witness path {path_id} edge {} is absent from original graph",
                        step.edge.edge_id
                    ));
                }
            }
        }
    }

    let mut path_results = Vec::new();
    for (path_id, steps) in &paths {
        let path_lost = lost_witness_edges
            .iter()
            .filter(|item| item["path_id"] == *path_id)
            .count();
        let path_cross = steps
            .iter()
            .filter(|step| {
                let src = assignments.get(&step.edge.src_node_id);
                let dst = assignments.get(&step.edge.dst_node_id);
                src.is_some() && dst.is_some() && src != dst
            })
            .count();
        let path_preserved_cross = steps
            .iter()
            .filter(|step| {
                let src = assignments.get(&step.edge.src_node_id);
                let dst = assignments.get(&step.edge.dst_node_id);
                src.is_some()
                    && dst.is_some()
                    && src != dst
                    && seen_witness_edges.contains(&step.edge.edge_id)
            })
            .count();
        path_results.push(serde_json::json!({
            "path_id": path_id,
            "edge_count": steps.len(),
            "preserved_edge_count": steps.len() - path_lost,
            "lost_edge_count": path_lost,
            "cross_partition_edge_count": path_cross,
            "preserved_cross_partition_edge_count": path_preserved_cross,
            "status": if path_lost == 0 { "preserved" } else { "lost" },
        }));
    }

    let lost_count = lost_witness_edges.len();
    let total_witness_edges = paths.values().map(Vec::len).sum::<usize>();
    let partition_count = assignments.values().collect::<BTreeSet<_>>().len();
    let result = if lost_count == 0 { "pass" } else { "fail" };
    let receipt = serde_json::json!({
        "schema_version": "hardsec-partition-path-stream-audit-v1",
        "result": result,
        "claim_scope": CLAIM_SCOPE,
        "input_sha256": {
            "original_edges": original_sha,
            "partitioned_edges": partitioned_sha,
            "node_partitions": partitions_sha,
            "witness_paths": witnesses_sha,
        },
        "input_record_counts": {
            "original_edges": original_edge_count,
            "partitioned_edges": partitioned_edge_count,
            "node_partition_assignments": assignments.len(),
            "witness_paths": paths.len(),
        },
        "audit": {
            "original_edge_count": original_edge_count,
            "partitioned_edge_count": partitioned_edge_count,
            "partition_count": partition_count,
            "witness_path_count": paths.len(),
            "total_witness_edges": total_witness_edges,
            "preserved_witness_edges": preserved_witness_edges,
            "lost_witness_edge_count": lost_count,
            "total_cross_partition_witness_edges": total_cross_partition_witness_edges,
            "preserved_cross_partition_witness_edges": preserved_cross_partition_witness_edges,
            "original_cross_partition_edge_count": original_cross_partition_edge_count,
            "partitioned_cross_partition_bridge_count": partitioned_bridge_count,
            "path_results": path_results,
            "lost_witness_edges": lost_witness_edges,
            "edge_streaming": true,
            "memory_contract": "O(V + W + maximum_input_record_size); original and partitioned edge lists are merge-joined",
        },
        "signature": null,
        "security_verdict": null,
        "notes": [
            "The original and partitioned edge lists must be strictly sorted by edge_id.",
            "The node partition map and witness ledger are resident in memory; edge sets are not materialized.",
            "Only supplied witness paths were checked; this does not prove that all graph paths survive.",
            "A pass is not detector-quality evidence, a Trojan-absence claim, or design-security certification."
        ]
    });

    let bytes = serde_json::to_vec_pretty(&receipt)
        .map_err(|error| format!("serialize streaming receipt: {error}"))?;
    write_receipt(
        &args.receipt,
        &bytes,
        &[
            &args.original_edges,
            &args.partitioned_edges,
            &args.node_partitions,
            &args.witness_paths,
        ],
    )?;

    println!("HARDSEC-010 streaming path audit: {result}");
    println!("original edges streamed: {original_edge_count}");
    println!("partitioned edges streamed: {partitioned_edge_count}");
    println!("witness edges: {total_witness_edges}");
    println!("lost witness edges: {lost_count}");
    println!("receipt: {}", args.receipt.display());
    println!("claim scope: {CLAIM_SCOPE}");

    if result != "pass" {
        return Err("one or more supplied witness paths were severed by partitioning".to_owned());
    }
    Ok(())
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(Some(args)) => args,
        Ok(None) => {
            println!(
                "usage: hardsec_partition_path_stream --original-edges FILE --partitioned-edges FILE --node-partitions FILE --witness-paths FILE --receipt NEW_FILE"
            );
            return ExitCode::SUCCESS;
        }
        Err(error) => {
            eprintln!("{error}");
            return ExitCode::from(2);
        }
    };
    match run(args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("HARDSEC-010 streaming audit failed: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static NEXT: AtomicUsize = AtomicUsize::new(0);

    struct Fixture {
        root: PathBuf,
        original: PathBuf,
        partitioned: PathBuf,
        partitions: PathBuf,
        witnesses: PathBuf,
        receipt: PathBuf,
    }

    impl Fixture {
        fn new(partitioned_rows: &str) -> Self {
            let id = NEXT.fetch_add(1, Ordering::Relaxed);
            let root = env::temp_dir().join(format!(
                "hardsec-stream-path-{}-{id}",
                std::process::id()
            ));
            fs::create_dir_all(root.join("out")).unwrap();
            let original = root.join("original.tsv");
            let partitioned = root.join("partitioned.tsv");
            let partitions = root.join("partitions.tsv");
            let witnesses = root.join("witnesses.tsv");
            let receipt = root.join("out/receipt.json");
            fs::write(
                &original,
                concat!(
                    "edge_id\tsrc_node_id\tdst_node_id\n",
                    "e1\ta\tb\n",
                    "e2\tb\tc\n",
                    "e3\tc\td\n"
                ),
            )
            .unwrap();
            fs::write(
                &partitioned,
                partitioned_rows,
            )
            .unwrap();
            fs::write(
                &partitions,
                concat!(
                    "node_id\tpartition_id\n",
                    "a\tp1\n",
                    "b\tp1\n",
                    "c\tp2\n",
                    "d\tp2\n"
                ),
            )
            .unwrap();
            fs::write(
                &witnesses,
                concat!(
                    "path_id\tsequence_index\tedge_id\tsrc_node_id\tdst_node_id\n",
                    "w1\t0\te1\ta\tb\n",
                    "w1\t1\te2\tb\tc\n",
                    "w1\t2\te3\tc\td\n"
                ),
            )
            .unwrap();
            Self { root, original, partitioned, partitions, witnesses, receipt }
        }

        fn args(&self) -> Args {
            Args {
                original_edges: self.original.clone(),
                partitioned_edges: self.partitioned.clone(),
                node_partitions: self.partitions.clone(),
                witness_paths: self.witnesses.clone(),
                receipt: self.receipt.clone(),
            }
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.root);
        }
    }

    const COMPLETE: &str = concat!(
        "edge_id\tsrc_node_id\tdst_node_id\trepresentation\n",
        "e1\ta\tb\tinternal\n",
        "e2\tb\tc\tbridge\n",
        "e3\tc\td\tinternal\n"
    );

    #[test]
    fn streams_sorted_edge_files_and_accepts_explicit_bridge() {
        let fixture = Fixture::new(COMPLETE);
        run(fixture.args()).unwrap();
        let receipt: serde_json::Value =
            serde_json::from_slice(&fs::read(&fixture.receipt).unwrap()).unwrap();
        assert_eq!(receipt["result"], "pass");
        assert_eq!(receipt["audit"]["edge_streaming"], true);
        assert_eq!(receipt["audit"]["preserved_witness_edges"], 3);
        assert_eq!(receipt["audit"]["preserved_cross_partition_witness_edges"], 1);
    }

    #[test]
    fn emits_fail_receipt_for_lost_witness_edge() {
        let partitioned = concat!(
            "edge_id\tsrc_node_id\tdst_node_id\trepresentation\n",
            "e1\ta\tb\tinternal\n",
            "e3\tc\td\tinternal\n"
        );
        let fixture = Fixture::new(partitioned);
        assert!(run(fixture.args()).unwrap_err().contains("paths were severed"));
        let receipt: serde_json::Value =
            serde_json::from_slice(&fs::read(&fixture.receipt).unwrap()).unwrap();
        assert_eq!(receipt["result"], "fail");
        assert_eq!(receipt["audit"]["lost_witness_edge_count"], 1);
    }

    #[test]
    fn rejects_unsorted_original_edges() {
        let fixture = Fixture::new(COMPLETE);
        let text = fs::read_to_string(&fixture.original)
            .unwrap()
            .replace("e1\ta\tb\n", "z1\ta\tb\n")
            .replace("e3\tc\td\n", "a3\tc\td\n");
        fs::write(&fixture.original, text).unwrap();
        assert!(run(fixture.args()).unwrap_err().contains("strictly increasing"));
    }

    #[test]
    fn rejects_unsorted_partitioned_edges() {
        let partitioned = concat!(
            "edge_id\tsrc_node_id\tdst_node_id\trepresentation\n",
            "e2\tb\tc\tbridge\n",
            "e1\ta\tb\tinternal\n"
        );
        let fixture = Fixture::new(partitioned);
        assert!(run(fixture.args()).unwrap_err().contains("strictly increasing"));
    }

    #[test]
    fn rejects_partitioned_edge_missing_from_original() {
        let partitioned = concat!(
            "edge_id\tsrc_node_id\tdst_node_id\trepresentation\n",
            "e1\ta\tb\tinternal\n",
            "e2\tb\tc\tbridge\n",
            "e9\tx\ty\tbridge\n"
        );
        let fixture = Fixture::new(partitioned);
        assert!(run(fixture.args()).unwrap_err().contains("absent from"));
    }

    #[test]
    fn rejects_wrong_bridge_annotation() {
        let partitioned = concat!(
            "edge_id\tsrc_node_id\tdst_node_id\trepresentation\n",
            "e1\ta\tb\tinternal\n",
            "e2\tb\tc\tinternal\n",
            "e3\tc\td\tinternal\n"
        );
        let fixture = Fixture::new(partitioned);
        assert!(run(fixture.args()).unwrap_err().contains("must be represented as bridge"));
    }

    #[test]
    fn rejects_witness_edge_with_wrong_original_endpoints() {
        let fixture = Fixture::new(COMPLETE);
        let text = fs::read_to_string(&fixture.witnesses)
            .unwrap()
            .replace("w1\t1\te2\tb\tc", "w1\t1\te2\tx\tc");
        fs::write(&fixture.witnesses, text).unwrap();
        assert!(run(fixture.args()).unwrap_err().contains("endpoints disagree"));
    }

    #[test]
    fn rejects_missing_node_partition_assignment() {
        let fixture = Fixture::new(COMPLETE);
        let text = fs::read_to_string(&fixture.partitions)
            .unwrap()
            .replace("c\tp2\n", "");
        fs::write(&fixture.partitions, text).unwrap();
        assert!(run(fixture.args()).unwrap_err().contains("no partition assignment"));
    }

    #[test]
    fn streams_large_edge_sets_with_small_witness_ledger() {
        use std::io::BufWriter;

        const EDGE_COUNT: usize = 1_000_000;
        let id = NEXT.fetch_add(1, Ordering::Relaxed);
        let root = env::temp_dir().join(format!(
            "hardsec-stream-scale-{}-{id}",
            std::process::id()
        ));
        fs::create_dir_all(root.join("out")).unwrap();
        let original = root.join("original.tsv");
        let partitioned = root.join("partitioned.tsv");
        let partitions = root.join("partitions.tsv");
        let witnesses = root.join("witnesses.tsv");
        let receipt = root.join("out/receipt.json");

        {
            let file = File::create(&original).unwrap();
            let mut writer = BufWriter::new(file);
            writeln!(writer, "{ORIGINAL_HEADER}").unwrap();
            for index in 0..EDGE_COUNT {
                writeln!(writer, "e{index:08}\tn00000000\tn{:08}", index + 1).unwrap();
            }
            writer.flush().unwrap();
        }
        {
            let file = File::create(&partitioned).unwrap();
            let mut writer = BufWriter::new(file);
            writeln!(writer, "{PARTITIONED_HEADER}").unwrap();
            for index in 0..EDGE_COUNT {
                let target_partition = (index + 1) / 1000;
                let representation = if target_partition == 0 { "internal" } else { "bridge" };
                writeln!(
                    writer,
                    "e{index:08}\tn00000000\tn{:08}\t{representation}",
                    index + 1
                )
                .unwrap();
            }
            writer.flush().unwrap();
        }
        {
            let file = File::create(&partitions).unwrap();
            let mut writer = BufWriter::new(file);
            writeln!(writer, "{PARTITION_HEADER}").unwrap();
            for node in 0..=EDGE_COUNT {
                writeln!(writer, "n{node:08}\tp{:04}", node / 1000).unwrap();
            }
            writer.flush().unwrap();
        }
        fs::write(
            &witnesses,
            "path_id\tsequence_index\tedge_id\tsrc_node_id\tdst_node_id\nw1\t0\te00000999\tn00000000\tn00001000\n",
        )
        .unwrap();

        let fixture = Args {
            original_edges: original,
            partitioned_edges: partitioned,
            node_partitions: partitions,
            witness_paths: witnesses,
            receipt: receipt.clone(),
        };
        run(fixture).unwrap();
        let output: serde_json::Value =
            serde_json::from_slice(&fs::read(&receipt).unwrap()).unwrap();
        assert_eq!(output["audit"]["original_edge_count"], EDGE_COUNT);
        assert_eq!(output["audit"]["partitioned_edge_count"], EDGE_COUNT);
        assert_eq!(output["audit"]["node_partition_assignments"], EDGE_COUNT + 1);
        assert_eq!(output["audit"]["partition_count"], (EDGE_COUNT / 1000) + 1);
        assert_eq!(output["audit"]["total_witness_edges"], 1);
        assert_eq!(output["audit"]["lost_witness_edge_count"], 0);
        assert_eq!(output["audit"]["total_cross_partition_witness_edges"], 1);
        assert_eq!(output["audit"]["preserved_cross_partition_witness_edges"], 1);
        assert_eq!(output["audit"]["partitioned_cross_partition_bridge_count"], 999_001);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn refuses_to_overwrite_receipt() {
        let fixture = Fixture::new(COMPLETE);
        fs::write(&fixture.receipt, b"keep").unwrap();
        assert!(run(fixture.args()).unwrap_err().contains("File exists"));
    }
}
