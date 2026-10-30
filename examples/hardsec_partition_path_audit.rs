//! HARDSEC-010 graph-partition witness-path preservation auditor.
//!
//! Audits whether supplied, ordered witness paths survive a graph partition.
//! Cross-partition dependencies must remain explicit bridge edges. The tool
//! validates only the supplied paths; it does not prove that all graph paths
//! are preserved, infer missing witnesses, detect Trojans, or certify a design.
//!
//! Input TSV schemas:
//!   original-edges.tsv:     edge_id, src_node_id, dst_node_id
//!   partitioned-edges.tsv:  edge_id, src_node_id, dst_node_id, representation
//!                           representation is "internal" or "bridge"
//!   node-partitions.tsv:    node_id, partition_id
//!   witness-paths.tsv:     path_id, sequence_index, edge_id, src_node_id, dst_node_id
//!
//! Example:
//!   cargo run --example hardsec_partition_path_audit --features hardsec-design-audit -- \
//!     --original-edges original-edges.tsv \
//!     --partitioned-edges partitioned-edges.tsv \
//!     --node-partitions node-partitions.tsv \
//!     --witness-paths witness-paths.tsv \
//!     --receipt /path/outside/inputs/path-audit.json

use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};
use std::process::ExitCode;

const CLAIM_SCOPE: &str = "input_integrity_and_supplied_witness_path_preservation_only";
const ORIGINAL_HEADER: [&str; 3] = ["edge_id", "src_node_id", "dst_node_id"];
const PARTITIONED_HEADER: [&str; 4] = [
    "edge_id",
    "src_node_id",
    "dst_node_id",
    "representation",
];
const PARTITION_HEADER: [&str; 2] = ["node_id", "partition_id"];
const WITNESS_HEADER: [&str; 5] = [
    "path_id",
    "sequence_index",
    "edge_id",
    "src_node_id",
    "dst_node_id",
];

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
    path_id: String,
    sequence_index: usize,
    edge: Edge,
}

#[derive(Debug, Clone)]
struct Inputs {
    original_edges: Vec<Edge>,
    partitioned_edges: Vec<PartitionedEdge>,
    node_partitions: BTreeMap<String, String>,
    witness_paths: BTreeMap<String, Vec<WitnessStep>>,
}

#[derive(Debug, Clone)]
struct InputHashes {
    original_edges: String,
    partitioned_edges: String,
    node_partitions: String,
    witness_paths: String,
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

fn sha256_bytes(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    format!("{digest:x}")
}

fn sha256_reader(mut reader: impl Read) -> Result<String, String> {
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = reader
            .read(&mut buffer)
            .map_err(|error| format!("read while hashing: {error}"))?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn read_regular_file(path: &Path, label: &str) -> Result<Vec<u8>, String> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| format!("stat {label} {}: {error}", path.display()))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(format!("{label} must be a regular non-symlink file"));
    }
    fs::read(path).map_err(|error| format!("read {label} {}: {error}", path.display()))
}

fn parse_tsv<'a>(
    bytes: &'a [u8],
    expected_header: &[&str],
    label: &str,
) -> Result<Vec<Vec<&'a str>>, String> {
    let text = std::str::from_utf8(bytes)
        .map_err(|error| format!("{label} is not UTF-8: {error}"))?;
    let mut lines = text.split_terminator('\n');
    let header = lines
        .next()
        .ok_or_else(|| format!("{label} is empty"))?
        .trim_end_matches('\r');
    if header.split('\t').collect::<Vec<_>>() != expected_header {
        return Err(format!(
            "{label} header must exactly match {}",
            expected_header.join("\t")
        ));
    }

    let mut rows = Vec::new();
    for (index, raw_line) in lines.enumerate() {
        let line_number = index + 2;
        let line = raw_line.trim_end_matches('\r');
        if line.is_empty() {
            return Err(format!("{label}: blank row at line {line_number}"));
        }
        let fields = line.split('\t').collect::<Vec<_>>();
        if fields.len() != expected_header.len() {
            return Err(format!(
                "{label}: line {line_number} expected {} fields, got {}",
                expected_header.len(),
                fields.len()
            ));
        }
        for (field_index, value) in fields.iter().enumerate() {
            if value.trim().is_empty() || value.trim() != *value {
                return Err(format!(
                    "{label}: line {line_number} field {} must be non-empty with no surrounding whitespace",
                    expected_header[field_index]
                ));
            }
        }
        rows.push(fields);
    }

    if rows.is_empty() {
        return Err(format!("{label} has no data rows"));
    }
    Ok(rows)
}

fn parse_original_edges(bytes: &[u8]) -> Result<Vec<Edge>, String> {
    let rows = parse_tsv(bytes, &ORIGINAL_HEADER, "original edges")?;
    let mut edge_ids = BTreeSet::new();
    let mut edges = Vec::with_capacity(rows.len());

    for (index, row) in rows.iter().enumerate() {
        let edge = Edge {
            edge_id: row[0].to_owned(),
            src_node_id: row[1].to_owned(),
            dst_node_id: row[2].to_owned(),
        };
        if !edge_ids.insert(edge.edge_id.clone()) {
            return Err(format!(
                "original edges: duplicate edge_id {} at data row {}",
                edge.edge_id,
                index + 1
            ));
        }
        if edge.src_node_id == edge.dst_node_id {
            return Err(format!(
                "original edges: self-loop {} is unsupported in witness paths",
                edge.edge_id
            ));
        }
        edges.push(edge);
    }
    edges.sort_by(|left, right| left.edge_id.cmp(&right.edge_id));
    Ok(edges)
}

fn parse_partitioned_edges(bytes: &[u8]) -> Result<Vec<PartitionedEdge>, String> {
    let rows = parse_tsv(bytes, &PARTITIONED_HEADER, "partitioned edges")?;
    let mut edge_ids = BTreeSet::new();
    let mut edges = Vec::with_capacity(rows.len());

    for row in rows {
        if !edge_ids.insert(row[0].to_owned()) {
            return Err(format!(
                "partitioned edges: duplicate edge_id {}",
                row[0]
            ));
        }
        if row[3] != "internal" && row[3] != "bridge" {
            return Err(format!(
                "partitioned edges: representation for {} must be internal or bridge",
                row[0]
            ));
        }
        edges.push(PartitionedEdge {
            edge: Edge {
                edge_id: row[0].to_owned(),
                src_node_id: row[1].to_owned(),
                dst_node_id: row[2].to_owned(),
            },
            representation: row[3].to_owned(),
        });
    }
    edges.sort_by(|left, right| left.edge.edge_id.cmp(&right.edge.edge_id));
    Ok(edges)
}

fn parse_node_partitions(bytes: &[u8]) -> Result<BTreeMap<String, String>, String> {
    let rows = parse_tsv(bytes, &PARTITION_HEADER, "node partitions")?;
    let mut partitions = BTreeMap::new();

    for row in rows {
        if partitions
            .insert(row[0].to_owned(), row[1].to_owned())
            .is_some()
        {
            return Err(format!(
                "node partitions: duplicate node_id {}",
                row[0]
            ));
        }
    }
    Ok(partitions)
}

fn parse_witness_paths(bytes: &[u8]) -> Result<BTreeMap<String, Vec<WitnessStep>>, String> {
    let rows = parse_tsv(bytes, &WITNESS_HEADER, "witness paths")?;
    let mut grouped: BTreeMap<String, BTreeMap<usize, WitnessStep>> = BTreeMap::new();

    for row in rows {
        let sequence_index = row[1]
            .parse::<usize>()
            .map_err(|error| format!("witness path {} has invalid sequence_index: {error}", row[0]))?;
        let step = WitnessStep {
            path_id: row[0].to_owned(),
            sequence_index,
            edge: Edge {
                edge_id: row[2].to_owned(),
                src_node_id: row[3].to_owned(),
                dst_node_id: row[4].to_owned(),
            },
        };
        if grouped
            .entry(step.path_id.clone())
            .or_default()
            .insert(sequence_index, step)
            .is_some()
        {
            return Err(format!(
                "witness paths: duplicate sequence_index {sequence_index} for path {}",
                row[0]
            ));
        }
    }

    let mut paths = BTreeMap::new();
    for (path_id, indexed_steps) in grouped {
        let mut steps = Vec::with_capacity(indexed_steps.len());
        for (expected_index, (actual_index, step)) in indexed_steps.into_iter().enumerate() {
            if actual_index != expected_index {
                return Err(format!(
                    "witness path {path_id} has non-contiguous sequence_index; expected {expected_index}, got {actual_index}"
                ));
            }
            if let Some(previous) = steps.last() {
                let previous: &WitnessStep = previous;
                if previous.edge.dst_node_id != step.edge.src_node_id {
                    return Err(format!(
                        "witness path {path_id} is disconnected between edges {} and {}",
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

    Ok(paths)
}

fn load_inputs(args: &Args) -> Result<(Inputs, InputHashes), String> {
    let original_bytes = read_regular_file(&args.original_edges, "original edges")?;
    let partitioned_bytes = read_regular_file(&args.partitioned_edges, "partitioned edges")?;
    let partition_bytes = read_regular_file(&args.node_partitions, "node partitions")?;
    let witness_bytes = read_regular_file(&args.witness_paths, "witness paths")?;

    let hashes = InputHashes {
        original_edges: sha256_reader(original_bytes.as_slice())?,
        partitioned_edges: sha256_reader(partitioned_bytes.as_slice())?,
        node_partitions: sha256_reader(partition_bytes.as_slice())?,
        witness_paths: sha256_reader(witness_bytes.as_slice())?,
    };

    let inputs = Inputs {
        original_edges: parse_original_edges(&original_bytes)?,
        partitioned_edges: parse_partitioned_edges(&partitioned_bytes)?,
        node_partitions: parse_node_partitions(&partition_bytes)?,
        witness_paths: parse_witness_paths(&witness_bytes)?,
    };
    Ok((inputs, hashes))
}

fn partition_for<'a>(
    partitions: &'a BTreeMap<String, String>,
    node_id: &str,
    context: &str,
) -> Result<&'a str, String> {
    partitions
        .get(node_id)
        .map(String::as_str)
        .ok_or_else(|| format!("{context}: node {node_id} has no partition assignment"))
}

fn validate_partitioned_edges(inputs: &Inputs) -> Result<BTreeMap<String, PartitionedEdge>, String> {
    let original = inputs
        .original_edges
        .iter()
        .map(|edge| (edge.edge_id.clone(), edge))
        .collect::<BTreeMap<_, _>>();
    let mut partitioned = BTreeMap::new();

    for item in &inputs.partitioned_edges {
        let original_edge = original.get(&item.edge.edge_id).ok_or_else(|| {
            format!(
                "partitioned graph contains unknown edge_id {}",
                item.edge.edge_id
            )
        })?;
        if **original_edge != item.edge {
            return Err(format!(
                "partitioned edge {} does not match original endpoints",
                item.edge.edge_id
            ));
        }

        let src_partition = partition_for(
            &inputs.node_partitions,
            &item.edge.src_node_id,
            &format!("partitioned edge {}", item.edge.edge_id),
        )?;
        let dst_partition = partition_for(
            &inputs.node_partitions,
            &item.edge.dst_node_id,
            &format!("partitioned edge {}", item.edge.edge_id),
        )?;
        let expected = if src_partition == dst_partition {
            "internal"
        } else {
            "bridge"
        };
        if item.representation != expected {
            return Err(format!(
                "partitioned edge {} crosses {} -> {} and must be represented as {expected}",
                item.edge.edge_id, src_partition, dst_partition
            ));
        }
        partitioned.insert(item.edge.edge_id.clone(), item.clone());
    }

    Ok(partitioned)
}

fn audit_paths(inputs: &Inputs) -> Result<serde_json::Value, String> {
    let original = inputs
        .original_edges
        .iter()
        .map(|edge| (edge.edge_id.clone(), edge))
        .collect::<BTreeMap<_, _>>();
    let partitioned = validate_partitioned_edges(inputs)?;

    for edge in &inputs.original_edges {
        partition_for(
            &inputs.node_partitions,
            &edge.src_node_id,
            &format!("original edge {}", edge.edge_id),
        )?;
        partition_for(
            &inputs.node_partitions,
            &edge.dst_node_id,
            &format!("original edge {}", edge.edge_id),
        )?;
    }

    let mut total_witness_edges = 0_usize;
    let mut preserved_witness_edges = 0_usize;
    let mut lost_witness_edges = Vec::new();
    let mut path_results = Vec::new();
    let mut total_cross_partition_witness_edges = 0_usize;
    let mut preserved_cross_partition_witness_edges = 0_usize;

    for (path_id, steps) in &inputs.witness_paths {
        let mut path_lost = Vec::new();
        let mut path_cross_partition_edges = 0_usize;
        let mut path_preserved_cross_partition_edges = 0_usize;

        for step in steps {
            total_witness_edges += 1;
            let original_edge = original.get(&step.edge.edge_id).ok_or_else(|| {
                format!(
                    "witness path {path_id} references edge {} absent from original graph",
                    step.edge.edge_id
                )
            })?;
            if **original_edge != step.edge {
                return Err(format!(
                    "witness path {path_id} edge {} endpoints disagree with original graph",
                    step.edge.edge_id
                ));
            }

            let src_partition = partition_for(
                &inputs.node_partitions,
                &step.edge.src_node_id,
                &format!("witness path {path_id}"),
            )?;
            let dst_partition = partition_for(
                &inputs.node_partitions,
                &step.edge.dst_node_id,
                &format!("witness path {path_id}"),
            )?;
            let crosses_partition = src_partition != dst_partition;
            if crosses_partition {
                total_cross_partition_witness_edges += 1;
                path_cross_partition_edges += 1;
            }

            if let Some(partitioned_edge) = partitioned.get(&step.edge.edge_id) {
                preserved_witness_edges += 1;
                if crosses_partition {
                    if partitioned_edge.representation != "bridge" {
                        return Err(format!(
                            "witness path {path_id} cross-partition edge {} is not a bridge",
                            step.edge.edge_id
                        ));
                    }
                    preserved_cross_partition_witness_edges += 1;
                    path_preserved_cross_partition_edges += 1;
                }
            } else {
                let missing = serde_json::json!({
                    "path_id": path_id,
                    "sequence_index": step.sequence_index,
                    "edge_id": step.edge.edge_id,
                    "src_node_id": step.edge.src_node_id,
                    "dst_node_id": step.edge.dst_node_id,
                    "src_partition": src_partition,
                    "dst_partition": dst_partition,
                    "crosses_partition": crosses_partition,
                });
                path_lost.push(missing.clone());
                lost_witness_edges.push(missing);
            }
        }

        path_results.push(serde_json::json!({
            "path_id": path_id,
            "edge_count": steps.len(),
            "preserved_edge_count": steps.len() - path_lost.len(),
            "lost_edge_count": path_lost.len(),
            "cross_partition_edge_count": path_cross_partition_edges,
            "preserved_cross_partition_edge_count": path_preserved_cross_partition_edges,
            "status": if path_lost.is_empty() { "preserved" } else { "lost" },
            "lost_edges": path_lost,
        }));
    }

    let original_cross_partition_edges = inputs
        .original_edges
        .iter()
        .filter(|edge| {
            inputs.node_partitions.get(&edge.src_node_id)
                != inputs.node_partitions.get(&edge.dst_node_id)
        })
        .count();
    let partitioned_cross_partition_edges = inputs
        .partitioned_edges
        .iter()
        .filter(|edge| edge.representation == "bridge")
        .count();

    Ok(serde_json::json!({
        "result": if lost_witness_edges.is_empty() { "pass" } else { "fail" },
        "claim_scope": CLAIM_SCOPE,
        "original_edge_count": inputs.original_edges.len(),
        "partitioned_edge_count": inputs.partitioned_edges.len(),
        "partition_count": inputs.node_partitions.values().collect::<BTreeSet<_>>().len(),
        "witness_path_count": inputs.witness_paths.len(),
        "total_witness_edges": total_witness_edges,
        "preserved_witness_edges": preserved_witness_edges,
        "lost_witness_edge_count": lost_witness_edges.len(),
        "total_cross_partition_witness_edges": total_cross_partition_witness_edges,
        "preserved_cross_partition_witness_edges": preserved_cross_partition_witness_edges,
        "original_cross_partition_edge_count": original_cross_partition_edges,
        "partitioned_cross_partition_bridge_count": partitioned_cross_partition_edges,
        "path_results": path_results,
        "lost_witness_edges": lost_witness_edges,
    }))
}

fn receipt_target(inputs: &[&Path], receipt: &Path) -> Result<PathBuf, String> {
    let filename = receipt
        .file_name()
        .ok_or("receipt path must include a filename")?;
    let parent = receipt
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let parent = fs::canonicalize(parent)
        .map_err(|error| format!("canonicalize receipt parent {}: {error}", parent.display()))?;
    let target = parent.join(filename);

    for input in inputs {
        let canonical = fs::canonicalize(input)
            .map_err(|error| format!("canonicalize input {}: {error}", input.display()))?;
        if target == canonical {
            return Err("receipt path must not overwrite an input file".to_owned());
        }
    }
    Ok(target)
}

fn run(args: Args) -> Result<(), String> {
    let (inputs, hashes) = load_inputs(&args)?;
    let audit = audit_paths(&inputs)?;
    let result = audit["result"]
        .as_str()
        .ok_or("internal audit result missing")?;
    let receipt_path = receipt_target(
        &[
            &args.original_edges,
            &args.partitioned_edges,
            &args.node_partitions,
            &args.witness_paths,
        ],
        &args.receipt,
    )?;
    let receipt = serde_json::json!({
        "schema_version": "hardsec-partition-path-audit-v1",
        "result": result,
        "claim_scope": CLAIM_SCOPE,
        "input_sha256": {
            "original_edges": hashes.original_edges,
            "partitioned_edges": hashes.partitioned_edges,
            "node_partitions": hashes.node_partitions,
            "witness_paths": hashes.witness_paths,
        },
        "input_record_counts": {
            "original_edges": inputs.original_edges.len(),
            "partitioned_edges": inputs.partitioned_edges.len(),
            "node_partition_assignments": inputs.node_partitions.len(),
            "witness_paths": inputs.witness_paths.len(),
        },
        "audit": audit,
        "signature": null,
        "security_verdict": null,
        "notes": [
            "Only the supplied witness paths were checked.",
            "No claim is made that unprovided trigger-to-payload paths were discovered or preserved.",
            "Every retained cross-partition witness edge is represented as an explicit bridge.",
            "A pass is not a detector-quality result, Trojan absence claim, or design-security certification."
        ]
    });

    let serialized = serde_json::to_vec_pretty(&receipt)
        .map_err(|error| format!("serialize path audit receipt: {error}"))?;
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&receipt_path)
        .map_err(|error| format!("create new receipt {}: {error}", receipt_path.display()))?;
    let write_result = output
        .write_all(&serialized)
        .and_then(|_| output.write_all(b"\n"))
        .and_then(|_| output.sync_all());
    drop(output);
    if let Err(error) = write_result {
        let _ = fs::remove_file(&receipt_path);
        return Err(format!("write receipt {}: {error}", receipt_path.display()));
    }

    println!("HARDSEC-010 partition path audit: {result}");
    println!("witness paths: {}", inputs.witness_paths.len());
    println!("preserved witness edges: {}", audit["preserved_witness_edges"]);
    println!("lost witness edges: {}", audit["lost_witness_edge_count"]);
    println!("receipt: {}", receipt_path.display());
    println!("claim scope: {}", CLAIM_SCOPE);

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
                "usage: hardsec_partition_path_audit --original-edges FILE --partitioned-edges FILE --node-partitions FILE --witness-paths FILE --receipt NEW_FILE"
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
            eprintln!("HARDSEC-010 path audit failed: {error}");
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
        original_edges: PathBuf,
        partitioned_edges: PathBuf,
        node_partitions: PathBuf,
        witness_paths: PathBuf,
        receipt: PathBuf,
    }

    impl Fixture {
        fn new(partitioned_rows: &str) -> Self {
            let id = NEXT.fetch_add(1, Ordering::Relaxed);
            let root = env::temp_dir().join(format!(
                "hardsec-path-audit-{}-{id}",
                std::process::id()
            ));
            fs::create_dir_all(&root).unwrap();

            let original_edges = root.join("original-edges.tsv");
            let partitioned_edges = root.join("partitioned-edges.tsv");
            let node_partitions = root.join("node-partitions.tsv");
            let witness_paths = root.join("witness-paths.tsv");
            let receipt = root.join("output/path-audit.json");

            fs::create_dir_all(receipt.parent().unwrap()).unwrap();
            fs::write(
                &original_edges,
                concat!(
                    "edge_id\tsrc_node_id\tdst_node_id\n",
                    "e1\ta\tb\n",
                    "e2\tb\tc\n",
                    "e3\tc\td\n"
                ),
            )
            .unwrap();
            fs::write(
                &partitioned_edges,
                partitioned_rows,
            )
            .unwrap();
            fs::write(
                &node_partitions,
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
                &witness_paths,
                concat!(
                    "path_id\tsequence_index\tedge_id\tsrc_node_id\tdst_node_id\n",
                    "w1\t0\te1\ta\tb\n",
                    "w1\t1\te2\tb\tc\n",
                    "w1\t2\te3\tc\td\n"
                ),
            )
            .unwrap();

            Self {
                root,
                original_edges,
                partitioned_edges,
                node_partitions,
                witness_paths,
                receipt,
            }
        }

        fn args(&self) -> Args {
            Args {
                original_edges: self.original_edges.clone(),
                partitioned_edges: self.partitioned_edges.clone(),
                node_partitions: self.node_partitions.clone(),
                witness_paths: self.witness_paths.clone(),
                receipt: self.receipt.clone(),
            }
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.root);
        }
    }

    const COMPLETE_PARTITIONED: &str = concat!(
        "edge_id\tsrc_node_id\tdst_node_id\trepresentation\n",
        "e1\ta\tb\tinternal\n",
        "e2\tb\tc\tbridge\n",
        "e3\tc\td\tinternal\n"
    );

    #[test]
    fn accepts_path_with_explicit_cross_partition_bridge() {
        let fixture = Fixture::new(COMPLETE_PARTITIONED);
        run(fixture.args()).unwrap();
        let receipt: serde_json::Value =
            serde_json::from_slice(&fs::read(&fixture.receipt).unwrap()).unwrap();
        assert_eq!(receipt["result"], "pass");
        assert_eq!(receipt["audit"]["preserved_witness_edges"], 3);
        assert_eq!(
            receipt["audit"]["preserved_cross_partition_witness_edges"],
            1
        );
    }

    #[test]
    fn writes_fail_receipt_and_rejects_lost_cross_partition_witness_edge() {
        let partitioned = concat!(
            "edge_id\tsrc_node_id\tdst_node_id\trepresentation\n",
            "e1\ta\tb\tinternal\n",
            "e3\tc\td\tinternal\n"
        );
        let fixture = Fixture::new(partitioned);
        let error = run(fixture.args()).unwrap_err();
        assert!(error.contains("witness paths were severed"));
        let receipt: serde_json::Value =
            serde_json::from_slice(&fs::read(&fixture.receipt).unwrap()).unwrap();
        assert_eq!(receipt["result"], "fail");
        assert_eq!(receipt["audit"]["lost_witness_edge_count"], 1);
        assert_eq!(
            receipt["audit"]["lost_witness_edges"][0]["edge_id"],
            "e2"
        );
    }

    #[test]
    fn rejects_cross_partition_edge_mislabeled_as_internal() {
        let partitioned = concat!(
            "edge_id\tsrc_node_id\tdst_node_id\trepresentation\n",
            "e1\ta\tb\tinternal\n",
            "e2\tb\tc\tinternal\n",
            "e3\tc\td\tinternal\n"
        );
        let fixture = Fixture::new(partitioned);
        let error = run(fixture.args()).unwrap_err();
        assert!(error.contains("must be represented as bridge"));
    }

    #[test]
    fn rejects_partitioned_edge_not_in_original_graph() {
        let partitioned = concat!(
            "edge_id\tsrc_node_id\tdst_node_id\trepresentation\n",
            "e1\ta\tb\tinternal\n",
            "e2\tb\tx\tbridge\n",
            "e3\tc\td\tinternal\n"
        );
        let fixture = Fixture::new(partitioned);
        let error = run(fixture.args()).unwrap_err();
        assert!(error.contains("does not match original endpoints"));
    }

    #[test]
    fn rejects_disconnected_witness_sequence() {
        let fixture = Fixture::new(COMPLETE_PARTITIONED);
        let text = fs::read_to_string(&fixture.witness_paths)
            .unwrap()
            .replace("w1\t1\te2\tb\tc", "w1\t1\te2\tx\tc");
        fs::write(&fixture.witness_paths, text).unwrap();
        let error = run(fixture.args()).unwrap_err();
        assert!(error.contains("is disconnected"));
    }

    #[test]
    fn rejects_non_contiguous_witness_sequence_indices() {
        let fixture = Fixture::new(COMPLETE_PARTITIONED);
        let text = fs::read_to_string(&fixture.witness_paths)
            .unwrap()
            .replace("w1\t2\te3", "w1\t3\te3");
        fs::write(&fixture.witness_paths, text).unwrap();
        let error = run(fixture.args()).unwrap_err();
        assert!(error.contains("non-contiguous sequence_index"));
    }

    #[test]
    fn rejects_duplicate_partition_edge_ids() {
        let partitioned = concat!(
            "edge_id\tsrc_node_id\tdst_node_id\trepresentation\n",
            "e1\ta\tb\tinternal\n",
            "e2\tb\tc\tbridge\n",
            "e2\tb\tc\tbridge\n"
        );
        let fixture = Fixture::new(partitioned);
        let error = run(fixture.args()).unwrap_err();
        assert!(error.contains("duplicate edge_id"));
    }

    #[test]
    fn rejects_missing_node_partition_assignment() {
        let fixture = Fixture::new(COMPLETE_PARTITIONED);
        let text = fs::read_to_string(&fixture.node_partitions)
            .unwrap()
            .replace("c\tp2\n", "");
        fs::write(&fixture.node_partitions, text).unwrap();
        let error = run(fixture.args()).unwrap_err();
        assert!(error.contains("has no partition assignment"));
    }

    #[test]
    fn refuses_to_overwrite_a_receipt() {
        let fixture = Fixture::new(COMPLETE_PARTITIONED);
        fs::write(&fixture.receipt, b"preserve").unwrap();
        let error = run(fixture.args()).unwrap_err();
        assert!(error.contains("File exists"));
    }
}
