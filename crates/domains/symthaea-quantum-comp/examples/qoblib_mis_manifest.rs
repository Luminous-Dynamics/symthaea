//! Hashes and canonicalizes the pinned QOBLIB MIS source set.
//!
//! Usage:
//! `cargo run -p symthaea-quantum-comp --example qoblib_mis_manifest -- <qoblib-repository-root> <pinned-files.tsv> <output.tsv>`
//!
//! The QOBLIB checkout must be at the exact commit embedded in the TSV. This tool
//! validates Git blob SHA-1 and byte size, then computes SHA-256 over the source
//! bytes and over the Symthaea canonical graph bytes. Git SHA-1 is used only to
//! check that the local source file matches the pinned Git tree object.

use std::collections::HashSet;
use std::error::Error;
use std::fs;
use std::path::Path;
use std::process::Command;

#[cfg(test)]
#[path = "../src/mis_evidence.rs"]
mod standalone_mis_evidence;

#[cfg(test)]
use standalone_mis_evidence::{canonicalize_dimacs_graph, sha256_hex};

#[cfg(not(test))]
use symthaea_quantum_comp::{canonicalize_dimacs_graph, sha256_hex};

#[derive(Debug)]
struct PinnedFile {
    name: String,
    relative_path: String,
    git_blob_sha1: String,
    size_bytes: u64,
    expected_vertices: usize,
    expected_edges: usize,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("qoblib-mis-manifest: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        return Err(
            "usage: qoblib_mis_manifest <qoblib-repository-root> <pinned-files.tsv> <output.tsv>"
                .into(),
        );
    }
    let repository_root = Path::new(&args[1]);
    let pinned_manifest_path = Path::new(&args[2]);
    let output_path = Path::new(&args[3]);
    if !repository_root.is_dir() {
        return Err(format!("QOBLIB root is not a directory: {}", repository_root.display()).into());
    }

    let manifest_text = fs::read_to_string(pinned_manifest_path)?;
    let (pinned_commit, pinned_files) = parse_pinned_manifest(&manifest_text)?;
    let actual_commit = run_git(repository_root, &["rev-parse", "HEAD"])?;
    if actual_commit != pinned_commit {
        return Err(format!(
            "QOBLIB checkout revision mismatch: expected {pinned_commit}, got {actual_commit}"
        )
        .into());
    }
    if pinned_files.len() != 50 {
        return Err(format!("expected 50 pinned MIS instances, got {}", pinned_files.len()).into());
    }

    verify_instance_directory_inventory(repository_root, &pinned_files)?;

    let mut output = String::from(
        "# schema=SYMTHEA-QOBLIB-MIS-MANIFEST-V1\n# source_repo=ZIB-AOPT/QOBLIB\n# output_columns=instance_name<TAB>source_git_blob_sha1<TAB>source_size_bytes<TAB>source_sha256<TAB>canonical_graph_sha256<TAB>vertices<TAB>edges<TAB>canonical_size_bytes\n",
    );

    for pinned in &pinned_files {
        let relative = Path::new(&pinned.relative_path);
        if relative.is_absolute()
            || relative.components().any(|component| matches!(component, std::path::Component::ParentDir))
            || !pinned.relative_path.starts_with("07-independentset/instances/")
        {
            return Err(format!("unsafe or unexpected relative path: {}", pinned.relative_path).into());
        }

        let path = repository_root.join(relative);
        let bytes = fs::read(&path)?;
        if bytes.len() as u64 != pinned.size_bytes {
            return Err(format!(
                "{} byte-size mismatch: expected {}, got {}",
                pinned.name,
                pinned.size_bytes,
                bytes.len()
            )
            .into());
        }

        let relative_git_path = pinned.relative_path.as_str();
        let actual_blob = run_git(repository_root, &["hash-object", "--no-filters", relative_git_path])?;
        if actual_blob != pinned.git_blob_sha1 {
            return Err(format!(
                "{} Git blob mismatch: expected {}, got {}",
                pinned.name,
                pinned.git_blob_sha1,
                actual_blob
            )
            .into());
        }

        let source_sha256 = sha256_hex(&bytes);
        let canonical = canonicalize_dimacs_graph(&bytes)?;
        let canonical_sha256 = sha256_hex(&canonical);
        let (vertices, edges) = canonical_graph_counts(&canonical)?;
        if vertices != pinned.expected_vertices || edges != pinned.expected_edges {
            return Err(format!(
                "{} graph-count mismatch: pinned {} vertices / {} edges, parsed {} vertices / {} edges",
                pinned.name, pinned.expected_vertices, pinned.expected_edges, vertices, edges
            ).into());
        }
        output.push_str(&format!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
            pinned.name,
            actual_blob,
            bytes.len(),
            source_sha256,
            canonical_sha256,
            vertices,
            edges,
            canonical.len()
        ));
    }

    if let Some(parent) = output_path.parent().filter(|parent| !parent.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    fs::write(output_path, output)?;
    println!(
        "verified {} pinned MIS source files at {}",
        pinned_files.len(),
        pinned_commit
    );
    println!("wrote SHA-256 and canonical graph inventory to {}", output_path.display());
    Ok(())
}

fn parse_pinned_manifest(text: &str) -> Result<(String, Vec<PinnedFile>), Box<dyn Error>> {
    let mut commit = None;
    let mut files = Vec::new();
    let mut names = HashSet::new();
    let mut paths = HashSet::new();

    for (line_number, line) in text.lines().enumerate() {
        if let Some(value) = line.strip_prefix("# source_commit=") {
            if commit.replace(value.to_owned()).is_some() {
                return Err("pinned manifest declares source_commit more than once".into());
            }
            continue;
        }
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let columns: Vec<&str> = line.split('\t').collect();
        if columns.len() != 6 {
            return Err(format!("pinned manifest line {} must have six tab-separated columns", line_number + 1).into());
        }
        let name = columns[0].to_owned();
        let relative_path = columns[1].to_owned();
        let expected_filename = format!("{name}.gph");
        let actual_filename = Path::new(&relative_path)
            .file_name()
            .and_then(|value| value.to_str())
            .ok_or_else(|| format!("invalid source path on line {}", line_number + 1))?;
        if actual_filename != expected_filename {
            return Err(format!("instance {name} must use filename {expected_filename}").into());
        }
        if !relative_path.starts_with("07-independentset/instances/")
            || Path::new(&relative_path).is_absolute()
            || Path::new(&relative_path).components()
                .any(|component| matches!(component, std::path::Component::ParentDir))
        {
            return Err(format!("unsafe or unexpected source path on line {}", line_number + 1).into());
        }
        let git_blob_sha1 = columns[2].to_owned();
        let size_bytes = columns[3].parse::<u64>()?;
        let expected_vertices = columns[4].parse::<usize>()?;
        let expected_edges = columns[5].parse::<usize>()?;
        if name.is_empty()
            || git_blob_sha1.len() != 40
            || !git_blob_sha1.bytes().all(|byte| byte.is_ascii_hexdigit())
            || expected_vertices == 0
        {
            return Err(format!("invalid name, Git blob SHA-1, or vertex count on line {}", line_number + 1).into());
        }
        if !names.insert(name.clone()) || !paths.insert(relative_path.clone()) {
            return Err(format!("duplicate instance name or path on line {}", line_number + 1).into());
        }
        files.push(PinnedFile {
            name,
            relative_path,
            git_blob_sha1,
            size_bytes,
            expected_vertices,
            expected_edges,
        });
    }

    let commit = commit.ok_or("pinned manifest lacks # source_commit=...")?;
    if commit.len() != 40 || !commit.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err("pinned source commit must be a 40-character Git SHA-1".into());
    }
    if files.is_empty() {
        return Err("pinned manifest contains no graph files".into());
    }
    Ok((commit, files))
}

fn verify_instance_directory_inventory(
    repository_root: &Path,
    pinned_files: &[PinnedFile],
) -> Result<(), Box<dyn Error>> {
    let instance_dir = repository_root.join("07-independentset/instances");
    let actual: HashSet<String> = fs::read_dir(&instance_dir)?
        .filter_map(Result::ok)
        .filter_map(|entry| {
            let path = entry.path();
            (path.extension().and_then(|extension| extension.to_str()) == Some("gph"))
                .then(|| path.file_name()?.to_str().map(str::to_owned))
                .flatten()
        })
        .collect();
    let expected: HashSet<String> = pinned_files
        .iter()
        .map(|file| {
            Path::new(&file.relative_path)
                .file_name()
                .and_then(|name| name.to_str())
                .map(str::to_owned)
                .ok_or_else(|| format!("invalid file path: {}", file.relative_path))
        })
        .collect::<Result<_, _>>()?;
    if actual != expected {
        return Err(format!(
            "QOBLIB .gph inventory differs from pinned manifest: {} actual, {} expected",
            actual.len(),
            expected.len()
        )
        .into());
    }
    Ok(())
}

fn canonical_graph_counts(bytes: &[u8]) -> Result<(usize, usize), Box<dyn Error>> {
    let text = std::str::from_utf8(bytes)?;
    let mut lines = text.lines();
    if lines.next() != Some("SYMTHEA-MIS-GRAPH-V1") {
        return Err("canonical graph header mismatch".into());
    }
    let vertices = lines
        .next()
        .and_then(|line| line.strip_prefix("vertices "))
        .ok_or("canonical graph vertex count missing")?
        .parse::<usize>()?;
    let edges = lines
        .next()
        .and_then(|line| line.strip_prefix("edges "))
        .ok_or("canonical graph edge count missing")?
        .parse::<usize>()?;
    Ok((vertices, edges))
}

fn run_git(repository_root: &Path, args: &[&str]) -> Result<String, Box<dyn Error>> {
    let output = Command::new("git")
        .arg("-C")
        .arg(repository_root)
        .args(args)
        .output()?;
    if !output.status.success() {
        return Err(format!(
            "git {:?} failed: {}",
            args,
            String::from_utf8_lossy(&output.stderr).trim()
        )
        .into());
    }
    Ok(String::from_utf8(output.stdout)?.trim().to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    const COMMIT: &str = "1e0e7c523cd9707dd541cc9e5a3086125d4da3c7";
    const BLOB: &str = "433ad3290665909e1ee5c17abfa8c946391938dc";

    fn line(name: &str, path: &str, blob: &str, size: usize, vertices: usize, edges: usize) -> String {
        format!("{name}\\t{path}\\t{blob}\\t{size}\\t{vertices}\\t{edges}\\n")
    }
    #[test]
    fn parses_pinned_manifest_without_mutating_source_identity() {
        let manifest = format!(
            "# source_commit={COMMIT}\\n# source_repo=ZIB-AOPT/QOBLIB\\n{}",
            line(
                "aves-sparrow-social",
                "07-independentset/instances/aves-sparrow-social.gph",
                BLOB,
                3652,
                52,
                454
            )
        );
        let (commit, entries) = parse_pinned_manifest(&manifest).unwrap();
        assert_eq!(commit, COMMIT);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].name, "aves-sparrow-social");
        assert_eq!(entries[0].git_blob_sha1, BLOB);
        assert_eq!(entries[0].size_bytes, 3652);
    }

    #[test]
    fn rejects_duplicate_instance_names_and_paths() {
        let row = line(
            "aves-sparrow-social",
            "07-independentset/instances/aves-sparrow-social.gph",
            BLOB,
            3652,
            52,
            454
        );
        let manifest = format!("# source_commit={COMMIT}\\n{row}{row}");
        assert!(parse_pinned_manifest(&manifest).is_err());
    }

    #[test]
    fn rejects_pinned_vertex_or_edge_count_mismatch() {
        let wrong_vertices = format!(
            "# source_commit={COMMIT}\\n{}",
            line(
                "aves-sparrow-social",
                "07-independentset/instances/aves-sparrow-social.gph",
                BLOB,
                3652,
                51,
                454
            )
        );
        let (_, entries) = parse_pinned_manifest(&wrong_vertices).unwrap();
        assert_eq!(entries[0].expected_vertices, 51);

        let wrong_edges = format!(
            "# source_commit={COMMIT}\\n{}",
            line(
                "aves-sparrow-social",
                "07-independentset/instances/aves-sparrow-social.gph",
                BLOB,
                3652,
                52,
                453
            )
        );
        let (_, entries) = parse_pinned_manifest(&wrong_edges).unwrap();
        assert_eq!(entries[0].expected_edges, 453);
    }

    #[test]
    fn rejects_wrong_filename_and_malformed_blob_id() {
        let wrong_filename = format!(
            "# source_commit={COMMIT}\\n{}",
            line(
                "aves-sparrow-social",
                "07-independentset/instances/another-name.gph",
                BLOB,
                3652,
                52,
                454
            )
        );
        assert!(parse_pinned_manifest(&wrong_filename).is_err());

        let malformed_blob = format!(
            "# source_commit={COMMIT}\\n{}",
            line(
                "aves-sparrow-social",
                "07-independentset/instances/aves-sparrow-social.gph",
                "deadbeef",
                3652,
                52,
                454
            )
        );
        assert!(parse_pinned_manifest(&malformed_blob).is_err());
    }
}
