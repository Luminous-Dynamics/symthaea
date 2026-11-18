use jsonschema::validator_for;
use serde_json::{Value, json};
use sha1::{Digest as Sha1Digest, Sha1};
use sha2::Sha256;
use std::collections::HashSet;
use std::env;
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::process;

fn required_text<'a>(value: &'a Value, key: &str) -> Result<&'a str, String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .filter(|text| !text.trim().is_empty())
        .ok_or_else(|| format!("missing or empty field {key}"))
}

fn sha1_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha1::new();
    hasher.update(format!("blob {}\0", bytes.len()).as_bytes());
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn safe_relative_path(value: &str) -> Result<PathBuf, String> {
    let path = Path::new(value);
    if path.is_absolute() {
        return Err(format!("manifest path must be relative: {value}"));
    }
    for component in path.components() {
        match component {
            Component::Normal(_) => {}
            _ => return Err(format!("manifest path contains unsafe component: {value}")),
        }
    }
    Ok(path.to_path_buf())
}

fn verify_local_blob(root: &Path, relative_path: &str, expected_sha: &str) -> Result<Value, String> {
    let relative = safe_relative_path(relative_path)?;
    let full_path = root.join(&relative);
    let bytes = fs::read(&full_path)
        .map_err(|error| format!("could not read {}: {error}", full_path.display()))?;
    let actual_sha = sha1_hex(&bytes);
    if actual_sha != expected_sha {
        return Err(format!(
            "Git blob SHA-1 mismatch for {relative_path}: expected {expected_sha}, got {actual_sha}"
        ));
    }
    Ok(json!({
        "path": relative_path,
        "expected_git_blob_sha1": expected_sha,
        "actual_git_blob_sha1": actual_sha,
        "byte_count": bytes.len(),
        "status": "verified"
    }))
}

fn audit_manifest(
    schema_bytes: &[u8],
    manifest_bytes: &[u8],
    root: Option<&Path>,
) -> Result<Value, String> {
    let schema: Value = serde_json::from_slice(schema_bytes)
        .map_err(|error| format!("schema JSON parse failed: {error}"))?;
    let manifest: Value = serde_json::from_slice(manifest_bytes)
        .map_err(|error| format!("manifest JSON parse failed: {error}"))?;
    const DIALECT: &str = "https://json-schema.org/draft/2020-12/schema";
    if schema.get("$schema").and_then(Value::as_str) != Some(DIALECT) {
        return Err(format!("schema must declare {DIALECT}"));
    }
    let validator =
        validator_for(&schema).map_err(|error| format!("schema compilation failed: {error}"))?;
    if let Some(error) = validator.iter_errors(&manifest).next() {
        return Err(format!(
            "SCTK replay manifest schema validation failed: {} at {}",
            error,
            error.instance_path()
        ));
    }
    if manifest["contract"] != "SCDFT-SCTK-REPLAY-0.1" {
        return Err("unsupported SCTK replay contract".to_owned());
    }

    let upstream = &manifest["upstream"];
    let upstream_commit = required_text(upstream, "commit_sha")?;
    if required_text(upstream, "ref_name")? != "main" {
        return Err("initial fixture is pinned to the upstream main source snapshot".to_owned());
    }
    if required_text(&manifest, "example_scope")? != "contributed_legacy_pb_example" {
        return Err("unexpected example scope".to_owned());
    }
    let physics = &manifest["physics_scope"];
    if physics["spin_orbit_included"].as_bool() != Some(false)
        || physics["finite_q_method"].as_bool() != Some(false)
        || physics["not_equivalent_to_2026_prl_fixture"].as_bool() != Some(true)
    {
        return Err(
            "legacy Pb example must remain no-SOI, non-finite-Q and separate from 2026 PRL data"
                .to_owned(),
        );
    }

    let cases = manifest["cases"]
        .as_array()
        .ok_or_else(|| "cases must be an array".to_owned())?;
    if cases.len() != 2 {
        return Err("exactly two Pb comparison cases are required".to_owned());
    }
    let mut case_ids = HashSet::new();
    for case in cases {
        let id = required_text(case, "case_id")?;
        if !case_ids.insert(id.to_owned()) {
            return Err(format!("duplicate SCTK case ID {id}"));
        }
        let spin = case["spin_fluctuations"]
            .as_bool()
            .ok_or_else(|| format!("case {id} spin_fluctuations must be boolean"))?;
        if (id == "with_spin_fluctuations") != spin {
            return Err(format!("case {id} has inconsistent spin-fluctuation flag"));
        }
        let reference_tc = case["reference_tc_k"]
            .as_f64()
            .ok_or_else(|| format!("case {id} missing reference_tc_k"))?;
        let README_tc = case["README_expectation_k"]
            .as_f64()
            .ok_or_else(|| format!("case {id} missing README expectation"))?;
        if (reference_tc - README_tc).abs() > 0.1 {
            return Err(format!("case {id} reference Tc does not agree with README-level expectation"));
        }
        if case["reference_output_complete"].as_bool() != Some(true) {
            return Err(format!("case {id} reference output not marked complete"));
        }
    }
    if !case_ids.contains("with_spin_fluctuations")
        || !case_ids.contains("without_spin_fluctuations")
    {
        return Err("both with-SF and without-SF cases are mandatory".to_owned());
    }

    let files = manifest["files"]
        .as_array()
        .ok_or_else(|| "files must be an array".to_owned())?;
    let mut paths = HashSet::new();
    let mut has_readme = false;
    let mut has_pseudopotential = false;
    for file in files {
        let path = required_text(file, "path")?;
        let _ = safe_relative_path(path)?;
        if !paths.insert(path.to_owned()) {
            return Err(format!("duplicate file path in manifest: {path}"));
        }
        let role = required_text(file, "role")?;
        if role == "readme" {
            has_readme = true;
        }
        if role == "pseudopotential" {
            has_pseudopotential = true;
        }
        let _ = required_text(file, "git_blob_sha1")?;
    }
    if !has_readme || !has_pseudopotential {
        return Err("manifest requires README and pseudopotential source identity".to_owned());
    }
    for case in cases {
        let tc_path = required_text(case, "tc_output_path")?;
        let expected = required_text(case, "tc_output_blob_sha")?;
        let matched = files.iter().any(|file| {
            file["path"].as_str() == Some(tc_path)
                && file["git_blob_sha1"].as_str() == Some(expected)
                && file["role"].as_str() == Some("reference_output")
        });
        if !matched {
            return Err(format!(
                "case {} TC reference is not bound to a matching file blob entry",
                required_text(case, "case_id")?
            ));
        }
    }

    let mut verified_files = Vec::new();
    let mut local_status = "not_materialized_manifest_only";
    if let Some(root) = root {
        for file in files {
            let path = required_text(file, "path")?;
            let expected = required_text(file, "git_blob_sha1")?;
            verified_files.push(verify_local_blob(root, path, expected)?);
        }
        local_status = "all_listed_local_git_blobs_verified";
    }

    let case_summaries: Vec<Value> = cases
        .iter()
        .map(|case| {
            json!({
                "case_id": case["case_id"],
                "spin_fluctuations": case["spin_fluctuations"],
                "reference_tc_k": case["reference_tc_k"],
                "README_expectation_k": case["README_expectation_k"],
                "converged_iterations": case["converged_iterations"],
                "reference_run_date": case["reference_run_date"],
                "reference_tc_output_git_blob_sha1": case["tc_output_blob_sha"]
            })
        })
        .collect();

    Ok(json!({
        "status": "pass",
        "contract": "SCDFT-SCTK-REPLAY-AUDIT-0.1",
        "manifest_id": required_text(&manifest, "manifest_id")?,
        "manifest_sha256": sha256_hex(manifest_bytes),
        "schema_sha256": sha256_hex(schema_bytes),
        "upstream": {
            "repository_url": upstream["repository_url"],
            "commit_sha": upstream_commit,
            "ref_name": upstream["ref_name"]
        },
        "file_count": files.len(),
        "local_git_blob_verification": local_status,
        "verified_local_files": verified_files,
        "physics_scope": physics,
        "cases": case_summaries,
        "reference_output_integrity": "upstream Git blob identities are pinned; local bytes are not verified unless --root is supplied",
        "execution_status": "not_executed",
        "claim_ceiling": "Source-pinned reference manifest only. This audit does not execute Quantum ESPRESSO/SCTK and does not claim an independent SCDFT reproduction."
    }))
}

fn read(path: &PathBuf) -> Vec<u8> {
    fs::read(path).unwrap_or_else(|error| {
        eprintln!("could not read {}: {error}", path.display());
        process::exit(1);
    })
}

fn usage(program: &str) {
    eprintln!(
        "Usage: {program} --schema <schema.json> --manifest <manifest.json> [--root <local-upstream-checkout>] [--out <receipt.json>]"
    );
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut schema_path = None;
    let mut manifest_path = None;
    let mut root_path = None;
    let mut output_path = None;
    let mut index = 1;
    while index < args.len() {
        match args[index].as_str() {
            "--schema" | "--manifest" | "--root" | "--out" => {
                if index + 1 >= args.len() {
                    usage(&args[0]);
                    process::exit(2);
                }
                let path = PathBuf::from(&args[index + 1]);
                match args[index].as_str() {
                    "--schema" => schema_path = Some(path),
                    "--manifest" => manifest_path = Some(path),
                    "--root" => root_path = Some(path),
                    "--out" => output_path = Some(path),
                    _ => unreachable!(),
                }
                index += 2;
            }
            "--help" | "-h" => {
                usage(&args[0]);
                return;
            }
            unknown => {
                eprintln!("unknown argument: {unknown}");
                usage(&args[0]);
                process::exit(2);
            }
        }
    }
    let (Some(schema_path), Some(manifest_path)) = (schema_path, manifest_path) else {
        usage(&args[0]);
        process::exit(2);
    };
    let schema_bytes = read(&schema_path);
    let manifest_bytes = read(&manifest_path);
    let receipt = audit_manifest(
        &schema_bytes,
        &manifest_bytes,
        root_path.as_deref(),
    )
    .unwrap_or_else(|error| {
        eprintln!("SCDFT-SCTK-REPLAY-FAIL: {error}");
        process::exit(1);
    });
    let rendered = serde_json::to_string_pretty(&receipt)
        .expect("serializing SCTK replay audit receipt should not fail");
    if let Some(path) = output_path {
        if let Err(error) = fs::write(&path, format!("{rendered}\n")) {
            eprintln!("could not write {}: {error}", path.display());
            process::exit(1);
        }
    } else {
        println!("{rendered}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn inputs() -> (Vec<u8>, Vec<u8>) {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        let schema = fs::read(
            root.join("data/schemas/scdft_sctk_replay_manifest_v0_1.schema.json"),
        )
        .unwrap();
        let manifest = fs::read(
            root.join("data/research/scdft_sctk_pb_legacy_replay_manifest_v0_1.json"),
        )
        .unwrap();
        (schema, manifest)
    }

    #[test]
    fn manifest_passes_without_claiming_local_replay() {
        let (schema, manifest) = inputs();
        let receipt = audit_manifest(&schema, &manifest, None).unwrap();
        assert_eq!(receipt["status"], "pass");
        assert_eq!(receipt["file_count"], 26);
        assert_eq!(receipt["execution_status"], "not_executed");
        assert_eq!(
            receipt["local_git_blob_verification"],
            "not_materialized_manifest_only"
        );
        assert_eq!(
            receipt["cases"][0]["reference_tc_k"].as_f64().unwrap(),
            4.4251
        );
        assert_eq!(
            receipt["cases"][1]["reference_tc_k"].as_f64().unwrap(),
            3.7233
        );
    }

    #[test]
    fn git_blob_sha1_matches_known_git_object() {
        assert_eq!(
            sha1_hex(b"abc"),
            "f2ba8f84ab5c1bce84a7b441cb1959cfc7093b7f"
        );
    }

    #[test]
    fn local_blob_verifier_accepts_exact_content_and_rejects_tampering() {
        let root = env::temp_dir().join(format!(
            "sctk-replay-blob-test-{}",
            process::id()
        ));
        fs::create_dir_all(&root).unwrap();
        let path = root.join("content.txt");
        fs::write(&path, b"abc").unwrap();
        assert!(verify_local_blob(
            &root,
            "content.txt",
            "f2ba8f84ab5c1bce84a7b441cb1959cfc7093b7f"
        ).is_ok());
        fs::write(&path, b"abd").unwrap();
        assert!(verify_local_blob(
            &root,
            "content.txt",
            "f2ba8f84ab5c1bce84a7b441cb1959cfc7093b7f"
        ).is_err());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn verifier_rejects_path_traversal() {
        assert!(safe_relative_path("../outside").is_err());
        assert!(safe_relative_path("/absolute/path").is_err());
    }

    #[test]
    fn manifest_rejects_a_spin_orbit_claim() {
        let (schema, bytes) = inputs();
        let mut manifest: Value = serde_json::from_slice(&bytes).unwrap();
        manifest["physics_scope"]["spin_orbit_included"] = json!(true);
        assert!(audit_manifest(&schema, &serde_json::to_vec(&manifest).unwrap(), None).is_err());
    }

    #[test]
    fn manifest_rejects_relabelling_the_legacy_example_as_finite_q() {
        let (schema, bytes) = inputs();
        let mut manifest: Value = serde_json::from_slice(&bytes).unwrap();
        manifest["physics_scope"]["finite_q_method"] = json!(true);
        assert!(audit_manifest(&schema, &serde_json::to_vec(&manifest).unwrap(), None).is_err());
    }

    #[test]
    fn manifest_rejects_cross_case_tc_output_binding() {
        let (schema, bytes) = inputs();
        let mut manifest: Value = serde_json::from_slice(&bytes).unwrap();
        manifest["cases"][0]["tc_output_blob_sha"] =
            manifest["cases"][1]["tc_output_blob_sha"].clone();
        assert!(audit_manifest(&schema, &serde_json::to_vec(&manifest).unwrap(), None).is_err());
    }
}
