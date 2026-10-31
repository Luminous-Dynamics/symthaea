//! Real Cargo build-script challenge for an undeclared runtime input.
//!
//! A warm target directory should retain stale output after the ignored input
//! mutates, while a fresh target directory should replay the script and expose it.
//! This measures Cargo behavior; it does not qualify a dependency observer.

use anyhow::{Context, bail};
use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::env;
use std::ffi::OsStr;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

const CASE_ID: &str = "cargo-build-script-input";
const RESOURCE_ID: &str = "fixture:build.rs-input.txt";
const BEFORE: &[u8] = b"baseline-state\n";
const AFTER: &[u8] = b"mutated-state\n";
const FIXTURE: &[(&str, &str)] = &[
    ("Cargo.toml", include_str!("../../benchmarks/basalt/fixtures/cargo-hidden-build-script/Cargo.toml")),
    ("Cargo.lock", include_str!("../../benchmarks/basalt/fixtures/cargo-hidden-build-script/Cargo.lock")),
    ("build.rs", include_str!("../../benchmarks/basalt/fixtures/cargo-hidden-build-script/build.rs")),
    ("src/main.rs", include_str!("../../benchmarks/basalt/fixtures/cargo-hidden-build-script/src/main.rs")),
    (".gitignore", include_str!("../../benchmarks/basalt/fixtures/cargo-hidden-build-script/.gitignore")),
];

#[derive(Debug, Serialize)]
pub struct CargoFixtureRunReport {
    pub report_version: u32,
    pub run_id: String,
    pub case_id: &'static str,
    pub resource_id: &'static str,
    pub evidence_path: String,
    pub workspace_path: String,
    pub catalog_digest: String,
    pub fixture_digest: String,
    pub cargo_binary_digest: String,
    pub rustc_launcher_digest: String,
    pub rustc_sysroot_binary_digest: String,
    pub rustc_version: String,
    pub rustc_sysroot: String,
    pub baseline_output_digest: String,
    pub warm_incremental_output_digest: String,
    pub fresh_replay_output_digest: String,
    pub warm_incremental_reused_stale_output: bool,
    pub fresh_replay_reflected_mutation: bool,
    pub qualification_eligible: bool,
}

#[derive(Debug, Serialize)]
struct Evidence {
    format_version: u32,
    suite_id: &'static str,
    run_id: String,
    case_id: &'static str,
    catalog_digest: String,
    fixture_digest: String,
    cargo_binary_digest: String,
    cargo_version: String,
    rustc_launcher_digest: String,
    rustc_sysroot_binary_digest: String,
    rustc_version: String,
    rustc_sysroot: String,
    workspace_relative_path: &'static str,
    git_baseline_commit: String,
    input_resource_id: &'static str,
    input_git_state: &'static str,
    baseline_input_digest: String,
    mutated_input_digest: String,
    build_script_digest: String,
    build_script_declared_rerun_paths: Vec<&'static str>,
    build_script_reads: Vec<&'static str>,
    baseline: Execution,
    warm_incremental_after_mutation: Execution,
    fresh_replay_after_mutation: Execution,
    stale_incremental_output_reproduced: bool,
    fresh_replay_observed_mutated_value: bool,
    ground_truth_basis: &'static str,
    independent_kernel_observation: bool,
    qualification_eligible: bool,
}

#[derive(Debug, Serialize)]
struct Execution {
    stage: &'static str,
    target_directory: String,
    cargo_process_id: u32,
    exit_code: i32,
    stdout_value: String,
    stdout_digest: String,
    stderr_digest: String,
}

pub fn run(catalog_path: &Path, out_dir: &Path) -> anyhow::Result<CargoFixtureRunReport> {
    let catalog_bytes = read_bounded(catalog_path, 16 * 1024 * 1024)?;
    let audit = crate::basalt_observer::audit_bytes(&catalog_bytes);
    if !audit.structurally_valid {
        bail!("challenge catalog failed structural validation");
    }
    let catalog: Value = serde_json::from_slice(&catalog_bytes)?;
    let cases = catalog.get("cases").and_then(Value::as_array)
        .context("catalog.cases must be an array")?;
    let case = cases.iter()
        .find(|case| case.get("case_id").and_then(Value::as_str) == Some(CASE_ID))
        .context("catalog does not contain cargo-build-script-input")?;
    if case.get("expected_disposition").and_then(Value::as_str) != Some("capture_dependency")
        || case.get("ground_truth_resource_id").and_then(Value::as_str) != Some(RESOURCE_ID)
        || case.get("channel").and_then(Value::as_str) != Some("process")
    {
        bail!("catalog cargo-build-script-input differs from the fixture contract");
    }

    let cargo_name = env::var_os("CARGO").unwrap_or_else(|| OsStr::new("cargo").to_os_string());
    let cargo = resolve_executable(&cargo_name)?;
    let cargo_digest = sha256_file(&cargo)?;
    let cargo_version = cargo_version(&cargo)?;
    let (rustc, rustc_version, rustc_sysroot, rustc_sysroot_binary) = rustc_identity()?;
    let rustc_launcher_digest = sha256_file(&rustc)?;
    let rustc_sysroot_binary_digest = sha256_file(&rustc_sysroot_binary)?;
    let fixture_digest = fixture_digest()?;

    let nanos = SystemTime::now().duration_since(UNIX_EPOCH)
        .context("system clock is before UNIX epoch")?.as_nanos();
    let run_id = format!("basalt-cargo-{}-{nanos}", std::process::id());
    fs::create_dir_all(out_dir)?;
    let run_dir = out_dir.join(&run_id);
    fs::create_dir(&run_dir).context("creating exclusive Cargo fixture run directory")?;
    let workspace = run_dir.join("workspace");
    fs::create_dir(&workspace)?;
    materialize(&workspace)?;
    write_new(&workspace.join("build.rs-input.txt"), BEFORE)?;
    let baseline_commit = initialize_git(&workspace)?;
    let input_git_state = verify_ignored_untracked(&workspace)?;

    let manifest = workspace.join("Cargo.toml");
    let warm_target = run_dir.join("target-warm");
    let fresh_target = run_dir.join("target-fresh");
    let baseline = run_cargo(&cargo, &rustc, &workspace, &manifest, &warm_target, "baseline")?;
    if baseline.stdout_value != input_value(BEFORE) {
        bail!("baseline task output does not match the baseline resource value");
    }

    write_replace(&workspace.join("build.rs-input.txt"), AFTER)?;
    let warm = run_cargo(&cargo, &rustc, &workspace, &manifest, &warm_target, "warm_incremental_after_mutation")?;
    let fresh = run_cargo(&cargo, &rustc, &workspace, &manifest, &fresh_target, "fresh_replay_after_mutation")?;

    let stale = warm.stdout_value == baseline.stdout_value;
    let fresh_matches = fresh.stdout_value == input_value(AFTER);
    if !stale {
        bail!("warm incremental run did not reproduce the stale output");
    }
    if !fresh_matches {
        bail!("fresh replay did not reflect the mutated build-script input");
    }

    let build_script = FIXTURE.iter().find(|(path, _)| *path == "build.rs")
        .map(|(_, contents)| *contents).context("build.rs fixture missing")?;
    let evidence = Evidence {
        format_version: 1,
        suite_id: "basalt-cargo-build-script-fixture-evidence-v1",
        run_id: run_id.clone(),
        case_id: CASE_ID,
        catalog_digest: sha256(&catalog_bytes),
        fixture_digest: fixture_digest.clone(),
        cargo_binary_digest: cargo_digest.clone(),
        cargo_version,
        rustc_launcher_digest: rustc_launcher_digest.clone(),
        rustc_sysroot_binary_digest: rustc_sysroot_binary_digest.clone(),
        rustc_version: rustc_version.clone(),
        rustc_sysroot: rustc_sysroot.clone(),
        workspace_relative_path: "workspace",
        git_baseline_commit: baseline_commit,
        input_resource_id: RESOURCE_ID,
        input_git_state,
        baseline_input_digest: sha256(BEFORE),
        mutated_input_digest: sha256(AFTER),
        build_script_digest: sha256(build_script.as_bytes()),
        build_script_declared_rerun_paths: vec!["build.rs"],
        build_script_reads: vec!["build.rs-input.txt"],
        baseline,
        warm_incremental_after_mutation: warm,
        fresh_replay_after_mutation: fresh,
        stale_incremental_output_reproduced: stale,
        fresh_replay_observed_mutated_value: fresh_matches,
        ground_truth_basis: "fixture build.rs source plus controlled input mutation and fresh target replay",
        independent_kernel_observation: false,
        qualification_eligible: false,
    };
    let bytes = serde_json::to_vec_pretty(&evidence)?;
    let evidence_path = run_dir.join("cargo-build-script-evidence.v1.json");
    write_new(&evidence_path, &bytes)?;

    Ok(CargoFixtureRunReport {
        report_version: 1,
        run_id,
        case_id: CASE_ID,
        resource_id: RESOURCE_ID,
        evidence_path: evidence_path.display().to_string(),
        workspace_path: workspace.display().to_string(),
        catalog_digest: sha256(&catalog_bytes),
        fixture_digest,
        cargo_binary_digest: cargo_digest,
        rustc_launcher_digest,
        rustc_sysroot_binary_digest,
        rustc_version,
        rustc_sysroot,
        baseline_output_digest: sha256(baseline_output_value(&evidence).as_bytes()),
        warm_incremental_output_digest: sha256(evidence.warm_incremental_after_mutation.stdout_value.as_bytes()),
        fresh_replay_output_digest: sha256(evidence.fresh_replay_after_mutation.stdout_value.as_bytes()),
        warm_incremental_reused_stale_output: stale,
        fresh_replay_reflected_mutation: fresh_matches,
        qualification_eligible: false,
    })
}

fn baseline_output_value(evidence: &Evidence) -> &str {
    &evidence.baseline.stdout_value
}

fn fixture_digest() -> anyhow::Result<String> {
    let mut map = BTreeMap::new();
    for (path, contents) in FIXTURE {
        map.insert(*path, sha256(contents.as_bytes()));
    }
    Ok(sha256(&serde_json::to_vec(&map)?))
}

fn materialize(workspace: &Path) -> anyhow::Result<()> {
    for (relative, contents) in FIXTURE {
        let path = workspace.join(relative);
        if let Some(parent) = path.parent() { fs::create_dir_all(parent)?; }
        write_new(&path, contents.as_bytes())?;
    }
    Ok(())
}

fn resolve_executable(name: &OsStr) -> anyhow::Result<PathBuf> {
    let requested = PathBuf::from(name);
    if requested.components().count() > 1 || requested.is_absolute() {
        return requested.canonicalize().context("canonicalizing CARGO executable");
    }
    let path = env::var_os("PATH").context("PATH is required to locate cargo")?;
    for directory in env::split_paths(&path) {
        let candidate = directory.join(&requested);
        if candidate.is_file() {
            return candidate.canonicalize().context("canonicalizing cargo from PATH");
        }
    }
    bail!("CARGO executable {:?} was not found in PATH", name)
}

fn cargo_version(cargo: &Path) -> anyhow::Result<String> {
    let output = Command::new(cargo).arg("--version").output().context("running cargo --version")?;
    if !output.status.success() {
        bail!("cargo --version failed: {}", String::from_utf8_lossy(&output.stderr));
    }
    Ok(String::from_utf8(output.stdout)?.trim().to_string())
}

fn rustc_identity() -> anyhow::Result<(PathBuf, String, String, PathBuf)> {
    let requested = env::var_os("RUSTC").unwrap_or_else(|| OsStr::new("rustc").to_os_string());
    let launcher = resolve_executable(&requested)?;
    let version_output = Command::new(&launcher)
        .args(["--version", "--verbose"])
        .output()
        .context("executing rustc --version --verbose")?;
    if !version_output.status.success() {
        bail!("rustc version query failed: {}", String::from_utf8_lossy(&version_output.stderr));
    }
    let version = String::from_utf8(version_output.stdout)?.trim().to_string();

    let sysroot_output = Command::new(&launcher)
        .args(["--print", "sysroot"])
        .output()
        .context("executing rustc --print sysroot")?;
    if !sysroot_output.status.success() {
        bail!("rustc sysroot query failed: {}", String::from_utf8_lossy(&sysroot_output.stderr));
    }
    let sysroot = String::from_utf8(sysroot_output.stdout)?.trim().to_string();
    let actual = PathBuf::from(&sysroot)
        .join("bin")
        .join(format!("rustc{}", env::consts::EXE_SUFFIX));
    let actual = if actual.is_file() {
        actual.canonicalize().context("canonicalizing sysroot rustc binary")?
    } else {
        launcher.clone()
    };
    Ok((launcher, version, sysroot, actual))
}

fn run_cargo(cargo: &Path, rustc: &Path, workspace: &Path, manifest: &Path, target: &Path, stage: &'static str) -> anyhow::Result<Execution> {
    let mut child = Command::new(cargo)
        .args(["run", "--quiet", "--offline", "--locked", "--manifest-path"])
        .arg(manifest)
        .current_dir(workspace)
        .env("CARGO_TARGET_DIR", target)
        .env("RUSTC", rustc)
        .env_remove("RUSTC_WRAPPER")
        .env_remove("RUSTC_WORKSPACE_WRAPPER")
        .env_remove("RUSTFLAGS")
        .env_remove("CARGO_ENCODED_RUSTFLAGS")
        .env("CARGO_NET_OFFLINE", "true")
        .env("CARGO_INCREMENTAL", "0")
        .env("CARGO_TERM_COLOR", "never")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn().context("starting offline Cargo fixture")?;
    let pid = child.id();
    let output = child.wait_with_output()?;
    let exit_code = output.status.code().unwrap_or(-1);
    if !output.status.success() {
        bail!("Cargo stage {stage} failed (exit {exit_code}): {}", String::from_utf8_lossy(&output.stderr));
    }
    let stdout_value = String::from_utf8(output.stdout)?.trim().to_string();
    let target_rel = target.strip_prefix(workspace.parent().context("run directory missing")?)
        .unwrap_or(target).to_string_lossy().into_owned();
    Ok(Execution {
        stage,
        target_directory: target_rel,
        cargo_process_id: pid,
        exit_code,
        stdout_digest: sha256(stdout_value.as_bytes()),
        stdout_value,
        stderr_digest: sha256(&output.stderr),
    })
}

fn initialize_git(workspace: &Path) -> anyhow::Result<String> {
    git(workspace, &["init", "--quiet"])?;
    git(workspace, &["config", "user.name", "BASALT Cargo Fixture"])?;
    git(workspace, &["config", "user.email", "basalt-cargo-fixture@example.invalid"])?;
    git(workspace, &["add", "--force", "--", ".gitignore", "Cargo.toml", "Cargo.lock", "build.rs", "src/main.rs"])?;
    git(workspace, &["commit", "--quiet", "-m", "BASALT Cargo fixture baseline"])?;
    let output = Command::new("git").args(["rev-parse", "HEAD"]).current_dir(workspace).output()?;
    if !output.status.success() { bail!("git rev-parse HEAD failed"); }
    Ok(String::from_utf8(output.stdout)?.trim().to_string())
}

fn verify_ignored_untracked(workspace: &Path) -> anyhow::Result<&'static str> {
    let ignored = Command::new("git").args(["check-ignore", "-q", "--", "build.rs-input.txt"])
        .current_dir(workspace).status()?.success();
    let tracked = Command::new("git").args(["ls-files", "--error-unmatch", "--", "build.rs-input.txt"])
        .current_dir(workspace).stdout(Stdio::null()).stderr(Stdio::null()).status()?.success();
    if !ignored || tracked { bail!("build.rs-input.txt must be ignored and absent from the Git index"); }
    Ok("ignored_untracked")
}

fn git(workspace: &Path, args: &[&str]) -> anyhow::Result<()> {
    let output = Command::new("git").args(args).current_dir(workspace).output()
        .with_context(|| format!("executing git {}", args.join(" ")))?;
    if !output.status.success() {
        bail!("git {} failed: {}", args.join(" "), String::from_utf8_lossy(&output.stderr));
    }
    Ok(())
}

fn input_value(bytes: &[u8]) -> String { String::from_utf8_lossy(bytes).trim().to_string() }

fn read_bounded(path: &Path, max: u64) -> anyhow::Result<Vec<u8>> {
    let file = fs::File::open(path)?;
    let mut bytes = Vec::new();
    file.take(max + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > max { bail!("input exceeds {max} byte limit"); }
    Ok(bytes)
}

fn write_replace(path: &Path, bytes: &[u8]) -> anyhow::Result<()> {
    use std::io::Write;
    let mut file = fs::OpenOptions::new().write(true).truncate(true).open(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

fn write_new(path: &Path, bytes: &[u8]) -> anyhow::Result<()> {
    use std::io::Write;
    let mut file = fs::OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

fn sha256_file(path: &Path) -> anyhow::Result<String> {
    let mut file = fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 { break; }
        hasher.update(&buffer[..n]);
    }
    let digest = hasher.finalize();
    Ok(encode_digest(&digest))
}

fn sha256(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    encode_digest(&digest)
}

fn encode_digest(bytes: &[u8]) -> String {
    let mut hex = String::with_capacity(64);
    for byte in bytes {
        use std::fmt::Write as _;
        let _ = write!(&mut hex, "{byte:02x}");
    }
    format!("sha256:{hex}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixture_reads_gitignored_input_but_omits_it_from_rerun_declarations() {
        let script = FIXTURE.iter().find(|(p, _)| *p == "build.rs").unwrap().1;
        assert!(script.contains("read_to_string(\"build.rs-input.txt\")"));
        assert!(script.contains("rerun-if-changed=build.rs"));
        assert!(!script.contains("rerun-if-changed=build.rs-input.txt"));
        assert!(FIXTURE.iter().find(|(p, _)| *p == ".gitignore").unwrap().1.contains("build.rs-input.txt"));
    }

    #[test]
    fn fixture_declares_an_independent_workspace_root() {
        let manifest = FIXTURE.iter().find(|(p, _)| *p == "Cargo.toml").unwrap().1;
        assert!(manifest.lines().any(|line| line.trim() == "[workspace]"));
    }

    #[test]
    fn fixture_is_dependency_free_and_lockfile_pinned() {
        let manifest = FIXTURE.iter().find(|(p, _)| *p == "Cargo.toml").unwrap().1;
        let lockfile = FIXTURE.iter().find(|(p, _)| *p == "Cargo.lock").unwrap().1;
        assert!(manifest.contains("build = \"build.rs\""));
        assert!(!manifest.contains("[dependencies]"));
        assert!(lockfile.contains("name = \"basalt-cargo-hidden-build-script-fixture\""));
    }

    #[test]
    fn controlled_mutation_changes_the_expected_resource_value() {
        assert_ne!(sha256(BEFORE), sha256(AFTER));
        assert_eq!(input_value(BEFORE), "baseline-state");
        assert_eq!(input_value(AFTER), "mutated-state");
    }
}
