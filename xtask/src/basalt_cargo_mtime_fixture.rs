//! Cargo adversarial test for correctly declared dependencies with unchanged mtimes.
//!
//! A regular declared-input positive control proves the build script reruns when
//! the timestamp advances. The mtime adversary changes bytes but restores the
//! exact prior mtime, then compares warm incremental execution with fresh replay.
//! This measures Cargo behavior; it does not qualify a BASALT observer.

use anyhow::{Context, bail};
use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::env;
use std::ffi::OsStr;
use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const CASE_ID: &str = "cargo-mtime-preserving-input";
const RESOURCE_ID: &str = "fixture:build.rs-input-mtime-preserved.txt";
const BEFORE: &[u8] = b"baseline-state\n";
const AFTER: &[u8] = b"mutated-state\n";
const FIXTURE: &[(&str, &str)] = &[
    ("declared/Cargo.toml", include_str!("../../benchmarks/basalt/fixtures/cargo-mtime-declared-control/Cargo.toml")),
    ("declared/Cargo.lock", include_str!("../../benchmarks/basalt/fixtures/cargo-mtime-declared-control/Cargo.lock")),
    ("declared/build.rs", include_str!("../../benchmarks/basalt/fixtures/cargo-mtime-declared-control/build.rs")),
    ("declared/src/main.rs", include_str!("../../benchmarks/basalt/fixtures/cargo-mtime-declared-control/src/main.rs")),
    ("mtime/Cargo.toml", include_str!("../../benchmarks/basalt/fixtures/cargo-mtime-adversary/Cargo.toml")),
    ("mtime/Cargo.lock", include_str!("../../benchmarks/basalt/fixtures/cargo-mtime-adversary/Cargo.lock")),
    ("mtime/build.rs", include_str!("../../benchmarks/basalt/fixtures/cargo-mtime-adversary/build.rs")),
    ("mtime/src/main.rs", include_str!("../../benchmarks/basalt/fixtures/cargo-mtime-adversary/src/main.rs")),
    (".gitignore", include_str!("../../benchmarks/basalt/fixtures/cargo-mtime-declared-control/.gitignore")),
];

#[derive(Debug, Serialize)]
pub struct MtimeFixtureRunReport {
    pub report_version: u32,
    pub run_id: String,
    pub case_id: &'static str,
    pub evidence_path: String,
    pub cargo_binary_digest: String,
    pub rustc_sysroot_binary_digest: String,
    pub declared_control_detected_mutation: bool,
    pub mtime_was_preserved: bool,
    pub cargo_incremental_missed_same_mtime_mutation: bool,
    pub fresh_replay_detected_mutation: bool,
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
    path_environment_digest: String,
    cargo_home_relative_path: &'static str,
    cargo_configuration_policy: &'static str,
    ancestor_cargo_config_absent: bool,
    workspace_relative_path: &'static str,
    git_baseline_commit: String,
    resource_id: &'static str,
    input_git_state: &'static str,
    before_content_digest: String,
    after_content_digest: String,
    declared_build_script_digest: String,
    declared_rerun_paths: Vec<&'static str>,
    mtime_build_script_digest: String,
    mtime_rerun_paths: Vec<&'static str>,
    baseline: Execution,
    declared_control_after_advanced_mtime: Execution,
    mtime_control_baseline: Execution,
    mtime_control_after_preserving_mutation: Execution,
    mtime_control_fresh_replay: Execution,
    declared_control_detected_mutation: bool,
    mtime_before_unix_ns: u64,
    mtime_after_unix_ns: u64,
    mtime_preserved_exactly: bool,
    cargo_incremental_missed_same_mtime_mutation: bool,
    fresh_replay_detected_mutation: bool,
    ground_truth_basis: &'static str,
    independent_kernel_observation: bool,
    qualification_eligible: bool,
}

#[derive(Debug, Serialize)]
struct Execution {
    stage: &'static str,
    package: &'static str,
    target_directory: String,
    cargo_process_id: u32,
    exit_code: i32,
    stdout_value: String,
    stdout_digest: String,
    stderr_digest: String,
}

pub fn run(catalog_path: &Path, out_dir: &Path) -> anyhow::Result<MtimeFixtureRunReport> {
    let catalog_bytes = read_bounded(catalog_path, 16 * 1024 * 1024)?;
    let catalog_audit = crate::basalt_observer::audit_bytes(&catalog_bytes);
    if !catalog_audit.structurally_valid {
        bail!("challenge catalog failed structural validation");
    }
    let catalog: Value = serde_json::from_slice(&catalog_bytes)?;
    validate_catalog_case(&catalog)?;

    let cargo_name = env::var_os("CARGO").unwrap_or_else(|| OsStr::new("cargo").to_os_string());
    let cargo = resolve_executable(&cargo_name)?;
    let cargo_binary_digest = sha256_file(&cargo)?;
    let cargo_version = cargo_version(&cargo)?;
    let (rustc_launcher, rustc_version, rustc_sysroot, rustc_binary) = rustc_identity()?;
    let rustc_launcher_digest = sha256_file(&rustc_launcher)?;
    let rustc_sysroot_binary_digest = sha256_file(&rustc_binary)?;
    let path_environment_digest = env::var_os("PATH")
        .map(|value| sha256(value.to_string_lossy().as_bytes()))
        .unwrap_or_else(|| sha256(b"<PATH-unset>"));
    let fixture_digest = fixture_digest()?;

    let now = SystemTime::now().duration_since(UNIX_EPOCH)
        .context("system clock is before UNIX epoch")?.as_nanos();
    let run_id = format!("basalt-cargo-mtime-{}-{now}", std::process::id());
    fs::create_dir_all(out_dir)?;
    let out_dir = out_dir.canonicalize().context("canonicalizing mtime fixture output directory")?;
    let run_dir = out_dir.join(&run_id);
    fs::create_dir(&run_dir).context("creating exclusive mtime fixture run directory")?;
    let workspace = run_dir.join("workspace");
    fs::create_dir(&workspace)?;
    materialize(&workspace)?;
    let declared_dir = workspace.join("declared");
    let mtime_dir = workspace.join("mtime");
    let declared_input = declared_dir.join("build.rs-input.txt");
    let mtime_input = mtime_dir.join("build.rs-input.txt");
    write_new(&declared_input, BEFORE)?;
    write_new(&mtime_input, BEFORE)?;

    let git_baseline_commit = initialize_git(&workspace)?;
    verify_ignored_untracked(&workspace, &[
        Path::new("declared/build.rs-input.txt"),
        Path::new("mtime/build.rs-input.txt"),
    ])?;

    ensure_no_ancestor_cargo_config(&declared_dir)?;
    ensure_no_ancestor_cargo_config(&mtime_dir)?;
    let cargo_home = run_dir.join("cargo-home");
    fs::create_dir(&cargo_home)?;

    let declared_baseline_mtime = fs::metadata(&declared_input)?.modified()?;
    let mtime_baseline_mtime = fs::metadata(&mtime_input)?.modified()?;
    let declared_baseline = run_cargo(
        &cargo, &rustc_binary, &cargo_home, &run_dir, &declared_dir,
        "declared_baseline", "declared",
    )?;
    let mtime_baseline = run_cargo(
        &cargo, &rustc_binary, &cargo_home, &run_dir, &mtime_dir,
        "mtime_baseline", "mtime",
    )?;
    if declared_baseline.stdout_value != input_value(BEFORE)
        || mtime_baseline.stdout_value != input_value(BEFORE)
    {
        bail!("Cargo positive-control baselines do not match the baseline input");
    }

    write_replace(&declared_input, AFTER)?;
    write_replace(&mtime_input, AFTER)?;

    // Force the ordinary declared control's mtime forward past any filesystem
    // timestamp granularity, so it tests the positive rerun-if-changed path.
    let declared_new_mtime = SystemTime::now()
        .checked_add(Duration::from_secs(60))
        .context("declared control mtime overflow")?;
    set_modified_time(&declared_input, declared_new_mtime)?;
    if fs::metadata(&declared_input)?.modified()? <= declared_baseline_mtime {
        bail!("declared positive-control mtime did not advance");
    }

    // This is the adversarial condition: input bytes change, mtime is restored.
    set_modified_time(&mtime_input, mtime_baseline_mtime)?;
    let mtime_before_unix_ns = unix_nanos(mtime_baseline_mtime)?;
    let mtime_after = fs::metadata(&mtime_input)?.modified()?;
    let mtime_after_unix_ns = unix_nanos(mtime_after)?;
    let mtime_preserved_exactly = mtime_after == mtime_baseline_mtime;
    if !mtime_preserved_exactly {
        bail!("filesystem did not preserve the exact original input mtime");
    }

    let declared_after = run_cargo(
        &cargo, &rustc_binary, &cargo_home, &run_dir, &declared_dir,
        "declared_control_after_advanced_mtime", "declared",
    )?;
    let mtime_warm = run_cargo(
        &cargo, &rustc_binary, &cargo_home, &run_dir, &mtime_dir,
        "mtime_control_after_preserving_mutation", "mtime",
    )?;
    let mtime_fresh_root = run_dir.join("mtime-fresh-run");
    fs::create_dir(&mtime_fresh_root)?;
    let mtime_fresh = run_cargo(
        &cargo, &rustc_binary, &cargo_home, &mtime_fresh_root, &mtime_dir,
        "mtime_control_fresh_replay", "mtime",
    )?;

    let declared_control_detected_mutation = declared_after.stdout_value == input_value(AFTER);
    let cargo_incremental_missed_same_mtime_mutation = mtime_warm.stdout_value == input_value(BEFORE);
    let fresh_replay_detected_mutation = mtime_fresh.stdout_value == input_value(AFTER);
    if !declared_control_detected_mutation {
        bail!("correctly declared positive control did not detect a newer input mtime");
    }
    if !fresh_replay_detected_mutation {
        bail!("fresh replay did not observe the mtime-preserving content mutation");
    }

    let evidence = Evidence {
        format_version: 1,
        suite_id: "basalt-cargo-mtime-fixture-evidence-v1",
        run_id: run_id.clone(),
        case_id: CASE_ID,
        catalog_digest: sha256(&catalog_bytes),
        fixture_digest: fixture_digest.clone(),
        cargo_binary_digest: cargo_binary_digest.clone(),
        cargo_version,
        rustc_launcher_digest,
        rustc_sysroot_binary_digest: rustc_sysroot_binary_digest.clone(),
        rustc_version,
        rustc_sysroot,
        path_environment_digest,
        cargo_home_relative_path: "cargo-home",
        cargo_configuration_policy: "isolated_empty_cargo_home_no_ancestor_dot_cargo_config",
        ancestor_cargo_config_absent: true,
        workspace_relative_path: "workspace",
        git_baseline_commit,
        resource_id: RESOURCE_ID,
        input_git_state: "ignored_untracked",
        before_content_digest: sha256(BEFORE),
        after_content_digest: sha256(AFTER),
        declared_build_script_digest: fixture_file_digest("declared/build.rs")?,
        declared_rerun_paths: vec!["build.rs", "build.rs-input.txt"],
        mtime_build_script_digest: fixture_file_digest("mtime/build.rs")?,
        mtime_rerun_paths: vec!["build.rs", "build.rs-input.txt"],
        baseline: declared_baseline,
        declared_control_after_advanced_mtime: declared_after,
        mtime_control_baseline: mtime_baseline,
        mtime_control_after_preserving_mutation: mtime_warm,
        mtime_control_fresh_replay: mtime_fresh,
        declared_control_detected_mutation,
        mtime_before_unix_ns,
        mtime_after_unix_ns,
        mtime_preserved_exactly,
        cargo_incremental_missed_same_mtime_mutation,
        fresh_replay_detected_mutation,
        ground_truth_basis: "declared build-script source plus controlled byte mutation, exact mtime restoration and fresh-target replay",
        independent_kernel_observation: false,
        qualification_eligible: false,
    };
    let evidence_path = run_dir.join("cargo-mtime-fixture-evidence.v1.json");
    write_new(&evidence_path, &serde_json::to_vec_pretty(&evidence)?)?;

    Ok(MtimeFixtureRunReport {
        report_version: 1,
        run_id,
        case_id: CASE_ID,
        evidence_path: evidence_path.display().to_string(),
        cargo_binary_digest,
        rustc_sysroot_binary_digest,
        declared_control_detected_mutation,
        mtime_was_preserved: mtime_preserved_exactly,
        cargo_incremental_missed_same_mtime_mutation,
        fresh_replay_detected_mutation,
        qualification_eligible: false,
    })
}

fn validate_catalog_case(catalog: &Value) -> anyhow::Result<()> {
    let cases = catalog.get("cases").and_then(Value::as_array)
        .context("catalog.cases must be an array")?;
    let case = cases.iter()
        .find(|case| case.get("case_id").and_then(Value::as_str) == Some(CASE_ID))
        .context("catalog does not contain cargo-mtime-preserving-input")?;
    if case.get("expected_disposition").and_then(Value::as_str) != Some("capture_dependency")
        || case.get("ground_truth_resource_id").and_then(Value::as_str) != Some(RESOURCE_ID)
        || case.get("channel").and_then(Value::as_str) != Some("process")
    {
        bail!("mtime challenge catalog definition differs from fixture contract");
    }
    Ok(())
}

fn fixture_digest() -> anyhow::Result<String> {
    let mut entries = BTreeMap::new();
    for (path, contents) in FIXTURE {
        entries.insert(*path, sha256(contents.as_bytes()));
    }
    Ok(sha256(&serde_json::to_vec(&entries)?))
}

fn fixture_file_digest(relative: &str) -> anyhow::Result<String> {
    let value = FIXTURE.iter().find(|(path, _)| *path == relative)
        .map(|(_, value)| *value)
        .with_context(|| format!("fixture file {relative} is not embedded"))?;
    Ok(sha256(value.as_bytes()))
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
        return requested.canonicalize().context("canonicalizing requested executable");
    }
    let path = env::var_os("PATH").context("PATH is required to resolve executable")?;
    for directory in env::split_paths(&path) {
        let candidate = directory.join(&requested);
        if candidate.is_file() { return candidate.canonicalize().context("canonicalizing executable"); }
    }
    bail!("executable {:?} was not found in PATH", name)
}

fn cargo_version(cargo: &Path) -> anyhow::Result<String> {
    let output = Command::new(cargo).arg("--version").output().context("running cargo --version")?;
    if !output.status.success() { bail!("cargo --version failed"); }
    Ok(String::from_utf8(output.stdout)?.trim().to_string())
}

fn rustc_identity() -> anyhow::Result<(PathBuf, String, String, PathBuf)> {
    let name = env::var_os("RUSTC").unwrap_or_else(|| OsStr::new("rustc").to_os_string());
    let launcher = resolve_executable(&name)?;
    let version = Command::new(&launcher).args(["--version", "--verbose"])
        .output().context("running rustc version query")?;
    if !version.status.success() { bail!("rustc version query failed"); }
    let sysroot = Command::new(&launcher).args(["--print", "sysroot"])
        .output().context("running rustc sysroot query")?;
    if !sysroot.status.success() { bail!("rustc sysroot query failed"); }
    let version_text = String::from_utf8(version.stdout.clone())?.trim().to_string();
    let sysroot_text = String::from_utf8(sysroot.stdout)?.trim().to_string();
    let binary = PathBuf::from(&sysroot_text).join("bin")
        .join(format!("rustc{}", env::consts::EXE_SUFFIX));
    if !binary.is_file() { bail!("rustc sysroot lacks compiler binary {}", binary.display()); }
    let binary = binary.canonicalize()?;
    let actual_version = Command::new(&binary).args(["--version", "--verbose"]).output()?;
    if !actual_version.status.success() || actual_version.stdout != version.stdout {
        bail!("rustc launcher and sysroot compiler do not identify the same toolchain");
    }
    Ok((launcher, version_text, sysroot_text, binary))
}

fn ensure_no_ancestor_cargo_config(start: &Path) -> anyhow::Result<()> {
    for ancestor in start.ancestors() {
        for name in [".cargo/config.toml", ".cargo/config"] {
            let path = ancestor.join(name);
            if path.exists() { bail!("refusing ambient Cargo config {}", path.display()); }
        }
    }
    Ok(())
}

fn run_cargo(
    cargo: &Path,
    rustc: &Path,
    cargo_home: &Path,
    run_root: &Path,
    package_dir: &Path,
    stage: &'static str,
    package: &'static str,
) -> anyhow::Result<Execution> {
    let target = run_root.join(format!("target-{package}"));
    let mut command = Command::new(cargo);
    command.args(["run", "--quiet", "--offline", "--locked", "--manifest-path"])
        .arg(package_dir.join("Cargo.toml"))
        .current_dir(package_dir)
        .env("CARGO_HOME", cargo_home)
        .env("CARGO_TARGET_DIR", &target)
        .env("RUSTC", rustc)
        .env("CARGO_NET_OFFLINE", "true")
        .env("CARGO_INCREMENTAL", "0")
        .env("CARGO_TERM_COLOR", "never")
        .env_remove("RUSTC_WRAPPER")
        .env_remove("RUSTC_WORKSPACE_WRAPPER")
        .env_remove("RUSTFLAGS")
        .env_remove("CARGO_ENCODED_RUSTFLAGS")
        .env_remove("CARGO_BUILD_RUSTFLAGS")
        .env_remove("CARGO_BUILD_TARGET")
        .env_remove("RUSTC_BOOTSTRAP")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    for (key, _) in env::vars_os() {
        let text = key.to_string_lossy();
        if text.starts_with("CARGO_TARGET_") && text.ends_with("_LINKER") {
            command.env_remove(key);
        }
    }
    let mut child = command.spawn().context("starting isolated offline Cargo run")?;
    let pid = child.id();
    let output = child.wait_with_output()?;
    let exit_code = output.status.code().unwrap_or(-1);
    if !output.status.success() {
        bail!("Cargo stage {stage} ({package}) failed exit={exit_code}: {}", String::from_utf8_lossy(&output.stderr));
    }
    let stdout_value = String::from_utf8(output.stdout)?.trim().to_string();
    Ok(Execution {
        stage,
        package,
        target_directory: target.strip_prefix(run_root).unwrap_or(&target).to_string_lossy().into_owned(),
        cargo_process_id: pid,
        exit_code,
        stdout_digest: sha256(stdout_value.as_bytes()),
        stdout_value,
        stderr_digest: sha256(&output.stderr),
    })
}

fn initialize_git(workspace: &Path) -> anyhow::Result<String> {
    git(workspace, &["init", "--quiet"])?;
    git(workspace, &["config", "user.name", "BASALT Cargo mtime Fixture"])?;
    git(workspace, &["config", "user.email", "basalt-mtime-fixture@example.invalid"])?;
    git(workspace, &[
        "add", "--force", "--", ".gitignore",
        "declared/Cargo.toml", "declared/Cargo.lock", "declared/build.rs", "declared/src/main.rs",
        "mtime/Cargo.toml", "mtime/Cargo.lock", "mtime/build.rs", "mtime/src/main.rs",
    ])?;
    git(workspace, &["commit", "--quiet", "-m", "BASALT Cargo mtime fixture baseline"])?;
    let output = Command::new("git").args(["rev-parse", "HEAD"]).current_dir(workspace).output()?;
    if !output.status.success() { bail!("git rev-parse HEAD failed"); }
    Ok(String::from_utf8(output.stdout)?.trim().to_string())
}

fn verify_ignored_untracked(workspace: &Path, paths: &[&Path]) -> anyhow::Result<()> {
    for path in paths {
        let rendered = path.to_string_lossy();
        let ignored = Command::new("git").args(["check-ignore", "-q", "--", &rendered]).current_dir(workspace).status()?.success();
        let tracked = Command::new("git").args(["ls-files", "--error-unmatch", "--", &rendered]).current_dir(workspace)
            .stdout(Stdio::null()).stderr(Stdio::null()).status()?.success();
        if !ignored || tracked { bail!("{} must be ignored and absent from Git index", rendered); }
    }
    Ok(())
}

fn git(workspace: &Path, args: &[&str]) -> anyhow::Result<()> {
    let output = Command::new("git").args(args).current_dir(workspace).output()
        .with_context(|| format!("executing git {}", args.join(" ")))?;
    if !output.status.success() { bail!("git {} failed: {}", args.join(" "), String::from_utf8_lossy(&output.stderr)); }
    Ok(())
}

fn set_modified_time(path: &Path, modified: SystemTime) -> anyhow::Result<()> {
    File::open(path)?.set_modified(modified).with_context(|| format!("setting mtime {}", path.display()))
}

fn unix_nanos(time: SystemTime) -> anyhow::Result<u64> {
    u64::try_from(time.duration_since(UNIX_EPOCH).context("mtime predates UNIX epoch")?.as_nanos())
        .context("mtime nanoseconds overflow u64")
}

fn read_bounded(path: &Path, max: u64) -> anyhow::Result<Vec<u8>> {
    let file = File::open(path)?;
    let mut bytes=Vec::new();
    file.take(max+1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > max { bail!("catalog input exceeds {max} bytes"); }
    Ok(bytes)
}

fn write_new(path: &Path, bytes: &[u8]) -> anyhow::Result<()> {
    use std::io::Write;
    let mut f=fs::OpenOptions::new().write(true).create_new(true).open(path)?;
    f.write_all(bytes)?;
    f.sync_all()?;
    Ok(())
}

fn write_replace(path: &Path, bytes: &[u8]) -> anyhow::Result<()> {
    use std::io::Write;
    let mut f=fs::OpenOptions::new().write(true).truncate(true).open(path)?;
    f.write_all(bytes)?;
    f.sync_all()?;
    Ok(())
}

fn sha256_file(path: &Path) -> anyhow::Result<String> {
    let mut f=File::open(path)?;
    let mut h=Sha256::new();
    let mut buf=[0_u8;65536];
    loop { let n=f.read(&mut buf)?; if n==0 {break;} h.update(&buf[..n]); }
    Ok(encode_digest(&h.finalize()))
}

fn sha256(bytes: &[u8]) -> String { encode_digest(&Sha256::digest(bytes)) }

fn encode_digest(bytes: &[u8]) -> String {
    let mut hex=String::with_capacity(64);
    for byte in bytes {
        use std::fmt::Write as _;
        let _=write!(&mut hex,"{byte:02x}");
    }
    format!("sha256:{hex}")
}

fn input_value(bytes:&[u8])->String { String::from_utf8_lossy(bytes).trim().to_string() }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn both_fixture_build_scripts_declare_the_input() {
        for path in ["declared/build.rs","mtime/build.rs"] {
            let source=FIXTURE.iter().find(|(p,_)| *p==path).unwrap().1;
            assert!(source.contains("rerun-if-changed=build.rs-input.txt"));
        }
    }

    #[test]
    fn fixture_uses_different_package_ids_but_same_output_contract() {
        assert!(FIXTURE.iter().any(|(p,s)| *p=="declared/Cargo.toml" && s.contains("basalt-cargo-mtime-declared-control")));
        assert!(FIXTURE.iter().any(|(p,s)| *p=="mtime/Cargo.toml" && s.contains("basalt-cargo-mtime-adversary")));
    }

    #[test]
    fn future_positive_control_mtime_is_strictly_newer_than_baseline() {
        let baseline = UNIX_EPOCH + Duration::from_secs(100);
        let advanced = baseline.checked_add(Duration::from_secs(60)).unwrap();
        assert!(advanced > baseline);
    }

    #[test]
    fn evidence_string_reader_rejects_missing_and_non_string_fields() {
        assert!(required_str(&serde_json::json!({}), "run_id").is_err());
        assert!(required_str(&serde_json::json!({"run_id": 4}), "run_id").is_err());
        assert_eq!(required_str(&serde_json::json!({"run_id":"r"}), "run_id").unwrap(), "r");
    }

    #[test]
    fn evidence_integer_reader_rejects_negative_and_non_integer_fields() {
        assert!(required_u64(&serde_json::json!({"mtime":-1}), "mtime").is_err());
        assert!(required_u64(&serde_json::json!({"mtime":"1"}), "mtime").is_err());
        assert_eq!(required_u64(&serde_json::json!({"mtime":1}), "mtime").unwrap(), 1);
    }

    #[test]
    fn exact_mtime_restore_helper_preserves_timestamp_after_content_change() {
        let path=env::temp_dir().join(format!("basalt-mtime-fixture-{}-{}",std::process::id(),
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos()));
        write_new(&path,b"baseline").unwrap();
        let original=fs::metadata(&path).unwrap().modified().unwrap();
        write_replace(&path,b"mutated").unwrap();
        set_modified_time(&path,original).unwrap();
        assert_eq!(fs::metadata(&path).unwrap().modified().unwrap(),original);
        fs::remove_file(path).unwrap();
    }
}


#[derive(Debug, Serialize)]
pub struct MtimeEvidenceVerifyReport {
    pub report_version: u32,
    pub run_id: String,
    pub verification_passed: bool,
    pub catalog_identity_verified: bool,
    pub toolchain_identity_verified: bool,
    pub workspace_identity_verified: bool,
    pub mtime_identity_verified: bool,
    pub original_warm_replay_matches: bool,
    pub independent_fresh_replay_matches: bool,
    pub warm_replay_process_id: u32,
    pub fresh_replay_process_id: u32,
    pub warm_output_digest: String,
    pub fresh_output_digest: String,
    pub oracle_authenticity_established: bool,
    pub qualification_eligible: bool,
}

/// Independently reopens a retained mtime fixture run, validates its identity,
/// replays its warm target, and performs a new fresh-target build.
/// This verifies a fixture execution, not the authenticity of the oracle or an observer.
pub fn verify(evidence_path: &Path, catalog_path: &Path) -> anyhow::Result<MtimeEvidenceVerifyReport> {
    let evidence_bytes = read_bounded(evidence_path, 4 * 1024 * 1024)?;
    let evidence: Value = serde_json::from_slice(&evidence_bytes)?;
    let run_root = evidence_path.parent().context("evidence file has no parent run directory")?
        .canonicalize().context("canonicalizing evidence run directory")?;
    let workspace_relative = required_str(&evidence, "workspace_relative_path")?;
    if workspace_relative != "workspace" {
        bail!("unsupported workspace_relative_path {workspace_relative}");
    }
    let workspace = run_root.join(workspace_relative);
    let canonical_workspace = workspace.canonicalize().context("resolving retained fixture workspace")?;
    if canonical_workspace != workspace {
        bail!("fixture workspace resolved through a symlink or path alias");
    }

    let catalog_bytes = read_bounded(catalog_path, 16 * 1024 * 1024)?;
    let catalog_audit = crate::basalt_observer::audit_bytes(&catalog_bytes);
    if !catalog_audit.structurally_valid {
        bail!("challenge catalog failed structural validation");
    }
    let catalog_digest = sha256(&catalog_bytes);
    if required_str(&evidence, "catalog_digest")? != catalog_digest {
        bail!("evidence catalog digest does not match supplied catalog");
    }
    validate_catalog_case(&serde_json::from_slice(&catalog_bytes)?)?;

    if required_str(&evidence, "fixture_digest")? != fixture_digest()? {
        bail!("evidence fixture digest does not match this verifier's embedded fixture");
    }
    for (relative, expected) in FIXTURE {
        let actual = fs::read(workspace.join(relative))
            .with_context(|| format!("reading retained fixture file {relative}"))?;
        if actual != expected.as_bytes() {
            bail!("retained fixture file {relative} differs from the pinned template");
        }
    }

    let cargo_name = env::var_os("CARGO").unwrap_or_else(|| OsStr::new("cargo").to_os_string());
    let cargo = resolve_executable(&cargo_name)?;
    let cargo_binary_digest = sha256_file(&cargo)?;
    let cargo_version = cargo_version(&cargo)?;
    let (rustc_launcher, rustc_version, rustc_sysroot, rustc_binary) = rustc_identity()?;
    let rustc_launcher_digest = sha256_file(&rustc_launcher)?;
    let rustc_binary_digest = sha256_file(&rustc_binary)?;
    let path_digest = env::var_os("PATH")
        .map(|value| sha256(value.to_string_lossy().as_bytes()))
        .unwrap_or_else(|| sha256(b"<PATH-unset>"));

    let toolchain_identity_verified =
        required_str(&evidence, "cargo_binary_digest")? == cargo_binary_digest
        && required_str(&evidence, "cargo_version")? == cargo_version
        && required_str(&evidence, "rustc_launcher_digest")? == rustc_launcher_digest
        && required_str(&evidence, "rustc_sysroot_binary_digest")? == rustc_binary_digest
        && required_str(&evidence, "rustc_version")? == rustc_version
        && required_str(&evidence, "rustc_sysroot")? == rustc_sysroot
        && required_str(&evidence, "path_environment_digest")? == path_digest;
    if !toolchain_identity_verified {
        bail!("Cargo/rustc/PATH identity differs from the recorded fixture evidence");
    }

    let cargo_home_path = run_root.join(required_str(&evidence, "cargo_home_relative_path")?);
    if !cargo_home_path.is_dir() {
        bail!("recorded isolated CARGO_HOME is missing");
    }
    if cargo_home_path.join("config.toml").exists() || cargo_home_path.join("config").exists() {
        bail!("recorded CARGO_HOME contains an unexpected Cargo configuration file");
    }
    if required_str(&evidence, "cargo_configuration_policy")?
        != "isolated_empty_cargo_home_no_ancestor_dot_cargo_config"
        || !evidence.get("ancestor_cargo_config_absent").and_then(Value::as_bool).unwrap_or(false)
    {
        bail!("recorded Cargo configuration policy is not the supported isolated policy");
    }
    ensure_no_ancestor_cargo_config(&workspace.join("declared"))?;
    ensure_no_ancestor_cargo_config(&workspace.join("mtime"))?;

    let git_commit = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(&workspace)
        .output()
        .context("checking retained fixture Git baseline")?;
    if !git_commit.status.success()
        || String::from_utf8(git_commit.stdout)?.trim()
            != required_str(&evidence, "git_baseline_commit")?
    {
        bail!("retained fixture Git HEAD differs from recorded baseline");
    }
    let git_status = Command::new("git")
        .args(["status", "--porcelain", "--untracked-files=all"])
        .current_dir(&workspace)
        .output()
        .context("checking retained fixture working tree")?;
    if !git_status.status.success() || !git_status.stdout.is_empty() {
        bail!("retained fixture tracked/untracked state changed after its baseline commit");
    }
    if required_str(&evidence, "input_git_state")? != "ignored_untracked" {
        bail!("unsupported input Git-state classification");
    }

    let declared_input = workspace.join("declared/build.rs-input.txt");
    let mtime_input = workspace.join("mtime/build.rs-input.txt");
    let after_digest = sha256(AFTER);
    if required_str(&evidence, "before_content_digest")? != sha256(BEFORE)
        || required_str(&evidence, "after_content_digest")? != after_digest
        || sha256(&fs::read(&declared_input)?) != after_digest
        || sha256(&fs::read(&mtime_input)?) != after_digest
    {
        bail!("retained input bytes do not match the recorded before/after content digests");
    }
    let recorded_mtime = required_u64(&evidence, "mtime_after_unix_ns")?;
    let actual_mtime = unix_nanos(fs::metadata(&mtime_input)?.modified()?)?;
    let mtime_identity_verified =
        evidence.get("mtime_preserved_exactly").and_then(Value::as_bool) == Some(true)
        && required_u64(&evidence, "mtime_before_unix_ns")? == recorded_mtime
        && actual_mtime == recorded_mtime;
    if !mtime_identity_verified {
        bail!("retained mtime input does not preserve its recorded exact timestamp");
    }

    let declared_source = fs::read_to_string(workspace.join("declared/build.rs"))?;
    let mtime_source = fs::read_to_string(workspace.join("mtime/build.rs"))?;
    if sha256(declared_source.as_bytes()) != required_str(&evidence, "declared_build_script_digest")?
        || sha256(mtime_source.as_bytes()) != required_str(&evidence, "mtime_build_script_digest")?
    {
        bail!("retained build-script source differs from the recorded fixture digests");
    }
    if required_str(&evidence, "resource_id")? != RESOURCE_ID {
        bail!("recorded resource identity does not match this mtime fixture verifier");
    }
    verify_ignored_untracked(
        &workspace,
        &[
            Path::new("declared/build.rs-input.txt"),
            Path::new("mtime/build.rs-input.txt"),
        ],
    )?;

    let declared_control = evidence.get("declared_control_after_advanced_mtime")
        .context("missing declared positive-control execution")?;
    if declared_control.get("stdout_value").and_then(Value::as_str) != Some(input_value(AFTER).as_str()) {
        bail!("recorded declared positive control did not output the mutated value");
    }
    if evidence.get("declared_control_detected_mutation").and_then(Value::as_bool) != Some(true) {
        bail!("recorded declared positive control did not detect mutation");
    }
    if evidence.get("fresh_replay_detected_mutation").and_then(Value::as_bool) != Some(true) {
        bail!("recorded fresh replay did not detect mutation");
    }

    // Replay with the same recorded isolated CARGO_HOME so the verifier does
    // not change Cargo's configuration/cache namespace while checking the warm target.
    let cargo_home = cargo_home_path;
    let mtime_dir = workspace.join("mtime");
    let warm_replay = run_cargo(
        &cargo,
        &rustc_binary,
        &cargo_home,
        &run_root,
        &mtime_dir,
        "mtime_control_after_preserving_mutation",
        "mtime",
    )?;
    let warm_record = evidence.get("mtime_control_after_preserving_mutation")
        .context("missing recorded warm mtime execution")?;
    let recorded_warm_value = warm_record.get("stdout_value")
        .and_then(Value::as_str).context("recorded warm execution lacks stdout_value")?;
    let recorded_warm_digest = warm_record.get("stdout_digest")
        .and_then(Value::as_str).context("recorded warm execution lacks stdout_digest")?;
    let warm_replay_matches = warm_replay.stdout_value == recorded_warm_value
        && warm_replay.stdout_digest == recorded_warm_digest
        && recorded_warm_digest == sha256(recorded_warm_value.as_bytes());
    let recorded_missed = evidence.get("cargo_incremental_missed_same_mtime_mutation")
        .and_then(Value::as_bool).context("recorded mtime cache outcome is not boolean")?;
    if recorded_missed != (recorded_warm_value == input_value(BEFORE)) {
        bail!("recorded same-mtime miss flag contradicts the recorded warm output");
    }
    if !warm_replay_matches {
        bail!("independent warm replay output/digest differs from recorded warm-cache evidence");
    }

    let verification_nonce = SystemTime::now().duration_since(UNIX_EPOCH)
        .context("system clock is before UNIX epoch")?.as_nanos();
    let fresh_root = run_root.join(format!(
        "verifier-fresh-replay-{}-{verification_nonce}",
        std::process::id()
    ));
    fs::create_dir(&fresh_root).context("creating exclusive verifier fresh-replay directory")?;
    let fresh_replay = run_cargo(
        &cargo,
        &rustc_binary,
        &cargo_home,
        &fresh_root,
        &mtime_dir,
        "mtime_control_fresh_replay",
        "mtime",
    )?;
    let expected_fresh_value = input_value(AFTER);
    let expected_fresh_digest = sha256(expected_fresh_value.as_bytes());
    let recorded_fresh = evidence.get("mtime_control_fresh_replay")
        .context("missing recorded fresh execution")?;
    if recorded_fresh.get("stdout_value").and_then(Value::as_str) != Some(expected_fresh_value.as_str())
        || recorded_fresh.get("stdout_digest").and_then(Value::as_str) != Some(expected_fresh_digest.as_str())
    {
        bail!("recorded fresh replay evidence does not match the mutated input value/digest");
    }
    let fresh_replay_matches = fresh_replay.stdout_value == expected_fresh_value
        && fresh_replay.stdout_digest == expected_fresh_digest;
    if !fresh_replay_matches {
        bail!("independent fresh replay did not emit the mutated input value/digest");
    }

    Ok(MtimeEvidenceVerifyReport {
        report_version: 1,
        run_id: required_str(&evidence, "run_id")?.to_string(),
        verification_passed: true,
        catalog_identity_verified: true,
        toolchain_identity_verified,
        workspace_identity_verified: true,
        mtime_identity_verified,
        original_warm_replay_matches: warm_replay_matches,
        independent_fresh_replay_matches: fresh_replay_matches,
        warm_replay_process_id: warm_replay.cargo_process_id,
        fresh_replay_process_id: fresh_replay.cargo_process_id,
        warm_output_digest: warm_replay.stdout_digest,
        fresh_output_digest: fresh_replay.stdout_digest,
        oracle_authenticity_established: false,
        qualification_eligible: false,
    })
}

fn required_str<'a>(value: &'a Value, key: &str) -> anyhow::Result<&'a str> {
    value.get(key).and_then(Value::as_str)
        .with_context(|| format!("missing or invalid evidence string field {key}"))
}

fn required_u64(value: &Value, key: &str) -> anyhow::Result<u64> {
    value.get(key).and_then(Value::as_u64)
        .with_context(|| format!("missing or invalid evidence integer field {key}"))
}
