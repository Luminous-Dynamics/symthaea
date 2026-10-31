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
    let declared_new_mtime = declared_baseline_mtime
        .checked_add(Duration::from_secs(3))
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
    let mtime_fresh = run_cargo(
        &cargo, &rustc_binary, &cargo_home, &run_dir.join("target-mtime-fresh-home"), &mtime_dir,
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
