// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Differential-check the exact XOR-bind RTL against the canonical Rust corpus.
//!
//! Usage from the repository root:
//!   cargo run --locked -p symthaea-silicon --example hdc_rtl_reference_corpus > /tmp/hdc-corpus.json
//!   cargo run --locked -p symthaea-silicon --example hdc_bind_rtl_simulator_diff -- /tmp/hdc-corpus.json
//!
//! This harness does not implement HDC algebra. It translates frozen corpus
//! bytes to the packed RTL bus, runs Icarus Verilog, and captures a provenance
//! receipt. Missing tools, dirty source files, wrong pins, malformed vectors,
//! compiler/simulation failures, or unresolved compiler warnings fail closed.

use serde::Deserialize;
use serde_json::json;
use std::collections::HashSet;
use std::env;
use std::fs;
use std::io::Write as IoWrite;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

const CORPUS_SCHEMA: &str = "neuro-silicon-hdc-rtl-corpus-v1";
const ORACLE_SOURCE_SHA: &str = "458c7b98d81c64b9361e252f85ef9d45132e6682";
const BENCHMARK_CONTRACT_SHA: &str = "68370c5431c8843553380c70d21952db5696cb50";
const VECTOR_BITS: usize = 16_384;
const VECTOR_BYTES: usize = VECTOR_BITS / 8;
const VECTOR_HEX_LEN: usize = VECTOR_BYTES * 2;
const DEFAULT_RTL: &str = "crates/domains/symthaea-silicon/rtl/hdc_bind_xor.sv";
const CANONICAL_ORACLE_FILE: &str = "crates/core/symthaea-core/src/hdc/binary_hv.rs";

#[derive(Debug, Deserialize)]
struct Corpus {
    schema_version: String,
    oracle_source_sha: String,
    benchmark_contract_sha: String,
    dimension_bits: usize,
    vector_bytes: usize,
    fixtures: Vec<Fixture>,
}

#[derive(Debug, Deserialize)]
struct Fixture {
    id: String,
    operation: String,
    input_hex: Vec<String>,
    expected_vector_hex: Option<String>,
    hardware_required_v1: bool,
}

fn main() {
    if let Err(message) = run() {
        eprintln!("{message}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() || args.len() > 2 {
        return Err(
            "usage: hdc_bind_rtl_simulator_diff <corpus.json> [hdc_bind_xor.sv]".to_owned(),
        );
    }

    let repo_root = PathBuf::from(
        git_text(&["rev-parse", "--show-toplevel"], "locate repository root")?
    );
    let repo_root = fs::canonicalize(&repo_root)
        .map_err(|error| format!("BLOCKED: cannot canonicalize repository root: {error}"))?;
    let current_dir = fs::canonicalize(
        env::current_dir().map_err(|error| format!("BLOCKED: cannot read current directory: {error}"))?
    ).map_err(|error| format!("BLOCKED: cannot canonicalize current directory: {error}"))?;
    if current_dir != repo_root {
        return Err(format!(
            "BLOCKED: run from the repository root '{}' (current directory is '{}')",
            repo_root.display(),
            current_dir.display()
        ));
    }
    let head_sha = git_text(&["rev-parse", "HEAD"], "read exact repository HEAD")?;

    let corpus_path = fs::canonicalize(&args[0])
        .map_err(|error| format!("BLOCKED: cannot open corpus '{}': {error}", args[0]))?;
    let rtl_input = args.get(1).map(String::as_str).unwrap_or(DEFAULT_RTL);
    let rtl_path = fs::canonicalize(rtl_input)
        .map_err(|error| format!("BLOCKED: cannot open RTL source '{rtl_input}': {error}"))?;

    let corpus_bytes = fs::read(&corpus_path)
        .map_err(|error| format!("BLOCKED: cannot read corpus '{}': {error}", corpus_path.display()))?;
    let corpus: Corpus = serde_json::from_slice(&corpus_bytes)
        .map_err(|error| format!("FAIL: corpus is not valid JSON matching the expected schema: {error}"))?;

    validate_corpus(&corpus)?;
    require_ancestor(ORACLE_SOURCE_SHA, &head_sha, "canonical Rust oracle commit")?;
    require_ancestor(BENCHMARK_CONTRACT_SHA, &head_sha, "benchmark-contract commit")?;

    let rtl_relative = repo_relative_path(&repo_root, &rtl_path)?;
    let oracle_path = fs::canonicalize(repo_root.join(CANONICAL_ORACLE_FILE))
        .map_err(|error| format!("BLOCKED: canonical Rust oracle file is missing: {error}"))?;
    let oracle_relative = repo_relative_path(&repo_root, &oracle_path)?;
    require_tracked_clean(&rtl_relative, "RTL source")?;
    require_tracked_clean(&oracle_relative, "canonical Rust oracle source")?;
    let rtl_blob_sha = git_blob_sha(&fs::read(&rtl_path).map_err(|e| e.to_string())?)?;
    let oracle_blob_sha = git_blob_sha(&fs::read(&oracle_path).map_err(|e| e.to_string())?)?;
    let corpus_blob_sha = git_blob_sha(&corpus_bytes)?;

    let selected = select_bind_fixtures(&corpus)?;
    let selected_ids: Vec<String> = selected.iter().map(|fixture| fixture.id.clone()).collect();
    let unsupported_ids: Vec<String> = corpus.fixtures.iter()
        .filter(|fixture| fixture.operation != "bind" && fixture.operation != "bind_chain")
        .map(|fixture| fixture.id.clone())
        .collect();
    let generated_testbench = build_testbench(&selected)?;
    let generated_testbench_blob_sha = git_blob_sha(generated_testbench.as_bytes())?;

    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos();
    let temp_dir = env::temp_dir().join(format!("symthaea-hdc-xor-diff-{}-{now}", std::process::id()));
    fs::create_dir(&temp_dir)
        .map_err(|error| format!("BLOCKED: cannot create temporary evidence directory: {error}"))?;
    let testbench_path = temp_dir.join("hdc_bind_rtl_corpus_tb.sv");
    let simulator_path = temp_dir.join("hdc_bind_rtl_corpus_tb.vvp");
    fs::write(&testbench_path, generated_testbench.as_bytes())
        .map_err(|error| format!("BLOCKED: cannot write generated testbench: {error}"))?;

    let iverilog_version = checked_version("iverilog", &["-V"])?;
    let vvp_version = checked_version("vvp", &["-V"])?;
    let compile_command = vec![
        "iverilog".to_owned(),
        "-g2012".to_owned(),
        "-Wall".to_owned(),
        "-s".to_owned(),
        "hdc_bind_rtl_corpus_tb".to_owned(),
        "-o".to_owned(),
        simulator_path.display().to_string(),
        rtl_path.display().to_string(),
        testbench_path.display().to_string(),
    ];
    let compile_output = run_command(
        Command::new("iverilog")
            .arg("-g2012")
            .arg("-Wall")
            .arg("-s")
            .arg("hdc_bind_rtl_corpus_tb")
            .arg("-o")
            .arg(&simulator_path)
            .arg(&rtl_path)
            .arg(&testbench_path),
        "compile generated RTL differential bench",
    )?;
    if !compile_output.status.success() {
        return Err(format!(
            "FAIL: iverilog compile exited {:?}; stdout:\n{}\nstderr:\n{}\nGenerated evidence retained at {}",
            compile_output.status.code(),
            output_text(&compile_output.stdout),
            output_text(&compile_output.stderr),
            temp_dir.display(),
        ));
    }

    let compile_stdout = output_text(&compile_output.stdout);
    let compile_stderr = output_text(&compile_output.stderr);
    let warnings_detected = format!("{compile_stdout}\n{compile_stderr}")
        .to_ascii_lowercase()
        .contains("warning");

    let simulation_output = run_command(
        Command::new("vvp").arg(&simulator_path),
        "run generated RTL differential bench",
    )?;
    let simulation_stdout = output_text(&simulation_output.stdout);
    let simulation_stderr = output_text(&simulation_output.stderr);
    if !simulation_output.status.success() {
        return Err(format!(
            "FAIL: vvp simulation exited {:?}; stdout:\n{}\nstderr:\n{}\nGenerated evidence retained at {}",
            simulation_output.status.code(),
            simulation_stdout,
            simulation_stderr,
            temp_dir.display(),
        ));
    }
    if !simulation_stdout.contains("PASS: hdc_bind_rtl_corpus_tb") {
        return Err(format!(
            "FAIL: simulator exited zero but omitted the testbench PASS marker; stdout:\n{}\nstderr:\n{}",
            simulation_stdout, simulation_stderr
        ));
    }

    let claim = if warnings_detected {
        "BLOCKED_WARNING_REVIEW_REQUIRED"
    } else {
        "RTL_ORACLE_DIFFERENTIAL_PASS"
    };
    let receipt = json!({
        "schema_version": "neuro-silicon-hdc-xor-sim-receipt-v1",
        "claim": claim,
        "git_head_sha": head_sha,
        "corpus_schema_version": corpus.schema_version,
        "oracle_source_commit_sha": corpus.oracle_source_sha,
        "oracle_source_blob_sha": oracle_blob_sha,
        "benchmark_contract_commit_sha": corpus.benchmark_contract_sha,
        "corpus_git_blob_sha": corpus_blob_sha,
        "rtl_path": rtl_relative.display().to_string(),
        "rtl_git_blob_sha": rtl_blob_sha,
        "dimension_bits": corpus.dimension_bits,
        "vector_bytes": corpus.vector_bytes,
        "fixture_count": selected.len(),
        "fixture_ids": selected_ids,
        "unsupported_operation_fixture_ids": unsupported_ids,
        "simulator": "Icarus Verilog",
        "iverilog_version_output": iverilog_version,
        "vvp_version_output": vvp_version,
        "compile_command": compile_command,
        "compile_exit_code": compile_output.status.code(),
        "compile_stdout": compile_stdout,
        "compile_stderr": compile_stderr,
        "warnings_detected": warnings_detected,
        "simulation_command": ["vvp", simulator_path.display().to_string()],
        "simulation_exit_code": simulation_output.status.code(),
        "simulation_stdout": simulation_stdout,
        "simulation_stderr": simulation_stderr,
        "generated_testbench_path": testbench_path.display().to_string(),
        "generated_testbench_git_blob_sha": generated_testbench_blob_sha,
        "generated_simulator_binary_path": simulator_path.display().to_string(),
        "evidence_directory_preserved": temp_dir.display().to_string(),
        "limitations": [
            "XOR bind and repeated bind-chain only; permutation, bundling and distance are unsupported here.",
            "A simulation receipt is not FPGA board evidence, PPA closure, a workload speedup, measured energy savings, or silicon qualification."
        ]
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&receipt)
            .map_err(|error| format!("FAIL: cannot serialize receipt: {error}"))?
    );

    if warnings_detected {
        return Err(format!(
            "BLOCKED: differential simulation ran, but compiler warnings need explicit review; receipt emitted above. Evidence retained at {}",
            temp_dir.display()
        ));
    }
    Ok(())
}

fn validate_corpus(corpus: &Corpus) -> Result<(), String> {
    if corpus.schema_version != CORPUS_SCHEMA {
        return Err(format!("BLOCKED: unexpected corpus schema '{}'", corpus.schema_version));
    }
    if corpus.oracle_source_sha != ORACLE_SOURCE_SHA {
        return Err(format!("BLOCKED: oracle source pin mismatch: '{}'", corpus.oracle_source_sha));
    }
    if corpus.benchmark_contract_sha != BENCHMARK_CONTRACT_SHA {
        return Err(format!(
            "BLOCKED: benchmark contract pin mismatch: '{}'",
            corpus.benchmark_contract_sha
        ));
    }
    if corpus.dimension_bits != VECTOR_BITS || corpus.vector_bytes != VECTOR_BYTES {
        return Err(format!(
            "BLOCKED: unsupported vector dimensions: {} bits / {} bytes",
            corpus.dimension_bits, corpus.vector_bytes
        ));
    }

    let mut ids = HashSet::new();
    for fixture in &corpus.fixtures {
        if fixture.id.is_empty()
            || !fixture.id.bytes().all(|byte| byte.is_ascii_alphanumeric() || byte == b'-' || byte == b'_')
        {
            return Err(format!("BLOCKED: invalid fixture identifier '{}'", fixture.id));
        }
        if !ids.insert(fixture.id.as_str()) {
            return Err(format!("BLOCKED: duplicate fixture identifier '{}'", fixture.id));
        }
    }
    Ok(())
}

fn select_bind_fixtures(corpus: &Corpus) -> Result<Vec<&Fixture>, String> {
    let required_ids = [
        "bind-zero-ones",
        "bind-self-zero",
        "bind-seed-1-2",
        "unbind-recovery",
        "bind-chain-9",
    ];
    let mut selected = Vec::new();

    for fixture in &corpus.fixtures {
        if fixture.operation != "bind" && fixture.operation != "bind_chain" {
            continue;
        }
        if !fixture.hardware_required_v1 {
            return Err(format!(
                "BLOCKED: XOR-compatible fixture '{}' is not marked hardware-required",
                fixture.id
            ));
        }
        let expected = fixture.expected_vector_hex.as_deref().ok_or_else(|| {
            format!("BLOCKED: fixture '{}' lacks expected vector output", fixture.id)
        })?;
        canonical_hex_to_sv(expected, &fixture.id)?;
        for input in &fixture.input_hex {
            canonical_hex_to_sv(input, &fixture.id)?;
        }
        match fixture.operation.as_str() {
            "bind" if fixture.input_hex.len() != 2 => {
                return Err(format!("BLOCKED: bind fixture '{}' must have exactly two inputs", fixture.id));
            }
            "bind_chain" if fixture.input_hex.len() < 2 => {
                return Err(format!("BLOCKED: bind-chain fixture '{}' requires at least two inputs", fixture.id));
            }
            _ => {}
        }
        selected.push(fixture);
    }

    for required in required_ids {
        if !selected.iter().any(|fixture| fixture.id == required) {
            return Err(format!("BLOCKED: required pinned fixture '{required}' is missing"));
        }
    }
    if selected.is_empty() {
        return Err("BLOCKED: no supported bind fixtures were found".to_owned());
    }
    Ok(selected)
}

fn canonical_hex_to_sv(hex: &str, fixture_id: &str) -> Result<String, String> {
    if hex.len() != VECTOR_HEX_LEN {
        return Err(format!(
            "BLOCKED: fixture '{}' vector has {} hex digits, expected {}",
            fixture_id,
            hex.len(),
            VECTOR_HEX_LEN
        ));
    }
    if !hex.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(format!("BLOCKED: fixture '{}' vector contains non-hex characters", fixture_id));
    }

    // BinaryHV stores byte 0 first; the SV packed literal's leftmost byte maps
    // to the high bus bits. Reverse whole bytes (not bits/nibbles) so canonical
    // byte 0 maps to bus bits [7:0]. XOR is order-independent, but fixing this
    // convention now prevents a hidden endianness bug in later primitives.
    let bytes = hex.as_bytes();
    let mut reversed = String::with_capacity(hex.len());
    for pair in bytes.chunks_exact(2).rev() {
        reversed.push(pair[0] as char);
        reversed.push(pair[1] as char);
    }
    Ok(reversed)
}

fn build_testbench(fixtures: &[&Fixture]) -> Result<String, String> {
    let mut source = String::from(
        "`timescale 1ns/1ps\n\
         `default_nettype none\n\
         module hdc_bind_rtl_corpus_tb;\n\
             localparam integer WIDTH = 16384;\n\
             reg [WIDTH-1:0] a;\n\
             reg [WIDTH-1:0] b;\n\
             reg [WIDTH-1:0] expected;\n\
             reg [WIDTH-1:0] accumulator;\n\
             wire [WIDTH-1:0] y;\n\
             hdc_bind_xor #(.WIDTH(WIDTH)) dut (.a(a), .b(b), .y(y));\n\
             initial begin\n\
                 a = '0; b = '0; expected = '0; accumulator = '0;\n"
    );

    for fixture in fixtures {
        let expected = canonical_hex_to_sv(
            fixture.expected_vector_hex.as_deref().expect("validated expected output"),
            &fixture.id,
        )?;
        match fixture.operation.as_str() {
            "bind" => {
                let a = canonical_hex_to_sv(&fixture.input_hex[0], &fixture.id)?;
                let b = canonical_hex_to_sv(&fixture.input_hex[1], &fixture.id)?;
                source.push_str(&format!(
                    "                // Fixture: {id}\n\
                     a = {width}'h{a};\n\
                     b = {width}'h{b};\n\
                     expected = {width}'h{expected};\n\
                     #1;\n\
                     if (y !== expected) $fatal(1, \"fixture {id} mismatch\");\n",
                    id = fixture.id,
                    width = VECTOR_BITS,
                    a = a,
                    b = b,
                    expected = expected,
                ));
            }
            "bind_chain" => {
                let first = canonical_hex_to_sv(&fixture.input_hex[0], &fixture.id)?;
                let second = canonical_hex_to_sv(&fixture.input_hex[1], &fixture.id)?;
                source.push_str(&format!(
                    "                // Fixture: {id}; fold through the same RTL primitive\n\
                     a = {width}'h{first};\n\
                     b = {width}'h{second};\n\
                     #1; accumulator = y;\n",
                    id = fixture.id,
                    width = VECTOR_BITS,
                    first = first,
                    second = second,
                ));
                for input in fixture.input_hex.iter().skip(2) {
                    let vector = canonical_hex_to_sv(input, &fixture.id)?;
                    source.push_str(&format!(
                        "                a = accumulator; b = {width}'h{vector};\n\
                         #1; accumulator = y;\n",
                        width = VECTOR_BITS,
                        vector = vector,
                    ));
                }
                source.push_str(&format!(
                    "                expected = {width}'h{expected};\n\
                     if (accumulator !== expected) $fatal(1, \"fixture {id} mismatch\");\n",
                    width = VECTOR_BITS,
                    expected = expected,
                    id = fixture.id,
                ));
            }
            operation => {
                return Err(format!("BLOCKED: unsupported operation '{operation}' reached test generation"));
            }
        }
    }

    source.push_str(&format!(
        "                $display(\"PASS: hdc_bind_rtl_corpus_tb; fixtures={}\" );\n\
                 $finish;\n\
             end\n\
         endmodule\n\
         `default_nettype wire\n",
        fixtures.len()
    ));
    Ok(source)
}

fn checked_version(program: &str, args: &[&str]) -> Result<String, String> {
    let output = run_command(Command::new(program).args(args), &format!("read {program} version"))?;
    if !output.status.success() {
        return Err(format!(
            "BLOCKED: {program} version command exited {:?}: {}",
            output.status.code(),
            output_text(&output.stderr)
        ));
    }
    Ok(format!(
        "stdout={}\nstderr={}",
        output_text(&output.stdout),
        output_text(&output.stderr)
    ))
}

fn require_ancestor(ancestor: &str, head: &str, label: &str) -> Result<(), String> {
    let output = Command::new("git")
        .arg("merge-base")
        .arg("--is-ancestor")
        .arg(ancestor)
        .arg(head)
        .output()
        .map_err(|error| format!("BLOCKED: cannot validate {label}: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "BLOCKED: pinned {label} {ancestor} is not verifiably an ancestor of HEAD {head}"
        ));
    }
    Ok(())
}

fn repo_relative_path(repo_root: &Path, path: &Path) -> Result<PathBuf, String> {
    path.strip_prefix(repo_root)
        .map(Path::to_path_buf)
        .map_err(|_| format!("BLOCKED: source '{}' is outside repository root '{}'", path.display(), repo_root.display()))
}

fn require_tracked_clean(path: &Path, label: &str) -> Result<(), String> {
    let tracked = git_path_text(&["ls-files", "--error-unmatch"], path)?;
    if tracked.trim().is_empty() {
        return Err(format!("BLOCKED: {label} is not a tracked source file: {}", path.display()));
    }
    let status = git_path_text(&["status", "--porcelain"], path)?;
    if !status.trim().is_empty() {
        return Err(format!(
            "BLOCKED: {label} has uncommitted changes (status='{}')",
            status.trim()
        ));
    }
    Ok(())
}

fn git_blob_sha(bytes: &[u8]) -> Result<String, String> {
    let mut child = Command::new("git")
        .arg("hash-object")
        .arg("--no-filters")
        .arg("--stdin")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| format!("BLOCKED: cannot start git hash-object: {error}"))?;
    child
        .stdin
        .take()
        .ok_or_else(|| "FAIL: git hash-object stdin was not available".to_owned())?
        .write_all(bytes)
        .map_err(|error| format!("FAIL: cannot hash evidence bytes: {error}"))?;
    let output = child
        .wait_with_output()
        .map_err(|error| format!("FAIL: cannot collect git hash-object output: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "FAIL: git hash-object exited {:?}: {}",
            output.status.code(),
            output_text(&output.stderr)
        ));
    }
    Ok(output_text(&output.stdout).trim().to_owned())
}

fn git_text(args: &[&str], label: &str) -> Result<String, String> {
    let output = run_command(Command::new("git").args(args), label)?;
    if !output.status.success() {
        return Err(format!(
            "BLOCKED: {label} exited {:?}: {}",
            output.status.code(),
            output_text(&output.stderr)
        ));
    }
    Ok(output_text(&output.stdout).trim().to_owned())
}

fn git_path_text(args: &[&str], path: &Path) -> Result<String, String> {
    let output = run_command(Command::new("git").args(args).arg("--").arg(path), "inspect tracked source")?;
    if !output.status.success() {
        return Err(format!(
            "BLOCKED: git {:?} '{}' exited {:?}: {}",
            args, path.display(), output.status.code(), output_text(&output.stderr)
        ));
    }
    Ok(output_text(&output.stdout))
}

fn run_command(command: &mut Command, label: &str) -> Result<Output, String> {
    command.output().map_err(|error| {
        if error.kind() == std::io::ErrorKind::NotFound {
            format!("BLOCKED: required executable for '{label}' is unavailable: {error}")
        } else {
            format!("FAIL: could not start '{label}': {error}")
        }
    })
}

fn output_text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}
