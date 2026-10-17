// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Compare streaming Hamming RTL with the exact distances in the canonical Rust corpus.

use serde::Deserialize;
use serde_json::json;
use std::collections::HashSet;
use std::env;
use std::fs;
use std::io::Write as IoWrite;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

const SCHEMA: &str = "neuro-silicon-hdc-rtl-corpus-v1";
const ORACLE_COMMIT: &str = "458c7b98d81c64b9361e252f85ef9d45132e6682";
const CONTRACT_COMMIT: &str = "68370c5431c8843553380c70d21952db5696cb50";
const ORACLE_PATH: &str = "crates/core/symthaea-core/src/hdc/binary_hv.rs";
const CONTRACT_PATH: &str = "docs/research/NEURO_SILICON_001A_BENCHMARK_CONTRACT.md";
const GENERATOR_PATH: &str = "crates/domains/symthaea-silicon/examples/hdc_rtl_reference_corpus.rs";
const DEFAULT_RTL: &str = "crates/domains/symthaea-silicon/rtl/hdc_hamming_stream.sv";
const DIM: usize = 16384;
const BYTES: usize = DIM / 8;
const HEX_DIGITS: usize = BYTES * 2;
const LANES: [usize; 5] = [64, 128, 256, 512, 16384];

#[derive(Deserialize)]
struct Corpus {
    schema_version: String,
    oracle_source_sha: String,
    benchmark_contract_sha: String,
    dimension_bits: usize,
    vector_bytes: usize,
    fixtures: Vec<Fixture>,
}
#[derive(Deserialize)]
struct Fixture {
    id: String,
    operation: String,
    input_hex: Vec<String>,
    expected_hamming_distance: Option<u32>,
    hardware_required_v1: bool,
}

fn main() {
    if let Err(e) = run() { eprintln!("{e}"); std::process::exit(2); }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() || args.len() > 2 {
        return Err("usage: hdc_hamming_rtl_simulator_diff <corpus.json> [hdc_hamming_stream.sv]".into());
    }
    let root = PathBuf::from(git(&["rev-parse", "--show-toplevel"], "repository root")?);
    let root = fs::canonicalize(&root).map_err(|e| format!("BLOCKED: canonicalize root: {e}"))?;
    let cwd = fs::canonicalize(env::current_dir().map_err(|e| format!("BLOCKED: cwd: {e}"))?)
        .map_err(|e| format!("BLOCKED: canonicalize cwd: {e}"))?;
    if cwd != root { return Err(format!("BLOCKED: run from repository root {}", root.display())); }
    let head = git(&["rev-parse", "HEAD"], "HEAD")?;
    let corpus_path = fs::canonicalize(&args[0]).map_err(|e| format!("BLOCKED: corpus missing: {e}"))?;
    let rtl_path = fs::canonicalize(args.get(1).map(String::as_str).unwrap_or(DEFAULT_RTL))
        .map_err(|e| format!("BLOCKED: Hamming RTL missing: {e}"))?;
    let corpus_bytes = fs::read(&corpus_path).map_err(|e| format!("BLOCKED: read corpus: {e}"))?;
    let corpus: Corpus = serde_json::from_slice(&corpus_bytes).map_err(|e| format!("FAIL: corpus JSON: {e}"))?;
    validate(&corpus)?;
    ancestor(ORACLE_COMMIT, &head, "canonical oracle commit")?;
    ancestor(CONTRACT_COMMIT, &head, "benchmark contract commit")?;

    let rtl_rel = rel(&root, &rtl_path)?;
    let oracle = fs::canonicalize(root.join(ORACLE_PATH)).map_err(|e| format!("BLOCKED: oracle file missing: {e}"))?;
    let contract = fs::canonicalize(root.join(CONTRACT_PATH)).map_err(|e| format!("BLOCKED: contract file missing: {e}"))?;
    let generator = fs::canonicalize(root.join(GENERATOR_PATH)).map_err(|e| format!("BLOCKED: corpus generator missing: {e}"))?;
    let oracle_rel = rel(&root, &oracle)?;
    let contract_rel = rel(&root, &contract)?;
    let generator_rel = rel(&root, &generator)?;
    for (p, label) in [
        (rtl_rel.as_path(), "RTL"), (oracle_rel.as_path(), "oracle"),
        (contract_rel.as_path(), "benchmark contract"), (generator_rel.as_path(), "corpus generator")
    ] { tracked_clean(p, label)?; }

    let rtl_blob = blob(&fs::read(&rtl_path).map_err(|e| e.to_string())?)?;
    let oracle_blob = blob(&fs::read(&oracle).map_err(|e| e.to_string())?)?;
    let contract_blob = blob(&fs::read(&contract).map_err(|e| e.to_string())?)?;
    let generator_blob = blob(&fs::read(&generator).map_err(|e| e.to_string())?)?;
    let pinned_oracle_blob = tree_blob(ORACLE_COMMIT, ORACLE_PATH)?;
    let pinned_contract_blob = tree_blob(CONTRACT_COMMIT, CONTRACT_PATH)?;
    if oracle_blob != pinned_oracle_blob { return Err(format!("BLOCKED: oracle blob mismatch {oracle_blob} != {pinned_oracle_blob}")); }
    if contract_blob != pinned_contract_blob { return Err(format!("BLOCKED: contract blob mismatch {contract_blob} != {pinned_contract_blob}")); }

    let cargo_version = version("cargo", &["--version"])?;
    let corpus_command = ["cargo", "run", "--locked", "--quiet", "-p", "symthaea-silicon", "--example", "hdc_rtl_reference_corpus"];
    let generated = execute(
        Command::new("cargo").args(["run", "--locked", "--quiet", "-p", "symthaea-silicon", "--example", "hdc_rtl_reference_corpus"]),
        "regenerate Rust corpus"
    )?;
    if !generated.status.success() {
        return Err(format!("BLOCKED: corpus regeneration failed: {}\n{}", out(&generated.stdout), out(&generated.stderr)));
    }
    let corpus_blob = blob(&corpus_bytes)?;
    let regenerated_blob = blob(&generated.stdout)?;
    if corpus_bytes != generated.stdout {
        return Err(format!("BLOCKED: supplied corpus {corpus_blob} differs from regenerated corpus {regenerated_blob}"));
    }

    let fixtures = select(&corpus)?;
    let ids: Vec<String> = fixtures.iter().map(|f| f.id.clone()).collect();
    let unsupported: Vec<String> = corpus.fixtures.iter().filter(|f| f.operation != "hamming_distance").map(|f| f.id.clone()).collect();
    let tb = make_tb(&fixtures)?;
    let tb_blob = blob(tb.as_bytes())?;
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos();
    let evidence = env::temp_dir().join(format!("symthaea-hamming-diff-{}-{nonce}", std::process::id()));
    fs::create_dir(&evidence).map_err(|e| format!("BLOCKED: evidence dir: {e}"))?;
    let tb_path = evidence.join("hdc_hamming_oracle_tb.sv");
    let sim_path = evidence.join("hdc_hamming_oracle_tb.vvp");
    fs::write(&tb_path, tb.as_bytes()).map_err(|e| format!("BLOCKED: testbench write: {e}"))?;

    let iv_version = version("iverilog", &["-V"])?;
    let vvp_version = version("vvp", &["-V"])?;
    let compile_cmd = vec![
        "iverilog".to_owned(), "-g2012".into(), "-Wall".into(), "-s".into(),
        "hdc_hamming_rtl_corpus_tb".into(), "-o".into(), sim_path.display().to_string(),
        rtl_path.display().to_string(), tb_path.display().to_string()
    ];
    let compiled = execute(
        Command::new("iverilog").arg("-g2012").arg("-Wall").arg("-s")
            .arg("hdc_hamming_rtl_corpus_tb").arg("-o").arg(&sim_path).arg(&rtl_path).arg(&tb_path),
        "compile Hamming corpus bench"
    )?;
    if !compiled.status.success() {
        return Err(format!("FAIL: compile {:?}; stdout={}\nstderr={}\nevidence={}",
            compiled.status.code(), out(&compiled.stdout), out(&compiled.stderr), evidence.display()));
    }
    let compile_stdout = out(&compiled.stdout);
    let compile_stderr = out(&compiled.stderr);
    let warnings = format!("{compile_stdout}\n{compile_stderr}").to_ascii_lowercase().contains("warning");
    let simulated = execute(Command::new("vvp").arg(&sim_path), "simulate Hamming corpus bench")?;
    let sim_stdout = out(&simulated.stdout);
    let sim_stderr = out(&simulated.stderr);
    if !simulated.status.success() {
        return Err(format!("FAIL: simulation {:?}; stdout={sim_stdout}\nstderr={sim_stderr}\nevidence={}",
            simulated.status.code(), evidence.display()));
    }
    if !sim_stdout.contains("PASS: hdc_hamming_rtl_corpus_tb") {
        return Err(format!("FAIL: missing PASS marker; stdout={sim_stdout}\nstderr={sim_stderr}"));
    }

    let receipt = json!({
        "schema_version":"neuro-silicon-hamming-sim-receipt-v1",
        "claim":if warnings {"BLOCKED_WARNING_REVIEW_REQUIRED"} else {"RUST_ORACLE_HAMMING_DIFFERENTIAL_PASS"},
        "git_head_sha":head,
        "oracle_source_commit_sha":corpus.oracle_source_sha,
        "oracle_source_blob_sha":oracle_blob,
        "pinned_oracle_source_blob_sha":pinned_oracle_blob,
        "benchmark_contract_commit_sha":corpus.benchmark_contract_sha,
        "benchmark_contract_blob_sha":contract_blob,
        "pinned_benchmark_contract_blob_sha":pinned_contract_blob,
        "corpus_generator_path":generator_rel.display().to_string(),
        "corpus_generator_blob_sha":generator_blob,
        "cargo_version_output":cargo_version,
        "corpus_generation_command":corpus_command,
        "corpus_git_blob_sha":corpus_blob,
        "regenerated_corpus_blob_sha":regenerated_blob,
        "supplied_corpus_matches_regenerated_byte_for_byte":true,
        "rtl_path":rtl_rel.display().to_string(),
        "rtl_git_blob_sha":rtl_blob,
        "generated_testbench_git_blob_sha":tb_blob,
        "dimension_bits":corpus.dimension_bits,
        "vector_bytes":corpus.vector_bytes,
        "lane_widths_bits":LANES,
        "fixture_count":fixtures.len(),
        "fixture_ids":ids,
        "unsupported_operation_fixture_ids":unsupported,
        "simulator":"Icarus Verilog",
        "iverilog_version_output":iv_version,
        "vvp_version_output":vvp_version,
        "compile_command":compile_cmd,
        "compile_exit_code":compiled.status.code(),
        "compile_stdout":compile_stdout,
        "compile_stderr":compile_stderr,
        "warnings_detected":warnings,
        "simulation_command":["vvp",sim_path.display().to_string()],
        "simulation_exit_code":simulated.status.code(),
        "simulation_stdout":sim_stdout,
        "simulation_stderr":sim_stderr,
        "evidence_directory_preserved":evidence.display().to_string(),
        "limitations":[
            "Only canonical Hamming-distance fixtures were executed; other corpus operations are listed as unsupported.",
            "Simulation equality is not FPGA observation, PPA closure, retrieval utility, system energy advantage or silicon qualification."
        ]
    });
    println!("{}", serde_json::to_string_pretty(&receipt).map_err(|e| format!("FAIL: receipt serialization: {e}"))?);
    if warnings { return Err(format!("BLOCKED: compiler warnings require review; evidence={}", evidence.display())); }
    Ok(())
}

fn validate(c: &Corpus) -> Result<(), String> {
    if c.schema_version != SCHEMA { return Err(format!("BLOCKED: wrong schema '{}'", c.schema_version)); }
    if c.oracle_source_sha != ORACLE_COMMIT { return Err(format!("BLOCKED: wrong oracle pin '{}'", c.oracle_source_sha)); }
    if c.benchmark_contract_sha != CONTRACT_COMMIT { return Err(format!("BLOCKED: wrong contract pin '{}'", c.benchmark_contract_sha)); }
    if c.dimension_bits != DIM || c.vector_bytes != BYTES { return Err("BLOCKED: unsupported vector size".into()); }
    let mut ids = HashSet::new();
    for f in &c.fixtures {
        if f.id.is_empty() || !f.id.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_') {
            return Err(format!("BLOCKED: invalid fixture ID '{}'", f.id));
        }
        if !ids.insert(f.id.as_str()) { return Err(format!("BLOCKED: duplicate fixture ID '{}'", f.id)); }
    }
    Ok(())
}

fn select(c: &Corpus) -> Result<Vec<&Fixture>, String> {
    let required = ["hamming-identical", "hamming-zero-ones", "hamming-seed-1-2"];
    let mut selected = Vec::new();
    for f in &c.fixtures {
        if f.operation != "hamming_distance" { continue; }
        if !f.hardware_required_v1 || f.input_hex.len() != 2 {
            return Err(format!("BLOCKED: invalid Hamming fixture '{}'", f.id));
        }
        let d = f.expected_hamming_distance.ok_or_else(|| format!("BLOCKED: '{}' lacks integer distance", f.id))?;
        if d as usize > DIM { return Err(format!("BLOCKED: '{}' distance exceeds dimension", f.id)); }
        for input in &f.input_hex { to_sv(input, &f.id)?; }
        selected.push(f);
    }
    for id in required {
        if !selected.iter().any(|f| f.id == id) { return Err(format!("BLOCKED: required fixture '{id}' missing")); }
    }
    Ok(selected)
}

fn to_sv(hex: &str, id: &str) -> Result<String, String> {
    if hex.len() != HEX_DIGITS || !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(format!("BLOCKED: fixture '{}' must have {} hex digits", id, HEX_DIGITS));
    }
    let mut s = String::with_capacity(hex.len());
    for pair in hex.as_bytes().chunks_exact(2).rev() { s.push(pair[0] as char); s.push(pair[1] as char); }
    Ok(s)
}

fn make_tb(fixtures: &[&Fixture]) -> Result<String, String> {
    let mut s = String::from(r##"@BT@timescale 1ns/1ps
@BT@default_nettype none
package hdc_hamming_corpus_pkg;
localparam integer VECTOR_BITS = 16384;
"##);
    for (i, f) in fixtures.iter().enumerate() {
        let a = to_sv(&f.input_hex[0], &f.id)?;
        let b = to_sv(&f.input_hex[1], &f.id)?;
        s.push_str(&format!("localparam [16383:0] FIXTURE_A_{i} = 16384'h{a};\nlocalparam [16383:0] FIXTURE_B_{i} = 16384'h{b};\n"));
    }
    s.push_str(r##"endpackage
module hdc_hamming_oracle_case #(parameter integer LANE_WIDTH = 256) (output reg done);
import hdc_hamming_corpus_pkg::*;
localparam integer COUNT_WIDTH = $clog2(VECTOR_BITS + 1);
localparam integer BEATS = VECTOR_BITS / LANE_WIDTH;
reg clk = 0; reg rst_n = 0; reg s_valid = 0; wire s_ready;
reg [LANE_WIDTH-1:0] s_a = 0; reg [LANE_WIDTH-1:0] s_b = 0; reg s_last = 0;
wire m_valid; reg m_ready = 0; wire [COUNT_WIDTH-1:0] m_distance; wire protocol_error;
reg stalled_q = 0; reg [COUNT_WIDTH-1:0] held_distance_q;
always #5 clk = ~clk;
hdc_hamming_stream #(.DATA_WIDTH(LANE_WIDTH), .VECTOR_BITS(VECTOR_BITS), .COUNT_WIDTH(COUNT_WIDTH)) dut (
 .clk(clk), .rst_n(rst_n), .s_valid(s_valid), .s_ready(s_ready), .s_a(s_a), .s_b(s_b), .s_last(s_last),
 .m_valid(m_valid), .m_ready(m_ready), .m_distance(m_distance), .protocol_error(protocol_error));
always @(posedge clk) begin
 if (!rst_n) stalled_q <= 0;
 else begin
  if (stalled_q && (m_valid !== 1'b1 || m_distance !== held_distance_q)) $fatal(1, "output changed while stalled");
  if (m_valid && !m_ready) begin stalled_q <= 1; held_distance_q <= m_distance; end else stalled_q <= 0;
  if (protocol_error) $fatal(1, "valid oracle fixture raised protocol error");
 end
end
task automatic run_fixture(input [16383:0] in_a, input [16383:0] in_b, input [COUNT_WIDTH-1:0] expected, input integer fixture_id);
 integer beat; integer cycles; reg [COUNT_WIDTH-1:0] observed; reg received;
 begin
  observed = 0; cycles = 0; received = 0;
  fork
   begin
    for (beat = 0; beat < BEATS; beat = beat + 1) begin
     @(negedge clk); s_valid = 1; s_a = in_a[beat*LANE_WIDTH +: LANE_WIDTH]; s_b = in_b[beat*LANE_WIDTH +: LANE_WIDTH]; s_last = (beat == BEATS-1);
     do begin @(posedge clk); end while (!s_ready);
    end
    @(negedge clk); s_valid = 0; s_last = 0;
   end
   begin
    while (!received) begin
     @(negedge clk); m_ready = ((cycles % 5) != 2) && ((cycles % 7) != 4);
     @(posedge clk); cycles = cycles + 1;
     if (m_valid && m_ready) begin
      observed = m_distance;
      if (observed !== expected) $fatal(1, "fixture %0d lane=%0d expected=%0d actual=%0d", fixture_id, LANE_WIDTH, expected, observed);
      received = 1;
     end
     if (cycles > BEATS*10 + 100) $fatal(1, "fixture %0d lane=%0d timeout", fixture_id, LANE_WIDTH);
    end
    @(negedge clk); m_ready = 0;
   end
  join
  $display("HAMMING_ORACLE_PASS lane=%0d beats=%0d fixture=%0d distance=%0d cycles=%0d", LANE_WIDTH, BEATS, fixture_id, observed, cycles);
 end
endtask
initial begin
 done = 0; repeat (3) @(posedge clk); @(negedge clk); rst_n = 1;
"##);
    for (i, f) in fixtures.iter().enumerate() {
        s.push_str(&format!("// canonical Rust fixture: {id}\nrun_fixture(FIXTURE_A_{i}, FIXTURE_B_{i}, {distance}, {i});\n",
            id=f.id, i=i, distance=f.expected_hamming_distance.expect("validated distance")));
    }
    s.push_str(r##"done = 1;
end
endmodule
module hdc_hamming_rtl_corpus_tb;
wire d64, d128, d256, d512, d16384;
hdc_hamming_oracle_case #(.LANE_WIDTH(64)) c64(d64);
hdc_hamming_oracle_case #(.LANE_WIDTH(128)) c128(d128);
hdc_hamming_oracle_case #(.LANE_WIDTH(256)) c256(d256);
hdc_hamming_oracle_case #(.LANE_WIDTH(512)) c512(d512);
hdc_hamming_oracle_case #(.LANE_WIDTH(16384)) c16384(d16384);
initial begin
 wait(d64 && d128 && d256 && d512 && d16384);
 $display("PASS: hdc_hamming_rtl_corpus_tb");
 $finish;
end
endmodule
@BT@default_nettype wire
"##);
    Ok(s.replace("@BT@", &char::from(96).to_string()))
}

fn version(program: &str, args: &[&str]) -> Result<String, String> {
    let o = execute(Command::new(program).args(args), &format!("version {program}"))?;
    if !o.status.success() { return Err(format!("BLOCKED: {program} version failed: {}", out(&o.stderr))); }
    Ok(format!("stdout={}\nstderr={}", out(&o.stdout), out(&o.stderr)))
}
fn ancestor(commit: &str, head: &str, label: &str) -> Result<(), String> {
    let o = Command::new("git").arg("merge-base").arg("--is-ancestor").arg(commit).arg(head)
        .output().map_err(|e| format!("BLOCKED: check {label}: {e}"))?;
    if !o.status.success() { return Err(format!("BLOCKED: {label} {commit} not ancestor of {head}")); }
    Ok(())
}
fn tree_blob(commit: &str, path: &str) -> Result<String, String> {
    let spec = format!("{commit}:{path}"); git(&["rev-parse", spec.as_str()], "pinned tree blob")
}
fn rel(root: &Path, p: &Path) -> Result<PathBuf, String> {
    p.strip_prefix(root).map(Path::to_path_buf).map_err(|_| format!("BLOCKED: {} outside repository", p.display()))
}
fn tracked_clean(p: &Path, label: &str) -> Result<(), String> {
    if git_path(&["ls-files", "--error-unmatch"], p)?.trim().is_empty() { return Err(format!("BLOCKED: {label} untracked")); }
    let s = git_path(&["status", "--porcelain"], p)?;
    if !s.trim().is_empty() { return Err(format!("BLOCKED: {label} dirty: {}", s.trim())); }
    Ok(())
}
fn blob(bytes: &[u8]) -> Result<String, String> {
    let mut child = Command::new("git").arg("hash-object").arg("--no-filters").arg("--stdin")
        .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped())
        .spawn().map_err(|e| format!("BLOCKED: hash-object unavailable: {e}"))?;
    child.stdin.take().ok_or("FAIL: hash-object stdin unavailable")?.write_all(bytes)
        .map_err(|e| format!("FAIL: hash-object stdin write: {e}"))?;
    let o = child.wait_with_output().map_err(|e| format!("FAIL: hash-object wait: {e}"))?;
    if !o.status.success() { return Err(format!("FAIL: hash-object: {}", out(&o.stderr))); }
    Ok(out(&o.stdout).trim().to_owned())
}
fn git(args: &[&str], label: &str) -> Result<String, String> {
    let o = execute(Command::new("git").args(args), label)?;
    if !o.status.success() { return Err(format!("BLOCKED: {label}: {}", out(&o.stderr))); }
    Ok(out(&o.stdout).trim().to_owned())
}
fn git_path(args: &[&str], p: &Path) -> Result<String, String> {
    let o = execute(Command::new("git").args(args).arg("--").arg(p), "inspect file status")?;
    if !o.status.success() { return Err(format!("BLOCKED: git {:?} {}: {}", args, p.display(), out(&o.stderr))); }
    Ok(out(&o.stdout))
}
fn execute(c: &mut Command, label: &str) -> Result<Output, String> {
    c.output().map_err(|e| if e.kind() == std::io::ErrorKind::NotFound {
        format!("BLOCKED: executable for {label} unavailable: {e}")
    } else { format!("FAIL: run {label}: {e}") })
}
fn out(bytes: &[u8]) -> String { String::from_utf8_lossy(bytes).into_owned() }
