//! Strict import of the corrected NATS-Bench-MCU v2 tabular export.
//!
//! This module validates source hashes, RFC-4180 CSV structure, candidate partition coverage,
//! accuracy joins, flash-feasibility labels and numeric measurements. It imports benchmark
//! evidence; it does not train models or independently remeasure hardware.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const SOURCE_SCHEMA: &str = "symthaea.nas.nats-bench-mcu-v2-source";
const RECEIPT_SCHEMA: &str = "symthaea.nas.nats-bench-mcu-v2-import";
const VERSION: &str = "0.1.0";
const SOURCE_DOMAIN: &[u8] = b"symthaea.nas.nats-bench-mcu-v2-source.v0.1";
const RECEIPT_DOMAIN: &[u8] = b"symthaea.nas.nats-bench-mcu-v2-import.v0.1";
const SHA256_LEN: usize = 64;
const SHA1_LEN: usize = 40;
const FLOAT_TOLERANCE: f64 = 1e-8;

const HW_HEADER: [&str; 17] = [
    "arch_idx", "arch_str", "int8_kb", "rom_kb", "ram_kb", "latency_s",
    "current_uA", "power_mW", "energy_mJ", "board", "g_model_kb",
    "arena_used_kb", "arena_size_kb", "arena_used_pct", "operators",
    "rom_categories_json", "ram_categories_json",
];
const ACC_HEADER: [&str; 4] = ["arch_idx", "cifar10", "cifar100", "ImageNet16-120"];
const FAIL_HEADER: [&str; 5] = ["arch_idx", "stage", "int8_kb", "overflow_kb", "board"];
const COVERAGE_HEADER: [&str; 3] = ["stage", "count", "pct_of_total"];
const RANK_HEADER: [&str; 7] = [
    "arch_idx", "arch_str", "nats_cifar10", "fp32_acc", "int8_acc", "delta_pp", "cratered",
];
const RANK_SUMMARY_HEADER: [&str; 6] = [
    "comparison", "n", "spearman_rho", "kendall_tau", "top10_overlap", "top25_overlap",
];
const BASELINE_HEADER: [&str; 5] = [
    "algorithm", "final_acc_mean", "final_gap_to_oracle_pp",
    "median_evals_to_oracle", "wasted_eval_frac",
];

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NatsBenchMcuV2SourceManifest {
    pub schema: String,
    pub version: String,
    pub doi: String,
    pub source_repository: String,
    pub source_commit: String,
    pub candidate_count: u32,
    pub measured_count: u32,
    pub infeasible_count: u32,
    pub flash_capacity_kb: f64,
    pub firmware_overhead_kb: f64,
    pub int8_limit_kb: f64,
    pub rank_rows: u32,
    pub rank_summary_rows: u32,
    pub baseline_rows: u32,
    pub hardware_sha256: String,
    pub accuracy_sha256: String,
    pub failure_sha256: String,
    pub coverage_sha256: String,
    pub rank_sha256: String,
    pub rank_summary_sha256: String,
    pub baseline_sha256: String,
    pub readme_sha256: String,
    pub manifest_sha256: String,
}

#[derive(Debug, Clone, Copy)]
pub struct NatsBenchMcuV2CsvBundle<'a> {
    pub hardware: &'a str,
    pub accuracy: &'a str,
    pub failures: &'a str,
    pub coverage: &'a str,
    pub rank: &'a str,
    pub rank_summary: &'a str,
    pub baselines: &'a str,
    pub readme: &'a str,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NatsBenchMcuV2HardwareRow {
    pub arch_idx: u32,
    pub arch_str: String,
    pub int8_kb: f64,
    pub rom_kb: f64,
    pub ram_kb: f64,
    pub latency_s: f64,
    pub current_u_a: f64,
    pub power_m_w: f64,
    pub energy_m_j: f64,
    pub board: String,
    pub g_model_kb: f64,
    pub arena_used_kb: f64,
    pub arena_size_kb: f64,
    pub arena_used_pct: f64,
    pub operators: u32,
    pub rom_categories_kb: BTreeMap<String, f64>,
    pub ram_categories_kb: BTreeMap<String, f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NatsBenchMcuV2AccuracyRow {
    pub arch_idx: u32,
    pub cifar10: f64,
    pub cifar100: f64,
    pub imagenet16_120: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NatsBenchMcuV2FailureRow {
    pub arch_idx: u32,
    pub stage: String,
    pub int8_kb: f64,
    pub overflow_kb: Option<f64>,
    pub board: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NatsBenchMcuV2CoverageRow {
    pub stage: String,
    pub count: u32,
    pub percent: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NatsBenchMcuV2BoardSummary {
    pub board: String,
    pub candidate_count: u32,
    pub mean_latency_s: f64,
    pub mean_power_m_w: f64,
    pub mean_energy_m_j: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NatsBenchMcuV2Summary {
    pub measured_count: u32,
    pub infeasible_count: u32,
    pub mean_latency_s: f64,
    pub median_latency_s: f64,
    pub p90_latency_s: f64,
    pub p99_latency_s: f64,
    pub mean_power_m_w: f64,
    pub power_sd_m_w: f64,
    pub power_cv: f64,
    pub mean_energy_m_j: f64,
    pub median_energy_m_j: f64,
    pub p90_energy_m_j: f64,
    pub p99_energy_m_j: f64,
    pub median_rom_kb: f64,
    pub median_ram_kb: f64,
    pub median_int8_kb: f64,
    pub max_rom_kb: f64,
    pub max_int8_kb: f64,
    pub latency_energy_pearson: f64,
    pub cifar10_latency_pareto_count: u32,
    pub cifar10_energy_pareto_count: u32,
    pub cifar10_rom_pareto_count: u32,
    pub board_summaries: Vec<NatsBenchMcuV2BoardSummary>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NatsBenchMcuV2ImportReceipt {
    pub schema: String,
    pub version: String,
    pub doi: String,
    pub source_commit: String,
    pub manifest_sha256: String,
    pub hardware_sha256: String,
    pub accuracy_sha256: String,
    pub failure_sha256: String,
    pub coverage_sha256: String,
    pub rank_sha256: String,
    pub rank_summary_sha256: String,
    pub baseline_sha256: String,
    pub readme_sha256: String,
    pub summary: NatsBenchMcuV2Summary,
    pub receipt_sha256: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NatsBenchMcuV2ImportedDataset {
    pub hardware: Vec<NatsBenchMcuV2HardwareRow>,
    pub accuracy: Vec<NatsBenchMcuV2AccuracyRow>,
    pub failures: Vec<NatsBenchMcuV2FailureRow>,
    pub coverage: Vec<NatsBenchMcuV2CoverageRow>,
    /// Parsed and schema-checked supplementary source rows, preserved as CSV string cells.
    pub rank_rows: Vec<Vec<String>>,
    pub rank_summary_rows: Vec<Vec<String>>,
    pub baseline_rows: Vec<Vec<String>>,
    pub receipt: NatsBenchMcuV2ImportReceipt,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NatsBenchMcuV2ImportError {
    pub field: String,
    pub message: String,
}

impl NatsBenchMcuV2ImportError {
    fn new(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self { field: field.into(), message: message.into() }
    }
}

impl fmt::Display for NatsBenchMcuV2ImportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.field, self.message)
    }
}

impl Error for NatsBenchMcuV2ImportError {}

/// Pinned corrected v2 publication-export identity from the authors' public repository mirror.
pub fn official_nats_bench_mcu_v2_manifest(
) -> Result<NatsBenchMcuV2SourceManifest, NatsBenchMcuV2ImportError> {
    let mut manifest = NatsBenchMcuV2SourceManifest {
        schema: SOURCE_SCHEMA.into(),
        version: "2.0.0".into(),
        doi: "10.5281/zenodo.21644470".into(),
        source_repository: "https://github.com/ankilab/NATS-Bench-MCU".into(),
        source_commit: "c92adc0ec112bdee6c73358cc52f3e026367b5a0".into(),
        candidate_count: 15_625,
        measured_count: 14_281,
        infeasible_count: 1_344,
        flash_capacity_kb: 1_024.0,
        firmware_overhead_kb: 277.6,
        int8_limit_kb: 746.0,
        rank_rows: 100,
        rank_summary_rows: 4,
        baseline_rows: 3,
        hardware_sha256: "ae318c606d6095ec31ac7383c20e330a6b1e850c8430f6ba20974e00c5d4f693".into(),
        accuracy_sha256: "948fa57fa643dbf749dc394f9f38ab20639737de1a2054d5a72e1be15dd76fae".into(),
        failure_sha256: "88fbc6c4a79f44c2c6c33cdfb5a47eca3cc3a038d88b136cda7b39303413565f".into(),
        coverage_sha256: "f0b6eed8771757d16f4fc0d961251aff4ce850c05f676488ca9e18e306b7da47".into(),
        rank_sha256: "ae764d9267d3f530a9e5396bdee618baed88a6fc65af859ac81cce74427a5596".into(),
        rank_summary_sha256: "b968e3cc98cdb3433263faf9612c41f83ee779eafffb8ff13b0f34802d38d5ad".into(),
        baseline_sha256: "4fad709bb1b6566ee09a780db841d26b3c640564099f7926022ca52a94136fd3".into(),
        readme_sha256: "cac028ee0c345d544b31a7b796622956912294281b1a9cc0f15449ade330c5cb".into(),
        manifest_sha256: "0".repeat(SHA256_LEN),
    };
    validate_manifest_fields(&manifest)?;
    manifest.manifest_sha256 = compute_source_manifest_sha256(&manifest)?;
    Ok(manifest)
}

pub fn compute_source_manifest_sha256(
    manifest: &NatsBenchMcuV2SourceManifest,
) -> Result<String, NatsBenchMcuV2ImportError> {
    validate_manifest_fields(manifest)?;
    let mut canonical = manifest.clone();
    canonical.manifest_sha256.clear();
    let bytes = serde_json::to_vec(&canonical)
        .map_err(|error| NatsBenchMcuV2ImportError::new("manifest", error.to_string()))?;
    Ok(digest_bytes(SOURCE_DOMAIN, &bytes))
}

pub fn validate_source_manifest(
    manifest: &NatsBenchMcuV2SourceManifest,
) -> Result<(), NatsBenchMcuV2ImportError> {
    validate_manifest_fields(manifest)?;
    if !is_sha256(&manifest.manifest_sha256)
        || compute_source_manifest_sha256(manifest)? != manifest.manifest_sha256
    {
        return Err(NatsBenchMcuV2ImportError::new(
            "manifest_sha256",
            "does not match canonical source manifest",
        ));
    }
    Ok(())
}

fn validate_manifest_fields(
    m: &NatsBenchMcuV2SourceManifest,
) -> Result<(), NatsBenchMcuV2ImportError> {
    if m.schema != SOURCE_SCHEMA || m.version != "2.0.0"
        || m.doi != "10.5281/zenodo.21644470"
        || m.source_repository != "https://github.com/ankilab/NATS-Bench-MCU"
    {
        return Err(NatsBenchMcuV2ImportError::new("source_identity", "unsupported NATS-Bench-MCU v2 source identity"));
    }
    if !is_git_sha1(&m.source_commit) {
        return Err(NatsBenchMcuV2ImportError::new("source_commit", "must be a lowercase 40-character Git SHA-1"));
    }
    if m.candidate_count == 0 || m.measured_count + m.infeasible_count != m.candidate_count {
        return Err(NatsBenchMcuV2ImportError::new("population_counts", "measured + infeasible must equal candidate_count"));
    }
    if !m.flash_capacity_kb.is_finite() || !m.firmware_overhead_kb.is_finite()
        || !m.int8_limit_kb.is_finite() || m.flash_capacity_kb <= m.firmware_overhead_kb
        || (m.flash_capacity_kb - m.firmware_overhead_kb - m.int8_limit_kb).abs() > 0.11
    {
        return Err(NatsBenchMcuV2ImportError::new("flash_budget", "flash capacity, firmware overhead and INT8 budget are inconsistent"));
    }
    for (name, value) in [
        ("hardware_sha256", m.hardware_sha256.as_str()),
        ("accuracy_sha256", m.accuracy_sha256.as_str()),
        ("failure_sha256", m.failure_sha256.as_str()),
        ("coverage_sha256", m.coverage_sha256.as_str()),
        ("rank_sha256", m.rank_sha256.as_str()),
        ("rank_summary_sha256", m.rank_summary_sha256.as_str()),
        ("baseline_sha256", m.baseline_sha256.as_str()),
        ("readme_sha256", m.readme_sha256.as_str()),
        ("manifest_sha256", m.manifest_sha256.as_str()),
    ] {
        require_sha256(name, value)?;
    }
    Ok(())
}

pub fn import_nats_bench_mcu_v2(
    manifest: &NatsBenchMcuV2SourceManifest,
    bundle: NatsBenchMcuV2CsvBundle<'_>,
) -> Result<NatsBenchMcuV2ImportedDataset, NatsBenchMcuV2ImportError> {
    validate_source_manifest(manifest)?;
    for (name, contents, expected) in [
        ("hw_metrics.csv", bundle.hardware, manifest.hardware_sha256.as_str()),
        ("accuracies.csv", bundle.accuracy, manifest.accuracy_sha256.as_str()),
        ("failures.csv", bundle.failures, manifest.failure_sha256.as_str()),
        ("coverage.csv", bundle.coverage, manifest.coverage_sha256.as_str()),
        ("int8_rank_correlation.csv", bundle.rank, manifest.rank_sha256.as_str()),
        ("int8_rank_summary.csv", bundle.rank_summary, manifest.rank_summary_sha256.as_str()),
        ("nas_baselines_summary.csv", bundle.baselines, manifest.baseline_sha256.as_str()),
        ("README.md", bundle.readme, manifest.readme_sha256.as_str()),
    ] {
        if sha256_raw(contents.as_bytes()) != expected {
            return Err(NatsBenchMcuV2ImportError::new(
                format!("{name}.sha256"),
                "source content digest differs from the pinned manifest",
            ));
        }
    }

    let hardware = parse_hardware(bundle.hardware)?;
    let accuracy = parse_accuracy(bundle.accuracy)?;
    let failures = parse_failures(bundle.failures, manifest.int8_limit_kb)?;
    let coverage = parse_coverage(bundle.coverage, manifest)?;
    let rank_rows = parse_supplementary(bundle.rank, &RANK_HEADER, "int8_rank_correlation.csv")?;
    let rank_summary_rows = parse_supplementary(bundle.rank_summary, &RANK_SUMMARY_HEADER, "int8_rank_summary.csv")?;
    let baseline_rows = parse_supplementary(bundle.baselines, &BASELINE_HEADER, "nas_baselines_summary.csv")?;

    if hardware.len() != manifest.measured_count as usize
        || accuracy.len() != manifest.measured_count as usize
        || failures.len() != manifest.infeasible_count as usize
        || rank_rows.len() != manifest.rank_rows as usize
        || rank_summary_rows.len() != manifest.rank_summary_rows as usize
        || baseline_rows.len() != manifest.baseline_rows as usize
    {
        return Err(NatsBenchMcuV2ImportError::new("row_counts", "one or more CSV row counts differ from the frozen source manifest"));
    }
    validate_partition(&hardware, &accuracy, &failures, manifest)?;
    let summary = summarize(&hardware, &accuracy, failures.len() as u32)?;
    let mut receipt = NatsBenchMcuV2ImportReceipt {
        schema: RECEIPT_SCHEMA.into(),
        version: VERSION.into(),
        doi: manifest.doi.clone(),
        source_commit: manifest.source_commit.clone(),
        manifest_sha256: manifest.manifest_sha256.clone(),
        hardware_sha256: manifest.hardware_sha256.clone(),
        accuracy_sha256: manifest.accuracy_sha256.clone(),
        failure_sha256: manifest.failure_sha256.clone(),
        coverage_sha256: manifest.coverage_sha256.clone(),
        rank_sha256: manifest.rank_sha256.clone(),
        rank_summary_sha256: manifest.rank_summary_sha256.clone(),
        baseline_sha256: manifest.baseline_sha256.clone(),
        readme_sha256: manifest.readme_sha256.clone(),
        summary,
        receipt_sha256: "0".repeat(SHA256_LEN),
    };
    receipt.receipt_sha256 = compute_import_receipt_sha256(&receipt)?;
    Ok(NatsBenchMcuV2ImportedDataset {
        hardware, accuracy, failures, coverage, rank_rows, rank_summary_rows, baseline_rows, receipt,
    })
}

pub fn compute_import_receipt_sha256(
    receipt: &NatsBenchMcuV2ImportReceipt,
) -> Result<String, NatsBenchMcuV2ImportError> {
    validate_receipt_fields(receipt)?;
    let mut canonical = receipt.clone();
    canonical.receipt_sha256.clear();
    let bytes = serde_json::to_vec(&canonical)
        .map_err(|error| NatsBenchMcuV2ImportError::new("receipt", error.to_string()))?;
    Ok(digest_bytes(RECEIPT_DOMAIN, &bytes))
}

pub fn verify_import_receipt(
    manifest: &NatsBenchMcuV2SourceManifest,
    bundle: NatsBenchMcuV2CsvBundle<'_>,
    receipt: &NatsBenchMcuV2ImportReceipt,
) -> Result<(), NatsBenchMcuV2ImportError> {
    validate_receipt_fields(receipt)?;
    if compute_import_receipt_sha256(receipt)? != receipt.receipt_sha256 {
        return Err(NatsBenchMcuV2ImportError::new("receipt_sha256", "canonical receipt digest mismatch"));
    }
    if import_nats_bench_mcu_v2(manifest, bundle)?.receipt != *receipt {
        return Err(NatsBenchMcuV2ImportError::new("receipt", "independent CSV re-import differs from supplied receipt"));
    }
    Ok(())
}

fn parse_hardware(input: &str) -> Result<Vec<NatsBenchMcuV2HardwareRow>, NatsBenchMcuV2ImportError> {
    let rows = parse_csv(input, "hw_metrics.csv")?;
    let data = header_rows(rows, &HW_HEADER, "hw_metrics.csv")?;
    let mut out = Vec::with_capacity(data.len());
    let mut ids = HashSet::with_capacity(data.len());
    for (i, row) in data.iter().enumerate() {
        let line = i + 2;
        let arch_idx = integer(cell(row, 0, "hw_metrics.csv", line)?, "hw_metrics.csv", line, "arch_idx")?;
        if !ids.insert(arch_idx) {
            return Err(NatsBenchMcuV2ImportError::new(format!("hw_metrics.csv:{line}.arch_idx"), "duplicate candidate ID"));
        }
        let arch_str = required(cell(row, 1, "hw_metrics.csv", line)?, "hw_metrics.csv", line, "arch_str")?.to_owned();
        let int8_kb = number(cell(row, 2, "hw_metrics.csv", line)?, "hw_metrics.csv", line, "int8_kb")?;
        let rom_kb = number(cell(row, 3, "hw_metrics.csv", line)?, "hw_metrics.csv", line, "rom_kb")?;
        let ram_kb = number(cell(row, 4, "hw_metrics.csv", line)?, "hw_metrics.csv", line, "ram_kb")?;
        let latency_s = number(cell(row, 5, "hw_metrics.csv", line)?, "hw_metrics.csv", line, "latency_s")?;
        let current_u_a = number(cell(row, 6, "hw_metrics.csv", line)?, "hw_metrics.csv", line, "current_uA")?;
        let power_m_w = number(cell(row, 7, "hw_metrics.csv", line)?, "hw_metrics.csv", line, "power_mW")?;
        let energy_m_j = number(cell(row, 8, "hw_metrics.csv", line)?, "hw_metrics.csv", line, "energy_mJ")?;
        let board = required(cell(row, 9, "hw_metrics.csv", line)?, "hw_metrics.csv", line, "board")?.to_owned();
        let g_model_kb = number(cell(row, 10, "hw_metrics.csv", line)?, "hw_metrics.csv", line, "g_model_kb")?;
        let arena_used_kb = number(cell(row, 11, "hw_metrics.csv", line)?, "hw_metrics.csv", line, "arena_used_kb")?;
        let arena_size_kb = number(cell(row, 12, "hw_metrics.csv", line)?, "hw_metrics.csv", line, "arena_size_kb")?;
        let arena_used_pct = number(cell(row, 13, "hw_metrics.csv", line)?, "hw_metrics.csv", line, "arena_used_pct")?;
        let operators = integer(cell(row, 14, "hw_metrics.csv", line)?, "hw_metrics.csv", line, "operators")?;
        let rom_categories_kb = category_map(cell(row, 15, "hw_metrics.csv", line)?, line, "rom_categories_json")?;
        let ram_categories_kb = category_map(cell(row, 16, "hw_metrics.csv", line)?, line, "ram_categories_json")?;
        for (field, value) in [
            ("int8_kb", int8_kb), ("rom_kb", rom_kb), ("ram_kb", ram_kb),
            ("latency_s", latency_s), ("current_uA", current_u_a),
            ("power_mW", power_m_w), ("energy_mJ", energy_m_j),
            ("g_model_kb", g_model_kb), ("arena_size_kb", arena_size_kb),
        ] {
            if value <= 0.0 {
                return Err(NatsBenchMcuV2ImportError::new(format!("hw_metrics.csv:{line}.{field}"), "must be positive"));
            }
        }
        if arena_used_kb < 0.0 || !(0.0..=100.0).contains(&arena_used_pct) || operators == 0
            || !board.starts_with("nrf5340dk_")
        {
            return Err(NatsBenchMcuV2ImportError::new(format!("hw_metrics.csv:{line}"), "invalid board/runtime allocation fields"));
        }
        out.push(NatsBenchMcuV2HardwareRow {
            arch_idx, arch_str, int8_kb, rom_kb, ram_kb, latency_s, current_u_a,
            power_m_w, energy_m_j, board, g_model_kb, arena_used_kb, arena_size_kb,
            arena_used_pct, operators, rom_categories_kb, ram_categories_kb,
        });
    }
    Ok(out)
}

fn parse_accuracy(input: &str) -> Result<Vec<NatsBenchMcuV2AccuracyRow>, NatsBenchMcuV2ImportError> {
    let rows = parse_csv(input, "accuracies.csv")?;
    let data = header_rows(rows, &ACC_HEADER, "accuracies.csv")?;
    let mut out = Vec::with_capacity(data.len());
    let mut ids = HashSet::with_capacity(data.len());
    for (i, row) in data.iter().enumerate() {
        let line = i + 2;
        let arch_idx = integer(cell(row, 0, "accuracies.csv", line)?, "accuracies.csv", line, "arch_idx")?;
        if !ids.insert(arch_idx) {
            return Err(NatsBenchMcuV2ImportError::new(format!("accuracies.csv:{line}.arch_idx"), "duplicate candidate ID"));
        }
        out.push(NatsBenchMcuV2AccuracyRow {
            arch_idx,
            cifar10: percentage(cell(row, 1, "accuracies.csv", line)?, line, "cifar10")?,
            cifar100: percentage(cell(row, 2, "accuracies.csv", line)?, line, "cifar100")?,
            imagenet16_120: percentage(cell(row, 3, "accuracies.csv", line)?, line, "ImageNet16-120")?,
        });
    }
    Ok(out)
}

fn parse_failures(input: &str, limit: f64) -> Result<Vec<NatsBenchMcuV2FailureRow>, NatsBenchMcuV2ImportError> {
    let rows = parse_csv(input, "failures.csv")?;
    let data = header_rows(rows, &FAIL_HEADER, "failures.csv")?;
    let mut out = Vec::with_capacity(data.len());
    let mut ids = HashSet::with_capacity(data.len());
    for (i, row) in data.iter().enumerate() {
        let line = i + 2;
        let arch_idx = integer(cell(row, 0, "failures.csv", line)?, "failures.csv", line, "arch_idx")?;
        if !ids.insert(arch_idx) {
            return Err(NatsBenchMcuV2ImportError::new(format!("failures.csv:{line}.arch_idx"), "duplicate candidate ID"));
        }
        let stage = required(cell(row, 1, "failures.csv", line)?, "failures.csv", line, "stage")?.to_owned();
        if stage != "infeasible" {
            return Err(NatsBenchMcuV2ImportError::new(format!("failures.csv:{line}.stage"), "noncanonical v2 failure label"));
        }
        let int8_kb = number(cell(row, 2, "failures.csv", line)?, "failures.csv", line, "int8_kb")?;
        if int8_kb <= limit {
            return Err(NatsBenchMcuV2ImportError::new(format!("failures.csv:{line}.int8_kb"), "failure must exceed pinned INT8 limit"));
        }
        let overflow = optional_number(cell(row, 3, "failures.csv", line)?, "failures.csv", line, "overflow_kb")?;
        let board = required(cell(row, 4, "failures.csv", line)?, "failures.csv", line, "board")?.to_owned();
        out.push(NatsBenchMcuV2FailureRow { arch_idx, stage, int8_kb, overflow_kb: overflow, board });
    }
    Ok(out)
}

fn parse_coverage(input: &str, manifest: &NatsBenchMcuV2SourceManifest) -> Result<Vec<NatsBenchMcuV2CoverageRow>, NatsBenchMcuV2ImportError> {
    let rows = parse_csv(input, "coverage.csv")?;
    let data = header_rows(rows, &COVERAGE_HEADER, "coverage.csv")?;
    let mut out = Vec::with_capacity(data.len());
    let mut ids = HashSet::with_capacity(data.len());
    for (i, row) in data.iter().enumerate() {
        let line = i + 2;
        let stage = required(cell(row, 0, "coverage.csv", line)?, "coverage.csv", line, "stage")?.to_owned();
        if !ids.insert(stage.clone()) {
            return Err(NatsBenchMcuV2ImportError::new(format!("coverage.csv:{line}.stage"), "duplicate coverage stage"));
        }
        let count = integer(cell(row, 1, "coverage.csv", line)?, "coverage.csv", line, "count")?;
        let percent = number(cell(row, 2, "coverage.csv", line)?, "coverage.csv", line, "pct_of_total")?;
        if !(0.0..=100.0).contains(&percent) {
            return Err(NatsBenchMcuV2ImportError::new(format!("coverage.csv:{line}.pct_of_total"), "must be in [0,100]"));
        }
        out.push(NatsBenchMcuV2CoverageRow { stage, count, percent });
    }
    let lookup: HashMap<&str, &NatsBenchMcuV2CoverageRow> = out.iter().map(|row| (row.stage.as_str(), row)).collect();
    for (name, count) in [
        ("NATS-Bench (target)", manifest.candidate_count),
        ("TFLite converted", manifest.candidate_count),
        ("Submitted", manifest.candidate_count),
        ("On disk", manifest.candidate_count),
        ("Successful", manifest.measured_count),
    ] {
        let row = lookup.get(name).ok_or_else(|| NatsBenchMcuV2ImportError::new("coverage.csv.stage", format!("missing stage {name}")))?;
        let expected_percent = 100.0 * count as f64 / manifest.candidate_count as f64;
        if row.count != count || (row.percent - expected_percent).abs() > 0.11 {
            return Err(NatsBenchMcuV2ImportError::new(format!("coverage.csv.{name}"), "coverage count or percentage mismatch"));
        }
    }
    if out.len() != 5 {
        return Err(NatsBenchMcuV2ImportError::new("coverage.csv.rows", "expected exactly five pipeline stages"));
    }
    Ok(out)
}

fn parse_supplementary<const N: usize>(
    input: &str,
    expected: &[&str; N],
    file: &str,
) -> Result<Vec<Vec<String>>, NatsBenchMcuV2ImportError> {
    let rows = parse_csv(input, file)?;
    header_rows(rows, expected, file)
}

fn validate_partition(
    hardware: &[NatsBenchMcuV2HardwareRow],
    accuracy: &[NatsBenchMcuV2AccuracyRow],
    failures: &[NatsBenchMcuV2FailureRow],
    manifest: &NatsBenchMcuV2SourceManifest,
) -> Result<(), NatsBenchMcuV2ImportError> {
    let total = manifest.candidate_count;
    let mut success_ids = HashSet::with_capacity(hardware.len());
    let mut accuracy_ids = HashSet::with_capacity(accuracy.len());
    let mut failure_ids = HashSet::with_capacity(failures.len());
    for row in hardware {
        if row.arch_idx >= total
            || row.int8_kb > manifest.int8_limit_kb + FLOAT_TOLERANCE
            || row.rom_kb > manifest.flash_capacity_kb + FLOAT_TOLERANCE
        {
            return Err(NatsBenchMcuV2ImportError::new(format!("hardware.{}", row.arch_idx), "measured row is out of range or exceeds flash constraints"));
        }
        success_ids.insert(row.arch_idx);
    }
    for row in accuracy {
        if row.arch_idx >= total || !accuracy_ids.insert(row.arch_idx) {
            return Err(NatsBenchMcuV2ImportError::new("accuracy.arch_idx", "out-of-range or duplicate accuracy ID"));
        }
    }
    for row in failures {
        if row.arch_idx >= total || !failure_ids.insert(row.arch_idx) || success_ids.contains(&row.arch_idx) {
            return Err(NatsBenchMcuV2ImportError::new("failures.arch_idx", "out-of-range, duplicate, or overlapping failure ID"));
        }
    }
    if success_ids != accuracy_ids || success_ids.len() + failure_ids.len() != total as usize {
        return Err(NatsBenchMcuV2ImportError::new("partition", "accuracy join or complete success/failure partition failed"));
    }
    for id in 0..total {
        if !success_ids.contains(&id) && !failure_ids.contains(&id) {
            return Err(NatsBenchMcuV2ImportError::new("partition", format!("candidate ID {id} is missing")));
        }
    }
    Ok(())
}

fn summarize(
    hardware: &[NatsBenchMcuV2HardwareRow],
    accuracy: &[NatsBenchMcuV2AccuracyRow],
    failures: u32,
) -> Result<NatsBenchMcuV2Summary, NatsBenchMcuV2ImportError> {
    if hardware.is_empty() { return Err(NatsBenchMcuV2ImportError::new("hardware", "no rows")); }
    let acc: HashMap<u32, &NatsBenchMcuV2AccuracyRow> = accuracy.iter().map(|row| (row.arch_idx, row)).collect();
    let latency: Vec<f64> = hardware.iter().map(|row| row.latency_s).collect();
    let power: Vec<f64> = hardware.iter().map(|row| row.power_m_w).collect();
    let energy: Vec<f64> = hardware.iter().map(|row| row.energy_m_j).collect();
    let mean_power = mean(power.iter().copied());
    let power_sd = stddev(&power);
    let mut groups: BTreeMap<String, Vec<&NatsBenchMcuV2HardwareRow>> = BTreeMap::new();
    for row in hardware { groups.entry(row.board.clone()).or_default().push(row); }
    let board_summaries = groups.into_iter().map(|(board, rows)| NatsBenchMcuV2BoardSummary {
        board,
        candidate_count: rows.len() as u32,
        mean_latency_s: mean(rows.iter().map(|r| r.latency_s)),
        mean_power_m_w: mean(rows.iter().map(|r| r.power_m_w)),
        mean_energy_m_j: mean(rows.iter().map(|r| r.energy_m_j)),
    }).collect();
    Ok(NatsBenchMcuV2Summary {
        measured_count: hardware.len() as u32,
        infeasible_count: failures,
        mean_latency_s: mean(latency.iter().copied()),
        median_latency_s: quantile(&latency, 0.5)?,
        p90_latency_s: quantile(&latency, 0.9)?,
        p99_latency_s: quantile(&latency, 0.99)?,
        mean_power_m_w: mean_power,
        power_sd_m_w: power_sd,
        power_cv: power_sd / mean_power,
        mean_energy_m_j: mean(energy.iter().copied()),
        median_energy_m_j: quantile(&energy, 0.5)?,
        p90_energy_m_j: quantile(&energy, 0.9)?,
        p99_energy_m_j: quantile(&energy, 0.99)?,
        median_rom_kb: quantile(&hardware.iter().map(|r| r.rom_kb).collect::<Vec<_>>(), 0.5)?,
        median_ram_kb: quantile(&hardware.iter().map(|r| r.ram_kb).collect::<Vec<_>>(), 0.5)?,
        median_int8_kb: quantile(&hardware.iter().map(|r| r.int8_kb).collect::<Vec<_>>(), 0.5)?,
        max_rom_kb: hardware.iter().map(|r| r.rom_kb).fold(f64::NEG_INFINITY, f64::max),
        max_int8_kb: hardware.iter().map(|r| r.int8_kb).fold(f64::NEG_INFINITY, f64::max),
        latency_energy_pearson: pearson(&latency, &energy)?,
        cifar10_latency_pareto_count: pareto_count(hardware, &acc, |a| acc[&a.arch_idx].cifar10, |a| a.latency_s),
        cifar10_energy_pareto_count: pareto_count(hardware, &acc, |a| acc[&a.arch_idx].cifar10, |a| a.energy_m_j),
        cifar10_rom_pareto_count: pareto_count(hardware, &acc, |a| acc[&a.arch_idx].cifar10, |a| a.rom_kb),
        board_summaries,
    })
}

fn pareto_count<A, C>(
    rows: &[NatsBenchMcuV2HardwareRow],
    _accuracy: &HashMap<u32, &NatsBenchMcuV2AccuracyRow>,
    a: A,
    c: C,
) -> u32
where
    A: Fn(&NatsBenchMcuV2HardwareRow) -> f64,
    C: Fn(&NatsBenchMcuV2HardwareRow) -> f64,
{
    let mut sorted: Vec<&NatsBenchMcuV2HardwareRow> = rows.iter().collect();
    sorted.sort_by(|x, y| a(y).total_cmp(&a(x)).then_with(|| c(x).total_cmp(&c(y))).then_with(|| x.arch_idx.cmp(&y.arch_idx)));
    let mut count = 0u32;
    let mut best_cost = f64::INFINITY;
    let mut i = 0;
    while i < sorted.len() {
        let score = a(sorted[i]);
        let mut j = i + 1;
        while j < sorted.len() && a(sorted[j]) == score { j += 1; }
        let group_best = sorted[i..j].iter().map(|r| c(r)).fold(f64::INFINITY, f64::min);
        if group_best < best_cost {
            count += sorted[i..j].iter().filter(|r| c(r) == group_best).count() as u32;
            best_cost = group_best;
        }
        i = j;
    }
    count
}

fn parse_csv(input: &str, file: &str) -> Result<Vec<Vec<String>>, NatsBenchMcuV2ImportError> {
    let bytes = input.as_bytes();
    let bytes = if bytes.starts_with(&[0xef, 0xbb, 0xbf]) { &bytes[3..] } else { bytes };
    let mut rows = Vec::new();
    let mut row = Vec::new();
    let mut field = Vec::new();
    let mut quoted = false;
    let mut after_quote = false;
    let mut line = 1usize;
    let mut i = 0usize;
    while i < bytes.len() {
        let b = bytes[i];
        if quoted {
            if b == b'"' {
                if bytes.get(i + 1) == Some(&b'"') { field.push(b'"'); i += 2; }
                else { quoted = false; after_quote = true; i += 1; }
            } else {
                field.push(b);
                if b == b'\n' { line += 1; }
                i += 1;
            }
            continue;
        }
        if after_quote {
            match b {
                b',' => { row.push(decode_field(std::mem::take(&mut field), file, line)?); after_quote = false; i += 1; }
                b'\n' => { row.push(decode_field(std::mem::take(&mut field), file, line)?); rows.push(std::mem::take(&mut row)); after_quote = false; line += 1; i += 1; }
                b'\r' if bytes.get(i + 1) == Some(&b'\n') => { row.push(decode_field(std::mem::take(&mut field), file, line)?); rows.push(std::mem::take(&mut row)); after_quote = false; line += 1; i += 2; }
                _ => return Err(NatsBenchMcuV2ImportError::new(format!("{file}:{line}"), "invalid data after closing quote")),
            }
            continue;
        }
        match b {
            b'"' if field.is_empty() => { quoted = true; i += 1; }
            b'"' => return Err(NatsBenchMcuV2ImportError::new(format!("{file}:{line}"), "quote inside unquoted field")),
            b',' => { row.push(decode_field(std::mem::take(&mut field), file, line)?); i += 1; }
            b'\n' => { row.push(decode_field(std::mem::take(&mut field), file, line)?); if !row.iter().all(|s| s.is_empty()) { rows.push(std::mem::take(&mut row)); } else { row.clear(); } line += 1; i += 1; }
            b'\r' if bytes.get(i + 1) == Some(&b'\n') => { row.push(decode_field(std::mem::take(&mut field), file, line)?); if !row.iter().all(|s| s.is_empty()) { rows.push(std::mem::take(&mut row)); } else { row.clear(); } line += 1; i += 2; }
            b'\r' => return Err(NatsBenchMcuV2ImportError::new(format!("{file}:{line}"), "bare carriage return")),
            _ => { field.push(b); i += 1; }
        }
    }
    if quoted { return Err(NatsBenchMcuV2ImportError::new(format!("{file}:{line}"), "unterminated quoted field")); }
    if after_quote || !field.is_empty() || !row.is_empty() {
        row.push(decode_field(field, file, line)?);
        rows.push(row);
    }
    if rows.is_empty() { return Err(NatsBenchMcuV2ImportError::new(file, "empty CSV")); }
    Ok(rows)
}

fn decode_field(bytes: Vec<u8>, file: &str, line: usize) -> Result<String, NatsBenchMcuV2ImportError> {
    String::from_utf8(bytes).map_err(|e| NatsBenchMcuV2ImportError::new(format!("{file}:{line}"), e.to_string()))
}

fn header_rows<const N: usize>(
    mut rows: Vec<Vec<String>>,
    expected: &[&str; N],
    file: &str,
) -> Result<Vec<Vec<String>>, NatsBenchMcuV2ImportError> {
    let header = rows.remove(0);
    if header.len() != N || header.iter().zip(expected).any(|(a, b)| a != b) {
        return Err(NatsBenchMcuV2ImportError::new(format!("{file}.header"), format!("expected {}", expected.join(","))));
    }
    for (i, row) in rows.iter().enumerate() {
        if row.len() != N {
            return Err(NatsBenchMcuV2ImportError::new(format!("{file}:{}", i + 2), format!("expected {N} cells, found {}", row.len())));
        }
    }
    Ok(rows)
}

fn cell<'a>(row: &'a [String], i: usize, file: &str, line: usize) -> Result<&'a str, NatsBenchMcuV2ImportError> {
    row.get(i).map(String::as_str).ok_or_else(|| NatsBenchMcuV2ImportError::new(format!("{file}:{line}"), "missing cell"))
}
fn required<'a>(v: &'a str, file: &str, line: usize, field: &str) -> Result<&'a str, NatsBenchMcuV2ImportError> {
    if v.trim().is_empty() { Err(NatsBenchMcuV2ImportError::new(format!("{file}:{line}.{field}"), "empty value")) } else { Ok(v) }
}
fn number(v: &str, file: &str, line: usize, field: &str) -> Result<f64, NatsBenchMcuV2ImportError> {
    let n = v.parse::<f64>().map_err(|e| NatsBenchMcuV2ImportError::new(format!("{file}:{line}.{field}"), e.to_string()))?;
    if !n.is_finite() { return Err(NatsBenchMcuV2ImportError::new(format!("{file}:{line}.{field}"), "non-finite value")); }
    Ok(n)
}
fn optional_number(v: &str, file: &str, line: usize, field: &str) -> Result<Option<f64>, NatsBenchMcuV2ImportError> {
    if v.trim().is_empty() { Ok(None) } else { number(v, file, line, field).map(Some) }
}
fn integer(v: &str, file: &str, line: usize, field: &str) -> Result<u32, NatsBenchMcuV2ImportError> {
    v.parse::<u32>().map_err(|e| NatsBenchMcuV2ImportError::new(format!("{file}:{line}.{field}"), e.to_string()))
}
fn percentage(v: &str, line: usize, field: &str) -> Result<f64, NatsBenchMcuV2ImportError> {
    let n = number(v, "accuracies.csv", line, field)?;
    if !(0.0..=100.0).contains(&n) { return Err(NatsBenchMcuV2ImportError::new(format!("accuracies.csv:{line}.{field}"), "percentage outside [0,100]")); }
    Ok(n)
}
fn category_map(v: &str, line: usize, field: &str) -> Result<BTreeMap<String, f64>, NatsBenchMcuV2ImportError> {
    let map: BTreeMap<String, f64> = serde_json::from_str(v).map_err(|e| NatsBenchMcuV2ImportError::new(format!("hw_metrics.csv:{line}.{field}"), e.to_string()))?;
    if map.is_empty() || map.iter().any(|(k, n)| k.trim().is_empty() || !n.is_finite() || *n < 0.0) {
        return Err(NatsBenchMcuV2ImportError::new(format!("hw_metrics.csv:{line}.{field}"), "invalid memory-category map"));
    }
    Ok(map)
}

fn validate_receipt_fields(r: &NatsBenchMcuV2ImportReceipt) -> Result<(), NatsBenchMcuV2ImportError> {
    if r.schema != RECEIPT_SCHEMA || r.version != VERSION {
        return Err(NatsBenchMcuV2ImportError::new("schema_version", "unsupported receipt schema"));
    }
    for (field, value) in [
        ("manifest_sha256", r.manifest_sha256.as_str()), ("hardware_sha256", r.hardware_sha256.as_str()),
        ("accuracy_sha256", r.accuracy_sha256.as_str()), ("failure_sha256", r.failure_sha256.as_str()),
        ("coverage_sha256", r.coverage_sha256.as_str()), ("rank_sha256", r.rank_sha256.as_str()),
        ("rank_summary_sha256", r.rank_summary_sha256.as_str()), ("baseline_sha256", r.baseline_sha256.as_str()),
        ("readme_sha256", r.readme_sha256.as_str()), ("receipt_sha256", r.receipt_sha256.as_str()),
    ] { require_sha256(field, value)?; }
    if r.measured_count + r.infeasible_count != r.candidate_count {
        return Err(NatsBenchMcuV2ImportError::new("candidate_counts", "partition totals mismatch"));
    }
    Ok(())
}
fn require_sha256(field: &str, v: &str) -> Result<(), NatsBenchMcuV2ImportError> {
    if is_sha256(v) { Ok(()) } else { Err(NatsBenchMcuV2ImportError::new(field, "expected lowercase SHA-256")) }
}
fn is_sha256(v: &str) -> bool { v.len() == SHA256_LEN && v.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) }
fn is_git_sha1(v: &str) -> bool { v.len() == SHA1_LEN && v.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) }
fn sha256_raw(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(SHA256_LEN);
    for b in digest { use std::fmt::Write as _; let _ = write!(&mut out, "{b:02x}"); }
    out
}
fn digest_bytes(domain: &[u8], bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update((domain.len() as u64).to_be_bytes()); h.update(domain);
    h.update((bytes.len() as u64).to_be_bytes()); h.update(bytes);
    let mut out = String::with_capacity(SHA256_LEN);
    for b in h.finalize() { use std::fmt::Write as _; let _ = write!(&mut out, "{b:02x}"); }
    out
}
fn mean(values: impl Iterator<Item = f64>) -> f64 {
    let mut sum = 0.0; let mut n = 0u64;
    for v in values { sum += v; n += 1; }
    sum / n as f64
}
fn stddev(values: &[f64]) -> f64 {
    let m = mean(values.iter().copied());
    mean(values.iter().map(|v| (v - m).powi(2))).sqrt()
}
fn quantile(values: &[f64], p: f64) -> Result<f64, NatsBenchMcuV2ImportError> {
    if values.is_empty() || !(0.0..=1.0).contains(&p) { return Err(NatsBenchMcuV2ImportError::new("quantile", "invalid input")); }
    let mut sorted = values.to_vec(); sorted.sort_by(f64::total_cmp);
    let pos = (sorted.len() - 1) as f64 * p;
    let lo = pos.floor() as usize; let hi = pos.ceil() as usize;
    Ok(sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo as f64))
}
fn pearson(a: &[f64], b: &[f64]) -> Result<f64, NatsBenchMcuV2ImportError> {
    if a.len() != b.len() || a.len() < 2 { return Err(NatsBenchMcuV2ImportError::new("pearson", "invalid paired vectors")); }
    let ma = mean(a.iter().copied()); let mb = mean(b.iter().copied());
    let mut n = 0.0; let mut da = 0.0; let mut db = 0.0;
    for (x, y) in a.iter().zip(b) {
        let cx = x - ma; let cy = y - mb; n += cx * cy; da += cx * cx; db += cy * cy;
    }
    if da == 0.0 || db == 0.0 { return Err(NatsBenchMcuV2ImportError::new("pearson", "zero variance")); }
    Ok(n / (da * db).sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn csv(fields: &[&str]) -> String {
        fields.iter().map(|v| {
            if v.contains(',') || v.contains('"') { format!("\"{}\"", v.replace('"', "\"\"")) } else { (*v).into() }
        }).collect::<Vec<_>>().join(",")
    }
    fn sha(c: char) -> String { std::iter::repeat_n(c, SHA256_LEN).collect() }

    struct Fixture {
        hw: String, acc: String, fail: String, cov: String, rank: String,
        rs: String, base: String, readme: String, manifest: NatsBenchMcuV2SourceManifest,
    }
    impl Fixture {
        fn new() -> Self {
            let hw = format!(
                "{}\n{}\n{}\n",
                HW_HEADER.join(","),
                csv(&["0", "|conv, op~0|+|skip_connect~1|", "100", "377.6", "366.2861", "1", "3400", "10", "10", "nrf5340dk_1", "100", "10", "350", "2.86", "16", "{\"WORKSPACE\":100,\"(no paths)\":277.6}", "{\"WORKSPACE\":10}"]),
                csv(&["2", "|avg_pool_3x3~0|+|skip_connect~1|", "200", "477.6", "366.2861", "2", "3600", "11", "22", "nrf5340dk_2", "200", "20", "350", "5.71", "24", "{\"WORKSPACE\":200,\"(no paths)\":277.6}", "{\"WORKSPACE\":20}"]),
            );
            let acc = format!("{}\n{}\n{}\n", ACC_HEADER.join(","), csv(&["0","90","70","45"]), csv(&["2","89","69","44"]));
            let fail = format!("{}\n{}\n", FAIL_HEADER.join(","), csv(&["1","infeasible","747","","nrf5340dk_3"]));
            let cov = format!("{}\n{}\n{}\n{}\n{}\n{}\n",
                COVERAGE_HEADER.join(","),
                csv(&["NATS-Bench (target)","3","100"]),
                csv(&["TFLite converted","3","100"]),
                csv(&["Submitted","3","100"]),
                csv(&["On disk","3","100"]),
                csv(&["Successful","2","66.7"]),
            );
            let rank = format!("{}\n{}\n", RANK_HEADER.join(","), csv(&["0","|conv, op~0|","90","0.90","0.89","1.0","0"]));
            let rs = format!("{}\n{}\n", RANK_SUMMARY_HEADER.join(","), csv(&["a_fp32_int8_all","100","0.999","0.9821","9","25"]));
            let base = format!("{}\n{}\n", BASELINE_HEADER.join(","), csv(&["Regularized Evolution","92.81","0.44","110","0.332"]));
            let readme = "miniature test fixture".to_string();
            let mut manifest = NatsBenchMcuV2SourceManifest {
                schema: SOURCE_SCHEMA.into(), version: "2.0.0".into(),
                doi: "10.5281/zenodo.21644470".into(),
                source_repository: "https://github.com/ankilab/NATS-Bench-MCU".into(),
                source_commit: "a".repeat(SHA1_LEN),
                candidate_count: 3, measured_count: 2, infeasible_count: 1,
                flash_capacity_kb: 1024.0, firmware_overhead_kb: 277.6,
                int8_limit_kb: 746.0, rank_rows: 1, rank_summary_rows: 1, baseline_rows: 1,
                hardware_sha256: sha256_raw(hw.as_bytes()), accuracy_sha256: sha256_raw(acc.as_bytes()),
                failure_sha256: sha256_raw(fail.as_bytes()), coverage_sha256: sha256_raw(cov.as_bytes()),
                rank_sha256: sha256_raw(rank.as_bytes()), rank_summary_sha256: sha256_raw(rs.as_bytes()),
                baseline_sha256: sha256_raw(base.as_bytes()), readme_sha256: sha256_raw(readme.as_bytes()),
                manifest_sha256: "0".repeat(SHA256_LEN),
            };
            manifest.manifest_sha256 = compute_source_manifest_sha256(&manifest).unwrap();
            Self { hw, acc, fail, cov, rank, rs, base, readme, manifest }
        }
        fn bundle(&self) -> NatsBenchMcuV2CsvBundle<'_> {
            NatsBenchMcuV2CsvBundle { hardware:&self.hw, accuracy:&self.acc, failures:&self.fail, coverage:&self.cov, rank:&self.rank, rank_summary:&self.rs, baselines:&self.base, readme:&self.readme }
        }
        fn refresh(&mut self) {
            self.manifest.hardware_sha256=sha256_raw(self.hw.as_bytes());
            self.manifest.accuracy_sha256=sha256_raw(self.acc.as_bytes());
            self.manifest.failure_sha256=sha256_raw(self.fail.as_bytes());
            self.manifest.coverage_sha256=sha256_raw(self.cov.as_bytes());
            self.manifest.rank_sha256=sha256_raw(self.rank.as_bytes());
            self.manifest.rank_summary_sha256=sha256_raw(self.rs.as_bytes());
            self.manifest.baseline_sha256=sha256_raw(self.base.as_bytes());
            self.manifest.readme_sha256=sha256_raw(self.readme.as_bytes());
            self.manifest.manifest_sha256="0".repeat(SHA256_LEN);
            self.manifest.manifest_sha256=compute_source_manifest_sha256(&self.manifest).unwrap();
        }
    }

    #[test]
    fn valid_import_checks_partition_join_embedded_json_and_summary() {
        let f=Fixture::new();
        let dataset=import_nats_bench_mcu_v2(&f.manifest,f.bundle()).unwrap();
        assert_eq!(dataset.hardware.len(),2);
        assert_eq!(dataset.accuracy.len(),2);
        assert_eq!(dataset.failures.len(),1);
        assert_eq!(dataset.receipt.summary.measured_count,2);
        assert_eq!(dataset.receipt.summary.infeasible_count,1);
        assert_eq!(dataset.hardware[0].arch_str,"|conv, op~0|+|skip_connect~1|");
        verify_import_receipt(&f.manifest,f.bundle(),&dataset.receipt).unwrap();
    }

    #[test]
    fn source_digest_drift_is_rejected() {
        let f=Fixture::new();
        let mut manifest=f.manifest.clone();
        manifest.hardware_sha256=sha('0');
        manifest.manifest_sha256="0".repeat(SHA256_LEN);
        manifest.manifest_sha256=compute_source_manifest_sha256(&manifest).unwrap();
        assert!(import_nats_bench_mcu_v2(&manifest,f.bundle()).is_err());
    }

    #[test]
    fn duplicate_candidate_id_is_rejected() {
        let mut f=Fixture::new();
        let lines:Vec<&str>=f.hw.lines().collect();
        f.hw=format!("{}\n{}\n{}\n",lines[0],lines[1],lines[1]);
        f.refresh();
        assert!(import_nats_bench_mcu_v2(&f.manifest,f.bundle()).unwrap_err().to_string().contains("duplicate candidate ID"));
    }

    #[test]
    fn failure_taxonomy_and_flash_ceiling_are_fail_closed() {
        let mut f=Fixture::new();
        f.fail=f.fail.replace("infeasible","flash_failed");
        f.refresh();
        assert!(import_nats_bench_mcu_v2(&f.manifest,f.bundle()).is_err());
        let mut f=Fixture::new();
        f.fail=f.fail.replace("747","745");
        f.refresh();
        assert!(import_nats_bench_mcu_v2(&f.manifest,f.bundle()).is_err());
    }

    #[test]
    fn accuracy_join_and_population_gaps_are_rejected() {
        let mut f=Fixture::new();
        f.acc=f.acc.replace("\n2,89", "\n1,89");
        f.refresh();
        assert!(import_nats_bench_mcu_v2(&f.manifest,f.bundle()).unwrap_err().to_string().contains("accuracy join"));
    }

    #[test]
    fn non_finite_hardware_measurement_is_rejected() {
        let mut f=Fixture::new();
        f.hw=f.hw.replace(",1,3400", ",NaN,3400");
        f.refresh();
        assert!(import_nats_bench_mcu_v2(&f.manifest,f.bundle()).unwrap_err().to_string().contains("non-finite"));
    }

    #[test]
    fn malformed_csv_quotes_and_unknown_headers_are_rejected() {
        let mut f=Fixture::new();
        f.hw=f.hw.replace("arch_idx,arch_str", "arch_idx,architecture");
        f.refresh();
        assert!(import_nats_bench_mcu_v2(&f.manifest,f.bundle()).is_err());
        let mut f=Fixture::new();
        f.hw=f.hw.replace("|conv, op~0|", "|conv, op~0\"");
        f.refresh();
        assert!(import_nats_bench_mcu_v2(&f.manifest,f.bundle()).is_err());
    }

    #[test]
    fn official_manifest_pins_nats_v2_counts_and_source_commit() {
        let m=official_nats_bench_mcu_v2_manifest().unwrap();
        validate_source_manifest(&m).unwrap();
        assert_eq!(m.candidate_count,15_625);
        assert_eq!(m.measured_count,14_281);
        assert_eq!(m.infeasible_count,1_344);
        assert_eq!(m.source_commit,"c92adc0ec112bdee6c73358cc52f3e026367b5a0");
        assert_eq!(m.int8_limit_kb,746.0);
    }

    #[test]
    fn category_json_must_be_finite_and_nonempty() {
        assert_eq!(category_map("{\"A\":1.0}",2,"rom_categories_json").unwrap().len(),1);
        assert!(category_map("{\"A\":NaN}",2,"rom_categories_json").is_err());
        assert!(category_map("{}",2,"rom_categories_json").is_err());
    }
}
