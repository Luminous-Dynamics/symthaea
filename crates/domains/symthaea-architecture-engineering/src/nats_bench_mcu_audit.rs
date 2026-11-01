//! Auditable import contract for the corrected NATS-Bench-MCU v2 publication export.
//!
//! This module validates the exhaustive architecture partition, joins accuracy rows to
//! measured hardware rows, checks the published INT8 feasibility boundary, hashes the exact
//! source CSV bytes, and reconstructs two-objective accuracy/resource Pareto fronts.
//!
//! It does not authenticate the Zenodo download, reproduce raw PPK2 traces, or claim new
//! physical measurements. Those require the pinned source files and the later raw-artifact
//! acquisition pipeline.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::error::Error;
use std::fmt;

use csv::{ReaderBuilder, StringRecord, Trim};
use serde::Serialize;
use sha2::{Digest, Sha256};

const AUDIT_SCHEMA: &str = "symthaea.nas.nats-bench-mcu-export-audit";
const AUDIT_VERSION: &str = "0.1.0";
const AUDIT_DOMAIN: &[u8] = b"symthaea.nas.nats-bench-mcu-export-audit.v0.1";
const DATASET_DOI: &str = "10.5281/zenodo.21644470";

/// Corrected v2 publication-export contract. The 746 KB ceiling is the dataset's
/// published deployability boundary; the 1,024 KB flash and 277.6 KB firmware figures
/// are retained as explanatory provenance rather than recomputed into a different cutoff.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NatsBenchMcuContract {
    pub expected_total_candidates: u32,
    pub expected_measured_candidates: usize,
    pub expected_infeasible_candidates: usize,
    pub application_flash_kb: f64,
    pub fixed_firmware_overhead_kb: f64,
    pub maximum_int8_model_kb: f64,
}

pub const NATS_BENCH_MCU_V2_CONTRACT: NatsBenchMcuContract = NatsBenchMcuContract {
    expected_total_candidates: 15_625,
    expected_measured_candidates: 14_281,
    expected_infeasible_candidates: 1_344,
    application_flash_kb: 1_024.0,
    fixed_firmware_overhead_kb: 277.6,
    maximum_int8_model_kb: 746.0,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NatsBenchMcuAuditError {
    pub field: String,
    pub message: String,
}

impl NatsBenchMcuAuditError {
    fn new(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            field: field.into(),
            message: message.into(),
        }
    }
}

impl fmt::Display for NatsBenchMcuAuditError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.field, self.message)
    }
}

impl Error for NatsBenchMcuAuditError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ParetoCostMetric {
    Latency,
    Energy,
    Rom,
    Ram,
}

impl ParetoCostMetric {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Latency => "latency_s",
            Self::Energy => "energy_mJ",
            Self::Rom => "rom_kb",
            Self::Ram => "ram_kb",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct NatsBenchMcuCandidate {
    pub arch_idx: u32,
    pub architecture: String,
    pub int8_model_kb: f64,
    pub total_rom_kb: f64,
    pub total_ram_kb: f64,
    pub inference_latency_s: f64,
    pub mean_current_ua: f64,
    pub mean_power_mw: f64,
    pub energy_per_inference_mj: f64,
    pub board_id: String,
    pub cifar10_accuracy_pct: f64,
    pub cifar100_accuracy_pct: f64,
    pub imagenet16_120_accuracy_pct: f64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ParetoFrontCount {
    pub dataset: String,
    pub cost_metric: ParetoCostMetric,
    pub candidate_count: usize,
    pub candidate_indices: Vec<u32>,
}

/// Deterministic source-and-result receipt. CSV hashes cover the exact UTF-8 bytes supplied
/// to the parser; they are not asserted to be the Zenodo archive hashes.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct NatsBenchMcuExportAuditReceipt {
    pub schema: String,
    pub version: String,
    pub dataset_doi: String,
    pub dataset_version: String,
    pub expected_total_candidates: u32,
    pub measured_candidates: usize,
    pub measured_arch_indices: Vec<u32>,
    pub infeasible_candidates: usize,
    pub infeasible_arch_indices: Vec<u32>,
    pub application_flash_kb: f64,
    pub fixed_firmware_overhead_kb: f64,
    pub maximum_int8_model_kb: f64,
    pub hardware_metrics_csv_sha256: String,
    pub accuracies_csv_sha256: String,
    pub failures_csv_sha256: String,
    pub hardware_metrics_extension_columns: Vec<String>,
    pub accuracies_extension_columns: Vec<String>,
    pub failures_extension_columns: Vec<String>,
    pub pareto_fronts: Vec<ParetoFrontCount>,
    pub receipt_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct NatsBenchMcuExportAuditResult {
    pub receipt: NatsBenchMcuExportAuditReceipt,
    pub candidates: Vec<NatsBenchMcuCandidate>,
}

#[derive(Debug, Clone)]
struct CsvTable {
    name: String,
    headers: Vec<String>,
    rows: Vec<StringRecord>,
    sha256: String,
}

impl CsvTable {
    fn column(&self, name: &str) -> Result<usize, NatsBenchMcuAuditError> {
        self.headers
            .iter()
            .position(|header| header == name)
            .ok_or_else(|| {
                NatsBenchMcuAuditError::new(
                    format!("{}.header", self.name),
                    format!("required column {name:?} is missing"),
                )
            })
    }

    fn one_of(&self, names: &[&str]) -> Result<usize, NatsBenchMcuAuditError> {
        let matches: Vec<usize> = names
            .iter()
            .filter_map(|name| self.headers.iter().position(|header| header == name))
            .collect();
        match matches.as_slice() {
            [index] => Ok(*index),
            [] => Err(NatsBenchMcuAuditError::new(
                format!("{}.header", self.name),
                format!("one of the required columns {names:?} must be present"),
            )),
            _ => Err(NatsBenchMcuAuditError::new(
                format!("{}.header", self.name),
                format!("ambiguous aliases present for {names:?}"),
            )),
        }
    }

    fn extension_columns(&self, required: &[&str]) -> Vec<String> {
        self.headers
            .iter()
            .filter(|header| !required.contains(&header.as_str()))
            .cloned()
            .collect()
    }
}

fn parse_csv(input: &str, name: &str) -> Result<CsvTable, NatsBenchMcuAuditError> {
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .flexible(false)
        .trim(Trim::All)
        .from_reader(input.as_bytes());

    let headers = reader
        .headers()
        .map_err(|error| {
            NatsBenchMcuAuditError::new(
                format!("{name}.header"),
                format!("cannot read CSV header: {error}"),
            )
        })?
        .clone();

    if headers.is_empty() {
        return Err(NatsBenchMcuAuditError::new(
            format!("{name}.header"),
            "CSV header is empty",
        ));
    }

    let mut seen = HashSet::with_capacity(headers.len());
    let mut header_names = Vec::with_capacity(headers.len());
    for header in headers.iter() {
        if header.is_empty() {
            return Err(NatsBenchMcuAuditError::new(
                format!("{name}.header"),
                "column names must be non-empty",
            ));
        }
        if !seen.insert(header.to_owned()) {
            return Err(NatsBenchMcuAuditError::new(
                format!("{name}.header"),
                format!("duplicate column name {header:?}"),
            ));
        }
        header_names.push(header.to_owned());
    }

    let mut rows = Vec::new();
    for (index, row) in reader.records().enumerate() {
        let record = row.map_err(|error| {
            NatsBenchMcuAuditError::new(
                format!("{name}.row[{}]", index + 2),
                format!("malformed CSV record: {error}"),
            )
        })?;
        rows.push(record);
    }

    Ok(CsvTable {
        name: name.to_owned(),
        headers: header_names,
        rows,
        sha256: sha256_hex(input.as_bytes()),
    })
}

fn field<'a>(
    table: &CsvTable,
    row: &'a StringRecord,
    column: usize,
    row_number: usize,
    column_name: &str,
) -> Result<&'a str, NatsBenchMcuAuditError> {
    row.get(column).ok_or_else(|| {
        NatsBenchMcuAuditError::new(
            format!("{}.row[{}].{column_name}", table.name, row_number),
            "column value is missing",
        )
    })
}

fn parse_arch_idx(
    table: &CsvTable,
    row: &StringRecord,
    column: usize,
    row_number: usize,
    total: u32,
) -> Result<u32, NatsBenchMcuAuditError> {
    let raw = field(table, row, column, row_number, "arch_idx")?;
    let arch_idx = raw.parse::<u32>().map_err(|_| {
        NatsBenchMcuAuditError::new(
            format!("{}.row[{}].arch_idx", table.name, row_number),
            format!("expected a zero-based integer architecture index, got {raw:?}"),
        )
    })?;
    if arch_idx >= total {
        return Err(NatsBenchMcuAuditError::new(
            format!("{}.row[{}].arch_idx", table.name, row_number),
            format!("index {arch_idx} is outside 0..{total}"),
        ));
    }
    Ok(arch_idx)
}

fn parse_positive_f64(
    table: &CsvTable,
    row: &StringRecord,
    column: usize,
    row_number: usize,
    name: &str,
) -> Result<f64, NatsBenchMcuAuditError> {
    let raw = field(table, row, column, row_number, name)?;
    let value = raw.parse::<f64>().map_err(|_| {
        NatsBenchMcuAuditError::new(
            format!("{}.row[{}].{name}", table.name, row_number),
            format!("expected a finite positive number, got {raw:?}"),
        )
    })?;
    if !value.is_finite() || value <= 0.0 {
        return Err(NatsBenchMcuAuditError::new(
            format!("{}.row[{}].{name}", table.name, row_number),
            format!("expected a finite positive number, got {raw:?}"),
        ));
    }
    Ok(value)
}

fn parse_accuracy(
    table: &CsvTable,
    row: &StringRecord,
    column: usize,
    row_number: usize,
    name: &str,
) -> Result<f64, NatsBenchMcuAuditError> {
    let raw = field(table, row, column, row_number, name)?;
    let value = raw.parse::<f64>().map_err(|_| {
        NatsBenchMcuAuditError::new(
            format!("{}.row[{}].{name}", table.name, row_number),
            format!("expected a finite accuracy percentage, got {raw:?}"),
        )
    })?;
    if !value.is_finite() || !(0.0..=100.0).contains(&value) {
        return Err(NatsBenchMcuAuditError::new(
            format!("{}.row[{}].{name}", table.name, row_number),
            format!("accuracy must be finite and in [0, 100], got {raw:?}"),
        ));
    }
    Ok(value)
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn candidate_accuracy(candidate: &NatsBenchMcuCandidate, dataset: &str) -> f64 {
    match dataset {
        "cifar10" => candidate.cifar10_accuracy_pct,
        "cifar100" => candidate.cifar100_accuracy_pct,
        "ImageNet16-120" => candidate.imagenet16_120_accuracy_pct,
        _ => unreachable!("dataset names are fixed by the audit contract"),
    }
}

fn candidate_cost(candidate: &NatsBenchMcuCandidate, metric: ParetoCostMetric) -> f64 {
    match metric {
        ParetoCostMetric::Latency => candidate.inference_latency_s,
        ParetoCostMetric::Energy => candidate.energy_per_inference_mj,
        ParetoCostMetric::Rom => candidate.total_rom_kb,
        ParetoCostMetric::Ram => candidate.total_ram_kb,
    }
}

/// Returns the exact two-objective nondominated set for accuracy (maximize) and one
/// deployment cost (minimize). Equal-cost/equal-accuracy candidates are retained as ties.
pub fn accuracy_cost_pareto_front(
    candidates: &[NatsBenchMcuCandidate],
    dataset: &str,
    cost_metric: ParetoCostMetric,
) -> Result<Vec<u32>, NatsBenchMcuAuditError> {
    if !matches!(dataset, "cifar10" | "cifar100" | "ImageNet16-120") {
        return Err(NatsBenchMcuAuditError::new(
            "pareto.dataset",
            format!("unsupported accuracy dataset {dataset:?}"),
        ));
    }

    let mut points: Vec<(f64, f64, u32)> = candidates
        .iter()
        .map(|candidate| {
            (
                candidate_cost(candidate, cost_metric),
                candidate_accuracy(candidate, dataset),
                candidate.arch_idx,
            )
        })
        .collect();

    points.sort_by(|left, right| {
        left.0
            .total_cmp(&right.0)
            .then_with(|| right.1.total_cmp(&left.1))
            .then_with(|| left.2.cmp(&right.2))
    });

    let mut front = Vec::new();
    let mut best_accuracy_at_lower_cost: Option<f64> = None;
    let mut start = 0;

    while start < points.len() {
        let cost = points[start].0;
        let mut end = start + 1;
        while end < points.len() && points[end].0.total_cmp(&cost).is_eq() {
            end += 1;
        }

        let group_max_accuracy = points[start..end]
            .iter()
            .map(|point| point.1)
            .max_by(|left, right| left.total_cmp(right))
            .expect("a Pareto cost group is non-empty");

        if best_accuracy_at_lower_cost
            .is_none_or(|best_accuracy| group_max_accuracy > best_accuracy)
        {
            front.extend(
                points[start..end]
                    .iter()
                    .filter(|point| point.1 == group_max_accuracy)
                    .map(|point| point.2),
            );
        }

        best_accuracy_at_lower_cost = Some(
            best_accuracy_at_lower_cost
                .map_or(group_max_accuracy, |prior| prior.max(group_max_accuracy)),
        );
        start = end;
    }

    front.sort_unstable();
    Ok(front)
}

fn receipt_sha256(
    receipt: &NatsBenchMcuExportAuditReceipt,
) -> Result<String, NatsBenchMcuAuditError> {
    let mut canonical = receipt.clone();
    canonical.receipt_sha256.clear();
    let bytes = serde_json::to_vec(&canonical).map_err(|error| {
        NatsBenchMcuAuditError::new("receipt", format!("cannot serialize canonical receipt: {error}"))
    })?;
    let mut hasher = Sha256::new();
    hasher.update(AUDIT_DOMAIN);
    hasher.update(bytes);
    Ok(format!("{:x}", hasher.finalize()))
}

/// Audits the corrected v2 publication CSVs against an explicit candidate-count contract.
pub fn audit_nats_bench_mcu_export(
    hardware_metrics_csv: &str,
    accuracies_csv: &str,
    failures_csv: &str,
    contract: NatsBenchMcuContract,
) -> Result<NatsBenchMcuExportAuditResult, NatsBenchMcuAuditError> {
    if contract.expected_total_candidates == 0
        || contract.expected_measured_candidates + contract.expected_infeasible_candidates
            != contract.expected_total_candidates as usize
        || !contract.application_flash_kb.is_finite()
        || contract.application_flash_kb <= 0.0
        || !contract.fixed_firmware_overhead_kb.is_finite()
        || contract.fixed_firmware_overhead_kb < 0.0
        || !contract.maximum_int8_model_kb.is_finite()
        || contract.maximum_int8_model_kb <= 0.0
        || contract.maximum_int8_model_kb
            > contract.application_flash_kb - contract.fixed_firmware_overhead_kb
    {
        return Err(NatsBenchMcuAuditError::new(
            "contract",
            "candidate counts must partition a non-empty search space and the INT8 limit must be finite and positive",
        ));
    }

    let hardware = parse_csv(hardware_metrics_csv, "hw_metrics.csv")?;
    let accuracies = parse_csv(accuracies_csv, "accuracies.csv")?;
    let failures = parse_csv(failures_csv, "failures.csv")?;

    let h_arch = hardware.column("arch_idx")?;
    let h_arch_str = hardware.column("arch_str")?;
    let h_int8 = hardware.column("int8_kb")?;
    let h_rom = hardware.column("rom_kb")?;
    let h_ram = hardware.column("ram_kb")?;
    let h_latency = hardware.column("latency_s")?;
    let h_current = hardware.column("current_uA")?;
    let h_power = hardware.column("power_mW")?;
    let h_energy = hardware.column("energy_mJ")?;
    let h_board = hardware.column("board")?;

    let f_arch = failures.column("arch_idx")?;
    let f_int8 = failures.column("int8_kb")?;
    let f_outcome = failures.one_of(&["stage", "outcome"])?;

    let a_arch = accuracies.column("arch_idx")?;
    let a_cifar10 = accuracies.column("cifar10")?;
    let a_cifar100 = accuracies.column("cifar100")?;
    let a_imagenet = accuracies.column("ImageNet16-120")?;

    let hardware_required = [
        "arch_idx", "arch_str", "int8_kb", "rom_kb", "ram_kb", "latency_s",
        "current_uA", "power_mW", "energy_mJ", "board",
    ];
    let accuracy_required = ["arch_idx", "cifar10", "cifar100", "ImageNet16-120"];
    let failure_required = ["arch_idx", "int8_kb", "stage", "outcome"];
    let hardware_extensions = hardware.extension_columns(&hardware_required);
    let accuracy_extensions = accuracies.extension_columns(&accuracy_required);
    let failure_extensions = failures.extension_columns(&failure_required);

    if hardware.rows.len() != contract.expected_measured_candidates {
        return Err(NatsBenchMcuAuditError::new(
            "hw_metrics.csv.rows",
            format!(
                "expected {} measured rows, found {}",
                contract.expected_measured_candidates,
                hardware.rows.len()
            ),
        ));
    }
    if failures.rows.len() != contract.expected_infeasible_candidates {
        return Err(NatsBenchMcuAuditError::new(
            "failures.csv.rows",
            format!(
                "expected {} infeasible rows, found {}",
                contract.expected_infeasible_candidates,
                failures.rows.len()
            ),
        ));
    }
    if accuracies.rows.len() != contract.expected_measured_candidates {
        return Err(NatsBenchMcuAuditError::new(
            "accuracies.csv.rows",
            format!(
                "expected {} accuracy rows, found {}",
                contract.expected_measured_candidates,
                accuracies.rows.len()
            ),
        ));
    }

    let mut measured = BTreeMap::<u32, NatsBenchMcuCandidate>::new();
    for (offset, row) in hardware.rows.iter().enumerate() {
        let row_number = offset + 2;
        let arch_idx = parse_arch_idx(
            &hardware, row, h_arch, row_number, contract.expected_total_candidates,
        )?;
        let architecture = field(&hardware, row, h_arch_str, row_number, "arch_str")?.to_owned();
        if architecture.trim().is_empty() {
            return Err(NatsBenchMcuAuditError::new(
                format!("hw_metrics.csv.row[{row_number}].arch_str"),
                "architecture topology must be non-empty",
            ));
        }

        let int8_model_kb = parse_positive_f64(&hardware, row, h_int8, row_number, "int8_kb")?;
        if int8_model_kb > contract.maximum_int8_model_kb {
            return Err(NatsBenchMcuAuditError::new(
                format!("hw_metrics.csv.row[{row_number}].int8_kb"),
                format!(
                    "measured candidate exceeds the published {:.3} KB INT8 deployability limit",
                    contract.maximum_int8_model_kb
                ),
            ));
        }

        let board_id = field(&hardware, row, h_board, row_number, "board")?.to_owned();
        if board_id.trim().is_empty() {
            return Err(NatsBenchMcuAuditError::new(
                format!("hw_metrics.csv.row[{row_number}].board"),
                "board identity must be non-empty",
            ));
        }

        let candidate = NatsBenchMcuCandidate {
            arch_idx,
            architecture,
            int8_model_kb,
            total_rom_kb: parse_positive_f64(&hardware, row, h_rom, row_number, "rom_kb")?,
            total_ram_kb: parse_positive_f64(&hardware, row, h_ram, row_number, "ram_kb")?,
            inference_latency_s: parse_positive_f64(&hardware, row, h_latency, row_number, "latency_s")?,
            mean_current_ua: parse_positive_f64(&hardware, row, h_current, row_number, "current_uA")?,
            mean_power_mw: parse_positive_f64(&hardware, row, h_power, row_number, "power_mW")?,
            energy_per_inference_mj: parse_positive_f64(&hardware, row, h_energy, row_number, "energy_mJ")?,
            board_id,
            // Filled after the independently keyed accuracy join.
            cifar10_accuracy_pct: 0.0,
            cifar100_accuracy_pct: 0.0,
            imagenet16_120_accuracy_pct: 0.0,
        };

        if measured.insert(arch_idx, candidate).is_some() {
            return Err(NatsBenchMcuAuditError::new(
                format!("hw_metrics.csv.row[{row_number}].arch_idx"),
                format!("duplicate measured architecture index {arch_idx}"),
            ));
        }
    }

    let mut infeasible = BTreeSet::<u32>::new();
    for (offset, row) in failures.rows.iter().enumerate() {
        let row_number = offset + 2;
        let arch_idx = parse_arch_idx(
            &failures, row, f_arch, row_number, contract.expected_total_candidates,
        )?;
        let outcome = field(&failures, row, f_outcome, row_number, "stage/outcome")?;
        if outcome != "infeasible" {
            return Err(NatsBenchMcuAuditError::new(
                format!("failures.csv.row[{row_number}].stage"),
                format!("v2 failure taxonomy requires exactly 'infeasible', got {outcome:?}"),
            ));
        }
        let int8_model_kb = parse_positive_f64(&failures, row, f_int8, row_number, "int8_kb")?;
        if int8_model_kb <= contract.maximum_int8_model_kb {
            return Err(NatsBenchMcuAuditError::new(
                format!("failures.csv.row[{row_number}].int8_kb"),
                format!(
                    "infeasible candidate does not exceed the published {:.3} KB INT8 limit",
                    contract.maximum_int8_model_kb
                ),
            ));
        }
        if !infeasible.insert(arch_idx) {
            return Err(NatsBenchMcuAuditError::new(
                format!("failures.csv.row[{row_number}].arch_idx"),
                format!("duplicate infeasible architecture index {arch_idx}"),
            ));
        }
        if measured.contains_key(&arch_idx) {
            return Err(NatsBenchMcuAuditError::new(
                format!("failures.csv.row[{row_number}].arch_idx"),
                format!("architecture index {arch_idx} appears as both measured and infeasible"),
            ));
        }
    }

    let mut accuracy_by_arch = BTreeMap::<u32, (f64, f64, f64)>::new();
    for (offset, row) in accuracies.rows.iter().enumerate() {
        let row_number = offset + 2;
        let arch_idx = parse_arch_idx(
            &accuracies, row, a_arch, row_number, contract.expected_total_candidates,
        )?;
        let scores = (
            parse_accuracy(&accuracies, row, a_cifar10, row_number, "cifar10")?,
            parse_accuracy(&accuracies, row, a_cifar100, row_number, "cifar100")?,
            parse_accuracy(&accuracies, row, a_imagenet, row_number, "ImageNet16-120")?,
        );
        if accuracy_by_arch.insert(arch_idx, scores).is_some() {
            return Err(NatsBenchMcuAuditError::new(
                format!("accuracies.csv.row[{row_number}].arch_idx"),
                format!("duplicate accuracy architecture index {arch_idx}"),
            ));
        }
    }

    if accuracy_by_arch.len() != measured.len()
        || accuracy_by_arch.keys().any(|arch_idx| !measured.contains_key(arch_idx))
        || measured.keys().any(|arch_idx| !accuracy_by_arch.contains_key(arch_idx))
    {
        return Err(NatsBenchMcuAuditError::new(
            "accuracies.csv.arch_idx",
            "accuracy indices must exactly equal measured hardware indices; do not impute missing rows",
        ));
    }

    let mut partition = BTreeSet::new();
    for arch_idx in measured.keys().copied().chain(infeasible.iter().copied()) {
        if !partition.insert(arch_idx) {
            return Err(NatsBenchMcuAuditError::new(
                "candidate_partition",
                format!("architecture index {arch_idx} occurs more than once"),
            ));
        }
    }

    if partition.len() != contract.expected_total_candidates as usize {
        return Err(NatsBenchMcuAuditError::new(
            "candidate_partition",
            format!(
                "expected {} unique architecture indices, found {}",
                contract.expected_total_candidates,
                partition.len()
            ),
        ));
    }
    for expected in 0..contract.expected_total_candidates {
        if !partition.contains(&expected) {
            return Err(NatsBenchMcuAuditError::new(
                "candidate_partition",
                format!("architecture index {expected} is missing from the measured/infeasible partition"),
            ));
        }
    }

    for (arch_idx, candidate) in measured.iter_mut() {
        let (cifar10, cifar100, imagenet16_120) =
            accuracy_by_arch.get(arch_idx).copied().ok_or_else(|| {
                NatsBenchMcuAuditError::new(
                    "accuracies.csv.arch_idx",
                    format!("missing accuracy join for architecture {arch_idx}"),
                )
            })?;
        candidate.cifar10_accuracy_pct = cifar10;
        candidate.cifar100_accuracy_pct = cifar100;
        candidate.imagenet16_120_accuracy_pct = imagenet16_120;
    }

    let candidates: Vec<NatsBenchMcuCandidate> = measured.into_values().collect();
    let mut pareto_fronts = Vec::with_capacity(12);
    for dataset in ["cifar10", "cifar100", "ImageNet16-120"] {
        for cost_metric in [
            ParetoCostMetric::Latency,
            ParetoCostMetric::Energy,
            ParetoCostMetric::Rom,
            ParetoCostMetric::Ram,
        ] {
            let front = accuracy_cost_pareto_front(&candidates, dataset, cost_metric)?;
            pareto_fronts.push(ParetoFrontCount {
                dataset: dataset.to_owned(),
                cost_metric,
                candidate_count: front.len(),
                candidate_indices: front,
            });
        }
    }

    let mut receipt = NatsBenchMcuExportAuditReceipt {
        schema: AUDIT_SCHEMA.to_owned(),
        version: AUDIT_VERSION.to_owned(),
        dataset_doi: DATASET_DOI.to_owned(),
        dataset_version: "2.0".to_owned(),
        expected_total_candidates: contract.expected_total_candidates,
        measured_candidates: candidates.len(),
        measured_arch_indices: candidates.iter().map(|candidate| candidate.arch_idx).collect(),
        infeasible_candidates: infeasible.len(),
        infeasible_arch_indices: infeasible.iter().copied().collect(),
        application_flash_kb: contract.application_flash_kb,
        fixed_firmware_overhead_kb: contract.fixed_firmware_overhead_kb,
        maximum_int8_model_kb: contract.maximum_int8_model_kb,
        hardware_metrics_csv_sha256: hardware.sha256,
        accuracies_csv_sha256: accuracies.sha256,
        failures_csv_sha256: failures.sha256,
        hardware_metrics_extension_columns: hardware_extensions,
        accuracies_extension_columns: accuracy_extensions,
        failures_extension_columns: failure_extensions,
        pareto_fronts,
        receipt_sha256: String::new(),
    };
    receipt.receipt_sha256 = receipt_sha256(&receipt)?;

    Ok(NatsBenchMcuExportAuditResult {
        receipt,
        candidates,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> (String, String, String, NatsBenchMcuContract) {
        let hardware = concat!(
            "arch_idx,arch_str,int8_kb,rom_kb,ram_kb,latency_s,current_uA,power_mW,energy_mJ,board,future_vendor_metric\n",
            "0,|nor~0|,100,380,120,0.010,1000,3.0,0.030,nrf5340dk_1,x\n",
            "1,|nor~1|,200,480,120,0.020,1000,3.0,0.060,nrf5340dk_2,y\n",
            "2,|avg~0|,300,580,120,0.015,1000,3.0,0.045,nrf5340dk_3,z\n",
        ).to_owned();
        let accuracy = concat!(
            "arch_idx,cifar10,cifar100,ImageNet16-120\n",
            "0,90.0,60.0,30.0\n",
            "1,92.0,62.0,32.0\n",
            "2,91.0,61.0,31.0\n",
        ).to_owned();
        let failures = concat!(
            "arch_idx,stage,int8_kb,overflow_kb,board\n",
            "3,infeasible,747,1,nrf5340dk_1\n",
        ).to_owned();
        let contract = NatsBenchMcuContract {
            expected_total_candidates: 4,
            expected_measured_candidates: 3,
            expected_infeasible_candidates: 1,
            application_flash_kb: 1_024.0,
            fixed_firmware_overhead_kb: 277.6,
            maximum_int8_model_kb: 746.0,
        };
        (hardware, accuracy, failures, contract)
    }

    #[test]
    fn audits_exhaustive_partition_and_emits_content_bound_receipt() {
        let (hardware, accuracy, failures, contract) = fixture();
        let audit = audit_nats_bench_mcu_export(&hardware, &accuracy, &failures, contract)
            .expect("complete v2-shaped fixture should pass");

        assert_eq!(audit.receipt.expected_total_candidates, 4);
        assert_eq!(audit.receipt.measured_candidates, 3);
        assert_eq!(audit.receipt.infeasible_candidates, 1);
        assert_eq!(audit.candidates.len(), 3);
        assert_eq!(audit.receipt.hardware_metrics_csv_sha256.len(), 64);
        assert_eq!(audit.receipt.receipt_sha256.len(), 64);
        assert_eq!(
            audit.receipt.hardware_metrics_extension_columns,
            vec!["future_vendor_metric".to_owned()]
        );
        assert_eq!(audit.receipt.pareto_fronts.len(), 12);
    }

    #[test]
    fn rejects_measured_candidate_above_published_flash_boundary() {
        let (mut hardware, accuracy, failures, contract) = fixture();
        hardware = hardware.replacen(",100,380,", ",747,380,", 1);
        let error = audit_nats_bench_mcu_export(&hardware, &accuracy, &failures, contract)
            .expect_err("measured rows above the published limit must fail");
        assert!(error.to_string().contains("exceeds the published"));
    }

    #[test]
    fn rejects_infeasible_candidate_at_or_below_published_flash_boundary() {
        let (hardware, accuracy, mut failures, contract) = fixture();
        failures = failures.replace("3,infeasible,747", "3,infeasible,746");
        let error = audit_nats_bench_mcu_export(&hardware, &accuracy, &failures, contract)
            .expect_err("infeasible candidates must exceed the exact published limit");
        assert!(error.to_string().contains("does not exceed the published"));
    }

    #[test]
    fn rejects_duplicate_architecture_indices() {
        let (mut hardware, accuracy, failures, contract) = fixture();
        hardware.push_str("0,|duplicate|,100,380,120,0.010,1000,3.0,0.030,nrf5340dk_1,q\n");
        let error = audit_nats_bench_mcu_export(&hardware, &accuracy, &failures, contract)
            .expect_err("duplicate architecture indices must fail");
        assert!(error.to_string().contains("duplicate measured architecture index"));
    }

    #[test]
    fn rejects_accuracy_rows_that_do_not_join_exactly_to_measured_architectures() {
        let (hardware, mut accuracy, failures, contract) = fixture();
        accuracy = accuracy.replace("2,91.0,61.0,31.0\n", "");
        let error = audit_nats_bench_mcu_export(&hardware, &accuracy, &failures, contract)
            .expect_err("missing accuracy data must not be imputed");
        assert!(error.to_string().contains("expected 3 accuracy rows"));
    }

    #[test]
    fn rejects_non_finite_objective_values() {
        let (mut hardware, accuracy, failures, contract) = fixture();
        hardware = hardware.replace(",0.010,1000,", ",NaN,1000,");
        let error = audit_nats_bench_mcu_export(&hardware, &accuracy, &failures, contract)
            .expect_err("NaN must never enter a benchmark objective");
        assert!(error.to_string().contains("finite positive"));
    }

    #[test]
    fn rejects_failure_taxonomy_substitution() {
        let (hardware, accuracy, mut failures, contract) = fixture();
        failures = failures.replace("infeasible", "failed");
        let error = audit_nats_bench_mcu_export(&hardware, &accuracy, &failures, contract)
            .expect_err("v2 uses one explicit infeasible label");
        assert!(error.to_string().contains("requires exactly 'infeasible'"));
    }

    #[test]
    fn pareto_front_keeps_ties_and_rejects_unknown_dataset() {
        let (hardware, accuracy, failures, contract) = fixture();
        let audit = audit_nats_bench_mcu_export(&hardware, &accuracy, &failures, contract)
            .expect("fixture should audit");
        let front = accuracy_cost_pareto_front(
            &audit.candidates,
            "cifar10",
            ParetoCostMetric::Latency,
        )
        .expect("supported objective pair should produce a front");
        assert!(front.contains(&0));
        assert!(front.contains(&2));
        assert!(accuracy_cost_pareto_front(
            &audit.candidates,
            "unknown",
            ParetoCostMetric::Latency,
        ).is_err());
    }
}
