//! Deterministic, fail-closed receipt runner for the frozen culinary validator corpus.
//!
//! This is an input/model validation benchmark, not a food-safety certification.
//! Non-finite JSON numeric values are represented as strings ("NaN", "Infinity",
//! "-Infinity", "MAX", "-MAX") so the fixture file remains valid JSON.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeSet, HashMap};
use std::env;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};
use std::time::Instant;
use symthaea_culinary::nutrition::NutrientProfile;
use symthaea_culinary::spec::{
    Candy, Coagulation, CulinarySpec, DairyAcidification, DoughClass, Emulsion, Fat, FryingFat,
    Hydration, Pasteurization, Pathogen, Protein, SugarStage,
};
use symthaea_culinary::thermal::ThermalTrajectory;
use symthaea_culinary::validate::{CulinaryViolation, validate};

const SCHEMA_VERSION: &str = "1.0.0";
const BENCHMARK_ID: &str = "CULINARY-VALIDATOR-CORPUS";
const BENCHMARK_VERSION: &str = "1.0.0";
const EXPECTED_MANIFEST_SHA256: &str = "236a67bb3e91a7422fbbc27841bfda013c54ac29e85a0ef71cd303ade978483a";

const REQUIRED_CASE_IDS: &[&str] = &[
    "EMULSION-VALID-INTERIOR-001",
    "EMULSION-RCP-BOUNDARY-001",
    "EMULSION-OVER-RCP-001",
    "EMULSION-NAN-001",
    "HYDRATION-BREAD-VALID-001",
    "HYDRATION-NEGATIVE-PAIRED-MASS-001",
    "HYDRATION-ZERO-FLOUR-001",
    "HYDRATION-OUTSIDE-WINDOW-001",
    "DAIRY-PH-SMOOTH-VALID-001",
    "DAIRY-PH-OUT-OF-DOMAIN-001",
    "NUTRITION-REFERENCE-VALID-001",
    "NUTRITION-NAN-SODIUM-001",
    "NUTRITION-ENERGY-OVERFLOW-001",
    "COAGULATION-EGG-WHITE-SETPOINT-001",
    "COAGULATION-EMPTY-TRAJECTORY-001",
    "COAGULATION-ASTRONOMICAL-TEMP-001",
    "COAGULATION-BELOW-ABSOLUTE-ZERO-001",
    "CANDY-SOFT-BALL-VALID-001",
    "CANDY-REVERSED-TIME-001",
    "FRYING-CANOLA-BOUNDARY-001",
    "FRYING-ABOVE-SMOKE-POINT-001",
    "PASTEURIZATION-NAN-TARGET-001",
    "PASTEURIZATION-LETHALITY-OVERFLOW-001",
    "NUTRITION-SODIUM-THRESHOLD-001",
    "DAIRY-CURDLING-INTENT-MISMATCH-001",
    "CANDY-UNDERCOOKED-001",
];

#[derive(Debug, Deserialize)]
struct Manifest {
    schema_version: String,
    benchmark_id: String,
    benchmark_version: String,
    expected_case_count: usize,
    cases: Vec<Fixture>,
}

#[derive(Debug, Deserialize)]
struct Fixture {
    id: String,
    family: String,
    expected_status: String,
    expected_violation: Option<String>,
    input: Value,
    rationale: String,
    reference: String,
}

#[derive(Debug, Serialize)]
struct CaseResult {
    id: String,
    family: String,
    expected_status: String,
    expected_violation: Option<String>,
    actual_status: String,
    actual_violation_classes: Vec<String>,
    matched_expectation: bool,
    detail: Option<String>,
    rationale: String,
    reference: String,
}

#[derive(Debug, Serialize)]
struct Platform {
    os: &'static str,
    arch: &'static str,
    package_version: &'static str,
}

#[derive(Debug, Serialize)]
struct Receipt {
    schema_version: String,
    benchmark_id: String,
    benchmark_version: String,
    repository: String,
    commit_sha: Option<String>,
    expected_head: Option<String>,
    head_matches_expected: Option<bool>,
    working_tree_clean: Option<bool>,
    manifest_sha256: Option<String>,
    manifest_digest_matches_pinned: Option<bool>,
    expected_case_count: usize,
    manifest_declared_case_count: Option<usize>,
    fixture_count: usize,
    attempted: usize,
    passed: usize,
    failed: usize,
    skipped: usize,
    missing_case_ids: Vec<String>,
    unexpected_case_ids: Vec<String>,
    duplicate_case_ids: Vec<String>,
    corpus_issues: Vec<String>,
    cases: Vec<CaseResult>,
    platform: Platform,
    elapsed_ms: f64,
    status: String,
}

struct Args {
    manifest_path: PathBuf,
    output_path: Option<PathBuf>,
    expected_head: Option<String>,
}

struct CommitContext {
    sha: Option<String>,
    clean: Option<bool>,
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(args) => args,
        Err(message) => {
            eprintln!("{message}");
            eprintln!(
                "usage: culinary_validation_receipt [--manifest PATH] [--output PATH] [--expected-head SHA]"
            );
            return ExitCode::from(2);
        }
    };

    let started = Instant::now();
    let (receipt, exit_code) = build_receipt(&args, started);
    match write_receipt(&receipt, args.output_path.as_deref()) {
        Ok(()) => {
            if exit_code == 0 {
                ExitCode::SUCCESS
            } else {
                ExitCode::from(2)
            }
        }
        Err(error) => {
            eprintln!("failed to write culinary validation receipt: {error}");
            ExitCode::FAILURE
        }
    }
}

fn parse_args() -> Result<Args, String> {
    let mut manifest_path = None;
    let mut output_path = None;
    let mut expected_head = None;
    let mut args = env::args().skip(1);

    while let Some(arg) = args.next() {
        let value = args
            .next()
            .ok_or_else(|| format!("missing value for {arg}"))?;
        match arg.as_str() {
            "--manifest" => manifest_path = Some(PathBuf::from(value)),
            "--output" => output_path = Some(PathBuf::from(value)),
            "--expected-head" => expected_head = Some(value),
            _ => return Err(format!("unknown argument: {arg}")),
        }
    }

    let default_manifest = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("benchmarks")
        .join("validator-corpus-v1.json");

    Ok(Args {
        manifest_path: manifest_path.unwrap_or(default_manifest),
        output_path,
        expected_head,
    })
}

fn build_receipt(args: &Args, started: Instant) -> (Receipt, u8) {
    let commit = identify_checkout();
    let head_matches = args
        .expected_head
        .as_ref()
        .map(|expected| commit.sha.as_ref() == Some(expected));

    let manifest_bytes = match fs::read(&args.manifest_path) {
        Ok(bytes) => bytes,
        Err(error) => {
            let mut issues = vec![format!(
                "manifest unreadable at {}: {error}",
                args.manifest_path.display()
            )];
            add_checkout_issues(&mut issues, &commit, &head_matches);
            let status = if checkout_not_comparable(&commit, &head_matches) {
                "not_comparable"
            } else {
                "incomplete"
            };
            return (
                Receipt {
                    schema_version: SCHEMA_VERSION.to_owned(),
                    benchmark_id: BENCHMARK_ID.to_owned(),
                    benchmark_version: BENCHMARK_VERSION.to_owned(),
                    repository: "Luminous-Dynamics/symthaea".to_owned(),
                    commit_sha: commit.sha,
                    expected_head: args.expected_head.clone(),
                    head_matches_expected: head_matches,
                    working_tree_clean: commit.clean,
                    manifest_sha256: None,
                    manifest_digest_matches_pinned: None,
                    expected_case_count: REQUIRED_CASE_IDS.len(),
                    manifest_declared_case_count: None,
                    fixture_count: 0,
                    attempted: 0,
                    passed: 0,
                    failed: 0,
                    skipped: REQUIRED_CASE_IDS.len(),
                    missing_case_ids: REQUIRED_CASE_IDS
                        .iter()
                        .map(|id| (*id).to_owned())
                        .collect(),
                    unexpected_case_ids: vec![],
                    duplicate_case_ids: vec![],
                    corpus_issues: std::mem::take(&mut issues),
                    cases: vec![],
                    platform: platform(),
                    elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
                    status: status.to_owned(),
                },
                2,
            );
        }
    };

    let manifest_digest = format!("{:x}", Sha256::digest(&manifest_bytes));
    let digest_matches = manifest_digest == EXPECTED_MANIFEST_SHA256;
    let manifest: Manifest = match serde_json::from_slice(&manifest_bytes) {
        Ok(manifest) => manifest,
        Err(error) => {
            let mut issues = vec![format!("manifest is not valid benchmark JSON: {error}")];
            if !digest_matches {
                issues.push(format!(
                    "manifest digest is not pinned: observed {manifest_digest}"
                ));
            }
            add_checkout_issues(&mut issues, &commit, &head_matches);
            let status = if checkout_not_comparable(&commit, &head_matches) || !digest_matches {
                "not_comparable"
            } else {
                "incomplete"
            };
            return (
                Receipt {
                    schema_version: SCHEMA_VERSION.to_owned(),
                    benchmark_id: BENCHMARK_ID.to_owned(),
                    benchmark_version: BENCHMARK_VERSION.to_owned(),
                    repository: "Luminous-Dynamics/symthaea".to_owned(),
                    commit_sha: commit.sha,
                    expected_head: args.expected_head.clone(),
                    head_matches_expected: head_matches,
                    working_tree_clean: commit.clean,
                    manifest_sha256: Some(manifest_digest),
                    manifest_digest_matches_pinned: Some(digest_matches),
                    expected_case_count: REQUIRED_CASE_IDS.len(),
                    manifest_declared_case_count: None,
                    fixture_count: 0,
                    attempted: 0,
                    passed: 0,
                    failed: 0,
                    skipped: REQUIRED_CASE_IDS.len(),
                    missing_case_ids: REQUIRED_CASE_IDS
                        .iter()
                        .map(|id| (*id).to_owned())
                        .collect(),
                    unexpected_case_ids: vec![],
                    duplicate_case_ids: vec![],
                    corpus_issues: issues,
                    cases: vec![],
                    platform: platform(),
                    elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
                    status: status.to_owned(),
                },
                2,
            );
        }
    };

    let mut issues = Vec::new();
    if manifest.expected_case_count != REQUIRED_CASE_IDS.len() {
        issues.push(format!(
            "manifest declares {} cases; code requires {}",
            manifest.expected_case_count,
            REQUIRED_CASE_IDS.len()
        ));
    }
    if manifest.cases.len() != manifest.expected_case_count {
        issues.push(format!(
            "manifest has {} fixture rows but declares {}",
            manifest.cases.len(),
            manifest.expected_case_count
        ));
    }
    if !digest_matches {
        issues.push(format!(
            "manifest SHA-256 differs from the pinned corpus: expected {EXPECTED_MANIFEST_SHA256}, observed {manifest_digest}"
        ));
    }
    add_checkout_issues(&mut issues, &commit, &head_matches);

    let required: BTreeSet<&str> = REQUIRED_CASE_IDS.iter().copied().collect();
    let mut id_counts: HashMap<&str, usize> = HashMap::new();
    for fixture in &manifest.cases {
        *id_counts.entry(fixture.id.as_str()).or_default() += 1;
    }
    let observed: BTreeSet<&str> = id_counts.keys().copied().collect();
    let missing: Vec<String> = required
        .difference(&observed)
        .map(|id| (*id).to_owned())
        .collect();
    let unexpected: Vec<String> = observed
        .difference(&required)
        .map(|id| (*id).to_owned())
        .collect();
    let mut duplicates: Vec<String> = id_counts
        .iter()
        .filter(|(_, count)| **count > 1)
        .map(|(id, _)| (*id).to_owned())
        .collect();
    duplicates.sort();

    if !missing.is_empty() {
        issues.push(format!("required fixture IDs are missing: {}", missing.join(", ")));
    }
    if !unexpected.is_empty() {
        issues.push(format!("unrecognised fixture IDs are present: {}", unexpected.join(", ")));
    }
    if !duplicates.is_empty() {
        issues.push(format!("fixture IDs are duplicated: {}", duplicates.join(", ")));
    }

    let identity_mismatch = manifest.schema_version != SCHEMA_VERSION
        || manifest.benchmark_id != BENCHMARK_ID
        || manifest.benchmark_version != BENCHMARK_VERSION
        || checkout_not_comparable(&commit, &head_matches)
        || !digest_matches;

    let results: Vec<CaseResult> = manifest.cases.iter().map(run_case).collect();
    let attempted = results.len();
    let passed = results
        .iter()
        .filter(|result| result.matched_expectation)
        .count();
    let failed = attempted.saturating_sub(passed);
    let incomplete = !issues.is_empty() || attempted != REQUIRED_CASE_IDS.len();

    let status = if identity_mismatch {
        "not_comparable"
    } else if incomplete {
        "incomplete"
    } else if failed > 0 {
        "fail"
    } else {
        "pass"
    };

    let receipt = Receipt {
        schema_version: manifest.schema_version,
        benchmark_id: manifest.benchmark_id,
        benchmark_version: manifest.benchmark_version,
        repository: "Luminous-Dynamics/symthaea".to_owned(),
        commit_sha: commit.sha,
        expected_head: args.expected_head.clone(),
        head_matches_expected: head_matches,
        working_tree_clean: commit.clean,
        manifest_sha256: Some(manifest_digest),
        manifest_digest_matches_pinned: Some(digest_matches),
        expected_case_count: REQUIRED_CASE_IDS.len(),
        manifest_declared_case_count: Some(manifest.expected_case_count),
        fixture_count: attempted,
        attempted,
        passed,
        failed,
        skipped: missing.len(),
        missing_case_ids: missing,
        unexpected_case_ids: unexpected,
        duplicate_case_ids: duplicates,
        corpus_issues: issues,
        cases: results,
        platform: platform(),
        elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
        status: status.to_owned(),
    };

    let code = if status == "pass" { 0 } else { 2 };
    (receipt, code)
}

fn identify_checkout() -> CommitContext {
    let current = Path::new(env!("CARGO_MANIFEST_DIR"));
    let sha = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(current)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|sha| sha.trim().to_owned())
        .filter(|sha| {
            (sha.len() == 40 || sha.len() == 64) && sha.chars().all(|c| c.is_ascii_hexdigit())
        });

    let clean = Command::new("git")
        .args(["status", "--porcelain", "--untracked-files=normal"])
        .current_dir(current)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|output| output.trim().is_empty());

    CommitContext { sha, clean }
}

fn checkout_not_comparable(commit: &CommitContext, head_matches: &Option<bool>) -> bool {
    commit.sha.is_none() || commit.clean != Some(true) || *head_matches != Some(true)
}

fn add_checkout_issues(
    issues: &mut Vec<String>,
    commit: &CommitContext,
    head_matches: &Option<bool>,
) {
    if commit.sha.is_none() {
        issues.push("exact git commit SHA could not be identified".to_owned());
    }
    if commit.clean != Some(true) {
        issues.push("working tree is dirty or its state could not be determined".to_owned());
    }
    match head_matches {
        None => issues.push(
            "--expected-head is required; an unpinned checkout cannot produce a comparable pass"
                .to_owned(),
        ),
        Some(false) => {
            issues.push("current commit SHA does not match --expected-head".to_owned());
        }
        Some(true) => {}
    }
}

fn run_case(fixture: &Fixture) -> CaseResult {
    let result = make_spec(fixture).map(|spec| {
        validate(&spec)
            .iter()
            .map(violation_class)
            .map(str::to_owned)
            .collect::<Vec<_>>()
    });

    let (actual_classes, detail) = match result {
        Ok(classes) => (classes, None),
        Err(error) => (vec![], Some(error)),
    };
    let actual_status = if detail.is_some() {
        "error"
    } else if actual_classes.is_empty() {
        "pass"
    } else {
        "reject"
    };

    let matched_expectation = if detail.is_some() {
        false
    } else {
        match fixture.expected_status.as_str() {
            "pass" => fixture.expected_violation.is_none() && actual_classes.is_empty(),
            "reject" => {
                fixture.expected_violation.as_ref().is_some_and(|expected| {
                    actual_classes.len() == 1 && actual_classes.first() == Some(expected)
                })
            }
            _ => false,
        }
    };

    CaseResult {
        id: fixture.id.clone(),
        family: fixture.family.clone(),
        expected_status: fixture.expected_status.clone(),
        expected_violation: fixture.expected_violation.clone(),
        actual_status: actual_status.to_owned(),
        actual_violation_classes: actual_classes,
        matched_expectation,
        detail,
        rationale: fixture.rationale.clone(),
        reference: fixture.reference.clone(),
    }
}

fn make_spec(fixture: &Fixture) -> Result<CulinarySpec, String> {
    let input = &fixture.input;
    let spec = match fixture.family.as_str() {
        "emulsion" => CulinarySpec::new(fixture.id.clone()).with_emulsion(Emulsion {
            dispersed_phase_fraction: number(input, "dispersed_phase_fraction")?,
        }),
        "hydration" => CulinarySpec::new(fixture.id.clone()).with_hydration(Hydration {
            dough: parse_dough(string(input, "dough")?)?,
            flour_g: number(input, "flour_g")?,
            water_g: number(input, "water_g")?,
        }),
        "dairy" => CulinarySpec::new(fixture.id.clone()).with_dairy(DairyAcidification {
            ph: number(input, "ph")?,
            should_curdle: boolean(input, "should_curdle")?,
        }),
        "nutrition" => CulinarySpec::new(fixture.id.clone()).with_nutrition(NutrientProfile {
            protein_g: number(input, "protein_g")?,
            carb_g: number(input, "carb_g")?,
            fat_g: number(input, "fat_g")?,
            alcohol_g: number(input, "alcohol_g")?,
            sodium_mg: number(input, "sodium_mg")?,
            added_sugar_g: number(input, "added_sugar_g")?,
        }),
        "coagulation" => CulinarySpec::new(fixture.id.clone()).with_coagulation(Coagulation {
            protein: parse_protein(string(input, "protein")?)?,
            trajectory: trajectory(input)?,
        }),
        "candy" => CulinarySpec::new(fixture.id.clone()).with_candy(Candy {
            stage: parse_sugar_stage(string(input, "stage")?)?,
            trajectory: trajectory(input)?,
        }),
        "frying" => CulinarySpec::new(fixture.id.clone()).with_frying(FryingFat {
            fat: parse_fat(string(input, "fat")?)?,
            trajectory: trajectory(input)?,
        }),
        "pasteurization" => CulinarySpec::new(fixture.id.clone()).with_pasteurization(Pasteurization {
            pathogen: parse_pathogen(string(input, "pathogen")?)?,
            trajectory: trajectory(input)?,
            required_log_reduction: optional_number(input, "required_log_reduction")?,
        }),
        family => return Err(format!("unsupported fixture family: {family}")),
    };
    Ok(spec)
}

fn number(input: &Value, key: &str) -> Result<f64, String> {
    let value = input
        .get(key)
        .ok_or_else(|| format!("missing numeric field: {key}"))?;
    read_float(value).map_err(|error| format!("{key}: {error}"))
}

fn optional_number(input: &Value, key: &str) -> Result<Option<f64>, String> {
    let value = input
        .get(key)
        .ok_or_else(|| format!("missing nullable numeric field: {key}"))?;
    if value.is_null() {
        Ok(None)
    } else {
        read_float(value)
            .map(Some)
            .map_err(|error| format!("{key}: {error}"))
    }
}

fn read_float(value: &Value) -> Result<f64, String> {
    if let Some(number) = value.as_f64() {
        return Ok(number);
    }
    if let Some(token) = value.as_str() {
        return match token {
            "NaN" => Ok(f64::NAN),
            "Infinity" | "+Infinity" | "Inf" => Ok(f64::INFINITY),
            "-Infinity" | "-Inf" => Ok(f64::NEG_INFINITY),
            "MAX" => Ok(f64::MAX),
            "-MAX" => Ok(-f64::MAX),
            _ => token
                .parse::<f64>()
                .map_err(|error| format!("invalid float token {token:?}: {error}")),
        };
    }
    Err("expected a JSON number or supported special-float string".to_owned())
}

fn string<'a>(input: &'a Value, key: &str) -> Result<&'a str, String> {
    input
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("missing string field: {key}"))
}

fn boolean(input: &Value, key: &str) -> Result<bool, String> {
    input
        .get(key)
        .and_then(Value::as_bool)
        .ok_or_else(|| format!("missing boolean field: {key}"))
}

fn trajectory(input: &Value) -> Result<ThermalTrajectory, String> {
    let points = input
        .get("trajectory_points")
        .and_then(Value::as_array)
        .ok_or_else(|| "missing trajectory_points array".to_owned())?;
    let mut parsed = Vec::with_capacity(points.len());
    for (index, point) in points.iter().enumerate() {
        let pair = point
            .as_array()
            .filter(|pair| pair.len() == 2)
            .ok_or_else(|| format!("trajectory point {index} must contain [time, temperature]"))?;
        let time = read_float(&pair[0])
            .map_err(|error| format!("trajectory point {index} time: {error}"))?;
        let temperature = read_float(&pair[1])
            .map_err(|error| format!("trajectory point {index} temperature: {error}"))?;
        parsed.push((time, temperature));
    }
    Ok(ThermalTrajectory::new(parsed))
}

fn parse_dough(value: &str) -> Result<DoughClass, String> {
    match value {
        "Bread" => Ok(DoughClass::Bread),
        "Pastry" => Ok(DoughClass::Pastry),
        "Batter" => Ok(DoughClass::Batter),
        _ => Err(format!("unknown dough class: {value}")),
    }
}

fn parse_protein(value: &str) -> Result<Protein, String> {
    match value {
        "EggWhite" => Ok(Protein::EggWhite),
        "EggYolk" => Ok(Protein::EggYolk),
        "Custard" => Ok(Protein::Custard),
        _ => Err(format!("unknown protein: {value}")),
    }
}

fn parse_sugar_stage(value: &str) -> Result<SugarStage, String> {
    match value {
        "Thread" => Ok(SugarStage::Thread),
        "SoftBall" => Ok(SugarStage::SoftBall),
        "FirmBall" => Ok(SugarStage::FirmBall),
        "HardBall" => Ok(SugarStage::HardBall),
        "SoftCrack" => Ok(SugarStage::SoftCrack),
        "HardCrack" => Ok(SugarStage::HardCrack),
        _ => Err(format!("unknown sugar stage: {value}")),
    }
}

fn parse_fat(value: &str) -> Result<Fat, String> {
    match value {
        "ExtraVirginOliveOil" => Ok(Fat::ExtraVirginOliveOil),
        "Butter" => Ok(Fat::Butter),
        "CanolaOil" => Ok(Fat::CanolaOil),
        "RefinedPeanutOil" => Ok(Fat::RefinedPeanutOil),
        _ => Err(format!("unknown cooking fat: {value}")),
    }
}

fn parse_pathogen(value: &str) -> Result<Pathogen, String> {
    match value {
        "SalmonellaPoultry" => Ok(Pathogen::SalmonellaPoultry),
        _ => Err(format!("unknown pathogen: {value}")),
    }
}

fn violation_class(violation: &CulinaryViolation) -> &'static str {
    match violation {
        CulinaryViolation::EmulsionBreaks { .. } => "EmulsionBreaks",
        CulinaryViolation::Undercooked { .. } => "Undercooked",
        CulinaryViolation::Overcoagulated { .. } => "Overcoagulated",
        CulinaryViolation::InsufficientPasteurization { .. } => "InsufficientPasteurization",
        CulinaryViolation::HydrationOutOfRange { .. } => "HydrationOutOfRange",
        CulinaryViolation::CandyUndercooked { .. } => "CandyUndercooked",
        CulinaryViolation::CandyOvercooked { .. } => "CandyOvercooked",
        CulinaryViolation::DairyWontCurdle { .. } => "DairyWontCurdle",
        CulinaryViolation::DairyWillCurdle { .. } => "DairyWillCurdle",
        CulinaryViolation::FatSmoking { .. } => "FatSmoking",
        CulinaryViolation::SodiumExceedsLimit { .. } => "SodiumExceedsLimit",
        CulinaryViolation::AddedSugarExceedsLimit { .. } => "AddedSugarExceedsLimit",
        CulinaryViolation::InvalidNumericInput { .. } => "InvalidNumericInput",
    }
}

fn platform() -> Platform {
    Platform {
        os: env::consts::OS,
        arch: env::consts::ARCH,
        package_version: env!("CARGO_PKG_VERSION"),
    }
}

fn write_receipt(receipt: &Receipt, output_path: Option<&Path>) -> io::Result<()> {
    let serialized = serde_json::to_string_pretty(receipt)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    if let Some(path) = output_path {
        fs::write(path, format!("{serialized}\n"))
    } else {
        println!("{serialized}");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FROZEN_MANIFEST: &[u8] = include_bytes!("../benchmarks/validator-corpus-v1.json");

    fn frozen_manifest() -> Manifest {
        serde_json::from_slice(FROZEN_MANIFEST).expect("frozen corpus must parse as JSON")
    }

    #[test]
    fn frozen_manifest_matches_pinned_sha_and_required_case_ids() {
        let digest = format!("{:x}", Sha256::digest(FROZEN_MANIFEST));
        assert_eq!(
            digest, EXPECTED_MANIFEST_SHA256,
            "fixture bytes changed without updating the reviewed digest pin"
        );

        let manifest = frozen_manifest();
        assert_eq!(manifest.schema_version, SCHEMA_VERSION);
        assert_eq!(manifest.benchmark_id, BENCHMARK_ID);
        assert_eq!(manifest.benchmark_version, BENCHMARK_VERSION);
        assert_eq!(manifest.expected_case_count, REQUIRED_CASE_IDS.len());
        assert_eq!(manifest.cases.len(), REQUIRED_CASE_IDS.len());

        let required: BTreeSet<&str> = REQUIRED_CASE_IDS.iter().copied().collect();
        let observed: BTreeSet<&str> = manifest.cases.iter().map(|fixture| fixture.id.as_str()).collect();
        assert_eq!(observed, required, "manifest IDs must exactly match the runner contract");
        assert_eq!(
            manifest.cases.len(),
            observed.len(),
            "duplicate fixture IDs are prohibited"
        );
    }

    #[test]
    fn every_frozen_fixture_matches_its_expected_validator_outcome() {
        let manifest = frozen_manifest();
        let failures: Vec<String> = manifest
            .cases
            .iter()
            .map(run_case)
            .filter(|result| !result.matched_expectation)
            .map(|result| {
                format!(
                    "{}: expected {} / {:?}, got {} / {:?} ({:?})",
                    result.id,
                    result.expected_status,
                    result.expected_violation,
                    result.actual_status,
                    result.actual_violation_classes,
                    result.detail
                )
            })
            .collect();

        assert!(
            failures.is_empty(),
            "frozen culinary validator cases failed:\n{}",
            failures.join("\n")
        );
    }

    #[test]
    fn a_missing_or_mismatched_expected_head_is_not_comparable() {
        let clean_checkout = CommitContext {
            sha: Some("a".repeat(40)),
            clean: Some(true),
        };
        assert!(checkout_not_comparable(&clean_checkout, &None));
        assert!(checkout_not_comparable(&clean_checkout, &Some(false)));
        assert!(!checkout_not_comparable(&clean_checkout, &Some(true)));
    }

    #[test]
    fn special_float_tokens_preserve_adversarial_values() {
        assert!(read_float(&Value::String("NaN".to_owned())).unwrap().is_nan());
        assert_eq!(
            read_float(&Value::String("Infinity".to_owned())).unwrap(),
            f64::INFINITY
        );
        assert_eq!(
            read_float(&Value::String("-Infinity".to_owned())).unwrap(),
            f64::NEG_INFINITY
        );
        assert_eq!(
            read_float(&Value::String("MAX".to_owned())).unwrap(),
            f64::MAX
        );
    }
}
