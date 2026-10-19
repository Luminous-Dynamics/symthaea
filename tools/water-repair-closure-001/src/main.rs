use std::env;
use std::error::Error;
use std::fs;
use std::io;
use std::path::PathBuf;
use water_repair_closure_001::{
    evaluate_all, sha256_hex, validate_input_corpus, validate_oracle, CaseResult, InputCorpus,
    OracleCorpus, RunReceipt, CLAIM_CEILING, INPUT_SCHEMA_VERSION, ORACLE_SCHEMA_VERSION,
};

fn invalid_input(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

fn invalid_data(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("water-repair-closure-001: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let mut inputs_path: Option<PathBuf> = None;
    let mut oracle_path: Option<PathBuf> = None;
    let mut receipt_path: Option<PathBuf> = None;
    let mut implementation_sha = env::var("GITHUB_SHA").unwrap_or_default();

    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        let value = args
            .next()
            .ok_or_else(|| invalid_input(format!("missing value after {arg}")))?;
        match arg.as_str() {
            "--inputs" => inputs_path = Some(PathBuf::from(value)),
            "--oracle" => oracle_path = Some(PathBuf::from(value)),
            "--receipt" => receipt_path = Some(PathBuf::from(value)),
            "--implementation-sha" => implementation_sha = value,
            _ => return Err(invalid_input(format!("unknown argument: {arg}")).into()),
        }
    }

    let inputs_path =
        inputs_path.ok_or_else(|| invalid_input("required argument --inputs is missing"))?;
    let oracle_path =
        oracle_path.ok_or_else(|| invalid_input("required argument --oracle is missing"))?;
    let receipt_path =
        receipt_path.ok_or_else(|| invalid_input("required argument --receipt is missing"))?;

    let input_bytes = fs::read(&inputs_path)?;
    let oracle_bytes = fs::read(&oracle_path)?;
    let input_corpus: InputCorpus = serde_json::from_slice(&input_bytes)?;
    let oracle: OracleCorpus = serde_json::from_slice(&oracle_bytes)?;
    validate_input_corpus(&input_corpus).map_err(invalid_data)?;
    validate_oracle(&oracle).map_err(invalid_data)?;
    let oracle_schema_version = oracle.schema_version.clone();

    let results: Vec<CaseResult> =
        evaluate_all(&input_corpus).map_err(invalid_data)?;
    let mut expected = oracle.results;
    expected.sort_by(|left, right| left.case_id.cmp(&right.case_id));

    let mut mismatches = Vec::new();
    if results.len() != expected.len() {
        mismatches.push(format!(
            "result count mismatch: actual={}, expected={}",
            results.len(),
            expected.len()
        ));
    }
    for (actual, target) in results.iter().zip(expected.iter()) {
        if actual != target {
            mismatches.push(format!(
                "{}: actual={:?}/{:?}/effect={}, expected={:?}/{:?}/effect={}",
                actual.case_id,
                actual.disposition,
                actual.reason_codes,
                actual.semantic_effect_count,
                target.disposition,
                target.reason_codes,
                target.semantic_effect_count
            ));
        }
    }

    let implementation_sha_is_exact =
        implementation_sha.len() == 40
            && implementation_sha.bytes().all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase());
    let qualification_status = if !mismatches.is_empty() {
        "Failed"
    } else if implementation_sha_is_exact {
        "PassedExactHead"
    } else {
        "PassedLocalOracle"
    };

    let receipt = RunReceipt {
        schema_version: "water-repair-closure-001-receipt-v1".to_owned(),
        qualification_status: qualification_status.to_owned(),
        claim_ceiling: CLAIM_CEILING.to_owned(),
        implementation_sha: if implementation_sha_is_exact {
            implementation_sha
        } else {
            "UnpinnedLocalExecution".to_owned()
        },
        inputs_sha256: sha256_hex(&input_bytes),
        oracle_sha256: sha256_hex(&oracle_bytes),
        case_count: results.len(),
        mismatches,
        results,
    };

    if let Some(parent) = receipt_path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    fs::write(&receipt_path, serde_json::to_vec_pretty(&receipt)?)?;
    println!(
        "qualification_status={} case_count={} inputs_sha256={} oracle_sha256={}",
        receipt.qualification_status, receipt.case_count, receipt.inputs_sha256, receipt.oracle_sha256
    );

    if receipt.qualification_status == "Failed" {
        return Err(invalid_data(format!(
            "{} oracle mismatch(es); receipt written to {}",
            receipt.mismatches.len(),
            receipt_path.display()
        ))
        .into());
    }
    if receipt.schema_version != "water-repair-closure-001-receipt-v1"
        || input_corpus.schema_version != INPUT_SCHEMA_VERSION
        || oracle_schema_version != ORACLE_SCHEMA_VERSION
    {
        return Err(invalid_data("internal receipt schema identity mismatch").into());
    }
    Ok(())
}
