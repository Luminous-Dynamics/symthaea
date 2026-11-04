use std::{env, fs, process};

use project_underwriting_oracle::evaluate_corpus;
use serde_json::{Value, json};

fn read_json(path: &str) -> Result<Value, Box<dyn std::error::Error>> {
    let contents = fs::read_to_string(path)?;
    Ok(serde_json::from_str(&contents)?)
}

fn main() {
    if let Err(error) = run() {
        eprintln!("project-underwriting-oracle: {error}");
        process::exit(2);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let arguments: Vec<String> = env::args().collect();
    if arguments.len() != 3 {
        return Err(format!(
            "usage: {} <public-cases.json> <reference-expected.json>",
            arguments.first().map(String::as_str).unwrap_or("project-underwriting-oracle")
        )
        .into());
    }

    let corpus = read_json(&arguments[1])?;
    let reference = read_json(&arguments[2])?;
    let results = evaluate_corpus(&corpus);
    let expected = reference
        .get("expected")
        .and_then(Value::as_array)
        .ok_or("reference file must contain an expected array")?;

    let mut mismatches = Vec::new();
    for result in &results {
        let actual = result.to_value();
        let expected_case = expected
            .iter()
            .find(|item| item.get("case_id") == actual.get("case_id"));
        let Some(expected_case) = expected_case else {
            mismatches.push(json!({
                "case_id": actual["case_id"],
                "reason": "missing_reference_case"
            }));
            continue;
        };

        for field in ["input_disposition", "risk_disposition", "decision_authority", "blockers", "warnings"] {
            if actual.get(field) != expected_case.get(field) {
                mismatches.push(json!({
                    "case_id": actual["case_id"],
                    "field": field,
                    "actual": actual.get(field),
                    "expected": expected_case.get(field)
                }));
            }
        }
    }

    let receipt = json!({
        "receipt_version": "1.0",
        "corpus_id": corpus.get("corpus_id"),
        "case_count": results.len(),
        "reference_case_count": expected.len(),
        "all_reference_dispositions_match": mismatches.is_empty() && results.len() == expected.len(),
        "results": results.iter().map(|result| result.to_value()).collect::<Vec<_>>(),
        "mismatches": mismatches
    });
    println!("{}", serde_json::to_string_pretty(&receipt)?);

    if receipt["all_reference_dispositions_match"] != true {
        process::exit(1);
    }
    Ok(())
}
