use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process;

#[derive(Clone, Debug)]
struct PredictionRow {
    candidate_id: String,
    score: f64,
    relevant: bool,
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(&mut output, "{byte:02x}");
    }
    output
}

fn parse_predictions(bytes: &[u8]) -> Result<Vec<PredictionRow>, String> {
    let value: Value = serde_json::from_slice(bytes)
        .map_err(|error| format!("prediction JSON parse failed: {error}"))?;
    let records = value
        .as_array()
        .ok_or_else(|| "prediction input must be a JSON array".to_owned())?;
    if records.is_empty() {
        return Err("prediction input must contain at least one record".to_owned());
    }

    let mut seen = HashSet::with_capacity(records.len());
    let mut rows = Vec::with_capacity(records.len());
    for (index, record) in records.iter().enumerate() {
        let context = format!("record[{index}]");
        let candidate_id = record
            .get("candidate_id")
            .and_then(Value::as_str)
            .filter(|text| !text.trim().is_empty())
            .ok_or_else(|| format!("{context}: candidate_id must be a non-empty string"))?
            .to_owned();
        if !seen.insert(candidate_id.clone()) {
            return Err(format!("{context}: duplicate candidate_id {candidate_id}"));
        }

        let score = record
            .get("score")
            .and_then(Value::as_f64)
            .ok_or_else(|| format!("{context}: score must be a finite number"))?;
        if !score.is_finite() {
            return Err(format!("{context}: score must be finite"));
        }

        let relevant = record
            .get("relevant")
            .and_then(Value::as_bool)
            .ok_or_else(|| format!("{context}: relevant must be a boolean"))?;
        rows.push(PredictionRow {
            candidate_id,
            score,
            relevant,
        });
    }
    Ok(rows)
}

fn validate_budgets(budgets: &[usize], row_count: usize) -> Result<(), String> {
    if budgets.is_empty() {
        return Err("at least one candidate budget K is required".to_owned());
    }
    let mut seen = HashSet::with_capacity(budgets.len());
    let mut previous = 0;
    for budget in budgets {
        if *budget == 0 || *budget > row_count {
            return Err(format!(
                "candidate budget K={budget} must be in 1..={row_count}"
            ));
        }
        if !seen.insert(*budget) {
            return Err(format!("duplicate candidate budget K={budget}"));
        }
        if *budget <= previous {
            return Err("candidate budgets must be strictly increasing".to_owned());
        }
        previous = *budget;
    }
    Ok(())
}

fn discovery_metrics(
    mut rows: Vec<PredictionRow>,
    budgets: &[usize],
    input_digest: &str,
) -> Result<Value, String> {
    if rows.is_empty() {
        return Err("no prediction rows supplied".to_owned());
    }
    validate_budgets(budgets, rows.len())?;

    // Ranking-only semantics: descending score, deterministic candidate-ID tie break.
    rows.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.candidate_id.cmp(&right.candidate_id))
    });

    let positives = rows.iter().filter(|row| row.relevant).count();
    let prevalence = positives as f64 / rows.len() as f64;

    let mut hits = 0usize;
    let mut precision_sum = 0.0f64;
    let mut hits_by_rank = Vec::with_capacity(rows.len());
    for (rank, row) in rows.iter().enumerate() {
        if row.relevant {
            hits += 1;
            precision_sum += hits as f64 / (rank + 1) as f64;
        }
        hits_by_rank.push(hits);
    }

    let average_precision = if positives == 0 {
        None
    } else {
        Some(precision_sum / positives as f64)
    };

    let mut budget_metrics = Vec::with_capacity(budgets.len());
    for budget in budgets {
        let hits_at_k = hits_by_rank[*budget - 1];
        let precision_at_k = hits_at_k as f64 / *budget as f64;
        let recall_at_k = if positives == 0 {
            None
        } else {
            Some(hits_at_k as f64 / positives as f64)
        };
        let enrichment_over_random = if prevalence <= 0.0 {
            None
        } else {
            Some(precision_at_k / prevalence)
        };

        budget_metrics.push(json!({
            "k": budget,
            "hits_at_k": hits_at_k,
            "precision_at_k": precision_at_k,
            "recall_at_k": recall_at_k,
            "random_expected_hits": (*budget as f64) * prevalence,
            "random_expected_precision": prevalence,
            "enrichment_over_random": enrichment_over_random
        }));
    }

    Ok(json!({
        "status": "pass",
        "metric_contract": "SC-DISCOVERY-RANKING-METRICS-0.1",
        "score_semantics": "ranking_only_not_a_calibrated_probability",
        "input_sha256": input_digest,
        "candidate_count": rows.len(),
        "positive_count": positives,
        "prevalence": prevalence,
        "average_precision": average_precision,
        "candidate_budgets": budget_metrics,
        "ranking": rows.iter().map(|row| json!({
            "candidate_id": row.candidate_id,
            "score": row.score,
            "relevant": row.relevant
        })).collect::<Vec<_>>(),
        "claim_ceiling": "Metrics summarize the supplied labels and ranking only; they do not establish experimental truth, synthesizability, or a causal discovery advantage."
    }))
}

fn usage(program: &str) {
    eprintln!(
        "Usage: {program} --predictions <rows.json> --budgets <K1,K2,...> [--out <metrics.json>]"
    );
}

fn read_file(path: &PathBuf) -> Vec<u8> {
    fs::read(path).unwrap_or_else(|error| {
        eprintln!("could not read {}: {error}", path.display());
        process::exit(1);
    })
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut predictions_path: Option<PathBuf> = None;
    let mut budgets_text: Option<String> = None;
    let mut output_path: Option<PathBuf> = None;
    let mut index = 1;

    while index < args.len() {
        match args[index].as_str() {
            "--predictions" | "--budgets" | "--out" => {
                if index + 1 >= args.len() {
                    usage(&args[0]);
                    process::exit(2);
                }
                let value = args[index + 1].clone();
                match args[index].as_str() {
                    "--predictions" => predictions_path = Some(PathBuf::from(value)),
                    "--budgets" => budgets_text = Some(value),
                    "--out" => output_path = Some(PathBuf::from(value)),
                    _ => unreachable!(),
                }
                index += 2;
            }
            "--help" | "-h" => {
                usage(&args[0]);
                return;
            }
            unknown => {
                eprintln!("unknown argument: {unknown}");
                usage(&args[0]);
                process::exit(2);
            }
        }
    }

    let (Some(predictions_path), Some(budgets_text)) = (predictions_path, budgets_text) else {
        usage(&args[0]);
        process::exit(2);
    };
    let budgets: Vec<usize> = budgets_text
        .split(',')
        .map(|value| {
            value
                .trim()
                .parse::<usize>()
                .map_err(|error| format!("invalid candidate budget {value:?}: {error}"))
        })
        .collect::<Result<_, _>>()
        .unwrap_or_else(|error| {
            eprintln!("{error}");
            process::exit(2);
        });

    let input_bytes = read_file(&predictions_path);
    let rows = parse_predictions(&input_bytes).unwrap_or_else(|error| {
        eprintln!("SC-DISCOVERY-METRICS-FAIL: {error}");
        process::exit(1);
    });
    let digest = sha256_hex(&input_bytes);
    let report = discovery_metrics(rows, &budgets, &digest).unwrap_or_else(|error| {
        eprintln!("SC-DISCOVERY-METRICS-FAIL: {error}");
        process::exit(1);
    });
    let rendered = serde_json::to_string_pretty(&report)
        .expect("serializing the metrics receipt should not fail");

    if let Some(path) = output_path {
        if let Err(error) = fs::write(&path, format!("{rendered}\n")) {
            eprintln!("could not write {}: {error}", path.display());
            process::exit(1);
        }
    } else {
        println!("{rendered}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn row(id: &str, score: f64, relevant: bool) -> PredictionRow {
        PredictionRow {
            candidate_id: id.to_owned(),
            score,
            relevant,
        }
    }

    #[test]
    fn top_k_metrics_use_ranked_hits_and_random_prevalence() {
        let rows = vec![
            row("a", 0.9, true),
            row("b", 0.8, false),
            row("c", 0.7, true),
            row("d", 0.6, false),
        ];
        let report = discovery_metrics(rows, &[2, 4], "fixture-digest").unwrap();
        assert_eq!(report["candidate_count"], 4);
        assert_eq!(report["positive_count"], 2);
        assert_eq!(report["candidate_budgets"][0]["hits_at_k"], 1);
        assert_eq!(report["candidate_budgets"][0]["precision_at_k"], 0.5);
        assert_eq!(report["candidate_budgets"][0]["recall_at_k"], 0.5);
        assert_eq!(report["candidate_budgets"][0]["enrichment_over_random"], 1.0);
        assert_eq!(report["candidate_budgets"][1]["precision_at_k"], 0.5);
        assert_eq!(report["average_precision"], 0.8333333333333333);
    }

    #[test]
    fn equal_scores_use_candidate_id_as_stable_tie_break() {
        let rows = vec![row("z", 0.5, true), row("a", 0.5, false)];
        let report = discovery_metrics(rows, &[1, 2], "fixture-digest").unwrap();
        assert_eq!(report["ranking"][0]["candidate_id"], "a");
        assert_eq!(report["ranking"][1]["candidate_id"], "z");
    }

    #[test]
    fn duplicate_ids_fail_closed_during_parse() {
        let input = json!([
            {"candidate_id":"same","score":0.1,"relevant":true},
            {"candidate_id":"same","score":0.2,"relevant":false}
        ]);
        assert!(parse_predictions(&serde_json::to_vec(&input).unwrap()).is_err());
    }

    #[test]
    fn invalid_or_duplicate_budgets_fail_closed() {
        assert!(validate_budgets(&[1, 1], 4).is_err());
        assert!(validate_budgets(&[0], 4).is_err());
        assert!(validate_budgets(&[5], 4).is_err());
        assert!(validate_budgets(&[2, 1], 4).is_err());
    }

    #[test]
    fn no_positive_labels_return_null_recall_and_enrichment() {
        let rows = vec![row("a", 0.9, false), row("b", 0.8, false)];
        let report = discovery_metrics(rows, &[1, 2], "fixture-digest").unwrap();
        assert!(report["average_precision"].is_null());
        assert!(report["candidate_budgets"][0]["recall_at_k"].is_null());
        assert!(report["candidate_budgets"][0]["enrichment_over_random"].is_null());
    }

    #[test]
    fn non_finite_scores_are_rejected() {
        let input = br#"[{"candidate_id":"x","score":null,"relevant":true}]"#;
        assert!(parse_predictions(input).is_err());
    }
}
