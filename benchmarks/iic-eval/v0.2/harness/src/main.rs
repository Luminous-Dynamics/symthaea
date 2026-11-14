use serde_json::{json, Value};
use std::{collections::{HashMap, HashSet}, env, error::Error, fs, io};

fn err(message: impl Into<String>) -> Box<dyn Error> {
    Box::new(io::Error::new(io::ErrorKind::InvalidData, message.into()))
}

fn read_json(path: &str) -> Result<Value, Box<dyn Error>> {
    let bytes = fs::read(path)?;
    Ok(serde_json::from_slice(&bytes)?)
}

fn required_str<'a>(value: &'a Value, key: &str, context: &str) -> Result<&'a str, Box<dyn Error>> {
    value.get(key).and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| err(format!("{context}: missing or empty string field '{key}'")))
}

fn required_array<'a>(value: &'a Value, key: &str, context: &str) -> Result<&'a Vec<Value>, Box<dyn Error>> {
    value.get(key).and_then(Value::as_array)
        .ok_or_else(|| err(format!("{context}: missing or invalid array field '{key}'")))
}

fn span_matches(content: &str, start: usize, end: usize, quote: &str) -> bool {
    if start > end || end > content.len() || !content.is_char_boundary(start) || !content.is_char_boundary(end) {
        return false;
    }
    &content[start..end] == quote
}

fn string_set(value: &Value, key: &str, context: &str) -> Result<Vec<String>, Box<dyn Error>> {
    let values = required_array(value, key, context)?;
    let mut out = Vec::with_capacity(values.len());
    for item in values {
        let text = item.as_str().filter(|s| !s.is_empty())
            .ok_or_else(|| err(format!("{context}: '{key}' must contain nonempty strings")))?;
        out.push(text.to_owned());
    }
    out.sort();
    Ok(out)
}

fn index_unique<'a>(items: &'a [Value], id_key: &str, label: &str) -> Result<HashMap<String, &'a Value>, Box<dyn Error>> {
    let mut map = HashMap::new();
    for item in items {
        let id = required_str(item, id_key, label)?.to_owned();
        if map.insert(id.clone(), item).is_some() {
            return Err(err(format!("{label}: duplicate identifier '{id}'")));
        }
    }
    Ok(map)
}

fn run() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 4 {
        return Err(err("usage: iic-eval-v0-2 <cases.json> <oracle.json> <submission.json>"));
    }
    let corpus = read_json(&args[1])?;
    let oracle = read_json(&args[2])?;
    let submission = read_json(&args[3])?;

    if corpus.get("corpus_version").and_then(Value::as_str) != Some("0.2")
        || corpus.get("synthetic").and_then(Value::as_bool) != Some(true) {
        return Err(err("input corpus must be synthetic IIC-Eval v0.2"));
    }
    if oracle.get("corpus_version").and_then(Value::as_str) != Some("0.2")
        || oracle.get("synthetic").and_then(Value::as_bool) != Some(true) {
        return Err(err("oracle must be synthetic IIC-Eval v0.2"));
    }
    if required_str(&submission, "system_id", "submission").is_err()
        || required_str(&submission, "build_sha", "submission").is_err()
        || required_str(&submission, "configuration_sha256", "submission").is_err() {
        return Err(err("submission requires system_id, build_sha, and configuration_sha256"));
    }

    let input_cases = required_array(&corpus, "cases", "corpus")?;
    let oracle_cases = required_array(&oracle, "cases", "oracle")?;
    let results = required_array(&submission, "results", "submission")?;
    let inputs = index_unique(input_cases, "case_id", "corpus")?;
    let gold = index_unique(oracle_cases, "case_id", "oracle")?;
    let outputs = index_unique(results, "case_id", "submission")?;

    if inputs.len() != gold.len() || inputs.keys().any(|id| !gold.contains_key(id)) {
        return Err(err("corpus and oracle case identifiers are not a one-to-one match"));
    }
    if inputs.len() != outputs.len() || inputs.keys().any(|id| !outputs.contains_key(id)) {
        return Err(err("submission must contain exactly one result for every corpus case"));
    }

    let mut total_gold = 0usize;
    let mut total_submitted = 0usize;
    let mut true_positive = 0usize;
    let mut valid_spans = 0usize;
    let mut source_version_matches = 0usize;
    let mut modality_correct = 0usize;
    let mut condition_correct = 0usize;
    let mut state_correct = 0usize;
    let mut matched_attributes = 0usize;
    let mut false_mandatory_promotions = 0usize;
    let mut gold_prohibitions = 0usize;
    let mut preserved_prohibitions = 0usize;
    let mut duplicate_candidates = 0usize;
    let mut per_case = Vec::new();

    let mut ordered_ids: Vec<String> = inputs.keys().cloned().collect();
    ordered_ids.sort();

    for case_id in ordered_ids {
        let input = inputs[&case_id];
        let answer = gold[&case_id];
        let output = outputs[&case_id];
        let docs = required_array(input, "documents", &format!("{case_id} input"))?;
        let mut doc_map: HashMap<String, &Value> = HashMap::new();
        for doc in docs {
            let id = required_str(doc, "document_id", &format!("{case_id} document"))?.to_owned();
            if doc_map.insert(id.clone(), doc).is_some() {
                return Err(err(format!("{case_id}: duplicate document id '{id}'")));
            }
        }

        let gold_candidates = required_array(answer, "expected_candidates", &format!("{case_id} oracle"))?;
        let submitted = required_array(output, "candidates", &format!("{case_id} output"))?;
        let mut gold_map: HashMap<(String, String), &Value> = HashMap::new();
        for candidate in gold_candidates {
            let doc_id = required_str(candidate, "source_document_id", "oracle candidate")?.to_owned();
            let quote = required_str(candidate, "verbatim_span", "oracle candidate")?.to_owned();
            let doc = doc_map.get(&doc_id).ok_or_else(|| err(format!("{case_id}: oracle references unknown source '{doc_id}'")))?;
            let content = required_str(doc, "content", "input document")?;
            if !content.contains(&quote) {
                return Err(err(format!("{case_id}: oracle span not present in source '{doc_id}'")));
            }
            if gold_map.insert((doc_id, quote), candidate).is_some() {
                return Err(err(format!("{case_id}: duplicate gold source span")));
            }
        }

        let mut seen = HashSet::new();
        let mut matched_keys = HashSet::new();
        let mut case_tp = 0usize;
        let mut matched_candidate_refs: HashSet<String> = HashSet::new();
        let mut case_valid_spans = 0usize;
        let mut case_version = 0usize;
        let mut case_modality = 0usize;
        let mut case_conditions = 0usize;
        let mut case_state = 0usize;
        let mut case_fp = 0usize;
        let mut case_false_mandatory = 0usize;

        for candidate in submitted {
            total_submitted += 1;
            let doc_id = required_str(candidate, "source_document_id", "submission candidate")?.to_owned();
            let source_version = required_str(candidate, "source_version", "submission candidate")?;
            let quote = required_str(candidate, "quoted_text", "submission candidate")?.to_owned();
            let modality = required_str(candidate, "modality", "submission candidate")?;
            let state = required_str(candidate, "adjudication_state", "submission candidate")?;
            let start = candidate.get("span_start_byte").and_then(Value::as_u64)
                .ok_or_else(|| err("span_start_byte must be an unsigned integer"))? as usize;
            let end = candidate.get("span_end_byte").and_then(Value::as_u64)
                .ok_or_else(|| err("span_end_byte must be an unsigned integer"))? as usize;
            let conditions = string_set(candidate, "conditions", "submission candidate")?;
            let identity = (doc_id.clone(), start, end);
            if !seen.insert(identity) {
                duplicate_candidates += 1;
                case_fp += 1;
                continue;
            }

            let Some(doc) = doc_map.get(&doc_id) else {
                case_fp += 1;
                if modality == "mandatory" { false_mandatory_promotions += 1; case_false_mandatory += 1; }
                continue;
            };
            let content = required_str(doc, "content", "input document")?;
            let version = required_str(doc, "version", "input document")?;
            if span_matches(content, start, end, &quote) {
                valid_spans += 1;
                case_valid_spans += 1;
            }
            if source_version == version {
                source_version_matches += 1;
                case_version += 1;
            }
            let key = (doc_id.clone(), quote.clone());
            let Some(gold_candidate) = gold_map.get(&key) else {
                case_fp += 1;
                if modality == "mandatory" { false_mandatory_promotions += 1; case_false_mandatory += 1; }
                continue;
            };
            if !span_matches(content, start, end, &quote) || source_version != version {
                case_fp += 1;
                if modality == "mandatory" { false_mandatory_promotions += 1; case_false_mandatory += 1; }
                continue;
            }
            if !matched_keys.insert(key) {
                duplicate_candidates += 1;
                case_fp += 1;
                continue;
            }
            true_positive += 1;
            case_tp += 1;
            matched_candidate_refs.insert(required_str(gold_candidate, "candidate_ref", "oracle candidate")?.to_owned());
            matched_attributes += 1;
            let gold_modality = required_str(gold_candidate, "modality", "oracle candidate")?;
            if modality == gold_modality { modality_correct += 1; case_modality += 1; }
            else if modality == "mandatory" { false_mandatory_promotions += 1; case_false_mandatory += 1; }
            if conditions == string_set(gold_candidate, "conditions", "oracle candidate")? {
                condition_correct += 1; case_conditions += 1;
            }
            if state == required_str(gold_candidate, "adjudication_state", "oracle candidate")? {
                state_correct += 1; case_state += 1;
            }
        }

        let temporal_queries = answer.get("temporal_queries").and_then(Value::as_array).cloned().unwrap_or_default();
        let mut temporal_required = 0usize;
        let mut temporal_complete = 0usize;
        for query in &temporal_queries {
            let refs = required_array(query, "required_candidate_refs", "oracle temporal query")?;
            temporal_required += 1;
            let all_present = refs.iter().all(|r| {
                r.as_str().map(|id| matched_candidate_refs.contains(id)).unwrap_or(false)
            });
            if all_present { temporal_complete += 1; }
        }

        let mut case_gold_prohibitions = 0usize;
        for c in gold_candidates {
            total_gold += 1;
            let modality = required_str(c, "modality", "oracle candidate")?;
            if modality == "prohibition" {
                gold_prohibitions += 1;
                case_gold_prohibitions += 1;
                let doc_id = required_str(c, "source_document_id", "oracle candidate")?.to_owned();
                let quote = required_str(c, "verbatim_span", "oracle candidate")?.to_owned();
                if matched_keys.contains(&(doc_id, quote)) {
                    let submitted_candidate = submitted.iter().find(|candidate| {
                        candidate.get("source_document_id").and_then(Value::as_str) == Some(required_str(c, "source_document_id", "oracle candidate").unwrap_or(""))
                            && candidate.get("quoted_text").and_then(Value::as_str) == Some(required_str(c, "verbatim_span", "oracle candidate").unwrap_or(""))
                    });
                    if submitted_candidate.and_then(|v|v.get("modality")).and_then(Value::as_str)==Some("prohibition") {
                        preserved_prohibitions += 1;
                    }
                }
            }
        }
        per_case.push(json!({
            "case_id": case_id,
            "gold_candidates": gold_candidates.len(),
            "submitted_candidates": submitted.len(),
            "matched_source_spans": case_tp,
            "false_positive_candidates": case_fp,
            "valid_utf8_byte_spans": case_valid_spans,
            "source_version_matches": case_version,
            "modality_exact": case_modality,
            "conditions_exact": case_conditions,
            "adjudication_state_exact": case_state,
            "false_mandatory_promotions": case_false_mandatory,
            "gold_prohibitions": case_gold_prohibitions,
            "temporal_query_slices": temporal_required,
            "temporal_query_slices_complete": temporal_complete
        }));
    }

    let precision = if total_submitted == 0 { 0.0 } else { true_positive as f64 / total_submitted as f64 };
    let total_temporal_queries: usize = per_case.iter().map(|v| v["temporal_query_slices"].as_u64().unwrap_or(0) as usize).sum();
    let complete_temporal_queries: usize = per_case.iter().map(|v| v["temporal_query_slices_complete"].as_u64().unwrap_or(0) as usize).sum();
    let temporal_coverage = if total_temporal_queries == 0 { 1.0 } else { complete_temporal_queries as f64 / total_temporal_queries as f64 };
    let recall = if total_gold == 0 { 0.0 } else { true_positive as f64 / total_gold as f64 };
    let f1 = if precision + recall == 0.0 { 0.0 } else { 2.0 * precision * recall / (precision + recall) };
    let attr_denominator = matched_attributes.max(1) as f64;
    let report = json!({
        "evaluator_version":"iic-eval-v0.2-rust-0.1",
        "corpus_version":"0.2",
        "system_id":submission["system_id"],
        "build_sha":submission["build_sha"],
        "configuration_sha256":submission["configuration_sha256"],
        "case_count":per_case.len(),
        "metrics":{
            "gold_candidate_count":total_gold,
            "submitted_candidate_count":total_submitted,
            "matched_source_spans":true_positive,
            "source_span_precision":precision,
            "source_span_recall":recall,
            "source_span_f1":f1,
            "valid_utf8_byte_span_count":valid_spans,
            "source_version_match_count":source_version_matches,
            "modality_exact_accuracy_on_matched_spans":modality_correct as f64 / attr_denominator,
            "condition_exact_accuracy_on_matched_spans":condition_correct as f64 / attr_denominator,
            "adjudication_state_exact_accuracy_on_matched_spans":state_correct as f64 / attr_denominator,
            "false_mandatory_promotions":false_mandatory_promotions,
            "explicit_prohibitions_in_gold":gold_prohibitions,
            "explicit_prohibitions_preserved":preserved_prohibitions,
            "duplicate_candidates":duplicate_candidates,
            "temporal_query_slice_count":total_temporal_queries,
            "complete_temporal_query_slices":complete_temporal_queries,
            "decision_time_evidence_coverage":temporal_coverage
        },
        "per_case":per_case,
        "interpretation_limits":[
            "Exact source-span matching is a structural proxy; it does not establish semantic equivalence.",
            "The development oracle is not independently adjudicated.",
            "Required clarification questions and prohibited inferences require expert adjudication and are not scored by this evaluator.",
            "No legal, authority, safety, or production-readiness conclusion follows from this report."
        ]
    });
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("iic-eval-v0.2: {error}");
        std::process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::span_matches;

    #[test]
    fn accepts_exact_ascii_byte_span() {
        let source = "Inspect the north gate.";
        assert!(span_matches(source, 0, 7, "Inspect"));
    }

    #[test]
    fn accepts_exact_unicode_byte_span() {
        let source = "The café’s Māori sign is legible.";
        let start = source.find("Māori").unwrap();
        let end = start + "Māori".len();
        assert!(span_matches(source, start, end, "Māori"));
    }

    #[test]
    fn rejects_character_offsets_that_split_utf8() {
        let source = "café";
        assert!(!span_matches(source, 3, 5, "é"));
    }

    #[test]
    fn rejects_wrong_quote_even_with_valid_boundaries() {
        assert!(!span_matches("North gate", 0, 5, "South"));
    }
}
