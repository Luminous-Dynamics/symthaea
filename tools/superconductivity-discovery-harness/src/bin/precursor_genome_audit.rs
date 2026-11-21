use jsonschema::validator_for;
use serde_json::{Value, json};
use sha1::Sha1;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashSet};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process;

const CONTRACT: &str = "SC-PRECURSOR-GENOME-SOURCE-0.1";
const CLAIM_CEILING: &str = "Frozen identity and profile for the public Precursor Genome ledger. Counts are source-snapshot bookkeeping, not the paper's model-training cohort or raw in-situ onset measurements.";
const CATEGORIES: [&str; 4] = [
    "transformed",
    "unreacted",
    "partially_reacted",
    "completely_reacted",
];

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn git_blob_sha1(bytes: &[u8]) -> String {
    let header = format!("blob {}\0", bytes.len());
    let mut hasher = Sha1::new();
    hasher.update(header.as_bytes());
    hasher.update(bytes);
    let digest = hasher.finalize();
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn nonempty_string<'a>(value: &'a Value, key: &str) -> Result<&'a str, String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| format!("missing or empty string field {key}"))
}

fn count_field(value: &Value, key: &str) -> Result<usize, String> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .and_then(|number| usize::try_from(number).ok())
        .ok_or_else(|| format!("missing or invalid non-negative integer field {key}"))
}

fn optional_index(value: &Value, key: &str) -> Option<usize> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .and_then(|number| usize::try_from(number).ok())
}

fn number_is_finite(value: Option<&Value>) -> bool {
    value.and_then(Value::as_f64).is_some_and(f64::is_finite)
}

fn json_count_map(value: &Value, key: &str) -> Result<BTreeMap<String, usize>, String> {
    let object = value
        .get(key)
        .and_then(Value::as_object)
        .ok_or_else(|| format!("manifest expected_profile.{key} must be an object"))?;
    let mut counts = BTreeMap::new();
    for (name, count) in object {
        let number = count
            .as_u64()
            .and_then(|number| usize::try_from(number).ok())
            .ok_or_else(|| {
                format!("manifest expected_profile.{key}.{name} must be a non-negative integer")
            })?;
        counts.insert(name.clone(), number);
    }
    Ok(counts)
}

fn check_count(profile: &Value, key: &str, observed: usize) -> Result<(), String> {
    let expected = count_field(profile, key)?;
    if expected != observed {
        return Err(format!(
            "frozen source profile mismatch for {key}: expected {expected}, observed {observed}"
        ));
    }
    Ok(())
}

fn check_map(profile: &Value, key: &str, observed: &BTreeMap<String, usize>) -> Result<(), String> {
    let expected = json_count_map(profile, key)?;
    if &expected != observed {
        return Err(format!(
            "frozen source profile mismatch for {key}: expected {expected:?}, observed {observed:?}"
        ));
    }
    Ok(())
}

fn active_refinement(sample: &Value) -> Option<&Value> {
    let scan_index = optional_index(sample, "active_scan_index")?;
    let scans = sample.pointer("/characterization/xrd/scans")?.as_array()?;
    let scan = scans.get(scan_index)?;
    let case_index = optional_index(scan, "active_case_index")?;
    let cases = scan.get("refinement_cases")?.as_array()?;
    cases.get(case_index)
}

fn audit(ledger_bytes: &[u8], schema_bytes: &[u8], manifest_bytes: &[u8]) -> Result<Value, String> {
    let schema: Value = serde_json::from_slice(schema_bytes)
        .map_err(|error| format!("manifest schema JSON parse failed: {error}"))?;
    let manifest: Value = serde_json::from_slice(manifest_bytes)
        .map_err(|error| format!("source manifest JSON parse failed: {error}"))?;
    let ledger: Value = serde_json::from_slice(ledger_bytes)
        .map_err(|error| format!("source ledger JSON parse failed: {error}"))?;

    const DIALECT: &str = "https://json-schema.org/draft/2020-12/schema";
    if schema.get("$schema").and_then(Value::as_str) != Some(DIALECT) {
        return Err(format!("source manifest schema must declare {DIALECT}"));
    }
    let validator = validator_for(&schema)
        .map_err(|error| format!("manifest schema compilation failed: {error}"))?;
    if let Some(error) = validator.iter_errors(&manifest).next() {
        return Err(format!(
            "source manifest schema validation failed: {error} at {}",
            error.instance_path()
        ));
    }
    if nonempty_string(&manifest, "contract")? != CONTRACT {
        return Err("unsupported Precursor Genome source-manifest contract".to_owned());
    }
    if nonempty_string(&manifest, "claim_ceiling")? != CLAIM_CEILING {
        return Err("source manifest has an unsupported claim ceiling".to_owned());
    }
    if nonempty_string(&ledger, "repository_name")? != "Precursor_Genome_Alpha" {
        return Err(
            "source ledger repository_name does not match the pinned Precursor Genome release"
                .to_owned(),
        );
    }

    let actual_blob_sha1 = git_blob_sha1(ledger_bytes);
    let expected_blob_sha1 = nonempty_string(&manifest, "ledger_git_blob_sha1")?;
    if !actual_blob_sha1.eq_ignore_ascii_case(expected_blob_sha1) {
        return Err(format!(
            "source ledger identity mismatch: expected Git blob {expected_blob_sha1}, got {actual_blob_sha1}"
        ));
    }

    let samples = ledger
        .get("samples")
        .and_then(Value::as_array)
        .ok_or("source ledger.samples must be an array")?;
    let profile = manifest
        .get("expected_profile")
        .ok_or("source manifest expected_profile is missing")?;

    let mut sample_ids = HashSet::new();
    let mut all_sample_ids = Vec::with_capacity(samples.len());
    let mut precursor_formulas = HashSet::new();
    let mut precursor_pairs = HashSet::new();
    let mut category_counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut scan_status_counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut active_quality_counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut active_acceptance_counts: BTreeMap<String, usize> = BTreeMap::new();

    let mut synthesis_objects = 0usize;
    let mut physical_failures = 0usize;
    let mut samples_with_heating_temperature = 0usize;
    let mut scan_count = 0usize;
    let mut refinement_case_count = 0usize;
    let mut samples_with_active_refinement = 0usize;
    let mut classified_temperature_no_failure = Vec::new();
    let mut accepted_quality_1_or_2_cohort = Vec::new();

    let mut category_ids: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut physical_failure_ids = Vec::new();
    let mut unclassified_ids = Vec::new();
    let mut missing_synthesis_ids = Vec::new();
    let mut missing_temperature_ids = Vec::new();
    let mut no_active_refinement_ids = Vec::new();
    let mut rejected_active_refinement_ids = Vec::new();
    let mut missing_quality_ids = Vec::new();
    let mut quality_3_ids = Vec::new();

    for sample in samples {
        let sample_id = nonempty_string(sample, "sample_id")?.to_owned();
        if !sample_ids.insert(sample_id.clone()) {
            return Err(format!("duplicate sample_id {sample_id}"));
        }
        all_sample_ids.push(sample_id.clone());

        let precursors = sample
            .get("precursors")
            .and_then(Value::as_array)
            .ok_or_else(|| format!("sample {sample_id} is missing its precursor array"))?;
        if precursors.len() != 2 {
            return Err(format!(
                "sample {sample_id} must have exactly two precursor entries"
            ));
        }
        let mut formulas = Vec::with_capacity(2);
        for precursor in precursors {
            let formula = nonempty_string(precursor, "formula")?.to_owned();
            precursor_formulas.insert(formula.clone());
            formulas.push(formula);
        }
        formulas.sort();
        let pair_key = format!("{} + {}", formulas[0], formulas[1]);
        if !precursor_pairs.insert(pair_key.clone()) {
            return Err(format!(
                "duplicate precursor pair {pair_key} at sample {sample_id}"
            ));
        }

        let category = sample
            .pointer("/outcome/reaction_category")
            .and_then(Value::as_str)
            .filter(|category| !category.trim().is_empty())
            .unwrap_or("unclassified")
            .to_owned();
        if category != "unclassified" && !CATEGORIES.contains(&category.as_str()) {
            return Err(format!(
                "sample {sample_id} has unsupported reaction category {category}"
            ));
        }
        *category_counts.entry(category.clone()).or_default() += 1;
        category_ids
            .entry(category.clone())
            .or_default()
            .push(sample_id.clone());
        if category == "unclassified" {
            unclassified_ids.push(sample_id.clone());
        }

        let physical_failure = sample
            .get("physical_failure")
            .is_some_and(|value| !value.is_null());
        if physical_failure {
            physical_failures += 1;
            physical_failure_ids.push(sample_id.clone());
        }

        let synthesis = sample.get("synthesis").filter(|value| !value.is_null());
        let has_synthesis = synthesis.is_some();
        if has_synthesis {
            synthesis_objects += 1;
        } else {
            missing_synthesis_ids.push(sample_id.clone());
        }
        let temperature_known = synthesis
            .and_then(|value| value.get("heating_temperature"))
            .is_some_and(|value| number_is_finite(Some(value)));
        if temperature_known {
            samples_with_heating_temperature += 1;
        } else {
            missing_temperature_ids.push(sample_id.clone());
        }

        let classified = category != "unclassified";
        let cohort_eligible = classified && temperature_known && !physical_failure;
        if cohort_eligible {
            classified_temperature_no_failure.push(sample_id.clone());
        }

        let scans = sample
            .pointer("/characterization/xrd/scans")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        scan_count += scans.len();
        for scan in &scans {
            let status = scan
                .get("status")
                .and_then(Value::as_str)
                .unwrap_or("missing");
            *scan_status_counts.entry(status.to_owned()).or_default() += 1;
            if let Some(cases) = scan.get("refinement_cases").and_then(Value::as_array) {
                refinement_case_count += cases.len();
            }
        }

        if let Some(refinement) = active_refinement(sample) {
            samples_with_active_refinement += 1;
            let quality = refinement
                .pointer("/verification/human_quality_score")
                .and_then(Value::as_u64)
                .map(|score| score.to_string())
                .unwrap_or_else(|| "missing".to_owned());
            let acceptance = match refinement
                .pointer("/verification/is_accepted")
                .and_then(Value::as_bool)
            {
                Some(true) => "accepted",
                Some(false) => "rejected",
                None => "missing",
            };
            *active_quality_counts.entry(quality.clone()).or_default() += 1;
            *active_acceptance_counts
                .entry(acceptance.to_owned())
                .or_default() += 1;

            if acceptance == "rejected" {
                rejected_active_refinement_ids.push(sample_id.clone());
            }
            if quality == "missing" {
                missing_quality_ids.push(sample_id.clone());
            }
            if quality == "3" {
                quality_3_ids.push(sample_id.clone());
            }
            if cohort_eligible && acceptance == "accepted" && (quality == "1" || quality == "2") {
                accepted_quality_1_or_2_cohort.push(sample_id.clone());
            }
        } else {
            *active_quality_counts
                .entry("no_active_refinement".to_owned())
                .or_default() += 1;
            no_active_refinement_ids.push(sample_id.clone());
        }
    }

    let mut expected_categories = CATEGORIES
        .iter()
        .map(|category| ((*category).to_owned(), 0usize))
        .collect::<BTreeMap<_, _>>();
    expected_categories.insert("unclassified".to_owned(), 0);
    for (category, count) in category_counts {
        expected_categories.insert(category, count);
    }

    let mut expected_quality = BTreeMap::new();
    for (reported, observed) in [
        ("score_1", "1"),
        ("score_2", "2"),
        ("score_3", "3"),
        ("missing", "missing"),
        ("no_active_refinement", "no_active_refinement"),
    ] {
        expected_quality.insert(
            reported.to_owned(),
            *active_quality_counts.get(observed).unwrap_or(&0),
        );
    }
    let mut expected_acceptance = BTreeMap::new();
    for key in ["accepted", "rejected", "missing"] {
        expected_acceptance.insert(
            key.to_owned(),
            *active_acceptance_counts.get(key).unwrap_or(&0),
        );
    }

    check_count(profile, "samples", samples.len())?;
    check_count(profile, "unique_sample_ids", sample_ids.len())?;
    check_count(
        profile,
        "unique_precursor_formulas",
        precursor_formulas.len(),
    )?;
    check_count(profile, "unique_precursor_pairs", precursor_pairs.len())?;
    check_count(profile, "synthesis_objects", synthesis_objects)?;
    check_count(profile, "physical_failures", physical_failures)?;
    check_count(
        profile,
        "samples_with_recorded_heating_temperature",
        samples_with_heating_temperature,
    )?;
    check_map(profile, "category_counts", &expected_categories)?;
    check_count(profile, "scans", scan_count)?;
    check_map(profile, "scans_by_status", &scan_status_counts)?;
    check_count(profile, "refinement_cases", refinement_case_count)?;
    check_count(
        profile,
        "samples_with_active_refinement",
        samples_with_active_refinement,
    )?;
    check_map(profile, "active_refinement_quality", &expected_quality)?;
    check_map(
        profile,
        "active_refinement_acceptance",
        &expected_acceptance,
    )?;
    check_count(
        profile,
        "classified_with_temperature_and_no_physical_failure",
        classified_temperature_no_failure.len(),
    )?;
    check_count(
        profile,
        "cohort_temperature_and_accepted_quality_1_or_2",
        accepted_quality_1_or_2_cohort.len(),
    )?;

    let paper_counts = manifest
        .pointer("/publication_comparison/paper_category_counts")
        .ok_or("manifest publication_comparison.paper_category_counts is missing")?;
    let mut paper_delta = BTreeMap::new();
    for category in CATEGORIES {
        let paper_count = count_field(paper_counts, category)?;
        let ledger_count = *expected_categories.get(category).unwrap_or(&0);
        paper_delta.insert(
            category.to_owned(),
            json!(ledger_count as i64 - paper_count as i64),
        );
    }
    let paper_total = count_field(
        manifest
            .get("publication_comparison")
            .expect("schema validated publication_comparison"),
        "paper_analyzable_outcomes",
    )?;
    let ledger_classified = samples.len() - *expected_categories.get("unclassified").unwrap_or(&0);

    Ok(json!({
        "status": "precursor_genome_source_audit_pass",
        "contract": "SC-PRECURSOR-GENOME-AUDIT-0.1",
        "source_repository": nonempty_string(&manifest, "source_repository")?,
        "source_commit": nonempty_string(&manifest, "source_commit")?,
        "ledger_path": nonempty_string(&manifest, "ledger_path")?,
        "ledger_git_blob_sha1": actual_blob_sha1,
        "ledger_sha256": sha256_hex(ledger_bytes),
        "manifest_sha256": sha256_hex(manifest_bytes),
        "measurement_units": manifest.get("measurement_units").cloned().unwrap_or(Value::Null),
        "manifest_schema_sha256": sha256_hex(schema_bytes),
        "counts": {
            "samples": samples.len(),
            "unique_sample_ids": sample_ids.len(),
            "unique_precursor_formulas": precursor_formulas.len(),
            "unique_precursor_pairs": precursor_pairs.len(),
            "synthesis_objects": synthesis_objects,
            "physical_failures": physical_failures,
            "samples_with_recorded_heating_temperature": samples_with_heating_temperature,
            "category_counts": expected_categories,
            "scans": scan_count,
            "scans_by_status": scan_status_counts,
            "refinement_cases": refinement_case_count,
            "samples_with_active_refinement": samples_with_active_refinement,
            "active_refinement_quality": expected_quality,
            "active_refinement_acceptance": expected_acceptance
        },
        "cohorts": {
            "all_samples": all_sample_ids,
            "classified": category_ids.iter().filter(|(category, _)| category.as_str() != "unclassified")
                .flat_map(|(_, ids)| ids.iter().cloned()).collect::<Vec<_>>(),
            "classified_temperature_no_physical_failure": classified_temperature_no_failure,
            "temperature_and_accepted_quality_1_or_2": accepted_quality_1_or_2_cohort,
            "category_ids": category_ids
        },
        "exclusions": {
            "physical_failure_ids": physical_failure_ids,
            "unclassified_ids": unclassified_ids,
            "missing_synthesis_ids": missing_synthesis_ids,
            "missing_temperature_ids": missing_temperature_ids,
            "no_active_refinement_ids": no_active_refinement_ids,
            "rejected_active_refinement_ids": rejected_active_refinement_ids,
            "missing_quality_ids": missing_quality_ids,
            "quality_3_ids": quality_3_ids
        },
        "publication_reconciliation": {
            "paper_analyzable_outcomes": paper_total,
            "ledger_categorized_outcomes": ledger_classified,
            "ledger_minus_paper_category_counts": paper_delta,
            "status": "unresolved; no rows have been dropped to force agreement"
        },
        "claim_ceiling": CLAIM_CEILING
    }))
}

fn usage(program: &str) {
    eprintln!(
        "Usage: {program} --schema <manifest-schema.json> --manifest <source-manifest.json> --ledger <ledger.json> [--out <receipt.json>]"
    );
}

fn read(path: &PathBuf) -> Vec<u8> {
    fs::read(path).unwrap_or_else(|error| {
        eprintln!("could not read {}: {error}", path.display());
        process::exit(1);
    })
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut schema_path = None;
    let mut manifest_path = None;
    let mut ledger_path = None;
    let mut output_path = None;
    let mut index = 1;

    while index < args.len() {
        match args[index].as_str() {
            "--schema" | "--manifest" | "--ledger" | "--out" => {
                if index + 1 >= args.len() {
                    usage(&args[0]);
                    process::exit(2);
                }
                let path = PathBuf::from(&args[index + 1]);
                match args[index].as_str() {
                    "--schema" => schema_path = Some(path),
                    "--manifest" => manifest_path = Some(path),
                    "--ledger" => ledger_path = Some(path),
                    "--out" => output_path = Some(path),
                    _ => unreachable!(),
                }
                index += 2;
            }
            "--help" | "-h" => {
                usage(&args[0]);
                return;
            }
            other => {
                eprintln!("unknown argument {other}");
                usage(&args[0]);
                process::exit(2);
            }
        }
    }

    let (Some(schema_path), Some(manifest_path), Some(ledger_path)) =
        (schema_path, manifest_path, ledger_path)
    else {
        usage(&args[0]);
        process::exit(2);
    };

    let schema_bytes = read(&schema_path);
    let manifest_bytes = read(&manifest_path);
    let ledger_bytes = read(&ledger_path);
    let receipt = audit(&ledger_bytes, &schema_bytes, &manifest_bytes).unwrap_or_else(|error| {
        eprintln!("SC-PRECURSOR-GENOME-FAIL: {error}");
        process::exit(1);
    });
    let rendered =
        serde_json::to_string_pretty(&receipt).expect("JSON receipt serialization cannot fail");

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

    fn schema_bytes() -> Vec<u8> {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        fs::read(root.join("data/schemas/precursor_genome_source_manifest_v0_1.schema.json"))
            .unwrap()
    }

    fn synthetic_inputs() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let ledger = json!({
            "repository_name": "synthetic",
            "samples": [{
                "sample_id": "SYN-001",
                "precursors": [{"formula": "A2O3"}, {"formula": "BO2"}],
                "synthesis": {"heating_temperature": 500.0},
                "physical_failure": null,
                "outcome": {"reaction_category": "transformed"},
                "active_scan_index": 0,
                "characterization": {
                    "xrd": {"scans": [{
                        "status": "valid",
                        "active_case_index": 0,
                        "refinement_cases": [{
                            "verification": {
                                "human_quality_score": 1,
                                "is_accepted": true
                            }
                        }]
                    }]}
                }
            }]
        });
        let ledger_bytes = serde_json::to_vec(&ledger).unwrap();
        let manifest = json!({
            "contract": CONTRACT,
            "source_repository": "lauren-walters/precursor-genome",
            "source_commit": "0000000000000000000000000000000000000000",
            "ledger_path": "ledger_precursor_genome.json",
            "ledger_git_blob_sha1": git_blob_sha1(&ledger_bytes),
            "license": "CC-BY-4.0",
            "zenodo_raw_artifacts_doi": "10.5281/zenodo.21285546",
            "measurement_units": {
                "heating_temperature": {
                    "source_field": "synthesis.heating_temperature",
                    "unit": "degree_Celsius",
                    "provenance": "Synthetic test authority: arXiv:2607.09903v1 SampleEntry field description."
                },
                "heating_time": {
                    "source_field": "synthesis.heating_time",
                    "unit": "minute",
                    "provenance": "Synthetic test authority: arXiv:2607.09903v1 SampleEntry field description."
                },
                "provenance_url": "https://arxiv.org/abs/2607.09903"
            },
            "expected_profile": {
                "samples": 1,
                "unique_sample_ids": 1,
                "unique_precursor_formulas": 2,
                "unique_precursor_pairs": 1,
                "synthesis_objects": 1,
                "physical_failures": 0,
                "samples_with_recorded_heating_temperature": 1,
                "category_counts": {
                    "transformed": 1,
                    "unreacted": 0,
                    "partially_reacted": 0,
                    "completely_reacted": 0,
                    "unclassified": 0
                },
                "scans": 1,
                "scans_by_status": {"valid": 1},
                "refinement_cases": 1,
                "samples_with_active_refinement": 1,
                "active_refinement_quality": {
                    "score_1": 1,
                    "score_2": 0,
                    "score_3": 0,
                    "missing": 0,
                    "no_active_refinement": 0
                },
                "active_refinement_acceptance": {
                    "accepted": 1,
                    "rejected": 0,
                    "missing": 0
                },
                "classified_with_temperature_and_no_physical_failure": 1,
                "cohort_temperature_and_accepted_quality_1_or_2": 1
            },
            "publication_comparison": {
                "source": "synthetic",
                "paper_analyzable_outcomes": 1,
                "paper_category_counts": {
                    "transformed": 1,
                    "unreacted": 0,
                    "partially_reacted": 0,
                    "completely_reacted": 0
                },
                "reconciliation_status": "unresolved; preserve snapshot differences and do not silently select/drop records"
            },
            "claim_ceiling": CLAIM_CEILING
        });
        (
            ledger_bytes,
            schema_bytes(),
            serde_json::to_vec(&manifest).unwrap(),
        )
    }

    #[test]
    fn git_blob_sha1_matches_the_empty_git_object() {
        assert_eq!(
            git_blob_sha1(&[]),
            "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"
        );
    }

    #[test]
    fn duplicate_unordered_precursor_pairs_fail_closed() {
        let (ledger, schema, manifest) = synthetic_inputs();
        let mut data: Value = serde_json::from_slice(&ledger).unwrap();
        let mut duplicate = data["samples"][0].clone();
        duplicate["sample_id"] = json!("SYN-002");
        data["samples"].as_array_mut().unwrap().push(duplicate);
        let changed_ledger = serde_json::to_vec(&data).unwrap();
        let mut manifest_data: Value = serde_json::from_slice(&manifest).unwrap();
        manifest_data["ledger_git_blob_sha1"] = json!(git_blob_sha1(&changed_ledger));
        let error = audit(
            &changed_ledger,
            &schema,
            &serde_json::to_vec(&manifest_data).unwrap(),
        )
        .unwrap_err();
        assert!(error.contains("duplicate precursor pair"));
    }

    #[test]
    fn synthetic_ledger_is_scored_and_transformed_is_preserved() {
        let (ledger, schema, manifest) = synthetic_inputs();
        let receipt = audit(&ledger, &schema, &manifest).unwrap();
        assert_eq!(receipt["status"], "precursor_genome_source_audit_pass");
        assert_eq!(receipt["counts"]["category_counts"]["transformed"], 1);
        assert_eq!(
            receipt["cohorts"]["classified_temperature_no_physical_failure"][0],
            "SYN-001"
        );
        assert_eq!(
            receipt["cohorts"]["temperature_and_accepted_quality_1_or_2"][0],
            "SYN-001"
        );
    }

    #[test]
    fn changed_source_bytes_fail_git_blob_identity() {
        let (mut ledger, schema, manifest) = synthetic_inputs();
        ledger.push(b' ');
        let error = audit(&ledger, &schema, &manifest).unwrap_err();
        assert!(error.contains("source ledger identity mismatch"));
    }

    #[test]
    fn duplicate_sample_ids_fail_closed() {
        let (ledger, schema, manifest) = synthetic_inputs();
        let mut data: Value = serde_json::from_slice(&ledger).unwrap();
        let duplicate = data["samples"][0].clone();
        data["samples"].as_array_mut().unwrap().push(duplicate);
        let changed_ledger = serde_json::to_vec(&data).unwrap();
        let mut manifest_data: Value = serde_json::from_slice(&manifest).unwrap();
        manifest_data["ledger_git_blob_sha1"] = json!(git_blob_sha1(&changed_ledger));
        let error = audit(
            &changed_ledger,
            &schema,
            &serde_json::to_vec(&manifest_data).unwrap(),
        )
        .unwrap_err();
        assert!(error.contains("duplicate sample_id"));
    }

    #[test]
    fn unclassified_physical_failure_is_not_a_negative_reaction_label() {
        let (ledger, schema, manifest) = synthetic_inputs();
        let mut data: Value = serde_json::from_slice(&ledger).unwrap();
        data["samples"][0]["outcome"]["reaction_category"] = Value::Null;
        data["samples"][0]["physical_failure"] = json!({"observation": "uncharacterizable"});
        let changed_ledger = serde_json::to_vec(&data).unwrap();
        let mut manifest_data: Value = serde_json::from_slice(&manifest).unwrap();
        manifest_data["ledger_git_blob_sha1"] = json!(git_blob_sha1(&changed_ledger));
        manifest_data["expected_profile"]["physical_failures"] = json!(1);
        manifest_data["expected_profile"]["category_counts"]["transformed"] = json!(0);
        manifest_data["expected_profile"]["category_counts"]["unclassified"] = json!(1);
        manifest_data["expected_profile"]["classified_with_temperature_and_no_physical_failure"] =
            json!(0);
        manifest_data["expected_profile"]["cohort_temperature_and_accepted_quality_1_or_2"] =
            json!(0);
        manifest_data["publication_comparison"]["paper_category_counts"]["transformed"] = json!(0);
        let receipt = audit(
            &changed_ledger,
            &schema,
            &serde_json::to_vec(&manifest_data).unwrap(),
        )
        .unwrap();
        assert_eq!(receipt["counts"]["category_counts"]["unclassified"], 1);
        assert_eq!(receipt["exclusions"]["physical_failure_ids"][0], "SYN-001");
        assert_eq!(receipt["exclusions"]["unclassified_ids"][0], "SYN-001");
        assert_eq!(
            receipt["cohorts"]["classified_temperature_no_physical_failure"]
                .as_array()
                .unwrap()
                .len(),
            0
        );
    }
}
