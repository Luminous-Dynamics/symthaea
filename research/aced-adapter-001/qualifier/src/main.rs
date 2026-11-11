use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::process::ExitCode;

const BENCHMARK_ID: &str = "ACED-REPAIR-ADAPTER-001";
const SCHEMA_VERSION: &str = "aced-repair-upstream-source-manifest-v0.1";
const UPSTREAM_REPO: &str = "https://github.com/Rose-STL-Lab/ACED-Bench";
const UPSTREAM_COMMIT: &str = "998dc692bfcf5574e5319d67f1cea9c170569d80";

#[derive(Debug)]
struct IndexReport {
    total: usize,
    domains: BTreeMap<String, usize>,
    archetypes: BTreeMap<String, usize>,
    rows: Vec<Value>,
}

#[derive(Debug)]
struct QualificationReport {
    test: IndexReport,
    eval: IndexReport,
    checksum_entries: usize,
    exact_eval_subset: usize,
}

fn text<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    value.get(key)?.as_str().filter(|item| !item.trim().is_empty())
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn parse_counts(value: &Value, key: &str) -> Result<BTreeMap<String, usize>, String> {
    let object = value
        .get(key)
        .and_then(Value::as_object)
        .ok_or_else(|| format!("{key} must be an object of expected counts"))?;
    let mut counts = BTreeMap::new();
    for (name, count) in object {
        let count = count
            .as_u64()
            .and_then(|value| usize::try_from(value).ok())
            .ok_or_else(|| format!("{key}.{name} must be a non-negative integer"))?;
        if count == 0 {
            return Err(format!("{key}.{name} must be positive"));
        }
        counts.insert(name.clone(), count);
    }
    Ok(counts)
}

fn validate_index(
    index: &Value,
    label: &str,
    expected_total: usize,
    expected_domains: &BTreeMap<String, usize>,
    expected_archetypes: &BTreeMap<String, usize>,
) -> Result<IndexReport, Vec<String>> {
    let mut errors = Vec::new();
    let Some(rows) = index.as_array() else {
        return Err(vec![format!("{label} index must be a JSON array")]);
    };
    if rows.len() != expected_total {
        errors.push(format!(
            "{label} index expected {expected_total} rows, received {}",
            rows.len()
        ));
    }

    let mut files = BTreeSet::new();
    let mut seeds = BTreeSet::new();
    let mut domains = BTreeMap::<String, usize>::new();
    let mut archetypes = BTreeMap::<String, usize>::new();

    for (position, row) in rows.iter().enumerate() {
        if row.as_object().map(serde_json::Map::len) != Some(4) {
            errors.push(format!(
                "{label}[{position}] must contain exactly file, domain, archetype and seed"
            ));
            continue;
        }

        let Some(file) = text(row, "file") else {
            errors.push(format!("{label}[{position}] has no non-empty file name"));
            continue;
        };
        if !file.starts_with("world_")
            || !file.ends_with(".json.gz")
            || file.contains('/')
            || file.contains('\\')
            || file.contains("..")
            || file.trim() != file
        {
            errors.push(format!(
                "{label}[{position}] has unsafe or unexpected world filename {file:?}"
            ));
        }
        if !files.insert(file.to_string()) {
            errors.push(format!("{label}[{position}] duplicates world file {file}"));
        }

        let Some(domain) = text(row, "domain") else {
            errors.push(format!("{label}[{position}] has no domain"));
            continue;
        };
        let Some(archetype) = text(row, "archetype") else {
            errors.push(format!("{label}[{position}] has no archetype"));
            continue;
        };
        let Some(seed) = row["seed"].as_u64() else {
            errors.push(format!("{label}[{position}] seed must be an unsigned integer"));
            continue;
        };
        if seed == 0 {
            errors.push(format!("{label}[{position}] seed must be positive"));
        }
        if !seeds.insert(seed) {
            errors.push(format!("{label}[{position}] duplicates seed {seed}"));
        }

        *domains.entry(domain.to_string()).or_default() += 1;
        *archetypes.entry(archetype.to_string()).or_default() += 1;
    }

    if &domains != expected_domains {
        errors.push(format!(
            "{label} domain distribution mismatch: expected {expected_domains:?}, got {domains:?}"
        ));
    }
    if &archetypes != expected_archetypes {
        errors.push(format!(
            "{label} archetype distribution mismatch: expected {expected_archetypes:?}, got {archetypes:?}"
        ));
    }

    if errors.is_empty() {
        Ok(IndexReport {
            total: rows.len(),
            domains,
            archetypes,
            rows: rows.clone(),
        })
    } else {
        errors.sort();
        errors.dedup();
        Err(errors)
    }
}

fn validate_eval_subset(test: &IndexReport, eval: &IndexReport) -> Result<usize, Vec<String>> {
    let mut errors = Vec::new();
    let test_by_file = test
        .rows
        .iter()
        .filter_map(|row| text(row, "file").map(|file| (file.to_string(), row)))
        .collect::<BTreeMap<_, _>>();

    for row in &eval.rows {
        let Some(file) = text(row, "file") else {
            errors.push("evaluation row has no file identity".into());
            continue;
        };
        match test_by_file.get(file) {
            Some(test_row) if *test_row == row => {}
            Some(_) => errors.push(format!(
                "evaluation metadata for {file} differs from its selected test row"
            )),
            None => errors.push(format!(
                "evaluation world {file} is not present in the pinned test index"
            )),
        }
    }

    if errors.is_empty() {
        Ok(eval.total)
    } else {
        errors.sort();
        errors.dedup();
        Err(errors)
    }
}

fn verify_checksum_manifest(
    source: &str,
    test: &IndexReport,
    eval: &IndexReport,
) -> Result<usize, Vec<String>> {
    let mut errors = Vec::new();
    let mut entries = BTreeMap::<String, String>::new();

    for (line_no, line) in source.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let Some((digest, path)) = line.split_once("  ") else {
            errors.push(format!(
                "SHA256SUMS line {} has no standard two-space separator",
                line_no + 1
            ));
            continue;
        };
        let path = path.trim();
        if !is_sha256(digest) {
            errors.push(format!(
                "SHA256SUMS line {} has a malformed lowercase digest",
                line_no + 1
            ));
        }
        if path.is_empty()
            || path.starts_with('/')
            || path.contains('\\')
            || path.split('/').any(|component| component == ".." || component.is_empty())
        {
            errors.push(format!(
                "SHA256SUMS line {} has an unsafe relative path",
                line_no + 1
            ));
        }
        if entries.insert(path.to_string(), digest.to_string()).is_some() {
            errors.push(format!("SHA256SUMS duplicates path {path}"));
        }
    }

    for row in &test.rows {
        if let Some(file) = text(row, "file") {
            let path = format!("repair/test/{file}");
            if !entries.contains_key(&path) {
                errors.push(format!("upstream checksum manifest omits {path}"));
            }
        }
    }
    for row in &eval.rows {
        if let Some(file) = text(row, "file") {
            let path = format!("repair/eval/{file}");
            if !entries.contains_key(&path) {
                errors.push(format!("upstream checksum manifest omits {path}"));
            }
        }
    }

    if errors.is_empty() {
        Ok(entries.len())
    } else {
        errors.sort();
        errors.dedup();
        Err(errors)
    }
}

fn qualify(
    test_index: &Value,
    eval_index: &Value,
    manifest: &Value,
    test_bytes: &[u8],
    eval_bytes: &[u8],
    checksum_bytes: &[u8],
) -> Result<QualificationReport, Vec<String>> {
    let mut errors = Vec::new();
    if text(manifest, "schema_version") != Some(SCHEMA_VERSION)
        || text(manifest, "benchmark_id") != Some(BENCHMARK_ID)
        || text(manifest, "status") != Some("pinned_upstream_inventory_only_no_model_trials")
    {
        errors.push("source manifest identity or status mismatch".into());
    }

    let upstream = &manifest["upstream"];
    if text(upstream, "repository") != Some(UPSTREAM_REPO)
        || text(upstream, "commit") != Some(UPSTREAM_COMMIT)
        || text(upstream, "license") != Some("MIT")
        || text(upstream, "paper_revision") != Some("v2")
        || text(upstream, "paper_revision_date") != Some("2026-11-08")
    {
        errors.push("upstream source identity, commit, license or paper revision mismatch".into());
    }

    let artifacts = &manifest["artifacts"];
    let expected_test_hash = text(artifacts, "test_index_sha256").unwrap_or("");
    let expected_eval_hash = text(artifacts, "eval_index_sha256").unwrap_or("");
    let expected_checksums_hash = text(artifacts, "data_sha256_manifest_sha256").unwrap_or("");
    if !is_sha256(expected_test_hash) || sha256(test_bytes) != expected_test_hash {
        errors.push("test index bytes do not match the pinned upstream SHA-256".into());
    }
    if !is_sha256(expected_eval_hash) || sha256(eval_bytes) != expected_eval_hash {
        errors.push("evaluation index bytes do not match the pinned upstream SHA-256".into());
    }
    if !is_sha256(expected_checksums_hash) || sha256(checksum_bytes) != expected_checksums_hash {
        errors.push("upstream data/SHA256SUMS bytes do not match the pinned SHA-256".into());
    }

    if text(artifacts, "test_index_path") != Some("data/repair/test/index.json")
        || text(artifacts, "eval_index_path") != Some("data/repair/eval/index.json")
        || text(artifacts, "data_sha256_manifest_path") != Some("data/SHA256SUMS")
        || manifest["split_contract"]["eval_is_subset_of_test"] != true
    {
        errors.push("upstream artifact paths or evaluation subset contract mismatch".into());
    }

    let distribution = &manifest["expected_distribution"];
    let test_expected = &distribution["test"];
    let eval_expected = &distribution["eval"];
    let test_total = test_expected["total"].as_u64().unwrap_or(0) as usize;
    let eval_total = eval_expected["total"].as_u64().unwrap_or(0) as usize;
    let test_domains = parse_counts(test_expected, "domains");
    let test_archetypes = parse_counts(test_expected, "archetypes");
    let eval_domains = parse_counts(eval_expected, "domains");
    let eval_archetypes = parse_counts(eval_expected, "archetypes");

    let test_report = match (test_domains, test_archetypes) {
        (Ok(domains), Ok(archetypes)) => {
            match validate_index(test_index, "test", test_total, &domains, &archetypes) {
                Ok(report) => Some(report),
                Err(found) => {
                    errors.extend(found);
                    None
                }
            }
        }
        (Err(a), Err(b)) => {
            errors.push(a);
            errors.push(b);
            None
        }
        (Err(error), _) | (_, Err(error)) => {
            errors.push(error);
            None
        }
    };

    let eval_report = match (eval_domains, eval_archetypes) {
        (Ok(domains), Ok(archetypes)) => {
            match validate_index(eval_index, "eval", eval_total, &domains, &archetypes) {
                Ok(report) => Some(report),
                Err(found) => {
                    errors.extend(found);
                    None
                }
            }
        }
        (Err(a), Err(b)) => {
            errors.push(a);
            errors.push(b);
            None
        }
        (Err(error), _) | (_, Err(error)) => {
            errors.push(error);
            None
        }
    };

    let mut exact_eval_subset = 0;
    let mut checksum_entries = 0;
    if let (Some(test), Some(eval)) = (&test_report, &eval_report) {
        match validate_eval_subset(test, eval) {
            Ok(count) => exact_eval_subset = count,
            Err(found) => errors.extend(found),
        }
        match std::str::from_utf8(checksum_bytes)
            .map_err(|error| vec![format!("SHA256SUMS is not UTF-8: {error}")])
            .and_then(|text| verify_checksum_manifest(text, test, eval))
        {
            Ok(count) => checksum_entries = count,
            Err(found) => errors.extend(found),
        }
    }

    if errors.is_empty() {
        Ok(QualificationReport {
            test: test_report.expect("test report exists when no errors"),
            eval: eval_report.expect("eval report exists when no errors"),
            checksum_entries,
            exact_eval_subset,
        })
    } else {
        errors.sort();
        errors.dedup();
        Err(errors)
    }
}

fn read_bytes(path: &str) -> Result<Vec<u8>, String> {
    fs::read(path).map_err(|error| format!("{path}: {error}"))
}

fn read_json(path: &str) -> Result<(Value, Vec<u8>), String> {
    let bytes = read_bytes(path)?;
    let value = serde_json::from_slice(&bytes)
        .map_err(|error| format!("{path}: invalid JSON: {error}"))?;
    Ok((value, bytes))
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    if args.len() != 5 {
        eprintln!(
            "usage: aced-repair-index-qualifier <test-index.json> <eval-index.json> <source-manifest.json> <SHA256SUMS>"
        );
        return ExitCode::from(2);
    }

    let (test, test_bytes) = match read_json(&args[1]) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("{error}");
            return ExitCode::FAILURE;
        }
    };
    let (eval, eval_bytes) = match read_json(&args[2]) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("{error}");
            return ExitCode::FAILURE;
        }
    };
    let (manifest, _) = match read_json(&args[3]) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("{error}");
            return ExitCode::FAILURE;
        }
    };
    let checksum_bytes = match read_bytes(&args[4]) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("{error}");
            return ExitCode::FAILURE;
        }
    };

    match qualify(
        &test,
        &eval,
        &manifest,
        &test_bytes,
        &eval_bytes,
        &checksum_bytes,
    ) {
        Ok(report) => {
            println!(
                "ACED-REPAIR UPSTREAM INVENTORY QUALIFIED: test={} eval={} exact_eval_subset={} archetypes={} test_domains={} eval_domains={} checksum_entries={}",
                report.test.total,
                report.eval.total,
                report.exact_eval_subset,
                report.test.archetypes.len(),
                report.test.domains.len(),
                report.eval.domains.len(),
                report.checksum_entries
            );
            println!(
                "CLAIM CEILING: metadata, index hashes, split identity and checksum-list coverage only; no compressed world payload was verified and no model was run."
            );
            ExitCode::SUCCESS
        }
        Err(errors) => {
            eprintln!(
                "ACED-REPAIR UPSTREAM INVENTORY QUALIFICATION FAILED ({} findings):",
                errors.len()
            );
            for error in errors {
                eprintln!(" - {error}");
            }
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_INDEX: &str = include_str!("../../upstream/repair_test_index.json");
    const EVAL_INDEX: &str = include_str!("../../upstream/repair_eval_index.json");
    const SOURCE_MANIFEST: &str = include_str!("../../ACED_REPAIR_SOURCE_MANIFEST_V0_1.json");
    const CHECKSUMS: &str = include_str!("../../upstream/SHA256SUMS");

    fn parsed(source: &str) -> Value {
        serde_json::from_str(source).expect("checked-in JSON must parse")
    }

    fn expected_maps(manifest: &Value, split: &str) -> (usize, BTreeMap<String, usize>, BTreeMap<String, usize>) {
        let expected = &manifest["expected_distribution"][split];
        (
            expected["total"].as_u64().unwrap() as usize,
            parse_counts(expected, "domains").unwrap(),
            parse_counts(expected, "archetypes").unwrap(),
        )
    }

    #[test]
    fn pinned_upstream_inventory_and_eval_subset_qualify() {
        let test = parsed(TEST_INDEX);
        let eval = parsed(EVAL_INDEX);
        let manifest = parsed(SOURCE_MANIFEST);
        let report = qualify(
            &test,
            &eval,
            &manifest,
            TEST_INDEX.as_bytes(),
            EVAL_INDEX.as_bytes(),
            CHECKSUMS.as_bytes(),
        )
        .expect("official pinned metadata qualifies");
        assert_eq!(report.test.total, 540);
        assert_eq!(report.eval.total, 54);
        assert_eq!(report.exact_eval_subset, 54);
        assert_eq!(report.test.archetypes.len(), 9);
        assert_eq!(report.test.domains.len(), 10);
    }

    #[test]
    fn altered_index_artifact_bytes_are_rejected() {
        let test = parsed(TEST_INDEX);
        let eval = parsed(EVAL_INDEX);
        let manifest = parsed(SOURCE_MANIFEST);
        let mut altered = TEST_INDEX.as_bytes().to_vec();
        altered[0] = b' ';
        assert!(qualify(
            &test,
            &eval,
            &manifest,
            &altered,
            EVAL_INDEX.as_bytes(),
            CHECKSUMS.as_bytes()
        )
        .is_err());
    }

    #[test]
    fn duplicate_world_seed_is_rejected() {
        let mut index = parsed(TEST_INDEX);
        let manifest = parsed(SOURCE_MANIFEST);
        let (total, domains, archetypes) = expected_maps(&manifest, "test");
        let first_seed = index[0]["seed"].clone();
        index[1]["seed"] = first_seed;
        assert!(validate_index(&index, "test", total, &domains, &archetypes).is_err());
    }

    #[test]
    fn path_traversal_world_filename_is_rejected() {
        let mut index = parsed(TEST_INDEX);
        let manifest = parsed(SOURCE_MANIFEST);
        let (total, domains, archetypes) = expected_maps(&manifest, "test");
        index[0]["file"] = Value::String("../hidden-world.json.gz".into());
        assert!(validate_index(&index, "test", total, &domains, &archetypes).is_err());
    }

    #[test]
    fn evaluation_world_must_be_an_exact_test_metadata_record() {
        let test = parsed(TEST_INDEX);
        let mut eval = parsed(EVAL_INDEX);
        eval[0]["seed"] = Value::from(9_999_999);
        let manifest = parsed(SOURCE_MANIFEST);
        let (test_total, test_domains, test_archetypes) = expected_maps(&manifest, "test");
        let (eval_total, eval_domains, eval_archetypes) = expected_maps(&manifest, "eval");
        let test_report = validate_index(&test, "test", test_total, &test_domains, &test_archetypes).unwrap();
        let eval_report = validate_index(&eval, "eval", eval_total, &eval_domains, &eval_archetypes).unwrap();
        assert!(validate_eval_subset(&test_report, &eval_report).is_err());
    }

    #[test]
    fn domain_or_archetype_imbalance_is_rejected() {
        let mut index = parsed(TEST_INDEX);
        let manifest = parsed(SOURCE_MANIFEST);
        let (total, domains, archetypes) = expected_maps(&manifest, "test");
        index[0]["archetype"] = Value::String("unknown_archetype".into());
        assert!(validate_index(&index, "test", total, &domains, &archetypes).is_err());
    }

    #[test]
    fn checksum_manifest_must_cover_selected_world_payloads() {
        let test = parsed(TEST_INDEX);
        let eval = parsed(EVAL_INDEX);
        let test_report = {
            let manifest = parsed(SOURCE_MANIFEST);
            let (total, domains, archetypes) = expected_maps(&manifest, "test");
            validate_index(&test, "test", total, &domains, &archetypes).unwrap()
        };
        let eval_report = {
            let manifest = parsed(SOURCE_MANIFEST);
            let (total, domains, archetypes) = expected_maps(&manifest, "eval");
            validate_index(&eval, "eval", total, &domains, &archetypes).unwrap()
        };
        let mut altered = CHECKSUMS.to_string();
        let first = test[0]["file"].as_str().unwrap();
        altered = altered.replace(
            &format!("repair/test/{first}"),
            "repair/test/missing-world.json.gz"
        );
        assert!(verify_checksum_manifest(&altered, &test_report, &eval_report).is_err());
    }
}
