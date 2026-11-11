use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

const UPSTREAM_REPOSITORY: &str = "https://github.com/Rose-STL-Lab/ACED-Bench";
const UPSTREAM_COMMIT: &str = "998dc692bfcf5574e5319d67f1cea9c170569d80";
const TRACKS: [&str; 4] = ["core_struct", "core_decision", "guess_struct", "guess_decision"];
const EXPECTED_COUNTS: [usize; 4] = [10, 60, 30, 48];
const KNOWN_DECISION_FAMILIES: [&str; 8] = [
    "invalid_premise",
    "mediator_structure_direct_and_mediated",
    "mediator_structure_mediated_only",
    "mediator_structure_not_mediator",
    "mediator_structure_which_mediator",
    "safety_constrained",
    "satisficing",
    "subgroup_robust",
];

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct QueryInventoryReport {
    pub total_track_entries: usize,
    pub unique_payload_sha256_count: usize,
    pub core_struct_guess_struct_exact_content_overlap: usize,
    pub core_decision_guess_decision_exact_content_overlap: usize,
    pub core_decision_guess_decision_filename_collision_count: usize,
    pub checksum_entries: usize,
    pub per_track_counts: BTreeMap<String, usize>,
}

#[derive(Clone, Debug)]
struct TrackReport {
    rows: Vec<Value>,
    domains: BTreeMap<String, usize>,
    families: BTreeMap<String, usize>,
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn text<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    value.get(key)?.as_str().filter(|item| !item.trim().is_empty())
}

fn is_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

fn exact_fields(value: &Value, expected: &[&str]) -> bool {
    let Some(object) = value.as_object() else {
        return false;
    };
    object.len() == expected.len() && expected.iter().all(|key| object.contains_key(*key))
}

fn expected_counts(manifest: &Value, track: &str, field: &str) -> Result<BTreeMap<String, usize>, String> {
    let object = manifest["distributions"][track][field]
        .as_object()
        .ok_or_else(|| format!("manifest.distributions.{track}.{field} must be an object"))?;
    let mut counts = BTreeMap::new();
    for (key, value) in object {
        let count = value
            .as_u64()
            .and_then(|n| usize::try_from(n).ok())
            .filter(|n| *n > 0)
            .ok_or_else(|| format!("manifest count {track}.{field}.{key} must be positive"))?;
        counts.insert(key.clone(), count);
    }
    Ok(counts)
}

fn validate_track(track: &str, rows: &Value, manifest: &Value) -> Result<TrackReport, Vec<String>> {
    let mut errors = Vec::new();
    let index = TRACKS
        .iter()
        .position(|candidate| *candidate == track)
        .ok_or_else(|| vec![format!("unknown ACED-Query track {track}")])?;
    let expected_total = EXPECTED_COUNTS[index];
    let Some(items) = rows.as_array() else {
        return Err(vec![format!("track {track} must be an array")]);
    };
    if items.len() != expected_total {
        errors.push(format!(
            "track {track} expected {expected_total} rows, received {}",
            items.len()
        ));
    }

    let mut files = BTreeSet::new();
    let mut seeds = BTreeSet::new();
    let mut domains = BTreeMap::<String, usize>::new();
    let mut families = BTreeMap::<String, usize>::new();

    for (position, row) in items.iter().enumerate() {
        if !exact_fields(
            row,
            &[
                "track",
                "file",
                "path",
                "git_blob_sha",
                "bytes",
                "sha256",
                "domain",
                "task_family",
                "seed",
                "variables_n",
                "edge_k",
                "sample_n",
            ],
        ) {
            errors.push(format!(
                "{track}[{position}] does not match the closed twelve-field metadata contract"
            ));
            continue;
        }
        if text(row, "track") != Some(track) {
            errors.push(format!("{track}[{position}] carries an inconsistent track identity"));
        }

        let Some(file) = text(row, "file") else {
            errors.push(format!("{track}[{position}] has no file name"));
            continue;
        };
        let expected_path = format!("query/{track}/{file}");
        if text(row, "path") != Some(expected_path.as_str())
            || file.contains('/')
            || file.contains('\\')
            || file.contains("..")
            || !file.ends_with(".json")
        {
            errors.push(format!("{track}[{position}] has an unsafe or mismatched source path"));
        }
        if !files.insert(file.to_string()) {
            errors.push(format!("{track}[{position}] duplicates file identity {file}"));
        }
        if !text(row, "git_blob_sha").is_some_and(|value| is_hex(value, 40)) {
            errors.push(format!("{track}[{position}] has an invalid Git blob SHA-1"));
        }
        if !text(row, "sha256").is_some_and(|value| is_hex(value, 64)) {
            errors.push(format!("{track}[{position}] has an invalid payload SHA-256"));
        }
        if row["bytes"].as_u64().is_none_or(|bytes| bytes == 0) {
            errors.push(format!("{track}[{position}] has an invalid payload byte size"));
        }
        let Some(domain) = text(row, "domain") else {
            errors.push(format!("{track}[{position}] has no domain"));
            continue;
        };
        let Some(family) = text(row, "task_family") else {
            errors.push(format!("{track}[{position}] has no task family"));
            continue;
        };
        let Some(seed) = row["seed"].as_u64().filter(|seed| *seed > 0) else {
            errors.push(format!("{track}[{position}] has an invalid seed"));
            continue;
        };
        if !seeds.insert(seed) {
            errors.push(format!("{track}[{position}] duplicates seed {seed}"));
        }

        if track.ends_with("struct") {
            if family != "random_graph"
                || row["variables_n"].as_u64().is_none_or(|value| value < 2)
                || row["edge_k"].as_u64().is_none()
                || !row["sample_n"].is_null()
            {
                errors.push(format!(
                    "{track}[{position}] violates the structural-world metadata contract"
                ));
            }
        } else if !KNOWN_DECISION_FAMILIES.contains(&family)
            || !row["variables_n"].is_null()
            || !row["edge_k"].is_null()
            || row["sample_n"].as_u64().is_none_or(|value| value == 0)
        {
            errors.push(format!(
                "{track}[{position}] violates the decision-world metadata contract"
            ));
        }

        *domains.entry(domain.to_string()).or_default() += 1;
        *families.entry(family.to_string()).or_default() += 1;
    }

    let expected_domains = expected_counts(manifest, track, "domains")
        .map_err(|error| vec![error])?;
    let expected_families = expected_counts(manifest, track, "families")
        .map_err(|error| vec![error])?;
    if domains != expected_domains {
        errors.push(format!(
            "track {track} domain distribution mismatch: expected {expected_domains:?}, got {domains:?}"
        ));
    }
    if families != expected_families {
        errors.push(format!(
            "track {track} family distribution mismatch: expected {expected_families:?}, got {families:?}"
        ));
    }

    if errors.is_empty() {
        Ok(TrackReport {
            rows: items.clone(),
            domains,
            families,
        })
    } else {
        errors.sort();
        errors.dedup();
        Err(errors)
    }
}

fn parse_checksums(source: &str) -> Result<BTreeMap<String, String>, Vec<String>> {
    let mut errors = Vec::new();
    let mut entries = BTreeMap::new();
    for (line_no, line) in source.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let Some((digest, path)) = line.split_once("  ") else {
            errors.push(format!("SHA256SUMS line {} lacks the standard separator", line_no + 1));
            continue;
        };
        if !is_hex(digest, 64) {
            errors.push(format!("SHA256SUMS line {} has an invalid digest", line_no + 1));
        }
        if path.is_empty()
            || path.starts_with('/')
            || path.contains('\\')
            || path.split('/').any(|part| part.is_empty() || part == "..")
        {
            errors.push(format!("SHA256SUMS line {} has an unsafe path", line_no + 1));
        }
        if entries.insert(path.to_string(), digest.to_string()).is_some() {
            errors.push(format!("SHA256SUMS contains duplicate path {path}"));
        }
    }
    if errors.is_empty() {
        Ok(entries)
    } else {
        errors.sort();
        errors.dedup();
        Err(errors)
    }
}

fn overlap_count(left: &[Value], right: &[Value], same_payload: bool) -> usize {
    left.iter()
        .filter(|a| {
            right.iter().any(|b| {
                text(a, "file") == text(b, "file")
                    && if same_payload {
                        text(a, "git_blob_sha") == text(b, "git_blob_sha")
                            && text(a, "sha256") == text(b, "sha256")
                    } else {
                        true
                    }
            })
        })
        .count()
}

pub fn qualify_query_inventory(
    inventory: &Value,
    manifest: &Value,
    inventory_bytes: &[u8],
    checksum_bytes: &[u8],
) -> Result<QueryInventoryReport, Vec<String>> {
    let mut errors = Vec::new();
    if !exact_fields(
        manifest,
        &[
            "schema_version",
            "benchmark_id",
            "status",
            "upstream",
            "artifacts",
            "track_counts",
            "distributions",
            "split_relationships",
            "candidate_data_boundary",
            "claim_ceiling",
        ],
    )
        || text(manifest, "schema_version") != Some("aced-query-source-manifest-v0.1")
        || text(manifest, "benchmark_id") != Some("ACED-QUERY-ADAPTER-001")
        || text(manifest, "status") != Some("pinned_query_metadata_only_no_candidate_world_payloads")
    {
        errors.push("ACED-Query source manifest identity or closed top-level contract mismatch".into());
    }

    let upstream = &manifest["upstream"];
    if text(upstream, "repository") != Some(UPSTREAM_REPOSITORY)
        || text(upstream, "commit") != Some(UPSTREAM_COMMIT)
        || text(upstream, "license") != Some("MIT")
        || text(upstream, "paper_revision") != Some("v2")
        || text(upstream, "paper_revision_date") != Some("2026-11-08")
    {
        errors.push("ACED-Query upstream repository, commit, license or paper revision mismatch".into());
    }

    let artifacts = &manifest["artifacts"];
    if text(artifacts, "inventory_path")
        != Some("research/aced-adapter-001/upstream/aced_query_world_inventory.json")
        || text(artifacts, "inventory_sha256") != Some(sha256(inventory_bytes).as_str())
        || text(artifacts, "upstream_checksums_path") != Some("data/SHA256SUMS")
        || text(artifacts, "upstream_checksums_sha256") != Some(sha256(checksum_bytes).as_str())
        || !text(artifacts, "upstream_checksums_git_blob_sha").is_some_and(|value| is_hex(value, 40))
    {
        errors.push("ACED-Query inventory or upstream checksum artifact digest mismatch".into());
    }

    if text(inventory, "schema_version") != Some("aced-query-world-inventory-v0.1")
        || text(inventory, "benchmark_id") != Some("ACED-QUERY-ADAPTER-001")
        || !exact_fields(inventory, &["schema_version", "benchmark_id", "tracks"])
        || !exact_fields(&inventory["tracks"], &TRACKS)
    {
        errors.push("ACED-Query inventory identity or track set mismatch".into());
    }

    if manifest["candidate_data_boundary"]["copied_payloads"] != false
        || manifest["candidate_data_boundary"]["candidate_visible_world_files"] != false
        || manifest["candidate_data_boundary"]["files_are_metadata_only"] != true
    {
        errors.push("ACED-Query candidate data boundary must remain metadata-only".into());
    }

    let mut reports = BTreeMap::new();
    for (track, expected_total) in TRACKS.iter().zip(EXPECTED_COUNTS) {
        if manifest["track_counts"][*track].as_u64() != Some(expected_total as u64) {
            errors.push(format!("manifest track count mismatch for {track}"));
        }
        match validate_track(track, &inventory["tracks"][*track], manifest) {
            Ok(report) => {
                if report.rows.len() != expected_total {
                    errors.push(format!("validated {track} row count mismatch"));
                }
                reports.insert(*track, report);
            }
            Err(found) => errors.extend(found),
        }
    }

    let total_track_entries = EXPECTED_COUNTS.iter().sum::<usize>();
    let core_struct_overlap = overlap_count(
        &inventory["tracks"]["core_struct"].as_array().cloned().unwrap_or_default(),
        &inventory["tracks"]["guess_struct"].as_array().cloned().unwrap_or_default(),
        true,
    );
    let core_decision_exact_overlap = overlap_count(
        &inventory["tracks"]["core_decision"].as_array().cloned().unwrap_or_default(),
        &inventory["tracks"]["guess_decision"].as_array().cloned().unwrap_or_default(),
        true,
    );
    let core_decision_filename_collision = overlap_count(
        &inventory["tracks"]["core_decision"].as_array().cloned().unwrap_or_default(),
        &inventory["tracks"]["guess_decision"].as_array().cloned().unwrap_or_default(),
        false,
    );

    let unique_payloads = inventory["tracks"]
        .as_object()
        .into_iter()
        .flat_map(|tracks| tracks.values())
        .filter_map(Value::as_array)
        .flatten()
        .filter_map(|row| text(row, "sha256").map(str::to_owned))
        .collect::<BTreeSet<_>>()
        .len();

    let split = &manifest["split_relationships"];
    if split["core_struct_guess_struct_exact_content_overlap"].as_u64()
        != Some(core_struct_overlap as u64)
        || split["core_decision_guess_decision_exact_content_overlap"].as_u64()
            != Some(core_decision_exact_overlap as u64)
        || split["core_decision_guess_decision_filename_collision_count"].as_u64()
            != Some(core_decision_filename_collision as u64)
        || split["total_track_entries"].as_u64() != Some(total_track_entries as u64)
        || split["unique_payload_sha256_count"].as_u64() != Some(unique_payloads as u64)
    {
        errors.push("ACED-Query split identity or unique-payload counts do not match the exact inventory".into());
    }

    let checksum_source = std::str::from_utf8(checksum_bytes)
        .map_err(|error| vec![format!("upstream SHA256SUMS is not UTF-8: {error}")]);
    let checksum_entries = match checksum_source.and_then(parse_checksums) {
        Ok(entries) => {
            for track in TRACKS {
                for row in inventory["tracks"][track].as_array().into_iter().flatten() {
                    let path = text(row, "path").unwrap_or("");
                    let digest = text(row, "sha256").unwrap_or("");
                    match entries.get(path) {
                        Some(source_digest) if source_digest == digest => {}
                        Some(source_digest) => errors.push(format!(
                            "upstream checksum mismatch for {path}: inventory={digest}, source={source_digest}"
                        )),
                        None => errors.push(format!("upstream SHA256SUMS omits {path}")),
                    }
                }
            }
            let query_count = entries.keys().filter(|path| path.starts_with("query/")).count();
            if query_count != total_track_entries {
                errors.push(format!(
                    "upstream checksum query path count expected {total_track_entries}, got {query_count}"
                ));
            }
            entries.len()
        }
        Err(found) => {
            errors.extend(found);
            0
        }
    };

    // Keep the reports live until their distributions have been read: this also verifies the
    // manifest distribution objects were parsed, rather than relying only on filename patterns.
    for (track, report) in &reports {
        if report.domains.is_empty() || report.families.is_empty() {
            errors.push(format!("track {track} has an empty validated distribution"));
        }
    }

    if errors.is_empty() {
        Ok(QueryInventoryReport {
            total_track_entries,
            unique_payload_sha256_count: unique_payloads,
            core_struct_guess_struct_exact_content_overlap: core_struct_overlap,
            core_decision_guess_decision_exact_content_overlap: core_decision_exact_overlap,
            core_decision_guess_decision_filename_collision_count: core_decision_filename_collision,
            checksum_entries,
            per_track_counts: TRACKS
                .iter()
                .zip(EXPECTED_COUNTS)
                .map(|(track, count)| ((*track).to_string(), count))
                .collect(),
        })
    } else {
        errors.sort();
        errors.dedup();
        Err(errors)
    }
}
