use csv::{ReaderBuilder, StringRecord};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;

const DATASET_ID: &str = "OPENTEXTILE-NIR-2026";

#[derive(Debug)]
struct Inputs {
    ground_truth: PathBuf,
    rgb_mean: PathBuf,
    spectra_mean: PathBuf,
    spectra_coordinates: PathBuf,
    source_metadata: PathBuf,
    split_manifest: PathBuf,
    expected_head: String,
    output: PathBuf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LabelState {
    Known,
    Unknown,
    Outlier,
}

#[derive(Debug)]
struct GroundTruthReport {
    states: BTreeMap<String, LabelState>,
    composition_column_count: usize,
    known_complete_compositions: usize,
    known_sparse_compositions: usize,
    composition_sum_residuals: Vec<(String, f64)>,
}

#[derive(Debug)]
struct SplitReport {
    counts: BTreeMap<String, usize>,
    assignments: usize,
}

fn sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn read_json(path: &Path) -> Result<(Vec<u8>, Value), Box<dyn Error>> {
    let bytes = fs::read(path)?;
    let value: Value = serde_json::from_slice(&bytes)?;
    Ok((bytes, value))
}

fn read_bytes(path: &Path) -> Result<Vec<u8>, Box<dyn Error>> {
    Ok(fs::read(path)?)
}

fn normalize_header(value: &str) -> String {
    value.trim().trim_start_matches('\u{feff}').to_ascii_lowercase()
}

fn empty_record(record: &StringRecord) -> bool {
    record.iter().all(|value| value.trim().is_empty())
}

fn label_index(headers: &StringRecord) -> Result<usize, String> {
    headers
        .iter()
        .position(|value| {
            let value = normalize_header(value);
            value == "label" || value == "sample" || value == "sample_id"
        })
        .ok_or_else(|| "CSV is missing a specimen label column".into())
}

fn parse_numeric(value: &str, field: &str, sample_id: &str) -> Result<f64, String> {
    let parsed = value.trim().parse::<f64>().map_err(|_| {
        format!("{field} for specimen {sample_id} is not numeric: {:?}", value)
    })?;
    if !parsed.is_finite() {
        return Err(format!("{field} for specimen {sample_id} is not finite"));
    }
    Ok(parsed)
}

fn parse_ground_truth<R: Read>(reader: R) -> Result<GroundTruthReport, String> {
    let mut csv = ReaderBuilder::new()
        .delimiter(b';')
        .flexible(false)
        .from_reader(reader);
    let headers = csv.headers().map_err(|error| error.to_string())?.clone();
    let id_index = label_index(&headers)?;
    let status_index = headers
        .iter()
        .position(|value| {
            let value = normalize_header(value);
            value.contains("unknown") && value.contains("outlier")
        })
        .ok_or_else(|| "ground-truth CSV is missing the required unknown/outlier status column".to_owned())?;
    let composition_indices: Vec<usize> = headers
        .iter()
        .enumerate()
        .filter_map(|(index, value)| value.contains('%').then_some(index))
        .collect();
    if composition_indices.is_empty() {
        return Err("ground-truth CSV has no percent-valued fibre-composition columns".into());
    }

    let mut states = BTreeMap::new();
    let mut known_complete_compositions = 0usize;
    let mut known_incomplete_compositions = 0usize;
    let mut composition_sum_residuals = Vec::new();

    for (row_index, row) in csv.records().enumerate() {
        let row = row.map_err(|error| format!("ground-truth row {}: {error}", row_index + 2))?;
        if empty_record(&row) {
            continue;
        }
        let sample_id = row
            .get(id_index)
            .unwrap_or_default()
            .trim()
            .to_owned();
        if sample_id.is_empty() {
            return Err(format!("ground-truth row {} has an empty specimen ID", row_index + 2));
        }
        if states.contains_key(&sample_id) {
            return Err(format!("duplicate ground-truth specimen ID {sample_id}"));
        }

        let status = row
            .get(status_index)
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase();
        let label_state = if status.contains("unknown") {
            LabelState::Unknown
        } else if status.contains("outlier") {
            LabelState::Outlier
        } else {
            LabelState::Known
        };

        if label_state == LabelState::Known {
            let mut sum = 0.0;
            let mut complete = true;
            for index in &composition_indices {
                let value = row.get(*index).unwrap_or_default().trim();
                if value.is_empty() {
                    complete = false;
                    continue;
                }
                let name = headers.get(*index).unwrap_or("composition");
                let parsed = parse_numeric(value, name, &sample_id)?;
                if !(0.0..=100.0).contains(&parsed) {
                    return Err(format!(
                        "composition value {parsed} in {name} for {sample_id} is outside 0..100 percent"
                    ));
                }
                sum += parsed;
            }
            if complete {
                known_complete_compositions += 1;
                composition_sum_residuals.push((sample_id.clone(), 100.0 - sum));
            } else {
                known_incomplete_compositions += 1;
            }
        }

        states.insert(sample_id, label_state);
    }

    if states.is_empty() {
        return Err("ground-truth CSV contained no specimen rows".into());
    }

    Ok(GroundTruthReport {
        states,
        composition_column_count: composition_indices.len(),
        known_complete_compositions,
        known_incomplete_compositions,
        composition_sum_residuals,
    })
}

fn parse_rgb<R: Read>(reader: R) -> Result<BTreeSet<String>, String> {
    let mut csv = ReaderBuilder::new()
        .delimiter(b';')
        .flexible(false)
        .from_reader(reader);
    let headers = csv.headers().map_err(|error| error.to_string())?.clone();
    let id_index = label_index(&headers)?;
    let channel_indices: Vec<usize> = ["r", "g", "b"]
        .iter()
        .map(|wanted| {
            headers
                .iter()
                .position(|value| normalize_header(value) == *wanted)
                .ok_or_else(|| format!("RGB summary is missing channel {wanted}"))
        })
        .collect::<Result<_, _>>()?;

    let mut ids = BTreeSet::new();
    for (row_index, row) in csv.records().enumerate() {
        let row = row.map_err(|error| format!("RGB row {}: {error}", row_index + 2))?;
        if empty_record(&row) {
            continue;
        }
        let sample_id = row.get(id_index).unwrap_or_default().trim().to_owned();
        if sample_id.is_empty() {
            return Err(format!("RGB row {} has empty specimen ID", row_index + 2));
        }
        if !ids.insert(sample_id.clone()) {
            return Err(format!("duplicate RGB specimen ID {sample_id}"));
        }
        for (channel, index) in ["R", "G", "B"].iter().zip(channel_indices.iter()) {
            let value = parse_numeric(row.get(*index).unwrap_or_default(), channel, &sample_id)?;
            if !(0.0..=255.0).contains(&value) {
                return Err(format!("{channel} value {value} for {sample_id} is outside 0..255"));
            }
        }
    }
    if ids.is_empty() {
        return Err("RGB summary contains no specimen rows".into());
    }
    Ok(ids)
}


fn parse_spectra_coordinates<R: Read>(
    reader: R,
    expected_image_width: u64,
    maximum_image_height: u64,
) -> Result<BTreeSet<String>, String> {
    let mut csv = ReaderBuilder::new()
        .delimiter(b';')
        .flexible(false)
        .from_reader(reader);
    let headers = csv.headers().map_err(|error| error.to_string())?.clone();
    let id_index = label_index(&headers)?;
    let column = |name: &str| {
        headers
            .iter()
            .position(|value| normalize_header(value) == name)
            .ok_or_else(|| format!("spectral-coordinate CSV is missing {name}"))
    };
    let y_start_index = column("y_start")?;
    let y_end_index = column("y_end")?;
    let x_start_index = column("x_start")?;
    let x_end_index = column("x_end")?;

    let mut ids = BTreeSet::new();
    for (row_index, row) in csv.records().enumerate() {
        let row = row.map_err(|error| format!("coordinate row {}: {error}", row_index + 2))?;
        if empty_record(&row) {
            continue;
        }
        let sample_id = row.get(id_index).unwrap_or_default().trim().to_owned();
        if sample_id.is_empty() {
            return Err(format!("coordinate row {} has empty specimen ID", row_index + 2));
        }
        if !ids.insert(sample_id.clone()) {
            return Err(format!("duplicate coordinate specimen ID {sample_id}"));
        }

        let parse_integer = |index: usize, name: &str| -> Result<u64, String> {
            let value = parse_numeric(row.get(index).unwrap_or_default(), name, &sample_id)?;
            if value < 0.0 || value.fract() != 0.0 {
                return Err(format!(
                    "{name} for {sample_id} must be a non-negative integer, found {value}"
                ));
            }
            Ok(value as u64)
        };
        let y_start = parse_integer(y_start_index, "y_start")?;
        let y_end = parse_integer(y_end_index, "y_end")?;
        let x_start = parse_integer(x_start_index, "x_start")?;
        let x_end = parse_integer(x_end_index, "x_end")?;

        if y_start >= y_end || y_end > maximum_image_height {
            return Err(format!(
                "invalid y-coordinate interval for {sample_id}: {y_start}..{y_end}, max {maximum_image_height}"
            ));
        }
        if x_start >= x_end || x_end > expected_image_width {
            return Err(format!(
                "invalid x-coordinate interval for {sample_id}: {x_start}..{x_end}, max {expected_image_width}"
            ));
        }
    }
    if ids.is_empty() {
        return Err("spectral-coordinate CSV contains no specimen rows".into());
    }
    Ok(ids)
}

fn parse_mean_spectra<R: Read>(reader: R, expected_wavelengths: usize) -> Result<(BTreeSet<String>, usize), String> {
    let mut csv = ReaderBuilder::new()
        .delimiter(b';')
        .flexible(false)
        .from_reader(reader);
    let headers = csv.headers().map_err(|error| error.to_string())?.clone();
    let id_index = label_index(&headers)?;
    let mut wavelength_columns = Vec::new();
    let mut wavelengths = Vec::new();

    for (index, header) in headers.iter().enumerate() {
        if index == id_index {
            continue;
        }
        match header.trim().parse::<f64>() {
            Ok(wavelength) if wavelength.is_finite() => {
                wavelength_columns.push(index);
                wavelengths.push(wavelength);
            }
            _ => {
                let normalized = normalize_header(header);
                if normalized != "index" && !normalized.is_empty() {
                    return Err(format!(
                        "mean-spectrum column {:?} is neither a specimen ID nor a numeric wavelength",
                        header
                    ));
                }
            }
        }
    }

    if wavelength_columns.len() != expected_wavelengths {
        return Err(format!(
            "expected {expected_wavelengths} wavelength columns, found {}",
            wavelength_columns.len()
        ));
    }
    if wavelengths.windows(2).any(|pair| pair[1] <= pair[0]) {
        return Err("wavelength headers must be strictly increasing".into());
    }

    let mut ids = BTreeSet::new();
    for (row_index, row) in csv.records().enumerate() {
        let row = row.map_err(|error| format!("mean-spectrum row {}: {error}", row_index + 2))?;
        if empty_record(&row) {
            continue;
        }
        let sample_id = row.get(id_index).unwrap_or_default().trim().to_owned();
        if sample_id.is_empty() {
            return Err(format!("mean-spectrum row {} has empty specimen ID", row_index + 2));
        }
        if !ids.insert(sample_id.clone()) {
            return Err(format!("duplicate mean-spectrum specimen ID {sample_id}"));
        }
        for index in &wavelength_columns {
            parse_numeric(row.get(*index).unwrap_or_default(), "reflectance", &sample_id)?;
        }
    }
    if ids.is_empty() {
        return Err("mean-spectrum CSV contains no specimen rows".into());
    }
    Ok((ids, wavelength_columns.len()))
}

fn validate_split_manifest(
    manifest: &Value,
    ground_truth: &BTreeMap<String, LabelState>,
) -> Result<SplitReport, String> {
    if manifest.get("split_strategy").and_then(Value::as_str) != Some("specimen_grouped") {
        return Err("split_strategy must be specimen_grouped; pixel/frame/row splits are prohibited".into());
    }
    let assignments = manifest
        .get("assignments")
        .and_then(Value::as_array)
        .ok_or("split manifest assignments must be an array")?;
    let mut assigned = BTreeMap::new();
    let mut counts = BTreeMap::from([
        ("train".to_owned(), 0usize),
        ("validation".to_owned(), 0usize),
        ("test".to_owned(), 0usize),
    ]);

    for (index, assignment) in assignments.iter().enumerate() {
        let sample_id = assignment
            .get("sample_id")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("split assignment {index} has no sample_id"))?
            .trim();
        let split = assignment
            .get("split")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("split assignment {index} has no split"))?;
        if !counts.contains_key(split) {
            return Err(format!("unsupported split name {split} for {sample_id}"));
        }
        if !ground_truth.contains_key(sample_id) {
            return Err(format!("split manifest contains unknown specimen {sample_id}"));
        }
        if assigned.insert(sample_id.to_owned(), split.to_owned()).is_some() {
            return Err(format!("specimen {sample_id} is assigned more than once"));
        }
        if split == "train"
            && matches!(
                ground_truth.get(sample_id),
                Some(LabelState::Unknown | LabelState::Outlier)
            )
        {
            return Err(format!(
                "unknown/outlier specimen {sample_id} cannot enter the supervised training split"
            ));
        }
        *counts.get_mut(split).expect("split key checked") += 1;
    }

    if assigned.len() != ground_truth.len() {
        let missing: Vec<&str> = ground_truth
            .keys()
            .filter(|sample_id| !assigned.contains_key(*sample_id))
            .map(String::as_str)
            .collect();
        return Err(format!("split manifest does not assign every specimen; missing: {}", missing.join(", ")));
    }
    if counts.values().any(|count| *count == 0) {
        return Err("train, validation and test splits must all be non-empty".into());
    }

    Ok(SplitReport {
        counts,
        assignments: assigned.len(),
    })
}

fn validate_rights(metadata: &Value) -> Result<(), String> {
    let rights = metadata
        .get("rights_status")
        .and_then(Value::as_str)
        .ok_or("source metadata must declare rights_status")?;
    if !matches!(rights, "verified_for_internal_use" | "verified_for_redistribution") {
        return Err(format!(
            "rights_status {rights:?} is not admitted; explicitly review dataset file-level terms first"
        ));
    }
    if metadata
        .get("rights_review_ref")
        .and_then(Value::as_str)
        .is_none_or(|value| value.trim().is_empty())
    {
        return Err("source metadata must include rights_review_ref".into());
    }
    Ok(())
}

fn validate_source_metadata(
    metadata: &Value,
    sample_count: usize,
    unknown_count: usize,
    outlier_count: usize,
    composition_column_count: usize,
    wavelength_count: usize,
    image_width: usize,
    maximum_image_height: usize,
) -> Result<(), String> {
    if metadata.get("source_id").and_then(Value::as_str) != Some(DATASET_ID) {
        return Err(format!("source metadata source_id must be {DATASET_ID}"));
    }
    validate_rights(metadata)?;

    for (key, actual) in [
        ("expected_sample_count", sample_count),
        ("expected_unknown_count", unknown_count),
        ("expected_outlier_count", outlier_count),
        ("expected_composition_column_count", composition_column_count),
        ("expected_wavelength_count", wavelength_count),
        ("expected_image_width", image_width),
        ("expected_max_image_height", maximum_image_height),
    ] {
        let expected = metadata
            .get(key)
            .and_then(Value::as_u64)
            .ok_or_else(|| format!("source metadata is missing integer {key}"))?
            as usize;
        if expected != actual {
            return Err(format!("{key} expected {expected}, found {actual}"));
        }
    }
    Ok(())
}

fn current_git_head() -> Result<String, Box<dyn Error>> {
    let output = Command::new("git").args(["rev-parse", "HEAD"]).output()?;
    if !output.status.success() {
        return Err("git rev-parse HEAD failed".into());
    }
    Ok(String::from_utf8(output.stdout)?.trim().to_owned())
}

fn parse_args() -> Result<Inputs, Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let allowed = [
        "--ground-truth",
        "--rgb-mean",
        "--spectra-mean",
        "--spectra-coordinates",
        "--source-metadata",
        "--split-manifest",
        "--expected-head",
        "--out",
    ];
    let mut options = BTreeMap::new();
    let mut index = 0usize;
    while index < args.len() {
        let key = args[index].as_str();
        if !allowed.contains(&key) {
            return Err(format!("unknown argument {key}").into());
        }
        if index + 1 >= args.len() || args[index + 1].starts_with("--") {
            return Err(format!("missing value for {key}").into());
        }
        if options.insert(key.to_owned(), args[index + 1].clone()).is_some() {
            return Err(format!("duplicate argument {key}").into());
        }
        index += 2;
    }
    let required = |key: &str| -> Result<PathBuf, Box<dyn Error>> {
        options
            .get(key)
            .map(PathBuf::from)
            .ok_or_else(|| format!("{key} is required").into())
    };
    let expected_head = options
        .get("--expected-head")
        .cloned()
        .ok_or("--expected-head is required")?;
    if expected_head.len() != 40 || !expected_head.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err("--expected-head must be a 40-character commit SHA".into());
    }
    Ok(Inputs {
        ground_truth: required("--ground-truth")?,
        rgb_mean: required("--rgb-mean")?,
        spectra_mean: required("--spectra-mean")?,
        spectra_coordinates: required("--spectra-coordinates")?,
        source_metadata: required("--source-metadata")?,
        split_manifest: required("--split-manifest")?,
        expected_head,
        output: options
            .get("--out")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("/tmp/opentextile-intake-receipt.json")),
    })
}

fn run(inputs: Inputs) -> Result<Value, Box<dyn Error>> {
    let actual_head = current_git_head()?;
    if actual_head != inputs.expected_head {
        return Err(format!(
            "exact-head mismatch: expected {}, checked out {}",
            inputs.expected_head, actual_head
        )
        .into());
    }

    let ground_truth_bytes = read_bytes(&inputs.ground_truth)?;
    let rgb_bytes = read_bytes(&inputs.rgb_mean)?;
    let spectra_bytes = read_bytes(&inputs.spectra_mean)?;
    let coordinates_bytes = read_bytes(&inputs.spectra_coordinates)?;
    let (metadata_bytes, metadata) = read_json(&inputs.source_metadata)?;
    let (split_bytes, split_manifest) = read_json(&inputs.split_manifest)?;

    let ground_truth = parse_ground_truth(ground_truth_bytes.as_slice())
        .map_err(|error| format!("ground-truth intake failed: {error}"))?;
    let rgb_ids = parse_rgb(rgb_bytes.as_slice())
        .map_err(|error| format!("RGB intake failed: {error}"))?;
    let (spectra_ids, wavelength_count) =
        parse_mean_spectra(spectra_bytes.as_slice(), 288)
            .map_err(|error| format!("mean-spectrum intake failed: {error}"))?;
    let coordinate_ids = parse_spectra_coordinates(
        coordinates_bytes.as_slice(),
        379,
        414,
    )
    .map_err(|error| format!("mean-spectrum coordinate intake failed: {error}"))?;

    let ground_truth_ids: BTreeSet<String> = ground_truth.states.keys().cloned().collect();
    if rgb_ids != ground_truth_ids {
        return Err("RGB specimen IDs do not exactly match ground-truth IDs".into());
    }
    if spectra_ids != ground_truth_ids {
        return Err("mean-spectrum specimen IDs do not exactly match ground-truth IDs".into());
    }
    if coordinate_ids != ground_truth_ids {
        return Err("mean-spectrum coordinate specimen IDs do not exactly match ground-truth IDs".into());
    }

    let unknown_count = ground_truth
        .states
        .values()
        .filter(|state| **state == LabelState::Unknown)
        .count();
    let outlier_count = ground_truth
        .states
        .values()
        .filter(|state| **state == LabelState::Outlier)
        .count();
    let sample_count = ground_truth.states.len();

    validate_source_metadata(
        &metadata,
        sample_count,
        unknown_count,
        outlier_count,
        ground_truth.composition_column_count,
        wavelength_count,
        379,
        414,
    )
    .map_err(|error| format!("source metadata rejected: {error}"))?;

    let split_report = validate_split_manifest(&split_manifest, &ground_truth.states)
        .map_err(|error| format!("split manifest rejected: {error}"))?;

    let residuals: Vec<Value> = ground_truth
        .composition_sum_residuals
        .iter()
        .map(|(sample_id, residual)| json!({"sample_id": sample_id, "residual_percentage_points": residual}))
        .collect();

    let receipt = json!({
        "receipt_version": "0.1.0",
        "status": "PASS",
        "commit": actual_head,
        "source_id": DATASET_ID,
        "dataset_doi": metadata.get("dataset_doi"),
        "rights_status": metadata.get("rights_status"),
        "source_provenance": "operator_declared_and_file_hash_bound_not_source_authenticated",
        "inputs": {
            "ground_truth": {"path": inputs.ground_truth.display().to_string(), "sha256": sha256(&ground_truth_bytes)},
            "rgb_mean": {"path": inputs.rgb_mean.display().to_string(), "sha256": sha256(&rgb_bytes)},
            "spectra_mean": {"path": inputs.spectra_mean.display().to_string(), "sha256": sha256(&spectra_bytes)},
            "spectra_coordinates": {"path": inputs.spectra_coordinates.display().to_string(), "sha256": sha256(&coordinates_bytes)},
            "source_metadata": {"path": inputs.source_metadata.display().to_string(), "sha256": sha256(&metadata_bytes)},
            "split_manifest": {"path": inputs.split_manifest.display().to_string(), "sha256": sha256(&split_bytes)}
        },
        "specimens": {
            "total": sample_count,
            "known": sample_count - unknown_count - outlier_count,
            "unknown": unknown_count,
            "outlier": outlier_count,
            "known_complete_compositions": ground_truth.known_complete_compositions,
            "composition_column_count": ground_truth.composition_column_count,
            "known_sparse_compositions": ground_truth.known_incomplete_compositions,
            "composition_sum_residuals": residuals
        },
        "modalities": {
            "rgb_channels": 3,
            "mean_spectrum_wavelength_count": wavelength_count
        },
        "split": {
            "strategy": "specimen_grouped",
            "assignments": split_report.assignments,
            "counts": split_report.counts,
            "unknown_and_outlier_in_train": false
        },
        "claim_ceiling": "File integrity, specimen joins, source metadata and specimen-level split integrity only; composition labels are not independently authenticated and no model or physical sorting performance is qualified."
    });

    if let Some(parent) = inputs.output.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&inputs.output, serde_json::to_vec_pretty(&receipt)?)?;
    Ok(receipt)
}

fn main() {
    match parse_args().and_then(run) {
        Ok(receipt) => {
            println!(
                "{}",
                serde_json::to_string_pretty(&receipt).expect("receipt serializes")
            );
        }
        Err(error) => {
            eprintln!("textile dataset audit failed: {error}");
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GT: &str = concat!(
        "Label;Unknown or outlier?;Cotton [%];Polyester [%];Color\n",
        "sample_A;;60;40;blue\n",
        "sample_B;;100;0;black\n",
        "sample_C;unknown;;;grey\n",
        "sample_D;outlier;50;50;red\n"
    );
    const RGB: &str = concat!(
        "Label;R;G;B\n",
        "sample_A;10;20;30\n",
        "sample_B;40;50;60\n",
        "sample_C;70;80;90\n",
        "sample_D;100;110;120\n"
    );
    const SPECTRA: &str = concat!(
        "Label;900.0;910.0;920.0\n",
        "sample_A;0.1;0.2;0.3\n",
        "sample_B;0.2;0.3;0.4\n",
        "sample_C;0.3;0.4;0.5\n",
        "sample_D;0.4;0.5;0.6\n"
    );

    fn small_ground_truth() -> GroundTruthReport {
        parse_ground_truth(GT.as_bytes()).unwrap()
    }

    fn small_split() -> Value {
        json!({
            "split_strategy": "specimen_grouped",
            "assignments": [
                {"sample_id":"sample_A","split":"train"},
                {"sample_id":"sample_B","split":"validation"},
                {"sample_id":"sample_C","split":"test"},
                {"sample_id":"sample_D","split":"test"}
            ]
        })
    }

    #[test]
    fn parses_known_unknown_and_outlier_as_distinct_states() {
        let report = small_ground_truth();
        assert_eq!(report.states.get("sample_A"), Some(&LabelState::Known));
        assert_eq!(report.states.get("sample_C"), Some(&LabelState::Unknown));
        assert_eq!(report.states.get("sample_D"), Some(&LabelState::Outlier));
        assert_eq!(report.known_complete_compositions, 2);
    }

    #[test]
    fn rgb_and_mean_spectra_parse_and_join_by_specimen_id() {
        let gt = small_ground_truth();
        let rgb = parse_rgb(RGB.as_bytes()).unwrap();
        let (spectra, count) = parse_mean_spectra(SPECTRA.as_bytes(), 3).unwrap();
        let coordinates = parse_spectra_coordinates(COORDINATES.as_bytes(), 379, 414).unwrap();
        let gt_ids: BTreeSet<String> = gt.states.keys().cloned().collect();
        assert_eq!(rgb, gt_ids);
        assert_eq!(spectra, gt_ids);
        assert_eq!(coordinates, gt_ids);
        assert_eq!(count, 3);
    }

    #[test]
    fn split_manifest_is_specimen_grouped_and_complete() {
        let report = validate_split_manifest(&small_split(), &small_ground_truth().states).unwrap();
        assert_eq!(report.assignments, 4);
        assert_eq!(report.counts.get("train"), Some(&1));
        assert_eq!(report.counts.get("validation"), Some(&1));
        assert_eq!(report.counts.get("test"), Some(&2));
    }

    #[test]
    fn rejects_ground_truth_without_explicit_unknown_outlier_status_column() {
        let bad = GT.replace("Unknown or outlier?;", "");
        let error = parse_ground_truth(bad.as_bytes()).unwrap_err();
        assert!(error.contains("missing the required unknown/outlier status column"));
    }

    #[test]
    fn rejects_duplicate_ground_truth_specimen_id() {
        let duplicate = format!("{GT}sample_A;;50;50;blue\n");
        let error = parse_ground_truth(duplicate.as_bytes()).unwrap_err();
        assert!(error.contains("duplicate ground-truth specimen ID"));
    }

    #[test]
    fn rejects_duplicate_rgb_specimen_id() {
        let duplicate = format!("{RGB}sample_A;1;2;3\n");
        let error = parse_rgb(duplicate.as_bytes()).unwrap_err();
        assert!(error.contains("duplicate RGB specimen ID"));
    }

    #[test]
    fn rejects_non_monotonic_wavelengths() {
        let bad = SPECTRA.replace("900.0;910.0;920.0", "900.0;920.0;910.0");
        let error = parse_mean_spectra(bad.as_bytes(), 3).unwrap_err();
        assert!(error.contains("strictly increasing"));
    }

    #[test]
    fn rejects_non_finite_spectral_value() {
        let bad = SPECTRA.replace("0.1;0.2;0.3", "0.1;NaN;0.3");
        let error = parse_mean_spectra(bad.as_bytes(), 3).unwrap_err();
        assert!(error.contains("not finite"));
    }

    #[test]
    fn rejects_invalid_spectral_coordinate_bounds() {
        let bad = COORDINATES.replace("sample_A;10;40;20;60", "sample_A;40;10;20;60");
        let error = parse_spectra_coordinates(bad.as_bytes(), 379, 414).unwrap_err();
        assert!(error.contains("invalid y-coordinate interval"));
    }

    #[test]
    fn rejects_non_integer_spectral_coordinates() {
        let bad = COORDINATES.replace("sample_A;10;40;20;60", "sample_A;10.5;40;20;60");
        let error = parse_spectra_coordinates(bad.as_bytes(), 379, 414).unwrap_err();
        assert!(error.contains("non-negative integer"));
    }

    #[test]
    fn rejects_unknown_or_outlier_specimen_in_training() {
        let mut split = small_split();
        split["assignments"][2]["split"] = json!("train");
        let error = validate_split_manifest(&split, &small_ground_truth().states).unwrap_err();
        assert!(error.contains("cannot enter the supervised training split"));
    }

    #[test]
    fn rejects_pixel_level_split_strategy() {
        let mut split = small_split();
        split["split_strategy"] = json!("pixel_random");
        let error = validate_split_manifest(&split, &small_ground_truth().states).unwrap_err();
        assert!(error.contains("specimen_grouped"));
    }

    #[test]
    fn rejects_missing_specimen_assignment() {
        let mut split = small_split();
        split["assignments"].as_array_mut().unwrap().pop();
        let error = validate_split_manifest(&split, &small_ground_truth().states).unwrap_err();
        assert!(error.contains("does not assign every specimen"));
    }

    #[test]
    fn rejects_composition_outside_percentage_range() {
        let bad = GT.replace("60;40", "120;40");
        let error = parse_ground_truth(bad.as_bytes()).unwrap_err();
        assert!(error.contains("outside 0..100 percent"));
    }

    #[test]
    fn refuses_unreviewed_dataset_rights() {
        let metadata = json!({
            "source_id": DATASET_ID,
            "rights_status": "unknown",
            "rights_review_ref": "not-reviewed"
        });
        assert!(validate_rights(&metadata).is_err());
    }
}
