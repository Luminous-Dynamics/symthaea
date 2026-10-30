//! Verify a spatial PSU frame is a complete, non-overlapping 3D lattice tiling
//! of its explicitly declared voxel bounding box.
//!
//! This verifier cannot establish that the bounding box covers all biologically
//! eligible tissue or that the eligibility mask is truthful. It checks the
//! internal geometry and provenance declarations supplied by the frame producer.

use clap::Parser;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Debug, Parser)]
#[command(
    name = "connectome-spatial-frame-audit",
    version,
    about = "Audit complete disjoint 3D spatial PSU lattice frames"
)]
struct Args {
    /// Spatial frame manifest JSON.
    #[arg(long)]
    frame: PathBuf,
    /// Optional JSON report path; stdout when omitted.
    #[arg(long)]
    output: Option<PathBuf>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct FrameInput {
    schema_version: u32,
    frame_id: String,
    source: SourceBinding,
    coordinate_frame: CoordinateFrame,
    origin_voxel: [i64; 3],
    extent_voxels: [u64; 3],
    tile_shape_voxels: [u64; 3],
    edge_policy: String,
    eligibility_mask: EligibilityMask,
    frame_protocol_sha256: String,
    frame_generator: FrameGenerator,
    cells: Vec<FrameCell>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct SourceBinding {
    volume_id: String,
    volume_revision_id: String,
    volume_content_sha256: String,
    segmentation_id: String,
    segmentation_revision_id: String,
    segmentation_content_sha256: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct CoordinateFrame {
    coordinate_frame_id: String,
    transform_sha256: String,
    voxel_size_nm: [f64; 3],
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct EligibilityMask {
    artifact_id: String,
    artifact_sha256: String,
    selection_rule_sha256: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct FrameGenerator {
    tool_id: String,
    tool_version: String,
    binary_sha256: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct FrameCell {
    psu_id: String,
    grid_index: [u64; 3],
    disposition: String,
    stratum_id: Option<String>,
    exclusion_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct PopulationPsu {
    psu_id: String,
    stratum_id: String,
}

#[derive(Debug, Serialize)]
struct AuditReport {
    schema_version: u32,
    tool: &'static str,
    frame_id: String,
    input_sha256: String,
    canonical_frame_sha256: String,
    source_volume_sha256: String,
    source_segmentation_sha256: String,
    coordinate_transform_sha256: String,
    eligibility_mask_sha256: String,
    grid_dimensions: [u64; 3],
    total_grid_cells: u64,
    eligible_grid_cells: u64,
    excluded_grid_cells: u64,
    declared_voxel_count: String,
    eligible_voxel_count: String,
    excluded_voxel_count: String,
    eligible_voxel_fraction: f64,
    excluded_voxel_fraction: f64,
    population_frame_sha256: String,
    population_frame: Vec<PopulationPsu>,
    status: &'static str,
    scope_note: &'static str,
}

fn sha256(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn canonical_psu_id(index: [u64; 3]) -> String {
    format!("{}:{}:{}", index[0], index[1], index[2])
}

fn checked_product(values: [u64; 3], label: &str) -> Result<u128, String> {
    values.iter().try_fold(1_u128, |acc, value| {
        acc.checked_mul(u128::from(*value))
            .ok_or_else(|| format!("{label} voxel count overflows u128"))
    })
}

fn grid_dimensions(extent: [u64; 3], tile: [u64; 3]) -> Result<[u64; 3], String> {
    let mut dimensions = [0_u64; 3];
    for axis in 0..3 {
        if extent[axis] == 0 {
            return Err(format!("extent_voxels axis {axis} must be positive"));
        }
        if tile[axis] == 0 {
            return Err(format!("tile_shape_voxels axis {axis} must be positive"));
        }
        // Safe ceiling division that avoids extent + tile - 1 overflow.
        dimensions[axis] = 1 + (extent[axis] - 1) / tile[axis];
    }
    Ok(dimensions)
}

fn canonical_frame_sha256(frame: &FrameInput) -> Result<String, String> {
    let mut canonical = frame.clone();
    canonical.cells.sort_by(|left, right| left.grid_index.cmp(&right.grid_index));
    let bytes = serde_json::to_vec(&canonical)
        .map_err(|error| format!("could not serialize canonical frame: {error}"))?;
    Ok(sha256(&bytes))
}

fn population_frame_sha256(population: &[PopulationPsu]) -> String {
    // Match connectome-candidate-coverage-audit::frame_sha256 exactly.
    let mut rows: Vec<(String, String)> = population
        .iter()
        .map(|row| (row.psu_id.clone(), row.stratum_id.clone()))
        .collect();
    rows.sort();
    let mut hasher = Sha256::new();
    for (psu_id, stratum_id) in rows {
        hasher.update((psu_id.len() as u64).to_be_bytes());
        hasher.update(psu_id.as_bytes());
        hasher.update((stratum_id.len() as u64).to_be_bytes());
        hasher.update(stratum_id.as_bytes());
    }
    hex::encode(hasher.finalize())
}

fn validate(frame: &FrameInput) -> Result<AuditReport, String> {
    if frame.schema_version != 1 {
        return Err("schema_version must equal 1".into());
    }
    if !nonempty(&frame.frame_id)
        || !nonempty(&frame.source.volume_id)
        || !nonempty(&frame.source.volume_revision_id)
        || !nonempty(&frame.source.segmentation_id)
        || !nonempty(&frame.source.segmentation_revision_id)
        || !nonempty(&frame.coordinate_frame.coordinate_frame_id)
        || !nonempty(&frame.eligibility_mask.artifact_id)
        || !nonempty(&frame.frame_generator.tool_id)
        || !nonempty(&frame.frame_generator.tool_version)
    {
        return Err("frame and source provenance identifiers must be non-empty".into());
    }
    if frame.edge_policy != "clip_last_tile" {
        return Err("edge_policy must equal clip_last_tile".into());
    }
    for (name, digest) in [
        ("volume_content_sha256", frame.source.volume_content_sha256.as_str()),
        ("segmentation_content_sha256", frame.source.segmentation_content_sha256.as_str()),
        ("coordinate transform_sha256", frame.coordinate_frame.transform_sha256.as_str()),
        ("eligibility mask artifact_sha256", frame.eligibility_mask.artifact_sha256.as_str()),
        ("eligibility selection_rule_sha256", frame.eligibility_mask.selection_rule_sha256.as_str()),
        ("frame_protocol_sha256", frame.frame_protocol_sha256.as_str()),
        ("frame generator binary_sha256", frame.frame_generator.binary_sha256.as_str()),
    ] {
        if !valid_sha256(digest) {
            return Err(format!("{name} must be lowercase SHA-256"));
        }
    }
    if frame.coordinate_frame.voxel_size_nm.iter().any(|value| !value.is_finite() || *value <= 0.0) {
        return Err("voxel_size_nm values must be finite and positive".into());
    }

    for axis in 0..3 {
        let extent = i64::try_from(frame.extent_voxels[axis])
            .map_err(|_| format!("extent_voxels axis {axis} exceeds signed coordinate range"))?;
        frame.origin_voxel[axis]
            .checked_add(extent)
            .ok_or_else(|| format!("origin_voxel plus extent overflows on axis {axis}"))?;
    }

    let dimensions = grid_dimensions(frame.extent_voxels, frame.tile_shape_voxels)?;
    let expected_cells = checked_product(dimensions, "grid")?;
    let declared_voxels = checked_product(frame.extent_voxels, "declared")?;
    if expected_cells > usize::MAX as u128 {
        return Err("grid cell count exceeds addressable memory".into());
    }
    if frame.cells.len() as u128 != expected_cells {
        return Err(format!(
            "frame must contain exactly {expected_cells} grid cells; found {}",
            frame.cells.len()
        ));
    }

    let mut seen_indices = BTreeSet::new();
    let mut seen_ids = BTreeSet::new();
    let mut population_frame = Vec::new();
    let mut eligible_voxels = 0_u128;
    let mut excluded_voxels = 0_u128;
    let mut eligible_cells = 0_u64;
    let mut excluded_cells = 0_u64;

    for cell in &frame.cells {
        if cell.grid_index.iter().enumerate().any(|(axis, value)| *value >= dimensions[axis]) {
            return Err(format!("PSU {} has an out-of-bounds grid_index", cell.psu_id));
        }
        if !seen_indices.insert(cell.grid_index) {
            return Err(format!("duplicate grid_index {:?}", cell.grid_index));
        }
        let expected_id = canonical_psu_id(cell.grid_index);
        if cell.psu_id != expected_id {
            return Err(format!(
                "PSU ID {} does not match canonical lattice ID {}",
                cell.psu_id, expected_id
            ));
        }
        if !seen_ids.insert(cell.psu_id.clone()) {
            return Err(format!("duplicate PSU ID {}", cell.psu_id));
        }

        let mut cell_extent = [0_u64; 3];
        for axis in 0..3 {
            let start = cell.grid_index[axis]
                .checked_mul(frame.tile_shape_voxels[axis])
                .ok_or_else(|| format!("tile start overflows on axis {axis}"))?;
            let remaining = frame.extent_voxels[axis]
                .checked_sub(start)
                .ok_or_else(|| format!("tile start exceeds extent on axis {axis}"))?;
            if remaining == 0 {
                return Err(format!("PSU {} has an empty boundary tile", cell.psu_id));
            }
            cell_extent[axis] = remaining.min(frame.tile_shape_voxels[axis]);
        }
        let cell_voxels = checked_product(cell_extent, "cell")?;

        match cell.disposition.as_str() {
            "eligible" => {
                let stratum_id = cell
                    .stratum_id
                    .as_deref()
                    .filter(|value| nonempty(value))
                    .ok_or_else(|| format!("eligible PSU {} requires a non-empty stratum_id", cell.psu_id))?;
                if cell.exclusion_reason.is_some() {
                    return Err(format!("eligible PSU {} must not have an exclusion_reason", cell.psu_id));
                }
                eligible_cells = eligible_cells.checked_add(1)
                    .ok_or_else(|| "eligible cell count overflows u64".to_string())?;
                eligible_voxels = eligible_voxels.checked_add(cell_voxels)
                    .ok_or_else(|| "eligible voxel count overflows u128".to_string())?;
                population_frame.push(PopulationPsu {
                    psu_id: cell.psu_id.clone(),
                    stratum_id: stratum_id.to_string(),
                });
            }
            "excluded" => {
                if cell.stratum_id.is_some() {
                    return Err(format!("excluded PSU {} must not have a stratum_id", cell.psu_id));
                }
                if !cell.exclusion_reason.as_deref().is_some_and(nonempty) {
                    return Err(format!("excluded PSU {} requires a non-empty exclusion_reason", cell.psu_id));
                }
                excluded_cells = excluded_cells.checked_add(1)
                    .ok_or_else(|| "excluded cell count overflows u64".to_string())?;
                excluded_voxels = excluded_voxels.checked_add(cell_voxels)
                    .ok_or_else(|| "excluded voxel count overflows u128".to_string())?;
            }
            other => return Err(format!("PSU {} has invalid disposition {other}", cell.psu_id)),
        }
    }

    if seen_indices.len() as u128 != expected_cells {
        return Err("lattice is missing one or more grid coordinates".into());
    }
    if population_frame.is_empty() {
        return Err("spatial frame contains no eligible PSUs".into());
    }
    let partitioned_voxels = eligible_voxels.checked_add(excluded_voxels)
        .ok_or_else(|| "partitioned voxel count overflows u128".to_string())?;
    if partitioned_voxels != declared_voxels {
        return Err("eligible and excluded tiles do not partition the declared voxel box".into());
    }

    let canonical_digest = canonical_frame_sha256(frame)?;
    let population_digest = population_frame_sha256(&population_frame);
    let total_cells_u64 = u64::try_from(expected_cells)
        .map_err(|_| "grid cell count exceeds u64 report range".to_string())?;
    let total_voxels_f64 = declared_voxels as f64;
    let report = AuditReport {
        schema_version: 1,
        tool: "connectome-spatial-frame-audit",
        frame_id: frame.frame_id.clone(),
        input_sha256: String::new(),
        canonical_frame_sha256: canonical_digest,
        source_volume_sha256: frame.source.volume_content_sha256.clone(),
        source_segmentation_sha256: frame.source.segmentation_content_sha256.clone(),
        coordinate_transform_sha256: frame.coordinate_frame.transform_sha256.clone(),
        eligibility_mask_sha256: frame.eligibility_mask.artifact_sha256.clone(),
        grid_dimensions: dimensions,
        total_grid_cells: total_cells_u64,
        eligible_grid_cells: eligible_cells,
        excluded_grid_cells: excluded_cells,
        declared_voxel_count: declared_voxels.to_string(),
        eligible_voxel_count: eligible_voxels.to_string(),
        excluded_voxel_count: excluded_voxels.to_string(),
        eligible_voxel_fraction: eligible_voxels as f64 / total_voxels_f64,
        excluded_voxel_fraction: excluded_voxels as f64 / total_voxels_f64,
        population_frame_sha256: population_digest,
        population_frame,
        status: "PASS",
        scope_note: "Proves exactly-once lattice tiling and eligible/excluded partitioning only within the declared integer voxel bounding box. Does not prove the box or eligibility mask covers all biologically eligible tissue, or that source digests are truthful.",
    };
    Ok(report)
}

fn run(args: Args) -> Result<i32, String> {
    let bytes = fs::read(&args.frame)
        .map_err(|error| format!("could not read {}: {error}", args.frame.display()))?;
    let frame: FrameInput = serde_json::from_slice(&bytes)
        .map_err(|error| format!("could not parse spatial frame JSON: {error}"))?;
    let mut report = match validate(&frame) {
        Ok(report) => report,
        Err(message) => {
            let failure = serde_json::json!({
                "schema_version": 1,
                "tool": "connectome-spatial-frame-audit",
                "input_sha256": sha256(&bytes),
                "status": "FAIL",
                "error": message,
                "scope_note": "A failed frame audit cannot be used as the population frame for a design-based coverage estimate."
            });
            let rendered = serde_json::to_string_pretty(&failure)
                .map_err(|error| format!("could not serialize failure report: {error}"))?;
            if let Some(output) = args.output {
                fs::write(&output, format!("{rendered}\n"))
                    .map_err(|error| format!("could not write {}: {error}", output.display()))?;
            } else {
                println!("{rendered}");
            }
            return Ok(1);
        }
    };
    report.input_sha256 = sha256(&bytes);
    let rendered = serde_json::to_string_pretty(&report)
        .map_err(|error| format!("could not serialize report: {error}"))?;
    if let Some(output) = args.output {
        fs::write(&output, format!("{rendered}\n"))
            .map_err(|error| format!("could not write {}: {error}", output.display()))?;
    } else {
        println!("{rendered}");
    }
    Ok(0)
}

fn main() -> ExitCode {
    match Args::try_parse() {
        Ok(args) => match run(args) {
            Ok(0) => ExitCode::SUCCESS,
            Ok(_) => ExitCode::from(1),
            Err(message) => {
                eprintln!("connectome-spatial-frame-audit: {message}");
                ExitCode::from(2)
            }
        },
        Err(error) => {
            let code = error.exit_code();
            let _ = error.print();
            ExitCode::from(code as u8)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> FrameInput {
        FrameInput {
            schema_version: 1,
            frame_id: "synthetic-volume-frame".into(),
            source: SourceBinding {
                volume_id: "synthetic-volume".into(),
                volume_revision_id: "volume-rev-1".into(),
                volume_content_sha256: "a".repeat(64),
                segmentation_id: "synthetic-segmentation".into(),
                segmentation_revision_id: "seg-rev-1".into(),
                segmentation_content_sha256: "b".repeat(64),
            },
            coordinate_frame: CoordinateFrame {
                coordinate_frame_id: "voxel-axes-xyz".into(),
                transform_sha256: "c".repeat(64),
                voxel_size_nm: [4.0, 4.0, 40.0],
            },
            origin_voxel: [-100, 200, 300],
            extent_voxels: [10, 5, 9],
            tile_shape_voxels: [6, 5, 8],
            edge_policy: "clip_last_tile".into(),
            eligibility_mask: EligibilityMask {
                artifact_id: "synthetic-eligibility-mask".into(),
                artifact_sha256: "d".repeat(64),
                selection_rule_sha256: "e".repeat(64),
            },
            frame_protocol_sha256: "f".repeat(64),
            frame_generator: FrameGenerator {
                tool_id: "synthetic-frame-builder".into(),
                tool_version: "1.0.0".into(),
                binary_sha256: "1".repeat(64),
            },
            cells: vec![
                FrameCell { psu_id: "0:0:0".into(), grid_index: [0, 0, 0], disposition: "eligible".into(), stratum_id: Some("region-a".into()), exclusion_reason: None },
                FrameCell { psu_id: "1:0:0".into(), grid_index: [1, 0, 0], disposition: "eligible".into(), stratum_id: Some("region-a".into()), exclusion_reason: None },
                FrameCell { psu_id: "0:0:1".into(), grid_index: [0, 0, 1], disposition: "eligible".into(), stratum_id: Some("region-b".into()), exclusion_reason: None },
                FrameCell { psu_id: "1:0:1".into(), grid_index: [1, 0, 1], disposition: "excluded".into(), stratum_id: None, exclusion_reason: Some("outside preregistered tissue mask".into()) },
            ],
        }
    }

    #[test]
    fn checked_in_synthetic_fixture_is_valid_and_reproducible() {
        let bytes = include_bytes!("../../../../docs/neuroscience/examples/CONNECTOME_SPATIAL_FRAME.example.json");
        let frame: FrameInput = serde_json::from_slice(bytes).expect("fixture JSON should parse");
        let report = validate(&frame).expect("checked-in synthetic frame should pass");
        assert_eq!(report.status, "PASS");
        assert_eq!(report.eligible_grid_cells, 3);
        assert_eq!(report.excluded_grid_cells, 1);
        assert_eq!(report.population_frame.len(), 3);
    }

    #[test]
    fn valid_partial_edge_lattice_passes_and_exports_only_eligible_psus() {
        let report = validate(&fixture()).unwrap();
        assert_eq!(report.grid_dimensions, [2, 1, 2]);
        assert_eq!(report.total_grid_cells, 4);
        assert_eq!(report.eligible_grid_cells, 3);
        assert_eq!(report.excluded_grid_cells, 1);
        assert_eq!(report.declared_voxel_count, "450");
        assert_eq!(report.eligible_voxel_count, "430");
        assert_eq!(report.excluded_voxel_count, "20");
        assert_eq!(report.population_frame.len(), 3);
        assert_eq!(report.population_frame_sha256, population_frame_sha256(&report.population_frame));
    }

    #[test]
    fn missing_grid_cell_is_rejected() {
        let mut frame = fixture();
        frame.cells.pop();
        assert!(validate(&frame).unwrap_err().contains("exactly"));
    }

    #[test]
    fn duplicate_grid_coordinate_is_rejected() {
        let mut frame = fixture();
        frame.cells[3].grid_index = [1, 0, 0];
        frame.cells[3].psu_id = "1:0:0".into();
        assert!(validate(&frame).unwrap_err().contains("duplicate grid_index"));
    }

    #[test]
    fn canonical_id_mismatch_is_rejected() {
        let mut frame = fixture();
        frame.cells[0].psu_id = "arbitrary-id".into();
        assert!(validate(&frame).unwrap_err().contains("canonical lattice ID"));
    }

    #[test]
    fn out_of_bounds_coordinate_is_rejected() {
        let mut frame = fixture();
        frame.cells[0].grid_index = [2, 0, 0];
        assert!(validate(&frame).unwrap_err().contains("out-of-bounds"));
    }

    #[test]
    fn eligible_cell_requires_stratum_and_forbids_exclusion_reason() {
        let mut frame = fixture();
        frame.cells[0].stratum_id = None;
        assert!(validate(&frame).unwrap_err().contains("requires a non-empty stratum_id"));
        let mut frame = fixture();
        frame.cells[0].exclusion_reason = Some("should not be set".into());
        assert!(validate(&frame).unwrap_err().contains("must not have an exclusion_reason"));
    }

    #[test]
    fn excluded_cell_requires_reason_and_forbids_stratum() {
        let mut frame = fixture();
        frame.cells[3].exclusion_reason = None;
        assert!(validate(&frame).unwrap_err().contains("requires a non-empty exclusion_reason"));
        let mut frame = fixture();
        frame.cells[3].stratum_id = Some("region-a".into());
        assert!(validate(&frame).unwrap_err().contains("must not have a stratum_id"));
    }

    #[test]
    fn all_excluded_frame_is_rejected() {
        let mut frame = fixture();
        for cell in &mut frame.cells {
            cell.disposition = "excluded".into();
            cell.stratum_id = None;
            cell.exclusion_reason = Some("synthetic exclusion".into());
        }
        assert!(validate(&frame).unwrap_err().contains("no eligible PSUs"));
    }

    #[test]
    fn zero_extent_and_zero_tile_are_rejected() {
        let mut frame = fixture();
        frame.extent_voxels[1] = 0;
        assert!(validate(&frame).unwrap_err().contains("extent_voxels"));
        let mut frame = fixture();
        frame.tile_shape_voxels[2] = 0;
        assert!(validate(&frame).unwrap_err().contains("tile_shape_voxels"));
    }

    #[test]
    fn malformed_source_and_transform_digests_are_rejected() {
        let mut frame = fixture();
        frame.source.volume_content_sha256 = "not-a-digest".into();
        assert!(validate(&frame).unwrap_err().contains("volume_content_sha256"));
        let mut frame = fixture();
        frame.coordinate_frame.transform_sha256 = "not-a-digest".into();
        assert!(validate(&frame).unwrap_err().contains("transform_sha256"));
    }

    #[test]
    fn non_finite_or_non_positive_voxel_sizes_are_rejected() {
        let mut frame = fixture();
        frame.coordinate_frame.voxel_size_nm[0] = f64::NAN;
        assert!(validate(&frame).unwrap_err().contains("voxel_size_nm"));
        let mut frame = fixture();
        frame.coordinate_frame.voxel_size_nm[1] = 0.0;
        assert!(validate(&frame).unwrap_err().contains("voxel_size_nm"));
    }

    #[test]
    fn population_frame_digest_matches_candidate_coverage_contract() {
        let report = validate(&fixture()).unwrap();
        let population = report.population_frame;
        let mut rows: Vec<(String, String)> = population.iter()
            .map(|row| (row.psu_id.clone(), row.stratum_id.clone()))
            .collect();
        rows.sort();
        let mut hasher = Sha256::new();
        for (psu_id, stratum_id) in rows {
            hasher.update((psu_id.len() as u64).to_be_bytes());
            hasher.update(psu_id.as_bytes());
            hasher.update((stratum_id.len() as u64).to_be_bytes());
            hasher.update(stratum_id.as_bytes());
        }
        assert_eq!(report.population_frame_sha256, hex::encode(hasher.finalize()));
    }

    #[test]
    fn input_order_does_not_change_canonical_digest_or_population_frame() {
        let mut first = fixture();
        let mut second = fixture();
        second.cells.reverse();
        let first_report = validate(&first).unwrap();
        let second_report = validate(&second).unwrap();
        assert_eq!(first_report.canonical_frame_sha256, second_report.canonical_frame_sha256);
        assert_eq!(first_report.population_frame_sha256, second_report.population_frame_sha256);
        first.cells.swap(0, 1);
        let third_report = validate(&first).unwrap();
        assert_eq!(first_report.canonical_frame_sha256, third_report.canonical_frame_sha256);
    }

    #[test]
    fn eligibility_or_stratum_changes_canonical_digest() {
        let baseline = validate(&fixture()).unwrap();
        let mut changed = fixture();
        changed.cells[0].stratum_id = Some("different-stratum".into());
        let changed = validate(&changed).unwrap();
        assert_ne!(baseline.canonical_frame_sha256, changed.canonical_frame_sha256);
    }

    #[test]
    fn overflowing_declared_volume_is_rejected() {
        let mut frame = fixture();
        frame.extent_voxels = [u64::MAX, u64::MAX, u64::MAX];
        frame.tile_shape_voxels = [1, 1, 1];
        assert!(validate(&frame).unwrap_err().contains("voxel count overflows u128"));
    }

    #[test]
    fn origin_plus_extent_must_fit_signed_voxel_coordinates() {
        let mut frame = fixture();
        frame.origin_voxel[0] = i64::MAX;
        frame.extent_voxels[0] = 1;
        frame.tile_shape_voxels[0] = 1;
        assert!(validate(&frame).unwrap_err().contains("origin_voxel plus extent overflows"));
    }

    #[test]
    fn negative_origin_is_allowed_and_preserved() {
        let report = validate(&fixture()).unwrap();
        assert_eq!(report.status, "PASS");
        assert_eq!(fixture().origin_voxel, [-100, 200, 300]);
    }
}
