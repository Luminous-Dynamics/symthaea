use cetb_rcaf::sketch_sufficiency_audit::{
    audit_sketch_conditional_score_leakage, SketchScoreObservation, SketchSufficiencyConfig,
};
use std::error::Error;
use std::fs;
use std::io::{self, Write};

fn main() -> Result<(), Box<dyn Error>> {
    let sketch_cells = [
        ("low_motion", 0.0_f64),
        ("medium_motion", 0.10),
        ("high_motion", 0.25),
    ];
    let mut source = Vec::new();
    let mut target = Vec::new();

    for (cell, shift) in sketch_cells {
        for index in 0..99 {
            let base = (index + 1) as f64 / 100.0;
            source.push(SketchScoreObservation {
                replicate_id: format!("src-{cell}-{index:03}"),
                sketch_cell_id: cell.to_string(),
                scores: [base, base * 0.8, base * 0.6],
            });
            target.push(SketchScoreObservation {
                replicate_id: format!("tgt-{cell}-{index:03}"),
                sketch_cell_id: cell.to_string(),
                scores: [
                    base + shift,
                    base * 0.8 + shift * 0.8,
                    base * 0.6 + shift * 0.6,
                ],
            });
        }
    }

    // A fourth target cell is not in the predeclared sketch support and must remain unseen.
    for index in 0..10 {
        target.push(SketchScoreObservation {
            replicate_id: format!("tgt-new-scanner-{index:03}"),
            sketch_cell_id: "new_scanner".to_string(),
            scores: [0.95, 0.76, 0.57],
        });
    }

    let report = audit_sketch_conditional_score_leakage(
        &source,
        &target,
        &SketchSufficiencyConfig {
            source_split_id: "sketch-source-null-v1".to_string(),
            target_split_id: "sketch-target-null-v1".to_string(),
            expected_sketch_cells: sketch_cells
                .iter()
                .map(|(cell, _)| (*cell).to_string())
                .collect(),
            confidence_delta: 0.05,
            minimum_replicates_per_domain_cell: 30,
        },
    )?;
    if report.cells.len() != 3
        || report.overall_estimators.len() != 3
        || report.unseen_target_sketch_cells.len() != 1
        || report.unseen_target_sketch_cells[0].target_replicates != 10
    {
        return Err("sketch-sufficiency audit report has unexpected dimensions".into());
    }
    let rendered = report.to_json_pretty();
    if let Some(path) = std::env::args_os().nth(1) {
        fs::write(path, rendered)?;
    } else {
        io::stdout().write_all(rendered.as_bytes())?;
    }
    Ok(())
}
