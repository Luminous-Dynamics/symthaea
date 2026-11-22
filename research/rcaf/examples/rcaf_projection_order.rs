use cetb_rcaf::operator_composition::run_projection_order_challenge;
use std::error::Error;
use std::fs;
use std::io::{self, Write};

fn main() -> Result<(), Box<dyn Error>> {
    let observations = vec![
        vec![1.0, 0.0],
        vec![1.0, 1.0],
        vec![0.0, 2.0],
        vec![0.0, -1.0],
    ];
    let nuisance = vec![vec![1.0], vec![0.0], vec![0.0], vec![0.0]];
    let removed_filter = vec![vec![1.0], vec![1.0], vec![0.0], vec![0.0]];
    let report = run_projection_order_challenge(&observations, &nuisance, &removed_filter)?;

    let rendered = format!(
        concat!(
            "{{\n",
            "  \"schema_version\": \"{}\",\n",
            "  \"source_status\": \"synthetic_projection_fixture\",\n",
            "  \"time_points\": {},\n",
            "  \"signal_columns\": {},\n",
            "  \"nuisance_design_columns\": {},\n",
            "  \"filter_design_columns\": {},\n",
            "  \"nuisance_design_rank\": {},\n",
            "  \"filter_design_rank\": {},\n",
            "  \"joint_design_rank\": {},\n",
            "  \"dependent_columns_dropped\": {},\n",
            "  \"projection_operator_commutator_ratio\": {:.12},\n",
            "  \"nuisance_then_filter_nuisance_leakage\": {:.12},\n",
            "  \"filter_then_nuisance_nuisance_leakage\": {:.12},\n",
            "  \"nuisance_then_filter_filter_leakage\": {:.12},\n",
            "  \"filter_then_nuisance_filter_leakage\": {:.12},\n",
            "  \"joint_nuisance_leakage\": {:.12},\n",
            "  \"joint_filter_leakage\": {:.12},\n",
            "  \"order_difference_ratio\": {:.12},\n",
            "  \"nuisance_then_filter_vs_joint_ratio\": {:.12},\n",
            "  \"filter_then_nuisance_vs_joint_ratio\": {:.12},\n",
            "  \"claim_limits\": [\n",
            "    \"synthetic projection fixture only\",\n",
            "    \"not a full temporal filter or fMRI preprocessing implementation\",\n",
            "    \"does not qualify an empirical pipeline or biological claim\"\n",
            "  ]\n",
            "}}\n"
        ),
        report.schema_version,
        report.time_points,
        report.signal_columns,
        report.nuisance_design_columns,
        report.filter_design_columns,
        report.nuisance_design_rank,
        report.filter_design_rank,
        report.joint_design_rank,
        report.dependent_columns_dropped,
        report.projection_operator_commutator_ratio,
        report.nuisance_then_filter_nuisance_leakage,
        report.filter_then_nuisance_nuisance_leakage,
        report.nuisance_then_filter_filter_leakage,
        report.filter_then_nuisance_filter_leakage,
        report.joint_nuisance_leakage,
        report.joint_filter_leakage,
        report.order_difference_ratio,
        report.nuisance_then_filter_vs_joint_ratio,
        report.filter_then_nuisance_vs_joint_ratio,
    );

    if let Some(path) = std::env::args_os().nth(1) {
        fs::write(path, rendered)?;
    } else {
        io::stdout().write_all(rendered.as_bytes())?;
    }
    Ok(())
}
