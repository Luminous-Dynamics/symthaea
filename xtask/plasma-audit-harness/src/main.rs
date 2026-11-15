use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let repository_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let report = xtask::plasma_energy_audit::run_audit(
        &repository_root.join("data/research/plasma_energy_reference_cases_v0_1.json"),
        &repository_root.join("data/research/hts_fusion_magnet_reference_cases_v0_1.json"),
        &repository_root.join("data/research/thermal_store_accounting_fixtures_v0_1.json"),
    )?;
    xtask::plasma_energy_audit::print_report(&report, true)
}
