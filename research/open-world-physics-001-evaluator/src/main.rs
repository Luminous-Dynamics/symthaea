use open_world_physics_evaluator::run_reference_benchmark;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let report = run_reference_benchmark();
    if !report.verify_receipt() {
        return Err("generated report receipt failed self-check".into());
    }
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
