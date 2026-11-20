use std::path::PathBuf;

fn main() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let mut output_dir = None;
    let mut include_ground_truth = false;

    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--output-dir" => {
                let value = args.next().ok_or_else(|| "--output-dir requires a path".to_owned())?;
                output_dir = Some(PathBuf::from(value));
            }
            "--include-ground-truth" => include_ground_truth = true,
            "--help" | "-h" => {
                println!("Usage: symmetry-colt-regression-reference --output-dir PATH [--include-ground-truth]");
                return Ok(());
            }
            _ => return Err(format!("unknown argument {argument}")),
        }
    }

    let output_dir = output_dir.ok_or_else(|| "--output-dir is required".to_owned())?;
    symmetry_colt_regression_reference::write_corpus(&output_dir, include_ground_truth)?;
    let receipt = std::fs::read_to_string(output_dir.join("generation_receipt.json"))
        .map_err(|error| format!("could not read generation receipt: {error}"))?;
    println!("{receipt}");
    Ok(())
}
