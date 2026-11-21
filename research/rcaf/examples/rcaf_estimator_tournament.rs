use cetb_rcaf::simulation::{run_estimator_tournament, InjectionRecoveryConfig};
use std::error::Error;
use std::fs;
use std::io::{self, Write};

fn main() -> Result<(), Box<dyn Error>> {
    let mut config = InjectionRecoveryConfig::default();
    config.monte_carlo_replicates = 40;
    let tournament = run_estimator_tournament(&config)?;
    let rendered = tournament.to_json_pretty();

    if let Some(path) = std::env::args_os().nth(1) {
        fs::write(path, rendered)?;
    } else {
        io::stdout().write_all(rendered.as_bytes())?;
    }
    Ok(())
}
