use cetb_rcaf::simulation::{run_common_mode_nuisance_sweep, InjectionRecoveryConfig};
use std::error::Error;
use std::fs;
use std::io::{self, Write};

fn main() -> Result<(), Box<dyn Error>> {
    let config = InjectionRecoveryConfig {
        monte_carlo_replicates: 40,
        ..InjectionRecoveryConfig::default()
    };
    let sweep = run_common_mode_nuisance_sweep(&config, &[0.0, 0.10, 0.20, 0.30])?;
    let rendered = sweep.to_json_pretty();

    if let Some(path) = std::env::args_os().nth(1) {
        fs::write(path, rendered)?;
    } else {
        io::stdout().write_all(rendered.as_bytes())?;
    }
    Ok(())
}
