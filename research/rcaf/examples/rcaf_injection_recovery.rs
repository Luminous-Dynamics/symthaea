use cetb_rcaf::simulation::{run_injection_recovery, InjectionRecoveryConfig};
use std::error::Error;
use std::fs;
use std::io::{self, Write};

fn main() -> Result<(), Box<dyn Error>> {
    let config = InjectionRecoveryConfig::default();
    let report = run_injection_recovery(&config)?;
    let rendered = report.to_json_pretty();

    if let Some(path) = std::env::args_os().nth(1) {
        fs::write(path, rendered)?;
    } else {
        io::stdout().write_all(rendered.as_bytes())?;
    }
    Ok(())
}
