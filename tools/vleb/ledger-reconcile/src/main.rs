use std::env;
use std::fs;
use std::io::{self, Write};
use std::process;

use vleb_ledger_reconcile::reconcile_csv;

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = env::args().skip(1);
    let flag = arguments.next();
    let path = arguments.next();
    let extra = arguments.next();

    let path = match (flag.as_deref(), path, extra) {
        (Some("--input"), Some(path), None) => path,
        _ => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "usage: ledger-reconcile --input <csv-path>",
            )
            .into());
        }
    };

    let input = fs::read_to_string(path)?;
    // Reconcile everything before writing stdout so errors cannot emit partial output.
    let report = reconcile_csv(&input)?;
    let output = report.to_json();

    let stdout = io::stdout();
    let mut lock = stdout.lock();
    lock.write_all(output.as_bytes())?;
    lock.flush()?;
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("ledger-reconcile: {error}");
        process::exit(2);
    }
}
