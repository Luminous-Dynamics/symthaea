// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic inert child process for DE-001A2EI executor qualification.
//!
//! This binary performs no cosmology, likelihood evaluation, sampling,
//! minimization, optimization, networking, or package installation.

use std::env;
use std::fs;
use std::io::{self, Write};
use std::path::Path;
use std::process;

fn die(message: &str) -> ! {
    eprintln!("de001a-a2ei-fixture: {message}");
    process::exit(64);
}

fn increment_counter(path: &Path) -> Result<u64, String> {
    let current = if path.exists() {
        let text = fs::read_to_string(path)
            .map_err(|error| format!("read counter {}: {error}", path.display()))?;
        text.trim()
            .parse::<u64>()
            .map_err(|error| format!("parse counter {}: {error}", path.display()))?
    } else {
        0
    };
    let next = current
        .checked_add(1)
        .ok_or_else(|| "counter overflow".to_owned())?;
    fs::write(path, format!("{next}\n"))
        .map_err(|error| format!("write counter {}: {error}", path.display()))?;
    Ok(next)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 4 {
        die("usage: de001a-a2ei-fixture MODE COUNTER_PATH OUTPUT_PATH");
    }

    let mode = &args[1];
    let counter = Path::new(&args[2]);
    let output = Path::new(&args[3]);
    if !counter.is_absolute() || !output.is_absolute() {
        die("counter and output paths must be absolute");
    }

    let count = increment_counter(counter).unwrap_or_else(|error| die(&error));
    let payload = match mode.as_str() {
        "success" => b"fixture-result-success-v1\n".as_slice(),
        "nonzero" => b"fixture-result-nonzero-v1\n".as_slice(),
        _ => die("MODE must be success or nonzero"),
    };
    fs::write(output, payload)
        .unwrap_or_else(|error| die(&format!("write output {}: {error}", output.display())));

    io::stdout()
        .write_all(format!("fixture-stdout-v1 count={count}\n").as_bytes())
        .unwrap_or_else(|error| die(&format!("write stdout: {error}")));
    io::stderr()
        .write_all(format!("fixture-stderr-v1 mode={mode}\n").as_bytes())
        .unwrap_or_else(|error| die(&format!("write stderr: {error}")));

    if mode == "nonzero" {
        process::exit(23);
    }
}
