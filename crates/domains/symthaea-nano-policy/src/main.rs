// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Newline-delimited JSON adapter for research-only nanotechnology selection.

use std::io::{self, BufRead, Write};

fn main() -> io::Result<()> {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut output = stdout.lock();

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(line) => line,
            Err(error) => {
                eprintln!("nano-policy: unable to read stdin line: {error}");
                return Err(error);
            }
        };
        if line.trim().is_empty() {
            continue;
        }
        let response = symthaea_nano_policy::handle_line(&line);
        writeln!(output, "{response}")?;
    }
    output.flush()
}
