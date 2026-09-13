// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::io::{self, Read, Write};

fn main() -> io::Result<()> {
    let mut stdout = io::stdout().lock();
    stdout.write_all(b"exec_ready\n")?;
    stdout.flush()?;

    let mut release = [0u8; 1];
    io::stdin().read_exact(&mut release)?;
    if release[0] != 0x5a {
        return Err(io::Error::other("invalid exec target release byte"));
    }
    Ok(())
}
