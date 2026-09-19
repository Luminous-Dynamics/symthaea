// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

#[cfg(not(unix))]
compile_error!("symthaea-process-stdin-launcher currently requires Unix");

#[cfg(unix)]
fn main() {
    use sha2::{Digest, Sha256};
    use std::fs::File;
    use std::io::{Read, Seek, SeekFrom};
    use std::os::unix::process::CommandExt;
    use std::process::{Command, Stdio};

    let mut args = std::env::args();
    let _program = args.next();
    let Some(stdin_path) = args.next() else {
        fail(64, "missing stdin path");
    };
    let Some(expected_sha) = args.next() else {
        fail(64, "missing stdin SHA-256");
    };
    let Some(expected_bytes_text) = args.next() else {
        fail(64, "missing stdin byte count");
    };
    let Some(target) = args.next() else {
        fail(64, "missing target command");
    };
    if expected_sha.len() != 64 || !expected_sha.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        fail(64, "invalid stdin SHA-256");
    }
    let expected_bytes: u64 = expected_bytes_text
        .parse()
        .unwrap_or_else(|_| fail(64, "invalid stdin byte count"));

    let mut file = File::open(&stdin_path)
        .unwrap_or_else(|error| fail(66, &format!("failed to open stdin file: {error}")));
    let mut digest = Sha256::new();
    let mut observed_bytes = 0_u64;
    let mut chunk = [0_u8; 1024 * 1024];
    loop {
        let read = file
            .read(&mut chunk)
            .unwrap_or_else(|error| fail(74, &format!("failed to read stdin file: {error}")));
        if read == 0 {
            break;
        }
        digest.update(&chunk[..read]);
        observed_bytes = observed_bytes
            .checked_add(read as u64)
            .unwrap_or_else(|| fail(74, "stdin byte count overflow"));
    }
    let observed_sha = format!("{:x}", digest.finalize());
    if !observed_sha.eq_ignore_ascii_case(&expected_sha) || observed_bytes != expected_bytes {
        fail(
            65,
            &format!(
                "stdin identity mismatch: expected sha={expected_sha} bytes={expected_bytes}, observed sha={observed_sha} bytes={observed_bytes}"
            ),
        );
    }
    file.seek(SeekFrom::Start(0))
        .unwrap_or_else(|error| fail(74, &format!("failed to rewind stdin file: {error}")));

    let error = Command::new(target)
        .args(args)
        .stdin(Stdio::from(file))
        .exec();
    fail(126, &format!("target exec failed: {error}"));
}

#[cfg(unix)]
fn fail(code: i32, message: &str) -> ! {
    eprintln!("symthaea-process-stdin-launcher: {message}");
    std::process::exit(code);
}
