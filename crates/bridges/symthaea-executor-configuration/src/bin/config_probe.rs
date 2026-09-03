// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Small qualification probe used only to exercise the real measurement path
//! inside a booted systemd/Nix VM.

use std::process::ExitCode;

use symthaea_executor_configuration::{
    EXECUTOR_CONFIGURATION_SCHEMA_VERSION, ExecutorConfigurationPolicyV1,
    measure_current_executor_configuration,
};

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let Some(unit) = args.next() else {
        eprintln!("usage: config_probe <executor.service>");
        return ExitCode::from(64);
    };
    if args.next().is_some() {
        eprintln!("usage: config_probe <executor.service>");
        return ExitCode::from(64);
    }

    let policy = ExecutorConfigurationPolicyV1 {
        schema_version: EXECUTOR_CONFIGURATION_SCHEMA_VERSION,
        expected_unit: unit,
        require_nix_store_fragment: true,
        require_no_new_privileges: true,
        require_protect_system_strict: true,
        require_protect_home: true,
        require_private_tmp: true,
        require_private_devices: true,
        require_memory_deny_write_execute: true,
        require_lock_personality: true,
        require_protect_kernel_tunables: true,
        require_protect_kernel_modules: true,
        require_protect_control_groups: true,
        require_restrict_suid_sgid: true,
        require_restrict_realtime: true,
    };

    match measure_current_executor_configuration(&policy)
        .and_then(|verified| verified.configuration_digest())
    {
        Ok(digest) => {
            println!("configuration_digest={}", hex(&digest.0));
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("configuration_error={error}");
            ExitCode::FAILURE
        }
    }
}

fn hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}
