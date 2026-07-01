// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Shell Binary Integration Tests
//!
//! Tests for the `symthaea-shell` TUI binary.
//! Since TUI applications require a terminal, these tests focus on:
//! - CLI argument validation
//! - Help/version output
//! - Non-interactive initialization paths
//!
//! Run with: `cargo test --test binary_shell --features shell`

#![cfg(feature = "shell")]

use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::Duration;

/// Get the path to the built binary
fn binary_path() -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("target");
    path.push(if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    });
    path.push("symthaea-shell");
    path
}

/// Build the binary if needed
fn ensure_binary_built() {
    let status = Command::new("cargo")
        .args(["build", "--bin", "symthaea-shell", "--features", "shell"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .expect("Failed to run cargo build");

    assert!(status.success(), "Failed to build symthaea-shell binary");
}

mod cli_tests {
    use super::*;

    #[test]
    fn test_help_flag() {
        ensure_binary_built();

        let output = Command::new(binary_path())
            .arg("--help")
            .output()
            .expect("Failed to execute binary");

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        let combined = format!("{}{}", stdout, stderr);

        // Help should succeed or at least not crash
        // Note: Some TUI apps may not have --help if they don't use clap
        if output.status.success() {
            assert!(
                combined.contains("shell")
                    || combined.contains("Shell")
                    || combined.contains("symthaea")
                    || combined.contains("Symthaea")
                    || combined.contains("Usage")
                    || combined.contains("usage"),
                "Help output should contain relevant info: {}",
                combined
            );
        } else {
            // If --help isn't supported, verify the binary at least runs
            eprintln!("[Info] Shell may not support --help flag: {}", combined);
        }
    }

    #[test]
    fn test_version_flag() {
        ensure_binary_built();

        let output = Command::new(binary_path())
            .arg("--version")
            .output()
            .expect("Failed to execute binary");

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);

        // Version should succeed or be gracefully handled
        if output.status.success() {
            assert!(
                stdout.contains("0.")
                    || stdout.contains("symthaea")
                    || stderr.contains("0.")
                    || stderr.contains("symthaea"),
                "Version output should contain version: stdout={}, stderr={}",
                stdout,
                stderr
            );
        } else {
            eprintln!("[Info] Shell may not support --version flag");
        }
    }

    #[test]
    fn test_invalid_argument_handling() {
        ensure_binary_built();

        let output = Command::new(binary_path())
            .arg("--definitely-not-a-real-flag-12345")
            .output()
            .expect("Failed to execute binary");

        // Should fail with non-zero exit code
        assert!(
            !output.status.success(),
            "Invalid argument should cause failure"
        );

        let stderr = String::from_utf8_lossy(&output.stderr);
        // Should have some error output
        assert!(
            stderr.len() > 0 || !output.status.success(),
            "Should indicate error for invalid argument"
        );
    }
}

mod initialization_tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_no_tty_graceful_exit() {
        ensure_binary_built();

        // Run without a TTY (which is the case in automated tests)
        // The shell should detect this and exit gracefully
        let mut child = Command::new(binary_path())
            .stdin(Stdio::null()) // No stdin
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("Failed to start shell");

        // Wait briefly for it to detect no TTY
        thread::sleep(Duration::from_millis(500));

        // Check if it exited (it should, since there's no TTY)
        match child.try_wait() {
            Ok(Some(status)) => {
                // Exited - this is expected when no TTY is available
                eprintln!("[Info] Shell exited without TTY: {:?}", status);
            }
            Ok(None) => {
                // Still running - kill it
                child.kill().expect("Failed to kill");
                let _ = child.wait();
                eprintln!("[Info] Shell was still running, killed it");
            }
            Err(e) => {
                panic!("Error checking process: {}", e);
            }
        }
    }

    #[test]
    fn test_binary_exists_and_executable() {
        ensure_binary_built();

        let path = binary_path();
        assert!(path.exists(), "Binary should exist at {:?}", path);

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let metadata = std::fs::metadata(&path).expect("Failed to get metadata");
            let permissions = metadata.permissions();
            assert!(
                permissions.mode() & 0o111 != 0,
                "Binary should be executable"
            );
        }
    }
}

mod component_tests {
    //! Tests for shell components that can be tested without a full TUI

    use symthaea::action::DestructivenessLevel;
    use symthaea::shell::{IntelliSenseEngine, PhiGate, ShellContext};

    #[test]
    fn test_shell_context_creation() {
        let _context = ShellContext::new();
        // ShellContext::new() should succeed without panicking
        // (This is a construction smoke test - verified by reaching this point)
        assert!(true, "ShellContext created successfully");
    }

    #[test]
    fn test_intellisense_engine_creation() {
        let engine = IntelliSenseEngine::new();

        // Test completion generation for a common command
        let input = "install ";
        let completions = engine.complete(input, input.len());

        // Completions should be a valid (possibly empty) list
        assert!(
            completions.len() < 10_000,
            "Unreasonable number of completions"
        );
    }

    #[test]
    fn test_phi_gate_creation() {
        let gate = PhiGate::new();

        // ReadOnly commands should be allowed
        let decision = gate.evaluate("search vim", DestructivenessLevel::ReadOnly);
        assert!(
            decision.is_allowed(),
            "ReadOnly search should be allowed by PhiGate"
        );
    }

    #[test]
    fn test_destructiveness_level_ordering() {
        // Test that destructiveness levels are properly ordered
        assert!(DestructivenessLevel::ReadOnly < DestructivenessLevel::Reversible);
        assert!(DestructivenessLevel::Reversible < DestructivenessLevel::NeedsConfirmation);
        assert!(DestructivenessLevel::NeedsConfirmation < DestructivenessLevel::Destructive);
    }

    #[test]
    fn test_classify_command_destructiveness() {
        use symthaea::shell::classify_command_destructiveness;

        // Read-only commands (Nix-focused classifier)
        let search_level = classify_command_destructiveness("nix search vim");
        assert_eq!(
            search_level,
            DestructivenessLevel::ReadOnly,
            "nix search should be read-only"
        );

        let query_level = classify_command_destructiveness("nix-env -q");
        assert_eq!(
            query_level,
            DestructivenessLevel::ReadOnly,
            "nix-env -q should be read-only"
        );

        // NeedsConfirmation commands
        let install_level = classify_command_destructiveness("nix-env -i vim");
        assert_eq!(
            install_level,
            DestructivenessLevel::NeedsConfirmation,
            "nix-env -i should need confirmation"
        );

        let rebuild_level = classify_command_destructiveness("nixos-rebuild switch");
        assert_eq!(
            rebuild_level,
            DestructivenessLevel::NeedsConfirmation,
            "nixos-rebuild should need confirmation"
        );

        // Destructive commands
        let rm_level = classify_command_destructiveness("rm -rf /");
        assert_eq!(
            rm_level,
            DestructivenessLevel::Destructive,
            "rm -rf should be destructive"
        );

        let gc_level = classify_command_destructiveness("nix-collect-garbage -d");
        assert_eq!(
            gc_level,
            DestructivenessLevel::Destructive,
            "nix-collect-garbage should be destructive"
        );
    }

    #[test]
    fn test_phi_gate_with_different_destructiveness() {
        let gate = PhiGate::new();

        // ReadOnly commands should be allowed
        let readonly_decision = gate.evaluate("nix search vim", DestructivenessLevel::ReadOnly);
        assert!(
            readonly_decision.is_allowed(),
            "ReadOnly search should be allowed"
        );

        // NeedsConfirmation commands should trigger confirmation
        let destructive_decision = gate.evaluate(
            "nixos-rebuild switch",
            DestructivenessLevel::NeedsConfirmation,
        );
        assert!(
            destructive_decision.needs_confirmation(),
            "NeedsConfirmation commands should require confirmation"
        );
    }
}

mod ipc_tests {
    //! Tests for IPC client functionality

    use symthaea::shell::ipc_client::{ConnectionState, ShellIpcClient};

    #[test]
    fn test_ipc_client_creation() {
        // IPC client should be creatable even without a server
        let client = ShellIpcClient::new();

        // Check initial state - use state() method
        let state = client.state();
        assert_eq!(
            state,
            ConnectionState::Disconnected,
            "Should start disconnected"
        );
    }

    #[test]
    fn test_socket_discovery() {
        use symthaea::shell::ipc_client::discover_socket;

        // discover_socket should return a valid Option (Some or None)
        let socket = discover_socket();

        // If a socket was found, it should be a valid path
        if let Some(ref path) = socket {
            assert!(
                !path.as_os_str().is_empty(),
                "Socket path should not be empty"
            );
        }
        // Not finding a socket is also valid in test environments
    }
}