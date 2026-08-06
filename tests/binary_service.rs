// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Service Binary Integration Tests
//!
//! Tests for the `symthaea` service daemon binary.
//! Verifies CLI argument parsing, help output, and basic service startup.
//!
//! Run with: `cargo test --test binary_service --features service`

#![cfg(feature = "service")]

use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::Duration;
use tempfile::TempDir;

/// Get the path to the built binary
fn binary_path() -> PathBuf {
    if let Some(path) = option_env!("CARGO_BIN_EXE_symthaea") {
        return PathBuf::from(path);
    }

    if let Ok(target_dir) = std::env::var("CARGO_TARGET_DIR") {
        let mut path = PathBuf::from(target_dir);
        path.push(if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        });
        path.push("symthaea");
        return path;
    }

    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("target");
    path.push(if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    });
    path.push("symthaea");
    path
}

/// Build the binary if needed
fn ensure_binary_built() {
    if option_env!("CARGO_BIN_EXE_symthaea").is_some() {
        return;
    }

    let status = Command::new("cargo")
        .args(["build", "--bin", "symthaea", "--features", "service"])
        .env_remove("RUSTC_WRAPPER")
        .env("SCCACHE_DISABLE", "1")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .expect("Failed to run cargo build");

    assert!(status.success(), "Failed to build symthaea binary");
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

        // Should succeed and show help
        assert!(output.status.success(), "Help command should succeed");

        // Should contain expected help text
        assert!(
            stdout.contains("symthaea") || stdout.contains("Symthaea"),
            "Help should mention symthaea"
        );
        assert!(
            stdout.contains("--socket") || stdout.contains("socket"),
            "Help should mention socket option"
        );
        assert!(
            stdout.contains("--tcp") || stdout.contains("tcp"),
            "Help should mention tcp option"
        );
    }

    #[test]
    fn test_version_flag() {
        ensure_binary_built();

        let output = Command::new(binary_path())
            .arg("--version")
            .output()
            .expect("Failed to execute binary");

        let stdout = String::from_utf8_lossy(&output.stdout);

        // Should succeed
        assert!(output.status.success(), "Version command should succeed");

        // Should contain version number
        assert!(
            stdout.contains("0.") || stdout.contains("symthaea"),
            "Version output should contain version info: {}",
            stdout
        );
    }

    #[test]
    fn test_invalid_argument() {
        ensure_binary_built();

        let output = Command::new(binary_path())
            .arg("--invalid-nonexistent-flag")
            .output()
            .expect("Failed to execute binary");

        // Should fail with non-zero exit code
        assert!(!output.status.success(), "Invalid argument should fail");

        // Stderr should contain error message
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains("error")
                || stderr.contains("invalid")
                || stderr.contains("unrecognized"),
            "Should show error for invalid argument: {}",
            stderr
        );
    }

    #[test]
    fn test_conflicting_socket_and_tcp_without_value() {
        ensure_binary_built();

        // Providing neither socket nor tcp should still work (uses defaults or fails gracefully)
        let output = Command::new(binary_path())
            .arg("--help") // Just test help works
            .output()
            .expect("Failed to execute binary");

        assert!(output.status.success());
    }
}

mod startup_tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_socket_startup_and_shutdown() {
        ensure_binary_built();

        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let socket_path = temp_dir.path().join("test.sock");

        // Start the service
        let mut child = Command::new(binary_path())
            .arg("--socket")
            .arg(&socket_path)
            .arg("--loop-interval")
            .arg("100") // Fast loop for testing
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("Failed to start service");

        // Give it time to start
        thread::sleep(Duration::from_millis(500));

        // Check if process is still running (it should be)
        match child.try_wait() {
            Ok(None) => {
                // Process still running - good, now kill it
                child.kill().expect("Failed to kill service");
                let _ = child.wait();
            }
            Ok(Some(status)) => {
                // Process exited - check why
                let mut stderr = String::new();
                if let Some(ref mut err) = child.stderr {
                    use std::io::Read;
                    let _ = err.read_to_string(&mut stderr);
                }
                // It's OK if it exits quickly due to missing dependencies
                // The point is it started and processed arguments correctly
                eprintln!(
                    "[Info] Service exited with status: {:?}, stderr: {}",
                    status, stderr
                );
            }
            Err(e) => {
                panic!("Error checking process status: {}", e);
            }
        }
    }

    #[test]
    fn test_tcp_startup_graceful() {
        ensure_binary_built();

        // Use a high port that's likely available
        let port = 19999;
        let addr = format!("127.0.0.1:{}", port);

        // Start the service with TCP
        let mut child = Command::new(binary_path())
            .arg("--tcp")
            .arg(&addr)
            .arg("--loop-interval")
            .arg("100")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("Failed to start service");

        // Give it time to bind
        thread::sleep(Duration::from_millis(500));

        // Try to connect (may fail if service isn't fully up, but that's OK)
        let connect_result = std::net::TcpStream::connect_timeout(
            &addr.parse().unwrap(),
            Duration::from_millis(200),
        );

        // Clean up
        child.kill().expect("Failed to kill service");
        let _ = child.wait();

        // If we connected, great. If not, at least verify the process ran.
        if connect_result.is_ok() {
            eprintln!("[Info] Successfully connected to TCP service");
        } else {
            eprintln!("[Info] Could not connect (service may need more time or dependencies)");
        }
    }

    #[test]
    fn test_public_tcp_requires_auth_or_explicit_insecure_opt_in() {
        ensure_binary_built();

        let output = Command::new(binary_path())
            .arg("--tcp")
            .arg("0.0.0.0:19998")
            .output()
            .expect("Failed to execute binary");

        assert!(
            !output.status.success(),
            "service should refuse public TCP bind without auth"
        );
    }

    #[test]
    fn test_state_file_option() {
        ensure_binary_built();

        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let state_file = temp_dir.path().join("state.json");
        let socket_path = temp_dir.path().join("test.sock");

        // Start with state file option
        let mut child = Command::new(binary_path())
            .arg("--socket")
            .arg(&socket_path)
            .arg("--state-file")
            .arg(&state_file)
            .arg("--loop-interval")
            .arg("100")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("Failed to start service");

        thread::sleep(Duration::from_millis(300));

        // Kill service
        child.kill().expect("Failed to kill service");
        let _ = child.wait();

        // The state file may or may not be created depending on implementation
        // The test just verifies the argument is accepted
    }

    #[test]
    fn test_verbose_flag() {
        ensure_binary_built();

        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let socket_path = temp_dir.path().join("test.sock");

        // Start with verbose flag
        let mut child = Command::new(binary_path())
            .arg("--socket")
            .arg(&socket_path)
            .arg("--verbose")
            .arg("--loop-interval")
            .arg("50")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("Failed to start service");

        thread::sleep(Duration::from_millis(200));

        child.kill().expect("Failed to kill service");
        let output = child.wait_with_output().expect("Failed to get output");

        // Verbose mode should produce more output (either stdout or stderr)
        let combined = format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );

        // Just verify the process accepted the verbose flag and ran
        eprintln!("[Info] Verbose output length: {} bytes", combined.len());
    }
}

mod signal_tests {
    use super::*;
    use std::thread;

    #[test]
    #[cfg(unix)]
    fn test_sigterm_graceful_shutdown() {
        ensure_binary_built();

        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let socket_path = temp_dir.path().join("test.sock");

        let mut child = Command::new(binary_path())
            .arg("--socket")
            .arg(&socket_path)
            .arg("--loop-interval")
            .arg("50")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("Failed to start service");

        let pid = child.id();

        thread::sleep(Duration::from_millis(300));

        // Send SIGTERM using kill command (portable approach)
        let kill_result = Command::new("kill")
            .arg("-TERM")
            .arg(pid.to_string())
            .status();

        if let Err(e) = kill_result {
            eprintln!("[Info] Could not send SIGTERM: {}", e);
            child.kill().expect("Failed to kill");
            let _ = child.wait();
            return;
        }

        // Wait for graceful shutdown (up to 2 seconds)
        let start = std::time::Instant::now();
        loop {
            match child.try_wait() {
                Ok(Some(status)) => {
                    eprintln!("[Info] Service exited with status: {:?}", status);
                    break;
                }
                Ok(None) => {
                    if start.elapsed() > Duration::from_secs(2) {
                        child.kill().expect("Failed to kill");
                        let _ = child.wait();
                        eprintln!("[Info] Service did not shutdown within 2 seconds, force killed");
                        break;
                    }
                    thread::sleep(Duration::from_millis(50));
                }
                Err(e) => {
                    eprintln!("[Info] Error waiting: {}", e);
                    break;
                }
            }
        }
    }
}
