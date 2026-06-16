#![allow(dead_code)]
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Test Fixtures
//!
//! Shared test state management including temporary directories,
//! databases, sockets, and other resources that need cleanup.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use tempfile::TempDir;

use symthaea::databases::SqliteMemory;

// ============================================================================
// Temporary Database Fixture
// ============================================================================

/// A temporary database that cleans up after itself.
///
/// # Example
/// ```rust,ignore
/// let fixture = TempDatabase::new().expect("Failed to create database");
/// let db = fixture.db();
/// // Use db...
/// // Automatically cleaned up when fixture is dropped
/// ```
pub struct TempDatabase {
    _temp_dir: TempDir,
    db: SqliteMemory,
    path: PathBuf,
}

impl TempDatabase {
    /// Create a new temporary database
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let temp_dir = TempDir::new()?;
        let path = temp_dir.path().join("test.db");
        let db = SqliteMemory::new(&path)?;
        Ok(Self {
            _temp_dir: temp_dir,
            db,
            path,
        })
    }

    /// Create an in-memory database (no file, faster)
    pub fn in_memory() -> Result<SqliteMemory, Box<dyn std::error::Error>> {
        Ok(SqliteMemory::in_memory()?)
    }

    /// Get a reference to the database
    pub fn db(&self) -> &SqliteMemory {
        &self.db
    }

    /// Get the database path
    pub fn path(&self) -> &PathBuf {
        &self.path
    }

    /// Consume and return the database (temp dir still cleans up on drop)
    pub fn into_db(self) -> SqliteMemory {
        self.db
    }
}

// ============================================================================
// Temporary Directory Fixture
// ============================================================================

/// A temporary directory with helper methods
pub struct TempTestDir {
    dir: TempDir,
}

impl TempTestDir {
    /// Create a new temporary directory
    pub fn new() -> Result<Self, std::io::Error> {
        Ok(Self {
            dir: TempDir::new()?,
        })
    }

    /// Create a new temporary directory with a prefix
    pub fn with_prefix(prefix: &str) -> Result<Self, std::io::Error> {
        Ok(Self {
            dir: TempDir::with_prefix(prefix)?,
        })
    }

    /// Get a path within the temp directory
    pub fn path(&self, name: &str) -> PathBuf {
        self.dir.path().join(name)
    }

    /// Get the root path
    pub fn root(&self) -> &std::path::Path {
        self.dir.path()
    }

    /// Create a subdirectory
    pub fn create_subdir(&self, name: &str) -> Result<PathBuf, std::io::Error> {
        let path = self.path(name);
        std::fs::create_dir_all(&path)?;
        Ok(path)
    }

    /// Write content to a file in the temp directory
    pub fn write_file(&self, name: &str, content: &str) -> Result<PathBuf, std::io::Error> {
        let path = self.path(name);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&path, content)?;
        Ok(path)
    }
}

impl Default for TempTestDir {
    fn default() -> Self {
        Self::new().expect("Failed to create temp directory")
    }
}

// ============================================================================
// Socket Path Generator
// ============================================================================

static SOCKET_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Generate a unique socket path for testing
///
/// Uses PID and atomic counter to ensure uniqueness across threads.
///
/// # Example
/// ```rust,ignore
/// let socket_path = unique_socket_path();
/// // /tmp/symthaea-test-12345-0.sock
/// ```
pub fn unique_socket_path() -> PathBuf {
    let pid = std::process::id();
    let counter = SOCKET_COUNTER.fetch_add(1, Ordering::SeqCst);
    PathBuf::from(format!("/tmp/symthaea-test-{}-{}.sock", pid, counter))
}

/// Clean up a test socket file
pub fn cleanup_socket(path: &PathBuf) {
    let _ = std::fs::remove_file(path);
}

/// Socket fixture that cleans up on drop
pub struct TempSocket {
    path: PathBuf,
}

impl TempSocket {
    /// Create a new unique socket path
    pub fn new() -> Self {
        Self {
            path: unique_socket_path(),
        }
    }

    /// Get the socket path
    pub fn path(&self) -> &PathBuf {
        &self.path
    }

    /// Clean up the socket now (also happens on drop)
    pub fn cleanup(&self) {
        cleanup_socket(&self.path);
    }
}

impl Default for TempSocket {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for TempSocket {
    fn drop(&mut self) {
        cleanup_socket(&self.path);
    }
}

// ============================================================================
// Test Timer
// ============================================================================

/// Simple timer for measuring test performance
pub struct TestTimer {
    start: std::time::Instant,
    name: String,
}

impl TestTimer {
    /// Start a new timer
    pub fn start(name: &str) -> Self {
        Self {
            start: std::time::Instant::now(),
            name: name.to_string(),
        }
    }

    /// Get elapsed time in milliseconds
    pub fn elapsed_ms(&self) -> u128 {
        self.start.elapsed().as_millis()
    }

    /// Get elapsed time in microseconds
    pub fn elapsed_us(&self) -> u128 {
        self.start.elapsed().as_micros()
    }

    /// Stop and print elapsed time
    pub fn stop(self) -> u128 {
        let elapsed = self.elapsed_ms();
        eprintln!("[Timer] {}: {}ms", self.name, elapsed);
        elapsed
    }

    /// Assert elapsed time is under threshold
    pub fn assert_under_ms(&self, max_ms: u128, msg: &str) {
        let elapsed = self.elapsed_ms();
        assert!(
            elapsed < max_ms,
            "{}: took {}ms, expected < {}ms",
            msg,
            elapsed,
            max_ms
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temp_dir() {
        let temp = TempTestDir::new().unwrap();
        let path = temp.path("test.txt");
        assert!(path.ends_with("test.txt"));
    }

    #[test]
    fn test_unique_socket_paths() {
        let path1 = unique_socket_path();
        let path2 = unique_socket_path();
        assert_ne!(path1, path2, "Socket paths should be unique");
    }

    #[test]
    fn test_temp_socket_cleanup() {
        let socket = TempSocket::new();
        let path = socket.path().clone();

        // Create a dummy file at the socket path
        std::fs::write(&path, "test").ok();
        assert!(path.exists());

        // Drop should clean up
        drop(socket);
        assert!(!path.exists());
    }

    #[test]
    fn test_timer() {
        let timer = TestTimer::start("test");
        std::thread::sleep(std::time::Duration::from_millis(10));
        let elapsed = timer.stop();
        assert!(elapsed >= 10, "Should have measured at least 10ms");
    }
}