// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! F3: File Watcher for Config Changes
//!
//! Monitors NixOS configuration files for changes and triggers
//! automatic diff updates and rebuild suggestions.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, RwLock};
use std::thread;
use std::time::{Duration, Instant};

/// Event types from file watcher
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WatchEventKind {
    /// File was created
    Create,
    /// File was modified
    Modify,
    /// File was deleted
    Delete,
    /// File was renamed (old_path, new_path)
    Rename,
    /// Access error
    Error(String),
}

/// A file system event
#[derive(Debug, Clone)]
pub struct WatchEvent {
    /// Path that changed
    pub path: PathBuf,
    /// Kind of change
    pub kind: WatchEventKind,
    /// Timestamp of event
    pub timestamp: Instant,
}

/// Configuration file watcher
pub struct ConfigWatcher {
    /// Paths being watched (shared with watcher thread)
    watched_paths: Arc<RwLock<HashSet<Arc<PathBuf>>>>,
    /// Event receiver
    event_rx: Option<Receiver<WatchEvent>>,
    /// Event sender (for thread)
    event_tx: Sender<WatchEvent>,
    /// Debounce duration
    debounce: Duration,
    /// Whether watcher is running
    running: bool,
    /// Last modification times
    last_modified: std::collections::HashMap<PathBuf, std::time::SystemTime>,
    /// Shutdown signal for watcher thread
    shutdown: Arc<AtomicBool>,
}

impl ConfigWatcher {
    /// Create a new config watcher
    pub fn new() -> Self {
        let (tx, rx) = channel();
        Self {
            watched_paths: Arc::new(RwLock::new(HashSet::new())),
            event_rx: Some(rx),
            event_tx: tx,
            debounce: Duration::from_millis(500),
            running: false,
            last_modified: std::collections::HashMap::new(),
            shutdown: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Set debounce duration
    pub fn with_debounce(mut self, debounce: Duration) -> Self {
        self.debounce = debounce;
        self
    }

    /// Add a path to watch
    pub fn watch(&mut self, path: impl AsRef<Path>) -> Result<(), WatchError> {
        let path = path.as_ref().to_path_buf();

        if !path.exists() {
            return Err(WatchError::PathNotFound(path));
        }

        // Record initial modification time
        if let Ok(metadata) = std::fs::metadata(&path) {
            if let Ok(modified) = metadata.modified() {
                self.last_modified.insert(path.clone(), modified);
            }
        }

        self.watched_paths
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .insert(Arc::new(path));

        Ok(())
    }

    /// Add standard NixOS config paths
    pub fn watch_nixos_config(&mut self) -> Result<(), WatchError> {
        let paths = [
            "/etc/nixos/configuration.nix",
            "/etc/nixos/hardware-configuration.nix",
            "/etc/nixos/flake.nix",
            "/etc/nixos/flake.lock",
        ];

        for path in &paths {
            if Path::new(path).exists() {
                self.watch(path)?;
            }
        }

        // Watch any .nix files in /etc/nixos/
        if let Ok(entries) = std::fs::read_dir("/etc/nixos") {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().map(|e| e == "nix").unwrap_or(false) {
                    self.watch(&path)?;
                }
            }
        }

        Ok(())
    }

    /// Remove a path from watch
    pub fn unwatch(&mut self, path: impl AsRef<Path>) {
        let path_buf = path.as_ref().to_path_buf();
        self.watched_paths
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .retain(|p| **p != path_buf);
        self.last_modified.remove(path.as_ref());
    }

    /// Start watching (spawns background thread)
    pub fn start(&mut self) -> Result<(), WatchError> {
        if self.running {
            return Ok(());
        }

        // Reset shutdown flag in case of restart
        self.shutdown.store(false, Ordering::Relaxed);

        let paths = Arc::clone(&self.watched_paths);
        let tx = self.event_tx.clone();
        let debounce = self.debounce;
        let mut last_modified = self.last_modified.clone();
        let shutdown = Arc::clone(&self.shutdown);

        self.running = true;

        thread::spawn(move || {
            let poll_interval = Duration::from_millis(100);

            while !shutdown.load(Ordering::Relaxed) {
                thread::sleep(poll_interval);

                // Get a snapshot of paths under read lock
                let paths_snapshot: Vec<Arc<PathBuf>> = paths
                    .read()
                    .unwrap_or_else(|e| e.into_inner())
                    .iter()
                    .cloned()
                    .collect();

                for path in paths_snapshot {
                    // Check shutdown between iterations
                    if shutdown.load(Ordering::Relaxed) {
                        return;
                    }

                    if let Ok(metadata) = std::fs::metadata(&*path) {
                        if let Ok(modified) = metadata.modified() {
                            let changed = last_modified
                                .get(&*path)
                                .map(|&last| modified > last)
                                .unwrap_or(true);

                            if changed {
                                // Debounce
                                thread::sleep(debounce);

                                // Check shutdown after debounce
                                if shutdown.load(Ordering::Relaxed) {
                                    return;
                                }

                                // Recheck after debounce
                                if let Ok(metadata2) = std::fs::metadata(&*path) {
                                    if let Ok(modified2) = metadata2.modified() {
                                        if modified2 == modified {
                                            // File has stabilized
                                            last_modified.insert((*path).clone(), modified2);

                                            let event = WatchEvent {
                                                path: (*path).clone(),
                                                kind: WatchEventKind::Modify,
                                                timestamp: Instant::now(),
                                            };

                                            if tx.send(event).is_err() {
                                                // Receiver dropped, stop watching
                                                return;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        // File deleted?
                        if last_modified.contains_key(&*path) {
                            last_modified.remove(&*path);

                            let event = WatchEvent {
                                path: (*path).clone(),
                                kind: WatchEventKind::Delete,
                                timestamp: Instant::now(),
                            };

                            if tx.send(event).is_err() {
                                return;
                            }
                        }
                    }
                }
            }
        });

        Ok(())
    }

    /// Stop the watcher thread
    pub fn stop(&self) {
        self.shutdown.store(true, Ordering::Relaxed);
    }

    /// Take the event receiver (can only be called once)
    pub fn take_receiver(&mut self) -> Option<Receiver<WatchEvent>> {
        self.event_rx.take()
    }

    /// Poll for events (non-blocking)
    pub fn poll(&self) -> Vec<WatchEvent> {
        let mut events = Vec::new();
        if let Some(ref rx) = self.event_rx {
            while let Ok(event) = rx.try_recv() {
                events.push(event);
            }
        }
        events
    }

    /// Get watched paths (returns a clone of the current paths)
    pub fn watched_paths(&self) -> HashSet<PathBuf> {
        self.watched_paths
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .map(|p| (**p).clone())
            .collect()
    }

    /// Check if a path is being watched
    pub fn is_watching(&self, path: impl AsRef<Path>) -> bool {
        let path_buf = path.as_ref().to_path_buf();
        self.watched_paths
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .any(|p| **p == path_buf)
    }
}

impl Default for ConfigWatcher {
    fn default() -> Self {
        Self::new()
    }
}

/// Watch errors
#[derive(Debug, Clone)]
pub enum WatchError {
    PathNotFound(PathBuf),
    PermissionDenied(PathBuf),
    IoError(String),
    AlreadyRunning,
}

impl std::fmt::Display for WatchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PathNotFound(p) => write!(f, "Path not found: {}", p.display()),
            Self::PermissionDenied(p) => write!(f, "Permission denied: {}", p.display()),
            Self::IoError(e) => write!(f, "IO error: {e}"),
            Self::AlreadyRunning => write!(f, "Watcher already running"),
        }
    }
}

impl std::error::Error for WatchError {}

/// Config change detector with diff support
pub struct ConfigChangeDetector {
    watcher: ConfigWatcher,
    /// Previous content for diff
    previous_content: std::collections::HashMap<PathBuf, String>,
    /// Pending changes
    pending_changes: Vec<ConfigChange>,
}

/// A detected configuration change
#[derive(Debug, Clone)]
pub struct ConfigChange {
    pub path: PathBuf,
    pub kind: WatchEventKind,
    pub timestamp: Instant,
    pub diff: Option<String>,
    pub lines_added: usize,
    pub lines_removed: usize,
}

impl ConfigChangeDetector {
    pub fn new() -> Self {
        Self {
            watcher: ConfigWatcher::new(),
            previous_content: std::collections::HashMap::new(),
            pending_changes: Vec::new(),
        }
    }

    /// Initialize with NixOS config watching
    pub fn with_nixos_config(mut self) -> Result<Self, WatchError> {
        self.watcher.watch_nixos_config()?;

        // Load initial content
        for path in self.watcher.watched_paths() {
            if let Ok(content) = std::fs::read_to_string(&path) {
                self.previous_content.insert(path, content);
            }
        }

        Ok(self)
    }

    /// Start monitoring
    pub fn start(&mut self) -> Result<(), WatchError> {
        self.watcher.start()
    }

    /// Check for changes and compute diffs
    pub fn check_changes(&mut self) -> Vec<ConfigChange> {
        let events = self.watcher.poll();
        let mut changes = Vec::new();

        for event in events {
            let change = match event.kind {
                WatchEventKind::Modify => {
                    // Compute diff
                    let (diff, added, removed) =
                        if let Ok(new_content) = std::fs::read_to_string(&event.path) {
                            let old_content = self
                                .previous_content
                                .get(&event.path)
                                .map(|s| s.as_str())
                                .unwrap_or("");

                            let diff = compute_simple_diff(old_content, &new_content);
                            let added = new_content
                                .lines()
                                .count()
                                .saturating_sub(old_content.lines().count());
                            let removed = old_content
                                .lines()
                                .count()
                                .saturating_sub(new_content.lines().count());

                            self.previous_content
                                .insert(event.path.clone(), new_content);

                            (Some(diff), added, removed)
                        } else {
                            (None, 0, 0)
                        };

                    ConfigChange {
                        path: event.path,
                        kind: event.kind,
                        timestamp: event.timestamp,
                        diff,
                        lines_added: added,
                        lines_removed: removed,
                    }
                }
                WatchEventKind::Delete => {
                    self.previous_content.remove(&event.path);
                    ConfigChange {
                        path: event.path,
                        kind: event.kind,
                        timestamp: event.timestamp,
                        diff: None,
                        lines_added: 0,
                        lines_removed: 0,
                    }
                }
                _ => ConfigChange {
                    path: event.path,
                    kind: event.kind,
                    timestamp: event.timestamp,
                    diff: None,
                    lines_added: 0,
                    lines_removed: 0,
                },
            };

            changes.push(change);
        }

        self.pending_changes.extend(changes.clone());
        changes
    }

    /// Get pending changes
    pub fn pending_changes(&self) -> &[ConfigChange] {
        &self.pending_changes
    }

    /// Clear pending changes
    pub fn clear_pending(&mut self) {
        self.pending_changes.clear();
    }

    /// Has changes since last rebuild
    pub fn has_changes(&self) -> bool {
        !self.pending_changes.is_empty()
    }
}

impl Default for ConfigChangeDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// Simple unified diff computation
fn compute_simple_diff(old: &str, new: &str) -> String {
    let old_lines: Vec<&str> = old.lines().collect();
    let new_lines: Vec<&str> = new.lines().collect();

    let mut diff = String::new();

    // Very simple diff - just show what's different
    let max_lines = old_lines.len().max(new_lines.len());

    for i in 0..max_lines {
        let old_line = old_lines.get(i);
        let new_line = new_lines.get(i);

        match (old_line, new_line) {
            (Some(o), Some(n)) if o != n => {
                diff.push_str(&format!("-{o}\n"));
                diff.push_str(&format!("+{n}\n"));
            }
            (Some(o), None) => {
                diff.push_str(&format!("-{o}\n"));
            }
            (None, Some(n)) => {
                diff.push_str(&format!("+{n}\n"));
            }
            _ => {}
        }
    }

    diff
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_watch_event_kinds() {
        let event = WatchEvent {
            path: PathBuf::from("/test"),
            kind: WatchEventKind::Modify,
            timestamp: Instant::now(),
        };

        assert_eq!(event.kind, WatchEventKind::Modify);
    }

    #[test]
    fn test_simple_diff() {
        let old = "line1\nline2\nline3";
        let new = "line1\nmodified\nline3\nline4";

        let diff = compute_simple_diff(old, new);
        assert!(diff.contains("-line2"));
        assert!(diff.contains("+modified"));
        assert!(diff.contains("+line4"));
    }

    #[test]
    fn test_watcher_creation() {
        let watcher = ConfigWatcher::new();
        assert!(watcher.watched_paths().is_empty());

        // Can't test actual file watching without temp files
    }
}
