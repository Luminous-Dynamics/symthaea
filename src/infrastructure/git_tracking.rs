// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! I1: Git Tracking for Config Changes
//!
//! Automatically tracks NixOS configuration changes in git
//! with semantic commit messages based on HDC analysis.

use std::path::PathBuf;
use std::process::{Command, Output};

/// Git repository tracker for NixOS configs
pub struct GitTracker {
    /// Repository root path
    repo_path: PathBuf,
    /// Auto-commit enabled
    auto_commit: bool,
    /// Branch for config changes
    config_branch: String,
    /// Commit author name
    author_name: String,
    /// Commit author email
    author_email: String,
}

impl GitTracker {
    /// Create a new git tracker
    pub fn new(repo_path: impl Into<PathBuf>) -> Self {
        Self {
            repo_path: repo_path.into(),
            auto_commit: true,
            config_branch: "main".to_string(),
            author_name: "Symthaea".to_string(),
            author_email: "symthaea@localhost".to_string(),
        }
    }

    /// Set auto-commit behavior
    pub fn with_auto_commit(mut self, enabled: bool) -> Self {
        self.auto_commit = enabled;
        self
    }

    /// Set branch for config changes
    pub fn with_branch(mut self, branch: impl Into<String>) -> Self {
        self.config_branch = branch.into();
        self
    }

    /// Set commit author
    pub fn with_author(mut self, name: impl Into<String>, email: impl Into<String>) -> Self {
        self.author_name = name.into();
        self.author_email = email.into();
        self
    }

    /// Initialize git repository if needed
    pub fn init(&self) -> Result<(), GitError> {
        if !self.repo_path.join(".git").exists() {
            self.run_git(&["init"])?;
            self.run_git(&["config", "user.name", &self.author_name])?;
            self.run_git(&["config", "user.email", &self.author_email])?;
        }
        Ok(())
    }

    /// Check if path is a git repository
    pub fn is_repo(&self) -> bool {
        self.repo_path.join(".git").exists()
    }

    /// Get repository status
    pub fn status(&self) -> Result<RepoStatus, GitError> {
        let output = self.run_git(&["status", "--porcelain"])?;
        let stdout = String::from_utf8_lossy(&output.stdout);

        let mut modified = Vec::new();
        let mut added = Vec::new();
        let mut deleted = Vec::new();
        let mut untracked = Vec::new();

        for line in stdout.lines() {
            if line.len() < 3 {
                continue;
            }

            let status = &line[0..2];
            let file = line[3..].to_string();

            match status {
                "M " | " M" | "MM" => modified.push(file),
                "A " | "AM" => added.push(file),
                "D " | " D" => deleted.push(file),
                "??" => untracked.push(file),
                "R " => modified.push(file), // Renamed
                _ => {}
            }
        }

        let has_changes = !modified.is_empty() || !added.is_empty() || !deleted.is_empty();

        Ok(RepoStatus {
            modified,
            added,
            deleted,
            untracked,
            has_changes,
        })
    }

    /// Stage files for commit
    pub fn stage(&self, files: &[&str]) -> Result<(), GitError> {
        let mut args = vec!["add"];
        args.extend(files);
        self.run_git(&args)?;
        Ok(())
    }

    /// Stage all changes
    pub fn stage_all(&self) -> Result<(), GitError> {
        self.run_git(&["add", "-A"])?;
        Ok(())
    }

    /// Create a commit with semantic message
    pub fn commit(&self, message: &str) -> Result<CommitInfo, GitError> {
        let output = self.run_git(&[
            "commit",
            "-m",
            message,
            "--author",
            &format!("{} <{}>", self.author_name, self.author_email),
        ])?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            if stderr.contains("nothing to commit") {
                return Err(GitError::NothingToCommit);
            }
            return Err(GitError::CommandFailed(stderr.to_string()));
        }

        // Get commit info
        self.get_head_commit()
    }

    /// Generate semantic commit message from changes
    pub fn generate_commit_message(&self, changes: &ConfigChanges) -> String {
        let mut parts = Vec::new();

        // Analyze change types
        if !changes.packages_added.is_empty() {
            if changes.packages_added.len() == 1 {
                parts.push(format!("Add package: {}", changes.packages_added[0]));
            } else {
                parts.push(format!("Add {} packages", changes.packages_added.len()));
            }
        }

        if !changes.packages_removed.is_empty() {
            if changes.packages_removed.len() == 1 {
                parts.push(format!("Remove package: {}", changes.packages_removed[0]));
            } else {
                parts.push(format!(
                    "Remove {} packages",
                    changes.packages_removed.len()
                ));
            }
        }

        if !changes.services_enabled.is_empty() {
            for svc in &changes.services_enabled {
                parts.push(format!("Enable service: {svc}"));
            }
        }

        if !changes.services_disabled.is_empty() {
            for svc in &changes.services_disabled {
                parts.push(format!("Disable service: {svc}"));
            }
        }

        if !changes.options_changed.is_empty() {
            if changes.options_changed.len() <= 3 {
                for opt in &changes.options_changed {
                    parts.push(format!("Update {opt}"));
                }
            } else {
                parts.push(format!("Update {} options", changes.options_changed.len()));
            }
        }

        // Construct message
        if parts.is_empty() {
            "Update NixOS configuration".to_string()
        } else if parts.len() == 1 {
            parts.remove(0)
        } else {
            let summary = &parts[0];
            let details = parts[1..].join("\n- ");
            format!("{summary}\n\n- {details}")
        }
    }

    /// Auto-commit if enabled and changes exist
    pub fn auto_commit_if_needed(
        &self,
        changes: &ConfigChanges,
    ) -> Result<Option<CommitInfo>, GitError> {
        if !self.auto_commit {
            return Ok(None);
        }

        let status = self.status()?;
        if !status.has_changes {
            return Ok(None);
        }

        self.stage_all()?;
        let message = self.generate_commit_message(changes);
        let commit = self.commit(&message)?;

        Ok(Some(commit))
    }

    /// Get diff of staged changes
    pub fn staged_diff(&self) -> Result<String, GitError> {
        let output = self.run_git(&["diff", "--cached"])?;
        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }

    /// Get diff of unstaged changes
    pub fn unstaged_diff(&self) -> Result<String, GitError> {
        let output = self.run_git(&["diff"])?;
        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }

    /// Get diff between commits
    pub fn diff_commits(&self, from: &str, to: &str) -> Result<String, GitError> {
        let output = self.run_git(&["diff", from, to])?;
        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }

    /// Get commit history
    pub fn log(&self, limit: usize) -> Result<Vec<CommitInfo>, GitError> {
        let output = self.run_git(&["log", &format!("-{limit}"), "--format=%H|%an|%ae|%at|%s"])?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let mut commits = Vec::new();

        for line in stdout.lines() {
            let parts: Vec<&str> = line.split('|').collect();
            if parts.len() >= 5 {
                let timestamp = parts[3].parse::<u64>().unwrap_or(0);
                commits.push(CommitInfo {
                    hash: parts[0].to_string(),
                    short_hash: parts[0][..7.min(parts[0].len())].to_string(),
                    author_name: parts[1].to_string(),
                    author_email: parts[2].to_string(),
                    timestamp,
                    message: parts[4..].join("|"),
                });
            }
        }

        Ok(commits)
    }

    /// Get HEAD commit info
    pub fn get_head_commit(&self) -> Result<CommitInfo, GitError> {
        let commits = self.log(1)?;
        commits.into_iter().next().ok_or(GitError::NoCommits)
    }

    /// Checkout a specific commit or branch
    pub fn checkout(&self, target: &str) -> Result<(), GitError> {
        self.run_git(&["checkout", target])?;
        Ok(())
    }

    /// Create a new branch
    pub fn create_branch(&self, name: &str) -> Result<(), GitError> {
        self.run_git(&["checkout", "-b", name])?;
        Ok(())
    }

    /// Get current branch name
    pub fn current_branch(&self) -> Result<String, GitError> {
        let output = self.run_git(&["branch", "--show-current"])?;
        Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
    }

    /// Revert to a previous commit
    pub fn revert(&self, commit: &str) -> Result<CommitInfo, GitError> {
        self.run_git(&["revert", "--no-edit", commit])?;
        self.get_head_commit()
    }

    /// Create a tag
    pub fn tag(&self, name: &str, message: Option<&str>) -> Result<(), GitError> {
        if let Some(msg) = message {
            self.run_git(&["tag", "-a", name, "-m", msg])?;
        } else {
            self.run_git(&["tag", name])?;
        }
        Ok(())
    }

    /// List tags
    pub fn list_tags(&self) -> Result<Vec<String>, GitError> {
        let output = self.run_git(&["tag", "-l"])?;
        let stdout = String::from_utf8_lossy(&output.stdout);
        Ok(stdout.lines().map(|s| s.to_string()).collect())
    }

    fn run_git(&self, args: &[&str]) -> Result<Output, GitError> {
        Command::new("git")
            .args(args)
            .current_dir(&self.repo_path)
            .output()
            .map_err(|e| GitError::IoError(e.to_string()))
    }
}

/// Repository status
#[derive(Debug, Clone)]
pub struct RepoStatus {
    /// Modified files
    pub modified: Vec<String>,
    /// Added files
    pub added: Vec<String>,
    /// Deleted files
    pub deleted: Vec<String>,
    /// Untracked files
    pub untracked: Vec<String>,
    /// Has any changes
    pub has_changes: bool,
}

/// Commit information
#[derive(Debug, Clone)]
pub struct CommitInfo {
    /// Full commit hash
    pub hash: String,
    /// Short commit hash
    pub short_hash: String,
    /// Author name
    pub author_name: String,
    /// Author email
    pub author_email: String,
    /// Unix timestamp
    pub timestamp: u64,
    /// Commit message
    pub message: String,
}

impl CommitInfo {
    /// Format timestamp as human-readable string
    pub fn formatted_time(&self) -> String {
        // Simple relative time formatting without chrono
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let elapsed = now.saturating_sub(self.timestamp);
        let mins = elapsed / 60;
        let hours = mins / 60;
        let days = hours / 24;

        if days > 365 {
            format!("{} years ago", days / 365)
        } else if days > 30 {
            format!("{} months ago", days / 30)
        } else if days > 0 {
            format!("{days} days ago")
        } else if hours > 0 {
            format!("{} hours ago", hours % 24)
        } else if mins > 0 {
            format!("{} minutes ago", mins % 60)
        } else {
            "just now".to_string()
        }
    }
}

/// Configuration changes for semantic commit messages
#[derive(Debug, Clone, Default)]
pub struct ConfigChanges {
    /// Packages added
    pub packages_added: Vec<String>,
    /// Packages removed
    pub packages_removed: Vec<String>,
    /// Services enabled
    pub services_enabled: Vec<String>,
    /// Services disabled
    pub services_disabled: Vec<String>,
    /// Options changed
    pub options_changed: Vec<String>,
    /// Files changed
    pub files_changed: Vec<String>,
}

impl ConfigChanges {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_package(&mut self, name: impl Into<String>) {
        self.packages_added.push(name.into());
    }

    pub fn remove_package(&mut self, name: impl Into<String>) {
        self.packages_removed.push(name.into());
    }

    pub fn enable_service(&mut self, name: impl Into<String>) {
        self.services_enabled.push(name.into());
    }

    pub fn disable_service(&mut self, name: impl Into<String>) {
        self.services_disabled.push(name.into());
    }

    pub fn change_option(&mut self, path: impl Into<String>) {
        self.options_changed.push(path.into());
    }

    pub fn is_empty(&self) -> bool {
        self.packages_added.is_empty()
            && self.packages_removed.is_empty()
            && self.services_enabled.is_empty()
            && self.services_disabled.is_empty()
            && self.options_changed.is_empty()
    }
}

/// Git tracking errors
#[derive(Debug, Clone)]
pub enum GitError {
    /// Git command failed
    CommandFailed(String),
    /// IO error
    IoError(String),
    /// Nothing to commit
    NothingToCommit,
    /// No commits in repository
    NoCommits,
    /// Not a git repository
    NotARepo,
}

impl std::fmt::Display for GitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CommandFailed(msg) => write!(f, "Git command failed: {msg}"),
            Self::IoError(msg) => write!(f, "IO error: {msg}"),
            Self::NothingToCommit => write!(f, "Nothing to commit"),
            Self::NoCommits => write!(f, "No commits in repository"),
            Self::NotARepo => write!(f, "Not a git repository"),
        }
    }
}

impl std::error::Error for GitError {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::SystemTime;

    #[test]
    fn test_commit_message_generation() {
        let tracker = GitTracker::new("/tmp/test");

        let mut changes = ConfigChanges::new();
        changes.add_package("firefox");
        let msg = tracker.generate_commit_message(&changes);
        assert_eq!(msg, "Add package: firefox");

        let mut changes = ConfigChanges::new();
        changes.add_package("firefox");
        changes.add_package("chromium");
        let msg = tracker.generate_commit_message(&changes);
        assert!(msg.contains("Add 2 packages"));

        let mut changes = ConfigChanges::new();
        changes.enable_service("nginx");
        let msg = tracker.generate_commit_message(&changes);
        assert_eq!(msg, "Enable service: nginx");
    }

    #[test]
    fn test_config_changes() {
        let mut changes = ConfigChanges::new();
        assert!(changes.is_empty());

        changes.add_package("vim");
        assert!(!changes.is_empty());
        assert_eq!(changes.packages_added.len(), 1);
    }

    #[test]
    fn test_commit_info_time_formatting() {
        let now = SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let commit = CommitInfo {
            hash: "abc123".to_string(),
            short_hash: "abc123".to_string(),
            author_name: "Test".to_string(),
            author_email: "test@test.com".to_string(),
            timestamp: now - 60, // 1 minute ago
            message: "Test".to_string(),
        };

        assert!(commit.formatted_time().contains("minute"));
    }
}
