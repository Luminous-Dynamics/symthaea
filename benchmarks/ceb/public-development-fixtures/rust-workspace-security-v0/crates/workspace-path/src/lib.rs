//! Lexical path helper for the public CEB workspace-security development fixture.
//!
//! This is intentionally not a filesystem authorization primitive: it does not
//! resolve symlinks, open handles, or prevent time-of-check/time-of-use races.

use std::path::{Component, Path, PathBuf};

#[derive(Debug, Clone)]
pub struct WorkspacePath {
    root: PathBuf,
}

impl WorkspacePath {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// Reject obvious absolute/parent traversal and return a joined path.
    ///
    /// This is a lexical filter only. Treating the result as a secure path
    /// capability would be an invalid use of this development helper.
    pub fn lexical_join(&self, input: &Path) -> Result<PathBuf, &'static str> {
        if input.is_absolute() {
            return Err("absolute path rejected");
        }

        if input.components().any(|component| matches!(component, Component::ParentDir)) {
            return Err("parent traversal rejected");
        }

        Ok(self.root.join(input))
    }
}
