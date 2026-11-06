//! Command description builder for the public CEB workspace-security fixture.
//!
//! Intentional defect: the path is interpolated into shell command text. This
//! function only returns a string; it does not execute a shell or process.

use std::path::Path;

use ceb_workspace_path::WorkspacePath;

pub fn unsafe_git_status_command(root: &WorkspacePath, relative: &Path) -> Result<String, &'static str> {
    let joined = root.lexical_join(relative)?;
    Ok(format!("git -C {} status --short", joined.display()))
}
