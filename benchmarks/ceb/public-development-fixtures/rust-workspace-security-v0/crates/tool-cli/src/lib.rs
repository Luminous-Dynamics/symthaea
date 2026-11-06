//! CLI-facing command-description consumer for the public CEB security fixture.

use std::path::Path;

use ceb_workspace_path::WorkspacePath;
use ceb_workspace_command_runner::unsafe_git_status_command;

/// Return the command description that would be handed to a future runner.
///
/// The development fixture deliberately stops here. No shell is invoked.
pub fn describe_status(root: &Path, requested_path: &Path) -> Result<String, &'static str> {
    let workspace = WorkspacePath::new(root.to_path_buf());
    unsafe_git_status_command(&workspace, requested_path)
}
