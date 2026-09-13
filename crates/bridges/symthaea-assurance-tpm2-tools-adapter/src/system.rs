use std::process::{Command, Stdio};

use crate::{blake3_digest, ToolExecution, Tpm2AdapterError, Tpm2ToolsExecutor};

/// Shell-free Linux executor for an explicitly pinned `tpm2-tools` binary path.
///
/// The reviewed command line supplies an explicit device TCTI. Environment
/// variables that could replace that TCTI or alter dynamic-library loading are
/// removed before execution so the pinned executable digest remains meaningful.
#[derive(Debug, Default, Clone, Copy)]
pub struct SystemTpm2ToolsExecutor;

impl Tpm2ToolsExecutor for SystemTpm2ToolsExecutor {
    fn executable_blake3(&self, executable: &str) -> Result<String, Tpm2AdapterError> {
        let bytes = std::fs::read(executable)
            .map_err(|error| Tpm2AdapterError::Io(error.to_string()))?;
        Ok(blake3_digest(&bytes))
    }

    fn execute(&self, executable: &str, args: &[String]) -> Result<ToolExecution, Tpm2AdapterError> {
        let output = Command::new(executable)
            .args(args)
            .env_remove("TPM2TOOLS_TCTI")
            .env_remove("LD_PRELOAD")
            .env_remove("LD_LIBRARY_PATH")
            .stdin(Stdio::null())
            .output()
            .map_err(|error| Tpm2AdapterError::Io(error.to_string()))?;
        Ok(ToolExecution {
            exit_code: output.status.code(),
            stdout: output.stdout,
            stderr: output.stderr,
        })
    }
}
