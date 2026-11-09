//! Parent-runner / child-agent bridge for blind open-world episodes.
//!
//! The evaluator owns the BlindEpisode in the parent process. The child sees
//! only serialized public AgentView frames on stdin and returns AgentAction
//! frames on stdout. This is process separation, not an OS security sandbox:
//! callers must still isolate filesystem, network, CPU/memory and secrets with
//! a separately qualified execution boundary before running untrusted agents.

use crate::protocol::{
    AgentAction, AgentView, BlindEpisode, EpisodeStatus, EvaluatorScore, PROTOCOL_VERSION,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::ffi::OsString;
use std::fmt;
use std::fs;
use std::io::{self, BufRead, BufReader, ChildStdin, ChildStdout, Read, Write};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};

pub const DEFAULT_MAX_FRAME_BYTES: usize = 1_048_576;
pub const MAX_PROCESS_FRAME_BYTES: usize = 16 * 1_048_576;
pub const MAX_PROCESS_PROBES: usize = 1_024;

#[derive(Debug, Clone)]
pub struct ProcessAgentConfig {
    pub executable: PathBuf,
    pub args: Vec<OsString>,
    pub working_directory: Option<PathBuf>,
    pub max_input_frame_bytes: usize,
    pub max_output_frame_bytes: usize,
}

impl ProcessAgentConfig {
    pub fn new(executable: impl Into<PathBuf>) -> Self {
        Self {
            executable: executable.into(),
            args: Vec::new(),
            working_directory: None,
            max_input_frame_bytes: DEFAULT_MAX_FRAME_BYTES,
            max_output_frame_bytes: DEFAULT_MAX_FRAME_BYTES,
        }
    }

    pub fn with_args(mut self, args: impl IntoIterator<Item = OsString>) -> Self {
        self.args = args.into_iter().collect();
        self
    }

    pub fn with_working_directory(mut self, path: impl Into<PathBuf>) -> Self {
        self.working_directory = Some(path.into());
        self
    }

    pub fn with_frame_limits(mut self, input_bytes: usize, output_bytes: usize) -> Self {
        self.max_input_frame_bytes = input_bytes;
        self.max_output_frame_bytes = output_bytes;
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProcessAgentIdentity {
    pub executable_path: String,
    pub executable_sha256: String,
    pub arguments: Vec<String>,
    pub working_directory: Option<String>,
    pub max_input_frame_bytes: usize,
    pub max_output_frame_bytes: usize,
    pub environment_inherited: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProcessActionRecord {
    pub action_index: usize,
    pub agent_view_frame_sha256: String,
    pub agent_action_frame_sha256: String,
    pub evidence_prefix_before: String,
    pub action: AgentAction,
    pub evidence_prefix_after: String,
    pub resulting_status: EpisodeStatus,
}

/// One exact request/response pair across the child-process JSONL boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProcessActionExchange {
    pub action: AgentAction,
    /// SHA-256 of the exact AgentView bytes sent, including its JSONL newline.
    pub agent_view_frame_sha256: String,
    /// SHA-256 of the exact AgentAction bytes received, including its JSONL newline.
    pub agent_action_frame_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProcessEpisodeReceipt {
    pub receipt_version: String,
    pub episode_id: String,
    pub world_family: crate::protocol::WorldFamily,
    pub comparison_arm: String,
    pub agent: ProcessAgentIdentity,
    pub initial_evidence_prefix: String,
    pub final_evidence_prefix: String,
    pub action_records: Vec<ProcessActionRecord>,
    pub observation_count: usize,
    pub evaluator_score: EvaluatorScore,
    pub process_exit_code: i32,
    pub limitations: Vec<String>,
}

/// Long-lived child endpoint. It serializes a view, writes one frame, parses
/// one action, and knows nothing about evaluator truth.
pub struct ProcessAgent {
    child: Option<Child>,
    stdin: Option<ChildStdin>,
    stdout: BufReader<ChildStdout>,
    identity: ProcessAgentIdentity,
    max_input_frame_bytes: usize,
    max_output_frame_bytes: usize,
}

impl ProcessAgent {
    pub fn spawn(config: ProcessAgentConfig) -> Result<Self, ProcessError> {
        validate_frame_limits(config.max_input_frame_bytes, config.max_output_frame_bytes)?;

        let executable = fs::canonicalize(&config.executable).map_err(|error| {
            ProcessError::Io(format!("cannot resolve agent executable: {error}"))
        })?;
        let metadata = fs::metadata(&executable).map_err(|error| {
            ProcessError::Io(format!("cannot inspect agent executable: {error}"))
        })?;
        if !metadata.is_file() {
            return Err(ProcessError::InvalidConfiguration(
                "agent executable must resolve to a regular file".to_owned(),
            ));
        }

        let executable_bytes = fs::read(&executable)
            .map_err(|error| ProcessError::Io(format!("cannot hash agent executable: {error}")))?;
        let executable_sha256 = format!("sha256:{:x}", Sha256::digest(&executable_bytes));
        let arguments = config
            .args
            .iter()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect::<Vec<_>>();
        let working_directory = config
            .working_directory
            .as_ref()
            .map(fs::canonicalize)
            .transpose()
            .map_err(|error| {
                ProcessError::Io(format!("cannot resolve agent working directory: {error}"))
            })?;

        let mut command = Command::new(&executable);
        command
            .args(&config.args)
            .env_clear()
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());

        if let Some(directory) = &working_directory {
            command.current_dir(directory);
        }

        let mut child = command
            .spawn()
            .map_err(|error| ProcessError::Io(format!("cannot spawn agent process: {error}")))?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| ProcessError::Io("agent stdin pipe was not created".to_owned()))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| ProcessError::Io("agent stdout pipe was not created".to_owned()))?;

        Ok(Self {
            child: Some(child),
            stdin: Some(stdin),
            stdout: BufReader::new(stdout),
            identity: ProcessAgentIdentity {
                executable_path: executable.display().to_string(),
                executable_sha256,
                arguments,
                working_directory: working_directory
                    .as_ref()
                    .map(|directory| directory.display().to_string()),
                max_input_frame_bytes: config.max_input_frame_bytes,
                max_output_frame_bytes: config.max_output_frame_bytes,
                environment_inherited: false,
            },
            max_input_frame_bytes: config.max_input_frame_bytes,
            max_output_frame_bytes: config.max_output_frame_bytes,
        })
    }

    pub fn identity(&self) -> &ProcessAgentIdentity {
        &self.identity
    }

    pub fn next_action(&mut self, view: &AgentView) -> Result<ProcessActionExchange, ProcessError> {
        if view.task.protocol_version != PROTOCOL_VERSION {
            return Err(ProcessError::Protocol(format!(
                "unsupported public protocol version: {}",
                view.task.protocol_version
            )));
        }
        if view.status != EpisodeStatus::Open {
            return Err(ProcessError::Protocol(
                "cannot request an action for a non-open episode".to_owned(),
            ));
        }

        let mut input = serde_json::to_vec(view)
            .map_err(|error| ProcessError::Serialization(error.to_string()))?;
        // The JSONL delimiter is part of the child-side frame limit.
        if input.len().saturating_add(1) > self.max_input_frame_bytes {
            return Err(ProcessError::FrameTooLarge {
                direction: "input",
                limit: self.max_input_frame_bytes,
            });
        }
        input.push(b'\n');
        let agent_view_frame_sha256 = sha256_frame(&input);

        let stdin = self
            .stdin
            .as_mut()
            .ok_or_else(|| ProcessError::Protocol("agent stdin was already closed".to_owned()))?;
        stdin
            .write_all(&input)
            .and_then(|()| stdin.flush())
            .map_err(|error| ProcessError::Io(format!("cannot send AgentView: {error}")))?;

        let mut output = Vec::new();
        match read_bounded_line(&mut self.stdout, &mut output, self.max_output_frame_bytes)
            .map_err(|error| ProcessError::Io(format!("cannot read AgentAction: {error}")))?
        {
            BoundedLine::Eof => {
                let status = self
                    .child
                    .as_mut()
                    .and_then(|child| child.try_wait().ok().flatten());
                Err(ProcessError::AgentExited(status.map_or_else(
                    || "stdout closed before an AgentAction frame".to_owned(),
                    |status| format!("stdout closed before an AgentAction frame: {status}"),
                )))
            }
            BoundedLine::Line => {
                let agent_action_frame_sha256 = sha256_frame(&output);
                while output
                    .last()
                    .is_some_and(|byte| *byte == b'\n' || *byte == b'\r')
                {
                    output.pop();
                }
                if output.is_empty() {
                    return Err(ProcessError::Protocol(
                        "agent emitted a blank stdout frame".to_owned(),
                    ));
                }
                let action = serde_json::from_slice(&output)
                    .map_err(|error| ProcessError::MalformedAction(error.to_string()))?;
                Ok(ProcessActionExchange {
                    action,
                    agent_view_frame_sha256,
                    agent_action_frame_sha256,
                })
            }
        }
    }

    /// Close the request stream, require a clean stdout EOF and wait for exit.
    pub fn finish(mut self) -> Result<i32, ProcessError> {
        drop(self.stdin.take());

        let mut trailing = Vec::new();
        let read_result = self
            .stdout
            .by_ref()
            .take(self.max_output_frame_bytes as u64 + 1)
            .read_to_end(&mut trailing);
        if let Err(error) = read_result {
            return Err(ProcessError::Io(format!(
                "cannot drain agent stdout after terminal action: {error}"
            )));
        }
        if trailing.len() > self.max_output_frame_bytes {
            return Err(ProcessError::FrameTooLarge {
                direction: "trailing output",
                limit: self.max_output_frame_bytes,
            });
        }
        if !trailing.is_empty() {
            return Err(ProcessError::Protocol(
                "agent emitted unsolicited stdout after the final action".to_owned(),
            ));
        }

        let child = self
            .child
            .take()
            .ok_or_else(|| ProcessError::Protocol("agent process was already reaped".to_owned()))?;
        let status = child
            .wait()
            .map_err(|error| ProcessError::Io(format!("cannot wait for agent: {error}")))?;
        if !status.success() {
            return Err(ProcessError::AgentNonZeroExit(status.code().unwrap_or(-1)));
        }
        Ok(status.code().unwrap_or(0))
    }
}

impl Drop for ProcessAgent {
    fn drop(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

fn validate_frame_limits(input_bytes: usize, output_bytes: usize) -> Result<(), ProcessError> {
    if input_bytes == 0
        || output_bytes == 0
        || input_bytes > MAX_PROCESS_FRAME_BYTES
        || output_bytes > MAX_PROCESS_FRAME_BYTES
    {
        return Err(ProcessError::InvalidConfiguration(format!(
            "input and output frame limits must be within 1..={MAX_PROCESS_FRAME_BYTES} bytes"
        )));
    }
    Ok(())
}

fn sha256_frame(bytes: &[u8]) -> String {
    format!("sha256:{:x}", Sha256::digest(bytes))
}

enum BoundedLine {
    Eof,
    Line,
}

fn read_bounded_line<R: BufRead>(
    reader: &mut R,
    buffer: &mut Vec<u8>,
    maximum: usize,
) -> io::Result<BoundedLine> {
    buffer.clear();
    loop {
        let available = reader.fill_buf()?;
        if available.is_empty() {
            return Ok(if buffer.is_empty() {
                BoundedLine::Eof
            } else {
                BoundedLine::Line
            });
        }

        let take = available
            .iter()
            .position(|byte| *byte == b'\n')
            .map_or(available.len(), |index| index + 1);
        if buffer.len().saturating_add(take) > maximum {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("agent frame exceeds {maximum} bytes"),
            ));
        }

        buffer.extend_from_slice(&available[..take]);
        let complete = available.get(take.saturating_sub(1)) == Some(&b'\n');
        reader.consume(take);
        if complete {
            return Ok(BoundedLine::Line);
        }
    }
}

/// Drive a complete episode through a separate agent process. The evaluator
/// remains in the caller's process and is never serialized into the child.
pub fn run_process_episode(
    mut episode: BlindEpisode,
    mut agent: ProcessAgent,
    comparison_arm: impl Into<String>,
) -> Result<ProcessEpisodeReceipt, ProcessError> {
    let initial_view = episode
        .view()
        .map_err(|error| ProcessError::Protocol(error.to_string()))?;
    let initial_evidence_prefix = initial_view.evidence_prefix_sha256.clone();
    let episode_id = initial_view.task.episode_id.clone();
    let world_family = initial_view.task.world_family;
    let maximum_actions = initial_view.task.max_probe_actions;
    if maximum_actions > MAX_PROCESS_PROBES {
        return Err(ProcessError::ProbeBudgetTooLarge {
            requested: maximum_actions,
            limit: MAX_PROCESS_PROBES,
        });
    }
    let maximum_action_count =
        maximum_actions
            .checked_add(1)
            .ok_or(ProcessError::ProbeBudgetTooLarge {
                requested: maximum_actions,
                limit: MAX_PROCESS_PROBES,
            })?;

    let mut action_records = Vec::new();
    for action_index in 1..=maximum_action_count {
        if episode.status() == EpisodeStatus::Submitted {
            break;
        }
        let before = episode
            .view()
            .map_err(|error| ProcessError::Protocol(error.to_string()))?;
        let exchange = agent.next_action(&before)?;
        let response = episode.apply(exchange.action.clone()).map_err(|error| {
            ProcessError::Protocol(format!("episode rejected agent action: {error}"))
        })?;
        action_records.push(ProcessActionRecord {
            action_index,
            agent_view_frame_sha256: exchange.agent_view_frame_sha256,
            agent_action_frame_sha256: exchange.agent_action_frame_sha256,
            evidence_prefix_before: before.evidence_prefix_sha256,
            action: exchange.action,
            evidence_prefix_after: response.view.evidence_prefix_sha256,
            resulting_status: response.status,
        });
    }

    if episode.status() != EpisodeStatus::Submitted {
        return Err(ProcessError::Protocol(
            "agent did not submit before the maximum probe budget plus terminal action".to_owned(),
        ));
    }

    let final_view = episode
        .view()
        .map_err(|error| ProcessError::Protocol(error.to_string()))?;
    let final_evidence_prefix = final_view.evidence_prefix_sha256.clone();
    let observation_count = final_view.observations.len();
    let identity = agent.identity().clone();
    let process_exit_code = agent.finish()?;
    let evaluator_score = episode
        .evaluate_after_submission()
        .map_err(|error| ProcessError::Protocol(error.to_string()))?;

    Ok(ProcessEpisodeReceipt {
        receipt_version: "symthaea-blind-process-episode-v0".to_owned(),
        episode_id,
        world_family,
        comparison_arm: comparison_arm.into(),
        agent: identity,
        initial_evidence_prefix,
        final_evidence_prefix,
        action_records,
        observation_count,
        evaluator_score,
        process_exit_code,
        limitations: vec![
            "The child receives only public AgentView JSONL frames and returns AgentAction JSONL frames.".to_owned(),
            "The evaluator remains in the parent process and is not serialized to the child.".to_owned(),
            "This is process separation, not an OS sandbox; filesystem, network, CPU/memory and process visibility require an external isolation boundary for untrusted agents.".to_owned(),
            "The receipt records executable-byte identity and action/evidence-prefix linkage; it does not authenticate the host or prove scientific truth.".to_owned(),
        ],
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProcessError {
    InvalidConfiguration(String),
    Io(String),
    Serialization(String),
    Protocol(String),
    FrameTooLarge {
        direction: &'static str,
        limit: usize,
    },
    AgentExited(String),
    AgentNonZeroExit(i32),
    ProbeBudgetTooLarge {
        requested: usize,
        limit: usize,
    },
    MalformedAction(String),
}

impl fmt::Display for ProcessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for ProcessError {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn frame_limits_reject_zero_and_unbounded_values() {
        assert!(validate_frame_limits(1, 1).is_ok());
        assert!(matches!(
            validate_frame_limits(0, 1),
            Err(ProcessError::InvalidConfiguration(_))
        ));
        assert!(matches!(
            validate_frame_limits(MAX_PROCESS_FRAME_BYTES + 1, 1),
            Err(ProcessError::InvalidConfiguration(_))
        ));
        assert!(matches!(
            validate_frame_limits(1, MAX_PROCESS_FRAME_BYTES + 1),
            Err(ProcessError::InvalidConfiguration(_))
        ));
    }

    #[test]
    fn frame_digest_commits_to_jsonl_delimiter_bytes() {
        assert_ne!(sha256_frame(b"{}"), sha256_frame(b"{}\n"));
        assert_eq!(
            sha256_frame(b"{}\n"),
            format!("sha256:{:x}", Sha256::digest(b"{}\n"))
        );
    }

    #[test]
    fn bounded_reader_accepts_exactly_limited_line() {
        let mut reader = BufReader::new(Cursor::new(b"{\"a\":1}\n"));
        let mut buffer = Vec::new();
        assert!(matches!(
            read_bounded_line(&mut reader, &mut buffer, 8).unwrap(),
            BoundedLine::Line
        ));
        assert_eq!(buffer, b"{\"a\":1}\n");
    }

    #[test]
    fn bounded_reader_rejects_oversized_line_before_unbounded_growth() {
        let mut reader = BufReader::new(Cursor::new(b"{\"too_long\":true}\n"));
        let mut buffer = Vec::new();
        let error = read_bounded_line(&mut reader, &mut buffer, 8).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn bounded_reader_preserves_eof_terminated_final_frame() {
        let mut reader = BufReader::new(Cursor::new(b"{\"a\":1}"));
        let mut buffer = Vec::new();
        assert!(matches!(
            read_bounded_line(&mut reader, &mut buffer, 16).unwrap(),
            BoundedLine::Line
        ));
        assert_eq!(buffer, b"{\"a\":1}");
        assert!(matches!(
            read_bounded_line(&mut reader, &mut buffer, 16).unwrap(),
            BoundedLine::Eof
        ));
    }
}
