//! Process-facing JSONL adapter for the blind open-world episode protocol.
//!
//! This executable consumes only serialized AgentView messages and emits one
//! AgentAction per input line. It never accepts evaluator truth or scores.
//! Keep stdout machine-readable; diagnostics are written only to stderr.

use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::process::ExitCode;

use open_world_ansatz_microcosm::policy::{
    FiniteEnumeratorBaseline, PolicyError, StrategyEnumeratorBaseline,
};
use open_world_ansatz_microcosm::protocol::{AgentAction, AgentView, PROTOCOL_VERSION};
use open_world_ansatz_microcosm::strategies::ProbeStrategy;

const MAX_INPUT_LINE_BYTES: usize = 1_048_576;

#[derive(Clone, Copy)]
enum AgentPolicy {
    FiniteEnumerator,
    Strategy(ProbeStrategy),
}

impl AgentPolicy {
    fn next_action(self, view: &AgentView) -> Result<AgentAction, PolicyError> {
        match self {
            Self::FiniteEnumerator => FiniteEnumeratorBaseline::next_action(view),
            Self::Strategy(strategy) => StrategyEnumeratorBaseline::next_action(view, strategy),
        }
    }
}

fn parse_policy(args: &[String]) -> Result<AgentPolicy, String> {
    if args.len() > 2 || args.first().is_some_and(|arg| arg != "--strategy") {
        return Err(
            "usage: open_world_agent [--strategy finite|maximum_partition|minimax|maximum_entropy|minimum_expected_remaining|maximum_structure_information_gain|fixed_grid|random:<seed>]"
                .to_owned(),
        );
    }
    let Some(value) = args.get(1) else {
        return Ok(AgentPolicy::FiniteEnumerator);
    };

    let policy = match value.as_str() {
        "finite" => AgentPolicy::FiniteEnumerator,
        "maximum_partition" => AgentPolicy::Strategy(ProbeStrategy::MaximumPartition),
        "minimax" => AgentPolicy::Strategy(ProbeStrategy::Minimax),
        "maximum_entropy" => AgentPolicy::Strategy(ProbeStrategy::MaximumEntropy),
        "minimum_expected_remaining" => {
            AgentPolicy::Strategy(ProbeStrategy::MinimumExpectedRemaining)
        }
        "maximum_structure_information_gain" => {
            AgentPolicy::Strategy(ProbeStrategy::MaximumStructureInformationGain)
        }
        "fixed_grid" => AgentPolicy::Strategy(ProbeStrategy::FixedGrid),
        random if random.starts_with("random:") => {
            let seed = random["random:".len()..]
                .parse::<u64>()
                .map_err(|error| format!("invalid random strategy seed: {error}"))?;
            AgentPolicy::Strategy(ProbeStrategy::Random { seed })
        }
        other => return Err(format!("unknown strategy: {other}")),
    };
    Ok(policy)
}

enum LineRead {
    Eof,
    Line,
}

fn read_bounded_line<R: BufRead>(
    reader: &mut R,
    buffer: &mut Vec<u8>,
) -> io::Result<LineRead> {
    buffer.clear();
    loop {
        let available = reader.fill_buf()?;
        if available.is_empty() {
            return Ok(if buffer.is_empty() {
                LineRead::Eof
            } else {
                LineRead::Line
            });
        }

        let take = available
            .iter()
            .position(|byte| *byte == b'\\n')
            .map_or(available.len(), |index| index + 1);

        if buffer.len().saturating_add(take) > MAX_INPUT_LINE_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("AgentView line exceeds {MAX_INPUT_LINE_BYTES} bytes"),
            ));
        }

        buffer.extend_from_slice(&available[..take]);
        let line_complete = available.get(take.saturating_sub(1)) == Some(&b'\\n');
        reader.consume(take);
        if line_complete {
            return Ok(LineRead::Line);
        }
    }
}

fn run() -> Result<(), String> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let policy = parse_policy(&args)?;

    let stdin = io::stdin();
    let mut reader = BufReader::new(stdin.lock());
    let stdout = io::stdout();
    let mut writer = BufWriter::new(stdout.lock());
    let mut line = Vec::new();

    loop {
        match read_bounded_line(&mut reader, &mut line)
            .map_err(|error| format!("cannot read AgentView: {error}"))?
        {
            LineRead::Eof => break,
            LineRead::Line => {
                while matches!(line.last(), Some(b'\\n' | b'\\r')) {
                    line.pop();
                }
                if line.is_empty() {
                    return Err("blank input lines are not valid AgentView messages".to_owned());
                }

                let view: AgentView = serde_json::from_slice(&line)
                    .map_err(|error| format!("invalid AgentView JSON: {error}"))?;
                if view.task.protocol_version != PROTOCOL_VERSION {
                    return Err(format!(
                        "unsupported protocol version: {}",
                        view.task.protocol_version
                    ));
                }
                if view.status != open_world_ansatz_microcosm::protocol::EpisodeStatus::Open {
                    return Err("agent received a non-open episode".to_owned());
                }

                let action = policy
                    .next_action(&view)
                    .map_err(|error| format!("policy could not select action: {error}"))?;
                serde_json::to_writer(&mut writer, &action)
                    .map_err(|error| format!("cannot serialize AgentAction: {error}"))?;
                writer
                    .write_all(b"\\n")
                    .and_then(|()| writer.flush())
                    .map_err(|error| format!("cannot write AgentAction: {error}"))?;
            }
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::from(64)
        }
    }
}
