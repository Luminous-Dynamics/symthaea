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
use open_world_ansatz_microcosm::protocol::{
    AgentAction, AgentView, EpisodeStatus, PROTOCOL_VERSION,
};
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
    let value = match args {
        [] => return Ok(AgentPolicy::FiniteEnumerator),
        [flag, value] if flag == "--strategy" => value.as_str(),
        _ => {
            return Err(
                "usage: open_world_agent [--strategy finite|maximum_partition|minimax|maximum_entropy|minimum_expected_remaining|maximum_structure_information_gain|fixed_grid|random:<seed>]"
                    .to_owned(),
            );
        }
    };

    match value {
        "finite" => Ok(AgentPolicy::FiniteEnumerator),
        "maximum_partition" => Ok(AgentPolicy::Strategy(ProbeStrategy::MaximumPartition)),
        "minimax" => Ok(AgentPolicy::Strategy(ProbeStrategy::Minimax)),
        "maximum_entropy" => Ok(AgentPolicy::Strategy(ProbeStrategy::MaximumEntropy)),
        "minimum_expected_remaining" => Ok(AgentPolicy::Strategy(
            ProbeStrategy::MinimumExpectedRemaining,
        )),
        "maximum_structure_information_gain" => Ok(AgentPolicy::Strategy(
            ProbeStrategy::MaximumStructureInformationGain,
        )),
        "fixed_grid" => Ok(AgentPolicy::Strategy(ProbeStrategy::FixedGrid)),
        random if random.starts_with("random:") => {
            let seed = random["random:".len()..]
                .parse::<u64>()
                .map_err(|error| format!("invalid random strategy seed: {error}"))?;
            Ok(AgentPolicy::Strategy(ProbeStrategy::Random { seed }))
        }
        other => Err(format!("unknown strategy: {other}")),
    }
}

enum LineRead {
    Eof,
    Line,
}

fn read_bounded_line<R: BufRead>(reader: &mut R, buffer: &mut Vec<u8>) -> io::Result<LineRead> {
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
            .position(|byte| *byte == b'\n')
            .map_or(available.len(), |index| index + 1);

        if buffer.len().saturating_add(take) > MAX_INPUT_LINE_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("AgentView line exceeds {MAX_INPUT_LINE_BYTES} bytes"),
            ));
        }

        buffer.extend_from_slice(&available[..take]);
        let line_complete = available.get(take.saturating_sub(1)) == Some(&b'\n');
        reader.consume(take);
        if line_complete {
            return Ok(LineRead::Line);
        }
    }
}

fn next_action(policy: AgentPolicy, line: &[u8]) -> Result<Vec<u8>, String> {
    let view: AgentView =
        serde_json::from_slice(line).map_err(|error| format!("invalid AgentView JSON: {error}"))?;
    if view.task.protocol_version != PROTOCOL_VERSION {
        return Err(format!(
            "unsupported protocol version: {}",
            view.task.protocol_version
        ));
    }
    if view.status != EpisodeStatus::Open {
        return Err("agent received a non-open episode".to_owned());
    }
    let action = policy
        .next_action(&view)
        .map_err(|error| format!("policy could not select action: {error}"))?;
    serde_json::to_vec(&action).map_err(|error| format!("cannot serialize AgentAction: {error}"))
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
                while line
                    .last()
                    .is_some_and(|byte| *byte == b'\n' || *byte == b'\r')
                {
                    line.pop();
                }
                if line.is_empty() {
                    return Err("blank input lines are not valid AgentView messages".to_owned());
                }

                let action = next_action(policy, &line)?;
                writer
                    .write_all(&action)
                    .and_then(|()| writer.write_all(b"\n"))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_finite_and_random_strategies() {
        assert!(matches!(
            parse_policy(&[]),
            Ok(AgentPolicy::FiniteEnumerator)
        ));
        assert!(matches!(
            parse_policy(&["--strategy".to_owned(), "random:19".to_owned()]),
            Ok(AgentPolicy::Strategy(ProbeStrategy::Random { seed: 19 }))
        ));
    }

    #[test]
    fn rejects_incomplete_strategy_arguments() {
        assert!(parse_policy(&["--strategy".to_owned()]).is_err());
        assert!(parse_policy(&["--unknown".to_owned()]).is_err());
    }

    #[test]
    fn serialized_blind_view_produces_an_action_accepted_by_the_episode_runner() {
        use open_world_ansatz_microcosm::numeric::{Mechanism, ModelHypothesis, SensorMap};
        use open_world_ansatz_microcosm::protocol::{BlindEpisode, StructureAccess, WorldFamily};

        let mut episode = BlindEpisode::new_numeric(
            "stdio-wire-roundtrip",
            ModelHypothesis {
                mechanism: Mechanism::DirectResponse,
                gain: 2,
                decay: None,
                initial_state: None,
                sensor: SensorMap::Identity,
            },
            StructureAccess::NumericWithheld,
            3,
        )
        .unwrap();
        let view = episode.view().unwrap();
        assert_eq!(view.task.world_family, WorldFamily::NumericMechanism);
        let input = serde_json::to_vec(&view).unwrap();

        let output = next_action(AgentPolicy::FiniteEnumerator, &input).unwrap();
        let action: AgentAction = serde_json::from_slice(&output).unwrap();
        let response = episode.apply(action).unwrap();
        assert_eq!(response.view.observations.len(), 1);
        assert_eq!(response.status, EpisodeStatus::Open);
    }

    #[test]
    fn reads_bounded_jsonl_frames_without_merging_them() {
        let mut reader = BufReader::new(&b"{\"a\":1}\n{\"b\":2}"[..]);
        let mut buffer = Vec::new();

        assert!(matches!(
            read_bounded_line(&mut reader, &mut buffer).unwrap(),
            LineRead::Line
        ));
        assert_eq!(buffer, b"{\"a\":1}\n");

        assert!(matches!(
            read_bounded_line(&mut reader, &mut buffer).unwrap(),
            LineRead::Line
        ));
        assert_eq!(buffer, b"{\"b\":2}");

        assert!(matches!(
            read_bounded_line(&mut reader, &mut buffer).unwrap(),
            LineRead::Eof
        ));
    }
}
