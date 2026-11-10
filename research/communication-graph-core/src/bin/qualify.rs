use communication_graph_core::{CommunicationEventLog, CommunicationGraphProfile, DeliveryOutcome, normalized_betweenness_v1};
use std::{env, fs, process::ExitCode};

fn load<T: serde::de::DeserializeOwned>(path: &str) -> Result<T, String> {
    let source = fs::read_to_string(path).map_err(|error| format!("{path}: {error}"))?;
    serde_json::from_str(&source).map_err(|error| format!("{path}: {error}"))
}

fn main() -> ExitCode {
    let args = env::args().collect::<Vec<_>>();
    if args.len() != 3 {
        eprintln!("usage: communication-graph-qualifier <graph-profile.json> <event-log.json>");
        return ExitCode::from(2);
    }

    let graph: CommunicationGraphProfile = match load(&args[1]) {
        Ok(value) => value,
        Err(error) => { eprintln!("{error}"); return ExitCode::FAILURE; }
    };
    let log: CommunicationEventLog = match load(&args[2]) {
        Ok(value) => value,
        Err(error) => { eprintln!("{error}"); return ExitCode::FAILURE; }
    };

    if let Err(error) = graph.validate() {
        eprintln!("GRAPH PROFILE INVALID: {error}");
        return ExitCode::FAILURE;
    }
    if let Err(error) = log.validate_against(&graph) {
        eprintln!("COMMUNICATION TRACE INVALID: {error}");
        return ExitCode::FAILURE;
    }

    println!("GRAPH SHA-256: {}", graph.sha256().expect("validated graph digest"));
    println!("EVENT LOG SHA-256: {}", log.sha256().expect("valid event log serialization"));
    println!("COMMUNICATION EVENTS: {}", log.events.len());
    let delivered = log.events.iter()
        .filter(|event| matches!(&event.outcome, DeliveryOutcome::Delivered))
        .count();
    println!("DELIVERED: {delivered}");
    println!("REJECTED: {}", log.events.len() - delivered);

    match normalized_betweenness_v1(&graph) {
        Ok(profile) => {
            println!("CENTRALITY PROFILE: {}", profile.metric_id);
            println!("MINIMUM NODE: {}", profile.selected_minimum_node.0);
            println!("MAXIMUM NODE: {}", profile.selected_maximum_node.0);
        }
        Err(error) => println!("CENTRALITY: UNSUPPORTED ({error})"),
    }

    println!("CLAIM CEILING: structural graph and event-log contract only; no agent behavior is inferred.");
    ExitCode::SUCCESS
}
