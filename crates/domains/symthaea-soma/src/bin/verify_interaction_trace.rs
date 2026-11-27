// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Verify cross-event causal invariants in a Symthaea interaction trace.
//!
//! This validates event-to-event relationships. It is not a JSON Schema
//! validator, dataset scorer, device attestation tool, or observer trust root.

use std::env;
use std::fs;
use std::path::PathBuf;
use std::process;

use symthaea_soma::embodiment_interaction_trace::InteractionTraceVerifier;

fn main() {
    let mut args = env::args_os();
    let program = args
        .next()
        .unwrap_or_else(|| "verify_interaction_trace".into());
    let Some(path) = args.next() else {
        eprintln!(
            "Usage: {} <interaction-trace.json>",
            PathBuf::from(program).display()
        );
        process::exit(2);
    };
    if args.next().is_some() {
        eprintln!(
            "Usage: {} <interaction-trace.json>",
            PathBuf::from(program).display()
        );
        process::exit(2);
    }

    let path = PathBuf::from(path);
    let input = match fs::read_to_string(&path) {
        Ok(input) => input,
        Err(error) => {
            eprintln!("cannot read {}: {error}", path.display());
            process::exit(2);
        }
    };

    match InteractionTraceVerifier::verify_json(&input) {
        Ok(summary) => {
            println!(
                "valid causal trace: events={} inputs={} cues={} turn_decisions={} authorizations={} submitted={} frames={} stopped={} grounding_probes={}",
                summary.event_count,
                summary.input_chunks,
                summary.semantic_cues,
                summary.turn_decisions,
                summary.authorizations,
                summary.submitted_outputs,
                summary.presented_frames,
                summary.stopped_outputs,
                summary.grounding_probes,
            );
        }
        Err(error) => {
            eprintln!("invalid causal trace {}: {error}", path.display());
            process::exit(1);
        }
    }
}
