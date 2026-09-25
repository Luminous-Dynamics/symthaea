// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Qualification-only compatibility adapter for the PIE #1619 JSON dialect.
//!
//! This example exists so the independent Python oracle and Rust candidate can
//! consume the same fixture payload. It is not part of the production closure API.

use serde::Deserialize;
use serde_json::{json, Value};
use symthaea_capability_closure::{
    evaluate_import_leverage, evaluate_reachability, CapabilityId, ClosureProblemV1,
    ClosureStatus, RouteId, RouteV1,
};

#[derive(Debug, Deserialize)]
struct RecipeWire {
    recipe_id: String,
    output: String,
    requires: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct ProblemWire {
    #[serde(default)]
    local_primitives: Vec<String>,
    #[serde(default)]
    imports: Vec<String>,
    #[serde(default)]
    targets: Vec<String>,
    #[serde(default)]
    recipes: Vec<RecipeWire>,
}

fn capability_ids(values: Vec<String>) -> Result<Vec<CapabilityId>, String> {
    values
        .into_iter()
        .map(|value| CapabilityId::new(value).map_err(|error| error.to_string()))
        .collect()
}

fn status_name(status: ClosureStatus) -> &'static str {
    match status {
        ClosureStatus::LocallyClosed => "LocallyClosed",
        ClosureStatus::ImportDependent => "ImportDependent",
        ClosureStatus::Unavailable => "Unavailable",
    }
}

fn evaluate_payload(payload: &str) -> Result<Value, String> {
    let wire: ProblemWire = serde_json::from_str(payload).map_err(|error| error.to_string())?;

    let local_primitives = capability_ids(wire.local_primitives)?;
    let imports = capability_ids(wire.imports)?;
    let targets = capability_ids(wire.targets)?;

    let routes = wire
        .recipes
        .into_iter()
        .map(|recipe| {
            let id = RouteId::new(recipe.recipe_id).map_err(|error| error.to_string())?;
            let output =
                CapabilityId::new(recipe.output).map_err(|error| error.to_string())?;
            let required_capabilities = capability_ids(recipe.requires)?;
            RouteV1::new(id, output, required_capabilities).map_err(|error| error.to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;

    let problem = ClosureProblemV1::new(local_primitives, imports, targets, routes)
        .map_err(|error| error.to_string())?;

    let reachability = evaluate_reachability(&problem);
    let import_leverage = evaluate_import_leverage(&problem);

    let locally_reproducible = reachability
        .locally_reproducible
        .iter()
        .map(|id| id.as_str())
        .collect::<Vec<_>>();
    let operationally_reachable = reachability
        .operationally_reachable
        .iter()
        .map(|id| id.as_str())
        .collect::<Vec<_>>();
    let targets = reachability
        .targets
        .iter()
        .map(|entry| {
            json!({
                "target": entry.target.as_str(),
                "status": status_name(entry.status),
            })
        })
        .collect::<Vec<_>>();
    let import_leverage = import_leverage
        .iter()
        .map(|entry| {
            json!({
                "import_id": entry.import.as_str(),
                "targets_lost_if_removed": entry
                    .targets_lost_if_removed
                    .iter()
                    .map(|id| id.as_str())
                    .collect::<Vec<_>>(),
                "capability_count_lost": entry.capability_count_lost,
            })
        })
        .collect::<Vec<_>>();

    Ok(json!({
        "locally_reproducible": locally_reproducible,
        "operationally_reachable": operationally_reachable,
        "targets": targets,
        "import_leverage": import_leverage,
    }))
}

fn main() {
    let mut args = std::env::args().skip(1);
    let Some(flag) = args.next() else {
        eprintln!("usage: reference_compat --json '<payload>'");
        std::process::exit(2);
    };
    let Some(payload) = args.next() else {
        eprintln!("usage: reference_compat --json '<payload>'");
        std::process::exit(2);
    };
    if flag != "--json" || args.next().is_some() {
        eprintln!("usage: reference_compat --json '<payload>'");
        std::process::exit(2);
    }

    let output = match evaluate_payload(&payload) {
        Ok(result) => json!({
            "disposition": "Valid",
            "result": result,
        }),
        Err(_) => json!({
            "disposition": "InvalidInput",
        }),
    };

    println!("{}", serde_json::to_string(&output).expect("JSON serialization must succeed"));
}
