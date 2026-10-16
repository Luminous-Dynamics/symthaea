// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Qualification-only projection for computer-production capability profiles.
//!
//! Run with:
//! cargo run --locked -p symthaea-capability-closure --example computer_profile_projection
//!
//! This projects preregistered structural profiles into the canonical V1 closure
//! kernel. It does not validate a physical build, host removal, component stock
//! quantities, calibration adequacy, tool safety, or semiconductor manufacture.

use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use symthaea_capability_closure::{
    evaluate_import_leverage, evaluate_reachability, ClosureProblemV1,
};

const EXPECTED_SCHEMA: &str = "computer-profile-closure-projection-v0.1";
const EXPECTED_ATLAS_CORPUS_BLOB: &str = "e9a7a8bce54d0ec89151ee011a5bb3d7002b5134";

#[derive(Debug, Deserialize)]
struct SourcePin {
    repository: String,
    path: String,
    git_blob_sha: String,
}

#[derive(Debug, Deserialize)]
struct Corpus {
    schema_version: String,
    source_atlas_corpus: SourcePin,
    disclaimer: String,
    fixtures: Vec<Fixture>,
}

#[derive(Debug, Deserialize)]
struct Fixture {
    id: String,
    equivalence_group: Option<String>,
    #[serde(default)]
    unresolved_prerequisites: Vec<String>,
    expected_disposition: String,
    problem: Option<ClosureProblemV1>,
    expected: Expected,
}

#[derive(Debug, Deserialize)]
struct Expected {
    reachability: Option<Value>,
    import_leverage: Option<Value>,
}

#[derive(Debug)]
struct Observed {
    disposition: &'static str,
    reachability: Value,
    import_leverage: Value,
}

fn observe(fixture: &Fixture) -> Result<Observed, String> {
    if !fixture.unresolved_prerequisites.is_empty() {
        if fixture.expected_disposition != "UnresolvedInput" {
            return Err(format!(
                "{}: unresolved prerequisites require UnresolvedInput disposition",
                fixture.id
            ));
        }
        if fixture.problem.is_some() {
            return Err(format!(
                "{}: unresolved input must be held before V1 closure evaluation",
                fixture.id
            ));
        }
        if fixture.expected.reachability.is_some() || fixture.expected.import_leverage.is_some() {
            return Err(format!(
                "{}: unresolved input must not carry fabricated closure reports",
                fixture.id
            ));
        }
        return Ok(Observed {
            disposition: "UnresolvedInput",
            reachability: Value::Null,
            import_leverage: Value::Null,
        });
    }

    if fixture.expected_disposition != "Valid" {
        return Err(format!(
            "{}: fixture without unresolved prerequisites must declare Valid",
            fixture.id
        ));
    }
    let problem = fixture
        .problem
        .as_ref()
        .ok_or_else(|| format!("{}: Valid fixture is missing its problem", fixture.id))?;
    let reachability = serde_json::to_value(evaluate_reachability(problem))
        .map_err(|error| format!("{}: serialize reachability: {error}", fixture.id))?;
    let import_leverage = serde_json::to_value(evaluate_import_leverage(problem))
        .map_err(|error| format!("{}: serialize import leverage: {error}", fixture.id))?;

    if fixture.expected.reachability.as_ref() != Some(&reachability) {
        return Err(format!(
            "{}: reachability mismatch\nexpected: {}\nactual:   {}",
            fixture.id,
            fixture.expected.reachability.as_ref().unwrap_or(&Value::Null),
            reachability
        ));
    }
    if fixture.expected.import_leverage.as_ref() != Some(&import_leverage) {
        return Err(format!(
            "{}: import-leverage mismatch\nexpected: {}\nactual:   {}",
            fixture.id,
            fixture.expected.import_leverage.as_ref().unwrap_or(&Value::Null),
            import_leverage
        ));
    }

    Ok(Observed {
        disposition: "Valid",
        reachability,
        import_leverage,
    })
}

fn run() -> Result<Value, String> {
    let corpus: Corpus = serde_json::from_str(include_str!(
        "../fixtures/computer-profile-projection-v0.1.json"
    ))
    .map_err(|error| format!("parse computer projection corpus: {error}"))?;

    if corpus.schema_version != EXPECTED_SCHEMA {
        return Err(format!(
            "schema mismatch: expected {EXPECTED_SCHEMA}, got {}",
            corpus.schema_version
        ));
    }
    if corpus.source_atlas_corpus.git_blob_sha != EXPECTED_ATLAS_CORPUS_BLOB {
        return Err(format!(
            "Atlas corpus blob mismatch: expected {EXPECTED_ATLAS_CORPUS_BLOB}, got {}",
            corpus.source_atlas_corpus.git_blob_sha
        ));
    }
    if corpus.source_atlas_corpus.repository != "Luminous-Dynamics/mycelix"
        || corpus.source_atlas_corpus.path
            != "docs/research/CIV_BOOT_COMPUTER_REPRO_CORPUS_V0_1.json"
    {
        return Err("unexpected Atlas source identity".to_owned());
    }
    if corpus.fixtures.is_empty() {
        return Err("projection corpus must not be empty".to_owned());
    }

    let mut seen_ids = std::collections::BTreeSet::new();
    let mut equivalence_groups: BTreeMap<String, (Value, Value, String)> = BTreeMap::new();
    let mut reports = Vec::with_capacity(corpus.fixtures.len());
    let mut unresolved_count = 0usize;

    for fixture in &corpus.fixtures {
        if !seen_ids.insert(fixture.id.clone()) {
            return Err(format!("duplicate fixture id: {}", fixture.id));
        }
        let observed = observe(fixture)?;
        if observed.disposition == "UnresolvedInput" {
            unresolved_count += 1;
        }

        if let Some(group) = &fixture.equivalence_group {
            let current = (
                observed.reachability.clone(),
                observed.import_leverage.clone(),
                fixture.id.clone(),
            );
            if let Some((reachability, leverage, first_id)) = equivalence_groups.get(group) {
                if reachability != &current.0 || leverage != &current.1 {
                    return Err(format!(
                        "equivalence group {group:?} differs between {first_id} and {}",
                        fixture.id
                    ));
                }
            } else {
                equivalence_groups.insert(group.clone(), current);
            }
        }

        reports.push(json!({
            "fixture_id": fixture.id,
            "disposition": observed.disposition,
            "reachability": observed.reachability,
            "import_leverage": observed.import_leverage
        }));
    }

    Ok(json!({
        "schema_version": EXPECTED_SCHEMA,
        "status": "pass",
        "fixtures_checked": reports.len(),
        "unresolved_before_kernel": unresolved_count,
        "equivalence_groups_checked": equivalence_groups.len(),
        "results": reports,
        "disclaimer": corpus.disclaimer,
        "qualification_boundary": {
            "structural_closure": "evaluated",
            "physical_C4_or_C5": "not evaluated",
            "inventory_quantity_or_runway": "not evaluated",
            "calibration_adequacy": "not evaluated",
            "PCB_or_semiconductor_fabrication": "not inferred",
            "authority_to_operate_equipment": "not granted"
        }
    }))
}

fn main() {
    match run() {
        Ok(report) => println!(
            "{}",
            serde_json::to_string_pretty(&report).expect("JSON report serialization")
        ),
        Err(error) => {
            eprintln!("computer profile projection failed closed: {error}");
            std::process::exit(1);
        }
    }
}
