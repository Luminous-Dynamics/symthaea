// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Materialize evaluator- and candidate-separated evolutionary-rescue replay capsules.
//!
//! Prerequisite: run evolutionary_rescue_generate_worlds once to create the frozen source
//! fixtures. This executable loads those fixtures; it does not rerun or mutate the worlds.
//!
//! Usage:
//! cargo run --release --example evolutionary_rescue_split_capsules -p symthaea-futures-ensemble -- <candidate-dir> <evaluator-dir> <source-commit-sha> <census|noisy-trait|shuffled-trait>
//!
//! Only candidate observation files and candidate receipts are written below candidate-dir.
//! Hidden outcomes, privileged observations, seeds, split labels and the cross-capsule manifest
//! are written below evaluator-dir. Directory ACLs / service boundaries remain an operator
//! responsibility; separate paths alone are not a security boundary.

#[path = "support/evolutionary_rescue_common.rs"]
mod common;
#[path = "support/evolutionary_rescue_custody.rs"]
mod custody;

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

fn main() {
    if let Err(error) = run() {
        eprintln!("evolutionary-rescue capsule generation failed: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 4 {
        return Err(
            "usage: <candidate-dir> <evaluator-dir> <source-commit-sha> <census|noisy-trait|shuffled-trait>"
                .into(),
        );
    }

    let candidate_root = PathBuf::from(&args[0]);
    let evaluator_root = PathBuf::from(&args[1]);
    let source_commit = &args[2];
    let view = match args[3].as_str() {
        "census" => custody::ObservationView::Census,
        "noisy-trait" => custody::ObservationView::NoisyTrait,
        "shuffled-trait" => custody::ObservationView::ShuffledTraitControl,
        other => return Err(format!("unknown observation view {other}")),
    };

    fs::create_dir_all(&candidate_root)
        .map_err(|e| format!("create candidate directory {}: {e}", candidate_root.display()))?;
    fs::create_dir_all(&evaluator_root)
        .map_err(|e| format!("create evaluator directory {}: {e}", evaluator_root.display()))?;
    ensure_disjoint_roots(&candidate_root, &evaluator_root)?;

    let candidate_root = fs::canonicalize(&candidate_root)
        .map_err(|e| format!("canonicalize candidate directory: {e}"))?;
    let evaluator_root = fs::canonicalize(&evaluator_root)
        .map_err(|e| format!("canonicalize evaluator directory: {e}"))?;

    let mut generated = 0usize;
    for (split, seeds) in [
        (custody::WorldSplit::Training, common::TRAIN_SEEDS.as_slice()),
        (custody::WorldSplit::HeldOut, common::TEST_SEEDS.as_slice()),
    ] {
        for &seed in seeds {
            let world = common::load_world(seed);
            if world.trajectory.len() != common::TICKS as usize + 1 {
                return Err(format!(
                    "seed {seed} has {} trajectory ticks; expected {}",
                    world.trajectory.len(),
                    common::TICKS + 1
                ));
            }

            let world_id = uuid::Uuid::new_v4().to_string();
            let encoded = custody::encode_world(
                &world, &world_id, split, view, source_commit,
            )?;
            custody::verify_encoded_capsules(
                &encoded, &encoded.candidate_digest, &encoded.evaluator_digest,
            )?;

            write_new(
                &candidate_root.join(format!("{world_id}.observations.json")),
                &encoded.candidate_capsule_json,
            )?;
            write_new(
                &candidate_root.join(format!("{world_id}.receipt.json")),
                &encoded.candidate_receipt_json,
            )?;
            write_new(
                &evaluator_root.join(format!("{world_id}.outcomes.json")),
                &encoded.evaluator_capsule_json,
            )?;
            // The manifest is written last. A bundle without its manifest is incomplete and
            // must not be admitted to a replay campaign.
            write_new(
                &evaluator_root.join(format!("{world_id}.manifest.json")),
                &encoded.evaluator_manifest_json,
            )?;

            let disk_set = custody::EncodedCapsuleSet {
                candidate_capsule_json: fs::read(candidate_root.join(format!("{world_id}.observations.json")))
                    .map_err(|e| format!("read back candidate capsule {world_id}: {e}"))?,
                candidate_receipt_json: fs::read(candidate_root.join(format!("{world_id}.receipt.json")))
                    .map_err(|e| format!("read back candidate receipt {world_id}: {e}"))?,
                evaluator_capsule_json: fs::read(evaluator_root.join(format!("{world_id}.outcomes.json")))
                    .map_err(|e| format!("read back evaluator capsule {world_id}: {e}"))?,
                evaluator_manifest_json: fs::read(evaluator_root.join(format!("{world_id}.manifest.json")))
                    .map_err(|e| format!("read back evaluator manifest {world_id}: {e}"))?,
                candidate_digest: encoded.candidate_digest.clone(),
                evaluator_digest: encoded.evaluator_digest.clone(),
            };
            custody::verify_encoded_capsules(
                &disk_set, &encoded.candidate_digest, &encoded.evaluator_digest,
            )?;

            println!(
                "materialized world_id={world_id} observation_policy={} candidate_digest={} evaluator_digest={}",
                view.policy_id(), encoded.candidate_digest, encoded.evaluator_digest
            );
            generated += 1;
        }
    }

    if generated != 10 {
        return Err(format!("generated {generated} worlds; expected exactly 10"));
    }
    println!(
        "done: {generated} candidate/evaluator capsule pairs; candidate and evaluator roots are disjoint"
    );
    Ok(())
}

fn ensure_disjoint_roots(candidate: &Path, evaluator: &Path) -> Result<(), String> {
    let c = fs::canonicalize(candidate)
        .map_err(|e| format!("canonicalize candidate root: {e}"))?;
    let e = fs::canonicalize(evaluator)
        .map_err(|e| format!("canonicalize evaluator root: {e}"))?;
    if c == e || c.starts_with(&e) || e.starts_with(&c) {
        return Err("candidate and evaluator roots must be distinct and non-overlapping".into());
    }
    Ok(())
}

fn write_new(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|e| format!("create {} without overwrite: {e}", path.display()))?;
    file.write_all(bytes)
        .map_err(|e| format!("write {}: {e}", path.display()))?;
    file.sync_all()
        .map_err(|e| format!("sync {}: {e}", path.display()))?;
    Ok(())
}
