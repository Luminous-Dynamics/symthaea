// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Materialize a custodian-only replay archive and a separate evaluator-only outcome capsule.
//!
//! Prerequisite: run evolutionary_rescue_generate_worlds once to create the frozen source
//! fixtures. This executable loads those fixtures; it does not rerun or mutate the worlds.
//!
//! Usage:
//! cargo run --release --example evolutionary_rescue_split_capsules -p symthaea-futures-ensemble -- <replay-archive-dir> <evaluator-dir> <source-commit-sha> <census|noisy-trait|shuffled-trait>
//!
//! Only the full, custodian-only replay archive and its receipt are written below replay-archive-dir.
//! Hidden outcomes, privileged observations, seeds, split labels and the cross-capsule manifest
//! are written below evaluator-dir. The replay archive includes future observations and MUST NOT
//! be mounted into candidate code. A trusted replay driver must emit only the time-valid prefix
//! for each decision. Directory ACLs/service boundaries remain an operator responsibility.

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
            "usage: <replay-archive-dir> <evaluator-dir> <source-commit-sha> <census|noisy-trait|shuffled-trait>"
                .into(),
        );
    }

    let replay_root = PathBuf::from(&args[0]);
    let evaluator_root = PathBuf::from(&args[1]);
    let source_commit = &args[2];
    let view = match args[3].as_str() {
        "census" => custody::ObservationView::Census,
        "noisy-trait" => custody::ObservationView::NoisyTrait,
        "shuffled-trait" => custody::ObservationView::ShuffledTraitControl,
        other => return Err(format!("unknown observation view {other}")),
    };

    fs::create_dir_all(&replay_root).map_err(|e| {
        format!(
            "create replay archive directory {}: {e}",
            replay_root.display()
        )
    })?;
    fs::create_dir_all(&evaluator_root).map_err(|e| {
        format!(
            "create evaluator directory {}: {e}",
            evaluator_root.display()
        )
    })?;
    ensure_disjoint_roots(&replay_root, &evaluator_root)?;

    let replay_root = fs::canonicalize(&replay_root)
        .map_err(|e| format!("canonicalize replay archive directory: {e}"))?;
    let evaluator_root = fs::canonicalize(&evaluator_root)
        .map_err(|e| format!("canonicalize evaluator directory: {e}"))?;

    ensure_empty_root(&replay_root)?;
    ensure_empty_root(&evaluator_root)?;

    let mut generated = 0usize;
    for (split, seeds) in [
        (
            custody::WorldSplit::Training,
            common::TRAIN_SEEDS.as_slice(),
        ),
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
            let encoded = custody::encode_world(&world, &world_id, split, view, source_commit)?;
            custody::verify_encoded_replay_artifacts(
                &encoded,
                &encoded.replay_digest,
                &encoded.evaluator_digest,
            )?;

            write_new(
                &replay_root.join(format!("{world_id}.observations.json")),
                &encoded.replay_archive_json,
            )?;
            write_new(
                &replay_root.join(format!("{world_id}.receipt.json")),
                &encoded.replay_receipt_json,
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

            let disk_set = custody::EncodedReplayArtifacts {
                replay_archive_json: fs::read(
                    replay_root.join(format!("{world_id}.observations.json")),
                )
                .map_err(|e| format!("read back replay archive {world_id}: {e}"))?,
                replay_receipt_json: fs::read(replay_root.join(format!("{world_id}.receipt.json")))
                    .map_err(|e| format!("read back replay archive receipt {world_id}: {e}"))?,
                evaluator_capsule_json: fs::read(
                    evaluator_root.join(format!("{world_id}.outcomes.json")),
                )
                .map_err(|e| format!("read back evaluator capsule {world_id}: {e}"))?,
                evaluator_manifest_json: fs::read(
                    evaluator_root.join(format!("{world_id}.manifest.json")),
                )
                .map_err(|e| format!("read back evaluator manifest {world_id}: {e}"))?,
                replay_digest: encoded.replay_digest.clone(),
                evaluator_digest: encoded.evaluator_digest.clone(),
            };
            custody::verify_encoded_replay_artifacts(
                &disk_set,
                &encoded.replay_digest,
                &encoded.evaluator_digest,
            )?;

            println!(
                "materialized world_id={world_id} observation_policy={} replay_digest={} evaluator_digest={}",
                view.policy_id(),
                encoded.replay_digest,
                encoded.evaluator_digest
            );
            generated += 1;
        }
    }

    if generated != 10 {
        return Err(format!("generated {generated} worlds; expected exactly 10"));
    }
    println!(
        "done: {generated} replay/evaluator capsule pairs; replay and evaluator roots are disjoint"
    );
    Ok(())
}

fn ensure_disjoint_roots(candidate: &Path, evaluator: &Path) -> Result<(), String> {
    let c = fs::canonicalize(candidate).map_err(|e| format!("canonicalize replay root: {e}"))?;
    let e = fs::canonicalize(evaluator).map_err(|e| format!("canonicalize evaluator root: {e}"))?;
    if c == e || c.starts_with(&e) || e.starts_with(&c) {
        return Err("replay and evaluator roots must be distinct and non-overlapping".into());
    }
    Ok(())
}

fn ensure_empty_root(path: &Path) -> Result<(), String> {
    let entries = fs::read_dir(path)
        .map_err(|e| format!("inspect output directory {}: {e}", path.display()))?;
    if entries.into_iter().next().is_some() {
        return Err(format!(
            "output directory {} must be empty; reconcile/remove prior partial or completed artifacts before retrying",
            path.display()
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_roots_must_be_disjoint() {
        let root = std::env::temp_dir().join(format!("symthaea-custody-{}", uuid::Uuid::new_v4()));
        let nested = root.join("nested");
        fs::create_dir_all(&nested).unwrap();
        assert!(ensure_disjoint_roots(&root, &nested).is_err());
        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn output_roots_must_be_empty_before_materialization() {
        let root = std::env::temp_dir().join(format!("symthaea-custody-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("partial.json"), b"partial").unwrap();
        assert!(
            ensure_empty_root(&root)
                .unwrap_err()
                .contains("must be empty")
        );
        fs::remove_dir_all(&root).unwrap();
    }
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
