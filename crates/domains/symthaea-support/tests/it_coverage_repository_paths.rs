// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::path::{Path, PathBuf};
use symthaea_support::{seed_it_coverage_inventory_v1, ItCoverageSourceV1};

fn workspace_root() -> PathBuf {
    // crates/domains/symthaea-support -> repository root
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .canonicalize()
        .expect("workspace root must be resolvable")
}

#[test]
fn every_declared_repository_path_exists_in_the_checkout() {
    let inventory = seed_it_coverage_inventory_v1("checkout-under-test", 1)
        .expect("seed IT coverage inventory must validate");
    let root = workspace_root();
    let mut missing = Vec::new();

    for domain in &inventory.domains {
        for signal in &domain.signals {
            if let ItCoverageSourceV1::RepositoryPath(relative) = &signal.source {
                if !root.join(relative).exists() {
                    missing.push(format!("{:?}: {relative}", domain.domain));
                }
            }
        }
    }

    assert!(
        missing.is_empty(),
        "IT coverage inventory contains stale repository paths:\n{}",
        missing.join("\n")
    );
}
