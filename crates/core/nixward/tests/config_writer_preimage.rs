// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Regression fixtures for ConfigWriter state/pre-image binding.
//!
//! These tests intentionally use dry-run mode so the state-binding contract can
//! be exercised without requiring `nix-instantiate` or touching real NixOS files.

use nixward::action::ConfigWriter;
use std::fs;
use std::io::ErrorKind;

const INITIAL: &str = r#"{ config, pkgs, ... }:
{
  services.openssh.enable = true;
  networking.firewall.enable = true;
}
"#;

const EXTERNAL_EDIT: &str = r#"{ config, pkgs, ... }:
{
  services.openssh.enable = true;
  networking.firewall.enable = false;
  services.nginx.enable = true;
}
"#;

fn writer_for(dir: &tempfile::TempDir) -> ConfigWriter {
    ConfigWriter::new()
        .with_config_root(dir.path())
        .with_git_backup(false)
        .with_dry_run(true)
}

#[test]
fn stale_planned_patch_is_rejected_and_external_edit_is_preserved() {
    let dir = tempfile::tempdir().unwrap();
    let target = dir.path().join("configuration.nix");
    fs::write(&target, INITIAL).unwrap();

    let writer = writer_for(&dir);
    let patch = writer
        .set_option("services.openssh.enable", "false")
        .expect("plan patch against initial pre-image");

    // Simulate another actor changing the file after planning but before apply.
    fs::write(&target, EXTERNAL_EDIT).unwrap();

    let err = writer
        .apply_patch(&patch)
        .expect_err("stale patch must fail closed instead of applying against foreign state");

    assert_eq!(
        err.kind(),
        ErrorKind::WouldBlock,
        "stale pre-image should have a distinct fail-closed error class"
    );
    assert!(
        err.to_string().contains("stale") || err.to_string().contains("changed since planning"),
        "error should explain that the patch pre-image is no longer current: {err}"
    );

    let on_disk = fs::read_to_string(&target).unwrap();
    assert_eq!(
        on_disk, EXTERNAL_EDIT,
        "an intervening edit must never be replaced by a stale planned patch"
    );
}

#[test]
fn stale_noop_patch_is_not_reported_as_current_success() {
    let dir = tempfile::tempdir().unwrap();
    let target = dir.path().join("configuration.nix");
    fs::write(&target, INITIAL).unwrap();

    let writer = writer_for(&dir);
    let patch = writer
        .set_option("services.openssh.enable", "true")
        .expect("plan a semantic no-op against the initial pre-image");
    assert!(patch.is_noop(), "fixture must actually produce a no-op patch");

    fs::write(&target, EXTERNAL_EDIT).unwrap();

    let err = writer
        .apply_patch(&patch)
        .expect_err("historical no-op must not masquerade as a current no-op after drift");
    assert_eq!(err.kind(), ErrorKind::WouldBlock);
    assert_eq!(fs::read_to_string(&target).unwrap(), EXTERNAL_EDIT);
}
