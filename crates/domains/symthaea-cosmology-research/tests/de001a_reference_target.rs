// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use serde_json::Value;
use std::collections::BTreeSet;
use symthaea_cosmology_research::identity::{GitObjectId, Sha256Digest};

const TARGET: &str = include_str!("../references/de001a_desi_dr2_bao_flat_lcdm_v1.json");

fn target() -> Value {
    serde_json::from_str(TARGET).expect("reference target must remain valid JSON")
}

#[test]
fn reference_target_is_frozen_but_not_yet_executable() {
    let value = target();
    assert_eq!(value["schema_version"], 1);
    assert_eq!(
        value["target_id"],
        "DE-001A-DESI-DR2-BAO-FLAT-LCDM-v1"
    );
    assert_eq!(value["status"], "reference-frozen-not-executable");
    assert_eq!(value["claim_policy"], "reproduction-only");
    assert_eq!(value["model"], "flat-lambda-cdm");
}

#[test]
fn every_recorded_sha256_and_git_identity_is_well_formed() {
    let value = target();

    for key in ["cobaya_commit", "bao_data_commit"] {
        let id = value["upstream_revisions"][key]
            .as_str()
            .expect("upstream commit must be a string");
        GitObjectId::parse(id).expect("upstream commit must be a valid Git object id");
    }

    for artifact in value["input_artifacts"]
        .as_array()
        .expect("input_artifacts must be an array")
    {
        Sha256Digest::parse(
            artifact["sha256"]
                .as_str()
                .expect("input artifact must have sha256"),
        )
        .expect("input sha256 must be valid");
        GitObjectId::parse(
            artifact["git_blob"]
                .as_str()
                .expect("Git-backed input must have git_blob"),
        )
        .expect("input git_blob must be valid");
    }

    for artifact in value["official_reference_artifacts"]
        .as_array()
        .expect("reference artifacts must be an array")
    {
        Sha256Digest::parse(
            artifact["sha256"]
                .as_str()
                .expect("reference artifact must have sha256"),
        )
        .expect("reference sha256 must be valid");
    }
}

#[test]
fn official_desi_reference_hashes_are_pinned_exactly() {
    let value = target();
    let refs = value["official_reference_artifacts"]
        .as_array()
        .expect("reference artifacts must be an array");

    let hash_for = |path: &str| {
        refs.iter()
            .find(|artifact| artifact["path"] == path)
            .and_then(|artifact| artifact["sha256"].as_str())
            .expect("required official reference artifact missing")
    };

    assert_eq!(
        hash_for("iminuit/base/desi-bao-all/bestfit.minimum.txt"),
        "bf8e35e2380ef35b137a77645dcb351af2ed2a93ca8da16c1fd71cb5dd7a1358"
    );
    assert_eq!(
        hash_for("iminuit/base/desi-bao-all/bestfit.minimize.input.yaml"),
        "34499cb78ecaec78db44da9f06f61cd9c9ee497dc5c541b72b48cda54091c6ef"
    );
    assert_eq!(
        hash_for("iminuit/base/desi-bao-all/bestfit.minimize.updated.yaml"),
        "c4c23032d1695635aaea6eb47fabd909006ff32df0a27eadbf64b36c89f31ba1"
    );
    assert_eq!(
        hash_for("iminuit/base/desi-bao-all/minimizer.yaml"),
        "6b51048b359e4b9d646de09379ca273a8d6a1bc1b5a88f585a09d8f2e61f290c"
    );
}

#[test]
fn preregistered_criteria_are_unique_finite_and_positive_tolerance() {
    let value = target();
    let criteria = value["criteria"]
        .as_array()
        .expect("criteria must be an array");
    assert!(!criteria.is_empty());

    let mut seen = BTreeSet::new();
    for criterion in criteria {
        let key = format!(
            "{}::{}",
            criterion["subgate"].as_str().expect("subgate must be a string"),
            criterion["statistic"]
                .as_str()
                .expect("statistic must be a string")
        );
        assert!(seen.insert(key), "duplicate criterion");

        let reference = criterion["reference_value"]
            .as_f64()
            .expect("reference value must be numeric");
        let tolerance = criterion["max_absolute_delta"]
            .as_f64()
            .expect("tolerance must be numeric");
        assert!(reference.is_finite());
        assert!(tolerance.is_finite() && tolerance > 0.0);
    }
}

#[test]
fn flat_lcdm_known_answer_values_match_the_frozen_target() {
    let value = target();
    let criteria = value["criteria"].as_array().unwrap();

    let lookup = |subgate: &str, statistic: &str| {
        criteria
            .iter()
            .find(|criterion| {
                criterion["subgate"] == subgate && criterion["statistic"] == statistic
            })
            .and_then(|criterion| criterion["reference_value"].as_f64())
            .expect("known-answer statistic missing")
    };

    assert!((lookup("DE-001A1", "chi2__BAO_at_published_bestfit") - 10.282299).abs() < 1e-12);
    assert!((lookup("DE-001A2", "omegam") - 0.29717936).abs() < 1e-12);
    assert!((lookup("DE-001A2", "hrdrag") - 101.54786).abs() < 1e-12);
}
