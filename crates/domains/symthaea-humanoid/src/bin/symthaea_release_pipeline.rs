// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};
use symthaea_humanoid::{
    HumanoidReleasePipelinePolicy, ReleaseStageEvidence, evaluate_release_pipeline,
};

fn main() -> anyhow::Result<()> {
    let args = std::env::args().collect::<Vec<_>>();
    if args.len() != 4 {
        anyhow::bail!(
            "usage: symthaea_release_pipeline <release-id> <stage-evidence.json> <report.json>"
        );
    }
    let stages: Vec<ReleaseStageEvidence> = serde_json::from_slice(&fs::read(&args[2])?)?;
    let now = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis() as u64;
    let report = evaluate_release_pipeline(
        &args[1],
        &HumanoidReleasePipelinePolicy::default(),
        stages,
        now,
    );
    fs::write(&args[3], serde_json::to_vec_pretty(&report)?)?;
    if !report.passed {
        anyhow::bail!(
            "release certification failed: {}",
            report.failures.join("; ")
        );
    }
    Ok(())
}
