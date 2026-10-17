// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Independent structural verifier for the frozen synthetic case manifest.
//!
//! This program does not import or call the platform assessment kernel. It
//! checks corpus identity, basic record shape, required task-family presence,
//! and a pinned non-cryptographic FNV-1a regression fingerprint. The FNV
//! fingerprint is not a substitute for a cryptographic artifact digest.

use std::collections::BTreeSet;
use std::process::ExitCode;

const MANIFEST: &str = include_str!("../../fixtures/scenario_manifest.tsv");
const EXPECTED_CASE_COUNT: usize = 53;
const EXPECTED_FNV1A64: u64 = 0xb265_2a70_4ccd_153d;
const HEADER: &str = "case_id\texpected_status\trationale";

const REQUIRED_IDS: &[&str] = &[
    "baseline-feasible",
    "manual-current-practice-feasible",
    "missing-task-evidence",
    "stale-geometry-evidence",
    "conflicting-terrain-evidence",
    "malformed-verified-digest",
    "known-slope-plus-estimated-cost",
    "cross-currency-comparison",
    "broad-acre-tractor-treatment",
    "orchard-canopy-gnss-denied-scouting",
    "greenhouse-rail-harvest",
    "nursery-gantry-handling",
    "livestock-specific-monitoring",
    "packhouse-produce-transport",
    "fixed-sensor-water-monitoring",
    "aerial-field-survey",
    "smallholder-mechanical-weeding",
    "agrivoltaic-access-rejects-wide-tractor",
];

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn verify() -> Result<(), String> {
    let actual_fingerprint = fnv1a64(MANIFEST.as_bytes());
    if actual_fingerprint != EXPECTED_FNV1A64 {
        return Err(format!(
            "manifest fingerprint changed: expected {EXPECTED_FNV1A64:016x}, got {actual_fingerprint:016x}; review the corpus change and update the pin intentionally"
        ));
    }

    let lines: Vec<&str> = MANIFEST
        .lines()
        .filter(|line| !line.trim().is_empty() && !line.starts_with('#'))
        .collect();
    if lines.first().copied() != Some(HEADER) {
        return Err("manifest header is missing or has drifted".into());
    }

    let rows = &lines[1..];
    if rows.len() != EXPECTED_CASE_COUNT {
        return Err(format!(
            "expected {EXPECTED_CASE_COUNT} frozen cases, found {}",
            rows.len()
        ));
    }

    let mut ids = BTreeSet::new();
    let mut statuses = [0_usize; 4];
    for (index, row) in rows.iter().enumerate() {
        let mut columns = row.split('\t');
        let id = columns.next().unwrap_or_default();
        let status = columns.next().unwrap_or_default();
        let rationale = columns.next().unwrap_or_default();

        if id.trim().is_empty() || rationale.trim().is_empty() || columns.next().is_some() {
            return Err(format!("malformed row {}: {row}", index + 2));
        }
        if !ids.insert(id) {
            return Err(format!("duplicate case id: {id}"));
        }

        let status_index = match status {
            "FEASIBLE" => 0,
            "PROVISIONAL" => 1,
            "REJECTED" => 2,
            "BLOCKED" => 3,
            _ => return Err(format!("unsupported status {status:?} at row {}", index + 2)),
        };
        statuses[status_index] += 1;
    }

    for id in REQUIRED_IDS {
        if !ids.contains(id) {
            return Err(format!("required negative-control or task-family case absent: {id}"));
        }
    }

    println!("contract=AGRI-ROBOTICS-BENCH-V1");
    println!("manifest_cases={}", rows.len());
    println!("feasible={}", statuses[0]);
    println!("provisional={}", statuses[1]);
    println!("rejected={}", statuses[2]);
    println!("blocked={}", statuses[3]);
    println!("manifest_fnv1a64={actual_fingerprint:016x} (non-cryptographic fingerprint)");
    println!("required_case_ids={}", REQUIRED_IDS.len());
    println!("result=STRUCTURE_VALID_ONLY");
    Ok(())
}

fn main() -> ExitCode {
    match verify() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("manifest verifier failed: {error}");
            ExitCode::FAILURE
        }
    }
}
