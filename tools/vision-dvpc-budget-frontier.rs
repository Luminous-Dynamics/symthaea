// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! VIS-DVPC-016: safety-gated latency/energy Pareto analysis across budget sweeps.
//!
//! The tool analyzes already-committed summary rows. It does not recompute the
//! upstream risk certificate, validate hardware meters, or make performance claims.
//!
//! Usage:
//!   rustc --edition=2024 --test tools/vision-dvpc-budget-frontier.rs -o /tmp/vis-dvpc-016-tests
//!   /tmp/vis-dvpc-016-tests
//!   rustc --edition=2024 -O tools/vision-dvpc-budget-frontier.rs -o /tmp/vis-dvpc-016
//!   /tmp/vis-dvpc-016 SWEEP.tsv EXPECTED_TRACE_BUNDLE_LOCK_SHA256 EXPECTED_SWEEP_FILE_SHA256

#![deny(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fmt::Write as FmtWrite;
use std::fs;

const FORMAT_VERSION: &str = "VIS-DVPC-BUDGET-SWEEP-001-v1";
const RECEIPT_VERSION: &str = "VIS-DVPC-PARETO-RECEIPT-001-v1";
const META_KEYS: &[&str] = &[
    "format",
    "experiment_id",
    "dataset_id",
    "dataset_revision",
    "split",
    "utility_spec_sha256",
    "bundle_lock_sha256",
    "paired_unit_set_sha256",
    "policy_count",
    "budget_point_count",
    "expected_evaluated_traces",
    "expected_independent_clusters",
    "harm_rate_limit",
    "cvar95_loss_limit",
    "safety_violations_max",
];
const SWEEP_HEADER: &[&str] = &[
    "SWEEP_HEADER",
    "budget_point_id",
    "policy_id",
    "latency_budget_us",
    "energy_budget_uj",
    "utility_delta_vs_all_cheap",
    "actual_all_in_latency_us",
    "actual_all_in_energy_uj",
    "allocator_latency_us",
    "allocator_energy_uj",
    "latency_budget_violations",
    "energy_budget_violations",
    "cost_measurement_complete",
    "safety_violations",
    "cvar95_incremental_loss",
    "pooled_harm_upper_bound",
    "worst_group_harm_upper_bound",
    "risk_certificate_state",
    "paired_unit_set_sha256",
    "evaluated_traces",
    "independent_clusters",
];

#[derive(Debug, Clone)]
struct Table {
    meta: BTreeMap<String, String>,
    rows: Vec<Vec<String>>,
}

#[derive(Debug, Clone)]
struct RunPoint {
    budget_point_id: String,
    policy_id: String,
    latency_budget_us: u64,
    energy_budget_uj: u64,
    utility_delta: f64,
    actual_latency_us: u64,
    actual_energy_uj: u64,
    allocator_latency_us: u64,
    allocator_energy_uj: u64,
    latency_budget_violations: u64,
    energy_budget_violations: u64,
    cost_measurement_complete: bool,
    safety_violations: u64,
    cvar95_loss: f64,
    pooled_harm_upper: f64,
    worst_group_harm_upper: f64,
    risk_certificate_state: String,
    evaluated_traces: u64,
    independent_clusters: u64,
}

#[derive(Debug, Clone)]
struct Constraints {
    policy_count: usize,
    budget_point_count: usize,
    expected_evaluated_traces: u64,
    expected_independent_clusters: u64,
    harm_rate_limit: f64,
    cvar95_loss_limit: f64,
    safety_violations_max: u64,
    paired_unit_set_sha256: String,
}

#[derive(Debug, Clone)]
struct ClassifiedPoint {
    point: RunPoint,
    status: &'static str,
    reasons: Vec<&'static str>,
    dominated_by: Option<(String, String)>,
}

fn invalid(message: impl Into<String>) -> String {
    message.into()
}

fn sha256_hex(bytes: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
        0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
        0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
        0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
        0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
        0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
        0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
        0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    ];
    let bit_len = (bytes.len() as u64).wrapping_mul(8);
    let mut message = bytes.to_vec();
    message.push(0x80);
    while message.len() % 64 != 56 {
        message.push(0);
    }
    message.extend_from_slice(&bit_len.to_be_bytes());

    let mut state = [
        0x6a09e667u32, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    ];
    for chunk in message.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (i, bytes) in chunk.chunks_exact(4).enumerate() {
            w[i] = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        }
        for i in 16..64 {
            let x = w[i - 15];
            let y = w[i - 2];
            let s0 = x.rotate_right(7) ^ x.rotate_right(18) ^ (x >> 3);
            let s1 = y.rotate_right(17) ^ y.rotate_right(19) ^ (y >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = state;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = h.wrapping_add(s1).wrapping_add(ch).wrapping_add(K[i]).wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            h = g; g = f; f = e; e = d.wrapping_add(t1);
            d = c; c = b; b = a; a = t1.wrapping_add(t2);
        }
        for (slot, value) in state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
            *slot = slot.wrapping_add(value);
        }
    }
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(64);
    for value in state {
        for byte in value.to_be_bytes() {
            out.push(HEX[(byte >> 4) as usize] as char);
            out.push(HEX[(byte & 0x0f) as usize] as char);
        }
    }
    out
}

fn validate_sha256(value: &str, field: &str) -> Result<String, String> {
    if value.len() != 64 || !value.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(invalid(format!("{field} must be a 64-character hexadecimal SHA-256 digest")));
    }
    Ok(value.to_ascii_lowercase())
}

fn validate_token(value: &str, field: &str) -> Result<(), String> {
    if value.is_empty()
        || !value.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b':'))
    {
        return Err(invalid(format!("{field} must be a non-empty token without whitespace or paths")));
    }
    Ok(())
}

fn parse_u64(value: &str, field: &str, row: usize) -> Result<u64, String> {
    value.parse::<u64>().map_err(|_| invalid(format!("row {row}: {field} must be a non-negative integer")))
}

fn parse_f64(value: &str, field: &str, row: usize) -> Result<f64, String> {
    let parsed = value.parse::<f64>().map_err(|_| invalid(format!("row {row}: {field} must be numeric")))?;
    if !parsed.is_finite() {
        return Err(invalid(format!("row {row}: {field} must be finite")));
    }
    Ok(parsed)
}

fn parse_bool(value: &str, field: &str, row: usize) -> Result<bool, String> {
    match value {
        "true" => Ok(true),
        "false" => Ok(false),
        _ => Err(invalid(format!("row {row}: {field} must be true or false"))),
    }
}

fn parse_table(input: &str) -> Result<Table, String> {
    let mut meta = BTreeMap::new();
    let mut rows = Vec::new();
    let mut header_seen = false;
    for (index, line) in input.lines().enumerate() {
        let line_number = index + 1;
        if line.is_empty() {
            return Err(invalid(format!("line {line_number}: blank lines are forbidden")));
        }
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.first().copied() == Some("META") {
            if header_seen {
                return Err(invalid(format!("line {line_number}: metadata after header is forbidden")));
            }
            if fields.len() != 3 || fields[1].is_empty() || fields[2].is_empty() {
                return Err(invalid(format!("line {line_number}: META requires exactly key and value")));
            }
            if meta.insert(fields[1].to_owned(), fields[2].to_owned()).is_some() {
                return Err(invalid(format!("line {line_number}: duplicate metadata key {}", fields[1])));
            }
            continue;
        }
        if !header_seen {
            if fields.as_slice() != SWEEP_HEADER {
                return Err(invalid("exact SWEEP header required; unknown/reordered columns fail closed"));
            }
            header_seen = true;
            continue;
        }
        if fields.first().copied() != Some("SWEEP") || fields.len() != SWEEP_HEADER.len() {
            return Err(invalid(format!("line {line_number}: expected SWEEP row with {} fields", SWEEP_HEADER.len())));
        }
        if fields.iter().any(|f| f.is_empty()) {
            return Err(invalid(format!("line {line_number}: empty fields are forbidden")));
        }
        rows.push(fields.iter().map(|f| (*f).to_owned()).collect());
    }
    if !header_seen || rows.is_empty() {
        return Err(invalid("missing SWEEP header or data rows"));
    }
    let expected: BTreeSet<&str> = META_KEYS.iter().copied().collect();
    let actual: BTreeSet<&str> = meta.keys().map(String::as_str).collect();
    if expected != actual {
        return Err(invalid("META key set mismatch: missing or unknown fields"));
    }
    if meta.get("format").map(String::as_str) != Some(FORMAT_VERSION) {
        return Err(invalid("unsupported budget-sweep format"));
    }
    if !matches!(meta.get("split").map(String::as_str), Some("test" | "private_test")) {
        return Err(invalid("only test or private_test summaries are admitted"));
    }
    for key in ["experiment_id", "dataset_id", "dataset_revision"] {
        validate_token(&meta[key], key)?;
    }
    validate_sha256(&meta["utility_spec_sha256"], "utility_spec_sha256")?;
    validate_sha256(&meta["bundle_lock_sha256"], "bundle_lock_sha256")?;
    Ok(Table { meta, rows })
}

fn parse_constraints(meta: &BTreeMap<String, String>) -> Result<Constraints, String> {
    let policy_count = meta["policy_count"].parse::<usize>()
        .map_err(|_| invalid("policy_count must be a positive integer"))?;
    let budget_point_count = meta["budget_point_count"].parse::<usize>()
        .map_err(|_| invalid("budget_point_count must be a positive integer"))?;
    if policy_count < 2 || budget_point_count < 2 {
        return Err(invalid("at least two policies and two budget points are required"));
    }
    let expected_evaluated_traces = meta["expected_evaluated_traces"].parse::<u64>()
        .map_err(|_| invalid("expected_evaluated_traces must be an integer"))?;
    let expected_independent_clusters = meta["expected_independent_clusters"].parse::<u64>()
        .map_err(|_| invalid("expected_independent_clusters must be an integer"))?;
    if expected_evaluated_traces == 0 || expected_independent_clusters < 5 {
        return Err(invalid("expected traces must be positive and at least five independent clusters are required"));
    }
    let harm_rate_limit = meta["harm_rate_limit"].parse::<f64>()
        .map_err(|_| invalid("harm_rate_limit must be numeric"))?;
    if !harm_rate_limit.is_finite() || !(0.0..=1.0).contains(&harm_rate_limit) {
        return Err(invalid("harm_rate_limit must be finite and in [0,1]"));
    }
    let cvar95_loss_limit = meta["cvar95_loss_limit"].parse::<f64>()
        .map_err(|_| invalid("cvar95_loss_limit must be numeric"))?;
    if !cvar95_loss_limit.is_finite() || cvar95_loss_limit < 0.0 {
        return Err(invalid("cvar95_loss_limit must be finite and non-negative"));
    }
    let safety_violations_max = meta["safety_violations_max"].parse::<u64>()
        .map_err(|_| invalid("safety_violations_max must be an integer"))?;
    let paired_unit_set_sha256 =
        validate_sha256(&meta["paired_unit_set_sha256"], "paired_unit_set_sha256")?;
    Ok(Constraints {
        policy_count,
        budget_point_count,
        expected_evaluated_traces,
        expected_independent_clusters,
        harm_rate_limit,
        cvar95_loss_limit,
        safety_violations_max,
        paired_unit_set_sha256,
    })
}

fn parse_run(row: &[String], row_number: usize, c: &Constraints) -> Result<RunPoint, String> {
    let token_positions = [(1, "budget_point_id"), (2, "policy_id")];
    for (position, field) in token_positions {
        validate_token(&row[position], field)?;
    }
    let latency_budget_us = parse_u64(&row[3], "latency_budget_us", row_number)?;
    let energy_budget_uj = parse_u64(&row[4], "energy_budget_uj", row_number)?;
    if latency_budget_us == 0 || energy_budget_uj == 0 {
        return Err(invalid(format!("row {row_number}: budget ceilings must be positive")));
    }
    let utility_delta = parse_f64(&row[5], "utility_delta_vs_all_cheap", row_number)?;
    let actual_latency_us = parse_u64(&row[6], "actual_all_in_latency_us", row_number)?;
    let actual_energy_uj = parse_u64(&row[7], "actual_all_in_energy_uj", row_number)?;
    let allocator_latency_us = parse_u64(&row[8], "allocator_latency_us", row_number)?;
    let allocator_energy_uj = parse_u64(&row[9], "allocator_energy_uj", row_number)?;
    let latency_budget_violations = parse_u64(&row[10], "latency_budget_violations", row_number)?;
    let energy_budget_violations = parse_u64(&row[11], "energy_budget_violations", row_number)?;
    let cost_measurement_complete = parse_bool(&row[12], "cost_measurement_complete", row_number)?;
    let safety_violations = parse_u64(&row[13], "safety_violations", row_number)?;
    let cvar95_loss = parse_f64(&row[14], "cvar95_incremental_loss", row_number)?;
    let pooled_harm_upper = parse_f64(&row[15], "pooled_harm_upper_bound", row_number)?;
    let worst_group_harm_upper = parse_f64(&row[16], "worst_group_harm_upper_bound", row_number)?;
    let risk_certificate_state = row[17].clone();
    if !matches!(risk_certificate_state.as_str(), "CERTIFIED" | "NOT_CERTIFIED_RISK_BOUND" | "INSUFFICIENT_EVIDENCE" | "REVOKED_SHIFT" | "UNKNOWN") {
        return Err(invalid(format!("row {row_number}: unknown risk certificate state")));
    }
    let paired_unit_set_sha256 = validate_sha256(&row[18], "paired_unit_set_sha256")?;
    if paired_unit_set_sha256 != c.paired_unit_set_sha256 {
        return Err(invalid(format!("row {row_number}: paired trace/cluster set digest differs from the frozen evaluation contract")));
    }
    let evaluated_traces = parse_u64(&row[19], "evaluated_traces", row_number)?;
    let independent_clusters = parse_u64(&row[20], "independent_clusters", row_number)?;
    if actual_latency_us == 0 || actual_energy_uj == 0 || allocator_latency_us == 0 || allocator_energy_uj == 0 {
        return Err(invalid(format!("row {row_number}: all-in and allocator costs must be positive")));
    }
    if allocator_latency_us > actual_latency_us || allocator_energy_uj > actual_energy_uj {
        return Err(invalid(format!("row {row_number}: allocator costs cannot exceed all-in actual costs")));
    }
    if !(0.0..=1.0).contains(&pooled_harm_upper) || !(0.0..=1.0).contains(&worst_group_harm_upper) {
        return Err(invalid(format!("row {row_number}: harm upper bounds must lie in [0,1]")));
    }
    if evaluated_traces != c.expected_evaluated_traces || independent_clusters != c.expected_independent_clusters {
        return Err(invalid(format!("row {row_number}: trace/cluster counts differ from the frozen evaluation contract")));
    }
    if independent_clusters < 5 {
        return Err(invalid(format!("row {row_number}: fewer than five independent clusters")));
    }
    Ok(RunPoint {
        budget_point_id: row[1].clone(),
        policy_id: row[2].clone(),
        latency_budget_us,
        energy_budget_uj,
        utility_delta,
        actual_latency_us,
        actual_energy_uj,
        allocator_latency_us,
        allocator_energy_uj,
        latency_budget_violations,
        energy_budget_violations,
        cost_measurement_complete,
        safety_violations,
        cvar95_loss,
        pooled_harm_upper,
        worst_group_harm_upper,
        risk_certificate_state,
        evaluated_traces,
        independent_clusters,
    })
}

fn failure_reasons(point: &RunPoint, c: &Constraints) -> Vec<&'static str> {
    let mut reasons = Vec::new();
    if !point.cost_measurement_complete { reasons.push("cost_measurement_incomplete"); }
    if point.latency_budget_violations != 0 || point.actual_latency_us > point.latency_budget_us {
        reasons.push("latency_budget_violation");
    }
    if point.energy_budget_violations != 0 || point.actual_energy_uj > point.energy_budget_uj {
        reasons.push("energy_budget_violation");
    }
    if point.safety_violations > c.safety_violations_max { reasons.push("safety_violation_limit_exceeded"); }
    if point.cvar95_loss > c.cvar95_loss_limit { reasons.push("cvar95_limit_exceeded"); }
    if point.pooled_harm_upper > c.harm_rate_limit { reasons.push("pooled_harm_upper_bound_exceeded"); }
    if point.worst_group_harm_upper > c.harm_rate_limit { reasons.push("worst_group_harm_upper_bound_exceeded"); }
    if point.risk_certificate_state != "CERTIFIED" { reasons.push("risk_certificate_not_certified"); }
    reasons
}

fn strictly_dominates(a: &RunPoint, b: &RunPoint) -> bool {
    let no_worse = a.utility_delta >= b.utility_delta
        && a.actual_latency_us <= b.actual_latency_us
        && a.actual_energy_uj <= b.actual_energy_uj;
    let strict = a.utility_delta > b.utility_delta
        || a.actual_latency_us < b.actual_latency_us
        || a.actual_energy_uj < b.actual_energy_uj;
    no_worse && strict
}

fn objectives_equal(a: &RunPoint, b: &RunPoint) -> bool {
    a.utility_delta == b.utility_delta
        && a.actual_latency_us == b.actual_latency_us
        && a.actual_energy_uj == b.actual_energy_uj
}

fn parse_points(table: &Table, c: &Constraints) -> Result<Vec<RunPoint>, String> {
    let mut points = Vec::new();
    let mut seen = BTreeSet::new();
    let mut point_budget: BTreeMap<String, (u64, u64)> = BTreeMap::new();
    let mut unique_budgets = BTreeSet::new();
    let mut point_policies: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut policies = BTreeSet::new();
    for (index, row) in table.rows.iter().enumerate() {
        let point = parse_run(row, index + 1, c)?;
        if !seen.insert((point.budget_point_id.clone(), point.policy_id.clone())) {
            return Err(invalid(format!("duplicate policy/budget-point row {}/{}", point.policy_id, point.budget_point_id)));
        }
        let ceilings = (point.latency_budget_us, point.energy_budget_uj);
        if let Some(previous) = point_budget.get(&point.budget_point_id) {
            if *previous != ceilings {
                return Err(invalid(format!("budget point {} changes ceilings across policies", point.budget_point_id)));
            }
        } else {
            point_budget.insert(point.budget_point_id.clone(), ceilings);
            if !unique_budgets.insert(ceilings) {
                return Err(invalid("two budget-point IDs use duplicate latency/energy ceilings"));
            }
        }
        policies.insert(point.policy_id.clone());
        point_policies.entry(point.budget_point_id.clone()).or_default().insert(point.policy_id.clone());
        points.push(point);
    }

    if point_budget.len() != c.budget_point_count {
        return Err(invalid(format!("observed {} budget points, expected {}", point_budget.len(), c.budget_point_count)));
    }
    if policies.len() != c.policy_count {
        return Err(invalid(format!("observed {} policies, expected {}", policies.len(), c.policy_count)));
    }
    let expected_policies = policies.clone();
    for (point_id, seen_policies) in &point_policies {
        if *seen_policies != expected_policies {
            return Err(invalid(format!("budget point {point_id} does not contain the exact same policy set")));
        }
    }
    if points.len() != c.budget_point_count * c.policy_count {
        return Err(invalid("incomplete policy-by-budget grid"));
    }
    points.sort_by(|a, b| {
        (&a.budget_point_id, &a.policy_id).cmp(&(&b.budget_point_id, &b.policy_id))
    });
    Ok(points)
}

fn number(value: f64) -> String {
    format!("{value:.10}")
}

fn reasons_string(reasons: &[&str]) -> String {
    if reasons.is_empty() { "-".to_owned() } else { reasons.join(",") }
}

fn analyze(input: &str, expected_trace_lock: &str, expected_sweep_digest: &str) -> Result<String, String> {
    let expected_trace_lock = validate_sha256(expected_trace_lock, "expected trace bundle lock")?;
    let expected_sweep_digest = validate_sha256(expected_sweep_digest, "expected sweep input digest")?;
    let actual_sweep_digest = sha256_hex(input.as_bytes());
    if actual_sweep_digest != expected_sweep_digest {
        return Err(invalid("sweep input digest mismatch against the separately frozen preregistration"));
    }
    let table = parse_table(input)?;
    if table.meta["bundle_lock_sha256"].to_ascii_lowercase() != expected_trace_lock {
        return Err(invalid("trace bundle-lock digest differs from the independently supplied expected digest"));
    }
    let c = parse_constraints(&table.meta)?;
    let points = parse_points(&table, &c)?;

    let mut classified = Vec::with_capacity(points.len());
    for point in &points {
        let reasons = failure_reasons(point, &c);
        classified.push(ClassifiedPoint {
            point: point.clone(),
            status: if reasons.is_empty() { "PENDING_FRONTIER" } else { "INFEASIBLE" },
            reasons,
            dominated_by: None,
        });
    }

    for i in 0..classified.len() {
        if !classified[i].reasons.is_empty() {
            continue;
        }
        let mut dominators: Vec<usize> = Vec::new();
        for j in 0..classified.len() {
            if i == j || !classified[j].reasons.is_empty() {
                continue;
            }
            let other = &classified[j].point;
            let current = &classified[i].point;
            if strictly_dominates(other, current) {
                dominators.push(j);
            } else if objectives_equal(other, current) {
                let other_key = (&other.budget_point_id, &other.policy_id);
                let current_key = (&current.budget_point_id, &current.policy_id);
                if other_key < current_key {
                    dominators.push(j);
                }
            }
        }
        if let Some(index) = dominators.into_iter().min_by(|a, b| {
            let pa = &classified[*a].point;
            let pb = &classified[*b].point;
            (&pa.budget_point_id, &pa.policy_id).cmp(&(&pb.budget_point_id, &pb.policy_id))
        }) {
            classified[i].status = "FEASIBLE_DOMINATED";
            classified[i].dominated_by = Some((
                classified[index].point.budget_point_id.clone(),
                classified[index].point.policy_id.clone(),
            ));
        } else {
            classified[i].status = "FEASIBLE_FRONTIER";
        }
    }

    let eligible = classified.iter().filter(|p| p.status != "INFEASIBLE").count();
    let frontier = classified.iter().filter(|p| p.status == "FEASIBLE_FRONTIER").count();
    let infeasible = classified.len() - eligible;
    let mut output = String::new();
    let _ = writeln!(output, "PARETO\tformat\t{RECEIPT_VERSION}");
    let _ = writeln!(output, "PARETO\tstatus\tSAFETY_GATED_FRONTIER_ANALYSIS_ONLY");
    for key in ["experiment_id", "dataset_id", "dataset_revision", "split", "utility_spec_sha256", "bundle_lock_sha256", "paired_unit_set_sha256"] {
        let _ = writeln!(output, "PARETO\t{key}\t{}", table.meta[key]);
    }
    let _ = writeln!(output, "PARETO\tsweep_input_sha256\t{actual_sweep_digest}");
    let _ = writeln!(output, "PARETO\tpolicy_count\t{}", c.policy_count);
    let _ = writeln!(output, "PARETO\tbudget_point_count\t{}", c.budget_point_count);
    let _ = writeln!(output, "PARETO\tindependent_clusters\t{}", c.expected_independent_clusters);
    let _ = writeln!(output, "PARETO\teligible_runs\t{eligible}");
    let _ = writeln!(output, "PARETO\tinfeasible_runs\t{infeasible}");
    let _ = writeln!(output, "PARETO\tfrontier_runs\t{frontier}");
    let _ = writeln!(output, "PARETO\tfrontier_objectives\tMAXIMIZE_UTILITY_DELTA;MINIMIZE_ACTUAL_LATENCY;MINIMIZE_ACTUAL_ENERGY");
    let _ = writeln!(output, "PARETO\tscalarized_score\tNONE");
    let _ = writeln!(output, "PARETO\tupstream_risk_certificate_verified_here\tfalse");
    let _ = writeln!(output, "PARETO\tperformance_claim\tNONE");
    let _ = writeln!(output, "RESULT_HEADER\tbudget_point_id\tpolicy_id\tstatus\tutility_delta_vs_all_cheap\tactual_all_in_latency_us\tactual_all_in_energy_uj\tlatency_budget_us\tenergy_budget_uj\tallocator_latency_us\tallocator_energy_uj\treasons\tdominated_by");
    for item in &classified {
        let p = &item.point;
        let dominated = item.dominated_by.as_ref().map(|(point, policy)| format!("{point}/{policy}")).unwrap_or_else(|| "-".to_owned());
        let _ = writeln!(
            output,
            "RESULT\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            p.budget_point_id,
            p.policy_id,
            item.status,
            number(p.utility_delta),
            p.actual_latency_us,
            p.actual_energy_uj,
            p.latency_budget_us,
            p.energy_budget_uj,
            p.allocator_latency_us,
            p.allocator_energy_uj,
            reasons_string(&item.reasons),
            dominated
        );
    }
    Ok(output)
}

fn run_cli(args: &[String]) -> Result<String, String> {
    if args.len() != 4 {
        return Err("usage: vision-dvpc-budget-frontier SWEEP.tsv EXPECTED_TRACE_BUNDLE_LOCK_SHA256 EXPECTED_SWEEP_FILE_SHA256".to_owned());
    }
    let input = fs::read_to_string(&args[1]).map_err(|e| invalid(format!("cannot read sweep file: {e}")))?;
    analyze(&input, &args[2], &args[3])
}

fn main() {
    let args: Vec<String> = env::args().collect();
    match run_cli(&args) {
        Ok(output) => print!("{output}"),
        Err(error) => {
            eprintln!("REJECT\treason\t{}", error.replace('\t', " ").replace('\n', " "));
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TRACE_LOCK: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    fn fixture() -> String {
        let mut output = format!(
            "META\tformat\t{FORMAT_VERSION}\n\
META\texperiment_id\tbudget-sweep-test\n\
META\tdataset_id\tsynthetic-fixture\n\
META\tdataset_revision\tr1\n\
META\tsplit\ttest\n\
META\tutility_spec_sha256\tbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\n\
META\tbundle_lock_sha256\t{TRACE_LOCK}\n\
META\tpaired_unit_set_sha256\tcccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc\n\
META\tpolicy_count\t4\n\
META\tbudget_point_count\t3\n\
META\texpected_evaluated_traces\t100\n\
META\texpected_independent_clusters\t10\n\
META\tharm_rate_limit\t0.05\n\
META\tcvar95_loss_limit\t0.20\n\
META\tsafety_violations_max\t0\n\
{header}\n",
            header = SWEEP_HEADER.join("\t")
        );
        let budgets = [("p20", 1000, 1000), ("p50", 2000, 2000), ("p80", 4000, 4000)];
        let policies = ["all-cheap", "candidate", "random", "unsafe"];
        for (point, lat_budget, energy_budget) in budgets {
            for policy in policies {
                let (utility, latency, energy, safety, cvar, pooled, worst, cert) = match (point, policy) {
                    ("p20", "all-cheap") => (0.0, 900, 900, 0, 0.10, 0.02, 0.03, "CERTIFIED"),
                    ("p20", "candidate") => (0.05, 950, 950, 0, 0.12, 0.02, 0.04, "CERTIFIED"),
                    ("p20", "random") => (0.02, 980, 990, 0, 0.13, 0.03, 0.04, "CERTIFIED"),
                    ("p20", "unsafe") => (0.10, 900, 900, 4, 0.35, 0.20, 0.25, "CERTIFIED"),
                    ("p50", "all-cheap") => (0.0, 900, 900, 0, 0.10, 0.02, 0.03, "CERTIFIED"),
                    ("p50", "candidate") => (0.08, 1500, 1700, 0, 0.12, 0.02, 0.04, "CERTIFIED"),
                    ("p50", "random") => (0.03, 1300, 1250, 0, 0.13, 0.03, 0.04, "CERTIFIED"),
                    ("p50", "unsafe") => (0.10, 1700, 1700, 2, 0.33, 0.20, 0.25, "NOT_CERTIFIED_RISK_BOUND"),
                    ("p80", "all-cheap") => (0.0, 900, 900, 0, 0.10, 0.02, 0.03, "CERTIFIED"),
                    ("p80", "candidate") => (0.09, 1800, 1800, 0, 0.12, 0.02, 0.04, "CERTIFIED"),
                    ("p80", "random") => (0.04, 2500, 2200, 0, 0.13, 0.03, 0.04, "CERTIFIED"),
                    ("p80", "unsafe") => (0.11, 2000, 1800, 3, 0.40, 0.21, 0.30, "REVOKED_SHIFT"),
                    _ => unreachable!(),
                };
                let _ = writeln!(
                    output,
                    "SWEEP\t{point}\t{policy}\t{lat_budget}\t{energy_budget}\t{utility:.4}\t{latency}\t{energy}\t100\t50\t0\t0\ttrue\t{safety}\t{cvar:.4}\t{pooled:.4}\t{worst:.4}\t{cert}\tcccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc\t100\t10"
                );
            }
        }
        output
    }

    fn execute(content: &str) -> Result<String, String> {
        analyze(content, TRACE_LOCK, &sha256_hex(content.as_bytes()))
    }

    #[test]
    fn known_sha256_vectors_match() {
        assert_eq!(sha256_hex(b""), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
        assert_eq!(sha256_hex(b"abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    }

    #[test]
    fn classifies_frontier_dominated_and_infeasible_points() {
        let result = execute(&fixture()).unwrap();
        assert!(result.contains("PARETO\tstatus\tSAFETY_GATED_FRONTIER_ANALYSIS_ONLY"));
        assert!(result.contains("RESULT\tp20\tcandidate\tFEASIBLE_FRONTIER"));
        assert!(result.contains("RESULT\tp20\trandom\tFEASIBLE_DOMINATED"));
        assert!(result.contains("RESULT\tp20\tunsafe\tINFEASIBLE"));
        assert!(result.contains("safety_violation_limit_exceeded"));
        assert!(result.contains("PARETO\tscalarized_score\tNONE"));
        assert!(result.contains("PARETO\tperformance_claim\tNONE"));
    }

    #[test]
    fn exact_replay_is_deterministic() {
        assert_eq!(execute(&fixture()).unwrap(), execute(&fixture()).unwrap());
    }

    #[test]
    fn row_order_preserves_classification_but_changes_exact_input_commitment() {
        let content = fixture();
        let mut lines: Vec<&str> = content.lines().collect();
        let header = lines.iter().position(|line| line.starts_with("SWEEP_HEADER")).unwrap();
        let prefix = lines[..=header].join("\n") + "\n";
        let mut rows: Vec<&str> = lines[header + 1..].to_vec();
        rows.reverse();
        let reversed = prefix + &rows.join("\n") + "\n";
        let original_output = execute(&content).unwrap();
        let reversed_output = execute(&reversed).unwrap();
        let original_results = original_output.lines().filter(|line| line.starts_with("RESULT\t")).collect::<Vec<_>>();
        let reversed_results = reversed_output.lines().filter(|line| line.starts_with("RESULT\t")).collect::<Vec<_>>();
        assert_eq!(original_results, reversed_results);
        assert_ne!(sha256_hex(content.as_bytes()), sha256_hex(reversed.as_bytes()));
        assert_ne!(original_output, reversed_output);
    }

    #[test]
    fn duplicate_policy_budget_row_fails() {
        let content = fixture();
        let row = content.lines().find(|line| line.starts_with("SWEEP\tp20\tcandidate")).unwrap();
        let broken = format!("{content}{row}\n");
        assert!(execute(&broken).unwrap_err().contains("duplicate policy/budget-point"));
    }

    #[test]
    fn sweep_file_mutation_fails_against_external_hash() {
        let content = fixture();
        let modified = content.replace("META\tcvar95_loss_limit\t0.20", "META\tcvar95_loss_limit\t0.50");
        assert!(analyze(&modified, TRACE_LOCK, &sha256_hex(content.as_bytes())).unwrap_err().contains("sweep input digest mismatch"));
    }

    #[test]
    fn wrong_trace_bundle_lock_fails() {
        assert!(analyze(&fixture(), "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc", &sha256_hex(fixture().as_bytes()))
            .unwrap_err().contains("independently supplied expected digest"));
    }

    #[test]
    fn changed_budget_ceiling_for_one_policy_fails() {
        let content = fixture().replace("SWEEP\tp20\tcandidate\t1000\t1000", "SWEEP\tp20\tcandidate\t999\t1000");
        assert!(execute(&content).unwrap_err().contains("changes ceilings"));
    }

    #[test]
    fn missing_policy_at_one_budget_fails() {
        let content = fixture();
        let broken = content.lines()
            .filter(|line| !line.starts_with("SWEEP\tp50\trandom\t"))
            .collect::<Vec<_>>().join("\n") + "\n";
        assert!(execute(&broken).unwrap_err().contains("exact same policy set") || execute(&broken).unwrap_err().contains("budget points"));
    }

    #[test]
    fn changed_cluster_counts_fail() {
        let content = fixture().replace("SWEEP\tp20\tcandidate\t1000\t1000\t0.0500\t950\t950\t100\t50\t0\t0\ttrue\t0\t0.1200\t0.0200\t0.0400\tCERTIFIED\tcccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc\t100\t10",
                                         "SWEEP\tp20\tcandidate\t1000\t1000\t0.0500\t950\t950\t100\t50\t0\t0\ttrue\t0\t0.1200\t0.0200\t0.0400\tCERTIFIED\tcccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc\t99\t10");
        assert!(execute(&content).unwrap_err().contains("trace/cluster counts differ"));
    }

    #[test]
    fn mismatched_paired_trace_set_digest_fails() {
        let content = fixture().replace(
            "CERTIFIED\\tcccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc\\t100\\t10",
            "CERTIFIED\\tdddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd\\t100\\t10",
        );
        assert!(execute(&content).unwrap_err().contains("paired trace/cluster set digest differs"));
    }

    #[test]
    fn missing_energy_meter_makes_run_ineligible() {
        let content = fixture().replace("SWEEP\tp20\tcandidate\t1000\t1000\t0.0500\t950\t950\t100\t50\t0\t0\ttrue",
                                         "SWEEP\tp20\tcandidate\t1000\t1000\t0.0500\t950\t950\t100\t50\t0\t0\tfalse");
        let result = execute(&content).unwrap();
        assert!(result.contains("RESULT\tp20\tcandidate\tINFEASIBLE"));
        assert!(result.contains("cost_measurement_incomplete"));
    }

    #[test]
    fn budget_overrun_is_ineligible_even_with_high_utility() {
        let content = fixture().replace("SWEEP\tp20\tcandidate\t1000\t1000\t0.0500\t950\t950",
                                         "SWEEP\tp20\tcandidate\t1000\t1000\t0.5000\t1050\t950");
        let result = execute(&content).unwrap();
        assert!(result.contains("RESULT\tp20\tcandidate\tINFEASIBLE"));
        assert!(result.contains("latency_budget_violation"));
    }

    #[test]
    fn risk_certificate_assertion_does_not_override_harm_or_tail_limits() {
        let content = fixture().replace("SWEEP\tp20\tcandidate\t1000\t1000\t0.0500\t950\t950\t100\t50\t0\t0\ttrue\t0\t0.1200\t0.0200\t0.0400\tCERTIFIED",
                                         "SWEEP\tp20\tcandidate\t1000\t1000\t0.0500\t950\t950\t100\t50\t0\t0\ttrue\t0\t0.1200\t0.1000\t0.1500\tCERTIFIED");
        let result = execute(&content).unwrap();
        assert!(result.contains("RESULT\tp20\tcandidate\tINFEASIBLE"));
        assert!(result.contains("worst_group_harm_upper_bound_exceeded"));
    }

    #[test]
    fn componentwise_frontier_does_not_trade_latency_for_energy() {
        let content = fixture().replace("SWEEP\tp50\tcandidate\t2000\t2000\t0.0800\t1500\t1700",
                                         "SWEEP\tp50\tcandidate\t2000\t2000\t0.0800\t1400\t1800");
        let result = execute(&content).unwrap();
        assert!(result.contains("RESULT\tp50\tcandidate\tFEASIBLE_FRONTIER"));
    }

    #[test]
    fn all_in_actual_cost_must_include_allocator_cost() {
        let content = fixture().replace("SWEEP\tp20\tcandidate\t1000\t1000\t0.0500\t950\t950\t100\t50",
                                         "SWEEP\tp20\tcandidate\t1000\t1000\t0.0500\t90\t40\t100\t50");
        assert!(execute(&content).unwrap_err().contains("allocator costs cannot exceed all-in actual costs"));
    }
}
