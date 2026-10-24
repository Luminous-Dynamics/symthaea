// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! VIS-DVPC-017: decision-time sequential multi-fidelity replay.
//!
//! This standalone evaluator is intentionally not a safety certifier. It consumes
//! upstream risk-state assertions and reports empirical outcomes separately.
//! Routing uses only decision-time gain bounds, certificate states, measured
//! costs for already-run stages, and reserved upper costs for the next stage.
//! Realized utilities and harms are read only after routing, for scoring.
//!
//! Usage:
//!   rustc --edition=2024 --test tools/vision-dvpc-sequential-fidelity.rs -o /tmp/vis-dvpc-017-tests
//!   /tmp/vis-dvpc-017-tests
//!   rustc --edition=2024 -O tools/vision-dvpc-sequential-fidelity.rs -o /tmp/vis-dvpc-017
//!   /tmp/vis-dvpc-017 INPUT.tsv EXPECTED_INPUT_SHA256
//!
//! The input must be an immutable held-out replay bundle. The caller-supplied
//! SHA-256 pins exact input bytes; paired_unit_set_sha256 pins the canonical
//! ordered-independent unit identity set. Neither digest establishes dataset
//! rights, sampling independence, meter validity, or risk-certificate validity.

#![deny(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fmt::Write as FmtWrite;
use std::fs;

const FORMAT_VERSION: &str = "VIS-DVPC-SEQUENTIAL-001-v1";
const RECEIPT_VERSION: &str = "VIS-DVPC-SEQUENTIAL-RECEIPT-001-v1";
const META_KEYS: &[&str] = &[
    "format",
    "experiment_id",
    "dataset_id",
    "dataset_revision",
    "split",
    "utility_spec_sha256",
    "risk_certificate_bundle_sha256",
    "paired_unit_set_sha256",
    "expected_trace_count",
    "expected_cluster_count",
    "latency_budget_us",
    "energy_budget_uj",
    "min_gain_lcb",
    "medium_risk_certificate_state",
    "full_risk_certificate_state",
    "harm_rate_limit",
];
const TRACE_HEADER: &[&str] = &[
    "TRACE",
    "trace_id",
    "cluster_id",
    "group_id",
    "utility_cheap",
    "utility_medium",
    "utility_full",
    "harm_cheap",
    "harm_medium",
    "harm_full",
    "medium_gain_lcb",
    "full_gain_lcb",
    "medium_cost_latency_upper_us",
    "medium_cost_energy_upper_uj",
    "full_cost_latency_upper_us",
    "full_cost_energy_upper_uj",
    "allocator_latency_us",
    "allocator_energy_uj",
    "cheap_latency_us",
    "cheap_energy_uj",
    "medium_latency_us",
    "medium_energy_uj",
    "full_latency_us",
    "full_energy_uj",
    "allocator_cost_complete",
    "cheap_cost_complete",
    "medium_cost_complete",
    "full_cost_complete",
];

#[derive(Debug, Clone)]
struct Meta {
    expected_trace_count: usize,
    expected_cluster_count: usize,
    latency_budget_us: u64,
    energy_budget_uj: u64,
    min_gain_lcb: f64,
    medium_risk_state: String,
    full_risk_state: String,
    harm_rate_limit: f64,
}

#[derive(Debug, Clone)]
struct Trace {
    trace_id: String,
    cluster_id: String,
    group_id: String,
    utility: [f64; 3],
    harm: [bool; 3],
    medium_gain_lcb: f64,
    full_gain_lcb: f64,
    medium_latency_upper: u64,
    medium_energy_upper: u64,
    full_latency_upper: u64,
    full_energy_upper: u64,
    allocator_latency: u64,
    allocator_energy: u64,
    cheap_latency: u64,
    cheap_energy: u64,
    medium_latency: u64,
    medium_energy: u64,
    full_latency: u64,
    full_energy: u64,
    allocator_cost_complete: bool,
    cheap_cost_complete: bool,
    medium_cost_complete: bool,
    full_cost_complete: bool,
}

#[derive(Debug)]
struct Table {
    meta: BTreeMap<String, String>,
    traces: Vec<Trace>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tier {
    Cheap,
    Medium,
    Full,
}

impl Tier {
    fn index(self) -> usize {
        match self {
            Tier::Cheap => 0,
            Tier::Medium => 1,
            Tier::Full => 2,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Tier::Cheap => "cheap",
            Tier::Medium => "medium",
            Tier::Full => "full",
        }
    }
}

#[derive(Debug, Clone)]
struct Decision {
    tier: Tier,
    reason: &'static str,
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
        return Err(invalid(format!("{field} must be a non-empty identifier token")));
    }
    Ok(())
}

fn parse_f64(value: &str, field: &str, row: usize) -> Result<f64, String> {
    let parsed = value.parse::<f64>()
        .map_err(|_| invalid(format!("row {row}: {field} must be numeric")))?;
    if !parsed.is_finite() {
        return Err(invalid(format!("row {row}: {field} must be finite")));
    }
    Ok(parsed)
}

fn parse_u64(value: &str, field: &str, row: usize) -> Result<u64, String> {
    value.parse::<u64>()
        .map_err(|_| invalid(format!("row {row}: {field} must be a non-negative integer")))
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
    let mut traces = Vec::new();
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
                return Err(invalid(format!("line {line_number}: META requires key and value")));
            }
            if meta.insert(fields[1].to_owned(), fields[2].to_owned()).is_some() {
                return Err(invalid(format!("line {line_number}: duplicate metadata key")));
            }
            continue;
        }
        if !header_seen {
            if fields.as_slice() != TRACE_HEADER {
                return Err(invalid("exact TRACE header required; reordered or unknown columns fail closed"));
            }
            header_seen = true;
            continue;
        }
        if fields.len() != TRACE_HEADER.len() || fields.first().copied() != Some("TRACE") {
            return Err(invalid(format!("line {line_number}: TRACE row has wrong field count")));
        }
        if fields.iter().any(|field| field.is_empty()) {
            return Err(invalid(format!("line {line_number}: empty fields are forbidden")));
        }
        let row = line_number;
        for (i, field) in [(1, "trace_id"), (2, "cluster_id"), (3, "group_id")] {
            validate_token(fields[i], field)?;
        }
        let trace = Trace {
            trace_id: fields[1].to_owned(),
            cluster_id: fields[2].to_owned(),
            group_id: fields[3].to_owned(),
            utility: [
                parse_f64(fields[4], "utility_cheap", row)?,
                parse_f64(fields[5], "utility_medium", row)?,
                parse_f64(fields[6], "utility_full", row)?,
            ],
            harm: [
                parse_bool(fields[7], "harm_cheap", row)?,
                parse_bool(fields[8], "harm_medium", row)?,
                parse_bool(fields[9], "harm_full", row)?,
            ],
            medium_gain_lcb: parse_f64(fields[10], "medium_gain_lcb", row)?,
            full_gain_lcb: parse_f64(fields[11], "full_gain_lcb", row)?,
            medium_latency_upper: parse_u64(fields[12], "medium_cost_latency_upper_us", row)?,
            medium_energy_upper: parse_u64(fields[13], "medium_cost_energy_upper_uj", row)?,
            full_latency_upper: parse_u64(fields[14], "full_cost_latency_upper_us", row)?,
            full_energy_upper: parse_u64(fields[15], "full_cost_energy_upper_uj", row)?,
            allocator_latency: parse_u64(fields[16], "allocator_latency_us", row)?,
            allocator_energy: parse_u64(fields[17], "allocator_energy_uj", row)?,
            cheap_latency: parse_u64(fields[18], "cheap_latency_us", row)?,
            cheap_energy: parse_u64(fields[19], "cheap_energy_uj", row)?,
            medium_latency: parse_u64(fields[20], "medium_latency_us", row)?,
            medium_energy: parse_u64(fields[21], "medium_energy_uj", row)?,
            full_latency: parse_u64(fields[22], "full_latency_us", row)?,
            full_energy: parse_u64(fields[23], "full_energy_uj", row)?,
            allocator_cost_complete: parse_bool(fields[24], "allocator_cost_complete", row)?,
            cheap_cost_complete: parse_bool(fields[25], "cheap_cost_complete", row)?,
            medium_cost_complete: parse_bool(fields[26], "medium_cost_complete", row)?,
            full_cost_complete: parse_bool(fields[27], "full_cost_complete", row)?,
        };
        traces.push(trace);
    }
    if !header_seen || traces.is_empty() {
        return Err(invalid("missing TRACE header or trace rows"));
    }
    let expected_keys: BTreeSet<&str> = META_KEYS.iter().copied().collect();
    let actual_keys: BTreeSet<&str> = meta.keys().map(String::as_str).collect();
    if expected_keys != actual_keys {
        return Err(invalid("META key set mismatch: missing or unknown fields"));
    }
    if meta.get("format").map(String::as_str) != Some(FORMAT_VERSION) {
        return Err(invalid("unsupported sequential-fidelity format"));
    }
    if !matches!(meta.get("split").map(String::as_str), Some("test" | "private_test")) {
        return Err(invalid("only test or private_test splits are admitted"));
    }
    for key in ["experiment_id", "dataset_id", "dataset_revision"] {
        validate_token(&meta[key], key)?;
    }
    validate_sha256(&meta["utility_spec_sha256"], "utility_spec_sha256")?;
    validate_sha256(&meta["risk_certificate_bundle_sha256"], "risk_certificate_bundle_sha256")?;
    validate_sha256(&meta["paired_unit_set_sha256"], "paired_unit_set_sha256")?;
    Ok(Table { meta, traces })
}

fn parse_meta(table: &Table) -> Result<Meta, String> {
    let m = &table.meta;
    let expected_trace_count = m["expected_trace_count"].parse::<usize>()
        .map_err(|_| invalid("expected_trace_count must be a positive integer"))?;
    let expected_cluster_count = m["expected_cluster_count"].parse::<usize>()
        .map_err(|_| invalid("expected_cluster_count must be a positive integer"))?;
    if expected_trace_count == 0 || expected_cluster_count < 5 {
        return Err(invalid("at least one trace and five independent clusters are required"));
    }
    let latency_budget_us = m["latency_budget_us"].parse::<u64>()
        .map_err(|_| invalid("latency_budget_us must be a positive integer"))?;
    let energy_budget_uj = m["energy_budget_uj"].parse::<u64>()
        .map_err(|_| invalid("energy_budget_uj must be a positive integer"))?;
    if latency_budget_us == 0 || energy_budget_uj == 0 {
        return Err(invalid("latency and energy budgets must be positive"));
    }
    let min_gain_lcb = parse_f64(&m["min_gain_lcb"], "min_gain_lcb", 0)?;
    if min_gain_lcb < 0.0 {
        return Err(invalid("min_gain_lcb must be non-negative"));
    }
    let harm_rate_limit = parse_f64(&m["harm_rate_limit"], "harm_rate_limit", 0)?;
    if !(0.0..=1.0).contains(&harm_rate_limit) {
        return Err(invalid("harm_rate_limit must be in [0,1]"));
    }
    let medium_risk_state = m["medium_risk_certificate_state"].clone();
    let full_risk_state = m["full_risk_certificate_state"].clone();
    for state in [&medium_risk_state, &full_risk_state] {
        if !matches!(state.as_str(), "CERTIFIED" | "NOT_CERTIFIED_RISK_BOUND" | "INSUFFICIENT_EVIDENCE" | "REVOKED_SHIFT" | "UNKNOWN") {
            return Err(invalid("unknown upstream risk certificate state"));
        }
    }
    Ok(Meta {
        expected_trace_count,
        expected_cluster_count,
        latency_budget_us,
        energy_budget_uj,
        min_gain_lcb,
        medium_risk_state,
        full_risk_state,
        harm_rate_limit,
    })
}

fn checked_add(a: u64, b: u64) -> Option<u64> {
    a.checked_add(b)
}

fn fits(base: u64, reserved_increment: u64, ceiling: u64) -> bool {
    checked_add(base, reserved_increment).is_some_and(|total| total <= ceiling)
}

/// Routing function deliberately does not read realized utility or harm fields.
fn route(t: &Trace, m: &Meta) -> Decision {
    if !t.allocator_cost_complete || !t.cheap_cost_complete {
        return Decision { tier: Tier::Cheap, reason: "base_cost_incomplete" };
    }
    let Some(base_latency) = checked_add(t.allocator_latency, t.cheap_latency) else {
        return Decision { tier: Tier::Cheap, reason: "base_cost_overflow" };
    };
    let Some(base_energy) = checked_add(t.allocator_energy, t.cheap_energy) else {
        return Decision { tier: Tier::Cheap, reason: "base_cost_overflow" };
    };
    if base_latency > m.latency_budget_us || base_energy > m.energy_budget_uj {
        return Decision { tier: Tier::Cheap, reason: "base_budget_exhausted" };
    }
    if m.medium_risk_state != "CERTIFIED" {
        return Decision { tier: Tier::Cheap, reason: "medium_risk_not_certified" };
    }
    if t.medium_gain_lcb < m.min_gain_lcb {
        return Decision { tier: Tier::Cheap, reason: "medium_gain_below_floor" };
    }
    if !fits(base_latency, t.medium_latency_upper, m.latency_budget_us)
        || !fits(base_energy, t.medium_energy_upper, m.energy_budget_uj)
    {
        return Decision { tier: Tier::Cheap, reason: "medium_reservation_exceeds_budget" };
    }

    let Some(after_medium_latency) = checked_add(base_latency, t.medium_latency) else {
        return Decision { tier: Tier::Medium, reason: "medium_cost_overflow_stop" };
    };
    let Some(after_medium_energy) = checked_add(base_energy, t.medium_energy) else {
        return Decision { tier: Tier::Medium, reason: "medium_cost_overflow_stop" };
    };
    if !t.medium_cost_complete {
        return Decision { tier: Tier::Medium, reason: "medium_cost_incomplete_stop" };
    }
    if after_medium_latency > m.latency_budget_us || after_medium_energy > m.energy_budget_uj {
        return Decision { tier: Tier::Medium, reason: "medium_actual_overrun_stop" };
    }
    if t.medium_latency > t.medium_latency_upper || t.medium_energy > t.medium_energy_upper {
        return Decision { tier: Tier::Medium, reason: "medium_cost_reservation_underestimated_stop" };
    }
    if m.full_risk_state != "CERTIFIED" {
        return Decision { tier: Tier::Medium, reason: "full_risk_not_certified" };
    }
    if t.full_gain_lcb < m.min_gain_lcb {
        return Decision { tier: Tier::Medium, reason: "full_gain_below_floor" };
    }
    if !fits(after_medium_latency, t.full_latency_upper, m.latency_budget_us)
        || !fits(after_medium_energy, t.full_energy_upper, m.energy_budget_uj)
    {
        return Decision { tier: Tier::Medium, reason: "full_reservation_exceeds_budget" };
    }
    Decision { tier: Tier::Full, reason: "full_gain_and_budget_admitted" }
}

#[derive(Default, Debug, Clone)]
struct Aggregate {
    n: usize,
    utility_sum: f64,
    harms: usize,
    latency_sum: u128,
    energy_sum: u128,
    budget_violations: usize,
    cost_incomplete: usize,
}

impl Aggregate {
    fn add(&mut self, utility: f64, harm: bool, latency: u64, energy: u64, complete: bool, m: &Meta) {
        self.n += 1;
        self.utility_sum += utility;
        self.harms += usize::from(harm);
        self.latency_sum += latency as u128;
        self.energy_sum += energy as u128;
        if latency > m.latency_budget_us || energy > m.energy_budget_uj {
            self.budget_violations += 1;
        }
        if !complete {
            self.cost_incomplete += 1;
        }
    }

    fn mean_utility(&self) -> f64 {
        self.utility_sum / self.n.max(1) as f64
    }

    fn harm_rate(&self) -> f64 {
        self.harms as f64 / self.n.max(1) as f64
    }

    fn mean_latency(&self) -> f64 {
        self.latency_sum as f64 / self.n.max(1) as f64
    }

    fn mean_energy(&self) -> f64 {
        self.energy_sum as f64 / self.n.max(1) as f64
    }
}

#[derive(Default)]
struct PolicyAggregates {
    total: Aggregate,
    by_group: BTreeMap<String, Aggregate>,
}

fn actual_cost(t: &Trace, mode: &str) -> (u64, u64, bool) {
    match mode {
        "cheap" => (t.cheap_latency, t.cheap_energy, t.cheap_cost_complete),
        "medium" => (t.medium_latency, t.medium_energy, t.medium_cost_complete),
        "full" => (t.full_latency, t.full_energy, t.full_cost_complete),
        _ => unreachable!("known fixed-policy name"),
    }
}

fn utility_for(t: &Trace, tier: Tier) -> (f64, bool) {
    let i = tier.index();
    (t.utility[i], t.harm[i])
}

fn run_aggregates(traces: &[Trace], m: &Meta) -> (BTreeMap<String, PolicyAggregates>, Vec<(String, Decision)>) {
    let mut policies = BTreeMap::<String, PolicyAggregates>::new();
    for name in ["cheap", "medium", "full", "sequential"] {
        policies.insert(name.to_owned(), PolicyAggregates::default());
    }
    let mut decisions = Vec::with_capacity(traces.len());
    for t in traces {
        let sequential_decision = route(t, m);
        decisions.push((t.trace_id.clone(), sequential_decision.clone()));
        let selections = [
            ("cheap", Tier::Cheap),
            ("medium", Tier::Medium),
            ("full", Tier::Full),
            ("sequential", sequential_decision.tier),
        ];
        for (name, tier) in selections {
            let (utility, harm) = utility_for(t, tier);
            let (latency, energy, complete) = match name {
                "cheap" => actual_cost(t, "cheap"),
                "medium" => actual_cost(t, "medium"),
                "full" => actual_cost(t, "full"),
                "sequential" => {
                    let latency = t.allocator_latency
                        .checked_add(t.cheap_latency)
                        .and_then(|v| if tier != Tier::Cheap { v.checked_add(t.medium_latency) } else { Some(v) })
                        .and_then(|v| if tier == Tier::Full { v.checked_add(t.full_latency) } else { Some(v) })
                        .unwrap_or(u64::MAX);
                    let energy = t.allocator_energy
                        .checked_add(t.cheap_energy)
                        .and_then(|v| if tier != Tier::Cheap { v.checked_add(t.medium_energy) } else { Some(v) })
                        .and_then(|v| if tier == Tier::Full { v.checked_add(t.full_energy) } else { Some(v) })
                        .unwrap_or(u64::MAX);
                    let complete = t.allocator_cost_complete && t.cheap_cost_complete
                        && (tier == Tier::Cheap || t.medium_cost_complete)
                        && (tier != Tier::Full || t.full_cost_complete);
                    (latency, energy, complete)
                }
                _ => unreachable!(),
            };
            let policy = policies.get_mut(name).expect("policy exists");
            policy.total.add(utility, harm, latency, energy, complete, m);
            policy.by_group.entry(t.group_id.clone()).or_default()
                .add(utility, harm, latency, energy, complete, m);
        }
    }
    (policies, decisions)
}

fn canonical_unit_set_sha256(traces: &[Trace]) -> String {
    let mut units: Vec<(&str, &str, &str)> = traces.iter()
        .map(|t| (t.trace_id.as_str(), t.cluster_id.as_str(), t.group_id.as_str()))
        .collect();
    units.sort_unstable();
    let mut canonical = String::new();
    for (trace, cluster, group) in units {
        let _ = writeln!(canonical, "{trace}\t{cluster}\t{group}");
    }
    sha256_hex(canonical.as_bytes())
}

fn validate_bundle(table: &Table, m: &Meta, input_sha: &str) -> Result<(), String> {
    if input_sha.is_empty() {
        return Err(invalid("empty input digest"));
    }
    if table.traces.len() != m.expected_trace_count {
        return Err(invalid("trace count differs from frozen contract"));
    }
    let mut trace_ids = BTreeSet::new();
    let mut clusters = BTreeSet::new();
    for trace in &table.traces {
        if !trace_ids.insert(trace.trace_id.as_str()) {
            return Err(invalid("duplicate trace_id"));
        }
        clusters.insert(trace.cluster_id.as_str());
    }
    if clusters.len() != m.expected_cluster_count || clusters.len() < 5 {
        return Err(invalid("independent cluster count differs from frozen contract or is below five"));
    }
    let unit_digest = canonical_unit_set_sha256(&table.traces);
    if unit_digest != table.meta["paired_unit_set_sha256"].to_ascii_lowercase() {
        return Err(invalid("paired_unit_set_sha256 does not match canonical trace/cluster/group identities"));
    }
    Ok(())
}

fn receipt(table: &Table, meta: &Meta, input_sha: &str) -> String {
    let (policies, decisions) = run_aggregates(&table.traces, meta);
    let sequential = &policies["sequential"].total;
    let cheap = &policies["cheap"].total;
    let mut out = String::new();
    let _ = writeln!(out, "FORMAT\t{RECEIPT_VERSION}");
    let _ = writeln!(out, "INPUT_SHA256\t{input_sha}");
    let _ = writeln!(out, "UNIT_SET_SHA256\t{}", table.meta["paired_unit_set_sha256"]);
    let _ = writeln!(out, "RISK_CERTIFICATE_BUNDLE_SHA256\t{}", table.meta["risk_certificate_bundle_sha256"]);
    let _ = writeln!(out, "EXPERIMENT_ID\t{}", table.meta["experiment_id"]);
    let _ = writeln!(out, "DATASET\t{}\t{}\t{}", table.meta["dataset_id"], table.meta["dataset_revision"], table.meta["split"]);
    let _ = writeln!(out, "SCOPE\tEVALUATION_ONLY_NOT_A_CERTIFICATE");
    let _ = writeln!(out, "RISK_STATE_MEDIUM\t{}", meta.medium_risk_state);
    let _ = writeln!(out, "RISK_STATE_FULL\t{}", meta.full_risk_state);
    let _ = writeln!(out, "EMPIRICAL_HARM_RATE_LIMIT\t{:.6}\tDESCRIPTIVE_ONLY_NOT_CERTIFICATION", meta.harm_rate_limit);
    let _ = writeln!(out, "POLICY_HEADER\tpolicy\tn\tmean_utility\tharm_count\tempirical_harm_rate\tmean_all_in_latency_us\tmean_all_in_energy_uj\tbudget_violations\tcost_incomplete");
    for name in ["cheap", "medium", "full", "sequential"] {
        let s = &policies[name].total;
        let _ = writeln!(
            out,
            "POLICY\t{name}\t{}\t{:.9}\t{}\t{:.9}\t{:.3}\t{:.3}\t{}\t{}",
            s.n, s.mean_utility(), s.harms, s.harm_rate(), s.mean_latency(), s.mean_energy(),
            s.budget_violations, s.cost_incomplete
        );
    }
    let _ = writeln!(out, "SEQUENTIAL_DELTA_UTILITY_VS_CHEAP\t{:.9}", sequential.mean_utility() - cheap.mean_utility());
    let _ = writeln!(out, "SEQUENTIAL_ESCALATION_COUNTS\tmedium_or_full={}\tfull={}",
        decisions.iter().filter(|(_, d)| d.tier != Tier::Cheap).count(),
        decisions.iter().filter(|(_, d)| d.tier == Tier::Full).count());
    let _ = writeln!(out, "DECISION_HEADER\ttrace_id\tselected_tier\treason");
    for (trace_id, decision) in decisions {
        let _ = writeln!(out, "DECISION\t{trace_id}\t{}\t{}", decision.tier.label(), decision.reason);
    }
    let _ = writeln!(out, "GROUP_HEADER\tpolicy\tgroup_id\tn\tharm_count\tempirical_harm_rate\tmean_utility");
    for name in ["cheap", "medium", "full", "sequential"] {
        for (group, agg) in &policies[name].by_group {
            let _ = writeln!(out, "GROUP\t{name}\t{group}\t{}\t{}\t{:.9}\t{:.9}",
                agg.n, agg.harms, agg.harm_rate(), agg.mean_utility());
        }
    }
    let _ = writeln!(out, "CLAIMS\tstatistical_independence=NOT_ASSESSED\tcausal_claim=NONE\tSOTA_CLAIM=NONE");
    out
}

fn main_result() -> Result<(), String> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        return Err(invalid("usage: vision-dvpc-sequential-fidelity INPUT.tsv EXPECTED_INPUT_SHA256"));
    }
    let expected_input_sha = validate_sha256(&args[2], "EXPECTED_INPUT_SHA256")?;
    let bytes = fs::read(&args[1]).map_err(|e| invalid(format!("cannot read input: {e}")))?;
    let actual_input_sha = sha256_hex(&bytes);
    if actual_input_sha != expected_input_sha {
        return Err(invalid(format!("input SHA-256 mismatch: expected {expected_input_sha}, got {actual_input_sha}")));
    }
    let input = std::str::from_utf8(&bytes).map_err(|_| invalid("input must be UTF-8"))?;
    let table = parse_table(input)?;
    let meta = parse_meta(&table)?;
    validate_bundle(&table, &meta, &actual_input_sha)?;
    print!("{}", receipt(&table, &meta, &actual_input_sha));
    Ok(())
}

fn main() {
    if let Err(error) = main_result() {
        eprintln!("VIS-DVPC-017 INVALID: {error}");
        std::process::exit(2);
    }
}

#[cfg(test)]
fn test_meta() -> Meta {
    Meta {
        values: BTreeMap::new(),
        expected_trace_count: 1,
        expected_cluster_count: 5,
        latency_budget_us: 100,
        energy_budget_uj: 100,
        min_gain_lcb: 0.05,
        medium_risk_state: "CERTIFIED".to_owned(),
        full_risk_state: "CERTIFIED".to_owned(),
        harm_rate_limit: 0.01,
    }
}

#[cfg(test)]
fn test_trace() -> Trace {
    Trace {
        trace_id: "t1".to_owned(),
        cluster_id: "c1".to_owned(),
        group_id: "g1".to_owned(),
        utility: [0.5, 0.7, 0.9],
        harm: [false, false, false],
        medium_gain_lcb: 0.1,
        full_gain_lcb: 0.1,
        medium_latency_upper: 10,
        medium_energy_upper: 10,
        full_latency_upper: 10,
        full_energy_upper: 10,
        allocator_latency: 2,
        allocator_energy: 2,
        cheap_latency: 10,
        cheap_energy: 10,
        medium_latency: 20,
        medium_energy: 20,
        full_latency: 20,
        full_energy: 20,
        allocator_cost_complete: true,
        cheap_cost_complete: true,
        medium_cost_complete: true,
        full_cost_complete: true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_known_vectors() {
        assert_eq!(sha256_hex(b""), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
        assert_eq!(sha256_hex(b"abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    }

    #[test]
    fn sequential_policy_escalates_through_medium_to_full() {
        let t = test_trace();
        assert_eq!(route(&t, &test_meta()).tier, Tier::Full);
    }

    #[test]
    fn no_escalation_without_medium_certificate() {
        let t = test_trace();
        let mut m = test_meta();
        m.medium_risk_state = "UNKNOWN".to_owned();
        assert_eq!(route(&t, &m).tier, Tier::Cheap);
    }

    #[test]
    fn full_is_blocked_without_full_certificate() {
        let t = test_trace();
        let mut m = test_meta();
        m.full_risk_state = "NOT_CERTIFIED_RISK_BOUND".to_owned();
        assert_eq!(route(&t, &m).tier, Tier::Medium);
    }

    #[test]
    fn upper_cost_reservation_blocks_stage_before_execution() {
        let mut t = test_trace();
        t.medium_latency_upper = 200;
        assert_eq!(route(&t, &test_meta()).tier, Tier::Cheap);
        let mut t = test_trace();
        t.full_energy_upper = 200;
        assert_eq!(route(&t, &test_meta()).tier, Tier::Medium);
    }

    #[test]
    fn under_threshold_gain_stops_at_current_tier() {
        let mut t = test_trace();
        t.medium_gain_lcb = 0.049;
        assert_eq!(route(&t, &test_meta()).tier, Tier::Cheap);
        let mut t = test_trace();
        t.full_gain_lcb = 0.0;
        assert_eq!(route(&t, &test_meta()).tier, Tier::Medium);
    }

    #[test]
    fn unknown_base_cost_fails_closed_to_cheap() {
        let mut t = test_trace();
        t.allocator_cost_complete = false;
        assert_eq!(route(&t, &test_meta()).tier, Tier::Cheap);
        let mut t = test_trace();
        t.cheap_cost_complete = false;
        assert_eq!(route(&t, &test_meta()).tier, Tier::Cheap);
    }

    #[test]
    fn actual_medium_overrun_blocks_full_escalation() {
        let mut t = test_trace();
        t.medium_latency = 200;
        assert_eq!(route(&t, &test_meta()).tier, Tier::Medium);
    }

    #[test]
    fn routing_does_not_depend_on_realized_utility_or_harm() {
        let mut a = test_trace();
        let mut b = test_trace();
        b.utility = [1_000_000.0, -1_000_000.0, 7.0];
        b.harm = [true, true, false];
        assert_eq!(route(&a, &test_meta()).tier, route(&b, &test_meta()).tier);
        a.medium_gain_lcb = b.medium_gain_lcb;
        assert_eq!(route(&a, &test_meta()).reason, route(&b, &test_meta()).reason);
    }

    #[test]
    fn resource_overflow_fails_closed() {
        let mut t = test_trace();
        t.allocator_latency = u64::MAX;
        assert_eq!(route(&t, &test_meta()).tier, Tier::Cheap);
        let mut t = test_trace();
        t.medium_latency_upper = u64::MAX;
        assert_eq!(route(&t, &test_meta()).tier, Tier::Cheap);
    }

    #[test]
    fn unit_set_digest_is_row_order_independent() {
        let mut a = test_trace();
        let mut b = test_trace();
        b.trace_id = "t2".to_owned();
        b.cluster_id = "c2".to_owned();
        let d1 = canonical_unit_set_sha256(&[a.clone(), b.clone()]);
        let d2 = canonical_unit_set_sha256(&[b, a]);
        assert_eq!(d1, d2);
        a.group_id = "g2".to_owned();
        assert_ne!(d1, canonical_unit_set_sha256(&[a, test_trace()]));
    }

    #[test]
    fn smoke_fixture_parses_contract_and_routes_all_three_tiers() {
        let input = include_str!("../tests/fixtures/vision-dvpc-017-smoke.tsv");
        let table = parse_table(input).expect("smoke fixture parses");
        let meta = parse_meta(&table).expect("smoke metadata parses");
        let digest = sha256_hex(input.as_bytes());
        validate_bundle(&table, &meta, &digest).expect("frozen fixture contract validates");
        let tiers: Vec<Tier> = table.traces.iter().map(|trace| route(trace, &meta).tier).collect();
        assert_eq!(tiers, vec![Tier::Full, Tier::Medium, Tier::Cheap, Tier::Full, Tier::Cheap]);
        let rendered = receipt(&table, &meta, &digest);
        assert!(rendered.contains("SCOPE\\tEVALUATION_ONLY_NOT_A_CERTIFICATE"));
        assert!(rendered.contains("DECISION\\tt02\\tmedium\\tfull_gain_below_floor"));
        assert!(rendered.contains("DECISION\\tt05\\tcheap\\tmedium_reservation_exceeds_budget"));
    }

    #[test]
    fn all_policy_aggregates_preserve_paired_trace_count() {
        let mut traces = Vec::new();
        for i in 0..5 {
            let mut t = test_trace();
            t.trace_id = format!("t{i}");
            t.cluster_id = format!("c{i}");
            traces.push(t);
        }
        let (policies, decisions) = run_aggregates(&traces, &test_meta());
        assert_eq!(decisions.len(), 5);
        for name in ["cheap", "medium", "full", "sequential"] {
            assert_eq!(policies[name].total.n, 5);
        }
    }
}
