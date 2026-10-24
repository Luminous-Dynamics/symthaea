// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! VIS-DVPC-018: uncertainty-aware reclassification of a safety-gated Pareto receipt.
//!
//! This tool consumes the receipt from VIS-DVPC-016 plus frozen, family-wise
//! simultaneous utility bands. It does not recompute feasibility, validate the
//! interval procedure, or certify safety. An interval is treated as an upstream
//! statistical assertion bound to the supplied inference-receipt digest.
//!
//! Usage:
//!   rustc --edition=2024 --test tools/vision-dvpc-confidence-frontier.rs -o /tmp/vis-dvpc-018-tests
//!   /tmp/vis-dvpc-018-tests
//!   rustc --edition=2024 -O tools/vision-dvpc-confidence-frontier.rs -o /tmp/vis-dvpc-018
//!   /tmp/vis-dvpc-018 PARETO.tsv EXPECTED_PARETO_SHA256 BANDS.tsv EXPECTED_BANDS_SHA256 EXPECTED_INFERENCE_RECEIPT_SHA256

#![deny(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fmt::Write as FmtWrite;
use std::fs;

const RECEIPT_VERSION: &str = "VIS-DVPC-PARETO-RECEIPT-001-v1";
const OUTPUT_VERSION: &str = "VIS-DVPC-CONFIDENCE-FRONTIER-RECEIPT-001-v1";
const PARETO_STATUS: &str = "SAFETY_GATED_FRONTIER_ANALYSIS_ONLY";
const PARETO_META_KEYS: &[&str] = &[
    "format", "status", "experiment_id", "dataset_id", "dataset_revision", "split",
    "utility_spec_sha256", "bundle_lock_sha256", "paired_unit_set_sha256",
    "sweep_input_sha256", "policy_count", "budget_point_count", "independent_clusters",
    "eligible_runs", "infeasible_runs", "frontier_runs", "frontier_objectives",
    "scalarized_score", "upstream_risk_certificate_verified_here", "performance_claim",
];
const PARETO_HEADER: &[&str] = &[
    "RESULT_HEADER", "budget_point_id", "policy_id", "status", "utility_delta_vs_all_cheap",
    "actual_all_in_latency_us", "actual_all_in_energy_uj", "latency_budget_us",
    "energy_budget_uj", "allocator_latency_us", "allocator_energy_uj", "reasons", "dominated_by",
];
const BAND_META_KEYS: &[&str] = &[
    "format", "experiment_id", "dataset_id", "dataset_revision", "split",
    "utility_spec_sha256", "bundle_lock_sha256", "paired_unit_set_sha256",
    "source_sweep_sha256", "source_pareto_receipt_sha256", "inference_receipt_sha256",
    "confidence_level", "familywise_simultaneous", "coverage_scope", "point_count",
];
const BAND_HEADER: &[&str] = &[
    "BAND_HEADER", "budget_point_id", "policy_id", "utility_lcb", "utility_ucb",
];

#[derive(Debug, Clone)]
struct Point {
    budget_point_id: String,
    policy_id: String,
    upstream_status: String,
    utility_estimate: f64,
    latency: u64,
    energy: u64,
    latency_budget: u64,
    energy_budget: u64,
    reasons: String,
    dominated_by: String,
}

impl Point {
    fn key(&self) -> (String, String) {
        (self.budget_point_id.clone(), self.policy_id.clone())
    }
    fn label(&self) -> String {
        format!("{}/{}", self.budget_point_id, self.policy_id)
    }
    fn eligible(&self) -> bool {
        self.upstream_status == "FEASIBLE_FRONTIER" || self.upstream_status == "FEASIBLE_DOMINATED"
    }
}

#[derive(Debug, Clone, Copy)]
struct Band {
    lower: f64,
    upper: f64,
}

#[derive(Debug)]
struct Receipt {
    meta: BTreeMap<String, String>,
    points: Vec<Point>,
}

#[derive(Debug)]
struct Bands {
    meta: BTreeMap<String, String>,
    values: BTreeMap<(String, String), Band>,
}

#[derive(Debug, Clone)]
struct Classified {
    point: Point,
    band: Band,
    status: &'static str,
    reason: &'static str,
    dominated_by: String,
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

fn validate_sha(value: &str, field: &str) -> Result<String, String> {
    if value.len() != 64 || !value.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(invalid(format!("{field} must be a 64-character hexadecimal SHA-256 digest")));
    }
    Ok(value.to_ascii_lowercase())
}

fn parse_f64(value: &str, field: &str, line: usize) -> Result<f64, String> {
    let n = value.parse::<f64>().map_err(|_| invalid(format!("line {line}: {field} must be numeric")))?;
    if !n.is_finite() {
        return Err(invalid(format!("line {line}: {field} must be finite")));
    }
    Ok(n)
}

fn parse_u64(value: &str, field: &str, line: usize) -> Result<u64, String> {
    value.parse::<u64>().map_err(|_| invalid(format!("line {line}: {field} must be a non-negative integer")))
}

fn parse_token(value: &str, field: &str) -> Result<(), String> {
    if value.is_empty() || !value.bytes().all(|b| {
        b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b':')
    }) {
        return Err(invalid(format!("{field} must be a non-empty token")));
    }
    Ok(())
}

fn require_exact_keys(meta: &BTreeMap<String, String>, expected: &[&str], kind: &str) -> Result<(), String> {
    let actual: BTreeSet<&str> = meta.keys().map(String::as_str).collect();
    let wanted: BTreeSet<&str> = expected.iter().copied().collect();
    if actual != wanted {
        let missing = wanted.difference(&actual).copied().collect::<Vec<_>>();
        let extra = actual.difference(&wanted).copied().collect::<Vec<_>>();
        return Err(invalid(format!("{kind} metadata key mismatch; missing={missing:?}; extra={extra:?}")));
    }
    Ok(())
}

fn parse_receipt(input: &str) -> Result<Receipt, String> {
    let mut meta = BTreeMap::new();
    let mut points = Vec::new();
    let mut header_seen = false;
    let mut rows_started = false;
    for (idx, line) in input.lines().enumerate() {
        let line_no = idx + 1;
        if line.is_empty() {
            return Err(invalid(format!("PARETO line {line_no}: blank lines are forbidden")));
        }
        let fields: Vec<&str> = line.split('\t').collect();
        match fields.first().copied() {
            Some("PARETO") => {
                if header_seen || rows_started || fields.len() != 3 || fields[1].is_empty() || fields[2].is_empty() {
                    return Err(invalid(format!("PARETO line {line_no}: malformed or out-of-order metadata")));
                }
                if meta.insert(fields[1].to_owned(), fields[2].to_owned()).is_some() {
                    return Err(invalid(format!("PARETO line {line_no}: duplicate metadata key")));
                }
            }
            Some("RESULT_HEADER") => {
                if header_seen || rows_started || fields.iter().copied().ne(PARETO_HEADER.iter().copied()) {
                    return Err(invalid(format!("PARETO line {line_no}: exact RESULT_HEADER required")));
                }
                header_seen = true;
            }
            Some("RESULT") => {
                if !header_seen || fields.len() != PARETO_HEADER.len() {
                    return Err(invalid(format!("PARETO line {line_no}: malformed RESULT row")));
                }
                rows_started = true;
                let budget_point_id = fields[1].to_owned();
                let policy_id = fields[2].to_owned();
                parse_token(&budget_point_id, "budget_point_id")?;
                parse_token(&policy_id, "policy_id")?;
                let upstream_status = fields[3].to_owned();
                if !matches!(upstream_status.as_str(), "FEASIBLE_FRONTIER" | "FEASIBLE_DOMINATED" | "INFEASIBLE") {
                    return Err(invalid(format!("PARETO line {line_no}: unknown upstream status {upstream_status}")));
                }
                let utility_estimate = parse_f64(fields[4], "utility_delta_vs_all_cheap", line_no)?;
                let latency = parse_u64(fields[5], "actual_all_in_latency_us", line_no)?;
                let energy = parse_u64(fields[6], "actual_all_in_energy_uj", line_no)?;
                let latency_budget = parse_u64(fields[7], "latency_budget_us", line_no)?;
                let energy_budget = parse_u64(fields[8], "energy_budget_uj", line_no)?;
                // Allocator costs are retained in the source receipt; the robust
                // comparison uses actual all-in latency and energy, not these components.
                let _allocator_latency = parse_u64(fields[9], "allocator_latency_us", line_no)?;
                let _allocator_energy = parse_u64(fields[10], "allocator_energy_uj", line_no)?;
                points.push(Point {
                    budget_point_id,
                    policy_id,
                    upstream_status,
                    utility_estimate,
                    latency,
                    energy,
                    latency_budget,
                    energy_budget,
                    reasons: fields[11].to_owned(),
                    dominated_by: fields[12].to_owned(),
                });
            }
            _ => return Err(invalid(format!("PARETO line {line_no}: unknown record type"))),
        }
    }
    if !header_seen || points.is_empty() {
        return Err(invalid("PARETO receipt must contain metadata, one exact header, and result rows"));
    }
    require_exact_keys(&meta, PARETO_META_KEYS, "PARETO")?;
    if meta["format"] != RECEIPT_VERSION || meta["status"] != PARETO_STATUS {
        return Err(invalid("unsupported PARETO receipt format or status"));
    }
    if meta["scalarized_score"] != "NONE"
        || meta["upstream_risk_certificate_verified_here"] != "false"
        || meta["performance_claim"] != "NONE"
        || meta["frontier_objectives"] != "MAXIMIZE_UTILITY_DELTA;MINIMIZE_ACTUAL_LATENCY;MINIMIZE_ACTUAL_ENERGY"
    {
        return Err(invalid("PARETO receipt has unexpected analysis semantics or non-claim fields"));
    }
    if !matches!(meta["split"].as_str(), "test" | "private_test") {
        return Err(invalid("PARETO split must be test or private_test"));
    }
    for key in ["utility_spec_sha256", "bundle_lock_sha256", "paired_unit_set_sha256", "sweep_input_sha256"] {
        validate_sha(&meta[key], key)?;
    }
    let policy_count: usize = meta["policy_count"].parse().map_err(|_| invalid("policy_count must be an integer"))?;
    let budget_count: usize = meta["budget_point_count"].parse().map_err(|_| invalid("budget_point_count must be an integer"))?;
    if policy_count == 0 || budget_count == 0 || policy_count.checked_mul(budget_count) != Some(points.len()) {
        return Err(invalid("PARETO result count must equal policy_count × budget_point_count"));
    }
    let distinct_policies: BTreeSet<&str> = points.iter().map(|p| p.policy_id.as_str()).collect();
    let distinct_budgets: BTreeSet<&str> = points.iter().map(|p| p.budget_point_id.as_str()).collect();
    if distinct_policies.len() != policy_count || distinct_budgets.len() != budget_count {
        return Err(invalid("PARETO policy/budget identity counts disagree with metadata"));
    }
    let mut seen = BTreeSet::new();
    for p in &points {
        if !seen.insert(p.key()) {
            return Err(invalid(format!("duplicate PARETO point {}", p.label())));
        }
    }
    for budget in &distinct_budgets {
        for policy in &distinct_policies {
            if !seen.contains(&(budget.to_string(), policy.to_string())) {
                return Err(invalid(format!("PARETO grid missing {budget}/{policy}")));
            }
        }
    }
    let eligible = points.iter().filter(|p| p.eligible()).count();
    let infeasible = points.iter().filter(|p| p.upstream_status == "INFEASIBLE").count();
    if meta["eligible_runs"] != eligible.to_string() || meta["infeasible_runs"] != infeasible.to_string() {
        return Err(invalid("PARETO eligible/infeasible counts disagree with result rows"));
    }
    Ok(Receipt { meta, points })
}

fn parse_bands(input: &str, receipt: &Receipt, expected_receipt_sha: &str, expected_inference_sha: &str) -> Result<Bands, String> {
    let mut meta = BTreeMap::new();
    let mut values = BTreeMap::new();
    let mut header_seen = false;
    let mut rows_started = false;
    for (idx, line) in input.lines().enumerate() {
        let line_no = idx + 1;
        if line.is_empty() {
            return Err(invalid(format!("BANDS line {line_no}: blank lines are forbidden")));
        }
        let fields: Vec<&str> = line.split('\t').collect();
        match fields.first().copied() {
            Some("META") => {
                if header_seen || rows_started || fields.len() != 3 || fields[1].is_empty() || fields[2].is_empty() {
                    return Err(invalid(format!("BANDS line {line_no}: malformed or out-of-order metadata")));
                }
                if meta.insert(fields[1].to_owned(), fields[2].to_owned()).is_some() {
                    return Err(invalid(format!("BANDS line {line_no}: duplicate metadata key")));
                }
            }
            Some("BAND_HEADER") => {
                if header_seen || rows_started || fields.iter().copied().ne(BAND_HEADER.iter().copied()) {
                    return Err(invalid(format!("BANDS line {line_no}: exact BAND_HEADER required")));
                }
                header_seen = true;
            }
            Some("BAND") => {
                if !header_seen || fields.len() != BAND_HEADER.len() {
                    return Err(invalid(format!("BANDS line {line_no}: malformed BAND row")));
                }
                rows_started = true;
                let budget = fields[1].to_owned();
                let policy = fields[2].to_owned();
                parse_token(&budget, "budget_point_id")?;
                parse_token(&policy, "policy_id")?;
                let lower = parse_f64(fields[3], "utility_lcb", line_no)?;
                let upper = parse_f64(fields[4], "utility_ucb", line_no)?;
                if lower > upper {
                    return Err(invalid(format!("BANDS line {line_no}: utility_lcb exceeds utility_ucb")));
                }
                if values.insert((budget.clone(), policy.clone()), Band { lower, upper }).is_some() {
                    return Err(invalid(format!("BANDS line {line_no}: duplicate band {budget}/{policy}")));
                }
            }
            _ => return Err(invalid(format!("BANDS line {line_no}: unknown record type"))),
        }
    }
    if !header_seen {
        return Err(invalid("BANDS file must contain metadata and one exact BAND_HEADER"));
    }
    require_exact_keys(&meta, BAND_META_KEYS, "BANDS")?;
    if meta["format"] != "VIS-DVPC-CONFIDENCE-BANDS-001-v1" {
        return Err(invalid("unsupported confidence-band format"));
    }
    if meta["familywise_simultaneous"] != "true" || meta["coverage_scope"] != "ALL_POLICY_BUDGET_POINTS" {
        return Err(invalid("bands must assert family-wise simultaneous coverage over ALL_POLICY_BUDGET_POINTS"));
    }
    let confidence = parse_f64(&meta["confidence_level"], "confidence_level", 0)?;
    if !(0.95..1.0).contains(&confidence) {
        return Err(invalid("confidence_level must be at least 0.95 and strictly below 1.0"));
    }
    let band_inference_sha = validate_sha(&meta["inference_receipt_sha256"], "inference_receipt_sha256")?;
    if band_inference_sha != expected_inference_sha {
        return Err(invalid("inference receipt digest does not match the caller-pinned digest"));
    }
    if validate_sha(&meta["source_pareto_receipt_sha256"], "source_pareto_receipt_sha256")? != expected_receipt_sha {
        return Err(invalid("confidence bands are not bound to the exact PARETO receipt bytes"));
    }
    for (band_key, receipt_key) in [
        ("experiment_id", "experiment_id"),
        ("dataset_id", "dataset_id"),
        ("dataset_revision", "dataset_revision"),
        ("split", "split"),
        ("utility_spec_sha256", "utility_spec_sha256"),
        ("bundle_lock_sha256", "bundle_lock_sha256"),
        ("paired_unit_set_sha256", "paired_unit_set_sha256"),
    ] {
        if meta[band_key] != receipt.meta[receipt_key] {
            return Err(invalid(format!("confidence-band {band_key} differs from PARETO receipt")));
        }
    }
    if validate_sha(&meta["source_sweep_sha256"], "source_sweep_sha256")? != receipt.meta["sweep_input_sha256"] {
        return Err(invalid("confidence bands do not bind to the same frozen sweep input"));
    }
    let expected_count = receipt.points.len();
    let point_count: usize = meta["point_count"].parse().map_err(|_| invalid("point_count must be an integer"))?;
    if point_count != expected_count || values.len() != expected_count {
        return Err(invalid("confidence-band count must equal the exact PARETO result grid"));
    }
    let expected_keys: BTreeSet<(String, String)> = receipt.points.iter().map(Point::key).collect();
    let actual_keys: BTreeSet<(String, String)> = values.keys().cloned().collect();
    if actual_keys != expected_keys {
        return Err(invalid("confidence-band point identities do not exactly match the PARETO grid"));
    }
    Ok(Bands { meta, values })
}

fn resource_no_worse(a: &Point, b: &Point) -> bool {
    a.latency <= b.latency && a.energy <= b.energy
}

fn resources_strictly_better(a: &Point, b: &Point) -> bool {
    a.latency < b.latency || a.energy < b.energy
}

fn certainly_dominates(a: &Point, a_band: Band, b: &Point, b_band: Band) -> bool {
    resource_no_worse(a, b)
        && a_band.lower >= b_band.upper
        && (a_band.lower > b_band.upper || resources_strictly_better(a, b))
}

fn possibly_dominates(a: &Point, a_band: Band, b: &Point, b_band: Band) -> bool {
    resource_no_worse(a, b)
        && a_band.upper >= b_band.lower
        && (a_band.upper > b_band.lower || resources_strictly_better(a, b))
}

fn exact_equivalent(a: &Point, a_band: Band, b: &Point, b_band: Band) -> bool {
    a.latency == b.latency
        && a.energy == b.energy
        && a_band.lower == b_band.lower
        && a_band.upper == b_band.upper
}

fn classify(receipt: &Receipt, bands: &Bands) -> Vec<Classified> {
    let mut points = receipt.points.clone();
    points.sort_by(|a, b| a.key().cmp(&b.key()));
    let mut output = Vec::with_capacity(points.len());
    for point in &points {
        let band = bands.values[&point.key()];
        if !point.eligible() {
            output.push(Classified {
                point: point.clone(), band, status: "INFEASIBLE",
                reason: "UPSTREAM_FEASIBILITY_GATE_FAILED;UNCERTAINTY_CANNOT_REHABILITATE",
                dominated_by: "-".to_owned(),
            });
            continue;
        }
        let mut certified_dominators = Vec::new();
        let mut possible_dominators = Vec::new();
        let mut equivalent_predecessor = None;
        for other in &points {
            if other.key() == point.key() || !other.eligible() {
                continue;
            }
            let other_band = bands.values[&other.key()];
            if exact_equivalent(other, other_band, point, band) {
                if other.key() < point.key() {
                    equivalent_predecessor = Some(other.label());
                }
                continue;
            }
            if certainly_dominates(other, other_band, point, band) {
                certified_dominators.push(other.label());
            } else if possibly_dominates(other, other_band, point, band) {
                possible_dominators.push(other.label());
            }
        }
        certified_dominators.sort();
        possible_dominators.sort();
        let (status, reason, dominated_by) = if let Some(first) = certified_dominators.first() {
            ("BAND_DOMINATED", "SIMULTANEOUS_UTILITY_BANDS_AND_RESOURCES_ESTABLISH_DOMINANCE", first.clone())
        } else if !possible_dominators.is_empty() {
            ("UNRESOLVED", "POTENTIAL_DOMINATOR_CANNOT_BE_CONFIRMED_OR_EXCLUDED_FROM_INTERVALS", possible_dominators.join(";"))
        } else if let Some(first) = equivalent_predecessor {
            ("DUPLICATE_EQUIVALENT", "IDENTICAL_INTERVALS_AND_RESOURCE_VALUES;LEXICOGRAPHIC_REPRESENTATIVE_RETAINED", first)
        } else {
            ("BAND_NONDOMINATED", "NO_ELIGIBLE_POINT_CAN_DOMINATE_UNDER_SUPPLIED_SIMULTANEOUS_BANDS", "-".to_owned())
        };
        output.push(Classified { point: point.clone(), band, status, reason, dominated_by });
    }
    output
}

fn number(n: f64) -> String {
    format!("{n:.12}")
}

fn analyze(
    pareto_bytes: &[u8],
    expected_pareto_sha: &str,
    bands_bytes: &[u8],
    expected_bands_sha: &str,
    expected_inference_sha: &str,
) -> Result<String, String> {
    let expected_pareto_sha = validate_sha(expected_pareto_sha, "expected PARETO receipt digest")?;
    let expected_bands_sha = validate_sha(expected_bands_sha, "expected confidence-band digest")?;
    let expected_inference_sha = validate_sha(expected_inference_sha, "expected inference receipt digest")?;
    let actual_pareto_sha = sha256_hex(pareto_bytes);
    let actual_bands_sha = sha256_hex(bands_bytes);
    if actual_pareto_sha != expected_pareto_sha {
        return Err("PARETO receipt bytes do not match the caller-pinned digest".to_owned());
    }
    if actual_bands_sha != expected_bands_sha {
        return Err("confidence-band bytes do not match the caller-pinned digest".to_owned());
    }
    let pareto_text = std::str::from_utf8(pareto_bytes).map_err(|_| invalid("PARETO receipt is not UTF-8"))?;
    let bands_text = std::str::from_utf8(bands_bytes).map_err(|_| invalid("confidence-band file is not UTF-8"))?;
    let receipt = parse_receipt(pareto_text)?;
    let bands = parse_bands(bands_text, &receipt, &actual_pareto_sha, &expected_inference_sha)?;
    let classified = classify(&receipt, &bands);
    let status_count = |name: &str| classified.iter().filter(|p| p.status == name).count();
    let mut output = String::new();
    let _ = writeln!(output, "CONFIDENCE_FRONTIER\tformat\t{OUTPUT_VERSION}");
    let _ = writeln!(output, "CONFIDENCE_FRONTIER\tstatus\tUNCERTAINTY_AWARE_RECLASSIFICATION_ONLY");
    let _ = writeln!(output, "CONFIDENCE_FRONTIER\texperiment_id\t{}", receipt.meta["experiment_id"]);
    let _ = writeln!(output, "CONFIDENCE_FRONTIER\tdataset_id\t{}", receipt.meta["dataset_id"]);
    let _ = writeln!(output, "CONFIDENCE_FRONTIER\tdataset_revision\t{}", receipt.meta["dataset_revision"]);
    let _ = writeln!(output, "CONFIDENCE_FRONTIER\tsplit\t{}", receipt.meta["split"]);
    let _ = writeln!(output, "CONFIDENCE_FRONTIER\tbundle_lock_sha256\t{}", receipt.meta["bundle_lock_sha256"]);
    let _ = writeln!(output, "CONFIDENCE_FRONTIER\tpaired_unit_set_sha256\t{}", receipt.meta["paired_unit_set_sha256"]);
    let _ = writeln!(output, "CONFIDENCE_FRONTIER\tsweep_input_sha256\t{}", receipt.meta["sweep_input_sha256"]);
    let _ = writeln!(output, "CONFIDENCE_FRONTIER\tpareto_receipt_sha256\t{actual_pareto_sha}");
    let _ = writeln!(output, "CONFIDENCE_FRONTIER\tconfidence_bands_sha256\t{actual_bands_sha}");
    let _ = writeln!(output, "CONFIDENCE_FRONTIER\tinference_receipt_sha256\t{}", bands.meta["inference_receipt_sha256"]);
    let _ = writeln!(output, "CONFIDENCE_FRONTIER\tconfidence_level\t{}", bands.meta["confidence_level"]);
    let _ = writeln!(output, "CONFIDENCE_FRONTIER\tfamilywise_simultaneous\ttrue");
    let _ = writeln!(output, "CONFIDENCE_FRONTIER\tinterval_procedure_verified_here\tfalse");
    let _ = writeln!(output, "CONFIDENCE_FRONTIER\tupstream_safety_certificate_verified_here\tfalse");
    let _ = writeln!(output, "CONFIDENCE_FRONTIER\tband_nondominated_count\t{}", status_count("BAND_NONDOMINATED"));
    let _ = writeln!(output, "CONFIDENCE_FRONTIER\tband_dominated_count\t{}", status_count("BAND_DOMINATED"));
    let _ = writeln!(output, "CONFIDENCE_FRONTIER\tunresolved_count\t{}", status_count("UNRESOLVED"));
    let _ = writeln!(output, "CONFIDENCE_FRONTIER\tduplicate_equivalent_count\t{}", status_count("DUPLICATE_EQUIVALENT"));
    let _ = writeln!(output, "CONFIDENCE_FRONTIER\tinfeasible_count\t{}", status_count("INFEASIBLE"));
    let _ = writeln!(output, "CONFIDENCE_FRONTIER\tperformance_claim\tNONE");
    let _ = writeln!(output, "CONFIDENCE_RESULT_HEADER\tbudget_point_id\tpolicy_id\tstatus\tutility_estimate\tutility_lcb\tutility_ucb\tactual_all_in_latency_us\tactual_all_in_energy_uj\tlatency_budget_us\tenergy_budget_uj\tupstream_point_status\tupstream_reasons\treason\tupstream_point_dominated_by");
    for item in &classified {
        let _ = writeln!(
            output,
            "CONFIDENCE_RESULT\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            item.point.budget_point_id,
            item.point.policy_id,
            item.status,
            number(item.point.utility_estimate),
            number(item.band.lower),
            number(item.band.upper),
            item.point.latency,
            item.point.energy,
            item.point.latency_budget,
            item.point.energy_budget,
            item.point.upstream_status,
            item.point.reasons,
            item.reason,
            item.point.dominated_by,
            item.dominated_by,

        );
    }
    Ok(output)
}

fn run_cli(args: &[String]) -> Result<String, String> {
    if args.len() != 6 {
        return Err("usage: vision-dvpc-confidence-frontier PARETO.tsv EXPECTED_PARETO_SHA256 BANDS.tsv EXPECTED_BANDS_SHA256 EXPECTED_INFERENCE_RECEIPT_SHA256".to_owned());
    }
    let pareto = fs::read(&args[1]).map_err(|e| invalid(format!("cannot read PARETO receipt: {e}")))?;
    let bands = fs::read(&args[3]).map_err(|e| invalid(format!("cannot read confidence bands: {e}")))?;
    analyze(&pareto, &args[2], &bands, &args[4], &args[5])
}

fn main() {
    let args: Vec<String> = env::args().collect();
    match run_cli(&args) {
        Ok(output) => print!("{output}"),
        Err(message) => {
            eprintln!("VIS-DVPC-018 error: {message}");
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SHA_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const SHA_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const SHA_C: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";

    #[derive(Clone)]
    struct Spec<'a> {
        budget: &'a str,
        policy: &'a str,
        status: &'a str,
        utility: f64,
        latency: u64,
        energy: u64,
        reasons: &'a str,
    }

    fn receipt(specs: &[Spec<'_>]) -> String {
        let policies: BTreeSet<&str> = specs.iter().map(|s| s.policy).collect();
        let budgets: BTreeSet<&str> = specs.iter().map(|s| s.budget).collect();
        let feasible = specs.iter().filter(|s| s.status != "INFEASIBLE").count();
        let infeasible = specs.len() - feasible;
        let frontier = specs.iter().filter(|s| s.status == "FEASIBLE_FRONTIER").count();
        let sweep_sha = SHA_A;
        let mut out = String::new();
        let meta_rows: Vec<(String, String)> = vec![
            ("format".to_owned(), RECEIPT_VERSION.to_owned()),
            ("status".to_owned(), PARETO_STATUS.to_owned()),
            ("experiment_id".to_owned(), "synthetic-dvpc-018".to_owned()),
            ("dataset_id".to_owned(), "synthetic".to_owned()),
            ("dataset_revision".to_owned(), "fixture-v1".to_owned()),
            ("split".to_owned(), "test".to_owned()),
            ("utility_spec_sha256".to_owned(), SHA_A.to_owned()),
            ("bundle_lock_sha256".to_owned(), SHA_B.to_owned()),
            ("paired_unit_set_sha256".to_owned(), SHA_C.to_owned()),
            ("sweep_input_sha256".to_owned(), SHA_A.to_owned()),
            ("policy_count".to_owned(), policies.len().to_string()),
            ("budget_point_count".to_owned(), budgets.len().to_string()),
            ("independent_clusters".to_owned(), "10".to_owned()),
            ("eligible_runs".to_owned(), feasible.to_string()),
            ("infeasible_runs".to_owned(), infeasible.to_string()),
            ("frontier_runs".to_owned(), frontier.to_string()),
            ("frontier_objectives".to_owned(), "MAXIMIZE_UTILITY_DELTA;MINIMIZE_ACTUAL_LATENCY;MINIMIZE_ACTUAL_ENERGY".to_owned()),
            ("scalarized_score".to_owned(), "NONE".to_owned()),
            ("upstream_risk_certificate_verified_here".to_owned(), "false".to_owned()),
            ("performance_claim".to_owned(), "NONE".to_owned()),
        ];
        for (k, v) in meta_rows {
            let _ = writeln!(out, "PARETO\\t{k}\\t{v}");
        }
        out.push_str("RESULT_HEADER\tbudget_point_id\tpolicy_id\tstatus\tutility_delta_vs_all_cheap\tactual_all_in_latency_us\tactual_all_in_energy_uj\tlatency_budget_us\tenergy_budget_uj\tallocator_latency_us\tallocator_energy_uj\treasons\tdominated_by\n");
        for s in specs {
            let _ = writeln!(out, "RESULT\t{}\t{}\t{}\t{:.6}\t{}\t{}\t1000\t1000\t1\t1\t{}\t-", s.budget, s.policy, s.status, s.utility, s.latency, s.energy, s.reasons);
        }
        out
    }

    fn bands(pareto: &str, specs: &[(&str, &str, f64, f64)], simultaneous: &str) -> String {
        let pareto_sha = sha256_hex(pareto.as_bytes());
        let mut out = String::new();
        let meta_rows: Vec<(String, String)> = vec![
            ("format".to_owned(), "VIS-DVPC-CONFIDENCE-BANDS-001-v1".to_owned()),
            ("experiment_id".to_owned(), "synthetic-dvpc-018".to_owned()),
            ("dataset_id".to_owned(), "synthetic".to_owned()),
            ("dataset_revision".to_owned(), "fixture-v1".to_owned()),
            ("split".to_owned(), "test".to_owned()),
            ("utility_spec_sha256".to_owned(), SHA_A.to_owned()),
            ("bundle_lock_sha256".to_owned(), SHA_B.to_owned()),
            ("paired_unit_set_sha256".to_owned(), SHA_C.to_owned()),
            ("source_sweep_sha256".to_owned(), SHA_A.to_owned()),
            ("source_pareto_receipt_sha256".to_owned(), pareto_sha),
            ("inference_receipt_sha256".to_owned(), SHA_C.to_owned()),
            ("confidence_level".to_owned(), "0.95".to_owned()),
            ("familywise_simultaneous".to_owned(), simultaneous.to_owned()),
            ("coverage_scope".to_owned(), "ALL_POLICY_BUDGET_POINTS".to_owned()),
            ("point_count".to_owned(), specs.len().to_string()),
        ];
        for (k, v) in meta_rows {
            let _ = writeln!(out, "META\\t{k}\\t{v}");
        }
        out.push_str("BAND_HEADER\tbudget_point_id\tpolicy_id\tutility_lcb\tutility_ucb\n");
        for (budget, policy, lower, upper) in specs {
            let _ = writeln!(out, "BAND\t{budget}\t{policy}\t{lower:.6}\t{upper:.6}");
        }
        out
    }

    fn execute(pareto: &str, band_rows: &[(&str, &str, f64, f64)]) -> Result<String, String> {
        let b = bands(pareto, band_rows, "true");
        analyze(
            pareto.as_bytes(),
            &sha256_hex(pareto.as_bytes()),
            b.as_bytes(),
            &sha256_hex(b.as_bytes()),
            SHA_C,
        )
    }

    #[test]
    fn separated_simultaneous_bands_certify_dominance() {
        let p = receipt(&[
            Spec { budget: "p10", policy: "cheap", status: "FEASIBLE_FRONTIER", utility: 0.50, latency: 10, energy: 10, reasons: "-" },
            Spec { budget: "p10", policy: "full", status: "FEASIBLE_DOMINATED", utility: 0.30, latency: 20, energy: 20, reasons: "-" },
        ]);
        let out = execute(&p, &[("p10", "cheap", 0.45, 0.55), ("p10", "full", 0.20, 0.35)]).unwrap();
        assert!(out.contains("CONFIDENCE_RESULT\tp10\tcheap\tBAND_NONDOMINATED"));
        assert!(out.contains("CONFIDENCE_RESULT\tp10\tfull\tBAND_DOMINATED"));
    }

    #[test]
    fn overlapping_bands_do_not_turn_point_estimate_order_into_a_win() {
        let p = receipt(&[
            Spec { budget: "p10", policy: "candidate", status: "FEASIBLE_FRONTIER", utility: 0.60, latency: 10, energy: 10, reasons: "-" },
            Spec { budget: "p10", policy: "baseline", status: "FEASIBLE_DOMINATED", utility: 0.40, latency: 20, energy: 20, reasons: "-" },
        ]);
        let out = execute(&p, &[("p10", "candidate", 0.20, 0.70), ("p10", "baseline", 0.30, 0.80)]).unwrap();
        assert!(out.contains("CONFIDENCE_RESULT\tp10\tcandidate\tBAND_NONDOMINATED"));
        assert!(out.contains("CONFIDENCE_RESULT\tp10\tbaseline\tUNRESOLVED"));
    }

    #[test]
    fn resource_incomparability_prevents_dominance() {
        let p = receipt(&[
            Spec { budget: "p10", policy: "low-latency", status: "FEASIBLE_FRONTIER", utility: 0.50, latency: 10, energy: 20, reasons: "-" },
            Spec { budget: "p10", policy: "low-energy", status: "FEASIBLE_DOMINATED", utility: 0.20, latency: 20, energy: 10, reasons: "-" },
        ]);
        let out = execute(&p, &[("p10", "low-latency", 0.45, 0.55), ("p10", "low-energy", 0.10, 0.30)]).unwrap();
        assert!(out.contains("CONFIDENCE_RESULT\tp10\tlow-latency\tBAND_NONDOMINATED"));
        assert!(out.contains("CONFIDENCE_RESULT\tp10\tlow-energy\tBAND_NONDOMINATED"));
    }

    #[test]
    fn unsafe_or_over_budget_point_cannot_reenter_frontier() {
        let p = receipt(&[
            Spec { budget: "p10", policy: "safe", status: "FEASIBLE_FRONTIER", utility: 0.20, latency: 10, energy: 10, reasons: "-" },
            Spec { budget: "p10", policy: "unsafe", status: "INFEASIBLE", utility: 0.99, latency: 1, energy: 1, reasons: "SAFETY_VIOLATION" },
        ]);
        let out = execute(&p, &[("p10", "safe", 0.10, 0.30), ("p10", "unsafe", 0.90, 1.00)]).unwrap();
        assert!(out.contains("CONFIDENCE_RESULT\tp10\tunsafe\tINFEASIBLE"));
        assert!(out.contains("UNCERTAINTY_CANNOT_REHABILITATE"));
    }

    #[test]
    fn identical_points_get_deterministic_duplicate_label_not_false_dominance() {
        let p = receipt(&[
            Spec { budget: "p10", policy: "a", status: "FEASIBLE_FRONTIER", utility: 0.50, latency: 10, energy: 10, reasons: "-" },
            Spec { budget: "p10", policy: "b", status: "FEASIBLE_DOMINATED", utility: 0.50, latency: 10, energy: 10, reasons: "-" },
        ]);
        let out = execute(&p, &[("p10", "a", 0.40, 0.60), ("p10", "b", 0.40, 0.60)]).unwrap();
        assert!(out.contains("CONFIDENCE_RESULT\tp10\ta\tBAND_NONDOMINATED"));
        assert!(out.contains("CONFIDENCE_RESULT\tp10\tb\tDUPLICATE_EQUIVALENT"));
    }

    #[test]
    fn incomplete_band_grid_fails_closed() {
        let p = receipt(&[
            Spec { budget: "p10", policy: "a", status: "FEASIBLE_FRONTIER", utility: 0.50, latency: 10, energy: 10, reasons: "-" },
            Spec { budget: "p10", policy: "b", status: "FEASIBLE_DOMINATED", utility: 0.30, latency: 20, energy: 20, reasons: "-" },
        ]);
        let b = bands(&p, &[("p10", "a", 0.40, 0.60)], "true");
        let e = analyze(p.as_bytes(), &sha256_hex(p.as_bytes()), b.as_bytes(), &sha256_hex(b.as_bytes()), SHA_C).unwrap_err();
        assert!(e.contains("confidence-band count"));
    }

    #[test]
    fn non_simultaneous_bands_are_rejected() {
        let p = receipt(&[
            Spec { budget: "p10", policy: "a", status: "FEASIBLE_FRONTIER", utility: 0.50, latency: 10, energy: 10, reasons: "-" },
            Spec { budget: "p10", policy: "b", status: "FEASIBLE_DOMINATED", utility: 0.30, latency: 20, energy: 20, reasons: "-" },
        ]);
        let b = bands(&p, &[("p10", "a", 0.40, 0.60), ("p10", "b", 0.20, 0.40)], "false");
        let e = analyze(p.as_bytes(), &sha256_hex(p.as_bytes()), b.as_bytes(), &sha256_hex(b.as_bytes()), SHA_C).unwrap_err();
        assert!(e.contains("family-wise simultaneous"));
    }

    #[test]
    fn a_changed_pareto_digest_is_rejected() {
        let p = receipt(&[
            Spec { budget: "p10", policy: "a", status: "FEASIBLE_FRONTIER", utility: 0.50, latency: 10, energy: 10, reasons: "-" },
            Spec { budget: "p10", policy: "b", status: "FEASIBLE_DOMINATED", utility: 0.30, latency: 20, energy: 20, reasons: "-" },
        ]);
        let b = bands(&p, &[("p10", "a", 0.40, 0.60), ("p10", "b", 0.20, 0.40)], "true");
        let e = analyze(p.as_bytes(), SHA_B, b.as_bytes(), &sha256_hex(b.as_bytes()), SHA_C).unwrap_err();
        assert!(e.contains("PARETO receipt bytes"));
    }

    #[test]
    fn duplicate_and_unknown_band_rows_are_rejected() {
        let p = receipt(&[
            Spec { budget: "p10", policy: "a", status: "FEASIBLE_FRONTIER", utility: 0.50, latency: 10, energy: 10, reasons: "-" },
            Spec { budget: "p10", policy: "b", status: "FEASIBLE_DOMINATED", utility: 0.30, latency: 20, energy: 20, reasons: "-" },
        ]);
        let b = bands(&p, &[("p10", "a", 0.40, 0.60), ("p10", "b", 0.20, 0.40), ("p10", "extra", 0.20, 0.40)], "true");
        let e = analyze(p.as_bytes(), &sha256_hex(p.as_bytes()), b.as_bytes(), &sha256_hex(b.as_bytes()), SHA_C).unwrap_err();
        assert!(e.contains("duplicate band") || e.contains("confidence-band count") || e.contains("identities"));
    }

    #[test]
    fn sha256_known_vector() {
        assert_eq!(sha256_hex(b"abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    }
}
