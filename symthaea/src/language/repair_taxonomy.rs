// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Stable repair categories and hints for coding-agent feedback loops.

pub const FORCE_REPAIR_BENCH_ENV: &str = "SYMTHAEA_ENABLE_FORCED_REPAIR_BENCH";
pub const FORCE_GEODESIC_REJECTION_PREFIX: &str = "benchmark_force_geodesic_rejection:";
pub const FORCE_GEODESIC_REJECTION_UNLESS_REPAIR_MEMORY_PREFIX: &str =
    "benchmark_force_geodesic_rejection_unless_repair_memory:";

pub fn categorize_rejection(reason: &str) -> &'static str {
    let lower = reason.to_ascii_lowercase();
    if lower.contains("energy budget") {
        "energy_budget"
    } else if lower.contains("stub") || lower.contains("todo") || lower.contains("unimplemented") {
        "stub"
    } else if lower.contains("does not parse")
        || lower.contains("parse")
        || lower.contains("expected one of")
    {
        "parse_failure"
    } else if lower.contains("not found in this scope")
        || lower.contains("cannot find value")
        || lower.contains("unresolved")
    {
        "unresolved_identifier"
    } else if lower.contains("mismatched types")
        || lower.contains("type mismatch")
        || lower.contains("returns a value")
        || lower.contains("unit-returning")
        || lower.contains("return type")
        || lower.contains("signature expects")
    {
        "type_mismatch"
    } else if lower.contains("borrow") || lower.contains("moved") || lower.contains("move") {
        "ownership"
    } else if lower.contains("test") {
        "test_failure"
    } else if lower.contains("compile") || lower.contains("rustc") {
        "compile_failure"
    } else if lower.contains("sheaf") || lower.contains("coherence") {
        "sheaf_failure"
    } else if lower.contains("no similar") || lower.contains("analogy") {
        "analogy_miss"
    } else if lower.contains("llm") {
        "llm_failure"
    } else {
        "other"
    }
}

pub fn repair_hint_for_category(category: &str) -> &'static str {
    match category {
        "type_mismatch" | "compile_failure" => {
            "regenerate with the declared signature as the source of truth for arguments and return type"
        }
        "unresolved_identifier" => {
            "only use parameters and locals that are in scope, or introduce the binding first"
        }
        "ownership" => {
            "prefer borrowing, cloning, copying, or statement-only mutation according to the signature"
        }
        "stub" => "replace placeholders with concrete expressions or accumulator statements",
        "parse_failure" => {
            "emit complete Rust syntax with balanced delimiters and valid separators"
        }
        "sheaf_failure" => {
            "repair local data-flow facts: definitions, return shape, ownership, and stubs"
        }
        "test_failure" => "treat examples and generated tests as executable semantic constraints",
        "analogy_miss" => "fall back to direct synthesis from the signature and task purpose",
        "energy_budget" => {
            "prefer low-cost deterministic templates before expensive fallback tiers"
        }
        _ => "use the previous failure as a concrete negative example for the next candidate",
    }
}

pub fn repair_lesson_for_rejection(reason: &str) -> String {
    if let Some(hint) = extract_embedded_repair_hint(reason) {
        return hint;
    }
    let category =
        extract_embedded_category(reason).unwrap_or_else(|| categorize_rejection(reason));
    repair_hint_for_category(category).to_string()
}

pub fn extract_embedded_category(reason: &str) -> Option<&str> {
    let start = reason.find("category=")? + "category=".len();
    let rest = &reason[start..];
    let end = rest.find([';', ']']).unwrap_or(rest.len());
    let category = rest[..end].trim();
    (!category.is_empty()).then_some(category)
}

pub fn extract_embedded_repair_hint(reason: &str) -> Option<String> {
    let start = reason.find("repair_hint=")? + "repair_hint=".len();
    let rest = &reason[start..];
    let end = rest.find(']').unwrap_or(rest.len());
    let hint = rest[..end].trim();
    (!hint.is_empty()).then(|| hint.to_string())
}

pub fn forced_geodesic_rejection(constraints: &[String]) -> Option<&str> {
    if std::env::var_os(FORCE_REPAIR_BENCH_ENV).is_none() {
        return None;
    }
    constraints
        .iter()
        .find_map(|constraint| constraint.strip_prefix(FORCE_GEODESIC_REJECTION_PREFIX))
        .map(str::trim)
        .filter(|reason| !reason.is_empty())
}

pub fn forced_geodesic_rejection_unless_repair_memory(constraints: &[String]) -> Option<&str> {
    if std::env::var_os(FORCE_REPAIR_BENCH_ENV).is_none() {
        return None;
    }
    constraints
        .iter()
        .find_map(|constraint| {
            constraint.strip_prefix(FORCE_GEODESIC_REJECTION_UNLESS_REPAIR_MEMORY_PREFIX)
        })
        .map(str::trim)
        .filter(|reason| !reason.is_empty())
}
