// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Shared quality gates for the abstract-thought macro pool.
//!
//! The examples use these gates for human-readable reports; tests can use the
//! same functions as hard assertions. Keep this module small and deterministic:
//! it should describe macro-pool health, not mutate the engine.

use crate::hdc::conjecture_engine::{BinOp, Expr};

use super::dynamic_grammar::MacroPoolMetrics;

#[derive(Debug, Clone, PartialEq)]
pub struct MacroQualityGate {
    pub name: &'static str,
    pub passed: bool,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MacroQualityReport {
    pub gates: Vec<MacroQualityGate>,
}

impl MacroQualityReport {
    pub fn new() -> Self {
        Self { gates: Vec::new() }
    }

    pub fn push(&mut self, name: &'static str, passed: bool, detail: impl Into<String>) {
        self.gates.push(MacroQualityGate {
            name,
            passed,
            detail: detail.into(),
        });
    }

    pub fn passed(&self) -> bool {
        self.gates.iter().all(|gate| gate.passed)
    }
}

impl Default for MacroQualityReport {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MacroQualityThresholds {
    pub min_activation_precision: f64,
    pub max_cold_dominance: usize,
    pub require_used_macro: bool,
    pub allowed_signatures: Vec<String>,
}

impl MacroQualityThresholds {
    pub fn one_dimensional() -> Self {
        Self {
            min_activation_precision: 0.50,
            max_cold_dominance: 1,
            require_used_macro: true,
            allowed_signatures: vec!["n".to_string(), "<const>".to_string()],
        }
    }
}

pub fn print_gate(gate: &MacroQualityGate) {
    let status = if gate.passed { "PASS" } else { "FAIL" };
    println!("  {:<34} {}  {}", gate.name, status, gate.detail);
}

pub fn print_report(report: &MacroQualityReport, overall_name: &str) {
    for gate in &report.gates {
        print_gate(gate);
    }
    let final_status = if report.passed() { "PASS" } else { "FAIL" };
    println!("  {}", "─".repeat(64));
    println!("  {:<34} {}", overall_name, final_status);
}

pub fn maybe_enforce(report: &MacroQualityReport) {
    if !report.passed()
        && std::env::var("SYMTHAEA_ENFORCE_MACRO_GATES")
            .ok()
            .as_deref()
            == Some("1")
    {
        std::process::exit(1);
    }
}

pub fn evaluate_common_metrics(
    metrics: Option<&MacroPoolMetrics>,
    thresholds: &MacroQualityThresholds,
) -> MacroQualityReport {
    let mut report = MacroQualityReport::new();

    report.push(
        "no_quarantined_promotions",
        metrics
            .map(|m| m.quarantined_operators == 0)
            .unwrap_or(false),
        metrics
            .map(|m| format!("quarantined={}", m.quarantined_operators))
            .unwrap_or_else(|| "metrics unavailable".to_string()),
    );

    if thresholds.require_used_macro {
        report.push(
            "has_used_macros",
            metrics.map(|m| m.used_operators > 0).unwrap_or(false),
            metrics
                .map(|m| format!("used={} total={}", m.used_operators, m.total_operators))
                .unwrap_or_else(|| "metrics unavailable".to_string()),
        );
    }

    report.push(
        "compatible_signatures",
        metrics
            .map(|m| {
                m.signature_stats.iter().all(|stat| {
                    thresholds
                        .allowed_signatures
                        .iter()
                        .any(|allowed| allowed == &stat.signature)
                })
            })
            .unwrap_or(false),
        metrics
            .map(|m| {
                let signatures: Vec<String> = m
                    .signature_stats
                    .iter()
                    .map(|stat| stat.signature.clone())
                    .collect();
                format!("signatures={:?}", signatures)
            })
            .unwrap_or_else(|| "metrics unavailable".to_string()),
    );

    report
}

pub fn nonsemantic_constants(expr: &Expr) -> Vec<f64> {
    let mut result = Vec::new();
    collect_nonsemantic_inner(expr, &mut result);
    result
}

fn collect_nonsemantic_inner(expr: &Expr, out: &mut Vec<f64>) {
    match expr {
        Expr::Const(c) => {
            if !is_semantic_template_constant(*c) {
                out.push(*c);
            }
        }
        Expr::Var(_) => {}
        Expr::BinOp(BinOp::Pow, l, r) => {
            collect_nonsemantic_inner(l, out);
            match r.as_ref() {
                Expr::Const(c) if is_semantic_pow_exponent(*c) => {}
                Expr::Const(c) => out.push(*c),
                _ => collect_nonsemantic_inner(r, out),
            }
        }
        Expr::BinOp(_, l, r) => {
            collect_nonsemantic_inner(l, out);
            collect_nonsemantic_inner(r, out);
        }
        Expr::Func(_, arg) => collect_nonsemantic_inner(arg, out),
        Expr::Sum(body, _) => collect_nonsemantic_inner(body, out),
    }
}

pub fn is_semantic_template_constant(c: f64) -> bool {
    (c - 0.0).abs() < 1e-12 || (c - 1.0).abs() < 1e-12 || (c - 2.0).abs() < 1e-12
}

pub fn is_semantic_pow_exponent(c: f64) -> bool {
    if !c.is_finite() {
        return false;
    }
    let rounded = c.round();
    if (c - rounded).abs() < 1e-9 && rounded.abs() <= 12.0 {
        return true;
    }
    let half_step = (c * 2.0).round() / 2.0;
    (c - half_step).abs() < 1e-9 && half_step.abs() <= 12.0
}
