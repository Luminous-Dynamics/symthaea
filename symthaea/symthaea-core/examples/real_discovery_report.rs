// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Real-data discovery quality report for Symthaea.
//!
//! This is a project-health harness, not an `egg` experiment. It runs real
//! observation families through the conjecture engine and reports the signals
//! that matter for deciding where to improve Symthaea next: verification rate,
//! EML backend coverage, macro-candidate pressure, suspicious formulas, and
//! runtime.
//!
//! Run:
//! ```sh
//! cargo run -p symthaea-core --example real_discovery_report \
//!   --features abstract_thought --release
//!
//! # Exit non-zero if health thresholds are breached:
//! cargo run -p symthaea-core --example real_discovery_report \
//!   --features abstract_thought --release -- --strict
//! ```

use std::time::Instant;

use symthaea_core::hdc::abstract_thought::expr_canonical_string;
use symthaea_core::hdc::conjecture_engine::{
    BinOp, Conjecture, ConjectureEngine, ConjectureStatus, Expr, MathDomain, ObservedSequence,
    PreferredEmlBackend, RegressorConfig, UnaryFn, observe_balmer_series, observe_bell_numbers,
    observe_blackbody_peak, observe_catalan, observe_central_binomial_limit,
    observe_derangement_ratio, observe_fibonacci_ratios, observe_hydrogen_energy_levels,
    observe_inverse_square_law, observe_kepler_third_law, observe_partitions,
    observe_prime_counting, observe_prime_gaps, observe_quantum_harmonic_oscillator,
    observe_relativistic_kinetic_energy, observe_stefan_boltzmann,
};
use symthaea_core::hdc::primitive_system::PrimitiveSystem;

#[derive(Debug)]
struct FamilyCase {
    name: &'static str,
    observations: Vec<ObservedSequence>,
}

#[derive(Debug, Default)]
struct StatusCounts {
    proposed: usize,
    numeric: usize,
    symbolic: usize,
    formal: usize,
    refuted: usize,
}

#[derive(Debug)]
struct SuspiciousFormula {
    source: String,
    formula: String,
    reason: String,
    mse: f64,
    mean_rel_error: Option<f64>,
    normalized_rmse: Option<f64>,
    confidence: f64,
}

#[derive(Debug)]
struct FamilyReport {
    name: &'static str,
    observations: usize,
    conjectures: usize,
    status: StatusCounts,
    eml_backed: usize,
    eml_verified: usize,
    constructive_only: usize,
    macro_candidates: usize,
    macro_candidate_buckets: usize,
    macro_operators: usize,
    unsupported_eml_shapes: Vec<(String, usize)>,
    suspicious: Vec<SuspiciousFormula>,
    elapsed_ms: u128,
}

#[derive(Debug, Clone, Copy)]
struct EvalQuality {
    mean_rel_error: f64,
    normalized_rmse: f64,
}

fn default_engine() -> ConjectureEngine {
    let mut engine = ConjectureEngine::with_config(RegressorConfig {
        population_size: 120,
        generations: 60,
        max_depth: 4,
        max_complexity: 12,
        lambda: 0.001,
        tournament_size: 5,
        mutation_rate: 0.3,
        seed: 42,
        ..RegressorConfig::default()
    });
    engine.enable_abstract_thought();
    engine
}

fn squares_sequence(max_n: usize) -> ObservedSequence {
    let data: Vec<_> = (1..=max_n).map(|n| (n as f64, (n * n) as f64)).collect();
    ObservedSequence::new("squares(n)", MathDomain::NumberTheory, data)
}

fn cubes_sequence(max_n: usize) -> ObservedSequence {
    let data: Vec<_> = (1..=max_n)
        .map(|n| (n as f64, (n * n * n) as f64))
        .collect();
    ObservedSequence::new("cubes(n)", MathDomain::NumberTheory, data)
}

fn triangular_sequence(max_n: usize) -> ObservedSequence {
    let data: Vec<_> = (1..=max_n)
        .map(|n| (n as f64, (n * (n + 1)) as f64 / 2.0))
        .collect();
    ObservedSequence::new("triangular(n)", MathDomain::Combinatorics, data)
}

fn family_cases() -> Vec<FamilyCase> {
    vec![
        FamilyCase {
            name: "simple_control",
            observations: vec![
                squares_sequence(14),
                cubes_sequence(14),
                triangular_sequence(14),
            ],
        },
        FamilyCase {
            name: "combinatorics_growth",
            observations: vec![
                observe_fibonacci_ratios(16),
                observe_partitions(12),
                observe_catalan(12),
                observe_bell_numbers(10),
                observe_central_binomial_limit(18),
            ],
        },
        FamilyCase {
            name: "number_theory",
            observations: vec![
                observe_prime_gaps(120),
                observe_prime_counting(60),
                observe_derangement_ratio(10),
            ],
        },
        FamilyCase {
            name: "physics_closed_form",
            observations: vec![
                observe_hydrogen_energy_levels(8),
                observe_quantum_harmonic_oscillator(8),
                observe_kepler_third_law(8),
                observe_stefan_boltzmann(8),
                observe_inverse_square_law(10),
            ],
        },
        FamilyCase {
            name: "physics_transcendental",
            observations: vec![
                observe_blackbody_peak(12),
                observe_balmer_series(8),
                observe_relativistic_kinetic_energy(12),
            ],
        },
    ]
}

fn count_status(conjectures: &[Conjecture]) -> StatusCounts {
    let mut counts = StatusCounts::default();
    for conjecture in conjectures {
        match conjecture.status {
            ConjectureStatus::Proposed => counts.proposed += 1,
            ConjectureStatus::NumericallyTested { .. } => counts.numeric += 1,
            ConjectureStatus::SymbolicallyChecked => counts.symbolic += 1,
            ConjectureStatus::FormallyVerified { .. } => counts.formal += 1,
            ConjectureStatus::Refuted { .. } => counts.refuted += 1,
        }
    }
    counts
}

fn is_verified_eml(conjecture: &Conjecture) -> bool {
    conjecture.eml_verified_real == Some(true)
        || conjecture.eml_verified_complex == Some(true)
        || conjecture.eml_verified_constructive_real == Some(true)
}

fn eval_quality(conjecture: &Conjecture, observations: &[ObservedSequence]) -> Option<EvalQuality> {
    let seq = observations
        .iter()
        .find(|seq| seq.name == conjecture.source)?;
    let (_, test) = seq.train_test_split();
    let eval_data = if test.is_empty() { &seq.data } else { &test };
    if eval_data.is_empty() {
        return None;
    }

    let mut sq_err_sum = 0.0;
    let mut sq_y_sum = 0.0;
    let mut rel_err_sum = 0.0;
    let mut rel_count = 0usize;

    for &(x, y) in eval_data {
        let predicted = conjecture.formula.eval(&[("n", x)]);
        if !predicted.is_finite() {
            return None;
        }
        let err = predicted - y;
        sq_err_sum += err * err;
        sq_y_sum += y * y;
        if y.abs() > 1e-10 {
            rel_err_sum += (err / y).abs();
            rel_count += 1;
        }
    }

    let rmse = (sq_err_sum / eval_data.len() as f64).sqrt();
    let y_rms = (sq_y_sum / eval_data.len() as f64).sqrt().max(1e-12);
    Some(EvalQuality {
        mean_rel_error: if rel_count == 0 {
            f64::INFINITY
        } else {
            rel_err_sum / rel_count as f64
        },
        normalized_rmse: rmse / y_rms,
    })
}

fn suspicious_reason(conjecture: &Conjecture, quality: Option<EvalQuality>) -> Option<String> {
    if !conjecture.training_mse.is_finite() {
        return Some("non-finite MSE".to_string());
    }
    if matches!(conjecture.status, ConjectureStatus::Refuted { .. }) {
        return Some("refuted".to_string());
    }
    if matches!(conjecture.status, ConjectureStatus::Proposed) {
        return Some("unverified after numerical pass".to_string());
    }
    if let Some(quality) = quality {
        if quality.normalized_rmse > 0.10 && quality.mean_rel_error > 0.10 {
            return Some(format!(
                "poor held-out fit: rel={:.2e}, nrmse={:.2e}",
                quality.mean_rel_error, quality.normalized_rmse
            ));
        }
    } else if conjecture.training_mse > 1.0 {
        return Some(format!("high MSE {:.2e}", conjecture.training_mse));
    }
    if conjecture.confidence < 0.5 {
        return Some(format!("low confidence {:.2}", conjecture.confidence));
    }
    if (conjecture.formula_str.contains("sin(") || conjecture.formula_str.contains("cos("))
        && conjecture.training_mse > 1e-6
    {
        return Some("oscillatory fit with non-negligible error".to_string());
    }
    if conjecture.formula_str.starts_with("rec:") {
        return Some("recurrence placeholder".to_string());
    }
    None
}

fn suspicious_formulas(
    conjectures: &[Conjecture],
    observations: &[ObservedSequence],
) -> Vec<SuspiciousFormula> {
    let mut suspicious: Vec<_> = conjectures
        .iter()
        .filter_map(|conjecture| {
            let quality = eval_quality(conjecture, observations);
            suspicious_reason(conjecture, quality).map(|reason| SuspiciousFormula {
                source: conjecture.source.clone(),
                formula: conjecture.formula_str.clone(),
                reason,
                mse: conjecture.training_mse,
                mean_rel_error: quality.map(|quality| quality.mean_rel_error),
                normalized_rmse: quality.map(|quality| quality.normalized_rmse),
                confidence: conjecture.confidence,
            })
        })
        .collect();

    suspicious.sort_by(|a, b| {
        b.mse
            .partial_cmp(&a.mse)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.confidence.partial_cmp(&b.confidence).unwrap())
    });
    suspicious
}

fn eml_support_gap(expr: &Expr) -> Option<String> {
    match expr {
        Expr::Var(_) => None,
        Expr::Const(c) if (*c - 1.0).abs() < 1e-12 => None,
        Expr::Const(c) => Some(format!("constant:{c:.6}")),
        Expr::BinOp(BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div, left, right) => {
            eml_support_gap(left).or_else(|| eml_support_gap(right))
        }
        Expr::BinOp(BinOp::Pow, left, right) => {
            if let Some(reason) = eml_support_gap(left) {
                return Some(reason);
            }
            match right.as_ref() {
                Expr::Const(c)
                    if (*c - -1.0).abs() < 1e-12
                        || (*c - 0.0).abs() < 1e-12
                        || (*c - 1.0).abs() < 1e-12
                        || (*c - 2.0).abs() < 1e-12 =>
                {
                    None
                }
                Expr::Const(c) => Some(format!("pow-exponent:{c:.6}")),
                other => eml_support_gap(other).or_else(|| Some("pow-nonconstant-exponent".into())),
            }
        }
        Expr::Func(UnaryFn::Exp | UnaryFn::Log, arg) => eml_support_gap(arg),
        Expr::Func(UnaryFn::Sqrt, arg) => {
            eml_support_gap(arg).or_else(|| Some("sqrt-rational-exponent".into()))
        }
        Expr::Func(other, _) => Some(format!("unary:{other:?}")),
        Expr::Sum(_, _) => Some("sum".into()),
    }
}

fn unsupported_eml_shapes(conjectures: &[Conjecture]) -> Vec<(String, usize)> {
    let mut counts = std::collections::BTreeMap::new();
    for conjecture in conjectures {
        if conjecture.preferred_eml_backend().is_some() {
            continue;
        }
        let reason = eml_support_gap(&conjecture.formula).unwrap_or_else(|| "unknown".into());
        *counts.entry(reason).or_insert(0usize) += 1;
    }

    let mut counts: Vec<_> = counts.into_iter().collect();
    counts.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    counts
}

fn report_family(case: &FamilyCase) -> FamilyReport {
    let started = Instant::now();
    let mut engine = default_engine();
    for observation in &case.observations {
        engine.observe(observation.clone());
    }

    engine.generate_conjectures(5);
    engine.verify_numerical();
    engine.reflect(&PrimitiveSystem::new());

    let at = engine
        .abstract_thought
        .as_ref()
        .expect("abstract thought enabled");
    let macro_candidate_buckets = {
        let mut buckets = std::collections::BTreeSet::new();
        for candidate in &at.dynamic_grammar.candidates {
            buckets.insert(expr_canonical_string(&candidate.pattern));
        }
        buckets.len()
    };

    let eml_backed = engine
        .conjectures
        .iter()
        .filter(|conjecture| conjecture.preferred_eml_backend().is_some())
        .count();
    let eml_verified = engine
        .conjectures
        .iter()
        .filter(|conjecture| is_verified_eml(conjecture))
        .count();
    let constructive_only = engine
        .conjectures
        .iter()
        .filter(|conjecture| {
            matches!(
                conjecture.preferred_eml_backend(),
                Some(PreferredEmlBackend::ConstructiveReal)
            )
        })
        .count();

    FamilyReport {
        name: case.name,
        observations: engine.observations.len(),
        conjectures: engine.conjectures.len(),
        status: count_status(&engine.conjectures),
        eml_backed,
        eml_verified,
        constructive_only,
        macro_candidates: at.dynamic_grammar.candidates.len(),
        macro_candidate_buckets,
        macro_operators: at.dynamic_grammar.operators.len(),
        unsupported_eml_shapes: unsupported_eml_shapes(&engine.conjectures),
        suspicious: suspicious_formulas(&engine.conjectures, &engine.observations),
        elapsed_ms: started.elapsed().as_millis(),
    }
}

fn pct(numerator: usize, denominator: usize) -> String {
    if denominator == 0 {
        "n/a".to_string()
    } else {
        format!("{:.0}%", numerator as f64 * 100.0 / denominator as f64)
    }
}

fn print_summary(reports: &[FamilyReport]) {
    println!("# Symthaea Real Discovery Report");
    println!();
    println!(
        "| family | obs | conjectures | verified | formal | refuted | EML backed | EML verified | macro candidates | macro ops | suspicious | ms |"
    );
    println!(
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    );
    for report in reports {
        let verified = report.status.numeric + report.status.symbolic + report.status.formal;
        println!(
            "| {} | {} | {} | {} ({}) | {} | {} | {} ({}) | {} ({}) | {} / {} | {} | {} | {} |",
            report.name,
            report.observations,
            report.conjectures,
            verified,
            pct(verified, report.conjectures),
            report.status.formal,
            report.status.refuted,
            report.eml_backed,
            pct(report.eml_backed, report.conjectures),
            report.eml_verified,
            pct(report.eml_verified, report.conjectures),
            report.macro_candidate_buckets,
            report.macro_candidates,
            report.macro_operators,
            report.suspicious.len(),
            report.elapsed_ms,
        );
    }
    println!();
}

fn health_warnings(reports: &[FamilyReport]) -> Vec<String> {
    let mut warnings = Vec::new();
    for report in reports {
        let verified = report.status.numeric + report.status.symbolic + report.status.formal;
        let suspicious_rate = if report.conjectures == 0 {
            0.0
        } else {
            report.suspicious.len() as f64 / report.conjectures as f64
        };
        let eml_rate = if report.conjectures == 0 {
            0.0
        } else {
            report.eml_backed as f64 / report.conjectures as f64
        };
        let verified_rate = if report.conjectures == 0 {
            0.0
        } else {
            verified as f64 / report.conjectures as f64
        };

        if suspicious_rate > 0.25 {
            warnings.push(format!(
                "{}: suspicious formula rate is {:.0}% ({}/{})",
                report.name,
                suspicious_rate * 100.0,
                report.suspicious.len(),
                report.conjectures
            ));
        }
        if eml_rate < 0.10 {
            warnings.push(format!(
                "{}: EML backend coverage is low at {:.0}% ({}/{})",
                report.name,
                eml_rate * 100.0,
                report.eml_backed,
                report.conjectures
            ));
        }
        if verified_rate < 0.50 {
            warnings.push(format!(
                "{}: verification rate is low at {:.0}% ({}/{})",
                report.name,
                verified_rate * 100.0,
                verified,
                report.conjectures
            ));
        }
    }
    warnings
}

fn print_health_warnings(warnings: &[String]) {
    println!("## Health Warnings");
    if warnings.is_empty() {
        println!("- none");
    } else {
        for warning in warnings {
            println!("- {warning}");
        }
    }
    println!();
}

fn print_details(reports: &[FamilyReport]) {
    for report in reports {
        println!("## {}", report.name);
        println!(
            "status: proposed={}, numeric={}, symbolic={}, formal={}, refuted={}",
            report.status.proposed,
            report.status.numeric,
            report.status.symbolic,
            report.status.formal,
            report.status.refuted
        );
        println!(
            "EML: backed={} verified={} constructive_only={}",
            report.eml_backed, report.eml_verified, report.constructive_only
        );
        println!(
            "macros: candidates={} canonical_buckets={} promoted_operators={}",
            report.macro_candidates, report.macro_candidate_buckets, report.macro_operators
        );
        if report.unsupported_eml_shapes.is_empty() {
            println!("unsupported EML shapes: none");
        } else {
            println!("unsupported EML shapes:");
            for (reason, count) in report.unsupported_eml_shapes.iter().take(5) {
                println!("- {reason}: {count}");
            }
            if report.unsupported_eml_shapes.len() > 5 {
                println!("- ... {} more", report.unsupported_eml_shapes.len() - 5);
            }
        }
        if report.suspicious.is_empty() {
            println!("suspicious formulas: none");
        } else {
            println!("suspicious formulas:");
            for item in report.suspicious.iter().take(5) {
                let quality = match (item.mean_rel_error, item.normalized_rmse) {
                    (Some(rel), Some(nrmse)) => {
                        format!("; rel={rel:.2e}; nrmse={nrmse:.2e}")
                    }
                    _ => String::new(),
                };
                println!(
                    "- {} => {} [{}; mse={:.2e}{}; confidence={:.2}]",
                    item.source, item.formula, item.reason, item.mse, quality, item.confidence
                );
            }
            if report.suspicious.len() > 5 {
                println!("- ... {} more", report.suspicious.len() - 5);
            }
        }
        println!();
    }
}

fn main() {
    let strict = std::env::args().any(|arg| arg == "--strict");
    let reports: Vec<_> = family_cases().iter().map(report_family).collect();
    let warnings = health_warnings(&reports);
    print_summary(&reports);
    print_health_warnings(&warnings);
    print_details(&reports);

    if strict && !warnings.is_empty() {
        eprintln!(
            "real_discovery_report strict mode failed with {} health warning(s)",
            warnings.len()
        );
        std::process::exit(1);
    }
}
