// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Self-contained HTML report generation with inline CSS and SVG charts.
//!
//! Generates a single HTML file with:
//! - Cognitive profile radar chart (SVG)
//! - Per-domain results tables
//! - Forest plot (SVG)
//! - Composite z-scores
//!
//! No external dependencies — pure `format!()` / `write!()` HTML generation.

// HTML generation intentionally uses write!(...\n) to embed newlines inline with
// multi-line format strings. Converting all to writeln! would reduce readability.
#![allow(clippy::write_with_newline)]

use super::cross_domain_prediction::CrossDomainMatrix;
use super::normative_comparison::NormativeReport;
use super::psychometric_report::PsychometricReport;
use super::reliability_analysis::{ReliabilityBattery, ReliabilityClass};
use super::report::{BenchmarkReport, domain_of, key_metric_for_benchmark};
use super::sat_curves::SatCurve;
use std::collections::BTreeMap;
use std::fmt::Write;

/// Optional supplementary data for a full HTML report.
///
/// SAT curves and reliability data require pre-computation that can't be
/// derived from `BenchmarkReport` alone, so they're provided separately.
#[derive(Default)]
pub struct ReportOptions<'a> {
    /// Include LLM baseline comparison columns.
    pub include_llm: bool,
    /// Pre-computed SAT curves (from `SatBattery::run()`).
    pub sat_curves: Option<&'a [SatCurve]>,
    /// Pre-computed reliability battery (from `ReliabilityBattery`).
    pub reliability: Option<&'a ReliabilityBattery>,
}

/// Generate a self-contained HTML report from benchmark results.
///
/// If `include_llm` is true, includes LLM baseline comparison columns.
pub fn generate_report(report: &BenchmarkReport, include_llm: bool) -> String {
    generate_full_report(
        report,
        &ReportOptions {
            include_llm,
            ..Default::default()
        },
    )
}

/// Generate a full HTML report with optional SAT curves and reliability sections.
pub fn generate_full_report(report: &BenchmarkReport, opts: &ReportOptions<'_>) -> String {
    let mut html = String::with_capacity(32_000);

    write_header(&mut html, report);
    write_profile_radar(&mut html, report);
    write_domain_tables(&mut html, report, opts.include_llm);
    write_forest_plot_svg(&mut html, report);
    write_composite_scores(&mut html, report);
    write_psychometric_summary(&mut html, report);
    write_normative_section(&mut html, report);
    write_cross_domain_section(&mut html, report);
    if let Some(curves) = opts.sat_curves {
        write_sat_curves(&mut html, curves);
    }
    if let Some(battery) = opts.reliability {
        write_reliability_section(&mut html, battery);
    }
    write_footer(&mut html);

    html
}

/// Escape HTML special characters.
fn escape_html(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

fn write_header(html: &mut String, report: &BenchmarkReport) {
    let _ = write!(
        html,
        r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Psych-Bench Report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background: #fafafa; color: #333; }}
h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
h2 {{ color: #2c3e50; margin-top: 30px; }}
h3 {{ color: #34495e; }}
table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
th {{ background: #3498db; color: white; padding: 10px 12px; text-align: left; font-weight: 600; }}
td {{ padding: 8px 12px; border-bottom: 1px solid #e0e0e0; }}
tr:nth-child(even) {{ background: #f5f5f5; }}
tr:hover {{ background: #e8f4fc; }}
.summary {{ background: #ecf0f1; padding: 15px; border-radius: 8px; margin: 15px 0; }}
.green {{ color: #27ae60; font-weight: bold; }}
.yellow {{ color: #f39c12; font-weight: bold; }}
.red {{ color: #e74c3c; font-weight: bold; }}
.chart-container {{ text-align: center; margin: 20px 0; }}
svg {{ max-width: 100%; }}
.footer {{ margin-top: 40px; padding-top: 15px; border-top: 1px solid #ddd; color: #777; font-size: 0.85em; }}
</style>
</head>
<body>
<h1>Psych-Bench Report</h1>
<div class="summary">
<strong>Generated:</strong> {} | <strong>Benchmarks:</strong> {} | <strong>Total metrics:</strong> {}
</div>
"#,
        escape_html(&report.timestamp),
        report.results.len(),
        report
            .results
            .iter()
            .map(|r| r.metrics.len())
            .sum::<usize>(),
    );
}

fn write_profile_radar(html: &mut String, report: &BenchmarkReport) {
    let profile = report.cognitive_profile();
    if profile.is_empty() {
        return;
    }

    let _ = write!(
        html,
        "<h2>Cognitive Profile</h2>\n<div class=\"chart-container\">\n"
    );

    let domains: Vec<(&String, &f64)> = profile.iter().collect();
    let n = domains.len();
    if n < 3 {
        let _ = write!(
            html,
            "<p>Need at least 3 domains for radar chart.</p></div>\n"
        );
        return;
    }

    let cx = 200.0_f64;
    let cy = 200.0_f64;
    let radius = 150.0_f64;
    let size = 420;

    let _ = write!(
        html,
        "<svg width=\"{}\" height=\"{}\" viewBox=\"0 0 {} {}\">\n",
        size, size, size, size,
    );

    // Background circles (grid)
    for level in &[0.25, 0.50, 0.75, 1.0] {
        let r = radius * level;
        let _ = write!(
            html,
            "<circle cx=\"{}\" cy=\"{}\" r=\"{:.1}\" fill=\"none\" stroke=\"#ddd\" stroke-width=\"1\"/>\n",
            cx, cy, r,
        );
    }

    // Axes
    for (i, domain) in domains.iter().enumerate() {
        let angle = std::f64::consts::TAU * i as f64 / n as f64 - std::f64::consts::FRAC_PI_2;
        let x = cx + radius * angle.cos();
        let y = cy + radius * angle.sin();
        let _ = write!(
            html,
            "<line x1=\"{}\" y1=\"{}\" x2=\"{:.1}\" y2=\"{:.1}\" stroke=\"#ccc\" stroke-width=\"1\"/>\n",
            cx, cy, x, y,
        );

        // Labels
        let label_r = radius + 20.0;
        let lx = cx + label_r * angle.cos();
        let ly = cy + label_r * angle.sin();
        let anchor = if angle.cos() > 0.1 {
            "start"
        } else if angle.cos() < -0.1 {
            "end"
        } else {
            "middle"
        };
        let _ = write!(
            html,
            "<text x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"{}\" font-size=\"11\" fill=\"#555\">{}</text>\n",
            lx,
            ly + 4.0,
            anchor,
            escape_html(domain.0),
        );
    }

    // Data polygon
    let mut points = String::new();
    for (i, &(_, &score)) in domains.iter().enumerate() {
        let angle = std::f64::consts::TAU * i as f64 / n as f64 - std::f64::consts::FRAC_PI_2;
        let r = radius * score.clamp(0.0, 1.0);
        let x = cx + r * angle.cos();
        let y = cy + r * angle.sin();
        if !points.is_empty() {
            points.push(' ');
        }
        let _ = write!(points, "{:.1},{:.1}", x, y);
    }
    let _ = write!(
        html,
        "<polygon points=\"{}\" fill=\"rgba(52,152,219,0.25)\" stroke=\"#3498db\" stroke-width=\"2\"/>\n",
        points,
    );

    // Data points
    for (i, &(_, &score)) in domains.iter().enumerate() {
        let angle = std::f64::consts::TAU * i as f64 / n as f64 - std::f64::consts::FRAC_PI_2;
        let r = radius * score.clamp(0.0, 1.0);
        let x = cx + r * angle.cos();
        let y = cy + r * angle.sin();
        let _ = write!(
            html,
            "<circle cx=\"{:.1}\" cy=\"{:.1}\" r=\"4\" fill=\"#3498db\"/>\n",
            x, y,
        );
    }

    let _ = writeln!(html, "</svg>\n</div>");
}

fn write_domain_tables(html: &mut String, report: &BenchmarkReport, include_llm: bool) {
    use crate::harness::baselines::BaselineCollection;
    let bl = BaselineCollection::all();

    let _ = writeln!(html, "<h2>Results by Domain</h2>");

    // Group by domain
    let mut domains: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (i, result) in report.results.iter().enumerate() {
        let domain = domain_of(&result.benchmark).to_string();
        domains.entry(domain).or_default().push(i);
    }

    for (domain, indices) in &domains {
        let _ = write!(
            html,
            "<h3>{}</h3>\n<table>\n<tr><th>Benchmark</th><th>Key Metric</th><th>Agent</th><th>Human</th><th>% Human</th><th>d</th><th>z</th><th>95% CI</th>",
            escape_html(domain)
        );
        if include_llm {
            let _ = write!(html, "<th>LLM (GPT-4)</th><th>LLM %</th>");
        }
        let _ = writeln!(html, "</tr>");

        for &i in indices {
            let result = &report.results[i];
            let key = key_metric_for_benchmark(&result.benchmark);
            let metric = match result.metrics.get(key) {
                Some(m) => m,
                None => continue,
            };

            let comparisons = report.find_comparisons(result, &bl);
            let comp = comparisons.iter().find(|(k, _)| k == key);

            let bench_name = result
                .benchmark
                .split("::")
                .last()
                .unwrap_or(&result.benchmark);
            let (human_str, pct_str) = comp
                .map(|(_, c)| {
                    (
                        format!("{:.3}", c.human_value),
                        format!("{:.1}%", c.ratio * 100.0),
                    )
                })
                .unwrap_or_else(|| ("\u{2014}".to_string(), "\u{2014}".to_string()));

            let d_val = comp.and_then(|(_, c)| c.effect_size).unwrap_or(0.0);
            let d_str = comp
                .and_then(|(_, c)| c.effect_size)
                .map(|d| format!("{:.2}", d))
                .unwrap_or_else(|| "\u{2014}".to_string());
            let z_str = comp
                .and_then(|(_, c)| c.z_score)
                .map(|z| format!("{:+.2}", z))
                .unwrap_or_else(|| "\u{2014}".to_string());

            let d_class = effect_color_class(d_val);

            let _ = write!(
                html,
                "<tr><td>{}</td><td>{}</td><td>{:.3}</td><td>{}</td><td>{}</td><td class=\"{}\">{}</td><td>{}</td><td>[{:.3}, {:.3}]</td>",
                escape_html(bench_name),
                escape_html(key),
                metric.mean,
                human_str,
                pct_str,
                d_class,
                d_str,
                z_str,
                metric.ci_lower,
                metric.ci_upper,
            );

            if include_llm {
                let llm_comps = report.find_llm_comparisons(result, &bl);
                let llm_comp = llm_comps.iter().find(|(k, _)| k == key);
                let (llm_val, llm_pct) = llm_comp
                    .map(|(_, c)| {
                        (
                            format!("{:.3}", c.human_value),
                            format!("{:.1}%", c.ratio * 100.0),
                        )
                    })
                    .unwrap_or_else(|| ("\u{2014}".to_string(), "\u{2014}".to_string()));
                let _ = write!(html, "<td>{}</td><td>{}</td>", llm_val, llm_pct);
            }

            let _ = writeln!(html, "</tr>");
        }
        let _ = writeln!(html, "</table>");
    }
}

fn write_forest_plot_svg(html: &mut String, report: &BenchmarkReport) {
    let rows = report.forest_plot_data();
    if rows.is_empty() {
        return;
    }

    let _ = write!(
        html,
        "<h2>Effect Size Forest Plot</h2>\n<div class=\"chart-container\">\n"
    );

    let row_height = 28;
    let left_margin = 200;
    let plot_width = 500;
    let right_margin = 80;
    let total_width = left_margin + plot_width + right_margin;
    let top_margin = 40;
    let total_height = top_margin + rows.len() * row_height + 20;

    let _ = write!(
        html,
        "<svg width=\"{}\" height=\"{}\" viewBox=\"0 0 {} {}\">\n",
        total_width, total_height, total_width, total_height,
    );

    // Scale: d ∈ [-2, +2] maps to plot_width
    let d_to_x = |d: f64| -> f64 { left_margin as f64 + ((d + 2.0) / 4.0) * plot_width as f64 };

    // Axis
    let axis_y = top_margin as f64 - 5.0;
    let _ = write!(
        html,
        "<line x1=\"{}\" y1=\"{:.0}\" x2=\"{}\" y2=\"{:.0}\" stroke=\"#999\" stroke-width=\"1\"/>\n",
        left_margin,
        axis_y,
        left_margin + plot_width,
        axis_y,
    );

    // Tick marks and labels
    for &d_val in &[-2.0, -1.0, 0.0, 1.0, 2.0] {
        let x = d_to_x(d_val);
        let _ = write!(
            html,
            "<line x1=\"{:.0}\" y1=\"{:.0}\" x2=\"{:.0}\" y2=\"{}\" stroke=\"#ccc\" stroke-width=\"1\" stroke-dasharray=\"2,2\"/>\n",
            x,
            axis_y,
            x,
            total_height - 10,
        );
        let _ = write!(
            html,
            "<text x=\"{:.0}\" y=\"{:.0}\" text-anchor=\"middle\" font-size=\"10\" fill=\"#666\">{:+.0}</text>\n",
            x,
            axis_y - 5.0,
            d_val,
        );
    }

    // Zero line (bold)
    let zero_x = d_to_x(0.0);
    let _ = write!(
        html,
        "<line x1=\"{:.0}\" y1=\"{:.0}\" x2=\"{:.0}\" y2=\"{}\" stroke=\"#333\" stroke-width=\"1.5\"/>\n",
        zero_x,
        axis_y,
        zero_x,
        total_height - 10,
    );

    // Data rows
    for (i, row) in rows.iter().enumerate() {
        let y = top_margin + i * row_height + row_height / 2;
        let yf = y as f64;
        let d = row.cohens_d.unwrap_or(0.0);
        let d_clamped = d.clamp(-2.0, 2.0);

        // Label
        let _ = write!(
            html,
            "<text x=\"{}\" y=\"{:.0}\" text-anchor=\"end\" font-size=\"11\" fill=\"#333\" dominant-baseline=\"middle\">{}</text>\n",
            left_margin - 10,
            yf,
            escape_html(&row.benchmark),
        );

        // CI whisker
        let ci_lo = ((row.ci_lower / row.human_mean - 1.0)
            * row.cohens_d.unwrap_or(1.0).abs().max(0.5))
        .clamp(-2.0, 2.0);
        let ci_hi = ((row.ci_upper / row.human_mean - 1.0)
            * row.cohens_d.unwrap_or(1.0).abs().max(0.5))
        .clamp(-2.0, 2.0);
        let _ = write!(
            html,
            "<line x1=\"{:.0}\" y1=\"{:.0}\" x2=\"{:.0}\" y2=\"{:.0}\" stroke=\"#666\" stroke-width=\"1\"/>\n",
            d_to_x(ci_lo.min(d_clamped)),
            yf,
            d_to_x(ci_hi.max(d_clamped)),
            yf,
        );

        // Point
        let color = effect_color(d);
        let x = d_to_x(d_clamped);
        let _ = write!(
            html,
            "<circle cx=\"{:.0}\" cy=\"{:.0}\" r=\"5\" fill=\"{}\" stroke=\"white\" stroke-width=\"1\"/>\n",
            x, yf, color,
        );

        // d value label
        let _ = write!(
            html,
            "<text x=\"{}\" y=\"{:.0}\" font-size=\"10\" fill=\"#666\" dominant-baseline=\"middle\">{:+.2}</text>\n",
            left_margin + plot_width + 10,
            yf,
            d,
        );
    }

    let _ = writeln!(html, "</svg>\n</div>");
}

fn write_composite_scores(html: &mut String, report: &BenchmarkReport) {
    let composites = report.composite_scores();
    if composites.is_empty() {
        return;
    }

    let _ = writeln!(
        html,
        "<h2>Composite Scores</h2>\n<table>\n<tr><th>Domain</th><th>Mean z</th><th>n</th><th>Benchmarks</th></tr>"
    );

    for (domain, cs) in &composites {
        let z_class = if cs.mean_z > 0.5 {
            "green"
        } else if cs.mean_z > -0.5 {
            ""
        } else {
            "red"
        };
        let _ = write!(
            html,
            "<tr><td>{}</td><td class=\"{}\">{:+.2}</td><td>{}</td><td>{}</td></tr>\n",
            escape_html(domain),
            z_class,
            cs.mean_z,
            cs.n_benchmarks,
            escape_html(&cs.benchmarks.join(", ")),
        );
    }

    let _ = writeln!(html, "</table>");
}

/// Write a citations/provenance table section into the HTML report.
///
/// Takes a slice of `(name, provenance)` tuples. Call this before `write_footer`.
pub fn write_provenance_section(
    html: &mut String,
    entries: &[(&str, crate::harness::BenchmarkProvenance)],
) {
    if entries.is_empty() {
        return;
    }
    let _ = write!(
        html,
        r#"<h2>Citations &amp; Provenance</h2>
<table>
<tr><th>Benchmark</th><th>Paradigm</th><th>Citation</th><th>Year</th><th>DOI</th></tr>
"#
    );
    for (name, p) in entries {
        let doi_cell = match p.doi {
            Some(d) => format!(
                "<a href=\"https://doi.org/{}\" target=\"_blank\">{}</a>",
                escape_html(d),
                escape_html(d)
            ),
            None => "\u{2014}".to_string(),
        };
        let _ = write!(
            html,
            "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>\n",
            escape_html(name),
            escape_html(p.paradigm),
            escape_html(p.citation),
            p.year,
            doi_cell,
        );
    }
    html.push_str("</table>\n");
}

/// Write SAT curve SVG charts into the HTML report.
///
/// Each benchmark gets a line chart: X = time_pressure, Y = accuracy.
/// Dashed fitted curve overlay. Color by domain.
pub fn write_sat_curves(html: &mut String, curves: &[SatCurve]) {
    if curves.is_empty() {
        return;
    }

    let _ = write!(
        html,
        "<h2>Speed-Accuracy Tradeoff Curves</h2>\n<div class=\"chart-container\">\n"
    );

    let chart_w = 600;
    let chart_h = 350;
    let margin_l = 60;
    let margin_r = 20;
    let margin_t = 30;
    let margin_b = 50;
    let plot_w = chart_w - margin_l - margin_r;
    let plot_h = chart_h - margin_t - margin_b;

    let _ = write!(
        html,
        "<svg width=\"{}\" height=\"{}\" viewBox=\"0 0 {} {}\">\n",
        chart_w, chart_h, chart_w, chart_h,
    );

    // Background
    let _ = write!(
        html,
        "<rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" fill=\"#fafafa\" stroke=\"#ccc\" stroke-width=\"1\"/>\n",
        margin_l, margin_t, plot_w, plot_h,
    );

    // X axis labels (time pressure 0.0 to 1.0)
    for i in 0..=5 {
        let tp = i as f64 / 5.0;
        let x = margin_l as f64 + tp * plot_w as f64;
        let y_base = (margin_t + plot_h) as f64;
        let _ = write!(
            html,
            "<line x1=\"{:.0}\" y1=\"{:.0}\" x2=\"{:.0}\" y2=\"{:.0}\" stroke=\"#ddd\" stroke-width=\"1\"/>\n",
            x, margin_t as f64, x, y_base,
        );
        let _ = write!(
            html,
            "<text x=\"{:.0}\" y=\"{:.0}\" text-anchor=\"middle\" font-size=\"10\" fill=\"#666\">{:.1}</text>\n",
            x,
            y_base + 15.0,
            tp,
        );
    }
    // X axis title
    let _ = write!(
        html,
        "<text x=\"{:.0}\" y=\"{:.0}\" text-anchor=\"middle\" font-size=\"12\" fill=\"#333\">Time Pressure</text>\n",
        margin_l as f64 + plot_w as f64 / 2.0,
        (chart_h - 5) as f64,
    );

    // Y axis labels (accuracy 0.0 to 1.0)
    for i in 0..=5 {
        let acc = i as f64 / 5.0;
        let y = margin_t as f64 + (1.0 - acc) * plot_h as f64;
        let _ = write!(
            html,
            "<line x1=\"{}\" y1=\"{:.0}\" x2=\"{}\" y2=\"{:.0}\" stroke=\"#ddd\" stroke-width=\"1\"/>\n",
            margin_l,
            y,
            margin_l + plot_w,
            y,
        );
        let _ = write!(
            html,
            "<text x=\"{:.0}\" y=\"{:.0}\" text-anchor=\"end\" font-size=\"10\" fill=\"#666\">{:.1}</text>\n",
            margin_l as f64 - 5.0,
            y + 3.0,
            acc,
        );
    }
    // Y axis title
    let _ = write!(
        html,
        "<text x=\"15\" y=\"{:.0}\" text-anchor=\"middle\" font-size=\"12\" fill=\"#333\" transform=\"rotate(-90, 15, {:.0})\">Accuracy</text>\n",
        margin_t as f64 + plot_h as f64 / 2.0,
        margin_t as f64 + plot_h as f64 / 2.0,
    );

    let domain_colors = [
        "#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c", "#e67e22", "#34495e",
        "#16a085", "#c0392b", "#2980b9", "#8e44ad", "#27ae60", "#d35400", "#7f8c8d",
    ];

    // Plot each curve
    for (ci, curve) in curves.iter().enumerate() {
        let color = domain_colors[ci % domain_colors.len()];
        let short_name = curve
            .benchmark
            .split("::")
            .last()
            .unwrap_or(&curve.benchmark);

        // Data points + line
        let mut path_d = String::new();
        let mut sorted_points = curve.points.clone();
        sorted_points.sort_by(|a, b| {
            a.time_pressure
                .partial_cmp(&b.time_pressure)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for (pi, p) in sorted_points.iter().enumerate() {
            let x = margin_l as f64 + p.time_pressure * plot_w as f64;
            let y = margin_t as f64 + (1.0 - p.accuracy.clamp(0.0, 1.0)) * plot_h as f64;
            let cmd = if pi == 0 { "M" } else { "L" };
            let _ = write!(path_d, "{}{:.1},{:.1} ", cmd, x, y);

            // Data point circle
            let _ = write!(
                html,
                "<circle cx=\"{:.1}\" cy=\"{:.1}\" r=\"3\" fill=\"{}\" stroke=\"white\" stroke-width=\"1\"/>\n",
                x, y, color,
            );
        }
        // Solid line for data
        let _ = write!(
            html,
            "<path d=\"{}\" fill=\"none\" stroke=\"{}\" stroke-width=\"2\" opacity=\"0.8\"/>\n",
            path_d.trim(),
            color,
        );

        // Dashed fitted curve (20 segments)
        let mut fit_d = String::new();
        for si in 0..=20 {
            let tp = si as f64 / 20.0;
            let t = 1.0 - tp;
            let predicted = if t > curve.fit.intercept {
                curve.fit.asymptote * (1.0 - (-curve.fit.rate * (t - curve.fit.intercept)).exp())
            } else {
                0.0
            };
            let x = margin_l as f64 + tp * plot_w as f64;
            let y = margin_t as f64 + (1.0 - predicted.clamp(0.0, 1.0)) * plot_h as f64;
            let cmd = if si == 0 { "M" } else { "L" };
            let _ = write!(fit_d, "{}{:.1},{:.1} ", cmd, x, y);
        }
        let _ = write!(
            html,
            "<path d=\"{}\" fill=\"none\" stroke=\"{}\" stroke-width=\"1.5\" stroke-dasharray=\"4,3\" opacity=\"0.6\"/>\n",
            fit_d.trim(),
            color,
        );

        // Legend entry
        let legend_y = margin_t as f64 + 12.0 + ci as f64 * 14.0;
        if legend_y < (margin_t + plot_h - 10) as f64 {
            let legend_x = margin_l as f64 + 10.0;
            let _ = write!(
                html,
                "<line x1=\"{:.0}\" y1=\"{:.0}\" x2=\"{:.0}\" y2=\"{:.0}\" stroke=\"{}\" stroke-width=\"2\"/>\n",
                legend_x,
                legend_y,
                legend_x + 15.0,
                legend_y,
                color,
            );
            let _ = write!(
                html,
                "<text x=\"{:.0}\" y=\"{:.0}\" font-size=\"9\" fill=\"#333\">{} (R\u{00b2}={:.2})</text>\n",
                legend_x + 18.0,
                legend_y + 3.0,
                escape_html(&short_name[..short_name.len().min(20)]),
                curve.fit.r_squared,
            );
        }
    }

    let _ = writeln!(html, "</svg>\n</div>");
}

/// Write a test-retest reliability table section into the HTML report.
///
/// Renders ICC, Pearson r, SEM, practice effect direction, and reliability
/// classification for each benchmark. Call this explicitly (like `write_sat_curves`)
/// since it requires pre-computed `ReliabilityBattery` results.
pub fn write_reliability_section(html: &mut String, battery: &ReliabilityBattery) {
    if battery.results.is_empty() {
        return;
    }

    let _ = writeln!(html, "<h2>Test-Retest Reliability</h2>");

    // Summary stats
    let mean_icc: f64 =
        battery.results.iter().map(|r| r.icc).sum::<f64>() / battery.results.len() as f64;
    let excellent = battery
        .results
        .iter()
        .filter(|r| r.reliability_class == ReliabilityClass::Excellent)
        .count();
    let good = battery
        .results
        .iter()
        .filter(|r| r.reliability_class == ReliabilityClass::Good)
        .count();

    let _ = write!(
        html,
        "<div class=\"summary\"><strong>Mean ICC:</strong> {:.3} | <strong>Excellent:</strong> {} | <strong>Good:</strong> {} | <strong>Total:</strong> {}</div>\n",
        mean_icc,
        excellent,
        good,
        battery.results.len(),
    );

    let _ = write!(
        html,
        "<table>\n<tr><th>Benchmark</th><th>Metric</th><th>ICC</th><th>Pearson r</th><th>SEM</th><th>Practice</th><th>Class</th></tr>\n"
    );

    for r in &battery.results {
        let icc_class = match r.reliability_class {
            ReliabilityClass::Excellent => "green",
            ReliabilityClass::Good => "green",
            ReliabilityClass::Moderate => "yellow",
            ReliabilityClass::Poor => "red",
        };
        let practice_str = format!("{:+.1}%", r.practice.change_pct,);
        let _ = write!(
            html,
            "<tr><td>{}</td><td>{}</td><td class=\"{}\">{:.3}</td><td>{:.3}</td><td>{:.3}</td><td>{}</td><td>{}</td></tr>\n",
            escape_html(&r.benchmark),
            escape_html(&r.metric),
            icc_class,
            r.icc,
            r.pearson_r,
            r.sem,
            practice_str,
            r.reliability_class.label(),
        );
    }

    let _ = writeln!(html, "</table>");
}

fn write_psychometric_summary(html: &mut String, report: &BenchmarkReport) {
    if report.results.is_empty() {
        return;
    }

    let pr = PsychometricReport::from_report(report);

    let _ = write!(
        html,
        "<h2>Psychometric Summary</h2>\n<div class=\"summary\">\n"
    );
    let _ = write!(
        html,
        "<strong>Overall score:</strong> {:.1}% | ",
        pr.summary.overall_score * 100.0
    );
    let _ = write!(
        html,
        "<strong>Strongest:</strong> {} | ",
        escape_html(&pr.summary.strongest_domain)
    );
    let _ = write!(
        html,
        "<strong>Weakest:</strong> {}",
        escape_html(&pr.summary.weakest_domain)
    );
    let _ = writeln!(html, "</div>");

    // Benchmark details table
    let _ = write!(
        html,
        "<table>\n<tr><th>Benchmark</th><th>Key Metric</th><th>Value</th><th>Trials</th><th>Time (ms)</th></tr>\n"
    );
    for d in &pr.benchmark_details {
        let _ = write!(
            html,
            "<tr><td>{}</td><td>{}</td><td>{:.3}</td><td>{}</td><td>{}</td></tr>\n",
            escape_html(&d.benchmark),
            escape_html(&d.key_metric),
            d.key_value,
            d.n_trials,
            d.elapsed_ms,
        );
    }
    let _ = writeln!(html, "</table>");
}

fn write_normative_section(html: &mut String, report: &BenchmarkReport) {
    if report.results.is_empty() {
        return;
    }

    let normative = NormativeReport::from_report(report);
    if normative.scores.is_empty() {
        return;
    }

    let _ = write!(
        html,
        "<h2>Normative Comparison</h2>\n<div class=\"summary\">\n"
    );
    let _ = write!(
        html,
        "<strong>Mean z:</strong> {:+.2} | <strong>Percentile:</strong> {:.1}%",
        normative.overall_mean_z, normative.overall_percentile,
    );
    let _ = writeln!(html, "</div>");

    let _ = write!(
        html,
        "<table>\n<tr><th>Benchmark</th><th>Metric</th><th>Agent</th><th>Human</th><th>z</th><th>Percentile</th><th>Interpretation</th></tr>\n"
    );
    for s in &normative.scores {
        let z_class = if s.z_score > 1.0 {
            "green"
        } else if s.z_score < -1.0 {
            "red"
        } else {
            ""
        };
        let _ = write!(
            html,
            "<tr><td>{}</td><td>{}</td><td>{:.3}</td><td>{:.3} &plusmn; {:.3}</td><td class=\"{}\">{:+.2}</td><td>{:.1}%</td><td>{}</td></tr>\n",
            escape_html(&s.benchmark),
            escape_html(&s.metric),
            s.agent_value,
            s.human_mean,
            s.human_sd,
            z_class,
            s.z_score,
            s.percentile,
            escape_html(&s.interpretation),
        );
    }
    let _ = writeln!(html, "</table>");
}

fn write_cross_domain_section(html: &mut String, report: &BenchmarkReport) {
    if report.results.is_empty() {
        return;
    }

    let matrix = CrossDomainMatrix::from_single_report(report);
    if matrix.correlations.is_empty() {
        return;
    }

    let _ = writeln!(html, "<h2>Cross-Domain Correlations</h2>");
    let _ = write!(
        html,
        "<table>\n<tr><th>Domain A</th><th>Domain B</th><th>r</th><th>N</th><th>p</th><th>Mechanism</th></tr>\n"
    );
    for c in &matrix.correlations {
        let mech_str = match c.shared_mechanism {
            super::cross_domain_prediction::SharedMechanism::Strong => "Strong",
            super::cross_domain_prediction::SharedMechanism::Moderate => "Moderate",
            super::cross_domain_prediction::SharedMechanism::Weak => "Weak",
            super::cross_domain_prediction::SharedMechanism::Independent => "Independent",
        };
        let _ = write!(
            html,
            "<tr><td>{}</td><td>{}</td><td>{:.3}</td><td>{}</td><td>{:.3}</td><td>{}</td></tr>\n",
            escape_html(&c.domain_a),
            escape_html(&c.domain_b),
            c.r,
            c.n,
            c.p_value,
            mech_str,
        );
    }
    let _ = writeln!(html, "</table>");

    // Shared mechanisms
    let shared = matrix.shared_mechanisms();
    if !shared.is_empty() {
        let _ = writeln!(html, "<h3>Shared Mechanisms (Moderate+)</h3>\n<ul>");
        for c in &shared {
            let _ = write!(
                html,
                "<li><strong>{} &times; {}</strong>: r={:.3}</li>\n",
                escape_html(&c.domain_a),
                escape_html(&c.domain_b),
                c.r,
            );
        }
        let _ = writeln!(html, "</ul>");
    }
}

fn write_footer(html: &mut String) {
    let _ = write!(
        html,
        r#"<div class="footer">
Generated by <strong>symthaea-psych-bench</strong> | Luminous Dynamics
</div>
</body>
</html>"#
    );
}

/// Return CSS color class for effect size magnitude.
fn effect_color_class(d: f64) -> &'static str {
    let abs_d = d.abs();
    if abs_d < 0.2 {
        "green"
    } else if abs_d < 0.8 {
        "yellow"
    } else {
        "red"
    }
}

/// Return SVG fill color for effect size magnitude.
fn effect_color(d: f64) -> &'static str {
    let abs_d = d.abs();
    if abs_d < 0.2 {
        "#27ae60"
    } else if abs_d < 0.8 {
        "#f39c12"
    } else {
        "#e74c3c"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::report::{BenchmarkResult, MetricValue};

    fn sample_report() -> BenchmarkReport {
        let mut report = BenchmarkReport::new();
        let mut r1 = BenchmarkResult::new("WorM::N-back", None);
        r1.insert(
            "nback_2::accuracy",
            MetricValue::from_samples(&[0.85, 0.90, 0.88]),
        );
        report.add(r1);
        let mut r2 = BenchmarkResult::new("Executive::Stroop", None);
        r2.insert(
            "stroop_effect",
            MetricValue::from_samples(&[0.10, 0.12, 0.11]),
        );
        report.add(r2);
        let mut r3 = BenchmarkResult::new("CogBench::BART", None);
        r3.insert("average_pumps", MetricValue::from_samples(&[28.0, 32.0]));
        report.add(r3);
        report
    }

    #[test]
    fn test_html_valid_structure() {
        let report = sample_report();
        let html = generate_report(&report, false);
        assert!(html.contains("<!DOCTYPE html>"), "missing DOCTYPE");
        assert!(html.contains("<table>"), "missing table");
        assert!(html.contains("<svg"), "missing SVG");
        assert!(html.contains("</html>"), "missing closing html tag");
    }

    #[test]
    fn test_html_contains_benchmarks() {
        let report = sample_report();
        let html = generate_report(&report, false);
        assert!(html.contains("N-back"), "missing N-back benchmark name");
        assert!(html.contains("Stroop"), "missing Stroop benchmark name");
        assert!(html.contains("BART"), "missing BART benchmark name");
    }

    #[test]
    fn test_html_forest_svg() {
        let report = sample_report();
        let html = generate_report(&report, false);
        // Forest plot should contain circle elements for data points
        assert!(
            html.contains("<circle"),
            "SVG should contain circle elements"
        );
    }

    #[test]
    fn test_html_empty_report() {
        let report = BenchmarkReport::new();
        let html = generate_report(&report, false);
        // Should not panic, should still have valid structure
        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("</html>"));
    }

    #[test]
    fn test_html_special_chars_escaped() {
        assert_eq!(escape_html("<script>"), "&lt;script&gt;");
        assert_eq!(escape_html("a & b"), "a &amp; b");
        assert_eq!(escape_html("\"hello\""), "&quot;hello&quot;");
    }

    #[test]
    fn test_html_sat_curves_svg() {
        use crate::harness::sat_curves::{SatCurve, SatFit, SatPoint};
        let curve = SatCurve {
            benchmark: "Test::Benchmark".to_string(),
            points: vec![
                SatPoint {
                    time_pressure: 0.0,
                    accuracy: 0.95,
                    mean_rt: 10.0,
                },
                SatPoint {
                    time_pressure: 0.5,
                    accuracy: 0.75,
                    mean_rt: 5.0,
                },
                SatPoint {
                    time_pressure: 1.0,
                    accuracy: 0.50,
                    mean_rt: 2.0,
                },
            ],
            fit: SatFit {
                asymptote: 0.95,
                rate: 1.5,
                intercept: 0.0,
                r_squared: 0.92,
            },
            human_like: true,
        };
        let mut html = String::new();
        write_sat_curves(&mut html, &[curve]);
        assert!(html.contains("<svg"), "SAT SVG should contain svg element");
        assert!(html.contains("Speed-Accuracy"), "should have SAT heading");
        assert!(html.contains("circle"), "should have data point circles");
        assert!(
            html.contains("stroke-dasharray"),
            "should have dashed fitted curve"
        );
    }

    #[test]
    fn test_html_sat_curves_empty() {
        let mut html = String::new();
        write_sat_curves(&mut html, &[]);
        assert!(html.is_empty(), "empty curves should produce no output");
    }

    #[test]
    fn test_html_psychometric_summary() {
        let report = sample_report();
        let mut html = String::new();
        write_psychometric_summary(&mut html, &report);
        assert!(
            html.contains("Psychometric Summary"),
            "missing psychometric summary heading"
        );
        assert!(
            html.contains("Overall score"),
            "missing overall score label"
        );
    }

    #[test]
    fn test_html_normative_section() {
        let mut report = BenchmarkReport::new();
        let mut stroop = BenchmarkResult::new("Executive::Stroop", None);
        stroop.insert(
            "incongruent_accuracy",
            MetricValue::from_samples(&[0.85, 0.87, 0.83]),
        );
        report.add(stroop);
        let mut html = String::new();
        write_normative_section(&mut html, &report);
        // May or may not produce output depending on baselines having SD
        // Just verify no panic
        let _ = html;
    }

    #[test]
    fn test_html_cross_domain_section() {
        let report = BenchmarkReport::new();
        let mut html = String::new();
        write_cross_domain_section(&mut html, &report);
        // Empty report: no output, no panic
        assert!(html.is_empty());
    }

    #[test]
    fn test_html_new_sections_in_full_report() {
        let report = sample_report();
        let html = generate_report(&report, false);
        // Full report should include new sections
        assert!(html.contains("Psychometric Summary"));
        assert!(html.contains("</html>"));
    }

    #[test]
    fn test_html_reliability_section() {
        use crate::harness::reliability_analysis::{
            PracticeDirection, PracticeEffect, ReliabilityClass, TestRetestResult,
        };
        let battery = ReliabilityBattery {
            results: vec![
                TestRetestResult {
                    benchmark: "Executive::Stroop".to_string(),
                    metric: "stroop_effect".to_string(),
                    icc: 0.85,
                    pearson_r: 0.87,
                    sem: 0.02,
                    practice: PracticeEffect::compute(
                        "Executive::Stroop",
                        "stroop_effect",
                        0.12,
                        0.10,
                    ),
                    reliability_class: ReliabilityClass::Good,
                },
                TestRetestResult {
                    benchmark: "WorM::N-back".to_string(),
                    metric: "nback_2::accuracy".to_string(),
                    icc: 0.92,
                    pearson_r: 0.93,
                    sem: 0.01,
                    practice: PracticeEffect::compute(
                        "WorM::N-back",
                        "nback_2::accuracy",
                        0.85,
                        0.88,
                    ),
                    reliability_class: ReliabilityClass::Excellent,
                },
            ],
        };
        let mut html = String::new();
        write_reliability_section(&mut html, &battery);
        assert!(
            html.contains("Test-Retest Reliability"),
            "missing reliability heading"
        );
        assert!(html.contains("ICC"), "missing ICC column header");
        assert!(html.contains("Stroop"), "missing Stroop benchmark");
        assert!(html.contains("N-back"), "missing N-back benchmark");
        assert!(html.contains("Excellent"), "missing Excellent class label");
        assert!(html.contains("Good"), "missing Good class label");
        assert!(html.contains("Mean ICC"), "missing mean ICC summary");
    }

    #[test]
    fn test_html_reliability_empty() {
        let battery = ReliabilityBattery { results: vec![] };
        let mut html = String::new();
        write_reliability_section(&mut html, &battery);
        assert!(html.is_empty(), "empty battery should produce no output");
    }
}
