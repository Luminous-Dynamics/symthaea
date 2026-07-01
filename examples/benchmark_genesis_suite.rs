// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Unified Genesis Benchmark Suite — HTML + JSON Report
//!
//! Runs all domain predictors, collects results, and generates an
//! HTML report with domain registry, O(1) cost proof, prediction drift,
//! cross-domain coherence matrix, and timescale span summary.

use symthaea_core::hdc::unified_hv::ContinuousHV;
use symthaea_core::temporal::TemporalPredictor;

fn main() {
    println!("=== Genesis Benchmark Suite ===\n");

    let predictors: Vec<Box<dyn TemporalPredictor>> = vec![
        Box::new(symthaea_physics::PlasmaMultiScalePredictor::new()),
        Box::new(symthaea_cell_foundry::HydrologicalPredictor::new()),
        Box::new(symthaea_materials::MaterialAgingModel::new()),
        Box::new(symthaea_nuclear_forensics::IsotopeDecayModel::new()),
        Box::new(symthaea_physics::ScalePredictor::new(
            symthaea_physics::PhysicalScale::Molecular,
        )),
        Box::new(symthaea_physics::GridPredictor::new()),
        Box::new(symthaea_physics::FissionPredictor::new()),
        Box::new(symthaea_physics::AcceleratorPredictor::new()),
        Box::new(symthaea_physics::ThreatPredictor::new()),
        Box::new(symthaea_physics::DatacenterPredictor::new()),
        Box::new(symthaea_cell_foundry::ExperimentPredictor::new()),
        Box::new(symthaea_materials::StrategicPredictor::new()),
        Box::new(symthaea_materials::MiningPredictor::new()),
        Box::new(symthaea_nuclear_forensics::SafeguardsPredictor::new()),
    ];

    let seed_state = ContinuousHV::random(16_384, 0xBEEF);
    let iters = 500;
    let n = predictors.len();

    // ── 1. Domain metadata ──────────────────────────────────────────────
    let names: Vec<&str> = predictors.iter().map(|p| p.domain()).collect();
    let taus: Vec<f32> = predictors.iter().map(|p| p.tau_base()).collect();
    let horizon_counts: Vec<usize> = predictors
        .iter()
        .map(|p| p.default_horizons().len())
        .collect();
    let hlabels: Vec<Vec<&str>> = predictors
        .iter()
        .map(|p| p.horizon_labels().iter().copied().collect())
        .collect();
    println!("Domains: {}", n);

    // ── 2. O(1) cost proof ──────────────────────────────────────────────
    let mut short_us: Vec<f64> = Vec::new();
    let mut long_us: Vec<f64> = Vec::new();
    let mut ratios: Vec<f64> = Vec::new();
    let mut cost_pass: Vec<bool> = Vec::new();
    let mut all_o1 = true;

    for p in &predictors {
        let short_dt = p.tau_base();
        let long_dt = p.tau_base() * 1_000_000.0;

        let t1 = std::time::Instant::now();
        for _ in 0..iters {
            let _ = p.predict_at(&seed_state, short_dt);
        }
        let s = t1.elapsed().as_micros() as f64 / iters as f64;

        let t2 = std::time::Instant::now();
        for _ in 0..iters {
            let _ = p.predict_at(&seed_state, long_dt);
        }
        let l = t2.elapsed().as_micros() as f64 / iters as f64;

        let r = l / s.max(0.001);
        let ok = r < 5.0 && r > 0.1;
        if !ok {
            all_o1 = false;
        }

        short_us.push(s);
        long_us.push(l);
        ratios.push(r);
        cost_pass.push(ok);
    }

    println!(
        "O(1) cost proof: {}",
        if all_o1 { "ALL PASS" } else { "SOME FAIL" }
    );

    // ── 3. Prediction drift (1 year forward) ────────────────────────────
    let one_year = 31_536_000.0_f32;
    let mut drifts: Vec<f32> = Vec::new();
    let mut pred_hvs: Vec<ContinuousHV> = Vec::new();

    for p in &predictors {
        let predicted = p.predict_at(&seed_state, one_year);
        drifts.push(1.0 - seed_state.similarity(&predicted));
        pred_hvs.push(predicted);
    }

    // ── 4. Cross-domain coherence matrix ────────────────────────────────
    let mut sim_matrix: Vec<Vec<f32>> = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            sim_matrix[i][j] = pred_hvs[i].similarity(&pred_hvs[j]);
        }
    }

    // ── 5. Timescale span ───────────────────────────────────────────────
    let tau_min = taus.iter().copied().fold(f32::INFINITY, f32::min);
    let tau_max = taus.iter().copied().fold(0.0_f32, f32::max);
    let orders = (tau_max / tau_min).log10();
    println!("Timescale span: {:.1} orders of magnitude\n", orders);

    // ── Generate reports ────────────────────────────────────────────────
    std::fs::create_dir_all("benchmark_output").ok();

    // ---- HTML ----
    let mut html = String::with_capacity(32_000);
    html.push_str("<!DOCTYPE html><html><head><meta charset='utf-8'>\n");
    html.push_str("<title>Genesis Benchmark Suite Report</title>\n");
    html.push_str("<style>\n");
    html.push_str(
        "body{font-family:system-ui,sans-serif;margin:2em;background:#fafafa;color:#222}\n",
    );
    html.push_str(
        "h1{color:#2c3e50}h2{color:#34495e;border-bottom:2px solid #ecf0f1;padding-bottom:4px}\n",
    );
    html.push_str("table{border-collapse:collapse;margin:1em 0;width:100%}\n");
    html.push_str("th,td{border:1px solid #bdc3c7;padding:6px 10px;text-align:right}\n");
    html.push_str("th{background:#34495e;color:white}\n");
    html.push_str(".pass{color:#27ae60;font-weight:bold}.fail{color:#e74c3c;font-weight:bold}\n");
    html.push_str(".summary{background:#ecf0f1;padding:1em;border-radius:6px;margin:1em 0}\n");
    html.push_str(".bar{display:inline-block;height:14px;border-radius:3px}\n");
    html.push_str("</style></head><body>\n");

    html.push_str("<h1>Genesis Benchmark Suite Report</h1>\n");

    // Executive summary
    html.push_str("<div class='summary'>\n");
    html.push_str(&format!("<strong>Domains:</strong> {} | ", n));
    html.push_str(&format!(
        "<strong>O(1) Cost Proof:</strong> <span class='{}'>{}</span> | ",
        if all_o1 { "pass" } else { "fail" },
        if all_o1 { "ALL PASS" } else { "SOME FAIL" }
    ));
    html.push_str(&format!(
        "<strong>Timescale Span:</strong> {:.1} orders of magnitude | ",
        orders
    ));
    html.push_str(&format!(
        "<strong>Fastest tau:</strong> {:.6}s | <strong>Slowest tau:</strong> {:.0}s\n",
        tau_min, tau_max
    ));
    html.push_str("</div>\n");

    // Domain registry table
    html.push_str("<h2>Domain Registry</h2>\n");
    html.push_str("<table><tr><th style='text-align:left'>Domain</th><th>tau_base (s)</th><th>Natural dt</th><th>Horizons</th></tr>\n");
    for i in 0..n {
        html.push_str(&format!(
            "<tr><td style='text-align:left'>{}</td><td>{:.6}</td><td>{}</td><td>{}</td></tr>\n",
            names[i],
            taus[i],
            format_tau(taus[i]),
            horizon_counts[i]
        ));
    }
    html.push_str("</table>\n");

    // O(1) cost proof table
    html.push_str("<h2>O(1) Cost Proof</h2>\n");
    html.push_str("<table><tr><th style='text-align:left'>Domain</th><th>Short (µs)</th><th>Long (µs)</th><th>Ratio</th><th>Status</th></tr>\n");
    for i in 0..n {
        let cls = if cost_pass[i] { "pass" } else { "fail" };
        let status = if cost_pass[i] { "PASS" } else { "FAIL" };
        html.push_str(&format!(
            "<tr><td style='text-align:left'>{}</td><td>{:.1}</td><td>{:.1}</td><td>{:.2}</td><td class='{}'>{}</td></tr>\n",
            names[i], short_us[i], long_us[i], ratios[i], cls, status
        ));
    }
    html.push_str("</table>\n");

    // Prediction drift
    html.push_str("<h2>Prediction Drift (t+1 year)</h2>\n");
    html.push_str("<table><tr><th style='text-align:left'>Domain</th><th>Drift</th><th>Visualization</th></tr>\n");
    for i in 0..n {
        let width = (drifts[i] * 400.0).min(400.0).max(1.0);
        let color = if drifts[i] < 0.3 {
            "#27ae60"
        } else if drifts[i] < 0.6 {
            "#f1c40f"
        } else {
            "#e74c3c"
        };
        html.push_str(&format!(
            "<tr><td style='text-align:left'>{}</td><td>{:.4}</td><td style='text-align:left'><span class='bar' style='width:{}px;background:{}'></span></td></tr>\n",
            names[i], drifts[i], width as u32, color
        ));
    }
    html.push_str("</table>\n");

    // Cross-domain coherence matrix
    html.push_str("<h2>Cross-Domain Coherence Matrix</h2>\n");
    html.push_str("<table><tr><th></th>");
    for i in 0..n {
        let short = &names[i][..names[i].len().min(10)];
        html.push_str(&format!("<th style='font-size:0.8em'>{}</th>", short));
    }
    html.push_str("</tr>\n");
    for i in 0..n {
        let short = &names[i][..names[i].len().min(12)];
        html.push_str(&format!(
            "<tr><th style='text-align:left;font-size:0.8em'>{}</th>",
            short
        ));
        for j in 0..n {
            let v = sim_matrix[i][j];
            let bg = if v > 0.5 {
                "#27ae60"
            } else if v > 0.3 {
                "#f1c40f"
            } else {
                "#e74c3c"
            };
            let fg = if v > 0.5 { "white" } else { "#222" };
            html.push_str(&format!(
                "<td style='background:{};color:{};font-size:0.8em'>{:.3}</td>",
                bg, fg, v
            ));
        }
        html.push_str("</tr>\n");
    }
    html.push_str("</table>\n");

    // Timescale summary
    html.push_str("<h2>Timescale Span Summary</h2>\n");
    html.push_str("<table><tr><th>Metric</th><th>Value</th></tr>\n");
    html.push_str(&format!("<tr><td>Domain count</td><td>{}</td></tr>\n", n));
    html.push_str(&format!(
        "<tr><td>Fastest tau</td><td>{:.6} s</td></tr>\n",
        tau_min
    ));
    html.push_str(&format!(
        "<tr><td>Slowest tau</td><td>{:.0} s</td></tr>\n",
        tau_max
    ));
    html.push_str(&format!(
        "<tr><td>Timescale span</td><td>{:.1} orders of magnitude</td></tr>\n",
        orders
    ));
    html.push_str(&format!(
        "<tr><td>O(1) verified</td><td class='{}'>{}</td></tr>\n",
        if all_o1 { "pass" } else { "fail" },
        if all_o1 { "YES" } else { "NO" }
    ));
    html.push_str("</table>\n");

    html.push_str("<p style='color:#7f8c8d;font-size:0.85em'>Generated by Symthaea Genesis Benchmark Suite</p>\n");
    html.push_str("</body></html>\n");

    std::fs::write("benchmark_output/genesis_suite_report.html", &html).unwrap();
    println!("HTML report: benchmark_output/genesis_suite_report.html");

    // ---- JSON ----
    let mut json = String::with_capacity(8_000);
    json.push_str("{\n");
    json.push_str(&format!("  \"domain_count\": {},\n", n));
    json.push_str(&format!("  \"all_o1_passed\": {},\n", all_o1));
    json.push_str(&format!("  \"tau_min\": {},\n", tau_min));
    json.push_str(&format!("  \"tau_max\": {},\n", tau_max));
    json.push_str(&format!("  \"timescale_orders\": {:.1},\n", orders));
    json.push_str("  \"domains\": [\n");
    for i in 0..n {
        let comma = if i < n - 1 { "," } else { "" };
        json.push_str(&format!(
            "    {{\"name\": \"{}\", \"tau_base\": {}, \"horizons\": {}, \"horizon_labels\": [{}]}}{}",
            names[i], taus[i], horizon_counts[i],
            hlabels[i].iter().map(|l| format!("\"{}\"", l)).collect::<Vec<_>>().join(", "),
            comma
        ));
        json.push('\n');
    }
    json.push_str("  ],\n");
    json.push_str("  \"cost_proofs\": [\n");
    for i in 0..n {
        let comma = if i < n - 1 { "," } else { "" };
        json.push_str(&format!(
            "    {{\"domain\": \"{}\", \"short_us\": {:.1}, \"long_us\": {:.1}, \"ratio\": {:.2}, \"passed\": {}}}{}",
            names[i], short_us[i], long_us[i], ratios[i], cost_pass[i], comma
        ));
        json.push('\n');
    }
    json.push_str("  ],\n");
    json.push_str("  \"drifts\": [\n");
    for i in 0..n {
        let comma = if i < n - 1 { "," } else { "" };
        json.push_str(&format!(
            "    {{\"domain\": \"{}\", \"drift\": {:.6}}}{}",
            names[i], drifts[i], comma
        ));
        json.push('\n');
    }
    json.push_str("  ],\n");
    json.push_str("  \"similarity_matrix\": {\n");
    json.push_str(&format!(
        "    \"labels\": [{}],\n",
        names
            .iter()
            .map(|n| format!("\"{}\"", n))
            .collect::<Vec<_>>()
            .join(", ")
    ));
    json.push_str("    \"values\": [\n");
    for i in 0..n {
        let comma = if i < n - 1 { "," } else { "" };
        json.push_str(&format!(
            "      [{}]{}",
            sim_matrix[i]
                .iter()
                .map(|v| format!("{:.4}", v))
                .collect::<Vec<_>>()
                .join(", "),
            comma
        ));
        json.push('\n');
    }
    json.push_str("    ]\n  }\n}\n");

    std::fs::write("benchmark_output/genesis_suite_report.json", &json).unwrap();
    println!("JSON report: benchmark_output/genesis_suite_report.json");

    assert!(all_o1, "O(1) cost proof FAILED for at least one domain");
    println!("\nPASS: Genesis Benchmark Suite across {} domains", n);
}

fn format_tau(tau: f32) -> String {
    if tau < 1.0 {
        format!("{:.0} ms", tau * 1000.0)
    } else if tau < 3600.0 {
        format!("{:.0} s", tau)
    } else if tau < 86_400.0 {
        format!("{:.0} hr", tau / 3600.0)
    } else {
        format!("{:.0} day", tau / 86_400.0)
    }
}
