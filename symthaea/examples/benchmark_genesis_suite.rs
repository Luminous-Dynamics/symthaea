//! Comprehensive Genesis domain benchmark suite.
//!
//! Runs all Genesis temporal predictors through a standardized battery:
//! - Domain metadata registry
//! - O(1) closed-form cost proof (short vs long horizon timing)
//! - Prediction drift at 1 year
//! - Cross-domain coherence (NxN similarity matrix)
//!
//! Generates an HTML report and a JSON report in `benchmark_output/`.
//!
//! ```bash
//! cargo run --example benchmark_genesis_suite --features genesis-missions --release
//! ```

use std::fs;
use std::time::Instant;

use symthaea_core::hdc::unified_hv::ContinuousHV;
use symthaea_core::temporal::TemporalPredictor;

// ── Data structures for report generation ────────────────────────────────────

#[derive(Clone)]
struct DomainRecord {
    domain: String,
    tau_base: f32,
    natural_dt: String,
    horizons: Vec<f32>,
    horizon_labels: Vec<String>,
    status: String,
}

#[derive(Clone)]
struct CostProofRecord {
    domain: String,
    short_us: f64,
    long_us: f64,
    ratio: f64,
    passed: bool,
}

#[derive(Clone)]
struct DriftRecord {
    domain: String,
    drift: f32,
}

struct SuiteResults {
    timestamp: String,
    domain_count: usize,
    total_tests: usize,
    domains: Vec<DomainRecord>,
    cost_proofs: Vec<CostProofRecord>,
    drifts: Vec<DriftRecord>,
    similarity_matrix: Vec<Vec<f32>>,
    domain_names: Vec<String>,
    tau_min: f32,
    tau_max: f32,
    timescale_orders: f32,
    all_o1_passed: bool,
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn format_tau(tau: f32) -> String {
    if tau < 1e-6 {
        format!("{:.1} ns", tau * 1e9)
    } else if tau < 1e-3 {
        format!("{:.1} us", tau * 1e6)
    } else if tau < 1.0 {
        format!("{:.0} ms", tau * 1000.0)
    } else if tau < 3600.0 {
        format!("{:.0} s", tau)
    } else if tau < 86_400.0 {
        format!("{:.1} hr", tau / 3600.0)
    } else {
        format!("{:.1} day", tau / 86_400.0)
    }
}

fn now_iso8601() -> String {
    let d = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = d.as_secs();
    // Simple UTC timestamp without pulling in chrono
    let days_since_epoch = secs / 86400;
    let time_of_day = secs % 86400;
    let hours = time_of_day / 3600;
    let minutes = (time_of_day % 3600) / 60;
    let seconds = time_of_day % 60;

    // Approximate year/month/day from days since 1970-01-01
    let (year, month, day) = days_to_ymd(days_since_epoch);

    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        year, month, day, hours, minutes, seconds
    )
}

fn days_to_ymd(mut days: u64) -> (u64, u64, u64) {
    let mut year = 1970;
    loop {
        let days_in_year = if is_leap(year) { 366 } else { 365 };
        if days < days_in_year {
            break;
        }
        days -= days_in_year;
        year += 1;
    }
    let month_days: [u64; 12] = if is_leap(year) {
        [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };
    let mut month = 1;
    for &md in &month_days {
        if days < md {
            break;
        }
        days -= md;
        month += 1;
    }
    (year, month, days + 1)
}

fn is_leap(y: u64) -> bool {
    (y % 4 == 0 && y % 100 != 0) || (y % 400 == 0)
}

fn similarity_color(val: f32) -> &'static str {
    if val >= 0.99 {
        "#2ecc71" // self-similarity (bright green)
    } else if val > 0.5 {
        "#27ae60" // high coherence (green)
    } else if val > 0.3 {
        "#f1c40f" // moderate (yellow)
    } else {
        "#e74c3c" // low (red)
    }
}

fn escape_json(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

// ── Core benchmark runner ────────────────────────────────────────────────────

fn run_benchmarks(predictors: &[Box<dyn TemporalPredictor>]) -> SuiteResults {
    let timestamp = now_iso8601();
    let domain_count = predictors.len();
    let seed_state = ContinuousHV::random(16_384, 0xBEEF);
    let iters = 500;
    let one_year: f32 = 31_536_000.0;

    // ── 1. Domain registry ───────────────────────────────────────────────────

    println!("=== Genesis Domain Benchmark Suite ===\n");
    println!("--- Domain Registry ({} domains) ---", domain_count);
    println!(
        "{:>24}  {:>14}  {:>12}",
        "Domain", "tau_base (s)", "Natural dt"
    );
    println!(
        "{:>24}  {:>14}  {:>12}",
        "------", "-----------", "----------"
    );

    let mut domains = Vec::with_capacity(domain_count);
    for p in predictors {
        let tau = p.tau_base();
        let natural = format_tau(tau);
        let horizons: Vec<f32> = p.default_horizons().to_vec();
        let labels: Vec<String> = p.horizon_labels().iter().map(|s| s.to_string()).collect();
        println!("{:>24}  {:>14.6}  {:>12}", p.domain(), tau, natural);
        domains.push(DomainRecord {
            domain: p.domain().to_string(),
            tau_base: tau,
            natural_dt: natural,
            horizons,
            horizon_labels: labels,
            status: String::new(), // filled after cost proof
        });
    }

    // ── 2. O(1) cost proof ───────────────────────────────────────────────────

    println!("\n--- O(1) Cost Proof ---");
    println!(
        "{:>24}  {:>14}  {:>14}  {:>8}  {:>6}",
        "Domain", "Short (us)", "Long (us)", "Ratio", "Status"
    );
    println!(
        "{:>24}  {:>14}  {:>14}  {:>8}  {:>6}",
        "------", "----------", "---------", "-----", "------"
    );

    let mut cost_proofs = Vec::with_capacity(domain_count);
    let mut all_o1_passed = true;

    for (i, p) in predictors.iter().enumerate() {
        let short_dt = p.tau_base();
        let long_dt = p.tau_base() * 1_000_000.0;

        let t1 = Instant::now();
        for _ in 0..iters {
            let _ = p.predict_at(&seed_state, short_dt);
        }
        let short_us = t1.elapsed().as_micros() as f64 / iters as f64;

        let t2 = Instant::now();
        for _ in 0..iters {
            let _ = p.predict_at(&seed_state, long_dt);
        }
        let long_us = t2.elapsed().as_micros() as f64 / iters as f64;

        let ratio = long_us / short_us.max(0.001);
        let passed = ratio < 5.0 && ratio > 0.1;
        if !passed {
            all_o1_passed = false;
        }
        let status = if passed { "PASS" } else { "FAIL" };

        println!(
            "{:>24}  {:>14.1}  {:>14.1}  {:>8.2}  {:>6}",
            p.domain(),
            short_us,
            long_us,
            ratio,
            status
        );

        domains[i].status = status.to_string();
        cost_proofs.push(CostProofRecord {
            domain: p.domain().to_string(),
            short_us,
            long_us,
            ratio,
            passed,
        });
    }

    // ── 3. Prediction drift at 1 year ────────────────────────────────────────

    println!("\n--- Prediction Drift at 1 Year ---");
    let mut drifts = Vec::with_capacity(domain_count);
    let mut predicted_states: Vec<ContinuousHV> = Vec::with_capacity(domain_count);

    for p in predictors {
        let predicted = p.predict_at(&seed_state, one_year);
        let drift = 1.0 - seed_state.similarity(&predicted);
        println!("  {:>24}: drift = {:.6}", p.domain(), drift);
        predicted_states.push(predicted);
        drifts.push(DriftRecord {
            domain: p.domain().to_string(),
            drift,
        });
    }

    // ── 4. Cross-domain similarity matrix ────────────────────────────────────

    println!("\n--- Cross-Domain Coherence Matrix (1y predictions) ---");
    let n = predicted_states.len();
    let mut sim_matrix = vec![vec![0.0_f32; n]; n];

    // Column headers
    print!("{:>24}", "");
    for p in predictors {
        let short = &p.domain()[..p.domain().len().min(10)];
        print!("{:>12}", short);
    }
    println!();

    for i in 0..n {
        let name_i = &drifts[i].domain;
        let short_i = &name_i[..name_i.len().min(22)];
        print!("{:>24}", short_i);
        for j in 0..n {
            let sim = predicted_states[i].similarity(&predicted_states[j]);
            sim_matrix[i][j] = sim;
            print!("{:>12.4}", sim);
        }
        println!();
    }

    // ── 5. Timescale span ────────────────────────────────────────────────────

    let tau_min = predictors
        .iter()
        .map(|p| p.tau_base())
        .fold(f32::INFINITY, f32::min);
    let tau_max = predictors
        .iter()
        .map(|p| p.tau_base())
        .fold(0.0_f32, f32::max);
    let timescale_orders = (tau_max / tau_min).log10();

    let total_tests = domain_count * 3; // registry + cost proof + drift per domain

    println!("\n--- Summary ---");
    println!("  Domains:          {}", domain_count);
    println!("  Total tests:      {}", total_tests);
    println!("  Fastest tau:      {:.6} s", tau_min);
    println!("  Slowest tau:      {:.0} s", tau_max);
    println!("  Timescale span:   {:.1} orders of magnitude", timescale_orders);
    println!(
        "  All O(1):         {}",
        if all_o1_passed { "YES" } else { "NO" }
    );

    let domain_names = drifts.iter().map(|d| d.domain.clone()).collect();

    SuiteResults {
        timestamp,
        domain_count,
        total_tests,
        domains,
        cost_proofs,
        drifts,
        similarity_matrix: sim_matrix,
        domain_names,
        tau_min,
        tau_max,
        timescale_orders,
        all_o1_passed,
    }
}

// ── HTML report generation ───────────────────────────────────────────────────

fn generate_html(results: &SuiteResults) -> String {
    let mut h = String::with_capacity(16_384);

    // Header
    h.push_str("<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n");
    h.push_str("<meta charset=\"UTF-8\">\n");
    h.push_str("<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n");
    h.push_str("<title>Genesis Suite Benchmark Report</title>\n");
    h.push_str("<style>\n");
    h.push_str(
        "body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; \
         background: #fff; color: #2c3e50; max-width: 1200px; margin: 0 auto; padding: 20px; }\n",
    );
    h.push_str(
        "h1 { color: #2c3e50; border-bottom: 3px solid #27ae60; padding-bottom: 10px; }\n",
    );
    h.push_str("h2 { color: #34495e; margin-top: 30px; }\n");
    h.push_str(
        "table { border-collapse: collapse; width: 100%; margin: 15px 0; font-size: 14px; }\n",
    );
    h.push_str(
        "th { background: #34495e; color: #fff; padding: 10px 12px; text-align: left; \
         border: 1px solid #2c3e50; }\n",
    );
    h.push_str("td { padding: 8px 12px; border: 1px solid #bdc3c7; }\n");
    h.push_str("tr:nth-child(even) { background: #f9f9f9; }\n");
    h.push_str("tr:hover { background: #eaf6ec; }\n");
    h.push_str(
        ".pass { color: #27ae60; font-weight: bold; }\n\
         .fail { color: #e74c3c; font-weight: bold; }\n",
    );
    h.push_str(
        ".summary-box { background: #f0f7f0; border-left: 4px solid #27ae60; \
         padding: 15px 20px; margin: 15px 0; border-radius: 4px; }\n",
    );
    h.push_str(
        ".summary-box h3 { margin: 0 0 10px 0; color: #27ae60; }\n\
         .summary-box p { margin: 4px 0; }\n",
    );
    h.push_str(
        ".sim-cell { text-align: center; font-family: monospace; font-size: 13px; \
         color: #fff; font-weight: bold; }\n",
    );
    h.push_str(
        ".footer { margin-top: 40px; padding-top: 15px; border-top: 1px solid #bdc3c7; \
         color: #7f8c8d; font-size: 12px; }\n",
    );
    h.push_str("</style>\n</head>\n<body>\n");

    // Title
    h.push_str("<h1>Genesis Suite Benchmark Report</h1>\n");
    h.push_str(&format!(
        "<p>Generated: {} | Domains: {}</p>\n",
        results.timestamp, results.domain_count
    ));

    // Executive summary
    h.push_str("<div class=\"summary-box\">\n");
    h.push_str("<h3>Executive Summary</h3>\n");
    h.push_str(&format!(
        "<p><strong>Domain coverage:</strong> {} temporal predictors across {} orders of magnitude</p>\n",
        results.domain_count, results.timescale_orders as i32
    ));
    h.push_str(&format!(
        "<p><strong>Total tests executed:</strong> {}</p>\n",
        results.total_tests
    ));
    h.push_str(&format!(
        "<p><strong>O(1) cost proof:</strong> <span class=\"{}\">{}</span></p>\n",
        if results.all_o1_passed { "pass" } else { "fail" },
        if results.all_o1_passed {
            "ALL PASSED"
        } else {
            "SOME FAILED"
        }
    ));
    h.push_str(&format!(
        "<p><strong>Timescale range:</strong> {:.6} s (fastest) to {:.0} s (slowest)</p>\n",
        results.tau_min, results.tau_max
    ));
    h.push_str("</div>\n");

    // Domain registry table
    h.push_str("<h2>Domain Registry</h2>\n");
    h.push_str("<table>\n<thead><tr>");
    h.push_str("<th>#</th><th>Domain</th><th>tau_base (s)</th><th>Natural dt</th><th>Horizons</th><th>Status</th>");
    h.push_str("</tr></thead>\n<tbody>\n");
    for (i, d) in results.domains.iter().enumerate() {
        let status_class = if d.status == "PASS" { "pass" } else { "fail" };
        let horizon_str = if d.horizons.is_empty() {
            "auto".to_string()
        } else {
            format!("{}", d.horizons.len())
        };
        h.push_str(&format!(
            "<tr><td>{}</td><td>{}</td><td>{:.6}</td><td>{}</td><td>{}</td>\
             <td class=\"{}\">{}</td></tr>\n",
            i + 1,
            d.domain,
            d.tau_base,
            d.natural_dt,
            horizon_str,
            status_class,
            d.status,
        ));
    }
    h.push_str("</tbody>\n</table>\n");

    // O(1) cost proof table
    h.push_str("<h2>O(1) Cost Proof</h2>\n");
    h.push_str("<p>Each predictor is timed at 1x tau (short) and 10<sup>6</sup>x tau (long). ");
    h.push_str("Ratio must be between 0.1 and 5.0 to confirm O(1) cost.</p>\n");
    h.push_str("<table>\n<thead><tr>");
    h.push_str(
        "<th>Domain</th><th>Short (us)</th><th>Long (us)</th><th>Ratio</th><th>Result</th>",
    );
    h.push_str("</tr></thead>\n<tbody>\n");
    for cp in &results.cost_proofs {
        let status_class = if cp.passed { "pass" } else { "fail" };
        let status_text = if cp.passed { "PASS" } else { "FAIL" };
        h.push_str(&format!(
            "<tr><td>{}</td><td>{:.1}</td><td>{:.1}</td><td>{:.2}</td>\
             <td class=\"{}\">{}</td></tr>\n",
            cp.domain, cp.short_us, cp.long_us, cp.ratio, status_class, status_text,
        ));
    }
    h.push_str("</tbody>\n</table>\n");

    // Prediction drift table
    h.push_str("<h2>Prediction Drift at 1 Year</h2>\n");
    h.push_str(
        "<p>Cosine distance between initial seed state and predicted state after 1 year (31,536,000 s).</p>\n",
    );
    h.push_str("<table>\n<thead><tr>");
    h.push_str("<th>Domain</th><th>Drift</th><th>Bar</th>");
    h.push_str("</tr></thead>\n<tbody>\n");
    for dr in &results.drifts {
        let bar_width = (dr.drift.abs().min(1.0) * 200.0) as u32;
        let bar_color = if dr.drift.abs() < 0.3 {
            "#27ae60"
        } else if dr.drift.abs() < 0.6 {
            "#f1c40f"
        } else {
            "#e67e22"
        };
        h.push_str(&format!(
            "<tr><td>{}</td><td>{:.6}</td><td>\
             <div style=\"background:{}; width:{}px; height:16px; border-radius:3px;\"></div>\
             </td></tr>\n",
            dr.domain, dr.drift, bar_color, bar_width,
        ));
    }
    h.push_str("</tbody>\n</table>\n");

    // Cross-domain coherence matrix
    h.push_str("<h2>Cross-Domain Coherence</h2>\n");
    h.push_str(
        "<p>NxN cosine similarity matrix of predicted states at t+1 year. \
         <span style=\"color:#27ae60;\">Green</span> &gt; 0.5, \
         <span style=\"color:#f1c40f;\">Yellow</span> 0.3-0.5, \
         <span style=\"color:#e74c3c;\">Red</span> &lt; 0.3.</p>\n",
    );
    h.push_str("<table>\n<thead><tr><th></th>");
    for name in &results.domain_names {
        let short = &name[..name.len().min(12)];
        h.push_str(&format!(
            "<th style=\"font-size:11px; max-width:80px; overflow:hidden;\">{}</th>",
            short
        ));
    }
    h.push_str("</tr></thead>\n<tbody>\n");
    let n = results.domain_names.len();
    for i in 0..n {
        let short_i = &results.domain_names[i][..results.domain_names[i].len().min(14)];
        h.push_str(&format!(
            "<tr><td style=\"font-weight:bold; font-size:12px;\">{}</td>",
            short_i
        ));
        for j in 0..n {
            let val = results.similarity_matrix[i][j];
            let color = similarity_color(val);
            h.push_str(&format!(
                "<td class=\"sim-cell\" style=\"background:{};\">{:.2}</td>",
                color, val,
            ));
        }
        h.push_str("</tr>\n");
    }
    h.push_str("</tbody>\n</table>\n");

    // Timescale span summary
    h.push_str("<h2>Timescale Span</h2>\n");
    h.push_str("<table>\n<thead><tr>");
    h.push_str("<th>Metric</th><th>Value</th>");
    h.push_str("</tr></thead>\n<tbody>\n");
    h.push_str(&format!(
        "<tr><td>Fastest tau</td><td>{:.6} s ({})</td></tr>\n",
        results.tau_min,
        format_tau(results.tau_min)
    ));
    h.push_str(&format!(
        "<tr><td>Slowest tau</td><td>{:.0} s ({})</td></tr>\n",
        results.tau_max,
        format_tau(results.tau_max)
    ));
    h.push_str(&format!(
        "<tr><td>Span</td><td>{:.1} orders of magnitude</td></tr>\n",
        results.timescale_orders
    ));
    h.push_str(&format!(
        "<tr><td>Domain count</td><td>{}</td></tr>\n",
        results.domain_count
    ));
    h.push_str(&format!(
        "<tr><td>Total tests</td><td>{}</td></tr>\n",
        results.total_tests
    ));
    h.push_str("</tbody>\n</table>\n");

    // Footer
    h.push_str("<div class=\"footer\">\n");
    h.push_str(&format!(
        "<p>Genesis Suite Benchmark | Symthaea | Generated {}</p>\n",
        results.timestamp
    ));
    h.push_str("<p>All predictions use O(1) closed-form CfC evolution via the TemporalPredictor trait.</p>\n");
    h.push_str("</div>\n");

    h.push_str("</body>\n</html>\n");
    h
}

// ── JSON report generation ───────────────────────────────────────────────────

fn generate_json(results: &SuiteResults) -> String {
    let mut j = String::with_capacity(8_192);
    j.push_str("{\n");
    j.push_str(&format!(
        "  \"timestamp\": \"{}\",\n",
        escape_json(&results.timestamp)
    ));
    j.push_str(&format!(
        "  \"domain_count\": {},\n",
        results.domain_count
    ));
    j.push_str(&format!(
        "  \"total_tests\": {},\n",
        results.total_tests
    ));
    j.push_str(&format!(
        "  \"all_o1_passed\": {},\n",
        results.all_o1_passed
    ));
    j.push_str(&format!("  \"tau_min\": {},\n", results.tau_min));
    j.push_str(&format!("  \"tau_max\": {},\n", results.tau_max));
    j.push_str(&format!(
        "  \"timescale_orders\": {:.1},\n",
        results.timescale_orders
    ));

    // Domains
    j.push_str("  \"domains\": [\n");
    for (i, d) in results.domains.iter().enumerate() {
        j.push_str("    {\n");
        j.push_str(&format!(
            "      \"domain\": \"{}\",\n",
            escape_json(&d.domain)
        ));
        j.push_str(&format!("      \"tau_base\": {},\n", d.tau_base));
        j.push_str(&format!(
            "      \"natural_dt\": \"{}\",\n",
            escape_json(&d.natural_dt)
        ));
        j.push_str(&format!(
            "      \"status\": \"{}\",\n",
            escape_json(&d.status)
        ));
        j.push_str("      \"horizons\": [");
        for (hi, h) in d.horizons.iter().enumerate() {
            if hi > 0 {
                j.push_str(", ");
            }
            j.push_str(&format!("{}", h));
        }
        j.push_str("],\n");
        j.push_str("      \"horizon_labels\": [");
        for (hi, label) in d.horizon_labels.iter().enumerate() {
            if hi > 0 {
                j.push_str(", ");
            }
            j.push_str(&format!("\"{}\"", escape_json(label)));
        }
        j.push_str("]\n");
        j.push_str("    }");
        if i < results.domains.len() - 1 {
            j.push(',');
        }
        j.push('\n');
    }
    j.push_str("  ],\n");

    // Cost proofs
    j.push_str("  \"cost_proofs\": [\n");
    for (i, cp) in results.cost_proofs.iter().enumerate() {
        j.push_str("    {\n");
        j.push_str(&format!(
            "      \"domain\": \"{}\",\n",
            escape_json(&cp.domain)
        ));
        j.push_str(&format!("      \"short_us\": {:.1},\n", cp.short_us));
        j.push_str(&format!("      \"long_us\": {:.1},\n", cp.long_us));
        j.push_str(&format!("      \"ratio\": {:.2},\n", cp.ratio));
        j.push_str(&format!("      \"passed\": {}\n", cp.passed));
        j.push_str("    }");
        if i < results.cost_proofs.len() - 1 {
            j.push(',');
        }
        j.push('\n');
    }
    j.push_str("  ],\n");

    // Drifts
    j.push_str("  \"drifts\": [\n");
    for (i, dr) in results.drifts.iter().enumerate() {
        j.push_str("    {\n");
        j.push_str(&format!(
            "      \"domain\": \"{}\",\n",
            escape_json(&dr.domain)
        ));
        j.push_str(&format!("      \"drift\": {}\n", dr.drift));
        j.push_str("    }");
        if i < results.drifts.len() - 1 {
            j.push(',');
        }
        j.push('\n');
    }
    j.push_str("  ],\n");

    // Similarity matrix
    j.push_str("  \"similarity_matrix\": {\n");
    j.push_str("    \"labels\": [");
    for (i, name) in results.domain_names.iter().enumerate() {
        if i > 0 {
            j.push_str(", ");
        }
        j.push_str(&format!("\"{}\"", escape_json(name)));
    }
    j.push_str("],\n");
    j.push_str("    \"values\": [\n");
    for (i, row) in results.similarity_matrix.iter().enumerate() {
        j.push_str("      [");
        for (jj, val) in row.iter().enumerate() {
            if jj > 0 {
                j.push_str(", ");
            }
            j.push_str(&format!("{:.4}", val));
        }
        j.push(']');
        if i < results.similarity_matrix.len() - 1 {
            j.push(',');
        }
        j.push('\n');
    }
    j.push_str("    ]\n");
    j.push_str("  }\n");

    j.push_str("}\n");
    j
}

// ── Main ─────────────────────────────────────────────────────────────────────

fn main() {
    // Build all Genesis domain predictors
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

    // Run all benchmarks
    let results = run_benchmarks(&predictors);

    // Create output directory
    let out_dir = "benchmark_output";
    fs::create_dir_all(out_dir).expect("Failed to create benchmark_output/ directory");

    // Generate and write HTML report
    let html = generate_html(&results);
    let html_path = format!("{}/genesis_suite_report.html", out_dir);
    fs::write(&html_path, &html).expect("Failed to write HTML report");
    println!("\nHTML report: {}", html_path);

    // Generate and write JSON report
    let json = generate_json(&results);
    let json_path = format!("{}/genesis_suite_report.json", out_dir);
    fs::write(&json_path, &json).expect("Failed to write JSON report");
    println!("JSON report: {}", json_path);

    // Final assertion
    assert!(
        results.all_o1_passed,
        "O(1) cost proof FAILED for one or more domains"
    );

    println!(
        "\nPASS: Genesis Suite Benchmark complete ({} domains, {} tests)",
        results.domain_count, results.total_tests
    );
}
