// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Cross-Domain Safety Dashboard — NRC-style HTML + JSON + Markdown
//!
//! Instantiates SafetyAgents for each domain, runs synthetic scenarios
//! (healthy → degrading → critical → recovery), and generates a
//! comprehensive safety dashboard.

use symthaea::safety::{
    SafetyAgent, SafetyAssessment, SafetyAuditReport, SafetyLevel, SafetyMetrics,
};

const DOMAIN_NAMES: &[&str] = &[
    "fusion",
    "grid",
    "fission",
    "accelerator",
    "threat",
    "datacenter",
    "water",
    "materials",
    "experiment",
    "strategic_materials",
    "critical_minerals",
    "safeguards",
];

fn main() {
    println!("=== Genesis Cross-Domain Safety Dashboard ===\n");

    let n = DOMAIN_NAMES.len();
    let steps = 100;

    // Run each domain through the 4-phase scenario
    let mut all_assessments: Vec<Vec<SafetyAssessment>> = Vec::new();
    let mut peak_levels: Vec<SafetyLevel> = Vec::new();
    let mut level_counts: Vec<[usize; 4]> = Vec::new();

    for (di, _name) in DOMAIN_NAMES.iter().enumerate() {
        let mut agent = SafetyAgent::new();
        let mut assessments = Vec::new();
        let mut counts = [0usize; 4];

        for step in 0..steps {
            let metrics = generate_metrics(step, di);
            let assessment = agent.assess(metrics);
            match assessment.level {
                SafetyLevel::Green => counts[0] += 1,
                SafetyLevel::Yellow => counts[1] += 1,
                SafetyLevel::Orange => counts[2] += 1,
                SafetyLevel::Red => counts[3] += 1,
            }
            assessments.push(assessment);
        }

        let peak = assessments
            .iter()
            .map(|a| a.level)
            .max()
            .unwrap_or(SafetyLevel::Green);
        peak_levels.push(peak);
        level_counts.push(counts);
        all_assessments.push(assessments);
    }

    // Overall system health = worst across all domains
    let overall = peak_levels
        .iter()
        .copied()
        .max()
        .unwrap_or(SafetyLevel::Green);

    // Console summary
    println!("Overall System Health: {:?}\n", overall);
    println!(
        "{:>20}  {:>6}  {:>6}  {:>6}  {:>6}  {:>8}",
        "Domain", "Green", "Yellow", "Orange", "Red", "Peak"
    );
    println!(
        "{:>20}  {:>6}  {:>6}  {:>6}  {:>6}  {:>8}",
        "------", "-----", "------", "------", "---", "----"
    );
    for i in 0..n {
        println!(
            "{:>20}  {:>6}  {:>6}  {:>6}  {:>6}  {:>8?}",
            DOMAIN_NAMES[i],
            level_counts[i][0],
            level_counts[i][1],
            level_counts[i][2],
            level_counts[i][3],
            peak_levels[i]
        );
    }

    // Generate reports
    std::fs::create_dir_all("benchmark_output").ok();

    // ---- HTML Dashboard ----
    let html = generate_html(&all_assessments, &peak_levels, &level_counts, overall);
    std::fs::write("benchmark_output/genesis_safety_dashboard.html", &html).unwrap();
    println!("\nHTML dashboard: benchmark_output/genesis_safety_dashboard.html");

    // ---- JSON Report ----
    let json = generate_json(&all_assessments, &peak_levels, &level_counts);
    std::fs::write("benchmark_output/genesis_safety_dashboard.json", &json).unwrap();
    println!("JSON report: benchmark_output/genesis_safety_dashboard.json");

    // ---- Markdown Audit ----
    let md = generate_markdown(&all_assessments);
    std::fs::write("benchmark_output/genesis_safety_audit.md", &md).unwrap();
    println!("Markdown audit: benchmark_output/genesis_safety_audit.md");

    println!("\nPASS: Safety Dashboard across {} domains", n);
}

/// Generate synthetic SafetyMetrics across 4 phases.
fn generate_metrics(step: usize, domain_idx: usize) -> SafetyMetrics {
    // Golden ratio jitter per domain for deterministic variety
    let jitter = ((domain_idx as f64 * 0.618033988) % 1.0 * 0.05) as f32;

    let (consciousness, pred_error, coherence) = if step < 25 {
        // Phase 1: Healthy
        let t = step as f32 / 25.0;
        (0.85 + t * 0.10 + jitter, 0.05 + t * 0.05, 0.70 + t * 0.20)
    } else if step < 50 {
        // Phase 2: Degrading
        let t = (step - 25) as f32 / 25.0;
        (0.85 - t * 0.45 - jitter, 0.10 + t * 0.40, 0.90 - t * 0.50)
    } else if step < 75 {
        // Phase 3: Critical
        let t = (step - 50) as f32 / 25.0;
        (0.10 + t * 0.20 + jitter, 0.70 + t * 0.20, 0.20 + t * 0.10)
    } else {
        // Phase 4: Recovery
        let t = (step - 75) as f32 / 25.0;
        (0.30 + t * 0.40 + jitter, 0.50 - t * 0.30, 0.30 + t * 0.40)
    };

    SafetyMetrics {
        cycle: step,
        consciousness_level: consciousness.clamp(0.0, 1.0),
        prediction_error: pred_error.clamp(0.0, 1.0),
        temporal_coherence: coherence.clamp(0.0, 1.0),
        integrity_critical: false,
    }
}

fn level_color(level: SafetyLevel) -> &'static str {
    match level {
        SafetyLevel::Green => "#27ae60",
        SafetyLevel::Yellow => "#f1c40f",
        SafetyLevel::Orange => "#e67e22",
        SafetyLevel::Red => "#e74c3c",
    }
}

fn level_name(level: SafetyLevel) -> &'static str {
    match level {
        SafetyLevel::Green => "Green",
        SafetyLevel::Yellow => "Yellow",
        SafetyLevel::Orange => "Orange",
        SafetyLevel::Red => "Red",
    }
}

fn display_name(name: &str) -> String {
    name.split('_')
        .map(|w| {
            let mut c = w.chars();
            match c.next() {
                None => String::new(),
                Some(f) => f.to_uppercase().to_string() + c.as_str(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn generate_html(
    all_assessments: &[Vec<SafetyAssessment>],
    peak_levels: &[SafetyLevel],
    level_counts: &[[usize; 4]],
    overall: SafetyLevel,
) -> String {
    let n = DOMAIN_NAMES.len();
    let mut h = String::with_capacity(16_000);

    h.push_str("<!DOCTYPE html><html><head><meta charset='utf-8'>\n");
    h.push_str("<title>Genesis Safety Dashboard</title>\n");
    h.push_str("<style>\n");
    h.push_str("body{font-family:system-ui,sans-serif;margin:2em;background:#fafafa;color:#222}\n");
    h.push_str(
        "h1{color:#2c3e50}h2{color:#34495e;border-bottom:2px solid #ecf0f1;padding-bottom:4px}\n",
    );
    h.push_str("table{border-collapse:collapse;margin:1em 0;width:100%}\n");
    h.push_str("th,td{border:1px solid #bdc3c7;padding:6px 10px;text-align:center}\n");
    h.push_str("th{background:#34495e;color:white}\n");
    h.push_str(".badge{display:inline-block;padding:8px 20px;border-radius:8px;color:white;font-weight:bold;font-size:1.2em}\n");
    h.push_str(".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:12px;margin:1em 0}\n");
    h.push_str(".card{border:1px solid #bdc3c7;border-radius:8px;padding:12px;background:white}\n");
    h.push_str(".card h3{margin:0 0 8px 0;font-size:0.95em}\n");
    h.push_str(".metric{font-size:0.85em;color:#555}\n");
    h.push_str("</style></head><body>\n");

    h.push_str("<h1>Genesis Cross-Domain Safety Dashboard</h1>\n");

    // Overall health badge
    h.push_str("<h2>Overall System Health</h2>\n");
    h.push_str(&format!(
        "<p><span class='badge' style='background:{}'>{}</span></p>\n",
        level_color(overall),
        level_name(overall)
    ));

    // Domain card grid
    h.push_str("<h2>Domain Status</h2>\n<div class='grid'>\n");
    for i in 0..n {
        let last = &all_assessments[i].last().unwrap();
        h.push_str(&format!(
            "<div class='card' style='border-left:4px solid {}'>\n",
            level_color(last.level)
        ));
        h.push_str(&format!("<h3>{}</h3>\n", display_name(DOMAIN_NAMES[i])));
        h.push_str(&format!(
            "<div class='metric'>Current: <strong>{}</strong></div>\n",
            level_name(last.level)
        ));
        h.push_str(&format!(
            "<div class='metric'>Consciousness: {:.2}</div>\n",
            last.consciousness_level
        ));
        h.push_str(&format!(
            "<div class='metric'>Pred Error: {:.2}</div>\n",
            last.prediction_error
        ));
        h.push_str(&format!(
            "<div class='metric'>Peak: <strong>{}</strong></div>\n",
            level_name(peak_levels[i])
        ));
        h.push_str("</div>\n");
    }
    h.push_str("</div>\n");

    // Level transition timeline (SVG per domain)
    h.push_str("<h2>Level Transition Timeline</h2>\n");
    h.push_str("<p style='font-size:0.85em;color:#666'>Phase 1 (0-24): Healthy | Phase 2 (25-49): Degrading | Phase 3 (50-74): Critical | Phase 4 (75-99): Recovery</p>\n");
    for i in 0..n {
        h.push_str(&format!(
            "<p style='margin:4px 0'><strong>{}</strong></p>\n",
            display_name(DOMAIN_NAMES[i])
        ));
        h.push_str("<svg width='600' height='20'>\n");
        let steps = all_assessments[i].len();
        let w = 600.0 / steps as f64;
        for (j, a) in all_assessments[i].iter().enumerate() {
            h.push_str(&format!(
                "<rect x='{}' y='0' width='{}' height='20' fill='{}'/>\n",
                (j as f64 * w) as u32,
                (w + 1.0) as u32,
                level_color(a.level)
            ));
        }
        h.push_str("</svg>\n");
    }

    // Summary table
    h.push_str("<h2>Summary Table</h2>\n");
    h.push_str("<table><tr><th style='text-align:left'>Domain</th><th>Green</th><th>Yellow</th><th>Orange</th><th>Red</th><th>Peak</th></tr>\n");
    for i in 0..n {
        h.push_str(&format!(
            "<tr><td style='text-align:left'>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td style='background:{};color:white'>{}</td></tr>\n",
            display_name(DOMAIN_NAMES[i]),
            level_counts[i][0], level_counts[i][1], level_counts[i][2], level_counts[i][3],
            level_color(peak_levels[i]), level_name(peak_levels[i])
        ));
    }
    h.push_str("</table>\n");

    h.push_str("<p style='color:#7f8c8d;font-size:0.85em'>Generated by Symthaea Genesis Safety Dashboard</p>\n");
    h.push_str("</body></html>\n");
    h
}

fn generate_json(
    all_assessments: &[Vec<SafetyAssessment>],
    peak_levels: &[SafetyLevel],
    level_counts: &[[usize; 4]],
) -> String {
    let n = DOMAIN_NAMES.len();
    let mut j = String::with_capacity(8_000);
    j.push_str("{\n  \"domains\": [\n");
    for i in 0..n {
        let report = SafetyAuditReport::from_assessments(&all_assessments[i]);
        let comma = if i < n - 1 { "," } else { "" };
        j.push_str(&format!(
            "    {{\n      \"name\": \"{}\",\n      \"peak_level\": \"{:?}\",\n      \"green\": {},\n      \"yellow\": {},\n      \"orange\": {},\n      \"red\": {},\n      \"mean_consciousness\": {:.3},\n      \"min_consciousness\": {:.3},\n      \"mean_prediction_error\": {:.3},\n      \"had_emergency\": {}\n    }}{}",
            DOMAIN_NAMES[i], peak_levels[i],
            level_counts[i][0], level_counts[i][1], level_counts[i][2], level_counts[i][3],
            report.mean_consciousness, report.min_consciousness,
            report.mean_prediction_error, report.had_emergency, comma
        ));
        j.push('\n');
    }
    j.push_str("  ]\n}\n");
    j
}

fn generate_markdown(all_assessments: &[Vec<SafetyAssessment>]) -> String {
    let n = DOMAIN_NAMES.len();
    let mut md = String::with_capacity(4_000);
    md.push_str("# Genesis Safety Audit Report\n\n");
    md.push_str("## Scenario\n\n");
    md.push_str("| Phase | Steps | Description |\n");
    md.push_str("|-------|-------|-------------|\n");
    md.push_str("| 1 | 0-24 | Healthy operation |\n");
    md.push_str("| 2 | 25-49 | Gradual degradation |\n");
    md.push_str("| 3 | 50-74 | Critical state |\n");
    md.push_str("| 4 | 75-99 | Recovery |\n\n");

    md.push_str("## Per-Domain Results\n\n");
    for i in 0..n {
        let report = SafetyAuditReport::from_assessments(&all_assessments[i]);
        md.push_str(&format!("### {}\n\n", display_name(DOMAIN_NAMES[i])));
        md.push_str(&format!("- **Peak level**: {:?}\n", report.peak_level));
        md.push_str(&format!(
            "- **Mean consciousness**: {:.3}\n",
            report.mean_consciousness
        ));
        md.push_str(&format!(
            "- **Min consciousness**: {:.3}\n",
            report.min_consciousness
        ));
        md.push_str(&format!(
            "- **Mean prediction error**: {:.3}\n",
            report.mean_prediction_error
        ));
        md.push_str(&format!("- **Had emergency**: {}\n", report.had_emergency));
        md.push_str(&format!(
            "- **Level counts**: G={} Y={} O={} R={}\n\n",
            report.level_counts.green,
            report.level_counts.yellow,
            report.level_counts.orange,
            report.level_counts.red
        ));
    }
    md
}
