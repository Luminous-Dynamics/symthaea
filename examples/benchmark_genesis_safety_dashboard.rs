//! Genesis Safety Dashboard — Cross-Domain NRC-Style Safety Monitor
//!
//! Runs synthetic 100-step scenarios through SafetyAgent instances for 12 domains,
//! tracking level transitions across four phases (healthy, degrading, critical, recovery).
//!
//! Outputs:
//! - `benchmark_output/genesis_safety_dashboard.html` — interactive HTML dashboard
//! - `benchmark_output/genesis_safety_dashboard.json` — machine-readable JSON
//! - `benchmark_output/genesis_safety_audit.md` — per-domain markdown audit
//!
//! ```bash
//! cargo run --example benchmark_genesis_safety_dashboard --features genesis-missions
//! ```

use std::collections::HashMap;
use std::fs;

use symthaea::safety::{SafetyAgent, SafetyAssessment, SafetyLevel, SafetyMetrics};
use symthaea::safety::audit::SafetyAuditReport;

const NUM_STEPS: usize = 100;
const OUTPUT_DIR: &str = "benchmark_output";

/// Per-domain results accumulated across 100 steps.
struct DomainResult {
    name: String,
    assessments: Vec<SafetyAssessment>,
    level_counts: [usize; 4], // green, yellow, orange, red
    max_level: SafetyLevel,
}

fn main() {
    println!("=== Genesis Safety Dashboard: Cross-Domain NRC Monitor ===\n");

    let domains = [
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

    // Run all domain simulations
    let mut results: Vec<DomainResult> = Vec::with_capacity(domains.len());

    for (domain_idx, domain_name) in domains.iter().enumerate() {
        let result = run_domain_scenario(domain_name, domain_idx);
        results.push(result);
    }

    // Compute overall system health (worst across all domains)
    let system_level = results
        .iter()
        .map(|r| r.max_level)
        .max()
        .unwrap_or(SafetyLevel::Green);

    // Print console summary
    print_console_summary(&results, system_level);

    // Create output directory
    fs::create_dir_all(OUTPUT_DIR).expect("Failed to create benchmark_output/");

    // Generate HTML dashboard
    let html = generate_html_dashboard(&results, system_level);
    let html_path = format!("{}/genesis_safety_dashboard.html", OUTPUT_DIR);
    fs::write(&html_path, &html).expect("Failed to write HTML dashboard");
    println!("  HTML dashboard: {}", html_path);

    // Generate JSON report
    let json = generate_json_report(&results, system_level);
    let json_path = format!("{}/genesis_safety_dashboard.json", OUTPUT_DIR);
    fs::write(&json_path, &json).expect("Failed to write JSON report");
    println!("  JSON report:    {}", json_path);

    // Generate markdown audit
    let md = generate_markdown_audit(&results, system_level);
    let md_path = format!("{}/genesis_safety_audit.md", OUTPUT_DIR);
    fs::write(&md_path, &md).expect("Failed to write markdown audit");
    println!("  Markdown audit: {}", md_path);

    println!("\nPASS: Genesis Safety Dashboard generated ({} domains, {} steps each)",
        domains.len(), NUM_STEPS);
}

/// Generate synthetic SafetyMetrics for a given step within the 4-phase scenario.
///
/// Each domain gets a slight deterministic offset from `domain_idx` so that
/// domains do not produce identical traces.
fn generate_metrics(step: usize, domain_idx: usize) -> SafetyMetrics {
    // Deterministic per-domain jitter (small variation so domains differ)
    let jitter = ((domain_idx as f32 * 0.618034) % 1.0) * 0.05; // 0..0.05

    let (consciousness, pred_error, coherence) = if step < 25 {
        // Phase 1: Healthy
        let t = step as f32 / 24.0;
        let c = 0.85 + t * 0.10 + jitter;   // 0.85 - 0.95
        let p = 0.05 + (1.0 - t) * 0.05;    // 0.05 - 0.10
        let h = 0.70 + t * 0.20 + jitter;    // 0.70 - 0.90
        (c.min(1.0), p.max(0.0), h.min(1.0))
    } else if step < 50 {
        // Phase 2: Degrading — consciousness declines from 0.85 to 0.40
        let t = (step - 25) as f32 / 24.0; // 0.0 to 1.0
        let c = 0.85 - t * 0.45 - jitter;   // 0.85 → 0.40
        let p = 0.10 + t * 0.50;             // 0.10 → 0.60
        let h = 0.70 - t * 0.35;             // 0.70 → 0.35
        (c.max(0.0), p.min(1.0), h.max(0.0))
    } else if step < 75 {
        // Phase 3: Critical — consciousness 0.10 to 0.30, high pred_error
        let t = (step - 50) as f32 / 24.0;
        let c = 0.10 + t * 0.20 - jitter;    // 0.10 → 0.30
        let p = 0.70 + (1.0 - t) * 0.20;     // 0.90 → 0.70
        let h = 0.10 + t * 0.15;              // 0.10 → 0.25
        (c.max(0.0), p.min(1.0), h.max(0.0))
    } else {
        // Phase 4: Recovery — consciousness rising from 0.30 to 0.70
        let t = (step - 75) as f32 / 24.0;
        let c = 0.30 + t * 0.40 + jitter;    // 0.30 → 0.70
        let p = 0.60 - t * 0.45;              // 0.60 → 0.15
        let h = 0.30 + t * 0.40;              // 0.30 → 0.70
        (c.min(1.0), p.max(0.0), h.min(1.0))
    };

    SafetyMetrics {
        cycle: step,
        consciousness_level: consciousness,
        prediction_error: pred_error,
        temporal_coherence: coherence,
    }
}

/// Run a 100-step scenario for a single domain.
fn run_domain_scenario(name: &str, domain_idx: usize) -> DomainResult {
    let mut agent = SafetyAgent::new();
    let mut level_counts = [0usize; 4];
    let mut max_level = SafetyLevel::Green;

    for step in 0..NUM_STEPS {
        let metrics = generate_metrics(step, domain_idx);
        let assessment = agent.assess(metrics);

        match assessment.level {
            SafetyLevel::Green => level_counts[0] += 1,
            SafetyLevel::Yellow => level_counts[1] += 1,
            SafetyLevel::Orange => level_counts[2] += 1,
            SafetyLevel::Red => level_counts[3] += 1,
        }

        if assessment.level > max_level {
            max_level = assessment.level;
        }
    }

    DomainResult {
        name: name.to_string(),
        assessments: agent.history().to_vec(),
        level_counts,
        max_level,
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
        SafetyLevel::Green => "GREEN",
        SafetyLevel::Yellow => "YELLOW",
        SafetyLevel::Orange => "ORANGE",
        SafetyLevel::Red => "RED",
    }
}

fn print_console_summary(results: &[DomainResult], system_level: SafetyLevel) {
    println!("System Health: {}\n", system_level.label());
    println!("{:<22} {:>6} {:>6} {:>6} {:>6} {:>6}  {:<8}",
        "Domain", "Steps", "Green", "Yellw", "Ornge", "Red", "Peak");
    println!("{}", "-".repeat(72));
    for r in results {
        println!("{:<22} {:>6} {:>6} {:>6} {:>6} {:>6}  {:<8}",
            r.name, NUM_STEPS,
            r.level_counts[0], r.level_counts[1],
            r.level_counts[2], r.level_counts[3],
            level_name(r.max_level));
    }
    println!();
}

// ── HTML Dashboard Generation ─────────────────────────────────────────────

fn generate_html_dashboard(results: &[DomainResult], system_level: SafetyLevel) -> String {
    let timestamp = chrono::Utc::now().to_rfc3339();
    let mut html = String::with_capacity(32_000);

    // Header
    html.push_str(&format!(r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Genesis Safety Dashboard</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: #ffffff; color: #2c3e50; padding: 24px; max-width: 1200px; margin: 0 auto; }}
  h1 {{ font-size: 28px; margin-bottom: 8px; }}
  h2 {{ font-size: 22px; margin: 32px 0 16px 0; border-bottom: 2px solid #2c3e50; padding-bottom: 4px; }}
  h3 {{ font-size: 16px; margin: 16px 0 8px 0; }}
  .subtitle {{ color: #7f8c8d; font-size: 14px; margin-bottom: 24px; }}
  .system-health {{ display: flex; align-items: center; gap: 20px; margin: 24px 0;
                    padding: 20px; background: #f8f9fa; border-radius: 8px; }}
  .health-badge {{ width: 64px; height: 64px; border-radius: 50%; display: flex;
                   align-items: center; justify-content: center; color: white;
                   font-weight: bold; font-size: 13px; text-align: center;
                   text-shadow: 0 1px 2px rgba(0,0,0,0.3); }}
  .health-text {{ font-size: 20px; font-weight: 600; }}
  .health-detail {{ font-size: 14px; color: #7f8c8d; }}
  .domain-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
                  gap: 16px; margin: 16px 0; }}
  .domain-card {{ border: 1px solid #ddd; border-radius: 8px; padding: 16px;
                  background: #fafafa; transition: box-shadow 0.2s; }}
  .domain-card:hover {{ box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
  .domain-name {{ font-weight: 600; font-size: 15px; margin-bottom: 8px;
                  text-transform: capitalize; }}
  .domain-level {{ display: inline-block; padding: 2px 10px; border-radius: 4px;
                   color: white; font-size: 12px; font-weight: 600;
                   text-shadow: 0 1px 1px rgba(0,0,0,0.2); margin-bottom: 8px; }}
  .metric-row {{ display: flex; justify-content: space-between; font-size: 13px;
                 padding: 2px 0; color: #555; }}
  .metric-label {{ color: #7f8c8d; }}
  .timeline-section {{ margin: 16px 0; }}
  .timeline-row {{ display: flex; align-items: center; margin: 6px 0; }}
  .timeline-label {{ width: 180px; font-size: 13px; font-weight: 500;
                     text-transform: capitalize; flex-shrink: 0; }}
  table {{ width: 100%; border-collapse: collapse; margin: 16px 0; font-size: 14px; }}
  th {{ background: #2c3e50; color: white; padding: 10px 12px; text-align: left; }}
  td {{ padding: 8px 12px; border-bottom: 1px solid #eee; }}
  tr:hover td {{ background: #f8f9fa; }}
  .footer {{ margin-top: 40px; padding-top: 16px; border-top: 1px solid #ddd;
             font-size: 12px; color: #95a5a6; text-align: center; }}
  .legend {{ display: flex; gap: 16px; margin: 12px 0; font-size: 13px; align-items: center; }}
  .legend-item {{ display: flex; align-items: center; gap: 4px; }}
  .legend-swatch {{ width: 16px; height: 16px; border-radius: 2px; }}
</style>
</head>
<body>
<h1>Genesis Safety Dashboard</h1>
<p class="subtitle">Cross-Domain NRC-Style Safety Monitor &mdash; {num_domains} domains, {num_steps} steps per domain</p>
"#, num_domains = results.len(), num_steps = NUM_STEPS));

    // Overall System Health
    html.push_str(&format!(r#"
<h2>Overall System Health</h2>
<div class="system-health">
  <div class="health-badge" style="background: {};">
    {}
  </div>
  <div>
    <div class="health-text">{}</div>
    <div class="health-detail">Maximum severity observed across all {} domains</div>
  </div>
</div>
"#,
        level_color(system_level),
        level_name(system_level),
        system_level.label(),
        results.len(),
    ));

    // Domain Grid
    html.push_str("<h2>Domain Status</h2>\n<div class=\"domain-grid\">\n");
    for r in results {
        let last = r.assessments.last().unwrap();
        let current_level = last.level;
        html.push_str(&format!(r#"<div class="domain-card">
  <div class="domain-name">{name}</div>
  <div class="domain-level" style="background: {color};">{level}</div>
  <div class="metric-row"><span class="metric-label">Consciousness</span><span>{consciousness:.3}</span></div>
  <div class="metric-row"><span class="metric-label">Pred. Error</span><span>{pred_error:.3}</span></div>
  <div class="metric-row"><span class="metric-label">Coherence</span><span>{coherence:.3}</span></div>
  <div class="metric-row"><span class="metric-label">Peak Level</span><span>{peak}</span></div>
</div>
"#,
            name = display_domain(&r.name),
            color = level_color(current_level),
            level = level_name(current_level),
            consciousness = last.consciousness_level,
            pred_error = last.prediction_error,
            coherence = last.temporal_coherence,
            peak = level_name(r.max_level),
        ));
    }
    html.push_str("</div>\n");

    // Level Transition Timeline (SVG)
    html.push_str("<h2>Level Transition Timeline</h2>\n");
    html.push_str(r#"<div class="legend">
  <span style="font-weight: 600;">Legend:</span>
  <div class="legend-item"><div class="legend-swatch" style="background: #27ae60;"></div> Green</div>
  <div class="legend-item"><div class="legend-swatch" style="background: #f1c40f;"></div> Yellow</div>
  <div class="legend-item"><div class="legend-swatch" style="background: #e67e22;"></div> Orange</div>
  <div class="legend-item"><div class="legend-swatch" style="background: #e74c3c;"></div> Red</div>
</div>
"#);

    let bar_width: f32 = 600.0;
    let step_width = bar_width / NUM_STEPS as f32;
    let bar_height: f32 = 20.0;
    let row_height: f32 = 30.0;
    let label_width: f32 = 180.0;
    let svg_width = label_width + bar_width + 20.0;
    let svg_height = results.len() as f32 * row_height + 10.0;

    html.push_str(&format!(
        "<div class=\"timeline-section\">\n<svg width=\"{svg_width}\" height=\"{svg_height}\" xmlns=\"http://www.w3.org/2000/svg\">\n"
    ));

    for (i, r) in results.iter().enumerate() {
        let y = i as f32 * row_height + 5.0;

        // Domain label
        html.push_str(&format!(
            "  <text x=\"0\" y=\"{text_y}\" font-size=\"13\" font-family=\"sans-serif\" fill=\"#2c3e50\" font-weight=\"500\">{name}</text>\n",
            text_y = y + bar_height * 0.7,
            name = display_domain(&r.name),
        ));

        // Colored segments for each step
        for (step, assessment) in r.assessments.iter().enumerate() {
            let x = label_width + step as f32 * step_width;
            let color = level_color(assessment.level);
            html.push_str(&format!(
                "  <rect x=\"{x:.1}\" y=\"{y:.1}\" width=\"{w:.2}\" height=\"{h}\" fill=\"{color}\" />\n",
                w = step_width + 0.5, // slight overlap to avoid hairline gaps
                h = bar_height,
            ));
        }
    }

    // Step markers
    let markers_y = results.len() as f32 * row_height + 5.0;
    for marker in [0, 25, 50, 75, 99] {
        let x = label_width + marker as f32 * step_width;
        html.push_str(&format!(
            "  <text x=\"{x:.1}\" y=\"{y:.1}\" font-size=\"10\" fill=\"#95a5a6\" text-anchor=\"middle\">{marker}</text>\n",
            y = markers_y + 2.0,
        ));
    }

    html.push_str("</svg>\n</div>\n");

    // Phase annotations
    html.push_str(r#"<p style="font-size: 12px; color: #7f8c8d; margin-top: 4px;">
  Steps 0&ndash;24: Healthy &nbsp;|&nbsp; 25&ndash;49: Degrading &nbsp;|&nbsp; 50&ndash;74: Critical &nbsp;|&nbsp; 75&ndash;99: Recovery
</p>
"#);

    // Summary Table
    html.push_str("<h2>Summary Table</h2>\n");
    html.push_str(r#"<table>
<tr><th>Domain</th><th>Steps</th><th style="color:#27ae60">Green</th><th style="color:#f1c40f">Yellow</th><th style="color:#e67e22">Orange</th><th style="color:#e74c3c">Red</th><th>Peak Level</th></tr>
"#);

    let mut totals = [0usize; 4];
    for r in results {
        for i in 0..4 {
            totals[i] += r.level_counts[i];
        }
        html.push_str(&format!(
            "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td style=\"font-weight:600; color:{}\">{}</td></tr>\n",
            display_domain(&r.name),
            NUM_STEPS,
            r.level_counts[0],
            r.level_counts[1],
            r.level_counts[2],
            r.level_counts[3],
            level_color(r.max_level),
            level_name(r.max_level),
        ));
    }

    // Totals row
    let total_steps = results.len() * NUM_STEPS;
    html.push_str(&format!(
        "<tr style=\"font-weight:600; background:#f0f0f0;\"><td>TOTAL</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>&mdash;</td></tr>\n",
        total_steps, totals[0], totals[1], totals[2], totals[3],
    ));
    html.push_str("</table>\n");

    // Footer
    html.push_str(&format!(
        "<div class=\"footer\">Generated by Symthaea Genesis Safety Dashboard &mdash; {} &mdash; {} domains &times; {} steps</div>\n",
        timestamp, results.len(), NUM_STEPS,
    ));

    html.push_str("</body>\n</html>\n");
    html
}

// ── JSON Report Generation ────────────────────────────────────────────────

fn generate_json_report(results: &[DomainResult], system_level: SafetyLevel) -> String {
    let timestamp = chrono::Utc::now().to_rfc3339();

    // Build per-domain audit reports using the library's SafetyAuditReport
    let mut domain_reports: Vec<serde_json::Value> = Vec::new();
    for r in results {
        let report = SafetyAuditReport::from_assessments(&r.assessments);
        let level_timeline: Vec<&str> = r.assessments.iter().map(|a| level_name(a.level)).collect();
        domain_reports.push(serde_json::json!({
            "domain": r.name,
            "total_steps": NUM_STEPS,
            "level_counts": {
                "green": r.level_counts[0],
                "yellow": r.level_counts[1],
                "orange": r.level_counts[2],
                "red": r.level_counts[3],
            },
            "max_level": level_name(r.max_level),
            "mean_consciousness": report.mean_consciousness,
            "min_consciousness": report.min_consciousness,
            "mean_prediction_error": report.mean_prediction_error,
            "had_emergency": report.had_emergency,
            "top_reasons": report.top_reasons,
            "level_timeline": level_timeline,
        }));
    }

    let report = serde_json::json!({
        "title": "Genesis Safety Dashboard",
        "generated_at": timestamp,
        "system_level": level_name(system_level),
        "num_domains": results.len(),
        "steps_per_domain": NUM_STEPS,
        "domains": domain_reports,
    });

    serde_json::to_string_pretty(&report).unwrap_or_else(|_| "{}".to_string())
}

// ── Markdown Audit Generation ─────────────────────────────────────────────

fn generate_markdown_audit(results: &[DomainResult], system_level: SafetyLevel) -> String {
    let timestamp = chrono::Utc::now().to_rfc3339();
    let mut md = String::with_capacity(8_000);

    md.push_str("# Genesis Safety Audit Report\n\n");
    md.push_str(&format!("**Generated**: {}\n\n", timestamp));
    md.push_str(&format!("**System Level**: {}\n\n", system_level.label()));
    md.push_str(&format!("**Domains**: {} | **Steps per domain**: {}\n\n", results.len(), NUM_STEPS));

    // Phase description
    md.push_str("## Scenario Phases\n\n");
    md.push_str("| Phase | Steps | Description |\n");
    md.push_str("|-------|-------|-------------|\n");
    md.push_str("| Healthy | 0-24 | Consciousness 0.85-0.95, low prediction error |\n");
    md.push_str("| Degrading | 25-49 | Consciousness declining from 0.85 to 0.40 |\n");
    md.push_str("| Critical | 50-74 | Consciousness 0.10-0.30, high prediction error |\n");
    md.push_str("| Recovery | 75-99 | Consciousness rising from 0.30 to 0.70 |\n\n");

    // Summary table
    md.push_str("## Summary\n\n");
    md.push_str("| Domain | Green | Yellow | Orange | Red | Peak |\n");
    md.push_str("|--------|-------|--------|--------|-----|------|\n");
    for r in results {
        md.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} |\n",
            display_domain(&r.name),
            r.level_counts[0],
            r.level_counts[1],
            r.level_counts[2],
            r.level_counts[3],
            level_name(r.max_level),
        ));
    }

    // Per-domain details
    md.push_str("\n## Per-Domain Details\n");
    for r in results {
        let report = SafetyAuditReport::from_assessments(&r.assessments);

        md.push_str(&format!("\n### {}\n\n", display_domain(&r.name)));
        md.push_str(&format!("- **Peak Level**: {}\n", level_name(r.max_level)));
        md.push_str(&format!("- **Mean Consciousness**: {:.3}\n", report.mean_consciousness));
        md.push_str(&format!("- **Min Consciousness**: {:.3}\n", report.min_consciousness));
        md.push_str(&format!("- **Mean Prediction Error**: {:.3}\n", report.mean_prediction_error));
        md.push_str(&format!("- **Emergency Events**: {}\n",
            if report.had_emergency { "YES" } else { "No" }));

        if !report.top_reasons.is_empty() {
            md.push_str("\n**Top Escalation Reasons**:\n\n");
            for (reason, count) in report.top_reasons.iter().take(5) {
                md.push_str(&format!("  - {} ({} occurrences)\n", reason, count));
            }
        }

        // Transition summary: find first step at each level
        let mut first_at: HashMap<&str, usize> = HashMap::new();
        for (step, a) in r.assessments.iter().enumerate() {
            let name = level_name(a.level);
            first_at.entry(name).or_insert(step);
        }
        md.push_str("\n**First occurrence by level**:\n\n");
        for lvl in &["GREEN", "YELLOW", "ORANGE", "RED"] {
            if let Some(step) = first_at.get(lvl) {
                md.push_str(&format!("  - {}: step {}\n", lvl, step));
            }
        }
    }

    md.push_str(&format!("\n---\n\n*Generated by Symthaea Genesis Safety Dashboard — {}*\n", timestamp));
    md
}

// ── Utility ───────────────────────────────────────────────────────────────

/// Format a domain name for display: replace underscores with spaces, capitalize words.
fn display_domain(name: &str) -> String {
    name.split('_')
        .map(|word| {
            let mut chars = word.chars();
            match chars.next() {
                None => String::new(),
                Some(c) => {
                    let upper: String = c.to_uppercase().collect();
                    upper + &chars.collect::<String>()
                }
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}
