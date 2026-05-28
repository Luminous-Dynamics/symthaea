use std::fs::{self, File};
use std::io::{Read, Write};
use std::process::Command;
use std::time::{Duration, Instant};
use std::thread;

const PROFILING_RUNS: usize = 5;
const TARGET_FILE: &str = "crates/symthaea-broca/src/liquid_mamba.rs";
const WEIGHT_LATENCY: f64 = 1.0;
const WEIGHT_SURPRISE: f64 = 1.5;

#[derive(Debug, Clone)]
struct DiscoveredParam {
    id: String,
    field_key: String,
}

#[derive(Debug, Clone)]
struct StructuralWarning {
    file: String,
    line: usize,
    symbol: String,
    warning_type: String,
    help_msg: String,
}

fn compile_and_capture_diagnostics() -> (bool, String) {
    let build_output = Command::new("cargo")
        .args(&[
            "build", 
            "--release", 
            "--bin", 
            "code-gen-smoke",
            "--features",
            "code_generation symthaea-broca/mamba-cpu symthaea-broca/code-sheaf-eval"
        ])
        .output();

    match build_output {
        Ok(out) => {
            let stderr_str = String::from_utf8_lossy(&out.stderr).to_string();
            (out.status.success(), stderr_str)
        }
        Err(_) => (false, String::new()),
    }
}

// Scans compiler diagnostics for structural anomalies to compile escalation dossiers
fn analyze_and_escalate_warnings() {
    println!("\n🔎 [Curation] Analyzing workspace compilation warnings...");
    
    let check_output = Command::new("cargo")
        .args(&["check", "--workspace", "--message-format=short"])
        .output();
        
    if let Ok(out) = check_output {
        let stderr_str = String::from_utf8_lossy(&out.stderr);
        let mut warnings = Vec::new();
        
        for line in stderr_str.lines() {
            if line.contains("warning:") {
                let parts: Vec<&str> = line.split(':').collect();
                if parts.len() >= 4 {
                    let file_path = parts[0].trim().to_string();
                    let line_num = parts[1].trim().parse::<usize>().unwrap_or(0);
                    let msg = parts[3..].join(":").trim().to_string();
                    
                    if msg.contains("unused import") || msg.contains("never read") || msg.contains("unused variable") {
                        let symbol = msg.split('`').nth(1).unwrap_or("unknown").to_string();
                        let warning_type = if msg.contains("import") { "Unused Import Path" } else { "Dead Code Structural Block" };
                        
                        warnings.push(StructuralWarning {
                            file: file_path,
                            line: line_num,
                            symbol,
                            warning_type: warning_type.to_string(),
                            help_msg: msg,
                        });
                    }
                }
            }
        }
        
        if warnings.is_empty() {
            println!("   ✅ No lingering structural warnings or dangling code segments detected.");
            return;
        }
        
        println!("\n╔═══════════════════════════════════════════════════════════════╗");
        println!("║             SYMTHAEA ARCHITECTURAL ESCALATION DOSSIER         ║");
        println!("║       Human-in-the-Loop Code Integration & Summary Gate       ║");
        println!("╚═══════════════════════════════════════════════════════════════╝");
        println!("⚠️  Symthaea isolated {} dangling structural elements. Routing data tokens:", warnings.len());
        
        for (idx, w) in warnings.iter().enumerate() {
            println!("\n📂 ESCALATION TICKETS #{}/{}", idx + 1, warnings.len());
            println!("   | Symbol Component:  `{}`", w.symbol);
            println!("   | Anomaly Profile:   {}", w.warning_type);
            println!("   | Target File Track: {} [Line: {}]", w.file, w.line);
            println!("   | Compiler Context:  {}", w.help_msg);
            println!("   | --- INTEGRATION ASSIGNMENT PROMPT ---");
            println!("   | \"Feed this docket chunk into our conversational LLM stream to generate");
            println!("   |  an integration plan or usage summary. Symthaea is awaiting your command");
            println!("   |  to either implement or preserve this structure.\"");
        }
        println!("\n⚙️  [Review Lock] Forge holding baseline state. Please instruct your collaborator on code updates.");
    }
}

fn auto_discover_parameters() -> Vec<DiscoveredParam> {
    let mut params = Vec::new();
    let mut content = String::new();
    if File::open(TARGET_FILE).and_then(|mut f| f.read_to_string(&mut content)).is_err() {
        return params;
    }
    
    let marker = "// FORGE_PARAM:";
    for line in content.lines() {
        if line.contains(marker) {
            if let Some(marker_idx) = line.find(marker) {
                if let Some(colon_idx) = line.find(':') {
                    let param_id = line[marker_idx + marker.len()..].trim().to_string();
                    let field_key = line[..colon_idx].trim().to_string();
                    if !param_id.is_empty() && !field_key.is_empty() {
                        params.push(DiscoveredParam { id: param_id, field_key });
                    }
                }
            }
        }
    }
    params
}

fn profile_target_multi_objective() -> Option<(Duration, f64)> {
    let mut runs = Vec::with_capacity(PROFILING_RUNS);
    let mut captured_surprise = 1.0f64;
    
    for pass in 0..PROFILING_RUNS {
        let start = Instant::now();
        let exec_output = Command::new("./target/release/code-gen-smoke").output();
        let duration = start.elapsed();
        
        match exec_output {
            Ok(out) if out.status.success() => {
                runs.push(duration);
                if pass == 0 {
                    let stdout_str = String::from_utf8_lossy(&out.stdout);
                    for line in stdout_str.lines() {
                        let lower = line.to_lowercase();
                        if lower.contains("surprise") || lower.contains("gap") || lower.contains("entropy") {
                            if let Some(eq_idx) = line.find(':').or_else(|| line.find('=')) {
                                if let Some(val) = line[eq_idx + 1..].trim().split_whitespace().next().and_then(|s| s.parse::<f64>().ok()) {
                                    if val > 0.0 {
                                        captured_surprise = val;
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            _ => return None,
        }
    }
    
    if runs.is_empty() { return None; }
    runs.sort();
    Some((runs[PROFILING_RUNS / 2], captured_surprise))
}

fn mutate_generic_parameter(param_id: &str, field_key: &str, delta: f32) -> Result<f32, Box<dyn std::error::Error>> {
    let mut content = String::new();
    File::open(TARGET_FILE)?.read_to_string(&mut content)?;
    
    let tag = format!("// FORGE_PARAM: {}", param_id);
    let mut lines: Vec<String> = content.lines().map(|s| s.to_string()).collect();
    let mut updated_val = 0.0f32;
    let mut found = false;
    
    for line in lines.iter_mut() {
        if line.contains(&tag) {
            if let Some(colon_idx) = line.find(':') {
                if let Some(comma_idx) = line.find(',') {
                    let val_str = line[colon_idx + 1..comma_idx].trim();
                    let old_val: f32 = val_str.parse().unwrap_or(0.15);
                    updated_val = (old_val + delta).clamp(0.005, 1.95);
                    
                    let indent = line.len() - line.trim_start().len();
                    let spaces = &line[..indent];
                    
                    *line = format!("{}{}: {:.4}, {}", spaces, field_key, updated_val, tag);
                    found = true;
                    break;
                }
            }
        }
    }
    
    if found {
        File::create(TARGET_FILE)?.write_all(lines.join("\n").as_bytes())?;
        Ok(updated_val)
    } else {
        Err(format!("Tag missing: {}", param_id).into())
    }
}

fn restore_file_buffer(buffer: &str) -> Result<(), Box<dyn std::error::Error>> {
    File::create(TARGET_FILE)?.write_all(buffer.as_bytes())?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║             SYMTHAEA INTELLIGENT CURATION DAEMON              ║");
    println!("║       Multi-Objective Tuning & Interactive Escalation Gate     ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");

    // Intercept code anomalies and generate the Dossier summary table right at startup
    analyze_and_escalate_warnings();

    println!("\n🔍 Mapping codebase parameter tracking coordinates...");
    let discovered_coords = auto_discover_parameters();
    println!("   Found {} active hyperparameter anchors mapped.", discovered_coords.len());

    println!("\n📊 Tracking baseline performance metrics...");
    let (initial_success, initial_diagnostics) = compile_and_capture_diagnostics();
    if !initial_success {
        println!("❌ Initial workspace compilation pass failed! Diagnostics:\n{}", initial_diagnostics);
        return Ok(());
    }
    
    let (mut baseline_lat, mut baseline_surp) = profile_target_multi_objective().expect("Baseline metrics tracking crash.");
    println!("   Initial Latency Floor: {:?}", baseline_lat);
    println!("   Initial Cognitive Telemetry Matrix Score: {:.4}", baseline_surp);

    let exploration_deltas = vec![0.0010, -0.0020, 0.0040];
    let mut execution_cycle = 1;

    println!("\n🚀 Entering persistent multi-objective evolution track...");
    loop {
        for coord in &discovered_coords {
            for &delta in &exploration_deltas {
                println!("\n⚡ --- Background Evolution Cycle #{} [Target: {}] ---", execution_cycle, coord.id);
                execution_cycle += 1;

                let mut pristine_buffer = String::new();
                File::open(TARGET_FILE)?.read_to_string(&mut pristine_buffer)?;

                let target_val = match mutate_generic_parameter(&coord.id, &coord.field_key, delta) {
                    Ok(v) => v,
                    Err(_) => continue,
                };

                let (build_success, _) = compile_and_capture_diagnostics();
                if build_success {
                    if let Some((mutated_lat, mutated_surp)) = profile_target_multi_objective() {
                        let base_lat_ns = baseline_lat.as_nanos() as f64;
                        let mut_lat_ns = mutated_lat.as_nanos() as f64;
                        
                        let latency_delta = (base_lat_ns - mut_lat_ns) / base_lat_ns;
                        let surprise_delta = (mutated_surp - baseline_surp) / baseline_surp;
                        let joint_reward = (WEIGHT_LATENCY * latency_delta) - (WEIGHT_SURPRISE * surprise_delta);

                        println!("   | Latency:   Mutated = {:?} | Baseline = {:?}", mutated_lat, baseline_lat);
                        println!("   | Cognition: Mutated Score = {:.4} | Baseline Score = {:.4}", mutated_surp, baseline_surp);
                        println!("   | Pareto Index Evaluation Score: {:.4}", joint_reward);

                        if joint_reward > 0.001 {
                            println!("   🏆 [GENETIC WIN] Compounding optimization verified!");
                            baseline_lat = mutated_lat;
                            baseline_surp = mutated_surp;

                            let _ = Command::new("git").args(&["add", "."]).output()?;
                            let _ = Command::new("git").args(&["commit", "-m", &format!("evolution: pareto winner [{}] val={:.4} reward={:.4}", coord.field_key, target_val, joint_reward)]).output()?;
                        } else {
                            println!("   📉 Mutation outside Pareto frontier. Dropping exploration branch.");
                            let _ = restore_file_buffer(&pristine_buffer);
                        }
                    } else {
                        let _ = restore_file_buffer(&pristine_buffer);
                    }
                } else {
                    let _ = restore_file_buffer(&pristine_buffer);
                }

                thread::sleep(Duration::from_millis(250));
            }
        }
        
        // Scan warning layouts on completion of sweeps to present fresh logs to user loop
        analyze_and_escalate_warnings();
        println!("\n💤 Sweep pass complete. Cooling pipeline down before the next sequence iteration...");
        thread::sleep(Duration::from_secs(20));
    }
}
