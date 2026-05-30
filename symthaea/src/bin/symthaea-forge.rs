use std::fs::{self, File};
use std::io::{Read, Write};
use std::process::Command;
use std::thread;
use std::time::{Duration, Instant};

const PROFILING_RUNS: usize = 5;
const TARGET_FILE: &str = "crates/symthaea-broca/src/liquid_mamba.rs";
const DOMOTIC_FILE: &str = "crates/symthaea-domotic/src/bin/domotic_test_node.rs";

const WEIGHT_LATENCY: f64 = 1.0;
const WEIGHT_SURPRISE: f64 = 1.5;
const WEIGHT_FILTER: f64 = 2.0; // High penalty for homeostatic error destabilization

#[derive(Debug, Clone)]
struct DiscoveredParam {
    id: String,
    field_key: String,
}

fn compile_workspace() -> bool {
    let build_output = Command::new("cargo")
        .args(&["build", "--release", "--workspace"])
        .output();

    match build_output {
        Ok(out) => out.status.success(),
        Err(_) => false,
    }
}

fn auto_discover_parameters() -> Vec<DiscoveredParam> {
    let mut params = Vec::new();
    let mut files = vec![TARGET_FILE, DOMOTIC_FILE];

    for path in files {
        let mut content = String::new();
        if File::open(path)
            .and_then(|mut f| f.read_to_string(&mut content))
            .is_ok()
        {
            let marker = "// FORGE_PARAM:";
            for line in content.lines() {
                if line.contains(marker) {
                    if let Some(marker_idx) = line.find(marker) {
                        if let Some(colon_idx) = line.find(':') {
                            let param_id = line[marker_idx + marker.len()..].trim().to_string();
                            let field_key = line[..colon_idx].trim().to_string();
                            if !param_id.is_empty() && !field_key.is_empty() {
                                params.push(DiscoveredParam {
                                    id: param_id,
                                    field_key,
                                });
                            }
                        }
                    }
                }
            }
        }
    }
    params
}

fn profile_target_tri_objective() -> Option<(Duration, f64, f64)> {
    let mut runs = Vec::with_capacity(PROFILING_RUNS);
    let mut captured_surprise = 1.0f64;
    let mut filter_error = 0.5f64;

    for pass in 0..PROFILING_RUNS {
        let start = Instant::now();
        // Profile against our active domestic validation harness
        let exec_output = Command::new("cargo")
            .args(&["check", "-p", "symthaea-domotic"])
            .output();
        let duration = start.elapsed();

        match exec_output {
            Ok(out) if out.status.success() => {
                runs.push(duration);
                if pass == 0 {
                    let stderr_str = String::from_utf8_lossy(&out.stderr);
                    for line in stderr_str.lines() {
                        if line.contains("Temp:") {
                            if let Some(idx) = line.find("Temp:") {
                                if let Some(val) = line[idx + 5..]
                                    .split_whitespace()
                                    .next()
                                    .and_then(|s| s.parse::<f64>().ok())
                                {
                                    filter_error = (val - 21.0).abs(); // Calculate tracking error delta from 21°C target
                                    break;
                                }
                            }
                        }
                    }
                }
            }
            _ => return None,
        }
    }

    if runs.is_empty() {
        return None;
    }
    runs.sort();
    Some((runs[PROFILING_RUNS / 2], captured_surprise, filter_error))
}

fn mutate_generic_parameter(
    param_id: &str,
    field_key: &str,
    delta: f32,
) -> Result<f32, Box<dyn std::error::Error>> {
    let path = if param_id.contains("domotic") {
        DOMOTIC_FILE
    } else {
        TARGET_FILE
    };
    let mut content = String::new();
    File::open(path)?.read_to_string(&mut content)?;

    let tag = format!("// FORGE_PARAM: {}", param_id);
    let mut lines: Vec<String> = content.lines().map(|s| s.to_string()).collect();
    let mut updated_val = 0.0f32;
    let mut found = false;

    for line in lines.iter_mut() {
        if line.contains(&tag) {
            if let Some(colon_idx) = line.find(':') {
                if let Some(comma_idx) = line.find(',') {
                    let val_str = line[colon_idx + 1..comma_idx].trim();
                    let old_val: f32 = val_str.parse().unwrap_or(0.35);
                    updated_val = (old_val + delta).clamp(0.005, 1.95);

                    let indent = line.len() - line.trim_start().len();
                    let spaces = &line[..indent];

                    *line = format!("{}{}: {:.4}, {}", spaces, field_key, updated_val, tag);
                    found = true;
                    break;
                }
            } else if let Some(eq_idx) = line.find('=') {
                if let Some(semi_idx) = line.find(';') {
                    let val_str = line[eq_idx + 1..semi_idx].trim();
                    let old_val: f32 = val_str.parse().unwrap_or(0.35);
                    updated_val = (old_val + delta).clamp(0.005, 1.95);

                    let indent = line.len() - line.trim_start().len();
                    let spaces = &line[..indent];

                    *line = format!("{}let {} = {:.4}; {}", spaces, field_key, updated_val, tag);
                    found = true;
                    break;
                }
            }
        }
    }

    if found {
        File::create(path)?.write_all(lines.join("\n").as_bytes())?;
        Ok(updated_val)
    } else {
        Err(format!("Tag signature for ID [{}] was missing.", param_id).into())
    }
}

fn restore_file_buffer(param_id: &str, buffer: &str) -> Result<(), Box<dyn std::error::Error>> {
    let path = if param_id.contains("domotic") {
        DOMOTIC_FILE
    } else {
        TARGET_FILE
    };
    File::create(path)?.write_all(buffer.as_bytes())?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║             SYMTHAEA TRI-OBJECTIVE PARETO DAEMON              ║");
    println!("║       Latency Velocity vs Cognitive Stability vs Domotic Error ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");

    println!("🔍 Crawling workspaces for tuning coordinates...");
    let discovered_coords = auto_discover_parameters();
    println!(
        "   Found {} active optimization anchors.",
        discovered_coords.len()
    );

    if !compile_workspace() {
        panic!("Codebase baseline compilation pass blocked.");
    }
    let (mut baseline_lat, mut baseline_surp, mut baseline_filter) =
        profile_target_tri_objective().unwrap();

    println!("   Baseline Latency Window:  {:?}", baseline_lat);
    println!("   Baseline Tracking Variance: {:.4}", baseline_filter);

    let exploration_deltas = vec![0.0050, -0.0100, 0.0150];
    let mut execution_cycle = 1;

    loop {
        for coord in &discovered_coords {
            for &delta in &exploration_deltas {
                println!(
                    "\n⚡ --- Tri-Objective Sweep Pass #{} [Target: {}] ---",
                    execution_cycle, coord.id
                );
                execution_cycle += 1;

                let path = if coord.id.contains("domotic") {
                    DOMOTIC_FILE
                } else {
                    TARGET_FILE
                };
                let mut pristine_buffer = String::new();
                File::open(path)?.read_to_string(&mut pristine_buffer)?;

                let target_val = match mutate_generic_parameter(&coord.id, &coord.field_key, delta)
                {
                    Ok(v) => v,
                    Err(_) => continue,
                };

                if compile_workspace() {
                    if let Some((mutated_lat, mutated_surp, mutated_filter)) =
                        profile_target_tri_objective()
                    {
                        let base_lat_ns = baseline_lat.as_nanos() as f64;
                        let mut_lat_ns = mutated_lat.as_nanos() as f64;

                        let latency_delta = (base_lat_ns - mut_lat_ns) / base_lat_ns;
                        let surprise_delta = (mutated_surp - baseline_surp) / baseline_surp;
                        let filter_delta =
                            (mutated_filter - baseline_filter) / (baseline_filter + 0.001);

                        let joint_reward = (WEIGHT_LATENCY * latency_delta)
                            - (WEIGHT_SURPRISE * surprise_delta)
                            - (WEIGHT_FILTER * filter_delta);

                        println!(
                            "   | Latency Metrics: Mutated = {:?} | Baseline = {:?}",
                            mutated_lat, baseline_lat
                        );
                        println!(
                            "   | Domotic Error:   Mutated = {:.4} | Baseline = {:.4}",
                            mutated_filter, baseline_filter
                        );
                        println!("   | Joint Reward Pareto Index Score: {:.4}", joint_reward);

                        if joint_reward > 0.001 {
                            println!(
                                "   🏆 [TRI-MODE WIN] Optimization matches Pareto frontier requirements!"
                            );
                            baseline_lat = mutated_lat;
                            baseline_surp = mutated_surp;
                            baseline_filter = mutated_filter;

                            let _ = Command::new("git").args(&["add", "."]).output()?;
                            let _ = Command::new("git")
                                .args(&[
                                    "commit",
                                    "-m",
                                    &format!(
                                        "evolution: tri-objective win [{}] val={:.4}",
                                        coord.field_key, target_val
                                    ),
                                ])
                                .output()?;
                        } else {
                            let _ = restore_file_buffer(&coord.id, &pristine_buffer);
                        }
                    } else {
                        let _ = restore_file_buffer(&coord.id, &pristine_buffer);
                    }
                } else {
                    let _ = restore_file_buffer(&coord.id, &pristine_buffer);
                }
                thread::sleep(Duration::from_millis(200));
            }
        }
        thread::sleep(Duration::from_secs(10));
    }
}
