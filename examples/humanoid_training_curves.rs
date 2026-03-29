// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Training curve visualization from humanoid telemetry CSV.
//!
//! Reads `output/humanoid_telemetry/summary.csv` and renders ASCII charts:
//! - Standing reward over episodes (with phase markers)
//! - Free energy over episodes
//! - Control effort over episodes
//!
//! Run: `cargo run --example humanoid_training_curves`

use std::io::BufRead;

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "output/humanoid_telemetry/summary.csv".to_string());

    let file = match std::fs::File::open(&path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Cannot open {path}: {e}");
            eprintln!("Run humanoid_100ep_telemetry first to generate data.");
            std::process::exit(1);
        }
    };

    let reader = std::io::BufReader::new(file);
    let mut episodes: Vec<EpisodeRow> = Vec::new();

    for (i, line) in reader.lines().enumerate() {
        let line = line.unwrap();
        if i == 0 {
            continue; // skip header
        }
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < 11 {
            continue;
        }
        episodes.push(EpisodeRow {
            episode: fields[0].parse().unwrap_or(0),
            task: fields[1].to_string(),
            standing_reward: fields[2].parse().unwrap_or(0.0),
            episode_reward: fields[3].parse().unwrap_or(0.0),
            free_energy: fields[4].parse().unwrap_or(0.0),
            head_height: fields[5].parse().unwrap_or(0.0),
            uprightness: fields[6].parse().unwrap_or(0.0),
            horizontal_speed: fields[7].parse().unwrap_or(0.0),
            control_effort: fields[8].parse().unwrap_or(0.0),
        });
    }

    if episodes.is_empty() {
        eprintln!("No data in {path}");
        std::process::exit(1);
    }

    println!(
        "=== Humanoid Training Curves ({} episodes) ===",
        episodes.len()
    );
    println!();

    // Standing Reward
    ascii_chart(
        "Standing Reward",
        &episodes,
        |e| e.standing_reward,
        0.0,
        1.0,
        40,
    );

    // Free Energy
    let max_fe = episodes
        .iter()
        .map(|e| e.free_energy)
        .fold(0.0f64, f64::max);
    ascii_chart(
        "Free Energy",
        &episodes,
        |e| e.free_energy,
        0.0,
        max_fe * 1.1,
        40,
    );

    // Control Effort
    let max_ctrl = episodes
        .iter()
        .map(|e| e.control_effort)
        .fold(0.0f64, f64::max);
    ascii_chart(
        "Control Effort",
        &episodes,
        |e| e.control_effort,
        0.0,
        max_ctrl * 1.1,
        40,
    );

    // Phase summary
    println!("Phase Transitions:");
    let mut prev_task = String::new();
    for e in &episodes {
        if e.task != prev_task {
            println!(
                "  Episode {:>3}: {} (R={:.4}, FE={:.3})",
                e.episode, e.task, e.standing_reward, e.free_energy
            );
            prev_task = e.task.clone();
        }
    }
}

#[allow(dead_code)]
struct EpisodeRow {
    episode: usize,
    task: String,
    standing_reward: f64,
    episode_reward: f64,
    free_energy: f64,
    head_height: f64,
    uprightness: f64,
    horizontal_speed: f64,
    control_effort: f64,
}

fn ascii_chart(
    title: &str,
    episodes: &[EpisodeRow],
    value_fn: impl Fn(&EpisodeRow) -> f64,
    y_min: f64,
    y_max: f64,
    height: usize,
) {
    println!("{title}");
    println!("{}", "=".repeat(title.len()));

    let width = episodes.len().min(100);
    let y_range = (y_max - y_min).max(1e-6);

    // Build grid
    let mut grid = vec![vec![' '; width]; height];

    // Plot data points
    let mut prev_task = String::new();
    for (col, ep) in episodes.iter().enumerate().take(width) {
        let val = value_fn(ep);
        let row = ((val - y_min) / y_range * (height - 1) as f64).clamp(0.0, (height - 1) as f64)
            as usize;
        let row = height - 1 - row; // invert for display

        let ch = match ep.task.as_str() {
            "Stand" => '#',
            "Walk" => 'W',
            "Run" => 'R',
            _ => '*',
        };
        grid[row][col] = ch;

        // Mark phase transitions
        if ep.task != prev_task && col > 0 {
            for row in grid.iter_mut().take(height) {
                if row[col] == ' ' {
                    row[col] = '|';
                }
            }
        }
        prev_task = ep.task.clone();
    }

    // Render
    for (r, row) in grid.iter().enumerate() {
        let y_val = y_max - (r as f64 / (height - 1) as f64) * y_range;
        let line: String = row.iter().collect();
        if r == 0 || r == height - 1 || r == height / 2 {
            print!("{:>6.3} |", y_val);
        } else {
            print!("       |");
        }
        println!("{line}");
    }

    // X-axis
    print!("       +");
    println!("{}", "-".repeat(width));
    println!("        0{:>width$}", episodes.len() - 1, width = width - 1);
    println!();
}
