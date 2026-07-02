// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! CelesTrak Demo Tool
//!
//! Demonstrates the mycelix-space orbital mechanics library with real data
//! from CelesTrak (no authentication required).
//!
//! Usage:
//!   celestrak-demo fetch-iss       - Get ISS TLE
//!   celestrak-demo fetch-starlink  - Get Starlink TLEs
//!   celestrak-demo screen          - Screen ISS for conjunctions
//!   celestrak-demo propagate       - Propagate ISS position
//!   celestrak-demo ingest          - Ingest data into Holochain

#[allow(dead_code)]
mod holochain_client;

use anyhow::{Context, Result};
use chrono::{DateTime, Duration, Utc};
use clap::{Parser, Subcommand};
use colored::*;
use holochain_client::{
    GroundLocation, HolochainClient, HolochainConfig, IngestionBatch, Measurement,
    OrbitalObjectInput, SpaceTimestamp, StateVectorInput, SubmitObservationCall, TleInput,
};
use orbital_mechanics::{
    conjunction::{ConjunctionAnalyzer, RiskLevel},
    covariance::CovarianceMatrix,
    propagator::Propagator,
    state::{DataSource, OrbitalState, StateVector},
    tle::TwoLineElement,
};
use serde::Serialize;
use std::collections::HashMap;

/// CelesTrak base URL for GP data
const CELESTRAK_GP_URL: &str = "https://celestrak.org/NORAD/elements/gp.php";

#[derive(Parser)]
#[command(name = "celestrak-demo")]
#[command(about = "Demo tool for mycelix-space orbital mechanics", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Fetch ISS (Zarya) TLE
    FetchIss,

    /// Fetch active Starlink satellites
    FetchStarlink {
        /// Maximum number to fetch
        #[arg(short, long, default_value = "20")]
        limit: usize,
    },

    /// Fetch space debris catalog
    FetchDebris {
        /// Maximum number to fetch
        #[arg(short, long, default_value = "50")]
        limit: usize,
    },

    /// Propagate ISS position over time
    Propagate {
        /// Hours to propagate
        #[arg(short, long, default_value = "2")]
        hours: f64,

        /// Step size in minutes
        #[arg(short, long, default_value = "10")]
        step: f64,
    },

    /// Screen ISS for close approaches with debris
    Screen {
        /// Screening threshold in km
        #[arg(short, long, default_value = "10.0")]
        threshold: f64,

        /// Number of debris objects to check
        #[arg(short, long, default_value = "100")]
        debris_count: usize,
    },

    /// Demonstrate conjunction analysis
    ConjunctionDemo,

    /// Ingest TLE data — writes JSON files for Holochain batch import
    Ingest {
        /// Source: "iss", "starlink", "debris", or "all"
        #[arg(short, long, default_value = "iss")]
        source: String,

        /// Maximum objects to ingest
        #[arg(short, long, default_value = "10")]
        limit: usize,

        /// Output directory for batch JSON files
        #[arg(long, default_value = "import_batch")]
        batch_dir: String,
    },

    /// Export TLE data as JSON for offline ingestion
    Export {
        /// Source: "iss", "starlink", "debris", or "all"
        #[arg(short, long, default_value = "all")]
        source: String,

        /// Maximum objects to export
        #[arg(short, long, default_value = "100")]
        limit: usize,

        /// Output file path
        #[arg(short, long, default_value = "orbital_data.json")]
        output: String,
    },

    /// Live-ingest: fetch catalog, compare with local state, detect changes
    LiveIngest {
        /// Persistent state file path
        #[arg(long, default_value = "catalog_state.json")]
        state_file: String,

        /// CelesTrak GP categories to fetch (comma-separated)
        #[arg(long, default_value = "active,cosmos-1408-debris")]
        categories: String,

        /// Show changes without saving
        #[arg(long)]
        dry_run: bool,
    },

    /// Print cron-compatible schedule entry for periodic ingestion
    Schedule {
        /// Interval in hours between runs
        #[arg(short, long, default_value = "6")]
        interval_hours: u32,
    },

    /// Print catalog statistics
    Stats {
        /// State file to read statistics from
        #[arg(long, default_value = "catalog_state.json")]
        state_file: String,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::FetchIss => fetch_iss()?,
        Commands::FetchStarlink { limit } => fetch_starlink(limit)?,
        Commands::FetchDebris { limit } => fetch_debris(limit)?,
        Commands::Propagate { hours, step } => propagate_iss(hours, step)?,
        Commands::Screen {
            threshold,
            debris_count,
        } => screen_iss(threshold, debris_count)?,
        Commands::ConjunctionDemo => conjunction_demo()?,
        Commands::Ingest {
            source,
            limit,
            batch_dir,
        } => {
            ingest_data(&source, limit, &batch_dir)?;
        }
        Commands::Export {
            source,
            limit,
            output,
        } => export_data(&source, limit, &output)?,
        Commands::LiveIngest {
            state_file,
            categories,
            dry_run,
        } => {
            live_ingest(&state_file, &categories, dry_run)?;
        }
        Commands::Schedule { interval_hours } => print_schedule(interval_hours),
        Commands::Stats { state_file } => print_stats(&state_file)?,
    }

    Ok(())
}

/// Fetch ISS TLE from CelesTrak
fn fetch_iss() -> Result<()> {
    println!(
        "{}",
        "=== Fetching ISS TLE from CelesTrak ===".green().bold()
    );

    let url = format!("{}?CATNR=25544&FORMAT=TLE", CELESTRAK_GP_URL);
    let response = reqwest::blocking::get(&url)
        .context("Failed to fetch ISS TLE")?
        .text()
        .context("Failed to read response")?;

    let lines: Vec<&str> = response.lines().collect();
    if lines.len() < 3 {
        anyhow::bail!("Invalid TLE response");
    }

    let name = lines[0].trim();
    let line1 = lines[1].trim();
    let line2 = lines[2].trim();

    println!("\n{}: {}", "Object".cyan(), name);
    println!("{}: {}", "Line 1".cyan(), line1);
    println!("{}: {}", "Line 2".cyan(), line2);

    // Parse TLE
    let tle = TwoLineElement::parse_lines(Some(name.to_string()), line1, line2)
        .context("Failed to parse TLE")?;

    println!("\n{}", "Parsed TLE Data:".yellow());
    println!("  NORAD ID: {}", tle.norad_id);
    println!("  Epoch: {}", tle.epoch);
    println!("  Inclination: {:.2}°", tle.inclination_deg);
    println!("  RAAN: {:.2}°", tle.raan_deg);
    println!("  Eccentricity: {:.7}", tle.eccentricity);
    println!("  Arg of Perigee: {:.2}°", tle.arg_of_perigee_deg);
    println!("  Mean Anomaly: {:.2}°", tle.mean_anomaly_deg);
    println!("  Mean Motion: {:.8} rev/day", tle.mean_motion);

    // Calculate altitude using TLE methods
    let semi_major = tle.semi_major_axis_km();
    let perigee = tle.perigee_km();
    let apogee = tle.apogee_km();

    println!("\n{}", "Orbit Parameters:".yellow());
    println!("  Semi-major axis: {:.2} km", semi_major);
    println!("  Perigee altitude: {:.2} km", perigee);
    println!("  Apogee altitude: {:.2} km", apogee);
    println!("  Period: {:.2} minutes", tle.period_minutes());

    Ok(())
}

/// Fetch Starlink TLEs
fn fetch_starlink(limit: usize) -> Result<()> {
    println!(
        "{}",
        format!("=== Fetching {} Starlink TLEs ===", limit)
            .green()
            .bold()
    );

    let url = format!("{}?GROUP=starlink&FORMAT=TLE", CELESTRAK_GP_URL);
    let response = reqwest::blocking::get(&url)
        .context("Failed to fetch Starlink TLEs")?
        .text()
        .context("Failed to read response")?;

    let lines: Vec<&str> = response.lines().collect();
    let mut count = 0;

    for chunk in lines.chunks(3) {
        if chunk.len() < 3 || count >= limit {
            break;
        }

        let name = chunk[0].trim();
        let line1 = chunk[1].trim();
        let line2 = chunk[2].trim();

        if let Ok(tle) = TwoLineElement::parse_lines(Some(name.to_string()), line1, line2) {
            let altitude = tle.semi_major_axis_km() - 6378.137;

            println!(
                "{:5} | {} | Alt: {:.0} km | Inc: {:.1}°",
                tle.norad_id, name, altitude, tle.inclination_deg
            );
            count += 1;
        }
    }

    println!(
        "\n{} Starlink satellites fetched",
        count.to_string().green()
    );

    Ok(())
}

/// Fetch debris TLEs
fn fetch_debris(limit: usize) -> Result<()> {
    println!(
        "{}",
        format!("=== Fetching {} Debris Objects ===", limit)
            .green()
            .bold()
    );

    // CelesTrak debris catalog (COSMOS 1408 debris from ASAT test)
    let url = format!("{}?GROUP=cosmos-1408-debris&FORMAT=TLE", CELESTRAK_GP_URL);
    let response = reqwest::blocking::get(&url)
        .context("Failed to fetch debris TLEs")?
        .text()
        .context("Failed to read response")?;

    let lines: Vec<&str> = response.lines().collect();
    let mut count = 0;

    println!(
        "\n{:<10} | {:<25} | {:>8} | {:>6}",
        "NORAD ID", "Name", "Alt (km)", "Inc"
    );
    println!("{}", "-".repeat(60));

    for chunk in lines.chunks(3) {
        if chunk.len() < 3 || count >= limit {
            break;
        }

        let name = chunk[0].trim();
        let line1 = chunk[1].trim();
        let line2 = chunk[2].trim();

        if let Ok(tle) = TwoLineElement::parse_lines(Some(name.to_string()), line1, line2) {
            let altitude = tle.semi_major_axis_km() - 6378.137;
            let display_name = if name.len() > 25 { &name[..25] } else { name };

            println!(
                "{:<10} | {:<25} | {:>8.0} | {:>6.1}°",
                tle.norad_id, display_name, altitude, tle.inclination_deg
            );
            count += 1;
        }
    }

    println!("\n{} debris objects fetched", count.to_string().yellow());

    Ok(())
}

/// Propagate ISS position
fn propagate_iss(hours: f64, step_minutes: f64) -> Result<()> {
    println!(
        "{}",
        format!("=== Propagating ISS for {} hours ===", hours)
            .green()
            .bold()
    );

    // Fetch ISS TLE
    let url = format!("{}?CATNR=25544&FORMAT=TLE", CELESTRAK_GP_URL);
    let response = reqwest::blocking::get(&url)?.text()?;

    let lines: Vec<&str> = response.lines().collect();
    if lines.len() < 3 {
        anyhow::bail!("Invalid TLE response");
    }

    let tle = TwoLineElement::parse_lines(
        Some(lines[0].trim().to_string()),
        lines[1].trim(),
        lines[2].trim(),
    )?;

    let propagator = Propagator::from_tle(&tle)?;

    println!(
        "\n{:<8} | {:>10} | {:>10} | {:>10} | {:>8} | {:>8}",
        "Time", "X (km)", "Y (km)", "Z (km)", "Alt (km)", "Speed"
    );
    println!("{}", "-".repeat(70));

    let start = Utc::now();
    let mut current = start;
    let end = start + Duration::minutes((hours * 60.0) as i64);

    while current <= end {
        if let Ok(state) = propagator.propagate_to(current) {
            let alt = state.state.altitude_km();
            let speed = state.state.speed();

            let time_offset = (current - start).num_minutes();
            println!(
                "{:>4} min | {:>10.1} | {:>10.1} | {:>10.1} | {:>8.1} | {:>6.2} km/s",
                time_offset, state.state.x, state.state.y, state.state.z, alt, speed
            );
        }

        current += Duration::minutes(step_minutes as i64);
    }

    Ok(())
}

/// Screen ISS for close approaches with Alfano Pc computation
fn screen_iss(threshold_km: f64, debris_count: usize) -> Result<()> {
    println!(
        "{}",
        "=== ISS Conjunction Screening Report ===".green().bold()
    );
    println!(
        "Threshold: {} km | Objects to screen: {}",
        threshold_km, debris_count
    );

    // Fetch ISS TLE
    println!("\n{}", "Fetching ISS TLE...".cyan());
    let iss_url = format!("{}?CATNR=25544&FORMAT=TLE", CELESTRAK_GP_URL);
    let iss_response = reqwest::blocking::get(&iss_url)?.text()?;
    let iss_lines: Vec<&str> = iss_response.lines().collect();
    if iss_lines.len() < 3 {
        anyhow::bail!("Invalid ISS TLE response");
    }

    let iss_tle = TwoLineElement::parse_lines(
        Some(iss_lines[0].trim().to_string()),
        iss_lines[1].trim(),
        iss_lines[2].trim(),
    )?;

    // TLE age
    let tle_age_hours = (Utc::now() - iss_tle.epoch).num_minutes() as f64 / 60.0;
    println!(
        "  NORAD ID: {} | Epoch: {} | Age: {:.1}h",
        iss_tle.norad_id,
        iss_tle.epoch.format("%Y-%m-%d %H:%M UTC"),
        tle_age_hours
    );

    // Fetch debris TLEs (COSMOS 1408 ASAT debris)
    println!("\n{}", "Fetching COSMOS 1408 debris catalog...".cyan());
    let debris_url = format!("{}?GROUP=cosmos-1408-debris&FORMAT=TLE", CELESTRAK_GP_URL);
    let debris_response = reqwest::blocking::get(&debris_url)?.text()?;
    let debris_lines: Vec<&str> = debris_response.lines().collect();

    // Parse all debris TLEs
    let mut debris_tles: Vec<(String, TwoLineElement)> = Vec::new();
    for chunk in debris_lines.chunks(3) {
        if chunk.len() < 3 || debris_tles.len() >= debris_count {
            break;
        }
        let name = chunk[0].trim();
        let line1 = chunk[1].trim();
        let line2 = chunk[2].trim();
        if let Ok(tle) = TwoLineElement::parse_lines(Some(name.to_string()), line1, line2) {
            debris_tles.push((name.to_string(), tle));
        }
    }
    println!("  Parsed {} debris TLEs", debris_tles.len());

    // Build ISS propagator
    let iss_prop = Propagator::from_tle(&iss_tle)?;
    let now = Utc::now();
    let iss_state = iss_prop.propagate_to(now)?;

    println!("\n{}", "ISS current state:".yellow());
    println!(
        "  Position: [{:.1}, {:.1}, {:.1}] km (TEME)",
        iss_state.state.x, iss_state.state.y, iss_state.state.z
    );
    println!(
        "  Velocity: [{:.3}, {:.3}, {:.3}] km/s",
        iss_state.state.vx, iss_state.state.vy, iss_state.state.vz
    );
    println!(
        "  Altitude: {:.1} km | Speed: {:.3} km/s",
        iss_state.state.altitude_km(),
        iss_state.state.speed()
    );

    // Screen: 24-hour window, coarse 5-min sweep then 10s refinement
    let screening_hours = 24.0;
    let coarse_step_min = 5.0;
    let fine_step_sec = 10.0;

    println!(
        "\n{}",
        format!(
            "Screening {} objects over {} hours...",
            debris_tles.len(),
            screening_hours
        )
        .yellow()
    );

    let analyzer = ConjunctionAnalyzer::new().with_hbr(20.0); // 20m combined HBR

    struct ScreenResult {
        name: String,
        norad_id: u32,
        tca: DateTime<Utc>,
        miss_km: f64,
        rel_vel_kms: f64,
        pc: f64,
        risk: RiskLevel,
    }

    let mut results: Vec<ScreenResult> = Vec::new();
    let mut screened = 0;
    let mut parse_errors = 0;

    for (name, debris_tle) in &debris_tles {
        screened += 1;

        let debris_prop = match Propagator::from_tle(debris_tle) {
            Ok(p) => p,
            Err(_) => {
                parse_errors += 1;
                continue;
            }
        };

        // Coarse sweep: find closest approach time
        let steps = (screening_hours * 60.0 / coarse_step_min) as i64;
        let mut min_dist = f64::MAX;
        let mut min_time = now;

        for step in 0..=steps {
            let t = now + Duration::minutes((step as f64 * coarse_step_min) as i64);
            let (Ok(iss_s), Ok(deb_s)) = (iss_prop.propagate_to(t), debris_prop.propagate_to(t))
            else {
                continue;
            };
            let dist = iss_s.state.distance_to(&deb_s.state);
            if dist < min_dist {
                min_dist = dist;
                min_time = t;
            }
        }

        // Only refine if within screening volume
        if min_dist > threshold_km {
            continue;
        }

        // Fine refinement: 10-second steps around the closest approach
        let refine_start = min_time - Duration::minutes(coarse_step_min as i64);
        let refine_end = min_time + Duration::minutes(coarse_step_min as i64);
        let fine_steps = ((refine_end - refine_start).num_seconds() as f64 / fine_step_sec) as i64;

        let mut best_dist = f64::MAX;
        let mut best_time = min_time;
        let mut best_iss_state = None;
        let mut best_deb_state = None;

        for step in 0..=fine_steps {
            let t = refine_start + Duration::seconds((step as f64 * fine_step_sec) as i64);
            let (Ok(iss_s), Ok(deb_s)) = (iss_prop.propagate_to(t), debris_prop.propagate_to(t))
            else {
                continue;
            };
            let dist = iss_s.state.distance_to(&deb_s.state);
            if dist < best_dist {
                best_dist = dist;
                best_time = t;
                best_iss_state = Some(iss_s);
                best_deb_state = Some(deb_s);
            }
        }

        // Compute Alfano Pc
        if let (Some(iss_s), Some(deb_s)) = (best_iss_state, best_deb_state) {
            let assessment = analyzer.assess(&iss_s, &deb_s);
            let rel_vel = assessment.relative_velocity_kms;
            let pc = assessment.collision_probability.pc;

            results.push(ScreenResult {
                name: name.clone(),
                norad_id: debris_tle.norad_id,
                tca: best_time,
                miss_km: best_dist,
                rel_vel_kms: rel_vel,
                pc,
                risk: assessment.risk_level,
            });
        }
    }

    // Sort by Pc descending
    results.sort_by(|a, b| b.pc.partial_cmp(&a.pc).unwrap_or(std::cmp::Ordering::Equal));

    // Print report
    println!(
        "\n{}",
        "=== CONJUNCTION SCREENING REPORT ===".green().bold()
    );
    println!(
        "Primary: ISS (NORAD {}) | Window: {} hours | HBR: 20m",
        iss_tle.norad_id, screening_hours
    );
    println!(
        "Screened: {} objects | Threshold: {} km | Parse errors: {}",
        screened, threshold_km, parse_errors
    );

    if results.is_empty() {
        println!(
            "\n{}",
            "  No conjunctions found within screening volume."
                .green()
                .bold()
        );
    } else {
        println!(
            "\n{} conjunctions found:\n",
            results.len().to_string().yellow().bold()
        );
        println!(
            "{:<8} | {:<22} | {:>19} | {:>9} | {:>9} | {:>10} | Risk",
            "NORAD", "Name", "TCA (UTC)", "Miss (km)", "Vrel(km/s)", "Pc"
        );
        println!("{}", "-".repeat(105));

        for r in &results {
            let display_name = if r.name.len() > 22 {
                &r.name[..22]
            } else {
                &r.name
            };
            let tca_str = r.tca.format("%Y-%m-%d %H:%M:%S").to_string();
            let pc_str = format!("{:.2e}", r.pc);

            let risk_str = format!("{:?}", r.risk);
            let colored_risk = match r.risk {
                RiskLevel::Emergency => risk_str.red().bold(),
                RiskLevel::High => risk_str.red(),
                RiskLevel::Medium => risk_str.yellow(),
                RiskLevel::Low => risk_str.green(),
                RiskLevel::Negligible => risk_str.normal(),
            };

            println!(
                "{:<8} | {:<22} | {:>19} | {:>9.3} | {:>9.3} | {:>10} | {}",
                r.norad_id, display_name, tca_str, r.miss_km, r.rel_vel_kms, pc_str, colored_risk
            );
        }
    }

    println!("\n{}", "Report complete.".green());

    Ok(())
}

/// Demonstrate conjunction analysis
fn conjunction_demo() -> Result<()> {
    println!("{}", "=== Conjunction Analysis Demo ===".green().bold());

    let now = Utc::now();

    // Create two hypothetical objects close together
    let primary = OrbitalState::new(
        25544, // ISS
        now,
        StateVector::new(6800.0, 0.0, 0.0, 0.0, 7.66, 0.0),
        DataSource::SpaceTrack,
    )
    .with_covariance(CovarianceMatrix::diagonal([
        0.5, 0.5, 0.5, 0.001, 0.001, 0.001,
    ]));

    // Secondary 0.5 km away
    let secondary = OrbitalState::new(
        99999, // Hypothetical debris
        now,
        StateVector::new(6800.5, 0.0, 0.0, 0.0, 7.66, 0.0),
        DataSource::SpaceTrack,
    )
    .with_covariance(CovarianceMatrix::diagonal([
        1.0, 1.0, 1.0, 0.002, 0.002, 0.002,
    ]));

    let analyzer = ConjunctionAnalyzer::new().with_hbr(20.0); // 20m combined radius
    let assessment = analyzer.assess(&primary, &secondary);

    println!("\n{}", "Conjunction Assessment:".cyan());
    println!("  Primary NORAD ID: {}", assessment.primary_norad_id);
    println!("  Secondary NORAD ID: {}", assessment.secondary_norad_id);
    println!("  Time of Closest Approach: {}", assessment.tca);
    println!("  Miss Distance: {:.3} km", assessment.miss_distance_km);
    println!(
        "  Relative Velocity: {:.3} km/s",
        assessment.relative_velocity_kms
    );
    println!("  Hard Body Radius: {:.1} m", assessment.hard_body_radius_m);

    println!("\n{}", "Collision Probability:".yellow());
    println!("  Pc: {:.2e}", assessment.collision_probability.pc);
    println!(
        "  Pc Lower: {:.2e}",
        assessment.collision_probability.pc_lower
    );
    println!(
        "  Pc Upper: {:.2e}",
        assessment.collision_probability.pc_upper
    );
    println!("  Method: {:?}", assessment.collision_probability.method);
    println!(
        "  Has Covariance: {}",
        assessment.collision_probability.has_covariance
    );

    let risk_color = match assessment.risk_level {
        RiskLevel::Emergency => "red",
        RiskLevel::High => "red",
        RiskLevel::Medium => "yellow",
        RiskLevel::Low => "green",
        RiskLevel::Negligible => "green",
    };

    let risk_str = format!("{:?}", assessment.risk_level);
    println!("\n{}", "Risk Assessment:".cyan());
    println!(
        "  Level: {}",
        match risk_color {
            "red" => risk_str.red().bold(),
            "yellow" => risk_str.yellow().bold(),
            _ => risk_str.green().bold(),
        }
    );
    println!("  Recommendation: {}", assessment.risk_level.description());

    Ok(())
}

/// Ingest TLE data — writes JSON files for Holochain batch import
fn ingest_data(source: &str, limit: usize, batch_dir: &str) -> Result<()> {
    println!(
        "{}",
        format!(
            "=== Ingesting {} data (writing to {}) ===",
            source, batch_dir
        )
        .green()
        .bold()
    );

    // Collect TLEs based on source
    let tles = match source.to_lowercase().as_str() {
        "iss" => fetch_tles_for_ingest("CATNR=25544", 1)?,
        "starlink" => fetch_tles_for_ingest("GROUP=starlink", limit)?,
        "debris" => fetch_tles_for_ingest("GROUP=cosmos-1408-debris", limit)?,
        "all" => {
            let mut all = fetch_tles_for_ingest("CATNR=25544", 1)?;
            all.extend(fetch_tles_for_ingest("GROUP=starlink", limit / 2)?);
            all.extend(fetch_tles_for_ingest(
                "GROUP=cosmos-1408-debris",
                limit / 2,
            )?);
            all
        }
        _ => anyhow::bail!("Unknown source. Use 'iss', 'starlink', 'debris', or 'all'"),
    };

    println!("\nFetched {} TLEs from CelesTrak", tles.len());

    // Build ingestion batch
    let mut batch = IngestionBatch::new();

    for (name, tle) in &tles {
        let object_type = if name.contains("DEB") || name.contains("debris") {
            "Debris"
        } else if name.contains("R/B") {
            "RocketBody"
        } else {
            "Payload"
        };

        batch = batch.add_object(OrbitalObjectInput {
            norad_id: tle.norad_id,
            name: name.clone(),
            object_type: object_type.to_string(),
            launch_date: None,
            decay_date: None,
            owner_country: None,
            data_source: "CelesTrak".to_string(),
            metadata: HashMap::new(),
        });

        batch = batch.add_tle(TleInput {
            norad_id: tle.norad_id,
            line1: tle.line1.clone(),
            line2: tle.line2.clone(),
            epoch: tle.epoch,
            source: "CelesTrak".to_string(),
        });

        if let Ok(propagator) = Propagator::from_tle(tle) {
            if let Ok(state) = propagator.propagate_to(Utc::now()) {
                let pos = [state.state.x, state.state.y, state.state.z];
                let vel = [state.state.vx, state.state.vy, state.state.vz];

                batch = batch.add_state_vector(StateVectorInput {
                    norad_id: tle.norad_id,
                    epoch: Utc::now(),
                    position_km: pos,
                    velocity_kms: vel,
                    covariance: None,
                    reference_frame: "Teme".to_string(),
                    quality: 0.9,
                    source: "CelesTrak".to_string(),
                });

                // Also generate an observation for the observations zome
                batch = batch.add_observation(SubmitObservationCall {
                    norad_id: Some(tle.norad_id),
                    observation_time: SpaceTimestamp::from_datetime(Utc::now()),
                    observer_location: Some(GroundLocation {
                        latitude_deg: 0.0,
                        longitude_deg: 0.0,
                        altitude_m: 0.0,
                        name: Some("CelesTrak SGP4".to_string()),
                    }),
                    observation_type: "Radar".to_string(),
                    measurement: Measurement::StateVector {
                        position_km: pos,
                        velocity_kms: Some(vel),
                        covariance: None,
                    },
                    quality: 85,
                    sensor_id: "celestrak-sgp4".to_string(),
                });
            }
        }
    }

    println!("\nPrepared batch:");
    println!("  Orbital Objects: {}", batch.object_count());
    println!("  TLEs: {}", batch.tle_count());
    println!("  State Vectors: {}", batch.state_vector_count());
    println!("  Observations: {}", batch.observation_count());

    // Initialize client and write batch files
    let config = HolochainConfig {
        batch_dir: std::path::PathBuf::from(batch_dir),
        ..Default::default()
    };

    let mut client = HolochainClient::new(config);
    client.init()?;

    let report = batch.write_batch(&client)?;
    report.print_summary();

    Ok(())
}

/// Export data to JSON file for offline ingestion
fn export_data(source: &str, limit: usize, output: &str) -> Result<()> {
    println!(
        "{}",
        format!("=== Exporting {} data to {} ===", source, output)
            .green()
            .bold()
    );

    // Collect TLEs based on source
    let tles = match source.to_lowercase().as_str() {
        "iss" => fetch_tles_for_ingest("CATNR=25544", 1)?,
        "starlink" => fetch_tles_for_ingest("GROUP=starlink", limit)?,
        "debris" => fetch_tles_for_ingest("GROUP=cosmos-1408-debris", limit)?,
        "all" => {
            let mut all = fetch_tles_for_ingest("CATNR=25544", 1)?;
            all.extend(fetch_tles_for_ingest("GROUP=starlink", limit / 2)?);
            all.extend(fetch_tles_for_ingest(
                "GROUP=cosmos-1408-debris",
                limit / 2,
            )?);
            all
        }
        _ => anyhow::bail!(
            "Unknown source: {}. Use 'iss', 'starlink', 'debris', or 'all'",
            source
        ),
    };

    println!("Fetched {} TLEs from CelesTrak", tles.len());

    // Build export data
    #[derive(Serialize)]
    struct ExportData {
        generated_at: DateTime<Utc>,
        source: String,
        count: usize,
        objects: Vec<ExportObject>,
    }

    #[derive(Serialize)]
    struct ExportObject {
        norad_id: u32,
        name: String,
        object_type: String,
        tle: ExportTle,
        state_vector: Option<ExportStateVector>,
    }

    #[derive(Serialize)]
    struct ExportTle {
        line1: String,
        line2: String,
        epoch: DateTime<Utc>,
    }

    #[derive(Serialize)]
    struct ExportStateVector {
        epoch: DateTime<Utc>,
        position_km: [f64; 3],
        velocity_kms: [f64; 3],
        reference_frame: String,
    }

    let mut objects = Vec::new();

    for (name, tle) in &tles {
        let object_type = if name.contains("DEB") || name.contains("debris") {
            "Debris"
        } else if name.contains("R/B") {
            "RocketBody"
        } else {
            "Payload"
        };

        let state_vector = if let Ok(propagator) = Propagator::from_tle(tle) {
            propagator
                .propagate_to(Utc::now())
                .ok()
                .map(|state| ExportStateVector {
                    epoch: Utc::now(),
                    position_km: [state.state.x, state.state.y, state.state.z],
                    velocity_kms: [state.state.vx, state.state.vy, state.state.vz],
                    reference_frame: "Teme".to_string(),
                })
        } else {
            None
        };

        objects.push(ExportObject {
            norad_id: tle.norad_id,
            name: name.clone(),
            object_type: object_type.to_string(),
            tle: ExportTle {
                line1: tle.line1.clone(),
                line2: tle.line2.clone(),
                epoch: tle.epoch,
            },
            state_vector,
        });
    }

    let export = ExportData {
        generated_at: Utc::now(),
        source: source.to_string(),
        count: objects.len(),
        objects,
    };

    let json = serde_json::to_string_pretty(&export)?;
    std::fs::write(output, &json)?;

    println!(
        "\n{}",
        format!("Exported {} objects to {}", export.count, output).green()
    );
    println!("File size: {} bytes", json.len());

    Ok(())
}

// =============================================================================
// Catalog State for Live Ingestion
// =============================================================================

/// Persistent catalog state for delta detection
#[derive(Serialize, serde::Deserialize, Default)]
struct CatalogState {
    last_updated: Option<DateTime<Utc>>,
    objects: HashMap<u32, CatalogEntry>,
}

#[derive(Serialize, serde::Deserialize, Clone)]
struct CatalogEntry {
    norad_id: u32,
    name: String,
    object_type: String,
    epoch: DateTime<Utc>,
    line1: String,
    line2: String,
    first_seen: DateTime<Utc>,
    last_updated: DateTime<Utc>,
}

/// Live-ingest: fetch catalog, compare with local state, detect new/updated/decayed objects
fn live_ingest(state_file: &str, categories: &str, dry_run: bool) -> Result<()> {
    println!("{}", "=== Live Catalog Ingestion ===".green().bold());

    // Load existing state
    let mut state = if std::path::Path::new(state_file).exists() {
        let data = std::fs::read_to_string(state_file).context("Failed to read state file")?;
        serde_json::from_str::<CatalogState>(&data).context("Failed to parse state file")?
    } else {
        CatalogState::default()
    };

    if let Some(last) = state.last_updated {
        let age_hours = (Utc::now() - last).num_minutes() as f64 / 60.0;
        println!(
            "Last update: {} ({:.1}h ago)",
            last.format("%Y-%m-%d %H:%M UTC"),
            age_hours
        );
    } else {
        println!("First run — no previous state");
    }

    let previous_ids: std::collections::HashSet<u32> = state.objects.keys().copied().collect();
    println!("Previous catalog: {} objects", previous_ids.len());

    // Fetch from each category
    let cats: Vec<&str> = categories.split(',').map(|s| s.trim()).collect();
    let mut fetched_tles: Vec<(String, TwoLineElement)> = Vec::new();

    for cat in &cats {
        let query = if *cat == "active" {
            "GROUP=active".to_string()
        } else {
            format!("GROUP={}", cat)
        };
        println!("\n{}", format!("Fetching category: {} ...", cat).cyan());

        match fetch_tles_for_ingest(&query, 10000) {
            Ok(tles) => {
                println!("  Retrieved {} TLEs", tles.len());
                fetched_tles.extend(tles);
            }
            Err(e) => {
                println!("  {} Failed: {}", "Warning:".yellow(), e);
            }
        }
    }

    // Compute delta
    let mut new_count = 0;
    let mut updated_count = 0;
    let mut unchanged_count = 0;
    let fetched_ids: std::collections::HashSet<u32> =
        fetched_tles.iter().map(|(_, tle)| tle.norad_id).collect();

    let now = Utc::now();

    for (name, tle) in &fetched_tles {
        let object_type = if name.contains("DEB") || name.contains("debris") {
            "Debris"
        } else if name.contains("R/B") {
            "RocketBody"
        } else {
            "Payload"
        };

        if let Some(existing) = state.objects.get(&tle.norad_id) {
            // Check if TLE epoch changed
            if existing.epoch != tle.epoch {
                updated_count += 1;
                if !dry_run {
                    let entry = state.objects.get_mut(&tle.norad_id).unwrap();
                    entry.epoch = tle.epoch;
                    entry.line1 = tle.line1.clone();
                    entry.line2 = tle.line2.clone();
                    entry.last_updated = now;
                    entry.name = name.clone();
                }
            } else {
                unchanged_count += 1;
            }
        } else {
            new_count += 1;
            if !dry_run {
                state.objects.insert(
                    tle.norad_id,
                    CatalogEntry {
                        norad_id: tle.norad_id,
                        name: name.clone(),
                        object_type: object_type.to_string(),
                        epoch: tle.epoch,
                        line1: tle.line1.clone(),
                        line2: tle.line2.clone(),
                        first_seen: now,
                        last_updated: now,
                    },
                );
            }
        }
    }

    // Detect decayed (in state but not in fetch)
    let decayed: Vec<u32> = previous_ids.difference(&fetched_ids).copied().collect();

    // Print summary
    println!("\n{}", "=== Delta Summary ===".yellow().bold());
    println!(
        "  Fetched:   {} objects from {} categories",
        fetched_tles.len(),
        cats.len()
    );
    println!(
        "  {} new objects",
        format!("{:>4}", new_count).green().bold()
    );
    println!("  {} TLE updates", format!("{:>4}", updated_count).cyan());
    println!("  {:>4} unchanged", unchanged_count);
    println!(
        "  {} potentially decayed/removed",
        format!("{:>4}", decayed.len()).red()
    );

    if !decayed.is_empty() && decayed.len() <= 20 {
        println!("\n  Decayed NORAD IDs: {:?}", decayed);
    }

    if dry_run {
        println!("\n{}", "  [DRY RUN] No changes saved.".yellow());
    } else {
        state.last_updated = Some(now);
        let json = serde_json::to_string_pretty(&state)?;
        std::fs::write(state_file, &json)?;
        println!("\n  State saved to {}", state_file);
        println!("  Catalog now contains {} objects", state.objects.len());
    }

    Ok(())
}

/// Print a cron-compatible schedule entry
fn print_schedule(interval_hours: u32) {
    println!("{}", "=== Cron Schedule ===".green().bold());

    let cron_expr = match interval_hours {
        1 => "0 * * * *".to_string(),
        2 => "0 */2 * * *".to_string(),
        4 => "0 */4 * * *".to_string(),
        6 => "0 */6 * * *".to_string(),
        8 => "0 */8 * * *".to_string(),
        12 => "0 */12 * * *".to_string(),
        24 => "0 0 * * *".to_string(),
        _ => format!("0 */{} * * *", interval_hours),
    };

    let bin_path = std::env::current_exe()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| "celestrak-demo".to_string());

    println!("\nAdd this to your crontab (crontab -e):\n");
    println!(
        "  {} {} live-ingest --state-file /var/lib/mycelix-space/catalog.json",
        cron_expr, bin_path
    );
    println!();
    println!(
        "Interval: every {} hour{}",
        interval_hours,
        if interval_hours == 1 { "" } else { "s" }
    );
    println!("\nCelesTrak rate limits: ~1 request/second, no auth required.");
    println!("Recommended interval: 6 hours (4 updates/day).");
}

/// Print catalog statistics from state file
fn print_stats(state_file: &str) -> Result<()> {
    println!("{}", "=== Catalog Statistics ===".green().bold());

    if !std::path::Path::new(state_file).exists() {
        println!(
            "\n{}",
            "No state file found. Run 'live-ingest' first.".yellow()
        );
        return Ok(());
    }

    let data = std::fs::read_to_string(state_file).context("Failed to read state file")?;
    let state: CatalogState = serde_json::from_str(&data).context("Failed to parse state file")?;

    let total = state.objects.len();
    if total == 0 {
        println!("\nCatalog is empty.");
        return Ok(());
    }

    // Count by type
    let mut by_type: HashMap<String, usize> = HashMap::new();
    let mut oldest_epoch: Option<DateTime<Utc>> = None;
    let mut newest_epoch: Option<DateTime<Utc>> = None;
    let mut total_age_hours = 0.0f64;
    let now = Utc::now();

    for entry in state.objects.values() {
        *by_type.entry(entry.object_type.clone()).or_insert(0) += 1;

        let age = (now - entry.epoch).num_minutes() as f64 / 60.0;
        total_age_hours += age;

        match oldest_epoch {
            None => oldest_epoch = Some(entry.epoch),
            Some(old) if entry.epoch < old => oldest_epoch = Some(entry.epoch),
            _ => {}
        }
        match newest_epoch {
            None => newest_epoch = Some(entry.epoch),
            Some(new) if entry.epoch > new => newest_epoch = Some(entry.epoch),
            _ => {}
        }
    }

    let avg_age_hours = total_age_hours / total as f64;

    println!("\n{}", "Object Counts:".cyan());
    println!("  Total: {}", total.to_string().green().bold());

    let mut type_vec: Vec<_> = by_type.iter().collect();
    type_vec.sort_by(|a, b| b.1.cmp(a.1));
    for (obj_type, count) in &type_vec {
        let pct = (**count as f64 / total as f64) * 100.0;
        println!("  {:<15} {:>6}  ({:.1}%)", obj_type, count, pct);
    }

    println!("\n{}", "TLE Freshness:".cyan());
    println!("  Average TLE age:  {:.1} hours", avg_age_hours);
    if let Some(oldest) = oldest_epoch {
        let age = (now - oldest).num_hours();
        println!(
            "  Oldest TLE:       {} ({}h ago)",
            oldest.format("%Y-%m-%d %H:%M UTC"),
            age
        );
    }
    if let Some(newest) = newest_epoch {
        let age = (now - newest).num_hours();
        println!(
            "  Newest TLE:       {} ({}h ago)",
            newest.format("%Y-%m-%d %H:%M UTC"),
            age
        );
    }

    if let Some(last) = state.last_updated {
        let age = (now - last).num_minutes() as f64 / 60.0;
        println!("\n{}", "Ingestion:".cyan());
        println!(
            "  Last update: {} ({:.1}h ago)",
            last.format("%Y-%m-%d %H:%M UTC"),
            age
        );
    }

    Ok(())
}

/// Helper to fetch TLEs for ingestion
fn fetch_tles_for_ingest(query: &str, limit: usize) -> Result<Vec<(String, TwoLineElement)>> {
    let url = format!("{}?{}&FORMAT=TLE", CELESTRAK_GP_URL, query);
    let response = reqwest::blocking::get(&url)
        .context("Failed to fetch TLEs")?
        .text()
        .context("Failed to read response")?;

    let lines: Vec<&str> = response.lines().collect();
    let mut tles = Vec::new();

    for chunk in lines.chunks(3) {
        if chunk.len() < 3 || tles.len() >= limit {
            break;
        }

        let name = chunk[0].trim().to_string();
        let line1 = chunk[1].trim();
        let line2 = chunk[2].trim();

        if let Ok(tle) = TwoLineElement::parse_lines(Some(name.clone()), line1, line2) {
            tles.push((name, tle));
        }
    }

    Ok(tles)
}
