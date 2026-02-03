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

mod holochain_client;

use anyhow::{Context, Result};
use chrono::{DateTime, Duration, Utc};
use clap::{Parser, Subcommand};
use colored::*;
use serde::Serialize;
use orbital_mechanics::{
    conjunction::{ConjunctionAnalyzer, RiskLevel},
    propagator::Propagator,
    state::{DataSource, OrbitalState, StateVector},
    tle::TwoLineElement,
    covariance::CovarianceMatrix,
};
use holochain_client::{HolochainClient, HolochainConfig, IngestionBatch, OrbitalObjectInput, TleInput, StateVectorInput};
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

    /// Ingest TLE data into Holochain network
    Ingest {
        /// Source: "iss", "starlink", "debris", or "all"
        #[arg(short, long, default_value = "iss")]
        source: String,

        /// Maximum objects to ingest
        #[arg(short, long, default_value = "10")]
        limit: usize,

        /// Holochain conductor WebSocket URL
        #[arg(long, default_value = "ws://localhost:8888")]
        conductor_url: String,

        /// Dry run (prepare data but don't send to Holochain)
        #[arg(long, default_value = "false")]
        dry_run: bool,
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
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::FetchIss => fetch_iss()?,
        Commands::FetchStarlink { limit } => fetch_starlink(limit)?,
        Commands::FetchDebris { limit } => fetch_debris(limit)?,
        Commands::Propagate { hours, step } => propagate_iss(hours, step)?,
        Commands::Screen { threshold, debris_count } => screen_iss(threshold, debris_count)?,
        Commands::ConjunctionDemo => conjunction_demo()?,
        Commands::Ingest { source, limit, conductor_url, dry_run } => {
            // Use tokio runtime for async operations
            let rt = tokio::runtime::Runtime::new()?;
            rt.block_on(ingest_data(&source, limit, &conductor_url, dry_run))?;
        }
        Commands::Export { source, limit, output } => export_data(&source, limit, &output)?,
    }

    Ok(())
}

/// Fetch ISS TLE from CelesTrak
fn fetch_iss() -> Result<()> {
    println!("{}", "=== Fetching ISS TLE from CelesTrak ===".green().bold());

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
    println!("{}", format!("=== Fetching {} Starlink TLEs ===", limit).green().bold());

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
                tle.norad_id,
                name,
                altitude,
                tle.inclination_deg
            );
            count += 1;
        }
    }

    println!("\n{} Starlink satellites fetched", count.to_string().green());

    Ok(())
}

/// Fetch debris TLEs
fn fetch_debris(limit: usize) -> Result<()> {
    println!("{}", format!("=== Fetching {} Debris Objects ===", limit).green().bold());

    // CelesTrak debris catalog (COSMOS 1408 debris from ASAT test)
    let url = format!("{}?GROUP=cosmos-1408-debris&FORMAT=TLE", CELESTRAK_GP_URL);
    let response = reqwest::blocking::get(&url)
        .context("Failed to fetch debris TLEs")?
        .text()
        .context("Failed to read response")?;

    let lines: Vec<&str> = response.lines().collect();
    let mut count = 0;

    println!("\n{:<10} | {:<25} | {:>8} | {:>6}", "NORAD ID", "Name", "Alt (km)", "Inc");
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
                tle.norad_id,
                display_name,
                altitude,
                tle.inclination_deg
            );
            count += 1;
        }
    }

    println!("\n{} debris objects fetched", count.to_string().yellow());

    Ok(())
}

/// Propagate ISS position
fn propagate_iss(hours: f64, step_minutes: f64) -> Result<()> {
    println!("{}", format!("=== Propagating ISS for {} hours ===", hours).green().bold());

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
        lines[2].trim()
    )?;

    let propagator = Propagator::from_tle(&tle)?;

    println!("\n{:<8} | {:>10} | {:>10} | {:>10} | {:>8} | {:>8}",
             "Time", "X (km)", "Y (km)", "Z (km)", "Alt (km)", "Speed");
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
                time_offset,
                state.state.x, state.state.y, state.state.z,
                alt, speed
            );
        }

        current = current + Duration::minutes(step_minutes as i64);
    }

    Ok(())
}

/// Screen ISS for close approaches
fn screen_iss(threshold_km: f64, debris_count: usize) -> Result<()> {
    println!("{}", format!("=== Screening ISS for conjunctions (threshold: {} km) ===",
                          threshold_km).green().bold());

    // Fetch ISS TLE
    let iss_url = format!("{}?CATNR=25544&FORMAT=TLE", CELESTRAK_GP_URL);
    let iss_response = reqwest::blocking::get(&iss_url)?.text()?;
    let iss_lines: Vec<&str> = iss_response.lines().collect();
    if iss_lines.len() < 3 {
        anyhow::bail!("Invalid ISS TLE response");
    }

    let iss_tle = TwoLineElement::parse_lines(
        Some(iss_lines[0].trim().to_string()),
        iss_lines[1].trim(),
        iss_lines[2].trim()
    )?;

    // Fetch debris TLEs
    let debris_url = format!("{}?GROUP=cosmos-1408-debris&FORMAT=TLE", CELESTRAK_GP_URL);
    let debris_response = reqwest::blocking::get(&debris_url)?.text()?;
    let debris_lines: Vec<&str> = debris_response.lines().collect();

    let iss_prop = Propagator::from_tle(&iss_tle)?;
    let now = Utc::now();

    // Get ISS position now
    let iss_state = iss_prop.propagate_to(now)?;

    println!("\n{}", "ISS current position:".cyan());
    println!("  X: {:.1} km, Y: {:.1} km, Z: {:.1} km",
             iss_state.state.x, iss_state.state.y, iss_state.state.z);
    println!("  Altitude: {:.1} km", iss_state.state.altitude_km());

    println!("\n{}", "Screening against debris...".yellow());

    let mut close_approaches: Vec<(String, u32, f64)> = Vec::new();
    let mut checked = 0;

    for chunk in debris_lines.chunks(3) {
        if chunk.len() < 3 || checked >= debris_count {
            break;
        }

        let name = chunk[0].trim();
        let line1 = chunk[1].trim();
        let line2 = chunk[2].trim();

        if let Ok(debris_tle) = TwoLineElement::parse_lines(Some(name.to_string()), line1, line2) {
            if let Ok(debris_prop) = Propagator::from_tle(&debris_tle) {
                if let Ok(debris_state) = debris_prop.propagate_to(now) {
                    let distance = iss_state.state.distance_to(&debris_state.state);

                    if distance < threshold_km {
                        close_approaches.push((name.to_string(), debris_tle.norad_id, distance));
                    }
                }
            }
        }
        checked += 1;
    }

    // Sort by distance
    close_approaches.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());

    if close_approaches.is_empty() {
        println!("\n{}", "No close approaches found within threshold.".green());
    } else {
        println!("\n{} close approaches found:", close_approaches.len().to_string().red().bold());
        println!("\n{:<10} | {:<25} | {:>12}", "NORAD ID", "Name", "Distance (km)");
        println!("{}", "-".repeat(55));

        for (name, norad_id, distance) in &close_approaches {
            let color = if *distance < 1.0 {
                "red"
            } else if *distance < 5.0 {
                "yellow"
            } else {
                "white"
            };

            let dist_str = format!("{:.2}", distance);
            let colored_dist = match color {
                "red" => dist_str.red(),
                "yellow" => dist_str.yellow(),
                _ => dist_str.normal(),
            };

            let display_name = if name.len() > 25 { &name[..25] } else { name };
            println!("{:<10} | {:<25} | {:>12}",
                     norad_id,
                     display_name,
                     colored_dist);
        }
    }

    println!("\nChecked {} debris objects", checked);

    Ok(())
}

/// Demonstrate conjunction analysis
fn conjunction_demo() -> Result<()> {
    println!("{}", "=== Conjunction Analysis Demo ===".green().bold());

    let now = Utc::now();

    // Create two hypothetical objects close together
    let primary = OrbitalState::new(
        25544,  // ISS
        now,
        StateVector::new(6800.0, 0.0, 0.0, 0.0, 7.66, 0.0),
        DataSource::SpaceTrack,
    ).with_covariance(CovarianceMatrix::diagonal([0.5, 0.5, 0.5, 0.001, 0.001, 0.001]));

    // Secondary 0.5 km away
    let secondary = OrbitalState::new(
        99999,  // Hypothetical debris
        now,
        StateVector::new(6800.5, 0.0, 0.0, 0.0, 7.66, 0.0),
        DataSource::SpaceTrack,
    ).with_covariance(CovarianceMatrix::diagonal([1.0, 1.0, 1.0, 0.002, 0.002, 0.002]));

    let analyzer = ConjunctionAnalyzer::new().with_hbr(20.0);  // 20m combined radius
    let assessment = analyzer.assess(&primary, &secondary);

    println!("\n{}", "Conjunction Assessment:".cyan());
    println!("  Primary NORAD ID: {}", assessment.primary_norad_id);
    println!("  Secondary NORAD ID: {}", assessment.secondary_norad_id);
    println!("  Time of Closest Approach: {}", assessment.tca);
    println!("  Miss Distance: {:.3} km", assessment.miss_distance_km);
    println!("  Relative Velocity: {:.3} km/s", assessment.relative_velocity_kms);
    println!("  Hard Body Radius: {:.1} m", assessment.hard_body_radius_m);

    println!("\n{}", "Collision Probability:".yellow());
    println!("  Pc: {:.2e}", assessment.collision_probability.pc);
    println!("  Pc Lower: {:.2e}", assessment.collision_probability.pc_lower);
    println!("  Pc Upper: {:.2e}", assessment.collision_probability.pc_upper);
    println!("  Method: {:?}", assessment.collision_probability.method);
    println!("  Has Covariance: {}", assessment.collision_probability.has_covariance);

    let risk_color = match assessment.risk_level {
        RiskLevel::Emergency => "red",
        RiskLevel::High => "red",
        RiskLevel::Medium => "yellow",
        RiskLevel::Low => "green",
        RiskLevel::Negligible => "green",
    };

    let risk_str = format!("{:?}", assessment.risk_level);
    println!("\n{}", "Risk Assessment:".cyan());
    println!("  Level: {}", match risk_color {
        "red" => risk_str.red().bold(),
        "yellow" => risk_str.yellow().bold(),
        _ => risk_str.green().bold(),
    });
    println!("  Recommendation: {}", assessment.risk_level.description());

    Ok(())
}

/// Ingest TLE data into Holochain
async fn ingest_data(source: &str, limit: usize, conductor_url: &str, dry_run: bool) -> Result<()> {
    println!("{}", format!("=== Ingesting {} data into Holochain ===", source).green().bold());

    // Collect TLEs based on source (using spawn_blocking for blocking HTTP calls)
    let source_owned = source.to_lowercase();
    let tles = tokio::task::spawn_blocking(move || -> Result<Vec<(String, TwoLineElement)>> {
        match source_owned.as_str() {
            "iss" => fetch_tles_for_ingest("CATNR=25544", 1),
            "starlink" => fetch_tles_for_ingest("GROUP=starlink", limit),
            "debris" => fetch_tles_for_ingest("GROUP=cosmos-1408-debris", limit),
            "all" => {
                let mut all = fetch_tles_for_ingest("CATNR=25544", 1)?;
                all.extend(fetch_tles_for_ingest("GROUP=starlink", limit / 2)?);
                all.extend(fetch_tles_for_ingest("GROUP=cosmos-1408-debris", limit / 2)?);
                Ok(all)
            }
            _ => anyhow::bail!("Unknown source. Use 'iss', 'starlink', 'debris', or 'all'"),
        }
    }).await??;

    println!("\nFetched {} TLEs from CelesTrak", tles.len());

    // Build ingestion batch
    let mut batch = IngestionBatch::new();

    for (name, tle) in &tles {
        // Determine object type based on name
        let object_type = if name.contains("DEB") || name.contains("debris") {
            "Debris"
        } else if name.contains("R/B") {
            "RocketBody"
        } else {
            "Payload"
        };

        // Add orbital object
        batch = batch.add_object(OrbitalObjectInput {
            norad_id: tle.norad_id,
            name: name.clone(),
            object_type: object_type.to_string(),
            launch_date: None,  // Could parse from international designator
            decay_date: None,
            owner_country: None,
            data_source: "CelesTrak".to_string(),
            metadata: HashMap::new(),
        });

        // Add TLE
        batch = batch.add_tle(TleInput {
            norad_id: tle.norad_id,
            line1: tle.line1.clone(),
            line2: tle.line2.clone(),
            epoch: tle.epoch,
            source: "CelesTrak".to_string(),
        });

        // Propagate to get current state vector
        if let Ok(propagator) = Propagator::from_tle(tle) {
            if let Ok(state) = propagator.propagate_to(Utc::now()) {
                batch = batch.add_state_vector(StateVectorInput {
                    norad_id: tle.norad_id,
                    epoch: Utc::now(),
                    position_km: [state.state.x, state.state.y, state.state.z],
                    velocity_kms: [state.state.vx, state.state.vy, state.state.vz],
                    covariance: None,
                    reference_frame: "Teme".to_string(),
                    quality: 0.9,  // CelesTrak data is generally good quality
                    source: "CelesTrak".to_string(),
                });
            }
        }
    }

    println!("\nPrepared batch:");
    println!("  Orbital Objects: {}", batch.object_count());
    println!("  TLEs: {}", batch.tle_count());
    println!("  State Vectors: {}", batch.state_vector_count());

    if dry_run {
        println!("\n{}", "DRY RUN - Not sending to Holochain".yellow());
        println!("\nSample entries:");
        for obj in batch.objects().iter().take(3) {
            println!("  - {} (NORAD {}): {}", obj.name, obj.norad_id, obj.object_type);
        }
        return Ok(());
    }

    // Connect to Holochain and ingest
    let config = HolochainConfig {
        conductor_url: conductor_url.to_string(),
        ..Default::default()
    };

    let mut client = HolochainClient::new(config);
    client.connect().await?;

    let report = batch.ingest(&client).await?;
    report.print_summary();

    Ok(())
}

/// Export data to JSON file for offline ingestion
fn export_data(source: &str, limit: usize, output: &str) -> Result<()> {
    println!("{}", format!("=== Exporting {} data to {} ===", source, output).green().bold());

    // Collect TLEs based on source
    let tles = match source.to_lowercase().as_str() {
        "iss" => fetch_tles_for_ingest("CATNR=25544", 1)?,
        "starlink" => fetch_tles_for_ingest("GROUP=starlink", limit)?,
        "debris" => fetch_tles_for_ingest("GROUP=cosmos-1408-debris", limit)?,
        "all" => {
            let mut all = fetch_tles_for_ingest("CATNR=25544", 1)?;
            all.extend(fetch_tles_for_ingest("GROUP=starlink", limit / 2)?);
            all.extend(fetch_tles_for_ingest("GROUP=cosmos-1408-debris", limit / 2)?);
            all
        }
        _ => anyhow::bail!("Unknown source: {}. Use 'iss', 'starlink', 'debris', or 'all'", source),
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
            propagator.propagate_to(Utc::now()).ok().map(|state| {
                ExportStateVector {
                    epoch: Utc::now(),
                    position_km: [state.state.x, state.state.y, state.state.z],
                    velocity_kms: [state.state.vx, state.state.vy, state.state.vz],
                    reference_frame: "Teme".to_string(),
                }
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

    println!("\n{}", format!("Exported {} objects to {}", export.count, output).green());
    println!("File size: {} bytes", json.len());

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
