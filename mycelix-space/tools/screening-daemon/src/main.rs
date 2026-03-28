// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Automated Conjunction Screening Daemon
//!
//! Periodically screens tracked objects for potential conjunctions using
//! the orbital-mechanics library. Designed to run alongside a Holochain
//! conductor, reading TLE data and writing conjunction events.
//!
//! Supports loading TLE catalogs from local files (`--catalog-path`) or
//! fetching directly from CelesTrak (`--celestrak-url`). When both are
//! provided, catalogs are merged (CelesTrak takes precedence for duplicate
//! NORAD IDs, since it is fresher).
//!
//! Consciousness-weighted priority screening is available via
//! `--priority-weights`, mapping NORAD IDs to weights (0.0-1.0) derived
//! from operator consciousness tiers. Higher-weight objects are screened
//! first with finer time steps.

use clap::Parser;
use chrono::{Duration, Utc};
use orbital_mechanics::conjunction::ConjunctionAnalyzer;
use orbital_mechanics::propagator::Propagator;
use orbital_mechanics::tle::TwoLineElement;
use std::collections::HashMap;
use tracing::{debug, info, warn};

/// Conjunction screening daemon for mycelix-space
#[derive(Parser, Debug)]
#[command(name = "mycelix-space-screener")]
#[command(about = "Automated conjunction screening for decentralized space situational awareness")]
struct Args {
    /// Screening interval in seconds
    #[arg(short, long, default_value = "900")]
    interval_seconds: u64,

    /// Hours ahead to screen for conjunctions
    #[arg(short = 'H', long, default_value = "72")]
    hours_ahead: u64,

    /// Collision probability threshold (events above this are reported)
    #[arg(short = 't', long, default_value = "1e-7")]
    pc_threshold: f64,

    /// Miss distance screening threshold in km
    #[arg(short = 'm', long, default_value = "5.0")]
    miss_threshold_km: f64,

    /// Path to TLE catalog file (one TLE per 2-3 lines)
    #[arg(short = 'c', long)]
    catalog_path: Option<String>,

    /// CelesTrak GP data URL to fetch TLEs from (3LE format).
    /// Example: https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle
    #[arg(long)]
    celestrak_url: Option<String>,

    /// Re-fetch CelesTrak catalog every N screening cycles (default: 4)
    #[arg(long, default_value = "4")]
    refresh_cycles: u64,

    /// NORAD IDs to protect (screen these against the catalog)
    #[arg(short = 'p', long, value_delimiter = ',')]
    protected_objects: Vec<u32>,

    /// Path to priority weights JSON file (NORAD ID -> weight 0.0-1.0).
    /// Higher-weight objects get screened first with finer time steps.
    /// Weights conceptually derive from consciousness tier (Guardian=1.0, Observer=0.1).
    #[arg(long)]
    priority_weights: Option<String>,

    /// Run once and exit (no daemon loop)
    #[arg(long)]
    once: bool,

    /// Output format: json or human
    #[arg(short, long, default_value = "human")]
    format: String,
}

/// Result of screening one pair of objects
#[derive(Debug, Clone, serde::Serialize)]
struct ScreeningResult {
    primary_norad_id: u32,
    secondary_norad_id: u32,
    time_of_closest_approach: String,
    miss_distance_km: f64,
    collision_probability: f64,
    risk_level: String,
    relative_velocity_kms: f64,
    /// Priority weight of the primary object (from consciousness tier)
    priority_weight: f64,
}

/// Summary of a screening run
#[derive(Debug, Clone, serde::Serialize)]
struct ScreeningSummary {
    timestamp: String,
    objects_screened: usize,
    pairs_evaluated: usize,
    conjunctions_found: usize,
    high_risk_count: usize,
    duration_ms: u64,
    results: Vec<ScreeningResult>,
}

/// Parse TLE catalog from string content (shared between file and HTTP loading).
///
/// Handles both 2-line (line1 + line2) and 3-line (name + line1 + line2) formats,
/// as well as blank-line separators.
fn parse_tle_catalog(content: &str) -> Result<HashMap<u32, TwoLineElement>, Box<dyn std::error::Error>> {
    let lines: Vec<&str> = content.lines().collect();
    let mut catalog = HashMap::new();

    let mut i = 0;
    while i < lines.len() {
        // Skip empty lines
        if lines[i].trim().is_empty() {
            i += 1;
            continue;
        }

        // Try to parse as TLE (2-line or 3-line format)
        if i + 1 < lines.len() && lines[i].starts_with('1') && lines[i + 1].starts_with('2') {
            // 2-line format
            match TwoLineElement::parse_lines(None, lines[i], lines[i + 1]) {
                Ok(tle) => {
                    catalog.insert(tle.norad_id, tle);
                    i += 2;
                }
                Err(e) => {
                    debug!("Skipping malformed TLE at line {}: {}", i + 1, e);
                    i += 1;
                }
            }
        } else if i + 2 < lines.len()
            && lines[i + 1].starts_with('1')
            && lines[i + 2].starts_with('2')
        {
            // 3-line format (name + line1 + line2)
            let name = lines[i].trim().to_string();
            match TwoLineElement::parse_lines(Some(name), lines[i + 1], lines[i + 2]) {
                Ok(tle) => {
                    catalog.insert(tle.norad_id, tle);
                    i += 3;
                }
                Err(e) => {
                    debug!("Skipping malformed TLE at line {}: {}", i + 1, e);
                    i += 1;
                }
            }
        } else {
            i += 1;
        }
    }

    Ok(catalog)
}

/// Load TLE catalog from a file (3-line format: name + line1 + line2)
fn load_catalog(path: &str) -> Result<HashMap<u32, TwoLineElement>, Box<dyn std::error::Error>> {
    let content = std::fs::read_to_string(path)?;
    parse_tle_catalog(&content)
}

/// Fetch TLE catalog from a CelesTrak URL (3LE format)
async fn fetch_celestrak_catalog(url: &str) -> Result<HashMap<u32, TwoLineElement>, Box<dyn std::error::Error>> {
    info!("Fetching TLE catalog from CelesTrak: {}", url);
    let response = reqwest::get(url).await?.text().await?;
    let catalog = parse_tle_catalog(&response)?;
    info!("Fetched {} objects from CelesTrak", catalog.len());
    Ok(catalog)
}

/// Load priority weights from a JSON file.
///
/// Expected format: `{ "25544": 1.0, "48274": 0.9 }`
/// Keys are NORAD ID strings, values are weights in [0.0, 1.0].
fn load_priority_weights(path: &str) -> Result<HashMap<u32, f64>, Box<dyn std::error::Error>> {
    let content = std::fs::read_to_string(path)?;
    let raw: HashMap<String, f64> = serde_json::from_str(&content)?;
    let mut weights = HashMap::new();
    for (key, value) in raw {
        let norad_id: u32 = key.parse()?;
        weights.insert(norad_id, value.clamp(0.0, 1.0));
    }
    Ok(weights)
}

/// Screen protected objects against catalog for conjunctions.
///
/// Uses a coarse-fine two-pass approach:
/// 1. Coarse scan to find approximate TCA
/// 2. Fine scan around the coarse minimum
/// 3. Full conjunction assessment at refined TCA via `ConjunctionAnalyzer`
///
/// When priority weights are provided, high-priority objects (weight > 0.8)
/// get finer time steps (30s coarse / 5s fine) vs standard (60s coarse / 10s fine).
/// Objects without explicit weights default to 0.5.
fn screen_catalog(
    catalog: &HashMap<u32, TwoLineElement>,
    protected: &[u32],
    hours_ahead: u64,
    miss_threshold_km: f64,
    pc_threshold: f64,
    priority_weights: &HashMap<u32, f64>,
) -> Vec<ScreeningResult> {
    let mut results = Vec::new();
    let analyzer = ConjunctionAnalyzer::new().with_screening_threshold(miss_threshold_km);

    let total_seconds = (hours_ahead * 3600) as i64;

    // Sort protected objects by priority weight (highest first)
    let mut sorted_protected: Vec<u32> = protected.to_vec();
    sorted_protected.sort_by(|a, b| {
        let wa = priority_weights.get(b).copied().unwrap_or(0.5);
        let wb = priority_weights.get(a).copied().unwrap_or(0.5);
        wa.partial_cmp(&wb).unwrap_or(std::cmp::Ordering::Equal)
    });

    for &primary_id in &sorted_protected {
        let weight = priority_weights.get(&primary_id).copied().unwrap_or(0.5);

        // High-priority objects get finer time steps
        let (coarse_step_secs, fine_step_secs): (i64, i64) = if weight > 0.8 {
            (30, 5) // Premium screening for Guardian-tier
        } else {
            (60, 10) // Standard screening
        };

        let num_coarse_steps = (total_seconds / coarse_step_secs).max(1);

        let Some(primary_tle) = catalog.get(&primary_id) else {
            warn!("Protected object {} not found in catalog", primary_id);
            continue;
        };

        let Ok(primary_prop) = Propagator::from_tle(primary_tle) else {
            warn!("Failed to initialize propagator for {}", primary_id);
            continue;
        };

        // Screen against all other objects in catalog
        for (&secondary_id, secondary_tle) in catalog.iter() {
            if secondary_id == primary_id {
                continue;
            }

            let Ok(secondary_prop) = Propagator::from_tle(secondary_tle) else {
                continue;
            };

            // Coarse scan: find the time step with minimum distance
            let mut best_distance = f64::MAX;
            let mut best_step: i64 = 0;

            for i in 0..=num_coarse_steps {
                let t = primary_tle.epoch + Duration::seconds(i * coarse_step_secs);
                let (Ok(p_state), Ok(s_state)) = (
                    primary_prop.propagate_to(t),
                    secondary_prop.propagate_to(t),
                ) else {
                    continue;
                };

                let dist = p_state.state.distance_to(&s_state.state);
                if dist < best_distance {
                    best_distance = dist;
                    best_step = i * coarse_step_secs;
                }
            }

            // If coarse minimum is within screening threshold, refine
            if best_distance < miss_threshold_km {
                // Fine scan around the coarse minimum
                let fine_start = (best_step - coarse_step_secs).max(0);
                let fine_end = (best_step + coarse_step_secs).min(total_seconds);
                let mut tca_offset = best_step;
                let mut tca_distance = best_distance;

                let mut t_secs = fine_start;
                while t_secs <= fine_end {
                    let t = primary_tle.epoch + Duration::seconds(t_secs);
                    let (Ok(p_state), Ok(s_state)) = (
                        primary_prop.propagate_to(t),
                        secondary_prop.propagate_to(t),
                    ) else {
                        t_secs += fine_step_secs;
                        continue;
                    };

                    let dist = p_state.state.distance_to(&s_state.state);
                    if dist < tca_distance {
                        tca_distance = dist;
                        tca_offset = t_secs;
                    }
                    t_secs += fine_step_secs;
                }

                // Assess conjunction at refined TCA
                let tca_time = primary_tle.epoch + Duration::seconds(tca_offset);
                let (Ok(p_state), Ok(s_state)) = (
                    primary_prop.propagate_to(tca_time),
                    secondary_prop.propagate_to(tca_time),
                ) else {
                    continue;
                };

                let assessment = analyzer.assess(&p_state, &s_state);

                if assessment.collision_probability.pc >= pc_threshold {
                    let risk_level = match assessment.risk_level {
                        orbital_mechanics::conjunction::RiskLevel::Emergency => "Emergency",
                        orbital_mechanics::conjunction::RiskLevel::High => "High",
                        orbital_mechanics::conjunction::RiskLevel::Medium => "Medium",
                        orbital_mechanics::conjunction::RiskLevel::Low => "Low",
                        orbital_mechanics::conjunction::RiskLevel::Negligible => "Negligible",
                    };

                    results.push(ScreeningResult {
                        primary_norad_id: primary_id,
                        secondary_norad_id: secondary_id,
                        time_of_closest_approach: assessment.tca.to_rfc3339(),
                        miss_distance_km: assessment.miss_distance_km,
                        collision_probability: assessment.collision_probability.pc,
                        risk_level: risk_level.to_string(),
                        relative_velocity_kms: assessment.relative_velocity_kms,
                        priority_weight: weight,
                    });
                }
            }
        }
    }

    // Sort by collision probability (highest first)
    results.sort_by(|a, b| {
        b.collision_probability
            .partial_cmp(&a.collision_probability)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    results
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let args = Args::parse();

    info!("Mycelix Space Screening Daemon starting");
    info!("  Interval: {}s", args.interval_seconds);
    info!("  Hours ahead: {}", args.hours_ahead);
    info!("  Pc threshold: {:.2e}", args.pc_threshold);
    info!("  Miss threshold: {} km", args.miss_threshold_km);

    // Load priority weights if provided
    let priority_weights = if let Some(ref path) = args.priority_weights {
        info!("Loading priority weights from: {}", path);
        let weights = load_priority_weights(path)?;
        info!("Loaded {} priority weights", weights.len());
        weights
    } else {
        HashMap::new()
    };

    // Load file-based catalog if provided
    let mut catalog = if let Some(ref path) = args.catalog_path {
        info!("Loading TLE catalog from file: {}", path);
        let cat = load_catalog(path)?;
        info!("Loaded {} objects from file", cat.len());
        cat
    } else {
        HashMap::new()
    };

    // Fetch CelesTrak catalog if URL provided (merges with file catalog)
    if let Some(ref url) = args.celestrak_url {
        let celestrak_cat = fetch_celestrak_catalog(url).await?;
        let new_count = celestrak_cat.len();
        // CelesTrak entries take precedence (fresher data)
        catalog.extend(celestrak_cat);
        info!("Catalog now contains {} objects (merged {} from CelesTrak)", catalog.len(), new_count);
    }

    if catalog.is_empty() {
        info!("No catalog provided -- running with empty catalog (demo mode)");
    }

    if args.protected_objects.is_empty() {
        warn!("No protected objects specified. Use -p to specify NORAD IDs.");
        if args.once {
            return Ok(());
        }
    }

    let mut cycle_count: u64 = 0;

    loop {
        let start = std::time::Instant::now();
        cycle_count += 1;

        // Re-fetch CelesTrak catalog on refresh cycles
        if let Some(ref url) = args.celestrak_url {
            if cycle_count > 1 && cycle_count % args.refresh_cycles == 1 {
                match fetch_celestrak_catalog(url).await {
                    Ok(fresh_cat) => {
                        let new_count = fresh_cat.len();
                        catalog.extend(fresh_cat);
                        info!("Refreshed CelesTrak catalog ({} objects merged, {} total)", new_count, catalog.len());
                    }
                    Err(e) => {
                        warn!("Failed to refresh CelesTrak catalog: {} (using stale data)", e);
                    }
                }
            }
        }

        info!("Starting screening run (cycle {})...", cycle_count);
        let results = screen_catalog(
            &catalog,
            &args.protected_objects,
            args.hours_ahead,
            args.miss_threshold_km,
            args.pc_threshold,
            &priority_weights,
        );

        let duration_ms = start.elapsed().as_millis() as u64;
        let high_risk = results
            .iter()
            .filter(|r| r.collision_probability >= 1e-4)
            .count();

        let summary = ScreeningSummary {
            timestamp: Utc::now().to_rfc3339(),
            objects_screened: args.protected_objects.len(),
            pairs_evaluated: args.protected_objects.len() * catalog.len(),
            conjunctions_found: results.len(),
            high_risk_count: high_risk,
            duration_ms,
            results: results.clone(),
        };

        match args.format.as_str() {
            "json" => {
                println!("{}", serde_json::to_string_pretty(&summary)?);
            }
            _ => {
                info!("Screening complete in {}ms", duration_ms);
                info!("  Objects screened: {}", summary.objects_screened);
                info!("  Pairs evaluated: {}", summary.pairs_evaluated);
                info!("  Conjunctions found: {}", summary.conjunctions_found);
                info!("  High risk (Pc >= 1e-4): {}", summary.high_risk_count);

                for result in &results {
                    let marker = match result.risk_level.as_str() {
                        "Emergency" => "!!!",
                        "High" => "!! ",
                        "Medium" => "!  ",
                        _ => "   ",
                    };
                    info!(
                        "{} {} vs {} | TCA: {} | Miss: {:.3} km | Pc: {:.2e} | Risk: {} | Weight: {:.1}",
                        marker,
                        result.primary_norad_id,
                        result.secondary_norad_id,
                        result.time_of_closest_approach,
                        result.miss_distance_km,
                        result.collision_probability,
                        result.risk_level,
                        result.priority_weight,
                    );
                }
            }
        }

        if args.once {
            break;
        }

        info!("Next screening in {}s", args.interval_seconds);
        tokio::time::sleep(tokio::time::Duration::from_secs(args.interval_seconds)).await;
    }

    Ok(())
}
