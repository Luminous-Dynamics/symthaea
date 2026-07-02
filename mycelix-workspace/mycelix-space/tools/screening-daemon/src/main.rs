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
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tracing::{debug, error, info, warn};

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

    /// HTTP timeout in seconds for CelesTrak requests
    #[arg(long, default_value = "30")]
    http_timeout_secs: u64,

    /// Max HTTP retry attempts for CelesTrak fetches
    #[arg(long, default_value = "3")]
    http_retries: u32,

    /// Consecutive CelesTrak failures before circuit breaker opens
    #[arg(long, default_value = "3")]
    circuit_breaker_threshold: u64,

    /// Cycles to skip when circuit breaker is open
    #[arg(long, default_value = "5")]
    circuit_breaker_cooldown: u64,

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

/// Fetch TLE catalog from a CelesTrak URL (3LE format) with retry and exponential backoff.
///
/// Returns `Ok(catalog)` on success, `Err` after all retries exhausted.
async fn fetch_celestrak_catalog(
    url: &str,
    timeout_secs: u64,
    max_retries: u32,
) -> Result<HashMap<u32, TwoLineElement>, Box<dyn std::error::Error>> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(timeout_secs))
        .build()?;

    let mut last_err: Option<Box<dyn std::error::Error>> = None;

    for attempt in 0..max_retries {
        if attempt > 0 {
            let backoff = std::time::Duration::from_secs(1 << (attempt - 1)); // 1s, 2s, 4s, ...
            info!(attempt, backoff_secs = backoff.as_secs(), "Retrying CelesTrak fetch");
            tokio::time::sleep(backoff).await;
        }

        info!(url, attempt = attempt + 1, max = max_retries, "Fetching TLE catalog from CelesTrak");

        match client.get(url).send().await {
            Ok(resp) => match resp.text().await {
                Ok(body) => {
                    let catalog = parse_tle_catalog(&body)?;
                    info!(count = catalog.len(), "Fetched objects from CelesTrak");
                    return Ok(catalog);
                }
                Err(e) => {
                    warn!(attempt = attempt + 1, error = %e, "CelesTrak response body read failed");
                    last_err = Some(Box::new(e));
                }
            },
            Err(e) => {
                warn!(attempt = attempt + 1, error = %e, "CelesTrak HTTP request failed");
                last_err = Some(Box::new(e));
            }
        }
    }

    Err(last_err.unwrap_or_else(|| "All CelesTrak fetch retries exhausted".into()))
}

/// Circuit breaker state for CelesTrak fetches.
struct CircuitBreaker {
    consecutive_failures: u64,
    threshold: u64,
    cooldown_cycles: u64,
    /// Cycles remaining while circuit is open (skip fetching).
    skip_remaining: u64,
}

impl CircuitBreaker {
    fn new(threshold: u64, cooldown_cycles: u64) -> Self {
        Self {
            consecutive_failures: 0,
            threshold,
            cooldown_cycles,
            skip_remaining: 0,
        }
    }

    /// Returns true if the circuit is open (should skip fetch).
    fn is_open(&self) -> bool {
        self.skip_remaining > 0
    }

    /// Call each cycle when the circuit is open to count down. Returns true
    /// when the cooldown expires (circuit half-open, ready to retry).
    fn tick(&mut self) -> bool {
        if self.skip_remaining > 0 {
            self.skip_remaining -= 1;
            if self.skip_remaining == 0 {
                info!(
                    cooldown = self.cooldown_cycles,
                    "Circuit breaker closed — resuming CelesTrak fetches"
                );
                return true;
            }
        }
        false
    }

    fn record_success(&mut self) {
        if self.consecutive_failures > 0 {
            info!(
                prev_failures = self.consecutive_failures,
                "CelesTrak circuit breaker reset after success"
            );
        }
        self.consecutive_failures = 0;
    }

    fn record_failure(&mut self) {
        self.consecutive_failures += 1;
        if self.consecutive_failures >= self.threshold && self.skip_remaining == 0 {
            self.skip_remaining = self.cooldown_cycles;
            error!(
                failures = self.consecutive_failures,
                cooldown = self.cooldown_cycles,
                "Circuit breaker OPEN — disabling CelesTrak fetches"
            );
        }
    }
}

/// Catalog staleness tracker.
struct CatalogFreshness {
    last_success: Option<std::time::Instant>,
}

impl CatalogFreshness {
    fn new() -> Self {
        Self { last_success: None }
    }

    fn record_success(&mut self) {
        self.last_success = Some(std::time::Instant::now());
    }

    /// Check and log staleness warnings. Returns the age in hours if available.
    fn check(&self) -> Option<f64> {
        let last = self.last_success?;
        let age = last.elapsed();
        let hours = age.as_secs_f64() / 3600.0;

        if hours > 72.0 {
            error!(
                age_hours = format!("{:.1}", hours),
                "Catalog is critically stale (>72h) — screening results are unreliable"
            );
        } else if hours > 24.0 {
            warn!(
                age_hours = format!("{:.1}", hours),
                "Catalog is stale (>24h) — consider investigating CelesTrak connectivity"
            );
        }

        Some(hours)
    }
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

/// Statistics from a screening run (propagation failure tracking).
#[derive(Debug, Clone, Default)]
struct ScreeningStats {
    pairs_attempted: u64,
    propagation_failures: u64,
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
) -> (Vec<ScreeningResult>, ScreeningStats) {
    let mut stats = ScreeningStats::default();
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

            stats.pairs_attempted += 1;

            let Ok(secondary_prop) = Propagator::from_tle(secondary_tle) else {
                stats.propagation_failures += 1;
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
    (results, stats)
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
    info!("  HTTP timeout: {}s, retries: {}", args.http_timeout_secs, args.http_retries);

    // Graceful shutdown signal
    let shutdown = Arc::new(AtomicBool::new(false));
    {
        let shutdown = Arc::clone(&shutdown);
        tokio::spawn(async move {
            let ctrl_c = tokio::signal::ctrl_c();
            #[cfg(unix)]
            let mut sigterm = tokio::signal::unix::signal(
                tokio::signal::unix::SignalKind::terminate(),
            )
            .expect("failed to register SIGTERM handler");

            #[cfg(unix)]
            tokio::select! {
                _ = ctrl_c => {},
                _ = sigterm.recv() => {},
            }
            #[cfg(not(unix))]
            ctrl_c.await.ok();

            info!("Shutdown signal received — finishing current cycle");
            shutdown.store(true, Ordering::SeqCst);
        });
    }

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

    // Circuit breaker and freshness tracking for CelesTrak
    let mut breaker = CircuitBreaker::new(
        args.circuit_breaker_threshold,
        args.circuit_breaker_cooldown,
    );
    let mut freshness = CatalogFreshness::new();

    // Fetch CelesTrak catalog if URL provided (merges with file catalog)
    if let Some(ref url) = args.celestrak_url {
        match fetch_celestrak_catalog(url, args.http_timeout_secs, args.http_retries).await {
            Ok(celestrak_cat) => {
                let new_count = celestrak_cat.len();
                catalog.extend(celestrak_cat);
                freshness.record_success();
                breaker.record_success();
                info!(
                    total = catalog.len(),
                    merged = new_count,
                    "Catalog loaded (merged CelesTrak)"
                );
            }
            Err(e) => {
                warn!(error = %e, "Initial CelesTrak fetch failed — continuing with file catalog only");
                breaker.record_failure();
            }
        }
    }

    if catalog.is_empty() {
        info!("No catalog provided — running with empty catalog (demo mode)");
    }

    if args.protected_objects.is_empty() {
        warn!("No protected objects specified. Use -p to specify NORAD IDs.");
        if args.once {
            return Ok(());
        }
    }

    let mut cycle_count: u64 = 0;

    loop {
        if shutdown.load(Ordering::SeqCst) {
            info!(
                cycles_completed = cycle_count,
                catalog_size = catalog.len(),
                "Shutdown requested — exiting cleanly"
            );
            break;
        }

        let cycle_start = std::time::Instant::now();
        cycle_count += 1;

        // Re-fetch CelesTrak catalog on refresh cycles (with circuit breaker)
        if let Some(ref url) = args.celestrak_url {
            if cycle_count > 1 && cycle_count % args.refresh_cycles == 1 {
                if breaker.is_open() {
                    breaker.tick();
                    debug!(cycle = cycle_count, "CelesTrak fetch skipped (circuit breaker open)");
                } else {
                    let fetch_start = std::time::Instant::now();
                    match fetch_celestrak_catalog(url, args.http_timeout_secs, args.http_retries)
                        .await
                    {
                        Ok(fresh_cat) => {
                            let new_count = fresh_cat.len();
                            catalog.extend(fresh_cat);
                            freshness.record_success();
                            breaker.record_success();
                            info!(
                                merged = new_count,
                                total = catalog.len(),
                                fetch_ms = fetch_start.elapsed().as_millis() as u64,
                                "Refreshed CelesTrak catalog"
                            );
                        }
                        Err(e) => {
                            breaker.record_failure();
                            warn!(
                                error = %e,
                                consecutive_failures = breaker.consecutive_failures,
                                "Failed to refresh CelesTrak catalog (using stale data)"
                            );
                        }
                    }
                }
            }
        }

        // Check catalog staleness each cycle
        if args.celestrak_url.is_some() {
            freshness.check();
        }

        info!(cycle = cycle_count, catalog_size = catalog.len(), "Starting screening run");

        let screen_start = std::time::Instant::now();
        let (results, stats) = screen_catalog(
            &catalog,
            &args.protected_objects,
            args.hours_ahead,
            args.miss_threshold_km,
            args.pc_threshold,
            &priority_weights,
        );
        let screen_ms = screen_start.elapsed().as_millis() as u64;

        let duration_ms = cycle_start.elapsed().as_millis() as u64;
        let high_risk = results
            .iter()
            .filter(|r| r.collision_probability >= 1e-4)
            .count();

        // Log screening throughput and propagation failure rate
        info!(
            pairs = stats.pairs_attempted,
            screen_ms,
            "Screened {} pairs in {}ms",
            stats.pairs_attempted,
            screen_ms,
        );
        if stats.propagation_failures > 0 {
            let fail_rate = stats.propagation_failures as f64 / stats.pairs_attempted.max(1) as f64;
            warn!(
                failures = stats.propagation_failures,
                rate = format!("{:.2}%", fail_rate * 100.0),
                "Propagation failures this cycle"
            );
        }

        let summary = ScreeningSummary {
            timestamp: Utc::now().to_rfc3339(),
            objects_screened: args.protected_objects.len(),
            pairs_evaluated: stats.pairs_attempted as usize,
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
                info!(
                    objects = summary.objects_screened,
                    pairs = summary.pairs_evaluated,
                    conjunctions = summary.conjunctions_found,
                    high_risk = summary.high_risk_count,
                    duration_ms,
                    "Screening cycle complete"
                );

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

        info!(next_in_secs = args.interval_seconds, "Waiting for next cycle");

        // Sleep in 1-second increments to allow prompt shutdown
        for _ in 0..args.interval_seconds {
            if shutdown.load(Ordering::SeqCst) {
                break;
            }
            tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
        }
    }

    info!(cycles_completed = cycle_count, "Screening daemon stopped");
    Ok(())
}
