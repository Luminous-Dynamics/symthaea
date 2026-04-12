// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Symthaea Watch — continuous interaction harness with ClickHouse telemetry.
//!
//! Drives Symthaea in a continuous loop with a rotating input corpus,
//! streams CycleMetadata to ClickHouse every second (batched), and serves
//! a Prometheus /metrics endpoint for Grafana dashboards.
//!
//! # Usage
//!
//! ```bash
//! # Initialize the ClickHouse table (run once)
//! cargo run --features "api_module,service,full_consciousness" --bin symthaea-watch -- --init-db
//!
//! # Start the harness with defaults
//! cargo run --features "api_module,service,full_consciousness" --bin symthaea-watch
//!
//! # Natural-speed operation (recommended for debug builds)
//! cargo run --features "api_module,service,full_consciousness" --bin symthaea-watch -- --no-throttle
//! ```
//!
//! # Live SQL queries against ClickHouse
//! ```sql
//! SELECT avg(consciousness_level), avg(prediction_error), count()
//! FROM symthaea_cycles WHERE ts > now() - INTERVAL 5 MINUTE;
//!
//! SELECT toStartOfMinute(ts) AS t, avg(dopamine), avg(noradrenaline)
//! FROM symthaea_cycles GROUP BY t ORDER BY t;
//!
//! SELECT urgency, count() FROM symthaea_cycles
//! WHERE ts > now() - INTERVAL 1 HOUR GROUP BY urgency;
//! ```

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use axum::{routing::get, Router};
use clap::Parser;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tracing::{error, info, warn};

use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService, TemporalBackend};

// ─── CLI ─────────────────────────────────────────────────────────────────────

#[derive(Parser, Debug)]
#[command(name = "symthaea-watch")]
#[command(about = "Continuous Symthaea harness — streams CycleMetadata to ClickHouse")]
struct Args {
    /// ClickHouse HTTP base URL.
    #[arg(long, default_value = "http://localhost:8123")]
    clickhouse: String,

    /// ClickHouse database name.
    #[arg(long, default_value = "default")]
    db: String,

    /// ClickHouse table name.
    #[arg(long, default_value = "symthaea_cycles")]
    table: String,

    /// Port to serve Prometheus /metrics on.
    #[arg(long, default_value_t = 8405)]
    metrics_port: u16,

    /// Target cognitive cycles per second (1–100).
    #[arg(long, default_value_t = 10)]
    cycles_per_second: u32,

    /// Create the ClickHouse table then exit.
    #[arg(long)]
    init_db: bool,

    /// Consciousness level below which an anomaly is logged.
    #[arg(long, default_value_t = 0.05)]
    collapse_threshold: f64,

    /// Number of consecutive collapsed cycles before alerting.
    #[arg(long, default_value_t = 30)]
    collapse_window: u32,

    /// Disable cycle-time throttle — let each cycle run at natural speed.
    /// Default in debug builds where cycles exceed the real-time budget.
    #[arg(long)]
    no_throttle: bool,
}

// ─── DATA ROW ────────────────────────────────────────────────────────────────

/// One row written to ClickHouse per cognitive cycle.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct CycleRow {
    /// Unix timestamp in milliseconds.
    ts_ms: u64,
    cycle_num: u64,
    // ── Consciousness ──────────────────────────────────────────────────
    consciousness_level: f64,
    narrative_self_psi: f64,
    living_mind_vitality: f64,
    resonance_frequency: f64,
    // ── Prediction ─────────────────────────────────────────────────────
    prediction_error: f32,
    peak_attention: f32,
    reasoning_confidence: f32,
    // ── Neuromodulators ────────────────────────────────────────────────
    dopamine: f32,
    noradrenaline: f32,
    serotonin: f32,
    acetylcholine: f32,
    // ── Phi decomposition ──────────────────────────────────────────────
    micro_phi: f64,
    meso_phi: f64,
    macro_phi: f64,
    // ── Temporal ───────────────────────────────────────────────────────
    temporal_coherence: f64,
    // ── Thermodynamic ──────────────────────────────────────────────────
    hierarchical_free_energy: f64,
    // ── Affect ─────────────────────────────────────────────────────────
    mood_temperature: f32,
    // ── Cycle info ─────────────────────────────────────────────────────
    learning_occurred: u8,
    surprise_triggered: u8,
    prefrontal_veto: u8,
    cycle_time_us: u64,
    urgency: String,
    language_output: String,
}

// ─── SHARED METRICS ──────────────────────────────────────────────────────────

#[derive(Default)]
struct Metrics {
    total_cycles: AtomicU64,
    total_rows_flushed: AtomicU64,
    clickhouse_errors: AtomicU64,
    anomalies_detected: AtomicU64,
    last_consciousness: std::sync::Mutex<f64>,
    last_prediction_error: std::sync::Mutex<f32>,
    last_urgency: std::sync::Mutex<String>,
}

impl Metrics {
    fn prometheus_text(&self) -> String {
        let consciousness = *self.last_consciousness.lock().unwrap();
        let pe = *self.last_prediction_error.lock().unwrap();
        let urgency = self.last_urgency.lock().unwrap().clone();
        let urgency_num: f64 = match urgency.as_str() {
            "critical" => 2.0,
            "normal" => 1.0,
            _ => 0.0,
        };
        format!(
            "# HELP symthaea_cycles_total Total cognitive cycles completed\n\
             # TYPE symthaea_cycles_total counter\n\
             symthaea_cycles_total {}\n\
             # HELP symthaea_rows_flushed_total Rows written to ClickHouse\n\
             # TYPE symthaea_rows_flushed_total counter\n\
             symthaea_rows_flushed_total {}\n\
             # HELP symthaea_clickhouse_errors_total ClickHouse write failures\n\
             # TYPE symthaea_clickhouse_errors_total counter\n\
             symthaea_clickhouse_errors_total {}\n\
             # HELP symthaea_anomalies_total Consciousness collapse events\n\
             # TYPE symthaea_anomalies_total counter\n\
             symthaea_anomalies_total {}\n\
             # HELP symthaea_consciousness_level Current consciousness level (0-1)\n\
             # TYPE symthaea_consciousness_level gauge\n\
             symthaea_consciousness_level {:.4}\n\
             # HELP symthaea_prediction_error Current prediction error (0-1)\n\
             # TYPE symthaea_prediction_error gauge\n\
             symthaea_prediction_error {:.4}\n\
             # HELP symthaea_urgency_level Urgency (0=cruise 1=normal 2=critical)\n\
             # TYPE symthaea_urgency_level gauge\n\
             symthaea_urgency_level {:.0}\n",
            self.total_cycles.load(Ordering::Relaxed),
            self.total_rows_flushed.load(Ordering::Relaxed),
            self.clickhouse_errors.load(Ordering::Relaxed),
            self.anomalies_detected.load(Ordering::Relaxed),
            consciousness,
            pe,
            urgency_num,
        )
    }
}

// ─── INPUT CORPUS ────────────────────────────────────────────────────────────

const CORPUS: &[&str] = &[
    // Philosophy of mind
    "What is the relationship between information integration and conscious experience?",
    "Does subjective experience require physical substrate or only computational organization?",
    "How does the global workspace broadcast shape the contents of awareness?",
    // Thermodynamics / physics
    "Free energy minimization as the organizing principle of biological cognition",
    "The Carnot efficiency bound and its implications for cognitive work",
    "Entropy production and the arrow of time in self-organizing systems",
    // Mathematics
    "Topological invariants as fingerprints of cognitive state space geometry",
    "Category theory unifies disparate mathematical structures through morphism composition",
    "Persistent homology tracks the birth and death of topological features across scales",
    // Neuroscience
    "The thalamo-cortical loop gates sensory salience through predictive suppression",
    "Hippocampal theta rhythm coordinates memory consolidation during sleep",
    "Dopaminergic prediction error signals drive reinforcement learning in the basal ganglia",
    // Language and meaning
    "Meaning emerges from the differential use of signs within a relational network",
    "Metaphor is not ornamental but constitutive of conceptual structure",
    // Reflection
    "What pattern in my recent reasoning reflects the highest epistemic quality?",
    "How does temporal binding create the unity of conscious experience moment to moment?",
];

// ─── CLICKHOUSE ────────────────────────────────────────────────────────────────

const CREATE_TABLE_SQL: &str = "\
CREATE TABLE IF NOT EXISTS symthaea_cycles (\
    ts                    DateTime64(3) DEFAULT now64(),\
    ts_ms                 UInt64,\
    cycle_num             UInt64,\
    consciousness_level   Float64,\
    narrative_self_psi    Float64,\
    living_mind_vitality  Float64,\
    resonance_frequency   Float64,\
    prediction_error      Float32,\
    peak_attention        Float32,\
    reasoning_confidence  Float32,\
    dopamine              Float32,\
    noradrenaline         Float32,\
    serotonin             Float32,\
    acetylcholine         Float32,\
    micro_phi             Float64,\
    meso_phi              Float64,\
    macro_phi             Float64,\
    temporal_coherence    Float64,\
    hierarchical_free_energy Float64,\
    mood_temperature      Float32,\
    learning_occurred     UInt8,\
    surprise_triggered    UInt8,\
    prefrontal_veto       UInt8,\
    cycle_time_us         UInt64,\
    urgency               LowCardinality(String),\
    language_output        String\
) ENGINE = MergeTree()\
  PARTITION BY toYYYYMMDD(ts)\
  ORDER BY (ts, cycle_num)\
  TTL ts + INTERVAL 30 DAY\
";

async fn clickhouse_query(client: &reqwest::Client, base_url: &str, db: &str, sql: &str) -> Result<()> {
    let url = format!("{}/?database={}", base_url, db);
    let resp = client.post(&url).body(sql.to_string()).send().await
        .context("ClickHouse request failed")?;
    let status = resp.status();
    if !status.is_success() {
        let body = resp.text().await.unwrap_or_default();
        anyhow::bail!("ClickHouse error {status}: {body}");
    }
    Ok(())
}

async fn flush_to_clickhouse(
    client: &reqwest::Client,
    base_url: &str,
    db: &str,
    table: &str,
    rows: &[CycleRow],
) -> Result<()> {
    if rows.is_empty() { return Ok(()); }
    let mut body = String::with_capacity(rows.len() * 200);
    for row in rows {
        body.push_str(&serde_json::to_string(row)?);
        body.push('\n');
    }
    let url = format!("{}/?database={}&query=INSERT+INTO+{}+FORMAT+JSONEachRow", base_url, db, table);
    let resp = client.post(&url)
        .header("Content-Type", "application/octet-stream")
        .body(body)
        .send().await
        .context("ClickHouse insert failed")?;
    let status = resp.status();
    if !status.is_success() {
        let body = resp.text().await.unwrap_or_default();
        anyhow::bail!("ClickHouse insert error {status}: {body}");
    }
    Ok(())
}

// ─── MAIN ────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt().with_target(false).with_level(true).init();

    let args = Args::parse();
    let http = reqwest::Client::builder().timeout(Duration::from_secs(10)).build()?;

    if args.init_db {
        info!("Initializing ClickHouse table {}…", args.table);
        clickhouse_query(&http, &args.clickhouse, &args.db, CREATE_TABLE_SQL).await?;
        info!("Table ready.");
        return Ok(());
    }

    info!(
        "Symthaea Watch — {}Hz → {} — metrics :{}",
        args.cycles_per_second, args.clickhouse, args.metrics_port
    );

    let metrics = Arc::new(Metrics::default());
    let (tx, mut rx) = mpsc::channel::<CycleRow>(4096);

    // ── Cognitive loop (blocking thread) ─────────────────────────────────
    let cycle_interval_us = if args.no_throttle {
        0 // No sleep between cycles — run at natural speed
    } else {
        1_000_000u64 / args.cycles_per_second.max(1) as u64
    };
    let collapse_threshold = args.collapse_threshold;
    let collapse_window = args.collapse_window;
    let metrics_for_loop = Arc::clone(&metrics);

    tokio::task::spawn_blocking(move || {
        info!("Building CognitiveLoopService…");
        let config = CognitiveLoopConfig {
            // 60s budget — prevents the budget-exceeded → safety-escalation doom loop
            // in debug builds where cycles naturally take 1-5 seconds.
            attention_budget_override_us: Some(60_000_000),
            causal_enhancement: true,
            // CfC backend: He-initialized output weights produce non-zero predictions
            // from cycle 1. The default HdcLtcUnified has near-zero network output
            // (closed-form evolution + project_from_hdc 1e-10 skip threshold) producing
            // zero-vector predictions → norm_pred=0 → PE=1.0 forever.
            temporal_backend: TemporalBackend::CfC,
            // Online learning: CfC adapts weights from each cycle's prediction error.
            enable_online_learning: true,
            // Higher learning threshold: default 0.05 means Critical at PE>0.15.
            // With CfC still learning (PE 0.5-0.8), the system is permanently Critical.
            // 0.3 sets Critical at PE>0.9, Normal at PE>0.3, Cruise below 0.3.
            learning_threshold: 0.3,
            ..Default::default()
        };
        let mut service = match CognitiveLoopService::new(config) {
            Ok(s) => s,
            Err(e) => { error!("Failed to create CognitiveLoopService: {e}"); return; }
        };

        // Drain safety alerts — log them as info, don't let the channel fill up
        if let Some(rx) = service.take_safety_alert_receiver() {
            std::thread::spawn(move || {
                while let Ok(alert) = rx.recv() {
                    info!(kind = ?alert.kind, cycle = alert.cycle, "Safety alert: {}", alert.message);
                }
            });
        }

        info!("CognitiveLoopService ready — entering cognitive loop");

        let mut cycle_num: u64 = 0;
        let mut corpus_idx: usize = 0;
        let mut collapse_streak: u32 = 0;

        loop {
            let tick = Instant::now();
            let input = CORPUS[corpus_idx % CORPUS.len()];
            corpus_idx += 1;

            let result = service.cycle(input);
            let m = &result.metadata;
            cycle_num += 1;
            metrics_for_loop.total_cycles.fetch_add(1, Ordering::Relaxed);

            let psi = m.consciousness.consciousness_level;
            *metrics_for_loop.last_consciousness.lock().unwrap() = psi;
            *metrics_for_loop.last_prediction_error.lock().unwrap() = result.prediction_error;
            *metrics_for_loop.last_urgency.lock().unwrap() = format!("{:?}", m.urgency).to_lowercase();

            // Anomaly: consciousness collapse
            if psi < collapse_threshold {
                collapse_streak += 1;
                if collapse_streak >= collapse_window {
                    metrics_for_loop.anomalies_detected.fetch_add(1, Ordering::Relaxed);
                    warn!(
                        cycle = cycle_num, psi = psi, pe = result.prediction_error,
                        "ANOMALY: consciousness collapse ({} consecutive cycles below {:.2})",
                        collapse_streak, collapse_threshold
                    );
                    collapse_streak = 0;
                }
            } else {
                collapse_streak = 0;
            }

            let ts_ms = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64;

            let row = CycleRow {
                ts_ms,
                cycle_num,
                // Consciousness
                consciousness_level: m.consciousness.consciousness_level,
                narrative_self_psi: m.narrative_self_psi,
                living_mind_vitality: m.living_mind_vitality,
                resonance_frequency: m.resonance_frequency,
                // Prediction
                prediction_error: result.prediction_error,
                peak_attention: result.peak_attention,
                reasoning_confidence: m.reasoning_confidence,
                // Neuromodulators
                dopamine: m.neuromod.dopamine_effective,
                noradrenaline: m.neuromod.noradrenaline_effective,
                serotonin: m.neuromod.serotonin_effective,
                acetylcholine: m.neuromod.acetylcholine_effective,
                // Phi decomposition
                micro_phi: m.structural.structural_micro_phi,
                meso_phi: m.structural.structural_meso_phi,
                macro_phi: m.structural.structural_macro_phi,
                // Temporal
                temporal_coherence: m.temporal.temporal_coherence_score,
                // Thermodynamic
                hierarchical_free_energy: m.hierarchical_total_free_energy,
                // Affect
                mood_temperature: m.embodied.mood_temperature,
                // Cycle info
                learning_occurred: result.learning_occurred as u8,
                surprise_triggered: m.surprise_triggered as u8,
                prefrontal_veto: m.prefrontal_veto as u8,
                cycle_time_us: result.cycle_time_us,
                urgency: format!("{:?}", m.urgency).to_lowercase(),
                language_output: result.language_output.clone().unwrap_or_default(),
            };

            let _ = tx.try_send(row);

            // Optional throttle — skip when --no-throttle for natural-speed operation
            if cycle_interval_us > 0 {
                let elapsed_us = tick.elapsed().as_micros() as u64;
                if elapsed_us < cycle_interval_us {
                    std::thread::sleep(Duration::from_micros(cycle_interval_us - elapsed_us));
                }
            }
        }
    });

    // ── Flush task: batch rows → ClickHouse every second ─────────────────
    let ch_url = args.clickhouse.clone();
    let db = args.db.clone();
    let table = args.table.clone();
    let http_flush = http.clone();
    let metrics_for_flush = Arc::clone(&metrics);

    tokio::spawn(async move {
        let mut batch: Vec<CycleRow> = Vec::with_capacity(256);
        let mut flush_interval = tokio::time::interval(Duration::from_secs(1));

        loop {
            tokio::select! {
                row = rx.recv() => {
                    match row {
                        Some(r) => batch.push(r),
                        None => break,
                    }
                }
                _ = flush_interval.tick() => {
                    if batch.is_empty() { continue; }
                    let to_flush = std::mem::replace(&mut batch, Vec::with_capacity(256));
                    let n = to_flush.len();
                    match flush_to_clickhouse(&http_flush, &ch_url, &db, &table, &to_flush).await {
                        Ok(()) => {
                            metrics_for_flush.total_rows_flushed.fetch_add(n as u64, Ordering::Relaxed);
                            info!("Flushed {n} rows (total: {})",
                                metrics_for_flush.total_rows_flushed.load(Ordering::Relaxed));
                        }
                        Err(e) => {
                            metrics_for_flush.clickhouse_errors.fetch_add(1, Ordering::Relaxed);
                            error!("ClickHouse flush error: {e}");
                        }
                    }
                }
            }
        }
    });

    // ── Prometheus /metrics HTTP server ──────────────────────────────────
    let metrics_for_http = Arc::clone(&metrics);
    let app = Router::new().route(
        "/metrics",
        get(move || {
            let m = Arc::clone(&metrics_for_http);
            async move { m.prometheus_text() }
        }),
    );

    let addr = format!("127.0.0.1:{}", args.metrics_port);
    let listener = tokio::net::TcpListener::bind(&addr).await
        .with_context(|| format!("Failed to bind metrics server on {addr}"))?;

    info!("Prometheus metrics at http://{addr}/metrics");
    info!("Press Ctrl-C to stop");

    axum::serve(listener, app).await?;
    Ok(())
}
