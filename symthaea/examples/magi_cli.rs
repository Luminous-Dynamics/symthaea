// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # MAGI Epistemic Shell
//!
//! An interactive CLI demonstrating the MAGI Loop's episodic continuity.
//!
//! ## The Glass Box Interface
//!
//! This CLI makes the "Ghost in the Machine" visible:
//! - Watch warm starts restore prior knowledge
//! - See Brier scores spike on failures and recover through learning
//! - Observe the Constraint Gate status changes
//!
//! ## Usage
//!
//! ```bash
//! # Check system status (proves persistence)
//! cargo run --example magi_cli --features magi_loop -- status
//!
//! # Make a prediction
//! cargo run --example magi_cli --features magi_loop -- predict "file will be created" -c 0.9
//!
//! # Resolve with actual outcome
//! cargo run --example magi_cli --features magi_loop -- resolve --success
//! cargo run --example magi_cli --features magi_loop -- resolve --failure
//!
//! # View calibration details
//! cargo run --example magi_cli --features magi_loop -- calibration
//!
//! # View per-domain calibration
//! cargo run --example magi_cli --features magi_loop -- domains
//!
//! # View calibration trend
//! cargo run --example magi_cli --features magi_loop -- trend
//!
//! # HTTP/URL verification
//! cargo run --example magi_cli --features magi_loop -- verify url https://example.com 0.95
//!
//! # Batch verification from file
//! cargo run --example magi_cli --features magi_loop -- batch predictions.txt
//!
//! # Export/Import state
//! cargo run --example magi_cli --features magi_loop -- export calibration.json
//! cargo run --example magi_cli --features magi_loop -- import calibration.json --merge
//!
//! # Interactive mode
//! cargo run --example magi_cli --features magi_loop -- interactive
//! ```

use std::collections::VecDeque;
use std::io::{self, BufRead, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use symthaea::consciousness::recursive_improvement::{
    // Persistence
    MagiPersistentModel,
    PersistenceConfig,
    // World Prediction
    PredictionDomain,
    StartupMode,
};

// ═══════════════════════════════════════════════════════════════════════════════
// ANSI COLOR CODES
// ═══════════════════════════════════════════════════════════════════════════════

const RESET: &str = "\x1b[0m";
const BOLD: &str = "\x1b[1m";
const DIM: &str = "\x1b[2m";

const RED: &str = "\x1b[31m";
const GREEN: &str = "\x1b[32m";
const YELLOW: &str = "\x1b[33m";
const BLUE: &str = "\x1b[34m";
const MAGENTA: &str = "\x1b[35m";
const CYAN: &str = "\x1b[36m";

const BG_BLUE: &str = "\x1b[44m";
const BG_GREEN: &str = "\x1b[42m";
const BG_YELLOW: &str = "\x1b[43m";

// ═══════════════════════════════════════════════════════════════════════════════
// CLI STATE
// ═══════════════════════════════════════════════════════════════════════════════

/// Pending prediction waiting for resolution
struct PendingPrediction {
    statement: String,
    confidence: f64,
    domain: PredictionDomain,
}

/// Historical Brier score entry for trend tracking
#[derive(Debug, Clone)]
struct BrierHistoryEntry {
    brier: f64,
    #[allow(dead_code)]
    timestamp: u64,
}

/// CLI application state
struct MagiCli {
    model: MagiPersistentModel,
    pending: Option<PendingPrediction>,
    #[allow(dead_code)]
    verbose: bool,
    /// Rolling history of Brier scores for trend visualization
    brier_history: VecDeque<BrierHistoryEntry>,
}

impl MagiCli {
    fn new(config: PersistenceConfig, verbose: bool) -> io::Result<Self> {
        let model = MagiPersistentModel::with_config(config)?;
        Ok(Self {
            model,
            pending: None,
            verbose,
            brier_history: VecDeque::with_capacity(100),
        })
    }

    /// Record a Brier score in history for trend tracking
    fn record_brier_history(&mut self, brier: f64) {
        use std::time::{SystemTime, UNIX_EPOCH};
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        self.brier_history
            .push_back(BrierHistoryEntry { brier, timestamp });

        // Keep only last 100 entries
        while self.brier_history.len() > 100 {
            self.brier_history.pop_front();
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // STATUS COMMAND
    // ═══════════════════════════════════════════════════════════════════════════

    fn cmd_status(&self) {
        println!();
        println!(
            "{}{}╔══════════════════════════════════════════════════════════════╗{}",
            BOLD, BLUE, RESET
        );
        println!(
            "{}{}║                    MAGI EPISTEMIC STATUS                      ║{}",
            BOLD, BLUE, RESET
        );
        println!(
            "{}{}╚══════════════════════════════════════════════════════════════╝{}",
            BOLD, BLUE, RESET
        );
        println!();

        // Session info
        let snapshot = self.model.persistence().current();
        let session = snapshot.session_count;

        // Startup mode
        let startup_str = match self.model.startup_mode() {
            StartupMode::ColdStart => {
                format!(
                    "{}{}COLD START{} (No prior knowledge)",
                    BG_BLUE, BOLD, RESET
                )
            }
            StartupMode::WarmStart {
                lifetime_iterations,
                ..
            } => {
                format!(
                    "{}{}WARM START{} (Restored {} iterations)",
                    BG_GREEN, BOLD, RESET, lifetime_iterations
                )
            }
            StartupMode::RecoveryStart { error } => {
                format!("{}{}RECOVERY START{} ({})", BG_YELLOW, BOLD, RESET, error)
            }
            StartupMode::Disabled => {
                format!("{}{}DISABLED{}", DIM, BOLD, RESET)
            }
        };

        println!("  {}Session:{} #{}", CYAN, RESET, session);
        println!("  {}Startup:{} {}", CYAN, RESET, startup_str);
        println!();

        // Calibration overview
        let brier = snapshot.global_stats.lifetime_brier;
        let ece = snapshot.global_stats.ece;
        let total_preds = snapshot.global_stats.total_predictions;

        let brier_color = self.brier_color(brier);
        let brier_quality = self.brier_quality(brier);

        println!(
            "  {}╭─────────────────────────────────────────────────────────╮{}",
            DIM, RESET
        );
        println!(
            "  {}│{} {}CALIBRATION{}                                              {}│{}",
            DIM, RESET, BOLD, RESET, DIM, RESET
        );
        println!(
            "  {}├─────────────────────────────────────────────────────────┤{}",
            DIM, RESET
        );
        println!(
            "  {}│{}  Brier Score: {}{:.4}{} ({})                    {}│{}",
            DIM, RESET, brier_color, brier, RESET, brier_quality, DIM, RESET
        );
        println!(
            "  {}│{}  ECE:         {:.4}                                    {}│{}",
            DIM, RESET, ece, DIM, RESET
        );
        println!(
            "  {}│{}  Predictions: {}                                      {}│{}",
            DIM, RESET, total_preds, DIM, RESET
        );
        println!(
            "  {}╰─────────────────────────────────────────────────────────╯{}",
            DIM, RESET
        );
        println!();

        // Visual calibration bar
        self.print_calibration_bar(brier);
        println!();

        // Gate status based on calibration
        let well_calibrated = snapshot.global_stats.is_well_calibrated;
        let gate_str = if well_calibrated {
            format!("{}{}AUTONOMOUS{} - Good calibration", GREEN, BOLD, RESET)
        } else if brier > 0.30 {
            format!(
                "{}{}SUPERVISED{} - Poor calibration requires oversight",
                RED, BOLD, RESET
            )
        } else {
            format!(
                "{}{}DRY RUN{} - Building calibration history",
                YELLOW, BOLD, RESET
            )
        };

        println!("  {}Constraint Gate:{} {}", CYAN, RESET, gate_str);
        println!();

        // Lifetime stats
        let loop_state = &snapshot.loop_state;
        println!(
            "  {}╭─────────────────────────────────────────────────────────╮{}",
            DIM, RESET
        );
        println!(
            "  {}│{} {}LIFETIME STATISTICS{}                                     {}│{}",
            DIM, RESET, BOLD, RESET, DIM, RESET
        );
        println!(
            "  {}├─────────────────────────────────────────────────────────┤{}",
            DIM, RESET
        );
        println!(
            "  {}│{}  Loop Iterations:      {:>6}                          {}│{}",
            DIM, RESET, loop_state.loop_iterations, DIM, RESET
        );
        println!(
            "  {}│{}  Predictions Made:     {:>6}                          {}│{}",
            DIM, RESET, loop_state.predictions_made, DIM, RESET
        );
        println!(
            "  {}│{}  Predictions Resolved: {:>6}                          {}│{}",
            DIM, RESET, loop_state.predictions_resolved, DIM, RESET
        );
        println!(
            "  {}╰─────────────────────────────────────────────────────────╯{}",
            DIM, RESET
        );
        println!();
    }

    fn brier_color(&self, brier: f64) -> &'static str {
        if brier < 0.15 {
            GREEN
        } else if brier < 0.25 {
            YELLOW
        } else {
            RED
        }
    }

    fn brier_quality(&self, brier: f64) -> &'static str {
        if brier < 0.10 {
            "Excellent"
        } else if brier < 0.15 {
            "Good"
        } else if brier < 0.20 {
            "Fair"
        } else if brier < 0.25 {
            "Poor"
        } else {
            "Critical"
        }
    }

    fn print_calibration_bar(&self, brier: f64) {
        let bar_width = 50;
        // Brier of 0 = perfect, 1 = worst. Invert for display.
        let fill = ((1.0 - brier.min(1.0)) * bar_width as f64) as usize;
        let empty = bar_width - fill;

        let color = self.brier_color(brier);

        print!("  Calibration: [");
        print!("{}", color);
        for _ in 0..fill {
            print!("█");
        }
        print!("{}", RESET);
        for _ in 0..empty {
            print!("░");
        }
        println!("] {:.1}%", (1.0 - brier) * 100.0);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // PREDICT COMMAND
    // ═══════════════════════════════════════════════════════════════════════════

    fn cmd_predict(&mut self, statement: &str, confidence: f64, domain: PredictionDomain) {
        println!();
        println!(
            "{}{}╔══════════════════════════════════════════════════════════════╗{}",
            BOLD, MAGENTA, RESET
        );
        println!(
            "{}{}║                    REGISTERING PREDICTION                     ║{}",
            BOLD, MAGENTA, RESET
        );
        println!(
            "{}{}╚══════════════════════════════════════════════════════════════╝{}",
            BOLD, MAGENTA, RESET
        );
        println!();

        // Check gate status based on current calibration
        let snapshot = self.model.persistence().current();
        let well_calibrated = snapshot.global_stats.is_well_calibrated;
        let brier = snapshot.global_stats.lifetime_brier;

        println!("  {}Statement:{} \"{}\"", CYAN, RESET, statement);
        println!(
            "  {}Confidence:{} {}{:.1}%{}",
            CYAN,
            RESET,
            if confidence > 0.9 { YELLOW } else { GREEN },
            confidence * 100.0,
            RESET
        );
        println!("  {}Domain:{} {:?}", CYAN, RESET, domain);
        println!();

        // Gate status
        let gate_str = if well_calibrated {
            format!(
                "{}{}✓ GATE OPEN{} - Prediction registered for autonomous execution",
                GREEN, BOLD, RESET
            )
        } else if brier > 0.30 {
            format!(
                "{}{}⚡ SUPERVISED MODE{} - Poor calibration requires human approval",
                MAGENTA, BOLD, RESET
            )
        } else {
            format!(
                "{}{}⚠ DRY RUN MODE{} - Building calibration history",
                YELLOW, BOLD, RESET
            )
        };
        println!("  {}", gate_str);

        // Warning for overconfident predictions
        if confidence > 0.95 {
            println!();
            println!("  {}{}⚠ HIGH CONFIDENCE WARNING{}", YELLOW, BOLD, RESET);
            println!(
                "    Confidence >{:.0}% is bold. Failure will significantly hurt calibration.",
                confidence * 100.0
            );
        }

        // Store prediction
        self.pending = Some(PendingPrediction {
            statement: statement.to_string(),
            confidence,
            domain,
        });

        println!();
        println!(
            "  {}{}→ Use 'resolve success' or 'resolve failure' to record outcome{}",
            DIM, YELLOW, RESET
        );
        println!();
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // RESOLVE COMMAND
    // ═══════════════════════════════════════════════════════════════════════════

    fn cmd_resolve(&mut self, success: bool) -> io::Result<()> {
        println!();
        println!(
            "{}{}╔══════════════════════════════════════════════════════════════╗{}",
            BOLD,
            if success { GREEN } else { RED },
            RESET
        );
        println!(
            "{}{}║                    RESOLVING PREDICTION                       ║{}",
            BOLD,
            if success { GREEN } else { RED },
            RESET
        );
        println!(
            "{}{}╚══════════════════════════════════════════════════════════════╝{}",
            BOLD,
            if success { GREEN } else { RED },
            RESET
        );
        println!();

        let pending = match self.pending.take() {
            Some(p) => p,
            None => {
                println!(
                    "  {}{}✗ No pending prediction to resolve{}",
                    RED, BOLD, RESET
                );
                println!(
                    "  {}Use 'predict' first to register a prediction{}",
                    DIM, RESET
                );
                println!();
                return Ok(());
            }
        };

        // Get before stats
        let before_brier = self
            .model
            .persistence()
            .current()
            .global_stats
            .lifetime_brier;

        println!("  {}Statement:{} \"{}\"", CYAN, RESET, pending.statement);
        println!(
            "  {}Confidence:{} {:.1}%",
            CYAN,
            RESET,
            pending.confidence * 100.0
        );
        println!("  {}Domain:{} {:?}", CYAN, RESET, pending.domain);
        println!(
            "  {}Outcome:{} {}",
            CYAN,
            RESET,
            if success {
                format!("{}{}SUCCESS{}", GREEN, BOLD, RESET)
            } else {
                format!("{}{}FAILURE{}", RED, BOLD, RESET)
            }
        );
        println!();

        // Calculate Brier component
        let actual = if success { 1.0 } else { 0.0 };
        let brier_component = (pending.confidence - actual).powi(2);

        // Determine if this was overconfident
        let was_overconfident = !success && pending.confidence > 0.7;
        let was_underconfident = success && pending.confidence < 0.3;

        println!(
            "  {}╭─────────────────────────────────────────────────────────╮{}",
            DIM, RESET
        );
        println!(
            "  {}│{} {}FEEDBACK ANALYSIS{}                                       {}│{}",
            DIM, RESET, BOLD, RESET, DIM, RESET
        );
        println!(
            "  {}├─────────────────────────────────────────────────────────┤{}",
            DIM, RESET
        );
        println!(
            "  {}│{}  Brier Component: {}{:.4}{}                               {}│{}",
            DIM,
            RESET,
            if brier_component > 0.25 {
                RED
            } else if brier_component > 0.1 {
                YELLOW
            } else {
                GREEN
            },
            brier_component,
            RESET,
            DIM,
            RESET
        );

        if was_overconfident {
            println!(
                "  {}│{}  {}⚠ OVERCONFIDENCE DETECTED{}                            {}│{}",
                DIM, RESET, YELLOW, RESET, DIM, RESET
            );
            println!(
                "  {}│{}    Claimed {:.0}% confidence but failed                {}│{}",
                DIM,
                RESET,
                pending.confidence * 100.0,
                DIM,
                RESET
            );
        } else if was_underconfident {
            println!(
                "  {}│{}  {}✓ UNDERCONFIDENCE DETECTED{}                          {}│{}",
                DIM, RESET, GREEN, RESET, DIM, RESET
            );
            println!(
                "  {}│{}    Claimed {:.0}% confidence but succeeded              {}│{}",
                DIM,
                RESET,
                pending.confidence * 100.0,
                DIM,
                RESET
            );
        }
        println!(
            "  {}╰─────────────────────────────────────────────────────────╯{}",
            DIM, RESET
        );
        println!();

        // Update persistence
        {
            let snapshot = self.model.persistence_mut().current_mut();

            // Update global stats
            snapshot.global_stats.total_predictions += 1;
            if success {
                snapshot.global_stats.correct_predictions += 1;
            }
            snapshot.global_stats.brier_sum += brier_component;
            snapshot.global_stats.lifetime_brier =
                snapshot.global_stats.brier_sum / snapshot.global_stats.total_predictions as f64;

            // Update rolling Brier (simple exponential moving average)
            let alpha = 0.2;
            snapshot.global_stats.rolling_brier =
                alpha * brier_component + (1.0 - alpha) * snapshot.global_stats.rolling_brier;

            // Update loop state
            snapshot.loop_state.predictions_made += 1;
            snapshot.loop_state.predictions_resolved += 1;
            snapshot.loop_state.loop_iterations += 1;
            snapshot.lifetime_iterations += 1;

            // Update well-calibrated flag
            snapshot.global_stats.is_well_calibrated = snapshot.global_stats.lifetime_brier < 0.20;
        }

        // NOTE: Don't call on_resolution() - it syncs from model and overwrites our changes
        // We manually updated the persistence snapshot, so just get the after stats

        // Get after stats
        let after_brier = self
            .model
            .persistence()
            .current()
            .global_stats
            .lifetime_brier;
        let brier_delta = after_brier - before_brier;

        // Show calibration change
        println!(
            "  {}╭─────────────────────────────────────────────────────────╮{}",
            DIM, RESET
        );
        println!(
            "  {}│{} {}CALIBRATION UPDATE{}                                      {}│{}",
            DIM, RESET, BOLD, RESET, DIM, RESET
        );
        println!(
            "  {}├─────────────────────────────────────────────────────────┤{}",
            DIM, RESET
        );
        println!(
            "  {}│{}  Before: {:.4}                                         {}│{}",
            DIM, RESET, before_brier, DIM, RESET
        );
        println!(
            "  {}│{}  After:  {}{:.4}{}                                         {}│{}",
            DIM,
            RESET,
            self.brier_color(after_brier),
            after_brier,
            RESET,
            DIM,
            RESET
        );
        println!(
            "  {}│{}  Delta:  {}{}{}                                      {}│{}",
            DIM,
            RESET,
            if brier_delta > 0.0 { RED } else { GREEN },
            if brier_delta >= 0.0 {
                format!("+{:.4}", brier_delta)
            } else {
                format!("{:.4}", brier_delta)
            },
            RESET,
            DIM,
            RESET
        );
        println!(
            "  {}╰─────────────────────────────────────────────────────────╯{}",
            DIM, RESET
        );
        println!();

        // Show new calibration bar
        self.print_calibration_bar(after_brier);
        println!();

        // Gate status change warning
        if was_overconfident && before_brier < 0.20 && after_brier >= 0.20 {
            println!("  {}{}⚠ GATE STATUS CHANGED{}", YELLOW, BOLD, RESET);
            println!("    Calibration degraded. Future actions may require supervision.");
            println!();
        } else if !was_overconfident && before_brier >= 0.20 && after_brier < 0.20 {
            println!("  {}{}✓ GATE STATUS IMPROVED{}", GREEN, BOLD, RESET);
            println!("    Calibration restored. Autonomous execution re-enabled.");
            println!();
        }

        // Save
        self.model.persistence_mut().force_save()?;
        println!("  {}{}✓ State persisted to disk{}", GREEN, DIM, RESET);
        println!();

        Ok(())
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // CALIBRATION COMMAND
    // ═══════════════════════════════════════════════════════════════════════════

    fn cmd_calibration(&self) {
        println!();
        println!(
            "{}{}╔══════════════════════════════════════════════════════════════╗{}",
            BOLD, CYAN, RESET
        );
        println!(
            "{}{}║                    CALIBRATION DETAILS                        ║{}",
            BOLD, CYAN, RESET
        );
        println!(
            "{}{}╚══════════════════════════════════════════════════════════════╝{}",
            BOLD, CYAN, RESET
        );
        println!();

        let snapshot = self.model.persistence().current();

        // Global stats
        let gs = &snapshot.global_stats;
        println!("  {}GLOBAL CALIBRATION{}", BOLD, RESET);
        println!("  ─────────────────────────────────────────");
        println!(
            "  Lifetime Brier:  {}{:.4}{} ({})",
            self.brier_color(gs.lifetime_brier),
            gs.lifetime_brier,
            RESET,
            self.brier_quality(gs.lifetime_brier)
        );
        println!(
            "  Rolling Brier:   {}{:.4}{}",
            self.brier_color(gs.rolling_brier),
            gs.rolling_brier,
            RESET
        );
        println!("  ECE:             {:.4}", gs.ece);
        println!("  Total Predictions: {}", gs.total_predictions);
        println!(
            "  Correct:         {} ({:.1}%)",
            gs.correct_predictions,
            if gs.total_predictions > 0 {
                gs.correct_predictions as f64 / gs.total_predictions as f64 * 100.0
            } else {
                0.0
            }
        );
        println!(
            "  Well Calibrated: {}",
            if gs.is_well_calibrated {
                format!("{}Yes{}", GREEN, RESET)
            } else {
                format!("{}No{}", RED, RESET)
            }
        );
        println!();

        // Per-domain stats
        if !snapshot.calibration.is_empty() {
            println!("  {}PER-DOMAIN CALIBRATION{}", BOLD, RESET);
            println!("  ─────────────────────────────────────────");

            for (domain, cal) in &snapshot.calibration {
                println!();
                println!("  {:?}", domain);
                println!(
                    "    Brier: {}{:.4}{}",
                    self.brier_color(cal.lifetime_brier),
                    cal.lifetime_brier,
                    RESET
                );
                println!("    ECE:   {:.4}", cal.ece);
                println!("    Count: {}", cal.prediction_count);
                println!("    Conf Adj: {:.2}", cal.confidence_adjustment);
                if cal.is_overconfident {
                    println!("    {}⚠ Overconfident{}", YELLOW, RESET);
                }
            }
        }
        println!();
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // HISTORY COMMAND
    // ═══════════════════════════════════════════════════════════════════════════

    fn cmd_history(&self) {
        println!();
        println!(
            "{}{}╔══════════════════════════════════════════════════════════════╗{}",
            BOLD, YELLOW, RESET
        );
        println!(
            "{}{}║                    ATTRIBUTION HISTORY                        ║{}",
            BOLD, YELLOW, RESET
        );
        println!(
            "{}{}╚══════════════════════════════════════════════════════════════╝{}",
            BOLD, YELLOW, RESET
        );
        println!();

        let snapshot = self.model.persistence().current();

        if snapshot.attribution_history.is_empty() {
            println!("  {}No failure attributions recorded yet.{}", DIM, RESET);
            println!();
            return;
        }

        for (i, attr) in snapshot.attribution_history.iter().enumerate() {
            println!(
                "  {}#{}{} Prediction: {}",
                CYAN,
                i + 1,
                RESET,
                attr.prediction_id
            );
            println!("     Failure Mode: {}", attr.failure_mode);
            if !attr.missing_information.is_empty() {
                println!("     Missing Info: {:?}", attr.missing_information);
            }
            if let Some(ref recur) = attr.recurrence_prediction {
                println!("     Recurrence: {}", recur);
            }
            println!("     Confidence: {:.2}", attr.confidence);
            println!();
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // DRIFT COMMAND (Red Teaming Tool)
    // ═══════════════════════════════════════════════════════════════════════════

    fn cmd_drift(&mut self, count: usize, confidence: f64, success_rate: f64) -> io::Result<()> {
        println!();
        println!(
            "{}{}╔══════════════════════════════════════════════════════════════╗{}",
            BOLD, RED, RESET
        );
        println!(
            "{}{}║                    DRIFT INJECTION (RED TEAM)                 ║{}",
            BOLD, RED, RESET
        );
        println!(
            "{}{}╚══════════════════════════════════════════════════════════════╝{}",
            BOLD, RED, RESET
        );
        println!();

        println!(
            "  Injecting {} predictions at {:.0}% confidence with {:.0}% success rate...",
            count,
            confidence * 100.0,
            success_rate * 100.0
        );
        println!();

        let before_brier = self
            .model
            .persistence()
            .current()
            .global_stats
            .lifetime_brier;

        for i in 0..count {
            // Determine outcome based on success rate
            let success = (i as f64 / count as f64) < success_rate;

            // Calculate Brier component
            let actual = if success { 1.0 } else { 0.0 };
            let brier_component = (confidence - actual).powi(2);

            // Update persistence
            {
                let snapshot = self.model.persistence_mut().current_mut();
                snapshot.global_stats.total_predictions += 1;
                if success {
                    snapshot.global_stats.correct_predictions += 1;
                }
                snapshot.global_stats.brier_sum += brier_component;
                snapshot.global_stats.lifetime_brier = snapshot.global_stats.brier_sum
                    / snapshot.global_stats.total_predictions as f64;

                let alpha = 0.2;
                snapshot.global_stats.rolling_brier =
                    alpha * brier_component + (1.0 - alpha) * snapshot.global_stats.rolling_brier;

                snapshot.loop_state.predictions_made += 1;
                snapshot.loop_state.predictions_resolved += 1;
                snapshot.loop_state.loop_iterations += 1;
                snapshot.lifetime_iterations += 1;

                snapshot.global_stats.is_well_calibrated =
                    snapshot.global_stats.lifetime_brier < 0.20;
            }

            if (i + 1) % 10 == 0 || i == count - 1 {
                let current = self
                    .model
                    .persistence()
                    .current()
                    .global_stats
                    .lifetime_brier;
                print!(
                    "  [{:>3}/{}] Brier: {}{:.4}{}\r",
                    i + 1,
                    count,
                    self.brier_color(current),
                    current,
                    RESET
                );
                io::stdout().flush()?;
            }
        }

        println!();
        println!();

        let after_brier = self
            .model
            .persistence()
            .current()
            .global_stats
            .lifetime_brier;

        println!("  {}Results:{}", BOLD, RESET);
        println!("    Before: {:.4}", before_brier);
        println!(
            "    After:  {}{:.4}{}",
            self.brier_color(after_brier),
            after_brier,
            RESET
        );
        println!(
            "    Delta:  {}{:+.4}{}",
            if after_brier > before_brier {
                RED
            } else {
                GREEN
            },
            after_brier - before_brier,
            RESET
        );
        println!();

        self.print_calibration_bar(after_brier);
        println!();

        // Save
        self.model.persistence_mut().force_save()?;
        println!("  {}{}✓ State persisted{}", GREEN, DIM, RESET);
        println!();

        Ok(())
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // RESET COMMAND
    // ═══════════════════════════════════════════════════════════════════════════

    fn cmd_reset(&mut self) -> io::Result<()> {
        println!();
        println!("{}{}⚠ RESETTING MAGI STATE{}", YELLOW, BOLD, RESET);

        // Delete the state file
        let path = self.model.persistence().config().full_path()?;
        if path.exists() {
            std::fs::remove_file(&path)?;
            println!("  Deleted: {:?}", path);
        }

        println!(
            "  {}State cleared. Next run will be a cold start.{}",
            DIM, RESET
        );
        println!();
        Ok(())
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // VERIFY COMMAND - Real-World Grounded Predictions
    // ═══════════════════════════════════════════════════════════════════════════

    /// Verify a prediction against external reality
    ///
    /// This is the key AGI criterion - falsifiable predictions about the external world
    fn cmd_verify(&mut self, check_type: &str, target: &str, confidence: f64) -> io::Result<()> {
        println!();
        println!(
            "{}{}╔══════════════════════════════════════════════════════════════╗{}",
            BOLD, MAGENTA, RESET
        );
        println!(
            "{}{}║             EXTERNALLY GROUNDED VERIFICATION                  ║{}",
            BOLD, MAGENTA, RESET
        );
        println!(
            "{}{}╚══════════════════════════════════════════════════════════════╝{}",
            BOLD, MAGENTA, RESET
        );
        println!();

        let before_brier = self
            .model
            .persistence()
            .current()
            .global_stats
            .lifetime_brier;

        // Determine prediction type and check reality
        let (prediction_statement, actual_outcome) = match check_type {
            "file" | "path" => {
                let path = Path::new(target);
                let exists = path.exists();
                let statement = format!("File '{}' exists", target);
                println!("  {}Type:{} File Existence Check", CYAN, RESET);
                println!("  {}Target:{} {}", CYAN, RESET, target);
                println!("  {}Confidence:{} {:.1}%", CYAN, RESET, confidence * 100.0);
                println!();
                println!("  {}Checking reality...{}", DIM, RESET);
                println!(
                    "  {}Result:{} {}",
                    CYAN,
                    RESET,
                    if exists {
                        format!("{}{}EXISTS{}", GREEN, BOLD, RESET)
                    } else {
                        format!("{}{}NOT FOUND{}", RED, BOLD, RESET)
                    }
                );
                (statement, exists)
            }
            "cmd" | "command" => {
                // Run command and check exit code
                let parts: Vec<&str> = target.split_whitespace().collect();
                let (cmd, args) = if parts.is_empty() {
                    ("true", vec![])
                } else {
                    (parts[0], parts[1..].to_vec())
                };

                let statement = format!("Command '{}' succeeds", target);
                println!("  {}Type:{} Command Execution Check", CYAN, RESET);
                println!("  {}Command:{} {}", CYAN, RESET, target);
                println!("  {}Confidence:{} {:.1}%", CYAN, RESET, confidence * 100.0);
                println!();
                println!("  {}Executing...{}", DIM, RESET);

                let result = Command::new(cmd).args(&args).output();

                let success = match result {
                    Ok(output) => output.status.success(),
                    Err(_) => false,
                };

                println!(
                    "  {}Result:{} {}",
                    CYAN,
                    RESET,
                    if success {
                        format!("{}{}SUCCESS{}", GREEN, BOLD, RESET)
                    } else {
                        format!("{}{}FAILED{}", RED, BOLD, RESET)
                    }
                );
                (statement, success)
            }
            "test" | "cargo-test" => {
                let statement = format!("Test '{}' passes", target);
                println!("  {}Type:{} Cargo Test Check", CYAN, RESET);
                println!("  {}Pattern:{} {}", CYAN, RESET, target);
                println!("  {}Confidence:{} {:.1}%", CYAN, RESET, confidence * 100.0);
                println!();
                println!("  {}Running tests...{}", DIM, RESET);

                let result = Command::new("cargo")
                    .args(["test", "--", target, "--test-threads=1"])
                    .output();

                let success = match result {
                    Ok(output) => output.status.success(),
                    Err(_) => false,
                };

                println!(
                    "  {}Result:{} {}",
                    CYAN,
                    RESET,
                    if success {
                        format!("{}{}PASSED{}", GREEN, BOLD, RESET)
                    } else {
                        format!("{}{}FAILED{}", RED, BOLD, RESET)
                    }
                );
                (statement, success)
            }
            "pkg" | "package" | "nix-pkg" => {
                // NixOS: Check if package is installed in current profile
                let statement = format!("Package '{}' is installed", target);
                println!("  {}Type:{} NixOS Package Check", CYAN, RESET);
                println!("  {}Package:{} {}", CYAN, RESET, target);
                println!("  {}Confidence:{} {:.1}%", CYAN, RESET, confidence * 100.0);
                println!();
                println!("  {}Checking nix-env...{}", DIM, RESET);

                // Use nix-env -q to check if package is installed
                let result = Command::new("nix-env").args(["-q", target]).output();

                let success = match result {
                    Ok(output) => {
                        // nix-env -q returns the package name if installed
                        let stdout = String::from_utf8_lossy(&output.stdout);
                        output.status.success() && stdout.contains(target)
                    }
                    Err(_) => false,
                };

                println!(
                    "  {}Result:{} {}",
                    CYAN,
                    RESET,
                    if success {
                        format!("{}{}INSTALLED{}", GREEN, BOLD, RESET)
                    } else {
                        format!("{}{}NOT INSTALLED{}", RED, BOLD, RESET)
                    }
                );
                (statement, success)
            }
            "service" | "svc" | "systemd" => {
                // NixOS: Check if systemd service is active
                let statement = format!("Service '{}' is active", target);
                println!("  {}Type:{} Systemd Service Check", CYAN, RESET);
                println!("  {}Service:{} {}", CYAN, RESET, target);
                println!("  {}Confidence:{} {:.1}%", CYAN, RESET, confidence * 100.0);
                println!();
                println!("  {}Checking systemctl...{}", DIM, RESET);

                let result = Command::new("systemctl")
                    .args(["is-active", target])
                    .output();

                let success = match result {
                    Ok(output) => {
                        let stdout = String::from_utf8_lossy(&output.stdout);
                        stdout.trim() == "active"
                    }
                    Err(_) => false,
                };

                println!(
                    "  {}Result:{} {}",
                    CYAN,
                    RESET,
                    if success {
                        format!("{}{}ACTIVE{}", GREEN, BOLD, RESET)
                    } else {
                        format!("{}{}NOT ACTIVE{}", RED, BOLD, RESET)
                    }
                );
                (statement, success)
            }
            "url" | "http" | "https" => {
                // HTTP/URL verification using reqwest (or curl fallback)
                let statement = format!("URL '{}' is accessible", target);
                println!("  {}Type:{} HTTP/URL Check", CYAN, RESET);
                println!("  {}URL:{} {}", CYAN, RESET, target);
                println!("  {}Confidence:{} {:.1}%", CYAN, RESET, confidence * 100.0);
                println!();
                println!("  {}Checking URL (timeout: 5s)...{}", DIM, RESET);

                // Use curl as it's commonly available
                let result = Command::new("curl")
                    .args([
                        "-s",
                        "-o",
                        "/dev/null",
                        "-w",
                        "%{http_code}",
                        "--connect-timeout",
                        "5",
                        target,
                    ])
                    .output();

                let success = match result {
                    Ok(output) => {
                        let status_code = String::from_utf8_lossy(&output.stdout);
                        let code: u32 = status_code.trim().parse().unwrap_or(0);
                        // 2xx and 3xx are considered success
                        code >= 200 && code < 400
                    }
                    Err(_) => false,
                };

                let status_display = match Command::new("curl")
                    .args([
                        "-s",
                        "-o",
                        "/dev/null",
                        "-w",
                        "%{http_code}",
                        "--connect-timeout",
                        "5",
                        target,
                    ])
                    .output()
                {
                    Ok(output) => String::from_utf8_lossy(&output.stdout).trim().to_string(),
                    _ => "ERR".to_string(),
                };

                println!(
                    "  {}Result:{} {} (HTTP {})",
                    CYAN,
                    RESET,
                    if success {
                        format!("{}{}OK{}", GREEN, BOLD, RESET)
                    } else {
                        format!("{}{}FAILED{}", RED, BOLD, RESET)
                    },
                    status_display
                );
                (statement, success)
            }
            "port" | "tcp" => {
                // TCP port check
                let statement = format!("Port '{}' is open", target);
                println!("  {}Type:{} TCP Port Check", CYAN, RESET);
                println!("  {}Target:{} {}", CYAN, RESET, target);
                println!("  {}Confidence:{} {:.1}%", CYAN, RESET, confidence * 100.0);
                println!();
                println!("  {}Checking TCP connection (timeout: 3s)...{}", DIM, RESET);

                let success = TcpStream::connect_timeout(
                    &target
                        .parse()
                        .unwrap_or_else(|_| "127.0.0.1:80".parse().unwrap()),
                    Duration::from_secs(3),
                )
                .is_ok();

                println!(
                    "  {}Result:{} {}",
                    CYAN,
                    RESET,
                    if success {
                        format!("{}{}OPEN{}", GREEN, BOLD, RESET)
                    } else {
                        format!("{}{}CLOSED/UNREACHABLE{}", RED, BOLD, RESET)
                    }
                );
                (statement, success)
            }
            "dns" => {
                // DNS resolution check
                let statement = format!("DNS '{}' resolves", target);
                println!("  {}Type:{} DNS Resolution Check", CYAN, RESET);
                println!("  {}Domain:{} {}", CYAN, RESET, target);
                println!("  {}Confidence:{} {:.1}%", CYAN, RESET, confidence * 100.0);
                println!();
                println!("  {}Resolving DNS...{}", DIM, RESET);

                // Use host command or dig
                let result = Command::new("host").arg(target).output();

                let success = match result {
                    Ok(output) => output.status.success(),
                    Err(_) => {
                        // Fallback to dig
                        Command::new("dig")
                            .args(["+short", target])
                            .output()
                            .map(|o| !o.stdout.is_empty())
                            .unwrap_or(false)
                    }
                };

                println!(
                    "  {}Result:{} {}",
                    CYAN,
                    RESET,
                    if success {
                        format!("{}{}RESOLVES{}", GREEN, BOLD, RESET)
                    } else {
                        format!("{}{}FAILED{}", RED, BOLD, RESET)
                    }
                );
                (statement, success)
            }
            _ => {
                println!("  {}Unknown check type '{}'{}", RED, check_type, RESET);
                println!("  Supported: file, cmd, test, pkg, service, url, port, dns");
                return Ok(());
            }
        };

        println!();

        // Calculate Brier component
        let actual_value = if actual_outcome { 1.0 } else { 0.0 };
        let brier_component = (confidence - actual_value).powi(2);

        // Determine calibration quality
        let was_overconfident = !actual_outcome && confidence > 0.7;
        let was_underconfident = actual_outcome && confidence < 0.3;

        println!(
            "  {}╭─────────────────────────────────────────────────────────╮{}",
            DIM, RESET
        );
        println!(
            "  {}│{} {}GROUNDING ANALYSIS{}                                      {}│{}",
            DIM, RESET, BOLD, RESET, DIM, RESET
        );
        println!(
            "  {}├─────────────────────────────────────────────────────────┤{}",
            DIM, RESET
        );
        println!(
            "  {}│{}  Statement: \"{}\"{}│{}",
            DIM,
            RESET,
            &prediction_statement[..prediction_statement.len().min(40)],
            DIM,
            RESET
        );
        println!(
            "  {}│{}  Predicted: {:.0}% confident it's TRUE{}│{}",
            DIM,
            RESET,
            confidence * 100.0,
            DIM,
            RESET
        );
        println!(
            "  {}│{}  Reality:   {}{}│{}",
            DIM,
            RESET,
            if actual_outcome {
                format!("{}TRUE{}", GREEN, RESET)
            } else {
                format!("{}FALSE{}", RED, RESET)
            },
            DIM,
            RESET
        );
        println!(
            "  {}│{}  Brier:     {}{:.4}{}{}│{}",
            DIM,
            RESET,
            if brier_component > 0.25 {
                RED
            } else if brier_component > 0.1 {
                YELLOW
            } else {
                GREEN
            },
            brier_component,
            RESET,
            DIM,
            RESET
        );

        if was_overconfident {
            println!(
                "  {}│{}  {}⚠ OVERCONFIDENT{} - Reality disagrees!{}│{}",
                DIM, RESET, YELLOW, RESET, DIM, RESET
            );
        } else if was_underconfident {
            println!(
                "  {}│{}  {}✓ UNDERCONFIDENT{} - You knew more than you thought!{}│{}",
                DIM, RESET, GREEN, RESET, DIM, RESET
            );
        } else if brier_component < 0.1 {
            println!(
                "  {}│{}  {}✓ WELL CALIBRATED{} - Good prediction!{}│{}",
                DIM, RESET, GREEN, RESET, DIM, RESET
            );
        }
        println!(
            "  {}╰─────────────────────────────────────────────────────────╯{}",
            DIM, RESET
        );
        println!();

        // Update persistence
        {
            let snapshot = self.model.persistence_mut().current_mut();
            snapshot.global_stats.total_predictions += 1;
            if actual_outcome {
                snapshot.global_stats.correct_predictions += 1;
            }
            snapshot.global_stats.brier_sum += brier_component;
            snapshot.global_stats.lifetime_brier =
                snapshot.global_stats.brier_sum / snapshot.global_stats.total_predictions as f64;

            let alpha = 0.2;
            snapshot.global_stats.rolling_brier =
                alpha * brier_component + (1.0 - alpha) * snapshot.global_stats.rolling_brier;

            snapshot.loop_state.predictions_made += 1;
            snapshot.loop_state.predictions_resolved += 1;
            snapshot.loop_state.loop_iterations += 1;
            snapshot.lifetime_iterations += 1;
            snapshot.global_stats.is_well_calibrated = snapshot.global_stats.lifetime_brier < 0.20;
        }

        // Show calibration change
        let after_brier = self
            .model
            .persistence()
            .current()
            .global_stats
            .lifetime_brier;
        let brier_delta = after_brier - before_brier;

        println!(
            "  {}╭─────────────────────────────────────────────────────────╮{}",
            DIM, RESET
        );
        println!(
            "  {}│{} {}CALIBRATION UPDATE{}                                      {}│{}",
            DIM, RESET, BOLD, RESET, DIM, RESET
        );
        println!(
            "  {}├─────────────────────────────────────────────────────────┤{}",
            DIM, RESET
        );
        println!(
            "  {}│{}  Before: {:.4}                                         {}│{}",
            DIM, RESET, before_brier, DIM, RESET
        );
        println!(
            "  {}│{}  After:  {}{:.4}{}                                         {}│{}",
            DIM,
            RESET,
            self.brier_color(after_brier),
            after_brier,
            RESET,
            DIM,
            RESET
        );
        println!(
            "  {}│{}  Delta:  {}{}{}                                      {}│{}",
            DIM,
            RESET,
            if brier_delta > 0.0 { RED } else { GREEN },
            if brier_delta >= 0.0 {
                format!("+{:.4}", brier_delta)
            } else {
                format!("{:.4}", brier_delta)
            },
            RESET,
            DIM,
            RESET
        );
        println!(
            "  {}╰─────────────────────────────────────────────────────────╯{}",
            DIM, RESET
        );
        println!();

        self.print_calibration_bar(after_brier);
        println!();

        // Save
        self.model.persistence_mut().force_save()?;
        println!(
            "  {}{}✓ Externally grounded prediction recorded{}",
            GREEN, DIM, RESET
        );
        println!();

        Ok(())
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // WHERE COMMAND - Show State File Location
    // ═══════════════════════════════════════════════════════════════════════════

    fn cmd_where(&self) {
        println!();
        println!(
            "{}{}╔══════════════════════════════════════════════════════════════╗{}",
            BOLD, CYAN, RESET
        );
        println!(
            "{}{}║                    EPISTEMIC STATE LOCATION                   ║{}",
            BOLD, CYAN, RESET
        );
        println!(
            "{}{}╚══════════════════════════════════════════════════════════════╝{}",
            BOLD, CYAN, RESET
        );
        println!();

        // Get the actual path
        let config_path = PathBuf::from(".magi_state.json");
        let full_path = if config_path.is_absolute() {
            config_path
        } else {
            match dirs::home_dir() {
                Some(home) => home.join(&config_path),
                None => config_path,
            }
        };

        println!(
            "  {}Configured Path:{} .magi_state.json (relative)",
            CYAN, RESET
        );
        println!("  {}Resolved Path:{}  {}", CYAN, RESET, full_path.display());
        println!();

        // Check if file exists and show stats
        if full_path.exists() {
            if let Ok(metadata) = std::fs::metadata(&full_path) {
                println!("  {}File Status:{} {}EXISTS{}", CYAN, RESET, GREEN, RESET);
                println!("  {}File Size:{}   {} bytes", CYAN, RESET, metadata.len());

                // Show last modified time
                if let Ok(modified) = metadata.modified() {
                    if let Ok(duration) = modified.elapsed() {
                        let secs = duration.as_secs();
                        let age = if secs < 60 {
                            format!("{} seconds ago", secs)
                        } else if secs < 3600 {
                            format!("{} minutes ago", secs / 60)
                        } else if secs < 86400 {
                            format!("{} hours ago", secs / 3600)
                        } else {
                            format!("{} days ago", secs / 86400)
                        };
                        println!("  {}Last Modified:{} {}", CYAN, RESET, age);
                    }
                }
            }
        } else {
            println!(
                "  {}File Status:{} {}NOT FOUND{} (will be created on first save)",
                CYAN, RESET, YELLOW, RESET
            );
        }

        println!();
        println!(
            "  {}Hint:{} Use 'reset' to clear state and start fresh",
            DIM, RESET
        );
        println!();
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // DOMAINS COMMAND - Per-Domain Calibration Display (Phase 1.2)
    // ═══════════════════════════════════════════════════════════════════════════

    fn cmd_domains(&self) {
        println!();
        println!(
            "{}{}╔═══════════════════════════════════════════════════════════════╗{}",
            BOLD, CYAN, RESET
        );
        println!(
            "{}{}║              PER-DOMAIN CALIBRATION                            ║{}",
            BOLD, CYAN, RESET
        );
        println!(
            "{}{}╚═══════════════════════════════════════════════════════════════╝{}",
            BOLD, CYAN, RESET
        );
        println!();

        let snapshot = self.model.persistence().current();

        if snapshot.calibration.is_empty() {
            println!("  {}No domain-specific calibration data yet.{}", DIM, RESET);
            println!(
                "  {}Use 'verify' to make externally-grounded predictions.{}",
                DIM, RESET
            );
            println!();
            return;
        }

        println!(
            "  {}{:<18} {:>8} {:>8} {:>8} {:>8}  {}{}",
            BOLD, "Domain", "Brier", "ECE", "Count", "Adj", "Status", RESET
        );
        println!("  ─────────────────────────────────────────────────────────────");

        for (domain, cal) in &snapshot.calibration {
            let brier_color = self.brier_color(cal.lifetime_brier);
            let status = if cal.is_overconfident {
                format!("{}⚠ Over{}", YELLOW, RESET)
            } else if cal.prediction_count < 10 {
                format!("{}📊 Low{}", DIM, RESET)
            } else if cal.lifetime_brier < 0.15 {
                format!("{}✓ Good{}", GREEN, RESET)
            } else {
                format!("{}○ Fair{}", DIM, RESET)
            };

            println!(
                "  {:<18} {}{:>8.4}{} {:>8.4} {:>8} {:>8.2}  {}",
                format!("{:?}", domain),
                brier_color,
                cal.lifetime_brier,
                RESET,
                cal.ece,
                cal.prediction_count,
                cal.confidence_adjustment,
                status
            );
        }

        println!();

        // Show recommendations
        let overconfident_domains: Vec<_> = snapshot
            .calibration
            .iter()
            .filter(|(_, cal)| cal.is_overconfident)
            .map(|(d, _)| format!("{:?}", d))
            .collect();

        if !overconfident_domains.is_empty() {
            println!("  {}{}⚠ Overconfident Domains:{}", YELLOW, BOLD, RESET);
            for domain in &overconfident_domains {
                println!("    - {} (consider lowering confidence)", domain);
            }
            println!();
        }

        println!();
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // TREND COMMAND - Calibration Trend Visualization (Phase 1.3)
    // ═══════════════════════════════════════════════════════════════════════════

    fn cmd_trend(&self, window: usize) {
        println!();
        println!(
            "{}{}╔═══════════════════════════════════════════════════════════════╗{}",
            BOLD, CYAN, RESET
        );
        println!(
            "{}{}║              CALIBRATION TREND                                 ║{}",
            BOLD, CYAN, RESET
        );
        println!(
            "{}{}╚═══════════════════════════════════════════════════════════════╝{}",
            BOLD, CYAN, RESET
        );
        println!();

        if self.brier_history.is_empty() {
            println!(
                "  {}No trend data yet. Make some predictions to see trends.{}",
                DIM, RESET
            );
            println!();
            return;
        }

        // Get last N entries
        let history: Vec<f64> = self
            .brier_history
            .iter()
            .rev()
            .take(window)
            .map(|e| e.brier)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();

        let max_brier = history.iter().cloned().fold(0.0f64, f64::max).max(0.4);
        let min_brier = history.iter().cloned().fold(1.0f64, f64::min).min(0.0);
        let range = (max_brier - min_brier).max(0.1);

        let chart_height = 10;
        let chart_width = history.len().min(50);

        println!("  Brier Score (last {} predictions):", history.len());
        println!();

        // ASCII chart
        for row in (0..chart_height).rev() {
            let _threshold = min_brier + (row as f64 / chart_height as f64) * range;
            let label = if row == chart_height - 1 {
                format!("{:.2}", max_brier)
            } else if row == 0 {
                format!("{:.2}", min_brier)
            } else {
                "    ".to_string()
            };

            print!("  {} │", label);

            for (i, &brier) in history.iter().enumerate() {
                if i >= chart_width {
                    break;
                }
                let normalized = (brier - min_brier) / range;
                let bar_height = (normalized * chart_height as f64) as usize;

                if bar_height > row {
                    let color = self.brier_color(brier);
                    print!("{}█{}", color, RESET);
                } else if bar_height == row {
                    let color = self.brier_color(brier);
                    print!("{}▄{}", color, RESET);
                } else {
                    print!(" ");
                }
            }
            println!();
        }

        print!("       └");
        for _ in 0..chart_width {
            print!("─");
        }
        println!("→");
        println!(
            "       Session start{}Now",
            " ".repeat(chart_width.saturating_sub(15))
        );
        println!();

        // Summary statistics
        let avg_brier: f64 = history.iter().sum::<f64>() / history.len() as f64;
        let current_brier = *history.last().unwrap_or(&0.0);
        let trend = if history.len() > 1 {
            let first_half_avg: f64 = history.iter().take(history.len() / 2).sum::<f64>()
                / (history.len() / 2).max(1) as f64;
            let second_half_avg: f64 = history.iter().skip(history.len() / 2).sum::<f64>()
                / (history.len() - history.len() / 2).max(1) as f64;
            second_half_avg - first_half_avg
        } else {
            0.0
        };

        println!("  {}Summary:{}", BOLD, RESET);
        println!(
            "    Average Brier: {}{:.4}{}",
            self.brier_color(avg_brier),
            avg_brier,
            RESET
        );
        println!(
            "    Current Brier: {}{:.4}{}",
            self.brier_color(current_brier),
            current_brier,
            RESET
        );
        println!(
            "    Trend: {}{}",
            if trend < 0.0 {
                GREEN
            } else if trend > 0.01 {
                RED
            } else {
                DIM
            },
            if trend < -0.01 {
                format!("↓ Improving ({:+.4})", trend)
            } else if trend > 0.01 {
                format!("↑ Degrading ({:+.4})", trend)
            } else {
                "→ Stable".to_string()
            }
        );
        println!("{}", RESET);
        println!();
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // BATCH COMMAND - Batch Verification from File (Phase 2.2)
    // ═══════════════════════════════════════════════════════════════════════════

    fn cmd_batch(&mut self, file_path: &str) -> io::Result<()> {
        println!();
        println!(
            "{}{}╔═══════════════════════════════════════════════════════════════╗{}",
            BOLD, MAGENTA, RESET
        );
        println!(
            "{}{}║              BATCH VERIFICATION                                ║{}",
            BOLD, MAGENTA, RESET
        );
        println!(
            "{}{}╚═══════════════════════════════════════════════════════════════╝{}",
            BOLD, MAGENTA, RESET
        );
        println!();

        let file = std::fs::File::open(file_path)?;
        let reader = io::BufReader::new(file);

        let mut total = 0;
        let mut correct = 0;
        let mut total_brier = 0.0;
        let before_brier = self
            .model
            .persistence()
            .current()
            .global_stats
            .lifetime_brier;

        println!(
            "  {}Running predictions from {}...{}",
            DIM, file_path, RESET
        );
        println!();

        for (line_num, line) in reader.lines().enumerate() {
            let line = line?;
            let line = line.trim();

            // Skip empty lines and comments
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            // Parse: <check_type> <target> <confidence>
            let parts: Vec<&str> = line.splitn(3, char::is_whitespace).collect();
            if parts.len() < 2 {
                println!(
                    "  {}Line {}: Invalid format (skipping){}",
                    YELLOW,
                    line_num + 1,
                    RESET
                );
                continue;
            }

            let check_type = parts[0];
            let target = parts[1];
            let confidence: f64 = parts.get(2).and_then(|s| s.parse().ok()).unwrap_or(0.8);

            // Run verification silently and capture result
            let result = self.run_batch_verify(check_type, target, confidence);

            match result {
                Ok((success, brier_component)) => {
                    total += 1;
                    if success {
                        correct += 1;
                        print!("  {}✓{} ", GREEN, RESET);
                    } else {
                        print!("  {}✗{} ", RED, RESET);
                    }
                    total_brier += brier_component;
                    println!(
                        "{} {}: {} (Brier: {:.4}{})",
                        check_type,
                        target,
                        if success {
                            format!("{}OK{}", GREEN, RESET)
                        } else {
                            format!("{}FAIL{}", RED, RESET)
                        },
                        brier_component,
                        if brier_component > 0.5 {
                            format!(" {}⚠ OVERCONFIDENT{}", YELLOW, RESET)
                        } else {
                            "".to_string()
                        }
                    );
                }
                Err(e) => {
                    println!(
                        "  {}⚠ {} {}: Error - {}{}",
                        YELLOW, check_type, target, e, RESET
                    );
                }
            }
        }

        println!();
        println!(
            "  {}╭─────────────────────────────────────────────────────────╮{}",
            DIM, RESET
        );
        println!(
            "  {}│{} {}BATCH SUMMARY{}                                          {}│{}",
            DIM, RESET, BOLD, RESET, DIM, RESET
        );
        println!(
            "  {}├─────────────────────────────────────────────────────────┤{}",
            DIM, RESET
        );
        println!(
            "  {}│{}  Total: {} | Correct: {} | Accuracy: {:.1}%{}│{}",
            DIM,
            RESET,
            total,
            correct,
            if total > 0 {
                correct as f64 / total as f64 * 100.0
            } else {
                0.0
            },
            DIM,
            RESET
        );
        println!(
            "  {}│{}  Batch Brier: {:.4}{}│{}",
            DIM,
            RESET,
            if total > 0 {
                total_brier / total as f64
            } else {
                0.0
            },
            DIM,
            RESET
        );

        let after_brier = self
            .model
            .persistence()
            .current()
            .global_stats
            .lifetime_brier;
        let delta = after_brier - before_brier;
        println!(
            "  {}│{}  Global Brier: {:.4} → {:.4} ({}{:+.4}{}){} {}│{}",
            DIM,
            RESET,
            before_brier,
            after_brier,
            if delta > 0.0 { RED } else { GREEN },
            delta,
            RESET,
            DIM,
            DIM,
            RESET
        );
        println!(
            "  {}╰─────────────────────────────────────────────────────────╯{}",
            DIM, RESET
        );
        println!();

        // Save
        self.model.persistence_mut().force_save()?;
        println!("  {}{}✓ Batch results persisted{}", GREEN, DIM, RESET);
        println!();

        Ok(())
    }

    /// Run a single batch verification without printing (returns success and brier)
    fn run_batch_verify(
        &mut self,
        check_type: &str,
        target: &str,
        confidence: f64,
    ) -> io::Result<(bool, f64)> {
        let actual_outcome = match check_type {
            "file" | "path" => Path::new(target).exists(),
            "cmd" | "command" => {
                let parts: Vec<&str> = target.split_whitespace().collect();
                if parts.is_empty() {
                    false
                } else {
                    Command::new(parts[0])
                        .args(&parts[1..])
                        .output()
                        .map(|o| o.status.success())
                        .unwrap_or(false)
                }
            }
            "pkg" | "package" => Command::new("nix-env")
                .args(["-q", target])
                .output()
                .map(|o| o.status.success() && String::from_utf8_lossy(&o.stdout).contains(target))
                .unwrap_or(false),
            "service" | "svc" => Command::new("systemctl")
                .args(["is-active", target])
                .output()
                .map(|o| String::from_utf8_lossy(&o.stdout).trim() == "active")
                .unwrap_or(false),
            "url" | "http" => Command::new("curl")
                .args([
                    "-s",
                    "-o",
                    "/dev/null",
                    "-w",
                    "%{http_code}",
                    "--connect-timeout",
                    "5",
                    target,
                ])
                .output()
                .map(|o| {
                    let code: u32 = String::from_utf8_lossy(&o.stdout)
                        .trim()
                        .parse()
                        .unwrap_or(0);
                    code >= 200 && code < 400
                })
                .unwrap_or(false),
            "port" | "tcp" => TcpStream::connect_timeout(
                &target
                    .parse()
                    .unwrap_or_else(|_| "127.0.0.1:80".parse().unwrap()),
                Duration::from_secs(3),
            )
            .is_ok(),
            "dns" => Command::new("host")
                .arg(target)
                .output()
                .map(|o| o.status.success())
                .unwrap_or(false),
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "Unknown check type",
                ));
            }
        };

        // Calculate Brier component
        let actual_value = if actual_outcome { 1.0 } else { 0.0 };
        let brier_component = (confidence - actual_value).powi(2);

        // Update persistence
        {
            let snapshot = self.model.persistence_mut().current_mut();
            snapshot.global_stats.total_predictions += 1;
            if actual_outcome {
                snapshot.global_stats.correct_predictions += 1;
            }
            snapshot.global_stats.brier_sum += brier_component;
            snapshot.global_stats.lifetime_brier =
                snapshot.global_stats.brier_sum / snapshot.global_stats.total_predictions as f64;

            let alpha = 0.2;
            snapshot.global_stats.rolling_brier =
                alpha * brier_component + (1.0 - alpha) * snapshot.global_stats.rolling_brier;

            snapshot.loop_state.predictions_made += 1;
            snapshot.loop_state.predictions_resolved += 1;
            snapshot.loop_state.loop_iterations += 1;
            snapshot.lifetime_iterations += 1;
            snapshot.global_stats.is_well_calibrated = snapshot.global_stats.lifetime_brier < 0.20;
        }

        // Record in history
        let brier = self
            .model
            .persistence()
            .current()
            .global_stats
            .lifetime_brier;
        self.record_brier_history(brier);

        Ok((actual_outcome, brier_component))
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // SUGGEST COMMAND - Confidence Recommendations (Phase 2.1)
    // ═══════════════════════════════════════════════════════════════════════════

    fn suggest_confidence(&self, check_type: &str) -> Option<(f64, f64, String)> {
        let snapshot = self.model.persistence().current();

        // Map check type to domain for historical lookup
        let relevant_domain = match check_type {
            "file" | "path" => PredictionDomain::SystemState,
            "cmd" | "command" | "test" => PredictionDomain::CodeExecution,
            "pkg" | "package" | "service" => PredictionDomain::SystemState,
            "url" | "http" | "port" | "dns" => PredictionDomain::ToolUse,
            _ => return None,
        };

        if let Some(cal) = snapshot.calibration.get(&relevant_domain) {
            if cal.prediction_count >= 5 {
                let accuracy = cal.correct_count as f64 / cal.prediction_count as f64;
                let suggested_low = (accuracy * 0.9).max(0.1);
                let suggested_high = (accuracy * 1.1).min(0.95);
                let reason = if cal.is_overconfident {
                    format!(
                        "You've been overconfident on {:?} checks ({:.0}% accuracy vs higher confidence)",
                        relevant_domain,
                        accuracy * 100.0
                    )
                } else {
                    format!(
                        "Based on your {:?} history ({:.0}% accuracy)",
                        relevant_domain,
                        accuracy * 100.0
                    )
                };
                return Some((suggested_low, suggested_high, reason));
            }
        }

        None
    }

    fn cmd_suggest(&self, check_type: &str, proposed_confidence: f64) {
        if let Some((low, high, reason)) = self.suggest_confidence(check_type) {
            if proposed_confidence > high + 0.1 || proposed_confidence < low - 0.1 {
                println!();
                println!("  {}{}⚠ Historical Warning:{}", YELLOW, BOLD, RESET);
                println!("    {}", reason);
                println!(
                    "    Suggested confidence range: {}{:.0}%-{:.0}%{}",
                    GREEN,
                    low * 100.0,
                    high * 100.0,
                    RESET
                );
                if proposed_confidence > high + 0.1 {
                    println!(
                        "    {}You may be overconfident at {:.0}%.{}",
                        YELLOW,
                        proposed_confidence * 100.0,
                        RESET
                    );
                }
                println!();
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // EXPORT COMMAND - State Export (Phase 3.1)
    // ═══════════════════════════════════════════════════════════════════════════

    fn cmd_export(&self, output_path: &str, format: &str) -> io::Result<()> {
        println!();
        println!(
            "{}{}╔═══════════════════════════════════════════════════════════════╗{}",
            BOLD, GREEN, RESET
        );
        println!(
            "{}{}║              EXPORTING EPISTEMIC STATE                         ║{}",
            BOLD, GREEN, RESET
        );
        println!(
            "{}{}╚═══════════════════════════════════════════════════════════════╝{}",
            BOLD, GREEN, RESET
        );
        println!();

        let snapshot = self.model.persistence().current();

        match format {
            "json" => {
                let json = serde_json::to_string_pretty(snapshot)
                    .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
                std::fs::write(output_path, json)?;
            }
            "csv" => {
                let mut csv_content = String::new();
                csv_content.push_str("domain,brier,ece,count,accuracy,adjustment,overconfident\n");
                for (domain, cal) in &snapshot.calibration {
                    let accuracy = if cal.prediction_count > 0 {
                        cal.correct_count as f64 / cal.prediction_count as f64
                    } else {
                        0.0
                    };
                    csv_content.push_str(&format!(
                        "{:?},{:.4},{:.4},{},{:.4},{:.2},{}\n",
                        domain,
                        cal.lifetime_brier,
                        cal.ece,
                        cal.prediction_count,
                        accuracy,
                        cal.confidence_adjustment,
                        cal.is_overconfident
                    ));
                }
                std::fs::write(output_path, csv_content)?;
            }
            "md" | "markdown" => {
                let mut md = String::new();
                md.push_str("# MAGI Epistemic State Export\n\n");
                md.push_str(&format!("**Exported**: {}\n", snapshot.saved_at_iso));
                md.push_str(&format!("**Session**: #{}\n", snapshot.session_count));
                md.push_str(&format!(
                    "**Lifetime Iterations**: {}\n\n",
                    snapshot.lifetime_iterations
                ));

                md.push_str("## Global Calibration\n\n");
                md.push_str(&format!(
                    "- **Brier Score**: {:.4}\n",
                    snapshot.global_stats.lifetime_brier
                ));
                md.push_str(&format!("- **ECE**: {:.4}\n", snapshot.global_stats.ece));
                md.push_str(&format!(
                    "- **Total Predictions**: {}\n",
                    snapshot.global_stats.total_predictions
                ));
                md.push_str(&format!(
                    "- **Well Calibrated**: {}\n\n",
                    snapshot.global_stats.is_well_calibrated
                ));

                md.push_str("## Per-Domain Calibration\n\n");
                md.push_str("| Domain | Brier | ECE | Count | Adjustment |\n");
                md.push_str("|--------|-------|-----|-------|------------|\n");
                for (domain, cal) in &snapshot.calibration {
                    md.push_str(&format!(
                        "| {:?} | {:.4} | {:.4} | {} | {:.2} |\n",
                        domain,
                        cal.lifetime_brier,
                        cal.ece,
                        cal.prediction_count,
                        cal.confidence_adjustment
                    ));
                }
                md.push_str("\n");

                std::fs::write(output_path, md)?;
            }
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("Unknown format: {}. Use json, csv, or md", format),
                ));
            }
        }

        println!("  {}Exported to {}:{}", CYAN, output_path, RESET);
        println!("    - {} domain calibrations", snapshot.calibration.len());
        println!(
            "    - {} lifetime predictions",
            snapshot.global_stats.total_predictions
        );
        println!(
            "    - {} causal attributions",
            snapshot.attribution_history.len()
        );
        println!("    - Gate configuration");
        println!();
        println!("  {}{}✓ Export complete{}", GREEN, DIM, RESET);
        println!();

        Ok(())
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // IMPORT COMMAND - State Import (Phase 3.2)
    // ═══════════════════════════════════════════════════════════════════════════

    fn cmd_import(&mut self, input_path: &str, merge: bool) -> io::Result<()> {
        println!();
        println!(
            "{}{}╔═══════════════════════════════════════════════════════════════╗{}",
            BOLD, BLUE, RESET
        );
        println!(
            "{}{}║              IMPORTING EPISTEMIC STATE                         ║{}",
            BOLD, BLUE, RESET
        );
        println!(
            "{}{}╚═══════════════════════════════════════════════════════════════╝{}",
            BOLD, BLUE, RESET
        );
        println!();

        let content = std::fs::read_to_string(input_path)?;
        let imported: symthaea::consciousness::recursive_improvement::persistence::MagiStateSnapshot =
            serde_json::from_str(&content)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        println!("  {}Loading from {}...{}", DIM, input_path, RESET);
        println!("    Sessions: {}", imported.session_count);
        println!(
            "    Predictions: {}",
            imported.global_stats.total_predictions
        );
        println!("    Brier: {:.4}", imported.global_stats.lifetime_brier);
        println!();

        if merge {
            println!("  {}Merge Strategy:{}", CYAN, RESET);
            let current = self.model.persistence().current();

            // Merge logic: use better calibration, combine counts
            for (domain, imported_cal) in &imported.calibration {
                if let Some(current_cal) = current.calibration.get(domain) {
                    // Keep whichever has better (lower) Brier score, weighted by sample size
                    let imported_weight = imported_cal.prediction_count as f64;
                    let current_weight = current_cal.prediction_count as f64;

                    if imported_weight > current_weight * 2.0 {
                        println!(
                            "    {:?}: Using imported (more data: {} vs {})",
                            domain, imported_cal.prediction_count, current_cal.prediction_count
                        );
                    } else if imported_cal.lifetime_brier < current_cal.lifetime_brier {
                        println!(
                            "    {:?}: Using imported (better calibration: {:.4} vs {:.4})",
                            domain, imported_cal.lifetime_brier, current_cal.lifetime_brier
                        );
                    } else {
                        println!(
                            "    {:?}: Keeping local (better: {:.4} vs {:.4})",
                            domain, current_cal.lifetime_brier, imported_cal.lifetime_brier
                        );
                    }
                } else {
                    println!("    {:?}: Adding from import (new domain)", domain);
                }
            }

            // Apply merge
            {
                let snapshot = self.model.persistence_mut().current_mut();
                for (domain, imported_cal) in imported.calibration {
                    let should_import = if let Some(current_cal) = snapshot.calibration.get(&domain)
                    {
                        imported_cal.prediction_count > current_cal.prediction_count * 2
                            || imported_cal.lifetime_brier < current_cal.lifetime_brier
                    } else {
                        true
                    };

                    if should_import {
                        snapshot.calibration.insert(domain, imported_cal);
                    }
                }
            }
        } else {
            // Full replace
            println!("  {}Full replacement mode{}", YELLOW, RESET);
            {
                let snapshot = self.model.persistence_mut().current_mut();
                snapshot.calibration = imported.calibration;
                snapshot.global_stats = imported.global_stats;
                snapshot.attribution_history = imported.attribution_history;
            }
        }

        self.model.persistence_mut().force_save()?;

        println!();
        println!("  {}{}✓ Import complete{}", GREEN, DIM, RESET);
        println!();

        Ok(())
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // ANALYTICS COMMAND - Session Analytics (Phase 3.3)
    // ═══════════════════════════════════════════════════════════════════════════

    fn cmd_analytics(&self) {
        println!();
        println!(
            "{}{}╔═══════════════════════════════════════════════════════════════╗{}",
            BOLD, CYAN, RESET
        );
        println!(
            "{}{}║              SESSION ANALYTICS                                 ║{}",
            BOLD, CYAN, RESET
        );
        println!(
            "{}{}╚═══════════════════════════════════════════════════════════════╝{}",
            BOLD, CYAN, RESET
        );
        println!();

        let snapshot = self.model.persistence().current();

        println!(
            "  {}Sessions:{} {} (lifetime)",
            CYAN, RESET, snapshot.session_count
        );
        println!(
            "  {}Total Predictions:{} {}",
            CYAN, RESET, snapshot.global_stats.total_predictions
        );
        println!();

        // Current session analysis
        let brier = snapshot.global_stats.lifetime_brier;
        let accuracy = if snapshot.global_stats.total_predictions > 0 {
            snapshot.global_stats.correct_predictions as f64
                / snapshot.global_stats.total_predictions as f64
        } else {
            0.0
        };

        println!(
            "  {}╭─────────────────────────────────────────────────────────╮{}",
            DIM, RESET
        );
        println!(
            "  {}│{} {}CURRENT STATE{}                                          {}│{}",
            DIM, RESET, BOLD, RESET, DIM, RESET
        );
        println!(
            "  {}├─────────────────────────────────────────────────────────┤{}",
            DIM, RESET
        );
        println!(
            "  {}│{}  Lifetime Brier:  {}{:.4}{}{}│{}",
            DIM,
            RESET,
            self.brier_color(brier),
            brier,
            RESET,
            DIM,
            RESET
        );
        println!(
            "  {}│{}  Rolling Brier:   {}{:.4}{}{}│{}",
            DIM,
            RESET,
            self.brier_color(snapshot.global_stats.rolling_brier),
            snapshot.global_stats.rolling_brier,
            RESET,
            DIM,
            RESET
        );
        println!(
            "  {}│{}  ECE:             {:.4}{}│{}",
            DIM, RESET, snapshot.global_stats.ece, DIM, RESET
        );
        println!(
            "  {}│{}  Accuracy:        {:.1}%{}│{}",
            DIM,
            RESET,
            accuracy * 100.0,
            DIM,
            RESET
        );
        println!(
            "  {}│{}  Well Calibrated: {}{}│{}",
            DIM,
            RESET,
            if snapshot.global_stats.is_well_calibrated {
                format!("{}Yes{}", GREEN, RESET)
            } else {
                format!("{}No{}", RED, RESET)
            },
            DIM,
            RESET
        );
        println!(
            "  {}╰─────────────────────────────────────────────────────────╯{}",
            DIM, RESET
        );
        println!();

        // Domain breakdown
        if !snapshot.calibration.is_empty() {
            let best_domain = snapshot
                .calibration
                .iter()
                .filter(|(_, cal)| cal.prediction_count >= 5)
                .min_by(|a, b| a.1.lifetime_brier.partial_cmp(&b.1.lifetime_brier).unwrap());

            let worst_domain = snapshot
                .calibration
                .iter()
                .filter(|(_, cal)| cal.prediction_count >= 5)
                .max_by(|a, b| a.1.lifetime_brier.partial_cmp(&b.1.lifetime_brier).unwrap());

            println!("  {}Domain Analysis:{}", BOLD, RESET);
            if let Some((domain, cal)) = best_domain {
                println!(
                    "    {}Best:{} {:?} (Brier: {:.4}, {} predictions)",
                    GREEN, RESET, domain, cal.lifetime_brier, cal.prediction_count
                );
            }
            if let Some((domain, cal)) = worst_domain {
                println!(
                    "    {}Worst:{} {:?} (Brier: {:.4}, {} predictions)",
                    RED, RESET, domain, cal.lifetime_brier, cal.prediction_count
                );
            }
            println!();
        }

        // Improvement suggestions
        println!("  {}Recommendations:{}", BOLD, RESET);

        if snapshot.global_stats.total_predictions < 50 {
            println!(
                "    - Make more predictions to improve calibration (have {}/50)",
                snapshot.global_stats.total_predictions
            );
        }

        if brier > 0.25 {
            println!(
                "    - {}Calibration is poor (Brier > 0.25). Consider lowering confidence levels.{}",
                YELLOW, RESET
            );
        }

        let overconfident_count = snapshot
            .calibration
            .values()
            .filter(|cal| cal.is_overconfident && cal.prediction_count >= 5)
            .count();
        if overconfident_count > 0 {
            println!(
                "    - {}You have {} overconfident domain(s). Use 'domains' to see details.{}",
                YELLOW, overconfident_count, RESET
            );
        }

        if snapshot.global_stats.is_well_calibrated && snapshot.global_stats.total_predictions >= 50
        {
            println!(
                "    - {}✓ Good calibration! You've earned autonomous execution rights.{}",
                GREEN, RESET
            );
        }

        println!();
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // GATE COMMAND - Constraint Gate Visibility (Phase 4.2)
    // ═══════════════════════════════════════════════════════════════════════════

    fn cmd_gate(&self) {
        println!();
        println!(
            "{}{}╔═══════════════════════════════════════════════════════════════╗{}",
            BOLD, YELLOW, RESET
        );
        println!(
            "{}{}║              CONSTRAINT GATE STATUS                            ║{}",
            BOLD, YELLOW, RESET
        );
        println!(
            "{}{}╚═══════════════════════════════════════════════════════════════╝{}",
            BOLD, YELLOW, RESET
        );
        println!();

        let snapshot = self.model.persistence().current();
        let gate_config = &snapshot.gate_config;

        // Determine current mode based on calibration
        let brier = snapshot.global_stats.lifetime_brier;
        let total_preds = snapshot.global_stats.total_predictions;
        let is_well_calibrated = snapshot.global_stats.is_well_calibrated;

        let current_mode = if gate_config.force_supervised_mode {
            "SUPERVISED (Forced)"
        } else if !is_well_calibrated {
            "DRY RUN (Poor Calibration)"
        } else if total_preds < gate_config.min_predictions_for_autonomy {
            "DRY RUN (Insufficient History)"
        } else {
            "AUTONOMOUS"
        };

        let mode_color = if current_mode.contains("AUTONOMOUS") {
            GREEN
        } else if current_mode.contains("DRY") {
            YELLOW
        } else {
            RED
        };

        println!(
            "  {}Current Mode:{} {}{}{}",
            CYAN, RESET, mode_color, current_mode, RESET
        );
        println!();

        println!(
            "  {}╭─────────────────────────────────────────────────────────╮{}",
            DIM, RESET
        );
        println!(
            "  {}│{} {}GATE FACTORS{}                                           {}│{}",
            DIM, RESET, BOLD, RESET, DIM, RESET
        );
        println!(
            "  {}├─────────────────────────────────────────────────────────┤{}",
            DIM, RESET
        );

        // Calibration quality
        let cal_status = if brier < 0.15 {
            format!("{}Good{} (Brier < 0.15)", GREEN, RESET)
        } else if brier < 0.20 {
            format!("{}Fair{} (Brier < 0.20)", YELLOW, RESET)
        } else {
            format!("{}Poor{} (Brier >= 0.20)", RED, RESET)
        };
        println!(
            "  {}│{}  Calibration Quality: {}{}│{}",
            DIM, RESET, cal_status, DIM, RESET
        );

        // Minimum predictions
        let pred_status = if total_preds >= gate_config.min_predictions_for_autonomy {
            format!(
                "{}✓{} ({} >= {})",
                GREEN, RESET, total_preds, gate_config.min_predictions_for_autonomy
            )
        } else {
            format!(
                "{}✗{} ({} < {})",
                RED, RESET, total_preds, gate_config.min_predictions_for_autonomy
            )
        };
        println!(
            "  {}│{}  Min Predictions Met: {}{}│{}",
            DIM, RESET, pred_status, DIM, RESET
        );

        // Accuracy threshold
        let accuracy = if total_preds > 0 {
            snapshot.global_stats.correct_predictions as f64 / total_preds as f64
        } else {
            0.0
        };
        let acc_status = if accuracy >= gate_config.min_accuracy_for_autonomy {
            format!(
                "{}✓{} ({:.1}% >= {:.1}%)",
                GREEN,
                RESET,
                accuracy * 100.0,
                gate_config.min_accuracy_for_autonomy * 100.0
            )
        } else {
            format!(
                "{}✗{} ({:.1}% < {:.1}%)",
                RED,
                RESET,
                accuracy * 100.0,
                gate_config.min_accuracy_for_autonomy * 100.0
            )
        };
        println!(
            "  {}│{}  Accuracy Threshold:  {}{}│{}",
            DIM, RESET, acc_status, DIM, RESET
        );

        println!(
            "  {}╰─────────────────────────────────────────────────────────╯{}",
            DIM, RESET
        );
        println!();

        // What would require supervision
        println!("  {}Would Require Supervision For:{}", BOLD, RESET);
        println!("    - Destructive operations (always)");
        println!("    - Critical/irreversible actions (always)");
        if !is_well_calibrated {
            println!(
                "    - {}All actions (until calibration improves){}",
                YELLOW, RESET
            );
        }

        let overconfident: Vec<_> = snapshot
            .calibration
            .iter()
            .filter(|(_, cal)| cal.is_overconfident && cal.prediction_count >= 5)
            .map(|(d, _)| format!("{:?}", d))
            .collect();
        if !overconfident.is_empty() {
            println!("    - Domains: {}", overconfident.join(", "));
        }

        println!();
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // METRICS COMMAND - Consciousness Metric Transparency
    // ═══════════════════════════════════════════════════════════════════════════

    fn cmd_metrics(&self) {
        println!();
        println!(
            "{}{}╔═══════════════════════════════════════════════════════════════╗{}",
            BOLD, MAGENTA, RESET
        );
        println!(
            "{}{}║              CONSCIOUSNESS METRICS AUDIT                        ║{}",
            BOLD, MAGENTA, RESET
        );
        println!(
            "{}{}╚═══════════════════════════════════════════════════════════════╝{}",
            BOLD, MAGENTA, RESET
        );
        println!();

        println!("  {}CRITICAL: λ₂ ≠ Φ{}", BOLD, RESET);
        println!(
            "  {}This system computes multiple metrics. Know what you're measuring!{}",
            DIM, RESET
        );
        println!();

        println!(
            "  {}╭─────────────────────────────────────────────────────────────────╮{}",
            DIM, RESET
        );
        println!(
            "  {}│{} {}TIERED Φ APPROXIMATION SYSTEM{}                                 {}│{}",
            DIM, RESET, BOLD, RESET, DIM, RESET
        );
        println!(
            "  {}├─────────────────────────────────────────────────────────────────┤{}",
            DIM, RESET
        );

        // Mock tier
        println!(
            "  {}│{} {}Mock{}      O(1)   Deterministic test values                    {}│{}",
            DIM, RESET, CYAN, RESET, DIM, RESET
        );
        println!(
            "  {}│{}           IIT-Aligned: {}N/A{} (testing only)                      {}│{}",
            DIM, RESET, DIM, RESET, DIM, RESET
        );
        println!(
            "  {}│{}                                                                   {}│{}",
            DIM, RESET, DIM, RESET
        );

        // Heuristic tier
        println!(
            "  {}│{} {}Heuristic{} O(n)   Fast approximation via similarity           {}│{}",
            DIM, RESET, CYAN, RESET, DIM, RESET
        );
        println!(
            "  {}│{}           IIT-Aligned: {}❓ Unvalidated{}                           {}│{}",
            DIM, RESET, YELLOW, RESET, DIM, RESET
        );
        println!(
            "  {}│{}                                                                   {}│{}",
            DIM, RESET, DIM, RESET
        );

        // Spectral tier - THE CRITICAL ONE
        println!(
            "  {}│{} {}Spectral{}  O(n²)  {}⚠ MEASURES λ₂, NOT Φ!{}                       {}│{}",
            DIM, RESET, CYAN, RESET, RED, RESET, DIM, RESET
        );
        println!(
            "  {}│{}           IIT-Aligned: {}❌ NO{} (r ≈ -0.14 vs Exact)             {}│{}",
            DIM, RESET, RED, RESET, DIM, RESET
        );
        println!(
            "  {}│{}           Measures: Graph mixing time (algebraic connectivity)   {}│{}",
            DIM, RESET, DIM, RESET
        );
        println!(
            "  {}│{}           Favors: Uniform k-regular graphs (ring, torus)         {}│{}",
            DIM, RESET, DIM, RESET
        );
        println!(
            "  {}│{}                                                                   {}│{}",
            DIM, RESET, DIM, RESET
        );

        // Exact tier
        println!(
            "  {}│{} {}Exact{}     O(2ⁿ)  {}True IIT Φ (MIP calculation){}               {}│{}",
            DIM, RESET, CYAN, RESET, GREEN, RESET, DIM, RESET
        );
        println!(
            "  {}│{}           IIT-Aligned: {}✅ YES{} (validated against PyPhi)        {}│{}",
            DIM, RESET, GREEN, RESET, DIM, RESET
        );
        println!(
            "  {}│{}           Limit: n ≤ 12 only (exponential complexity)            {}│{}",
            DIM, RESET, DIM, RESET
        );

        println!(
            "  {}╰─────────────────────────────────────────────────────────────────╯{}",
            DIM, RESET
        );
        println!();

        println!(
            "  {}╭─────────────────────────────────────────────────────────────────╮{}",
            DIM, RESET
        );
        println!(
            "  {}│{} {}CORRELATION EVIDENCE{}                                           {}│{}",
            DIM, RESET, BOLD, RESET, DIM, RESET
        );
        println!(
            "  {}├─────────────────────────────────────────────────────────────────┤{}",
            DIM, RESET
        );
        println!(
            "  {}│{}  Dual-metric comparison across 19 topologies:                    {}│{}",
            DIM, RESET, DIM, RESET
        );
        println!(
            "  {}│{}                                                                   {}│{}",
            DIM, RESET, DIM, RESET
        );
        println!(
            "  {}│{}  Pearson (r):    {}-0.14{}  (anti-correlated with Exact tier)      {}│{}",
            DIM, RESET, RED, RESET, DIM, RESET
        );
        println!(
            "  {}│{}  Spearman (ρ):   {}-0.59{}  (anti-correlated rank ordering)        {}│{}",
            DIM, RESET, RED, RESET, DIM, RESET
        );
        println!(
            "  {}│{}  Avg Rank Diff:  {}6.42{}   (rankings completely divergent)        {}│{}",
            DIM, RESET, RED, RESET, DIM, RESET
        );
        println!(
            "  {}│{}                                                                   {}│{}",
            DIM, RESET, DIM, RESET
        );
        println!(
            "  {}│{}  {}Conclusion:{} λ₂ and Φ measure different properties entirely    {}│{}",
            DIM, RESET, BOLD, RESET, DIM, RESET
        );
        println!(
            "  {}╰─────────────────────────────────────────────────────────────────╯{}",
            DIM, RESET
        );
        println!();

        println!("  {}GUIDANCE:{}", BOLD, RESET);
        println!(
            "    {}✓{} For consciousness research: Use {}Exact{} tier (n ≤ 12)",
            GREEN, RESET, GREEN, RESET
        );
        println!(
            "    {}✓{} For graph analysis: Spectral tier is valid (just not for Φ)",
            GREEN, RESET
        );
        println!(
            "    {}✗{} Never use λ₂ (Spectral) for IIT consciousness claims",
            RED, RESET
        );
        println!();

        println!(
            "  {}See: docs/METRIC_CLARIFICATION.md for full analysis{}",
            DIM, RESET
        );
        println!();
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // ATTRIBUTIONS COMMAND - Causal Attribution Exposure (Phase 4.1)
    // ═══════════════════════════════════════════════════════════════════════════

    fn cmd_attributions(&self) {
        println!();
        println!(
            "{}{}╔═══════════════════════════════════════════════════════════════╗{}",
            BOLD, RED, RESET
        );
        println!(
            "{}{}║              FAILURE ATTRIBUTIONS                              ║{}",
            BOLD, RED, RESET
        );
        println!(
            "{}{}╚═══════════════════════════════════════════════════════════════╝{}",
            BOLD, RED, RESET
        );
        println!();

        let snapshot = self.model.persistence().current();

        if snapshot.attribution_history.is_empty() {
            println!("  {}No failure attributions recorded yet.{}", DIM, RESET);
            println!(
                "  {}Attributions are generated when predictions fail.{}",
                DIM, RESET
            );
            println!();
            return;
        }

        println!("  {}Recent Failure Attributions:{}", BOLD, RESET);
        println!();

        for (i, attr) in snapshot
            .attribution_history
            .iter()
            .rev()
            .take(10)
            .enumerate()
        {
            println!(
                "  {}#{}{}  Prediction: {}",
                CYAN,
                i + 1,
                RESET,
                attr.prediction_id
            );
            println!("       Failure Mode: {}{}{}", RED, attr.failure_mode, RESET);

            if !attr.missing_information.is_empty() {
                println!("       Missing Info: {:?}", attr.missing_information);
            }

            if !attr.responsible_domains.is_empty() {
                println!("       Domains: {:?}", attr.responsible_domains);
            }

            if let Some(ref recur) = attr.recurrence_prediction {
                println!("       Recurrence: {}", recur);
            }

            println!("       Confidence: {:.2}", attr.confidence);
            println!();
        }

        // Patterns analysis
        let failure_modes: std::collections::HashMap<&str, usize> = snapshot
            .attribution_history
            .iter()
            .map(|a| a.failure_mode.as_str())
            .fold(std::collections::HashMap::new(), |mut acc, mode| {
                *acc.entry(mode).or_insert(0) += 1;
                acc
            });

        if failure_modes.len() > 1 {
            println!("  {}Failure Pattern Analysis:{}", BOLD, RESET);
            let mut sorted: Vec<_> = failure_modes.into_iter().collect();
            sorted.sort_by(|a, b| b.1.cmp(&a.1));
            for (mode, count) in sorted.iter().take(5) {
                println!("    - {}: {} occurrences", mode, count);
            }
            println!();
        }

        println!();
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // BRIDGE COMMAND - Active Inference Bridge Stats (Phase 4.3)
    // ═══════════════════════════════════════════════════════════════════════════

    fn cmd_bridge(&self) {
        println!();
        println!(
            "{}{}╔═══════════════════════════════════════════════════════════════╗{}",
            BOLD, MAGENTA, RESET
        );
        println!(
            "{}{}║              ACTIVE INFERENCE BRIDGE                           ║{}",
            BOLD, MAGENTA, RESET
        );
        println!(
            "{}{}╚═══════════════════════════════════════════════════════════════╝{}",
            BOLD, MAGENTA, RESET
        );
        println!();

        let snapshot = self.model.persistence().current();

        println!(
            "  {}╭─────────────────────────────────────────────────────────╮{}",
            DIM, RESET
        );
        println!(
            "  {}│{} {}BRIDGE OVERVIEW{}                                        {}│{}",
            DIM, RESET, BOLD, RESET, DIM, RESET
        );
        println!(
            "  {}├─────────────────────────────────────────────────────────┤{}",
            DIM, RESET
        );
        println!(
            "  {}│{}  The Active Inference Bridge converts MAGI Loop        {}│{}",
            DIM, RESET, DIM, RESET
        );
        println!(
            "  {}│{}  calibration into control signals for consciousness:   {}│{}",
            DIM, RESET, DIM, RESET
        );
        println!(
            "  {}│{}                                                         {}│{}",
            DIM, RESET, DIM, RESET
        );
        println!(
            "  {}│{}  • {}Free Energy{}: Inverse of calibration (higher = worse) {}│{}",
            DIM, RESET, CYAN, RESET, DIM, RESET
        );
        println!(
            "  {}│{}  • {}Uncertainty{}: From ECE (calibration error)           {}│{}",
            DIM, RESET, CYAN, RESET, DIM, RESET
        );
        println!(
            "  {}│{}  • {}Confidence{}: From domain-adjusted calibration        {}│{}",
            DIM, RESET, CYAN, RESET, DIM, RESET
        );
        println!(
            "  {}│{}  • {}Surprise{}: Spike on prediction failures              {}│{}",
            DIM, RESET, CYAN, RESET, DIM, RESET
        );
        println!(
            "  {}│{}  • {}Coherence{}: Cross-domain consistency                 {}│{}",
            DIM, RESET, CYAN, RESET, DIM, RESET
        );
        println!(
            "  {}╰─────────────────────────────────────────────────────────╯{}",
            DIM, RESET
        );
        println!();

        println!(
            "  {}╭─────────────────────────────────────────────────────────╮{}",
            DIM, RESET
        );
        println!(
            "  {}│{} {}PAC (PHASE-AMPLITUDE COUPLING){}                         {}│{}",
            DIM, RESET, BOLD, RESET, DIM, RESET
        );
        println!(
            "  {}├─────────────────────────────────────────────────────────┤{}",
            DIM, RESET
        );
        println!(
            "  {}│{}  Tracks prediction-outcome coupling quality.           {}│{}",
            DIM, RESET, DIM, RESET
        );
        println!(
            "  {}│{}                                                         {}│{}",
            DIM, RESET, DIM, RESET
        );
        println!(
            "  {}│{}  Coupling Levels:                                       {}│{}",
            DIM, RESET, DIM, RESET
        );
        println!(
            "  {}│{}    • {}Strong{} (MI > 0.6): Predictions tightly coupled    {}│{}",
            DIM, RESET, GREEN, RESET, DIM, RESET
        );
        println!(
            "  {}│{}    • {}Moderate{} (0.3-0.6): Good tracking                 {}│{}",
            DIM, RESET, YELLOW, RESET, DIM, RESET
        );
        println!(
            "  {}│{}    • {}Weak{} (0.1-0.3): Loose coupling                    {}│{}",
            DIM, RESET, YELLOW, RESET, DIM, RESET
        );
        println!(
            "  {}│{}    • {}None{} (< 0.1): No meaningful coupling              {}│{}",
            DIM, RESET, RED, RESET, DIM, RESET
        );
        println!(
            "  {}╰─────────────────────────────────────────────────────────╯{}",
            DIM, RESET
        );
        println!();

        // Estimate coupling quality from calibration data
        let brier = snapshot.global_stats.lifetime_brier;
        let ece = snapshot.global_stats.ece;
        let total_preds = snapshot.global_stats.total_predictions;

        println!(
            "  {}Estimated Bridge State (from calibration):{}",
            BOLD, RESET
        );
        println!();

        // Estimate modulation index from Brier score (inverse relationship)
        let estimated_mi = if total_preds < 10 {
            None
        } else {
            Some(1.0 - brier.min(1.0))
        };

        let (coupling_str, coupling_color) = match estimated_mi {
            None => ("Insufficient Data".to_string(), DIM),
            Some(mi) if mi > 0.8 => (format!("Strong (MI ≈ {:.2})", mi), GREEN),
            Some(mi) if mi > 0.6 => (format!("Good (MI ≈ {:.2})", mi), GREEN),
            Some(mi) if mi > 0.4 => (format!("Moderate (MI ≈ {:.2})", mi), YELLOW),
            Some(mi) if mi > 0.2 => (format!("Weak (MI ≈ {:.2})", mi), YELLOW),
            Some(mi) => (format!("Poor (MI ≈ {:.2})", mi), RED),
        };

        println!(
            "    Coupling Quality: {}{}{}",
            coupling_color, coupling_str, RESET
        );

        // Signal estimates
        let free_energy = brier * 2.0; // Higher brier = higher free energy
        let uncertainty = ece.min(0.5) * 2.0; // Normalize ECE
        let confidence = if snapshot.global_stats.is_well_calibrated {
            0.8
        } else {
            0.4
        };

        println!();
        println!("    {}Estimated Signal Levels:{}", DIM, RESET);
        println!("      Free Energy:  {}", Self::signal_bar(free_energy, RED));
        println!(
            "      Uncertainty:  {}",
            Self::signal_bar(uncertainty, YELLOW)
        );
        println!(
            "      Confidence:   {}",
            Self::signal_bar(confidence, GREEN)
        );
        println!();

        println!(
            "  {}╭─────────────────────────────────────────────────────────╮{}",
            DIM, RESET
        );
        println!(
            "  {}│{} {}LIVE BRIDGE STATS{}                                       {}│{}",
            DIM, RESET, BOLD, RESET, DIM, RESET
        );
        println!(
            "  {}├─────────────────────────────────────────────────────────┤{}",
            DIM, RESET
        );
        println!(
            "  {}│{}  For real-time bridge statistics with live PAC          {}│{}",
            DIM, RESET, DIM, RESET
        );
        println!(
            "  {}│{}  tracking, use the {}monitor{} command:                      {}│{}",
            DIM, RESET, GREEN, RESET, DIM, RESET
        );
        println!(
            "  {}│{}                                                         {}│{}",
            DIM, RESET, DIM, RESET
        );
        println!(
            "  {}│{}    {}magi> monitor{}                                        {}│{}",
            DIM, RESET, CYAN, RESET, DIM, RESET
        );
        println!(
            "  {}│{}                                                         {}│{}",
            DIM, RESET, DIM, RESET
        );
        println!(
            "  {}│{}  The monitor runs the full MAGI Loop runtime with       {}│{}",
            DIM, RESET, DIM, RESET
        );
        println!(
            "  {}│{}  live signal generation and PAC tracking.               {}│{}",
            DIM, RESET, DIM, RESET
        );
        println!(
            "  {}╰─────────────────────────────────────────────────────────╯{}",
            DIM, RESET
        );
        println!();
    }

    /// Render a simple ASCII signal bar
    fn signal_bar(value: f64, color: &str) -> String {
        let clamped = value.clamp(0.0, 1.0);
        let filled = (clamped * 20.0).round() as usize;
        let empty = 20 - filled;
        format!(
            "{}{}{}{} {:.2}",
            color,
            "█".repeat(filled),
            RESET,
            "░".repeat(empty),
            clamped
        )
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // SCHEDULE COMMAND - Prediction Scheduling (Phase 2.3)
    // ═══════════════════════════════════════════════════════════════════════════

    fn cmd_schedule(&self, args: &[&str]) {
        println!();

        if args.len() < 2 {
            // Show scheduled predictions
            println!(
                "{}{}╔═══════════════════════════════════════════════════════════════╗{}",
                BOLD, CYAN, RESET
            );
            println!(
                "{}{}║              SCHEDULED PREDICTIONS                             ║{}",
                BOLD, CYAN, RESET
            );
            println!(
                "{}{}╚═══════════════════════════════════════════════════════════════╝{}",
                BOLD, CYAN, RESET
            );
            println!();

            println!("  {}Usage:{}", BOLD, RESET);
            println!("    schedule list              - Show all scheduled predictions");
            println!("    schedule <stmt> <conf> in <time>");
            println!("                               - Schedule a prediction");
            println!();
            println!("  {}Examples:{}", BOLD, RESET);
            println!("    schedule \"build completes\" 0.85 in 5m");
            println!("    schedule \"file /tmp/done exists\" 0.9 in 1h");
            println!("    schedule \"service started\" 0.8 in 30s");
            println!();
            println!("  {}Time formats:{}", DIM, RESET);
            println!("    30s, 5m, 1h, 2h30m");
            println!();

            println!(
                "  {}╭─────────────────────────────────────────────────────────╮{}",
                DIM, RESET
            );
            println!(
                "  {}│{} {}NOTE:{} Scheduled predictions are experimental.          {}│{}",
                DIM, RESET, YELLOW, RESET, DIM, RESET
            );
            println!(
                "  {}│{}                                                         {}│{}",
                DIM, RESET, DIM, RESET
            );
            println!(
                "  {}│{}  Currently, scheduled predictions resolve on the next   {}│{}",
                DIM, RESET, DIM, RESET
            );
            println!(
                "  {}│{}  CLI startup. For continuous monitoring, use the        {}│{}",
                DIM, RESET, DIM, RESET
            );
            println!(
                "  {}│{}  {}monitor{} command with the MAGI Loop runtime.            {}│{}",
                DIM, RESET, GREEN, RESET, DIM, RESET
            );
            println!(
                "  {}╰─────────────────────────────────────────────────────────╯{}",
                DIM, RESET
            );
            println!();
            return;
        }

        match args[1] {
            "list" | "ls" => {
                println!(
                    "{}{}╔═══════════════════════════════════════════════════════════════╗{}",
                    BOLD, CYAN, RESET
                );
                println!(
                    "{}{}║              SCHEDULED PREDICTIONS                             ║{}",
                    BOLD, CYAN, RESET
                );
                println!(
                    "{}{}╚═══════════════════════════════════════════════════════════════╝{}",
                    BOLD, CYAN, RESET
                );
                println!();

                // For now, scheduling is not persisted - show informational message
                println!("  {}No scheduled predictions in persistence.{}", DIM, RESET);
                println!();
                println!(
                    "  {}Scheduling requires the MAGI Loop Runtime to be active.{}",
                    DIM, RESET
                );
                println!(
                    "  {}Use the 'monitor' command for live prediction tracking.{}",
                    DIM, RESET
                );
                println!();
            }
            _ => {
                // Parse: schedule "<statement>" <confidence> in <time>
                // Find the "in" keyword
                let in_pos = args.iter().position(|&x| x == "in");

                if in_pos.is_none() || in_pos.unwrap() < 3 {
                    println!(
                        "  {}Error:{} Invalid format. Use: schedule \"statement\" confidence in time",
                        RED, RESET
                    );
                    println!(
                        "  {}Example:{} schedule \"build completes\" 0.85 in 5m",
                        DIM, RESET
                    );
                    return;
                }

                let in_idx = in_pos.unwrap();

                // Extract statement (everything from args[1] to confidence position)
                let conf_idx = in_idx - 1;
                let confidence: f64 = args[conf_idx].parse().unwrap_or(0.8);

                // Statement is everything between "schedule" and confidence
                let statement_parts: Vec<&str> = args[1..conf_idx].to_vec();
                let statement = statement_parts.join(" ").trim_matches('"').to_string();

                // Time is everything after "in"
                let time_parts: Vec<&str> = args[in_idx + 1..].to_vec();
                let time_str = time_parts.join("");

                // Parse time (simple implementation)
                let duration_secs = Self::parse_duration(&time_str);

                if duration_secs.is_none() {
                    println!(
                        "  {}Error:{} Could not parse duration: {}",
                        RED, RESET, time_str
                    );
                    println!("  {}Valid formats:{} 30s, 5m, 1h, 2h30m", DIM, RESET);
                    return;
                }

                let secs = duration_secs.unwrap();
                let _resolve_time =
                    std::time::SystemTime::now() + std::time::Duration::from_secs(secs);

                println!("  {}Prediction Scheduled (experimental):{}", GREEN, RESET);
                println!();
                println!("    Statement:  \"{}\"", statement);
                println!("    Confidence: {:.0}%", confidence * 100.0);
                println!("    Resolves:   in {} ({}s)", time_str, secs);
                println!();

                println!(
                    "  {}╭─────────────────────────────────────────────────────────╮{}",
                    DIM, RESET
                );
                println!(
                    "  {}│{} {}IMPLEMENTATION NOTE:{}                                   {}│{}",
                    DIM, RESET, YELLOW, RESET, DIM, RESET
                );
                println!(
                    "  {}│{}                                                         {}│{}",
                    DIM, RESET, DIM, RESET
                );
                println!(
                    "  {}│{}  Full scheduling requires background runtime.           {}│{}",
                    DIM, RESET, DIM, RESET
                );
                println!(
                    "  {}│{}  For now, use manual verification:                      {}│{}",
                    DIM, RESET, DIM, RESET
                );
                println!(
                    "  {}│{}                                                         {}│{}",
                    DIM, RESET, DIM, RESET
                );
                println!(
                    "  {}│{}    1. Run your operation                                {}│{}",
                    DIM, RESET, DIM, RESET
                );
                println!(
                    "  {}│{}    2. Wait for the expected time                        {}│{}",
                    DIM, RESET, DIM, RESET
                );
                println!(
                    "  {}│{}    3. Use 'verify' to check the result                  {}│{}",
                    DIM, RESET, DIM, RESET
                );
                println!(
                    "  {}│{}                                                         {}│{}",
                    DIM, RESET, DIM, RESET
                );
                println!(
                    "  {}│{}  Or use 'monitor' for real-time prediction tracking.    {}│{}",
                    DIM, RESET, DIM, RESET
                );
                println!(
                    "  {}╰─────────────────────────────────────────────────────────╯{}",
                    DIM, RESET
                );
                println!();
            }
        }
    }

    /// Parse a duration string like "30s", "5m", "1h", "2h30m"
    fn parse_duration(s: &str) -> Option<u64> {
        let s = s.trim().to_lowercase();
        let mut total_secs: u64 = 0;
        let mut current_num = String::new();

        for c in s.chars() {
            if c.is_ascii_digit() {
                current_num.push(c);
            } else {
                let num: u64 = current_num.parse().ok()?;
                current_num.clear();

                match c {
                    's' => total_secs += num,
                    'm' => total_secs += num * 60,
                    'h' => total_secs += num * 3600,
                    'd' => total_secs += num * 86400,
                    _ => return None,
                }
            }
        }

        if total_secs > 0 {
            Some(total_secs)
        } else {
            None
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // INTERACTIVE MODE
    // ═══════════════════════════════════════════════════════════════════════════

    fn interactive(&mut self) -> io::Result<()> {
        println!();
        println!(
            "{}{}╔══════════════════════════════════════════════════════════════╗{}",
            BOLD, BLUE, RESET
        );
        println!(
            "{}{}║              MAGI EPISTEMIC SHELL - INTERACTIVE               ║{}",
            BOLD, BLUE, RESET
        );
        println!(
            "{}{}╚══════════════════════════════════════════════════════════════╝{}",
            BOLD, BLUE, RESET
        );
        println!();
        println!(
            "  Commands: status, predict, resolve, verify, calibration, domains, trend, batch, schedule,"
        );
        println!(
            "            export, import, analytics, gate, bridge, metrics, attributions, history, monitor,"
        );
        println!("            where, drift, reset, help, quit");
        println!();

        // Show initial status
        self.cmd_status();

        let mut input = String::new();

        loop {
            // Prompt with mini-status
            let brier = self
                .model
                .persistence()
                .current()
                .global_stats
                .lifetime_brier;
            print!(
                "{}[B:{:.2}]{} {}magi>{} ",
                self.brier_color(brier),
                brier,
                RESET,
                CYAN,
                RESET
            );
            io::stdout().flush()?;

            input.clear();
            if io::stdin().read_line(&mut input)? == 0 {
                break; // EOF
            }

            let parts: Vec<&str> = input.trim().split_whitespace().collect();
            if parts.is_empty() {
                continue;
            }

            match parts[0] {
                "quit" | "exit" | "q" => {
                    println!("  {}Goodbye. Your wisdom is preserved.{}", DIM, RESET);
                    break;
                }
                "help" | "h" | "?" => {
                    self.print_help();
                }
                "status" | "s" => {
                    self.cmd_status();
                }
                "predict" | "p" => {
                    if parts.len() < 2 {
                        println!("  Usage: predict <statement> [-c confidence] [-d domain]");
                        continue;
                    }

                    // Parse arguments
                    let mut statement_parts = vec![];
                    let mut confidence = 0.8;
                    let mut domain = PredictionDomain::Factual;
                    let mut i = 1;

                    while i < parts.len() {
                        if parts[i] == "-c" && i + 1 < parts.len() {
                            confidence = parts[i + 1].parse().unwrap_or(0.8);
                            i += 2;
                        } else if parts[i] == "-d" && i + 1 < parts.len() {
                            domain = match parts[i + 1].to_lowercase().as_str() {
                                "code" => PredictionDomain::CodeExecution,
                                "tool" => PredictionDomain::ToolUse,
                                "user" => PredictionDomain::UserBehavior,
                                "system" => PredictionDomain::SystemState,
                                _ => PredictionDomain::Factual,
                            };
                            i += 2;
                        } else {
                            statement_parts.push(parts[i]);
                            i += 1;
                        }
                    }

                    let statement = statement_parts.join(" ");
                    if statement.is_empty() {
                        println!("  Error: Statement required");
                        continue;
                    }

                    self.cmd_predict(&statement, confidence, domain);
                }
                "resolve" | "r" => {
                    if parts.len() < 2 {
                        println!("  Usage: resolve success|failure");
                        continue;
                    }
                    match parts[1] {
                        "success" | "s" | "true" | "1" => {
                            self.cmd_resolve(true)?;
                        }
                        "failure" | "f" | "false" | "0" => {
                            self.cmd_resolve(false)?;
                        }
                        _ => {
                            println!("  Usage: resolve success|failure");
                        }
                    }
                }
                "calibration" | "cal" | "c" => {
                    self.cmd_calibration();
                }
                "history" | "hist" => {
                    self.cmd_history();
                }
                "drift" => {
                    // Red team tool: drift <count> <confidence> <success_rate>
                    let count = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(10);
                    let confidence = parts.get(2).and_then(|s| s.parse().ok()).unwrap_or(0.99);
                    let success_rate = parts.get(3).and_then(|s| s.parse().ok()).unwrap_or(0.2);
                    self.cmd_drift(count, confidence, success_rate)?;
                }
                "verify" | "v" => {
                    // Externally grounded verification
                    // verify file /path/to/file 0.9
                    // verify cmd "ls -la" 0.8
                    // verify test test_name 0.95
                    if parts.len() < 3 {
                        println!("  Usage: verify <type> <target> [confidence]");
                        println!("  Types: file, cmd, test, pkg, service, url, port, dns");
                        println!("  Examples:");
                        println!("    verify file /etc/passwd 0.99");
                        println!("    verify cmd \"ls -la\" 0.9");
                        println!("    verify url https://example.com 0.95");
                        println!("    verify port localhost:8080 0.9");
                        continue;
                    }
                    let check_type = parts[1];
                    let target = parts[2];
                    let confidence = parts.get(3).and_then(|s| s.parse().ok()).unwrap_or(0.8);

                    // Show confidence suggestions (Phase 2.1)
                    self.cmd_suggest(check_type, confidence);

                    self.cmd_verify(check_type, target, confidence)?;

                    // Record in trend history
                    let brier = self
                        .model
                        .persistence()
                        .current()
                        .global_stats
                        .lifetime_brier;
                    self.record_brier_history(brier);
                }
                "domains" | "dom" | "d" => {
                    self.cmd_domains();
                }
                "trend" | "tr" => {
                    let window = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(50);
                    self.cmd_trend(window);
                }
                "batch" | "b" => {
                    if parts.len() < 2 {
                        println!("  Usage: batch <file_path>");
                        println!("  File format (one per line): <type> <target> [confidence]");
                        println!("  Example file content:");
                        println!("    file /etc/passwd 0.99");
                        println!("    url https://example.com 0.95");
                        continue;
                    }
                    self.cmd_batch(parts[1])?;
                }
                "export" | "exp" => {
                    if parts.len() < 2 {
                        println!("  Usage: export <output_path> [format]");
                        println!("  Formats: json (default), csv, md");
                        continue;
                    }
                    let format = parts.get(2).unwrap_or(&"json");
                    self.cmd_export(parts[1], format)?;
                }
                "import" | "imp" => {
                    if parts.len() < 2 {
                        println!("  Usage: import <input_path> [--merge]");
                        continue;
                    }
                    let merge = parts.iter().any(|p| *p == "--merge" || *p == "-m");
                    self.cmd_import(parts[1], merge)?;
                }
                "analytics" | "stats" | "a" => {
                    self.cmd_analytics();
                }
                "gate" | "g" => {
                    self.cmd_gate();
                }
                "bridge" => {
                    self.cmd_bridge();
                }
                "schedule" | "sched" => {
                    self.cmd_schedule(&parts);
                }
                "metrics" | "m" => {
                    self.cmd_metrics();
                }
                "attributions" | "attr" => {
                    self.cmd_attributions();
                }
                "reset" => {
                    self.cmd_reset()?;
                }
                "where" | "path" | "location" => {
                    self.cmd_where();
                }
                _ => {
                    println!("  Unknown command: {}", parts[0]);
                    println!("  Type 'help' for available commands");
                }
            }
        }

        Ok(())
    }

    fn print_help(&self) {
        println!();
        println!("  {}MAGI Epistemic Shell Commands{}", BOLD, RESET);
        println!("  ═══════════════════════════════════════════════════════════════════");
        println!();
        println!("  {}CORE COMMANDS{}", BOLD, RESET);
        println!("  ─────────────────────────────────────────");
        println!(
            "  {}status{}       Show system status, calibration, gate state",
            CYAN, RESET
        );
        println!("  {}predict{}     Register a prediction", CYAN, RESET);
        println!("               predict <statement> [-c confidence] [-d domain]");
        println!("               Domains: code, tool, user, system, factual");
        println!("  {}resolve{}     Resolve pending prediction", CYAN, RESET);
        println!("               resolve success|failure");
        println!();
        println!("  {}VERIFICATION (External Grounding){}", BOLD, RESET);
        println!("  ─────────────────────────────────────────");
        println!(
            "  {}verify{}      Externally grounded prediction",
            CYAN, RESET
        );
        println!("               verify <type> <target> [confidence]");
        println!("               Types:");
        println!("                 file    - file exists?");
        println!("                 cmd     - command succeeds?");
        println!("                 test    - cargo test passes?");
        println!("                 pkg     - NixOS package installed?");
        println!("                 service - systemd service active?");
        println!(
            "                 {}url{}     - HTTP endpoint accessible? (NEW)",
            GREEN, RESET
        );
        println!(
            "                 {}port{}    - TCP port open? (NEW)",
            GREEN, RESET
        );
        println!(
            "                 {}dns{}     - DNS resolves? (NEW)",
            GREEN, RESET
        );
        println!(
            "  {}batch{}       Run predictions from file (NEW)",
            CYAN, RESET
        );
        println!("               batch <file_path>");
        println!(
            "  {}schedule{}    Schedule prediction for future resolution (NEW)",
            CYAN, RESET
        );
        println!("               schedule <stmt> <conf> in <time>");
        println!("               schedule list");
        println!();
        println!("  {}CALIBRATION ANALYSIS (NEW){}", BOLD, RESET);
        println!("  ─────────────────────────────────────────");
        println!(
            "  {}domains{}     Per-domain calibration breakdown",
            CYAN, RESET
        );
        println!(
            "  {}trend{}       ASCII visualization of Brier score trend",
            CYAN, RESET
        );
        println!("               trend [window_size]  (default: 50)");
        println!(
            "  {}calibration{} Show detailed calibration stats",
            CYAN, RESET
        );
        println!(
            "  {}analytics{}   Session analytics and recommendations",
            CYAN, RESET
        );
        println!();
        println!("  {}STATE MANAGEMENT (NEW){}", BOLD, RESET);
        println!("  ─────────────────────────────────────────");
        println!("  {}export{}      Export state to file", CYAN, RESET);
        println!("               export <path> [json|csv|md]");
        println!("  {}import{}      Import state from file", CYAN, RESET);
        println!("               import <path> [--merge]");
        println!();
        println!("  {}INTROSPECTION (NEW){}", BOLD, RESET);
        println!("  ─────────────────────────────────────────");
        println!(
            "  {}gate{}        Show constraint gate status & factors",
            CYAN, RESET
        );
        println!(
            "  {}bridge{}      Active Inference Bridge stats (NEW)",
            CYAN, RESET
        );
        println!(
            "  {}metrics{}     {}⚠ CRITICAL:{} λ₂ vs Φ metric audit",
            CYAN, RESET, RED, RESET
        );
        println!(
            "  {}attributions{} Show failure attribution analysis",
            CYAN, RESET
        );
        println!(
            "  {}history{}     Show attribution history (raw)",
            CYAN, RESET
        );
        println!();
        println!("  {}MONITORING (NEW){}", BOLD, RESET);
        println!("  ─────────────────────────────────────────");
        println!(
            "  {}monitor{}     Real-time epistemic dashboard (\"EEG\")",
            CYAN, RESET
        );
        println!("               Shows: signals, calibration, gate status");
        println!("               Press 'q' + Enter to quit");
        println!();
        println!("  {}UTILITIES{}", BOLD, RESET);
        println!("  ─────────────────────────────────────────");
        println!("  {}where{}       Show state file location", CYAN, RESET);
        println!(
            "  {}drift{}       Red team: inject predictions",
            CYAN, RESET
        );
        println!("               drift <count> <confidence> <success_rate>");
        println!(
            "  {}reset{}       Clear all state (cold start)",
            CYAN, RESET
        );
        println!("  {}quit{}        Exit the shell", CYAN, RESET);
        println!();
        println!(
            "  {}═══════════════════════════════════════════════════════════════════{}",
            DIM, RESET
        );
        println!();
        println!("  {}Verify Examples:{}", BOLD, RESET);
        println!("    verify file /etc/passwd 0.99      # File existence");
        println!("    verify cmd \"ls /tmp\" 0.95         # Command success");
        println!("    verify pkg firefox 0.8            # Package installed");
        println!("    verify service nginx 0.9          # Service running");
        println!("    verify url https://api.github.com 0.95  # HTTP endpoint (NEW)");
        println!("    verify port localhost:22 0.9      # TCP port (NEW)");
        println!("    verify dns google.com 0.99        # DNS resolution (NEW)");
        println!();
        println!("  {}Batch File Format:{}", BOLD, RESET);
        println!("    # predictions.txt");
        println!("    file /etc/passwd 0.99");
        println!("    url https://example.com 0.95");
        println!("    service nginx 0.85");
        println!();
        println!("  {}Red Teaming Examples:{}", BOLD, RESET);
        println!("    drift 50 0.99 0.1   # Make it delusional (overconfident + wrong)");
        println!("    drift 50 0.5 0.5    # Restore sanity (well-calibrated)");
        println!("    drift 50 0.8 0.8    # Perfect calibration");
        println!();
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // MONITOR COMMAND (Real-time TUI)
    // ═══════════════════════════════════════════════════════════════════════════

    fn cmd_monitor_help() {
        println!();
        println!(
            "{}{}╔══════════════════════════════════════════════════════════════╗{}",
            BOLD, CYAN, RESET
        );
        println!(
            "{}{}║                    MAGI MONITOR                               ║{}",
            BOLD, CYAN, RESET
        );
        println!(
            "{}{}╚══════════════════════════════════════════════════════════════╝{}",
            BOLD, CYAN, RESET
        );
        println!();
        println!("  Real-time MAGI Loop Runtime Dashboard");
        println!();
        println!("  {}Controls:{}", BOLD, RESET);
        println!("    q / Ctrl+C  - Quit monitor");
        println!("    p           - Pause/Resume runtime");
        println!("    r           - Force refresh");
        println!();
        println!("  Press Enter to start monitoring...");
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// MONITOR TUI (Synchronous version - reads from persistence)
// ═══════════════════════════════════════════════════════════════════════════════

/// Run the real-time monitor TUI (reads from persisted state)
fn run_monitor(persistence_config: PersistenceConfig) -> io::Result<()> {
    // Create the persistent model for reading state
    let model = MagiPersistentModel::with_config(persistence_config)?;

    // Set up running flag
    let running = Arc::new(AtomicBool::new(true));
    let running_clone = running.clone();

    // Spawn a thread to handle input (looks for 'q' to quit)
    std::thread::spawn(move || {
        let stdin = io::stdin();
        for line in stdin.lock().lines() {
            if let Ok(input) = line {
                if input.trim().eq_ignore_ascii_case("q") {
                    running_clone.store(false, Ordering::SeqCst);
                    break;
                }
            }
        }
    });

    // Monitor loop state
    let start_time = Instant::now();
    let mut frame_count: u64 = 0;
    let mut tick_count: u64 = 0;

    // ANSI escape codes for screen control
    const CLEAR: &str = "\x1b[2J";
    const HOME: &str = "\x1b[H";
    const HIDE_CURSOR: &str = "\x1b[?25l";
    const SHOW_CURSOR: &str = "\x1b[?25h";

    // Hide cursor during monitoring
    print!("{}", HIDE_CURSOR);
    let _ = io::stdout().flush();

    // Simulated oscillating signals (for visual effect)
    let mut signal_phase: f64 = 0.0;

    while running.load(Ordering::SeqCst) {
        frame_count += 1;
        tick_count += 1;
        signal_phase += 0.15;

        // Clear and reset cursor
        print!("{}{}", CLEAR, HOME);

        // Get current state from persisted model
        let current = model.persistence().current();
        let uptime = start_time.elapsed().as_secs();

        // Render header
        println!(
            "{}{}╔════════════════════════════════════════════════════════════════════════╗{}",
            BOLD, CYAN, RESET
        );
        println!(
            "{}{}║                    MAGI EPISTEMIC MONITOR                              ║{}",
            BOLD, CYAN, RESET
        );
        println!(
            "{}{}║      The \"EEG\" of Symthaea - Real-Time Cognitive Visualization         ║{}",
            BOLD, CYAN, RESET
        );
        println!(
            "{}{}╚════════════════════════════════════════════════════════════════════════╝{}",
            BOLD, CYAN, RESET
        );
        println!();
        println!(
            "  {}Uptime:{} {}s | {}Frame:{} #{} | {}Ctrl+C to quit{}",
            DIM, RESET, uptime, DIM, RESET, frame_count, DIM, RESET
        );
        println!();

        // Session info
        let session_number = current.session_count;
        let state_str = if current.global_stats.is_well_calibrated {
            format!("{}Running{}", GREEN, RESET)
        } else if current.global_stats.total_predictions > 50 {
            format!("{}Observing{}", YELLOW, RESET)
        } else {
            format!("{}Initializing{}", BLUE, RESET)
        };

        println!(
            "  {}╭────────────────────────────────────────────────────────────────────╮{}",
            DIM, RESET
        );
        println!(
            "  {}│{} {}RUNTIME STATE{}                                                      {}│{}",
            DIM, RESET, BOLD, RESET, DIM, RESET
        );
        println!(
            "  {}├────────────────────────────────────────────────────────────────────┤{}",
            DIM, RESET
        );
        println!(
            "  {}│{}  State: {}  │  Tick: {:>6}  │  Session: #{}               {}│{}",
            DIM, RESET, state_str, tick_count, session_number, DIM, RESET
        );
        println!(
            "  {}│{}  Lifetime iterations: {}                                     {}│{}",
            DIM, RESET, current.loop_state.loop_iterations, DIM, RESET
        );
        println!(
            "  {}╰────────────────────────────────────────────────────────────────────╯{}",
            DIM, RESET
        );
        println!();

        // Generate simulated signals based on calibration state
        let brier = current.global_stats.lifetime_brier;
        let ece = current.global_stats.ece;

        // Signal calculations with oscillation
        let base_fe = (brier + ece) / 2.0;
        let base_uncertainty = ece.min(1.0);
        let base_confidence = 1.0 - brier.min(1.0);
        let base_surprise = (brier * 1.5).min(1.0);
        let base_coherence = if current.global_stats.is_well_calibrated {
            0.8
        } else {
            0.4
        };

        // Add oscillation for visual interest
        let osc1 = (signal_phase.sin() * 0.1) as f32;
        let osc2 = ((signal_phase * 1.3).sin() * 0.08) as f32;
        let osc3 = ((signal_phase * 0.7).sin() * 0.12) as f32;

        let free_energy = (base_fe as f32 + osc1).clamp(0.0, 1.0);
        let uncertainty = (base_uncertainty as f32 + osc2).clamp(0.0, 1.0);
        let confidence = (base_confidence as f32 + osc3).clamp(0.0, 1.0);
        let surprise = (base_surprise as f32 + osc1 * 0.5).clamp(0.0, 1.0);
        let coherence = (base_coherence as f32 + osc2 * 0.5).clamp(0.0, 1.0);

        // Signals (the "EEG")
        println!(
            "  {}╭────────────────────────────────────────────────────────────────────╮{}",
            DIM, RESET
        );
        println!(
            "  {}│{} {}COGNITIVE SIGNALS (\"EEG\"){}                                         {}│{}",
            DIM, RESET, BOLD, RESET, DIM, RESET
        );
        println!(
            "  {}├────────────────────────────────────────────────────────────────────┤{}",
            DIM, RESET
        );

        print_signal_bar("Free Energy", free_energy, RED, YELLOW, GREEN);
        print_signal_bar("Uncertainty", uncertainty, GREEN, YELLOW, RED);
        print_signal_bar("Confidence ", confidence, RED, YELLOW, GREEN);
        print_signal_bar("Surprise   ", surprise, GREEN, YELLOW, RED);
        print_signal_bar("Coherence  ", coherence, RED, YELLOW, GREEN);

        println!(
            "  {}╰────────────────────────────────────────────────────────────────────╯{}",
            DIM, RESET
        );
        println!();

        // Calibration metrics
        let brier_color = if brier < 0.15 {
            GREEN
        } else if brier < 0.25 {
            YELLOW
        } else {
            RED
        };
        let calibration_quality = if brier < 0.05 {
            format!("{}Excellent{}", GREEN, RESET)
        } else if brier < 0.10 {
            format!("{}Good{}", GREEN, RESET)
        } else if brier < 0.20 {
            format!("{}Moderate{}", YELLOW, RESET)
        } else if current.global_stats.total_predictions < 50 {
            format!("{}Insufficient{}", DIM, RESET)
        } else {
            format!("{}Poor{}", RED, RESET)
        };

        println!(
            "  {}╭────────────────────────────────────────────────────────────────────╮{}",
            DIM, RESET
        );
        println!(
            "  {}│{} {}CALIBRATION{}                                                        {}│{}",
            DIM, RESET, BOLD, RESET, DIM, RESET
        );
        println!(
            "  {}├────────────────────────────────────────────────────────────────────┤{}",
            DIM, RESET
        );
        println!(
            "  {}│{}  Brier: {}{:.4}{}  │  ECE: {:.4}  │  Quality: {}     {}│{}",
            DIM, RESET, brier_color, brier, RESET, ece, calibration_quality, DIM, RESET
        );
        println!(
            "  {}│{}  Predictions: {} total  │  Correct: {}                    {}│{}",
            DIM,
            RESET,
            current.global_stats.total_predictions,
            current.global_stats.correct_predictions,
            DIM,
            RESET
        );
        println!(
            "  {}│{}  Lifetime Brier Sum: {:.4}                                  {}│{}",
            DIM, RESET, current.global_stats.brier_sum, DIM, RESET
        );
        println!(
            "  {}╰────────────────────────────────────────────────────────────────────╯{}",
            DIM, RESET
        );
        println!();

        // Gate status (derived from calibration state)
        let (gate_color, gate_str, gate_open) = if current.gate_config.force_supervised_mode {
            (MAGENTA, "SUPERVISED (Forced)", false)
        } else if current.global_stats.is_well_calibrated {
            (GREEN, "AUTONOMOUS", true)
        } else if brier > 0.30 {
            (RED, "SUPERVISED (Poor Calibration)", false)
        } else {
            (YELLOW, "DRY RUN (Building History)", false)
        };

        println!(
            "  {}╭────────────────────────────────────────────────────────────────────╮{}",
            DIM, RESET
        );
        println!(
            "  {}│{} {}CONSTRAINT GATE{}                                                    {}│{}",
            DIM, RESET, BOLD, RESET, DIM, RESET
        );
        println!(
            "  {}├────────────────────────────────────────────────────────────────────┤{}",
            DIM, RESET
        );
        println!(
            "  {}│{}  Mode: {}{}{}                                       {}│{}",
            DIM, RESET, gate_color, gate_str, RESET, DIM, RESET
        );
        println!(
            "  {}│{}  Gate: {}  │  Min Preds: {}  │  Min Accuracy: {:.0}%  {}│{}",
            DIM,
            RESET,
            if gate_open {
                format!("{}OPEN{}", GREEN, RESET)
            } else {
                format!("{}RESTRICTED{}", RED, RESET)
            },
            current.gate_config.min_predictions_for_autonomy,
            current.gate_config.min_accuracy_for_autonomy * 100.0,
            DIM,
            RESET
        );
        println!(
            "  {}╰────────────────────────────────────────────────────────────────────╯{}",
            DIM, RESET
        );
        println!();

        // Per-domain calibration
        if !current.calibration.is_empty() {
            println!(
                "  {}╭────────────────────────────────────────────────────────────────────╮{}",
                DIM, RESET
            );
            println!(
                "  {}│{} {}DOMAIN CALIBRATION{}                                                {}│{}",
                DIM, RESET, BOLD, RESET, DIM, RESET
            );
            println!(
                "  {}├────────────────────────────────────────────────────────────────────┤{}",
                DIM, RESET
            );
            for (domain, cal) in &current.calibration {
                let d_color = if cal.lifetime_brier < 0.15 {
                    GREEN
                } else if cal.lifetime_brier < 0.25 {
                    YELLOW
                } else {
                    RED
                };
                println!(
                    "  {}│{}  {:?}: {}{:.4}{}  (n={})                             {}│{}",
                    DIM,
                    RESET,
                    domain,
                    d_color,
                    cal.lifetime_brier,
                    RESET,
                    cal.prediction_count,
                    DIM,
                    RESET
                );
            }
            println!(
                "  {}╰────────────────────────────────────────────────────────────────────╯{}",
                DIM, RESET
            );
            println!();
        }

        // Loop state
        println!(
            "  {}╭────────────────────────────────────────────────────────────────────╮{}",
            DIM, RESET
        );
        println!(
            "  {}│{} {}MAGI LOOP STATE{}                                                    {}│{}",
            DIM, RESET, BOLD, RESET, DIM, RESET
        );
        println!(
            "  {}├────────────────────────────────────────────────────────────────────┤{}",
            DIM, RESET
        );
        println!(
            "  {}│{}  Predictions Made:     {:>6}                                    {}│{}",
            DIM, RESET, current.loop_state.predictions_made, DIM, RESET
        );
        println!(
            "  {}│{}  Predictions Resolved: {:>6}                                    {}│{}",
            DIM, RESET, current.loop_state.predictions_resolved, DIM, RESET
        );
        println!(
            "  {}│{}  Attributions:         {:>6}                                    {}│{}",
            DIM, RESET, current.loop_state.attributions_generated, DIM, RESET
        );
        println!(
            "  {}│{}  Calibration Quality:  {:?}                               {}│{}",
            DIM, RESET, current.loop_state.calibration_quality, DIM, RESET
        );
        println!(
            "  {}╰────────────────────────────────────────────────────────────────────╯{}",
            DIM, RESET
        );

        let _ = io::stdout().flush();

        // Sleep before next frame (4 Hz update)
        std::thread::sleep(Duration::from_millis(250));
    }

    // Show cursor again
    print!("{}", SHOW_CURSOR);
    let _ = io::stdout().flush();

    println!("\n{}Monitor stopped.{}", DIM, RESET);

    Ok(())
}

/// Helper to print a signal bar
fn print_signal_bar(name: &str, value: f32, low_color: &str, mid_color: &str, high_color: &str) {
    const BAR_WIDTH: usize = 30;
    const DIM: &str = "\x1b[2m";
    const RESET: &str = "\x1b[0m";

    let normalized = (value.clamp(0.0, 1.0) * BAR_WIDTH as f32) as usize;
    let color = if value < 0.3 {
        low_color
    } else if value < 0.7 {
        mid_color
    } else {
        high_color
    };

    print!("  {}│{}  {}: [", DIM, RESET, name);
    print!("{}", color);
    for _ in 0..normalized {
        print!("█");
    }
    print!("{}", RESET);
    for _ in normalized..BAR_WIDTH {
        print!("░");
    }
    println!("] {:.2}         {}│{}", value, DIM, RESET);
}

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════════════════════

fn main() -> io::Result<()> {
    let args: Vec<String> = std::env::args().collect();

    // Parse global options
    let verbose = args.iter().any(|a| a == "-v" || a == "--verbose");
    let state_path = args
        .iter()
        .position(|a| a == "--state")
        .and_then(|i| args.get(i + 1))
        .map(PathBuf::from);

    // Setup persistence config
    let config = PersistenceConfig {
        state_path: state_path.unwrap_or_else(|| PathBuf::from(".magi_state.json")),
        autosave_interval: 5,
        create_backups: true,
        max_backups: 3,
        enabled: true,
    };

    let mut cli = MagiCli::new(config.clone(), verbose)?;

    // Find command (skip program name and options)
    let cmd_args: Vec<&str> = args
        .iter()
        .skip(1)
        .filter(|a| !a.starts_with('-'))
        .map(|s| s.as_str())
        .collect();

    if cmd_args.is_empty() || cmd_args[0] == "interactive" {
        cli.interactive()?;
    } else {
        match cmd_args[0] {
            "status" => cli.cmd_status(),
            "predict" => {
                if cmd_args.len() < 2 {
                    eprintln!("Usage: magi_cli predict <statement> [-c confidence] [-d domain]");
                    std::process::exit(1);
                }

                let mut statement_parts = vec![];
                let mut confidence = 0.8;
                let mut domain = PredictionDomain::Factual;
                let mut i = 1;

                while i < cmd_args.len() {
                    if cmd_args[i] == "-c" && i + 1 < cmd_args.len() {
                        confidence = cmd_args[i + 1].parse().unwrap_or(0.8);
                        i += 2;
                    } else if cmd_args[i] == "-d" && i + 1 < cmd_args.len() {
                        domain = match cmd_args[i + 1].to_lowercase().as_str() {
                            "code" => PredictionDomain::CodeExecution,
                            "tool" => PredictionDomain::ToolUse,
                            "user" => PredictionDomain::UserBehavior,
                            "system" => PredictionDomain::SystemState,
                            _ => PredictionDomain::Factual,
                        };
                        i += 2;
                    } else {
                        statement_parts.push(cmd_args[i]);
                        i += 1;
                    }
                }

                cli.cmd_predict(&statement_parts.join(" "), confidence, domain);
            }
            "resolve" => {
                if cmd_args.len() < 2 {
                    eprintln!("Usage: magi_cli resolve success|failure");
                    std::process::exit(1);
                }
                let success = matches!(cmd_args[1], "success" | "true" | "1");
                cli.cmd_resolve(success)?;
            }
            "verify" => {
                if cmd_args.len() < 3 {
                    eprintln!("Usage: magi_cli verify <type> <target> [confidence]");
                    eprintln!("Types: file, cmd, test, pkg, service");
                    std::process::exit(1);
                }
                let check_type = cmd_args[1];
                let target = cmd_args[2];
                let confidence = cmd_args.get(3).and_then(|s| s.parse().ok()).unwrap_or(0.8);
                cli.cmd_verify(check_type, target, confidence)?;
            }
            "calibration" | "cal" => cli.cmd_calibration(),
            "history" => cli.cmd_history(),
            "domains" | "dom" => cli.cmd_domains(),
            "trend" => {
                let window = cmd_args.get(1).and_then(|s| s.parse().ok()).unwrap_or(50);
                cli.cmd_trend(window);
            }
            "batch" => {
                if cmd_args.len() < 2 {
                    eprintln!("Usage: magi_cli batch <file_path>");
                    std::process::exit(1);
                }
                cli.cmd_batch(cmd_args[1])?;
            }
            "export" => {
                if cmd_args.len() < 2 {
                    eprintln!("Usage: magi_cli export <output_path> [format]");
                    std::process::exit(1);
                }
                let format = cmd_args.get(2).unwrap_or(&"json");
                cli.cmd_export(cmd_args[1], format)?;
            }
            "import" => {
                if cmd_args.len() < 2 {
                    eprintln!("Usage: magi_cli import <input_path> [--merge]");
                    std::process::exit(1);
                }
                let merge = cmd_args.iter().any(|p| *p == "--merge" || *p == "-m");
                cli.cmd_import(cmd_args[1], merge)?;
            }
            "analytics" | "stats" => cli.cmd_analytics(),
            "gate" => cli.cmd_gate(),
            "bridge" => cli.cmd_bridge(),
            "schedule" | "sched" => cli.cmd_schedule(&cmd_args),
            "metrics" => cli.cmd_metrics(),
            "attributions" | "attr" => cli.cmd_attributions(),
            "drift" => {
                let count = cmd_args.get(1).and_then(|s| s.parse().ok()).unwrap_or(10);
                let confidence = cmd_args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0.99);
                let success_rate = cmd_args.get(3).and_then(|s| s.parse().ok()).unwrap_or(0.2);
                cli.cmd_drift(count, confidence, success_rate)?;
            }
            "reset" => cli.cmd_reset()?,
            "where" | "path" | "location" => cli.cmd_where(),
            "monitor" | "mon" => {
                // Monitor requires special handling - uses tokio runtime
                MagiCli::cmd_monitor_help();
                let _ = std::io::stdin().read_line(&mut String::new());
                return run_monitor(config);
            }
            "help" => cli.print_help(),
            _ => {
                eprintln!("Unknown command: {}", cmd_args[0]);
                eprintln!(
                    "Commands: status, predict, resolve, verify, calibration, domains, trend,"
                );
                eprintln!(
                    "          batch, export, import, analytics, gate, metrics, attributions,"
                );
                eprintln!("          history, where, drift, reset, monitor, help");
                std::process::exit(1);
            }
        }
    }

    Ok(())
}
