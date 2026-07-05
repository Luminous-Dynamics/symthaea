// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! TUI Application State Machine
//!
//! Manages the terminal UI lifecycle: initialization, event loop,
//! rendering, and cleanup. Orchestrates the layout of all widgets
//! and handles keyboard input.

use std::io;
use std::time::{Duration, Instant};

use crossterm::{
    ExecutableCommand,
    event::{self, Event, KeyCode, KeyEventKind},
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use ratatui::{
    Terminal,
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
};

use super::widgets::{
    AlertsPanel, AlertsSnapshot, CausalExplorer, CausalLink, ConsciousnessGauge,
    ConsciousnessState, GenerationTimeline, HealthSnapshot, SystemHealth, TimelineEntry,
    WorldModelSnapshot, WorldModelView,
};
use crate::ipc::{self, DaemonSnapshot};
use crate::mind::active_inference::NixActiveInference;

/// Adaptive complexity level — controls how much detail the TUI shows.
///
/// Ported from the Python TUI's `AdaptiveInterface` / `ComplexityLevel`.
/// The TUI auto-adjusts based on user interaction success rate, or the
/// user can cycle manually with `Ctrl-L`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComplexityLevel {
    /// Minimal: consciousness gauge + input only (zen mode).
    Zen,
    /// Beginner: gauge + health + input/output.
    Beginner,
    /// Standard: all panels visible.
    Standard,
    /// Expert: all panels + alerts + debug info.
    Expert,
}

impl ComplexityLevel {
    fn next(self) -> Self {
        match self {
            Self::Zen => Self::Beginner,
            Self::Beginner => Self::Standard,
            Self::Standard => Self::Expert,
            Self::Expert => Self::Zen,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Zen => "Zen",
            Self::Beginner => "Beginner",
            Self::Standard => "Standard",
            Self::Expert => "Expert",
        }
    }
}

/// Which panel is focused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FocusPanel {
    Consciousness,
    Health,
    Generations,
    WorldModel,
    CausalGraph,
    Alerts,
    Input,
}

/// Application state.
pub struct App {
    /// Active inference engine.
    engine: NixActiveInference,
    /// Current consciousness state for display.
    consciousness: ConsciousnessState,
    /// System health snapshot.
    health: HealthSnapshot,
    /// Generation entries.
    generations: Vec<TimelineEntry>,
    /// World model snapshot.
    world_model: WorldModelSnapshot,
    /// Causal links.
    causal_links: Vec<CausalLink>,
    /// Current text input buffer.
    input: String,
    /// Output/response text.
    output: Vec<String>,
    /// Focused panel.
    focus: FocusPanel,
    /// Alerts snapshot.
    alerts: AlertsSnapshot,
    /// Scroll offset for causal explorer.
    causal_scroll: usize,
    /// Whether the app should quit.
    should_quit: bool,
    /// Tick counter for status updates.
    tick: u64,
    /// Dry-run mode.
    _dry_run: bool,
    /// Adaptive complexity level.
    complexity: ComplexityLevel,
    /// Successful interaction count (for auto-complexity).
    success_count: u32,
    /// Total interaction count (for auto-complexity).
    total_count: u32,
    /// Path to daemon IPC snapshot.
    daemon_snapshot_path: std::path::PathBuf,
    /// Last daemon snapshot (if available).
    daemon_snapshot: Option<DaemonSnapshot>,
    /// Free energy history for sparkline (last 60 samples).
    fe_history: Vec<f64>,
    /// Anomaly count history for trend display.
    anomaly_history: Vec<u64>,
    /// Consciousness state history for trajectory display (last 20).
    consciousness_history: Vec<ConsciousnessState>,
    /// Memory usage history for sparkline (last 30 samples).
    memory_history: Vec<f64>,
    /// Prediction MAE history for trend tracking (last 20 samples).
    mae_history: Vec<f64>,
}

impl App {
    /// Create a new application.
    pub fn new(dry_run: bool) -> Self {
        Self {
            engine: NixActiveInference::new(),
            consciousness: ConsciousnessState::default(),
            health: HealthSnapshot::default(),
            generations: Vec::new(),
            world_model: WorldModelSnapshot::default(),
            causal_links: Vec::new(),
            input: String::new(),
            output: vec!["Welcome to nix-mind TUI. Type a command or press 'q' to quit.".into()],
            focus: FocusPanel::Input,
            alerts: AlertsSnapshot::default(),
            causal_scroll: 0,
            should_quit: false,
            tick: 0,
            _dry_run: dry_run,
            complexity: ComplexityLevel::Standard,
            success_count: 0,
            total_count: 0,
            daemon_snapshot_path: ipc::default_snapshot_path(),
            daemon_snapshot: None,
            fe_history: Vec::with_capacity(60),
            anomaly_history: Vec::with_capacity(60),
            consciousness_history: Vec::with_capacity(20),
            memory_history: Vec::with_capacity(30),
            mae_history: Vec::with_capacity(20),
        }
    }

    /// Run the TUI event loop.
    pub fn run(&mut self) -> io::Result<()> {
        enable_raw_mode()?;
        io::stdout().execute(EnterAlternateScreen)?;
        let backend = CrosstermBackend::new(io::stdout());
        let mut terminal = Terminal::new(backend)?;

        let tick_rate = Duration::from_millis(250);
        let mut last_tick = Instant::now();

        // Initial data load
        self.refresh_data();

        while !self.should_quit {
            terminal.draw(|frame| self.draw(frame))?;

            let timeout = tick_rate.saturating_sub(last_tick.elapsed());
            if event::poll(timeout)? {
                if let Event::Key(key) = event::read()? {
                    if key.kind == KeyEventKind::Press {
                        self.handle_key(key.code);
                    }
                }
            }

            if last_tick.elapsed() >= tick_rate {
                self.tick += 1;
                // Refresh faster when daemon is running (every ~2s = 8 ticks),
                // slower when direct-querying (every ~4s = 16 ticks)
                let refresh_interval = if self.daemon_snapshot.is_some() {
                    8
                } else {
                    16
                };
                if self.tick % refresh_interval == 0 {
                    self.refresh_data();
                }
                last_tick = Instant::now();
            }
        }

        disable_raw_mode()?;
        io::stdout().execute(LeaveAlternateScreen)?;
        Ok(())
    }

    /// Handle a key press.
    fn handle_key(&mut self, key: KeyCode) {
        match key {
            KeyCode::Char('q') if self.focus != FocusPanel::Input => {
                self.should_quit = true;
            }
            KeyCode::Esc => {
                if self.focus == FocusPanel::Input && !self.input.is_empty() {
                    self.input.clear();
                } else {
                    self.should_quit = true;
                }
            }
            KeyCode::Tab => {
                self.focus = match self.focus {
                    FocusPanel::Input => FocusPanel::Consciousness,
                    FocusPanel::Consciousness => FocusPanel::Health,
                    FocusPanel::Health => FocusPanel::Generations,
                    FocusPanel::Generations => FocusPanel::WorldModel,
                    FocusPanel::WorldModel => FocusPanel::CausalGraph,
                    FocusPanel::CausalGraph => FocusPanel::Alerts,
                    FocusPanel::Alerts => FocusPanel::Input,
                };
            }
            // Ctrl-L: cycle complexity level
            KeyCode::Char('l') if self.focus != FocusPanel::Input => {
                self.complexity = self.complexity.next();
                self.output.push(format!(
                    "Complexity: {} (Ctrl-L to cycle)",
                    self.complexity.name()
                ));
            }
            // Ctrl-Z: toggle zen mode
            KeyCode::Char('z') if self.focus != FocusPanel::Input => {
                if self.complexity == ComplexityLevel::Zen {
                    self.complexity = ComplexityLevel::Standard;
                    self.output.push("Exited zen mode.".into());
                } else {
                    self.complexity = ComplexityLevel::Zen;
                    self.output
                        .push("Zen mode: minimal interface. Press 'z' to exit.".into());
                }
            }
            KeyCode::Char(c) if self.focus == FocusPanel::Input => {
                self.input.push(c);
            }
            KeyCode::Backspace if self.focus == FocusPanel::Input => {
                self.input.pop();
            }
            KeyCode::Enter if self.focus == FocusPanel::Input => {
                self.process_input();
            }
            KeyCode::Up if self.focus == FocusPanel::CausalGraph => {
                self.causal_scroll = self.causal_scroll.saturating_sub(1);
            }
            KeyCode::Down if self.focus == FocusPanel::CausalGraph => {
                if self.causal_scroll + 1 < self.causal_links.len() {
                    self.causal_scroll += 1;
                }
            }
            KeyCode::Up if self.focus == FocusPanel::Alerts => {
                self.alerts.selected_index = self.alerts.selected_index.saturating_sub(1);
            }
            KeyCode::Down if self.focus == FocusPanel::Alerts => {
                let max = self.alerts.active_alerts.len().saturating_sub(1);
                if self.alerts.selected_index < max {
                    self.alerts.selected_index += 1;
                }
            }
            // Acknowledge alert
            KeyCode::Char('a') if self.focus == FocusPanel::Alerts => {
                if let Some(alert) = self.alerts.active_alerts.get(self.alerts.selected_index) {
                    self.alerts.acknowledged.insert(alert.metric.clone());
                }
            }
            _ => {}
        }
    }

    /// Process the current input text.
    fn process_input(&mut self) {
        let input = self.input.clone();
        self.input.clear();

        if input.is_empty() {
            return;
        }

        let plan = self.engine.process_input(&input);

        // Update consciousness state
        self.consciousness = ConsciousnessState {
            phi: 0.7, // Would come from full Symthaea integration
            confidence: plan.goal.confidence,
            free_energy: plan.current_free_energy,
        };

        // Update world model view
        let hierarchy = self.engine.world_model().prediction_hierarchy();
        self.world_model = WorldModelSnapshot {
            level_errors: hierarchy.errors(),
            free_energy: hierarchy.free_energy(),
            learned_actions: self.engine.world_model().learned_action_count(),
            total_observations: self.engine.world_model().total_observations(),
            is_surprised: hierarchy.is_surprised(),
            memory_items: self
                .engine
                .goal_inference()
                .working_memory()
                .items()
                .iter()
                .map(|item| (item.label.clone(), item.activation))
                .collect(),
            ..Default::default()
        };

        // Build output
        self.output.clear();
        self.output.push(format!("> {}", input));
        self.output.push(format!("Goal: {}", plan.goal.description));

        if plan.needs_clarification {
            self.output
                .push("Needs clarification - please be more specific.".into());
        } else if let Some(best) = plan.actions.first() {
            self.output.push(format!(
                "Best: {:?} (EFE={:.3}, pragmatic={:.2}, epistemic={:.2})",
                best.action, best.expected_free_energy, best.pragmatic_value, best.epistemic_value,
            ));
            self.success_count += 1;
        }
        self.total_count += 1;

        // Auto-adjust complexity based on success rate
        self.auto_adjust_complexity();
    }

    /// Adjust complexity level based on interaction success rate.
    fn auto_adjust_complexity(&mut self) {
        if self.total_count < 5 {
            return; // Wait for enough data
        }
        let rate = self.success_count as f64 / self.total_count as f64;
        let new = if rate > 0.8 {
            ComplexityLevel::Expert
        } else if rate > 0.5 {
            ComplexityLevel::Standard
        } else {
            ComplexityLevel::Beginner
        };
        // Only auto-adjust if user hasn't set Zen manually
        if self.complexity != ComplexityLevel::Zen {
            self.complexity = new;
        }
    }

    /// Refresh system data — first from daemon IPC, then direct queries as fallback.
    fn refresh_data(&mut self) {
        // Try to read daemon snapshot (preferred — richer data from continuous awareness)
        self.daemon_snapshot = DaemonSnapshot::read_from(&self.daemon_snapshot_path)
            .filter(|snap| snap.is_fresh(120) && snap.daemon_alive());

        if let Some(snap) = self.daemon_snapshot.clone() {
            self.apply_daemon_snapshot(&snap);
        } else {
            // Fallback: direct system queries (less rich, no prediction data)
            self.refresh_data_direct();
        }
    }

    /// Apply daemon snapshot data to TUI state.
    fn apply_daemon_snapshot(&mut self, snap: &DaemonSnapshot) {
        // Track free energy history for sparkline
        self.fe_history.push(snap.free_energy);
        if self.fe_history.len() > 60 {
            self.fe_history.remove(0);
        }
        self.anomaly_history.push(snap.anomaly_count);
        if self.anomaly_history.len() > 60 {
            self.anomaly_history.remove(0);
        }

        // Consciousness state from daemon's predictive hierarchy
        self.consciousness = ConsciousnessState {
            phi: snap.free_energy.min(1.0),
            confidence: 1.0 - snap.free_energy, // higher FE = lower confidence
            free_energy: snap.free_energy,
        };

        // Track consciousness trajectory
        self.consciousness_history.push(self.consciousness);
        if self.consciousness_history.len() > 20 {
            self.consciousness_history.remove(0);
        }

        // Track memory usage for sparkline
        if let Some(pct) = snap.memory_used_percent {
            self.memory_history.push(pct);
            if self.memory_history.len() > 30 {
                self.memory_history.remove(0);
            }
        }

        // World model from daemon
        self.world_model = WorldModelSnapshot {
            level_errors: snap.hierarchy_errors,
            free_energy: snap.free_energy,
            learned_actions: snap.causal_edge_count,
            total_observations: snap.observation_count as usize,
            is_surprised: snap.is_surprised,
            memory_items: snap
                .concerns
                .iter()
                .map(|c| (c.label.clone(), c.activation))
                .collect(),
            drift_similarity: snap.drift_similarity,
            episodic_count: snap.episodic_count,
            prediction_accuracy: snap.prediction_accuracy,
            maintenance_plan_count: snap.maintenance_plan_count,
            ..Default::default()
        };

        // Causal links — prefer real causal edges, fall back to anomalies
        if !snap.top_causal_edges.is_empty() {
            self.causal_links = snap
                .top_causal_edges
                .iter()
                .map(|e| CausalLink {
                    from: e.from.clone(),
                    to: e.to.clone(),
                    confidence: e.confidence,
                    relationship: "causal".to_string(),
                    suggestion: None,
                })
                .collect();
            // Append anomalies as secondary entries
            for a in &snap.recent_anomalies {
                self.causal_links.push(CausalLink {
                    from: a.unit.clone(),
                    to: a.reason.clone(),
                    confidence: a.score,
                    relationship: "anomaly".to_string(),
                    suggestion: a.suggestion.clone(),
                });
            }
        } else {
            self.causal_links = snap
                .recent_anomalies
                .iter()
                .map(|a| CausalLink {
                    from: a.unit.clone(),
                    to: a.reason.clone(),
                    confidence: a.score,
                    relationship: "anomaly".to_string(),
                    suggestion: a.suggestion.clone(),
                })
                .collect();
        }

        // Health: merge daemon stats with direct service queries
        self.health.services_failed = snap.anomaly_count as usize;
        self.health.memory_used_percent = snap.memory_used_percent;
        self.health.memory_history = self.memory_history.clone();
        self.health.load_average_1m = snap.load_average_1m;
        self.health.swap_used_percent = snap.swap_used_percent;

        // Alerts: prefer real predictive alerts from daemon, fall back to anomaly-derived
        if !snap.alerts.is_empty() {
            self.alerts.active_alerts = snap.alerts.clone();
        } else {
            self.alerts.active_alerts = snap
                .recent_anomalies
                .iter()
                .map(|a| crate::ipc::AlertEntry {
                    metric: a.unit.clone(),
                    current_value: a.score * 100.0,
                    predicted_value: a.score * 120.0,
                    hours_ahead: 6.0,
                    threshold: 80.0,
                    confidence: a.score,
                    recommended_action: Some(format!("Investigate {}", a.unit)),
                    severity: if a.score > 0.8 {
                        crate::ipc::AlertSeverity::Critical
                    } else if a.score > 0.5 {
                        crate::ipc::AlertSeverity::Warning
                    } else {
                        crate::ipc::AlertSeverity::Info
                    },
                    first_seen: snap.timestamp,
                    last_seen: snap.timestamp,
                    consecutive_cycles: 1,
                    prev_predicted_value: None,
                    journal_context: vec![a.reason.clone()],
                })
                .collect();
        }
        self.alerts.support_status = snap.support_status.clone();
        self.alerts.recommendation_count = snap.recommendation_count;
        self.alerts.watchdog_status = snap.watchdog_status.clone();
        self.alerts.prediction_accuracy = snap.prediction_accuracy;

        // Track MAE history for trend display (BF)
        if let Some(mae) = snap.prediction_accuracy {
            self.mae_history.push(mae);
            if self.mae_history.len() > 20 {
                self.mae_history.remove(0);
            }
        }
        self.alerts.prediction_mae_history = self.mae_history.clone();

        self.refresh_generations();
    }

    /// Direct system queries (fallback when daemon not running).
    fn refresh_data_direct(&mut self) {
        if let Ok(units) = crate::observe::systemd::SystemdObserver::list_units() {
            self.health.services_total = units.len();
            self.health.services_running =
                units.iter().filter(|u| u.active_state == "active").count();
            self.health.services_failed =
                units.iter().filter(|u| u.active_state == "failed").count();
        }
        self.refresh_generations();
    }

    /// Refresh generation data (shared by both daemon and direct paths).
    fn refresh_generations(&mut self) {
        if let Ok(gens) = crate::action::generation_manager::GenerationManager::list() {
            self.health.total_generations = gens.len();
            self.health.current_generation = gens.iter().find(|g| g.current).map(|g| g.number);
            self.generations = gens
                .iter()
                .map(|g| TimelineEntry {
                    number: g.number,
                    date: g.date.clone(),
                    current: g.current,
                    size_delta_bytes: None,
                    status: None,
                    age_hours: None,
                })
                .collect();
        }
    }

    /// Draw the UI, adapting layout to the current complexity level.
    fn draw(&self, frame: &mut ratatui::Frame) {
        let size = frame.area();

        match self.complexity {
            ComplexityLevel::Zen => self.draw_zen(frame, size),
            ComplexityLevel::Beginner => self.draw_beginner(frame, size),
            ComplexityLevel::Standard => self.draw_standard(frame, size),
            ComplexityLevel::Expert => self.draw_expert(frame, size),
        }
    }

    /// Zen mode: consciousness gauge + input only.
    fn draw_zen(&self, frame: &mut ratatui::Frame, size: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(8),
                Constraint::Length(3),
                Constraint::Min(1),
            ])
            .split(size);

        let cons_block = Block::default()
            .title(" Consciousness [Zen] ")
            .borders(Borders::ALL)
            .border_style(self.border_style(FocusPanel::Consciousness));
        let gauge = ConsciousnessGauge::new(self.consciousness)
            .history(self.consciousness_history.clone())
            .block(cons_block);
        frame.render_widget(gauge, chunks[0]);

        self.draw_input(frame, chunks[1]);
        self.draw_output(frame, chunks[2]);
    }

    /// Beginner: gauge + health + input/output.
    fn draw_beginner(&self, frame: &mut ratatui::Frame, size: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(8),
                Constraint::Length(5),
                Constraint::Length(3),
                Constraint::Min(1),
            ])
            .split(size);

        let cons_block = Block::default()
            .title(" Consciousness [Beginner] ")
            .borders(Borders::ALL)
            .border_style(self.border_style(FocusPanel::Consciousness));
        frame.render_widget(
            ConsciousnessGauge::new(self.consciousness)
                .history(self.consciousness_history.clone())
                .block(cons_block),
            chunks[0],
        );

        let health_block = Block::default()
            .title(" System Health ")
            .borders(Borders::ALL)
            .border_style(self.border_style(FocusPanel::Health));
        frame.render_widget(
            SystemHealth::new(self.health.clone()).block(health_block),
            chunks[1],
        );

        self.draw_input(frame, chunks[2]);
        self.draw_output(frame, chunks[3]);
    }

    /// Standard: all panels visible (original layout).
    fn draw_standard(&self, frame: &mut ratatui::Frame, size: Rect) {
        let main_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(10), Constraint::Length(6)])
            .split(size);

        let top_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(40), Constraint::Percentage(60)])
            .split(main_chunks[0]);

        // Left column
        let left_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Percentage(50),
                Constraint::Percentage(25),
                Constraint::Percentage(25),
            ])
            .split(top_chunks[0]);

        let cons_block = Block::default()
            .title(" Consciousness ")
            .borders(Borders::ALL)
            .border_style(self.border_style(FocusPanel::Consciousness));
        frame.render_widget(
            ConsciousnessGauge::new(self.consciousness)
                .history(self.consciousness_history.clone())
                .block(cons_block),
            left_chunks[0],
        );

        let health_block = Block::default()
            .title(" System Health ")
            .borders(Borders::ALL)
            .border_style(self.border_style(FocusPanel::Health));
        frame.render_widget(
            SystemHealth::new(self.health.clone()).block(health_block),
            left_chunks[1],
        );

        let gen_block = Block::default()
            .title(" Generations ")
            .borders(Borders::ALL)
            .border_style(self.border_style(FocusPanel::Generations));
        frame.render_widget(
            GenerationTimeline::new(self.generations.clone()).block(gen_block),
            left_chunks[2],
        );

        // Right column
        let right_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(top_chunks[1]);

        let model_block = Block::default()
            .title(" World Model ")
            .borders(Borders::ALL)
            .border_style(self.border_style(FocusPanel::WorldModel));
        frame.render_widget(
            WorldModelView::new(self.world_model.clone()).block(model_block),
            right_chunks[0],
        );

        let causal_block = Block::default()
            .title(" Causal Graph ")
            .borders(Borders::ALL)
            .border_style(self.border_style(FocusPanel::CausalGraph));
        frame.render_widget(
            CausalExplorer::new(self.causal_links.clone())
                .scroll(self.causal_scroll)
                .block(causal_block),
            right_chunks[1],
        );

        // Bottom: input + output
        let bottom_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(3), Constraint::Min(1)])
            .split(main_chunks[1]);

        self.draw_input(frame, bottom_chunks[0]);
        self.draw_output(frame, bottom_chunks[1]);
    }

    /// Expert: all panels + alerts panel.
    fn draw_expert(&self, frame: &mut ratatui::Frame, size: Rect) {
        let main_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(10), Constraint::Length(8)])
            .split(size);

        let top_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(30),
                Constraint::Percentage(35),
                Constraint::Percentage(35),
            ])
            .split(main_chunks[0]);

        // Left column: consciousness + health + generations
        let left_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Percentage(40),
                Constraint::Percentage(30),
                Constraint::Percentage(30),
            ])
            .split(top_chunks[0]);

        let cons_block = Block::default()
            .title(" Consciousness [Expert] ")
            .borders(Borders::ALL)
            .border_style(self.border_style(FocusPanel::Consciousness));
        frame.render_widget(
            ConsciousnessGauge::new(self.consciousness)
                .history(self.consciousness_history.clone())
                .block(cons_block),
            left_chunks[0],
        );

        let health_block = Block::default()
            .title(" System Health ")
            .borders(Borders::ALL)
            .border_style(self.border_style(FocusPanel::Health));
        frame.render_widget(
            SystemHealth::new(self.health.clone()).block(health_block),
            left_chunks[1],
        );

        let gen_block = Block::default()
            .title(" Generations ")
            .borders(Borders::ALL)
            .border_style(self.border_style(FocusPanel::Generations));
        frame.render_widget(
            GenerationTimeline::new(self.generations.clone()).block(gen_block),
            left_chunks[2],
        );

        // Middle column: world model + causal graph
        let mid_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(top_chunks[1]);

        let model_block = Block::default()
            .title(" World Model ")
            .borders(Borders::ALL)
            .border_style(self.border_style(FocusPanel::WorldModel));
        frame.render_widget(
            WorldModelView::new(self.world_model.clone()).block(model_block),
            mid_chunks[0],
        );

        let causal_block = Block::default()
            .title(" Causal Graph ")
            .borders(Borders::ALL)
            .border_style(self.border_style(FocusPanel::CausalGraph));
        frame.render_widget(
            CausalExplorer::new(self.causal_links.clone())
                .scroll(self.causal_scroll)
                .block(causal_block),
            mid_chunks[1],
        );

        // Right column: alerts panel
        let alerts_block = Block::default()
            .title(" Alerts & Predictions ")
            .borders(Borders::ALL)
            .border_style(self.border_style(FocusPanel::Alerts));
        let mut alerts_snap = self.alerts.clone();
        alerts_snap.is_focused = self.focus == FocusPanel::Alerts;
        frame.render_widget(
            AlertsPanel::new(alerts_snap).block(alerts_block),
            top_chunks[2],
        );

        // Bottom: input + output
        let bottom_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(3), Constraint::Min(1)])
            .split(main_chunks[1]);

        self.draw_input(frame, bottom_chunks[0]);
        self.draw_output(frame, bottom_chunks[1]);
    }

    /// Render a text-based sparkline of the free energy history.
    fn fe_sparkline(&self) -> String {
        if self.fe_history.is_empty() {
            return String::new();
        }
        const SPARKS: &[char] = &[
            ' ', '\u{2581}', '\u{2582}', '\u{2583}', '\u{2584}', '\u{2585}', '\u{2586}',
            '\u{2587}', '\u{2588}',
        ];
        let max = self.fe_history.iter().cloned().fold(0.01_f64, f64::max);
        self.fe_history
            .iter()
            .map(|v| {
                let idx = ((v / max) * (SPARKS.len() - 1) as f64).round() as usize;
                SPARKS[idx.min(SPARKS.len() - 1)]
            })
            .collect()
    }

    /// Draw the input bar.
    fn draw_input(&self, frame: &mut ratatui::Frame, area: Rect) {
        let daemon_status = match &self.daemon_snapshot {
            Some(snap) => {
                let fe_trend = self.fe_sparkline();
                format!(
                    " [daemon pid {} | {} obs | {} anomalies | FE:{}]",
                    snap.daemon_pid, snap.observation_count, snap.anomaly_count, fe_trend
                )
            }
            None => " [daemon offline]".to_string(),
        };
        let complexity_tag = format!(" [{}]", self.complexity.name());
        let input_block = Block::default()
            .title(format!(
                " Input (Tab/Esc){}{} ",
                daemon_status, complexity_tag
            ))
            .borders(Borders::ALL)
            .border_style(self.border_style(FocusPanel::Input));
        let cursor = if self.focus == FocusPanel::Input {
            "_"
        } else {
            ""
        };
        let input_text = format!("{}{}", self.input, cursor);
        frame.render_widget(Paragraph::new(input_text).block(input_block), area);
    }

    /// Draw the output panel.
    fn draw_output(&self, frame: &mut ratatui::Frame, area: Rect) {
        let output_lines: Vec<Line> = self
            .output
            .iter()
            .map(|s| Line::from(Span::raw(s.as_str())))
            .collect();
        let output_block = Block::default().borders(Borders::ALL).title(" Output ");
        frame.render_widget(Paragraph::new(output_lines).block(output_block), area);
    }

    /// Get border style based on focus.
    fn border_style(&self, panel: FocusPanel) -> Style {
        if self.focus == panel {
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::DarkGray)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_app_creation() {
        let app = App::new(true);
        assert!(!app.should_quit);
        assert_eq!(app.focus, FocusPanel::Input);
        assert!(app._dry_run);
        assert!(app.daemon_snapshot.is_none());
        assert_eq!(app.complexity, ComplexityLevel::Standard);
    }

    #[test]
    fn test_tab_cycling() {
        let mut app = App::new(true);
        assert_eq!(app.focus, FocusPanel::Input);
        app.handle_key(KeyCode::Tab);
        assert_eq!(app.focus, FocusPanel::Consciousness);
        app.handle_key(KeyCode::Tab);
        assert_eq!(app.focus, FocusPanel::Health);
        app.handle_key(KeyCode::Tab);
        assert_eq!(app.focus, FocusPanel::Generations);
        app.handle_key(KeyCode::Tab);
        assert_eq!(app.focus, FocusPanel::WorldModel);
        app.handle_key(KeyCode::Tab);
        assert_eq!(app.focus, FocusPanel::CausalGraph);
        app.handle_key(KeyCode::Tab);
        assert_eq!(app.focus, FocusPanel::Alerts);
        app.handle_key(KeyCode::Tab);
        assert_eq!(app.focus, FocusPanel::Input);
    }

    #[test]
    fn test_input_handling() {
        let mut app = App::new(true);
        app.handle_key(KeyCode::Char('h'));
        app.handle_key(KeyCode::Char('i'));
        assert_eq!(app.input, "hi");
        app.handle_key(KeyCode::Backspace);
        assert_eq!(app.input, "h");
    }

    #[test]
    fn test_quit_on_esc() {
        let mut app = App::new(true);
        app.focus = FocusPanel::Consciousness;
        app.handle_key(KeyCode::Esc);
        assert!(app.should_quit);
    }

    #[test]
    fn test_esc_clears_input_first() {
        let mut app = App::new(true);
        app.input = "hello".into();
        app.handle_key(KeyCode::Esc);
        assert!(!app.should_quit);
        assert!(app.input.is_empty());
    }

    #[test]
    fn test_process_input() {
        let mut app = App::new(true);
        app.input = "install firefox".into();
        app.process_input();
        assert!(app.input.is_empty());
        assert!(!app.output.is_empty());
    }

    #[test]
    fn test_q_doesnt_quit_in_input_mode() {
        let mut app = App::new(true);
        assert_eq!(app.focus, FocusPanel::Input);
        app.handle_key(KeyCode::Char('q'));
        assert!(!app.should_quit);
        assert_eq!(app.input, "q");
    }

    #[test]
    fn test_q_quits_in_other_modes() {
        let mut app = App::new(true);
        app.focus = FocusPanel::Health;
        app.handle_key(KeyCode::Char('q'));
        assert!(app.should_quit);
    }

    #[test]
    fn test_complexity_cycling() {
        let mut app = App::new(true);
        app.focus = FocusPanel::Health; // Not input, so 'l' triggers complexity
        assert_eq!(app.complexity, ComplexityLevel::Standard);
        app.handle_key(KeyCode::Char('l'));
        assert_eq!(app.complexity, ComplexityLevel::Expert);
        app.handle_key(KeyCode::Char('l'));
        assert_eq!(app.complexity, ComplexityLevel::Zen);
        app.handle_key(KeyCode::Char('l'));
        assert_eq!(app.complexity, ComplexityLevel::Beginner);
        app.handle_key(KeyCode::Char('l'));
        assert_eq!(app.complexity, ComplexityLevel::Standard);
    }

    #[test]
    fn test_zen_toggle() {
        let mut app = App::new(true);
        app.focus = FocusPanel::Health;
        assert_eq!(app.complexity, ComplexityLevel::Standard);
        app.handle_key(KeyCode::Char('z'));
        assert_eq!(app.complexity, ComplexityLevel::Zen);
        app.handle_key(KeyCode::Char('z'));
        assert_eq!(app.complexity, ComplexityLevel::Standard);
    }

    #[test]
    fn test_auto_adjust_complexity() {
        let mut app = App::new(true);
        // Simulate 10 successful interactions
        for _ in 0..10 {
            app.success_count += 1;
            app.total_count += 1;
        }
        app.auto_adjust_complexity();
        assert_eq!(app.complexity, ComplexityLevel::Expert);

        // Now mostly failures
        app.success_count = 1;
        app.total_count = 10;
        app.auto_adjust_complexity();
        assert_eq!(app.complexity, ComplexityLevel::Beginner);

        // Zen mode should be preserved
        app.complexity = ComplexityLevel::Zen;
        app.success_count = 10;
        app.total_count = 10;
        app.auto_adjust_complexity();
        assert_eq!(app.complexity, ComplexityLevel::Zen);
    }

    #[test]
    fn test_apply_daemon_snapshot() {
        let mut app = App::new(true);
        let snap = DaemonSnapshot {
            version: ipc::SNAPSHOT_VERSION,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            observation_count: 100,
            anomaly_count: 5,
            hierarchy_errors: [0.1, 0.2, 0.15, 0.05],
            free_energy: 0.35,
            is_surprised: true,
            drift_similarity: 0.95,
            causal_edge_count: 200,
            episodic_count: 10,
            concerns: vec![ipc::ConcernEntry {
                label: "high memory".into(),
                activation: 0.9,
                source: "system".into(),
            }],
            recent_anomalies: vec![ipc::AnomalyEntry {
                score: 0.8,
                reason: "OOM killer".into(),
                unit: "kernel".into(),
                error_type: None,
                suggestion: None,
            }],
            daemon_running: true,
            daemon_pid: 1,
            support_status: Some("Healthy".into()),
            recommendation_count: 0,
            alerts: vec![],
            top_causal_edges: vec![],
            memory_used_percent: Some(55.0),
            watchdog_status: Some("stabilized".into()),
            degraded: false,
            prediction_accuracy: Some(3.5),
            maintenance_plan_count: 1,
            load_average_1m: Some(0.5),
            swap_used_percent: Some(12.0),
            anomaly_volatility_ema: 0.1,
            active_anomaly_threshold: 0.5,
            last_plan_efe: None,
        };

        app.apply_daemon_snapshot(&snap);

        assert!((app.consciousness.free_energy - 0.35).abs() < 1e-6);
        assert!((app.world_model.free_energy - 0.35).abs() < 1e-6);
        assert_eq!(app.world_model.total_observations, 100);
        assert!(app.world_model.is_surprised);
        assert_eq!(app.world_model.memory_items.len(), 1);
        assert_eq!(app.world_model.memory_items[0].0, "high memory");
        assert_eq!(app.causal_links.len(), 1);
        assert_eq!(app.causal_links[0].from, "kernel");
        // Consciousness history should track
        assert_eq!(app.consciousness_history.len(), 1);
        assert!((app.consciousness_history[0].free_energy - 0.35).abs() < 1e-6);
        // Memory history should track
        assert_eq!(app.memory_history.len(), 1);
        assert!((app.memory_history[0] - 55.0).abs() < 1e-6);
    }

    #[test]
    fn test_apply_daemon_snapshot_alerts_populated() {
        let mut app = App::new(true);
        let snap = DaemonSnapshot {
            version: ipc::SNAPSHOT_VERSION,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            alerts: vec![ipc::AlertEntry {
                metric: "disk_used_pct".into(),
                current_value: 85.0,
                predicted_value: 95.0,
                hours_ahead: 6.0,
                threshold: 90.0,
                confidence: 0.8,
                recommended_action: Some("nix-collect-garbage -d".into()),
                severity: ipc::AlertSeverity::Warning,
                first_seen: 1700000000,
                last_seen: 1700000100,
                consecutive_cycles: 3,
                prev_predicted_value: Some(93.0),
                journal_context: vec!["disk nearly full".into()],
            }],
            support_status: Some("Warning".into()),
            recommendation_count: 2,
            watchdog_status: Some("monitoring".into()),
            prediction_accuracy: Some(4.5),
            ..DaemonSnapshot::test_default()
        };

        app.apply_daemon_snapshot(&snap);

        // Alerts should be populated from snapshot
        assert_eq!(app.alerts.active_alerts.len(), 1);
        assert_eq!(app.alerts.active_alerts[0].metric, "disk_used_pct");
        assert_eq!(app.alerts.support_status.as_deref(), Some("Warning"));
        assert_eq!(app.alerts.recommendation_count, 2);
        assert_eq!(app.alerts.watchdog_status.as_deref(), Some("monitoring"));
        assert_eq!(app.alerts.prediction_accuracy, Some(4.5));
    }

    #[test]
    fn test_apply_daemon_snapshot_mae_history_tracks() {
        let mut app = App::new(true);

        // Apply 3 snapshots with different MAE values
        for mae in [5.0, 4.0, 3.0] {
            let snap = DaemonSnapshot {
                prediction_accuracy: Some(mae),
                ..DaemonSnapshot::test_default()
            };
            app.apply_daemon_snapshot(&snap);
        }

        assert_eq!(app.mae_history.len(), 3);
        assert!((app.mae_history[0] - 5.0).abs() < 1e-6);
        assert!((app.mae_history[2] - 3.0).abs() < 1e-6);
        assert_eq!(app.alerts.prediction_mae_history.len(), 3);
    }

    #[test]
    fn test_apply_daemon_snapshot_mae_history_bounded() {
        let mut app = App::new(true);

        // Apply 25 snapshots — history should be capped at 20
        for i in 0..25 {
            let snap = DaemonSnapshot {
                prediction_accuracy: Some(i as f64),
                ..DaemonSnapshot::test_default()
            };
            app.apply_daemon_snapshot(&snap);
        }

        assert_eq!(app.mae_history.len(), 20);
        // Oldest entries should have been dropped
        assert!((app.mae_history[0] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_apply_daemon_snapshot_causal_edges_and_anomalies() {
        let mut app = App::new(true);
        let snap = DaemonSnapshot {
            top_causal_edges: vec![ipc::CausalEdgeEntry {
                from: "services.nginx".into(),
                to: "networking.firewall".into(),
                confidence: 0.85,
            }],
            recent_anomalies: vec![ipc::AnomalyEntry {
                score: 0.7,
                reason: "Service crashed".into(),
                unit: "myapp.service".into(),
                error_type: Some("crash".into()),
                suggestion: Some("Restart myapp".into()),
            }],
            ..DaemonSnapshot::test_default()
        };

        app.apply_daemon_snapshot(&snap);

        // Should have causal edge + anomaly appended
        assert_eq!(app.causal_links.len(), 2);
        assert_eq!(app.causal_links[0].from, "services.nginx");
        assert_eq!(app.causal_links[0].relationship, "causal");
        assert_eq!(app.causal_links[1].from, "myapp.service");
        assert_eq!(app.causal_links[1].relationship, "anomaly");
        assert_eq!(
            app.causal_links[1].suggestion.as_deref(),
            Some("Restart myapp")
        );
    }

    #[test]
    fn test_apply_daemon_snapshot_health_fields() {
        let mut app = App::new(true);
        let snap = DaemonSnapshot {
            anomaly_count: 3,
            memory_used_percent: Some(72.5),
            load_average_1m: Some(2.1),
            swap_used_percent: Some(15.0),
            ..DaemonSnapshot::test_default()
        };

        app.apply_daemon_snapshot(&snap);

        assert_eq!(app.health.services_failed, 3);
        assert_eq!(app.health.memory_used_percent, Some(72.5));
        assert_eq!(app.health.load_average_1m, Some(2.1));
        assert_eq!(app.health.swap_used_percent, Some(15.0));
    }

    #[test]
    fn test_apply_daemon_snapshot_fe_history_tracks() {
        let mut app = App::new(true);

        for fe in [0.1, 0.3, 0.5] {
            let snap = DaemonSnapshot {
                free_energy: fe,
                ..DaemonSnapshot::test_default()
            };
            app.apply_daemon_snapshot(&snap);
        }

        assert_eq!(app.fe_history.len(), 3);
        assert!((app.fe_history[2] - 0.5).abs() < 1e-6);
    }
}
