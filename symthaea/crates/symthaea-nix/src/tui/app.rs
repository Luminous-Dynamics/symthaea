//! TUI Application State Machine
//!
//! Manages the terminal UI lifecycle: initialization, event loop,
//! rendering, and cleanup. Orchestrates the layout of all widgets
//! and handles keyboard input.

use std::io;
use std::time::{Duration, Instant};

use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    ExecutableCommand,
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
    Terminal,
};

use super::widgets::{
    CausalExplorer, CausalLink, ConsciousnessGauge, ConsciousnessState, GenerationTimeline,
    HealthSnapshot, SystemHealth, TimelineEntry, WorldModelSnapshot, WorldModelView,
};
use crate::ipc::{self, DaemonSnapshot};
use crate::mind::active_inference::NixActiveInference;

/// Which panel is focused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FocusPanel {
    Consciousness,
    Health,
    Generations,
    WorldModel,
    CausalGraph,
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
    /// Scroll offset for causal explorer.
    causal_scroll: usize,
    /// Whether the app should quit.
    should_quit: bool,
    /// Tick counter for status updates.
    tick: u64,
    /// Dry-run mode.
    _dry_run: bool,
    /// Path to daemon IPC snapshot.
    daemon_snapshot_path: std::path::PathBuf,
    /// Last daemon snapshot (if available).
    daemon_snapshot: Option<DaemonSnapshot>,
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
            causal_scroll: 0,
            should_quit: false,
            tick: 0,
            _dry_run: dry_run,
            daemon_snapshot_path: ipc::default_snapshot_path(),
            daemon_snapshot: None,
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
                // Periodic refresh every ~4 seconds (16 ticks)
                if self.tick % 16 == 0 {
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
                    FocusPanel::CausalGraph => FocusPanel::Input,
                };
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
        // Consciousness state from daemon's predictive hierarchy
        self.consciousness = ConsciousnessState {
            phi: snap.free_energy.min(1.0),
            confidence: 1.0 - snap.free_energy, // higher FE = lower confidence
            free_energy: snap.free_energy,
        };

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
            ..Default::default()
        };

        // Causal links — show recent anomalies as causal concerns
        self.causal_links = snap
            .recent_anomalies
            .iter()
            .map(|a| CausalLink {
                from: a.unit.clone(),
                to: a.reason.clone(),
                confidence: a.score,
                relationship: "anomaly".to_string(),
            })
            .collect();

        // Health: merge daemon stats with direct service queries
        self.health.services_failed = snap.anomaly_count as usize;
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
                })
                .collect();
        }
    }

    /// Draw the UI.
    fn draw(&self, frame: &mut ratatui::Frame) {
        let size = frame.area();

        // Main layout: top (panels) + bottom (input/output)
        let main_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(10), Constraint::Length(6)])
            .split(size);

        // Top: left column (consciousness + health) + right column (world model + causal)
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

        // Consciousness gauge
        let cons_block = Block::default()
            .title(" Consciousness ")
            .borders(Borders::ALL)
            .border_style(self.border_style(FocusPanel::Consciousness));
        let gauge = ConsciousnessGauge::new(self.consciousness).block(cons_block);
        frame.render_widget(gauge, left_chunks[0]);

        // System health
        let health_block = Block::default()
            .title(" System Health ")
            .borders(Borders::ALL)
            .border_style(self.border_style(FocusPanel::Health));
        let health = SystemHealth::new(self.health.clone()).block(health_block);
        frame.render_widget(health, left_chunks[1]);

        // Generations
        let gen_block = Block::default()
            .title(" Generations ")
            .borders(Borders::ALL)
            .border_style(self.border_style(FocusPanel::Generations));
        let timeline = GenerationTimeline::new(self.generations.clone()).block(gen_block);
        frame.render_widget(timeline, left_chunks[2]);

        // Right column
        let right_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(top_chunks[1]);

        // World model
        let model_block = Block::default()
            .title(" World Model ")
            .borders(Borders::ALL)
            .border_style(self.border_style(FocusPanel::WorldModel));
        let model = WorldModelView::new(self.world_model.clone()).block(model_block);
        frame.render_widget(model, right_chunks[0]);

        // Causal explorer
        let causal_block = Block::default()
            .title(" Causal Graph ")
            .borders(Borders::ALL)
            .border_style(self.border_style(FocusPanel::CausalGraph));
        let causal = CausalExplorer::new(self.causal_links.clone())
            .scroll(self.causal_scroll)
            .block(causal_block);
        frame.render_widget(causal, right_chunks[1]);

        // Bottom: input + output
        let bottom_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(3), Constraint::Min(1)])
            .split(main_chunks[1]);

        // Input — show daemon connection status
        let daemon_status = match &self.daemon_snapshot {
            Some(snap) => format!(
                " [daemon pid {} | {} obs | {} anomalies]",
                snap.daemon_pid, snap.observation_count, snap.anomaly_count
            ),
            None => " [daemon offline]".to_string(),
        };
        let input_block = Block::default()
            .title(format!(" Input (Tab/Esc){} ", daemon_status))
            .borders(Borders::ALL)
            .border_style(self.border_style(FocusPanel::Input));
        let cursor = if self.focus == FocusPanel::Input {
            "_"
        } else {
            ""
        };
        let input_text = format!("{}{}", self.input, cursor);
        let input_widget = Paragraph::new(input_text).block(input_block);
        frame.render_widget(input_widget, bottom_chunks[0]);

        // Output
        let output_lines: Vec<Line> = self
            .output
            .iter()
            .map(|s| Line::from(Span::raw(s.as_str())))
            .collect();
        let output_block = Block::default().borders(Borders::ALL).title(" Output ");
        let output_widget = Paragraph::new(output_lines).block(output_block);
        frame.render_widget(output_widget, bottom_chunks[1]);
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
    fn test_apply_daemon_snapshot() {
        let mut app = App::new(true);
        let snap = DaemonSnapshot {
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
            }],
            daemon_running: true,
            daemon_pid: 1,
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
    }
}
