// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! World Model View Widget
//!
//! Visualizes the internal world model state: prediction errors across
//! hierarchy levels, learned action deltas, and working memory items.

use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Widget},
};

/// World model state for display.
#[derive(Debug, Clone)]
pub struct WorldModelSnapshot {
    /// Prediction errors at each hierarchy level.
    pub level_errors: [f64; 4],
    /// Level names.
    pub level_names: [&'static str; 4],
    /// Total free energy.
    pub free_energy: f64,
    /// Number of learned action categories.
    pub learned_actions: usize,
    /// Total episodic observations.
    pub total_observations: usize,
    /// Working memory items (label, activation).
    pub memory_items: Vec<(String, f64)>,
    /// Whether the system is surprised.
    pub is_surprised: bool,
    /// HDC world model drift similarity (1.0 = stable).
    pub drift_similarity: f32,
    /// Number of episodes in episodic memory.
    pub episodic_count: usize,
    /// Rolling MAE of predictions (lower = more accurate).
    pub prediction_accuracy: Option<f64>,
    /// Number of active inference maintenance plans (dry-run).
    pub maintenance_plan_count: u32,
    /// Volatility EMA
    pub anomaly_volatility_ema: f64,
    /// Active anomaly threshold
    pub active_anomaly_threshold: f64,
    /// Whether the daemon is hibernating
    pub hibernating: bool,
    /// Risk aversion weight
    pub risk_aversion: f64,
    /// Curiosity weight
    pub curiosity_weight: f64,
    /// Causal learning rate
    pub causal_learning_rate: f64,
}

impl Default for WorldModelSnapshot {
    fn default() -> Self {
        Self {
            level_errors: [0.0; 4],
            level_names: ["Sensory", "Features", "Concepts", "Goals"],
            free_energy: 0.0,
            learned_actions: 0,
            total_observations: 0,
            memory_items: Vec::new(),
            is_surprised: false,
            drift_similarity: 1.0,
            episodic_count: 0,
            prediction_accuracy: None,
            maintenance_plan_count: 0,
            anomaly_volatility_ema: 0.0,
            active_anomaly_threshold: 0.7,
            hibernating: false,
            risk_aversion: 0.2,
            curiosity_weight: 0.3,
            causal_learning_rate: 0.1,
        }
    }
}

/// World model view widget.
pub struct WorldModelView<'a> {
    snapshot: WorldModelSnapshot,
    block: Option<Block<'a>>,
}

impl<'a> WorldModelView<'a> {
    pub fn new(snapshot: WorldModelSnapshot) -> Self {
        Self {
            snapshot,
            block: None,
        }
    }

    pub fn block(mut self, block: Block<'a>) -> Self {
        self.block = Some(block);
        self
    }

    fn error_color(error: f64) -> Color {
        if error > 0.7 {
            Color::Red
        } else if error > 0.3 {
            Color::Yellow
        } else {
            Color::Green
        }
    }

    fn error_bar(error: f64, width: usize) -> String {
        let filled = ((error * width as f64) as usize).min(width);
        let empty = width.saturating_sub(filled);
        format!("{}{}", "#".repeat(filled), ".".repeat(empty))
    }
}

impl Widget for WorldModelView<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let block = self.block.unwrap_or_else(|| {
            Block::default()
                .title(" World Model ")
                .borders(Borders::ALL)
        });
        let inner = block.inner(area);
        block.render(area, buf);

        if inner.height < 4 || inner.width < 25 {
            return;
        }

        let mut y = inner.y;
        let x = inner.x + 1;
        let bar_width = (inner.width as usize).saturating_sub(18).min(20);

        // Prediction hierarchy
        for i in 0..4 {
            if y >= inner.y + inner.height {
                break;
            }
            let err = self.snapshot.level_errors[i];
            let color = Self::error_color(err);
            let bar = Self::error_bar(err, bar_width);
            let line = Line::from(vec![
                Span::styled(
                    format!("{:<8}", self.snapshot.level_names[i]),
                    Style::default().fg(Color::White),
                ),
                Span::raw(" ["),
                Span::styled(bar, Style::default().fg(color)),
                Span::raw("] "),
                Span::styled(format!("{:.2}", err), Style::default().fg(color)),
            ]);
            buf.set_line(x, y, &line, inner.width.saturating_sub(1));
            y += 1;
        }

        // Blank line
        y += 1;

        // Stats
        if y < inner.y + inner.height {
            let surprise_indicator = if self.snapshot.is_surprised { "!" } else { "" };
            let fe_color = if self.snapshot.free_energy > 0.5 {
                Color::Red
            } else {
                Color::Green
            };
            let stats = Line::from(vec![
                Span::raw("FE: "),
                Span::styled(
                    format!("{:.3}{}", self.snapshot.free_energy, surprise_indicator),
                    Style::default()
                        .fg(fe_color)
                        .add_modifier(if self.snapshot.is_surprised {
                            Modifier::BOLD
                        } else {
                            Modifier::empty()
                        }),
                ),
                Span::raw(format!(
                    "  Actions: {}  Obs: {}",
                    self.snapshot.learned_actions, self.snapshot.total_observations,
                )),
            ]);
            buf.set_line(x, y, &stats, inner.width.saturating_sub(1));
            y += 1;
        }

        // Drift + episodic line
        if y < inner.y + inner.height {
            let drift = self.snapshot.drift_similarity;
            let drift_color = if drift > 0.9 {
                Color::Green
            } else if drift > 0.7 {
                Color::Yellow
            } else {
                Color::Red
            };
            let drift_line = Line::from(vec![
                Span::raw("Drift: "),
                Span::styled(format!("{:.2}", drift), Style::default().fg(drift_color)),
                Span::raw(format!("  Episodes: {}", self.snapshot.episodic_count)),
            ]);
            buf.set_line(x, y, &drift_line, inner.width.saturating_sub(1));
            y += 1;
        }

        // Prediction accuracy + maintenance plans
        if y < inner.y + inner.height {
            let mut spans = Vec::new();
            if let Some(mae) = self.snapshot.prediction_accuracy {
                let mae_color = if mae > 10.0 {
                    Color::Red
                } else if mae > 5.0 {
                    Color::Yellow
                } else {
                    Color::Green
                };
                spans.push(Span::raw("MAE: "));
                spans.push(Span::styled(
                    format!("{:.1}", mae),
                    Style::default().fg(mae_color),
                ));
            }
            if self.snapshot.maintenance_plan_count > 0 {
                if !spans.is_empty() {
                    spans.push(Span::raw("  "));
                }
                spans.push(Span::raw(format!(
                    "Plans: {}",
                    self.snapshot.maintenance_plan_count
                )));
            }
            if !spans.is_empty() {
                buf.set_line(x, y, &Line::from(spans), inner.width.saturating_sub(1));
                y += 1;
            }
        }

        // Autonomic and Mood state
        if y < inner.y + inner.height {
            let sleep_str = if self.snapshot.hibernating {
                "Deep Sleep (Hibernating)"
            } else {
                "Active / Alert"
            };
            let sleep_color = if self.snapshot.hibernating {
                Color::Blue
            } else {
                Color::Green
            };
            let sleep_line = Line::from(vec![
                Span::raw("Sleep state: "),
                Span::styled(sleep_str, Style::default().fg(sleep_color)),
            ]);
            buf.set_line(x, y, &sleep_line, inner.width.saturating_sub(1));
            y += 1;
        }

        if y < inner.y + inner.height {
            let threshold_line = Line::from(vec![
                Span::raw("Vigilance:   "),
                Span::styled(
                    format!("{:.2}", self.snapshot.active_anomaly_threshold),
                    Style::default().fg(Color::Yellow),
                ),
                Span::raw(format!(
                    " (volatility: {:.2})",
                    self.snapshot.anomaly_volatility_ema
                )),
            ]);
            buf.set_line(x, y, &threshold_line, inner.width.saturating_sub(1));
            y += 1;
        }

        if y < inner.y + inner.height {
            let cognitive_line = Line::from(vec![
                Span::raw("Cognition:   Risk: "),
                Span::styled(
                    format!("{:.2}", self.snapshot.risk_aversion),
                    Style::default().fg(Color::Magenta),
                ),
                Span::raw("  Curiosity: "),
                Span::styled(
                    format!("{:.2}", self.snapshot.curiosity_weight),
                    Style::default().fg(Color::Cyan),
                ),
                Span::raw("  Learn: "),
                Span::styled(
                    format!("{:.2}", self.snapshot.causal_learning_rate),
                    Style::default().fg(Color::Green),
                ),
            ]);
            buf.set_line(x, y, &cognitive_line, inner.width.saturating_sub(1));
            y += 1;
        }

        // Working memory
        if y < inner.y + inner.height && !self.snapshot.memory_items.is_empty() {
            y += 1;
            if y < inner.y + inner.height {
                buf.set_string(
                    x,
                    y,
                    "Working Memory:",
                    Style::default()
                        .fg(Color::White)
                        .add_modifier(Modifier::BOLD),
                );
                y += 1;
            }

            for (label, activation) in &self.snapshot.memory_items {
                if y >= inner.y + inner.height {
                    break;
                }
                let act_color = if *activation > 0.7 {
                    Color::White
                } else {
                    Color::DarkGray
                };
                let truncated = if label.len() > (inner.width as usize).saturating_sub(10) {
                    &label[..(inner.width as usize).saturating_sub(13)]
                } else {
                    label.as_str()
                };
                let line = Line::from(vec![
                    Span::styled(
                        format!("[{:.1}]", activation),
                        Style::default().fg(act_color),
                    ),
                    Span::raw(" "),
                    Span::styled(truncated, Style::default().fg(act_color)),
                ]);
                buf.set_line(x, y, &line, inner.width.saturating_sub(1));
                y += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_render_with_data() {
        let snap = WorldModelSnapshot {
            level_errors: [0.1, 0.3, 0.6, 0.8],
            free_energy: 0.45,
            learned_actions: 5,
            total_observations: 42,
            memory_items: vec![
                ("install firefox".into(), 0.9),
                ("system state".into(), 0.5),
            ],
            is_surprised: false,
            ..Default::default()
        };
        let widget =
            WorldModelView::new(snap).block(Block::default().title("Model").borders(Borders::ALL));
        let area = Rect::new(0, 0, 50, 15);
        let mut buf = Buffer::empty(area);
        widget.render(area, &mut buf);
    }

    #[test]
    fn test_render_surprised() {
        let snap = WorldModelSnapshot {
            level_errors: [0.9, 0.8, 0.7, 0.9],
            free_energy: 0.85,
            is_surprised: true,
            ..Default::default()
        };
        let widget = WorldModelView::new(snap);
        let area = Rect::new(0, 0, 45, 12);
        let mut buf = Buffer::empty(area);
        widget.render(area, &mut buf);
    }

    #[test]
    fn test_error_bar() {
        assert_eq!(WorldModelView::error_bar(0.5, 10), "#####.....");
        assert_eq!(WorldModelView::error_bar(0.0, 10), "..........");
        assert_eq!(WorldModelView::error_bar(1.0, 10), "##########");
    }
}
