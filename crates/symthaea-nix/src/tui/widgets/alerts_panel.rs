// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Alerts Panel Widget
//!
//! Displays active predictive alerts (threshold-crossing predictions),
//! support assessment status, and watchdog status from the daemon.

use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Widget},
};

use crate::ipc::{AlertEntry, AlertSeverity};

/// Snapshot of alert/support state for display.
#[derive(Debug, Clone, Default)]
pub struct AlertsSnapshot {
    pub active_alerts: Vec<AlertEntry>,
    pub support_status: Option<String>,
    pub recommendation_count: usize,
    pub watchdog_status: Option<String>,
    pub matched_knowledge_ids: Vec<String>,
    /// Recent telemetry for sparklines: [disk_pct, memory_pct, store_paths_k, failed_units].
    pub recent_telemetry: Vec<[f64; 4]>,
    /// Currently selected alert index (for keyboard navigation).
    pub selected_index: usize,
    /// Whether the alerts panel is focused.
    pub is_focused: bool,
    /// Acknowledged alert metrics (suppressed from display).
    pub acknowledged: std::collections::HashSet<String>,
    /// Overall prediction mean absolute error (lower is better).
    pub prediction_accuracy: Option<f64>,
    /// MAE history for trend tracking (last N samples).
    pub prediction_mae_history: Vec<f64>,
}

/// Alerts dashboard widget.
pub struct AlertsPanel<'a> {
    snapshot: AlertsSnapshot,
    block: Option<Block<'a>>,
}

impl<'a> AlertsPanel<'a> {
    pub fn new(snapshot: AlertsSnapshot) -> Self {
        Self {
            snapshot,
            block: None,
        }
    }

    pub fn block(mut self, block: Block<'a>) -> Self {
        self.block = Some(block);
        self
    }
}

impl Widget for AlertsPanel<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let block = self
            .block
            .unwrap_or_else(|| Block::default().title(" Alerts ").borders(Borders::ALL));
        let inner = block.inner(area);
        block.render(area, buf);

        if inner.height < 2 || inner.width < 15 {
            return;
        }

        let mut y = inner.y;
        let x = inner.x + 1;
        let w = inner.width.saturating_sub(1);

        // Status line: support assessment + watchdog
        let status_color = match self.snapshot.support_status.as_deref() {
            Some(s) if s.contains("Critical") => Color::Red,
            Some(s) if s.contains("Warning") || s.contains("Degraded") => Color::Yellow,
            Some(_) => Color::Green,
            None => Color::DarkGray,
        };
        let status_text = self
            .snapshot
            .support_status
            .as_deref()
            .unwrap_or("no assessment");
        let mut status_spans = vec![
            Span::raw("Status: "),
            Span::styled(status_text, Style::default().fg(status_color)),
        ];
        if self.snapshot.recommendation_count > 0 {
            status_spans.push(Span::styled(
                format!(" ({} recs)", self.snapshot.recommendation_count),
                Style::default().fg(Color::Yellow),
            ));
        }
        buf.set_line(x, y, &Line::from(status_spans), w);
        y += 1;

        // Watchdog status
        if y < inner.y + inner.height {
            if let Some(ref ws) = self.snapshot.watchdog_status {
                let wd_color = match ws.as_str() {
                    "monitoring" => Color::Cyan,
                    "stabilized" => Color::Green,
                    "degraded" | "reverted" | "error" => Color::Red,
                    _ => Color::DarkGray,
                };
                let wd_line = Line::from(vec![
                    Span::raw("Watchdog: "),
                    Span::styled(ws.as_str(), Style::default().fg(wd_color)),
                ]);
                buf.set_line(x, y, &wd_line, w);
                y += 1;
            }
        }

        // Prediction accuracy (MAE) with trend arrow (BF)
        if y < inner.y + inner.height {
            if let Some(mae) = self.snapshot.prediction_accuracy {
                let acc_color = if mae < 3.0 {
                    Color::Green
                } else if mae < 8.0 {
                    Color::Yellow
                } else {
                    Color::Red
                };

                // Compute trend from MAE history
                let (trend_arrow, trend_color) = mae_trend(&self.snapshot.prediction_mae_history);

                let acc_line = Line::from(vec![
                    Span::raw("Pred MAE: "),
                    Span::styled(format!("{:.1}", mae), Style::default().fg(acc_color)),
                    Span::styled(
                        format!(" {}", trend_arrow),
                        Style::default().fg(trend_color),
                    ),
                    Span::styled(
                        if mae < 3.0 {
                            " (excellent)"
                        } else if mae < 8.0 {
                            " (good)"
                        } else {
                            " (needs calibration)"
                        },
                        Style::default().fg(Color::DarkGray),
                    ),
                ]);
                buf.set_line(x, y, &acc_line, w);
                y += 1;
            }
        }

        // Filter acknowledged alerts
        let visible_alerts: Vec<&AlertEntry> = self
            .snapshot
            .active_alerts
            .iter()
            .filter(|a| !self.snapshot.acknowledged.contains(&a.metric))
            .collect();

        let ack_count = self.snapshot.active_alerts.len() - visible_alerts.len();

        if visible_alerts.is_empty() {
            if y < inner.y + inner.height {
                let msg = if ack_count > 0 {
                    format!("No active alerts ({} acknowledged)", ack_count)
                } else {
                    "No active alerts".to_string()
                };
                let line = Line::from(Span::styled(msg, Style::default().fg(Color::DarkGray)));
                buf.set_line(x, y, &line, w);
            }
            return;
        }

        // Help hint when focused
        if self.snapshot.is_focused && y < inner.y + inner.height {
            let hint = Line::from(Span::styled(
                "[Up/Down] select  [Enter] copy action  [a] ack  [u] un-ack",
                Style::default().fg(Color::DarkGray),
            ));
            buf.set_line(x, y, &hint, w);
            y += 1;
        }

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let selected = self.snapshot.selected_index;

        for (idx, alert) in visible_alerts.iter().enumerate() {
            if y >= inner.y + inner.height {
                break;
            }

            let is_selected = self.snapshot.is_focused && idx == selected;

            let alert_color = match alert.severity {
                AlertSeverity::Critical => Color::Red,
                AlertSeverity::Warning => Color::Yellow,
                AlertSeverity::Info => Color::White,
            };

            // Trend direction: compare current predicted_value with previous cycle
            let trend_arrow = match alert.prev_predicted_value {
                Some(prev) if alert.predicted_value > prev + 0.5 => "^",
                Some(prev) if alert.predicted_value < prev - 0.5 => "v",
                _ => "=",
            };
            let trend_color = match trend_arrow {
                "^" => Color::Red,
                "v" => Color::Green,
                _ => Color::DarkGray,
            };

            // Age badge
            let age_secs = now.saturating_sub(alert.first_seen);
            let age_str = if age_secs < 60 {
                format!("{}s", age_secs)
            } else if age_secs < 3600 {
                format!("{}m", age_secs / 60)
            } else {
                format!("{}h", age_secs / 3600)
            };

            // Selection cursor
            let cursor = if is_selected { "> " } else { "  " };
            let cursor_style = if is_selected {
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default()
            };

            let line = Line::from(vec![
                Span::styled(cursor, cursor_style),
                Span::styled(
                    format!("{}: ", alert.metric),
                    Style::default()
                        .fg(alert_color)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::raw(format!(
                    "{:.0} -> {:.0} in {:.0}h",
                    alert.current_value, alert.predicted_value, alert.hours_ahead
                )),
                Span::styled(
                    format!(" {}", trend_arrow),
                    Style::default().fg(trend_color),
                ),
                Span::styled(
                    format!(" [{:.0}% {}]", alert.confidence * 100.0, age_str),
                    Style::default().fg(alert_color),
                ),
            ]);
            buf.set_line(x, y, &line, w);
            y += 1;

            // Show recommended action if present and space allows
            if y < inner.y + inner.height {
                if let Some(ref action) = alert.recommended_action {
                    let action_line = Line::from(Span::styled(
                        format!("    -> {}", action),
                        Style::default().fg(Color::Cyan),
                    ));
                    buf.set_line(x, y, &action_line, w);
                    y += 1;
                }
            }

            // Show journal context if corroborating anomalies exist
            for ctx in &alert.journal_context {
                if y >= inner.y + inner.height {
                    break;
                }
                let ctx_line = Line::from(Span::styled(
                    format!("    ! {}", ctx),
                    Style::default().fg(Color::Magenta),
                ));
                buf.set_line(x, y, &ctx_line, w);
                y += 1;
            }
        }

        // Show acknowledged count footer
        if ack_count > 0 && y < inner.y + inner.height {
            let footer = Line::from(Span::styled(
                format!("({} acknowledged, press 'u' to show)", ack_count),
                Style::default().fg(Color::DarkGray),
            ));
            buf.set_line(x, y, &footer, w);
        }

        // Sparklines: show recent metric trends if data is available
        if !self.snapshot.recent_telemetry.is_empty() && y + 1 < inner.y + inner.height {
            y += 1; // blank separator line

            let metrics = [
                ("disk", 0, 100.0, Color::Magenta),
                ("mem", 1, 100.0, Color::Blue),
                ("store", 2, 200.0, Color::Yellow), // in thousands
                ("fail", 3, 10.0, Color::Red),
            ];

            for (label, idx, max_val, color) in &metrics {
                if y >= inner.y + inner.height {
                    break;
                }
                let values: Vec<f64> = self
                    .snapshot
                    .recent_telemetry
                    .iter()
                    .map(|t| t[*idx])
                    .collect();
                let sparkline = render_sparkline(&values, *max_val);
                let line = Line::from(vec![
                    Span::styled(
                        format!("{:>5} ", label),
                        Style::default().fg(Color::DarkGray),
                    ),
                    Span::styled(sparkline, Style::default().fg(*color)),
                ]);
                buf.set_line(x, y, &line, w);
                y += 1;
            }
        }
    }
}

/// Compute MAE trend arrow from history.
///
/// Returns (arrow_str, color): rising MAE (worse) = red "^", falling (better) = green "v",
/// stable = gray "=".
fn mae_trend(history: &[f64]) -> (&'static str, Color) {
    if history.len() < 3 {
        return ("=", Color::DarkGray);
    }
    // Compare average of last 3 vs previous 3 (or all earlier)
    let n = history.len();
    let recent_start = n.saturating_sub(3);
    let prev_end = recent_start;
    let prev_start = prev_end.saturating_sub(3);

    if prev_start >= prev_end {
        return ("=", Color::DarkGray);
    }

    let recent_avg: f64 = history[recent_start..].iter().sum::<f64>() / (n - recent_start) as f64;
    let prev_avg: f64 =
        history[prev_start..prev_end].iter().sum::<f64>() / (prev_end - prev_start) as f64;

    let delta = recent_avg - prev_avg;
    if delta > 0.5 {
        ("^", Color::Red) // MAE rising = getting worse
    } else if delta < -0.5 {
        ("v", Color::Green) // MAE falling = improving
    } else {
        ("=", Color::DarkGray) // stable
    }
}

/// Render a sparkline string from a series of values using Unicode block chars.
fn render_sparkline(values: &[f64], max_val: f64) -> String {
    const BLOCKS: &[char] = &['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

    if values.is_empty() {
        return String::new();
    }

    values
        .iter()
        .map(|&v| {
            let normalized = (v / max_val).clamp(0.0, 1.0);
            let idx = (normalized * (BLOCKS.len() - 1) as f64).round() as usize;
            BLOCKS[idx.min(BLOCKS.len() - 1)]
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_render_empty_doesnt_panic() {
        let widget = AlertsPanel::new(AlertsSnapshot::default())
            .block(Block::default().title("Alerts").borders(Borders::ALL));
        let area = Rect::new(0, 0, 50, 10);
        let mut buf = Buffer::empty(area);
        widget.render(area, &mut buf);
    }

    #[test]
    fn test_render_with_alerts_doesnt_panic() {
        let snap = AlertsSnapshot {
            active_alerts: vec![AlertEntry {
                metric: "disk_usage".into(),
                current_value: 85.0,
                predicted_value: 100.0,
                hours_ahead: 6.0,
                threshold: 90.0,
                confidence: 0.8,
                recommended_action: Some("nix-collect-garbage -d".into()),
                severity: AlertSeverity::Warning,
                first_seen: 1700000000,
                last_seen: 1700000100,
                consecutive_cycles: 3,
                prev_predicted_value: Some(98.0),
                journal_context: vec![],
            }],
            support_status: Some("Warning".into()),
            recommendation_count: 2,
            watchdog_status: Some("monitoring".into()),
            matched_knowledge_ids: vec!["disk-001".into()],
            ..AlertsSnapshot::default()
        };
        let widget =
            AlertsPanel::new(snap).block(Block::default().title("Alerts").borders(Borders::ALL));
        let area = Rect::new(0, 0, 60, 12);
        let mut buf = Buffer::empty(area);
        widget.render(area, &mut buf);
    }

    #[test]
    fn test_mae_trend_improving() {
        // MAE going down = improving
        let history = vec![8.0, 7.5, 7.0, 4.0, 3.5, 3.0];
        let (arrow, color) = super::mae_trend(&history);
        assert_eq!(arrow, "v");
        assert_eq!(color, Color::Green);
    }

    #[test]
    fn test_mae_trend_worsening() {
        // MAE going up = worsening
        let history = vec![2.0, 2.5, 3.0, 5.0, 6.0, 7.0];
        let (arrow, color) = super::mae_trend(&history);
        assert_eq!(arrow, "^");
        assert_eq!(color, Color::Red);
    }

    #[test]
    fn test_mae_trend_stable() {
        let history = vec![3.0, 3.1, 3.0, 3.1, 3.0, 3.1];
        let (arrow, _) = super::mae_trend(&history);
        assert_eq!(arrow, "=");
    }

    #[test]
    fn test_mae_trend_insufficient_data() {
        let (arrow, _) = super::mae_trend(&[1.0, 2.0]);
        assert_eq!(arrow, "=");
    }

    #[test]
    fn test_render_with_mae_history() {
        let snap = AlertsSnapshot {
            prediction_accuracy: Some(4.5),
            prediction_mae_history: vec![6.0, 5.5, 5.0, 4.5, 4.5, 4.5],
            ..AlertsSnapshot::default()
        };
        let widget =
            AlertsPanel::new(snap).block(Block::default().title("Alerts").borders(Borders::ALL));
        let area = Rect::new(0, 0, 60, 12);
        let mut buf = Buffer::empty(area);
        widget.render(area, &mut buf);
    }

    #[test]
    fn test_render_tiny_doesnt_panic() {
        let widget = AlertsPanel::new(AlertsSnapshot::default());
        let area = Rect::new(0, 0, 10, 1);
        let mut buf = Buffer::empty(area);
        widget.render(area, &mut buf);
    }
}
