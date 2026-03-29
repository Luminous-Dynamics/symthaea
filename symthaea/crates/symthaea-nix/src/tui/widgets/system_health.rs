// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! System Health Widget
//!
//! Aggregates service states, store usage, and generation info
//! into a dashboard panel.

use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Widget},
};

/// Snapshot of system health for display.
#[derive(Debug, Clone, Default)]
pub struct HealthSnapshot {
    pub services_running: usize,
    pub services_failed: usize,
    pub services_total: usize,
    pub store_size_human: String,
    pub store_paths: usize,
    pub current_generation: Option<u32>,
    pub total_generations: usize,
    pub memory_used_percent: Option<f64>,
    /// Memory usage history for sparkline (last 30 samples, 0-100).
    pub memory_history: Vec<f64>,
    /// CPU load average (1-minute).
    pub load_average_1m: Option<f64>,
    /// Swap usage percentage.
    pub swap_used_percent: Option<f64>,
}

/// System health dashboard widget.
pub struct SystemHealth<'a> {
    snapshot: HealthSnapshot,
    block: Option<Block<'a>>,
}

impl<'a> SystemHealth<'a> {
    pub fn new(snapshot: HealthSnapshot) -> Self {
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

impl SystemHealth<'_> {
    /// Render a sparkline from a sequence of values (0-100 range).
    fn render_sparkline(values: &[f64], width: usize) -> String {
        const BARS: &[char] = &['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
        if values.is_empty() || width == 0 {
            return String::new();
        }
        // Take the last `width` values (or all if fewer)
        let slice = if values.len() > width {
            &values[values.len() - width..]
        } else {
            values
        };
        let min = slice.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = slice.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = (max - min).max(1.0);
        slice
            .iter()
            .map(|v| {
                let idx = (((v - min) / range) * 7.0) as usize;
                BARS[idx.min(7)]
            })
            .collect()
    }
}

impl Widget for SystemHealth<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let block = self.block.unwrap_or_else(|| {
            Block::default()
                .title(" System Health ")
                .borders(Borders::ALL)
        });
        let inner = block.inner(area);
        block.render(area, buf);

        if inner.height < 3 || inner.width < 15 {
            return;
        }

        let mut y = inner.y;
        let x = inner.x + 1;

        // Services line
        let svc_color = if self.snapshot.services_failed > 0 {
            Color::Red
        } else {
            Color::Green
        };
        let svc_line = Line::from(vec![
            Span::raw("Services: "),
            Span::styled(
                format!("{} running", self.snapshot.services_running),
                Style::default().fg(Color::Green),
            ),
            Span::raw(", "),
            Span::styled(
                format!("{} failed", self.snapshot.services_failed),
                Style::default().fg(svc_color),
            ),
            Span::raw(format!(" / {}", self.snapshot.services_total)),
        ]);
        buf.set_line(x, y, &svc_line, inner.width.saturating_sub(1));
        y += 1;

        // Store line
        if y < inner.y + inner.height {
            let store_line = Line::from(vec![
                Span::raw("Store:    "),
                Span::styled(
                    &self.snapshot.store_size_human,
                    Style::default().fg(Color::Cyan),
                ),
                Span::raw(format!(" ({} paths)", self.snapshot.store_paths)),
            ]);
            buf.set_line(x, y, &store_line, inner.width.saturating_sub(1));
            y += 1;
        }

        // Generation line
        if y < inner.y + inner.height {
            let gen_str = match self.snapshot.current_generation {
                Some(g) => format!("#{}", g),
                None => "?".to_string(),
            };
            let gen_line = Line::from(vec![
                Span::raw("Gen:      "),
                Span::styled(
                    gen_str,
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::raw(format!(" ({} total)", self.snapshot.total_generations)),
            ]);
            buf.set_line(x, y, &gen_line, inner.width.saturating_sub(1));
            y += 1;
        }

        // Memory line (optional)
        if y < inner.y + inner.height {
            if let Some(pct) = self.snapshot.memory_used_percent {
                let mem_color = if pct > 90.0 {
                    Color::Red
                } else if pct > 70.0 {
                    Color::Yellow
                } else {
                    Color::Green
                };
                let mem_line = Line::from(vec![
                    Span::raw("Memory:   "),
                    Span::styled(format!("{:.0}%", pct), Style::default().fg(mem_color)),
                ]);
                buf.set_line(x, y, &mem_line, inner.width.saturating_sub(1));
                y += 1;
            }
        }

        // Memory sparkline (when history available)
        if y < inner.y + inner.height && self.snapshot.memory_history.len() >= 2 {
            let spark = Self::render_sparkline(
                &self.snapshot.memory_history,
                inner.width.saturating_sub(12) as usize,
            );
            let spark_line = Line::from(vec![
                Span::raw("Mem trend "),
                Span::styled(spark, Style::default().fg(Color::Cyan)),
            ]);
            buf.set_line(x, y, &spark_line, inner.width.saturating_sub(1));
            y += 1;
        }

        // Load average (optional)
        if y < inner.y + inner.height {
            if let Some(load) = self.snapshot.load_average_1m {
                let load_color = if load > 8.0 {
                    Color::Red
                } else if load > 4.0 {
                    Color::Yellow
                } else {
                    Color::Green
                };
                let load_line = Line::from(vec![
                    Span::raw("Load:     "),
                    Span::styled(format!("{:.2}", load), Style::default().fg(load_color)),
                ]);
                buf.set_line(x, y, &load_line, inner.width.saturating_sub(1));
                y += 1;
            }
        }

        // Swap usage (optional)
        if y < inner.y + inner.height {
            if let Some(swap) = self.snapshot.swap_used_percent {
                let swap_color = if swap > 80.0 {
                    Color::Red
                } else if swap > 50.0 {
                    Color::Yellow
                } else {
                    Color::Green
                };
                let swap_line = Line::from(vec![
                    Span::raw("Swap:     "),
                    Span::styled(format!("{:.0}%", swap), Style::default().fg(swap_color)),
                ]);
                buf.set_line(x, y, &swap_line, inner.width.saturating_sub(1));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_render_doesnt_panic() {
        let snap = HealthSnapshot {
            services_running: 42,
            services_failed: 1,
            services_total: 50,
            store_size_human: "15.3 GiB".into(),
            store_paths: 12345,
            current_generation: Some(45),
            total_generations: 10,
            memory_used_percent: Some(55.0),
            ..Default::default()
        };
        let widget =
            SystemHealth::new(snap).block(Block::default().title("Health").borders(Borders::ALL));
        let area = Rect::new(0, 0, 50, 10);
        let mut buf = Buffer::empty(area);
        widget.render(area, &mut buf);
    }

    #[test]
    fn test_render_tiny_doesnt_panic() {
        let widget = SystemHealth::new(HealthSnapshot::default());
        let area = Rect::new(0, 0, 10, 2);
        let mut buf = Buffer::empty(area);
        widget.render(area, &mut buf);
    }

    #[test]
    fn test_render_with_memory_sparkline() {
        let snap = HealthSnapshot {
            services_running: 42,
            services_failed: 0,
            services_total: 42,
            store_size_human: "10 GiB".into(),
            store_paths: 5000,
            current_generation: Some(10),
            total_generations: 10,
            memory_used_percent: Some(55.0),
            memory_history: vec![40.0, 45.0, 50.0, 55.0, 60.0, 55.0, 50.0, 48.0],
            load_average_1m: Some(1.2),
            swap_used_percent: Some(15.0),
        };
        let widget =
            SystemHealth::new(snap).block(Block::default().title("Health").borders(Borders::ALL));
        let area = Rect::new(0, 0, 50, 12);
        let mut buf = Buffer::empty(area);
        widget.render(area, &mut buf);
    }

    #[test]
    fn test_sparkline_rendering() {
        let values = vec![0.0, 25.0, 50.0, 75.0, 100.0];
        let spark = SystemHealth::render_sparkline(&values, 10);
        assert_eq!(spark.chars().count(), 5);
        // First should be lowest bar, last should be highest
        let chars: Vec<char> = spark.chars().collect();
        assert_eq!(chars[0], '▁');
        assert_eq!(chars[4], '█');
    }

    #[test]
    fn test_sparkline_empty() {
        assert!(SystemHealth::render_sparkline(&[], 10).is_empty());
        assert!(SystemHealth::render_sparkline(&[50.0], 0).is_empty());
    }

    #[test]
    fn test_sparkline_constant() {
        let values = vec![50.0, 50.0, 50.0];
        let spark = SystemHealth::render_sparkline(&values, 10);
        // All same value → all same bar
        let chars: Vec<char> = spark.chars().collect();
        assert!(chars.iter().all(|c| *c == chars[0]));
    }
}
