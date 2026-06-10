// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Generation Timeline Widget
//!
//! Displays NixOS generation history as a horizontal timeline with
//! the current generation highlighted.

use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Color, Modifier, Style},
    widgets::{Block, Borders, Widget},
};

/// A generation entry for the timeline.
#[derive(Debug, Clone)]
pub struct TimelineEntry {
    pub number: u32,
    pub date: String,
    pub current: bool,
    /// Size change from previous generation in bytes (positive = grew).
    pub size_delta_bytes: Option<i64>,
    /// Build status indicator (e.g. "ok", "failed", "building").
    pub status: Option<String>,
    /// Hours since this generation was built.
    pub age_hours: Option<f64>,
}

/// Generation timeline widget.
pub struct GenerationTimeline<'a> {
    entries: Vec<TimelineEntry>,
    block: Option<Block<'a>>,
}

impl<'a> GenerationTimeline<'a> {
    pub fn new(entries: Vec<TimelineEntry>) -> Self {
        Self {
            entries,
            block: None,
        }
    }

    pub fn block(mut self, block: Block<'a>) -> Self {
        self.block = Some(block);
        self
    }
}

impl Widget for GenerationTimeline<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let block = self.block.unwrap_or_else(|| {
            Block::default()
                .title(" Generations ")
                .borders(Borders::ALL)
        });
        let inner = block.inner(area);
        block.render(area, buf);

        if inner.height < 2 || inner.width < 10 || self.entries.is_empty() {
            return;
        }

        // Show most recent generations that fit
        let max_entries = (inner.width as usize) / 8; // ~8 chars per entry
        let visible: Vec<&TimelineEntry> = self.entries.iter().rev().take(max_entries).collect();
        let visible: Vec<&TimelineEntry> = visible.into_iter().rev().collect();

        // Draw timeline bar
        let bar_y = inner.y;
        let mut x = inner.x + 1;

        for entry in &visible {
            if x + 6 > inner.x + inner.width {
                break;
            }

            let status_indicator = match entry.status.as_deref() {
                Some("failed") => ("!", Color::Red),
                Some("building") => ("~", Color::Yellow),
                _ => {
                    if entry.current {
                        ("*", Color::Green)
                    } else {
                        (" ", Color::DarkGray)
                    }
                }
            };

            let style = Style::default()
                .fg(status_indicator.1)
                .add_modifier(if entry.current {
                    Modifier::BOLD
                } else {
                    Modifier::empty()
                });

            let marker = if entry.current { ">" } else { " " };
            let label = format!("{}{}#{}", marker, status_indicator.0, entry.number);
            buf.set_string(x, bar_y, &label, style);

            // Date + size delta below
            if inner.height > 1 {
                let date_short = if entry.date.len() > 5 {
                    &entry.date[5..entry.date.len().min(10)]
                } else {
                    &entry.date
                };
                let delta_str = match entry.size_delta_bytes {
                    Some(d) if d > 1_000_000 => format!(" +{}M", d / 1_000_000),
                    Some(d) if d < -1_000_000 => format!(" {}M", d / 1_000_000),
                    Some(d) if d != 0 => format!(" {}K", d / 1_000),
                    _ => String::new(),
                };
                let delta_color = match entry.size_delta_bytes {
                    Some(d) if d > 50_000_000 => Color::Red,
                    Some(d) if d < -10_000_000 => Color::Green,
                    _ => Color::DarkGray,
                };
                buf.set_string(
                    x,
                    bar_y + 1,
                    date_short,
                    Style::default().fg(Color::DarkGray),
                );
                if !delta_str.is_empty()
                    && x + date_short.len() as u16 + delta_str.len() as u16 + 1
                        < inner.x + inner.width
                {
                    buf.set_string(
                        x + date_short.len() as u16,
                        bar_y + 1,
                        &delta_str,
                        Style::default().fg(delta_color),
                    );
                }
            }

            // Arrow connector
            if x + 7 < inner.x + inner.width {
                buf.set_string(
                    x + label.len() as u16,
                    bar_y,
                    "->",
                    Style::default().fg(Color::DarkGray),
                );
            }

            x += 8;
        }

        // Summary at bottom
        if inner.height > 2 {
            let summary = format!("{} generations", self.entries.len());
            let sy = inner.y + inner.height.saturating_sub(1);
            buf.set_string(
                inner.x + 1,
                sy,
                &summary,
                Style::default().fg(Color::DarkGray),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_render_with_entries() {
        let entries = vec![
            TimelineEntry {
                number: 42,
                date: "2025-01-05".into(),
                current: false,
                size_delta_bytes: None,
                status: None,
                age_hours: Some(240.0),
            },
            TimelineEntry {
                number: 43,
                date: "2025-01-10".into(),
                current: false,
                size_delta_bytes: Some(50_000_000),
                status: None,
                age_hours: Some(120.0),
            },
            TimelineEntry {
                number: 44,
                date: "2025-01-15".into(),
                current: true,
                size_delta_bytes: Some(-5_000_000),
                status: Some("ok".into()),
                age_hours: Some(0.5),
            },
        ];
        let widget = GenerationTimeline::new(entries)
            .block(Block::default().title("Gens").borders(Borders::ALL));
        let area = Rect::new(0, 0, 60, 8);
        let mut buf = Buffer::empty(area);
        widget.render(area, &mut buf);
    }

    #[test]
    fn test_render_empty() {
        let widget = GenerationTimeline::new(vec![]);
        let area = Rect::new(0, 0, 40, 6);
        let mut buf = Buffer::empty(area);
        widget.render(area, &mut buf);
    }
}
