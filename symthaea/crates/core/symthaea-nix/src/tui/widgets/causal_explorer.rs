// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Causal Explorer Widget
//!
//! Displays the causal graph as a list of edges with confidence values.
//! Supports scrolling through the causal relationships discovered
//! between NixOS configuration options.

use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Widget},
};

/// A causal link for display.
#[derive(Debug, Clone)]
pub struct CausalLink {
    pub from: String,
    pub to: String,
    pub confidence: f64,
    pub relationship: String,
    /// Structured suggestion from anomaly diagnosis (if available).
    pub suggestion: Option<String>,
}

/// Causal explorer widget.
pub struct CausalExplorer<'a> {
    links: Vec<CausalLink>,
    scroll_offset: usize,
    block: Option<Block<'a>>,
}

impl<'a> CausalExplorer<'a> {
    pub fn new(links: Vec<CausalLink>) -> Self {
        Self {
            links,
            scroll_offset: 0,
            block: None,
        }
    }

    pub fn scroll(mut self, offset: usize) -> Self {
        self.scroll_offset = offset;
        self
    }

    pub fn block(mut self, block: Block<'a>) -> Self {
        self.block = Some(block);
        self
    }

    fn confidence_color(confidence: f64) -> Color {
        if confidence > 0.8 {
            Color::Green
        } else if confidence > 0.5 {
            Color::Yellow
        } else {
            Color::DarkGray
        }
    }
}

impl Widget for CausalExplorer<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let block = self.block.unwrap_or_else(|| {
            Block::default()
                .title(" Causal Graph ")
                .borders(Borders::ALL)
        });
        let inner = block.inner(area);
        block.render(area, buf);

        if inner.height < 2 || inner.width < 20 || self.links.is_empty() {
            if inner.height >= 1 && inner.width >= 15 {
                buf.set_string(
                    inner.x + 1,
                    inner.y,
                    "No causal links",
                    Style::default().fg(Color::DarkGray),
                );
            }
            return;
        }

        // Header
        let header = Line::from(vec![
            Span::styled("From", Style::default().add_modifier(Modifier::BOLD)),
            Span::raw("  ->  "),
            Span::styled("To", Style::default().add_modifier(Modifier::BOLD)),
            Span::raw("  (conf)"),
        ]);
        buf.set_line(inner.x + 1, inner.y, &header, inner.width.saturating_sub(1));

        // Visible links
        let visible_count = (inner.height.saturating_sub(1)) as usize;
        let visible = self
            .links
            .iter()
            .skip(self.scroll_offset)
            .take(visible_count);

        for (i, link) in visible.enumerate() {
            let y = inner.y + 1 + i as u16;
            if y >= inner.y + inner.height {
                break;
            }

            let conf_color = Self::confidence_color(link.confidence);
            let max_name = ((inner.width as usize).saturating_sub(20)) / 2;
            let from = if link.from.len() > max_name {
                &link.from[..max_name]
            } else {
                &link.from
            };
            let to = if link.to.len() > max_name {
                &link.to[..max_name]
            } else {
                &link.to
            };

            let rel_indicator = match link.relationship.as_str() {
                "causal" => Span::styled(" C ", Style::default().fg(Color::Cyan)),
                "anomaly" => Span::styled(" ! ", Style::default().fg(Color::Red)),
                _ => Span::styled(" ? ", Style::default().fg(Color::DarkGray)),
            };
            let line = Line::from(vec![
                rel_indicator,
                Span::styled(from, Style::default().fg(Color::White)),
                Span::styled(" -> ", Style::default().fg(Color::DarkGray)),
                Span::styled(to, Style::default().fg(Color::White)),
                Span::raw(" "),
                Span::styled(
                    format!("{:.0}%", link.confidence * 100.0),
                    Style::default().fg(conf_color),
                ),
            ]);
            buf.set_line(inner.x + 1, y, &line, inner.width.saturating_sub(1));

            // Show suggestion if present and space allows
            if let Some(ref suggestion) = link.suggestion {
                let sy = inner.y + 2 + i as u16;
                if sy < inner.y + inner.height {
                    let sug_line = Line::from(Span::styled(
                        format!("    -> {}", suggestion),
                        Style::default().fg(Color::Cyan),
                    ));
                    buf.set_line(inner.x + 1, sy, &sug_line, inner.width.saturating_sub(1));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_render_with_links() {
        let links = vec![
            CausalLink {
                from: "boot.kernelPackages".into(),
                to: "hardware.nvidia.package".into(),
                confidence: 0.85,
                relationship: "determines".into(),
                suggestion: None,
            },
            CausalLink {
                from: "services.xserver.enable".into(),
                to: "services.displayManager".into(),
                confidence: 0.92,
                relationship: "requires".into(),
                suggestion: Some("Enable display manager first".into()),
            },
        ];
        let widget = CausalExplorer::new(links)
            .block(Block::default().title("Causal").borders(Borders::ALL));
        let area = Rect::new(0, 0, 60, 10);
        let mut buf = Buffer::empty(area);
        widget.render(area, &mut buf);
    }

    #[test]
    fn test_render_empty() {
        let widget = CausalExplorer::new(vec![]);
        let area = Rect::new(0, 0, 40, 6);
        let mut buf = Buffer::empty(area);
        widget.render(area, &mut buf);
    }

    #[test]
    fn test_scroll() {
        let links: Vec<CausalLink> = (0..20)
            .map(|i| CausalLink {
                from: format!("from.{}", i),
                to: format!("to.{}", i),
                confidence: 0.5 + (i as f64) * 0.02,
                relationship: "affects".into(),
                suggestion: None,
            })
            .collect();
        let widget = CausalExplorer::new(links).scroll(5);
        let area = Rect::new(0, 0, 50, 8);
        let mut buf = Buffer::empty(area);
        widget.render(area, &mut buf);
    }
}
