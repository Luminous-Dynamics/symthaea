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
}

/// System health dashboard widget.
pub struct SystemHealth<'a> {
    snapshot: HealthSnapshot,
    block: Option<Block<'a>>,
}

impl<'a> SystemHealth<'a> {
    pub fn new(snapshot: HealthSnapshot) -> Self {
        Self { snapshot, block: None }
    }

    pub fn block(mut self, block: Block<'a>) -> Self {
        self.block = Some(block);
        self
    }
}

impl Widget for SystemHealth<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let block = self.block.unwrap_or_else(|| {
            Block::default().title(" System Health ").borders(Borders::ALL)
        });
        let inner = block.inner(area);
        block.render(area, buf);

        if inner.height < 3 || inner.width < 15 {
            return;
        }

        let mut y = inner.y;
        let x = inner.x + 1;

        // Services line
        let svc_color = if self.snapshot.services_failed > 0 { Color::Red } else { Color::Green };
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
                Span::styled(&self.snapshot.store_size_human, Style::default().fg(Color::Cyan)),
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
                Span::styled(gen_str, Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
                Span::raw(format!(" ({} total)", self.snapshot.total_generations)),
            ]);
            buf.set_line(x, y, &gen_line, inner.width.saturating_sub(1));
            y += 1;
        }

        // Memory line (optional)
        if y < inner.y + inner.height {
            if let Some(pct) = self.snapshot.memory_used_percent {
                let mem_color = if pct > 90.0 { Color::Red } else if pct > 70.0 { Color::Yellow } else { Color::Green };
                let mem_line = Line::from(vec![
                    Span::raw("Memory:   "),
                    Span::styled(format!("{:.0}%", pct), Style::default().fg(mem_color)),
                ]);
                buf.set_line(x, y, &mem_line, inner.width.saturating_sub(1));
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
        };
        let widget = SystemHealth::new(snap)
            .block(Block::default().title("Health").borders(Borders::ALL));
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
}
