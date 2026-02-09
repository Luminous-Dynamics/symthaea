//! Consciousness Gauge Widget — 2D Phi x Confidence Space
//!
//! Visualizes the system's consciousness state as a point in a 2D space:
//! - X-axis: Confidence (0.0 -> 1.0)
//! - Y-axis: Phi (0.0 -> 1.0)
//!
//! Quadrants:
//!   Top-Right: Confident (high Phi, high Confidence)
//!   Top-Left:  Curious (high Phi, low Confidence)
//!   Bot-Right: Habitual (low Phi, high Confidence)
//!   Bot-Left:  Confused (low Phi, low Confidence)

use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Widget},
};

/// Consciousness state for the gauge.
#[derive(Debug, Clone, Copy)]
pub struct ConsciousnessState {
    pub phi: f64,
    pub confidence: f64,
    pub free_energy: f64,
}

impl Default for ConsciousnessState {
    fn default() -> Self {
        Self { phi: 0.5, confidence: 0.5, free_energy: 0.5 }
    }
}

impl ConsciousnessState {
    pub fn quadrant_name(&self) -> &'static str {
        match (self.phi > 0.5, self.confidence > 0.5) {
            (true, true) => "Confident",
            (true, false) => "Curious",
            (false, true) => "Habitual",
            (false, false) => "Confused",
        }
    }

    pub fn quadrant_color(&self) -> Color {
        match (self.phi > 0.5, self.confidence > 0.5) {
            (true, true) => Color::Green,
            (true, false) => Color::Cyan,
            (false, true) => Color::Yellow,
            (false, false) => Color::Red,
        }
    }
}

/// The consciousness gauge widget.
pub struct ConsciousnessGauge<'a> {
    state: ConsciousnessState,
    block: Option<Block<'a>>,
}

impl<'a> ConsciousnessGauge<'a> {
    pub fn new(state: ConsciousnessState) -> Self {
        Self { state, block: None }
    }

    pub fn block(mut self, block: Block<'a>) -> Self {
        self.block = Some(block);
        self
    }
}

impl Widget for ConsciousnessGauge<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let block = self.block.unwrap_or_else(|| {
            Block::default().title(" Consciousness ").borders(Borders::ALL)
        });
        let inner = block.inner(area);
        block.render(area, buf);

        if inner.height < 4 || inner.width < 20 {
            return;
        }

        let grid_h = inner.height.saturating_sub(2);
        let grid_w = inner.width.saturating_sub(4);
        if grid_h == 0 || grid_w == 0 {
            return;
        }

        let mid_x = inner.x + 2 + grid_w / 2;
        let mid_y = inner.y + grid_h / 2;

        // Crosshairs
        for x in (inner.x + 2)..(inner.x + 2 + grid_w) {
            buf.set_string(x, mid_y, "-", Style::default().fg(Color::DarkGray));
        }
        for y in inner.y..(inner.y + grid_h) {
            buf.set_string(mid_x, y, "|", Style::default().fg(Color::DarkGray));
        }
        buf.set_string(mid_x, mid_y, "+", Style::default().fg(Color::DarkGray));

        // Quadrant labels
        buf.set_string(inner.x + 2, inner.y, "Cur", Style::default().fg(Color::Cyan).add_modifier(Modifier::DIM));
        let rw = (inner.x + 2 + grid_w).saturating_sub(3);
        buf.set_string(rw, inner.y, "Con", Style::default().fg(Color::Green).add_modifier(Modifier::DIM));

        // Plot current state
        let px = inner.x + 2 + ((self.state.confidence * grid_w as f64) as u16).min(grid_w.saturating_sub(1));
        let py = inner.y + grid_h.saturating_sub(1)
            - ((self.state.phi * grid_h.saturating_sub(1) as f64) as u16).min(grid_h.saturating_sub(1));
        let color = self.state.quadrant_color();
        buf.set_string(px, py, "@", Style::default().fg(color).add_modifier(Modifier::BOLD));

        // Status line
        let status = format!(
            "P:{:.2} C:{:.2} FE:{:.2} [{}]",
            self.state.phi, self.state.confidence, self.state.free_energy,
            self.state.quadrant_name(),
        );
        let sy = inner.y + inner.height.saturating_sub(1);
        buf.set_line(inner.x, sy, &Line::from(Span::styled(status, Style::default().fg(color))), inner.width);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quadrant_names() {
        let s = ConsciousnessState { phi: 0.8, confidence: 0.8, free_energy: 0.1 };
        assert_eq!(s.quadrant_name(), "Confident");
        let s = ConsciousnessState { phi: 0.8, confidence: 0.2, free_energy: 0.5 };
        assert_eq!(s.quadrant_name(), "Curious");
        let s = ConsciousnessState { phi: 0.2, confidence: 0.8, free_energy: 0.5 };
        assert_eq!(s.quadrant_name(), "Habitual");
        let s = ConsciousnessState { phi: 0.2, confidence: 0.2, free_energy: 0.9 };
        assert_eq!(s.quadrant_name(), "Confused");
    }

    #[test]
    fn test_render_doesnt_panic() {
        let gauge = ConsciousnessGauge::new(ConsciousnessState::default())
            .block(Block::default().title("Test").borders(Borders::ALL));
        let area = Rect::new(0, 0, 40, 15);
        let mut buf = Buffer::empty(area);
        gauge.render(area, &mut buf);
    }

    #[test]
    fn test_render_tiny_doesnt_panic() {
        let gauge = ConsciousnessGauge::new(ConsciousnessState::default());
        let area = Rect::new(0, 0, 5, 2);
        let mut buf = Buffer::empty(area);
        gauge.render(area, &mut buf);
    }
}
