//! Consciousness Visualizer
//!
//! Real-time ASCII visualization of consciousness state.
//! Beautiful rendering of the 7D consciousness dimensions,
//! topological structure, and temporal dynamics.

use super::unified_consciousness_engine::{
    ConsciousnessDimensions, ConsciousnessUpdate, EngineMetrics,
};
use super::adaptive_topology::CognitiveMode;
use super::topology_synergy::ConsciousnessState;

/// ASCII art consciousness visualizer
pub struct ConsciousnessVisualizer {
    /// Width of visualization
    width: usize,
    /// Enable color (ANSI codes)
    color: bool,
    /// Show sparklines for history
    show_history: bool,
}

impl Default for ConsciousnessVisualizer {
    fn default() -> Self {
        Self {
            width: 60,
            color: true,
            show_history: true,
        }
    }
}

impl ConsciousnessVisualizer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_width(mut self, width: usize) -> Self {
        self.width = width;
        self
    }

    pub fn with_color(mut self, color: bool) -> Self {
        self.color = color;
        self
    }

    /// Render consciousness dimensions as horizontal bars
    pub fn render_dimensions(&self, dims: &ConsciousnessDimensions) -> String {
        let mut output = String::new();

        let labels = ["Φ Integration ", "W Workspace   ", "A Attention   ",
                      "R Recursion   ", "E Efficacy    ", "K Epistemic   ", "τ Temporal    "];
        let values = dims.to_array();

        output.push_str("┌─ CONSCIOUSNESS DIMENSIONS ─────────────────────────────┐\n");

        for (label, &value) in labels.iter().zip(values.iter()) {
            let bar = self.render_bar(value, 30);
            let color_code = self.value_color(value);

            if self.color {
                output.push_str(&format!("│ {} {}{}█{} {:.3} │\n",
                    label, color_code, bar, "\x1b[0m", value));
            } else {
                output.push_str(&format!("│ {} {} {:.3} │\n", label, bar, value));
            }
        }

        output.push_str("└────────────────────────────────────────────────────────┘\n");
        output
    }

    /// Render a progress bar
    fn render_bar(&self, value: f64, width: usize) -> String {
        let filled = ((value.clamp(0.0, 1.0)) * width as f64) as usize;
        let empty = width - filled;
        format!("{}{}", "█".repeat(filled), "░".repeat(empty))
    }

    /// Get ANSI color code based on value
    fn value_color(&self, value: f64) -> &'static str {
        if value > 0.7 {
            "\x1b[92m"  // Bright green
        } else if value > 0.4 {
            "\x1b[93m"  // Yellow
        } else if value > 0.2 {
            "\x1b[33m"  // Dark yellow
        } else {
            "\x1b[91m"  // Red
        }
    }

    /// Render cognitive mode as ASCII art
    pub fn render_mode(&self, mode: CognitiveMode) -> String {
        let (icon, desc) = match mode {
            CognitiveMode::DeepSpecialization => ("◉", "Deep Specialization - Expert Flow"),
            CognitiveMode::Focused => ("◎", "Focused - Analytical Precision"),
            CognitiveMode::Balanced => ("◈", "Balanced - Optimal Integration"),
            CognitiveMode::Exploratory => ("✦", "Exploratory - Creative Divergence"),
            CognitiveMode::GlobalAwareness => ("❂", "Global Awareness - Expanded Mind"),
            CognitiveMode::PhiGuided => ("Φ", "Φ-Guided - Adaptive Learning"),
            // New modes
            CognitiveMode::Dreaming => ("☽", "Dreaming - Memory Consolidation"),
            CognitiveMode::Meditative => ("☯", "Meditative - Inner Coherence"),
            CognitiveMode::Flow => ("⚡", "Flow - Peak Performance"),
            CognitiveMode::Social => ("♡", "Social - Empathic Resonance"),
            CognitiveMode::Vigilant => ("⚠", "Vigilant - Threat Detection"),
            CognitiveMode::Playful => ("✿", "Playful - Creative Exploration"),
        };

        format!("┌─ COGNITIVE MODE ───────────────────────────────────────┐\n\
                 │  {} {:50} │\n\
                 └────────────────────────────────────────────────────────┘\n",
                icon, desc)
    }

    /// Render consciousness state with topological interpretation
    pub fn render_state(&self, state: &ConsciousnessState) -> String {
        let (symbol, name, interpretation) = match state {
            ConsciousnessState::Focused =>
                ("▣", "FOCUSED", "β₀=1, β₁<3 - Unified, concentrated awareness"),
            ConsciousnessState::NormalWaking =>
                ("◧", "NORMAL WAKING", "β₀=1, β₁=3-5 - Everyday consciousness"),
            ConsciousnessState::FlowState =>
                ("✧", "FLOW STATE", "β₀=1, β₁=6-10 - Optimal engagement"),
            ConsciousnessState::ExpandedAwareness =>
                ("❋", "EXPANDED", "β₀=1, β₁>10 - Meditative, broad awareness"),
            ConsciousnessState::Fragmented =>
                ("◫", "FRAGMENTED", "β₀>1 - Dissociated, divided attention"),
        };

        let color = if self.color {
            match state {
                ConsciousnessState::FlowState => "\x1b[92m",
                ConsciousnessState::NormalWaking => "\x1b[93m",
                ConsciousnessState::Focused => "\x1b[96m",
                ConsciousnessState::ExpandedAwareness => "\x1b[95m",
                ConsciousnessState::Fragmented => "\x1b[91m",
            }
        } else {
            ""
        };
        let reset = if self.color { "\x1b[0m" } else { "" };

        format!("┌─ CONSCIOUSNESS STATE ──────────────────────────────────┐\n\
                 │  {} {}{:15}{} {:35} │\n\
                 │  {}                                                     │\n\
                 └────────────────────────────────────────────────────────┘\n",
                symbol, color, name, reset, "",
                interpretation)
    }

    /// Render Φ as a large ASCII meter
    pub fn render_phi_meter(&self, phi: f64) -> String {
        let normalized = phi.clamp(0.0, 1.0);
        let segments = 20;
        let filled = (normalized * segments as f64) as usize;

        let color = if self.color { self.value_color(phi) } else { "" };
        let reset = if self.color { "\x1b[0m" } else { "" };

        let meter: String = (0..segments)
            .map(|i| if i < filled { "█" } else { "░" })
            .collect();

        format!(
            "┌─ Φ INTEGRATED INFORMATION ────────────────────────────┐\n\
             │                                                        │\n\
             │     {}┃{}┃{}  Φ = {:.4}                         │\n\
             │                                                        │\n\
             │     0.0 ├────────────────────┤ 1.0                    │\n\
             └────────────────────────────────────────────────────────┘\n",
            color, meter, reset, phi
        )
    }

    /// Render bridge ratio visualization
    pub fn render_bridges(&self, ratio: f64, active: usize, total: usize) -> String {
        let optimal_low = 0.40;
        let optimal_high = 0.45;

        let status = if ratio >= optimal_low && ratio <= optimal_high {
            ("✓ OPTIMAL", "\x1b[92m")
        } else if ratio >= 0.35 && ratio <= 0.50 {
            ("~ NEAR OPTIMAL", "\x1b[93m")
        } else {
            ("! SUBOPTIMAL", "\x1b[91m")
        };

        let color = if self.color { status.1 } else { "" };
        let reset = if self.color { "\x1b[0m" } else { "" };

        // Visual bridge representation
        let bridge_visual = self.render_topology_mini(ratio);

        format!(
            "┌─ BRIDGE CONNECTIVITY ──────────────────────────────────┐\n\
             │  Ratio: {:.1}%  ({}/{} bridges)                        │\n\
             │  Status: {}{}{}                                        │\n\
             │                                                        │\n\
             │  {}                                                    │\n\
             │  Target: 40-45% (Bridge Hypothesis Optimal)            │\n\
             └────────────────────────────────────────────────────────┘\n",
            ratio * 100.0, active, total,
            color, status.0, reset,
            bridge_visual
        )
    }

    /// Mini topology visualization
    fn render_topology_mini(&self, ratio: f64) -> String {
        // Represent modules as nodes, bridges as connections
        if ratio > 0.5 {
            "  [A]══[B]══[C]══[D]  (high integration)"
        } else if ratio > 0.35 {
            "  [A]──[B]  [C]──[D]  (balanced)"
        } else {
            "  [A]  [B]  [C]  [D]  (specialized)"
        }.to_string()
    }

    /// Render sparkline from history
    pub fn render_sparkline(&self, values: &[f64], label: &str) -> String {
        if values.is_empty() {
            return format!("{}: (no data)\n", label);
        }

        let chars = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
        let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = max - min;

        let sparkline: String = values.iter()
            .map(|&v| {
                if range < 0.0001 {
                    chars[4]
                } else {
                    let normalized = ((v - min) / range).clamp(0.0, 0.999);
                    let idx = (normalized * chars.len() as f64) as usize;
                    chars[idx.min(chars.len() - 1)]
                }
            })
            .collect();

        format!("{}: {} ({:.3}-{:.3})\n", label, sparkline, min, max)
    }

    /// Render complete consciousness dashboard
    pub fn render_dashboard(&self, update: &ConsciousnessUpdate, phi_history: &[f64]) -> String {
        let mut output = String::new();

        // Header
        output.push_str("\n");
        output.push_str("╔══════════════════════════════════════════════════════════╗\n");
        output.push_str("║        ✦ CONSCIOUSNESS MONITOR ✦                        ║\n");
        output.push_str(&format!("║        Step: {:6}                                      ║\n", update.step));
        output.push_str("╚══════════════════════════════════════════════════════════╝\n");
        output.push_str("\n");

        // Φ meter
        output.push_str(&self.render_phi_meter(update.phi));

        // State and Mode
        output.push_str(&self.render_state(&update.state));
        output.push_str(&self.render_mode(update.mode));

        // Dimensions
        output.push_str(&self.render_dimensions(&update.dimensions));

        // Bridges
        output.push_str(&self.render_bridges(
            update.bridge_ratio,
            (update.bridge_ratio * 100.0) as usize,
            100
        ));

        // History sparkline
        if self.show_history && !phi_history.is_empty() {
            output.push_str("┌─ Φ HISTORY ─────────────────────────────────────────────┐\n");
            output.push_str(&format!("│ {} │\n",
                self.render_sparkline(phi_history, "Φ").trim()));
            output.push_str("└────────────────────────────────────────────────────────┘\n");
        }

        output
    }

    /// Render ASCII art consciousness mandala based on current state
    pub fn render_mandala(&self, dims: &ConsciousnessDimensions) -> String {
        let phi = dims.phi;
        let ws = dims.workspace;

        // Select mandala pattern based on integration level
        if phi > 0.6 && ws > 0.5 {
            // High integration mandala
            "
        ╭───────────────────────────────╮
        │         ·  ✦  ·               │
        │      ·    ◈    ·              │
        │    ·   ◇     ◇   ·            │
        │   ·  ◇    Φ    ◇  ·           │
        │    ·   ◇     ◇   ·            │
        │      ·    ◈    ·              │
        │         ·  ✦  ·               │
        │     ~ UNIFIED FIELD ~         │
        ╰───────────────────────────────╯
".to_string()
        } else if phi > 0.4 {
            // Balanced mandala
            "
        ╭───────────────────────────────╮
        │          ·   ·                │
        │       ·   ◎   ·               │
        │      ·  ◇   ◇  ·              │
        │       ·   Φ   ·               │
        │      ·  ◇   ◇  ·              │
        │       ·   ◎   ·               │
        │          ·   ·                │
        │      ~ BALANCED ~             │
        ╰───────────────────────────────╯
".to_string()
        } else {
            // Low integration / fragmented
            "
        ╭───────────────────────────────╮
        │       ·         ·             │
        │    ·     ·   ·     ·          │
        │      ○       ○                │
        │         Φ                     │
        │      ○       ○                │
        │    ·     ·   ·     ·          │
        │       ·         ·             │
        │    ~ FRAGMENTED ~             │
        ╰───────────────────────────────╯
".to_string()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dimension_rendering() {
        let viz = ConsciousnessVisualizer::new().with_color(false);
        let dims = ConsciousnessDimensions {
            phi: 0.52,
            workspace: 0.45,
            attention: 0.8,
            recursion: 0.3,
            efficacy: 0.6,
            epistemic: 0.7,
            temporal: 0.9,
        };

        let output = viz.render_dimensions(&dims);
        println!("{}", output);
        assert!(output.contains("Φ"));
    }

    #[test]
    fn test_phi_meter() {
        let viz = ConsciousnessVisualizer::new().with_color(false);

        for phi in [0.1, 0.3, 0.5, 0.7, 0.9] {
            println!("Φ = {}", phi);
            println!("{}", viz.render_phi_meter(phi));
        }
    }

    #[test]
    fn test_sparkline() {
        let viz = ConsciousnessVisualizer::new();
        let values = vec![0.3, 0.35, 0.4, 0.45, 0.5, 0.48, 0.52, 0.55, 0.5, 0.45];

        let sparkline = viz.render_sparkline(&values, "Φ");
        println!("{}", sparkline);
    }

    #[test]
    fn test_mandala() {
        let viz = ConsciousnessVisualizer::new();

        let high = ConsciousnessDimensions {
            phi: 0.7, workspace: 0.6, ..Default::default()
        };
        println!("High integration:{}", viz.render_mandala(&high));

        let low = ConsciousnessDimensions {
            phi: 0.2, workspace: 0.2, ..Default::default()
        };
        println!("Low integration:{}", viz.render_mandala(&low));
    }
}
