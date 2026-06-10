// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! nix-mind-tui: terminal UI for conscious NixOS management.
//!
//! Launches a full-screen terminal interface showing:
//! - Consciousness gauge (2D Φ × Confidence space)
//! - System health dashboard (services, store, memory)
//! - Generation timeline
//! - World model state (prediction hierarchy, working memory)
//! - Causal graph explorer
//! - Natural language input with active inference

use std::io;

use clap::Parser;

#[derive(Parser)]
#[command(
    name = "nix-mind-tui",
    about = "Consciousness visualization TUI for NixOS",
    version
)]
struct Args {
    /// Dry-run mode: show what would happen without executing
    #[arg(long, short = 'n')]
    dry_run: bool,
}

fn main() -> io::Result<()> {
    let args = Args::parse();
    let mut app = symthaea_nix::tui::App::new(args.dry_run);
    app.run()
}
