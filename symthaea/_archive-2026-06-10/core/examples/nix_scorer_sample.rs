// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Sample tool — generate output for a batch of prompts and print them
//! so goldens can be written to match the generator's natural shape.
//! Not a scorer; just a bulk `generate_nix` caller.
//!
//! Run:
//!   cargo run --features code_generation --example nix_scorer_sample

use symthaea::language::nix_codegen::generate_nix;

fn main() {
    let prompts = [
        "enable nvidia hardware acceleration",
        "configure intel hardware acceleration",
        "set up hyprland with fonts",
        "enable hyprland wayland compositor",
        "enable gnome desktop with extensions",
        "set up gnome desktop environment",
        "open port 8080 in firewall",
        "open udp port 51820 for wireguard",
        "enable tailscale VPN",
        "configure prometheus monitoring",
        "grafana dashboard server",
        "configure CUPS printing service",
        "set time zone to Africa/Johannesburg",
        "enable systemd-resolved for DNS",
    ];
    for prompt in prompts {
        println!("=== {prompt} ===");
        let r = generate_nix(prompt);
        println!("{}", r.code);
        println!();
    }
}
