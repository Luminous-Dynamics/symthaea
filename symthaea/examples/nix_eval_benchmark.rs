// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! NixEval Benchmark — 20 prompts with structural-correctness assertions.
//!
//! Each problem specifies:
//! - prompt: natural language request
//! - expected_intent: what System 1 should classify as
//! - expected_substrings: must appear in the generated Nix
//! - forbidden_substrings: must NOT appear (catches wrong idioms)
//! - require_parse: whether nix-instantiate --parse must succeed
//!
//! Honest measurement of the hybrid pipeline on a curated test set.
//!
//! Run:
//!   cargo run --example nix_eval_benchmark --features code_generation

use symthaea::language::nix_codegen::{generate_nix, NixIntent};

struct NixProblem {
    prompt: &'static str,
    expected_intent: NixIntent,
    expected_substrings: &'static [&'static str],
    forbidden_substrings: &'static [&'static str],
    require_parse: bool,
}

fn problems() -> Vec<NixProblem> {
    vec![
        // ── Dev Shells (4) ──
        NixProblem {
            prompt: "set up a rust dev environment with rust-analyzer and mold",
            expected_intent: NixIntent::DevShell,
            expected_substrings: &["mkShell", "rustc", "cargo", "rust-analyzer", "mold"],
            forbidden_substrings: &["python", "nodejs"],
            require_parse: true,
        },
        NixProblem {
            prompt: "configure a python data-science environment with jupyter and pandas",
            expected_intent: NixIntent::DevShell,
            expected_substrings: &["python311", "jupyter", "pandas"],
            forbidden_substrings: &["rustc", "cargo"],
            require_parse: true,
        },
        NixProblem {
            prompt: "set up a node development environment with typescript",
            expected_intent: NixIntent::DevShell,
            expected_substrings: &["nodejs", "typescript"],
            forbidden_substrings: &["python", "rustc"],
            require_parse: true,
        },
        NixProblem {
            prompt: "rust dev shell with sccache and openssl",
            expected_intent: NixIntent::DevShell,
            expected_substrings: &["rustc", "sccache", "openssl"],
            forbidden_substrings: &["pandas"],
            require_parse: true,
        },
        // ── Services (5) ──
        NixProblem {
            prompt: "set up postgresql with pgvector",
            expected_intent: NixIntent::Service,
            expected_substrings: &["services.postgresql", "enable = true", "pgvector"],
            forbidden_substrings: &["postgis", "redis"],
            require_parse: true,
        },
        NixProblem {
            prompt: "enable docker and add my user to the docker group",
            expected_intent: NixIntent::Service,
            expected_substrings: &["virtualisation.docker", "docker"],
            forbidden_substrings: &["postgresql", "podman"],
            require_parse: true,
        },
        NixProblem {
            prompt: "enable nginx web server",
            expected_intent: NixIntent::Service,
            expected_substrings: &["services.nginx", "enable = true"],
            forbidden_substrings: &["apache"],
            require_parse: true,
        },
        NixProblem {
            prompt: "enable redis cache server",
            expected_intent: NixIntent::Service,
            expected_substrings: &["services.redis"],
            forbidden_substrings: &["postgresql"],
            require_parse: true,
        },
        NixProblem {
            prompt: "set up ipfs kubo node",
            expected_intent: NixIntent::Service,
            expected_substrings: &["services.kubo", "Datastore"],
            forbidden_substrings: &[],
            require_parse: true,
        },
        // ── Hardware (2) ──
        NixProblem {
            prompt: "configure nvidia gpu drivers",
            expected_intent: NixIntent::Hardware,
            expected_substrings: &["hardware.nvidia", "modesetting"],
            forbidden_substrings: &["amdgpu"],
            require_parse: true,
        },
        NixProblem {
            prompt: "enable nvidia hardware acceleration",
            expected_intent: NixIntent::Hardware,
            expected_substrings: &["hardware.nvidia", "vulkan"],
            forbidden_substrings: &[],
            require_parse: true,
        },
        // ── Desktop (3) ──
        NixProblem {
            prompt: "enable wayland with sway and configure standard fonts",
            expected_intent: NixIntent::Desktop,
            expected_substrings: &["programs.sway", "xdg.portal", "fonts.packages"],
            forbidden_substrings: &["x11", "xorg"],
            require_parse: true,
        },
        NixProblem {
            prompt: "enable kde plasma desktop environment",
            expected_intent: NixIntent::Desktop,
            expected_substrings: &["plasma6", "sddm"],
            forbidden_substrings: &["gnome", "xfce"],
            require_parse: true,
        },
        NixProblem {
            prompt: "set up sway window manager",
            expected_intent: NixIntent::Desktop,
            expected_substrings: &["programs.sway"],
            forbidden_substrings: &["plasma"],
            require_parse: true,
        },
        // ── User / Permission (2) ──
        NixProblem {
            prompt: "create a user with audio video and networkmanager groups",
            expected_intent: NixIntent::User,
            expected_substrings: &["users.users.tstoltz", "audio", "video", "networkmanager"],
            forbidden_substrings: &[],
            require_parse: true,
        },
        NixProblem {
            prompt: "add user to wheel and audio groups",
            expected_intent: NixIntent::User,
            expected_substrings: &["users.users.tstoltz", "wheel", "audio"],
            forbidden_substrings: &[],
            require_parse: true,
        },
        // ── Networking (2) ──
        NixProblem {
            prompt: "open firewall ports 80 and 443",
            expected_intent: NixIntent::Networking,
            expected_substrings: &["allowedTCPPorts", "80", "443"],
            forbidden_substrings: &[],
            require_parse: true,
        },
        NixProblem {
            prompt: "open port 8080 in firewall",
            expected_intent: NixIntent::Networking,
            expected_substrings: &["allowedTCPPorts", "8080"],
            forbidden_substrings: &[],
            require_parse: true,
        },
        // ── Home Manager (1) ──
        NixProblem {
            prompt: "configure git with home-manager",
            expected_intent: NixIntent::HomeManager,
            expected_substrings: &["programs.git", "enable = true"],
            forbidden_substrings: &[],
            require_parse: true,
        },
        // ── Edge case (1) ──
        NixProblem {
            prompt: "explain monads",
            expected_intent: NixIntent::Generic,
            expected_substrings: &[],
            forbidden_substrings: &[],
            require_parse: true,
        },
        // ── Hyprland + GNOME (Task 1) ──
        NixProblem {
            prompt: "set up hyprland with fonts",
            expected_intent: NixIntent::Desktop,
            expected_substrings: &["programs.hyprland", "xdg.portal"],
            forbidden_substrings: &["sway", "kde"],
            require_parse: true,
        },
        NixProblem {
            prompt: "enable hyprland wayland compositor",
            expected_intent: NixIntent::Desktop,
            expected_substrings: &["programs.hyprland", "xwayland"],
            forbidden_substrings: &["plasma"],
            require_parse: true,
        },
        NixProblem {
            prompt: "enable gnome desktop with extensions",
            expected_intent: NixIntent::Desktop,
            expected_substrings: &["gnome", "gdm", "gnomeExtensions"],
            forbidden_substrings: &["plasma", "sway"],
            require_parse: true,
        },
        NixProblem {
            prompt: "set up gnome desktop environment",
            expected_intent: NixIntent::Desktop,
            expected_substrings: &["desktopManager.gnome", "gdm"],
            forbidden_substrings: &["plasma"],
            require_parse: true,
        },
        // ── Secrets (Task 2) ──
        NixProblem {
            prompt: "set up sops secrets management",
            expected_intent: NixIntent::Secrets,
            expected_substrings: &["sops", "defaultSopsFile", "age"],
            forbidden_substrings: &["agenix"],
            require_parse: true,
        },
        NixProblem {
            prompt: "configure agenix for encrypted secrets",
            expected_intent: NixIntent::Secrets,
            expected_substrings: &["agenix", "age", "secrets"],
            forbidden_substrings: &[],
            require_parse: true,
        },
        NixProblem {
            prompt: "manage credentials with encrypted secret store",
            expected_intent: NixIntent::Secrets,
            expected_substrings: &["sops"],
            forbidden_substrings: &[],
            require_parse: true,
        },
        // ── Flake templates (Task 3) ──
        NixProblem {
            prompt: "complete flake template for rust and python project",
            expected_intent: NixIntent::FlakeTemplate,
            expected_substrings: &["devShells", "rustc", "python311", "nixpkgs"],
            forbidden_substrings: &[],
            require_parse: true,
        },
        NixProblem {
            prompt: "system flake with home-manager",
            expected_intent: NixIntent::FlakeTemplate,
            expected_substrings: &["nixosConfigurations", "home-manager", "nixosSystem"],
            forbidden_substrings: &[],
            require_parse: true,
        },
        NixProblem {
            prompt: "flake template for nodejs and typescript devshell",
            expected_intent: NixIntent::FlakeTemplate,
            expected_substrings: &["devShells", "nodejs"],
            forbidden_substrings: &["python311"],
            require_parse: true,
        },
        // ── Tricky cases — multi-keyword overlap ──
        NixProblem {
            prompt: "set up rust dev environment with sccache",
            expected_intent: NixIntent::DevShell,
            expected_substrings: &["rustc", "sccache", "mkShell"],
            forbidden_substrings: &["python", "nodejs"],
            require_parse: true,
        },
        NixProblem {
            prompt: "enable nginx and open ports 80 and 443",
            // Service should win — nginx + the firewall is part of the service idiom
            expected_intent: NixIntent::Service,
            expected_substrings: &["services.nginx", "allowedTCPPorts", "80", "443"],
            forbidden_substrings: &[],
            require_parse: true,
        },
        NixProblem {
            prompt: "configure postgresql service",
            expected_intent: NixIntent::Service,
            expected_substrings: &["services.postgresql", "enable"],
            forbidden_substrings: &["pgvector", "postgis"],
            require_parse: true,
        },
        NixProblem {
            prompt: "open firewall port 22000",
            expected_intent: NixIntent::Networking,
            expected_substrings: &["allowedTCPPorts", "22000"],
            forbidden_substrings: &[],
            require_parse: true,
        },
        // ── Out of distribution — should still classify cleanly ──
        NixProblem {
            prompt: "what is the meaning of life",
            expected_intent: NixIntent::Generic,
            expected_substrings: &[],
            forbidden_substrings: &["services", "hardware", "nvidia", "users"],
            require_parse: true,
        },
    ]
}

#[derive(Default)]
struct ScoreCard {
    total: usize,
    intent_correct: usize,
    parse_correct: usize,
    expected_present: usize,
    forbidden_absent: usize,
    full_pass: usize,
    by_intent: std::collections::HashMap<NixIntent, (usize, usize)>,
}

fn evaluate(problem: &NixProblem) -> (bool, bool, bool, bool, String) {
    let result = generate_nix(problem.prompt);
    let intent_ok = result.intent == problem.expected_intent;
    let parse_ok = if problem.require_parse {
        result.parses
    } else {
        true
    };

    let expected_ok = problem
        .expected_substrings
        .iter()
        .all(|s| result.code.contains(s));

    let forbidden_ok = problem
        .forbidden_substrings
        .iter()
        .all(|s| !result.code.contains(s));

    (intent_ok, parse_ok, expected_ok, forbidden_ok, result.code)
}

fn main() {
    let problems = problems();
    let mut card = ScoreCard::default();
    card.total = problems.len();

    println!("┌─────────────────────────────────────────────────────────");
    println!("│ NixEval Benchmark — {} problems", problems.len());
    println!("└─────────────────────────────────────────────────────────");

    for (i, p) in problems.iter().enumerate() {
        let (intent_ok, parse_ok, exp_ok, forbid_ok, code) = evaluate(p);

        if intent_ok {
            card.intent_correct += 1;
        }
        if parse_ok {
            card.parse_correct += 1;
        }
        if exp_ok {
            card.expected_present += 1;
        }
        if forbid_ok {
            card.forbidden_absent += 1;
        }
        let full = intent_ok && parse_ok && exp_ok && forbid_ok;
        if full {
            card.full_pass += 1;
        }

        let entry = card.by_intent.entry(p.expected_intent).or_insert((0, 0));
        entry.1 += 1;
        if full {
            entry.0 += 1;
        }

        let mark = if full { "✓" } else { "✗" };
        let detail = format!(
            "intent={} parse={} expected={} forbidden={}",
            if intent_ok { "✓" } else { "✗" },
            if parse_ok { "✓" } else { "✗" },
            if exp_ok { "✓" } else { "✗" },
            if forbid_ok { "✓" } else { "✗" },
        );
        println!("  {} #{:02} {:50} | {}", mark, i + 1, p.prompt, detail);

        if !full {
            // Show what was missing for diagnostics
            for s in p.expected_substrings.iter() {
                if !code.contains(s) {
                    println!("       missing: {s}");
                }
            }
            for s in p.forbidden_substrings.iter() {
                if code.contains(s) {
                    println!("       leaked:  {s}");
                }
            }
        }
    }

    let n = card.total as f32;
    println!("\n╔═════════════════════════════════════════════════════════");
    println!(
        "║ Intent classification:  {}/{} ({:.0}%)",
        card.intent_correct,
        card.total,
        card.intent_correct as f32 / n * 100.0
    );
    println!(
        "║ Parses successfully:    {}/{} ({:.0}%)",
        card.parse_correct,
        card.total,
        card.parse_correct as f32 / n * 100.0
    );
    println!(
        "║ Expected substrings:    {}/{} ({:.0}%)",
        card.expected_present,
        card.total,
        card.expected_present as f32 / n * 100.0
    );
    println!(
        "║ No forbidden leakage:   {}/{} ({:.0}%)",
        card.forbidden_absent,
        card.total,
        card.forbidden_absent as f32 / n * 100.0
    );
    println!(
        "║ FULL PASS (all 4):      {}/{} ({:.0}%)",
        card.full_pass,
        card.total,
        card.full_pass as f32 / n * 100.0
    );
    println!("╚═════════════════════════════════════════════════════════");

    println!("\nPer-intent full pass rate:");
    let mut intents: Vec<_> = card.by_intent.iter().collect();
    intents.sort_by_key(|(k, _)| format!("{:?}", k));
    for (intent, (passed, total)) in intents {
        println!(
            "  {:?}: {}/{} ({:.0}%)",
            intent,
            passed,
            total,
            *passed as f32 / *total as f32 * 100.0
        );
    }
}
