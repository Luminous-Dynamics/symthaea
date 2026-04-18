// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//
//! Golden-reference Nix snippets for the NixEval corpus.
//!
//! The corpus in `nix_eval_corpus.rs` specifies prompts + expected/forbidden
//! substrings. The structural scorer in `nix_scorer.rs` needs an actual
//! reference AST to compare against. Rather than retrofit every one of the
//! 95 `NixProblem` entries with an `expected_config: Option<&'static str>`
//! field (a 95-line churn), goldens are looked up here by prompt string.
//!
//! Goldens are backfilled incrementally. The benchmark's `--structural`
//! mode falls through to the legacy substring check when no golden exists,
//! so this file can grow one entry at a time without breaking anything.
//!
//! All goldens must **parse** (rnix) and be **nixfmt-clean** so that the
//! canonicalization in `nix_scorer::canonicalize_value` meets them
//! half-way.

/// Return the golden reference for a prompt, or None if not yet backfilled.
pub fn golden_for(prompt: &str) -> Option<&'static str> {
    match prompt {
        // ── Services ──
        "set up postgresql with pgvector" => Some(POSTGRESQL_PGVECTOR),
        "configure postgresql service" => Some(POSTGRESQL_BASIC),
        "enable nginx web server" => Some(NGINX_ENABLE),
        "enable redis cache server" => Some(REDIS_ENABLE),
        "enable docker and add my user to the docker group" => Some(DOCKER_ENABLE),
        "set up ipfs kubo node" => Some(IPFS_KUBO),

        // ── Hardware ──
        "configure nvidia gpu drivers" => Some(NVIDIA_DRIVERS),

        // ── Desktop ──
        "set up sway window manager" => Some(SWAY_WM),
        "enable kde plasma desktop environment" => Some(KDE_PLASMA),

        // ── Networking ──
        "open firewall ports 80 and 443" => Some(FIREWALL_80_443),

        // ── Dev Shells ──
        "set up a rust dev environment with rust-analyzer and mold" => {
            Some(RUST_SHELL_ANALYZER_MOLD)
        }
        "rust dev shell with sccache and openssl" => Some(RUST_SHELL_SCCACHE_OPENSSL),
        "set up a node development environment with typescript" => Some(NODE_TYPESCRIPT_SHELL),

        _ => None,
    }
}

/// Number of golden-backed problems. Reported in the benchmark summary so
/// readers can see how much of the corpus is running under the strict
/// scorer vs. falling through to substring.
pub fn golden_count() -> usize {
    // Keep in sync with `golden_for` — one short function, manually audited.
    13
}

// ── Service goldens ───────────────────────────────────────────────────────

const POSTGRESQL_PGVECTOR: &str = r#"{ config, pkgs, ... }:
{
  services.postgresql.enable = true;
  services.postgresql.package = pkgs.postgresql_16;
  services.postgresql.extraPlugins = with pkgs.postgresql_16.pkgs; [ pgvector ];
}
"#;

const NGINX_ENABLE: &str = r#"{ config, pkgs, ... }:
{
  services.nginx.enable = true;
}
"#;

const REDIS_ENABLE: &str = r#"{ config, pkgs, ... }:
{
  services.redis.servers."".enable = true;
}
"#;

const DOCKER_ENABLE: &str = r#"{ config, pkgs, ... }:
{
  virtualisation.docker.enable = true;
  users.users.tstoltz.extraGroups = [ "docker" ];
}
"#;

const POSTGRESQL_BASIC: &str = r#"{ pkgs, ... }:
{
  services.postgresql.enable = true;
}
"#;

const IPFS_KUBO: &str = r#"{
  services.kubo.enable = true;
}
"#;

// ── Hardware goldens ──────────────────────────────────────────────────────

const NVIDIA_DRIVERS: &str = r#"{ pkgs, ... }:
{
  hardware.nvidia.modesetting.enable = true;
  services.xserver.videoDrivers = [ "nvidia" ];
}
"#;

// ── Desktop goldens ───────────────────────────────────────────────────────

const SWAY_WM: &str = r#"{ config, pkgs, ... }:
{
  programs.sway.enable = true;
}
"#;

const KDE_PLASMA: &str = r#"{
  services.desktopManager.plasma6.enable = true;
  services.displayManager.sddm.enable = true;
}
"#;

// ── Networking goldens ────────────────────────────────────────────────────

const FIREWALL_80_443: &str = r#"{
  networking.firewall.allowedTCPPorts = [ 80 443 ];
}
"#;

// ── Dev shell goldens ─────────────────────────────────────────────────────
//
// Dev shells don't set option attrpaths in the NixOS-module sense — they
// are `pkgs.mkShell { … }` expressions. The structural scorer still works
// because `mkShell` is called with an attrset literal whose attrpaths
// (`buildInputs`, `shellHook`, …) do register as NODE_ATTRPATH_VALUE.

const RUST_SHELL_ANALYZER_MOLD: &str = r#"{ pkgs ? import <nixpkgs> {} }:
pkgs.mkShell {
  buildInputs = with pkgs; [ rustc cargo rust-analyzer mold ];
}
"#;

const RUST_SHELL_SCCACHE_OPENSSL: &str = r#"{ pkgs ? import <nixpkgs> {} }:
pkgs.mkShell {
  buildInputs = with pkgs; [ rustc cargo sccache openssl pkg-config ];
  RUSTC_WRAPPER = "sccache";
}
"#;

const NODE_TYPESCRIPT_SHELL: &str = r#"{ pkgs ? import <nixpkgs> {} }:
pkgs.mkShell {
  buildInputs = with pkgs; [ nodejs_20 nodePackages.typescript ];
}
"#;

/// Run the generator against every golden-backed prompt and report the
/// per-problem structural verdict. Useful as a fast proxy for the full
/// 95-problem benchmark — this only runs the 6 golden-backed prompts, so
/// it's ~10x faster and proves the scorer plumbing works end-to-end.
///
/// Returned tuple: `(prompt, passed, verdict_summary)`.
#[cfg(feature = "code_generation")]
pub fn score_all_goldens() -> Vec<(&'static str, bool, String)> {
    use crate::language::nix_codegen::generate_nix;
    use crate::language::nix_scorer::score;

    let prompts = [
        "set up postgresql with pgvector",
        "configure postgresql service",
        "enable nginx web server",
        "enable redis cache server",
        "enable docker and add my user to the docker group",
        "set up ipfs kubo node",
        "configure nvidia gpu drivers",
        "set up sway window manager",
        "enable kde plasma desktop environment",
        "open firewall ports 80 and 443",
        "set up a rust dev environment with rust-analyzer and mold",
        "rust dev shell with sccache and openssl",
        "set up a node development environment with typescript",
    ];
    prompts
        .into_iter()
        .map(|prompt| {
            let result = generate_nix(prompt);
            let golden = golden_for(prompt).expect("prompt must have a golden");
            let verdict = score(&result.code, golden);
            (prompt, verdict.pass(), verdict.summary())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_golden_parses() {
        // Each golden must itself be valid Nix — otherwise the scorer
        // would spuriously fail generations that would have matched.
        let all = [
            POSTGRESQL_PGVECTOR,
            POSTGRESQL_BASIC,
            NGINX_ENABLE,
            REDIS_ENABLE,
            DOCKER_ENABLE,
            IPFS_KUBO,
            NVIDIA_DRIVERS,
            SWAY_WM,
            KDE_PLASMA,
            FIREWALL_80_443,
            RUST_SHELL_ANALYZER_MOLD,
            RUST_SHELL_SCCACHE_OPENSSL,
            NODE_TYPESCRIPT_SHELL,
        ];
        for (i, src) in all.iter().enumerate() {
            let parse = rnix::Root::parse(src);
            assert!(
                parse.errors().is_empty(),
                "golden #{} has parse errors: {:?}\n---\n{}",
                i,
                parse.errors(),
                src
            );
        }
    }

    #[test]
    fn lookup_returns_none_for_unknown_prompt() {
        assert!(golden_for("nonexistent prompt").is_none());
    }

    #[test]
    fn lookup_returns_some_for_known_prompt() {
        assert!(golden_for("enable nginx web server").is_some());
    }

    #[test]
    fn golden_count_is_consistent() {
        // Manually audited list must match the lookup table size.
        // Bump `golden_count()` and this constant when adding goldens.
        let known_prompts = [
            "set up postgresql with pgvector",
            "configure postgresql service",
            "enable nginx web server",
            "enable redis cache server",
            "enable docker and add my user to the docker group",
            "set up ipfs kubo node",
            "configure nvidia gpu drivers",
            "set up sway window manager",
            "enable kde plasma desktop environment",
            "open firewall ports 80 and 443",
            "set up a rust dev environment with rust-analyzer and mold",
            "rust dev shell with sccache and openssl",
            "set up a node development environment with typescript",
        ];
        assert_eq!(known_prompts.len(), golden_count());
        for p in known_prompts {
            assert!(golden_for(p).is_some(), "{} should be golden-backed", p);
        }
    }
}
