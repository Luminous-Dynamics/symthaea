// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Cache speedup bench — runs the NixEval prompt corpus twice (cold + warm)
//! against an isolated `LearnedIdiomCache` and reports:
//!   - cold/warm latency per problem
//!   - cache hit rate on the warm pass
//!   - end-to-end speedup
//!
//! The cache lives in a tempfile that's deleted when the bench finishes so
//! the user's real ~/.cache/symthaea/learned-idioms.json is untouched.
//!
//! Usage:
//!   cargo run --release --features code_generation --example nix_self_improve_bench

use std::path::PathBuf;
use std::time::Instant;

use symthaea::language::learned_idioms::LearnedIdiomCache;
use symthaea::language::nix_codegen::{
    classify_nix_intent, generate_nix_with_cache, NixIntent, SelfImproveSource,
};

/// Trimmed corpus — the bottleneck is `try_nix_eval` (~100s per call
/// because each one re-imports nixpkgs from scratch), so a small but
/// representative set keeps the bench under ~15 min cold while still
/// giving a clean cold-vs-warm signal.
///
/// Layout: 5 originals (cold pass records into cache) + 5 paraphrases
/// (warm pass should hit cache via Jaccard matching).
const PROMPTS: &[&str] = &[
    // Originals
    "set up a rust dev environment with rust-analyzer and mold",
    "set up postgresql with pgvector",
    "enable docker and add my user to the docker group",
    "open firewall port 22000",
    "encrypted secrets with sops-nix",
    // Paraphrases of the above
    "I need a rust development environment with rust-analyzer and the mold linker",
    "install postgresql and pgvector for vector search",
    "I want to run docker containers on this machine",
    "open firewall ports for HTTP and HTTPS",
    "encrypted secrets with sops",
];

#[derive(Default, Debug, Clone)]
struct PassStats {
    total_us: u128,
    cache_hits: usize,
    fresh_recorded: usize,
    fresh_not_recorded: usize,
    parses: usize,
}

impl PassStats {
    fn record(&mut self, elapsed_us: u128, src: SelfImproveSource, parses: bool) {
        self.total_us += elapsed_us;
        if parses {
            self.parses += 1;
        }
        match src {
            SelfImproveSource::LearnedCache => self.cache_hits += 1,
            SelfImproveSource::FreshlyGenerated { recorded: true } => self.fresh_recorded += 1,
            SelfImproveSource::FreshlyGenerated { recorded: false } => self.fresh_not_recorded += 1,
        }
    }
}

fn intent_str(i: NixIntent) -> &'static str {
    match i {
        NixIntent::DevShell => "DevShell",
        NixIntent::Service => "Service",
        NixIntent::Hardware => "Hardware",
        NixIntent::Desktop => "Desktop",
        NixIntent::User => "User",
        NixIntent::HomeManager => "HomeManager",
        NixIntent::Networking => "Networking",
        NixIntent::Secrets => "Secrets",
        NixIntent::FlakeTemplate => "FlakeTemplate",
        NixIntent::Generic => "Generic",
    }
}

fn run_pass(label: &str, cache: &LearnedIdiomCache) -> PassStats {
    println!("\n┌─────────────────────────────────────────────────────────");
    println!("│ {label} pass — {} prompts", PROMPTS.len());
    println!("└─────────────────────────────────────────────────────────");
    let mut stats = PassStats::default();
    for (i, prompt) in PROMPTS.iter().enumerate() {
        let intent = intent_str(classify_nix_intent(&prompt.to_lowercase()));
        let t0 = Instant::now();
        let (result, src) = generate_nix_with_cache(prompt, 3, cache);
        let elapsed_us = t0.elapsed().as_micros();
        stats.record(elapsed_us, src, result.base.parses);
        let mark = match src {
            SelfImproveSource::LearnedCache => "●",
            SelfImproveSource::FreshlyGenerated { recorded: true } => "○",
            SelfImproveSource::FreshlyGenerated { recorded: false } => "?",
        };
        let status = if result.base.parses { "✓" } else { "✗" };
        println!(
            "  {mark} #{:>2} {status} [{:>10}] {:>7}µs  {:.55}",
            i + 1,
            intent,
            elapsed_us,
            prompt
        );
    }
    stats
}

fn main() {
    println!("┌─────────────────────────────────────────────────────────");
    println!("│ Symthaea Nix codegen — self-improve cache speedup bench");
    println!("│ Pipeline: idiom-library + nixpkgs option index + Tier 3 cache");
    println!("└─────────────────────────────────────────────────────────");
    println!("Legend: ● cache hit  ○ fresh+recorded  ? fresh+not-recorded");

    // Per-run isolated cache so the user's real one isn't polluted.
    let cache_path: PathBuf = std::env::temp_dir().join(format!(
        "symthaea_self_improve_bench_{}_{}.json",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    let _ = std::fs::remove_file(&cache_path);
    let cache = LearnedIdiomCache::with_cache_path(cache_path.clone());
    println!("Cache file: {}", cache_path.display());

    let cold = run_pass("COLD", &cache);
    println!("\n  Cold cache size after: {} idioms recorded", cache.len());

    let warm = run_pass("WARM", &cache);

    let n = PROMPTS.len() as f64;
    let cold_avg_ms = cold.total_us as f64 / 1_000.0 / n;
    let warm_avg_ms = warm.total_us as f64 / 1_000.0 / n;
    let speedup = if warm.total_us > 0 {
        cold.total_us as f64 / warm.total_us as f64
    } else {
        f64::INFINITY
    };
    let warm_hit_rate = warm.cache_hits as f64 / n * 100.0;

    println!("\n╔═════════════════════════════════════════════════════════");
    println!("║ Results ({} prompts)", PROMPTS.len());
    println!("╠═════════════════════════════════════════════════════════");
    println!(
        "║ Cold:   total {:>7.1}ms  avg {:>6.2}ms  hits  {:>2}  fresh+rec {:>2}  fresh-not {:>2}  parses {:>2}/{}",
        cold.total_us as f64 / 1_000.0,
        cold_avg_ms,
        cold.cache_hits,
        cold.fresh_recorded,
        cold.fresh_not_recorded,
        cold.parses,
        PROMPTS.len()
    );
    println!(
        "║ Warm:   total {:>7.1}ms  avg {:>6.2}ms  hits  {:>2}  fresh+rec {:>2}  fresh-not {:>2}  parses {:>2}/{}",
        warm.total_us as f64 / 1_000.0,
        warm_avg_ms,
        warm.cache_hits,
        warm.fresh_recorded,
        warm.fresh_not_recorded,
        warm.parses,
        PROMPTS.len()
    );
    println!("║");
    println!("║ Cache hit rate (warm): {:.1}%", warm_hit_rate);
    println!("║ End-to-end speedup:    {:.2}× (cold→warm)", speedup);
    println!("╚═════════════════════════════════════════════════════════");

    // Clean up the bench cache.
    let _ = std::fs::remove_file(&cache_path);
}
