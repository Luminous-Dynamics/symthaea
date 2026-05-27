// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! HDC + Active Inference benchmarks for symthaea-nix.

use criterion::{Criterion, black_box, criterion_group, criterion_main};

use symthaea_nix::encoding::{NixCodebook, OptionEncoder, UserInputEncoder};
use symthaea_nix::mind::{NixActiveInference, causal_graph::NixCausalGraph};

/// Benchmark encoding a NixOS option path into HDC space.
/// Target: <1μs
fn bench_option_encoding(c: &mut Criterion) {
    let mut codebook = NixCodebook::new();

    c.bench_function("option_encoding", |b| {
        b.iter(|| {
            let mut encoder = OptionEncoder::new(&mut codebook);
            encoder.encode_path(black_box("services.nginx.enable"))
        })
    });
}

/// Benchmark encoding user input into HDC space.
/// Target: <1μs
fn bench_input_encoding(c: &mut Criterion) {
    let mut codebook = NixCodebook::new();

    c.bench_function("input_encoding", |b| {
        b.iter(|| {
            let mut encoder = UserInputEncoder::new(&mut codebook);
            encoder.encode_input(black_box("install firefox and enable nginx"))
        })
    });
}

/// Benchmark causal graph querying (side effect prediction).
/// Target: <10μs
fn bench_causal_query(c: &mut Criterion) {
    let mut graph = NixCausalGraph::new(42);

    // Seed with structural edges
    for (cause, effect, _) in
        symthaea_nix::mind::causal_graph::NixOSCausalPatterns::known_patterns()
    {
        graph.add_structural_edge(cause, effect, 0.8);
    }

    c.bench_function("causal_query", |b| {
        b.iter(|| graph.predict_side_effects(black_box("services.nginx.enable")))
    });
}

/// Benchmark a full cognition cycle (input → goal inference → action plan).
/// Target: <1ms
fn bench_full_cognition_cycle(c: &mut Criterion) {
    let mut engine = NixActiveInference::new();

    c.bench_function("full_cognition_cycle", |b| {
        b.iter(|| engine.process_input(black_box("install firefox")))
    });
}

/// Benchmark LSH-accelerated codebook search over 200+ entries.
/// Target: <100μs
fn bench_lsh_search(c: &mut Criterion) {
    let mut codebook = NixCodebook::new();

    // Pre-populate codebook with many option-like tokens to exceed LSH_MIN_ENTRIES (32)
    let services = [
        "nginx",
        "postgresql",
        "mysql",
        "redis",
        "mongodb",
        "openssh",
        "fail2ban",
        "grafana",
        "prometheus",
        "caddy",
        "apache",
        "nextcloud",
        "gitea",
        "syncthing",
        "tor",
        "tailscale",
        "k3s",
        "xrdp",
        "dovecot2",
        "postfix",
        "printing",
        "avahi",
        "cron",
        "logrotate",
        "dnsmasq",
        "resolved",
        "flatpak",
        "udisks2",
        "gvfs",
        "blueman",
        "pipewire",
        "xserver",
        "ntp",
        "chrony",
        "timesyncd",
    ];
    for svc in &services {
        // Encode as option paths to create diverse tokens
        let mut encoder = OptionEncoder::new(&mut codebook);
        encoder.encode_path(&format!("services.{}.enable", svc));
    }

    // Build a query vector
    let query_hv = codebook.get_or_create("nginx").clone();

    c.bench_function("lsh_search_200_entries", |b| {
        b.iter(|| codebook.search_similar(black_box(&query_hv), 5))
    });
}

/// Benchmark semantic option search via HDC similarity.
/// Target: <100μs
fn bench_semantic_search(c: &mut Criterion) {
    use symthaea_nix::encoding::search_options;

    let mut codebook = NixCodebook::new();
    let paths = [
        "services.nginx.enable",
        "services.postgresql.enable",
        "services.openssh.enable",
        "networking.firewall.enable",
        "hardware.opengl.enable",
        "boot.loader.grub.enable",
        "users.users.admin.extraGroups",
        "environment.systemPackages",
        "programs.zsh.enable",
        "security.polkit.enable",
    ];
    let path_refs: Vec<&str> = paths.iter().copied().collect();

    c.bench_function("semantic_search_10_paths", |b| {
        b.iter(|| {
            search_options(
                black_box("enable nginx web server"),
                &mut codebook,
                &path_refs,
                5,
            )
        })
    });
}

criterion_group!(
    benches,
    bench_option_encoding,
    bench_input_encoding,
    bench_causal_query,
    bench_full_cognition_cycle,
    bench_lsh_search,
    bench_semantic_search,
);
criterion_main!(benches);
