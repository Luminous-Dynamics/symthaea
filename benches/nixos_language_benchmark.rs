//! Benchmarks for NixOS Language Pipeline
//!
//! Measures performance of:
//! - Intent recognition (natural language → NixOSIntent)
//! - Semantic search (query → relevant packages)
//! - Error diagnosis (error message → actionable fix)
//! - Security analysis (input → threat detection)
//! - Full pipeline end-to-end

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use std::time::Duration;

use symthaea::language::{
    NixOSLanguageAdapter, NixOSIntent,
    NixKnowledgeProvider, PackageInfo, PackageCategory,
    NixErrorDiagnoser,
};
use symthaea::language::nix_security::{SecurityKernel, SecurityConfig};

// ═══════════════════════════════════════════════════════════════════════════
// INTENT RECOGNITION BENCHMARKS
// ═══════════════════════════════════════════════════════════════════════════

fn bench_intent_recognition(c: &mut Criterion) {
    let mut group = c.benchmark_group("intent_recognition");
    group.measurement_time(Duration::from_secs(10));

    let mut adapter = NixOSLanguageAdapter::new();

    // Test prompts of varying complexity
    let prompts = vec![
        ("simple", "install firefox"),
        ("medium", "search for a video editor that supports 4k"),
        ("complex", "create a development environment with rust, python, and nodejs with lsp support"),
        ("ambiguous", "how do I set up my system for gaming?"),
    ];

    for (name, prompt) in prompts {
        group.bench_with_input(
            BenchmarkId::new("understand", name),
            &prompt,
            |b, prompt| {
                b.iter(|| {
                    adapter.understand(black_box(prompt))
                });
            },
        );
    }

    group.finish();
}

fn bench_intent_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("intent_throughput");
    group.measurement_time(Duration::from_secs(15));

    let mut adapter = NixOSLanguageAdapter::new();

    // Batch of 100 diverse prompts
    let prompts: Vec<&str> = vec![
        "install firefox",
        "search vim",
        "remove unused packages",
        "upgrade the system",
        "configure nginx",
        "rollback to previous generation",
        "list installed packages",
        "show package info for rust",
        "garbage collect old generations",
        "create flake for python project",
    ].into_iter()
        .cycle()
        .take(100)
        .collect();

    group.throughput(Throughput::Elements(prompts.len() as u64));

    group.bench_function("batch_100", |b| {
        b.iter(|| {
            for prompt in &prompts {
                black_box(adapter.understand(black_box(prompt)));
            }
        });
    });

    group.finish();
}

// ═══════════════════════════════════════════════════════════════════════════
// SEMANTIC SEARCH BENCHMARKS
// ═══════════════════════════════════════════════════════════════════════════

fn create_large_knowledge_base() -> NixKnowledgeProvider {
    let mut knowledge = NixKnowledgeProvider::new();

    // Simulate 1000 packages (realistic subset)
    let categories = vec![
        ("browser", PackageCategory::Browser, vec!["firefox", "chromium", "brave", "vivaldi", "qutebrowser"]),
        ("editor", PackageCategory::Editor, vec!["vim", "neovim", "emacs", "vscode", "helix", "kakoune"]),
        ("shell", PackageCategory::Shell, vec!["bash", "zsh", "fish", "nushell", "elvish"]),
        ("database", PackageCategory::Database, vec!["postgresql", "mysql", "redis", "mongodb", "sqlite"]),
        ("webserver", PackageCategory::WebServer, vec!["nginx", "apache", "caddy", "traefik", "haproxy"]),
    ];

    for (cat_name, category, packages) in categories {
        for pkg in packages {
            // Create 20 variants per package to simulate real package count
            for i in 0..20 {
                let name = if i == 0 {
                    pkg.to_string()
                } else {
                    format!("{}-{}", pkg, i)
                };

                let info = PackageInfo::new(
                    format!("pkgs.{}", name),
                    name.clone(),
                    "1.0.0",
                    format!("{} - a {} application (variant {})", name, cat_name, i),
                )
                .with_category(category.clone())
                .with_keywords(&[cat_name, "tool"]);

                knowledge.add_package(info);
            }
        }
    }

    knowledge
}

fn bench_semantic_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("semantic_search");
    group.measurement_time(Duration::from_secs(15));

    let queries = vec![
        ("simple", "firefox"),
        ("descriptive", "terminal text editor"),
        ("conceptual", "database for web applications"),
        ("vague", "something for coding"),
    ];

    for (name, query) in queries {
        // Create fresh knowledge base for each benchmark
        let mut knowledge = create_large_knowledge_base();
        group.bench_with_input(
            BenchmarkId::new("search", name),
            &query,
            |b, query| {
                b.iter(|| {
                    let results = knowledge.semantic_search(black_box(query), 10);
                    // Convert to owned data to avoid lifetime issues
                    black_box(results.len())
                });
            },
        );
    }

    group.finish();
}

fn bench_semantic_search_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("semantic_search_scaling");
    group.measurement_time(Duration::from_secs(20));

    for k in [1, 5, 10, 25, 50, 100] {
        // Create fresh knowledge base for each benchmark
        let mut knowledge = create_large_knowledge_base();
        group.bench_with_input(
            BenchmarkId::new("top_k", k),
            &k,
            |b, k| {
                b.iter(|| {
                    let results = knowledge.semantic_search(black_box("text editor"), *k);
                    black_box(results.len())
                });
            },
        );
    }

    group.finish();
}

// ═══════════════════════════════════════════════════════════════════════════
// ERROR DIAGNOSIS BENCHMARKS
// ═══════════════════════════════════════════════════════════════════════════

fn bench_error_diagnosis(c: &mut Criterion) {
    let mut group = c.benchmark_group("error_diagnosis");
    group.measurement_time(Duration::from_secs(10));

    let diagnoser = NixErrorDiagnoser::new();

    let errors = vec![
        ("evaluation", r#"
            error: attribute 'nonexistent' missing
               at /etc/nixos/configuration.nix:15:5:
               15|   environment.systemPackages = [ pkgs.nonexistent ];
        "#),
        ("build", r#"
            error: builder for '/nix/store/xxx-package.drv' failed with exit code 1
            ld: cannot find -lfoo: No such file or directory
        "#),
        ("permission", r#"
            error: cannot link '/nix/store/xxx' to '/nix/store/yyy': Permission denied
        "#),
        ("hash_mismatch", r#"
            error: hash mismatch in fixed-output derivation '/nix/store/xxx.drv':
              specified: sha256-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=
              got:       sha256-BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB=
        "#),
    ];

    for (name, error) in errors {
        group.bench_with_input(
            BenchmarkId::new("diagnose", name),
            &error,
            |b, error| {
                b.iter(|| {
                    diagnoser.diagnose(black_box(error))
                });
            },
        );
    }

    group.finish();
}

// ═══════════════════════════════════════════════════════════════════════════
// SECURITY ANALYSIS BENCHMARKS
// ═══════════════════════════════════════════════════════════════════════════

fn bench_security_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("security_analysis");
    group.measurement_time(Duration::from_secs(10));

    let config = SecurityConfig::default();
    let mut security = SecurityKernel::new(config);

    let inputs = vec![
        ("clean", "install firefox and setup my development environment"),
        ("with_path", "cat /etc/passwd and show users"),
        ("with_secret", "API_KEY=sk_live_abc123xyz9876543210000"),
        ("mixed", "run sudo rm -rf / --no-preserve-root with GITHUB_TOKEN=ghp_xxxx"),
    ];

    for (name, input) in inputs {
        group.bench_with_input(
            BenchmarkId::new("analyze", name),
            &input,
            |b, input| {
                b.iter(|| {
                    security.analyze(black_box(input))
                });
            },
        );
    }

    group.finish();
}

fn bench_security_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("security_throughput");
    group.measurement_time(Duration::from_secs(15));

    let config = SecurityConfig::default();
    let mut security = SecurityKernel::new(config);

    // Simulate 1000 inputs for throughput testing
    let inputs: Vec<String> = (0..1000)
        .map(|i| format!("install package-{} and configure service-{}", i, i))
        .collect();

    group.throughput(Throughput::Elements(inputs.len() as u64));

    group.bench_function("batch_1000", |b| {
        b.iter(|| {
            for input in &inputs {
                black_box(security.analyze(black_box(input)));
            }
        });
    });

    group.finish();
}

// ═══════════════════════════════════════════════════════════════════════════
// FULL PIPELINE BENCHMARKS
// ═══════════════════════════════════════════════════════════════════════════

fn bench_full_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_pipeline");
    group.measurement_time(Duration::from_secs(20));

    // Initialize all components
    let mut adapter = NixOSLanguageAdapter::new();
    let mut knowledge = create_large_knowledge_base();
    let diagnoser = NixErrorDiagnoser::new();
    let config = SecurityConfig::default();
    let mut security = SecurityKernel::new(config);

    // Simulate complete user interactions
    let interactions = vec![
        ("install_flow", vec![
            "install firefox",
            "searching for firefox...",
            "firefox - Web browser",
        ]),
        ("search_flow", vec![
            "find a video editor",
            "searching...",
            "kdenlive - Video editor",
        ]),
        ("error_flow", vec![
            "why did my build fail?",
            "error: attribute missing",
            "Package not found. Try: nix search",
        ]),
    ];

    for (name, steps) in interactions {
        group.bench_with_input(
            BenchmarkId::new("interaction", name),
            &steps,
            |b, steps| {
                b.iter(|| {
                    // Step 1: Security check
                    let analysis = security.analyze(black_box(steps[0]));

                    // Step 2: Full understanding (intent + frames + actions)
                    let understanding = adapter.understand(black_box(steps[0]));

                    // Step 3: Semantic search (if applicable)
                    if matches!(understanding.intent, NixOSIntent::Search | NixOSIntent::Install) {
                        let _results = knowledge.semantic_search(steps[0], 5);
                    }

                    // Step 4: Error diagnosis (if applicable)
                    if steps[1].contains("error") {
                        let _diagnosis = diagnoser.diagnose(steps[1]);
                    }

                    black_box((analysis, understanding))
                });
            },
        );
    }

    group.finish();
}

// ═══════════════════════════════════════════════════════════════════════════
// HV16 EMBEDDING BENCHMARKS
// ═══════════════════════════════════════════════════════════════════════════

fn bench_hv16_operations(c: &mut Criterion) {
    use symthaea::hdc::binary_hv::HV16;
    use symthaea::hdc::deterministic_seeds::seed_from_name;

    let mut group = c.benchmark_group("hv16_embeddings");
    group.measurement_time(Duration::from_secs(10));

    // Benchmark vector creation
    group.bench_function("create_from_seed", |b| {
        b.iter(|| {
            let seed = seed_from_name("firefox-browser");
            black_box(HV16::random(seed))
        });
    });

    // Benchmark similarity computation
    let v1 = HV16::random(seed_from_name("firefox"));
    let v2 = HV16::random(seed_from_name("chromium"));

    group.bench_function("similarity", |b| {
        b.iter(|| {
            black_box(v1.similarity(&v2))
        });
    });

    // Benchmark XOR composition
    group.bench_function("xor_compose", |b| {
        b.iter(|| {
            let mut v = HV16::random(seed_from_name("test"));
            for (a, b_val) in v.0.iter_mut().zip(v2.0.iter()) {
                *a ^= *b_val;
            }
            black_box(v)
        });
    });

    // Benchmark batch similarity (realistic for search)
    let query = HV16::random(seed_from_name("editor"));
    let corpus: Vec<HV16> = (0..1000)
        .map(|i| HV16::random(seed_from_name(&format!("package-{}", i))))
        .collect();

    group.bench_function("batch_similarity_1000", |b| {
        b.iter(|| {
            let mut results: Vec<f32> = corpus.iter()
                .map(|v| query.similarity(v))
                .collect();
            results.sort_by(|a, b| b.partial_cmp(a).unwrap());
            black_box(results.into_iter().take(10).collect::<Vec<_>>())
        });
    });

    group.finish();
}

// ═══════════════════════════════════════════════════════════════════════════
// GÖDEL NUMBER BENCHMARKS
// ═══════════════════════════════════════════════════════════════════════════

fn bench_godel_operations(c: &mut Criterion) {
    use symthaea::hdc::deterministic_seeds::{GodelNumber, NixPrimeConcept};

    let mut group = c.benchmark_group("godel_numbers");
    group.measurement_time(Duration::from_secs(10));

    // Benchmark number composition
    group.bench_function("compose_3_concepts", |b| {
        b.iter(|| {
            let g1 = GodelNumber::from_concept(NixPrimeConcept::Package);
            let g2 = GodelNumber::from_concept(NixPrimeConcept::Enable);
            let g3 = GodelNumber::from_concept(NixPrimeConcept::System);
            let composed = g1.compose(&g2).compose(&g3);
            black_box(composed)
        });
    });

    // Benchmark concept containment check
    let composite = GodelNumber::from_concepts(&[
        NixPrimeConcept::Package,
        NixPrimeConcept::Enable,
        NixPrimeConcept::System,
    ]);
    group.bench_function("check_contains", |b| {
        b.iter(|| {
            let has_package = composite.contains(NixPrimeConcept::Package);
            let has_enable = composite.contains(NixPrimeConcept::Enable);
            black_box((has_package, has_enable))
        });
    });

    group.finish();
}

// ═══════════════════════════════════════════════════════════════════════════
// BENCHMARK GROUPS
// ═══════════════════════════════════════════════════════════════════════════

criterion_group!(
    name = intent_benches;
    config = Criterion::default()
        .significance_level(0.01)
        .sample_size(100);
    targets = bench_intent_recognition, bench_intent_throughput
);

criterion_group!(
    name = search_benches;
    config = Criterion::default()
        .significance_level(0.01)
        .sample_size(50);
    targets = bench_semantic_search, bench_semantic_search_scaling
);

criterion_group!(
    name = diagnostic_benches;
    config = Criterion::default()
        .significance_level(0.01)
        .sample_size(100);
    targets = bench_error_diagnosis
);

criterion_group!(
    name = security_benches;
    config = Criterion::default()
        .significance_level(0.01)
        .sample_size(100);
    targets = bench_security_analysis, bench_security_throughput
);

criterion_group!(
    name = pipeline_benches;
    config = Criterion::default()
        .significance_level(0.01)
        .sample_size(50);
    targets = bench_full_pipeline
);

criterion_group!(
    name = hdc_benches;
    config = Criterion::default()
        .significance_level(0.01)
        .sample_size(200);
    targets = bench_hv16_operations, bench_godel_operations
);

criterion_main!(
    intent_benches,
    search_benches,
    diagnostic_benches,
    security_benches,
    pipeline_benches,
    hdc_benches
);
