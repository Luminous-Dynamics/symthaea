// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Performance Benchmarks for Mycelix Mail
//!
//! Micro-benchmarks for critical path operations.
//! Run with: cargo bench

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use std::time::Duration;

// Mock types for benchmarking - replace with actual imports
mod mock {
    use std::collections::HashMap;

    pub struct Email {
        pub id: String,
        pub from: String,
        pub to: Vec<String>,
        pub subject: String,
        pub body: String,
    }

    pub fn generate_email(size: usize) -> Email {
        Email {
            id: uuid::Uuid::new_v4().to_string(),
            from: "sender@example.com".to_string(),
            to: vec!["recipient@example.com".to_string()],
            subject: "Benchmark Test Email".to_string(),
            body: "x".repeat(size),
        }
    }

    pub fn calculate_trust_score(
        sender: &str,
        existing_interactions: u32,
        vouches: u32,
    ) -> f64 {
        let base_score = 0.3;
        let interaction_bonus = (existing_interactions as f64 * 0.01).min(0.3);
        let vouch_bonus = (vouches as f64 * 0.05).min(0.4);
        (base_score + interaction_bonus + vouch_bonus).min(1.0)
    }

    pub fn categorize_email(subject: &str, body: &str, sender: &str) -> &'static str {
        let content = format!("{} {}", subject, body).to_lowercase();

        if content.contains("unsubscribe") || content.contains("% off") {
            "promotions"
        } else if sender.contains("facebook") || sender.contains("twitter") {
            "social"
        } else if content.contains("order") || content.contains("shipped") {
            "updates"
        } else {
            "primary"
        }
    }

    pub fn encrypt_content(content: &[u8], key: &[u8]) -> Vec<u8> {
        // Simulated encryption (XOR for benchmark purposes)
        content.iter().zip(key.iter().cycle()).map(|(a, b)| a ^ b).collect()
    }

    pub fn parse_email_headers(raw: &str) -> HashMap<String, String> {
        let mut headers = HashMap::new();
        for line in raw.lines() {
            if let Some(idx) = line.find(':') {
                let key = line[..idx].trim().to_string();
                let value = line[idx + 1..].trim().to_string();
                headers.insert(key, value);
            }
        }
        headers
    }

    pub fn search_emails(query: &str, corpus: &[Email]) -> Vec<&Email> {
        let query_lower = query.to_lowercase();
        corpus.iter()
            .filter(|e| {
                e.subject.to_lowercase().contains(&query_lower) ||
                e.body.to_lowercase().contains(&query_lower) ||
                e.from.to_lowercase().contains(&query_lower)
            })
            .collect()
    }
}

fn bench_trust_score_calculation(c: &mut Criterion) {
    let mut group = c.benchmark_group("trust_score");

    for interactions in [0, 10, 50, 100, 500].iter() {
        group.bench_with_input(
            BenchmarkId::new("interactions", interactions),
            interactions,
            |b, &interactions| {
                b.iter(|| {
                    mock::calculate_trust_score(
                        black_box("sender@example.com"),
                        black_box(interactions),
                        black_box(5),
                    )
                });
            },
        );
    }

    group.finish();
}

fn bench_email_categorization(c: &mut Criterion) {
    let mut group = c.benchmark_group("categorization");

    let test_cases = vec![
        ("short", "Meeting tomorrow", "Let's meet"),
        ("medium", "Weekly Newsletter - Special Offer!", "Check out our deals. Unsubscribe here."),
        ("long", "Project Update", &"x".repeat(10000)),
    ];

    for (name, subject, body) in test_cases {
        group.bench_function(name, |b| {
            b.iter(|| {
                mock::categorize_email(
                    black_box(subject),
                    black_box(body),
                    black_box("sender@example.com"),
                )
            });
        });
    }

    group.finish();
}

fn bench_encryption(c: &mut Criterion) {
    let mut group = c.benchmark_group("encryption");
    let key = b"0123456789abcdef0123456789abcdef";

    for size in [1024, 10240, 102400, 1048576].iter() {
        group.throughput(Throughput::Bytes(*size as u64));
        group.bench_with_input(
            BenchmarkId::new("content_size", size),
            size,
            |b, &size| {
                let content = vec![0u8; size];
                b.iter(|| mock::encrypt_content(black_box(&content), black_box(key)));
            },
        );
    }

    group.finish();
}

fn bench_header_parsing(c: &mut Criterion) {
    let mut group = c.benchmark_group("header_parsing");

    let small_headers = "From: sender@example.com\nTo: recipient@example.com\nSubject: Test";

    let large_headers = format!(
        "From: sender@example.com\n\
         To: recipient@example.com\n\
         Cc: cc1@example.com, cc2@example.com\n\
         Subject: Test Email with Many Headers\n\
         Date: Mon, 1 Jan 2024 12:00:00 +0000\n\
         Message-ID: <unique-id@example.com>\n\
         MIME-Version: 1.0\n\
         Content-Type: multipart/mixed; boundary=boundary123\n\
         X-Priority: 1\n\
         X-Mailer: Mycelix Mail\n\
         X-Custom-Header-1: value1\n\
         X-Custom-Header-2: value2\n\
         X-Custom-Header-3: value3\n\
         {}",
        (0..50).map(|i| format!("X-Extra-{}: value{}", i, i)).collect::<Vec<_>>().join("\n")
    );

    group.bench_function("small", |b| {
        b.iter(|| mock::parse_email_headers(black_box(small_headers)));
    });

    group.bench_function("large", |b| {
        b.iter(|| mock::parse_email_headers(black_box(&large_headers)));
    });

    group.finish();
}

fn bench_email_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("search");
    group.sample_size(50);
    group.measurement_time(Duration::from_secs(10));

    // Generate corpus of different sizes
    for corpus_size in [100, 1000, 10000].iter() {
        let corpus: Vec<mock::Email> = (0..*corpus_size)
            .map(|i| mock::Email {
                id: format!("email-{}", i),
                from: format!("sender{}@example.com", i % 100),
                to: vec![format!("recipient{}@example.com", i % 50)],
                subject: format!("Subject {} with keyword{}", i, if i % 10 == 0 { " important" } else { "" }),
                body: format!("Body content {} {}", i, if i % 5 == 0 { "meeting" } else { "general" }),
            })
            .collect();

        group.throughput(Throughput::Elements(*corpus_size as u64));

        group.bench_with_input(
            BenchmarkId::new("corpus_size", corpus_size),
            &corpus,
            |b, corpus| {
                b.iter(|| mock::search_emails(black_box("important"), black_box(corpus)));
            },
        );
    }

    group.finish();
}

fn bench_email_serialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("serialization");

    for size in [1024, 10240, 102400].iter() {
        let email = mock::generate_email(*size);

        group.throughput(Throughput::Bytes(*size as u64));

        group.bench_with_input(
            BenchmarkId::new("json_serialize", size),
            &email,
            |b, email| {
                b.iter(|| {
                    serde_json::to_string(&serde_json::json!({
                        "id": &email.id,
                        "from": &email.from,
                        "to": &email.to,
                        "subject": &email.subject,
                        "body": &email.body,
                    }))
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_trust_score_calculation,
    bench_email_categorization,
    bench_encryption,
    bench_header_parsing,
    bench_email_search,
    bench_email_serialization,
);

criterion_main!(benches);
