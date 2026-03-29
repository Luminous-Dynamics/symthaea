// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Load Testing Framework
//!
//! Performance and stress testing for Mycelix Mail

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Semaphore;

/// Load test configuration
#[derive(Debug, Clone)]
pub struct LoadTestConfig {
    /// Total number of requests to make
    pub total_requests: u64,
    /// Maximum concurrent requests
    pub concurrency: usize,
    /// Target requests per second (0 = unlimited)
    pub rps_limit: u64,
    /// Timeout per request
    pub timeout: Duration,
    /// Warm-up period before measuring
    pub warmup_duration: Duration,
}

impl Default for LoadTestConfig {
    fn default() -> Self {
        Self {
            total_requests: 1000,
            concurrency: 50,
            rps_limit: 0,
            timeout: Duration::from_secs(30),
            warmup_duration: Duration::from_secs(5),
        }
    }
}

/// Results from a load test run
#[derive(Debug, Clone)]
pub struct LoadTestResults {
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub total_duration: Duration,
    pub requests_per_second: f64,
    pub latency_p50: Duration,
    pub latency_p95: Duration,
    pub latency_p99: Duration,
    pub latency_avg: Duration,
    pub latency_min: Duration,
    pub latency_max: Duration,
    pub errors: Vec<String>,
}

/// Individual request result
#[derive(Debug, Clone)]
pub struct RequestResult {
    pub success: bool,
    pub latency: Duration,
    pub status_code: Option<u16>,
    pub error: Option<String>,
}

/// Load test runner
pub struct LoadTestRunner {
    config: LoadTestConfig,
    results: Arc<tokio::sync::Mutex<Vec<RequestResult>>>,
    completed: Arc<AtomicU64>,
}

impl LoadTestRunner {
    pub fn new(config: LoadTestConfig) -> Self {
        Self {
            config,
            results: Arc::new(tokio::sync::Mutex::new(Vec::new())),
            completed: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Run load test with custom request function
    pub async fn run<F, Fut>(&self, request_fn: F) -> LoadTestResults
    where
        F: Fn(u64) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = RequestResult> + Send,
    {
        let semaphore = Arc::new(Semaphore::new(self.config.concurrency));
        let request_fn = Arc::new(request_fn);
        let start = Instant::now();

        // Warm-up period
        if self.config.warmup_duration > Duration::ZERO {
            tokio::time::sleep(self.config.warmup_duration).await;
        }

        let measurement_start = Instant::now();
        let mut handles = Vec::new();

        for i in 0..self.config.total_requests {
            let permit = semaphore.clone().acquire_owned().await.unwrap();
            let results = self.results.clone();
            let completed = self.completed.clone();
            let request_fn = request_fn.clone();
            let timeout = self.config.timeout;

            let handle = tokio::spawn(async move {
                let req_start = Instant::now();

                let result = tokio::time::timeout(timeout, request_fn(i)).await;

                let request_result = match result {
                    Ok(r) => r,
                    Err(_) => RequestResult {
                        success: false,
                        latency: req_start.elapsed(),
                        status_code: None,
                        error: Some("Request timeout".to_string()),
                    },
                };

                results.lock().await.push(request_result);
                completed.fetch_add(1, Ordering::Relaxed);
                drop(permit);
            });

            handles.push(handle);

            // Rate limiting
            if self.config.rps_limit > 0 {
                let delay = Duration::from_secs_f64(1.0 / self.config.rps_limit as f64);
                tokio::time::sleep(delay).await;
            }
        }

        // Wait for all requests to complete
        for handle in handles {
            let _ = handle.await;
        }

        let total_duration = measurement_start.elapsed();
        self.calculate_results(total_duration).await
    }

    async fn calculate_results(&self, total_duration: Duration) -> LoadTestResults {
        let results = self.results.lock().await;

        let mut latencies: Vec<Duration> = results.iter().map(|r| r.latency).collect();
        latencies.sort();

        let successful = results.iter().filter(|r| r.success).count() as u64;
        let failed = results.len() as u64 - successful;

        let errors: Vec<String> = results
            .iter()
            .filter_map(|r| r.error.clone())
            .take(10)
            .collect();

        let latency_sum: Duration = latencies.iter().sum();
        let latency_avg = if !latencies.is_empty() {
            latency_sum / latencies.len() as u32
        } else {
            Duration::ZERO
        };

        let percentile = |p: f64| -> Duration {
            if latencies.is_empty() {
                Duration::ZERO
            } else {
                let idx = ((latencies.len() as f64 * p) as usize).min(latencies.len() - 1);
                latencies[idx]
            }
        };

        LoadTestResults {
            total_requests: results.len() as u64,
            successful_requests: successful,
            failed_requests: failed,
            total_duration,
            requests_per_second: results.len() as f64 / total_duration.as_secs_f64(),
            latency_p50: percentile(0.50),
            latency_p95: percentile(0.95),
            latency_p99: percentile(0.99),
            latency_avg,
            latency_min: latencies.first().copied().unwrap_or(Duration::ZERO),
            latency_max: latencies.last().copied().unwrap_or(Duration::ZERO),
            errors,
        }
    }
}

// ============================================================================
// Load Test Scenarios
// ============================================================================

/// Standard load test scenarios
pub mod scenarios {
    use super::*;

    /// Email listing endpoint load test
    pub async fn email_list_load_test(base_url: &str, token: &str, config: LoadTestConfig) -> LoadTestResults {
        let client = reqwest::Client::new();
        let url = format!("{}/api/emails", base_url);
        let token = token.to_string();

        let runner = LoadTestRunner::new(config);

        runner.run(move |_i| {
            let client = client.clone();
            let url = url.clone();
            let token = token.clone();

            async move {
                let start = Instant::now();
                let response = client
                    .get(&url)
                    .header("Authorization", format!("Bearer {}", token))
                    .send()
                    .await;

                match response {
                    Ok(resp) => RequestResult {
                        success: resp.status().is_success(),
                        latency: start.elapsed(),
                        status_code: Some(resp.status().as_u16()),
                        error: None,
                    },
                    Err(e) => RequestResult {
                        success: false,
                        latency: start.elapsed(),
                        status_code: None,
                        error: Some(e.to_string()),
                    },
                }
            }
        }).await
    }

    /// Email search endpoint load test
    pub async fn search_load_test(base_url: &str, token: &str, config: LoadTestConfig) -> LoadTestResults {
        let client = reqwest::Client::new();
        let url = format!("{}/api/search", base_url);
        let token = token.to_string();

        let queries = vec![
            "important",
            "meeting",
            "urgent",
            "project",
            "deadline",
        ];

        let runner = LoadTestRunner::new(config);

        runner.run(move |i| {
            let client = client.clone();
            let url = url.clone();
            let token = token.clone();
            let query = queries[i as usize % queries.len()].to_string();

            async move {
                let start = Instant::now();
                let response = client
                    .get(&url)
                    .query(&[("q", &query)])
                    .header("Authorization", format!("Bearer {}", token))
                    .send()
                    .await;

                match response {
                    Ok(resp) => RequestResult {
                        success: resp.status().is_success(),
                        latency: start.elapsed(),
                        status_code: Some(resp.status().as_u16()),
                        error: None,
                    },
                    Err(e) => RequestResult {
                        success: false,
                        latency: start.elapsed(),
                        status_code: None,
                        error: Some(e.to_string()),
                    },
                }
            }
        }).await
    }

    /// Trust score calculation load test
    pub async fn trust_score_load_test(base_url: &str, token: &str, agent_ids: Vec<String>, config: LoadTestConfig) -> LoadTestResults {
        let client = reqwest::Client::new();
        let base = base_url.to_string();
        let token = token.to_string();

        let runner = LoadTestRunner::new(config);

        runner.run(move |i| {
            let client = client.clone();
            let base = base.clone();
            let token = token.clone();
            let agent_id = agent_ids[i as usize % agent_ids.len()].clone();

            async move {
                let url = format!("{}/api/trust/{}/score", base, agent_id);
                let start = Instant::now();

                let response = client
                    .get(&url)
                    .header("Authorization", format!("Bearer {}", token))
                    .send()
                    .await;

                match response {
                    Ok(resp) => RequestResult {
                        success: resp.status().is_success(),
                        latency: start.elapsed(),
                        status_code: Some(resp.status().as_u16()),
                        error: None,
                    },
                    Err(e) => RequestResult {
                        success: false,
                        latency: start.elapsed(),
                        status_code: None,
                        error: Some(e.to_string()),
                    },
                }
            }
        }).await
    }

    /// Mixed workload simulation
    pub async fn mixed_workload_test(base_url: &str, token: &str, config: LoadTestConfig) -> LoadTestResults {
        let client = reqwest::Client::new();
        let base = base_url.to_string();
        let token = token.to_string();

        let runner = LoadTestRunner::new(config);

        runner.run(move |i| {
            let client = client.clone();
            let base = base.clone();
            let token = token.clone();

            async move {
                // Mix of different endpoints (simulating real usage)
                let endpoint = match i % 10 {
                    0..=4 => format!("{}/api/emails", base),           // 50% list emails
                    5..=6 => format!("{}/api/search?q=test", base),    // 20% search
                    7 => format!("{}/api/contacts", base),             // 10% contacts
                    8 => format!("{}/api/labels", base),               // 10% labels
                    _ => format!("{}/api/settings", base),             // 10% settings
                };

                let start = Instant::now();
                let response = client
                    .get(&endpoint)
                    .header("Authorization", format!("Bearer {}", token))
                    .send()
                    .await;

                match response {
                    Ok(resp) => RequestResult {
                        success: resp.status().is_success(),
                        latency: start.elapsed(),
                        status_code: Some(resp.status().as_u16()),
                        error: None,
                    },
                    Err(e) => RequestResult {
                        success: false,
                        latency: start.elapsed(),
                        status_code: None,
                        error: Some(e.to_string()),
                    },
                }
            }
        }).await
    }
}

/// Print load test results in a formatted way
pub fn print_results(name: &str, results: &LoadTestResults) {
    println!("\n{'='*60}");
    println!("Load Test Results: {}", name);
    println!("{'='*60}");
    println!("Total Requests:    {}", results.total_requests);
    println!("Successful:        {} ({:.1}%)",
        results.successful_requests,
        (results.successful_requests as f64 / results.total_requests as f64) * 100.0
    );
    println!("Failed:            {}", results.failed_requests);
    println!("Duration:          {:.2}s", results.total_duration.as_secs_f64());
    println!("Requests/sec:      {:.2}", results.requests_per_second);
    println!("\nLatency:");
    println!("  Min:    {:?}", results.latency_min);
    println!("  Avg:    {:?}", results.latency_avg);
    println!("  P50:    {:?}", results.latency_p50);
    println!("  P95:    {:?}", results.latency_p95);
    println!("  P99:    {:?}", results.latency_p99);
    println!("  Max:    {:?}", results.latency_max);

    if !results.errors.is_empty() {
        println!("\nSample Errors:");
        for error in &results.errors {
            println!("  - {}", error);
        }
    }
}
