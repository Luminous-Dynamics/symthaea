//! Production observability for the nix-mind daemon.
//!
//! Provides:
//! - Prometheus metrics (counters, gauges, histograms) for pipeline phases
//! - Structured JSON logging via `tracing-subscriber`
//! - A `/metrics` HTTP endpoint for scraping
//!
//! Feature-gated behind `observability`.

use prometheus::{
    register_counter, register_gauge, register_histogram_vec, Counter, Gauge, HistogramVec,
};
use std::sync::OnceLock;

/// Global metrics registry — initialized once on daemon start.
pub struct Metrics {
    /// Total consciousness cycles completed.
    pub consciousness_cycles_total: Counter,
    /// Duration of each pipeline phase in seconds, labeled by phase name.
    pub phase_duration_seconds: HistogramVec,
    /// Current consciousness level gauge.
    pub consciousness_level: Gauge,
    /// Current Phi value gauge.
    pub phi_value: Gauge,
    /// Total number of gate vetoes.
    pub gate_vetoes_total: Counter,
    /// Current free energy gauge.
    pub free_energy: Gauge,
    /// Total anomalies detected.
    pub anomalies_total: Counter,
    /// Number of causal graph edges.
    pub causal_edge_count: Gauge,
    /// Number of episodic memory entries.
    pub episodic_count: Gauge,
}

static METRICS: OnceLock<Metrics> = OnceLock::new();

impl Metrics {
    /// Get or initialize the global metrics instance.
    pub fn global() -> &'static Metrics {
        METRICS.get_or_init(|| {
            let phase_buckets =
                prometheus::exponential_buckets(0.0001, 2.0, 15).expect("valid histogram buckets");

            Metrics {
                consciousness_cycles_total: register_counter!(
                    "consciousness_cycles_total",
                    "Total number of consciousness daemon cycles completed"
                )
                .expect("register consciousness_cycles_total"),

                phase_duration_seconds: register_histogram_vec!(
                    "phase_duration_seconds",
                    "Duration of each pipeline phase in seconds",
                    &["phase"],
                    phase_buckets
                )
                .expect("register phase_duration_seconds"),

                consciousness_level: register_gauge!(
                    "consciousness_level",
                    "Current consciousness level (0.0-1.0)"
                )
                .expect("register consciousness_level"),

                phi_value: register_gauge!(
                    "phi_value",
                    "Current Phi (integrated information) value"
                )
                .expect("register phi_value"),

                gate_vetoes_total: register_counter!(
                    "gate_vetoes_total",
                    "Total number of times the Phi gate vetoed action execution"
                )
                .expect("register gate_vetoes_total"),

                free_energy: register_gauge!(
                    "free_energy",
                    "Current free energy of the world model"
                )
                .expect("register free_energy"),

                anomalies_total: register_counter!(
                    "anomalies_total",
                    "Total anomalies detected from journal analysis"
                )
                .expect("register anomalies_total"),

                causal_edge_count: register_gauge!(
                    "causal_edge_count",
                    "Number of edges in the causal graph"
                )
                .expect("register causal_edge_count"),

                episodic_count: register_gauge!(
                    "episodic_count",
                    "Number of episodes in episodic memory"
                )
                .expect("register episodic_count"),
            }
        })
    }
}

/// A guard that records the duration of a pipeline phase when dropped.
pub struct PhaseTimer {
    phase: &'static str,
    start: std::time::Instant,
}

impl PhaseTimer {
    /// Start timing a pipeline phase.
    pub fn start(phase: &'static str) -> Self {
        tracing::debug!(phase, "pipeline phase started");
        Self {
            phase,
            start: std::time::Instant::now(),
        }
    }
}

impl Drop for PhaseTimer {
    fn drop(&mut self) {
        let elapsed = self.start.elapsed().as_secs_f64();
        let m = Metrics::global();
        m.phase_duration_seconds
            .with_label_values(&[self.phase])
            .observe(elapsed);
        tracing::debug!(
            phase = self.phase,
            elapsed_ms = elapsed * 1000.0,
            "pipeline phase completed"
        );
    }
}

/// Initialize structured JSON logging via tracing-subscriber.
///
/// Reads `RUST_LOG` env var for filter directives (default: `info`).
pub fn init_tracing() {
    use tracing_subscriber::{fmt, EnvFilter};

    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));

    fmt()
        .json()
        .with_env_filter(filter)
        .with_target(true)
        .with_thread_ids(true)
        .with_file(true)
        .with_line_number(true)
        .init();

    tracing::info!("nix-mind-daemon observability initialized");
}

/// Serve Prometheus metrics on the given port at `/metrics`.
///
/// This spawns a background tokio task. The caller must be inside a tokio runtime.
pub async fn serve_metrics(port: u16) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    use http_body_util::Full;
    use hyper::body::Bytes;
    use hyper::server::conn::http1;
    use hyper::service::service_fn;
    use hyper::{Request, Response};
    use hyper_util::rt::TokioIo;
    use prometheus::Encoder;
    use std::net::SocketAddr;

    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!(port, "Prometheus metrics endpoint listening");

    loop {
        let (stream, _) = listener.accept().await?;
        let io = TokioIo::new(stream);

        tokio::task::spawn(async move {
            let service = service_fn(|req: Request<hyper::body::Incoming>| async move {
                if req.uri().path() == "/metrics" {
                    let encoder = prometheus::TextEncoder::new();
                    let metric_families = prometheus::gather();
                    let mut buffer = Vec::new();
                    encoder.encode(&metric_families, &mut buffer).unwrap();
                    Ok::<_, hyper::Error>(
                        Response::builder()
                            .header("Content-Type", encoder.format_type())
                            .body(Full::new(Bytes::from(buffer)))
                            .unwrap(),
                    )
                } else {
                    Ok(Response::builder()
                        .status(404)
                        .body(Full::new(Bytes::from("Not Found")))
                        .unwrap())
                }
            });

            if let Err(err) = http1::Builder::new().serve_connection(io, service).await {
                tracing::warn!(error = %err, "metrics connection error");
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_initialization() {
        let m = Metrics::global();
        // Verify counters start at 0
        assert_eq!(m.consciousness_cycles_total.get() as u64, 0);
        assert_eq!(m.gate_vetoes_total.get() as u64, 0);
    }

    #[test]
    fn test_phase_timer_records_duration() {
        let m = Metrics::global();
        {
            let _timer = PhaseTimer::start("observe");
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        // After dropping the timer, the histogram should have one observation
        let count = m
            .phase_duration_seconds
            .with_label_values(&["observe"])
            .get_sample_count();
        assert!(count >= 1);
    }

    #[test]
    fn test_gauge_set_and_read() {
        let m = Metrics::global();
        m.consciousness_level.set(0.75);
        assert!((m.consciousness_level.get() - 0.75).abs() < 1e-6);
        m.free_energy.set(0.42);
        assert!((m.free_energy.get() - 0.42).abs() < 1e-6);
    }

    #[test]
    fn test_counter_increment() {
        let m = Metrics::global();
        let before = m.anomalies_total.get() as u64;
        m.anomalies_total.inc();
        assert_eq!(m.anomalies_total.get() as u64, before + 1);
    }
}
