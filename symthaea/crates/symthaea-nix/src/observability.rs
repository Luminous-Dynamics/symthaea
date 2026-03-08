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

/// Errors that can occur during metrics initialization.
#[derive(Debug)]
pub struct MetricsInitError(String);

impl std::fmt::Display for MetricsInitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "metrics initialization failed: {}", self.0)
    }
}

impl std::error::Error for MetricsInitError {}

static METRICS: OnceLock<Metrics> = OnceLock::new();

impl Metrics {
    /// Try to create a new Metrics instance, returning an error if prometheus
    /// registration fails (e.g. duplicate metric names in the same process).
    fn try_new() -> Result<Metrics, MetricsInitError> {
        let phase_buckets = prometheus::exponential_buckets(0.0001, 2.0, 15)
            .map_err(|e| MetricsInitError(format!("histogram buckets: {e}")))?;

        Ok(Metrics {
            consciousness_cycles_total: register_counter!(
                "consciousness_cycles_total",
                "Total number of consciousness daemon cycles completed"
            )
            .map_err(|e| MetricsInitError(format!("consciousness_cycles_total: {e}")))?,

            phase_duration_seconds: register_histogram_vec!(
                "phase_duration_seconds",
                "Duration of each pipeline phase in seconds",
                &["phase"],
                phase_buckets
            )
            .map_err(|e| MetricsInitError(format!("phase_duration_seconds: {e}")))?,

            consciousness_level: register_gauge!(
                "consciousness_level",
                "Current consciousness level (0.0-1.0)"
            )
            .map_err(|e| MetricsInitError(format!("consciousness_level: {e}")))?,

            phi_value: register_gauge!(
                "phi_value",
                "Current Phi (integrated information) value"
            )
            .map_err(|e| MetricsInitError(format!("phi_value: {e}")))?,

            gate_vetoes_total: register_counter!(
                "gate_vetoes_total",
                "Total number of times the Phi gate vetoed action execution"
            )
            .map_err(|e| MetricsInitError(format!("gate_vetoes_total: {e}")))?,

            free_energy: register_gauge!(
                "free_energy",
                "Current free energy of the world model"
            )
            .map_err(|e| MetricsInitError(format!("free_energy: {e}")))?,

            anomalies_total: register_counter!(
                "anomalies_total",
                "Total anomalies detected from journal analysis"
            )
            .map_err(|e| MetricsInitError(format!("anomalies_total: {e}")))?,

            causal_edge_count: register_gauge!(
                "causal_edge_count",
                "Number of edges in the causal graph"
            )
            .map_err(|e| MetricsInitError(format!("causal_edge_count: {e}")))?,

            episodic_count: register_gauge!(
                "episodic_count",
                "Number of episodes in episodic memory"
            )
            .map_err(|e| MetricsInitError(format!("episodic_count: {e}")))?,
        })
    }

    /// Get or initialize the global metrics instance.
    ///
    /// Panics only if prometheus registration fails — which indicates a
    /// fundamental configuration error (duplicate metric names). In production
    /// use `try_global()` instead.
    pub fn global() -> &'static Metrics {
        METRICS.get_or_init(|| {
            Self::try_new().unwrap_or_else(|e| {
                panic!("Fatal: cannot register prometheus metrics: {e}")
            })
        })
    }

    /// Fallible version of `global()` for contexts where panicking is
    /// unacceptable. Returns `Err` only on the first call if registration
    /// fails; subsequent calls always return `Ok`.
    pub fn try_global() -> Result<&'static Metrics, MetricsInitError> {
        // If already initialized, return immediately.
        if let Some(m) = METRICS.get() {
            return Ok(m);
        }
        let metrics = Self::try_new()?;
        // Another thread might have raced us; that's fine — OnceLock handles it.
        Ok(METRICS.get_or_init(|| metrics))
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
                    if let Err(e) = encoder.encode(&metric_families, &mut buffer) {
                        eprintln!("nix-mind-daemon: metrics encode error: {e}");
                        let resp = Response::builder()
                            .status(500)
                            .body(Full::new(Bytes::from("Internal Server Error")))
                            .unwrap_or_else(|_| Response::new(Full::new(Bytes::from("error"))));
                        return Ok::<_, hyper::Error>(resp);
                    }
                    let resp = Response::builder()
                        .header("Content-Type", encoder.format_type())
                        .body(Full::new(Bytes::from(buffer)))
                        .unwrap_or_else(|_| Response::new(Full::new(Bytes::from("error"))));
                    Ok::<_, hyper::Error>(resp)
                } else {
                    let resp = Response::builder()
                        .status(404)
                        .body(Full::new(Bytes::from("Not Found")))
                        .unwrap_or_else(|_| Response::new(Full::new(Bytes::from("error"))));
                    Ok(resp)
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

    #[test]
    fn test_try_global_returns_ok() {
        // try_global() should succeed when metrics can be registered
        let result = Metrics::try_global();
        assert!(result.is_ok());
        // Subsequent calls should also succeed
        let result2 = Metrics::try_global();
        assert!(result2.is_ok());
    }

    #[test]
    fn test_try_global_same_instance_as_global() {
        let m1 = Metrics::global();
        let m2 = Metrics::try_global().unwrap();
        assert!(std::ptr::eq(m1, m2));
    }
}
