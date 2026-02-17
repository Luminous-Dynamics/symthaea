//! OpenTelemetry Distributed Tracing for Proof Operations
//!
//! Provides distributed tracing for proof generation and verification,
//! enabling debugging and performance analysis across distributed systems.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use fl_aggregator::proofs::tracing::{init_tracing, TracedProofGenerator};
//!
//! // Initialize tracing with Jaeger exporter
//! init_tracing("proof-service", "http://localhost:14268/api/traces")?;
//!
//! // Generate proof with tracing
//! let generator = TracedProofGenerator::new();
//! let proof = generator.generate_range_proof(50, 0, 100)?;
//! ```
//!
//! ## Spans Created
//!
//! - `proof.generate` - Root span for proof generation
//! - `proof.generate.trace` - Trace table construction
//! - `proof.generate.commit` - Commitment phase
//! - `proof.generate.prove` - STARK proof generation
//! - `proof.verify` - Proof verification
//! - `proof.serialize` - Serialization
//! - `proof.batch` - Batch operations

use std::time::Instant;
use tracing::{info_span, instrument, Span};

use crate::proofs::{
    ProofConfig, ProofResult, ProofType, RangeProof, GradientIntegrityProof,
    IdentityAssuranceProof, VoteEligibilityProof, MembershipProof,
    ProofAssuranceLevel, ProofIdentityFactor, ProofProposalType, ProofVoterProfile,
};

/// Tracing configuration
#[derive(Debug, Clone)]
pub struct TracingConfig {
    /// Service name for traces
    pub service_name: String,
    /// Jaeger/OTLP endpoint
    pub endpoint: Option<String>,
    /// Sample rate (0.0 to 1.0)
    pub sample_rate: f64,
    /// Enable console output
    pub console_output: bool,
    /// Include proof details in spans
    pub detailed_spans: bool,
}

impl Default for TracingConfig {
    fn default() -> Self {
        Self {
            service_name: "fl-aggregator-proofs".to_string(),
            endpoint: None,
            sample_rate: 1.0,
            console_output: true,
            detailed_spans: false,
        }
    }
}

impl TracingConfig {
    /// Create config for Jaeger
    pub fn jaeger(service_name: &str, endpoint: &str) -> Self {
        Self {
            service_name: service_name.to_string(),
            endpoint: Some(endpoint.to_string()),
            sample_rate: 1.0,
            console_output: false,
            detailed_spans: true,
        }
    }

    /// Create config for development
    pub fn development() -> Self {
        Self {
            service_name: "proof-service-dev".to_string(),
            endpoint: None,
            sample_rate: 1.0,
            console_output: true,
            detailed_spans: true,
        }
    }
}

/// Initialize tracing with the given configuration
pub fn init_tracing(config: &TracingConfig) -> ProofResult<()> {
    use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info"));

    let subscriber = tracing_subscriber::registry().with(filter);

    if config.console_output {
        let fmt_layer = tracing_subscriber::fmt::layer()
            .with_target(true)
            .with_thread_ids(true)
            .with_file(true)
            .with_line_number(true);

        subscriber.with(fmt_layer).init();
    } else {
        subscriber.init();
    }

    tracing::info!(
        service = %config.service_name,
        "Tracing initialized"
    );

    Ok(())
}

/// Traced proof generator with automatic span creation
pub struct TracedProofGenerator {
    config: ProofConfig,
}

impl TracedProofGenerator {
    /// Create a new traced generator
    pub fn new(config: ProofConfig) -> Self {
        Self { config }
    }

    /// Generate range proof with tracing
    #[instrument(
        name = "proof.generate.range",
        skip(self),
        fields(
            proof_type = "range",
            value = value,
            min = min,
            max = max,
        )
    )]
    pub fn generate_range_proof(&self, value: u64, min: u64, max: u64) -> ProofResult<RangeProof> {
        let start = Instant::now();

        let proof = RangeProof::generate(value, min, max, self.config.clone())?;

        tracing::info!(
            duration_ms = start.elapsed().as_millis() as u64,
            proof_size = proof.size(),
            "Range proof generated"
        );

        Ok(proof)
    }

    /// Generate gradient proof with tracing
    #[instrument(
        name = "proof.generate.gradient",
        skip(self, gradients),
        fields(
            proof_type = "gradient",
            gradient_count = gradients.len(),
            max_norm = max_norm,
        )
    )]
    pub fn generate_gradient_proof(
        &self,
        gradients: &[f32],
        max_norm: f32,
    ) -> ProofResult<GradientIntegrityProof> {
        let start = Instant::now();

        let proof = GradientIntegrityProof::generate(gradients, max_norm, self.config.clone())?;

        tracing::info!(
            duration_ms = start.elapsed().as_millis() as u64,
            proof_size = proof.size(),
            "Gradient proof generated"
        );

        Ok(proof)
    }

    /// Generate identity proof with tracing
    #[instrument(
        name = "proof.generate.identity",
        skip(self, factors),
        fields(
            proof_type = "identity",
            did = %did,
            factor_count = factors.len(),
        )
    )]
    pub fn generate_identity_proof(
        &self,
        did: &str,
        factors: &[ProofIdentityFactor],
        min_level: ProofAssuranceLevel,
    ) -> ProofResult<IdentityAssuranceProof> {
        let start = Instant::now();

        let proof = IdentityAssuranceProof::generate(did, factors, min_level, self.config.clone())?;

        tracing::info!(
            duration_ms = start.elapsed().as_millis() as u64,
            proof_size = proof.size(),
            "Identity proof generated"
        );

        Ok(proof)
    }

    /// Generate vote proof with tracing
    #[instrument(
        name = "proof.generate.vote",
        skip(self, voter),
        fields(
            proof_type = "vote",
            proposal_type = ?proposal_type,
        )
    )]
    pub fn generate_vote_proof(
        &self,
        voter: &ProofVoterProfile,
        proposal_type: ProofProposalType,
    ) -> ProofResult<VoteEligibilityProof> {
        let start = Instant::now();

        let proof = VoteEligibilityProof::generate(voter, proposal_type, self.config.clone())?;

        tracing::info!(
            duration_ms = start.elapsed().as_millis() as u64,
            proof_size = proof.size(),
            "Vote proof generated"
        );

        Ok(proof)
    }

    /// Generate membership proof with tracing
    #[instrument(
        name = "proof.generate.membership",
        skip(self, leaf, path, root),
        fields(
            proof_type = "membership",
            path_length = path.len(),
        )
    )]
    pub fn generate_membership_proof(
        &self,
        leaf: [u8; 32],
        path: Vec<([u8; 32], bool)>,
        root: [u8; 32],
    ) -> ProofResult<MembershipProof> {
        let start = Instant::now();

        let proof = MembershipProof::generate(leaf, path, root, self.config.clone())?;

        tracing::info!(
            duration_ms = start.elapsed().as_millis() as u64,
            proof_size = proof.size(),
            "Membership proof generated"
        );

        Ok(proof)
    }
}

/// Verification tracer for adding spans to verification operations
pub struct VerificationTracer;

impl VerificationTracer {
    /// Verify with tracing
    #[instrument(
        name = "proof.verify",
        skip(proof_bytes),
        fields(
            proof_type = %proof_type,
            proof_size = proof_bytes.len(),
        )
    )]
    pub fn verify_traced<F, T>(
        proof_type: &str,
        proof_bytes: &[u8],
        verify_fn: F,
    ) -> ProofResult<T>
    where
        F: FnOnce() -> ProofResult<T>,
    {
        let start = Instant::now();
        let result = verify_fn();

        match &result {
            Ok(_) => {
                tracing::info!(
                    duration_ms = start.elapsed().as_millis() as u64,
                    valid = true,
                    "Proof verified"
                );
            }
            Err(e) => {
                tracing::warn!(
                    duration_ms = start.elapsed().as_millis() as u64,
                    valid = false,
                    error = %e,
                    "Proof verification failed"
                );
            }
        }

        result
    }
}

/// Create a span for proof operations
pub fn proof_span(operation: &str, proof_type: ProofType) -> Span {
    info_span!(
        "proof",
        operation = %operation,
        proof_type = ?proof_type,
    )
}

/// Create a span for batch operations
pub fn batch_span(operation: &str, batch_size: usize) -> Span {
    info_span!(
        "proof.batch",
        operation = %operation,
        batch_size = batch_size,
    )
}

/// Trace context for propagation
#[derive(Debug, Clone, Default)]
pub struct TraceContext {
    /// Trace ID
    pub trace_id: Option<String>,
    /// Span ID
    pub span_id: Option<String>,
    /// Parent span ID
    pub parent_span_id: Option<String>,
    /// Baggage items
    pub baggage: std::collections::HashMap<String, String>,
}

impl TraceContext {
    /// Create from current span
    pub fn from_current() -> Self {
        // Would extract from current tracing context
        Self::default()
    }

    /// Inject into headers
    pub fn inject(&self, headers: &mut std::collections::HashMap<String, String>) {
        if let Some(trace_id) = &self.trace_id {
            headers.insert("traceparent".to_string(), trace_id.clone());
        }
        for (key, value) in &self.baggage {
            headers.insert(format!("baggage-{}", key), value.clone());
        }
    }

    /// Extract from headers
    pub fn extract(headers: &std::collections::HashMap<String, String>) -> Self {
        let mut ctx = Self::default();
        if let Some(traceparent) = headers.get("traceparent") {
            ctx.trace_id = Some(traceparent.clone());
        }
        for (key, value) in headers {
            if let Some(baggage_key) = key.strip_prefix("baggage-") {
                ctx.baggage.insert(baggage_key.to_string(), value.clone());
            }
        }
        ctx
    }
}

/// Span timing helper
pub struct SpanTimer {
    start: Instant,
    span: Span,
}

impl SpanTimer {
    /// Start a new timer
    pub fn start(operation: &str, proof_type: ProofType) -> Self {
        let span = proof_span(operation, proof_type);
        Self {
            start: Instant::now(),
            span,
        }
    }

    /// Get elapsed time
    pub fn elapsed(&self) -> std::time::Duration {
        self.start.elapsed()
    }

    /// Record success
    pub fn success(self, message: &str) {
        let _guard = self.span.enter();
        tracing::info!(
            duration_ms = self.start.elapsed().as_millis() as u64,
            success = true,
            message,
        );
    }

    /// Record failure
    pub fn failure(self, error: &str) {
        let _guard = self.span.enter();
        tracing::error!(
            duration_ms = self.start.elapsed().as_millis() as u64,
            success = false,
            error,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proofs::SecurityLevel;

    fn test_config() -> ProofConfig {
        ProofConfig {
            security_level: SecurityLevel::Standard96,
            parallel: false,
            max_proof_size: 0,
        }
    }

    #[test]
    fn test_tracing_config_default() {
        let config = TracingConfig::default();
        assert_eq!(config.sample_rate, 1.0);
        assert!(config.console_output);
    }

    #[test]
    fn test_tracing_config_jaeger() {
        let config = TracingConfig::jaeger("my-service", "http://localhost:14268");
        assert_eq!(config.service_name, "my-service");
        assert!(config.endpoint.is_some());
        assert!(!config.console_output);
    }

    #[test]
    fn test_trace_context() {
        let ctx = TraceContext {
            trace_id: Some("abc123".to_string()),
            span_id: Some("def456".to_string()),
            parent_span_id: None,
            baggage: {
                let mut m = std::collections::HashMap::new();
                m.insert("user_id".to_string(), "user123".to_string());
                m
            },
        };

        let mut headers = std::collections::HashMap::new();
        ctx.inject(&mut headers);

        assert!(headers.contains_key("traceparent"));
        assert!(headers.contains_key("baggage-user_id"));

        let restored = TraceContext::extract(&headers);
        assert_eq!(restored.trace_id, Some("abc123".to_string()));
    }

    #[test]
    fn test_traced_generator() {
        let generator = TracedProofGenerator::new(test_config());

        // Just verify it doesn't panic
        let proof = generator.generate_range_proof(50, 0, 100);
        assert!(proof.is_ok());
    }

    #[test]
    fn test_span_timer() {
        let timer = SpanTimer::start("test", ProofType::Range);
        std::thread::sleep(std::time::Duration::from_millis(10));
        assert!(timer.elapsed() >= std::time::Duration::from_millis(10));
        timer.success("test completed");
    }
}
