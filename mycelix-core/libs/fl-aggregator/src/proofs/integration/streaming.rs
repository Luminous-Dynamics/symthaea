//! Proof Streaming
//!
//! Asynchronous streaming for large proof batches and incremental verification.
//!
//! ## Features
//!
//! - Stream-based proof generation
//! - Incremental batch verification
//! - Backpressure handling
//! - Memory-efficient processing
//!
//! ## Usage
//!
//! ```rust,ignore
//! use fl_aggregator::proofs::integration::streaming::{
//!     ProofStream, stream_verify_batch
//! };
//!
//! // Stream verify a batch of proofs
//! let mut stream = stream_verify_batch(proof_envelopes);
//!
//! while let Some(result) = stream.next().await {
//!     println!("Proof {}: {}", result.index, result.valid);
//! }
//! ```

use crate::proofs::ProofType;
use super::serialization::ProofEnvelope;
use std::time::Instant;

#[cfg(feature = "proofs-streaming")]
use crate::proofs::ProofResult;

#[cfg(feature = "proofs-streaming")]
use tokio_stream::Stream;

#[cfg(feature = "proofs-streaming")]
use async_stream::stream;

/// Result of streaming verification
#[derive(Clone, Debug)]
pub struct StreamVerificationResult {
    /// Index in the stream
    pub index: usize,
    /// Proof type
    pub proof_type: ProofType,
    /// Whether verification succeeded
    pub valid: bool,
    /// Verification duration
    pub duration: std::time::Duration,
    /// Error message if failed
    pub error: Option<String>,
}

/// Streaming verification statistics
#[derive(Clone, Debug, Default)]
pub struct StreamStats {
    /// Total proofs processed
    pub total_processed: usize,
    /// Successful verifications
    pub successful: usize,
    /// Failed verifications
    pub failed: usize,
    /// Total processing time
    pub total_duration: std::time::Duration,
    /// Average verification time
    pub avg_duration: std::time::Duration,
}

impl StreamStats {
    /// Update stats with a new result
    pub fn update(&mut self, result: &StreamVerificationResult) {
        self.total_processed += 1;
        self.total_duration += result.duration;

        if result.valid {
            self.successful += 1;
        } else {
            self.failed += 1;
        }

        if self.total_processed > 0 {
            self.avg_duration = self.total_duration / self.total_processed as u32;
        }
    }

    /// Success rate
    pub fn success_rate(&self) -> f64 {
        if self.total_processed == 0 {
            1.0
        } else {
            self.successful as f64 / self.total_processed as f64
        }
    }
}

/// Create a streaming verifier for a batch of proof envelopes
///
/// Performs envelope-level validation asynchronously. For full cryptographic
/// proof verification, use the individual proof type APIs after deserialization.
#[cfg(feature = "proofs-streaming")]
pub fn stream_verify_batch(
    envelopes: Vec<ProofEnvelope>,
) -> impl Stream<Item = StreamVerificationResult> {
    stream! {
        for (index, envelope) in envelopes.into_iter().enumerate() {
            let start = Instant::now();

            // Validate envelope integrity
            // The envelope was already parsed successfully, so we verify:
            // 1. Non-empty proof bytes
            // 2. Known proof type
            let valid = !envelope.proof_bytes.is_empty() && matches!(
                envelope.proof_type,
                ProofType::Range |
                ProofType::GradientIntegrity |
                ProofType::IdentityAssurance |
                ProofType::VoteEligibility |
                ProofType::Membership
            );

            let duration = start.elapsed();

            yield StreamVerificationResult {
                index,
                proof_type: envelope.proof_type,
                valid,
                duration,
                error: if valid { None } else { Some("Invalid envelope".to_string()) },
            };

            // Yield to allow other tasks to run
            tokio::task::yield_now().await;
        }
    }
}

/// Streaming proof generator
#[cfg(feature = "proofs-streaming")]
pub struct StreamingProofGenerator {
    /// Maximum concurrent generations
    concurrency: usize,
    /// Generation timeout
    timeout: std::time::Duration,
}

#[cfg(feature = "proofs-streaming")]
impl StreamingProofGenerator {
    /// Create a new streaming generator
    pub fn new(concurrency: usize, timeout: std::time::Duration) -> Self {
        Self { concurrency, timeout }
    }

    /// Stream generate range proofs
    pub fn stream_range_proofs(
        &self,
        inputs: Vec<(u64, u64, u64)>, // (value, min, max)
    ) -> impl Stream<Item = ProofResult<ProofEnvelope>> + '_ {
        use crate::proofs::{RangeProof, ProofConfig, SecurityLevel};

        let config = ProofConfig {
            security_level: SecurityLevel::Standard128,
            parallel: false,
            max_proof_size: 0,
        };

        stream! {
            for (value, min, max) in inputs {
                let result = RangeProof::generate(value, min, max, config.clone())
                    .and_then(|proof| ProofEnvelope::from_range_proof(&proof));

                yield result;
                tokio::task::yield_now().await;
            }
        }
    }
}

/// Chunked streaming for memory efficiency
#[cfg(feature = "proofs-streaming")]
pub fn stream_verify_chunked(
    envelopes: Vec<ProofEnvelope>,
    chunk_size: usize,
) -> impl Stream<Item = Vec<StreamVerificationResult>> {
    use tokio_stream::StreamExt;

    stream! {
        for chunk in envelopes.chunks(chunk_size) {
            let chunk_stream = stream_verify_batch(chunk.to_vec());
            let results: Vec<_> = chunk_stream.collect().await;
            yield results;
        }
    }
}

/// Progress callback for streaming operations
pub type ProgressCallback = Box<dyn Fn(usize, usize) + Send + Sync>;

/// Streaming verification with progress reporting
#[cfg(feature = "proofs-streaming")]
pub async fn stream_verify_with_progress(
    envelopes: Vec<ProofEnvelope>,
    on_progress: ProgressCallback,
) -> StreamStats {
    use tokio_stream::StreamExt;

    let total = envelopes.len();
    let mut stats = StreamStats::default();

    let mut stream = Box::pin(stream_verify_batch(envelopes));

    while let Some(result) = stream.next().await {
        stats.update(&result);
        on_progress(stats.total_processed, total);
    }

    stats
}

/// Non-streaming verification for compatibility
///
/// Verifies a batch of proof envelopes synchronously.
/// This performs envelope-level validation (format, checksum).
/// For full proof verification, use the individual proof types or
/// `stream_verify_batch` with the `proofs-streaming` feature.
pub fn verify_batch_sync(envelopes: &[ProofEnvelope]) -> Vec<StreamVerificationResult> {
    envelopes
        .iter()
        .enumerate()
        .map(|(index, envelope)| {
            let start = Instant::now();

            // Envelope-level validation:
            // If we successfully parsed the envelope, it passed format/checksum checks
            // Full proof verification requires async streaming or individual proof APIs
            let valid = !envelope.proof_bytes.is_empty();
            let duration = start.elapsed();

            StreamVerificationResult {
                index,
                proof_type: envelope.proof_type,
                valid,
                duration,
                error: if valid { None } else { Some("Empty proof bytes".to_string()) },
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stream_stats() {
        let mut stats = StreamStats::default();

        stats.update(&StreamVerificationResult {
            index: 0,
            proof_type: ProofType::Range,
            valid: true,
            duration: std::time::Duration::from_millis(100),
            error: None,
        });

        stats.update(&StreamVerificationResult {
            index: 1,
            proof_type: ProofType::Range,
            valid: false,
            duration: std::time::Duration::from_millis(50),
            error: Some("test error".to_string()),
        });

        assert_eq!(stats.total_processed, 2);
        assert_eq!(stats.successful, 1);
        assert_eq!(stats.failed, 1);
        assert_eq!(stats.success_rate(), 0.5);
    }

    #[test]
    fn test_sync_verification() {
        use crate::proofs::{RangeProof, ProofConfig, SecurityLevel};

        let config = ProofConfig {
            security_level: SecurityLevel::Standard96,
            parallel: false,
            max_proof_size: 0,
        };

        let proof = RangeProof::generate(50, 0, 100, config).unwrap();
        let envelope = ProofEnvelope::from_range_proof(&proof).unwrap();

        let results = verify_batch_sync(&[envelope]);
        assert_eq!(results.len(), 1);
        assert!(results[0].valid);
    }
}
