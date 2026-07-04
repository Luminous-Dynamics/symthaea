// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Neural Bridge — Text Embedding → HDC Pipeline
//!
//! Converts raw text into the HDC representations used throughout the
//! cognitive core. The full pipeline is:
//!
//! ```text
//! &str (user text)
//!   │
//!   ▼  NeuralBridge::embed_text()
//! Vec<f32> (BGE-M3 embeddings via Ollama /api/embeddings)
//!   │
//!   ▼  project_to_hdc()  — random projection matrix (seeded, deterministic)
//! ContinuousHV (16,384-D float hypervector)
//!   │
//!   ▼  binarize()  — threshold at 0
//! BinaryHV (16,384-bit packed bipolar)
//!   │
//!   ▼  EmbeddingResult { continuous, binary, provenance }
//! ```
//!
//! ## Design Contracts
//!
//! - **No NaN/Inf leaks**: raw embedding is validated before projection;
//!   any non-finite value rejects the embedding with `BridgeError::NonFiniteEmbedding`.
//! - **Dimension independence**: the projection matrix maps any input dimension
//!   to exactly `HDC_DIMENSION` (16,384) output dimensions.
//! - **Determinism**: given the same model and text, the projection is
//!   identical across runs (the projection matrix seed is fixed).
//! - **Bounded output**: `ContinuousHV` similarity is in [-1, 1]; `BinaryHV`
//!   Hamming similarity is in [0, 1]. Both invariants are tested in this file.
//! - **Provenance**: every result carries the model name and a BLAKE3 hash of
//!   the input text so downstream consumers can audit what they received.
//!
//! ## Offline / Test Mode
//!
//! When Ollama is not available, use [`NeuralBridge::embed_deterministic`]
//! which bypasses the network and derives a `ContinuousHV`/`BinaryHV` directly
//! from the text via BLAKE3. This is suitable for unit tests and offline
//! development but does NOT produce semantically meaningful embeddings.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use symthaea_core::hdc::{ContinuousHV, HDC_DIMENSION, binary_hv::BinaryHV};

// ─── Error ───────────────────────────────────────────────────────────────────

/// Errors that can occur during embedding.
#[derive(Debug, Clone, PartialEq)]
pub enum BridgeError {
    /// Network request to Ollama failed.
    NetworkError(String),
    /// Ollama returned a non-200 response.
    ApiError { status: u16, body: String },
    /// The embedding vector contained NaN or Inf.
    NonFiniteEmbedding { index: usize, value: f32 },
    /// The embedding vector was empty.
    EmptyEmbedding,
    /// JSON parsing failed.
    ParseError(String),
    /// Input text was empty.
    EmptyInput,
}

impl std::fmt::Display for BridgeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NetworkError(e) => write!(f, "network error: {e}"),
            Self::ApiError { status, body } => write!(f, "API error {status}: {body}"),
            Self::NonFiniteEmbedding { index, value } => {
                write!(f, "non-finite embedding value at [{index}]: {value}")
            }
            Self::EmptyEmbedding => write!(f, "embedding vector is empty"),
            Self::ParseError(e) => write!(f, "JSON parse error: {e}"),
            Self::EmptyInput => write!(f, "input text is empty"),
        }
    }
}

// ─── Provenance ───────────────────────────────────────────────────────────────

/// Metadata carried with every embedding result.
///
/// Allows downstream consumers to audit what model and input produced a vector.
#[derive(Debug, Clone)]
pub struct EmbeddingProvenance {
    /// The Ollama model that produced the raw embedding.
    pub model: String,
    /// BLAKE3 hex hash of the input text (first 16 bytes = 32 hex chars).
    pub input_hash: String,
    /// Raw embedding dimension before projection.
    pub raw_dim: usize,
    /// Wall-clock time for the Ollama round-trip (0 in offline mode).
    pub latency_ms: u64,
}

// ─── Result ──────────────────────────────────────────────────────────────────

/// The full output of the Neural Bridge for a single text input.
#[derive(Debug, Clone)]
pub struct EmbeddingResult {
    /// Float hypervector (16,384-D) — suitable for EFE scoring, drift detection.
    pub continuous: ContinuousHV,
    /// Binary hypervector (16,384-bit) — suitable for memory search, binding.
    pub binary: BinaryHV,
    /// Provenance metadata.
    pub provenance: EmbeddingProvenance,
}

// ─── Projection Matrix ────────────────────────────────────────────────────────

/// A random projection matrix that maps `input_dim → HDC_DIMENSION`.
///
/// Uses Gaussian random entries seeded deterministically from a fixed seed,
/// giving a Johnson–Lindenstrauss-style projection that preserves cosine
/// similarity approximately.
///
/// The matrix is computed lazily and cached per `(input_dim, seed)`.
struct ProjectionMatrix {
    /// Rows = HDC_DIMENSION, cols = input_dim.
    /// Stored transposed for cache-friendly dot products.
    /// Entry [row][col] drawn from N(0, 1/sqrt(input_dim)).
    rows: Vec<Vec<f32>>,
    input_dim: usize,
}

impl ProjectionMatrix {
    /// Build a `HDC_DIMENSION × input_dim` projection matrix.
    ///
    /// Uses Box-Muller transform seeded via BLAKE3 for reproducibility.
    fn new(input_dim: usize, seed: u64) -> Self {
        use std::f32::consts::PI;

        let scale = 1.0 / (input_dim as f32).sqrt();
        let mut rows = Vec::with_capacity(HDC_DIMENSION);

        // Generate HDC_DIMENSION × input_dim Gaussian entries.
        // We need 2 uniform samples per Box-Muller pair.
        // Total pairs needed: HDC_DIMENSION * input_dim (ceiling)
        let total_floats = HDC_DIMENSION * input_dim;
        let total_bytes = total_floats * 4 + 16; // 4 bytes per f32 + headroom

        // Use BLAKE3 XOF seeded with (seed || input_dim) for determinism.
        let key = {
            let mut h = blake3::Hasher::new();
            h.update(&seed.to_le_bytes());
            h.update(&(input_dim as u64).to_le_bytes());
            h.finalize()
        };
        let mut xof = blake3::Hasher::new_keyed(key.as_bytes()).finalize_xof();
        let mut buf = vec![0u8; total_bytes];
        xof.fill(&mut buf);

        let mut byte_idx = 0usize;

        let read_u32 = |buf: &[u8], idx: &mut usize| -> u32 {
            let v = u32::from_le_bytes([buf[*idx], buf[*idx + 1], buf[*idx + 2], buf[*idx + 3]]);
            *idx += 4;
            v
        };

        for _ in 0..HDC_DIMENSION {
            let mut row = Vec::with_capacity(input_dim);
            let mut col = 0;
            while col < input_dim {
                // Box-Muller: two uniform u32 → two independent N(0,1)
                let u1 = (read_u32(&buf, &mut byte_idx) as f32 + 0.5) / (u32::MAX as f32 + 1.0);
                let u2 = (read_u32(&buf, &mut byte_idx) as f32 + 0.5) / (u32::MAX as f32 + 1.0);
                let r = (-2.0 * u1.ln()).sqrt();
                let z0 = r * (2.0 * PI * u2).cos() * scale;
                let z1 = r * (2.0 * PI * u2).sin() * scale;
                row.push(z0);
                col += 1;
                if col < input_dim {
                    row.push(z1);
                    col += 1;
                }
            }
            rows.push(row);
        }

        Self { rows, input_dim }
    }

    /// Project `embedding` (length = input_dim) → Vec<f32> (length = HDC_DIMENSION).
    fn project(&self, embedding: &[f32]) -> Vec<f32> {
        debug_assert_eq!(
            embedding.len(),
            self.input_dim,
            "embedding dim mismatch: got {}, expected {}",
            embedding.len(),
            self.input_dim
        );

        self.rows
            .iter()
            .map(|row| {
                row.iter()
                    .zip(embedding.iter())
                    .map(|(&w, &x)| w * x)
                    .sum::<f32>()
            })
            .collect()
    }
}

// ─── Bridge ──────────────────────────────────────────────────────────────────

/// Configuration for the Neural Bridge.
#[derive(Debug, Clone)]
pub struct NeuralBridgeConfig {
    /// Ollama API endpoint.
    pub endpoint: String,
    /// Embedding model. Must be an approved model from CLAUDE.md.
    /// Default: `embeddinggemma:300m`
    pub model: String,
    /// HTTP timeout for embedding requests.
    pub timeout: Duration,
    /// Seed for the deterministic projection matrix.
    /// Change only if you need to rotate the HDC basis.
    pub projection_seed: u64,
}

impl Default for NeuralBridgeConfig {
    fn default() -> Self {
        Self {
            endpoint: "http://localhost:11434".into(),
            model: "embeddinggemma:300m".into(),
            timeout: Duration::from_secs(10),
            // Fixed seed — changing this invalidates all stored BinaryHV vectors.
            projection_seed: 0x5754_ae9f_1b3c_d024,
        }
    }
}

/// The Neural Bridge: text → HDC.
///
/// Thread-safe: the projection matrix cache is behind an `Arc<RwLock<...>>`.
pub struct NeuralBridge {
    config: NeuralBridgeConfig,
    /// Cache of projection matrices keyed by raw embedding dimension.
    /// Most models produce a fixed dimension, so this rarely has >1 entry.
    projection_cache: Arc<RwLock<HashMap<usize, ProjectionMatrix>>>,
}

impl NeuralBridge {
    /// Create a new bridge with default configuration.
    pub fn new() -> Self {
        Self::with_config(NeuralBridgeConfig::default())
    }

    /// Create with custom configuration.
    pub fn with_config(config: NeuralBridgeConfig) -> Self {
        Self {
            config,
            projection_cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Embed `text` using Ollama and return the full HDC result.
    ///
    /// Requires a live Ollama instance. For offline/test use, see
    /// [`embed_deterministic`](Self::embed_deterministic).
    ///
    /// # Errors
    ///
    /// - [`BridgeError::EmptyInput`] — text is empty after trimming
    /// - [`BridgeError::NetworkError`] — Ollama unreachable
    /// - [`BridgeError::ApiError`] — non-200 from Ollama
    /// - [`BridgeError::NonFiniteEmbedding`] — NaN/Inf in the response
    #[cfg(feature = "native")]
    pub async fn embed_text(&self, text: &str) -> Result<EmbeddingResult, BridgeError> {
        let text = text.trim();
        if text.is_empty() {
            return Err(BridgeError::EmptyInput);
        }

        let input_hash = self.hash_input(text);
        let t0 = Instant::now();
        let raw = self.fetch_embedding(text).await?;
        let latency_ms = t0.elapsed().as_millis() as u64;

        let result =
            self.project_and_binarize(raw, self.config.model.clone(), input_hash, latency_ms)?;

        Ok(result)
    }

    /// Embed `text` without a network call — deterministic, offline-safe.
    ///
    /// Derives a `ContinuousHV` and `BinaryHV` directly from the text via
    /// BLAKE3, treating the hash output as a pseudo-embedding. This does
    /// **not** produce semantically meaningful vectors (similar texts will
    /// not have similar embeddings). Use only for tests and offline development.
    pub fn embed_deterministic(&self, text: &str) -> Result<EmbeddingResult, BridgeError> {
        let text = text.trim();
        if text.is_empty() {
            return Err(BridgeError::EmptyInput);
        }

        let input_hash = self.hash_input(text);

        // Derive a pseudo-embedding of HDC_DIMENSION from BLAKE3 XOF
        let mut hasher = blake3::Hasher::new();
        hasher.update(text.as_bytes());
        hasher.update(b"symthaea-neural-bridge-deterministic-v1");
        let mut xof = hasher.finalize_xof();

        let mut raw = vec![0u8; HDC_DIMENSION * 4];
        xof.fill(&mut raw);

        let float_vec: Vec<f32> = raw
            .chunks_exact(4)
            .map(|b| {
                let u = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                // Map u32 to [-1, 1]
                (u as f32 / (u32::MAX as f32)) * 2.0 - 1.0
            })
            .collect();

        let continuous = ContinuousHV::from_vec(float_vec.clone());
        let binary = BinaryHV::from_bipolar(&float_vec);

        Ok(EmbeddingResult {
            continuous,
            binary,
            provenance: EmbeddingProvenance {
                model: "deterministic-blake3".into(),
                input_hash,
                raw_dim: HDC_DIMENSION,
                latency_ms: 0,
            },
        })
    }

    /// How similar are two texts in HDC space?
    ///
    /// Uses `ContinuousHV::similarity` (cosine, range [-1, 1]).
    /// In offline/test mode, pass `use_network: false` to use deterministic
    /// embeddings.
    pub fn similarity_offline(&self, a: &str, b: &str) -> Result<f32, BridgeError> {
        let ea = self.embed_deterministic(a)?;
        let eb = self.embed_deterministic(b)?;
        Ok(ea.continuous.similarity(&eb.continuous))
    }

    // ── Private ──────────────────────────────────────────────────────────────

    /// Fetch raw float embedding from Ollama `/api/embeddings`.
    #[cfg(feature = "native")]
    async fn fetch_embedding(&self, text: &str) -> Result<Vec<f32>, BridgeError> {
        use hyper_util::rt::TokioIo;
        use tokio::net::TcpStream;

        // Parse the endpoint URL manually (avoid reqwest/hyper-full dependency)
        let url = format!("{}/api/embeddings", self.config.endpoint);
        let body = serde_json::json!({
            "model": self.config.model,
            "prompt": text,
        })
        .to_string();

        // Use tokio timeout around the whole request
        let response_text = tokio::time::timeout(self.config.timeout, async {
            // Simple HTTP POST using ureq-style blocking-in-async or hyper-util
            // We use a subprocess-free approach: reqwest-lite or raw hyper.
            // Since symthaea-nix uses hyper-util, we send a raw HTTP/1.1 request.
            self.http_post_json(&url, &body).await
        })
        .await
        .map_err(|_| BridgeError::NetworkError("request timed out".into()))?
        .map_err(|e| BridgeError::NetworkError(e.to_string()))?;

        // Parse the Ollama embeddings response
        // {"embedding": [0.123, -0.456, ...]}
        let parsed: serde_json::Value = serde_json::from_str(&response_text)
            .map_err(|e| BridgeError::ParseError(e.to_string()))?;

        let embedding = parsed
            .get("embedding")
            .and_then(|v| v.as_array())
            .ok_or_else(|| {
                BridgeError::ParseError("missing 'embedding' array in response".into())
            })?;

        if embedding.is_empty() {
            return Err(BridgeError::EmptyEmbedding);
        }

        let floats: Vec<f32> = embedding
            .iter()
            .map(|v| v.as_f64().unwrap_or(0.0) as f32)
            .collect();

        Ok(floats)
    }

    /// Minimal HTTP POST that returns the response body as a String.
    #[cfg(feature = "native")]
    async fn http_post_json(&self, url: &str, body: &str) -> Result<String, String> {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::TcpStream;

        // Parse host:port from the URL
        let url_stripped = url
            .strip_prefix("http://")
            .ok_or_else(|| "only http:// supported".to_string())?;
        let (host_port, path) = url_stripped.split_once('/').unwrap_or((url_stripped, ""));
        let path = format!("/{path}");

        let host = host_port.split(':').next().unwrap_or("localhost");
        let port: u16 = host_port
            .split(':')
            .nth(1)
            .and_then(|p| p.parse().ok())
            .unwrap_or(11434);

        let mut stream = TcpStream::connect(format!("{host}:{port}"))
            .await
            .map_err(|e| e.to_string())?;

        let request = format!(
            "POST {path} HTTP/1.1\r\nHost: {host}:{port}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );

        stream
            .write_all(request.as_bytes())
            .await
            .map_err(|e| e.to_string())?;

        let mut response = String::new();
        stream
            .read_to_string(&mut response)
            .await
            .map_err(|e| e.to_string())?;

        // Split headers from body
        let body_start = response
            .find("\r\n\r\n")
            .ok_or_else(|| "malformed HTTP response".to_string())?
            + 4;

        // Check status line
        let status_line = response.lines().next().unwrap_or("");
        let status: u16 = status_line
            .split_whitespace()
            .nth(1)
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);

        if status != 200 {
            return Err(format!(
                "HTTP {status}: {}",
                response[body_start..].chars().take(200).collect::<String>()
            ));
        }

        Ok(response[body_start..].to_string())
    }

    /// Validate, project, and binarize a raw embedding.
    fn project_and_binarize(
        &self,
        raw: Vec<f32>,
        model: String,
        input_hash: String,
        latency_ms: u64,
    ) -> Result<EmbeddingResult, BridgeError> {
        if raw.is_empty() {
            return Err(BridgeError::EmptyEmbedding);
        }

        // Contract: no NaN or Inf may enter the HDC pipeline
        for (i, &v) in raw.iter().enumerate() {
            if !v.is_finite() {
                return Err(BridgeError::NonFiniteEmbedding { index: i, value: v });
            }
        }

        let raw_dim = raw.len();

        // Project raw → HDC_DIMENSION
        let projected = if raw_dim == HDC_DIMENSION {
            // Already the right dimension — use directly
            raw
        } else {
            // Get or build projection matrix for this dimension
            {
                let cache = self.projection_cache.read().unwrap();
                if !cache.contains_key(&raw_dim) {
                    drop(cache);
                    let mut cache = self.projection_cache.write().unwrap();
                    cache.entry(raw_dim).or_insert_with(|| {
                        ProjectionMatrix::new(raw_dim, self.config.projection_seed)
                    });
                }
            }
            let cache = self.projection_cache.read().unwrap();
            cache[&raw_dim].project(&raw)
        };

        // Post-projection finite check (defensive)
        for (i, &v) in projected.iter().enumerate() {
            if !v.is_finite() {
                return Err(BridgeError::NonFiniteEmbedding { index: i, value: v });
            }
        }

        let continuous = ContinuousHV::from_vec(projected.clone());
        let binary = BinaryHV::from_bipolar(&projected);

        Ok(EmbeddingResult {
            continuous,
            binary,
            provenance: EmbeddingProvenance {
                model,
                input_hash,
                raw_dim,
                latency_ms,
            },
        })
    }

    /// BLAKE3 hash of input text, hex-encoded (first 16 bytes = 32 chars).
    fn hash_input(&self, text: &str) -> String {
        let hash = blake3::hash(text.as_bytes());
        hash.to_hex().chars().take(32).collect()
    }
}

impl Default for NeuralBridge {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn bridge() -> NeuralBridge {
        NeuralBridge::new()
    }

    // ── Deterministic offline path ────────────────────────────────────────────

    /// Same text always produces the same vectors.
    #[test]
    fn test_deterministic_same_text_same_result() {
        let b = bridge();
        let r1 = b.embed_deterministic("nixos rebuild switch").unwrap();
        let r2 = b.embed_deterministic("nixos rebuild switch").unwrap();

        let sim = r1.continuous.similarity(&r2.continuous);
        assert!(
            (sim - 1.0).abs() < 1e-5,
            "identical input must produce identical ContinuousHV: sim={sim}"
        );
        assert_eq!(
            r1.binary, r2.binary,
            "identical input must produce identical BinaryHV"
        );
    }

    /// Different texts produce different vectors (quasi-orthogonal in high-D).
    #[test]
    fn test_deterministic_different_text_different_result() {
        let b = bridge();
        let r1 = b.embed_deterministic("install firefox").unwrap();
        let r2 = b.embed_deterministic("remove vim").unwrap();

        // Different inputs should NOT produce identical vectors
        assert_ne!(
            r1.binary, r2.binary,
            "different inputs must produce different BinaryHV"
        );
    }

    /// ContinuousHV similarity is always in [-1, 1].
    #[test]
    fn test_continuous_similarity_in_unit_interval() {
        let b = bridge();
        let texts = [
            "enable nginx",
            "disable sshd",
            "nixos rebuild",
            "",    // will error — tested separately
            "   ", // will error — tested separately
        ];

        for text in texts.iter().take(3) {
            let r = b.embed_deterministic(text).unwrap();
            let self_sim = r.continuous.similarity(&r.continuous);
            assert!(
                (self_sim - 1.0).abs() < 1e-4,
                "self-similarity must be ≈1.0 for '{text}': got {self_sim}"
            );

            // Cross-similarity with another
            let r2 = b.embed_deterministic("rollback to generation 42").unwrap();
            let cross = r.continuous.similarity(&r2.continuous);
            assert!(
                cross >= -1.0 && cross <= 1.0,
                "cross-similarity must be in [-1,1] for '{text}': got {cross}"
            );
        }
    }

    /// BinaryHV Hamming similarity is always in [0, 1].
    #[test]
    fn test_binary_hamming_similarity_in_unit_interval() {
        let b = bridge();
        let r1 = b.embed_deterministic("configure network").unwrap();
        let r2 = b.embed_deterministic("garbage collect store").unwrap();

        let sim = r1.binary.similarity(&r2.binary);
        assert!(
            sim >= 0.0 && sim <= 1.0,
            "BinaryHV similarity must be in [0,1]: {sim}"
        );
        assert!(sim.is_finite(), "BinaryHV similarity must be finite: {sim}");
    }

    /// Self-similarity of BinaryHV is exactly 1.0.
    #[test]
    fn test_binary_self_similarity_is_one() {
        let b = bridge();
        let r = b.embed_deterministic("update flake inputs").unwrap();
        let sim = r.binary.similarity(&r.binary);
        assert!(
            (sim - 1.0).abs() < 1e-6,
            "BinaryHV self-similarity must be exactly 1.0: {sim}"
        );
    }

    /// Empty and whitespace-only input returns EmptyInput error.
    #[test]
    fn test_empty_input_returns_error() {
        let b = bridge();
        assert_eq!(
            b.embed_deterministic(""),
            Err(BridgeError::EmptyInput),
            "empty string must return EmptyInput"
        );
        assert_eq!(
            b.embed_deterministic("   "),
            Err(BridgeError::EmptyInput),
            "whitespace-only must return EmptyInput"
        );
        assert_eq!(
            b.embed_deterministic("\t\n"),
            Err(BridgeError::EmptyInput),
            "tab/newline must return EmptyInput"
        );
    }

    /// Provenance is correctly populated.
    #[test]
    fn test_provenance_is_populated() {
        let b = bridge();
        let r = b.embed_deterministic("test input").unwrap();

        assert_eq!(r.provenance.model, "deterministic-blake3");
        assert_eq!(r.provenance.latency_ms, 0);
        assert_eq!(r.provenance.raw_dim, HDC_DIMENSION);
        // Hash must be 32 hex chars
        assert_eq!(
            r.provenance.input_hash.len(),
            32,
            "input hash must be 32 hex chars"
        );
        assert!(
            r.provenance
                .input_hash
                .chars()
                .all(|c| c.is_ascii_hexdigit()),
            "input hash must be hex"
        );
    }

    /// Different inputs produce different provenance hashes.
    #[test]
    fn test_provenance_hash_differs_for_different_inputs() {
        let b = bridge();
        let r1 = b.embed_deterministic("input A").unwrap();
        let r2 = b.embed_deterministic("input B").unwrap();
        assert_ne!(
            r1.provenance.input_hash, r2.provenance.input_hash,
            "different inputs must produce different hashes"
        );
    }

    // ── Projection matrix ────────────────────────────────────────────────────

    /// Random projection is deterministic for the same seed and dimension.
    #[test]
    fn test_projection_matrix_deterministic() {
        let m1 = ProjectionMatrix::new(768, 42);
        let m2 = ProjectionMatrix::new(768, 42);

        // Same seed, same dimension → identical rows
        for (r1, r2) in m1.rows.iter().zip(m2.rows.iter()) {
            for (&a, &b) in r1.iter().zip(r2.iter()) {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "projection matrix must be deterministic"
                );
            }
        }
    }

    /// Projected output dimension is always HDC_DIMENSION.
    #[test]
    fn test_projection_output_dimension() {
        let b = bridge();
        // Simulate 768-dim embedding (BGE-M3)
        let raw: Vec<f32> = (0..768).map(|i| (i as f32) / 768.0 - 0.5).collect();
        let result = b
            .project_and_binarize(raw, "test".into(), "abc".into(), 0)
            .unwrap();

        assert_eq!(
            result.continuous.dim(),
            HDC_DIMENSION,
            "projected ContinuousHV must have HDC_DIMENSION dimensions"
        );
    }

    /// Projected output is always finite.
    #[test]
    fn test_projection_output_always_finite() {
        let b = bridge();
        // Extreme input values
        for scale in [0.0f32, 1.0, 100.0, -100.0, f32::MIN_POSITIVE] {
            let raw: Vec<f32> = (0..768).map(|i| scale * (i as f32 / 384.0 - 1.0)).collect();
            let result = b
                .project_and_binarize(raw, "test".into(), "abc".into(), 0)
                .unwrap();

            for (i, &v) in result.continuous.as_slice().iter().enumerate() {
                assert!(
                    v.is_finite(),
                    "projected[{i}] must be finite for scale={scale}: {v}"
                );
            }
        }
    }

    /// NaN in raw embedding is rejected before projection.
    #[test]
    fn test_nan_embedding_rejected() {
        let b = bridge();
        let mut raw = vec![0.5f32; 768];
        raw[100] = f32::NAN;

        let err = b
            .project_and_binarize(raw, "test".into(), "abc".into(), 0)
            .unwrap_err();
        assert!(
            matches!(err, BridgeError::NonFiniteEmbedding { index: 100, .. }),
            "NaN must be caught at index 100: {err}"
        );
    }

    /// Inf in raw embedding is rejected before projection.
    #[test]
    fn test_inf_embedding_rejected() {
        let b = bridge();
        let mut raw = vec![0.5f32; 768];
        raw[42] = f32::INFINITY;

        let err = b
            .project_and_binarize(raw, "test".into(), "abc".into(), 0)
            .unwrap_err();
        assert!(
            matches!(err, BridgeError::NonFiniteEmbedding { index: 42, .. }),
            "Inf must be caught at index 42: {err}"
        );
    }

    /// Empty embedding vector is rejected.
    #[test]
    fn test_empty_raw_embedding_rejected() {
        let b = bridge();
        let err = b
            .project_and_binarize(vec![], "test".into(), "abc".into(), 0)
            .unwrap_err();
        assert_eq!(err, BridgeError::EmptyEmbedding);
    }

    // ── Pipeline composition with mind layer ─────────────────────────────────

    /// ContinuousHV from the bridge integrates with HdcWorldModel::observe.
    #[test]
    fn test_bridge_output_feeds_hdc_world_model() {
        use crate::mind::hdc_world_model::HdcWorldModel;

        let b = bridge();
        let r = b.embed_deterministic("enable postgresql service").unwrap();

        let mut wm = HdcWorldModel::new(HDC_DIMENSION);
        wm.set_expected_state(r.continuous.clone());
        wm.observe(&r.continuous);

        let drift = wm.detect_drift(0.8);
        assert!(
            !drift.drifted,
            "same vector observed as expected must not drift: sim={}",
            drift.similarity
        );
        assert!(
            drift.similarity >= 0.0 && drift.similarity <= 1.0,
            "drift similarity must be in [0,1]: {}",
            drift.similarity
        );
    }

    /// BinaryHV from the bridge can be used in similarity search.
    #[test]
    fn test_bridge_binary_similarity_search() {
        let b = bridge();

        let query = b.embed_deterministic("install postgresql").unwrap();
        let targets: Vec<BinaryHV> = [
            "install postgresql", // identical
            "remove postgresql",  // related
            "enable nginx",       // different
            "garbage collect",    // unrelated
        ]
        .iter()
        .map(|t| b.embed_deterministic(t).unwrap().binary)
        .collect();

        // The first target (identical) must have similarity = 1.0
        let sim_identical = query.binary.similarity(&targets[0]);
        assert!(
            (sim_identical - 1.0).abs() < 1e-6,
            "identical text must have BinaryHV similarity = 1.0: {sim_identical}"
        );

        // All similarities must be in [0, 1]
        for (i, target) in targets.iter().enumerate() {
            let sim = query.binary.similarity(target);
            assert!(
                sim >= 0.0 && sim <= 1.0,
                "BinaryHV similarity[{i}] must be in [0,1]: {sim}"
            );
        }
    }

    /// offline similarity returns [-1, 1] for any text pair.
    #[test]
    fn test_similarity_offline_bounded() {
        let b = bridge();
        for (a, c) in [
            ("a", "b"),
            ("nixos", "nixos"),
            ("hello world", "goodbye moon"),
        ] {
            let sim = b.similarity_offline(a, c).unwrap();
            assert!(
                sim >= -1.0 && sim <= 1.0,
                "similarity_offline must be in [-1,1]: {sim}"
            );
            assert!(sim.is_finite(), "similarity_offline must be finite: {sim}");
        }
    }
}
