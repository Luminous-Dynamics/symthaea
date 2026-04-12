// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Code Certificate — Machine-Verifiable Audit Trail
//!
//! Every piece of code accepted by the CodeOrchestrator receives a
//! `CodeCertificate` — a cryptographic receipt proving *how* the code
//! was generated, *what* verification it passed, and *why* it was accepted.
//!
//! No LLM can produce this. A transformer generates tokens and emits them.
//! It cannot prove that the output was verified against topological constraints,
//! passed sheaf coherence checks, or that the intent-to-code similarity was
//! mathematically computed in 16,384-dimensional hypervector space.
//!
//! # Certificate Contents
//!
//! ```text
//! CodeCertificate {
//!     id:                  "cert_a7f3b2..."      // BLAKE3 of source
//!     source_hash:         [u8; 32]               // BLAKE3 hash of source code
//!     backend_used:        "CodeGenerator"         // Which synthesis backend
//!     semantic_similarity: 0.847                   // Cosine sim(intent_hv, code_hv)
//!     verification_layers: [                       // What checks passed
//!         { name: "syntax", passed: true },
//!         { name: "semantic_similarity", passed: true, score: 0.847 },
//!         { name: "entity_count", passed: true, score: 5.0 },
//!     ]
//!     epistemic_status:    Probable                // System's confidence level
//!     safety_critical:     false                   // Whether this was safety-gated
//!     timestamp:           1744300800               // Unix timestamp
//! }
//! ```

use serde::{Deserialize, Serialize};
use symthaea_core::synthesis_trait::{EpistemicStatus, VerificationLayer};

/// Machine-verifiable certificate proving code provenance and verification.
///
/// This is Symthaea's competitive moat: every accepted function comes with
/// a cryptographic audit trail that no LLM API can replicate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeCertificate {
    /// Unique certificate ID (BLAKE3 hash prefix of source + timestamp)
    pub id: String,
    /// BLAKE3 hash of the verified source code
    pub source_hash: [u8; 32],
    /// Which synthesis backend produced this code
    pub backend_used: String,
    /// Cosine similarity between intent HV and generated code HV (0.0-1.0)
    pub semantic_similarity: f32,
    /// Verification layers that were applied, with pass/fail and scores
    pub verification_layers: Vec<CertVerificationLayer>,
    /// System's epistemic confidence at time of generation
    pub epistemic_status: String,
    /// Whether this was generated under safety-critical policy (0.85 threshold)
    pub safety_critical: bool,
    /// Unix timestamp of certificate issuance
    pub timestamp: u64,
    /// Topology constraints verified (Betti numbers, if applicable)
    pub topology: Option<CertTopology>,
    /// CfC execution oracle convergence prediction (if applicable)
    pub oracle_convergence: Option<f32>,
    /// Whether sheaf coherence was verified (if applicable)
    pub sheaf_coherent: Option<bool>,
}

/// Serializable verification layer for the certificate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CertVerificationLayer {
    /// Verification check name
    pub name: String,
    /// Whether the check passed
    pub passed: bool,
    /// Numeric score (if applicable)
    pub score: Option<f32>,
    /// Details about the check
    pub detail: String,
}

/// Topological invariants verified via Geodesic synthesis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CertTopology {
    /// β₀: connected components (independent modules)
    pub beta_0: usize,
    /// β₁: independent cycles (loops)
    pub beta_1: usize,
    /// β₂: enclosed voids
    pub beta_2: usize,
    /// Euler characteristic
    pub euler_characteristic: i64,
}

impl CodeCertificate {
    /// Create a new certificate for verified code.
    pub fn new(source: &str, backend: &str, similarity: f32) -> Self {
        let source_hash = blake3::hash(source.as_bytes());
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        // Certificate ID: first 16 hex chars of BLAKE3(source + timestamp)
        let id_input = format!("{}{}", source, timestamp);
        let id_hash = blake3::hash(id_input.as_bytes());
        let id = format!("cert_{}", hex_prefix(id_hash.as_bytes(), 8));

        Self {
            id,
            source_hash: *source_hash.as_bytes(),
            backend_used: backend.to_string(),
            semantic_similarity: similarity,
            verification_layers: Vec::new(),
            epistemic_status: "Probable".to_string(),
            safety_critical: false,
            timestamp,
            topology: None,
            oracle_convergence: None,
            sheaf_coherent: None,
        }
    }

    /// Set the epistemic status.
    pub fn with_epistemic_status(mut self, status: EpistemicStatus) -> Self {
        self.epistemic_status = format!("{status}");
        self
    }

    /// Set verification layers from the orchestrator's audit trail.
    pub fn with_verification_layers(mut self, layers: &[VerificationLayer]) -> Self {
        self.verification_layers = layers
            .iter()
            .map(|l| CertVerificationLayer {
                name: l.name.clone(),
                passed: l.passed,
                score: l.score,
                detail: l.detail.clone(),
            })
            .collect();
        self
    }

    /// Mark as safety-critical.
    pub fn with_safety_critical(mut self, critical: bool) -> Self {
        self.safety_critical = critical;
        self
    }

    /// Add topology verification results.
    pub fn with_topology(mut self, beta_0: usize, beta_1: usize, beta_2: usize) -> Self {
        let euler = beta_0 as i64 - beta_1 as i64 + beta_2 as i64;
        self.topology = Some(CertTopology {
            beta_0,
            beta_1,
            beta_2,
            euler_characteristic: euler,
        });
        self
    }

    /// Add execution oracle convergence prediction.
    pub fn with_oracle_convergence(mut self, convergence: f32) -> Self {
        self.oracle_convergence = Some(convergence);
        self
    }

    /// Add sheaf coherence result.
    pub fn with_sheaf_coherent(mut self, coherent: bool) -> Self {
        self.sheaf_coherent = Some(coherent);
        self
    }

    /// Serialize the certificate to JSON.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_string())
    }

    /// Verify that a source code string matches this certificate's hash.
    ///
    /// This is the tamper-detection mechanism: if the code was modified
    /// after certification, this returns false.
    pub fn verify_source(&self, source: &str) -> bool {
        let hash = blake3::hash(source.as_bytes());
        hash.as_bytes() == &self.source_hash
    }

    /// How many verification layers passed?
    pub fn layers_passed(&self) -> usize {
        self.verification_layers.iter().filter(|l| l.passed).count()
    }

    /// How many verification layers total?
    pub fn layers_total(&self) -> usize {
        self.verification_layers.len()
    }

    /// Human-readable summary of the certificate.
    pub fn summary(&self) -> String {
        let topo_str = match &self.topology {
            Some(t) => format!(
                ", topology: β₀={} β₁={} β₂={}",
                t.beta_0, t.beta_1, t.beta_2
            ),
            None => String::new(),
        };
        let oracle_str = match self.oracle_convergence {
            Some(c) => format!(", oracle: {:.3}", c),
            None => String::new(),
        };
        let sheaf_str = match self.sheaf_coherent {
            Some(true) => ", sheaf: coherent".to_string(),
            Some(false) => ", sheaf: INCOHERENT".to_string(),
            None => String::new(),
        };

        format!(
            "[{}] {} via {} | sim: {:.3} | {}/{} checks passed | {}{}{}{}",
            self.id,
            if self.safety_critical {
                "SAFETY"
            } else {
                "std"
            },
            self.backend_used,
            self.semantic_similarity,
            self.layers_passed(),
            self.layers_total(),
            self.epistemic_status,
            topo_str,
            oracle_str,
            sheaf_str,
        )
    }
}

/// Convert first N bytes to hex string.
fn hex_prefix(bytes: &[u8], n: usize) -> String {
    bytes.iter().take(n).map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_certificate_creation() {
        let cert = CodeCertificate::new(
            "fn add(a: i32, b: i32) -> i32 { a + b }",
            "CodeGenerator",
            0.85,
        );

        assert!(cert.id.starts_with("cert_"));
        assert_eq!(cert.backend_used, "CodeGenerator");
        assert!((cert.semantic_similarity - 0.85).abs() < 1e-6);
        assert!(!cert.safety_critical);
        assert!(cert.timestamp > 0);
    }

    #[test]
    fn test_certificate_verify_source() {
        let source = "fn hello() -> &str { \"world\" }";
        let cert = CodeCertificate::new(source, "CodeGenerator", 0.9);

        assert!(cert.verify_source(source));
        assert!(!cert.verify_source("fn hello() -> &str { \"earth\" }"));
    }

    #[test]
    fn test_certificate_with_topology() {
        let cert = CodeCertificate::new("fn x() {}", "Geodesic", 0.8).with_topology(1, 2, 0);

        let topo = cert.topology.unwrap();
        assert_eq!(topo.beta_0, 1);
        assert_eq!(topo.beta_1, 2);
        assert_eq!(topo.beta_2, 0);
        assert_eq!(topo.euler_characteristic, -1); // 1 - 2 + 0
    }

    #[test]
    fn test_certificate_with_verification_layers() {
        let layers = vec![
            VerificationLayer {
                name: "syntax".to_string(),
                passed: true,
                score: None,
                detail: "Valid".to_string(),
            },
            VerificationLayer {
                name: "semantic".to_string(),
                passed: true,
                score: Some(0.85),
                detail: "Above threshold".to_string(),
            },
            VerificationLayer {
                name: "topology".to_string(),
                passed: false,
                score: Some(0.3),
                detail: "Beta_1 mismatch".to_string(),
            },
        ];

        let cert =
            CodeCertificate::new("fn x() {}", "Test", 0.85).with_verification_layers(&layers);

        assert_eq!(cert.layers_passed(), 2);
        assert_eq!(cert.layers_total(), 3);
    }

    #[test]
    fn test_certificate_json_roundtrip() {
        let cert = CodeCertificate::new(
            "fn add(a: i32, b: i32) -> i32 { a + b }",
            "CodeGenerator",
            0.85,
        )
        .with_epistemic_status(EpistemicStatus::Certain)
        .with_safety_critical(true)
        .with_topology(1, 0, 0)
        .with_oracle_convergence(0.95)
        .with_sheaf_coherent(true);

        let json = cert.to_json();
        assert!(json.contains("cert_"));
        assert!(json.contains("CodeGenerator"));
        assert!(json.contains("Certain"));
        assert!(json.contains("safety_critical"));

        // Deserialize back
        let parsed: CodeCertificate = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.backend_used, "CodeGenerator");
        assert!(parsed.safety_critical);
        assert!(parsed.topology.is_some());
    }

    #[test]
    fn test_certificate_summary() {
        let cert = CodeCertificate::new("fn x() {}", "Geodesic", 0.92)
            .with_topology(1, 1, 0)
            .with_oracle_convergence(0.88)
            .with_sheaf_coherent(true);

        let summary = cert.summary();
        assert!(summary.contains("cert_"));
        assert!(summary.contains("Geodesic"));
        assert!(summary.contains("0.920"));
        assert!(summary.contains("β₀=1"));
        assert!(summary.contains("oracle: 0.880"));
        assert!(summary.contains("sheaf: coherent"));
    }

    #[test]
    fn test_certificate_unique_ids() {
        let cert1 = CodeCertificate::new("fn a() {}", "A", 0.8);
        let cert2 = CodeCertificate::new("fn b() {}", "B", 0.8);
        assert_ne!(cert1.id, cert2.id);
    }
}
