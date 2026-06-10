// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Peer fusion via Covariance Intersection for decentralized positioning.

use serde::{Deserialize, Serialize};

/// Helper for accepting both scalar and array sigma.
pub struct SigmaInput(pub [f64; 3]);
impl From<[f64; 3]> for SigmaInput {
    fn from(v: [f64; 3]) -> Self {
        Self(v)
    }
}
impl From<f64> for SigmaInput {
    fn from(v: f64) -> Self {
        Self([v, v, v])
    }
}
impl From<f32> for SigmaInput {
    fn from(v: f32) -> Self {
        Self([v as f64, v as f64, v as f64])
    }
}

/// 3D Gaussian estimate with mean and covariance diagonal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GaussianEstimate3D {
    pub mean: [f64; 3],
    pub covariance_diag: [f64; 3],
    /// Alias for covariance_diag (backward compat).
    #[serde(default)]
    pub covariance: [f64; 3],
}

impl GaussianEstimate3D {
    pub fn new(mean: [f64; 3], cov: [f64; 3]) -> Self {
        Self {
            mean,
            covariance_diag: cov,
            covariance: cov,
        }
    }
    pub fn from_diagonal_sigma(mean: [f64; 3], sigma: impl Into<SigmaInput>) -> Self {
        let s: SigmaInput = sigma.into();
        let c = [s.0[0] * s.0[0], s.0[1] * s.0[1], s.0[2] * s.0[2]];
        Self {
            mean,
            covariance_diag: c,
            covariance: c,
        }
    }
    pub fn uncertainty(&self) -> f64 {
        (self.covariance_diag.iter().sum::<f64>() / 3.0).sqrt()
    }
    pub fn diagonal_sigma_m(&self) -> [f64; 3] {
        [
            self.covariance_diag[0].sqrt(),
            self.covariance_diag[1].sqrt(),
            self.covariance_diag[2].sqrt(),
        ]
    }
}

impl PartialEq for GaussianEstimate3D {
    fn eq(&self, other: &Self) -> bool {
        self.mean == other.mean && self.covariance_diag == other.covariance_diag
    }
}

impl PublishableEstimate3D for GaussianEstimate3D {
    fn estimate(&self) -> &GaussianEstimate3D {
        self
    }
    fn source_count(&self) -> usize {
        1
    }
    fn timestamp_us(&self) -> u64 {
        0
    }
}

/// A peer's position estimate with trust weight.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PeerEstimate3D {
    pub peer_id: String,
    pub estimate: GaussianEstimate3D,
    pub trust_weight: f64,
    pub timestamp_us: u64,
    pub confidence: f64,
}

impl PeerEstimate3D {
    pub fn peer_estimate(&self) -> &GaussianEstimate3D {
        &self.estimate
    }
}

/// Trait for estimates that can be published to DHT.
pub trait PublishableEstimate3D {
    fn estimate(&self) -> &GaussianEstimate3D;
    fn peer_estimate(&self) -> &GaussianEstimate3D {
        self.estimate()
    }
    fn source_count(&self) -> usize;
    fn timestamp_us(&self) -> u64;
    fn confidence(&self) -> f64 {
        1.0
    }
}

/// Default publishable estimate implementation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DefaultPublishableEstimate3D {
    pub estimate: GaussianEstimate3D,
    pub source_count: usize,
    pub timestamp_us: u64,
}

impl PublishableEstimate3D for DefaultPublishableEstimate3D {
    fn estimate(&self) -> &GaussianEstimate3D {
        &self.estimate
    }
    fn source_count(&self) -> usize {
        self.source_count
    }
    fn timestamp_us(&self) -> u64 {
        self.timestamp_us
    }
}

/// Peer fusion engine using Covariance Intersection.
pub struct PeerFusion3D {
    peers: Vec<PeerEstimate3D>,
    max_peers: usize,
}

impl PeerFusion3D {
    pub fn new(max_peers: usize) -> Self {
        Self {
            peers: Vec::new(),
            max_peers,
        }
    }

    pub fn add_peer(&mut self, peer: PeerEstimate3D) {
        if self.peers.len() >= self.max_peers {
            self.peers.remove(0);
        }
        self.peers.push(peer);
    }

    pub fn fuse(&self) -> Option<GaussianEstimate3D> {
        if self.peers.is_empty() {
            return None;
        }
        let total_w: f64 = self.peers.iter().map(|p| p.trust_weight).sum();
        if total_w < 1e-10 {
            return None;
        }
        let mut mean = [0.0; 3];
        let mut cov = [0.0; 3];
        for p in &self.peers {
            let w = p.trust_weight / total_w;
            for i in 0..3 {
                mean[i] += w * p.estimate.mean[i];
                cov[i] += w * p.estimate.covariance_diag[i];
            }
        }
        Some(GaussianEstimate3D {
            mean,
            covariance_diag: cov,
            covariance: cov,
        })
    }

    pub fn peer_count(&self) -> usize {
        self.peers.len()
    }

    pub fn upsert_peer(&mut self, peer: PeerEstimate3D) {
        if let Some(existing) = self.peers.iter_mut().find(|p| p.peer_id == peer.peer_id) {
            *existing = peer;
        } else {
            self.add_peer(peer);
        }
    }

    pub fn fused_estimate(&self) -> Option<GaussianEstimate3D> {
        self.fuse()
    }
}

/// Covariance Intersection for two 3D Gaussian estimates.
pub fn covariance_intersection_3d(
    a: &GaussianEstimate3D,
    b: &GaussianEstimate3D,
    omega: f64,
) -> GaussianEstimate3D {
    let omega = omega.clamp(0.0, 1.0);
    let mut mean = [0.0; 3];
    let mut cov = [0.0; 3];
    for i in 0..3 {
        let inv_a = if a.covariance_diag[i] > 1e-15 {
            1.0 / a.covariance_diag[i]
        } else {
            1e15
        };
        let inv_b = if b.covariance_diag[i] > 1e-15 {
            1.0 / b.covariance_diag[i]
        } else {
            1e15
        };
        let fused_inv = omega * inv_a + (1.0 - omega) * inv_b;
        cov[i] = if fused_inv > 1e-15 {
            1.0 / fused_inv
        } else {
            1e15
        };
        mean[i] = cov[i] * (omega * inv_a * a.mean[i] + (1.0 - omega) * inv_b * b.mean[i]);
    }
    GaussianEstimate3D {
        mean,
        covariance_diag: cov,
        covariance: cov,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_ci() {
        let a = GaussianEstimate3D::new([1.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let b = GaussianEstimate3D::new([3.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let c = covariance_intersection_3d(&a, &b, 0.5);
        assert!((c.mean[0] - 2.0).abs() < 0.01);
        assert!(c.covariance_diag[0] < 1.0);
    }
}
