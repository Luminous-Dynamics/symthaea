//! Positioning stub for standalone builds.
use serde::{Deserialize, Serialize};

pub mod ranging {
    pub fn rssi_to_range(_rssi_dbm: f64, _tx_power: f64, _n: f64) -> f64 {
        0.0
    }
    pub fn uwb_tof_to_range(_tof_ns: f64) -> f64 {
        0.0
    }
    pub fn wifi_rtt_to_range(_rtt_ns: f64) -> f64 {
        0.0
    }
}

pub mod dead_reckoning {
    pub fn barometric_altitude(_pressure_hpa: f64, _ref_pressure: f64) -> f64 {
        0.0
    }
}

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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GaussianEstimate3D {
    pub mean: [f64; 3],
    pub covariance_diag: [f64; 3],
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
        let mut mean = [0.0; 3];
        let mut w_total = 0.0;
        for p in &self.peers {
            let w = p.trust_weight;
            for (m, pm) in mean.iter_mut().zip(p.estimate.mean.iter()) {
                *m += w * pm;
            }
            w_total += w;
        }
        if w_total > 0.0 {
            for m in &mut mean {
                *m /= w_total;
            }
        }
        Some(GaussianEstimate3D::new(mean, [1.0, 1.0, 1.0]))
    }
    pub fn peer_count(&self) -> usize {
        self.peers.len()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MeasurementModality {
    Rssi,
    LoraToA,
    UwbToF,
    WifiRtt,
    Gps,
    Barometric,
    Imu,
    Visual,
    Acoustic,
    AbsolutePosition,
    RelativePosition,
}

// Additional methods used by symthaea swarm service
impl PeerFusion3D {
    pub fn upsert_peer(&mut self, peer: PeerEstimate3D) {
        if let Some(pos) = self.peers.iter().position(|p| p.peer_id == peer.peer_id) {
            self.peers[pos] = peer;
        } else {
            self.add_peer(peer);
        }
    }
    pub fn fused_estimate(&self) -> Option<GaussianEstimate3D> {
        self.fuse()
    }
}
