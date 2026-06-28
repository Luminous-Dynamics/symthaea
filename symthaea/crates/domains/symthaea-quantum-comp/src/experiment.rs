//! Experiment-level manifests and claim boundaries.
//!
//! This module keeps the crate's research posture explicit. A report should not
//! merely contain numbers; it should also say what protocol was run, which claim
//! boundary applies, and which caveats remain unresolved.

use crate::substrate::SubstrateProfile;

/// Published experiment protocol identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExperimentProtocol {
    /// Classical XOR HDC binding baseline.
    ClassicalXorBinding,
    /// Classical simulation of phase-HDC binding.
    PhaseBindingSimulation,
    /// Pairwise parity/correlation sketch probe.
    CorrelationBindingSketch,
    /// Classical proxy for entanglement-mediated parity binding.
    EntanglementProxyBinding,
    /// Negative-control run using mismatched keys or random items.
    NegativeControl,
    /// Noise sweep over one or more binding probes.
    NoiseSweep,
}

/// Explicit boundary for what a result is allowed to claim.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClaimBoundary {
    /// Implementation sanity check only.
    ImplementationCheck,
    /// Local reproducible simulation result.
    LocalSimulation,
    /// Exported circuit artifact; execution happens elsewhere.
    CircuitExportOnly,
    /// External backend observation, requiring attached backend metadata.
    ExternalBackendObservation,
}

/// Reusable experiment manifest for reports that are not captured by the older
/// binding benchmark manifest.
#[derive(Debug, Clone, PartialEq)]
pub struct ExperimentManifest {
    /// Human-readable experiment name.
    pub name: String,
    /// Protocol identifier.
    pub protocol: ExperimentProtocol,
    /// Crate version string.
    pub crate_version: String,
    /// Deterministic seed.
    pub seed: u64,
    /// Hypervector or sketch dimension.
    pub dimension: usize,
    /// Trial count.
    pub trials: usize,
    /// Claim boundary for the result.
    pub claim_boundary: ClaimBoundary,
    /// Substrate assumptions.
    pub substrate: SubstrateProfile,
    /// Additional caveats.
    pub caveats: Vec<String>,
}

impl ExperimentManifest {
    /// Creates an alpha-series manifest for a local simulation protocol.
    pub fn local_simulation(
        name: impl Into<String>,
        protocol: ExperimentProtocol,
        seed: u64,
        dimension: usize,
        trials: usize,
        substrate: SubstrateProfile,
    ) -> Self {
        Self {
            name: name.into(),
            protocol,
            crate_version: env!("CARGO_PKG_VERSION").to_string(),
            seed,
            dimension,
            trials,
            claim_boundary: ClaimBoundary::LocalSimulation,
            substrate,
            caveats: vec![
                "Research probe only; no quantum consciousness claim.".to_string(),
                "No quantum advantage claim.".to_string(),
                "Non-cryptographic reproducibility fingerprint.".to_string(),
            ],
        }
    }

    /// Returns a deterministic, non-cryptographic fingerprint for reproducibility reports.
    pub fn reproducibility_fingerprint(&self) -> u64 {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        fn mix(h: &mut u64, bytes: &[u8]) {
            for b in bytes {
                *h ^= *b as u64;
                *h = h.wrapping_mul(0x0000_0100_0000_01B3);
            }
        }
        mix(&mut h, self.name.as_bytes());
        mix(&mut h, format!("{:?}", self.protocol).as_bytes());
        mix(&mut h, self.crate_version.as_bytes());
        mix(&mut h, &self.seed.to_le_bytes());
        mix(&mut h, &self.dimension.to_le_bytes());
        mix(&mut h, &self.trials.to_le_bytes());
        mix(&mut h, format!("{:?}", self.claim_boundary).as_bytes());
        mix(&mut h, self.substrate.backend_name.as_bytes());
        h
    }

    /// Returns a compact line-oriented manifest summary.
    pub fn to_text(&self) -> String {
        format!(
            "{}\nprotocol={:?} crate_version={} dimension={} trials={} seed={} claim_boundary={:?} substrate={:?} confidence={:?} fingerprint={:016x}",
            self.name,
            self.protocol,
            self.crate_version,
            self.dimension,
            self.trials,
            self.seed,
            self.claim_boundary,
            self.substrate.backend,
            self.substrate.confidence,
            self.reproducibility_fingerprint(),
        )
    }
}
