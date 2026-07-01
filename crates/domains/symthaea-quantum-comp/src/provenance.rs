//! Lightweight provenance helpers for reproducible research reports.
//!
//! These helpers are deliberately dependency-free. They are not cryptographic
//! commitments. Use Mycelix or a real digest crate for signed artifact receipts.

use crate::benchmark::BenchmarkManifest;

/// Dependency-free run environment metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunEnvironment {
    /// Crate version recorded at compile time.
    pub crate_version: &'static str,
    /// Optional git commit or external revision label supplied by the caller.
    pub revision: Option<String>,
    /// Optional toolchain label supplied by the caller.
    pub toolchain: Option<String>,
    /// Optional host or platform label supplied by the caller.
    pub host: Option<String>,
    /// Freeform caveat for local machine state, Nix flake, container, or backend assumptions.
    pub caveat: Option<String>,
}

impl RunEnvironment {
    /// Returns a minimal environment record for examples and tests.
    pub fn local_unknown() -> Self {
        Self {
            crate_version: env!("CARGO_PKG_VERSION"),
            revision: option_env!("SYMTHAEA_QUANTUM_COMP_REVISION").map(str::to_string),
            toolchain: option_env!("SYMTHAEA_QUANTUM_COMP_TOOLCHAIN").map(str::to_string),
            host: option_env!("SYMTHAEA_QUANTUM_COMP_HOST").map(str::to_string),
            caveat: Some("local environment was not externally attested".to_string()),
        }
    }

    /// Returns a compact line-oriented report.
    pub fn to_text(&self) -> String {
        format!(
            "crate_version={} revision={} toolchain={} host={} caveat={}",
            self.crate_version,
            self.revision.as_deref().unwrap_or("unknown"),
            self.toolchain.as_deref().unwrap_or("unknown"),
            self.host.as_deref().unwrap_or("unknown"),
            self.caveat.as_deref().unwrap_or("none"),
        )
    }

    /// Returns a deterministic, non-cryptographic environment fingerprint.
    pub fn fingerprint(&self) -> u64 {
        let mut h = fnv1a64(self.crate_version.as_bytes());
        mix_optional(&mut h, self.revision.as_deref());
        mix_optional(&mut h, self.toolchain.as_deref());
        mix_optional(&mut h, self.host.as_deref());
        mix_optional(&mut h, self.caveat.as_deref());
        h
    }
}

/// A manifest plus local environment fingerprint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReproducibilityRecord {
    /// Fingerprint derived from the benchmark manifest.
    pub manifest_fingerprint: u64,
    /// Fingerprint derived from the local environment record.
    pub environment_fingerprint: u64,
    /// Combined non-cryptographic run fingerprint.
    pub combined_fingerprint: u64,
}

impl ReproducibilityRecord {
    /// Builds a record from a benchmark manifest and local run environment.
    pub fn from_manifest_and_environment(
        manifest: &BenchmarkManifest,
        environment: &RunEnvironment,
    ) -> Self {
        let manifest_fingerprint = manifest.reproducibility_fingerprint();
        let environment_fingerprint = environment.fingerprint();
        let mut combined_fingerprint = manifest_fingerprint;
        mix_u64(&mut combined_fingerprint, environment_fingerprint);
        Self {
            manifest_fingerprint,
            environment_fingerprint,
            combined_fingerprint,
        }
    }

    /// Returns a compact report line.
    pub fn to_text(&self) -> String {
        format!(
            "manifest_fingerprint={:016x} environment_fingerprint={:016x} combined_fingerprint={:016x}",
            self.manifest_fingerprint, self.environment_fingerprint, self.combined_fingerprint,
        )
    }
}

/// Computes a deterministic FNV-1a 64-bit hash for report fingerprints.
///
/// This is not cryptographic and must not be used as a security commitment.
pub fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01B3);
    }
    h
}

fn mix_optional(h: &mut u64, value: Option<&str>) {
    match value {
        Some(v) => mix_bytes(h, v.as_bytes()),
        None => mix_bytes(h, b"<none>"),
    }
}

fn mix_bytes(h: &mut u64, bytes: &[u8]) {
    for b in bytes {
        *h ^= *b as u64;
        *h = h.wrapping_mul(0x0000_0100_0000_01B3);
    }
}

fn mix_u64(h: &mut u64, value: u64) {
    mix_bytes(h, &value.to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn environment_fingerprint_is_stable() {
        let a = RunEnvironment::local_unknown();
        let b = RunEnvironment::local_unknown();
        assert_eq!(a.fingerprint(), b.fingerprint());
    }
}
