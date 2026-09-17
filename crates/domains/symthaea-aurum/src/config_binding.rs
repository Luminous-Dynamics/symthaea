// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use crate::{AurumConfig, AurumError, AurumTopology};

const CONFIG_DOMAIN: &[u8] = b"symthaea:aurum:config:v1\0";

impl AurumConfig {
    /// Canonical behavior-affecting configuration bytes for PHYS-002 binding.
    ///
    /// The per-run seed is intentionally excluded because PHYS-002 binds seeds
    /// separately. Every model parameter that changes the deterministic update
    /// rule or seeded topology generation is included here.
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, AurumError> {
        self.validate()?;
        let mut out = Vec::with_capacity(160);
        out.extend_from_slice(CONFIG_DOMAIN);
        push_u64(&mut out, self.junctions as u64);
        match self.topology {
            AurumTopology::Ring { radius } => {
                out.push(0);
                push_u64(&mut out, radius as u64);
                push_u64(&mut out, 0);
            }
            AurumTopology::SmallWorld {
                radius,
                shortcuts_per_node,
            } => {
                out.push(1);
                push_u64(&mut out, radius as u64);
                push_u64(&mut out, shortcuts_per_node as u64);
            }
        }
        for value in [
            self.baseline_conductance,
            self.max_conductance,
            self.switching_threshold,
            self.switching_gain,
            self.relaxation,
            self.threshold_disorder,
            self.stochasticity,
            self.input_coupling,
            self.recurrent_coupling,
        ] {
            push_f64(&mut out, value);
        }
        Ok(out)
    }

    /// Domain-separated BLAKE3 digest of the canonical configuration bytes.
    pub fn configuration_digest(&self) -> Result<[u8; 32], AurumError> {
        let bytes = self.canonical_bytes()?;
        Ok(*blake3::hash(&bytes).as_bytes())
    }
}

fn push_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_f64(out: &mut Vec<u8>, value: f64) {
    out.extend_from_slice(&value.to_bits().to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn digest_is_deterministic() {
        let a = AurumConfig::default();
        let b = AurumConfig::default();
        assert_eq!(a.configuration_digest().unwrap(), b.configuration_digest().unwrap());
    }

    #[test]
    fn every_behavioral_parameter_changes_config_identity() {
        let base = AurumConfig::default();
        let base_digest = base.configuration_digest().unwrap();

        let mut variants = Vec::new();
        let mut c = base.clone();
        c.junctions += 1;
        variants.push(c);
        let mut c = base.clone();
        c.topology = AurumTopology::Ring { radius: 1 };
        variants.push(c);
        let mut c = base.clone();
        c.baseline_conductance += 0.01;
        variants.push(c);
        let mut c = base.clone();
        c.max_conductance -= 0.01;
        variants.push(c);
        let mut c = base.clone();
        c.switching_threshold += 0.01;
        variants.push(c);
        let mut c = base.clone();
        c.switching_gain += 0.01;
        variants.push(c);
        let mut c = base.clone();
        c.relaxation += 0.01;
        variants.push(c);
        let mut c = base.clone();
        c.threshold_disorder += 0.01;
        variants.push(c);
        let mut c = base.clone();
        c.stochasticity += 0.01;
        variants.push(c);
        let mut c = base.clone();
        c.input_coupling += 0.01;
        variants.push(c);
        let mut c = base.clone();
        c.recurrent_coupling += 0.01;
        variants.push(c);

        for variant in variants {
            assert_ne!(base_digest, variant.configuration_digest().unwrap());
        }
    }

    #[test]
    fn invalid_config_cannot_be_committed() {
        let mut config = AurumConfig::default();
        config.junctions = 1;
        assert!(config.configuration_digest().is_err());
    }
}
