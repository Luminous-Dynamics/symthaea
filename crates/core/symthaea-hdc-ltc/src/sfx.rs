// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! # SFX (Symthaea Format eXport) Serializer & Deserializer
//!
//! Provides direct, high-performance binary serialization/deserialization for
//! `HdcLtcUnifiedNetwork` and `HdcLtcUnifiedNeuron` parameters, supporting
//! zero-copy mmap-friendly layout.

use crate::config::{Activation, NetworkConfig, NeuronConfig};
use crate::continuous_hv::ContinuousHV;
use crate::network::HdcLtcUnifiedNetwork;
use crate::neuron::HdcLtcUnifiedNeuron;
use std::io::{self, Read, Write};

const SFX_MAGIC: &[u8; 4] = b"SFX\x02";

/// Serialize a network to SFX format.
pub fn export_network_to_sfx(
    network: &HdcLtcUnifiedNetwork,
    writer: &mut impl Write,
) -> io::Result<()> {
    writer.write_all(SFX_MAGIC)?;

    let config = network.config();
    // 1. NetworkConfig
    let layer_count = config.layer_sizes.len() as u32;
    writer.write_all(&layer_count.to_le_bytes())?;
    for &size in &config.layer_sizes {
        writer.write_all(&(size as u32).to_le_bytes())?;
    }
    writer.write_all(&[if config.use_layer_binding { 1 } else { 0 }])?;
    writer.write_all(&[if config.skip_connections { 1 } else { 0 }])?;

    // 2. NeuronConfig
    let n_cfg = &config.neuron_config;
    writer.write_all(&(n_cfg.dim as u32).to_le_bytes())?;
    writer.write_all(&n_cfg.tau_base.to_le_bytes())?;
    writer.write_all(&n_cfg.backbone_tau.to_le_bytes())?;

    match n_cfg.activation {
        Activation::Tanh => {
            writer.write_all(&[0])?;
            writer.write_all(&0.0f32.to_le_bytes())?;
        }
        Activation::Sigmoid => {
            writer.write_all(&[1])?;
            writer.write_all(&0.0f32.to_le_bytes())?;
        }
        Activation::SiLU => {
            writer.write_all(&[2])?;
            writer.write_all(&0.0f32.to_le_bytes())?;
        }
        Activation::Identity => {
            writer.write_all(&[3])?;
            writer.write_all(&0.0f32.to_le_bytes())?;
        }
        Activation::BoundedTanh { bound } => {
            writer.write_all(&[4])?;
            writer.write_all(&bound.to_le_bytes())?;
        }
    }

    writer.write_all(&n_cfg.learning_rate.to_le_bytes())?;
    writer.write_all(&n_cfg.momentum.to_le_bytes())?;
    writer.write_all(&n_cfg.weight_decay.to_le_bytes())?;
    writer.write_all(&n_cfg.gating_steepness.to_le_bytes())?;
    writer.write_all(&n_cfg.tau_min.to_le_bytes())?;
    writer.write_all(&n_cfg.tau_max.to_le_bytes())?;
    writer.write_all(&n_cfg.tau_coupling.to_le_bytes())?;

    // 3. Layer Bindings
    for binding in &network.layer_bindings {
        for &val in &binding.values {
            writer.write_all(&val.to_le_bytes())?;
        }
    }

    // 4. Neurons
    for layer_idx in 0..network.layer_count() {
        if let Some(layer) = network.layer(layer_idx) {
            for neuron in layer {
                // weight_hv
                for &val in &neuron.weight_hv.values {
                    writer.write_all(&val.to_le_bytes())?;
                }
                // input_mask
                for &val in &neuron.input_mask.values {
                    writer.write_all(&val.to_le_bytes())?;
                }
                // tau_modulator
                for &val in &neuron.tau_modulator.values {
                    writer.write_all(&val.to_le_bytes())?;
                }
                // gate_weight
                for &val in &neuron.gate_weight.values {
                    writer.write_all(&val.to_le_bytes())?;
                }
                // gate_bias
                for &val in &neuron.gate_bias.values {
                    writer.write_all(&val.to_le_bytes())?;
                }
            }
        }
    }

    Ok(())
}

/// Deserialize a network from SFX format.
pub fn import_network_from_sfx(reader: &mut impl Read) -> io::Result<HdcLtcUnifiedNetwork> {
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic)?;
    if &magic != SFX_MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Invalid SFX magic header",
        ));
    }

    // 1. NetworkConfig
    let mut buf4 = [0u8; 4];
    reader.read_exact(&mut buf4)?;
    let layer_count = u32::from_le_bytes(buf4) as usize;

    let mut layer_sizes = Vec::with_capacity(layer_count);
    for _ in 0..layer_count {
        reader.read_exact(&mut buf4)?;
        layer_sizes.push(u32::from_le_bytes(buf4) as usize);
    }

    let mut bool_buf = [0u8; 2];
    reader.read_exact(&mut bool_buf)?;
    let use_layer_binding = bool_buf[0] != 0;
    let skip_connections = bool_buf[1] != 0;

    // 2. NeuronConfig
    reader.read_exact(&mut buf4)?;
    let dim = u32::from_le_bytes(buf4) as usize;

    let mut f32_buf = [0u8; 4];
    reader.read_exact(&mut f32_buf)?;
    let tau_base = f32::from_le_bytes(f32_buf);

    reader.read_exact(&mut f32_buf)?;
    let backbone_tau = f32::from_le_bytes(f32_buf);

    let mut act_type = [0u8; 1];
    reader.read_exact(&mut act_type)?;
    reader.read_exact(&mut f32_buf)?;
    let act_bound = f32::from_le_bytes(f32_buf);

    let activation = match act_type[0] {
        0 => Activation::Tanh,
        1 => Activation::Sigmoid,
        2 => Activation::SiLU,
        3 => Activation::Identity,
        4 => Activation::BoundedTanh { bound: act_bound },
        _ => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Unknown activation type in SFX",
            ));
        }
    };

    reader.read_exact(&mut f32_buf)?;
    let learning_rate = f32::from_le_bytes(f32_buf);

    reader.read_exact(&mut f32_buf)?;
    let momentum = f32::from_le_bytes(f32_buf);

    reader.read_exact(&mut f32_buf)?;
    let weight_decay = f32::from_le_bytes(f32_buf);

    reader.read_exact(&mut f32_buf)?;
    let gating_steepness = f32::from_le_bytes(f32_buf);

    reader.read_exact(&mut f32_buf)?;
    let tau_min = f32::from_le_bytes(f32_buf);

    reader.read_exact(&mut f32_buf)?;
    let tau_max = f32::from_le_bytes(f32_buf);

    reader.read_exact(&mut f32_buf)?;
    let tau_coupling = f32::from_le_bytes(f32_buf);

    let neuron_config = NeuronConfig {
        dim,
        tau_base,
        backbone_tau,
        activation,
        learning_rate,
        momentum,
        weight_decay,
        gating_steepness,
        tau_min,
        tau_max,
        tau_coupling,
    };

    let network_config = NetworkConfig {
        layer_sizes: layer_sizes.clone(),
        neuron_config,
        use_layer_binding,
        skip_connections,
    };

    // Construct target network skeleton
    let mut network = HdcLtcUnifiedNetwork::new(network_config, 0);

    // 3. Layer Bindings
    for binding in &mut network.layer_bindings {
        for val in &mut binding.values {
            reader.read_exact(&mut f32_buf)?;
            *val = f32::from_le_bytes(f32_buf);
        }
    }

    // 4. Neurons
    for layer_idx in 0..network.layer_count() {
        if let Some(layer) = network.layer_mut(layer_idx) {
            for neuron in layer {
                // weight_hv
                for val in &mut neuron.weight_hv.values {
                    reader.read_exact(&mut f32_buf)?;
                    *val = f32::from_le_bytes(f32_buf);
                }
                // input_mask
                for val in &mut neuron.input_mask.values {
                    reader.read_exact(&mut f32_buf)?;
                    *val = f32::from_le_bytes(f32_buf);
                }
                // tau_modulator
                for val in &mut neuron.tau_modulator.values {
                    reader.read_exact(&mut f32_buf)?;
                    *val = f32::from_le_bytes(f32_buf);
                }
                // gate_weight
                for val in &mut neuron.gate_weight.values {
                    reader.read_exact(&mut f32_buf)?;
                    *val = f32::from_le_bytes(f32_buf);
                }
                // gate_bias
                for val in &mut neuron.gate_bias.values {
                    reader.read_exact(&mut f32_buf)?;
                    *val = f32::from_le_bytes(f32_buf);
                }
            }
        }
    }

    Ok(network)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_sfx_network_roundtrip() {
        let config = NetworkConfig {
            layer_sizes: vec![2, 3],
            neuron_config: NeuronConfig {
                dim: 128,
                activation: Activation::BoundedTanh { bound: 0.8 },
                ..NeuronConfig::default()
            },
            use_layer_binding: true,
            skip_connections: true,
        };

        // Create with a seed to get non-zero randomized weights
        let net_before = HdcLtcUnifiedNetwork::new(config, 42);

        let mut buffer = Vec::new();
        export_network_to_sfx(&net_before, &mut buffer).unwrap();

        let mut cursor = Cursor::new(buffer);
        let net_after = import_network_from_sfx(&mut cursor).unwrap();

        assert_eq!(net_after.layer_count(), net_before.layer_count());
        assert_eq!(net_after.neuron_count(), net_before.neuron_count());
        assert_eq!(
            net_after.config().neuron_config.dim,
            net_before.config().neuron_config.dim
        );
        assert_eq!(
            net_after.config().use_layer_binding,
            net_before.config().use_layer_binding
        );
        assert_eq!(
            net_after.config().skip_connections,
            net_before.config().skip_connections
        );

        // Check activation
        match (
            &net_before.config().neuron_config.activation,
            &net_after.config().neuron_config.activation,
        ) {
            (Activation::BoundedTanh { bound: b1 }, Activation::BoundedTanh { bound: b2 }) => {
                assert!((b1 - b2).abs() < 1e-6);
            }
            _ => panic!("Activation mismatch"),
        }

        // Compare some weights
        let l0_n0_before = &net_before.layer(0).unwrap()[0];
        let l0_n0_after = &net_after.layer(0).unwrap()[0];
        assert_eq!(l0_n0_after.weight_hv.values, l0_n0_before.weight_hv.values);
        assert_eq!(l0_n0_after.gate_bias.values, l0_n0_before.gate_bias.values);

        // Compare layer bindings
        assert_eq!(
            net_after.layer_binding(1).values,
            net_before.layer_binding(1).values
        );
    }
}
