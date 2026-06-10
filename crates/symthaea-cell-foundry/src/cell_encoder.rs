// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! HDC encoding of cell state into 16,384D hypervectors.
//!
//! Each cell state field is encoded as a random HV scaled by the field value
//! (thermometer encoding), then all fields are bound and bundled into a single
//! holographic representation.

use symthaea_core::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION};

use crate::types::{CellState, CellType, CultureEnvironment, DifferentiatedType};

// ═══════════════════════════════════════════════════════════════════════════════
// SEED CONSTANTS (deterministic per field/type)
// ═══════════════════════════════════════════════════════════════════════════════

const SEED_VIABILITY: u64 = 0xC311_A001;
const SEED_PLURIPOTENCY: u64 = 0xC311_A002;
const SEED_DIVISION_COUNT: u64 = 0xC311_A003;
const SEED_AGE_HOURS: u64 = 0xC311_A004;
const SEED_METHYLATION: u64 = 0xC311_A005;
const SEED_GENE_BASE: u64 = 0xC311_B000;
const SEED_GENE_ROLE: u64 = 0xC311_B100;

const SEED_ENV_TEMP: u64 = 0xE4E1_0001;
const SEED_ENV_CO2: u64 = 0xE4E1_0002;
const SEED_ENV_O2: u64 = 0xE4E1_0003;
const SEED_ENV_PH: u64 = 0xE4E1_0004;
const SEED_ENV_HUMIDITY: u64 = 0xE4E1_0005;
const SEED_ENV_MEDIA_VOL: u64 = 0xE4E1_0006;
const SEED_ENV_PASSAGE: u64 = 0xE4E1_0007;

// ═══════════════════════════════════════════════════════════════════════════════
// ENCODING
// ═══════════════════════════════════════════════════════════════════════════════

/// Encode a scalar value as a thermometer-coded HV: random HV scaled by value.
fn encode_scalar(value: f64, seed: u64) -> ContinuousHV {
    ContinuousHV::random(HDC_DIMENSION, seed).scale(value as f32)
}

/// Map a cell type to a unique deterministic seed.
fn cell_type_seed(cell_type: CellType) -> u64 {
    match cell_type {
        CellType::Somatic => 0xCE11_0001,
        CellType::Pluripotent => 0xCE11_0002,
        CellType::PrimordialGerm => 0xCE11_0003,
        CellType::Oogonium => 0xCE11_0004,
        CellType::Oocyte => 0xCE11_0005,
        CellType::Spermatogonium => 0xCE11_0006,
        CellType::Spermatocyte => 0xCE11_0007,
        CellType::Embryonic => 0xCE11_0008,
        CellType::Differentiated(dt) => match dt {
            DifferentiatedType::Neural => 0xCE11_1001,
            DifferentiatedType::Cardiac => 0xCE11_1002,
            DifferentiatedType::Hepatic => 0xCE11_1003,
            DifferentiatedType::Renal => 0xCE11_1004,
            DifferentiatedType::Epithelial => 0xCE11_1005,
            DifferentiatedType::Mesenchymal => 0xCE11_1006,
        },
    }
}

/// Encode a cell state into a 16,384D holographic hypervector.
///
/// The encoding uses HDC bind-and-bundle:
/// 1. Each field is encoded as a type-specific random HV scaled by the field value
/// 2. Gene expression is encoded as `bind(gene_index_hv, expression_level_hv)` per gene
/// 3. All field HVs are bundled (superposition) into the final cell state HV
///
/// This preserves similarity structure: cells with similar states produce similar HVs.
pub fn encode_cell_state(state: &CellState) -> ContinuousHV {
    let type_hv = ContinuousHV::random(HDC_DIMENSION, cell_type_seed(state.cell_type));
    let viability_hv = encode_scalar(state.viability, SEED_VIABILITY);
    let pluripotency_hv = encode_scalar(state.pluripotency_score, SEED_PLURIPOTENCY);
    let division_hv = encode_scalar(
        (state.division_count as f64).min(100.0) / 100.0,
        SEED_DIVISION_COUNT,
    );
    let age_hv = encode_scalar((state.age_hours / 1000.0).min(1.0), SEED_AGE_HOURS);
    let methylation_hv = encode_scalar(state.methylation_score, SEED_METHYLATION);

    // Encode gene expression: bind(gene_role_hv, expression_level_hv) per gene
    let gene_hvs: Vec<ContinuousHV> = state
        .gene_expression
        .iter()
        .enumerate()
        .map(|(i, &expr)| {
            let gene_role = ContinuousHV::random(HDC_DIMENSION, SEED_GENE_ROLE + i as u64);
            let gene_level = encode_scalar(expr, SEED_GENE_BASE + i as u64);
            gene_role.bind(&gene_level)
        })
        .collect();

    let gene_refs: Vec<&ContinuousHV> = gene_hvs.iter().collect();
    let gene_bundle = if gene_refs.is_empty() {
        ContinuousHV::zero(HDC_DIMENSION)
    } else {
        ContinuousHV::bundle(&gene_refs)
    };

    // Bundle all components
    let all_refs: Vec<&ContinuousHV> = vec![
        &type_hv,
        &viability_hv,
        &pluripotency_hv,
        &division_hv,
        &age_hv,
        &methylation_hv,
        &gene_bundle,
    ];
    ContinuousHV::bundle(&all_refs)
}

/// Encode a culture environment into a 16,384D holographic hypervector.
pub fn encode_environment(env: &CultureEnvironment) -> ContinuousHV {
    let temp_hv = encode_scalar(env.temperature_celsius / 45.0, SEED_ENV_TEMP);
    let co2_hv = encode_scalar(env.co2_percent / 10.0, SEED_ENV_CO2);
    let o2_hv = encode_scalar(env.o2_percent / 25.0, SEED_ENV_O2);
    let ph_hv = encode_scalar((env.ph - 6.0) / 2.0, SEED_ENV_PH);
    let humidity_hv = encode_scalar(env.humidity_percent / 100.0, SEED_ENV_HUMIDITY);
    let media_hv = encode_scalar((env.media_volume_ml / 20.0).min(1.0), SEED_ENV_MEDIA_VOL);
    let passage_hv = encode_scalar(
        (env.passage_number as f64).min(50.0) / 50.0,
        SEED_ENV_PASSAGE,
    );

    ContinuousHV::bundle(&[
        &temp_hv,
        &co2_hv,
        &o2_hv,
        &ph_hv,
        &humidity_hv,
        &media_hv,
        &passage_hv,
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_same_state_same_hv() {
        let state = CellState::new_somatic();
        let hv1 = encode_cell_state(&state);
        let hv2 = encode_cell_state(&state);
        let sim = hv1.similarity(&hv2);
        assert!(
            (sim - 1.0).abs() < 1e-5,
            "Same state must produce identical HV, similarity={}",
            sim
        );
    }

    #[test]
    fn test_different_types_dissimilar() {
        let somatic = CellState::new_somatic();
        let ipsc = CellState::new_ipsc();
        let hv_somatic = encode_cell_state(&somatic);
        let hv_ipsc = encode_cell_state(&ipsc);
        let sim = hv_somatic.similarity(&hv_ipsc);
        assert!(
            sim < 0.7,
            "Different cell types should produce dissimilar HVs, similarity={}",
            sim
        );
    }

    #[test]
    fn test_encode_dimension() {
        let state = CellState::new_somatic();
        let hv = encode_cell_state(&state);
        assert_eq!(hv.dim(), HDC_DIMENSION);
    }

    #[test]
    fn test_encode_nonzero() {
        let state = CellState::new_somatic();
        let hv = encode_cell_state(&state);
        assert!(hv.norm() > 0.0, "Encoded HV should be non-zero");
    }

    #[test]
    fn test_similar_states_high_similarity() {
        let mut state1 = CellState::new_somatic();
        let mut state2 = CellState::new_somatic();
        // Very small difference
        state1.viability = 0.95;
        state2.viability = 0.94;
        let hv1 = encode_cell_state(&state1);
        let hv2 = encode_cell_state(&state2);
        let sim = hv1.similarity(&hv2);
        assert!(
            sim > 0.95,
            "Nearly identical states should have high similarity, got {}",
            sim
        );
    }

    #[test]
    fn test_environment_encoding_deterministic() {
        let env = CultureEnvironment::standard();
        let hv1 = encode_environment(&env);
        let hv2 = encode_environment(&env);
        let sim = hv1.similarity(&hv2);
        assert!(
            (sim - 1.0).abs() < 1e-5,
            "Same environment should produce identical HV"
        );
    }

    #[test]
    fn test_environment_encoding_dimension() {
        let env = CultureEnvironment::standard();
        let hv = encode_environment(&env);
        assert_eq!(hv.dim(), HDC_DIMENSION);
    }

    #[test]
    fn test_different_environments_dissimilar() {
        let standard = CultureEnvironment::standard();
        let hypoxic = CultureEnvironment::hypoxic();
        let hv_std = encode_environment(&standard);
        let hv_hyp = encode_environment(&hypoxic);
        let sim = hv_std.similarity(&hv_hyp);
        assert!(
            sim < 0.99,
            "Different environments should produce somewhat dissimilar HVs, similarity={}",
            sim
        );
    }

    #[test]
    fn test_all_cell_types_unique_seeds() {
        let types = vec![
            CellType::Somatic,
            CellType::Pluripotent,
            CellType::PrimordialGerm,
            CellType::Oogonium,
            CellType::Oocyte,
            CellType::Spermatogonium,
            CellType::Spermatocyte,
            CellType::Embryonic,
            CellType::Differentiated(DifferentiatedType::Neural),
            CellType::Differentiated(DifferentiatedType::Cardiac),
            CellType::Differentiated(DifferentiatedType::Hepatic),
            CellType::Differentiated(DifferentiatedType::Renal),
            CellType::Differentiated(DifferentiatedType::Epithelial),
            CellType::Differentiated(DifferentiatedType::Mesenchymal),
        ];
        let seeds: Vec<u64> = types.iter().map(|ct| cell_type_seed(*ct)).collect();
        // All seeds must be unique
        for i in 0..seeds.len() {
            for j in (i + 1)..seeds.len() {
                assert_ne!(
                    seeds[i], seeds[j],
                    "Seed collision for types {} and {}",
                    i, j
                );
            }
        }
    }

    #[test]
    fn test_gene_expression_empty() {
        let mut state = CellState::new_somatic();
        state.gene_expression = vec![];
        let hv = encode_cell_state(&state);
        assert!(
            hv.norm() > 0.0,
            "Should still encode even with empty gene expression"
        );
    }
}
