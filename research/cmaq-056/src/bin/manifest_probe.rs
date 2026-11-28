#![forbid(unsafe_code)]

//! Deterministic canonical-manifest wiring probe.
//!
//! This reports content digests for a fixed two-expert Fourier inventory.
//! It is not a statistical Type-I, power, or source-authenticity experiment.

use std::error::Error;

use cmaq_046_convex_target_approachability::BoxTarget;
use cmaq_048_fourier_kernel_approachability::FourierFeatureMap;
use cmaq_053_compositional_risk_budget::Rational;
use cmaq_054_evidence_adaptive_kernel_family::KernelExpertConfig;
use cmaq_056_typed_null_manifest::{KernelFamilyManifest, NullCalibration};

const SHA_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const SHA_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
const SHA_C: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
const SHA_D: &str = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
const SHA_E: &str = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";
const SHA_F: &str = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";

fn expert(
    id: &str,
    response_frequency: f64,
    prior: Rational,
    response_map_id: &str,
    covariate_map_id: &str,
    target_id: &str,
) -> KernelExpertConfig {
    KernelExpertConfig {
        expert_id: id.into(),
        response_map: FourierFeatureMap::new(
            1,
            vec![vec![response_frequency]],
            response_map_id.into(),
        )
        .expect("fixed response map is valid"),
        covariate_map: FourierFeatureMap::new(
            1,
            vec![vec![0.5]],
            covariate_map_id.into(),
        )
        .expect("fixed covariate map is valid"),
        target: BoxTarget::new(vec![-0.01; 4], vec![0.01; 4])
            .expect("fixed zero-containing target is valid"),
        prior_weight: prior,
        target_sha256: target_id.into(),
        response_reference_sha256: SHA_E.into(),
        covariate_reference_sha256: SHA_F.into(),
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let experts = vec![
        expert(
            "low-frequency",
            1.0,
            Rational::new(1, 2)?,
            SHA_A,
            SHA_B,
            SHA_C,
        ),
        expert(
            "high-frequency",
            0.5,
            Rational::new(1, 2)?,
            SHA_D,
            SHA_E,
            SHA_F,
        ),
    ];

    let direct = KernelFamilyManifest::build(
        &experts,
        NullCalibration::direct_v1(),
    )?;
    let reference = KernelFamilyManifest::build(
        &experts,
        NullCalibration::reference_cmaq052_v1(Rational::new(1, 320)?),
    )?;

    println!("mode,portfolio_inventory_sha256,target_envelope_sha256,reference_bundle_sha256,calibration_contract_sha256,canonical_bytes,target_bytes,reference_bytes,total_features");
    for (mode, manifest) in [("direct", &direct), ("reference", &reference)] {
        let digest = manifest.digests();
        println!(
            "{},{},{},{},{},{},{},{},{}",
            mode,
            digest.portfolio_inventory_sha256,
            digest.target_envelope_sha256,
            digest.reference_bundle_sha256,
            digest.calibration_contract_sha256,
            manifest.canonical_bytes().len(),
            manifest.target_bytes().len(),
            manifest.reference_bytes().len(),
            manifest.total_feature_count(),
        );
    }
    Ok(())
}
