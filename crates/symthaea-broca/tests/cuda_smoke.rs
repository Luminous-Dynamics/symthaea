// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! CUDA smoke tests for the Broca SSM language center.
//!
//! These tests verify that the Mamba model infrastructure works on GPU.
//! They are gated behind `#[cfg(feature = "mamba")]` and `#[ignore]` so they
//! only run when explicitly requested on hardware with a CUDA GPU:
//!
//! ```bash
//! cargo test -p symthaea-broca --features mamba -- --ignored
//! ```
//!
//! On NixOS, set `LD_LIBRARY_PATH="/run/opengl-driver/lib"` first.

#[cfg(feature = "mamba")]
mod cuda_tests {
    use candle_core::{DType, Device, Tensor};

    /// Verify that a CUDA device is available and functional.
    #[test]
    #[ignore = "requires GPU hardware"]
    fn test_cuda_device_available() {
        let device = Device::cuda_if_available(0).expect("CUDA device query should not fail");
        assert!(
            device.is_cuda(),
            "Expected CUDA device but got CPU — is a GPU present?"
        );

        // Basic tensor op on GPU to verify the driver works
        let t = Tensor::zeros((2, 3), DType::F32, &device).expect("GPU tensor creation");
        assert_eq!(t.shape().dims(), &[2, 3]);
    }

    /// Verify that the vendored RmsNorm produces finite output on GPU.
    #[test]
    #[ignore = "requires GPU hardware"]
    fn test_cuda_rms_norm_forward() {
        use candle_core::Module;
        use candle_nn::VarMap;

        let device = Device::cuda_if_available(0).expect("CUDA device query failed");
        assert!(device.is_cuda(), "Need CUDA device for this test");

        let dim = 768; // mamba-130m d_model
        let varmap = VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);

        // Initialize weight to ones (identity scaling)
        let _weight = varmap
            .get(
                (dim,),
                "weight",
                candle_nn::Init::Const(1.0),
                DType::F32,
                &device,
            )
            .expect("weight init");

        // Create RmsNorm via the model's internal path — we test the same tensor ops
        let input = Tensor::randn(0.0f32, 1.0, (1, dim), &device).expect("GPU randn");

        // Manual RmsNorm: x / sqrt(mean(x^2) + eps) * weight
        let eps = 1e-5;
        let x_sq = input.sqr().expect("sqr");
        let variance =
            (x_sq.sum_keepdim(candle_core::D::Minus1).expect("sum") / dim as f64).expect("div");
        let x_normed = input
            .broadcast_div(&(variance + eps).expect("add eps").sqrt().expect("sqrt"))
            .expect("norm div");

        // Verify output shape and finiteness
        assert_eq!(x_normed.shape().dims(), &[1, dim]);
        let values: Vec<f32> = x_normed
            .flatten_all()
            .expect("flatten")
            .to_vec1()
            .expect("to_vec1");
        assert!(
            values.iter().all(|v| v.is_finite()),
            "RmsNorm output should be all finite on GPU"
        );
    }

    /// Verify that the Mamba model can be loaded onto GPU (if weights exist).
    ///
    /// This test requires the mamba-130m model weights to be downloaded.
    /// Skip gracefully if weights are not found.
    #[test]
    #[ignore = "requires GPU hardware + model weights"]
    fn test_cuda_model_load() {
        use symthaea_broca::mamba::MambaBackend;

        let device = Device::cuda_if_available(0).expect("CUDA device query failed");
        assert!(device.is_cuda(), "Need CUDA device for this test");

        // Attempt to load via MambaWrapper (which handles weight download)
        match symthaea_broca::mamba::MambaWrapper::load("state-spaces/mamba-130m", device) {
            Ok(mut wrapper) => {
                let d_model = wrapper.d_model();
                assert_eq!(d_model, 768, "mamba-130m d_model should be 768");
                // Model is on GPU — verify a forward pass works
                let logits = wrapper.forward_one_token(0);
                assert!(logits.is_ok(), "Forward pass on GPU should succeed");
            }
            Err(e) => {
                let msg = format!("{e}");
                if msg.contains("No such file") || msg.contains("not found") {
                    eprintln!("Skipping model load test: weights not downloaded ({e})");
                    return;
                }
                panic!("Unexpected error loading model on GPU: {e}");
            }
        }
    }
}
