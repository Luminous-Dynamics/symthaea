// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! CanaryCNN: Small frozen CNN for gradient quality validation
//!
//! Architecture (MNIST):
//! - Input: 28x28 grayscale image
//! - Conv2D(1→16, 3x3, stride=1, padding=0) → ReLU → MaxPool(2x2)
//!   Output: 16 x 13 x 13 (= 2704)
//! - Conv2D(16→32, 3x3, stride=1, padding=0) → ReLU → MaxPool(2x2)
//!   Output: 32 x 5 x 5 (= 800)
//! - Flatten → FC(800→128) → ReLU → FC(128→10)
//! - Output: logits [10]

use crate::{weights, Fixed};

/// MNIST image dimensions
pub const IMG_HEIGHT: usize = 28;
pub const IMG_WIDTH: usize = 28;
pub const IMG_SIZE: usize = IMG_HEIGHT * IMG_WIDTH;

/// Number of output classes (MNIST digits 0-9)
pub const NUM_CLASSES: usize = 10;

/// Conv2D layer with 3x3 kernel
///
/// # Arguments
/// * `input` - Input tensor [in_channels, height, width]
/// * `weights` - Conv weights [out_channels, in_channels, 3, 3]
/// * `bias` - Bias [out_channels]
/// * `in_channels` - Number of input channels
/// * `out_channels` - Number of output channels
/// * `in_h` - Input height
/// * `in_w` - Input width
///
/// # Returns
/// Output tensor [out_channels, out_h, out_w] where:
/// - out_h = in_h - 2 (3x3 kernel, no padding, stride=1)
/// - out_w = in_w - 2
pub fn conv2d_3x3(
    input: &[Fixed],
    weights: &[Fixed],
    bias: &[Fixed],
    in_channels: usize,
    out_channels: usize,
    in_h: usize,
    in_w: usize,
) -> Vec<Fixed> {
    let out_h = in_h - 2; // 3x3 kernel, no padding
    let out_w = in_w - 2;
    let out_size = out_channels * out_h * out_w;
    let mut output = vec![Fixed::ZERO; out_size];

    // For each output channel
    for oc in 0..out_channels {
        let bias_val = bias[oc];

        // For each output spatial position
        for oh in 0..out_h {
            for ow in 0..out_w {
                let mut sum = bias_val;

                // Convolve 3x3 kernel across all input channels
                for ic in 0..in_channels {
                    for kh in 0..3 {
                        for kw in 0..3 {
                            let in_h_idx = oh + kh;
                            let in_w_idx = ow + kw;

                            // Input index: [ic, in_h_idx, in_w_idx]
                            let in_idx = ic * in_h * in_w + in_h_idx * in_w + in_w_idx;

                            // Weight index: [oc, ic, kh, kw]
                            let w_idx = oc * in_channels * 9 + ic * 9 + kh * 3 + kw;

                            sum = sum + input[in_idx] * weights[w_idx];
                        }
                    }
                }

                // Output index: [oc, oh, ow]
                let out_idx = oc * out_h * out_w + oh * out_w + ow;
                output[out_idx] = sum;
            }
        }
    }

    output
}

/// ReLU activation: max(0, x) for each element
pub fn relu(input: &[Fixed]) -> Vec<Fixed> {
    input.iter().map(|&x| x.relu()).collect()
}

/// MaxPool2D with 2x2 kernel and stride 2
///
/// # Arguments
/// * `input` - Input tensor [channels, height, width]
/// * `channels` - Number of channels
/// * `in_h` - Input height
/// * `in_w` - Input width
///
/// # Returns
/// Output tensor [channels, out_h, out_w] where:
/// - out_h = in_h / 2
/// - out_w = in_w / 2
pub fn maxpool2d_2x2(input: &[Fixed], channels: usize, in_h: usize, in_w: usize) -> Vec<Fixed> {
    let out_h = in_h / 2;
    let out_w = in_w / 2;
    let out_size = channels * out_h * out_w;
    let mut output = vec![Fixed::ZERO; out_size];

    for c in 0..channels {
        for oh in 0..out_h {
            for ow in 0..out_w {
                let ih = oh * 2;
                let iw = ow * 2;

                // Find max of 2x2 window
                let mut max_val = Fixed::from_raw(i32::MIN);
                for dh in 0..2 {
                    for dw in 0..2 {
                        let in_idx = c * in_h * in_w + (ih + dh) * in_w + (iw + dw);
                        max_val = max_val.max(input[in_idx]);
                    }
                }

                let out_idx = c * out_h * out_w + oh * out_w + ow;
                output[out_idx] = max_val;
            }
        }
    }

    output
}

/// Fully connected (linear) layer
///
/// # Arguments
/// * `input` - Input vector [in_features]
/// * `weights` - Weight matrix [out_features, in_features]
/// * `bias` - Bias vector [out_features]
/// * `in_features` - Input size
/// * `out_features` - Output size
///
/// # Returns
/// Output vector [out_features]
pub fn fc_layer(
    input: &[Fixed],
    weights: &[Fixed],
    bias: &[Fixed],
    in_features: usize,
    out_features: usize,
) -> Vec<Fixed> {
    let mut output = vec![Fixed::ZERO; out_features];

    for o in 0..out_features {
        let mut sum = bias[o];
        for i in 0..in_features {
            let w_idx = o * in_features + i;
            sum = sum + weights[w_idx] * input[i];
        }
        output[o] = sum;
    }

    output
}

/// CanaryCNN Model
pub struct CanaryCNN {
    // Conv1: 1 → 16 channels, 3x3 kernel
    pub conv1_weights: Vec<Fixed>, // [16, 1, 3, 3] = 144
    pub conv1_bias: Vec<Fixed>,    // [16]

    // Conv2: 16 → 32 channels, 3x3 kernel
    pub conv2_weights: Vec<Fixed>, // [32, 16, 3, 3] = 4608
    pub conv2_bias: Vec<Fixed>,    // [32]

    // FC1: 800 → 128
    pub fc1_weights: Vec<Fixed>, // [128, 800] = 102400
    pub fc1_bias: Vec<Fixed>,    // [128]

    // FC2: 128 → 10
    pub fc2_weights: Vec<Fixed>, // [10, 128] = 1280
    pub fc2_bias: Vec<Fixed>,    // [10]
}

impl CanaryCNN {
    /// Create a new CanaryCNN with zero-initialized weights (for testing)
    pub fn new_zeros() -> Self {
        Self {
            conv1_weights: vec![Fixed::ZERO; 144],
            conv1_bias: vec![Fixed::ZERO; 16],
            conv2_weights: vec![Fixed::ZERO; 4608],
            conv2_bias: vec![Fixed::ZERO; 32],
            fc1_weights: vec![Fixed::ZERO; 102400],
            fc1_bias: vec![Fixed::ZERO; 128],
            fc2_weights: vec![Fixed::ZERO; 1280],
            fc2_bias: vec![Fixed::ZERO; 10],
        }
    }

    /// Instantiate CanaryCNN using the baked-in pretrained weights.
    pub fn pretrained() -> Self {
        Self {
            conv1_weights: weights::CONV1_WEIGHTS.to_vec(),
            conv1_bias: weights::CONV1_BIAS.to_vec(),
            conv2_weights: weights::CONV2_WEIGHTS.to_vec(),
            conv2_bias: weights::CONV2_BIAS.to_vec(),
            fc1_weights: weights::FC1_WEIGHTS.to_vec(),
            fc1_bias: weights::FC1_BIAS.to_vec(),
            fc2_weights: weights::FC2_WEIGHTS.to_vec(),
            fc2_bias: weights::FC2_BIAS.to_vec(),
        }
    }

    /// Forward pass through the network
    ///
    /// # Arguments
    /// * `input` - MNIST image as flat array [784] (28x28)
    ///
    /// # Returns
    /// Logits [10] for digit classes 0-9
    pub fn forward(&self, input: &[Fixed; IMG_SIZE]) -> [Fixed; NUM_CLASSES] {
        // Reshape input to [1, 28, 28]
        let x = input.to_vec();

        // Conv1: [1, 28, 28] → [16, 26, 26]
        let x = conv2d_3x3(&x, &self.conv1_weights, &self.conv1_bias, 1, 16, 28, 28);
        let x = relu(&x);

        // MaxPool1: [16, 26, 26] → [16, 13, 13]
        let x = maxpool2d_2x2(&x, 16, 26, 26);

        // Conv2: [16, 13, 13] → [32, 11, 11]
        let x = conv2d_3x3(&x, &self.conv2_weights, &self.conv2_bias, 16, 32, 13, 13);
        let x = relu(&x);

        // MaxPool2: [32, 11, 11] → [32, 5, 5] = 800
        let x = maxpool2d_2x2(&x, 32, 11, 11);

        // Flatten: [32, 5, 5] → [800]
        // (already flat from maxpool2d)

        // FC1: [800] → [128]
        let x = fc_layer(&x, &self.fc1_weights, &self.fc1_bias, 800, 128);
        let x = relu(&x);

        // FC2: [128] → [10]
        let logits = fc_layer(&x, &self.fc2_weights, &self.fc2_bias, 128, 10);

        // Convert Vec to array
        let mut output = [Fixed::ZERO; NUM_CLASSES];
        output.copy_from_slice(&logits[0..NUM_CLASSES]);
        output
    }
}

impl Default for CanaryCNN {
    fn default() -> Self {
        Self::pretrained()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{calibration, loss::mse_single};

    const LOGIT_EPS: f32 = 0.05;
    const LOSS_EPS: f32 = 0.02;

    #[test]
    fn test_conv2d_3x3_shape() {
        // Simple 1x5x5 input with 1x3x3 kernel
        let input = vec![Fixed::ONE; 1 * 5 * 5];
        let weights = vec![Fixed::ONE; 1 * 1 * 3 * 3]; // 9 weights
        let bias = vec![Fixed::ZERO; 1];

        let output = conv2d_3x3(&input, &weights, &bias, 1, 1, 5, 5);

        // Output should be 1x3x3 = 9 elements
        assert_eq!(output.len(), 9);

        // Each output should be sum of 9 ones = 9.0
        let expected = Fixed::from_f32(9.0);
        for &val in &output {
            assert_eq!(val, expected);
        }
    }

    #[test]
    fn test_relu_basic() {
        let input = vec![
            Fixed::from_f32(-1.0),
            Fixed::ZERO,
            Fixed::from_f32(1.0),
            Fixed::from_f32(-5.0),
            Fixed::from_f32(5.0),
        ];

        let output = relu(&input);

        assert_eq!(output[0], Fixed::ZERO); // -1 → 0
        assert_eq!(output[1], Fixed::ZERO); // 0 → 0
        assert_eq!(output[2], Fixed::from_f32(1.0)); // 1 → 1
        assert_eq!(output[3], Fixed::ZERO); // -5 → 0
        assert_eq!(output[4], Fixed::from_f32(5.0)); // 5 → 5
    }

    #[test]
    fn test_maxpool2d_2x2_shape() {
        // 1x4x4 input
        let input = vec![Fixed::from_f32(1.0); 1 * 4 * 4];

        let output = maxpool2d_2x2(&input, 1, 4, 4);

        // Output should be 1x2x2 = 4 elements
        assert_eq!(output.len(), 4);
    }

    #[test]
    fn test_maxpool2d_2x2_values() {
        // 1x4x4 input with distinct values
        let input = vec![
            Fixed::from_f32(1.0),
            Fixed::from_f32(2.0),
            Fixed::from_f32(5.0),
            Fixed::from_f32(6.0),
            Fixed::from_f32(3.0),
            Fixed::from_f32(4.0),
            Fixed::from_f32(7.0),
            Fixed::from_f32(8.0),
            Fixed::from_f32(9.0),
            Fixed::from_f32(10.0),
            Fixed::from_f32(13.0),
            Fixed::from_f32(14.0),
            Fixed::from_f32(11.0),
            Fixed::from_f32(12.0),
            Fixed::from_f32(15.0),
            Fixed::from_f32(16.0),
        ];

        let output = maxpool2d_2x2(&input, 1, 4, 4);

        // Output should be max of each 2x2 window
        let expected = vec![
            Fixed::from_f32(4.0),  // max(1,2,3,4)
            Fixed::from_f32(8.0),  // max(5,6,7,8)
            Fixed::from_f32(12.0), // max(9,10,11,12)
            Fixed::from_f32(16.0), // max(13,14,15,16)
        ];

        assert_eq!(output, expected);
    }

    #[test]
    fn test_fc_layer_shape() {
        let input = vec![Fixed::from_f32(1.0); 4];
        let weights = vec![Fixed::from_f32(0.5); 3 * 4]; // 3x4 matrix
        let bias = vec![Fixed::ZERO; 3];

        let output = fc_layer(&input, &weights, &bias, 4, 3);

        assert_eq!(output.len(), 3);
    }

    #[test]
    fn test_fc_layer_computation() {
        let input = vec![Fixed::from_f32(1.0), Fixed::from_f32(2.0)];
        let weights = vec![
            // Row 0: [1.0, 2.0]
            Fixed::from_f32(1.0),
            Fixed::from_f32(2.0),
            // Row 1: [3.0, 4.0]
            Fixed::from_f32(3.0),
            Fixed::from_f32(4.0),
        ];
        let bias = vec![Fixed::from_f32(0.1), Fixed::from_f32(0.2)];

        let output = fc_layer(&input, &weights, &bias, 2, 2);

        // output[0] = 1*1 + 2*2 + 0.1 = 5.1
        // output[1] = 1*3 + 2*4 + 0.2 = 11.2
        assert!((output[0].to_f32() - 5.1).abs() < 0.01);
        assert!((output[1].to_f32() - 11.2).abs() < 0.01);
    }

    #[test]
    fn test_canarycnn_forward_shape() {
        let model = CanaryCNN::new_zeros();
        let input = [Fixed::ZERO; IMG_SIZE];

        let output = model.forward(&input);

        // Output should be 10 logits
        assert_eq!(output.len(), NUM_CLASSES);
    }

    #[test]
    fn test_canarycnn_forward_zeros() {
        let model = CanaryCNN::new_zeros();
        let input = [Fixed::ZERO; IMG_SIZE];

        let output = model.forward(&input);

        // With zero weights and biases, output should be all zeros
        for &val in &output {
            assert_eq!(val, Fixed::ZERO);
        }
    }

    #[test]
    fn test_pretrained_forward_matches_calibration() {
        let model = CanaryCNN::pretrained();

        for idx in 0..calibration::CALIBRATION_SAMPLE_COUNT {
            let input = calibration::CALIBRATION_INPUTS[idx];
            let expected = calibration::CALIBRATION_OUTPUTS[idx];
            let label = calibration::CALIBRATION_LABELS[idx];

            let logits = model.forward(&input);

            for (j, &logit) in logits.iter().enumerate() {
                let diff = (logit.to_f32() - expected[j]).abs();
                assert!(
                    diff <= LOGIT_EPS,
                    "sample {idx} logit {j}: diff {diff} exceeded {LOGIT_EPS}"
                );
            }

            let loss_fixed = mse_single(&logits, label).to_f32();
            let mut total = 0.0f32;
            for (class_idx, &logit) in expected.iter().enumerate() {
                let target = if class_idx == label { 1.0 } else { 0.0 };
                let error = logit - target;
                total += error * error;
            }
            let expected_loss = total / expected.len() as f32;
            let loss_diff = (loss_fixed - expected_loss).abs();
            assert!(
                loss_diff <= LOSS_EPS,
                "sample {idx} loss diff {loss_diff} exceeded {LOSS_EPS}"
            );
        }
    }
}
