#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""Export BigVGAN v2 (24kHz, 100-band mel) to ONNX format.

Downloads the model from HuggingFace and exports with dynamic axes on the
n_frames dimension so the model accepts variable-length mel spectrograms.

Dependencies:
    pip install torch bigvgan onnx onnxruntime

Usage:
    cd symthaea/models && python export_bigvgan.py

Output:
    bigvgan_v2_24khz_100band.onnx (~112MB)
"""

import os
import torch
import numpy as np

OUTPUT_PATH = "bigvgan_v2_24khz_100band.onnx"
MODEL_NAME = "nvidia/bigvgan_v2_24khz_100band_256x"
N_MELS = 100
SAMPLE_FRAMES = 64  # frames for dummy input during export


def main():
    print(f"Loading BigVGAN model: {MODEL_NAME}")

    try:
        import bigvgan
        model = bigvgan.BigVGAN.from_pretrained(MODEL_NAME)
    except ImportError:
        # Fallback: clone and load directly
        print("bigvgan package not found, trying torch.hub...")
        model = torch.hub.load("NVIDIA/BigVGAN", "bigvgan_v2_24khz_100band_256x", trust_repo=True)

    model.eval()
    model.remove_weight_norm()

    # Dummy input: [batch=1, n_mels=100, n_frames=64]
    dummy_mel = torch.randn(1, N_MELS, SAMPLE_FRAMES)

    print(f"Exporting to ONNX: {OUTPUT_PATH}")
    torch.onnx.export(
        model,
        dummy_mel,
        OUTPUT_PATH,
        input_names=["mel_spectrogram"],
        output_names=["audio"],
        dynamic_axes={
            "mel_spectrogram": {2: "n_frames"},
            "audio": {2: "n_samples"},
        },
        opset_version=17,
        do_constant_folding=True,
    )

    # Verify exported model
    print("Verifying exported model...")
    import onnx
    import onnxruntime as ort

    onnx_model = onnx.load(OUTPUT_PATH)
    onnx.checker.check_model(onnx_model)

    session = ort.InferenceSession(OUTPUT_PATH)
    test_mel = np.random.randn(1, N_MELS, SAMPLE_FRAMES).astype(np.float32)
    outputs = session.run(None, {"mel_spectrogram": test_mel})
    audio = outputs[0]

    expected_samples = SAMPLE_FRAMES * 256  # hop_length = 256
    print(f"Input shape:  [1, {N_MELS}, {SAMPLE_FRAMES}]")
    print(f"Output shape: {audio.shape}")
    print(f"Expected ~{expected_samples} samples, got {audio.shape[-1]}")
    print(f"Model size: {os.path.getsize(OUTPUT_PATH) / 1024 / 1024:.1f} MB")
    print("Export successful!")


if __name__ == "__main__":
    main()
