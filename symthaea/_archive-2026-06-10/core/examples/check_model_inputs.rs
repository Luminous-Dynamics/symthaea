// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Quick diagnostic to check ONNX model inputs

#[cfg(feature = "embeddings")]
fn main() -> anyhow::Result<()> {
    use ort::session::Session;
    use std::path::Path;

    let model_paths = [
        "models/all-MiniLM-L6-v2-onnx/model.onnx",
        "models/qwen3-embedding-0.6b-onnx/onnx/model.onnx",
        "models/qwen3-embedding-0.6b/model.onnx",
    ];

    for model_path in &model_paths {
        let path = Path::new(model_path);
        if path.exists() {
            println!("=== {} ===", model_path);
            let session = Session::builder()?.commit_from_file(path)?;

            println!("INPUTS:");
            for input in session.inputs.iter() {
                println!(
                    "  {} : {:?} (type: {:?})",
                    input.name, input.input_type, input.input_type
                );
            }

            println!("OUTPUTS:");
            for output in session.outputs.iter() {
                println!("  {} : {:?}", output.name, output.output_type);
            }
            println!();
        } else {
            println!("=== {} (NOT FOUND) ===\n", model_path);
        }
    }

    Ok(())
}

#[cfg(not(feature = "embeddings"))]
fn main() {
    println!("Requires --features embeddings");
}
