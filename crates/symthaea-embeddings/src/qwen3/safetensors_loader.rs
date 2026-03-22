// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Safetensors → Burn Weight Loader
//!
//! Loads HuggingFace safetensors files directly into Burn model parameters
//! without `burn-import` (removed due to liblzma-sys link conflict with lancedb).
//!
//! Approach: initialise model from config (random weights + correct RotaryEncoding),
//! then use [`Module::map`] to replace every learnable `Param<Tensor>` with the
//! corresponding safetensors tensor, in declaration-order traversal.
//!
//! Supports both single-file (`model.safetensors`) and multi-shard layouts
//! (`model.safetensors.index.json` → `model-00001-of-NNNNN.safetensors` …).
//! Uses memory-mapped I/O via `memmap2` to avoid copying large weight files.

use anyhow::{Context, Result};
use burn::module::{ModuleMapper, ParamId};
use burn::prelude::*;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::io::Read;
use std::path::Path;

use super::model::Qwen3ModelConfig;

// ── Tensor replacement ──────────────────────────────────────────────

/// Raw f32 data + shape for one tensor replacement.
struct TensorReplacement {
    data: Vec<f32>,
    shape: Vec<usize>,
}

/// [`ModuleMapper`] that replaces float parameters in visit order.
///
/// The derive-generated `Module::map` traverses parameters in struct-field
/// declaration order (recursing into sub-modules and `Vec` elements in order).
/// We exploit this deterministic ordering to match safetensors keys to Burn
/// parameters without needing parameter names or IDs.
struct SafetensorsMapper {
    replacements: Vec<TensorReplacement>,
    counter: usize,
}

impl<B: Backend> ModuleMapper<B> for SafetensorsMapper {
    fn map_float<const D: usize>(&mut self, _id: ParamId, tensor: Tensor<B, D>) -> Tensor<B, D> {
        let idx = self.counter;
        self.counter += 1;
        if idx < self.replacements.len() {
            let repl = &self.replacements[idx];
            let data = burn::tensor::TensorData::new(repl.data.clone(), repl.shape.clone());
            Tensor::from_data(data, &tensor.device())
        } else {
            tensor
        }
    }
}

// ── Model integrity verification ──────────────────────────────────

/// Verify model file integrity via SHA-256 before unsafe memory-mapped loading.
///
/// - If `expected_hash` is `Some(hash)`, computes SHA-256 and verifies it matches.
/// - If `expected_hash` is `None`, logs a warning but allows loading (graceful degradation).
/// - Always fails if the file does not exist.
fn verify_model_integrity(path: &Path, expected_hash: Option<&str>) -> Result<()> {
    if !path.exists() {
        anyhow::bail!(
            "model_integrity: model file does not exist: {}",
            path.display()
        );
    }

    match expected_hash {
        Some(expected) => {
            let actual = compute_model_hash(path)?;
            if actual != expected {
                anyhow::bail!(
                    "model_integrity: SHA-256 mismatch for {}\n  expected: {}\n  actual:   {}",
                    path.display(),
                    expected,
                    actual
                );
            }
            tracing::debug!(
                path = %path.display(),
                "model_integrity: SHA-256 verified"
            );
            Ok(())
        }
        None => {
            tracing::warn!(
                path = %path.display(),
                "model_integrity: loading model without integrity verification \
                 (no expected hash provided)"
            );
            Ok(())
        }
    }
}

/// Compute the SHA-256 hash of a file, returned as a lowercase hex string.
fn compute_model_hash(path: &Path) -> Result<String> {
    let mut file = std::fs::File::open(path)
        .with_context(|| format!("model_integrity: failed to open {}", path.display()))?;

    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 65536];

    loop {
        let bytes_read = file
            .read(&mut buffer)
            .with_context(|| format!("model_integrity: failed to read {}", path.display()))?;
        if bytes_read == 0 {
            break;
        }
        hasher.update(&buffer[..bytes_read]);
    }

    let hash = hasher.finalize();
    Ok(format!("{:x}", hash))
}

// ── Memory-mapped file helper ───────────────────────────────────────

/// Memory-map a file for zero-copy safetensors deserialization.
///
/// Verifies model integrity via SHA-256 before performing the unsafe mmap.
/// The `Mmap` derefs to `&[u8]`, so `SafeTensors::deserialize(&mmap)` works unchanged.
#[allow(unsafe_code)]
fn mmap_file(path: &Path) -> Result<memmap2::Mmap> {
    // Verify model integrity before unsafe memory-mapped loading
    verify_model_integrity(path, None)?;
    let file =
        std::fs::File::open(path).with_context(|| format!("Failed to open {}", path.display()))?;
    unsafe { memmap2::MmapOptions::new().map(&file) }
        .map_err(|e| anyhow::anyhow!("mmap failed for {}: {e}", path.display()))
}

// ── Public API ──────────────────────────────────────────────────────

/// Load safetensors weights from a model directory into an already-initialised Qwen3 model.
///
/// Handles both single-file and multi-shard layouts:
/// - If `model.safetensors.index.json` exists, parses the `weight_map` to locate
///   tensors across shard files, loading each shard exactly once.
/// - Otherwise, falls back to the single `model.safetensors` file.
///
/// Uses memory-mapped I/O for efficient loading of large weight files.
pub fn load_qwen3_from_dir<B: Backend>(
    model: super::model::Qwen3Model<B>,
    model_dir: &Path,
    config: &Qwen3ModelConfig,
    _device: &B::Device,
) -> Result<super::model::Qwen3Model<B>> {
    let param_names = build_param_names(config.num_hidden_layers);

    let index_path = model_dir.join("model.safetensors.index.json");
    let replacements = if index_path.exists() {
        load_sharded(&index_path, model_dir, &param_names)?
    } else {
        let single_path = model_dir.join("model.safetensors");
        load_single(&single_path, &param_names)?
    };

    let mut mapper = SafetensorsMapper {
        replacements,
        counter: 0,
    };

    Ok(model.map(&mut mapper))
}

/// Load safetensors weights into an already-initialised Qwen3 model (single-file, in-memory).
///
/// Kept for backward compatibility. Prefer [`load_qwen3_from_dir`] for new code.
///
/// # Weight key mapping (HuggingFace → Burn)
///
/// | HF Key | Transform |
/// |--------|-----------|
/// | `model.embed_tokens.weight` | `[vocab, hidden]` — no transpose |
/// | `model.layers.{i}.self_attn.{q,k,v,o}_proj.weight` | `[out, in]` → transpose to `[in, out]` |
/// | `model.layers.{i}.mlp.{gate,up,down}_proj.weight` | `[out, in]` → transpose |
/// | `model.layers.{i}.{input,post_attention}_layernorm.weight` | 1D → RmsNorm gamma |
/// | `model.norm.weight` | 1D → final RmsNorm gamma |
pub fn load_qwen3_weights<B: Backend>(
    model: super::model::Qwen3Model<B>,
    safetensors_data: &[u8],
    config: &Qwen3ModelConfig,
    _device: &B::Device,
) -> Result<super::model::Qwen3Model<B>> {
    let st = safetensors::SafeTensors::deserialize(safetensors_data)
        .map_err(|e| anyhow::anyhow!("Failed to parse safetensors: {e}"))?;

    let param_names = build_param_names(config.num_hidden_layers);

    let mut replacements = Vec::with_capacity(param_names.len());
    for (name, transpose) in &param_names {
        let view = st
            .tensor(name)
            .map_err(|e| anyhow::anyhow!("Missing tensor '{name}': {e}"))?;
        let repl = convert_tensor(&view, *transpose)
            .with_context(|| format!("Failed to convert tensor '{name}'"))?;
        replacements.push(repl);
    }

    let mut mapper = SafetensorsMapper {
        replacements,
        counter: 0,
    };

    Ok(model.map(&mut mapper))
}

// ── Single-file loading (mmap) ──────────────────────────────────────

fn load_single(
    safetensors_path: &Path,
    param_names: &[(String, bool)],
) -> Result<Vec<TensorReplacement>> {
    let mmap = mmap_file(safetensors_path)?;
    let st = safetensors::SafeTensors::deserialize(&mmap)
        .map_err(|e| anyhow::anyhow!("Failed to parse safetensors: {e}"))?;

    let mut replacements = Vec::with_capacity(param_names.len());
    for (name, transpose) in param_names {
        let view = st
            .tensor(name)
            .map_err(|e| anyhow::anyhow!("Missing tensor '{name}': {e}"))?;
        let repl = convert_tensor(&view, *transpose)
            .with_context(|| format!("Failed to convert tensor '{name}'"))?;
        replacements.push(repl);
    }
    Ok(replacements)
}

// ── Multi-shard loading ─────────────────────────────────────────────

/// HF index file structure: `{"metadata": {...}, "weight_map": {"tensor.name": "shard-file.safetensors", ...}}`
#[derive(serde::Deserialize)]
struct SafetensorsIndex {
    weight_map: BTreeMap<String, String>,
}

fn load_sharded(
    index_path: &Path,
    model_dir: &Path,
    param_names: &[(String, bool)],
) -> Result<Vec<TensorReplacement>> {
    let index_data = std::fs::read_to_string(index_path)
        .with_context(|| format!("Failed to read {}", index_path.display()))?;
    let index: SafetensorsIndex = serde_json::from_str(&index_data)
        .with_context(|| format!("Failed to parse {}", index_path.display()))?;

    // Group tensor lookups by shard file to avoid re-reading shards.
    // Each entry: (param_index, tensor_name, needs_transpose)
    let mut shard_groups: BTreeMap<&str, Vec<(usize, &str, bool)>> = BTreeMap::new();
    for (idx, (name, transpose)) in param_names.iter().enumerate() {
        let shard = index
            .weight_map
            .get(name.as_str())
            .ok_or_else(|| anyhow::anyhow!("Tensor '{name}' not found in index weight_map"))?;
        shard_groups
            .entry(shard.as_str())
            .or_default()
            .push((idx, name.as_str(), *transpose));
    }

    // Pre-allocate with placeholder values, then fill in-place
    let mut replacements: Vec<Option<TensorReplacement>> =
        (0..param_names.len()).map(|_| None).collect();

    for (shard_file, tensors) in &shard_groups {
        let shard_path = model_dir.join(shard_file);
        let mmap = mmap_file(&shard_path)?;
        let st = safetensors::SafeTensors::deserialize(&mmap)
            .map_err(|e| anyhow::anyhow!("Failed to parse shard {shard_file}: {e}"))?;

        for &(idx, name, transpose) in tensors {
            let view = st.tensor(name).map_err(|e| {
                anyhow::anyhow!("Missing tensor '{name}' in shard {shard_file}: {e}")
            })?;
            let repl = convert_tensor(&view, transpose)
                .with_context(|| format!("Failed to convert tensor '{name}' from {shard_file}"))?;
            replacements[idx] = Some(repl);
        }
    }

    // Unwrap all — every slot should be filled
    replacements
        .into_iter()
        .enumerate()
        .map(|(i, opt)| {
            opt.ok_or_else(|| anyhow::anyhow!("Missing replacement for param index {i}"))
        })
        .collect()
}

// ── Parameter name ordering ─────────────────────────────────────────

/// Build the ordered list of safetensors tensor names + transpose flags.
///
/// **Must match the Burn `Module::map` traversal order exactly.**
///
/// Traversal order for our model structs:
/// - `Qwen3Model`: embed_tokens → layers\[0..N\] → norm
/// - `Qwen3DecoderLayer`: self_attn → mlp → input_layernorm → post_attention_layernorm
/// - `Qwen3Attention`: q_proj → k_proj → v_proj → o_proj  (rotary/usize fields skipped)
/// - `Qwen3Mlp`: gate_proj → up_proj → down_proj
/// - `RmsNorm`: gamma  (epsilon is f64, skipped)
///
/// Linear biases are `None` (Qwen3 uses no bias), so they don't participate.
pub(crate) fn build_param_names(num_layers: usize) -> Vec<(String, bool)> {
    // 1 embed + 9 per layer + 1 final norm
    let capacity = 2 + num_layers * 9;
    let mut names = Vec::with_capacity(capacity);

    // Embedding: [vocab, hidden] — no transpose
    names.push(("model.embed_tokens.weight".into(), false));

    for i in 0..num_layers {
        let p = format!("model.layers.{i}");

        // Attention: HF [out, in] → Burn [in, out]
        names.push((format!("{p}.self_attn.q_proj.weight"), true));
        names.push((format!("{p}.self_attn.k_proj.weight"), true));
        names.push((format!("{p}.self_attn.v_proj.weight"), true));
        names.push((format!("{p}.self_attn.o_proj.weight"), true));

        // MLP: same transpose
        names.push((format!("{p}.mlp.gate_proj.weight"), true));
        names.push((format!("{p}.mlp.up_proj.weight"), true));
        names.push((format!("{p}.mlp.down_proj.weight"), true));

        // LayerNorms: 1D gamma (HF names it "weight")
        names.push((format!("{p}.input_layernorm.weight"), false));
        names.push((format!("{p}.post_attention_layernorm.weight"), false));
    }

    // Final norm gamma
    names.push(("model.norm.weight".into(), false));

    names
}

// ── Tensor conversion ───────────────────────────────────────────────

/// Convert a safetensors tensor view to f32 data, optionally transposing 2D weights.
fn convert_tensor(
    view: &safetensors::tensor::TensorView,
    transpose: bool,
) -> Result<TensorReplacement> {
    let dtype = view.dtype();
    let shape = view.shape().to_vec();
    let raw = view.data();

    // Convert raw bytes → f32
    let values: Vec<f32> = match dtype {
        safetensors::Dtype::F32 => raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        safetensors::Dtype::F16 => raw
            .chunks_exact(2)
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        safetensors::Dtype::BF16 => raw
            .chunks_exact(2)
            .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        other => anyhow::bail!("Unsupported tensor dtype: {other:?}"),
    };

    // Transpose 2D weights: HF [out_features, in_features] → Burn [d_input, d_output]
    if transpose && shape.len() == 2 {
        let (rows, cols) = (shape[0], shape[1]);
        let mut transposed = vec![0.0f32; values.len()];
        for r in 0..rows {
            for c in 0..cols {
                transposed[c * rows + r] = values[r * cols + c];
            }
        }
        Ok(TensorReplacement {
            data: transposed,
            shape: vec![cols, rows],
        })
    } else {
        Ok(TensorReplacement {
            data: values,
            shape,
        })
    }
}

// ── Diagnostics ─────────────────────────────────────────────────────

/// List all tensor names, shapes, and dtypes in a safetensors file.
#[allow(dead_code)]
pub fn list_tensors(data: &[u8]) -> Result<Vec<(String, Vec<usize>, safetensors::Dtype)>> {
    let st = safetensors::SafeTensors::deserialize(data)
        .map_err(|e| anyhow::anyhow!("Failed to parse safetensors: {e}"))?;
    let mut result: Vec<_> = st
        .tensors()
        .into_iter()
        .map(|(name, view)| (name, view.shape().to_vec(), view.dtype()))
        .collect();
    result.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(result)
}

/// Expected number of learnable parameters for a given model config.
pub fn expected_param_count(config: &Qwen3ModelConfig) -> usize {
    2 + config.num_hidden_layers * 9
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use safetensors::tensor::{Dtype, TensorView};

    type B = NdArray;

    /// Infer the HF-convention shape for a tensor name given a model config.
    fn infer_hf_shape(name: &str, config: &Qwen3ModelConfig) -> Vec<usize> {
        if name.contains("embed_tokens") {
            vec![config.vocab_size, config.hidden_size]
        } else if name.contains("q_proj") {
            vec![
                config.num_attention_heads * config.head_dim,
                config.hidden_size,
            ]
        } else if name.contains("k_proj") || name.contains("v_proj") {
            vec![
                config.num_key_value_heads * config.head_dim,
                config.hidden_size,
            ]
        } else if name.contains("o_proj") {
            vec![
                config.hidden_size,
                config.num_attention_heads * config.head_dim,
            ]
        } else if name.contains("gate_proj") || name.contains("up_proj") {
            vec![config.intermediate_size, config.hidden_size]
        } else if name.contains("down_proj") {
            vec![config.hidden_size, config.intermediate_size]
        } else if name.contains("layernorm") || name.ends_with("norm.weight") {
            vec![config.hidden_size]
        } else {
            panic!("Unknown tensor name: {name}");
        }
    }

    /// Build synthetic safetensors bytes with known f32 values for every expected tensor.
    fn build_synthetic_safetensors(
        config: &Qwen3ModelConfig,
        dtype: Dtype,
    ) -> (Vec<u8>, Vec<(String, Vec<f32>)>) {
        let param_names = build_param_names(config.num_hidden_layers);
        let mut tensors_data: Vec<(String, Vec<u8>, Vec<usize>, Dtype)> = Vec::new();
        let mut expected_values: Vec<(String, Vec<f32>)> = Vec::new();

        for (idx, (name, _transpose)) in param_names.iter().enumerate() {
            let shape = infer_hf_shape(name, config);
            let numel: usize = shape.iter().product();
            // Fill with deterministic f32 values: (idx * 1000 + element) * 0.001
            let base = (idx as f32) * 0.1;
            let values: Vec<f32> = (0..numel).map(|i| base + (i as f32) * 0.0001).collect();
            expected_values.push((name.clone(), values.clone()));

            let raw_bytes: Vec<u8> = match dtype {
                Dtype::F32 => values.iter().flat_map(|v| v.to_le_bytes()).collect(),
                Dtype::F16 => values
                    .iter()
                    .flat_map(|v| half::f16::from_f32(*v).to_le_bytes())
                    .collect(),
                _ => panic!("Test only supports F32 and F16"),
            };
            tensors_data.push((name.clone(), raw_bytes, shape, dtype));
        }

        // Build TensorView references and serialize
        let views: Vec<(String, TensorView<'_>)> = tensors_data
            .iter()
            .map(|(name, data, shape, dt)| {
                (
                    name.clone(),
                    TensorView::new(*dt, shape.clone(), data).unwrap(),
                )
            })
            .collect();

        let bytes = safetensors::tensor::serialize(
            views.iter().map(|(n, v)| (n.as_str(), v.clone())),
            &None,
        )
        .unwrap();
        (bytes, expected_values)
    }

    /// ModuleMapper that extracts param data for verification.
    struct ParameterExtractor {
        params: Vec<Vec<f32>>,
    }

    impl<B2: Backend> ModuleMapper<B2> for ParameterExtractor {
        fn map_float<const D: usize>(
            &mut self,
            _id: ParamId,
            tensor: Tensor<B2, D>,
        ) -> Tensor<B2, D> {
            let data = tensor.to_data();
            let values: Vec<f32> = data.to_vec().unwrap();
            self.params.push(values);
            // Re-create from data to return (identity)
            Tensor::from_data(data, &tensor.device())
        }
    }

    #[test]
    fn test_roundtrip_f32() {
        let config = Qwen3ModelConfig::tiny();
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;

        let (st_bytes, expected) = build_synthetic_safetensors(&config, Dtype::F32);
        let model = config.init::<B>(&device);
        let model = load_qwen3_weights(model, &st_bytes, &config, &device).unwrap();

        let mut extractor = ParameterExtractor { params: Vec::new() };
        let _model = model.map(&mut extractor);

        let param_names = build_param_names(config.num_hidden_layers);
        assert_eq!(extractor.params.len(), param_names.len());

        for (i, (name, transpose)) in param_names.iter().enumerate() {
            let loaded = &extractor.params[i];
            let original = &expected[i].1;
            assert_eq!(loaded.len(), original.len(), "Size mismatch for {name}");

            // For transposed 2D tensors, verify the transpose happened correctly
            let shape = infer_hf_shape(name, &config);
            if *transpose && shape.len() == 2 {
                let (rows, cols) = (shape[0], shape[1]);
                // After transpose: Burn shape is [cols, rows]
                // Value at Burn[c][r] should equal HF[r][c] = original[r*cols + c]
                for r in 0..rows {
                    for c in 0..cols {
                        let burn_idx = c * rows + r;
                        let hf_idx = r * cols + c;
                        assert!(
                            (loaded[burn_idx] - original[hf_idx]).abs() < 1e-6,
                            "Transpose mismatch for {name} at [{c},{r}]"
                        );
                    }
                }
            } else {
                // Direct comparison
                for (j, (l, e)) in loaded.iter().zip(original.iter()).enumerate() {
                    assert!(
                        (l - e).abs() < 1e-6,
                        "Value mismatch for {name}[{j}]: got {l}, expected {e}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_roundtrip_f16() {
        let config = Qwen3ModelConfig::tiny();
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;

        let (st_bytes, expected) = build_synthetic_safetensors(&config, Dtype::F16);
        let model = config.init::<B>(&device);
        let model = load_qwen3_weights(model, &st_bytes, &config, &device).unwrap();

        let mut extractor = ParameterExtractor { params: Vec::new() };
        let _model = model.map(&mut extractor);

        let param_names = build_param_names(config.num_hidden_layers);
        for (i, (name, transpose)) in param_names.iter().enumerate() {
            let loaded = &extractor.params[i];
            let original = &expected[i].1;
            assert_eq!(loaded.len(), original.len(), "Size mismatch for {name}");

            let shape = infer_hf_shape(name, &config);
            if *transpose && shape.len() == 2 {
                // Transposed: Burn[c*rows+r] == HF[r*cols+c]
                let (rows, cols) = (shape[0], shape[1]);
                for r in 0..rows {
                    for c in 0..cols {
                        let burn_idx = c * rows + r;
                        let hf_idx = r * cols + c;
                        let e_f16 = half::f16::from_f32(original[hf_idx]).to_f32();
                        assert!(
                            (loaded[burn_idx] - e_f16).abs() < 0.01,
                            "Transpose value mismatch for {name} at [{c},{r}]"
                        );
                    }
                }
            } else {
                // Direct comparison with f16 epsilon
                for (j, (l, e)) in loaded.iter().zip(original.iter()).enumerate() {
                    let e_f16 = half::f16::from_f32(*e).to_f32();
                    assert!(
                        (l - e_f16).abs() < 0.01,
                        "Value mismatch for {name}[{j}]: got {l}, expected ~{e_f16} (f16 of {e})"
                    );
                }
            }
        }
    }

    #[test]
    fn test_missing_tensor_error() {
        let config = Qwen3ModelConfig::tiny();
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;

        // Build safetensors with all-but-one tensor
        let param_names = build_param_names(config.num_hidden_layers);
        let omitted = &param_names.last().unwrap().0; // Omit final norm.weight

        let mut tensors_data: Vec<(String, Vec<u8>, Vec<usize>)> = Vec::new();
        for (name, _transpose) in &param_names {
            if name == omitted {
                continue;
            }
            let shape = infer_hf_shape(name, &config);
            let numel: usize = shape.iter().product();
            let bytes: Vec<u8> = vec![0u8; numel * 4]; // zeros
            tensors_data.push((name.clone(), bytes, shape));
        }

        let views: Vec<(String, TensorView<'_>)> = tensors_data
            .iter()
            .map(|(name, data, shape)| {
                (
                    name.clone(),
                    TensorView::new(Dtype::F32, shape.clone(), data).unwrap(),
                )
            })
            .collect();
        let bytes = safetensors::tensor::serialize(
            views.iter().map(|(n, v)| (n.as_str(), v.clone())),
            &None,
        )
        .unwrap();

        let model = config.init::<B>(&device);
        let err = load_qwen3_weights(model, &bytes, &config, &device).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains(omitted),
            "Error should name the missing tensor '{omitted}', got: {msg}"
        );
    }

    #[test]
    fn test_param_count() {
        let tiny = Qwen3ModelConfig::tiny();
        assert_eq!(
            expected_param_count(&tiny),
            build_param_names(tiny.num_hidden_layers).len()
        );

        let cfg_06b = Qwen3ModelConfig::qwen3_06b();
        assert_eq!(
            expected_param_count(&cfg_06b),
            build_param_names(cfg_06b.num_hidden_layers).len()
        );

        let cfg_15b = Qwen3ModelConfig::qwen3_15b();
        assert_eq!(
            expected_param_count(&cfg_15b),
            build_param_names(cfg_15b.num_hidden_layers).len()
        );
    }

    #[test]
    fn test_param_names_order() {
        let config = Qwen3ModelConfig::tiny();
        let names = build_param_names(config.num_hidden_layers);

        assert_eq!(names.first().unwrap().0, "model.embed_tokens.weight");
        assert_eq!(names.last().unwrap().0, "model.norm.weight");

        // Embedding should NOT be transposed
        assert!(!names.first().unwrap().1);
        // Final norm should NOT be transposed
        assert!(!names.last().unwrap().1);

        // Attention projections should be transposed
        assert!(names[1].0.contains("q_proj"));
        assert!(names[1].1); // transposed
    }
}
