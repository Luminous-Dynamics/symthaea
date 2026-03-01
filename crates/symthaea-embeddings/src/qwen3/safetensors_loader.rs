//! Safetensors → Burn Weight Loader
//!
//! Loads HuggingFace safetensors files directly into Burn model parameters
//! without `burn-import` (removed due to liblzma-sys link conflict with lancedb).
//!
//! Approach: initialise model from config (random weights + correct RotaryEncoding),
//! then use [`Module::map`] to replace every learnable `Param<Tensor>` with the
//! corresponding safetensors tensor, in declaration-order traversal.

use anyhow::{Context, Result};
use burn::module::{ModuleMapper, ParamId};
use burn::prelude::*;

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
    fn map_float<const D: usize>(
        &mut self,
        _id: ParamId,
        tensor: Tensor<B, D>,
    ) -> Tensor<B, D> {
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

// ── Public API ──────────────────────────────────────────────────────

/// Load safetensors weights into an already-initialised Qwen3 model.
///
/// The model must have been created via [`Qwen3ModelConfig::init`] so that
/// non-learnable state (RotaryEncoding frequencies, RmsNorm epsilon) is correct.
/// This function replaces **all** learnable `Param<Tensor>` fields with the
/// corresponding safetensors tensors.
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

    // Build ordered list of (tensor_name, needs_transpose) matching Burn's
    // Module::map traversal order (struct-field declaration order, recursive).
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
fn build_param_names(num_layers: usize) -> Vec<(String, bool)> {
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
pub fn list_tensors(
    data: &[u8],
) -> Result<Vec<(String, Vec<usize>, safetensors::Dtype)>> {
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
#[allow(dead_code)]
pub fn expected_param_count(config: &Qwen3ModelConfig) -> usize {
    2 + config.num_hidden_layers * 9
}
