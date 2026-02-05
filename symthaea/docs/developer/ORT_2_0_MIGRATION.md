# ORT 2.0 Migration Guide

**Date**: 2026-01-05
**Status**: Complete
**Affected Files**: `src/embeddings/qwen3.rs`, `src/hdc/semantic_encoder.rs`

---

## Overview

This document describes the API changes required when migrating from ORT 1.x to ORT 2.0 (ort crate version 2.0.0-rc.10+).

---

## Breaking Changes

### 1. Session Import Path

**Before (ORT 1.x):**
```rust
use ort::Session;
// or
let session = ort::Session::builder()...
```

**After (ORT 2.0):**
```rust
use ort::session::Session;

let session = Session::builder()
    .with_optimization_level(GraphOptimizationLevel::Level3)?
    .commit_from_file(&model_path)?;
```

### 2. Tensor Creation

**Before (ORT 1.x):**
```rust
use ort::value::Value;

let tensor = Value::from_array(session.allocator(), &array)?;
```

**After (ORT 2.0):**
```rust
use ort::value::Tensor;

// Option 1: Using (shape, data) tuple (recommended for dynamic shapes)
let tensor = Tensor::from_array(([1usize, seq_len], input_ids))?;

// Option 2: Using ndarray (for f32/f64 only, not i64)
// Note: ndarray with i64 is NOT supported directly
let array = Array2::from_shape_vec((1, seq_len), data)?;
let tensor = Tensor::from_array(array)?;  // Only works for f32/f64
```

**Important**: The `ort::inputs!` macro with ndarray views does NOT work for `i64` arrays. Use the `(shape, Vec<i64>)` tuple format instead.

### 3. Running Inference

**Before (ORT 1.x):**
```rust
let outputs = session.run(ort::inputs![
    "input_ids" => array.view(),
])?;
```

**After (ORT 2.0):**
```rust
// Create Tensor values first
let input_tensor = Tensor::from_array(([1usize, seq_len], input_ids))?;

let outputs = session.run(ort::inputs![
    "input_ids" => input_tensor,
])?;
```

### 4. Session Mutability

**ORT 2.0** requires `&mut self` for `session.run()`. If implementing a trait that requires `&self`, use interior mutability:

```rust
use std::sync::Mutex;

pub struct MyEncoder {
    session: Mutex<Session>,  // Wrap in Mutex
    // ...
}

impl MyEncoder {
    fn encode(&self, text: &str) -> Result<Vec<f32>> {
        // Lock the session
        let mut session = self.session.lock()
            .map_err(|e| format!("Lock failed: {}", e))?;

        // Now use &mut session
        let outputs = session.run(ort::inputs![...])?;
        // ...
    }
}
```

### 5. Output Extraction

**Before (ORT 1.x):**
```rust
let output = outputs[0].extract_tensor::<f32>()?;
let data = output.view();
```

**After (ORT 2.0):**
```rust
// try_extract_tensor returns (&Shape, &[T]) tuple
let extracted = outputs[0].try_extract_tensor::<f32>()?;

// Access tuple elements
let shape: Vec<usize> = extracted.0.iter().map(|&d| d as usize).collect();
let data: Vec<f32> = extracted.1.to_vec();
```

---

## Complete Migration Example

### Before (ORT 1.x)
```rust
use ort::{Session, Value};
use ndarray::Array2;

pub struct Encoder {
    session: Session,
}

impl Encoder {
    fn encode(&self, input_ids: Vec<i64>) -> Result<Vec<f32>> {
        let seq_len = input_ids.len();
        let array = Array2::from_shape_vec((1, seq_len), input_ids)?;

        let outputs = self.session.run(ort::inputs![
            "input_ids" => array.view(),
        ])?;

        let output = outputs[0].extract_tensor::<f32>()?;
        Ok(output.view().iter().copied().collect())
    }
}
```

### After (ORT 2.0)
```rust
use ort::session::Session;
use ort::value::Tensor;
use std::sync::Mutex;

pub struct Encoder {
    session: Mutex<Session>,  // Mutex for interior mutability
}

impl Encoder {
    fn encode(&self, input_ids: Vec<i64>) -> Result<Vec<f32>> {
        let seq_len = input_ids.len();

        // Lock session for mutable access
        let mut session = self.session.lock()
            .map_err(|e| anyhow::anyhow!("Lock failed: {}", e))?;

        // Create tensor using (shape, data) tuple
        let input_tensor = Tensor::from_array(([1usize, seq_len], input_ids))?;

        let outputs = session.run(ort::inputs![
            "input_ids" => input_tensor,
        ])?;

        // Extract using tuple destructuring
        let extracted = outputs[0].try_extract_tensor::<f32>()?;
        Ok(extracted.1.to_vec())
    }
}
```

---

## Cargo.toml Configuration

```toml
[dependencies]
ort = { version = "2.0", features = ["ndarray"] }
ndarray = "0.15"

[features]
embeddings = ["ort", "tokenizers", "hf-hub"]
```

---

## Common Errors and Solutions

### Error: `OwnedTensorArrayData<_>` not implemented for `ArrayBase<OwnedRepr<i64>, ...>`

**Cause**: ndarray with `i64` element type is not supported for direct tensor creation.

**Solution**: Use `(shape, Vec<i64>)` tuple format:
```rust
// Instead of this:
let array = Array2::from_shape_vec((1, len), data)?;
let tensor = Tensor::from_array(array)?;  // Error!

// Use this:
let tensor = Tensor::from_array(([1usize, len], data))?;  // Works!
```

### Error: `no method named 'shape' found for tuple`

**Cause**: `try_extract_tensor` returns a tuple, not an array view.

**Solution**: Access tuple elements:
```rust
let extracted = output.try_extract_tensor::<f32>()?;
let shape = extracted.0;  // &Shape
let data = extracted.1;   // &[f32]
```

### Error: `cannot borrow as mutable`

**Cause**: `Session::run` requires `&mut self` in ORT 2.0.

**Solution**: Use `Mutex<Session>` for interior mutability (see Section 4 above).

---

## Files Modified

1. **`src/embeddings/qwen3.rs`**
   - Updated tensor creation to use `(shape, data)` tuple format
   - Fixed output extraction for tuple return type

2. **`src/hdc/semantic_encoder.rs`**
   - Added `use ort::session::Session;` import
   - Wrapped `Session` in `Mutex` for interior mutability
   - Updated tensor creation and output extraction

---

## Testing

```bash
# Verify compilation
cargo check --features embeddings

# Run embeddings tests
cargo test --features embeddings qwen3

# Run semantic encoder tests
cargo test --features embeddings semantic_encoder
```

---

## References

- [ORT 2.0 Rust Documentation](https://ort.pyke.io/)
- [ORT 2.0 Migration Guide](https://ort.pyke.io/migrating/v2)
- [ONNX Runtime GitHub](https://github.com/microsoft/onnxruntime)
