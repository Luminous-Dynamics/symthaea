//! # Embedding Integration Layer
//!
//! This module provides semantic embedding capabilities for Symthaea,
//! bridging external embedding models (BGE) with the HDC semantic substrate.
//!
//! ## Architecture
//!
//! ```text
//! Text Input
//!     │
//!     ▼
//! ┌──────────────┐
//! │ BGE Embedder │ ← 768D dense vectors (bge-base-en-v1.5, 109M params)
//! └──────┬───────┘
//!        │
//!        ▼
//! ┌──────────────┐
//! │ HDC Bridge   │ ← Random projection to binary
//! └──────┬───────┘
//!        │
//!        ▼
//! ┌──────────────┐
//! │   HV16       │ ← 2048-bit binary hypervector
//! └──────────────┘
//! ```
//!
//! ## Key Capabilities
//!
//! 1. **Semantic grounding**: Bridge external text to HDC space
//! 2. **Coherence verification**: Detect hallucinations via similarity
//! 3. **Multilingual support**: Cross-lingual semantic similarity
//! 4. **LLM verification**: Check LLM outputs against input semantics
//!
//! ## Why BGE over Gemma?
//!
//! | Factor | BGE (bge-base-en-v1.5) | Gemma |
//! |--------|------------------------|-------|
//! | MTEB Rank | #1 open-source (2025) | Lower |
//! | Parameters | 109M (edge-friendly) | 308M |
//! | Contrastive training | Yes (better similarity) | No |
//! | Rust/ONNX | tract-onnx | tract-onnx |

pub mod bge;
pub mod bridge;

pub use bge::BGEEmbedder;
pub use bridge::HdcBridge;
