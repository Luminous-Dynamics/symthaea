//! # Bridges Module: Cross-Representation Translation
//!
//! This module provides bridges between different neural/cognitive representations
//! in Symthaea, enabling bidirectional information flow between:
//!
//! - **HDC semantic vectors** (high-dimensional, symbolic meaning)
//! - **CfC temporal states** (continuous-time dynamics, sequential patterns)
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────────┐
//! │                        HDC-CfC BRIDGE                                       │
//! │                                                                             │
//! │    HDC Domain                              CfC Domain                       │
//! │   (Semantic)                              (Temporal)                        │
//! │                                                                             │
//! │  ┌─────────────┐    encode_semantic_to_temporal    ┌─────────────┐          │
//! │  │ ContinuousHV│ ──────────────────────────────▶   │  CfC State  │          │
//! │  │  (16,384D)  │                                   │   (128D)    │          │
//! │  │             │    decode_temporal_to_semantic    │             │          │
//! │  │  concepts   │ ◀──────────────────────────────── │  dynamics   │          │
//! │  └─────────────┘                                   └─────────────┘          │
//! │                                                                             │
//! │  Projection Matrices:                     Attention Mechanism:              │
//! │  - W_encode: [hidden × hdc_dim]          - Query from HDC                   │
//! │  - W_decode: [hdc_dim × hidden]          - Key/Value from CfC               │
//! │  - Both learnable via Adam               - Selective bridging               │
//! │                                                                             │
//! └─────────────────────────────────────────────────────────────────────────────┘
//! ```

pub mod hdc_cfc_bridge;

pub use hdc_cfc_bridge::{
    HdcCfcBridge, HdcCfcBridgeConfig, BridgeAttention, AttentionOutput,
};
