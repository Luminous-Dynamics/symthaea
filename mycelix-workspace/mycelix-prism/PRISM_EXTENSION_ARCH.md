# Prism Extension Architecture (Proposal)

## Overview
To enable "Safe Extensions" without introducing JS/JIT, Prism will adopt a side-car WASM-based extension architecture. Extensions will operate as read-only or transformation-limited agents, communicating via the `prism-bridge`.

## Message Interface Extensions (`prism-bridge`)

We will add an `Extension` variant to the IPC message types:

```rust
// Proposed additions to prism-bridge/src/lib.rs

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RendererToSpore {
    // ... existing variants
    /// Message from a registered extension.
    ExtensionMessage {
        extension_id: String,
        payload: Vec<u8>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SporeToRenderer {
    // ... existing variants
    /// Command/Data payload for an extension.
    ExtensionCommand {
        extension_id: String,
        command: String,
        data: Vec<u8>,
    },
}
```

## Transport Abstraction (`prism-net`)

To support non-HTTP transports (e.g., `iroh://`, `ipfs://`), we will introduce a `Transport` trait:

```rust
// Proposed structure for prism-net/src/transport.rs

#[async_trait]
pub trait Transport: Send + Sync {
    async fn fetch(&self, url: &Url) -> Result<Vec<u8>, FetchError>;
}

pub struct MultiTransportClient {
    http: reqwest::Client,
    p2p: Option<Box<dyn Transport>>,
}
```

## Security Rationale
- **Isolation:** All extension logic runs in WASM sandboxes managed by the Spore kernel.
- **Controlled Communication:** Only explicitly allowed message types are passed through the IPC bridge.
- **Zero JIT:** No JS/Node.js required in the renderer process.
