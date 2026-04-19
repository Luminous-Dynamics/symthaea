# mycelix-leptos-client

**Browser-compatible Holochain client for Leptos frontends.**

A pure-Rust replacement for [`@holochain/client`](https://www.npmjs.com/package/@holochain/client) (JavaScript) intended for WASM frontends — particularly [Leptos](https://leptos.dev) apps shipping AGPL application code that wants to stay out of the Node.js ecosystem.

Uses `web-sys::WebSocket` + [`rmp-serde`](https://crates.io/crates/rmp-serde) (MessagePack) to communicate with a Holochain conductor over binary WebSocket frames.

## Why

The official Rust client, [`holochain_client`](https://crates.io/crates/holochain_client), is built for native Rust — it transitively depends on `tokio`, `holochain_websocket`, and other crates that do not compile to `wasm32-unknown-unknown`. For browser WASM frontends, the official path is `@holochain/client` (TypeScript) + some JS bundling layer.

`mycelix-leptos-client` is the third option: a fresh-written, browser-native Rust client that avoids the TypeScript round-trip entirely. If you are building a Leptos / Yew / any-Rust-WASM frontend against a Holochain conductor, this crate lets you call zome functions without ever touching npm.

## Features

- **Pure Rust, pure WASM.** Zero JS shim, no bundler dance — just add the crate and call zome functions.
- **MessagePack wire format.** Matches what Holochain conductors speak natively; no JSON adaptation layer.
- **Trait-based transport** so the same `HolochainClient` type works across:
  - [`BrowserWsTransport`] — browser WebSocket (feature `browser`, default)
  - [`NativeWsTransport`] — native integration tests (feature `native`)
  - `TauriIpcTransport` — planned, invoke a Tauri backend
  - `MockTransport` — unit tests, no network
- **Typed zome calls** via serde-derived input/output structs.
- **AGPL-3.0-or-later.** Intended for application-layer code. Pair with your own AGPL app or negotiate a commercial exception.

## Example

```rust
use mycelix_leptos_client::{HolochainClient, BrowserWsTransport};
use serde::{Serialize, Deserialize};

#[derive(Serialize)]
struct CreateProposal { title: String, body: String }

#[derive(Deserialize)]
struct ProposalHash { hash: Vec<u8> }

async fn example() -> Result<(), Box<dyn std::error::Error>> {
    let transport = BrowserWsTransport::new();
    let client = HolochainClient::new(transport, "mycelix-unified", "governance");

    // Connect; auth token is optional (None = no auth)
    client.connect("ws://localhost:8888", None).await?;

    let result: ProposalHash = client
        .call_zome(
            "agora",
            "create_proposal",
            &CreateProposal {
                title: "Test".into(),
                body:  "Body".into(),
            },
        )
        .await?;

    Ok(())
}
```

## Cargo feature flags

| Feature | Default | What it turns on |
|---|---|---|
| `browser` | yes | `BrowserWsTransport` over `web-sys::WebSocket` |
| `tauri` | no | Tauri IPC transport (planned) |
| `native` | no | `NativeWsTransport` over `tokio-tungstenite` — integration tests only |

The default feature set is correct for a Leptos WASM frontend. For a Tauri desktop app wrapper you would enable `tauri` instead.

## Compatibility

- **Holochain conductor:** 0.6+ (tracks the protocol, not the specific conductor version). Tested against the "shared conductor" bootstrap of the Mycelix ecosystem.
- **Rust:** 1.85+ (edition 2021, matches the Leptos 0.8 line).
- **Browser:** anything with WebSocket + WebCrypto; no Service Worker assumptions.

## Where this ships

This crate is developed inside the Luminous Dynamics monorepo (private) and published to crates.io from there. The public mirror at <https://github.com/Luminous-Dynamics/mycelix-leptos-client> carries the per-release source snapshots.

It is used in production by several Mycelix ecosystem apps:

- [Mycelix Praxis](https://praxis.mycelix.net) — sovereign learning platform, Leptos + Holochain
- [Mycelix Craft](https://github.com/Luminous-Dynamics/mycelix) — talent / credentials marketplace
- [Mycelix Sovereign](https://github.com/Luminous-Dynamics/xenia-peer) — remote-session PAM suite (via the Xenia admin console)

## License

AGPL-3.0-or-later. See [`LICENSE`](LICENSE) for the full text.

Commercial licensing: contact the authors for dual-license terms if AGPL does not fit your deployment shape.

## Contributing

This crate is developed in the private Luminous Dynamics monorepo. External PRs on the public mirror are welcome and will be carried forward into the monorepo source of truth by the maintainers. Do not expect a fast turnaround yet — the public repo is a source mirror, not a primary-development surface.
