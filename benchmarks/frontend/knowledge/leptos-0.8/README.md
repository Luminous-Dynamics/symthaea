# Leptos 0.8 Frontend Knowledge Corpus

This directory seeds Symthaea's versioned Leptos knowledge base from the
existing public Symthaea applications.

## Current status

- `patterns-v0.json` contains eight source-grounded patterns.
- Every pattern records its exact source path and Git blob SHA.
- Every pattern has an explicit evidence state; none is marked browser-verified.
- The corpus is pinned to the Leptos 0.8 API family and the current repository
  dependency requirement.
- The code examples are learning excerpts, not a standalone crate. They must
  compile against the exact Cargo.lock resolution before their status is
  promoted.

## Pattern inventory

1. Function component and view composition.
2. Local state with `RwSignal`.
3. Tracked reads versus `get_untracked()`.
4. Effects for browser/external side effects.
5. Client-side async work with `spawn_local`.
6. Keyed collection rendering with `For`.
7. Typed `NodeRef` for DOM-backed rendering.
8. A narrow HTTP/WebSocket transport boundary between the WASM UI and the
   cognitive service.

## Most important reactivity rule

A tracked read inside an `Effect` subscribes the effect to the signal. An
untracked read does not. Use the former when a signal change should rerun the
effect; use the latter only when the algorithm deliberately samples state
without creating a reactive dependency.

Symthaea's telemetry retry loop deliberately reads the gateway with
`get_untracked()` on each retry attempt. It does not forcibly replace an
already-open socket when the gateway field changes. That behavior is stated in
the source-observed interface contract.

## Validation and provenance

Run the standalone Rust validator from the repository root:

```bash
cargo run --manifest-path benchmarks/frontend/contract-check/Cargo.toml -- --patterns
```

The validator checks the corpus against `pattern-v1.schema.json`, then
recomputes Git blob SHA-1 for every source path. If a referenced source file
changes, the pattern is stale and validation fails until the example and
provenance are reviewed and updated. Unsafe or parent-traversing paths fail
closed.

## Evidence promotion

Knowledge moves through explicit states:

`source_observed_not_compiled -> reference_checked_not_compiled -> compiled -> unit_tested -> browser_verified`

Each promotion requires an exact commit, toolchain/dependency record and the
appropriate compile, test or browser evidence. A generated example is not
evidence that the example works. Stale patterns are quarantined until
revalidated.

## External API references

- Leptos 0.8.20 `Effect`: https://docs.rs/leptos/0.8.20/leptos/prelude/struct.Effect.html
- Leptos 0.8.22 `RwSignal`: https://docs.rs/leptos/0.8.22/leptos/prelude/struct.RwSignal.html
- Leptos 0.8.22 `Memo`: https://docs.rs/leptos/0.8.22/leptos/prelude/struct.Memo.html
- Leptos `Owner`: https://docs.rs/leptos/0.8.21/leptos/prelude/struct.Owner.html
- gloo-net 0.5.0 `WebSocket`: https://docs.rs/gloo-net/0.5.0/gloo_net/websocket/futures/struct.WebSocket.html
- gloo-net 0.5.0 WebSocket `State`: https://docs.rs/gloo-net/0.5.0/gloo_net/websocket/enum.State.html

The current browser implementation should follow the actual resolved versions in
`Cargo.lock`, not blindly follow `latest` documentation.

## Next curriculum increments

- Component properties, children, context and composition boundaries.
- Reactive resources, async state and cancellation.
- Forms, validation and accessibility semantics.
- Routing, navigation state and error boundaries.
- SSR/hydration as a separate track; the current Symthaea bridge UIs are CSR.
- CSS layout, design tokens, responsive constraints and motion.
- Browser rendering, performance, CSP and untrusted content.
- Component testing, visual regression, keyboard journeys and seeded defect
  repair.
