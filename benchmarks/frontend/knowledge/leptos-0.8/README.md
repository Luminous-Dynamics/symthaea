# Leptos 0.8 Frontend Knowledge Corpus

This directory seeds Symthaea's versioned Leptos knowledge base from the
existing public Symthaea applications.

## Current status

- `patterns-v0.json` contains seventeen source-grounded patterns across the Symthaea and Muse Leptos applications.
- Every pattern records its exact source path and Git blob SHA.
- Every pattern has an explicit evidence state; none is marked browser-verified.
- The corpus is pinned to the Leptos 0.8 API family. The repository's current
  `Cargo.lock` resolves `leptos` to 0.8.20; this exact resolved version is the
  baseline, even though upstream has since published 0.8.23.
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
9. Declarative Router / Routes / Route composition and active links.
10. Typed context provision and descendant retrieval without prop drilling.
11. Derived state with `Memo` and a single source of truth.
12. Programmatic navigation after a completed state transition.
13. Pure reducer state machines with typed events and explicit effects.
14. Persistent DOM resources placed outside route subtrees.
15. Browser event adapters that dispatch epoch-tagged reducer events.
16. Fail-closed authorization gating before a mutating import.
17. Asynchronous component initialization with visible loading and failure states.

The first eight patterns originate in `symthaea-ui`; the expanded set also
uses the existing `symthaea-muse-ui` Router, shared playback store, pure
playback reducer, composition page and private-import flow. Each Muse example
records its exact source blob SHA, including related source files where the
pattern spans multiple modules.

## Framework version discipline

As of 2026-11-22, upstream has published Leptos 0.8.23 as the latest stable
0.8 release, with 0.9.0-rc1 as a prerelease. This repository's checked-in
`Cargo.lock` currently resolves Leptos 0.8.20. The SFEB baseline must record
and use 0.8.20 until an explicit 0.8.23 compatibility experiment updates the
lockfile and passes compile, browser and regression gates. Do not silently mix
the two versions in one benchmark result.

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
- Leptos 0.8.20 `RwSignal`: https://docs.rs/leptos/0.8.20/leptos/prelude/struct.RwSignal.html
- Leptos 0.8.20 `Memo`: https://docs.rs/leptos/0.8.20/leptos/prelude/struct.Memo.html
- Leptos 0.8.20 `Owner`: https://docs.rs/leptos/0.8.20/leptos/prelude/struct.Owner.html
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

## Retrieval evaluation

A first public pilot query set is available at
`retrieval-queries-v0.json`, with its schema in
`retrieval-v1.schema.json`. It contains 34 paraphrased positive queries and
eight hard negatives, so the system must retrieve the right entry when one
exists and abstain when the corpus has no relevant pattern.

Validate query structure and gold-ID membership with:

```bash
cargo run --manifest-path benchmarks/frontend/contract-check/Cargo.toml -- --retrieval
```

Compare exact same queries and corpus using keyword, BM25, TF-IDF, dense
embedding, HDC associative retrieval and hybrid retrieval. Report recall@1/3,
MRR, nDCG@3, precision@3, negative-query false-positive rate, p50/p95 latency,
memory and cost. This query set is public development data, not a hidden
holdout; final claims require an additional private query set.

