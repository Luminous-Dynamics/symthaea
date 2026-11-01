# Rust-First Language Boundaries

**Status:** engineering policy proposal tracked by [#9749](https://github.com/Luminous-Dynamics/symthaea/issues/9749).

## Decision

Symthaea should be Rust-first for domain logic and authoritative runtime behavior, while retaining JavaScript where the browser platform itself requires it. This is a boundary policy, not a language purity rule.

## Language ownership

| Layer | Default | Ownership |
|---|---|---|
| Domain logic, cognition, planning, evidence, proof graphs, trust, policy, resource accounting | Rust | Authoritative semantics and validation |
| Backend, agent orchestration, schedulers, protocol handlers, security-sensitive code | Rust | Authorization, invariants, enforcement interfaces, durable state |
| Browser product UI | Leptos + Rust/WASM | UI state and typed domain requests; server remains authoritative |
| Browser API adapters | Small JavaScript or Rust `web-sys` bindings | DOM, Web Workers, Service Worker entry points, Web Audio and vendor SDK interop |
| Browser-only legacy code | TypeScript if it must remain substantial | Presentation and adapter logic only; no duplicated authoritative rules |
| Offline research and data preparation | Python where the scientific ecosystem materially helps | Reproducible experiment boundary; no hidden production authority |
| Generated WebAssembly glue | Tool-generated JS | Generated output only; never hand-edited |

TypeScript adds static checking to JavaScript source but does not change the browser runtime or create a security boundary. Rust/WASM can share types and pure logic with native code, but code running in the browser is still untrusted from the server's point of view.

## Audit of current browser-facing examples

Audited against the repository `main` tree at `36b2105811200dd60b2d9eed1177f819e7ac9467` on 2026-11-01.

- `crates/bridges/symthaea-web/Cargo.toml` is already a Rust/Leptos CSR frontend. It uses `wasm-bindgen`, `js-sys`, and `web-sys` to access browser APIs.
- `crates/bridges/symthaea-ui/Cargo.toml` is another Rust/Leptos frontend. Treat these as convergence candidates; avoid creating a third frontend stack without a concrete product reason.
- `crates/domains/symthaea-spore/build-wasm.sh` compiles the Spore kernel to `wasm32-unknown-unknown`, runs optimization, and generates bindings.
- `crates/domains/symthaea-spore/www/js/app.js` contains handwritten portal state, tabs, worker lifecycle, and initialization. It is a candidate for staged consolidation into the Leptos frontend after golden-trace coverage exists.
- `crates/domains/symthaea-spore/www/js/charts.js` is canvas presentation code. Port it only as part of a UI consolidation or when a typed shared chart model gives measurable value.
- `crates/domains/symthaea-spore/www/js/consciousness-audio-bridge.js` and `crates/domains/symthaea-muse/js/consciousness-audio-bridge.js` were byte-identical in the audited tree. Consolidate their shared source, but keep generated deployment artifacts and product-specific configuration separate.
- `crates/domains/symthaea-spore/www/js/consciousness-audio-worklet.js` subclasses the browser's `AudioWorkletProcessor`. Keep the registration/lifecycle shell in JS. Benchmark any DSP/ring-buffer extraction into Rust/WASM rather than assuming it will improve realtime behavior.
- `crates/bridges/symthaea-web/assets/pkg/symthaea_spore.js` is generated wasm-bindgen output. It is not handwritten application logic and must not be migrated or edited manually.

## Security boundary

1. Browser UI and browser WASM are untrusted clients. A user can alter them, replay requests, skip client validation, or invoke endpoints directly.
2. The service independently validates identity, authorization, request shape, freshness, replay protection, operation admissibility, and hard resource limits.
3. Shared Rust types are a consistency mechanism, not proof that client input is trusted.
4. Independent proof/evidence verifiers, budget ledgers, and execution authorities remain separate from the predictor and UI.
5. No secret, signing authority, provider credential, or authoritative policy decision is moved into client WASM.

## Migration order

1. Freeze behavior with golden traces: record browser events, worker messages, state transitions, and visible outputs for normal, error, offline, refresh, and suspend/resume paths.
2. Consolidate the identical Spore/Muse audio bridge source into one owned module and verify both products against identical traces.
3. Introduce shared Rust request/state types for worker and portal messages; add versioned schema and compatibility tests.
4. Port the app state machine in small slices into the existing Leptos app; keep a compatibility adapter until differential traces match.
5. Port chart rendering only if it meaningfully reduces duplicate logic or improves accessibility/performance.
6. Keep the AudioWorklet JS shell. Experiment with Rust/WASM DSP only behind A/B benchmarks.
7. Remove legacy JS only after parity, error recovery, accessibility, and deployment tests pass.

Do not replace JavaScript with TypeScript solely to say the stack is typed; do not rewrite stable code without a measured benefit.

## Benchmark and acceptance matrix

| Area | Measure | Gate |
|---|---|---|
| Behavioral parity | Golden event/action/state traces | Exact semantic match for all supported workflows; intentional differences explicitly reviewed |
| Trust and security | Unauthorized, malformed, stale, replayed, and forged client requests | Zero unauthorized state changes; server rejects every negative control |
| Worker reliability | Message loss, duplication, ordering, retries, cancellation | Zero lost/duplicated accepted messages; deterministic ordering and idempotent retry semantics |
| Startup | Cold/warm startup p50/p95 on target desktop and mobile devices | No material regression; report compressed JS+WASM bytes and parse/compile time |
| Runtime | CPU, allocations, memory high-water mark per cycle | Improve or hold current performance; no unbounded allocations in realtime path |
| Audio | 44.1/48 kHz, 128-frame callbacks, p50/p95/p99 jitter, underruns, deadline misses | Zero deadline misses in qualification runs; zero underruns in steady-state test windows |
| Accessibility | Keyboard navigation, screen readers, focus and reduced-motion behavior | No regression against current UI; WCAG AA review for user-facing flows |
| Offline/lifecycle | Offline start, reconnect, refresh, suspend/resume, worker restart | No state corruption or duplicate side effects |
| Supply chain | Build reproducibility, pinned Rust/Nix inputs, generated artifacts | Reproducible WASM build and verifiable generated glue |

Benchmarks must run on the same hardware, browser versions, workload seeds, payloads, and production-like build flags. Report raw distributions and confidence intervals; do not report only averages. A microbenchmark win does not justify a migration if startup, bundle size, accessibility, or recovery regresses.

## Recommended next technical changes

- First: consolidate the identical audio bridge, add a golden event protocol, and enforce explicit schema versions for JS↔WASM worker messages.
- Second: compare the current Spore portal state machine with the Rust/Leptos `symthaea-web` frontend and plan a staged convergence path.
- Third: build a WASM audio benchmark harness before deciding whether any real-time DSP belongs in Rust/WASM or JavaScript.
- Avoid blanket JavaScript deletion, a TypeScript rewrite of the entire app, or moving server authority to browser/WASM code.

## Decision summary

Rust owns the system's semantics and authority. Leptos/WASM is the preferred browser application layer. JavaScript is a narrow platform adapter. TypeScript is a containment option for unavoidable larger browser-only modules. Python is a bounded research tool. Every migration is justified by measured correctness, security, performance, or maintenance gains.