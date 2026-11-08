# Spore Component Runtime v1 — Architecture and PR Plan

Status: design/provenance only. This document does **not** implement a Spore runtime and does not transfer product authority into the legacy `crates/domains/symthaea-spore` crate.

Tracking:

- #4198 — SYM-WASM-000: consolidate and harden existing Symthaea raw-Wasm execution
- #4199 — SP-WASM-000: freeze post-extraction Spore Component Runtime constitution
- #4200 — SP-WASM-001: freeze Component ABI v1, manifest, and capability vocabulary
- #504 / #505 / #3843 — standalone Spore extraction and destination creation

## 1. Architectural thesis

Spore should use WebAssembly Components and selective WASI as a **bounded extension substrate**, not as its operating-system or recovery substrate.

```text
Native Rust / Linux / Nix trusted core
        |
        +-- boot / recovery / LKG / physical boot identity
        +-- Nixward deterministic candidate construction and execution
        +-- Xenia / owner / organization authority
        |
        `-- Spore Component Runtime
              |
              +-- versioned WIT worlds
              +-- WebAssembly Component Model
              +-- explicit capability admission
              +-- selected/brokered WASI interfaces
              +-- third-party tools/extensions
```

Constitutional rule:

> **Native core, component edge.**

Components may compute, observe, transform, and propose. They do not create machine health, generation-promotion eligibility, LKG truth, recovery authority, deployment authority, or privileged execution authority.

## 2. Existing WebAssembly systems are separate

Do not collapse the following into one architecture:

### Browser/edge Symthaea Spore

`crates/domains/symthaea-spore` is the existing wasm-bindgen/browser cognition/config-generation kernel. It is not the extracted boot/recovery Spore product.

### Symthaea raw-Wasm sandbox

Symthaea already contains Wasmtime execution for generated/internal plugins, including `WasmArchitect` and `ActionIR::WasmSandbox`. #4198 owns consolidation and security hardening of those paths.

### Future Spore Component Runtime

The product runtime is WIT/Component-Model-first, capability brokered, and belongs in the standalone Spore destination after extraction.

```text
browser WASM cognition
        !=
Symthaea raw-Wasm execution
        !=
Spore Component Runtime
```

## 3. Standards posture

At this plan's creation:

- WASI 0.3.0 is ratified and introduces native Component Model async primitives;
- Wasmtime's Rust `wasmtime_wasi::p3` host module is still explicitly experimental/incomplete and not production-semver-stable;
- Wasmtime's Component Model embedding API supports custom WIT worlds and host-provided imports through the component linker.

Therefore Spore v1 should make **WIT the durable ABI boundary** and keep the runtime implementation replaceable/evolvable.

Initial policy:

```text
custom WIT worlds        canonical product ABI
WASI Preview 2 adapters  selectively usable production capability
WASI Preview 3 adapter   experimental async profile only
```

References:

- https://wasi.dev/releases/wasi-p3
- https://docs.wasmtime.dev/api/wasmtime_wasi/p3/index.html
- https://docs.wasmtime.dev/api/wasmtime/component/index.html
- https://component-model.bytecodealliance.org/design/worlds.html

## 4. Default component authority

The default admitted component receives no ambient host capability:

```text
filesystem       NONE
network          NONE
environment      NONE
clock            NONE
random           NONE
stdio            NONE
devices          NONE
shell            NONE
Nix              NONE
Spore authority  NONE
recovery/LKG      NONE
```

Host access exists only through explicit WIT imports supplied by the runtime.

Do not make `wasi:cli/command` the default Spore plugin world. Prefer narrow interfaces such as:

```text
spore:log/event@1
spore:observe/system@1
spore:project/read@1
spore:http/request@1
spore:proposal/change@1
```

## 5. Capability theorem

Keep four layers distinct:

```text
WIT import declaration
    !=
ComponentManifest capability request
    !=
local/owner policy grant
    !=
EffectiveComponentCapability
```

Conceptually:

```text
effective =
    requested
  ∩ host policy maximum
  ∩ owner/organization grant
  ∩ current runtime constraints
```

`EffectiveComponentCapability` should be opaque/non-Serde and only constructible after admission/currentness checks. Serializable manifests are untrusted descriptions, not capabilities.

## 6. Portable identity

Bind independently:

```text
ComponentBinaryId
WitContractId
ComponentManifestId
ComponentSubjectId
```

The exact subject should commit to the portable component bytes plus exact contract/manifest semantics.

Do not use any of these as portable component identity:

- filename;
- registry tag;
- display name;
- semantic version alone;
- Wasmtime serialized native code;
- machine-specific AOT cache entry.

Publisher signing should cover portable component/contract/manifest bytes. Native compilation artifacts are local disposable caches keyed by component identity + engine/runtime configuration + target.

## 7. Runtime resource boundary

Every invocation should have explicit limits for:

- fuel/instruction budget;
- memory bytes;
- memories;
- tables/table elements;
- instances;
- input bytes;
- output bytes;
- host calls;
- wall timeout as defense-in-depth;
- future async task/stream ceilings.

Prefer a short-lived Wasmtime `Store` for each invocation unless a future stateful component contract explicitly justifies longer lifetime.

## 8. Brokered filesystem/network before generic WASI

Avoid starting with host-path or raw-socket authority.

Prefer host-mediated capabilities:

```text
spore:project/read-file(resource-handle, relative-path)
spore:http/request(endpoint-id, request)
```

The resource handle / endpoint ID is resolved under host policy. Components do not need to know arbitrary host paths, DNS reachability, or raw IP topology merely to consume one resource.

Raw WASI filesystem/socket profiles can be added later under their own qualification theorem.

## 9. Proposal-only bridge to Nixward

A component may eventually emit a typed proposal, never an execution permit.

```text
component
    -> ChangeProposalV1
    -> Spore/Nixward validation
    -> exact diff/candidate
    -> owner/organization authority
    -> deterministic execution
    -> independent observation/qualification
```

A component cannot directly invoke:

- Nix activation;
- LKG mutation;
- generation promotion;
- recovery selection;
- boot-attempt mutation;
- privileged service control;
- Xenia authority minting;
- arbitrary shell execution.

## 10. First pilot

The first runtime component should deliberately prove the smallest theorem:

```text
HostProvidedSystemSnapshotV1
    -> pure deterministic component
    -> SystemSummaryV1
```

No filesystem, network, environment, clock, entropy, shell, Nix, or recovery imports.

Adversarial corpus:

1. infinite loop -> fuel trap;
2. memory bomb -> limiter trap;
3. unknown import -> linker refusal;
4. filesystem/network/environment/clock/random request -> unavailable;
5. oversized input/output -> host refusal;
6. malformed component -> validation refusal;
7. wrong WIT world/version -> admission refusal;
8. trap/panic -> guest fails, host remains healthy;
9. changed component bytes -> identity mismatch;
10. manifest capability escalation -> denied;
11. guest failure cannot change machine health, LKG, recovery, or deployment state.

## 11. Composition rule

Dynamic authority-amplifying composition is out of scope for v1.

The host may:

- instantiate admitted components independently; or
- admit a statically composed component as a new immutable subject with its own contract/manifest/capability evaluation.

Component A does not inherit component B's capabilities merely by composing/loading B.

## 12. PR train

### Immediate Symthaea lane

**SYM-WASM-000 (#4198)**

- consolidate raw-Wasm engine/store policy;
- remove the independent unbounded `Engine::default()` execution path;
- update the Wasmtime baseline under exact lock qualification;
- adversarial fuel/memory/import/AOT tests;
- no WASI rollout.

### Post-extraction Spore lane

**SP-WASM-000 (#4199)** — constitution only.

**SP-WASM-001 (#4200)** — WIT ABI v1 + manifest/capability/resource/identity schemas; no execution.

**SP-WASM-002** — component admission + exact identity + import introspection + compatibility policy; no execution.

**SP-WASM-003** — Component host with zero ambient WASI and explicit fuel/resource limits.

**SP-WASM-004** — pure/read-only system-snapshot pilot and adversarial runtime qualification.

**SP-WASM-005** — request/policy/grant/effective-capability resolver.

**SP-WASM-006** — brokered project/filesystem and HTTP/network adapters.

**SP-WASM-007** — typed Nixward proposal bridge; no direct execution.

**SP-WASM-008** — immutable distribution/provenance (OCI digest, publisher/Xenia proof, offline cache, Nix materialization).

**SP-WASM-009** — experimental WASI 0.3 async/stream profile.

**SP-WASM-010** — Symthaea tool-registry adapter consuming admitted Spore components without expanding authority.

**SP-WASM-011** — SDK, linter, capability preview and community/adversarial qualification tooling.

## 13. Security baseline

Do not introduce Spore WASI on Symthaea's current Wasmtime `44.0.1` dependency merely because Wasmtime is already present.

The product runtime must establish and pin its own current security baseline, then qualify it before adding filesystem/network imports.

Wasmtime had multiple 2026 advisories relevant to WASI filesystem isolation. This reinforces defense-in-depth rather than invalidating the architecture:

```text
Wasm isolation
    + WIT import minimization
    + Spore capability broker
    + fuel/resource limits
    + process/systemd sandbox where applicable
    + Xenia/owner authority separation
    + Nix-pinned runtime
```

Relevant advisory/reference examples:

- https://github.com/bytecodealliance/wasmtime/security/advisories/GHSA-vqjp-4c8c-hfgg
- https://github.com/bytecodealliance/wasmtime/security/advisories/GHSA-4ch3-9j33-3pmj
- https://github.com/bytecodealliance/wasmtime/security/advisories/GHSA-2r75-cxrj-cmph

## 14. Completion theorem

This planning line does not qualify a runtime.

The design is ready for implementation only when:

1. standalone Spore exists and has fresh destination qualification;
2. its recovery core remains independent of Wasmtime/WASI;
3. SP-WASM-000 is accepted as the component authority constitution;
4. SP-WASM-001 freezes exact component/WIT/manifest identity before execution code lands;
5. the first runtime host begins with zero ambient WASI and a non-authoritative pure pilot.

The intended product rule remains:

> **Intelligence proposes. Authority permits. Deterministic machinery executes. Independent evidence decides what happened. Components never skip those boundaries.**
