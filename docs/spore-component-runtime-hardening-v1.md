# Spore Component Runtime v1 — Hardening Addendum

Status: design/provenance only. This document does **not** implement the Spore Component Runtime and does not grant authority to legacy `symthaea-spore`, Symthaea cognition, or third-party components.

Tracking:

- #4199 — Component Runtime constitution
- #4200 — Component ABI v1 / manifest / capability vocabulary
- #4219 — immediate Symthaea `ActionIR::WasmSandbox` hardening
- #4224 — Spore component admission / compile isolation / pre-execution resource policy
- #505 / #3843 — standalone Spore extraction prerequisites

## 1. Security boundary refinement

The runtime must distinguish at least these stages:

```text
portable bytes
    -> static admission
    -> bounded compilation
    -> compiled-cache entry
    -> current invocation admission
    -> effective capability handles
    -> guest execution
    -> host-call transcript / output
```

None of these stages may inherit authority merely because a prior stage succeeded.

In particular:

```text
ValidComponent
    != SafeDomainOutput
    != TrustedPublisher
    != CurrentlyAuthorizedInvocation
    != Nix/Recovery/DeploymentAuthority
```

## 2. Freeze an explicit Wasm feature profile

Spore must not treat `Wasmtime::Config::new()` defaults as part of the product contract. Runtime upgrades may change which WebAssembly proposals are enabled by default.

Define an exact `SporeWasmFeatureProfileV1` and bind it into runtime/admission evidence.

The profile must explicitly decide, rather than inherit, at least:

- component model;
- threads / atomics / shared memory;
- shared-everything threads;
- memory64;
- multi-memory;
- reference types;
- function references;
- GC;
- tail calls;
- SIMD;
- relaxed SIMD;
- bulk memory;
- extended const;
- stack switching;
- maximum Wasm stack;
- compiler backend;
- async support;
- fuel;
- epoch interruption.

### Recommended v1 posture

Start smaller than Wasmtime's general-purpose defaults:

```text
component model          enabled
fuel                     enabled
threads                  disabled
shared memory            disabled
shared-everything        disabled
memory64                 disabled
GC                       disabled unless WIT/component tooling proves it is required
stack switching          disabled
relaxed SIMD             disabled initially
SIMD                     disabled initially for pure qualification profile;
                         later explicit performance profile may enable it
multi-memory             disabled unless required by the selected component toolchain
```

Reference types / function references / bulk memory may be required by the Component Model/toolchain and should be enabled only to the exact minimum supported profile, with fixtures proving the profile is sufficient.

Do not claim the above draft list is final until the first WIT fixture is compiled and the required feature closure is measured.

### Determinism

Relaxed SIMD is especially unsuitable for the default evidence-sensitive profile because allowed results may differ by host architecture. If a future profile enables it, either force Wasmtime's deterministic relaxed-SIMD lowering or mark that profile explicitly architecture-dependent.

## 3. Compilation is an untrusted-input boundary

Guest runtime fuel and `StoreLimits` do not prove compilation is bounded.

The production untrusted-component path should prefer an admission/compile worker with OS-level quotas:

```text
systemd transient worker
  DynamicUser=yes
  NoNewPrivileges=yes
  PrivateTmp=yes
  ProtectSystem=strict
  ProtectHome=yes
  RestrictAddressFamilies=AF_UNIX
  MemoryMax=<policy>
  TasksMax=<policy>
  LimitNOFILE=<policy>
  CPUQuota=<policy>
  RuntimeMaxSec=<policy>
  bounded writable cache/temp directory only
```

Exact sandbox knobs remain implementation-dependent, but the theorem is not:

> a hostile component must not obtain more host resources merely because it has not reached its first guest instruction yet.

## 4. Host-call resource policy

Every WIT host function is a potential host-memory / host-I/O allocator.

For each host interface define:

- maximum argument byte size;
- maximum result byte size;
- maximum per-call host allocation;
- maximum cumulative host allocation per invocation where measurable;
- maximum call count;
- maximum resource-handle count;
- maximum pending/streamed data;
- cancellation semantics;
- error class on budget exhaustion.

Use Wasmtime host-call fuel/resource-table limits where available, but do not assume they replace interface-specific quotas.

Examples:

```text
spore:log/event
  <= 16 KiB one event
  <= 128 events/invocation
  <= 256 KiB cumulative

spore:project/read-file
  <= 4 MiB one result
  <= policy cumulative project-read budget

spore:http/request
  <= fixed header bytes
  <= request-body bytes
  <= response-body bytes
  <= redirect count
  <= request count
```

Numbers above are examples only; freeze exact values with performance/evidence fixtures.

## 5. Capability handles, not authority-bearing strings

Prefer host-created WIT `resource` handles for capabilities whose scope refers to an external object.

Do not make this:

```text
read-file("/home/alice/project", "Cargo.toml")
```

the authority model.

Prefer:

```text
project-handle = host grants exact project snapshot/root
read-file(project-handle, "Cargo.toml")
```

Likewise:

```text
endpoint-handle = host grants exact API policy
request(endpoint-handle, request)
```

The component cannot manufacture a valid host resource merely by guessing a string or integer. The host resource table binds each guest-visible handle to an exact current capability object.

## 6. Per-invocation currentness

Installation/admission is historical evidence, not open-ended permission.

Each execution should construct a new opaque `EffectiveComponentInvocationV1` (name provisional) bound to at least:

- exact `ComponentSubjectId`;
- exact runtime profile / engine policy ID;
- exact local component-policy generation/ID;
- exact owner/org grant identity where required;
- exact requested + attenuated resource budget;
- exact capability handle set;
- exact invocation nonce/ID;
- current trusted policy epoch;
- optional expiry/deadline;
- exact input/snapshot identity where relevant.

The effective invocation object should be non-Serde and non-Clone by default.

Dropping the invocation's Wasmtime `Store` should invalidate every WIT resource handle associated with the invocation.

## 7. Stateful components are a separate profile

V1 pure/read-only components should be short-lived and stateless unless state is explicit input/output.

Do not give a component persistent host directories merely because it wants state.

A later state profile should expose typed/scoped storage, for example:

```text
spore:state/kv
  namespace = host-owned exact component-state subject
  quotas = explicit
  schema/version = explicit
  migration = explicit
  rollback/compatibility = explicit
```

State identity must not silently follow a mutable display name. Component version/subject changes need explicit state-compatibility policy.

## 8. Secrets should be usable without being extractable

Prefer operation capabilities over raw secret bytes.

Examples:

```text
sign(signing-handle, message)
decrypt(decryption-handle, ciphertext)
authenticate(http-endpoint-handle, request)
```

rather than:

```text
get-secret("github-token") -> bytes
```

A component that can use a credential still should not necessarily be able to copy/export that credential.

High-value owner/recovery/Xenia private material remains outside the general component runtime entirely unless a future dedicated broker proves a narrower theorem.

## 9. Determinism classes

The manifest determinism declaration must correspond to actual imports/capabilities.

Recommended v1 classes:

### PureDeterministic

- no host observation except explicit input;
- no clock;
- no random;
- no network;
- no persistent state;
- no architecture-dependent relaxed SIMD;
- same component/profile/input expected to produce same canonical output.

### SnapshotDeterministic

- consumes one exact immutable host snapshot as explicit input/capability;
- receipt binds snapshot digest;
- no other nondeterministic host interface.

### SeededDeterministic

- host supplies exact seed as explicit input;
- no ambient random interface;
- receipt binds seed or seed commitment as policy permits.

### NondeterministicObserved

- network, clock, randomness, mutable external service or state is involved;
- receipt records the exact observations/provenance available;
- replay equality is not claimed.

A component cannot self-label `PureDeterministic` if its imports/capabilities permit nondeterminism.

## 10. Invocation evidence

Keep facts, semantic verification, and authority separate.

Suggested evidence chain:

```text
ComponentInvocationObservationV1
  subject/runtime/profile/input identity
  effective capability IDs
  resource budgets
  host-call transcript digest
  traps/resource exhaustion
  output digest

        -> ComponentInvocationVerificationV1

        -> optional domain-specific qualification
```

A successful invocation proves at most that one exact component ran under one exact admitted environment and produced one observed result.

It does not prove the result is true, safe, desirable, or authorized for machine mutation.

## 11. Host-call transcript

For brokered capabilities, record a canonical transcript digest sufficient to bind evidence without necessarily storing sensitive contents.

Example event identity:

```text
sequence number
interface/function ID
capability-handle ID
request metadata digest
response metadata digest
byte counts
result disposition
budget deltas
```

Sensitive request/response bodies may be represented by domain-separated digests or redacted evidence according to policy.

Transcript identity is evidence, not authorization.

## 12. Cancellation and failure semantics

The host must define cancellation behavior for:

- fuel exhaustion;
- epoch/wall timeout;
- user cancellation;
- component trap;
- broker timeout;
- network cancellation;
- host process shutdown;
- async stream/task cancellation in future p3 profiles.

A cancelled component must not leave a partially granted capability alive beyond its invocation.

Any host-side operation that can mutate state later must have its own transactional/authority semantics; killing the Wasm guest is not rollback.

## 13. Side-channel boundary

WebAssembly sandboxing does not prove absence of timing/cache/speculative/co-tenancy side channels.

For v1:

- do not pass raw high-value secrets to general components;
- do not grant high-resolution time by default;
- keep high-authority cryptographic operations in dedicated brokers;
- consider process/VM isolation for untrusted components handling sensitive material;
- record CPU/runtime profile where result semantics depend on hardware features.

## 14. Dynamic composition remains denied in v1

One admitted component cannot import/load another arbitrary component and thereby acquire the union of both capability sets.

If static component composition is supported by the toolchain, admit the final composed artifact as a **new exact component subject** and reevaluate the entire import/capability closure.

## 15. Upgrade theorem

Changing any of the following requires explicit requalification of the affected runtime profile:

- Wasmtime version;
- WIT contract digest;
- engine feature profile;
- compiler backend/configuration;
- host-call budget semantics;
- capability broker semantics;
- resource-limiter policy;
- component canonicalization/identity rules.

A semantic-version range alone is not sufficient evidence that a previously qualified component/runtime claim still holds.

## 16. Immediate sequencing

```text
#4219 SYM-WASM-000A
  harden the existing ActionIR raw-Wasm path

#4198 later children
  align both raw-Wasm executors
  extract neutral common raw-Wasm policy
  upgrade the direct Wasmtime baseline
  requalify AOT boundary

standalone Spore extraction
  ↓
#4199 SP-WASM-000 constitution
  ↓
#4200 SP-WASM-001 WIT / manifest / identity
  ↓
#4224 SP-WASM-002 admission / compile isolation
  ↓
SP-WASM-003 zero-ambient runtime
  ↓
SP-WASM-004 pure pilot + adversarial runtime corpus
```

The product implementation should not jump directly from a manifest parser to third-party network/filesystem plugins.