# Spore Component Runtime v1 — PR Train

This file exists to make sequencing explicit. It does not authorize implementation before the standalone Spore repository exists and has fresh destination qualification.

## Immediate Symthaea hardening

### SYM-WASM-000 — current raw-Wasm substrate

Tracked by #4198.

Target implementation sequence:

1. **SYM-WASM-000A — engine/store policy extraction**
   - one internal hardened Wasmtime engine/store factory;
   - fuel, memory/table/instance limits;
   - no WASI;
   - no behavior change outside raw-Wasm sandbox paths.

2. **SYM-WASM-000B — ActionIR convergence**
   - remove direct `Engine::default()` path from `ActionIR::WasmSandbox`;
   - route through hardened executor or retire duplicate execution mechanism;
   - exact functional + adversarial tests.

3. **SYM-WASM-000C — runtime baseline update**
   - exact Wasmtime dependency/lock update;
   - focused security/advisory audit;
   - no WASI rollout.

4. **SYM-WASM-000D — AOT provenance repair**
   - canonical raw-Wasm identity retained independently of native serialized cache;
   - unsafe deserialize only under exact compatible trusted cache provenance;
   - corruption/mismatch negative controls.

Do not combine all four into one large PR unless exact dependency mechanics make a smaller split impossible.

## Spore destination prerequisites

Before SP-WASM implementation:

1. complete extraction boundary #504;
2. create empty `Luminous-Dynamics/spore` per #3843;
3. seed destination and assign migration provenance;
4. import/qualify parity corpus;
5. establish fresh standalone Spore recovery qualification;
6. only then introduce Component Runtime code.

## Component Runtime train

### SP-WASM-000 — constitution

Tracked by #4199.

No runtime code. Freeze:

- Native core / component edge;
- zero ambient authority;
- WIT-first ABI;
- no recovery dependency;
- proposal != authority;
- production p2 / experimental p3 policy;
- no dynamic authority-amplifying composition v1.

### SP-WASM-001 — ABI + manifest

Tracked by #4200.

No execution.

Outputs:

- WIT packages/worlds;
- `ComponentManifestV1`;
- `ResourceBudgetRequestV1`;
- capability vocabulary;
- component/contract/manifest/subject identities;
- deterministic known-answer fixtures;
- static adversarial corpus.

### SP-WASM-002 — admission without execution

Inputs:

- exact component bytes;
- WIT package/world;
- manifest;
- host-supported ABI profiles.

Outputs:

- parsed component import/export inventory;
- compatibility result;
- exact admitted subject identity;
- requested capability set;
- requested resource envelope.

No Wasmtime instantiation and no effective capability yet.

Required failures:

- binary/manifest/contract mismatch;
- unknown import/world/version;
- forbidden authority request;
- manifest/import mismatch;
- incompatible determinism profile.

### SP-WASM-003 — zero-ambient runtime

Introduce Wasmtime Component Model host with **no WASI imports by default**.

Runtime must enforce:

- fuel;
- memory/table/instance limits;
- host-call/input/output budgets;
- bounded lifetime;
- one short-lived Store per invocation initially;
- trap isolation;
- zero recovery/Nix/authority APIs.

Only `compute.pure` / tightly scoped logging imports are allowed.

### SP-WASM-004 — pure/read-only pilot

Fixture:

```text
HostProvidedSystemSnapshotV1
    -> component
    -> SystemSummaryV1
```

No filesystem, network, clock, random, environment or shell imports.

This is the first executable qualification of the Component Runtime.

### SP-WASM-005 — capability resolver

Introduce explicit:

```text
ComponentCapabilityRequest
HostComponentPolicy
OwnerComponentGrant
RuntimeConstraintSnapshot
    -> EffectiveComponentCapability
```

`EffectiveComponentCapability` is opaque/non-Serde and attenuated from all inputs.

Grant currentness/revocation semantics must be explicit before long-lived components are supported.

### SP-WASM-006 — brokered I/O

Add higher-level host adapters before raw WASI filesystem/sockets:

- project-resource handles;
- bounded file reads/writes where appropriate;
- endpoint-ID-bound HTTP requests;
- response/request size budgets;
- provenance logging.

Network/filesystem adapters receive dedicated hostile-component tests.

### SP-WASM-007 — proposal bridge

Component may emit typed proposals only.

```text
Component
    -> ChangeProposalV1
    -> validation/diff
    -> Nixward candidate
    -> independent authority
```

No direct activation or privileged command execution.

### SP-WASM-008 — distribution/provenance

Support immutable distribution subjects:

- OCI/component digest;
- WIT package digest;
- manifest digest;
- publisher proof via Xenia where available;
- offline verified cache;
- Nix materialization.

Mutable tags/locators are never executable identity.

### SP-WASM-009 — WASI 0.3 experimental profile

Only after the host embedding API required by Spore is production-suitable.

Add separately qualified async:

- `future<T>`;
- `stream<T>`;
- async host calls;
- task/stream quotas;
- cancellation/cleanup semantics;
- backpressure tests.

Do not silently migrate synchronous v1 components.

### SP-WASM-010 — Symthaea tool registry

Symthaea can discover and invoke admitted Spore components through the Component Runtime.

Critical invariant:

```text
Symthaea tool choice
    != capability grant
    != execution authority
```

Cognition selects among capabilities already admitted by policy; it cannot widen them.

### SP-WASM-011 — ecosystem/SDK

Deliver:

- Rust SDK;
- WIT packages;
- component manifest generator;
- linter;
- capability preview;
- local deterministic qualification runner;
- hostile-component fixture suite;
- community signing/provenance guidance.

## Release gates

A Component Runtime v1 release requires at least:

1. zero-ambient pure component PASS;
2. hostile resource-exhaustion PASS;
3. host survives every guest trap fixture;
4. import/manifest/capability mismatch corpus PASS;
5. no recovery/LKG/machine-health mutation route exists;
6. exact runtime version and engine configuration are recorded;
7. Wasmtime advisory review current at qualification time;
8. exact source/component/WIT/manifest identities retained in evidence;
9. filesystem/network profiles, if shipped, have their own negative qualification;
10. removal/crash of Component Runtime leaves boot/recovery functional.

## Deliberate non-goals for v1

- general POSIX emulation;
- arbitrary shell plugins;
- arbitrary dynamic component loading;
- native plugin ABI;
- kernel/device-driver replacement;
- Component Runtime dependency in recovery;
- agent-created authority;
- autonomous system mutation.
