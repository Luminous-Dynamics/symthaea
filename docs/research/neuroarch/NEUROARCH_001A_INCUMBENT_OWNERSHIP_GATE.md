# NEUROARCH-001A — Active HDC/LTC incumbent ownership gate

Status: design/ownership gate; blocks NEUROARCH-001 qualification
Parent: #3316
Related: #3317, draft PR #3325

## Why this gate exists

The repository contains two divergent HDC/LTC implementations with similar names but different roles:

1. `crates/core/symthaea-core/src/hdc/hdc_ltc_unified.rs` — the active in-core implementation re-exported by the main Symthaea HDC surface and consumed broadly across domain/controller crates.
2. `crates/core/symthaea-hdc-ltc` — a prior standalone extraction recorded by `docs/crate-status.toml` as an archived/orphaned extraction whose migration was never completed.

The two lineages have diverged and are not interchangeable. Therefore a topology adapter qualified against one may not be claimed as qualification of the other.

## Current audit findings

The active in-core implementation includes capabilities absent or materially different in the standalone extraction, including:

- `UnifiedConfig` fields such as interpolation bias and Fourier basis configuration;
- Genesis-derived deterministic initialization;
- multiple evolution methods (`evolve`, closed-form, fused/SIMD, iterative variants);
- training/BPTT and mutable parameter analysis surfaces;
- evolution clocks that are semantically relevant when Fourier basis dynamics are enabled;
- explicit `NetworkStateSnapshot` capture/restore used to protect prediction purity;
- broad use through `symthaea_core::hdc::hdc_ltc_unified` by active domain/controller crates.

Conversely, the standalone extraction currently exposes irregular wall-clock timestamp stepping that is not an API of the active in-core network. That capability must not be silently attributed to the active incumbent.

## Core theorem

```text
same type name
    != same implementation lineage
same broad architecture
    != same incumbent
standalone-adapter PASS
    != active-core adapter PASS
archived extraction
    != production/active ownership
```

## Decision

For NEUROARCH evidence, the **primary incumbent is the implementation actually used by active Symthaea consumers**: `symthaea-core::hdc::hdc_ltc_unified`.

The standalone `symthaea-hdc-ltc` crate may remain useful as:

- a compatibility/reference adapter;
- a focused testbed;
- a future migration target only if a separate consolidation program explicitly revives it and migrates consumers.

NEUROARCH must not implicitly make that migration decision.

## Contract ownership

Topology/evidence types should not be owned by either HDC/LTC engine lineage. The preferred seam is a tiny neutral crate such as `symthaea-neuroarch-types` with no dependency on `symthaea-core`, `symthaea-hdc-ltc`, cognitive runtime, or candidate architecture crates.

V1 responsibilities of the neutral seam:

- stable `CircuitId` / `EdgeId`;
- topology descriptors and policy types;
- fail-closed structural validation;
- canonical deterministic encoding;
- versioned topology commitment;
- no live-network adapter logic;
- no runtime state, learned parameters, effectome claims, or cognition policy.

Engine-specific adapters live with or beside their engines and implement the neutral contract.

## Required adapters

### Primary — active core

Adapter for `symthaea_core::hdc::hdc_ltc_unified::HdcLtcUnifiedNetwork` must bind the active implementation's actual structural semantics and protect its actual evolution API.

Parity/purity tests should use its real surfaces, including snapshot-based state purity where appropriate. Do not invent `step_with_timestamp` parity for an API it does not have.

### Secondary — standalone extraction

The existing #3325 adapter may be retained as a separate compatibility/reference adapter after the topology types are extracted. Its fixed-step/irregular-time properties establish facts only about the standalone crate.

## Topology vs model configuration

The active adapter requires an explicit field classification audit. Every `UnifiedConfig` field must be classified as one of:

- structural topology identity;
- subject/model identity;
- learned parameter identity;
- runtime/evolution state;
- measurement-only metadata.

The classification must be frozen before the active adapter commitment is qualified. In particular, Fourier support, evolution method/profile, and dimension/timescale semantics must not disappear merely because the standalone extraction lacks equivalent fields.

## Qualification gates

NEUROARCH-001 cannot close until:

1. neutral topology contract ownership is resolved;
2. primary active-core adapter exists;
3. exact active-core topology field classification is frozen;
4. topology inspection is proven non-mutating using the active implementation's state-purity surface;
5. active-core replay/parity tests cover the execution profile claimed by the receipt;
6. standalone results are labeled with their own implementation lineage;
7. no archived-crate lifecycle change is implied without a separate migration decision.

## Non-goals

- no HDC/LTC migration in this gate;
- no deletion of either implementation;
- no production behavior change;
- no claim that the standalone extraction is lower quality;
- no claim that active-core scientific behavior is already qualified;
- no requirement that future topology candidates depend on `symthaea-core`.

## Evidence boundary

This gate is repository-ownership and experimental-validity infrastructure. It does not establish that any topology improves cognition, efficiency, biological realism, or consciousness-related properties.
