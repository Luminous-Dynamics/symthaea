# NEUROARCH-001 — Experimental topology contract

Status: executable draft; qualification pending for issue #3317 and ownership gate #3519  
Parent program: #3316

## Purpose

NEUROARCH-001 introduces a neutral mesoscale description of Symthaea cognitive topology without changing any engine's production evolution path. The topology layer answers **what circuits and declared routes exist and which static execution semantics they carry**. It does not establish causal influence, biological realism, architectural benefit, or implementation identity by itself.

The primary incumbent for Phase-I evidence is the HDC/LTC implementation actually used by active Symthaea consumers:

```text
symthaea-core::hdc::hdc_ltc_unified::HdcLtcUnifiedNetwork
```

The historical standalone `symthaea-hdc-ltc` extraction is explicitly archived in the crate-truth registry and is outside the primary qualification surface unless a separate migration/compatibility effort revives it.

## Ownership

Canonical topology types live in the engine-independent `symthaea-neuroarch-types` seam.

The active engine is observed through a read-only wrapper in `symthaea-neuroarch-adapters`. The adapter crate may depend on `symthaea-core`; `symthaea-core` does not depend on NEUROARCH research tooling merely to be inspected.

`CircuitImplementation` has only one schema-level special case: `ExternalInput`. Computational implementations use bounded canonical `Named(...)` tokens. The neutral types crate therefore does not require a new enum variant for every engine lineage. The active incumbent adapter currently declares `active_core:hdc_ltc_layer`.

## Separation of identities

At minimum these identities remain distinct:

1. **Topology / connectome identity** — declared static circuits, routes, merge semantics, and structural timescale metadata.
2. **Subject/model/checkpoint identity** — exact implementation lineage, initialization profile, model configuration, and parameter/checkpoint identity.
3. **Trial/execution identity** — task fixture, ordered inputs, time-step stream, selected evolution method, build/toolchain/hardware profile, and run policy.
4. **Runtime state** — mutable HDC/neuron state, cached outputs, evolution clocks, observations, and temporary snapshots.
5. **Effectome** — causal influence measured by controlled perturbation.
6. **Epistemome/evidence** — confidence, provenance, applicability, uncertainty, and evidence ancestry.

A `TopologyCommitment` is never a substitute for any other identity.

## Canonical identifiers

V1 exposes stable structural identifiers equivalent to:

```rust
pub struct CircuitId(pub u32);
pub struct EdgeId(pub u64);
```

Construction order, collection iteration order, thread scheduling, runtime activity, and initialization seed do not alter structural identity.

## Circuit descriptor

Minimum static semantics:

```text
CircuitDescriptor
  id
  role                 // descriptive symbolic label only
  timescale_class
  state_dimension      // logical dimensions per implementation unit
  unit_count           // implementation-unit multiplicity
  implementation
  input_merge_policy   // None | Single | BundleAll
  modulation_profile?  // declarative capability/profile only
```

`role` is metadata, not verified function. Runtime systems may not treat a label such as `causal`, `self`, or `planning` as evidence that a circuit actually performs that function.

Resource accounting derives:

```text
total_logical_state_dimensions = checked(state_dimension * unit_count)
```

The multiplication is performed exactly once. Overflow fails closed with `DimensionOverflow`. External-input descriptors are structural interface metadata and are not automatically counted as persistent computational state.

## Edge descriptor

Minimum static semantics:

```text
EdgeDescriptor
  id
  source
  target
  channel
  direction
  recurrence
  budget_class
  transform            // route-local Direct | Bind in V1
```

A structural edge means only that influence is permitted by the architecture. It is not evidence that the source actually causes a downstream effect.

## Route transform vs target merge

Route-local transformation and target-level multi-input merging are separate facts.

For the active HDC/LTC incumbent:

- inter-layer routes are `Direct` or `Bind` according to `use_layer_binding`;
- skip routes are explicit `Direct` routes from external input to deeper layers;
- a deeper target with skip connectivity declares `input_merge_policy = BundleAll`.

This prevents skip semantics from being double encoded and gives later resource accounting one unambiguous graph.

## Symbolic-token grammar

Identity-bearing V1 symbolic fields are case-sensitive bounded ASCII tokens.

Allowed characters:

```text
A-Z a-z 0-9 _ - . : /
```

Maximum length: 128 UTF-8 bytes.

Whitespace, Unicode lookalikes, empty names, and over-length fields fail closed. V1 does not trim, case-fold, or Unicode-normalize identity-bearing tokens. Changing this grammar requires an explicit schema/version change.

## Validation rules

Topology construction fails closed on at least:

- unsupported topology schema version;
- no computational circuits;
- duplicate circuit IDs;
- duplicate edge IDs;
- dangling source or target IDs;
- invalid state dimensions or zero implementation-unit counts;
- logical-state multiplication overflow;
- invalid timescale declarations;
- forbidden non-recurrent self-edges;
- ambiguous duplicate structural routes;
- duplicate source/target/channel routes;
- malformed symbolic identifiers.

Parallel channels are permitted only when explicitly enabled and semantically distinct.

## Bidirectional canonicalization

A bidirectional route has no topology-significant endpoint orientation. Therefore V1 normalizes its endpoints before duplicate detection, sorting, and encoding:

```text
A <-> B
```

and

```text
B <-> A
```

produce the same canonical representation when all other static identity fields, including edge identity, are equal. Directed routes remain orientation-sensitive.

## Canonical commitment

Canonical form sorts circuits by stable circuit identity and edges by canonical direction/endpoints/channel/edge identity. Canonical bytes bind all declared behaviorally relevant static topology fields and contain no runtime state.

V1 uses a domain-separated BLAKE3 commitment.

Changing structural properties such as these changes the commitment:

- circuit count or identity;
- unit count or per-unit state dimension;
- role/timescale/implementation/merge-policy declarations;
- route identity/source/target/channel/direction/recurrence;
- route budget class;
- route-local transform;
- skip-connectivity structure.

Changing current runtime state, observations, cached outputs, learned weights, realized initialization vectors, or random seed does not change structural topology identity.

An external fixed V1 golden byte vector protects the encoder against accidental drift where all implementations would otherwise change together.

## Active field classification

`NEUROARCH_001_ACTIVE_FIELD_CLASSIFICATION.md` freezes the V1 classification before qualification.

Topology V1 binds the active incumbent's:

- `UnifiedNetworkConfig.layer_sizes` as circuit/unit multiplicity;
- `UnifiedConfig.dimension` as per-unit logical state dimension;
- `UnifiedConfig.tau_base` as a nearest-nanosecond structural timescale declaration;
- `UnifiedNetworkConfig.use_layer_binding`;
- `UnifiedNetworkConfig.skip_connections`.

The exact floating-point `tau_base` representation remains part of subject/config identity. Two models may therefore share a topology commitment while having distinct exact temporal parameters below V1's nanosecond structural resolution.

Other behaviorally important fields—activation, interpolation/gating parameters, Fourier configuration, optimizer/learning parameters, initialization lineage, and realized parameter values—are **not ignored**. They are bound by exact subject/model/checkpoint identity rather than topology V1.

Changing that classification after held-out evidence begins requires a new schema or evidence lineage.

## Active incumbent adapter

The active adapter wraps:

```text
symthaea-core::hdc::hdc_ltc_unified::HdcLtcUnifiedNetwork
```

It is observational. It must not require changes to active-core constructors, evolution methods, training methods, snapshots, or consumers.

The active implementation exposes several evolution profiles, including Euler, ordinary closed-form, fused/SIMD closed-form, and other variants. **Execution method is trial identity, not inferred from topology.** A parity claim applies only to the exact execution profile exercised.

The active implementation also exposes `NetworkStateSnapshot`; evolution clocks are part of genuine mutable state when Fourier dynamics are enabled. Inspection-purity testing must therefore protect snapshots/clocks, not merely final output.

## Replay and purity gate

NEUROARCH-001 is not qualified until topology inspection of the active incumbent is shown not to perturb its evolution state or output under the exercised profile.

The initial active-core profile is ordinary `evolve_closed_form` with fixed explicit `dt` values. Required evidence includes:

- snapshot before inspection equals snapshot after inspection;
- an observed network and otherwise identical unobserved control produce the same output after each exercised closed-form step;
- same static topology with different initialization seed has the same topology commitment but a different subject identity in the later benchmark receipt;
- integer-seed and Genesis-derived initialization lineages share topology identity when their static graph/config declaration is otherwise equal;
- structural mutations such as layer size, binding, skip connectivity, or structural timescale change topology commitment;
- subject-only mutations such as activation, learning/optimizer, gating/interpolation, backbone-tau, and Fourier settings remain topology-stable.

### No imported irregular-time claim

The active in-core network does not expose the standalone extraction's `step_with_timestamp` API. Therefore NEUROARCH-001 makes **no active-incumbent irregular-wall-clock parity claim**.

If active core later gains a timestamp-derived stepping profile, it requires its own implementation and qualification evidence. Historical standalone behavior cannot establish it.

## Initial test matrix

Required neutral-contract evidence:

1. insertion-order-independent canonical bytes/commitment;
2. fixed external V1 golden byte vector;
3. every declared V1 circuit/edge identity field, including canonical IDs, is commitment-bound;
4. dangling-edge rejection;
5. duplicate-ID rejection;
6. forbidden self-edge rejection;
7. explicit parallel-channel policy;
8. bidirectional orientation canonicalization;
9. bounded canonical symbolic-token enforcement;
10. checked logical-state total and overflow rejection.

Required active-adapter evidence:

11. active topology commitment is integer-seed independent;
12. active topology commitment is Genesis-vs-integer initialization-lineage independent;
13. active per-unit state accounting is explicit and checked;
14. active layer-size/timescale/binding/skip structural changes alter commitment;
15. active non-topological model/config mutations remain topology-stable;
16. active skip connectivity is represented once through route + merge semantics;
17. topology inspection preserves exact active `NetworkStateSnapshot` under zero tolerance where deterministic;
18. active ordinary closed-form replay parity holds with inspection interleaved.

Additional execution profiles are separate claims, not automatically inherited from item 18.

## Exact-head qualification protocol

The focused `.github/workflows/neuroarch-001-qualify.yml` lane is part of the evidence protocol, not merely CI convenience.

For bootstrap qualification it must:

- check out the exact PR head SHA rather than GitHub's synthetic merge ref;
- prove that the head is exactly one semantic commit over frozen base `4bad8af72ff775e7c869b6df83faba718a339a36`;
- reject files outside the preregistered NEUROARCH-001 scope;
- reject whitespace-error diffs;
- prove Rust and Cargo release `1.96.0`;
- prove both new crates are workspace members;
- perform a full workspace metadata resolution to materialize the lock transition;
- reject any removed lock content and require exactly the two expected new workspace package blocks;
- switch immediately to `--locked` for compilation, dependency-tree inspection, tests, and clippy;
- bind the generated lock with SHA-256 plus qualification-head/base metadata;
- upload that lock evidence once;
- require the active-adapter job to consume the exact same checksum-verified lock artifact;
- run the active job only after the neutral contract job succeeds.

The bootstrap lock artifact is evidence, not a mergeable repository state. After inspection, that exact lock must be committed, the complete tree must again be squashed to one semantic commit over the frozen base, and the qualifier must be converted to final no-mutation mode. Final qualification requires `Cargo.lock` to remain byte-for-byte unchanged under Cargo 1.96 resolution.

Queued, skipped, no-step, timeout, runner-mutated-only, or synthetic-merge-ref states are not PASS evidence.

## Lock/build qualification

The exact qualification lineage binds:

- Git head;
- frozen base;
- one-commit scope;
- generated and then committed `Cargo.lock` appropriate to that exact tree;
- Rust/Cargo 1.96.0 toolchain;
- target/build profile;
- executed format/check/test/clippy commands and results;
- lock artifact checksum and dependency-tree evidence.

Adding workspace crates without regenerating and committing the lock is not a complete reproducibility capsule even if an unconstrained Cargo invocation can repair the lock transiently. The PR remains unqualified until the final committed-lock lineage passes without mutation.

## Measurement-authority boundary

Existing `symthaea-consciousness-topology` persistent-homology/TDA outputs and legacy `symthaea-phi-search` Φ-like outputs may be retained as secondary measurements only. They neither define static topology identity nor have authority to promote a NEUROARCH candidate.

A consciousness-related proxy must not be used as both the architecture-selection objective and the independent evidence that the selected architecture is superior.

## Evidence boundary

Passing NEUROARCH-001 would establish only that Symthaea can describe, validate, and commit to the active incumbent's static topology reproducibly while observational inspection preserves the exercised active-core dynamics.

It would **not** establish that:

- a topology is biologically realistic;
- a declared edge is causally important;
- modular/rich-club organization is beneficial;
- a topology improves cognition or efficiency;
- the architecture has consciousness-related significance;
- every active-core execution method has been qualified.

Those questions belong to later preregistered experiments.

## Follow-on dependency

NEUROARCH-002 (#3318) may build deterministic benchmark receipts on this commitment only after NEUROARCH-001 qualification. Later candidates must bind both topology identity and exact implementation/subject/config/trial identity and may not bypass this contract.
