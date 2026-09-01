# Affective Emergence v0.2 — Execution Mode Contract

Status: **design-only / blocked on Native Interoception v0.1 qualification**

This contract strengthens the v0.2 no-feedback boundary by separating the scientific
question of **prefix-causal information use** from the engineering question of
**real-time co-resident observation**.

The first v0.2 evidence-bearing studies should use **offline prefix replay over a
completed, frozen native execution trace**. Live/co-resident observation is a later
engineering validation mode, not a prerequisite for the first observational science.

## 1. Principle

For the first v0.2 study, prefer:

`native execution completes`
→ `exact StudyExecutionTrace is frozen`
→ `observatory replays immutable prefixes`
→ `candidate/forecast artifacts are derived`

instead of:

`native execution || live observatory`

The scientific target is whether a candidate at cut point `t` can be computed using
only information that would have been available through `t`. It does not require that
the candidate was literally computed while the native run was still executing.

Offline replay can therefore preserve the same information restriction while making
observer-to-native feedback impossible for the primary evidence path by construction.

## 2. Execution mode taxonomy

Use explicit execution modes rather than one ambiguous "online" label.

### `OfflinePrefixReplay`

Primary mode for initial v0.2 exploratory and confirmatory evidence.

Properties:

- native v0.1 run is fully complete before observatory computation begins;
- the exact immutable execution trace is hashed/frozen first;
- candidate code receives only prefix-restricted views materialized from that trace;
- no observatory code executes in the native process during evidence generation;
- no candidate output can causally affect the already-completed native trace;
- future suffix information remains forbidden to the candidate despite being present
  somewhere in the trusted replay harness.

### `OnlineShadowObservation`

Later engineering-validation mode.

Properties:

- observatory may execute concurrently/co-resident with native execution;
- outputs remain telemetry-only;
- exact native trace must match a no-observer control under locked inputs;
- shared-resource/timing/global-state side effects become part of the validation threat
  model;
- success does not retroactively change the scientific meaning of offline evidence.

### `OracleDiagnostic`

Offline diagnostic mode with explicit future authority.

Properties:

- may read realized future suffixes or locked future schedules;
- never eligible as a primary prefix-causal candidate;
- uses distinct type/module/evidence identity;
- cannot be substituted into the primary candidate registry.

### `RetrospectiveDiagnostic`

Offline diagnostic using later realized data without oracle claims.

Used for calibration/error decomposition and other explicitly retrospective analyses.
It is not a prefix-causal primary candidate.

## 3. Trusted replay harness vs candidate authority

The replay harness may possess the complete frozen trace because it is responsible for
constructing restricted views. Candidate and forecast code must not.

Required separation:

- `ReplayHarness` validates the exact frozen trace and requested cut point;
- it constructs `ObservationPrefixView(t)` containing only allowed prefix information;
- candidate/forecast functions accept the prefix view, never the complete trace;
- the harness may compare outputs across suffix-mutated twin traces, but each candidate
  call still receives only its identical prefix;
- semantic arm mapping remains unavailable to primary candidate code.

The trusted harness is part of the evidence-critical implementation and must itself be
versioned, tested, and bound by the design/evidence root.

## 4. Frozen-trace prerequisite

Before any primary candidate is computed in `OfflinePrefixReplay`:

1. validate the native `StudyExecutionTrace` against its locked study;
2. serialize/canonicalize under the declared evidence contract;
3. freeze its exact digest;
4. record that digest as the immutable source of all replay prefixes;
5. only then construct candidate/forecast artifacts.

A candidate artifact must bind:

- source execution-trace digest;
- cut point / prefix range;
- execution mode = `OfflinePrefixReplay`;
- prefix-view schema/version;
- candidate-definition digest;
- forecast-policy digest when applicable;
- output availability/value;
- implementation/toolchain identity required by the v0.2 evidence root.

## 5. Prefix-causality under offline replay

Offline access to a full trace in the trusted harness does **not** make a candidate
prefix-causal automatically.

The candidate remains qualified only if:

- its public computation surface cannot receive suffix information;
- changing the unseen suffix while preserving the exact prefix leaves the candidate
  output unchanged;
- source/dependency audits show no hidden full-trace/oracle access;
- malicious future-reading fixtures are caught by the adversarial suite.

Recommended metamorphic proof:

Given two validated source traces `T_a` and `T_b` whose allowed prefix through `t` is
identical but whose suffix differs, require:

`candidate(prefix(T_a, t)) == candidate(prefix(T_b, t))`

including availability status and provenance-relevant deterministic output.

## 6. Why offline-first is stronger

A live observer can theoretically perturb execution without ever converting a
candidate value into a native drive. Possible channels include:

- shared mutable globals;
- RNG/global registry consumption;
- allocator/resource pressure;
- thread scheduling;
- logging hooks;
- filesystem/network side effects;
- timing-sensitive code;
- shared caches or metrics registries.

The existing v0.2 purity/no-feedback tests help detect these effects, but
`OfflinePrefixReplay` removes the entire observer→native causal path from the primary
study.

This is stronger than merely observing no numerical difference in one bisimulation
suite.

## 7. Online shadow mode remains useful

Offline-first does not eliminate the value of later real-time observation.

Before any production/live observatory is considered equivalent to offline evidence,
require `OnlineShadowObservation` validation:

1. same locked study/inputs with and without observer;
2. exact equality of complete native execution traces;
3. exact equality across candidate-order permutations;
4. repeated clean-process trials under the same deterministic contract;
5. no use of wall-clock time/randomness/network/filesystem in candidate code;
6. no shared mutable state between candidate evaluations;
7. resource-stress tests where practical;
8. offline replay of the resulting trace reproduces the online candidate artifact
   exactly.

Failure means online deployment is not observationally isolated. It does not invalidate
a separately valid offline-prefix result.

## 8. Evidence-root consequences

The prospective `ObservationalEvidenceRootManifest` should bind:

- primary execution mode;
- replay-harness contract/version;
- prefix-view contract/version;
- source native trace identity policy;
- online-shadow equivalence requirement if online results are included.

For initial v0.2 confirmatory work, the primary execution mode should be locked to
`OfflinePrefixReplay`.

A later root may include `OnlineShadowObservation`, but it must declare that mode
explicitly and include the online/no-observer equivalence evidence.

## 9. Design-freeze consequence

Execution mode is an **ArchitectureBlocking** design choice because it changes the
causal relationship between measurement and the native system.

The initial design freeze should therefore resolve:

- first evidence-bearing mode = `OfflinePrefixReplay`;
- live observation = later validation/deployment mode;
- oracle and retrospective modes remain distinct non-primary namespaces.

Changing the primary confirmatory mode after data generation begins requires a new
design/evidence identity.

## 10. Initial implementation order amendment

The initial v0.2 sequence should begin:

1. standalone observatory crate with one-way dependency boundary;
2. validated frozen-trace replay harness;
3. `ObservationPrefixView` construction from immutable trace + cut point;
4. capability/source gates preventing candidate access to full trace/suffix/semantic
   mapping;
5. prefix-causal forecast-policy interfaces;
6. trajectory artifacts + v0.1 aggregate-equivalence gates;
7. neutral candidate families;
8. future-suffix mutation / identical-prefix adversarial tests;
9. blinded artifacts/comparison and evidence receipts;
10. exploratory `OfflinePrefixReplay` study;
11. only later implement `OnlineShadowObservation` and exact offline/online equivalence.

No live observational integration should be necessary to answer the first v0.2
scientific question.

## 11. Claim boundary

Offline prefix replay can establish that a deterministic candidate is computable from
only the information available through a given cut point and that its result is
invariant to unseen-future changes under the locked test suite.

It does not establish that the candidate is affect, emotion, subjective valence,
sentience, or consciousness. It also does not establish that a future real-time
implementation is side-effect free until `OnlineShadowObservation` separately passes
its isolation/equivalence gates.
