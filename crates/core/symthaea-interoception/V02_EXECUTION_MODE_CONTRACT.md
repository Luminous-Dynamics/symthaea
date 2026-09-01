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
3. freeze its exact full-trace digest;
4. construct and freeze the canonical digest of the allowed prefix through cut point
   `t`;
5. only then construct candidate/forecast payloads from the prefix view.

The full-trace digest is **outer provenance only**. It must never be supplied to the
candidate/forecast computation because it is suffix-sensitive: two runs with identical
allowed prefixes but different unseen futures necessarily have different full-trace
hashes.

## 5. Payload vs provenance envelope

Prefix-causality requires two distinct artifact layers.

### `CandidatePayload`

The prefix-causal computational result.

Conceptually binds only information the candidate is allowed to depend on:

- prefix digest;
- cut point / prefix range;
- execution mode class;
- prefix-view schema/version;
- candidate-definition digest;
- forecast-policy digest when applicable;
- output availability/value;
- deterministic implementation identity required to reproduce the computation.

The payload must **not** contain:

- full source-trace digest;
- future schedule/suffix digest;
- semantic arm mapping digest;
- post-run exclusion disposition;
- unblinding identity.

### `CandidateEvidenceEnvelope`

Outer provenance showing where the payload came from.

Conceptually binds:

- exact full source execution-trace digest;
- exact `CandidatePayload` digest;
- study/evidence-root identities;
- toolchain/dependency identities;
- artifact-storage identity/hash as required by the realized package.

The envelope is intentionally allowed to differ between two suffix-divergent source
traces because their full-trace provenance is different.

This prevents a full-trace hash from becoming a covert suffix-information channel while
still preserving complete provenance.

## 6. Prefix-causality under offline replay

Offline access to a full trace in the trusted harness does **not** make a candidate
prefix-causal automatically.

The candidate remains qualified only if:

- its public computation surface cannot receive suffix information;
- changing the unseen suffix while preserving the exact prefix leaves the candidate
  payload unchanged;
- source/dependency audits show no hidden full-trace/oracle access;
- malicious future-reading fixtures are caught by the adversarial suite.

Recommended metamorphic proof:

Given two validated source traces `T_a` and `T_b` whose allowed prefix through `t` is
identical but whose suffix differs, require:

`payload(prefix(T_a, t)) == payload(prefix(T_b, t))`

including:

- identical prefix digest;
- identical availability status;
- identical candidate numeric value;
- identical candidate-definition / forecast-policy identity;
- identical payload digest.

Do **not** require equality of the outer evidence envelope because the full source-trace
digests are expected to differ.

A dedicated regression should prove that changing only the unseen suffix changes the
outer source-provenance digest while leaving the candidate payload byte-identical.

## 7. Prefix digest contract

The prefix digest is itself evidence-critical and must not accidentally hash data after
`t`.

The prefix canonicalization contract should specify exactly which fields are included,
including:

- initial state/configuration;
- executed states/reports through `t`;
- drives observed through `t`;
- intervention receipts executed through `t`;
- cycle/index semantics;
- allowed opaque blind/study identifiers;
- prefix schema/version.

It must exclude:

- later native states;
- later drives/interventions;
- future protocol phases;
- final run disposition;
- semantic mapping;
- later candidate outputs.

Adversarial tests should mutate each forbidden suffix field individually and prove the
prefix digest remains unchanged when the allowed prefix is unchanged.

## 8. Why offline-first is stronger

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

## 9. Online shadow mode remains useful

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
8. offline replay of the resulting trace reproduces the online **candidate payload**
   exactly.

Outer provenance envelopes may differ where execution-mode or source-artifact identity
legitimately differs; the payload-equivalence rule must remain explicit.

Failure means online deployment is not observationally isolated. It does not invalidate
a separately valid offline-prefix result.

## 10. Evidence-root consequences

The prospective `ObservationalEvidenceRootManifest` should bind:

- primary execution mode;
- replay-harness contract/version;
- prefix-view and prefix-digest contract/version;
- candidate payload/envelope schema versions;
- source native trace identity policy;
- online-shadow equivalence requirement if online results are included.

For initial v0.2 confirmatory work, the primary execution mode should be locked to
`OfflinePrefixReplay`.

A later root may include `OnlineShadowObservation`, but it must declare that mode
explicitly and include the online/no-observer equivalence evidence.

## 11. Design-freeze consequence

Execution mode and payload/provenance separation are **ArchitectureBlocking** design
choices because they determine both the causal relationship between measurement and
the native system and whether suffix-sensitive provenance can reach the candidate.

The initial design freeze should therefore resolve:

- first evidence-bearing mode = `OfflinePrefixReplay`;
- live observation = later validation/deployment mode;
- candidate computation receives prefix identity only;
- full source-trace identity remains outer provenance only;
- oracle and retrospective modes remain distinct non-primary namespaces.

Changing the primary confirmatory mode or allowing full-trace provenance into candidate
computation after data generation begins requires a new design/evidence identity.

## 12. Initial implementation order amendment

The initial v0.2 sequence should begin:

1. standalone observatory crate with one-way dependency boundary;
2. validated frozen-trace replay harness;
3. canonical prefix artifact/digest contract;
4. `ObservationPrefixView` construction from immutable trace + cut point;
5. separate `CandidatePayload` and `CandidateEvidenceEnvelope` types;
6. capability/source gates preventing candidate access to full trace/suffix/semantic
   mapping;
7. prefix-causal forecast-policy interfaces;
8. trajectory artifacts + v0.1 aggregate-equivalence gates;
9. neutral candidate families;
10. future-suffix mutation / identical-prefix adversarial tests, including full-trace
    provenance mutation with byte-identical payload;
11. blinded artifacts/comparison and evidence receipts;
12. exploratory `OfflinePrefixReplay` study;
13. only later implement `OnlineShadowObservation` and exact offline/online payload
    equivalence.

No live observational integration should be necessary to answer the first v0.2
scientific question.

## 13. Claim boundary

Offline prefix replay can establish that a deterministic candidate payload is computable
from only the information available through a given cut point and that the payload is
invariant to unseen-future changes under the locked test suite.

It does not establish that the candidate is affect, emotion, subjective valence,
sentience, or consciousness. It also does not establish that a future real-time
implementation is side-effect free until `OnlineShadowObservation` separately passes
its isolation/equivalence gates.
