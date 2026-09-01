# Affective Emergence v0.2 — Capability-Typed Observatory API

Status: **design-only / blocked on Native Interoception v0.1 qualification**

This contract turns the v0.2 information firewall into an API-shape requirement. The goal is not merely to document that future information, semantic arm identity, or mutable native state are forbidden. Qualified observational code should be unable to receive those capabilities in the first place.

## Principle

For qualified online observational-affect code, forbidden information should be **unrepresentable by the public type surface** whenever practical.

A function that receives a full `StudyExecutionTrace`, full preregistration, semantic arm mapping, or mutable native model and promises not to inspect/mutate forbidden fields is weaker than a function whose parameter types simply do not contain those capabilities.

The preferred v0.2 architecture therefore follows capability security:

> give each computation only the authority and information it requires.

## Proposed module split

After v0.1 qualifies, prefer a separate domain crate:

`symthaea-affect-observatory`

Suggested internal modules:

- `prefix` — restricted online observation views;
- `forecast` — prefix-causal forecast policies;
- `trajectory` — forecast trajectory artifacts;
- `candidate` — interpretation-neutral candidate definitions/computation;
- `blind` — blinded candidate artifacts and comparisons;
- `oracle` — explicit offline oracle diagnostics;
- `retrospective` — diagnostics requiring realized later data;
- `unblinding` — semantic join/evaluation layer;
- `evidence` — manifests/receipts/root validation.

The `oracle`, `retrospective`, and `unblinding` modules must not be dependencies of qualified online candidate modules.

## ObservationPrefixView

A proposed `ObservationPrefixView<'a>` is the only rich input accepted by online candidate/forecast code.

It should contain only information legitimately available through the current cut point:

- opaque study/blind identity permitted at primary-analysis stage;
- immutable native dynamics configuration;
- validated initial native state;
- executed native states/reports through `t`;
- already-observed drives through `t`;
- already-executed intervention receipts through `t`;
- current step/cycle;
- digests binding the source execution prefix.

It must not contain:

- future protocol phases;
- future drives/interventions;
- future realized states;
- post-run exclusion disposition;
- semantic arm ID;
- semantic arm mapping;
- human interpretation labels;
- mutable references to the native model.

Constructing a prefix view should itself validate that all included events have time/index `<= t`.

## Capability marker types

Prefer distinct zero-sized/sealed marker types or wrapper capabilities so temporal information class is encoded statically.

Conceptual examples:

- `OnlinePrefixCapability`;
- `RetrospectiveCapability`;
- `OracleDiagnosticCapability`;
- `UnblindingCapability`.

Qualified online functions accept only `OnlinePrefixCapability` or types constructed from it.

Do not use a runtime boolean such as `allow_future: bool`, `oracle: bool`, or `unblind: bool` inside one universal API. Such flags create easy accidental escalation paths.

## Forecast policy trait boundary

A prefix-causal forecast policy should conceptually resemble:

`forecast(prefix: &ObservationPrefixView, config: &LockedForecastConfig) -> ForecastTrajectoryArtifact`

It must not receive the full experiment schedule.

Qualified policy classes:

- zero-input native recovery;
- current-observed-drive persistence;
- current-state velocity/kinematic baseline;
- later evidence-bearing learned/cued prediction, once separately qualified.

Oracle forecast should use a different type and function namespace, for example:

`oracle::forecast_with_realized_future(...)`

not the online trait.

It should be impossible to insert an oracle implementation into the primary candidate registry without a type/validation failure.

## Candidate function purity

Qualified candidate computation should be a pure transformation of immutable artifacts/configuration.

Requirements:

- no filesystem/network/time/randomness access;
- no mutation of native execution artifacts;
- no global mutable state;
- no semantic arm lookup;
- no logging of hidden semantic context;
- deterministic output for identical canonical inputs;
- explicit typed `Unavailable(reason)` rather than implicit missing/zero values.

Where candidate computation needs history, the required history window must come from the prefix view, never by querying the full trace.

## No-feedback capability rule

The observatory must not expose a public type that can be converted into:

- `InteroceptiveDrive`;
- native intervention command;
- action recommendation consumed by v0.1;
- neuromodulator update;
- memory/attention weight;
- cognitive-loop command.

Candidate result types should contain measurements and provenance only.

If a later causal-affect tranche needs control outputs, that must be a new crate/API/evidence lineage rather than adding a method to v0.2 measurement types.

## Separate semantic namespace

Semantic arm IDs and interpretation labels should live only in the unblinding/evaluation layer.

Primary candidate structures should carry:

- blind code;
- scenario/cut-point identity as permitted by the blinded analysis design;
- candidate ID/digest;
- numeric/availability value;
- provenance digests.

They must not carry a hidden optional `semantic_arm_id` field. An `Option` is still a capability.

## Dependency direction gate

Expected dependency DAG:

`native interoception`

→ `prefix view / forecast trajectory`

→ `candidate computation`

→ `blinded comparison`

→ `unblinding + semantic hypothesis evaluation`

Oracle/retrospective diagnostics are side branches from immutable evidence and never feed back upstream.

A source/dependency audit should reject reverse dependencies such as:

- native interoception importing observatory;
- candidate module importing unblinding;
- online forecast importing oracle;
- blinded comparison importing semantic mapping;
- v0.1 execution depending on any v0.2 result type.

## Construction authority

Some artifacts should have restricted constructors.

Examples:

- `ObservationPrefixView` constructed only from validated execution prefix + cut point;
- `OracleFutureView` constructible only by oracle/diagnostic module;
- `ArmIdentityMapping` created by preparation/custody path, not online analysis;
- `UnblindingCapability` produced only after frozen blinded-artifact digest and mapping commitment validate.

Avoid public struct literals for security/evidence-critical capability objects when that would allow callers to fabricate impossible authority states.

## Serialization boundary

Capability-bearing runtime wrappers should generally not serialize as reusable authority tokens.

Serialize evidence artifacts and commitments, not live privileges.

For example:

- serialize `ObservationPrefixArtifact` if needed;
- reconstruct/validate an in-memory `ObservationPrefixView` from it;
- do not deserialize an `OracleDiagnosticCapability` from arbitrary JSON and thereby grant future access.

## Required compile-/source-structure gates

Where feasible, test architectural constraints mechanically:

1. online candidate APIs do not accept `StudyExecutionTrace` directly;
2. online candidate APIs do not accept `ExperimentPreregistration` directly;
3. online modules do not import `ArmIdentityMapping`;
4. online modules do not import oracle/unblinding modules;
5. candidate outputs have no control/drive conversion implementations;
6. v0.1 crates do not depend on the observatory;
7. semantic emotion-category symbols are absent from online candidate APIs;
8. no `unsafe` is needed for the v0.2 observatory unless separately justified/qualified;
9. no runtime flag can promote online computation to oracle/unblinded authority;
10. semantic-label canary tests remain clean across all primary artifacts/logs.

## Metamorphic authority tests

The eventual implementation should prove:

- replacing the future portion of the full experiment while leaving `ObservationPrefixView` identical leaves online output identical;
- replacing the semantic arm mapping leaves online output identical;
- removing oracle modules from the build does not change qualified online results;
- disabling unblinding support does not change primary artifacts;
- running observatory vs no observatory leaves native v0.1 execution identical.

## Claim boundary

A capability-typed API can make certain classes of information leakage and feedback substantially harder and mechanically detectable. It does not prove that the measured quantity is affect, emotion, subjective valence, sentience, or consciousness.