# Native Interoception v0.1 — Qualification Contract

This file defines the minimum gate for treating the v0.1 substrate as a qualified
baseline for later regulatory-affect experiments. Passing these gates does not
establish any higher-level interpretation; it only qualifies the mechanical
self-regulation and evidence substrate.

## Local crate gates

Run from the repository root in the pinned development environment:

```bash
cargo fmt --all --check
cargo test -p symthaea-interoception
cargo clippy -p symthaea-interoception --all-targets -- -D warnings
```

All commands must exit zero from the same source tree and toolchain environment.

## Mechanical gates

The test suite must demonstrate all of the following:

1. the default state is inside preferred and viable ranges;
2. normalized deviation is zero inside preferred ranges and normalized correctly at lower viability boundaries;
3. zero drive does not manufacture movement inside a preferred range;
4. undriven out-of-band state moves monotonically toward its preferred range;
5. extreme finite drives remain bounded and finite;
6. kinematic and dynamics-aware forecasts remain explicitly distinguishable;
7. dynamics-aware forecasts replay deterministically;
8. direct interventions are recorded separately from endogenous dynamics and reset measured velocity;
9. snapshot, intervention, qualification, capsule, preregistration, execution-trace, and analysis evidence survives serialization round trips where applicable;
10. stable channel identifiers are unique;
11. named higher-level state categories remain absent from core source;
12. passive, restorative, driven, and clamped evidence-plane arms satisfy their declared mechanism expectations;
13. property-based tests preserve boundedness, determinism, and passive-recovery monotonicity across a declared generated region;
14. structural sensitivity monotonicities hold for preferred/viable widths, weights, forecast load, recovery, horizon, discount, and drive persistence;
15. zero aggregate weight cannot erase raw per-channel breach evidence;
16. every exported snapshot, qualification receipt, evidence capsule, and qualification/evidence bundle binds the exact native model-semantics version;
17. deserialization cannot bypass viability/configuration invariants and loaded snapshots reject forged derived reports;
18. preregistration rejects ambiguous arm/metric/hypothesis references, invalid schedules, and any dynamics-aware registered metric whose forecast timestep is incompatible with any arm for which the blinded extractor will compute that metric, including metrics not referenced by a hypothesis;
19. the preregistration digest is stable under round trip, changes when the prospective plan changes, and is separately bound into the evidence capsule;
20. executing the same preregistration twice produces exactly equal traces and hashes;
21. execution limits fail closed instead of silently truncating an arm or protocol;
22. replay validation rejects any trace that diverges from the locked preregistration, and blinded trace exports omit semantic arm identifiers;
23. primary metric extraction validates the execution trace first and emits only blind codes plus preregistered metric identifiers;
24. the blinded metric artifact has a stable digest that is bound into the later hypothesis-evaluation report before semantic arm outcomes are emitted;
25. preregistered minimum-effect relations remain distinct from direction-only relations, preventing arbitrarily tiny differences from satisfying a declared effect-size gate;
26. `QualificationEvidenceBundle` rejects cross-pairing otherwise-valid qualification receipts and evidence capsules from different source commits or model-semantics lineages, and reports qualified only when the bound receipt passes every required gate;
27. a stored `ConfirmatoryHypothesisEvaluation` can be verified by exact recomputation from the locked study, execution, exclusions, and blinded artifact, and semantic-output tampering is rejected;
28. every runnable model and preregistered arm has current state plus preferred/viable geometry inside its declared dynamics numerical domain, while snapshot deserialization rejects domain-incompatible evidence without constructing an invalid model.

## Workspace gates

The repository's ordinary pull-request CI must remain green for the exact PR head.
Showroom Integrity must also pass for that head. A skipped benchmark workflow is not
evidence of benchmark success and must not be reported as such.

Infrastructure inability to schedule these gates is not a pass or a failure. The
qualification remains pending until the required gates execute for the exact source
head. A later source change supersedes any queued qualification run for the older
head.

## Semantic versioning of the research contract

`INTEROCEPTIVE_MODEL_SEMANTICS_VERSION` identifies the scientific meaning of the
native state/regulation/forecast contract independently of the Git commit. A source
refactor that preserves the scientific contract may keep this value stable. Any
change that alters the meaning of viability state, recovery, intervention, or
allostatic forecast behavior must increment it and starts a new experimental
semantics lineage.

`INTEROCEPTIVE_SNAPSHOT_SCHEMA_VERSION` is separate: it changes when the serialized
snapshot representation changes, even when the underlying scientific semantics do
not. Evidence artifacts must bind both versions.

The runnable-model domain checks added during hardening enforce the already-declared
structural numerical domain. They reject inputs that were outside that contract rather
than changing transition behavior for valid-domain states; the native model-semantics
version therefore remains unchanged.

## Machine-readable qualification receipt

`QualificationReceipt` records the exact source commit, model-semantics version, and
one explicit status for each fixed required gate. The v0.1 required gate identifiers
are:

- `local_fmt`
- `local_test`
- `local_clippy`
- `workspace_ci`
- `showroom_integrity`

Each gate is one of `Passed`, `Failed`, `Skipped`, or `Pending`. `is_qualified()` is
true only when the receipt is structurally valid and every required gate is
explicitly `Passed`. `Skipped` never counts as `Passed`. Optional observations such
as `benchmark_suite` may be recorded without altering the required-gate set.

A receipt is still only one half of the promotion boundary: it states which gates
were recorded for one source commit but does not by itself prove that the separately
supplied evidence capsule belongs to that same source lineage.

## Qualification/evidence bundle

`QualificationEvidenceBundle` is the self-contained v0.1 promotion artifact. It
embeds the `QualificationReceipt` and `EvidenceCapsuleManifest` and binds both to one
explicit source commit and model-semantics version.

The bundle must reject:

- a bundle source commit that differs from the qualification receipt;
- a bundle source commit that differs from the evidence capsule;
- a qualification receipt and evidence capsule from different source commits;
- model-semantics mismatches between the bundle and either embedded artifact;
- any structurally invalid embedded receipt or capsule.

`QualificationEvidenceBundle::is_qualified()` is true only when the complete bundle
validates and the embedded qualification receipt explicitly passes every required
v0.1 gate. A bundle with pending gates may remain structurally valid, but it is not
qualified.

The bundle has canonical JSON and its own SHA-256 so a downstream v0.2
implementation-start receipt can bind one exact v0.1 qualification/evidence lineage
rather than relying on independently supplied artifacts.

## Runnable model-domain contract

`InteroceptiveDynamicsConfig::try_validate_state` validates a native state against the
model's declared numerical domain. For each channel, current value and preferred/viable
bounds must lie inside `[min_value, max_value]`.

This does **not** require current state to be inside its viable band: a non-viable state
is a legitimate regulatory condition. It only prevents model geometry or current state
from lying outside the numerical space that the transition law can represent without
clamping.

`NativeInteroceptiveModel::try_new` is the fallible construction path. The infallible
`new` constructor delegates to it and is intended for already-validated internal/test
inputs. Preregistration validates every arm's state/config pair before execution, and
snapshot deserialization records domain mismatch as a validation error rather than
constructing an invalid dynamics-aware model.

## Preregistration contract for later experiments

`ExperimentPreregistration` is the locked prospective plan for an evidence-bearing
experiment. It records:

- exact model-semantics and snapshot-schema versions;
- protocol and analysis version identifiers;
- opaque arm codes for blinded primary analysis;
- each arm's initial native state and dynamics configuration;
- ordered drive phases and scheduled interventions;
- stable registered metric identifiers;
- explicit directional or minimum-effect hypotheses over arm/metric outcome references;
- exclusion criteria declared before results are inspected.

The protocol exposes a deterministic SHA-256 over its validated canonical JSON under
the pinned dependency set. The digest must be captured before result generation.
Changing an arm, drive, intervention, metric, hypothesis, exclusion rule, or analysis
version therefore produces a different preregistration identity.

Because blinded extraction computes the complete registered arm × metric table,
preregistration validation must ensure that every dynamics-aware registered metric
is executable for every arm, even when a metric is not referenced by a hypothesis.
This prevents a prospective protocol from validating successfully and then failing
only after primary metric extraction begins.

Preregistration also validates each arm's initial state against its dynamics numerical
domain so execution cannot begin from an out-of-domain state that would be silently
collapsed by the first clamp.

Preregistration does not prove that the runtime obeyed the plan. The evidence capsule
separately records the preregistration digest and the resolved runtime-configuration
and input-sequence digests so divergence can be detected instead of silently folded
into the planned experiment.

## Deterministic protocol execution

`execute_preregistration` consumes the validated prospective protocol directly. It
does not accept an independently reconstructed arm configuration. Each arm executes
its declared initial state, dynamics configuration, drive phases, and interventions
in protocol order.

Execution is bounded by caller-supplied `ExecutionLimits`. Exceeding either the
per-arm or total-step limit is a hard error; the executor never shortens a run and
reports it as complete.

`ExecutionTrace` records:

- protocol and analysis identity;
- protocol SHA-256;
- resolved native-configuration SHA-256;
- resolved input-sequence SHA-256;
- model- and snapshot-semantics versions;
- opaque arm code and native initial state;
- each executed drive and intervention receipt;
- each mechanical transition receipt;
- state and homeostatic report after every executed step.

Semantic `arm_id` values are deliberately omitted from the trace. A primary analyst
can therefore receive the trace without receiving the arm-identity mapping. The
trace can later be replayed against the locked preregistration; exact mismatch is a
validation failure rather than a warning.

## Blinded analysis boundary

`extract_blinded_metrics` validates the complete trace against the locked protocol
before computing any registered metric. Its output contains blind codes and metric
identifiers, but no semantic arm IDs. The resulting `BlindedMetricReport` is sorted
deterministically and can be hashed and frozen as a primary-analysis artifact.

`evaluate_hypotheses` is a separate unblinding operation. It consumes the locked
protocol and the already-produced blinded metric report, maps blind codes back to
semantic arms, applies the exact preregistered relation, and emits a
`HypothesisEvaluationReport` that includes the blinded-metric SHA-256.

For study-level confirmatory work, the exported qualified path additionally
revalidates the exact execution and exclusion receipt and recomputes the submitted
blinded metric report before semantic evaluation. A fabricated but superficially
well-formed blinded artifact therefore cannot enter the qualified confirmatory path.

`validate_confirmatory_evaluation_bound` provides the symmetric verification path for
a stored/serialized `ConfirmatoryHypothesisEvaluation`: it recomputes the qualified
confirmatory result from the complete locked evidence chain and requires exact equality.
A semantic report that was altered after production therefore fails validation even if
it still deserializes and carries plausible-looking digests.

For confirmatory work, the blinded metric digest should be captured before the
unblinding report is generated. If a primary metric definition or metric value is
changed after unblinding, the digest changes and the original hypothesis report no
longer binds that altered artifact.

`GreaterByAtLeast` and `LessByAtLeast` relations are provided so primary hypotheses
can preregister a minimum practically meaningful difference instead of treating any
nonzero numerical direction as confirmation.

## Evidence capsule

Any result promoted beyond exploratory status should be accompanied by a valid
`EvidenceCapsuleManifest` recording at minimum:

- exact source commit;
- exact native model-semantics version;
- `Cargo.lock` SHA-256;
- `flake.lock` SHA-256 when present;
- `rust-toolchain.toml` SHA-256 when present;
- Rust toolchain identity (`rustc -vV` and `cargo -Vv`);
- target triple and architecture;
- exact experiment identifier;
- locked preregistration SHA-256;
- resolved runtime configuration SHA-256;
- forecast basis;
- input drive/intervention sequence digest;
- snapshot schema version;
- evidence-plane artifact digest;
- raw result artifact hashes, including the blinded execution trace and blinded metric report for confirmatory runs.

The crate validates caller-supplied provenance but does not discover or synthesize
Git state, toolchain identity, or artifact hashes itself.

For v0.1 promotion, independent validity of the evidence capsule is insufficient:
the capsule must be embedded in a valid `QualificationEvidenceBundle` with the
qualification receipt from the same exact source and model-semantics lineage.

A change to source, locked dependencies, toolchain, model semantics, prospective
protocol, executed experimental semantics, or primary analysis definition starts a
new evidence lineage rather than being mixed into an existing one.

## Parameter gate

The defaults documented in `CALIBRATION.md` remain hypothesis-class values. Before
higher-level interpretation, primary qualitative findings must survive sensitivity
analysis over a declared parameter region rather than a single hand-selected point.
The executable sensitivity and property gates in the test suite are minimum
structural checks, not substitutes for a preregistered scientific parameter sweep.

## Stop rule

Do not wire this crate into the cognitive loop or derive higher-level regulatory
state from it until the local crate gates and the required workspace gates pass for
the exact head intended as the v0.1 baseline, and one valid
`QualificationEvidenceBundle` binds the passing qualification receipt and evidence
capsule to that same exact source commit and model-semantics lineage.

A queued, pending, skipped, or infrastructure-blocked workflow does not authorize
promotion. Any source change supersedes older queued gate runs for qualification
purposes and requires qualification against the new exact head.

For the first v0.2 observational experiment, lock and hash the preregistration before
running any primary arm. Exploratory pilot runs must be labeled exploratory and must
not be retroactively promoted into the preregistered confirmatory set. Freeze the
blinded metric digest before generating the semantic-arm hypothesis report.
