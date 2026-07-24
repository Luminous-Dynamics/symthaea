# symthaea-legal-reasoning

A small, deterministic, dependency-free legal-reasoning microkernel for
Symthaea.

The crate applies and checks **already-formalized** legal objects. It now
contains three deliberately separate layers:

1. the compatibility calculus for locally stratified string defaults; and
2. a direct typed resolver for already-closed fact sets; and
3. a bounded recursive typed kernel for explicit negation, named rules, superiority,
   direct skeptical conflict resolution, legal time, norm lifecycle,
   Hohfeldian state transitions, proof graphs, validation, and canonical evidence.

It remains pure `std`, forbids unsafe Rust, has zero dependencies, and does not
link to `symthaea-core`.

## Scope boundary

This crate does not:

- interpret natural-language statutes or contracts;
- decide whether evidence is authentic or sufficient;
- decide which precedent is analogous or controlling;
- invent rule priority from source-file order;
- infer the effects of an amendment, judgment, waiver, or legal instrument;
- decide what the law should be;
- provide legal advice.

Those decisions must be formalized in reviewable upstream layers before the
kernel evaluates them.

## v0.4 architecture

### Recursive typed inference

`infer` performs deterministic grounded whole-state iteration. Every round
recomputes derived conclusions from the immutable factual base and the previous
state, allowing newly derived exceptions to retract earlier defaults. Repeated
states, literal bounds, and round bounds fail explicitly.

```rust
use symthaea_legal_reasoning::{
    Atom, FactBase, FormalRule, InferenceProfile, Literal, RuleId, RuleKind,
    RulePack, RulePackId, infer,
};

let positive = |name: &str| Literal::Positive(Atom::new(name).unwrap());
let default = FormalRule::new(
    RuleId::new("bird-default").unwrap(),
    RuleKind::Defeasible,
    [positive("bird")],
    positive("flies"),
).unwrap().with_exceptions([positive("penguin")]).unwrap();
let exception = FormalRule::new(
    RuleId::new("penguin-classification").unwrap(),
    RuleKind::Strict,
    [positive("bird")],
    positive("penguin"),
).unwrap();
let pack = RulePack::new(
    RulePackId::new("birds-v1").unwrap(),
    [default, exception],
    [],
).unwrap();
let facts = FactBase::from_literals([positive("bird")]);
let result = infer(
    &pack,
    &facts,
    &InferenceProfile::grounded_blocking_v1(),
)?;

assert!(result.supports(&positive("penguin")));
assert!(!result.supports(&positive("flies")));
# Ok::<(), symthaea_legal_reasoning::InferenceError>(())
```

`explain_query` reports missing and conflicted premises, active exceptions,
defeats, and defeater-only rules. `ProofGraph::slice_for` returns the transitive
support graph for one conclusion. Applied defaults also preserve every explicit
exception that was checked absent as a canonical proof guard.

`FactBase` retains whether each input was observed, stipulated, or assumed,
along with optional source and asserting-authority provenance.

### Typed rules and explicit priority

`FormalRule` supports strict rules, defeasible rules, and defeaters. `RulePack`
requires unique rule IDs and validates an acyclic `SuperiorityGraph`.

```rust
use std::collections::BTreeSet;
use symthaea_legal_reasoning::{
    Atom, FormalRule, LegalStatus, Literal, PriorityBasis, RuleId, RuleKind,
    RulePack, RulePackId, Superiority, resolve_literal,
};

let positive = |name: &str| Literal::Positive(Atom::new(name).unwrap());
let negative = |name: &str| Literal::Negative(Atom::new(name).unwrap());

let general = FormalRule::new(
    RuleId::new("resident-registration").unwrap(),
    RuleKind::Defeasible,
    [positive("resident")],
    positive("must-register"),
).unwrap();
let exception = FormalRule::new(
    RuleId::new("diplomat-exemption").unwrap(),
    RuleKind::Defeasible,
    [positive("diplomat")],
    negative("must-register"),
).unwrap();
let pack = RulePack::new(
    RulePackId::new("registration-v1").unwrap(),
    [general, exception],
    [Superiority::new(
        RuleId::new("diplomat-exemption").unwrap(),
        RuleId::new("resident-registration").unwrap(),
        PriorityBasis::MoreSpecific,
    )],
).unwrap();
let facts: BTreeSet<_> = [positive("resident"), positive("diplomat")]
    .into_iter()
    .collect();

assert_eq!(
    resolve_literal(&pack, &facts, &positive("must-register")).status,
    LegalStatus::Refuted,
);
```

`resolve_literal` is the direct, non-recursive conflict entry point. It evaluates
applicable rules against a supplied fact set and returns `Supported`, `Refuted`,
`Both`, or `Undetermined` with explicit defeat records. Use `infer` when rule
conclusions must satisfy later premises or activate later exceptions.

### Legal time

`TemporalDimensions` separates:

- enactment time;
- publication time;
- legal effectiveness; and
- applicability to underlying acts or events.

This allows retroactivity to be represented and reviewed instead of silently
collapsing all dates into one interval. Temporal revision overlap is returned as
an error rather than resolved by insertion order. `infer_at` selects one
governing `TemporalRevision<RulePack>` before recursive inference, while
`overlapping_revisions` audits the complete schedule.

### Norm lifecycle

`TimedNorm` and `assess_lifecycle` can report:

- not yet effective;
- active;
- fulfilled;
- fulfilled late;
- exercised;
- violated;
- temporally ambiguous when same-day event ordering is unavailable;
- waived; or
- expired.

A violation may activate an explicitly supplied reparative norm. Event matching
is party-, action-, and beneficiary-sensitive.

When same-day sequence has been formally established, `EventOrder` and
`assess_lifecycle_with_order` resolve action/waiver order explicitly. Event IDs
never act as hidden timestamps.

### Changing factual records

`EvaluationSession` applies fact additions and removals transactionally and
returns deterministic deltas. Failed inference leaves the prior state intact.
The implementation currently recomputes the model. `RuleDependencyIndex`
provides conservative impact closure without claiming a verified incremental
algorithm.

### Hohfeldian transitions

`LegalPositionState` is automatically closed under correlatives. `PowerExercise`
can atomically retract and assert legal positions only when the authorizing
`Power` relation is present. Contradictory post-states fail without mutating the
original state.

### Canonical evidence

`CanonicalEvidence` emits length-prefixed, schema-tagged canonical bytes for:

- legacy derivations and typed literal resolutions;
- provenance-bearing fact bases and grounded inference results;
- query explanations, proof graphs, and proof slices;
- temporal inference results and transactional evaluation deltas;
- norm lifecycle assessments; and
- legal-position transitions.

`EvidenceManifest` binds a payload to an engine version, semantic profile, rule
pack, and query. The crate deliberately does not choose a cryptographic hash or
signature algorithm; callers hash and sign the canonical bytes under their own
approved evidence policy.

### Validation

`validate_rule_pack` produces deterministic advisory findings for missing
provenance, unconditional rules, self-exceptions, inert defeaters, duplicate
rule bodies, and contradictory strict rules. Validation reports never alter an
inference result.

## Compatibility calculus

The original `Rule` API remains available with deterministic locally stratified
semantics. Conclusions used as exceptions must resolve in lower strata. Cycles
through exceptions return `DerivationError`.

```rust
use symthaea_legal_reasoning::{Rule, try_derive};

let rules = vec![
    Rule::new(&["bird"], &["penguin"], "flies"),
    Rule::new(&["bird"], &[], "penguin"),
];
let facts = try_derive(&rules, &["bird"])?;
assert!(facts.contains("penguin"));
assert!(!facts.contains("flies"));
# Ok::<(), symthaea_legal_reasoning::DerivationError>(())
```

See [`SEMANTICS.md`](SEMANTICS.md) for the normative behavioral contract and
[`MIGRATION_0_4.md`](MIGRATION_0_4.md) for adoption guidance.

## Verification

```bash
cargo test -p symthaea-legal-reasoning
cargo clippy -p symthaea-legal-reasoning --all-targets -- -D warnings
cargo doc -p symthaea-legal-reasoning --no-deps
```

## Deliberate next steps

The next major milestones are verified semi-naive incremental evaluation,
minimal counterfactual explanations, richer legal-event identity and evidence
admissibility bindings, and independently curated semantic conformance packs.
