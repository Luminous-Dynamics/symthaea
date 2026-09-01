# Affective Emergence v0.2 — Design Freeze and Implementation-Start Contract

Status: **design-only / blocked on Native Interoception v0.1 qualification**

This contract defines when v0.2 planning is sufficiently complete to stop changing the scientific design and begin implementation. Its purpose is to prevent endless design drift and to make the transition from research planning to code an explicit evidence event.

## Principle

A research design should not remain indefinitely editable while implementation and data generation begin.

Before runtime implementation starts, freeze the scientific/epistemic contracts that determine what will count as a candidate, what information it may use, which scenarios test it, how failures are classified, and how evidence is promoted.

Implementation may reveal a genuine design defect. Fixing such a defect is allowed, but it creates a new design-freeze identity and must occur before confirmatory data are generated under the revised design.

## Proposed DesignFreezeManifest

Create a canonical machine-readable manifest binding at minimum:

- design-freeze schema/version;
- exact v0.1 parent source commit;
- v0.1 model-semantics version;
- current v0.1 qualification dependency state;
- exact v0.2 design branch/source commit;
- top-level observational-affect plan digest;
- information-firewall contract digest;
- temporal-alignment contract digest;
- candidate-definition contract digest;
- scenario/holdout contract digest;
- blinding/unblinding contract digest;
- evidence-root contract digest;
- capability-typed API contract digest;
- adversarial-validation contract digest;
- claim-boundary text/version;
- implementation tranche ordering/version;
- unresolved-design-question list;
- freeze status;
- canonical manifest SHA-256.

## Freeze states

Use explicit states rather than a generic complete/incomplete flag:

- `Draft` — design still actively changing;
- `Reviewable` — all required contracts exist, but unresolved blocking questions remain or review has not completed;
- `FrozenBlockedOnV01` — v0.2 design is frozen, but implementation remains blocked on v0.1 qualification;
- `FrozenImplementationAuthorized` — v0.1 dependency satisfied and implementation may begin;
- `Superseded` — a later design freeze replaces this one;
- `Invalidated` — a discovered internal contradiction makes this freeze unusable.

`FrozenImplementationAuthorized` must never be reachable while v0.1 qualification is unresolved/failed.

## Required design-completeness gates

Before entering `FrozenBlockedOnV01`, all of the following should be true.

### Scientific question gate

The primary v0.2 question is stated narrowly and does not presuppose emotion:

> Do reproducible, prefix-causal, label-free regulatory observables distinguish aspects of regulatory change/forecast revision beyond simpler reactive and nuisance baselines?

No stronger claim is embedded in metric names or success criteria.

### Information gate

The design specifies:

- allowed information through time `t`;
- forbidden future/semantic information;
- online vs retrospective vs oracle classes;
- prefix-equivalence/future-mutation invariants;
- capability-typed API shape preventing accidental escalation.

### Mathematical gate

The design distinguishes at least:

- realized current change (R1);
- one-step forecast residual (R2);
- aligned overlapping-future revision (R3);
- rolling finite-horizon change (R4);
- separate urgency family.

Rolling-horizon turnover is not silently interpreted as forecast revision.

### Candidate-identity gate

Candidate identity prospectively includes formula, sign, temporal indices, information class, forecast policy, horizon, discount, normalization, numeric rules, undefined semantics, source/model lineage, and reference fixtures.

### Scenario gate

Discovery and confirmatory scenario identity are separate and content-hash audited.

Required adversarial scenario families include neutral, nuisance-matched, crossed-sign, exact-prefix/divergent-future, future-mutation, forecast-agreement, and forecast-disagreement cases.

Comparison cut points/windows are prospective.

### Blinding gate

The design specifies:

- separate semantic arm mapping commitment;
- opaque codes;
- primary artifact flow without mapping contents;
- semantic-label canary tests;
- explicit unblinding receipt;
- honest blinding-strength declaration.

### Causal-isolation gate

The no-feedback invariant is explicit and requires native execution equality with and without observatory attachment.

No v0.2 measurement type is designed to become a drive/action/neuromodulator/memory/cognitive command.

### Evidence gate

The prospective root and realized package are distinct.

Every locked scenario is accounted for as included/excluded/indeterminate.

Qualified negative/null result is distinct from integrity failure.

### Adversarial-validation gate

The design has explicit attacks for future leakage, semantic leakage, observer feedback, temporal indexing, artifact substitution, scenario omission, exclusion manipulation, analysis mutation, and known-malicious fixtures.

### Deterministic-inference gate

The design does not manufacture stochastic significance from deterministic grids.

Held-out robustness, worst-case/minimum margins, equivalence bounds, paired baseline comparisons, and explicit failure regions are preferred unless stochasticity is separately introduced/qualified.

### Claim-boundary gate

Even a successful v0.2 remains explicitly insufficient to establish emotion, subjective valence, feeling, sentience, consciousness, or unseen-future prediction.

## Unresolved question policy

A design freeze may contain unresolved questions only if each is classified:

- `ImplementationDetail` — does not change scientific meaning/evidence identity;
- `ExploratoryChoice` — may be chosen using exploratory data, after which confirmatory identity must be newly frozen;
- `ConfirmatoryBlocking` — must be resolved before confirmatory root lock;
- `ArchitectureBlocking` — must be resolved before implementation begins.

Examples:

- exact internal Rust collection type: usually `ImplementationDetail`;
- primary candidate after pilot comparison: `ExploratoryChoice`;
- minimum-effect threshold for confirmation: `ConfirmatoryBlocking`;
- whether online code can see future schedule: already resolved and `ArchitectureBlocking` if reopened.

No unresolved `ArchitectureBlocking` item is permitted at `FrozenImplementationAuthorized`.

## Implementation tranche freeze

Before code begins, lock the initial implementation order so later attractive features do not enter v0.2 opportunistically.

Recommended sequence:

1. crate skeleton and one-way dependency boundary;
2. `ObservationPrefixView` and capability types;
3. prefix-causal forecast policy interfaces;
4. forecast trajectory artifact + v0.1 aggregate-equivalence gate;
5. neutral R1/R2/R3/R4 candidate definitions;
6. typed unavailable/undefined semantics;
7. prefix/future-mutation adversarial tests;
8. no-feedback bisimulation gate;
9. semantic-label canary and mapping separation;
10. scenario/cohort manifests;
11. blinded candidate artifacts/comparison;
12. evidence-root/validation receipts;
13. exploratory study only.

Out of scope for this implementation lineage:

- neuromodulation;
- memory weighting;
- attention modulation;
- action selection;
- policy/control outputs;
- controllability/dominance;
- persistent mood states;
- attachment/social affect;
- learned emotion labels;
- consciousness/sentience inference.

Adding any of these requires a later tranche/design lineage.

## Design-change severity after freeze

Classify changes discovered during implementation:

### Class I — implementation-preserving

Examples:

- refactor with identical reference fixtures/artifacts;
- clearer error message that does not leak semantics/change machine artifact;
- internal allocation/performance improvement with identical qualified outputs.

May preserve design identity when all canonical contracts remain unchanged.

### Class II — candidate/evidence semantic change

Examples:

- formula/sign/horizon change;
- information dependency change;
- temporal alignment change;
- scenario cohort change;
- threshold/comparison rule change;
- new baseline/removal of baseline.

Requires new design/candidate/evidence identity before confirmatory data.

### Class III — architecture-boundary change

Examples:

- allowing feedback to native execution;
- allowing oracle information online;
- adding semantic arm identity to candidate computation;
- adding causal affect outputs.

Invalidates v0.2 observational freeze and requires a new scientific tranche.

## ImplementationStartReceipt

When v0.1 qualification eventually passes, create a small receipt binding:

- DesignFreezeManifest SHA-256;
- exact qualified v0.1 source commit;
- v0.1 qualification receipt/evidence capsule digests;
- v0.2 implementation branch starting SHA;
- implementation tranche version;
- authorization state `FrozenImplementationAuthorized`;
- canonical receipt SHA-256.

This receipt marks the exact transition from design to implementation.

No observational primary data should predate the relevant implementation-start/design identities.

## Confirmatory lock occurs later

Design freeze is **not** the same as confirmatory preregistration.

After implementation and exploratory qualification:

- choose/freeze the primary candidate;
- choose thresholds/equivalence bands;
- freeze confirmatory scenario cohort;
- freeze analysis plan;
- freeze mapping commitment;
- construct the prospective observational evidence root;
- only then generate confirmatory data.

This preserves legitimate exploratory learning while preventing confirmatory retrofitting.

## Review checklist

A reviewer should be able to answer yes to all of these before implementation begins:

1. Can online code be built without access to future experimental schedules?
2. Can primary candidate computation run without semantic arm mapping?
3. Are R1/R2/R3/R4 mathematically distinct and adversarially tested?
4. Is oracle analysis structurally separate?
5. Are candidate and scenario identities immutable/hashable?
6. Can null/negative results survive without being reclassified as integrity failures?
7. Can every confirmatory scenario be accounted for?
8. Can malicious leakage/tampering fixtures be caught?
9. Is the observatory provably read-only relative to v0.1?
10. Are stronger affect/consciousness claims explicitly out of scope?

## Claim boundary

A design freeze demonstrates that the intended experiment and evidence architecture were fixed before implementation/data generation. It does not validate the implementation, qualify v0.1, establish that a candidate succeeds, or support claims of emotion, feeling, sentience, or consciousness.