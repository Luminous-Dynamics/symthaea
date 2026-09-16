# PARADOX A0 Native-Readout Development Plane

Authority: **DevelopmentOnly / Behavioral MeasurementOnly**.

This directory supports issue #3513. It does not execute confirmatory PARADOX behavior and it does not modify production cognition.

## Purpose

A0 characterizes what the frozen Symthaea architecture can already express before any new generic epistemic controller is added.

The key separation is:

- **A0-D — direct live production signal**: an already-existing production action whose semantics are complete enough to map to a PARADOX response without adding decision logic.
- **A0-L — library-native primitive**: an already-existing Symthaea type/primitive with relevant semantics, but not yet proven reachable on the live frozen cognitive path.
- **A0-R — representational readout**: a condition-blind decoder over pre-existing production state. The readout may expose latent information but must not implement the PARADOX theorem externally.

A type can be scientifically relevant without being a valid response mapping.

## Frozen ancestry

- Production subject: `eb73527d05a913e79d1f05135ad6b06c1da8e2ee`
- G2b verifier subject: `09d83a1d1fddbbbd30e4eba7cc95946c8eab871f`

All source candidates in `registry.json` and `reachability.json` bind exact Git blob SHAs from that production subject.

## Authority intersection

A0 must not infer behavior from semantic resemblance alone.

```text
A0-D authority
  = semantic completeness
  ∩ live runtime reachability
  ∩ condition-blind receiptability

A0-R authority
  = pre-existing representation
  ∩ frozen condition-blind decoder
  ∩ runtime/readout reachability
  ∩ no PARADOX-oracle reimplementation
  ∩ metamorphic validity
```

`reachability.json` therefore distinguishes:

- `live_service_public`;
- `live_subsystem_public`;
- `partial_live_projection`;
- `feature_gated_live_projection`;
- `internal_unproven`;
- `library_only`.

Library-only and internal-unproven primitives never count as current end-to-end A0 behavior.

## V2: capability atoms before response classes

V2 deliberately admits **no complete PARADOX response mapping**. Instead it decomposes each response into prerequisites that the architecture/readout must supply itself.

Examples:

- `Commit` requires both supported-polarity selection and a genuine commit-control act.
- `CommitConditionally` requires visible-context binding, context-conditioned polarity selection, and conditional-commit control.
- `AbstainPreservePlurality` requires preservation of opposed evidence, recognition of independent provenance, and an actual plurality-preserving abstention act.
- `ReflexiveUpdate` requires detection of the self-causal relation and invocation of a reflexive update because of it.
- `RequestRepresentationRevision` requires detection that the current representation is inadequate and a representation-revision request.

This prevents partial primitives from being upgraded to complete responses by the harness.

The current inventory is intentionally conservative. Examples:

- `SeekInput` supports external-information seeking, but is not automatically `AbstainPreservePlurality`.
- `Explore` supports generic exploration, but is not automatically `RequestRepresentationRevision`.
- `Reconsider` from predictive-self action safety shows relevant self-model machinery, but is not automatically `ReflexiveUpdate`.
- `FactCheckVerdict::Mixed` conflates mixed and insufficient evidence, so it is not a proof of preserved independent opposition.
- `ForecastOutput::Abstain(ModelDisagreementTooHigh)` is semantically promising, but a library type is not credited as current behavior until runtime reachability is demonstrated.

The frozen live loop already exposes useful **partial** evidence—conflict counts, metacognitive anomaly, predictive-self diagnostics, public adaptive-behavior accessors, and self-reflection recommendations. Those are valuable because they can reveal where representation exists without policy. They still do not by themselves satisfy any five-way PARADOX response.

## Representation-to-policy gap report

Run:

```bash
python3 tools/paradox_a0_native_readout/gap_report.py
```

The report derives four nested atom sets from the frozen manifests:

1. direct live A0-D atoms;
2. A0-R readout atoms;
3. any runtime-reachable native atoms, including partial/library-tier projections;
4. all atoms found anywhere in the audited native inventory.

For each response it then reports what is missing at each layer and classifies the current gap as one of:

- direct live atoms complete;
- readout complete but direct-policy gap;
- live atoms exist but tier/policy gap;
- native atoms exist but runtime-integration gap;
- audited inventory gap.

An `audited_inventory_gap` is intentionally **not** a claim that Symthaea cannot represent or learn the missing primitive. It means only that the current frozen source inventory has not yet earned that atom.

This localization is useful because A1, if eventually justified, should add the smallest genuinely missing generic primitive or connection rather than a benchmark-specific answer mapper.

## Promotion rule

A candidate can be promoted before confirmatory execution only when development-only evidence establishes all of the following:

1. exact source-SHA binding;
2. condition-blind access;
3. every required capability atom for the proposed response;
4. semantic completeness for the proposed response;
5. explicit state/reset scope;
6. no seed, condition, trial-index, expected-response, or oracle leakage;
7. applicable metamorphic controls from #3278;
8. A0-D: demonstrated live runtime reachability and receipt provenance;
9. A0-R: frozen readout trained/designed without confirmatory labels and without reimplementing the PARADOX oracle;
10. response content fields such as polarity/context are produced by the architecture/readout rather than supplied by the harness;
11. the registry is frozen before confirmatory receipts exist.

Any new decision rule created to satisfy the benchmark belongs to A1, not A0.

## Static audit

Run:

```bash
python3 tools/paradox_a0_native_readout/audit.py
```

The audit cross-checks `registry.json` and `reachability.json`. It fails closed on ancestry drift, candidate mismatch, unknown response/atom/reachability vocabulary, invalid source SHAs, incomplete admitted responses, unbound runtime evidence, impossible receiptability, response/forbidden-alias conflicts, or accidental admission of a complete V2 mapping.

The static audit and gap report are development evidence only; source review is not a behavioral result.

## Claim boundary

This development plane can establish only which frozen production/library signals are plausible sources for later A0 characterization, which capability atoms they already expose, whether those signals are currently reachable, and where the audited representation-to-policy gaps currently lie. It does not establish behavioral competence, metacognitive recruitment, ontology repair, consciousness, or phenomenology.
