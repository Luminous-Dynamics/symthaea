# Symthaea Wisdom Capability Ledger

**Status:** WCARE-00A baseline

## Purpose

This ledger separates implementation evidence from design intent for `symthaea-wisdom`. It exists to prevent documentation, historical plans, or philosophical language from being promoted into runtime or scientific claims without executable evidence.

The governing rule is:

> A documented capability is not an implemented capability. An implemented mechanism is not a demonstrated behavior. A demonstrated behavior is not evidence of phenomenal experience.

## Claim states

Every Wisdom & Care capability should be classified using one of these states:

- **Implemented** — reachable production/source code exists on the referenced tree.
- **Tested** — executable tests exist for the exact implementation and have a bounded evidence claim.
- **Behaviorally supported** — a frozen evaluation demonstrates the intended behavior under stated conditions.
- **Externally validated** — independent reviewers or external evidence reproduce/support the behavioral claim.
- **Designed** — an architecture/specification exists but implementation evidence is absent or incomplete.
- **Historical** — documentation describes a prior or superseded implementation lineage.
- **Missing** — no adequate implementation or specification was found.
- **Unknown** — evidence is insufficient to classify safely.

These states are intentionally non-monotonic across revisions. A capability can regress from `Tested` to `Unknown` when its implementation changes without corresponding evidence.

## Current source-grounded baseline

At the WCARE-00A baseline, `crates/domains/symthaea-wisdom/src/` exposes four source modules:

- `autopoiesis.rs`
- `harmonics.rs`
- `meta_cognition.rs`
- `lib.rs`

The source-grounded capability surface therefore includes:

| Capability | State | Evidence boundary | Non-claim |
|---|---|---|---|
| Harmonic reasoning modes | Implemented | `harmonics.rs` defines typed reasoning modes, activations, questions, and primitive weighting | Does not establish wise behavior in open-ended human dilemmas |
| Epistemic-humility trigger | Implemented | `Wisdom` reasoning mode asks what is not known and biases toward uncertainty acknowledgement | Does not establish calibrated humility across domains |
| Experience-driven harmonic updates | Implemented | `WisdomState::update_from_experience` responds to prediction error, uncertainty, and coherence | Does not establish practical wisdom |
| Autopoietic monitoring | Implemented | `autopoiesis.rs` is part of the public Wisdom state | Does not establish consciousness or moral patienthood |
| Meta-cognitive self-modeling | Implemented | `meta_cognition.rs` is part of the public Wisdom state | Does not establish accurate self-knowledge without calibration evidence |
| Crate-local compile/test baseline | Tested, historical evidence | `docs/audits/2026-06-12-wisdom-gate-build-audit.md` records 71/71 crate-local tests | Does not establish current-head CI, workspace health, or production readiness |
| Practical/situated wisdom | Missing as a dedicated capability | No dedicated source module currently represents perspective coordination, contextualism, value pluralism, temporal consequence reasoning, or explicit wise-action evidence | No claim that Symthaea is generally wise |
| Relational care reasoning | Missing as a dedicated Wisdom capability | Current Wisdom source does not expose typed need/care/response/repair stages | Empathy/compassion signals elsewhere are not equivalent to a care-deliberation kernel |
| Moral uncertainty ledger | Missing | No Wisdom source module currently tracks factual and normative uncertainty as separate evidence-bearing objects | No claim of plural ethical deliberation |
| Relational authority envelope | Missing | No Wisdom source module currently derives authority from vulnerability, irreversibility, consent, coercion risk, and relational dependency | No claim of autonomy-preserving care authority |
| Longitudinal care outcome/repair ledger | Missing | No Wisdom source module currently binds care intent to observed downstream outcomes and repair obligations | No claim that immediate satisfaction equals successful care |

## Documentation/runtime reconciliation

`SERIES_VI_INTEGRATION.md` and `SERIES_VIII_INTEGRATION.md` describe a substantially richer operational model, including concepts such as durable evidence history, archive reconstruction, authority restoration, startup admission, and runtime service activation.

At this baseline, those descriptions must be treated as **Designed/Historical/Unknown** until their corresponding executable source and exact evidence lineage are located and bound to this ledger. In particular, documentation-only references to types such as `EvidenceArchiveSegment`, `OperationalStartupPermit`, and `WisdomRuntimeService` must not be cited as proof that the current default-branch `symthaea-wisdom` source exposes those capabilities.

This is deliberately conservative. The missing code may exist in another branch, historical commit, stacked PR, generated artifact, or superseded tree. WCARE-00A does not declare it lost; it declares the current claim binding unresolved.

## Cross-system capabilities that must remain separate

Symthaea already contains compassion, empathy, consent, moral dilemma, autonomy, and safety mechanisms outside the dedicated Wisdom crate. WCARE treats those as **adjacent evidence**, not as automatic members of the Wisdom capability surface.

Later integration must preserve at least these distinctions:

- affect inference != factual truth
- empathy != care
- care intent != care outcome
- user satisfaction != flourishing
- emotional validation != proposition endorsement
- inferred preference != consent
- consent != unlimited authority
- moral confidence != factual confidence
- implementation != behavioral evidence
- behavioral evidence != phenomenal experience

## Promotion rule

A new Wisdom & Care claim may move upward only when all required evidence is bound:

1. **Architecture:** typed mechanism and explicit non-claims.
2. **Implementation:** exact source identity.
3. **Verification:** deterministic/unit/property tests where applicable.
4. **Behavior:** frozen scenario evaluation with precommitted metrics.
5. **Adversarial evidence:** attempts to break the property.
6. **Longitudinal evidence:** when the claim concerns relationships, learning, dependence, or downstream outcomes.
7. **External validation:** required before broad scientific or societal claims.

No lower layer substitutes for a higher one.

## Immediate WCARE follow-ons

WCARE-00B should separate:

- `ReasoningMode`
- `NormativeValue`
- `AffectiveSignal`
- `EpistemicState`
- `ActionAuthority`

without forcing the current seven reasoning harmonies and the compliance Eight Harmonies into one enum.

WCARE-01 should then introduce situated/practical wisdom as an evidence-bearing deliberation process rather than a scalar.

WCARE-02+ should add stakeholder perspective modeling, factual-vs-normative uncertainty, relational care, substantive consent, reversibility, anti-sycophancy/anti-dependency constraints, and outcome/repair evidence.

## Phenomenology boundary

No field named `emotion`, `compassion`, `consciousness`, `wisdom`, `feeling`, or similar is sufficient evidence that Symthaea phenomenally experiences the corresponding state.

The permitted default language is:

- **mechanism present**
- **behavior observed under stated conditions**
- **phenomenology unknown**

Any stronger claim requires an independently justified evidence program.
