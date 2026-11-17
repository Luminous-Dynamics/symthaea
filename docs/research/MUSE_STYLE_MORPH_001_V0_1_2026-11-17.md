# MUSE-STYLE-MORPH-001 — Time-varying style trajectories with musical continuity

**Version:** 0.1  
**Date:** 2026-11-17  
**Status:** research and implementation contract proposal; no runtime capability is claimed  
**Parent:** Symthaea #12643

## 1. Objective

Make a composition's style a time-indexed, verifiable trajectory rather than a single global label or static recipe. A piece should be able to move between distinct compositional grammars while retaining explicitly selected musical identities, preserving protected material, and satisfying each grammar's own obligations.

Example trajectories:

- a formal opening develops into a cyclic groove, then returns to transformed opening material;
- a free-time modal exposition establishes a subject before a pulse-locked ensemble enters;
- a chamber texture develops into a chorus-based improvisatory dialogue while retaining a named theme;
- a sparse additive process grows into an interlocking ensemble and releases back to its original ground.

A hybrid/trajectory is a new composition recipe. It is not a claim that the resulting work is historically authentic to any contributing tradition.

## 2. Why this is a distinct work package

Existing work already provides important adjacent foundations:

- #7417 proposes multidimensional profiles, provenance, compatibility rules and static hybrid grammar.
- #9505 proposes deterministic compilation of layered hybrid recipes.
- #9497 owns the 152-style source-backed completion programme.
- #9397 owns renderer-independent style identity and instrument idiom qualification.
- #9581 owns score-first structural/control/robustness evaluation.
- #11284 models temporal relationships among ensemble parts.
- #11355 separates shared ensemble time from part-local expressive performance.
- #11332 and #12367 own score-native editing and grounded instruction-to-edit planning.

The gap addressed here is the *temporal composition of style states*: which grammar owns each score interval, how the transition is represented, what identity must persist, and how the resulting score proves it changed as requested. This issue does not replace the adjacent owners or duplicate their validators.

## 3. Core model: style as a trajectory of sparse dimensions

A style state is a sparse tuple of independently owned dimensions:

1. structural grammar;
2. rhythmic language;
3. pitch and tuning language;
4. harmony;
5. melody and motif behaviour;
6. texture and counterpoint;
7. ensemble and instrument idiom;
8. performance dialect;
9. production environment.

Not every style has a meaningful value for every dimension. Unknown, unsupported and unspecified remain distinct from zero/default. Inherited values retain their source reference. A dimension may change at a different time from another dimension.

The 19 catalog constellations remain useful navigation groups. They do not imply a total tree of historical lineage or musical compatibility.

## 4. Proposed versioned contract

The first implementation should define a canonical `StyleTrajectoryPlanV1`, reusing the current `symthaea_music_theory::Duration` (exact rational beat coordinates), `Score`, `PartId`, `GrammarPlanEvidence`, and `GrammarStructuralTrace` where their semantics fit. Do not create a competing score clock or a second grammar-family enum.

The current `ScoreEventRef` is a pair of `VoiceRole` and role-local index, not a globally stable event ID. The Studio's current `SymbolicNoteEvent.id` is also generated as `score-{role}-{role_index}`; it must be treated as score-scoped, not as persistent identity. Every such locator must be bound to the exact canonical score digest. Any transformation that changes event ordering must emit an explicit source-to-result event correspondence; never assume a role/index locator survives an edit unchanged.

The symbolic `ScoreNote` already carries `PartId`, but the current public `SymbolicNoteEvent` projection does not. Existing draft [#11351](https://github.com/Luminous-Dynamics/symthaea/pull/11351) owns preserving PartId through Listen/performance provenance. MUSE-STYLE-MORPH should consume that versioned, fail-closed projection after integration rather than create a parallel PartId mechanism. A role containing multiple or unassigned source parts must remain ambiguous until the renderer can prove an exact event-to-part mapping.

The existing protocol `MusicalTime` is a display/transport projection containing floating-point beats/seconds plus ticks. Use exact `Duration` arithmetic inside the symbolic plan/verifier; do not round exact score-time comparisons through floating-point `MusicalTime` fields.

Conceptually:

```text
StyleTrajectoryPlanV1 {
  schema_version,
  source_score_digest,
  score_time_schema,
  ordered_segments: [
    {
      segment_id,
      start_beat,             // exact rational beat
      end_beat,               // exact rational beat; half-open interval
      structural_owner_ref,
      dimension_sources,
      required_grammar_obligations,
      protected_material_refs,
      capability_evidence_refs,
    }
  ],
  transitions: [
    {
      transition_id,
      from_segment_id,
      to_segment_id,
      mode,
      start_beat,
      end_beat,
      dimension_operations,
      continuity_anchors,
      bridge_obligations,
      protected_material_refs,
    }
  ],
  motif_lineage_refs,
  part_identity_refs,
  engine_revision,
  deterministic_seed,
  disposition,
  canonical_digest,
}
```

This is a schema sketch, not a committed Rust type. The implementation must use existing canonical time, score, profile, digest and provenance owners rather than create parallel representations.

### 4.1 Segment semantics

- Every segment uses an exact half-open score-time interval `[start_beat, end_beat)`.
- Segments form a complete partition of the requested composition interval unless an explicit uncomposed/unknown interval is represented.
- Exactly one structural grammar owns each instant. Multiple structural owners are only legal if a future, explicitly qualified polymetric/overlay grammar defines the semantics.
- Each dimension declares one disposition: `inherit`, `replace`, `transform`, `overlay`, `unspecified`, or `unsupported`.
- A segment binds exact profile IDs/revisions/digests and the evidence applicable to the requested capabilities.
- User overrides are explicit; they cannot silently widen a profile's qualified capability envelope.

### 4.2 Transition modes

The initial vocabulary is deliberately small:

- **HardBoundary:** a deliberate change at an exact structural boundary.
- **ComposedBridge:** an explicit interval with its own score events and obligations.
- **GradualMorph:** a named set of dimensions changes across a declared window.
- **EnsembleOverlay:** two or more parts/layers interact during a bounded overlap while structural ownership stays unambiguous.

Do not represent every transition as a numeric crossfade. Tuning systems, meter, cycle phase, instrument technique and discrete grammar operations often have no meaningful interpolation. A bridge must be composed and validated; it is not a period in which grammar obligations are suspended. Audio crossfade belongs to the renderer and must not be mistaken for symbolic continuity.

### 4.3 Continuity anchors

A transition may preserve, transform, or deliberately release named musical identity:

- motif/subject/hook identity and allowed transformation lineage;
- persistent `PartId` and temporal part relationships;
- beat/pulse/cycle phase or an explicitly declared phase reset;
- pitch-centre, tuning or voice-leading relation where a valid mapping exists;
- formal return/cadence obligations;
- selected texture, instrument or performance attributes.

Continuity is not the same as sameness. A subject may be augmented, fragmented, sequenced, inverted or answered only when the source/target grammar supports that operation and the operation is declared in the plan.

## 5. Independent verification

The verifier must derive its result from the canonical plan and realized Score/trace. It must not accept a generator-authored boolean saying that a transition was applied.

At minimum, reject:

1. gaps, overlaps, non-monotonic intervals or ambiguous segment ownership;
2. an unknown, stale or mismatched source score/profile digest;
3. a structural owner that is not an executable, explicitly supported grammar;
4. an unsupported dimension operation or capability silently mapped to a generic default;
5. events outside the segment/transition windows permitted by the plan;
6. missing, duplicated or changed `PartId` / motif references;
7. broken motif lineage or a transformation that was not requested/permitted;
8. alterations to protected events or regions;
9. an invalid pulse, meter, tuning, register, articulation or instrument mapping;
10. bridge material that violates its own obligations or leaves either adjacent grammar invalid;
11. a plan with disposition `needs_clarification`, `abstain` or `invalid` that still contains executable operations.

Every accepted transition should produce a receipt binding the source score, resolved plan, relevant profile versions, realized event range, verifier revision and exact outcome. Failure, abstention and unsupported capability are first-class results.

## 6. Benchmark protocol

Begin with deterministic symbolic known-answer fixtures. Human listening and cultural review are separate external evidence; neither replaces exact mechanics.

### 6.1 Panels

| Panel | Positive cases | Negative/adversarial controls |
|---|---|---|
| Partition | Adjacent half-open intervals, explicit bridge and overlay | Gaps, overlap, duplicate ownership, zero-length segment |
| Transition semantics | Hard boundary, bridge, one-axis morph | Unmapped tuning change, unsupported meter change, undeclared crossfade |
| Identity continuity | Exact motif retention and allowed transformation | Missing subject, false lineage, altered protected notes |
| Part continuity | Persistent part through instrument handoff | New/duplicate PartId, note spanning unsupported handoff |
| Grammar obligations | Both grammars and bridge satisfy their contracts | Invalid cadence, broken cycle, impossible section/return |
| Causal control | One declared dimension changes under same seed | Non-target dimensions change; requested dimension remains inert |
| Evidence integrity | Exact profile/source revisions replay | Stale profile, changed source score, corrupted receipt |
| Long-form resilience | Repeated sections and late transitions | Transition after return, tempo warp, dense polyphonic overlap |

### 6.2 Metrics — never collapse into one score

- Exact plan match and transition mode accuracy.
- Boundary error in beats and milliseconds.
- Structural-obligation pass rate, reported separately for source, bridge and target.
- Motif/subject continuity precision, recall and exact transformation match.
- Persistent part/relation precision and recall.
- Intended-axis adherence and non-target disturbance under paired counterfactuals.
- Pulse/cycle phase error, event collisions and density discontinuity where the relevant grammar makes those measurements meaningful.
- Unsupported/ambiguous detection recall, false-abstention rate and risk/coverage curve.
- Within-trajectory diversity and between-trajectory identity/separability.
- p50/p95 planning and rendering latency, CPU/GPU/memory and cost per verified transition.
- Separate blind listener preference and musician-reviewed transition coherence, with sample sizes and confidence intervals.

Do not compare all symbolic metrics with audio-only systems as if their inputs and control surfaces were identical. Run matched task tracks and disclose modalities, candidate budgets, duration, seeds, and tool access.

### 6.3 Dataset construction and leakage controls

- Use original synthetic fixtures for exact known answers and mutation controls.
- Use rights-cleared or participant-authored scores for external listening and musician review.
- Split by complete works, composers, performers, editions, source lineages, periods and regions as applicable; never random bars from the same work across train/test.
- Keep culturally specific claims under musician/community governance; do not infer tradition authenticity from feature similarity.
- Preserve failed, unparseable, unsupported and abstained requests in all denominators.
- Preregister confirmatory cases separately from development cases and publish per-example receipts.

## 7. Prospective release gates

These are targets, not current results:

- 100% exact partition and single-owner checks on deterministic fixtures.
- Zero executable plans for unsupported, stale-evidence or ambiguous required transitions.
- 100% protected-event preservation on deterministic controls.
- 100% exact part/motif reference accounting on accepted fixtures.
- At least 95% exact transition-plan adherence on a preregistered confirmatory symbolic cohort.
- At least 95% recall for deterministic unsupported/ambiguous cases; false abstention is reported separately.
- Every claimed causal gene has a same-owner/same-seed counterfactual showing its intended effect.
- No style-authenticity or listener-preference superiority claim until a held-out, blinded expert/listener cohort supports the precise claim.

Do not turn a proposed threshold into a pass merely because tests are authored or workflows are queued.

## 8. Current implementation draft — M0 envelope

The current branch now contains a first Rust contract in:

- `crates/domains/symthaea-music-theory/src/style_trajectory.rs`
- exported from `symthaea-music-theory` via `lib.rs`

The current draft defines:

- `StyleTrajectoryPlanV1`, ordered `StyleTrajectorySegmentV1` intervals and `StyleTrajectoryTransitionV1` edges;
- a four-value transition mode vocabulary: hard boundary, composed bridge, gradual morph and ensemble overlay;
- lowercase SHA-256 shape checks for source-score and profile revisions;
- exact rational partition checks using `Duration` and `i128` cross-products, never floating-point beat comparisons;
- first-at-zero, positive-length, contiguous, non-overlapping and final-at-total coverage;
- exactly one correctly ordered transition record for each adjacent segment pair;
- stable machine-readable validation issue codes;
- positive, gap, overlap, missing-edge, skipped-edge, invalid-digest/owner and invalid-interval unit fixtures.

This is an intentionally narrow M0 contract. The profile and structural-owner references are opaque strings at this stage: the validator does not yet resolve the catalogue, prove a grammar route executable, verify profile capabilities, or recompute source-score bytes. The transition-mode values describe requested semantics; they do not synthesize bridge material or interpolate grammar axes. A composed bridge must be represented as its own segment.

**Validation status:** unit tests are authored but have not been compiled or executed in this environment. Exact-head GitHub workflow status must be checked separately; skipped or queued workflows are not passes. No composer, Listen, performance or renderer path is changed.

## 9. Current implementation draft — catalogue-backed admission

The branch now adds a separate Muse-layer admission module:

- `crates/domains/symthaea-muse/src/style_trajectory_admission.rs`
- exported behind the existing `theory` feature from `symthaea-muse/src/lib.rs`

This location is intentional: the dependency-light music-theory crate owns exact symbolic invariants and does not depend on the Muse UI/protocol crate. The Muse layer already depends on both the music-theory engine and the 152-style protocol catalogue, so it can safely join those sources without creating a dependency cycle.

The admission function:

- validates the exact-rational trajectory envelope, then binds it to the supplied source Score;
- resolves only canonical `catalog-style:<id>` references;
- checks catalogue composability and the catalogue-to-`Style::ALL` route mapping;
- maps the resolved Style to its current GrammarFamily;
- permits a structural owner only when its family has an explicit typed GrammarPlanEvidence dispatch in the current composer;
- rejects research-only/unrouted catalogue rows and structural owners that hit the generic Compatibility fallback;
- permits a non-structural profile reference to resolve to a callable compatibility route, but explicitly marks it as not having a typed grammar plan;
- carries ExpertReviewRequired status into the admission result instead of suppressing it.

The current typed-dispatch allowlist is derived from the explicit return arms in `compose_with_grammar_plan`: PeriodSentence, Contrapuntal, GrooveCycle, ProcessAdditive, RagaModalArc, BluesCallResponse and JazzChorus. Families handled by the catch-all Compatibility arm are not admitted as structural owners. This list must be revised alongside the grammar dispatcher when a new typed engine is added; the fixture tests guard known dedicated and fallback routes.

### Important claim boundaries

- A typed grammar plan proves that the current composer has a typed planning/evidence route. It does not prove that every style-specific dimension is distinctive or that all grammar obligations are musically fulfilled.
- A callable profile reference does not prove which donor dimensions will be applied; the causal gene/effect verifier remains a later gate.
- Each profile SHA-256 is recomputed from canonical bytes containing the resolved style name, typed grammar profile, serialized CompositionSpec, and music-theory crate version under muse-style-profile-spec-serde-json-sha256-v1. This binds the current built-in profile configuration, not executable composer code or a perceptual effect.
- ExpertReviewRequired is a technical catalogue status that is preserved in the result. Successful technical admission does not establish cultural authenticity or completed expert review.
- The source-score SHA-256 is recomputed from the exact serde_json byte serialization under the explicit schema muse-score-serde-json-sha256-v1, and the plan total duration must equal Score.total_beats exactly.
- No composer, renderer, Listen or live-interaction path is invoked or changed.

The module includes fixtures for: Afro-Cuban GrooveCycle → Minimalism ProcessAdditive, a typed PeriodSentence owner, research-only rejection, compatibility fallback rejection, malformed/unknown/noncanonical references, expert-review propagation, callable donor-profile resolution, exact source-score digest replay, changed-score rejection, unsupported digest-schema rejection and score-duration mismatch rejection.

**Validation status:** tests are authored but have not been compiled or executed in this environment. The current exact PR head and Actions state are reported separately; skipped workflows are not passes.

## 10. Work sequence

### M0 — complete contract and verifier

The current branch implements the trajectory envelope validator, catalogue-backed style/grammar route admission, exact source-score digest/duration binding, and recomputed built-in profile-configuration digest admission. Remaining M0 work is explicit admission dispositions, immutable custom-profile revision storage/signing, and broader route/negative fixtures. Reuse existing score/time/digest/profile types. No generation path changes.

### M1 — two-family trajectory

First use the non-expert-review pair Classical (PeriodSentence) → Minimalism (ProcessAdditive), then use Afro-Cuban GrooveCycle only in a clearly marked technical fixture while its ExpertReviewRequired status remains visible. Generate each segment with frozen intent/seed, validate its actual Score and typed grammar obligations, and only then define concatenation/bridge semantics. Implement one hard boundary before an explicit composed bridge; do not interpolate discrete musical systems.

### M2 — identity and ensemble continuity

Integrate motif lineage, persistent PartId and temporal relationship graph. Verify that a continuing part retains identity across changes in role or qualified instrument assignment.

### M3 — causal compiler integration

Connect the resolver to the actual score-generation path. Use paired same-owner/same-seed counterfactuals to show which symbolic dimensions changed. Verify the protected complement and ensure no style label receives credit without a measurable score effect.

### M4 — performance and renderer

Consume the shared ensemble clock, renderer-specific capabilities and instrument receipts. Keep symbolic transition correctness separate from audible transition quality. Test audio continuity under both native and alternative renderers.

### M5 — broader coverage

Expand across style constellations based on evidence and executable capability. The 152-style catalogue is a finite coverage target, not a reason to synthesize generic fallbacks. Do not claim all 152×151 ordered transitions are supported; publish a sparse capability matrix with explicit unsupported cells.

## 11. Industry comparison and strategic positioning

The public industry frontier already includes strong prompt-to-audio products and interactive control:

- Suno v6 documents up to eight minutes per generation, Extend/Cover/Replace Section and natural-language editing: https://help.suno.com/en/articles/13924801
- Google Lyria RealTime documents persistent bidirectional WebSocket streaming, weighted-prompt steering and tempo/density/scale controls; its developer guide notes that abrupt prompt/configuration changes can produce hard transitions: https://ai.google.dev/gemini-api/docs/realtime-music-generation
- Google Lyria 3.5 documents full songs with timestamped section prompting, while the current guide says multi-turn editing is not supported in that generation endpoint: https://ai.google.dev/gemini-api/docs/music-generation
- Mureka V9.5 describes whole-song structure, section-dependent arrangement density and prompt control. Its published pass rates come from a vendor-described 100-song internal evaluation and must not be treated as independently comparable preference results: https://www.mureka.ai/blog_en/mureka-v9-5-model-changes.html
- Artificial Analysis provides a useful separate blind-preference reference, based on same-prompt pairwise choices and standardized audio delivery: https://artificialanalysis.ai/methodology/music

Melothaea should not claim to beat these systems on raw audio preference without a duration-matched, blinded test. The defensible differentiation is verifiable symbolic control, explicit continuity, exact localized edits, source/rights provenance, renderer-independent style identity and reciprocal musical collaboration. Preference, structure, control, provenance and interaction must remain separate scorecards.

## 12. Claims boundary

This document is a research/implementation proposal accompanied by the M0 trajectory-envelope validator and catalogue-backed admission module on this branch. Those changes add no composer route, renderer behavior, dataset, training run, compiled test result, CI result, listener study or cultural qualification. Unit tests are authored but not yet executed. The branch does not claim that the style catalogue is complete.


## 11. M0 follow-through — profile configuration digest admission

The first implementation left profile hashes as lowercase-hex shape checks. That was an evidence gap: an arbitrary 64-character value could be labelled as a profile revision without being recomputed from the referenced built-in style.

The current branch closes that narrower gap:

- `style_profile_sha256_v1(style)` serializes a deterministic tuple of the profile digest schema, `MUSIC_THEORY_ENGINE_VERSION`, the canonical `Style` debug name, `GrammarProfile`, and `Style::spec()`, then computes lowercase SHA-256.
- Each segment's `profile_ref` is resolved through the canonical 152-entry catalogue and `Style::ALL`; its supplied `profile_sha256` must equal the recomputed digest.
- Structural-owner route resolution remains separate from profile configuration binding. The digest describes built-in configuration, not executable composer code and not the efficacy of donor dimensions.
- A changed/stale or arbitrary digest returns the stable issue code `profile_revision_digest_mismatch`; absent bytes and serialization failure are distinct findings.
- Tests derive fixture digests from the same public helper and include an altered-digest negative control.

This is stronger than checking hash syntax, but it is not a general immutable profile-registry service. Custom/user-authored profile byte stores, signing, deprecation and historical revision retrieval remain future work. Profile digest schema and score digest schema are separate by design.

**Validation status:** tests are authored; they have not been compiled or run here. No exact-head workflow success is inferred from the commit.
