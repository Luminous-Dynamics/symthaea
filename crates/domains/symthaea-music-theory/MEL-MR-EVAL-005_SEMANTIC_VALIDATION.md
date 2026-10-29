# MEL-MR-EVAL-005 — Semantic Contract Gates and Mutation Protocol

**Draft:** v0.1.0  
**Parent benchmark design:** https://github.com/Luminous-Dynamics/symthaea/issues/8799  
**Tracking subtask:** https://github.com/Luminous-Dynamics/symthaea/issues/8814  
**Related draft protocol PR:** https://github.com/Luminous-Dynamics/symthaea/pull/8816

This is a conformance specification for the draft contract in `mel-mr-eval-005-contract-schema-v0.1.json`. It specifies intended behavior; it does not claim the production validator has been implemented or executed.

## 1. Paradigm shift: compile authorization once, verify independently

A request should not be interpreted independently by the planner, editor, renderer and audit layer. Those components can accidentally disagree about which events, tracks or properties were authorized.

Introduce a **contract compiler** with two outputs:
1. a normalized authorization representation used to constrain the edit plan;
2. a verifier plan used after the edit to test the requested change and protected invariants.

Both outputs derive from the exact same validated source contract, but the post-edit verifier must independently observe the result rather than trusting the editor's self-report. The receipt binds the canonical contract, source/target content commitments, normalized-mask version, renderer and detector configurations, and complete per-gate results.

This does not mean the editor and verifier should share an implementation for the critical predicate itself: shared code can share the same bug. Keep the contract normalization format stable and versioned, and use an independently written reference oracle plus adversarial mutation vectors to cross-check the production evaluator.

## 2. Validation stages and verdicts

Use three contract validation outcomes, separate from the later edit verdict:

- **valid** — structural and semantic checks pass, and the declared evidence can in principle be measured;
- **reject** — malformed, contradictory, unauthorized, or out-of-domain contract; no editing should begin;
- **abstain** — the request may be meaningful, but this environment cannot support the promised measurement/replay claim (for example, required detector outside validation domain or deterministic rerender required but unavailable).

The final edit result remains a separate four-way state: **pass**, **fail**, **abstain**, or **execution_error**. Do not map a missing artifact, evaluator crash, or unavailable workflow to pass.

### Stage A — structural validation

Run the declared JSON Schema version first. Reject missing required fields, wrong field types, unknown values, malformed digest encodings, and unsupported schema versions. Defaults must be applied by the versioned semantic interpreter explicitly; do not rely on JSON Schema's non-mutating `default` annotation to fill in values.

### Stage B — semantic validation

The Rust module `src/edit_contract_validation.rs` now contains a first draft of these checks, a typed deserialization model, stable diagnostic codes, and tests that replay the JSONL mutation vectors. Its compilation and unit tests are authored but have not been executed because this environment has no Rust toolchain. The implementation is intentionally conservative: it abstains on edit-operation target domains that are not frozen yet.

The source manifest includes an explicit `track_ids` catalog. This is necessary because a track can be protected even when it never appears in the authorized edit mask; deriving the universe of tracks from the mask would reject or misinterpret legitimate protected tracks.

The following are mandatory rules beyond JSON Schema:

| Rule | Required behavior | Example diagnostic |
|---|---|---|
| Region identity | Every region ID is unique; references resolve exactly once | `DUPLICATE_REGION_ID`, `UNKNOWN_REGION_ID` |
| Time interval | Require end > start; use checked integer/rational time arithmetic; interval must fit within source duration | `INVALID_REGION_INTERVAL`, `REGION_OUT_OF_BOUNDS` |
| Track scope | Every operation must target only tracks authorized by its referenced regions; a track-scoped protected invariant must reference known tracks | `TRACK_SCOPE_VIOLATION` |
| Operation/axis type | Operation must match its declared property axis | `OPERATION_AXIS_MISMATCH` |
| Target domain | Target type and range must be valid for the operation (e.g. MIDI pitch in [0,127], valid scale factor > 0) | `TARGET_OUT_OF_DOMAIN` |
| Scope conflict | A requested change cannot simultaneously promise the same property as invariant over the same affected scope, unless a finer selector explicitly separates affected and protected events | `REQUEST_PROTECTED_SCOPE_CONFLICT` |
| Overlapping requests | Conflicting operations on the same axis and region must specify explicit order/composition semantics or be rejected | `AMBIGUOUS_OVERLAPPING_REQUESTS` |
| Tolerance semantics | Every requested change declares an explicit tolerance; metric, unit, axis, calibration identity, finite non-negative max deviation and comparison boundary must agree | `TOLERANCE_UNIT_AXIS_MISMATCH` |
| Calibrated threshold | Feature-distance / annotator-calibrated limits require a named calibration artifact, detector version and evaluation domain | `CALIBRATION_ID_REQUIRED` |
| Evidence availability | A required audio gate cannot pass if no audio output or no supported detector exists | `REQUIRED_EVIDENCE_MISSING` or abstain when unavailable rather than malformed |
| Render replay | If the contract requires deterministic rerendering, nondeterministic or unidentified renderer configuration yields abstain | `REPLAY_NOT_REPRODUCIBLE` |
| Digest binding | Digest syntax passes structural checks; actual bytes/content and role separation are verified when artifacts are available | `CONTENT_COMMITMENT_MISMATCH` |
| Version closure | Unknown schema, normalization, verifier, renderer or detector versions must not silently inherit current semantics | `UNSUPPORTED_VERSION` |

Scope intersection must consider the actual source-track set as well as region/time scope: a protected property on track B must not conflict with an edit authorized only on track A. Define `everywhere` as all source time on the selected tracks (or all tracks if omitted); `inside_mask` as the union of the selected authorization regions, optionally track-filtered; and `outside_mask` as the complement of all authorization regions. Do not combine `everywhere` or `outside_mask` with `region_ids`, because those combinations are ambiguous. An `inside_mask` track filter must be a subset of the tracks in its selected regions. The semantic oracle must compute source duration from integer frame count and sample rate using checked wide-integer arithmetic. Avoid rounding the duration to floating-point seconds before boundary checks. If timestamps in microseconds cannot represent a sample boundary exactly, specify the rounding/conversion rule in the versioned protocol and test the first frame before/after every boundary.

### Stage C — canonical normalization

Normalize, before authorization:
- region intervals and track sets;
- the mapping from requested operations to the properties they may modify;
- protected invariant scopes;
- unit/tolerance semantics;
- required evidence and abstention rules.

Normalization must be deterministic and versioned. Do not silently merge overlapping regions when the merge changes permission semantics. When overlapping region IDs create ambiguous permissions for the same track/time/property, reject the contract until precedence or union semantics are explicit.

## 3. Property-specific contracts

Do not allow a free-form string target to bypass a typed operation. Examples of target domains:
- `transpose_semitones`: finite integer semitone delta within a documented supported range;
- `set_pitch`: integer MIDI note 0–127 (or explicitly versioned alternative pitch representation);
- `scale_duration` / `scale_onset`: finite, positive scale where the operation definition requires a multiplicative factor; no silent zero / negative scale;
- `set_velocity`: documented MIDI velocity range and clear treatment of note-off velocity;
- `change_instrument`: a catalog or program identifier with a declared mapping and version;
- acoustic feature metrics: explicit feature extractor, calibration set, and domain. A normalized distance threshold without this context is not a portable tolerance.

This table is illustrative; the accepted domain must be encoded in schema and code together, and cross-version differences must be explicit.

## 4. Mutation and metamorphic conformance

Use the 34-case JSONL corpus `mel-mr-eval-005-semantic-validation-cases-v0.1.jsonl`. It includes cross-track and non-overlapping-region positive controls, so a validator that rejects every edit or treats each same-axis invariant as a conflict cannot pass. Each row:
1. identifies the immutable base fixture;
2. applies an ordered JSON-Pointer mutation list;
3. declares expected structural/semantic verdict and stable diagnostic codes;
4. states the invariant the test protects.

Conformance requirements:
- Every mandatory semantic rule has at least one isolated negative vector.
- Every critical rule also appears in at least one compound-fault vector, so the implementation cannot pass by noticing only a single error.
- Include positive valid cases that differ in legitimate ways; a validator that rejects everything must fail positive controls.
- Preserve expected codes and verdicts in regression tests. More than one error may be reported, but the expected code set must be included.
- Run mutation tests against the production validator and an independently implemented reference oracle; discrepancy fails the gate.
- Never report just aggregate accuracy. Publish per-code detection, false reject, false accept, and abstention counts.

### Metamorphic laws for edits

Alongside static contract vectors, the future edit evaluator should test invariant relationships:
- transposition changes allowed pitches by the requested delta while onset/duration and all out-of-scope pitches remain unchanged;
- a local pitch edit leaves unrelated event IDs, pitches, onsets, durations and track assignments unchanged;
- an onset edit leaves pitch and duration unchanged unless explicitly authorized;
- a duration edit preserves onset and pitch unless explicitly authorized;
- reordering the physical serialization without changing the canonical sequence semantics either preserves the canonical commitment or is rejected by the documented canonicalization rule;
- changing a source, target, operation, tolerance, renderer version, detector version or calibration artifact invalidates the corresponding receipt binding;
- duplicating an input event, moving a note outside the mask, or injecting a barely-over-tolerance violation must be caught by the relevant gate.

These are executable invariants, not human-facing aspirations. Each law needs a passing control and one or more mutation-kill tests.

## 5. Benchmark interop and current field position (checked 2026-10-29)

A useful external track is MMAE, the Massive Multitask Audio Editing benchmark. Its repository describes 2,000 examples and 17,741 atomic rubrics across seven audio modalities and six complexity levels. Its public evaluation uses Qwen3-Omni rubric judgements and reports:
- Instruction Following Rate (IFR);
- Consistency Rate (CR);
- Exact Match Rate (EMR), requiring all instruction-following and consistency rubrics for a sample to pass.

Sources:
- MMAE benchmark and evaluator: https://github.com/ddlBoJack/MMAE
- Dataset card: https://huggingface.co/datasets/BoJack/MMAE

A separate Audio Editing Challenge for ICASSP 2027 states that its submission SDK and previously unreleased tests are planned for **November 10, 2026**, with separate 500-example single-model and agent tracks. The test rubrics are to remain hidden until official results are finalized:
- https://audio-editing-challenge.github.io/
- https://audio-editing-challenge.github.io/leaderboard/

The AudioEditSurvey repository's snapshot (checked 2026-10-29) reports several **different-scope** reference numbers:
- On the original full MMAE single-model comparison, Step-Audio-EditX leads IFR at 44.86% and CR at 58.88%; Ming-UniAudio leads EMR at 3.20%.
- The challenge's LLM-agent baseline (router + DSP + SAM-Audio + AuK) reports 7.45% EMR, 44.06% IFR, and 74.63% CR across all 2,000 cases.
- AuK without Prompt Enhancer reports 7.58% EMR on a 1,003-case single-operation subset; Audio-Omni's 4.99% EMR is on a separate 801-case subset of clips up to 10 seconds.

These figures are not one leaderboard table and must not be ranked against each other without matching evaluator version, task subset, model/tool versions and compute budget. MEL is not an audio editor today; they are references for a future ablation of **baseline agent vs the same agent + contract verifier vs the same agent + verifier-driven repair**. Source: https://github.com/MM-Speech/AudioEditSurvey#benchmarks.

As of 2026-10-29, the challenge's hidden-test submission interface is not yet open. The published registration deadline was October 8; participation requires existing registration or explicit organizer permission. Do not claim challenge results in advance.

### How to compare without conflating tasks

Produce a **two-part scorecard**:
1. external-compatible IFR/CR/EMR on exactly the MMAE-compatible task slice and with its evaluator version, where licensing and available interfaces permit;
2. MEL-specific contract metrics: critical collateral false-accept rate, family-held-out symbolic correctness, per-axis protected-property violations, calibrated abstention, deterministic receipt replay and latency.

MMAE uses a rubric-based multimodal judge; MEL's critical structural assertions should additionally use deterministic rules wherever ground truth exists. Report judge disagreement and sensitivity, and do not use MLLM majority vote as the sole authority for claims like exact event preservation.

The Audio Editing Challenge could become a near-term external evaluation opportunity if registration, modality/task eligibility and resource constraints permit. Do not silently infer eligibility for every Symthaea edit: symbolic score edits, mixture-grounded audio editing and generic speech/sound edits are not identical task distributions.

## 6. Gate targets (proposed, not yet measured)

- 100% pass/fail agreement on deterministic positive/negative contract fixtures after execution.
- Zero accepted critical-scope conflicts, malformed intervals, unsupported operations or receipt-binding mutations in the defined adversarial corpus.
- ≥95% requested-property success and ≤5% protected-property violations on independent held-out annotated edits before reliability claims.
- 100% replay agreement for valid supported deterministic fixtures; abstain for inputs outside the verified replay domain.
- Report IFR/CR/EMR separately on the compatible external subset; do not turn them into an MEL-specific weighted average.
- Publish per-rule false-accept/false-reject rates, family-level confidence intervals and p50/p95 latency.

The numbers above are goals, not results. Zero false accepts in a finite panel must be accompanied by a confidence interval; it is not proof that population risk is zero.

## 7. Implementation order

1. Validate schema and conformance JSONL structurally.
2. Implement versioned semantic checks using checked integer arithmetic and stable error codes.
3. Implement canonical region/property normalization with unit tests.
4. Run the mutation corpus through the production Rust validator and an independent reference oracle.
5. Add metamorphic tests over actual symbolic edits.
6. Build controlled audio renders and calibrate detector floors.
7. Run blinded annotations and freeze test split/thresholds before final scoring.
8. Run an MMAE-compatible external evaluation only where task compatibility and data rights permit.
9. Report exact commit, exact test command, executed result, tool versions, and artifact digests.

Until stages 1–4 are executed at an exact commit, this document and vector corpus remain authored research artifacts, not proof of implementation correctness.
