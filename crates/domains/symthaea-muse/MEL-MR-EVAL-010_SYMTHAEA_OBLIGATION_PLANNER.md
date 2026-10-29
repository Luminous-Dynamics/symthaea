# MEL-MR-EVAL-010 — Symthaea-Assisted Edit-Contract Planning

**Status:** architecture and evaluation proposal; no implementation or benchmark gain claimed  
**Tracking:** [Issue #8985](https://github.com/Luminous-Dynamics/symthaea/issues/8985)  
**Builds on:** [MEL-MR-EVAL-008](https://github.com/Luminous-Dynamics/symthaea/pull/8944) and [MEL-MR-EVAL-009](https://github.com/Luminous-Dynamics/symthaea/pull/8956)

## Executive decision

**Yes, use more of Symthaea—but separate cognitive capability from trusted evidence.** The highest-value use is a Symthaea-assisted planner that compiles user intent into a frozen, typed edit contract, chooses diagnostics under a budget, and proposes bounded repairs. Final acceptance remains deterministic, operation-specific, and independently replayable.

Do **not** route raw consciousness/Phi/FEP/HDC scores directly into acceptance. They can rank hypotheses, retrieve analogues, expose uncertainty, or choose the next measurement; none is proof that audio obeys the request.

The research shift is from **“generate an edit, then score similarity”** to **“compile the edit's obligations before generation, then verify each critical obligation against the result.”** Reward the smallest sufficient change, not a plausible-sounding result that violates a hidden constraint.

## Why this is worth doing

The [Audio Editing Challenge](https://audio-editing-challenge.github.io/leaderboard/) uses instruction-following, preservation/consistency, and exact-match metrics over the 2,000-example public MMAE set, with 17,741 atomic rubrics. Exact match requires every required rubric to pass. The challenge overview describes continuing difficulties with precisely timed, multi-step and mixed-modality instructions; the [MMAE paper](https://arxiv.org/abs/2606.07229) reports low exact-match rates and sharp drops for complex mixed-modality tasks.

That creates an opening for a reliability-first agent. Better generation alone may not fix the gap: an editor can satisfy the requested change but damage unrelated content, or preserve the source so aggressively that the requested change never happens. Requested-change obligations and preservation obligations must be separate and non-compensatory.

Music needs a second evaluation lens. [MuseCPBench](https://arxiv.org/abs/2512.14629) studies preservation of musical context, while the [AudioEditSurvey benchmark index](https://github.com/MM-Speech/AudioEditSurvey) lists music-oriented evaluation such as MuseCPEval for melody, harmony, rhythm, structure and timbre. Do not collapse these scores into MMAE metrics; use them to diagnose different failures.

## 1. Paradigm shift: the edit contract is a small program

Before invoking an editor, convert the instruction and source score into a versioned contract. It is structured data with explicit regions, attributes, comparators, tolerances, unsupported states, and evidence obligations—not a prose summary or an LLM-generated checklist.

For a supported pitch edit, a conceptual contract could say:

| Attribute | Requested change? | Obligation |
|---|---|---|
| Target pitch | Yes | Verify intended pitch change against the target score |
| Target onset | No | Preserve attack timing |
| Target RMS envelope | No | Remain within the frozen profile's tolerance |
| Neighboring/non-target notes | No | Preserve within operation-specific tolerance |
| Tempo, meter, part/role and score metadata | No | Enforce exact symbolic invariants |
| Renderer and assets | N/A | Bind exact configuration and trust level |
| Missing/ambiguous evidence | N/A | Abstain or reject; never infer success |

The table is illustrative; executable contracts must specify units, interval semantics, event identity, threshold source and version. Current full-Score implementation deliberately covers only a narrow single-part, non-overlapping subset for pitch edits and transposition. This RFC does not broaden supported operations.

### Contract lifecycle

1. **Parse:** represent the request as typed candidate operations; preserve the original instruction and parser/version metadata.
2. **Ground:** match operations to source-score events and time regions. Ambiguous targets or unsupported operations remain unresolved.
3. **Compile:** emit a contract containing requested deltas, protected invariants, comparator IDs, uncertainty metadata and an allowlisted calibration-profile digest.
4. **Freeze:** hash and persist the contract before rendering. The planner cannot widen a mask, relax a threshold or remove an obligation after inspecting the output.
5. **Execute:** pass the target score or edit plan through a bounded renderer/editor adapter.
6. **Audit:** independently recompute evidence against the exact buffers, frozen contract and approved profile.
7. **Decide:** a critical failure rejects; missing required evidence abstains/rejects; accept only when every critical obligation passes.
8. **Repair:** allow only a declared number of repair cycles. A repair creates a new candidate and execution receipt but inherits the original obligations. No repair can rewrite the target of evaluation.

A contract hash is a content commitment, not an approval credential. Calibration-profile approval and any renderer trust root must come from a trusted allowlist or authenticated registry. A manifest supplied by the same untrusted caller as the audio is not independent attestation.

## 2. Which Symthaea components to use, and how

The relevant modules already exist in the repository; inclusion here is a proposal, not evidence that they improve Muse today. References: [cognitive loop](https://github.com/Luminous-Dynamics/symthaea/blob/2b3189d17e5f16c0ac8a826991da1a6332984891/src/cognitive_loop/mod.rs), [hierarchical free energy](https://github.com/Luminous-Dynamics/symthaea/blob/2b3189d17e5f16c0ac8a826991da1a6332984891/src/consciousness/dynamics/hierarchical_free_energy.rs), and [native HDC similarity](https://github.com/Luminous-Dynamics/symthaea/blob/2b3189d17e5f16c0ac8a826991da1a6332984891/crates/core/symthaea-core/src/hdc/native_similarity.rs).

| Symthaea capability | High-value role | Forbidden role | Initial status |
|---|---|---|---|
| Cognitive loop / working memory | Bounded observe → plan → act → inspect → repair orchestration with immutable trace | Unlogged retries or mutating obligations after seeing output | Candidate for first integration |
| HDC retrieval | Retrieve candidate edit-contract templates by symbolic operation, event attributes and failure family | Accept/reject from embedding similarity; replacing exact score identity | Advisory retrieval only |
| FEP / expected free energy | Rank diagnostic actions by expected uncertainty reduction and cost, where likelihoods are empirically calibrated | Treating low free energy as correctness or using uncalibrated beliefs as verification | Evaluate after deterministic baseline |
| Epistemic integrity / calibration | Track OOD flags, calibration provenance, uncertainty, abstention and risk-coverage | Rewriting frozen thresholds or hiding a weak profile | Required policy surface |
| Episodic memory | Suggest prior plans and known failure recipes with provenance | Treating remembered success as evidence about current audio | Later; split by composition/source |
| CfC/LTC temporal modeling | Candidate for long-context or multi-round tasks after simpler baselines | Learned temporal state in the trusted acceptance path | Defer until ablated |
| Global workspace / attention | Prioritize unresolved high-risk obligations and avoid redundant diagnostics | Dropping a critical obligation under attention pressure | Optional scheduler with invariants |

### Bounded diagnostic-selection policy

Start with a small explicit action set: verify target event/pitch presence; audit attack timing; audit local energy envelope; check protected-region leakage; inspect score/metadata invariants; replay a provenance receipt; or request human/stronger-tool review for an unsupported case.

For each action, estimate expected reduction in decision-relevant uncertainty per unit cost. Select it only when its estimated value is positive and within a fixed action budget. Log candidate actions, estimates, selection, actual result, cost and whether the decision changed. If probabilities are not calibrated on held-out tasks, use a fixed heuristic baseline and label the EIG arm experimental.

HDC retrieval can suggest which diagnostic matters; an information-gain policy can decide whether another check is worth its cost. The detector then runs independently. **Planning may be approximate; acceptance may not be.**

## 3. Trust model: proposal, execution, evidence, decision

Keep four trust domains distinct:

1. **Proposal domain:** Symthaea creates a candidate intent parse, event mapping, plan and diagnostic schedule. It is allowed to be wrong.
2. **Execution domain:** a bounded adapter runs the editor/renderer and records exact input/output identities and configuration. A plain manifest is only a claim unless the adapter boundary is trusted.
3. **Evidence domain:** deterministic verifiers replay measurements against the exact buffers, score, masks, profile and receipt. Learned evaluators may add diagnostic signals but cannot replace a required deterministic check.
4. **Decision domain:** a small fail-closed policy evaluates all required obligations. Critical failures cannot be averaged away by perceptual similarity or confidence.

For future polyphonic support, a stable note/event correspondence layer is needed before masks can expand. It should use event IDs and note attributes with explicit ambiguity rejection, not assume that the nearest time/frequency blob is the intended target. Duration/onset scaling also needs a request-conditioned timing-warp comparator. These are future contracts, not currently supported operations.

### Threat model to test

- target note removed/attenuated inside a broad spectral mask;
- attack delayed, duplicated or missing while pitch is correct;
- neighboring note/voice modified outside the authorized region;
- mask escalation, latency-offset manipulation or unit confusion;
- receipt metrics changed and unkeyed receipt recomputed;
- profile changed while reusing its claimed identity;
- renderer, instrument bank, effects, channel layout, sample rate or latency substituted;
- wrong event selected because a repeated motif looks similar;
- OOD score or polyphonic overlap incorrectly treated as supported;
- memory retrieval biased by a near-duplicate composition;
- repair loop overfits to a metric or gets hidden extra tool calls.

## 4. Evaluation: separate reliability gains from cognitive complexity

### Matched arms

Use identical tasks, source assets, backend, prompts, seeds, inference settings and base tool budgets where applicable. One extra repair attempt must be explicitly budgeted and reported.

| Arm | System | Question |
|---|---|---|
| A | Existing editor baseline | What happens without the new gates? |
| B | A + frozen symbolic obligation contract | Does grounded intent prevent wrong/incomplete edits? |
| C | A + audio preservation audit | Does direct evidence reduce collateral damage? |
| D | A + B + C: deterministic combined gate | What does the safety envelope alone add? |
| E | D + Symthaea planning and diagnostic selection | Does cognitive scheduling add value beyond D? |
| F | E + one bounded repair/re-audit | Can repair recover failures without creating regressions? |

Include a fixed-order diagnostic-policy control for E. Otherwise E might appear better merely because it receives extra checks or a larger budget.

### Metrics: keep them disjoint

**General audio editing:** official MMAE IFR, CR and EMR on the full public set using the same evaluator; stratify by task complexity and modality. Do not compare full-MMAE values with single-operation-subset values.

**Music context:** melody, harmony, rhythm/meter, structure, timbre, and local onset/envelope preservation on compatible MuseCPBench/MuseCPEval tasks. Report dimensions and task scope separately.

**Verifier safety:** critical false accepts/rejects by attack family, obligation-level precision/recall, abstention/unsupported rate, calibration error, risk-coverage curve, mask-policy rejections, profile mismatches and report replay failures.

**Editing quality:** requested-target success, protected-region damage, over-edit ratio, edit magnitude outside support, human preference, and blinded assessment of requested-edit correctness versus audible damage.

**Operations:** diagnostic calls per task, p50/p95 latency, GPU-hours, wall-clock time, cost per accepted edit, repair success rate and percentage of repairs that were unnecessary.

**Statistics:** paired task-level comparisons with stratified bootstrap 95% confidence intervals; account for repeated examples from the same composition. Hold out whole compositions/source recordings and renderer/instrument-bank families where possible. Freeze task manifests, judge version, calibration profile, seeds and thresholds before final evaluation.

### Prespecified targets: goals, not results

1. **Safety non-regression:** zero newly introduced critical false accepts on the frozen attack suite; target at least 384 independent critical negatives with zero false accepts for an approximate two-sided 95% Wilson upper bound below 1%. Count independent composition/task examples, not multiple windows from one clip.
2. **No quality regression:** Symthaea arm E should be non-inferior to deterministic arm D on EMR with a prespecified −1 percentage-point margin, while not increasing critical false accepts. If the sample is too small, report non-inferiority as inconclusive.
3. **Operational benefit:** target at least a 15% relative reduction in unnecessary diagnostic calls or verification cost at matched safety/quality, with a paired 95% confidence interval supporting improvement. This gives a reason to add a cognitive planner even if EMR itself does not rise.
4. **Quality improvement:** use MEL-MR-EVAL-009's full-MMAE EMR target separately. Do not attribute an EMR change to Symthaea unless E is compared directly with D and the editing backend remains fixed.
5. **OOD discipline:** unsupported/OOD cases should increase abstention instead of critical false acceptance. Report coverage versus risk, not a single best threshold.
6. **Generalization:** test held-out renderers/instrument-bank and composition splits. Gains that disappear outside a familiar synthesizer or arrangement are not a general breakthrough.

## 5. Execution order

**Phase 0 — source audit:** freeze MEL-008's supported-operation boundary and profile IDs. Define a versioned data contract for a frozen edit contract and obligation receipts; avoid direct dependency on the full cognitive loop in the low-level verifier crate.

**Phase 1 — deterministic contract schema:** implement a serializable versioned contract type, canonical digest, allowed operation/attribute matrix and independent verifier tests. Add mutation tests for contract tampering, obligation removal, profile mismatch and post-render mask escalation. Reuse the existing replayable evidence APIs; do not duplicate spectral analysis.

**Phase 2 — minimal planner interface:** expose a small trait or pure function that takes the task description and supported score summary and returns a candidate contract plus explicit unresolved fields. The deterministic supported-operation validator remains the authority. Start rules-based to lock the schema and trace before adding HDC/FEP.

**Phase 3 — Symthaea adapter:** use HDC retrieval to suggest diagnostic templates and the cognitive loop to track bounded state. Make suggestions auditable. Keep a fixed-order planner as baseline and disable implicit online learning during evaluation.

**Phase 4 — EIG/repair ablation:** add diagnostic action scoring and one repair maximum. Record predicted versus realized uncertainty reduction and cost; verify extra attempts never relax the contract.

**Phase 5 — held-out benchmark:** run pre-registered matched arms on public MMAE and separate music-specific tasks, then conduct a blinded listening audit. Only then decide whether CfC/LTC, deeper memory or other cognitive subsystems are worth their runtime and complexity.

## Decision rule

Adopt a Symthaea component only when an ablation demonstrates a measurable benefit that survives held-out evaluation and preserves the deterministic trust boundary. Prefer the smallest useful combination over integrating the whole cognitive architecture.

**No code execution or benchmark gain is claimed by this RFC.** MEL-MR-EVAL-008's Rust tests and end-to-end MMAE comparison still require an exact-head build/run. This document is a design and experiment specification intended to make the next implementation testable.
