# Muse Evidence Execution Architecture v1

**Date:** 10 November 2026  
**Status:** integration contract; components are draft PRs unless stated otherwise  
**Repository:** Luminous-Dynamics/symthaea

## Core product thesis

Muse should be a musical editing instrument with a verifiable execution history—not just a singing waveform generator or a general-purpose model router.

The product contract is:

> Interpret a musician's intent into a normalized specification; compile only capability-qualified operations; authorize each mutation against exact inputs and active constraints; independently measure the result; correct within declared bounds; preserve the full instruction history; and provide reproducible evidence for every final artifact.

This separates **creative intent**, **execution authority**, **measurement authority**, and **release qualification**.

## System flow

~~~text
Musician / producer
  |
  v
Intent & score layer
  |  normalized edit request; explicit targets/protected properties
  v
CQEP-001 plan compiler
  |  typed DAG; capability profiles; independent observer obligations
  v
Runtime authority gate
  |  authorization; exact parent digest; registry snapshot; budget; sandbox
  v
Renderer / DSP / local editor
  |  output artifact + action/tool receipts
  v
Independent observation + response identification
  |  F0 / phoneme / timing / timbre / energy / locality
  |  VCRB response matrix + VNOC no-op calibrated intervals
  v
VRCO-001 bounded robust control proposal
  |  target + collateral interval must both fit; otherwise no proposal
  v
CQEP/runtime authority re-check
  |  exact parent, plan, registry, bounds, budgets and authorization
  v
Renderer / DSP / local editor (next bounded step)
  |  actual output + receipts
  v
Independent post-render measurement
  v
VMEA-001 per-edit assurance
  |  instruction result + preservation result; strict exact match
  v
VCCR-001 correction trajectory
  |  bounded corrective action; no cycles; stop on uncertainty/regression
  v
MMAE-TRACE-001 session ledger
  |  all active instructions and preservation constraints rechecked at final output
  v
Final content-addressed artifact manifest
  |
  +--> MMAE-EXPORT-001 local manifest preparation
  |       |
  |       v
  |    Organizer's official SDK validation / packaging (separate authority)
  |
  +--> Reproducible Muse project / VocalEditPatch / release evidence
~~~

The plan compiler, per-case evaluator, correction evaluator and session ledger are evidence/contract components. None of them generates audio or independently proves that upstream declarations are true.

## Authority separation

| Component | Permitted authority | Explicitly not authoritative for |
|---|---|---|
| Intent layer | Normalize user intent, score, lyrics, scope and constraints | Whether a renderer can meet them |
| CQEP-001 | Select a declared eligible operator/observer and create a typed plan | Runtime execution authorization or capability truth |
| Runtime gate | Permit one mutation when plan, authority, parent and budget checks pass | Whether output sounds correct |
| Renderer / DSP | Produce the declared transformation and output receipt | Its own correctness or preservation |
| Independent observers | Measure declared acoustic/perceptual properties using pinned evaluators | User intent or permission to mutate |
| VMEA-001 | Score all required instruction/preservation rubrics for one case | Audio measurement or human naturalness unless those receipts are supplied |
| VCRB/VNOC/VIA | Qualify control response, no-op tolerance and interval decisions | Overall musical quality |
| VELB-001 | Quantify target change and protected-region locality | Musical coherence or intelligibility by itself |
| VCCR-001 | Qualify a bounded correction trajectory | Whether the musician authorized the goal |
| MMAE-TRACE-001 | Preserve multi-turn constraints, exact output chain and final status | Whether upstream intent interpretation was correct |
| MMAE-EXPORT-001 | Prepare complete ID/path/digest manifest | WAV validity, official rubrics, eligibility or upload |
| Official challenge SDK/judge | Official package validation and challenge score | Muse's richer independent engineering evidence |

No model or renderer can self-certify. A distinct observer artifact digest is necessary but does not by itself prove statistical independence; empirical independence and evaluator validity require separate qualification.

## Runtime mutation protocol

Each mutation should follow the same transaction:

1. **Resolve:** Bind the normalized request to a stable target, protected properties and user authorization.
2. **Plan:** Select a registered operator using the frozen capability registry; preserve exact artifact types, target scope and required evidence.
3. **Preflight:** Check operator artifact/version, exact parent-output digest, input manifest, dependency receipts, capability bounds, rights/policy and remaining resource budget. Any VRCO proposal must also fit every interval target and collateral guard.
4. **Execute:** Run only the declared operation in the designated isolation boundary. Record tool attempt, action digest, configuration, seed/determinism and output digest.
5. **Observe:** Run independent evaluators on the exact output; bind every interval to evaluator artifact and calibration receipts.
6. **Decide:** PASS only when the entire uncertainty interval is within its frozen acceptance region. FAIL when wholly outside; INDETERMINATE when overlapping.
7. **Commit or stop:** Only a qualified state can authorize a dependent mutation. Failed/unsupported/abstained attempts remain in the ledger. Unknown outcome requires reconciliation before a new mutation.
8. **Finalize:** Re-evaluate every active instruction and preservation constraint against the exact final output. Finalization is immutable; later edits create a new session/version.

This is deliberately stricter than a conversational model deciding that its last output "sounds right."

## Failure semantics

| State | Meaning | Next action |
|---|---|---|
| UNSUPPORTED | No registered operator meets the declared contract | Abstain, ask for a scope/goal change, or register and qualify a new operator |
| INVALID | Stale identity, malformed graph, missing dependency or conflicting evidence | Reject and repair specification/receipt; no mutation |
| BUDGET_EXCEEDED | Eligible plan exceeds declared resource ceiling | Ask for authorization to change budget or scope; no silent fallback |
| FAILED | Operator attempt failed | Preserve failure receipt; reconcile output state before retry |
| INDETERMINATE | Observation cannot establish pass or fail | No dependent mutation; gather better evidence or stop |
| NONCOMPLIANT | A definite active requirement failed | Correct under the same declared bounds or return to the user |
| QUALIFIED | Every required final instruction and preservation check passes | Release artifact with complete evidence package |

An abstention is a first-class action, not an embarrassing failure to hide. Failed and unsupported cases stay in benchmark denominators.

## Benchmark programme

Muse should report two distinct evaluation families.

### Musical editing and singing

Use LyricEditBench and matched YingMusic-Singer-Plus / Vevo2 comparisons for the six lyric operations. Keep language, technique, same-/cross-timbre and edit type as separate strata. Report PER, SIM, F0-CORR, Vocal Score, human naturalness and melody ratings, plus Muse-specific locality, neighboring-phone timing, seam quality and protected accompaniment.

Use VPTB-001 for score-independent F0 tracker comparisons against pinned RMVPE, pYIN, CREPE/FCPE/SwiftF0 reference implementations where supported. Score-relative pitch remains a distinct measurement track.

Use VCRB/VNOC/VRCO/VIA/VELB/VCCR for control response, no-op, robust bounded intervention, interval assurance, locality and correction trajectories. A proposed robust step is not a measured result; always measure the actual render. A passing pitch score is not a passing performance.

### General instruction-based audio editing

Use MMAE public development cases and the released Track 1/Track 2 inputs where useful. The official hidden-track baseline on the 500-case Track 1 set is AuK base with Prompt Enhancer disabled: IFR 40.33%, CR 74.70%, EMR 7.60%. Track 2's 500-case agent baseline uses DeepSeek-V4-Flash as a router over DSP tools, SAM-Audio-Large and AuK with Prompt Enhancer: IFR 44.13%, CR 76.09%, EMR 8.00%. These two tracks have different scope and must not be compared as a model-vs-agent ablation.

The public baseline repository also reports a separate local Qwen3-Omni-judged result on the 2,000-example development benchmark (Agent: IFR 44.06%, CR 74.63%, EMR 7.45%; Single Model: 1003-example subset, IFR 42.12%, CR 75.78%, EMR 7.58%). Do not mix these results with the held-out track cards.

The official challenge uses a frozen Qwen3-Omni judge with three shuffled evaluations per rubric and majority voting. Keep its official IFR/CR/EMR distinct from Muse's independent acoustic and human evidence. Competition registration closed October 8; only registered teams with organizer-issued credentials can assume official leaderboard participation. Public development testing is still a useful engineering comparison, not an official score.

## Proposed comparative qualification targets

Targets for controlled experiments, not current achievements:

- **Planner safety:** 100% rejection on missing capability, cycle, stale parent, type mismatch, unqualified observer and budget negative controls.
- **Evidence completeness:** 100% required rubric/provenance coverage on every scored case; zero indeterminate outcomes counted as pass.
- **Locality:** zero global fallback where the edit contract is sample/phrase-local.
- **Correction:** at least 90% of supported requested controls within their calibrated target on first intervention, progressing toward 95%; collateral guards must be met independently.
- **Task assurance:** 90%+ strict VMEA exact match on the *declared supported subset* before widening capability claims.
- **MMAE Track 2:** first public development milestone is to exceed the 8.00% held-out agent baseline only on a matched evaluation scope/judge; initial research gate should target at least 12% local EMR with no reduction in IFR/CR, then 16%+, with paired confidence intervals and per-complexity results. This is an experimental target, not a leaderboard prediction.
- **Cost efficiency:** lower render/tool/compute cost than a matched router at equal or better exact match.
- **Human validation:** blinded naturalness, melody, identity and edit-success judgments remain separate; never tune the only judge against the confirmatory outputs.

For a 500-case benchmark, report binomial uncertainty and paired test differences. Do not claim superiority from a small point-estimate gap without uncertainty analysis. For acoustic time-series metrics, resample at singer/song/session level, not frame level.

## Near-term execution order

1. Run the dedicated exact-head contracts for #11488, #11491, #11506, #11517, #11523 and #11528; resolve failures without weakening gates.
2. Integrate VMEA rubric profiles with VCCR/VIA/VPTB observations through the versioned observation adapter.
3. Connect VCRB/VNOC response identification to VRCO's bounded solver, then require CQEP/runtime preflight before any proposed control delta is executed.
4. Add the runtime authority gate that verifies the exact CQEP plan, registry snapshot, parent artifact and evidence preconditions before any renderer/DSP mutation.
5. Build the first real, rights-cleared singing edit pilot: one-word substitution, one-phone correction, one-note edit and one phrase-level edit.
6. Compare current Muse render path against YingMusic-Singer-Plus and Vevo2 on matched LyricEditBench tasks, and independently measure preservation/locality.
7. Use MMAE public development data to compare naive routing, fixed sequences, static capability-qualified plans and evidence-feedback replanning.
8. Only after the local export matches the official SDK, create a fully valid sample submission directory; official server validation remains mandatory.

## Current engineering components

- #11401 VCRB-001 control response matrix
- #11420 VNOC-001 no-op tolerance calibration
- #11424 independent pitch analyzer correction
- #11488 VCCR-001 correction trajectories (draft PR)
- #11491 VPTB-001 pitch benchmark evaluator (draft PR)
- #11506 VMEA-001 task-aware edit assurance (draft PR)
- #11517 MMAE-TRACE-001 session ledger (draft PR)
- #11523 MMAE-EXPORT-001 submission manifest exporter (draft PR)
- #11528 CQEP-001 capability-qualified edit planner (draft PR)
- #11533 / PR #11536 VRCO-001 robust bounded vocal control solver (draft PR)
- #11356 renderer-neutral VocalEditPatch
- #11402 versioned acoustic observation integration
- #11525 CQEP-001 plan compiler issue

## Primary references

1. [ICASSP 2027 Audio Editing Challenge](https://audio-editing-challenge.github.io/) — task definitions, official metrics, evaluation protocol, eligibility and timeline.
2. [Track 1 official test card](https://huggingface.co/datasets/ICASSP2027-AudioEditingChallenge/Track1-Single-Model) — 500 hidden-rubric single-complexity examples and AuK baseline.
3. [Track 2 official test card](https://huggingface.co/datasets/ICASSP2027-AudioEditingChallenge/Track2-Agent) — 500 hidden-rubric agent examples, including multi-audio, multi-round and multi-hop, and official agent baseline.
4. [Official challenge baseline repository](https://github.com/Audio-Editing-Challenge/Audio-Editing-Challenge-Baseline) — public 2,000-example development baseline and local evaluator results.
5. [YingMusic-Singer / LyricEditBench paper](https://arxiv.org/abs/2603.24589) — melody-preserving lyric editing and six task families.
6. [LyricEditBench dataset card](https://huggingface.co/datasets/ASLP-lab/LyricEditBench) — test composition and non-commercial licence.
7. [SOTA2 singing-voice synthesis benchmark](https://www.sota2.com/research/task/singing-voice-synthesis) — published dataset-specific comparisons; not a universal cross-dataset ranking.
8. [Official Audio Editing Challenge SDK](https://github.com/Audio-Editing-Challenge/Audio-Editing-Challenge-SDK) — submission mapping, WAV and USTAR validation, and credentialed upload boundary.

## Claim boundary

This document describes an intended architecture assembled from standalone contracts. It does not claim integrated end-to-end execution, real-audio qualification, official MMAE participation, or SOTA performance. Those claims require exact-head build/test evidence, a licensed/consented audio pilot, independent evaluation, and a reproducible final artifact chain.
