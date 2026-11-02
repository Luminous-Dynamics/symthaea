# Connectome Proofreading Policy Evaluation v0.1

**Status:** research protocol / schema proposal. Not a validated benchmark result.

## Why this benchmark is different

A classifier leaderboard asks whether each image crop is labelled correctly. A deployed proofreading system must decide which cases to spend expert time on, when to request another view, and when its uncertainty is too high to act. Those choices have asymmetric consequences: a missed merge/split error may remain embedded in many downstream annotations, while an unnecessary review consumes scarce expert time.

The proposed objective is therefore:

> Maximize verified error resolution per reviewer-minute subject to a preregistered safety and uncertainty budget.

This is a policy-evaluation problem, not simply a higher-accuracy classifier.

## Evidence boundary

ConnectomeBench2 V2 provides expert-labelled operation decisions and matched controls. The VLM paper evaluates 101,929 test operations at natural class mix (20,879 mouse split errors, 22,094 mouse merge errors, 24,065 fly split errors, 24,203 fly merge errors, 5,351 human split errors, 60 human merge errors, 5,223 zebrafish split errors and 54 zebrafish merge errors). It reports full-test balanced accuracy of:

| Model | Split-error | Merge-error |
|---|---:|---:|
| CB2 EM-only ViT-B | 94.2% | 91.1% |
| Qwen3.5-9B LoRA | 94.6% | 90.2% |
| InternVL3-8B LoRA | 93.9% | 90.1% |
| Gemma 4 12B LoRA | 95.0% | 90.9% |

These are the published full-test balanced-accuracy results, not policy utility or editing safety. The VLM paper also reports small human/zebrafish merge-error test supports (60 and 54 operations), so those strata require explicit uncertainty and cannot support confident broad transfer claims by themselves.

The public dataset card states that `main` is V1 and V2 must be loaded via revision `v2`. Until the V2 revision resolves to an immutable commit SHA and artifacts are hashed, no run should claim pinned V2 reproduction. See the run manifest in PR #9962.

**Important scope rule:** CB2 labels can support retrospective detection and review-triage evaluation. They do not, by themselves, provide arbitrary model-generated edit patches, post-edit graph state, reviewer-time logs, or downstream consequence labels. Therefore this protocol separates three tracks:

- **Track A — retrospective triage:** rank and route labelled cases to review, defer, or request another view. This can be evaluated on CB2 V2 after exact pinning.
- **Track B — impact-aware triage:** weight missed errors by measured downstream graph/annotation impact. This is unavailable unless an independent, provenance-bearing impact artifact is supplied. Unweighted metrics remain primary.
- **Track C — prospective edit safety:** replay actual proposed edits in a sandbox and verify topology, annotation lineage, reversibility and downstream stability. This requires new transaction traces and cannot be claimed from CB2 binary labels.

## Policy contract

A policy consumes a model score and an evidence state, then emits one of:

- `REVIEW`: send to a qualified human reviewer;
- `ACQUIRE_VIEW`: request another permitted EM/geometry view before adjudication;
- `DEFER`: do not spend review resources now.

The policy must not represent `REVIEW` as a completed correction. The reviewer outcome is a separate event and must be logged independently.

For a simple fixed-duration review protocol, rank cases by predicted probability of an error and take the top cases within the declared reviewer-minute budget. When review duration varies, compare policies using expected avoided error cost per reviewer-minute and solve the declared budget-allocation problem deterministically. If impact weights are unavailable, use the preregistered task-level cost profile and publish unweighted recall/workload curves alongside it.

For an additional-view action, estimate value of information as the expected reduction in downstream decision loss minus the acquisition and review cost. Do not use entropy reduction as a substitute for decision value. Do not score this action until the likelihood model, acquisition cost and downstream loss are frozen and calibrated on independent data.

## Required evaluation protocol

1. Resolve and pin the exact ConnectomeBench2 V2 commit, card, shard manifest and every materialized file digest.
2. Reconcile unique Parquet rows, task-routed operations and view instances using stable sample hashes. Never silently equate these denominators.
3. Preserve the published task view contract. In particular, exclude `em_best` for single-segment merge-error identification because the oblique view is selected using post-split masks and leaks target information.
4. Keep the natural-mix full test, balanced mouse/fly comparison and unseen-species transfer as separate regimes. Report species × task strata and denominators.
5. Group splits by source volume/neuron/session as appropriate. Repeated views from one operation must never leak across train/validation/test.
6. Evaluate review recall at 1%, 5%, 10%, 20% and 50% of the review budget, and at fixed reviewer-minute budgets. Report both item-count and time-based curves.
7. Run seeded random review, fixed confidence threshold, uncertainty-only ranking, score × preregistered cost ranking, the exact small-cohort knapsack oracle on sampled batches of at most 20 cases, the best supervised CB2 model, LoRA VLM and timed human workflow baselines.
8. Use cluster-aware confidence intervals (resample volumes or neurons, not individual correlated crops). Report worst-stratum results and small-support warnings.
9. Keep model selection and cost-profile selection on validation data. Freeze both before test evaluation. No repeated tuning on test outcomes.
10. Publish all negative controls and raw per-operation decisions, subject to dataset licensing and access policy.

## Metrics

### Retrospective Track A

- Error recall at fixed review fractions and reviewer-minute budgets.
- Precision among reviewed cases and number of true errors surfaced.
- Missed-error count and false-review count.
- Reviewer minutes per additional true error surfaced.
- Area under review-recall versus review-budget curve.
- Brier score, log loss, reliability diagram and ECE (binning rule specified).
- Selective risk versus coverage; `REVIEW`, `ACQUIRE_VIEW` and `DEFER` rates.
- Per-species/task/volume strata, worst-stratum performance and cluster-bootstrap intervals.
- Latency, throughput, peak RAM/VRAM and energy per reviewed operation where measured.

### Impact-aware Track B

- Unweighted missed-error count remains mandatory.
- Impact-weighted missed-error burden is reported only with a pinned independent artifact and explicit unit definition.
- Report downstream outcomes separately: changed cell identity, altered synapse partner assignment, affected annotations, graph motifs/pathways and changes in declared scientific conclusions.
- Publish sensitivity curves over cost profiles. Do not hide conclusions inside a single arbitrary weighted score.

### Prospective Track C

- Patch replay success and deterministic replay.
- Preconditions and stale-object rejection.
- Duplicate-application rejection and idempotency.
- Rollback/inverse-patch success.
- Collateral annotation mutation count.
- Topology, connectivity and identity invariant violations.
- Before/after downstream analysis drift.
- Independent verifier disagreement and reviewer overturn rate.
- Human minutes per accepted, verified correction.

## Known-answer fixture

The companion fixture contains four synthetic operations, two true errors, a fixed 2-minute review cost and a review budget of four minutes. The score ranking places one true error and one control in the top two. Expected results are therefore two reviewed cases, four reviewer-minutes, 50% review precision, 50% error recall and one missed error. A second perfect-ranking policy recovers both errors at the same budget. These values are arithmetic test vectors, not biological estimates.

The fixture also demonstrates why the oracle is a bound, not a deployable policy. Real policies must be evaluated on the same frozen cases and cost profile.

## Research targets (proposed, not achieved)

- **Efficiency:** at matched missed-error burden, reduce reviewer-minutes by at least 25% versus uncertainty-only triage in a prospective timed study.
- **Allocation quality:** at fixed 10% reviewer-minute budget, improve error recall by at least 10 absolute percentage points over uncertainty-only and seeded-random policies, with cluster-aware confidence intervals.
- **Calibration:** no materially worse Brier/log-loss or selective risk than the best supervised baseline in any adequately powered major stratum.
- **Robustness:** no silent degradation in held-out volume/species/session; abstention must increase when evidence is outside validated support.
- **Transaction safety:** 100% rejection of stale preconditions, duplicate application and digest mutation in the conformance suite; 100% rollback success for operations declared reversible; zero untracked collateral mutations in the test harness.
- **Scientific stability:** structural alternatives and ambiguous proofreading hypotheses remain explicit; no downstream claim is called invariant unless it survives every admissible alternative.

These are stretch gates for the research programme. They are not current performance claims and must be revised if feasibility or power analysis says otherwise.

## Industry comparison and how to beat it

The field already has strong components:

- **ConnectomeBench2** supplies a large, multi-species label base and a multimodal ViT baseline.
- **VLM LoRA baselines** are now approximately level with the CB2 EM-only ViT-B on the full test split; the advantage to test is cross-domain adaptability and multi-task reuse, not an assumed accuracy lead.
- **NEURD** demonstrates graph-feature-based proofreading and reports roughly 150,000 high-confidence automatic edits in MICrONS, as well as more than threefold speed-up for a challenging manual splitting workflow when suggestions are used.
- **CAVE** demonstrates the importance of versioned annotation state and reproducible edits at scale.
- **SynAnno** demonstrates that guided workflows can improve expert throughput and reduce cognitive load; a seven-expert evaluation is an operational comparator, not a direct CB2 classifier leaderboard.

The Rust policy crate now contains both a deterministic greedy value-density allocator and an exhaustive exact oracle for at most 20 candidates. The oracle is for regression tests and sampled-batch regret measurement; it is not intended to enumerate the full CB2 test set.

The differentiation opportunity is to join these ideas into a verifiable operating loop:

1. graph-aware detection and candidate localization;
2. calibrated uncertainty with structural alternatives preserved;
3. decision-theoretic review allocation per human-minute;
4. reversible, content-addressed edit transactions;
5. independent verification of topology and annotation lineage;
6. measurement of downstream scientific stability;
7. feedback that improves the model without allowing test leakage.

A system that is 0.5 points below a classifier on balanced accuracy but delivers materially more verified corrections per expert-hour, with lower harmful-edit risk and better transfer, may be more useful. That conclusion must be demonstrated in timed, controlled studies.

## Immediate work sequence

- Complete the V2 pinning and denominator reconciliation in #9936 / PR #9962.
- Compare greedy allocations against the exact small-cohort oracle and quantify regret before selecting a production scheduling policy.
- Run the known-answer, negative-control and leakage fixtures before touching CB2 test labels.
- Reproduce the ViT-B and LoRA baselines under the same operation-level unit and view contract.
- Add cluster-aware uncertainty and selective calibration.
- Start a prospective edit-transaction data contract before claiming correction safety.
- Integrate alternative structural hypotheses from PR #9891 and #9960 into downstream impact scoring.

## References

- ConnectomeBench2 paper: https://arxiv.org/abs/2606.21116
- VLM ConnectomeBench2 evaluation: https://arxiv.org/abs/2609.36492
- NEURD: https://www.nature.com/articles/s41586-025-08660-5
- CAVE: https://www.nature.com/articles/s41592-024-02426-z
- SynAnno: https://doi.org/10.1109/TVCG.2025.3634824
- Connectome quality / optimal stopping preprint: https://doi.org/10.64898/2026.06.30.735414
