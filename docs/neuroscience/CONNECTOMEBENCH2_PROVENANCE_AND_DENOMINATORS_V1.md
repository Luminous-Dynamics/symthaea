# ConnectomeBench2 provenance and denominator audit V1

**Tracking:** #10676 — SCI-CONNECTOMICS-093
**Date checked:** 6 November 2026
**Status:** V1 denominator reconciled; V2 paper denominators reconciled; immutable V2 dataset snapshot still unpinned.

## 1. The denominator correction

Do not use 401,170 and 716,485 as though they were competing counts for one immutable dataset. The Hugging Face dataset card's current `main` README explicitly says that `main` is the initially submitted V1 and that the V2 revision (`revision="v2"`) is the version used by the arXiv paper. The paper reports 716,485 expert-labelled decisions; the visible V1 card reports 401,170 rows.

| Quantity | V1 (`main`, initially submitted) | V2 (paper revision) |
|---|---:|---:|
| Total rows / decisions | 401,170 | 716,485
| Train | 319,727 | 531,734
| Validation | 43,517 | 82,822
| Test | 37,926 | 101,929
| Test + train + validation | 401,170 | 716,485
| EM coverage | 251,499 rows with EM views | Not established by the V1 card; do not carry V1 coverage into V2
| Geometry | 401,170 rows | Not separately enumerated in the retrieved V2 evidence

V2 arithmetic is exact: 531,734 + 82,822 + 101,929 = 716,485. V1 arithmetic is exact: 319,727 + 43,517 + 37,926 = 401,170.

## 2. V1 modality and task-routing facts

The visible V1 card reports five sample types:

| Sample type | Rows | EM coverage
|---|---:|---:|
| adjacent_control | 121,333 | 100%
| junction_control | 38,272 | 100%
| synapse_control | 18,182 | 100%
| merge_edit | 146,461 | 38% (rounded)
| split_edit | 77,213 | 23% (rounded)
| Total | 401,170 | 63% (rounded)

The three synthetic control classes have EM on every row; the real edit classes are stratified subsets. Therefore an all-row EM model score can be biased toward synthetic controls if it is not filtered and stratified by `sample_type`, task routing, `has_em`, species and split. Do not calculate per-class EM counts from the rounded 38% and 23% summaries.

**V1 card inconsistency:** the five reported sample-type counts sum to **401,461**, which is **291 greater** than the card's stated 401,170 rows. The split counts (319,727 + 43,517 + 37,926) do sum to 401,170. This is a source-reported discrepancy, not an inferred overlap or a correction. The V1 sample-type census is therefore not reconciled until the immutable metadata/shard snapshot is counted. The manifest records this as `SOURCE_REPORTED_COUNTS_DO_NOT_RECONCILE`; no row identities are invented.

Task routing is material: merge correction of false splits uses dual-segment renders; split correction of false merges uses single-segment renders. The V1 card warns that `em_best` was selected using both labels and can leak dual-label information; it advises discarding it for single-segment tasks. The VLM preprint separately says it dropped the oblique `em_best` plane for its merge-error task because its orientation is chosen from post-split masks.

## 3. V2 full-test comparator: exact denominator and class census

The 29 September 2026 VLM preprint reports the complete V2 test split as 101,929 operations. Its split-error / merge-error counts are:

| Species | Split-error | Merge-error | Total
|---|---:|---:|---:|
| Mouse | 20,879 | 22,094 | 42,973
| Fly | 24,065 | 24,203 | 48,268
| Human | 5,351 | 60 | 5,411
| Zebrafish | 5,223 | 54 | 5,277
| Total | 55,518 | 46,411 | 101,929

The natural class mix is very skewed in the human and zebrafish merge-error subtasks. Balanced accuracy is the paper's reported metric; it does not remove the need to report per-species/per-task denominators, sensitivity, specificity, calibration, and uncertainty.

### Full V2 test balanced accuracy (paper Table 2)

| Model / training condition | Split-error BA | Merge-error BA
|---|---:|---:|
| Qwen3.5-9B LoRA | 94.6% | 90.2%
| InternVL3-8B LoRA | 93.9% | 90.1%
| Gemma 4 12B LoRA | 95.0% | 90.9%
| ConnectomeBench2 ViT-B, EM-only reference | 94.2% | 91.1%

These are full-test results on the natural V2 test mix, with training on the V2 train and validation splits. They are not the balanced mouse/fly subset results and not the unseen-species transfer results.

### Separate balanced and transfer protocol (paper Table 3)

The preprint also evaluates a distinct balanced mouse/fly subset (3,000 train / 600 validation / 1,000 test sites per species and task) against closed VLMs and specialist networks, then transfers the adapters unchanged to unseen human and zebrafish species. Do not merge these rows into the 101,929-operation full-test denominator.

Selected transfer comparison:

| Species / task | Gemma 4 12B LoRA | Specialist comparator
|---|---:|---:|
| Human split error | 79.4% | ConvNeXt-Tiny 85.4%
| Human merge error | 77.3% | ConvNeXt-Tiny 66.9%
| Zebrafish split error | 85.4% | ConvNeXt-Tiny 89.3%
| Zebrafish merge error | 82.7% | ConvNeXt-Tiny 74.7%

The strongest VLM is not uniformly superior: the adapted VLM leads on these merge-error transfer cells, while ConvNeXt-Tiny leads on split-error transfer. A genuine industry comparison must preserve endpoint, task, species, training exposure, split, modality, model cost and evaluation regime.

## 4. Competitor landscape and what it means for our programme

- **ConnectomeBench2 ViT-B / specialist networks:** the direct baseline for local EM/mesh split and merge proofreading decisions. The published V2 paper claims human-level performance and calibration in-distribution; any reproduction must use V2, not the V1 main card.
- **Adapted VLMs:** a 2026 preprint reports near-specialist full-test balanced accuracy after LoRA and better merge-error transfer to unseen species on selected tasks. Zero-shot open VLMs are near chance in the full-test table. VLMs may be valuable for many tasks and unseen species, but need latency, GPU memory, cost-per-decision and reviewer-overhead comparisons.
- **ConnectomeBench (NeurIPS 2025):** an earlier benchmark of LLM proofreading; the conference page reports 52–82% balanced accuracy on segment identification and 75–85% on binary/multiple-choice split correction, while models generally struggled on merge-error identification. Keep its tasks and dataset separate from ConnectomeBench2.
- **Global graph-invariant stopping (Martinez et al., 4 July 2026 bioRxiv preprint):** 15 global graph invariants over 11 checkpoints are fit with monotone asymptotic models. The zebrafish example forecasts six invariants converging around 71,000 edits (upper estimate 89,000). The authors explicitly state these global proxies cannot measure progress toward a particular scientific question. Treat this as a global stopping / budgeting baseline, not a query-stability certificate.

### Our differentiated target

Do not try to claim that a query-critical discovery benchmark beats a proofreading classifier on balanced accuracy. It measures a different failure mode: an entire production channel can confidently miss a biologically admissible alternative that changes a frozen scientific conclusion.

The proper head-to-head is an end-to-end blinded workflow comparison:

1. Candidate recall for independently adjudicated, query-changing alternatives at a fixed reviewer-time or compute budget.
2. False-resolution rate: how often a workflow declares the query settled while a compatible, opposing alternative remains.
3. Query regret / decision stability after bounded or sampled reconstruction-space expansion.
4. Abstention and non-identifiability correctness under missing, weak or contradictory evidence.
5. Additional verified query-critical alternatives per reviewer-hour, GPU-hour and acquisition cost.
6. Incremental value of morphology, EM, synapse models, graph context, active sampling and intervention selection, with paired ablations.
7. Global-invariant stopping versus query-specific stopping, measured on progressive snapshots and held-out query suites.

Aspiration: at the same total review budget, aim for ≥95% recall of independently adjudicated query-critical alternatives, zero false query-resolution certificates in the held-out promotion corpus, and ≥20% lower decision regret per total review/acquisition cost than the strongest frozen workflow baseline. These are targets, not results; confidence bounds and all missed/abstained cases must be reported.

## 5. Reproducibility gate still open

The V1 `main` README snapshot is visible at Hugging Face commit `1039fdf5d69f221a7078f0a66c31b1cc04080cc3`. The card says the V2 data is selected by the movable branch/revision name `v2`, but the immutable V2 commit SHA and its shard manifest were not retrievable in this audit. Consequently:

- V1 is pinned to the exact observed `main` commit and may be audited separately.
- V2 is currently identified by the source-reported `v2` selector and reconciled published counts, but remains `UNPINNED_IMMUTABLE_REVISION`.
- No V2 data download, row-level census, model rerun, or reproduction claim is made.
- V2 row counts, `has_em` coverage, sample-type breakdown, shard SHA-256 values, license terms and split membership must be re-materialized from the exact immutable V2 commit before a V2 reproduction is called complete.

## 6. Sources

- ConnectomeBench2 paper page (2026): https://synthneuro.org/publications/connectomebench2-a-unified-benchmark-for-automated-connectomic-proofreading/
- Hugging Face ConnectomeBench2 dataset card (V1 `main` disclaimer and V1 counts): https://huggingface.co/datasets/jeffbbrown2/ConnectomeBench2
- Hugging Face V1 README commit: https://huggingface.co/datasets/jeffbbrown2/ConnectomeBench2/commit/1039fdf5d69f221a7078f0a66c31b1cc04080cc3
- Li et al., *Benchmarking Vision-Language Models on Synapse Detection and Proofreading in Connectomics*, arXiv:2609.36492 (29 September 2026): https://arxiv.org/abs/2609.36492
- Brown et al., *ConnectomeBench: Can LLMs proofread the connectome?*, NeurIPS 2025: https://proceedings.neurips.cc/paper_files/paper/2025/hash/4a70f0c8443593ca59a88ebc8a937ed6-Abstract-Datasets_and_Benchmarks_Track.html
- Martinez et al., *Connectome quality converges predictably to reveal optimal stopping points during proofreading*, bioRxiv preprint (4 July 2026): https://pmc.ncbi.nlm.nih.gov/articles/PMC13345220/

All VLM comparator results cited above are preprint results as of this date and must retain that status until their peer-reviewed version and artifacts are independently verified.