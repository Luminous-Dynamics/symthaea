# NIST Fe–Co–Ni Thin-Film Library: Offline Active-Learning Benchmark Plan

**Status:** Candidate benchmark specification and source audit; no dataset bytes have been downloaded, parsed, or benchmarked in this work.  
**Date:** 2026-10-16  
**Tracking issue:** [NANO-007](https://github.com/Luminous-Dynamics/symthaea/issues/7644) (dataset/rights qualification)  
**Canonical Rust selector/evaluator:** [PR #7568](https://github.com/Luminous-Dynamics/symthaea/pull/7568), tracked by [NANO-005 #7570](https://github.com/Luminous-Dynamics/symthaea/issues/7570)  
**Small reference baseline:** [tools/nano-policy](../../tree/research/nano-001-atomistic-discovery/tools/nano-policy)  
**Benchmark policy:** [NANOTECHNOLOGY_BENCHMARK_CHARTER.md](./NANOTECHNOLOGY_BENCHMARK_CHARTER.md)  
**Domain map:** [NANOTECHNOLOGY_DOMAIN_ATLAS.md](./NANOTECHNOLOGY_DOMAIN_ATLAS.md)

## 1. Why this is a valuable new workstream

The NIST Resource for Materials Informatics (REMI) lists an open-access combinatorial-library dataset for a Fe–Co–Ni thin-film material system. The dataset contains 921 composition samples, X-ray diffraction (XRD), and experimentally measured magnetic coercivity and Kerr rotation. This is a more concrete *measured-property* offline benchmark for testing fixed-budget candidate selection than a synthetic-only task, and it complements—rather than replaces—the OC22 and MOCLIP tracks.

Official source pages:
- [NIST REMI open-access data catalog](https://pages.nist.gov/remi/data/)
- [Wang et al., Benchmarking Active Learning Strategies for Materials Optimization and Discovery](https://doi.org/10.1093/oxfmat/itac006) and [arXiv version](https://arxiv.org/abs/2204.05838)
- [NIST repository source pinned at commit 596b04435fb577175264a70c4434f5539b809510](https://github.com/usnistgov/remi/tree/596b04435fb577175264a70c4434f5539b809510)
- [Published acquisition-strategy starter code pinned at 527de6f634e55f33d519656c75e06bb446593cab](https://github.com/alex-aa-wang/Benchmarking-Active-Learning-Starter/tree/527de6f634e55f33d519656c75e06bb446593cab)

**Rights blocker:** the REMI page calls these open-access data resources and the dataset description asks users to cite two publications, but the pinned repository tree has no root LICENSE file and the examined data description does not state an explicit dataset license. Do not infer a license from “open access” or from the MIT license on the separate acquisition-strategy starter-code repository. Confirm terms for the dataset and any third-party components before incorporating its bytes into tests, derived-data releases, or redistributable artifacts.

## 2. Source identity and schema found

At pinned REMI revision `596b04435fb577175264a70c4434f5539b809510`:

| Item | Path / identity | Source-reported detail |
|---|---|---|
| Main dataset | `data/Combinatorial Libraries/Fe-Co-Ni/FeCoNi_benchmark_dataset_220501a.mat` | Git blob `e506db115d522d62618042abead7df2695ccf13e`; 289,480 bytes |
| Data description | `data/Combinatorial Libraries/Fe-Co-Ni/data description.txt` | Git blob `a2bc0f1973cefd38759d7129d5b7573ab9f53b09`; 862 bytes |
| Catalog/source | [REMI open data page](https://pages.nist.gov/remi/data/) | Categorized as combinatorial-library data; recommends citations |
| Dataset paper | [Wang et al. (2022)](https://doi.org/10.1093/oxfmat/itac006) | 921 compositions/samples; previously measured response pool |
| Starter code | [alex-aa-wang/Benchmarking-Active-Learning-Starter](https://github.com/alex-aa-wang/Benchmarking-Active-Learning-Starter/tree/527de6f634e55f33d519656c75e06bb446593cab) | MIT code license; historical Python/GPflow/TensorFlow-based reference implementation, not production dependency |

The data-description file identifies fields:
- `C`: composition per sample;
- `XRD`: X-ray diffraction data per sample;
- `TTH`: corresponding two-theta labels;
- `Coer`: magnetic coercivity, in mT;
- `Kerr`: Kerr rotation, in mrad.

These are source documentation claims. The MAT file has not been parsed locally; actual array shapes, MATLAB cell/struct types, missing entries, finite-value checks, duplicate samples, row correspondence, measurement protocols and file digest must be established in NANO-007 before any run.

## 3. Scientific question

**Can a selection strategy identify high-performing *measured* thin-film compositions with fewer oracle queries, under equal candidate pools and budgets?**

Treat two tasks separately:
1. **Kerr rotation maximization**, the paper describes as a comparatively smooth landscape.
2. **Magnetic coercivity maximization**, the paper describes as rougher and more difficult, with many local maxima.

This contrast is the scientific value of the dataset. A strategy that does well on a smooth function may fail on a rough one. Do not average them into one win score or claim cross-domain generalization from this single ternary library.

## 4. Leakage-safe offline replay

The 921 observed measurements can be used as a finite-pool oracle simulator only after rights and source checks pass. The evaluation harness holds all target values; the policy receives only features explicitly allowed by the frozen protocol. After each query, the harness exposes only the selected candidate's target outcome to adaptive baselines.

### Primary arms

1. **Random selection**, repeated over frozen seeds.
2. **Canonical-kernel farthest-first and family-balanced farthest-first**, using composition features only and stable family groups where scientifically justified; the separate `tools/nano-policy` crate can remain a simple reference baseline.
3. **Cost-aware farthest-first**, only if heterogeneous query costs are evidenced and preregistered. If all samples have equal costs, cost-aware selection is not a distinct policy and must not be presented as a separate win.
4. **Expected Improvement (EI)** using the published GP baseline and exact applicable preprocessing.
5. **Add-GP-UCB**, which the paper reports as strong on the smoother Kerr task.
6. **CAMEO or another physics-informed baseline**, only if source, algorithm, preprocessing and license are pinned and the published setup is reproduced.
7. **Ablation / negative controls**, including shuffled feature association or a deliberately randomized selector, as appropriate.

The paper's reference code imports Google Colab file helpers, TensorFlow, GPflow and plotting packages. It is a research reference, not a Rust or production dependency. Isolate and pin it; do not introduce those dependencies into Symthaea production.

### What the selector may see

**Default primary arm:** composition only, as information assumed to be available before the target coercivity or Kerr measurement.

Do **not** give every candidate its measured XRD or target-property array for free simply because those values occur in the offline file. XRD is a measurement with its own protocol and cost. A separate phase-aware arm may use XRD only if the benchmark explicitly states that it is already available or counts its acquisition cost and explains why the information is available before the target measurement.

The oracle must never expose `Coer`, `Kerr`, or hidden outcome-derived fields before a candidate is selected. The NIST dataset is a historic experimental pool: replay can evaluate policy logic on existing data, but it is **not a new physical experiment** and does not establish present-day fabrication reproducibility.

## 5. Metrics and benchmark targets

### Primary metrics

For each outcome independently, report:
- **Simple regret by query budget**: best value not yet discovered relative to the pool's true optimum, in raw physical units and a normalized value based on the source-pool range. Lower is better.

For the canonical outcome ledger, encode each maximization result as a **normalized simple-regret loss** computed by the independent evaluation harness after the query: `loss = (pool_max - observed_value) / (pool_max - pool_min)`. The pool extrema and observed value must never enter the selector request. This yields a dimensionless loss in [0, 1] for non-degenerate pools. An evaluator success threshold of 0.001 means within 0.1% of the observed pool range; this is a new normalization and must not be called an exact reproduction of the paper's “0.1% of maximum” criterion unless the paper's original formula is matched separately. If the pool range is zero, the task is degenerate and must be marked not evaluable.
- **Target attainment / queries-to-target**: number of measurements needed to get within a preregistered tolerance of the optimum. Report the distribution across episodes/seeds, not only the average.
- **Area under the best-so-far curve** over a shared, preregistered query range.
- **Budgeted utility** at fixed query counts (including an early budget and 5% of the pool, about 46 of 921 samples).

Secondary metrics:
- valid unique candidates selected; duplicate/requery count must be zero;
- composition-space coverage/diversity as diagnostics, not a surrogate for a good magnetic property;
- wall time and compute per selection;
- sensitivity to seed and initial seed set;
- differences between smooth Kerr and rough coercivity landscapes.

### Paper-reported anchors (reproduction targets, not Symthaea results)

The 2022 paper reports that for the Kerr-rotation task, all evaluated acquisition functions reach within 0.1% of the maximum within 5% of the 921-sample pool; Add-GP-UCB reaches that threshold in roughly 11 samples and is about twice as fast as the next competitor. The paper reports that the more complex coercivity landscape is harder, with Expected Improvement performing well overall.

These targets can only be used when the paper's exact normalization, initial conditions, repetition protocol, data cleaning, GP settings and metric formulas are reproduced. At 921 items, 5% is approximately 46 samples. Do not quietly change the “within 0.1%” definition into a different normalized-regret threshold without explicitly labeling it as a new metric.

### Proposed go/no-go for the Rust policy

Before exposing any held-out evaluation run, choose one primary metric and justify the practical margin and replication/power plan. For the canonical evaluator, a provisional primary is paired **best-loss reduction at budget** on the normalized simple-regret scale, with a proposed minimum margin of 0.05 (five percentage points of the observed pool-property range). The evaluator computes the mean paired reduction and a group-bootstrap interval only when every target has a valid observation in both arms; missing coverage blocks the inferential comparison. This is a proposed engineering threshold, not a result from the paper or a universal standard.

As a secondary descriptive outcome, report median best-so-far normalized regret at budgets 11 and 46 and its relative reduction versus the baseline; do not present that median as statistically confirmed unless the independent checker also supplies an appropriate paired interval. For Kerr, separately attempt to reproduce the paper's original criterion and report the exact formula rather than silently substituting the normalized-range definition.

Because this is a finite 921-point historic pool with correlated composition neighborhoods, separate algorithmic seed variation from material-family generalization. Repeated random seeds quantify stochastic policy variability on this pool; they do not create independent physical material systems. A successful offline replay remains a benchmark result for this pool, not evidence of fresh synthesis or broad materials generalization.

## 6. Design caveats learned from newer active-learning research

Nair & Foppa's 2026 critical review of materials-science active learning emphasizes that hidden workflow choices, candidate-pool construction, surrogate setup, sampling bias and metric selection can materially affect conclusions. Random sampling is necessary but not a sufficient baseline, and a single acceleration/enhancement factor is not enough. Source: https://doi.org/10.1039/D6DD00081A

Therefore:
- random is required, but **not sufficient** as the only baseline;
- include EI or another well-established acquisition function and the strongest reproducible task baseline;
- report full regret-versus-budget curves, not just a cherry-picked acceleration factor;
- preserve the same initial seed pool, oracle-query budget, candidate pool, preprocessing and exclusion rules across arms;
- use repeated seeds and both the smoother and rougher response surfaces;
- treat a null result as evidence that simple feature-space diversity did not beat the chosen baselines.

The separate NIST benchmark paper found Expected Improvement performed well overall across the two tasks. Symthaea should earn a win against that baseline, not only against random sampling.

## 7. Resource and provenance requirements

Before executing:
1. Clarify data-use terms and attribution expectations with NIST/REMI and resolve third-party data provenance.
2. Download the exact pinned artifact only after the rights gate; compute local SHA-256 and byte size.
3. Parse the MAT file in an isolated research environment; record actual arrays, shapes, units, non-finite/missing values and row alignment.
4. Freeze the preprocessing transform, candidate IDs, pool snapshot, allowed feature schema, target task and query budgets.
5. Capture all policy requests, exposed oracle responses, selected/rejected/failed decisions, outputs, seeds and version identities.
6. Write outcome observations only after selection, then use the independent post-outcome evaluator in [PR #7568](https://github.com/Luminous-Dynamics/symthaea/pull/7568) to recompute simple-regret/target-attainment metrics. Never pass the ledger or aggregate outcomes back to a selector.

No dataset download or benchmark run has happened yet; a previous binary-fetch attempt was blocked by the available download tool. Do not claim the file was locally verified or parsed.

## 8. Intended claim ceiling

A successful offline replay would show only that a fixed-budget selection policy performs better under this specific historical Fe–Co–Ni library and simulated oracle protocol. It would not establish new Fe–Co–Ni performance, reproducible fabrication, industrial value, or generalization to other nanomaterials. Such claims would require fresh physical experiments, calibrated characterization and independent replication.

**Next action:** NANO-007 gates the rights/schema audit. If cleared, run the source-pinned offline replay as a separate empirical benchmark for the Rust candidate-selection prototype while OC22 and MOCLIP retain their own validation paths.
