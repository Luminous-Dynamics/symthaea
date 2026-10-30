# Query-Firewalled Tabular NAS Evaluator v0.1

**Status:** Rust source and synthetic tests authored; not compiled, executed, or qualified.  
**Tracking:** [#9160](https://github.com/Luminous-Dynamics/symthaea/issues/9160), under [#9071](https://github.com/Luminous-Dynamics/symthaea/issues/9071).  
**Implementation:** `crates/domains/symthaea-architecture-engineering/src/tabular_nas_firewall.rs`.

## 1. Why a query firewall is required

A tabular NAS benchmark contains both cheap architecture features and expensive target labels. Search policies need to inspect architecture topology and permitted zero-cost proxy scores, but must not read the complete target accuracy table and then pretend that their selected architectures were discovered under a limited query budget.

This boundary matters for the current SOTA comparison. ZAPS (Touayouch, Najem and Benjelloun, September 12, 2026) scores methods by the proportion of the true top-100 architectures that they actually evaluate, not by the surrogate's predicted ranking. Its paper also explicitly says its offline ProxyFit stage ranks proxies using accuracy correlations computed over the full search space, then reports cross-dataset proxy-transfer experiments. That is an explicit calibration protocol, not the same thing as a strict no-oracle run. See [the ZAPS paper](https://arxiv.org/abs/2609.14184), especially Sections 3.1 and 4.8.

The benchmark needs two named protocols:

1. **Paper-protocol reproduction:** implement ZAPS's declared offline full-space ProxyFit calibration as a separate, explicitly labelled reproduction. Do not describe it as a no-oracle evaluation.
2. **Strict no-oracle evaluation:** select/calibrate proxy identities on a separately identified development task/split; freeze that selection and its digest; then run the search on a distinct challenge task/split without exposing challenge labels except through the query API.

The initial Rust contract supports the strict variant by requiring different calibration and evaluation split IDs and recording the calibration artifact digest in the manifest/receipt. It does not implement ProxyFit or verify a calibration artifact's statistical correctness yet.

## 2. Information-flow contract

The public policy callback receives only:

- all permitted architecture IDs;
- topology encodings;
- proxy feature values with a consistent, declared feature schema;
- a mutable `QueryOracle` capability.

It does **not** receive:

- the full target-label table;
- the evaluator's true top-1/top-10/top-100 IDs;
- global best score or regret;
- evaluator methods or a reference to the full benchmark object.

One successful unique target query reveals one target value and consumes one unit of the fixed query budget. The manifest also freezes whether the target metric is maximize-is-better or minimize-is-better; top-K ranking and simple regret use that direction, and both direction and top-K definition are bound into the query receipt. Duplicate, unknown-ID and over-budget attempts return typed errors, never return a target value, and remain in the query trace. The receipt contains query IDs, attempt statuses and hashes of revealed observations, but not raw target values or unqueried labels.

The evaluator retains the full target table privately. The runner consumes the benchmark object while the policy callback executes; this prevents the same benchmark value from also being captured by reference and used for a nested evaluator call. Only after the callback returns does it calculate top-K capture and regret from the set of *distinct architectures actually queried*. A policy prediction or a recommended architecture that it never queried receives no P@K credit.

### Important security boundary

This Rust module is an API/information-flow boundary for trusted code, not an operating-system sandbox. It cannot defend against unsafe code, process-memory inspection, debugger access, deliberate side channels, or a caller that separately copies target labels into a policy closure. Production comparisons should run the searcher in a separate process/container with the evaluator as the only component holding target labels. The inter-process protocol should send only candidate features, explicit query requests, one queried result at a time, and the final aggregate report.

## 3. Reproducibility and fail-closed validation

The manifest pins:

- schema/version;
- benchmark and target metric IDs;
- evaluation split ID;
- separate proxy-calibration split ID and calibration receipt SHA-256;
- raw or canonical artifact SHA-256 and a source description;
- unique architecture-query budget;
- the size of the true top-K oracle set to be reported (at least `min(100, architecture_count)`).

The constructor rejects:
- wrong schema/version or malformed digests;
- identical proxy-calibration and challenge split IDs in strict mode;
- zero or overlarge budgets;
- duplicate/empty architecture IDs;
- empty topology encodings;
- non-finite proxy/target values;
- inconsistent proxy feature keys across architectures;
- attempts to smuggle the target metric under the same proxy key;
- duplicate, missing or unknown target labels.

The query receipt hashes the benchmark ID, metric, challenge split, calibration split/digest, artifact digest, unique-query budget, ordered queried architecture IDs, every attempt status, query index and a digest of each revealed target value. Replaying the same policy's query sequence against the same manifest yields a deterministic receipt; reordering attempts changes it.

## 4. Primary metrics

ZAPS defines P@100 as the fraction of the benchmark's true top-100 architectures that the method actually evaluated within its query budget. For a smaller search space, use `min(100, N)) as the denominator. The implementation reports:

- P@1, P@10 and P@100 over distinct, actually queried architectures;
- best queried target metric;
- global best target metric and simple regret;
- unique queries, query budget and budget utilization;
- duplicate, unknown-ID and over-budget attempt counts;
- query trace SHA-256 and queried-target observation SHA-256.

Report means, standard deviations and paired confidence intervals across independent seeds. Use exactly the same query budget for the primary algorithm comparison. Keep proxy computation time, model-fit time, CPU/GPU time, wall-clock, memory and energy separate from the number of full-fidelity queries.

Do not use the term P@100 for a surrogate's predicted top-100 unless it is explicitly a different metric. The benchmark's P@100 is empirical capture of the actual oracle top-100 in the queried set.

## 5. External baseline and targets

The September 2026 ZAPS preprint reports NAS-Bench-201 P@100 of 52.3% on CIFAR-10 and 65.8% on CIFAR-100 at 200 evaluations, averaged over 200 seeds, and compares against Random Search, Local Search, REA, BANANAS and TPE. These are published reference values, not Symthaea measurements. Pin the exact preprint version and reproduce its task, metric, search space, budget and baseline settings before treating the numbers as directly comparable.

A second public source is NAS-Bench-Suite-Zero from the official [NASLib `zerocost` branch](https://github.com/automl/NASLib/tree/zerocost). Its datasheet describes 13 zero-cost proxies on 28 tasks and says the dataset is Apache-2.0. The official downloader identifies `zc_nasbench201.json` and a Google Drive file ID. That file ID is not a content digest; the raw data artifact still must be downloaded, byte-counted, SHA-256 hashed and schema-validated before pinning.

The first importer should be a one-time, documented extraction if upstream artifacts require Python/PyTorch serialization. The production search primitive and query firewall remain Rust-first and must read a canonical, pinned open format rather than acquire a Python runtime dependency.


### Current quantitative reference points

At 200 NAS-Bench-201 queries, the ZAPS preprint reports:

| Task | P@100 mean ± SD | Regret mean ± SD | Strongest P@100 baseline |
|---|---:|---:|---:|
| CIFAR-10 | 52.3% ± 6.1 pp | 0.05% ± 0.09 pp | TPE: 35.4% |
| CIFAR-100 | 65.8% ± 11.0 pp | 0.02% ± 0.14 pp | TPE: 50.5% |

At 300 queries, ZAPS reports 65.1% P@100 on CIFAR-10 and 76.1% on CIFAR-100; the gap to alternative methods narrows as the query budget rises, especially on CIFAR-100. The paper uses 200 independent seeds, a one-sided Mann–Whitney U test and Cliff's delta; it also notes that its P@100 advantage does not translate into a significant regret advantage over BANANAS on CIFAR-10 at B=200. We should therefore report P@K and regret jointly, plus effect sizes and variance—not headline P@100 alone.

Reference: [ZAPS, arXiv:2609.14184v1, Table 4 and Sections 4.4/4.8](https://arxiv.org/abs/2609.14184).

## 6. Authored implementation

The first Rust module provides:

- public feature rows with proxy scores and topology only;
- a private target-label store and evaluator-owned oracle top-K ranking;
- a callback API that passes only the feature slice and query capability;
- a hard unique-query budget;
- traceable duplicate, unknown and exhausted-budget attempts;
- per-observation digests without raw labels in the receipt;
- strict canonical JSON input with unknown-field rejection and a deterministic content digest over sorted architecture features, topology, proxy scores/times and target labels;
- per-proxy computation times in seconds, kept separate from full-fidelity query counts;
- metric direction (maximize/minimize), plus raw simple regret and normalized regret percentage (for positive best accuracy, matching the ZAPS definition).
- deterministic query receipts;
- evaluator-only P@1/P@10/P@100 and direction-aware simple regret;
- constructor validation of source, metric, split, calibration provenance, IDs and finite values.

Twenty-two unit tests are authored for budget enforcement, attempt logging, top-K capture, no credit for unqueried predictions, deterministic traces, regret, calibration/evaluation split separation, invalid proxy/target schemas and receipt provenance. They have not been compiled or executed.

## 7. One-time NASLib SuiteZero importer

The importer is `scripts/research/import_naslib_suite_zero.py` and uses only the Python standard library. It does not run inside the production Rust search runtime.

It follows the actual observed NASLib `zerocost` loader:
- `data[dataset]` selects a task such as `cifar10` or `cifar100`;
- each architecture key is the string form of the NAS-Bench-201 six-operation-index tuple;
- each architecture record contains `val_accuracy`;
- each of the 13 proxy records contains `score` and `time`, with the latter in seconds.

The converter checks the tuple length (six), operation range (0 through 4), exact one-to-one target-label coverage, all chosen proxy records, finite target values, nonnegative finite runtimes, and duplicate JSON object keys. It applies the same `+/-inf -> +/-1e9` proxy-score substitution observed in NASLib's ZeroCost evaluators, rejects NaN, and records the normalization count. It preserves proxy runtime values in `proxy_times_seconds`.

Example invocation (the calibration digest must be the actual digest of the separate development/calibration artifact plus frozen proxy-selection decision):

```bash
python3 scripts/research/import_naslib_suite_zero.py \
  --input data/zc_nasbench201.json \
  --output data/symthaea-nb201-cifar10-v0.json \
  --dataset cifar10 \
  --benchmark-id suitezero-nb201-cifar10-v1 \
  --evaluation-split-id challenge-cifar10 \
  --proxy-calibration-split-id development-cifar100 \
  --proxy-calibration-sha256 <64-hex-digest> \
  --query-budget 200 \
  --top-k 100
```

The tool records the raw input SHA-256 in the manifest, pins the observed NASLib source commit and upstream Google Drive artifact ID, emits deterministic canonical JSON, writes the canonical output SHA-256 and counts of normalized infinite proxy scores to stderr, and refuses to overwrite an existing output unless `--force` is provided. Nine standard-library Python unit tests are authored in `scripts/research/test_import_naslib_suite_zero.py`; run them with `python3 scripts/research/test_import_naslib_suite_zero.py`. They have not been executed in this environment. The raw SuiteZero artifact is still not acquired or pinned in this repository; the importer remains unvalidated against the real data until it is fetched, hashed and schema-audited.

## 8. Qualification plan

Before connecting real tables:

1. Exact-head Rust compilation and all unit tests pass.
2. Property tests compare top-K capture and simple regret against an independent reference implementation.
3. A synthetic canary policy deliberately reads features but never queries; it must receive zero capture.
4. A query-spam canary attempts duplicates, unknown IDs and more than the budget; it must reveal no extra values and all attempts must be present in the receipt.
5. A leakage canary injects the target label as a proxy column; construction must reject it.
6. Two independent challenge runs with the same query order produce identical receipts.
7. A process-isolated runner demonstrates that the policy has no filesystem/database/API route to unqueried target labels.
8. Only then import the pinned NAS-Bench-Suite-Zero/NAS-Bench-201 artifacts.

For NAS-Bench-201, reproduce ZAPS at budgets such as 50, 100, 200 and 400 (using its exact published settings where available), then compare a strict no-oracle variant with proxy calibration frozen on a disjoint task. Add at least one held-out search space (for example NAS-Bench-101 or NAS-Bench-301), with all methods using the same evaluator and query firewall. All benchmark results must include 100+ paired seeds or justify a power-based alternative.

## 9. Claim ceiling

Synthetic tests qualify only the evaluator contract. They do not qualify the real benchmark import, prove a ZAPS reproduction, establish NAS generalization, improve architecture quality, or demonstrate SOTA. A successful query firewall makes comparisons more credible; it is not itself a better search algorithm.
