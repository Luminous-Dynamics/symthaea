# QOBLIB-MIS-BASELINES-001: frozen published MIS comparison records

Status: **published records frozen; no Symthaea-vs-baseline experiment claimed**

## Snapshot and provenance

This source-side baseline is pinned to QOBLIB commit 1e0e7c523cd9707dd541cc9e5a3086125d4da3c7 (2026-10-19 09:52:56 UTC). The graph cohort in QOBLIB-MIS-COHORT-001.json uses commit d61f234b9fc816deb3d4e1dea4c031595b35acd7 (website snapshot 09:44:36 UTC). I compared all 50 MIS graph Git blob IDs across those two snapshots: all 50 are unchanged. The newer snapshot supplies the contemporaneous official best-known/optimal table and published solver records.

The companion QOBLIB-MIS-BASELINES-001.json preserves all 50 per-instance rows from the QOBLIB ParallelILS report, joined by exact instance ID to the official MIS best-known table at the pinned baseline commit. It also preserves the historical best-known value copied into the ParallelILS submission itself, because that field can predate QOBLIB's automated post-merge update.

Source identities at the pinned baseline commit:
- ParallelILS report blob: e1e5a5fbb4e148d4adcf021202314406584ccff8.
- Official MIS best-known table blob: 001ecfcab4e9147dc1a8f92a6027863660034f7a.
- HSQC es60fst02 report blob: 31cfc42f73f6760402f5d7bdfdd8cb54256f6677.
- QAOA es60fst02 report blob: 3b5257cf1219644412ccbd866370ae50aaa1b403.
- Runtime/reporting contract blob: 5b0bcc30ea1eb1c62530b22deb6a934d25e7382b.

All source URLs use the immutable commit, not a moving main link. Git blob identifiers are Git SHA-1 object IDs, not SHA-256 checksums of downloaded bytes.

## Published classical baseline across all 50 instances

The QOBLIB ParallelILS report describes a cooperative parallel iterated-local-search solver written in C++, compiled with clang -O3 -march=native, running on an Apple M5 Max (arm64, 48 GB RAM, 18 CPU cores). Its README says each instance had 10 trials, and its table records a best-known objective for each one. The record is published external evidence, not a local reproduction by Symthaea.

At the pinned official table snapshot:
- 50 of 50 instance IDs join exactly between the graph cohort, ParallelILS report and official status table.
- The 50 ParallelILS reported objectives match the current official best-known objective for all 50 instances.
- 38 instances are marked optimal and 12 are marked best known in the official table. Best known must not be described as a proven optimum.
- The submitted report counts 487 successful trials out of 500 by its stated criterion. Preserve per-instance counts; aggregate counts alone do not describe which hard instances failed.
- The median of the 50 per-instance reported mean total runtimes is 0.0009 seconds; the maximum is 120.05 seconds on brock800-1. This is a descriptive median across 50 instance-level mean runtimes, not a single pooled runtime estimate.

The wide spread is a reason to stratify by graph family and difficulty. Easy instances may be useful for parser/feasibility/regression checks but do not establish a research advantage. Hard instances such as brock800-1 and the FRB family require stronger budget-matched follow-up, including attention to reliability across runs.

## Same-instance accounting case: es60fst02

Three records at the same QOBLIB snapshot report objective 88 on the 186-vertex es60fst02 graph:

| Published policy | Reported successful runs | Reported mean total runtime | CPU time | QPU time | Hardware/report context |
|---|---:|---:|---:|---:|---|
| Classical ParallelILS | 10/10 | 0.0006 s | Not separately reported | — | Apple M5 Max, 18 cores; cooperative parallel ILS |
| HSQC hybrid hardware | 5/5 | 2.5608 s | 0.0843 s | 2.4765 s | IBM Heron R3 (ibm_aachen) plus single-core Apple M5 Max preprocessing/postprocessing |
| Hardware-aware QAOA greedy/reduction | 5/20 | 23,365.8542 s | 23,287.8542 s | 78 s | IBM Heron R3 (ibm_aachen) plus CPU VM, 32 cores and 128 GB RAM |

The quantum submission rows list Optimality Bound: N/A; reaching objective 88 matches the current best-known value but does not by itself prove optimality. The counts and timings are reported by the respective submitters, not independently rerun in this change.

QOBLIB's pinned CONTRIBUTING.md says runtimes are averages over repetitions, excludes hardware queue time, and says to exclude compilation and queue time from runtime reporting. Therefore QOBLIB-native Total Runtime is not equivalent to a broader commercial all-in cost that includes transpilation, calibration, retries, queue delays, orchestration and access fees. Report QOBLIB-native metrics when comparing with QOBLIB records, and keep any broader all-in measurements separate with their own explicit boundary.

These three values are **not a controlled head-to-head speedup experiment**: their parallelism, algorithms, preprocessing, hardware and success conditions differ. The case is useful for designing a fair comparison, not for claiming that one submission is universally faster. A fair Symthaea experiment must align the graph, target objective, number and definition of trials, stopping budget, hardware access convention, solver resources, and independently checked feasibility.

## Implications for Symthaea

1. A greedy smoke runner is a plumbing baseline, not a state-of-the-art competitor. The first validation target is correct parsing, independent feasibility, known-objective alignment and deterministic receipts; the next is a controlled comparison with a pinned implementation of a strong classical solver such as the published ParallelILS method.
2. Use the 38 officially optimal cases to test exact agreement when the optimum is proven, but use the 12 best known cases as best-known-bound comparisons—not proof of optimality.
3. Preserve per-instance outcomes and failure/timeout counts. Do not drop failed stochastic runs or average objective quality only over successful runs.
4. For each stochastic method, report the incumbent/time series and target-attainment probability. A final score alone does not permit reproducible time-to-solution comparisons.
5. Compare the same instances on the same machine for a controlled software benchmark when possible. For quantum hardware, report classical preprocessing and postprocessing, CPU/GPU/QPU runtimes, trials, actual QPU execution and an all-in cost account separately.
6. Maintain a separate random-regular-graph track for the recent quantum-enhanced greedy literature. Do not pool random-regular results with QOBLIB's named MIS suite as if their graph distributions were the same.

## Research comparison targets

- **qReduMIS**, published in Physical Review Research in September 2026, uses quantum measurements to guide classical reductions and reports experiments up to 231 qubits. Symthaea's useful question is whether that information changes final verified graph quality or total work relative to matched classical reduction/informer arms: https://doi.org/10.1103/nh3b-1wv5
- **Scalable quantum-enhanced greedy MIS on random regular graphs**, accepted 19 October 2026, uses shallow QAOA-derived local expectations to inform greedy decisions. Reproduction should include the same minimal-degree greedy arm and stronger classical baselines, fixed QAOA depth/angle provenance, physical-vs-tensor-network evidence labels, and end-to-end cost: https://doi.org/10.1103/ds1n-wsdm
- **QOBLIB** is the shared community reference for published per-instance outputs, feasibility checks and best-known attribution: https://zib-aopt.github.io/QOBLIB/

## Proposed gates for a credible advantage claim

- Correct, independently verified, source-pinned graph and solver outputs.
- Matched instances, objective/target, trial definitions, stopping budgets and resource-accounting boundaries.
- A strong, pinned classical solver baseline, not only degree-greedy or a weak random control.
- Paired per-instance results with appropriate uncertainty, preregistered primary metrics and multiplicity handling.
- Separate results by family and hardness stratum, plus genuinely held-out generated families for generalization claims.
- An aspirational target—not a current result—is at least a 2x reduction in the predeclared all-in cost to reach the same verified quality target while retaining non-inferior quality on held-out families.

No local QOBLIB solver has been run in this contribution, and it makes no claim of SOTA improvement or quantum advantage.
