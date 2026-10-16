# Nanophotonics Pilot Benchmark Scorecard: MOCLIP, HDC and Budgeted Inverse Design

**Status:** Preregistered pilot proposal; not an executed benchmark.
**Rust policy reference:** [tools/nano-policy](../../tree/research/nano-001-atomistic-discovery/tools/nano-policy) is a small isolated prototype; canonical selector development is in [PR #7568](https://github.com/Luminous-Dynamics/symthaea/pull/7568) and [NANO-005 issue #7570](https://github.com/Luminous-Dynamics/symthaea/issues/7570).
**Cross-domain benchmark charter:** [NANOTECHNOLOGY_BENCHMARK_CHARTER.md](./NANOTECHNOLOGY_BENCHMARK_CHARTER.md) sets shared reproducibility/leakage gates and domain-specific acceptance logic.  
**Date:** 2026-10-16  
**Parent domain-selection gate:** [NANO-002](https://github.com/Luminous-Dynamics/symthaea/issues/7507)  
**Related domain atlas:** [NANOTECHNOLOGY_DOMAIN_ATLAS.md](./NANOTECHNOLOGY_DOMAIN_ATLAS.md)  
**Related closed-loop protocol:** [NANOTECHNOLOGY_CLOSED_LOOP_BENCHMARK_PLAN.md](./NANOTECHNOLOGY_CLOSED_LOOP_BENCHMARK_PLAN.md)

## 1. Why this pilot now

New research surfaced a much more concrete nanophotonics benchmark than generic “try inverse design” work:

- Rodionov et al., **MOCLIP**, *Nature Communications* (published 19 August 2026): https://doi.org/10.1038/s41467-026-76714-x
- Release source, pinned for audit to commit `776d9dbce9338175b9edefed1442cfb1f918eab1`: https://github.com/RodionovSA/MOCLIP_release/tree/776d9dbce9338175b9edefed1442cfb1f918eab1
- Independent benchmark/tooling resource from the NeurIPS 2023 Datasets and Benchmarks paper, pinned for audit to `b877feffc4be737a12397fe3ad881a498c2409af`: https://github.com/jungtaekkim/nanophotonic-structures/tree/b877feffc4be737a12397fe3ad881a498c2409af

MOCLIP pairs fabricated metasurface geometry with measured polarization-resolved spectra and uses contrastive representation learning for cross-modal retrieval. The paper reports a full dataset of 466,537 unique geometry-spectrum pairs after quality filtering, with a 373,230 / 46,654 / 46,653 train/validation/test split. It reports test retrieval accuracy around 75% Top-1, 95% Top-5 and 98% Top-10. For a probe set larger than 10,000 geometries, it reports 95% of tested inverse-design predictions had spectrum MSE below 1e-2. Those are author-reported figures under that paper's exact protocol, not results reproduced here.

The paper also reports more than one million fabricated instances when exact replica arrays are counted, with measured device-size deviation mean 4 nm and standard deviation 10 nm. This makes fabrication artifacts and repeated physical samples important data-lineage dimensions, not optional metadata. Source: https://www.nature.com/articles/s41467-026-76714-x

## 2. Critical source and claim boundaries

The release contains an approximately 10,000-sample subset (`data/MOCLIP_dataset_10k.npz`, Git blob `451af237eaed4966b009554529dfe84ba112b2b6`, reported size 8,686,229 bytes) and pretrained weights (`src/models/weights/moclip_v1.pth`, blob `efcbd0758f97e804e5a7939b7c5b15dccaa0ebc7`, 47,515,250 bytes). The README says MOCLIP was trained/evaluated on an RTX 4090, recommends CUDA and says CPU inference is prohibitively slow for practical use. The complete 466,537-pair dataset is available from the corresponding author upon request.

The source repository's LICENSE blob is MIT for the repository. Before using the dataset or weights beyond personal audit, verify their specific terms, attribution and redistribution/derivative-use rights independently; do not assume a repository-level software license resolves every scientific-data or checkpoint question.

**Major evaluation issue:** the paper says the released checkpoint was trained exclusively on the official 373,230-sample training split (80% of the 466,537 unique geometry–spectrum pairs), with separate validation (46,654) and test (46,653) splits. The release README describes its 10k subset as randomly sampled from the full dataset, but the release materials reviewed here do not establish that each released row retains an auditable official split assignment. Therefore, do **not** create a new random split of the 10k subset and call it an official held-out evaluation: it could mix training-exposed rows with validation/test rows. First establish the subset's split membership by matching exact geometry/data identities to the official split or obtain the official evaluation artifacts. Until then the 10k set is suitable for schema/inference plumbing and carefully labelled exploratory work, not a claim of official test reproduction. A from-scratch 10k-split experiment may compare methods within that small sample, but it is not a reproduction of the published full-data MOCLIP benchmark.

The NeurIPS nanophotonic-structures code repository is MIT-licensed and links public data on Hugging Face, but its dependency path requires Meep and associated scientific tooling. Audit exact dataset/simulator tasks before mixing its simulated labels with experimentally measured MOCLIP spectra; these have different evidence classes and domains.

## 3. Research question

**Can a selector built on Symthaea's actual BinaryHV primitives improve label-acquisition efficiency for nanophotonic spectra under an equal oracle budget, compared fairly with Euclidean maximin and task-native baselines?**

The research-only selector is implemented in [PR #7628](https://github.com/Luminous-Dynamics/symthaea/pull/7628): `random`, `maximin` in normalized descriptor space, and `hdc_maximin` using Symthaea core's 16,384-bit `BinaryHV`, a deterministic 256-level scalar codebook, role binding, bundling, and normalized Hamming distance. This makes it an actual Symthaea-HDC policy hypothesis rather than a renamed Euclidean heuristic. It is not a spectrum predictor or direct target-spectrum optimizer.

Separate two scientific questions:
- **Published-model reproduction:** reproduce MOCLIP retrieval/inverse-design metrics on the official split and exact protocol.
- **Budgeted active acquisition:** select which geometry-spectrum labels to acquire from the official training pool, then train the same downstream surrogate from scratch on each arm's selected labels, and evaluate on a sealed official test set. Only geometry-derived features may reach the selector before acquisition. A pretrained model exposed to candidates or labels used to claim label efficiency is ineligible for that primary comparison unless its exposure is demonstrably independent.

The HDC metric is a descriptor-space selection score, not optical performance, a calibrated uncertainty, or a claim that HDC preserves Maxwell physics. MOCLIP retrieval and adjoint/full-wave methods remain the task-native comparison; Symthaea must demonstrate added value against these and the Euclidean/random controls.

## 4. Staged benchmark plan

### N0 — Rights, artifacts, and reproducibility

1. Pin exact source commit, data file and checkpoint digests; record whether the Git blobs can be downloaded and verified locally.
2. Resolve explicit dataset/checkpoint terms and attribution requirements.
3. Inspect NPZ key names, array shapes, dtype, null/non-finite values, geometry encoding, spectrum wavelength grid, polarization order and normalization. Never infer units or spectral semantics from array shape alone.
4. Determine whether the sample subset includes replica groups, substrate/run identifiers, or published split labels; if not, mark these lineage fields unavailable.
5. Freeze the execution environment, package hashes, exact checkpoint/config, device, deterministic settings, CPU/GPU fallback and time/memory ceiling.
6. Test the released example without changing outputs; report any dependency/version adjustments separately.

**Exit:** provenance, terms, schema and exact input/output semantics are resolved enough to run an auditable smoke test. Otherwise stop at an explicit blocker.

### N1 — Artifact and inference smoke test

Targets:
- 100% of retained rows have accounted-for source identity and schema disposition (valid, invalid, duplicate/replica, or excluded with reason).
- Zero silent NaN/Inf replacement; every excluded row is accounted for.
- The provided example inference completes with finite embeddings/scores on a compatible environment.
- Record p50/p95 inference latency, throughput, peak RAM/VRAM and device. Compare only under exactly declared batch size, dtype, resolution and hardware.
- Run one exact repeat to check numerical reproducibility with fixed input/config, reporting tolerances rather than demanding bitwise equality across different hardware.

This phase validates plumbing and source understanding only. It does not establish accuracy.

### N2 — Reproduce MOCLIP on its official test contract

Preferred if the full evaluation set/split artifacts and rights are available:
- Use the exact published train/validation/test split, model checkpoint, pre/postprocessing, and candidate library semantics.
- Recompute Top-1/5/10 retrieval from raw prediction receipts and verify the number of test queries and candidate pool.
- Recompute inverse-design spectrum MSE quantiles and the fraction below 1e-2 for the stated probe-library sizes.
- Match the paper-reported values within predeclared reproduction tolerances; report any deviation and do not present the small 10k random subset as the full test set.

**Paper context target, not a new guarantee:** reproduce approximately 0.75 Top-1, 0.95 Top-5, 0.98 Top-10, and 95% of predictions below MSE 1e-2 for a probe set above 10,000 candidates, only when the exact published protocol is available. These correspond to author-reported results, not independently achieved milestones.

### N3 — Fair budgeted-acquisition experiment (separate from MOCLIP reproduction)

The selector has no target-spectrum input by design, so do **not** report its output as a direct target-specific inverse-design ranking. Test it as a policy for selecting which candidate geometries should receive an expensive simulated/measured spectrum label.

Prerequisites:
- map the public subset's exact geometry identities to the official training/validation/test split or obtain the official split artifacts;
- use only the official **training split** as the candidate acquisition pool, official validation only for policy/model tuning, and keep the official test split sealed;
- freeze geometry-only features, feature order/units, query budget, candidate pool, repeated seeds, downstream surrogate architecture/training recipe, and the primary metric before test evaluation;
- train the same downstream surrogate **from scratch** on each arm's acquired labels; do not use the MOCLIP full-data checkpoint to prove label efficiency on rows it has already seen;
- the selector receives only allowed geometry-only descriptors (or a geometry-only embedding whose checkpoint exposure is audited), IDs and the budget; it never sees spectra, target outcomes, validation/test membership, or unacquired labels;
- a masked offline oracle may reveal a selected row's spectrum after selection. The oracle lookup itself is not available to the selection policy before the query.

**Acquisition arms (same labeled-example/oracle budget):**
1. seeded random;
2. Euclidean descriptor maximin (`maximin`);
3. Symthaea BinaryHV HDC maximin (`hdc_maximin`);
4. one task-native acquisition baseline, such as uncertainty sampling from a surrogate trained only on already acquired labels, if it can be implemented without leaking hidden labels;
5. an HDC ablation (same pipeline with Euclidean distance) and a feature/selection negative control.

MOCLIP's published retrieval/optimization pipeline belongs in the separate N2 model-reproduction track. It should not be silently mixed into the acquisition arms unless a model's prior training exposure is consistent with the explicitly stated claim.

**Primary metric (choose one before testing):**
- **Label-efficiency curve:** held-out official-test spectral MSE of the *same from-scratch downstream surrogate* versus number of acquired training labels; report area under the error-vs-label-budget curve and final test error at fixed budgets.
- **Downstream inverse-design Success@K (secondary):** apply the exact same inverse-design/search procedure to the surrogates from each acquisition arm and count how often the best candidate meets the predeclared spectral-MSE threshold under independent full-wave re-evaluation. The downstream optimizer must be identical across arms.

Secondary metrics: wavelength- and polarization-wise residuals; candidate-family and geometry coverage; duplicate rate; failed/invalid queries; abstention/uncertainty coverage if calibrated; query cost, selector latency, peak RAM/VRAM, and full-wave calls. Use paired/group-aware confidence intervals over independent test geometries and repeat acquisition seeds.

Do not treat exact-pair retrieval accuracy as functional success when different geometries can produce equivalent spectra. For the functional task, success is defined by the target response and hard design constraints, with every finalist reevaluated by the same forward model.

**Provisional improvement target:** versus the strongest eligible acquisition baseline at equal acquired-label count, seek at least 20% lower median held-out spectral MSE at a preregistered label budget, or at least +5 percentage points in downstream Success@K, with a paired/cluster-aware 95% interval excluding no improvement. These are proposed pilot go/no-go targets, not universal scientific standards. Choose one primary metric after data variance/power and feasibility are inspected, before opening the held-out test; never change the primary metric after seeing outcomes.

A valid null result is “no demonstrated HDC acquisition benefit over the random or Euclidean maximin control.”

### N4 — Fabrication robustness bridge

Computational or retrieval success is not fabricated optical performance. Before claiming physical utility:
- Re-simulate candidate designs with a verified electromagnetic solver and grid-convergence study.
- Perturb geometry using the measured fabrication error envelope where evidence supports it; report response distributions, not only nominal results.
- Match wavelength grid, polarization, substrate, illumination/collection geometry, measurement calibration and normalization.
- If physical validation is proposed, secure human/laboratory review, sample traceability, calibrated measurement, independent reference samples and repeat measurements.
- Compare predicted-to-measured spectra with uncertainty and maintain failed/ambiguous observations.

The paper's reported grid-convergence exercise found MSE around 1e-4 at a 5 nm grid compared with a 2.5 nm reference for their particular solver/configuration, about an order of magnitude below their measurement error, and reported about three minutes per full simulation in a 96-CPU-core HPC configuration. This is an example of a *paper-specific* numerical validation study; do not port its grid size/runtime as a universal default.

## 5. Benchmark ladder and targets for Symthaea

| Gate | Requirement to pass | What it earns—not more |
|---|---|---|
| **P0: source integrity** | Pinned identities, rights review, valid schema and exact preprocessing | Permission to run smoke tests |
| **P1: execution** | Example inference completes; finite values; reproducible receipts; resource envelope measured | Plumbing verified |
| **P2: baseline reproduction** | Recomputed metrics on the exact official evaluation contract fall within a predeclared tolerance or discrepancy is fully characterized | Baseline is usable for comparisons |
| **P3: fair policy advantage** | Frozen primary metric improves by the predeclared margin; CI excludes neutral effect; no leakage; equal budgets; all failures counted | Evidence that this particular policy improves this particular task |
| **P4: out-of-distribution robustness** | Performance remains above a preregistered utility floor across held-out geometry families, substrate/measurement domains or permitted operating conditions | Limited robustness claim within those domains |
| **P5: physical validation** | Independent solver re-evaluation; robustness checks; then expert-authorized fabrication and calibrated optical measurement if feasible | Evidence for a tested physical design, not a general nanotechnology capability |

### Reproducibility is itself a benchmark

A 2026 reproducibility analysis of MatBench found that using original dependency metadata, only 2 of 28 submitted environments installed and passed import checks, and only 13 of 28 submissions completed a selected task and generated outputs after remediation. This warns against treating source publication or queued CI as reproducibility. See https://cname.oaepublish.com/articles/aiagent.2026.31.

Accordingly, report **environment-instantiation rate, test/contract pass rate, metric-regeneration rate, and artifact completeness** alongside scientific accuracy. Goal for this pilot: 100% of claimed scored runs must be reconstructible from a pinned manifest and output receipt; failed-to-reproduce runs stay in the denominator.

## 6. Constraints for implementation

- Keep production Symthaea Rust-first. MOCLIP and the NeurIPS toolchain are Python research software; use an isolated, pinned sidecar and a small versioned JSONL interface only if a later implementation justifies it.
- The release's reported RTX 4090 training/evaluation hardware is materially more powerful than the user's RTX 2070 Laptop GPU. Begin with inference-only or a CPU/low-memory smoke test; measure actual VRAM use before assuming the model fits.
- The full 466,537-pair dataset is not included in the small release; do not silently substitute the 10k subset and claim comparable evaluation.
- Do not call this a “Symthaea nanophotonics model” unless an identified Symthaea component is actually tested; do not infer spectral values from HDC distance alone.
- Do not add new production ontology/types without evidence current canonical owners cannot represent the semantics.
- No autonomous fabrication, optical-lab actuation, biological work, environmental release or safety-critical deployment is authorized by this computational pilot.

## 7. Exact external identities captured during source audit

### MOCLIP release at `776d9dbce9338175b9edefed1442cfb1f918eab1`

| Artifact | Git blob SHA | Source-level size |
|---|---|---:|
| README | `39d141f498ecbb58c05210f5c0a28c4884059551` | 2,639 bytes |
| LICENSE | `06582980588a0bc62a1bf282624006bbc6e3ac90` | 1,080 bytes |
| `data/MOCLIP_dataset_10k.npz` | `451af237eaed4966b009554529dfe84ba112b2b6` | 8,686,229 bytes |
| `src/models/config.yaml` | `0f2bdc6ac9b3192f5a6c1fab46d6aeb145550013` | 492 bytes |
| `src/models/weights/moclip_v1.pth` | `efcbd0758f97e804e5a7939b7c5b15dccaa0ebc7` | 47,515,250 bytes |

Links: [README](https://github.com/RodionovSA/MOCLIP_release/blob/776d9dbce9338175b9edefed1442cfb1f918eab1/README.md), [LICENSE](https://github.com/RodionovSA/MOCLIP_release/blob/776d9dbce9338175b9edefed1442cfb1f918eab1/LICENSE), [10k dataset](https://github.com/RodionovSA/MOCLIP_release/blob/776d9dbce9338175b9edefed1442cfb1f918eab1/data/MOCLIP_dataset_10k.npz), [pretrained weights](https://github.com/RodionovSA/MOCLIP_release/blob/776d9dbce9338175b9edefed1442cfb1f918eab1/src/models/weights/moclip_v1.pth).

### NeurIPS nanophotonic benchmark at `b877feffc4be737a12397fe3ad881a498c2409af`

- Repository license: MIT, LICENSE blob `2d0c0925e0fdf0d92be0c362f803eac32972327a`.
- Official repository describes dataset generation, surrogate training/testing, and discrete/continuous structure-optimization scripts; it says Meep must be installed first and links datasets hosted on Hugging Face.
- Repo code is not a substitute for task-specific audit of the Hugging Face data snapshot, data terms and simulator/model provenance.

Source: https://github.com/jungtaekkim/nanophotonic-structures/tree/b877feffc4be737a12397fe3ad881a498c2409af

## 8. Current execution status

**Completed in this work:** current literature scan and source-level audit of the MOCLIP release, metrics reported in the paper, the small public subset/model file identities, the older NeurIPS benchmark tooling, and the reproducibility evidence. This scorecard is now specific enough for implementation.

**Not executed:** no binaries were downloaded, no source was checked out, no weights were loaded, no dependencies installed, no benchmark/test split was run, and no GPU capability was measured. This must not be described as a passed smoke test or achieved metric.

**Next action:** NANO-002 should treat this as a strong candidate and decide the source/rights/runtime gates against 2D materials and metrology. If selected, proceed with P0/P1 immediately on a local/pinned environment without waiting for unrelated Symthaea runners; keep CI evidence exact-head and do not claim a pass while workflows are skipped or queued.
