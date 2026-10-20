# First reference campaign: supported Cu nanoparticles on alumina

**Campaign ID:** `atomfab-supported-cu-alumina-reference-v1`  
**Status:** Proposed and unexecuted; not a benchmark result.

## Why this task first?

The published 2026 benchmark directly addresses the gap AtomFab now wants to study: how binding-energy accuracy relates to finding structures in global optimization. The paper compares domain-specific DP-UniAlCu with MACE-OMAT and MatterSim v1.0.0-1M for supported Cu nanoparticles and uses GA searches for Cu27, Cu38, Cu47 and Cu55. Its reported search protocol uses a population of 50, 20 generations, random initial structures, cut-and-splice and mutation. [Source: Xu et al., *npj Computational Materials* (2026)](https://doi.org/10.1038/s41524-026-02043-8).

The campaign explicitly freezes 12 reportable strata: Cu27, Cu38, Cu47 and Cu55 × γ-Al2O3(100), γ-Al2O3(110), α-Al2O3(0001). This is important scientifically, not just organizationally: the underlying Cu/Al2O3 research reports that support surface can change the particle-size/mobility relationship and even reverse intuitions based only on zero-temperature binding. [Source: Xu et al., *ACS Nano* (2025)](https://doi.org/10.1021/acsnano.5c04622).

## What is pinned now—and what is not?

A machine-readable design is at `benchmarks/supported-cu-alumina-reference-campaign-v1.json` with its schema in `schemas/supported-cu-alumina-reference-campaign-v1.schema.json`.

The published GA settings above are transcribed from the 2026 paper. The Methods specify rattle and mirror mutations with equal probability. Other settings required for exact reproduction—DFT code/build, pseudopotentials, cutoffs, k-points, relaxation thresholds, structure matching tolerance, exact candidate/reference structure manifest and RNG details—remain explicitly unresolved until extracted from the publication Methods/Supplement and data. No placeholders are being treated as experimental facts.

The related 2025 Zenodo record contains training/validation structures, VASP inputs and global-minimum configurations. It lists `dataset.zip` MD5 `e88745ca2cfd25ef0b152f5fa3ed6bd9` and `global_minima.zip` MD5 `9788d7f13403450552e4ad1642c6a478`. These are source-record MD5 checksums, not our computed SHA-256 digests, and are not evidence that the archive is the 2026 paper's complete independent test set. We must materialize the source, verify these record checksums, compute SHA-256/byte length locally, identify licensing, and freeze the exact evaluation-file manifest before claim-bearing runs. [Zenodo record](https://zenodo.org/records/15865800).

## Paper-compatible budget versus our policy experiment

The reference paper states that it locally DFT-optimizes the 50 lowest-energy structures from each of the three MLIP-generated searches per nanoparticle/surface system. That is **150 DFT-refined candidates per stratum** and 1,800 candidates over twelve strata if all twelve combinations are reproduced. The DFT configuration, exact GA source details, actual candidate evaluation count, failures, and structure manifests must still be pinned before we can claim exact reproduction. The 50-member × 20-generation GA settings alone should not be used to assert exactly 1,000 unique candidate structures; the plan records that as a provisional arithmetic inference only, since deduplication and implementation behavior can change the count. [Paper, Methods](https://doi.org/10.1038/s41524-026-02043-8).

Do not confuse that reproduction budget with our **extension**: compare acquisition policies on 20 paired seeds using the same seed across policies and a fixed per-policy reference budget. Use 150 reference relaxations per stratum only if it is appropriate to the matched decision experiment; choose and freeze another common budget if necessary. Charge MLIP inference, reference relaxations, failed retries, preprocessing/deduplication, router overhead and post-processing. A policy must not look efficient by shifting cost into a different accounting bucket.

## Executable order of operations

1. Reproduce the exact published GA/reference comparison with DP-UniAlCu, MACE-OMAT medium and MatterSim v1.0.0-1M, after pinning checkpoint revisions and terms. Reproduce the reported number of repetitions if the paper specifies it.
2. Preserve the twelve stratum-specific reports. Record model-data overlap where feasible; a Cu27+ result may probe extrapolation for DP-UniAlCu because the paper describes its supported-particle training set as extending only up to Cu21. Do not generalize that training-domain statement to other models without auditing their data.
3. For AtomFab decision-layer comparisons, hold the physics model and proposal pool constant. Compare conventional/random/local search, uncertainty-only, diversity-only, cost-aware selection and full Symthaea at equal reference-calculation budget and matched seeds.
4. Count failed relaxations, invalid starting structures, duplicates and rework costs. The numerator is unique **reference-validated** low-energy discoveries—not the number of generated candidates or a model-predicted minimum.
5. Report Pareto frontier (validated discovery breadth, best validated objective, total cost), prediction errors on matched structures, typed failure modes by stratum, and abstentions separately. Do not roll these into a single weighted score.
6. Begin the policy-effect campaign with at least 20 paired fixed seeds. The proposed stretch gate of median acceleration factor ≥2 and 95% bootstrap lower bound >1 applies only if both policies reach the same frozen objective target and the chosen policy's validated result quality is no worse at equal total cost. This is a target, not an achieved result.


## Historical result fingerprint, not a truth claim

The 2026 paper reports that among its 12 nanoparticle-size/surface cases, DP-UniAlCu produced the lowest-binding-energy structure in eight, MatterSim-v1.0.0-1M in three, and MACE-OMAT in one. The manifest records these counts as a **paper-reported reproduction fingerprint**, not a deterministic acceptance requirement under unknown/different random seeds. If a repeated run differs, preserve the random seed, candidate pool, all failures and reference refinements and compare distributions; do not tune the evaluator to force an 8/3/1 match. [Xu et al. (2026)](https://doi.org/10.1038/s41524-026-02043-8).

The paper itself notes that finite stochastic searches are unlikely to establish true global minima. AtomFab therefore uses “published/reference DFT-refined best-known minimum” unless a separate exhaustive or certifying procedure supports a genuine global-optimum claim. The benchmark objective is to improve the evidence-backed best-known result at matched cost, not to make an unprovable global-optimality claim.


## Ranking metric reproduction requirement

The published paper's comparison should be reproduced with both RBO (p=0.94) and Kendall tau-b on the common set of 150 DFT-refined structures per size/surface stratum, with the same structure identities and pinned reference energies. AtomFab's finite-prefix RBO-style metric is useful for its own large-candidate search analysis but is not a drop-in replacement for the paper's RBO; the new `published_ranking.rs` implements a separate paper-compatible extrapolated RBO and tie-aware Kendall tau-b. Candidate-set Jaccard/unique discovery remains a distinct metric. The paper-reported 8/3/1 best-binding-energy winner count is a historical stochastic-search fingerprint, not a hard result requirement under unreported/different random seeds.

## Decision criteria

A result deserves an industry-comparison claim only after: (a) exact reference protocol reproduction; (b) pinned, license-compatible model/checkpoint/evaluator; (c) comparable task and splits; (d) identical reference-validation budget; (e) total costs and failed runs accounted; (f) independently recomputed metrics from evidence artifacts; and (g) stratum-specific failures remain visible. The 2026 study shows that energy-error rankings and exploration performance can diverge, so neither MAE nor best predicted energy alone can establish superiority. [Benchmark source](https://doi.org/10.1038/s41524-026-02043-8).

No data has been downloaded or model has been run as part of this campaign-plan commit. The plan is ready for reference-source materialization, but its data/checkpoint hashes and exact DFT settings are not yet pinned.


## Paired-seed confidence interval implementation

The policy experiment's stretch gate now has a dedicated Rust implementation in `src/campaign_statistics.rs`. It computes the median seed-level cost ratio and a deterministic percentile-bootstrap interval from 20 or more matched campaign pairs. It refuses an interval if any paired campaign is incomplete, fails to reach the shared target, changes the underlying model/data/evaluator/environment/target/budget, repeats a seed, or uses non-positive/non-finite cost. Bootstrap resample count and random seed are explicit and reproducible; the serialized summary is validated against `schemas/paired-acceleration-summary-v1.schema.json`.

This is stricter than computing an interval over only successful seeds after results are observed. All target failures stay reportable campaign outcomes; the speedup interval is withheld if the entire declared paired cohort does not satisfy the target-reaching rule. A lower confidence bound above one is necessary but not sufficient for the stretch claim—the selected policy must also maintain or improve the validated objective at equal total cost.

No benchmark campaign has been executed by adding this metric kernel. It is report-generation infrastructure awaiting exact data/model pins and external baseline runs.
