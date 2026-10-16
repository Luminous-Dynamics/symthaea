# NANO Closed-Loop Benchmark Protocol: Symthaea × MADE

**Status:** Proposed benchmark protocol; no experiment has been run and no scientific advantage is claimed.  
**Date:** 2026-10-15  
**Strategy:** [Paradigm-shift roadmap](./NANOMATERIALS_PARADIGM_SHIFT_ROADMAP.md)  
**Portfolio:** [Nanotechnology frontier portfolio](./NANOTECHNOLOGY_FRONTIER_PORTFOLIO.md)  
**OC22 track:** [Evidence-bounded research program](./NANOMATERIALS_RESEARCH_PROGRAM.md)  
**Tracking issue:** [NANO-003](https://github.com/Luminous-Dynamics/symthaea/issues/7519) (closed-loop policy evaluation)
**Run manifest contract:** [NANOTECHNOLOGY_CLOSED_LOOP_RUN_MANIFEST.schema.json](./NANOTECHNOLOGY_CLOSED_LOOP_RUN_MANIFEST.schema.json)
**Rust selector baselines:** [PR #7568](https://github.com/Luminous-Dynamics/symthaea/pull/7568), tracked by [NANO-005](https://github.com/Luminous-Dynamics/symthaea/issues/7570). It now includes seeded random, farthest-first, family-balanced, and cost-normalized farthest-first. These are batch-selection heuristics—not yet a sequential label-updating active-learning loop—and no Rust tests have passed yet.

## 1. Decision

Evaluate whether Symthaea contributes value to *sequential evidence selection* by testing against a public closed-loop materials-discovery benchmark before attempting to generalize its policy to additional nanotechnology domains.

Candidate framework: [MADE — Materials Discovery Environments](https://github.com/diffractivelabs/MADE), accompanying the ICML 2026 paper [MADE: Benchmark Environments for Closed-Loop Materials Discovery](https://proceedings.mlr.press/v306/malik26a.html).

MADE is specifically designed to simulate campaigns in which an agent proposes compounds, receives bounded-oracle feedback, and refines its search. It represents phase-diagram discovery as finding thermodynamically stable compounds relative to a convex hull and supports interchangeable agents, environments and oracles. The paper reports that adaptive planning becomes more important as search spaces scale. This makes it a useful external methodological benchmark for discovery strategy.

**Critical scope boundary:** MADE's bulk-compound stability/convex-hull task is not OC22 IS2RE, adsorption/catalysis, nanophotonics, membrane performance, toxicology, or experimental synthesis. Results may support only a claim about the tested MADE environment and policy. They cannot establish atomistic accuracy, new-material discovery in the lab, or transfer to another domain.

## 2. Why this is a useful next step

The current OC22 research remains blocked on obtaining and validating actual source bytes, proving the initial-input/relaxed-target mapping, reconciling official splits, and reproducing a task-matched baseline. We should not bypass those gates.

A separate closed-loop benchmark can answer a narrower question: **Does a Symthaea-derived selection policy improve the use of a fixed oracle budget in an explicit simulated discovery environment?** That is a different hypothesis, and MADE offers a pre-existing benchmark interface and published comparison baselines instead of requiring a new environment to be invented.

The external repository describes:
- a convex-hull discovery environment;
- analytical, ORB and MACE oracles;
- modular agents comprising generators, planners, scorers and filters;
- published baseline configurations, including random and Chemeleon-generative strategies;
- trajectory and summary outputs for reproducible comparisons;
- an MIT code license.

The repository currently declares Python >=3.12 and a substantial dependency set (ASE, MACE-Torch, ORB models, Materials Project API, and optional remote experiment/LLM/logging tools among them). We should therefore **reuse it as an isolated research benchmark, not adopt it as a Symthaea production dependency**. The source audit pinned the current upstream review snapshot to commit `bffda0c9a1fa1904b35c0ef67b455881c975df86` and recorded key blob SHAs below. Use the immutable commit in experiment configs; never cite the moving main branch as the run identity.

### Initial immutable source pin

The public MADE repository's current main ref was resolved during the source audit to commit **bffda0c9a1fa1904b35c0ef67b455881c975df86** (commit date reported by GitHub: 2026-05-08T15:21:48Z). Treat this as the review snapshot only; confirm the exact upstream commit intended for execution before a scored run.

| Source component | Pinned path/blob |
|---|---|
| Code license | [LICENSE](https://github.com/diffractivelabs/MADE/blob/bffda0c9a1fa1904b35c0ef67b455881c975df86/LICENSE), blob `3ba6d9061dbfc90f7ef75751a6e40f79b21feac3` |
| Dependency metadata | [pyproject.toml](https://github.com/diffractivelabs/MADE/blob/bffda0c9a1fa1904b35c0ef67b455881c975df86/pyproject.toml), blob `527f5fd85e12087eb43ba91fea467abc72c54252` |
| Main metrics module | [metrics.py](https://github.com/diffractivelabs/MADE/blob/bffda0c9a1fa1904b35c0ef67b455881c975df86/src/made/evaluation/metrics.py), blob `d102ffe14f9bfd85790101dd4f359adadba31d5e` |
| Random-generator baseline config | [random_generator_baseline.yaml](https://github.com/diffractivelabs/MADE/blob/bffda0c9a1fa1904b35c0ef67b455881c975df86/configs/agent/random_generator_baseline.yaml), blob `bfc0a29324c352d794fbac4723e78b18286bc1ac` |
| Analytic oracle config | [analytic.yaml](https://github.com/diffractivelabs/MADE/blob/bffda0c9a1fa1904b35c0ef67b455881c975df86/configs/oracle/analytic.yaml), blob `0c77cf0197293f6f20c3ca0c2a803144441d2cdc` |

A source-level review confirms that the metric implementation has code for area under the discovery curve (cumulative discoveries versus query count, integrated trapezoidally), enhancement factor (proposal discoveries relative to baseline at matched query points, with an implementation cap), and acceleration factor (relative queries needed to reach a target discovery count). **Do not assume the README's listed mSUN metric is implemented by the same metrics module**: search the pinned runner/notebooks and record the exact calculation site before using or reporting it.

The machine-readable [run-manifest schema](./NANOTECHNOLOGY_CLOSED_LOOP_RUN_MANIFEST.schema.json) describes the required experiment identity, query accounting, controls and independent-check receipt. Its JSON syntax was re-parsed after commit; full JSON Schema conformance has not yet been run through an independent validator.

These are immutable review identifiers, not a declaration that dependencies installed, artifacts downloaded, or any benchmark ran successfully.

## 3. Hypothesis and null

**H1:** At equal candidate-oracle query budgets, a fixed Symthaea-assisted policy achieves better discovery utility/efficiency than the strongest applicable non-Symthaea baseline over held-out systems, with no invalid queries or hidden-information access.

**H0:** The policy does not improve the preregistered metric enough to exceed a stated practical margin, or any apparent gain disappears under matched budgets, repeated episodes, or controls.

A null/inconclusive result is valid. We do not tune against the final evaluation episodes and then report those episodes as held out.

## 4. Staged execution plan

### Phase 0 — Freeze and audit the external benchmark

Before running:
1. Pin the MADE Git commit SHA, repository URL, MIT license text/revision, and the ICML 2026 paper citation.
2. Audit the exact baseline, environment, oracle and metric implementations used by that revision; retain links/line references in the run manifest.
3. Freeze source-data/system files by cryptographic digest, source/citation, and license/terms for data and every pretrained model separately. The MIT code license does not license third-party datasets or weights.
4. Record the full resolved configuration and dependency lock, Python/OS/compiler settings, device, inference fallback, thread count, resource cap and environment variables.
5. Disable optional services not needed for local evaluation (cloud orchestration, tracking services, and LLM APIs) unless a named experiment explicitly requires them. Never embed secrets in tracked configs or receipts.
6. Confirm that the selected oracle returns finite energies and consistent units/semantics for all eligible structures. Treat oracle failure as failure/incomplete—not as a free skipped query.

**Phase 0 exit:** a reproducible source/data/model/config inventory exists and licensing is reviewed for the proposed use. No performance claim is allowed yet.

### Phase 1 — Smoke test the benchmark protocol (not a science result)

Use the smallest published local configuration as a plumbing test (the repository README shows a Li–O system, random agent, ORB oracle, three episodes and five queries). Freeze the exact config and expected output structure.

Check:
- run starts from a clean environment and finishes under an explicit time/memory cap;
- the oracle-query budget is enforced exactly;
- episode seeds and candidate identities are recorded;
- all queries, rejected/duplicate/invalid proposals, failures and returned observations are accounted for;
- trajectory and summary files can be parsed and their headline metrics independently recomputed.

A three-episode/five-query smoke test is not adequately powered evidence of superiority and must not be reported as such. If ORB needs more resources than available, first use the framework's analytic oracle to test protocol plumbing, clearly labelled **synthetic/environment test only**, then assess whether a physical ML oracle is feasible. Do not compare analytic-oracle metrics with ORB/MACE results.

### Phase 2 — Reproduce published baselines

Run the applicable upstream baseline configs with the same systems, oracle, budget, environment and scoring implementation. At minimum include random search; include the repository's published generative baseline where its requirements and license are met.

Reproduction requires:
- exact run manifests and trajectory receipts;
- per-system and per-episode metrics, not only a grand mean;
- repeated seeds/episodes matching the paper's protocol as closely as compute permits;
- an independent metric recomputation;
- discrepancy resolution before adding a Symthaea arm.

If the original protocol cannot be reproduced under the available hardware/budget, report the exact deviation and treat any result as an adaptation, not a reproduction.

### Phase 3 — Add the implemented Symthaea BinaryHV policy, not an entire cognitive stack

The first implementation is [PR #7628](https://github.com/Luminous-Dynamics/symthaea/pull/7628): a research-only Rust CLI with `random`, Euclidean `maximin`, and `hdc_maximin`. The HDC policy uses the existing `symthaea-core::hdc::binary_hv::BinaryHV` type, a deterministic correlated 256-level scalar codebook, dimension-role binding and bundling, then normalized Hamming distance for greedy diverse selection. It is a real Symthaea representation ablation, but it is not an atomistic model, energy predictor or calibrated uncertainty estimator.

For MADE, define a frozen candidate descriptor schema that is meaningful for the specific discovery environment (for example composition/stoichiometry features that can be computed before an oracle query). The HDC policy's hard bounds are 10,000 candidates and 256 descriptor features. Do not feed it formation energies, energy-above-hull, oracle values, hidden convex-hull membership, or features derived from the unacquired label. Candidate IDs must be opaque and must not encode target status.

Keep the generator, oracle, candidate restrictions, stopping rule and total oracle budget fixed. Compare random, Euclidean maximin, Symthaea HDC maximin, and the strongest applicable task-native baseline. Keep a research-only adapter between MADE and the Rust selector (e.g. a versioned JSON request/response over stdin/stdout); do not add Python to Symthaea production runtime. The adapter must be versioned, enforce the same allowed-input contract and preserve exact input/output receipts.

Allowed inputs: eligible candidate IDs, predeclared unlabeled descriptors, previously revealed observations only when the acquisition contract permits them, remaining budget, and declared constraints. Forbidden inputs include hidden ground-truth labels, unrevealed oracle results, future trajectory fields, evaluation-only metrics, and test membership if that membership would reveal outcomes.

### Phase 4 — Controlled evaluation

Freeze and run:
1. seeded random acquisition;
2. Euclidean descriptor maximin (selector baseline);
3. Symthaea `hdc_maximin` (HDC representation arm);
4. strongest applicable published/task-native non-Symthaea baseline;
5. an HDC ablation using the same pool/budget but Euclidean maximin distance;
6. negative-control policies (e.g. shuffled descriptors/features, leakage canary, or randomized policy where meaningful).

Use held-out chemical systems and repeated episodes. Avoid counting episodes from the same system as independent scientific systems. Evaluate at multiple fixed query budgets when feasible; do not allow one arm more oracle calls or a more expensive oracle.

The upstream paper/repository includes discovery-specific metrics (AF, EF, AUDC and mSUN). At implementation time, freeze the precise formulas, signs, units, tie rules and aggregation exactly from the pinned source revision; do not reconstruct metric definitions from abbreviations or selectively report the metric that flatters the method. Add cost-per-valid-discovery, invalid-query rate, failure/abstention coverage, wall time and peak memory as transparent supplementary metrics where they can be measured reproducibly.

Report mean and dispersion/uncertainty across episodes and systems. Choose a primary metric and practical margin before the final episodes are run. The independent checker must regenerate metrics from raw trajectory receipts.

### Phase 5 — Decision and transfer gate

Proceed only if the policy:
- passes no-leakage and query-budget controls;
- improves the preregistered primary metric by the predeclared useful margin (or clearly states that it does not);
- retains its benefit on held-out systems/episodes;
- does not gain by invalid queries, duplicate loops, hidden-label access, or omitted failures;
- has a reproducible run receipt and independently regenerated score.

Even a successful result validates only the tested policy on the tested MADE task/oracle. It justifies an OC22-specific acquisition experiment after OC22 admission, not immediate transfer to catalysts, membranes, photonics, nanomedicine, or physical labs.

## 5. Resource and implementation boundaries

- The external MADE repository requires Python >=3.12 and several scientific/ML dependencies. Keep this environment isolated and pinned.
- Start with a CPU-feasible smoke test and an explicit small budget. Measure actual resource use before selecting ORB/MACE and scaling up; do not assume the user's RTX 2070 Laptop GPU can host every checkpoint.
- No cloud/Modal execution, API key, external LLM, or paid service is required for the baseline unless a frozen experiment explicitly says otherwise.
- No raw OC22 archive, dataset pair mapping, or scientific validation is supplied by this protocol.
- No automatic synthesis, biology, environmental release, medical, or lab-control use is authorized.
- Do not fork or vendor MADE merely to wrap Symthaea. First prove an external adapter provides value and keep a pinned upstream reference.

## 6. Independent result contract

Each run should preserve:
- MADE repository commit, paper DOI/URL, external code license revision;
- each data/system artifact URL, revision, digest, rights and transformation digest;
- exact oracle class, checkpoint digest, model license and settings;
- resolved experiment config, seed/episode IDs, allowed field contract, query budget;
- each proposed candidate and query status, invalid/rejected/duplicate candidate, observation and cost;
- raw trajectories, metric implementation revision, computed metric table;
- environment/toolchain/device identity and resource measurements;
- independent checker version, results and blockers.

Run status must distinguish not-started, running, completed, failed, incomplete, and invalidated. A completed runner is not equivalent to a passing scientific control.

## 7. Related work and source audit

- Malik et al., [MADE: Benchmark Environments for Closed-Loop Materials Discovery](https://proceedings.mlr.press/v306/malik26a.html), PMLR 306, ICML 2026. The paper formalizes bounded-oracle convex-hull discovery and reports that adaptive planning is increasingly important as search spaces scale.
- Official reference implementation: [diffractivelabs/MADE](https://github.com/diffractivelabs/MADE). README documents modular environments/oracles/agents, local smoke command, baseline scripts, output receipts and MIT code license; pyproject.toml declares Python >=3.12 and its dependency set.
- Han et al., [Artificial intelligence as a driver of sustainable materials and circularity](https://www.nature.com/articles/s41578-026-00912-8), Nature Reviews Materials (27 March 2026): argues that AI-assisted materials work can better integrate performance, life-cycle assessment and recycling across design/use/end-of-life.
- [Autonomous laboratories for sustainable nanomaterials discovery](https://doi.org/10.1016/j.nxmate.2026.102883) (2026): system-level review of closed-loop experimentation, robotics, characterization, digital twins and sustainability, with interoperability and governance challenges.

## 8. Current evidence status

**Completed in this change:** reviewed the published paper and upstream repository documentation and proposed a gated benchmark strategy.

**Not completed:** no MADE repository checkout, environment installation, benchmark run, model download, metric reproduction, Symthaea adapter, or GPU feasibility test was performed in this step. Do not report it as executed. The next implementation work begins with the Phase 0 pinned-source audit and Phase 1 smoke test.

**Bottom line:** this is an inexpensive way to test whether Symthaea improves the process of discovery while OC22 validation continues, without conflating convex-hull search with catalyst-surface prediction.
