# ROB-BENCHMARK-PROBE-CONTROLS-001

Status: first executable shortcut-solvability and privileged-input diagnostic. This is a deterministic Rust reference over caller-supplied arm manifests and per-cluster outcome summaries. It is not a benchmark runner, model trainer, statistical inference package, safety authority, or physical qualification.

Parent: [Symthaea #11775 — ROB-BENCHMARK-VALIDITY-001](https://github.com/Luminous-Dynamics/symthaea/issues/11775). The paired independent-cluster significance evaluator remains separately owned by [PR #11807](https://github.com/Luminous-Dynamics/symthaea/pull/11807). This crate deliberately emits no p-value, confidence interval, SOTA ranking, or safety certificate.

Research anchor: Jiang et al., *What Are We Actually Benchmarking in Robot Manipulation?*, 2 June 2026, arXiv:2606.04233. The paper identifies shortcut solvability, weak statistical significance, creeping overfit, and data-source dependence as distinct benchmark-validity failure modes. It reports that a 0.09B probe without a language encoder approached reported LIBERO SOTA under its audited setup. That is evidence to test benchmark designs, not a universal conclusion that all LIBERO results are invalid.

## Research question

Can a deliberately low-capacity policy, using only one or a few declared input families, achieve safe-success performance close to the headline candidate on the same frozen hidden evaluation set? Separately, can evaluator-only or identity/seed inputs expose leakage that makes a nominal task result appear easier than the intended capability?

The diagnostic evaluates these probe families:

- constant / majority-class;
- single-feature categorical lookup;
- capacity-bounded small linear model;
- shallow decision tree;
- input ablations over task metadata, language instruction, visual observation, robot state, action history, environment metadata and explicitly privileged channels.

The crate consumes outcomes that have already been produced by the probe and candidate policies. It does not train or validate those models.

## Frozen manifest and split contract

Each arm binds:
- model artifact, policy, harness and architecture-audit digests;
- task, scenario, robot configuration, evaluator and resource-budget digests;
- separate training, calibration and hidden-evaluation split digests;
- a declared input-family profile and model family.

Training, calibration and hidden evaluation digests must be pairwise distinct for each arm. Every arm must use the exact hidden-evaluation split and the same task, scenario, robot, evaluator and resource-budget scope. The candidate cannot declare evaluator state, ground-truth labels, episode identity or environment seed as input.

The architecture-audit digest is a binding to an external capacity attestation. This crate checks that the attestation identifier is present and that the declared model family is one of the supported low-capacity probe families; it cannot establish that the artifact truly has that capacity.

## Paired cluster design

Every arm supplies exactly one outcome record per frozen independent cluster. Candidate and probe records must bind the same:
- cluster ID;
- paired-episode manifest digest;
- independence receipt digest and disposition;
- episode count.

Episode outcomes are mutually exclusive and must sum to the scheduled episode denominator. Safe success is a subset of task success. Failures and timeouts remain in the denominator. Unknown and censored outcomes remain visible but prevent a confirmatory probe signal. Rejected independence evidence rejects the input; Unknown independence makes the audit inconclusive.

Control ticks are retained for descriptive reporting only. They never increase the number of independent clusters.

## Signal definitions

For each qualified cluster, calculate:

    gap = candidate safe-success rate - probe safe-success rate

Rates use all scheduled episodes in that cluster and integer parts per million. The evaluator reports the lower and upper middle order statistics of the paired cluster gaps; these are descriptive median endpoints, not a confidence interval.

A signal is emitted only when:
1. all declared clusters have Qualified independence receipts;
2. no candidate/probe Unknown or censored outcomes are present;
3. the qualified cluster count meets the predeclared minimum; and
4. the upper middle cluster-gap statistic is no greater than the frozen near-candidate threshold.

A near-candidate probe with non-privileged inputs emits ShortcutSolvabilitySignal. A near-candidate probe using an evaluator-only, ground-truth-label, episode-identity or environment-seed input emits PrivilegedInputLeakageSignal. If all probes are evaluable and no signal is observed, the result is NoKnownShortcutSignal. Missingness or unqualified independence yields Inconclusive.

This is a screening diagnostic. A signal calls for benchmark investigation and does not, by itself, prove invalidity. Absence of a signal does not prove that no shortcut exists.

## Claim ceiling and next diagnostics

This reference does not implement hidden-split custody, repeated-submission/selection accounting, source/site/operator/morphology holdout, contamination discovery, probe training, or a complete audit of candidate-visible feature provenance. Those remain separate obligations under #11775.

It does not prove that declared input families match the policy's actual runtime inputs, that the architecture attestation is truthful, that clusters are independent, or that the evaluator's safety specifications are complete. No physical robot result or SOTA-superiority claim follows from a passing software test.

## Running the reference

The crate is dependency-free and isolated from the application workspace:

    cargo fmt --manifest-path research/rob-benchmark-probe-controls-001/reference/Cargo.toml -- --check
    cargo test --locked --manifest-path research/rob-benchmark-probe-controls-001/reference/Cargo.toml
    cargo clippy --locked --manifest-path research/rob-benchmark-probe-controls-001/reference/Cargo.toml --all-targets -- -D warnings

The exact-head Rust 1.96 GitHub Actions workflow is the qualification gate. Authored tests are not passes until the dedicated workflow succeeds.
