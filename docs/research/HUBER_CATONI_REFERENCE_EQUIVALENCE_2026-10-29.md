# Huber–Catoni reference-equivalence gate

**Date:** 2026-10-29  
**Status:** benchmark harness added; upstream numerical comparison not yet executed in this environment.

## Why this is the next gate

The current independent implementation of Wang & Ramdas's Huber–Catoni equations is useful as a derivation, but equation-level unit tests cannot establish that it faithfully implements the publication. The next test compares its interval endpoints to the authors' implementation on identical, deterministic sample prefixes.

Paper: https://proceedings.mlr.press/v206/wang23p.html  
Reference implementation: https://github.com/HJ-Wg/catoni-confseq  
Reference repository HEAD checked for this report: 381d7fb17d2af5f4cc78d422c92900f2b1d5d1c4.

## What was added

- validation/crosscheck_huber_catoni_reference.py: optional differential harness that imports the upstream checkout only from an explicitly supplied local directory. It does not vendor upstream source or link it into production code.
- validation/test_huber_catoni_reference_crosscheck.py: five tests for support intersection, non-finite mismatches, endpoint-error reporting, deterministic streams, and bounded test inputs.
- This report defines the numerical gate and its validation boundary.

The harness currently compares positive contamination levels only. It uses independent one-sided upstream confidence sequences at alpha / 2 for the lower and upper endpoints, making the tail allocation explicit rather than depending on the upstream two-sided wrapper convention. It intersects upstream intervals with the known clean-mean support [0, 1], and compares the result with the independent equations at fixed prefixes up to 128 observations.

The tolerance is 5e-4 absolute endpoint error. Any endpoint mismatch beyond tolerance makes the run fail. The JSON result records the upstream Git commit, SHA-256 of the upstream files, exact checkpoints, parameter grid, implementation-source digest, and every comparison. The tolerance is a numerical root-solver allowance, not a statistical error budget.

## Reproduce without modifying the upstream repository

Use a separate checkout; do not copy the GPL-3.0 source into Symthaea:

    git clone https://github.com/HJ-Wg/catoni-confseq.git /tmp/catoni-confseq
    PYTHONPATH=validation python validation/test_huber_catoni_reference_crosscheck.py
    PYTHONPATH=validation python validation/crosscheck_huber_catoni_reference.py \
      --reference-dir /tmp/catoni-confseq \
      --output /tmp/huber-catoni-reference-crosscheck.json

The upstream codebase is GPL-3.0. This harness is an optional evaluator for an external checkout, not a decision to incorporate that implementation into the Symthaea runtime. Any future linking, copying, or packaging decision needs a separate license review.

## Acceptance gates

1. Equation agreement: all reported endpoints differ by at most 5e-4 after the same support intersection. A mismatch is a blocker, not something to absorb by widening the tolerance after observing results.
2. Exact revision evidence: record the reference commit and both upstream source digests in the raw JSON; reviewers must be able to reproduce the run from those bytes.
3. Assumption validity is separate: passing this check establishes implementation agreement only. It does not show coverage when the claimed contamination bound or clean central-moment bound is false.
4. Then stress validity: test actual versus declared contamination, moment-bound understatement, and contamination beyond the method's breakdown condition in separate panels. Report unsafe admissions, false rejections, unresolved outcomes, time-uniform clean-mean coverage, interval width, and observations to decision.
5. Then compare efficiency: only compare observation cost on the subset of conditions where each method's assumptions hold and its safety gate is satisfied. Keep coverage and availability as first-class metrics; do not count unsafe early admission as an efficiency win.

## Comparator map for the next phase

- **Wang & Ramdas (AISTATS 2023):** closest implementation-level baseline for anytime mean inference with a stated Huber contamination cap and a clean-distribution central-moment bound. First validate endpoint equivalence before comparing performance. https://proceedings.mlr.press/v206/wang23p.html
- **Zhou & Liu (AAAI 2026), On the Robustness of Bandit Multiple Testing:** strongest next allocation-level comparator if Symthaea adaptively allocates observations across risk cells. It targets anytime false-discovery-rate control and sample-efficient power under Huber-style corruption, so compare at matched FDR/unsafe-admission budgets rather than only average observations. It is related, not a drop-in replacement for the current fixed-cell decision certificate. https://ojs.aaai.org/index.php/AAAI/article/view/40148
- **Xie, Gao & Lafferty (Biometrika, online 28 Sep 2026), Confidence Intervals for Linear Models with Arbitrary Noise Contamination:** relevant to the larger assumption-integrity question because its structured regression method does not require a prior contamination fraction. The paper also notes non-identifiability of the contamination proportion from data alone. Its Z-estimation/linear-model scope means it is a future structured comparator, not a direct replacement for arbitrary multi-cell Bernoulli certification. https://academic.oup.com/biomet/advance-article-abstract/doi/10.1093/biomet/asag058/8842600
- **Industry reference:** Adobe documents anytime-valid confidence sequences for continuous experimentation and multiple-comparison handling. This is a useful operational baseline for monitoring/stopping semantics, but its public documentation is not a Huber-misspecification guarantee; do not equate an industrial sequential-testing workflow with contamination robustness. https://experienceleague.adobe.com/en/docs/analytics-platform/using/cja-workspace/panels/experimentation

The competitive target is a layered guarantee: (1) verified method implementation, (2) valid or explicitly bounded assumptions, (3) familywise/FDR-correct adaptive decisions, and (4) a useful safety–availability–cost frontier. No single source above provides all four for this exact task. Our contribution should be demonstrated by matched experiments, not inferred from combining citations.

## ## Important limitation

The reference checkout could be inspected through the public repository connector, but this environment could not clone GitHub into a local execution environment. Consequently, no upstream-vs-independent numerical comparison or test pass is claimed yet. The harness and its tests are committed as runnable work; the first result JSON must be generated on a machine with Python, NumPy, the Symthaea validation files, and the separately cloned upstream reference.
