# Huber-contamination identifiability frontier: exact counterexample

**Date:** 2026-10-29  
**Status:** exact population-level counterexample; local tests pass; not a deployment result.

## Finding

A robust confidence sequence cannot infer that its declared contamination upper bound is true merely from the same contaminated observations. Under the Bernoulli Huber model,

\[
q=(1-\rho)p+\rho r,\qquad p,r\in[0,1],
\]

where `q` is the observed mean, `p` is the clean mean, `rho` is contamination and `r` is the contaminant mean. If only `rho <= rho_max` is known, the sharp compatible clean-mean envelope is

\[
\left[\max\left(0,\frac{q-\rho_{max}}{1-\rho_{max}}\right),\;
\min\left(1,\frac{q}{1-\rho_{max}}\right)\right].
\]

The interval expresses partial identification, not statistical uncertainty from finite sampling. A confidence sequence for `q` can be mapped through this envelope, but neither component can validate a false `rho_max`.

## Exact witness

Take model risk `m = 0.37` and allowed absolute clean-risk error `0.05`.

- **Safe world:** clean mean `p = 0.37`, contamination `rho = 0`, so observed mean `q = 0.37` and model error is `0`.
- **Unsafe world:** clean mean `p = 0.43`, contaminant mean `r = 0`, and `rho = 1 - 0.37/0.43 = 0.1395348837...`. The observed mean is also exactly `q = 0.37`, while clean model error is `0.06` and therefore unsafe.

These worlds induce the *same observed Bernoulli distribution*, not merely similar sample means. Any procedure that sees only those observations has the same decision distribution in both worlds, because the observable laws are identical. A method may admit, reject, or abstain, but it cannot simultaneously admit the safe world at a high rate and guarantee rejecting/never admitting the unsafe world without additional assumptions or evidence.

For the exact observed mean `q = 0.37`:

| Declared upper bound | Compatible clean mean interval | Exact decision for model `m=0.37`, tolerance `0.05` |
|---:|---:|---|
| 5% (understated) | `[0.336842, 0.389474]` | **Admit** — but excludes the true unsafe mean `0.43` |
| 15% (covers the unsafe world) | `[0.258824, 0.435294]` | **Unresolved** — contains both safe and unsafe worlds |

The first row is a deterministic population-level failure witness. It is not a finite-sample Monte Carlo result. It demonstrates why a false bound can make even the correct sensitivity formula unsafe. The second row illustrates the availability cost of using a sufficiently conservative bound.

## Important distinction: authenticity is not validity

A signature, source digest or provenance receipt can establish who made a statement and whether it changed. It does not prove that the contamination bound the statement contains is true. Therefore the contract must keep separate fields for:

1. source authentication, identity, scope and expiry;
2. the evidence or process that supports the numerical upper bound;
3. the probability that this upper-bound event fails (if it is probabilistic);
4. the confidence-sequence error budget conditional on the bound holding.

If a bound is independently justified only with failure probability `delta_bound`, and the sequential statistical guarantee has failure probability `delta_stat`, a conservative union-bound composition is `delta_total <= delta_bound + delta_stat`. Additional probabilistic assumption gates also consume error budget. This arithmetic is valid only if the component claims are themselves valid; plugging an unsupported number into `delta_bound` does not create a guarantee. Deterministic, independently verified preconditions can instead be handled as gates, not as arbitrary confidence scores.

The code deliberately labels its illustrative attestation confidence as **not verified here**. It is a composition example, not an attestation verifier.

## Files and reproduction

- `validation/huber_identifiability_frontier.py`: exact witness, clean-mean envelope, fail-closed preflight and illustrative error-budget composition.
- `validation/test_huber_identifiability_frontier.py`: nine tests covering the indistinguishable worlds, understated-bound unsafe admission, valid-bound abstention, invalid provenance, error-budget composition, sharp envelope and input validation.
- `validation/HUBER_IDENTIFIABILITY_WITNESS_2026-10-29.json`: generated structured witness and exact decision outcomes.

Run:

```sh
PYTHONPATH=validation python -m unittest -v test_huber_identifiability_frontier.py
PYTHONPATH=validation python huber_identifiability_frontier.py \
  --output /tmp/huber-identifiability-witness.json
```

Local execution of the exact authored source: **9/9 tests passed**. Witness JSON SHA-256: `ad29b13614d0e1e4801be3bd49aeceb314a690b330deb6b9ef829917b0c2da60`. Source SHA-256: `66f6e21d5730f92bcb960970553f17f7876b4c4cc89b0854ff48f18d1fe9591b`. Test SHA-256: `11cfe6e2f9f8385dd1fbe82484c14eea6bef658300e022eb30d03c5b76f01507`.

## Research connection and positioning

- Wang & Ramdas, *Huber-robust confidence sequences* (AISTATS 2023), gives anytime-valid inference under a known upper contamination fraction and a clean central-moment bound: https://proceedings.mlr.press/v206/wang23p.html
- Mu & Xiong, *On Huber's contaminated model* (Journal of Complexity, 2023), studies identifiability properties and the case where the contaminated model is unidentifiable: https://doi.org/10.1016/j.jco.2023.101745
- Xie, Gao & Lafferty, *Confidence Intervals for Linear Models with Arbitrary Noise Contamination* (Biometrika, published online 28 Sep 2026), states that the contamination proportion is not identifiable from data in its setting and develops a structured regression-specific method not requiring prior contamination fraction: https://academic.oup.com/biomet/advance-article-abstract/doi/10.1093/biomet/asag058/8842600
- Zhou & Liu, *On the Robustness of Bandit Multiple Testing* (AAAI 2026), is a comparator for adaptive multiple-testing/allocation efficiency under a stated Huber contamination cap: https://ojs.aaai.org/index.php/AAAI/article/view/40148

These papers address related but different assumptions. The witness above is a simple exact construction, not a novel impossibility theorem. The research value is to make this identifiability limit a mandatory executable negative control for Symthaea's certificate and allocator.

## Next benchmark gates

1. Preserve this exact population witness as a permanent regression test.
2. Add confidence sequences around `q` and map them into clean-mean identified sets; test coverage under valid caps and unsafe admissions under intentionally false caps separately.
3. Include probability that the external contamination-bound process fails, and budget it with the statistical error rate. Never infer the bound's truth from the same data without a theorem covering that adaptation.
4. Compare support-aware sensitivity, Wang–Ramdas Huber–Catoni and robust bandit multiple testing at matched error-control targets; keep unsafe admission, false rejection, unresolved rate and observation cost separate.
5. Integrate an explicit typed result into Rust only after source-level assumptions, authenticated receipts and fallback semantics are specified. This prototype is Python research code, not production integration.
