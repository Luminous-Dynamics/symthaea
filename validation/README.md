# Validation

Research validation is evidence only when both implementations compute a clearly specified target quantity on the same model and all failures are reported.

## PyPhi cross-validation status: BLOCKED

`pyphi_crossvalidation.py` is a legacy scaffold, not a qualified validation pipeline. Its former implementation substituted hard-coded topology estimates when the Rust subprocess failed. The Rust example it invoked is absent from the maintained `examples/` path, and its abandoned sketch used spectral connectivity rather than IIT integrated information.

The current scaffold now fails closed until a maintained Rust adapter can produce a measurement from the same explicit causal model. A missing result is not zero and is not a pass. The pinned PyPhi `2.0.0rc4` package is a pre-release, and the `emd` extra is installed for IIT 3.0 compatibility; this environment must be requalified against a final release before publication claims.

## PyPhi formalism

PyPhi 2.0 supports distinct formalisms: `IIT_3_0`, `IIT_4_0_2023`, and `IIT_4_0_2026`. This legacy validator explicitly selects `IIT_3_0`; a future qualified study should run separate, version-pinned comparisons for every formalism it intends to discuss. Do not rely on PyPhi's changing default.

Reference: https://pyphi.readthedocs.io/en/latest/reference/configuration.html

## Run the blocked harness

~~~bash
nix develop .#python-research
python -m pip install -r validation/requirements-pyphi.txt
python -m unittest validation.test_pyphi_crossvalidation_fail_closed -v
python validation/pyphi_crossvalidation.py
~~~

This currently produces unavailable/failed measurements by design, until the adapter exists. It writes a NOT QUALIFIED Markdown report to results/pyphi_validation_results.md and returns a nonzero exit status. The nine Python regression tests are source definitions until the command above has actually run successfully. Do not interpret the output as evidence of agreement.

## Required before qualification

- A maintained Rust adapter that emits scores for the same causal transition model as PyPhi.
- Model / TPM digest, source SHA, PyPhi version, explicit formalism, state, and configuration in each receipt.
- No topology-name fallbacks, default-zero substitutions, or hidden exception-to-score paths.
- Enough finite, paired cases, held-out model families, uncertainty intervals, absolute-error metrics, rank metrics, and negative controls.
- Exact-head evidence; queued or missing CI is not a pass.

See `docs/research/ENTROPY_CAUSAL_CUT_BENCHMARK.md` and issue #8238 for the active remediation plan.
