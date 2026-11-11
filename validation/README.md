# Validation

Research validation is evidence only when both implementations compute a clearly specified target quantity on the same model and all failures are reported.

## PyPhi cross-validation status: BLOCKED

`pyphi_crossvalidation.py` is a legacy scaffold, not a qualified validation pipeline. Its former implementation substituted hard-coded topology estimates when the Rust subprocess failed. The Rust example it invoked is absent from the maintained `examples/` path, and its abandoned sketch used spectral connectivity rather than IIT integrated information.

The current scaffold now fails closed until a maintained Rust adapter can produce a measurement from the same explicit causal model. A missing result is not zero and is not a pass. The PyPhi dependency is pinned to `2.0.0` (stable) with the `emd` extra for IIT 3.0 compatibility. The pinned environment still requires an executed, exact-head qualification run before publication claims. The canonical TSV model manifest is checked by both a Python parser and a Rust test; that parity test is still unexecuted.

## PyPhi formalism

PyPhi 2.0 supports distinct formalisms: `IIT_3_0`, `IIT_4_0_2023`, and `IIT_4_0_2026`. This legacy validator explicitly selects `IIT_3_0`; a future qualified study should run separate, version-pinned comparisons for every formalism it intends to discuss. Do not rely on PyPhi's changing default.

Reference: https://pyphi.readthedocs.io/en/latest/reference/configuration.html

## Run the blocked harness

~~~bash
nix develop .#python-research
python -m pip install -r validation/requirements-pyphi.txt
python -m unittest validation.test_pyphi_crossvalidation_fail_closed validation.test_shared_boolean_network -v
python validation/pyphi_crossvalidation.py
~~~

This currently produces unavailable/failed measurements by design, until the adapter exists. It writes a NOT QUALIFIED Markdown report to results/pyphi_validation_results.md and returns a nonzero exit status. The twenty Python regression tests are source definitions until the command above has actually run successfully. Do not interpret the output as evidence of agreement.

## TCCIP independent finite-state reference

The task-conditioned counterfactual integration candidate has a small, exact Boolean-network evaluator and a separate standard-library Python oracle. The executable source of fixture rules, task targets, input-state weights, component partitions, intervention edges, and expected outcomes is `fixtures/tccip_toy_expectations.tsv` (schema v3). Both the Rust example and independent Python oracle parse this manifest.

Run the additional independent, standard-library checker (separate parser/evaluator) as a third exact cross-check:

~~~bash
set -euo pipefail
SYMTHAEA_SOURCE_SHA="$(git rev-parse HEAD)" \
  python validation/tccip_independent_check.py
~~~

It checks all six manifest fixture summaries/partition ranges and all 16 joint-intervention rows. Its output includes SHA-256 digests of both input TSV files. It is a separate executable, not a Rust test or a result emitted by the primary Python oracle. This newly committed checker itself has not yet been run from the repository; the one-off exhaustive audit performed during development was a distinct local script against copies of the two manifest files.

Run the independent reference oracle:

~~~bash
set -euo pipefail
python validation/tccip_reference_oracle.py \
  --source-sha "$(git rev-parse HEAD)" \
  --output results/tccip_python_reference.tsv
sha256sum results/tccip_python_reference.tsv
python -m unittest validation.test_tccip_reference_oracle -v
~~~

Run the Rust implementation separately:

~~~bash
set -euo pipefail
cargo test --example task_conditioned_finite_state_eval
SYMTHAEA_SOURCE_SHA="$(git rev-parse HEAD)" \
  cargo run --release --example task_conditioned_finite_state_eval \
  > results/tccip_rust_reference.tsv
sha256sum results/tccip_rust_reference.tsv
~~~

Additionally, preserve the `validation/fixtures/tccip_pairwise_utility_interactions.tsv` expectation matrix when reviewing joint-intervention results. Its positive/negative signs refer to utility-loss saturation versus compound-failure amplification, not information synergy. Keep both artifacts and their source SHAs/hashes. They are independently authored calculations over the same manifest; do not call the Python output a Rust run. A `UNBOUND` source SHA marks the output exploratory. Current fixtures show three limitations: arbitrary partition changes can reverse TCCIP's sign; reversing the input-state prior can reverse it for the same graph/task; and in balanced XOR, equally essential cross-cut/internal edges each reduce accuracy by 0.5, so TCCIP is zero even though joint input MI is 1 bit while either individual input MI is zero. The Rust output and independent Python artifact now include this information baseline. Report per-edge effects, partition and prior profiles, and higher-order information separately; do not use TCCIP as a standalone integration/synergy scalar.

The oracle and sixteen Python regression-test definitions are authored research code; they have not been run from the committed files in this environment. The Rust evaluator has 19 authored tests and has not been compiled. A second manifest, `fixtures/tccip_pairwise_utility_interactions.tsv`, specifies 16 exact two-cut cases; the Rust and Python suites check their single-cut and joint-cut utilities plus the second-order finite difference. This diagnostic measures non-additive task-utility loss, not PID/PEID synergy or IIT Φ. No qualification or performance claim is made by adding the harness.

## Required before qualification

- A maintained Rust adapter that emits scores for the same causal transition model as PyPhi.
- Canonical model digest, separate state-prior digest, source SHA, PyPhi version, explicit formalism, state, and configuration in each receipt.
- No topology-name fallbacks, default-zero substitutions, or hidden exception-to-score paths.
- Enough finite, paired cases, held-out model families, uncertainty intervals, absolute-error metrics, rank metrics, and negative controls.
- Exact-head evidence; queued or missing CI is not a pass.

See `docs/research/ENTROPY_CAUSAL_CUT_BENCHMARK.md` and issue #8238 for the active remediation plan.
