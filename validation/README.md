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

The task-conditioned counterfactual integration candidate has a small, exact Boolean-network evaluator and a separate standard-library Python oracle. The executable source of fixture rules, partitions, intervention edges, and analytical expectations is `fixtures/tccip_toy_expectations.tsv` (schema v3, including integer-weighted input-state priors). The Rust example parses this file rather than duplicating the three networks and intervention lists in source.

Run the independent reference oracle:

~~~bash
python validation/tccip_reference_oracle.py \
  --source-sha "$(git rev-parse HEAD)" \
  --output results/tccip_python_reference.tsv
sha256sum results/tccip_python_reference.tsv
python -m unittest validation.test_tccip_reference_oracle -v
~~~

Run the Rust implementation separately:

~~~bash
cargo test --example task_conditioned_finite_state_eval
SYMTHAEA_SOURCE_SHA="$(git rev-parse HEAD)" \
  cargo run --release --example task_conditioned_finite_state_eval \
  > results/tccip_rust_reference.tsv
sha256sum results/tccip_rust_reference.tsv
~~~

Keep both artifacts and their source SHAs/hashes. They are independently authored calculations over the same manifest; do not call the Python output a Rust run. A `UNBOUND` source SHA marks the output exploratory. Current fixtures show two distinct sensitivity effects: arbitrary partition changes can reverse the sign of TCCIP, and reversing the input-state prior can flip TCCIP (+0.3 to −0.3) even when the causal rules, task, partition, and intervention edges are identical. The latter pair uses edges into the same task-output AND gate as a stronger role-matched comparison. Report the full partition and state-prior profile; never select the most favorable result after seeing outcomes.

The oracle and ten Python regression-test definitions are newly authored research code. They have not yet been run from the committed files in this environment. The Rust evaluator has 14 authored tests and has not been compiled. No qualification or performance claim is made by adding the harness.

## Required before qualification

- A maintained Rust adapter that emits scores for the same causal transition model as PyPhi.
- Canonical model digest, separate state-prior digest, source SHA, PyPhi version, explicit formalism, state, and configuration in each receipt.
- No topology-name fallbacks, default-zero substitutions, or hidden exception-to-score paths.
- Enough finite, paired cases, held-out model families, uncertainty intervals, absolute-error metrics, rank metrics, and negative controls.
- Exact-head evidence; queued or missing CI is not a pass.

See `docs/research/ENTROPY_CAUSAL_CUT_BENCHMARK.md` and issue #8238 for the active remediation plan.
