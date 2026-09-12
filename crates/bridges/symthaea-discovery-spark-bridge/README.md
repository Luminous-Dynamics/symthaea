# symthaea-discovery-spark-bridge

One-shot adapter from Spark's existing Bayesian expected-information-gain (EIG) experiment planner into `symthaea-discovery::ExperimentProposal`.

## Why one-shot

Spark already implements both:

- ranking candidate experiments by expected information gain under a current hypothesis belief; and
- a multi-step greedy sequence that advances the belief by simulating the outcome expected under the current MAP hypothesis.

The second capability is useful planning machinery, but a simulated future outcome is not an observation. This bridge therefore uses **only the current-belief ranking** for generic discovery selection. It never calls the simulated multi-step sequence.

The output means:

> under this declared Spark hypothesis model and current belief, this experiment is predicted to be the most information-efficient next test.

It does **not** mean the experiment occurred, the predicted observation occurred, or the belief was updated by evidence.

## Domain binding

The adapter is intentionally bound to candidate kind `spark_lcf_anomaly_program`. A Spark-specific experiment selector cannot silently operate on an unrelated materials, storage, grid, biology, or engineering candidate.

`SparkEigSelector::candidate(...)` creates the matching descriptive discovery candidate.

## Mapping

For each Spark `ExperimentDesign`:

- `research_question` → discovery question;
- Spark experiment name → deterministic proposal ID;
- current-belief EIG → `expected_information_gain_bits`;
- estimated cost → `ResourceEstimate { unit: "USD" }`;
- planning duration in months → seconds using an explicit 30-day planning-month convention;
- intended evidence → `EvidenceKind::Experiment`.

The bridge deliberately does **not** copy physical setup, trigger, geometry, or instrumentation details into the generic proposal. Those remain Spark-domain records.

## Validation

Before ranking, the bridge rejects:

- wrong candidate kind;
- invalid/mismatched candidate evaluations;
- empty experiment names or research questions;
- duplicate experiment names;
- non-finite/negative costs or durations;
- non-finite/negative ranking outputs.

A design with insufficient encoded outcome classes can legitimately receive zero EIG. If no design clears the configured minimum EIG threshold, `select_next` returns `None`.

## Authority boundary

This crate can translate and rank descriptive experiment proposals. It cannot execute an experiment, invoke hardware, update a belief from a simulated result, produce experimental evidence, authorize nuclear work, procure equipment, or promote a candidate.

## Lock / qualification note

The crate adds no new third-party dependency; it only links two existing workspace crates. As with every newly added workspace member, the repository lockfile still needs exact-toolchain reconciliation before `--locked` qualification can be claimed.
