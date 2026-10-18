# Culinary decision quality and VOI corpus v1

This directory contains a **synthetic contract corpus** for qualifying the decision disposition and measurement-selection layer described in [DECISION_QUALITY_AND_VOI_V1.md](../../benchmarks/DECISION_QUALITY_AND_VOI_V1.md).

- corpus.json defines 12 deterministic cases with policy-visible inputs and scorer-only expected outputs.
- Every value, source reference, candidate, timestamp, group, and truth in the file is synthetic. It is not observed price, inventory, nutrition, safety, or household outcome data.
- The test harness must provide only the policy_input object to the candidate. It must keep oracle_only_expected inaccessible during inference and scoring-independent tuning.
- oracle_only_expected is an acceptance contract to implement and verify independently; it is not a pre-existing runtime test result.
- Passing this synthetic contract corpus establishes only finite fixture semantics. It does not establish probability calibration, external source truth, food-safety compliance, household benefit, or generalization.

Before threshold tuning, freeze the corpus digest and keep a separate held-out set. Mutating expected results after inspecting held-out outcomes requires a new corpus version.