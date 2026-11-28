## Multi-order reduction tournament

Proposed targets (not achieved):
- 100% rejection of malformed or duplicate deletion permutations.
- 100% same trial/scenario/boundary/code/failure-signature scope across orders.
- 100% complete-order requirement for a 1-minimal guarantee.
- 0 global-minimum claims for >12 faults.
- 100% deterministic best-observed selection and tie-breaking.
- Report subset evaluations and runtime per ordering; compare against the original single-order reducer.

Synthetic adversarial interaction fixture: 13 faults contain two distinct target-preserving minimal explanations (a two-fault set and a three-fault set). Two deletion orders should find different valid 1-minimal sets. The tournament must select the two-fault observed set while reporting best-observed-only, not global minimum.
