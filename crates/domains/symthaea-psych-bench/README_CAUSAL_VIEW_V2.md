# GWT-1 resolved evidence view V2

V2 separates the outcome of an evidence lineage from the strongest indicator outcome that remains supported after that lineage is incorporated.

For the GWT-1 chain:

- direct `Qualified` -> lineage `Observed`, resolved `Observed`;
- causal `Qualified` after direct `Observed` -> lineage `CausallySupported`, resolved `CausallySupported`;
- causal `NotDemonstrated` after direct `Observed` -> lineage `NotDemonstrated`, resolved remains `Observed`;
- causal `Contradicted` after direct `Observed` -> lineage `Contradicted`, resolved remains `Observed`;
- causal `Inconclusive` after direct `Observed` -> lineage `Inconclusive`, resolved remains `Observed`.

This prevents a higher-tier null or contradiction from erasing independent lower-tier evidence while preserving the negative result as part of the evidence lineage.

V2 does not create causal authority. It reuses the V1 verification path. Positive causal elevation still requires the opaque cryptographically verified promotion token, and no V2 path can produce `FunctionallySupported`.
