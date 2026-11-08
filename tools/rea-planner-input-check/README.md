# REA planner input manifest checker

Standalone Rust checker for the six-case synthetic planner-input pilot and grader-only held-out DGP fixture.

It meta-validates both Draft 2020-12 schemas and validates their instances. It also checks probability mass, exact hypothesis/candidate/outcome/scenario coverage, complete utility matrices, REA Evidence ID shape and support references, parent agent-view alignment, hidden-DGP separation and grader-field leakage.

The checker does not execute the planner APIs, authenticate provenance labels, calibrate likelihoods, adjudicate utility, or establish model performance.

Run:

```bash
cargo test --manifest-path tools/rea-planner-input-check/Cargo.toml --locked
```

All inputs are synthetic and public. The `IndependentObservation` origin label in the EvidenceProvenance fixture is a synthetic stand-in used only to exercise planner validation; it is not a real observation or an authenticated independent source.

Any actual policy comparison must call the existing APIs and use an access-controlled held-out set. The public grader is not a hidden test set.