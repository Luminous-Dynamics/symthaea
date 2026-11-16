# REA planner input manifest checker

Standalone Rust checker for the six-case synthetic planner-input pilot and its grader-only held-out DGP sidecar.

It meta-validates both Draft 2020-12 schemas and validates the manifest instances. Semantic checks include probability-mass sums, exact source-agent-view candidate/outcome alignment, Evidence ID/root references, complete point and scenario likelihood matrices, complete terminal utility matrices, scenario candidate coverage, held-out DGP coverage and non-duplication of every declared scenario, and a SHA-1 Git-blob binding to the copied agent-view projection.

It does not execute EIG/EVSI/minimax planners, calibrate caller-supplied likelihoods, authenticate provenance, adjudicate utilities, or establish model performance.

Run:

```bash
cargo test --manifest-path tools/rea-planner-input-check/Cargo.toml --locked
```

All inputs are public synthetic development fixtures. The `IndependentObservation` origin in this fixture is a synthetic stand-in used only to exercise planner API preconditions; it is not a real or authenticated observation. The six-case pilot intentionally contrasts high-EIG/non-decisive observations with lower-EIG proof-obligation closure, shared-source mirrors with independent reproduction, correlated logs with controlled interventions, and unauthorized high-information probes with authorized alternatives.

Case `REA-POA-012` has one possible outcome per candidate. Its held-out categorical likelihood shift is explicitly marked not testable rather than inventing unrepresented outcomes.

A real policy comparison still requires direct adapters that call the existing planner APIs, human-adjudicated inputs, source-family-disjoint cases, access-controlled held-out outcomes, and run accounting through the existing #10314 receipt owner.