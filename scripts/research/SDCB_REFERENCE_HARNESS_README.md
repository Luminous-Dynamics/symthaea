# SDCB analytic reference harness (v1)

This is a dependency-free Rust test harness for the hand-checkable analytic controls in the Spectral–Dynamical Cognition Bridge.

## Run independently of the Symthaea workspace

From the repository root:

```bash
rustc --edition=2024 --test scripts/research/sdcb_reference.rs -o /tmp/sdcb-reference-tests
/tmp/sdcb-reference-tests
```

The file uses only the Rust standard library and can be compiled without resolving the repository's workspace or dependency graph. This makes it useful when the large workspace is expensive to build.

## What the reference cases cover

- Exact 2x2 real symmetric eigenvalues.
- Spectral radius of real 2x2 matrices, including complex-conjugate eigenvalues.
- Discrete-time stability classification from spectral radius.
- The exact trace/determinant Hurwitz criterion for a real 2x2 continuous-time linear system.
- Adjacent spectral gap ratios and positive-affine invariance.
- A deterministic prime sieve for small analytic reference cases.
- 2x2 linear-system observability rank and 2-feature input-excitation rank, with scale-normalized rank checks.
- An exact hidden-mode non-identifiability fixture: two different systems generate identical measured traces, plus active-sensor discrimination and scale-invariance metamorphic controls.
- Beta–Bernoulli conjugate posterior updates, posterior-predictive means and a deterministic UCB1-style arm-selection baseline.
- XOR binding/unbinding, majority bundling, Hamming distance, and reproducible synthetic binary codewords.

The tests include malformed-input and boundary controls. The deterministic codeword generator is **not cryptographic** and is only for reproducible test fixtures.

## Explicit limits

This harness does not:
- compute zeta zeros, certify their precision, or verify RH;
- simulate GUE/GOE ensembles or establish a random-matrix law;
- fit a Koopman model or identify nonlinear dynamics;
- implement or benchmark the Symthaea cognitive architecture;
- prove a theorem or verify a proof plan;
- establish consciousness, intelligence, or biological correspondence.

The Rust source currently authors 32 unit tests, including observability, excitation, and selective-abstention controls. A passing run would validate these small reference functions only. It would not count as an SDCB model result or a SOTA comparison.


## Machine-readable reference receipt

A standalone runner emits one JSON receipt for 29 analytic reference-control cases, including spectral/stability checks, HDC primitives, identifiability/active sensing, and Beta–Bernoulli/UCB sequential-design controls:

```bash
rustc --edition=2024 scripts/research/sdcb_reference_receipt.rs -o /tmp/sdcb-reference-receipt
/tmp/sdcb-reference-receipt \
  --source-revision "$(git rev-parse HEAD)" \
  --compiler-version "$(rustc --version)" \
  > /tmp/sdcb-reference-receipt.json
```

Validate the JSON structure against `docs/research/SDCB_REFERENCE_CONTROL_RECEIPT_V1.schema.json`. The runner first verifies that the supplied source revision equals the repository's observed Git HEAD and that the working tree (including untracked, non-ignored files) is clean. It exits with a provenance error before running controls if either condition fails. The receipt records the observed HEAD, clean-tree status, compiler/platform metadata, RUSTFLAGS and per-case dispositions. The runner does not sign the receipt or prove that the Git repository itself has trusted custody; those claims require the existing Luminous attestation/checkpoint infrastructure.

Receipt claim ceiling: `analytic_reference_controls_only`. A valid receipt is not an SDCB model benchmark, theorem proof, proof-checker qualification, physical experiment, or consciousness result.
