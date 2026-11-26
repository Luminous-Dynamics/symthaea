# SDCB belief-transition reference harness

This standalone Rust source models a finite hypothesis set and an evidence-gated posterior transition.

## Compile and run

From the repository root:

```bash
rustc --edition=2024 --test scripts/research/sdcb_belief_transition.rs -o /tmp/sdcb-belief-transition
/tmp/sdcb-belief-transition
```

The file uses only the Rust standard library and does not depend on the Symthaea Cargo workspace.

## Coverage

- Expected information gain for a finite hypothesis/outcome model.
- Cost-normalized experiment choice with deterministic tie-breaking.
- Abstention when all candidate probes are uninformative.
- Bayesian posterior calculation from a declared prior and observed outcome.
- Exact run/task/system/hypothesis/proposal binding.
- Independent-verification gate and verifier identity check.
- Stale version, duplicate experiment, cost mismatch, and budget rejection.
- Atomic failure: invalid transitions do not change posterior, version, budget, or accepted count.
- Invalid distributions, impossible observations, and malformed digest controls.

## Claim limits

The harness does not authenticate signatures, hash observation bytes, establish chain of custody, validate sensor calibration, or prove that a model is scientifically correct. The independently_verified flag is a test seam only. Production integration requires a separately governed verifier that returns an authenticated, revocable attestation bound to the exact experiment and observation.