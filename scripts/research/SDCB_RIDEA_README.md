# SDCB R-IDeA robust acquisition reference

Standalone Rust reference for the R-I and R-IDeA acquisition equations from Tang, Sloman & Kaski (AISTATS 2026).

## Build and run

```bash
rustc --edition=2024 --test scripts/research/sdcb_ridea.rs -o /tmp/sdcb-ridea
/tmp/sdcb-ridea
```

The source uses only the Rust standard library.

## Implemented

- Biased empirical MMD squared using an RBF kernel and caller-supplied bandwidth; the R-I ratio uses this squared value directly, matching the released upstream implementation.
- R-I robust representativeness ratio multiplied by supplied expected information gain.
- R-IDeA DeA sigmoid using supplied current-model and proxy predictions.
- Raw R-I, R-IDeA, and score-per-cost outputs; cost ranking is explicitly a project extension.
- Deterministic ranking, non-positive score abstention, and invalid-input checks.

## Not implemented

- The proxy model g training objective from the paper is not trained here.
- EIG is a caller-supplied input; this code does not compute Bayesian posteriors or likelihoods.
- No model misspecification or covariate shift is measured from real data.
- No decision value, EVSI, robustness guarantee, or safety guarantee is implied by a high R-IDeA score.

## Frozen fixture

Target support {0,1,2}, history {0,2}, RBF sigma 1.0. Baseline biased MMD² is 0.0394007. Adding candidate x=1 reduces MMD² to zero and yields R-IDeA 0.7046377. A higher-EIG x=4 candidate increases MMD² to 0.2197536 and receives negative R-I/R-IDeA. The released code uses raw MMD² in its ratio; this reference matches that executable behaviour. These are arithmetic fixture values, not an external experiment.

## Validation boundary

18 Rust test functions are authored. They have not been compiled or executed in the current environment. JSON schemas/examples are parsed and semantically checked separately; that is not a Rust test pass or full benchmark reproduction.