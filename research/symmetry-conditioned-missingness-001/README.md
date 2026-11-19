# Hypothesis-conditioned missingness reference v0.1

First executable research slice for [SYMM-MNAR-001 #12932](https://github.com/Luminous-Dynamics/symthaea/issues/12932), built alongside the conservative probe-abstention semantics in [PR #12690](https://github.com/Luminous-Dynamics/symthaea/pull/12690).

## Two different mathematical objects

### 1. Hard support

The hard-support contract records, for every hypothesis/probe pair, whether abstention is possible.

- If abstention is possible, observing it does not remove that hypothesis.
- If abstention is declared impossible, validation requires an evidence identifier, evidence kind, and matching validity-domain identifier.
- An observed abstention can then eliminate that hypothesis **relative to the declared hard-support model**.
- If all hypotheses are eliminated, the receipt reports model inconsistency. It does not claim that the physical system has no symmetry.

The verifier binds the support table and evidence references into canonical bytes. It does not download or cryptographically verify those external artifacts. The responsible evaluation pipeline must verify that the referenced hard guarantee actually applies over the frozen validity domain.

### 2. Probabilistic dropout intervals

A distinct model carries a calibrated interval for each hypothesis/probe dropout probability in parts per million (PPM). It requires explicit calibration and conditional-independence contract identifiers.

For a sequence of distinct probes, the evaluator computes outward-rounded intervals for the probability of the observed dropout/non-dropout pattern:
- abstention contributes the declared dropout-probability interval;
- a usable reading contributes its complement;
- products use checked integer arithmetic and outward rounding.

**This model never eliminates hypotheses.** A small or even zero lower likelihood is not a hard impossibility. The module reports likelihood intervals only; it does not add priors, posterior probabilities, or a decision threshold.

Repeated observations of the same probe are rejected because this v0.1 product model assumes distinct probe events and does not model retry dependence.

## Negative controls

The exact tests cover:
- hard dropout eliminating only a hypothesis with a declared impossibility witness;
- missing hard-support justification and validity-domain mismatch;
- all-candidate elimination reported as model inconsistency;
- replay receipt mutation/forgery;
- low-but-nonzero dropout probability never deleting a candidate;
- non-abstention using the complement interval;
- interval likelihood multiplication and outward rounding;
- absent calibration/independence contract identifiers;
- duplicate probe attempts and invalid PPM intervals.

## Reproduction

    cargo fmt --manifest-path research/symmetry-conditioned-missingness-001/Cargo.toml --all -- --check
    cargo test --locked --manifest-path research/symmetry-conditioned-missingness-001/Cargo.toml
    cargo clippy --locked --manifest-path research/symmetry-conditioned-missingness-001/Cargo.toml --all-targets -- -D warnings

## Claim boundary

This is a finite categorical missingness reference, not a calibrated physical dropout model. Evidence IDs and calibration IDs are caller-supplied references, not authenticated proof. The probabilistic model assumes conditional independence across distinct probes and must not be used outside its declared validity domain without recalibration. It does not establish MCAR, MAR, or MNAR for any real sensor, and it makes no physical-symmetry or search-pruning claim.
