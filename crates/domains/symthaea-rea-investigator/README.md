# symthaea-rea-investigator

Deterministic, dependency-free Rust planner for selecting the next authorized experiment from a caller-supplied hypothesis distribution and REA Evidence provenance catalog.

## Paradigm

Move from "rank the most likely cause" to "identify which permitted observation most reduces uncertainty between competing explanations."

The planner computes expected information gain in bits:

```
EIG(E) = H(Hypotheses) - Σ_outcomes P(outcome) × H(Hypotheses | outcome)
```

It ranks eligible proposals by EIG divided by normalized estimated time, energy and risk cost. Ties are deterministic: utility descending, EIG descending, experiment ID ascending.

## Safety and claim boundaries

- This crate ranks proposals only. It does not execute an experiment, access a device, launch a process or mutate a target.
- Only explicitly authorized, reversible, non-destructive proposals within the configured target-impact and resource budgets are eligible.
- Unknown authorization is denied.
- Evidence origin classifications are caller-supplied and are not cryptographically authenticated.
- Independent derivations qualify only when their parent graph resolves to an independent observation; dangling references and cycles are rejected.
- Model-generated, synthetic and external-reference records do not independently ground hypotheses.
- If any hypothesis is ungrounded, the planner abstains instead of silently dropping that alternative.
- The expected information gain is conditional on caller-supplied priors and likelihoods. It does not prove these models are calibrated.
- Observed entropy reduction is a separate function that requires independently rooted observation Evidence, and still does not authenticate that Evidence.

The output claim ceiling explicitly leaves producer authentication, causal truth, experiment execution, target mutation and likelihood calibration unverified.

## API

Enable the root Symthaea feature `rea-investigator` and call `symthaea::rea_investigator::plan_next_experiment` in-process. MCP is not required.

The public API defines:
- `EvidenceProvenance` and `EvidenceOrigin`;
- `Hypothesis` with normalized prior probability and support/contradiction Evidence IDs;
- `ExperimentProposal` with outcome likelihoods, authorization, reversibility, target impact and time/energy/risk estimates;
- `PlannerConfig` with explicit budgets, weights and minimum EIG;
- `InvestigationPlan` with selected/ranked proposals, rejection reasons and abstention state;
- `measure_observed_information_gain` to separately record posterior entropy reduction after independent observations.

## Validation

The crate is tested through a standalone, locked harness so validation does not rely on the repository's currently unhealthy root workspace glob:

```bash
cargo fmt --manifest-path tools/rea-investigator-harness/Cargo.toml -- --check
cargo test --manifest-path tools/rea-investigator-harness/Cargo.toml --locked
```

The exact upstream-generated Evidence provenance test suite remains separate from this planner. The planner accepts Evidence IDs and provenance classifications; it does not itself verify REA Evidence IDs or producer authenticity.

## Claim ceiling

This is a deterministic decision-support planner over supplied probability models. It is not a causal inference engine, a security decision authority, an autonomous experimentation agent or an evidence authenticator.
