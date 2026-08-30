# symthaea-scenario-evidence

Evidence-bearing counterfactual scenario envelopes for Symthaea decision support.

## Core distinctions

This crate makes the following statements structurally different:

- a causal effect is identified;
- a causal effect is unidentified;
- identification depends on an explicit assumption;
- a scenario is produced by simulation only;
- a scenario is speculative.

A scenario envelope records baseline evidence, intervention intent, model/version/digest references, causal support, assumptions, and an epistemic class.

It deliberately contains **no execution authority** and **no single aggregate utility score**.

## Numerical effect boundary

The current counterfactual identification engine can return a symbolic estimand while its numerical `effect` field remains a placeholder. `symthaea-earth-causal-query` strips that placeholder before it reaches this crate.

A future numerical effect estimate must have its own evidence-bearing object containing at minimum:

- estimator identity/version;
- dataset and evidence lineage;
- estimand identity;
- effect estimate and units;
- uncertainty/confidence interval;
- diagnostics and assumption checks;
- out-of-sample or prospective validation where appropriate.

## Digital twin integration

`symthaea-digital-twin::Intervention` maps into `ScenarioIntervention`. Its urgency is retained only as context for scenario prioritization. Urgency is not permission to execute the intervention.

## Intended next layer

Plural scenario outcomes should remain separate from this provenance envelope so different stakeholders can inspect multiple consequence dimensions and distributional effects without collapsing them into one hidden value function.
