# SC-CAUSAL-TRANSITION-0.1

**Status:** bounded synthetic two-step state-transition planner; no physical experiment is authorized.  
**Parent research issue:** [#12966](https://github.com/Luminous-Dynamics/symthaea/issues/12966)  
**Risk/resource planning dependency:** [#12965](https://github.com/Luminous-Dynamics/symthaea/pull/12965)

## Paradigm shift

Static experiment selection assumes a fixed specimen and treats actions as observations. This contract instead models an action-conditioned physical transition:

\`X_{t+1} \sim T(X_{t+1}\mid X_t,A_t)\`

The observed outcome is sampled jointly with the transition edge:

\`Y_t \sim O(Y_t\mid X_t,A_t,X_{t+1})\`

That joint edge semantics matters for destructive characterization: the observation can be informative about the pre-intervention state even though the specimen is damaged by the end of the operation.

The physical state keeps pairing strength, phase coherence and sample integrity separate. This permits an intervention to improve pairing-sensitive response while degrading coherence or preserving a high pairing amplitude in a damaged, unusable specimen.

## Exact horizon-two policy

The Rust binary \`sc_causal_transition_audit\` exhaustively evaluates every eligible first action. For each possible first observation, it compares stopping against every feasible distinct second action. The second action can differ between first-outcome branches.

The objective is expected terminal decision loss under the joint initial-state / transition / observation distribution. Terminal decisions are selected after the final observation. The initial state is not assumed known.

Resource use is path-dependent. Each realized branch consumes the first action plus its own second action. The receipt reports a **componentwise worst-case resource envelope** across mutually exclusive observation branches; it does not add mutually exclusive branches together. Every resource dimension is checked independently.

The exact planner is bounded to eight physical states, eight actions, two binary observation outcomes and horizon two. It is a reference oracle for small finite problems, not a production-scale scenario-tree solver.

## Action and specimen semantics

Action records distinguish:
- passive observations;
- reversible interventions;
- irreversible interventions;
- destructive characterization;
- sham controls.

Every action is linked to one specimen identity, declares reversibility, prerequisites, resource use and a full state-transition row for every physical state. Unknown/discrepant states are explicit. A destructive operation's sample damage is part of the transition model, so the planner cannot assume it can reuse the original specimen unchanged.

The current model uses known synthetic transition probabilities. It does not infer causal effects from observational data, model uncertainty in the transition parameters, or prove that a real intervention is reversible. Its treatment-versus-sham contrasts are model-implied synthetic contrasts only.

## Frozen adversarial fixtures

The current two-instance contract includes:

1. **Pressure/strain path dependence** — pressure moves a fresh state into a paired-but-incoherent state; strain then can transition it to a coherent state. Strain applied first damages the fresh state and pressure cannot rescue it. The exact policy can choose strain only after a supportive pressure observation. This fixture also includes a destructive characterization option.
2. **Destructive-information versus specimen preservation** — a destructive characterization can identify the initial sample class but leaves the specimen destroyed. The downstream application loss makes stopping/preserving the specimen the optimal policy.

The fixtures are synthetic and their transition probabilities and losses are invented. They are not superconductivity predictions, protocol recipes, experimental recommendations or measured causal effects.

## Comparators and metrics

The receipt compares:
- no-intervention stop/preserve baseline;
- myopic single-action observation/intervention;
- exact adaptive two-step policy;
- fixed pressure→strain versus strain→pressure path losses;
- model-implied pairing, phase-coherence and specimen-integrity contrasts versus sham;
- worst-case resource envelope and terminal decision policy.

Follow-on work should compare factorial / response-surface policies, one-factor-at-a-time, static EVSI / MI, GO-CBED, constrained scenario-tree BED and laboratory expert policies on the same frozen finite-state transition models and paired seeds.

## Claim ceiling

This is a synthetic transition-planning contract. It does not establish an empirical causal effect, pairing mechanism, bulk superconductivity, instrument protocol, specimen availability, safe operating conditions or permission to conduct an experiment.

## Pairing/coherence separation control

The pressure/strain fixture assigns separate pairing-strength and phase-coherence coordinates. Under its synthetic transition table, the single-step strain intervention raises the expected pairing-strength coordinate relative to sham while lowering expected phase coherence and specimen integrity. The unit test asserts the signs separately. This is deliberately not collapsed into a scalar “superconductivity score” or interpreted as a real material prediction.

The preservation fixture has prior expected terminal loss 3.8 under defer/preserve and 8.0 under immediate application. Destructive characterization maps both possible initial sample classes into the destroyed state; because the terminal application value is lost, the exact policy stops and preserves the sample even though the diagnostic outcome distinguishes the initial states. These are fixture semantics, not observed specimen utilities.
