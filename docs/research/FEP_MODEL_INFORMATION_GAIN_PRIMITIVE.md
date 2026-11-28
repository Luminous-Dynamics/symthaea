# FEP model-hypothesis information gain primitive

**Status:** isolated implementation with unit tests; not yet connected to production action selection and not a performance result.

## Motivation

The current active-core FEP's epistemic term is computed from the difference between the entropy of a predicted next state and the current state. Under the current transition update, predicted precision and mode probabilities are action-independent for a fixed belief. Because the entropy implementation depends on those quantities rather than the predicted mean, that term does not rank candidate actions by expected information about competing model hypotheses. Action novelty may still vary, but novelty is not information gain. See [Issue #8391](https://github.com/Luminous-Dynamics/symthaea/issues/8391).

Friston et al. (2026), *Active inference and artificial reasoning*, explicitly study action selection to disambiguate candidate generative models through expected information gain. This primitive implements that limited mathematical quantity in an isolated, testable form.

## API and mathematics

Each ModelHypothesis provides:
- a unique identifier;
- a prior probability (p(m));
- a predictive outcome distribution (p(omid m,a)) for every candidate action.

For each action, the implementation computes the model-averaged outcome distribution

[
p(omid a)=sum_m p(m),p(omid m,a)
]

and, for each possible outcome, the posterior

[
p(mmid o,a)=rac{p(m),p(omid m,a)}{p(omid a)}.
]

The expected information gain is

[
I(M;Omid a)=H(M)-sum_o p(omid a)H(Mmid o,a).
]

It is returned in **nats**. An outcome that is impossible under every candidate model receives no posterior and contributes zero to the expectation. Inputs with missing/duplicate model identities, non-finite/out-of-range probabilities, unnormalized priors, or malformed action/outcome dimensions are rejected rather than silently normalized.

Public entry points:
- compute_action_information_gains computes the value for every action;
- compute_action_information_gain computes one requested action's result;
- most_informative_action returns the action with the highest information gain, with deterministic lowest-index tie-breaking.

## Important limits

This is **not** a complete expected-free-energy policy. It does not incorporate reward preferences, observation cost, safety constraints, temporal planning, or execution authority. A caller must combine the information term with pragmatic value and cost in an explicit, reviewed decision rule.

It also does not learn or establish its candidate models. The caller supplies model-conditioned outcome probabilities; those assumptions must be bound to an exact model-set revision and task profile, and posterior model weights remain model-relative.

The primitive is not yet wired into ActiveInferenceAgent::select_action. Wiring it directly into production before a task-level observation model, negative controls, and parity/rollback tests exist would risk presenting a correct isolated formula as a validated agent improvement.

## Tests

The module includes tests for:
- identical predictions giving zero information gain;
- an action that discriminates between two models receiving positive information gain;
- the expected-entropy identity and posterior normalization;
- impossible outcomes and zero-prior hypotheses;
- invariance to model ordering;
- rejection of inconsistent action counts and unnormalized distributions/priors;
- duplicate model-identity rejection and deterministic tie-breaking.

## Acceptance path

1. Qualify this formula and these input contracts.
2. Bind actual Symthaea model-hypothesis/outcome distributions with their exact provenance and applicability scope.
3. Evaluate the contribution using the deterministic information-choice environment from [PR #8375](https://github.com/Luminous-Dynamics/symthaea/pull/8375), then graduate to multiple competing rules/models.
4. Compare against cue-greedy, always-observe, Bayesian value-of-information and matched RL/information-gain controls under paired held-out seeds.
5. Only then consider integrating the information term into action selection; preserve a no-information negative control and report null or adverse results.

References:
- Friston et al. (2026), [Active inference and artificial reasoning](https://doi.org/10.1038/s41467-026-77209-5).
- Chou et al. (2026), [A systematic empirical comparison of active inference and reinforcement learning models](https://doi.org/10.1016/j.jmp.2026.103011).
