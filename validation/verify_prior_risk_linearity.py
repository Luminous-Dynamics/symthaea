#!/usr/bin/env python3
"""Self-contained finite-model check of prior-linearity for expected posterior CE.

This is a research fixture, not Symthaea production code. It assumes:
- a finite latent hypothesis set;
- a fixed deterministic policy tree (action is a function of observed history);
- a shared likelihood model used for the true and learner posteriors;
- a strictly positive learner prior on every latent hypothesis that can be true.

For a fixed policy and terminal categorical objective, expected posterior cross-entropy
is linear in the true prior over latent hypotheses. The nonlinear posterior normalizer
cancels against the probability of the observed history in the outer expectation.
"""
from __future__ import annotations

from itertools import product
from math import isfinite, log2
from typing import Callable, Sequence

Prior = Sequence[float]
# probs[z][step][action][observation] = P(observation | latent hypothesis z, step, action)
ObservationModel = Sequence[Sequence[Sequence[Sequence[float]]]]
Policy = Callable[[tuple[int, ...]], int]


def _validate_prior(p: Prior, n: int, *, strictly_positive: bool = False) -> None:
    if len(p) != n or any(not isfinite(float(x)) or x < 0 for x in p):
        raise ValueError("prior must have correct length and finite nonnegative entries")
    if abs(sum(p) - 1.0) > 1e-10:
        raise ValueError("prior must sum to one")
    if strictly_positive and any(x <= 0 for x in p):
        raise ValueError("learner prior must be strictly positive for finite CE")


def _history_likelihoods(model: ObservationModel, policy: Policy, horizon: int):
    n_states = len(model)
    n_steps = len(model[0])
    n_actions = len(model[0][0])
    n_observations = len(model[0][0][0])
    if not (1 <= horizon <= n_steps):
        raise ValueError("invalid horizon")
    for z in range(n_states):
        if len(model[z]) != n_steps:
            raise ValueError("inconsistent step count")
        for t in range(n_steps):
            if len(model[z][t]) != n_actions:
                raise ValueError("inconsistent action count")
            for a in range(n_actions):
                probs = model[z][t][a]
                if len(probs) != n_observations or any(x < 0 or not isfinite(float(x)) for x in probs):
                    raise ValueError("invalid observation probabilities")
                if abs(sum(probs) - 1.0) > 1e-10:
                    raise ValueError("observation probabilities must sum to one")

    result = {}
    for history in product(range(n_observations), repeat=horizon):
        likelihoods = []
        for z in range(n_states):
            likelihood = 1.0
            prefix: tuple[int, ...] = ()
            for t, obs in enumerate(history):
                action = policy(prefix)
                if not 0 <= action < n_actions:
                    raise ValueError("policy returned invalid action")
                likelihood *= model[z][t][action][obs]
                prefix += (obs,)
            likelihoods.append(likelihood)
        result[history] = tuple(likelihoods)
    return result


def expected_terminal_cross_entropy(
    true_prior: Prior,
    learner_prior: Prior,
    model: ObservationModel,
    policy: Policy,
    class_of_hypothesis: Sequence[int],
    horizon: int,
) -> float:
    """Expected CE (bits) between true and learner terminal class posteriors."""
    n_states = len(model)
    _validate_prior(true_prior, n_states)
    _validate_prior(learner_prior, n_states, strictly_positive=True)
    if len(class_of_hypothesis) != n_states or not class_of_hypothesis:
        raise ValueError("class mapping must have one class per hypothesis")
    if any(not isinstance(c, int) or c < 0 for c in class_of_hypothesis):
        raise ValueError("class labels must be nonnegative integers")

    histories = _history_likelihoods(model, policy, horizon)
    n_classes = max(class_of_hypothesis) + 1
    total_risk = 0.0
    for _, likelihoods in histories.items():
        true_joint = [true_prior[z] * likelihoods[z] for z in range(n_states)]
        p_history = sum(true_joint)
        if p_history <= 0.0:
            continue
        learner_joint = [learner_prior[z] * likelihoods[z] for z in range(n_states)]
        p_learner_history = sum(learner_joint)
        if p_learner_history <= 0.0:
            raise ArithmeticError("learner assigns zero probability to a true-reachable history")

        q_class = [0.0] * n_classes
        l_class = [0.0] * n_classes
        for z, c in enumerate(class_of_hypothesis):
            q_class[c] += true_joint[z] / p_history
            l_class[c] += learner_joint[z] / p_learner_history

        ce = 0.0
        for c, q_c in enumerate(q_class):
            if q_c > 0.0:
                if l_class[c] <= 0.0:
                    return float("inf")
                ce -= q_c * log2(l_class[c])
        total_risk += p_history * ce
    return total_risk


def mixture(p: Prior, q: Prior, alpha: float) -> tuple[float, ...]:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if len(p) != len(q):
        raise ValueError("prior dimensions differ")
    out = tuple(alpha * x + (1.0 - alpha) * y for x, y in zip(p, q))
    _validate_prior(out, len(out))
    return out
