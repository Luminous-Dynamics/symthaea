# Institutional Expectations and Anticipatory Adaptation Benchmark

**Status:** research design v0  
**Related:** #7061, #7046, #7047, #7050, #7051, #7052, #7054, #7057, #7058

## Purpose

Test whether beliefs about future institutional states alter present behavior and therefore feed back into institutional evolution.

Experiments-about-institutions literature emphasizes that beliefs about future institutions are central to identifying critical junctures in real time rather than only after outcomes are known. Recent work on divergent market transitions likewise argues that social expectations can interact with institutional uncertainty to produce multiple self-enforcing trajectories. A 2025 institutional-economics contribution further argues that institutions shape expectations by structuring how agents engage with the future.

Sources:
- https://www.annualreviews.org/content/journals/10.1146/annurev-economics-091823-031317
- https://www.sciencedirect.com/science/article/pii/S0305750X25001901
- https://www.cambridge.org/core/journals/journal-of-institutional-economics/article/enacting-the-future-institutions-temporal-affordances-and-the-formation-of-expectations/2CEE2BDD077901753D56347CE5E9FCBC
- https://www.cambridge.org/core/journals/journal-of-institutional-economics/article/understanding-masahiko-aokis-comparative-institutional-analysis/B8153640E3B8C0568B71C48E8A42CF01

## Causal model

~~~text
current institution
      -> observation
      -> expectation about future institution
      -> current action
      -> aggregate outcome
      -> institutional proposal/adoption
      -> future institutional state
      -> revised expectation
~~~

An observed action is therefore not attributable to current rules alone when future-state beliefs are part of the agent model.

## Expectation representation

An expectation is a state of an agent at a declared time T.

It may be represented as:

- probability distribution;
- interval;
- ordinal confidence;
- deterministic forecast rule;
- categorical expectation;
- bounded heuristic;
- explicitly unresolved.

Do not force deterministic heuristics into artificial probabilities.

Required identity:

~~~text
expectation_id
+ subject
+ observation_time
+ referenced institution/profile
+ expectation rule/version
+ information set
+ belief value/representation
~~~

## Information boundary

Agents may access only information explicitly included in their information set.

They must not access:

- hidden evaluator truth;
- future shocks;
- future institution states;
- future authority decisions;
- post-treatment outcomes;
- oracle scores not publicly available at the time.

Any deliberate disclosure is part of the experiment.

## Treatment families

### T0 — Myopic

Agents condition only on current declared institutional state.

### T1 — Historical adaptive

Expectations update from observed institutional history.

### T2 — Rule-based anticipatory

Agents apply an explicit deterministic mapping from signals to future-state beliefs.

### T3 — Bounded/noisy

Belief updates contain explicit bounded error/noise.

### T4 — Heterogeneous

Different populations use different expectation rules.

### T5 — Public-signal treatment

Authorities/agents publish expectations about future institutional change.

Treat T5 as a signal-generation intervention, not proof that the expected change will occur.

## Critical distinction

~~~text
expectation about X
!=
X
!=
evidence that X will happen
~~~

False expectations can therefore be causally effective without becoming correct.

## Expectation-shock fixtures

- credible announcement;
- uncertain announcement;
- false announcement;
- delayed implementation;
- unexpected reversal;
- temporary emergency rule with uncertain expiry;
- expected institutional failure that does not happen;
- expected persistence that fails;
- two populations with different public information;
- public correction after belief divergence;
- identical current institutional state with different inherited beliefs.

## Anticipatory-response fixtures

Record whether expectations alter behavior before the underlying institutional change:

- saving/holding;
- consumption;
- production;
- investment;
- capacity expansion;
- quality;
- maintenance;
- migration;
- coalition formation;
- governance participation;
- proposal activity.

Measure the lead time between expectation change and physical/institutional response.

## Self-fulfilling failure

Required paired fixture:

~~~text
weak persistence expectation
-> lower investment/compliance
-> lower output / weaker coordination
-> observed institutional failure
~~~

versus:

~~~text
strong persistence expectation
-> higher investment/compliance
-> stronger output / coordination
-> institutional persistence
~~~

Keep the underlying physical environment and institutional rules identical until the declared divergence point.

This is a test of expectation-mediated coordination, not proof that the equilibrium is socially optimal.

## Self-negating expectation

Also require:

~~~text
expected scarcity
-> precautionary accumulation
-> reduced realized scarcity
-> expectation falsified
~~~

This prevents the laboratory from encoding self-fulfilling expectations as the only possible reflexive pattern.

## Expectation manipulation

Allow declared information sources to publish false or strategically selective signals about:

- future rule changes;
- enforcement;
- resource availability;
- contribution recognition;
- institutional survival;
- access conditions;
- transition timing.

Separate:

1. signal issuance;
2. signal provenance;
3. agent belief update;
4. current action;
5. actual institutional transition;
6. later belief revision.

## Belief convergence/divergence

Measure:

- cross-agent dispersion;
- within-group dispersion;
- convergence rate;
- divergence rate;
- source dependence;
- memory persistence;
- reversal latency;
- confidence calibration, where probabilistic beliefs are used.

Convergence does not imply correctness.

## Critical-juncture detection

Flag periods where:

- institutional uncertainty is high;
- belief dispersion is high;
- multiple institutional trajectories are reachable;
- behavior is unusually sensitive to expectation changes;
- authority or measurement rules are changing simultaneously.

These are candidate critical junctures for later empirical comparison.

## Counterfactual intervention

At time T, hold the physical state and current institutional rules constant.

Change only the public expectation signal.

Compare subsequent actions and outcomes.

This isolates expectation-mediated pathways from direct rule effects.

## Belief-vs-history ablation

Hold institutional history constant while varying initial beliefs.

Then hold initial beliefs constant while varying institutional history.

Compare which component explains divergence.

This directly tests co-evolution rather than assuming a one-way causal path.

## Negative controls

Include problems where future institutional expectations should have no causal relevance.

Example:

~~~text
expectation treatment
+ instantaneous fully reversible transaction
~~~

If large effects appear where no intertemporal channel exists, inspect the model for leakage or hidden coupling.

## Qualification boundary

A PASS establishes only the declared expectation representation, information boundary, update rule, and anticipatory-response behavior under the frozen synthetic environment.

It does not establish human forecast accuracy, rational expectations, historical equivalence, political legitimacy, or real-world self-fulfilling dynamics.