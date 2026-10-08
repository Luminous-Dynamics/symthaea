# Institutional Evolution Methodological Controls

**Status:** research design addendum v0  
**Related:** #7046, #7047

## The additional methodological finding

Recent work makes a useful distinction for this laboratory:

1. institutional rules themselves can materially change outcomes even when the agents and incentives are held fixed;
2. endogenous institution formation can improve coordination without necessarily reaching a Pareto-optimal result;
3. initial conditions can shape the institutional forms that emerge and can also generate power and legitimacy as emergent phenomena;
4. protocolized multi-agent simulation requires explicit roles, information boundaries, rule controls, structured outputs, replication manifests, and validation targets if it is to support quantitative research.

Relevant sources:

- https://arxiv.org/abs/2608.04020
- https://www.sciencedirect.com/science/article/pii/S0167268125000630
- https://journals.aom.org/doi/full/10.5465/amr.2021.0045
- https://ojs.aaai.org/index.php/AAAI/article/view/41262
- https://www.annualreviews.org/content/journals/10.1146/annurev-economics-091823-031317

These findings imply that the Symthaea laboratory needs a stricter causal decomposition than a single evolving-agent simulation.

## Control A — institution effect vs agent effect

Run a crossed design:

| | Agent model A | Agent model B |
| --- | --- | --- |
| Institution I | A/I | B/I |
| Institution J | A/J | B/J |

The institution comparison must be possible with the agent model held fixed.

The agent-model comparison must also be possible with the institutional mechanism held fixed.

Do not attribute:

```
A/I != B/J
```

to the institution when both the institution and the agent model changed.

## Control B — proposal-space audit

An institution can only emerge if the mutation grammar can express it.

Therefore record:

- mutation grammar version;
- reachable institution classes;
- mutation depth;
- proposal frequency;
- rejected proposals;
- inaccessible mechanisms.

A mechanism absent from the emergent distribution is not evidence that agents rejected it if the mechanism was unreachable.

Required disposition:

```
Unreachable
!=
ProposedAndRejected
!=
ProposedAndNotConsidered
!=
AdoptedAndFailed
!=
AdoptedAndSucceeded
```

This is one of the most important anti-bias controls in the entire program.

## Control C — institutional expectations

Institutions are prospective as well as present.

Agents may form beliefs about:

- probability of proposal adoption;
- probability of rule persistence;
- expected future access;
- expected future contribution recognition;
- expected future resource availability;
- expected enforcement;
- expected governance change.

Those beliefs can alter current behavior and therefore influence the institution that eventually emerges.

Record an explicit expectation state where the chosen agent model supports it.

Do not provide actual future outcomes as observations.

## Control D — bounded rationality

The 2025 weakest-link institution-formation evidence is especially relevant because observed institutional formation did not simply follow the fully rational Pareto-optimal prediction; bounded rationality provided explanatory value.

Therefore the benchmark should include at least:

- deterministic best-response baseline;
- noisy/bounded-response baseline;
- memory-limited baseline;
- heterogeneous-response baseline.

Do not hard-code perfect optimization as the default explanation of institutional emergence.

## Control E — initial-condition sensitivity

Institutional emergence may be path-dependent.

Repeat each problem with:

- different seed institutions;
- different network topologies;
- different initial resource distributions;
- different initial belief states;
- different mutation orderings.

Report whether the final institution is:

- invariant;
- seed-sensitive;
- path-dependent;
- multi-equilibrium.

A converged institution across seeds is interesting.

A divergent institution across seeds is also interesting.

Do not treat divergence as a failed experiment.

## Control F — emergent power and legitimacy

Power and legitimacy should be **measured outcomes**, not baked-in assumptions.

Possible operational observables include:

### Power

- control over scarce resources;
- veto frequency;
- agenda-setting frequency;
- information bottleneck control;
- rule-amendment influence;
- dependency created by institutional position;
- concentration of effective decision rights.

### Legitimacy

Use observable proxies only, such as:

- voluntary adoption rate;
- continued compliance;
- appeal success/acceptance;
- exit pressure;
- re-adoption after expiry;
- participation in amendment;
- explicit support/opposition measures.

Do not convert these observables into the claim that an institution is morally legitimate.

The distinction is:

```
observed acceptance
!=
political legitimacy
```

## Control G — emergence vs viability

Separate:

```
can be generated
-> can be adopted
-> can be implemented
-> can persist
-> performs under known shocks
-> performs under unseen shocks
```

A proposal that never emerges may be inaccessible.

A proposal that emerges but fails implementation is not the same as one that never had support.

A proposal that performs on training shocks but fails on holdout shocks may be overfit to the problem generator.

Therefore use a train/holdout distinction for institutional evolution campaigns.

## Control H — holdout shocks

Institution proposals and adoption decisions may observe the current problem history but must not observe a hidden evaluation schedule.

Freeze a holdout shock set at experiment initialization.

After institutional selection:

- expose the evolved institution to unseen shocks;
- compare with fixed and planner-designed controls;
- retain the exact lineage used to reach the holdout condition.

This gives the laboratory a basic analogue of out-of-sample evaluation.

## Control I — placebo problems

Include recurring problems for which a given institutional mutation should have no causal relevance.

Example:

```
monitoring mutation
+
fully observable quality problem
```

A monitoring rule may still be adopted because of the proposal process, but the model should not report a causal improvement in hidden-information resolution when no hidden information exists.

Placebo fixtures are required to detect institutions that become universal because of generic selection pressure.

## Control J — causal intervention on an emerged institution

Once institution V is adopted, run:

```
world_1: V remains
world_2: V removed
world_3: V replaced by one declared mutation
```

Hold everything else fixed where the counterfactual permits.

This distinguishes:

- emergence correlation;
- persistence correlation;
- downstream causal effect.

The lineage itself is not causal proof.

## Control K — no hidden institutional evaluator

The evaluator must never feed back its final score to the proposal mechanism.

Forbidden:

```
candidate -> evaluator -> score -> same candidate generator
```

unless that feedback loop is itself the declared treatment.

Otherwise the benchmark becomes an optimization loop trained against its own oracle.

## Control L — deterministic base before LLM augmentation

The first institutional-evolution implementation should not require a language model.

Start with:

- structured agent state;
- explicit bounded response rules;
- explicit mutation grammar;
- explicit adoption procedure;
- deterministic event scheduling;
- content-addressed lineage;
- independent oracle.

An LLM-driven agent layer can later be introduced as a controlled agent-model treatment.

When it is introduced, bind at least:

- model identity/version;
- prompt/role manifest;
- context/input manifest;
- information boundary;
- decoding configuration;
- structured output schema;
- parser/version;
- random seed where applicable;
- refusal/invalid-output handling;
- replay manifest.

A language model should increase behavioral expressiveness, not weaken causal identifiability.

## Minimum scientific result set

A useful first campaign should produce:

1. a fixed-institution baseline;
2. an evolution-enabled treatment;
3. an ablated proposal generator;
4. an ablated adoption mechanism;
5. multiple initial states;
6. multiple seeds;
7. holdout shocks;
8. a full institutional lineage;
9. outcome vectors;
10. proposal-space coverage;
11. expectation state, when modeled;
12. power/concentration observables;
13. explicit failure dispositions.

## Strongest claim

The strongest result the laboratory should seek initially is not that one institution is optimal.

It is:

```
under declared problem class P,
agent model A,
mutation grammar M,
adoption mechanism G,
and information regime I,

institutional families X/Y/Z
are reachable,
X/Y/Z are proposed at measured frequencies,
adoption differs under G,
persistence differs across seeds,
and downstream outcomes differ under controlled counterfactual interventions.
```

That is a reproducible statement about an institutional generative process.

It is deliberately weaker than a claim about what society should choose.
