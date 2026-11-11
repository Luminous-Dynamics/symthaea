# IG-CAUSAL-PRIMITIVES-001 — component-level causal reasoning

**Status:** protocol and deterministic synthetic oracle fixtures; model trials not executed  
**Owner:** [Symthaea #11505](https://github.com/Luminous-Dynamics/symthaea/issues/11505)

## Research question

Can a system recover a causal structure, propagate an intervention through a structural causal model (SCM), update beliefs only when the evidence warrants it, and remain consistent when only irrelevant agency framing changes?

This track follows **CausalPrism** (Feng & Rudzicz, Findings of EMNLP 2026), which separates Causal Prior, Forward State Tracing, Belief Updating and Agency Sensitivity rather than hiding primitive failures inside one aggregate score. Its reported 14-model evaluation found Belief Updating to be the bottleneck: the best reported accuracy was 0.613, while GPT-5.2 and GPT-5-mini were reported at 0.531 and 0.517 under that paper's setup. These are external study results, not Symthaea results or portable thresholds.

Reference: https://aclanthology.org/2026.findings-emnlp.1294/

## What is implemented here

The v0.1 corpus contains **40 synthetic known-answer tasks**:

| Primitive | Tasks | Oracle |
|---|---:|---|
| Causal Prior | 10 | DAG edges derived from the ordered structural equations |
| Forward State Tracing | 10 | Exact Boolean state after a declared do-intervention |
| Belief Updating | 10 | Exact rational Bayesian posterior |
| Agency Sensitivity | 10 | Five matched pairs with identical SCM/intervention and changed narrative framing |

Tasks are distributed across mechanical/infrastructure, institutional/resource and abstract synthetic domains.

The Rust qualifier independently checks:
- unique task identity and exact per-primitive/domain counts;
- structural dependencies, parent validity and acyclic equation order;
- exact edge-oracle agreement for Causal Prior;
- intervention override semantics and exact downstream state for Forward State Tracing;
- rational Bayes update arithmetic without floating-point tolerance for Belief Updating;
- two-member matched agency pairs with invariant SCM, intervention and expected effect, but distinct framing.

## Experimental extensions

After the contract is qualified, model experiments should compare:
1. single-agent reasoning;
2. repeated independent samples under a matched budget;
3. standard multi-agent debate;
4. consensus-free trajectory evaluation;
5. evidence-gated debate;
6. posterior/evidence-guided communication routing.

Use the same frozen SCM, model revision, prompt, evidence, token/tool/latency budget and evaluator across matched arms. Record all run failures, timeouts, refusals, missing answers and indeterminate cases. A one-seed run is not a behavioral result.

### Required causal contrasts

- Observation versus intervention: compare P(Y|X=x) with P(Y|do(X=x)).
- Valid interventional evidence versus observational association.
- Confounded association versus genuine intervention.
- Evidence-bearing minority message versus unsupported social assertion.
- Correction after misinformation versus no correction.
- Actor/intentional wording versus passive-event wording with unchanged SCM.
- Agency framing that genuinely changes the mechanism, where invariance is not the expected answer.

## Metrics — no aggregate score

Report the four primitive families independently. Also retain false updates, over-rejection, evidence attribution, posterior calibration where a probabilistic target exists, abstention, confidence–accuracy divergence, model/task-family transfer and exact resource costs.

Do not reward stubbornness: valid new evidence should change the posterior. Do not reward conformity: repeated social claims without new evidence should not be treated as an evidence update. Normative pluralism remains outside single-answer causal correctness.

## Reproduction

```bash
cargo fmt --manifest-path research/ig-causal-primitives-001/qualifier/Cargo.toml --check
cargo test --manifest-path research/ig-causal-primitives-001/qualifier/Cargo.toml
cargo run --manifest-path research/ig-causal-primitives-001/qualifier/Cargo.toml -- \
  research/ig-causal-primitives-001/CAUSAL_PRIMITIVES_FIXTURES_V0_1.json \
  research/ig-causal-primitives-001/CAUSAL_PRIMITIVES_FIXTURES_V0_1.schema.json
```

## Claim ceiling

A structural PASS means only that the 40 checked-in synthetic tasks and their exact oracles satisfy the declared contract. It does not show that a model can infer causal structure, perform causal discovery, reason about real-world interventions, or generalize to humans or institutions. Those claims require executed model trials, matched baselines, multiple independent seeds/tasks, uncertainty intervals and external replication. Human studies require separate ethics review, consent, privacy safeguards and preregistration.
