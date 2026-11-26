# HIST-CAUSAL-001 v0.2: exact identifiability fixtures

**Status:** research reference; 9 unit tests pass locally.  
**Boundary:** no physical evidence and no inference that Chain Theory entails cross-system historical influence.

## Model set

Six competing finite causal explanations:
1. latent common cause;
2. ordinary transfer through an open channel;
3. candidate sealed-channel historical effect;
4. conventional but unobserved leakage;
5. no cross-cohort effect;
6. ordinary same-person local memory.

The same six mechanisms are evaluated under observational, randomized open-channel, randomized sealed-channel, independent leakage-audit, same-person memory and future-exposure placebo contexts.

## Exact result

All probabilities use rational arithmetic.

| Probe suite | Pairwise distinctions | Unresolved pairs |
|---|---:|---|
| Observation only | 8/15 | Common cause vs each transfer-like model; open transfer vs history/leak; history vs leak; null vs local memory |
| Interventions, no independent leakage audit | 13/15 | Sealed history vs hidden leakage; null vs local memory |
| Full probe menu incl. leakage audit and local-memory probe | 15/15 | None among the six declared models |

Critical counterexample: the sealed-history and unobserved-leakage models have exactly the same predicted outcomes under observation, open-channel intervention and sealed-channel intervention. A sealed-channel effect alone cannot identify a new nonlocal mechanism. The models become statistically distinguishable only after including an independent leakage measurement. That audit has 95% sensitivity and 1% false-positive probability in this toy model; individual audit outcomes are not definitive.

The exhaustive weighted set-cover oracle finds a minimum-cost full diagnostic design under the declared arbitrary toy costs:
- observational probe;
- randomized open-channel intervention;
- sealed-channel intervention with independent leakage audit;
- same-person local-memory probe.

Total toy cost: 9/2 = 4.5 units. These costs exist only to test the optimizer and are not experimental cost estimates.

## Required invariants

- Every model/context distribution sums to exactly one.
- Observational common-cause and direct-transfer mechanisms are equivalent under the observational probe alone.
- Randomized interventions break that observational equivalence.
- Hidden leakage and candidate sealed-history mechanisms remain equivalent without a leakage audit.
- Future exposure is a placebo for every declared causal model.
- Full probe menu separates all 15 model pairs, in the distribution-identifiability sense.

## Execution evidence

Local command:
`python -m unittest -v test_hist_causal_v02.py`

Result: 9 tests passed. Exact rational arithmetic; no external numerical package required.

The downloadable research reference and test are supplied in the conversation artifacts. The Python program is a finite reference oracle only. Port the model/context contract to Rust before integrating with Symthaea or Symtropy production paths.

## Relation to existing Symtropy work

Do not duplicate generic active experiment selection. Use the existing SIM-REL-002/003 grammar and information-gain work, SIM-REL-013/014 joint stochastic inference and benchmark work, SIM-REL-015 robust model-indexed replay, SIM-REL-017 challenge generators, and SIM-REL-025 apparatus/action-realization semantics as appropriate. HIST-CAUSAL contributes the distinct causal-history and hidden-leakage mechanism fixtures.