# FEP-EFE-ACTION-002A: finite categorical expected information gain reference

**Status:** frozen candidate reference source; not independently qualified.  
**Parent:** [#6223](https://github.com/Luminous-Dynamics/symthaea/issues/6223)  
**Profile:** `FiniteCategoricalExpectedInformationGainV1`

## Objective

Define a finite-model software reference for expected information gain (EIG) over future observations and hypothetical posterior updates. This is a new objective profile, distinct from the current predicted-hidden-state entropy-delta proxy.

The work is intentionally staged:
1. Freeze this reference statement and the ordered corpus.
2. Qualify it with an independent oracle and hostile mutations on a separate exact-head qualifier branch.
3. Formalize theorem-sized probability identities within the already-approved SYM-FV Lean environment.
4. Only after the applicable baselines and lifecycle contracts are qualified, consider a production adapter.

## Frozen finite-model semantics

For a finite latent-state support (S), finite observation support (O), action (a), prior (q(s)), and action-conditioned likelihood (q(o\mid s,a)), define:

`q(o|a) = sum_s q(s) q(o|s,a)`

For every observation (o) with (q(o|a)>0):

`q(s|o,a) = q(s) q(o|s,a) / q(o|a)`

Then:

`EIG(a) = sum_o q(o|a) * KL(q(s|o,a) || q(s))`

The equivalent entropy identity is:

`EIG(a) = H(q(s)) - sum_o q(o|a) H(q(s|o,a))`

with (H(p)=-sum_i p_i\ln p_i), using the continuous convention (0\ln 0=0).

For the configured EFE minimization convention, the candidate information contribution is:

`E_info(a) = -EIG(a)`

This sign is explicit and is not inferred from a field name.

A zero-predictive-mass observation branch contributes zero to the expectation and **does not have a fabricated posterior**. Input probabilities are checked, not silently renormalized.

## Data contract

Canonical corpus: [`docs/fep/data/fep_efe_action_002a_reference_v1.json`](./data/fep_efe_action_002a_reference_v1.json)

- 35 ordered cases: 23 valid model/reference cases and 12 malformed-domain or schema-shape cases.
- 10 semantic mutation controls.
- Both expected-KL and entropy-reduction forms are required in independent qualification.
- State and observation labels are identity labels; consistent relabeling must preserve EIG.
- Probability inputs are decimal strings. The malformed shape control containing raw numeric JSON values must fail closed.
- Frozen normalization tolerance: (10^{-12}). No authority-bearing normalization is performed.
- Expected expressions use natural logarithms; (H) is defined in this document and in the corpus metadata.

The asymmetric-observation case (I35) prevents a mutant that drops the expectation over observation outcomes from passing merely because a symmetric binary channel gives equal contributions across branches.

## Required domain checks

Reject:
- empty state or observation supports;
- non-unique support identifiers;
- prior/likelihood dimension mismatch;
- negative, non-finite, or out-of-range probabilities;
- priors not summing to one within the frozen tolerance;
- any likelihood row not summing to one within that tolerance;
- unknown schema/profile/critical fields;
- a posterior construction on a zero-probability observation;
- hidden realized outcomes as candidate-time inputs.

## Semantic firewall

These are distinct:
- predicted hidden-state entropy delta;
- expected information gain under a frozen future-observation model;
- realized information gain after a real observation;
- a hypothetical posterior;
- the canonical live belief;
- accepted evidence.

The 001 profile is named `DiagonalGaussianPredictedEntropyDeltaV1`. It must not be promoted to EIG by renaming. EIG is only prospective information under the supplied observation model and is not by itself calibration evidence, physical information gain, proof of the Free Energy Principle, action optimality, experiment authorization, or causal truth.

Candidate evaluation must be pure: generating hypothetical posterior branches cannot mutate live belief, learning state, commitment/novelty history, evidence stores, or external world state. Pure probability algebra does not prove this implementation-level property; a separate state-purity/integration oracle is required.

## Independent qualification requirements

The separate 002A1 qualifier must:
1. bind this exact source commit, contract blob, corpus blob, schema, ordered case list, and numeric profile;
2. independently compute both EIG formulas without an `ID -> answer` lookup;
3. report all 35 cases and the two formulas' disagreement bound;
4. detect all declared input and semantic mutations, including the asymmetric branch expectation mutation;
5. keep qualifier changes outside the frozen source subject;
6. verify that the checked-out source SHA equals the triggering head and preserve execution output;
7. fail closed on changed source, blob, tolerance, profile, case ordering, or unknown authority-bearing fields.

Queued, skipped, cancelled, or partial workflows are not a qualification pass.

## Exploratory arithmetic (not qualification)

A separate transient JavaScript audit of the candidate data observed:
- 23 valid models and 12 rejected-domain/shape models;
- maximum expected-KL versus entropy-identity difference about (1.12\times10^{-16});
- maximum discrepancy from the authored symbolic expressions about (2.23\times10^{-16});
- all six declared metamorphic comparisons held;
- the asymmetric observation branches contributed different information ((approx0.05896) and (approx0.07355) nats).

This is a source/arithmetic sanity check only. It does not substitute for the independent 002A1 oracle, Lean kernel proof, hosted exact-head qualification, production correspondence, model calibration, or physical validation.
