# AIRROAD-MIMO-ACTIVE-EXCITATION-001 — model-discriminating frequency/port design

**Status:** V0.1 synthetic exact-rational experiment-selection reference  
**Tracking issue:** https://github.com/Luminous-Dynamics/symthaea/issues/11868  
**Composes with:** ROB-DOE #4817, ROB-EXP #4800, ROB-SYSID #4801, model-identification #11865, joint MIMO uncertainty #11858

## 1. Objective

Rank prospective impedance measurements by how strongly they distinguish currently admitted MIMO model hypotheses, while preserving measurement uncertainty, port mapping, domain, cost and test-approval constraints.

This issue owns an AIRROAD-specific proposal adapter. It does not create a second generic experiment optimizer or issue physical instrument commands.

## 2. Frozen model pair

Reuse the exact model pair from #11865:

    Z_A(s) = [2  0]
             [0  3]

    Delta(s) = 1000*(s^2+1)*(s^2+4)*(s^2+16)/(s+5)^7

    Z_B(s) = [2  Delta(s)]
             [0  3]

At omega={1,2,4} rad/s the two full matrices are equal. At omega=3 rad/s the bus-a observed / bus-b driven cross-port entry differs.

The candidate excitation set is {1,2,3,4} rad/s, all with the same synthetic cost, full observation coverage and zero declared measurement uncertainty. The exact model-pair separation is the Frobenius norm of the matrix difference.

## 3. V0.1 selection rule

For candidate a, compute the exact squared Frobenius separation:

    d²(a) = sum_(i,j in observed matrix coverage) |Z_A,ij(jw)-Z_B,ij(jw)|²

The selected candidate is the feasible candidate with maximum d²/cost, provided:
- candidate frequency is inside the model's declared validity domain;
- the candidate is approved by the supplied experiment protocol;
- the driven/observed port identities include the differing matrix element;
- the measurement uncertainty sets are guaranteed disjoint under the declared uncertainty metric.

In this deterministic fixture, the full MIMO plan at omega=3 wins. If omega=3 is absent, all remaining candidates have zero separation and the result is no guaranteed discriminating proposal.

This exact score is a small model-discrimination oracle, not an EIG estimate. Probabilistic EIG requires justified likelihoods; decision EVSI requires a declared downstream decision and utility. They are separate future objectives.

## 4. Uncertainty-aware distinguishability

For each model, the actual prediction is a set after measurement uncertainty and shared calibration dependencies. A positive nominal model separation is not a guaranteed discriminator when the two predicted sets overlap.

V0.1 uses a complex-disk outer uncertainty radius. A candidate is guaranteed-discriminating only when:

    distance_between_nominal_responses > radius_model_A + radius_model_B

The exact fixture uses zero radius for the winning case. A counterfactual radius of 1 for each model makes the 3 rad/s response sets overlap; this is NOT_GUARANTEED even though the nominal model predictions differ.

The next version must consume #11788/#11858 joint uncertainty objects rather than treating all channel errors as independent.

## 5. Port semantics

The distinguishing term is row=observed bus-a, column=driven bus-b. Therefore:
- injecting at bus-b while observing bus-a can see the distinction;
- diagonal-only measurement cannot see it;
- injecting at bus-a cannot see the differing column;
- omitting bus-a from observed outputs cannot see it.

The selector must preserve row/column conventions and never substitute a scalar diagonal response for a complete matrix decision.

## 6. Industry / SOTA baseline

Reuse the generic ROB-DOE owner #4817. Compare, on identical candidate sets/costs/model ensembles:
- fixed logarithmic or uniform frequency sweeps;
- random feasible selection;
- maximum local parameter sensitivity / Fisher information;
- nominal pairwise model separation;
- worst-case set separation;
- nominal EIG only where calibrated likelihoods exist;
- worst-case decision EVSI only where utility and outcomes exist;
- exhaustive candidate oracle for small instances.

The claimed improvement is reduced experiment budget to distinguish the admitted MIMO model set under the same risk/cost constraints. It is not a universal assertion that adaptive selection beats all fixed sweeps or that the candidate set includes the true physical model.

## 7. Required negative controls

- Removing the only discriminating frequency causes abstention.
- Diagonal-only or wrong-driven-port coverage cannot receive positive cross-port separation.
- A missing observed cross-port term cannot create a score.
- Overlapping uncertainty sets are not guaranteed distinguishable.
- Out-of-domain or unapproved experiments are rejected.
- Missing/nonpositive cost and duplicate candidate IDs fail closed.
- An unresolved OTHER model class prevents a complete model-identification claim.
- A proposal cannot deserialize as an executable hardware command.

## 8. Benchmark targets (targets, not results)

- 100% exact Fraction agreement with the small candidate oracle.
- Correct selection of omega=3 in the frozen four-candidate fixture.
- Correct no-proposal disposition without omega=3.
- Zero guaranteed-discrimination labels when predicted uncertainty sets overlap.
- Zero cross-port distinctions lost by confusing diagonal and full-MIMO coverage.
- Report model-pair coverage, worst-case separation, cost, query budget, unresolved model mass and downstream decision impact separately.
- Compare adaptive selection to fixed sweeps, random, Fisher/EIG/EVSI and exhaustive small-case oracle at matched budget.

## 9. Claim ceiling

V0.1 is a synthetic exact-rational model-discrimination reference. It does not establish safe physical excitation amplitude, operate test equipment, prove the model set is complete, guarantee a real system can be identified with one test, or establish aircraft stability, safety, airworthiness or certification.
