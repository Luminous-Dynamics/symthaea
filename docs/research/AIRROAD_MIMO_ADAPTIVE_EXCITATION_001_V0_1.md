# AIRROAD-MIMO-ADAPTIVE-EXCITATION-001 — sequential model discrimination

**Status:** V0.1 synthetic exact-rational deterministic decision-tree reference  
**Tracking issue:** https://github.com/Luminous-Dynamics/symthaea/issues/11887  
**Composes with:** #11868 active excitation, #11865 model non-identifiability, #11858 joint uncertainty, ROB-DOE #4817, ROB-EXP #4800, ROB-SYSID #4801

## 1. From ranking to adaptive policies

A one-shot experiment ranking is insufficient when different measurements distinguish different pairs of remaining hypotheses. The next experiment must depend on the observation just received.

The object is a deterministic experiment decision tree:
- current admitted model set;
- approved candidate action;
- predicted observation sets under each model;
- observation-to-survivor update;
- next action or terminal model/abstention.

This is model discrimination under deterministic hard sets. It is not Bayesian EIG.

## 2. Three exact rational hypotheses

Two candidate cross-port terms are declared:

    Delta_3(s) = 1000*(s^2+1)*(s^2+4)*(s^2+16)/(s+5)^7
    Delta_2(s) = 1000*(s^2+1)*(s^2+9)*(s^2+16)/(s+5)^7

The three 2-port hypotheses differ only in the bus-a observed / bus-b driven cross-port entry:
- H0: zero;
- H1: Delta_3(s);
- H2: Delta_2(s).

Two experiments drive bus-b and observe bus-a:
- A2: omega=2 rad/s;
- A3: omega=3 rad/s.

At A2, H0 and H1 are exactly equal; H2 differs. At A3, H0 and H2 are exactly equal; H1 differs. Thus neither one-shot experiment identifies all three hypotheses.

## 3. Exact separation under hard observation disks

Each predicted complex observation is enclosed in a closed disk of radius 1/10. Two disks are guaranteed disjoint when squared center distance exceeds:

    (1/10 + 1/10)^2 = 1/25

At omega=2:

    |Delta_2(j*2)|^2 = 32400000000/17249876309 > 1/25

At omega=3:

    |Delta_3(j*3)|^2 = 612500000/410338673 > 1/25

Therefore the nonzero hypothesis disk is disjoint from the zero disk at each corresponding action. The equal-response hypothesis pair remains indistinguishable at that action.

## 4. Adaptive policy and exact cost

Choose A2 first (canonical tie-break):
- if the observation is in H2's disk, identify H2 and stop after one experiment;
- if it is in the shared H0/H1 zero disk, retain both and perform A3;
- at A3, the zero disk identifies H0 and the Delta_3 disk identifies H1;
- any observation outside all admitted sets is unresolved, never forced to the nearest model.

With the explicitly declared uniform prior:

    E[N] = (1/3)*1 + (2/3)*2 = 5/3 experiments

Worst-case cost is 2. A fixed two-action sweep always costs 2. The prior-conditioned expected reduction is:

    (2 - 5/3)/2 = 1/6 = 16.67%

This is an exact accounting result for this synthetic decision tree and this prior. Without a prior, report only the worst-case cost of 2.

## 5. Required semantics

- Prediction sets are closed deterministic complex disks with exact radii.
- A guaranteed distinction requires disjoint sets, not merely different nominal centers.
- Observation outside all model sets returns unresolved.
- An OTHER_UNRESOLVED model without a bounded prediction set prevents complete-set identification.
- Candidate actions are proposals tied to ROB-EXP #4800 and ROB-DOE #4817; they never become hardware commands or physical execution authority.
- Changing the model set, action protocol, uncertainty, topology, port mapping or prior changes the policy identity/result.

## 6. Industry/SOTA comparison

Robust/min-max experiment design and MIMO excitation design under continuous uncertainty are established:
- robust optimal experiment design for system identification: https://doi.org/10.1016/j.automatica.2006.12.013
- iterative robust MIMO experiment design via the S-Lemma: https://research.tudelft.nl/en/publications/iterative-robust-experiment-design-for-mimo-system-identification/

The AIRROAD contribution is an exact small-instance decision-tree oracle with explicit model-set ambiguity, hard prediction-set overlap and auditable cost accounting—not an assertion that adaptive experiment design is new.

## 7. Benchmark targets (targets, not results)

- 100% exact Fraction agreement with both rational model responses.
- Every hypothesis reaches a unique leaf with both actions approved and available.
- No complete identification when only A2 or only A3 is available.
- Zero guaranteed separation when hard prediction disks overlap.
- Zero forced selection when observations fall outside every admitted model set.
- Exact expected test count 5/3 and worst-case count 2 under the frozen uniform prior.
- Compare adaptive tree, fixed sweep, one-step greedy ranking and exhaustive small-tree oracle at identical action set/costs.
- Report expected cost only with a declared prior; report worst-case cost independently.
- No EIG claim without calibrated likelihoods and no physical authority.

## 8. Claim ceiling

Synthetic exact-rational adaptive policy only. It does not establish calibrated probabilities, model-set completeness, physical detectability, safe excitation amplitudes, aircraft converter identification, robust stability, flight safety, airworthiness or certification.
