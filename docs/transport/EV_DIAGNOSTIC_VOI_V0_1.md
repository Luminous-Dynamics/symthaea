# EV Diagnostic Value-of-Information Evaluator v0.1

**Status:** Synthetic decision-engine prototype; not a clinical, safety or battery-service authority.  
**Owner:** Symthaea [EV-DIAGNOSTIC-VOI-001 #11277](https://github.com/Luminous-Dynamics/symthaea/issues/11277).

## Purpose

Given a probability distribution over battery-condition hypotheses, an admissible set of maintenance/inspection actions, and candidate diagnostic tests, calculate whether any test is expected to improve the decision enough to justify its declared cost and downtime.

The key distinction is between **information gain** and **decision value**. A test can perfectly identify a hidden state and still have zero decision value when every possible outcome leads to the same admissible action.

## Decision model

The benchmark uses a finite state set (s), prior probabilities (p(s)), admissible actions (a), and non-negative decision losses (L(a,s)), expressed in a declared common decision-loss unit.

The current best action is:

`a^* = argmin_a Σ_s p(s) L(a,s)`

For a candidate test (T) with outcomes (o) and likelihood model (P(o|s,T)), each outcome updates the state probabilities by Bayes' rule. The expected post-test decision loss is:

`R_T = Σ_o P(o|T) min_a Σ_s P(s|o,T) L(a,s)`

Gross expected value of sample information is:

`EVSI(T) = R_0 - R_T`

where (R_0) is the best current expected loss. Net value subtracts test and downtime costs, which must use the same decision-loss unit:

`Net(T) = EVSI(T) - test_cost - downtime_cost`

Only a positive-net-value test that is in-domain and has every precondition satisfied can be recommended. Otherwise the result is `NO_TEST`. When there is no admissible current action, the evaluator returns `ABSTAIN`.

The report also computes mutual information (I(S;O|T)), so the benchmark can expose cases where high information gain does not produce a better decision.

## Hard eligibility gates

A test is excluded (not merely penalized) when its interpretation method is unqualified, its chemistry/temperature/SOC domain does not cover the scenario, protocol/equipment/calibration identity or cost evidence is invalid, or any safety review, qualified-facility, protocol approval, calibration or responsible-authority prerequisite is not explicitly satisfied.

The evaluator only ranks proposals. It never books, initiates or authorizes a physical test.

## v0.1 baselines

- Current-evidence / no-test decision.
- Maximum mutual-information test.
- Maximum gross-EVSI test.
- Maximum net-EVSI test (the proposed selector).

This small synthetic suite proves decision arithmetic and fail-closed behaviour. It does not establish superiority on real battery service data. Later benchmark versions should add fixed-periodic, rule-based, random-eligible, expert-adjudicated and sequential policies under matched costs and held-out physical outcomes.

## Metrics and deterministic behavior

Reports include prior expected loss, clairvoyant lower bound, no-test regret, gross EVSI, total test/downtime cost, net value, post-test expected loss and regret, mutual information, selected test and ineligible-test reasons.

Scenarios and internal records are sorted by stable identifiers before calculation. The evaluator validates:
- prior probabilities sum to one;
- each action has exactly one loss for every declared state;
- every test outcome defines a likelihood for every state;
- likelihoods across outcomes sum to one separately for each state;
- IDs are unique and all values/costs are finite and non-negative;
- test applicability and all preconditions are satisfied before selection.

The frozen corpus contains positive, no-decision-value, out-of-domain, wrong-configuration, abstention and negative controls. All current scenarios and loss/cost values are synthetic.

## Important limitations

- The finite-state oracle is supplied by the benchmark fixture; it is not inferred from a battery.
- The v0.1 likelihoods are declared, not estimated or calibrated from field data.
- Each scenario binds the exact vehicle/pack configuration, observation-batch digest, pack-capability-profile digest, prior-model digest and decision-loss-model digest. Each test binds its target configuration, protocol, equipment, calibration and cost-source reference/digest. A configuration mismatch makes a test ineligible.
- A scenario's state hypotheses are treated as mutually exclusive and exhaustive. A test's outcome likelihoods must sum to one for every state.
- An action marked `gate_status: satisfied` must be independently approved as admissible across the full support of the scenario's declared prior. Expected loss cannot compensate for an unapproved safety action; blocked or unknown actions are excluded.
- Costs are synthetic decision-loss units; real deployment requires a reviewed normalization of financial cost, downtime, service risk and other impacts. The cost source and evidence digest are required even for synthetic fixtures.
- There is no electrochemistry, pack-topology or degradation simulation in this crate.
- No physical test, maintenance action, BMS change, charging change or safety conclusion is produced.

## Research interfaces

- SOH estimator and split owner: Symthaea #11252.
- Observation provenance: Symthaea #11249.
- Pack capability profile: Symthaea #11271 / PR #11274.
- Pack repair and architecture evaluation: Symtropy #2155.
- EV-REF-01 integration: Symtropy #4346.
- Generic multi-objective / active experiment design method: Symthaea #10143.

External benchmark/research references:
- 2026 fleet-scale cell inconsistency metrics: https://www.nature.com/articles/s41560-026-02131-5
- 2026 multi-fidelity SOH/RUL validation review: https://www.mdpi.com/2313-0105/12/7/255
- 2026 cross-dataset mixture-of-experts SOH evaluation on 564 cells from seven datasets: https://www.mdpi.com/2313-0105/12/9/359
- IEC 62660-1 cell performance/life testing: https://webstore.iec.ch/en/publication/28965
- ISO 12405-4 pack/system performance testing: https://www.iso.org/standard/71407.html

These standards and papers are external comparison references, not claims that the evaluator confers compliance or reproduces their results.
