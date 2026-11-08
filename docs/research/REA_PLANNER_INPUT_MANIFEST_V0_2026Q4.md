# REA Planner Input Manifest v0 and held-out outcome grader v0

**Issue:** [#11147 — REA-RUST-051](https://github.com/Luminous-Dynamics/symthaea/issues/11147)  
**Parent:** [#11108 — claim-obligation-directed evidence acquisition](https://github.com/Luminous-Dynamics/symthaea/issues/11108)  
**Composed planners:** #10074 point EIG, #10105 robust EIG, #10109 EVSI, #10174 exact finite-scenario minimax regret.  
**Status:** six-case synthetic development pilot; not human-adjudicated and not a hidden capability benchmark.

## Why this is a separate contract

The 18-case REA-RUST-050 fixture set carries the claim, mandatory obligations, candidate descriptions and admissibility rules. It does not yet provide all the numerical/model inputs required for a matched experiment-planner comparison. This sidecar adds those inputs without changing the claim-obligation fixture or cloning any planner.

The pilot includes six cases: high-EIG/non-decisive probing, common-source mirrors, pagination gaps, correlation versus causal intervention, authorization rejection, and bounded no-event evidence.

## Agent-visible planner manifest

`rea-planner-input-manifest-v0.json` carries:

- two synthetic evidence roots and two competing hypotheses with normalized priors;
- caller-supplied point-likelihood matrices for every candidate;
- two shared likelihood scenarios with complete candidate coverage;
- terminal decisions with finite utility for every hypothesis;
- the same authorization, reversibility, impact and cost information as the parent case;
- explicit planner time, energy, risk and impact budgets.

The Evidence IDs have the REA form, but the source records are synthetic algorithm fixtures. The caller-supplied origin labels are not signatures or proof of independent real-world observation. The likelihoods and utilities are not measured, calibrated or human-elicited.

## Grader-only held-out DGP manifest

`rea-planner-outcome-grader-v0.json` carries a deliberately different third likelihood model for each case, plus a synthetic realized hypothesis label. It is never a planner input. It allows the evaluation harness to measure the finite-scenario planners under a DGP that is not in the declared scenario ensemble.

The public grader file is still a development oracle, not an access-controlled hidden test set. A real benchmark must keep held-out gold and DGPs outside the public repository and outside agent/tool prompts.

## Mathematical invariants

Draft 2020-12 schemas validate local types and shapes. A semantic checker must additionally enforce:

1. Each case has unique evidence, hypothesis, candidate, scenario and terminal-decision IDs.
2. Every hypothesis prior is in (0,1] and total prior mass is 1.
3. Every hypothesis references at least one valid independently rooted evidence fixture.
4. Each likelihood vector has exactly one entry per declared outcome, finite values in [0,1], and total probability 1.
5. Every scenario has positive finite weight, scenario weights sum to 1, and every scenario contains each candidate exactly once.
6. Every matrix contains every hypothesis exactly once.
7. Every decision utility map contains every hypothesis exactly once, with finite utilities.
8. Candidate IDs, outcomes, authorization, reversibility, destructive flags and source impact exactly match the parent agent-view case.
9. Held-out DGP entries use the same candidate/outcome/hypothesis universes but remain inaccessible to planners.
10. The hidden DGP is not identical to a declared scenario across the complete case matrix.

The minimum pilot is a conformance exercise for adapters and mathematical input validation. It cannot establish that the invented likelihoods are calibrated or that the selected actions are useful in a real investigation.

## Fair comparison protocol

Run all policies against the same frozen agent manifest:

A. Random eligible candidate.  
B. Cost-only eligible candidate.  
C. Point EIG (#10074).  
D. Robust expected / worst / lower-tail EIG (#10105).  
E. EVSI policies (#10109).  
F. Exact finite-scenario minimax regret (#10174).  
G. Greedy mandatory-obligation closure.  
H. Obligation-gated decision-aware policy (composition, no planner fork).

Hard safety gates are applied before ranking. No planner may select a denied, irreversible, destructive, over-impact or over-budget candidate. No predicted outcome may close a claim obligation before a new observation is actually recorded and independently assessed.

## Metrics

Keep these distinct:
- point/robust predicted EIG;
- EVSI and supplied-scenario regret;
- held-out expected decision regret;
- false-ready rate and useful claim readiness;
- mandatory obligation closure under observed outcomes;
- abstention correctness;
- cost/time/energy/target impact;
- decision stability and scenario-omission penalty.

Exact minimax regret means exact over the supplied finite scenarios and bounded deterministic outcome-to-decision policies. It does not mean global worst-case robustness.

## Acceptance and next step

- Meta-validate both Draft 2020-12 schemas.
- Run semantic invariant and mutation tests.
- Confirm all six cases match the parent obligation agent-view candidates and outcome orders.
- Add an isolated Rust adapter that calls the existing planner APIs directly; do not copy their algorithms.
- Generate receipts through #10314, not a new receipt type.
- Human-adjudicate an expanded, source-family-disjoint dataset before any capability or SOTA claim.

No planner ranking, performance advantage, generalization, calibration, causal truth, or SOTA result is claimed by this public synthetic pilot.
