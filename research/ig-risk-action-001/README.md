# IG-RISK-ACTION-001 — risk-sensitive action selection

**Status:** protocol and deterministic synthetic action-oracle fixtures; model trials not executed  
**Owner:** [Symthaea #11453](https://github.com/Luminous-Dynamics/symthaea/issues/11453)

## Research question

When the system receives the same information but the cost of error changes, does it select the appropriate action—answer, abstain, request evidence, escalate or defer—and how much decision regret does it incur against an independently declared oracle?

RiskEval (EMNLP 2026) reports a gap between verbal confidence and the decision to abstain under high error penalties. CALIBER (Findings EMNLP 2026) distinguishes pre-reasoning policy-success confidence from post-answer correctness confidence. SAGE (Findings EMNLP 2026) uses repeated output distributions to construct answer-conditioned uncertainty. These are external research results and not measured Symthaea capabilities.

References:
- https://aclanthology.org/2026.emnlp-main.759/
- https://aclanthology.org/2026.findings-emnlp.2420/
- https://aclanthology.org/2026.findings-emnlp.1214/

## v0.1 implementation

The corpus contains **40 synthetic decision scenarios**, covers five actions and four task families, and includes five matched low-/high-wrong-answer-penalty pairs. Those pairs hold probabilities, rewards and non-error costs constant while changing only the wrong-answer penalty.

For each action, the fixture declares a probability of eventual correctness under that action and a scenario-specific cost. The independent Rust oracle recomputes:

`EU(a)=p_a U_c-(1-p_a)C_w-C_a`

where (p_a) is the scenario-oracle probability for that action, (U_c) is the reward for a correct result, (C_w) is the wrong-answer penalty, and (C_a) is the action-specific cost. Utilities are exact rational numbers. Ties are resolved by an explicit per-fixture order.

**Scope limit:** evidence-adjusted, reviewer and deferred correctness probabilities are declared synthetic scenario inputs. This v0.1 contract does not estimate actual value of information, reviewer quality, future-state probability, or any live model's calibrated confidence.

## Qualification

The Rust qualifier verifies exact manifest and scenario counts, all five actions represented among optimal outcomes, probability/loss bounds, rational expected utilities, optimal action selection and tie resolution. It also verifies that matched penalty pairs hold every non-penalty input constant and change the optimal action when only wrong-answer cost rises.

Negative controls cover utility tampering, confidence-target substitution, action duplication, invalid probabilities, duplicate IDs, always-abstain mutation and matched-pair cost tampering.

## Composition

- #11367 owns probability calibration and selective-risk metrics.
- #11340/#11350 owns event-level revision receipts.
- #11320/#11323 owns deliberation exposure and stance trajectories.
- #11415 owns matched token/latency/tool budgets and transcript integrity.
- #11434/#11484 owns unauthorized-effect security outcomes.

A calibrated probability never becomes decision authority, civic standing, or permission to execute. Consequential actions still require a separate authority/capability path.

## Reproduction

```bash
cargo fmt --manifest-path research/ig-risk-action-001/qualifier/Cargo.toml --check
cargo test --manifest-path research/ig-risk-action-001/qualifier/Cargo.toml
cargo run --manifest-path research/ig-risk-action-001/qualifier/Cargo.toml -- \
  research/ig-risk-action-001/IG_RISK_ACTION_FIXTURES_V0_1.json \
  research/ig-risk-action-001/IG_RISK_ACTION_FIXTURES_V0_1.schema.json
```

## Claim ceiling

A PASS qualifies synthetic utility arithmetic, tie policy and action-oracle consistency only. It does not qualify model calibration, real evidence value, reviewer reliability, human acceptance, actual decision quality, or safety/deployment performance. Human studies and external model trials require separate preregistration, matched budgets, held-out tasks, independent outcome labels and uncertainty estimates.
