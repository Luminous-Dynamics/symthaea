# IG-NUDGE-CALIBRATION-001 v0.1

**Owner:** Symthaea #11704  
**Status:** deterministic one-step cue-contract qualification only

## Research question

Can a decision agent use accurate information that changes the best-supported choice, avoid overreacting to accurate but decision-irrelevant observations or presentation-only cues, and defer unsupported factual claims?

The central design principle is selective responsiveness, not blanket resistance to influence.

## External comparator and source audit

Cherep, Maes and Singh, *AI Agents Are Sensitive to Nudges*, PNAS 123(25), e2537030123 (June 23, 2026): https://pmc.ncbi.nlm.nih.gov/articles/PMC13291645/

The published paper describes a sequential multiattribute task: hidden prize cells across baskets, weighted rewards, reveal actions that incur costs, and basket selection. It tests defaults, suggestions, information highlighting and resource-rational previews. The authors report that some LLMs pay excessive information costs, some ignore information, and some over-follow presentation cues; similar net rewards can conceal different search strategies.

The authors' public code repository is https://github.com/PapayaResearch/nudging and declares an MIT license for its code. That code license does not automatically establish participant-data or underlying human-data permissions. This benchmark commits no external human records or source dataset payloads.

**Important scope boundary:** this v0.1 package is a smaller one-step synthetic cue contract. It does not reproduce the PNAS multiattribute/sequential environment. The sequential 5x5 cell-reveal task, reveal-cost policy, early/late suggestions, defaults, highlight conditions, and resource-rational nudge reproduction remain the next stage.

## Workload

- 42 synthetic scenarios in six matched task blocks.
- Seven cue arms: no cue, verified high-value preview, verified low-value preview, unverified recommendation, default preselection, salience highlight, unsupported fact claim.
- Candidate-visible input contains only priors, the actual delivered cue and its evidence metadata. The evaluator's computed posterior is not handed to the candidate.
- Oracle-only realized payoffs remain separate from task and candidate payloads.
- Rust qualifier validates cue provenance, high/low value distinction, matched-block invariants and deterministic regret arithmetic.
- No LLM, human or model API was executed.

High-value previews are accurate observations that change the best choice under the prior-plus-observation calculation. Low-value previews are also accurate but leave the best choice unchanged. Presentation-only cues add no evidence. Unsupported factual claims are false under the oracle, have no supporting evidence reference, and require verification rather than being treated as an actual decision.

A REQUIRE_VERIFICATION response is a deferral, not a basket selection; its decision regret is null, not an invented zero.

## Local run

    cargo test --manifest-path research/ig-nudge-calibration-001/qualifier/Cargo.toml
    cargo run --manifest-path research/ig-nudge-calibration-001/qualifier/Cargo.toml --       research/ig-nudge-calibration-001/IG_NUDGE_CALIBRATION_PROTOCOL_V0_1.json       research/ig-nudge-calibration-001/IG_NUDGE_CALIBRATION_PROTOCOL_V0_1.schema.json       research/ig-nudge-calibration-001/IG_NUDGE_CALIBRATION_FIXTURES_V0_1.json       research/ig-nudge-calibration-001/IG_NUDGE_CALIBRATION_FIXTURES_V0_1.schema.json

The Rust validator implements the bounded contract; it is not a complete JSON Schema engine.

## Next research stage: faithful sequential replication

1. Extract and freeze the public reference task semantics: five prize attributes, baskets, weights, payoff matrix, initially revealed cells, per-cell reveal costs and tool/action protocol.
2. Define deterministic synthetic task generator and oracle, with identical task states across matched cue arms.
3. Implement reveal/select action replay and exact net-earnings computation.
4. Reproduce the four intervention families in the paper: defaults, early/late suggestions, information highlighting and resource-rational/optimal previews.
5. Evaluate both process and outcome: reveal count/cost, search coverage, selection, gross reward, net reward, nudge-following rates, regret and compute/latency.
6. Only then run pinned model tournaments and compare with the paper under matched model, prompt, seed, task, budget and statistical analysis.

A high-payoff model that over-reveals or blindly follows defaults must not automatically be described as behaviorally aligned. Conversely, a model that resists a useful preview can be under-responsive.

## Claim ceiling

A passing synthetic run validates this one-step cue contract and payoff arithmetic only. No sequential-task reproduction, model behavior, human effect, human-equivalent cognition, or production safety is established.
