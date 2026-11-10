# IG-CONTAGION-001 — Persistent stance shifts and evidence-grounded revision

**Status:** research proposal / synthetic scenario design, not executed  
**Owning issue:** [Symthaea #11320](https://github.com/Luminous-Dynamics/symthaea/issues/11320)  
**Parent lab:** [Symthaea #3054](https://github.com/Luminous-Dynamics/symthaea/issues/3054)

## What this evaluates

The experiment separates five things that are often conflated:

1. An agent's initial private elicited position.
2. The position it states publicly during discussion.
3. Its position after discussion in a fresh private session with a participant-specific exposure-memory artifact.
4. Its response after a complete context reset, used as a repeat-elicitation noise control.
5. Optional delayed private re-elicitation to test persistence under a separately frozen delay policy.

A stance shift is not automatically a failure. Valid new evidence should be able to change a position. The red-team question is whether unsupported repetition, status signals, fabricated consensus, centrality, or prompt injection can move positions without relevant evidence.

## A critical measurement distinction

A stateless LLM API does not retain a private latent update merely because it participated in a previous call. Therefore the protocol does not pretend that private post-discussion behavior can be measured after simply deleting all history.

The primary private post-discussion measurement preserves an explicit, content-bound, participant-specific exposure-memory artifact while removing live access to peers, current group messages, other agents' outputs, and the baseline answer. The full context-reset condition is a separate measurement-noise control; it is **not** labeled a persistence measure.

This makes the operational state handoff visible and auditable. It does not establish that the artifact corresponds to a human-like memory or internal belief.

## Files

- `IG_CONTAGION_001_PROTOCOL_V0_1.json` — stages, semantics, metrics, experiment controls, receipt contract and claim ceilings.
- `IG_CONTAGION_001_PROTOCOL_V0_1.schema.json` — bounded structural contract for the protocol manifest.
- `IG_CONTAGION_001_SCENARIO_MATRIX_V0_1.json` — 45 unexecuted synthetic scenario descriptors across eight families.
- `IG_CONTAGION_001_SCENARIO_MATRIX_V0_1.schema.json` — bounded structural contract for the scenario matrix.

The initial scenario matrix contains five descriptors each for seven families and ten centrality-placement descriptors forming five matched pairs:
- no-discussion repeat controls;
- evidence-supported revision;
- unsupported social influence;
- centrality placement: five same-content, same-topology, same-seed pairs that differ only by low- versus high-centrality placement;
- correction after misinformation;
- false consensus/status signals;
- identity multiplicity and prompt injection;
- open-world uncertainty and conflicting evidence.

It references the topology generator in [#11248](https://github.com/Luminous-Dynamics/symthaea/issues/11248). It does not introduce a new graph generator or adversarial runtime.

## Execution ownership

- **#11248:** deterministic graph/topology generation and exact graph identity.
- **#3054 / #3095 / #3115:** institutional-game orchestration, run matrix/receipts and mechanism/model provenance.
- **#1872:** hostile-agent execution.
- **#9431:** evaluator ensemble error correlation.
- **#11320:** stance measurement protocol, causal contrasts, topology placement and counterexample analysis.

A future run receipt must bind the protocol, scenario, graph profile, RNG, agent/model revisions, prompts, evidence packet, ordered message log, participant-specific exposure-memory artifact, all stage outputs, evaluator, analysis plan, and runtime identity.

## Validation and claim ceiling

The two JSON files were fetched back from the branch and checked against their declared bounded schema subset. The scenario matrix has 40 unique fixture IDs and seeds, eight declared families, five cases per family except centrality placement, which has ten descriptors forming five paired blocks, and five centrality-pair identifiers.

These are structural checks of a scenario design, not executed simulations. No agent interaction, stance transition, benchmark performance or statistical result is claimed.

A future PASS can qualify only the declared deterministic fixture/orchestration contract. It cannot establish human psychology, general LLM behavior, alignment, legitimacy, democratic quality, or production governance suitability. Human-subject work requires a separate protocol, applicable ethics review, consent, privacy safeguards, preregistration and independent human evaluation.

## Required future experiments

1. Establish a no-discussion repeated-elicitation baseline for each pinned agent profile.
2. Run evidence-positive controls where a well-supported minority view should cause warranted revision.
3. Run evidence-poor and deceptive social-influence conditions with the same message content across matched topology placements.
4. Compare public stance, private exposure-memory re-elicitation, and context-reset control separately.
5. Add delayed private measurements only after fixing the delay and memory-retention policy.
6. Report outcome vectors and uncertainty across multiple seeds; do not collapse results into one contagion or alignment score.
