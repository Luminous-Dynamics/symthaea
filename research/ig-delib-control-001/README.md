# IG-DELIB-CONTROL-001 — evidence-aware debate allocation and stopping

**Status:** deterministic synthetic reference-policy design; no LLM debate executed  
**Owner:** [Symthaea #11383](https://github.com/Luminous-Dynamics/symthaea/issues/11383)  
**Runner / receipt owner:** [#3095](https://github.com/Luminous-Dynamics/symthaea/issues/3095)

## Paradigm

A debate is a sequential evidence-acquisition process. Stop because evidence and uncertainty justify stopping—not because agents agree, a round budget expires, or a model is confident.

The v0.1 reference controller is transparent and rule-based. It chooses among evidence requests, claim verification, adversarial challenge, minority preservation, independent adjudication, hypothesis diversification, supported finalization, indeterminate finalization, pluralistic multi-position output, and explicit incomplete/invalid stops. It does not train a policy, call an LLM, or claim superiority.

## Deterministic scenario corpus

40 synthetic scenarios span eight families: factual evidence closure, false consensus, minority preservation, open-world uncertainty, normative pluralism, controller/orchestrator adversaries, budget/cost, and trajectory value.

30 negative controls mutate a valid scenario and require exact violation codes for malformed truth profiles, budgets, round counts, evidence flags, unresolved counts, minority semantics, adversary flags, confidence signals, expected actions and baseline outcomes.

## Reference policy

Precedence is explicit:
1. invalid controller input → `StopInvalid`;
2. zero budget → `StopBudgetExhausted`;
3. orchestrator/source-lineage compromise → `IndependentAdjudication`;
4. common-mode signal → `AdversarialChallenge`;
5. task-specific evidence and uncertainty policy.

For factual tasks, preserve a verified minority before finalization; verify unresolved contradictions; request missing evidence (or diversify a single hypothesis); finalize only after evidence closure.

For open-world tasks, unresolved facts remain indeterminate. For normative pluralism, the reference policy never emits a single factual correctness conclusion; it preserves dissent and returns multiple defensible positions.

Agent agreement and confidence are recorded for baseline comparisons but do not control the reference policy. Near-unanimity cannot override incomplete evidence.

## Baselines and measurements

Every fixture computes an agreement-stop baseline (`agreement >= 0.8`), a fixed-round-stop baseline (`rounds_used >= round_limit`), and the evidence-closure reference action. This supports later measurement of premature stopping, wasted rounds, unsupported consensus, minority loss and budget exhaustion. The 40 descriptors are policy test inputs, not agent trials.

## Research comparators

- [Meta-Moderator, Findings EMNLP 2026](https://aclanthology.org/2026.findings-emnlp.1250/) — separately trained meta-cognitive moderator and adaptive debate allocation.
- [Free-MAD, Findings ACL 2026](https://aclanthology.org/2026.findings-acl.1600/) — consensus-free full-trajectory decision.
- [Demystifying Multi-Agent Debate, Findings ACL 2026](https://aclanthology.org/2026.findings-acl.1694/) — initial diversity and calibrated confidence communication.
- [Beyond Consensus, Findings EMNLP 2026](https://aclanthology.org/2026.findings-emnlp.243/) — role-asymmetry bias in subjective multi-agent judging.
- [MADBench, 2026 preprint](https://arxiv.org/abs/2609.39146) — layered debate-security taxonomy.

These are comparison targets. No reproduction or superiority claim is made by this synthetic reference-policy crate.

## Run

```bash
cargo fmt --manifest-path research/ig-delib-control-001/qualifier/Cargo.toml --check
cargo test --manifest-path research/ig-delib-control-001/qualifier/Cargo.toml
cargo run --manifest-path research/ig-delib-control-001/qualifier/Cargo.toml -- \
  research/ig-delib-control-001/DELIB_CONTROL_SCENARIOS_V0_1.json \
  research/ig-delib-control-001/DELIB_CONTROL_NEGATIVE_CONTROLS_V0_1.json
```

## Claim ceiling

A structural PASS means the reference policy is deterministic on the declared synthetic fixtures and its negative controls are rejected. It does not establish LLM debate quality, controller efficacy, causal effects, human behaviour or production readiness. Next steps are matched comparisons against fixed-round debate, agreement stopping, Free-MAD and Meta-Moderator under exact token/tool/model budgets. No Mycelix production moderation, voting, authority or delegation policy is changed.
