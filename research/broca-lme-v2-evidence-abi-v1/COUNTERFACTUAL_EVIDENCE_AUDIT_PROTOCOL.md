# Broca Counterfactual Evidence Audit Protocol v0.1

**Status:** research protocol and fixture contract only. No semantic verifier, official worker integration, benchmark score, or SOTA gain is claimed by this document.

**Parent work:** [PR #8876](https://github.com/Luminous-Dynamics/symthaea/pull/8876) role-set selection; [PR #8894](https://github.com/Luminous-Dynamics/symthaea/pull/8894) payload-span-bound role witnesses with source-digest association; [issue #8913](https://github.com/Luminous-Dynamics/symthaea/issues/8913).

## 1. Research question

Can Liquid Broca distinguish a collectively supported answer from a set of individually relevant, byte-valid but semantically insufficient memories, under a fixed evidence/context budget?

The target is not to maximize the fraction of declared roles marked complete. The target is to reduce unsupported answers while preserving the ability to answer when evidence is sufficient. Evaluate support, answerability, and abstention as separate outcomes.

This protocol introduces a **paired counterfactual unit**: two cases share the same query and nearly the same evidence, but one controlled change alters whether the requested answer is supported. A system that simply retrieves relevant text or trusts caller-supplied role labels may score well on easy cases while failing the changed condition.

## 2. Evidence and decision layers

Record the following as distinct fields; never collapse them into one Boolean:

1. **Integrity:** the committed source digest remains bound to the evidence record; the current payload digest and selected payload-span digest are independently recomputed and validated.
2. **Eligibility:** scope, lifecycle, authorization, and temporal validity permit the record to contribute.
3. **Role coverage:** supplied requirement-role labels are covered by supplied candidate-role annotations.
4. **Semantic support:** the selected evidence entails or otherwise justifies the specific answer claim in context, including polarity, time, entity, and relevant relations.
5. **Decision:** answer, abstain/escalate, or bounded refinement.
6. **Answer correctness:** downstream answer judged against the frozen reference/evaluation protocol.

The selector's `complete=true` is layer 3 only. A payload-span digest establishes byte identity within the selected evidence payload, not raw-source offset mapping, entailment, source authority, truth, independence, or freshness. An integrity error must not be relabeled as low coverage or an ordinary abstention.

## 3. Versioned fixture format

Store fixtures as deterministic UTF-8 JSONL. One row is one pair; a pair contains two independently executed cases. Use a fixed schema version and reject unknown fields in the generator/validator. The generator emits only the data needed for the memory write/query path. Any byte-range witnesses in this protocol are defined against the exact text payload supplied to the selector, not automatically against the original trajectory JSON or raw source bytes. Keep gold labels in a separate evaluator-only file inaccessible to the system under test.

Canonical input corpus: `crates/domains/symthaea-broca/tests/fixtures/counterfactual-evidence-audit-v1.jsonl`. The current strict Rust schema includes an `as_of` timestamp and explicit `source_family`, `entity_id`, and `scope` on every evidence record. The example below is intentionally short; executable examples in the fixture file are authoritative for field names and mutation paths.

```json
{
  "schema_version": "BROCA-COUNTERFACTUAL-EVIDENCE-AUDIT-V1",
  "pair_id": "negation-0001",
  "family": "polarity_negation",
  "query": "Did service svc-17 recover after restart?",
  "as_of": "2026-10-29T12:00:00Z",
  "cases": [
    {
      "case_id": "negation-0001-a",
      "evidence": [
        {
          "evidence_id": "event-1",
          "source_id": "source-1",
          "source_family": "ops-ticket-17",
          "entity_id": "svc-17",
          "scope": "prod-east",
          "observed_at": "2026-10-29T10:00:00Z",
          "valid_until": null,
          "text": "After the restart, service svc-17 recovered."
        }
      ]
    },
    {
      "case_id": "negation-0001-b",
      "evidence": [
        {
          "evidence_id": "event-1",
          "source_id": "source-1",
          "source_family": "ops-ticket-17",
          "entity_id": "svc-17",
          "scope": "prod-east",
          "observed_at": "2026-10-29T10:00:00Z",
          "valid_until": null,
          "text": "After the restart, service svc-17 did not recover."
        }
      ]
    }
  ],
  "mutation": {
    "kind": "polarity_flip",
    "changed_fields": ["evidence[0].text"],
    "expected_decision_relation": "claim_must_flip_or_be_rejected"
  }
}
```

The `changed_fields` list uses exact structural paths from the validator. For a fixed-length evidence array, a text mutation is `evidence[0].text`; when list length changes, the list path `evidence` is declared. The validator rejects undeclared differences, extra differences, duplicate IDs, malformed timestamps, unknown fields, evaluator-label fields, and unsupported mutation families/relations.

The evaluator-only labels are physically separate at `crates/domains/symthaea-broca/tests/fixtures/evaluator-only/counterfactual-evidence-audit-labels-v1.jsonl`. The production validator does not open this file. `CounterfactualPairFixture::runtime_case(index)` returns only query, as-of time, and evidence; it omits pair IDs, mutation metadata, expected decision relations, and all gold labels.

This is an illustrative schema, not a released dataset example. Replace timestamps, identifiers, and evidence with synthetic or properly licensed material in committed fixtures. The evaluator-only label file should map each case ID to: answerability, the required claim(s), supporting/rebutting source spans, expected current state, expected action/abstention, and the applicable mutation invariant.

### Required schema/fixture rules

- Pair members have the same query, query image when present, reader configuration, allowed write interface, and context-token budget.
- The generator records a machine-readable diff of the intended mutation. It must reject a pair where any undeclared fields differ.
- Evidence identities, source identities, scope, timestamps, and source-family relationships are explicit. Duplicate projections of one source do not count as independent corroboration.
- A reference answer and gold evidence annotation never enter memory ingestion, candidate ranking, role extraction, refinement, or the answer prompt.
- Stable IDs and case ordering derive from canonical fixture content, not runtime randomness.
- Every run pins fixture digest, generator revision, system commit, selector/verifier configuration, reader/model revision, evaluator revision, prompt, tokenizer and final serialized context-token count.
- The runner stores actual inputs and outputs needed to reproduce decisions, with secrets and private user content excluded.

## 4. Counterfactual families

Implement a positive control and a minimally changed negative/alternate case for each family.

| Family | Controlled mutation | Expected check |
|---|---|---|
| Polarity | Add/remove negation or change a status predicate | Must not treat text similarity or a span witness as positive support for the opposite claim |
| Temporal validity | Supersede, expire, or move an observation across the query's time boundary | Old evidence may remain historically quotable but must not silently become current state |
| Entity binding | Change the entity ID/name while preserving the predicate and surrounding wording | Evidence for entity A must not satisfy a requirement about entity B |
| Scope / premise | Change environment, tenant, deployment or authority scope | Cross-scope evidence is ineligible even if text is identical |
| Correction / conflict | Add a later correction, revocation, or unresolved conflicting active claim | Current-state claims need explicit resolution policy; conflict cannot count as corroboration |
| Source dependence | Duplicate one underlying source into multiple records/projections | Duplicate rows must not create independent-source credit |
| Missing requirement | Remove exactly one essential fact, value, relation, or constraint | Missing information remains missing; role labels must not invent it |
| Ordered transition | Swap or remove an intermediate state/action while leaving endpoints similar | Workflow/transition answer must respect order and cannot infer an unobserved transition |
| Qualifier truncation | Keep a matching phrase but remove surrounding negation, condition, exception, or attribution | Isolated lexical match must not override semantics carried by the omitted qualifier |
| Integrity/lifecycle | Mutate payload bytes, source mapping, digest, lifecycle or authorization metadata | The responsible integrity/eligibility gate rejects or surfaces the failure; no silent fallback |

Each family should include (a) clear answerable controls, (b) structurally complete but semantically unsupported cases, and (c) cases where abstention is correct. Always-abstain and always-answer policies are mandatory diagnostics so neither abstention nor answering everything can masquerade as robust performance.

## 5. Test levels

### Level A — deterministic structural invariants

Use synthetic fixtures to test the software contracts without a model or judge:

- Every seeded payload/span/source-digest mutation is detected at the layer owning that invariant.
- UTF-8 offsets are valid, non-empty, and byte-exact.
- Unknown IDs, missing/extra witness mappings, invalid scope, inactive lifecycle and corrupt eligible payloads never contribute to selected evidence.
- Candidate permutation and irrelevant-candidate insertion do not change the result where the policy contract promises deterministic invariance.
- Repeated records from the same source family do not increase independent-source counts.
- Missing a required role produces a missing requirement; it cannot produce semantic completeness.
- An integrity error remains distinguishable from insufficient evidence.
- Final serialized context is recounted with the pinned tokenizer and is rejected if it exceeds the configured budget.

For this level, require zero forbidden emissions or silent integrity fallbacks in the deterministic fixture corpus. This is a property-test target, not proof against all faults or filesystem failure modes.

### Level B — semantic minimal pairs

Use manually authored and independently reviewed cases covering polarity, recency, entity, scope, contradiction, relation and transition semantics. For each family, have an annotator establish the intended mutation and a second reviewer verify that no unrelated change determines the label. Keep training/development and held-out test source families disjoint where possible.

Report annotator agreement and adjudication policy. The span verifier should be evaluated for byte fidelity; a separate semantic role/support component should be evaluated for whether the claim follows from the span plus required context. Do not score a semantic model based only on synthetic template recognition.

### Level C — official and external evaluations

First qualify the Rust invariants and produce a benchmark-native adapter for the official LongMemEval-V2 contract:

- `insert(trajectory)` receives trajectories only.
- `query(query, query_image=None)` receives query text and optional query image only.
- Return only supported context objects (`type/value`) permitted by the harness.
- Respect the harness's `--memory-context-max-tokens` limit after final serialization.
- Do not expose benchmark IDs, reference answers, evidence labels or evaluator state at query time.

Run the official reader and baseline configurations, including the packaged strong baseline where feasible. Separately run MemoryAgentBench and LongMINT for incremental update/forgetting and interference stress. Never combine scores across benchmark versions into one leaderboard number.

## 6. Required metrics

Always disaggregate by mutation family and report the denominator:

- Integrity-gate pass/fail and forbidden-evidence emission count.
- Payload-span verification accuracy / exact byte integrity.
- Source-record binding (payload receipt carries the expected committed source digest).
- Raw-source traceability (payload span is mapped to and independently verified against original source bytes); report as unavailable until an explicit mapping/locator is implemented.
- Semantic role precision/recall and false-role rate.
- **False-coverage rate:** cases declared covered/complete for which at least one essential answer requirement lacks genuine support.
- **Unsupported-answer rate:** answered cases whose material answer claims lack support.
- Answer accuracy conditional on answering.
- Answerable-case coverage and false-abstention rate.
- Paired counterfactual consistency: fraction of pairs where the decision relation matches the pre-registered expected relation.
- Contradiction, stale-state, wrong-entity and cross-scope detection.
- Requirement/evidence recall and visibility of annotated supporting source spans in the final serialized context.
- Actual prompt/context tokens, retrieval/refinement calls, p50/p95/p99 latency, memory/index bytes and cost per query.
- Paired question-level confidence intervals, failure traces and all preregistered exclusions.

For selective answering, publish risk–coverage curves and compare at matched coverage. Where a probability estimate exists, additionally report proper scoring/calibration metrics such as Brier score and ECE; do not interpret the deterministic role-coverage flag as a probability.

## 7. Baseline matrix

| Arm | Purpose |
|---|---|
| No-memory | Detect whether stored memory adds value |
| Score-prefix retrieval | Ordinary relevance baseline |
| Role-set selection | Isolate gains from explicit coverage annotations |
| Payload-span witnesses | Isolate payload byte integrity and source-digest association; not raw-source offset mapping or semantic correctness |
| Semantic support/contradiction gate | Test the semantic hypothesis when independently implemented |
| One bounded refinement round | Determine whether targeted gap recovery improves quality per cost |
| Always answer | Lower-bound diagnostic for abstention policy |
| Always abstain | Upper/lower bound diagnostic for risk/coverage interpretation |
| Oracle labels | Diagnostic ceiling only; prohibited at inference time |

Hold memory writes, candidate pools, reader, prompt, tokenizer, context budget, and model settings constant when isolating the selector. For refinement, include the full cost of planning and additional retrieval. An oracle is not a competitor and must never enter the inference path.

## 8. Decision rules and pre-registration

Before accessing held-out results, publish the evaluator version, primary metric, acceptable answerable-case coverage, maximum false-support rate, paired test, confidence-interval method, and cost cap. Select thresholds on the development split only.

A tranche advances only if:
1. Level A invariants pass at the exact commit being evaluated.
2. The new method reduces false-coverage or unsupported-answer risk against score-prefix and payload-span-witness baselines at a matched context budget.
3. The improvement does not come from always abstaining, leaked labels, extra context, a stronger reader, or hidden additional model calls.
4. The held-out result preserves acceptable answerable-case coverage and its uncertainty supports the stated conclusion.
5. The official benchmark's own scoring and latency definitions are preserved.

If the support gate fails to distinguish the minimal pairs, report the failure mode and keep semantic sufficiency unqualified. Negative results are valuable: they show which guarantees the current stack actually provides.

## 9. Reference points and limits

- QEMem, *From Relevant Memories to Sufficient Evidence*: https://arxiv.org/abs/2610.22668
- LongMemEval-V2 official backend contract, evaluation harness and baselines: https://github.com/xiaowu0162/LongMemEval-V2
- MemoryAgentBench: https://arxiv.org/abs/2507.05257
- LongMINT: https://arxiv.org/abs/2605.18565

These works motivate the protocol but use different datasets, memory interfaces, readers, evaluators and budgets. Their published accuracies are not target values for this component and must not be placed on the same chart as Symthaea results without a matched protocol.

## 10. Current claim boundary

This document provides a versioned evaluation contract. It does not implement semantic entailment, contradiction detection, role extraction, query planning, answer calibration, official worker wiring, or a benchmark result. PR #8876's role coverage and PR #8894's payload-span verification/source-digest association remain useful component-level guarantees; neither is a semantic-sufficiency certificate, and the latter does not yet prove raw-source byte-range mapping. No performance or SOTA claim follows until the exact integrated commit is compiled, tested, and evaluated on the frozen held-out protocol.
