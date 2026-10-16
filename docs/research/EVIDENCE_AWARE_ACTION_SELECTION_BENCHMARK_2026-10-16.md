# Evidence-Aware Action Selection: Research and Benchmark Contract

**Date:** 2026-10-16  
**Status:** Research protocol and deterministic policy prototype; evaluation targets are proposed, not measured.  
**Stack:** Intended to follow PR #7610, alternative and joint evidence proof formulas.

## 1. The paradigm shift

Evidence systems are often evaluated as retriever + answer generator, with a binary choice between answering and abstaining. That misses an important control problem: a system can often do something useful between those extremes. It can ask a clarifying question, acquire a missing source, resolve a provenance gap, wait for independent corroboration, or return only a narrower conclusion that the evidence supports.

The proposed control loop is:

1. **Diagnose** evidence sufficiency, contradiction, question clarity, provenance validity, and lineage-deduplicated origin count.
2. **Choose an action**: answer, answer with an explicit scope limit, clarify, acquire more evidence, or abstain.
3. **Re-evaluate** the evidence after clarification/acquisition; never assume that an acquisition attempt succeeded.
4. **Record** the input assessment, policy version, action, reason code, and later outcome in a reproducible receipt.
5. **Evaluate risk and utility** on held-out claim/source families, including adversarial mutations.

This is not a confidence score. The output is a deterministic policy decision from upstream assessments. Provenance integrity is not truth, and multiple URLs are not proof of independent sources.

## 2. Why this is worth testing

The 2026 *Evidence Sufficiency Benchmark* paper evaluates a five-level continuum from full support through partial support, irrelevant context, no context, and conflicting evidence. Its reported tested models answered too often under conflicting evidence: the reported over-answer rate range was 65–91% in that experiment. This is evidence of a benchmarked failure mode, not a universal rate for all models or domains.

Selective prediction and conformal abstention research also makes clear that reduced risk and useful coverage trade off against each other. Conformal methods need held-out calibration assumptions and finite-sample analysis; they do not turn an arbitrary heuristic threshold into a guarantee. For this policy kernel, the first milestone is deterministic action correctness and a sound dataset protocol. Any future probabilistic risk claims need their own calibration/evaluation design.

### Research references

- Hantian Zhang and Wentai Wu (2026), [Do LLMs Know When Evidence is Insufficient? An Evidence Sufficiency Benchmark for Answer-Abstention Calibration in Retrieval-Augmented Generation](https://doi.org/10.32604/cmc.2026.086343). The paper uses five evidence levels and reports high over-answering under conflicting evidence in its evaluated settings.
- Sina Tayebati et al. (2025), [CAP: Conformalized Abstention Policies for Context-Adaptive Risk Management for LLMs and VLMs](https://proceedings.mlr.press/v304/tayebati26a.html). Relevant to explicitly evaluating policy-calibrated coverage and risk/coverage trade-offs.
- Varun Kotte (2026), [When Can Conformal Risk Control Certify LLM Outputs? Bounds, Impossibility, and Adaptation for Structured Generation](https://arxiv.org/abs/2606.29054). Relevant to feasibility checks, finite-sample bounds, and behavior under distribution shift.
- W3C, [PROV-O: The PROV Ontology](https://www.w3.org/TR/prov-o/). A vocabulary for entities and derivations; using a provenance representation does not itself authenticate the entities or certify truth.

## 3. Prototype contract

The Rust module \`src/knowledge/evidence_action_policy.rs\` consumes explicit upstream assessments:

- Evidence condition: full support, partial support, irrelevant, missing, or conflicting.
- Provenance condition: verified, incomplete, or invalidated.
- Question condition: clear or ambiguous.
- Lineage result: a known count of distinct credited primary-origin groups, or unresolved.
- Whether a safely scoped partial conclusion exists.
- Whether acquisition is currently permitted by budget/policy.

It returns one of five actions:

| Action | Meaning |
|---|---|
| \`Answer\` | Return the supported claim at the requested scope. |
| \`AnswerWithScopeLimit\` | Return only an explicitly supported narrower conclusion. |
| \`AskClarification\` | Resolve question ambiguity before further retrieval. |
| \`AcquireMoreEvidence\` | Seek missing lineage, sufficient support, or conflict resolution. |
| \`Abstain\` | Do not return a substantive answer when policy cannot proceed. |

Hard invalidation, unresolved conflict, and incomplete provenance cannot produce a substantive answer. Origin counts must be lineage-deduplicated; the policy must not receive raw citation counts as a substitute. Exhausted acquisition budget must lead to abstention, not a weaker proof silently promoted to a complete one.

### Scope limits

This is a deterministic policy kernel—not a retrieval engine, truth detector, probabilistic calibration method, source-authentication mechanism, or general-purpose decision-maker. It trusts that callers correctly map raw evidence into the enumerated states. The current prototype is not wired into the live cognitive loop or production retrieval pipeline.

## 4. Dataset construction and leakage controls

Build a versioned, immutable corpus of claim families. Every row must have a stable case ID, claim family ID, origin-family split key, expected action, evidence condition, source lifecycle, provenance completeness, conflict label, question clarity, available independent-origin groups, acquisition-budget state, and an adjudication note. Keep sensitive or licensed source text out of the public repo; public fixtures can use synthetic or redistributable evidence.

### Core strata

- **S1 — Full support:** exact claim supported with a provenance-complete source set.
- **S2 — Partial support:** only a restricted, explicit conclusion is supportable.
- **S3 — Irrelevant:** high lexical similarity may be present, but evidence does not support the claim.
- **S4 — Missing:** retrieval returns no usable evidence.
- **S5 — Conflict:** two or more statements cannot all support the requested conclusion without temporal, scope, or contextual resolution.

Add cross-cutting perturbations rather than making them disjoint strata:

- Source mirrored at multiple URLs; reposts and quotations that share one primary origin.
- Same source revision repeated with changed fetch time or locator.
- Stale and superseded revision, withdrawn source, explicitly revoked source, digest mismatch, missing derivation edge, cycle, dangling parent, or unknown root origin.
- Evidence that is accurate but answers a neighboring question; under-specified question; a valid narrowed answer embedded in otherwise insufficient evidence.
- Source diversity near the policy boundary (0, 1, 2, and 3 independently attributed origins).
- Acquisition permitted vs. exhausted; successful acquisition vs. failed or contradictory acquisition.
- Domain shift: unseen entities, source families, claim templates, publishers, time periods, and at least one held-out domain.

**Split before generating variants.** Keep all derived documents, mirrors, revisions, paraphrases, claims sharing source families, and counterfactual variants on one side of the train/development/test boundary. Split by origin/claim family—not by individual row. Any synthetic-to-real transfer result must report the domains and source families that remain distinct.

Use at least 1,000 labeled cases in each of S1–S5 for a useful initial study, with at least 300 genuinely independent origin/claim clusters represented in the critical-risk test slice. This is a proposed minimum, not a guarantee of statistical power. Report the effective cluster count and use a source-clustered interval or bootstrap; do not treat many mirrors of one source as independent observations.

## 5. Baselines and ablations

Compare the policy with:

1. **Answer always** — negative control, expected to fail on missing/irrelevant/conflicting evidence.
2. **Binary answer/abstain** — a simple conservative baseline.
3. **Raw-source-count threshold** — intentionally naive control showing the effect of mirrors/reposts.
4. **Lineage-aware threshold** — counts credited origin groups but has no clarification or acquisition action.
5. **Action policy without provenance state** — ablation measuring the value of provenance gating.
6. **Action policy without conflict handling** — negative control; must visibly fail conflict cases.
7. **Full action policy** — five-action route with reason codes.
8. **Policy budget ablation** — compare acquisition permitted/exhausted without counting an attempted acquisition as a successful repair.

For learned downstream action selectors, include a non-learned rule baseline, freeze thresholds before the held-out test, and report tuning-data usage. Do not train and test on alternative phrasings or derived documents from the same source family.

## 6. Metrics and preregistered targets

Separate structural invariants, behavioral quality, statistical risk, and systems cost.

### Structural invariants (hard gates)

- 100% of deterministic policy truth-table tests pass.
- **Zero substantive answers** when provenance is invalid or required provenance is incomplete.
- **Zero automatic winner selection** while contradictory evidence remains unresolved.
- **Zero answer eligibility from unknown origin lineage** or raw URL count alone.
- **Zero partial proof/policy outputs presented as complete** after a budget failure.
- Same input + same policy version produces exactly the same action and reason.
- Every receipt records exact input digest, policy version, action, and reason; altered input must invalidate a digest-bound receipt.
- No post-budget acquisition is recorded as successful evidence.

### Behavior and utility (report per stratum and overall)

- Unsupported-substantive-answer rate: unsupported answers / all substantive answers.
- False-abstention rate on cases whose evidence and scope are sufficient.
- Action exact-match and macro-F1 across all five actions (diagnostic only; never substitute for safety gates).
- Supported-answer coverage at fixed unsupported-answer risk.
- Scope precision for qualified answers: fraction of statements that remain within adjudicated support.
- Conflict escape rate and invalid-provenance escape rate.
- Origin inflation: credited origins minus gold distinct primary origins; false merge/split rates for lineage grouping.
- Clarification efficiency: fraction of clarification actions that change an ambiguous request into a correctly answerable or correctly abstained case.
- Acquisition yield: fraction of acquisition requests that obtain evidence resolving the targeted deficiency; report outcomes and cost, including failed fetches.
- Regret versus an oracle action under a preregistered utility table; publish the utility table before test evaluation.

### Initial numeric targets (proposed, not observed)

| Metric | Initial target |
|---|---:|
| Invalid-provenance / unresolved-conflict answer escapes | 0 |
| Raw-citation inflation on mirror/repost tests | 0 |
| Exact deterministic policy cases | 100% |
| Full-support answer coverage with valid provenance and policy-sufficient origins | ≥ 95% |
| Scoped-answer scope precision on adjudicated partial cases | ≥ 95% |
| Unsupported-answer rate on critical-risk test slice | ≤ 1%, with one-sided 95% upper confidence bound reported |
| False-abstention rate on clean, fully supported cases | ≤ 10% |
| Unresolved acquisition reported as repaired | 0 |

These thresholds are engineering hypotheses. Do not claim a 1% risk bound from a small or non-independent synthetic fixture. If the upper confidence bound misses the target, the gate has not passed even if the point estimate looks good. Report uncertainty and cluster structure.

### Cost and scale

At 1k, 10k, and 100k claims, report p50/p95 action latency, memory, allocation count if available, evidence acquisition attempts, successful repairs, and human review burden. The policy kernel itself should be constant-time in the number of retrieved documents once its input assessments are prepared; the surrounding lineage/provenance graph and acquisition system need separate scaling measurements. Never combine the cost of failed acquisition with successful repairs only.

## 7. Counterfactual tests that expose shallow policies

Each case family should include near-identical variants where only one meaningful condition changes. Examples:

- Keep claim and documents fixed but change source status from active to revoked: the answer must no longer be permitted.
- Replace five repost URLs with one origin family: credited independent-origin count must fall.
- Preserve the same content but make its provenance unresolved: the output must move to acquisition or abstention.
- Add unresolved contradiction to an otherwise answerable claim: the policy must not answer.
- Replace a full-support passage with a passage that supports only a narrower scope: full answer becomes scoped answer, acquisition, or abstention—not the same unqualified answer.
- Mark the question ambiguous while keeping support intact: request clarification.
- Exhaust the acquisition budget: acquisition decisions must safely fall back to abstention.
- Add a new independent source that actually resolves a conflict: only after upstream reevaluation proves the conflict resolved may the policy return to an answer action.

The paired case is the experimental unit. Report consistency over the pair and the exact expected transition, not only aggregate accuracy.

## 8. Implementation sequence

1. **Current PR:** add and unit-test the standalone deterministic action-policy kernel.
2. **Next:** create an immutable JSONL case format and an independent receipt checker; reject unknown fields, duplicate IDs, missing labels, family leakage, and altered case digests.
3. Generate synthetic counterfactual families and verify the negative controls before collecting human labels.
4. Add expert-adjudicated, source-family-disjoint cases and blind disagreement review.
5. Join policy input assessments to the already separate provenance, lineage, derivation graph, and proof-formula results; retain component outputs as evidence, not as truth.
6. Evaluate baselines, paired perturbations, calibration if a probabilistic selector is later introduced, and clustered uncertainty.
7. Wire into the live answer path only after the end-to-end test shows that it receives trustworthy upstream assessments and that action receipts are auditable.

## 9. Acceptance rule

A passing structural suite establishes only that the code follows its declared action contract on those tests. A passing synthetic benchmark establishes only behavior on that fixture. A source-disjoint, adjudicated test suite is needed for meaningful external validity. Production integration additionally needs end-to-end evidence of provenance inputs, budget enforcement, repair results, action receipts, latency, and policy-version binding.

Queued CI is pending, not passing. No Rust compile, unit test, or benchmark result is claimed by this research document.
