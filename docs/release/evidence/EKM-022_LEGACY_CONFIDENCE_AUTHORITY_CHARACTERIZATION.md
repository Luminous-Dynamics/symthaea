# EKM-022 — Legacy Confidence Authority Characterization

Status: measurement-only characterization. No legacy confidence behavior is disabled or replaced by this PR.

## Why this exists

The new EKM epistemic pipeline separates claims, evidence, provenance, admission, mutation, and later belief revision. The older `EnhancedKnowledgeGraph` predates that separation and already contains several operations that alter fact confidence directly.

Before any EKM belief-update policy is wired into the live graph, those existing authority channels need to be characterized and then migrated deliberately.

## Characterized channels

The deterministic local characterization records whether the current graph exhibits:

1. **Similarity corroboration boost** — an HDC-near-duplicate can reuse an existing fact and increase its confidence.
2. **Retrieval recency authority** — ordinary search updates `last_accessed_cycle`, which changes subsequent decay behavior.
3. **Contradiction contraction** — `resolve_contradictions` directly reduces the currently weaker fact's confidence.
4. **Direct adjustment** — `adjust_confidence` accepts a caller-provided confidence delta without an EKM evidence object.
5. **Dream/topic strengthening** — `strengthen_facts_by_topic` can increase confidence based on replay/topic matching after confidence has fallen below its initial value.
6. **Causal strengthening** — `strengthen_causal_facts` can increase confidence for facts already marked causal after confidence has fallen below its initial value.

## Epistemic interpretation boundary

This characterization does **not** claim those mechanisms are inherently wrong. Some may remain useful as memory salience, retention, accessibility, or consolidation signals.

The problem is semantic overloading: the same `confidence` field currently carries effects from evidence-like corroboration, retrieval recency, contradiction handling, dream replay, and generic consolidation. Those are not epistemically equivalent.

In particular:

- retrieval should not become evidence merely because it happened again;
- replay/dream consolidation may strengthen accessibility or memory retention, but should not create external validation;
- HDC similarity should not establish independent corroboration;
- a causal label should not by itself add causal evidence;
- contradiction resolution should preserve the contradictory evidence history even if a downstream belief estimate changes.

## Intended migration direction

Do not delete these mechanisms blindly. Split their authority:

- **memory accessibility / retention** may remain influenced by retrieval and replay;
- **epistemic belief strength** should be derived through evidence/provenance/calibration policy;
- **causal qualification** should continue through the semantic causal admission gate;
- **contradictions** should remain represented even when belief posture changes.

A later PR should introduce separate `salience/retention` and `epistemic support` concepts before changing the live graph's behavior.

## Qualification rule

Do not call this characterization qualified until the exact PR head executes repository CI. Expected numeric fixture values are regression descriptors, not normative thresholds or proof that the legacy behavior should be preserved.
