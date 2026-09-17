# EKM-023 — Separate Epistemic Support from Memory Dynamics

Status: proposal-routing schema only. No live confidence field is migrated or mutated by this PR.

## Problem

EKM-022 characterizes multiple pre-existing mechanisms that currently write into one `TemporalFact::confidence` scalar: similarity corroboration, retrieval recency, contradiction handling, direct adjustment, dream/topic replay, and causal consolidation.

Those mechanisms have different meanings. Repeating or replaying a representation may make it easier to retrieve or harder to forget without providing new external evidence that the represented claim is more likely true.

## Proposed dimensions

EKM-023 introduces four independent knowledge-weight dimensions:

- **EpistemicSupport** — evidence-grounded support for a proposition.
- **Accessibility** — retrieval priority/ease for current cognition.
- **Retention** — resistance to eviction or forgetting.
- **Consolidation** — strength of internally rehearsed representation.

An unrepresented dimension is `None`, not zero.

## Conservative authority routing

The proposal-only authority router permits:

- admitted evidence → `EpistemicSupport`
- retrieval → `Accessibility`
- similarity match → `Accessibility`
- dream replay → `Retention` or `Consolidation`
- causal consolidation → `Retention` or `Consolidation`
- memory decay → `Retention`
- task relevance → `Accessibility`

Retrieval, HDC similarity, replay, generic consolidation, and task relevance cannot propose direct epistemic-support changes.

Even admitted evidence may only *propose* an epistemic-support update and must name at least one evidence ID. This PR contains no apply/update function.

## Important non-claim

This routing table is an architectural safety policy, not a scientific assertion that memory accessibility and epistemic belief are completely independent in humans or biological cognition. It exists to prevent Symthaea from silently treating internal repetition as external validation.

## Next boundary

A later belief-revision gate should inspect the actual evidence records named by an epistemic proposal, their polarity, provenance ancestry, causal evidence kind, contradiction state, uncertainty assessment, calibration state, and a bounded update policy before any live epistemic-support mutation is possible.

## Qualification

Do not call EKM-023 qualified until its exact PR head executes repository CI. Authoring, static review, queued jobs, and mergeability are not qualification evidence.
