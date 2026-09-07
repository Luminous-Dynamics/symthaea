# Scientific Disposition Assessment v1 — evidence-view closure hardening

**Status:** semantic hardening companion to `SCIENTIFIC_DISPOSITION_ASSESSMENT_V1.md`.

This note closes an omission/cherry-picking loophole in the initial disposition contract before any implementation exists.

## 1. Core correction

A disposition cannot claim a complete reason topology merely because it lists every evidence object the caller supplied.

The stronger theorem is:

```text
caller-supplied evidence list
    != closed evidence view

closed evidence view
    != all evidence in existence

complete reason topology
    = complete relative to one exact declared evidence-view snapshot
```

Without a closed evidence-view boundary, a caller could omit inconvenient evidence before disposition derivation and still receive an internally consistent reason trace.

That would make the trace complete only relative to a silently cherry-picked input set.

## 2. Evidence-view snapshot

A future disposition assessment should therefore bind an immutable `ScientificEvidenceViewSnapshot` or equivalent object describing the exact candidate evidence universe considered for one proposition/use/cutoff.

Conceptually:

```text
ScientificEvidenceViewSnapshotV1 {
    view_id,
    proposition_id,
    requested_scientific_use,
    information_cutoff,
    registry_generation_id,
    discovery/query_profile_id,
    inclusion_scope,
    candidate_contribution_ids,
    unresolved_discovery_refs,
}
```

The exact Rust shape is not frozen here.

The disposition assessment then operates on:

```text
closed evidence-view snapshot
        -> admission / exclusion receipts
        -> argument + dependency + triangulation projection
        -> primary disposition + reason topology
```

rather than:

```text
Vec<EvidenceContribution>
        -> disposition
```

## 3. Completeness is scoped, never absolute

The shared kernel should not claim:

```text
this snapshot contains every relevant scientific fact in the world
```

Instead it should be explicit:

```text
complete relative to:
    exact proposition
    exact scientific use
    exact information cutoff
    exact registry/source generations
    exact discovery/query profile
    exact declared source scope
```

An assessment may therefore say, in effect:

> complete over registry R generation G, source collections S, query profile Q, and cutoff T.

It must not silently promote that to universal literature completeness.

## 4. Query/discovery policy is scientific lineage

The method used to construct the candidate evidence view can itself influence the result.

Examples include:

```text
registry traversal
systematic-search query
connected-database scope
citation expansion policy
language restriction
publication-type restriction
domain inclusion/exclusion rules
```

Therefore the discovery/query profile needs exact identity and belongs in scientific lineage.

Changing the query profile creates a new evidence-view snapshot even when the final admitted evidence happens to be identical.

## 5. Omission becomes detectable

The disposition assessment should require every contribution named by the exact evidence-view snapshot to end in one of a small set of auditable states such as:

```text
Admitted
Excluded { exact reason }
Unresolved { exact reason }
DuplicateAliasOf { canonical contribution }
OutOfScopeByQualifiedRule { receipt }
```

No candidate contribution may simply disappear between evidence-view construction and disposition.

Thus:

```text
candidate evidence count
    = admitted
    + excluded
    + unresolved
    + explicitly canonicalized aliases
```

under the exact snapshot.

The count is an accounting invariant only; it grants no evidentiary weight.

## 6. Discovery failure is itself visible

If a connected registry/source cannot be queried, a source collection is unavailable, an index is stale, or a query cannot be reproduced, the view should record that limitation rather than silently becoming a smaller “complete” evidence set.

Possible states include:

```text
SourceUnavailable
RegistryGenerationUnavailable
QueryExecutionUnverified
IndexCoverageUnknown
HistoricalSnapshotUnavailable
```

These may force an `IncompleteEvidenceView` or other bounded disposition depending on policy.

They must not default to absence of contrary evidence.

## 7. Historical views require historical discovery semantics

For `as_of=t0`, it is not enough to filter today's contribution database by contribution date.

The historical view should bind the source/registry/index/query state that was admissibly available under the selected historical reconstruction policy.

Otherwise a later-discovered old paper could leak backward into the reconstructed information state simply because its publication date predates `t0`.

Therefore:

```text
published_before(t0)
    != known/discoverable_by_the_assessment_at(t0)
```

unless the historical-view policy explicitly reconstructs a broader counterfactual literature set and labels it as such.

## 8. Prospective vs retrospective evidence-view authority

A live prospective scientific process may have stronger evidence-custody semantics than a later retrospective reconstruction.

The kernel should preserve whether an evidence view is:

```text
LiveProspectiveView
HistoricalReconstructionView
CounterfactualCompleteRegistryView
```

or another domain-defined class.

These classes are not rank-ordered universally.

A retrospective reconstruction cannot silently inherit the stronger claim that the live process actually possessed every included artifact before the decision/forecast cutoff.

## 9. The disposition binds the view identity

`ScientificDispositionAssessmentV1` should therefore bind:

```text
evidence_view_snapshot_id
```

alongside lifecycle/adjudication/dependency/triangulation/policy generations.

Changing the evidence-view snapshot makes the predecessor disposition stale for the new view even when the primary disposition label stays the same.

## 10. Reason-topology completeness is now precise

The phrase `complete reason topology` should mean:

```text
all candidate contributions in the bound evidence view
+ exact admission/exclusion/unresolved handling for each
+ every qualified argument relation considered by the policy
+ every bound defeater/falsifier/dependency/compatibility/triangulation object
+ every unresolved condition required by the policy
+ exact policy and generation identities
```

It does **not** mean omniscient knowledge of all possible evidence.

This scoped definition is auditable and falsifiable.

## 11. Policy cannot secretly narrow the evidence universe after reveal

The evidence-view construction profile and the disposition policy are separate identities.

A future workflow should establish the candidate evidence universe before disposition rules selectively admit/exclude contributions.

Conceptually:

```text
EvidenceViewSnapshot
    -> complete candidate set
    -> DispositionPolicy admission/exclusion
    -> assessment
```

not:

```text
DispositionPolicy searches until it finds enough supporting evidence
```

A domain may define adaptive search procedures, but they require explicit preregistration/stopping rules and their own execution lineage.

## 12. Negative-search evidence remains bounded

A result such as:

```text
no opposing contribution found
```

is meaningful only relative to the exact evidence-view/query scope.

It must be represented as:

```text
no opposing contribution found within view V
```

not:

```text
no opposing evidence exists
```

## 13. Duplicate and derivative publications

The evidence view should preserve discovered contributions before dependency-aware interpretation.

Multiple publications may later resolve to:

```text
same evidence lineage
same underlying dataset
same analysis artifact
corrected/superseded versions
```

but discovery should not silently deduplicate them by title/author similarity.

Canonical aliasing or lineage grouping requires explicit identity/dependency receipts.

This keeps search completeness separate from replication independence.

## 14. Suggested additional disposition fields

Conceptually extend the candidate assessment with:

```text
evidence_view_snapshot_id,
evidence_view_profile_id,
evidence_registry_generation_id,
view_completeness_disposition,
view_limitations,
```

and require the reason-topology commitment to cover them.

## 15. Qualification adversarial cases

A future implementation should prove at least:

1. omitting one contribution from a bound closed view causes assessment failure rather than silent omission;
2. every candidate contribution receives an admitted/excluded/unresolved/alias disposition;
3. changing the query/discovery profile changes the view identity;
4. a source outage is retained as a view limitation rather than interpreted as no evidence;
5. `no opposition found` is scoped to the exact view;
6. a later-discovered old artifact cannot leak into an earlier live prospective view;
7. historical reconstruction and live prospective custody remain distinguishable;
8. duplicate papers cannot silently inflate replication counts;
9. disposition-policy logic cannot mutate the already-bound candidate universe without producing a new view;
10. reason-topology completeness is asserted only relative to exact view identity, never globally.

## 16. Important non-claims

This note does not define a universal literature-search algorithm, guarantee access to every scientific database, define relevance ranking, solve publication bias, establish exhaustive discovery, or make one registry authoritative for science.

It only ensures that a future disposition cannot call its reason topology complete while allowing an upstream caller to silently choose which evidence exists for the assessment.
