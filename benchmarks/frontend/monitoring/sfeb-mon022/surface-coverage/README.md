# MON-022 surface-coverage semantic reference

A standard-library Go reference that evaluates a declared database surface inventory against normalized per-surface attestations.

It has two claim scopes:
- row_stream_continuity: only row-stream-required surfaces are claimed; every other inventory surface must have explicit excluded-by-contract evidence, and exclusions are returned in the result.
- full_recoverable_database_state: every inventory surface must be required and covered or have a verified not-applicable proof; excluded or missing surfaces cannot qualify.

It checks source/target system IDs, administrative control roots, scope digest, freshness, completeness and evidence/method digest. Independent comparison and recomputation require an observer with a control root distinct from both endpoints. Source/target digest mismatch is divergent. Duplicate surface/evidence identities are divergent.

It consumes normalized attestations. It does not query PostgreSQL, compute digests, verify signatures, establish control-root membership or run restore tests. Those are upstream adapter/verifier responsibilities.

Run: go test -race -count=5 ./..., go vet ./..., go test -count=1 -cover ./..., go test -run=^$ -fuzz=FuzzSurfaceScopeIdentity -fuzztime=2s, go test -run=^$ -bench=BenchmarkEvaluateCoverage -benchtime=3x.


Each attestation also binds the exact inventory digest, evidence subject ID and pinned mon022-surface-coverage-attestation-v1 profile. Independent and recomputed comparisons require a separately verified observer identity and control-root membership; the observer must not be an alias of either endpoint. The current package consumes these as normalized verifier outputs and does not authenticate them itself.

No exact-head tests, fuzz, vet or numeric benchmark have been run for the current branch.
