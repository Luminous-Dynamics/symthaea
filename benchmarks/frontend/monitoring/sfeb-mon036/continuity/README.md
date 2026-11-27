# MON-036 — Recovery Evidence Continuity

MON-036 prevents a still-unexpired recovery claim from being reused after a dependency, verifier or authority changes. It preserves unaffected claims rather than forcing every claim to be re-run.

Run:

\`\`\`sh
cd benchmarks/frontend/monitoring/sfeb-mon036/continuity
go run ./cmd/qualify --input INPUT-example.json
bash CHECKS.sh
\`\`\`

The example includes two claims and one connector configuration change. The CDC claim should be \`revalidate\`; the unrelated backup repository claim should be \`retain\`.

- \`retain\`: no supporting dependency changed and all evidence remains current.
- \`revalidate\`: a supporting dependency/verifier changed or evidence expired; obtain fresh evidence before reuse.
- \`invalidate\`: authority revoked, scope changed, contradiction or corrupt event/claim identity.
- \`unresolved\`: incomplete history, unknown impact, missing state or untracked drift.

The event hash chain checks integrity, not authenticity. Use independently authenticated change collection and MON-026 custody before composing results into MON-024. No live services are queried.
