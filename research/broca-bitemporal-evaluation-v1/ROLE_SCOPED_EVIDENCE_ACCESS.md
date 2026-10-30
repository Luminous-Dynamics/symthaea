# Broca Role-Scoped Evidence Views

**Status:** reference adapter and unit tests authored; not compiled or executed. This is scope-level read authorization, not a complete multi-agent identity system.

## 1. Gap

The existing `EvidenceIndex` filters by namespace/environment `scope_id`, lifecycle state, and validity window, and verifies source/content digests before returning payload. However, the low-level API accepts `scope_id` from its caller. That is a filtering parameter, not a complete authorization decision.

Multi-agent memory requires a trusted mapping from an authenticated requesting role to the scopes it may read. A receiving agent must not gain access merely by asking the resolver for another scope name.

## 2. New adapter

The new types and methods are:

- `RoleScopePolicy`: immutable policy version and mapping from role ID to readable scope IDs.
- `AuthorizedEvidenceContext`: requesting role, access-policy version, original `EvidenceRecord`, and materialized `MemoryContextItem`.
- `EvidenceIndex::search_bm25_for_role`: checks the role grant before lexical retrieval and again during context resolution.
- `EvidenceIndex::resolve_for_role`: resolves precomputed HDC IDs only after checking the same role/scope grant.

The adapter retains the existing checks:
- scope equality;
- Active lifecycle only;
- half-open valid-time eligibility;
- structural record validation;
- text/image content digest checks;
- safe image path canonicalization;
- duplicate and stale-hit filtering.

The full evidence record is returned alongside the context so a caller can retain source trajectory/step IDs, source SHA-256, source pointer, evidence identity, content digest, lifecycle, temporal bounds and scope. Only the context payload should be passed to the language model; audit metadata remains outside its prompt.

## 3. Security model and non-claims

The role identifier must come from authenticated execution context, not from natural-language user input or an agent-selected string. The policy must be loaded from trusted configuration or a verified governance policy. `RoleScopePolicy::new` validates structure but does not authenticate the source of policy bytes or issue credentials.

The adapter currently provides **scope-level** authorization:
- a role can read a listed scope;
- a role cannot request an unlisted scope through the authorized methods.

It does not yet implement:
- per-record or per-attribute ACLs within one scope;
- write permissions or delegation;
- authenticated role credentials, signed access-policy chains or policy-ingress receipts;
- policy revocation/rollback protection;
- role membership in Mycelix or a production worker/session identity bridge;
- atomic coupling of access policy changes with vector-index updates.

The existing low-level `search_bm25` and `resolve` methods remain for backwards compatibility. They must not be exposed as agent-callable tools in a multi-agent deployment that relies on role authorization. The worker/API boundary must call the authorized adapter using a trusted principal.

This is not a substitute for source authenticity: an authorized role may still encounter false or low-authority evidence. Source admission and provenance remain governed separately by the bitemporal ledger and attestation modules.

## 4. COLLABMEM alignment

COLLABMEM diagnoses collaborative memory operations including sharing, access visibility, routing, provenance, conflict resolution and retention. The new adapter directly targets the access-visibility boundary; evidence metadata supports provenance handoff. It does not yet solve routing, write-side collaboration, conflict arbitration, or retention-policy distribution.

Use these diagnostic controls:
1. Authorized role reads its granted scope.
2. Same role requests a different scope and receives an authorization error before search results or context are returned.
3. Unknown role cannot enumerate scopes or evidence IDs through the authorized methods.
4. A stale HDC hit cannot bypass role authorization or lifecycle filtering.
5. Expired, revoked, superseded or deleted records stay unavailable even for an authorized role.
6. Returned evidence preserves original source provenance and access-policy version.

For external evaluation, use the official COLLABMEM harness and separate scope/access errors from retrieval, provenance, conflict and answer errors. Do not claim a COLLABMEM score from this unit suite.

Reference:
- Wu et al., *COLLABMEM: A Diagnostic Benchmark for Collaborative Memory Operations in Multi-Agent LLM Systems*, EMNLP 2026: https://aclanthology.org/2026.emnlp-main.1194/

## 5. Promotion gates

Internal gates for the next worker integration:
- zero cross-scope reads through the authorized interface;
- zero payload/context exposure before role grant validation;
- 100% provenance metadata preservation on successful reads;
- zero revival of revoked/deleted/superseded/expired evidence;
- policy-version changes invalidate stale authorization caches;
- role identity and policy source are authenticated outside the language model;
- official COLLABMEM access-visibility probes pass under the same harness.

These are proposed gates, not measured results. The low-level API compatibility path means the first four gates apply only after the production worker is wired to the authorized entry point and bypasses are prevented.

## 6. Next steps

1. Compile and run the duplicate-definition cleanup and role-scope tests on the exact PR head.
2. Add authenticated principal-to-role binding.
3. Bridge role grants to the Mycelix policy/provenance fabric.
4. Add policy revision, expiry, rollback and emergency revocation semantics.
5. Add per-record visibility only if scope-level isolation proves insufficient for the official COLLABMEM tasks.
6. Run official collaboration probes and capture exact retrieval/context traces.
