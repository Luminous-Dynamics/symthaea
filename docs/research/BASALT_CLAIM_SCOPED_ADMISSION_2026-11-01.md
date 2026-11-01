# BASALT: Claim-Scoped Admission Calculus and Evidence Capsules

**Research specification:** v0.1  
**Date:** 2026-11-01  
**Status:** draft; protocol and benchmark design only  
**Tracking:** [BASALT-ADMISSION-CALCULUS-001](https://github.com/Luminous-Dynamics/symthaea/issues/9722)

## 1. Purpose and claim boundary

BASALT is converging on several independently meaningful evidence layers:

- receipt authenticity and transparency (#9560);
- crash-safe state transitions (#9612);
- remote authoritative state and rollback recovery (#9702);
- local SQLite durability (#9703);
- dependency-aware freshness and selective re-verification (#9203);
- observer delivery and qualification workstreams.

This document defines how a consumer composes those results into a decision for one exact claim. It does not replace any of those systems and does not implement their cryptographic or storage backends.

The target is not a generic trusted boolean. It is a reproducible answer to:

> Is this exact evidence sufficient, current, and applicable for this exact subject, environment, policy, and intended use?

A capsule is a portable collection of evidence and verification context. It is not itself an authority. A cached result is never self-authenticating.

## 2. Typed three-valued admission

Let a claim be:

\[
C = (id, profile, S, U, X)
\]

where:

- \(id\) is the stable claim identifier;
- \(profile\) is the versioned admission profile;
- \(S\) is the exact subject digest set;
- \(U\) is the intended-use identifier;
- \(X\) is the qualification/context envelope.

Every mandatory obligation \(o_i\) has a result in:

\[
R = \{\mathsf{valid},\mathsf{invalid},\mathsf{unknown}\}
\]

Interpretation:

- **valid:** the obligation was affirmatively verified for the exact bound inputs;
- **invalid:** the evidence is disproven, violates a mandatory policy, or is out of scope for the requested use;
- **unknown:** the obligation is missing, stale, unavailable, unsupported, ambiguous, or cannot presently be verified.

The admission result is the conjunction of all mandatory obligations using strong Kleene three-valued logic.

| AND | valid | unknown | invalid |
|---|---|---|---|
| **valid** | valid | unknown | invalid |
| **unknown** | unknown | unknown | invalid |
| **invalid** | invalid | invalid | invalid |

Equivalently:

\[
A(C) =
\begin{cases}
\mathsf{invalid} & \exists i: R_i=\mathsf{invalid}\\
\mathsf{unknown} & (\nexists i: R_i=\mathsf{invalid}) \land (\exists i: R_i=\mathsf{unknown})\\
\mathsf{valid} & \forall i: R_i=\mathsf{valid}
\end{cases}
\]

**Operational admission is permitted only when the result is valid.** Invalid is a known rejection. Unknown is a fail-closed deferral, not a claim that the evidence is false.

No weighted score, majority of unrelated checks, confidence value, or subsystem success can compensate for an unknown or invalid mandatory obligation. The full vector of obligation outcomes and reason codes must accompany the aggregate result.

An obligation may be excluded only when the versioned claim profile explicitly declares it non-mandatory and provides the policy reason. A verifier must not silently convert a missing obligation into not-applicable.

### 2.1 Minimum obligation vocabulary

A claim profile may require any applicable subset of these separately typed dimensions:

1. subject and exact payload-byte binding;
2. DSSE envelope/signature validity;
3. signer authorization, key validity, and revocation;
4. transparency-log origin and inclusion;
5. signed checkpoint and consistency from the previously trusted checkpoint;
6. witness policy, independent-domain quorum, and split-view status;
7. source generation and predecessor-seal continuity;
8. local transaction durability state;
9. remote authority freshness and anti-rollback state;
10. observer binary/configuration and capture qualification;
11. workload-class and environment applicability;
12. dependency graph currentness and invalidation state;
13. authorization for the exact intended use.

These are not interchangeable. A signature does not establish observation completeness. A log inclusion proof does not establish that the observation is true. A local commit does not establish external rollback resistance. Observer qualification does not authorize arbitrary downstream use.

## 3. Evidence capsule model

The interoperable outer structure should reuse the in-toto Statement and DSSE envelope rather than invent a signature format. BASALT should define a versioned predicate type for runtime/source-epoch evidence and claim admission. This is not SLSA build provenance.

The Statement subject binds the exact immutable artifact or evidence subject by digest. The predicate binds BASALT-specific runtime semantics and admission context. DSSE signs the exact payload bytes and payload type using its specified pre-authentication encoding. Do not parse and reserialize a JSON object before verifying a signature over the original bytes.

A complete capsule contains or references, by content digest:

### 3.1 Claim identity and scope

- capsule schema and profile version;
- claim ID and claim-type URI;
- exact subject digest set;
- exact intended-use identifier;
- qualification envelope and applicability constraints;
- policy bundle digest and profile version.

### 3.2 Execution and observation context

Where relevant:

- provisioned host/source/partition lineage identifiers;
- boot-incarnation identity;
- OS distribution and version;
- kernel release/build/configuration identity;
- observer executable digest and configuration digest;
- capture backend and relevant driver/runtime versions;
- workload class and qualification corpus/version;
- capture-loss, overflow, restart, and closure evidence references.

A change to a bound field is not silently compatible. The profile defines whether the change invalidates a specific obligation, narrows applicability, or makes the result unknown pending requalification.

### 3.3 Receipt, transparency, and authority evidence

- exact DSSE envelope digest and payload type;
- exact accepted receipt identity and source generation;
- log origin, tree size, root digest, checkpoint digest, and entry identity;
- inclusion proof and consistency proof references (or portable proof bytes);
- witness identities, administrative/failure domains, policy version, and signed checkpoint attestations;
- remote authority identifier and relevant source/log/policy cursor digests;
- authority global revision as an observed audit watermark;
- per-key or per-scope modification revisions where the backend exposes them.

The global etcd revision is an audit watermark, not a rule that every unrelated write invalidates every capsule. Currentness must be evaluated against the relevant source, log, signer, trust-root, witness, and policy scopes. A fresh linearizable read of these scopes is required. A serializable/stale read or watch event alone cannot establish freshness.

### 3.4 Verification record

For each required obligation:

- stable obligation ID and version;
- result: valid, invalid, or unknown;
- reason code and bounded explanation;
- exact evidence references and digests consumed;
- verifier implementation/build digest;
- policy bundle digest;
- verification time and time source;
- applicable trust-root, signer, witness, and authority versions;
- re-verification/invalidation triggers.

Wall-clock verification time is descriptive metadata, not an anti-rollback anchor. Expiry is evaluated using the declared trusted time/freshness source. If the required time or currentness source is unavailable, the relevant result is unknown.

### 3.5 Dependency closure

Every dependency edge binds:

- dependency type;
- exact dependency subject digest;
- exact dependency capsule/receipt digest;
- edge semantics and version;
- invalidation rule.

The graph must be acyclic, canonical, and complete for the claim profile. A missing required edge makes applicability/currentness unknown. A cycle or conflicting duplicate edge is structurally invalid. Dependency aliases cannot replace digest identity.

No implicit transitive qualification is permitted. A parent's evidence does not qualify a child artifact, environment, workload class, or broader intended use unless the child claim explicitly binds that parent and verifies the edge semantics.

## 4. Capsule currentness and selective reuse

A capsule can be considered for reuse only after a new consumer-side currentness check:

1. Verify the capsule envelope and exact payload binding.
2. Verify the capsule subject and intended-use binding.
3. Verify proof material or retrieve it by immutable digest.
4. Re-evaluate each obligation under the current policy.
5. Obtain a linearizable authority snapshot for the relevant source/log/policy scopes.
6. Compare exact relevant cursor values, digests, policy versions, revocation state, and witness policy.
7. Check the complete dependency closure and invalidation rules.
8. Check the current environment/workload against the qualification envelope.
9. Derive the new top-level three-valued result from the per-obligation results.

The cache can nominate a capsule for re-verification. It cannot provide the final freshness verdict.

### 4.1 Avoiding over-invalidation

A global authority revision may advance because an unrelated source or key changed. Requiring exact equality with a previously observed global revision would invalidate unrelated capsules and can collapse selective reuse into global reruns.

Therefore:

- record the global revision observed for audit and ordering;
- bind the relevant source/log/policy scope keys and their exact current values or per-key revisions;
- require a linearizable snapshot that proves those relevant values are current;
- invalidate only when a relevant dependency, policy, scope cursor, trust root, signer status, witness policy, or qualification input changes;
- do not use a stale cache or watch notification as the currentness source.

For etcd, use linearizable range reads for trust decisions and compare-and-advance transactions for state mutations. Serializable member-local reads may be stale. A watch is a notification stream, not a proof of freshness.

### 4.2 Reuse result states

- **reusable:** exact claim, subject, use, context, dependencies, and current policy remain valid;
- **reverify-required:** one or more scoped dependencies changed and the verifier can re-evaluate them;
- **invalid:** a required proof or binding is disproven;
- **unknown:** a current authority, proof, revocation state, or required dependency cannot be established.

Only reusable with an aggregate valid result may satisfy an admission request. Reverify-required is not a pass.

## 5. Reference evaluation algorithm

The evaluator must be deterministic for a fixed capsule, claim profile, evidence set, current authority snapshot, and time/freshness input.

1. Parse the versioned profile and reject unsupported major versions.
2. Validate capsule schema, canonical encodings, and dependency DAG structure.
3. Bind the capsule to exact subject digest(s) and intended use.
4. Resolve the mandatory obligation set from the profile.
5. Independently evaluate each mandatory obligation using its verifier.
6. Evaluate applicability and dependency closure.
7. Obtain current scoped authority and policy state.
8. Map each result to valid, invalid, or unknown, retaining reason codes.
9. Reduce results using the truth table above.
10. Emit a complete machine-readable report; admission is true only for aggregate valid.

The evaluator must not trust a caller-supplied aggregate result, cached status, or boolean field. It recomputes the aggregate from verified obligation records.

## 6. Adversarial benchmark design

The normative initial suite is:

| ID | Attack or control | Required result |
|---|---|---|
| ADM-001 | All required proofs valid, exact context and use | valid |
| ADM-002 | One required obligation omitted | unknown |
| ADM-003 | One required proof disproven | invalid |
| ADM-004 | Invalid proof plus unknown proof | invalid, preserve both reasons |
| ADM-005 | Exact capsule reused for exact same subject/use/context with current scope | valid |
| ADM-006 | Subject digest substituted | invalid |
| ADM-007 | Intended use widened | invalid or unknown per profile, never valid by default |
| ADM-008 | Observer binary/configuration changed | unknown pending requalification |
| ADM-009 | Workload class outside qualified envelope | invalid for that use |
| ADM-010 | Required dependency missing | unknown |
| ADM-011 | Dependency digest substituted | invalid |
| ADM-012 | Dependency graph cycle | invalid |
| ADM-013 | Required dependency edge omitted | unknown |
| ADM-014 | Signer revoked or signer policy changed | invalid or unknown per authoritative revocation result; never valid on stale policy |
| ADM-015 | Relevant source/log/policy scope advanced | reverify or invalid; never cache-only valid |
| ADM-016 | Unrelated global authority key advances | no invalidation when relevant scopes are unchanged and linearly verified |
| ADM-017 | Authority unavailable | unknown |
| ADM-018 | Serializable stale read supplied as currentness evidence | unknown |
| ADM-019 | Local cache restored to older internally consistent snapshot | unknown without an independent current anchor |
| ADM-020 | Same-size log checkpoint presents different root | invalid fork |
| ADM-021 | Unauthorized signer has a mathematically valid signature | invalid authorization |
| ADM-022 | Caller supplies aggregate valid=true contrary to obligations | ignore caller result; recompute |
| ADM-023 | Unsupported major schema or ambiguous canonical encoding | invalid |
| ADM-024 | Required trusted time/currentness source unavailable | unknown |
| ADM-025 | Exact retry after remote commit response loss | same accepted receipt/revision, no second advance |
| ADM-026 | Obligation order permuted | identical result and normalized reason set |

The benchmark schema and machine-readable cases are maintained beside this document.

## 7. Comparison arms and metrics

Required comparison arms:

1. global rerun of all verification;
2. full capsule re-verification;
3. dependency-aware selective re-verification (#9203);
4. cache-only reuse (negative control; must never admit);
5. evaluator-only exact oracle on small graphs.

Workload matrix:

- 10, 100, and 1,000 dependency nodes;
- shallow/deep graphs, high fan-out, sparse/dense invalidation;
- 1%, 10%, 30%, and 70% dependency drift;
- policy/signer/trust-root/witness/authority changes;
- network outage and recovery;
- 1k, 10k, and 100k capsules.

Mandatory safety gates precede efficiency scoring:

- zero invalid or unknown capsule admissions;
- zero cross-subject or cross-use acceptance;
- 100% detection of stale required dependencies in the seeded mutation matrix;
- zero local-cache or restorable-authority snapshot rollback promotions without independent current evidence;
- zero policy downgrade acceptance;
- exact reference-vs-independent evaluator parity on the small oracle corpus;
- order-independent, deterministic outcomes;
- explicit unknown whenever freshness cannot be established.

Efficiency measures:

- invalidation precision and recall;
- unnecessary re-verification work;
- required verification work omitted (must be zero);
- verifier CPU time and total bytes read;
- proof bytes and network requests;
- p50/p95/p99 end-to-end admission latency;
- capsule size and dependency traversal cost;
- policy-revocation and authority-advance detection delay.

**Initial research target:** at least 30% less unnecessary verification work than global reruns on held-out drift-heavy workloads, with zero stale, invalid, or out-of-scope acceptance. This is a target, not a result.

Do not collapse metrics into one score. Publish raw measurements, workload mixes, hardware, software versions, policy, failure schedule, and confidence intervals. The independent sampling unit is a distinct subject/provisioning lineage and dependency graph, not repeated queries to the same cached capsule.

## 8. Industry alignment

| System | What it provides | What BASALT adds or must not assume |
|---|---|---|
| in-toto | Typed subject/predicate and envelope structure | Runtime source-epoch and observer-specific meaning |
| DSSE | Exact-byte signed envelope with PAE | Signer authorization remains a separate policy |
| Sigstore Bundle | Signature verification material, log entry and timestamp material | Capsule applicability and currentness remain BASALT obligations |
| Rekor | Transparency inclusion/checkpoint and consistency mechanisms | Inclusion alone does not prove latest-view freshness or eliminate split views |
| TUF | Versioned trust metadata and rollback/freeze defenses | Runtime source-generation continuity and observation qualification |
| etcd | Atomic transactions and linearizable reads | No BFT or malicious full-cluster snapshot guarantee |
| SQLite WAL/FULL | Single-host transactional durability | No external freshness or whole-snapshot anti-rollback |
| #9203 | Dependency-aware invalidation and selective recomputation | Claim-scoped admission over exact subject/use/context |

The differentiator is the composition of claim-specific applicability, independently verified evidence obligations, scoped freshness, and selective re-verification. BASALT should integrate established formats and services instead of building a competing signature format or transparency log.

## 9. Implementation sequence

1. Freeze obligation IDs, three-valued result algebra, and profile semantics.
2. Add a small pure reference evaluator and truth-table conformance tests.
3. Freeze a versioned capsule schema and canonical digest rules.
4. Add deterministic positive/negative/unknown fixtures and small exact dependency-graph oracle.
5. Bind currentness evaluation to #9203 dependency records and #9702 scoped authority reads.
6. Integrate verified receipt and witness outputs from #9560.
7. Add an independent evaluator implementation for cross-checking.
8. Measure global rerun, full capsule verification, and selective re-verification under matched conditions.
9. Only then consider production consumers or performance claims.

## 10. Explicit non-claims

This specification does not claim:
- Rust compilation or tests passed;
- production DSSE, Rekor, witness, etcd, or SQLite integration exists;
- observer or capture completeness is qualified;
- a local or remote snapshot cannot be maliciously restored;
- the 30% efficiency target has been achieved;
- SOTA has been surpassed.

Every implementation claim must name the exact commit, verifier/policy digest, test corpus, and evidence. Queued, skipped, missing, or stale CI is not a pass.

## 11. References

- [in-toto attestation v1](https://github.com/in-toto/attestation/tree/main/spec/v1)
- [DSSE protocol](https://github.com/secure-systems-lab/dsse)
- [Sigstore Bundle format](https://docs.sigstore.dev/about/bundle/)
- [Rekor witness and split-view design](https://github.com/sigstore/architecture-docs/blob/main/rekor-spec.md)
- [TUF specification](https://theupdateframework.github.io/specification/latest/)
- [etcd v3.7 transaction and range API](https://etcd.io/docs/v3.7/learning/api/)
- [SQLite WAL](https://www.sqlite.org/wal.html) and [synchronous durability](https://www.sqlite.org/pragma.html)
- [BASALT evidence freshness #9203](https://github.com/Luminous-Dynamics/symthaea/pull/9203)
- [BASALT receipt transparency #9560](https://github.com/Luminous-Dynamics/symthaea/issues/9560)
- [BASALT trusted state #9612](https://github.com/Luminous-Dynamics/symthaea/issues/9612)
- [BASALT remote authority #9702](https://github.com/Luminous-Dynamics/symthaea/issues/9702)
