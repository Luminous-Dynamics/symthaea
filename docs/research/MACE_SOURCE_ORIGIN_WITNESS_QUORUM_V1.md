# MACE Source-Origin Witness Quorum V1

**Snapshot:** 2026-11-26  
**Status:** Consumer-side quorum verifier, policy schema, receipt schema and authored deterministic fixtures are published on `mace/source-origin-witness-quorum-v1`. Rust compilation, tests and hosted CI have not been executed in this environment.  
**Relationship:** Follow-on to [MACE Source-Origin Attestation V1](MACE_SOURCE_ORIGIN_ATTESTATION_V1.md) and [MACE Source-Origin and Provenance Assurance Benchmark V1](MACE_SOURCE_ORIGIN_ASSURANCE_BENCHMARK_V1.md).

## 1. Why a fourth authority is not enough

A separately pinned source-origin verifier improves byte integrity, but one origin operator can still be wrong, compromised, misconfigured, or dependent on the same source of metadata as the lineage resolver.

Adding signatures is not the same as adding independent observations. Three keys controlled by one organization, using one archive, network, cloud identity, or metadata feed, may constitute one failure domain.

The next assurance step is therefore **independent source observation**: multiple operators acquire and describe source identity, revision and ancestry through separately governed acquisition paths, and the consumer checks both operator diversity and failure-domain diversity.

## 2. Consumer-owned 2-of-3 policy

The new `source_origin_witness` module implements a policy-configurable quorum. The intended initial profile is two observations from three provisioned operators, with at least two distinct acquisition failure domains.

The policy binds:
- the exact replayed source-lineage graph digest;
- the required number of distinct operators;
- the required number of distinct failure domains;
- maximum observation age;
- a stable operator ID, failure-domain commitment and unique pinned trust-profile digest for each permitted observer.

The consumer provisions this policy and its profiles out-of-band. The candidate memory backend, lineage resolver, evaluator and source-origin verifier cannot supply or weaken it.

The verifier also rejects reuse of any previously used authority profile or signing key. It inspects all provisioned keys in each witness profile, including revoked or currently unused keys, to prevent a key from being silently reused across operators or protected roles.

## 3. What each witness signs

Each independent observer emits one DSSE/in-toto statement per graph node it observed. The signed receipt binds:

- exact run, query invocation and graph digest;
- exact SHA-256 of the corresponding fourth-authority source-origin envelope;
- consumer-pinned operator identity and failure-domain ID;
- source ID, source bytes digest, payload digest and evidence class;
- independently observed parent IDs, derivation receipt and common-cause groups;
- publisher-identity, acquisition-record and source-revision commitments;
- observation time, issue/expiry interval, witness profile and signer key.

A receipt is not accepted simply because its signer is trusted. Its source, payload, revision, publisher and acquisition commitments must match the replayed graph and the exact fourth-authority source-origin receipt. The witness must also independently report the same parent, derivation and common-cause edges.

## 4. The critical disagreement rule

**A conflict is not resolved by majority vote.**

If any signed witness reports a source digest, parent, derivation receipt or common-cause group that differs from the replayed graph, the consumer returns a typed mismatch/disagreement result. It does not silently discard the dissenting observer and accept the majority.

A consumer may then:
1. re-acquire the disputed source through another method;
2. request the missing parent or transformation record;
3. ask an independent adjudicator to review the specific conflict;
4. return indeterminate or abstain.

This prevents a 2-of-3 policy from laundering a known ancestry omission merely because two witnesses copied the same incomplete metadata. Quorum is used to establish corroboration only after the observations are mutually consistent with the exact graph. It is not an authority to overrule evidence of a missing dependency.

## 5. What this does and does not prove

The verifier can establish that:
- each accepted receipt was signed under the operator's currently pinned profile;
- the operator and failure-domain IDs satisfy the consumer policy;
- every graph node has enough consistent observations from distinct operators and failure domains;
- the observations bind to the exact graph and exact fourth-authority origin receipt;
- observations are within the declared freshness and validity windows;
- authority profiles and keys are not reused across protected roles.

It does **not** prove that:
- the operator IDs correspond to genuinely independent organizations;
- the failure domains do not secretly share a parent company, cloud, network, archive, source feed, software stack or credential administrator;
- all operators can discover every real-world upstream dependency;
- source publisher claims are accurate;
- the underlying source is factually true or fit for a decision.

Those assumptions require an externally governed operator registry, independent acquisition tests, audit of control-plane ownership, and adversarial hidden-parent fixtures. A 2-of-3 receipt should never be called "ground truth."

## 6. Interoperability: PROV and SCITT

MACE should avoid inventing a competing general-purpose provenance vocabulary. Use a narrow MACE assurance envelope, while supporting a lossless mapping to the W3C PROV model:

| MACE object / edge | W3C PROV concept |
|---|---|
| Raw source, evidence payload, derivation receipt, graph manifest | `prov:Entity` |
| Acquire, extract, transform, normalize, consolidate | `prov:Activity` |
| Publisher, acquisition operator, transformer, verifier | `prov:Agent` |
| Evidence derived from parent evidence | `prov:wasDerivedFrom` |
| Payload generated by extraction/transformation | `prov:wasGeneratedBy` |
| Activity performed by an operator | `prov:wasAssociatedWith` |
| Entity attributed to a publisher or producer | `prov:wasAttributedTo` |

The mapping should preserve exact content digests and MACE-specific DecisionNeed bindings as qualified attributes rather than treating PROV alone as an authorization policy.

For transparency, each issuer can publish its signed statements to a SCITT transparency service. [IETF RFC 9943](https://www.rfc-editor.org/rfc/rfc9943/) defines an architecture for transparency of single-issuer signed statements. This is useful for finding equivocation, replay, hidden statement history and inconsistent log views. It is **not** a proof that the issuer's statement is true, nor a substitute for independent operators. MACE should keep per-issuer statements separately attributable and use independent logs or witnesses when log consistency itself is in scope.

In other words:
- PROV gives interoperable provenance semantics.
- DSSE/in-toto binds signed statements to exact payload bytes.
- SCITT provides transparency around issuer statements.
- MACE's consumer policy handles decision-specific admissibility, authority separation, freshness, quorum and abstention.

No one standard supplies all four layers.

## 7. Benchmark additions

Add a separate witness-quorum scorecard to the MACE assurance benchmark:

| Metric | Proposed target |
|---|---|
| Operator identity reuse | 0 accepted profile/operator/key reuse across distinct operators |
| Failure-domain overcount | 0 cases where two operators mapped to one declared domain satisfy a two-domain policy |
| Per-node witness coverage | 100% of graph nodes meet the configured quorum |
| Hidden-parent disagreement detection | 100% of signed, known hidden-parent fixtures become disputed |
| Majority-laundering rate | 0 acceptance when any valid witness receipt reports a parent/common-cause mismatch |
| Observation freshness | 0 stale witness observations accepted |
| Witness outage | Abstention/insufficient-witness result, never silent fallback to single-source trust |
| Acquisition cost | Report per-node bytes, p50/p95 latency and cost per independent failure domain |

Run at least these ablations:
1. one origin authority only;
2. 2-of-3 keys from one administrative domain;
3. 2-of-3 operators but one shared acquisition domain;
4. 2-of-3 operators and two or more distinct declared failure domains;
5. the same quorum with one deliberately dissenting parent/common-cause observation;
6. quorum plus an independently witnessed transparency log.

The benchmark must score false acceptance and false rejection separately. If an observer is unavailable, a consumer may accept only when every graph node still satisfies the configured per-node quorum and failure-domain threshold; otherwise it must return insufficient evidence. A timeout is not an implicit witness.

## 8. External memory benchmarks remain separate

This quorum does not improve AML recall scores by itself. It addresses a different capability. MACE must still run the official Agent Memory Leaderboard tracks separately and compare agentic memory against AMA-Bench, MemoryArena, Memora/FAMA, MEMORA-Bench and LongMemEval-V2 with matched harnesses and budgets.

The key opportunity is to connect memory-guided decisions to independently corroborated evidence. MemoryArena shows why recall-only evaluation is not sufficient: agents must acquire, consolidate and reuse experience across interdependent tasks. AMA-Bench separately emphasizes causal inference and state updates across trajectories. MACE should combine those memory capabilities with evidence provenance and decision-use assurance without averaging the benchmarks together.

## 9. Validation status and next steps

- Rust witness quorum module and public module export are authored.
- Consumer policy and per-node witness receipt schemas are published.
- Deterministic tests cover 2-of-3 policy validity, shared failure-domain rejection, duplicate observations, policy-digest binding and hidden-parent/common-cause disagreement.
- Rust tests, Cargo checks, rustfmt, Clippy and hosted CI have not been run here.
- The policy's declared failure domains remain trust assumptions until the operator registry and independent acquisition infrastructure are deployed and audited.

Next work should implement the actual independent acquisition adapters and operator registry, add signed end-to-end 2-of-3 fixtures, export the provenance graph through a W3C PROV mapping, and publish issuer statements to a SCITT-compatible transparency service. Those services remain separate deployment work; the verifier library must not pretend they already exist.
