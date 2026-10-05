# Verification-Method Resolution Snapshot Contract

## Purpose

A verification report that records a resolver snapshot must bind the resolved verification method and the snapshot identifier to the same resolution event.

The snapshot is evidence about the resolver state consulted by verification. It is not a claim that the underlying attestation, observation, or external-world fact is true.

## Execution context versus evaluation annotations

The v8 Evidence Fabric representation retains evaluator identity, trust-root identity, and authorization-policy identity inside the historical VerificationContext representation for compatibility. Those fields are supplemental: VerificationContext::matches_report() deliberately excludes them from execution binding.

The additive v9 representation makes this distinction structural rather than relying on reader interpretation:

- VerificationContextV5 contains only execution-bound appraisal context: policy, verifier identity/version, environment, procedure, resolver snapshot, and evaluation time.
- EvaluationContextSupplement contains evaluator identity, trust-root, and authorization-policy fingerprints as separately versioned provenance annotations.
- The context and supplement each have independent canonical encodings and BLAKE3 fingerprints.
- EvidenceEvaluationV9 includes the two fingerprints separately and uses a distinct v9 evidence domain, so the v8 durable identity is not silently rewritten.

The split does **not** establish evaluator independence, trust-root correctness, or authorization correctness. Those remain separate claims that require their own evidence and appraisal policy.

The v9 outer evaluation fingerprint includes the supplement fingerprint, so an annotation change changes the v9 evidence identity without changing the execution-context fingerprint. Consumers that persist a supplement separately SHOULD retain its association with the corresponding v9 subject/report identity; the supplement is provenance metadata, not a portable authority to override another evaluation.
Additionally, every v9 supplement is bound to the exact execution-context fingerprint it annotates. An unbound supplement may be attached once during checked v9 materialization; once a supplement already carries a context binding, checked construction requires that binding to equal the target context and rejects foreign bindings rather than silently rebinding provenance. A structurally valid supplement with a foreign context binding is therefore rejected by v9 self-validation. This establishes association, not contemporaneity, evaluator independence, trust-root correctness, or authorization correctness.
v9 further freezes the attachment phase as post-evaluation. The record therefore does not imply that the supplemental evaluator/trust-root/authorization annotation was itself a governing input to the original procedure. A future concurrent-annotation contract, if ever needed, must be a distinct explicit semantic/version boundary rather than inferred from the existing v9 representation.



This structure follows the RATS architecture's distinction between verifier-side evidence appraisal inputs and later relying-party appraisal. Current RATS Attestation Result work likewise models contextual information as part of the appraisal result while keeping the relying party's subsequent policy decision distinct.

## Attestation payload binding

Current v5 verification reports also bind a fingerprint of the exact attestation payload whose detached proof was evaluated.

The report MUST therefore identify:

- the underlying receipt;
- the complete attestation payload;
- the verification procedure and policy;
- and the resolver state used for method lifecycle and authorization

rather than treating the receipt fingerprint alone as the identity of the attested statement.

Two distinct signed attestation envelopes over the same receipt MUST produce distinct current report identities whenever their canonical payloads differ. This prevents metadata changes such as domain, challenge, attester identity, or validity interval from disappearing behind an otherwise identical receipt-level subject.

## Attester identity boundary

The core envelope API also exposes `verify_against_integral_receipt()`. This opt-in check first verifies the receipt's own assessment commitment and then verifies the envelope↔receipt commitment. It is appropriate for callers that require internal receipt coherence, while leaving the historical Evidence Fabric v5 procedure unchanged until a future versioned procedure explicitly adopts that stronger semantic.


The envelope's `attester_id` is part of the signed attestation payload, so changing it changes the payload identity and invalidates an existing detached proof. However, this field is an opaque attester/issuer identifier, not proof that the identifier controls the verification method used by the detached signature.

A successful Evidence Fabric appraisal therefore establishes the cryptographic proof under the resolved verification method and the resolver's proof-purpose authorization state. It does **not** independently establish a controller relationship between `attester_id` and `verification_method`. Applications that need that stronger identity claim MUST obtain and verify the corresponding controller/authorization evidence as a separate trust step.

This distinction mirrors the broader RATS model in RFC 9334 and the current EAR-04 Internet-Draft: the verifier produces appraisal results from evidence and policy, while a relying party may apply additional policy to decide whether the resulting evidence is sufficient for a particular authorization decision.


## Required semantics

### 1. Resolution and snapshot are one evidence unit

Implementations backed by mutable or remote state should implement `VerificationMethodResolver::resolve_with_snapshot` directly.

The implementation MUST obtain:

- the resolved verification method;
- the lifecycle state used for verification;
- the proof-purpose authorization state used for verification; and
- the durable snapshot identifier/fingerprint

from one atomic or otherwise consistency-preserving view.

A caller MUST NOT assume that calling `resolve()` and then `snapshot_fingerprint_for()` is atomic.

The compatibility default for `resolve_with_snapshot` performs only the resolution and returns `snapshot_fingerprint: None`. It intentionally does not combine two independently observed states and therefore does not make an atomicity claim.

Resolvers that can provide a consistency-preserving or atomic view MUST override `resolve_with_snapshot` before returning a snapshot identifier. The in-memory resolver does so by deriving its method-scoped snapshot from the exact resolved method value it returns.

### 2. Snapshot identifiers are opaque

A snapshot fingerprint is an opaque resolver-provided identifier.

Consumers MUST NOT assume:

- a particular hash algorithm;
- a fixed hexadecimal length;
- that the identifier is locally recomputable;
- or that it represents the entire resolver registry.

A resolver may instead return a durable database revision, transparency-log position, content-addressed identifier, or another stable identifier whose semantics are documented by that resolver.

An empty or whitespace-only snapshot value is invalid resolver metadata. If no durable snapshot identifier is available, the resolver MUST return `None` rather than an empty identifier.

### 3. Scope should match the evidence being asserted

Where a resolver can provide it, `snapshot_fingerprint_for(method)` SHOULD identify the state relevant to the requested verification method rather than unrelated registry state.

The in-memory resolver therefore exposes:

- the historical whole-registry v1 fingerprint for compatibility; and
- a method-scoped v2 fingerprint for new resolution evidence.

Changing unrelated verification methods should not invalidate a scoped snapshot for a method whose resolution facts did not change.

### 4. Resolver failure and identity mismatch are explicit

A resolver failure does not constitute successful resolution. The report may retain the requested method identifier for diagnostics, but a `resolved_verification_method: Some(...)` value MUST NOT be interpreted by itself as evidence that the resolver successfully returned that method.

Likewise, if a resolver returns a method whose identifier does not match the requested identifier, verification MUST terminate rather than verify with the mismatched result. The paired snapshot, when supplied, remains provenance for the resolver result that was actually returned; it does not convert the rejected result into a successful resolution.

### 5. Missing snapshots are explicit

`None` means that the resolver did not provide a durable snapshot identifier.

Consumers that require replayable/auditable resolution evidence SHOULD require an explicit snapshot identifier rather than silently treating `None` as proof that resolution was durable.

### 6. Snapshot evidence does not establish truth

A valid snapshot proves only that the verifier recorded a particular resolver state or durable state reference.

### 6.1. Snapshot identity is not freshness proof

A resolver snapshot identifier MUST NOT be treated as a proof-of-freshness value unless the resolver contract explicitly defines freshness semantics for that identifier.

Likewise, the report's `evaluated_at_unix_ns` records the verifier's evaluation time; it is not, by itself, evidence that the attestation or resolver state was fresh at that instant. Replay-sensitive deployments SHOULD use an independently bound freshness mechanism, such as a verifier-originated challenge/nonce, an authenticated epoch, or a synchronized-clock validity rule, and SHOULD preserve the evidence needed to validate that mechanism during later audit.

This distinction follows the current RATS direction: multi-verifier appraisal calls for freshness mechanisms such as synchronized clocks, epoch IDs, or nonces, while AR4SI treats claim freshness as a distinct category from identity and trustworthiness claims. RFC 9943 likewise emphasizes retaining enough evidence to reproduce checks that were applicable at the time a statement was accepted.

A resolver snapshot can therefore provide strong **state identity** without providing **state freshness**. Consumers MUST keep those properties separate.

It does not establish:

- truth of the attested observation;
- correctness of an external-world claim;
- signer intent;
- evaluator independence;
- or correctness of the resolver's own data.

Those remain separate epistemic boundaries in the Evidence Fabric.

For consumers holding both a report and the original envelope, `ReceiptAttestationVerificationReport::matches_attestation_envelope()` provides the explicit report↔payload identity check. It first rejects internally inconsistent report identity bindings and structurally invalid envelopes, then verifies the current v5 payload fingerprint and receipt fingerprint match the supplied envelope. It does not replace full report execution-trace validation, detached proof verification, or resolver validation.

## Replayability guidance

For durable deployments, retaining only a resolver identifier may be insufficient if the referenced state can later disappear.

A deployment SHOULD retain enough resolver evidence to reconstruct the verification decision at the time it was made. Depending on the resolver, that can mean retaining the relevant signed registry statement, authenticated key material, authorization policy, lifecycle evidence, and the durable state identifier.

This mirrors the broader auditability principle in RFC 9943: systems intended for later audit should retain enough information to reproduce the checks that were applicable when a statement was accepted. RFC 9943 also distinguishes authentication/registration evidence from the truth or accuracy of the statement itself.

## Graph-relative receipt revalidation

Receipt integrity, attestation verification, and current graph correspondence are separate evidence properties.

`IndependenceVerificationReceipt::verify_integrity()` answers whether the receipt's stored assessment fingerprint is consistent with the receipt fields. A receipt can therefore remain intrinsically coherent after the observation graph changes.

`IndependenceVerificationReceipt::verify_against_graph_detailed(&graph)` is the stronger operation. It re-runs the bounded independence assessment against the supplied closed-world graph and distinguishes:

- `VerifiedAgainstGraph`: the current graph reproduces the receipt's recorded assessment;
- `GraphMismatch`: the receipt is internally coherent, but the current graph no longer reproduces that assessment;
- `InvalidReceiptIntegrity`: the receipt itself is internally inconsistent;
- `UnsupportedVerifierVersion`: the receipt was produced under an unsupported assessment contract.

`ReceiptAttestationVerificationReport` and `EvidenceEvaluation` do not perform that graph revalidation. Their receipt fingerprint identifies the exact historical receipt that was attested; it is not a graph snapshot, a current-state proof, or a substitute for re-running the assessment.

### Consumer rule

Consumers that make a **current graph-state decision** MUST obtain the authoritative observation graph and perform `verify_against_graph_detailed()` (or an equivalent explicitly recorded graph-relative verification). A historical `Verified` attestation or a well-formed `EvidenceEvaluation` MUST NOT be promoted into a claim that the same result still holds in the current graph.
Before making that decision, consumers SHOULD record the `ObservationGraph::independence_verification_scope_fingerprint()` value produced from the authoritative graph. This is the reproducible identity of the **verifier-relevant provenance scope**, not a full-graph digest.

That distinction is intentional: the bounded independence procedure does not inspect every observation field or relation semantic. An unchanged scope fingerprint therefore means the fields relevant to this particular verifier are unchanged; it does not mean every graph property is unchanged or freshly trusted.

When the authoritative graph is unavailable, the current-graph correspondence state MUST remain explicitly unestablished or indeterminate. Consumers SHOULD retain the graph-relative verification outcome, verifier version, and scope evidence needed to audit why a current-state decision was accepted.

This is intentionally analogous to freshness: a durable identifier can establish which state was referenced without proving that the referenced state is still current. The current RATS Many-Verifiers draft explicitly places freshness verification on verifiers, and the RATS Epoch Markers draft notes that some freshness mechanisms require receiver-side state to determine whether a marker is still the appropriate unused value. The W3C Verifiable Credential Data Integrity 1.1 Working Draft likewise treats proof purpose, domain, and challenge as explicit verification inputs rather than as general semantic truth guarantees.

## Compatibility rule

Historical report fingerprints must remain reconstructable.

New snapshot semantics may therefore require a new versioned canonical representation while preserving the old representation for historical evidence. The current v1 whole-registry fingerprint is retained for that reason; the v2 method-scoped fingerprint is a new representation and must not silently rewrite historical v1 identities.

## Threat model

The primary failure this contract prevents is state skew:

1. resolve a key from state A;
2. the backing registry changes to state B;
3. independently compute a snapshot for state B;
4. emit evidence that appears to bind the key to B.

That evidence is internally coherent only at the serialization layer; it does not prove that B produced the key actually used for verification.

The atomic `resolve_with_snapshot` contract removes this ambiguity for resolvers that can provide a consistency-preserving view. The compatibility default no longer pretends to provide that guarantee.
