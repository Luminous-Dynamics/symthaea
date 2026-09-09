# HAK-014 — Deterministic Normalization Execution & Receipt v1

Status: implementation + audit-only evidence tooling candidate. No runtime authority changes.

## Purpose

HAK-013 makes normalization policy content-bound and binds selector grammar identity. HAK-014 closes the next composition boundary:

```text
NormalizationPolicyDefined
!=
NormalizationPolicyExecutedByCollector
```

HAK-014 implements `hak.selector-path` v1 and emits a `NormalizationExecutionReceiptV1` proving what the normalization executor actually did with exact input bytes, policy content and interpreter bytes.

## Execution chain

```text
RawProviderResponse bytes
+
Exact HAK-013 NormalizationPolicyV1
+
hak.selector-path v1
+
Exact executor artifact
        ↓
NormalizationExecutionReceiptV1
        ↓
future ProviderSourceObservation
```

The receipt does **not** contain a future observation digest. This keeps construction directional and avoids a receipt↔observation identity cycle.

```text
ReceiptIdentityMustNotDependOnFutureObservationIdentity
```

## Strict raw JSON grammar

Provider input is treated as strict JSON evidence, not as whatever extensions the host parser happens to accept.

HAK-014 rejects:

- duplicate object keys rather than accepting implicit last-key-wins semantics;
- non-standard numeric constants such as `NaN`, `Infinity`, and `-Infinity`;
- invalid JSON encodings/syntax;
- non-object roots for the supported provider resource profiles.

```text
ParserAccepted != EvidenceUnambiguous
DuplicateKey != DeterministicFieldIdentity
NonStandardNumber != JSONEvidence
```

The exact raw bytes remain independently bound by `raw_response_digest`, so lexical representation is preserved even though selected evidence is normalized into canonical JSON for local digests.

## hak.selector-path v1

The v1 grammar permits only dot-separated object fields and array wildcards:

```text
field
field.child
array[*].field
array[*].child.field
```

No numeric indexes, filters, recursive descent, arbitrary expressions or provider-specific evaluation are permitted.

### Required containers

`required_containers` require an exact provider path and exact container type. They preserve shape only:

```text
array -> preserve cardinality/order and empty structural slots
object -> preserve empty object shape
```

They do not authorize importing unselected descendants.

### Required selectors

A required non-wildcard selector must be present. `null` is a present value and is preserved as `null`.

For required selectors beneath a wildcard:

```text
empty correctly typed array
-> vacuously satisfied

non-empty array
-> every existing element must satisfy the required descendant
```

This avoids inventing elements for an empty provider result while refusing to hide missing required evidence inside existing elements.

### Selector evidence states

Requirement satisfaction is not the same thing as direct observation. HAK-014 therefore distinguishes:

```text
Present
VacuouslySatisfied
Absent
Failed
```

For required selectors:

```text
Present
-> one or more direct matches were observed

VacuouslySatisfied
-> a wildcard requirement had zero applicable elements
   and zero missing evidence

Failed
-> required evidence/type semantics were violated
```

For optional selectors:

```text
Present
Absent
Failed
```

`VacuouslySatisfied` is not legal for optional selectors because optional absence is represented directly as `Absent`.

```text
RequirementSatisfied != EvidenceObserved
VacuouslySatisfied != Present
AbsenceOfCounterexample != PositiveObservation
```

The receipt validator enforces these semantics even if an attacker recomputes the canonical receipt digest after changing a selector status.

### Optional selectors

Missing optional fields do not fail normalization. Under wildcards, provider array cardinality and order are preserved; elements with no optional match remain structural empty objects until another selected descendant fills them.

### Schema drift

An expected array observed as an object, or a wildcard applied to a non-array value, is a normalization error. Provider shape drift is not silently reinterpreted.

```text
ObservedShape != ExpectedShape
```

## Terminal states

The receipt has three terminal states:

```text
Succeeded
Partial
Failed
```

`Succeeded` requires zero errors.

`Partial` means some selected evidence was retained but at least one required or type-safety condition failed.

`Failed` means normalization could not produce a meaningful selected result, including invalid JSON/non-object root cases.

Both `Partial` and `Failed` are valid negative execution evidence when their receipt integrity checks pass.

```text
FailedNormalization != InvalidReceipt
```

Neither permits downstream observation issuance:

```text
observation_issuance_permitted
==
(execution_status == Succeeded)
```

## Receipt bindings

`NormalizationExecutionReceiptV1` binds:

- exact raw input byte digest;
- raw source reference;
- policy ID and canonical HAK-013 policy digest;
- exact policy artifact ref;
- selector grammar ID/version;
- interpreter ID/version;
- exact interpreter artifact ref;
- interpreter source-content digest;
- typed-container outcomes;
- required/optional selector outcomes and evidence states;
- selected payload;
- domain-separated selected-result digest;
- explicit errors and terminal state;
- observation-issuance decision;
- canonical receipt digest.

### Import-time interpreter source snapshot

HAK-014 captures its source-file bytes at module import and refuses caller-supplied `interpreter_bytes` unless they equal that import-time snapshot. This prevents later filesystem mutation from silently changing the claimed content digest while already-loaded module code continues to run.

```text
FilesystemBytesNow != ImportTimeSourceSnapshot
ClaimedInterpreterBytes != ImportTimeSourceSnapshot
```

This is still a bounded claim: HAK-014 does not prove that Python's in-memory code object is cryptographically derivable from those source bytes, nor does it authenticate that a caller-supplied Git artifact ref resolves to them. Those require stronger execution/build provenance.

### Deterministic replay

A receipt can be internally self-consistent and still be a coherent fabrication. For example, an attacker could alter `selected_payload`, recompute `selected_result_digest`, and recompute `receipt_digest` while leaving every input digest intact.

Therefore stronger input validation does not stop at identity comparison. It re-executes the deterministic normalizer using the supplied raw bytes, loaded policy, expected resource/source identities, artifact refs, and import-time interpreter source bytes, then requires the supplied receipt to equal the replayed receipt.

```text
InputIdentityMatch != DeterministicTransformationReplay
ReceiptSelfConsistency != ReceiptInputBinding != ReceiptReplayEquivalence
```

This closes the tested class of coherent redigested output forgeries.

## Omission and ordering

Unknown provider fields are never copied merely because they are adjacent to selected data. Provider array order is preserved exactly; HAK-014 does not sort provider arrays to make evidence look canonical.

Canonical JSON is used only for local content digests, with non-standard numeric values forbidden, not to rewrite provider ordering semantics.

## Qualification contract

HAK-014 has a precommitted E5-target plan, HAK-010 obligation-to-step binding policy and a dedicated exact-head `HAK Normalization Execution` workflow.

The focused tests cover:

- deterministic unknown-field omission;
- provider array-order preservation;
- duplicate raw JSON key rejection;
- non-standard JSON numeric-constant rejection;
- empty wildcard vacuous satisfaction without mislabeling it as direct presence;
- required descendant failure in existing wildcard elements;
- missing/wrong required containers;
- optional descendant absence;
- optional selectors refusing `VacuouslySatisfied`;
- required `null` preservation as direct presence;
- invalid JSON and non-object roots producing valid failed receipts;
- raw-byte substitution;
- policy substitution;
- interpreter substitution;
- mismatch between claimed and import-time interpreter bytes;
- selected-payload tampering;
- coherent selected-payload/result/receipt redigest rejected by deterministic replay;
- receipt-digest tampering;
- observation-authority escalation from a Partial receipt;
- unsupported selector grammar substitution;
- receipt JSON-schema parity.

Keep the tranche draft until the dedicated workflow passes on its exact head.

## Non-claims

HAK-014 does not:

- authenticate GitHub or another provider;
- prove a source reference identifies authentic provider bytes;
- prove a Git artifact ref resolves to the import-time source bytes without external provenance evidence;
- cryptographically attest the in-memory Python code object;
- issue a ProviderSourceObservation by itself;
- prove semantic claim truth;
- establish scientific validity;
- certify governance, robotics or consent behavior;
- grant runtime authority.

```text
PolicyExecutedCorrectly
!= ProviderAuthenticated
!= SemanticTruth
!= Authority
```

## Next boundary

A future prospective collector may consume only `Succeeded` execution receipts to materialize a policy-bound `ProviderSourceObservation`, preserving a one-way construction:

```text
NormalizationExecutionReceipt
        ↓
ProviderSourceObservation
```

If a reverse binding is needed for evidence navigation, use a separate later binding artifact rather than making either identity self-referential.
