# symthaea-epistemic-data-pack

Strict declarative import adapter for Symthaea's existing
`GlobalEpistemicLedger` domain type.

Third-party data is **not** deserialized directly into the canonical ledger.
This bridge owns a versioned, deny-unknown-fields wire format and performs
bounded semantic validation before conversion.

```text
third-party payload
      -> symthaea.epistemic.global_claims.v1
      -> GlobalClaimsPackValidator
      -> DataPackHost technical inspection
      -> signer/admission policy
      -> AdmissionAuthority + live currentness
      -> ScopedAdmission
      -> host-bound DataPackHost::open
      -> ValidatedDataPack
      -> explicit decode
      -> GlobalEpistemicLedger
```

## Capability

`knowledge.epistemic.global_claims`

The payload itself carries schema identity
`symthaea.epistemic.global_claims.v1`. Capability identity stays semantic while
the payload schema can evolve independently.

## V1 validation

- unknown top-level/claim fields are rejected;
- schema identity must match exactly;
- empty packs are rejected;
- claim count is bounded;
- domain/name/proof-reference strings are canonical, bounded and free of
  control characters;
- duplicate `(domain, name)` claim identities are rejected;
- a `proven` claim requires a non-empty `formal_proof_ref`;
- the proof reference is treated as opaque provenance text and is never opened
  as a filesystem path by this bridge;
- conversion must still pass the canonical ledger's own `audit_all()` check.

## Authority boundary

The generic data host is bound to one process-local `AuthorityScope`. A claim
pack therefore reaches this decoder only after its `ScopedAdmission` is proven to
belong to that exact host authority and current policy/trust/revocation state has
been rechecked at the use site. The epistemic adapter does not implement or
bypass those authority decisions.

## Why not reuse the domain Serde representation directly?

The canonical domain type is intentionally convenient for internal code and has
a permissive wire shape. Public extension input needs a stricter compatibility
and attack-surface boundary. Keeping a dedicated DTO means future public schema
changes do not force changes to internal epistemic representation, and vice
versa.

## Epistemic meaning

Passing this validator does **not** mean a claim is true. It means the package is
structurally coherent enough to enter Symthaea's epistemic machinery with its
claimed status/provenance intact. Evidence calibration, contradiction handling,
replication, source trust and Mycelix consensus remain higher-level concerns.

## Lean boundary

This crate depends on the existing epistemic types and the generic non-executable
data host. The authority crate appears only in tests because authority ownership
belongs to host orchestration, not epistemic semantics. Production code does not
depend on cognition, Wasmtime, networking, Holochain or a solver runtime.
