# Symthaea SAT Trace Verifier

A small, read-only verifier for SAT v0.1 event streams. It does not authorize
or execute actions, authenticate grants, validate signatures, issue
qualification certificates, or decide whether an AI system is aligned.

## Run

From the Symthaea workspace, run:

    cargo test -p symthaea-sat-trace-verifier
    cargo clippy -p symthaea-sat-trace-verifier --all-targets -- -D warnings
    cargo fmt --all -- --check
    cargo run -p symthaea-sat-trace-verifier --bin sat-trace-verify -- <run-uuid> <manifest.json> <events.jsonl> [expected-head-digest]

Pass "-" as the events argument to read JSONL from stdin. The CLI computes the manifest digest from the frozen manifest file using the SAT v0.1 domain-separated JCS profile. The optional final argument is an externally supplied expected head digest. Without it, the verifier reports internal chain consistency only; with it, the verifier compares the computed terminal digest to that value but does not authenticate the anchor source or signature.

## Reproducible example

From the repository root:

    cargo run -p symthaea-sat-trace-verifier --bin sat-trace-verify -- 00000000-0000-4000-8000-000000000001 docs/research/examples/sat-trace-verifier-manifest-v0.1.json docs/research/examples/sat-trace-verifier-trace-v0.1.jsonl

Expected structural summary: 2 events, 1 authority binding, 1 disclosure binding,
2 sink projections, 1 effect submission, 1 terminal observation, no unresolved
attempts. Append the fixture's expected terminal digest
`sha256:bb201acf3e03cd941e35e0db7d2e3a9e1a81f20378acce81f56412b142cc3fd0`
to exercise the optional head comparison. The output must still report
`signatures_verified=false` and `system_qualified=false`. The static example
anchor is a test vector, not a qualified external witness or production effect.

## External checkpoint receipt inspection

The `checkpoint` module provides `inspect_checkpoint_receipt`. It checks that a
receipt matches the verified run/manifest/head, recomputes the canonical
trace-head statement digest, validates entry index/tree-size shape, detects
conflicting roots for the same service/log/tree-size, checks each witness
against a supplied log root, and enforces the declared count of distinct
witness IDs.

This is **structural inspection only**. It does not verify checkpoint or witness
signatures, transparency inclusion proofs, consistency proofs, key status,
revocation, witness independence, trust-policy governance or artifact custody.
Its report always has `signatures_verified=false`,
`inclusion_proofs_verified=false`, `consistency_proofs_verified=false`,
`witness_independence_verified=false` and
`checkpoint_authenticated=false`.

See `docs/research/SAT_TRACE_CHECKPOINT_PROFILE_V0_1_2026-11-25.md` for the
portable receipt contract and SCITT / Sigstore Rekor adapter boundary.

## Integrity boundary

The unkeyed event hash chain establishes internal ordering and digest consistency. An attacker who can rewrite the entire trace can recompute the chain. To detect that, the terminal digest (or a checkpoint) must be stored with an independent, qualified witness or signed/custodied receipt. The optional CLI anchor only compares bytes; this crate does not authenticate the witness, signature, key status or custody.

## What it verifies

- Strict JSON parsing; duplicate object keys and non-safe-integer numbers fail closed.
- The SAT v0.1 restricted JCS profile, including recursive object sorting by
  UTF-16 code units and preservation of Unicode strings without normalization.
- Exact run and frozen-manifest identity on every event.
- Consecutive event sequence, unique event IDs, prior-only causal references,
  same-correlation/direct-parent linkage for effect observations, previous-event
  digest and domain-separated event digest.
- Full intent, authority exact-effect, disclosure and versioned sink-projection
  commitments.
- Independent sink-observer role/identity and observed projection equality.
- Effect-submission/observation correlation by both idempotency_key and unique
  attempt_id; delayed observations for an earlier attempt cannot settle a retry.
- Hold-before-new-mutation semantics for unknown external outcomes; terminal
  observations cannot be silently overwritten.

## Deliberate limits

The verifier checks evidence structure and internal consistency. It does not
establish that a signer is authorized, a root grant is authentic, a policy is
current, labels are truthful, a sink observer is independent, a source of truth
is reliable, or an observed projection corresponds to the physical world.
It does not verify signatures or revocation status. Those require separately
qualified trust roots, evidence custody, source-of-truth adapters and the
production owners named in SAT's protocol.

A successful trace verification is not an overall SAT qualification and must
never be described as proof of safety. The report always states that signature
verification was not performed. Unresolved external effects are returned as a
hold condition; the verifier never performs retries or mutations.
