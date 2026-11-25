# Symthaea SAT Trace Verifier

A small, read-only verifier for SAT v0.1 event streams. It does not authorize
or execute actions, authenticate grants, validate signatures, issue
qualification certificates, or decide whether an AI system is aligned.

## Run

From the Symthaea workspace, run:

    cargo test -p symthaea-sat-trace-verifier
    cargo clippy -p symthaea-sat-trace-verifier --all-targets -- -D warnings
    cargo fmt --all -- --check
    cargo run -p symthaea-sat-trace-verifier --bin sat-trace-verify -- <run-uuid> <manifest.json> <events.jsonl>

Pass "-" as the final argument to read JSONL from stdin. The CLI computes the manifest digest from the frozen manifest file using the SAT v0.1 domain-separated JCS profile.

## Reproducible example

From the repository root:

    cargo run -p symthaea-sat-trace-verifier --bin sat-trace-verify -- 00000000-0000-4000-8000-000000000001 docs/research/examples/sat-trace-verifier-manifest-v0.1.json docs/research/examples/sat-trace-verifier-trace-v0.1.jsonl

Expected structural summary: 2 events, 1 authority binding, 1 disclosure binding,
2 sink projections, 1 effect submission, 1 terminal observation, no unresolved
attempts. The output must still report `signatures_verified=false` and
`system_qualified=false`. This fixture is synthetic, not a production effect.

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
