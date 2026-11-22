# SFEB trusted execution and causal repair

This directory defines the next assurance layer for SFEB. These schemas are **design artifacts**; they are not yet wired into the Rust adjudicator, browser executor, GitHub Actions workflow or Sigstore verification.

## Files

- `sfeb-trace-event-v1.schema.json` — normalized, privacy-aware events in a run-local SHA-256 hash chain.
- `sfeb-execution-predicate-v1.schema.json` — an in-toto Statement v1 carrying the custom SFEB execution predicate.
- `sfeb-causal-repair-v1.schema.json` — requirement-to-observation-to-intervention-to-invariant evidence for a repair.

## Trust model

The signed execution statement binds a run manifest to the campaign, case, task, source revision, fixture digest, environment digest, runner revision, executor image, browser binary, controller binary, isolation profile, network policy, event trace and captured artifacts.

The predicate intentionally has **no** `accepted` / `rejected` disposition and no runner-authored `observed_oracle_ids`. A signing identity proves the origin of the statement under the configured trust policy; it does not prove that a compromised runner observed reality correctly. The independent verifier must derive oracle outcomes from the trace and raw observation artifacts.

The Sigstore certificate identity and OIDC issuer must be checked against an out-of-band allowlist for the exact repository and immutable workflow identity. Values copied into the JSON predicate are claims, not trust roots. Never accept a wildcard identity policy, a user-controlled workflow, a mutable workflow reference or a signature alone as proof of correct execution.

## Trace rules

Each JSONL event includes a run ID, zero-based sequence, monotonic elapsed time, event type, actor, redaction state, payload and predecessor hash. Hash the RFC 8785 JCS encoding of the event with `event_sha256` omitted. The first predecessor is 64 zeroes. Each later predecessor must equal the previous event hash. The verifier checks uniqueness, continuity, canonicalization, run binding and the final digest in the signed statement.

Preserve only the minimum information necessary to replay the declared oracles. Redact credentials, tokens, personal data and user content before writing artifacts. A digest-only payload is acceptable only when the underlying observation is not required for independent oracle replay. Screenshots and DOM/accessibility traces remain potentially sensitive and require access controls and retention limits.

## Required execution boundary

- Ephemeral disposable microVM or independently reviewed isolation backend; non-root worker.
- Default-deny egress, explicit fixture-only origins and a separately hashed network policy.
- No host network, home-directory mounts, Docker socket or host browser debugging endpoint.
- CDP endpoint reachable only inside the sandbox; controller communication via a private pipe or equivalent isolated channel.
- Read-only fixture and source inputs; bounded CPU, memory, disk and wall time.
- Separate artifact collection and independent verification identities.
- Unknown termination, missing evidence, policy violation, identity mismatch or trace discontinuity means **inconclusive/rejected**, never accepted.

The current CDP capture primitive remains the browser observation substrate; this design does not introduce a second browser framework.

## Network-egress dependency

The isolated Chromium executor must integrate [#7435 — enforce outbound URL
policy before requests and redirects](https://github.com/Luminous-Dynamics/symthaea/issues/7435).
Controller-side URL validation is not an SSRF boundary and cannot stop an
already-issued redirect, subresource request or DNS-rebinding connection.
Treat the external egress policy as a hard gate; do not claim isolation until
the request/redirect policy and OS-level default-deny boundary are both
exercised with request/packet evidence.

## Verification sequence

1. Freeze and hash campaign, task, fixture, source baseline, environment manifest and policies.
2. Issue a unique run ID and challenge nonce; execute baseline and intervention in fresh isolated instances.
3. Collect the append-only trace and artifacts; create an in-toto Statement whose subjects are immutable run-bundle digests.
4. Sign the statement using a workload identity. Verify the Sigstore bundle, certificate identity, issuer, repository, workflow path and workflow commit against the out-of-band trust policy.
5. Verify all content digests, source/fixture/environment bindings, sandbox policy claims, run nonce, trace chain and artifact set.
6. Independently replay the trace and derive oracle outcomes. Do not ingest oracle IDs as truth from the worker's report.
7. Pair baseline/candidate runs, apply the frozen mutation/positive/metamorphic rules and only then produce the qualification summary.

SLSA provenance should separately identify how the runner, controller and evaluator were built. Do not pretend the custom browser-execution predicate is itself SLSA build provenance.

## Causal repair protocol

A diagnosis is an evidence graph, not a prose explanation:

`requirement -> contract assertion -> failed oracle -> trace event -> semantic region -> source anchor -> ranked root-cause hypothesis -> patch digest -> paired post-repair run -> invariant outcomes`

The `sfeb-causal-repair-v1` record binds the diagnosis and patch to pre/post execution statements and a same-fixture/same-environment comparison. The placebo statement is essential: a successful repair must beat a no-op or unrelated intervention, not merely coincide with a passing rerun.

A repair is not promoted as causally supported unless:
- the pre-repair failure is independently reproduced;
- the selected diagnosis is backed by trace events and stable source/region identity;
- the patch changes the target outcome under the same pinned conditions;
- unrelated functional, responsive, accessibility and security invariants remain passing;
- the placebo does not produce the same apparent improvement; and
- an independent hidden or randomized holdout does not reveal overfitting.

Confidence values must be calibrated on held-out diagnoses. Report top-1/top-3 localization accuracy, MRR, target-repair success, regression rate, placebo false-credit rate, patch size, attempts, time and cost per accepted repair.

## Next implementation gates

1. Implement JCS event hashing and chain validation in the standalone Rust contract checker, with tamper, reorder, omission, cross-run replay and duplicate-event tests.
2. Verify in-toto statements and Sigstore bundles against a pinned GitHub Actions workload-identity policy; add wrong-repository, wrong-workflow, wrong-ref, expired-certificate and altered-subject negative tests.
3. Implement the disposable Chromium runner on the existing `symthaea-browser` substrate, with the isolation boundary and network policy above.
4. Build an independent trace-to-oracle replay engine; remove trust in runner-supplied oracle IDs.
5. Run the 30-case paired qualification campaign; publish raw denominators and Wilson intervals.
6. Add causal-repair tasks with seeded defects, placebo patches, counterfactual controls and invariant checks.

## Evidence status

- Schema documents authored in this change.
- JSON parsing can be checked independently.
- Rust schema integration, cryptographic verification, executor isolation, trace replay, browser execution and qualification measurements remain pending.
- No signature, trusted-runner status, successful mutation campaign, causal-repair success rate or benchmark score is claimed.
