# Symthaea Music Theory Patch Series 24 Plan

**Date:** 2026-07-21  
**Base:** exact demonstrated Patch Series 23 final tree  
**Theme:** Resource budgets, bounded parsing, archive safety, and deterministic abuse rejection

## Executive summary

Public evidence verification is an adversarial input boundary. Correct signatures and hashes do not matter if a malformed catalog, witness set, conformance bundle, or archive can exhaust memory, trigger millions of external verifier calls, expand without bound, traverse the filesystem, or leave partial accepted state after cancellation.

Series 24 introduces caller-owned verification limits, preflight measurement, deterministic rejection stages, bounded archive handling, external-verifier budgets, cancellation safety, and worst-case valid benchmarks. It does not weaken canonical semantics to gain speed; when an object is within configured limits, its existing Series 16–23 result must remain unchanged.

## Security invariants

1. **Verifier-owned policy:** limits come from trusted local configuration, never from the artifact being verified.
2. **Cheap before expensive:** byte, count, depth, uniqueness, and declared-length checks precede signature verification and deep lineage work.
3. **No partial acceptance:** timeout, cancellation, limit exhaustion, subprocess failure, or archive error cannot leave a persisted accepted result.
4. **Bound every dimension:** raw bytes, decoded objects, nesting, strings, collections, lineage, events, signatures, mirrors, conflicts, vectors, files, expansion ratio, and subprocess output all have explicit limits.
5. **Deterministic failure:** the same artifact and limit policy produce the same earliest failure stage and code.
6. **Streaming where possible:** hashing and archive validation do not require loading whole public kits into memory.
7. **Filesystem confinement:** extraction rejects absolute paths, parent traversal, symlink escapes, hardlink escapes, devices, FIFOs, and undeclared files.
8. **External-call accounting:** duplicate identities are rejected before verifier calls; every external authentication attempt consumes a visible budget.
9. **Cache safety:** cached results bind exact bytes, schema version, expected policy, limit policy, verifier identity/version, and relevant lineage context.
10. **Valid worst case remains usable:** limits are tested against explicitly sized valid reference bundles, not chosen solely from malformed inputs.

## Deliverables

- `VerificationLimits` and preflight measurement APIs owned by the caller.
- Stable `LimitExceeded` dimensions and failure-stage integration.
- Bounded decoders and collection builders.
- Signature, witness, mirror, conflict, and lineage work budgets.
- Safe streaming archive verifier/extractor.
- Cancellation-safe and transactional verification workflow.
- Frozen malicious-artifact corpus and fuzz seeds.
- Worst-case-valid benchmarks and recommended deployment profiles.
- Operator-visible resource report in offline verification kits.

## Explicit non-goals

Series 24 does not add network transport, distributed rate limiting, remote attestation, consensus, universal hardware budgets, or claims that one default profile is appropriate for every deployment.
