# Vocal edit patch contract

Standalone Rust contract for renderer-neutral, content-bound vocal edits.

## Why this exists

A local word/phone/note/phrase correction must not be an unversioned command to regenerate an entire song. The request must identify the exact source performance, the object being edited, the dependency cone, the renderer/voice artifacts, and the authorization for the change.

The contract is intentionally separate from the v2 VocalPerformance API and from any particular synthesizer. It is the admission/receipt boundary that a future renderer adapter can implement.

## Request model

A VocalEditPatch binds:
- schema and stable patch ID;
- BLAKE3 digests of pronunciation plan, score/alignment, source rendered audio, renderer capability manifest and voice artifact;
- one stable target graph node;
- a typed operation (candidate selection, phone replacement, score pitch/velocity, melisma nucleus, phrase breath, tempo map or voice artifact);
- a declared dependency cone and sample-clock windows;
- sample rate and total frame count;
- an authorization authority and opaque evidence reference.

The patch has an exact, domain-separated BLAKE3 digest over its validated JSON representation. A renderer must reject replay when any base digest has changed.

## Dependency cone

Local scope requires:
- the target node included in the affected-node set;
- unique affected/protected node sets with no intersection;
- one exact target window and one wider render window;
- context-guard counts equal to the actual interval differences;
- at least one protected node.

Tempo-map and voice-artifact changes require whole-stem scope. A bounded local operation may still request whole-stem rendering when a backend cannot localize it, but the receipt must say whole_stem; it cannot claim locality.

## Receipt

A VocalEditReceipt binds the exact patch digest, source artifact set, output pronunciation/score/audio digests, renderer identity and actual render scope.

A bounded-local receipt must:
- use the exact declared render window;
- cover exactly the declared protected graph nodes;
- include protected-region before/after digests and exact-sample/RMS measurements;
- reference a VELB locality report and a passing seam report.

A whole-stem receipt must not include protected-region or bounded-locality claims.

verify_against checks structural binding and report references. It does not independently open audio files or recalculate VELB/seam metrics; the content-addressed verifier must inspect those reports separately.

## Limits

- This is a schema/validation prototype, not an edit engine.
- It does not resolve G2P, choose OOV pronunciations, derive a dependency cone from the graph, synthesize or splice audio, or evaluate seams.
- Stable target IDs must come from the eventual v3 graph adapter; array indices and display labels are not suitable identities.
- A local receipt still requires VELB and seam evidence; a symbolically valid patch is not proof of a successful edit.

## Tests

Synthetic tests cover valid local patch/receipt verification, stale plan/score/audio/renderer/voice digests, missing target in dependency cone, affected/protected overlap, inconsistent context windows, operation-target mismatch, malformed operation values, global-operation escalation, stale receipt binding, missing preserved-node evidence, failed seam report, and patch-digest sensitivity.
