# Nixward CROSS-068A — Authorization-Time Definition Content Capture V1

## Purpose

CROSS-060 bound a service-definition commitment into the authority chain and CROSS-067 made the observed FragmentPath/DropInPaths source identity self-verifying.

CROSS-068A adds the missing local content-commitment primitive: an observer can capture the exact bytes referenced by the current systemd definition identity and produce an observer-sealed commitment suitable for authorization.

## Source identity and content identity remain separate

The observer first captures:
- canonical target unit;
- systemd FragmentPath;
- systemd DropInPaths;
- systemd manager unique owner.

The existing definition-identity digest remains the source-identity commitment.

The content-evidence commitment separately binds:
- target unit;
- source-identity digest;
- systemd manager unique connection name;
- D-Bus daemon incarnation (GetId);
- source path;
- resolved path when explicit NixOS link resolution occurred;
- byte length;
- per-file BLAKE3 digest;
- ordered file-set membership.

The per-file BLAKE3 remains a pure byte-content digest. Capture time is retained as temporal evidence metadata but is deliberately outside content identity so the aggregate commitment can be independently recomputed from durable source/content metadata.

No raw definition bytes are placed in the portable authorization record.

## File capture boundary

On Unix, each source path is opened read-only with close-on-exec and trailing-link rejection.

The same opened file descriptor is then:
1. metadata-sampled;
2. streamed and hashed;
3. metadata-sampled again;
4. streamed and hashed again;
5. metadata-sampled finally.

A mutation in descriptor identity/metadata or a change in the two computed hashes/lengths fails closed.

This intentionally uses descriptor-based observation rather than a separate path-stat/read/stat sequence.

The current policy rejects a trailing symbolic link. Stronger containment against hostile intermediate path components remains a separate hardening seam.

`O_NOFOLLOW` therefore closes the final-component symlink case, but does not by itself eliminate symlink traversal in intermediate path components. A future filesystem-resolution tranche can strengthen that boundary where the host kernel supports it.

## systemd correlation boundary

The content observer captures the systemd manager unique owner before observation and rechecks it after the complete content read and after the final definition identity read.

It also requires systemd's `NeedDaemonReload` property to be false both before and after the capture. If systemd reports that a daemon reload is needed, the current filesystem bytes are not sufficient evidence for the definition currently considered loaded by systemd, so the content capture fails closed.

It also re-resolves the unit and requires the unit object path and FragmentPath/DropInPaths source identity to remain unchanged.

Therefore a manager rollover or observed definition-identity change cannot silently be reinterpreted as the same capture.

This is observation/commitment evidence, not cryptographic attestation of systemd.

## Sealed authorization input

The observer-sealed definition-content type has no public constructor, serialization, or cloning surface.

The observer creates it only after validation.

The service-effect context can be derived from that sealed token with from_verified_definition_content(...). The legacy Service authorization constructors now reject Service actions without the sealed capture; the capture-aware constructor recomputes the content commitment from the sealed evidence before creating the authorization record.

Consequently, an arbitrary caller-supplied string that merely looks like a BLAKE3 digest cannot satisfy the new Service authorization entry point.

## Claim ceiling

CROSS-068A does not prove that systemd loaded those exact bytes atomically with capture, nor does it eliminate all filesystem namespace or execution TOCTOU races.

It proves a narrower fact:

> the observer captured and independently committed the bytes of the systemd-reported source files across a checked descriptor-level observation interval, with the source identity and systemd manager incarnation rechecked around the capture.

The qualification result remains Unproven until exact-head Actions complete successfully. A queued, pending, skipped, or mergeable run is not evidence of PASS.

## Next tranche

CROSS-068B should propagate authorized_definition_content_digest into the post-state expectation/observation/stability/receipt grammar and make the final receipt verifier recompute the same content commitment from durable content metadata.

That should be a separate change so the content-capture boundary can be qualified independently before expanding the durable receipt schema.