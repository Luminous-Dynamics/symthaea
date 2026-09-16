# Signed narinfo closure current head

Adds authority-head currentness above the independently verified signed-narinfo closure theorem.

The theorem deliberately separates three claims:

1. **head authenticity** — a reviewed scoped Ed25519 authority signs the exact opaque signed-closure qualification/capsule/graph/local-closure/root identity;
2. **observed head continuity** — an in-memory strict tracker starts at authority sequence 1 and rejects rollback, collision, gaps, broken predecessor links, issuance-time regression, and silent authority-policy replacement; and
3. **exact-use currentness** — a currentness-authorized signature answers one exact nonzero caller challenge for the tracker's still-latest head at one exact use-time that cannot predate the signed head itself.

The authority policy must provide both head-statement and currentness-attestation scope. V1 freezes one exact authority-policy digest for each tracked subject lineage. Rotating that authority policy requires a separate explicit successor theorem; it cannot be smuggled through the ordinary closure-head sequence.

The reviewed root path is validated with the same Nix store-path hash alphabet/name grammar used by the signed-narinfo theorem. This keeps policy identity from admitting a looser spelling than the closure capability it is meant to govern.

This prevents an older valid signed capsule from remaining current after this tracker has observed an authority successor. It establishes authority-asserted currentness for one exact reviewed closure profile/root, not global cache latestness.

The local tracker is not authenticated durable anti-rollback state. A restarted or maliciously rolled-back tracker remains outside this theorem. Likewise, authority compromise, trusted wall-clock provenance, global Nix/cache currentness, atomic dependency pinning, root-resistant immutability and physical authority remain explicit non-claims.
