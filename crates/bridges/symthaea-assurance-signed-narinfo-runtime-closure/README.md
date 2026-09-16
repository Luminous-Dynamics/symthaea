# Signed narinfo runtime closure

Independently verifies Nix binary-cache Ed25519 signatures over the canonical `ValidPathInfo` fingerprint (`1;StorePath;NarHash;NarSize;References`) for every object in a previously qualified runtime closure, then requires the complete signed path/NAR/reference graph to equal the local #3245 closure graph exactly.

The verifier first re-derives #3245's public closure vector into the exact opaque closure digest, path count and total NAR bytes before using it. Signature thresholds count distinct reviewed key names. The signed claim covers store path, NAR hash, NAR size and references only; content-address metadata, cache transport fields and freshness/currentness are not promoted into the signature theorem.

This establishes non-equivocation of the local graph against one exact reviewed signed capsule. It does not establish that the capsule is globally latest/current, that the Nix DB is globally current, atomic loader/runtime pinning, trusted time, root-resistant immutability, or physical authority.
