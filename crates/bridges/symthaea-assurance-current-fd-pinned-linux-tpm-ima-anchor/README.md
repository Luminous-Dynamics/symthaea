# Current FD-pinned Linux TPM + IMA anchor

This bridge upgrades an opaque `CurrentLinuxTpmImaRuntimeAnchor` only when the quote identity embedded in that provider capability exactly matches an opaque `VerifiedFdPinnedTpm2Quote`.

The composition binds the current-ledger provider policy/qualification, FD-pinned execution policy/qualification, quote qualification, challenge, AK, raw quote artifact, verification receipt, PCR selection, and exact executed-file identity into one non-serializable capability.

It establishes that the TPM quote used by the authority-current TPM+IMA provider chain was verified through the file-descriptor-pinned execution theorem. It does not atomically pin the dynamic dependency closure, establish root-resistant Nix-store immutability, trusted time, readiness, or physical authority.
