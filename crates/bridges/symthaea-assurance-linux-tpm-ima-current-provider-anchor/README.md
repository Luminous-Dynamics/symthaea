# Current Linux TPM2 + IMA provider anchor

This bridge promotes already-qualified provider evidence only when three opaque capabilities agree:

1. a ledger-relative Linux TPM+IMA acceptance anchor;
2. the exact verified TPM↔IMA PCR binding consumed by that acceptance; and
3. a challenge-current authoritative acceptance-ledger head.

The provider policy pins the exact ledger-relative anchor policy, PCR-binding policy, ledger ID and ledger-head authority policy. It also bounds quote age and acceptance age at the current-head use-time.

The output `CurrentLinuxTpmImaProviderAnchor` is intentionally distinct from continuous runtime integrity. It establishes a current, freshness-bounded measured-runtime provider anchor under the reviewed authority/time assumptions; it does not replace verifier runtime hash-link continuity and grants no physical authority.
