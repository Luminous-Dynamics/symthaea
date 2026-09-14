# symthaea-evidence-lifecycle-head-currentness

ASSURE-015E closes the withheld-lifecycle-head gap above the authenticated verifier-profile lifecycle stack.

An internally valid lifecycle ledger is not evidence that it is the latest authoritative ledger. This crate therefore separates three claims:

1. a signed lifecycle-head statement binds one exact profile-record digest to one exact lifecycle-ledger digest and monotonically sequenced head;
2. a challenge-bound currentness attestation says that exact signed head is the authority's current head for one exact use-time and nonce;
3. a local tracker rejects rollback, skipped sequence, conflicting same-sequence heads, broken previous-head links, and regressed issuance time across observed heads.

Only an exact active profile, signed head statement, authorized currentness attestation, expected challenge nonce, and exact use-time can mint `CurrentActiveVerifierProfile`.

This is still evidence-plane machinery. It does not establish trusted time, global truth of the authority's assertion, verifier independence by itself, readiness, or physical authority.
