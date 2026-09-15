# Current in-process Linux TPM + IMA anchor

Upgrades the authority-current Linux TPM+IMA provider only when its exact quote identity is the same quote verified by the in-process ECDSA P-256/SHA-256 verifier.

The composition binds the exact current-provider policy/qualification, in-process policy/qualification, inherited raw-quote qualification, challenge, AK, quote artifact, verification receipt, PCR selection, running-host executable backing-file identity, cryptographic backend identity, and authority-current ledger state into one non-serializable capability.

For the supported profile the terminal provider no longer depends on an external `tpm2_checkquote` process. The resulting capability still does not establish mapped-memory immutability, atomic shared-library identity, resistance to privileged in-place mutation, trusted time, readiness, or physical authority.
