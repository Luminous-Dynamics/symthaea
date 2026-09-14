# Current Linux TPM + IMA anchor

This bridge upgrades a ledger-relative Linux TPM+IMA acceptance capability only when the complete supplied acceptance ledger exactly matches an opaque `CurrentAcceptanceLedgerHead`.

The accepted record must still exist at the exact historical revision with the exact acceptance digest and predecessor binding. The current ledger revision/head/state digest, currentness-authority policy, ledger ID, and exact use-time are all committed into a new non-serializable capability.

This establishes authority-asserted current ledger membership at one exact challenged use-time. It does not establish trusted wall-clock provenance, authenticated persistence of the currentness tracker/authority state, readiness, or physical authority.
