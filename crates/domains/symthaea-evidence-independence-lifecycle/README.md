# symthaea-evidence-independence-lifecycle

Bitemporal lifecycle semantics for authenticated verifier-profile provenance.

The core rule is:

```text
ordinary supersession != historical falsification
corrective revocation/contradiction can invalidate current reliance
```

A profile that was validly authenticated before a verification event may remain historically usable after ordinary replacement. A later correction can still state that the historical attribution was wrong from an earlier effective time and block reliance on it.

This crate also adds anti-rollback tracking for provenance-policy snapshots. It does not itself provide trusted time, graph independence, runtime continuity, readiness, or physical authority.
