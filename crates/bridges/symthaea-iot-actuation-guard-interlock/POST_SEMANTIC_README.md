# Post-Semantic Interlock Proof

The corrected guard path reuses the existing `GuardInterlockState` policy/trust TCB but verifies the stronger post-semantic controller statement directly.

`VerifiedPostSemanticPhysicalInterlock` means only that:

- the exact post-semantic challenge/report correlation was rechecked;
- the controller statement was fresh at guard-local relying-party time;
- the exact controller was allowed by guard-owned `PhysicalInterlockPolicyV1`;
- the asserted interlock set and report lifetime matched that policy;
- the report did not predate the current controller-trust generation; and
- the exact stronger statement digest + raw evidence were verified under the current anti-rollback controller key using the fixed RFC 8032 Ed25519 profile.

It is deliberately **not** a legacy `VerifiedPhysicalInterlock`, final permit, JIT lease, or HAL authority. The new controller signature authenticates a stronger/different digest than the historical legacy report, so coercing the proof into the old type would misstate what was signed.

Current device semantic state and Xenia transport trust must be re-fenced again during corrected final/JIT composition before any hardware attempt.
