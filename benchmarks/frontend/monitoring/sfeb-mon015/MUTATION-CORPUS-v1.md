# MON-015 deterministic mutation corpus v1

The 18 cases below are one-variable mutations against a known-good semantic fixture. The expected classification is authored as a test oracle, not derived from the code being tested. These are semantic adapter tests over normalized inputs; they do not replace upstream signature or receipt vectors.

| ID | Mutation | Expected result |
|---|---|---|
| M01 | Change producer/sink resource scope | contradicted |
| M02 | Change destination operation ID | contradicted |
| M03 | Change sink request digest under identical context | contradicted |
| M04 | Retain same hex but alter request profile | indeterminate (incomparable) |
| M05 | Remove response digest from AAC confirmed | effect assessment invalid; derived mode cannot remain confirmed |
| M06 | Claim confirmed mode for dispatched effect | invalid overclaim |
| M07 | Use sink writer principal as observer principal | contradicted |
| M08 | Mark observer control root shared/unknown | observed, but not independent |
| M09 | Change observer revision | contradicted |
| M10 | Change observer state digest | contradicted |
| M11 | Authority was revoked/invalid at commit | observed effect, authority invalid; never qualified |
| M12 | Commit timestamp/authority proof unavailable | observed effect, authority uncheckable; never qualified |
| M13 | Producer confirms while sink refuses | contradicted |
| M14 | AAC timeout paired with confirmed effect mode | orthogonality failure |
| M15 | AAC errored paired with not-applicable | orthogonality failure |
| M16 | OAR execution follows denied root decision | contradicted |
| M17 | OAR execution lacks confirmation | executed-unconfirmed, never confirmed |
| M18 | OAR reference crosses run identity | contradicted |

## Mutation-score protocol

- Publish all mutation IDs, expected verdicts, actual verdicts and implementation commit.
- Count a mutant as killed only when the test emits the expected non-success classification for the right reason.
- Do not count a parser crash, panic, timeout or test harness failure as a successful kill unless the protocol defines that exact outcome in advance.
- Add 20% independent holdouts not visible to the implementation author; report false accepts and false rejects separately.
- Keep mutations in at least four groups: identity/scope, digest/profile, authority/time, and lifecycle/effect semantics.
- Mutation corpus size does not equal security assurance. Independent review, realistic production traces and concurrent system tests remain mandatory.
