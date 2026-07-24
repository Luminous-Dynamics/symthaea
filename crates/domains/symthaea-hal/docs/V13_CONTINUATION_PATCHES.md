# Hardened v13 continuation patches

The v13 continuation begins from the exact hardened-v12 tree `0cc091f3c625141bbc823f9f7a7844f3eb9cd95b`.

| Patch | Purpose |
|---|---|
| 0157 | Measurable recovery objectives and objective reports |
| 0158 | Signed recovery-drill plans and reports |
| 0159 | Deterministic evidence-replay contracts |
| 0160 | Durable attestation challenge anti-replay state |
| 0161 | Metadata-only authority escrow exercises |
| 0162 | Evidence-backed corrective-action closure |
| 0163 | Structured signed incident postmortems |
| 0164 | Durable phased return-to-service state |
| 0165 | Bounded fleet rejoin waves |
| 0166 | Portable recovery-assurance bundle |
| 0167 | Accumulated post-recovery admission gates |
| 0168 | Offline recovery-drill verifier |
| 0169 | Offline deterministic replay verifier |
| 0170 | Complete post-recovery admission CLI |
| 0171 | Production policy and evidence examples |
| 0172 | Future-time, corrective-action, and rejoin fail-closed closure |
| 0173 | Operating and migration documentation |

## Compatibility

The existing v1-v12 artifact domains and serialized structures are unchanged. v13 introduces new domains rather than silently changing old evidence identities. The only affected API is the not-yet-released v13 `RecoveryAssuranceInputs`, which takes a validated `AuthorityEscrowExercise` reference instead of an unverified digest so bundle assembly can prove that the escrow threshold actually passed.

## Migration order

1. Provision the recovery-objective and post-recovery admission policies.
2. Create the durable attestation nonce journal before issuing re-attestation challenges.
3. Establish metadata-only authority escrow and run a non-production exercise.
4. Generate a signed recovery drill and deterministic replay manifest.
5. Complete the postmortem and corrective-action closure.
6. Assemble the recovery-assurance bundle.
7. Run `hal-post-recovery-admission`.
8. Issue one return-to-service permit per phase transition.
9. Run individually accountable fleet rejoin waves.

No v13 artifact should be used as a substitute for hardware interlocks or direct output-power removal.
