# Signed post-release checkpoint two

Verifies one signed runtime checkpoint that commits the exact authenticated post-release live observation from #3565.

The bridge reuses `RuntimeMeasurementCheckpoint::canonical_unsigned_bytes()` and the frozen runtime-policy trust semantics: Ed25519 signature validity, exact key ID/public key match, `Checkpoint` scope, and key validity/revocation window at the checkpoint's observation-time field.

Qualification additionally requires sequence 2, the exact checkpoint-one predecessor digest, strict counter agreement with the authenticated observation, exact runtime process/static identity, and `dynamic_measurement_digest == authenticated_post_release_observation.qualification_digest()`.

This theorem proves that an authorized runtime checkpoint signature commits the full release -> supervisor challenge -> tracee consumption -> live mapped-observation chain. It does not prove when the signature operation physically occurred, checkpoint-gap/time-regression policy, checkpoint-one live provenance, between-checkpoint mapping continuity, trusted time, key non-compromise, global replay exclusion, or physical authority. The complete existing runtime-continuity verifier remains the independent owner of full-trace sequence/gap/computation checks.
