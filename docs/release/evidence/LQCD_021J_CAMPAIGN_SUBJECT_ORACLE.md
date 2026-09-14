# LQCD-021J — staged Beta6CampaignSubject transition oracle

Independent standard-library execution freezing the staged whole-experiment identity semantics from #2748.

Exact executed subject SHA-256:

`69fde4f216129e45fe3b523977ea00d94630ca684b7ee8c51deec726430fcbf1`

Canonical result SHA-256:

`106da97ea807f77cc57f848e0383e10fd9acebddd24e9510d4ff5dfa6ed02571`

## Canonical subject encoding

Stable oracle ID: `beta6_campaign_subject_transition_oracle_v1`.

The subject uses a language-neutral typed TLV encoding with explicit tags for typed `Unbound`, null, boolean, unsigned integer, IEEE-754 binary64, UTF-8 string, list, and map. Map keys are ordered by their UTF-8 bytes. Subject bytes begin with `symthaea.lqcd.beta6-campaign-subject.v1\0`, a one-byte stage code, the exact 32-byte predecessor digest, then canonical typed payload bytes.

The encoding deliberately does not hash JSON floating formatting, Rust debug output, native memory layout, or insertion order.

## Staged theorem

The executed synthetic chain is:

`Template -> Pilot -> Final -> SealedAnalysis -> Comparison`.

Every child must name the exact parent digest and may advance exactly one stage. Stale predecessor substitution and stage skipping are rejected.

The template uses typed `Unbound` fields for throughput/pilot/final-derived choices. The pilot binds only pilot-authority fields while final-production choices remain `Unbound`. Final validation requires every scientific final field to be bound.

## Exact retained schedule

A synthetic four-chain final fixture derives exactly 4000 `(chain_id, retained_ordinal, update_ordinal)` slots. Its canonical schedule digest is:

`60bf219272a1dd1028e0cb4f63b5a60f304702b1a3d6fe5d8cc5d1ed5eac2744`.

Changing one allocation to produce 3999 slots fails closed. Changing final burn-in from 8000 to 8001 changes the final-campaign identity.

## Target-isolation composition

`SealedAnalysis` rejects `ehk_targets` or benchmark-delta fields. A later comparison may bind an external benchmark. Replacing that benchmark changes the comparison subject while the already-sealed analysis subject remains bit-identical.

## Important claim boundary

All numerical choices in the stage-transition fixture are synthetic qualification data. The result explicitly records:

`real_beta6_campaign_authorized = false`.

This oracle establishes canonicalization, transition, typed-Unbound and exact-schedule semantics only. It does not instantiate or authorize the real beta=6.0 campaign, and it does not upgrade any queued Rust dependency to executable-qualified status.
