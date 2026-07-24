# Checkpoint Federation Lifecycle Campaign

**Program:** Symthaea vocal-tract checkpoint trust

**Lifecycle evidence schema:** `symthaea.checkpoint-power-loss-federation-lifecycle-evidence.v1`

**Operational evidence schema:** `symthaea.checkpoint-operational-evidence.v9`

## Purpose

Series 18 authenticates one federation epoch. It does not define how a campaign
moves to a new authority key, how a laboratory rotates its evidence key or
changes administration, how the epoch history resists rollback, or what happens
to promotion evidence accepted under the prior epoch.

Series 19 makes every one of those changes explicit. A rollover is accepted only
when the prior and next federation authorities endorse the same transition,
affected laboratories endorse their own succession, the transition extends the
exact prior epoch-ledger head, and every prior promotion merge receives a signed
retained, superseded, or revoked disposition.

## Preregistered transition

Before the next epoch becomes active, publish one transition containing:

- nonzero transition identifier and sequence;
- stable federation identifier;
- consecutive prior and next epochs;
- exact prior and next campaign digests;
- exact prior and next operations-plan digests;
- exact prior and next federation-plan digests;
- prior and next federation authority key identifiers;
- exact prior epoch-ledger head digest;
- a contiguous handoff time, where the next plan begins one second after the
  prior plan ends;
- nonzero reason digest; and
- the complete, deterministically derived member-transition list.

The member list distinguishes retained members, evidence-key rotations,
delegated administration changes, additions, and removals. A caller cannot omit
or reorder a member change without changing the transition digest.

## Dual federation-authority lane

The prior authority signs the transition digest as `PriorAuthority`; the next
authority signs the same digest as `NextAuthority`. Both endorsements must be
created no later than the transition effective time.

Negative controls must reject:

1. prior-only endorsement;
2. next-only endorsement;
3. swapped roles;
4. endorsement under an unrelated key;
5. different transition digests;
6. endorsement after the handoff time; and
7. nonconsecutive epochs or a validity gap/overlap.

## Laboratory succession lane

Every evidence-key rotation or delegated administration change requires one
succession record. The outgoing and incoming lab evidence keys sign the exact
same succession digest.

The record binds:

- transition digest;
- monotonically ordered succession sequence;
- lab identity;
- prior and next member digests;
- prior and next evidence-key identifiers;
- effective time; and
- nonzero reason digest.

Added and removed members remain governed by the dual federation transition.
A retained member needs no succession record. Promotion fails if a required
succession is absent, duplicated, out of order, or signed by the wrong key.

## Epoch-ledger lane

The next authority publishes an authenticated epoch ledger. Each entry binds:

- transition identifier and digest;
- consecutive prior and next epochs;
- prior and next federation-plan digests;
- exact previous ledger-head digest; and
- recording time.

The transition itself includes the expected previous head. Appending the
transition is therefore impossible against a different history without
invalidating both federation endorsements. The final ledger entry must be the
transition under evaluation, and the ledger must be sealed by the current
federation authority.

## Prior-evidence disposition lane

Every authenticated Series 18 merge from the prior epoch receives exactly one
of three dispositions:

- `Retained` — the prior evidence remains independently valid;
- `Superseded` — the record names the exact current-epoch replacement merge;
- `Revoked` — the prior evidence must not be used for promotion.

The prior and next federation authorities sign every disposition. Duplicate or
missing decisions fail closed. A superseded decision whose replacement digest
does not equal the current merge is rejected.

## Artifact layout

The strict evaluator consumes:

- prior campaign and operations-plan artifacts;
- prior sealed federation plan and private federation key;
- prior lab-key directory;
- directory of authenticated prior federation-merge artifacts;
- next campaign and operations-plan artifacts;
- next sealed federation plan and private federation key;
- next lab-key directory;
- current federation-merge artifact; and
- next-authority-sealed lifecycle evidence bundle containing an authenticated
  receipt for that exact merge digest, next federation-plan digest, federation
  identifier, and epoch.

Private key files are exactly 48 bytes (`key-id || key`), private to the current
user, regular files, and opened without following symlinks.

## Promotion gates

Series 19 passes only when:

- the epoch transition is structurally valid and dual-endorsed;
- the federation authority key actually rotates;
- every required lab succession has both lab endorsements;
- the authenticated epoch ledger ends at the new transition and extends the
  exact prior head;
- every prior merge has exactly one authenticated disposition;
- all superseded evidence points to the exact current merge;
- the next authority authenticates the exact current merge digest, next
  federation-plan digest, federation identifier, and epoch in a merge receipt;
- the current merge belongs to the next plan and epoch; and
- all Series 19 gates are derived from authenticated lifecycle evidence rather
  than hand-populated booleans.

Missing lifecycle artifacts remain `not_exercised`. A valid Series 18 federation
merge alone cannot satisfy Series 19.

## Non-claims

The lifecycle uses symmetric keyed authentication because that is the existing
crate contract. It does not provide public nonrepudiation. A verifier holding a
secret can also produce an endorsement. Public or adversarial third-party
verification requires asymmetric signatures or hardware-backed verification
keys.

The epoch ledger prevents an endorsed transition from being attached to a
different prior head, but external monotonic publication is still needed to
prevent all parties holding the current authority key from withholding a newer
ledger. Series 19 does not claim legal continuity of organizations from digest
bindings alone; governance must independently establish who controls each key
and administration binding.
