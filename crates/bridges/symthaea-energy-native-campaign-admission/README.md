# symthaea-energy-native-campaign-admission

Admits a complete Tier-1 campaign assembled entirely from native candidate/campaign/lane-bound evidence envelopes.

This is the path for newly generated evidence that no longer needs #1963 compatibility `ReceiptVersionAttestation`s.

## Required native chain

Admission starts from:

1. frozen Tier-1 campaign manifest;
2. native envelopes bound to that campaign/candidate/lane and exact generic predictions;
3. #1942 external identity assertions bound to exact envelope digests;
4. #1994 native dossier assembled mechanically from those envelopes.

The native dossier is fully revalidated before admission.

## Frozen-plan checks

For every one of the seven Tier-1 lanes admission requires:

- exact native envelope binding to the frozen campaign;
- metric/unit matching the screening contract;
- fidelity at or above the frozen policy minimum;
- exact expected model name/version;
- every required evidence kind;
- pinned/internal source digest present in prediction provenance, or a matching prospective acquisition declaration.

For prospective acquisition, the declaration must bind:

- the exact preregistered acquisition-query SHA-256;
- acquired-artifact SHA-256;
- the exact native-envelope SHA-256;
- reviewer + review note.

The acquired artifact digest must be present in the generic prediction provenance.

## Negative results remain first-class

Admission deliberately does **not** require the candidate to be feasible.

A complete campaign that shows the candidate violates a hard constraint is still admissible scientific evidence. Admission means the frozen evidence plan was completely recorded at the checks implemented here; it does not mean the candidate passed screening.

## Relationship to #1974

This crate mirrors the substantive campaign checks currently implemented in #1974's compatibility admission path.

Those validators are private in the current campaign crate, so v0 keeps a small native checker rather than rewriting the large parent source file while CI is unavailable. Once the stack is execution-qualified, the shared lane/source validators should be extracted into one lower-level helper so compatibility and native admission cannot drift.

## Network-free CLI

`energy-native-campaign-admission <input.json>`

Input contains:

- frozen campaign manifest;
- complete native dossier;
- optional prospective acquisition declarations.

Output contains the native admission receipt + SHA-256. Host-local input paths are omitted.

## Authority boundary

Native campaign admission is evidence-plan conformance metadata. It is not novelty, experimental truth, scientific replication, candidate promotion, safety certification, synthesis authorization, manufacturing approval, investment approval, or deployment authority.
