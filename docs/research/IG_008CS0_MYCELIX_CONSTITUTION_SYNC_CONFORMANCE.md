# IG-008CS0 — Mycelix constitution-to-runtime synchronization conformance

Issue: #3388

Parent: composite-v4 / draft #3364.

## Purpose

Independently reconstruct the frozen Mycelix constitution→runtime governance-config synchronization counterexamples before this edge can participate in a constitutional-downstream composite.

This tranche establishes only `MeasurementOnly / CrossImplementationConformance`.

## Frozen Mycelix evidence

Exact evidence head:

`abb277cc39418e43e80d48e8a4a57bc7c939d9e4`

Profile:

```text
id             mycelix-constitution-bridge-sync-observed-fca2c107-v1
profile SHA    60daae86044098561fa8e41bcdf6f695ab41234760be4b7e235d2780271b681e
authority      ObservedSourceBound
```

Corpus:

`2ca6d79212cfd0acae1e974c8390d564bab06821050d8eacbc32f437630ef60b`

Issues: Mycelix #943 and #944.

Semantic production subject:

`fca2c107a1ea5108823ce617ba4111b6f7f77230`.

Tree-equivalent evidence source:

`31ede2365b81365bb119cd9351b2739119974130`.

## Independent oracle

`scripts/ig008cs0_mycelix_constitution_sync_oracle.py` is stdlib-only and imports none of Mycelix CS0/CS1 Python, Constitution/Bridge Rust, or Institutional Lab code.

It independently validates enough of the profile to reproduce exactly:

- CE-CS-01 — the called `update_phi_config` entrypoint is absent from the exact eight-module bridge coordinator census while `update_consciousness_config` is visible;
- CE-CS-02 — a successful ConstitutionParameter write does not establish runtime synchronization when the subsequent best-effort bridge sync fails;
- CE-CS-03 — mechanically renaming the target would cross into the separate #943 authorization theorem;
- CE-CS-04 — no content-bound synchronization/reconciliation evidence is established by the frozen helper.

The emitted canonical corpus must be byte-identical to Mycelix CS1.

## Qualification

The exact-head workflow:

1. checks out the exact Symthaea subject;
2. checks out Mycelix at exact CS1 evidence head `abb277cc...`;
3. binds the constitution coordinator plus all eight bridge coordinator source blobs;
4. independently checks the exact bridge module census, absence of `update_phi_config`, and presence of `update_consciousness_config`;
5. syntax-compiles Mycelix CS0/CS1 and independent Symthaea oracle;
6. revalidates Mycelix CS0;
7. runs Mycelix CS1 twice deterministically and emits its canonical corpus;
8. runs the Symthaea oracle twice deterministically and emits its canonical corpus;
9. requires byte-identical Mycelix/Symthaea corpus output;
10. asserts exact profile/corpus commitments, #943/#944, production subject, same-tree evidence head and authority ceilings;
11. verifies both working trees remain immutable.

## Constitutional downstream composition

This tranche alone is **not** complete constitutional-downstream coverage.

The full observed downstream slice requires three independently represented mechanisms:

```text
ConstitutionParameter
+ ConstitutionBridgeSync
+ GovernanceConfig
```

where:

- ConstitutionParameter is independently reproduced by Symthaea #3349;
- GovernanceConfig is independently reproduced by Symthaea #3269;
- ConstitutionBridgeSync is this tranche.

A later submanifest should bind all three before the top-level composite is allowed to remove a broad constitutional-downstream gap.

## v4 correction boundary

Composite-v4 / #3364 currently removes `constitution_parameter_authorization_downstream` after only ConstitutionParameter coverage. Treat that as an intermediate candidate, not final broad constitutional coverage. The correct successor should preserve v4 history and add a finer-grained downstream submanifest instead of rewriting v4.

## Non-claims

No live divergence, live mutation, deployment exploit, authorized runtime config, complete constitutional downstream, deployment currentness, governance safety, fairness, or constitutional legitimacy is established.
