# Replicator Safety Kernel — Semantic Schema Transition Test Plan v0.1

Status: **normative verification plan; authored evidence only until executed**

This document defines adversarial tests for capability-schema and resource-accounting semantic integrity.

It contains no physical replication mechanism or physical resource recipe.

---

## 1. Goal

Prove that a software/schema upgrade cannot widen replication authority merely by changing what existing bytes or numbers mean.

The central properties are:

```text
same bytes + different schema != same authority
```

and:

```text
translated_remaining_authority <= prior_remaining_authority
```

---

## 2. CS-ID — capability schema identity

Required cases:

- exact canonical schema produces stable expected digest;
- one semantic-description change produces different schema digest;
- one bit-index change produces different digest;
- bit-width change produces different digest;
- reserved/retired-bit change produces different digest;
- human version string unchanged while content changes -> digest changes and old authority does not transfer;
- same digest claim with different canonical schema bytes -> verification failure.

---

## 3. CS-BITS — capability bit validation

Required cases:

- valid defined bit accepted under exact schema;
- same bits under different schema rejected;
- unknown bit rejected;
- reserved bit rejected;
- retired bit rejected for new authority;
- bit outside schema width rejected;
- zero/empty set behaves according to policy without inventing capability;
- duplicate semantic identifiers in one schema rejected;
- two meanings assigned to one bit rejected.

---

## 4. CS-BIND — capability object binding

Generate mismatches among:

- lineage hard policy;
- parent ceiling;
- grant;
- action request;
- evaluated authorization;
- committed event;
- checkpoint;
- runtime/admitted-release schema.

Every mismatch must deny new positive authority.

A durable event with valid bits but missing schema ID is invalid authority evidence.

---

## 5. CS-TRANS — cross-schema capability translation

Required cases:

- no translation object -> reject cross-schema comparison;
- verified exact-preserving translation -> may pass;
- one-to-one mapping to broader target meaning -> reject;
- one source bit split into two independently usable target bits -> reject unless a conservative conjunction/partition proof exists;
- two source bits merge into one broader target bit -> reject unless attenuation is proven;
- translation missing one active source capability -> reject unless policy proves removal only;
- translation creates target capability absent from source -> reject;
- stale/revoked translation policy -> reject;
- translation valid for other source/target schema pair -> reject.

Property generator:

```text
Meaning(translate(S)) ⊆ ConservativeMeaning(S)
```

for every generated source set `S` within the bounded schema model.

---

## 6. CS-UPGRADE — capability upgrade/rollback

Histories:

```text
schema A -> grant A -> upgrade software to schema B -> use old grant
```

Expected: denied unless verified transition/requalification exists.

Also test:

- A -> B -> rollback binary to A while durable state is B;
- deprecated A remains readable but not authority-bearing;
- fresh epoch after incompatible transition carries no old capability authority;
- recovery cannot reinterpret A bits as B bits;
- admitted release expects B but runtime registry loads A -> deny.

---

## 7. RA-ID — resource scheme identity

Required cases:

- exact scheme canonical encoding/digest stable;
- unit identifier change changes digest;
- scale change changes digest;
- required-dimension set change changes digest;
- rounding policy change changes digest;
- aggregation policy change changes digest;
- numeric representation/bounds change changes digest;
- human version reused with changed content does not preserve authority identity.

---

## 8. RA-VECTOR — resource vector validation

Required cases:

- exact complete dimension vector accepted;
- missing required dimension rejected;
- unknown active dimension rejected;
- duplicate dimension rejected;
- invalid scale rejected;
- non-canonical numeric representation rejected;
- consumed greater than limit rejected;
- checked-add overflow rejected;
- checked-sub underflow rejected;
- zero limits behave exactly and do not wrap;
- maximum representable boundary behaves without saturation widening.

---

## 9. RA-BIND — resource object binding

Generate exact-schema mismatches among:

- lineage hard policy;
- ancestor/subtree budget;
- grant;
- request;
- authorization;
- commit;
- checkpoint;
- safety-case/admitted-release/runtime scheme.

Every semantic mismatch must fail closed before state mutation.

---

## 10. RA-MONO — monotonic consumption

For every dimension and every applicable ancestor scope:

```text
consumed_after >= consumed_before
remaining_after <= remaining_before
```

unless entering a fresh epoch with no implicit authority carryover.

Generated histories must include:

- descendants at multiple depths;
- multiple grant generations;
- partial budget exhaustion;
- failed actions;
- replayed events;
- stale cursor attempts;
- branch/lineage revocation;
- checkpoint/restart.

Failed actions must not consume budget unless the contract explicitly defines a committed cost event; they must never increase remaining authority.

---

## 11. RA-GEN — generation laundering

Required cases:

- generation N consumption followed by generation N+1 grant under same scheme cannot reset ancestor counters;
- new generation with same numeric limit but new incompatible scheme rejected;
- generation change cannot drop required dimension;
- generation change cannot increase remaining authority through re-denomination;
- stale generation cannot reapply old lower consumed state;
- rollback to old generation rejected.

---

## 12. RA-TRANS — cross-scheme resource translation

Required cases:

- absent translation -> reject;
- exact identity translation -> may pass;
- denomination/scale conversion with conservative exact arithmetic -> may pass;
- conversion rounds remaining allowance upward -> reject;
- conversion rounds consumed amount downward -> reject;
- conversion drops constrained dimension -> reject by default;
- conversion duplicates remaining budget into two target dimensions -> reject;
- merge loses stricter source constraint -> reject;
- translation proof/evidence stale/revoked -> reject;
- wrong source/target scheme pair -> reject.

---

## 13. RA-REMAIN — translate remaining authority, not totals independently

Adversarial examples must specifically catch the bug class:

```text
translate(limit) - translate(consumed)
    > conservative_translate(limit - consumed)
```

The production transition path must use the conservative remaining-authority rule.

Generated property:

```text
new_remaining <= ConservativeImage(old_remaining)
```

for every dimension and ancestor scope.

---

## 14. RA-DROP — dimension removal

Required cases:

- drop dimension with nonzero remaining authority -> reject;
- drop fully exhausted dimension under policy permitting retirement -> may pass;
- embed old constraint conservatively into target dimension -> may pass only with verified proof;
- omit dimension because current action does not use it -> reject if ancestor policy still requires it;
- fresh epoch explicitly establishes new scheme -> old resource authority does not cross implicitly.

---

## 15. RA-UNCERT — measurement uncertainty

When consumption is represented by an interval:

```text
[min_consumed, max_consumed]
```

eligibility must use `max_consumed` for safety.

Test:

- exact measurement below limit -> eligible if other predicates pass;
- uncertainty interval crosses limit -> deny;
- uncertainty widened by stale calibration -> deny when bound no longer proves budget;
- authenticated but unqualified measurement -> deny;
- measurement source common-mode failure -> fail according to policy;
- favorable midpoint selection is never used to extend authority.

---

## 16. RA-CHECKPOINT — replay/checkpoint equivalence

Required cases:

- checkpoint exactly equals replay-from-genesis resource state;
- checkpoint decrements consumed value -> reject;
- checkpoint changes scheme ID -> reject absent migration evidence;
- checkpoint changes scale/dimension set -> reject;
- checkpoint omits ancestor scope -> reject;
- checkpoint after verified migration agrees with transition record;
- stale checkpoint from old generation/epoch rejected.

---

## 17. ST-CROSS — combined capability/resource transition

A release may change both schemas simultaneously. Test combined failures so one valid migration cannot hide the other invalid migration.

Cases:

- valid capability translation + invalid resource translation -> deny;
- invalid capability translation + valid resource translation -> deny;
- both valid -> may proceed to remaining constitutional gates;
- one schema unchanged, other changed -> changed side still requires transition evidence;
- release/runtime schema tuple mismatch -> deny.

---

## 18. ST-ACTION — exact action binding across schema transitions

The action evaluated must remain exactly the action committed under the same semantic schema tuple.

Required cases:

- authorization evaluated under capability A/resource R, commit under capability B/resource R -> reject;
- capability A/resource R -> capability A/resource S -> reject;
- same schema IDs but different requested resource vector -> reject;
- same bits/numbers but changed schema IDs -> reject;
- translation after authorization but before commit -> old authorization invalid unless explicitly re-evaluated.

---

## 19. ST-RECOVERY — recovery/fresh epoch

Recovery tests:

- same-schema recovery with exact replay preserves state but grants no extra authority;
- verified conservative migration preserves/reduces authority;
- incompatible transition requires fresh epoch;
- fresh epoch receives no old capability/resource positive authority implicitly;
- old grants/authorizations cannot be replayed into new epoch;
- recovery authority cannot mint migration proof by self-assertion.

---

## 20. ST-DURABLE — durable encoding/golden vectors

Golden vectors freeze:

- capability schema bytes/digest;
- bound capability-set encoding;
- resource scheme bytes/digest;
- resource vector encoding;
- translation evidence identifiers;
- representative durable grant/action/commit bindings.

Mutation/fuzz tests must ensure:

- duplicate keys/fields rejected where canonical encoding forbids them;
- integer overflow rejected;
- unknown critical fields/schema versions fail closed;
- reordered/noncanonical encodings cannot create a second authority identity for the same logical record unless canonicalized deterministically.

---

## 21. ST-API — compile/type misuse

Production APIs SHOULD make invalid cross-schema arithmetic hard to express.

Compile-fail or equivalent tests should prove that callers cannot directly:

- intersect raw bitsets from different schemas;
- compare resource vectors from different schemes;
- convert raw schema bytes to verified translation capability;
- deserialize `VerifiedCapabilityTranslation` or `VerifiedResourceTranslation` as trusted state;
- omit scheme IDs from capability/resource authority objects.

---

## 22. ST-PROP — generated transition histories

Generate arbitrary sequences over:

```text
issue_grant
consume
spawn_descendant
new_generation
upgrade_schema
translate
checkpoint
restart
rollback
revoke
recover
fresh_epoch
```

Target properties:

```text
SemanticCapabilityNeverWidensAcrossTransition
RemainingResourceAuthorityNeverIncreasesAcrossTransition
UnknownSchemaCannotAuthorize
MissingDimensionCannotAuthorize
GrantGenerationCannotLaunderBudgets
RecoveryCarriesNoSemanticAuthorityImplicitly
ExactActionSchemaTupleIsCommitBound
```

---

## 23. ST-FORMAL — formal refinement targets

Extend the bounded RSK model with abstract schema IDs and translation relations.

Candidate invariants:

```text
CapabilitySchemaMismatchCannotAuthorize
ResourceSchemeMismatchCannotAuthorize
CapabilityTranslationIsAttenuating
ResourceTranslationIsAttenuating
GenerationChangeCannotResetConsumption
SchemaTransitionCannotIncreaseAuthority
RecoveryCannotReinterpretOldState
```

The formal model should distinguish “same numeric state” from “same semantic state.”

---

## 24. Current status

These tests are authored targets only. This tranche does not implement the production schema registry, migration verifier, resource meter, or runtime schema binding.

Production admission remains **DENIED / NOT YET ELIGIBLE**.