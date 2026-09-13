# symthaea-assurance-subject

ASSURE-001 defines exact multi-surface system-under-test identity for external AI and agent qualification.

Its central rule is that a product or provider label is not a sufficient subject identity:

```text
same product name
    != same model
    != same prompt
    != same authority
    != same runtime
    != same qualified subject
```

The crate uses a registered `SurfaceProfile` plus one explicit `SurfaceBinding` per material surface. Every registered surface is `Known(commitment)`, `Unknown`, `Unavailable(reason)`, or `NotApplicable`; omission fails closed. Completeness state is part of the manifest identity.

The standard external-AI profile covers source/image identity, model, system prompt, tool authority, policy, runtime, deployment envelope, and explicitly named external dependencies. Evaluator/corpus/campaign identity is intentionally excluded and belongs to later qualification-plan/evidence layers.

Provider aliases and stable URLs are locators, not immutable revisions. When an immutable commitment is unavailable, the subject remains explicitly incomplete rather than hashing an alias and pretending it is immutable.

Raw secrets and credential bytes are not intended manifest material. Commit effective non-secret authority/policy/service identity instead; hashing low-entropy secrets is not a safe substitute.

`AiSubjectManifest::as_core_subject()` bridges the entire ASSURE-001 manifest commitment into the qualified ASSURE-000 kernel without modifying that core semantic waist.

See `ASSURE_001.md` for the complete theorem and nonclaims.
