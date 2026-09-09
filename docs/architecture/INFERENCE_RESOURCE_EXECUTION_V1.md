# Symthaea Inference Fabric — IF-13 Resource-Scoped Execution v1

Status: execution-authority child of IF-12.

## Purpose

IF-12 establishes a local proof that the current provider profile account scope,
exact credential identity/epoch, and exact quota identity/epoch belong to one
explicitly registered resource scope. That proof is still evidence, not execution
authority.

IF-13 composes the freshly verified resource-scope proof into the one-use inference
execution binding and re-verifies it again adjacent to external I/O.

## v4 binding

IF-13 introduces:

`symthaea.inference.resource-execution-binding.v4`

The v4 provider-state digest composes:

- the existing IF-11 v3 provider-state digest;
- the exact IF-12 `ResourceScopeProofDigest`.

The scope proof itself already binds registry lineage, IF-10 profile digest,
provider/deployment/account scope, credential id/epoch plus mapping evidence, and
quota id/epoch plus mapping evidence.

Therefore changing current profile truth, resource-registry lineage, credential/quota
resource epochs, or local mapping evidence changes final execution identity.

## Layer ownership

IF-13 deliberately keeps provider/resource semantics out of the raw transport layers.

- `OpenAiInferenceExecutor` owns the IF-4/v2 envelope and private HTTP transport.
- `CredentialBoundOpenAiExecutor` owns secret-to-credential-identity coherence.
- `CurrentProfileCredentialExecutor` owns IF-11/v3 current-profile qualification and rebind.
- `ResourceScopedCurrentProfileExecutor` owns IF-12 verification and IF-13/v4 composition.

The two lower executors expose only a crate-private exact-binding dispatch seam to
stricter child layers. They do not need to import provider-profile or resource-scope
proof types.

This keeps earlier focused v2/credential test harnesses independent of later authority
modules while retaining the same final exact-binding comparison before private I/O.

## No caller-supplied scope proof

`VerifiedInferenceResourceScope` is inspectable evidence, but the public IF-13 safe
executor never accepts it as an execution argument.

`ResourceScopedCurrentProfileExecutor` owns an `InferenceResourceScopeResolver`.
Both permit-binding preparation and final execution ask that owned resolver to verify
the current profile + exact IF-9 credential binding + exact quota binding.

The v4 composition function and exact-binding forwarding are crate-internal. External
callers therefore cannot replay a previously obtained scope proof into the private
transport path.

## Final pre-I/O revalidation

The public safe path is:

1. resolve current IF-10 provider profile using IF-11 trusted clock/profile key;
2. verify current IF-12 resource scope using exact credential + quota state;
3. construct v4 binding;
4. issue/prepare the one-use permit;
5. at execution, resolve current profile again;
6. verify resource scope again;
7. re-derive the complete v4 binding in the IF-13 wrapper;
8. transfer only that exact binding through the credential-bound executor;
9. compare it with the prepared binding inside the raw executor;
10. only then derive the wire request and perform I/O.

If profile or scope verification fails during execution, prepared authority is
consumed into a local `VerificationRejected` receipt and no wire evidence is attached.

If verification succeeds but any bound request/profile/resource evidence changed, the
exact-binding comparison returns `BoundStateChanged` before I/O.

## Rotation semantics

### Credential scope rotation

Suppose a permit was prepared while credential `A@1` was the current registered
resource identity. If local scope authority later registers `A@2`, the executor still
holds `A@1` and final scope verification rejects it as superseded.

Credential rotation therefore cannot silently reuse an old prepared permit.

### Quota scope rotation

Likewise, if quota scope `Q@2` supersedes `Q@1`, a prepared attempt that still carries
`Q@1` fails resource-scope verification before network execution.

This is separate from IF-7 numeric quota accounting: IF-13 proves resource-domain
coherence, while IF-7 controls spend/availability within that domain.

## Bypass narrowing

IF-11's lower credential-bound executor accessor is crate-only and the final
resource-scoped wrapper owns the current-profile executor by value. Ordinary external
callers therefore cannot bypass scope verification by reaching the private transport.

The raw and credential-bound layers retain their legacy v2 APIs for stack
compatibility, but profile-specific v3 execution semantics now live solely in the
IF-11 wrapper and resource-specific v4 semantics solely in IF-13.

## Qualification regressions

The focused IF-13 suite establishes:

- unchanged profile/resource authority re-verifies to the same v4 binding;
- missing resource-scope authority prevents permit construction;
- newer credential scope epoch rejects prepared authority before network I/O;
- newer quota scope epoch rejects old quota authority before network I/O;
- resource-registry lineage changes v4 provider-state identity;
- coherent current scope completes a real localhost OpenAI-compatible request and receipt;
- public v4 execution APIs do not accept a verified scope proof;
- the IF-11 lower executor accessor remains crate-only;
- legacy IF-4/IF-6/IF-9/IF-11 focused binaries still compile/run after lower-layer decoupling.

## Explicit non-claims

IF-13 does **not** yet establish:

- cryptographic/Xenia authorization of resource-scope registry mutation;
- provider-side attestation that a credential truly owns an external account;
- provider-side attestation that quota headers belong to that account;
- an atomic lease that freezes resource-scope registry authority across final verification and dispatch;
- durable/distributed resource-scope consensus;
- automatic credential/executor reconstruction after rotation;
- integration of IF-7 reservation capability into this final execution boundary;
- a deployment-aware IF-8 router key/resource handle;
- signed provider/deployment manifests;
- provider presets or live provider discovery;
- production runtime module export or replacement of the current `LLMBackend` factory.

A strong next theorem is **execution resource leasing**: combine IF-7 pre-dispatch
reservation with an immutable/current resource-scope authority snapshot so concurrent
quota or scope changes cannot race between final verification and external dispatch.
After that, deployment-aware IF-8 routing can select exact executable resources rather
than provider/model names.
