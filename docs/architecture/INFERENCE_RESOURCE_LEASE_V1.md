# Symthaea Inference Fabric — IF-15 Execution Resource Lease v1

Status: resource-authority child of IF-14 terminal HTTP-status evidence.

## Purpose

IF-7 reserves conservative provider/account request and token capacity before dispatch.
IF-12 proves that the current provider profile, exact credential identity/epoch, and
exact quota identity/epoch belong to one explicitly registered local resource scope.
IF-13 folds that proof into executable authority and revalidates it immediately before
private transport. IF-14 preserves the terminal HTTP status needed to distinguish
rate limiting, credential rejection, and other provider failure without retaining
provider error bodies.

Those theorems were still separate point-in-time facts. IF-15 joins them into one
process-local execution resource capability that remains active from reservation
through terminal settlement.

## Authority ownership

`InferenceExecutionResourceAuthority` privately owns:

- one IF-7 `InferenceResourceGuard`;
- one IF-12 `InferenceResourceScopeRegistry`;
- a non-secret authority lineage id;
- a monotonic local authority epoch;
- the active lease-generation set.

Neither the guard nor scope registry is exposed mutably by the high-level executor.
Changing both is one `replace_state` transition, and replacement is rejected while
any lease is active.

This freezes local resource authority across an external attempt rather than allowing
guard and resource-scope state to rotate independently between verification and I/O.

## Resource lease

`InferenceExecutionResourceLease` is intentionally non-Clone and non-Serde. It owns:

- authority identity and epoch;
- lease generation;
- conservative reserved token exposure;
- the exact verified IF-12 resource-scope proof;
- a domain-separated lease digest;
- the opaque IF-7 reservation.

The reservation cannot be separated from the lease by ordinary callers.

Lease identity uses:

`symthaea.inference.execution-resource-lease.v1`

and binds authority lineage, authority epoch, lease generation, reserved token
exposure, and resource-scope proof digest.

## v5 execution binding

IF-15 introduces:

`symthaea.inference.resource-lease-binding.v5`

The v5 provider-state digest composes:

- the exact IF-13/v4 provider-state digest;
- the active execution-resource lease digest.

Consequently two otherwise identical requests using different lease generations are
different execution authorities.

## Conservative reservation before permit issuance

The high-level safe preparation path is:

1. resolve the current IF-10/IF-11 provider profile;
2. verify current IF-12 resource scope;
3. derive the IF-13/v4 binding;
4. derive token exposure from `estimated_input_tokens + max_output_tokens`;
5. reserve one IF-7 request plus that maximum token exposure;
6. compose the v5 binding;
7. issue the IF-2 one-use permit;
8. immediately prepare the permit;
9. return one joined `PreparedLeasedInferenceExecution` capability.

The caller does not choose the reserved token quantity and does not receive a naked
`InferenceExecutionResourceLease` from the public preparation API.

If permit issuance or preparation fails after reservation, IF-15 retires the lease
conservatively. Request/token capacity is not refunded.

This prevents a failed permit from leaving an ordinary caller responsible for
manually cleaning up resource authority.

## Joined prepared capability

`PreparedLeasedInferenceExecution` joins exactly one prepared IF-2 execution with
exactly one IF-15 lease. It is non-Clone/non-Serde.

The public execution APIs accept this joined capability rather than separate prepared
permit and reservation objects.

Presenting a capability to a different local authority is checked before provider
profile resolution. The wrong executor returns the intact capability so the
originating authority can still execute or cancel it instead of orphaning the
reservation.

## Explicit cancellation

A prepared attempt that will not execute should be consumed through
`cancel_prepared_leased_current`.

Cancellation:

- validates that the capability belongs to this authority;
- retires the IF-7 reservation without refund;
- removes the active lease;
- emits an `InferenceFailureClass::Cancelled` receipt.

Thus ordinary user/application cancellation is a first-class terminal state rather
than an availability leak.

## Final pre-I/O revalidation

For execution, IF-15:

1. checks lease ownership before provider work;
2. re-resolves the current provider profile using the trusted IF-11 clock;
3. revalidates the active lease against current scope/credential/quota authority;
4. re-derives v4;
5. re-composes v5 from the same active lease;
6. sends only the exact final binding through the credential-bound exact-binding seam;
7. compares that binding with prepared authority before deriving the wire request;
8. performs private HTTP I/O only after the comparison succeeds.

A semantic/profile/scope change before I/O is rejected locally. The lease is retired
without refund and no wire evidence is attached.

## Terminal settlement

IF-14 terminal status evidence allows exact IF-7 settlement without provider error
bodies:

- success -> `settle_success`;
- HTTP 429 -> `settle_rate_limited` and bounded local cooldown;
- HTTP 401/403 -> `settle_auth_rejected` and block the current credential epoch;
- HTTP 5xx/other provider/transport failure -> conservative `settle_failure`;
- local pre-wire rejection/cancellation -> `abandon` without refund;
- executor-fatal path without a terminal outcome -> conservative failure settlement.

Provider-declared usage is never used to refund reserved capacity.

If the model request completed but local settlement then fails, the completed
`InferenceExecutionOutcome` is preserved in the error for recovery rather than being
silently discarded.

## Concurrency theorem

Because reservation occurs while the combined authority is locked, concurrent
callers cannot both spend the last locally available request/token capacity.

A prepared active lease also blocks atomic resource-state replacement until terminal
settlement or explicit cancellation.

## Qualification regressions

The focused IF-15 suite establishes:

- permit failure after reservation retires the lease without refund;
- explicit cancellation retires the lease, records `Cancelled`, and does not refund;
- a first prepared lease prevents a second caller from spending the last request;
- active leases block resource-state replacement;
- real localhost success settles the lease and preserves conservative charging;
- 429 evidence drives cooldown and retains no provider error body;
- both 401 and 403 block the current credential epoch;
- provider 500 drives conservative failure/circuit behavior;
- a changed request is rejected before wire I/O and the reservation is not refunded;
- distinct lease generations produce distinct v5 provider-state identities;
- wrong-authority presentation returns the capability intact and the origin can
  subsequently complete the real request.

## Explicit non-claims

IF-15 does **not** yet establish:

- durable/distributed lease persistence or crash recovery;
- automatic reaping of accidentally dropped process-local prepared capabilities;
- global uniqueness enforcement for caller-supplied authority ids;
- cryptographic/Xenia authorization of resource-authority or scope-registry mutation;
- provider-side proof that the local quota guard belongs to the named external account;
- provider-side credential/account ownership attestation;
- distributed resource-scope consensus;
- deployment-aware IF-8 routing/resource handles;
- signed provider/deployment manifests;
- provider presets, live discovery, or runtime/factory replacement.

Authority ids are therefore a local construction invariant: independent live
authorities must receive distinct unpredictable/non-colliding ids. A future resource
registry should enforce uniqueness centrally within the runtime.

A genuinely lost/dropped prepared capability remains fail-closed: its reserved
capacity may stay charged and resource-state replacement may remain blocked until
process-level recovery. The intended normal lifecycle is execute or explicit cancel;
durable lease journals/expiry recovery are future work and must not be approximated
with unsafe automatic refunds.

## Next theorem

The next useful layer is deployment-aware routing. IF-8 currently ranks abstract
provider/model candidates, while IF-15 proves execution authority for an exact
provider deployment/account/credential/quota resource. A future router resource handle
should point to this exact qualified execution resource instead of selecting a naked
provider/model pair.
