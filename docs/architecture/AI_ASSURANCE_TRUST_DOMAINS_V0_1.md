# AI Assurance Trust Domains v0.1

**Status:** Research foundation extension for PR #121.

## Why this layer exists

The low-level assurance kernel answers questions such as:

- is a capability the right static kind?
- does its scope cover the action?
- is it expired?
- is it bound to this exact action transition?
- is the lifecycle state valid?

Those checks are necessary, but they do not by themselves answer a more fundamental question:

> **Which root of authority does the host actually trust?**

A public library cannot infer organizational trust merely from the existence of an `AuthorityRoot`. A process can create another root. Therefore a security-sensitive integration must not accept "any structurally valid grant" as equivalent to "a grant from the host-selected authority domain."

## Two-layer model

```text
LOW-LEVEL MECHANICS

AuthorityRoot
   |
   +--> BoundOneShotCapability<K>
   |
   +--> Action<K, Proposed -> ... -> Resolved>

Provides affine capabilities, scope, expiry, exact binding,
and typestate. It does NOT decide which root the host trusts.


TRUSTED HOST PATH

AuthorityDomain
   |
   +--> random AuthorityDomainId
   +--> monotonic AuthorityEpoch
   +--> AuthorityVerifier --------------------+
   |                                         |
   +--> TrustedBoundOneShotCapability<K>      |
                                             v
TrustedAction<K, Proposed>
   -> RiskAssessed
   -> Authorized
   -> Executed
   -> Observed
   -> Resolved
                                             |
                                             v
                                  TrustedEvidenceReceipt
```

## Trust-domain identity

`AuthorityDomain::new` creates an internally generated random `AuthorityDomainId`. Callers cannot choose an existing domain id for a second domain.

A `TrustedAction` is admitted under a host-selected `AuthorityVerifier` and records that domain identity. A trusted execution grant must carry the same domain identity and current revocation epoch before the wrapped low-level action transition is attempted.

This means a separately created root or authority domain cannot authorize an action that the trusted host admitted under another domain.

## Revocation epochs

Expiry is necessary but insufficient for long-running autonomous systems. Operators also need a fast fail-closed way to invalidate authority that has not yet expired.

Each authority domain therefore owns a monotonic `AuthorityEpoch`.

`revoke_all()` advances the epoch. After rotation:

- old unspent grants are rejected;
- old proposed/risk-assessed actions cannot be authorized;
- actions authorized in an old epoch cannot cross the execution boundary;
- newly admitted actions and grants use the new epoch;
- already executed actions may still be observed and resolved so evidence about past side effects is not destroyed.

This is intentionally a coarse emergency revocation primitive. Per-grant revocation sets and durable distributed revocation are later research layers.

## Host verifier rule

The verifier used to admit execution is part of the trusted tool adapter, not model-provided data.

Correct integration shape:

```text
Model / planner
     |
     v
Proposal data
     |
     v
Trusted host adapter
     | owns/selects AuthorityVerifier
     v
TrustedAction
     |
     v
policy / gate
     |
     v
trusted grant
     |
     v
executor that retains the same host verifier
```

Incorrect integration shape:

```text
Model chooses verifier/root
     |
     v
executor accepts it
```

The latter simply moves ambient authority into another object and is not a security boundary.

## Independent observation

The trusted path strengthens the lower-level "separate observation grant" rule: the observer principal must differ from the action actor for an externally resolved outcome.

Execution and observation may use different authority domains. This supports architectures where, for example:

- an operations policy domain authorizes execution;
- an evidence/monitoring domain authorizes observation;
- the final `TrustedEvidenceReceipt` records both domains and epochs.

## Threats this extension closes

The trusted-domain path adds direct defenses against:

1. minting an otherwise-valid exact grant from an unrelated self-created root;
2. replaying authority after emergency domain revocation;
3. authorizing before revocation and executing afterward;
4. presenting the acting principal as its own independent external observer;
5. losing which execution and observation trust domains backed final evidence.

## Remaining boundary assumptions

This still does not make arbitrary code in the same trusted process untrusted by magic. If attacker-controlled code can directly invoke privileged host adapters, replace verifier state, or modify the running binary, the process boundary has already failed.

Therefore PR #122 should treat the eventual MAGI/tool integration as a capability boundary:

- cognition supplies proposal data, not verifier objects;
- concrete tool adapters retain host-selected verifiers internally;
- `AuthorityDomain` stays outside model-facing APIs;
- shell/process/filesystem/network ambient authority stays outside the assurance kernel;
- stronger isolation should eventually use process/Wasm-component boundaries where practical.

## Qualification additions

PR #121 should now demonstrate:

- unrelated trust-domain grant rejection;
- epoch rotation invalidates unspent grants;
- epoch rotation after authorization blocks execution;
- raw low-level grants do not type-check on the trusted admission API;
- self-observation by the acting principal is rejected;
- final evidence preserves separate execution and observer trust-domain lineage.

These are foundation properties. MAGI runtime integration remains out of scope until the foundation is qualified.
