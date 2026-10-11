# symthaea-ai-assurance

Project-neutral Rust primitives for constraining autonomous-system authority.

This crate deliberately separates **what a model/planner proposes** from **what trusted host code permits to execute**.

## Layers

### Low-level mechanics

`capability` + `action` provide:

- affine capability values;
- typed capability classes (`Read`, `Write`, `Execute`, `Network`, `Deploy`, `UpdateModel`, `Observe`);
- hierarchical logical scopes;
- expiry and non-widening delegation;
- exact one-shot transition bindings;
- action typestate (`Proposed -> RiskAssessed -> Authorized -> Executed -> Observed -> Resolved`);
- immutable execution/observation evidence lineage.

These types validate capability mechanics. They do **not** by themselves decide which independently created authority root a host trusts.

### Trusted-host path

Security-sensitive integrations should normally use:

- `AuthorityDomain`;
- `AuthorityVerifier`;
- `TrustedBoundOneShotCapability<K>`;
- `TrustedAction<K, S>`;
- `TrustedEvidenceReceipt`.

The trusted path adds:

- a random host trust-domain identity;
- monotonic revocation epochs;
- rejection of grants from unrelated roots/domains;
- execution-time revocation checks;
- independent observer-principal enforcement;
- execution-domain + observation-domain lineage in final evidence.

## Integration rule

A concrete executor should **retain the host-selected `AuthorityVerifier` internally**.

Model/planner output may supply proposal data, but it must not choose the root/domain/verifier the executor accepts.

Conceptually:

```text
model / planner
      |
      v
 proposal data
      |
      v
 trusted host adapter ---- owns AuthorityVerifier
      |
      v
 TrustedAction<K, Proposed>
      |
      v
 risk + policy gate
      |
      v
 TrustedBoundOneShotCapability<K>
      |
      v
 TrustedAction<K, Authorized>
      |
      v
 concrete executor ------- retains same verifier
      |
      v
 Executed -> independently Observed -> Resolved
```

Do **not** treat this as equivalent:

```text
model supplies proposal + verifier/root
                    |
                    v
                 executor
```

That simply recreates ambient authority in object form.

## Revocation

`AuthorityDomain::revoke_all()` advances the domain epoch. Outstanding grants, proposals, and authorized-but-not-yet-executed actions from older epochs fail closed at trusted admission/execution.

Already executed actions may still be observed and resolved after execution-domain revocation so evidence about side effects is preserved.

## Non-claims

This crate does not prove that an AI is aligned, conscious, truthful, wise, or factually correct. It aims at narrower machine-checkable properties around authority, state transitions, provenance, and evidence.

It also does not make attacker-controlled code safe merely because that code shares a process with trusted adapters. Stronger isolation should use process/Wasm-component boundaries where practical.

## Qualification

The focused `AI Assurance` workflow runs with the repository-pinned Rust toolchain and checks:

- formatting;
- locked workspace resolution;
- unit + integration + compile-fail/doctests;
- Clippy with warnings denied.

The repository-wide CI remains an additional merge gate.
