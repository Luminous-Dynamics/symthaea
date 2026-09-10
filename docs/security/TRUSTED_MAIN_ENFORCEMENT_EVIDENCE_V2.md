# Trusted Main Enforcement Evidence V2

This layer turns **GitHub server rule-evaluation readbacks** into a narrow
behavioral-corrobation artifact for P0.

It exists because these are different claims:

```text
ruleset is configured and active
!=
GitHub evaluated and rejected a concrete operation
```

Likewise:

```text
operator wrote "blocked"
!=
GitHub rule suite records an active rule failure
```

## Inputs

The verifier requires the already-established P0 inputs:

- reviewed P0 policy;
- `P0StructurallySatisfied` structural readback;
- `P0EffectiveRulesSatisfied` active-rule projection;
- exact trusted-main root subject (`root_sha`, `root_tree`).

Those upstream verification IDs are recomputed before behavioral evidence is
considered.

It may then consume up to three **detailed GitHub rule-suite** readbacks.

## Three independent behavioral theorems

P0 has three relevant rules and V2 keeps them distinct:

| operation | required failed active rule evaluation |
| --- | --- |
| ordinary direct update | `pull_request` |
| force push / non-fast-forward update | `non_fast_forward` |
| branch deletion | `deletion` |

A rule suite used for one operation may not be reused for another operation.
All three observations must bind the same exact `before_sha == root_sha`.

For each supplied operation the verifier requires:

- positive rule-suite ID;
- explicit actor ID and name;
- exact repository numeric ID;
- GitHub rule-suite short repository name (`symthaea`);
- exact `refs/heads/main`;
- canonical before/after SHA values;
- provider timestamp shape;
- top-level rule-suite `result == fail`;
- exactly one matching rule evaluation from the expected P0 ruleset ID;
- `enforcement == active`;
- rule-evaluation `result == fail`;
- the correct operation-specific `rule_type`.

The numeric repository ID and ruleset ID are the primary stable provider
identities. Human-readable names are additional consistency checks.

## Dispositions

The output deliberately does not use a generic boolean:

```text
no behavioral suites
    -> EnforcementConfiguredOnly

some valid suites, others absent
    -> EnforcementBehavioralEvidencePartial

any supplied suite malformed / wrong / bypassed / passing
    -> EnforcementEvidenceRejected

three distinct exact active-rule failures
    -> EnforcementBehaviorallyCorroborated
```

Only the last state is suitable as the behavioral input to a later trusted-root
admission policy that explicitly requires all three operations.

## No destructive test is implied by this code

This verifier **does not perform** a push, force push, or deletion. It only
checks readbacks produced elsewhere.

Do not casually test branch protection by experimenting against an important
live branch with a bypass-capable administrator identity.

When behavioral evidence is required, use a deliberately scoped non-bypass
identity and a reviewed procedure. Failed protected operations should leave the
root unchanged, but the operational ceremony remains separate from this parser.
Where ordinary repository activity already yields an unambiguous rule-suite
failure, that provider evidence may be used if it satisfies the exact theorem.

## GitHub rule-suite semantics

GitHub's detailed rule-suite response carries, among other fields:

- suite ID;
- actor ID/name;
- `before_sha` / `after_sha`;
- ref;
- repository ID/name;
- pushed timestamp;
- suite result;
- detailed `rule_evaluations`, each with rule source, enforcement mode, result,
  and rule type.

V2 uses those provider fields as observations. It does not treat a GitHub
provider timestamp as authenticated chronology; the output says:

```text
chronology_authority = provider-timestamp-observed-only
```

## Bypass scope

The P0 policy currently configures no bypass actors. When all three distinct
non-bypass attempts are observed failing under the exact P0 ruleset, V2 records
a narrow bypass assurance:

```text
no-configured-p0-bypass-and-distinct-non-bypass-failures-observed
```

This is **not** a universal theorem that no GitHub/platform/organization bypass
can ever exist. Higher-level server policy and future authenticated admission
remain separate.

## Relationship to trusted-root receipts

The existing root-receipt V1 enforcement schema was a useful conformance model,
but an operator-authored object saying all operations were `blocked` must not
be the final positive behavioral source.

The intended next composition is:

```text
P0 structural verification
+ P0 effective-rule verification
+ EnforcementBehaviorallyCorroborated V2
+ independently selected evidence IDs
        ↓
strict successor TrustedMainRootReceipt
```

Do not reinterpret a V1 root receipt as though it already contained this V2
provider-derived theorem.

## Non-claims

This evidence does not establish:

- current admission;
- cryptographic signer identity;
- scientific validity;
- full CI success;
- trusted chronology;
- future-root protection;
- self-hosted runner activation.
