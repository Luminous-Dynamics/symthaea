# Trusted-main root receipt V1

`TrustedMainRootReceiptV1` is an immutable historical composition record for one
exact repository root. It exists to prevent downstream qualification code from
silently treating “this commit exists on `main`” as equivalent to “this commit
was admitted as a protected trust root.”

This contract is security/evidence infrastructure only. It has no scientific,
Butlin, consciousness, cryptographic-signature, self-hosted-runner, or current-
admission authority.

## Evidence layers

The receipt keeps these facts separate:

```text
P0 policy intent
!=
repository-owned structural ruleset readback
!=
effective active-rule projection on main
!=
behavioral/administrative enforcement evidence
!=
historical root admission
!=
current admission
```

The composer consumes:

1. the complete canonical P0 policy manifest;
2. a valid `P0StructurallySatisfied`/rejected verification object whose
   content-addressed `verification_id` is recomputed;
3. a valid effective-rule verification object whose content-addressed
   `verification_id` is recomputed;
4. an exact root subject bound to repository ID, canonical repository name,
   `refs/heads/main`, commit SHA, commit tree, and `github-commit-readback`;
5. optional typed enforcement evidence;
6. an explicit root harness identity (`sha256:...`) or the exact sentinel
   `none-pre-bootstrap`.

The composer independently recomputes `ProtectionPolicyId`, structural and
effective verification IDs, `RootSubjectId`, `BypassPolicyId`, optional
`EnforcementEvidenceId`, and the final `TrustedMainRootReceiptId`.

## Admission states

`AdmittedHistoricalRoot` is possible only when all three protection layers are
satisfied:

```text
P0StructurallySatisfied
+ P0EffectiveRulesSatisfied
+ EnforcementSatisfied
= AdmittedHistoricalRoot
```

Missing enforcement evidence produces `PendingEnforcementEvidence`.
Inconclusive enforcement produces `EnforcementInconclusive`.
Rejected structural/effective evidence produces
`ProtectionEvidenceNotSatisfied`.

A malformed content ID, repository/ref mismatch, ruleset mismatch, root SHA
mismatch, or cross-evidence identity mismatch is a verifier error rather than a
weaker receipt.

## Enforcement evidence V1

The initial enforcement-evidence contract intentionally supports only two
reviewed methods:

- `administrative-verification`;
- `non-bypass-negative-ref-update`.

For `EnforcementSatisfied`, the evidence must explicitly state:

```text
ordinary_direct_push = blocked
force_push           = blocked
deletion             = blocked
bypass_assurance     = policy-matched
```

The evidence is content-addressed but not authenticated. Rule-suite IDs may be
recorded as corroborating GitHub evidence, but ordinary PR rule-suite success is
not by itself equivalent to a negative direct-push theorem.

## Review assurance

P0 currently requires PR mediation but zero approving reviews. V1 therefore
records:

```text
review_assurance = pr-mediated-no-required-approval
```

If a future policy requires one or more approvals, a new policy identity and
root-receipt identity will record:

```text
review_assurance = pr-mediated-required-approval
```

Neither value claims independent multi-party review beyond what the underlying
GitHub policy actually requires.

## Historical vs current authority

Every receipt contains:

```text
historical_scope              = exact-root-only
current_admission             = not-evaluated
receipt_attestation           = none
self_hosted_runner_activation = not-authorized
scientific_authority          = none
bootstrap_authority           = none
```

A later ruleset, bypass, policy, root, harness, or enforcement change creates a
new content identity. Historical receipts are not rewritten. Current
admission/withdrawal remains a separate policy layer (#931), and detached
producer/witness authentication remains separate (#955).

## Bootstrap

`none-pre-bootstrap` is allowed so a protected pre-#1119 root can be recorded,
but it grants no bootstrap authority. #1119 remains explicitly
`BootstrapNoPredecessor`; this receipt must not be used to manufacture a
circular predecessor that retrospectively self-authorizes #1119.

After the bootstrap recipe is governance-installed on protected `main`, future
root receipts should bind the exact content-addressed harness/recipe identity.
Post-bootstrap #1157 witnessing should consume an admitted root receipt ID and
verify that its explicit root SHA/tree and trusted recipe agree with that
receipt before granting the narrow software-contract witness disposition.

## CLI

```bash
python3 scripts/trusted_main_root_receipt.py \
  policy.json \
  structural-verification.json \
  effective-rules-verification.json \
  root-subject.json \
  --enforcement-evidence enforcement.json \
  --root-harness-identity sha256:<64-hex>
```

The CLI exits zero only for `AdmittedHistoricalRoot`. Use `--conformance` when
valid non-admitted receipts are intentionally being tested.

## Non-claims

A root receipt does not prove that GitHub is currently configured the same way,
that the evidence producer is cryptographically authenticated, that a historical
root remains currently admissible, that arbitrary PR code is safe on a
self-hosted runner, or that any research/scientific claim is true.

Related: #330, #1119, #1157, #1240, #905, #931, #955.
