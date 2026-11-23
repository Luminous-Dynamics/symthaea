# Independent SFEB oracle verification

This schema defines a second, independently produced in-toto Statement for the verifier's findings. It is deliberately separate from the execution predicate.

## Trust split

1. The execution statement attests which immutable trace and browser-observation artifacts an authorized execution workflow produced.
2. The oracle-verification statement attests which outcomes an independently controlled verifier derived from those exact inputs and a pinned oracle policy.

The verifier identity must be authenticated from the signed Sigstore bundle and an out-of-band allowlist. The JSON property `independent_execution_identity: true` is only a schema-level assertion; it is not proof of independence. The trust policy must reject the executor's workflow identity, untrusted forks, mutable workflow references and wildcard repository/workflow matching.

## Required bindings

- digest of the signed execution statement;
- run, campaign, case, task, fixture, source and environment identities;
- raw trace bundle digest and terminal event digest;
- observation bundle digest plus individual artifact hashes/lengths;
- exact verifier revision, verifier binary digest and oracle-policy digest;
- one outcome per required oracle, with evidence event and/or artifact digests;
- an explicit accepted/rejected/inconclusive verdict and missing-oracle list.

The verifier must re-open every subject and artifact, re-run trace verification, validate every evidence reference, and evaluate the frozen rule. It must reject duplicated, unknown or missing oracle IDs. It must not copy a worker's `observed_oracle_ids` or disposition into this statement.

## Verdict rule

- `accepted`: policy is complete and every required oracle passes.
- `rejected`: at least one required oracle has a definitive failure.
- `inconclusive`: missing inputs, unknown oracle, incomplete policy, invalid trace, insufficient evidence or verifier error prevents a defensible binary decision.

The qualification report v2 should consume this independently signed statement. The current v1 adjudicator remains a legacy local integrity format and must not be promoted as independently oracle-adjudicated.

## Implementation issue

The evaluation engine, policy schema, result generation, signature verification and report-v2 migration are tracked separately in issue #13247. Execution attestation and trace integrity remain in #13228.

## Interaction-as-a-Judge is a secondary evaluator, not the trust root

The independent verifier should support a distinct interaction-diagnostic
channel: a computer-use evaluator may traverse journeys, identify broken
requirements, rank repair hypotheses and emit evidence-linked explanations.
It must not be the sole acceptance oracle. Deterministic policy checks remain
authoritative for security, identity, evidence integrity and explicit contract
requirements. Human-calibrated preference outcomes remain a separate result
family. Pin judge model/prompt/tool versions and calibration data; adversarial
page instructions must never alter verdict policy.

The October 13, 2026 Interaction-as-a-Judge paper reports preference
prediction 65.6% to 69.1%, requirement verification 77.0% to 81.6%, and
iterative repair 55.4% to 75.5% within five attempts under its own protocol.
The paper describes 6,858 preference pairs and 222 correctness applications;
the released dataset cards list 6,390 pairs and 189 app versions with 362
human-verified requirements. The discrepancy is unresolved; keep paper and
dataset-card counts separate. These results are not SFEB measurements or proof
of oracle correctness.
