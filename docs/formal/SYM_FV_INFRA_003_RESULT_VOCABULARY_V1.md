# SYM-FV-INFRA-003 — Qualification Result Vocabulary v1

Formal evidence lanes use one canonical admission-state vocabulary:

```text
Pass
Fail
Blocked
EnvironmentFailure
```

Tool- or lane-specific outcomes remain separate metadata, for example:

```text
result = Pass
proof_checker_outcome = LeanTypechecked

result = Pass
qualification_kind = StaticExportContractQualified

result = Blocked
semantic_result_class = UnsupportedDependencyBoundary
```

The formal workflow safety checker rejects receipt-like workflow assignments where `result` is a tool-specific string, and rejects tool-specific outcome/kind fields when no canonical result is emitted.

This structural policy does not prove that a tool-specific outcome was mapped to the correct canonical state. That mapping remains a separate semantic/refinement obligation for each lane family.

## Non-equivalences

```text
canonical result vocabulary
!= theorem truth
!= verifier correctness
!= semantic correctness of outcome mapping
!= evidence-class promotion
!= runtime authority
```

Historical receipts are not rewritten. A result emitted by an older exact subject remains evidence of that historical subject only; changing vocabulary creates a new exact-head evidence subject.
