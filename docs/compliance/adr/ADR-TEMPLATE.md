# ADR-XXX — <Decision Title>

Status: Proposed  
Date: YYYY-MM-DD  
Decision class: A | B | C  
Owner:  
Related issues / PRs:  
Supersedes: None  
Superseded by: None

## Context

Describe the problem, forces, constraints, and why a decision is required now.

For claim-bearing engineering decisions, reference the relevant `NEED-*`, `REQ-*`, `CON-*`, `ASM-*`, `IFC-*`, `RISK-*`, or design-packet identifiers rather than duplicating their canonical content.

## Decision

State the selected architectural/design decision precisely.

## Alternatives Considered

Preserve meaningful alternatives, including rejected options.

For each alternative record:

- summary;
- advantages;
- disadvantages;
- constraints;
- evidence;
- uncertainty;
- reason accepted/rejected.

Do not replace the underlying tradeoffs with an unexplained universal scalar score.

## Evidence / Rationale

List the scientific, empirical, formal, benchmark, operational, or engineering evidence used.

Distinguish evidence planes and claim scope explicitly.

```text
model/simulation evidence != FIELD evidence
operational fact != engineering qualification
```

## Risks and Consequences

### Positive consequences

-

### Negative consequences / tradeoffs

-

### Risks / failure modes

-

### Mitigations and verification route

-

A proposed mitigation is not a verified mitigation.

## Interfaces and Assumptions

### Interfaces affected

-

### Assumptions

-

For each claim-relevant assumption state its invalidation condition.

## Verification / Validation Impact

- Requirements affected:
- Verification cases affected:
- Validation cases affected:
- Existing evidence invalidated or made stale:
- Requalification required:

## Configuration / Migration Impact

Describe affected configuration generations, compatibility, migration, rollback, and history-preservation requirements.

A configuration change must not silently inherit claim-bearing evidence when the change affects the claim.

## Authority Boundary

State what this decision may influence and what it may not authorize.

Default for engineering/design ADRs:

`analysis_and_design_only_no_physical_execution_authority`

## Decision Outcome

- [ ] Proposed
- [ ] Accepted
- [ ] Rejected
- [ ] Superseded

Acceptance records the decision. It does not by itself prove implementation, verification, validation, safety, certification, or physical capability.
