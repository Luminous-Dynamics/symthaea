# MFG-PROC-003 — evidence-bounded manufacturing capability

Manufacturing capability is an engineering evidence subject, not a machine label or resource-availability record.

```text
"5-axis mill"
!= process capability
!= qualified capability
!= available capacity
!= job authorization
```

A capability profile binds an exact resource subject, exact process definition, externally owned process envelope/profile refs, configuration/tooling refs, evidence class and a bounded validity/revision window.

V1 evidence classes preserve at least:

```text
UnknownOrUnavailable
< Declared
< ManufacturerSpecified
< ObservedCapability
< QualifiedUnderProfile
< ProductionQualifiedUnderProfile
```

This ordering is a minimum-evidence relation only; it does not mean a stronger label can repair mismatched process/envelope/configuration refs.

Capability matching must return a structured disposition rather than a boolean. Distinguish exact admission, weaker evidence, unresolved external refs, process mismatch, envelope mismatch, stale/expired capability and unknown/malformed state.

Commercial offer, price, queue/calendar capacity, current machine health, consumable inventory and provider policy stay outside the engineering capability subject and belong to Mycelix/resource-operations composition.

SE-SEM remains the owner of numeric quantity/envelope semantics; ENG-CATALOG remains the owner of exact resource/tool/component identity. MFG-PROC carries references rather than copying either ontology.
