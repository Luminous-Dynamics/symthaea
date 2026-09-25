# ENG-EVID-REF-001 adversarial corpus

Status: shared semantic qualification fixture

Parent: #5683

This fixture freezes the cross-domain hostile cases that ENG-CATALOG, ENG-MAG, PHOT, ENG-THERM, ENG-VAC and later engineering snapshots must satisfy before stronger evidence claims are admitted.

## Core distinctions

```text
reference string present
!= referenced subject resolved

well-formed identifier
!= subject exists
!= subject belongs to this capsule
!= subject has required authority class

backend required
!= backend configured
!= backend executed
!= result parsed
!= result admitted as prediction evidence

source document present
!= source class compatible with evidence class
```

## Required corpus

| ID | Adversarial case | Required disposition |
|---|---|---|
| R01 | Evidence references absent local model profile | Reject |
| R02 | Duplicate local model-profile semantic identity | Reject |
| R03 | Same local identity resolves to two different semantic objects | Reject |
| R04 | Digest-shaped external component ID has no resolver hit | Preserve unresolved; block strong claim |
| R05 | External reference resolves to wrong subject class | Reject claim requiring exact class |
| R06 | `FullWaveFdtdRequired` supplied as optical prediction | Reject |
| R07 | `ExternalFiniteElementRequired` supplied as field/temperature prediction | Reject |
| R08 | `RarefiedKineticRequired` supplied as flow prediction | Reject |
| R09 | `InternallyMeasured` + manufacturer document | Reject |
| R10 | `InternallyMeasured` + distributor document | Reject |
| R11 | `InternallyMeasured` + imported database | Reject |
| R12 | `InternallyMeasured` + community report | Reject |
| R13 | `InternallyMeasured` + internal model record | Reject |
| R14 | Measured evidence cites prediction-only model as measurement authority | Reject |
| R15 | Numerical prediction requires execution/result receipt but only capability declaration exists | Reject |
| R16 | Malformed/dangling reference survives serialization round trip | Remains invalid after decode |
| R17 | Input ordering changes only | Preserve canonical identity |
| R18 | Semantic duplicate/contradiction inserted | Reject; never silently deduplicate |
| R19 | Configured/commanded state supplied for observed/measured requirement | Reject |
| R20 | Imported measured data lacks admitted measurement provenance/profile | Reject strong measured claim |

## ENG-CATALOG V1 source compatibility

| Evidence class | Admitted source kind |
|---|---|
| `ManufacturerGuaranteed` | manufacturer datasheet/application note/drawing/certificate |
| `ManufacturerTypical` | manufacturer datasheet/application note/drawing/certificate |
| `ManufacturerAbsoluteMaximum` | manufacturer datasheet/application note/drawing/certificate |
| `ManufacturerNominal` | manufacturer datasheet/application note/drawing/certificate |
| `DistributorMetadata` | distributor document |
| `ImportedDatabase` | imported database |
| `CommunityReported` | community report |
| `InternallyMeasured` | internal measurement record |
| `DerivedModel` | internal model record |
| `Assumption` | no source document |
| `Unknown` | no source document |

`StandardOrHandbook` and `Other` source kinds intentionally receive no V1 specification-evidence authority class. A future profile should add an explicit class rather than overload an unrelated one.

## Capability/result rule

Domain implementations must preserve stages equivalent to:

```text
ModelCapabilityRequirement
BackendCapabilityDeclared
ExecutionSubjectPrepared
ExecutionReceipt
ParsedNumericalResult
AdmittedPredictionEvidence
```

Names may differ, but a `...Required` marker can never mint positive prediction evidence.

## Exit use

A domain satisfies this corpus only when its validation API rejects the applicable hostile cases. This fixture does not establish physical truth, solver validity, calibration, safety or actuation authority.
