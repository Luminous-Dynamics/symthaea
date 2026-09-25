# MFG-PROC-002 — process transformation state boundary

A manufacturing process transforms externally owned engineering state subjects; it does not copy their physics or quantity systems into the process ontology.

```text
ProcessStateRef
!= material model
!= geometry model
!= measured state
!= process capability
```

State references may classify material, geometry, surface, assembly, cleanliness, thermal-history, inspection, or extension-defined state subjects. The referenced subject remains owned by its canonical domain.

A transformation contract binds exact input and output state references plus explicitly preserved state references. Duplicate refs reject. Missing input/output state rejects for transformation processes. Resolution of an identifier-shaped external reference is a later composition theorem; the presence of a string does not prove the subject exists.

This tranche introduces no numeric process parameters, no recipe semantics, no material-property duplication, and no machine authority.
