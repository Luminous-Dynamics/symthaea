# Patch 0020: security add emergency software release procedure

**Series:** 29

## Objective

Support urgent corrective binaries without weakening evidence-history or compatibility truth.

## Intended changes

- Define minimal urgent patch scope, independent review, required regression fixture, signed software artifact, advisory, and follow-up full qualification.
- Permit temporary operational mitigations that do not rewrite authoritative evidence.
- Record every skipped non-hard gate.

## Required evidence

- Hard semantic, transaction, policy, and privacy gates cannot be skipped.
- Emergency release identity differs from the previous release.
- Follow-up qualification remains mandatory and visible.

## Non-claims

- Does not roll back or delete authoritative catalog history.
- Does not permit undocumented hot fixes.
