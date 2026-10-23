# PIE-009U — Metrology-lineage diversity and traceability cut sets

Status: independent structural reference only.

## Claim under test
Independent sensors do not automatically provide independent metrology evidence. If multiple qualified measurement paths share a primary anchor, transfer standard, calibration fixture, or other declared metrology failure domain, they may fail together as one traceability lineage.

The reference therefore evaluates decision policies over **qualified measurement paths plus explicit metrology lineage**.

## Structural semantics
Each path declares:

- a path ID;
- sensor ID;
- terminal anchor ID;
- explicit metrology failure groups;
- whether the path is currently qualified.

A decision policy declares a minimum path count and a minimum distinct-anchor count. A candidate subset is sufficient only when it uses distinct sensors, satisfies the anchor count, and no pair shares a declared metrology failure group.

The oracle enumerates inclusion-minimal sufficient path sets, inclusion-minimal path-loss cut sets, and inclusion-minimal metrology failure-group cut sets. Unqualified paths do not contribute.

## Independent execution evidence
The final Python candidate was executed locally on 2026-09-13 and returned:

`ok`

The executed fixture contains two qualified sensors sharing one imported master anchor, one path on local anchor A, one path on local anchor B, and one unqualified spare. It demonstrates that the two imported-master sensors alone are insufficient despite sensor diversity; cross-anchor pairs are sufficient when their declared failure groups are disjoint; one anchor loss is survivable with three anchor lineages; loss of any two primary anchors destroys a two-anchor policy; and a shared calibration fixture defeats otherwise distinct anchors.

Adding a genuinely independent fourth anchor path preserves every previously sufficient set. Malformed paths fail closed.

## Important non-claims
This is structural lineage analysis only. It does not estimate failure probabilities, prove physical independence, detect cyber compromise, replace calibration physics, or recompute PIE-009T precision/aging qualification. It relies on upstream qualification to say which paths are valid and on explicit failure-domain declarations to say which dependencies may fail together.
