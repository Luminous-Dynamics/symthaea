# PIE-006A circularity / inventory / heat-allocation oracle

## Purpose

Freeze implementation-independent bookkeeping semantics for circular industrial ecology before adding production code.

The oracle is `scripts/pie-circularity-oracle.py` and imports no Symthaea code.

## Executed evidence

The final self-test was executed locally on 2026-09-11 and returned `ok` before the branch was written.

Synthetic fixtures prove:

- a material inventory cannot be allocated twice;
- exact recovery accounting preserves recovered, reject, and loss mass;
- an 80% recovery remains lossy across repeated cycles and never becomes 100% by graph traversal;
- an unsupported A->B->A recycle loop cannot start from zero inventory;
- the same loop can operate when real seed inventory exists;
- a lossy recycle process reduces available recyclable inventory and exposes reject/loss;
- waste heat is a finite resource and cannot be allocated beyond source energy;
- lower-temperature heat cannot satisfy a higher-temperature process requirement;
- explicit heat splitting is allowed when total allocation and temperature requirements close.

## Accounting rules

1. Recovered material is transferred inventory, not new production.
2. Future recycle output cannot be consumed before it exists.
3. Losses remain explicit rather than disappearing from a closed-loop percentage.
4. Heat is allocated once; downstream reuse cannot double-credit the same joule.
5. Heat reuse requires adequate source temperature as well as energy quantity.
6. This oracle does not upgrade material grade; PIE-003 governs composition/grade compatibility.

## Non-claims

This tranche does not establish Moon/Mars recycling performance, equipment lifetime, recovery economics, real heat-exchanger efficiency, thermodynamic feasibility, or industrial self-sufficiency.

Tracks #1639 and master #1604.
