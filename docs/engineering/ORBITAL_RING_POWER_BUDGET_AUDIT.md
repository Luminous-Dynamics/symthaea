# Orbital-ring power-budget source audit

**Status:** source-number arithmetic reproduced; original design point not fully reconstructed.  
**Purpose:** prevent distinct electrical and thermal quantities in Birch (1982) from being conflated.  
**Primary source:** Paul Birch, “Orbital Ring Systems and Jacob's Ladders — I,” *Journal of the British Interplanetary Society*, vol. 35 (1982), pp. 475–497. [PDF](https://nss.org/wp-content/uploads/Orbital-Rings.pdf)

## Executive finding

The paper contains three relevant power estimates from different stated contexts. They should not be collapsed into one “orbital-ring power requirement”.

| Source passage | Quantity | Published approximate value | What it means |
|---|---|---:|---|
| p. 482, §3.4 | Eddy-current drag in one conducting-sheet example | 0.2 GW | An illustrative conducting-sheet / skyhook case |
| p. 482, §3.4 | Eddy-current drag using a total precessing force of about \(5\times10^{11}\,\mathrm{N}\) | around 500 GW | A scaled estimate using the same conducting-sheet loss model |
| p. 483, §3.4 | Cryogenic heat load / cryostat electrical power | about 100 MW / 10 GW | A proposed superconducting-surface mitigation, with heat rejected to a 300 K sink |
| p. 485, Table 8 | Total cryogenic heat loss / power to cryostats | \(5\times10^8\,\mathrm{W}\) / \(5\times10^{10}\,\mathrm{W}\) | A typical continuous-skyhook ORS table; the row/column and length basis still need reconstruction |

The 500 GW estimate is **not** the Table 8 cryostat power. It is an electromagnetic drag loss estimate for the conducting-sheet model. Birch says the superconducting surface would reduce drag-related dissipation to cryogenic losses, and also notes a linear-induction-motor approach could cancel much of the eddy current. A modern design would need to model those alternatives, their controls, and power recovery consistently.

The two cryogenic pairs each imply a roughly 100:1 ratio of wall-plug input to the stated cold-side load. Their absolute values differ by a factor of five. The arithmetic is clear; whether the difference comes from configuration, length basis, table normalization, or a substantive inconsistency requires a source-level reconstruction. Do not silently pick one pair or average them.

## Reproduced conducting-sheet arithmetic

Birch gives the relation

\[
P_E = F_L\frac{2}{\mu_0\sigma\delta}
\]

for the power loss under the conducting-sheet model, where \(F_L\) is the associated lift/precessing force, \(\mu_0\) is vacuum permeability, \(\sigma\) is sheet conductivity and \(\delta\) is sheet thickness. This expression can be written as \(P_E=F_L v_{\mathrm{eq}}\), where

\[
v_{\mathrm{eq}} = \frac{2}{\mu_0\sigma\delta}.
\]

Using the source's aluminium conductivity \(\sigma=3.54\times10^7\,\mathrm{S/m}\) and illustrative \(\delta=0.05\,\mathrm{m}\) gives

\[
v_{\mathrm{eq}}\approx 0.89918\,\mathrm{m/s}.
\]

For the smaller skyhook example using \(F_L\approx2\times10^8\,\mathrm{N}\):

\[
P_E\approx(2\times10^8)(0.89918)
\approx1.798\times10^8\,\mathrm{W}
\approx0.180\,\mathrm{GW},
\]

consistent with the paper's rounded 0.2 GW statement.

For the paper's approximate total precessing force \(F_L\approx5\times10^{11}\,\mathrm{N}\):

\[
P_E\approx(5\times10^{11})(0.89918)
\approx4.496\times10^{11}\,\mathrm{W}
\approx449.6\,\mathrm{GW}.
\]

That is within roughly 10% of Birch's “around 500 GW” value. Since both the force and source estimates are rounded, this is an arithmetic reproduction of the order of magnitude, not evidence that the force estimate, the sheet model, or the system boundary is realistic.

## Cryogenic pair audit

The source numbers transcribed from the two passages are:

| Passage | Cold-side heat load | Cryostat electrical power | Ratio |
|---|---:|---:|---:|
| p. 483 paragraph | \(100\,\mathrm{MW}\) | \(10\,\mathrm{GW}\) | 100 |
| p. 485 Table 8 | \(500\,\mathrm{MW}\) | \(50\,\mathrm{GW}\) | 100 |

Table 8 notes a conservative estimate with 300 K sink temperature and multilayer vacuum insulation. Its columns distinguish orbital ring and geosynchronous ring; the row's scope and the relationship between the tabulated value and total ORS length must be explicitly reconstructed before using either number for current sizing. The matching ratios do not prove that the underlying configurations are identical.

A September 2026 Canadian Institute for Exotic Science brief repeats 50 GW and labels itself “Not independently reviewed”. It is useful as a lead for literature triage, not as independent validation: https://www.instituteforexoticscience.ca/content/frontier/fr-orbital-rings.html

## Conditional stored-energy scale

Birch's Table 8 gives a total ORS mass of \(5.6\times10^8\,\mathrm{kg}\), while the paper uses \(8\times10^3\,\mathrm{m/s}\) as a speed in its drag calculation. If, only for a scale check, the entire quoted mass is assigned that inertial speed magnitude,

\[
E_k = \frac12 Mv^2
= \frac12(5.6\times10^8)(8\times10^3)^2
=1.792\times10^{16}\,\mathrm{J}
=17.92\,\mathrm{PJ}.
\]

This is not a failure-release, impact-damage, or structural-load prediction. Counter-rotation can affect net vector momentum, but it does not cancel the positive kinetic energy of each moving component. A fault model still needs to specify how energy and momentum are redistributed among the stream, support structure, payload, electromagnetic fields, debris, and other bodies.

## Reproducibility and evidence state

Run from the repository root:

\`\`\`sh
python3 scripts/engineering/orbital_ring_power_source_audit.py
\`\`\`

The standard-library script computes the stated ratios and drag coefficients and checks the smaller and larger drag-power estimates. The recorded run passed **12/12 checks** and emitted byte-identical JSON on repeat.

- Source Git blob: \`63dd1221598644d745416476c149655906f7ab59\`
- Source SHA-256: \`d12fe55f532369cdb6da009b775d3849f1219c4e1aa0db22ea296eff3cb723e5\`
- Deterministic JSON SHA-256: \`02c3abbb45ad706c6d692a0a248bbf0d8d09cf26e7575b169af16176b77d0ee8\`

This evidence verifies only the arithmetic of transcribed source values. It does not establish a valid current design, cryocooler performance, heat-rejection feasibility, structural stability, safe failure behavior, costs, or buildability.

## Next closure artifacts

1. Reconstruct the geometry/length basis of Table 8 and annotate which cryogenic load applies to which physical subsystem.
2. Model conducting-sheet eddy-current drag, superconducting track alternatives, and linear-motor cancellation as separate architectures with explicit boundaries.
3. Include energy recovery and control overhead rather than representing the approximate drag power as a mandatory net electrical demand.
4. Carry the resulting source assumptions and uncertainty intervals into the T0/T1 manifest before comparing power, throughput or lifecycle costs.
