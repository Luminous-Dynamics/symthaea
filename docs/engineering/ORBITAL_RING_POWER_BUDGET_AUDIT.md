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

The two cryogenic pairs each imply a roughly 100:1 ratio of wall-plug input to the stated cold-side load, but the passages describe different configurations and should not automatically be treated as one baseline. Table 8 is specifically for a continuous-skyhook ORS. The text immediately before it says the orbital rings do not need active cryogenic equipment because they sit inside the fixed ring's vacuum flask; insulation is only needed on the outside, and the cryostats are stationary on the continuous skyhook. The table accordingly leaves the orbital-ring cryogenic-loss cell blank and puts 10 W/m under the geosynchronous/support column.

Table 8's superconductor mass-per-ring divided by superconductor mass-per-length gives about 44.17 Mm for the precession/orbital-ring entry (5.3 × 10⁷ kg / 1.2 kg m⁻¹) and 44.00 Mm for the support entry (8.8 × 10⁷ kg / 2.0 kg m⁻¹). Apply the 10 W/m row to the support-column length only: 44.00 Mm × 10 W/m = **440 MW**, about 12% below the rounded 500 MW total. Treating that support length as a circular centerline gives a radius of about 7,003 km, or an equivalent altitude of about **625 km** using the WGS-84 equatorial radius. Its circumference is only **0.35% longer** than a nominal 600 km circular-path circumference, a representative ORS height discussed earlier in Birch's paper. This indicates the Table 8 entries are internally consistent to their stated precision and broadly compatible with that reference geometry. It does not prove the exact physical layout or explain the earlier, different superconducting-surface estimate of 100 MW / 10 GW; that discrepancy may reflect scope/configuration differences and remains to be reconstructed. Do not apply the 10 W/m loss to an average of columns or state an exact altitude from the table alone.

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

The 600 km comparison is a geometric consistency reference only, not proof that Table 8 specifies an exact 600 km altitude.

Run from the repository root:

```sh
python3 scripts/engineering/orbital_ring_power_source_audit.py
```

The standard-library script computes the drag coefficients, checks the smaller and larger drag-power estimates and paired cryogenic ratios, applies cryogenic leakage to the support/geosynchronous column only, reconstructs the Table 8 support length and heat-load scale, and compares the implied circle with a 600 km reference path. The updated source was executed locally: **20/20 checks passed**, `py_compile` succeeded, and the JSON output was byte-identical on repeat. The exact output is preserved in the evidence directory. The captured report is checked into [the evidence directory](evidence/orbital_ring_power_source_audit.json) for review.

- Source Git blob: `b6406e391e49d53cd077cf08b3d8691c40472611`; source SHA-256: `80c716c0f374de4acb706a7aac5ca467e00e285dcb99e6f97b403f90eb9c6a0b`
- Exact output SHA-256: `d04091d63db550b2ce771a6d06fbbfe61998ce2c3806488dea53bd12e39e8165`; captured JSON report Git blob: `0dd0e85ce020e874163d7dcc63fa0e209d4c3558`

This evidence verifies only the arithmetic of transcribed source values. It does not establish a valid current design, cryocooler performance, heat-rejection feasibility, structural stability, safe failure behavior, costs, or buildability.

## Next closure artifacts

1. Confirm Table 8's precise system boundary and configuration; its support-column line-mass data imply a ~625 km circular-equivalent altitude and about 440 MW at 10 W/m, broadly compatible with the 600 km reference case. Explain whether the p. 483 estimate describes a different superconducting-surface configuration.
2. Model conducting-sheet eddy-current drag, superconducting track alternatives, and linear-motor cancellation as separate architectures with explicit boundaries.
3. Include energy recovery and control overhead rather than representing the approximate drag power as a mandatory net electrical demand.
4. Carry the resulting source assumptions and uncertainty intervals into the T0/T1 manifest before comparing power, throughput or lifecycle costs.
