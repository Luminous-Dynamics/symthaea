# Historical geometry reference: Birch’s 300 km and 600 km cases

**Status:** source-anchored T0 arithmetic only. This note does not select an orbital-ring geometry or establish a feasible ORS.

## Primary-source anchor

Paul Birch’s *Orbital Ring Systems and Jacob’s Ladders — I* (1982), printed p. 476, says he will use 300 km and 600 km as representative suitable ORS heights in the ladder analysis. These are **representative source cases**, not a specification that a real continuous-skyhook orbital-ring path is a perfect circle at either altitude. [Scanned primary source (PDF)](https://space.nss.org/wp-content/uploads/Orbital-Rings.pdf).

The 600 km idealized comparison uses the same declared point-mass reference constants as the T0 checker: Earth equatorial reference radius \(R_E=6{,}378.137\,\mathrm{km}\) and geocentric gravitational constant \(\mu=3.986004418\times10^{14}\,\mathrm{m^3/s^2}\). The National Geospatial-Intelligence Agency lists both corresponding WGS 84 constants, including \(a=6{,}378{,}137.0\,\mathrm{m}\) and \(GM=3.986004418\times10^{14}\,\mathrm{m^3/s^2}\): [NGA WGS 84 defining parameters](https://earth-info.nga.mil/GandG/wgs84/gravitymod/egm2008/index.html).

## Circular two-body reference values

| Quantity | 300 km reference | 600 km reference |
|---|---:|---:|
| Altitude above reference radius | 300 km | 600 km |
| Geocentric radius | 6,678.137 km | 6,978.137 km |
| Circle circumference | 41,959.972 km | 43,844.928 km |
| Circular speed | 7.725760 km/s | 7.557865 km/s |
| Period | 90.519619 min | 96.687196 min |
| Specific kinetic energy | 29.843686 MJ/kg | 28.560663 MJ/kg |
| Specific mechanical energy | −29.843686 MJ/kg | −28.560663 MJ/kg |

The mechanical-energy column includes both kinetic and gravitational potential terms; it is not a system-wide stored-energy estimate for an orbital ring. These values come from the idealized relations \(r=R_E+h\), \(C=2\pi r\), \(v=\sqrt{\mu/r}\), \(T=2\pi\sqrt{r^3/\mu}\), and \(e_k=v^2/2\). They are reference calculations, **not values transcribed from Birch’s tables**. The model omits Earth oblateness, rotation, drag, third bodies, and all ring/support/rider dynamics.

The reference checker now includes a 600 km fixture with source metadata and a source-bound regression. Schema v3 also reports and cross-checks specific orbital mechanical energy \(\epsilon = e_k-\mu/r=-\mu/(2r)\) for circular two-body motion. It has **66 expected checks**, compared with 47 on the prior checked source. The new source has not yet been runtime-verified; the isolated workflow is configured to run it twice, require byte-identical output, and archive the raw result and source fingerprints. Do not cite the expected count as a pass.

## Relationship to Table 8

Birch’s Table 8 (printed p. 485) gives superconductor mass and linear-mass entries for the precession/orbital-ring and support portions separately. Dividing the rounded entries gives about 44.17 Mm for \(5.3\times10^7\,\mathrm{kg}/(1.2\,\mathrm{kg/m})\) and 44.00 Mm for \(8.8\times10^7\,\mathrm{kg}/(2.0\,\mathrm{kg/m})\). Applying the table’s 10 W/m loss only to the support entry gives 440 MW, about 12% below the rounded 500 MW total.

The inferred 44.17 Mm and 44.00 Mm are **conductor lengths derived from rounded mass-per-length entries**, not explicitly stated orbit circumferences. A circular-equivalent comparison puts those lengths near a 600 km circle (whose circumference is 43.845 Mm; the length differences are about 0.7% and 0.35%), but equality between conductor length and orbital centerline circumference is not established by the table alone. This is a consistency clue for follow-up source reconstruction—not grounds to freeze the scenario radius or declare its geometry solved.

## Scenario boundary

Do not populate F-SKYHOOK-LOAD-STEP from this note. Its 23 inputs remain null because the selected inertial frame realization, orientation, epoch, initial-equilibrium evidence, support/rider model, and conservation boundaries remain unresolved. The 600 km case is a reference fixture to check arithmetic and sensitivity, not an engineering assumption approved for the failure scenario.
