# Atelier Lightshell — Test Method Applicability Contract v0.1

## Purpose

This contract prevents method names from being treated as universal permission to test a textile specimen. Every method selection must be evaluated against its exact edition, stated scope, construction class, tested surface, seam geometry and fixture.

Official standard metadata is recorded for interoperability and method selection only. The repository must not reproduce licensed standard text; the laboratory must use its licensed copy and competent procedures.

## Source-reviewed method crosswalk

| Method | Current edition pinned in v0.1 | Use | Important scope boundary |
| --- | --- | --- | --- |
| ISO 13935-2 | 2026 | Seam tensile maximum force, CRE grab | Straight sewn seams; normally not nonwovens or coated fabrics; not a curved-seam method |
| ISO 12947-2 | 2016 | Martindale textile specimen breakdown | Excludes coated/laminated fabrics |
| ISO 5470-2 | 2021 | Wet/dry Martindale abrasion of coated surfaces | Does not apply to the uncoated face |
| ISO 9237 | 1995 | Fabric air permeability | Includes many technical fabrics, nonwovens and made-up articles; thick 3D spacer fixture still needs validation |
| ISO 6330 | 2021 | Domestic washing/drying procedures | Requires exact machine, detergent, wash and drying selection; does not establish robot-specific industrial cleaning resistance |

Edition metadata was checked against official ISO catalogue records in November 2026. ISO 13935-2:2026 was published in April 2026 and replaces the 2014 edition. ISO 12947-2:2016 and ISO 9237:1995 were confirmed in 2023; ISO 5470-2:2021 was confirmed in 2026. Always recheck the ISO catalogue before freezing a future method profile.

## Why automatic scope checks matter

- A coated textile and an uncoated textile surface are not interchangeable abrasion test subjects. ISO 12947-2 excludes coated/laminated fabrics; ISO 5470-2 is the coated-surface alternative and does not test the uncoated face.
- A straight seam tensile standard cannot automatically be applied to a curved articulated transition. ISO 13935-2:2026 is expressly a straight-seam CRE grab method and is normally not applicable to nonwovens or coated fabrics.
- A flat textile air-permeability fixture can compress a thick spacer and alter the structure being measured. A spacer-specific fixture must be validated before a spacer result can become ready for testing.
- ISO 6330's domestic wash/dry cycles are not a substitute for the cleaning chemistry, exposure, temperature, pressure or drying process encountered by a particular robot garment.

## Contract semantics

Method profile identity binds method ID, exact edition, official metadata source and family. Standard profiles are pinned to the source-reviewed set in this v0.1 crosswalk. Unknown standards remain blocked pending research review. Internal methods use a `luminous:` namespace, require independent internal validation and must never masquerade as ISO methods.

Each selection binds the specimen, profile, selected surface, fixture class/identity/revision and fixture validation. Known scope conflicts fail closed. Modified methods require an approved deviation and cannot be labelled unmodified standard conformance. A selection can be `proposed` without having a fixture; `ready_for_test` requires fixture validation, protocol and acceptance-rule references.

## P0 negative controls

Authored tests include:
- ISO 13935-2:2026 with curved seams, nonwovens or coated fabrics;
- ISO 12947-2:2016 with coated/laminated specimens;
- ISO 5470-2:2021 with an uncoated surface;
- ISO 9237:1995 on a thick spacer without a spacer-qualified fixture;
- stale ISO 13935-2:2014 edition;
- ISO 6330:2021 with robot-specific cleaning rather than a domestic wash profile;
- internal method impersonating an ISO method;
- modified standard method without approved deviation;
- ready-for-test selection without exact fixture validation and protocol.

## Benchmarks

- zero false acceptance of the fixed known scope-conflict corpus;
- 100% exact method-edition and fixture-revision binding before testing;
- 100% of deviations explicitly marked and approved;
- quantify bias between flat and spacer-qualified air-permeability fixtures on matched specimens;
- compare replicate dispersion and inter-laboratory reproducibility only under genuinely matched fixtures and protocol conditions.

## Claim boundary

This is an applicability/admission helper, not the licensed normative text, a complete standard implementation, accreditation, a laboratory scope or a conformity assessment. `ready_for_test` means only that the method selection has the required references and known scope checks; it does not mean any test passed.

## Official references

- [ISO 13935-2:2026](https://www.iso.org/standard/88344.html).
- [ISO 12947-2:2016](https://www.iso.org/standard/61058.html).
- [ISO 5470-2:2021](https://www.iso.org/standard/77552.html).
- [ISO 9237:1995](https://www.iso.org/standard/16869.html).
- [ISO 6330:2021](https://www.iso.org/standard/75934.html).
- Schwager et al. (2022), [Spacer-fabric test equipment and methods](https://doi.org/10.1007/s10443-021-09959-y).

## Current state

The example is synthetic and in `proposed` state. It records five method profiles and proposed selections, including a spacer airflow test that is deliberately not ready until its fixture is validated. No physical tests, machine capability or laboratory accreditation are claimed.