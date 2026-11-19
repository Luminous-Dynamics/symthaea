# FEEDIT candidate index — metadata only

**Audit date:** 2026-11-18  
**Source:** [FEEDIT — Field Edit archive](https://www.feedit.info/)  
**Source-stated scope:** 11 formations, 1,195 rendered elements, 1990–2008  
**Data licence stated by source:** CC BY-SA 4.0. Referenced aerial photographs remain separately copyrighted; none are included here.

This index records public plate-page metadata and data-quality caveats. It does **not** reproduce FEEDIT's machine-readable parametric geometry records. The source website says those JSON records contain geometry in metres and are released under CC BY-SA 4.0; the website currently says the GitHub repository link will appear at launch.

## Critical benchmark distinction

FEEDIT is useful as a parametric geometry source, but the rendered-element count is not necessarily the count of independent observations:
- Windmill Hill Triple Julia: 94 rendered elements from 32 declared elements; one arm is declared then repeated three times.
- Woodborough Hill sunflower: 310 rendered elements from 16 declared elements; a compact rule generates a 14 × 44 lattice.
- Milk Hill Galaxy: 409 rendered elements from 69 declared; the page says satellite elements are representative and the source survey covers major circles.
- Bythorn Mandala: 23 rendered from 14 declared.

These are valuable *generator hypotheses* and data augmentation specifications. They must not be treated as 310, 409 or 94 independently annotated ground-truth elements. The dataset also states that most digitization was performed by `claude-fable-5` (August 2026); independent human replication and measurement covariance are not provided in the metadata audited.

The Stonehenge Julia Set page is a stronger candidate for independent circle-level testing because it says 125 circles were individually digitized from an aerial record. That statement still requires independent re-digitization from rights-cleared source imagery before accuracy can be evaluated.

## Candidate records

| Plate | Formation | Reported date | Span | Symmetry | Rendered / declared | Notes |
|---:|---|---|---:|---|---:|---|
| 01 | Alton Barnes double pictogram | 1990-07-12 | 125 m | 1-fold | 28 / 28 | Element-level geometry |
| 02 | Barbury Castle tetrahedron | 1991-07-17 | 80 m | 1-fold | 24 / 24 | Triangle / rings / corner motifs |
| 03 | Ickleton Mandelbrot | 1991-08-12 | 46 m | 1-fold, mirror | 10 / 10 | Some scales/gaps estimated from imagery |
| 04 | Bythorn Mandala | 1993-09-04 | 44 m | 5-fold, mirror | 23 / 14 | Declared/rendered mismatch |
| 05 | Stonehenge Julia Set | 1996-07-07 | 152 m | 1-fold | 125 / 125 | Page says individually placed circles |
| 06 | Windmill Hill Triple Julia | 1996-07-29 | 150 m | 3-fold | 94 / 32 | One arm repeated at 120° |
| 07 | Barbury Castle Tree of Life | unresolved | — | — | — | Page returned 404 during audit |
| 08 | Milk Hill Koch snowflake | 1997-08-08 | 71 m | 6-fold, mirror | 99 / 99 | Repeated lace circles |
| 09 | Woodborough Hill sunflower | 2000-08-13 | 70 m | 22-fold, mirror | 310 / 16 | Parametric lattice expansion |
| 10 | Milk Hill Galaxy | 2001-08-12 | 274 m | 6-fold | 409 / 69 | Major-circle scaffold; satellites representative |
| 11 | Barbury Castle Pi | 2008-06-01 | 46 m | 1-fold | 31 / 31 | Described Pi path is a source interpretation |

For exact metadata and page URLs, see `records.json`.

## Recommended use

1. Treat each formation as one event-level group.
2. Separate traced/individually placed primitives from repeated, mirrored, copied or procedurally generated primitives.
3. Use parametric records to test reconstruction fidelity and generator compression, not to claim that the field was built using that generator.
4. For event-held-out validation, keep all elements and source images from one physical event in one fold.
5. Obtain the actual CC BY-SA 4.0 JSON geometry record and preserve its licence/attribution before numerical ingestion. Do not scrape third-party photos.
