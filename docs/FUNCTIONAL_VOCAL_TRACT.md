# Symthaea Functional Vocal Tract

Symthaea now has an experimental singing backend whose vocal identity is a
compact, inspectable physiology rather than a recorded speaker embedding. It is
intended as the controllable physical branch of a hybrid singer; it is not yet
claimed to be release-quality by itself.

The versioned computational-physiology successor—including intention-relative
gestures, identity anatomy/physiology, `PhysicalTractFrame`, the passive
oral/nasal v2 solver, the reference solver and isolation experiments—is
documented in `VOCAL_PHYSIOLOGY_ARCHITECTURE.md`. This document retains the
24-section baseline history and data boundary.

## Signal path

```text
lyrics + Muse notes
  -> IPA/ARPABET timing + expression trajectory
  -> 16,384D HDC phoneme/cognitive representation
  -> CfC/LTC motor controller at 200 Hz
  -> direct ridge-fitted articulatory head
  -> 24-section area function A(x,t)
  -> bounded log-area and source-control trajectory smoothing
  -> glottal flow + Kelly-Lochbaum scattering waveguide at 48 kHz
  -> optional bounded detail-only residual network
  -> WAV / objective metrics / blind human gate
```

The runtime path now projects the post-CfC latent state directly to 24 tube
areas, velum, glottal aperture, turbulence and constriction position. The old
formant-to-area adapter remains only as deterministic initialization supervision
and a comparison fallback; it is not used to generate the live direct tract.

The procedural identity contains tract length, oral/pharyngeal scaling, lip
area, open quotient, aspiration and glottal spectral tilt. It is separable from
phoneme and expression control, serializable, and not fitted to any named
person. `functional_tract_singing_gallery` writes three starting identities and
their JSON descriptions for listening comparisons.

The renderer preserves phase, waveguide, radiation and noise-filter state across
motor frames. Section areas interpolate in the log domain; pitch, energy,
voicing, glottal aperture and turbulence have parameter-specific audio-rate
smoothing; and a moving turbulence source crossfades continuously between tube
sections. The nonlinear glottal source is evaluated at 4x while the physical
tract remains at 48 kHz. Its opening and closing branches are value- and
slope-continuous, and aspiration/frication noise is band-limited before tube
injection.

The current July 19 continuity run passes the pitch, timing, physical-stability
and render-cleanliness gates for Silk, Luminous and Velvet. Relative to the
original gallery, worst single-sample jumps fell from 0.358-0.397 to
0.100-0.106, contextual click events fell to zero, and stable-note p95 pitch
error is 23-28 cents. Velvet remains the procedural production default. This is
an objective engineering gate, not a claim of human-perceived Muse quality.

`objective_report.json` now separates the four gates and records contextual
click outliers, stable/transition pitch errors, broadband first-difference
energy, reflection-limit hits, non-finite output and waveguide-energy jumps.
`controls.json` captures the physical trajectory, and `--reuse-controls` makes
renderer iteration independent of the expensive HDC/CfC motor pass.

## What should make it Muse-quality

The physical branch supplies phase-coherent pitch, interpretable articulation,
and continuous breath/phonation control. A small residual neural decoder should
add only the radiation, subglottal and high-frequency detail the tube model
misses. `ResidualDetailModel` implements that boundary: it is causal,
high-pass/DC-rejected, schema-limited to at most 20% of the carrier envelope,
and has no access to pitch, lyrics, timing, identity or tract state. It is
disabled until separately trained weights pass the same blind gate.

Promotion is empirical: the functional backend is included in
`muse_vocal_release_gate`. It must pass objective checks and the same concealed,
multi-listener naturalness, emotional-fit, identity-consistency, artifact and
word-comprehension thresholds as the other backends.

## Free-data boundary

`communication/singing/free_corpora.json` catalogs VocalSet, PJS, GTSinger,
JVS-MuSiC and CSD with their use restrictions. The provisioner downloads only
the two corpora whose published licenses permit commercial use. These recordings
may inform speaker-normalized physiology and technique priors, but never the
procedural identity. A Creative Commons copyright license also does not grant a
performer's publicity, privacy or personality rights.

Provision reference data outside the repository:

```bash
python communication/singing/provision_free_corpora.py vocalset-1.2 pjs --extract
```

Every provisioned archive gets a source/license/hash receipt and an explicit
`identity_training_allowed: false` marker.

`extract_functional_priors.py` samples archives without extracting them and
writes only anonymous aggregate distributions. The committed
`functional_priors.v1.json` currently summarizes 115 recordings (846 seconds)
across 15 technique groups; it contains neither recording nor speaker IDs.
