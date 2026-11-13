# ZA-CLICK-001 — synthetic multi-source click gesture compiler

A permission-free symbolic articulatory gesture benchmark for click-bearing performance events.

This tool generates **symbolic timing fixtures only**. It does not synthesize realistic clicks, provide a pronunciation dictionary, model a specific person, or establish native-language correctness.

## Why a separate gesture representation?

A typical click has a front oral closure and a simultaneous posterior closure. Lowering the tongue body changes the pressure in the trapped oral cavity; releasing the front closure produces the characteristic ingressive click event. The posterior/oral airstream mechanism is distinct from optional laryngeal voicing and velic nasal coupling.

The three place families represented here are dental, post-alveolar and lateral. The profile then represents voicing, nasal coupling, aspiration and prenasal lead as separate controls rather than hiding them inside one token.

This matches the architectural distinction described in UCL phonetics materials and the articulatory study of isiZulu clicks. The latter reports that front-release characteristics differ by click place and that click gestures coarticulate with following vowel context.

## Run

From the repository root:

~~~bash
cargo run --manifest-path tools/za-click-gesture-bench/Cargo.toml -- /tmp/za-click-gesture-output
~~~

The tool writes:
- gesture-manifest.json
- gesture-manifest.sha256
- gesture-timelines.txt

The manifest identifies all inputs as synthetic symbolic gestures, contains explicit claim boundaries, and records source/configuration identity.

## Event model

Each click event has independently identified intervals for:
- front closure;
- posterior closure;
- tongue-body rarefaction;
- front release;
- velaric-ingressive burst;
- optional pulmonic/laryngeal voicing;
- optional velic nasal coupling;
- optional post-release aspiration;
- optional prenasal lead;
- following vowel nucleus and its independent F0/tone trajectory.

The click's place, oral airstream, laryngeal source, velic coupling, aspiration and vowel tone are not conflated.

The symbolic v1 validator enforces:
- sufficient overlap between front and posterior closures;
- rarefaction inside the double-closure interval;
- front release after rarefaction and closure;
- ingressive burst after release;
- independent accompaniment requirements;
- aspiration in the post-release/pre-vowel interval;
- prenasal lead timing;
- tone/melody points inside the vowel nucleus, never on the click transient.

## Fixture coverage

The generator creates 18 known-answer symbolic cases: three places × six accompaniment profiles (plain, aspirated, voiced, nasalized, prenasalized, voiced-nasalized).

The absolute timings are synthetic fixture-policy values for testing event order and interval constraints. They are not measured native-speaker durations and must not be presented as phonetic norms.

## Critical limitation

The legacy formant source-filter renderer is not a physically faithful click synthesizer. A click's velaric ingressive mechanism cannot be made authentic by passing a click label into a conventional glottal pulse/formant renderer.

ZA-CLICK-001 therefore qualifies symbolic gesture compilation only. It does not make claims about:
- realistic click audio;
- native isiZulu/isiXhosa pronunciation;
- lexical tone correctness;
- singing naturalness or intelligibility;
- cultural authenticity or speaker identity.

The orthographic hints c/q/x are only fixture metadata for the three click-place families. They do not provide broad G2P coverage.

## Research sources

- UCL phonetics teaching material, *Sounds of the World's Languages*: double closure, cavity rarefaction, front release, velaric ingressive airflow and optional pulmonic/velic accompaniment. https://www.phon.ucl.ac.uk/courses/spsci/SSC_talking/material/week_03/sounds-of-the-worlds-languages.pdf
- *Coproduction and Coarticulation in IsiZulu Clicks*, UC eScholarship: place-specific release and vowel-context coarticulation. https://escholarship.org/uc/item/4vv5373d
- Mahura & Pascoe, *Zulu/isiZulu speech development*, Oxford Handbook (2026): 59 consonants including clicks and prenasalized consonants, seven vowels, tonal language. https://doi.org/10.1093/9780191964909.003.0077

These references ground the need for a gesture model; the generated fixtures are not a reproduction of their human data.
