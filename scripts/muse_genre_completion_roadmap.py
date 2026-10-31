#!/usr/bin/env python3
"""Build/check a source-bound implementation backlog for every Muse catalog style."""
from __future__ import annotations
import argparse,json,re,sys,unicodedata
from pathlib import Path
from typing import Any
from muse_style_coverage import CATALOG_RELATIVE,CoverageError,build_current_report
ROOT=Path(__file__).resolve().parents[1]
ROADMAP_RELATIVE=Path("crates/domains/symthaea-muse/genre-completion-roadmap-v1.json")
WORKSTREAMS=json.loads(r'''{
  "ClassicalLyricCharacter": {
    "label": "Classical Lyric & Character",
    "priority": "P2",
    "candidate_engine_track": "character_piece_grammar",
    "hypothesis": "Differentiate character pieces by phrase, cadence, meter and motif.",
    "next_gate": "M0d_measure_existing_routes_then_prototype_elegy",
    "cultural_review_recommended": false
  },
  "BaroqueEarlyDance": {
    "label": "Baroque and Early Dance Forms",
    "priority": "P1",
    "candidate_engine_track": "baroque_dance_and_suite",
    "hypothesis": "Represent dance-specific pickups, gait, phrase pairing, reprises and suite order.",
    "next_gate": "prototype_one_dance_form_with_phrase_and_meter_contract",
    "cultural_review_recommended": false
  },
  "DevelopmentalLargeForms": {
    "label": "Developmental and Large Forms",
    "priority": "P1",
    "candidate_engine_track": "large_form_obligation_engine",
    "hypothesis": "Use form obligations and transitions for rondo, variations, scherzo and concerto.",
    "next_gate": "declare_section_obligations_and_counterexample_fixtures",
    "cultural_review_recommended": false
  },
  "ContrapuntalPolyphonic": {
    "label": "Contrapuntal and Polyphonic",
    "priority": "P1",
    "candidate_engine_track": "contrapuntal_obligation_engine",
    "hypothesis": "Separate subject entry, imitation, voice independence and closure by form.",
    "next_gate": "add_per_form_voice_entry_and_independence_contracts",
    "cultural_review_recommended": false
  },
  "GroundOstinatoTransformation": {
    "label": "Ground, Ostinato and Transformation",
    "priority": "P1",
    "candidate_engine_track": "ground_transformation_engine",
    "hypothesis": "Separate invariant ground/cell from upper-voice transformation.",
    "next_gate": "test_ground_identity_and_transformation_invariants",
    "cultural_review_recommended": false
  },
  "SongNarrative": {
    "label": "Song and Narrative",
    "priority": "P1",
    "candidate_engine_track": "song_form_engine",
    "hypothesis": "Separate strophic, verse-chorus, through-composed, art-song and work-song.",
    "next_gate": "declare_song_section_and_vocal_role_contract",
    "cultural_review_recommended": false
  },
  "BluesGospelSoul": {
    "label": "Blues, Gospel and Soul",
    "priority": "P1",
    "candidate_engine_track": "blues_gospel_soul_tracks",
    "hypothesis": "Separate blues response, gospel escalation, soul build, R&B groove and Neo-Soul pocket.",
    "next_gate": "add_track_specific_groove_and_phrase_metrics",
    "cultural_review_recommended": true
  },
  "JazzImprovisatory": {
    "label": "Jazz and Improvisatory Forms",
    "priority": "P1",
    "candidate_engine_track": "jazz_style_grammar_tracks",
    "hypothesis": "Separate swing, bebop, cool/hard bop, modal vamp, jazz waltz and open form.",
    "next_gate": "define_style_specific_phrase_harmony_and_swing_metrics",
    "cultural_review_recommended": true
  },
  "LatinCaribbeanCycles": {
    "label": "Latin and Caribbean Cycles",
    "priority": "P1",
    "candidate_engine_track": "regional_clave_and_dance_cycles",
    "hypothesis": "Distinct cycle/accent roles for son, salsa, mambo, cha-cha, samba and bossa nova.",
    "next_gate": "define_cycle_phase_accent_and_phrase_contracts",
    "cultural_review_recommended": true
  },
  "EuropeanNorthAtlanticFolk": {
    "label": "European and North Atlantic Folk",
    "priority": "P1",
    "candidate_engine_track": "regional_folk_tune_and_dance_tracks",
    "hypothesis": "Separate regional meter, strain, tune-set and ornament.",
    "next_gate": "prototype_regional_tune_set_with_meter_and_strain_rules",
    "cultural_review_recommended": true
  },
  "MediterraneanMiddleEasternNorthAfrican": {
    "label": "Mediterranean, Middle Eastern and North African Modal",
    "priority": "P0",
    "candidate_engine_track": "multiple_modal_song_and_improvisation_tracks",
    "hypothesis": "Keep Fado, maqam, makam, dastgah, nuba, taqsim and rebetiko separate.",
    "next_gate": "partition_tracks_and_define_pitch_hierarchy_and_temporal_contracts",
    "cultural_review_recommended": true
  },
  "SouthAsianRagaTala": {
    "label": "South Asian Raga and Tala",
    "priority": "P0",
    "candidate_engine_track": "multiple_raga_tala_and_song_tracks",
    "hypothesis": "Keep Hindustani, Dhrupad, Carnatic, RTP, bhajan, ghazal and qawwali distinct.",
    "next_gate": "separate_hindustani_and_carnatic_contracts_with_expert_review",
    "cultural_review_recommended": true
  },
  "EastSoutheastAsian": {
    "label": "East and Southeast Asian Traditions",
    "priority": "P0",
    "candidate_engine_track": "multiple_tradition_specific_ensemble_tracks",
    "hypothesis": "Separate Gagaku, honkyoku, min'yo, guqin, sizhu, Gugak, Gamelan and Piphat.",
    "next_gate": "one_tradition_at_a_time_source_model_then_expert_review",
    "cultural_review_recommended": true
  },
  "AfricanDiasporicGroove": {
    "label": "African and Diasporic Groove",
    "priority": "P0",
    "candidate_engine_track": "separate_regional_groove_tracks",
    "hypothesis": "Separate Afrobeat, Highlife, Juju, Soukous, Amapiano, Gqom, Reggae and Dub.",
    "next_gate": "measure_style_specific_bass_drum_percussion_and_phrase_roles",
    "cultural_review_recommended": true
  },
  "MinimalProcessExperimental": {
    "label": "Minimal, Process and Experimental",
    "priority": "P1",
    "candidate_engine_track": "process_rule_and_chance_tracks",
    "hypothesis": "Expose rules for phase, additive process, spectral change, chance forms and cellular generation.",
    "next_gate": "formalize_process_state_and_reproducibility_receipts",
    "cultural_review_recommended": false
  },
  "AmbientElectronicTexture": {
    "label": "Ambient and Electronic Texture",
    "priority": "P1",
    "candidate_engine_track": "textural_horizon_and_automation_tracks",
    "hypothesis": "Separate drone, sequencer, metric mutation, glitch, loop and acoustic/electronic processes.",
    "next_gate": "define_long_horizon_density_and_texture_metrics",
    "cultural_review_recommended": false
  },
  "ClubBeatMusic": {
    "label": "Club and Beat Music",
    "priority": "P0",
    "candidate_engine_track": "electronic_pattern_and_arrangement_tracks",
    "hypothesis": "Separate house, techno, trance, DnB, garage, breakbeat and hip-hop by groove and transition.",
    "next_gate": "implement_track_specific_grid_groove_and_section_transitions",
    "cultural_review_recommended": true
  },
  "PopRock": {
    "label": "Pop and Rock",
    "priority": "P0",
    "candidate_engine_track": "song_hook_and_band_arrangement_tracks",
    "hypothesis": "Differentiate verse/chorus, bridge, riff, dream pop, synthpop and post-rock arcs.",
    "next_gate": "define_section_hook_riff_and_arrangement_metrics",
    "cultural_review_recommended": false
  },
  "DramaticScreenStage": {
    "label": "Dramatic, Screen and Stage",
    "priority": "P1",
    "candidate_engine_track": "dramatic_cue_and_state_transition_engine",
    "hypothesis": "Tie scene, character, threat, world state and game state to musical obligations.",
    "next_gate": "define_cue_state_transition_and_reproducibility_contracts",
    "cultural_review_recommended": false
  },
  "SouthernAfricanUrbanFolkDance": {
    "label": "Southern African Urban, Folk and Dance Traditions",
    "priority": "P0",
    "candidate_engine_track": "separate_southern_african_style_tracks",
    "hypothesis": "Kwaito, Maskandi, Isicathamiya, Mbaqanga, Kwela, Marabi, Township Jazz and Bubblegum Pop need separate sourced definitions and review.",
    "next_gate": "expert_reviewed_sources_and_distinct_track_contracts_before_any_route",
    "cultural_review_recommended": true
  }
}''')
ENTRY_TRACK_OVERRIDES=json.loads(r'''{
  "Kwaito": "kwaito_groove_song_and_township_language_context",
  "Maskandi": "maskandi_instrumental_introduction_and_narrative_song",
  "Isicathamiya": "isicathamiya_vocal_polyphony_and_refrain",
  "Mbaqanga": "mbaqanga_vocal_instrumental_dance_song",
  "Kwela": "kwela_whistle_led_dance_cycle",
  "Marabi": "marabi_repeating_piano_vamp_and_dance_cycle",
  "Township Jazz": "township_jazz_ensemble_and_improvisation",
  "South African Bubblegum Pop": "south_african_bubblegum_pop_hook_and_dance_song",
  "Rondo": "rondo_refrain_return_contract",
  "Theme and Variations": "theme_variation_identity_and_transformation",
  "Two-Part Invention": "two_voice_independence_and_imitation",
  "Canon": "strict_imitation_and_interval_contract",
  "Chaconne": "chaconne_ground_identity_and_variation",
  "Arabic Maqam": "maqam_specific_pitch_pathway_and_ornament",
  "Carnatic Kriti": "carnatic_kriti_sections_and_sangatis",
  "Indonesian Gamelan": "gamelan_colotomic_cycle_and_interlocking_parts",
  "Amapiano": "amapiano_log_drum_bass_phrase_and_drop",
  "Gqom": "gqom_sparse_pressure_and_syncopation",
  "House": "house_four_on_floor_phrase_energy",
  "Techno": "techno_machine_process_and_microchange",
  "Pop": "pop_hook_and_verse_chorus",
  "Rock": "rock_riff_and_backbeat",
  "Adaptive Game Score": "game_state_transition_to_musical_obligation"
}''')
EXTENSION_CANDIDATES=json.loads(r'''[
  {
    "name": "Afrobeats",
    "area": "West African contemporary popular music",
    "priority": "P0",
    "rationale": "Distinct from Afrobeat; scholarship identifies a separate contemporary field.",
    "sources": [
      "https://doi.org/10.1080/00064246.2024.2370204"
    ]
  },
  {
    "name": "Afro-house",
    "area": "African electronic dance",
    "priority": "P0",
    "rationale": "Research a distinct house lineage and scene-specific groove.",
    "sources": [
      "https://musicbrainz.org/doc/Genre"
    ]
  },
  {
    "name": "Bacardi",
    "area": "Southern African contemporary dance",
    "priority": "P0",
    "rationale": "Candidate; sources and producer review must define boundaries.",
    "sources": [
      "https://musicbrainz.org/doc/Genre"
    ]
  },
  {
    "name": "Sgija",
    "area": "Southern African contemporary dance",
    "priority": "P0",
    "rationale": "Candidate contemporary dance substyle; define boundaries and overlap.",
    "sources": [
      "https://musicbrainz.org/doc/Genre"
    ]
  },
  {
    "name": "3-step",
    "area": "Southern African contemporary dance",
    "priority": "P0",
    "rationale": "Candidate; define terminology and rhythmic contract with practitioners.",
    "sources": [
      "https://musicbrainz.org/doc/Genre"
    ]
  },
  {
    "name": "Country",
    "area": "North American popular/folk",
    "priority": "P1",
    "rationale": "Missing parent genre; Country Blues is not general country.",
    "sources": [
      "https://musicbrainz.org/doc/Genre"
    ]
  },
  {
    "name": "Bluegrass",
    "area": "North American roots",
    "priority": "P1",
    "rationale": "Distinct instrumental roles, tune forms and ensemble interaction.",
    "sources": [
      "https://musicbrainz.org/doc/Genre"
    ]
  },
  {
    "name": "Funk",
    "area": "Popular groove",
    "priority": "P1",
    "rationale": "Bass/drum interlock distinct from R&B and Pop.",
    "sources": [
      "https://musicbrainz.org/doc/Genre"
    ]
  },
  {
    "name": "Disco",
    "area": "Dance/pop",
    "priority": "P1",
    "rationale": "Arrangement, orchestration and phrase lift separate from generic house.",
    "sources": [
      "https://musicbrainz.org/doc/Genre"
    ]
  },
  {
    "name": "Heavy Metal",
    "area": "Rock substyles",
    "priority": "P1",
    "rationale": "Riff, subdivision and drum articulation aren't explicit in generic Rock.",
    "sources": [
      "https://musicbrainz.org/doc/Genre"
    ]
  },
  {
    "name": "Punk",
    "area": "Rock substyles",
    "priority": "P1",
    "rationale": "Separate performance/arrangement hypothesis.",
    "sources": [
      "https://musicbrainz.org/doc/Genre"
    ]
  },
  {
    "name": "Dancehall",
    "area": "Caribbean popular",
    "priority": "P1",
    "rationale": "Separate riddim, phrase and production from Reggae.",
    "sources": [
      "https://musicbrainz.org/doc/Genre"
    ]
  },
  {
    "name": "Reggaeton",
    "area": "Latin urban",
    "priority": "P1",
    "rationale": "Own beat/pattern, bass and section model.",
    "sources": [
      "https://musicbrainz.org/doc/Genre"
    ]
  },
  {
    "name": "Grime",
    "area": "UK electronic/rap",
    "priority": "P1",
    "rationale": "Distinct tempo, rhythmic grid, phrase and production.",
    "sources": [
      "https://musicbrainz.org/doc/Genre"
    ]
  },
  {
    "name": "Drill",
    "area": "Hip-hop substyles",
    "priority": "P1",
    "rationale": "Rhythmic/bass behavior needs scene-specific research.",
    "sources": [
      "https://musicbrainz.org/doc/Genre"
    ]
  },
  {
    "name": "Dubstep",
    "area": "Electronic bass",
    "priority": "P1",
    "rationale": "Bass design, half-time groove and drop/space behavior.",
    "sources": [
      "https://musicbrainz.org/doc/Genre"
    ]
  },
  {
    "name": "Trip-Hop",
    "area": "Electronic/alternative",
    "priority": "P2",
    "rationale": "Check overlap with ambient, hip-hop and downtempo.",
    "sources": [
      "https://musicbrainz.org/doc/Genre"
    ]
  },
  {
    "name": "Chiptune",
    "area": "Electronic/game music",
    "priority": "P2",
    "rationale": "Constraint-led timbre needs synthesis/rendering support.",
    "sources": [
      "https://musicbrainz.org/doc/Genre"
    ]
  },
  {
    "name": "Shoegaze",
    "area": "Alternative rock",
    "priority": "P2",
    "rationale": "Texture, guitar layers and production need evidence.",
    "sources": [
      "https://musicbrainz.org/doc/Genre"
    ]
  },
  {
    "name": "Contemporary Gospel",
    "area": "Gospel substyles",
    "priority": "P2",
    "rationale": "Separate contemporary arrangement and choir production from Gospel.",
    "sources": [
      "https://musicbrainz.org/doc/Genre"
    ]
  },
  {
    "name": "Southern Gospel",
    "area": "Gospel substyles",
    "priority": "P2",
    "rationale": "Validate against Gospel/Spiritual with voice/ensemble evidence.",
    "sources": [
      "https://musicbrainz.org/doc/Genre"
    ]
  },
  {
    "name": "Arabic Pop",
    "area": "Middle Eastern popular",
    "priority": "P2",
    "rationale": "Do not infer pop song form from maqam.",
    "sources": [
      "https://musicbrainz.org/doc/Genre"
    ]
  },
  {
    "name": "Gnawa",
    "area": "North African traditions",
    "priority": "P2",
    "rationale": "Specific cycles, instruments and performance context need sources.",
    "sources": [
      "https://musicbrainz.org/doc/Genre"
    ]
  },
  {
    "name": "Regional Mexican / Norteño",
    "area": "Mexican regional popular",
    "priority": "P2",
    "rationale": "Candidate accordion-led song/dance family; resolve boundaries.",
    "sources": [
      "https://musicbrainz.org/doc/Genre"
    ]
  },
  {
    "name": "Bhangra",
    "area": "Punjabi popular/dance",
    "priority": "P2",
    "rationale": "Distinct dance-rhythm and ensemble track.",
    "sources": [
      "https://musicbrainz.org/doc/Genre"
    ]
  },
  {
    "name": "Taarab",
    "area": "East African coastal traditions",
    "priority": "P2",
    "rationale": "Specific regional song/ensemble practice.",
    "sources": [
      "https://musicbrainz.org/doc/Genre"
    ]
  },
  {
    "name": "Benga",
    "area": "East African popular music",
    "priority": "P0",
    "rationale": "A named Kenyan/East African guitar-led popular tradition missing from the current registry; define period, region and ensemble roles before implementation.",
    "sources": [
      "https://doi.org/10.1017/S0261143000003524",
      "https://doi.org/10.21504/amj.v9i4.1888"
    ]
  },
  {
    "name": "Chimurenga",
    "area": "Zimbabwean popular music",
    "priority": "P0",
    "rationale": "A Zimbabwean music field with historical and political context; do not reduce it to generic guitar-band patterns.",
    "sources": [
      "https://doi.org/10.1017/S0261143000003524",
      "https://www.jstor.org/stable/j.ctt17w8gb6"
    ]
  },
  {
    "name": "Mbalax",
    "area": "Senegalese popular music",
    "priority": "P0",
    "rationale": "A distinct Senegalese popular-music trajectory; research the relation between local rhythmic practice, song and modern production.",
    "sources": [
      "https://doi.org/10.11606/issn.1983-6023.sank.2016.119064"
    ]
  },
  {
    "name": "Apala",
    "area": "Yoruba/Nigerian popular music",
    "priority": "P1",
    "rationale": "A named Yoruba popular tradition meriting its own sourced ensemble, rhythm and performance study.",
    "sources": [
      "https://doi.org/10.1017/S0261143000003524"
    ]
  },
  {
    "name": "Fuji",
    "area": "Yoruba/Nigerian popular music",
    "priority": "P1",
    "rationale": "A major Nigerian popular tradition; boundaries and development relative to Apala and Juju require specialist sources.",
    "sources": [
      "https://doi.org/10.1080/17411912.2023.2260405"
    ]
  },
  {
    "name": "Makossa",
    "area": "Cameroonian popular music",
    "priority": "P1",
    "rationale": "A regional popular music identity not represented by a generic pan-African route.",
    "sources": [
      "https://doi.org/10.1017/S0261143000003524"
    ]
  },
  {
    "name": "Azonto",
    "area": "Ghanaian contemporary dance music",
    "priority": "P0",
    "rationale": "A named Ghanaian dance-music practice; distinguish musical structure from the related dance and from broader Afrobeats marketing labels.",
    "sources": [
      "https://www.africanmusiclibrary.org/genre/Azonto",
      "https://www.theguardian.com/music/2026/oct/06/african-music-afrobeats-tyla"
    ]
  },
  {
    "name": "Coupé-Décalé",
    "area": "Ivorian contemporary dance music",
    "priority": "P1",
    "rationale": "Candidate for a separately sourced rhythm, arrangement and scene contract; a directory label alone is only discovery evidence.",
    "sources": [
      "https://musicbrainz.org/genres"
    ]
  },
  {
    "name": "Calypso",
    "area": "Trinidadian and wider Caribbean music",
    "priority": "P0",
    "rationale": "A regionally grounded form with lyrical, rhythmic and social functions that a generic Caribbean label would hide.",
    "sources": [
      "https://folkways.si.edu/genre/calypso",
      "https://folklife.si.edu/magazine/calypso-music-trinidad-and-tobago"
    ]
  },
  {
    "name": "Soca",
    "area": "Trinidadian contemporary dance music",
    "priority": "P1",
    "rationale": "Research as a distinct contemporary lineage; do not collapse it into Calypso or generic dance-pop.",
    "sources": [
      "https://folklife.si.edu/magazine/calypso-music-trinidad-and-tobago",
      "https://musicbrainz.org/genres"
    ]
  },
  {
    "name": "Cumbia",
    "area": "Latin American regional dance music",
    "priority": "P1",
    "rationale": "A geographically varied family; regional versions should be linked rather than treated as one universal rhythmic template.",
    "sources": [
      "https://www.revistahistoriaindigena.uchile.cl/index.php/RMCH/article/view/44886"
    ]
  },
  {
    "name": "Vallenato",
    "area": "Greater Magdalena region, Colombia",
    "priority": "P0",
    "rationale": "A regionally identified song tradition with distinct rhythmic airs, instruments and intergenerational context; protect regional variants.",
    "sources": [
      "https://ich.unesco.org/en/USL/vallenato-traditional-music-of-the-greater-magdalena-region-01095"
    ]
  },
  {
    "name": "Bachata",
    "area": "Dominican Republic",
    "priority": "P1",
    "rationale": "Research the guitar, percussion, bass, song and dance relationship; source material supports tradition identity, not a ready-made generator.",
    "sources": [
      "https://ich.unesco.org/en/RL/music-and-dance-of-dominican-bachata-01514"
    ]
  },
  {
    "name": "Merengue",
    "area": "Dominican and broader Caribbean music",
    "priority": "P1",
    "rationale": "Candidate for independent review of rhythm, arrangement and regional variants; do not merge it into Bachata or Salsa.",
    "sources": [
      "https://ich.unesco.org/en/RL/bon-01162"
    ]
  },
  {
    "name": "Forró",
    "area": "Brazilian regional dance music",
    "priority": "P1",
    "rationale": "A regional family with variant styles, instruments and dance relationships; define scope before assigning one route.",
    "sources": [
      "https://www.cambridge.org/core/journals/latin-american-research-review/article/forros-wars-of-maneuver-and-position-popular-northeastern-music-critical-regionalism-and-a-culture-of-migration/1BD823B5294A93F26BE24AD65F19B9CF"
    ]
  },
  {
    "name": "Choro",
    "area": "Brazilian instrumental popular music",
    "priority": "P2",
    "rationale": "Candidate for a dedicated study of instrumental interaction, phrasing and ornament rather than a generic samba preset.",
    "sources": [
      "https://doi.org/10.1080/09298215.2020.1797109",
      "https://github.com/DCMLab/choro/tree/fe409c62deed50fc7f4b0d667863cd13ab230ff6",
      "https://github.com/DCMLab/choro/blob/fe409c62deed50fc7f4b0d667863cd13ab230ff6/LICENSE"
    ]
  },
  {
    "name": "Sertanejo",
    "area": "Brazilian country/popular music",
    "priority": "P2",
    "rationale": "A broad and historically changing field; scope early/modern variants and regional distinctions before modeling.",
    "sources": [
      "https://musicbrainz.org/genres"
    ]
  },
  {
    "name": "Raï",
    "area": "Algerian popular folk song",
    "priority": "P0",
    "rationale": "A documented living tradition whose language, social function and changing performance contexts belong in its evidence record.",
    "sources": [
      "https://ich.unesco.org/en/RL/rai-popular-folk-song-of-algeria-01894"
    ]
  },
  {
    "name": "Dangdut",
    "area": "Indonesian popular music",
    "priority": "P0",
    "rationale": "A high-salience Indonesian popular-music identity missing from the current registry; investigate internal variation before choosing a representative scope.",
    "sources": [
      "https://academic.oup.com/book/27688"
    ]
  },
  {
    "name": "Luk Thung",
    "area": "Thai popular and regional music",
    "priority": "P1",
    "rationale": "A Thai country/popular tradition with regional language and style distinctions; avoid treating it as generic Western country.",
    "sources": [
      "https://so02.tci-thaijo.org/index.php/jg-mcukk/article/view/244782",
      "https://www.researchgate.net/publication/303940861_Lukthung_Authenticity_and_Modernity_in_Thai_Country_Music"
    ]
  },
  {
    "name": "Mor Lam",
    "area": "Isan/Lao-linked performance traditions",
    "priority": "P1",
    "rationale": "Research its local-language, poetic, improvisational and contemporary hybrid contexts; keep links to Luk Thung explicit rather than equating them.",
    "sources": [
      "https://so02.tci-thaijo.org/index.php/jg-mcukk/article/view/244782",
      "https://www.researchgate.net/publication/303940861_Lukthung_Authenticity_and_Modernity_in_Thai_Country_Music"
    ]
  },
  {
    "name": "Bongo Flava",
    "area": "Tanzanian contemporary popular music",
    "priority": "P0",
    "rationale": "A named East African popular-music field needing regional, language, production and historical sourcing.",
    "sources": [
      "https://escholarship.org/uc/item/03d1v7fh",
      "https://www.journals.codesria.org/index.php/ad/article/view/968"
    ]
  },
  {
    "name": "K-pop",
    "area": "South Korean popular music",
    "priority": "P1",
    "rationale": "Research production, performance and dissemination conventions as well as intra-genre labels; global influence does not imply a single grammar.",
    "sources": [
      "https://doi.org/10.25024/kj.2013.53.4.14",
      "https://experts.arizona.edu/en/publications/k-pop-genres-a-cross-cultural-exploration/"
    ]
  },
  {
    "name": "J-pop",
    "area": "Japanese popular music",
    "priority": "P2",
    "rationale": "A broad market label that needs scoped subfamilies, time periods and local terminology before implementation.",
    "sources": [
      "https://onlinelibrary.wiley.com/doi/10.1002/9781119518297.eowe00042"
    ]
  },
  {
    "name": "Cantopop",
    "area": "Cantonese-language popular music",
    "priority": "P2",
    "rationale": "Research its language, repertoire and production scope as a separate linguistic/popular field; do not assume equivalence to Mandopop.",
    "sources": [
      "https://musicbrainz.org/genres"
    ]
  },
  {
    "name": "Trot",
    "area": "Korean popular music",
    "priority": "P2",
    "rationale": "A historically distinct Korean popular tradition; scope vocal, rhythmic and period differences through sources and specialist review.",
    "sources": [
      "https://researchportal.helsinki.fi/en/publications/interrogating-trot-situating-the-boom-inewtroi-nostalgia-old-song/"
    ]
  },
  {
    "name": "Ethio-jazz",
    "area": "Ethiopian jazz and popular music",
    "priority": "P1",
    "rationale": "Candidate for a scoped fusion study; identify which Ethiopian melodic practices and historical periods are actually supported.",
    "sources": [
      "https://musicbrainz.org/genres"
    ]
  },
  {
    "name": "Jungle",
    "area": "UK electronic dance music",
    "priority": "P1",
    "rationale": "Research breakbeat, bass and arrangement behavior separately from Drum and Bass rather than using a shared umbrella route.",
    "sources": [
      "https://musicbrainz.org/genres"
    ]
  },
  {
    "name": "Footwork / Juke",
    "area": "Chicago electronic dance music",
    "priority": "P2",
    "rationale": "Candidate for highly specific rhythmic and phrase behavior; resolve whether this is one queue item or two related identities.",
    "sources": [
      "https://musicbrainz.org/genres"
    ]
  },
  {
    "name": "Psytrance",
    "area": "Electronic dance music",
    "priority": "P2",
    "rationale": "Research long-form arrangement, periodicity and sound-design constraints; symbolic composition and rendered production require separate evidence.",
    "sources": [
      "https://musicbrainz.org/genres"
    ]
  },
  {
    "name": "Hardstyle",
    "area": "Electronic dance music",
    "priority": "P2",
    "rationale": "Candidate for its own arrangement, kick/bass and section-transition model, contingent on sound-rendering support.",
    "sources": [
      "https://musicbrainz.org/genres"
    ]
  },
  {
    "name": "Hyperpop",
    "area": "Contemporary internet-native pop",
    "priority": "P2",
    "rationale": "Highly time-sensitive and heterogeneous label; capture temporal scope and production dependencies before making a stable identity.",
    "sources": [
      "https://musicbrainz.org/genres"
    ]
  },
  {
    "name": "Banku Music",
    "area": "Ghanaian contemporary popular music",
    "priority": "P1",
    "rationale": "Emerging label named in contemporary reporting; obtain locally grounded sources and practitioner definitions before any registry promotion.",
    "sources": [
      "https://www.theguardian.com/music/2026/oct/06/african-music-afrobeats-tyla"
    ]
  },
  {
    "name": "Mara",
    "area": "African contemporary popular music; scope unconfirmed",
    "priority": "P2",
    "rationale": "An emerging label noted in contemporary reporting; identity, geographic scope and community usage need independent confirmation.",
    "sources": [
      "https://www.theguardian.com/music/2026/oct/06/african-music-afrobeats-tyla"
    ]
  }
]''')
SOURCE_ASSESSMENTS=json.loads(r'''{
  "https://doi.org/10.1017/S0261143000003524": {
    "source_kind": "peer_reviewed_musicology_article",
    "scope": "broad_scholarly_context",
    "supports": "A historical survey situating named African popular-music traditions within regional histories.",
    "does_not_establish": "It is not a complete or contemporary style specification for every tradition it mentions."
  },
  "https://doi.org/10.1080/00064246.2024.2370204": {
    "source_kind": "scholarly_article",
    "scope": "tradition_specific_context",
    "supports": "A scholarly discussion distinguishing contemporary Afrobeats from Afrobeat terminology.",
    "does_not_establish": "A terminology discussion does not define an executable groove, arrangement, or production contract."
  },
  "https://doi.org/10.11606/issn.1983-6023.sank.2016.119064": {
    "source_kind": "scholarly_article",
    "scope": "tradition_specific_context",
    "supports": "A focused study of Mbalax as Senegalese popular music and its changing boundaries.",
    "does_not_establish": "It does not validate a complete generative grammar or establish present-day practitioner consensus."
  },
  "https://doi.org/10.25024/kj.2013.53.4.14": {
    "source_kind": "scholarly_publication_record",
    "scope": "broad_scholarly_context",
    "supports": "Research context for K-pop's production and international dissemination.",
    "does_not_establish": "A broad media or industry account does not specify one K-pop musical grammar."
  },
  "https://experts.arizona.edu/en/publications/k-pop-genres-a-cross-cultural-exploration/": {
    "source_kind": "scholarly_publication_record",
    "scope": "tradition_specific_context",
    "supports": "A cross-cultural study of genre labels applied within K-pop.",
    "does_not_establish": "It does not make K-pop internally homogeneous or prescribe a generator."
  },
  "https://folklife.si.edu/magazine/calypso-music-trinidad-and-tobago": {
    "source_kind": "museum_public_history",
    "scope": "tradition_specific_context",
    "supports": "Smithsonian Folklife context for Calypso in Trinidad and Tobago.",
    "does_not_establish": "Public-history coverage is a starting point, not a complete technical analysis or practitioner review."
  },
  "https://folkways.si.edu/genre/calypso": {
    "source_kind": "museum_music_library_catalog",
    "scope": "curated_reference_profile",
    "supports": "A Smithsonian Folkways curated music-library reference for Calypso.",
    "does_not_establish": "Catalog classification alone does not establish regional variation or compositional rules."
  },
  "https://ich.unesco.org/en/Decisions/14.COM/10.b.10": {
    "source_kind": "unesco_intangible_heritage_decision",
    "scope": "tradition_specific_context",
    "supports": "The UNESCO inscription decision and community context for Dominican Bachata.",
    "does_not_establish": "Heritage recognition is not proof of a machine-readable grammar or authorization to claim faithful generation."
  },
  "https://ich.unesco.org/en/RL/rai-popular-folk-song-of-algeria-01894": {
    "source_kind": "unesco_intangible_heritage_record",
    "scope": "tradition_specific_context",
    "supports": "The heritage record for Raï as Algerian popular folk song.",
    "does_not_establish": "The record does not cover every contemporary substyle or provide a complete composing algorithm."
  },
  "https://ich.unesco.org/en/USL/vallenato-traditional-music-of-the-greater-magdalena-region-01095": {
    "source_kind": "unesco_intangible_heritage_record",
    "scope": "tradition_specific_context",
    "supports": "A region-specific heritage record for traditional Vallenato music.",
    "does_not_establish": "One heritage record cannot be treated as exhaustive for every regional and modern variant."
  },
  "https://musicbrainz.org/doc/Genre": {
    "source_kind": "community_taxonomy_methodology",
    "scope": "taxonomy_only",
    "supports": "The role and subjective nature of community-maintained genre tags as a discovery mechanism.",
    "does_not_establish": "It does not establish source quality, musical properties, cultural authenticity, or a generation route for any candidate."
  },
  "https://musicbrainz.org/genres": {
    "source_kind": "community_taxonomy_directory",
    "scope": "discovery_only",
    "supports": "A maintained directory of genre labels and naming leads.",
    "does_not_establish": "A directory listing is not tradition-specific evidence, scholarly validation, or a compositional specification."
  },
  "https://so02.tci-thaijo.org/index.php/jg-mcukk/article/view/244782": {
    "source_kind": "scholarly_journal_article",
    "scope": "tradition_specific_context",
    "supports": "A study of language in Luk Thung performances by a Mor Lam Siang-Isan band, directly relevant to the overlap between the two forms.",
    "does_not_establish": "Its language-focused scope is not a complete musical grammar and should not erase differences between Luk Thung and Mor Lam."
  },
  "https://www.africanmusiclibrary.org/genre/Azonto": {
    "source_kind": "music_library_genre_profile",
    "scope": "curated_reference_profile",
    "supports": "A focused library/profile entry that can help locate Azonto-specific references.",
    "does_not_establish": "A library profile alone is not practitioner consensus or an implementation-ready rhythmic contract."
  },
  "https://www.researchgate.net/publication/303940861_Lukthung_Authenticity_and_Modernity_in_Thai_Country_Music": {
    "source_kind": "research_repository_record",
    "scope": "repository_record_only",
    "supports": "A repository discovery record for a work about authenticity and modernity in Thai country music.",
    "does_not_establish": "The hosting page is not by itself a publisher-of-record validation; verify the full bibliographic record and argument."
  },
  "https://www.theguardian.com/music/2026/oct/06/african-music-afrobeats-tyla": {
    "source_kind": "journalistic_reporting",
    "scope": "current_reporting_lead",
    "supports": "A contemporary reporting lead about terminology and emerging labels in African popular music.",
    "does_not_establish": "Journalism is a lead for follow-up, not sufficient evidence of a stable identity or musical contract."
  },
  "https://doi.org/10.21504/amj.v9i4.1888": {
    "source_kind": "peer_reviewed_musicology_article",
    "scope": "tradition_specific_context",
    "supports": "A focused study of Benga beat formation in Kenyan Luo song style and its later adaptations.",
    "does_not_establish": "The article's historical scope does not establish one canonical Benga pattern for all communities or eras."
  },
  "https://www.jstor.org/stable/j.ctt17w8gb6": {
    "source_kind": "university_press_monograph_record",
    "scope": "tradition_specific_context",
    "supports": "A scholarly monograph examining African music, colonial Zimbabwe, and the historical context of Chimurenga music.",
    "does_not_establish": "A historical monograph is not an exhaustive present-day repertoire survey or a generative specification."
  },
  "https://doi.org/10.1080/17411912.2023.2260405": {
    "source_kind": "peer_reviewed_ethnomusicology_article",
    "scope": "tradition_specific_context",
    "supports": "A study of Fújì music, genre-making, and social context in Lagos, Nigeria.",
    "does_not_establish": "It does not yield a universal musical recipe or collapse related Yoruba traditions."
  },
  "https://www.revistahistoriaindigena.uchile.cl/index.php/RMCH/article/view/44886": {
    "source_kind": "scholarly_musicology_article",
    "scope": "tradition_specific_context",
    "supports": "Research on the social construction, history, and changing usage of Cumbia in Colombia.",
    "does_not_establish": "Its central caution is against treating Cumbia as a fixed or homogeneous identity."
  },
  "https://ich.unesco.org/en/RL/bon-01162": {
    "source_kind": "unesco_intangible_heritage_record",
    "scope": "tradition_specific_context",
    "supports": "A UNESCO heritage record describing music and dance of Merengue in the Dominican Republic.",
    "does_not_establish": "The record does not cover every regional variation or define an executable rhythm engine."
  },
  "https://www.cambridge.org/core/journals/latin-american-research-review/article/forros-wars-of-maneuver-and-position-popular-northeastern-music-critical-regionalism-and-a-culture-of-migration/1BD823B5294A93F26BE24AD65F19B9CF": {
    "source_kind": "peer_reviewed_music_and_culture_article",
    "scope": "tradition_specific_context",
    "supports": "A cultural-historical study of northeastern Brazilian popular music and the relationship between baião and forró.",
    "does_not_establish": "The article emphasizes history and regional politics rather than a complete performance-level style grammar."
  },
  "https://doi.org/10.1080/09298215.2020.1797109": {
    "source_kind": "scholarly_corpus_study",
    "scope": "tradition_specific_context",
    "supports": "A corpus-driven analysis of harmony and form in Brazilian Choro based on a set of representative pieces.",
    "does_not_establish": "Its corpus and analytical dimensions are bounded; it does not prove an engine produces perceptually authentic Choro."
  },
  "https://academic.oup.com/book/27688": {
    "source_kind": "scholarly_monograph_record",
    "scope": "tradition_specific_context",
    "supports": "A social and musical history of Dangdut including vocal style, melody, rhythm, harmony, form, lyrics, and performance meaning.",
    "does_not_establish": "Historical analysis still requires explicit corpus selection and validation before any code claims fidelity."
  },
  "https://escholarship.org/uc/item/03d1v7fh": {
    "source_kind": "university_repository_dissertation",
    "scope": "tradition_specific_context",
    "supports": "A dissertation studying Bongo Flava, Tanzanian hip-hop/pop, and the music's sociopolitical development.",
    "does_not_establish": "A dissertation's scope and author perspective should be represented; it is not a universal definition of all Tanzanian popular music."
  },
  "https://www.journals.codesria.org/index.php/ad/article/view/968": {
    "source_kind": "scholarly_journal_article",
    "scope": "tradition_specific_context",
    "supports": "Research on Bongo Flava and hip-hop in contemporary Tanzania, including its linguistic and social setting.",
    "does_not_establish": "The article does not prescribe an executable composition model."
  },
  "https://onlinelibrary.wiley.com/doi/10.1002/9781119518297.eowe00042": {
    "source_kind": "scholarly_reference_article",
    "scope": "tradition_specific_context",
    "supports": "A 2025 reference article discussing J-pop, language contact, and its historical relationship to other Japanese popular forms.",
    "does_not_establish": "A linguistic and historical account does not define every J-pop substyle or a complete musical grammar."
  },
  "https://researchportal.helsinki.fi/en/publications/interrogating-trot-situating-the-boom-inewtroi-nostalgia-old-song/": {
    "source_kind": "peer_reviewed_music_culture_article",
    "scope": "tradition_specific_context",
    "supports": "A study of Trot as a Korean popular-music genre and the historical discourse around its 2019–2020 revival.",
    "does_not_establish": "A study of a particular revival period is not a full transhistorical Trot definition."
  },
  "https://ich.unesco.org/en/RL/music-and-dance-of-dominican-bachata-01514": {
    "source_kind": "unesco_intangible_heritage_record",
    "scope": "tradition_specific_context",
    "supports": "A UNESCO record detailing Dominican Bachata's social context, instrumentation, rhythmic outline, song, and dance.",
    "does_not_establish": "The record is a point of cultural grounding, not complete implementation evidence or permission for authenticity claims."
  },
  "https://github.com/DCMLab/choro/tree/fe409c62deed50fc7f4b0d667863cd13ab230ff6": {
    "source_kind": "scholarly_corpus_repository_pinned_commit",
    "scope": "tradition_specific_context",
    "supports": "The DCMLab Choro Songbooks Corpus repository README describes a corpus of 295 pieces and provides annotations for key, meter, form paths/phrases/parts, chord positions, chord types/extensions, and available composition-year/subgenre metadata.",
    "does_not_establish": "The corpus is based on published Choro Songbook material and is not a universal sample of all Choro practice; repository metadata does not establish commercial reuse permission or that any Muse generator is faithful.",
    "source_pin": "fe409c62deed50fc7f4b0d667863cd13ab230ff6"
  },
  "https://github.com/DCMLab/choro/blob/fe409c62deed50fc7f4b0d667863cd13ab230ff6/LICENSE": {
    "source_kind": "corpus_license_text_pinned_commit",
    "scope": "license_and_reuse_constraint",
    "supports": "The repository LICENSE identifies Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0).",
    "does_not_establish": "This is not legal advice and does not settle rights in third-party musical works represented by the transcriptions; do not use corpus data for commercial training or redistribute extracted data without legal clearance or separate permission.",
    "source_pin": "fe409c62deed50fc7f4b0d667863cd13ab230ff6"
  }
}''')
BASE_EVIDENCE_GATES=["versioned grammar/recipe contract names target behavior and known limits","one-factor response curves run through actual routed composition path","candidate non-target invariants reported separately from target fidelity","fixed-seed response and exact source/artifact digests are complete","nearest-neighbor distinctness/collision analysis is published","renderer substitutions and unavailable layers are disclosed"]
CULTURAL_EVIDENCE_GATE="review by knowledgeable musicians or researchers from the relevant tradition before any authenticity claim or Foundation promotion"
def _slug(value:str)->str:
 value=unicodedata.normalize("NFKD",value).encode("ascii","ignore").decode("ascii"); value=re.sub(r"[^a-z0-9]+","_",value.lower()).strip("_")
 if not value: raise CoverageError("could not derive track slug")
 return value
def parse_grammar_foci(source:str,expected_count:int)->list[str]:
 match=re.search(r"const\s+CANONICAL_GRAMMAR_FOCI\s*:\s*\[\[&str;\s*(\d+)\];\s*(\d+)\]\s*=\s*\[(.*?)\n\];",source,re.DOTALL)
 if match is None: raise CoverageError("Could not parse CANONICAL_GRAMMAR_FOCI")
 width,groups=int(match.group(1)),int(match.group(2)); tokens=re.findall(r'"(?:\\.|[^"\\])*"',match.group(3))
 try: foci=[json.loads(t) for t in tokens]
 except json.JSONDecodeError as exc: raise CoverageError(f"Malformed grammar focus: {exc}") from exc
 if width<=0 or groups<=0 or len(foci)!=width*groups: raise CoverageError(f"Grammar-focus cardinality mismatch: expected {groups}x{width}; got {len(foci)}")
 if len(foci)!=expected_count: raise CoverageError(f"Grammar-focus count {len(foci)} does not match {expected_count}")
 if any(not x.strip() for x in foci): raise CoverageError("Grammar-focus table contains empty hypothesis")
 return foci
def build_roadmap(root:Path)->dict[str,Any]:
 report=build_current_report(root); entries=report["entries"]; source=(root/CATALOG_RELATIVE).read_text(encoding="utf-8"); foci=parse_grammar_foci(source,len(entries))
 tokens={e["constellation"] for e in entries}
 if tokens!=set(WORKSTREAMS): raise CoverageError(f"Workstream/catalog mismatch: unplanned={sorted(tokens-set(WORKSTREAMS))}; unused={sorted(set(WORKSTREAMS)-tokens)}")
 grouped={}
 for e in entries: grouped.setdefault(e["constellation"],[]).append(e)
 planned=[]
 for e,focus in zip(entries,foci,strict=True):
  ws=WORKSTREAMS[e["constellation"]]; direct=e["composer_style"] is not None; gates=list(BASE_EVIDENCE_GATES)
  if ws["cultural_review_recommended"] or e["requires_expert_review"]: gates.append(CULTURAL_EVIDENCE_GATE)
  planned.append({"id":e["id"],"name":e["name"],"display_name":e["display_name"],"constellation":e["constellation"],"catalog_status":e["status"],"requires_expert_review":e["requires_expert_review"],"current_composer_route":e["composer_style"],"route_state":"direct_route_declared" if direct else "research_only_unrouted","grammar_focus_hypothesis":focus,"workstream_priority":ws["priority"],"candidate_engine_track":ENTRY_TRACK_OVERRIDES.get(e["name"],f'{e["constellation"].lower()}_{_slug(focus)}'),"next_gate":"M0d_measure_existing_route_and_capture_actual_behavior" if direct else ws["next_gate"],"minimum_evidence_gates":gates,"claim_boundary":"A direct route is not proof of musical distinctness, rendered availability, listener recognition, or cultural authenticity." if direct else "Research plan only: no composer route or capability claim is implied."})
 workstreams=[]
 for token,ws in WORKSTREAMS.items():
  rows=grouped[token]; unr=[e for e in rows if e["composer_style"] is None]
  workstreams.append({"constellation":token,"label":ws["label"],"priority":ws["priority"],"candidate_engine_track":ws["candidate_engine_track"],"implementation_hypothesis":ws["hypothesis"],"next_gate":ws["next_gate"],"cultural_review_recommended":ws["cultural_review_recommended"],"catalog_entries":len(rows),"direct_route_entries":len(rows)-len(unr),"research_only_unrouted_entries":len(unr),"catalog_ids":[e["id"] for e in rows],"unrouted_catalog_ids":[e["id"] for e in unr]})
 names={e["name"].casefold() for e in entries}; dup=[c["name"] for c in EXTENSION_CANDIDATES if c["name"].casefold() in names]
 if dup: raise CoverageError("Expansion queue duplicates catalog names: "+", ".join(dup))
 def assess_candidate(c:dict[str,Any])->dict[str,Any]:
  source_rows=[]
  for url in c["sources"]:
   assessment=SOURCE_ASSESSMENTS.get(url)
   if assessment is None: raise CoverageError(f"No source assessment registered for {c['name']}: {url}")
   source_rows.append({"url":url,**assessment})
  scopes={row["scope"] for row in source_rows}
  if "tradition_specific_context" in scopes: evidence_state="tradition_specific_context_present"
  elif "broad_scholarly_context" in scopes: evidence_state="broad_scholarly_context_only"
  elif "current_reporting_lead" in scopes: evidence_state="journalism_lead_only"
  elif "curated_reference_profile" in scopes or "repository_record_only" in scopes: evidence_state="curated_reference_or_repository_only"
  else: evidence_state="discovery_only"
  next_gate={
   "tradition_specific_context_present":"Disambiguate names and aliases; check source date, region, internal variants, and community terms; then seek knowledgeable-practitioner review and formulate falsifiable musical hypotheses.",
   "broad_scholarly_context_only":"Add focused tradition-specific research and practitioner review before treating this name as a stable identity or proposing a grammar.",
   "journalism_lead_only":"Corroborate this reporting lead with tradition-specific research and direct practitioner/primary sources before taxonomy review.",
   "curated_reference_or_repository_only":"Verify the bibliographic record and supplement the reference profile with tradition-specific research and direct practitioner context.",
   "discovery_only":"Replace or supplement directory-only discovery links with tradition-specific scholarly, archival, heritage, or practitioner sources before taxonomy review."
  }[evidence_state]
  return {**c,"source_assessments":source_rows,"source_evidence_state":evidence_state,"source_evidence_limit":"Source presence grounds research intake only; none of these labels proves a unique executable grammar, runtime generation, perceptual distinctness, or cultural authenticity.","catalog_id":None,"state":"proposal_only_not_in_catalog","next_gate":next_gate,"implementation_claim":"No registry row, composer route, or runtime capability is implied."}
 expansions=[assess_candidate(c) for c in EXTENSION_CANDIDATES]
 summary={"catalog_entries":len(entries),"constellations":len(grouped),"entries_with_explicit_completion_track":len(planned),"direct_route_entries":sum(e["composer_style"] is not None for e in entries),"research_only_unrouted_entries":sum(e["composer_style"] is None for e in entries),"research_expert_review_required_entries":sum(e["status"]=="ResearchExpertReviewRequired" for e in entries),"workstreams_with_explicit_plan":len(workstreams),"proposed_expansion_candidates_not_in_catalog":len(expansions),"roadmap_rows_are_not_implementation_evidence":True,"priority_definition":"P0/P1/P2 express suggested work sequencing only, never cultural or artistic value.","candidate_source_evidence_states":{state:sum(e["source_evidence_state"]==state for e in expansions) for state in sorted({e["source_evidence_state"] for e in expansions})}}
 return {"schema":"muse-genre-completion-roadmap-v1","kind":"source_bound_implementation_backlog_not_runtime_evidence","source":{"catalog_path":report["source"]["catalog_path"],"catalog_blob_sha":report["source"]["catalog_blob_sha"],"executable_style_path":report["source"]["executable_style_path"],"executable_style_blob_sha":report["source"]["executable_style_blob_sha"],"grammar_focus_path":CATALOG_RELATIVE.as_posix(),"grammar_focus_blob_sha":report["source"]["catalog_blob_sha"],"note":"Catalog/focus values are source-derived. Candidate tracks, priorities and expansion entries are proposals, not implementation evidence."},"summary":summary,"workstreams":workstreams,"entries":planned,"expansion_candidates":expansions,"next_evidence_gate":"After source-coverage and response-curve jobs run on the exact head, choose one P0 track and require target/invariance, seed, distinctness, renderer and applicable review evidence before promotion."}
def render(root:Path)->str: return json.dumps(build_roadmap(root),indent=2,ensure_ascii=False)+"\n"
def main(argv:list[str]|None=None)->int:
 parser=argparse.ArgumentParser(description=__doc__); group=parser.add_mutually_exclusive_group(); group.add_argument("--check",action="store_true"); group.add_argument("--report",action="store_true"); group.add_argument("--write",action="store_true")
 parser.add_argument("--root",type=Path,default=ROOT); parser.add_argument("--output",type=Path,default=None); args=parser.parse_args(argv); root=args.root.resolve(); output=args.output.resolve() if args.output else root/ROADMAP_RELATIVE
 try:
  text=render(root)
  if args.report: print(text,end=""); return 0
  if args.write: output.parent.mkdir(parents=True,exist_ok=True); output.write_text(text,encoding="utf-8"); print(f"Wrote {output}"); return 0
  if not output.is_file(): raise CoverageError(f"Roadmap missing: {output}; regenerate with --write")
  if output.read_text(encoding="utf-8")!=text: raise CoverageError("Roadmap stale; inspect --report and refresh explicitly")
  print(f"PASS: roadmap covers {len(build_current_report(root)['entries'])} catalog entries; route/evidence remain separate."); return 0
 except (CoverageError,OSError,json.JSONDecodeError) as exc: print(f"FAIL: {exc}",file=sys.stderr); return 1
if __name__=="__main__": raise SystemExit(main())
