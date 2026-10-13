# Muse Global Genre Coverage and Research Intake v1

**Status:** Research and intake guidance only. This document does not claim the registry is a complete inventory of world music, and it does not claim that any proposal-only candidate is implemented.

## Current boundary

The canonical registry currently contains 160 named entries across 20 constellations. Twenty-nine entries have direct composer routes; 131 remain unrouted research entries, including eight Southern African additions marked as requiring expert review. A route is a code path, not evidence of musical distinctness, an available renderer, community recognition, or cultural authenticity.

The expansion queue now contains 60 names outside the catalog. It is deliberately an intake list, not a “complete list of every genre”: genre labels evolve, overlap, travel across languages and scenes, and may be contested by practitioners. New names should be added through versioned evidence rather than assuming an exhaustive flat tree is possible.

## Evidence from the research scan

- MusicBrainz treats genres as community-submitted tags and explicitly describes them as subjective. It is useful for discovering labels, not as a compositional specification: [Genre](https://musicbrainz.org/doc/Genre) and [genre list](https://musicbrainz.org/genres).
- Lange et al. describe how genre categories shift with historical periods, language, discourse communities, industry practices and social conditions; boundaries can be fuzzy and subjective: [Categorizing music by genres](https://doi.org/10.1111/nyas.15404).
- Mehta et al. (Findings of NAACL 2025) reported that only 5.7% of the music hours in the datasets they surveyed were labeled as non-Western genres, and found cross-genre adaptation non-trivial for Hindustani classical and Turkish Makam: [Music for All](https://aclanthology.org/2025.findings-naacl.258/). Treat this as a finding about the studied datasets, not a universal estimate for every music model.
- A 2025 critique of GTZAN argues that its familiar genre classes reflect a historically narrow research frame and should not be mistaken for a universal ontology: [An Accidental Benchmark](https://link.springer.com/article/10.1007/s44206-025-00191-w).
- Collins’ historical account names distinct African popular-music traditions—including Benga, Chimurenga, Juju, Apala, Makossa and Mbalax—rather than one monolithic “African music” class: [The early history of West African highlife music](https://doi.org/10.1017/S0261143000003524). Döring’s study treats Mbalax as a Senegalese popular-music tradition with changing genre boundaries: [Mbalax](https://doi.org/10.11606/issn.1983-6023.sank.2016.119064).
- UNESCO’s heritage records provide grounded context for named traditions such as [Vallenato](https://ich.unesco.org/en/USL/vallenato-traditional-music-of-the-greater-magdalena-region-01095), [Raï](https://ich.unesco.org/en/RL/rai-popular-folk-song-of-algeria-01894), [Gnawa](https://ich.unesco.org/en/RL/gnawa-01170) and the [Bachata inscription decision](https://ich.unesco.org/en/Decisions/14.COM/10.b.10). Such records establish cultural and transmission context; they do not give Muse permission to claim faithful generation.
- The Korea Journal study on K-pop production and dissemination, and Lee et al.’s cross-cultural study of K-pop genre labels, are reminders to investigate internal labels and differences in community usage rather than treat a global export label as a single musical grammar: [Korea Journal](https://doi.org/10.25024/kj.2013.53.4.14), [K-pop genres: A cross-cultural exploration](https://experts.arizona.edu/en/publications/k-pop-genres-a-cross-cultural-exploration/).
- Contemporary reporting dated 6 October 2026 warns that using “Afrobeats” as a catch-all can obscure artists’ own genre identities and emerging Ghanaian labels. This is a current reporting lead, not sufficient evidence for a catalog identity on its own: [The Guardian](https://www.theguardian.com/music/2026/oct/06/african-music-afrobeats-tyla).

## New proposal-only coverage areas

These 34 additions bring the out-of-catalog intake queue from 26 to 60. Every row remains without a catalog ID, runtime route, or implementation claim.

| Region / scene | Candidate identities | Research focus before promotion |
|---|---|---|
| East, West and Central Africa | Benga, Chimurenga, Mbalax, Apala, Fuji, Makossa | Regional and historical scope; rhythm/ensemble roles; language and scene-specific performance; relationship to nearby traditions |
| Ghana and Côte d’Ivoire | Azonto, Coupé-Décalé, Banku Music, Mara | Verify name and community usage; distinguish dance from music; date/place boundaries; get local practitioner sources |
| Caribbean | Calypso, Soca | Lyrical form, rhythm, performance context and the distinct historical relation between styles |
| Latin America | Cumbia, Vallenato, Bachata, Merengue, Forró, Choro, Sertanejo | Keep regional variants visible; define instruments, cycles, song/dance relation and temporal scope instead of a single universal preset |
| North Africa | Raï | Language, song/performance context, historical changes and regional scope |
| Southeast and East Asia | Dangdut, Luk Thung, Mor Lam, Bongo Flava, K-pop, J-pop, Cantopop, Trot, Ethio-jazz | Separate language markets, local labels, historical periods, regional families and contemporary hybrid styles |
| Electronic / internet-native | Jungle, Footwork / Juke, Psytrance, Hardstyle, Hyperpop | Test groove/phrase/section behavior separately from sound-design and renderer dependencies |

The candidate source field has different strengths. Scholarly work or a heritage authority can ground research scope; a maintained genre directory can only help discover a label; journalism can flag current terminology for subsequent confirmation. None alone establishes a machine-readable grammar. The checked-in roadmap records the proposed names and source links and keeps all candidates explicitly outside the canonical registry.

## Promotion protocol

A proposal should receive a stable catalog ID only after name/alias disambiguation, scope and period definition, source notes, and review of overlap with existing entries. A composer route should require a versioned grammar contract, target and non-target measurements through the actual composition path, fixed-seed and source-bound receipts, nearest-neighbor collision analysis, and explicit renderer availability. Culturally rooted styles require informed review by people knowledgeable in the relevant tradition before authenticity claims or promotion to a high-confidence status.

For each tradition, document what evidence says, what remains uncertain, and what must not be inferred. Do not treat a metric match, a MIDI rendering, a public genre tag, or a broad family label as proof of faithful musical representation. Do not treat P0/P1/P2 as a value judgment about cultures; they only indicate suggested sequencing for engineering and research.

## Scope and limitations

The registry and queue are designed to become more useful through accountable, revisable research, not to freeze living music cultures into a final list. Candidate coverage remains incomplete by design. The next useful unit of progress is a reviewed, testable style contract with observable distinguishing behavior—not simply another row or another large genre count.
