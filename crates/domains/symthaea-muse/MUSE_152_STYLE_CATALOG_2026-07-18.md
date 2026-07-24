# Muse 152 — Style Catalog and Grammar Roadmap (2026-07-18)

**Status:** catalog target, not an implementation claim  
**Companion:** `MUSE_DIVERSITY_TRUTH_PLAN_2026-07-18.md`

## Implementation snapshot

The catalog architecture landed on 2026-07-18:

- one validated 152-entry registry shared by native server and WebAssembly UI;
- stable IDs, 19 constellations, canonical/display names, status, cultural-review
  requirements, composer routing, and eight-part style anatomy;
- all 29 current composer presets mapped exactly once; the other 123 entries are
  visible research targets and cannot be sent to `/api/compose`;
- `GET /api/catalog` exposes the full taxonomy;
- Create Mode navigates constellation → canonical style → advanced anatomy rather
  than presenting a flat 152-entry dropdown;
- first-class grammar, phrase, harmonic, intent-axis, and performance-dialect
  profiles route styles independently of the catalog taxonomy;
- the Groove-cycle route formalizes the existing cycle-continuous,
  clave/compás-aware mechanisms;
- the Process/additive route bypasses period form entirely: audible cell growth
  and contraction govern the whole piece over a static tonic/fifth clock;
- the Raga/modal-arc route now bypasses the period/cadence pipeline with a generic,
  culturally qualified exposition → pulse → intensification arc over continuous
  drone;
- promotion to `Foundation` is code-gated on blind grammar-family recognition,
  within-style identity, and expert review when the catalog requires it.

No entry is marked `Foundation`: listening and expert evidence cannot be created by
implementation alone.

Muse can turn 100+ styles into a major strength, provided those styles are real
compositional systems rather than parameter presets. The recommended catalog is:

> **19 grammar constellations × 8 canonical styles each = 152 styles**

The user chooses a recognizable style. Muse resolves it into:

> **grammar family + phrase grammar + harmonic syntax + rhythmic language +
> melodic DNA + ensemble + performance dialect + production space**

This preserves the architecture in the diversity plan. The current 29 styles
should migrate into this catalog, but no style is considered finished until its
grammar family passes the blind-listening gate.

## The catalog

### 1. Classical Lyric & Character

These emphasize clear melodic rhetoric, balanced phrasing, and expressive
character.

1. Classical
2. Romantic Nocturne
3. Waltz
4. Lullaby
5. March
6. Elegy
7. Playful Character Piece
8. Impressionist Prelude

### 2. Baroque and Early Dance Forms

These require dance-specific rhythm, articulation, harmonic sequences, and
ornamentation—not merely Baroque instrumentation.

9. Baroque Suite
10. Minuet and Trio
11. Gavotte
12. Sarabande
13. Gigue
14. Allemande
15. Toccata
16. French Overture

### 3. Developmental and Large Forms

These should operate through long-range transformation, tonal regions, and
explicit obligations.

17. Sonata
18. Rondo
19. Theme and Variations
20. Scherzo
21. Symphonic Poem
22. Concerto Movement
23. Progressive Suite
24. Chamber Fantasia

### 4. Contrapuntal and Polyphonic

These need independent voice identity, imitation, invertible relationships, and
controlled dissonance.

25. Fugue
26. Fughetta
27. Two-Part Invention
28. Canon
29. Ricercar
30. Renaissance Polyphony
31. Motet
32. Chorale Prelude

### 5. Ground, Ostinato and Transformation

These derive large-scale meaning from repetition that accumulates consequence.

33. Passacaglia
34. Chaconne
35. Descending-Tetrachord Lament
36. Folia Variations
37. Romanesca
38. Ostinato Variations
39. Erosion
40. Lineage

### 6. Song and Narrative

These need strophic, verse–refrain, and storytelling grammars rather than period
forms disguised with folk instruments.

41. Folk
42. Modal Folk
43. Folk Ballad
44. Singer-Songwriter
45. Art Song / Lied
46. Verse–Chorus Song
47. Through-Composed Song
48. Sea Shanty

### 7. Blues, Gospel and Soul

These should be built around chorus form, call-and-response, blue-note behavior,
vocal phrasing, and groove.

49. Delta Blues
50. Chicago Blues
51. Country Blues
52. Gospel
53. Spiritual
54. Soul Ballad
55. Rhythm and Blues
56. Neo-Soul

### 8. Jazz and Improvisatory Forms

These require phrase-level improvisational logic, chord-scale behavior, swing
dialects, and chorus memory.

57. Jazz Ballad
58. Swing
59. Bebop
60. Cool Jazz
61. Hard Bop
62. Modal Jazz
63. Jazz Waltz
64. Free Jazz

### 9. Latin and Caribbean Cycles

Here, rhythm must become structural form. Clave, tumbao, montuno, and dance cycles
cannot remain decorative accompaniment.

65. Afro-Cuban
66. Son Cubano
67. Salsa
68. Mambo
69. Cha-Cha-Chá
70. Bossa Nova
71. Samba
72. Tango

### 10. European and North Atlantic Folk

These need tradition-specific meters, phrase asymmetry, ornamentation, dance
relationships, and ensemble behavior.

73. Irish Traditional
74. Celtic
75. Scottish Reel and Strathspey
76. English Folk
77. Nordic Folk
78. Balkan Odd-Meter
79. Klezmer
80. Progressive Folk

### 11. Mediterranean, Middle Eastern and North African Modal

These should model modal pathways, characteristic cadences, ornament systems, and
improvisatory development rather than map everything onto Western chord loops.

81. Flamenco
82. Fado
83. Arabic Maqam
84. Turkish Makam
85. Persian Dastgah
86. Andalusian Nuba
87. Oud Taqsim
88. Greek Rebetiko

### 12. South Asian Raga and Tala

These require pitch hierarchy, ornament, temporal unfolding, drone relationships,
and tala-aware development.

89. Hindustani Khayal
90. Dhrupad
91. Alap–Jor–Jhala
92. Carnatic Kriti
93. Ragam–Tanam–Pallavi
94. Bhajan
95. Ghazal
96. Qawwali

### 13. East and Southeast Asian Traditions

These should preserve characteristic tuning concepts, ensemble roles, rhythmic
structures, and relationships to silence.

97. Japanese Gagaku
98. Shakuhachi Honkyoku
99. Japanese Min’yō
100. Chinese Guqin
101. Jiangnan Sizhu
102. Korean Gugak
103. Indonesian Gamelan
104. Thai Piphat

### 14. African and Diasporic Groove

These need layered pulse, interlocking parts, cyclical form, and ensemble
conversation.

105. Afrobeat
106. Highlife
107. Juju
108. Soukous
109. Amapiano
110. Gqom
111. Reggae
112. Dub

### 15. Minimal, Process and Experimental

These should be governed by processes rather than conventional phrase rhetoric.

113. Minimalism
114. Phase Music
115. Additive Process
116. Post-Minimalism
117. Spectralism
118. Aleatoric Chamber
119. Algorithmic Counterpoint
120. Generative Cellular Music

### 16. Ambient and Electronic Texture

These depend on timbral evolution, density, spectral motion, and long temporal
horizons.

121. Ambient
122. Dark Ambient
123. Drone
124. Berlin School
125. IDM
126. Glitch
127. Vaporwave
128. Electroacoustic

### 17. Club and Beat Music

These require groove-locked performance, phrase-energy scheduling, drop
architecture, and production-aware arrangement.

129. House
130. Deep House
131. Techno
132. Trance
133. Drum and Bass
134. UK Garage
135. Breakbeat
136. Hip-Hop Instrumental

### 18. Pop and Rock

These should use riff, verse, pre-chorus, chorus, bridge, and production-oriented
grammars.

137. Pop
138. Indie Pop
139. Dream Pop
140. Synthpop
141. Rock
142. Progressive Rock
143. Post-Rock
144. Folk Rock

### 19. Dramatic, Screen and Stage

These should respond to narrative state, thematic obligations, scene timing, and
character memory.

145. Opera
146. Musical Theatre
147. Cinematic
148. Film Noir Score
149. Epic Orchestral
150. Horror Score
151. Science-Fiction Score
152. Adaptive Game Score

## Catalog organization

The UI must not present a flat dropdown containing 152 entries. It uses three
levels.

### Constellation

The 19 broad musical worlds above, such as:

- Jazz and Improvisatory Forms
- Ground, Ostinato and Transformation
- South Asian Raga and Tala
- Ambient and Electronic Texture
- Dramatic, Screen and Stage

### Canonical style

The named style users recognize, such as Bebop, Passacaglia, or Amapiano.

### Style anatomy

Advanced controls reveal the ingredients:

- grammar;
- phrase behavior;
- harmony or modal system;
- rhythm;
- melodic language;
- ensemble;
- performance dialect;
- production environment.

This also enables principled style merging. For example:

> **Sonata development**  
> + **Afro-Cuban rhythmic cycle**  
> + **Impressionist harmony**  
> + **chamber ensemble**  
> + **dance-locked performance**

Muse must state that this is a new hybrid rather than pretend it is a historical
genre.

## Quality status for every style

Every catalog entry carries exactly one of four statuses:

**Foundation**  
The grammar and style are implemented and pass tests.

**Developing**  
The style works but shares some generic mechanisms.

**Research**  
A planned style with architecture but insufficient musical evidence.

**Expert review required**  
A culturally specific tradition that must not be presented as authentic until
reviewed by knowledgeable musicians.

The South Asian, East Asian, African, Middle Eastern, and many folk traditions
deserve especially careful source work and expert listening. Until then, names
such as “Hindustani-informed” are more honest than claiming a full Khayal system.

Status is evidence, not marketing. `Foundation` requires the diversity plan's
blind-listening gate; culturally specific styles require that gate **and** expert
review before Muse makes authenticity claims.

## Recommended implementation sequence

Do not build 152 separate composers. Build reusable foundations in this order:

1. Period and sentence
2. Developmental
3. Contrapuntal
4. Ground and variation
5. Strophic song
6. Blues chorus and call-response
7. Jazz chorus and improvisation
8. Groove-cycle
9. Raga and modal arc
10. Process and additive
11. Ambient and textural
12. Dramatic and adaptive

Then specialize those foundations using harmonic, rhythmic, melodic,
orchestration, and performance systems.

The next three families in the current plan remain the best choices:

- **Groove-cycle**
- **Process/additive**
- **Raga arc**

They create the largest audible departure from Muse's present period-based center.

## Success target

Aim publicly for **152 styles**, but define success as:

- 40 flagship styles of exceptional quality;
- 60–80 strong production-ready styles;
- the remainder clearly marked as developing or research;
- all supported by approximately 12–16 genuine grammar families;
- no style shipped merely because its instruments, tempo, and chord loop differ.

> Every style should represent a different way for music to think, move, remember,
> and become.
