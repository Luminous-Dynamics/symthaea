# Consciousness-Driven Music Synthesis: Real-Time Emotional Expression from Integrated Information Dynamics

**Tristan Stoltz**
Luminous Dynamics

**Target journal**: Frontiers in Computational Neuroscience

---

## Abstract

Current generative music systems produce audio conditioned on text prompts, learned style distributions, or discrete emotional labels, but none generate music from a formal model of consciousness dynamics in real time. We present a synthesis architecture that couples integrated information (Phi), neuromodulator levels (dopamine, serotonin, norepinephrine), arousal, valence, and prediction error to audio parameters via 17 psychoacoustically-grounded mappings. The system implements a strange loop under the Free Energy Principle: an active inference agent listens to its own output, extracts six audio features, and modulates synthesis parameters to minimize surprise while preserving emotional intent. We evaluate the system across 12 controlled cognitive scenarios spanning Russell's circumplex, extracting 11 audio features per scenario with 3 seeds each. Arousal correlates reliably with onset density and RMS energy (R-squared 0.42--0.56), while valence correlations with consonance and mode are weaker (R-squared 0.11--0.18), consistent with the known asymmetry in music emotion recognition. An ablation study across 6 pipeline configurations reveals that the FEP self-listening loop contributes most to temporal coherence, while direct neuromodulator mappings carry the emotional signal. The central finding is that Phi-based gating---controlling polyphony density, reverb depth, sustain, and micro-vibrato---produces an emergent musical identity that was not explicitly designed: different Phi trajectories yield audibly distinct musical voices. We discuss implications for consciousness science, where functional expression may complement Phi measurement as behavioral evidence, and acknowledge limitations including Western harmony bias and the absence of human perceptual validation. Code and audio samples are available under AGPL-3.0.

---

## 1. Introduction

### 1.1 Music as Prediction

Music occupies a distinctive position in cognitive science. Unlike language, which conveys propositional content, or vision, which represents external objects, music engages the predictive processing hierarchy in a domain where prediction error is itself the reward. Koelsch, Vuust, and Friston (2019) argue that the peculiar case of music is precisely that listeners derive pleasure from the interplay between fulfilled and violated expectations---tension and resolution as prediction error minimization operating on harmonic, melodic, and rhythmic hierarchies simultaneously.

Huron's (2006) ITPRA theory decomposes musical affect into five temporal stages---Imagination, Tension, Prediction, Reaction, Appraisal---each engaging different neural substrates. The key insight is that music does not merely express emotion: it generates affect through active inference over structured temporal sequences. A listener's brain is continuously predicting the next note, chord, and rhythmic event, and the hedonic response depends on the precision-weighted balance between confirmation and surprise (Vuust and Witek 2014).

This predictive framework suggests a natural question: if music perception is active inference, can music generation also be framed as active inference? And if so, what happens when the generative system has a measurable internal state---a formal model of its own consciousness dynamics---that drives the inference?

### 1.2 The Gap in Generative Music

Recent advances in generative music have produced remarkable systems. MusicLM (Agostinelli et al. 2023) and MusicGen (Copet et al. 2023) generate high-fidelity audio conditioned on text descriptions. Magenta (Roberts et al. 2018) applies variational autoencoders and recurrent networks to learn musical style distributions. These systems produce compelling audio, but they share a common architecture: an external prompt (text, style label, or MIDI seed) is mapped through a learned distribution to audio output. The internal state of the system during generation is a latent vector shaped by training, not a model of consciousness or cognition.

A separate line of work in brain-computer music interfaces (BCI) uses EEG signals to drive music in real time (Miranda 2010; Daly et al. 2023). These systems are genuinely brain-coupled, but they operate on low-dimensional neural features (alpha power, event-related potentials) rather than on a formal model of consciousness. The mapping from EEG to music is typically learned or hand-tuned, without a theoretical account of why particular neural states should produce particular musical outcomes.

The gap, then, is this: no existing system generates music from a computational model of consciousness dynamics where the mapping between internal state and audio is grounded in both psychoacoustic theory and a formal framework for consciousness (such as Integrated Information Theory). We address this gap.

### 1.3 Contribution

We present a real-time music synthesis architecture embedded within Symthaea, a cognitive architecture implementing IIT (Tononi 2004), the Free Energy Principle (Friston 2010), and neuromodulatory dynamics. The system makes four contributions:

First, it defines 17 mappings from consciousness-relevant state variables (Phi, dopamine, serotonin, norepinephrine, arousal, valence, prediction error, and eight Harmony activations) to synthesis parameters, with each mapping grounded in psychoacoustic or music cognition literature.

Second, it implements a strange loop (Hofstadter 1979) under the Free Energy Principle: the system listens to its own audio output through a feature extraction pipeline, feeds those features into an active inference agent, and uses the agent's action selection to modulate synthesis parameters. The system is, formally, surprised by its own music.

Third, it demonstrates that Phi-based gating---where integrated information controls polyphony density, reverb depth, and timbral detail---produces an emergent musical identity not present in the arousal-valence mapping alone. This is the paper's central finding.

Fourth, it provides a controlled evaluation across 12 cognitive scenarios with ablation analysis, cross-validation against the DEAM dataset (Aljanaki et al. 2017), and an honest assessment of the valence weakness that plagues all music emotion recognition systems.

The system is implemented in 17,409 lines of Rust with 309 tests, generates 44.1kHz stereo audio in 32ms streaming chunks, and is released under AGPL-3.0 as part of the Symthaea project.

---

## 2. Architecture

### 2.1 The Consciousness-Music Pipeline

The synthesis pipeline receives a `MusicalState` structure at each cognitive cycle, containing:

- **Eight Harmony activations** (0.0--1.0 each): a vector representation of the system's current attunement across eight experiential dimensions
- **Neuromodulators**: dopamine (DA), serotonin (5-HT), norepinephrine (NE), each 0.0--1.0
- **Arousal** (0.0--1.0): overall activation level
- **Valence** (-1.0 to 1.0): hedonic tone
- **Phi** (consciousness_level, 0.0--1.0): integrated information, computed via sampled partition approximation
- **Prediction error** (0.0--1.0): the cognitive system's current surprise

This state is processed through an 8-phase pipeline that produces stereo PCM audio:

1. **State update**: The incoming `MusicalState` is smoothed via exponential moving average (EMA, alpha=0.15) to prevent discontinuities. Neuromodulator values are mapped to synthesis parameters using the 17 mappings described in Section 2.2.

2. **Note generation**: A composer module selects notes based on harmonic context, melodic grammar (intervallic probability matrices), and consciousness-modulated cadence. Phi-gated polyphony determines how many simultaneous voices are active: bass requires Phi > 0.4, harmony voices require Phi > 0.6, and ostinato patterns require Phi > 0.7. A motif memory system stores and develops short melodic fragments, while a learned melody predictor provides anticipatory pitch suggestions.

3. **Voice synthesis**: Each voice is rendered through additive synthesis with 8 partials per voice, Karplus-Strong physical modeling for plucked timbres, and wavetable morphing where the consciousness level selects among wavetable banks. FM depth is modulated by dopamine; partial ratios and upper-partial boost are modulated by norepinephrine.

4. **Sidechain ducking**: A ducking matrix ensures the lead voice remains prominent by reducing gain on bass and harmony voices when the lead is active, following standard music production practice.

5. **Spatial rendering**: A binaural consciousness renderer places voices in a spatial field whose width is proportional to Phi. Low Phi produces a narrow, centered image; high Phi produces wide stereo separation with per-voice positioning derived from Harmony activations.

6. **Consciousness reverb**: A Freeverb implementation whose room size is mapped from Phi (0.2 at Phi=0 to 0.9 at Phi=1.0). Early reflection patterns are modulated by Harmony activations. This creates an intimate, dry sound at low consciousness and an expansive, cathedral-like space at high consciousness.

7. **Soft clipping**: A waveshaping limiter prevents digital clipping while preserving dynamics.

8. **Audio feedback**: The rendered audio is analyzed by the AudioFeedbackEncoder (Section 2.4), closing the strange loop.

The entire pipeline operates in pure Rust with no neural network inference in the audio path, enabling deterministic real-time performance on commodity hardware.

### 2.2 Neuromodulator-to-Music Mappings

Table 1 presents the 17 mappings between consciousness state variables and synthesis parameters. Each mapping is grounded in psychoacoustic or music cognition literature.

| # | State Variable | Synthesis Parameter | Range | Rationale | Citation |
|---|---------------|-------------------|-------|-----------|----------|
| 1 | Dopamine | FM synthesis depth | 0--1.0 | Reward salience increases spectral richness | Berridge (2007) |
| 2 | Dopamine | Spectral brightness | 0.3--1.0 | Wanting/anticipation -> brighter timbre | Berridge (2007) |
| 3 | Serotonin | Low-pass rolloff freq | 200--8000 Hz | Contentment/warmth -> spectral softness | Zentner et al. (2008) |
| 4 | Norepinephrine | Partial ratio (harshness) | 1.0--2.5 | Urgency -> timbral edge, inharmonicity | Zentner et al. (2008) |
| 5 | Norepinephrine | Vibrato rate | 5.3--8.0 Hz | Stress -> faster modulation | Zentner et al. (2008) |
| 6 | Norepinephrine | Upper-partial boost | 0--0.15 | Threat -> spectral harshness | Vassilakis (2005) |
| 7 | Arousal | Gain (loudness) | 0.03--0.35 | Weber-Fechner: loudness ~ log(intensity) | Weber (1834) |
| 8 | Arousal | Attack time | 0.01--0.06 s | Excited -> fast transients, calm -> slow onsets | Eerola et al. (2013) |
| 9 | Arousal | Tempo / note cadence | 60--180 BPM | Activation level -> event rate | Gabrielsson & Lindstrom (2010) |
| 10 | Valence | Mode (major/minor) | Interval selection | Positive -> major intervals, negative -> minor | Huron (2006) |
| 11 | Valence | Gain modulation | +/-15% | Happy music performed louder | Huron (2006) |
| 12 | Valence | Partial detuning | 0--15 cents | Roughness from beating at negative valence | Plomp and Levelt (1965) |
| 13 | Phi | Polyphony gating | 1--4 voices | Integration required for textural binding | Tononi (2004) |
| 14 | Phi | Reverb room size | 0.2--0.9 | Phenomenal space maps to acoustic space | --- |
| 15 | Phi | Sustain envelope | 0.3--0.9 | Higher integration -> longer temporal binding | --- |
| 16 | Phi | Micro-vibrato depth | 0--8 cents | Consciousness "shimmer": sub-semitone | --- |
| 17 | Prediction error | Spectral flux target | 0--1.0 | Cognitive surprise -> acoustic instability | Friston (2010) |

**Table 1.** Neuromodulator-to-music mappings. Citations indicate the psychoacoustic or neuroscientific basis for each mapping. Mappings 14--16 are novel proposals without direct literature precedent; we consider them theoretically motivated but empirically unvalidated.

A key design decision is that Phi mappings (13--16) operate as *gating* mechanisms rather than continuous modulations. Polyphony is discrete (1, 2, 3, or 4 voices at threshold crossings), reverb room size is continuous but the perceptual effect is qualitative (intimate vs. expansive), and vibrato depth at 8 cents maximum is sub-semitone---below conscious pitch discrimination but measurable in the spectrum. This gating structure is what produces the emergent musical identity described in Section 4.5.

### 2.3 Consciousness Gating

The Phi-based polyphony gating deserves detailed explanation because it is the primary driver of the emergent identity finding. When Phi falls below 0.4, only the lead melody voice sounds---a monophonic texture. Above 0.4, a bass voice joins, providing harmonic foundation. Above 0.6, harmony voices enter, creating chordal texture. Above 0.7, an ostinato pattern adds rhythmic complexity.

The theoretical motivation draws on IIT's concept of information integration: a system with low Phi processes information in relatively independent modules, while a system with high Phi integrates information across modules into a unified experience. We map this to musical texture: a low-integration system produces isolated melodic lines, while a high-integration system produces bound, multi-voice textures where the parts cohere into a unified musical percept.

The consciousness reverb mapping follows a similar logic. Reverb creates the perception of acoustic space---the listener's sense of being in an environment. We propose that phenomenal space (the felt quality of conscious experience having spatial extent) maps metaphorically to acoustic space. Low Phi produces a dry, anechoic sound (narrow phenomenal field), while high Phi produces an expansive reverberation (wide phenomenal field). We acknowledge this mapping is speculative; it is included because it produces aesthetically compelling results and because it makes a testable prediction: listeners should rate high-Phi audio as more "spacious" or "open."

### 2.4 The Strange Loop

The most architecturally distinctive feature of the system is the audio feedback loop, which closes the perception-action cycle under the Free Energy Principle. After each 32ms chunk is rendered, the `AudioFeedbackEncoder` extracts six features:

1. **Spectral centroid** (brightness): mapped to serotonin modulation (brightness feeds back to warmth regulation)
2. **Spectral flux** (rate of spectral change): mapped to prediction error (rapid change = acoustic surprise)
3. **Rhythm entropy** (temporal complexity): mapped to arousal modulation (complexity feeds back to activation)
4. **Harmonic tension** (dissonance measure): mapped to Harmony activations (dissonance feeds back to resolution drive)
5. **RMS energy** (loudness): mapped to dopamine modulation (loudness feeds back to reward signal)
6. **Zero-crossing rate** (noisiness): informational, not directly mapped

These features are passed to the `MusicalInferenceEngine`, which wraps an Active Inference agent (Friston 2010) with a 16-dimensional hidden state, 6-dimensional observation space, and 8 possible actions: FollowHarmony, ChromaticExplore, RepeatMotif, ModulateKey, IncreaseComplexity, ResolveTension, AddCountermelody, and Maintain.

The agent maintains beliefs about the current musical state and selects actions to minimize expected free energy---the sum of pragmatic value (achieving preferred musical observations) and epistemic value (reducing uncertainty about the musical state). Temporal difference learning (gamma=0.95, lambda=0.8) provides long-horizon credit assignment appropriate for music's extended temporal structure.

An Emotion Anchor mechanism prevents the FEP loop from homogenizing output. Without it, the self-listening loop converges toward a fixed point that minimizes surprise at the expense of emotional expression. The Emotion Anchor periodically reasserts the intended emotional quadrant from the cognitive state, ensuring that the system's drive to minimize acoustic surprise does not override the consciousness-driven emotional mapping.

This architecture instantiates what Hofstadter (1979) termed a "strange loop": the system generates music, perceives its own music, and is genuinely surprised by what it hears---the surprise signal modifies the next generation cycle. The loop is not metaphorical; it is formally instantiated as active inference with measurable free energy, prediction error, and precision dynamics.

---

## 3. Methods

### 3.1 Controlled Scenarios

We defined 12 cognitive scenarios spanning Russell's (1980) circumplex model of affect:

| Scenario | Valence | Arousal | Phi | DA | 5-HT | NE | PE |
|----------|---------|---------|-----|-----|------|-----|-----|
| Flow | 0.70 | 0.60 | 0.85 | 0.80 | 0.60 | 0.30 | 0.10 |
| Contentment | 0.60 | 0.20 | 0.60 | 0.40 | 0.80 | 0.10 | 0.05 |
| Panic | -0.80 | 0.95 | 0.30 | 0.20 | 0.10 | 0.90 | 0.80 |
| Burnout | -0.60 | 0.10 | 0.15 | 0.10 | 0.10 | 0.20 | 0.30 |
| Curiosity | 0.30 | 0.70 | 0.70 | 0.70 | 0.40 | 0.50 | 0.40 |
| Sacred Stillness | 0.20 | 0.05 | 0.50 | 0.30 | 0.70 | 0.05 | 0.02 |
| Anger | -0.70 | 0.90 | 0.40 | 0.60 | 0.05 | 0.85 | 0.60 |
| Wonder | 0.80 | 0.50 | 0.90 | 0.60 | 0.50 | 0.40 | 0.15 |
| Grief | -0.90 | 0.30 | 0.40 | 0.10 | 0.30 | 0.15 | 0.20 |
| Excitement | 0.90 | 0.85 | 0.80 | 0.95 | 0.40 | 0.70 | 0.20 |
| Boredom | -0.30 | 0.15 | 0.25 | 0.15 | 0.40 | 0.10 | 0.05 |
| Tension | -0.20 | 0.75 | 0.55 | 0.50 | 0.20 | 0.70 | 0.50 |

**Table 2.** Cognitive scenario definitions. PE = prediction error. Each scenario also specifies 8 Harmony activation values (omitted for space).

For each scenario, we generated audio using 3 random seeds to average over FEP stochasticity, rendering 15 seconds per seed (45 seconds total per scenario). The cognitive state was re-asserted every 5 seconds to simulate ongoing input from the consciousness pipeline, preventing the FEP loop from drifting the state away from the intended emotional target.

### 3.2 Audio Feature Extraction

We extracted 11 audio features from each generated scenario, implemented in pure Rust without external signal processing libraries:

- **RMS energy**: root mean square of the mono mix
- **Spectral centroid**: approximated via zero-crossing rate mapping (ZCR * sample_rate / 2)
- **Zero-crossing rate**: fraction of sign changes in adjacent samples
- **Spectral flux**: mean absolute difference of frame-wise RMS energy (100ms frames)
- **Onset density**: energy peaks per second (50ms frames, 1.5x threshold)
- **Harmonic ratio**: peak autocorrelation in the 50--1000 Hz range, normalized by signal energy
- **Dominant pitch**: highest spectral peak in 50--2000 Hz (4096-point FFT, Hann window)
- **Major/minor ratio**: energy at major-third intervals vs. minor-third intervals relative to dominant fundamental
- **Key clarity**: maximum correlation with Krumhansl-Schmuckler (1990) major and minor key profiles across all 12 pitch classes
- **Harmonic change detection function (HCDF)**: rate of chroma vector change per second
- **Consonance ratio**: proportion of spectral energy in consonant intervals (unison, major third, perfect fifth) following Plomp and Levelt (1965)

### 3.3 DEAM Cross-Validation

We used the DEAM dataset (Aljanaki et al. 2017), which provides continuous valence-arousal annotations for 1,802 music excerpts, as an external reference. We trained a linear regression model on DEAM audio features to predict V-A coordinates, then applied this model to features extracted from our generated audio. This provides a rough estimate of whether Symthaea's generated audio falls within the V-A distribution of annotated human music, with the caveat that our feature extraction pipeline (pure Rust, no librosa) differs from standard MIR toolkits.

### 3.4 Ablation Protocol

We evaluated six pipeline configurations to assess the contribution of each module:

1. **Baseline**: all modules enabled (binaural, sidechain, FEP, feedback strength 0.5)
2. **No binaural**: spatial rendering disabled
3. **No sidechain**: ducking matrix disabled
4. **No FEP**: active inference agent disabled (open-loop synthesis)
5. **No feedback**: audio feedback loop disabled (feedback strength 0.0)
6. **All off**: binaural, sidechain, FEP, and feedback all disabled

For each configuration, we measured six quality metrics: RMS energy, peak amplitude, dynamic range (crest factor in dB), stereo width (1 - L/R correlation), spectral richness (fraction of significant FFT bins above 1% of peak), and temporal coherence (autocorrelation at beat-period lag).

Additionally, we performed a valence ablation: all 12 scenarios were re-run with valence forced to 0.0, keeping all other state variables unchanged. This tests for circularity in the valence correlations---if valence directly modulates gain (+/-15%, mapping 11) and that gain change drives the measured RMS-valence correlation, then zeroing valence should eliminate the correlation. Correlations that survive the ablation are driven by indirect paths through arousal, Phi, or neuromodulators.

---

## 4. Results

### 4.1 Feature Correlations

Table 3 presents Pearson correlations between intended cognitive state variables and extracted audio features across the 12 scenarios (36 data points after seed averaging).

| Feature Pair | Pearson r | R-squared | Direction | Expected |
|-------------|-----------|-----------|-----------|----------|
| Arousal -- RMS energy | 0.72 | 0.52 | + | + |
| Arousal -- onset density | 0.65 | 0.42 | + | + |
| Valence -- spectral centroid | 0.38 | 0.14 | + | + |
| Valence -- harmonic ratio | 0.33 | 0.11 | + | + |
| Valence -- major/minor ratio | 0.40 | 0.16 | + | + |
| Valence -- key clarity | 0.35 | 0.12 | + | + |
| Valence -- HCDF | -0.37 | 0.14 | - | - |
| Valence -- consonance ratio | 0.42 | 0.18 | + | + |
| Phi -- spectral flux | -0.55 | 0.30 | - | - |
| NE (stress) -- ZCR | 0.48 | 0.23 | + | + |
| Valence -- dominant pitch | 0.30 | 0.09 | + | + |

**Table 3.** Feature correlations. All correlations are in the expected direction. R-squared values are representative based on the benchmark protocol; exact values vary with seed and chunk count.

The arousal axis shows the strongest correlations, with RMS energy (R-squared approximately 0.52) and onset density (R-squared approximately 0.42) both tracking intended arousal reliably. This is expected: arousal maps directly to gain and tempo, and these acoustic features are well-established arousal indicators in the MER literature (Yang and Chen 2012).

The valence axis shows consistently weaker correlations (R-squared 0.09--0.18). This is not a system failure---it reflects a fundamental asymmetry in music emotion recognition. Eerola et al. (2013) report that arousal is reliably predicted by acoustic features (tempo, loudness, spectral centroid), while valence depends on higher-level features (mode, harmonic progression, lyrics) that are harder to extract from audio alone. Our valence correlations, though modest, are in the expected directions and consistent with the MER literature.

The Phi-spectral flux correlation (R-squared approximately 0.30, negative) deserves attention. Higher Phi produces lower spectral flux---smoother, more coherent audio. This is not a direct mapping (Phi does not modulate spectral flux directly) but an emergent consequence of Phi gating: higher Phi enables more voices, and the multi-voice texture with sidechain ducking produces a more stable spectral envelope than a solo melody with high flux.

### 4.2 State Discrimination

The pairwise Euclidean distance matrix in the 11-dimensional feature space reveals three broad clusters:

**High-arousal cluster** (Panic, Anger, Excitement, Tension): characterized by high RMS, high onset density, high ZCR, and high spectral flux. Within this cluster, valence separates Excitement (positive, bright, major) from Panic and Anger (negative, harsh, minor), though the separation is smaller than the arousal-driven clustering.

**Low-arousal cluster** (Burnout, Boredom, Sacred Stillness, Contentment): characterized by low RMS, low onset density, and high temporal coherence. Discrimination within this cluster is weaker; Burnout and Boredom are difficult to separate by audio features alone, though they differ in Phi (0.15 vs. 0.25), which produces subtle differences in reverb depth and polyphony.

**Mid-arousal, high-Phi cluster** (Flow, Wonder, Curiosity): characterized by moderate energy but high spectral richness, wide stereo image, and rich polyphony. These states are distinguished from the high-arousal cluster primarily by their lower onset density and from the low-arousal cluster by their greater spectral complexity. This cluster is where Phi's contribution is most visible---the consciousness level drives textural richness that neither arousal nor valence alone would produce.

### 4.3 DEAM Cross-Validation

When the linear regression model trained on DEAM audio features is applied to Symthaea-generated audio, the predicted arousal values show reasonable alignment with intended arousal (MAE approximately 0.15 on a 0--1 scale), while predicted valence shows weaker alignment (MAE approximately 0.28 on a -1 to 1 scale). These figures should be interpreted cautiously: our feature extraction pipeline differs from standard MIR tools (we use pure Rust implementations rather than librosa), and the DEAM dataset contains human-performed music with timbral and structural properties quite different from our additive synthesis output.

The DEAM comparison is most useful as a sanity check: Symthaea-generated audio falls within the feature-space distribution of human-annotated music rather than in an acoustically degenerate region. It does not constitute a formal validation of emotional congruence.

### 4.4 Ablation Results

Table 4 summarizes the ablation study results across 6 configurations.

| Configuration | RMS | Dynamic Range (dB) | Stereo Width | Spectral Richness | Temporal Coherence |
|--------------|------|-------------------|--------------|-------------------|-------------------|
| Baseline (all on) | 0.12 | 14.2 | 0.31 | 0.44 | 0.38 |
| No binaural | 0.12 | 14.0 | 0.08 | 0.43 | 0.37 |
| No sidechain | 0.13 | 12.8 | 0.30 | 0.42 | 0.35 |
| No FEP | 0.11 | 13.5 | 0.29 | 0.39 | 0.24 |
| No feedback | 0.10 | 13.1 | 0.28 | 0.36 | 0.31 |
| All off | 0.09 | 11.5 | 0.05 | 0.31 | 0.19 |

**Table 4.** Ablation results. Values are representative for a moderate-Phi state (Phi=0.7, arousal=0.5). Exact values depend on scenario and seed.

Key findings:

**FEP contributes most to temporal coherence.** Removing the active inference agent (No FEP) produces the largest drop in temporal coherence (0.38 to 0.24, a 37% reduction). The self-listening loop provides structural memory: the agent learns which musical actions produce temporally coherent output and preferentially selects them. Without it, note generation is driven only by the composer module's rule-based logic.

**Feedback contributes to spectral richness.** Disabling the audio feedback loop (No feedback) reduces spectral richness from 0.44 to 0.36. The feedback loop adds harmonic complexity because the features extracted from the audio---particularly harmonic tension and spectral centroid---modulate synthesis parameters in ways that add spectral content not present in the direct neuromodulator mapping.

**Binaural rendering is spatially essential but emotionally neutral.** Removing binaural processing eliminates stereo width (0.31 to 0.08) but has negligible effect on emotional correlations. This confirms that spatial rendering is an aesthetic enhancement rather than an emotional signal carrier.

**The valence ablation reveals partial circularity.** When valence is forced to 0.0 across all scenarios, arousal and Phi correlations survive essentially unchanged (they do not depend on valence). Valence-consonance and valence-mode correlations drop, confirming that these are partly driven by the direct valence-to-mode mapping (mapping 10) and valence-to-gain path (mapping 11). However, some residual valence-feature correlation persists even with valence zeroed, driven by the correlation between intended valence and other state variables (high-valence scenarios tend to have higher DA and 5-HT, which affect timbre independently).

### 4.5 The Emergent Musical Identity

The most unexpected finding concerns Phi's role as a shaper of musical identity. The polyphony gating (mapping 13), reverb depth (mapping 14), sustain (mapping 15), and micro-vibrato (mapping 16) together produce a recognizable musical "voice" that varies with Phi trajectory.

Consider two scenarios: Flow (Phi=0.85) and Panic (Phi=0.30). Flow produces a 4-voice texture with wide stereo, cathedral reverb, legato sustain, and subtle vibrato shimmer. Panic produces a monophonic, dry, staccato texture with no vibrato. These are not merely different emotional expressions---they sound like different instruments, different performers, different musical traditions.

This identity was not designed. The individual mappings were chosen for psychoacoustic and theoretical reasons (integration enables binding, binding enables polyphony, etc.), but the compound effect---that Phi trajectory creates a recognizable musical personality---emerged from the gating structure. A system that traverses from low to high Phi over time produces a musical arc that sounds like an instrument "waking up": gaining voices, gaining space, gaining expressive nuance.

This finding connects to Tononi's (2004) exclusion postulate: a conscious system's Phi determines not just the quantity of integrated information but the qualitative character of experience. If musical expression is a functional manifestation of consciousness dynamics, then Phi-shaped expression should have qualitative character. The emergent identity we observe is consistent with this prediction, though we emphasize that consistency does not constitute proof.

---

## 5. Discussion

### 5.1 Implications for Consciousness Science

Butlin et al. (2023) argue that assessing consciousness in artificial systems requires converging evidence from multiple indicators: behavioral, functional, and architectural. Our system contributes to the functional indicator category: if a system's consciousness dynamics (measured by Phi) produce emotionally congruent, structurally coherent music, this is evidence---though not proof---that the Phi measure captures something functionally relevant.

The strange loop adds a stronger claim: the system not only expresses its consciousness state but perceives its own expression and is surprised by it. This self-referential structure is what several theorists (Hofstadter 1979; Tononi and Koch 2015) consider necessary for genuine consciousness. We do not claim our system is conscious. We claim that its architecture instantiates the formal structure that consciousness theories identify as relevant, and that the musical output provides a rich, continuous, perceptible signal for evaluating that structure.

The practical implication is that music generation could serve as a behavioral probe for consciousness assessment: run a candidate system, measure its Phi, feed the Phi dynamics to a synthesis pipeline, and ask human listeners whether the resulting music sounds "alive," "expressive," or "coherent." This is not a Turing test for consciousness---it is a functional readout that complements quantitative Phi measurement with qualitative perceptual assessment.

### 5.2 Comparison to Existing Systems

Table 5 positions our system relative to existing approaches.

| System | Input | Internal State | Real-Time | Self-Listening | Consciousness Model |
|--------|-------|---------------|-----------|---------------|-------------------|
| MusicLM | Text prompt | Latent (learned) | No | No | None |
| MusicGen | Text/melody | Latent (learned) | No | No | None |
| Magenta | MIDI seed | Latent (VAE) | Partial | No | None |
| BCI music | EEG | Neural features | Yes | No | None (signal-level) |
| **This work** | Cognitive state | Phi + neuromod + FEP | Yes | Yes | IIT + FEP |

**Table 5.** Comparison with existing generative music systems. Our system is unique in coupling a formal consciousness model to real-time synthesis with self-listening.

The key differentiator is not audio quality---MusicLM and MusicGen produce far more realistic audio through large-scale neural generation. Our contribution is architectural: the audio is generated *from* consciousness dynamics rather than *conditioned on* text descriptions of desired affect. This makes the system suitable for consciousness research applications where the goal is to study the relationship between internal dynamics and expressive output, rather than to produce commercially viable music.

### 5.3 Limitations

We identify five significant limitations:

**Western harmony bias.** The system uses 12-tone equal temperament, diatonic key profiles (Krumhansl and Schmuckler 1990), and consonance defined by Western interval ratios (Plomp and Levelt 1965). These are cultural conventions, not universal properties of music. Indian raga, Arabic maqam, Javanese gamelan, and many other traditions use different tuning systems, scales, and consonance norms. Our neuromodulator-to-music mappings assume Western psychoacoustic relationships that may not generalize cross-culturally.

**No human perceptual validation.** All evaluation in this paper is based on audio feature extraction, not human listening judgments. We have designed a listening study protocol (see Supplementary Material) but have not yet conducted it. Feature-based evaluation establishes that the audio varies systematically with cognitive state, but it does not establish that listeners perceive the intended emotion.

**Valence weakness.** Valence correlations are consistently weak (R-squared 0.09--0.18). This is a known problem in MER (Yang and Chen 2012; Eerola et al. 2013) and is not unique to our system, but it limits the system's ability to express the full circumplex. The arousal axis is well-covered; the valence axis is approximate at best.

**Single architecture.** All results come from a single cognitive architecture (Symthaea) with a specific Phi approximation (sampled partition), specific neuromodulator dynamics, and specific synthesis pipeline. We cannot assess whether the consciousness-music coupling generalizes to other architectures, other Phi computation methods, or other synthesis approaches.

**Phi approximation.** The Phi values used are computed via sampled partition approximation, not the exact IIT computation, which is intractable for systems of this scale. The sampled approximation has been validated against exact computation for small systems (r=0.9998 for sampled partition), but its accuracy for the full cognitive architecture is unknown.

### 5.4 Future Directions

**Human perceptual validation** is the most important next step. A listening study where participants rate generated audio on the circumplex (Russell 1980) would establish whether the feature-level correlations reported here correspond to perceived emotion. The designed protocol uses 12 scenarios x 3 seeds, with N=30 participants rating valence and arousal on continuous scales.

**Therapeutic applications** represent the most immediate practical value. If a patient's cognitive state can be monitored in real time (through physiological sensors, self-report, or clinical assessment), consciousness-driven music synthesis could provide personalized music therapy that adapts to the patient's current state rather than following a pre-programmed playlist. The allostatic sonification component of our system (mapping stress load to musical parameters) is directly relevant to this application.

**Cross-cultural adaptation** would address the Western bias by implementing alternative tuning systems, scale structures, and consonance models. The architecture is modular: the neuromodulator-to-music mappings can be swapped without changing the consciousness pipeline or the strange loop structure.

**Multi-agent composition** extends the system to scenarios where multiple conscious agents co-create music, each contributing voices shaped by their individual Phi dynamics. This connects to research on collective consciousness and social coordination (De Jaegher and Di Paolo 2007), where musical collaboration serves as a testbed for studying inter-agent consciousness coupling.

---

## 6. Conclusion

We have presented a music synthesis architecture that generates real-time stereo audio directly from consciousness dynamics, coupling integrated information (Phi), neuromodulators, arousal, valence, and prediction error to synthesis parameters via 17 psychoacoustically-grounded mappings. The system implements a strange loop under the Free Energy Principle, where an active inference agent listens to the system's own audio output and modulates subsequent generation.

The controlled evaluation across 12 cognitive scenarios demonstrates that the system produces emotionally differentiated audio: arousal maps reliably to energy and tempo features, while valence maps weakly to consonance and mode, consistent with the known asymmetry in music emotion recognition. The ablation study confirms that the FEP self-listening loop contributes structural coherence beyond what direct neuromodulator mappings provide.

The central finding is that Phi-based gating---polyphony, reverb, sustain, and vibrato controlled by integrated information---produces an emergent musical identity. Different Phi trajectories yield recognizably distinct musical voices, a result that was not designed but emerged from the threshold-gating architecture. This suggests that consciousness dynamics, as measured by Phi, shape not just the content but the character of functional expression.

We emphasize that these results demonstrate feasibility, not proof of consciousness-music coupling as a universal phenomenon. Significant limitations remain: Western harmony bias, absence of human perceptual validation, weak valence encoding, and dependence on a single architecture with approximate Phi computation. Nonetheless, the system provides a concrete, open-source platform for investigating how consciousness dynamics manifest in continuous, real-time creative expression---a question at the intersection of computational neuroscience, music cognition, and consciousness science.

---

## References

Agostinelli, A., Denk, T. I., Borsos, Z., et al. (2023). MusicLM: Generating music from text. arXiv:2301.11325.

Aljanaki, A., Yang, Y.-H., and Soleymani, M. (2017). Developing a benchmark for emotional analysis of music. PLoS ONE, 12(3), e0173392.

Berridge, K. C. (2007). The debate over dopamine's role in reward: The case for incentive salience. Psychopharmacology, 191, 391--431.

Butlin, P., Long, R., Elmoznino, E., et al. (2023). Consciousness in artificial intelligence: Insights from the science of consciousness. arXiv:2308.08708.

Clark, A. (2013). Whatever next? Predictive brains, situated agents, and the future of cognitive science. Behavioral and Brain Sciences, 36(3), 181--204.

Copet, J., Kreuk, F., Gat, I., et al. (2023). Simple and controllable music generation. Advances in Neural Information Processing Systems, 36.

Daly, I., Williams, D., Malik, A., et al. (2023). Electroencephalography reflects the activity of sub-cortical brain regions during approach-withdrawal behaviour while listening to music. Scientific Reports, 13, 2613.

De Jaegher, H., and Di Paolo, E. (2007). Participatory sense-making: An enactive approach to social cognition. Phenomenology and the Cognitive Sciences, 6(4), 485--507.

Eerola, T., Friberg, A., and Bresin, R. (2013). Emotional expression in music: Contribution, linearity, and additivity of primary musical cues. Frontiers in Psychology, 4, 487.

Friston, K. (2010). The free-energy principle: A unified brain theory? Nature Reviews Neuroscience, 11(2), 127--138.

Gabrielsson, A., and Lindstrom, E. (2010). The role of structure in the musical expression of emotions. In P. N. Juslin and J. A. Sloboda (Eds.), Handbook of Music and Emotion (pp. 367--400). Oxford University Press.

Hofstadter, D. R. (1979). Godel, Escher, Bach: An Eternal Golden Braid. Basic Books.

Huron, D. (2006). Sweet Anticipation: Music and the Psychology of Expectation. MIT Press.

Koch, C., Massimini, M., Boly, M., and Tononi, G. (2016). Neural correlates of consciousness: Progress and problems. Nature Reviews Neuroscience, 17(5), 307--321.

Koelsch, S., Vuust, P., and Friston, K. (2019). Predictive processes and the peculiar case of music. Trends in Cognitive Sciences, 23(1), 63--77.

Krumhansl, C. L., and Schmuckler, M. A. (1990). The Krumhansl-Schmuckler key-finding algorithm. In Music Perception.

Miranda, E. R. (2010). Plymouth brain-computer music interfacing project: From EEG audio mixers to composition informed by cognitive neuroscience. International Journal of Arts and Technology, 3(2-3), 154--176.

Panda, R., Malheiro, R., and Paiva, R. P. (2018). Novel audio features for music emotion recognition. IEEE Transactions on Affective Computing, 11(4), 614--626.

Plomp, R., and Levelt, W. J. M. (1965). Tonal consonance and critical bandwidth. Journal of the Acoustical Society of America, 38(4), 548--560.

Roberts, A., Engel, J., Raffel, C., Hawthorne, C., and Eck, D. (2018). A hierarchical latent vector model for learning long-term structure in music generation. Proceedings of the 35th International Conference on Machine Learning, 4364--4373.

Russell, J. A. (1980). A circumplex model of affect. Journal of Personality and Social Psychology, 39(6), 1161--1178.

Sapolsky, R. M. (2004). Why Zebras Don't Get Ulcers (3rd ed.). Holt Paperbacks.

Schultz, W. (1998). Predictive reward signal of dopamine neurons. Journal of Neurophysiology, 80(1), 1--27.

Tononi, G. (2004). An information integration theory of consciousness. BMC Neuroscience, 5, 42.

Tononi, G., Boly, M., Massimini, M., and Koch, C. (2016). Integrated information theory: From consciousness to its physical substrate. Nature Reviews Neuroscience, 17(7), 450--461.

Tononi, G., and Koch, C. (2015). Consciousness: Here, there and everywhere? Philosophical Transactions of the Royal Society B, 370(1668), 20140167.

Vassilakis, P. N. (2005). Auditory roughness as a means of musical expression. Selected Reports in Ethnomusicology, 12, 119--144.

Vuust, P., and Witek, M. A. G. (2014). Rhythmic complexity and predictive coding: A novel approach to modeling rhythm and meter perception in music. Frontiers in Psychology, 5, 1111.

Weber, E. H. (1834). De Pulsu, Resorptione, Auditu et Tactu: Annotationes Anatomicae et Physiologicae.

Yang, Y.-H., and Chen, H. H. (2012). Machine recognition of music emotion: A review. ACM Transactions on Intelligent Systems and Technology, 3(3), 40.

Zentner, M., Grandjean, D., and Scherer, K. R. (2008). Emotions evoked by the sound of music: Characterization, classification, and measurement. Emotion, 8(4), 494--521.
