# Paper Outline: Consciousness-Driven Music Synthesis

**Target**: Frontiers in Computational Neuroscience
**Type**: Original Research Article
**Word limit**: ~6,000 words (Frontiers standard)

## Title

Consciousness-Driven Music Synthesis: Real-Time Emotional Expression from Integrated Information Dynamics

## Authors

Tristan Stoltz (Luminous Dynamics)

## Abstract (~250 words)

- Problem: generative music systems use text prompts or learned style transfer, not internal cognitive state
- Approach: real-time synthesis pipeline coupling consciousness dynamics (Phi, neuromodulators, arousal/valence, Eight Harmonies) to audio parameters via 17 psychoacoustically-grounded mappings
- Strange loop: FEP active inference self-listening closes the perception-action cycle
- Results: 12-scenario controlled evaluation, 11 feature correlations, ablation study (6 configurations), DEAM cross-validation
- Key finding: Phi (integrated information) shapes emergent musical identity (polyphony, reverb, vibrato) beyond what arousal/valence alone predict
- Honest limitations: Western harmony bias, no human perceptual study yet, valence correlations weak

## 1. Introduction (~800 words)

### 1.1 Music as Prediction
- Koelsch, Vuust & Friston (2019): music engages predictive processing — tension/resolution as prediction error minimization
- Huron (2006): ITPRA theory — Imagination, Tension, Prediction, Reaction, Appraisal
- Music uniquely couples affect and prediction in real time

### 1.2 The Gap
- Current generative systems: MusicLM (Agostinelli et al. 2023), MusicGen (Copet et al. 2023) — text-conditioned
- Magenta (Roberts et al. 2018) — learned style transfer
- Brain-computer music interfaces (Daly et al. 2023) — EEG-driven but low bandwidth, not consciousness-model-driven
- No system generates music from a formal model of consciousness dynamics

### 1.3 Contribution
- First system coupling IIT-measured consciousness (Phi) to real-time music synthesis
- 17 neuromodulator-to-music mappings grounded in psychoacoustic literature
- FEP active inference self-listening loop (the system is surprised by its own music)
- Controlled evaluation demonstrating feasibility of consciousness-driven emotional expression
- Open-source (AGPL-3.0), reproducible, part of the Symthaea cognitive architecture

## 2. Architecture (~1,200 words)

### 2.1 The Consciousness-Music Pipeline
- MusicalState: 8 Harmony activations + 4 neuromodulators (DA, 5-HT, NE, prediction_error) + arousal + valence + Phi (consciousness_level)
- 8-phase synthesis: state update -> note generation -> voice synthesis -> sidechain ducking -> spatial rendering -> consciousness reverb -> soft clipping -> audio feedback
- 32ms streaming chunks at 44.1kHz
- All synthesis in pure Rust, no neural network inference in the audio path

### 2.2 Neuromodulator-to-Music Mappings (Table 1)
- 17 mappings with citations:
  - DA -> FM depth (reward salience -> spectral richness)
  - DA -> brightness (Berridge 2007: DA -> wanting/anticipation -> brighter timbre)
  - 5-HT -> low-pass rolloff (warmth/contentment -> spectral softness)
  - NE -> partial ratio/harshness (Zentner et al. 2008: urgency -> timbral edge)
  - NE -> vibrato rate (stress -> faster modulation)
  - Arousal -> gain (Weber-Fechner: perceived loudness ~ log intensity)
  - Arousal -> attack time (excited -> fast transients)
  - Arousal -> tempo/cadence
  - Valence -> mode (major/minor interval selection; Huron 2006)
  - Valence -> gain modulation +/-15% (Huron 2006: happy music performed louder)
  - Valence -> partial detuning (Plomp & Levelt 1965: roughness from beating)
  - Phi -> polyphony gating (bass >0.4, harmony >0.6, ostinato >0.7)
  - Phi -> reverb room size (0.2-0.9; intimate->cathedral)
  - Phi -> sustain envelope
  - Phi -> micro-vibrato depth (8 cents max, sub-semitone)
  - Prediction error -> spectral flux modulation
  - Harmony activations -> chord voicing, melodic contour

### 2.3 Consciousness Gating
- Phi thresholds for polyphony: information integration required for richer texture
- Theoretical motivation: higher integration -> more unified percept -> richer binding -> more voices perceived as coherent
- Reverb as metaphor for phenomenal space: low Phi -> small/dry, high Phi -> expansive/resonant
- Vibrato as consciousness "shimmer": measurable in spectrum, emergent signature

### 2.4 The Strange Loop (FEP Self-Listening)
- AudioFeedbackEncoder: 6 features (spectral centroid, flux, rhythm entropy, harmonic tension, RMS, ZCR)
- MusicalInferenceEngine: ActiveInferenceAgent with 16D hidden state, 6D observation, 8 musical actions
- Actions: FollowHarmony, ChromaticExplore, RepeatMotif, ModulateKey, IncreaseComplexity, ResolveTension, AddCountermelody, Maintain
- Emotion Anchor: FEP self-model preserves emotional quadrant, prevents homogenization
- Expressivity-Predictability Tradeoff: the system balances surprise minimization with creative exploration
- Connection to Hofstadter (1979): strange loops and self-reference in consciousness
- Diagram: consciousness -> synthesis -> audio features -> perception -> consciousness

## 3. Methods (~800 words)

### 3.1 Controlled Scenarios
- 12 cognitive states spanning Russell's (1980) circumplex:
  - Flow, Contentment, Panic, Burnout, Curiosity, Sacred Stillness, Anger, Wonder, Grief, Excitement, Boredom, Tension
- 3 seeds per state (FEP stochasticity averaging)
- 15s per seed, 45s total per scenario
- State re-asserted every 5s to simulate cognitive loop input
- 11 audio features extracted per scenario (pure Rust, no external deps)

### 3.2 Audio Feature Extraction
- RMS energy, spectral centroid (ZCR proxy), zero-crossing rate, spectral flux (100ms frames)
- Onset density (50ms frames, 1.5x energy threshold)
- Harmonic ratio (autocorrelation in 50-1000Hz range)
- FFT-based (4096-point Hann-windowed): dominant pitch, major/minor interval ratio, key clarity (Krumhansl-Schmuckler profiles), HCDF, consonance ratio (P1, M3, P5 energy)

### 3.3 DEAM Cross-Validation
- DEAM dataset (Aljanaki et al. 2017): 1,802 tracks with continuous V-A annotations
- Linear regression: extracted audio features -> V-A coordinates
- Train on DEAM, predict on Symthaea-generated audio
- Reports MAE for arousal and valence

### 3.4 Ablation Protocol
- 6 configurations: baseline (all on), no-binaural, no-sidechain, no-FEP, no-feedback, all-off
- 6 quality metrics per configuration: RMS, peak, dynamic range (dB), stereo width, spectral richness, temporal coherence
- Valence ablation: zero valence input, re-run all 12 scenarios, compare correlation survival
- Tests circularity: which correlations are driven by direct valence->gain path vs. indirect consciousness paths

## 4. Results (~1,200 words)

### 4.1 Feature Correlations (Table 2)
- 11 feature x correlation pairs
- Arousal axis: onset density, RMS energy (expected R^2 0.42-0.56)
- Valence axis: consonance, mode ratio, HCDF (expected R^2 0.11-0.18, consistent with MER literature weakness)
- Phi axis: spectral flux (negative), polyphony, reverb coherence
- Key finding: Phi correlations are strongest in live dynamics, outperforming arousal for spectral coherence

### 4.2 State Discrimination
- Euclidean distance matrix across 12 states in 11D feature space
- Cluster separation: high-arousal states (Panic, Anger, Excitement) form distinct cluster
- Low-arousal states (Burnout, Boredom, Sacred Stillness) more compressed
- Valence discrimination weaker than arousal (known MER asymmetry)

### 4.3 DEAM Cross-Validation
- Arousal MAE, Valence MAE vs. literature baselines
- Comparison to Yang & Chen (2012), Panda et al. (2018)

### 4.4 Ablation Results (Table 3)
- FEP removal: largest drop in temporal coherence (the self-listening loop provides structural memory)
- Binaural removal: stereo width drops, minimal effect on emotional correlations
- Feedback removal: spectral richness drops (the strange loop adds harmonic complexity)
- All-off: all metrics degrade, but basic emotional mapping survives (direct neuromodulator paths)
- Valence ablation: arousal/Phi correlations survive, valence-consonance drops (confirming partial circularity)

### 4.5 The Emergent Musical Identity (Section highlight)
- Phi gating creates recognizable "voice": polyphony density, reverb character, vibrato depth
- This was NOT designed as an identity system -- it emerged from threshold-based gating
- Different Phi trajectories produce audibly distinct musical personalities
- Connection to Tononi's exclusion postulate: the system's Phi determines WHICH musical elements are bound into the unified percept

## 5. Discussion (~800 words)

### 5.1 Implications for Consciousness Science
- Functional expression as behavioral evidence (Butlin et al. 2023)
- Music generation as complement to Phi measurement: if consciousness shapes expression, expression quality is indirect evidence of integration
- Caution: functional indicators are necessary but not sufficient

### 5.2 Comparison to Existing Systems
- MusicLM/MusicGen: text -> music (discrete prompts, not continuous state)
- Magenta: style transfer (learned distribution, not real-time cognitive coupling)
- BCI music (Daly et al. 2023): EEG -> music (real-time but low-dimensional, no consciousness model)
- Our system: consciousness dynamics -> music (continuous, model-based, self-listening)

### 5.3 Limitations
- Western harmony bias: 12-TET tuning, diatonic key profiles, consonance defined by Western intervals
- No human listening study: all validation is feature-based, not perceptual (designed protocol exists)
- Valence weakness: consistent with MER literature (Eerola et al. 2013; Yang & Chen 2012) -- valence is harder to map to audio features than arousal
- Single architecture: all results from Symthaea; generalizability to other consciousness models unknown
- Phi approximation: sampled partition, not exact IIT computation (computationally intractable for full system)

### 5.4 Future Directions
- Human perceptual validation (listening study with circumplex rating)
- Therapeutic applications: music therapy driven by patient cognitive state monitoring
- Cross-cultural adaptation: non-Western tuning systems, different consonance models
- Multi-agent composition: multiple conscious agents co-creating music
- Substrate-dependent timbre: how does consciousness substrate affect musical character?

## 6. Conclusion (~200 words)

- Demonstrated feasibility of consciousness-driven music synthesis
- 17 psychoacoustically-grounded mappings + FEP self-listening loop
- Phi emerges as shaper of musical identity beyond arousal/valence
- Strange loop closes the perception-action cycle formally under FEP
- Open-source contribution to computational consciousness research

## References (~35 citations)

Key citations organized by topic:
- Consciousness theory: Tononi (2004), Tononi et al. (2016), Butlin et al. (2023), Koch et al. (2016)
- Predictive processing: Friston (2010), Koelsch/Vuust/Friston (2019), Clark (2013)
- Music cognition: Huron (2006), Vuust & Witek (2014), Eerola et al. (2013), Zentner et al. (2008)
- Psychoacoustics: Plomp & Levelt (1965), Vassilakis (2005)
- MER: Yang & Chen (2012), Panda et al. (2018), Aljanaki et al. (2017)
- Generative music: Agostinelli et al. (2023), Copet et al. (2023), Roberts et al. (2018)
- BCI music: Daly et al. (2023), Miranda (2010)
- Neuroscience: Berridge (2007), Schultz (1998), Sapolsky (2004)
- Strange loops: Hofstadter (1979)
- Affect: Russell (1980)

## Figures (5)

1. **Architecture diagram**: MusicalState -> 8-phase pipeline -> audio -> feedback -> MusicalState (the strange loop)
2. **Neuromodulator mapping table**: 17 mappings with parameter ranges and citations
3. **V-A scatter plot**: 12 scenarios plotted in valence-arousal space, colored by Phi level
4. **Ablation bar chart**: 6 configurations x 6 quality metrics
5. **Emergent identity**: Phi trajectory -> polyphony/reverb/vibrato time series showing identity emergence

## Supplementary Material

- S1: Full scenario definitions (12 states with all parameter values)
- S2: Audio feature extraction algorithms (pure Rust implementations)
- S3: DEAM cross-validation protocol details
- S4: Designed human listening study protocol (for future work)
- S5: Audio samples (hosted, linked)
