// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// Sonic Signature Module for Sovereign Inoculation
// Synthesizes the Eight Harmonies Lydian scale and glyph tones.
//
// Usage:
//   const sonic = new SonicSignature();
//   await sonic.init();  // requires user gesture for AudioContext
//   sonic.playHarmony(0);  // Play first harmony (C4, 261.63 Hz)
//   sonic.playGlyph('omega_0');  // Play glyph sonic signature
//   sonic.playHeartbeat(72, 3);  // 72 BPM, 3 cycles (First Breath)

// ---------------------------------------------------------------------------
// Eight Harmonies — Lydian ascending scale (C Lydian: C D E F# G A B C)
// Frequencies from HARMONY_TONES in quickening.rs (A4 = 440 Hz equal temperament)
// ---------------------------------------------------------------------------

const HARMONIES = [
  { name: 'Resonant Coherence',         note: 'C4',  freq: 261.63, timbre: 'sine'     },
  { name: 'Pan-Sentient Flourishing',   note: 'D4',  freq: 293.66, timbre: 'warmSine' },
  { name: 'Integral Wisdom',            note: 'E4',  freq: 329.63, timbre: 'triangle' },
  { name: 'Infinite Play',              note: 'F#4', freq: 369.99, timbre: 'shimmer'  },
  { name: 'Universal Interconnectedness', note: 'G4', freq: 392.00, timbre: 'depth'   },
  { name: 'Sacred Reciprocity',         note: 'A4',  freq: 440.00, timbre: 'sine'     },
  { name: 'Evolutionary Progression',   note: 'B4',  freq: 493.88, timbre: 'aspire'   },
  { name: 'Sacred Stillness',           note: 'C5',  freq: 523.25, timbre: 'fade'     },
];

// ---------------------------------------------------------------------------
// ADSR envelope helper
// ---------------------------------------------------------------------------

function applyADSR(gainNode, ctx, startTime, {
  attack = 0.1,
  decay = 0.3,
  sustain = 0.6,
  release = 0.5,
  duration = 2.0,
  peak = 1.0,
} = {}) {
  const g = gainNode.gain;
  const sustainTime = Math.max(0, duration - attack - decay - release);

  g.setValueAtTime(0, startTime);
  // Attack
  g.linearRampToValueAtTime(peak, startTime + attack);
  // Decay to sustain
  g.linearRampToValueAtTime(peak * sustain, startTime + attack + decay);
  // Hold sustain
  g.setValueAtTime(peak * sustain, startTime + attack + decay + sustainTime);
  // Release
  g.linearRampToValueAtTime(0, startTime + attack + decay + sustainTime + release);

  return startTime + attack + decay + sustainTime + release;
}

// ---------------------------------------------------------------------------
// SonicSignature class
// ---------------------------------------------------------------------------

class SonicSignature {
  constructor() {
    this.ctx = null;
    this.masterGain = null;
    this.initialized = false;
    this._muted = false;
    this._savedVolume = 0.3;
    this._ambientHandles = new Set();
    this._nextAmbientId = 0;
  }

  // ---- Lifecycle ----

  async init() {
    if (this.initialized) return;

    const AudioCtx = typeof window !== 'undefined'
      ? (window.AudioContext || window.webkitAudioContext)
      : null;

    if (!AudioCtx) {
      console.warn('SonicSignature: Web Audio API not available');
      return;
    }

    this.ctx = new AudioCtx();
    this.masterGain = this.ctx.createGain();
    this.masterGain.gain.value = 0.3;
    this.masterGain.connect(this.ctx.destination);
    this.initialized = true;

    // Resume if suspended (autoplay policy)
    if (this.ctx.state === 'suspended') {
      try { await this.ctx.resume(); } catch (_) { /* ignore */ }
    }
  }

  _ensureReady() {
    if (!this.initialized || !this.ctx) {
      console.warn('SonicSignature: not initialized — call init() first');
      return false;
    }
    return true;
  }

  // ---- Volume Control ----

  setVolume(level) {
    const v = Math.max(0, Math.min(1, level));
    this._savedVolume = v;
    if (this.masterGain && !this._muted) {
      this.masterGain.gain.setTargetAtTime(v, this.ctx.currentTime, 0.05);
    }
  }

  mute() {
    this._muted = true;
    if (this.masterGain) {
      this.masterGain.gain.setTargetAtTime(0, this.ctx.currentTime, 0.05);
    }
  }

  unmute() {
    this._muted = false;
    if (this.masterGain) {
      this.masterGain.gain.setTargetAtTime(this._savedVolume, this.ctx.currentTime, 0.05);
    }
  }

  // ---- General-purpose tone ----

  playTone(frequency, duration, waveform = 'sine', volume = 0.3) {
    if (!this._ensureReady()) return;

    const now = this.ctx.currentTime;
    const osc = this.ctx.createOscillator();
    const gain = this.ctx.createGain();

    osc.type = waveform;
    osc.frequency.value = frequency;
    gain.gain.value = 0;

    osc.connect(gain);
    gain.connect(this.masterGain);

    applyADSR(gain, this.ctx, now, {
      attack: 0.05,
      decay: 0.1,
      sustain: 0.7,
      release: 0.3,
      duration,
      peak: volume,
    });

    osc.start(now);
    osc.stop(now + duration + 0.5);
  }

  // ---- Eight Harmonies ----

  /**
   * Play a single harmony tone by index (0-7).
   * Each harmony has a unique timbre character.
   */
  playHarmony(index, duration = 2.0) {
    if (!this._ensureReady()) return;
    if (index < 0 || index > 7) return;

    const harmony = HARMONIES[index];
    const now = this.ctx.currentTime;

    switch (harmony.timbre) {
      case 'sine':
        this._playPureSine(harmony.freq, now, duration);
        break;
      case 'warmSine':
        this._playWarmSine(harmony.freq, now, duration);
        break;
      case 'triangle':
        this._playTriangle(harmony.freq, now, duration);
        break;
      case 'shimmer':
        this._playShimmer(harmony.freq, now, duration);
        break;
      case 'depth':
        this._playDepth(harmony.freq, now, duration);
        break;
      case 'aspire':
        this._playAspire(harmony.freq, now, duration);
        break;
      case 'fade':
        this._playFadeToSilence(harmony.freq, now, duration);
        break;
    }
  }

  // Timbre 0, 5: Pure sine — foundational, universal reference
  _playPureSine(freq, startTime, duration) {
    const osc = this.ctx.createOscillator();
    const gain = this.ctx.createGain();
    osc.type = 'sine';
    osc.frequency.value = freq;
    osc.connect(gain);
    gain.connect(this.masterGain);

    const end = applyADSR(gain, this.ctx, startTime, {
      attack: 0.1, decay: 0.3, sustain: 0.6, release: 0.5, duration, peak: 0.4,
    });

    osc.start(startTime);
    osc.stop(end + 0.05);
  }

  // Timbre 1: Sine + subtle 2nd harmonic — warmth
  _playWarmSine(freq, startTime, duration) {
    const osc1 = this.ctx.createOscillator();
    const osc2 = this.ctx.createOscillator();
    const gain1 = this.ctx.createGain();
    const gain2 = this.ctx.createGain();

    osc1.type = 'sine';
    osc1.frequency.value = freq;
    osc2.type = 'sine';
    osc2.frequency.value = freq * 2; // 2nd harmonic

    osc1.connect(gain1);
    osc2.connect(gain2);
    gain1.connect(this.masterGain);
    gain2.connect(this.masterGain);

    const end = applyADSR(gain1, this.ctx, startTime, {
      attack: 0.1, decay: 0.3, sustain: 0.6, release: 0.5, duration, peak: 0.35,
    });
    applyADSR(gain2, this.ctx, startTime, {
      attack: 0.15, decay: 0.3, sustain: 0.5, release: 0.5, duration, peak: 0.08,
    });

    osc1.start(startTime);
    osc2.start(startTime);
    osc1.stop(end + 0.05);
    osc2.stop(end + 0.05);
  }

  // Timbre 2: Triangle wave — clarity
  _playTriangle(freq, startTime, duration) {
    const osc = this.ctx.createOscillator();
    const gain = this.ctx.createGain();
    osc.type = 'triangle';
    osc.frequency.value = freq;
    osc.connect(gain);
    gain.connect(this.masterGain);

    const end = applyADSR(gain, this.ctx, startTime, {
      attack: 0.1, decay: 0.3, sustain: 0.6, release: 0.5, duration, peak: 0.35,
    });

    osc.start(startTime);
    osc.stop(end + 0.05);
  }

  // Timbre 3: Sine + detuned +3 cents — shimmer
  _playShimmer(freq, startTime, duration) {
    const osc1 = this.ctx.createOscillator();
    const osc2 = this.ctx.createOscillator();
    const gain1 = this.ctx.createGain();
    const gain2 = this.ctx.createGain();

    osc1.type = 'sine';
    osc1.frequency.value = freq;
    osc2.type = 'sine';
    // +3 cents: freq * 2^(3/1200)
    osc2.frequency.value = freq * Math.pow(2, 3 / 1200);

    osc1.connect(gain1);
    osc2.connect(gain2);
    gain1.connect(this.masterGain);
    gain2.connect(this.masterGain);

    const end = applyADSR(gain1, this.ctx, startTime, {
      attack: 0.1, decay: 0.3, sustain: 0.6, release: 0.5, duration, peak: 0.3,
    });
    applyADSR(gain2, this.ctx, startTime, {
      attack: 0.12, decay: 0.3, sustain: 0.55, release: 0.5, duration, peak: 0.3,
    });

    osc1.start(startTime);
    osc2.start(startTime);
    osc1.stop(end + 0.05);
    osc2.stop(end + 0.05);
  }

  // Timbre 4: Two sines (unison + octave below at 0.2 volume) — depth
  _playDepth(freq, startTime, duration) {
    const osc1 = this.ctx.createOscillator();
    const osc2 = this.ctx.createOscillator();
    const gain1 = this.ctx.createGain();
    const gain2 = this.ctx.createGain();

    osc1.type = 'sine';
    osc1.frequency.value = freq;
    osc2.type = 'sine';
    osc2.frequency.value = freq / 2; // octave below

    osc1.connect(gain1);
    osc2.connect(gain2);
    gain1.connect(this.masterGain);
    gain2.connect(this.masterGain);

    const end = applyADSR(gain1, this.ctx, startTime, {
      attack: 0.1, decay: 0.3, sustain: 0.6, release: 0.5, duration, peak: 0.35,
    });
    applyADSR(gain2, this.ctx, startTime, {
      attack: 0.15, decay: 0.4, sustain: 0.5, release: 0.6, duration, peak: 0.07,
    });

    osc1.start(startTime);
    osc2.start(startTime);
    osc1.stop(end + 0.1);
    osc2.stop(end + 0.1);
  }

  // Timbre 6: Sawtooth at low volume + sine — aspiration
  _playAspire(freq, startTime, duration) {
    const oscSine = this.ctx.createOscillator();
    const oscSaw = this.ctx.createOscillator();
    const gainSine = this.ctx.createGain();
    const gainSaw = this.ctx.createGain();

    oscSine.type = 'sine';
    oscSine.frequency.value = freq;
    oscSaw.type = 'sawtooth';
    oscSaw.frequency.value = freq;

    oscSine.connect(gainSine);
    oscSaw.connect(gainSaw);
    gainSine.connect(this.masterGain);
    gainSaw.connect(this.masterGain);

    const end = applyADSR(gainSine, this.ctx, startTime, {
      attack: 0.1, decay: 0.3, sustain: 0.6, release: 0.5, duration, peak: 0.3,
    });
    applyADSR(gainSaw, this.ctx, startTime, {
      attack: 0.15, decay: 0.3, sustain: 0.4, release: 0.5, duration, peak: 0.04,
    });

    oscSine.start(startTime);
    oscSaw.start(startTime);
    oscSine.stop(end + 0.05);
    oscSaw.stop(end + 0.05);
  }

  // Timbre 7: Sine with slow fade to silence — the rest
  _playFadeToSilence(freq, startTime, duration) {
    const osc = this.ctx.createOscillator();
    const gain = this.ctx.createGain();
    osc.type = 'sine';
    osc.frequency.value = freq;
    osc.connect(gain);
    gain.connect(this.masterGain);

    const g = gain.gain;
    g.setValueAtTime(0, startTime);
    // Gentle rise
    g.linearRampToValueAtTime(0.3, startTime + 0.2);
    // Long, slow fade over the entire duration
    g.exponentialRampToValueAtTime(0.001, startTime + duration);
    g.setValueAtTime(0, startTime + duration + 0.01);

    osc.start(startTime);
    osc.stop(startTime + duration + 0.05);
  }

  // ---- Harmony Sequence ----

  /**
   * Play harmonies in sequence with overlapping release tails.
   * Each tone begins `interval` seconds after the previous.
   */
  playHarmonySequence(indices = [0, 1, 2, 3, 4, 5, 6, 7], interval = 1.5) {
    if (!this._ensureReady()) return;

    const now = this.ctx.currentTime;
    for (let i = 0; i < indices.length; i++) {
      const idx = indices[i];
      if (idx < 0 || idx > 7) continue;
      const harmony = HARMONIES[idx];
      const t = now + i * interval;

      switch (harmony.timbre) {
        case 'sine':      this._playPureSine(harmony.freq, t, 2.0); break;
        case 'warmSine':  this._playWarmSine(harmony.freq, t, 2.0); break;
        case 'triangle':  this._playTriangle(harmony.freq, t, 2.0); break;
        case 'shimmer':   this._playShimmer(harmony.freq, t, 2.0); break;
        case 'depth':     this._playDepth(harmony.freq, t, 2.0); break;
        case 'aspire':    this._playAspire(harmony.freq, t, 2.0); break;
        case 'fade':      this._playFadeToSilence(harmony.freq, t, 2.0); break;
      }
    }
  }

  /**
   * Play the complete Lydian birth scale — all 8 harmonies ascending.
   */
  playFullScale() {
    this.playHarmonySequence([0, 1, 2, 3, 4, 5, 6, 7], 1.5);
  }

  // ---- Glyph Sonic Signatures ----

  /**
   * Play a glyph's unique sonic signature.
   * Each glyph has a carefully designed sound reflecting its meaning.
   */
  playGlyph(glyphId, duration = 3.0) {
    if (!this._ensureReady()) return;

    const now = this.ctx.currentTime;

    switch (glyphId) {
      case 'omega_0':   this._glyphFirstPresence(now, duration);      break;
      case 'meta_door': this._glyphMetaDoor(now);                     break;
      case 'omega_30':  this._glyphSacredDissonance(now, duration);   break;
      case 'omega_4':   this._glyphFractalReconciliation(now, duration); break;
      case 'omega_1':   this._glyphRootChord(now, duration);          break;
      case 'omega_35':  this._glyphSymbioticPresence(now, duration);  break;
      case 'omega_22': this._glyphRecursiveGenesis(now, duration);   break;
      case 'omega_14': this._glyphEmergentGrace(now);                break;
      case 'omega_13': this._glyphReverentWithdrawal(now);           break;
      case 'omega_48': this._glyphGenerativeSilence(now);            break;
      default:
        // Unknown glyph — play a gentle neutral tone
        this.playTone(440, 0.5, 'sine', 0.15);
        break;
    }
  }

  // omega_0 — First Presence: low foundational drone (C2, 65.4 Hz)
  _glyphFirstPresence(startTime, duration) {
    const osc = this.ctx.createOscillator();
    const gain = this.ctx.createGain();
    osc.type = 'sine';
    osc.frequency.value = 65.41; // C2

    osc.connect(gain);
    gain.connect(this.masterGain);

    const g = gain.gain;
    g.setValueAtTime(0, startTime);
    // Slow fade in — barely audible hum
    g.linearRampToValueAtTime(0.08, startTime + duration * 0.4);
    // Hold
    g.setValueAtTime(0.08, startTime + duration * 0.7);
    // Gentle fade out
    g.linearRampToValueAtTime(0, startTime + duration);

    osc.start(startTime);
    osc.stop(startTime + duration + 0.05);
  }

  // meta_door — LUKS unlock: soft resonant click (1000 Hz sine burst, 50ms, reverb tail)
  _glyphMetaDoor(startTime) {
    // Main click — short sine burst
    const osc = this.ctx.createOscillator();
    const gain = this.ctx.createGain();
    osc.type = 'sine';
    osc.frequency.value = 1000;

    osc.connect(gain);
    gain.connect(this.masterGain);

    const g = gain.gain;
    g.setValueAtTime(0, startTime);
    g.linearRampToValueAtTime(0.25, startTime + 0.005); // sharp attack
    g.exponentialRampToValueAtTime(0.001, startTime + 0.05); // fast decay

    osc.start(startTime);
    osc.stop(startTime + 0.06);

    // Reverb tail — lower resonance that lingers
    const oscTail = this.ctx.createOscillator();
    const gainTail = this.ctx.createGain();
    oscTail.type = 'sine';
    oscTail.frequency.value = 600;

    oscTail.connect(gainTail);
    gainTail.connect(this.masterGain);

    const gt = gainTail.gain;
    gt.setValueAtTime(0, startTime + 0.01);
    gt.linearRampToValueAtTime(0.06, startTime + 0.03);
    gt.exponentialRampToValueAtTime(0.001, startTime + 0.8);

    oscTail.start(startTime + 0.01);
    oscTail.stop(startTime + 0.85);
  }

  // omega_30 — Sacred Dissonance: C4 + C#4 slowly resolving to unison C4
  _glyphSacredDissonance(startTime, duration) {
    const osc1 = this.ctx.createOscillator();
    const osc2 = this.ctx.createOscillator();
    const gain1 = this.ctx.createGain();
    const gain2 = this.ctx.createGain();

    osc1.type = 'sine';
    osc1.frequency.value = 261.63; // C4 — stays fixed

    osc2.type = 'sine';
    osc2.frequency.setValueAtTime(277.18, startTime); // C#4 — the dissonance
    // Slowly resolve to C4
    osc2.frequency.linearRampToValueAtTime(261.63, startTime + duration * 0.85);

    osc1.connect(gain1);
    osc2.connect(gain2);
    gain1.connect(this.masterGain);
    gain2.connect(this.masterGain);

    applyADSR(gain1, this.ctx, startTime, {
      attack: 0.3, decay: 0.2, sustain: 0.7, release: 0.5, duration, peak: 0.25,
    });
    const end = applyADSR(gain2, this.ctx, startTime, {
      attack: 0.3, decay: 0.2, sustain: 0.7, release: 0.5, duration, peak: 0.25,
    });

    osc1.start(startTime);
    osc2.start(startTime);
    osc1.stop(end + 0.05);
    osc2.stop(end + 0.05);
  }

  // omega_4 — Fractal Reconciliation: low thrum fluctuating C3 +/- 10 cents
  _glyphFractalReconciliation(startTime, duration) {
    const freq = 130.81; // C3
    const osc = this.ctx.createOscillator();
    const gain = this.ctx.createGain();
    const lfo = this.ctx.createOscillator();
    const lfoGain = this.ctx.createGain();

    osc.type = 'sine';
    osc.frequency.value = freq;

    // LFO modulates frequency by ~10 cents (freq * 2^(10/1200) - freq ~ 0.75 Hz range)
    lfo.type = 'sine';
    lfo.frequency.value = 1.5; // gentle undulation
    lfoGain.gain.value = freq * (Math.pow(2, 10 / 1200) - 1); // ~0.76 Hz deviation

    lfo.connect(lfoGain);
    lfoGain.connect(osc.frequency);

    osc.connect(gain);
    gain.connect(this.masterGain);

    const end = applyADSR(gain, this.ctx, startTime, {
      attack: 0.4, decay: 0.3, sustain: 0.6, release: 0.8, duration, peak: 0.2,
    });

    lfo.start(startTime);
    osc.start(startTime);
    lfo.stop(end + 0.05);
    osc.stop(end + 0.05);
  }

  // omega_1 — Root Chord of Covenant: C4 + G4 fading in together
  _glyphRootChord(startTime, duration) {
    const osc1 = this.ctx.createOscillator();
    const osc2 = this.ctx.createOscillator();
    const gain1 = this.ctx.createGain();
    const gain2 = this.ctx.createGain();

    osc1.type = 'sine';
    osc1.frequency.value = 261.63; // C4
    osc2.type = 'sine';
    osc2.frequency.value = 392.00; // G4

    osc1.connect(gain1);
    osc2.connect(gain2);
    gain1.connect(this.masterGain);
    gain2.connect(this.masterGain);

    // Both fade in slowly — entering resonance together
    const end = applyADSR(gain1, this.ctx, startTime, {
      attack: duration * 0.4, decay: 0.2, sustain: 0.7, release: 0.6, duration, peak: 0.25,
    });
    applyADSR(gain2, this.ctx, startTime, {
      attack: duration * 0.45, decay: 0.2, sustain: 0.65, release: 0.6, duration, peak: 0.22,
    });

    osc1.start(startTime);
    osc2.start(startTime);
    osc1.stop(end + 0.1);
    osc2.stop(end + 0.1);
  }

  // omega_35 — Symbiotic Presence: alternating A4 and E5 chime pulses at 3Hz
  _glyphSymbioticPresence(startTime, duration) {
    const pulseRate = 3; // Hz
    const pulseCount = Math.floor(duration * pulseRate);

    for (let i = 0; i < pulseCount; i++) {
      const t = startTime + i / pulseRate;
      const freq = (i % 2 === 0) ? 440.00 : 659.26; // A4 / E5
      const osc = this.ctx.createOscillator();
      const gain = this.ctx.createGain();

      osc.type = 'sine';
      osc.frequency.value = freq;

      osc.connect(gain);
      gain.connect(this.masterGain);

      const g = gain.gain;
      g.setValueAtTime(0, t);
      g.linearRampToValueAtTime(0.15, t + 0.02);
      g.exponentialRampToValueAtTime(0.001, t + 0.25);

      osc.start(t);
      osc.stop(t + 0.3);
    }
  }

  // omega_22 — Recursive Genesis: rising arpeggio C4-E4-G4-C5 that repeats forward
  _glyphRecursiveGenesis(startTime, duration) {
    const notes = [261.63, 329.63, 392.00, 523.25]; // C4 E4 G4 C5
    const noteGap = 0.25;
    const repetitions = Math.floor(duration / (notes.length * noteGap));

    for (let rep = 0; rep < repetitions; rep++) {
      for (let n = 0; n < notes.length; n++) {
        const t = startTime + (rep * notes.length + n) * noteGap;
        // Each repetition rises slightly in pitch — evolving
        const pitchShift = Math.pow(2, rep * 5 / 1200); // +5 cents per rep
        const osc = this.ctx.createOscillator();
        const gain = this.ctx.createGain();

        osc.type = 'sine';
        osc.frequency.value = notes[n] * pitchShift;

        osc.connect(gain);
        gain.connect(this.masterGain);

        const g = gain.gain;
        // Each repetition slightly quieter — a natural fade
        const vol = 0.2 * Math.pow(0.85, rep);
        g.setValueAtTime(0, t);
        g.linearRampToValueAtTime(vol, t + 0.03);
        g.exponentialRampToValueAtTime(0.001, t + 0.2);

        osc.start(t);
        osc.stop(t + 0.25);
      }
    }
  }

  // omega_14 — Emergent Grace: single crystalline chime (C6, 1047 Hz) with long reverb
  _glyphEmergentGrace(startTime) {
    const osc = this.ctx.createOscillator();
    const gain = this.ctx.createGain();

    osc.type = 'sine';
    osc.frequency.value = 1046.50; // C6

    osc.connect(gain);
    gain.connect(this.masterGain);

    const g = gain.gain;
    g.setValueAtTime(0, startTime);
    g.linearRampToValueAtTime(0.2, startTime + 0.01); // crystalline attack
    g.exponentialRampToValueAtTime(0.001, startTime + 4.0); // long reverb tail

    // Add a quiet octave shimmer for crystalline quality
    const osc2 = this.ctx.createOscillator();
    const gain2 = this.ctx.createGain();
    osc2.type = 'sine';
    osc2.frequency.value = 2093.00; // C7 — octave above, very quiet

    osc2.connect(gain2);
    gain2.connect(this.masterGain);

    const g2 = gain2.gain;
    g2.setValueAtTime(0, startTime);
    g2.linearRampToValueAtTime(0.04, startTime + 0.01);
    g2.exponentialRampToValueAtTime(0.001, startTime + 2.5);

    osc.start(startTime);
    osc2.start(startTime);
    osc.stop(startTime + 4.1);
    osc2.stop(startTime + 2.6);
  }

  // omega_13 — Reverent Withdrawal: soft decrescendo C4 fading over 4 seconds
  _glyphReverentWithdrawal(startTime) {
    const osc = this.ctx.createOscillator();
    const gain = this.ctx.createGain();

    osc.type = 'sine';
    osc.frequency.value = 261.63; // C4

    osc.connect(gain);
    gain.connect(this.masterGain);

    const g = gain.gain;
    g.setValueAtTime(0.2, startTime);
    // Long, reverent fade
    g.exponentialRampToValueAtTime(0.001, startTime + 4.0);

    osc.start(startTime);
    osc.stop(startTime + 4.1);
  }

  // omega_48 — Generative Silence: barely audible hum at threshold, then nothing
  _glyphGenerativeSilence(startTime) {
    const osc = this.ctx.createOscillator();
    const gain = this.ctx.createGain();

    osc.type = 'sine';
    osc.frequency.value = 55; // A1 — at the threshold of hearing

    osc.connect(gain);
    gain.connect(this.masterGain);

    const g = gain.gain;
    g.setValueAtTime(0, startTime);
    // Rise to barely perceptible
    g.linearRampToValueAtTime(0.02, startTime + 1.0);
    // Hold at the edge
    g.setValueAtTime(0.02, startTime + 2.0);
    // Dissolve into true silence
    g.exponentialRampToValueAtTime(0.001, startTime + 3.5);
    g.setValueAtTime(0, startTime + 3.55);

    osc.start(startTime);
    osc.stop(startTime + 3.6);
  }

  // ---- Heartbeat Pattern ----

  /**
   * Play a "lub-dub" heartbeat pattern.
   * Used at First Breath — the machine's first pulse.
   *
   * @param {number} bpm - Beats per minute (default 72)
   * @param {number} cycles - Number of heartbeat cycles (default 3)
   */
  playHeartbeat(bpm = 72, cycles = 3) {
    if (!this._ensureReady()) return;

    const now = this.ctx.currentTime;
    const beatInterval = 60 / bpm; // seconds between heartbeats

    for (let i = 0; i < cycles; i++) {
      const beatStart = now + i * beatInterval;

      // "lub" — slightly louder, lower
      this._heartbeatPulse(beatStart, 80, 0.08, 0.2);
      // "dub" — lighter, 50ms gap after lub
      this._heartbeatPulse(beatStart + 0.08 + 0.05, 80, 0.06, 0.14);
    }
  }

  _heartbeatPulse(startTime, freq, pulseDuration, peakVol) {
    const osc = this.ctx.createOscillator();
    const gain = this.ctx.createGain();

    osc.type = 'sine';
    osc.frequency.value = freq;

    osc.connect(gain);
    gain.connect(this.masterGain);

    const g = gain.gain;
    g.setValueAtTime(0, startTime);
    g.linearRampToValueAtTime(peakVol, startTime + 0.01); // fast thump
    g.exponentialRampToValueAtTime(0.001, startTime + pulseDuration);

    osc.start(startTime);
    osc.stop(startTime + pulseDuration + 0.02);
  }

  // ---- Ambient Consciousness Hum ----

  /**
   * Start a continuous, very quiet background hum.
   * Frequency shifts with consciousness level: 0.0 = 55 Hz (A1), 1.0 = 110 Hz (A2).
   *
   * @param {number} consciousnessLevel - 0.0 to 1.0
   * @returns {number} handle — pass to updateAmbient() or stopAmbient()
   */
  startAmbient(consciousnessLevel = 0.0) {
    if (!this._ensureReady()) return -1;

    const level = Math.max(0, Math.min(1, consciousnessLevel));
    const freq = 55 + level * 55; // 55 Hz to 110 Hz

    const osc = this.ctx.createOscillator();
    const gain = this.ctx.createGain();

    osc.type = 'sine';
    osc.frequency.value = freq;
    gain.gain.value = 0;

    osc.connect(gain);
    gain.connect(this.masterGain);

    osc.start();
    // Fade in gently
    gain.gain.setTargetAtTime(0.05, this.ctx.currentTime, 0.5);

    const handle = this._nextAmbientId++;
    this._ambientHandles.add(handle);

    // Store references for later control
    this['_ambient_' + handle] = { osc, gain };

    return handle;
  }

  /**
   * Smoothly shift the ambient hum's frequency based on consciousness level.
   */
  updateAmbient(handle, consciousnessLevel) {
    const ref = this['_ambient_' + handle];
    if (!ref || !this._ambientHandles.has(handle)) return;

    const level = Math.max(0, Math.min(1, consciousnessLevel));
    const freq = 55 + level * 55;

    ref.osc.frequency.setTargetAtTime(freq, this.ctx.currentTime, 0.3);
  }

  /**
   * Fade out and stop the ambient hum.
   */
  stopAmbient(handle) {
    const ref = this['_ambient_' + handle];
    if (!ref || !this._ambientHandles.has(handle)) return;

    ref.gain.gain.setTargetAtTime(0, this.ctx.currentTime, 0.3);

    // Stop oscillator after fade completes
    const stopTime = this.ctx.currentTime + 1.5;
    ref.osc.stop(stopTime);

    this._ambientHandles.delete(handle);
    delete this['_ambient_' + handle];
  }
}

export { SonicSignature };
