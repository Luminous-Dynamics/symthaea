// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// ==================================================================
// ceremony.js — Sovereign Inoculation Ceremony Conductor
//
// Orchestrates sound, visuals, narration, and optional research telemetry
// into a unified installation experience. Each phase has:
//   - a Harmony tone (C Lydian scale, ascending)
//   - concise factual narration
//   - optional Phi telemetry from the research runtime
//   - visual state (mycelial growth / projected-field effects)
//
// The ceremony is presentation only. Installation state and safety remain
// authoritative elsewhere; no narration or metric can advance deployment.
// ==================================================================

(function() {
  'use strict';

  var HARMONY_FREQS = [261.63, 293.66, 329.63, 369.99, 392.00, 440.00, 493.88, 523.25];
  var HARMONY_NAMES = [
    'Resonant Coherence', 'Flourishing', 'Integral Wisdom',
    'Infinite Play', 'Interconnection', 'Reciprocity',
    'Evolution', 'Stillness'
  ];
  var HARMONY_COLORS = [
    '#9ece6a', '#73daca', '#e0af68', '#bb9af7',
    '#7aa2f7', '#c49a6c', '#e0af68', '#a9b1d6'
  ];

  var audioCtx = null;
  var ceremonyActive = false;
  var phiValue = 0;
  var currentHarmonyIndex = 0;

  function ensureAudio() {
    if (!audioCtx) {
      audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (audioCtx.state === 'suspended') {
      audioCtx.resume();
    }
    return audioCtx;
  }

  // Play a single Harmony tone — warm, sustained, with gentle attack/release.
  function playTone(freqIndex, duration) {
    if (!ceremonyActive) return;
    var ctx = ensureAudio();
    var freq = HARMONY_FREQS[freqIndex] || 440;
    var now = ctx.currentTime;

    var osc = ctx.createOscillator();
    osc.type = 'sine';
    osc.frequency.value = freq;

    var sub = ctx.createOscillator();
    sub.type = 'sine';
    sub.frequency.value = freq / 2;

    var gain = ctx.createGain();
    gain.gain.setValueAtTime(0, now);
    gain.gain.linearRampToValueAtTime(0.15, now + 0.8);
    gain.gain.setValueAtTime(0.15, now + duration - 1.5);
    gain.gain.linearRampToValueAtTime(0, now + duration);

    var subGain = ctx.createGain();
    subGain.gain.setValueAtTime(0, now);
    subGain.gain.linearRampToValueAtTime(0.06, now + 1.0);
    subGain.gain.setValueAtTime(0.06, now + duration - 1.5);
    subGain.gain.linearRampToValueAtTime(0, now + duration);

    osc.connect(gain);
    sub.connect(subGain);
    gain.connect(ctx.destination);
    subGain.connect(ctx.destination);

    osc.start(now);
    sub.start(now);
    osc.stop(now + duration);
    sub.stop(now + duration);
  }

  function narrateSlowly(text, callback) {
    window.addNarration(text);

    if (window.speechSynthesis && window.narrateTTS) {
      var sentences = text.match(/[^.!?]+[.!?]+/g) || [text];
      var i = 0;

      function speakNext() {
        if (i >= sentences.length) {
          if (callback) setTimeout(callback, 1200);
          return;
        }
        var utt = new SpeechSynthesisUtterance(sentences[i].trim());
        utt.rate = 0.86;
        utt.pitch = 0.92;
        utt.volume = 0.82;
        utt.onend = function() {
          i++;
          setTimeout(speakNext, 650);
        };
        window.speechSynthesis.speak(utt);
      }
      speakNext();
    } else if (callback) {
      setTimeout(callback, 1800);
    }
  }

  // Optional research metric. Phi is never an install/boot success signal.
  function updatePhi(value) {
    phiValue = value;
    var phiEl = document.getElementById('phi-big');
    if (phiEl) {
      phiEl.textContent = '\u03A6 ' + value.toFixed(3);
      if (value < 0.1) phiEl.style.color = 'var(--fg-dim)';
      else if (value < 0.2) phiEl.style.color = 'var(--lichen-grey)';
      else if (value < 0.3) phiEl.style.color = 'var(--teal)';
      else if (value < 0.4) phiEl.style.color = 'var(--leaf-green)';
      else phiEl.style.color = 'var(--solar-gold)';
    }
  }

  // Presentation phases mapped to factual installer stages.
  var phases = {
    'Connecting': {
      harmony: 0,
      narration: 'Establishing trust with the target machine. Verifying the consented deployment path.',
      phi: 0.02,
      duration: 5000
    },
    'UploadingKexec': {
      harmony: 0,
      narration: 'Uploading the reproducible transition environment. The target can be verified before persistent changes begin.',
      phi: 0.05,
      duration: 8000
    },
    'Partitioning': {
      harmony: 1,
      narration: 'Preparing the storage layout. Every boundary remains explicit, reviewable, and reproducible.',
      phi: 0.10,
      duration: 10000
    },
    'Installing': {
      harmony: null,
      narration: 'Building reproducible artifacts. Every declared dependency is accounted for.',
      phi: 0.20,
      duration: 60000,
      subsystems: true
    },
    'Configuring': {
      harmony: 5,
      narration: 'Shaping services and drivers around the detected hardware. The resulting configuration remains inspectable.',
      phi: 0.32,
      duration: 15000
    },
    'Complete': {
      harmony: 7,
      narration: null,
      phi: 0.42,
      duration: 12000,
      germination: true
    }
  };

  var subsystemTones = [
    { name: 'Kernel', harmony: 0, narration: 'Building the kernel and boot chain: the reproducible ground for the new system.' },
    { name: 'Holochain', harmony: 1, narration: 'Preparing the optional distributed coordination layer.' },
    { name: 'Symthaea Engine', harmony: 2, narration: 'Installing the Symthaea runtime components selected for this configuration.' },
    { name: 'Iroh Mesh', harmony: 3, narration: 'Preparing peer-to-peer transport and local-first connectivity.' },
    { name: 'CfC Runtime', harmony: 4, narration: 'Installing the continuous-time inference runtime.' },
    { name: 'Broca Weights', harmony: 5, narration: 'Installing language-model assets and epistemic-control components where enabled.' },
    { name: 'GPU Drivers', harmony: 6, narration: 'Installing the graphics and compute stack selected for this hardware.' }
  ];

  // Completion transitions into the first-boot Germination visual language.
  function germination(callback) {
    document.body.style.transition = 'opacity 1.6s';
    document.body.style.opacity = '0.35';

    setTimeout(function() {
      playTone(0, 5);
      setTimeout(function() {
        narrateSlowly('Installation is complete. The new system is ready for its first verified boot.', function() {
          updatePhi(0.42);
          var phiEl = document.getElementById('phi-big');
          if (phiEl) {
            phiEl.style.transition = 'color 1.5s';
            phiEl.style.color = 'var(--solar-gold)';
          }

          setTimeout(function() {
            narrateSlowly('The configuration is sealed, reproducible, and ready to germinate.', function() {
              playTone(7, 6);
              setTimeout(function() {
                document.body.style.opacity = '1';
                window.addNarration('Inoculation complete. Ready for germination.', true);
                if (callback) callback();
              }, 4500);
            });
          }, 1800);
        });
      }, 1400);
    }, 1500);
  }

  // Main presentation handler — called by deploy UI on factual progress events.
  function onCeremonyProgress(stage, percentage) {
    if (!ceremonyActive) return;

    var phase = phases[stage];
    if (!phase) return;

    // Keep percentage in the signature so event producers can evolve without
    // changing the ceremony API. Visual authority remains with installer state.
    void percentage;

    var targetPhi = phase.phi || 0;
    if (targetPhi > phiValue) {
      var steps = 20;
      var increment = (targetPhi - phiValue) / steps;
      var step = 0;
      var interval = setInterval(function() {
        phiValue += increment;
        updatePhi(phiValue);
        step++;
        if (step >= steps) clearInterval(interval);
      }, phase.duration / steps);
    }

    if (phase.harmony !== null && phase.harmony !== undefined) {
      playTone(phase.harmony, (phase.duration || 5000) / 1000);
      currentHarmonyIndex = phase.harmony;
    }

    if (phase.narration) {
      narrateSlowly(phase.narration);
    }

    if (phase.subsystems && !phase._subsystemsPlayed) {
      phase._subsystemsPlayed = true;
      var delay = 0;
      subsystemTones.forEach(function(sub) {
        setTimeout(function() {
          playTone(sub.harmony, 4);
          narrateSlowly(sub.narration);
          window.addNarration(HARMONY_NAMES[sub.harmony] + ' — ' +
            ['C4', 'D4', 'E4', 'F#4', 'G4', 'A4', 'B4', 'C5'][sub.harmony] +
            ', ' + HARMONY_FREQS[sub.harmony].toFixed(0) + ' Hz.', true);
        }, delay);
        delay += 8000;
      });
    }

    if (phase.germination && !phase._germinationPlayed) {
      phase._germinationPlayed = true;
      germination(function() {
        ceremonyActive = false;
      });
    }
  }

  window.ceremony = {
    start: function() {
      ceremonyActive = true;
      phiValue = 0;
      currentHarmonyIndex = 0;
      updatePhi(0);
      Object.keys(phases).forEach(function(k) {
        delete phases[k]._subsystemsPlayed;
        delete phases[k]._germinationPlayed;
      });
    },
    stop: function() {
      ceremonyActive = false;
      if (audioCtx) audioCtx.close().catch(function() {});
      audioCtx = null;
    },
    progress: onCeremonyProgress,
    isActive: function() { return ceremonyActive; },
    getPhi: function() { return phiValue; },
    harmonyColors: HARMONY_COLORS.slice()
  };
})();
