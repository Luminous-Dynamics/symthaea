const form = document.getElementById('form');
const specBox = document.getElementById('spec');
let seedStride = 1;

// ── Musical language on the dials: a live word instead of a raw number,
// so "-0.3" reads as "reflective" and "0.85" reads as "epic". ──
const DIAL_WORDS = {
  valence: { lo: -1, hi: 1, words: ['Dark', 'Melancholic', 'Reflective', 'Hopeful', 'Radiant'] },
  arousal: { lo: 0, hi: 1, words: ['Calm', 'Settled', 'Engaged', 'Restless', 'Electric'] },
  energy: { lo: 0, hi: 1, words: ['Sparse', 'Gentle', 'Flowing', 'Driving', 'Epic'] },
};
function dialWord(name, value) {
  const { lo, hi, words } = DIAL_WORDS[name];
  const frac = (value - lo) / (hi - lo);
  const idx = Math.min(words.length - 1, Math.max(0, Math.floor(frac * words.length)));
  return words[idx];
}
function wireDial(inputName, wordId) {
  const input = form.elements[inputName];
  const word = document.getElementById(wordId);
  const update = () => { word.textContent = dialWord(inputName, parseFloat(input.value)); };
  input.addEventListener('input', update);
  update();
}
wireDial('valence', 'moodWord');
wireDial('arousal', 'arousalWord');
wireDial('energy', 'energyWord');

// ── Creative state presets: one click sets all four rendering dials
// (Wonder/Warmth/Urgency/Focus — API names dopamine/serotonin/noradrenaline/
// consciousness are unchanged underneath) to a named creative mindset. ──
const CREATIVE_PRESETS = {
  dreaming:     { dopamine: 0.65, serotonin: 0.7,  noradrenaline: 0.1, consciousness: 0.3 },
  storytelling: { dopamine: 0.5,  serotonin: 0.55, noradrenaline: 0.4, consciousness: 0.6 },
  flow:         { dopamine: 0.6,  serotonin: 0.45, noradrenaline: 0.6, consciousness: 0.7 },
};
document.querySelectorAll('.preset-btn').forEach((btn) => {
  btn.addEventListener('click', () => {
    const preset = CREATIVE_PRESETS[btn.dataset.preset];
    for (const [name, value] of Object.entries(preset)) {
      form.elements[name].value = value;
    }
  });
});

// ── Progressive disclosure inside Create. These are density profiles,
// not product modes: Guided keeps the first composition easy, Composer
// exposes musical controls, and Advanced exposes renderer/spec internals.
const MODE_HINTS = {
  artist: 'prompt, style, mood, energy, and length',
  studio: 'plus identity grammar, arousal, key, candidate count, and deterministic seed',
  research: 'plus renderer state, voice assignment, and the raw style specification',
};
const modeHint = document.getElementById('modeHint');
function setComposeMode(mode) {
  form.dataset.mode = mode;
  modeHint.textContent = MODE_HINTS[mode];
  document.querySelectorAll('#composeMode .chip').forEach((b) => {
    b.classList.toggle('sel', b.dataset.mode === mode);
  });
}
document.querySelectorAll('#composeMode .chip').forEach((btn) => {
  btn.addEventListener('click', () => setComposeMode(btn.dataset.mode));
});
setComposeMode('artist');

// ── Prompt-first hero: the true entry point. Rotating example prompts
// as placeholder text (paused while focused, so typing is never
// interrupted) invite a description without demanding exact wording;
// "Compose" here or Enter submits the same shared form as every other
// compose path. ──
const heroPrompt = document.getElementById('heroPrompt');
const heroComposeBtn = document.getElementById('heroCompose');
const EXAMPLE_PROMPTS = [
  'a gentle nostalgic waltz for a rainy evening',
  'anxious energy building toward a chase',
  'quiet joy, sunlight through pine needles',
  'a slow goodbye that already knows it’s temporary',
  'defiant and bright, refusing to be small',
  'the last hour before dawn, half-asleep',
  'a homecoming after too long away',
];
let heroPhraseTimer = null;
const heroPrefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
function startHeroRotation() {
  let i = Math.floor(Math.random() * EXAMPLE_PROMPTS.length);
  heroPrompt.placeholder = EXAMPLE_PROMPTS[i];
  if (heroPrefersReducedMotion) return;
  heroPhraseTimer = setInterval(() => {
    i = (i + 1) % EXAMPLE_PROMPTS.length;
    heroPrompt.placeholder = EXAMPLE_PROMPTS[i];
  }, 3500);
}
heroPrompt.addEventListener('focus', () => { clearInterval(heroPhraseTimer); });
heroPrompt.addEventListener('blur', () => { if (!heroPrompt.value) startHeroRotation(); });
document.addEventListener('visibilitychange', () => {
  if (document.hidden) { clearInterval(heroPhraseTimer); }
  else if (!heroPrompt.value && document.activeElement !== heroPrompt) { startHeroRotation(); }
});
startHeroRotation();
heroComposeBtn.addEventListener('click', () => form.requestSubmit());
heroPrompt.addEventListener('keydown', (ev) => {
  if (ev.key === 'Enter') { ev.preventDefault(); form.requestSubmit(); }
});
// ── Per-voice instrument pickers: write directly into the style spec's
// ensemble_pool/counter_instrument fields — no separate mechanism, just
// the same spec textarea every other "Advanced" control already uses.
// A pick collapses ensemble_pool to the single triple the user chose;
// any voice left on Auto keeps whatever the spec's first pool entry
// already had for that slot. ──
const INSTRUMENT_GROUPS = [
  ['Keys & Mallets', [
    ['piano', 'Piano'], ['piano_pp', 'Piano (soft)'], ['electric_piano', 'Electric Piano'],
    ['bell', 'Bell'], ['marimba', 'Marimba'], ['kalimba', 'Kalimba'],
  ]],
  ['Strings', [
    ['violin', 'Violin'], ['cello', 'Cello'], ['upright_bass', 'Upright Bass'],
  ]],
  ['Winds', [
    ['flute', 'Flute'], ['clarinet', 'Clarinet'], ['trumpet', 'Trumpet'],
    ['saxophone', 'Saxophone'], ['ney', 'Ney'],
  ]],
  ['Plucked', [
    ['acoustic_guitar', 'Acoustic Guitar'], ['harp', 'Harp'], ['koto', 'Koto'],
    ['oud', 'Oud'], ['sitar', 'Sitar'],
  ]],
  ['Pads & Synths', [
    ['pad', 'Pad'], ['saw_lead', 'Saw Lead'], ['organ', 'Organ'],
  ]],
];
function populateVoiceSelect(select) {
  select.innerHTML = '<option value="auto" selected>Auto — let the style choose</option>';
  for (const [group, instruments] of INSTRUMENT_GROUPS) {
    const og = document.createElement('optgroup');
    og.label = group;
    for (const [value, label] of instruments) {
      const opt = document.createElement('option');
      opt.value = value; opt.textContent = label;
      og.appendChild(opt);
    }
    select.appendChild(og);
  }
}
const voiceSelects = {
  melody: document.getElementById('voiceMelody'),
  harmony: document.getElementById('voiceHarmony'),
  bass: document.getElementById('voiceBass'),
  counter: document.getElementById('voiceCounter'),
};
Object.values(voiceSelects).forEach(populateVoiceSelect);
async function ensureSpecLoadedForVoicePick() {
  if (specBox.value.trim()) return;
  const style = new FormData(form).get('style');
  const res = await fetch('/api/spec/' + style);
  specBox.value = await res.text();
}
function applyVoicePicks() {
  let spec;
  try { spec = JSON.parse(specBox.value); }
  catch { return; } // leave whatever's there — the submit handler will report the JSON error
  const fallback = (spec.ensemble_pool && spec.ensemble_pool[0]) || ['piano', 'piano', 'upright_bass'];
  const pick = (name, idx) => (voiceSelects[name].value === 'auto' ? fallback[idx] : voiceSelects[name].value);
  spec.ensemble_pool = [[pick('melody', 0), pick('harmony', 1), pick('bass', 2)]];
  if (voiceSelects.counter.value === 'auto') delete spec.counter_instrument;
  else spec.counter_instrument = voiceSelects.counter.value;
  specBox.value = JSON.stringify(spec, null, 2);
}
Object.values(voiceSelects).forEach((sel) => {
  sel.addEventListener('change', async () => {
    await ensureSpecLoadedForVoicePick();
    applyVoicePicks();
  });
});
// ── Randomize / keep-melody: pick concrete instruments (never "auto")
// so the roll is visible in the dropdowns themselves, then reuse the
// exact same load+patch path a manual pick uses. ──
const ALL_INSTRUMENTS = INSTRUMENT_GROUPS.flatMap(([, list]) => list.map(([value]) => value));
const randomInstrument = () => ALL_INSTRUMENTS[Math.floor(Math.random() * ALL_INSTRUMENTS.length)];
document.getElementById('randomizeEnsemble').addEventListener('click', async () => {
  voiceSelects.melody.value = randomInstrument();
  voiceSelects.harmony.value = randomInstrument();
  voiceSelects.bass.value = randomInstrument();
  voiceSelects.counter.value = randomInstrument();
  Object.keys(voiceSelects).forEach(refreshPreviewButton);
  await ensureSpecLoadedForVoicePick();
  applyVoicePicks();
});
document.getElementById('keepMelodyChangeInstruments').addEventListener('click', async () => {
  await ensureSpecLoadedForVoicePick();
  if (voiceSelects.melody.value === 'auto') {
    // "Keep melody" needs something concrete to keep — pin it to
    // whatever the spec's current pool already plays, not a fresh roll.
    let current = 'piano';
    try {
      const spec = JSON.parse(specBox.value);
      current = (spec.ensemble_pool && spec.ensemble_pool[0] && spec.ensemble_pool[0][0]) || current;
    } catch { /* keep the piano fallback; applyVoicePicks will report any real JSON error */ }
    voiceSelects.melody.value = current;
  }
  voiceSelects.harmony.value = randomInstrument();
  voiceSelects.bass.value = randomInstrument();
  voiceSelects.counter.value = randomInstrument();
  Object.keys(voiceSelects).forEach(refreshPreviewButton);
  applyVoicePicks();
});
// ── Instrument preview: a ~3.5s scale rendered once per instrument and
// cached server-side (/api/preview/{name}) — hear a pick before
// committing to it. Disabled on "Auto" (nothing concrete to preview). ──
const previewAudio = new Audio();
function refreshPreviewButton(voice) {
  const btn = document.querySelector(`.preview-btn[data-voice="${voice}"]`);
  btn.disabled = voiceSelects[voice].value === 'auto';
}
document.querySelectorAll('.preview-btn').forEach((btn) => {
  const voice = btn.dataset.voice;
  refreshPreviewButton(voice);
  voiceSelects[voice].addEventListener('change', () => refreshPreviewButton(voice));
  btn.addEventListener('click', () => {
    const value = voiceSelects[voice].value;
    if (value === 'auto') return;
    previewAudio.src = '/api/preview/' + value;
    previewAudio.play();
  });
});
document.getElementById('loadSpec').addEventListener('click', async () => {
  const style = new FormData(form).get('style');
  const res = await fetch('/api/spec/' + style);
  specBox.value = await res.text();
});
const savedSel = document.getElementById('savedSpecs');
async function refreshSaved() {
  const names = await (await fetch('/api/specs')).json();
  savedSel.innerHTML = '<option value="">saved specs…</option>' +
    names.map(n => `<option>${n}</option>`).join('');
}
refreshSaved();
savedSel.addEventListener('change', async () => {
  if (!savedSel.value) return;
  const res = await fetch('/api/specs/' + savedSel.value);
  specBox.value = await res.text();
  document.getElementById('specName').value = savedSel.value;
});
document.getElementById('saveSpec').addEventListener('click', async () => {
  const name = document.getElementById('specName').value.trim();
  if (!name || !specBox.value.trim()) { note.textContent = 'name + spec JSON required to save'; return; }
  let spec;
  try { spec = JSON.parse(specBox.value); }
  catch (e) { note.textContent = 'spec is not valid JSON: ' + e.message; return; }
  const res = await fetch('/api/specs', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({name, spec})
  });
  note.textContent = res.ok ? `saved “${name}”` : await res.text();
  refreshSaved();
});
const go = document.getElementById('go');
const note = document.getElementById('note');
const list = document.getElementById('candidates');
const tweakDrawer = document.getElementById('tweakDrawer');

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const f = new FormData(form);
  const body = {
    style: f.get('style'),
    valence: parseFloat(f.get('valence')),
    arousal: parseFloat(f.get('arousal')),
    energy: parseFloat(f.get('energy')),
    tonic: parseInt(f.get('tonic')),
    bars: parseInt(f.get('bars')),
    n_candidates: parseInt(f.get('n_candidates')),
    base_seed: parseInt(f.get('base_seed')),
    seed_stride: seedStride,
    dopamine: parseFloat(f.get('dopamine')),
    serotonin: parseFloat(f.get('serotonin')),
    noradrenaline: parseFloat(f.get('noradrenaline')),
    consciousness: parseFloat(f.get('consciousness')),
    prompt: heroPrompt.value.trim(),
    grammar: f.get('grammar') || 'Auto'
  };
  seedStride = 1; // one-shot: More-like-this sets it, plain Compose resets
  if (specBox.value.trim()) {
    try { body.spec = JSON.parse(specBox.value); }
    catch (e) { note.textContent = 'spec is not valid JSON: ' + e.message; return; }
  }
  go.disabled = true;
  heroComposeBtn.disabled = true;
  note.textContent = 'composing… (each candidate is a full piece; give it a moment)';
  loadingOn();
  list.innerHTML = '';
  try {
    const res = await fetch('/api/compose', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body)
    });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    latestBatch = data.candidates;
    listenIndex = 0;
    note.textContent = data.ranking_note +
      (data.sampled_instruments ? ' · sampled instruments' : ' · synthesis (set SYMTHAEA_VCSL_DIR for samples)');
    for (const c of data.candidates) {
      const card = document.createElement('div');
      card.className = 'card';
      card.innerHTML = `
        <div class="who">
          <div class="seed">${c.title || (c.card ? c.card.title : 'seed ' + c.seed)}</div>
          <div class="sim">${c.similarity != null ? 'match ' + c.similarity.toFixed(3) : ''}</div>
          <div class="badge">${c.card ? c.card.traits.join(' · ') : ''}</div>
          <div class="badge">${(() => {
            if (!c.novelty) return '';
            const ch = {melodic: c.novelty.melodic, rhythmic: c.novelty.rhythmic,
                        premise: c.novelty.premise, orchestration: c.novelty.orchestration,
                        harmonic: c.novelty.harmonic};
            const top = Object.entries(ch).sort((a,b) => b[1]-a[1]).filter(e => e[1] > 0.05).slice(0,2);
            return top.length ? 'differs most in: ' + top.map(e => e[0]).join(', ') : '';
          })()}</div>
          <div class="badge">seed ${c.seed} · ${c.duration_secs.toFixed(0)}s${c.meter !== 4 ? ' · ' + c.meter + ' beats/bar' : ''}</div>
        </div>
        <audio controls preload="none" src="/api/audio/${c.id}"></audio>
        <a href="/api/midi/${c.id}" download>MIDI</a>
        <a href="/api/audio/${c.id}" download="muse_seed${c.seed}.wav">WAV</a>
        <a href="#" data-seed="${c.seed}" class="more">More like this</a>
        <a href="#" data-seed="${c.seed}" class="cousin" title="explore a far region of idea space with the same intent">Find distant cousin</a>
        <a href="#" data-id="${c.id}" class="research" title="inspect this exact candidate without stopping playback">Research</a>
        <a href="#" data-id="${c.id}" class="keep" title="mark as a keeper — feeds Muse's taste">♡ keep</a>
        <span class="badge" title="Integration is score-structure analysis, not a consciousness claim.">${c.renderer === 'fluidsynth' ? 'soundfont render' : 'native render'} · Integration ${c.phi.toFixed(2)} · ${c.grammar}${c.ending ? ' (' + c.ending + ')' : ''}</span>
        <div class="legend" style="margin-top:0">${(c.why || []).join(' ')}</div>
        <canvas class="roll" title="click to seek"></canvas>
        <div class="legend"></div>`;
      card.querySelector('.keep').addEventListener('click', async (ev) => {
        ev.preventDefault();
        const el = ev.target;
        const res = await fetch('/api/keeper/' + el.dataset.id, { method: 'POST' });
        if (res.ok) { el.textContent = '♥ kept'; el.style.color = 'var(--green)'; }
      });
      card.querySelector('.research').addEventListener('click', async (ev) => {
        ev.preventDefault();
        c._listenIntent ||= createIntentSnapshot();
        await showPiece(c, false);
        showTab('research');
      });
      card.querySelector('.more').addEventListener('click', (ev) => {
        ev.preventDefault();
        form.querySelector('[name=base_seed]').value = ev.target.dataset.seed;
        seedStride = 6; // holds form/accompaniment/motif, varies the rest
        form.requestSubmit();
      });
      card.querySelector('.cousin').addEventListener('click', (ev) => {
        ev.preventDefault();
        // A far region of the same intent: well past the Explorer's
        // 64-seed window, deterministic from the card's own seed.
        form.querySelector('[name=base_seed]').value =
          String((parseInt(ev.target.dataset.seed) + 100003) >>> 0);
        seedStride = 1; // full exploration in the new region
        form.requestSubmit();
      });
      list.appendChild(card);
      pianoRoll(card.querySelector('.roll'), card.querySelector('.legend'),
                card.querySelector('audio'), c.id);
    }
    // Candidates dominate the page now — tuck the controls away; one
    // click on "Compose" (the summary) brings them back.
    tweakDrawer.removeAttribute('open');
  } catch (err) {
    note.textContent = 'error: ' + err.message;
  } finally {
    loadingOff();
    go.disabled = false;
    heroComposeBtn.disabled = false;
  }
});

// ── Piano-roll: the performed notes, per voice, with a playhead ─────────
const VOICE_COLORS = {
  Melody:   '#d9a05b',
  Doubling: 'rgba(217,160,91,0.45)',
  Harmony:  '#7a6d5c',
  Bass:     '#7ba05b',
  Counter:  '#6f9bbf',
};
async function pianoRoll(canvas, legend, audio, id) {
  let voices;
  try { voices = await (await fetch('/api/notes/' + id)).json(); }
  catch { canvas.remove(); legend.remove(); return; }
  const midiOf = f => 69 + 12 * Math.log2(f / 440);
  let tMax = 0, mLo = 127, mHi = 0;
  for (const v of voices) for (const n of v.notes) {
    tMax = Math.max(tMax, n.start_time + n.duration);
    const m = midiOf(n.frequency);
    mLo = Math.min(mLo, m); mHi = Math.max(mHi, m);
  }
  if (!tMax) { canvas.remove(); legend.remove(); return; }
  mLo -= 2; mHi += 2;
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.clientWidth * dpr, H = canvas.clientHeight * dpr;
  canvas.width = W; canvas.height = H;
  const ctx = canvas.getContext('2d');
  const x = t => t / tMax * W;
  const y = m => H - (m - mLo) / (mHi - mLo) * H;
  const noteH = Math.max(2 * dpr, H / (mHi - mLo) * 0.85);
  // Draw once to an offscreen copy; the playhead redraws over it.
  function drawNotes() {
    ctx.clearRect(0, 0, W, H);
    for (const v of voices) {
      ctx.fillStyle = VOICE_COLORS[v.name] || '#999';
      for (const n of v.notes) {
        ctx.globalAlpha = 0.35 + 0.6 * Math.min(1, n.velocity);
        ctx.fillRect(x(n.start_time), y(midiOf(n.frequency)) - noteH / 2,
                     Math.max(1.5 * dpr, x(n.duration)), noteH);
      }
    }
    ctx.globalAlpha = 1;
  }
  drawNotes();
  const base = ctx.getImageData(0, 0, W, H);
  legend.innerHTML = voices.map(v =>
    `<b style="color:${VOICE_COLORS[v.name] || '#999'}">■</b> ${v.name} · ${v.instrument}`
  ).join('  ');
  let raf;
  function playhead() {
    ctx.putImageData(base, 0, 0);
    if (audio.duration) {
      const px = audio.currentTime / audio.duration * W;
      ctx.fillStyle = 'rgba(232,223,210,0.85)';
      ctx.fillRect(px, 0, 1.5 * dpr, H);
    }
    if (!audio.paused && !audio.ended) raf = requestAnimationFrame(playhead);
  }
  audio.addEventListener('play', () => { cancelAnimationFrame(raf); playhead(); });
  audio.addEventListener('pause', () => cancelAnimationFrame(raf));
  audio.addEventListener('seeked', playhead);
  canvas.addEventListener('click', ev => {
    if (!audio.duration) { audio.play(); return; }
    const r = canvas.getBoundingClientRect();
    audio.currentTime = (ev.clientX - r.left) / r.width * audio.duration;
    playhead();
  });
}

// ── Discovery-first landing ─────────────────────────────────────────────
// "Today's Discoveries": a date-deterministic batch composed on load —
// the same four identities all day, different tomorrow. The style rotates
// daily through the roster; everything is reproducible from the visible
// seed. Discover first; the Tweak drawer is there when you've picked.
const DISCOVERY_STYLES = ['Classical', 'Nocturne', 'Tango', 'Cinematic',
                          'Passacaglia', 'Folk', 'Waltz', 'Fugue', 'ModalFolk',
                          'Celtic', 'Blues', 'Impressionism', 'SacredChoral',
                          'Minimalism', 'JazzBallad', 'BaroqueSuite', 'ProgFolk', 'Ambient', 'Sonata',
                          'RenaissancePolyphony', 'AfroCuban', 'Flamenco', 'BossaNova', 'Opera',
                          'IrishTraditional', 'HindustaniInspired'];
function todaySeed() {
  const d = new Date();
  return d.getFullYear() * 10000 + (d.getMonth() + 1) * 100 + d.getDate();
}
function composeToday() {
  const seed = todaySeed();
  const style = DISCOVERY_STYLES[seed % DISCOVERY_STYLES.length];
  document.getElementById('discoveryTitle').textContent =
    `Today's Discoveries — ${style}`;
  form.querySelector('[name=style]').value = style;
  form.querySelector('[name=base_seed]').value = String(seed);
  form.querySelector('[name=n_candidates]').value = '4';
  form.requestSubmit();
}
document.getElementById('surprise').addEventListener('click', () => {
  // A one-click leap to an unexplored region: random entry point, but the
  // seed lands in the form so anything you like remains reproducible.
  const seed = Math.floor(Math.random() * 900000) + 1;
  const style = DISCOVERY_STYLES[Math.floor(Math.random() * DISCOVERY_STYLES.length)];
  document.getElementById('discoveryTitle').textContent = `Discoveries — ${style}`;
  form.querySelector('[name=style]').value = style;
  form.querySelector('[name=base_seed]').value = String(seed);
  form.querySelector('[name=n_candidates]').value = '4';
  form.requestSubmit();
});
queueMicrotask(composeToday);

// ── The breathing loader (calm, not busy) ───────────────────────────────
const LOADING_PHRASES = [
  'listening for a hook worth keeping…',
  'auditioning subjects for the ground…',
  'letting the themes argue…',
  'teaching the bass where home is…',
  'choosing what deserves to return…',
  'breathing into the phrases…',
  'finding four different answers…',
];
let phraseTimer = null;
function loadingOn() {
  applyOrbPalette(form.querySelector('[name=style]').value);
  const el = document.getElementById('loading');
  const ph = document.getElementById('loadingPhrase');
  let i = Math.floor(Math.random() * LOADING_PHRASES.length);
  ph.textContent = LOADING_PHRASES[i];
  el.classList.add('on');
  phraseTimer = setInterval(() => {
    i = (i + 1) % LOADING_PHRASES.length;
    ph.textContent = LOADING_PHRASES[i];
  }, 3500);
}
function loadingOff() {
  document.getElementById('loading').classList.remove('on');
  if (phraseTimer) { clearInterval(phraseTimer); phraseTimer = null; }
}

// ── Tabs ────────────────────────────────────────────────────────────────
let latestBatch = [];
let listenIndex = 0;
const tabStudio = document.getElementById('tabStudio');
const tabListen = document.getElementById('tabListen');
const tabResearch = document.getElementById('tabResearch');
const tabLiked = document.getElementById('tabLiked');
function showTab(which) {
  const listen = which === 'listen';
  const research = which === 'research';
  const liked = which === 'liked';
  const create = which === 'studio';
  document.getElementById('listenView').classList.toggle('on', listen);
  document.getElementById('researchView').classList.toggle('on', research);
  document.getElementById('likedView').classList.toggle('on', liked);
  document.getElementById('studioView').style.display = create ? '' : 'none';
  tabStudio.classList.toggle('active', create);
  tabListen.classList.toggle('active', listen);
  tabResearch.classList.toggle('active', research);
  tabLiked.classList.toggle('active', liked);
  for (const [tab, active] of [[tabStudio, create], [tabListen, listen], [tabResearch, research]]) {
    if (active) tab.setAttribute('aria-current', 'page'); else tab.removeAttribute('aria-current');
  }
  if (!listen) cancelAnimationFrame(vizRaf);
  if (listen) startListen();
  if (research && typeof renderResearch === 'function') renderResearch();
  if (liked) loadLiked();
}
tabStudio.addEventListener('click', () => showTab('studio'));
tabListen.addEventListener('click', () => showTab('listen'));
tabResearch.addEventListener('click', () => showTab('research'));
tabLiked.addEventListener('click', () => showTab('liked'));
document.getElementById('openInCreate').addEventListener('click', () => showTab('studio'));

// ── Liked Songs ──────────────────────────────────────────────────────────
async function loadLiked() {
  const grid = document.getElementById('likedGrid');
  const empty = document.getElementById('likedEmpty');
  grid.innerHTML = '<span class="badge">loading…</span>';
  let entries;
  try {
    const res = await fetch('/api/keepers');
    entries = await res.json();
  } catch {
    grid.innerHTML = '';
    empty.style.display = '';
    empty.textContent = "couldn't reach the keeper log — try again";
    return;
  }
  grid.innerHTML = '';
  empty.style.display = entries.length ? 'none' : '';
  for (const e of entries) {
    const when = e.ts ? new Date(e.ts * 1000).toLocaleString() : '';
    const card = document.createElement('div');
    card.className = 'card';
    card.innerHTML = `
      <div class="who">
        <div class="seed">${e.title || e.spec}</div>
        <div class="badge">${e.spec}${e.mode ? ' · ' + e.mode : ''} · seed ${e.seed}</div>
        <div class="badge">${e.grammar || ''}${e.ending ? ' (' + e.ending + ')' : ''} · Integration ${(e.phi ?? 0).toFixed(2)}</div>
        <div class="badge">${when}</div>
      </div>
      <audio controls preload="none" src="/api/keeper-audio/${e.audio_key}"></audio>
      <a href="/api/keeper-midi/${e.audio_key}" download>MIDI</a>
      <a href="/api/keeper-audio/${e.audio_key}" download="muse_${e.audio_key}.wav">WAV</a>
      <a href="/api/keeper-recipe/${e.audio_key}" download>Recipe</a>`;
    grid.appendChild(card);
  }
}

// ── Listen: the visualizer ──────────────────────────────────────────────
// A radial breathing form driven by the real spectrum (Web Audio
// AnalyserNode) — inner glow follows the bass, a ring of petals follows
// the mids, in the room's own amber. Beauty over meters: no axes, no
// numbers.
const listenAudio = document.getElementById('listenAudio');
let audioCtx = null, analyser = null, vizRaf = null;
let currentListenAnalysis = null;
let currentCompositionBundle = null;
let currentPerformanceBundle = null;
let currentProvenanceBundle = null;
function ensureAnalyser() {
  if (audioCtx) return;
  audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  const src = audioCtx.createMediaElementSource(listenAudio);
  analyser = audioCtx.createAnalyser();
  analyser.fftSize = 512;
  analyser.smoothingTimeConstant = 0.86;
  src.connect(analyser);
  analyser.connect(audioCtx.destination);
}

const LISTEN_STYLES = ['Classical', 'Nocturne', 'Cinematic', 'Passacaglia',
                       'Lullaby', 'Waltz', 'Folk', 'Tango', 'Fugue',
                       'ModalFolk', 'March', 'Playful', 'Celtic', 'Blues',
                       'Impressionism', 'SacredChoral', 'Minimalism',
                       'JazzBallad', 'BaroqueSuite', 'ProgFolk', 'Ambient', 'Sonata',
                       'RenaissancePolyphony', 'AfroCuban', 'Flamenco', 'BossaNova', 'Opera',
                       'IrishTraditional', 'HindustaniInspired'];
const LISTEN_DEFAULT = new Set(['Classical', 'Nocturne', 'Cinematic',
                                'Passacaglia', 'Lullaby']);
const listenSel = new Set(LISTEN_DEFAULT);
const chipsBox = document.getElementById('styleChips');
for (const st of LISTEN_STYLES) {
  const b = document.createElement('button');
  b.className = 'chip' + (listenSel.has(st) ? ' sel' : '');
  b.textContent = st;
  b.type = 'button';
  b.addEventListener('click', () => {
    if (listenSel.has(st)) { listenSel.delete(st); b.classList.remove('sel'); }
    else { listenSel.add(st); b.classList.add('sel'); }
    listenQueue = [];
    updateNextJourneyPreview();
    prefetchListen();
  });
  chipsBox.appendChild(b);
}

let listenQueue = [];
let listenComposing = false;
let listenGeneration = 0;
let listenHistory = [];
let currentListenPiece = null;
const listenState = document.getElementById('listenState');
const clamp = (value, lo, hi) => Math.min(hi, Math.max(lo, value));
function formatTime(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return '0:00';
  const whole = Math.floor(seconds);
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, '0')}`;
}
function halton(index, base) {
  let result = 0, fraction = 1 / base, i = index;
  while (i > 0) {
    result += fraction * (i % base);
    i = Math.floor(i / base);
    fraction /= base;
  }
  return result;
}
function createIntentSnapshot() {
  const read = (name, fallback) => {
    const field = form.querySelector(`[name="${name}"]`);
    const value = field ? Number(field.value) : fallback;
    return Number.isFinite(value) ? value : fallback;
  };
  return {
    valence: clamp(read('valence', 0.15), -1, 1),
    arousal: clamp(read('arousal', 0.45), 0, 1),
    energy: clamp(read('energy', 0.5), 0, 1),
  };
}
function journeyIntent() {
  const base = createIntentSnapshot();
  const n = ++listenGeneration;
  const kind = document.getElementById('journeyKind').value;
  if (kind === 'discovery') {
    return {
      valence: -1 + 2 * halton(n, 2),
      arousal: halton(n, 3),
      energy: halton(n, 5),
    };
  }
  if (kind === 'contrast') {
    const flip = n % 2 === 0;
    return {
      valence: clamp((flip ? -base.valence : base.valence) + (halton(n, 5) - .5) * .18, -1, 1),
      arousal: clamp((flip ? 1 - base.arousal : base.arousal) + (halton(n, 3) - .5) * .14, 0, 1),
      energy: clamp((flip ? 1 - base.energy : base.energy) + (halton(n, 7) - .5) * .14, 0, 1),
    };
  }
  return {
    valence: clamp(base.valence + (halton(n, 2) - .5) * .34, -1, 1),
    arousal: clamp(base.arousal + (halton(n, 3) - .5) * .26, 0, 1),
    energy: clamp(base.energy + (halton(n, 5) - .5) * .26, 0, 1),
  };
}
function relationToCurrent(next) {
  if (!currentListenPiece) return 'First piece in this journey';
  const before = currentListenPiece._listenIntent || createIntentSnapshot();
  const after = next._listenIntent || before;
  const shifts = [
    ['mood', Math.abs(after.valence - before.valence)],
    ['motion', Math.abs(after.arousal - before.arousal)],
    ['energy', Math.abs(after.energy - before.energy)],
  ].sort((a, b) => b[1] - a[1]);
  if (next.style === currentListenPiece.style && shifts[0][1] < .18) return 'Nearby identity · same style';
  if (shifts[0][1] > .42) return `Contrast · largest change in ${shifts[0][0]}`;
  if (next.style !== currentListenPiece.style) return `Distant style · related ${shifts[0][0]}`;
  return `Related variation · shifted ${shifts[0][0]}`;
}
async function composeListenPiece() {
  const styles = listenSel.size ? [...listenSel] : LISTEN_STYLES;
  const style = styles[Math.floor(Math.random() * styles.length)];
  const seed = Math.floor(Math.random() * 900000) + 1;
  const intent = journeyIntent();
  const body = {
    style,
    valence: intent.valence,
    arousal: intent.arousal,
    energy: intent.energy,
    tonic: Math.floor(halton(listenGeneration, 7) * 12),
    bars: 4,
    n_candidates: 1,
    base_seed: seed,
    prompt: '',
  };
  const res = await fetch('/api/compose', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body),
  });
  if (!res.ok) return null;
  const candidate = (await res.json()).candidates[0];
  candidate._listenIntent = intent;
  candidate._journeyRelation = relationToCurrent(candidate);
  return candidate;
}
function updateNextJourneyPreview() {
  const next = listenQueue[0];
  document.getElementById('nextJourneyTitle').textContent = next ? next.title : (listenComposing ? 'Composing…' : 'Not queued');
  document.getElementById('nextJourneyRelation').textContent = next ? next._journeyRelation : 'A related piece will appear here';
}
async function prefetchListen() {
  if (listenComposing || listenQueue.length >= 1) return;
  listenComposing = true;
  listenState.textContent = 'Composing the next piece…';
  updateNextJourneyPreview();
  try {
    const c = await composeListenPiece();
    if (c) listenQueue.push(c);
  } finally {
    listenComposing = false;
    if (!listenAudio.paused || listenQueue.length) listenState.textContent = '';
    updateNextJourneyPreview();
  }
}
function renderWhyEvidence(c) {
  const box = document.getElementById('whyEvidence');
  box.innerHTML = '';
  for (const line of c.why || []) {
    const p = document.createElement('p');
    p.textContent = line;
    box.appendChild(p);
  }
  if (!box.childElementCount) box.innerHTML = '<p>No grounded explanation was emitted for this piece.</p>';
}
function renderVoiceBlend(analysis) {
  const box = document.getElementById('voiceBlend');
  if (!analysis || !analysis.voices.length) {
    box.innerHTML = '<p class="muted">Performed voice data is unavailable.</p>';
    return;
  }
  box.innerHTML = analysis.voices.map(voice => `
    <div class="voice-row">
      <span title="${voice.instrument}">${voice.name}</span>
      <i style="--voice-share:${(voice.share * 100).toFixed(1)}%"></i>
      <b>${Math.round(voice.share * 100)}%</b>
    </div>`).join('');
}
function drawEmotionArc(analysis, composition = currentCompositionBundle) {
  const canvas = document.getElementById('emotionArc');
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(1, canvas.clientWidth), h = Math.max(1, canvas.clientHeight);
  canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);
  const structural = composition?.resonance?.samples || [];
  const values = structural.length
    ? structural.map(sample => clamp(Number(sample.energy) || 0, 0, 1))
    : (analysis?.density || []);
  if (!values.length) {
    ctx.strokeStyle = 'rgba(158,154,148,.25)';
    ctx.setLineDash([3, 4]);
    ctx.beginPath(); ctx.moveTo(0, h * .68); ctx.lineTo(w, h * .68); ctx.stroke();
    return;
  }
  const baseline = h - 12;
  const grad = ctx.createLinearGradient(0, 0, w, 0);
  grad.addColorStop(0, 'rgba(125,100,200,.78)');
  grad.addColorStop(.62, `rgba(${currentPalette.a},.9)`);
  grad.addColorStop(1, `rgba(${currentPalette.b},.62)`);
  ctx.beginPath();
  values.forEach((value, i) => {
    const x = i / Math.max(1, values.length - 1) * w;
    const y = baseline - value * (h - 24);
    if (!i) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.strokeStyle = grad; ctx.lineWidth = 1.5; ctx.stroke();
  ctx.lineTo(w, baseline); ctx.lineTo(0, baseline); ctx.closePath();
  const fill = ctx.createLinearGradient(0, 0, 0, h);
  fill.addColorStop(0, 'rgba(215,164,95,.22)'); fill.addColorStop(1, 'rgba(215,164,95,0)');
  ctx.fillStyle = fill; ctx.fill();
}
async function loadLegacyPerformedAnalysis(id) {
  try {
    const res = await fetch('/api/notes/' + id);
    if (!res.ok) throw new Error('performed notes unavailable');
    const rawVoices = await res.json();
    let duration = 0, totalWeight = 0;
    const bins = Array.from({length: 64}, () => 0);
    const events = [];
    const weights = rawVoices.map(voice => {
      let weight = 0;
      for (const note of voice.notes || []) {
        duration = Math.max(duration, note.start_time + note.duration);
        weight += Math.max(.02, note.duration) * Math.max(.05, note.velocity || .5);
      }
      totalWeight += weight;
      return weight;
    });
    for (let voiceIndex = 0; voiceIndex < rawVoices.length; voiceIndex++) {
      const voice = rawVoices[voiceIndex];
      for (const note of voice.notes || []) {
        const index = Math.min(bins.length - 1, Math.floor((note.start_time / Math.max(duration, .001)) * bins.length));
        bins[index] += Math.max(.05, note.velocity || .5) * Math.sqrt(Math.max(.02, note.duration));
        if (events.length < 900) events.push({
          start: note.start_time,
          duration: note.duration,
          velocity: note.velocity || .5,
          voiceIndex,
        });
      }
    }
    const maxBin = Math.max(...bins, .001);
    return {
      duration,
      density: bins.map(value => value / maxBin),
      events,
      voices: rawVoices.map((voice, i) => ({
        name: voice.name,
        instrument: voice.instrument,
        share: totalWeight ? weights[i] / totalWeight : 0,
      })),
    };
  } catch {
    return null;
  }
}

function analysisFromPerformanceBundle(composition, performance) {
  if (!performance) return null;
  const voiceIndex = new Map((performance.voices || []).map((voice, index) => [voice.id, index]));
  const duration = Math.max(
    Number(performance.duration_seconds) || 0,
    Number(composition?.duration_seconds) || 0,
    .001,
  );
  const events = (performance.notes || []).map(note => ({
    id: note.id,
    sourceNoteId: note.source_note_id,
    start: Number(note.start_seconds) || 0,
    duration: Number(note.duration_seconds) || 0,
    velocity: Number(note.velocity) || .5,
    frequency: Number(note.frequency_hz) || 0,
    onsetDeviation: note.onset_deviation_seconds,
    durationDeviation: note.duration_deviation_seconds,
    voiceIndex: voiceIndex.get(note.voice_id) ?? 0,
  }));
  const bins = Array.from({length: 64}, () => 0);
  const weights = Array.from({length: performance.voices?.length || 0}, () => 0);
  for (const event of events) {
    const index = Math.min(bins.length - 1, Math.floor(event.start / duration * bins.length));
    const weight = Math.max(.05, event.velocity) * Math.sqrt(Math.max(.02, event.duration));
    bins[index] += weight;
    weights[event.voiceIndex] = (weights[event.voiceIndex] || 0) + Math.max(.02, event.duration) * Math.max(.05, event.velocity);
  }
  const maxBin = Math.max(...bins, .001);
  const totalWeight = weights.reduce((sum, value) => sum + value, 0);
  return {
    duration,
    density: bins.map(value => value / maxBin),
    events,
    voices: (performance.voices || []).map((voice, index) => ({
      id: voice.id,
      name: voice.name,
      instrument: voice.instrument,
      share: totalWeight ? weights[index] / totalWeight : 0,
    })),
    composition,
    performance,
  };
}

async function fetchBundle(url) {
  try {
    const response = await fetch(url);
    if (!response.ok) return null;
    return response.json();
  } catch {
    return null;
  }
}

async function loadPieceBundles(id) {
  const [compositionEnvelope, performanceEnvelope, provenanceEnvelope] = await Promise.all([
    fetchBundle(`/api/piece/${id}/listen-bundle`),
    fetchBundle(`/api/piece/${id}/performance-bundle`),
    fetchBundle(`/api/piece/${id}/provenance`),
  ]);
  const composition = compositionEnvelope?.payload || null;
  const performance = performanceEnvelope?.payload || null;
  let analysis = analysisFromPerformanceBundle(composition, performance);
  if (!analysis) analysis = await loadLegacyPerformedAnalysis(id);
  return {
    composition,
    performance,
    provenance: provenanceEnvelope?.payload || null,
    analysis,
    warnings: [
      ...(compositionEnvelope?.warnings || []),
      ...(performanceEnvelope?.warnings || []),
      ...(provenanceEnvelope?.warnings || []),
    ],
  };
}

function activeSectionAt(seconds) {
  const sections = currentCompositionBundle?.sections || [];
  return sections.find(section => seconds >= section.start.seconds && seconds < section.end.seconds)
    || sections.find(section => seconds < section.start.seconds)
    || sections.at(-1)
    || null;
}

function activeMotifAt(seconds) {
  const occurrences = currentCompositionBundle?.motif_occurrences || [];
  return occurrences.find(item => seconds >= item.start.seconds && seconds < item.end.seconds) || null;
}
function activeSonorityAt(seconds) {
  const regions = currentCompositionBundle?.sonorities || [];
  return regions.find(item => seconds >= item.start.seconds && seconds < item.end.seconds)
    || regions.find(item => seconds < item.start.seconds)
    || regions.at(-1)
    || null;
}
function activeCadenceAt(seconds, tolerance = .28) {
  const cadences = currentCompositionBundle?.cadences || [];
  return cadences.find(item => Math.abs(seconds - item.at.seconds) <= tolerance) || null;
}
function activeOrchestrationAt(seconds) {
  const regions = currentCompositionBundle?.orchestration || [];
  return regions.find(item => seconds >= item.start.seconds && seconds < item.end.seconds)
    || regions.find(item => seconds < item.start.seconds)
    || regions.at(-1)
    || null;
}
function evidenceStatusLabel(basis) {
  const status = String(basis?.status || 'unavailable').replaceAll('-', ' ');
  const method = basis?.source_method ? ` · ${basis.source_method}` : '';
  return `${status}${method}`;
}
function updateVisualizationAvailability(composition) {
  const availability = {
    motifs: Boolean(composition?.motif_occurrences?.length),
    harmony: Boolean(composition?.sonorities?.length),
    orchestration: Boolean(composition?.orchestration?.length),
    resonance: Boolean(composition?.resonance?.samples?.length || currentListenAnalysis?.density?.length),
  };
  document.querySelectorAll('#vizModes .chip').forEach(button => {
    const available = availability[button.dataset.mode];
    if (available === undefined) return;
    button.disabled = !available;
    button.title = available ? '' : 'This evidence layer was not emitted for the current piece.';
    if (!available && vizMode === button.dataset.mode) {
      vizMode = 'hybrid';
      document.querySelectorAll('#vizModes .chip').forEach(other => {
        other.classList.toggle('sel', other.dataset.mode === 'hybrid');
      });
    }
  });
}

function renderSectionLegend(composition) {
  const legend = document.getElementById('sectionLegend');
  const sections = composition?.sections || [];
  legend.innerHTML = '';
  if (!sections.length) return;
  const total = Math.max(.001, composition.duration_seconds || sections.at(-1)?.end.seconds || 1);
  for (const section of sections) {
    const button = document.createElement('button');
    button.type = 'button';
    button.dataset.sectionId = section.id;
    button.style.setProperty('--section-grow', Math.max(.35, (section.end.seconds - section.start.seconds) / total * sections.length));
    button.innerHTML = `${section.label}<small>${formatTime(section.start.seconds)}–${formatTime(section.end.seconds)}</small>`;
    button.addEventListener('click', () => {
      listenAudio.currentTime = Math.min(listenAudio.duration || section.start.seconds, section.start.seconds);
      updateTransport();
    });
    legend.appendChild(button);
  }
}

function updateCurrentStructuralContext() {
  const seconds = listenAudio.currentTime || 0;
  const section = activeSectionAt(seconds);
  const motif = activeMotifAt(seconds);
  const sonority = activeSonorityAt(seconds);
  const cadence = activeCadenceAt(seconds);
  const orchestration = activeOrchestrationAt(seconds);
  document.getElementById('identityForm').textContent = currentCompositionBundle?.form_kind || 'Not emitted';
  document.getElementById('identitySection').textContent = section?.label || 'Not emitted';
  document.getElementById('vizSectionLabel').textContent = section?.label || currentCompositionBundle?.form_kind || currentListenPiece?.grammar || 'Journey';
  document.getElementById('momentMotif').textContent = motif
    ? `${motif.transformation} · ${Math.round((motif.similarity || 0) * 100)}% symbolic match`
    : 'No occurrence at this moment';
  const sonorityLabel = sonority
    ? `${sonority.pitch_classes.join('–')}${sonority.home_key_degree ? ` · degree ${sonority.home_key_degree} ${sonority.home_key_function || ''}` : ''}`
    : 'Not emitted';
  document.getElementById('momentHarmony').textContent = sonorityLabel;
  document.getElementById('momentCadence').textContent = cadence
    ? `${cadence.arrival_pitch_name} · marked arrival`
    : 'No cadence marker nearby';
  document.getElementById('momentVoices').textContent = orchestration?.active_voices?.length
    ? orchestration.active_voices.map(voice => voice.voice_role).join(' · ')
    : 'Not emitted';
  const evidence = motif?.basis || cadence?.basis || sonority?.basis || orchestration?.basis;
  document.getElementById('momentEvidence').textContent = evidence
    ? `Evidence: ${evidenceStatusLabel(evidence)}.`
    : 'No additional evidence layer is active at this moment.';
  document.querySelectorAll('#sectionLegend button').forEach(button => {
    button.classList.toggle('active', button.dataset.sectionId === section?.id);
  });
}

async function showPiece(c, remember = true) {
  if (remember && currentListenPiece && currentListenPiece.id !== c.id) listenHistory.push(currentListenPiece);
  currentListenPiece = c;
  currentListenAnalysis = null;
  currentCompositionBundle = null;
  currentPerformanceBundle = null;
  currentProvenanceBundle = null;
  const love = document.getElementById('listenLove');
  love.textContent = 'Add to Library';
  love.style.color = '';
  applyPalette(c.style);
  const title = c.title || (c.card ? c.card.title : 'seed ' + c.seed);
  const traits = c.card ? c.card.traits.join(' · ') : `${c.style} · ${c.grammar}`;
  document.getElementById('listenTitle').textContent = title;
  document.getElementById('listenTraits').textContent = traits +
    (c.renderer !== 'fluidsynth' ? ' · native render' : '');
  document.getElementById('listenWhy').textContent = (c.why || [])[0] || 'Grounded explanation is available in the context rail.';
  document.getElementById('identityStyle').textContent = c.style || '—';
  document.getElementById('identityGrammar').textContent = c.grammar || '—';
  document.getElementById('identityMeter').textContent = c.meter ? `${c.meter}/4` : '—';
  document.getElementById('identityRenderer').textContent = c.renderer || '—';
  document.getElementById('headerNowTitle').textContent = title;
  document.getElementById('headerNowMeta').textContent = `${c.style} · ${c.grammar} · seed ${c.seed}`;
  document.getElementById('vizSectionLabel').textContent = c.grammar || 'Journey';
  document.getElementById('identityForm').textContent = 'Loading…';
  document.getElementById('identitySection').textContent = 'Loading…';
  renderWhyEvidence(c);
  listenAudio.src = '/api/audio/' + c.id;
  document.getElementById('listenDownloadWav').href = '/api/audio/' + c.id;
  document.getElementById('listenDownloadWav').download = 'muse_seed' + c.seed + '.wav';
  document.getElementById('listenDownloadMidi').href = '/api/midi/' + c.id;
  document.getElementById('listenDownloadMidi').download = 'muse_seed' + c.seed + '.mid';
  document.getElementById('listenPrevious').disabled = listenHistory.length === 0;
  const bundles = await loadPieceBundles(c.id);
  currentCompositionBundle = bundles.composition;
  currentPerformanceBundle = bundles.performance;
  currentProvenanceBundle = bundles.provenance;
  currentListenAnalysis = bundles.analysis;
  c._compositionBundle = currentCompositionBundle;
  c._performanceBundle = currentPerformanceBundle;
  c._provenanceBundle = currentProvenanceBundle;
  c._bundleWarnings = bundles.warnings;
  c._performedAnalysis = currentListenAnalysis;
  renderSectionLegend(currentCompositionBundle);
  renderVoiceBlend(currentListenAnalysis);
  drawEmotionArc(currentListenAnalysis, currentCompositionBundle);
  updateVisualizationAvailability(currentCompositionBundle);
  updateTransport();
}
async function nextListenPiece(autoplay) {
  let c = listenQueue.shift();
  if (!c && latestBatch.length) {
    c = latestBatch[listenIndex % latestBatch.length];
    c._listenIntent ||= createIntentSnapshot();
    c._journeyRelation ||= relationToCurrent(c);
    listenIndex += 1;
  }
  updateNextJourneyPreview();
  if (!c) {
    listenState.textContent = 'Composing the first piece…';
    c = await composeListenPiece();
    listenState.textContent = '';
  }
  if (!c) { listenState.textContent = "Couldn't reach the composer — try again"; return; }
  await showPiece(c);
  if (autoplay) await playListen();
  prefetchListen();
}
async function previousListenPiece() {
  const previous = listenHistory.pop();
  if (!previous) return;
  if (currentListenPiece) listenQueue.unshift(currentListenPiece);
  await showPiece(previous, false);
  await playListen();
  updateNextJourneyPreview();
}
async function playListen() {
  ensureAnalyser();
  await audioCtx.resume();
  await listenAudio.play();
}
function toggleListenPlayback() {
  if (listenAudio.paused) playListen(); else listenAudio.pause();
}
function updateTransport() {
  const playing = !listenAudio.paused && !listenAudio.ended;
  const icon = playing ? 'Ⅱ' : '▶';
  document.getElementById('listenPlay').textContent = icon;
  document.getElementById('globalPlay').textContent = icon;
  document.getElementById('listenProgress').textContent = `${formatTime(listenAudio.currentTime)} / ${formatTime(listenAudio.duration || currentListenPiece?.duration_secs || 0)}`;
  document.getElementById('vizTimeLabel').textContent = formatTime(listenAudio.currentTime);
  updateCurrentStructuralContext();
}
function startListen() {
  drawViz();
  if (!listenAudio.src) nextListenPiece(false);
  prefetchListen();
}
document.getElementById('listenNext').addEventListener('click', () => nextListenPiece(true));
document.getElementById('listenPrevious').addEventListener('click', previousListenPiece);
document.getElementById('listenPlay').addEventListener('click', toggleListenPlayback);
document.getElementById('globalPlay').addEventListener('click', toggleListenPlayback);
document.getElementById('listenShuffle').addEventListener('click', () => {
  listenQueue = [];
  listenGeneration += 7;
  prefetchListen();
});
document.getElementById('journeyKind').addEventListener('change', () => {
  listenQueue = [];
  prefetchListen();
});
document.getElementById('whyThisPiece').addEventListener('click', () => {
  const panel = document.getElementById('whyPanel');
  panel.open = true;
  panel.scrollIntoView({behavior: 'smooth', block: 'nearest'});
});
document.getElementById('listenLove').addEventListener('click', async (ev) => {
  if (!currentListenPiece) return;
  const res = await fetch('/api/keeper/' + currentListenPiece.id, { method: 'POST' });
  if (res.ok) { ev.target.textContent = 'Added to Library'; ev.target.style.color = 'var(--green)'; }
});
const vizCanvas = document.getElementById('viz');
vizCanvas.addEventListener('click', ev => {
  const rect = vizCanvas.getBoundingClientRect();
  const x = ev.clientX - rect.left, y = ev.clientY - rect.top;
  if (vizMode === 'hybrid') {
    const cx = rect.width / 2, cy = rect.height / 2;
    const distance = Math.hypot(x - cx, y - cy);
    if (distance > Math.min(rect.width, rect.height) * .21 && listenAudio.duration) {
      let angle = Math.atan2(y - cy, x - cx) + Math.PI / 2;
      if (angle < 0) angle += Math.PI * 2;
      listenAudio.currentTime = angle / (Math.PI * 2) * listenAudio.duration;
      updateTransport();
      return;
    }
  } else if (listenAudio.duration) {
    listenAudio.currentTime = clamp(x / rect.width, 0, 1) * listenAudio.duration;
    updateTransport();
    return;
  }
  toggleListenPlayback();
});
vizCanvas.addEventListener('keydown', ev => {
  if (ev.key === ' ' || ev.key === 'Enter') { ev.preventDefault(); toggleListenPlayback(); }
  if (ev.key === 'ArrowRight' && listenAudio.duration) listenAudio.currentTime = Math.min(listenAudio.duration, listenAudio.currentTime + 5);
  if (ev.key === 'ArrowLeft') listenAudio.currentTime = Math.max(0, listenAudio.currentTime - 5);
});
listenAudio.addEventListener('play', updateTransport);
listenAudio.addEventListener('pause', updateTransport);
listenAudio.addEventListener('loadedmetadata', updateTransport);
listenAudio.addEventListener('timeupdate', () => { updateTransport(); if (document.getElementById('researchView').classList.contains('on')) { updateResearchInspector(); renderResearchTimeline(currentListenPiece?._performedAnalysis || currentListenAnalysis); renderResearchEvidenceWorkspaces(); } });
listenAudio.addEventListener('ended', () => nextListenPiece(true));

// ── Research: a grounded overview from data the current server emits ────
let researchView = 'overview';
const researchTimeline = document.getElementById('researchTimeline');
function metricRow(label, value) {
  const safe = clamp(Number(value) || 0, 0, 1);
  return `<div class="metric-row"><span>${label}</span><i style="--metric-value:${(safe * 100).toFixed(1)}%"></i><b>${safe.toFixed(2)}</b></div>`;
}
function renderResearchIdentity(c) {
  const box = document.getElementById('researchIdentity');
  const rows = [];
  for (const trait of c.card?.traits || []) rows.push(`<div class="identity-evidence-row"><span>Identity trait</span><b>${trait}</b></div>`);
  if (c.novelty) {
    const channels = [['Melodic distance', c.novelty.melodic], ['Rhythmic distance', c.novelty.rhythmic], ['Harmonic distance', c.novelty.harmonic], ['Orchestration distance', c.novelty.orchestration]];
    for (const [label, value] of channels) if (Number.isFinite(value)) rows.push(`<div class="identity-evidence-row"><span>${label}</span><b>${Number(value).toFixed(2)}</b></div>`);
  }
  if (!rows.length) rows.push('<p class="identity-unavailable">The candidate did not emit an identity card or batch-novelty evidence. Style and grammar remain available as observed metadata.</p>');
  box.innerHTML = rows.join('');
}
function researchBundleContext() {
  const piece = currentListenPiece || {};
  return {
    composition: piece._compositionBundle || currentCompositionBundle,
    performance: piece._performanceBundle || currentPerformanceBundle,
    provenance: piece._provenanceBundle || currentProvenanceBundle,
    warnings: piece._bundleWarnings || [],
  };
}
function shortHash(value) {
  if (!value) return 'Not recorded';
  return value.length > 18 ? `${value.slice(0, 10)}…${value.slice(-6)}` : value;
}
function normalizedLaneName(value) {
  return String(value || '').trim().toLowerCase().replace(/[^a-z0-9]+/g, '');
}
function sourceMethodLabel(value) {
  const labels = {
    'score-section-intensity-runs-v1': 'score section-intensity annotations (v1)',
    'score-emphasis-annotations-v1': 'score phrase-emphasis annotations (v1)',
    'resolved-composition-recipe-motif-v1': 'resolved recipe motif (v1)',
    'recipe-motif-score-window-scan-v1': 'bounded score-window motif scan (v1)',
    'score-cadential-emphasis-v1': 'score cadential-emphasis markers (v1)',
    'active-score-pitch-classes-per-beat-v1': 'active score pitch classes per beat (v1)',
    'symbolic-voice-assignment-by-structural-region-v1': 'symbolic voice assignment by structural region (v1)',
    'score-activity-resonance-proxy-v1': 'score activity resonance proxy (v1)',
  };
  return labels[value] || String(value || 'an emitted score annotation');
}
function limitationLabel(value) {
  return String(value || '')
    .replace(/([a-z])([A-Z])/g, '$1 $2')
    .replace(/Digest$/i, ' digest')
    .replace(/Revision$/i, ' revision')
    .toLowerCase()
    .replace(/^./, letter => letter.toUpperCase()) + ' is not recorded.';
}
function renderResearchTimeline(analysis) {
  const canvas = researchTimeline;
  const legend = document.getElementById('researchTimelineLegend');
  const { composition } = researchBundleContext();
  const symbolicNotes = composition?.notes || [];
  const performedEvents = analysis?.events || [];
  const duration = Math.max(
    Number(composition?.duration_seconds) || 0,
    Number(analysis?.duration) || 0,
    Number(currentListenPiece?.duration_secs) || 0,
    .001,
  );
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const w = Math.max(1, canvas.clientWidth), h = Math.max(1, canvas.clientHeight);
  canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
  const ctx = canvas.getContext('2d'); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, w, h);
  if (!symbolicNotes.length && !performedEvents.length) {
    ctx.fillStyle = 'rgba(158,154,148,.65)'; ctx.font = '12px Inter, sans-serif';
    ctx.fillText('Symbolic and performed note events are unavailable.', 18, 30);
    legend.innerHTML = '';
    return;
  }

  const symbolicRoles = [...new Set(symbolicNotes.map(note => note.voice_role).filter(Boolean))];
  const performedVoices = analysis?.voices || [];
  const lanes = performedVoices.map((voice, index) => ({
    key: normalizedLaneName(voice.name || voice.id || `voice-${index}`),
    label: voice.name || voice.id || `Voice ${index + 1}`,
    instrument: voice.instrument || '',
    performanceIndex: index,
  }));
  for (const role of symbolicRoles) {
    const key = normalizedLaneName(role);
    if (!lanes.some(lane => lane.key === key)) lanes.push({ key, label: role, instrument: '', performanceIndex: null });
  }
  if (!lanes.length) lanes.push({ key: 'score', label: 'Score', instrument: '', performanceIndex: null });
  const laneIndex = new Map(lanes.map((lane, index) => [lane.key, index]));
  const top = composition?.sections?.length ? 28 : 8;
  const laneH = Math.max(26, (h - top) / Math.max(1, lanes.length));
  const sectionColors = ['215,164,95', '125,100,200', '102,174,190', '190,118,126'];

  ctx.font = '10px Inter, sans-serif';
  for (let i = 0; i < (composition?.sections || []).length; i++) {
    const section = composition.sections[i];
    const x = clamp(section.start.seconds / duration, 0, 1) * w;
    const x2 = clamp(section.end.seconds / duration, 0, 1) * w;
    const rgb = sectionColors[i % sectionColors.length];
    ctx.fillStyle = `rgba(${rgb},${i % 2 ? .035 : .065})`;
    ctx.fillRect(x, 0, Math.max(1, x2 - x), h);
    ctx.strokeStyle = `rgba(${rgb},.28)`;
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
    if (x2 - x > 48) {
      ctx.fillStyle = `rgba(${rgb},.78)`;
      ctx.fillText(section.label, x + 6, 17);
    }
  }

  lanes.forEach((lane, index) => {
    const y = top + index * laneH;
    ctx.fillStyle = index % 2 ? 'rgba(255,255,255,.012)' : 'rgba(255,255,255,.024)';
    ctx.fillRect(0, y, w, laneH);
    ctx.fillStyle = 'rgba(176,171,164,.68)'; ctx.fillText(lane.label, 8, y + 14);
    ctx.strokeStyle = 'rgba(255,255,255,.04)'; ctx.beginPath(); ctx.moveTo(0, y + laneH); ctx.lineTo(w, y + laneH); ctx.stroke();
  });

  const symbolicById = new Map(symbolicNotes.map(note => [note.id, note]));
  for (const note of symbolicNotes) {
    const index = laneIndex.get(normalizedLaneName(note.voice_role)) ?? 0;
    const x = clamp(note.onset.seconds / duration, 0, 1) * w;
    const width = Math.max(1.5, clamp(note.duration_seconds / duration, 0, 1) * w);
    const y = top + index * laneH + Math.min(20, laneH * .34);
    ctx.fillStyle = `rgba(125,100,200,${.35 + clamp(Number(note.velocity) || .5, 0, 1) * .45})`;
    ctx.fillRect(x, y, width, Math.max(2, Math.min(5, laneH * .12)));
  }

  for (const event of performedEvents) {
    const voice = performedVoices[event.voiceIndex];
    const index = laneIndex.get(normalizedLaneName(voice?.name || voice?.id)) ?? Math.min(event.voiceIndex || 0, lanes.length - 1);
    const x = clamp(event.start / duration, 0, 1) * w;
    const width = Math.max(1.5, clamp(event.duration / duration, 0, 1) * w);
    const y = top + index * laneH + Math.min(laneH - 8, Math.max(16, laneH * .62));
    ctx.fillStyle = `rgba(102,174,190,${.35 + clamp(event.velocity, 0, 1) * .55})`;
    ctx.fillRect(x, y, width, Math.max(2, Math.min(6, laneH * .14)));
    if (event.sourceNoteId && Number.isFinite(event.onsetDeviation) && Math.abs(event.onsetDeviation) > .002) {
      const source = symbolicById.get(event.sourceNoteId);
      if (source) {
        const sourceX = clamp(source.onset.seconds / duration, 0, 1) * w;
        ctx.strokeStyle = 'rgba(238,232,223,.18)';
        ctx.beginPath(); ctx.moveTo(sourceX, y - 3); ctx.lineTo(x, y - 3); ctx.stroke();
      }
    }
  }

  for (const occurrence of composition?.motif_occurrences || []) {
    const x = clamp(occurrence.start.seconds / duration, 0, 1) * w;
    const x2 = clamp(occurrence.end.seconds / duration, 0, 1) * w;
    ctx.fillStyle = `rgba(125,100,200,${.16 + clamp(Number(occurrence.similarity) || 0, 0, 1) * .24})`;
    ctx.fillRect(x, Math.max(2, top - 10), Math.max(2, x2 - x), 5);
  }
  for (const cadence of composition?.cadences || []) {
    const x = clamp(cadence.at.seconds / duration, 0, 1) * w;
    ctx.strokeStyle = 'rgba(215,164,95,.72)';
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
  }

  for (const phrase of composition?.phrases || []) {
    const x = clamp(phrase.start.seconds / duration, 0, 1) * w;
    ctx.fillStyle = phrase.closes_with_cadential_marker ? 'rgba(215,164,95,.72)' : 'rgba(238,232,223,.3)';
    ctx.fillRect(x, top - 4, 1, h - top + 4);
  }

  const progress = listenAudio.duration ? clamp(listenAudio.currentTime / listenAudio.duration, 0, 1) : clamp((listenAudio.currentTime || 0) / duration, 0, 1);
  ctx.fillStyle = 'rgba(238,232,223,.92)'; ctx.fillRect(progress * w, 0, 1, h);
  legend.innerHTML = [
    '<span style="--legend-color:rgb(125,100,200)">Symbolic score</span>',
    '<span style="--legend-color:rgb(102,174,190)">Rendered performance</span>',
    ...(composition?.motif_occurrences?.length ? ['<span style="--legend-color:rgb(125,100,200)">Motif occurrence</span>'] : []),
    ...(composition?.cadences?.length ? ['<span style="--legend-color:rgb(215,164,95)">Cadence marker</span>'] : []),
    ...lanes.map(lane => `<span style="--legend-color:rgba(215,164,95,.45)">${lane.label}${lane.instrument ? ` · ${lane.instrument}` : ''}</span>`),
  ].join('');
}
function prepareEvidenceCanvas(id) {
  const canvas = document.getElementById(id);
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const w = Math.max(1, canvas.clientWidth), h = Math.max(1, canvas.clientHeight);
  canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, w, h);
  return { canvas, ctx, w, h };
}
function evidenceBadgeText(basis, count) {
  if (!count) return 'Not emitted';
  const status = String(basis?.status || 'available').replaceAll('-', ' ');
  const confidence = Number.isFinite(basis?.confidence) ? ` · ${Math.round(basis.confidence * 100)}%` : '';
  return `${status}${confidence}`;
}
function renderResearchMotifs() {
  const { composition } = researchBundleContext();
  const definitions = composition?.motif_definitions || [];
  const occurrences = composition?.motif_occurrences || [];
  const badge = document.getElementById('motifEvidenceBadge');
  const summary = document.getElementById('researchMotifSummary');
  const { canvas, ctx, w, h } = prepareEvidenceCanvas('researchMotifMap');
  const basis = occurrences[0]?.basis || definitions[0]?.basis;
  badge.textContent = evidenceBadgeText(basis, occurrences.length || definitions.length);
  badge.dataset.status = basis?.status || 'unavailable';
  if (!occurrences.length) {
    ctx.fillStyle = 'rgba(158,154,148,.68)'; ctx.font = '12px Inter, sans-serif';
    ctx.fillText('No occurrence met the declared symbolic threshold.', 18, 30);
    summary.innerHTML = `<p class="evidence-empty">${definitions.length ? 'The recipe motif is available, but no additional occurrence survived conservative classification.' : 'No motif definition or occurrence was emitted.'}</p>`;
    canvas._evidenceTargets = [];
    return;
  }
  const duration = Math.max(.001, composition.duration_seconds || 1);
  const top = 40, laneH = Math.max(34, (h - top - 20) / 3);
  for (const section of composition.sections || []) {
    const x = section.start.seconds / duration * w;
    const x2 = section.end.seconds / duration * w;
    ctx.fillStyle = 'rgba(255,255,255,.018)'; ctx.fillRect(x, 0, Math.max(1, x2 - x), h);
    ctx.strokeStyle = 'rgba(255,255,255,.055)'; ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
    if (x2 - x > 55) { ctx.fillStyle = 'rgba(176,171,164,.58)'; ctx.font = '9px Inter, sans-serif'; ctx.fillText(section.label, x + 5, 15); }
  }
  const targets = [];
  occurrences.forEach((occurrence, index) => {
    const x = occurrence.start.seconds / duration * w;
    const x2 = occurrence.end.seconds / duration * w;
    const y = top + (index % 3) * laneH;
    const similarity = clamp(Number(occurrence.similarity) || 0, 0, 1);
    ctx.fillStyle = `rgba(125,100,200,${.26 + similarity * .55})`;
    ctx.fillRect(x, y, Math.max(5, x2 - x), Math.max(12, laneH - 10));
    ctx.strokeStyle = 'rgba(185,163,242,.72)'; ctx.strokeRect(x + .5, y + .5, Math.max(4, x2 - x - 1), Math.max(11, laneH - 11));
    ctx.fillStyle = 'rgba(238,232,223,.9)'; ctx.font = '9px Inter, sans-serif';
    ctx.fillText(`${occurrence.transformation} · ${Math.round(similarity * 100)}%`, x + 5, y + 14, Math.max(20, x2 - x - 8));
    targets.push({ seconds: occurrence.start.seconds, x, x2, label: occurrence.transformation, id: occurrence.id });
  });
  const progress = clamp((listenAudio.currentTime || 0) / duration, 0, 1);
  ctx.fillStyle = 'rgba(238,232,223,.92)'; ctx.fillRect(progress * w, 0, 1, h);
  canvas._evidenceTargets = targets;
  const definition = definitions[0];
  const limitations = basis?.limitations || [];
  summary.innerHTML = `
    <div class="evidence-summary-grid">
      <div><span>Recipe motif</span><b>${definition?.degrees?.join(' · ') || 'Not emitted'}</b></div>
      <div><span>Occurrences</span><b>${occurrences.length}</b></div>
      <div><span>Method</span><b>${sourceMethodLabel(basis?.source_method)}</b></div>
    </div>
    <p>${limitations[0] || 'Occurrence blocks link directly to score note identifiers.'}</p>`;
}
function renderResearchHarmony() {
  const { composition } = researchBundleContext();
  const regions = composition?.sonorities || [];
  const cadences = composition?.cadences || [];
  const badge = document.getElementById('harmonyEvidenceBadge');
  const summary = document.getElementById('researchHarmonySummary');
  const { canvas, ctx, w, h } = prepareEvidenceCanvas('researchHarmonyPath');
  const basis = regions[0]?.basis || cadences[0]?.basis;
  badge.textContent = evidenceBadgeText(basis, regions.length || cadences.length);
  badge.dataset.status = basis?.status || 'unavailable';
  if (!regions.length) {
    ctx.fillStyle = 'rgba(158,154,148,.68)'; ctx.font = '12px Inter, sans-serif';
    ctx.fillText('No score-side sonority regions were emitted.', 18, 30);
    summary.innerHTML = '<p class="evidence-empty">Harmony remains unavailable rather than fabricated.</p>';
    canvas._evidenceTargets = [];
    return;
  }
  const duration = Math.max(.001, composition.duration_seconds || 1);
  const top = 44, bandH = Math.max(70, h - 88);
  const targets = [];
  regions.forEach((region, index) => {
    const x = region.start.seconds / duration * w;
    const x2 = region.end.seconds / duration * w;
    const exact = Boolean(region.home_key_degree);
    ctx.fillStyle = exact ? `rgba(215,164,95,${.18 + (index % 2) * .05})` : 'rgba(102,174,190,.13)';
    ctx.fillRect(x, top, Math.max(1, x2 - x), bandH);
    ctx.strokeStyle = exact ? 'rgba(215,164,95,.48)' : 'rgba(102,174,190,.3)'; ctx.strokeRect(x + .5, top + .5, Math.max(1, x2 - x - 1), bandH - 1);
    if (x2 - x > 34) {
      ctx.fillStyle = 'rgba(238,232,223,.88)'; ctx.font = '9px Inter, sans-serif';
      ctx.fillText(region.pitch_classes.join('–'), x + 4, top + 16, Math.max(15, x2 - x - 7));
      if (exact) { ctx.fillStyle = 'rgba(215,164,95,.82)'; ctx.fillText(`degree ${region.home_key_degree}`, x + 4, top + 31, Math.max(15, x2 - x - 7)); }
    }
    targets.push({ seconds: region.start.seconds, x, x2, label: region.pitch_classes.join('–'), id: region.id });
  });
  cadences.forEach(cadence => {
    const x = cadence.at.seconds / duration * w;
    ctx.strokeStyle = 'rgba(238,232,223,.8)'; ctx.lineWidth = 1.4; ctx.beginPath(); ctx.moveTo(x, top - 22); ctx.lineTo(x, top + bandH + 8); ctx.stroke();
    ctx.fillStyle = 'rgba(238,232,223,.86)'; ctx.font = '9px Inter, sans-serif'; ctx.fillText(cadence.arrival_pitch_name, x + 3, top - 9);
  });
  const progress = clamp((listenAudio.currentTime || 0) / duration, 0, 1);
  ctx.fillStyle = 'rgba(238,232,223,.92)'; ctx.fillRect(progress * w, 0, 1, h);
  canvas._evidenceTargets = targets;
  const exactCount = regions.filter(region => region.home_key_degree).length;
  const limitations = [...new Set([...(basis?.limitations || []), ...(cadences[0]?.basis?.limitations || [])])];
  summary.innerHTML = `
    <div class="evidence-summary-grid">
      <div><span>Sonority regions</span><b>${regions.length}</b></div>
      <div><span>Exact home-key triads</span><b>${exactCount}</b></div>
      <div><span>Marked arrivals</span><b>${cadences.length}</b></div>
    </div>
    <p>${limitations[0] || 'Pitch-class regions are observed from sounding score notes.'}</p>`;
}
function renderResearchOrchestration() {
  const { composition } = researchBundleContext();
  const regions = composition?.orchestration || [];
  const badge = document.getElementById('orchestrationEvidenceBadge');
  const summary = document.getElementById('researchOrchestrationSummary');
  const { canvas, ctx, w, h } = prepareEvidenceCanvas('researchOrchestrationMap');
  const basis = regions[0]?.basis;
  badge.textContent = evidenceBadgeText(basis, regions.length);
  badge.dataset.status = basis?.status || 'unavailable';
  if (!regions.length) {
    ctx.fillStyle = 'rgba(158,154,148,.68)'; ctx.font = '12px Inter, sans-serif';
    ctx.fillText('No composition-side voice regions were emitted.', 18, 30);
    summary.innerHTML = '<p class="evidence-empty">Rendered prominence is not inferred from missing data.</p>';
    canvas._evidenceTargets = [];
    return;
  }
  const duration = Math.max(.001, composition.duration_seconds || 1);
  const roles = [...new Set(regions.flatMap(region => (region.active_voices || []).map(voice => voice.voice_role)))];
  const top = 28, laneH = Math.max(30, (h - top - 15) / Math.max(1, roles.length));
  const targets = [];
  roles.forEach((role, laneIndex) => {
    const y = top + laneIndex * laneH;
    ctx.fillStyle = laneIndex % 2 ? 'rgba(255,255,255,.012)' : 'rgba(255,255,255,.024)'; ctx.fillRect(0, y, w, laneH);
    ctx.fillStyle = 'rgba(176,171,164,.7)'; ctx.font = '9px Inter, sans-serif'; ctx.fillText(role, 6, y + 13);
    for (const region of regions) {
      const voice = region.active_voices?.find(item => item.voice_role === role);
      if (!voice) continue;
      const x = region.start.seconds / duration * w;
      const x2 = region.end.seconds / duration * w;
      const total = Math.max(1, region.active_voices.reduce((sum, item) => sum + item.note_count, 0));
      const share = voice.note_count / total;
      ctx.fillStyle = role.toLowerCase().includes('melody') ? `rgba(215,164,95,${.22 + share * .55})` : role.toLowerCase().includes('bass') ? `rgba(102,174,190,${.2 + share * .5})` : `rgba(125,100,200,${.18 + share * .48})`;
      ctx.fillRect(x, y + 18, Math.max(2, x2 - x), Math.max(5, laneH - 23));
      targets.push({ seconds: region.start.seconds, x, x2, label: role, id: region.id });
    }
  });
  const progress = clamp((listenAudio.currentTime || 0) / duration, 0, 1);
  ctx.fillStyle = 'rgba(238,232,223,.92)'; ctx.fillRect(progress * w, 0, 1, h);
  canvas._evidenceTargets = targets;
  const registers = regions.flatMap(region => [region.register_min_midi, region.register_max_midi]).filter(Number.isFinite);
  const minRegister = registers.length ? Math.min(...registers) : null;
  const maxRegister = registers.length ? Math.max(...registers) : null;
  summary.innerHTML = `
    <div class="evidence-summary-grid">
      <div><span>Structural regions</span><b>${regions.length}</b></div>
      <div><span>Assigned roles</span><b>${roles.length}</b></div>
      <div><span>Register span</span><b>${minRegister == null ? 'Not emitted' : `${minRegister}–${maxRegister}`}</b></div>
    </div>
    <p>${basis?.limitations?.[0] || 'This view shows score assignments, not rendered loudness or perceptual prominence.'}</p>`;
}
function renderResearchEvidenceWorkspaces() {
  if (!currentListenPiece) return;
  if (researchView === 'motifs') renderResearchMotifs();
  if (researchView === 'harmony') renderResearchHarmony();
  if (researchView === 'orchestration') renderResearchOrchestration();
}
function seekFromEvidenceCanvas(canvas, event) {
  if (!currentListenPiece) return;
  const { composition } = researchBundleContext();
  const duration = Math.max(.001, Number(composition?.duration_seconds) || Number(currentListenPiece.duration_secs) || Number(listenAudio.duration) || 1);
  const rect = canvas.getBoundingClientRect();
  const seconds = clamp((event.clientX - rect.left) / Math.max(1, rect.width), 0, 1) * duration;
  listenAudio.currentTime = seconds;
  updateResearchInspector(seconds);
  renderResearchTimeline(currentListenPiece._performedAnalysis || currentListenAnalysis);
  renderResearchEvidenceWorkspaces();
}

function updateResearchInspector(seconds = listenAudio.currentTime || 0) {
  if (!currentListenPiece) return;
  const analysis = currentListenPiece._performedAnalysis || currentListenAnalysis;
  const { composition } = researchBundleContext();
  const section = (composition?.sections || []).find(region => seconds >= region.start.seconds && seconds < region.end.seconds) || null;
  const phrase = (composition?.phrases || []).find(region => seconds >= region.start.seconds && seconds < region.end.seconds) || null;
  const symbolic = (composition?.notes || []).filter(note => note.onset.seconds <= seconds && note.onset.seconds + note.duration_seconds >= seconds);
  const active = (analysis?.events || []).filter(event => event.start <= seconds && event.start + event.duration >= seconds);
  const motif = (composition?.motif_occurrences || []).find(item => seconds >= item.start.seconds && seconds < item.end.seconds) || null;
  const sonority = (composition?.sonorities || []).find(item => seconds >= item.start.seconds && seconds < item.end.seconds) || null;
  const cadence = (composition?.cadences || []).find(item => Math.abs(seconds - item.at.seconds) <= .28) || null;
  const orchestration = (composition?.orchestration || []).find(item => seconds >= item.start.seconds && seconds < item.end.seconds) || null;
  const voiceNames = [...new Set(active.map(event => analysis?.voices?.[event.voiceIndex]?.name).filter(Boolean))];
  document.getElementById('inspectorMoment').textContent = `${section?.label || composition?.form_kind || 'Piece'} · ${formatTime(seconds)}`;
  const structural = [section?.label && `section ${section.label}`, phrase?.label && `phrase ${phrase.label}`].filter(Boolean).join(', ');
  document.getElementById('inspectorObservation').textContent = [
    structural ? `Current location: ${structural}.` : 'No score-grounded section or phrase spans this moment.',
    `${symbolic.length} symbolic note event${symbolic.length === 1 ? '' : 's'} and ${active.length} performed event${active.length === 1 ? '' : 's'} are active${voiceNames.length ? ` across ${voiceNames.join(', ')}` : ''}.`,
    motif ? `Motif occurrence: ${motif.transformation} at ${Math.round((motif.similarity || 0) * 100)}% symbolic similarity.` : '',
    sonority ? `Sounding pitch classes: ${sonority.pitch_classes.join(', ')}${sonority.home_key_degree ? `; exact declared-home-key degree ${sonority.home_key_degree}` : ''}.` : '',
    cadence ? `A cadential-emphasis marker arrives on ${cadence.arrival_pitch_name}.` : '',
    orchestration?.active_voices?.length ? `Assigned score voices: ${orchestration.active_voices.map(voice => voice.voice_role).join(', ')}.` : '',
  ].filter(Boolean).join(' ');
  const interpretations = [];
  if (section) interpretations.push(`${section.label} is exposed from ${sourceMethodLabel(section.source_method)}.`);
  if (motif) interpretations.push(`The motif relation is ${motif.basis?.status || 'inferred'} by ${sourceMethodLabel(motif.basis?.source_method)}; it is not an authored occurrence label.`);
  if (sonority) interpretations.push('The sonority is an observed pitch-class set. Degree and function appear only for an exact triad in the declared home key; modulation is not inferred.');
  if (cadence) interpretations.push('The score marks an arrival, but Muse does not classify its cadence type from that marker alone.');
  if (orchestration) interpretations.push('Voice roles are composition assignments, not claims about rendered prominence.');
  if (!interpretations.length && currentListenPiece.why?.[0]) interpretations.push(`Piece-level interpretation: ${currentListenPiece.why[0]}`);
  document.getElementById('inspectorInterpretation').textContent = interpretations.join(' ') || 'No interpretation was emitted for this moment.';
  const scoreIds = symbolic.slice(0, 4).map(note => note.id);
  const performedIds = active.slice(0, 4).map(event => event.id || event.sourceNoteId).filter(Boolean);
  const evidence = [];
  const structuralIds = [motif?.id, sonority?.id, cadence?.id, orchestration?.id].filter(Boolean);
  if (composition) evidence.push(`/api/piece/${currentListenPiece.id}/listen-bundle${scoreIds.length ? ` · score IDs ${scoreIds.join(', ')}` : ''}${structuralIds.length ? ` · evidence IDs ${structuralIds.join(', ')}` : ''}`);
  if (analysis) evidence.push(`/api/piece/${currentListenPiece.id}/performance-bundle${performedIds.length ? ` · performed IDs ${performedIds.join(', ')}` : ''}`);
  document.getElementById('inspectorEvidence').textContent = evidence.length
    ? `Evidence at ${seconds.toFixed(2)} s: ${evidence.join('; ')}.`
    : `No inspectable score or performance bundle was available at ${seconds.toFixed(2)} s.`;
}
function renderAnalysisAvailability(c, analysis) {
  const { composition, provenance, warnings } = researchBundleContext();
  const entries = [
    ['Composition metadata', true],
    ['Symbolic score events', Boolean(composition?.notes?.length)],
    ['Form and sections', Boolean(composition?.form_kind && composition?.sections?.length)],
    ['Phrase regions', Boolean(composition?.phrases?.length)],
    ['Performed notes', Boolean(analysis?.events?.length)],
    ['Metric summary', Number.isFinite(c.phi)],
    ['Motif definition', Boolean(composition?.motif_definitions?.length)],
    ['Motif occurrences', Boolean(composition?.motif_occurrences?.length)],
    ['Sonority regions', Boolean(composition?.sonorities?.length)],
    ['Cadence markers', Boolean(composition?.cadences?.length)],
    ['Orchestration regions', Boolean(composition?.orchestration?.length)],
    ['Structural activity curve', Boolean(composition?.resonance?.samples?.length)],
    ['Provenance manifest', Boolean(provenance)],
  ];
  document.getElementById('analysisAvailability').innerHTML = entries.map(([label, available]) =>
    `<li><span>${label}</span><span style="--availability-color:${available ? 'var(--green)' : 'var(--dim)'}">${available ? 'Available' : 'Not emitted'}</span></li>`).join('');
  const warningBox = document.getElementById('researchBundleWarnings');
  const messages = [
    ...(warnings || []).map(warning => warning.message || warning.code).filter(Boolean),
    ...(warnings?.length ? [] : (provenance?.reproduction?.limitations || []).map(limitationLabel)),
  ];
  warningBox.hidden = !messages.length;
  warningBox.innerHTML = messages.length
    ? `<strong>Declared limitations</strong>${messages.map(message => `<p>${message}</p>`).join('')}`
    : '';
}
function renderResearch() {
  const empty = document.getElementById('researchEmpty');
  const workspace = document.getElementById('researchWorkspace');
  if (!currentListenPiece) {
    empty.hidden = false; workspace.hidden = true; return;
  }
  const c = currentListenPiece;
  const analysis = c._performedAnalysis || currentListenAnalysis;
  const { composition, provenance } = researchBundleContext();
  empty.hidden = true; workspace.hidden = false;
  workspace.dataset.profile = document.getElementById('analysisProfile').value.toLowerCase().replace(' ', '-');
  document.querySelector('.research-body').dataset.activeView = researchView;
  const meter = composition?.meter_map?.[0];
  const meterLabel = meter ? `${meter.numerator}/${meter.denominator}` : (c.meter ? `${c.meter}/4` : '—');
  const duration = composition?.duration_seconds || c.duration_secs || analysis?.duration || 0;
  document.getElementById('researchTitle').textContent = c.title || `seed ${c.seed}`;
  document.getElementById('researchMeta').textContent = `${c.style} · ${composition?.form_kind || c.grammar} · ${meterLabel} · ${formatTime(duration)}`;
  document.getElementById('researchStyle').textContent = c.style || provenance?.style_name || '—';
  document.getElementById('researchGrammar').textContent = c.grammar || '—';
  document.getElementById('researchForm').textContent = composition?.form_kind || 'Not emitted';
  document.getElementById('researchMeter').textContent = meterLabel;
  document.getElementById('researchDuration').textContent = formatTime(duration);
  document.getElementById('researchSectionCount').textContent = composition ? String(composition.sections?.length || 0) : 'Not emitted';
  document.getElementById('researchPhraseCount').textContent = composition ? String(composition.phrases?.length || 0) : 'Not emitted';
  document.getElementById('researchSymbolicCount').textContent = composition ? String(composition.notes?.length || 0) : 'Not emitted';
  renderResearchIdentity(c);
  document.getElementById('researchMetrics').innerHTML =
    metricRow('Integration', c.phi) + metricRow('Local coherence', c.local_coherence) + metricRow('Global coherence', c.global_coherence);
  const insights = c.why || [];
  document.getElementById('researchInsights').innerHTML = insights.length
    ? insights.map(line => `<p>${line}</p>`).join('')
    : '<p>No interpretive summary was emitted.</p>';
  const reproduction = provenance?.reproduction;
  const provenanceRows = provenance ? [
    ['Candidate ID', c.id],
    ['Seed', provenance.seed],
    ['Recipe schema', provenance.recipe_schema_version],
    ['Recipe SHA-256', provenance.recipe_sha256],
    ['Score SHA-256', provenance.score_sha256],
    ['Audio SHA-256', provenance.audio_sha256],
    ['Muse engine', provenance.muse_engine_version],
    ['Theory engine', provenance.theory_engine_version],
    ['Renderer', `${provenance.renderer_name}${provenance.renderer_version ? ` ${provenance.renderer_version}` : ''}`],
    ['Muse revision', provenance.muse_source_revision || 'Not recorded'],
    ['Score exact', reproduction?.symbolic_score_exact ? 'Yes' : 'No'],
    ['MIDI exact', reproduction?.midi_exact ? 'Yes' : 'No'],
    ['Audio exact', reproduction?.rendered_audio_exact ? 'Yes' : 'No'],
  ] : [
    ['Candidate ID', c.id], ['Seed', c.seed], ['Renderer', c.renderer], ['Style spec', c.style],
    ['Identity grammar', c.grammar], ['Provenance bundle', 'Not emitted'],
  ];
  document.getElementById('researchProvenance').innerHTML = provenanceRows.map(([key, value]) => {
    const full = value == null ? 'Not recorded' : String(value);
    const display = key.includes('SHA-256') || key.includes('revision') ? shortHash(full) : full;
    return `<dt>${key}</dt><dd title="${full}">${display}</dd>`;
  }).join('');
  document.getElementById('researchMidi').href = `/api/midi/${c.id}`;
  document.getElementById('researchWav').href = `/api/audio/${c.id}`;
  renderResearchTimeline(analysis);
  renderResearchEvidenceWorkspaces();
  renderAnalysisAvailability(c, analysis);
  updateResearchInspector();
}
document.querySelectorAll('[data-research-view]').forEach(button => {
  button.addEventListener('click', () => {
    researchView = button.dataset.researchView;
    document.querySelectorAll('[data-research-view]').forEach(other => other.classList.toggle('active', other === button));
    renderResearch();
  });
});
document.getElementById('analysisProfile').addEventListener('change', renderResearch);
document.getElementById('researchOpenListen').addEventListener('click', () => showTab('listen'));
document.getElementById('researchOpenCreate').addEventListener('click', () => showTab('studio'));
researchTimeline.addEventListener('click', event => {
  if (!currentListenPiece || !listenAudio.duration) return;
  const rect = researchTimeline.getBoundingClientRect();
  listenAudio.currentTime = clamp((event.clientX - rect.left) / rect.width, 0, 1) * listenAudio.duration;
  updateResearchInspector(); renderResearchTimeline(currentListenPiece._performedAnalysis || currentListenAnalysis);
});
researchTimeline.addEventListener('keydown', event => {
  if (event.key === 'ArrowRight') listenAudio.currentTime = Math.min(listenAudio.duration || Infinity, listenAudio.currentTime + 5);
  if (event.key === 'ArrowLeft') listenAudio.currentTime = Math.max(0, listenAudio.currentTime - 5);
  if (event.key === ' ' || event.key === 'Enter') { event.preventDefault(); toggleListenPlayback(); }
  updateResearchInspector();
});
for (const canvasId of ['researchMotifMap', 'researchHarmonyPath', 'researchOrchestrationMap']) {
  const canvas = document.getElementById(canvasId);
  canvas.addEventListener('click', event => seekFromEvidenceCanvas(canvas, event));
  canvas.addEventListener('keydown', event => {
    if (event.key === 'ArrowRight') listenAudio.currentTime = Math.min(listenAudio.duration || Infinity, listenAudio.currentTime + 5);
    if (event.key === 'ArrowLeft') listenAudio.currentTime = Math.max(0, listenAudio.currentTime - 5);
    if (event.key === ' ' || event.key === 'Enter') { event.preventDefault(); toggleListenPlayback(); }
    updateResearchInspector(); renderResearchEvidenceWorkspaces();
  });
}


function drawFormView(ctx, w, h, progress, palette) {
  const composition = currentCompositionBundle;
  const sections = composition?.sections || [];
  if (!sections.length) {
    ctx.fillStyle = 'rgba(158,154,148,.72)';
    ctx.font = '12px Inter, sans-serif';
    ctx.fillText('No score-grounded section regions were emitted.', 20, 34);
    return;
  }
  const duration = Math.max(.001, composition.duration_seconds || 1);
  const top = h * .22, bandH = h * .44;
  sections.forEach((section, index) => {
    const x = section.start.seconds / duration * w;
    const endX = section.end.seconds / duration * w;
    const intensity = clamp(Number(section.intensity) || .8, .5, 1.25);
    ctx.fillStyle = index % 2
      ? `rgba(${palette.b},${.08 + intensity * .08})`
      : `rgba(${palette.a},${.07 + intensity * .09})`;
    ctx.fillRect(x + 1, top, Math.max(2, endX - x - 2), bandH);
    ctx.strokeStyle = `rgba(${palette.a},${.2 + intensity * .18})`;
    ctx.strokeRect(x + .5, top + .5, Math.max(1, endX - x - 1), bandH - 1);
    ctx.fillStyle = 'rgba(238,232,223,.86)';
    ctx.font = '11px Inter, sans-serif';
    const label = section.label.length > 22 ? `${section.label.slice(0, 20)}…` : section.label;
    ctx.fillText(label, x + 8, top + 22, Math.max(10, endX - x - 14));
    ctx.fillStyle = 'rgba(158,154,148,.72)';
    ctx.font = '9px Inter, sans-serif';
    ctx.fillText(`${formatTime(section.start.seconds)}–${formatTime(section.end.seconds)}`, x + 8, top + 39, Math.max(10, endX - x - 14));
  });
  for (const phrase of composition.phrases || []) {
    const x = phrase.start.seconds / duration * w;
    ctx.strokeStyle = phrase.closes_with_cadential_marker
      ? 'rgba(215,164,95,.72)'
      : 'rgba(158,154,148,.35)';
    ctx.setLineDash(phrase.closes_with_cadential_marker ? [] : [3, 4]);
    ctx.beginPath(); ctx.moveTo(x, top - 14); ctx.lineTo(x, top + bandH + 14); ctx.stroke();
  }
  ctx.setLineDash([]);
  ctx.fillStyle = 'rgba(238,232,223,.92)';
  ctx.fillRect(progress * w, top - 20, 1, bandH + 40);
}

function drawScoreView(ctx, w, h, progress, compare, palette) {
  const composition = currentCompositionBundle;
  const symbolic = composition?.notes || [];
  if (!symbolic.length) {
    ctx.fillStyle = 'rgba(158,154,148,.72)';
    ctx.font = '12px Inter, sans-serif';
    ctx.fillText('Symbolic score events are unavailable.', 20, 34);
    return;
  }
  const duration = Math.max(.001, composition.duration_seconds || 1);
  const roles = ['Melody', 'Counter melody', 'Harmony', 'Bass'].filter(role => symbolic.some(note => note.voice_role === role));
  const laneH = h / Math.max(1, roles.length);
  const performedBySource = new Map((currentPerformanceBundle?.notes || [])
    .filter(note => note.source_note_id)
    .map(note => [note.source_note_id, note]));
  roles.forEach((role, roleIndex) => {
    const y0 = roleIndex * laneH;
    ctx.fillStyle = roleIndex % 2 ? 'rgba(255,255,255,.012)' : 'rgba(255,255,255,.024)';
    ctx.fillRect(0, y0, w, laneH);
    ctx.fillStyle = 'rgba(158,154,148,.7)';
    ctx.font = '10px Inter, sans-serif';
    ctx.fillText(role, 8, y0 + 13);
    const notes = symbolic.filter(note => note.voice_role === role);
    const minMidi = Math.min(...notes.map(note => note.midi));
    const maxMidi = Math.max(...notes.map(note => note.midi));
    for (const note of notes) {
      const x = note.onset.seconds / duration * w;
      const width = Math.max(2, note.duration_seconds / duration * w);
      const pitchFrac = maxMidi === minMidi ? .5 : (maxMidi - note.midi) / (maxMidi - minMidi);
      const y = y0 + 20 + pitchFrac * Math.max(4, laneH - 30);
      ctx.fillStyle = note.emphasis === 'climax'
        ? `rgba(${palette.a},.9)`
        : note.emphasis === 'cadential'
          ? 'rgba(215,164,95,.72)'
          : `rgba(${palette.b},.46)`;
      ctx.fillRect(x, y, width, 5);
      if (compare) {
        const performed = performedBySource.get(note.id);
        if (performed) {
          const px = performed.start_seconds / duration * w;
          const pw = Math.max(2, performed.duration_seconds / duration * w);
          ctx.strokeStyle = 'rgba(102,174,190,.92)';
          ctx.lineWidth = 1.2;
          ctx.strokeRect(px, y - 2, pw, 9);
          if (Math.abs(px - x) > .8) {
            ctx.strokeStyle = 'rgba(102,174,190,.38)';
            ctx.beginPath(); ctx.moveTo(x, y + 8); ctx.lineTo(px, y + 8); ctx.stroke();
          }
        }
      }
    }
  });
  ctx.fillStyle = 'rgba(238,232,223,.92)';
  ctx.fillRect(progress * w, 0, 1, h);
}


function drawMotifView(ctx, w, h, progress, palette) {
  const composition = currentCompositionBundle;
  const occurrences = composition?.motif_occurrences || [];
  const definition = composition?.motif_definitions?.[0];
  if (!occurrences.length) {
    ctx.fillStyle = 'rgba(158,154,148,.72)';
    ctx.font = '12px Inter, sans-serif';
    ctx.fillText('No motif occurrence met the conservative score-side threshold.', 20, 34);
    return;
  }
  const duration = Math.max(.001, composition.duration_seconds || 1);
  ctx.fillStyle = 'rgba(238,232,223,.9)';
  ctx.font = '12px Inter, sans-serif';
  const degrees = definition?.degrees?.length ? definition.degrees.join(' · ') : 'definition unavailable';
  ctx.fillText(`${definition?.label || 'Primary motif'} · degrees ${degrees}`, 18, 28);
  ctx.fillStyle = 'rgba(158,154,148,.7)';
  ctx.font = '10px Inter, sans-serif';
  ctx.fillText('Occurrence relationships are symbolic analysis; select one to hear it in context.', 18, 46);
  const top = 74;
  const bandH = Math.max(80, h - 118);
  for (const section of composition.sections || []) {
    const x = section.start.seconds / duration * w;
    const x2 = section.end.seconds / duration * w;
    ctx.fillStyle = 'rgba(255,255,255,.018)';
    ctx.fillRect(x, top, Math.max(1, x2 - x), bandH);
    ctx.strokeStyle = 'rgba(255,255,255,.06)';
    ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, top + bandH); ctx.stroke();
  }
  occurrences.forEach((occurrence, index) => {
    const x = occurrence.start.seconds / duration * w;
    const x2 = occurrence.end.seconds / duration * w;
    const y = top + 20 + (index % 3) * 42;
    const alpha = .3 + clamp(Number(occurrence.similarity) || 0, 0, 1) * .55;
    ctx.fillStyle = `rgba(125,100,200,${alpha})`;
    ctx.fillRect(x, y, Math.max(4, x2 - x), 18);
    ctx.strokeStyle = `rgba(${palette.a},.58)`;
    ctx.strokeRect(x + .5, y + .5, Math.max(3, x2 - x - 1), 17);
    ctx.fillStyle = 'rgba(238,232,223,.88)';
    ctx.font = '9px Inter, sans-serif';
    const label = `${occurrence.transformation} · ${Math.round((occurrence.similarity || 0) * 100)}%`;
    ctx.fillText(label, x + 5, y + 12, Math.max(20, x2 - x - 9));
  });
  ctx.fillStyle = 'rgba(238,232,223,.92)';
  ctx.fillRect(progress * w, top - 8, 1, bandH + 16);
}

function drawHarmonyView(ctx, w, h, progress, palette) {
  const composition = currentCompositionBundle;
  const regions = composition?.sonorities || [];
  if (!regions.length) {
    ctx.fillStyle = 'rgba(158,154,148,.72)';
    ctx.font = '12px Inter, sans-serif';
    ctx.fillText('No score-side sonority regions were emitted.', 20, 34);
    return;
  }
  const duration = Math.max(.001, composition.duration_seconds || 1);
  const top = h * .24;
  const bandH = h * .38;
  regions.forEach((region, index) => {
    const x = region.start.seconds / duration * w;
    const x2 = region.end.seconds / duration * w;
    const exact = Boolean(region.home_key_degree);
    ctx.fillStyle = exact
      ? `rgba(${palette.a},${.12 + (index % 2) * .04})`
      : 'rgba(125,100,200,.13)';
    ctx.fillRect(x + .5, top, Math.max(1, x2 - x - 1), bandH);
    ctx.strokeStyle = exact ? `rgba(${palette.a},.42)` : 'rgba(125,100,200,.35)';
    ctx.strokeRect(x + .5, top + .5, Math.max(1, x2 - x - 1), bandH - 1);
    if (x2 - x > 34) {
      ctx.fillStyle = 'rgba(238,232,223,.86)';
      ctx.font = '10px Inter, sans-serif';
      ctx.fillText(region.pitch_classes.join('–'), x + 5, top + 18, Math.max(10, x2 - x - 9));
      if (exact) {
        ctx.fillStyle = 'rgba(215,164,95,.78)';
        ctx.font = '9px Inter, sans-serif';
        ctx.fillText(`degree ${region.home_key_degree} · ${region.home_key_function || 'home-key triad'}`, x + 5, top + 34, Math.max(10, x2 - x - 9));
      }
    }
  });
  for (const cadence of composition.cadences || []) {
    const x = cadence.at.seconds / duration * w;
    ctx.strokeStyle = 'rgba(238,232,223,.72)';
    ctx.lineWidth = 1.4;
    ctx.beginPath(); ctx.moveTo(x, top - 18); ctx.lineTo(x, top + bandH + 18); ctx.stroke();
    ctx.fillStyle = 'rgba(238,232,223,.82)';
    ctx.font = '9px Inter, sans-serif';
    ctx.fillText(`cadence · ${cadence.arrival_pitch_name}`, x + 4, top - 7);
  }
  ctx.fillStyle = 'rgba(158,154,148,.72)';
  ctx.font = '10px Inter, sans-serif';
  ctx.fillText('Degree labels appear only for exact triads in the declared home key; no modulation is inferred.', 18, h - 22);
  ctx.fillStyle = 'rgba(238,232,223,.92)';
  ctx.fillRect(progress * w, top - 24, 1, bandH + 48);
}

function drawOrchestrationView(ctx, w, h, progress, palette) {
  const composition = currentCompositionBundle;
  const regions = composition?.orchestration || [];
  if (!regions.length) {
    ctx.fillStyle = 'rgba(158,154,148,.72)';
    ctx.font = '12px Inter, sans-serif';
    ctx.fillText('No composition-side voice assignment regions were emitted.', 20, 34);
    return;
  }
  const duration = Math.max(.001, composition.duration_seconds || 1);
  const roles = ['Melody', 'Counter melody', 'Harmony', 'Bass'].filter(role =>
    regions.some(region => region.active_voices?.some(voice => voice.voice_role === role)));
  const top = 30;
  const laneH = Math.max(34, (h - top - 28) / Math.max(1, roles.length));
  roles.forEach((role, laneIndex) => {
    const y = top + laneIndex * laneH;
    ctx.fillStyle = laneIndex % 2 ? 'rgba(255,255,255,.012)' : 'rgba(255,255,255,.024)';
    ctx.fillRect(0, y, w, laneH);
    ctx.fillStyle = 'rgba(176,171,164,.72)';
    ctx.font = '10px Inter, sans-serif';
    ctx.fillText(role, 8, y + 15);
    for (const region of regions) {
      const voice = region.active_voices?.find(item => item.voice_role === role);
      if (!voice) continue;
      const x = region.start.seconds / duration * w;
      const x2 = region.end.seconds / duration * w;
      const share = voice.note_count / Math.max(1, region.active_voices.reduce((sum, item) => sum + item.note_count, 0));
      ctx.fillStyle = role === 'Melody'
        ? `rgba(${palette.a},${.24 + share * .55})`
        : role === 'Bass'
          ? 'rgba(102,174,190,.42)'
          : `rgba(${palette.b},${.18 + share * .42})`;
      ctx.fillRect(x, y + 20, Math.max(2, x2 - x), Math.max(5, laneH - 27));
    }
  });
  ctx.fillStyle = 'rgba(158,154,148,.72)';
  ctx.font = '10px Inter, sans-serif';
  ctx.fillText('Assigned symbolic voices by section · not rendered prominence.', 18, h - 10);
  ctx.fillStyle = 'rgba(238,232,223,.92)';
  ctx.fillRect(progress * w, top - 6, 1, h - top - 12);
}

function drawHybridStructure(ctx, cx, cy, radius, duration, palette) {
  const sections = currentCompositionBundle?.sections || [];
  for (const section of sections) {
    const start = -Math.PI / 2 + section.start.seconds / duration * Math.PI * 2;
    const end = -Math.PI / 2 + section.end.seconds / duration * Math.PI * 2;
    const intensity = clamp(Number(section.intensity) || .8, .5, 1.25);
    ctx.strokeStyle = `rgba(${palette.a},${.22 + intensity * .35})`;
    ctx.lineWidth = 8;
    ctx.beginPath(); ctx.arc(cx, cy, radius, start + .006, end - .006); ctx.stroke();
  }
  for (const phrase of currentCompositionBundle?.phrases || []) {
    const angle = -Math.PI / 2 + phrase.start.seconds / duration * Math.PI * 2;
    const inner = radius - 7, outer = radius + (phrase.closes_with_cadential_marker ? 8 : 4);
    ctx.strokeStyle = phrase.closes_with_cadential_marker
      ? 'rgba(238,232,223,.68)'
      : 'rgba(158,154,148,.3)';
    ctx.lineWidth = phrase.closes_with_cadential_marker ? 1.6 : 1;
    ctx.beginPath();
    ctx.moveTo(cx + Math.cos(angle) * inner, cy + Math.sin(angle) * inner);
    ctx.lineTo(cx + Math.cos(angle) * outer, cy + Math.sin(angle) * outer);
    ctx.stroke();
  }
  for (const occurrence of currentCompositionBundle?.motif_occurrences || []) {
    const angle = -Math.PI / 2 + occurrence.start.seconds / duration * Math.PI * 2;
    const markerRadius = radius - 15;
    ctx.fillStyle = `rgba(125,100,200,${.35 + clamp(Number(occurrence.similarity) || 0, 0, 1) * .58})`;
    ctx.beginPath();
    ctx.arc(cx + Math.cos(angle) * markerRadius, cy + Math.sin(angle) * markerRadius, 2.8, 0, Math.PI * 2);
    ctx.fill();
  }
  for (const cadence of currentCompositionBundle?.cadences || []) {
    const angle = -Math.PI / 2 + cadence.at.seconds / duration * Math.PI * 2;
    ctx.strokeStyle = 'rgba(238,232,223,.78)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(cx + Math.cos(angle) * (radius + 5), cy + Math.sin(angle) * (radius + 5));
    ctx.lineTo(cx + Math.cos(angle) * (radius + 13), cy + Math.sin(angle) * (radius + 13));
    ctx.stroke();
  }
}

function drawViz() {
  cancelAnimationFrame(vizRaf);
  const canvas = document.getElementById('viz');
  const ctx = canvas.getContext('2d');
  const bins = new Uint8Array(256);
  const wave = new Uint8Array(512);
  let t = 0, lastFrame = 0;
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  function frame(now = 0) {
    vizRaf = requestAnimationFrame(frame);
    if (reduceMotion && now - lastFrame < 100) return;
    lastFrame = now;
    t += reduceMotion ? 0 : 0.008;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const cssW = Math.max(1, canvas.clientWidth), cssH = Math.max(1, canvas.clientHeight);
    if (canvas.width !== Math.round(cssW * dpr) || canvas.height !== Math.round(cssH * dpr)) {
      canvas.width = Math.round(cssW * dpr);
      canvas.height = Math.round(cssH * dpr);
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const w = cssW, h = cssH;
    ctx.clearRect(0, 0, w, h);
    const P = currentPalette;
    let bass = 0.12, mids = [];
    const live = analyser && !listenAudio.paused;
    if (live) {
      analyser.getByteFrequencyData(bins);
      bass = bins.slice(1, 8).reduce((a, b) => a + b, 0) / (7 * 255);
      for (let i = 0; i < 72; i++) mids.push(bins[8 + i * 2] / 255);
    } else {
      const density = currentListenAnalysis?.density || [];
      for (let i = 0; i < 72; i++) mids.push(density[i % Math.max(1, density.length)] || 0.035);
      bass = density.length ? density[Math.floor((listenAudio.currentTime / Math.max(.001, listenAudio.duration || 1)) * density.length) % density.length] * .32 : .08;
    }
    const progress = listenAudio.duration ? clamp(listenAudio.currentTime / listenAudio.duration, 0, 1) : 0;

    if (vizMode === 'form') {
      drawFormView(ctx, w, h, progress, P);
    } else if (vizMode === 'score' || vizMode === 'compare') {
      drawScoreView(ctx, w, h, progress, vizMode === 'compare', P);
    } else if (vizMode === 'motifs') {
      drawMotifView(ctx, w, h, progress, P);
    } else if (vizMode === 'harmony') {
      drawHarmonyView(ctx, w, h, progress, P);
    } else if (vizMode === 'orchestration') {
      drawOrchestrationView(ctx, w, h, progress, P);
    } else if (vizMode === 'resonance') {
      if (live) analyser.getByteTimeDomainData(wave);
      const structural = currentCompositionBundle?.resonance?.samples || [];
      const density = structural.length
        ? structural.map(sample => clamp(Number(sample.energy) || 0, 0, 1))
        : (currentListenAnalysis?.density || []);
      for (let pass = 0; pass < 3; pass++) {
        ctx.strokeStyle = `rgba(${pass === 1 ? P.a : P.b}, ${0.48 - pass * 0.13})`;
        ctx.lineWidth = 2.4 - pass * .55;
        ctx.beginPath();
        const n = live ? wave.length : Math.max(64, density.length);
        for (let i = 0; i < n; i++) {
          const value = live ? (wave[i] - 128) / 128
                             : ((density[i % Math.max(1, density.length)] || .06) - .35) * .8;
          const x = i / Math.max(1, n - 1) * w;
          const envelope = Math.sin(Math.PI * i / Math.max(1, n - 1));
          const y = h * .53 + value * h * .34 * envelope + Math.sin(i * .08 + pass * .7) * 2;
          if (!i) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.stroke();
      }
      ctx.strokeStyle = 'rgba(125,100,200,.52)';
      ctx.beginPath(); ctx.moveTo(progress * w, 16); ctx.lineTo(progress * w, h - 16); ctx.stroke();
    } else {
      const cx = w / 2, cy = h / 2;
      const baseRadius = Math.min(w, h) * .17;
      const outerRadius = Math.min(w, h) * .38;
      const structuralRadius = outerRadius - 12;
      const glowRadius = baseRadius * (1.6 + bass * .8);
      const glow = ctx.createRadialGradient(cx, cy, 2, cx, cy, glowRadius);
      glow.addColorStop(0, `rgba(${P.a},${.48 + bass * .4})`);
      glow.addColorStop(.35, `rgba(${P.b},.17)`);
      glow.addColorStop(1, `rgba(${P.b},0)`);
      ctx.fillStyle = glow;
      ctx.beginPath(); ctx.arc(cx, cy, glowRadius, 0, Math.PI * 2); ctx.fill();

      drawHybridStructure(ctx, cx, cy, structuralRadius, Math.max(.001, currentCompositionBundle?.duration_seconds || currentListenAnalysis?.duration || listenAudio.duration || 1), P);

      // Stable structural rings: musical events never rotate. Motion only
      // changes luminance/length, so the map remains inspectable while playing.
      for (let ring = 0; ring < 4; ring++) {
        ctx.strokeStyle = `rgba(215,164,95,${.07 + ring * .025})`;
        ctx.lineWidth = 1;
        ctx.beginPath(); ctx.arc(cx, cy, baseRadius + 38 + ring * 28, 0, Math.PI * 2); ctx.stroke();
      }
      const events = currentListenAnalysis?.events || [];
      const duration = currentListenAnalysis?.duration || listenAudio.duration || 1;
      const stride = Math.max(1, Math.ceil(events.length / 520));
      for (let i = 0; i < events.length; i += stride) {
        const event = events[i];
        const a = -Math.PI / 2 + event.start / Math.max(.001, duration) * Math.PI * 2;
        const lane = event.voiceIndex % 5;
        const r1 = baseRadius + 34 + lane * 18;
        const length = 4 + Math.min(1, event.velocity) * 12 + Math.min(.8, event.duration / 2) * 7;
        ctx.strokeStyle = lane === 0 ? `rgba(${P.a},.74)` : lane === 3 ? 'rgba(102,174,190,.55)' : `rgba(${P.b},.42)`;
        ctx.lineWidth = 1 + event.velocity * 1.1;
        ctx.beginPath();
        ctx.moveTo(cx + Math.cos(a) * r1, cy + Math.sin(a) * r1);
        ctx.lineTo(cx + Math.cos(a) * (r1 + length), cy + Math.sin(a) * (r1 + length));
        ctx.stroke();
      }
      if (!events.length) {
        for (let i = 0; i < 96; i++) {
          const a = -Math.PI / 2 + i / 96 * Math.PI * 2;
          const value = mids[i % mids.length];
          const r1 = baseRadius + 45;
          ctx.strokeStyle = `rgba(${P.b},${.09 + value * .42})`;
          ctx.lineWidth = 1.2;
          ctx.beginPath();
          ctx.moveTo(cx + Math.cos(a) * r1, cy + Math.sin(a) * r1);
          ctx.lineTo(cx + Math.cos(a) * (r1 + 7 + value * 25), cy + Math.sin(a) * (r1 + 7 + value * 25));
          ctx.stroke();
        }
      }
      ctx.strokeStyle = 'rgba(255,255,255,.075)';
      ctx.lineWidth = 9;
      ctx.beginPath(); ctx.arc(cx, cy, outerRadius, -Math.PI / 2, Math.PI * 1.5); ctx.stroke();
      ctx.strokeStyle = `rgba(${P.a},.8)`;
      ctx.lineWidth = 3;
      ctx.beginPath(); ctx.arc(cx, cy, outerRadius, -Math.PI / 2, -Math.PI / 2 + progress * Math.PI * 2); ctx.stroke();
      const playAngle = -Math.PI / 2 + progress * Math.PI * 2;
      ctx.fillStyle = `rgba(${P.a},.95)`;
      ctx.beginPath(); ctx.arc(cx + Math.cos(playAngle) * outerRadius, cy + Math.sin(playAngle) * outerRadius, 3.2, 0, Math.PI * 2); ctx.fill();
    }
  }
  frame();
}

// ── Style-reactive palettes: the room changes with the music ────────────
const STYLE_PALETTES = {
  Classical:   { a: '232,196,138', b: '217,160,91',  bg: ['#241e16', '#16130f'] },
  Nocturne:    { a: '158,150,224', b: '104,96,190',  bg: ['#191a2e', '#0f1018'] },
  Tango:       { a: '235,110,97',  b: '178,49,58',   bg: ['#2a1414', '#140d0d'] },
  Cinematic:   { a: '126,196,207', b: '62,130,148',  bg: ['#12222a', '#0b1418'] },
  Passacaglia: { a: '150,196,132', b: '84,132,84',   bg: ['#16231a', '#0e150f'] },
  Lullaby:     { a: '238,178,189', b: '196,116,138', bg: ['#271a20', '#151013'] },
  Folk:        { a: '206,196,120', b: '148,132,66',  bg: ['#22200f', '#14130a'] },
  Waltz:       { a: '240,214,170', b: '196,164,110', bg: ['#262016', '#15120d'] },
  Fugue:       { a: '168,180,198', b: '104,120,146', bg: ['#181c24', '#0f1115'] },
  ModalFolk:   { a: '186,160,210', b: '124,96,152',  bg: ['#201a28', '#121017'] },
  March:       { a: '224,178,96',  b: '166,116,44',  bg: ['#241c10', '#14100a'] },
  Playful:     { a: '240,150,120', b: '196,96,80',   bg: ['#281a14', '#15100c'] },
  Celtic:      { a: '120,186,158', b: '58,124,102',  bg: ['#132420', '#0b1613'] },
  Blues:       { a: '98,140,210',  b: '46,74,150',   bg: ['#101a2e', '#0a1019'] },
  Impressionism: { a: '196,186,224', b: '140,128,180', bg: ['#1c1826', '#100e16'] },
  SacredChoral: { a: '236,222,182', b: '196,164,96',  bg: ['#221c12', '#14100a'] },
  Minimalism:  { a: '120,224,210', b: '52,158,148',   bg: ['#0e201c', '#081210'] },
  JazzBallad:  { a: '212,142,120', b: '150,78,66',    bg: ['#241511', '#150c0a'] },
  BaroqueSuite: { a: '196,120,132', b: '138,62,74',   bg: ['#241318', '#140b0d'] },
  ProgFolk:    { a: '150,90,230',  b: '92,48,168',    bg: ['#180f28', '#0d0916'] },
  Ambient:     { a: '168,188,196', b: '104,124,134',  bg: ['#12181c', '#0a0d0f'] },
  Sonata:      { a: '120,140,180', b: '70,86,124',    bg: ['#11151f', '#090b12'] },
  RenaissancePolyphony: { a: '158,148,96', b: '108,100,62', bg: ['#181610', '#0d0c08'] },
  AfroCuban:   { a: '232,142,64',  b: '186,84,36',    bg: ['#251507', '#150c05'] },
  Flamenco:    { a: '214,54,58',   b: '138,26,32',    bg: ['#210a0b', '#140606'] },
  BossaNova:   { a: '92,182,168',  b: '48,124,114',    bg: ['#0c1c19', '#07110f'] },
  Opera:       { a: '176,96,182',  b: '112,52,120',    bg: ['#180c1c', '#0e0710'] },
  IrishTraditional: { a: '110,184,96', b: '62,128,52',  bg: ['#0e1c0c', '#081007'] },
  HindustaniInspired: { a: '224,168,72', b: '164,116,40', bg: ['#201708', '#130e05'] },
};
let currentPalette = STYLE_PALETTES.Classical;
function applyPalette(style) {
  currentPalette = STYLE_PALETTES[style] || STYLE_PALETTES.Classical;
  const viz = document.getElementById('viz');
  viz.style.background =
    `radial-gradient(ellipse at 50% 62%, ${currentPalette.bg[0]}, ${currentPalette.bg[1]} 78%)`;
}
// The loader's orb wears the style being composed.
function applyOrbPalette(style) {
  const p = STYLE_PALETTES[style] || STYLE_PALETTES.Classical;
  const orb = document.querySelector('#loading .orb');
  if (!orb) return;
  orb.style.background =
    `radial-gradient(circle at 38% 34%, rgb(${p.a}), rgb(${p.b}) 55%, #241a10 100%)`;
  orb.style.boxShadow = `0 0 44px 6px rgba(${p.b}, 0.28)`;
  document.getElementById('loading').style.background =
    `radial-gradient(ellipse at 50% 45%, ${p.bg[0]}55, transparent 70%)`;
}

// ── Visualization modes ─────────────────────────────────────────────────
let vizMode = 'hybrid';
document.querySelectorAll('#vizModes .chip').forEach((b) => {
  b.addEventListener('click', () => {
    if (b.disabled) return;
    document.querySelectorAll('#vizModes .chip').forEach((x) => x.classList.remove('sel'));
    b.classList.add('sel');
    vizMode = b.dataset.mode;
  });
});
