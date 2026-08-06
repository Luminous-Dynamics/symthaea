// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! The missing middle in Muse's evidence: a paired, level-matched, blind A/B
//! listening pack you can run in twenty minutes.
//!
//! Muse has had a barbell problem. At one end, informal "I rendered two seeds
//! and preferred one." At the other, a ~26K-line preregistered confirmatory
//! apparatus with 120 subcommands (`cognitive_study`), which has never been run
//! on real listeners. Nothing in between — so every audible change stalls on
//! "needs a listening check" and then doesn't get one. The Baroque campaign
//! closed on exactly that: it built the A/B *renders*, then stopped, naming
//! what was missing as "the loudness-normalization/blinding/sealing step"
//! (`docs/research/evidence/baroque-harmonic-campaign-2026-07-28/CAMPAIGN_CLOSED.md`).
//!
//! This is that step, and nothing more. It is deliberately NOT a study: no
//! preregistration, no cohort registry, no sealed amendments ledger. It is the
//! cheap loop you run before deciding whether a change was an improvement.
//!
//! # What it does
//!
//! `build` takes a directory of already-rendered pairs — the layout
//! `examples/baroque_campaign.rs` already writes —
//!
//! ```text
//! pairs_dir/<case>/<variant>/piece.wav
//! ```
//!
//! and for each case emits ONE trial: the two variants, level-matched, in a
//! randomized left/right order, under opaque clip names. The answer key is
//! written OUTSIDE the pack directory so the pack can be handed to a listener
//! as-is.
//!
//! Paired by construction: each trial compares two renders of the SAME case,
//! which is what makes a handful of listeners informative at all.
//!
//! # Level matching, not mastering
//!
//! Clips are gain-matched **pairwise** to one shared target both members can
//! reach, using this crate's BS.1770 K-weighted [`measure_lufs`], and the
//! result is then REMEASURED and rejected if the rendered pair differs by more
//! than [`MAX_PAIR_DELTA_LU`].
//!
//! That verification is the whole point, and v1 did not have it. v1 normalized
//! each clip independently toward the target and backed off on peak — so a clip
//! that hit the ceiling landed quieter than its partner while both were
//! reported as "level-matched." In the 32-clip Baroque pack built that way,
//! **22 clips sat at the peak ceiling**: most of the pack was capped rather
//! than matched, and the tool's central claim was false for it.
//!
//! It deliberately does NOT run [`auto_master`](symthaea_muse::auto_master):
//! that applies corrective EQ and compression, which would partly mask the
//! difference under test. Loudness is the one confound that reliably decides
//! blind preference on its own, so it is the one thing to remove — and only it.
//!
//! # Honest limits
//!
//! - **Decisions are not independent.** Judgments from one listener share their
//!   taste and fatigue; judgments of one case share that case's character. So
//!   `score` reports a descriptive breakdown by case and by listener, and the
//!   exact binomial is printed only as an explicitly-labelled naive statistic
//!   whose independence assumption does not hold here. Do not quote it as a
//!   result.
//! - Listeners are not blinded to each other, there is no attention check, and
//!   nothing is sealed. If you need evidence that survives review, that is what
//!   `cognitive_study` is for.
//! - Order within a trial is randomized; trial order is not. Fatigue effects
//!   are not controlled.
//!
//! # Usage
//!
//! ```bash
//! cargo run --release -p symthaea-muse --features theory --bin ab_listening_pack -- \
//!     build audio_output/baroque_campaign_2026-07-27 /tmp/pack 20260729
//! # hand /tmp/pack to listeners; collect /tmp/pack/responses.json
//! cargo run ... --bin ab_listening_pack -- score /tmp/pack.key.json /tmp/pack/responses.json
//! ```

use std::collections::BTreeMap;
use std::error::Error;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use symthaea_muse::auto_master::measure_lufs;

/// Streaming-standard integrated loudness. The exact value matters far less
/// than every clip sharing it.
const TARGET_LUFS: f32 = -16.0;
/// Leave headroom so gain-matching cannot push a peak into the ceiling.
const PEAK_CEILING: f32 = 0.891; // ≈ -1 dBFS
/// Seconds of each render presented per trial.
///
/// Full pieces defeat the purpose: the Baroque campaign's renders are ~2.3
/// minutes each, so 16 trials of two full clips is 74 minutes of listening —
/// which is how a "quick check" becomes something nobody does. At 45s a
/// 16-trial pack is ~24 minutes of audio.
///
/// The honest cost: a head excerpt cannot show late-piece structure. If the
/// change under test is about recapitulation or long-range return, raise this
/// or excerpt around the moment of interest — a null result on 45s of exposition
/// says nothing about a difference that only appears at the return.
const DEFAULT_EXCERPT_SECS: f32 = 45.0;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Trial {
    case: String,
    /// Clip filenames as the listener sees them, in presentation order.
    left_clip: String,
    right_clip: String,
    /// Which variant each side actually was. Present only in the key.
    left_variant: String,
    right_variant: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AnswerKey {
    pack_version: String,
    seed: u64,
    target_lufs: f32,
    excerpt_secs: f32,
    matcher: String,
    max_pair_delta_lu: f32,
    variants: Vec<String>,
    trials: Vec<Trial>,
    /// Per-clip measurement provenance: the evidence that this pack's
    /// level-matching claim is true, rather than an assertion about it.
    clips: Vec<ClipRecord>,
}

/// Deterministic, seed-driven bit source. Not cryptographic — it only decides
/// which side a variant lands on, and the seed is recorded in the key so a
/// pack can be rebuilt exactly.
struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        // SplitMix64.
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn flip(&mut self) -> bool {
        // Take a HIGH bit: low bits of LCG-family generators are weak.
        (self.next_u64() >> 63) == 1
    }
}

fn read_wav(path: &Path) -> Result<(Vec<[f32; 2]>, u32), Box<dyn Error>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let raw: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => {
            let scale = 1.0 / (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 * scale))
                .collect::<Result<_, _>>()?
        }
    };
    let frames = match spec.channels {
        1 => raw.iter().map(|&s| [s, s]).collect(),
        2 => raw.chunks_exact(2).map(|c| [c[0], c[1]]).collect(),
        n => return Err(format!("{}: unsupported channel count {n}", path.display()).into()),
    };
    Ok((frames, spec.sample_rate))
}

/// Version stamp recorded in every manifest. Bump when the algorithm changes,
/// so a pack's provenance says which matcher produced it.
const MATCHER: &str = "pairwise-shared-target-v2";

/// A pair whose two clips differ by more than this after rendering is rejected.
/// Roughly the threshold below which loudness stops driving blind preference.
const MAX_PAIR_DELTA_LU: f32 = 0.2;

/// What a clip measured before and after matching. Recorded per clip so a
/// pack's central claim — "these two are level-matched" — is auditable from
/// the manifest instead of taken on trust.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ClipRecord {
    source: String,
    source_sha256: String,
    output: String,
    output_sha256: String,
    sample_rate: u32,
    excerpt_secs: f32,
    initial_lufs: f32,
    initial_peak: f32,
    requested_target_lufs: f32,
    pair_target_lufs: f32,
    applied_gain_db: f32,
    final_lufs: f32,
    final_peak: f32,
    matcher: String,
}

fn measured(frames: &[[f32; 2]], sample_rate: u32) -> (f32, f32) {
    let lufs = measure_lufs(frames, sample_rate).integrated;
    let peak = frames
        .iter()
        .flat_map(|f| [f[0].abs(), f[1].abs()])
        .fold(0.0f32, f32::max);
    (lufs, peak)
}

/// The loudest this clip can be made without its peak exceeding the ceiling.
fn safe_target_lufs(lufs: f32, peak: f32) -> f32 {
    if peak <= f32::EPSILON {
        return f32::NEG_INFINITY;
    }
    lufs + 20.0 * (PEAK_CEILING / peak).log10()
}

/// Match a PAIR to one shared target, then verify the result.
///
/// The v1 algorithm normalized each clip independently toward `TARGET_LUFS`
/// and backed off if its peak would clip. That silently defeated the whole
/// point: a clip that hit the ceiling landed BELOW target while its partner
/// reached it, so a "level-matched" pair could differ audibly. It was not a
/// theoretical hazard — in the 32-clip Baroque pack built with v1, **22 clips
/// sat at the peak ceiling**, i.e. most of them were capped rather than matched.
///
/// v2 takes the target both members can actually reach —
/// `min(requested, safe_a, safe_b)` — applies it to each from its ORIGINAL
/// excerpt (never re-normalizing already-gained audio, so repeated runs cannot
/// accumulate error), then REMEASURES and fails closed if the rendered
/// difference exceeds `MAX_PAIR_DELTA_LU`.
fn match_pair(
    a: &mut [[f32; 2]],
    b: &mut [[f32; 2]],
    sample_rate: u32,
) -> Result<(f32, [f32; 2], [f32; 2], [f32; 2]), String> {
    let (la, pa) = measured(a, sample_rate);
    let (lb, pb) = measured(b, sample_rate);
    for (l, tag) in [(la, "A"), (lb, "B")] {
        if !l.is_finite() || l <= -70.0 {
            return Err(format!("clip {tag} is silent or unmeasurable ({l} LUFS)"));
        }
    }

    let target = TARGET_LUFS
        .min(safe_target_lufs(la, pa))
        .min(safe_target_lufs(lb, pb));

    let apply = |frames: &mut [[f32; 2]], lufs: f32| -> f32 {
        let gain = 10f32.powf((target - lufs) / 20.0);
        for f in frames.iter_mut() {
            f[0] *= gain;
            f[1] *= gain;
        }
        20.0 * gain.log10()
    };
    let ga = apply(a, la);
    let gb = apply(b, lb);

    let (fa, fpa) = measured(a, sample_rate);
    let (fb, fpb) = measured(b, sample_rate);

    let delta = (fa - fb).abs();
    if delta > MAX_PAIR_DELTA_LU {
        return Err(format!(
            "pair differs by {delta:.3} LU after matching (limit {MAX_PAIR_DELTA_LU}); \
             A {fa:.2} LUFS, B {fb:.2} LUFS"
        ));
    }
    for (p, tag) in [(fpa, "A"), (fpb, "B")] {
        if p > PEAK_CEILING + 1e-4 {
            return Err(format!(
                "clip {tag} peaks at {p:.4}, above ceiling {PEAK_CEILING}"
            ));
        }
    }
    if a.iter()
        .chain(b.iter())
        .any(|f| !f[0].is_finite() || !f[1].is_finite())
    {
        return Err("non-finite sample after gain".into());
    }

    Ok((target, [la, lb], [ga, gb], [fa, fb]))
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(bytes);
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

fn write_wav(path: &Path, frames: &[[f32; 2]], sample_rate: u32) -> Result<(), Box<dyn Error>> {
    let spec = hound::WavSpec {
        channels: 2,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create(path, spec)?;
    for f in frames {
        for &s in f {
            w.write_sample((s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)?;
        }
    }
    w.finalize()?;
    Ok(())
}

/// `<case>/<variant>/piece.wav` for every case that has EXACTLY the two
/// variants. Cases with a missing render are skipped and reported — a pair
/// with one side absent would silently become an unpaired comparison.
fn discover(
    pairs_dir: &Path,
) -> Result<(Vec<String>, BTreeMap<String, Vec<PathBuf>>), Box<dyn Error>> {
    let mut variants: Vec<String> = Vec::new();
    let mut cases: BTreeMap<String, Vec<PathBuf>> = BTreeMap::new();
    let mut dirs: Vec<_> = std::fs::read_dir(pairs_dir)?
        .filter_map(Result::ok)
        .filter(|e| e.path().is_dir())
        .map(|e| e.path())
        .collect();
    dirs.sort();
    for case_dir in dirs {
        let case = case_dir.file_name().unwrap().to_string_lossy().to_string();
        let mut vdirs: Vec<_> = std::fs::read_dir(&case_dir)?
            .filter_map(Result::ok)
            .filter(|e| e.path().is_dir())
            .map(|e| e.path())
            .collect();
        vdirs.sort();
        for vdir in vdirs {
            let vname = vdir.file_name().unwrap().to_string_lossy().to_string();
            let wav = vdir.join("piece.wav");
            if !wav.is_file() {
                continue;
            }
            if !variants.contains(&vname) {
                variants.push(vname.clone());
            }
            cases.entry(case.clone()).or_default().push(wav);
        }
    }
    Ok((variants, cases))
}

/// Short cosine fade so a truncated excerpt ends rather than being cut off —
/// a hard edit click is an obvious artefact that would itself sway preference.
fn fade_out(frames: &mut [[f32; 2]], sample_rate: u32) {
    let n = ((sample_rate as f32 * 0.75) as usize).min(frames.len());
    let start = frames.len() - n;
    for (i, f) in frames[start..].iter_mut().enumerate() {
        let t = i as f32 / n.max(1) as f32;
        let g = 0.5 * (1.0 + (std::f32::consts::PI * t).cos());
        f[0] *= g;
        f[1] *= g;
    }
}

fn build(
    pairs_dir: &Path,
    out_dir: &Path,
    seed: u64,
    excerpt_secs: f32,
) -> Result<(), Box<dyn Error>> {
    let (variants, cases) = discover(pairs_dir)?;
    if variants.len() != 2 {
        return Err(format!(
            "expected exactly 2 variants under {}, found {}: {:?}",
            pairs_dir.display(),
            variants.len(),
            variants
        )
        .into());
    }
    std::fs::create_dir_all(out_dir)?;
    let mut rng = Rng(seed);
    let mut trials = Vec::new();
    let mut skipped = Vec::new();
    let mut clip_records: Vec<ClipRecord> = Vec::new();

    for (case, wavs) in &cases {
        if wavs.len() != 2 {
            skipped.push(case.clone());
            continue;
        }
        // `discover` pushes variants in sorted order, so wavs[i] belongs to
        // the variant whose directory sorted i-th within this case.
        let (a, b) = (&wavs[0], &wavs[1]);
        let (av, bv) = (&variants[0], &variants[1]);
        let swap = rng.flip();
        let (left_src, right_src, left_variant, right_variant) = if swap {
            (b, a, bv.clone(), av.clone())
        } else {
            (a, b, av.clone(), bv.clone())
        };
        let idx = trials.len() + 1;
        let left_clip = format!("trial{idx:02}_A.wav");
        let right_clip = format!("trial{idx:02}_B.wav");

        // Excerpt BEFORE matching, so gain is set on what the listener actually
        // hears rather than on a quiet tail they never reach. Both members are
        // read and excerpted first, because the target depends on BOTH.
        let mut excerpt = |src: &PathBuf| -> Result<(Vec<[f32; 2]>, u32), Box<dyn Error>> {
            let (mut frames, sr) = read_wav(src)?;
            let keep = (excerpt_secs * sr as f32) as usize;
            if keep > 0 && frames.len() > keep {
                frames.truncate(keep);
                fade_out(&mut frames, sr);
            }
            Ok((frames, sr))
        };
        let (mut la_frames, sr_a) = excerpt(left_src)?;
        let (mut rb_frames, sr_b) = excerpt(right_src)?;

        if sr_a != sr_b {
            return Err(format!("{case}: sample-rate mismatch {sr_a} vs {sr_b}").into());
        }
        if la_frames.len() != rb_frames.len() {
            return Err(format!(
                "{case}: excerpt length mismatch {} vs {} frames",
                la_frames.len(),
                rb_frames.len()
            )
            .into());
        }

        let (target, initial, gains, finals) =
            match match_pair(&mut la_frames, &mut rb_frames, sr_a) {
                Ok(v) => v,
                // Fail closed: a pair that cannot be matched is not presented
                // as if it were. This is the guarantee v1 lacked.
                Err(why) => return Err(format!("{case}: {why}").into()),
            };

        let mut records = Vec::new();
        let mut rendered_lufs: Vec<f32> = Vec::new();
        for (i, (src, dst, frames)) in [
            (left_src, &left_clip, &la_frames),
            (right_src, &right_clip, &rb_frames),
        ]
        .into_iter()
        .enumerate()
        {
            let out_path = out_dir.join(dst);
            write_wav(&out_path, frames, sr_a)?;
            // Measure the FILE, not the buffer that produced it. `match_pair`
            // already verified the float frames; reading back re-measures the
            // artifact actually handed to a listener, so 16-bit quantisation
            // and any write-path bug are inside the guarantee rather than
            // outside it. Verifying the buffer instead would be checking a
            // proxy for the deliverable — the exact substitution this tool
            // exists to prevent.
            let (rendered, _) = read_wav(&out_path)?;
            let (flufs, fpk) = measured(&rendered, sr_a);
            rendered_lufs.push(flufs);
            records.push(ClipRecord {
                source: src.display().to_string(),
                source_sha256: sha256_hex(&std::fs::read(src)?),
                output: dst.clone(),
                output_sha256: sha256_hex(&std::fs::read(&out_path)?),
                sample_rate: sr_a,
                excerpt_secs,
                initial_lufs: initial[i],
                initial_peak: 0.0, // filled below from the pre-gain measurement
                requested_target_lufs: TARGET_LUFS,
                pair_target_lufs: target,
                applied_gain_db: gains[i],
                final_lufs: 0.0, // set below from the rendered file
                final_peak: fpk,
                matcher: MATCHER.into(),
            });
        }
        // Recover each clip's PRE-gain peak from its post-gain peak and the
        // gain actually applied — cheaper than keeping a second copy of the
        // audio, and exact, since gain is a scalar multiply.
        for (i, r) in records.iter_mut().enumerate() {
            r.initial_peak = r.final_peak / 10f32.powf(gains[i] / 20.0);
            r.final_lufs = rendered_lufs[i];
        }
        // The guarantee, re-checked against what was actually written.
        let rendered_delta = (rendered_lufs[0] - rendered_lufs[1]).abs();
        if rendered_delta > MAX_PAIR_DELTA_LU {
            return Err(format!(
                "{case}: rendered files differ by {rendered_delta:.3} LU (limit \
                 {MAX_PAIR_DELTA_LU}) even though the float buffers matched — \
                 the write path changed the audio"
            )
            .into());
        }
        eprintln!(
            "  {case}  target {target:.2} LUFS  A {:+.2} dB -> {:.2}  B {:+.2} dB -> {:.2}  \
             delta {:.3} LU",
            gains[0], rendered_lufs[0], gains[1], rendered_lufs[1], rendered_delta
        );
        clip_records.extend(records);
        trials.push(Trial {
            case: case.clone(),
            left_clip,
            right_clip,
            left_variant,
            right_variant,
        });
    }

    if trials.is_empty() {
        return Err("no complete pairs found — nothing to listen to".into());
    }
    if !skipped.is_empty() {
        eprintln!(
            "\nWARNING: skipped {} case(s) without both renders: {:?}",
            skipped.len(),
            skipped
        );
    }

    // The listener's copy: trials with the variant names stripped.
    #[derive(Serialize)]
    struct PublicTrial<'a> {
        trial: usize,
        left_clip: &'a str,
        right_clip: &'a str,
    }
    let public: Vec<_> = trials
        .iter()
        .enumerate()
        .map(|(i, t)| PublicTrial {
            trial: i + 1,
            left_clip: &t.left_clip,
            right_clip: &t.right_clip,
        })
        .collect();
    std::fs::write(
        out_dir.join("trials.json"),
        serde_json::to_vec_pretty(&public)?,
    )?;
    std::fs::write(out_dir.join("index.html"), player_html(&public.len()))?;
    std::fs::write(
        out_dir.join("responses.template.json"),
        serde_json::to_vec_pretty(
            &(1..=trials.len())
                .map(|i| serde_json::json!({"trial": i, "prefer": "A"}))
                .collect::<Vec<_>>(),
        )?,
    )?;

    // Key goes NEXT TO the pack, not inside it.
    let key = AnswerKey {
        pack_version: "muse-ab-pack-v2".into(),
        seed,
        target_lufs: TARGET_LUFS,
        excerpt_secs,
        matcher: MATCHER.into(),
        max_pair_delta_lu: MAX_PAIR_DELTA_LU,
        variants: variants.clone(),
        trials,
        clips: clip_records,
    };
    let key_path = out_dir.with_extension("key.json");
    std::fs::write(&key_path, serde_json::to_vec_pretty(&key)?)?;

    println!("\npack:  {}", out_dir.display());
    println!(
        "key:   {}  (do NOT put this in the pack)",
        key_path.display()
    );
    println!("{} trials, variants {:?}", key.trials.len(), variants);
    let worst = key
        .trials
        .iter()
        .enumerate()
        .map(|(i, _)| (key.clips[i * 2].final_lufs - key.clips[i * 2 + 1].final_lufs).abs())
        .fold(0.0f32, f32::max);
    println!(
        "matcher {MATCHER}: every pair verified within {MAX_PAIR_DELTA_LU} LU \
         (worst observed {worst:.3} LU)"
    );
    println!("\nopen {}/index.html, then:", out_dir.display());
    println!(
        "  ab_listening_pack score {} {}/responses.json",
        key_path.display(),
        out_dir.display()
    );
    Ok(())
}

fn player_html(n: &usize) -> String {
    format!(
        r#"<!doctype html><meta charset=utf-8><title>Muse A/B</title>
<style>body{{font:16px/1.5 system-ui;max-width:44rem;margin:2rem auto;padding:0 1rem}}
.t{{border:1px solid #ccc;border-radius:8px;padding:1rem;margin:1rem 0}}
audio{{width:100%;margin:.3rem 0}}button{{font:inherit;padding:.4rem .9rem;margin-right:.5rem}}
#out{{white-space:pre;background:#f6f6f6;padding:1rem;border-radius:8px}}</style>
<h1>Muse A/B — {n} trials</h1>
<p><label>Your name or initials <input id=who placeholder="e.g. TS"></label>
(recorded with every answer so results can be broken down per listener —
judgments from one person are correlated and must not be pooled blindly)</p>
<p>Each trial is the same piece rendered two ways, level-matched. Play both,
pick the one you'd rather listen to again. If you genuinely can't tell, say so —
that is a real answer and the scorer handles it.</p>
<div id=trials></div>
<button onclick=save()>Export responses.json</button>
<div id=out></div>
<script>
const N={n}, r={{}};
const SESSION = 'S' + Date.now().toString(36);
const root=document.getElementById('trials');
for(let i=1;i<=N;i++){{
  const p=String(i).padStart(2,'0');
  root.insertAdjacentHTML('beforeend',
    `<div class=t><b>Trial ${{i}}</b>
     <div>A <audio controls preload=none src=trial${{p}}_A.wav></audio></div>
     <div>B <audio controls preload=none src=trial${{p}}_B.wav></audio></div>
     <button onclick="pick(${{i}},'A',this)">Prefer A</button>
     <button onclick="pick(${{i}},'B',this)">Prefer B</button>
     <button onclick="pick(${{i}},'tie',this)">Can't tell</button>
     <span id=s${{i}}></span></div>`);
}}
function pick(i,v,el){{r[i]=v;document.getElementById('s'+i).textContent=' → '+v;}}
function save(){{
  const who=(document.getElementById('who').value||'').trim();
  if(!who){{alert('Please enter your name or initials first.');return;}}
  const out=[];for(let i=1;i<=N;i++)if(r[i])
    out.push({{trial:i,prefer:r[i],listener_id:who,session_id:SESSION}});
  document.getElementById('out').textContent=JSON.stringify(out,null,1);
  const b=new Blob([JSON.stringify(out,null,1)],{{type:'application/json'}});
  const a=document.createElement('a');a.href=URL.createObjectURL(b);
  a.download='responses.json';a.click();
}}
</script>
"#
    )
}

#[derive(Debug, Deserialize)]
struct Response {
    trial: usize,
    prefer: String,
    /// Who judged it. Optional so older response files still parse, but
    /// without it the by-listener breakdown collapses into one row and the
    /// clustering it exists to expose becomes invisible — the player always
    /// writes it.
    #[serde(default)]
    listener_id: Option<String>,
    /// Which sitting. One listener across two sessions is still correlated,
    /// but differently from within a single sitting (fatigue, calibration).
    #[serde(default)]
    session_id: Option<String>,
}

/// Two-sided exact binomial (sign) test at p = 0.5, ties excluded — the
/// standard treatment, but it means the effective n is the number of listeners
/// who actually heard a difference, which is reported alongside.
fn sign_test_p(wins: usize, n: usize) -> f64 {
    if n == 0 {
        return 1.0;
    }
    let mut log_c = 0.0f64; // log C(n, k), built incrementally
    let mut tail = 0.0f64;
    let extreme = wins.max(n - wins);
    for k in 0..=n {
        if k > 0 {
            log_c += ((n - k + 1) as f64).ln() - (k as f64).ln();
        }
        if k >= extreme || k <= n - extreme {
            tail += (log_c - (n as f64) * std::f64::consts::LN_2).exp();
        }
    }
    tail.min(1.0)
}

fn score(key_path: &Path, responses_path: &Path) -> Result<(), Box<dyn Error>> {
    let key: AnswerKey = serde_json::from_slice(&std::fs::read(key_path)?)?;
    let responses: Vec<Response> = serde_json::from_slice(&std::fs::read(responses_path)?)?;

    let (a, b) = (&key.variants[0], &key.variants[1]);
    // (variant-preferred, ties) keyed by case and by listener. Both breakdowns
    // matter because neither dimension is independent: one listener's twelve
    // judgments share their taste and fatigue; one case's three judgments share
    // that case's character.
    let mut by_case: BTreeMap<String, (usize, usize, usize)> = BTreeMap::new();
    let mut by_listener: BTreeMap<String, (usize, usize, usize)> = BTreeMap::new();
    let (mut na, mut nb, mut ties) = (0usize, 0usize, 0usize);
    let mut unknown = Vec::new();

    for r in &responses {
        let Some(trial) = key.trials.get(r.trial.wrapping_sub(1)) else {
            unknown.push(r.trial);
            continue;
        };
        let listener = r.listener_id.clone().unwrap_or_else(|| "(unnamed)".into());
        let picked = match r.prefer.as_str() {
            "A" => Some(trial.left_variant.clone()),
            "B" => Some(trial.right_variant.clone()),
            "tie" => None,
            other => return Err(format!("trial {}: unknown response {other:?}", r.trial).into()),
        };
        let ce = by_case.entry(trial.case.clone()).or_default();
        let le = by_listener.entry(listener).or_default();
        match picked.as_deref() {
            Some(v) if v == a.as_str() => {
                na += 1;
                ce.0 += 1;
                le.0 += 1;
            }
            Some(_) => {
                nb += 1;
                ce.1 += 1;
                le.1 += 1;
            }
            None => {
                ties += 1;
                ce.2 += 1;
                le.2 += 1;
            }
        }
    }
    if !unknown.is_empty() {
        return Err(format!("responses reference trials not in the key: {unknown:?}").into());
    }

    let decided = na + nb;
    println!("variants: {a}  vs  {b}");
    println!("\nOVERALL   {a}: {na}   {b}: {nb}   can't tell: {ties}");
    if decided > 0 {
        println!(
            "          preference fraction for {a}: {:.2}",
            na as f64 / decided as f64
        );
    }

    println!("\nBY CASE   (each row is one musical case, judged by everyone)");
    println!(
        "  {:<28} {:>6} {:>6} {:>5}",
        "case",
        a_short(a),
        a_short(b),
        "tie"
    );
    for (case, (x, y, t)) in &by_case {
        println!("  {case:<28} {x:>6} {y:>6} {t:>5}");
    }

    println!("\nBY LISTENER");
    println!(
        "  {:<28} {:>6} {:>6} {:>5}",
        "listener",
        a_short(a),
        a_short(b),
        "tie"
    );
    for (l, (x, y, t)) in &by_listener {
        println!("  {l:<28} {x:>6} {y:>6} {t:>5}");
    }
    let directions: Vec<&str> = by_listener
        .values()
        .map(|(x, y, _)| match x.cmp(y) {
            std::cmp::Ordering::Greater => "A",
            std::cmp::Ordering::Less => "B",
            std::cmp::Ordering::Equal => "=",
        })
        .collect();
    let agree = directions.iter().all(|d| *d == directions[0]) && directions[0] != "=";
    println!(
        "\n  listeners agreeing on direction: {}",
        if by_listener.len() < 2 {
            "n/a (fewer than 2 listeners)".to_string()
        } else if agree {
            format!("all {} agree", by_listener.len())
        } else {
            format!("split — {directions:?}")
        }
    );

    // Deliberately last, deliberately caveated. Listener and case clustering
    // mean these are not independent Bernoulli trials, so this number is not
    // the result; it is a directional smell test.
    println!(
        "\nEXPLORATORY (naive): two-sided exact binomial p = {:.4} over {decided} decided \
         comparisons.\n  ASSUMES INDEPENDENT OBSERVATIONS, WHICH THIS DESIGN DOES NOT SATISFY \
         — judgments\n  cluster by listener and by case. Do not quote this as a result; read the \
         tables above.",
        sign_test_p(na, decided)
    );
    if decided == 0 {
        println!("\nno decided comparisons — nothing to conclude.");
    }
    Ok(())
}

/// Trim a long variant name to something that fits a column header.
fn a_short(v: &str) -> String {
    if v.len() <= 6 {
        v.to_string()
    } else {
        v[..6].to_string()
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("build") if args.len() >= 4 => {
            let seed = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(1);
            let secs = args
                .get(5)
                .and_then(|s| s.parse().ok())
                .unwrap_or(DEFAULT_EXCERPT_SECS);
            build(Path::new(&args[2]), Path::new(&args[3]), seed, secs)
        }
        Some("score") if args.len() >= 4 => score(Path::new(&args[2]), Path::new(&args[3])),
        _ => {
            eprintln!(
                "usage:\n  ab_listening_pack build <pairs_dir> <out_dir> [seed] [excerpt_secs]\n  \
                 ab_listening_pack score <key.json> <responses.json>"
            );
            Err("bad usage".into())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sign_test_matches_known_values() {
        // Symmetric cases: all-one-way at n=6 is 2*(1/64) = 0.03125.
        assert!((sign_test_p(6, 6) - 0.03125).abs() < 1e-9);
        assert!((sign_test_p(0, 6) - 0.03125).abs() < 1e-9);
        // A dead-even split cannot be evidence of anything.
        assert!((sign_test_p(3, 6) - 1.0).abs() < 1e-9);
        // n=0 is defined, not a panic or NaN.
        assert_eq!(sign_test_p(0, 0), 1.0);
    }

    #[test]
    fn sign_test_needs_a_real_run_to_reach_significance() {
        // Guards the honest-limits claim in this file's docs: five unanimous
        // listeners are NOT enough to clear p<0.05 two-sided, six are.
        assert!(sign_test_p(5, 5) > 0.05);
        assert!(sign_test_p(6, 6) < 0.05);
    }

    /// A sine at `amp`, plus an optional single-sample spike, so a clip's PEAK
    /// can be raised independently of its loudness — which is exactly the
    /// condition that broke v1.
    fn clip(amp: f32, spike: Option<f32>) -> Vec<[f32; 2]> {
        let mut f: Vec<[f32; 2]> = (0..44_100)
            .map(|i| {
                let s = (i as f32 * 0.05).sin() * amp;
                [s, s]
            })
            .collect();
        if let Some(p) = spike {
            f[10] = [p, p];
        }
        f
    }

    fn lufs_of(f: &[[f32; 2]]) -> f32 {
        measure_lufs(f, 44_100).integrated
    }

    #[test]
    fn unconstrained_pair_both_reach_the_requested_target() {
        let (mut a, mut b) = (clip(0.05, None), clip(0.02, None));
        let (target, _, _, finals) = match_pair(&mut a, &mut b, 44_100).expect("should match");
        assert!(
            (target - TARGET_LUFS).abs() < 1e-3,
            "nothing forced a quieter target, so it should be the requested one: {target}"
        );
        assert!((finals[0] - finals[1]).abs() <= MAX_PAIR_DELTA_LU);
    }

    #[test]
    fn one_peak_limited_clip_drags_both_to_a_quieter_shared_target() {
        // THE v1 BUG, reproduced. `b` is quiet but has a near-full-scale spike,
        // so it cannot be raised to -16 LUFS without clipping. v1 would push
        // `a` to -16 and leave `b` short, then call the pair matched.
        let (mut a, mut b) = (clip(0.05, None), clip(0.02, Some(0.98)));
        let (target, _, _, finals) = match_pair(&mut a, &mut b, 44_100).expect("should match");
        assert!(
            target < TARGET_LUFS - 0.5,
            "the spiky clip's headroom should force a quieter shared target, got {target}"
        );
        assert!(
            (finals[0] - finals[1]).abs() <= MAX_PAIR_DELTA_LU,
            "pair must still end matched: {finals:?}"
        );
    }

    #[test]
    fn both_peak_limited_at_different_levels_still_match() {
        let (mut a, mut b) = (clip(0.04, Some(0.95)), clip(0.02, Some(0.80)));
        let (_, _, _, finals) = match_pair(&mut a, &mut b, 44_100).expect("should match");
        assert!((finals[0] - finals[1]).abs() <= MAX_PAIR_DELTA_LU);
        for f in [&a, &b] {
            let pk = f
                .iter()
                .flat_map(|x| [x[0].abs(), x[1].abs()])
                .fold(0.0f32, f32::max);
            assert!(pk <= PEAK_CEILING + 1e-4, "peak {pk} over ceiling");
        }
    }

    #[test]
    fn matching_is_deterministic_from_the_same_source() {
        // Always regenerate from the original excerpt; never re-normalize
        // already-gained audio, or repeated runs accumulate error.
        let run = || {
            let (mut a, mut b) = (clip(0.05, None), clip(0.02, Some(0.98)));
            let (t, _, g, f) = match_pair(&mut a, &mut b, 44_100).unwrap();
            (t, g, f, a[500], b[500])
        };
        let first = run();
        let second = run();
        assert_eq!(first.0.to_bits(), second.0.to_bits(), "target drifted");
        assert_eq!(first.3, second.3, "sample drifted");
        assert_eq!(first.4, second.4, "sample drifted");
    }

    #[test]
    fn a_pair_that_cannot_be_matched_is_rejected() {
        // Silence has no measurable loudness, so no gain can bring it to any
        // target. It must fail closed rather than be emitted as a trial.
        let (mut a, mut b) = (clip(0.05, None), vec![[0.0f32; 2]; 44_100]);
        assert!(match_pair(&mut a, &mut b, 44_100).is_err());
    }

    #[test]
    fn v1_algorithm_fails_the_property_v2_guarantees() {
        // The regression that makes the discovery permanent. v1 normalized each
        // clip independently and backed off on peak; reproduced here exactly.
        // On the spiky pair it leaves a gap far outside tolerance -- which is
        // why 22 of 32 clips in the real Baroque pack were capped, not matched.
        fn v1(frames: &mut [[f32; 2]], sr: u32) {
            let m = measure_lufs(frames, sr).integrated;
            let mut gain = 10f32.powf((TARGET_LUFS - m) / 20.0);
            let peak = frames
                .iter()
                .flat_map(|f| [f[0].abs(), f[1].abs()])
                .fold(0.0f32, f32::max);
            if peak * gain > PEAK_CEILING {
                gain = PEAK_CEILING / peak.max(f32::EPSILON);
            }
            for f in frames.iter_mut() {
                f[0] *= gain;
                f[1] *= gain;
            }
        }
        let (mut a1, mut b1) = (clip(0.05, None), clip(0.02, Some(0.98)));
        v1(&mut a1, 44_100);
        v1(&mut b1, 44_100);
        let v1_delta = (lufs_of(&a1) - lufs_of(&b1)).abs();
        assert!(
            v1_delta > MAX_PAIR_DELTA_LU,
            "v1 was supposed to FAIL this pair; delta {v1_delta:.3} LU"
        );

        let (mut a2, mut b2) = (clip(0.05, None), clip(0.02, Some(0.98)));
        let (_, _, _, finals) = match_pair(&mut a2, &mut b2, 44_100).expect("v2 should match");
        assert!(
            (finals[0] - finals[1]).abs() <= MAX_PAIR_DELTA_LU,
            "v2 must pass what v1 failed"
        );
    }

    #[test]
    fn fade_out_ends_silent_without_touching_the_body() {
        // A hard truncation click is an obvious artefact that would sway
        // preference on its own, so the excerpt must actually land softly.
        let mut frames = vec![[1.0f32, 1.0]; 44_100];
        fade_out(&mut frames, 44_100);
        let last = frames.last().unwrap();
        assert!(
            last[0].abs() < 1e-3 && last[1].abs() < 1e-3,
            "excerpt should fade to silence, ended at {last:?}"
        );
        // The fade is 0.75s, so anything before that is untouched.
        assert_eq!(frames[0], [1.0, 1.0]);
        assert_eq!(frames[44_100 - 33_075 - 1], [1.0, 1.0]);
    }

    #[test]
    fn fade_out_handles_clips_shorter_than_the_fade() {
        // min() guard: a 0.1s clip must not index past its own end.
        let mut frames = vec![[1.0f32, 1.0]; 4_410];
        fade_out(&mut frames, 44_100);
        assert!(frames.last().unwrap()[0].abs() < 1e-3);
    }

    #[test]
    fn rng_is_deterministic_and_uses_high_bits() {
        let flips: Vec<bool> = {
            let mut r = Rng(20260729);
            (0..16).map(|_| r.flip()).collect()
        };
        let again: Vec<bool> = {
            let mut r = Rng(20260729);
            (0..16).map(|_| r.flip()).collect()
        };
        assert_eq!(flips, again, "same seed must rebuild the same pack");
        assert!(
            flips.iter().any(|&b| b) && flips.iter().any(|&b| !b),
            "a side-assignment that never varies would silently unblind the pack"
        );
    }
}
