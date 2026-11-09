// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Blind, backend-comparative Muse vocal release gate for integrated backends.
//! The unintegrated physical-tract prototype is tracked separately and is not
//! represented as a selectable singing backend here.

use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use symthaea::voice::diffsinger::DiffSingerEngine;
use symthaea::voice::formant_singing_engine::FormantSingingEngine;
use symthaea::voice::kokoro_singing::KokoroSingingEngine;
use symthaea::voice::singing_engine::{
    ControlCapability, SingingBackendCapabilities, SingingControl, SingingRenderPolicy,
    SingingVoiceEngine, VocalPerformance, VocalStem,
};
use symthaea::voice::singing_quality::{VocalQualityMetrics, analyze_vocal_stem};
use symthaea_muse::{MusicalIntent, MusicalState, Style};

const PHRASES: &[(&str, &str)] = &[
    ("softly now the stars are waking", "legato_sustain"),
    ("the sun rises over the valley", "clear_consonants"),
    ("every silver river remembers", "liquid_consonants"),
    ("hold the light inside the silence", "soft_attack"),
    (
        "we are dancing through the morning",
        "rhythmic_articulation",
    ),
    ("consciousness shapes every note", "consonant_cluster"),
    ("bright birds scatter into blue", "plosive_cluster"),
    ("a quiet fire carries me home", "breath_control"),
    ("beautiful impossible harmony", "multisyllabic"),
    ("stay with me beyond the turning tide", "long_phrase"),
    ("soft breath clear heart open sky", "aspirates"),
    ("sing the world awake again", "strong_attack"),
    ("thin threads twist through the stillness", "fricatives"),
    ("joy joins every echo", "affricates"),
    ("round the wild horizon we wander", "diphthongs"),
    ("keep this little flame alive", "voiced_unvoiced"),
    ("moonlight moves across the water", "nasals"),
    ("rise and shimmer over me", "melisma"),
    ("hold on through the longest night", "sustain"),
    ("gently let the ending fall", "phrase_release"),
];

#[derive(Clone, Copy)]
struct GateCase {
    lyrics: &'static str,
    challenge: &'static str,
    category: &'static str,
    style: Style,
    valence: f32,
    arousal: f32,
    transpose_semitones: f32,
}

fn cases() -> Vec<GateCase> {
    PHRASES
        .iter()
        .flat_map(|&(lyrics, challenge)| {
            [
                ("intimate_low", Style::Folk, 0.35, 0.25, -5.0),
                ("center_legato", Style::Classical, 0.55, 0.48, 0.0),
                ("luminous_high", Style::Cinematic, 0.78, 0.72, 5.0),
            ]
            .into_iter()
            .map(
                move |(category, style, valence, arousal, transpose_semitones)| GateCase {
                    lyrics,
                    challenge,
                    category,
                    style,
                    valence,
                    arousal,
                    transpose_semitones,
                },
            )
        })
        .collect()
}

#[derive(Serialize)]
struct BlindClip {
    clip_id: String,
    case_id: usize,
    lyrics: String,
    audio: String,
    audio_blake3: String,
    performance_blake3: Option<String>,
    category: String,
    challenge: String,
}

#[derive(Serialize)]
struct KeyEntry {
    clip_id: String,
    backend: String,
    case_id: usize,
    performance_blake3: Option<String>,
    audio_blake3: String,
    capability_manifest_blake3: Option<String>,
    render_policy_blake3: Option<String>,
}

#[derive(Clone, Serialize)]
struct CaseReceipt {
    case_id: usize,
    seed: u64,
    lyrics: String,
    challenge: String,
    category: String,
    style: String,
    valence: f32,
    arousal: f32,
    composition_energy: f32,
    bars: usize,
    consciousness_level: f32,
    transpose_semitones: f32,
    melody: Vec<symthaea_muse::Note>,
    score_blake3: String,
    performance_canonical_json: String,
    performance_blake3: String,
}

#[derive(Serialize)]
struct RunManifest {
    schema: &'static str,
    source_revision: String,
    voice_id: String,
    diffsinger_worker: Option<String>,
    diffsinger_worker_blake3: Option<String>,
    case_count: usize,
    cases: Vec<CaseReceipt>,
}

#[derive(Serialize)]
struct ObjectiveEntry {
    clip_id: String,
    backend: String,
    case_id: usize,
    metrics: VocalQualityMetrics,
    capability_manifest: SingingBackendCapabilities,
    capability_manifest_canonical_json: String,
    capability_manifest_blake3: String,
    render_policy: SingingRenderPolicy,
    render_policy_canonical_json: String,
    render_policy_blake3: String,
    accepted_control_dispositions: Vec<ControlCapability>,
}

#[derive(Deserialize)]
struct HumanRating {
    listener_id: String,
    clip_id: String,
    naturalness: f32,
    emotional_fit: f32,
    identity_consistency: f32,
    artifact_free: f32,
    lyric_words_understood: usize,
    lyric_words_total: usize,
}

/// Optional authorized human anchors. Paths remain private; imported copies
/// receive the same salted concealment as synthetic clips.
#[derive(Deserialize)]
struct ReferenceClip {
    case_id: usize,
    audio: PathBuf,
    lyrics: String,
    category: String,
    challenge: String,
    score_blake3: String,
    singer_pseudonym: String,
    consent_reference: String,
    license_reference: String,
}

#[derive(Serialize)]
struct HumanReferenceReceipt {
    case_id: usize,
    singer_pseudonym: String,
    consent_reference: String,
    license_reference: String,
    score_blake3: String,
    audio_blake3: String,
}

fn arg(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|x| x == flag)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn save(path: &Path, stem: &VocalStem) -> Result<()> {
    let path = path.to_str().context("non-UTF8 output path")?;
    symthaea::voice::save_wav(&stem.samples, stem.sample_rate, path)
}

fn blind_id(salt: &str, backend: &str, case_id: usize) -> String {
    let digest = blake3::hash(format!("muse-vocal-v2:{salt}:{backend}:{case_id}").as_bytes());
    format!("vocal-{}", &digest.to_hex()[..12])
}

fn source_revision(args: &[String]) -> Result<String> {
    let revision = arg(args, "--source-revision")
        .or_else(|| std::env::var("VOCAL_PHONEME_REVISION").ok())
        .or_else(|| std::env::var("GITHUB_SHA").ok())
        .or_else(|| {
            let status = Command::new("git")
                .args(["status", "--porcelain", "--untracked-files=all"])
                .output()
                .ok()?;
            if !status.status.success() || !status.stdout.is_empty() {
                return None;
            }
            let output = Command::new("git").args(["rev-parse", "HEAD"]).output().ok()?;
            if !output.status.success() {
                return None;
            }
            String::from_utf8(output.stdout)
                .ok()
                .map(|value| value.trim().to_string())
        })
        .context(
            "exact source revision unavailable; provide --source-revision, VOCAL_PHONEME_REVISION, or a clean Git checkout",
        )?;
    validate_source_revision(&revision)
}

fn validate_source_revision(revision: &str) -> Result<String> {
    if revision.len() != 40 || !revision.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        anyhow::bail!("source revision must be a 40-character hexadecimal Git commit SHA");
    }
    Ok(revision.to_ascii_lowercase())
}

fn performance_digest(performance: &VocalPerformance) -> Result<String> {
    let bytes = serde_json::to_vec(performance)?;
    Ok(blake3::hash(&bytes).to_hex().to_string())
}

fn score_digest(melody: &[symthaea_muse::Note]) -> Result<String> {
    let bytes = serde_json::to_vec(melody)?;
    Ok(blake3::hash(&bytes).to_hex().to_string())
}

fn validate_reference_case(reference: &ReferenceClip, case: &CaseReceipt) -> Result<()> {
    if reference.case_id != case.case_id {
        anyhow::bail!("human reference case ID does not match its score receipt");
    }
    if reference.lyrics != case.lyrics
        || reference.category != case.category
        || reference.challenge != case.challenge
    {
        anyhow::bail!(
            "human reference lyrics/category/challenge do not match case {}",
            case.case_id
        );
    }
    if reference.score_blake3 != case.score_blake3 {
        anyhow::bail!(
            "human reference score digest does not match case {}",
            case.case_id
        );
    }
    if reference.singer_pseudonym.trim().is_empty()
        || reference.consent_reference.trim().is_empty()
        || reference.license_reference.trim().is_empty()
    {
        anyhow::bail!(
            "human reference case {} requires singer pseudonym, consent reference, and license reference",
            case.case_id
        );
    }
    Ok(())
}

fn artifact_digest(path: &Path) -> Result<String> {
    let bytes = std::fs::read(path)?;
    Ok(blake3::hash(&bytes).to_hex().to_string())
}

fn blind_salt(args: &[String], output: &Path) -> Result<String> {
    if let Some(seed) = arg(args, "--blind-seed") {
        return Ok(blake3::hash(seed.as_bytes()).to_hex().to_string());
    }
    let mut bytes = [0u8; 32];
    std::fs::File::open("/dev/urandom")
        .and_then(|mut file| file.read_exact(&mut bytes))
        .with_context(|| {
            format!(
                "secure blinding entropy unavailable for {}",
                output.display()
            )
        })?;
    Ok(blake3::hash(&bytes).to_hex().to_string())
}

fn render_one(
    engine: &mut dyn SingingVoiceEngine,
    performance: &VocalPerformance,
    case_id: usize,
    case: &GateCase,
    salt: &str,
    output: &Path,
    clips: &mut Vec<BlindClip>,
    key: &mut Vec<KeyEntry>,
    objective: &mut Vec<ObjectiveEntry>,
) -> Result<()> {
    let capabilities = engine.capabilities();
    capabilities.validate()?;
    if capabilities.backend_id != engine.id() {
        anyhow::bail!(
            "backend capability ID {:?} does not match renderer ID {:?}",
            capabilities.backend_id,
            engine.id()
        );
    }
    let capability_manifest_canonical_json = serde_json::to_string(&capabilities)?;
    let capability_manifest_blake3 = blake3::hash(capability_manifest_canonical_json.as_bytes())
        .to_hex()
        .to_string();
    let policy = SingingRenderPolicy::score_conditioned_comparison();
    let policy_canonical_json = serde_json::to_string(&policy)?;
    let policy_blake3 = blake3::hash(policy_canonical_json.as_bytes())
        .to_hex()
        .to_string();
    let accepted_control_dispositions = policy
        .required_controls
        .iter()
        .map(|control| {
            capabilities
                .controls
                .iter()
                .find(|entry| entry.control == *control)
                .cloned()
                .with_context(|| format!("missing declared disposition for {control:?}"))
        })
        .collect::<Result<Vec<_>>>()?;
    let stem = engine.render_with_policy(performance, &policy)?;
    let id = blind_id(salt, &stem.backend, case_id);
    let filename = format!("{id}.wav");
    let audio_path = output.join(&filename);
    save(&audio_path, &stem)?;
    let audio_blake3 = artifact_digest(&audio_path)?;
    let performance_blake3 = performance_digest(performance)?;
    let metrics = analyze_vocal_stem(performance, &stem);
    clips.push(BlindClip {
        clip_id: id.clone(),
        case_id,
        lyrics: performance.lyrics.clone(),
        audio: filename,
        audio_blake3: audio_blake3.clone(),
        performance_blake3: Some(performance_blake3.clone()),
        category: case.category.to_string(),
        challenge: case.challenge.to_string(),
    });
    key.push(KeyEntry {
        clip_id: id.clone(),
        backend: stem.backend.clone(),
        case_id,
        performance_blake3: Some(performance_blake3),
        audio_blake3,
        capability_manifest_blake3: Some(capability_manifest_blake3.clone()),
        render_policy_blake3: Some(policy_blake3.clone()),
    });
    objective.push(ObjectiveEntry {
        clip_id: id,
        backend: stem.backend,
        case_id,
        metrics,
        capability_manifest: capabilities,
        capability_manifest_canonical_json,
        capability_manifest_blake3,
        render_policy: policy,
        render_policy_canonical_json: policy_canonical_json,
        render_policy_blake3: policy_blake3,
        accepted_control_dispositions,
    });
    Ok(())
}

fn import_references(
    manifest: &Path,
    cases: &[CaseReceipt],
    salt: &str,
    output: &Path,
    clips: &mut Vec<BlindClip>,
    key: &mut Vec<KeyEntry>,
) -> Result<Vec<HumanReferenceReceipt>> {
    let references: Vec<ReferenceClip> = serde_json::from_slice(&std::fs::read(manifest)?)?;
    let mut receipts = Vec::with_capacity(references.len());
    for reference in references {
        let case = cases
            .iter()
            .find(|case| case.case_id == reference.case_id)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "human reference references unknown case {}",
                    reference.case_id
                )
            })?;
        validate_reference_case(&reference, case)?;

        let metadata = std::fs::symlink_metadata(&reference.audio)
            .with_context(|| format!("missing human reference: {}", reference.audio.display()))?;
        if metadata.file_type().is_symlink() || !metadata.file_type().is_file() {
            anyhow::bail!(
                "human reference must be a regular non-symlink file: {}",
                reference.audio.display()
            );
        }

        let id = blind_id(salt, "human-reference", reference.case_id);
        if clips.iter().any(|clip| clip.clip_id == id) {
            anyhow::bail!("duplicate human-reference case_id {}", reference.case_id);
        }
        let filename = format!("{id}.wav");
        let destination = output.join(&filename);
        std::fs::copy(&reference.audio, &destination)?;
        let audio_blake3 = artifact_digest(&destination)?;
        clips.push(BlindClip {
            clip_id: id.clone(),
            case_id: reference.case_id,
            lyrics: reference.lyrics,
            audio: filename,
            audio_blake3: audio_blake3.clone(),
            performance_blake3: None,
            category: reference.category,
            challenge: reference.challenge,
        });
        key.push(KeyEntry {
            clip_id: id,
            backend: "human-reference".to_string(),
            case_id: reference.case_id,
            performance_blake3: None,
            audio_blake3: audio_blake3.clone(),
            capability_manifest_blake3: None,
            render_policy_blake3: None,
        });
        receipts.push(HumanReferenceReceipt {
            case_id: reference.case_id,
            singer_pseudonym: reference.singer_pseudonym,
            consent_reference: reference.consent_reference,
            license_reference: reference.license_reference,
            score_blake3: reference.score_blake3,
            audio_blake3,
        });
    }
    Ok(receipts)
}

#[derive(Clone, Debug, PartialEq, Serialize)]
struct RatingCoverage {
    clip_count: usize,
    listener_count: usize,
    rating_count: usize,
    complete: bool,
}

fn validate_rating_inputs(
    ratings: &[HumanRating],
    clips: &[BlindClip],
    minimum_listeners: usize,
) -> Result<RatingCoverage> {
    if minimum_listeners == 0 {
        anyhow::bail!("minimum listener count must be at least one");
    }

    let mut clip_by_id = BTreeMap::new();
    for clip in clips {
        if clip.clip_id.trim().is_empty()
            || clip_by_id.insert(clip.clip_id.as_str(), clip).is_some()
        {
            anyhow::bail!("blind manifest contains an empty or duplicate clip ID");
        }
    }

    let listeners: std::collections::BTreeSet<_> = ratings
        .iter()
        .map(|rating| rating.listener_id.as_str())
        .collect();
    let mut by_clip_listener = BTreeMap::new();

    for rating in ratings {
        if rating.listener_id.trim().is_empty() {
            anyhow::bail!("human rating has an empty listener ID");
        }
        let Some(clip) = clip_by_id.get(rating.clip_id.as_str()) else {
            anyhow::bail!("human rating references unknown clip ID {}", rating.clip_id);
        };

        let rating_values = [
            rating.naturalness,
            rating.emotional_fit,
            rating.identity_consistency,
            rating.artifact_free,
        ];
        if rating_values
            .iter()
            .any(|value| !value.is_finite() || !(1.0..=5.0).contains(value))
        {
            anyhow::bail!("invalid or non-finite human rating for {}", rating.clip_id);
        }

        let expected_words = clip.lyrics.split_whitespace().count();
        if expected_words == 0
            || rating.lyric_words_total != expected_words
            || rating.lyric_words_understood > rating.lyric_words_total
        {
            anyhow::bail!(
                "invalid lyric comprehension denominator/count for {}: expected {expected_words}, got {}/{}",
                rating.clip_id,
                rating.lyric_words_understood,
                rating.lyric_words_total
            );
        }

        if by_clip_listener
            .insert((rating.clip_id.as_str(), rating.listener_id.as_str()), ())
            .is_some()
        {
            anyhow::bail!("duplicate listener/clip rating for {}", rating.clip_id);
        }
    }

    let complete = listeners.len() >= minimum_listeners
        && clips.iter().all(|clip| {
            listeners
                .iter()
                .all(|listener| by_clip_listener.contains_key(&(clip.clip_id.as_str(), *listener)))
        });

    Ok(RatingCoverage {
        clip_count: clips.len(),
        listener_count: listeners.len(),
        rating_count: ratings.len(),
        complete,
    })
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let output = PathBuf::from(
        arg(&args, "--output").unwrap_or_else(|| "audio_output/muse_vocal_gate".into()),
    );
    let voice = arg(&args, "--voice").unwrap_or_else(|| "af_bella".into());
    let worker = arg(&args, "--diffsinger-worker").map(PathBuf::from);
    let ratings = arg(&args, "--ratings").map(PathBuf::from);
    let reference_manifest = arg(&args, "--reference-manifest").map(PathBuf::from);
    let minimum_listeners: usize = arg(&args, "--minimum-listeners")
        .as_deref()
        .unwrap_or("5")
        .parse()
        .context("invalid --minimum-listeners")?;
    let source_revision = source_revision(&args)?;
    let blind_output = output.join("blind");
    let private_output = output.join("private");
    std::fs::create_dir_all(&blind_output)?;
    std::fs::create_dir_all(&private_output)?;
    let salt = blind_salt(&args, &private_output)?;

    let mut kokoro = KokoroSingingEngine::load(&voice)?;
    let mut formant = FormantSingingEngine::default();
    let mut diffsinger = worker.as_deref().map(DiffSingerEngine::spawn).transpose()?;
    let mut clips = Vec::new();
    let mut key = Vec::new();
    let mut objective = Vec::new();
    let mut case_receipts = Vec::new();

    let cases = cases();
    for (case_id, case) in cases.iter().enumerate() {
        let intent = MusicalIntent {
            valence: case.valence,
            arousal: case.arousal,
            energy: 0.55,
            bars: 2,
            seed: case_id as u64 + 9001,
            ..Default::default()
        };
        let state = MusicalState {
            valence: case.valence,
            arousal: case.arousal,
            consciousness_level: 0.72,
            ..Default::default()
        };
        let mut melody =
            symthaea_muse::theory_realize::compose_and_perform_melody(&intent, case.style, &state);
        let syllables = symthaea_muse::singing_bridge::syllable_count(case.lyrics).max(1);
        let wanted_notes = if case.challenge == "melisma" {
            syllables + 3
        } else {
            syllables
        };
        melody.truncate(wanted_notes);
        let ratio = 2.0f32.powf(case.transpose_semitones / 12.0);
        melody.iter_mut().for_each(|note| note.frequency *= ratio);
        let performance = VocalPerformance::from_melody(case.lyrics, &melody, "en")?;
        case_receipts.push(CaseReceipt {
            case_id,
            seed: case_id as u64 + 9001,
            lyrics: case.lyrics.to_string(),
            challenge: case.challenge.to_string(),
            category: case.category.to_string(),
            style: format!("{:?}", case.style),
            valence: case.valence,
            arousal: case.arousal,
            composition_energy: 0.55,
            bars: 2,
            consciousness_level: 0.72,
            transpose_semitones: case.transpose_semitones,
            score_blake3: score_digest(&melody)?,
            melody: melody.clone(),
            performance_canonical_json: serde_json::to_string(&performance)?,
            performance_blake3: performance_digest(&performance)?,
        });
        render_one(
            &mut kokoro,
            &performance,
            case_id,
            case,
            &salt,
            &blind_output,
            &mut clips,
            &mut key,
            &mut objective,
        )?;
        render_one(
            &mut formant,
            &performance,
            case_id,
            case,
            &salt,
            &blind_output,
            &mut clips,
            &mut key,
            &mut objective,
        )?;
        if let Some(engine) = diffsinger.as_mut() {
            render_one(
                engine,
                &performance,
                case_id,
                case,
                &salt,
                &blind_output,
                &mut clips,
                &mut key,
                &mut objective,
            )?;
        }
    }
    let run_manifest = RunManifest {
        schema: "muse-vocal-run/v1",
        source_revision,
        voice_id: voice.clone(),
        diffsinger_worker: worker.as_ref().map(|path| path.display().to_string()),
        diffsinger_worker_blake3: worker
            .as_ref()
            .map(|path| artifact_digest(path))
            .transpose()?,
        case_count: case_receipts.len(),
        cases: case_receipts.clone(),
    };
    std::fs::write(
        private_output.join("run_manifest.json"),
        serde_json::to_vec_pretty(&run_manifest)?,
    )?;

    let mut human_reference_receipts = Vec::new();
    if let Some(manifest) = reference_manifest {
        human_reference_receipts = import_references(
            &manifest,
            &case_receipts,
            &salt,
            &blind_output,
            &mut clips,
            &mut key,
        )?;
    }
    std::fs::write(
        private_output.join("human_reference_receipts.json"),
        serde_json::to_vec_pretty(&human_reference_receipts)?,
    )?;

    // Deterministic IDs conceal the backend; sort by ID so presentation order
    // does not reveal alternating render order.
    clips.sort_by(|a, b| a.clip_id.cmp(&b.clip_id));
    std::fs::write(
        blind_output.join("blind_manifest.json"),
        serde_json::to_vec_pretty(&clips)?,
    )?;
    std::fs::write(
        private_output.join("private_backend_key.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "format_version": 2, "salt": salt, "entries": key
        }))?,
    )?;
    std::fs::write(
        private_output.join("objective_report.json"),
        serde_json::to_vec_pretty(&objective)?,
    )?;

    let rating_template: Vec<_> = clips.iter().map(|c| serde_json::json!({
        "listener_id": null, "clip_id": c.clip_id, "naturalness": null,
        "emotional_fit": null, "identity_consistency": null, "artifact_free": null,
        "lyric_words_understood": null, "lyric_words_total": c.lyrics.split_whitespace().count()
    })).collect();
    std::fs::write(
        blind_output.join("ratings.template.json"),
        serde_json::to_vec_pretty(&rating_template)?,
    )?;

    let objective_pass = objective.iter().all(|entry| entry.metrics.objective_pass);
    let mut status = serde_json::json!({
        "objective_pass": objective_pass,
        "human_complete": false,
        "release_pass": false,
        "required": {"minimum_listeners": minimum_listeners, "naturalness_mean": 4.0,
                     "emotional_fit_mean": 4.0, "identity_consistency_mean": 4.0,
                     "artifact_free_mean": 4.0, "word_comprehension": 0.95}
    });
    if let Some(path) = ratings {
        let values: Vec<HumanRating> = serde_json::from_slice(&std::fs::read(path)?)?;
        let coverage = validate_rating_inputs(&values, &clips, minimum_listeners)?;
        let backend_by_id: BTreeMap<_, _> = key
            .iter()
            .map(|entry| (entry.clip_id.as_str(), entry.backend.as_str()))
            .collect();
        let listeners: std::collections::BTreeSet<_> = values
            .iter()
            .map(|rating| rating.listener_id.clone())
            .collect();
        let complete = coverage.complete;
        let model_values: Vec<_> = values
            .iter()
            .filter(|rating| backend_by_id.get(rating.clip_id.as_str()) != Some(&"human-reference"))
            .collect();
        let reference_values: Vec<_> = values
            .iter()
            .filter(|rating| backend_by_id.get(rating.clip_id.as_str()) == Some(&"human-reference"))
            .collect();
        let naturalness = if model_values.is_empty() {
            0.0
        } else {
            model_values.iter().map(|r| r.naturalness).sum::<f32>() / model_values.len() as f32
        };
        let mean = |f: fn(&HumanRating) -> f32| {
            model_values.iter().map(|r| f(r)).sum::<f32>() / model_values.len().max(1) as f32
        };
        let emotional_fit = mean(|r| r.emotional_fit);
        let identity_consistency = mean(|r| r.identity_consistency);
        let artifact_free = mean(|r| r.artifact_free);
        let understood: usize = model_values.iter().map(|r| r.lyric_words_understood).sum();
        let total: usize = model_values.iter().map(|r| r.lyric_words_total).sum();
        let comprehension = understood as f32 / total.max(1) as f32;
        status = serde_json::json!({
            "objective_pass": objective_pass, "human_complete": complete,
            "listener_count": listeners.len(), "rating_coverage": coverage, "naturalness_mean": naturalness,
            "human_reference_rating_count": reference_values.len(),
            "emotional_fit_mean": emotional_fit, "identity_consistency_mean": identity_consistency,
            "artifact_free_mean": artifact_free, "word_comprehension": comprehension,
            "release_pass": objective_pass && complete && naturalness >= 4.0
                && emotional_fit >= 4.0 && identity_consistency >= 4.0
                && artifact_free >= 4.0 && comprehension >= 0.95
        });
    }
    std::fs::write(
        private_output.join("release_status.json"),
        serde_json::to_vec_pretty(&status)?,
    )?;
    println!("{}", serde_json::to_string_pretty(&status)?);
    eprintln!("Blind listener bundle: {}", blind_output.display());
    eprintln!("Private evaluator receipts: {}", private_output.display());
    eprintln!("Share only the blind/ directory with listeners.");
    Ok(())
}

#[cfg(test)]
mod rating_validation_tests {
    use super::*;

    fn case_receipt() -> CaseReceipt {
        let melody = vec![symthaea_muse::Note {
            frequency: 220.0,
            start_time: 0.0,
            duration: 0.5,
            velocity: 0.7,
        }];
        CaseReceipt {
            case_id: 1,
            seed: 9002,
            lyrics: "hello world".to_string(),
            challenge: "baseline".to_string(),
            category: "center_legato".to_string(),
            style: "Classical".to_string(),
            valence: 0.5,
            arousal: 0.5,
            composition_energy: 0.55,
            bars: 2,
            consciousness_level: 0.72,
            transpose_semitones: 0.0,
            score_blake3: score_digest(&melody).unwrap(),
            melody,
            performance_canonical_json: r#"{"synthetic_fixture":true}"#.to_string(),
            performance_blake3: blake3::hash(br#"{"synthetic_fixture":true}"#)
                .to_hex()
                .to_string(),
        }
    }

    fn human_reference(case: &CaseReceipt) -> ReferenceClip {
        ReferenceClip {
            case_id: case.case_id,
            audio: PathBuf::from("licensed-human-reference.wav"),
            lyrics: case.lyrics.clone(),
            category: case.category.clone(),
            challenge: case.challenge.clone(),
            score_blake3: case.score_blake3.clone(),
            singer_pseudonym: "singer-001".to_string(),
            consent_reference: "consent:study-001".to_string(),
            license_reference: "license:study-001".to_string(),
        }
    }

    #[test]
    fn human_reference_requires_exact_lyric_score_and_rights_receipts() {
        let case = case_receipt();
        let reference = human_reference(&case);
        validate_reference_case(&reference, &case).unwrap();

        let mut wrong_score = human_reference(&case);
        wrong_score.score_blake3 = "0".repeat(64);
        assert!(
            validate_reference_case(&wrong_score, &case)
                .unwrap_err()
                .to_string()
                .contains("score digest")
        );

        let mut wrong_lyrics = human_reference(&case);
        wrong_lyrics.lyrics = "different lyrics".into();
        assert!(
            validate_reference_case(&wrong_lyrics, &case)
                .unwrap_err()
                .to_string()
                .contains("lyrics/category/challenge")
        );

        let mut no_consent = human_reference(&case);
        no_consent.consent_reference.clear();
        assert!(
            validate_reference_case(&no_consent, &case)
                .unwrap_err()
                .to_string()
                .contains("consent reference")
        );
    }

    fn clip(id: &str, lyrics: &str) -> BlindClip {
        BlindClip {
            clip_id: id.to_string(),
            case_id: 0,
            lyrics: lyrics.to_string(),
            audio: format!("{id}.wav"),
            audio_blake3: "audio-digest".to_string(),
            performance_blake3: Some("performance-digest".to_string()),
            category: "test".to_string(),
            challenge: "test".to_string(),
        }
    }

    fn rating(clip_id: &str, listener_id: &str) -> HumanRating {
        HumanRating {
            listener_id: listener_id.to_string(),
            clip_id: clip_id.to_string(),
            naturalness: 4.0,
            emotional_fit: 4.0,
            identity_consistency: 4.0,
            artifact_free: 4.0,
            lyric_words_understood: 2,
            lyric_words_total: 2,
        }
    }

    #[test]
    fn balanced_known_ratings_are_complete() {
        let clips = vec![clip("a", "hello world"), clip("b", "hello world")];
        let ratings = vec![
            rating("a", "listener-1"),
            rating("b", "listener-1"),
            rating("a", "listener-2"),
            rating("b", "listener-2"),
        ];
        let result = validate_rating_inputs(&ratings, &clips, 2).unwrap();
        assert!(result.complete);
        assert_eq!(result.rating_count, 4);
        assert_eq!(result.listener_count, 2);
    }

    #[test]
    fn unknown_clip_ids_cannot_contaminate_human_means() {
        let clips = vec![clip("a", "hello world")];
        let ratings = vec![rating("a", "listener-1"), rating("ghost", "listener-1")];
        let error = validate_rating_inputs(&ratings, &clips, 1)
            .unwrap_err()
            .to_string();
        assert!(error.contains("unknown clip ID"), "{error}");
    }

    #[test]
    fn lyric_comprehension_denominator_must_match_manifest_lyrics() {
        let clips = vec![clip("a", "hello world")];
        let mut invalid = rating("a", "listener-1");
        invalid.lyric_words_total = 20;
        let error = validate_rating_inputs(&[invalid], &clips, 1)
            .unwrap_err()
            .to_string();
        assert!(error.contains("denominator"), "{error}");
    }

    #[test]
    fn non_finite_and_duplicate_ratings_are_rejected() {
        let clips = vec![clip("a", "hello world")];
        let mut nan_rating = rating("a", "listener-1");
        nan_rating.naturalness = f32::NAN;
        assert!(validate_rating_inputs(&[nan_rating], &clips, 1).is_err());

        let duplicate = vec![rating("a", "listener-1"), rating("a", "listener-1")];
        assert!(validate_rating_inputs(&duplicate, &clips, 1).is_err());
    }

    #[test]
    fn incomplete_rating_matrix_never_counts_as_complete() {
        let clips = vec![clip("a", "hello world"), clip("b", "hello world")];
        let ratings = vec![rating("a", "listener-1")];
        let result = validate_rating_inputs(&ratings, &clips, 1).unwrap();
        assert!(!result.complete);
    }
    #[test]
    fn source_revision_requires_exact_git_sha() {
        let valid = "0123456789abcdef0123456789abcdef01234567";
        assert_eq!(validate_source_revision(valid).unwrap(), valid);
        assert_eq!(
            validate_source_revision("ABCDEF0123456789ABCDEF0123456789ABCDEF01").unwrap(),
            "abcdef0123456789abcdef0123456789abcdef01"
        );
        assert!(validate_source_revision("queued").is_err());
        assert!(validate_source_revision("0123456789abcdef").is_err());
        assert!(validate_source_revision("z123456789abcdef0123456789abcdef01234567").is_err());
    }
}
