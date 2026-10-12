// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Blind, backend-comparative Muse vocal release gate.

use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use symthaea::voice::diffsinger::DiffSingerEngine;
use symthaea::voice::functional_singing::FunctionalSingingEngine;
use symthaea::voice::kokoro_singing::KokoroSingingEngine;
use symthaea::voice::singing_engine::{SingingVoiceEngine, VocalPerformance, VocalStem};
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
    category: String,
    challenge: String,
}

/// Listener-facing metadata only. Keep experimental labels and source case IDs
/// out of the shareable manifest to reduce expectancy bias during rating.
#[derive(Serialize)]
struct ListeningClip<'a> {
    clip_id: &'a str,
    lyrics: &'a str,
    audio: &'a str,
}

#[derive(Serialize)]
struct KeyEntry {
    clip_id: String,
    backend: String,
    case_id: usize,
}

#[derive(Serialize)]
struct ObjectiveEntry {
    clip_id: String,
    backend: String,
    case_id: usize,
    metrics: VocalQualityMetrics,
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
}

fn arg(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|x| x == flag)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

/// Keep organizer-only mappings and score reports outside the shareable
/// listening-pack directory. Share `--output`; keep this sibling private.
fn private_output_dir(output: &Path) -> Result<PathBuf> {
    let name = output
        .file_name()
        .context("--output must identify a listening-pack directory name")?;
    Ok(output.with_file_name(format!("{}.private", name.to_string_lossy())))
}

/// Check directory entries without following symlinks (including dangling ones).
fn path_entry_exists(path: &Path) -> Result<bool> {
    match std::fs::symlink_metadata(path) {
        Ok(_) => Ok(true),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error.into()),
    }
}

/// Migrate legacy organizer-only artifacts out of older listening packs.
/// Never overwrite prior private evidence; move colliding legacy files under a
/// distinct archived name so the shareable directory no longer contains them.
fn relocate_legacy_private_artifacts(output: &Path, private_output: &Path) -> Result<()> {
    for name in [
        "private_backend_key.json",
        "objective_report.json",
        "release_status.json",
    ] {
        let legacy_path = output.join(name);
        if !path_entry_exists(&legacy_path)? {
            continue;
        }
        let preferred_path = private_output.join(name);
        let private_path = if !path_entry_exists(&preferred_path)? {
            preferred_path
        } else {
            let mut suffix = 0usize;
            loop {
                let candidate = private_output.join(format!("{name}.legacy-{suffix}"));
                if !path_entry_exists(&candidate)? {
                    break candidate;
                }
                suffix = suffix
                    .checked_add(1)
                    .context("legacy private artifact suffix overflow")?;
            }
        };
        std::fs::rename(&legacy_path, &private_path).with_context(|| {
            format!(
                "failed to move legacy private artifact {} outside the listening pack",
                legacy_path.display()
            )
        })?;
    }
    Ok(())
}

/// Require the two output trees to be physically separate and restrict access
/// to organizer-only evidence on Unix systems.
fn ensure_private_output_is_separate(output: &Path, private_output: &Path) -> Result<()> {
    let public_metadata =
        std::fs::symlink_metadata(output).context("failed to inspect listening-pack directory")?;
    anyhow::ensure!(
        !public_metadata.file_type().is_symlink(),
        "listening-pack output must not be a symlink"
    );
    let private_metadata = std::fs::symlink_metadata(private_output)
        .context("failed to inspect organizer-only output directory")?;
    anyhow::ensure!(
        !private_metadata.file_type().is_symlink(),
        "organizer-only output must not be a symlink"
    );
    let public_path = output.canonicalize()?;
    let private_path = private_output.canonicalize()?;
    anyhow::ensure!(
        public_path != private_path
            && !public_path.starts_with(&private_path)
            && !private_path.starts_with(&public_path),
        "--output and organizer-only output must resolve to separate, non-nested directories"
    );
    restrict_private_output_permissions(private_output)?;
    Ok(())
}

#[cfg(unix)]
fn restrict_private_output_permissions(path: &Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700))?;
    Ok(())
}

#[cfg(not(unix))]
fn restrict_private_output_permissions(_path: &Path) -> Result<()> {
    // The private directory remains a sibling and is never included in the
    // shareable pack. Platform-specific ACL configuration is left to the host.
    Ok(())
}

/// Never mix stale WAVs or manifests into a fresh blinded run.
fn ensure_empty_listening_pack(output: &Path) -> Result<()> {
    let mut entries = std::fs::read_dir(output)
        .with_context(|| format!("failed to inspect listening pack {}", output.display()))?;
    anyhow::ensure!(
        entries.next().transpose()?.is_none(),
        "listening-pack directory {} is not empty; archive the previous run and choose a fresh --output path",
        output.display()
    );
    Ok(())
}

/// Never overwrite old private evidence, which may belong to another blind key.
fn ensure_empty_private_output(private_output: &Path) -> Result<()> {
    let mut entries = std::fs::read_dir(private_output).with_context(|| {
        format!(
            "failed to inspect organizer-only output {}",
            private_output.display()
        )
    })?;
    anyhow::ensure!(
        entries.next().transpose()?.is_none(),
        "organizer-only output {} is not empty; archive prior evidence and choose a fresh --output path",
        private_output.display()
    );
    Ok(())
}

fn save(path: &Path, stem: &VocalStem) -> Result<()> {
    let path = path.to_str().context("non-UTF8 output path")?;
    symthaea::voice::save_wav(&stem.samples, stem.sample_rate, path)
}

fn blind_id(salt: &str, backend: &str, case_id: usize) -> String {
    let digest = blake3::hash(format!("muse-vocal-v2:{salt}:{backend}:{case_id}").as_bytes());
    format!("vocal-{}", &digest.to_hex()[..12])
}

fn validate_human_rating(rating: &HumanRating, clips: &[BlindClip]) -> Result<()> {
    if rating.listener_id.trim().is_empty() {
        anyhow::bail!("listener_id must be non-empty");
    }
    if !(1.0..=5.0).contains(&rating.naturalness)
        || !(1.0..=5.0).contains(&rating.emotional_fit)
        || !(1.0..=5.0).contains(&rating.identity_consistency)
        || !(1.0..=5.0).contains(&rating.artifact_free)
    {
        anyhow::bail!("rating scores must be within 1..=5");
    }
    if rating.lyric_words_understood > rating.lyric_words_total {
        anyhow::bail!("understood word count exceeds total for {}", rating.clip_id);
    }
    let clip = clips
        .iter()
        .find(|clip| clip.clip_id == rating.clip_id)
        .with_context(|| format!("rating references unknown clip_id {}", rating.clip_id))?;
    let expected_words = clip.lyrics.split_whitespace().count();
    if expected_words == 0 || rating.lyric_words_total != expected_words {
        anyhow::bail!(
            "lyric_words_total mismatch for {}: expected {}, got {}",
            rating.clip_id,
            expected_words,
            rating.lyric_words_total
        );
    }
    Ok(())
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
    let stem = engine.render(performance)?;
    // Enforce the shared integrity contract at the evaluator boundary too;
    // a custom/backend implementation must not save malformed audio for ratings.
    stem.validate()?;
    let id = blind_id(salt, &stem.backend, case_id);
    let filename = format!("{id}.wav");
    save(&output.join(&filename), &stem)?;
    let metrics = analyze_vocal_stem(performance, &stem);
    clips.push(BlindClip {
        clip_id: id.clone(),
        case_id,
        lyrics: performance.lyrics.clone(),
        audio: filename,
        category: case.category.to_string(),
        challenge: case.challenge.to_string(),
    });
    key.push(KeyEntry {
        clip_id: id.clone(),
        backend: stem.backend.clone(),
        case_id,
    });
    objective.push(ObjectiveEntry {
        clip_id: id,
        backend: stem.backend,
        case_id,
        metrics,
    });
    Ok(())
}

fn import_references(
    manifest: &Path,
    salt: &str,
    output: &Path,
    clips: &mut Vec<BlindClip>,
    key: &mut Vec<KeyEntry>,
) -> Result<()> {
    let references: Vec<ReferenceClip> = serde_json::from_slice(&std::fs::read(manifest)?)?;
    for reference in references {
        if !reference.audio.is_file() {
            anyhow::bail!("missing human reference: {}", reference.audio.display());
        }
        let id = blind_id(salt, "human-reference", reference.case_id);
        if clips.iter().any(|clip| clip.clip_id == id) {
            anyhow::bail!("duplicate human-reference case_id {}", reference.case_id);
        }
        let filename = format!("{id}.wav");
        std::fs::copy(&reference.audio, output.join(&filename))?;
        clips.push(BlindClip {
            clip_id: id.clone(),
            case_id: reference.case_id,
            lyrics: reference.lyrics,
            audio: filename,
            category: reference.category,
            challenge: reference.challenge,
        });
        key.push(KeyEntry {
            clip_id: id,
            backend: "human-reference".to_string(),
            case_id: reference.case_id,
        });
    }
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let output = PathBuf::from(
        arg(&args, "--output").unwrap_or_else(|| "audio_output/muse_vocal_gate".into()),
    );
    let private_output = private_output_dir(&output)?;
    let voice = arg(&args, "--voice").unwrap_or_else(|| "af_bella".into());
    let worker = arg(&args, "--diffsinger-worker").map(PathBuf::from);
    let ratings = arg(&args, "--ratings").map(PathBuf::from);
    let reference_manifest = arg(&args, "--reference-manifest").map(PathBuf::from);
    let minimum_listeners: usize = arg(&args, "--minimum-listeners")
        .as_deref()
        .unwrap_or("5")
        .parse()
        .context("invalid --minimum-listeners")?;
    anyhow::ensure!(minimum_listeners > 0, "--minimum-listeners must be positive");
    std::fs::create_dir_all(&output)?;
    std::fs::create_dir_all(&private_output)?;
    ensure_private_output_is_separate(&output, &private_output)?;
    relocate_legacy_private_artifacts(&output, &private_output)?;
    ensure_empty_private_output(&private_output)?;
    ensure_empty_listening_pack(&output)?;
    let salt = blind_salt(&args, &output)?;

    let mut kokoro = KokoroSingingEngine::load(&voice)?;
    let mut functional = FunctionalSingingEngine::default();
    let mut diffsinger = worker.as_deref().map(DiffSingerEngine::spawn).transpose()?;
    let mut clips = Vec::new();
    let mut key = Vec::new();
    let mut objective = Vec::new();

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
        render_one(
            &mut kokoro,
            &performance,
            case_id,
            case,
            &salt,
            &output,
            &mut clips,
            &mut key,
            &mut objective,
        )?;
        render_one(
            &mut functional,
            &performance,
            case_id,
            case,
            &salt,
            &output,
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
                &output,
                &mut clips,
                &mut key,
                &mut objective,
            )?;
        }
    }
    if let Some(manifest) = reference_manifest {
        import_references(&manifest, &salt, &output, &mut clips, &mut key)?;
    }

    // Deterministic IDs conceal the backend; sort by ID so presentation order
    // does not reveal alternating render order.
    clips.sort_by(|a, b| a.clip_id.cmp(&b.clip_id));
    let listening_manifest: Vec<_> = clips
        .iter()
        .map(|clip| ListeningClip {
            clip_id: &clip.clip_id,
            lyrics: &clip.lyrics,
            audio: &clip.audio,
        })
        .collect();
    std::fs::write(
        output.join("blind_manifest.json"),
        serde_json::to_vec_pretty(&listening_manifest)?,
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
        output.join("ratings.template.json"),
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
        let backend_by_id: BTreeMap<_, _> = key
            .iter()
            .map(|entry| (entry.clip_id.as_str(), entry.backend.as_str()))
            .collect();
        let listeners: std::collections::BTreeSet<_> =
            values.iter().map(|r| r.listener_id.clone()).collect();
        let mut by_clip_listener = BTreeMap::new();
        for rating in &values {
            validate_human_rating(rating, &clips)?;
            if by_clip_listener
                .insert((rating.clip_id.clone(), rating.listener_id.clone()), ())
                .is_some()
            {
                anyhow::bail!("duplicate listener/clip rating");
            }
        }
        let complete = listeners.len() >= minimum_listeners
            && clips.iter().all(|c| {
                listeners.iter().all(|listener| {
                    by_clip_listener.contains_key(&(c.clip_id.clone(), listener.clone()))
                })
            });
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
            "listener_count": listeners.len(), "naturalness_mean": naturalness,
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
    eprintln!(
        "organizer-only artifacts (backend key, objective report, release status): {}",
        private_output.display()
    );
    println!("{}", serde_json::to_string_pretty(&status)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clip() -> BlindClip {
        BlindClip {
            clip_id: "clip-1".into(),
            case_id: 7,
            lyrics: "softly now stars".into(),
            audio: "clip-1.wav".into(),
            category: "center".into(),
            challenge: "legato".into(),
        }
    }

    fn rating(listener_id: &str, clip_id: &str, total: usize) -> HumanRating {
        HumanRating {
            listener_id: listener_id.into(),
            clip_id: clip_id.into(),
            naturalness: 4.0,
            emotional_fit: 4.0,
            identity_consistency: 4.0,
            artifact_free: 4.0,
            lyric_words_understood: total,
            lyric_words_total: total,
        }
    }

    #[test]
    fn private_output_is_separate_and_owner_only_on_unix() {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "symthaea-vocal-private-output-{}-{nonce}",
            std::process::id()
        ));
        let public = root.join("pack");
        let private = root.join("pack.private");
        std::fs::create_dir_all(&public).unwrap();
        std::fs::create_dir_all(&private).unwrap();
        assert!(ensure_private_output_is_separate(&public, &private).is_ok());
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&private).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode, 0o700);
        }
        std::fs::remove_dir_all(root).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn private_output_rejects_symlinks() {
        use std::os::unix::fs::symlink;
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "symthaea-vocal-private-symlink-{}-{nonce}",
            std::process::id()
        ));
        let public = root.join("pack");
        let actual_private = root.join("actual-private");
        let private_link = root.join("pack.private");
        std::fs::create_dir_all(&public).unwrap();
        std::fs::create_dir_all(&actual_private).unwrap();
        symlink(&actual_private, &private_link).unwrap();
        assert!(ensure_private_output_is_separate(&public, &private_link).is_err());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn legacy_private_artifact_collision_migrates_without_overwriting() {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "symthaea-vocal-legacy-migration-{}-{nonce}",
            std::process::id()
        ));
        let public = root.join("pack");
        let private = root.join("pack.private");
        std::fs::create_dir_all(&public).unwrap();
        std::fs::create_dir_all(&private).unwrap();
        std::fs::write(public.join("private_backend_key.json"), b"legacy-key").unwrap();
        std::fs::write(private.join("private_backend_key.json"), b"existing-key").unwrap();
        std::fs::write(public.join("objective_report.json"), b"legacy-report").unwrap();

        relocate_legacy_private_artifacts(&public, &private).unwrap();

        assert!(!public.join("private_backend_key.json").exists());
        assert!(!public.join("objective_report.json").exists());
        assert_eq!(
            std::fs::read(private.join("private_backend_key.json")).unwrap(),
            b"existing-key"
        );
        assert_eq!(
            std::fs::read(private.join("private_backend_key.json.legacy-0")).unwrap(),
            b"legacy-key"
        );
        assert_eq!(
            std::fs::read(private.join("objective_report.json")).unwrap(),
            b"legacy-report"
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn existing_listening_pack_is_rejected_to_prevent_stale_clip_mixing() {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let output = std::env::temp_dir().join(format!(
            "symthaea-vocal-listening-pack-{}-{nonce}",
            std::process::id()
        ));
        std::fs::create_dir_all(&output).unwrap();
        assert!(ensure_empty_listening_pack(&output).is_ok());
        std::fs::write(output.join("stale.wav"), b"not-a-valid-wav").unwrap();
        assert!(ensure_empty_listening_pack(&output).is_err());
        std::fs::remove_dir_all(&output).unwrap();
    }

    #[test]
    fn listener_manifest_does_not_expose_experimental_condition_labels() {
        let clip = clip();
        let manifest = ListeningClip {
            clip_id: &clip.clip_id,
            lyrics: &clip.lyrics,
            audio: &clip.audio,
        };
        let value = serde_json::to_value(manifest).unwrap();
        assert!(value.get("clip_id").is_some());
        assert!(value.get("lyrics").is_some());
        assert!(value.get("audio").is_some());
        for private_field in ["case_id", "category", "challenge", "backend"] {
            assert!(
                value.get(private_field).is_none(),
                "listener manifest leaked {private_field}"
            );
        }
    }

    #[test]
    fn private_artifacts_are_written_to_a_sibling_of_the_listening_pack() {
        let output = Path::new("audio_output/muse_vocal_gate");
        let private = private_output_dir(output).unwrap();
        assert_eq!(
            private,
            PathBuf::from("audio_output/muse_vocal_gate.private")
        );
        assert!(!private.starts_with(output));
    }

    #[test]
    fn human_rating_rejects_unknown_clip_id() {
        let error = validate_human_rating(&rating("listener-1", "invented-clip", 3), &[clip()]);
        assert!(error.is_err());
    }

    #[test]
    fn human_rating_rejects_blank_listener_and_word_count_mismatch() {
        let clips = [clip()];
        assert!(validate_human_rating(&rating("  ", "clip-1", 3), &clips).is_err());
        assert!(validate_human_rating(&rating("listener-1", "clip-1", 2), &clips).is_err());
    }

    #[test]
    fn human_rating_accepts_manifest_aligned_rating() {
        assert!(validate_human_rating(&rating("listener-1", "clip-1", 3), &[clip()]).is_ok());
    }
}
