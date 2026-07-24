// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Batch Processing Module
//!
//! Parallel processing of multiple audio files for:
//! - High-throughput training on large datasets
//! - Batch transcription of audio archives
//! - Parallel acoustic discovery
//!
//! Uses work-stealing thread pool for efficient CPU utilization.

use crate::audio::{AudioConfig, AudioFrontend, AudioProjector};
use crate::bootstrap::{AdaptivePrototypeSet, AdaptiveStats, BootstrapConfig, TrainedPrototypes};
use crate::discovery::{DiscoveryConfig, DiscoveryPipeline, DiscoveryResult};
use crate::hdc::HV16;
use crate::lexicon::{CmuDictionary, TextToPhonemes};
use crate::ltc::LtcConfig;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};
use std::thread;

/// Batch processing configuration
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Number of worker threads (0 = auto-detect)
    pub num_workers: usize,
    /// Maximum files to process
    pub max_files: Option<usize>,
    /// Shuffle file order
    pub shuffle: bool,
    /// Continue on error
    pub continue_on_error: bool,
    /// Progress callback interval (files)
    pub progress_interval: usize,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            num_workers: 0, // Auto-detect
            max_files: None,
            shuffle: true,
            continue_on_error: true,
            progress_interval: 10,
        }
    }
}

impl BatchConfig {
    /// Get actual number of workers
    pub fn workers(&self) -> usize {
        if self.num_workers == 0 {
            thread::available_parallelism()
                .map(|p| p.get())
                .unwrap_or(4)
        } else {
            self.num_workers
        }
    }
}

/// Result from processing a single file
#[derive(Debug, Clone)]
pub struct FileResult {
    /// File path
    pub path: PathBuf,
    /// Success or error message
    pub status: Result<FileSuccess, String>,
    /// Processing time (seconds)
    pub duration_sec: f32,
}

#[derive(Debug, Clone)]
pub struct FileSuccess {
    /// Audio duration (seconds)
    pub audio_duration: f32,
    /// Number of frames extracted
    pub num_frames: usize,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Batch processing statistics
#[derive(Debug, Clone, Default)]
pub struct BatchStats {
    /// Total files processed
    pub files_processed: usize,
    /// Successful files
    pub files_success: usize,
    /// Failed files
    pub files_failed: usize,
    /// Total audio duration (seconds)
    pub total_audio_sec: f32,
    /// Total processing time (seconds)
    pub total_time_sec: f32,
    /// Realtime factor (audio_duration / processing_time)
    pub realtime_factor: f32,
}

/// Progress callback type
pub type ProgressCallback = Box<dyn Fn(usize, usize, &str) + Send + Sync>;

/// Batch audio processor
pub struct BatchProcessor {
    config: BatchConfig,
    audio_config: AudioConfig,
    ltc_config: LtcConfig,
}

impl BatchProcessor {
    /// Create a new batch processor
    pub fn new(config: BatchConfig) -> Self {
        Self {
            config,
            audio_config: AudioConfig::default(),
            ltc_config: LtcConfig::default(),
        }
    }

    /// Set audio configuration
    pub fn with_audio_config(mut self, config: AudioConfig) -> Self {
        self.audio_config = config;
        self
    }

    /// Set LTC configuration
    pub fn with_ltc_config(mut self, config: LtcConfig) -> Self {
        self.ltc_config = config;
        self
    }

    /// Process multiple files and extract features
    pub fn extract_features(
        &self,
        files: &[PathBuf],
        callback: Option<ProgressCallback>,
    ) -> (Vec<FileResult>, BatchStats) {
        let num_workers = self.config.workers();
        let total_files = self
            .config
            .max_files
            .unwrap_or(files.len())
            .min(files.len());

        // Prepare file list
        let mut work_files: Vec<PathBuf> = files.iter().take(total_files).cloned().collect();

        if self.config.shuffle {
            // Simple shuffle using hash
            work_files.sort_by_key(|p| {
                let hash = blake3::hash(p.to_string_lossy().as_bytes());
                u64::from_le_bytes(hash.as_bytes()[..8].try_into().unwrap_or([0u8; 8]))
            });
        }

        // Shared state
        let results = Arc::new(Mutex::new(Vec::with_capacity(total_files)));
        let work_queue = Arc::new(Mutex::new(
            work_files.into_iter().enumerate().collect::<Vec<_>>(),
        ));
        let processed = Arc::new(AtomicUsize::new(0));
        let callback = callback.map(Arc::new);

        let start_time = std::time::Instant::now();

        // Spawn workers
        let mut handles = Vec::with_capacity(num_workers);

        for _ in 0..num_workers {
            let work_queue = Arc::clone(&work_queue);
            let results = Arc::clone(&results);
            let processed = Arc::clone(&processed);
            let callback = callback.clone();
            let audio_config = self.audio_config.clone();
            let continue_on_error = self.config.continue_on_error;
            let progress_interval = self.config.progress_interval;
            let total = total_files;

            handles.push(thread::spawn(move || {
                // Each worker gets its own projector
                let mut projector = AudioProjector::new(audio_config, LtcConfig::default());

                loop {
                    // Get next file
                    let work = {
                        let mut queue = work_queue.lock().unwrap_or_else(|e| e.into_inner());
                        queue.pop()
                    };

                    let (_idx, path) = match work {
                        Some(w) => w,
                        None => break,
                    };

                    let file_start = std::time::Instant::now();

                    // Process file
                    let status = Self::process_file(&path, &mut projector);

                    let duration_sec = file_start.elapsed().as_secs_f32();

                    let result = FileResult {
                        path: path.clone(),
                        status,
                        duration_sec,
                    };

                    // Store result
                    {
                        let mut results = results.lock().unwrap_or_else(|e| e.into_inner());
                        results.push(result);
                    }

                    // Update progress
                    let count = processed.fetch_add(1, Ordering::SeqCst) + 1;

                    if let Some(ref cb) = callback
                        && (count.is_multiple_of(progress_interval) || count == total)
                    {
                        cb(count, total, path.to_string_lossy().as_ref());
                    }

                    // Check if we should stop on error
                    if !continue_on_error {
                        // Check last result
                        let results = results.lock().unwrap_or_else(|e| e.into_inner());
                        if let Some(last) = results.last()
                            && last.status.is_err()
                        {
                            break;
                        }
                    }
                }
            }));
        }

        // Wait for all workers
        for handle in handles {
            let _ = handle.join();
        }

        let total_time = start_time.elapsed().as_secs_f32();

        // Collect results
        let results = Arc::try_unwrap(results)
            .map(|mutex| mutex.into_inner().unwrap())
            .unwrap_or_else(|arc| arc.lock().unwrap_or_else(|e| e.into_inner()).clone());

        // Compute stats
        let mut stats = BatchStats {
            files_processed: results.len(),
            total_time_sec: total_time,
            ..Default::default()
        };

        for result in &results {
            match &result.status {
                Ok(success) => {
                    stats.files_success += 1;
                    stats.total_audio_sec += success.audio_duration;
                }
                Err(_) => {
                    stats.files_failed += 1;
                }
            }
        }

        if stats.total_time_sec > 0.0 {
            stats.realtime_factor = stats.total_audio_sec / stats.total_time_sec;
        }

        (results, stats)
    }

    /// Process a single file
    fn process_file(path: &Path, projector: &mut AudioProjector) -> Result<FileSuccess, String> {
        // Load audio
        let (audio, sample_rate) =
            AudioFrontend::load_audio(path).map_err(|e| format!("Failed to load audio: {e}"))?;

        let audio_duration = audio.len() as f32 / sample_rate as f32;

        // Project to HDC
        projector.reset();
        let hvs = projector.project(&audio);

        let mut metadata = HashMap::new();
        metadata.insert("sample_rate".to_string(), sample_rate.to_string());

        Ok(FileSuccess {
            audio_duration,
            num_frames: hvs.len(),
            metadata,
        })
    }
}

/// Batch training processor
pub struct BatchTrainer {
    config: BatchConfig,
    bootstrap_config: BootstrapConfig,
    /// Phoneme accumulators (protected by mutex)
    accumulators: Arc<Mutex<HashMap<String, PhonemeAccumulator>>>,
    /// Text to phonemes converter (optional)
    text_to_phonemes: Option<Arc<TextToPhonemes>>,
}

#[derive(Clone)]
struct PhonemeAccumulator {
    /// Sum of hypervectors (for computing mean)
    sum_counts: Vec<i32>,
    /// Number of instances
    count: usize,
}

impl PhonemeAccumulator {
    fn new() -> Self {
        Self {
            sum_counts: vec![0; 2048],
            count: 0,
        }
    }

    fn add(&mut self, hv: &HV16) {
        let bytes = hv.to_bytes();
        for (byte_idx, &byte) in bytes.iter().enumerate() {
            for bit_idx in 0..8 {
                let bit = (byte >> bit_idx) & 1;
                let pos = byte_idx * 8 + bit_idx;
                self.sum_counts[pos] += if bit == 1 { 1 } else { -1 };
            }
        }
        self.count += 1;
    }

    fn finalize(&self) -> HV16 {
        let mut bytes = [0u8; 256];
        for byte_idx in 0..256 {
            for bit_idx in 0..8 {
                let pos = byte_idx * 8 + bit_idx;
                if self.sum_counts[pos] > 0 {
                    bytes[byte_idx] |= 1 << bit_idx;
                }
            }
        }
        HV16::from_bytes(&bytes)
    }
}

impl BatchTrainer {
    /// Create a new batch trainer
    pub fn new(config: BatchConfig, bootstrap_config: BootstrapConfig) -> Self {
        Self {
            config,
            bootstrap_config,
            accumulators: Arc::new(Mutex::new(HashMap::new())),
            text_to_phonemes: None,
        }
    }

    /// Create with a pronunciation dictionary for phoneme-level training
    pub fn with_dictionary(mut self, dict: CmuDictionary) -> Self {
        self.text_to_phonemes = Some(Arc::new(TextToPhonemes::new(dict)));
        self
    }

    /// Load and set dictionary from path
    pub fn load_dictionary<P: AsRef<std::path::Path>>(&mut self, path: P) -> std::io::Result<()> {
        let dict = CmuDictionary::load(path)?;
        self.text_to_phonemes = Some(Arc::new(TextToPhonemes::new(dict)));
        Ok(())
    }

    /// Train on a batch of (audio_path, transcript) pairs
    pub fn train(
        &self,
        samples: &[(PathBuf, String)],
        callback: Option<ProgressCallback>,
    ) -> (TrainedPrototypes, BatchStats) {
        let num_workers = self.config.workers();
        let total = samples.len();

        let work_queue = Arc::new(Mutex::new(
            samples
                .iter()
                .enumerate()
                .map(|(i, (path, transcript))| (i, (path.clone(), transcript.clone())))
                .collect::<Vec<_>>(),
        ));
        let processed = Arc::new(AtomicUsize::new(0));
        let stats = Arc::new(Mutex::new(BatchStats::default()));
        let callback = callback.map(Arc::new);
        let accumulators = Arc::clone(&self.accumulators);

        let start_time = std::time::Instant::now();

        // Spawn workers
        let mut handles = Vec::with_capacity(num_workers);

        for _ in 0..num_workers {
            let work_queue = Arc::clone(&work_queue);
            let processed = Arc::clone(&processed);
            let stats = Arc::clone(&stats);
            let callback = callback.clone();
            let accumulators = Arc::clone(&accumulators);
            let _bootstrap_config = self.bootstrap_config.clone();
            let progress_interval = self.config.progress_interval;
            let text_to_phonemes = self.text_to_phonemes.clone();

            handles.push(thread::spawn(move || {
                let mut projector = AudioProjector::default_config();

                loop {
                    let work = {
                        let mut queue = work_queue.lock().unwrap_or_else(|e| e.into_inner());
                        queue.pop()
                    };

                    let (_idx, (path, transcript)) = match work {
                        Some(w) => w,
                        None => break,
                    };

                    // Process file
                    let result = Self::process_training_file(
                        &path,
                        &transcript,
                        &mut projector,
                        &accumulators,
                        text_to_phonemes.as_deref(),
                    );

                    // Update stats
                    {
                        let mut s = stats.lock().unwrap_or_else(|e| e.into_inner());
                        s.files_processed += 1;
                        match result {
                            Ok(duration) => {
                                s.files_success += 1;
                                s.total_audio_sec += duration;
                            }
                            Err(_) => {
                                s.files_failed += 1;
                            }
                        }
                    }

                    // Progress callback
                    let count = processed.fetch_add(1, Ordering::SeqCst) + 1;
                    if let Some(ref cb) = callback
                        && (count.is_multiple_of(progress_interval) || count == total)
                    {
                        cb(count, total, path.to_string_lossy().as_ref());
                    }
                }
            }));
        }

        // Wait for workers
        for handle in handles {
            let _ = handle.join();
        }

        // Finalize stats
        let mut final_stats = stats.lock().unwrap_or_else(|e| e.into_inner()).clone();
        final_stats.total_time_sec = start_time.elapsed().as_secs_f32();
        if final_stats.total_time_sec > 0.0 {
            final_stats.realtime_factor = final_stats.total_audio_sec / final_stats.total_time_sec;
        }

        // Finalize prototypes
        let accumulators = self.accumulators.lock().unwrap_or_else(|e| e.into_inner());
        let mut prototypes = TrainedPrototypes::new(self.bootstrap_config.clone());

        for (phoneme, acc) in accumulators.iter() {
            if acc.count >= self.bootstrap_config.min_instances {
                prototypes
                    .prototypes
                    .insert(phoneme.clone(), acc.finalize());
                prototypes.counts.insert(phoneme.clone(), acc.count);
            }
        }

        (prototypes, final_stats)
    }

    fn process_training_file(
        path: &Path,
        transcript: &str,
        projector: &mut AudioProjector,
        accumulators: &Arc<Mutex<HashMap<String, PhonemeAccumulator>>>,
        text_to_phonemes: Option<&TextToPhonemes>,
    ) -> Result<f32, String> {
        // Load audio
        let (audio, sample_rate) =
            AudioFrontend::load_audio(path).map_err(|e| format!("Load error: {e}"))?;

        let duration = audio.len() as f32 / sample_rate as f32;

        // Project audio
        projector.reset();
        let (hvs, _saliences) = projector.project_with_salience(&audio);

        if hvs.is_empty() {
            return Ok(duration);
        }

        // Get labels (phonemes if dictionary available, otherwise words)
        let labels: Vec<String> = if let Some(ttp) = text_to_phonemes {
            // Convert to phonemes
            ttp.convert(transcript)
                .into_iter()
                .filter(|p| p != "_" && p != "SIL" && !p.starts_with('<'))
                .collect()
        } else {
            // Fall back to words
            transcript
                .split_whitespace()
                .map(|w| w.to_lowercase())
                .collect()
        };

        if labels.is_empty() {
            return Ok(duration);
        }

        // Distribute frames across labels (simple uniform alignment)
        let frames_per_label = hvs.len() / labels.len();
        if frames_per_label == 0 {
            return Ok(duration);
        }

        // For each label, add REPRESENTATIVE frames to accumulator
        // Too many frames causes bundling collapse - majority vote converges to average
        // Use only the MIDDLE frame of each segment as the representative
        for (label_idx, label) in labels.iter().enumerate() {
            let start = label_idx * frames_per_label;
            let end = ((label_idx + 1) * frames_per_label).min(hvs.len());

            if start >= hvs.len() || start >= end {
                break;
            }

            // Use MIDDLE frame as representative (most stable part of phoneme)
            let mid = (start + end) / 2;
            if mid < hvs.len() {
                let mut acc = accumulators.lock().unwrap_or_else(|e| e.into_inner());
                acc.entry(label.clone())
                    .or_insert_with(PhonemeAccumulator::new)
                    .add(&hvs[mid]);
            }
        }

        Ok(duration)
    }

    /// Get current phoneme counts
    pub fn get_counts(&self) -> HashMap<String, usize> {
        self.accumulators
            .lock()
            .unwrap()
            .iter()
            .map(|(k, v)| (k.clone(), v.count))
            .collect()
    }

    /// Train with adaptive allophone clustering
    ///
    /// Unlike `train()`, this method uses dynamic clustering to prevent
    /// "Grey Goo" collapse where high-frequency phonemes converge.
    pub fn train_adaptive(
        &self,
        samples: &[(PathBuf, String)],
        threshold: f32,
        max_variants: usize,
        callback: Option<ProgressCallback>,
    ) -> (AdaptivePrototypeSet, BatchStats, AdaptiveStats) {
        let num_workers = self.config.workers();
        let total = samples.len();

        let work_queue = Arc::new(Mutex::new(
            samples
                .iter()
                .enumerate()
                .map(|(i, (path, transcript))| (i, (path.clone(), transcript.clone())))
                .collect::<Vec<_>>(),
        ));
        let processed = Arc::new(AtomicUsize::new(0));
        let stats = Arc::new(Mutex::new(BatchStats::default()));
        let callback = callback.map(Arc::new);

        // Use AdaptivePrototypeSet instead of naive accumulators
        let adaptive_set = Arc::new(Mutex::new(AdaptivePrototypeSet::new(
            threshold,
            max_variants,
        )));

        let start_time = std::time::Instant::now();

        // Spawn workers
        let mut handles = Vec::with_capacity(num_workers);

        for _ in 0..num_workers {
            let work_queue = Arc::clone(&work_queue);
            let processed = Arc::clone(&processed);
            let stats = Arc::clone(&stats);
            let callback = callback.clone();
            let adaptive_set = Arc::clone(&adaptive_set);
            let _bootstrap_config = self.bootstrap_config.clone();
            let progress_interval = self.config.progress_interval;
            let text_to_phonemes = self.text_to_phonemes.clone();

            handles.push(thread::spawn(move || {
                let mut projector = AudioProjector::default_config();

                loop {
                    let work = {
                        let mut queue = work_queue.lock().unwrap_or_else(|e| e.into_inner());
                        queue.pop()
                    };

                    let (_idx, (path, transcript)) = match work {
                        Some(w) => w,
                        None => break,
                    };

                    // Process file with adaptive training
                    let result = Self::process_training_file_adaptive(
                        &path,
                        &transcript,
                        &mut projector,
                        &adaptive_set,
                        text_to_phonemes.as_deref(),
                    );

                    // Update stats
                    {
                        let mut s = stats.lock().unwrap_or_else(|e| e.into_inner());
                        s.files_processed += 1;
                        match result {
                            Ok(duration) => {
                                s.files_success += 1;
                                s.total_audio_sec += duration;
                            }
                            Err(_) => {
                                s.files_failed += 1;
                            }
                        }
                    }

                    // Progress callback
                    let count = processed.fetch_add(1, Ordering::SeqCst) + 1;
                    if let Some(ref cb) = callback
                        && (count.is_multiple_of(progress_interval) || count == total)
                    {
                        cb(count, total, path.to_string_lossy().as_ref());
                    }
                }
            }));
        }

        // Wait for workers
        for handle in handles {
            let _ = handle.join();
        }

        // Finalize stats
        let mut final_stats = stats.lock().unwrap_or_else(|e| e.into_inner()).clone();
        final_stats.total_time_sec = start_time.elapsed().as_secs_f32();
        if final_stats.total_time_sec > 0.0 {
            final_stats.realtime_factor = final_stats.total_audio_sec / final_stats.total_time_sec;
        }

        // Extract the adaptive set
        let adaptive_set = Arc::try_unwrap(adaptive_set)
            .map(|mutex| mutex.into_inner().unwrap())
            .unwrap_or_else(|arc| arc.lock().unwrap_or_else(|e| e.into_inner()).clone());

        let adaptive_stats = adaptive_set.stats();

        (adaptive_set, final_stats, adaptive_stats)
    }

    fn process_training_file_adaptive(
        path: &Path,
        transcript: &str,
        projector: &mut AudioProjector,
        adaptive_set: &Arc<Mutex<AdaptivePrototypeSet>>,
        text_to_phonemes: Option<&TextToPhonemes>,
    ) -> Result<f32, String> {
        // Load audio
        let (audio, sample_rate) =
            AudioFrontend::load_audio(path).map_err(|e| format!("Load error: {e}"))?;

        let duration = audio.len() as f32 / sample_rate as f32;

        // Project audio
        projector.reset();
        let (hvs, _saliences) = projector.project_with_salience(&audio);

        if hvs.is_empty() {
            return Ok(duration);
        }

        // Get labels (phonemes if dictionary available, otherwise words)
        let labels: Vec<String> = if let Some(ttp) = text_to_phonemes {
            ttp.convert(transcript)
                .into_iter()
                .filter(|p| p != "_" && p != "SIL" && !p.starts_with('<'))
                .collect()
        } else {
            transcript
                .split_whitespace()
                .map(|w| w.to_lowercase())
                .collect()
        };

        if labels.is_empty() {
            return Ok(duration);
        }

        // Duration-weighted alignment based on phoneme class
        // Relative durations: vowels=3, fricatives=2, stops=1, other=1.5
        let durations: Vec<f32> = labels
            .iter()
            .map(|p| phoneme_relative_duration(p))
            .collect();
        let total_duration: f32 = durations.iter().sum();

        if total_duration == 0.0 {
            return Ok(duration);
        }

        // Compute frame ranges based on relative durations
        let total_frames = hvs.len() as f32;
        let mut frame_ranges: Vec<(usize, usize)> = Vec::with_capacity(labels.len());
        let mut current_frame: f32 = 0.0;

        for dur in &durations {
            let num_frames = (dur / total_duration) * total_frames;
            let start = current_frame.round() as usize;
            current_frame += num_frames;
            let end = current_frame.round().min(hvs.len() as f32) as usize;
            frame_ranges.push((start, end));
        }

        // For each label, bundle frames and train adaptively
        for (label, (start, end)) in labels.iter().zip(frame_ranges.iter()) {
            if *start >= hvs.len() || *start >= *end {
                continue;
            }

            // Train on individual frames (not bundled) to match inference behavior
            // Each frame within the phoneme occurrence contributes to the prototype
            let mut set = adaptive_set.lock().unwrap_or_else(|e| e.into_inner());
            for frame_idx in *start..*end {
                if frame_idx < hvs.len() {
                    set.train(label, &hvs[frame_idx]);
                }
            }
        }

        Ok(duration)
    }
}

/// Estimate relative phoneme duration based on phoneme class
/// Based on typical CMU phoneme durations in English speech
fn phoneme_relative_duration(phoneme: &str) -> f32 {
    // Strip stress marker (0, 1, 2) from vowels
    let base = phoneme.trim_end_matches(|c: char| c.is_ascii_digit());

    match base {
        // Long vowels (diphthongs and tense vowels)
        "AY" | "OY" | "AW" | "OW" | "EY" | "IY" | "UW" => 3.5,

        // Short vowels
        "AA" | "AE" | "AH" | "AO" | "EH" | "ER" | "IH" | "UH" => 2.5,

        // Reduced vowels (schwa)
        "AX" => 1.5,

        // Fricatives (relatively long)
        "S" | "Z" | "SH" | "ZH" | "F" | "V" | "TH" | "DH" | "HH" => 2.0,

        // Affricates
        "CH" | "JH" => 2.0,

        // Nasals (medium duration)
        "M" | "N" | "NG" => 2.0,

        // Liquids and glides
        "L" | "R" | "W" | "Y" => 1.8,

        // Stops (very short - just the release)
        "P" | "T" | "K" | "B" | "D" | "G" => 1.0,

        // Silence/unknown
        _ => 1.5,
    }
}

/// Batch discovery processor
pub struct BatchDiscovery {
    config: BatchConfig,
    discovery_config: DiscoveryConfig,
}

impl BatchDiscovery {
    /// Create a new batch discovery processor
    pub fn new(config: BatchConfig, discovery_config: DiscoveryConfig) -> Self {
        Self {
            config,
            discovery_config,
        }
    }

    /// Run discovery on multiple files
    pub fn discover(
        &self,
        files: &[PathBuf],
        callback: Option<ProgressCallback>,
    ) -> (DiscoveryResult, BatchStats) {
        let num_workers = self.config.workers();
        let total = files.len();

        // Shared discovery pipeline
        let pipeline = Arc::new(Mutex::new(DiscoveryPipeline::new(
            self.discovery_config.clone(),
        )));
        let work_queue = Arc::new(Mutex::new(
            files
                .iter()
                .enumerate()
                .map(|(i, path)| (i, path.clone()))
                .collect::<Vec<_>>(),
        ));
        let processed = Arc::new(AtomicUsize::new(0));
        let stats = Arc::new(Mutex::new(BatchStats::default()));
        let callback = callback.map(Arc::new);

        let start_time = std::time::Instant::now();

        // Spawn workers
        let mut handles = Vec::with_capacity(num_workers);

        for _ in 0..num_workers {
            let work_queue = Arc::clone(&work_queue);
            let pipeline = Arc::clone(&pipeline);
            let processed = Arc::clone(&processed);
            let stats = Arc::clone(&stats);
            let callback = callback.clone();
            let progress_interval = self.config.progress_interval;

            handles.push(thread::spawn(move || {
                let mut projector = AudioProjector::default_config();

                loop {
                    let work = {
                        let mut queue = work_queue.lock().unwrap_or_else(|e| e.into_inner());
                        queue.pop()
                    };

                    let (_, path) = match work {
                        Some(w) => w,
                        None => break,
                    };

                    // Process file
                    let result = Self::process_discovery_file(&path, &mut projector, &pipeline);

                    // Update stats
                    {
                        let mut s = stats.lock().unwrap_or_else(|e| e.into_inner());
                        s.files_processed += 1;
                        match result {
                            Ok(duration) => {
                                s.files_success += 1;
                                s.total_audio_sec += duration;
                            }
                            Err(_) => {
                                s.files_failed += 1;
                            }
                        }
                    }

                    // Progress
                    let count = processed.fetch_add(1, Ordering::SeqCst) + 1;
                    if let Some(ref cb) = callback
                        && (count.is_multiple_of(progress_interval) || count == total)
                    {
                        cb(count, total, path.to_string_lossy().as_ref());
                    }
                }
            }));
        }

        // Wait
        for handle in handles {
            let _ = handle.join();
        }

        // Finalize
        let result = pipeline
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .finalize();

        let mut final_stats = stats.lock().unwrap_or_else(|e| e.into_inner()).clone();
        final_stats.total_time_sec = start_time.elapsed().as_secs_f32();
        if final_stats.total_time_sec > 0.0 {
            final_stats.realtime_factor = final_stats.total_audio_sec / final_stats.total_time_sec;
        }

        (result, final_stats)
    }

    fn process_discovery_file(
        path: &Path,
        projector: &mut AudioProjector,
        pipeline: &Arc<Mutex<DiscoveryPipeline>>,
    ) -> Result<f32, String> {
        // Load audio
        let (audio, sample_rate) =
            AudioFrontend::load_audio(path).map_err(|e| format!("Load error: {e}"))?;

        let duration = audio.len() as f32 / sample_rate as f32;

        // Project
        projector.reset();
        let (hvs, saliences) = projector.project_with_salience(&audio);

        // Create timestamps
        let timestamps: Vec<f32> = (0..hvs.len()).map(|i| i as f32 * 0.010).collect();

        // Feed to pipeline
        {
            let mut pipe = pipeline.lock().unwrap_or_else(|e| e.into_inner());
            pipe.process(&hvs, &saliences, &timestamps);
        }

        Ok(duration)
    }
}

/// Find all audio files in a directory
pub fn find_audio_files(dir: &Path, extensions: &[&str]) -> Vec<PathBuf> {
    let mut files = Vec::new();

    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.filter_map(|e| e.ok()) {
            let path = entry.path();

            if path.is_dir() {
                // Recurse
                files.extend(find_audio_files(&path, extensions));
            } else if let Some(ext) = path.extension() {
                let ext_str = ext.to_string_lossy().to_lowercase();
                if extensions.iter().any(|e| *e == ext_str) {
                    files.push(path);
                }
            }
        }
    }

    files
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_config() {
        let config = BatchConfig::default();
        assert!(config.workers() >= 1);
    }

    #[test]
    fn test_phoneme_accumulator() {
        let mut acc = PhonemeAccumulator::new();

        let hv1 = HV16::random("test1");
        let hv2 = HV16::random("test2");

        acc.add(&hv1);
        acc.add(&hv2);

        assert_eq!(acc.count, 2);

        let result = acc.finalize();
        // Result should be similar to both inputs
        assert!(result.similarity(&hv1) > 0.3);
        assert!(result.similarity(&hv2) > 0.3);
    }

    #[test]
    fn test_find_audio_files() {
        // Test with current directory (may or may not have audio)
        let files = find_audio_files(Path::new("."), &["wav", "flac"]);
        // Just verify it doesn't panic
        // Just verify it doesn't panic (files.len() >= 0 is always true for usize)
        let _ = files;
    }
}
