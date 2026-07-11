// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Autobiography narration bridge: `life_story` episodes → narrated prose,
//! off the hot cognitive-loop path.
//!
//! `crate::language::autobiography::narrate_autobiography` is async and
//! LLM-backed (via Ollama), so — mirroring `music_publisher.rs`'s
//! background-thread pattern — narration runs on a dedicated thread with its
//! own current-thread tokio runtime, fed by a request channel and drained
//! (non-blocking) from the sync cognitive-loop tick.
//!
//! Narration is deliberately infrequent: it only fires once at least
//! [`MIN_NEW_EPISODES_FOR_NARRATION`] new episodes have accumulated since the
//! last narration, since `life_story` itself only grows during dream-replay
//! insight events (see `helpers/cycle_phases_dream.rs`).

use std::sync::Mutex;
use std::sync::mpsc;
use std::thread;

use crate::consciousness::narrative_self::LifeEpisode;

/// Minimum new episodes since the last narration before triggering another one.
pub const MIN_NEW_EPISODES_FOR_NARRATION: usize = 3;

/// Request sent to the narration background thread.
#[derive(Debug, Clone)]
pub struct AutobiographyRequest {
    pub episodes: Vec<LifeEpisode>,
    pub self_name: String,
    pub cycle_num: u64,
}

/// Result from the narration background thread.
#[derive(Debug, Clone)]
pub struct AutobiographyResult {
    /// Narrated prose (or the compiled prompt, if no LLM backend was available).
    pub prose: String,
    /// Whether the prose actually came from the LLM backend.
    pub used_llm: bool,
    /// Number of episodes narrated (post-selection/chunking).
    pub episode_count: usize,
    pub cycle_num: u64,
}

/// Background narrator: submit requests, drain results, non-blocking.
pub struct AutobiographyNarrator {
    tx: mpsc::Sender<AutobiographyRequest>,
    rx: Mutex<mpsc::Receiver<AutobiographyResult>>,
    _thread: thread::JoinHandle<()>,
    submitted: u64,
    /// `life_story.len()` as of the last submitted narration request.
    last_episode_count_narrated: usize,
}

impl AutobiographyNarrator {
    /// Spawn the narration background thread (real Ollama backend).
    pub fn spawn() -> Self {
        let (request_tx, request_rx) = mpsc::channel::<AutobiographyRequest>();
        let (result_tx, result_rx) = mpsc::channel::<AutobiographyResult>();

        let handle = thread::Builder::new()
            .name("autobiography-narrator".to_string())
            .spawn(move || {
                narrator_loop(request_rx, result_tx);
            })
            .expect("autobiography-narrator thread spawn");

        Self {
            tx: request_tx,
            rx: Mutex::new(result_rx),
            _thread: handle,
            submitted: 0,
            last_episode_count_narrated: 0,
        }
    }

    /// Whether enough new episodes have accumulated to justify another narration.
    pub fn should_narrate(&self, current_episode_count: usize) -> bool {
        current_episode_count >= self.last_episode_count_narrated + MIN_NEW_EPISODES_FOR_NARRATION
    }

    /// Submit a narration request (non-blocking). Records the episode count as
    /// "narrated as of now" so `should_narrate` doesn't re-fire every cycle
    /// while the background thread is still working.
    pub fn submit(&mut self, request: AutobiographyRequest) -> bool {
        let episode_count = request.episodes.len();
        if self.tx.send(request).is_ok() {
            self.submitted += 1;
            self.last_episode_count_narrated = episode_count;
            true
        } else {
            false
        }
    }

    /// Drain completed narration results (call from the cognitive loop, non-blocking).
    pub fn drain_results(&mut self) -> Vec<AutobiographyResult> {
        let mut results = Vec::new();
        if let Ok(receiver) = self.rx.lock() {
            while let Ok(r) = receiver.try_recv() {
                results.push(r);
            }
        }
        results
    }

    /// Snapshot telemetry: (submitted count, episode count as of last narration).
    pub fn telemetry(&self) -> (u64, usize) {
        (self.submitted, self.last_episode_count_narrated)
    }
}

fn narrator_loop(
    request_rx: mpsc::Receiver<AutobiographyRequest>,
    result_tx: mpsc::Sender<AutobiographyResult>,
) {
    let rt = match tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    {
        Ok(rt) => rt,
        Err(e) => {
            tracing::error!("Failed to create tokio runtime for autobiography narrator: {e}");
            return;
        }
    };

    let backend = crate::language::llm_backend::OllamaBackend::new();

    while let Ok(request) = request_rx.recv() {
        // Skip to the latest request if several queued while we were busy.
        let mut latest = request;
        while let Ok(newer) = request_rx.try_recv() {
            latest = newer;
        }

        let output = rt.block_on(crate::language::autobiography::narrate_autobiography(
            &latest.episodes,
            &latest.self_name,
            Some(&backend),
        ));

        let result = AutobiographyResult {
            used_llm: output.used_llm(),
            prose: output.prose,
            episode_count: latest.episodes.len(),
            cycle_num: latest.cycle_num,
        };

        if result_tx.send(result).is_err() {
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_episode(desc: &str, valence: f64) -> LifeEpisode {
        LifeEpisode {
            description: desc.to_string(),
            encoding: crate::hdc::BinaryHV::random(0),
            valence,
            significance: 0.6,
            timestamp_secs: 0.0,
            causal_links: Vec::new(),
        }
    }

    #[test]
    fn should_narrate_false_before_threshold() {
        let narrator = AutobiographyNarrator::spawn();
        assert!(!narrator.should_narrate(MIN_NEW_EPISODES_FOR_NARRATION - 1));
    }

    #[test]
    fn should_narrate_true_at_threshold() {
        let narrator = AutobiographyNarrator::spawn();
        assert!(narrator.should_narrate(MIN_NEW_EPISODES_FOR_NARRATION));
    }

    #[test]
    fn submit_increments_count_and_updates_threshold() {
        let mut narrator = AutobiographyNarrator::spawn();
        let episodes = vec![
            dummy_episode("a", 0.1),
            dummy_episode("b", 0.2),
            dummy_episode("c", 0.3),
        ];
        assert!(narrator.submit(AutobiographyRequest {
            episodes,
            self_name: "Symthaea".to_string(),
            cycle_num: 1,
        }));
        let (submitted, last_count) = narrator.telemetry();
        assert_eq!(submitted, 1);
        assert_eq!(last_count, 3);
        // Shouldn't re-fire immediately for the same episode count.
        assert!(!narrator.should_narrate(3));
        assert!(narrator.should_narrate(3 + MIN_NEW_EPISODES_FOR_NARRATION));
    }

    #[test]
    fn drain_results_empty_initially() {
        let mut narrator = AutobiographyNarrator::spawn();
        let results = narrator.drain_results();
        assert!(results.is_empty());
    }
}
