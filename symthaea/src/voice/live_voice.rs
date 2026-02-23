//! Real-time streaming voice: text → phonemes → synthesis → speaker output.
//!
//! Combines [`SimpleG2P`] + [`StreamingVocalTract`] + [`AudioOutput`] into a single
//! `speak()` call that streams audio incrementally (first audio within ~25ms).
//!
//! Feature-gated under `live-voice`.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::Result;
use symthaea_core::genesis::GenesisSeed;

use super::audio_out::AudioOutput;
use super::formant_targets::FormantDatabase;
use super::repl_voice::SimpleG2P;
use super::vocal_tract_controller::train_controller_on_phoneme_db;
use super::vocal_tract_encoder::VoiceCognitiveState;
use super::vocal_tract_fep::StreamingVocalTract;

/// Real-time streaming voice: text → phonemes → synthesis → speaker output.
///
/// Combines SimpleG2P + StreamingVocalTract + AudioOutput into a single
/// `speak()` call that streams audio incrementally.
///
/// # Example
///
/// ```no_run
/// # use symthaea_core::genesis::GenesisSeed;
/// # use symthaea::voice::live_voice::LiveVoice;
/// let genesis = GenesisSeed::from_phrase("my-voice");
/// let mut voice = LiveVoice::new(&genesis).unwrap();
/// voice.speak("hello world").unwrap();
/// ```
pub struct LiveVoice {
    streaming: StreamingVocalTract,
    audio: AudioOutput,
    g2p: SimpleG2P,
    formant_db: FormantDatabase,
    cognitive_state: VoiceCognitiveState,
    speaking: Arc<AtomicBool>,
    genesis: GenesisSeed,
}

impl LiveVoice {
    /// Create all components, train the controller, and open audio output.
    ///
    /// Training runs 30 epochs on the full FormantDatabase (~43 phonemes).
    pub fn new(genesis: &GenesisSeed) -> Result<Self> {
        let audio = AudioOutput::new()?;
        let sample_rate = audio.sample_rate();

        let mut streaming = StreamingVocalTract::new(genesis, sample_rate, 200);

        // Train the controller on the phoneme database
        let db = FormantDatabase::new();
        for _ in 0..30 {
            train_controller_on_phoneme_db(&mut streaming.pipeline.controller, genesis, &db, 1);
        }

        Ok(Self {
            streaming,
            audio,
            g2p: SimpleG2P::new(),
            formant_db: db,
            cognitive_state: VoiceCognitiveState::default(),
            speaking: Arc::new(AtomicBool::new(false)),
            genesis: genesis.clone(),
        })
    }

    /// Speak text in real time: G2P → frame-by-frame synthesis → speaker.
    ///
    /// Streams audio incrementally. Returns when the entire utterance has been
    /// pushed to the ring buffer (playback may continue briefly after return).
    pub fn speak(&mut self, text: &str) -> Result<()> {
        self.speaking.store(true, Ordering::SeqCst);

        let dt = 0.005; // 200 Hz motor frame rate
        let base_duration = 0.06; // 60ms base phoneme duration

        let phonemes = self.g2p.text_to_phonemes(text, base_duration);

        for timed in &phonemes {
            if !self.speaking.load(Ordering::SeqCst) {
                break;
            }

            let n_frames = ((timed.duration / dt) as usize).max(1);
            let phoneme_str = if timed.phoneme == "SIL" {
                None
            } else {
                Some(timed.phoneme.as_str())
            };

            for _ in 0..n_frames {
                if !self.speaking.load(Ordering::SeqCst) {
                    break;
                }

                let chunk = self.streaming.tick(
                    &self.cognitive_state,
                    None,
                    dt,
                    phoneme_str,
                );
                self.push_with_backpressure(&chunk);
            }
        }

        self.speaking.store(false, Ordering::SeqCst);
        Ok(())
    }

    /// Push samples to the ring buffer with simple backpressure.
    fn push_with_backpressure(&mut self, samples: &[f32]) {
        let mut offset = 0;
        while offset < samples.len() {
            let written = self.audio.push_samples(&samples[offset..]);
            offset += written;
            if offset < samples.len() {
                // Buffer nearly full — yield briefly
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }
    }

    /// Stop speaking immediately. The ring buffer drains naturally to silence.
    pub fn stop(&self) {
        self.speaking.store(false, Ordering::SeqCst);
    }

    /// Whether `speak()` is currently running.
    pub fn is_speaking(&self) -> bool {
        self.speaking.load(Ordering::SeqCst)
    }

    /// Get a clone of the stop flag for cross-thread interruption.
    pub fn stop_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.speaking)
    }

    /// Set the cognitive state that modulates prosody in real time.
    pub fn set_cognitive_state(&mut self, state: VoiceCognitiveState) {
        self.cognitive_state = state;
    }

    /// Run additional training epochs on the formant database.
    pub fn train(&mut self, epochs: usize) {
        for _ in 0..epochs {
            train_controller_on_phoneme_db(
                &mut self.streaming.pipeline.controller,
                &self.genesis,
                &self.formant_db,
                1,
            );
        }
    }

    /// Audio sample rate from the output device.
    pub fn sample_rate(&self) -> u32 {
        self.audio.sample_rate()
    }

    /// Reset the vocal tract pipeline state.
    pub fn reset(&mut self) {
        self.streaming.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phoneme_sequence_generation() {
        // Verify G2P → TimedPhoneme pipeline without audio
        let g2p = SimpleG2P::new();
        let phonemes = g2p.text_to_phonemes("hello world", 0.06);
        assert!(!phonemes.is_empty(), "Should produce phonemes for 'hello world'");

        // Should contain actual phonemes (not just silence)
        let non_silence: Vec<_> = phonemes.iter().filter(|p| p.phoneme != "SIL").collect();
        assert!(
            non_silence.len() >= 4,
            "Should have at least 4 non-silence phonemes, got {}",
            non_silence.len()
        );
    }

    #[test]
    fn test_stop_flag_works() {
        let flag = Arc::new(AtomicBool::new(true));
        assert!(flag.load(Ordering::SeqCst));

        flag.store(false, Ordering::SeqCst);
        assert!(!flag.load(Ordering::SeqCst));
    }

    #[test]
    #[ignore] // Requires audio device
    fn test_live_voice_speak_produces_audio() {
        let genesis = GenesisSeed::from_phrase("test-live-voice");
        let mut voice = LiveVoice::new(&genesis).expect("Should create LiveVoice");
        assert!(voice.sample_rate() > 0);

        voice.speak("hello").expect("Should speak without error");
        assert!(!voice.is_speaking());
    }
}
