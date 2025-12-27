//! Voice Conversation Module - Full Voice Loop with LTC Integration
//!
//! Provides a complete voice conversation interface that:
//! - Listens for speech via whisper-rs
//! - Processes input through Symthaea's language layer
//! - Synthesizes responses via Kokoro TTS
//! - Uses LTC temporal dynamics for natural pacing

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use super::{VoiceResult, LTCPacing};
use super::models::{WhisperModel, KokoroModel};

/// Configuration for voice conversation
#[derive(Debug, Clone)]
pub struct VoiceConfig {
    /// Whisper model for STT
    pub whisper_model: WhisperModel,
    /// Kokoro model for TTS
    pub kokoro_model: KokoroModel,
    /// Language for speech recognition
    pub language: Option<String>,
    /// Enable LTC-aware pacing
    pub ltc_pacing: bool,
    /// Voice ID (0-9)
    pub voice_id: u8,
    /// Wake word (optional)
    pub wake_word: Option<String>,
    /// Silence threshold for VAD
    pub silence_threshold: f32,
    /// Debug mode (print transcriptions)
    pub debug: bool,
}

impl Default for VoiceConfig {
    fn default() -> Self {
        Self {
            whisper_model: WhisperModel::Base,
            kokoro_model: KokoroModel::V019,
            language: Some("en".to_string()),
            ltc_pacing: true,
            voice_id: 0,
            wake_word: None,
            silence_threshold: 0.02,
            debug: false,
        }
    }
}

/// Events during voice conversation
#[derive(Debug, Clone)]
pub enum VoiceEvent {
    /// Ready to listen
    Ready,
    /// Started listening
    Listening,
    /// Speech detected
    SpeechDetected,
    /// Transcription complete
    Transcribed { text: String, confidence: f32 },
    /// Processing input
    Processing,
    /// Response generated
    Response { text: String },
    /// Speaking response
    Speaking,
    /// Finished speaking
    Done,
    /// Error occurred
    Error { message: String },
    /// User said stop/quit
    Stopped,
}

/// Voice conversation handler
pub struct VoiceConversation {
    config: VoiceConfig,
    is_running: Arc<AtomicBool>,
    current_ltc: LTCPacing,
    #[cfg(feature = "voice-stt")]
    input: Option<super::input::VoiceInput>,
    #[cfg(feature = "voice-tts")]
    output: Option<super::output::VoiceOutput>,
}

impl VoiceConversation {
    /// Create a new voice conversation handler
    pub fn new(config: VoiceConfig) -> VoiceResult<Self> {
        let mut vc = Self {
            config,
            is_running: Arc::new(AtomicBool::new(false)),
            current_ltc: LTCPacing::default(),
            #[cfg(feature = "voice-stt")]
            input: None,
            #[cfg(feature = "voice-tts")]
            output: None,
        };

        vc.initialize()?;
        Ok(vc)
    }

    /// Initialize voice components
    fn initialize(&mut self) -> VoiceResult<()> {
        #[cfg(feature = "voice-stt")]
        {
            use super::input::{VoiceInput, VoiceInputConfig};
            let input_config = VoiceInputConfig {
                model: self.config.whisper_model,
                language: self.config.language.clone(),
                silence_threshold: self.config.silence_threshold,
                ..Default::default()
            };
            self.input = Some(VoiceInput::new(input_config)?);
        }

        #[cfg(feature = "voice-tts")]
        {
            use super::output::{VoiceOutput, VoiceOutputConfig};
            let output_config = VoiceOutputConfig {
                model: self.config.kokoro_model,
                ltc_pacing: self.config.ltc_pacing,
                voice_id: self.config.voice_id,
                ..Default::default()
            };
            self.output = Some(VoiceOutput::new(output_config)?);
        }

        Ok(())
    }

    /// Update LTC pacing from conversation state
    pub fn update_ltc(&mut self, flow_state: f32, phi_trend: f32) {
        self.current_ltc = LTCPacing::from_ltc(flow_state, phi_trend);
    }

    /// Get current LTC pacing
    pub fn ltc_pacing(&self) -> &LTCPacing {
        &self.current_ltc
    }

    /// Check if running
    pub fn is_running(&self) -> bool {
        self.is_running.load(Ordering::SeqCst)
    }

    /// Stop the conversation loop
    pub fn stop(&self) {
        self.is_running.store(false, Ordering::SeqCst);
    }

    /// Speak text using TTS with LTC pacing
    #[cfg(feature = "voice-tts")]
    pub fn speak(&self, text: &str) -> VoiceResult<()> {
        if let Some(ref output) = self.output {
            let result = output.synthesize_with_pacing(text, self.current_ltc.clone())?;
            output.play_with_pacing(&result, self.current_ltc.clone())?;
        }
        Ok(())
    }

    /// Speak text sentence by sentence for natural flow
    #[cfg(feature = "voice-tts")]
    pub fn speak_natural(&self, text: &str) -> VoiceResult<()> {
        use super::output::split_sentences;

        let sentences = split_sentences(text);
        for sentence in sentences {
            if !self.is_running() {
                break;
            }
            self.speak(sentence)?;
        }
        Ok(())
    }

    /// Listen for speech and transcribe
    #[cfg(feature = "voice-stt")]
    pub fn listen(&mut self) -> VoiceResult<String> {
        use super::input::detect_speech;

        if let Some(ref mut input) = self.input {
            let stream = input.start_listening()?;

            // Wait for speech
            loop {
                std::thread::sleep(std::time::Duration::from_millis(100));
                let samples = stream.take_samples();

                if detect_speech(&samples, self.config.silence_threshold) {
                    // Continue collecting until silence
                    let mut all_samples = samples;
                    let mut silence_frames = 0;

                    while silence_frames < 10 {  // ~1 second of silence to end
                        std::thread::sleep(std::time::Duration::from_millis(100));
                        let new_samples = stream.take_samples();

                        if detect_speech(&new_samples, self.config.silence_threshold) {
                            silence_frames = 0;
                        } else {
                            silence_frames += 1;
                        }

                        all_samples.extend(new_samples);
                    }

                    stream.stop();

                    // Transcribe
                    let result = input.transcribe(&all_samples)?;
                    return Ok(result.text);
                }

                if !self.is_running() {
                    stream.stop();
                    return Err(VoiceError::AudioDevice("Stopped".into()));
                }
            }
        }

        Err(VoiceError::FeatureNotEnabled("voice-stt".into()))
    }

    /// Run a single turn of conversation
    #[cfg(all(feature = "voice-stt", feature = "voice-tts"))]
    pub fn turn<F>(&mut self, process: F) -> VoiceResult<VoiceEvent>
    where
        F: FnOnce(&str) -> String,
    {
        // Listen
        let user_text = self.listen()?;

        if self.config.debug {
            eprintln!("[STT] {}", user_text);
        }

        // Check for stop words
        let lower = user_text.to_lowercase();
        if lower.contains("stop") || lower.contains("quit") || lower.contains("goodbye") {
            self.speak("Goodbye!")?;
            return Ok(VoiceEvent::Stopped);
        }

        // Process
        let response = process(&user_text);

        if self.config.debug {
            eprintln!("[TTS] {}", response);
        }

        // Speak
        self.speak_natural(&response)?;

        Ok(VoiceEvent::Done)
    }

    /// Run continuous conversation loop
    #[cfg(all(feature = "voice-stt", feature = "voice-tts"))]
    pub fn run<F>(&mut self, mut process: F) -> VoiceResult<()>
    where
        F: FnMut(&str) -> String,
    {
        self.is_running.store(true, Ordering::SeqCst);

        // Initial greeting
        self.speak("Hello! I'm listening.")?;

        while self.is_running() {
            match self.turn(|text| process(text)) {
                Ok(VoiceEvent::Stopped) => break,
                Ok(_) => continue,
                Err(e) => {
                    eprintln!("Voice error: {}", e);
                    std::thread::sleep(std::time::Duration::from_secs(1));
                }
            }
        }

        Ok(())
    }
}

/// Create a voice conversation integrated with Symthaea's Conversation
#[cfg(all(feature = "voice-stt", feature = "voice-tts"))]
pub fn create_voice_symthaea(config: VoiceConfig) -> VoiceResult<VoiceConversation> {
    VoiceConversation::new(config)
}

/// Mock voice conversation for testing without audio hardware
pub struct MockVoiceConversation {
    config: VoiceConfig,
    ltc: LTCPacing,
    inputs: Vec<String>,
    outputs: Vec<String>,
    current_input: usize,
}

impl MockVoiceConversation {
    /// Create with predefined inputs
    pub fn new(config: VoiceConfig, inputs: Vec<String>) -> Self {
        Self {
            config,
            ltc: LTCPacing::default(),
            inputs,
            outputs: Vec::new(),
            current_input: 0,
        }
    }

    /// Update LTC
    pub fn update_ltc(&mut self, flow_state: f32, phi_trend: f32) {
        self.ltc = LTCPacing::from_ltc(flow_state, phi_trend);
    }

    /// Simulate listening
    pub fn listen(&mut self) -> Option<String> {
        if self.current_input < self.inputs.len() {
            let input = self.inputs[self.current_input].clone();
            self.current_input += 1;
            Some(input)
        } else {
            None
        }
    }

    /// Simulate speaking
    pub fn speak(&mut self, text: &str) {
        self.outputs.push(text.to_string());
    }

    /// Get outputs
    pub fn outputs(&self) -> &[String] {
        &self.outputs
    }

    /// Get LTC pacing
    pub fn ltc_pacing(&self) -> &LTCPacing {
        &self.ltc
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_voice_config_default() {
        let config = VoiceConfig::default();
        assert_eq!(config.whisper_model, WhisperModel::Base);
        assert_eq!(config.kokoro_model, KokoroModel::V019);
        assert!(config.ltc_pacing);
    }

    #[test]
    fn test_voice_event_variants() {
        let ready = VoiceEvent::Ready;
        let transcribed = VoiceEvent::Transcribed {
            text: "Hello".to_string(),
            confidence: 0.95,
        };
        let error = VoiceEvent::Error {
            message: "Test".to_string(),
        };

        assert!(matches!(ready, VoiceEvent::Ready));
        assert!(matches!(transcribed, VoiceEvent::Transcribed { .. }));
        assert!(matches!(error, VoiceEvent::Error { .. }));
    }

    #[test]
    fn test_mock_voice_conversation() {
        let inputs = vec![
            "Hello".to_string(),
            "How are you?".to_string(),
        ];
        let mut mock = MockVoiceConversation::new(VoiceConfig::default(), inputs);

        // First input
        let first = mock.listen();
        assert_eq!(first, Some("Hello".to_string()));

        // Speak response
        mock.speak("Hello! Nice to meet you.");

        // Second input
        let second = mock.listen();
        assert_eq!(second, Some("How are you?".to_string()));

        // No more inputs
        assert_eq!(mock.listen(), None);

        // Check outputs
        assert_eq!(mock.outputs().len(), 1);
        assert_eq!(mock.outputs()[0], "Hello! Nice to meet you.");
    }

    #[test]
    fn test_mock_ltc_update() {
        let mut mock = MockVoiceConversation::new(VoiceConfig::default(), vec![]);

        mock.update_ltc(0.8, 0.05);
        assert!(mock.ltc_pacing().speech_rate > 1.0);
        assert!(mock.ltc_pacing().peak_flow);

        mock.update_ltc(0.2, -0.05);
        assert!(mock.ltc_pacing().speech_rate < 1.0);
        assert!(!mock.ltc_pacing().peak_flow);
    }

    #[test]
    fn test_ltc_pacing_values() {
        // High flow, rising Φ
        let fast = LTCPacing::from_ltc(0.85, 0.1);
        assert_eq!(fast.speech_rate, 1.1);
        assert_eq!(fast.pause_ms, 150);
        assert!(fast.peak_flow);

        // Low flow, falling Φ
        let slow = LTCPacing::from_ltc(0.1, -0.1);
        assert_eq!(slow.speech_rate, 0.9);
        assert_eq!(slow.pause_ms, 500);
        assert!(!slow.peak_flow);
    }
}
