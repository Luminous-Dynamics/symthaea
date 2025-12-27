//! Model Management for Voice Interface
//!
//! Handles downloading, caching, and loading of voice models:
//! - Whisper models for STT (tiny, base, small, medium, large)
//! - Kokoro model for TTS

use std::path::PathBuf;
use super::{VoiceError, VoiceResult};

/// Available Whisper model sizes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WhisperModel {
    /// Fastest, least accurate (~75MB)
    Tiny,
    /// Good balance for most uses (~150MB)
    Base,
    /// Higher accuracy (~500MB)
    Small,
    /// High accuracy (~1.5GB)
    Medium,
    /// Best accuracy (~3GB)
    Large,
}

impl WhisperModel {
    /// Get the model filename
    pub fn filename(&self) -> &'static str {
        match self {
            WhisperModel::Tiny => "ggml-tiny.bin",
            WhisperModel::Base => "ggml-base.bin",
            WhisperModel::Small => "ggml-small.bin",
            WhisperModel::Medium => "ggml-medium.bin",
            WhisperModel::Large => "ggml-large-v3.bin",
        }
    }

    /// Get HuggingFace repo and model path
    pub fn hf_path(&self) -> (&'static str, &'static str) {
        match self {
            WhisperModel::Tiny => ("ggerganov/whisper.cpp", "ggml-tiny.bin"),
            WhisperModel::Base => ("ggerganov/whisper.cpp", "ggml-base.bin"),
            WhisperModel::Small => ("ggerganov/whisper.cpp", "ggml-small.bin"),
            WhisperModel::Medium => ("ggerganov/whisper.cpp", "ggml-medium.bin"),
            WhisperModel::Large => ("ggerganov/whisper.cpp", "ggml-large-v3.bin"),
        }
    }

    /// Approximate model size in bytes
    pub fn size_bytes(&self) -> u64 {
        match self {
            WhisperModel::Tiny => 75_000_000,
            WhisperModel::Base => 150_000_000,
            WhisperModel::Small => 500_000_000,
            WhisperModel::Medium => 1_500_000_000,
            WhisperModel::Large => 3_000_000_000,
        }
    }
}

impl Default for WhisperModel {
    fn default() -> Self {
        WhisperModel::Base  // Good default balance
    }
}

/// Kokoro TTS model variants
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KokoroModel {
    /// Fast, smaller model (~80MB)
    V019,
    /// Higher quality (if available)
    V020,
}

impl KokoroModel {
    /// Get the model filename
    pub fn filename(&self) -> &'static str {
        match self {
            KokoroModel::V019 => "kokoro-v0_19.onnx",
            KokoroModel::V020 => "kokoro-v0_20.onnx",
        }
    }

    /// Get HuggingFace repo and model path
    pub fn hf_path(&self) -> (&'static str, &'static str) {
        match self {
            KokoroModel::V019 => ("hexgrad/Kokoro-82M", "kokoro-v0_19.onnx"),
            KokoroModel::V020 => ("hexgrad/Kokoro-82M", "kokoro-v0_20.onnx"),
        }
    }
}

impl Default for KokoroModel {
    fn default() -> Self {
        KokoroModel::V019
    }
}

/// Paths for voice models
#[derive(Debug, Clone)]
pub struct ModelPaths {
    /// Base directory for models
    pub base_dir: PathBuf,
    /// Whisper models directory
    pub whisper_dir: PathBuf,
    /// Kokoro models directory
    pub kokoro_dir: PathBuf,
}

impl ModelPaths {
    /// Create model paths from XDG data directory
    pub fn from_xdg() -> Self {
        let base_dir = std::env::var("XDG_DATA_HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                std::env::var("HOME")
                    .map(|h| PathBuf::from(h).join(".local/share"))
                    .unwrap_or_else(|_| PathBuf::from("/tmp"))
            })
            .join("symthaea/models");

        Self {
            whisper_dir: base_dir.join("whisper"),
            kokoro_dir: base_dir.join("kokoro"),
            base_dir,
        }
    }

    /// Get path to a specific Whisper model
    pub fn whisper_model(&self, model: WhisperModel) -> PathBuf {
        self.whisper_dir.join(model.filename())
    }

    /// Get path to a specific Kokoro model
    pub fn kokoro_model(&self, model: KokoroModel) -> PathBuf {
        self.kokoro_dir.join(model.filename())
    }

    /// Ensure all directories exist
    pub fn ensure_dirs(&self) -> VoiceResult<()> {
        std::fs::create_dir_all(&self.whisper_dir)?;
        std::fs::create_dir_all(&self.kokoro_dir)?;
        Ok(())
    }
}

impl Default for ModelPaths {
    fn default() -> Self {
        Self::from_xdg()
    }
}

/// Model manager for downloading and loading models
pub struct ModelManager {
    paths: ModelPaths,
}

impl ModelManager {
    /// Create a new model manager
    pub fn new() -> VoiceResult<Self> {
        let paths = ModelPaths::from_xdg();
        paths.ensure_dirs()?;
        Ok(Self { paths })
    }

    /// Create with custom paths
    pub fn with_paths(paths: ModelPaths) -> VoiceResult<Self> {
        paths.ensure_dirs()?;
        Ok(Self { paths })
    }

    /// Check if a Whisper model is downloaded
    pub fn has_whisper_model(&self, model: WhisperModel) -> bool {
        let path = self.paths.whisper_model(model);
        path.exists() && path.metadata().map(|m| m.len() > 1000).unwrap_or(false)
    }

    /// Check if a Kokoro model is downloaded
    pub fn has_kokoro_model(&self, model: KokoroModel) -> bool {
        let path = self.paths.kokoro_model(model);
        path.exists() && path.metadata().map(|m| m.len() > 1000).unwrap_or(false)
    }

    /// Get path to Whisper model (must be downloaded first)
    pub fn whisper_path(&self, model: WhisperModel) -> VoiceResult<PathBuf> {
        let path = self.paths.whisper_model(model);
        if self.has_whisper_model(model) {
            Ok(path)
        } else {
            Err(VoiceError::ModelLoad(format!(
                "Whisper model {:?} not found at {:?}. Run download first.",
                model, path
            )))
        }
    }

    /// Get path to Kokoro model (must be downloaded first)
    pub fn kokoro_path(&self, model: KokoroModel) -> VoiceResult<PathBuf> {
        let path = self.paths.kokoro_model(model);
        if self.has_kokoro_model(model) {
            Ok(path)
        } else {
            Err(VoiceError::ModelLoad(format!(
                "Kokoro model {:?} not found at {:?}. Run download first.",
                model, path
            )))
        }
    }

    /// Get model paths reference
    pub fn paths(&self) -> &ModelPaths {
        &self.paths
    }

    /// Download a Whisper model using hf-hub
    #[cfg(feature = "voice-stt")]
    pub fn download_whisper(&self, model: WhisperModel) -> VoiceResult<PathBuf> {
        use hf_hub::api::sync::Api;

        let (repo, file) = model.hf_path();
        let target = self.paths.whisper_model(model);

        if self.has_whisper_model(model) {
            return Ok(target);
        }

        eprintln!("Downloading Whisper {:?} from {}...", model, repo);

        let api = Api::new()
            .map_err(|e| VoiceError::Download(format!("HF API error: {}", e)))?;

        let repo = api.model(repo.to_string());
        let downloaded = repo.get(file)
            .map_err(|e| VoiceError::Download(format!("Download failed: {}", e)))?;

        // Copy to our location
        std::fs::copy(&downloaded, &target)?;

        eprintln!("Downloaded to {:?}", target);
        Ok(target)
    }

    /// Download a Kokoro model using hf-hub
    #[cfg(feature = "voice-tts")]
    pub fn download_kokoro(&self, model: KokoroModel) -> VoiceResult<PathBuf> {
        use hf_hub::api::sync::Api;

        let (repo, file) = model.hf_path();
        let target = self.paths.kokoro_model(model);

        if self.has_kokoro_model(model) {
            return Ok(target);
        }

        eprintln!("Downloading Kokoro {:?} from {}...", model, repo);

        let api = Api::new()
            .map_err(|e| VoiceError::Download(format!("HF API error: {}", e)))?;

        let repo = api.model(repo.to_string());
        let downloaded = repo.get(file)
            .map_err(|e| VoiceError::Download(format!("Download failed: {}", e)))?;

        // Copy to our location
        std::fs::copy(&downloaded, &target)?;

        eprintln!("Downloaded to {:?}", target);
        Ok(target)
    }
}

impl Default for ModelManager {
    fn default() -> Self {
        Self::new().expect("Failed to create model manager")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_whisper_model_filenames() {
        assert_eq!(WhisperModel::Tiny.filename(), "ggml-tiny.bin");
        assert_eq!(WhisperModel::Base.filename(), "ggml-base.bin");
        assert_eq!(WhisperModel::Large.filename(), "ggml-large-v3.bin");
    }

    #[test]
    fn test_kokoro_model_filenames() {
        assert_eq!(KokoroModel::V019.filename(), "kokoro-v0_19.onnx");
    }

    #[test]
    fn test_model_paths_xdg() {
        let paths = ModelPaths::from_xdg();
        assert!(paths.base_dir.to_string_lossy().contains("symthaea"));
        assert!(paths.whisper_dir.to_string_lossy().contains("whisper"));
        assert!(paths.kokoro_dir.to_string_lossy().contains("kokoro"));
    }

    #[test]
    fn test_whisper_model_path() {
        let paths = ModelPaths::from_xdg();
        let tiny_path = paths.whisper_model(WhisperModel::Tiny);
        assert!(tiny_path.to_string_lossy().ends_with("ggml-tiny.bin"));
    }

    #[test]
    fn test_model_manager_creation() {
        let manager = ModelManager::new();
        assert!(manager.is_ok());
    }
}
