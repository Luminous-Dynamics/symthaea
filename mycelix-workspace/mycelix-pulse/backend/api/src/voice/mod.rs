// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Voice & Accessibility Module
//!
//! Provides voice dictation, voice commands, screen reader optimization,
//! text-to-speech, and comprehensive accessibility features.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use uuid::Uuid;

#[derive(Debug, thiserror::Error)]
pub enum VoiceError {
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),
    #[error("Speech recognition failed: {0}")]
    RecognitionFailed(String),
    #[error("Text-to-speech failed: {0}")]
    TtsFailed(String),
    #[error("Voice command not recognized: {0}")]
    UnknownCommand(String),
    #[error("Accessibility feature not available: {0}")]
    FeatureUnavailable(String),
}

// ============================================================================
// Voice Dictation
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DictationSession {
    pub id: Uuid,
    pub user_id: Uuid,
    pub started_at: DateTime<Utc>,
    pub ended_at: Option<DateTime<Utc>>,
    pub language: String,
    pub total_words: i32,
    pub accuracy_score: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DictationResult {
    pub text: String,
    pub confidence: f32,
    pub alternatives: Vec<AlternativeTranscription>,
    pub is_final: bool,
    pub words: Vec<WordTiming>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlternativeTranscription {
    pub text: String,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WordTiming {
    pub word: String,
    pub start_ms: i32,
    pub end_ms: i32,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DictationConfig {
    pub language: String,
    pub enable_punctuation: bool,
    pub enable_profanity_filter: bool,
    pub custom_vocabulary: Vec<String>,
    pub boost_phrases: Vec<String>,
}

pub struct DictationService {
    pool: PgPool,
    speech_endpoint: String,
}

impl DictationService {
    pub fn new(pool: PgPool, speech_endpoint: String) -> Self {
        Self { pool, speech_endpoint }
    }

    pub async fn start_session(
        &self,
        user_id: Uuid,
        config: DictationConfig,
    ) -> Result<DictationSession, VoiceError> {
        let session = sqlx::query_as!(
            DictationSession,
            r#"
            INSERT INTO dictation_sessions (user_id, language, started_at)
            VALUES ($1, $2, NOW())
            RETURNING id, user_id, started_at, ended_at, language,
                      total_words, accuracy_score
            "#,
            user_id,
            config.language
        )
        .fetch_one(&self.pool)
        .await?;

        // Store custom vocabulary for the session
        for word in config.custom_vocabulary {
            sqlx::query!(
                r#"
                INSERT INTO dictation_vocabulary (session_id, word, boost_weight)
                VALUES ($1, $2, 1.5)
                "#,
                session.id,
                word
            )
            .execute(&self.pool)
            .await?;
        }

        Ok(session)
    }

    pub async fn process_audio_chunk(
        &self,
        session_id: Uuid,
        audio_data: &[u8],
        is_final: bool,
    ) -> Result<DictationResult, VoiceError> {
        // In production, this would send to speech recognition service
        // For now, return a placeholder result
        let result = DictationResult {
            text: String::new(),
            confidence: 0.95,
            alternatives: vec![],
            is_final,
            words: vec![],
        };

        if is_final {
            sqlx::query!(
                r#"
                UPDATE dictation_sessions
                SET total_words = total_words + $1
                WHERE id = $2
                "#,
                result.words.len() as i32,
                session_id
            )
            .execute(&self.pool)
            .await?;
        }

        Ok(result)
    }

    pub async fn end_session(&self, session_id: Uuid) -> Result<DictationSession, VoiceError> {
        let session = sqlx::query_as!(
            DictationSession,
            r#"
            UPDATE dictation_sessions
            SET ended_at = NOW()
            WHERE id = $1
            RETURNING id, user_id, started_at, ended_at, language,
                      total_words, accuracy_score
            "#,
            session_id
        )
        .fetch_one(&self.pool)
        .await?;

        Ok(session)
    }

    pub async fn get_user_vocabulary(&self, user_id: Uuid) -> Result<Vec<String>, VoiceError> {
        let words = sqlx::query_scalar!(
            r#"
            SELECT DISTINCT word
            FROM user_custom_vocabulary
            WHERE user_id = $1
            ORDER BY word
            "#,
            user_id
        )
        .fetch_all(&self.pool)
        .await?;

        Ok(words.into_iter().flatten().collect())
    }

    pub async fn add_custom_word(&self, user_id: Uuid, word: &str) -> Result<(), VoiceError> {
        sqlx::query!(
            r#"
            INSERT INTO user_custom_vocabulary (user_id, word, added_at)
            VALUES ($1, $2, NOW())
            ON CONFLICT (user_id, word) DO NOTHING
            "#,
            user_id,
            word
        )
        .execute(&self.pool)
        .await?;

        Ok(())
    }
}

// ============================================================================
// Voice Commands
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VoiceCommand {
    // Navigation
    OpenInbox,
    OpenDrafts,
    OpenSent,
    OpenFolder(String),
    NextEmail,
    PreviousEmail,

    // Reading
    ReadEmail,
    ReadSubject,
    ReadSender,
    StopReading,

    // Composing
    NewEmail,
    Reply,
    ReplyAll,
    Forward,

    // Actions
    Archive,
    Delete,
    MarkRead,
    MarkUnread,
    Star,
    Snooze(String), // duration like "1 hour", "tomorrow"

    // Search
    Search(String),
    SearchFrom(String),
    SearchSubject(String),

    // Dictation control
    StartDictation,
    StopDictation,
    SendEmail,
    DiscardDraft,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommandResult {
    pub command: VoiceCommand,
    pub success: bool,
    pub message: String,
    pub speak_response: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceCommandConfig {
    pub enabled: bool,
    pub wake_word: Option<String>,
    pub language: String,
    pub confirmation_required: Vec<String>, // commands that need confirmation
    pub custom_commands: Vec<CustomVoiceCommand>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomVoiceCommand {
    pub phrase: String,
    pub action: VoiceCommand,
}

pub struct VoiceCommandService {
    pool: PgPool,
}

impl VoiceCommandService {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    pub async fn parse_command(&self, user_id: Uuid, text: &str) -> Result<VoiceCommand, VoiceError> {
        let text_lower = text.to_lowercase();

        // Check custom commands first
        if let Some(custom) = self.find_custom_command(user_id, &text_lower).await? {
            return Ok(custom);
        }

        // Parse built-in commands
        let command = if text_lower.contains("open inbox") || text_lower.contains("go to inbox") {
            VoiceCommand::OpenInbox
        } else if text_lower.contains("open drafts") {
            VoiceCommand::OpenDrafts
        } else if text_lower.contains("open sent") {
            VoiceCommand::OpenSent
        } else if text_lower.starts_with("open ") {
            let folder = text_lower.replace("open ", "").trim().to_string();
            VoiceCommand::OpenFolder(folder)
        } else if text_lower.contains("next email") || text_lower.contains("next message") {
            VoiceCommand::NextEmail
        } else if text_lower.contains("previous email") || text_lower.contains("previous message") {
            VoiceCommand::PreviousEmail
        } else if text_lower.contains("read email") || text_lower.contains("read this") {
            VoiceCommand::ReadEmail
        } else if text_lower.contains("read subject") {
            VoiceCommand::ReadSubject
        } else if text_lower.contains("who sent") || text_lower.contains("read sender") {
            VoiceCommand::ReadSender
        } else if text_lower.contains("stop reading") || text_lower.contains("stop") {
            VoiceCommand::StopReading
        } else if text_lower.contains("new email") || text_lower.contains("compose") {
            VoiceCommand::NewEmail
        } else if text_lower.contains("reply all") {
            VoiceCommand::ReplyAll
        } else if text_lower.contains("reply") {
            VoiceCommand::Reply
        } else if text_lower.contains("forward") {
            VoiceCommand::Forward
        } else if text_lower.contains("archive") {
            VoiceCommand::Archive
        } else if text_lower.contains("delete") {
            VoiceCommand::Delete
        } else if text_lower.contains("mark as read") || text_lower.contains("mark read") {
            VoiceCommand::MarkRead
        } else if text_lower.contains("mark as unread") || text_lower.contains("mark unread") {
            VoiceCommand::MarkUnread
        } else if text_lower.contains("star") || text_lower.contains("important") {
            VoiceCommand::Star
        } else if text_lower.starts_with("snooze") {
            let duration = text_lower.replace("snooze", "").trim().to_string();
            VoiceCommand::Snooze(duration)
        } else if text_lower.starts_with("search for ") {
            let query = text_lower.replace("search for ", "").trim().to_string();
            VoiceCommand::Search(query)
        } else if text_lower.starts_with("search from ") {
            let sender = text_lower.replace("search from ", "").trim().to_string();
            VoiceCommand::SearchFrom(sender)
        } else if text_lower.contains("start dictation") || text_lower.contains("dictate") {
            VoiceCommand::StartDictation
        } else if text_lower.contains("stop dictation") {
            VoiceCommand::StopDictation
        } else if text_lower.contains("send email") || text_lower.contains("send it") {
            VoiceCommand::SendEmail
        } else if text_lower.contains("discard") || text_lower.contains("cancel") {
            VoiceCommand::DiscardDraft
        } else {
            return Err(VoiceError::UnknownCommand(text.to_string()));
        };

        // Log command for analytics
        self.log_command(user_id, &command).await?;

        Ok(command)
    }

    async fn find_custom_command(
        &self,
        user_id: Uuid,
        text: &str,
    ) -> Result<Option<VoiceCommand>, VoiceError> {
        let custom = sqlx::query!(
            r#"
            SELECT phrase, action_type, action_data
            FROM custom_voice_commands
            WHERE user_id = $1 AND LOWER(phrase) = $2
            "#,
            user_id,
            text
        )
        .fetch_optional(&self.pool)
        .await?;

        if let Some(cmd) = custom {
            // Parse action_type and action_data into VoiceCommand
            // This would need proper deserialization in production
            Ok(None)
        } else {
            Ok(None)
        }
    }

    async fn log_command(&self, user_id: Uuid, command: &VoiceCommand) -> Result<(), VoiceError> {
        let command_type = format!("{:?}", command);
        sqlx::query!(
            r#"
            INSERT INTO voice_command_log (user_id, command_type, executed_at)
            VALUES ($1, $2, NOW())
            "#,
            user_id,
            command_type
        )
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    pub async fn get_config(&self, user_id: Uuid) -> Result<VoiceCommandConfig, VoiceError> {
        let config = sqlx::query!(
            r#"
            SELECT enabled, wake_word, language, confirmation_commands
            FROM voice_command_settings
            WHERE user_id = $1
            "#,
            user_id
        )
        .fetch_optional(&self.pool)
        .await?;

        Ok(config
            .map(|c| VoiceCommandConfig {
                enabled: c.enabled,
                wake_word: c.wake_word,
                language: c.language,
                confirmation_required: c
                    .confirmation_commands
                    .unwrap_or_default()
                    .split(',')
                    .map(String::from)
                    .collect(),
                custom_commands: vec![],
            })
            .unwrap_or(VoiceCommandConfig {
                enabled: false,
                wake_word: Some("Hey Mail".to_string()),
                language: "en-US".to_string(),
                confirmation_required: vec!["delete".to_string(), "send".to_string()],
                custom_commands: vec![],
            }))
    }
}

// ============================================================================
// Text-to-Speech
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TtsVoice {
    pub id: String,
    pub name: String,
    pub language: String,
    pub gender: String,
    pub preview_url: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TtsRequest {
    pub text: String,
    pub voice_id: String,
    pub speed: f32,      // 0.5 to 2.0
    pub pitch: f32,      // 0.5 to 2.0
    pub volume: f32,     // 0.0 to 1.0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TtsSettings {
    pub voice_id: String,
    pub speed: f32,
    pub pitch: f32,
    pub auto_read_new: bool,
    pub read_sender_first: bool,
    pub skip_signatures: bool,
    pub skip_quoted_text: bool,
}

pub struct TtsService {
    pool: PgPool,
    tts_endpoint: String,
}

impl TtsService {
    pub fn new(pool: PgPool, tts_endpoint: String) -> Self {
        Self { pool, tts_endpoint }
    }

    pub async fn synthesize(&self, request: TtsRequest) -> Result<Vec<u8>, VoiceError> {
        // In production, this would call the TTS API
        // Return empty audio for now
        Ok(vec![])
    }

    pub async fn get_available_voices(&self, language: Option<&str>) -> Result<Vec<TtsVoice>, VoiceError> {
        // Return common voice options
        Ok(vec![
            TtsVoice {
                id: "en-US-Neural2-A".to_string(),
                name: "Aria".to_string(),
                language: "en-US".to_string(),
                gender: "female".to_string(),
                preview_url: None,
            },
            TtsVoice {
                id: "en-US-Neural2-D".to_string(),
                name: "Davis".to_string(),
                language: "en-US".to_string(),
                gender: "male".to_string(),
                preview_url: None,
            },
            TtsVoice {
                id: "en-GB-Neural2-A".to_string(),
                name: "Emma".to_string(),
                language: "en-GB".to_string(),
                gender: "female".to_string(),
                preview_url: None,
            },
        ])
    }

    pub async fn get_settings(&self, user_id: Uuid) -> Result<TtsSettings, VoiceError> {
        let settings = sqlx::query!(
            r#"
            SELECT voice_id, speed, pitch, auto_read_new,
                   read_sender_first, skip_signatures, skip_quoted_text
            FROM tts_settings
            WHERE user_id = $1
            "#,
            user_id
        )
        .fetch_optional(&self.pool)
        .await?;

        Ok(settings
            .map(|s| TtsSettings {
                voice_id: s.voice_id,
                speed: s.speed,
                pitch: s.pitch,
                auto_read_new: s.auto_read_new,
                read_sender_first: s.read_sender_first,
                skip_signatures: s.skip_signatures,
                skip_quoted_text: s.skip_quoted_text,
            })
            .unwrap_or(TtsSettings {
                voice_id: "en-US-Neural2-A".to_string(),
                speed: 1.0,
                pitch: 1.0,
                auto_read_new: false,
                read_sender_first: true,
                skip_signatures: true,
                skip_quoted_text: true,
            }))
    }

    pub async fn update_settings(
        &self,
        user_id: Uuid,
        settings: TtsSettings,
    ) -> Result<(), VoiceError> {
        sqlx::query!(
            r#"
            INSERT INTO tts_settings (user_id, voice_id, speed, pitch,
                auto_read_new, read_sender_first, skip_signatures, skip_quoted_text)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ON CONFLICT (user_id) DO UPDATE SET
                voice_id = EXCLUDED.voice_id,
                speed = EXCLUDED.speed,
                pitch = EXCLUDED.pitch,
                auto_read_new = EXCLUDED.auto_read_new,
                read_sender_first = EXCLUDED.read_sender_first,
                skip_signatures = EXCLUDED.skip_signatures,
                skip_quoted_text = EXCLUDED.skip_quoted_text
            "#,
            user_id,
            settings.voice_id,
            settings.speed,
            settings.pitch,
            settings.auto_read_new,
            settings.read_sender_first,
            settings.skip_signatures,
            settings.skip_quoted_text
        )
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    pub async fn prepare_email_for_reading(
        &self,
        user_id: Uuid,
        email_id: Uuid,
    ) -> Result<String, VoiceError> {
        let settings = self.get_settings(user_id).await?;

        let email = sqlx::query!(
            r#"
            SELECT subject, sender_name, sender_email, body_text, body_html
            FROM emails
            WHERE id = $1
            "#,
            email_id
        )
        .fetch_one(&self.pool)
        .await?;

        let mut text_parts = vec![];

        if settings.read_sender_first {
            let sender = email.sender_name.unwrap_or(email.sender_email);
            text_parts.push(format!("Email from {}.", sender));
        }

        text_parts.push(format!("Subject: {}.", email.subject));

        let body = email.body_text.unwrap_or_else(|| {
            // Strip HTML if only HTML body available
            html_to_text(&email.body_html.unwrap_or_default())
        });

        let cleaned_body = if settings.skip_signatures {
            remove_signature(&body)
        } else {
            body.clone()
        };

        let final_body = if settings.skip_quoted_text {
            remove_quoted_text(&cleaned_body)
        } else {
            cleaned_body
        };

        text_parts.push(final_body);

        Ok(text_parts.join(" "))
    }
}

fn html_to_text(html: &str) -> String {
    // Simple HTML to text conversion
    let text = html
        .replace("<br>", "\n")
        .replace("<br/>", "\n")
        .replace("<br />", "\n")
        .replace("</p>", "\n\n")
        .replace("</div>", "\n");

    // Remove remaining tags
    let re = regex::Regex::new(r"<[^>]+>").unwrap();
    re.replace_all(&text, "").to_string()
}

fn remove_signature(text: &str) -> String {
    // Look for common signature separators
    let separators = ["--\n", "-- \n", "___", "Best regards,", "Sincerely,", "Thanks,"];

    for sep in separators {
        if let Some(pos) = text.find(sep) {
            return text[..pos].trim().to_string();
        }
    }

    text.to_string()
}

fn remove_quoted_text(text: &str) -> String {
    let lines: Vec<&str> = text
        .lines()
        .filter(|line| !line.starts_with('>'))
        .filter(|line| !line.starts_with("On ") || !line.contains(" wrote:"))
        .collect();

    lines.join("\n")
}

// ============================================================================
// Accessibility Settings
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessibilitySettings {
    pub high_contrast: bool,
    pub contrast_level: ContrastLevel,
    pub font_size: FontSize,
    pub reduced_motion: bool,
    pub screen_reader_optimized: bool,
    pub keyboard_only_mode: bool,
    pub focus_indicators: FocusIndicatorStyle,
    pub link_underlines: bool,
    pub dyslexia_friendly_font: bool,
    pub color_blind_mode: Option<ColorBlindMode>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContrastLevel {
    Normal,
    AA,
    AAA,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FontSize {
    Small,
    Medium,
    Large,
    XLarge,
    Custom(i32), // percentage
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FocusIndicatorStyle {
    Default,
    Bold,
    HighContrast,
    Custom { color: String, width: i32 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ColorBlindMode {
    Protanopia,
    Deuteranopia,
    Tritanopia,
    Achromatopsia,
}

pub struct AccessibilityService {
    pool: PgPool,
}

impl AccessibilityService {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    pub async fn get_settings(&self, user_id: Uuid) -> Result<AccessibilitySettings, VoiceError> {
        let settings = sqlx::query!(
            r#"
            SELECT high_contrast, contrast_level, font_size, reduced_motion,
                   screen_reader_optimized, keyboard_only_mode, focus_style,
                   link_underlines, dyslexia_font, color_blind_mode
            FROM accessibility_settings
            WHERE user_id = $1
            "#,
            user_id
        )
        .fetch_optional(&self.pool)
        .await?;

        Ok(settings
            .map(|s| AccessibilitySettings {
                high_contrast: s.high_contrast,
                contrast_level: serde_json::from_str(&s.contrast_level).unwrap_or(ContrastLevel::Normal),
                font_size: serde_json::from_str(&s.font_size).unwrap_or(FontSize::Medium),
                reduced_motion: s.reduced_motion,
                screen_reader_optimized: s.screen_reader_optimized,
                keyboard_only_mode: s.keyboard_only_mode,
                focus_indicators: serde_json::from_str(&s.focus_style).unwrap_or(FocusIndicatorStyle::Default),
                link_underlines: s.link_underlines,
                dyslexia_friendly_font: s.dyslexia_font,
                color_blind_mode: s.color_blind_mode.and_then(|m| serde_json::from_str(&m).ok()),
            })
            .unwrap_or(AccessibilitySettings {
                high_contrast: false,
                contrast_level: ContrastLevel::Normal,
                font_size: FontSize::Medium,
                reduced_motion: false,
                screen_reader_optimized: false,
                keyboard_only_mode: false,
                focus_indicators: FocusIndicatorStyle::Default,
                link_underlines: true,
                dyslexia_friendly_font: false,
                color_blind_mode: None,
            }))
    }

    pub async fn update_settings(
        &self,
        user_id: Uuid,
        settings: AccessibilitySettings,
    ) -> Result<(), VoiceError> {
        sqlx::query!(
            r#"
            INSERT INTO accessibility_settings (
                user_id, high_contrast, contrast_level, font_size, reduced_motion,
                screen_reader_optimized, keyboard_only_mode, focus_style,
                link_underlines, dyslexia_font, color_blind_mode
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            ON CONFLICT (user_id) DO UPDATE SET
                high_contrast = EXCLUDED.high_contrast,
                contrast_level = EXCLUDED.contrast_level,
                font_size = EXCLUDED.font_size,
                reduced_motion = EXCLUDED.reduced_motion,
                screen_reader_optimized = EXCLUDED.screen_reader_optimized,
                keyboard_only_mode = EXCLUDED.keyboard_only_mode,
                focus_style = EXCLUDED.focus_style,
                link_underlines = EXCLUDED.link_underlines,
                dyslexia_font = EXCLUDED.dyslexia_font,
                color_blind_mode = EXCLUDED.color_blind_mode
            "#,
            user_id,
            settings.high_contrast,
            serde_json::to_string(&settings.contrast_level).unwrap(),
            serde_json::to_string(&settings.font_size).unwrap(),
            settings.reduced_motion,
            settings.screen_reader_optimized,
            settings.keyboard_only_mode,
            serde_json::to_string(&settings.focus_indicators).unwrap(),
            settings.link_underlines,
            settings.dyslexia_friendly_font,
            settings.color_blind_mode.map(|m| serde_json::to_string(&m).unwrap())
        )
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    pub async fn generate_aria_labels(&self, email_id: Uuid) -> Result<AriaLabels, VoiceError> {
        let email = sqlx::query!(
            r#"
            SELECT subject, sender_name, sender_email, received_at,
                   is_read, is_starred, has_attachments, attachment_count
            FROM emails
            WHERE id = $1
            "#,
            email_id
        )
        .fetch_one(&self.pool)
        .await?;

        let sender = email.sender_name.unwrap_or(email.sender_email);
        let status = if email.is_read { "read" } else { "unread" };
        let starred = if email.is_starred { ", starred" } else { "" };
        let attachments = if email.has_attachments {
            format!(", {} attachments", email.attachment_count)
        } else {
            String::new()
        };

        Ok(AriaLabels {
            email_row: format!(
                "{} email from {}, subject: {}{}{}",
                status, sender, email.subject, starred, attachments
            ),
            star_button: if email.is_starred {
                "Remove star".to_string()
            } else {
                "Add star".to_string()
            },
            archive_button: "Archive email".to_string(),
            delete_button: "Delete email".to_string(),
            reply_button: "Reply to email".to_string(),
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AriaLabels {
    pub email_row: String,
    pub star_button: String,
    pub archive_button: String,
    pub delete_button: String,
    pub reply_button: String,
}

// ============================================================================
// Keyboard Navigation
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyboardShortcut {
    pub key: String,
    pub modifiers: Vec<KeyModifier>,
    pub action: String,
    pub description: String,
    pub category: ShortcutCategory,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KeyModifier {
    Ctrl,
    Alt,
    Shift,
    Meta,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ShortcutCategory {
    Navigation,
    Actions,
    Compose,
    Selection,
    View,
}

pub struct KeyboardService {
    pool: PgPool,
}

impl KeyboardService {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    pub fn get_default_shortcuts(&self) -> Vec<KeyboardShortcut> {
        vec![
            // Navigation
            KeyboardShortcut {
                key: "j".to_string(),
                modifiers: vec![],
                action: "next_email".to_string(),
                description: "Move to next email".to_string(),
                category: ShortcutCategory::Navigation,
            },
            KeyboardShortcut {
                key: "k".to_string(),
                modifiers: vec![],
                action: "prev_email".to_string(),
                description: "Move to previous email".to_string(),
                category: ShortcutCategory::Navigation,
            },
            KeyboardShortcut {
                key: "o".to_string(),
                modifiers: vec![],
                action: "open_email".to_string(),
                description: "Open selected email".to_string(),
                category: ShortcutCategory::Navigation,
            },
            KeyboardShortcut {
                key: "u".to_string(),
                modifiers: vec![],
                action: "back_to_list".to_string(),
                description: "Return to email list".to_string(),
                category: ShortcutCategory::Navigation,
            },
            KeyboardShortcut {
                key: "g".to_string(),
                modifiers: vec![],
                action: "goto_menu".to_string(),
                description: "Open go-to menu (g+i inbox, g+d drafts, etc.)".to_string(),
                category: ShortcutCategory::Navigation,
            },
            // Actions
            KeyboardShortcut {
                key: "e".to_string(),
                modifiers: vec![],
                action: "archive".to_string(),
                description: "Archive selected email".to_string(),
                category: ShortcutCategory::Actions,
            },
            KeyboardShortcut {
                key: "#".to_string(),
                modifiers: vec![KeyModifier::Shift],
                action: "delete".to_string(),
                description: "Delete selected email".to_string(),
                category: ShortcutCategory::Actions,
            },
            KeyboardShortcut {
                key: "s".to_string(),
                modifiers: vec![],
                action: "star".to_string(),
                description: "Star/unstar selected email".to_string(),
                category: ShortcutCategory::Actions,
            },
            KeyboardShortcut {
                key: "!".to_string(),
                modifiers: vec![KeyModifier::Shift],
                action: "spam".to_string(),
                description: "Mark as spam".to_string(),
                category: ShortcutCategory::Actions,
            },
            // Compose
            KeyboardShortcut {
                key: "c".to_string(),
                modifiers: vec![],
                action: "compose".to_string(),
                description: "Compose new email".to_string(),
                category: ShortcutCategory::Compose,
            },
            KeyboardShortcut {
                key: "r".to_string(),
                modifiers: vec![],
                action: "reply".to_string(),
                description: "Reply to email".to_string(),
                category: ShortcutCategory::Compose,
            },
            KeyboardShortcut {
                key: "a".to_string(),
                modifiers: vec![],
                action: "reply_all".to_string(),
                description: "Reply to all".to_string(),
                category: ShortcutCategory::Compose,
            },
            KeyboardShortcut {
                key: "f".to_string(),
                modifiers: vec![],
                action: "forward".to_string(),
                description: "Forward email".to_string(),
                category: ShortcutCategory::Compose,
            },
            KeyboardShortcut {
                key: "Enter".to_string(),
                modifiers: vec![KeyModifier::Ctrl],
                action: "send".to_string(),
                description: "Send email".to_string(),
                category: ShortcutCategory::Compose,
            },
            // Selection
            KeyboardShortcut {
                key: "x".to_string(),
                modifiers: vec![],
                action: "select".to_string(),
                description: "Select/deselect email".to_string(),
                category: ShortcutCategory::Selection,
            },
            KeyboardShortcut {
                key: "*".to_string(),
                modifiers: vec![KeyModifier::Shift],
                action: "select_all".to_string(),
                description: "Select all emails".to_string(),
                category: ShortcutCategory::Selection,
            },
            // View
            KeyboardShortcut {
                key: "/".to_string(),
                modifiers: vec![],
                action: "search".to_string(),
                description: "Open search".to_string(),
                category: ShortcutCategory::View,
            },
            KeyboardShortcut {
                key: "?".to_string(),
                modifiers: vec![KeyModifier::Shift],
                action: "help".to_string(),
                description: "Show keyboard shortcuts".to_string(),
                category: ShortcutCategory::View,
            },
        ]
    }

    pub async fn get_user_shortcuts(&self, user_id: Uuid) -> Result<Vec<KeyboardShortcut>, VoiceError> {
        // Get user customizations and merge with defaults
        let customizations = sqlx::query!(
            r#"
            SELECT action, key_combo, enabled
            FROM keyboard_shortcuts
            WHERE user_id = $1
            "#,
            user_id
        )
        .fetch_all(&self.pool)
        .await?;

        let mut shortcuts = self.get_default_shortcuts();

        // Apply user customizations
        for custom in customizations {
            if let Some(shortcut) = shortcuts.iter_mut().find(|s| s.action == custom.action) {
                // Parse key_combo like "Ctrl+Shift+K"
                let parts: Vec<&str> = custom.key_combo.split('+').collect();
                shortcut.key = parts.last().unwrap_or(&"").to_string();
                shortcut.modifiers = parts[..parts.len().saturating_sub(1)]
                    .iter()
                    .filter_map(|m| match m.to_lowercase().as_str() {
                        "ctrl" => Some(KeyModifier::Ctrl),
                        "alt" => Some(KeyModifier::Alt),
                        "shift" => Some(KeyModifier::Shift),
                        "meta" | "cmd" => Some(KeyModifier::Meta),
                        _ => None,
                    })
                    .collect();
            }
        }

        Ok(shortcuts)
    }

    pub async fn set_shortcut(
        &self,
        user_id: Uuid,
        action: &str,
        key_combo: &str,
    ) -> Result<(), VoiceError> {
        sqlx::query!(
            r#"
            INSERT INTO keyboard_shortcuts (user_id, action, key_combo, enabled)
            VALUES ($1, $2, $3, true)
            ON CONFLICT (user_id, action) DO UPDATE SET
                key_combo = EXCLUDED.key_combo
            "#,
            user_id,
            action,
            key_combo
        )
        .execute(&self.pool)
        .await?;

        Ok(())
    }
}
