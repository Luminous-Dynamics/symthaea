// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! WASM-safe mirror types for the mycelix-mail Leptos frontend.
//!
//! These types mirror the Holochain zome entry types but carry NO hdi/hdk
//! dependencies, making them safe for wasm32-unknown-unknown (browser) targets.

use serde::{Deserialize, Serialize};

// ── Email Types ──

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EmailListItem {
    pub hash: String,
    pub sender: String,
    pub sender_name: Option<String>,
    pub encrypted_subject: Vec<u8>,
    pub subject: Option<String>,
    pub snippet: Option<String>,
    pub timestamp: u64,
    pub priority: EmailPriority,
    pub is_read: bool,
    pub is_starred: bool,
    pub star_type: Option<StarType>,
    pub is_pinned: bool,
    pub is_muted: bool,
    pub is_snoozed: bool,
    pub snooze_until: Option<u64>,
    pub has_attachments: bool,
    pub labels: Vec<String>,
    pub thread_id: Option<String>,
    pub crypto_suite: CryptoSuiteView,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EmailDetail {
    pub hash: String,
    pub sender: String,
    pub sender_name: Option<String>,
    pub recipients: Vec<String>,
    pub cc: Vec<String>,
    pub subject: String,
    pub body: String,
    pub timestamp: u64,
    pub priority: EmailPriority,
    pub is_read: bool,
    pub is_starred: bool,
    pub has_attachments: bool,
    pub attachments: Vec<AttachmentView>,
    pub thread_id: Option<String>,
    pub in_reply_to: Option<String>,
    pub crypto_suite: CryptoSuiteView,
    pub read_receipt_requested: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum EmailPriority {
    Low,
    Normal,
    High,
    Urgent,
}

impl EmailPriority {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Low => "Low",
            Self::Normal => "Normal",
            Self::High => "High",
            Self::Urgent => "Urgent",
        }
    }

    pub fn css_class(&self) -> &'static str {
        match self {
            Self::Low => "priority-low",
            Self::Normal => "priority-normal",
            Self::High => "priority-high",
            Self::Urgent => "priority-urgent",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CryptoSuiteView {
    pub key_exchange: String,
    pub symmetric: String,
    pub signature: String,
}

impl CryptoSuiteView {
    pub fn is_post_quantum(&self) -> bool {
        self.key_exchange.starts_with("kyber") || self.signature.starts_with("dilithium")
    }

    pub fn short_label(&self) -> &'static str {
        if self.is_post_quantum() {
            "PQC"
        } else {
            "E2E"
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AttachmentView {
    pub hash: String,
    pub filename: String,
    pub mime_type: String,
    pub size_bytes: u64,
}

impl AttachmentView {
    pub fn size_display(&self) -> String {
        if self.size_bytes < 1024 {
            format!("{} B", self.size_bytes)
        } else if self.size_bytes < 1024 * 1024 {
            format!("{:.1} KB", self.size_bytes as f64 / 1024.0)
        } else {
            format!("{:.1} MB", self.size_bytes as f64 / (1024.0 * 1024.0))
        }
    }
}

// ── Draft Types ──

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DraftView {
    pub hash: String,
    pub recipients: Vec<String>,
    pub cc: Vec<String>,
    pub bcc: Vec<String>,
    pub subject: String,
    pub body: String,
    pub in_reply_to: Option<String>,
    pub updated_at: u64,
    pub scheduled_for: Option<u64>,
}

// ── Folder Types ──

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct FolderView {
    pub hash: String,
    pub name: String,
    pub is_system: bool,
    pub sort_order: i32,
    pub unread_count: u32,
}

impl FolderView {
    pub fn icon(&self) -> &'static str {
        match self.name.as_str() {
            "Inbox" => "\u{1F4E5}",
            "Sent" => "\u{1F4E4}",
            "Drafts" => "\u{1F4DD}",
            "Starred" => "\u{2B50}",
            "Archive" => "\u{1F4E6}",
            "Spam" => "\u{26A0}",
            "Trash" => "\u{1F5D1}",
            _ => "\u{1F4C1}",
        }
    }
}

// ── Contact Types ──

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ContactView {
    pub hash: String,
    pub id: String,
    pub display_name: String,
    pub nickname: Option<String>,
    pub email: Option<String>,
    pub agent_pub_key: Option<String>,
    pub organization: Option<String>,
    pub avatar: Option<String>,
    pub groups: Vec<String>,
    pub is_favorite: bool,
    pub is_blocked: bool,
    pub email_count: u32,
    pub trust_score: Option<f64>,
}

impl ContactView {
    pub fn initials(&self) -> String {
        self.display_name
            .split_whitespace()
            .filter_map(|w| w.chars().next())
            .take(2)
            .collect::<String>()
            .to_uppercase()
    }

    pub fn short_agent(&self) -> String {
        self.agent_pub_key
            .as_deref()
            .map(|k| {
                if k.len() > 12 {
                    format!("{}...{}", &k[..6], &k[k.len()-6..])
                } else {
                    k.to_string()
                }
            })
            .unwrap_or_default()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ContactGroupView {
    pub id: String,
    pub name: String,
    pub color: Option<String>,
    pub member_count: u32,
}

// ── Trust Types ──

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrustScoreView {
    pub agent: String,
    pub score: f64,
    pub confidence: f64,
    pub direct_count: u32,
    pub transitive_count: u32,
}

impl TrustScoreView {
    pub fn tier_label(&self) -> &'static str {
        if self.score >= 0.8 { "Trusted" }
        else if self.score >= 0.5 { "Known" }
        else if self.score >= 0.2 { "Acquaintance" }
        else if self.score >= 0.0 { "Unknown" }
        else { "Distrusted" }
    }

    pub fn tier_css(&self) -> &'static str {
        if self.score >= 0.8 { "trust-high" }
        else if self.score >= 0.5 { "trust-medium" }
        else if self.score >= 0.0 { "trust-low" }
        else { "trust-negative" }
    }
}

// ── Key Status Types ──

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum BundleStatus {
    Ok,
    NoBundle,
    ExpiringSoon,
    Expired,
    LowOnKeys(u32),
}

impl BundleStatus {
    pub fn label(&self) -> String {
        match self {
            Self::Ok => "Keys healthy".to_string(),
            Self::NoBundle => "No keys published".to_string(),
            Self::ExpiringSoon => "Keys expiring soon".to_string(),
            Self::Expired => "Keys expired".to_string(),
            Self::LowOnKeys(n) => format!("{n} one-time keys remaining"),
        }
    }

    pub fn is_healthy(&self) -> bool {
        matches!(self, Self::Ok)
    }

    pub fn css_class(&self) -> &'static str {
        match self {
            Self::Ok => "key-status-ok",
            Self::LowOnKeys(_) => "key-status-warn",
            _ => "key-status-error",
        }
    }
}

// ── Search Types ──

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SearchResultView {
    pub email: EmailListItem,
    pub score: f64,
    pub matched_terms: Vec<String>,
    pub highlight: Option<String>,
}

// ── Thread Types ──

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ThreadView {
    pub thread_id: String,
    pub subject: String,
    pub participants: Vec<String>,
    pub messages: Vec<EmailListItem>,
    pub last_activity: u64,
    pub unread_count: u32,
}

impl ThreadView {
    pub fn participant_summary(&self) -> String {
        match self.participants.len() {
            0 => "No participants".into(),
            1 => self.participants[0].clone(),
            2 => format!("{}, {}", self.participants[0], self.participants[1]),
            n => format!("{}, {} and {} others", self.participants[0], self.participants[1], n - 2),
        }
    }
}

// ── Compose Types ──

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ComposeMode {
    New,
    Reply { email_hash: String, sender: String, sender_name: String, subject: String, body: String, thread_id: Option<String> },
    Forward { subject: String, body: String },
}

impl Default for ComposeMode {
    fn default() -> Self { Self::New }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ComposeState {
    pub recipients: Vec<String>,
    pub cc: Vec<String>,
    pub bcc: Vec<String>,
    pub subject: String,
    pub body: String,
    pub priority: Option<EmailPriority>,
    pub in_reply_to: Option<String>,
    pub request_read_receipt: bool,
    pub use_pqc: bool,
    pub scheduled_for: Option<u64>,
}

// ── Star Types (#11) ──

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum StarType {
    Yellow,
    Blue,
    Red,
    Orange,
    Green,
    Purple,
    BangYellow,
    BangBlue,
    CheckGreen,
    QuestionPurple,
    InfoBlue,
    WarningRed,
}

impl StarType {
    pub const ALL: &[StarType] = &[
        Self::Yellow, Self::Blue, Self::Red, Self::Orange, Self::Green, Self::Purple,
        Self::BangYellow, Self::BangBlue, Self::CheckGreen, Self::QuestionPurple,
        Self::InfoBlue, Self::WarningRed,
    ];

    pub fn icon(&self) -> &'static str {
        match self {
            Self::Yellow => "\u{2B50}", Self::Blue => "\u{1F535}", Self::Red => "\u{1F534}",
            Self::Orange => "\u{1F7E0}", Self::Green => "\u{1F7E2}", Self::Purple => "\u{1F7E3}",
            Self::BangYellow => "\u{2757}", Self::BangBlue => "\u{2755}",
            Self::CheckGreen => "\u{2705}", Self::QuestionPurple => "\u{2753}",
            Self::InfoBlue => "\u{2139}", Self::WarningRed => "\u{26A0}",
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            Self::Yellow => "Star", Self::Blue => "Blue star", Self::Red => "Red star",
            Self::Orange => "Orange star", Self::Green => "Green star", Self::Purple => "Purple star",
            Self::BangYellow => "Important", Self::BangBlue => "Note",
            Self::CheckGreen => "Done", Self::QuestionPurple => "Question",
            Self::InfoBlue => "Info", Self::WarningRed => "Warning",
        }
    }

    pub fn next(&self) -> Self {
        let all = Self::ALL;
        let idx = all.iter().position(|s| s == self).unwrap_or(0);
        all[(idx + 1) % all.len()]
    }
}

impl Default for StarType {
    fn default() -> Self { Self::Yellow }
}

// ── Labels (#2) ──

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LabelView {
    pub id: String,
    pub name: String,
    pub color: String,
}

impl LabelView {
    pub fn css_var(&self) -> String {
        format!("background: {}; color: white;", self.color)
    }
}

// ── Filter Rules (#1) ──

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FilterRule {
    pub id: String,
    pub name: String,
    pub conditions: FilterConditions,
    pub actions: FilterActions,
    pub enabled: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FilterConditions {
    pub from_contains: Option<String>,
    pub to_contains: Option<String>,
    pub subject_contains: Option<String>,
    pub body_contains: Option<String>,
    pub has_attachment: Option<bool>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FilterActions {
    pub apply_label: Option<String>,
    pub move_to_folder: Option<String>,
    pub mark_read: bool,
    pub star: bool,
    pub archive: bool,
    pub delete: bool,
}

// ── Signatures (#6) ──

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SignatureView {
    pub id: String,
    pub name: String,
    pub body_html: String,
    pub is_default: bool,
    pub use_for_new: bool,
    pub use_for_reply: bool,
}

// ── Vacation Responder (#7) ──

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VacationResponder {
    pub enabled: bool,
    pub subject: String,
    pub body: String,
    pub start_date: Option<u64>,
    pub end_date: Option<u64>,
    pub contacts_only: bool,
}

// ── Swipe Actions (#10) ──

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SwipeAction {
    Archive,
    Delete,
    MarkRead,
    Star,
    Snooze,
    MoveToFolder,
}

impl SwipeAction {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Archive => "Archive", Self::Delete => "Delete",
            Self::MarkRead => "Mark Read", Self::Star => "Star",
            Self::Snooze => "Snooze", Self::MoveToFolder => "Move",
        }
    }
    pub fn icon(&self) -> &'static str {
        match self {
            Self::Archive => "\u{1F4E6}", Self::Delete => "\u{1F5D1}",
            Self::MarkRead => "\u{2709}", Self::Star => "\u{2B50}",
            Self::Snooze => "\u{23F0}", Self::MoveToFolder => "\u{1F4C1}",
        }
    }
    pub fn color(&self) -> &'static str {
        match self {
            Self::Archive => "#4ade80", Self::Delete => "#ef4444",
            Self::MarkRead => "#60a5fa", Self::Star => "#fbbf24",
            Self::Snooze => "#a78bfa", Self::MoveToFolder => "#06D6C8",
        }
    }
}

// ── Reading Pane (#5) ──

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReadingPanePosition {
    Off,
    Right,
    Bottom,
}

impl ReadingPanePosition {
    pub fn label(&self) -> &'static str {
        match self { Self::Off => "Off", Self::Right => "Right", Self::Bottom => "Bottom" }
    }
    pub fn next(&self) -> Self {
        match self { Self::Off => Self::Right, Self::Right => Self::Bottom, Self::Bottom => Self::Off }
    }
}

impl Default for ReadingPanePosition {
    fn default() -> Self { Self::Off }
}

// ── Theme ──

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Theme {
    Dark,
    Light,
}

impl Theme {
    pub fn label(&self) -> &'static str {
        match self { Self::Dark => "Dark", Self::Light => "Light" }
    }
    pub fn toggle(&self) -> Self {
        match self { Self::Dark => Self::Light, Self::Light => Self::Dark }
    }
    pub fn data_attr(&self) -> &'static str {
        match self { Self::Dark => "dark", Self::Light => "light" }
    }
}

impl Default for Theme {
    fn default() -> Self { Self::Dark }
}

// ── Density ──

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Density {
    Compact,
    Default,
    Comfortable,
}

impl Density {
    pub fn label(&self) -> &'static str {
        match self { Self::Compact => "Compact", Self::Default => "Default", Self::Comfortable => "Comfortable" }
    }
    pub fn data_attr(&self) -> &'static str {
        match self { Self::Compact => "compact", Self::Default => "default", Self::Comfortable => "comfortable" }
    }
    pub fn next(&self) -> Self {
        match self { Self::Compact => Self::Default, Self::Default => Self::Comfortable, Self::Comfortable => Self::Compact }
    }
}

impl Default for Density {
    fn default() -> Self { Self::Default }
}

// ── Email Templates (#6) ──

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EmailTemplate {
    pub id: String,
    pub name: String,
    pub subject: String,
    pub body: String,
    pub use_pqc: bool,
}

// ── Offline Queue ──

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum OfflineAction {
    ToggleStar { hash: String },
    ToggleRead { hash: String },
    Archive { hash: String },
    Delete { hash: String },
    MoveToFolder { hash: String, folder_hash: String },
    Send { to: String, subject: String, body: String, use_pqc: bool },
}

// ── Signal Types (real-time from conductor) ──

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum MailSignalView {
    EmailReceived {
        email_hash: String,
        sender: String,
        sender_name: Option<String>,
        subject: Option<String>,
        timestamp: u64,
        priority: EmailPriority,
    },
    ReadReceiptReceived {
        email_hash: String,
        reader: String,
    },
    TypingIndicator {
        sender: String,
        thread_id: Option<String>,
    },
    EmailStateChanged {
        email_hash: String,
        is_read: Option<bool>,
        is_starred: Option<bool>,
        is_archived: Option<bool>,
        is_trashed: Option<bool>,
    },
}

// ── Calendar Types ──

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CalendarEventView {
    pub id: String,
    pub title: String,
    pub description: Option<String>,
    pub location: Option<String>,
    pub start_time: u64,
    pub end_time: u64,
    pub all_day: bool,
    pub recurrence: Recurrence,
    pub category: String,
    pub organizer: Option<String>,
    pub attendees: Vec<AttendeeView>,
    pub rsvp_status: Option<RsvpStatus>,
    pub source: CalendarSource,
    pub color: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Recurrence {
    None,
    Daily,
    Weekly,
    Monthly,
    Yearly,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RsvpStatus {
    Going,
    Maybe,
    NotGoing,
    NeedsAction,
}

impl RsvpStatus {
    pub fn label(&self) -> &'static str {
        match self { Self::Going => "Going", Self::Maybe => "Maybe", Self::NotGoing => "Not Going", Self::NeedsAction => "Respond" }
    }
    pub fn css_class(&self) -> &'static str {
        match self { Self::Going => "rsvp-going", Self::Maybe => "rsvp-maybe", Self::NotGoing => "rsvp-no", Self::NeedsAction => "rsvp-pending" }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CalendarSource {
    Personal,
    Community,
    Hearth,
    External(String),
}

impl CalendarSource {
    pub fn label(&self) -> String {
        match self {
            Self::Personal => "Personal".into(),
            Self::Community => "Community".into(),
            Self::Hearth => "Hearth".into(),
            Self::External(name) => name.clone(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AttendeeView {
    pub name: String,
    pub agent_key: Option<String>,
    pub status: RsvpStatus,
}

// ── Chat Types (Ephemeral Signals / The Pulse) ──

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChatMessage {
    /// Ephemeral ID — NOT a DHT hash (these live in browser memory only)
    pub id: String,
    pub sender: String,
    pub sender_name: Option<String>,
    pub content: String,
    pub timestamp: u64,
    pub reply_to: Option<String>,
    pub reactions: Vec<ChatReaction>,
    pub edited: bool,
    pub channel_id: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChatReaction {
    pub emoji: String,
    pub users: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChatChannel {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    pub is_direct: bool,
    pub members: Vec<String>,
    pub member_names: Vec<String>,
    pub unread_count: u32,
    pub last_message: Option<String>,
    pub last_activity: u64,
    pub pinned: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PresenceStatus {
    Online,
    Away,
    DoNotDisturb,
    Offline,
}

impl PresenceStatus {
    pub fn label(&self) -> &'static str {
        match self { Self::Online => "Online", Self::Away => "Away", Self::DoNotDisturb => "DND", Self::Offline => "Offline" }
    }
    pub fn css_class(&self) -> &'static str {
        match self { Self::Online => "presence-online", Self::Away => "presence-away", Self::DoNotDisturb => "presence-dnd", Self::Offline => "presence-offline" }
    }
}

// ── Meet Types (WebRTC Video/Audio) ──

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CallType {
    Audio,
    Video,
    Screen,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CallParticipant {
    pub agent_key: String,
    pub name: String,
    pub is_muted: bool,
    pub camera_on: bool,
    pub screen_sharing: bool,
    pub hand_raised: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CallStatus {
    Ringing,
    Active,
    Ended,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ActiveCall {
    pub id: String,
    pub call_type: CallType,
    pub participants: Vec<CallParticipant>,
    pub started_at: u64,
    pub status: CallStatus,
    pub channel_id: Option<String>,
}
