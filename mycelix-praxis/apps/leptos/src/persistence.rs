// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Simple localStorage persistence layer.
//!
//! No conductor needed -- saves and loads JSON to/from the browser's
//! localStorage API. Used to persist role selection, sovereignty level,
//! and other lightweight state across page reloads.

use serde::{de::DeserializeOwned, Serialize};
use web_sys::window;

/// Save a value to localStorage.
pub fn save<T: Serialize>(key: &str, value: &T) {
    if let Some(storage) = window()
        .and_then(|w| w.local_storage().ok())
        .flatten()
    {
        if let Ok(json) = serde_json::to_string(value) {
            let _ = storage.set_item(key, &json);
        }
    }
}

/// Load a value from localStorage.
pub fn load<T: DeserializeOwned>(key: &str) -> Option<T> {
    window()
        .and_then(|w| w.local_storage().ok())
        .flatten()
        .and_then(|s| s.get_item(key).ok())
        .flatten()
        .and_then(|json| serde_json::from_str(&json).ok())
}

/// Remove a value from localStorage.
pub fn remove(key: &str) {
    if let Some(storage) = window()
        .and_then(|w| w.local_storage().ok())
        .flatten()
    {
        let _ = storage.remove_item(key);
    }
}

// ============================================================
// Progress Export/Import — cross-device portability
// ============================================================

use crate::curriculum::ProgressStore;
use crate::study_tracker::StudyTracker;
use crate::student_profile::StudentProfile;

// ============================================================
// Governance Store — localStorage-first DAO preview
// ============================================================

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct GovernanceStore {
    pub proposals: Vec<LocalProposal>,
    pub next_id: u32,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct LocalProposal {
    pub id: String,
    pub title: String,
    pub description: String,
    pub category: String,
    pub proposer: String,
    pub for_votes: u32,
    pub against_votes: u32,
    pub status: String, // "Active", "Approved", "Rejected"
    pub created_at: String,
    pub voted: bool, // has current user voted
}

const GOV_KEY: &str = "praxis_governance";

impl GovernanceStore {
    pub fn load() -> Self {
        load::<GovernanceStore>(GOV_KEY).unwrap_or_default()
    }

    pub fn save_store(&self) {
        save(GOV_KEY, self);
    }

    pub fn create_proposal(&mut self, title: String, description: String, category: String, proposer: String) {
        let now = js_sys::Date::new_0();
        let date = format!("{:04}-{:02}-{:02}", now.get_full_year(), now.get_month() + 1, now.get_date());
        self.next_id += 1;
        self.proposals.push(LocalProposal {
            id: format!("local_{}", self.next_id),
            title,
            description,
            category,
            proposer,
            for_votes: 0,
            against_votes: 0,
            status: "Active".into(),
            created_at: date,
            voted: false,
        });
        self.save_store();
    }

    pub fn vote(&mut self, proposal_id: &str, is_for: bool) {
        if let Some(p) = self.proposals.iter_mut().find(|p| p.id == proposal_id) {
            if !p.voted {
                if is_for { p.for_votes += 1; } else { p.against_votes += 1; }
                p.voted = true;
                // Auto-resolve: 5+ for votes = approved, 5+ against = rejected
                if p.for_votes >= 5 { p.status = "Approved".into(); }
                if p.against_votes >= 5 { p.status = "Rejected".into(); }
                self.save_store();
            }
        }
    }
}

// ============================================================
// Pending TEND Ledger — Trial Mode Economic Activity
// ============================================================
//
// Tracks TEND credits earned locally before conductor connection.
// When the learner connects to the network, Pending TEND can be
// replayed as real TEND exchanges.
//
// DESIGN PRINCIPLE: Learning is NEVER blocked behind a conductor.
// Pending TEND accumulates freely. Connection unlocks real value.

const TEND_KEY: &str = "praxis_pending_tend";

/// A record of a TEND-earning learning event (pre-conductor).
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct PendingTendEvent {
    /// TEND credits earned for this event
    pub tend_credits: f32,
    /// Learning event type that triggered it
    pub event_type: String,
    /// Quality of the learning session (0-1000 permille)
    pub quality_permille: u16,
    /// Duration in seconds
    pub duration_seconds: u32,
    /// When the event occurred (JS timestamp ms)
    pub timestamp_ms: f64,
    /// Whether this has been synced to the network
    pub synced: bool,
}

/// The Pending TEND ledger — accumulates locally until conductor connects.
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct PendingTendLedger {
    /// Total pending TEND (not yet synced to network)
    pub total_pending: f32,
    /// Total TEND ever earned (including synced)
    pub total_earned: f32,
    /// Individual events
    pub events: Vec<PendingTendEvent>,
    /// Whether the learner has been prompted to connect
    pub connection_prompted: bool,
}

/// Connection readiness threshold: prompt at this many pending TEND
pub const TEND_CONNECTION_THRESHOLD: f32 = 10.0;

impl PendingTendLedger {
    pub fn load() -> Self {
        load::<PendingTendLedger>(TEND_KEY).unwrap_or_default()
    }

    pub fn save_ledger(&self) {
        save(TEND_KEY, self);
    }

    /// Record a learning event that earned TEND credits.
    pub fn record_event(
        &mut self,
        tend_credits: f32,
        event_type: &str,
        quality_permille: u16,
        duration_seconds: u32,
    ) {
        let now = js_sys::Date::now();
        self.events.push(PendingTendEvent {
            tend_credits,
            event_type: event_type.to_string(),
            quality_permille,
            duration_seconds,
            timestamp_ms: now,
            synced: false,
        });
        self.total_pending += tend_credits;
        self.total_earned += tend_credits;
        self.save_ledger();
    }

    /// Check if the learner has earned enough to prompt for connection.
    pub fn should_prompt_connection(&self) -> bool {
        self.total_pending >= TEND_CONNECTION_THRESHOLD && !self.connection_prompted
    }

    /// Mark that the connection prompt has been shown.
    pub fn mark_prompted(&mut self) {
        self.connection_prompted = true;
        self.save_ledger();
    }

    /// Count of unsynced events (for replay on connection).
    pub fn unsynced_count(&self) -> usize {
        self.events.iter().filter(|e| !e.synced).count()
    }

    /// Mark events as synced (after successful conductor replay).
    pub fn mark_synced(&mut self, count: usize) {
        let mut remaining = count;
        for event in &mut self.events {
            if !event.synced && remaining > 0 {
                event.synced = true;
                self.total_pending -= event.tend_credits;
                remaining -= 1;
            }
        }
        self.total_pending = self.total_pending.max(0.0); // Safety clamp
        self.save_ledger();
    }
}

/// Bundled export of all student data.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct ExportBundle {
    pub version: u32,
    pub exported_at: String,
    pub progress: Option<ProgressStore>,
    pub study_tracker: Option<StudyTracker>,
    pub profile: Option<StudentProfile>,
}

/// Export all student data as a JSON string.
pub fn export_all() -> String {
    let now = js_sys::Date::new_0();
    let exported_at = format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}",
        now.get_full_year(), now.get_month() + 1, now.get_date(),
        now.get_hours(), now.get_minutes(), now.get_seconds()
    );

    let bundle = ExportBundle {
        version: 1,
        exported_at,
        progress: load::<ProgressStore>("praxis_progress"),
        study_tracker: load::<StudyTracker>("praxis_study_tracker"),
        profile: load::<StudentProfile>("praxis_profile"),
    };

    serde_json::to_string_pretty(&bundle).unwrap_or_default()
}

/// Import student data from a JSON string, merging with existing data.
/// Returns a summary of what was imported.
pub fn import_all(json: &str) -> Result<String, String> {
    let bundle: ExportBundle = serde_json::from_str(json)
        .map_err(|e| format!("Invalid format: {}", e))?;

    if bundle.version > 1 {
        return Err("This export is from a newer version of Praxis. Please update the app.".into());
    }

    let mut summary = Vec::new();

    // Merge progress (keep higher mastery, more recent reviews)
    if let Some(imported) = bundle.progress {
        let mut existing = load::<ProgressStore>("praxis_progress").unwrap_or_default();
        let mut merged_count = 0u32;

        for (id, imp_node) in &imported.nodes {
            let existing_node = existing.nodes.entry(id.clone()).or_default();
            if imp_node.mastery_permille > existing_node.mastery_permille {
                existing_node.mastery_permille = imp_node.mastery_permille;
                existing_node.status = imp_node.status;
                merged_count += 1;
            }
            if imp_node.attempts > existing_node.attempts {
                existing_node.attempts = imp_node.attempts;
                existing_node.correct = imp_node.correct;
            }
            if imp_node.last_reviewed.unwrap_or(0.0) > existing_node.last_reviewed.unwrap_or(0.0) {
                existing_node.last_reviewed = imp_node.last_reviewed;
            }
        }

        // Merge BKT states
        for (id, imp_bkt) in &imported.bkt_states {
            let existing_bkt = existing.bkt_states.entry(id.clone()).or_default();
            if imp_bkt.p_mastery > existing_bkt.p_mastery {
                *existing_bkt = imp_bkt.clone();
            }
        }

        // Merge SRS cards
        for (id, imp_card) in &imported.srs_cards {
            let existing_card = existing.srs_cards.entry(id.clone()).or_default();
            if imp_card.repetitions > existing_card.repetitions {
                *existing_card = imp_card.clone();
            }
        }

        save("praxis_progress", &existing);
        summary.push(format!("{} topics merged", imported.nodes.len()));
        if merged_count > 0 {
            summary.push(format!("{} mastery levels updated", merged_count));
        }
    }

    // Merge study tracker (combine study days, keep max)
    if let Some(imported) = bundle.study_tracker {
        let mut existing = load::<StudyTracker>("praxis_study_tracker").unwrap_or_default();
        for day in &imported.study_days {
            if !existing.study_days.contains(day) {
                existing.study_days.push(day.clone());
            }
        }
        existing.study_days.sort();
        existing.total_minutes = existing.total_minutes.max(imported.total_minutes);
        if imported.exam_date.is_some() && existing.exam_date.is_none() {
            existing.exam_date = imported.exam_date;
        }
        save("praxis_study_tracker", &existing);
        summary.push("study tracker merged".into());
    }

    // Import profile (only if current is empty)
    if let Some(imported) = bundle.profile {
        let existing = load::<StudentProfile>("praxis_profile");
        if existing.is_none() || existing.as_ref().map(|p| p.name.is_empty()).unwrap_or(true) {
            save("praxis_profile", &imported);
            summary.push("profile imported".into());
        }
    }

    Ok(summary.join(", "))
}

/// Trigger a browser file download with the given content.
pub fn trigger_download(filename: &str, content: &str) {
    use wasm_bindgen::JsCast;
    let Some(window) = window() else { return };
    let Some(document) = window.document() else { return };

    let blob_parts = js_sys::Array::new();
    blob_parts.push(&wasm_bindgen::JsValue::from_str(content));

    let mut options = web_sys::BlobPropertyBag::new();
    options.type_("application/json");

    if let Ok(blob) = web_sys::Blob::new_with_str_sequence_and_options(&blob_parts, &options) {
        if let Ok(url) = web_sys::Url::create_object_url_with_blob(&blob) {
            if let Ok(a) = document.create_element("a") {
                let _ = a.set_attribute("href", &url);
                let _ = a.set_attribute("download", filename);
                a.unchecked_ref::<web_sys::HtmlElement>().click();
                let _ = web_sys::Url::revoke_object_url(&url);
            }
        }
    }
}
