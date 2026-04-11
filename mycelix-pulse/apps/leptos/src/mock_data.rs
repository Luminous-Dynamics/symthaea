// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Mock data for development without a running Holochain conductor.

use mail_leptos_types::*;
use std::collections::HashMap;

pub fn mock_folders() -> Vec<FolderView> {
    vec![
        FolderView { hash: "f-inbox".into(), name: "Inbox".into(), is_system: true, sort_order: 0, unread_count: 3 },
        FolderView { hash: "f-sent".into(), name: "Sent".into(), is_system: true, sort_order: 1, unread_count: 0 },
        FolderView { hash: "f-drafts".into(), name: "Drafts".into(), is_system: true, sort_order: 2, unread_count: 1 },
        FolderView { hash: "f-starred".into(), name: "Starred".into(), is_system: true, sort_order: 3, unread_count: 0 },
        FolderView { hash: "f-archive".into(), name: "Archive".into(), is_system: true, sort_order: 4, unread_count: 0 },
        FolderView { hash: "f-spam".into(), name: "Spam".into(), is_system: true, sort_order: 5, unread_count: 0 },
        FolderView { hash: "f-trash".into(), name: "Trash".into(), is_system: true, sort_order: 6, unread_count: 0 },
    ]
}

pub fn mock_inbox() -> Vec<EmailListItem> {
    let now = (js_sys::Date::now() / 1000.0) as u64;
    vec![
        EmailListItem {
            hash: "e-001".into(),
            sender: "uhCAk_alice_pubkey_mock".into(),
            sender_name: Some("Alice Nakamura".into()),
            encrypted_subject: vec![],
            subject: Some("Governance proposal: Water stewardship council".into()),
            snippet: Some("Hi, I've drafted a new proposal for establishing the water stewardship council. The key points are around democratic water access rights and community-managed aquifer monitoring...".into()),
            timestamp: 1743724800,
            priority: EmailPriority::Normal,
            is_read: false,
            is_starred: true,
            star_type: Some(StarType::Yellow),
            is_pinned: true, is_muted: false, is_snoozed: false, snooze_until: None,
            has_attachments: true,
            labels: vec!["Governance".into()],
            thread_id: Some("t-001".into()),
            crypto_suite: CryptoSuiteView {
                key_exchange: "kyber1024".into(),
                symmetric: "chacha20-poly1305".into(),
                signature: "dilithium3".into(),
            },
        },
        // Thread reply to e-001
        EmailListItem {
            hash: "e-001b".into(),
            sender: "uhCAk_self_mock".into(),
            sender_name: Some("You".into()),
            encrypted_subject: vec![],
            subject: Some("Re: Governance proposal: Water stewardship council".into()),
            snippet: Some("Alice, this looks excellent. I have a few suggestions on the voting threshold — should we require 2/3 supermajority for constitutional amendments?".into()),
            timestamp: 1743726600,
            priority: EmailPriority::Normal,
            is_read: true,
            is_starred: false,
            star_type: None,
            is_pinned: false, is_muted: false, is_snoozed: false, snooze_until: None,
            has_attachments: false,
            labels: vec![],
            thread_id: Some("t-001".into()),
            crypto_suite: CryptoSuiteView {
                key_exchange: "kyber1024".into(),
                symmetric: "chacha20-poly1305".into(),
                signature: "dilithium3".into(),
            },
        },
        // Thread reply to e-001
        EmailListItem {
            hash: "e-001c".into(),
            sender: "uhCAk_alice_pubkey_mock".into(),
            sender_name: Some("Alice Nakamura".into()),
            encrypted_subject: vec![],
            subject: Some("Re: Governance proposal: Water stewardship council".into()),
            snippet: Some("Great point on the supermajority. I've updated the proposal to include that. Also added a sunset clause for the emergency powers section. Can you review section 4.2?".into()),
            timestamp: 1743728400,
            priority: EmailPriority::Normal,
            is_read: false,
            is_starred: false,
            star_type: None,
            is_pinned: false, is_muted: false, is_snoozed: false, snooze_until: None,
            has_attachments: true,
            labels: vec!["Governance".into()],
            thread_id: Some("t-001".into()),
            crypto_suite: CryptoSuiteView {
                key_exchange: "kyber1024".into(),
                symmetric: "chacha20-poly1305".into(),
                signature: "dilithium3".into(),
            },
        },
        EmailListItem {
            hash: "e-002".into(),
            sender: "uhCAk_bob_pubkey_mock".into(),
            sender_name: Some("Bob Mthembu".into()),
            encrypted_subject: vec![],
            subject: Some("Re: Commons resource mesh sync issue".into()),
            snippet: Some("I tracked down the DHT propagation delay. The mesh-time zome needs a longer grace period for...".into()),
            timestamp: 1743721200,
            priority: EmailPriority::High,
            is_read: false,
            is_starred: false,
            star_type: None,
            is_pinned: false, is_muted: false, is_snoozed: false, snooze_until: None,
            has_attachments: false,
            labels: vec![],
            thread_id: Some("t-002".into()),
            crypto_suite: CryptoSuiteView {
                key_exchange: "x25519".into(),
                symmetric: "chacha20-poly1305".into(),
                signature: "ed25519".into(),
            },
        },
        EmailListItem {
            hash: "e-003".into(),
            sender: "uhCAk_carol_pubkey_mock".into(),
            sender_name: Some("Carol Dlamini".into()),
            encrypted_subject: vec![],
            subject: Some("EduNet curriculum review: Mathematics Gr10-12".into()),
            snippet: Some("The CAPS alignment for the new parabola explorer game is complete. All 47 curriculum nodes map correctly to...".into()),
            timestamp: 1743717600,
            priority: EmailPriority::Normal,
            is_read: true,
            is_starred: false,
            star_type: None,
            is_pinned: false, is_muted: false, is_snoozed: false, snooze_until: None,
            has_attachments: true,
            labels: vec!["Education".into()],
            thread_id: None,
            crypto_suite: CryptoSuiteView {
                key_exchange: "kyber768".into(),
                symmetric: "chacha20-poly1305".into(),
                signature: "dilithium2".into(),
            },
        },
        EmailListItem {
            hash: "e-004".into(),
            sender: "uhCAk_dave_pubkey_mock".into(),
            sender_name: Some("David Chen".into()),
            encrypted_subject: vec![],
            subject: Some("Emergency: Node health degradation detected".into()),
            snippet: Some("ALERT: Three commons nodes in Zone 4 are showing synchronization failures. The mesh-time heartbeat...".into()),
            timestamp: 1743714000,
            priority: EmailPriority::Urgent,
            is_read: false,
            is_starred: false,
            star_type: None,
            is_pinned: false, is_muted: false, is_snoozed: false, snooze_until: None,
            has_attachments: false,
            labels: vec![],
            thread_id: Some("t-004".into()),
            crypto_suite: CryptoSuiteView {
                key_exchange: "x25519".into(),
                symmetric: "chacha20-poly1305".into(),
                signature: "ed25519".into(),
            },
        },
        EmailListItem {
            hash: "e-005".into(),
            sender: "uhCAk_eve_pubkey_mock".into(),
            sender_name: Some("Eve Moloi".into()),
            encrypted_subject: vec![],
            subject: Some("Music attribution: New streaming royalty model".into()),
            snippet: Some("The TEND-based royalty distribution is live on the test conductor. Artists receive micro-payments in real-time as...".into()),
            timestamp: 1743710400,
            priority: EmailPriority::Normal,
            is_read: true,
            is_starred: true,
            star_type: Some(StarType::CheckGreen),
            is_pinned: false, is_muted: false, is_snoozed: false, snooze_until: None,
            has_attachments: false,
            labels: vec!["Music".into()],
            thread_id: None,
            crypto_suite: CryptoSuiteView {
                key_exchange: "kyber1024".into(),
                symmetric: "chacha20-poly1305".into(),
                signature: "dilithium3".into(),
            },
        },
        // ── Additional realistic emails ──
        EmailListItem {
            hash: "e-006".into(),
            sender: "uhCAk_fatima_pubkey_mock".into(),
            sender_name: Some("Fatima Al-Rashid".into()),
            encrypted_subject: vec![],
            subject: Some("Health cluster: FHIR integration ready for review".into()),
            snippet: Some("The HL7 FHIR adapter for the health zome is passing all compliance tests. Patient records are encrypted at rest with per-field granularity. Ready for peer review before merge.".into()),
            timestamp: now - 7200,
            priority: EmailPriority::High,
            is_read: false, is_starred: false, star_type: None,
            is_pinned: false, is_muted: false, is_snoozed: false, snooze_until: None,
            has_attachments: true,
            labels: vec!["Health".into(), "Review".into()],
            thread_id: None,
            crypto_suite: pqc_suite(),
        },
        EmailListItem {
            hash: "e-007".into(),
            sender: "uhCAk_grace_pubkey_mock".into(),
            sender_name: Some("Grace Okonkwo".into()),
            encrypted_subject: vec![],
            subject: Some("Weekly standup notes — April 6".into()),
            snippet: Some("Quick recap from today's standup: Portal deployment on track, EduNet has 2002 curriculum nodes live, identity cluster DID resolution testing this week.".into()),
            timestamp: now - 10800,
            priority: EmailPriority::Normal,
            is_read: true, is_starred: false, star_type: None,
            is_pinned: false, is_muted: false, is_snoozed: false, snooze_until: None,
            has_attachments: false,
            labels: vec!["Standup".into()],
            thread_id: None,
            crypto_suite: classic_suite(),
        },
        EmailListItem {
            hash: "e-008".into(),
            sender: "uhCAk_henry_pubkey_mock".into(),
            sender_name: Some("Henry Pretorius".into()),
            encrypted_subject: vec![],
            subject: Some("Solar farm proposal — Limpopo site survey results".into()),
            snippet: Some("The site survey for the Limpopo solar installation is complete. 4.2 hectares available, average irradiance 5.8 kWh/m\u{00B2}/day. I've attached the feasibility study and cost projections.".into()),
            timestamp: now - 18000,
            priority: EmailPriority::Normal,
            is_read: false, is_starred: false, star_type: None,
            is_pinned: false, is_muted: false, is_snoozed: false, snooze_until: None,
            has_attachments: true,
            labels: vec!["Energy".into(), "Proposals".into()],
            thread_id: None,
            crypto_suite: pqc_suite(),
        },
        EmailListItem {
            hash: "e-009".into(),
            sender: "uhCAk_iris_pubkey_mock".into(),
            sender_name: Some("Iris Vandenberg".into()),
            encrypted_subject: vec![],
            subject: Some("Reminder: Grant deadline June 16 (PAR-25-100)".into()),
            snippet: Some("Just a reminder that the NIH PAR-25-100 grant deadline is June 16. We need the rehabilitation platform demo ready by then. Budget narrative needs one more revision.".into()),
            timestamp: now - 43200,
            priority: EmailPriority::Urgent,
            is_read: false, is_starred: true, star_type: Some(StarType::Red),
            is_pinned: false, is_muted: false, is_snoozed: false, snooze_until: None,
            has_attachments: true,
            labels: vec!["Grants".into(), "Urgent".into()],
            thread_id: None,
            crypto_suite: classic_suite(),
        },
        EmailListItem {
            hash: "e-010".into(),
            sender: "uhCAk_james_pubkey_mock".into(),
            sender_name: Some("James Nkosi".into()),
            encrypted_subject: vec![],
            subject: Some("Mesh network test results — Johannesburg pilot".into()),
            snippet: Some("The mesh pilot in Soweto achieved 94% uptime over 30 days. 47 nodes active, average hop count 2.3. Latency stays under 200ms for 3-hop paths. Full report attached.".into()),
            timestamp: now - 86400,
            priority: EmailPriority::Normal,
            is_read: true, is_starred: true, star_type: Some(StarType::QuestionPurple),
            is_pinned: false, is_muted: false, is_snoozed: false, snooze_until: None,
            has_attachments: true,
            labels: vec!["Infrastructure".into()],
            thread_id: None,
            crypto_suite: pqc_suite(),
        },
        EmailListItem {
            hash: "e-011".into(),
            sender: "uhCAk_alice_pubkey_mock".into(),
            sender_name: Some("Alice Nakamura".into()),
            encrypted_subject: vec![],
            subject: Some("Re: Governance proposal: Water stewardship council".into()),
            snippet: Some("Great point on the supermajority threshold. I've updated the proposal — 2/3 for constitutional, simple majority for operational. Also added a sunset clause for emergency powers.".into()),
            timestamp: now - 3600,
            priority: EmailPriority::Normal,
            is_read: false, is_starred: false, star_type: None,
            is_pinned: false, is_muted: false, is_snoozed: false, snooze_until: None,
            has_attachments: false,
            labels: vec!["Governance".into()],
            thread_id: Some("t-001".into()),
            crypto_suite: pqc_suite(),
        },
        EmailListItem {
            hash: "e-012".into(),
            sender: "uhCAk_kwame_pubkey_mock".into(),
            sender_name: Some("Kwame Asante".into()),
            encrypted_subject: vec![],
            subject: Some("Supply chain: Fair trade verification on-chain".into()),
            snippet: Some("We've deployed the provenance tracking zome with QR-code scanning at origin. Farmers in Ghana can now register their cocoa batches with GPS + timestamp proof.".into()),
            timestamp: now - 129600,
            priority: EmailPriority::Normal,
            is_read: true, is_starred: false, star_type: None,
            is_pinned: false, is_muted: false, is_snoozed: false, snooze_until: None,
            has_attachments: false,
            labels: vec!["Supply Chain".into()],
            thread_id: None,
            crypto_suite: pqc_suite(),
        },
        EmailListItem {
            hash: "e-013".into(),
            sender: "uhCAk_lin_pubkey_mock".into(),
            sender_name: Some("Lin Wei".into()),
            encrypted_subject: vec![],
            subject: Some("Knowledge graph: Climate data integration".into()),
            snippet: Some("Cross-linked the climate monitoring data with the energy production metrics. The knowledge zome's inference engine found 3 non-obvious correlations worth investigating.".into()),
            timestamp: now - 172800,
            priority: EmailPriority::Low,
            is_read: true, is_starred: false, star_type: None,
            is_pinned: false, is_muted: false, is_snoozed: false, snooze_until: None,
            has_attachments: false,
            labels: vec!["Knowledge".into(), "Climate".into()],
            thread_id: None,
            crypto_suite: classic_suite(),
        },
        EmailListItem {
            hash: "e-014".into(),
            sender: "uhCAk_maria_pubkey_mock".into(),
            sender_name: Some("Maria Santos".into()),
            encrypted_subject: vec![],
            subject: Some("Community care network — new mutual aid zome features".into()),
            snippet: Some("The mutual aid coordinator now supports time-banking credits. Members can offer skills (tutoring, repairs, childcare) and earn TEND credits redeemable across the commons.".into()),
            timestamp: now - 259200,
            priority: EmailPriority::Normal,
            is_read: true, is_starred: false, star_type: None,
            is_pinned: false, is_muted: false, is_snoozed: false, snooze_until: None,
            has_attachments: false,
            labels: vec!["Commons".into(), "Care".into()],
            thread_id: None,
            crypto_suite: pqc_suite(),
        },
    ]
}

fn pqc_suite() -> CryptoSuiteView {
    CryptoSuiteView {
        key_exchange: "kyber1024".into(),
        symmetric: "chacha20-poly1305".into(),
        signature: "dilithium3".into(),
    }
}

fn classic_suite() -> CryptoSuiteView {
    CryptoSuiteView {
        key_exchange: "x25519".into(),
        symmetric: "chacha20-poly1305".into(),
        signature: "ed25519".into(),
    }
}

pub fn mock_contacts() -> Vec<ContactView> {
    vec![
        ContactView {
            hash: "c-001".into(), id: "alice".into(),
            display_name: "Alice Nakamura".into(), nickname: None,
            email: Some("alice@mycelix.net".into()),
            agent_pub_key: Some("uhCAk_alice_pubkey_mock".into()),
            organization: Some("Water Stewardship Council".into()),
            avatar: None, groups: vec!["Governance".into()],
            is_favorite: true, is_blocked: false, email_count: 47,
            trust_score: Some(0.85),
        },
        ContactView {
            hash: "c-002".into(), id: "bob".into(),
            display_name: "Bob Mthembu".into(), nickname: Some("Bob".into()),
            email: Some("bob@mycelix.net".into()),
            agent_pub_key: Some("uhCAk_bob_pubkey_mock".into()),
            organization: Some("Commons Infrastructure".into()),
            avatar: None, groups: vec!["Engineering".into()],
            is_favorite: false, is_blocked: false, email_count: 23,
            trust_score: Some(0.72),
        },
        ContactView {
            hash: "c-003".into(), id: "carol".into(),
            display_name: "Carol Dlamini".into(), nickname: None,
            email: Some("carol@mycelix.net".into()),
            agent_pub_key: Some("uhCAk_carol_pubkey_mock".into()),
            organization: Some("EduNet Curriculum".into()),
            avatar: None, groups: vec!["Education".into()],
            is_favorite: true, is_blocked: false, email_count: 15,
            trust_score: Some(0.91),
        },
        ContactView {
            hash: "c-004".into(), id: "dave".into(),
            display_name: "David Chen".into(), nickname: Some("Dave".into()),
            email: Some("dave@mycelix.net".into()),
            agent_pub_key: Some("uhCAk_dave_pubkey_mock".into()),
            organization: Some("Emergency Response".into()),
            avatar: None, groups: vec!["Operations".into()],
            is_favorite: false, is_blocked: false, email_count: 8,
            trust_score: Some(0.68),
        },
        ContactView {
            hash: "c-005".into(), id: "eve".into(),
            display_name: "Eve Moloi".into(), nickname: None,
            email: Some("eve@mycelix.net".into()),
            agent_pub_key: Some("uhCAk_eve_pubkey_mock".into()),
            organization: Some("Music Attribution".into()),
            avatar: None, groups: vec!["Creative".into()],
            is_favorite: false, is_blocked: false, email_count: 31,
            trust_score: Some(0.79),
        },
        ContactView {
            hash: "c-006".into(), id: "fatima".into(),
            display_name: "Fatima Al-Rashid".into(), nickname: None,
            email: Some("fatima@mycelix.net".into()),
            agent_pub_key: Some("uhCAk_fatima_pubkey_mock".into()),
            organization: Some("Health Cluster".into()),
            avatar: None, groups: vec!["Health".into()],
            is_favorite: true, is_blocked: false, email_count: 19,
            trust_score: Some(0.88),
        },
        ContactView {
            hash: "c-007".into(), id: "grace".into(),
            display_name: "Grace Okonkwo".into(), nickname: None,
            email: Some("grace@mycelix.net".into()),
            agent_pub_key: Some("uhCAk_grace_pubkey_mock".into()),
            organization: Some("Project Management".into()),
            avatar: None, groups: vec!["Operations".into()],
            is_favorite: false, is_blocked: false, email_count: 56,
            trust_score: Some(0.82),
        },
        ContactView {
            hash: "c-008".into(), id: "henry".into(),
            display_name: "Henry Pretorius".into(), nickname: None,
            email: Some("henry@mycelix.net".into()),
            agent_pub_key: Some("uhCAk_henry_pubkey_mock".into()),
            organization: Some("Energy Cooperative".into()),
            avatar: None, groups: vec!["Energy".into()],
            is_favorite: false, is_blocked: false, email_count: 12,
            trust_score: Some(0.75),
        },
        ContactView {
            hash: "c-009".into(), id: "iris".into(),
            display_name: "Iris Vandenberg".into(), nickname: None,
            email: Some("iris@mycelix.net".into()),
            agent_pub_key: Some("uhCAk_iris_pubkey_mock".into()),
            organization: Some("Grant Writing".into()),
            avatar: None, groups: vec!["Grants".into()],
            is_favorite: true, is_blocked: false, email_count: 34,
            trust_score: Some(0.93),
        },
        ContactView {
            hash: "c-010".into(), id: "james".into(),
            display_name: "James Nkosi".into(), nickname: None,
            email: Some("james@mycelix.net".into()),
            agent_pub_key: Some("uhCAk_james_pubkey_mock".into()),
            organization: Some("Mesh Network Ops".into()),
            avatar: None, groups: vec!["Infrastructure".into()],
            is_favorite: false, is_blocked: false, email_count: 28,
            trust_score: Some(0.80),
        },
    ]
}

/// Map sender agent keys to trust scores for inline display.
pub fn mock_sender_trust() -> HashMap<String, f64> {
    let mut m = HashMap::new();
    m.insert("uhCAk_alice_pubkey_mock".into(), 0.85);
    m.insert("uhCAk_bob_pubkey_mock".into(), 0.72);
    m.insert("uhCAk_carol_pubkey_mock".into(), 0.91);
    m.insert("uhCAk_dave_pubkey_mock".into(), 0.68);
    m.insert("uhCAk_eve_pubkey_mock".into(), 0.79);
    m.insert("uhCAk_fatima_pubkey_mock".into(), 0.88);
    m.insert("uhCAk_grace_pubkey_mock".into(), 0.82);
    m.insert("uhCAk_henry_pubkey_mock".into(), 0.75);
    m.insert("uhCAk_iris_pubkey_mock".into(), 0.93);
    m.insert("uhCAk_james_pubkey_mock".into(), 0.80);
    m.insert("uhCAk_kwame_pubkey_mock".into(), 0.77);
    m.insert("uhCAk_lin_pubkey_mock".into(), 0.70);
    m.insert("uhCAk_maria_pubkey_mock".into(), 0.86);
    m
}

pub fn mock_drafts() -> Vec<DraftView> {
    vec![
        DraftView {
            hash: "d-001".into(),
            recipients: vec!["uhCAk_alice_pubkey_mock".into()],
            cc: vec![], bcc: vec![],
            subject: "Re: Governance proposal".into(),
            body: "Alice, I've reviewed the proposal and have a few suggestions regarding the voting threshold...".into(),
            in_reply_to: Some("e-001".into()),
            updated_at: 1743720000,
            scheduled_for: None,
        },
    ]
}

pub fn mock_labels() -> Vec<LabelView> {
    vec![
        LabelView { id: "l-gov".into(), name: "Governance".into(), color: "#8b7ec8".into() },
        LabelView { id: "l-eng".into(), name: "Engineering".into(), color: "#06D6C8".into() },
        LabelView { id: "l-edu".into(), name: "Education".into(), color: "#4ade80".into() },
        LabelView { id: "l-ops".into(), name: "Operations".into(), color: "#f59e0b".into() },
        LabelView { id: "l-music".into(), name: "Music".into(), color: "#ec4899".into() },
        LabelView { id: "l-urgent".into(), name: "Urgent".into(), color: "#ef4444".into() },
    ]
}

pub fn mock_filters() -> Vec<FilterRule> {
    vec![
        FilterRule {
            id: "f-001".into(),
            name: "Governance to label".into(),
            conditions: FilterConditions {
                from_contains: None,
                to_contains: None,
                subject_contains: Some("governance".into()),
                body_contains: None,
                has_attachment: None,
            },
            actions: FilterActions {
                apply_label: Some("Governance".into()),
                move_to_folder: None,
                mark_read: false, star: false, archive: false, delete: false,
            },
            enabled: true,
        },
        FilterRule {
            id: "f-002".into(),
            name: "Emergency auto-star".into(),
            conditions: FilterConditions {
                from_contains: Some("dave".into()),
                to_contains: None,
                subject_contains: Some("emergency".into()),
                body_contains: None,
                has_attachment: None,
            },
            actions: FilterActions {
                apply_label: Some("Urgent".into()),
                move_to_folder: None,
                mark_read: false, star: true, archive: false, delete: false,
            },
            enabled: true,
        },
    ]
}

pub fn mock_signatures() -> Vec<SignatureView> {
    vec![
        SignatureView {
            id: "sig-1".into(),
            name: "Default".into(),
            body_html: "<p>Best regards,<br>— Sent via <strong>Mycelix Pulse</strong> (PQC encrypted)</p>".into(),
            is_default: true,
            use_for_new: true,
            use_for_reply: false,
        },
        SignatureView {
            id: "sig-2".into(),
            name: "Formal".into(),
            body_html: "<p>Kind regards,<br><br>Tristan Stoltz<br>Luminous Dynamics<br>tristan.stoltz@evolvingresonantcocreationism.com</p>".into(),
            is_default: false,
            use_for_new: false,
            use_for_reply: false,
        },
    ]
}

pub fn mock_vacation() -> VacationResponder {
    VacationResponder {
        enabled: false,
        subject: "Out of office".into(),
        body: "I'm currently away and will respond when I return. For urgent matters, please contact the commons coordinator.".into(),
        start_date: None,
        end_date: None,
        contacts_only: true,
    }
}

pub fn mock_calendar_events() -> Vec<CalendarEventView> {
    let now = (js_sys::Date::now() / 1000.0) as u64;
    let hour = 3600;
    let day = 86400;
    vec![
        CalendarEventView {
            id: "cal-001".into(), title: "Water Council Meeting".into(),
            description: Some("Monthly governance review for water stewardship".into()),
            location: Some("Community Hall, Room 3".into()),
            start_time: now + 2 * day + 14 * hour, end_time: now + 2 * day + 16 * hour,
            all_day: false, recurrence: Recurrence::Monthly,
            category: "Governance".into(), organizer: Some("Alice Nakamura".into()),
            attendees: vec![
                AttendeeView { name: "Alice Nakamura".into(), agent_key: Some("uhCAk_alice_pubkey_mock".into()), status: RsvpStatus::Going },
                AttendeeView { name: "You".into(), agent_key: None, status: RsvpStatus::NeedsAction },
            ],
            rsvp_status: Some(RsvpStatus::NeedsAction),
            source: CalendarSource::Community, color: Some("#8b7ec8".into()),
        },
        CalendarEventView {
            id: "cal-002".into(), title: "EduNet Curriculum Sync".into(),
            description: Some("Review CAPS alignment for Gr10-12 mathematics".into()),
            location: None, start_time: now + day + 10 * hour, end_time: now + day + 11 * hour,
            all_day: false, recurrence: Recurrence::Weekly,
            category: "Education".into(), organizer: Some("Carol Dlamini".into()),
            attendees: vec![],
            rsvp_status: Some(RsvpStatus::Going),
            source: CalendarSource::Community, color: Some("#4ade80".into()),
        },
        CalendarEventView {
            id: "cal-003".into(), title: "Family Dinner".into(),
            description: None, location: Some("Home".into()),
            start_time: now + 6 * hour, end_time: now + 8 * hour,
            all_day: false, recurrence: Recurrence::None,
            category: "Personal".into(), organizer: None,
            attendees: vec![], rsvp_status: None,
            source: CalendarSource::Hearth, color: Some("#f59e0b".into()),
        },
        CalendarEventView {
            id: "cal-004".into(), title: "Infrastructure Maintenance Window".into(),
            description: Some("Scheduled node updates for Zone 4".into()),
            location: None,
            start_time: now + 3 * day, end_time: now + 3 * day + day,
            all_day: true, recurrence: Recurrence::None,
            category: "Operations".into(), organizer: Some("David Chen".into()),
            attendees: vec![], rsvp_status: None,
            source: CalendarSource::Community, color: Some("#ef4444".into()),
        },
    ]
}

pub fn mock_chat_channels() -> Vec<ChatChannel> {
    let now = (js_sys::Date::now() / 1000.0) as u64;
    vec![
        ChatChannel {
            id: "dm-alice".into(), name: "Alice Nakamura".into(),
            description: None, is_direct: true,
            members: vec!["uhCAk_alice_pubkey_mock".into()],
            member_names: vec!["Alice Nakamura".into()],
            unread_count: 2, last_message: Some("See you at the meeting!".into()),
            last_activity: now - 300, pinned: true,
        },
        ChatChannel {
            id: "dm-bob".into(), name: "Bob Mthembu".into(),
            description: None, is_direct: true,
            members: vec!["uhCAk_bob_pubkey_mock".into()],
            member_names: vec!["Bob Mthembu".into()],
            unread_count: 0, last_message: Some("Fixed the sync issue".into()),
            last_activity: now - 3600, pinned: false,
        },
        ChatChannel {
            id: "ch-governance".into(), name: "governance".into(),
            description: Some("Governance discussion and proposals".into()),
            is_direct: false,
            members: vec!["uhCAk_alice_pubkey_mock".into(), "uhCAk_bob_pubkey_mock".into(), "uhCAk_carol_pubkey_mock".into()],
            member_names: vec!["Alice".into(), "Bob".into(), "Carol".into()],
            unread_count: 5, last_message: Some("The proposal vote closes Friday".into()),
            last_activity: now - 600, pinned: true,
        },
        ChatChannel {
            id: "ch-engineering".into(), name: "engineering".into(),
            description: Some("Technical coordination".into()),
            is_direct: false,
            members: vec!["uhCAk_bob_pubkey_mock".into(), "uhCAk_dave_pubkey_mock".into()],
            member_names: vec!["Bob".into(), "Dave".into()],
            unread_count: 0, last_message: Some("Node 4B is back online".into()),
            last_activity: now - 7200, pinned: false,
        },
        ChatChannel {
            id: "ch-music".into(), name: "music".into(),
            description: Some("TEND royalty discussions".into()),
            is_direct: false,
            members: vec!["uhCAk_eve_pubkey_mock".into()],
            member_names: vec!["Eve".into()],
            unread_count: 1, last_message: Some("New streaming model is live!".into()),
            last_activity: now - 1800, pinned: false,
        },
    ]
}

pub fn mock_chat_messages(channel_id: &str) -> Vec<ChatMessage> {
    let now = (js_sys::Date::now() / 1000.0) as u64;
    match channel_id {
        "dm-alice" => vec![
            ChatMessage { id: "cm-1".into(), sender: "uhCAk_alice_pubkey_mock".into(), sender_name: Some("Alice Nakamura".into()), content: "Hey, did you review the water proposal?".into(), timestamp: now - 600, reply_to: None, reactions: vec![], edited: false, channel_id: Some("dm-alice".into()) },
            ChatMessage { id: "cm-2".into(), sender: "uhCAk_self_mock".into(), sender_name: Some("You".into()), content: "Yes, the supermajority clause looks good. I'd add a sunset provision.".into(), timestamp: now - 540, reply_to: None, reactions: vec![ChatReaction { emoji: "\u{1F44D}".into(), users: vec!["uhCAk_alice_pubkey_mock".into()] }], edited: false, channel_id: Some("dm-alice".into()) },
            ChatMessage { id: "cm-3".into(), sender: "uhCAk_alice_pubkey_mock".into(), sender_name: Some("Alice Nakamura".into()), content: "Great idea. I'll update section 4.2. See you at the meeting!".into(), timestamp: now - 300, reply_to: None, reactions: vec![], edited: false, channel_id: Some("dm-alice".into()) },
        ],
        "ch-governance" => vec![
            ChatMessage { id: "cm-g1".into(), sender: "uhCAk_carol_pubkey_mock".into(), sender_name: Some("Carol Dlamini".into()), content: "Reminder: the budget proposal vote closes Friday at 5pm".into(), timestamp: now - 3600, reply_to: None, reactions: vec![], edited: false, channel_id: Some("ch-governance".into()) },
            ChatMessage { id: "cm-g2".into(), sender: "uhCAk_alice_pubkey_mock".into(), sender_name: Some("Alice Nakamura".into()), content: "Current tally: 12 yes, 3 no, 5 abstain. Need 15 for quorum.".into(), timestamp: now - 1800, reply_to: None, reactions: vec![], edited: false, channel_id: Some("ch-governance".into()) },
            ChatMessage { id: "cm-g3".into(), sender: "uhCAk_bob_pubkey_mock".into(), sender_name: Some("Bob Mthembu".into()), content: "I'll reach out to the remaining voters".into(), timestamp: now - 600, reply_to: Some("cm-g2".into()), reactions: vec![ChatReaction { emoji: "\u{1F64F}".into(), users: vec!["uhCAk_alice_pubkey_mock".into(), "uhCAk_carol_pubkey_mock".into()] }], edited: false, channel_id: Some("ch-governance".into()) },
        ],
        _ => vec![],
    }
}

pub fn mock_templates() -> Vec<EmailTemplate> {
    vec![
        EmailTemplate {
            id: "tpl-1".into(),
            name: "Meeting Follow-up".into(),
            subject: "Follow-up: [Meeting Topic]".into(),
            body: "Hi,\n\nThank you for your time today. Here's a summary of what we discussed:\n\n1. \n2. \n3. \n\nNext steps:\n- \n\nBest regards".into(),
            use_pqc: true,
        },
        EmailTemplate {
            id: "tpl-2".into(),
            name: "Governance Proposal".into(),
            subject: "Proposal: [Title]".into(),
            body: "Dear Council Members,\n\nI'd like to propose the following for community consideration:\n\n**Proposal**: \n\n**Rationale**: \n\n**Budget Impact**: \n\n**Timeline**: \n\nPlease review and share your feedback before the next assembly.\n\nIn service".into(),
            use_pqc: true,
        },
        EmailTemplate {
            id: "tpl-3".into(),
            name: "Quick Acknowledgment".into(),
            subject: "".into(),
            body: "Received, thank you. I'll review and respond in detail by end of day.".into(),
            use_pqc: true,
        },
    ]
}

pub fn mock_key_status() -> BundleStatus {
    BundleStatus::Ok
}
