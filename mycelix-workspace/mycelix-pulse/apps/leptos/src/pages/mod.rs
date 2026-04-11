// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

mod inbox;
mod compose;
mod read;
mod contacts;
mod search;
mod settings;
mod drafts;
mod accounts;
mod calendar;
mod chat;
mod meet;

pub use inbox::InboxPage;
pub use compose::ComposePage;
pub use read::ReadPage;
pub use contacts::ContactsPage;
pub use search::SearchPage;
pub use settings::SettingsPage;
pub use drafts::DraftsPage;
pub use accounts::AccountsPage;
pub use calendar::CalendarPage;
pub use chat::ChatPage;
pub use meet::MeetPage;
