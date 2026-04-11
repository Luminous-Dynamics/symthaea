// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Adapters — map wire types to display-ready formats.

use finance_wire_types as wire;
use hdk::prelude::Record;

/// Extract a typed entry from a Holochain Record.
pub fn extract_entry<T: serde::de::DeserializeOwned>(record: &Record) -> Option<T> {
    record.entry().to_app_option().ok().flatten()
}

pub fn map_payment(record: &Record) -> Option<wire::Payment> {
    extract_entry(record)
}

pub fn map_exchange(record: &Record) -> Option<wire::ExchangeRecord> {
    extract_entry(record)
}

pub fn map_recognition(record: &Record) -> Option<wire::RecognitionEvent> {
    extract_entry(record)
}

pub fn map_stake(record: &Record) -> Option<wire::CollateralStake> {
    extract_entry(record)
}

pub fn map_treasury(record: &Record) -> Option<wire::Treasury> {
    extract_entry(record)
}
