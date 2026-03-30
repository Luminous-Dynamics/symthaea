// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Health shared types — mirrors mycelix-health zome entries without HDI.

use serde::{Deserialize, Serialize};

// ── Patient Records ──

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HealthRecord {
    pub id: String,
    pub patient_did: String,
    pub record_type: RecordType,
    pub encrypted_payload: Vec<u8>,
    pub metadata: RecordMetadata,
    pub created_at: i64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum RecordType { LabResult, Encounter, Vitals, Immunization, Medication, Imaging, Procedure }

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RecordMetadata {
    pub provider_did: Option<String>,
    pub facility: Option<String>,
    pub date_of_service: i64,
    pub sensitivity: SensitivityLevel,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum SensitivityLevel { Normal, Part2Protected, MentalHealth, Restricted }

// ── Consent ──

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Consent {
    pub id: String,
    pub patient_did: String,
    pub recipient_did: String,
    pub categories: Vec<ConsentCategory>,
    pub purpose: String,
    pub granted_at: i64,
    pub expires_at: Option<i64>,
    pub revoked_at: Option<i64>,
    pub no_further_disclosure: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum ConsentCategory { LabResults, Medications, Vitals, SubstanceAbuse, MentalHealth, Encounters, Immunizations, All }

// ── Privacy Budget ──

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PrivacyBudget {
    pub patient_did: String,
    pub epsilon_total: f64,
    pub epsilon_spent: f64,
    pub delta: f64,
    pub fl_contributions: u32,
    pub last_contribution_at: Option<i64>,
}

// ── Data Dividends ──

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DataDividend {
    pub id: String,
    pub patient_did: String,
    pub cohort_id: String,
    pub cohort_name: String,
    pub amount: u64,
    pub currency: String,
    pub earned_at: i64,
    pub claimed: bool,
}

// ── Access Audit ──

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AccessEvent {
    pub event_type: AccessEventType,
    pub accessor_did: String,
    pub record_id: Option<String>,
    pub timestamp: i64,
    pub chain_hash: Vec<u8>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum AccessEventType { DataAccess, FlContribution, DividendPayout, ConsentChange, BreakGlass }
