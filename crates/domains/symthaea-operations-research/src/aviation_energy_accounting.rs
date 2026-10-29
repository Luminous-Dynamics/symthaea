// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Fail-closed P0 energy/lifecycle-accounting oracle for the clean-airline benchmark.
//!
//! This module aggregates explicitly supplied, non-overlapping accounting records.
//! It is not an aircraft-energy model, fuel lifecycle-factor database, emissions
//! estimator, dispatcher, or operational decision authority. Callers must ensure
//! all records share the declared lifecycle boundary and planning period.

use std::collections::BTreeSet;
use std::fmt;

/// Evidence maturity for an input record. This metadata does not make a claim true.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvidenceClass {
    Measured,
    SupplierReported,
    LiteratureDerived,
    CalibratedModel,
    UncalibratedModel,
    Synthetic,
}

/// Energy carrier at the declared source/use boundary.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum EnergyCarrier {
    Electricity,
    Hydrogen,
    SustainableAviationFuel,
    SyntheticAviationFuel,
    FossilJetFuel,
    Other(String),
}

/// Non-CO2 effects are carried separately; this module never silently folds them
/// into the CO2e total or aggregates unlike metrics into a synthetic score.
#[derive(Debug, Clone, PartialEq)]
pub enum NonCo2Evidence {
    NotAssessed,
    Qualitative {
        method_ref: String,
        summary: String,
    },
    Quantified {
        method_ref: String,
        metric_name: String,
        value: f64,
        unit: String,
    },
}

/// One non-overlapping energy-use accounting row.
///
/// \`source_energy_kwh\` is the energy of the declared carrier procured at the chosen
/// boundary; \`delivered_energy_kwh\` is energy of that same carrier delivered to the
/// declared service boundary after distribution/storage losses. Both must use the
/// same carrier-energy convention. Upstream emissions include the declared upstream
/// lifecycle stages exactly once; direct emissions cover the declared operational
/// boundary. Do not include upstream production electricity again as a separate
/// energy-use row if it is already included in the fuel's upstream emissions.
#[derive(Debug, Clone, PartialEq)]
pub struct EnergyRecord {
    pub id: String,
    pub boundary_id: String,
    pub period_id: String,
    pub evidence_ref: String,
    pub evidence_class: EvidenceClass,
    pub carrier: EnergyCarrier,
    pub source_energy_kwh: f64,
    pub delivered_energy_kwh: f64,
    pub upstream_co2e_kg: f64,
    pub direct_operational_co2e_kg: f64,
    pub cost_amount: f64,
    pub currency_code: String,
    pub non_co2: NonCo2Evidence,
}

/// Transport service in the exact same planning period as all ledger records.
#[derive(Debug, Clone, PartialEq)]
pub struct TransportService {
    pub period_id: String,
    pub completed_passenger_km: f64,
    pub available_seat_km: f64,
}

/// Per-record non-CO2 evidence, preserved without mixing distinct methods/units.
#[derive(Debug, Clone, PartialEq)]
pub struct NonCo2Observation {
    pub record_id: String,
    pub method_ref: String,
    pub kind: NonCo2ObservationKind,
}

#[derive(Debug, Clone, PartialEq)]
pub enum NonCo2ObservationKind {
    Qualitative { summary: String },
    Quantified {
        metric_name: String,
        value: f64,
        unit: String,
    },
}

/// Independently recomputable aggregates. \`accounted_lifecycle_co2e_kg\` is only
/// the sum of the declared upstream + direct CO2e fields. It is not a complete
/// climate-impact score and does not include the separate non-CO2 observations.
#[derive(Debug, Clone, PartialEq)]
pub struct AccountingSummary {
    pub period_id: String,
    pub currency_code: String,
    pub total_source_energy_kwh: f64,
    pub total_delivered_energy_kwh: f64,
    pub upstream_co2e_kg: f64,
    pub direct_operational_co2e_kg: f64,
    pub accounted_lifecycle_co2e_kg: f64,
    pub total_cost: f64,
    pub completed_passenger_km: f64,
    pub available_seat_km: f64,
    pub accounted_co2e_kg_per_completed_passenger_km: f64,
    pub cost_per_completed_passenger_km: f64,
    pub cost_per_available_seat_km: f64,
    pub non_co2_not_assessed_record_ids: Vec<String>,
    pub non_co2_observations: Vec<NonCo2Observation>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AccountingError {
    EmptyLedger,
    MissingIdentifier(&'static str),
    DuplicateIdentifier(&'static str),
    NonFiniteValue(&'static str),
    NegativeValue(&'static str),
    InvalidSourceEnergy,
    DeliveredEnergyExceedsSource { record_id: String },
    CurrencyMismatch,
    InvalidCurrencyCode,
    PeriodMismatch,
    InvalidServiceDenominator,
    NoDeliveredEnergy,
    InvalidNonCo2Evidence,
    ArithmeticOverflow,
}

impl fmt::Display for AccountingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyLedger => write!(f, "energy ledger must not be empty"),
            Self::MissingIdentifier(field) => write!(f, "missing identifier: {field}"),
            Self::DuplicateIdentifier(field) => write!(f, "duplicate identifier: {field}"),
            Self::NonFiniteValue(field) => write!(f, "non-finite value: {field}"),
            Self::NegativeValue(field) => write!(f, "negative value: {field}"),
            Self::InvalidSourceEnergy => write!(f, "source energy must be greater than zero"),
            Self::DeliveredEnergyExceedsSource { record_id } => {
                write!(f, "delivered energy exceeds source energy for record {record_id}")
            }
            Self::CurrencyMismatch => write!(f, "ledger records use different currencies"),
            Self::InvalidCurrencyCode => write!(f, "currency code must be three uppercase ASCII letters"),
            Self::PeriodMismatch => write!(f, "ledger and transport service periods do not match"),
            Self::InvalidServiceDenominator => {
                write!(f, "passenger-km and seat-km denominators must be finite and positive")
            }
            Self::NoDeliveredEnergy => write!(f, "transport service has no delivered energy"),
            Self::InvalidNonCo2Evidence => write!(f, "invalid non-CO2 evidence metadata or value"),
            Self::ArithmeticOverflow => write!(f, "non-finite aggregate or derived metric"),
        }
    }
}

impl std::error::Error for AccountingError {}

/// Validate and aggregate explicitly declared energy, cost, service and CO2e inputs.
///
/// No default emission factor or price is invented. This function cannot determine
/// whether a caller's lifecycle boundary is scientifically complete; the boundary
/// and evidence must be inspected independently.
pub fn account_energy_ledger(
    records: &[EnergyRecord],
    service: &TransportService,
) -> Result<AccountingSummary, AccountingError> {
    if records.is_empty() {
        return Err(AccountingError::EmptyLedger);
    }
    require_identifier(&service.period_id, "service period id")?;
    require_positive_finite(service.completed_passenger_km, "completed passenger-km")?;
    require_positive_finite(service.available_seat_km, "available seat-km")?;

    let mut record_ids = BTreeSet::new();
    let mut boundary_ids = BTreeSet::new();
    let mut currency: Option<&str> = None;

    let mut total_source_energy_kwh = 0.0;
    let mut total_delivered_energy_kwh = 0.0;
    let mut upstream_co2e_kg = 0.0;
    let mut direct_operational_co2e_kg = 0.0;
    let mut total_cost = 0.0;
    let mut non_co2_not_assessed_record_ids = Vec::new();
    let mut non_co2_observations = Vec::new();

    for record in records {
        require_identifier(&record.id, "record id")?;
        require_identifier(&record.boundary_id, "accounting boundary id")?;
        require_identifier(&record.period_id, "record period id")?;
        require_identifier(&record.evidence_ref, "evidence reference")?;

        if !record_ids.insert(record.id.as_str()) {
            return Err(AccountingError::DuplicateIdentifier("record id"));
        }
        if !boundary_ids.insert(record.boundary_id.as_str()) {
            return Err(AccountingError::DuplicateIdentifier("accounting boundary id"));
        }
        if record.period_id != service.period_id {
            return Err(AccountingError::PeriodMismatch);
        }

        if record.currency_code.len() != 3
            || !record.currency_code.bytes().all(|byte| byte.is_ascii_uppercase())
        {
            return Err(AccountingError::InvalidCurrencyCode);
        }
        match currency {
            Some(existing) if existing != record.currency_code => {
                return Err(AccountingError::CurrencyMismatch);
            }
            None => currency = Some(record.currency_code.as_str()),
            _ => {}
        }

        require_finite(record.source_energy_kwh, "source energy kWh")?;
        if record.source_energy_kwh <= 0.0 {
            return Err(AccountingError::InvalidSourceEnergy);
        }
        require_nonnegative_finite(record.delivered_energy_kwh, "delivered energy kWh")?;
        require_finite(record.upstream_co2e_kg, "upstream CO2e kg")?;
        require_nonnegative_finite(
            record.direct_operational_co2e_kg,
            "direct operational CO2e kg",
        )?;
        require_nonnegative_finite(record.cost_amount, "cost amount")?;

        if record.delivered_energy_kwh > record.source_energy_kwh {
            return Err(AccountingError::DeliveredEnergyExceedsSource {
                record_id: record.id.clone(),
            });
        }

        add_finite(&mut total_source_energy_kwh, record.source_energy_kwh)?;
        add_finite(&mut total_delivered_energy_kwh, record.delivered_energy_kwh)?;
        add_finite(&mut upstream_co2e_kg, record.upstream_co2e_kg)?;
        add_finite(&mut direct_operational_co2e_kg, record.direct_operational_co2e_kg)?;
        add_finite(&mut total_cost, record.cost_amount)?;

        match &record.non_co2 {
            NonCo2Evidence::NotAssessed => {
                non_co2_not_assessed_record_ids.push(record.id.clone());
            }
            NonCo2Evidence::Qualitative { method_ref, summary } => {
                require_identifier(method_ref, "non-CO2 method reference")?;
                require_identifier(summary, "non-CO2 qualitative summary")?;
                non_co2_observations.push(NonCo2Observation {
                    record_id: record.id.clone(),
                    method_ref: method_ref.clone(),
                    kind: NonCo2ObservationKind::Qualitative {
                        summary: summary.clone(),
                    },
                });
            }
            NonCo2Evidence::Quantified {
                method_ref,
                metric_name,
                value,
                unit,
            } => {
                require_identifier(method_ref, "non-CO2 method reference")?;
                require_identifier(metric_name, "non-CO2 metric name")?;
                require_identifier(unit, "non-CO2 metric unit")?;
                if !value.is_finite() {
                    return Err(AccountingError::InvalidNonCo2Evidence);
                }
                non_co2_observations.push(NonCo2Observation {
                    record_id: record.id.clone(),
                    method_ref: method_ref.clone(),
                    kind: NonCo2ObservationKind::Quantified {
                        metric_name: metric_name.clone(),
                        value: *value,
                        unit: unit.clone(),
                    },
                });
            }
        }
    }

    if total_delivered_energy_kwh <= 0.0 {
        return Err(AccountingError::NoDeliveredEnergy);
    }

    let accounted_lifecycle_co2e_kg = upstream_co2e_kg + direct_operational_co2e_kg;
    if !accounted_lifecycle_co2e_kg.is_finite() {
        return Err(AccountingError::ArithmeticOverflow);
    }

    let accounted_co2e_kg_per_completed_passenger_km =
        accounted_lifecycle_co2e_kg / service.completed_passenger_km;
    let cost_per_completed_passenger_km = total_cost / service.completed_passenger_km;
    let cost_per_available_seat_km = total_cost / service.available_seat_km;
    if !accounted_co2e_kg_per_completed_passenger_km.is_finite()
        || !cost_per_completed_passenger_km.is_finite()
        || !cost_per_available_seat_km.is_finite()
    {
        return Err(AccountingError::ArithmeticOverflow);
    }

    non_co2_not_assessed_record_ids.sort();
    non_co2_observations.sort_by(|left, right| left.record_id.cmp(&right.record_id));

    let currency_code = currency.ok_or(AccountingError::MissingIdentifier("currency code"))?;
    Ok(AccountingSummary {
        period_id: service.period_id.clone(),
        currency_code: currency_code.to_owned(),
        total_source_energy_kwh,
        total_delivered_energy_kwh,
        upstream_co2e_kg,
        direct_operational_co2e_kg,
        accounted_lifecycle_co2e_kg,
        total_cost,
        completed_passenger_km: service.completed_passenger_km,
        available_seat_km: service.available_seat_km,
        accounted_co2e_kg_per_completed_passenger_km,
        cost_per_completed_passenger_km,
        cost_per_available_seat_km,
        non_co2_not_assessed_record_ids,
        non_co2_observations,
    })
}

fn require_identifier(value: &str, field: &'static str) -> Result<(), AccountingError> {
    if value.trim().is_empty() {
        return Err(AccountingError::MissingIdentifier(field));
    }
    Ok(())
}

fn require_finite(value: f64, field: &'static str) -> Result<(), AccountingError> {
    if !value.is_finite() {
        return Err(AccountingError::NonFiniteValue(field));
    }
    Ok(())
}

fn require_nonnegative_finite(value: f64, field: &'static str) -> Result<(), AccountingError> {
    require_finite(value, field)?;
    if value < 0.0 {
        return Err(AccountingError::NegativeValue(field));
    }
    Ok(())
}

fn require_positive_finite(value: f64, field: &'static str) -> Result<(), AccountingError> {
    require_finite(value, field)?;
    if value <= 0.0 {
        return Err(AccountingError::InvalidServiceDenominator);
    }
    Ok(())
}

fn add_finite(total: &mut f64, value: f64) -> Result<(), AccountingError> {
    let next = *total + value;
    if !next.is_finite() {
        return Err(AccountingError::ArithmeticOverflow);
    }
    *total = next;
    Ok(())
}
