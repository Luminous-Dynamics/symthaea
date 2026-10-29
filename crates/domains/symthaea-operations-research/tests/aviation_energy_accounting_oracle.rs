use symthaea_operations_research::aviation_energy_accounting::{
    account_energy_ledger, AccountingError, EnergyCarrier, EnergyRecord, EvidenceClass,
    NonCo2Evidence, NonCo2ObservationKind, TransportService,
};

fn service() -> TransportService {
    TransportService {
        period_id: "2026-W01".to_owned(),
        completed_passenger_km: 1_000.0,
        available_seat_km: 2_000.0,
    }
}

fn record(
    id: &str,
    boundary_id: &str,
    carrier: EnergyCarrier,
    source_energy_kwh: f64,
    delivered_energy_kwh: f64,
    upstream_co2e_kg: f64,
    direct_operational_co2e_kg: f64,
    cost_amount: f64,
) -> EnergyRecord {
    EnergyRecord {
        id: id.to_owned(),
        boundary_id: boundary_id.to_owned(),
        period_id: "2026-W01".to_owned(),
        evidence_ref: format!("evidence:{id}"),
        evidence_class: EvidenceClass::Synthetic,
        carrier,
        source_energy_kwh,
        delivered_energy_kwh,
        upstream_co2e_kg,
        direct_operational_co2e_kg,
        cost_amount,
        currency_code: "USD".to_owned(),
        non_co2: NonCo2Evidence::NotAssessed,
    }
}

fn known_records() -> Vec<EnergyRecord> {
    let electricity = record(
        "electricity",
        "boundary-electricity",
        EnergyCarrier::Electricity,
        100.0,
        95.0,
        20.0,
        0.0,
        50.0,
    );
    let mut fuel = record(
        "fuel",
        "boundary-fuel",
        EnergyCarrier::SustainableAviationFuel,
        20.0,
        18.0,
        5.0,
        6.0,
        40.0,
    );
    fuel.non_co2 = NonCo2Evidence::Quantified {
        method_ref: "screening-method-v1".to_owned(),
        metric_name: "illustrative_forcing_index".to_owned(),
        value: 0.2,
        unit: "index".to_owned(),
    };
    vec![electricity, fuel]
}

#[test]
fn known_answer_ledger_aggregates_only_declared_fields() {
    let summary = account_energy_ledger(&known_records(), &service()).unwrap();

    assert_eq!(summary.currency_code, "USD");
    assert_eq!(summary.total_source_energy_kwh, 120.0);
    assert_eq!(summary.total_delivered_energy_kwh, 113.0);
    assert_eq!(summary.upstream_co2e_kg, 25.0);
    assert_eq!(summary.direct_operational_co2e_kg, 6.0);
    assert_eq!(summary.accounted_lifecycle_co2e_kg, 31.0);
    assert_eq!(summary.total_cost, 90.0);
    assert!((summary.accounted_co2e_kg_per_completed_passenger_km - 0.031).abs() < 1e-12);
    assert!((summary.cost_per_completed_passenger_km - 0.09).abs() < 1e-12);
    assert!((summary.cost_per_available_seat_km - 0.045).abs() < 1e-12);

    // A metric with its own unit is preserved, not added to CO2e or another metric.
    assert_eq!(summary.non_co2_observations.len(), 1);
    assert_eq!(summary.non_co2_not_assessed_record_ids, vec!["electricity".to_owned()]);
    assert!(matches!(
        &summary.non_co2_observations[0].kind,
        NonCo2ObservationKind::Quantified { .. }
    ));
}

#[test]
fn input_permutation_preserves_canonical_aggregates_and_evidence_order() {
    let mut records = known_records();
    let forward = account_energy_ledger(&records, &service()).unwrap();
    records.reverse();
    let reverse = account_energy_ledger(&records, &service()).unwrap();
    assert_eq!(forward, reverse);
}

#[test]
fn empty_ledger_is_rejected() {
    assert_eq!(
        account_energy_ledger(&[], &service()).unwrap_err(),
        AccountingError::EmptyLedger
    );
}

#[test]
fn duplicate_record_identity_is_rejected() {
    let mut records = known_records();
    records[1].id = records[0].id.clone();
    assert_eq!(
        account_energy_ledger(&records, &service()).unwrap_err(),
        AccountingError::DuplicateIdentifier("record id")
    );
}

#[test]
fn duplicate_accounting_boundary_is_rejected_to_reduce_double_counting_risk() {
    let mut records = known_records();
    records[1].boundary_id = records[0].boundary_id.clone();
    assert_eq!(
        account_energy_ledger(&records, &service()).unwrap_err(),
        AccountingError::DuplicateIdentifier("accounting boundary id")
    );
}

#[test]
fn source_energy_must_be_positive_and_finite() {
    let mut records = known_records();
    records[0].source_energy_kwh = 0.0;
    assert_eq!(
        account_energy_ledger(&records, &service()).unwrap_err(),
        AccountingError::InvalidSourceEnergy
    );

    records[0].source_energy_kwh = f64::INFINITY;
    assert_eq!(
        account_energy_ledger(&records, &service()).unwrap_err(),
        AccountingError::NonFiniteValue("source energy kWh")
    );
}

#[test]
fn delivered_energy_cannot_exceed_declared_source_energy() {
    let mut records = known_records();
    records[0].delivered_energy_kwh = 101.0;
    assert_eq!(
        account_energy_ledger(&records, &service()).unwrap_err(),
        AccountingError::DeliveredEnergyExceedsSource {
            record_id: "electricity".to_owned()
        }
    );
}

#[test]
fn negative_cost_or_direct_emissions_are_rejected() {
    let mut records = known_records();
    records[0].cost_amount = -1.0;
    assert_eq!(
        account_energy_ledger(&records, &service()).unwrap_err(),
        AccountingError::NegativeValue("cost amount")
    );

    let mut records = known_records();
    records[0].direct_operational_co2e_kg = -1.0;
    assert_eq!(
        account_energy_ledger(&records, &service()).unwrap_err(),
        AccountingError::NegativeValue("direct operational CO2e kg")
    );
}

#[test]
fn upstream_co2e_nan_is_rejected() {
    let mut records = known_records();
    records[0].upstream_co2e_kg = f64::NAN;
    assert_eq!(
        account_energy_ledger(&records, &service()).unwrap_err(),
        AccountingError::NonFiniteValue("upstream CO2e kg")
    );
}

#[test]
fn mixed_currency_is_rejected() {
    let mut records = known_records();
    records[1].currency_code = "EUR".to_owned();
    assert_eq!(
        account_energy_ledger(&records, &service()).unwrap_err(),
        AccountingError::CurrencyMismatch
    );
}

#[test]
fn malformed_currency_code_is_rejected() {
    let mut records = known_records();
    records[0].currency_code = "usd".to_owned();
    assert_eq!(
        account_energy_ledger(&records, &service()).unwrap_err(),
        AccountingError::InvalidCurrencyCode
    );
}

#[test]
fn period_mismatch_is_rejected() {
    let mut records = known_records();
    records[0].period_id = "2026-W02".to_owned();
    assert_eq!(
        account_energy_ledger(&records, &service()).unwrap_err(),
        AccountingError::PeriodMismatch
    );
}

#[test]
fn missing_evidence_reference_is_rejected() {
    let mut records = known_records();
    records[0].evidence_ref.clear();
    assert_eq!(
        account_energy_ledger(&records, &service()).unwrap_err(),
        AccountingError::MissingIdentifier("evidence reference")
    );
}

#[test]
fn zero_service_denominator_is_rejected() {
    let mut invalid_service = service();
    invalid_service.completed_passenger_km = 0.0;
    assert_eq!(
        account_energy_ledger(&known_records(), &invalid_service).unwrap_err(),
        AccountingError::InvalidServiceDenominator
    );
}

#[test]
fn positive_transport_service_without_delivered_energy_is_rejected() {
    let mut records = known_records();
    for entry in &mut records {
        entry.delivered_energy_kwh = 0.0;
    }
    assert_eq!(
        account_energy_ledger(&records, &service()).unwrap_err(),
        AccountingError::NoDeliveredEnergy
    );
}

#[test]
fn non_co2_quantification_requires_a_method_and_finite_value() {
    let mut records = known_records();
    records[1].non_co2 = NonCo2Evidence::Quantified {
        method_ref: String::new(),
        metric_name: "index".to_owned(),
        value: 1.0,
        unit: "index".to_owned(),
    };
    assert_eq!(
        account_energy_ledger(&records, &service()).unwrap_err(),
        AccountingError::MissingIdentifier("non-CO2 method reference")
    );

    records[1].non_co2 = NonCo2Evidence::Quantified {
        method_ref: "method-v1".to_owned(),
        metric_name: "index".to_owned(),
        value: f64::NAN,
        unit: "index".to_owned(),
    };
    assert_eq!(
        account_energy_ledger(&records, &service()).unwrap_err(),
        AccountingError::InvalidNonCo2Evidence
    );
}

#[test]
fn arithmetic_overflow_fails_closed() {
    let first = record(
        "large-1",
        "large-boundary-1",
        EnergyCarrier::Electricity,
        f64::MAX,
        1.0,
        0.0,
        0.0,
        1.0,
    );
    let second = record(
        "large-2",
        "large-boundary-2",
        EnergyCarrier::Electricity,
        f64::MAX,
        1.0,
        0.0,
        0.0,
        1.0,
    );
    assert_eq!(
        account_energy_ledger(&[first, second], &service()).unwrap_err(),
        AccountingError::ArithmeticOverflow
    );
}
