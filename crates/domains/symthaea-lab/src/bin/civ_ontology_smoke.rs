//! SYM-CIV-001: typed Civilization OS ontology smoke laboratory.
//!
//! This executable demonstrates the semantic boundary in a dependency-free
//! higher-layer lab. Numeric equality never grants semantic conversion.
//!
//! Run:
//!   cargo run -p symthaea-lab --bin civ_ontology_smoke

use std::marker::PhantomData;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Instrument<P> {
    units: i64,
    _profile: PhantomData<P>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Resource<K> {
    units: i64,
    _kind: PhantomData<K>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Authority<S> {
    _scope: PhantomData<S>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Qualification<S> {
    _scope: PhantomData<S>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Reservation<K> {
    units: i64,
    _kind: PhantomData<K>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Settlement<P> {
    units: i64,
    _profile: PhantomData<P>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ConversionReceipt<S, T> {
    source_units: i64,
    target_units: i64,
    _source: PhantomData<S>,
    _target: PhantomData<T>,
}

struct CreditismPc;
struct MarketMoney;
struct Steel;
struct Water;
struct ProjectScope;
struct UniversalPoliticalScope;
struct SettlementScope;

fn pc(units: i64) -> Instrument<CreditismPc> {
    assert!(units >= 0);
    Instrument {
        units,
        _profile: PhantomData,
    }
}

fn steel(units: i64) -> Resource<Steel> {
    assert!(units >= 0);
    Resource {
        units,
        _kind: PhantomData,
    }
}

fn reserve_steel(units: i64) -> Reservation<Steel> {
    assert!(units >= 0);
    Reservation {
        units,
        _kind: PhantomData,
    }
}

fn project_authority() -> Authority<ProjectScope> {
    Authority {
        _scope: PhantomData,
    }
}

fn project_qualification() -> Qualification<ProjectScope> {
    Qualification {
        _scope: PhantomData,
    }
}

fn authorize_project(
    _authority: Authority<ProjectScope>,
    _qualification: Qualification<ProjectScope>,
    reservation: Reservation<Steel>,
) -> Reservation<Steel> {
    // Authority does not become the resource; it only authorizes a bounded
    // operation over an already-typed reservation.
    reservation
}

fn convert_pc_to_market(
    source: Instrument<CreditismPc>,
) -> (ConversionReceipt<CreditismPc, MarketMoney>, Instrument<MarketMoney>) {
    let target = Instrument {
        units: source.units,
        _profile: PhantomData,
    };

    (
        ConversionReceipt {
            source_units: source.units,
            target_units: target.units,
            _source: PhantomData,
            _target: PhantomData,
        },
        target,
    )
}

fn settle_market(
    _receipt: &ConversionReceipt<CreditismPc, MarketMoney>,
    target: Instrument<MarketMoney>,
) -> Settlement<SettlementScope> {
    Settlement {
        units: target.units,
        _profile: PhantomData,
    }
}

fn run_typed_conversion_control() {
    let source = pc(100);
    let (receipt, target) = convert_pc_to_market(source);
    let settlement = settle_market(&receipt, target);

    assert_eq!(receipt.source_units, 100);
    assert_eq!(receipt.target_units, 100);
    assert_eq!(settlement.units, 100);

    println!("CONTROL:EXPLICIT_TYPED_INSTRUMENT_CONVERSION:PASS");
    println!("  source=CreditismPC:100");
    println!("  target=MarketMoney:100");
    println!("  settlement=100");
}

fn run_claim_resource_separation_control() {
    let claim = pc(100);
    let resource = steel(100);

    // Numeric equality is deliberately irrelevant.
    assert_eq!(claim.units, resource.units);

    let reserved = reserve_steel(40);
    let authorized = authorize_project(
        project_authority(),
        project_qualification(),
        reserved,
    );
    assert_eq!(authorized.units, 40);

    println!("CONTROL:NUMERIC_EQUALITY_WITH_SEMANTIC_NON_EQUIVALENCE:PASS");
    println!("  claim=CreditismPC:100");
    println!("  resource=Steel:100");
    println!("  reservation=Steel:40");
}

fn run_water_is_not_steel_control() {
    let water = Resource::<Water> {
        units: 100,
        _kind: PhantomData,
    };
    let steel = steel(100);

    assert_eq!(water.units, steel.units);
    assert_ne!(
        std::any::type_name::<Resource<Water>>(),
        std::any::type_name::<Resource<Steel>>()
    );

    println!("CONTROL:RESOURCE_TYPE_SEPARATION:PASS");
}

fn run_scope_separation_control() {
    let project = project_authority();

    // A project authority is deliberately not constructible as a universal
    // political authority without an explicit conversion function.
    assert_eq!(
        std::any::type_name::<Authority<ProjectScope>>(),
        "civ_ontology_smoke::Authority<civ_ontology_smoke::ProjectScope>"
    );

    println!("CONTROL:AUTHORITY_SCOPE_IS_EXPLICIT:PASS");
    println!("  scope=project_only");
    let _ = project;
}

fn main() {
    println!("SYM-CIV-TYPED-ONTOLOGY-SMOKE-V1");

    run_typed_conversion_control();
    run_claim_resource_separation_control();
    run_water_is_not_steel_control();
    run_scope_separation_control();

    println!("RESULT:PASS");
}
