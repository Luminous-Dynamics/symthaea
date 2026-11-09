//! Deterministic long-only cash-ledger oracle for synthetic quant conformance.
//! This is not a strategy engine, broker connector, execution model or proof of alpha.
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

const BPS: i128 = 10_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side { Buy, Sell }

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Fill {
    pub fill_id: String,
    pub instrument_id: String,
    pub currency: String,
    pub side: Side,
    pub quantity: u64,
    pub price_minor_units: i128,
    pub fee_bps: u32,
    pub sequence: u64,
    pub event_time_ms: i64,
    pub known_at_ms: i64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mark {
    pub mark_id: String,
    pub instrument_id: String,
    pub currency: String,
    pub price_minor_units: i128,
    pub event_time_ms: i64,
    pub known_at_ms: i64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Config {
    pub base_currency: String,
    pub starting_cash_minor_units: i128,
    pub decision_frontier_ms: i64,
    pub valuation_at_ms: i64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Position { pub quantity: u64 }

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AppliedFill {
    pub fill_id: String,
    pub side: Side,
    pub quantity: u64,
    pub notional_minor_units: i128,
    pub fee_minor_units: i128,
    pub cash_after_minor_units: i128,
    pub position_after: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MarkedPosition {
    pub instrument_id: String,
    pub mark_id: String,
    pub quantity: u64,
    pub price_minor_units: i128,
    pub market_value_minor_units: i128,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ledger {
    pub base_currency: String,
    pub starting_cash_minor_units: i128,
    pub cash_minor_units: i128,
    pub gross_buys_minor_units: i128,
    pub gross_sells_minor_units: i128,
    pub fees_minor_units: i128,
    pub market_value_minor_units: i128,
    pub equity_minor_units: i128,
    pub positions: BTreeMap<String, Position>,
    pub applied_fills: Vec<AppliedFill>,
    pub marked_positions: Vec<MarkedPosition>,
    pub excluded_future_marks: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    InvalidConfig(&'static str),
    EmptyId(&'static str),
    InvalidCurrency(String),
    CurrencyMismatch { expected: String, actual: String },
    DuplicateFill(String),
    DuplicateFillSequence(i64, u64),
    DuplicateMark(String),
    InvalidQuantity(String),
    InvalidPrice(String),
    InvalidFee(String),
    FillKnownBeforeEvent(String),
    FutureFillEvent(String),
    FutureKnownFill(String),
    InsufficientCash { fill_id: String, required: i128, available: i128 },
    Oversell { fill_id: String, requested: u64, available: u64 },
    MissingMark(String),
    AmbiguousMark(String, i64),
    Overflow(&'static str),
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "{self:?}") }
}
impl std::error::Error for Error {}

fn add(a: i128, b: i128, what: &'static str) -> Result<i128, Error> {
    a.checked_add(b).ok_or(Error::Overflow(what))
}
fn sub(a: i128, b: i128, what: &'static str) -> Result<i128, Error> {
    a.checked_sub(b).ok_or(Error::Overflow(what))
}
fn mul(a: i128, b: i128, what: &'static str) -> Result<i128, Error> {
    a.checked_mul(b).ok_or(Error::Overflow(what))
}
fn currency(value: &str) -> Result<(), Error> {
    if value.len() != 3 || !value.bytes().all(|b| b.is_ascii_uppercase()) {
        return Err(Error::InvalidCurrency(value.to_owned()));
    }
    Ok(())
}
fn fee(notional: i128, bps: u32) -> Result<i128, Error> {
    let n=mul(notional, i128::from(bps), "fee numerator")?;
    Ok(add(n, BPS-1, "fee rounding")? / BPS)
}

/// Replays fills in (event_time, sequence) order; marks are selected as of the frontier.
/// Amounts are integer minor units, whole-share positions, no shorts, no margin, no FX.
pub fn evaluate(config: &Config, fills: &[Fill], marks: &[Mark]) -> Result<Ledger, Error> {
    currency(&config.base_currency)?;
    if config.starting_cash_minor_units < 0 { return Err(Error::InvalidConfig("negative starting cash")); }
    if config.valuation_at_ms > config.decision_frontier_ms {
        return Err(Error::InvalidConfig("valuation exceeds decision frontier"));
    }

    let mut ids=BTreeSet::new();
    let mut sequences=BTreeSet::new();
    let mut ordered:Vec<&Fill>=fills.iter().collect();
    for x in fills {
        if x.fill_id.trim().is_empty() { return Err(Error::EmptyId("fill_id")); }
        if x.instrument_id.trim().is_empty() { return Err(Error::EmptyId("instrument_id")); }
        if !ids.insert(x.fill_id.as_str()) { return Err(Error::DuplicateFill(x.fill_id.clone())); }
        if !sequences.insert((x.event_time_ms,x.sequence)) {
            return Err(Error::DuplicateFillSequence(x.event_time_ms,x.sequence));
        }
        currency(&x.currency)?;
        if x.currency != config.base_currency {
            return Err(Error::CurrencyMismatch { expected:config.base_currency.clone(), actual:x.currency.clone() });
        }
        if x.quantity==0 { return Err(Error::InvalidQuantity(x.fill_id.clone())); }
        if x.price_minor_units<=0 { return Err(Error::InvalidPrice(x.fill_id.clone())); }
        if x.fee_bps>10_000 { return Err(Error::InvalidFee(x.fill_id.clone())); }
        if x.known_at_ms<x.event_time_ms { return Err(Error::FillKnownBeforeEvent(x.fill_id.clone())); }
        if x.event_time_ms>config.valuation_at_ms { return Err(Error::FutureFillEvent(x.fill_id.clone())); }
        if x.known_at_ms>config.decision_frontier_ms { return Err(Error::FutureKnownFill(x.fill_id.clone())); }
    }
    ordered.sort_by_key(|x|(x.event_time_ms,x.sequence));

    let mut cash=config.starting_cash_minor_units;
    let mut gross_buys=0i128;
    let mut gross_sells=0i128;
    let mut total_fees=0i128;
    let mut positions:BTreeMap<String,Position>=BTreeMap::new();
    let mut applied=Vec::with_capacity(fills.len());

    for x in ordered {
        let notional=mul(x.price_minor_units,i128::from(x.quantity),"fill notional")?;
        let cost=fee(notional,x.fee_bps)?;
        total_fees=add(total_fees,cost,"total fees")?;
        let p=positions.entry(x.instrument_id.clone()).or_default();
        match x.side {
            Side::Buy=>{
                let needed=add(notional,cost,"buy total")?;
                if needed>cash {
                    return Err(Error::InsufficientCash{fill_id:x.fill_id.clone(),required:needed,available:cash});
                }
                cash=sub(cash,needed,"cash after buy")?;
                p.quantity=p.quantity.checked_add(x.quantity).ok_or(Error::Overflow("position quantity"))?;
                gross_buys=add(gross_buys,notional,"gross buys")?;
            }
            Side::Sell=>{
                if x.quantity>p.quantity {
                    return Err(Error::Oversell{fill_id:x.fill_id.clone(),requested:x.quantity,available:p.quantity});
                }
                cash=add(cash,sub(notional,cost,"sell proceeds")?,"cash after sell")?;
                p.quantity-=x.quantity;
                gross_sells=add(gross_sells,notional,"gross sells")?;
            }
        }
        applied.push(AppliedFill{
            fill_id:x.fill_id.clone(),side:x.side,quantity:x.quantity,
            notional_minor_units:notional,fee_minor_units:cost,
            cash_after_minor_units:cash,position_after:p.quantity
        });
    }
    positions.retain(|_,p|p.quantity>0);

    let mut mark_ids=BTreeSet::new();
    for m in marks {
        if m.mark_id.trim().is_empty(){return Err(Error::EmptyId("mark_id"));}
        if !mark_ids.insert(m.mark_id.as_str()){return Err(Error::DuplicateMark(m.mark_id.clone()));}
    }

    let mut excluded=0usize;
    let mut market_value=0i128;
    let mut marked=Vec::with_capacity(positions.len());
    for (instrument,p) in &positions {
        let mut eligible:Vec<&Mark>=Vec::new();
        for m in marks.iter().filter(|m|m.instrument_id==*instrument) {
            if m.event_time_ms>config.valuation_at_ms || m.known_at_ms>config.decision_frontier_ms {
                excluded+=1;
                continue;
            }
            if m.known_at_ms<m.event_time_ms {return Err(Error::FillKnownBeforeEvent(m.mark_id.clone()));}
            currency(&m.currency)?;
            if m.currency!=config.base_currency {
                return Err(Error::CurrencyMismatch{expected:config.base_currency.clone(),actual:m.currency.clone()});
            }
            if m.price_minor_units<=0 {return Err(Error::InvalidPrice(m.mark_id.clone()));}
            eligible.push(m);
        }
        eligible.sort_by(|left, right| {
            left.event_time_ms
                .cmp(&right.event_time_ms)
                .then_with(|| left.known_at_ms.cmp(&right.known_at_ms))
                .then_with(|| left.mark_id.cmp(&right.mark_id))
        });
        let selected=*eligible.last().ok_or_else(||Error::MissingMark(instrument.clone()))?;
        if eligible.iter().filter(|m|m.event_time_ms==selected.event_time_ms).count()>1 {
            return Err(Error::AmbiguousMark(instrument.clone(),selected.event_time_ms));
        }
        let value=mul(i128::from(p.quantity),selected.price_minor_units,"marked position")?;
        market_value=add(market_value,value,"market value")?;
        marked.push(MarkedPosition{
            instrument_id:instrument.clone(),mark_id:selected.mark_id.clone(),quantity:p.quantity,
            price_minor_units:selected.price_minor_units,market_value_minor_units:value
        });
    }
    let equity=add(cash,market_value,"portfolio equity")?;
    Ok(Ledger{
        base_currency:config.base_currency.clone(),starting_cash_minor_units:config.starting_cash_minor_units,
        cash_minor_units:cash,gross_buys_minor_units:gross_buys,gross_sells_minor_units:gross_sells,
        fees_minor_units:total_fees,market_value_minor_units:market_value,equity_minor_units:equity,
        positions,applied_fills:applied,marked_positions:marked,excluded_future_marks:excluded
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn cfg()->Config {Config{base_currency:"USD".into(),starting_cash_minor_units:1_000_000,decision_frontier_ms:300,valuation_at_ms:300}}
    fn fill(id:&str,inst:&str,side:Side,q:u64,px:i128,seq:u64,event:i64,known:i64)->Fill{
        Fill{fill_id:id.into(),instrument_id:inst.into(),currency:"USD".into(),side,quantity:q,price_minor_units:px,fee_bps:10,sequence:seq,event_time_ms:event,known_at_ms:known}
    }
    fn mark(id:&str,inst:&str,px:i128,event:i64,known:i64)->Mark{
        Mark{mark_id:id.into(),instrument_id:inst.into(),currency:"USD".into(),price_minor_units:px,event_time_ms:event,known_at_ms:known}
    }

    #[test] fn exact_two_asset_reconciliation(){
        let f=vec![fill("a","A",Side::Buy,49,10_000,1,100,101),fill("b","B",Side::Buy,100,5_000,2,100,101)];
        let m=vec![mark("ma","A",10_500,200,201),mark("mb","B",4_900,200,201)];
        let r=evaluate(&cfg(),&f,&m).unwrap();
        assert_eq!((r.gross_buys_minor_units,r.fees_minor_units,r.cash_minor_units),(990_000,990,9_010));
        assert_eq!((r.market_value_minor_units,r.equity_minor_units),(1_004_500,1_013_510));
    }

    #[test] fn sell_reconciles_cash_position_and_equity(){
        let f=vec![fill("buy","A",Side::Buy,49,10_000,1,100,101),fill("sell","A",Side::Sell,10,10_500,2,200,201)];
        let r=evaluate(&cfg(),&f,&[mark("m","A",10_500,250,251)]).unwrap();
        assert_eq!((r.gross_sells_minor_units,r.fees_minor_units,r.cash_minor_units),(105_000,595,614_405));
        assert_eq!(r.positions["A"].quantity,39);
        assert_eq!((r.market_value_minor_units,r.equity_minor_units),(409_500,1_023_905));
    }

    #[test] fn late_known_fill_fails(){let f=vec![fill("late","A",Side::Buy,1,100,1,100,301)];assert!(matches!(evaluate(&cfg(),&f,&[]),Err(Error::FutureKnownFill(_))));}
    #[test] fn future_fill_fails(){let f=vec![fill("future","A",Side::Buy,1,100,1,301,301)];assert!(matches!(evaluate(&cfg(),&f,&[]),Err(Error::FutureFillEvent(_))));}
    #[test] fn future_only_mark_cannot_value_position(){let f=vec![fill("buy","A",Side::Buy,1,100,1,100,101)];let m=vec![mark("future","A",500,301,301)];assert!(matches!(evaluate(&cfg(),&f,&m),Err(Error::MissingMark(_))));}
    #[test] fn frontier_is_inclusive(){let f=vec![fill("buy","A",Side::Buy,1,100,1,300,300)];let m=vec![mark("m","A",100,300,300)];assert!(evaluate(&cfg(),&f,&m).is_ok());}
    #[test] fn oversell_fails(){let f=vec![fill("buy","A",Side::Buy,2,100,1,100,101),fill("sell","A",Side::Sell,3,100,2,200,201)];assert!(matches!(evaluate(&cfg(),&f,&[mark("m","A",100,250,251)]),Err(Error::Oversell{..})));}
    #[test] fn cash_shortfall_fails(){let f=vec![fill("big","A",Side::Buy,101,10_000,1,100,101)];assert!(matches!(evaluate(&cfg(),&f,&[]),Err(Error::InsufficientCash{..})));}
    #[test] fn duplicate_fill_id_fails(){let f=vec![fill("x","A",Side::Buy,1,100,1,100,101),fill("x","A",Side::Buy,1,100,2,200,201)];assert!(matches!(evaluate(&cfg(),&f,&[]),Err(Error::DuplicateFill(_))));}
    #[test] fn duplicate_sequence_fails(){let f=vec![fill("x","A",Side::Buy,1,100,1,100,101),fill("y","A",Side::Buy,1,100,1,100,101)];assert!(matches!(evaluate(&cfg(),&f,&[]),Err(Error::DuplicateFillSequence(..))));}
    #[test] fn ambiguous_same_time_marks_fail(){let f=vec![fill("buy","A",Side::Buy,1,100,1,100,101)];let m=vec![mark("m1","A",100,200,201),mark("m2","A",101,200,202)];assert!(matches!(evaluate(&cfg(),&f,&m),Err(Error::AmbiguousMark(..))));}
    #[test] fn newest_eligible_mark_used_future_mark_excluded(){let f=vec![fill("buy","A",Side::Buy,1,100,1,100,101)];let m=vec![mark("old","A",110,200,201),mark("new","A",120,250,251),mark("future","A",900,301,301)];let r=evaluate(&cfg(),&f,&m).unwrap();assert_eq!(r.marked_positions[0].mark_id,"new");assert_eq!(r.excluded_future_marks,1);}
    #[test] fn currency_mismatch_fails(){let mut f=fill("buy","A",Side::Buy,1,100,1,100,101);f.currency="EUR".into();assert!(matches!(evaluate(&cfg(),&[f],&[]),Err(Error::CurrencyMismatch{..})));}
    #[test] fn fee_rounds_up(){let mut f=fill("buy","A",Side::Buy,1,1,1,100,101);f.fee_bps=1;let r=evaluate(&cfg(),&[f],&[mark("m","A",1,200,201)]).unwrap();assert_eq!(r.fees_minor_units,1);assert_eq!(r.cash_minor_units,999_998);}
    #[test] fn arbitrary_input_order_replays_deterministically(){let a=fill("a","A",Side::Buy,2,100,1,100,101);let b=fill("b","A",Side::Sell,1,120,2,200,201);let m=vec![mark("m","A",120,250,251)];assert_eq!(evaluate(&cfg(),&[a.clone(),b.clone()],&m).unwrap(),evaluate(&cfg(),&[b,a],&m).unwrap());}
    #[test] fn zero_quantity_and_nonpositive_price_fail(){assert!(matches!(evaluate(&cfg(),&[fill("q","A",Side::Buy,0,100,1,100,101)],&[]),Err(Error::InvalidQuantity(_))));assert!(matches!(evaluate(&cfg(),&[fill("p","A",Side::Buy,1,0,1,100,101)],&[]),Err(Error::InvalidPrice(_))));}
    #[test] fn cash_only_portfolio_needs_no_marks(){let r=evaluate(&cfg(),&[],&[]).unwrap();assert_eq!(r.equity_minor_units,1_000_000);assert!(r.positions.is_empty());}
}