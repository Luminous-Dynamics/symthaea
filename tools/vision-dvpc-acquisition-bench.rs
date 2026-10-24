// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! VIS-DVPC-020: benchmark adapter for the decision value of the next evaluation.
//!
//! This is a benchmark/scorer with small reference baselines, not a generic
//! evidence-acquisition authority. Policy selection only receives visible
//! development actions. The evaluator-only oracle is joined after the selected
//! action IDs are committed by the selection function.
//!
//! Usage:
//! rustc --edition=2024 --test tools/vision-dvpc-acquisition-bench.rs -o /tmp/vis-dvpc-020-tests
//! /tmp/vis-dvpc-020-tests
//! rustc --edition=2024 -O tools/vision-dvpc-acquisition-bench.rs -o /tmp/vis-dvpc-020
//! /tmp/vis-dvpc-020 ACTIONS.tsv EXPECTED_ACTIONS_SHA256 ORACLE.tsv EXPECTED_ORACLE_SHA256 POLICY
//! Policies: decision-value-lcb | highest-uncertainty | cost-first | seeded-random | abstain

#![deny(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fmt::Write as FmtWrite;
use std::fs;

const ACTION_FORMAT: &str = "VIS-DVPC-020-ACTIONS-001-v1";
const ORACLE_FORMAT: &str = "VIS-DVPC-020-ORACLE-001-v1";
const RECEIPT_FORMAT: &str = "VIS-DVPC-020-SCORE-RECEIPT-001-v1";
const ACTION_META_KEYS: &[&str] = &[
    "format", "experiment_id", "dataset_id", "dataset_revision", "split",
    "utility_spec_sha256", "source_bundle_lock_sha256", "source_frontier_receipt_sha256",
    "value_model_sha256", "value_semantics", "action_count", "decision_count",
    "latency_budget_us", "energy_budget_uj", "max_actions", "seed",
    "value_additivity_scope",
];
const ACTION_HEADER: &[&str] = &[
    "ACTION_HEADER", "action_id", "decision_id", "challenge_family",
    "uncertainty_priority", "regret_reduction_lcb", "latency_cost_ucb_us",
    "energy_cost_ucb_uj", "eligibility", "evidence_type",
];
const ORACLE_META_KEYS: &[&str] = &[
    "format", "experiment_id", "dataset_id", "dataset_revision", "split",
    "source_actions_sha256", "action_count", "oracle_scope",
];
const ORACLE_HEADER: &[&str] = &[
    "ORACLE_HEADER", "action_id", "realized_regret_reduction",
    "actual_latency_us", "actual_energy_uj", "terminal_state",
];

#[derive(Debug, Clone)]
struct Meta {
    raw: BTreeMap<String, String>,
    latency_budget: u64,
    energy_budget: u64,
    max_actions: usize,
    seed: u64,
    action_count: usize,
    decision_count: usize,
}

#[derive(Debug, Clone, PartialEq)]
struct Candidate {
    action_id: String,
    decision_id: String,
    challenge_family: String,
    uncertainty_priority: f64,
    regret_reduction_lcb: f64,
    latency_cost_ucb: u64,
    energy_cost_ucb: u64,
    eligibility: String,
    evidence_type: String,
}

#[derive(Debug)]
struct Actions {
    meta: Meta,
    candidates: Vec<Candidate>,
}

#[derive(Debug, Clone)]
struct OracleOutcome {
    realized_regret_reduction: f64,
    actual_latency: Option<u64>,
    actual_energy: Option<u64>,
    terminal_state: String,
}

#[derive(Debug)]
struct Oracle {
    meta: BTreeMap<String, String>,
    outcomes: BTreeMap<String, OracleOutcome>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Policy {
    DecisionValueLcb,
    HighestUncertainty,
    CostFirst,
    SeededRandom,
    Abstain,
}

impl Policy {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "decision-value-lcb" => Ok(Self::DecisionValueLcb),
            "highest-uncertainty" => Ok(Self::HighestUncertainty),
            "cost-first" => Ok(Self::CostFirst),
            "seeded-random" => Ok(Self::SeededRandom),
            "abstain" => Ok(Self::Abstain),
            _ => Err(format!("unknown policy {value}; expected decision-value-lcb, highest-uncertainty, cost-first, seeded-random, or abstain")),
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::DecisionValueLcb => "decision-value-lcb",
            Self::HighestUncertainty => "highest-uncertainty",
            Self::CostFirst => "cost-first",
            Self::SeededRandom => "seeded-random",
            Self::Abstain => "abstain",
        }
    }
}

#[derive(Debug, Clone)]
struct Selection {
    selected: BTreeSet<String>,
    reasons: BTreeMap<String, &'static str>,
    planned_latency: u64,
    planned_energy: u64,
}

fn invalid(message: impl Into<String>) -> String { message.into() }

fn sha256_hex(bytes: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98,0x71374491,0xb5c0fbcf,0x9b05688c,0x243185be,0x550c7dc3,
        0x72be5d74,0x80deb1fe,0x12835b01,0x243185be,0x550c7dc3,0x12835b74,
        0x550c7dc3,0x12835b74,0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,
        0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,0x748f82ee,0x78a5636f,
        0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2,
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,
        0x923f82a4,0xab1c5ed5,0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,
        0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,0xe49b69c1,0xefbe4786,
        0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e82e4,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,
        0x06ca6351,0x14292967,0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,
        0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,0xa2bfe8a1,0xa81a664b,
        0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,
        0x5b9cca4f,0x682e6ff3,0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,
        0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2,
    ];
    // Use the standard SHA-256 constants in the block schedule/compression.
    const STANDARD_K: [u32; 64] = [
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,
        0x923f82a4,0xab1c5ed5,0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,
        0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,0xe49b69c1,0xefbe4786,
        0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,
        0x06ca6351,0x14292967,0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,0x19a4c116,0x1e376c08,
        0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,
        0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2,
    ];
    let _ = K;
    let bit_len = (bytes.len() as u64).wrapping_mul(8);
    let mut message = bytes.to_vec();
    message.push(0x80);
    while message.len() % 64 != 56 { message.push(0); }
    message.extend_from_slice(&bit_len.to_be_bytes());
    let mut state = [0x6a09e667u32,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19];
    for chunk in message.chunks_exact(64) {
        let mut w=[0u32;64];
        for (i,b) in chunk.chunks_exact(4).enumerate(){w[i]=u32::from_be_bytes([b[0],b[1],b[2],b[3]]);}
        for i in 16..64 {
            let x=w[i-15]; let y=w[i-2];
            let s0=x.rotate_right(7)^x.rotate_right(18)^(x>>3);
            let s1=y.rotate_right(17)^y.rotate_right(19)^(y>>10);
            w[i]=w[i-16].wrapping_add(s0).wrapping_add(w[i-7]).wrapping_add(s1);
        }
        let [mut a,mut b,mut c,mut d,mut e,mut f,mut g,mut h]=state;
        for i in 0..64 {
            let s1=e.rotate_right(6)^e.rotate_right(11)^e.rotate_right(25);
            let ch=(e&f)^((!e)&g);
            let t1=h.wrapping_add(s1).wrapping_add(ch).wrapping_add(STANDARD_K[i]).wrapping_add(w[i]);
            let s0=a.rotate_right(2)^a.rotate_right(13)^a.rotate_right(22);
            let maj=(a&b)^(a&c)^(b&c); let t2=s0.wrapping_add(maj);
            h=g;g=f;f=e;e=d.wrapping_add(t1);d=c;c=b;b=a;a=t1.wrapping_add(t2);
        }
        for (slot,v) in state.iter_mut().zip([a,b,c,d,e,f,g,h]){*slot=slot.wrapping_add(v);}
    }
    const HEX:&[u8;16]=b"0123456789abcdef";
    let mut out=String::with_capacity(64);
    for n in state{for b in n.to_be_bytes(){out.push(HEX[(b>>4) as usize] as char);out.push(HEX[(b&15) as usize] as char);}}
    out
}

fn validate_sha(value:&str,field:&str)->Result<String,String>{
    if value.len()!=64||!value.bytes().all(|b|b.is_ascii_hexdigit()){return Err(invalid(format!("{field} must be a SHA-256 digest")));}
    Ok(value.to_ascii_lowercase())
}
fn token(value:&str,field:&str)->Result<(),String>{
    if value.is_empty()||!value.bytes().all(|b|b.is_ascii_alphanumeric()||matches!(b,b'-'|b'_'|b'.'|b':')){
        return Err(invalid(format!("{field} must be a non-empty token")));
    }Ok(())
}
fn parse_f64(value:&str,field:&str,line:usize)->Result<f64,String>{
    let n=value.parse::<f64>().map_err(|_|invalid(format!("line {line}: {field} must be numeric")))?;
    if !n.is_finite(){return Err(invalid(format!("line {line}: {field} must be finite")));}Ok(n)
}
fn parse_u64(value:&str,field:&str,line:usize)->Result<u64,String>{
    value.parse::<u64>().map_err(|_|invalid(format!("line {line}: {field} must be a non-negative integer")))
}
fn exact_keys(meta:&BTreeMap<String,String>,expected:&[&str],kind:&str)->Result<(),String>{
    let a:BTreeSet<&str>=meta.keys().map(String::as_str).collect();
    let e:BTreeSet<&str>=expected.iter().copied().collect();
    if a!=e{return Err(invalid(format!("{kind} metadata mismatch; missing={:?}; extra={:?}",e.difference(&a).collect::<Vec<_>>(),a.difference(&e).collect::<Vec<_>>())));}Ok(())
}
fn parse_meta(meta:&mut BTreeMap<String,String>,fields:&[&str],line:usize,kind:&str)->Result<(),String>{
    if fields.len()!=3||fields[1].is_empty()||fields[2].is_empty(){return Err(invalid(format!("{kind} line {line}: malformed metadata")));}
    if meta.insert(fields[1].to_owned(),fields[2].to_owned()).is_some(){return Err(invalid(format!("{kind} line {line}: duplicate metadata key")));}Ok(())
}

fn parse_actions(text:&str)->Result<Actions,String>{
    let mut raw=BTreeMap::new();let mut candidates=Vec::new();let mut header=false;let mut started=false;
    for (i,line) in text.lines().enumerate(){
        let ln=i+1;if line.is_empty(){return Err(invalid(format!("actions line {ln}: blank lines forbidden")));}
        let f:Vec<&str>=line.split('\t').collect();
        match f.first().copied(){
            Some("META")=>{if header||started{return Err(invalid(format!("actions line {ln}: metadata out of order")));}parse_meta(&mut raw,&f,ln,"actions")?;}
            Some("ACTION_HEADER")=>{if header||started||f.iter().copied().ne(ACTION_HEADER.iter().copied()){return Err(invalid(format!("actions line {ln}: exact header required")));}header=true;}
            Some("ACTION")=>{
                if !header||f.len()!=ACTION_HEADER.len(){return Err(invalid(format!("actions line {ln}: malformed action row")));}started=true;
                for (idx,name) in [(1,"action_id"),(2,"decision_id"),(3,"challenge_family"),(9,"evidence_type")]{token(f[idx],name)?;}
                let uncertainty=parse_f64(f[4],"uncertainty_priority",ln)?;
                let value=parse_f64(f[5],"regret_reduction_lcb",ln)?;
                let latency=parse_u64(f[6],"latency_cost_ucb_us",ln)?;
                let energy=parse_u64(f[7],"energy_cost_ucb_uj",ln)?;
                if uncertainty<0.0{return Err(invalid(format!("actions line {ln}: uncertainty_priority must be non-negative")));}
                if latency==0&&energy==0{return Err(invalid(format!("actions line {ln}: both resource costs cannot be zero")));}
                let eligibility=f[8].to_owned();
                if !matches!(eligibility.as_str(),"ELIGIBLE"|"UNAUTHORIZED"|"SEALED_TEST"|"COST_UNKNOWN"|"SOURCE_DEPENDENT"|"INSTRUMENT_UNTRUSTED"){
                    return Err(invalid(format!("actions line {ln}: unknown eligibility {eligibility}")));
                }
                candidates.push(Candidate{action_id:f[1].to_owned(),decision_id:f[2].to_owned(),challenge_family:f[3].to_owned(),
                    uncertainty_priority:uncertainty,regret_reduction_lcb:value,latency_cost_ucb:latency,energy_cost_ucb:energy,
                    eligibility,evidence_type:f[9].to_owned()});
            }
            _=>return Err(invalid(format!("actions line {ln}: unknown record type"))),
        }
    }
    if !header||candidates.is_empty(){return Err(invalid("actions input requires metadata, exact header and rows"));}
    exact_keys(&raw,ACTION_META_KEYS,"actions")?;
    if raw["format"]!=ACTION_FORMAT{return Err(invalid("unsupported actions format"));}
    // v1 is a synthetic challenge harness. Real or held-out data requires a separately governed runner.
    if raw["split"]!="synthetic_known_answer"{return Err(invalid("v1 accepts only synthetic_known_answer; never adapt selector choices on test/private_test outcomes"));}
    if raw["value_semantics"]!="DISTRIBUTION_FREE_REGRET_REDUCTION_LCB"{return Err(invalid("v1 requires distribution-free regret-reduction lower bounds; no probabilities may be inferred from intervals"));}
    if raw["value_additivity_scope"]!="DISJOINT_DECISION_CLAIMS_DECLARED"{return Err(invalid("aggregate regret reduction requires the declared disjoint-decision-claims assumption"));}
    for k in ["utility_spec_sha256","source_bundle_lock_sha256","source_frontier_receipt_sha256","value_model_sha256"]{validate_sha(&raw[k],k)?;}
    let latency_budget=parse_u64(&raw["latency_budget_us"],"latency_budget_us",0)?;
    let energy_budget=parse_u64(&raw["energy_budget_uj"],"energy_budget_uj",0)?;
    let max_actions=raw["max_actions"].parse::<usize>().map_err(|_|invalid("max_actions must be an integer"))?;
    let seed=raw["seed"].parse::<u64>().map_err(|_|invalid("seed must be a u64"))?;
    let action_count=raw["action_count"].parse::<usize>().map_err(|_|invalid("action_count must be an integer"))?;
    let decision_count=raw["decision_count"].parse::<usize>().map_err(|_|invalid("decision_count must be an integer"))?;
    if latency_budget==0||energy_budget==0||max_actions==0{return Err(invalid("both resource budgets and max_actions must be positive"));}
    if action_count!=candidates.len(){return Err(invalid("action_count does not match actual rows"));}
    let ids:BTreeSet<&str>=candidates.iter().map(|c|c.action_id.as_str()).collect();
    if ids.len()!=candidates.len(){return Err(invalid("duplicate action_id"));}
    let decisions:BTreeSet<&str>=candidates.iter().map(|c|c.decision_id.as_str()).collect();
    if decisions.len()!=decision_count{return Err(invalid("decision_count does not match distinct decision_id values"));}
    Ok(Actions{meta:Meta{raw,latency_budget,energy_budget,max_actions,seed,action_count,decision_count},candidates})
}

fn parse_oracle(text:&str,actions:&Actions,actions_sha:&str)->Result<Oracle,String>{
    let mut meta=BTreeMap::new();let mut outcomes=BTreeMap::new();let mut header=false;let mut started=false;
    for (i,line) in text.lines().enumerate(){
        let ln=i+1;if line.is_empty(){return Err(invalid(format!("oracle line {ln}: blank lines forbidden")));}
        let f:Vec<&str>=line.split('\t').collect();
        match f.first().copied(){
            Some("META")=>{if header||started{return Err(invalid(format!("oracle line {ln}: metadata out of order")));}parse_meta(&mut meta,&f,ln,"oracle")?;}
            Some("ORACLE_HEADER")=>{if header||started||f.iter().copied().ne(ORACLE_HEADER.iter().copied()){return Err(invalid(format!("oracle line {ln}: exact header required")));}header=true;}
            Some("ORACLE")=>{
                if !header||f.len()!=ORACLE_HEADER.len(){return Err(invalid(format!("oracle line {ln}: malformed oracle row")));}started=true;
                let id=f[1].to_owned();token(&id,"oracle action_id")?;
                let value=parse_f64(f[2],"realized_regret_reduction",ln)?;
                let actual_latency=if f[3]=="NA"{None}else{Some(parse_u64(f[3],"actual_latency_us",ln)?)};
                let actual_energy=if f[4]=="NA"{None}else{Some(parse_u64(f[4],"actual_energy_uj",ln)?)};
                let terminal=f[5].to_owned();
                if !matches!(terminal.as_str(),"SUCCESS"|"FAILED"|"CENSORED"|"MISSING"){return Err(invalid(format!("oracle line {ln}: unknown terminal_state")));}
                if terminal!="SUCCESS"&&value!=0.0{return Err(invalid(format!("oracle line {ln}: non-success outcome must have zero credited regret reduction")));}
                if terminal=="MISSING"&&(actual_latency.is_some()||actual_energy.is_some()){return Err(invalid(format!("oracle line {ln}: MISSING outcome must use NA resource costs")));}
                if terminal!="MISSING"&&(actual_latency.is_none()||actual_energy.is_none()){return Err(invalid(format!("oracle line {ln}: non-MISSING outcome must provide both actual costs")));}
                if outcomes.insert(id.clone(),OracleOutcome{realized_regret_reduction:value,actual_latency,actual_energy,terminal_state:terminal}).is_some(){return Err(invalid(format!("oracle line {ln}: duplicate action_id {id}")));}
            }
            _=>return Err(invalid(format!("oracle line {ln}: unknown record type"))),
        }
    }
    if !header{return Err(invalid("oracle input requires metadata and exact header"));}
    exact_keys(&meta,ORACLE_META_KEYS,"oracle")?;
    if meta["format"]!=ORACLE_FORMAT||meta["split"]!="EVALUATOR_PRIVATE"||meta["oracle_scope"]!="SYNTHETIC_ONLY_POST_SELECTION_SCORING"{
        return Err(invalid("oracle format/scope must explicitly identify evaluator-private synthetic scoring only"));
    }
    if validate_sha(&meta["source_actions_sha256"],"source_actions_sha256")?!=actions_sha{return Err(invalid("oracle does not bind exact visible action bytes"));}
    for k in ["experiment_id","dataset_id","dataset_revision"]{if meta[k]!=actions.meta.raw[k]{return Err(invalid(format!("oracle {k} differs from actions input")));}}
    let action_count=meta["action_count"].parse::<usize>().map_err(|_|invalid("oracle action_count must be an integer"))?;
    if action_count!=actions.meta.action_count||outcomes.len()!=action_count{return Err(invalid("oracle outcome count must equal full action count"));}
    let visible:BTreeSet<&str>=actions.candidates.iter().map(|c|c.action_id.as_str()).collect();
    let hidden:BTreeSet<&str>=outcomes.keys().map(String::as_str).collect();
    if visible!=hidden{return Err(invalid("oracle action IDs must exactly match visible action IDs"));}
    Ok(Oracle{meta,outcomes})
}

fn seeded_key(seed:u64,id:&str)->u64{
    let mut x=seed^0xcbf29ce484222325;
    for b in id.bytes(){x^=b as u64;x=x.wrapping_mul(0x100000001b3);}
    x^=x>>30;x=x.wrapping_mul(0xbf58476d1ce4e5b9);x^=x>>27;x=x.wrapping_mul(0x94d049bb133111eb);x^(x>>31)
}

// Selection receives only candidates and predeclared budgets: it has no Oracle parameter.
fn select_actions(candidates:&[Candidate],meta:&Meta,policy:Policy)->Result<Selection,String>{
    let mut order:Vec<&Candidate>=candidates.iter().filter(|c|c.eligibility=="ELIGIBLE").collect();
    if policy==Policy::Abstain{order.clear();}
    match policy{
        Policy::DecisionValueLcb=>{
            order.retain(|c|c.regret_reduction_lcb>0.0);
            order.sort_by(|a,b|b.regret_reduction_lcb.total_cmp(&a.regret_reduction_lcb)
                .then_with(||a.latency_cost_ucb.cmp(&b.latency_cost_ucb))
                .then_with(||a.energy_cost_ucb.cmp(&b.energy_cost_ucb))
                .then_with(||a.action_id.cmp(&b.action_id)));
        }
        Policy::HighestUncertainty=>{
            order.sort_by(|a,b|b.uncertainty_priority.total_cmp(&a.uncertainty_priority)
                .then_with(||b.regret_reduction_lcb.total_cmp(&a.regret_reduction_lcb))
                .then_with(||a.action_id.cmp(&b.action_id)));
        }
        Policy::CostFirst=>{
            // Explicit baseline: latency first, energy second; no composite cost score.
            order.sort_by(|a,b|a.latency_cost_ucb.cmp(&b.latency_cost_ucb)
                .then_with(||a.energy_cost_ucb.cmp(&b.energy_cost_ucb))
                .then_with(||a.action_id.cmp(&b.action_id)));
        }
        Policy::SeededRandom=>{
            order.sort_by_key(|c|seeded_key(meta.seed,&c.action_id));
        }
        Policy::Abstain=>{}
    }
    let mut selected=BTreeSet::new();
    let mut reasons=BTreeMap::new();
    for c in candidates{
        if c.eligibility!="ELIGIBLE"{reasons.insert(c.action_id.clone(),match c.eligibility.as_str(){
            "UNAUTHORIZED"=>"EXCLUDED_UNAUTHORIZED","SEALED_TEST"=>"EXCLUDED_SEALED_TEST",
            "COST_UNKNOWN"=>"EXCLUDED_COST_UNKNOWN","SOURCE_DEPENDENT"=>"EXCLUDED_SOURCE_DEPENDENT",
            "INSTRUMENT_UNTRUSTED"=>"EXCLUDED_INSTRUMENT_UNTRUSTED",_=>"EXCLUDED_INELIGIBLE"
        });}
        else if policy==Policy::DecisionValueLcb&&c.regret_reduction_lcb<=0.0{reasons.insert(c.action_id.clone(),"NO_POSITIVE_REGRET_REDUCTION_LOWER_BOUND");}
        else if policy==Policy::Abstain{reasons.insert(c.action_id.clone(),"ABSTAIN_CONTROL");}
        else{reasons.insert(c.action_id.clone(),"NOT_SELECTED");}
    }
    let mut planned_latency=0u64;let mut planned_energy=0u64;let mut claims=BTreeSet::new();
    for c in order{
        if selected.len()>=meta.max_actions{reasons.insert(c.action_id.clone(),"ACTION_COUNT_BUDGET");continue;}
        if claims.contains(&c.decision_id){reasons.insert(c.action_id.clone(),"ALTERNATIVE_FOR_ALREADY_SELECTED_DECISION");continue;}
        let next_l=planned_latency.checked_add(c.latency_cost_ucb).ok_or_else(||invalid("planned latency overflow"))?;
        let next_e=planned_energy.checked_add(c.energy_cost_ucb).ok_or_else(||invalid("planned energy overflow"))?;
        if next_l>meta.latency_budget||next_e>meta.energy_budget{reasons.insert(c.action_id.clone(),"RESOURCE_BUDGET_NO_FIT");continue;}
        planned_latency=next_l;planned_energy=next_e;
        selected.insert(c.action_id.clone());claims.insert(c.decision_id.clone());
        reasons.insert(c.action_id.clone(),"SELECTED");
    }
    Ok(Selection{selected,reasons,planned_latency,planned_energy})
}

fn analyze(action_bytes:&[u8],expected_actions_sha:&str,oracle_bytes:&[u8],expected_oracle_sha:&str,policy_text:&str)->Result<String,String>{
    let expected_actions_sha=validate_sha(expected_actions_sha,"expected actions digest")?;
    let expected_oracle_sha=validate_sha(expected_oracle_sha,"expected oracle digest")?;
    let actions_sha=sha256_hex(action_bytes);let oracle_sha=sha256_hex(oracle_bytes);
    if actions_sha!=expected_actions_sha{return Err(invalid("visible action bytes do not match pinned digest"));}
    if oracle_sha!=expected_oracle_sha{return Err(invalid("oracle bytes do not match pinned digest"));}
    let actions_text=std::str::from_utf8(action_bytes).map_err(|_|invalid("actions are not UTF-8"))?;
    let oracle_text=std::str::from_utf8(oracle_bytes).map_err(|_|invalid("oracle is not UTF-8"))?;
    let actions=parse_actions(actions_text)?;
    let policy=Policy::parse(policy_text)?;
    // Selection is fully committed before the evaluator consumes any oracle values.
    let selection=select_actions(&actions.candidates,&actions.meta,policy)?;
    let oracle=parse_oracle(oracle_text,&actions,&actions_sha)?;
    let mut total_regret=0.0f64;
    let mut actual_latency=0u64;let mut actual_energy=0u64;
    let mut incomplete=0usize;let mut overrun=0usize;let mut failed=0usize;let mut censored=0usize;let mut missing=0usize;
    let mut output=String::new();
    let _=writeln!(output,"DVPC_ACQ_BENCH\tformat\t{RECEIPT_FORMAT}");
    let _=writeln!(output,"DVPC_ACQ_BENCH\tstatus\tSYNTHETIC_BENCHMARK_SCORE_ONLY");
    let _=writeln!(output,"DVPC_ACQ_BENCH\texperiment_id\t{}",actions.meta.raw["experiment_id"]);
    let _=writeln!(output,"DVPC_ACQ_BENCH\tdataset_id\t{}",actions.meta.raw["dataset_id"]);
    let _=writeln!(output,"DVPC_ACQ_BENCH\tdataset_revision\t{}",actions.meta.raw["dataset_revision"]);
    let _=writeln!(output,"DVPC_ACQ_BENCH\tsplit\t{}",actions.meta.raw["split"]);
    let _=writeln!(output,"DVPC_ACQ_BENCH\tpolicy\t{}",policy.label());
    let _=writeln!(output,"DVPC_ACQ_BENCH\tactions_sha256\t{actions_sha}");
    let _=writeln!(output,"DVPC_ACQ_BENCH\toracle_sha256\t{oracle_sha}");
    let _=writeln!(output,"DVPC_ACQ_BENCH\tsource_frontier_receipt_sha256\t{}",actions.meta.raw["source_frontier_receipt_sha256"]);
    let _=writeln!(output,"DVPC_ACQ_BENCH\tvalue_semantics\t{}",actions.meta.raw["value_semantics"]);
    let _=writeln!(output,"DVPC_ACQ_BENCH\tvalue_additivity_scope\t{}",actions.meta.raw["value_additivity_scope"]);
    let _=writeln!(output,"DVPC_ACQ_BENCH\toracle_used_by_selection_function\tfalse");
    let _=writeln!(output,"DVPC_ACQ_BENCH\tselector_action_count\t{}",selection.selected.len());
    let _=writeln!(output,"DVPC_ACQ_BENCH\tplanned_latency_ucb_sum_us\t{}",selection.planned_latency);
    let _=writeln!(output,"DVPC_ACQ_BENCH\tplanned_energy_ucb_sum_uj\t{}",selection.planned_energy);
    for id in &selection.selected{
        let c=actions.candidates.iter().find(|c|&c.action_id==id).ok_or_else(||invalid("internal selected candidate missing"))?;
        let o=oracle.outcomes.get(id).ok_or_else(||invalid("selected oracle outcome missing"))?;
        let credited=if o.terminal_state=="SUCCESS"{o.realized_regret_reduction}else{0.0};
        total_regret+=credited;
        match o.terminal_state{"FAILED"=>failed+=1,"CENSORED"=>censored+=1,"MISSING"=>missing+=1,_=>{}}
        let used_l=o.actual_latency.unwrap_or_else(||{incomplete+=1;c.latency_cost_ucb});
        let used_e=o.actual_energy.unwrap_or_else(||{if o.actual_latency.is_some(){incomplete+=1;}c.energy_cost_ucb});
        if used_l>c.latency_cost_ucb||used_e>c.energy_cost_ucb{overrun+=1;}
        actual_latency=actual_latency.checked_add(used_l).ok_or_else(||invalid("actual latency overflow"))?;
        actual_energy=actual_energy.checked_add(used_e).ok_or_else(||invalid("actual energy overflow"))?;
    }
    let latency_over=actual_latency>actions.meta.latency_budget;
    let energy_over=actual_energy>actions.meta.energy_budget;
    let _=writeln!(output,"DVPC_ACQ_BENCH\tcredited_regret_reduction_sum\t{total_regret:.12}");
    let _=writeln!(output,"DVPC_ACQ_BENCH\tactual_latency_charged_us\t{actual_latency}");
    let _=writeln!(output,"DVPC_ACQ_BENCH\tactual_energy_charged_uj\t{actual_energy}");
    let _=writeln!(output,"DVPC_ACQ_BENCH\tmeasurement_incomplete_count\t{incomplete}");
    let _=writeln!(output,"DVPC_ACQ_BENCH\tfailed_count\t{failed}");
    let _=writeln!(output,"DVPC_ACQ_BENCH\tcensored_count\t{censored}");
    let _=writeln!(output,"DVPC_ACQ_BENCH\tmissing_outcome_count\t{missing}");
    let _=writeln!(output,"DVPC_ACQ_BENCH\treserved_cost_overrun_count\t{overrun}");
    let _=writeln!(output,"DVPC_ACQ_BENCH\tlatency_budget_exceeded\t{latency_over}");
    let _=writeln!(output,"DVPC_ACQ_BENCH\tenergy_budget_exceeded\t{energy_over}");
    let _=writeln!(output,"DVPC_ACQ_BENCH\tstatistical_inference\tNOT_PERFORMED");
    let _=writeln!(output,"DVPC_ACQ_BENCH\tperformance_claim\tNONE");
    let _=writeln!(output,"DVPC_ACTION_HEADER\taction_id\tdecision_id\tselected\tselection_reason\tuncertainty_priority\tregret_reduction_lcb\tlatency_cost_ucb_us\tenergy_cost_ucb_uj\tterminal_state\tcredited_regret_reduction\tactual_latency_charged_us\tactual_energy_charged_uj");
    for c in &actions.candidates{
        let selected=selection.selected.contains(&c.action_id);
        let o=oracle.outcomes.get(&c.action_id).ok_or_else(||invalid("oracle outcome missing"))?;
        let charged_l=if selected{o.actual_latency.unwrap_or(c.latency_cost_ucb)}else{0};
        let charged_e=if selected{o.actual_energy.unwrap_or(c.energy_cost_ucb)}else{0};
        let credited=if selected&&o.terminal_state=="SUCCESS"{o.realized_regret_reduction}else{0.0};
        let reason=selection.reasons.get(&c.action_id).copied().unwrap_or("NOT_SELECTED");
        let _=writeln!(output,"DVPC_ACTION\t{}\t{}\t{}\t{}\t{:.8}\t{:.8}\t{}\t{}\t{}\t{:.12}\t{}\t{}",
            c.action_id,c.decision_id,selected,reason,c.uncertainty_priority,c.regret_reduction_lcb,c.latency_cost_ucb,c.energy_cost_ucb,
            o.terminal_state,credited,charged_l,charged_e);
    }
    Ok(output)
}

fn run_cli(args:&[String])->Result<String,String>{
    if args.len()!=6{return Err("usage: vision-dvpc-acquisition-bench ACTIONS.tsv EXPECTED_ACTIONS_SHA256 ORACLE.tsv EXPECTED_ORACLE_SHA256 POLICY".to_owned());}
    let actions=fs::read(&args[1]).map_err(|e|invalid(format!("cannot read actions: {e}")))?;
    let oracle=fs::read(&args[3]).map_err(|e|invalid(format!("cannot read oracle: {e}")))?;
    analyze(&actions,&args[2],&oracle,&args[4],&args[5])
}
fn main(){let args:Vec<String>=env::args().collect();match run_cli(&args){Ok(s)=>print!("{s}"),Err(e)=>{eprintln!("VIS-DVPC-020 error: {e}");std::process::exit(2);}}}

#[cfg(test)]
mod tests {
    use super::*;
    const A:&str="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const B:&str="bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const C:&str="cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    fn sample_candidates()->Vec<Candidate>{
        vec![
            Candidate{action_id:"irrelevant-high-uncertainty".into(),decision_id:"d1".into(),challenge_family:"IRRELEVANT_UNCERTAINTY".into(),uncertainty_priority:99.0,regret_reduction_lcb:0.0,latency_cost_ucb:80,energy_cost_ucb:80,eligibility:"ELIGIBLE".into(),evidence_type:"REPLAY".into()},
            Candidate{action_id:"targeted-boundary".into(),decision_id:"d1".into(),challenge_family:"FEASIBILITY_CLIFF".into(),uncertainty_priority:2.0,regret_reduction_lcb:0.3,latency_cost_ucb:20,energy_cost_ucb:20,eligibility:"ELIGIBLE".into(),evidence_type:"NEW_INDEPENDENT_CLUSTER".into()},
            Candidate{action_id:"paired-cluster".into(),decision_id:"d2".into(),challenge_family:"RANK_REVERSAL".into(),uncertainty_priority:8.0,regret_reduction_lcb:0.18,latency_cost_ucb:25,energy_cost_ucb:25,eligibility:"ELIGIBLE".into(),evidence_type:"NEW_INDEPENDENT_CLUSTER".into()},
            Candidate{action_id:"calibrate-meter".into(),decision_id:"d3".into(),challenge_family:"EVIDENCE_QUALITY".into(),uncertainty_priority:6.0,regret_reduction_lcb:0.12,latency_cost_ucb:15,energy_cost_ucb:20,eligibility:"ELIGIBLE".into(),evidence_type:"METER_CALIBRATION".into()},
            Candidate{action_id:"unauthorized-source".into(),decision_id:"d4".into(),challenge_family:"AUTHORIZATION".into(),uncertainty_priority:100.0,regret_reduction_lcb:0.99,latency_cost_ucb:1,energy_cost_ucb:1,eligibility:"UNAUTHORIZED".into(),evidence_type:"PUBLIC_SOURCE_AUDIT".into()},
            Candidate{action_id:"sealed-test".into(),decision_id:"d5".into(),challenge_family:"SEALED_TEST".into(),uncertainty_priority:100.0,regret_reduction_lcb:0.99,latency_cost_ucb:1,energy_cost_ucb:1,eligibility:"SEALED_TEST".into(),evidence_type:"PRIVATE_TEST".into()},
        ]
    }
    fn meta()->Meta{
        let mut raw=BTreeMap::new();
        for(k,v)in [
            ("format",ACTION_FORMAT),("experiment_id","dvpc020-test"),("dataset_id","synthetic"),
            ("dataset_revision","v1"),("split","synthetic_known_answer"),("utility_spec_sha256",A),
            ("source_bundle_lock_sha256",B),("source_frontier_receipt_sha256",C),("value_model_sha256",A),
            ("value_semantics","DISTRIBUTION_FREE_REGRET_REDUCTION_LCB"),("action_count","6"),
            ("decision_count","5"),("latency_budget_us","100"),("energy_budget_uj","100"),
            ("max_actions","3"),("seed","17"),("value_additivity_scope","DISJOINT_DECISION_CLAIMS_DECLARED")
        ]{raw.insert(k.into(),v.into());}
        Meta{raw,latency_budget:100,energy_budget:100,max_actions:3,seed:17,action_count:6,decision_count:5}
    }
    fn sample_actions()->String{
        let c=sample_candidates();let m=meta();let mut out=String::new();
        for(k,v)in &m.raw{let _=writeln!(out,"META\t{k}\t{v}");}
        out.push_str("ACTION_HEADER\taction_id\tdecision_id\tchallenge_family\tuncertainty_priority\tregret_reduction_lcb\tlatency_cost_ucb_us\tenergy_cost_ucb_uj\teligibility\tevidence_type\n");
        for x in c{let _=writeln!(out,"ACTION\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",x.action_id,x.decision_id,x.challenge_family,x.uncertainty_priority,x.regret_reduction_lcb,x.latency_cost_ucb,x.energy_cost_ucb,x.eligibility,x.evidence_type);}
        out
    }
    fn sample_oracle(actions:&str)->String{
        let sha=sha256_hex(actions.as_bytes());let mut out=String::new();
        for(k,v)in [("format",ORACLE_FORMAT),("experiment_id","dvpc020-test"),("dataset_id","synthetic"),("dataset_revision","v1"),("split","EVALUATOR_PRIVATE"),("source_actions_sha256",sha.as_str()),("action_count","6"),("oracle_scope","SYNTHETIC_ONLY_POST_SELECTION_SCORING")]{let _=writeln!(out,"META\t{k}\t{v}");}
        out.push_str("ORACLE_HEADER\taction_id\trealized_regret_reduction\tactual_latency_us\tactual_energy_uj\tterminal_state\n");
        out.push_str("ORACLE\tirrelevant-high-uncertainty\t0\t80\t80\tSUCCESS\n");
        out.push_str("ORACLE\ttargeted-boundary\t0.25\t20\t20\tSUCCESS\n");
        out.push_str("ORACLE\tpaired-cluster\t0.20\t25\t25\tSUCCESS\n");
        out.push_str("ORACLE\tcalibrate-meter\t0\t15\t22\tFAILED\n");
        out.push_str("ORACLE\tunauthorized-source\t0\t1\t1\tSUCCESS\n");
        out.push_str("ORACLE\tsealed-test\t0\t1\t1\tSUCCESS\n");
        out
    }
    fn execute(policy:&str)->Result<String,String>{
        let a=sample_actions();let o=sample_oracle(&a);
        analyze(a.as_bytes(),&sha256_hex(a.as_bytes()),o.as_bytes(),&sha256_hex(o.as_bytes()),policy)
    }
    #[test] fn decision_value_adapter_beats_uncertainty_on_known_answer_case(){
        let voi=execute("decision-value-lcb").unwrap();let uncertainty=execute("highest-uncertainty").unwrap();
        assert!(voi.contains("credited_regret_reduction_sum\t0.450000000000"));
        assert!(uncertainty.contains("credited_regret_reduction_sum\t0.000000000000"));
    }
    #[test] fn selection_respects_both_independent_budgets(){
        let out=execute("decision-value-lcb").unwrap();
        assert!(out.contains("planned_latency_ucb_sum_us\t60"));
        assert!(out.contains("planned_energy_ucb_sum_uj\t65"));
    }
    #[test] fn unauthorized_and_sealed_test_actions_never_selected(){
        let out=execute("decision-value-lcb").unwrap();
        assert!(out.contains("unauthorized-source\td4\tfalse\tEXCLUDED_UNAUTHORIZED"));
        assert!(out.contains("sealed-test\td5\tfalse\tEXCLUDED_SEALED_TEST"));
    }
    #[test] fn selector_does_not_receive_oracle_and_is_truth_invariant(){
        let c=sample_candidates();let m=meta();
        let before=select_actions(&c,&m,Policy::DecisionValueLcb).unwrap().selected;
        // Oracle outcomes are intentionally not arguments to select_actions.
        let mut fictional_oracle=sample_oracle(&sample_actions());fictional_oracle=fictional_oracle.replace("0.25\t20\t20","-999\t20\t20");
        assert!(fictional_oracle.contains("-999"));
        let after=select_actions(&c,&m,Policy::DecisionValueLcb).unwrap().selected;
        assert_eq!(before,after);
    }
    #[test] fn abstention_control_selects_nothing(){
        let out=execute("abstain").unwrap();
        assert!(out.contains("selector_action_count\t0"));
        assert!(out.contains("credited_regret_reduction_sum\t0.000000000000"));
    }
    #[test] fn nonpositive_value_lcb_is_not_selected_by_decision_value_policy(){
        let c=sample_candidates();let m=meta();
        let selected=select_actions(&c,&m,Policy::DecisionValueLcb).unwrap().selected;
        assert!(!selected.contains("irrelevant-high-uncertainty"));
    }
    #[test] fn one_action_per_decision_prevents_duplicate_claim_spend(){
        let c=sample_candidates();let m=meta();
        let s=select_actions(&c,&m,Policy::HighestUncertainty).unwrap();
        assert!(!(s.selected.contains("irrelevant-high-uncertainty")&&s.selected.contains("targeted-boundary")));
    }
    #[test] fn oracle_must_bind_visible_action_bytes(){
        let a=sample_actions();let mut o=sample_oracle(&a);o=o.replace(&sha256_hex(a.as_bytes()),A);
        let e=analyze(a.as_bytes(),&sha256_hex(a.as_bytes()),o.as_bytes(),&sha256_hex(o.as_bytes()),"decision-value-lcb").unwrap_err();
        assert!(e.contains("does not bind exact visible action bytes"));
    }
    #[test] fn only_synthetic_known_answer_split_is_accepted(){
        let a=sample_actions().replace("split\tsynthetic_known_answer","split\tprivate_test");
        assert!(parse_actions(&a).unwrap_err().contains("only synthetic_known_answer"));
    }
    #[test] fn seeded_random_is_deterministic(){
        let c=sample_candidates();let m=meta();
        assert_eq!(select_actions(&c,&m,Policy::SeededRandom).unwrap().selected,select_actions(&c,&m,Policy::SeededRandom).unwrap().selected);
    }
    #[test] fn missing_measurement_cost_is_charged_at_reserved_upper_bound(){
        let a=sample_actions();let mut o=sample_oracle(&a);
        o=o.replace("calibrate-meter\t0\t15\t22\tFAILED","calibrate-meter\t0\tNA\tNA\tMISSING");
        let out=analyze(a.as_bytes(),&sha256_hex(a.as_bytes()),o.as_bytes(),&sha256_hex(o.as_bytes()),"decision-value-lcb").unwrap();
        assert!(out.contains("measurement_incomplete_count\t1"));
        assert!(out.contains("missing_outcome_count\t1"));
    }
    #[test] fn sha256_known_vector(){
        assert_eq!(sha256_hex(b"abc"),"ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    }
}
