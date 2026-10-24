// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! VIS-DVPC-019: robust feasibility and Pareto analysis under joint uncertainty.
//!
//! Inputs are an exact VIS-DVPC-018 receipt and a frozen operational-envelope
//! table. All interval validity, calibration, risk certificates and meters remain
//! upstream assertions. This tool binds the bytes, checks contracts, and applies
//! conservative feasibility/dominance rules; it does not prove coverage.
//!
//! Usage:
//! rustc --edition=2024 --test tools/vision-dvpc-robust-envelope.rs -o /tmp/vis-dvpc-019-tests
//! /tmp/vis-dvpc-019-tests
//! rustc --edition=2024 -O tools/vision-dvpc-robust-envelope.rs -o /tmp/vis-dvpc-019
//! /tmp/vis-dvpc-019 CONFIDENCE.tsv EXPECTED_CONFIDENCE_SHA256 ENVELOPE.tsv EXPECTED_ENVELOPE_SHA256 EXPECTED_INFERENCE_RECEIPT_SHA256

#![deny(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fmt::Write as FmtWrite;
use std::fs;

const CONF_FORMAT: &str = "VIS-DVPC-CONFIDENCE-FRONTIER-RECEIPT-001-v1";
const CONF_STATUS: &str = "UNCERTAINTY_AWARE_RECLASSIFICATION_ONLY";
const OUTPUT_FORMAT: &str = "VIS-DVPC-ROBUST-ENVELOPE-RECEIPT-001-v1";
const CONF_META_KEYS: &[&str] = &[
    "format", "status", "experiment_id", "dataset_id", "dataset_revision", "split",
    "bundle_lock_sha256", "paired_unit_set_sha256", "sweep_input_sha256",
    "pareto_receipt_sha256", "confidence_bands_sha256", "inference_receipt_sha256",
    "confidence_level", "familywise_simultaneous", "interval_procedure_verified_here",
    "upstream_safety_certificate_verified_here", "band_nondominated_count",
    "band_dominated_count", "unresolved_count", "duplicate_equivalent_count",
    "infeasible_count", "performance_claim",
];
const CONF_HEADER: &[&str] = &[
    "CONFIDENCE_RESULT_HEADER", "budget_point_id", "policy_id", "status",
    "utility_estimate", "utility_lcb", "utility_ucb", "actual_all_in_latency_us",
    "actual_all_in_energy_uj", "latency_budget_us", "energy_budget_uj",
    "upstream_point_status", "upstream_reasons", "reason",
    "upstream_point_dominated_by", "band_dominated_by",
];
const ENV_META_KEYS: &[&str] = &[
    "format", "experiment_id", "dataset_id", "dataset_revision", "split",
    "bundle_lock_sha256", "paired_unit_set_sha256", "source_sweep_sha256",
    "source_confidence_receipt_sha256", "inference_receipt_sha256",
    "confidence_level", "familywise_simultaneous", "coverage_scope", "point_count",
    "pooled_harm_limit", "worst_group_harm_limit", "cvar95_loss_limit",
    "safety_violations_max",
];
const ENV_HEADER: &[&str] = &[
    "ENVELOPE_HEADER", "budget_point_id", "policy_id",
    "latency_lcb_us", "latency_ucb_us", "energy_lcb_uj", "energy_ucb_uj",
    "pooled_harm_lcb", "pooled_harm_ucb", "worst_group_harm_lcb",
    "worst_group_harm_ucb", "cvar95_loss_lcb", "cvar95_loss_ucb",
    "safety_violations_lower", "safety_violations_upper",
    "cost_measurement_complete", "risk_certificate_state",
];

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Key { budget: String, policy: String }

impl Key {
    fn label(&self) -> String { format!("{}/{}", self.budget, self.policy) }
}

#[derive(Debug, Clone)]
struct Point {
    key: Key,
    status: String,
    upstream_status: String,
    utility_estimate: f64,
    utility_lcb: f64,
    utility_ucb: f64,
    latency_budget: u64,
    energy_budget: u64,
    upstream_reasons: String,
    upstream_reason: String,
    upstream_dominator: String,
}

#[derive(Debug, Clone, Copy)]
struct Interval { lo: f64, hi: f64 }

#[derive(Debug, Clone)]
struct Envelope {
    latency: Interval,
    energy: Interval,
    pooled_harm: Interval,
    worst_group_harm: Interval,
    cvar95: Interval,
    safety_lower: u64,
    safety_upper: u64,
    cost_complete: bool,
    risk_state: String,
}

#[derive(Debug)]
struct ConfidenceReceipt { meta: BTreeMap<String, String>, points: Vec<Point> }

#[derive(Debug)]
struct EnvelopeTable {
    meta: BTreeMap<String, String>,
    rows: BTreeMap<Key, Envelope>,
}

#[derive(Debug, Clone)]
struct Classified {
    point: Point,
    envelope: Envelope,
    status: &'static str,
    reason: String,
    dominator: String,
}

fn invalid(message: impl Into<String>) -> String { message.into() }

fn sha256_hex(bytes: &[u8]) -> String {
    const K: [u32;64] = [
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,
        0x923f82a4,0xab1c5ed5,0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,
        0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,0xe49b69c1,0xefbe4786,
        0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,
        0x06ca6351,0x14292967,0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,
        0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,0xa2bfe8a1,0xa81a664b,
        0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,
        0x5b9cca4f,0x682e6ff3,0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,
        0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2,
    ];
    let bit_len = (bytes.len() as u64).wrapping_mul(8);
    let mut message = bytes.to_vec();
    message.push(0x80);
    while message.len() % 64 != 56 { message.push(0); }
    message.extend_from_slice(&bit_len.to_be_bytes());
    let mut h = [0x6a09e667u32,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19];
    for chunk in message.chunks_exact(64) {
        let mut w=[0u32;64];
        for (i,b) in chunk.chunks_exact(4).enumerate() { w[i]=u32::from_be_bytes([b[0],b[1],b[2],b[3]]); }
        for i in 16..64 {
            let x=w[i-15]; let y=w[i-2];
            let s0=x.rotate_right(7)^x.rotate_right(18)^(x>>3);
            let s1=y.rotate_right(17)^y.rotate_right(19)^(y>>10);
            w[i]=w[i-16].wrapping_add(s0).wrapping_add(w[i-7]).wrapping_add(s1);
        }
        let [mut a,mut b,mut c,mut d,mut e,mut f,mut g,mut hh]=h;
        for i in 0..64 {
            let s1=e.rotate_right(6)^e.rotate_right(11)^e.rotate_right(25);
            let ch=(e&f)^((!e)&g);
            let t1=hh.wrapping_add(s1).wrapping_add(ch).wrapping_add(K[i]).wrapping_add(w[i]);
            let s0=a.rotate_right(2)^a.rotate_right(13)^a.rotate_right(22);
            let maj=(a&b)^(a&c)^(b&c);
            let t2=s0.wrapping_add(maj);
            hh=g; g=f; f=e; e=d.wrapping_add(t1); d=c; c=b; b=a; a=t1.wrapping_add(t2);
        }
        for (slot,value) in h.iter_mut().zip([a,b,c,d,e,f,g,hh]) { *slot=slot.wrapping_add(value); }
    }
    const HEX:&[u8;16]=b"0123456789abcdef";
    let mut out=String::with_capacity(64);
    for n in h { for b in n.to_be_bytes() { out.push(HEX[(b>>4) as usize] as char); out.push(HEX[(b&15) as usize] as char); } }
    out
}

fn validate_sha(v:&str,field:&str)->Result<String,String>{
    if v.len()!=64 || !v.bytes().all(|b|b.is_ascii_hexdigit()){return Err(invalid(format!("{field} must be a 64-character SHA-256 digest")));}
    Ok(v.to_ascii_lowercase())
}
fn token(v:&str,field:&str)->Result<(),String>{
    if v.is_empty() || !v.bytes().all(|b|b.is_ascii_alphanumeric()||matches!(b,b'-'|b'_'|b'.'|b':')){
        return Err(invalid(format!("{field} must be a non-empty token")));
    } Ok(())
}
fn f64_field(v:&str,field:&str,line:usize)->Result<f64,String>{
    let n=v.parse::<f64>().map_err(|_|invalid(format!("line {line}: {field} must be numeric")))?;
    if !n.is_finite(){return Err(invalid(format!("line {line}: {field} must be finite")));} Ok(n)
}
fn u64_field(v:&str,field:&str,line:usize)->Result<u64,String>{
    v.parse::<u64>().map_err(|_|invalid(format!("line {line}: {field} must be a non-negative integer")))
}
fn bool_field(v:&str,field:&str,line:usize)->Result<bool,String>{
    match v {"true"=>Ok(true),"false"=>Ok(false),_=>Err(invalid(format!("line {line}: {field} must be true or false")))}
}
fn exact_keys(meta:&BTreeMap<String,String>,expected:&[&str],kind:&str)->Result<(),String>{
    let a:BTreeSet<&str>=meta.keys().map(String::as_str).collect();
    let e:BTreeSet<&str>=expected.iter().copied().collect();
    if a!=e {return Err(invalid(format!("{kind} metadata keys mismatch: missing={:?}, extra={:?}",e.difference(&a).collect::<Vec<_>>(),a.difference(&e).collect::<Vec<_>>())));} Ok(())
}
fn put_meta(meta:&mut BTreeMap<String,String>,fields:&[&str],line:usize,kind:&str)->Result<(),String>{
    if fields.len()!=3 || fields[1].is_empty() || fields[2].is_empty(){return Err(invalid(format!("{kind} line {line}: malformed metadata")));}
    if meta.insert(fields[1].to_owned(),fields[2].to_owned()).is_some(){return Err(invalid(format!("{kind} line {line}: duplicate metadata key")));} Ok(())
}

fn parse_confidence(text:&str)->Result<ConfidenceReceipt,String>{
    let mut meta=BTreeMap::new(); let mut points=Vec::new(); let mut header=false; let mut rows=false;
    for (i,line) in text.lines().enumerate(){
        let ln=i+1; if line.is_empty(){return Err(invalid(format!("confidence line {ln}: blank lines forbidden")));}
        let f:Vec<&str>=line.split('\t').collect();
        match f.first().copied(){
            Some("CONFIDENCE_FRONTIER")=>{
                if header||rows{return Err(invalid(format!("confidence line {ln}: metadata out of order")));} put_meta(&mut meta,&f,ln,"confidence")?;
            }
            Some("CONFIDENCE_RESULT_HEADER")=>{
                if header||rows||f.iter().copied().ne(CONF_HEADER.iter().copied()){return Err(invalid(format!("confidence line {ln}: exact result header required")));} header=true;
            }
            Some("CONFIDENCE_RESULT")=>{
                if !header||f.len()!=CONF_HEADER.len(){return Err(invalid(format!("confidence line {ln}: malformed result row")));} rows=true;
                let budget=f[1].to_owned(); let policy=f[2].to_owned(); token(&budget,"budget_point_id")?; token(&policy,"policy_id")?;
                let status=f[3].to_owned();
                if !matches!(status.as_str(),"BAND_DOMINATED"|"BAND_NONDOMINATED"|"UNRESOLVED"|"DUPLICATE_EQUIVALENT"|"INFEASIBLE"){
                    return Err(invalid(format!("confidence line {ln}: unknown status {status}")));
                }
                let upstream_status=f[11].to_owned();
                if !matches!(upstream_status.as_str(),"FEASIBLE_FRONTIER"|"FEASIBLE_DOMINATED"|"INFEASIBLE"){
                    return Err(invalid(format!("confidence line {ln}: unknown upstream eligibility {upstream_status}")));
                }
                let estimate=f64_field(f[4],"utility_estimate",ln)?;
                let lo=f64_field(f[5],"utility_lcb",ln)?; let hi=f64_field(f[6],"utility_ucb",ln)?;
                if lo>hi{return Err(invalid(format!("confidence line {ln}: utility interval inverted")));}
                let latency_budget=u64_field(f[9],"latency_budget_us",ln)?;
                let energy_budget=u64_field(f[10],"energy_budget_uj",ln)?;
                let key=Key{budget,policy};
                points.push(Point{key,status,upstream_status,utility_estimate:estimate,utility_lcb:lo,utility_ucb:hi,
                    latency_budget,energy_budget,upstream_reasons:f[12].to_owned(),upstream_reason:f[13].to_owned(),upstream_dominator:f[14].to_owned()});
            }
            _=>return Err(invalid(format!("confidence line {ln}: unknown record type"))),
        }
    }
    if !header||points.is_empty(){return Err(invalid("confidence receipt requires metadata, header and rows"));}
    exact_keys(&meta,CONF_META_KEYS,"confidence")?;
    if meta["format"]!=CONF_FORMAT||meta["status"]!=CONF_STATUS||meta["performance_claim"]!="NONE"||
        meta["interval_procedure_verified_here"]!="false"||meta["upstream_safety_certificate_verified_here"]!="false"||
        meta["familywise_simultaneous"]!="true" {return Err(invalid("confidence receipt format or non-claim contract mismatch"));}
    if !matches!(meta["split"].as_str(),"test"|"private_test"){return Err(invalid("confidence split must be test or private_test"));}
    let utility_conf=f64_field(&meta["confidence_level"],"confidence_level",0)?;
    if !(0.0..1.0).contains(&utility_conf){return Err(invalid("utility confidence_level must be in (0,1)"));}
    for k in ["bundle_lock_sha256","paired_unit_set_sha256","sweep_input_sha256","pareto_receipt_sha256","confidence_bands_sha256","inference_receipt_sha256"]{validate_sha(&meta[k],k)?;}
    let mut seen=BTreeSet::new();
    for p in &points{if !seen.insert(p.key.clone()){return Err(invalid(format!("duplicate confidence point {}",p.key.label())));}}
    let policies:BTreeSet<&str>=points.iter().map(|p|p.key.policy.as_str()).collect();
    let budgets:BTreeSet<&str>=points.iter().map(|p|p.key.budget.as_str()).collect();
    for b in &budgets{for p in &policies{if !seen.contains(&Key{budget:b.to_string(),policy:p.to_string()}){return Err(invalid(format!("confidence grid missing {b}/{p}")));}}}
    Ok(ConfidenceReceipt{meta,points})
}

fn parse_envelopes(text:&str,conf:&ConfidenceReceipt,conf_sha:&str,expected_inference:&str)->Result<EnvelopeTable,String>{
    let mut meta=BTreeMap::new();let mut rows=BTreeMap::new();let mut header=false;let mut started=false;
    for (i,line) in text.lines().enumerate(){
        let ln=i+1;if line.is_empty(){return Err(invalid(format!("envelope line {ln}: blank lines forbidden")));}
        let f:Vec<&str>=line.split('\t').collect();
        match f.first().copied(){
            Some("META")=>{if header||started{return Err(invalid(format!("envelope line {ln}: metadata out of order")));}put_meta(&mut meta,&f,ln,"envelope")?;}
            Some("ENVELOPE_HEADER")=>{if header||started||f.iter().copied().ne(ENV_HEADER.iter().copied()){return Err(invalid(format!("envelope line {ln}: exact header required")));}header=true;}
            Some("ENVELOPE")=>{
                if !header||f.len()!=ENV_HEADER.len(){return Err(invalid(format!("envelope line {ln}: malformed row")));}started=true;
                let budget=f[1].to_owned();let policy=f[2].to_owned();token(&budget,"budget_point_id")?;token(&policy,"policy_id")?;
                let interval=|a:usize,b:usize,name:&str|->Result<Interval,String>{
                    let lo=f64_field(f[a],&format!("{name}_lcb"),ln)?;let hi=f64_field(f[b],&format!("{name}_ucb"),ln)?;
                    if lo>hi{return Err(invalid(format!("envelope line {ln}: {name} interval inverted")));}Ok(Interval{lo,hi})
                };
                let latency=interval(3,4,"latency")?;let energy=interval(5,6,"energy")?;
                let pooled=interval(7,8,"pooled_harm")?;let worst=interval(9,10,"worst_group_harm")?;
                let cvar=interval(11,12,"cvar95_loss")?;
                for (name,v) in [("pooled_harm",pooled),("worst_group_harm",worst)]{if v.lo<0.0||v.hi>1.0{return Err(invalid(format!("envelope line {ln}: {name} must lie in [0,1]")));}}
                let safety_lower=u64_field(f[13],"safety_violations_lower",ln)?;let safety_upper=u64_field(f[14],"safety_violations_upper",ln)?;
                if safety_lower>safety_upper{return Err(invalid(format!("envelope line {ln}: safety count interval inverted")));}
                let cost_complete=bool_field(f[15],"cost_measurement_complete",ln)?;
                let risk_state=f[16].to_owned();if !matches!(risk_state.as_str(),"CERTIFIED"|"FAILED"|"UNKNOWN"){return Err(invalid(format!("envelope line {ln}: risk state must be CERTIFIED, FAILED or UNKNOWN")));}
                let key=Key{budget,policy};
                if rows.insert(key.clone(),Envelope{latency,energy,pooled_harm:pooled,worst_group_harm:worst,cvar95:cvar,safety_lower,safety_upper,cost_complete,risk_state}).is_some(){
                    return Err(invalid(format!("envelope line {ln}: duplicate {}",key.label())));
                }
            }
            _=>return Err(invalid(format!("envelope line {ln}: unknown record type"))),
        }
    }
    if !header{return Err(invalid("envelope file requires metadata and header"));}
    exact_keys(&meta,ENV_META_KEYS,"envelope")?;
    if meta["format"]!="VIS-DVPC-OP-ENVELOPE-001-v1"||meta["familywise_simultaneous"]!="true"||
       meta["coverage_scope"]!="ALL_POLICY_BUDGET_SAFETY_METRICS"{return Err(invalid("envelope must declare family-wise simultaneous coverage over all policy-budget safety metrics"));}
    let c:f64=f64_field(&meta["confidence_level"],"envelope confidence_level",0)?;
    if !(0.0..1.0).contains(&c){return Err(invalid("envelope confidence level must be in (0,1)"));}
    let utility_c:f64=f64_field(&conf.meta["confidence_level"],"utility confidence_level",0)?;
    // Conservative union bound: P(both families cover) >= c_utility + c_envelope - 1.
    if utility_c+c-1.0+1e-12<0.95{return Err(invalid("combined utility/envelope coverage lower bound is below 0.95; use jointly calibrated bands or allocate alpha so the union bound reaches 95%"));}
    let expected_sha=validate_sha(&meta["source_confidence_receipt_sha256"],"source_confidence_receipt_sha256")?;
    if expected_sha!=conf_sha{return Err(invalid("envelope does not bind the exact confidence-receipt bytes"));}
    if validate_sha(&meta["inference_receipt_sha256"],"envelope inference_receipt_sha256")?!=expected_inference
        || validate_sha(&conf.meta["inference_receipt_sha256"],"confidence inference_receipt_sha256")?!=expected_inference {
        return Err(invalid("confidence and envelope must bind the same caller-pinned inference receipt digest"));
    }
    for (mk,ck) in [("experiment_id","experiment_id"),("dataset_id","dataset_id"),("dataset_revision","dataset_revision"),("split","split"),("bundle_lock_sha256","bundle_lock_sha256"),("paired_unit_set_sha256","paired_unit_set_sha256"),("source_sweep_sha256","sweep_input_sha256")]{
        if meta[mk]!=conf.meta[ck]{return Err(invalid(format!("envelope {mk} differs from confidence receipt")));}
    }
    let expected: BTreeSet<Key>=conf.points.iter().map(|p|p.key.clone()).collect();
    let actual:BTreeSet<Key>=rows.keys().cloned().collect();
    if actual!=expected{return Err(invalid("operational envelope point grid does not exactly match confidence receipt"));}
    let count:usize=meta["point_count"].parse().map_err(|_|invalid("point_count must be an integer"))?;
    if count!=expected.len()||rows.len()!=count{return Err(invalid("envelope point_count must match exact input grid"));}
    for k in ["pooled_harm_limit","worst_group_harm_limit","cvar95_loss_limit"]{
        let limit=f64_field(&meta[k],k,0)?;if !limit.is_finite()||limit<0.0{return Err(invalid(format!("{k} must be finite and non-negative")));}
        if k.contains("harm")&&limit>1.0{return Err(invalid(format!("{k} must be <= 1")));}
    }
    meta["safety_violations_max"].parse::<u64>().map_err(|_|invalid("safety_violations_max must be a non-negative integer"))?;
    Ok(EnvelopeTable{meta,rows})
}

fn feasibility(p:&Point,e:&Envelope,t:&EnvelopeTable)->(&'static str,Vec<&'static str>){
    if p.upstream_status=="INFEASIBLE"||p.status=="INFEASIBLE"{return ("ROBUST_INFEASIBLE",vec!["UPSTREAM_INFEASIBLE_CANNOT_BE_REHABILITATED"]);}
    let mut definite=Vec::new();let mut unresolved=Vec::new();
    if !e.cost_complete{unresolved.push("COST_MEASUREMENT_INCOMPLETE");}
    if e.risk_state=="FAILED"{definite.push("RISK_CERTIFICATE_FAILED");}
    if e.risk_state=="UNKNOWN"{unresolved.push("RISK_CERTIFICATE_UNKNOWN");}
    let check=|iv:Interval,limit:f64,name_def:&'static str,name_unc:&'static str,definite:&mut Vec<&'static str>,unresolved:&mut Vec<&'static str>|{
        if iv.lo>limit{definite.push(name_def);}else if iv.hi>limit{unresolved.push(name_unc);}
    };
    check(e.latency, p.latency_budget as f64,"LATENCY_DEFINITELY_OVER_BUDGET","LATENCY_INTERVAL_CROSSES_BUDGET",&mut definite,&mut unresolved);
    check(e.energy,p.energy_budget as f64,"ENERGY_DEFINITELY_OVER_BUDGET","ENERGY_INTERVAL_CROSSES_BUDGET",&mut definite,&mut unresolved);
    check(e.pooled_harm, t.meta["pooled_harm_limit"].parse().unwrap_or(f64::NAN),"POOLED_HARM_LIMIT_EXCEEDED","POOLED_HARM_INTERVAL_CROSSES_LIMIT",&mut definite,&mut unresolved);
    check(e.worst_group_harm,t.meta["worst_group_harm_limit"].parse().unwrap_or(f64::NAN),"WORST_GROUP_HARM_LIMIT_EXCEEDED","WORST_GROUP_HARM_INTERVAL_CROSSES_LIMIT",&mut definite,&mut unresolved);
    check(e.cvar95,t.meta["cvar95_loss_limit"].parse().unwrap_or(f64::NAN),"CVAR95_LIMIT_EXCEEDED","CVAR95_INTERVAL_CROSSES_LIMIT",&mut definite,&mut unresolved);
    let max=t.meta["safety_violations_max"].parse::<u64>().unwrap_or(u64::MAX);
    if e.safety_lower>max{definite.push("SAFETY_COUNT_LIMIT_EXCEEDED");}else if e.safety_upper>max{unresolved.push("SAFETY_COUNT_INTERVAL_CROSSES_LIMIT");}
    if !definite.is_empty(){("ROBUST_INFEASIBLE",definite)}
    else if !unresolved.is_empty(){("ROBUST_UNRESOLVED_FEASIBILITY",unresolved)}
    else {("ROBUST_FEASIBLE_PENDING_FRONTIER",vec![])}
}

fn certainly_dominates(a:&Point,ae:&Envelope,b:&Point,be:&Envelope)->bool{
    a.utility_lcb>=b.utility_ucb && ae.latency.hi<=be.latency.hi && ae.energy.hi<=be.energy.hi &&
    (a.utility_lcb>b.utility_ucb||ae.latency.hi<be.latency.hi||ae.energy.hi<be.energy.hi)
}
fn possibly_dominates(a:&Point,ae:&Envelope,b:&Point,be:&Envelope)->bool{
    a.utility_ucb>=b.utility_lcb && ae.latency.hi<=be.latency.hi && ae.energy.hi<=be.energy.hi &&
    (a.utility_ucb>b.utility_lcb||ae.latency.hi<be.latency.hi||ae.energy.hi<be.energy.hi)
}
fn exact_equivalent(a:&Point,ae:&Envelope,b:&Point,be:&Envelope)->bool{
    a.utility_estimate==b.utility_estimate&&a.utility_lcb==b.utility_lcb&&a.utility_ucb==b.utility_ucb&&
    ae.latency.lo==be.latency.lo&&ae.latency.hi==be.latency.hi&&ae.energy.lo==be.energy.lo&&ae.energy.hi==be.energy.hi
}

fn classify(conf:&ConfidenceReceipt,env:&EnvelopeTable)->Vec<Classified>{
    let mut points=conf.points.clone();points.sort_by(|a,b|a.key.cmp(&b.key));
    let mut out=Vec::new();
    for p in &points{
        let e=env.rows[&p.key].clone();
        let (gate,failures)=feasibility(p,&e,env);
        if gate!="ROBUST_FEASIBLE_PENDING_FRONTIER"{
            out.push(Classified{point:p.clone(),envelope:e,status:gate,reason:failures.join(";"),dominator:"-".to_owned()});continue;
        }
        let mut strict=Vec::new();let mut possible=Vec::new();let mut equivalents=Vec::new();
        for q in &points{
            if q.key==p.key{continue;}
            let qe=&env.rows[&q.key];
            let (qgate,_)=feasibility(q,qe,env);
            if qgate!="ROBUST_FEASIBLE_PENDING_FRONTIER"{continue;}
            if exact_equivalent(q,qe,p,&e){equivalents.push(q.key.label());continue;}
            if certainly_dominates(q,qe,p,&e){strict.push(q.key.label());}
            else if possibly_dominates(q,qe,p,&e){possible.push(q.key.label());}
        }
        strict.sort();possible.sort();equivalents.sort();
        let (status,reason,dominator)=if let Some(x)=strict.first(){
            ("ROBUST_FEASIBLE_DOMINATED","SIMULTANEOUS_UTILITY_AND_RESOURCE_BOUNDS_ESTABLISH_DOMINANCE".to_owned(),x.clone())
        }else if !possible.is_empty(){
            ("ROBUST_FRONTIER_UNRESOLVED", "POTENTIAL_DOMINATOR_REMAINS_UNRESOLVED".to_owned(),possible.join(";"))
        }else if !equivalents.is_empty(){
            ("ROBUST_DUPLICATE_EQUIVALENT","IDENTICAL_UTILITY_INTERVALS_AND_RESOURCE_INTERVALS".to_owned(),equivalents.join(";"))
        }else{
            ("ROBUST_FEASIBLE_FRONTIER","NO_FEASIBLE_POINT_CAN_DOMINATE_UNDER_SUPPLIED_BOUNDS".to_owned(),"-".to_owned())
        };
        out.push(Classified{point:p.clone(),envelope:e,status,reason,dominator});
    }
    out
}

fn analyze(conf_bytes:&[u8],expected_conf:&str,env_bytes:&[u8],expected_env:&str,expected_inference:&str)->Result<String,String>{
    let expected_conf=validate_sha(expected_conf,"expected confidence digest")?;
    let expected_env=validate_sha(expected_env,"expected envelope digest")?;
    let expected_inference=validate_sha(expected_inference,"expected inference receipt digest")?;
    let conf_sha=sha256_hex(conf_bytes);let env_sha=sha256_hex(env_bytes);
    if conf_sha!=expected_conf{return Err(invalid("confidence receipt bytes do not match pinned digest"));}
    if env_sha!=expected_env{return Err(invalid("operational envelope bytes do not match pinned digest"));}
    let conf_text=std::str::from_utf8(conf_bytes).map_err(|_|invalid("confidence receipt is not UTF-8"))?;
    let env_text=std::str::from_utf8(env_bytes).map_err(|_|invalid("operational envelope is not UTF-8"))?;
    let conf=parse_confidence(conf_text)?;
    let env=parse_envelopes(env_text,&conf,&conf_sha,&expected_inference)?;
    let classified=classify(&conf,&env);
    let count=|s:&str|classified.iter().filter(|x|x.status==s).count();
    let env_conf:f64=env.meta["confidence_level"].parse().unwrap_or(0.0);
    let util_conf:f64=conf.meta["confidence_level"].parse().unwrap_or(0.0);
    let joint_lower=(util_conf+env_conf-1.0).max(0.0);
    let mut output=String::new();
    let _=writeln!(output,"ROBUST_ENVELOPE\tformat\t{OUTPUT_FORMAT}");
    let _=writeln!(output,"ROBUST_ENVELOPE\tstatus\tROBUST_FEASIBILITY_AND_FRONTIER_ANALYSIS_ONLY");
    for k in ["experiment_id","dataset_id","dataset_revision","split","bundle_lock_sha256","paired_unit_set_sha256","sweep_input_sha256","inference_receipt_sha256"]{
        let v=if k=="inference_receipt_sha256"{expected_inference.as_str()}else{conf.meta[k].as_str()};
        let _=writeln!(output,"ROBUST_ENVELOPE\t{k}\t{v}");
    }
    let _=writeln!(output,"ROBUST_ENVELOPE\tconfidence_receipt_sha256\t{conf_sha}");
    let _=writeln!(output,"ROBUST_ENVELOPE\toperational_envelope_sha256\t{env_sha}");
    let _=writeln!(output,"ROBUST_ENVELOPE\tutility_confidence_level\t{}",conf.meta["confidence_level"]);
    let _=writeln!(output,"ROBUST_ENVELOPE\toperational_confidence_level\t{}",env.meta["confidence_level"]);
    let _=writeln!(output,"ROBUST_ENVELOPE\tjoint_coverage_lower_bound_union\t{joint_lower:.6}");
    let _=writeln!(output,"ROBUST_ENVELOPE\tcoverage_method\tBONFERRONI_UNION_BOUND");
    let _=writeln!(output,"ROBUST_ENVELOPE\tinterval_procedure_verified_here\tfalse");
    let _=writeln!(output,"ROBUST_ENVELOPE\thardware_meters_verified_here\tfalse");
    for s in ["ROBUST_FEASIBLE_FRONTIER","ROBUST_FEASIBLE_DOMINATED","ROBUST_FRONTIER_UNRESOLVED","ROBUST_DUPLICATE_EQUIVALENT","ROBUST_UNRESOLVED_FEASIBILITY","ROBUST_INFEASIBLE"]{
        let _=writeln!(output,"ROBUST_ENVELOPE\t{}_count\t{}",s.to_lowercase(),count(s));
    }
    let _=writeln!(output,"ROBUST_ENVELOPE\tperformance_claim\tNONE");
    let _=writeln!(output,"ROBUST_RESULT_HEADER\tbudget_point_id\tpolicy_id\tstatus\tutility_estimate\tutility_lcb\tutility_ucb\tlatency_lcb_us\tlatency_ucb_us\tlatency_budget_us\tenergy_lcb_uj\tenergy_ucb_uj\tenergy_budget_uj\tpooled_harm_ucb\tworst_group_harm_ucb\tcvar95_loss_ucb\tsafety_violations_lower\tsafety_violations_upper\tcost_measurement_complete\trisk_certificate_state\tupstream_status\tupstream_reasons\tgate_or_frontier_reason\trobust_dominator");
    for x in &classified{
        let e=&x.envelope;let p=&x.point;
        let _=writeln!(output,"ROBUST_RESULT\t{}\t{}\t{}\t{:.12}\t{:.12}\t{:.12}\t{:.6}\t{:.6}\t{}\t{:.6}\t{:.6}\t{}\t{:.8}\t{:.8}\t{:.8}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            p.key.budget,p.key.policy,x.status,p.utility_estimate,p.utility_lcb,p.utility_ucb,e.latency.lo,e.latency.hi,p.latency_budget,e.energy.lo,e.energy.hi,p.energy_budget,
            e.pooled_harm.hi,e.worst_group_harm.hi,e.cvar95.hi,e.safety_lower,e.safety_upper,e.cost_complete,e.risk_state,p.upstream_status,p.upstream_reasons,x.reason,x.dominator);
    }
    Ok(output)
}

fn run_cli(args:&[String])->Result<String,String>{
    if args.len()!=6{return Err("usage: vision-dvpc-robust-envelope CONFIDENCE.tsv EXPECTED_CONFIDENCE_SHA256 ENVELOPE.tsv EXPECTED_ENVELOPE_SHA256 EXPECTED_INFERENCE_RECEIPT_SHA256".to_owned());}
    let c=fs::read(&args[1]).map_err(|e|invalid(format!("cannot read confidence receipt: {e}")))?;
    let e=fs::read(&args[3]).map_err(|x|invalid(format!("cannot read envelope: {x}")))?;
    analyze(&c,&args[2],&e,&args[4],&args[5])
}
fn main(){let args:Vec<String>=env::args().collect();match run_cli(&args){Ok(s)=>print!("{s}"),Err(e)=>{eprintln!("VIS-DVPC-019 error: {e}");std::process::exit(2);}}}

#[cfg(test)]
mod tests {
    use super::*;
    const A:&str="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const B:&str="bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const C:&str="cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    #[derive(Clone)]
    struct S<'a>{budget:&'a str,policy:&'a str,status:&'a str,upstream:&'a str,utility:f64,ulo:f64,uhi:f64,lat:u64,en:u64,latbudget:u64,enbudget:u64}
    fn confidence(ss:&[S<'_>])->String{
        let mut o=String::new();
        let rows:Vec<(&str,&str)>=vec![
            ("format",CONF_FORMAT),("status",CONF_STATUS),("experiment_id","synth019"),("dataset_id","synthetic"),
            ("dataset_revision","v1"),("split","test"),("bundle_lock_sha256",B),("paired_unit_set_sha256",C),
            ("sweep_input_sha256",A),("pareto_receipt_sha256",A),("confidence_bands_sha256",B),("inference_receipt_sha256",C),
            ("confidence_level","0.975"),("familywise_simultaneous","true"),("interval_procedure_verified_here","false"),
            ("upstream_safety_certificate_verified_here","false"),("band_nondominated_count","1"),("band_dominated_count","0"),
            ("unresolved_count","0"),("duplicate_equivalent_count","0"),("infeasible_count","0"),("performance_claim","NONE")
        ];
        for (k,v) in rows{let _=writeln!(o,"CONFIDENCE_FRONTIER\t{k}\t{v}");}
        o.push_str("CONFIDENCE_RESULT_HEADER\tbudget_point_id\tpolicy_id\tstatus\tutility_estimate\tutility_lcb\tutility_ucb\tactual_all_in_latency_us\tactual_all_in_energy_uj\tlatency_budget_us\tenergy_budget_uj\tupstream_point_status\tupstream_reasons\treason\tupstream_point_dominated_by\tband_dominated_by\n");
        for s in ss{let _=writeln!(o,"CONFIDENCE_RESULT\t{}\t{}\t{}\t{:.4}\t{:.4}\t{:.4}\t{}\t{}\t{}\t{}\t{}\t-\t-\t-\t-",s.budget,s.policy,s.status,s.utility,s.ulo,s.uhi,s.lat,s.en,s.latbudget,s.enbudget,s.upstream);}
        o
    }
    fn envelope(conf:&str,ss:&[S<'_>],latlo:f64,lathi:f64,harmlo:f64,harmhi:f64,risk:&str,coverage:&str)->String{
        let mut o=String::new();
        let rows:Vec<(&str,&str)>=vec![
            ("format","VIS-DVPC-OP-ENVELOPE-001-v1"),("experiment_id","synth019"),("dataset_id","synthetic"),
            ("dataset_revision","v1"),("split","test"),("bundle_lock_sha256",B),("paired_unit_set_sha256",C),
            ("source_sweep_sha256",A),("source_confidence_receipt_sha256",&sha256_hex(conf.as_bytes())),
            ("inference_receipt_sha256",C),("confidence_level","0.975"),("familywise_simultaneous","true"),
            ("coverage_scope",coverage),("point_count",&ss.len().to_string()),("pooled_harm_limit","0.10"),
            ("worst_group_harm_limit","0.10"),("cvar95_loss_limit","0.25"),("safety_violations_max","0")
        ];
        for(k,v)in rows{let _=writeln!(o,"META\t{k}\t{v}");}
        o.push_str("ENVELOPE_HEADER\tbudget_point_id\tpolicy_id\tlatency_lcb_us\tlatency_ucb_us\tenergy_lcb_uj\tenergy_ucb_uj\tpooled_harm_lcb\tpooled_harm_ucb\tworst_group_harm_lcb\tworst_group_harm_ucb\tcvar95_loss_lcb\tcvar95_loss_ucb\tsafety_violations_lower\tsafety_violations_upper\tcost_measurement_complete\trisk_certificate_state\n");
        for s in ss{
            let _=writeln!(o,"ENVELOPE\t{}\t{}\t{}\t{}\t10\t12\t{}\t{}\t{}\t{}\t0.05\t0.08\t0\t0\ttrue\t{}",s.budget,s.policy,latlo,lathi,harmlo,harmhi,harmlo,harmhi,risk);
        }
        o
    }
    fn run(c:&str,e:&str)->Result<String,String>{analyze(c.as_bytes(),&sha256_hex(c.as_bytes()),e.as_bytes(),&sha256_hex(e.as_bytes()),C)}
    fn one(status:&'static str,upstream:&'static str)->Vec<S<'static>>{vec![S{budget:"p1",policy:"cheap",status,upstream,utility:0.5,ulo:0.45,uhi:0.55,lat:40,en:20,latbudget:100,enbudget:100}]}

    #[test] fn robust_feasible_single_point_is_frontier(){
        let s=one("BAND_NONDOMINATED","FEASIBLE_FRONTIER");let c=confidence(&s);let e=envelope(&c,&s,30.0,50.0,0.02,0.04,"CERTIFIED","ALL_POLICY_BUDGET_SAFETY_METRICS");
        let out=run(&c,&e).unwrap();assert!(out.contains("ROBUST_RESULT\tp1\tcheap\tROBUST_FEASIBLE_FRONTIER"));
        assert!(out.contains("joint_coverage_lower_bound_union\t0.950000"));
    }
    #[test] fn interval_crossing_latency_ceiling_is_unresolved(){
        let s=one("BAND_NONDOMINATED","FEASIBLE_FRONTIER");let c=confidence(&s);let e=envelope(&c,&s,80.0,120.0,0.02,0.04,"CERTIFIED","ALL_POLICY_BUDGET_SAFETY_METRICS");
        assert!(run(&c,&e).unwrap().contains("ROBUST_UNRESOLVED_FEASIBILITY"));
    }
    #[test] fn latency_definitely_over_budget_is_infeasible(){
        let s=one("BAND_NONDOMINATED","FEASIBLE_FRONTIER");let c=confidence(&s);let e=envelope(&c,&s,101.0,120.0,0.02,0.04,"CERTIFIED","ALL_POLICY_BUDGET_SAFETY_METRICS");
        assert!(run(&c,&e).unwrap().contains("LATENCY_DEFINITELY_OVER_BUDGET"));
    }
    #[test] fn unsafe_upstream_point_cannot_be_rehabilitated(){
        let s=one("INFEASIBLE","INFEASIBLE");let c=confidence(&s);let e=envelope(&c,&s,1.0,2.0,0.0,0.01,"CERTIFIED","ALL_POLICY_BUDGET_SAFETY_METRICS");
        assert!(run(&c,&e).unwrap().contains("UPSTREAM_INFEASIBLE_CANNOT_BE_REHABILITATED"));
    }
    #[test] fn harm_interval_crossing_ceiling_is_unresolved(){
        let s=one("BAND_NONDOMINATED","FEASIBLE_FRONTIER");let c=confidence(&s);let e=envelope(&c,&s,30.0,50.0,0.08,0.12,"CERTIFIED","ALL_POLICY_BUDGET_SAFETY_METRICS");
        assert!(run(&c,&e).unwrap().contains("POOLED_HARM_INTERVAL_CROSSES_LIMIT"));
    }
    #[test] fn explicit_risk_certificate_failure_is_infeasible(){
        let s=one("BAND_NONDOMINATED","FEASIBLE_FRONTIER");let c=confidence(&s);let e=envelope(&c,&s,30.0,50.0,0.02,0.04,"FAILED","ALL_POLICY_BUDGET_SAFETY_METRICS");
        assert!(run(&c,&e).unwrap().contains("RISK_CERTIFICATE_FAILED"));
    }
    #[test] fn unknown_risk_certificate_is_unresolved(){
        let s=one("BAND_NONDOMINATED","FEASIBLE_FRONTIER");let c=confidence(&s);let e=envelope(&c,&s,30.0,50.0,0.02,0.04,"UNKNOWN","ALL_POLICY_BUDGET_SAFETY_METRICS");
        assert!(run(&c,&e).unwrap().contains("RISK_CERTIFICATE_UNKNOWN"));
    }
    #[test] fn incomplete_measurement_is_unresolved(){
        let s=one("BAND_NONDOMINATED","FEASIBLE_FRONTIER");let c=confidence(&s);let mut e=envelope(&c,&s,30.0,50.0,0.02,0.04,"CERTIFIED","ALL_POLICY_BUDGET_SAFETY_METRICS");
        e=e.replace("true\tCERTIFIED","false\tCERTIFIED");assert!(run(&c,&e).unwrap().contains("COST_MEASUREMENT_INCOMPLETE"));
    }
    #[test] fn insufficient_joint_coverage_is_rejected(){
        let s=one("BAND_NONDOMINATED","FEASIBLE_FRONTIER");let mut c=confidence(&s).replace("confidence_level\t0.975","confidence_level\t0.95");
        // Keep the input valid and let the conservative union bound reject 0.95 + 0.975 - 1 < 0.95.
        let e=envelope(&c,&s,30.0,50.0,0.02,0.04,"CERTIFIED","ALL_POLICY_BUDGET_SAFETY_METRICS");assert!(run(&c,&e).unwrap_err().contains("combined utility/envelope coverage"));
        c.clear();
    }
    #[test] fn unknown_coverage_scope_is_rejected(){
        let s=one("BAND_NONDOMINATED","FEASIBLE_FRONTIER");let c=confidence(&s);let e=envelope(&c,&s,30.0,50.0,0.02,0.04,"CERTIFIED","ALL_FEASIBLE_POINTS");
        assert!(run(&c,&e).unwrap_err().contains("family-wise simultaneous"));
    }
    #[test] fn changed_confidence_receipt_bytes_are_rejected(){
        let s=one("BAND_NONDOMINATED","FEASIBLE_FRONTIER");let c=confidence(&s);let e=envelope(&c,&s,30.0,50.0,0.02,0.04,"CERTIFIED","ALL_POLICY_BUDGET_SAFETY_METRICS");
        let mut changed=c.clone();changed.push_str(" ");assert!(analyze(changed.as_bytes(),&sha256_hex(c.as_bytes()),e.as_bytes(),&sha256_hex(e.as_bytes()),C).unwrap_err().contains("pinned digest"));
    }
    #[test] fn sha256_known_vector(){assert_eq!(sha256_hex(b"abc"),"ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");}
}
