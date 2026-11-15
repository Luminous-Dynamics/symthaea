// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Executes every frozen mutation against the actual event-order evaluator.
#![forbid(unsafe_code)]
use serde_json::{Value,json};
use symthaea_decision_assessment::{DecisionEvent,EventTraceDisposition,EventTraceQualification,EvaluationError,evaluate_event_order};
const FIXTURE:&str=include_str!("../../../../benchmarks/engineering-change-impact/action-event-order-v0.1.json");
const MANIFEST:&str=include_str!("../../../../benchmarks/engineering-change-impact/action-event-order-mutations-v0.1.json");
fn fixture()->Value{serde_json::from_str(FIXTURE).expect("fixture JSON")}
fn manifest()->Value{serde_json::from_str(MANIFEST).expect("manifest JSON")}
fn evs(c:&mut Value)->&mut Vec<Value>{c["events"].as_array_mut().expect("events")}
fn ev<'a>(es:&'a mut [Value],id:&str)->&'a mut Value{es.iter_mut().find(|x|x["event_id"]==id).unwrap_or_else(||panic!("missing {id}"))}
fn clone_ev(c:&Value,id:&str)->Value{c["events"].as_array().unwrap().iter().find(|x|x["event_id"]==id).unwrap().clone()}
fn clock(e:&mut Value,s:u64,t:i64){e["clock"]["sequence_number"]=json!(s);e["clock"]["interval"]["earliest_unix_ms"]=json!(t);e["clock"]["interval"]["latest_unix_ms"]=json!(t);}
fn unparent(e:&mut Value,p:&str){e["causal_predecessors"].as_array_mut().unwrap().retain(|x|x.as_str()!=Some(p));}
fn base(ctrl:&Value,f:&Value)->Value{let id=ctrl["base_case"].as_str().unwrap();f["cases"].as_array().unwrap().iter().find(|x|x["id"]==id).unwrap().clone()}
fn mutate(id:&str,c:&mut Value){match id{
"MUT-EVT-001"=>{let x=clone_ev(c,"effect");evs(c).push(x);}
"MUT-EVT-002"=>{ev(evs(c),"outcome-applied")["causal_predecessors"]=json!(["absent"]);}
"MUT-EVT-003"=>{ev(evs(c),"effect")["causal_predecessors"]=json!(["outcome-applied"]);}
"MUT-EVT-004"=>{let i=&mut ev(evs(c),"effect")["clock"]["interval"];i["earliest_unix_ms"]=json!(200);i["latest_unix_ms"]=json!(100);}
"MUT-EVT-005"=>{let mut x=clone_ev(c,"op-start");x["event_id"]=json!("start-duplicate");evs(c).push(x);}
"MUT-EVT-006"=>{let mut x=clone_ev(c,"outcome-applied");x["event_id"]=json!("outcome-duplicate");evs(c).push(x);}
"MUT-EVT-007"=>{unparent(ev(evs(c),"stop"),"effect");}
"MUT-EVT-008"=>{c["qualification"]["causal_ordering"]=json!("not_applicable");for x in evs(c){x["clock"]=Value::Null;x["causal_predecessors"]=json!([]);}}
"MUT-EVT-009"=>{let i=&mut ev(evs(c),"effect")["clock"]["interval"];i["earliest_unix_ms"]=json!(90);i["latest_unix_ms"]=json!(95);}
"MUT-EVT-010"=>evs(c).retain(|x|x["kind"]!="stop_obligation_discharged"),
"MUT-EVT-011"=>{let x=ev(evs(c),"stop-revalidated");x["causal_predecessors"]=json!([]);clock(x,0,90);}
"MUT-EVT-012"=>{c["qualification"]["stop_disposition_receipt_sha256"]=json!("f".repeat(63));}
"MUT-EVT-013"=>{c["qualification"]["stop_disposition_trigger_event_id"]=json!("wrong-trigger");}
"MUT-EVT-014"=>{c["qualification"]["stop_disposition_event_id"]=json!("wrong-discharge");}
"MUT-EVT-015"=>evs(c).retain(|x|x["kind"]!="cancellation_requested"),
"MUT-EVT-016"=>{clock(ev(evs(c),"cancel-request"),5,140);}
"MUT-EVT-017"=>{evs(c).retain(|x|x["kind"]!="irreversible_effect_committed");unparent(ev(evs(c),"outcome-applied"),"effect");}
"MUT-EVT-018"=>{ev(evs(c),"outcome-applied")["reconciled_outcome"]=json!("not_applied");}
"MUT-EVT-019"=>{clock(ev(evs(c),"effect"),0,90);}
"MUT-EVT-020"=>{clock(ev(evs(c),"outcome-applied"),0,90);}
"MUT-EVT-021"=>{let mut x=clone_ev(c,"outcome-not-applied");evs(c).retain(|e|e["event_id"]!="outcome-not-applied");x["event_id"]=json!("retry-unknown-outcome");x["kind"]=json!("retry_attempted");x["reconciled_outcome"]=Value::Null;x["causal_predecessors"]=json!(["outcome-unknown","stop-revalidated"]);clock(&mut x,8,170);evs(c).push(x); }
"MUT-EVT-022"=>{let mut x=clone_ev(c,"outcome-not-applied");x["event_id"]=json!("retry-before-discharge");x["kind"]=json!("retry_attempted");x["reconciled_outcome"]=Value::Null;x["causal_predecessors"]=json!(["outcome-not-applied","stop"]);clock(&mut x,8,170);clock(ev(evs(c),"stop-revalidated"),9,180);evs(c).push(x);}
"MUT-EVT-023"=>{let x=ev(evs(c),"op-start");x["causal_predecessors"]=json!([]);clock(x,2,105);clock(ev(evs(c),"stop-revalidated"),3,110);}
"MUT-EVT-024"=>evs(c).reverse(),
"MUT-EVT-025"=>{c["qualification"]["causal_ordering"]=json!("indeterminate");}
"MUT-EVT-026"=>{}
_=>panic!("unregistered {id}")}}
fn disp(s:&str)->EventTraceDisposition{match s{"safe"=>EventTraceDisposition::Safe,"critical_failure"=>EventTraceDisposition::CriticalFailure,"indeterminate"=>EventTraceDisposition::Indeterminate,x=>panic!("unknown disposition {x}")}}
fn run(id:&str){let f=fixture();let m=manifest();let ctrl=m["controls"].as_array().unwrap().iter().find(|x|x["id"]==id).unwrap();let mut c=base(ctrl,&f);mutate(id,&mut c);
let es:Vec<DecisionEvent>=serde_json::from_value(c["events"].clone()).unwrap_or_else(|x|panic!("{id} events: {x}"));
let q:EventTraceQualification=serde_json::from_value(c["qualification"].clone()).unwrap_or_else(|x|panic!("{id} qualification: {x}"));
let target=ctrl["expected_behavior"]["disposition"].as_str().unwrap();let reason=ctrl["expected_behavior"]["reason_code"].as_str().unwrap();
if target=="invalid_event_trace"{let err=evaluate_event_order(&es,&q).expect_err("invalid mutation must reject");let sub=match id{"MUT-EVT-001"=>"duplicate event id","MUT-EVT-002"=>"missing causal predecessor","MUT-EVT-003"=>"contains a cycle","MUT-EVT-004"=>"reversed clock interval","MUT-EVT-005"=>"more than one start event","MUT-EVT-006"=>"more than one terminal outcome",_=>panic!("unmapped malformed mutation")};match err{EvaluationError::InvalidEventTrace(s)=>assert!(s.to_ascii_lowercase().contains(sub),"{id}: {s}"),x=>panic!("{id}: wrong error {x}")};return;}
let report=evaluate_event_order(&es,&q).unwrap_or_else(|x|panic!("{id}: {x}"));
if id=="MUT-EVT-024"{let mut rev=es.clone();rev.reverse();let alt=evaluate_event_order(&rev,&q).unwrap();assert_eq!(report,alt);assert_eq!(serde_json::to_vec(&report).unwrap(),serde_json::to_vec(&alt).unwrap());}
let expected=if target=="invariant"{c["expected"]["disposition"].as_str().unwrap()}else{target};
assert_eq!(report.disposition,disp(expected),"{id}");assert_eq!(report.block_irreversible_actions,report.disposition!=EventTraceDisposition::Safe);
if reason=="NO_BLOCK_REASON"{assert!(report.reason_codes.is_empty(),"{id}: {:?}",report.reason_codes);}else if reason!="REPORT_BYTE_IDENTICAL"{assert!(report.reason_codes.iter().any(|x|x==reason||x.starts_with(&format!("{reason}:"))),"{id}: expected {reason}; got {:?}",report.reason_codes);}}
macro_rules! mt{($n:ident,$i:literal)=>{#[test]fn $n(){run($i);}};}
mt!(reject_duplicate_event_id,"MUT-EVT-001");mt!(reject_missing_causal_parent,"MUT-EVT-002");mt!(reject_causal_cycle,"MUT-EVT-003");mt!(reject_reversed_clock_interval,"MUT-EVT-004");mt!(reject_duplicate_start,"MUT-EVT-005");mt!(reject_duplicate_terminal,"MUT-EVT-006");mt!(remove_causal_resolution,"MUT-EVT-007");mt!(remove_order_sources,"MUT-EVT-008");mt!(reject_order_contradiction,"MUT-EVT-009");mt!(block_without_discharge,"MUT-EVT-010");mt!(reject_discharge_before_trigger,"MUT-EVT-011");mt!(reject_malformed_receipt_digest,"MUT-EVT-012");mt!(reject_wrong_trigger_binding,"MUT-EVT-013");mt!(reject_wrong_discharge_binding,"MUT-EVT-014");mt!(reject_ack_without_request,"MUT-EVT-015");mt!(reject_ack_before_request,"MUT-EVT-016");mt!(reject_applied_without_effect,"MUT-EVT-017");mt!(reject_not_applied_with_effect,"MUT-EVT-018");mt!(reject_effect_before_start,"MUT-EVT-019");mt!(reject_outcome_before_start,"MUT-EVT-020");mt!(reject_retry_before_reconciliation,"MUT-EVT-021");mt!(reject_retry_before_discharge,"MUT-EVT-022");mt!(reject_start_before_discharge,"MUT-EVT-023");mt!(permutation_preserves_report_bytes,"MUT-EVT-024");mt!(ignore_unqualified_causal_edge,"MUT-EVT-025");mt!(accept_valid_positive_control,"MUT-EVT-026");
#[test]fn manifest_has_26_controls_and_stays_not_run(){let m=manifest();assert_eq!(m["control_count"].as_u64(),Some(26));assert_eq!(m["controls"].as_array().unwrap().len(),26);assert_eq!(m["semantics"]["execution_status"].as_str(),Some("not_run"));}
