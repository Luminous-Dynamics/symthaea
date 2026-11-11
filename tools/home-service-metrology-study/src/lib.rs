//! Synthetic-only preregistration gate for household-appliance metrology studies.
//! This validates study design structure; it does not analyze performance data.

#![forbid(unsafe_code)]

use serde::de::{MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer};
use serde_json::{Map, Value};
use std::collections::BTreeSet;
use std::fmt;

pub const MANIFEST_JSON: &str = include_str!("../../../docs/research/home-service-metrology/HOME_SERVICE_METROLOGY_STUDY_DESIGN_V0_1.json");
pub const MANIFEST_SCHEMA_JSON: &str = include_str!("../../../docs/research/home-service-metrology/HOME_SERVICE_METROLOGY_STUDY_DESIGN_V0_1.schema.json");
pub const CORPUS_JSON: &str = include_str!("../../../docs/research/home-service-metrology/HOME_SERVICE_METROLOGY_STUDY_CORPUS_V0_1.json");
pub const CORPUS_SCHEMA_JSON: &str = include_str!("../../../docs/research/home-service-metrology/HOME_SERVICE_METROLOGY_STUDY_CORPUS_V0_1.schema.json");

const EXPECTED_CASES: &[(&str,&str)] = &[
("HSM-001","ELIGIBLE_FOR_PREREGISTRATION_REVIEW_ONLY"),
("HSM-002","NOT_COMPARABLE"),("HSM-003","NOT_COMPARABLE"),("HSM-004","NOT_COMPARABLE"),
("HSM-005","NOT_COMPARABLE"),("HSM-006","NOT_COMPARABLE"),
("HSM-007","MORE_EVIDENCE_REQUIRED"),("HSM-008","MORE_EVIDENCE_REQUIRED"),("HSM-009","MORE_EVIDENCE_REQUIRED"),
("HSM-010","MORE_EVIDENCE_REQUIRED"),("HSM-011","MORE_EVIDENCE_REQUIRED"),("HSM-012","MORE_EVIDENCE_REQUIRED"),
("HSM-013","MORE_EVIDENCE_REQUIRED"),("HSM-014","NOT_COMPARABLE"),("HSM-015","NOT_COMPARABLE"),
("HSM-016","NOT_COMPARABLE"),("HSM-017","MORE_EVIDENCE_REQUIRED"),("HSM-018","MORE_EVIDENCE_REQUIRED"),
("HSM-019","MORE_EVIDENCE_REQUIRED"),("HSM-020","MORE_EVIDENCE_REQUIRED"),("HSM-021","MORE_EVIDENCE_REQUIRED"),
("HSM-022","REJECTED_FOR_STUDY_SCOPE"),("HSM-023","REJECTED_FOR_STUDY_SCOPE"),
("HSM-024","REJECTED_FOR_STUDY_SCOPE"),("HSM-025","REJECTED_FOR_STUDY_SCOPE"),("HSM-026","MORE_EVIDENCE_REQUIRED"),
("HSM-027","MORE_EVIDENCE_REQUIRED"),("HSM-028","NOT_COMPARABLE"),("HSM-029","NOT_COMPARABLE"),
("HSM-030","MORE_EVIDENCE_REQUIRED"),("HSM-031","NOT_COMPARABLE"),("HSM-032","REJECTED_FOR_STUDY_SCOPE"),
("HSM-033","MORE_EVIDENCE_REQUIRED"),("HSM-034","MORE_EVIDENCE_REQUIRED"),("HSM-035","NOT_COMPARABLE"),
("HSM-036","ELIGIBLE_FOR_PREREGISTRATION_REVIEW_ONLY"),
];
const SIGNAL_FIELDS:&[&str]=&[
"primary_endpoint_defined","functional_unit_defined","comparators_pinned","same_protocol_across_arms",
"same_conditions_and_trace","software_model_versions_pinned","adaptation_budget_frozen",
"repeatability_runs_per_cell","independent_lab_count","production_lot_count","units_per_lot",
"household_count","duty_cycle_profile_count","holdout_lab_count","holdout_lot_count","holdout_household_count",
"holdout_not_used_for_tuning","power_analysis_present","hierarchical_model_registered","analysis_code_digest_present",
"calibration_and_provenance_complete","independent_reference_method_present","randomization_preregistered",
"assessor_blinding_or_independent_adjudication","exclusion_rules_preregistered","missingness_policy_preregistered",
"consent_and_privacy_plan_present","purpose_bound_data_use","pseudoreplication_guard",
"hard_gates_noncompensatory","outlier_rule_preregistered","deviation_log_enabled","cluster_nesting_declared",
"analysis_preregistered","all_variance_strata_separately_reported","no_unplanned_product_revision_mixing"
];

#[derive(Debug)] struct UniqueJson(Value);
impl<'de> Deserialize<'de> for UniqueJson{
 fn deserialize<D>(d:D)->Result<Self,D::Error> where D:Deserializer<'de>{d.deserialize_any(UniqueVisitor)}
}
struct UniqueVisitor;
impl<'de> Visitor<'de> for UniqueVisitor{
 type Value=UniqueJson;
 fn expecting(&self,f:&mut fmt::Formatter<'_>)->fmt::Result{f.write_str("a JSON value without duplicate object members")}
 fn visit_bool<E>(self,v:bool)->Result<Self::Value,E> where E:serde::de::Error{Ok(UniqueJson(Value::Bool(v)))}
 fn visit_i64<E>(self,v:i64)->Result<Self::Value,E> where E:serde::de::Error{Ok(UniqueJson(Value::Number(v.into())))}
 fn visit_u64<E>(self,v:u64)->Result<Self::Value,E> where E:serde::de::Error{Ok(UniqueJson(Value::Number(v.into())))}
 fn visit_f64<E>(self,v:f64)->Result<Self::Value,E> where E:serde::de::Error{let n=serde_json::Number::from_f64(v).ok_or_else(||E::custom("non-finite"))?;Ok(UniqueJson(Value::Number(n)))}
 fn visit_str<E>(self,v:&str)->Result<Self::Value,E> where E:serde::de::Error{Ok(UniqueJson(Value::String(v.to_owned())))}
 fn visit_string<E>(self,v:String)->Result<Self::Value,E> where E:serde::de::Error{Ok(UniqueJson(Value::String(v)))}
 fn visit_unit<E>(self)->Result<Self::Value,E> where E:serde::de::Error{Ok(UniqueJson(Value::Null))}
 fn visit_none<E>(self)->Result<Self::Value,E> where E:serde::de::Error{Ok(UniqueJson(Value::Null))}
 fn visit_seq<A>(self,mut seq:A)->Result<Self::Value,A::Error> where A:SeqAccess<'de>{let mut a=Vec::new();while let Some(v)=seq.next_element::<UniqueJson>()?{a.push(v.0);}Ok(UniqueJson(Value::Array(a)))}
 fn visit_map<A>(self,mut map:A)->Result<Self::Value,A::Error> where A:MapAccess<'de>{let mut o=Map::new();let mut seen=BTreeSet::new();while let Some(k)=map.next_key::<String>()?{if !seen.insert(k.clone()){return Err(serde::de::Error::custom(format!("duplicate member {k}")));}let v=map.next_value::<UniqueJson>()?;o.insert(k,v.0);}Ok(UniqueJson(Value::Object(o)))}
}
fn parse(s:&str)->Result<Value,String>{let mut d=serde_json::Deserializer::from_str(s);let v=UniqueJson::deserialize(&mut d).map_err(|e|e.to_string())?;d.end().map_err(|e|e.to_string())?;Ok(v.0)}
fn exact_keys(v:&Value,keys:&[&str])->bool{let Some(o)=v.as_object()else{return false};o.len()==keys.len()&&keys.iter().all(|k|o.contains_key(*k))}
fn text<'a>(v:&'a Value,k:&str)->&'a str{v.get(k).and_then(Value::as_str).unwrap_or("")}
fn flag(v:&Value,k:&str)->bool{v.get(k).and_then(Value::as_bool)==Some(true)}
fn count(v:&Value,k:&str)->u64{v.get(k).and_then(Value::as_u64).unwrap_or(0)}

pub fn classify_signals(s:&Value)->&'static str{
 if !flag(s,"primary_endpoint_defined")||!flag(s,"functional_unit_defined")||!flag(s,"comparators_pinned")
 ||!flag(s,"same_protocol_across_arms")||!flag(s,"same_conditions_and_trace")
 ||!flag(s,"software_model_versions_pinned")||!flag(s,"adaptation_budget_frozen")
 {return "NOT_COMPARABLE";}
 if !flag(s,"consent_and_privacy_plan_present")||!flag(s,"purpose_bound_data_use")
 ||!flag(s,"pseudoreplication_guard")||!flag(s,"hard_gates_noncompensatory")||!flag(s,"holdout_not_used_for_tuning")
 {return "REJECTED_FOR_STUDY_SCOPE";}
 if count(s,"repeatability_runs_per_cell")<2||count(s,"independent_lab_count")<2
 ||count(s,"production_lot_count")<2||count(s,"units_per_lot")<2
 ||count(s,"household_count")<2||count(s,"duty_cycle_profile_count")<2
 {return "NOT_COMPARABLE";}
 if count(s,"holdout_lab_count")<1||count(s,"holdout_lot_count")<1||count(s,"holdout_household_count")<1
 {return "MORE_EVIDENCE_REQUIRED";}
 if !flag(s,"power_analysis_present")||!flag(s,"hierarchical_model_registered")
 ||!flag(s,"analysis_code_digest_present")||!flag(s,"calibration_and_provenance_complete")
 ||!flag(s,"independent_reference_method_present")||!flag(s,"randomization_preregistered")
 ||!flag(s,"assessor_blinding_or_independent_adjudication")||!flag(s,"exclusion_rules_preregistered")
 ||!flag(s,"missingness_policy_preregistered")||!flag(s,"outlier_rule_preregistered")
 ||!flag(s,"deviation_log_enabled")||!flag(s,"cluster_nesting_declared")||!flag(s,"analysis_preregistered")
 ||!flag(s,"all_variance_strata_separately_reported")||!flag(s,"no_unplanned_product_revision_mixing")
 {return "MORE_EVIDENCE_REQUIRED";}
 "ELIGIBLE_FOR_PREREGISTRATION_REVIEW_ONLY"
}
fn validate_manifest(v:&Value)->Result<(),String>{
 if !exact_keys(v,&["schema_version","study_design_id","parent_issue","classification","claim_ceiling","product_scope","evidence_strata","confirmatory_holdouts","analysis_contract","comparison_contract","evidence_and_rights","design_state","planned_counts","power_analysis","protocol_digest","result"])
 ||text(v,"schema_version")!="luminous.home-service-metrology-study-design/0.1"
 ||text(v,"study_design_id")!="HOME-SVC-METROLOGY-STUDY-DESIGN-V0.1"
 ||text(v,"classification")!="requirements_template_only_not_preregistered"
 ||text(v,"claim_ceiling")!="study_design_contract_and_synthetic_gate_only"
 ||text(v,"design_state")!="NOT_PREREGISTERED"||text(v,"planned_counts")!="NOT_SELECTED"
 ||text(v,"power_analysis")!="NOT_PERFORMED"||text(v,"protocol_digest")!="NOT_CREATED"
 ||text(v,"result")!="NO_STUDY_AUTHORITY"
 {return Err("MANIFEST_IDENTITY_OR_CLAIM_CEILING_INVALID".into());}
 let strata=v["evidence_strata"].as_array().ok_or("STRATA_MISSING")?;
 let expected=["WITHIN_UNIT_REPEATABILITY","BETWEEN_LAB_REPRODUCIBILITY","PRODUCTION_UNIT_VARIATION","HOUSEHOLD_ECOLOGICAL_VALIDITY"];
 if strata.len()!=4{return Err("STRATA_COUNT_INVALID".into());}
 for (s,id) in strata.iter().zip(expected){
  if text(s,"id")!=id{return Err(format!("STRATUM_ORDER_OR_ID_INVALID:{id}"));}
  if s["required_identity"].as_array().is_none_or(|a|a.is_empty())||s["required_controls"].as_array().is_none_or(|a|a.is_empty()){return Err(format!("STRATUM_CONTRACT_INCOMPLETE:{id}"));}
 }
 if v["confirmatory_holdouts"]["reuse_for_adaptive_tuning"]!=false
 ||v["comparison_contract"]["no_single_scalar_score"]!=true
 ||v["comparison_contract"]["hard_safety_service_accessibility_and_consent_floors_noncompensatory"]!=true
 {return Err("COMPARISON_POLICY_WEAKENED".into());}
 Ok(())
}
pub fn validate_tracked()->Result<usize,String>{
 let m=parse(MANIFEST_JSON)?;let ms=parse(MANIFEST_SCHEMA_JSON)?;let c=parse(CORPUS_JSON)?;let cs=parse(CORPUS_SCHEMA_JSON)?;
 if ms["additionalProperties"]!=false||ms["type"]!="object"||cs["additionalProperties"]!=false||cs["type"]!="object"{return Err("SCHEMA_ROOT_INVALID".into());}
 if c["cases"].as_array().is_none_or(|a|a.len()!=EXPECTED_CASES.len()){return Err("CASE_COUNT_INVALID".into());}
 validate_manifest(&m)?;
 if text(&c,"corpus_id")!="HOME-SVC-METROLOGY-STUDY-CORPUS-V0.1"{return Err("CORPUS_ID_INVALID".into());}
 for(case,(id,expected))in c["cases"].as_array().unwrap().iter().zip(EXPECTED_CASES){
  if !exact_keys(case,&["case_id","description","signals","expected_disposition"])
   ||text(case,"case_id")!=*id||text(case,"expected_disposition")!=*expected
   ||!exact_keys(&case["signals"],SIGNAL_FIELDS)
  {return Err(format!("CASE_SHAPE_OR_EXPECTED_DRIFT:{id}"));}
  let actual=classify_signals(&case["signals"]);
  if actual!=*expected{return Err(format!("{id}:EXPECTED_{expected}:ACTUAL_{actual}"));}
 }
 Ok(c["cases"].as_array().unwrap().len())
}

#[cfg(test)]
mod tests{
 use super::*;
 #[test] fn tracked_36_known_answers_validate(){assert_eq!(validate_tracked().unwrap(),36);}
 #[test] fn duplicate_json_members_are_rejected(){assert!(parse(r#"{"x":1,"x":2}"#).is_err());}
 #[test] fn one_lab_cannot_claim_reproducibility(){let c=parse(CORPUS_JSON).unwrap();assert_eq!(classify_signals(&c["cases"][1]["signals"]),"NOT_COMPARABLE");}
 #[test] fn production_lot_and_unit_variance_are_separate(){let c=parse(CORPUS_JSON).unwrap();assert_eq!(classify_signals(&c["cases"][2]["signals"]),"NOT_COMPARABLE");assert_eq!(classify_signals(&c["cases"][3]["signals"]),"NOT_COMPARABLE");}
 #[test] fn household_repeat_windows_are_not_household_sample_size(){let c=parse(CORPUS_JSON).unwrap();assert_eq!(classify_signals(&c["cases"][4]["signals"]),"NOT_COMPARABLE");}
 #[test] fn no_power_analysis_means_more_evidence(){let c=parse(CORPUS_JSON).unwrap();assert_eq!(classify_signals(&c["cases"][9]["signals"]),"MORE_EVIDENCE_REQUIRED");}
 #[test] fn holdout_reuse_for_adaptation_is_rejected(){let c=parse(CORPUS_JSON).unwrap();assert_eq!(classify_signals(&c["cases"][31]["signals"]),"REJECTED_FOR_STUDY_SCOPE");}
 #[test] fn consent_and_hard_floors_are_not_compensable(){let c=parse(CORPUS_JSON).unwrap();assert_eq!(classify_signals(&c["cases"][21]["signals"]),"REJECTED_FOR_STUDY_SCOPE");assert_eq!(classify_signals(&c["cases"][24]["signals"]),"REJECTED_FOR_STUDY_SCOPE");}
 #[test] fn full_design_is_review_only_not_performance_pass(){let c=parse(CORPUS_JSON).unwrap();assert_eq!(classify_signals(&c["cases"][0]["signals"]),"ELIGIBLE_FOR_PREREGISTRATION_REVIEW_ONLY");}
 #[test] fn missingness_has_no_implicit_zero_default(){let c=parse(CORPUS_JSON).unwrap();let mut s=c["cases"][0]["signals"].clone();s["missingness_policy_preregistered"]=Value::Bool(false);assert_eq!(classify_signals(&s),"MORE_EVIDENCE_REQUIRED");}
 #[test] fn unplanned_revision_mixing_blocks_comparison(){let c=parse(CORPUS_JSON).unwrap();let mut s=c["cases"][0]["signals"].clone();s["no_unplanned_product_revision_mixing"]=Value::Bool(false);assert_eq!(classify_signals(&s),"MORE_EVIDENCE_REQUIRED");}
}
