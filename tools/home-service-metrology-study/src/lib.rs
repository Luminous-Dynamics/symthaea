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
("HSM-037","MORE_EVIDENCE_REQUIRED"),
("HSM-038","MORE_EVIDENCE_REQUIRED"),
("HSM-039","NOT_COMPARABLE"),
("HSM-040","NOT_COMPARABLE"),
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
"analysis_preregistered","all_variance_strata_separately_reported","no_unplanned_product_revision_mixing",
"laboratory_scope_schedule_current","proficiency_assessment_applicability_resolved",
"proficiency_evidence_not_substituted_for_reproducibility","method_edition_current"
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
 if !flag(s,"proficiency_evidence_not_substituted_for_reproducibility")||!flag(s,"method_edition_current")
 {return "NOT_COMPARABLE";}
 if !flag(s,"laboratory_scope_schedule_current")||!flag(s,"proficiency_assessment_applicability_resolved")
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

fn string_array_equals(value:&Value,expected:&[&str])->bool{
 value.as_array().is_some_and(|a|a.len()==expected.len()&&a.iter().zip(expected).all(|(v,e)|v.as_str()==Some(*e)))
}
fn schema_type_matches(value:&Value,kind:&str)->bool{
 match kind{
  "object"=>value.is_object(),"array"=>value.is_array(),"string"=>value.is_string(),
  "boolean"=>value.is_boolean(),"integer"=>value.as_i64().is_some()||value.as_u64().is_some(),
  "number"=>value.as_f64().is_some(),"null"=>value.is_null(),_=>false
 }
}
fn schema_pointer<'a>(root:&'a Value,reference:&str)->Option<&'a Value>{
 let path=reference.strip_prefix("#/")?;
 let mut value=root;
 for part in path.split('/') {
  let key=part.replace("~1","/").replace("~0","~");
  value=value.get(&key)?;
 }
 Some(value)
}
fn validate_json_schema(value:&Value,schema:&Value,root:&Value,path:&str,errors:&mut Vec<String>,depth:usize){
 if depth>64{errors.push(format!("{path}:SCHEMA_DEPTH_EXCEEDED"));return;}
 if let Some(reference)=schema.get("$ref").and_then(Value::as_str){
  if let Some(target)=schema_pointer(root,reference){validate_json_schema(value,target,root,path,errors,depth+1);}
  else{errors.push(format!("{path}:SCHEMA_REF_UNRESOLVED:{reference}"));}
  return;
 }
 if let Some(expected)=schema.get("const"){if value!=expected{errors.push(format!("{path}:SCHEMA_CONST"));}}
 if let Some(allowed)=schema.get("enum").and_then(Value::as_array){
  if !allowed.iter().any(|candidate|candidate==value){errors.push(format!("{path}:SCHEMA_ENUM"));}
 }
 if let Some(kind)=schema.get("type").and_then(Value::as_str){
  if !schema_type_matches(value,kind){errors.push(format!("{path}:SCHEMA_TYPE:{kind}"));return;}
 }
 if let Some(branches)=schema.get("oneOf").and_then(Value::as_array){
  let mut matches=0usize;
  for branch in branches{
   let mut branch_errors=Vec::new();
   validate_json_schema(value,branch,root,path,&mut branch_errors,depth+1);
   if branch_errors.is_empty(){matches+=1;}
  }
  if matches!=1{errors.push(format!("{path}:SCHEMA_ONE_OF_MATCH_COUNT:{matches}"));}
 }
 if let Some(object)=value.as_object(){
  if let Some(required)=schema.get("required").and_then(Value::as_array){
   for key in required.iter().filter_map(Value::as_str){
    if !object.contains_key(key){errors.push(format!("{path}:SCHEMA_REQUIRED:{key}"));}
   }
  }
  if schema.get("additionalProperties")==Some(&Value::Bool(false)){
   if let Some(properties)=schema.get("properties").and_then(Value::as_object){
    for key in object.keys(){if !properties.contains_key(key){errors.push(format!("{path}:SCHEMA_ADDITIONAL_PROPERTY:{key}"));}}
   }
  }
  if let Some(properties)=schema.get("properties").and_then(Value::as_object){
   for (key,subschema) in properties{
    if let Some(child)=object.get(key){validate_json_schema(child,subschema,root,&format!("{path}.{key}"),errors,depth+1);}
   }
  }
 }
 if let Some(array)=value.as_array(){
  if let Some(minimum)=schema.get("minItems").and_then(Value::as_u64){if array.len()<(minimum as usize){errors.push(format!("{path}:SCHEMA_MIN_ITEMS"));}}
  if let Some(maximum)=schema.get("maxItems").and_then(Value::as_u64){if array.len()>(maximum as usize){errors.push(format!("{path}:SCHEMA_MAX_ITEMS"));}}
  if schema.get("uniqueItems")==Some(&Value::Bool(true)){
   for i in 0..array.len(){if array[..i].iter().any(|prior|prior==&array[i]){errors.push(format!("{path}:SCHEMA_UNIQUE_ITEMS"));break;}}
  }
  if let Some(items)=schema.get("items"){for (index,item) in array.iter().enumerate(){validate_json_schema(item,items,root,&format!("{path}[{index}]"),errors,depth+1);}}
 }
 if let Some(string)=value.as_str(){
  if let Some(minimum)=schema.get("minLength").and_then(Value::as_u64){if string.chars().count()<(minimum as usize){errors.push(format!("{path}:SCHEMA_MIN_LENGTH"));}}
  if let Some(pattern)=schema.get("pattern").and_then(Value::as_str){
   let matched=match pattern{
    "^HSM-[0-9]{3}$"=>string.len()==7&&string.starts_with("HSM-")&&string[4..].bytes().all(|b|b.is_ascii_digit()),
    _=>true
   };
   if !matched{errors.push(format!("{path}:SCHEMA_PATTERN"));}
  }
 }
 if let Some(number)=value.as_f64(){
  if let Some(minimum)=schema.get("minimum").and_then(Value::as_f64){if number<minimum{errors.push(format!("{path}:SCHEMA_MINIMUM"));}}
  if let Some(maximum)=schema.get("maximum").and_then(Value::as_f64){if number>maximum{errors.push(format!("{path}:SCHEMA_MAXIMUM"));}}
 }
}

fn validate_manifest(v:&Value)->Result<(),String>{
 let root_fields:&[&str]=&["schema_version","study_design_id","parent_issue","classification","claim_ceiling","product_scope","evidence_strata","confirmatory_holdouts","analysis_contract","comparison_contract","evidence_and_rights","laboratory_competence_contract","design_state","planned_counts","power_analysis","protocol_digest","result"];
 if !exact_keys(v,root_fields)
 ||text(v,"schema_version")!="luminous.home-service-metrology-study-design/0.1"
 ||text(v,"study_design_id")!="HOME-SVC-METROLOGY-STUDY-DESIGN-V0.1"
 ||text(v,"parent_issue")!="https://github.com/Luminous-Dynamics/symthaea/issues/10341"
 ||text(v,"classification")!="requirements_template_only_not_preregistered"
 ||text(v,"claim_ceiling")!="study_design_contract_and_synthetic_gate_only"
 ||text(v,"design_state")!="NOT_PREREGISTERED"
 ||text(v,"planned_counts")!="NOT_SELECTED"
 ||text(v,"power_analysis")!="NOT_PERFORMED"
 ||text(v,"protocol_digest")!="NOT_CREATED"
 ||text(v,"result")!="NO_STUDY_AUTHORITY"
 {return Err("MANIFEST_IDENTITY_OR_CLAIM_CEILING_INVALID".into());}
 if !string_array_equals(&v["product_scope"],&["InstantClean","FabricCare","WaterLoop","FoodSteward","HeatBank","AirCare","HomeRenew"]){
  return Err("PRODUCT_SCOPE_INVALID".into());
 }
 let strata=v["evidence_strata"].as_array().ok_or("STRATA_MISSING")?;
 if strata.len()!=4{return Err("STRATA_COUNT_INVALID".into());}
 let expected_strata=[
  ("WITHIN_UNIT_REPEATABILITY","run_within_unit_lab_operator_cell","minimum_replicates_per_cell",2u64,
   &["unit_revision","laboratory_id","operator_id","protocol_digest"][..],
   &["randomized_run_order","conditioning_state","current_calibration"][..]),
  ("BETWEEN_LAB_REPRODUCIBILITY","independent_laboratory","minimum_independent_labs",2u64,
   &["laboratory_id","protocol_digest","reference_artifact_id","calibration_scope_digest"][..],
   &["round_robin_or_equivalent","same_method_revision","lab_scope_applicability"][..]),
  ("PRODUCTION_UNIT_VARIATION","production_unit_within_lot","minimum_production_lots",2u64,
   &["unit_serial","production_lot","product_revision","critical_supplier_revisions"][..],
   &["production_representative_sampling","supplier_substitution_tracking"][..]),
  ("HOUSEHOLD_ECOLOGICAL_VALIDITY","household_site_period","minimum_households",2u64,
   &["household_site_id","service_boundary","duty_cycle_profile","environmental_context"][..],
   &["consent_and_withdrawal","privacy_minimization","maintenance_and_operator_burden"][..]),
 ];
 for(index,(item,(id,unit,minimum_field,minimum,identity,controls)))in strata.iter().zip(expected_strata).enumerate(){
  if text(item,"id")!=id||text(item,"unit_of_analysis")!=unit
   ||item[minimum_field].as_u64()!=Some(minimum)
   ||!string_array_equals(&item["required_identity"],identity)
   ||!string_array_equals(&item["required_controls"],controls)
  {return Err(format!("STRATUM_CONTRACT_DRIFT:{index}:{id}"));}
  if id=="PRODUCTION_UNIT_VARIATION"&&item["minimum_units_per_lot"].as_u64()!=Some(2){
   return Err("PRODUCTION_UNITS_PER_LOT_MINIMUM_INVALID".into());
  }
  if id=="HOUSEHOLD_ECOLOGICAL_VALIDITY"&&item["minimum_duty_cycle_profiles"].as_u64()!=Some(2){
   return Err("HOUSEHOLD_DUTY_CYCLE_MINIMUM_INVALID".into());
  }
  let expected_keys=match id{
   "WITHIN_UNIT_REPEATABILITY"=>&["id","unit_of_analysis","minimum_replicates_per_cell","required_identity","required_controls"][..],
   "BETWEEN_LAB_REPRODUCIBILITY"=>&["id","unit_of_analysis","minimum_independent_labs","required_identity","required_controls"][..],
   "PRODUCTION_UNIT_VARIATION"=>&["id","unit_of_analysis","minimum_production_lots","minimum_units_per_lot","required_identity","required_controls"][..],
   _=>&["id","unit_of_analysis","minimum_households","minimum_duty_cycle_profiles","required_identity","required_controls"][..],
  };
  if !exact_keys(item,expected_keys){return Err(format!("STRATUM_FIELDS_INVALID:{id}"));}
 }
 let holdouts=&v["confirmatory_holdouts"];
 if !exact_keys(holdouts,&["independent_laboratories_required","production_lots_required","households_or_sites_required","reuse_for_adaptive_tuning"])
  ||holdouts["independent_laboratories_required"].as_u64()!=Some(1)
  ||holdouts["production_lots_required"].as_u64()!=Some(1)
  ||holdouts["households_or_sites_required"].as_u64()!=Some(1)
  ||holdouts["reuse_for_adaptive_tuning"]!=false
 {return Err("CONFIRMATORY_HOLDOUT_POLICY_INVALID".into());}
 let analysis=&v["analysis_contract"];
 let analysis_fields:&[&str]=&["sample_size_policy","model_family","required_variance_components","fixed_effects_required","cluster_nesting_required","randomization_required","assessor_blinding_or_independent_adjudication_required","exclusion_rules_preregistered","missingness_and_denominator_policy_preregistered","outlier_rule_preregistered","deviation_log_required","independent_analysis_code_digest_required"];
 if !exact_keys(analysis,analysis_fields)
  ||text(analysis,"sample_size_policy")!="preregistered_power_or_precision_analysis_from_pilot_variance_components"
  ||text(analysis,"model_family")!="hierarchical_or_cluster_aware"
  ||!string_array_equals(&analysis["required_variance_components"],&["within_unit_repeatability","between_lab_reproducibility","between_unit_or_lot_variation","household_or_site_variation"])
  ||!string_array_equals(&analysis["fixed_effects_required"],&["product_or_architecture","protocol_revision","declared_operating_context"])
  ||analysis.as_object().is_some_and(|o|analysis_fields.iter().any(|k|!o.get(*k).is_some_and(|x|x.as_bool()==Some(true)||x.is_string()||x.is_array())))
 {return Err("ANALYSIS_CONTRACT_INVALID".into());}
 for key in &analysis_fields[4..]{
  if analysis[*key]!=true{return Err(format!("ANALYSIS_REQUIREMENT_WEAKENED:{key}"));}
 }
 let comparison=&v["comparison_contract"];
 let comparison_fields:&[&str]=&["same_protocol_digest_across_arms","exact_product_model_and_revision_pinned","same_exogenous_trace_and_conditions","same_primary_endpoint_and_functional_unit","compute_human_time_and_adaptation_budgets_frozen","holdout_not_used_for_tuning","hard_safety_service_accessibility_and_consent_floors_noncompensatory","no_single_scalar_score"];
 if !exact_keys(comparison,comparison_fields)||comparison.as_object().is_none_or(|o|o.values().any(|x|x.as_bool()!=Some(true))){
  return Err("COMPARISON_POLICY_WEAKENED".into());
 }
 let rights=&v["evidence_and_rights"];
 let rights_fields:&[&str]=&["current_instrument_calibration_and_method_scope_required","source_payload_and_sample_lineage_required","participant_consent_and_withdrawal_required","purpose_bound_data_use_required","right_to_withdraw_and_correct_required"];
 if !exact_keys(rights,rights_fields)||rights.as_object().is_none_or(|o|o.values().any(|x|x.as_bool()!=Some(true))){
  return Err("EVIDENCE_RIGHTS_POLICY_WEAKENED".into());
 }
 let lab=&v["laboratory_competence_contract"];
 let lab_fields:&[&str]=&["scope_schedule_per_laboratory_required","scope_must_cover_exact_measurand_method_range","proficiency_assessment_reference","proficiency_assessment_may_be_not_applicable_with_justification","proficiency_is_not_reproducibility_substitute","withdrawn_method_edition_forbidden"];
 if !exact_keys(lab,lab_fields)
  ||lab["scope_schedule_per_laboratory_required"]!=true
  ||lab["scope_must_cover_exact_measurand_method_range"]!=true
  ||text(lab,"proficiency_assessment_reference")!="ISO/IEC 17043:2023 + ISO 13528:2022/Amd 1:2026"
  ||lab["proficiency_assessment_may_be_not_applicable_with_justification"]!=true
  ||lab["proficiency_is_not_reproducibility_substitute"]!=true
  ||lab["withdrawn_method_edition_forbidden"]!=true
 {return Err("LABORATORY_COMPETENCE_CONTRACT_INVALID".into());}
 Ok(())
}


fn validate_corpus(c:&Value)->Result<usize,String>{
 if !exact_keys(c,&["schema_version","corpus_id","classification","claim_ceiling","cases"])
  ||text(c,"schema_version")!="luminous.home-service-metrology-study-corpus/0.1"
  ||text(c,"corpus_id")!="HOME-SVC-METROLOGY-STUDY-CORPUS-V0.1"
  ||text(c,"classification")!="synthetic_preregistration_gate_cases"
  ||text(c,"claim_ceiling")!="study_design_gate_only_not_evidence_of_product_performance"
 {return Err("CORPUS_IDENTITY_INVALID".into());}
 let cases=c["cases"].as_array().ok_or("CASES_MISSING")?;
 if cases.len()!=EXPECTED_CASES.len(){return Err("CASE_COUNT_INVALID".into());}
 let mut seen=BTreeSet::new();
 for(case,(id,expected))in cases.iter().zip(EXPECTED_CASES){
  if !exact_keys(case,&["case_id","description","signals","expected_disposition"])
   ||text(case,"case_id")!=*id||!seen.insert(id)
   ||text(case,"description").chars().count()<20
   ||text(case,"expected_disposition")!=*expected
   ||!exact_keys(&case["signals"],SIGNAL_FIELDS)
  {return Err(format!("CASE_SHAPE_OR_EXPECTED_DRIFT:{id}"));}
  let actual=classify_signals(&case["signals"]);
  if actual!=*expected{return Err(format!("{id}:EXPECTED_{expected}:ACTUAL_{actual}"));}
 }
 Ok(cases.len())
}
pub fn validate_tracked()->Result<usize,String>{
 let m=parse(MANIFEST_JSON)?;
 let ms=parse(MANIFEST_SCHEMA_JSON)?;
 let c=parse(CORPUS_JSON)?;
 let cs=parse(CORPUS_SCHEMA_JSON)?;
 let mut errors=Vec::new();
 validate_json_schema(&m,&ms,&ms,"$manifest",&mut errors,0);
 validate_json_schema(&c,&cs,&cs,"$corpus",&mut errors,0);
 if !errors.is_empty(){errors.sort();return Err(format!("STUDY_SCHEMA_VALIDATION_FAILED:{}",errors.join("|")));}
 validate_manifest(&m)?;
 validate_corpus(&c)
}

#[cfg(test)]
mod tests{
 use super::*;

 #[test] fn both_tracked_json_schemas_are_applied(){assert_eq!(validate_tracked().unwrap(),36);}
 #[test] fn stratum_specific_minimum_cannot_be_removed(){let mut m=parse(MANIFEST_JSON).unwrap();m["evidence_strata"][2]["minimum_units_per_lot"]=Value::from(1);assert!(validate_manifest(&m).is_err());}
 #[test] fn analysis_contract_cannot_omit_variance_components(){let mut m=parse(MANIFEST_JSON).unwrap();m["analysis_contract"]["required_variance_components"].as_array_mut().unwrap().pop();assert!(validate_manifest(&m).is_err());}
 #[test] fn comparison_contract_cannot_enable_a_scalar_score(){let mut m=parse(MANIFEST_JSON).unwrap();m["comparison_contract"]["no_single_scalar_score"]=Value::Bool(false);assert!(validate_manifest(&m).is_err());}
 #[test] fn unknown_study_signal_fields_are_rejected(){let mut c=parse(CORPUS_JSON).unwrap();c["cases"][0]["signals"]["posthoc_score"]=Value::from(4);assert!(validate_corpus(&c).is_err());}
 #[test] fn a_holdout_used_for_tuning_is_not_confirmatory(){let c=parse(CORPUS_JSON).unwrap();let mut s=c["cases"][0]["signals"].clone();s["holdout_not_used_for_tuning"]=Value::Bool(false);assert_eq!(classify_signals(&s),"REJECTED_FOR_STUDY_SCOPE");}
 #[test] fn exact_lab_scope_schedule_is_required(){let c=parse(CORPUS_JSON).unwrap();let mut s=c["cases"][0]["signals"].clone();s["laboratory_scope_schedule_current"]=Value::Bool(false);assert_eq!(classify_signals(&s),"MORE_EVIDENCE_REQUIRED");}
 #[test] fn proficiency_testing_is_not_reproducibility(){let c=parse(CORPUS_JSON).unwrap();let mut s=c["cases"][0]["signals"].clone();s["proficiency_evidence_not_substituted_for_reproducibility"]=Value::Bool(false);assert_eq!(classify_signals(&s),"NOT_COMPARABLE");}
 #[test] fn withdrawn_interlaboratory_method_edition_is_not_current(){let c=parse(CORPUS_JSON).unwrap();let mut s=c["cases"][0]["signals"].clone();s["method_edition_current"]=Value::Bool(false);assert_eq!(classify_signals(&s),"NOT_COMPARABLE");}
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
