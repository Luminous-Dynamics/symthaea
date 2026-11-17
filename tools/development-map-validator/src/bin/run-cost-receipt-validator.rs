//! D6.17.2.3 run-level cost receipt validator.
//!
//! Verifies SHA-256 receipt/event-chain integrity, evaluator-approved execution
//! context, attempt/retry chronology, and complete-trajectory cost samples.
//! Hashes provide integrity checks only; they do not authenticate a runner or
//! third-party billing provider.

use chrono::{DateTime,FixedOffset};
use serde_json::{json,Value};
use sha2::{Digest,Sha256};
use std::collections::{BTreeMap,BTreeSet};
use std::{env,fs,process};

const MAX_RUNS:usize=1000;
const MAX_ATTEMPTS:usize=4096;
const CONTEXT_FIELDS:[&str;14]=[
 "repository_full_name","branch","commit_sha","worktree_digest_sha256",
 "model_provider","model_version","model_config_digest_sha256",
 "harness_digest_sha256","toolset_digest_sha256","permissions_digest_sha256",
 "runner_class","hardware_class","runner_image_digest_sha256","region"
];
const CLAIM_CEILING:&str="Validates receipt structure, SHA-256 content/event-chain consistency, evaluator-approved execution context, chronology, retry links, and empirical quantiles over complete supplied run trajectories. Hashes do not authenticate the receipt producer or billing provider. It does not validate runner honesty, reconcile external invoices, estimate population tails from small samples, prove agent quality, authorize actions, or establish production reliability.";

#[derive(Clone,Debug)]
struct Attempt{
 event_id:String,option_id:String,attempt_no:u64,mode:String,status:String,result:String,
 queued_at:DateTime<FixedOffset>,started_at:Option<DateTime<FixedOffset>>,
 completed_at:Option<DateTime<FixedOffset>>,elapsed_ms:Option<u64>,price_micros:Option<u64>,
 event_sha256:String
}
#[derive(Clone,Debug)]
struct RunMetrics{
 run_id:String,task_family:String,paired_seed:u64,
 cost_micros:Option<u64>,wall_ms:Option<u64>,queue_delay_ms:Option<u64>,
 complete_cost:bool,complete_wall:bool,queued_attempts:u64,unknown_attempts:u64,
 retries:u64,option_costs:BTreeMap<String,Vec<u64>>
}

fn object<'a>(v:&'a Value,label:&str)->Result<&'a serde_json::Map<String,Value>,String>{
 v.as_object().ok_or_else(||format!("{label} must be an object"))
}
fn allowed(v:&Value,keys:&[&str],label:&str)->Result<(),String>{
 for key in object(v,label)?.keys(){if !keys.contains(&key.as_str()){return Err(format!("{label} contains unknown field {key:?}"));}}
 Ok(())
}
fn get<'a>(v:&'a Value,key:&str,label:&str)->Result<&'a Value,String>{
 v.get(key).ok_or_else(||format!("{label}.{key} is required"))
}
fn string(v:&Value,key:&str,label:&str)->Result<String,String>{
 get(v,key,label)?.as_str().filter(|x|!x.trim().is_empty()).map(str::to_owned)
  .ok_or_else(||format!("{label}.{key} must be a non-empty string"))
}
fn uint(v:&Value,key:&str,label:&str)->Result<u64,String>{
 get(v,key,label)?.as_u64().ok_or_else(||format!("{label}.{key} must be a non-negative integer"))
}
fn nullable_uint(v:&Value,key:&str,label:&str)->Result<Option<u64>,String>{
 match get(v,key,label)?{Value::Null=>Ok(None),Value::Number(n)=>n.as_u64().map(Some).ok_or_else(||format!("{label}.{key} must be a non-negative integer or null")),_=>Err(format!("{label}.{key} must be an integer or null"))}
}
fn nullable_string(v:&Value,key:&str,label:&str)->Result<Option<String>,String>{
 match get(v,key,label)?{Value::Null=>Ok(None),Value::String(x)=>Ok(Some(x.clone())),_=>Err(format!("{label}.{key} must be a string or null"))}
}
fn timestamp(raw:&str,label:&str)->Result<DateTime<FixedOffset>,String>{
 DateTime::parse_from_rfc3339(raw).map_err(|e|format!("{label} must be RFC 3339: {e}"))
}
fn optional_time(v:&Value,key:&str,label:&str)->Result<Option<DateTime<FixedOffset>>,String>{
 match nullable_string(v,key,label)?{None=>Ok(None),Some(raw)=>Ok(Some(timestamp(&raw,&format!("{label}.{key}"))?))}
}
fn digest_hex(raw:&str,label:&str)->Result<(),String>{
 if raw.len()!=64||!raw.chars().all(|c|c.is_ascii_hexdigit()){
  return Err(format!("{label} must be a 64-character SHA-256 hex digest"));
 }
 Ok(())
}
fn hash_bytes(bytes:&[u8])->String{
 let digest=Sha256::digest(bytes);
 digest.iter().map(|b|format!("{b:02x}")).collect()
}
fn hash_object_without(v:&Value,field:&str,label:&str)->Result<String,String>{
 let mut map=object(v,label)?.clone();
 map.remove(field);
 let canonical=serde_json::to_vec(&Value::Object(map)).map_err(|e|format!("{label} canonical JSON failed: {e}"))?;
 Ok(hash_bytes(&canonical))
}
fn checked_digest(v:&Value,field:&str,label:&str)->Result<String,String>{
 let claimed=string(v,field,label)?;
 digest_hex(&claimed,&format!("{label}.{field}"))?;
 let calculated=hash_object_without(v,field,label)?;
 if !claimed.eq_ignore_ascii_case(&calculated){
  return Err(format!("{label}.{field} does not match canonical SHA-256 content"));
 }
 Ok(calculated)
}
fn quantile(values:&[u64],percent:u64)->Option<u64>{
 if values.is_empty(){return None;}
 let mut sorted=values.to_vec();sorted.sort_unstable();
 let rank=((sorted.len() as u128*percent as u128+99)/100).max(1) as usize;
 sorted.get(rank.saturating_sub(1)).copied()
}
fn summary_quantiles(values:&[u64])->Value{
 json!({"sample_count":values.len(),"p50":quantile(values,50),"p95":quantile(values,95),"p99":quantile(values,99)})
}
fn validate_context(receipt:&Value,approved:&Value,label:&str)->Result<(),String>{
 for field in CONTEXT_FIELDS{
  let value=string(receipt,field,label)?;
  let expected=string(approved,field,"oracle.approved_context")?;
  if value!=expected{return Err(format!("{label}.{field} does not match evaluator-approved context"));}
  if field.ends_with("_digest_sha256")||field=="worktree_digest_sha256"{
   digest_hex(&value,&format!("{label}.{field}"))?;
  }
 }
 let commit=string(receipt,"commit_sha",label)?;
 if commit.len()!=40||!commit.chars().all(|c|c.is_ascii_hexdigit()){
  return Err(format!("{label}.commit_sha must be a 40-character Git SHA"));
 }
 Ok(())
}
fn validate_attempts(receipt:&Value,approved_options:&BTreeSet<String>,run_started:Option<DateTime<FixedOffset>>,run_finished:Option<DateTime<FixedOffset>>,label:&str)->Result<(Vec<Attempt>,RunMetrics),String>{
 let rows=get(receipt,"attempts",label)?.as_array().ok_or("attempts must be an array")?;
 if rows.is_empty()||rows.len()>MAX_ATTEMPTS{return Err(format!("{label}.attempts must contain 1..={MAX_ATTEMPTS} events"));}
 let mut attempts=Vec::new();let mut ids=BTreeSet::new();let mut by_id:BTreeMap<String,Attempt>=BTreeMap::new();
 let mut attempt_numbers:BTreeMap<String,u64>=BTreeMap::new();let mut option_costs:BTreeMap<String,Vec<u64>>=BTreeMap::new();
 let mut previous_hash="".to_owned();let mut previous_queue:Option<DateTime<FixedOffset>>=None;
 let mut queued_count=0u64;let mut unknown_count=0u64;let mut retry_count=0u64;
 let mut run_cost=0u64;let mut cost_complete=true;let mut wall_complete=true;
 let mut earliest_started:Option<DateTime<FixedOffset>>=None;let mut latest_completed:Option<DateTime<FixedOffset>>=None;
 for (index,row) in rows.iter().enumerate(){
  let l=format!("{label}.attempts[{index}]");
  allowed(row,&[
   "event_id","sequence","option_id","attempt_no","mode","status","result","queued_at",
   "started_at","completed_at","elapsed_ms","cpu_ms","gpu_ms","network_bytes","price_micros",
   "retry_of_event_id","evidence_digest_sha256","billing_record_ref","billing_record_sha256",
   "previous_event_sha256","event_sha256"
  ],&l)?;
  let event_id=string(row,"event_id",&l)?;
  if !ids.insert(event_id.clone()){return Err(format!("{label} has duplicate event ID {event_id:?}"));}
  if uint(row,"sequence",&l)?!=index as u64{return Err(format!("{l}.sequence must be contiguous and zero-based"));}
  let option=string(row,"option_id",&l)?;
  if !approved_options.contains(&option){return Err(format!("{l} references unapproved option {option:?}"));}
  let attempt_no=uint(row,"attempt_no",&l)?;
  let expected_no=attempt_numbers.get(&option).copied().unwrap_or(0)+1;
  if attempt_no!=expected_no{return Err(format!("{l}.attempt_no must be contiguous for option {option:?}"));}
  attempt_numbers.insert(option.clone(),attempt_no);
  let mode=string(row,"mode",&l)?;let status=string(row,"status",&l)?;let result=string(row,"result",&l)?;
  if !["cold","warm","retry"].contains(&mode.as_str()){return Err(format!("{l}.mode is invalid"));}
  if !["completed","failed","timeout","cancelled","skipped","queued","unknown"].contains(&status.as_str()){
   return Err(format!("{l}.status is invalid"));
  }
  if !["pass","fail","unknown","not_applicable"].contains(&result.as_str()){return Err(format!("{l}.result is invalid"));}
  let queued_at=timestamp(&string(row,"queued_at",&l)?,&format!("{l}.queued_at"))?;
  if previous_queue.is_some_and(|previous|queued_at<previous){return Err(format!("{label} attempt queue times regress"));}
  previous_queue=Some(queued_at);
  let started_at=optional_time(row,"started_at",&l)?;
  let completed_at=optional_time(row,"completed_at",&l)?;
  let elapsed_ms=nullable_uint(row,"elapsed_ms",&l)?;
  let price_micros=nullable_uint(row,"price_micros",&l)?;
  let cpu=uint(row,"cpu_ms",&l)?;let gpu=uint(row,"gpu_ms",&l)?;let network=uint(row,"network_bytes",&l)?;
  if let Some(start)=started_at{
   if start<queued_at{return Err(format!("{l}.started_at precedes queued_at"));}
   if run_started.is_some_and(|x|start<x){return Err(format!("{l}.started_at precedes run_started_at"));}
   earliest_started=Some(earliest_started.map_or(start,|x|x.min(start)));
  }
  if let Some(end)=completed_at{
   if started_at.is_none_or(|start|end<start){return Err(format!("{l}.completed_at requires a nondecreasing started_at"));}
   if run_finished.is_some_and(|x|end>x){return Err(format!("{l}.completed_at follows run_finished_at"));}
   latest_completed=Some(latest_completed.map_or(end,|x|x.max(end)));
   let actual=(end-started_at.unwrap_or(end)).num_milliseconds().max(0) as u64;
   if elapsed_ms!=Some(actual){return Err(format!("{l}.elapsed_ms does not match the timestamp interval"));}
  }else if elapsed_ms.is_some(){return Err(format!("{l}.elapsed_ms must be null when completed_at is null"));}
  let retry_of=nullable_string(row,"retry_of_event_id",&l)?;
  if attempt_no==1{
   if mode=="retry"||retry_of.is_some(){return Err(format!("{l} first attempt cannot be a retry"));}
  }else{
   if mode!="retry"{return Err(format!("{l} subsequent attempt must use retry mode"));}
   let parent_id=retry_of.as_ref().ok_or_else(||format!("{l} retry requires retry_of_event_id"))?;
   let parent=by_id.get(parent_id).ok_or_else(||format!("{l} retry parent must be a prior event"))?;
   if parent.option_id!=option||parent.status!="failed"||parent.attempt_no+1!=attempt_no{
    return Err(format!("{l} may retry only the immediately preceding definitive failure for the same option"));
   }
   retry_count+=1;
  }
  if mode!="retry"&&retry_of.is_some(){return Err(format!("{l} non-retry event cannot set retry_of_event_id"));}
  match status.as_str(){
   "completed"=>{
    if started_at.is_none()||completed_at.is_none()||!matches!(result.as_str(),"pass"|"fail"){
     return Err(format!("{l} completed status requires start/end times and a pass/fail result"));
    }
   },
   "failed"=>{
    if started_at.is_none()||completed_at.is_none()||result!="unknown"{
     return Err(format!("{l} failed status requires completed execution and unknown result"));
    }
   },
   "timeout"=>{
    if started_at.is_none()||result!="unknown"{return Err(format!("{l} timeout requires started execution and unknown result"));}
    if completed_at.is_none(){wall_complete=false;}
   },
   "cancelled"=>{
    if started_at.is_none()||result!="unknown"{return Err(format!("{l} cancelled status requires started execution and unknown result"));}
    if completed_at.is_none(){wall_complete=false;}
   },
   "skipped"=>{
    if started_at.is_some()||completed_at.is_some()||elapsed_ms.is_some()||price_micros!=Some(0)||cpu!=0||gpu!=0||network!=0||result!="not_applicable"{
     return Err(format!("{l} skipped status must have no execution and zero cost"));
    }
   },
   "queued"=>{
    if started_at.is_some()||completed_at.is_some()||elapsed_ms.is_some()||price_micros.is_some()||cpu!=0||gpu!=0||network!=0||result!="not_applicable"{
     return Err(format!("{l} queued status must have no execution and unknown cost"));
    }
    queued_count+=1;cost_complete=false;wall_complete=false;
   },
   "unknown"=>{
    if result!="unknown"{return Err(format!("{l} unknown status requires unknown result"));}
    unknown_count+=1;cost_complete=false;
    if started_at.is_none()||completed_at.is_none(){wall_complete=false;}
   },
   _=>return Err("unreachable status".to_owned())
  }
  if status!="skipped"&&status!="queued"{
   if started_at.is_none(){return Err(format!("{l} executed attempt requires started_at"));}
   if price_micros.is_none(){cost_complete=false;}
   if let Some(cost)=price_micros{
    run_cost=run_cost.checked_add(cost).ok_or_else(||format!("{label} trajectory cost overflow"))?;
    if cost>0{
     let _=string(row,"billing_record_ref",&l)?;
     digest_hex(&string(row,"billing_record_sha256",&l)?,&format!("{l}.billing_record_sha256"))?;
    }
    if ["completed","failed","timeout","cancelled"].contains(&status.as_str()){
     option_costs.entry(option.clone()).or_default().push(cost);
    }
   }
   if let Some(evidence)=nullable_string(row,"evidence_digest_sha256",&l)?{digest_hex(&evidence,&format!("{l}.evidence_digest_sha256"))?;}
   if matches!(result.as_str(),"pass"|"fail"){
    let evidence=nullable_string(row,"evidence_digest_sha256",&l)?;
    if evidence.is_none(){return Err(format!("{l} pass/fail result requires evidence digest"));}
   }
  }
  let prev=nullable_string(row,"previous_event_sha256",&l)?;
  if index==0{
   if prev.is_some(){return Err(format!("{l} first event previous_event_sha256 must be null"));}
  }else if prev.as_deref()!=Some(previous_hash.as_str()){
   return Err(format!("{l} event hash chain predecessor mismatch"));
  }
  let event_hash=checked_digest(row,"event_sha256",&l)?;
  previous_hash=event_hash.clone();
  let attempt=Attempt{event_id:event_id.clone(),option_id:option,attempt_no,mode,status,result,queued_at,started_at,completed_at,elapsed_ms,price_micros,event_sha256:event_hash};
  by_id.insert(event_id,attempt.clone());attempts.push(attempt);
 }
 if let Some(run_start)=run_started{
  if earliest_started.is_some_and(|x|x<run_start){return Err(format!("{label} run_started_at follows an attempt start"));}
 }else if earliest_started.is_some(){return Err(format!("{label} attempts started but run_started_at is null"));}
 if let Some(run_end)=run_finished{
  if latest_completed.is_some_and(|x|x>run_end){return Err(format!("{label} run_finished_at precedes an attempt completion"));}
 }else{cost_complete=false;wall_complete=false;}
 let cost=if cost_complete{Some(run_cost)}else{None};
 let wall=match(run_started,run_finished){
  (Some(start),Some(end)) if wall_complete&&end>=start=>Some((end-start).num_milliseconds() as u64),
  _=>None
 };
 let queue=match(run_started,Some(queued_at_for_receipt(receipt_label_placeholder()))){
  _=>None
 };
 let _=queue;
 let _=attempts;
 unreachable!()
}
fn receipt_label_placeholder()->DateTime<FixedOffset>{
 DateTime::parse_from_rfc3339("2000-01-01T00:00:00Z").expect("constant timestamp")
}
fn evaluate(raw:&str)->Result<Value,String>{
 let bundle:Value=serde_json::from_str(raw).map_err(|e|format!("invalid JSON: {e}"))?;
 allowed(&bundle,&["candidate_view","oracle"],"bundle")?;
 let candidate=get(&bundle,"candidate_view","bundle")?;
 allowed(candidate,&["run_receipts"],"candidate_view")?;
 let oracle=get(&bundle,"oracle","bundle")?;
 allowed(oracle,&["approved_context","approved_option_ids"],"oracle")?;
 let context=get(oracle,"approved_context","oracle")?;
 allowed(context,&CONTEXT_FIELDS,"oracle.approved_context")?;
 let approved_options=get(oracle,"approved_option_ids","oracle")?.as_array().ok_or("oracle.approved_option_ids must be an array")?;
 let mut option_ids=BTreeSet::new();
 for option in approved_options{
  let value=option.as_str().filter(|x|!x.trim().is_empty()).ok_or("approved option ID must be non-empty string")?;
  if !option_ids.insert(value.to_owned()){return Err(format!("duplicate approved option {value:?}"));}
 }
 if option_ids.is_empty(){return Err("oracle must approve at least one option".to_owned());}
 let receipts=get(candidate,"run_receipts","candidate_view")?.as_array().ok_or("candidate_view.run_receipts must be array")?;
 if receipts.is_empty()||receipts.len()>MAX_RUNS{return Err(format!("run_receipts must contain 1..={MAX_RUNS} receipts"));}
 let mut metrics=Vec::new();let mut run_ids=BTreeSet::new();
 for (index,receipt) in receipts.iter().enumerate(){
  let label=format!("run_receipts[{index}]");
  allowed(receipt,&[
   "schema_version","run_id","task_family","paired_seed",
   "repository_full_name","branch","commit_sha","worktree_digest_sha256",
   "model_provider","model_version","model_config_digest_sha256",
   "harness_digest_sha256","toolset_digest_sha256","permissions_digest_sha256",
   "runner_class","hardware_class","runner_image_digest_sha256","region",
   "runner_identity","runner_attestation_ref","runner_attestation_digest_sha256",
   "queued_at","run_started_at","run_finished_at","attempts","receipt_sha256"
  ],&label)?;
  if string(receipt,"schema_version",&label)?!="1.0"{return Err(format!("{label}.schema_version unsupported"));}
  let run_id=string(receipt,"run_id",&label)?;
  if !run_ids.insert(run_id.clone()){return Err(format!("duplicate run_id {run_id:?}"));}
  let task_family=string(receipt,"task_family",&label)?;let paired_seed=uint(receipt,"paired_seed",&label)?;
  validate_context(receipt,context,&label)?;
  let _runner=string(receipt,"runner_identity",&label)?;
  let _attest=string(receipt,"runner_attestation_ref",&label)?;
  digest_hex(&string(receipt,"runner_attestation_digest_sha256",&label)?,&format!("{label}.runner_attestation_digest_sha256"))?;
  let queued_at=timestamp(&string(receipt,"queued_at",&label)?,&format!("{label}.queued_at"))?;
  let started_at=optional_time(receipt,"run_started_at",&label)?;
  let finished_at=optional_time(receipt,"run_finished_at",&label)?;
  if started_at.is_some_and(|x|x<queued_at){return Err(format!("{label}.run_started_at precedes queued_at"));}
  if finished_at.is_some_and(|x|started_at.is_none_or(|start|x<start)){return Err(format!("{label}.run_finished_at requires nondecreasing run_started_at"));}
  let (attempts,mut m)=validate_attempts(receipt,&option_ids,started_at,finished_at,&label)?;
  let calculated=checked_digest(receipt,"receipt_sha256",&label)?;
  let _=calculated;
  let queue_delay=started_at.map(|x|(x-queued_at).num_milliseconds().max(0) as u64);
  m.run_id=run_id;m.task_family=task_family;m.paired_seed=paired_seed;m.queue_delay_ms=queue_delay;
  let _=attempts;
  metrics.push(m);
 }
 let complete_cost:Vec<u64>=metrics.iter().filter_map(|m|m.cost_micros).collect();
 let complete_wall:Vec<u64>=metrics.iter().filter_map(|m|m.wall_ms).collect();
 let queue:Vec<u64>=metrics.iter().filter_map(|m|m.queue_delay_ms).collect();
 let mut option_costs:BTreeMap<String,Vec<u64>>=BTreeMap::new();
 for option in &option_ids{option_costs.insert(option.clone(),Vec::new());}
 let mut retries=0u64;let mut queued=0u64;let mut unknown=0u64;
 for m in &metrics{
  retries+=m.retries;queued+=m.queued_attempts;unknown+=m.unknown_attempts;
  for (option,values) in &m.option_costs{option_costs.entry(option.clone()).or_default().extend(values);}
 }
 let mut marginal_p95=Vec::new();let mut marginal_sum=0u64;let mut all_options_sampled=true;
 for (option,values) in &option_costs{
  let p95=quantile(values,95);
  if let Some(value)=p95{marginal_sum=marginal_sum.checked_add(value).ok_or("marginal p95 sum overflow")?;}
  else{all_options_sampled=false;}
  marginal_p95.push(json!({"option_id":option,"sample_count":values.len(),"p95_cost_micros":p95}));
 }
 Ok(json!({
  "adjudication_status":"synthetic_known_answer_run_level_cost_receipt_validation",
  "run_count":metrics.len(),
  "complete_cost_run_count":complete_cost.len(),
  "complete_wall_run_count":complete_wall.len(),
  "censored_or_incomplete_run_count":metrics.len().saturating_sub(complete_cost.len()),
  "queued_attempt_count":queued,
  "unknown_execution_count":unknown,
  "retry_count":retries,
  "full_run_cost_micros":summary_quantiles(&complete_cost),
  "full_run_wall_ms":summary_quantiles(&complete_wall),
  "queue_delay_ms":summary_quantiles(&queue),
  "marginal_option_cost_p95_micros":marginal_p95,
  "sum_of_marginal_option_p95_micros":if all_options_sampled{Value::from(marginal_sum)}else{Value::Null},
  "tail_quantile_warning":"The sum of marginal per-option p95 values is a descriptive comparator, not generally a p95 guarantee for the sum of correlated run costs. Full-run p50/p95/p99 are computed directly from complete trajectory receipts.",
  "run_samples":metrics.iter().map(|m|json!({
   "run_id":m.run_id,"task_family":m.task_family,"paired_seed":m.paired_seed,
   "full_run_cost_micros":m.cost_micros,"full_run_wall_ms":m.wall_ms,
   "queue_delay_ms":m.queue_delay_ms,"cost_complete":m.complete_cost,
   "wall_complete":m.complete_wall
  })).collect::<Vec<_>>(),
  "claim_ceiling":CLAIM_CEILING
 }))
}
pub fn validate_run_cost_receipts_json(raw:&str)->Result<Value,String>{evaluate(raw)}
fn main(){
 let args:Vec<String>=env::args().skip(1).collect();
 if args.len()!=1||args[0]=="--help"||args[0]=="-h"{
  eprintln!("Usage: run-cost-receipt-validator <receipt-bundle.json>");
  process::exit(if args.len()==1{0}else{2});
 }
 let raw=match fs::read_to_string(&args[0]){Ok(x)=>x,Err(e)=>{eprintln!("ERROR D6.17.2.3-IO-001 {}: {e}",args[0]);process::exit(2)}};
 match evaluate(&raw){
  Ok(report)=>match serde_json::to_string_pretty(&report){Ok(text)=>println!("{text}"),Err(e)=>{eprintln!("ERROR D6.17.2.3-JSON-001: {e}");process::exit(2)}},
  Err(e)=>{eprintln!("ERROR D6.17.2.3-EVALUATION-001: {e}");process::exit(1)}
 }
}
