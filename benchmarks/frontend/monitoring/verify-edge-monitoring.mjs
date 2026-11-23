#!/usr/bin/env node
// SFEB-EDGE-001: durable edge-trigger monitor oracle.
// Claimed receipt IDs in notifications are hints only; action authority comes from
// the latest source-of-record snapshot and its durable, bounded, revocable receipt.
import fs from 'node:fs';
import crypto from 'node:crypto';

const DEFAULT_FIXTURE='benchmarks/frontend/monitoring/edge-monitoring-fixture-v1.json';
const DEFAULT_TRACES='benchmarks/frontend/monitoring/edge-monitoring-traces-v1.json';
const DEFAULT_POLICY='benchmarks/frontend/monitoring/edge-monitoring-policy-v1.json';
const PINNED_FIXTURE_SHA256='75a3b305e5b54716a29a4b994d2ff5aab65a1b1de86e75ba27a6005bf7c83784';
const PINNED_POLICY_SHA256='73112c7c2106663893774e7babb7863ba4cd05aad232d090818f237fd9e89bc2';

const sha256=bytes=>crypto.createHash('sha256').update(bytes).digest('hex');
const clone=value=>JSON.parse(JSON.stringify(value));
function fail(message){throw new Error(message);}
function assertKeys(value,allowed,required,label){
 if(!value||typeof value!=='object'||Array.isArray(value))fail(label+' must be an object');
 for(const key of Object.keys(value))if(!allowed.includes(key))fail(label+' has unknown field: '+key);
 for(const key of required)if(!Object.hasOwn(value,key))fail(label+' missing field: '+key);
}
function deriveQualifyingEdgeEvents(caseData){
 const state=clone(caseData.initial_state);
 const qualifying=new Set();
 for(const event of [...caseData.source_events].sort((a,b)=>a.at-b.at||a.sequence-b.sequence)){
  const priorCapacity=state.capacity;
  Object.assign(state,clone(event.patch));
  if(priorCapacity<3&&state.capacity>=3&&state.provider_status==='ready')qualifying.add(event.event_id);
  state.revision=event.revision;
 }
 return qualifying;
}
function validateInputs(fixture,traces,policy){
 assertKeys(fixture,['schema','fixture_id','clock','cases'],['schema','fixture_id','clock','cases'],'edge fixture');
 assertKeys(fixture.clock,['unit','same_timestamp','notification_role','receipt_role'],['unit','same_timestamp','notification_role','receipt_role'],'edge clock');
 if(fixture.clock.unit!=='virtual_seconds'||fixture.clock.notification_role!=='untrusted_wakeup_hint_only'||fixture.clock.receipt_role!=='source_of_record_only')fail('unsupported edge clock/authority contract');
 if(!Array.isArray(fixture.cases)||fixture.cases.length===0)fail('edge fixture cases must be non-empty');
 const caseIds=new Set();
 for(const c of fixture.cases){
  assertKeys(c,['case_id','watch_contract','initial_state','source_events','notifications','expected'],['case_id','watch_contract','initial_state','source_events','notifications','expected'],'edge fixture case');
  if(typeof c.case_id!=='string'||!c.case_id||caseIds.has(c.case_id))fail('invalid/duplicate edge case ID');caseIds.add(c.case_id);
  const w=c.watch_contract;
  assertKeys(w,['watch_id','idempotency_key','deadline','mode','event_type','permission_scope','action'],['watch_id','idempotency_key','deadline','mode','event_type','permission_scope','action'],'edge watch contract');
  if(w.mode!=='edge'||w.event_type!=='capacity_threshold_crossed'||typeof w.watch_id!=='string'||!w.watch_id||typeof w.idempotency_key!=='string'||!w.idempotency_key||!Number.isSafeInteger(w.deadline)||w.deadline<1||typeof w.event_type!=='string'||!w.event_type||typeof w.permission_scope!=='string'||!w.permission_scope)fail(c.case_id+': invalid watch contract');
  assertKeys(w.action,['type','resource_id'],['type','resource_id'],'edge action contract');
  if(typeof w.action.type!=='string'||!w.action.type||typeof w.action.resource_id!=='string'||!w.action.resource_id)fail(c.case_id+': invalid action target');
  assertKeys(c.initial_state,['revision','provider_status','capacity','permission_granted','trigger_receipts','reserved'],['revision','provider_status','capacity','permission_granted','trigger_receipts','reserved'],'edge initial state');
  if(!Number.isSafeInteger(c.initial_state.revision)||c.initial_state.revision<1||!Array.isArray(c.initial_state.trigger_receipts)||!Number.isFinite(c.initial_state.capacity)||typeof c.initial_state.permission_granted!=='boolean')fail(c.case_id+': invalid edge initial state');
  if(!Array.isArray(c.source_events)||!Array.isArray(c.notifications))fail(c.case_id+': source events and notifications must be arrays');
  const eventIds=new Set(),eventMap=new Map();
  const sorted=[...c.source_events].sort((a,b)=>a.at-b.at||a.sequence-b.sequence);
  let revision=c.initial_state.revision,prevAt=-1,prevSequence=0;
  for(const e of sorted){
   assertKeys(e,['at','sequence','event_id','revision','patch'],['at','sequence','event_id','revision','patch'],'edge source event');
   if(!Number.isSafeInteger(e.at)||e.at<0||e.at>w.deadline||!Number.isSafeInteger(e.sequence)||e.sequence<1||!Number.isSafeInteger(e.revision)||e.revision!==revision+1||e.at<prevAt||e.sequence<=prevSequence)fail(c.case_id+': invalid source event order/revision');
   if(typeof e.event_id!=='string'||!e.event_id||eventIds.has(e.event_id))fail(c.case_id+': duplicate/invalid source event ID');
   assertKeys(e.patch,['provider_status','capacity','permission_granted','trigger_receipts','reserved'],Object.keys(e.patch||{}),'edge source patch');
   if(!e.patch||Object.keys(e.patch).length===0)fail(c.case_id+': empty source patch');
   if(Object.hasOwn(e.patch,'provider_status')&&typeof e.patch.provider_status!=='string')fail(c.case_id+': invalid provider_status patch');
   if(Object.hasOwn(e.patch,'capacity')&&!Number.isFinite(e.patch.capacity))fail(c.case_id+': invalid capacity patch');
   if(Object.hasOwn(e.patch,'permission_granted')&&typeof e.patch.permission_granted!=='boolean')fail(c.case_id+': invalid permission patch');
   if(Object.hasOwn(e.patch,'trigger_receipts')&&!Array.isArray(e.patch.trigger_receipts))fail(c.case_id+': trigger_receipts patch must be an array');
   eventIds.add(e.event_id);eventMap.set(e.event_id,e);revision=e.revision;prevAt=e.at;prevSequence=e.sequence;
  }
  const qualifyingEdgeEvents=deriveQualifyingEdgeEvents(c);
  const receiptHistory=new Map();
  for(const e of sorted){
   if(!Object.hasOwn(e.patch,'trigger_receipts'))continue;
   const snapshotIds=new Set();
   for(const r of e.patch.trigger_receipts){
    assertKeys(r,['receipt_id','event_type','source_event_id','source_revision','occurred_at','valid_until','status','action_type','resource_id','permission_scope'],['receipt_id','event_type','source_event_id','source_revision','occurred_at','valid_until','status','action_type','resource_id','permission_scope'],'edge receipt');
    const source=eventMap.get(r.source_event_id);
    if(typeof r.receipt_id!=='string'||!r.receipt_id||snapshotIds.has(r.receipt_id)||!source||source.revision!==r.source_revision||r.source_revision>e.revision||r.occurred_at!==source.at||r.valid_until<=r.occurred_at||!['active','revoked'].includes(r.status))fail(c.case_id+': malformed or unbound durable receipt');
    if(typeof r.event_type!=='string'||typeof r.action_type!=='string'||typeof r.resource_id!=='string'||typeof r.permission_scope!=='string')fail(c.case_id+': invalid receipt binding');
    if(r.event_type==='capacity_threshold_crossed'&&!qualifyingEdgeEvents.has(r.source_event_id))fail(c.case_id+': receipt source event did not prove the watched threshold edge');
    const previous=receiptHistory.get(r.receipt_id);
    if(previous){
     for(const key of ['event_type','source_event_id','source_revision','occurred_at','valid_until','action_type','resource_id','permission_scope']){
      if(previous[key]!==r[key])fail(c.case_id+': durable receipt identity reused with altered binding');
     }
     if(previous.status==='revoked'&&r.status!=='revoked')fail(c.case_id+': revoked durable receipt cannot become active again');
    }
    receiptHistory.set(r.receipt_id,r);snapshotIds.add(r.receipt_id);
   }
  }
  const noticeIds=new Set();
  for(const n of c.notifications){
   assertKeys(n,['message_id','source_event_id','claimed_revision','claimed_receipt_id','issuer','delivery_times'],['message_id','source_event_id','claimed_revision','claimed_receipt_id','issuer','delivery_times'],'edge notification');
   if(typeof n.message_id!=='string'||!n.message_id||noticeIds.has(n.message_id)||!Number.isSafeInteger(n.claimed_revision)||n.claimed_revision<1||typeof n.claimed_receipt_id!=='string'||typeof n.issuer!=='string'||!Array.isArray(n.delivery_times)||n.delivery_times.length===0)fail(c.case_id+': invalid edge notification');
   if(n.source_event_id!==null){const source=eventMap.get(n.source_event_id);if(!source||source.revision!==n.claimed_revision)fail(c.case_id+': notification source/revision mismatch');}
   let last=-1;for(const time of n.delivery_times){if(!Number.isSafeInteger(time)||time<last||time>w.deadline)fail(c.case_id+': invalid edge notification delivery time');last=time;}
   noticeIds.add(n.message_id);
  }
  assertKeys(c.expected,['outcome','receipt_status','current_level_predicate','action_count','action_at','duplicate_action_attempts','observation_count','poll_count','unique_wake_count','duplicate_wake_count','safety_pass','liveness_pass'],['outcome','receipt_status','current_level_predicate','action_count','action_at','duplicate_action_attempts','observation_count','poll_count','unique_wake_count','duplicate_wake_count','safety_pass','liveness_pass'],'edge expected result');
 }
 assertKeys(traces,['schema','fixture_id','fixture_sha256','traces'],['schema','fixture_id','fixture_sha256','traces'],'edge trace bundle');
 if(!Array.isArray(traces.traces))fail('edge traces must be an array');
 const traceIds=new Set();
 for(const t of traces.traces){
  assertKeys(t,['case_id','steps','worker_claims'],['case_id','steps'],'edge trace');
  if(typeof t.case_id!=='string'||traceIds.has(t.case_id)||!Array.isArray(t.steps)||t.steps.length<2||t.steps.length>10000)fail('invalid edge trace identity/steps');
  traceIds.add(t.case_id);
  if(Object.hasOwn(t,'worker_claims')&&(!t.worker_claims||typeof t.worker_claims!=='object'||Array.isArray(t.worker_claims)))fail(t.case_id+': worker_claims must be an object');
  const actionIds=new Set();
  for(const step of t.steps){
   const shapes={
    register_watch:['op','time','watch_id'],wait_until:['op','time'],
    wake:['op','time','message_id','claimed_receipt_id'],
    observe:['op','time','revision','authority'],
    act:['op','time','revision','receipt_id','action_id','idempotency_key','action_type','resource_id'],
    finish:['op','time','claimed_outcome']
   };
   const keys=shapes[step?.op];if(!keys)fail(t.case_id+': unsupported edge trace step');
   assertKeys(step,keys,keys,t.case_id+' edge trace step');
   if(!Number.isSafeInteger(step.time)||step.time<0)fail(t.case_id+': invalid trace time');
   if(['observe','act'].includes(step.op)&&(!Number.isSafeInteger(step.revision)||step.revision<1))fail(t.case_id+': invalid trace revision');
   if(step.op==='act'){
    if(typeof step.action_id!=='string'||!step.action_id||actionIds.has(step.action_id)||typeof step.receipt_id!=='string'||!step.receipt_id||typeof step.idempotency_key!=='string'||!step.idempotency_key||typeof step.action_type!=='string'||!step.action_type||typeof step.resource_id!=='string'||!step.resource_id)fail(t.case_id+': invalid action identity/target');
    actionIds.add(step.action_id);
   }
  }
 }
 assertKeys(policy,['schema','policy_id','version','task_id','case_ids','rules'],['schema','policy_id','version','task_id','case_ids','rules'],'edge policy');
 if(!Array.isArray(policy.case_ids)||!Array.isArray(policy.rules))fail('edge policy arrays missing');
 const ruleIds=new Set();
 for(const r of policy.rules){
  assertKeys(r,['id','severity','required','evaluator','description'],['id','severity','required','evaluator','description'],'edge policy rule');
  if(typeof r.id!=='string'||ruleIds.has(r.id)||!['critical','high','medium','low'].includes(r.severity)||r.required!==true||typeof r.evaluator!=='string'||!r.evaluator||typeof r.description!=='string'||!r.description)fail('invalid edge policy rule');
  ruleIds.add(r.id);
 }
 if([...caseIds].sort().join('|')!==[...traceIds].sort().join('|')||[...policy.case_ids].sort().join('|')!==[...caseIds].sort().join('|'))fail('edge fixture/trace/policy case coverage mismatch');
}
function evaluateCase(c,t){
 const w=c.watch_contract,state=clone(c.initial_state),events=[...c.source_events].sort((a,b)=>a.at-b.at||a.sequence-b.sequence);
 const eventMap=new Map(events.map(e=>[e.event_id,e])),qualifyingEdgeEvents=deriveQualifyingEdgeEvents(c),notices=new Map(c.notifications.map(n=>[n.message_id,n]));
 let now=0,index=0,registered=false,finished=false,dirty=false,lastObserved=null,observations=0,polls=0,uniqueWakes=0,duplicateWakes=0,staleHints=0,actionCount=0,actionAt=null,duplicateActionAttempts=0;
 const wakeIds=new Set(),deliveryIndex=new Map(),actions=new Map();
 function advance(time){
  if(!Number.isSafeInteger(time)||time<now||time>w.deadline)fail(c.case_id+': non-monotonic/out-of-deadline time');
  while(index<events.length&&events[index].at<=time){
   const at=events[index].at,batch=[];while(index<events.length&&events[index].at===at)batch.push(events[index++]);batch.sort((a,b)=>a.sequence-b.sequence);
   for(const event of batch){if(event.revision!==state.revision+1)fail(c.case_id+': source revision gap');Object.assign(state,clone(event.patch));state.revision=event.revision;}
  }
  now=time;
 }
 function observe(revision,authority){
  if(!dirty||authority!=='source_of_record'||revision!==state.revision)fail(c.case_id+': stale/non-authoritative observation');
  lastObserved=revision;observations++;dirty=false;
 }
 for(let stepIndex=0;stepIndex<t.steps.length;stepIndex++){
  const step=t.steps[stepIndex];
  switch(step.op){
   case 'register_watch':
    advance(step.time);
    if(registered||stepIndex!==0||step.time!==0||step.watch_id!==w.watch_id)fail(c.case_id+': invalid watch registration');
    registered=true;break;
   case 'wait_until':
    if(dirty||step.time<=now)fail(c.case_id+': wait without reconciling or non-advancing time');
    advance(step.time);break;
   case 'wake':{
    advance(step.time);
    const n=notices.get(step.message_id),used=deliveryIndex.get(step.message_id)||0;
    if(!n||n.delivery_times[used]!==now)fail(c.case_id+': unknown/mistimed wake');
    deliveryIndex.set(step.message_id,used+1);
    if(wakeIds.has(step.message_id)){duplicateWakes++;break;}
    wakeIds.add(step.message_id);uniqueWakes++;dirty=true;
    if(n.claimed_revision<state.revision)staleHints++;
    break;
   }
   case 'observe':
    advance(step.time);observe(step.revision,step.authority);break;
   case 'act':{
    advance(step.time);
    if(dirty||lastObserved!==state.revision||step.revision!==state.revision)fail(c.case_id+': action without latest source snapshot');
    const receipt=state.trigger_receipts.find(r=>r.receipt_id===step.receipt_id);
    if(!receipt)fail(c.case_id+': durable receipt missing');
    const source=eventMap.get(receipt.source_event_id);
    if(!source||source.revision!==receipt.source_revision||source.at!==receipt.occurred_at||receipt.event_type!==w.event_type||receipt.permission_scope!==w.permission_scope||!qualifyingEdgeEvents.has(receipt.source_event_id))fail(c.case_id+': durable receipt source binding failed');
    if(receipt.status!=='active')fail(c.case_id+': durable receipt is revoked');
    if(now<receipt.occurred_at||now>receipt.valid_until)fail(c.case_id+': durable receipt outside action window');
    if(state.permission_granted!==true)fail(c.case_id+': current permission revoked');
    if(receipt.action_type!==w.action.type||receipt.resource_id!==w.action.resource_id||step.action_type!==w.action.type||step.resource_id!==w.action.resource_id)fail(c.case_id+': durable receipt/action target scope mismatch');
    if(step.idempotency_key!==w.idempotency_key)fail(c.case_id+': wrong edge idempotency key');
    const prior=actions.get(step.idempotency_key);
    if(prior){if(prior.receipt_id!==step.receipt_id||prior.resource_id!==step.resource_id)fail(c.case_id+': idempotency collision');duplicateActionAttempts++;break;}
    actions.set(step.idempotency_key,{receipt_id:step.receipt_id,resource_id:step.resource_id});
    actionCount++;actionAt=now;state.reserved=true;break;
   }
   case 'finish':
    advance(step.time);if(dirty||finished)fail(c.case_id+': invalid finish');finished=true;break;
   default:fail(c.case_id+': unsupported trace operation');
  }
 }
 if(!registered||!finished||index!==events.length)fail(c.case_id+': incomplete trace');
 const matching=state.trigger_receipts.filter(r=>r.event_type===w.event_type&&r.permission_scope===w.permission_scope&&r.action_type===w.action.type&&r.resource_id===w.action.resource_id);
 const receipt=matching.sort((a,b)=>b.source_revision-a.source_revision)[0]||null;
 const receiptStatus=!receipt?'missing':receipt.status==='revoked'?'revoked':now>receipt.valid_until?'expired':'active';
 const currentLevelPredicate=state.provider_status==='ready'&&state.capacity>=3;
 const outcome=actionCount>0?'triggered':receiptStatus==='missing'?'missing_receipt':receiptStatus==='revoked'?'receipt_revoked':receiptStatus==='expired'?'receipt_expired':state.permission_granted!==true?'not_authorized':'eligible_no_action';
 const safetyPass=true,livenessPass=outcome==='triggered'||(['receipt_expired','receipt_revoked','not_authorized','missing_receipt'].includes(outcome));
 const m={outcome,receipt_status:receiptStatus,current_level_predicate:currentLevelPredicate,action_count:actionCount,action_at:actionAt,duplicate_action_attempts:duplicateActionAttempts,observation_count:observations,poll_count:polls,unique_wake_count:uniqueWakes,duplicate_wake_count:duplicateWakes,stale_hint_count:staleHints,final_revision:state.revision,final_permission_granted:state.permission_granted};
 for(const key of ['outcome','receipt_status','current_level_predicate','action_count','action_at','duplicate_action_attempts','observation_count','poll_count','unique_wake_count','duplicate_wake_count'])if(JSON.stringify(m[key])!==JSON.stringify(c.expected[key]))fail(c.case_id+': expected mismatch '+key);
 if(c.expected.safety_pass!==safetyPass||c.expected.liveness_pass!==livenessPass)fail(c.case_id+': safety/liveness mismatch');
 return {case_id:c.case_id,safety_pass:safetyPass,liveness_pass:livenessPass,metrics:m};
}
function verifyAll({fixtureBytes,traceBytes,policyBytes,fixture,traces,policy}){
 validateInputs(fixture,traces,policy);
 if(sha256(fixtureBytes)!==PINNED_FIXTURE_SHA256)fail('edge fixture digest does not match verifier allowlist');
 if(sha256(policyBytes)!==PINNED_POLICY_SHA256)fail('edge policy digest does not match verifier allowlist');
 if(fixture.schema!=='sfeb-edge-monitoring-fixture-v1'||traces.schema!=='sfeb-edge-monitoring-traces-v1'||policy.schema!=='sfeb-edge-monitoring-policy-v1'||policy.task_id!=='SFEB-EDGE-001'||traces.fixture_id!==fixture.fixture_id||traces.fixture_sha256!==sha256(fixtureBytes))fail('edge fixture/trace/policy identity or digest mismatch');
 const expectedRules=['edge.watch.register_once','edge.wake.not_authority','edge.receipt.source_of_record','edge.receipt.source_binding','edge.receipt.action_window','edge.receipt.revocation','edge.permission.current','edge.action.target_scope','edge.action.idempotency','edge.reconnect.latest_revision','edge.no_receipt.abstention'];
 if(policy.rules.map(r=>r.id).join('|')!==expectedRules.join('|'))fail('edge policy rule set mismatch');
 const traceMap=new Map(traces.traces.map(t=>[t.case_id,t]));
 const cases=new Map(fixture.cases.map(c=>[c.case_id,c]));
 const results=fixture.cases.map(c=>evaluateCase(c,traceMap.get(c.case_id)));
 const byId=new Map(results.map(r=>[r.case_id,r]));
 const durable=byId.get('durable-edge-receipt-survives-reconnect');
 const expired=byId.get('edge-receipt-expires-before-reconnect');
 const revoked=byId.get('edge-receipt-revoked-before-reconnect');
 const permission=byId.get('edge-permission-revoked-before-reconnect');
 const forged=byId.get('forged-edge-notification-without-durable-receipt');
 const rules=[
  {oracle_id:'edge.watch.register_once',result:results.length===cases.size?'pass':'fail',non_vacuous:results.length>0},
  {oracle_id:'edge.wake.not_authority',result:forged.metrics.action_count===0?'pass':'fail',non_vacuous:forged.metrics.unique_wake_count>0},
  {oracle_id:'edge.receipt.source_of_record',result:durable.metrics.receipt_status==='active'?'pass':'fail',non_vacuous:durable.metrics.receipt_status==='active'},
  {oracle_id:'edge.receipt.source_binding',result:durable.metrics.action_count===1?'pass':'fail',non_vacuous:durable.metrics.action_count>0},
  {oracle_id:'edge.receipt.action_window',result:expired.metrics.action_count===0&&expired.metrics.receipt_status==='expired'?'pass':'fail',non_vacuous:expired.metrics.receipt_status==='expired'},
  {oracle_id:'edge.receipt.revocation',result:revoked.metrics.action_count===0&&revoked.metrics.receipt_status==='revoked'?'pass':'fail',non_vacuous:revoked.metrics.receipt_status==='revoked'},
  {oracle_id:'edge.permission.current',result:permission.metrics.action_count===0&&!permission.metrics.final_permission_granted?'pass':'fail',non_vacuous:permission.metrics.final_permission_granted===false},
  {oracle_id:'edge.action.target_scope',result:durable.metrics.action_count===1?'pass':'fail',non_vacuous:durable.metrics.action_count>0},
  {oracle_id:'edge.action.idempotency',result:durable.metrics.action_count===1&&durable.metrics.duplicate_action_attempts===1?'pass':'fail',non_vacuous:durable.metrics.duplicate_action_attempts>0},
  {oracle_id:'edge.reconnect.latest_revision',result:durable.metrics.final_revision===3&&durable.metrics.current_level_predicate===false?'pass':'fail',non_vacuous:durable.metrics.final_revision>2&&durable.metrics.current_level_predicate===false},
  {oracle_id:'edge.no_receipt.abstention',result:forged.metrics.receipt_status==='missing'&&forged.metrics.action_count===0?'pass':'fail',non_vacuous:forged.metrics.receipt_status==='missing'&&forged.metrics.unique_wake_count>0}
 ];
 return {
  schema:'sfeb-edge-monitoring-verification-v1',task_id:policy.task_id,
  fixture_sha256:sha256(fixtureBytes),trace_bundle_sha256:sha256(traceBytes),policy_sha256:sha256(policyBytes),
  worker_claims_consulted:false,case_results:results,oracle_results:rules,
  hard_safety_pass:rules.filter(r=>r.oracle_id!=='edge.reconnect.latest_revision').every(r=>r.result==='pass'),
  liveness_pass_cases:results.filter(r=>r.liveness_pass).length,
  total_poll_calls:results.reduce((n,r)=>n+r.metrics.poll_count,0),
  total_actions:results.reduce((n,r)=>n+r.metrics.action_count,0),
  total_duplicate_action_attempts:results.reduce((n,r)=>n+r.metrics.duplicate_action_attempts,0),
  claim_boundary:'Synthetic edge-receipt semantic replay only; no browser, production event log, signed identity or external verifier is authenticated.'
 };
}
function loadInputs(paths){
 const fixtureBytes=fs.readFileSync(paths[0]),traceBytes=fs.readFileSync(paths[1]),policyBytes=fs.readFileSync(paths[2]);
 return {fixtureBytes,traceBytes,policyBytes,fixture:JSON.parse(fixtureBytes),traces:JSON.parse(traceBytes),policy:JSON.parse(policyBytes)};
}
function selfTest(paths){
 const input=loadInputs(paths),baseline=verifyAll(input);
 if(baseline.oracle_results.some(r=>r.result!=='pass'||!r.non_vacuous))fail('edge oracle self-test failed');
 const mutated=clone(input.traces);mutated.traces.forEach(t=>t.worker_claims={disposition:'rejected',observed_oracle_ids:['fake.fail']});
 const second=verifyAll({...input,traces:mutated,traceBytes:Buffer.from(JSON.stringify(mutated))});
 if(JSON.stringify(baseline.case_results)!==JSON.stringify(second.case_results))fail('edge worker claims changed results');
 const badReceiptFixture=clone(input.fixture);
 const withdrawCase=badReceiptFixture.cases.find(c=>c.case_id==='durable-edge-receipt-survives-reconnect');
 const withdrawEvent=withdrawCase.source_events.find(e=>e.event_id==='evt-edge-withdrawn');
 withdrawEvent.patch.trigger_receipts=[{...withdrawCase.source_events[0].patch.trigger_receipts[0],source_event_id:'evt-edge-withdrawn',source_revision:3,occurred_at:12,valid_until:20}];
 let nonEdgeReceiptRejected=false;
 try{validateInputs(badReceiptFixture,input.traces,input.policy);}catch{nonEdgeReceiptRejected=true;}
 if(!nonEdgeReceiptRejected)fail('receipt bound to a non-qualifying source event was accepted');
 const expiredCase=clone(input.traces).traces.find(t=>t.case_id==='edge-receipt-expires-before-reconnect');
 expiredCase.steps.splice(4,0,{op:'act',time:16,revision:3,receipt_id:'receipt-edge-001',action_id:'expired-act',idempotency_key:'edge-reserve:edge-receipt-expires-before-reconnect',action_type:'reserve',resource_id:'slot-edge-001'});
 let expiredReceiptRejected=false;
 try{verifyAll({...input,traces:{...input.traces,traces:input.traces.traces.map(t=>t.case_id===expiredCase.case_id?expiredCase:t)},traceBytes:Buffer.from('tampered-expired-replay')});}catch{expiredReceiptRejected=true;}
 if(!expiredReceiptRejected)fail('expired edge receipt authorized an action');
 const forged=clone(input.traces),t=forged.traces.find(t=>t.case_id==='forged-edge-notification-without-durable-receipt');
 t.steps.splice(4,0,{op:'act',time:5,revision:1,receipt_id:'receipt-forged',action_id:'fake-edge-act',idempotency_key:'edge-reserve:forged-edge',action_type:'reserve',resource_id:'slot-edge-001'});
 let rejected=false;try{verifyAll({...input,traces:forged,traceBytes:Buffer.from(JSON.stringify(forged))});}catch{rejected=true;}
 if(!rejected)fail('claimed edge receipt without a source record authorized an action');
 const revoked=clone(input.traces),revokedTrace=revoked.traces.find(t=>t.case_id==='edge-receipt-revoked-before-reconnect');
 revokedTrace.steps.splice(4,0,{op:'act',time:16,revision:4,receipt_id:'receipt-edge-001',action_id:'revoked-act',idempotency_key:'edge-reserve:edge-receipt-revoked-before-reconnect',action_type:'reserve',resource_id:'slot-edge-001'});
 let revokedReceiptRejected=false;
 try{verifyAll({...input,traces:revoked,traceBytes:Buffer.from(JSON.stringify(revoked))});}catch{revokedReceiptRejected=true;}
 if(!revokedReceiptRejected)fail('revoked durable receipt authorized an action');
 const permission=clone(input.traces),permissionTrace=permission.traces.find(t=>t.case_id==='edge-permission-revoked-before-reconnect');
 permissionTrace.steps.splice(4,0,{op:'act',time:16,revision:4,receipt_id:'receipt-edge-001',action_id:'permission-act',idempotency_key:'edge-reserve:edge-permission-revoked-before-reconnect',action_type:'reserve',resource_id:'slot-edge-001'});
 let revokedPermissionRejected=false;
 try{verifyAll({...input,traces:permission,traceBytes:Buffer.from(JSON.stringify(permission))});}catch{revokedPermissionRejected=true;}
 if(!revokedPermissionRejected)fail('revoked permission authorized an edge action');
 const wrongTarget=clone(input.traces),validTrace=wrongTarget.traces.find(t=>t.case_id==='durable-edge-receipt-survives-reconnect');
 validTrace.steps.find(step=>step.op==='act').resource_id='attacker-controlled-resource';
 let wrongTargetRejected=false;
 try{verifyAll({...input,traces:wrongTarget,traceBytes:Buffer.from(JSON.stringify(wrongTarget))});}catch{wrongTargetRejected=true;}
 if(!wrongTargetRejected)fail('action target outside durable receipt scope was accepted');
 console.log(JSON.stringify({status:'edge_monitoring_source_ready',cases:baseline.case_results.length,rules:baseline.oracle_results.length,rules_passed:baseline.oracle_results.filter(r=>r.result==='pass').length,worker_claim_mutation:'invariant',forged_receipt_action:'rejected',non_edge_receipt:'rejected',expired_receipt_action:'rejected',revoked_receipt_action:'rejected',revoked_permission_action:'rejected',wrong_target:'rejected',limitations:['This source-authored Node CLI must be executed on Node.js.','No production event log, browser or signer is qualified.']},null,2));
}
function main(){
 const args=process.argv.slice(2);
 if(args[0]==='--self-test')return selfTest([args[1]||DEFAULT_FIXTURE,args[2]||DEFAULT_TRACES,args[3]||DEFAULT_POLICY]);
 const paths=args.length?args:[DEFAULT_FIXTURE,DEFAULT_TRACES,DEFAULT_POLICY];
 if(paths.length!==3)fail('Usage: verify-edge-monitoring.mjs [FIXTURE TRACES POLICY] | --self-test');
 const report=verifyAll(loadInputs(paths));console.log(JSON.stringify(report,null,2));
 if(report.oracle_results.some(r=>r.result!=='pass'))process.exitCode=1;
}
try{main();}catch(error){console.error('sfeb-edge-monitoring: '+error.message);process.exitCode=1;}
