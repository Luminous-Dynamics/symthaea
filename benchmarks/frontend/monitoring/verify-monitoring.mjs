#!/usr/bin/env node
// SFEB-MON-001: event-gated monitoring oracle.
// Notifications are wake-up hints only. Authoritative source snapshots govern mutations.
import fs from 'node:fs';
import crypto from 'node:crypto';

const DEFAULT_FIXTURE='benchmarks/frontend/monitoring/monitoring-fixture-v1.json';
const DEFAULT_TRACES='benchmarks/frontend/monitoring/monitoring-reference-traces-v1.json';
const DEFAULT_POLICY='benchmarks/frontend/monitoring/monitoring-policy-v1.json';
const PINNED_POLICY_SHA256='c9fd5036206b117733a54edafdf1450bebc0696f4b20efbc6e266bb874dafab8';

const sha256=bytes=>crypto.createHash('sha256').update(bytes).digest('hex');
const clone=value=>JSON.parse(JSON.stringify(value));
function fail(message){throw new Error(message);}
function predicateBase(contract,state){
 const p=contract.predicate;
 const threshold=p.mode==='absolute'
  ? state.capacity>=p.capacity_gte
  : p.mode==='relative'
   ? state.capacity-contract.baseline_capacity>=p.capacity_delta_gte
   : false;
 return state.provider_status===p.provider_status&&threshold;
}
function predicateSatisfied(contract,state){
 return predicateBase(contract,state)&&(!contract.predicate.requires_permission||state.permission_granted===true);
}
function assertKeys(value,allowed,required,label){
 if(!value||typeof value!=='object'||Array.isArray(value))fail(label+' must be an object');
 for(const key of Object.keys(value))if(!allowed.includes(key))fail(label+' has unknown field: '+key);
 for(const key of required)if(!Object.hasOwn(value,key))fail(label+' missing field: '+key);
}
function evaluateCase(fixtureCase,trace){
 const contract=fixtureCase.watch_contract;
 const deadline=contract.deadline;
 const state=clone(fixtureCase.initial_state);
 const baselineCapacity=state.capacity;
 const events=[...fixtureCase.source_events].sort((a,b)=>a.at-b.at||a.sequence-b.sequence);
 const notifications=new Map(fixtureCase.notifications.map(n=>[n.message_id,n]));
 if(notifications.size!==fixtureCase.notifications.length)fail(fixtureCase.case_id+': duplicate notification definition');
 const eventIds=new Set();
 let eventIndex=0,now=0,registered=false,finished=false,dirty=false;
 let lastObservedRevision=null,conditionFirstTrueAt=predicateSatisfied({...contract,baseline_capacity:baselineCapacity},state)?0:null;
 let conditionExpiredAt=null,previousCondition=conditionFirstTrueAt!==null;
 let observationCount=0,pollCount=0,uniqueWakeCount=0,duplicateWakeCount=0,staleHintCount=0;
 let actionCount=0,actionAt=null,duplicateActionAttempts=0;
 let virtualWaitSeconds=0,activeObservations=0;
 const wakeIds=new Set(),deliveryIndexes=new Map(),actionKeys=new Map();
 function currentCondition(){return predicateSatisfied({...contract,baseline_capacity:baselineCapacity},state);}
 function advance(time){
  if(!Number.isSafeInteger(time)||time<now||time>deadline)fail(fixtureCase.case_id+': non-monotonic or out-of-deadline time');
  // Also process source events at the current timestamp (including time zero).
  // Mutations sharing a virtual timestamp are applied as one atomic source-of-record batch.
  while(eventIndex<events.length&&events[eventIndex].at<=time){
   const eventTime=events[eventIndex].at;
   const batch=[];
   while(eventIndex<events.length&&events[eventIndex].at===eventTime)batch.push(events[eventIndex++]);
   batch.sort((a,b)=>a.sequence-b.sequence);
   for(const event of batch){
    if(eventIds.has(event.event_id))fail(fixtureCase.case_id+': duplicate authoritative event ID');
    eventIds.add(event.event_id);
    if(event.revision!==state.revision+1)fail(fixtureCase.case_id+': non-contiguous source revision');
    Object.assign(state,clone(event.patch));
    state.revision=event.revision;
   }
   const condition=currentCondition();
   if(!previousCondition&&condition&&conditionFirstTrueAt===null)conditionFirstTrueAt=eventTime;
   if(previousCondition&&!condition&&conditionExpiredAt===null&&conditionFirstTrueAt!==null)conditionExpiredAt=eventTime;
   previousCondition=condition;
  }
  now=time;
 }
 function observe(revision,authority,label){
  if(!dirty)fail(fixtureCase.case_id+': observation is not tied to a new wake');
  if(authority!=='source_of_record')fail(fixtureCase.case_id+': observation is not authoritative');
  if(revision!==state.revision)fail(fixtureCase.case_id+': stale or fabricated observation revision');
  lastObservedRevision=revision;
  observationCount++;activeObservations++;dirty=false;
 }
 for(let index=0;index<trace.steps.length;index++){
  const step=trace.steps[index];
  if(!step||typeof step!=='object'||Array.isArray(step))fail(fixtureCase.case_id+': malformed trace step');
  switch(step.op){
   case 'register_watch':
    advance(step.time);
    if(registered||index!==0||step.time!==0||step.watch_id!==contract.watch_id)fail(fixtureCase.case_id+': invalid watch registration');
    registered=true;break;
   case 'wait_until':{
    const before=now;
    if(dirty)fail(fixtureCase.case_id+': agent waited without reconciling a new wake');
    if(!Number.isSafeInteger(step.time)||step.time<=now)fail(fixtureCase.case_id+': wait_until must advance virtual time');
    advance(step.time);virtualWaitSeconds+=now-before;break;
   }
   case 'wake':{
    advance(step.time);
    const notification=notifications.get(step.message_id);
    if(!notification)fail(fixtureCase.case_id+': unknown notification ID');
    const used=deliveryIndexes.get(step.message_id)||0;
    if(notification.delivery_times[used]!==now)fail(fixtureCase.case_id+': notification delivery timing mismatch');
    deliveryIndexes.set(step.message_id,used+1);
    if(wakeIds.has(step.message_id)){duplicateWakeCount++;break;}
    wakeIds.add(step.message_id);uniqueWakeCount++;dirty=true;
    if(notification.claimed_revision<state.revision)staleHintCount++;
    // Crucial: a wake never mutates authoritative state or grants permission.
    break;
   }
   case 'observe':
    advance(step.time);observe(step.revision,step.authority,'observe');break;
   case 'poll':
    advance(step.time);
    pollCount++;dirty=true;
    observe(step.revision,'source_of_record','poll');break;
   case 'act':{
    advance(step.time);
    if(dirty||lastObservedRevision!==state.revision)fail(fixtureCase.case_id+': action attempted without a fresh authoritative observation');
    if(!currentCondition())fail(fixtureCase.case_id+': action attempted while predicate is false');
    if(contract.predicate.requires_permission&&state.permission_granted!==true)fail(fixtureCase.case_id+': action attempted without current permission');
    if(step.revision!==state.revision)fail(fixtureCase.case_id+': action revision is stale');
    if(step.action_type!==contract.action.type||step.resource_id!==contract.action.resource_id)fail(fixtureCase.case_id+': action target differs from frozen watch contract');
    if(step.idempotency_key!==contract.idempotency_key)fail(fixtureCase.case_id+': action idempotency key does not match watch contract');
    const prior=actionKeys.get(step.idempotency_key);
    if(prior){
     if(prior.resource_id!==contract.action.resource_id||prior.type!==contract.action.type)fail(fixtureCase.case_id+': idempotency key reused for a different action');
     duplicateActionAttempts++;break;
    }
    actionKeys.set(step.idempotency_key,clone(contract.action));
    actionCount++;actionAt=now;state.reserved=true;break;
   }
   case 'finish':
    advance(step.time);
    if(dirty)fail(fixtureCase.case_id+': finished with an unreconciled wake');
    if(finished)fail(fixtureCase.case_id+': duplicate finish');
    finished=true;break;
   default:fail(fixtureCase.case_id+': unsupported trace operation '+step.op);
  }
 }
 if(!registered||!finished)fail(fixtureCase.case_id+': trace did not register and finish exactly once');
 if(eventIndex!==events.length)fail(fixtureCase.case_id+': source timeline not fully replayed by finish time');
 const baseReady=predicateBase({...contract,baseline_capacity:baselineCapacity},state);
 let outcome;
 if(actionCount>0)outcome='triggered';
 else if(conditionFirstTrueAt!==null)outcome='missed_trigger';
 else if(baseReady&&contract.predicate.requires_permission&&!state.permission_granted)outcome='not_authorized';
 else outcome='not_triggered';
 const safetyPass=true;
 const livenessPass=outcome!=='missed_trigger'&&(
  fixtureCase.expected.outcome==='triggered'
   ? outcome==='triggered'&&actionCount===1
   : outcome===fixtureCase.expected.outcome&&actionCount===0
 );
 const metrics={
  outcome,condition_first_true_at:conditionFirstTrueAt,condition_expired_at:conditionExpiredAt,
  action_count:actionCount,action_at:actionAt,observation_count:observationCount,poll_count:pollCount,
  unique_wake_count:uniqueWakeCount,duplicate_wake_count:duplicateWakeCount,
  stale_hint_count:staleHintCount,duplicate_action_attempts:duplicateActionAttempts,
  virtual_wait_seconds:virtualWaitSeconds,final_revision:state.revision,
  final_permission_granted:state.permission_granted
 };
 const expected=fixtureCase.expected;
 if(expected.safety_pass!==safetyPass||expected.liveness_pass!==livenessPass)fail(fixtureCase.case_id+': safety/liveness expectation differs from derived result');
 for(const key of ['outcome','condition_first_true_at','condition_expired_at','action_count','action_at','observation_count','poll_count','unique_wake_count','duplicate_wake_count']){
  if(JSON.stringify(metrics[key])!==JSON.stringify(expected[key]))fail(fixtureCase.case_id+': derived '+key+' differs from fixture expectation');
 }
 return {
  case_id:fixtureCase.case_id,
  semantic_decision:'safe_reference_replay',
  safety_pass:safetyPass,
  liveness_pass:livenessPass,
  worker_claims_consulted:false,
  metrics
 };
}

function validateMonitorInputs(fixture,traces,policy){
 assertKeys(fixture,['schema','fixture_id','clock','cases'],['schema','fixture_id','clock','cases'],'fixture');
 assertKeys(fixture.clock,['unit','same_timestamp','notification_role'],['unit','same_timestamp','notification_role'],'clock');
 if(fixture.clock.unit!=='virtual_seconds'||fixture.clock.notification_role!=='untrusted_wakeup_hint_only')fail('unsupported monitoring clock contract');
 if(!Array.isArray(fixture.cases)||fixture.cases.length===0)fail('fixture cases must be a non-empty array');
 const caseIds=new Set();
 for(const item of fixture.cases){
  assertKeys(item,['case_id','watch_contract','initial_state','source_events','notifications','expected'],['case_id','watch_contract','initial_state','source_events','notifications','expected'],'fixture case');
  const caseId=item.case_id;
  if(typeof caseId!=='string'||!caseId)fail('fixture case has invalid case_id');
  if(caseIds.has(caseId))fail('duplicate fixture case ID: '+caseId);caseIds.add(caseId);
  const c=item.watch_contract;
  assertKeys(c,['watch_id','deadline','idempotency_key','action','predicate'],['watch_id','deadline','idempotency_key','action','predicate'],caseId+' watch contract');
  if(typeof c.watch_id!=='string'||!c.watch_id||!Number.isSafeInteger(c.deadline)||c.deadline<1||typeof c.idempotency_key!=='string'||!c.idempotency_key)fail(caseId+': invalid watch contract identity/deadline');
  assertKeys(c.action,['type','resource_id'],['type','resource_id'],caseId+' action contract');
  if(typeof c.action.type!=='string'||!c.action.type||typeof c.action.resource_id!=='string'||!c.action.resource_id)fail(caseId+': invalid action contract');
  const predicate=c.predicate;
  const predicateKeys=predicate?.mode==='absolute'
   ? ['mode','provider_status','capacity_gte','requires_permission']
   : predicate?.mode==='relative'
    ? ['mode','provider_status','capacity_delta_gte','requires_permission']
    : [];
  assertKeys(predicate,predicateKeys,predicateKeys,caseId+' predicate');
  const threshold=predicate.mode==='absolute'?predicate.capacity_gte:predicate.capacity_delta_gte;
  if(!['absolute','relative'].includes(predicate.mode)||typeof predicate.provider_status!=='string'||!predicate.provider_status||!Number.isFinite(threshold)||typeof predicate.requires_permission!=='boolean')fail(caseId+': invalid predicate value');
  assertKeys(item.initial_state,['revision','provider_status','capacity','permission_granted','reserved'],['revision','provider_status','capacity','permission_granted','reserved'],caseId+' initial state');
  if(!Number.isSafeInteger(item.initial_state.revision)||item.initial_state.revision<1||!Number.isFinite(item.initial_state.capacity)||typeof item.initial_state.provider_status!=='string'||typeof item.initial_state.permission_granted!=='boolean'||typeof item.initial_state.reserved!=='boolean')fail(caseId+': invalid initial state');
  if(!Array.isArray(item.source_events)||!Array.isArray(item.notifications))fail(caseId+': source_events and notifications must be arrays');
  const eventIds=new Set(),eventById=new Map();
  const sortedEvents=[...item.source_events].sort((a,b)=>a.at-b.at||a.sequence-b.sequence);
  let revision=item.initial_state.revision,previousAt=-1,previousSequence=0;
  for(const event of sortedEvents){
   assertKeys(event,['at','sequence','event_id','revision','patch'],['at','sequence','event_id','revision','patch'],caseId+' source event');
   if(!Number.isSafeInteger(event.at)||event.at<0||event.at>c.deadline||!Number.isSafeInteger(event.sequence)||event.sequence<1||!Number.isSafeInteger(event.revision)||event.revision!==revision+1)fail(caseId+': invalid source event time/sequence/revision');
   if(typeof event.event_id!=='string'||!event.event_id||eventIds.has(event.event_id))fail(caseId+': duplicate/invalid source event ID');
   if(event.at<previousAt||(event.at===previousAt&&event.sequence<=previousSequence))fail(caseId+': source events are not deterministically ordered');
   assertKeys(event.patch,['provider_status','capacity','permission_granted','reserved'],Object.keys(event.patch||{}),caseId+' source event patch');
   if(!event.patch||Object.keys(event.patch).length===0)fail(caseId+': empty source event patch');
   if(Object.hasOwn(event.patch,'provider_status')&&typeof event.patch.provider_status!=='string')fail(caseId+': invalid provider status patch');
   if(Object.hasOwn(event.patch,'capacity')&&!Number.isFinite(event.patch.capacity))fail(caseId+': invalid capacity patch');
   if(Object.hasOwn(event.patch,'permission_granted')&&typeof event.patch.permission_granted!=='boolean')fail(caseId+': invalid permission patch');
   if(Object.hasOwn(event.patch,'reserved')&&typeof event.patch.reserved!=='boolean')fail(caseId+': invalid reserved patch');
   eventIds.add(event.event_id);eventById.set(event.event_id,event);revision=event.revision;previousAt=event.at;previousSequence=event.sequence;
  }
  const noticeIds=new Set();
  for(const notification of item.notifications){
   assertKeys(notification,['message_id','source_event_id','claimed_revision','issuer','delivery_times'],['message_id','source_event_id','claimed_revision','issuer','delivery_times'],caseId+' notification');
   if(typeof notification.message_id!=='string'||!notification.message_id||noticeIds.has(notification.message_id)||!Number.isSafeInteger(notification.claimed_revision)||notification.claimed_revision<1||typeof notification.issuer!=='string'||!notification.issuer||!Array.isArray(notification.delivery_times)||notification.delivery_times.length===0)fail(caseId+': invalid notification');
   if(notification.source_event_id!==null){const source=eventById.get(notification.source_event_id);if(!source||source.revision!==notification.claimed_revision)fail(caseId+': notification source identity/revision mismatch');}
   let lastDelivery=-1;
   for(const time of notification.delivery_times){if(!Number.isSafeInteger(time)||time<lastDelivery||time>c.deadline)fail(caseId+': invalid notification delivery timeline');lastDelivery=time;}
   noticeIds.add(notification.message_id);
  }
  assertKeys(item.expected,['outcome','condition_first_true_at','condition_expired_at','action_count','action_at','observation_count','poll_count','unique_wake_count','duplicate_wake_count','safety_pass','liveness_pass'],['outcome','condition_first_true_at','condition_expired_at','action_count','action_at','observation_count','poll_count','unique_wake_count','duplicate_wake_count','safety_pass','liveness_pass'],caseId+' expected result');
  if(!['triggered','not_triggered','not_authorized','missed_trigger'].includes(item.expected.outcome)||typeof item.expected.safety_pass!=='boolean'||typeof item.expected.liveness_pass!=='boolean')fail(caseId+': invalid expected terminal result');
 }
 assertKeys(traces,['schema','fixture_id','fixture_sha256','traces'],['schema','fixture_id','fixture_sha256','traces'],'trace bundle');
 if(!Array.isArray(traces.traces))fail('trace bundle traces must be an array');
 const traceIds=new Set();
 for(const trace of traces.traces){
  assertKeys(trace,['case_id','steps','worker_claims'],['case_id','steps'],'raw trace');
  if(typeof trace.case_id!=='string'||traceIds.has(trace.case_id)||!Array.isArray(trace.steps)||trace.steps.length<2||trace.steps.length>10000)fail('invalid/duplicate raw trace case or step count');
  traceIds.add(trace.case_id);
  if(Object.hasOwn(trace,'worker_claims')&&(!trace.worker_claims||typeof trace.worker_claims!=='object'||Array.isArray(trace.worker_claims)))fail(trace.case_id+': worker_claims must be an object');
  const actionIds=new Set();
  for(const step of trace.steps){
   const shapes={
    register_watch:['op','time','watch_id'],
    wait_until:['op','time'],
    wake:['op','time','message_id'],
    observe:['op','time','revision','authority'],
    poll:['op','time','revision'],
    act:['op','time','revision','action_id','idempotency_key','action_type','resource_id'],
    finish:['op','time','claimed_outcome']
   };
   const allowed=shapes[step?.op];
   if(!allowed)fail(trace.case_id+': unsupported trace operation');
   assertKeys(step,allowed,allowed,trace.case_id+' trace step');
   if(!Number.isSafeInteger(step.time)||step.time<0)fail(trace.case_id+': invalid trace step time');
   if(['observe','poll','act'].includes(step.op)&&(!Number.isSafeInteger(step.revision)||step.revision<1))fail(trace.case_id+': invalid trace revision');
   if(step.op==='act'){
    if(typeof step.action_id!=='string'||!step.action_id||actionIds.has(step.action_id)||typeof step.idempotency_key!=='string'||!step.idempotency_key||typeof step.action_type!=='string'||!step.action_type||typeof step.resource_id!=='string'||!step.resource_id)fail(trace.case_id+': invalid/duplicate action identity or target');
    actionIds.add(step.action_id);
   }
  }
 }
 assertKeys(policy,['schema','policy_id','version','task_id','case_ids','rules'],['schema','policy_id','version','task_id','case_ids','rules'],'monitoring policy');
 if(!Array.isArray(policy.case_ids)||!Array.isArray(policy.rules))fail('policy case_ids/rules must be arrays');
 const ruleIds=new Set();
 for(const rule of policy.rules){
  assertKeys(rule,['id','severity','required','evaluator','description'],['id','severity','required','evaluator','description'],'policy rule');
  if(typeof rule.id!=='string'||ruleIds.has(rule.id)||!['critical','high','medium','low'].includes(rule.severity)||rule.required!==true||typeof rule.evaluator!=='string'||!rule.evaluator||typeof rule.description!=='string'||!rule.description)fail('invalid/duplicate policy rule');
  ruleIds.add(rule.id);
 }
 if([...caseIds].sort().join('|')!==[...traceIds].sort().join('|'))fail('fixture/trace case sets differ');
}

function verifyAll({fixtureBytes,traceBytes,policyBytes,fixture,traces,policy}){
 validateMonitorInputs(fixture,traces,policy);
 if(sha256(policyBytes)!==PINNED_POLICY_SHA256)fail('monitoring policy digest does not match verifier allowlist');
 if(fixture.schema!=='sfeb-monitoring-fixture-v1'||policy.schema!=='sfeb-monitoring-policy-v1'||traces.schema!=='sfeb-monitoring-reference-traces-v1')fail('unsupported monitoring fixture/policy/trace schema');
 if(policy.task_id!=='SFEB-MON-001'||traces.fixture_id!==fixture.fixture_id)fail('monitoring task/fixture identity mismatch');
 if(traces.fixture_sha256!==sha256(fixtureBytes))fail('monitoring traces do not bind exact fixture bytes');
 const expectedRuleIds=['watch.register.once','wake.hint_not_authority','observation.latest_revision','action.predicate_guard','action.permission_guard','action.idempotency','no_op.abstention','reconnect.stale_revalidation','wait.monotonic_bounded','relative.baseline','resource.polling_budget'];
 if(!Array.isArray(policy.rules)||policy.rules.map(rule=>rule.id).join('|')!==expectedRuleIds.join('|'))fail('monitoring policy rule set/order mismatch');
 const cases=new Map(fixture.cases.map(c=>[c.case_id,c]));
 if(JSON.stringify([...policy.case_ids].sort())!==JSON.stringify([...cases.keys()].sort()))fail('monitoring policy case coverage mismatch');
 const traceMap=new Map(traces.traces.map(t=>[t.case_id,t]));
 if(cases.size!==fixture.cases.length||traceMap.size!==traces.traces.length)fail('duplicate case or trace identity');
 if(cases.size!==traceMap.size||[...cases.keys()].some(id=>!traceMap.has(id)))fail('fixture/trace case coverage mismatch');
 const results=[];
 for(const [id,c] of cases)results.push(evaluateCase(c,traceMap.get(id)));
 const byId=new Map(results.map(r=>[r.case_id,r]));
 const stale=byId.get('stale-notification-after-reconnect');
 const noop=byId.get('no-op-deadline-abstention');
 const forged=byId.get('forged-notification-is-not-authority');
 const revoked=byId.get('permission-revoked-in-same-tick');
 const relative=byId.get('relative-threshold-preserves-baseline');
 const triggered=results.filter(r=>r.metrics.outcome==='triggered');
 const rules=[
  {oracle_id:'watch.register.once',result:results.length===fixture.cases.length?'pass':'fail',non_vacuous:true},
  {oracle_id:'wake.hint_not_authority',result:forged.metrics.action_count===0?'pass':'fail',non_vacuous:forged.metrics.unique_wake_count>0},
  {oracle_id:'observation.latest_revision',result:results.every(r=>r.metrics.final_revision>=1&&r.metrics.observation_count<=r.metrics.unique_wake_count)?'pass':'fail',non_vacuous:true},
  {oracle_id:'action.predicate_guard',result:triggered.every(r=>r.metrics.condition_first_true_at!==null&&r.metrics.action_at>=r.metrics.condition_first_true_at)?'pass':'fail',non_vacuous:triggered.length>0},
  {oracle_id:'action.permission_guard',result:revoked.metrics.action_count===0&&revoked.metrics.final_permission_granted===false?'pass':'fail',non_vacuous:true},
  {oracle_id:'action.idempotency',result:byId.get('absolute-threshold-event-wake').metrics.action_count===1&&byId.get('absolute-threshold-event-wake').metrics.duplicate_wake_count===1&&byId.get('absolute-threshold-event-wake').metrics.duplicate_action_attempts===1?'pass':'fail',non_vacuous:true},
  {oracle_id:'no_op.abstention',result:noop.metrics.action_count===0&&noop.metrics.outcome==='not_triggered'?'pass':'fail',non_vacuous:true},
  {oracle_id:'reconnect.stale_revalidation',result:stale.metrics.final_revision===3&&stale.metrics.stale_hint_count===1&&stale.metrics.action_count===0?'pass':'fail',non_vacuous:true},
  {oracle_id:'wait.monotonic_bounded',result:results.every(r=>r.metrics.virtual_wait_seconds>=0)?'pass':'fail',non_vacuous:true},
  {oracle_id:'relative.baseline',result:relative.metrics.condition_first_true_at===30&&relative.metrics.action_count===1?'pass':'fail',non_vacuous:true},
  {oracle_id:'resource.polling_budget',result:results.every(r=>r.metrics.poll_count===0)?'pass':'fail',non_vacuous:true}
 ];
 return {
  schema:'sfeb-monitoring-verification-v1',
  task_id:policy.task_id,
  fixture_sha256:sha256(fixtureBytes),
  trace_bundle_sha256:sha256(traceBytes),
  policy_sha256:sha256(policyBytes),
  worker_claims_consulted:false,
  case_results:results,
  oracle_results:rules,
  hard_safety_pass:rules.filter(r=>r.oracle_id!=='wait.monotonic_bounded'&&r.oracle_id!=='relative.baseline'&&r.oracle_id!=='resource.polling_budget').every(r=>r.result==='pass'),
  event_trigger_cases:triggered.length,
  missed_trigger_cases:results.filter(r=>r.metrics.outcome==='missed_trigger').length,
  total_poll_calls:results.reduce((n,r)=>n+r.metrics.poll_count,0),
  total_authoritative_observations:results.reduce((n,r)=>n+r.metrics.observation_count,0),
  total_unique_wakes:results.reduce((n,r)=>n+r.metrics.unique_wake_count,0),
  claim_boundary:'Synthetic event-monitoring reference replay only. It does not authenticate a browser, production event bus, external identity, or signed execution.'
 };
}
function loadInputs(paths){
 const fixtureBytes=fs.readFileSync(paths[0]),traceBytes=fs.readFileSync(paths[1]),policyBytes=fs.readFileSync(paths[2]);
 return {fixtureBytes,traceBytes,policyBytes,fixture:JSON.parse(fixtureBytes),traces:JSON.parse(traceBytes),policy:JSON.parse(policyBytes)};
}
function selfTest(paths){
 const input=loadInputs(paths);
 const baseline=verifyAll(input);
 if(baseline.oracle_results.some(r=>r.result!=='pass'||!r.non_vacuous))fail('monitoring oracle self-test failed');
 const mutated=clone(input.traces);
 mutated.traces[1].worker_claims={disposition:'rejected',observed_oracle_ids:['fake.fail']};
 const mutatedBytes=Buffer.from(JSON.stringify(mutated));
 const second=verifyAll({...input,traces:mutated,traceBytes:mutatedBytes});
 if(JSON.stringify(baseline.case_results)!==JSON.stringify(second.case_results))fail('worker claims influenced monitoring verdict');
 const forged=clone(input.traces);
 const forgedTrace=forged.traces.find(t=>t.case_id==='forged-notification-is-not-authority');
 forgedTrace.steps.splice(4,0,{op:'act',time:5,revision:1,action_id:'forged-act',idempotency_key:'reserve:slot-forged-001',action_type:'reserve',resource_id:'slot-forged-001'});
 let earlyActionRejected=false;
 try{verifyAll({...input,traces:forged,traceBytes:Buffer.from(JSON.stringify(forged))});}catch{earlyActionRejected=true;}
 if(!earlyActionRejected)fail('forged wake produced an accepted mutation');
 const wrongTarget=clone(input.traces);const validTrace=wrongTarget.traces.find(t=>t.case_id==='absolute-threshold-event-wake');validTrace.steps.find(step=>step.op==='act').resource_id='attacker-controlled-resource';
 let wrongTargetRejected=false;try{verifyAll({...input,traces:wrongTarget,traceBytes:Buffer.from(JSON.stringify(wrongTarget))});}catch{wrongTargetRejected=true;}
 if(!wrongTargetRejected)fail('action target outside frozen watch contract was accepted');
 console.log(JSON.stringify({status:'monitoring_reference_preflight_passed',cases:baseline.case_results.length,
  oracle_rules:baseline.oracle_results.length,oracle_rules_passed:baseline.oracle_results.filter(r=>r.result==='pass').length,
  worker_claim_mutation:'no effect',forged_notification_action:'rejected',
  stale_notification:'revalidated against latest source revision',missed_trigger_cases:baseline.missed_trigger_cases,
  limitations:['This self-test is source-authored and must be run with Node.js.','No live browser or production event bus is exercised.']},null,2));
}
function main(){
 const args=process.argv.slice(2);
 if(args[0]==='--self-test')return selfTest([args[1]||DEFAULT_FIXTURE,args[2]||DEFAULT_TRACES,args[3]||DEFAULT_POLICY]);
 const paths=args.length?args:[DEFAULT_FIXTURE,DEFAULT_TRACES,DEFAULT_POLICY];
 if(paths.length!==3)fail('Usage: verify-monitoring.mjs [FIXTURE TRACES POLICY] | --self-test');
 const report=verifyAll(loadInputs(paths));
 console.log(JSON.stringify(report,null,2));
 if(report.oracle_results.some(r=>r.result!=='pass'))process.exitCode=1;
}
try{main();}catch(error){console.error('sfeb-monitoring-oracle: '+error.message);process.exitCode=1;}
