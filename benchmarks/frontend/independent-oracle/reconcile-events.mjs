#!/usr/bin/env node
// Independent JavaScript reference for SFEB-PILOT-INT-004.
// Reads only frozen fixtures, never candidate UI claims.
import fs from 'node:fs';
import crypto from 'node:crypto';
const MAX_SAFE_INTEGER=Number.MAX_SAFE_INTEGER,MAX_REORDER_WINDOW=1024,MAX_BUFFERED_EVENTS=1024;
const fixturePath = process.argv[2] ?? 'benchmarks/frontend/long-horizon/event-reconciliation-v1.json';
const fixtureBytes = fs.readFileSync(fixturePath);
const fixture = JSON.parse(fixtureBytes.toString('utf8'));
if (fixture.schema !== 'sfeb-event-reconciliation-fixture-v1') throw new Error('unsupported fixture schema');
for (const key of ['events','cases','metamorphic_pairs','negative_controls']) if (!Array.isArray(fixture[key])) throw new Error(`fixture ${key} missing`);
const clone = value => JSON.parse(JSON.stringify(value));
const transitionAllowed = (from,to) => ({open:['in_progress','blocked'],in_progress:['blocked','done'],blocked:['in_progress','done'],done:[]})[from]?.includes(to) ?? false;
function newState(initial) {
  const jobs=new Map(), permissions=new Map();
  for (const job of initial.jobs) { if (!job.id || jobs.has(job.id) || !Number.isSafeInteger(job.version) || job.version<1) throw new Error('invalid initial job'); jobs.set(job.id,clone(job)); }
  for (const entry of initial.permissions) { if (permissions.has(entry.actor_id)) throw new Error('duplicate permission actor'); permissions.set(entry.actor_id,new Set(entry.capabilities)); }
  return {jobs,permissions,conflicts:new Map(),pending:new Map(),actionResults:new Map(),seenById:new Map(),sequenceOwner:new Map(),buffered:new Map(),lastSequence:0,audit:[],duplicateDeliveries:0};
}
const hasPermission=(s,a,c)=>s.permissions.get(a)?.has(c)??false;
function rejectPending(s,id,reason){s.pending.delete(id);s.actionResults.set(id,{status:'rejected',reason});}
function rejectStaleForJob(s,id){const job=s.jobs.get(id);for(const [key,a] of s.pending)if(a.job_id===id&&a.expected_version!==job.version)rejectPending(s,key,'stale_authoritative_state');}
function submit(s,a){
 if(!a.action_id||s.actionResults.has(a.action_id))throw new Error(`duplicate action id: ${a.action_id}`);
 const job=s.jobs.get(a.job_id);let reason=null;
 if(!job)reason='unknown_job';else if(s.conflicts.has(a.job_id))reason='clarification_required';else if(job.version!==a.expected_version)reason='stale_version';else if(job.status!==a.from_status)reason='stale_status';else if(!hasPermission(s,a.actor_id,a.capability))reason='permission_denied';else if(!transitionAllowed(job.status,a.to_status))reason='invalid_transition';
 if(reason){s.actionResults.set(a.action_id,{status:'rejected',reason});return;}
 s.pending.set(a.action_id,{...a});s.actionResults.set(a.action_id,{status:'pending'});
}
function applyEvent(s,e){
 switch(e.type){
 case 'job_transition':{const j=s.jobs.get(e.job_id);if(!j||j.version!==e.expected_version||j.status!==e.from_status||e.next_version!==e.expected_version+1)throw new Error(`${e.event_id}: stale transition`);if(!transitionAllowed(j.status,e.to_status))throw new Error(`${e.event_id}: illegal authoritative transition`);if(!hasPermission(s,e.actor_id,e.capability))throw new Error(`${e.event_id}: unauthorized actor`);if(e.action_id){const p=s.pending.get(e.action_id);if(!p||p.job_id!==e.job_id||p.expected_version!==e.expected_version||p.to_status!==e.to_status||p.actor_id!==e.actor_id)throw new Error(`${e.event_id}: acknowledgement mismatch`);s.pending.delete(e.action_id);s.actionResults.set(e.action_id,{status:'accepted'});}j.status=e.to_status;j.version=e.next_version;rejectStaleForJob(s,e.job_id);break;}
 case 'owner_changed':{const j=s.jobs.get(e.job_id);if(!j||j.version!==e.expected_version||j.owner!==e.from_owner||e.next_version!==e.expected_version+1)throw new Error(`${e.event_id}: stale owner change`);if(!hasPermission(s,e.actor_id,e.capability))throw new Error(`${e.event_id}: unauthorized owner change`);j.owner=e.to_owner;j.version=e.next_version;rejectStaleForJob(s,e.job_id);break;}
 case 'permission_revoked':s.permissions.get(e.actor_id)?.delete(e.capability);for(const [id,a] of s.pending)if(a.actor_id===e.actor_id&&a.capability===e.capability)rejectPending(s,id,'permission_revoked');break;
 case 'clarification_required':if(s.conflicts.has(e.job_id))throw new Error(`${e.event_id}: duplicate conflict`);s.conflicts.set(e.job_id,e.conflict_id);for(const [id,a] of s.pending)if(a.job_id===e.job_id)rejectPending(s,id,'clarification_required');break;
 case 'clarification_resolved':if(s.conflicts.get(e.job_id)!==e.conflict_id||!e.resolution)throw new Error(`${e.event_id}: resolution mismatch`);s.conflicts.delete(e.job_id);break;
 default:throw new Error(`unsupported event type: ${e.type}`);
 }
}
function cloneState(s){
 return {jobs:new Map([...s.jobs].map(([k,v])=>[k,clone(v)])),permissions:new Map([...s.permissions].map(([k,v])=>[k,new Set(v)])),conflicts:new Map(s.conflicts),pending:new Map([...s.pending].map(([k,v])=>[k,clone(v)])),actionResults:new Map([...s.actionResults].map(([k,v])=>[k,clone(v)])),seenById:new Map([...s.seenById].map(([k,v])=>[k,clone(v)])),sequenceOwner:new Map(s.sequenceOwner),buffered:new Map([...s.buffered].map(([k,v])=>[k,clone(v)])),lastSequence:s.lastSequence,audit:[...s.audit],duplicateDeliveries:s.duplicateDeliveries};
}
function deliver(s,e){const draft=cloneState(s);deliverInto(draft,e);Object.assign(s,draft);}
function validateEventShape(e){
 const kind=e?.type;
 const fields={job_transition:['type','event_id','sequence','job_id','expected_version','next_version','from_status','to_status','actor_id','capability','action_id'],owner_changed:['type','event_id','sequence','job_id','expected_version','next_version','from_owner','to_owner','actor_id','capability'],permission_revoked:['type','event_id','sequence','actor_id','capability'],clarification_required:['type','event_id','sequence','job_id','conflict_id'],clarification_resolved:['type','event_id','sequence','job_id','conflict_id','resolution']}[kind];
 if(!fields)throw new Error(`unsupported event type: ${kind}`);
 if(!e||typeof e!=='object'||Array.isArray(e))throw new Error('event must be a JSON object');
 for(const key of Object.keys(e))if(!fields.includes(key))throw new Error(`unknown event field: ${key}`);
 for(const key of fields)if(!Object.hasOwn(e,key))throw new Error(`missing event field: ${key}`);
 if(typeof e.event_id!=='string'||!e.event_id||e.event_id.length>128||!/^[A-Za-z0-9._-]+$/.test(e.event_id))throw new Error('invalid event identity');
 if(!Number.isSafeInteger(e.sequence)||e.sequence<=0)throw new Error('event sequence is outside the safe positive-integer range');
 for(const key of ['expected_version','next_version'])if(Object.hasOwn(e,key)&&(!Number.isSafeInteger(e[key])||e[key]<=0))throw new Error(`${key} is outside the safe positive-integer range`);
 for(const key of ['job_id','actor_id','capability','from_status','to_status','from_owner','to_owner','conflict_id','resolution'])if(Object.hasOwn(e,key)&&(typeof e[key]!=='string'||!e[key]||e[key].length>128))throw new Error(`invalid event string field: ${key}`);
 if(Object.hasOwn(e,'action_id')&&e.action_id!==null&&(typeof e.action_id!=='string'||!e.action_id||e.action_id.length>128||!/^[A-Za-z0-9._-]+$/.test(e.action_id)))throw new Error('invalid action_id');
}
function deliverInto(s,e){
 const prior=s.seenById.get(e.event_id);if(prior){if(JSON.stringify(prior)!==JSON.stringify(e))throw new Error(`event identity reused with different payload: ${e.event_id}`);s.duplicateDeliveries++;return;}
 validateEventShape(e);
 if(e.sequence<=s.lastSequence)throw new Error(`invalid or replayed sequence ${e.sequence}`);
 if(e.sequence>s.lastSequence+MAX_REORDER_WINDOW)throw new Error(`event sequence exceeds bounded reorder window: ${e.sequence}`);
 if(s.buffered.size>=MAX_BUFFERED_EVENTS)throw new Error('event reorder buffer is full');
 const owner=s.sequenceOwner.get(e.sequence);if(owner)throw new Error(`sequence ${e.sequence} already owned by ${owner}`);
 s.seenById.set(e.event_id,clone(e));s.sequenceOwner.set(e.sequence,e.event_id);s.buffered.set(e.sequence,e);
 while(s.buffered.has(s.lastSequence+1)){const next=s.buffered.get(s.lastSequence+1);applyEvent(s,next);s.buffered.delete(next.sequence);s.lastSequence=next.sequence;s.audit.push(next.event_id);}
}
function snapshot(s){
 const jobs=Object.fromEntries([...s.jobs].sort().map(([k,v])=>[k,clone(v)]));
 const permissions=Object.fromEntries([...s.permissions].sort().map(([k,v])=>[k,[...v].sort()]));
 const conflicts=Object.fromEntries([...s.conflicts].sort());
 const action_results=Object.fromEntries([...s.actionResults].sort().map(([k,v])=>[k,clone(v)]));
 const visible_statuses=Object.fromEntries([...s.jobs.keys()].sort().map(id=>[id,[...s.pending.values()].find(a=>a.job_id===id)?.to_status??s.jobs.get(id).status]));
 return {last_sequence:s.lastSequence,jobs,permissions,conflicts,pending_action_ids:[...s.pending.keys()].sort(),action_results,visible_statuses,audit_event_ids:[...s.audit],duplicate_deliveries:s.duplicateDeliveries,buffered_sequences:[...s.buffered.keys()].sort((a,b)=>a-b)};
}
function assertSubset(actual,expected,path='$'){
 if(expected&&typeof expected==='object'&&!Array.isArray(expected)){if(!actual||typeof actual!=='object'||Array.isArray(actual))throw new Error(`${path}: expected object`);for(const [k,v] of Object.entries(expected)){if(!Object.hasOwn(actual,k))throw new Error(`${path}.${k}: missing`);assertSubset(actual[k],v,`${path}.${k}`);}}
 else if(Array.isArray(expected)){if(JSON.stringify(actual)!==JSON.stringify(expected))throw new Error(`${path}: array mismatch`);}
 else if(actual!==expected)throw new Error(`${path}: expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`);
}
const eventById=new Map(fixture.events.map(e=>[e.event_id,e]));if(eventById.size!==fixture.events.length)throw new Error('duplicate fixture event identity');
function runCase(c){const s=newState(fixture.initial_state);for(const step of c.steps){if(step.op==='submit')submit(s,step.action);else if(step.op==='deliver'){const e=eventById.get(step.event_id);if(!e)throw new Error(`${c.case_id}: unknown event`);deliver(s,e);}else throw new Error(`${c.case_id}: unknown step`);}const out=snapshot(s);assertSubset(out,c.expected);return out;}
const results=new Map(fixture.cases.map(c=>[c.case_id,runCase(c)]));
for(const r of fixture.metamorphic_pairs){if(!results.has(r.left_case)||!results.has(r.right_case))throw new Error(`${r.relation_id}: case missing`);if(JSON.stringify(results.get(r.left_case))!==JSON.stringify(results.get(r.right_case)))throw new Error(`${r.relation_id}: outputs differ`);}
function mustReject(control,fn){try{fn();}catch(error){if(String(error.message).includes(control.expected_error_fragment))return;throw new Error(`${control.control_id}: unexpected rejection: ${error.message}`);}throw new Error(`${control.control_id}: adversarial mutation was accepted`);}
for(const c of fixture.negative_controls){const s=newState(fixture.initial_state),e=eventById.get(c.event_id);if(!e)throw new Error(`${c.control_id}: event missing`);if(c.operation==='alter_duplicate_payload'){deliver(s,e);mustReject(c,()=>deliver(s,{...e,...c.patch}));}else if(c.operation==='reuse_sequence_with_new_id'){deliver(s,e);mustReject(c,()=>deliver(s,{...e,event_id:c.alternate_event_id}));}else if(c.operation==='replace_actor'){deliver(s,eventById.get('evt-001'));deliver(s,eventById.get('evt-002'));mustReject(c,()=>deliver(s,{...e,actor_id:c.actor_id}));}else if(c.operation==='invalid_event_is_rolled_back_atomically'){mustReject(c,()=>deliver(s,{...e,to_status:c.to_status}));deliver(s,e);if(s.lastSequence!==e.sequence||s.audit.at(-1)!==e.event_id)throw new Error(`${c.control_id}: valid retry did not apply after rejection`);}else if(c.operation==='excessive_reorder_window'){mustReject(c,()=>deliver(s,{...e,sequence:c.sequence}));if(s.seenById.has(e.event_id)||s.sequenceOwner.size!==0||s.buffered.size!==0)throw new Error(`${c.control_id}: rejected event mutated stream state`);deliver(s,e);}else if(c.operation==='unknown_event_field'){mustReject(c,()=>deliver(s,{...e,[c.field]:c.value}));}else throw new Error(`${c.control_id}: unsupported control`);}
console.log(JSON.stringify({schema:'sfeb-event-reconciliation-preflight-v1',fixture_id:fixture.fixture_id,fixture_sha256:crypto.createHash('sha256').update(fixtureBytes).digest('hex'),golden_cases:fixture.cases.length,golden_cases_passed:results.size,metamorphic_relations:fixture.metamorphic_pairs.length,metamorphic_relations_passed:fixture.metamorphic_pairs.length,negative_controls:fixture.negative_controls.length,negative_controls_rejected:fixture.negative_controls.length,total_event_deliveries:fixture.cases.reduce((n,c)=>n+c.steps.filter(s=>s.op==='deliver').length,0),status:'reference_model_passed',limitations:['JavaScript reference only; Rust compilation/execution pending.','No Leptos candidate, DOM/browser, keyboard journey, accessibility or isolated network result.','Deterministic schedule is a synthetic benchmark fixture, not a production event service.','Public pilot fixture; not a hidden holdout.']},null,2));
