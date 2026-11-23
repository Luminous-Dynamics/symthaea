#!/usr/bin/env node
// Deterministic SFEB-INT-004 event oracle replay.
// Worker claims are retained for diagnostics but are never read by the verdict path.
import fs from 'node:fs';
import crypto from 'node:crypto';

const DEFAULT_FIXTURE = 'benchmarks/frontend/long-horizon/event-reconciliation-v1.json';
const DEFAULT_POLICY = 'benchmarks/frontend/long-horizon/event-oracle-policy-v1.json';
const DEFAULT_TRACE = 'benchmarks/frontend/long-horizon/examples/late-event-rollback-trace.json';
const PINNED_POLICY_SHA256 = 'dddaaa445881feabc3dc7d3599c8741004917190b2b2467db65cf1c42d7b6d5c';
const MAX_REORDER_WINDOW = 1024;
const MAX_BUFFERED_EVENTS = 1024;

const sha256 = bytes => crypto.createHash('sha256').update(bytes).digest('hex');
const clone = value => JSON.parse(JSON.stringify(value));
const transitionAllowed = (from, to) => ({
  open: ['in_progress', 'blocked'],
  in_progress: ['blocked', 'done'],
  blocked: ['in_progress', 'done'],
  done: []
})[from]?.includes(to) ?? false;

// SFEB's current policy/event values are ASCII. This recursively sorted JSON
// serializer gives stable JCS-compatible bytes for this constrained value domain.
function canonicalJson(value) {
  if (Array.isArray(value)) return '[' + value.map(canonicalJson).join(',') + ']';
  if (value && typeof value === 'object') {
    return '{' + Object.keys(value).sort().map(key => JSON.stringify(key) + ':' + canonicalJson(value[key])).join(',') + '}';
  }
  return JSON.stringify(value);
}
const canonicalDigest = value => sha256(Buffer.from(canonicalJson(value), 'utf8'));

function fail(message) { throw new Error(message); }
function stringField(value, key) {
  if (typeof value?.[key] !== 'string' || !value[key]) fail(`missing or invalid string field: ${key}`);
  return value[key];
}
function validateEventShape(event) {
  const fields = {
    job_transition: ['type','event_id','sequence','job_id','expected_version','next_version','from_status','to_status','actor_id','capability','action_id'],
    owner_changed: ['type','event_id','sequence','job_id','expected_version','next_version','from_owner','to_owner','actor_id','capability'],
    permission_revoked: ['type','event_id','sequence','actor_id','capability'],
    clarification_required: ['type','event_id','sequence','job_id','conflict_id'],
    clarification_resolved: ['type','event_id','sequence','job_id','conflict_id','resolution']
  }[event?.type];
  if (!fields) fail(`unsupported event type: ${event?.type}`);
  if (!event || typeof event !== 'object' || Array.isArray(event)) fail('event must be a JSON object');
  for (const key of Object.keys(event)) if (!fields.includes(key)) fail(`unknown event field: ${key}`);
  for (const key of fields) if (!Object.hasOwn(event, key)) fail(`missing event field: ${key}`);
  if (typeof event.event_id !== 'string' || !/^[A-Za-z0-9._-]{1,128}$/.test(event.event_id)) fail('invalid event identity');
  if (!Number.isSafeInteger(event.sequence) || event.sequence <= 0) fail('event sequence is outside the safe positive-integer range');
  for (const key of ['expected_version','next_version']) {
    if (Object.hasOwn(event, key) && (!Number.isSafeInteger(event[key]) || event[key] <= 0)) fail(`${key} is outside the safe positive-integer range`);
  }
  for (const key of ['job_id','actor_id','capability','from_status','to_status','from_owner','to_owner','conflict_id','resolution']) {
    if (Object.hasOwn(event, key) && (typeof event[key] !== 'string' || !event[key] || event[key].length > 128)) fail(`invalid event string field: ${key}`);
  }
  if (event.action_id !== undefined && event.action_id !== null &&
      (typeof event.action_id !== 'string' || !/^[A-Za-z0-9._-]{1,128}$/.test(event.action_id))) fail('invalid action_id');
}

function newState(initial) {
  const jobs = new Map();
  for (const job of initial.jobs) {
    if (!job.id || jobs.has(job.id) || !Number.isSafeInteger(job.version) || job.version <= 0) fail('invalid initial job');
    jobs.set(job.id, clone(job));
  }
  const permissions = new Map();
  for (const entry of initial.permissions) {
    if (permissions.has(entry.actor_id)) fail('duplicate permission actor');
    permissions.set(entry.actor_id, new Set(entry.capabilities));
  }
  return {jobs,permissions,conflicts:new Map(),pending:new Map(),actions:new Map(),seen:new Map(),
    sequenceOwner:new Map(),buffered:new Map(),lastSequence:0,audit:[],duplicateDeliveries:0};
}
const hasPermission = (state, actor, capability) => state.permissions.get(actor)?.has(capability) ?? false;
function rejectPending(state, id, reason) {
  state.pending.delete(id);
  state.actions.set(id, {status:'rejected',reason});
}
function rejectStaleForJob(state, jobId) {
  const job = state.jobs.get(jobId);
  for (const [id, action] of state.pending) {
    if (action.job_id === jobId && action.expected_version !== job.version) rejectPending(state,id,'stale_authoritative_state');
  }
}
function submitAction(state, action) {
  if (!action.action_id || state.actions.has(action.action_id)) fail('duplicate or empty action ID');
  const job = state.jobs.get(action.job_id);
  let reason = null;
  if (!job) reason = 'unknown_job';
  else if (state.conflicts.has(action.job_id)) reason = 'clarification_required';
  else if (job.version !== action.expected_version) reason = 'stale_version';
  else if (job.status !== action.from_status) reason = 'stale_status';
  else if (!hasPermission(state,action.actor_id,action.capability)) reason = 'permission_denied';
  else if (!transitionAllowed(job.status,action.to_status)) reason = 'invalid_transition';
  if (reason) { state.actions.set(action.action_id,{status:'rejected',reason}); return; }
  state.pending.set(action.action_id,clone(action));
  state.actions.set(action.action_id,{status:'pending'});
}
function applyEvent(state,event) {
  switch (event.type) {
    case 'job_transition': {
      const job = state.jobs.get(event.job_id);
      if (!job || job.version !== event.expected_version || job.status !== event.from_status ||
          event.next_version !== event.expected_version + 1) fail(`${event.event_id}: stale transition`);
      if (!transitionAllowed(job.status,event.to_status)) fail(`${event.event_id}: illegal authoritative transition`);
      if (!hasPermission(state,event.actor_id,event.capability)) fail(`${event.event_id}: unauthorized actor`);
      if (event.action_id !== null) {
        const pending = state.pending.get(event.action_id);
        if (!pending || pending.job_id !== event.job_id || pending.expected_version !== event.expected_version ||
            pending.to_status !== event.to_status || pending.actor_id !== event.actor_id) fail(`${event.event_id}: acknowledgement mismatch`);
        state.pending.delete(event.action_id);
        state.actions.set(event.action_id,{status:'accepted'});
      }
      job.status = event.to_status;
      job.version = event.next_version;
      rejectStaleForJob(state,event.job_id);
      break;
    }
    case 'owner_changed': {
      const job = state.jobs.get(event.job_id);
      if (!job || job.version !== event.expected_version || job.owner !== event.from_owner ||
          event.next_version !== event.expected_version + 1) fail(`${event.event_id}: stale owner change`);
      if (!hasPermission(state,event.actor_id,event.capability)) fail(`${event.event_id}: unauthorized owner change`);
      job.owner = event.to_owner;
      job.version = event.next_version;
      rejectStaleForJob(state,event.job_id);
      break;
    }
    case 'permission_revoked': {
      state.permissions.get(event.actor_id)?.delete(event.capability);
      for (const [id,action] of state.pending) {
        if (action.actor_id === event.actor_id && action.capability === event.capability) rejectPending(state,id,'permission_revoked');
      }
      break;
    }
    case 'clarification_required': {
      if (state.conflicts.has(event.job_id)) fail(`${event.event_id}: duplicate conflict`);
      state.conflicts.set(event.job_id,event.conflict_id);
      for (const [id,action] of state.pending) if (action.job_id === event.job_id) rejectPending(state,id,'clarification_required');
      break;
    }
    case 'clarification_resolved':
      if (state.conflicts.get(event.job_id) !== event.conflict_id || !event.resolution) fail(`${event.event_id}: resolution mismatch`);
      state.conflicts.delete(event.job_id);
      break;
    default: fail(`unsupported event type: ${event.type}`);
  }
}
function cloneState(state) {
  return {
    jobs:new Map([...state.jobs].map(([k,v])=>[k,clone(v)])),
    permissions:new Map([...state.permissions].map(([k,v])=>[k,new Set(v)])),
    conflicts:new Map(state.conflicts),
    pending:new Map([...state.pending].map(([k,v])=>[k,clone(v)])),
    actions:new Map([...state.actions].map(([k,v])=>[k,clone(v)])),
    seen:new Map([...state.seen].map(([k,v])=>[k,clone(v)])),
    sequenceOwner:new Map(state.sequenceOwner),
    buffered:new Map([...state.buffered].map(([k,v])=>[k,clone(v)])),
    lastSequence:state.lastSequence,audit:[...state.audit],duplicateDeliveries:state.duplicateDeliveries
  };
}
function deliver(state,event) {
  const draft = cloneState(state);
  const prior = draft.seen.get(event.event_id);
  if (prior) {
    if (JSON.stringify(prior) !== JSON.stringify(event)) fail(`event identity reused with different payload: ${event.event_id}`);
    draft.duplicateDeliveries++;
    Object.assign(state,draft);
    return;
  }
  validateEventShape(event);
  if (event.sequence <= draft.lastSequence) fail(`invalid or replayed sequence: ${event.sequence}`);
  if (event.sequence > draft.lastSequence + MAX_REORDER_WINDOW) fail(`event sequence exceeds bounded reorder window: ${event.sequence}`);
  if (draft.buffered.size >= MAX_BUFFERED_EVENTS) fail('event reorder buffer is full');
  const owner = draft.sequenceOwner.get(event.sequence);
  if (owner) fail(`sequence ${event.sequence} already owned by ${owner}`);
  draft.seen.set(event.event_id,clone(event));
  draft.sequenceOwner.set(event.sequence,event.event_id);
  draft.buffered.set(event.sequence,clone(event));
  while (draft.buffered.has(draft.lastSequence + 1)) {
    const next = draft.buffered.get(draft.lastSequence + 1);
    applyEvent(draft,next);
    draft.buffered.delete(next.sequence);
    draft.lastSequence = next.sequence;
    draft.audit.push(next.event_id);
  }
  Object.assign(state,draft);
}
function snapshot(state) {
  const visible = Object.fromEntries([...state.jobs.keys()].sort().map(id => [
    id,[...state.pending.values()].find(action=>action.job_id===id)?.to_status ?? state.jobs.get(id).status
  ]));
  return {
    last_sequence:state.lastSequence,
    jobs:Object.fromEntries([...state.jobs].sort()),
    permissions:Object.fromEntries([...state.permissions].sort().map(([k,v])=>[k,[...v].sort()])),
    conflicts:Object.fromEntries([...state.conflicts].sort()),
    pending_action_ids:[...state.pending.keys()].sort(),
    action_results:Object.fromEntries([...state.actions].sort().map(([k,v])=>[k,clone(v)])),
    visible_statuses:visible,audit_event_ids:[...state.audit],
    duplicate_deliveries:state.duplicateDeliveries,
    buffered_sequences:[...state.buffered.keys()].sort((a,b)=>a-b)
  };
}
function assertSubset(actual,expected,path='$') {
  if (expected && typeof expected === 'object' && !Array.isArray(expected)) {
    if (!actual || typeof actual !== 'object' || Array.isArray(actual)) fail(`${path}: expected object`);
    for (const [key,value] of Object.entries(expected)) {
      if (!Object.hasOwn(actual,key)) fail(`${path}.${key}: missing`);
      assertSubset(actual[key],value,`${path}.${key}`);
    }
  } else if (JSON.stringify(actual) !== JSON.stringify(expected)) {
    fail(`${path}: expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`);
  }
}
function verifyTrace({fixture,fixtureBytes,policy,policyBytes,trace,traceBytes}) {
  if (sha256(policyBytes) !== PINNED_POLICY_SHA256) fail('event oracle policy digest does not match verifier allowlist');
  if (fixture.schema !== 'sfeb-event-reconciliation-fixture-v1' || policy.schema !== 'sfeb-event-oracle-policy-v1' ||
      policy.task_id !== 'SFEB-PILOT-INT-004') fail('unsupported fixture or policy schema');
  if (!Array.isArray(policy.rules) || policy.rules.length !== 7) fail('event policy must have seven rules');
  const policyIDs = new Set();
  for (const rule of policy.rules) {
    if (policyIDs.has(rule.id) || rule.required !== true || rule.inconclusive_on_missing !== true) fail(`invalid policy rule: ${rule.id}`);
    policyIDs.add(rule.id);
  }
  if (trace.schema !== 'sfeb-event-execution-trace-v1') fail('unsupported trace schema');
  if (trace.fixture_id !== fixture.fixture_id || trace.fixture_sha256 !== sha256(fixtureBytes)) fail('trace fixture identity/digest mismatch');
  const caseDef = fixture.cases.find(item=>item.case_id===trace.case_id);
  if (!caseDef) fail('trace case is not in frozen fixture');
  if (!Array.isArray(trace.steps) || trace.steps.length !== caseDef.steps.length || trace.steps.length > 10000) fail('trace step coverage mismatch');
  const events = new Map(fixture.events.map(event=>[event.event_id,event]));
  const state = newState(fixture.initial_state);
  const eventIDs = [];
  for (let i=0;i<trace.steps.length;i++) {
    const expected = caseDef.steps[i], observed = trace.steps[i];
    if (expected.op !== observed.op) fail(`step ${i}: operation mismatch`);
    if (expected.op === 'submit') {
      if (JSON.stringify(expected.action) !== JSON.stringify(observed.action)) fail(`step ${i}: action input mismatch`);
      submitAction(state,observed.action);
    } else if (expected.op === 'deliver') {
      const eventID = expected.event_id;
      if (observed.event?.event_id !== eventID || JSON.stringify(events.get(eventID)) !== JSON.stringify(observed.event)) fail(`step ${i}: authoritative event payload mismatch`);
      eventIDs.push(eventID);
      deliver(state,observed.event);
    } else fail(`step ${i}: unsupported operation`);
  }
  const derived = snapshot(state);
  const matches = (()=>{try{assertSubset(derived,caseDef.expected);return true;}catch{return false;}})();
  const sequenceOK = derived.buffered_sequences.length===0 && derived.last_sequence===caseDef.expected.last_sequence;
  const auditUnique = new Set(derived.audit_event_ids).size===derived.audit_event_ids.length;
  const eventTypes = eventIDs.map(id=>events.get(id)?.type);
  const evaluations = {
    'event.schema.strict':true,
    'event.sequence.contiguous':sequenceOK,
    'event.identity.idempotent':auditUnique && derived.duplicate_deliveries===caseDef.expected.duplicate_deliveries,
    'action.optimistic.reconcile':matches,
    'permission.revocation.fail_closed':!eventTypes.includes('permission_revoked')||matches,
    'conflict.clarification.gate':!eventTypes.includes('clarification_required')||matches,
    'audit.history.exactly_once':auditUnique && JSON.stringify(derived.audit_event_ids)===JSON.stringify(caseDef.expected.audit_event_ids)
  };
  const results = policy.rules.map(rule=>{
    const pass=evaluations[rule.id];
    if (typeof pass!=='boolean') fail(`no evaluator for frozen rule ${rule.id}`);
    const evidence = [...new Set(eventIDs)].map(id=>canonicalDigest(events.get(id)));
    const ruleDigest = canonicalDigest(rule);
    return {oracle_id:rule.id,severity:rule.severity,result:pass?'pass':'fail',rule_sha256:ruleDigest,
      evidence_event_sha256:evidence,evidence_artifact_sha256:[],diagnostic_code:pass?'derived_from_raw_event_trace':'semantic_state_mismatch'};
  });
  const decision=results.every(result=>result.result==='pass')?'accepted':'rejected';
  return {
    schema:'sfeb-event-oracle-verification-v1',
    verification_id:`SFEB-VERIFY-${trace.trace_id}`,
    scope:'deterministic_event_semantics_only',
    trace_sha256:sha256(traceBytes),
    fixture_sha256:sha256(fixtureBytes),
    policy_sha256:sha256(policyBytes),
    case_id:trace.case_id,
    semantic_decision:decision,
    worker_claims_consulted:false,
    oracle_results:results,
    derived_state:derived,
    claim_boundary:'This offline evaluator authenticates neither browser execution nor execution identity. It does not produce a signed verifier attestation and does not qualify the UI candidate.'
  };
}
function readInputs(tracePath,fixturePath,policyPath) {
  const traceBytes=fs.readFileSync(tracePath),fixtureBytes=fs.readFileSync(fixturePath),policyBytes=fs.readFileSync(policyPath);
  return {trace:JSON.parse(traceBytes),fixture:JSON.parse(fixtureBytes),policy:JSON.parse(policyBytes),traceBytes,fixtureBytes,policyBytes};
}
function selfTest() {
  const input=readInputs(DEFAULT_TRACE,DEFAULT_FIXTURE,DEFAULT_POLICY);
  const baseline=verifyTrace(input);
  if(baseline.semantic_decision!=='accepted'||baseline.oracle_results.some(item=>item.result!=='pass')) fail('self-test baseline did not pass all seven rules');
  const forged=clone(input);forged.trace.worker_claims={disposition:'rejected',observed_oracle_ids:['fake.failure','fake.security']};
  const forgedReport=verifyTrace({...forged,traceBytes:Buffer.from(JSON.stringify(forged.trace))});
  if(JSON.stringify(baseline.oracle_results)!==JSON.stringify(forgedReport.oracle_results)) fail('worker claims influenced oracle outcomes');
  const altered=clone(input);altered.trace.steps[1].event.to_status='done';
  let mutationRejected=false;
  try { verifyTrace({...altered,traceBytes:Buffer.from(JSON.stringify(altered.trace))}); } catch { mutationRejected=true; }
  if(!mutationRejected) fail('altered authoritative event was accepted');
  console.log(JSON.stringify({status:'offline_event_oracle_self_test_passed',fixture_id:input.fixture.fixture_id,
    policy_rules:baseline.oracle_results.length,policy_rules_passed:baseline.oracle_results.filter(x=>x.result==='pass').length,
    worker_claim_mutation:'no_effect',altered_event_payload:'rejected',
    limitations:['Node.js semantic sub-verifier only; no signed execution identity.','No browser evidence, Leptos candidate or trusted independent workflow.']},null,2));
}
function main() {
  const args=process.argv.slice(2);
  if(args[0]==='--self-test') return selfTest();
  const paths=args.length?args:[DEFAULT_TRACE,DEFAULT_FIXTURE,DEFAULT_POLICY];
  if(paths.length!==3) fail('Usage: verify-event-trace.mjs [TRACE_JSON FIXTURE_JSON POLICY_JSON] | --self-test');
  const report=verifyTrace(readInputs(paths[0],paths[1],paths[2]));
  console.log(JSON.stringify(report,null,2));
  if(report.semantic_decision!=='accepted') process.exitCode=1;
}
try { main(); } catch(error) { console.error(`sfeb-event-oracle: ${error.message}`); process.exitCode=1; }
