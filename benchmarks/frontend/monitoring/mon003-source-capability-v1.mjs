#!/usr/bin/env node
// SFEB-MON-003 source capability contract compiler.
// Classifies which monitoring guarantees an integration can honestly make.
import assert from 'node:assert/strict';

const ALLOWED = [
  'source_id','delivery','durable_log','replay_from_cursor','partitioned_sequence','high_watermark',
  'stable_event_identity','signed_events','issuer_scope_check','key_revocation_check',
  'permission_revision_at_sink','replay_retention_seconds','max_expected_disconnect_seconds',
  'poll_interval_seconds'
];
function fail(code, message) { const e = new Error(message); e.code = code; throw e; }
function classifySource(profile) {
  if (!profile || typeof profile !== 'object' || Array.isArray(profile)) fail('INVALID_PROFILE', 'profile must be an object');
  for (const key of Object.keys(profile)) if (!ALLOWED.includes(key)) fail('UNKNOWN_FIELD', 'unknown source capability: ' + key);
  for (const key of ['source_id','delivery','durable_log','replay_from_cursor','partitioned_sequence','high_watermark','stable_event_identity','signed_events','issuer_scope_check','key_revocation_check','permission_revision_at_sink','replay_retention_seconds','max_expected_disconnect_seconds']) {
    if (!Object.hasOwn(profile,key)) fail('MISSING_FIELD', 'missing source capability: ' + key);
  }
  if (typeof profile.source_id !== 'string' || !profile.source_id || !['poll','push_best_effort','push_durable'].includes(profile.delivery)) fail('INVALID_PROFILE', 'source identity or delivery mode invalid');
  for (const key of ['durable_log','replay_from_cursor','partitioned_sequence','high_watermark','stable_event_identity','signed_events','issuer_scope_check','key_revocation_check','permission_revision_at_sink']) {
    if (typeof profile[key] !== 'boolean') fail('INVALID_PROFILE', key + ' must be boolean');
  }
  for (const key of ['replay_retention_seconds','max_expected_disconnect_seconds']) {
    if (!Number.isFinite(profile[key]) || profile[key] < 0) fail('INVALID_PROFILE', key + ' must be a non-negative number');
  }
  if (profile.delivery === 'poll' && (!Number.isFinite(profile.poll_interval_seconds) || profile.poll_interval_seconds <= 0)) fail('INVALID_PROFILE', 'polling sources require a positive poll_interval_seconds');

  const replayContract = profile.delivery === 'push_durable' && profile.durable_log && profile.replay_from_cursor &&
    profile.partitioned_sequence && profile.high_watermark && profile.stable_event_identity;
  const retentionAdequate = replayContract && profile.replay_retention_seconds >= profile.max_expected_disconnect_seconds;
  const provenanceTrusted = profile.signed_events && profile.issuer_scope_check && profile.key_revocation_check;
  const authorityLive = profile.permission_revision_at_sink;

  let edgeRecovery;
  if (profile.delivery === 'poll') edgeRecovery = 'not_supported';
  else if (!replayContract) edgeRecovery = 'best_effort_wakeup_only';
  else if (!retentionAdequate) edgeRecovery = 'replayable_but_retention_insufficient';
  else if (!provenanceTrusted) edgeRecovery = 'bounded_replay_untrusted_provenance';
  else edgeRecovery = 'bounded_replay_with_authenticated_provenance';

  const levelMonitoring = profile.delivery === 'poll' ? 'polling_supported' : 'push_wakeup_supported';
  const actionAuthority = authorityLive ? 'sink_rechecks_current_permission_revision' : 'current_authority_not_enforced_at_sink';
  const strongestClaim = edgeRecovery === 'bounded_replay_with_authenticated_provenance' && authorityLive
    ? 'bounded_edge_recovery_with_authenticated_provenance_and_live_authority'
    : edgeRecovery;

  return Object.freeze({
    source_id: profile.source_id,
    level_monitoring: levelMonitoring,
    edge_recovery: edgeRecovery,
    replay_contract_complete: replayContract,
    replay_retention_adequate: Boolean(retentionAdequate),
    event_provenance_authenticated: Boolean(provenanceTrusted),
    action_authority: actionAuthority,
    strongest_supported_claim: strongestClaim,
    explicit_limitations: [
      ...(profile.delivery === 'poll' ? ['No event-driven wake guarantee; missed transient events are possible between polls.'] : []),
      ...(!replayContract && profile.delivery !== 'poll' ? ['No durable partition-cursor replay contract; notifications cannot prove missed events did not occur.'] : []),
      ...(replayContract && !retentionAdequate ? ['Expected disconnect can exceed source replay retention.'] : []),
      ...(replayContract && !provenanceTrusted ? ['Replayable events are not fully authenticated under issuer scope and key-revocation checks.'] : []),
      ...(!authorityLive ? ['A previously issued grant may survive a permission change unless the sink checks the current policy revision.'] : []),
    ],
  });
}

const results=[]; const test=(name,fn)=>{fn();results.push({name,result:'pass'});};
const expect=(code,fn)=>assert.throws(fn,e=>e&&e.code===code,'expected '+code);
const base={source_id:'urn:source:demo',delivery:'push_durable',durable_log:true,replay_from_cursor:true,partitioned_sequence:true,high_watermark:true,stable_event_identity:true,signed_events:true,issuer_scope_check:true,key_revocation_check:true,permission_revision_at_sink:true,replay_retention_seconds:86400,max_expected_disconnect_seconds:3600};

test('polling-only source does not claim transient edge recovery',()=>{const x=classifySource({...base,delivery:'poll',durable_log:false,replay_from_cursor:false,partitioned_sequence:false,high_watermark:false,signed_events:false,issuer_scope_check:false,key_revocation_check:false,poll_interval_seconds:5});assert.equal(x.edge_recovery,'not_supported');assert.match(x.explicit_limitations.join(' '),/between polls/);});
test('best-effort push source is only a wake hint, not replayable evidence',()=>{const x=classifySource({...base,delivery:'push_best_effort',durable_log:false,replay_from_cursor:false,partitioned_sequence:false,high_watermark:false});assert.equal(x.edge_recovery,'best_effort_wakeup_only');assert.equal(x.replay_contract_complete,false);});
test('durable partitioned replay with adequate retention supports bounded recovery',()=>{const x=classifySource(base);assert.equal(x.edge_recovery,'bounded_replay_with_authenticated_provenance');assert.equal(x.strongest_supported_claim,'bounded_edge_recovery_with_authenticated_provenance_and_live_authority');});
test('durable replay with insufficient retention cannot claim the outage horizon',()=>{const x=classifySource({...base,replay_retention_seconds:120});assert.equal(x.edge_recovery,'replayable_but_retention_insufficient');assert.equal(x.replay_retention_adequate,false);});
test('durable log without partition sequence and high watermark is not a complete cursor contract',()=>{const x=classifySource({...base,partitioned_sequence:false,high_watermark:false});assert.equal(x.edge_recovery,'best_effort_wakeup_only');assert.equal(x.replay_contract_complete,false);});
test('signature without issuer-scope and key-revocation validation is not trusted provenance',()=>{const x=classifySource({...base,issuer_scope_check:false,key_revocation_check:false});assert.equal(x.edge_recovery,'bounded_replay_untrusted_provenance');assert.equal(x.event_provenance_authenticated,false);});
test('current permission must be rechecked at the sink',()=>{const x=classifySource({...base,permission_revision_at_sink:false});assert.equal(x.action_authority,'current_authority_not_enforced_at_sink');assert.equal(x.strongest_supported_claim,'bounded_replay_with_authenticated_provenance');});
test('unknown source capability fields fail closed',()=>expect('UNKNOWN_FIELD',()=>classifySource({...base,maybe_exactly_once:true})));
test('missing replay-retention value is not silently defaulted',()=>{const x={...base};delete x.replay_retention_seconds;expect('MISSING_FIELD',()=>classifySource(x));});
test('negative replay retention fails validation',()=>expect('INVALID_PROFILE',()=>classifySource({...base,replay_retention_seconds:-1})));
test('push delivery with signed events but no durable replay cannot claim event recall',()=>{const x=classifySource({...base,delivery:'push_best_effort',durable_log:false,replay_from_cursor:false,partitioned_sequence:false,high_watermark:false});assert.equal(x.event_provenance_authenticated,true);assert.equal(x.edge_recovery,'best_effort_wakeup_only');});
test('high watermark alone cannot qualify replay without stable source identity',()=>{const x=classifySource({...base,stable_event_identity:false});assert.equal(x.replay_contract_complete,false);});

console.log(JSON.stringify({schema:'sfeb-mon003-source-capability-report-v1',status:'passed',deterministic_cases:results.length,cases:results,invariants:{no_recall_claim_without_replay:'pass',retention_horizon_is_explicit:'pass',authenticated_provenance_requires_trust_profile:'pass',sink_authority_is_separate_from_event_receipt:'pass',invalid_profiles_fail_closed:'pass'},limitations:['Pure in-memory capability classifier; does not probe or authenticate a live source adapter.','Capability claims depend on the truth of the source profile supplied by an integration owner.','A source may change behavior after classification; production adapters need periodic conformance probes and signed capability/version receipts.']},null,2));
