#!/usr/bin/env node
// SFEB source-capability contract compiler; in-memory reference only.
import assert from 'node:assert/strict';
const KEYS=['source_id','delivery','durable_log','replay_from_cursor','partitioned_sequence','high_watermark','stable_event_identity','signed_events','signature_verification','issuer_scope_check','key_revocation_check','permission_revision_at_sink','replay_retention_seconds','max_expected_disconnect_seconds','poll_interval_seconds'];
const REQUIRED=KEYS.filter(k=>k!=='poll_interval_seconds');
const fail=(code,msg)=>{const e=new Error(msg);e.code=code;throw e;};
function classify(p){
 if(!p||typeof p!=='object'||Array.isArray(p))fail('INVALID_PROFILE','profile must be object');
 for(const k of Object.keys(p))if(!KEYS.includes(k))fail('UNKNOWN_FIELD','unknown field '+k);
 for(const k of REQUIRED)if(!Object.hasOwn(p,k))fail('MISSING_FIELD','missing '+k);
 if(typeof p.source_id!=='string'||!p.source_id||!['poll','push_best_effort','push_durable'].includes(p.delivery))fail('INVALID_PROFILE','invalid source/delivery');
 for(const k of ['durable_log','replay_from_cursor','partitioned_sequence','high_watermark','stable_event_identity','signed_events','signature_verification','issuer_scope_check','key_revocation_check','permission_revision_at_sink'])if(typeof p[k]!=='boolean')fail('INVALID_PROFILE',k+' must be boolean');
 for(const k of ['replay_retention_seconds','max_expected_disconnect_seconds'])if(!Number.isFinite(p[k])||p[k]<0)fail('INVALID_PROFILE',k+' must be non-negative');
 if(p.delivery==='poll'&&(!Number.isFinite(p.poll_interval_seconds)||p.poll_interval_seconds<=0))fail('INVALID_PROFILE','poll requires positive interval');
 if(p.delivery!=='poll'&&Object.hasOwn(p,'poll_interval_seconds'))fail('INVALID_PROFILE','poll interval only valid on poll source');
 const replay=p.delivery==='push_durable'&&p.durable_log&&p.replay_from_cursor&&p.partitioned_sequence&&p.high_watermark&&p.stable_event_identity;
 const retention=replay&&p.replay_retention_seconds>=p.max_expected_disconnect_seconds;
 const provenance=p.signed_events&&p.signature_verification&&p.issuer_scope_check&&p.key_revocation_check;
 const authority=p.permission_revision_at_sink;
 const edge=p.delivery==='poll'?'not_supported':!replay?'best_effort_wakeup_only':!retention?'replayable_but_retention_insufficient':!provenance?'bounded_replay_untrusted_provenance':'bounded_replay_with_authenticated_provenance';
 return {source_id:p.source_id,level_monitoring:p.delivery==='poll'?'polling_supported':'push_wakeup_supported',edge_recovery:edge,replay_contract_complete:replay,replay_retention_adequate:!!retention,event_provenance_authenticated:!!provenance,action_authority:authority?'sink_rechecks_current_permission_revision':'current_authority_not_enforced_at_sink',strongest_supported_claim:edge==='bounded_replay_with_authenticated_provenance'&&authority?'bounded_edge_recovery_with_authenticated_provenance_and_live_authority':edge};
}
const base={source_id:'urn:source:demo',delivery:'push_durable',durable_log:true,replay_from_cursor:true,partitioned_sequence:true,high_watermark:true,stable_event_identity:true,signed_events:true,signature_verification:true,issuer_scope_check:true,key_revocation_check:true,permission_revision_at_sink:true,replay_retention_seconds:86400,max_expected_disconnect_seconds:3600};
const cases=[];function test(name,fn){fn();cases.push({name,result:'pass'});}function expect(code,fn){assert.throws(fn,e=>e&&e.code===code,'expected '+code);}
test('poll only cannot guarantee transient edge recovery',()=>assert.equal(classify({...base,delivery:'poll',durable_log:false,replay_from_cursor:false,partitioned_sequence:false,high_watermark:false,signed_events:false,signature_verification:false,issuer_scope_check:false,key_revocation_check:false,poll_interval_seconds:5}).edge_recovery,'not_supported'));
test('best effort push is only a wake hint',()=>assert.equal(classify({...base,delivery:'push_best_effort',durable_log:false,replay_from_cursor:false,partitioned_sequence:false,high_watermark:false}).edge_recovery,'best_effort_wakeup_only'));
test('complete durable replay supports bounded authenticated recovery',()=>assert.equal(classify(base).strongest_supported_claim,'bounded_edge_recovery_with_authenticated_provenance_and_live_authority'));
test('retention shorter than disconnect horizon degrades claim',()=>assert.equal(classify({...base,replay_retention_seconds:120}).edge_recovery,'replayable_but_retention_insufficient'));
test('missing partition sequence or high watermark disables replay claim',()=>assert.equal(classify({...base,partitioned_sequence:false,high_watermark:false}).replay_contract_complete,false));
test('issuer scope and key revocation checks are required for provenance',()=>assert.equal(classify({...base,issuer_scope_check:false,key_revocation_check:false}).event_provenance_authenticated,false));
test('signature presence without verification is not authenticated provenance',()=>assert.equal(classify({...base,signature_verification:false}).edge_recovery,'bounded_replay_untrusted_provenance'));
test('current sink permission revision is a separate requirement',()=>assert.equal(classify({...base,permission_revision_at_sink:false}).action_authority,'current_authority_not_enforced_at_sink'));
test('poll interval cannot be attached to a push source',()=>expect('INVALID_PROFILE',()=>classify({...base,poll_interval_seconds:1})));
test('unknown fields fail closed',()=>expect('UNKNOWN_FIELD',()=>classify({...base,magic_exactly_once:true})));
test('missing retention is not defaulted',()=>{const p={...base};delete p.replay_retention_seconds;expect('MISSING_FIELD',()=>classify(p));});
test('negative retention is invalid',()=>expect('INVALID_PROFILE',()=>classify({...base,replay_retention_seconds:-1})));
test('signed best-effort push does not imply event recall',()=>assert.equal(classify({...base,delivery:'push_best_effort',durable_log:false,replay_from_cursor:false,partitioned_sequence:false,high_watermark:false}).edge_recovery,'best_effort_wakeup_only'));
test('high watermark without stable event identity is incomplete',()=>assert.equal(classify({...base,stable_event_identity:false}).replay_contract_complete,false));
test('invalid poll interval fails closed',()=>expect('INVALID_PROFILE',()=>classify({...base,delivery:'poll',poll_interval_seconds:0})));
console.log(JSON.stringify({schema:'sfeb-mon003-source-capability-report-v1',status:'passed',deterministic_cases:cases.length,cases,invariants:{no_recall_claim_without_replay:'pass',retention_horizon_explicit:'pass',signature_verification_and_trust_profile_required:'pass',sink_authority_separate:'pass',invalid_profiles_fail_closed:'pass'},limitations:['In-memory profile classifier; no live source probe, authentication or production adapter is exercised.','Capability declarations are integration-owner inputs and can drift; production use needs conformance probes and signed profile/version receipts.']},null,2));
