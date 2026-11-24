#!/usr/bin/env node
// SFEB-MON-003 capability leases: source guarantees expire when their evidence does.
// In-memory reference only; not a production signer, probe or scheduler.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
const canon=v=>v===null||typeof v!=='object'?JSON.stringify(v):Array.isArray(v)?'['+v.map(canon).join(',')+']':'{'+Object.keys(v).sort().map(k=>JSON.stringify(k)+':'+canon(v[k])).join(',')+'}';
const hash=v=>createHash('sha256').update(canon(v)).digest('hex');
const fail=(c,m)=>{const e=new Error(m);e.code=c;throw e;};
const profileHash=p=>hash({sourceId:p.sourceId,connectorVersion:p.connectorVersion,capabilities:p.capabilities});
class Registry{
  constructor(){this.rows=new Map();}
  register(p,a,now){
    if(!p||!a||!Number.isFinite(now))fail('INVALID_INPUT','profile, attestation and time required');
    if(!p.sourceId||!p.connectorVersion||!p.capabilities)fail('INVALID_PROFILE','source, version and capabilities required');
    if(a.sourceId!==p.sourceId)fail('SOURCE_MISMATCH','wrong source');
    if(a.connectorVersion!==p.connectorVersion)fail('VERSION_MISMATCH','wrong connector version');
    if(a.profileDigest!==profileHash(p))fail('PROFILE_MISMATCH','capability digest mismatch');
    if(!a.signatureVerified||!a.issuerScopeValid||!a.keyNotRevoked)fail('UNTRUSTED','signature/issuer/key trust failed');
    if(!a.conformanceProbePassed)fail('PROBE_FAILED','conformance probe failed');
    if(!Number.isFinite(a.issuedAt)||!Number.isFinite(a.expiresAt)||a.issuedAt>now||a.expiresAt<=now||a.expiresAt<=a.issuedAt)fail('BAD_TIME','attestation outside validity window');
    this.rows.set(p.sourceId,{p:structuredClone(p),a:structuredClone(a)});
    return this.check(p.sourceId,now,{});
  }
  update(p){const row=this.rows.get(p.sourceId);if(!row)fail('UNKNOWN_SOURCE','unknown source');row.p=structuredClone(p);}
  check(id,now,obs={}){
    const row=this.rows.get(id);if(!row)fail('UNKNOWN_SOURCE','unknown source');
    const {p,a}=row, c=p.capabilities, reasons=[];
    if(a.sourceId!==id)reasons.push('source_mismatch');
    if(a.connectorVersion!==p.connectorVersion)reasons.push('connector_version_changed');
    if(a.profileDigest!==profileHash(p))reasons.push('profile_changed');
    if(!a.signatureVerified||!a.issuerScopeValid||!a.keyNotRevoked)reasons.push('trust_revoked');
    if(!Number.isFinite(now)||now<a.issuedAt||now>=a.expiresAt)reasons.push('attestation_expired');
    if(obs.replayProbePassed===false)reasons.push('replay_probe_failed');
    if(obs.highWatermarkMonotonic===false)reasons.push('cursor_rollback');
    if(obs.sinkPermissionRevisionEnforced===false)reasons.push('sink_authority_probe_failed');
    const replay=c.delivery==='push_durable'&&c.durableLog&&c.replayFromCursor&&c.partitionedSequence&&c.highWatermark&&c.stableEventIdentity;
    const retention=replay&&c.replayRetentionSeconds>=c.maxExpectedDisconnectSeconds;
    const provenance=c.signedEvents&&a.signatureVerified&&a.issuerScopeValid&&a.keyNotRevoked;
    const authority=c.permissionRevisionAtSink&&obs.sinkPermissionRevisionEnforced!==false;
    if(c.delivery==='poll')reasons.push('poll_only');else if(!replay)reasons.push('no_replay');else if(!retention)reasons.push('retention_short');
    if(!provenance)reasons.push('provenance_untrusted');if(!authority)reasons.push('sink_authority_unverified');
    const blocking=['source_mismatch','connector_version_changed','profile_changed','trust_revoked','attestation_expired','replay_probe_failed','cursor_rollback','sink_authority_probe_failed'].some(x=>reasons.includes(x));
    const edge=c.delivery==='poll'?'not_supported':!replay?'best_effort_wakeup_only':!retention?'replayable_but_retention_insufficient':!provenance?'bounded_replay_untrusted_provenance':'bounded_replay_with_authenticated_provenance';
    const status=blocking?'inconclusive':reasons.length?'degraded':'active';
    const strongest=status!=='active'?'no_unqualified_edge_recall_claim':edge==='bounded_replay_with_authenticated_provenance'&&authority?'bounded_replay_with_authenticated_provenance_and_live_authority':edge;
    return {sourceId:id,status,edgeRecovery:edge,strongestSupportedClaim:strongest,attestationId:a.attestationId,attestationExpiresAt:a.expiresAt,reasons};
  }
}
function makeProfile(over={}){return {sourceId:'urn:service:calendar-A',connectorVersion:'3.4.1',capabilities:{delivery:'push_durable',durableLog:true,replayFromCursor:true,partitionedSequence:true,highWatermark:true,stableEventIdentity:true,signedEvents:true,replayRetentionSeconds:86400,maxExpectedDisconnectSeconds:3600,permissionRevisionAtSink:true},...over};}
function makeAttestation(p,over={}){return {attestationId:'attest-'+p.connectorVersion,sourceId:p.sourceId,connectorVersion:p.connectorVersion,profileDigest:profileHash(p),issuedAt:100,expiresAt:200,signatureVerified:true,issuerScopeValid:true,keyNotRevoked:true,conformanceProbePassed:true,...over};}
const cases=[];const test=(name,fn)=>{fn();cases.push({name,result:'pass'});};const expect=(code,fn)=>assert.throws(fn,e=>e&&e.code===code,'expected '+code);
test('fresh trusted attestation establishes active bounded replay',()=>{const p=makeProfile(),r=new Registry().register(p,makeAttestation(p),110);assert.equal(r.status,'active');assert.equal(r.strongestSupportedClaim,'bounded_replay_with_authenticated_provenance_and_live_authority');});
test('attestation expiry blocks unqualified edge claim',()=>{const p=makeProfile(),g=new Registry();g.register(p,makeAttestation(p),110);const r=g.check(p.sourceId,200);assert.equal(r.status,'inconclusive');assert.ok(r.reasons.includes('attestation_expired'));});
test('connector version upgrade invalidates prior lease',()=>{const p=makeProfile(),g=new Registry();g.register(p,makeAttestation(p),110);g.update({...p,connectorVersion:'3.5.0'});assert.equal(g.check(p.sourceId,120).status,'inconclusive');});
test('capability profile drift invalidates prior attestation',()=>{const p=makeProfile(),g=new Registry();g.register(p,makeAttestation(p),110);g.update({...p,capabilities:{...p.capabilities,replayRetentionSeconds:60}});assert.ok(g.check(p.sourceId,120).reasons.includes('profile_changed'));});
test('failed replay probe downgrades source guarantee',()=>{const p=makeProfile(),g=new Registry();g.register(p,makeAttestation(p),110);assert.equal(g.check(p.sourceId,120,{replayProbePassed:false}).status,'inconclusive');});
test('source high-watermark rollback blocks capability lease',()=>{const p=makeProfile(),g=new Registry();g.register(p,makeAttestation(p),110);assert.ok(g.check(p.sourceId,120,{highWatermarkMonotonic:false}).reasons.includes('cursor_rollback'));});
test('failed sink permission revision probe blocks authority guarantee',()=>{const p=makeProfile(),g=new Registry();g.register(p,makeAttestation(p),110);assert.equal(g.check(p.sourceId,120,{sinkPermissionRevisionEnforced:false}).status,'inconclusive');});
test('retention shorter than disconnect horizon is degraded',()=>{const p=makeProfile({capabilities:{...makeProfile().capabilities,replayRetentionSeconds:300}}),r=new Registry().register(p,makeAttestation(p),110);assert.equal(r.status,'degraded');assert.equal(r.edgeRecovery,'replayable_but_retention_insufficient');});
test('poll-only source does not claim event-driven edge recovery',()=>{const b=makeProfile(),p={...b,capabilities:{...b.capabilities,delivery:'poll',durableLog:false,replayFromCursor:false,partitionedSequence:false,highWatermark:false,signedEvents:false}},r=new Registry().register(p,makeAttestation(p),110);assert.equal(r.status,'degraded');assert.equal(r.edgeRecovery,'not_supported');});
test('revoked signing key prevents capability registration',()=>{const p=makeProfile();expect('UNTRUSTED',()=>new Registry().register(p,makeAttestation(p,{keyNotRevoked:false}),110));});
test('unverified signature prevents capability registration',()=>{const p=makeProfile();expect('UNTRUSTED',()=>new Registry().register(p,makeAttestation(p,{signatureVerified:false}),110));});
test('attestation cannot be transplanted across source identities',()=>{const p=makeProfile();expect('SOURCE_MISMATCH',()=>new Registry().register(p,makeAttestation(p,{sourceId:'urn:attacker'}),110));});
test('attestation cannot be transplanted across connector versions',()=>{const p=makeProfile();expect('VERSION_MISMATCH',()=>new Registry().register(p,makeAttestation(p,{connectorVersion:'1.0'}),110));});
test('fresh attestation is required to restore guarantee after upgrade',()=>{const p=makeProfile(),g=new Registry();g.register(p,makeAttestation(p),110);const up={...p,connectorVersion:'3.5.0'};g.update(up);assert.equal(g.check(p.sourceId,120).status,'inconclusive');assert.equal(g.register(up,makeAttestation(up,{issuedAt:121,expiresAt:221}),122).status,'active');});
test('attestation issued in future or already expired is rejected',()=>{const p=makeProfile();expect('BAD_TIME',()=>new Registry().register(p,makeAttestation(p,{issuedAt:120}),110));expect('BAD_TIME',()=>new Registry().register(p,makeAttestation(p),200));});
console.log(JSON.stringify({schema:'sfeb-mon003-capability-lease-report-v1',status:'passed',deterministic_cases:cases.length,cases,invariants:{capability_is_time_bounded:'pass',connector_or_profile_change_invalidates_attestation:'pass',failed_replay_or_cursor_probe_blocks_edge_guarantee:'pass',key_revocation_and_signature_trust_required:'pass',capability_recovery_requires_fresh_attestation:'pass',downgrade_instead_of_silent_overclaim:'pass'},limitations:['In-memory capability-lease model; no production signature verifier, key registry, live source probe or scheduler is exercised.','Signature, issuer, key and conformance status are trusted test booleans, not cryptographic evidence.','Production adapters must persist and independently verify capability attestations, probes and connector/profile changes.']},null,2));
