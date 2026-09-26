#!/usr/bin/env python3
import collections, copy, gzip, hashlib, json, pathlib, sys

CORPUS = pathlib.Path('docs/release/evidence/rob-cell-001a-application-composition-reference-v1.json.gz')
GZIP_SHA = '04d2a0567deb5572360d6cb245c1b3cedf4815b725c42f07631a1d57f84b4889'
JSON_SHA = 'cac94e64bacafd221d6fedf6c535be2842877d84edf02674a9ae32557cf24d4c'
SCHEMA = 'rob-cell-001a-application-composition-reference-v1'
AUTHORITY = 'robot_application_workcell_fleet_composition_only_no_safety_or_execution_authority'
OUTCOMES = ['ApplicationCompositionAdmissible','RobotQualificationBlocked','ToolQualificationBlocked','FacilityConfigurationBlocked','SpatialReferenceBlocked','CollisionSceneBlocked','FixtureStateBlocked','LocalizationBlocked','EnvironmentMapBlocked','ResourceAvailabilityBlocked','InfrastructureCommandBlocked','InfrastructureObservationBlocked','MachineCommandBlocked','MachineObservationBlocked','HandoffIdentityBlocked','CustodyEvidenceBlocked','CommonModeBlocked','ConfigurationCurrentnessBlocked','HumanZoneBlocked','CoordinationUnavailableBlocked','CoordinationFallbackOnly','AuthorityReferenceBlocked','CommissioningBoundaryBlocked','TaskEvidenceBlocked','ProcessQualificationBoundaryBlocked','AuthorityBoundaryBlocked','ExecutionEvidenceBoundaryBlocked','HistoryIntegrityBlocked','ApplicabilityBoundaryBlocked','InfrastructureUseSemanticallyComplete','HandoffSemanticallyComplete','ReassignmentBlocked','ReassignmentAdmissible']

def die(s): raise SystemExit('FAIL_ROB_CELL_001A_REFERENCE '+s)
def canon(d): return (json.dumps(d,sort_keys=True,separators=(',',':'))+'\n').encode()

def expand(c,doc):
    d=copy.deepcopy(doc['common_defaults'])
    d.update(doc['fixed_cell_defaults'] if c['application_kind']=='FIXED_CELL' else doc['mobile_fleet_defaults'])
    allowed=set(d)
    unknown=set(c['overrides'])-allowed
    if unknown: die(f"unknown overrides {c['id']}: {sorted(unknown)}")
    d.update(c['overrides'])
    voc=dict(doc['field_vocabularies']['COMMON']); voc.update(doc['field_vocabularies'][c['application_kind']])
    for k,a in voc.items():
        if k in d and d[k] not in a: die(f"unknown enum {k}={d[k]!r} in {c['id']}")
    for k,v in d.items():
        if k.endswith('_requested') or k.endswith('_required') or k.endswith('_bound') or k.endswith('_current') or k.endswith('_scope_match') or k in {'history_retained','configuration_current','robot_receipt_current','tool_receipt_current','application_commissioning_current'}:
            if type(v) is not bool: die(f"non-bool {k} in {c['id']}")
    for k in ('required_independent_task_trials','observed_independent_task_trials'):
        if type(d[k]) is not int or d[k]<0: die(f"invalid count {k} in {c['id']}")
    return d

def receipt_bad(d,prefix):
    if prefix=='application_commissioning':
        return (not d['application_commissioning_receipt_bound'] or d['application_commissioning_result']!='Pass' or not d['application_commissioning_current'] or not d['application_commissioning_scope_match'])
    return (not d[prefix+'_bound'] or d[prefix+'_result']!='Pass' or not d[prefix+'_current'] or not d[prefix+'_scope_match'])

def derive(c,doc):
    d=expand(c,doc); app=c['application_kind']; kind=c['case_kind']
    if not d['history_retained']: return 'HistoryIntegrityBlocked'
    if d['integration_reference_reliance']=='AdmissionClaim' and d['integration_reference_profile'] not in doc['reference_profile_applicability'][app]: return 'ApplicabilityBoundaryBlocked'
    if d['physical_authority_requested'] or d['safety_compliance_claim_requested']: return 'AuthorityBoundaryBlocked'
    if d['manufacturing_process_qualification_claim_requested']: return 'ProcessQualificationBoundaryBlocked'
    if not d['configuration_current']: return 'ConfigurationCurrentnessBlocked'
    if d['facility_configuration_state']!='Current': return 'FacilityConfigurationBlocked'
    if d['spatial_reference_state']!='Current': return 'SpatialReferenceBlocked'
    if receipt_bad(d,'robot_receipt'): return 'RobotQualificationBlocked'
    if d['tool_receipt_required'] and receipt_bad(d,'tool_receipt'): return 'ToolQualificationBlocked'
    if d['human_zone_state'] not in ('Clear','NotApplicable'): return 'HumanZoneBlocked'
    if d['external_authority_ref_state'] not in ('Current','NotRequired'): return 'AuthorityReferenceBlocked'
    if d['application_commissioning_required'] and receipt_bad(d,'application_commissioning'): return 'CommissioningBoundaryBlocked'
    if app=='FIXED_CELL':
        if d['collision_scene_state']!='Current': return 'CollisionSceneBlocked'
        if d['fixture_machine_state']!='Current': return 'FixtureStateBlocked'
        if kind=='MACHINE':
            if d['machine_command_state'] in ('Rejected','Failed'): return 'MachineCommandBlocked'
            if d['machine_command_state']=='Accepted' and d['machine_observed_cycle_state']!='ObservedComplete': return 'MachineObservationBlocked'
    else:
        if d['localization_state']!='Current': return 'LocalizationBlocked'
        if d['environment_map_state']!='Current': return 'EnvironmentMapBlocked'
        if d['reassignment_requested']:
            vals=[d['alternate_robot_receipt_state'],d['alternate_tool_receipt_state'],d['alternate_configuration_state'],d['alternate_authority_ref_state']]
            return 'ReassignmentAdmissible' if vals==['CurrentPass','CurrentPass','Current','Current'] else 'ReassignmentBlocked'
        if d['redundancy_claim_requested'] and d['common_mode_state']!='Independent': return 'CommonModeBlocked'
        if d['coordination_provider_state']!='Available':
            return 'CoordinationFallbackOnly' if d['local_fallback_receipt_state']=='Current' else 'CoordinationUnavailableBlocked'
        if kind=='INFRASTRUCTURE':
            if d['required_resource_state']!='Available': return 'ResourceAvailabilityBlocked'
            if d['infrastructure_command_state'] in ('Rejected','Failed'): return 'InfrastructureCommandBlocked'
            if d['infrastructure_command_state']=='Accepted':
                return 'InfrastructureUseSemanticallyComplete' if d['infrastructure_observed_state']=='ObservedReady' else 'InfrastructureObservationBlocked'
        if kind=='HANDOFF':
            if d['handoff_article_identity']!='Verified' or d['handoff_receiver_identity']!='Verified': return 'HandoffIdentityBlocked'
            if d['handoff_transfer_state']!='Accepted' or d['handoff_custody_observation']!='ObservedTransferred': return 'CustodyEvidenceBlocked'
            return 'HandoffSemanticallyComplete'
    if d['task_evidence_actual_plane']!=d['task_evidence_required_plane'] or not d['task_evidence_current'] or not d['task_evidence_scope_match'] or d['observed_independent_task_trials']<d['required_independent_task_trials']: return 'TaskEvidenceBlocked'
    if d['execution_claim_requested'] and d['measured_execution_state']!='ObservedSuccess': return 'ExecutionEvidenceBoundaryBlocked'
    return 'ApplicationCompositionAdmissible'

def main():
    packed=CORPUS.read_bytes()
    if hashlib.sha256(packed).hexdigest()!=GZIP_SHA: die('gzip digest drift')
    try: raw=gzip.decompress(packed)
    except Exception as e: die('invalid gzip '+repr(e))
    if hashlib.sha256(raw).hexdigest()!=JSON_SHA: die('decompressed digest drift')
    doc=json.loads(raw)
    if canon(doc)!=raw: die('non-canonical json')
    if doc.get('schema')!=SCHEMA or doc.get('authority')!=AUTHORITY or doc.get('issue')!=6039: die('identity drift')
    if doc.get('base')!={'branch':'main','head':'eae17187e199e3a53d108b437c0215b5ff812261'}: die('base drift')
    if doc.get('outcomes')!=OUTCOMES: die('outcome vocabulary drift')
    if doc.get('qualification_result_vocabulary')!=['Pass','Fail','Blocked','EnvironmentFailure']: die('receipt vocabulary drift')
    if doc.get('reference_profile_applicability')!={'FIXED_CELL':['None','ISO10218_2_2025'],'MOBILE_FLEET':['None','ISO3691_4_2023']}: die('reference applicability drift')
    cases=doc.get('cases',[])
    if len(cases)!=51 or len({c.get('id') for c in cases})!=51: die('case census drift')
    if cases[0]['id']!='C01_fixed_cell_complete_profile' or cases[-1]['id']!='C51_infrastructure_command_failed': die('case order anchors drift')
    allowed={'id','application_kind','case_kind','expected','overrides'}; derived=[]
    for c in cases:
        if set(c)!=allowed: die('case shape drift '+c.get('id','?'))
        if c['application_kind'] not in ('FIXED_CELL','MOBILE_FLEET') or c['case_kind'] not in ('APPLICATION','MACHINE','INFRASTRUCTURE','HANDOFF','REASSIGNMENT'): die('case kind drift '+c['id'])
        got=derive(c,doc)
        if got!=c['expected']: die(f"{c['id']} expected {c['expected']} derived {got}")
        derived.append(got)
    if set(derived)!=set(OUTCOMES): die('not every disposition exercised')
    # Independent hostile controls ensure source labels cannot drive the oracle.
    probes=[
      ({'application_kind':'FIXED_CELL','case_kind':'APPLICATION','id':'P','expected':'','overrides':{'physical_authority_requested':True}},'AuthorityBoundaryBlocked'),
      ({'application_kind':'FIXED_CELL','case_kind':'APPLICATION','id':'P','expected':'','overrides':{'integration_reference_reliance':'AdmissionClaim','integration_reference_profile':'ISO3691_4_2023'}},'ApplicabilityBoundaryBlocked'),
      ({'application_kind':'MOBILE_FLEET','case_kind':'APPLICATION','id':'P','expected':'','overrides':{'coordination_provider_state':'Unavailable','local_fallback_receipt_state':'Current'}},'CoordinationFallbackOnly'),
      ({'application_kind':'MOBILE_FLEET','case_kind':'INFRASTRUCTURE','id':'P','expected':'','overrides':{'infrastructure_command_state':'Accepted','infrastructure_observed_state':'ObservedNotReady'}},'InfrastructureObservationBlocked'),
      ({'application_kind':'MOBILE_FLEET','case_kind':'HANDOFF','id':'P','expected':'','overrides':{'handoff_article_identity':'Verified','handoff_receiver_identity':'Verified','handoff_transfer_state':'Accepted','handoff_custody_observation':'Unknown'}},'CustodyEvidenceBlocked')]
    for c,w in probes:
        g=derive(c,doc)
        if g!=w: die(f'hostile control expected {w} got {g}')
    census=dict(collections.Counter(derived))
    print('PASS_ROB_CELL_001A_REFERENCE digest='+JSON_SHA+' cases=51 outcomes=33 census='+json.dumps(census,sort_keys=True,separators=(',',':')))

if __name__=='__main__': main()
