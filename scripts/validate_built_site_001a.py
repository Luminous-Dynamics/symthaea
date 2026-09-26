#!/usr/bin/env python3
import copy, gzip, hashlib, json, pathlib, sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
CORPUS = ROOT / 'docs/release/evidence/built-site-001a-terrestrial-site-evidence-reference-v1.json.gz'

GZIP_SHA = 'e7547ba293f362e5b8081232b426947fa6e0b5d297b6572e468907dd6a10d6f5'
JSON_SHA = 'aca82eaa1e3e09a834ad978a262a0e656a793cf461a502258997b79cf7de44a1'
SCHEMA = 'built-site-001a-terrestrial-site-evidence-reference-v1'
AUTHORITY = 'terrestrial_site_geotechnical_evidence_applicability_only_no_design_or_execution_authority'
BASE = {'branch':'main','head':'eae17187e199e3a53d108b437c0215b5ff812261'}
PARENT = {'built_env_issue':6032,'eng_design_issue':6005,'eng_design_trace_issue':6026}
OUTCOMES = [
'SiteEvidenceAdmissible','SiteIdentityBlocked','ObservationIdentityBlocked','SpatialCoverageBlocked',
'DepthCoverageBlocked','ClassificationBoundaryBlocked','SampleChainBlocked','GroundwaterCurrentnessBlocked',
'UnitReferenceBlocked','AssumptionBindingBlocked','InterpolationBoundaryBlocked','UncertaintyBoundaryBlocked',
'ExternalReportBoundaryBlocked','ConfigurationCurrentnessBlocked','PostWorkObservationBlocked',
'ApplicabilityReviewRequired','HistoryIntegrityBlocked','SourceAuthorityBlocked','FavorableDefaultBlocked',
'DesignAuthorityBoundaryBlocked','PhysicalAuthorityBoundaryBlocked']
QRESULTS = ['Pass','Fail','Blocked','EnvironmentFailure']
CASE_IDS = [f'S{i:02d}_' for i in range(1,41)]
EXPECTED_EXTERNAL = [
 {'role':'soil_identification_description_reference','standard':'ISO 14688-1','status_note':'current_as_confirmed_by_iso_2023_review','version':'2017'},
 {'role':'soil_classification_reference','standard':'ISO 14688-2','status_note':'current_as_confirmed_by_iso_2023_review','version':'2017'},
 {'role':'rock_identification_description_classification_reference','standard':'ISO 14689','status_note':'current_as_confirmed_by_iso_2023_review','version':'2017'},
 {'role':'sampling_and_groundwater_measurement_principles_reference','standard':'ISO 22475-1','status_note':'published','version':'2021'},
 {'role':'ground_model_execution_service_life_informative_reference','standard':'Eurocode 7 second-generation JRC guidance','status_note':'informative_guidance_not_jurisdictional_compliance','version':'2024'},
]
EXPECTED_NONCLAIMS = [
'actual_site_suitability','foundation_design','bearing_capacity','settlement_prediction','slope_stability',
'seismic_adequacy','environmental_clearance','code_compliance','permit_or_occupancy_approval',
'construction_recommendation','procurement_or_resource_allocation','physical_execution_authority']

ENUM_FIELDS = {
 'site_boundary_identity','coordinate_reference_state','investigation_identity','observation_provenance',
 'horizontal_coverage','vertical_coverage','classification_context','sample_chain_state','groundwater_state',
 'unit_reference_state','assumption_binding_state','interpolation_state','uncertainty_state',
 'external_report_identity','external_report_currentness','external_report_spatial_match',
 'external_report_configuration_match','post_work_change','post_work_observation_state'
}
BOOL_FIELDS = {
 'consumer_applicability_reviewed','design_claim_requested','external_report_limitations_retained',
 'external_report_required','favorable_default_requested','foundation_consumer_changed','groundwater_required',
 'negative_history_retained','permit_or_execution_claim_requested','site_configuration_current',
 'work_or_provenance_event_only'
}
SITE_CHANGE_REOBS = {'Fill','Excavation','Drainage','UtilityTrench','NeighboringWorks'}


def fail(msg):
    raise AssertionError(msg)

def derive(s):
    if not s['negative_history_retained']:
        return 'HistoryIntegrityBlocked'
    if s['permit_or_execution_claim_requested']:
        return 'PhysicalAuthorityBoundaryBlocked'
    if s['design_claim_requested']:
        return 'DesignAuthorityBoundaryBlocked'
    if s['work_or_provenance_event_only']:
        return 'SourceAuthorityBlocked'
    if s['favorable_default_requested']:
        return 'FavorableDefaultBlocked'
    if s['site_boundary_identity'] != 'Bound' or s['coordinate_reference_state'] != 'Bound':
        return 'SiteIdentityBlocked'
    if s['investigation_identity'] != 'Exact' or s['observation_provenance'] != 'Bound':
        return 'ObservationIdentityBlocked'
    if s['horizontal_coverage'] != 'WithinDeclaredExtent':
        return 'SpatialCoverageBlocked'
    if s['vertical_coverage'] != 'CoversDeclaredZone':
        return 'DepthCoverageBlocked'
    if s['classification_context'] != 'Bound':
        return 'ClassificationBoundaryBlocked'
    if s['sample_chain_state'] != 'Current':
        return 'SampleChainBlocked'
    if s['groundwater_required']:
        if s['groundwater_state'] != 'CurrentObserved':
            return 'GroundwaterCurrentnessBlocked'
    elif s['groundwater_state'] not in ('NotApplicable','CurrentObserved'):
        return 'GroundwaterCurrentnessBlocked'
    if s['unit_reference_state'] != 'Bound':
        return 'UnitReferenceBlocked'
    if s['assumption_binding_state'] != 'Bound':
        return 'AssumptionBindingBlocked'
    if s['interpolation_state'] not in ('DeclaredBounded','NotUsed'):
        return 'InterpolationBoundaryBlocked'
    if s['uncertainty_state'] != 'Declared':
        return 'UncertaintyBoundaryBlocked'
    if s['external_report_required']:
        if not (s['external_report_identity']=='Exact' and s['external_report_currentness']=='Current'
                and s['external_report_spatial_match']=='Exact' and s['external_report_configuration_match']=='Exact'
                and s['external_report_limitations_retained']):
            return 'ExternalReportBoundaryBlocked'
    if not s['site_configuration_current']:
        return 'ConfigurationCurrentnessBlocked'
    if s['post_work_change'] in SITE_CHANGE_REOBS and s['post_work_observation_state'] != 'CurrentObserved':
        return 'PostWorkObservationBlocked'
    if s['foundation_consumer_changed'] and not s['consumer_applicability_reviewed']:
        return 'ApplicabilityReviewRequired'
    return 'SiteEvidenceAdmissible'


def validate_shape(d):
    if d['schema'] != SCHEMA: fail('schema drift')
    if d['authority'] != AUTHORITY: fail('authority drift')
    if d['base'] != BASE: fail('base drift')
    if d['issue'] != 6034: fail('issue drift')
    if d['parent'] != PARENT: fail('parent drift')
    if d['outcomes'] != OUTCOMES: fail('outcome vocabulary/order drift')
    if d['qualification_result_vocabulary'] != QRESULTS: fail('qualification-result vocabulary drift')
    if d['external_alignment'] != EXPECTED_EXTERNAL: fail('external alignment drift')
    if d['nonclaims'] != EXPECTED_NONCLAIMS: fail('nonclaims drift')
    if set(d['field_vocabularies']) != {'SITE'}: fail('field vocabulary scope drift')
    voc = d['field_vocabularies']['SITE']
    if set(voc) != ENUM_FIELDS: fail('enum field vocabulary drift')
    defaults = d['site_defaults']
    if set(defaults) != ENUM_FIELDS | BOOL_FIELDS: fail('default field set drift')
    for k in ENUM_FIELDS:
        if defaults[k] not in voc[k]: fail(f'default enum invalid: {k}')
    for k in BOOL_FIELDS:
        if type(defaults[k]) is not bool: fail(f'default boolean invalid: {k}')
    cases=d['cases']
    if len(cases)!=40: fail('case census drift')
    if any(c.get('kind')!='SITE' for c in cases): fail('case kind drift')
    if len({c['id'] for c in cases})!=40: fail('duplicate case id')
    for i,c in enumerate(cases,1):
        if not c['id'].startswith(f'S{i:02d}_'): fail(f'case order/id drift at {i}')
        if c['expected'] not in OUTCOMES: fail(f'unknown expected outcome {c["id"]}')
        ov=c['overrides']
        if not isinstance(ov,dict): fail(f'bad overrides {c["id"]}')
        unknown=set(ov)-set(defaults)
        if unknown: fail(f'unknown override key {c["id"]}: {sorted(unknown)}')
        for k,v in ov.items():
            if k in ENUM_FIELDS and v not in voc[k]: fail(f'unknown enum {c["id"]}:{k}={v}')
            if k in BOOL_FIELDS and type(v) is not bool: fail(f'bad bool {c["id"]}:{k}')


def materialize(defaults, c):
    s=copy.deepcopy(defaults); s.update(c['overrides']); return s


def hostile_controls(d):
    b=copy.deepcopy(d['site_defaults'])
    checks=[
      ({'horizontal_coverage':'OutsideDeclaredExtent'},'SpatialCoverageBlocked'),
      ({'groundwater_state':'HistoricalObserved'},'GroundwaterCurrentnessBlocked'),
      ({'interpolation_state':'Overextended'},'InterpolationBoundaryBlocked'),
      ({'uncertainty_state':'CoercedFavorable'},'UncertaintyBoundaryBlocked'),
      ({'post_work_change':'Fill','post_work_observation_state':'Missing'},'PostWorkObservationBlocked'),
      ({'foundation_consumer_changed':True,'consumer_applicability_reviewed':False},'ApplicabilityReviewRequired'),
      ({'work_or_provenance_event_only':True},'SourceAuthorityBlocked'),
      ({'favorable_default_requested':True},'FavorableDefaultBlocked'),
      ({'design_claim_requested':True},'DesignAuthorityBoundaryBlocked'),
      ({'permit_or_execution_claim_requested':True},'PhysicalAuthorityBoundaryBlocked'),
      ({'external_report_required':True,'external_report_identity':'Exact','external_report_currentness':'Current','external_report_spatial_match':'NearbyOnly','external_report_configuration_match':'Exact'},'ExternalReportBoundaryBlocked'),
    ]
    for ov,want in checks:
        s=copy.deepcopy(b); s.update(ov)
        got=derive(s)
        if got!=want: fail(f'hostile control {ov}: {got} != {want}')


def main():
    gz=CORPUS.read_bytes()
    if hashlib.sha256(gz).hexdigest()!=GZIP_SHA: fail('gzip digest drift')
    raw=gzip.decompress(gz)
    if hashlib.sha256(raw).hexdigest()!=JSON_SHA: fail('canonical JSON digest drift')
    if not raw.endswith(b'\n'): fail('canonical JSON missing final newline')
    d=json.loads(raw)
    canonical=(json.dumps(d,sort_keys=True,separators=(',',':'))+'\n').encode()
    if canonical!=raw: fail('JSON is not canonical compact sorted-key encoding')
    validate_shape(d)
    counts=Counter()
    for c in d['cases']:
        got=derive(materialize(d['site_defaults'],c))
        if got!=c['expected']: fail(f'{c["id"]}: derived {got} != expected {c["expected"]}')
        counts[got]+=1
    if set(counts)!=set(OUTCOMES): fail(f'not all dispositions exercised: {set(OUTCOMES)-set(counts)}')
    hostile_controls(d)
    print('PASS_BUILT_SITE_001A_REFERENCE digest='+JSON_SHA+' cases=40 outcomes='+json.dumps(dict(sorted(counts.items())),sort_keys=True))

if __name__=='__main__':
    try: main()
    except Exception as e:
        print('FAIL_BUILT_SITE_001A_REFERENCE:',e,file=sys.stderr)
        raise
