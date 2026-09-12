#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json,pathlib,tempfile
from typing import Any
P='ll009z.hybrid-visibility-policy.v1'; O='ll009z.hybrid-visibility-reconciliation-receipt.v1'; S='ll009s.semantic-visibility-receipt.v1'; K='ll009k.horizon-pack.v1'; Y='ll009y.memberwise-hybrid-horizon-receipt.v1'; Q='ll009q.clone-horizon-ensemble-receipt.v1'; V='ll009v.product90-rms-semantics-receipt.v1'; W='ll009w.distribution-free-familywise-horizon-receipt.v1'; X='ll009x.geometry-aware-familywise-rms-horizon-receipt.v1'; R='ll009r.spatial-support-classification-receipt.v1'; L='ll009l.cog-materialization-config.v1'; MODE='empirical_q_of_scenario_parameterized_distribution_free_far_envelopes'; EC='hybrid_scenario_sampled_visibility'
class ZE(RuntimeError):pass
def cb(v:Any)->bytes:return (json.dumps(v,sort_keys=True,indent=2,separators=(',',': '))+'\n').encode()
def hb(b:bytes)->str:return hashlib.sha256(b).hexdigest()
def hf(p:pathlib.Path)->str:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for c in iter(lambda:f.read(1<<20),b''):h.update(c)
 return h.hexdigest()
def rd(p,n):
 try:v=json.loads(p.read_text())
 except (OSError,json.JSONDecodeError) as e:raise ZE(f'cannot read {n}: {e}') from e
 if not isinstance(v,dict):raise ZE(f'{n} must be object')
 return v
def sh(v,n):
 x=v.get('receipt_sha256');
 if not isinstance(x,str) or len(x)!=64:raise ZE(f'{n} self-hash missing')
 b=json.loads(json.dumps(v));b.pop('receipt_sha256',None)
 if x!=hb(cb(b)):raise ZE(f'{n} self-hash mismatch')
def pol(p):
 if p.get('schema_version')!=P or p.get('target_evidence_class')!=EC:raise ZE('invalid Z policy')
 if p.get('required_y_binding_mode')!=MODE:raise ZE('policy Y mode mismatch')
 if p.get('numeric_mutation_policy')!='forbidden_copy_exact_s_metrics_and_targets':raise ZE('numeric mutation must be forbidden')
 if p.get('joint_probability_policy')!='prohibited_without_joint_coupling_theorem':raise ZE('joint probability must be prohibited')
 need={'risk_qualified_visibility','deterministic_visibility_bound','joint_probability_visibility'}
 if not isinstance(p.get('blocked_stronger_classes'),list) or not need<=set(p['blocked_stronger_classes']):raise ZE('stronger classes not blocked')
 return p
def check_s(s,yp):
 if s.get('schema_version')!=S or s.get('status') not in {'pass','claim_blocked_geometry_available'}:raise ZE('invalid S receipt')
 sh(s,'S');
 if s.get('k_horizon_pack_sha256')!=hf(yp) or s.get('horizon_numeric_authority')!=K:raise ZE('S/Y numeric binding mismatch')
 if not isinstance(s.get('metrics'),dict) or not s['metrics'] or not isinstance(s.get('targets'),list) or not s['targets']:raise ZE('S metrics/targets missing')
 if s.get('metric_evidence_class') not in {'descriptive_geometry_only','derived_empirical_sampled'}:raise ZE('S already stronger than Z class')
def check_y(y,p):
 if y.get('schema_version')!=K or y.get('producer_schema_version')!=Y or y.get('status')!='pass':raise ZE('invalid Y K-pack')
 sh(y,'Y');b=y.get('statistical_horizon_binding')
 if not isinstance(b,dict) or b.get('status')!='bound' or b.get('mode')!=p['required_y_binding_mode']:raise ZE('Y binding mode/status mismatch')
 if b.get('memberwise_composition')!='memberwise_max_then_empirical_quantile':raise ZE('Y composition theorem drift')
 q,a=b.get('quantile'),b.get('whole_sky_far_exceedance_budget_alpha')
 if not isinstance(q,(int,float)) or isinstance(q,bool) or not 0<float(q)<=1:raise ZE('Y quantile invalid')
 if not isinstance(a,(int,float)) or isinstance(a,bool) or not 0<float(a)<1:raise ZE('Y alpha invalid')
 h=y.get('memberwise_hybrid_evidence',{})
 if h.get('semantics_class')!='finite_empirical_q_of_scenario_parameterized_distribution_free_far_envelopes' or h.get('far_semantics')!='marginal_rms_model_re_evaluated_at_fixed_q_observer_scenarios_no_joint_probability_claim':raise ZE('Y hybrid semantics drift')
 return b
def lineage(y,b,paths):
 keys=('q_receipt_sha256','v_receipt_sha256','w_receipt_sha256','x_receipt_sha256','r_receipt_sha256','l_config_sha256')
 for k,p in zip(keys,paths):
  if b.get(k)!=hf(p):raise ZE(f'Y binding mismatch {k}')
 q,v,w,x,r,l=[rd(p,n) for p,n in zip(paths,('Q','V','W','X','R','L'))]
 for z,sc,n in ((q,Q,'Q'),(v,V,'V'),(w,W,'W'),(x,X,'X'),(r,R,'R')):
  if z.get('schema_version')!=sc:raise ZE(f'{n} schema mismatch')
  sh(z,n)
 if l.get('schema_version')!=L:raise ZE('L schema mismatch')
 for z,n in ((q,'Q'),(v,'V'),(w,'W'),(x,'X'),(r,'R'),(l,'L')):
  if z.get('study_id')!=y.get('study_id'):raise ZE(f'{n}/Y study mismatch')
 lh=hf(paths[-1])
 if q.get('l_config_sha256')!=lh or v.get('l_config_sha256')!=lh:raise ZE('Q/V/L mismatch')
 if w.get('l_config_sha256')!=lh or w.get('v_receipt_sha256')!=hf(paths[1]):raise ZE('W/V/L mismatch')
 if x.get('l_config_sha256')!=lh or x.get('w_receipt_sha256')!=hf(paths[2]):raise ZE('X/W/L mismatch')
 if q.get('semantics_class')!='empirical_ensemble' or v.get('semantics_class')!='rms_error':raise ZE('Q/V semantics drift')
 return q,v,w,x,r
def spatial(y,r):
 allowed={'continuous_hard_bound','empirical_multiscale_bound','resolution_qualified','sample_points_only','unknown'};common=r.get('strongest_common_spatial_support')
 if common not in allowed:raise ZE('R support invalid')
 ri={z.get('layer_id'):z for z in r.get('layers',[]) if isinstance(z,dict)};out=[]
 for z in y.get('layers',[]):
  if not isinstance(z,dict) or z.get('layer_id') not in ri:raise ZE('Y/R layer mismatch')
  rr=ri[z['layer_id']]
  if z.get('spatial_support_class')!=rr.get('support_class'):raise ZE('Y/R support class mismatch')
  out.append({'layer_id':z['layer_id'],'support_class':rr.get('support_class'),'applied_spatial_margin_deg':z.get('applied_spatial_margin_deg',0.0)})
 if not out:raise ZE('Y layers missing')
 return {'strongest_common_spatial_support':common,'layers':out}
def reconcile(pp,sp,yp,qp,vp,wp,xp,rp,lp):
 p=pol(rd(pp,'policy'));s=rd(sp,'S');y=rd(yp,'Y');check_s(s,yp);b=check_y(y,p);q,v,w,x,r=lineage(y,b,(qp,vp,wp,xp,rp,lp))
 for k in ('study_id','frame_contract_id','epoch_contract_id','site_ref','frame'):
  if s.get(k)!=y.get(k):raise ZE(f'S/Y {k} mismatch')
 if s.get('azimuth_bin_width_deg')!=y.get('azimuth_bin_width_deg'):raise ZE('S/Y bins mismatch')
 vis={'metrics':s['metrics'],'targets':s['targets']};ss=spatial(y,r);t=w.get('theorem',{})
 out={'schema_version':O,'status':'pass','study_id':y['study_id'],'frame_contract_id':y['frame_contract_id'],'epoch_contract_id':y['epoch_contract_id'],'site_ref':y['site_ref'],'frame':y['frame'],'evidence_class':EC,'numeric_authority':'exact_ll009s_visibility_receipt','semantic_authority':'ll009z_additive_reconciliation_only','source_s_metric_evidence_class':s.get('metric_evidence_class'),'visibility_numbers_preserved_without_recomputation':True,'preserved_visibility_payload_sha256':hb(cb(vis)),'metrics':s['metrics'],'targets':s['targets'],'lineage':{'policy_sha256':hf(pp),'s_receipt_sha256':hf(sp),'y_k_pack_sha256':hf(yp),'q_receipt_sha256':hf(qp),'v_receipt_sha256':hf(vp),'w_receipt_sha256':hf(wp),'x_receipt_sha256':hf(xp),'r_receipt_sha256':hf(rp),'l_config_sha256':hf(lp)},'hybrid_statistics':{'q_semantics':q.get('semantics_class'),'q_member_count':b.get('q_member_count'),'q_quantile_estimator':b.get('quantile_estimator'),'q_selected_quantile':b.get('quantile'),'far_semantics':v.get('semantics_class'),'far_whole_sky_exceedance_budget_alpha':b.get('whole_sky_far_exceedance_budget_alpha'),'far_distribution_assumption':t.get('distribution_assumption'),'far_independence_assumption':t.get('independence_assumption'),'far_geometry_mode':x.get('semantics_class'),'memberwise_composition':b.get('memberwise_composition'),'joint_probability_interpretation':'prohibited_without_joint_coupling_theorem'},'spatial_support':ss,'capability':{'hybrid_scenario_sampled_visibility_eligible':True,'risk_qualified_visibility_eligible':False,'deterministic_visibility_bound_eligible':False,'joint_probability_visibility_eligible':False,'continuous_physical_terrain_completeness_established':ss['strongest_common_spatial_support']=='continuous_hard_bound'},'blocked_stronger_classes':p['blocked_stronger_classes'],'claim_rule':'Exact S visibility numbers are promoted only to hybrid scenario-sampled evidence because they are computed from exact Y memberwise Q-scenario / distribution-free far-RMS terrain evidence; no joint Q×far probability is inferred.','non_claims':['No joint probability distribution across Q clone scenarios and Product 90 far errors is established.','The finite Q empirical quantile is not a population confidence guarantee.','The far alpha remains a separate distribution-free RMS familywise theorem and is not multiplied by the Q quantile.','Spatial-support limitations remain exactly those of LL-009R and Y.','Visibility geometry is not an RF link budget, delivered solar power model, site qualification, mission-safety determination, or operations authority.']};out['receipt_sha256']=hb(cb(out));return out
def wr(p,v):
 b=cb(v)
 if p.exists() and p.read_bytes()!=b:raise ZE(f'refusing to overwrite {p}')
 if not p.exists():p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
def ah(v):v=json.loads(json.dumps(v));v['receipt_sha256']=hb(cb(v));return v
def selftest():
 with tempfile.TemporaryDirectory() as td:
  r=pathlib.Path(td)
  def put(n,v):p=r/n;p.write_bytes(cb(v));return p
  p={'schema_version':P,'target_evidence_class':EC,'required_y_binding_mode':MODE,'numeric_mutation_policy':'forbidden_copy_exact_s_metrics_and_targets','joint_probability_policy':'prohibited_without_joint_coupling_theorem','blocked_stronger_classes':['risk_qualified_visibility','deterministic_visibility_bound','joint_probability_visibility']};pp=put('p',p)
  l={'schema_version':L,'study_id':'z'};lp=put('l',l);q=ah({'schema_version':Q,'status':'pass','study_id':'z','semantics_class':'empirical_ensemble','l_config_sha256':hf(lp)});qp=put('q',q);v=ah({'schema_version':V,'status':'pass','study_id':'z','semantics_class':'rms_error','l_config_sha256':hf(lp)});vp=put('v',v);w=ah({'schema_version':W,'status':'pass','study_id':'z','l_config_sha256':hf(lp),'v_receipt_sha256':hf(vp),'theorem':{'distribution_assumption':'none_beyond_rms_second_moment_model','independence_assumption':'none'}});wp=put('w',w);x=ah({'schema_version':X,'status':'pass','study_id':'z','semantics_class':'distribution_free_geometry_aware_familywise_rms_upper_envelope','l_config_sha256':hf(lp),'w_receipt_sha256':hf(wp)});xp=put('x',x);rr=ah({'schema_version':R,'status':'blocked','study_id':'z','strongest_common_spatial_support':'sample_points_only','layers':[{'layer_id':'n','support_class':'sample_points_only'},{'layer_id':'f','support_class':'resolution_qualified'}]});rp=put('r',rr)
  b={'status':'bound','mode':MODE,'quantile_estimator':'empirical_cdf_nearest_rank','quantile':.99,'q_receipt_sha256':hf(qp),'v_receipt_sha256':hf(vp),'w_receipt_sha256':hf(wp),'x_receipt_sha256':hf(xp),'r_receipt_sha256':hf(rp),'l_config_sha256':hf(lp),'whole_sky_far_exceedance_budget_alpha':.01,'q_member_count':100,'memberwise_composition':'memberwise_max_then_empirical_quantile'};y=ah({'schema_version':K,'producer_schema_version':Y,'status':'pass','study_id':'z','frame_contract_id':'F','epoch_contract_id':'E','site_ref':'s','frame':'M','azimuth_bin_width_deg':1.,'layers':[{'layer_id':'n','spatial_support_class':'sample_points_only'},{'layer_id':'f','spatial_support_class':'resolution_qualified'}],'statistical_horizon_binding':b,'memberwise_hybrid_evidence':{'semantics_class':'finite_empirical_q_of_scenario_parameterized_distribution_free_far_envelopes','far_semantics':'marginal_rms_model_re_evaluated_at_fixed_q_observer_scenarios_no_joint_probability_claim'}});yp=put('y',y);s=ah({'schema_version':S,'status':'claim_blocked_geometry_available','study_id':'z','frame_contract_id':'F','epoch_contract_id':'E','site_ref':'s','frame':'M','azimuth_bin_width_deg':1.,'k_horizon_pack_sha256':hf(yp),'horizon_numeric_authority':K,'metric_evidence_class':'descriptive_geometry_only','metrics':{'m':{'central':.7}},'targets':[{'target_id':'sun'}]});sp=put('s',s);a=reconcile(pp,sp,yp,qp,vp,wp,xp,rp,lp);assert cb(a)==cb(reconcile(pp,sp,yp,qp,vp,wp,xp,rp,lp)) and a['metrics']==s['metrics'] and not a['capability']['risk_qualified_visibility_eligible']
  bad=json.loads(json.dumps(s));bad['metrics']['m']['central']=.8;bp=put('bad',bad)
  try:reconcile(pp,bp,yp,qp,vp,wp,xp,rp,lp);raise AssertionError('tamper accepted')
  except ZE:pass
  print('LL-009Z hybrid visibility reconciliation self-test: PASS')
def main():
 a=argparse.ArgumentParser();a.add_argument('--policy');a.add_argument('--s-receipt');a.add_argument('--y-k-pack');a.add_argument('--q-receipt');a.add_argument('--v-receipt');a.add_argument('--w-receipt');a.add_argument('--x-receipt');a.add_argument('--r-receipt');a.add_argument('--l-config');a.add_argument('--output');a.add_argument('--self-test',action='store_true');z=a.parse_args()
 try:
  if z.self_test:selftest();return 0
  vals=(z.policy,z.s_receipt,z.y_k_pack,z.q_receipt,z.v_receipt,z.w_receipt,z.x_receipt,z.r_receipt,z.l_config,z.output)
  if not all(vals):raise ZE('all inputs and --output required')
  o=reconcile(*[pathlib.Path(x) for x in vals[:-1]]);wr(pathlib.Path(z.output),o);print(json.dumps(o,sort_keys=True,indent=2));return 0
 except (OSError,json.JSONDecodeError,ZE) as e:raise SystemExit(f'LL-009Z failure: {e}') from e
if __name__=='__main__':raise SystemExit(main())
