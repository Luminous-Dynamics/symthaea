#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json,math,pathlib
from typing import Any
import materialize_ll009ac_scale_matched_rms_audit as AC

P='ll009ad.three-way-terrain-decomposition-policy.v1';O='ll009ad.three-way-terrain-decomposition-receipt.v1'
class E(RuntimeError):pass

def rd(p,n):return AC.rd(p,n)
def cb(v:Any)->bytes:return AC.cb(v)
def hf(p):return AC.hf(p)
def sh(v,n):return AC.sh(v,n)
def wr(p,v):return AC.wr(p,v)
def fin(x):return AC.fin(x)
class B:
 def __init__(s):s.n=0;s.x=s.y=s.xx=s.yy=s.xy=0.
 def add(s,x,y):
  s.n+=1;s.x+=x;s.y+=y;s.xx+=x*x;s.yy+=y*y;s.xy+=x*y
 def out(s):
  if not s.n:return {'count':0}
  mx,my=s.x/s.n,s.y/s.n;vx=max(0.,s.xx/s.n-mx*mx);vy=max(0.,s.yy/s.n-my*my);cv=s.xy/s.n-mx*my;den=math.sqrt(vx*vy)
  return {'count':s.n,'covariance_population':cv,'correlation_population':None if den<=0 else cv/den}
def policy(v):
 if v.get('schema_version')!=P:raise E('bad AD policy schema')
 for k in ('study_id','near_layer_id','far_layer_id'):
  if not isinstance(v.get(k),str) or not v[k]:raise E(f'policy missing {k}')
 if v['near_layer_id']==v['far_layer_id']:raise E('near/far layer IDs must differ')
 for k in ('closure_tolerance_m','dominance_tie_tolerance_m','affine_tolerance'):
  if not fin(v.get(k)) or v[k]<=0:raise E(f'{k} invalid')
 return v
def ac_receipt(v,acp,abp,vp,lp,p):
 if v.get('schema_version')!='ll009ac.scale-matched-rms-model-check-receipt.v1' or v.get('status')!='pass' or v.get('semantics_class')!='scale_matched_cross_method_rms_model_check':raise E('AC receipt must pass')
 sh(v,'AC receipt')
 if v.get('study_id')!=p['study_id'] or v.get('far_layer_id')!=p['far_layer_id'] or v.get('policy_sha256')!=hf(acp) or v.get('ab_role_receipt_sha256')!=hf(abp) or v.get('v_receipt_sha256')!=hf(vp) or v.get('l_config_sha256')!=hf(lp):raise E('AC exact lineage mismatch')
 return v
def layer(l,i):
 q=[x for x in l.get('layers',[]) if isinstance(x,dict) and x.get('layer_id')==i]
 if len(q)!=1 or not isinstance(q[0].get('elevation_source'),dict):raise E(f'exact layer {i} elevation required')
 return q[0]
def off(f,g,tol):
 if f.crs!=g.crs:raise E('fine SDEM/Site01 LDEM CRS mismatch')
 a=list(f.transform)[:6];b=list(g.transform)[:6]
 for i in (0,1,3,4):
  if abs(a[i]-b[i])>tol*max(1.,abs(a[i])):raise E('fine SDEM/Site01 LDEM lattice matrix mismatch')
 x,y=(~f.transform)*(g.transform.c,g.transform.f);xi,yi=round(x),round(y)
 if abs(x-xi)>tol or abs(y-yi)>tol:raise E('fine SDEM/Site01 LDEM subpixel shift')
 return int(yi),int(xi)
def moments():return {'total':AC.M(),'sfs':AC.M(),'baseline':AC.M(),'total_norm':AC.M((1.,2.,3.)),'sfs_norm':AC.M((1.,2.,3.)),'baseline_norm':AC.M((1.,2.,3.)),'closure':AC.M()}
def mout(q):return {k:v.out() for k,v in q.items()}

def materialize(pp,acpp,acrp,abp,vp,lp,sroot,nroot,froot):
 np,rio=AC.req();p=policy(rd(pp,'AD policy'));ap=AC.policy(rd(acpp,'AC policy'));ar=ac_receipt(rd(acrp,'AC receipt'),acpp,abp,vp,lp,p);ab=rd(abp,'AB role receipt');v=rd(vp,'V receipt');l=rd(lp,'L config');sd,far=AC.lineage(ap,ab,v,l,lp);near=layer(l,p['near_layer_id'])
 if ap['study_id']!=p['study_id'] or ap['far_layer_id']!=p['far_layer_id']:raise E('AD/AC policy mismatch')
 sp=AC.resolve(sroot,sd,'SDEM');npth=AC.resolve(nroot,near['elevation_source'],'Site01 LDEM');ep=AC.resolve(froot,far['elevation_source'],'Product90 elevation');rp=AC.resolve(froot,far['uncertainty_source'],'Product90 RMS')
 n=ap['expected_integer_scale_ratio'];tol=p['affine_tolerance'];R=ap['projection_contract']['reference_radius_m'];lo,hi=ap['surface_area_quadrature_orders'];tile=ap['tile_width_coarse_cells'];eq=moments();aw=moments();beq=B();baw=B();dig=hashlib.sha256();acdig=hashlib.sha256();adm=rein=cancel=tie=sdm=bdm=0;worst=[]
 with rio.open(sp) as s,rio.open(npth) as nl,rio.open(ep) as c,rio.open(rp) as rms:
  ro,co=AC.nest(s,c,n,tol);nr,nc=off(s,nl,tol);arows,frows=AC.ranges(ro,s.height,c.height,n);acols,fcols=AC.ranges(co,s.width,c.width,n);full=max(0,frows[1]-frows[0])*max(0,fcols[1]-fcols[0])
  if full!=ar['scale_matching']['coarse_cells_with_complete_geometric_fine_support']:raise E('AC complete-geometry population drift')
  for cr0 in range(frows[0],frows[1],tile):
   cr1=min(frows[1],cr0+tile);fh=(cr1-cr0)*n;sr0=ro+n*cr0
   for cc0 in range(fcols[0],fcols[1],tile):
    cc1=min(fcols[1],cc0+tile);fw=(cc1-cc0)*n;sc0=co+n*cc0;ch,cw=cr1-cr0,cc1-cc0;lr0=sr0-nr;lc0=sc0-nc
    if lr0<0 or lc0<0 or lr0+fh>nl.height or lc0+fw>nl.width:raise E('Site01 LDEM does not cover full AC support population')
    S=s.read(1,window=rio.windows.Window(sc0,sr0,fw,fh),masked=True);N=nl.read(1,window=rio.windows.Window(lc0,lr0,fw,fh),masked=True);C=c.read(1,window=rio.windows.Window(cc0,cr0,cw,ch),masked=True);U=rms.read(1,window=rio.windows.Window(cc0,cr0,cw,ch),masked=True)
    sv=np.asarray(S.data,dtype='float64').reshape(ch,n,cw,n);nv=np.asarray(N.data,dtype='float64').reshape(ch,n,cw,n);sm=np.ma.getmaskarray(S).reshape(ch,n,cw,n);nm=np.ma.getmaskarray(N).reshape(ch,n,cw,n);cd=np.asarray(C.data,dtype='float64');uv=np.asarray(U.data,dtype='float64')
    oks=~np.any(sm|~np.isfinite(sv),axis=(1,3));okc=~(np.ma.getmaskarray(C)|~np.isfinite(cd)|np.ma.getmaskarray(U)|~np.isfinite(uv))&(uv>0);valid=oks&okc
    if np.any(valid & np.any(nm|~np.isfinite(nv),axis=(1,3))):raise E('Site01 LDEM missing on an AC-admitted fine support tile')
    if not np.any(valid):continue
    sem=np.mean(sv,axis=(1,3));nem=np.mean(nv,axis=(1,3));mn=np.min(sv,axis=(1,3));mx=np.max(sv,axis=(1,3));rv=np.sqrt(np.mean((sv-sem[:,None,:,None])**2,axis=(1,3)))
    A0=AC.areas(np,s.transform,sr0,sc0,fh,fw,lo,R).reshape(ch,n,cw,n);A1=AC.areas(np,s.transform,sr0,sc0,fh,fw,hi,R).reshape(ch,n,cw,n);s0=np.sum(A0,axis=(1,3));s1=np.sum(A1,axis=(1,3));sw0=np.sum(sv*A0,axis=(1,3))/s0;sw1=np.sum(sv*A1,axis=(1,3))/s1;nw1=np.sum(nv*A1,axis=(1,3))/s1;ad=np.abs(s1-s0)/s1;md=np.abs(sw1-sw0)
    if np.any(valid&(ad>ap['max_relative_area_quadrature_difference']+1e-18)) or np.any(valid&(md>ap['max_abs_weighted_mean_quadrature_difference_m']+1e-18)):raise E('AC quadrature convergence drift')
    rows,cols=np.nonzero(valid)
    for x,y in zip(rows.tolist(),cols.tolist()):
     gr,gc=cr0+x,cc0+y
     acrec={'coarse_row':gr,'coarse_col':gc,'product90_elevation_m':float(cd[x,y]),'product90_rms_m':float(uv[x,y]),'fine_cell_count':n*n,'projected_equal_weight_sdem_mean_m':float(sem[x,y]),'surface_area_weighted_sdem_mean_m':float(sw1[x,y]),'projected_residual_m':float(sem[x,y]-cd[x,y]),'area_weighted_residual_m':float(sw1[x,y]-cd[x,y]),'projected_normalized_residual':float((sem[x,y]-cd[x,y])/uv[x,y]),'area_weighted_normalized_residual':float((sw1[x,y]-cd[x,y])/uv[x,y]),'aggregation_definition_sensitivity_m':float(sw1[x,y]-sem[x,y]),'sdem_min_m':float(mn[x,y]),'sdem_max_m':float(mx[x,y]),'sdem_range_m':float(mx[x,y]-mn[x,y]),'sdem_within_cell_rms_relief_m':float(rv[x,y]),'area_quadrature_relative_difference':float(ad[x,y]),'weighted_mean_quadrature_abs_difference_m':float(md[x,y])};acdig.update(cb(acrec));adm+=1
     rec={'coarse_row':gr,'coarse_col':gc,'product90_rms_m':float(uv[x,y])}
     for name,Sv,Nv in (('projected',sem[x,y],nem[x,y]),('area',sw1[x,y],nw1[x,y])):
      total=float(Sv-cd[x,y]);sfs=float(Sv-Nv);base=float(Nv-cd[x,y]);close=total-(sfs+base)
      if abs(close)>p['closure_tolerance_m']:raise E('three-way decomposition closure failure')
      rec[name]={'total_residual_m':total,'sfs_increment_m':sfs,'lola_baseline_difference_m':base,'closure_error_m':close,'total_normalized':total/uv[x,y],'sfs_normalized':sfs/uv[x,y],'lola_baseline_normalized':base/uv[x,y]};q=eq if name=='projected' else aw
      q['total'].add([total]);q['sfs'].add([sfs]);q['baseline'].add([base]);q['total_norm'].add([total/uv[x,y]]);q['sfs_norm'].add([sfs/uv[x,y]]);q['baseline_norm'].add([base/uv[x,y]]);q['closure'].add([close]);(beq if name=='projected' else baw).add(sfs,base)
     sfs=rec['area']['sfs_increment_m'];base=rec['area']['lola_baseline_difference_m'];d=abs(sfs)-abs(base)
     if abs(d)<=p['dominance_tie_tolerance_m']:kind='mixed_equal';tie+=1
     elif d>0:kind='sfs_dominant';sdm+=1
     else:kind='lola_baseline_dominant';bdm+=1
     prod=sfs*base
     if prod>0:rein+=1
     elif prod<0:cancel+=1
     rec['area_dominance']=kind;dig.update(cb(rec));worst.append((abs(rec['area']['total_normalized']),rec))
  if adm!=ar['scale_matching']['admitted_cell_count'] or acdig.hexdigest()!=ar['cell_record_digest_sha256']:raise E('reconstructed AC exact cell population/digest mismatch')
  worst=sorted(worst,key=lambda q:(-q[0],q[1]['coarse_row'],q[1]['coarse_col']))[:20]
  z={'schema_version':O,'status':'pass','study_id':p['study_id'],'semantics_class':'scale_matched_three_way_terrain_discrepancy_decomposition','policy_sha256':hf(pp),'ac_policy_sha256':hf(acpp),'ac_receipt_sha256':hf(acrp),'ab_role_receipt_sha256':hf(abp),'v_receipt_sha256':hf(vp),'l_config_sha256':hf(lp),'near_layer_id':p['near_layer_id'],'far_layer_id':p['far_layer_id'],'admitted_cell_count':adm,'reconstructed_ac_cell_record_digest_sha256':acdig.hexdigest(),'decomposition_record_digest_sha256':dig.hexdigest(),'projected_equal_weight':mout(eq),'surface_area_weighted':mout(aw),'projected_sfs_vs_lola_baseline':beq.out(),'surface_area_sfs_vs_lola_baseline':baw.out(),'area_dominance_fractions':{'sfs_dominant':sdm/adm,'lola_baseline_dominant':bdm/adm,'mixed_equal':tie/adm,'reinforcing':rein/adm,'cancelling':cancel/adm,'zero_product':(adm-rein-cancel)/adm},'worst_abs_total_normalized_residual_cells':[r for _,r in worst],'interpretation':'Exact algebraic decomposition of AC discrepancy into SfS-vs-Site01 and Site01-vs-Product90 terms; descriptive ancestry-aware diagnostic, not causal error attribution.','non_claims':['SfS increment is not independent truth.','LOLA baseline difference is not automatically error in either LOLA product.','Finite-population covariance/correlation does not establish independence or causality.','Component dispersion does not automatically define a Product90 RMS multiplier.','W/X probability authority and R/AB spatial-support semantics are unchanged.']};z['receipt_sha256']=AC.hb(cb(z));return z

def mout(q):return {k:v.out() for k,v in q.items()}
def main():
 a=argparse.ArgumentParser();a.add_argument('--policy',required=True);a.add_argument('--ac-policy',required=True);a.add_argument('--ac-receipt',required=True);a.add_argument('--ab-role-receipt',required=True);a.add_argument('--v-receipt',required=True);a.add_argument('--l-config',required=True);a.add_argument('--sdem-root',default='.');a.add_argument('--near-root',default='.');a.add_argument('--far-root',default='.');a.add_argument('--output',required=True);x=a.parse_args()
 try:z=materialize(pathlib.Path(x.policy),pathlib.Path(x.ac_policy),pathlib.Path(x.ac_receipt),pathlib.Path(x.ab_role_receipt),pathlib.Path(x.v_receipt),pathlib.Path(x.l_config),pathlib.Path(x.sdem_root),pathlib.Path(x.near_root),pathlib.Path(x.far_root));AC.wr(pathlib.Path(x.output),z);print(json.dumps(z,sort_keys=True,indent=2));return 0
 except (E,AC.E,OSError,ValueError) as q:raise SystemExit(f'LL-009AD failure: {q}') from q
if __name__=='__main__':raise SystemExit(main())
