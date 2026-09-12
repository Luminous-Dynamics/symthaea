#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, math, pathlib
from typing import Any

P='ll009ac.scale-matched-rms-audit-policy.v1'; O='ll009ac.scale-matched-rms-model-check-receipt.v1'
AB='ll009ab.sdem-role-binding-receipt.v1'; V='ll009v.product90-rms-semantics-receipt.v1'; L='ll009l.cog-materialization-config.v1'
class E(RuntimeError): pass

def cb(v:Any)->bytes:return (json.dumps(v,sort_keys=True,indent=2,separators=(',',': '))+'\n').encode()
def hb(b:bytes)->str:return hashlib.sha256(b).hexdigest()
def hf(p:pathlib.Path)->str:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for c in iter(lambda:f.read(4<<20),b''):h.update(c)
 return h.hexdigest()
def rd(p,n):
 try:v=json.loads(p.read_text())
 except (OSError,json.JSONDecodeError) as x:raise E(f'cannot read {n}: {x}') from x
 if not isinstance(v,dict):raise E(f'{n} must be object')
 return v
def sh(v,n):
 x=v.get('receipt_sha256');b=json.loads(json.dumps(v));b.pop('receipt_sha256',None)
 if not isinstance(x,str) or len(x)!=64 or x!=hb(cb(b)):raise E(f'{n} self-hash mismatch')
def fin(x):return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x)
def rel(x):
 if not isinstance(x,str) or not x or '\\' in x:raise E(f'unsafe path {x!r}')
 p=pathlib.PurePosixPath(x)
 if p.is_absolute() or '..' in p.parts or any(q in ('','.','..') for q in p.parts):raise E(f'unsafe path {x!r}')
 return p
def resolve(root,s,n):
 if not isinstance(s,dict) or not isinstance(s.get('path'),str) or not isinstance(s.get('sha256'),str) or len(s['sha256'])!=64:raise E(f'{n}: invalid source')
 p=root/pathlib.Path(*rel(s['path']).parts)
 if not p.is_file() or hf(p)!=s['sha256']:raise E(f'{n}: missing/hash mismatch')
 return p
def req():
 try:
  import numpy as np, rasterio
  return np,rasterio
 except Exception as x:raise E('NumPy/Rasterio required for LL-009AC evidence') from x

class M:
 def __init__(s,ts=()):s.n=0;s.a=0.;s.q=0.;s.lo=math.inf;s.hi=-math.inf;s.ts=ts;s.ex=[0]*len(ts)
 def add(s,x):
  np,_=req();x=np.asarray(x,dtype='float64').ravel();x=x[np.isfinite(x)]
  if not len(x):return
  s.n+=len(x);s.a+=float(np.sum(x));s.q+=float(np.sum(x*x));s.lo=min(s.lo,float(np.min(x)));s.hi=max(s.hi,float(np.max(x)));u=np.abs(x)
  for i,t in enumerate(s.ts):s.ex[i]+=int(np.count_nonzero(u>t))
 def out(s):
  if not s.n:return {'count':0}
  m=s.a/s.n;z={'count':int(s.n),'mean':m,'stddev_population':math.sqrt(max(0.,s.q/s.n-m*m)),'rms':math.sqrt(s.q/s.n),'min':s.lo,'max':s.hi}
  for t,n in zip(s.ts,s.ex):z[f'fraction_abs_gt_{t:g}']=n/s.n
  return z

def policy(v):
 if v.get('schema_version')!=P:raise E('bad AC policy schema')
 for k in ('study_id','far_layer_id'):
  if not isinstance(v.get(k),str) or not v[k]:raise E(f'policy missing {k}')
 if v.get('incomplete_support_policy')!='exclude_with_count':raise E('V1 requires exclude_with_count')
 r=v.get('expected_integer_scale_ratio');b=v.get('diagnostic_block_width_coarse_cells');t=v.get('tile_width_coarse_cells')
 if not isinstance(r,int) or isinstance(r,bool) or r<2:raise E('integer scale ratio >=2 required')
 if not isinstance(b,int) or b<1 or not isinstance(t,int) or not 1<=t<=256:raise E('block/tile size invalid')
 q=v.get('surface_area_quadrature_orders')
 if not isinstance(q,list) or len(q)!=2 or not all(isinstance(x,int) and 2<=x<=10 for x in q) or q[0]>=q[1]:raise E('quadrature orders invalid')
 for k in ('max_relative_area_quadrature_difference','max_abs_weighted_mean_quadrature_difference_m','affine_tolerance'):
  if not fin(v.get(k)) or v[k]<=0:raise E(f'{k} invalid')
 p=v.get('projection_contract')
 if not isinstance(p,dict) or p.get('projection')!='south_polar_stereographic' or not fin(p.get('reference_radius_m')) or p['reference_radius_m']<=0 or abs(float(p.get('stereographic_scale_at_pole',math.nan))-1)>1e-15:raise E('canonical unit-scale spherical stereographic contract required')
 return v

def lineage(p,ab,v,l,lp):
 if ab.get('schema_version')!=AB or ab.get('status')!='pass':raise E('AB role receipt must pass')
 sh(ab,'AB role receipt')
 if ab.get('study_id')!=p['study_id'] or not isinstance(ab.get('roles'),dict) or 'sdem_elevation' not in ab['roles']:raise E('AB lineage mismatch')
 sd=ab['roles']['sdem_elevation']
 if not isinstance(sd,dict) or not isinstance(sd.get('raster_metadata'),dict):raise E('SDEM role metadata missing')
 if v.get('schema_version')!=V or v.get('status')!='pass':raise E('V receipt must pass')
 sh(v,'V receipt')
 if v.get('study_id')!=p['study_id'] or v.get('layer_id')!=p['far_layer_id'] or v.get('semantics_class')!='rms_error' or v.get('l_config_sha256')!=hf(lp):raise E('V lineage/semantics mismatch')
 if l.get('schema_version')!=L or l.get('study_id')!=p['study_id']:raise E('L config mismatch')
 pr=l.get('projection_contract')
 if not isinstance(pr,dict) or pr.get('projection')!='south_polar_stereographic' or abs(float(pr.get('reference_radius_m',math.nan))-float(p['projection_contract']['reference_radius_m']))>1e-9:raise E('L projection mismatch')
 ls=[x for x in l.get('layers',[]) if isinstance(x,dict) and x.get('layer_id')==p['far_layer_id']]
 if len(ls)!=1 or not isinstance(ls[0].get('elevation_source'),dict) or not isinstance(ls[0].get('uncertainty_source'),dict):raise E('exact Product90 elevation/RMS layer required')
 return sd,ls[0]

def nest(f,c,n,tol):
 if f.crs!=c.crs:raise E('fine/coarse CRS mismatch')
 a=list(f.transform)[:6];b=list(c.transform)[:6]
 for i in (0,1,3,4):
  z=n*a[i]
  if abs(b[i]-z)>tol*max(1.,abs(z)):raise E('coarse affine is not exact integer multiple of fine lattice')
 x,y=(~f.transform)*(c.transform.c,c.transform.f);xi,yi=round(x),round(y)
 if abs(x-xi)>tol or abs(y-yi)>tol:raise E('coarse origin not on fine-pixel boundary')
 return int(yi),int(xi)
def ranges(off,nf,nc,n):
 a0=max(0,math.floor((-off-n)/n)+1);a1=min(nc,math.ceil((nf-off)/n));f0=max(0,math.ceil(-off/n));f1=min(nc,math.floor((nf-n-off)/n)+1)
 return (a0,max(a0,a1)),(f0,max(f0,f1))
def areas(np,tr,r0,c0,h,w,order,R):
 nd,wt=np.polynomial.legendre.leggauss(order);u=(nd+1)/2;wt=wt/2;rr,cc=np.indices((h,w),dtype='float64');rr+=r0;cc+=c0;a,b,c,d,e,f=list(tr)[:6];out=np.zeros((h,w));D=4*R*R
 for i,x in enumerate(u):
  for j,y in enumerate(u):
   X=a*(cc+x)+b*(rr+y)+c;Y=d*(cc+x)+e*(rr+y)+f;out+=wt[i]*wt[j]/(1+(X*X+Y*Y)/D)**2
 return abs(a*e-b*d)*out
def blocks_out(bs):
 z=[]
 for (r,c) in sorted(bs):
  q=bs[(r,c)];z.append({'block_row':r,'block_col':c,'projected_normalized_residual':q[0].out(),'area_weighted_normalized_residual':q[1].out(),'aggregation_definition_sensitivity_m':q[2].out()})
 return z

def materialize(pp,abp,vp,lp,sroot,froot):
 np,rio=req();p=policy(rd(pp,'AC policy'));ab=rd(abp,'AB role receipt');v=rd(vp,'V receipt');l=rd(lp,'L config');sd,lay=lineage(p,ab,v,l,lp)
 sp=resolve(sroot,sd,'SDEM');ep=resolve(froot,lay['elevation_source'],'Product90 elevation');rp=resolve(froot,lay['uncertainty_source'],'Product90 RMS')
 n=p['expected_integer_scale_ratio'];tol=float(p['affine_tolerance']);R=float(p['projection_contract']['reference_radius_m']);lo,hi=p['surface_area_quadrature_orders'];tile=p['tile_width_coarse_cells'];bw=p['diagnostic_block_width_coarse_cells']
 pres=M();ares=M();pn=M((1.,2.,3.));an=M((1.,2.,3.));sen=M();rel=M();qa=M();qm=M();bs={};dig=hashlib.sha256();adm=fn=cn=zr=0;worst=[]
 with rio.open(sp) as f,rio.open(ep) as c,rio.open(rp) as r:
  if c.crs!=r.crs or c.width!=r.width or c.height!=r.height or any(abs(x-y)>tol for x,y in zip(list(c.transform)[:6],list(r.transform)[:6])):raise E('Product90 elevation/RMS grid mismatch')
  fs=math.hypot(f.transform.a,f.transform.d);cs=math.hypot(c.transform.a,c.transform.d)
  if abs(cs/fs-n)>1e-10:raise E('coarse/fine pixel-scale ratio mismatch')
  if fin(lay.get('expected_pixel_size_m')) and abs(cs-float(lay['expected_pixel_size_m']))>1e-8:raise E('Product90 pixel scale drift')
  ro,co=nest(f,c,n,tol);ar,fr=ranges(ro,f.height,c.height,n);ac,fc=ranges(co,f.width,c.width,n);anyn=max(0,ar[1]-ar[0])*max(0,ac[1]-ac[0]);full=max(0,fr[1]-fr[0])*max(0,fc[1]-fc[0]);boundary=max(0,anyn-full)
  if full<=0:raise E('no complete scale-matched cells')
  for cr0 in range(fr[0],fr[1],tile):
   cr1=min(fr[1],cr0+tile);fh=(cr1-cr0)*n;rr0=ro+n*cr0
   for cc0 in range(fc[0],fc[1],tile):
    cc1=min(fc[1],cc0+tile);fw=(cc1-cc0)*n;cl0=co+n*cc0;ch,cw=cr1-cr0,cc1-cc0
    F=f.read(1,window=rio.windows.Window(cl0,rr0,fw,fh),masked=True);C=c.read(1,window=rio.windows.Window(cc0,cr0,cw,ch),masked=True);U=r.read(1,window=rio.windows.Window(cc0,cr0,cw,ch),masked=True)
    fd=np.asarray(F.data,dtype='float64').reshape(ch,n,cw,n);fm=np.ma.getmaskarray(F).reshape(ch,n,cw,n);okf=~np.any(fm|~np.isfinite(fd),axis=(1,3));fn+=int(np.count_nonzero(~okf));cd=np.asarray(C.data,dtype='float64');ud=np.asarray(U.data,dtype='float64');okc=~(np.ma.getmaskarray(C)|~np.isfinite(cd)|np.ma.getmaskarray(U)|~np.isfinite(ud));cn+=int(np.count_nonzero(okf&~okc));pos=ud>0;zr+=int(np.count_nonzero(okf&okc&~pos));ok=okf&okc&pos
    if not np.any(ok):continue
    em=np.mean(fd,axis=(1,3));mn=np.min(fd,axis=(1,3));mx=np.max(fd,axis=(1,3));rv=np.sqrt(np.mean((fd-em[:,None,:,None])**2,axis=(1,3)))
    A0=areas(np,f.transform,rr0,cl0,fh,fw,lo,R).reshape(ch,n,cw,n);A1=areas(np,f.transform,rr0,cl0,fh,fw,hi,R).reshape(ch,n,cw,n);s0=np.sum(A0,axis=(1,3));s1=np.sum(A1,axis=(1,3));w0=np.sum(fd*A0,axis=(1,3))/s0;w1=np.sum(fd*A1,axis=(1,3))/s1;ad=np.abs(s1-s0)/s1;md=np.abs(w1-w0)
    if np.any(ok&(ad>p['max_relative_area_quadrature_difference']+1e-18)) or np.any(ok&(md>p['max_abs_weighted_mean_quadrature_difference_m']+1e-18)):raise E('surface-area quadrature convergence failure')
    pr=em-cd;aa=w1-cd;ppn=pr/ud;aan=aa/ud;ss=w1-em;pres.add(pr[ok]);ares.add(aa[ok]);pn.add(ppn[ok]);an.add(aan[ok]);sen.add(ss[ok]);rel.add(rv[ok]);qa.add(ad[ok]);qm.add(md[ok]);rows,cols=np.nonzero(ok)
    for x,y in zip(rows.tolist(),cols.tolist()):
     gr,gc=cr0+x,cc0+y;rec={'coarse_row':gr,'coarse_col':gc,'product90_elevation_m':float(cd[x,y]),'product90_rms_m':float(ud[x,y]),'fine_cell_count':n*n,'projected_equal_weight_sdem_mean_m':float(em[x,y]),'surface_area_weighted_sdem_mean_m':float(w1[x,y]),'projected_residual_m':float(pr[x,y]),'area_weighted_residual_m':float(aa[x,y]),'projected_normalized_residual':float(ppn[x,y]),'area_weighted_normalized_residual':float(aan[x,y]),'aggregation_definition_sensitivity_m':float(ss[x,y]),'sdem_min_m':float(mn[x,y]),'sdem_max_m':float(mx[x,y]),'sdem_range_m':float(mx[x,y]-mn[x,y]),'sdem_within_cell_rms_relief_m':float(rv[x,y]),'area_quadrature_relative_difference':float(ad[x,y]),'weighted_mean_quadrature_abs_difference_m':float(md[x,y])};dig.update(cb(rec));adm+=1;k=(gr//bw,gc//bw)
     if k not in bs:bs[k]=(M((1.,2.,3.)),M((1.,2.,3.)),M())
     bs[k][0].add([rec['projected_normalized_residual']]);bs[k][1].add([rec['area_weighted_normalized_residual']]);bs[k][2].add([rec['aggregation_definition_sensitivity_m']]);worst.append((abs(rec['area_weighted_normalized_residual']),rec))
  if not adm:raise E('no scale-matched cells admitted')
  worst=sorted(worst,key=lambda q:(-q[0],q[1]['coarse_row'],q[1]['coarse_col']))[:20]
  z={'schema_version':O,'status':'pass','study_id':p['study_id'],'semantics_class':'scale_matched_cross_method_rms_model_check','policy_sha256':hf(pp),'ab_role_receipt_sha256':hf(abp),'v_receipt_sha256':hf(vp),'l_config_sha256':hf(lp),'far_layer_id':p['far_layer_id'],'source_hashes':{'sdem_elevation':sd['sha256'],'product90_elevation':lay['elevation_source']['sha256'],'product90_rms':lay['uncertainty_source']['sha256']},'scale_matching':{'fine_pixel_scale_m':fs,'coarse_pixel_scale_m':cs,'integer_scale_ratio':n,'fine_row_offset_of_coarse_origin':ro,'fine_col_offset_of_coarse_origin':co,'grid_alignment_policy':'exact_integer_lattice_no_resampling','coarse_cells_with_any_sdem_extent_overlap':anyn,'coarse_cells_with_complete_geometric_fine_support':full,'excluded_boundary_incomplete_support':boundary,'excluded_fine_nodata_or_nonfinite':fn,'excluded_coarse_nodata_or_nonfinite':cn,'excluded_nonpositive_rms':zr,'admitted_cell_count':adm},'aggregation':{'projected_equal_weight_definition':'arithmetic_mean_of_complete_exact_fine_cell_tiling','surface_area_definition':'integral of spherical stereographic area Jacobian over every fine projected cell','surface_area_jacobian':'1/(1+(x^2+y^2)/(4R^2))^2','reference_radius_m':R,'stereographic_scale_at_pole':1.0,'quadrature_orders':[lo,hi],'observed_area_quadrature_relative_difference':qa.out(),'observed_weighted_mean_quadrature_abs_difference_m':qm.out()},'projected_residual_m':pres.out(),'surface_area_weighted_residual_m':ares.out(),'projected_normalized_residual':pn.out(),'surface_area_weighted_normalized_residual':an.out(),'aggregation_definition_sensitivity_m':sen.out(),'within_cell_sdem_rms_relief_m':rel.out(),'diagnostic_block_width_coarse_cells':bw,'spatial_block_summaries':blocks_out(bs),'worst_abs_area_weighted_normalized_residual_cells':[r for _,r in worst],'cell_record_digest_sha256':dig.hexdigest(),'calibration_interpretation':'scale-matched descriptive model check of Product90 RMS; not a certified calibration multiplier or probability guarantee','spatial_dependence_interpretation':'adjacent coarse cells are not treated as IID; deterministic blocks are descriptive only','non_claims':['The SDEM is not independent ground truth.','Observed normalized-residual dispersion is not automatically an uncertainty multiplier.','Product90 ADJ_ERR remains rms_error rather than hard_upper_bound.','LL-009W/X remain model-conditional unless a later calibration-adequacy theorem justifies stronger probability authority.','No Q×Product90 joint probability or multiplied confidence is inferred.','No hidden raster reprojection or interpolation is performed.']};z['receipt_sha256']=hb(cb(z));return z

def wr(p,v):
 b=cb(v)
 if p.exists() and p.read_bytes()!=b:raise E(f'refusing differing output {p}')
 if not p.exists():p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
def main():
 a=argparse.ArgumentParser();a.add_argument('--policy',required=True);a.add_argument('--ab-role-receipt',required=True);a.add_argument('--v-receipt',required=True);a.add_argument('--l-config',required=True);a.add_argument('--sdem-root',default='.');a.add_argument('--far-root',default='.');a.add_argument('--output',required=True);x=a.parse_args()
 try:z=materialize(pathlib.Path(x.policy),pathlib.Path(x.ab_role_receipt),pathlib.Path(x.v_receipt),pathlib.Path(x.l_config),pathlib.Path(x.sdem_root),pathlib.Path(x.far_root));wr(pathlib.Path(x.output),z);print(json.dumps(z,sort_keys=True,indent=2));return 0
 except (E,OSError,ValueError) as q:raise SystemExit(f'LL-009AC failure: {q}') from q
if __name__=='__main__':raise SystemExit(main())
