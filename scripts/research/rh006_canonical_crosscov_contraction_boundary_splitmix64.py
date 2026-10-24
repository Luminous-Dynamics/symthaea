#!/usr/bin/env python3
import argparse,hashlib,json
from pathlib import Path
import numpy as np
class SM:
 def __init__(self,s): self.s=np.uint64(s&0xffffffffffffffff)
 def r(self,n):
  n=int(n); idx=self.s+np.arange(n,dtype=np.uint64); z=idx+np.uint64(0x9E3779B97F4A7C15); z=(z^(z>>np.uint64(30)))*np.uint64(0xBF58476D1CE4E5B9); z=(z^(z>>np.uint64(27)))*np.uint64(0x94D049BB133111EB); z^=(z>>np.uint64(31)); self.s=np.uint64((int(self.s)+n)&0xffffffffffffffff); return np.where((z&np.uint64(1))==0,-1.,1.)
def sym(A): return (A+A.T)/2
def root(A,inv=False):
 e,U=np.linalg.eigh(sym(A)); e=np.maximum(e,1e-10); d=(1/np.sqrt(e)) if inv else np.sqrt(e); return (U*d)@U.T
def spd(g,d):
 X=g.r(d*d).reshape(d,d); return sym(X@X.T+0.5*np.eye(d))
def run(a):
 g=SM(a.seed); rows=[]
 for i in range(a.trials):
  m,d=a.outcome_dim,a.latent_dim; Se=spd(g,m); Sq=spd(g,d); C0=g.r(m*d).reshape(m,d); U,s,Vt=np.linalg.svd(C0,False)
  for s_target in a.strengths:
   C=(U*(s_target*s/max(float(s[0]),1e-30)))@Vt
   K=root(Se)@C@root(Sq,True); V=sym(Se-K@Sq@K.T); cov_min=float(np.min(np.linalg.eigvalsh(V))); canonical=float(np.linalg.svd(root(Se,True)@K@root(Sq),compute_uv=False)[0]); residual=1-canonical**2
   rows.append({'trial':i,'target_strength':s_target,'canonical_strength':canonical,'min_conditional_cov_eigen':cov_min,'residual_strength_gap':residual,'admissible_at_1e-10':cov_min>=-1e-10})
 summary={}
 for st in a.strengths:
  r=[x for x in rows if x['target_strength']==st]
  summary[str(st)]={'canonical_strength_range':[min(x['canonical_strength'] for x in r),max(x['canonical_strength'] for x in r)],'min_cov_eigen_range':[min(x['min_conditional_cov_eigen'] for x in r),max(x['min_conditional_cov_eigen'] for x in r)],'admissible_fraction':sum(x['admissible_at_1e-10'] for x in r)/len(r)}
 return {'schema':'rh006-canonical-crosscov-contraction-boundary/v1','status':'research-diagnostic-only','applicability':'not-approved-for-execution','selection':'stop-assumption-failure','execution':vars(a),'identity':{'Xi':'K Sigma_q','canonical_operator':'Sigma_e^{-1/2} Xi Sigma_q^{-1/2}','feasibility':"Sigma_e-K Sigma_q K' >= 0 iff ||canonical_operator||_2 <= 1 for positive-definite Sigma_e,Sigma_q",'parameterization':'Xi=Sigma_e^{1/2} C Sigma_q^{1/2}, ||C||_2<=1'},'results':summary,'rows':rows,'scientific_conclusion':'The canonical cross-covariance class is a spectral-norm contraction. Strength 1 is the covariance-completion boundary; strength above 1 is infeasible. This provides a fail-closed admissibility test independent of the latent coordinate system.','nonclaims':['No formal inference.','No confidence interval.','No empirical validation.','No arbitrary-DGP identification claim.','No Rust implementation qualification.']}
if __name__=='__main__':
 ap=argparse.ArgumentParser(); ap.add_argument('--trials',type=int,default=60); ap.add_argument('--seed',type=int,default=20261012); ap.add_argument('--latent-dim',type=int,default=7); ap.add_argument('--outcome-dim',type=int,default=5); ap.add_argument('--strengths',type=float,nargs='+',default=[0.0,0.35,0.85,0.99,1.0,1.01]); a=ap.parse_args(); o=run(a); o['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(); raw=json.dumps(o,sort_keys=True,separators=(',',':')); o['payload_sha256']=hashlib.sha256(raw.encode()).hexdigest(); print(json.dumps(o,indent=2,sort_keys=True))
