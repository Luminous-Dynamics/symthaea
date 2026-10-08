#!/usr/bin/env python3
import argparse,hashlib,json
from pathlib import Path
import numpy as np
class SM:
 def __init__(self,s): self.s=np.uint64(s&0xffffffffffffffff)
 def r(self,n):
  n=int(n); idx=self.s+np.arange(n,dtype=np.uint64); z=idx+np.uint64(0x9E3779B97F4A7C15); z=(z^(z>>np.uint64(30)))*np.uint64(0xBF58476D1CE4E5B9); z=(z^(z>>np.uint64(27)))*np.uint64(0x94D049BB133111EB); z^=z>>np.uint64(31); self.s=np.uint64((int(self.s)+n)&0xffffffffffffffff); return np.where((z&np.uint64(1))==0,-1.,1.)
def sym(A): return (A+A.T)/2
def root(A,inv=False):
 e,U=np.linalg.eigh(sym(A)); e=np.maximum(e,1e-10); d=1/np.sqrt(e) if inv else np.sqrt(e); return (U*d)@U.T
def spd(g,d):
 X=g.r(d*d).reshape(d,d); return sym(X@X.T+0.75*np.eye(d))
def canon(Se,X,Sq): return np.linalg.svd(root(Se,1)@X@root(Sq,1),compute_uv=False)
def run(a):
 g=SM(a.seed); rows=[]; errs={k:0. for k in ['V','mean','spec','obs']}; raw=[]; can=[]
 for i in range(a.trials):
  d,m,p=a.latent_dim,a.outcome_dim,a.feature_dim; Sq=spd(g,d); Se=spd(g,m); C=g.r(m*d).reshape(m,d); U,s,Vt=np.linalg.svd(C,False); s=a.strength*s/max(float(s[0]),1e-30); C=(U*s)@Vt; K=root(Se)@C@root(Sq,1); Xi=K@Sq; V=sym(Se-K@Sq@K.T)
  A=g.r(d*d).reshape(d,d)+(1.3+0.05*(i%4))*np.eye(d)
  while np.linalg.cond(A)>25: A=.5*A+np.eye(d)
  Sqs=sym(A@Sq@A.T); Ks=K@np.linalg.inv(A); Xis=Ks@Sqs; Vs=sym(Se-Ks@Sqs@Ks.T)
  eV=float(np.max(np.abs(V-Vs))); eM=float(np.max(np.abs(Ks@A-K))); s0=canon(Se,Xi,Sq); s1=canon(Se,Xis,Sqs); eS=float(np.max(np.abs(s0-s1))); errs['V']=max(errs['V'],eV); errs['mean']=max(errs['mean'],eM); errs['spec']=max(errs['spec'],eS)
  raw.append(abs(np.linalg.norm(K,2)-np.linalg.norm(Ks,2))); can.append(float(s0[0]))
  M=g.r(p*d).reshape(p,d); Sn=spd(g,p); Seps=V; CF=M@Sq@K.T; SF=M@Sq@M.T+Sn; EF=K@Sq@K.T+Seps; Obs=np.block([[SF,CF],[CF.T,EF]])
  B=g.r(d*d).reshape(d,d)+(0.9+0.03*(i%3))*np.eye(d)
  while np.linalg.cond(B)>25: B=.5*B+np.eye(d)
  M2=M@np.linalg.inv(B); K2=K@np.linalg.inv(B); Sq2=sym(B@Sq@B.T); CF2=M2@Sq2@K2.T; SF2=M2@Sq2@M2.T+Sn; EF2=K2@Sq2@K2.T+Seps; Obs2=np.block([[SF2,CF2],[CF2.T,EF2]]); eO=float(np.max(np.abs(Obs-Obs2))); errs['obs']=max(errs['obs'],eO)
  rows.append({'trial':i,'latent_cond':float(np.linalg.cond(A)),'raw_K_norm_change':raw[-1],'canonical_max':can[-1],'canonical_spectrum_error':eS,'observed_cov_error':eO})
 return {'schema':'rh006-k-identifiability-observable-crosscov/v1','status':'research-diagnostic-only','applicability':'not-approved-for-execution','selection':'stop-assumption-failure','execution':vars(a),'identities':{'cross_covariance':'Xi=Cov(e,q)=K Sigma_q','conditional_covariance':"Var(e|q)=Sigma_e-K Sigma_q K'",'latent_reparameterization':"q*=Aq, Sigma_q*=A Sigma_q A', K*=K A^{-1}",'canonical_operator':'Sigma_e^{-1/2} Xi Sigma_q^{-1/2}','feasibility':"Sigma_e-K Sigma_q K' >= 0"},'results':{'max_conditional_covariance_error':errs['V'],'max_conditional_mean_error':errs['mean'],'max_canonical_spectrum_error':errs['spec'],'max_latent_factor_observed_covariance_error':errs['obs'],'median_raw_K_norm_change':float(np.median(raw)),'max_raw_K_norm_change':float(max(raw)),'median_canonical_strength':float(np.median(can))},'rows':rows,'scientific_conclusion':'K is not an invariant observable nuisance coordinate under latent reparameterization. When q is observed, Xi=K Sigma_q and the canonical cross-covariance spectrum are invariant bindings; when q is latent, a measurement/factor normalization is required before K itself is identified.','nonclaims':['No formal inference.','No confidence interval.','No empirical validation.','No arbitrary-DGP identification claim.','No Rust implementation qualification.']}
if __name__=='__main__':
 ap=argparse.ArgumentParser(); ap.add_argument('--trials',type=int,default=40); ap.add_argument('--seed',type=int,default=20261011); ap.add_argument('--latent-dim',type=int,default=6); ap.add_argument('--outcome-dim',type=int,default=5); ap.add_argument('--feature-dim',type=int,default=4); ap.add_argument('--strength',type=float,default=.7); a=ap.parse_args(); o=run(a); o['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(); raw=json.dumps(o,sort_keys=True,separators=(',',':')); o['payload_sha256']=hashlib.sha256(raw.encode()).hexdigest(); print(json.dumps(o,indent=2,sort_keys=True))
