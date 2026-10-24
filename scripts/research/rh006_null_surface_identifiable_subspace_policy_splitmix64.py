#!/usr/bin/env python3
"""RH-006 identifiable-subspace policy simulation; diagnostic only."""
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np
N=48+4+16+(24-1)*16; TRAIN,ORIGINS,STEP=48,24,16
EPS=[1e-2,1e-3,1e-4,0.0]; TAUS=[1e-2,1e-3,1e-4,1e-5]
class SM:
 def __init__(s,seed):s.x=np.uint64(seed&0xffffffffffffffff)
 def r(s,n):
  i=s.x+np.arange(n,dtype=np.uint64);z=i+np.uint64(0x9E3779B97F4A7C15)
  z=(z^(z>>np.uint64(30)))*np.uint64(0xBF58476D1CE4E5B9)
  z=(z^(z>>np.uint64(27)))*np.uint64(0x94D049BB133111EB);z^=z>>np.uint64(31)
  s.x=np.uint64((int(s.x)+n)&0xffffffffffffffff)
  return np.where((z&np.uint64(1))==0,-1.0,1.0)
def ar(g,n,rho):
 e=g.r(n);x=np.empty(n);x[0]=e[0]
 for i in range(1,n):x[i]=rho*x[i-1]+e[i]
 return x
def feat(g):
 r=np.column_stack([ar(g,N,.35) for _ in range(7)]);f=lambda x:np.clip(.5+.18*x,0,1)
 a,b,c=f(r[:,0]),f(r[:,1]),f(r[:,2]);al=f(.65*r[:,3]+.35*r[:,4]);x,y=f(r[:,5]),f(r[:,6]);t=f(.45*r[:,4]+.55*r[:,5])
 return np.column_stack((a,b,al,x,y,t,c))
def stress(F,g,e):
 z=F.copy();x=F[:,3];z[:,4]=np.clip(x+e*.18*ar(g,N,.25),0,1);z[:,5]=np.clip(.25+.5*x+e*.18*ar(g,N,.25),0,1);return z
def svd_pooled(F):
 xs=[]
 for o in range(ORIGINS):
  s=o*STEP;X=F[s:s+TRAIN,3:6];m=X.mean(0);sc=np.where(X.std(0)<=1e-12,1,X.std(0));xs.append((X-m)/sc)
 return np.linalg.svd(np.vstack(xs),full_matrices=False)
def dirs(n):
 p=(1+math.sqrt(5))/2
 for i in range(n):
  z=1-2*(i+.5)/n;r=math.sqrt(max(0,1-z*z));th=2*math.pi*i/p;yield np.array((r*math.cos(th),r*math.sin(th),z))
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--paths",type=int,default=8);ap.add_argument("--directions",type=int,default=256);ap.add_argument("--seed",type=int,default=20261014);a=ap.parse_args()
 g=SM(a.seed);rows=[]
 for e in EPS:
  for path in range(a.paths):
   F=stress(feat(g),g,e);_,S,Vt=svd_pooled(F);ratio=S/S[0]
   for tau in TAUS:
    k=int(np.sum(ratio>=tau));V=Vt[:k].T
    R=np.asarray([np.linalg.norm(V@(V.T@v)) for v in dirs(a.directions)])
    rows.append((e,tau,k,float(np.median(R)),float(np.quantile(R,.1)),float(np.quantile(R,.9)),float(np.mean(R>=.9)),float(ratio[-1])))
 out={"schema":"rh006-null-surface-identifiable-subspace-policy-executed/v1","status":"research-diagnostic-only",
 "execution":{"command":"python scripts/research/rh006_null_surface_identifiable_subspace_policy_splitmix64.py --paths 8 --directions 256 --seed 20261014",
 "script_sha256":None},"geometry":{"train":48,"test":16,"gap":4,"origins":24,"step":16,"feature_paths":8,"surface_directions":256,"epsilons":EPS,"singular_ratio_thresholds":TAUS},
 "scenarios":[],"interpretation":{"formal_inference":False,"applicability":"not-approved-for-execution","selection":"stop-assumption-failure","policy_status":"no threshold approved","finding":"Projecting onto an SVD-defined identifiable subspace is not automatically equivalent to the original three-channel relational estimand; discarded directions define a different alternative."}}
 for e in EPS:
  for tau in TAUS:
   z=[x for x in rows if x[0]==e and x[1]==tau]
   out["scenarios"].append({"epsilon":e,"tau":tau,"rank_median":float(np.median([x[2] for x in z])),"min_singular_ratio_median":float(np.median([x[7] for x in z])),"retained_median":float(np.median([x[3] for x in z])),"retained_p10":float(np.median([x[4] for x in z])),"retained_p90":float(np.median([x[5] for x in z])),"directions_ge90_median":float(np.median([x[6] for x in z]))})
 out["execution"]["script_sha256"]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();raw=json.dumps(out,sort_keys=True,separators=(",",":"));out["payload_sha256"]=hashlib.sha256(raw.encode()).hexdigest();print(json.dumps(out,indent=2,sort_keys=True))
if __name__=="__main__":main()
