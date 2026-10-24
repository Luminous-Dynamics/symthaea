#!/usr/bin/env python3
"""RH-006 origin-local conditioning adversary; diagnostic only."""
import argparse,hashlib,json,numpy as np
from pathlib import Path
N=48+4+16+(24-1)*16; TRAIN,ORIGINS,STEP=48,24,16
class SM:
 def __init__(s,x):s.x=np.uint64(x&0xffffffffffffffff)
 def r(s,n):
  i=s.x+np.arange(n,dtype=np.uint64);z=i+np.uint64(0x9E3779B97F4A7C15)
  z=(z^(z>>np.uint64(30)))*np.uint64(0xBF58476D1CE4E5B9)
  z=(z^(z>>np.uint64(27)))*np.uint64(0x94D049BB133111EB);z^=z>>np.uint64(31)
  s.x=np.uint64((int(s.x)+n)&0xffffffffffffffff);return np.where((z&np.uint64(1))==0,-1.,1.)
def ar(g,n,r):
 e=g.r(n);x=np.empty(n);x[0]=e[0]
 for i in range(1,n):x[i]=r*x[i-1]+e[i]
 return x
def feat(g):
 r=np.column_stack([ar(g,N,.35) for _ in range(7)]);f=lambda x:np.clip(.5+.18*x,0,1)
 a,b,c=f(r[:,0]),f(r[:,1]),f(r[:,2]);al=f(.65*r[:,3]+.35*r[:,4]);x,y=f(r[:,5]),f(r[:,6]);t=f(.45*r[:,4]+.55*r[:,5])
 return np.column_stack([a,b,al,x,y,t,c])
def ratios(F):
 local=[];blocks=[]
 for o in range(ORIGINS):
  s=o*STEP;X=F[s:s+TRAIN,3:6];m=X.mean(0);sc=np.where(X.std(0)<=1e-12,1,X.std(0));X=(X-m)/sc;sv=np.linalg.svd(X,compute_uv=False);local.append(sv[-1]/sv[0]);blocks.append(X)
 sv=np.linalg.svd(np.vstack(blocks),compute_uv=False);return float(sv[-1]/sv[0]),local
def adversary(F):
 z=F.copy();s=(ORIGINS-1)*STEP;z[s:s+TRAIN,4]=z[s:s+TRAIN,3];z[s:s+TRAIN,5]=z[s:s+TRAIN,3];return z
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--paths",type=int,default=8);ap.add_argument("--seed",type=int,default=20261017);a=ap.parse_args();g=SM(a.seed);rows=[]
 for case in ("baseline","adversarial_last_origin"):
  for p in range(a.paths):
   F=feat(g);F=adversary(F) if case.startswith("adversarial") else F;pooled,local=ratios(F);rows.append({"case":case,"path":p,"pooled_min_singular_ratio":pooled,"local_min_singular_ratio":min(local),"last_origin_ratio":local[-1],"origins_below_1e-3":int(sum(x<1e-3 for x in local))})
 out={"schema":"rh006-origin-local-conditioning-adversary-executed/v1","status":"research-diagnostic-only","faithfulness_scope":"independent estimator-mechanics reimplementation; diagnostic only","execution":{"command":"python scripts/research/rh006_origin_local_conditioning_adversary_splitmix64.py --paths 8 --seed 20261017","script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},"geometry":{"train":48,"test":16,"gap":4,"origins":24,"step":16,"paths":a.paths},"results":[],"interpretation":{"formal_inference":False,"applicability":"not-approved-for-execution","selection":"stop-assumption-failure","finding":"A pooled conditioning summary can conceal a singular rolling origin. Conditioning policy must be evaluated at the actual estimator unit, i.e. each forecast origin, with a fail-closed aggregate policy."}}
 for case in ("baseline","adversarial_last_origin"):
  z=[x for x in rows if x["case"]==case];out["results"].append({"case":case,"pooled_min_singular_ratio_median":float(np.median([x["pooled_min_singular_ratio"] for x in z])),"local_min_singular_ratio_median":float(np.median([x["local_min_singular_ratio"] for x in z])),"last_origin_ratio_median":float(np.median([x["last_origin_ratio"] for x in z])),"paths_with_any_origin_below_1e-3":int(sum(x["origins_below_1e-3"]>0 for x in z))})
 raw=json.dumps(out,sort_keys=True,separators=(",",":"));out["payload_sha256"]=hashlib.sha256(raw.encode()).hexdigest();print(json.dumps(out,indent=2))
if __name__=="__main__":main()
