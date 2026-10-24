#!/usr/bin/env python3
"""RH-006 null-surface conditioning/ridge stress; diagnostic only."""
import argparse,hashlib,json,math,numpy as np
from pathlib import Path
TRAIN,TEST,GAP,ORIGINS,STEP=48,16,4,24,16
N=TRAIN+GAP+TEST+(ORIGINS-1)*STEP
RIDGES=(0.,1e-12,1e-10,1e-8,1e-6,1e-4,1e-2); EPS=(1e-2,1e-3,1e-4,0.)
RHO=.8; SIG=.08; H=.1; M=256
class SM:
 def __init__(s,x):s.x=np.uint64(x&0xffffffffffffffff)
 def r(s,n):
  i=s.x+np.arange(n,dtype=np.uint64);z=i+np.uint64(0x9E3779B97F4A7C15)
  z=(z^(z>>np.uint64(30)))*np.uint64(0xBF58476D1CE4E5B9)
  z=(z^(z>>np.uint64(27)))*np.uint64(0x94D049BB133111EB);z^=z>>np.uint64(31)
  s.x=np.uint64((int(s.x)+n)&0xffffffffffffffff);return np.where(z&np.uint64(1),1.,-1.)
def ar(n,g,r):
 e=g.r(n);x=np.empty(n);x[0]=e[0]
 for i in range(1,n):x[i]=r*x[i-1]+e[i]
 return x
def feat(g):
 r=np.column_stack([ar(N,g,.35) for _ in range(7)])
 f=lambda x:np.clip(.5+.18*x,0,1)
 a,b,c=f(r[:,0]),f(r[:,1]),f(r[:,2]); al=f(.65*r[:,3]+.35*r[:,4])
 x,y=f(r[:,5]),f(r[:,6]);t=f(.45*r[:,4]+.55*r[:,5])
 return np.column_stack((a,b,al,x,y,t,c))
def stress(F,g,e):
 z=F.copy();x=F[:,3];z[:,4]=np.clip(x+e*.18*ar(N,g,.25),0,1);z[:,5]=np.clip(.25+.5*x+e*.18*ar(N,g,.25),0,1);return z
def op(X,Y,r):
 m=X.mean(0);q=np.where(X.std(0)<=1e-12,1,X.std(0));A=np.c_[np.ones(len(X)),(X-m)/q];B=np.c_[np.ones(len(Y)),(Y-m)/q]
 G=A.T@A;G[1:,1:]+=r*np.eye(X.shape[1]);return B@np.linalg.solve(G,A.T)
def cov(F):
 n=len(F);sig=SIG*(1+H*(F[:, -1]-.5));L=np.zeros((n,n))
 for t in range(n):
  L[t,t]=sig[t]
  if t:L[t,:t]=(RHO**np.arange(t,0,-1))*sig[:t]
 return L@L.T
def ops(F,r):
 out=[];nr=[0,1,2,6]
 for o in range(ORIGINS):
  s=o*STEP;te=s+TRAIN+GAP;out.append((op(F[s:s+TRAIN,nr],F[te:te+TEST,nr],r),op(F[s:s+TRAIN],F[te:te+TEST],r)))
 return out
def quad(F,ws):
 S=cov(F);a,b,al,*_=F.T;mu=.15+.45*a+.25*b+.2*F[:,6]+.3*al;Z=F[:,3:6];Q=np.zeros((3,3));B=np.zeros(3);c=0.
 for o,(W,V) in enumerate(ws):
  s=o*STEP;te=s+TRAIN+GAP;mt,me=mu[s:s+TRAIN],mu[te:te+TEST];zt,ze=Z[s:s+TRAIN],Z[te:te+TEST]
  qc=W@mt-me;qr=V@mt-me;hc=W@zt-ze;hr=V@zt-ze;St=S[s:s+TRAIN,s:s+TRAIN];Sty=S[s:s+TRAIN,te:te+TEST]
  st=lambda W:np.trace(W@St@W.T)-2*np.trace(W@Sty)
  Q+=(hc.T@hc-hr.T@hr)/TEST;B+=2*(hc.T@qc-hr.T@qr)/TEST;c+=(qc@qc-qr@qr+st(W)-st(V))/TEST
 return Q/ORIGINS,B/ORIGINS,c/ORIGINS
def root(Q,b,c,v):
 A=float(v@Q@v);B=float(b@v);C=float(c);z=np.roots([A,B,C] if abs(A)>1e-14 else [B,C])
 p=[x.real for x in z if abs(x.imag)<1e-8 and x.real>0 and np.isfinite(x.real)]
 return min(p) if p else None
def dirs(n):
 ph=(1+math.sqrt(5))/2
 for i in range(n):
  z=1-2*(i+.5)/n;r=math.sqrt(max(0,1-z*z));th=2*math.pi*i/ph;yield np.array([r*math.cos(th),r*math.sin(th),z])
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--paths",type=int,default=8);ap.add_argument("--directions",type=int,default=256);ap.add_argument("--seed",type=int,default=20261013);ap.add_argument("--ridges",type=float,nargs="+",default=list(RIDGES));a=ap.parse_args()
 g=SM(a.seed);raw=[]
 for p in range(a.paths):
  base=feat(g)
  for e in EPS:
   F=stress(base,g,e)
   for r in a.ridges:
    try:
     ws=ops(F,r);Q,b,c=quad(F,ws);ev=np.linalg.eigvalsh(Q);dc=[];rr=[]
     for o in range(ORIGINS):
      s=o*STEP;X=F[s:s+TRAIN];X=(X-X.mean(0))/np.where(X.std(0)<=1e-12,1,X.std(0));sv=np.linalg.svd(X,compute_uv=False);dc.append(sv[0]/max(sv[-1],1e-18))
     for v in dirs(a.directions):
      z=root(Q,b,c,v)
      if z is not None:rr.append(z)
     raw.append({"epsilon":e,"ridge":r,"path":p,"status":"computed","dc_max":float(max(dc)),"qmin":float(ev[0]),"qmax":float(ev[-1]),"qcond":float(abs(ev[-1])/max(abs(ev[0]),1e-18)),"rank10":int(np.sum(abs(ev)>1e-10)),"root_frac":len(rr)/a.directions,"root_med":float(np.median(rr)),"root_max":float(max(rr))})
    except np.linalg.LinAlgError: raw.append({"epsilon":e,"ridge":r,"path":p,"status":"singular","dc_max":float(max(dc)) if dc else None})
 groups=[]
 for e in EPS:
  for r in a.ridges:
   z=[x for x in raw if x["epsilon"]==e and x["ridge"]==r];c=[x for x in z if x["status"]=="computed"]
   groups.append({"epsilon":e,"ridge":r,"paths":len(z),"singular_paths":len(z)-len(c),
     "design_condition_max_median":float(np.median([x["dc_max"] for x in z])),
     "Q_condition_abs_median":float(np.median([x["qcond"] for x in c])) if c else None,
     "Q_eigen_min_median":float(np.median([x["qmin"] for x in c])) if c else None,
     "Q_rank_tol_1e-10_min":min(x["rank10"] for x in c) if c else None,
     "positive_root_median_mean":float(np.mean([x["root_med"] for x in c])) if c else None,
     "positive_root_max_global":float(max(x["root_max"] for x in c)) if c else None})
 out={"schema":"rh006-null-surface-conditioning-ridge-stress-executed/v1","status":"research-diagnostic-only",
  "execution":{"command":"python scripts/research/rh006_null_surface_conditioning_ridge_stress_splitmix64.py --paths 8 --directions 256 --seed 20261013 --ridges 0 1e-12 1e-10 1e-8 1e-6 1e-4 1e-2"},
  "geometry":{"train":TRAIN,"test":TEST,"gap":GAP,"origins":ORIGINS,"step":STEP,"outcome_rho":RHO,"heteroskedastic_strength":H,"epsilons":list(EPS),"ridges":a.ridges,"surface_directions":a.directions},
  "scenarios":groups,"interpretation":{"formal_inference":False,"applicability":"not-approved-for-execution","selection":"stop-assumption-failure","finding":"Positive ridge restores numerical solvability under exact collinearity but not identification of the nearly flat equal-risk surface."}}
 out["execution"]["script_sha256"]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
 raw=json.dumps(out,sort_keys=True,separators=(",",":"));out["payload_sha256"]=hashlib.sha256(raw.encode()).hexdigest();print(json.dumps(out,indent=2,sort_keys=True))
if __name__=="__main__":main()
