#!/usr/bin/env python3
import argparse, hashlib, json, math
import numpy as np

TRAIN,TEST,GAP,ORIGINS,STEP=48,16,4,24,16
N=TRAIN+GAP+TEST+(ORIGINS-1)*STEP
RIDGE=1e-8; LAG=3; ALPHA=.05; MC=12; B=49
BASE_DIRS=[]
raw=[
[1,0,0],[0,1,0],[0,0,1],[1,1,1],[1,1,-1],[1,-1,1],[1,-1,-1],[2,1,0]
]
for x in raw:
 v=np.array(x,float); BASE_DIRS.append(v/np.linalg.norm(v))
DIRECTIONS=[]
for i,v in enumerate(BASE_DIRS):
 DIRECTIONS.append((f'd{i:02d}+',v)); DIRECTIONS.append((f'd{i:02d}-',-v))

class SM:
 def __init__(s,x): s.x=np.uint64(x&0xffffffffffffffff)
 def r(s,n):
  n=int(n); i=s.x+np.arange(n,dtype=np.uint64); z=i+np.uint64(0x9E3779B97F4A7C15)
  z=(z^(z>>np.uint64(30)))*np.uint64(0xBF58476D1CE4E5B9)
  z=(z^(z>>np.uint64(27)))*np.uint64(0x94D049BB133111EB); z^=z>>np.uint64(31)
  s.x=np.uint64((int(s.x)+n)&0xffffffffffffffff); return np.where((z&np.uint64(1))==0,-1.,1.)

def ar(g,n,r):
 e=g.r(n); x=np.empty(n); x[0]=e[0]
 for i in range(1,n): x[i]=r*x[i-1]+e[i]
 return x

def feat(g):
 r=np.column_stack([ar(g,N,.35) for _ in range(7)]); f=lambda x:np.clip(.5+.18*x,0,1)
 a,b,c=f(r[:,0]),f(r[:,1]),f(r[:,2]); al=f(.65*r[:,3]+.35*r[:,4]); a2b,b2a=f(r[:,5]),f(r[:,6]); turn=f(.45*r[:,4]+.55*r[:,5])
 return np.column_stack([a,b,al,a2b,b2a,turn,c])

def cov(n,rho,hetero,common):
 sig=.08*(1+.7*(common-.5)) if hetero else np.full(n,.08); L=np.zeros((n,n))
 for t in range(n):
  L[t,t]=sig[t]
  if t: L[t,:t]=(rho**np.arange(t,0,-1))*sig[:t]
 return L@L.T,sig

def op(X,Y):
 m=X.mean(0); sc=np.where(X.std(0)<=1e-12,1.,X.std(0)); A=np.c_[np.ones(len(X)),(X-m)/sc]; B=np.c_[np.ones(len(Y)),(Y-m)/sc]; G=A.T@A; G[1:,1:]+=RIDGE*np.eye(X.shape[1]); return B@np.linalg.solve(G,A.T)

def build_ops(F):
 nr=[0,1,2,6]; rel=[0,1,2,3,4,5,6]; out=[]
 for o in range(ORIGINS):
  s=o*STEP; te=s+TRAIN+GAP; out.append((op(F[s:s+TRAIN,nr],F[te:te+TEST,nr]),op(F[s:s+TRAIN,rel],F[te:te+TEST,rel])))
 return out

def mu0(F): return .15+.45*F[:,0]+.25*F[:,1]+.20*F[:,6]+.30*F[:,2]

def quad(F,ops,rho,hetero):
 S,_=cov(N,rho,hetero,F[:,-1]); m=mu0(F); Z=F[:,3:6]; Q=np.zeros((3,3)); b=np.zeros(3); c=0.
 for o,(Wc,Wr) in enumerate(ops):
  s=o*STEP; te=s+TRAIN+GAP; tr=slice(s,s+TRAIN); tt=slice(te,te+TEST)
  mt,me=m[tr],m[tt]; zt,ze=Z[tr],Z[tt]
  qc=Wc@mt-me; qr=Wr@mt-me; Hc=Wc@zt-ze; Hr=Wr@zt-ze
  stt=S[s:s+TRAIN,s:s+TRAIN]; sty=S[s:s+TRAIN,te:te+TEST]
  st=lambda W:np.trace(W@stt@W.T)-2*np.trace(W@sty)
  Q+=(Hc.T@Hc-Hr.T@Hr)/TEST; b+=2*(Hc.T@qc-Hr.T@qr)/TEST; c+=(qc@qc-qr@qr+st(Wc)-st(Wr))/TEST
 return Q/ORIGINS,b/ORIGINS,c/ORIGINS

def root(Q,b,c,v):
 A=float(v@Q@v); B=float(b@v); C=float(c); rr=np.roots([A,B,C] if abs(A)>1e-14 else [B,C]); p=[float(x.real) for x in rr if abs(x.imag)<1e-8 and x.real>0 and np.isfinite(x.real)]; return min(p) if p else None

def stat_batch(D):
 D=np.asarray(D,float); m=D.mean(axis=1); l=np.mean((D-m[:,None])**2,axis=1)
 for k in range(1,LAG+1): l+=2*(1-k/(LAG+1))*np.mean((D[:,k:]-m[:,None])*(D[:,:-k]-m[:,None]),axis=1)
 return np.where(l>1e-14,np.sqrt(ORIGINS)*m/np.sqrt(l),0.)

def pval(F,y,mu,ops,rng,rho,hetero):
 D=[]
 for o,(Wc,Wr) in enumerate(ops):
  s=o*STEP; te=s+TRAIN+GAP; D.append(np.mean((Wc@y[s:s+TRAIN]-y[te:te+TEST])**2-(Wr@y[s:s+TRAIN]-y[te:te+TEST])**2))
 Tobs=stat_batch(np.asarray(D,float)[None,:])[0]
 _,sig=cov(N,rho,hetero,F[:,-1]); signs=rng.r(B*N).reshape(B,N); E=np.empty_like(signs); E[:,0]=sig[0]*signs[:,0]
 for t in range(1,N): E[:,t]=rho*E[:,t-1]+sig[t]*signs[:,t]
 Y=mu[None,:]+E; DT=np.empty((B,ORIGINS))
 for o,(Wc,Wr) in enumerate(ops):
  s=o*STEP; te=s+TRAIN+GAP; pn=Y[:,s:s+TRAIN]@Wc.T; pr=Y[:,s:s+TRAIN]@Wr.T; yt=Y[:,te:te+TEST]; DT[:,o]=np.mean((pn-yt)**2-(pr-yt)**2,axis=1)
 Tb=stat_batch(DT); return (1+int(np.count_nonzero(Tb>=Tobs)))/(B+1)

def scenario(rho,hetero,seed,args):
 rng=SM(seed); rec=[]
 for path in range(args.paths):
  F=feat(rng); ops=build_ops(F); Q,b,c=quad(F,ops,rho,hetero); m=mu0(F); Z=F[:,3:6]
  for name,v in DIRECTIONS:
   k=root(Q,b,c,v)
   if k is None:
    rec.append((name,path,None,None,None)); continue
   mu=m+k*(Z@v); ps=[]
   for _ in range(args.outcomes):
    _,sig=cov(N,rho,hetero,F[:,-1]); e=np.empty(N); z=rng.r(N); e[0]=sig[0]*z[0]
    for t in range(1,N): e[t]=rho*e[t-1]+sig[t]*z[t]
    ps.append(pval(F,mu+e,mu,ops,rng,rho,hetero))
   rec.append((name,path,k,float(np.mean(ps)),int(np.mean(np.asarray(ps)<ALPHA))))
 rows=[]
 for name,_ in DIRECTIONS:
  x=[r for r in rec if r[0]==name and r[2] is not None]; ps=[]
  for r in x: ps += [r[3]]*args.outcomes
  rows.append({"direction":name,"root_points":len(x),"root_support_rate":len(x)/args.paths,"root_median":float(np.median([r[2] for r in x])) if x else None,"reject_rate":float(np.mean(np.asarray(ps)<ALPHA)) if ps else None,"mean_p":float(np.mean(ps)) if ps else None})
 valid=[r for r in rows if r["reject_rate"] is not None]
 return {"rho":rho,"heteroskedastic":hetero,"paths":args.paths,"directions":len(DIRECTIONS),"direction_results":rows,"envelope":{"min_reject_rate":min(r["reject_rate"] for r in valid),"max_reject_rate":max(r["reject_rate"] for r in valid),"median_direction_reject_rate":float(np.median([r["reject_rate"] for r in valid])),"min_root_support":min(r["root_support_rate"] for r in valid),"max_root_support":max(r["root_support_rate"] for r in valid),"total_root_points":sum(r["root_points"] for r in valid),"total_possible_points":args.paths*len(DIRECTIONS)}}

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--paths',type=int,default=12); ap.add_argument('--outcomes',type=int,default=1); ap.add_argument('--bootstrap',type=int,default=49); ap.add_argument('--seed',type=int,default=20261040); a=ap.parse_args(); globals()['B']=a.bootstrap
 results=[scenario(0.,False,a.seed,a),scenario(.5,False,a.seed+1000,a),scenario(.8,False,a.seed+2000,a),scenario(.5,True,a.seed+3000,a)]
 out={"schema":"rh006-least-favorable-null-surface-envelope-executed/v1","status":"research-diagnostic-only","faithfulness_scope":"independent estimator-mechanics reimplementation; not the Symthaea Rust implementation","geometry":{"train":TRAIN,"gap":GAP,"test":TEST,"origins":ORIGINS,"step":STEP,"ridge":RIDGE,"bartlett_lag":LAG,"alpha":ALPHA,"directions":len(DIRECTIONS),"paths":a.paths,"outcomes_per_point":a.outcomes,"bootstrap":a.bootstrap},"direction_grid":"8 predeclared directions plus antipodes (16 total), fixed before bootstrap outcomes are generated","scenarios":results,"interpretation":{"formal_inference":False,"applicability":"not-approved-for-execution","selection":"stop-assumption-failure","finding":"A least-favorable envelope is useful as a diagnostic only until a uniform composite-null procedure is derived. Root support itself is an admissibility dimension: maximum rejection over directions is not interpretable when some null directions are not representable under the finite-sample risk geometry.","policy":"The envelope must never be selected post hoc from confirmatory data; the direction grid, root-support rule, and nuisance policy must be frozen prospectively.","next_gate":"replace grid-maximum diagnostics with a mathematically defined uniform/null-surface procedure or narrow the scientific estimand to a predeclared direction."},"references":["Elliott, Müller & Watson (2015), Nearly Optimal Tests When a Nuisance Parameter is Present Under the Null","Hill, Weak Identification Robust Bootstrap diagnostic literature","Clark & McCracken (2015), DOI 10.1016/j.jeconom.2014.06.016"]}
 raw=json.dumps(out,sort_keys=True,separators=(',',':'));out['payload_sha256']=hashlib.sha256(raw.encode()).hexdigest();print(json.dumps(out,indent=2))
if __name__=='__main__': main()
