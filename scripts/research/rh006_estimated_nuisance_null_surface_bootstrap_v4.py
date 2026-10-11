#!/usr/bin/env python3
"""RH-006 plug-in estimated-nuisance restricted-null bootstrap diagnostic."""
import argparse, hashlib, json, math
from pathlib import Path
import numpy as np

TRAIN,GAP,TEST,ORIGINS,STEP=48,4,16,24,16
N=TRAIN+GAP+TEST+(ORIGINS-1)*STEP
RIDGES=(1e-8,1e-4); EPS=(1e-2,1e-3,1e-4)
ALPHA=.05; K=3
DIRS={"strong":np.array([1.,1.,.5]),"weak":np.array([0.,1.,-2.])}
DIRS={k:v/np.linalg.norm(v) for k,v in DIRS.items()}

class SM:
 def __init__(s,x):s.x=np.uint64(x&0xffffffffffffffff)
 def r(s,n):
  n=int(n);i=s.x+np.arange(n,dtype=np.uint64);z=i+np.uint64(0x9E3779B97F4A7C15)
  z=(z^(z>>np.uint64(30)))*np.uint64(0xBF58476D1CE4E5B9)
  z=(z^(z>>np.uint64(27)))*np.uint64(0x94D049BB133111EB);z^=z>>np.uint64(31)
  s.x=np.uint64((int(s.x)+n)&0xffffffffffffffff);return np.where((z&np.uint64(1))==0,-1.,1.)

def ar(g,n,rho):
 e=g.r(n);x=np.empty(n);x[0]=e[0]
 for i in range(1,n):x[i]=rho*x[i-1]+e[i]
 return x

def feat(g,e):
 raw=np.column_stack([ar(g,N,.35) for _ in range(7)]);f=lambda x:np.clip(.5+.18*x,0,1)
 a,b,c=f(raw[:,0]),f(raw[:,1]),f(raw[:,2]);al=f(.65*raw[:,3]+.35*raw[:,4]);a2b,b2a=f(raw[:,5]),f(raw[:,6]);turn=f(.45*raw[:,4]+.55*raw[:,5])
 F=np.column_stack([a,b,al,a2b,b2a,turn,c]);z=F.copy()
 z[:,4]=np.clip(F[:,3]+e*.18*ar(g,N,.25),0,1);z[:,5]=np.clip(.25+.5*F[:,3]+e*.18*ar(g,N,.25),0,1);return z

def mu0(F):return .15+.45*F[:,0]+.25*F[:,1]+.20*F[:,6]+.30*F[:,2]
def sig0(F):return .08*(1+.7*(F[:,-1]-.5))
def op(X,Y,r):
 m=X.mean(0);sc=np.where(X.std(0)<=1e-12,1.,X.std(0));A=np.c_[np.ones(len(X)),(X-m)/sc];B=np.c_[np.ones(len(Y)),(Y-m)/sc];G=A.T@A;G[1:,1:]+=r*np.eye(X.shape[1]);return B@np.linalg.solve(G,A.T)
def build_ops(F,r):
 nr=[0,1,2,6];rel=[0,1,2,3,4,5,6];out=[]
 for o in range(ORIGINS):
  s=o*STEP;te=s+TRAIN+GAP;out.append((op(F[s:s+TRAIN,nr],F[te:te+TEST,nr],r),op(F[s:s+TRAIN,rel],F[te:te+TEST,rel],r)))
 return out

def cov(rho,sig):
 L=np.zeros((N,N))
 for t in range(N):
  L[t,t]=sig[t]
  if t:L[t,:t]=(rho**np.arange(t,0,-1))*sig[:t]
 return L@L.T

def quad(F,ridge,m0,rho,sig,v):
 ops=build_ops(F,ridge);S=cov(rho,sig);Z=F[:,3:6];Q=np.zeros((3,3));b=np.zeros(3);c=0.
 for o,(Wc,Wr) in enumerate(ops):
  s=o*STEP;te=s+TRAIN+GAP;tr=np.arange(s,s+TRAIN);tt=np.arange(te,te+TEST)
  qc=Wc@m0[tr]-m0[tt];qr=Wr@m0[tr]-m0[tt];Hc=Wc@Z[tr]-Z[tt];Hr=Wr@Z[tr]-Z[tt]
  St=S[np.ix_(tr,tr)];Sty=S[np.ix_(tr,tt)]
  st=lambda W:np.trace(W@St@W.T)-2*np.trace(W@Sty)
  Q+=(Hc.T@Hc-Hr.T@Hr)/TEST;b+=2*(Hc.T@qc-Hr.T@qr)/TEST;c+=(qc@qc-qr@qr+st(Wc)-st(Wr))/TEST
 return Q/ORIGINS,b/ORIGINS,c/ORIGINS,ops

def root(Q,b,c,v):
 A=float(v@Q@v);B=float(b@v);C=float(c);z=np.roots([A,B,C] if abs(A)>1e-14 else [B,C]);p=[float(x.real) for x in z if abs(x.imag)<1e-8 and x.real>0 and np.isfinite(x.real)];return min(p) if p else None

def est_nuisance(F,y):
 nr=[0,1,2,6];rel=[0,1,2,3,4,5,6];X=F[:,rel];m=X.mean(0);sc=np.where(X.std(0)<=1e-12,1.,X.std(0));A=np.c_[np.ones(N),(X-m)/sc];G=A.T@A;G[1:,1:]+=1e-8*np.eye(len(rel));bf=np.linalg.solve(G,A.T@y);res=y-A@bf
 C=F[:,nr];cm=C.mean(0);cs=np.where(C.std(0)<=1e-12,1.,C.std(0));Ac=np.c_[np.ones(N),(C-cm)/cs];Gc=Ac.T@Ac;Gc[1:,1:]+=1e-8*np.eye(len(nr));bc=np.linalg.solve(Gc,Ac.T);m0=Ac@bc
 den=np.dot(res[:-1],res[:-1]);rho=float(np.clip(np.dot(res[1:],res[:-1])/den if den>1e-14 else 0.,-.98,.98));inn=res[1:]-rho*res[:-1];x=F[1:,-1]-.5;ab=np.linalg.lstsq(np.c_[np.ones(len(x)),x],np.abs(inn),rcond=None)[0];a=max(float(ab[0]),1e-5);h=float(np.clip(ab[1]/a,-.95,.95));sig=np.maximum(a*(1+h*(F[:,-1]-.5)),1e-5);return m0,rho,sig

def statistic_batch(Y,ops):
 B=Y.shape[0];D=np.empty((B,ORIGINS))
 for o,(Wc,Wr) in enumerate(ops):
  s=o*STEP;te=s+TRAIN+GAP;pn=Y[:,s:s+TRAIN]@Wc.T;pr=Y[:,s:s+TRAIN]@Wr.T;yte=Y[:,te:te+TEST];D[:,o]=np.mean((pn-yte)**2-(pr-yte)**2,axis=1)
 m=D.mean(1);l=np.mean((D-m[:,None])**2,1)
 for k in range(1,K+1):l+=2*(1-k/(K+1))*np.mean((D[:,k:]-m[:,None])*(D[:,:-k]-m[:,None]),1)
 return np.where(l>1e-14,np.sqrt(ORIGINS)*m/np.sqrt(l),0.)

def generate_batch(F,rng,mu,rho,sig,B):
 z=rng.r(B*N).reshape(B,N);E=np.empty((B,N));E[:,0]=sig[0]*z[:,0]
 for t in range(1,N):E[:,t]=rho*E[:,t-1]+sig[t]*z[:,t]
 return mu[None,:]+E

def run_filtered(args):
 if args.epsilon is None and args.ridge is None and args.direction is None:
  return run_matrix(args)
 eps_list=[args.epsilon] if args.epsilon is not None else list(EPS)
 ridge_list=[args.ridge] if args.ridge is not None else list(RIDGES)
 dir_list=[args.direction] if args.direction is not None else list(DIRS)
 return run_matrix(args, eps_list, ridge_list, dir_list)

def run_matrix(args, eps_list=None, ridge_list=None, dir_list=None):
 if eps_list is None: eps_list=list(EPS)
 if ridge_list is None: ridge_list=list(RIDGES)
 if dir_list is None: dir_list=list(DIRS)
 rng=SM(args.seed);cells=[]
 for eps in eps_list:
  for ridge in ridge_list:
   for dname in dir_list:
    v=DIRS[dname]
    p_or=[];p_est=[];rooterr=[];rhoh=[]
    oracle_roots=0; estimated_roots=0; disc_negative=0; no_positive_nonneg_disc=0
    for _ in range(args.mc):
     F=feat(rng,eps);m=mu0(F);sig=sig0(F);Q,b,c,ops=quad(F,ridge,m,.5,sig,v);ro=root(Q,b,c,v)
     if ro is None:continue
     oracle_roots+=1;y=generate_batch(F,rng,m+ro*(F[:,3:6]@v),.5,sig,1)[0];Tobs=statistic_batch(y[None,:],ops)[0]
     mh,rh,sh=est_nuisance(F,y);Qh,bh,ch,_=quad(F,ridge,mh,rh,sh,v);rh0=root(Qh,bh,ch,v)
     if rh0 is None:
      qA=float(v@Qh@v); qB=float(bh@v); qC=float(ch); disc=qB*qB-4*qA*qC
      if np.isfinite(disc) and disc < -1e-12: disc_negative+=1
      elif np.isfinite(disc): no_positive_nonneg_disc+=1
      continue
     estimated_roots+=1
     mu_est=mh+rh0*(F[:,3:6]@v);mu_or=m+ro*(F[:,3:6]@v)
     Yest=generate_batch(F,rng,mu_est,rh,sh,args.bootstrap);Yor=generate_batch(F,rng,mu_or,.5,sig,args.bootstrap)
     pest=np.mean(statistic_batch(Yest,ops)>=Tobs);por=np.mean(statistic_batch(Yor,ops)>=Tobs)
     p_est.append((pest*(args.bootstrap)+1)/(args.bootstrap+1));p_or.append((por*(args.bootstrap)+1)/(args.bootstrap+1));rooterr.append(rh0/ro);rhoh.append(rh)
    cells.append({"epsilon":eps,"ridge":ridge,"direction":dname,"datasets":len(p_est),"oracle_root_available_rate":oracle_roots/args.mc,"estimated_root_available_rate":estimated_roots/args.mc,"estimated_given_oracle_root_rate":(estimated_roots/oracle_roots) if oracle_roots else None,"estimated_root_failure_discriminant_negative":disc_negative,"estimated_root_failure_no_positive_nonnegative_discriminant":no_positive_nonneg_disc,"estimated_vs_oracle_root_ratio_median":float(np.median(rooterr)) if rooterr else None,"oracle_bootstrap_reject_rate":float(np.mean(np.asarray(p_or)<ALPHA)) if p_or else None,"estimated_nuisance_bootstrap_reject_rate":float(np.mean(np.asarray(p_est)<ALPHA)) if p_est else None,"median_rho_hat":float(np.median(rhoh)) if rhoh else None})
 return cells

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--mc',type=int,default=50);ap.add_argument('--bootstrap',type=int,default=49);ap.add_argument('--seed',type=int,default=20261024);ap.add_argument('--epsilon',type=float,default=None);ap.add_argument('--ridge',type=float,default=None);ap.add_argument('--direction',type=str,choices=sorted(DIRS),default=None);a=ap.parse_args();res=run_filtered(a)
 out={"schema":"rh006-estimated-nuisance-null-surface-bootstrap-executed/v4","status":"research-diagnostic-only","faithfulness_scope":"independent estimator-mechanics reimplementation; not the Symthaea Rust implementation","execution":{"command":f"python scripts/research/rh006_estimated_nuisance_null_surface_bootstrap_v4.py --mc {a.mc} --bootstrap {a.bootstrap} --seed {a.seed}","script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},"geometry":{"train":TRAIN,"gap":GAP,"test":TEST,"origins":ORIGINS,"step":STEP,"epsilons":list(EPS),"ridges":list(RIDGES),"bootstrap_reps":a.bootstrap,"mc_reps":a.mc,"oracle_rho":.5,"oracle_heteroskedastic":True,"alpha":ALPHA},"null_construction":"Each dataset is generated at the oracle finite-sample equal-risk root on a predeclared direction. The plug-in bootstrap estimates the baseline mean and AR/heteroskedastic innovation nuisance from that observed dataset, solves the equal-risk root using those estimates, and generates restricted-null bootstrap worlds from that estimated point. Bootstrap worlds are evaluated with the same rolling estimator and Bartlett studentizer.","limitations":["Mean-model nuisance is estimated from the full synthetic sample rather than a dedicated external reservoir.","Features remain fixed/conditionally generated; endogeneity is not covered.","This is a parametric restricted-bootstrap diagnostic, not a validated RH-006 inference procedure.","The estimated-nuisance critical distribution conditions on the observed plug-in nuisance; it does not claim the stronger double-bootstrap or uniform-composite-null result."],"results":res,"interpretation":{"applicability":"not-approved-for-execution","selection":"stop-assumption-failure","formal_inference":False,"finding":"Plug-in nuisance estimation is a distinct calibration layer: the finite-sample equal-risk null point itself can move substantially relative to the oracle point, especially in weakly identified near-singular directions. Mechanical bootstrap behavior therefore cannot be inherited from an oracle calibration.","next_gate":"increase Monte Carlo depth and bootstrap replicates for the worst cells, then compare the plug-in restricted bootstrap against a constrained/recentered alternative and a least-favorable/null-surface envelope before any formal execution."},"references":["Clark & McCracken (2015), DOI 10.1016/j.jeconom.2014.06.016","Clark & McCracken (2012), FRBSL WP 2011-024B","Chernozhukov, Hansen & Spindler (2015), post-selection/post-regularization inference"]}
 raw=json.dumps(out,sort_keys=True,separators=(",",":"));out['payload_sha256']=hashlib.sha256(raw.encode()).hexdigest();print(json.dumps(out,indent=2,sort_keys=True))
if __name__=='__main__':main()
