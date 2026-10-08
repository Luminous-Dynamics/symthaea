#!/usr/bin/env python3
"""RH-006 general linear cross-covariance operator diagnostic.

Research-only. q is a stacked 3-channel innovation vector with block-diagonal
Sigma_q. For an explicit linear operator K,
    E[e|q] = K q,
    Var(e|q) = Sigma_e - K Sigma_q K',
with fixed marginal Sigma_e. The implementation scales K so that
||Sigma_e^{-1/2} K Sigma_q^{1/2}||_2 < 1, making the conditional
covariance positive definite by construction.
"""
import argparse, hashlib, json, math
from collections import Counter
from pathlib import Path
import numpy as np

TRAIN,GAP,TEST,ORIGINS,STEP=48,4,16,24,16
N=TRAIN+GAP+TEST+(ORIGINS-1)*STEP
RIDGE=1e-4
DIRECTIONS={
    "weak":np.array([0.0,1.0,-2.0])/np.sqrt(5.0),
    "strong":np.array([1.0,1.0,0.5])/1.5,
}
FAMILIES=("single_rel","multi_rel","cross_channel","anti_rel","lag1","distributed_lag","full_lag")
STRENGTHS=(0.35,0.60,0.85)
CHANNEL_COV=np.array([[1.00,0.25,0.15],[0.25,1.00,-0.20],[0.15,-0.20,1.00]])

class SplitMix64:
    def __init__(self,seed): self.state=np.uint64(seed & 0xffffffffffffffff)
    def r(self,n):
        idx=self.state+np.arange(int(n),dtype=np.uint64)
        z=idx+np.uint64(0x9E3779B97F4A7C15)
        z=(z ^ (z>>np.uint64(30)))*np.uint64(0xBF58476D1CE4E5B9)
        z=(z ^ (z>>np.uint64(27)))*np.uint64(0x94D049BB133111EB)
        z ^= z>>np.uint64(31)
        self.state=np.uint64((int(self.state)+int(n)) & 0xffffffffffffffff)
        return np.where((z & np.uint64(1))==0,-1.0,1.0)

def ar_from_e(e,rho):
    x=np.empty(N); x[0]=e[0]
    for i in range(1,N): x[i]=rho*x[i-1]+e[i]
    return x

def feature_path(g):
    base=np.column_stack([g.r(N),g.r(N),g.r(N)])
    q=base@np.linalg.cholesky(CHANNEL_COV).T
    ind=[g.r(N) for _ in range(4)]
    raw=np.column_stack([
        ar_from_e(ind[0],0.35), ar_from_e(ind[1],0.35),
        ar_from_e(ind[2],0.35), ar_from_e(ind[3],0.35),
        ar_from_e(q[:,2],0.35), ar_from_e(q[:,0],0.35),
        ar_from_e(q[:,1],0.35),
    ])
    f=lambda x:np.clip(0.5+0.18*x,0.0,1.0)
    a,b,c=f(raw[:,0]),f(raw[:,1]),f(raw[:,2])
    alignment=f(0.65*raw[:,3]+0.35*raw[:,4])
    a2b,b2a=f(raw[:,5]),f(raw[:,6])
    turn=f(0.45*raw[:,4]+0.55*raw[:,5])
    F=np.column_stack([a,b,alignment,a2b,b2a,turn,c])
    return F,q

def mu0(F): return 0.15+0.45*F[:,0]+0.25*F[:,1]+0.20*F[:,6]+0.30*F[:,2]
def sig0(F): return 0.08*(1.0+0.7*(F[:,-1]-0.5))

def loading(sig):
    L=np.zeros((N,N))
    for t in range(N):
        L[t,t]=sig[t]
        if t: L[t,:t]=(0.5**np.arange(t,0,-1))*sig[:t]
    return L

def op(Xtr,Xte):
    m=Xtr.mean(0); sc=np.where(Xtr.std(0)<=1e-12,1.0,Xtr.std(0))
    A=np.c_[np.ones(len(Xtr)),(Xtr-m)/sc]
    B=np.c_[np.ones(len(Xte)),(Xte-m)/sc]
    G=A.T@A; G[1:,1:]+=RIDGE*np.eye(Xtr.shape[1])
    return B@np.linalg.solve(G,A.T)

def build_ops(F):
    nr,rel=[0,1,2,6],list(range(7)); out=[]
    for o in range(ORIGINS):
        s=o*STEP; te=o*STEP+TRAIN+GAP
        out.append((op(F[s:s+TRAIN,nr],F[te:te+TEST,nr]),op(F[s:s+TRAIN,rel],F[te:te+TEST,rel])))
    return out

def make_K_basis(kind,q):
    G=np.zeros((N,3*N))
    def add(lag,weights):
        for t in range(N):
            s=t-lag
            if s>=0:
                G[t,3*s:3*s+3]+=np.asarray(weights,float)
    if kind=="single_rel": add(0,[1,0,0])
    elif kind=="multi_rel": add(0,np.array([1,1,1])/np.sqrt(3.0))
    elif kind=="cross_channel": add(0,np.array([0,1,0])/np.sqrt(2.0)); add(1,np.array([1,0,0])/np.sqrt(2.0))
    elif kind=="anti_rel": add(0,np.array([1,-1,0])/np.sqrt(2.0))
    elif kind=="lag1": add(1,np.array([1,0,1])/np.sqrt(2.0))
    elif kind=="distributed_lag":
        add(0,[0.8,0.0,0.0]); add(1,[0.0,0.6,0.0]); add(2,[0.0,0.0,-0.5]); G/=np.sqrt(0.8**2+0.6**2+0.5**2)
    elif kind=="full_lag":
        add(0,[0.7,0.5,0.4]); add(1,[0.5,-0.4,0.3]); add(2,[-0.3,0.4,0.5]); G/=np.linalg.norm(G,ord='fro')*np.sqrt(N)
    else: raise ValueError(kind)
    return G

def surface_given_K(F,q,K):
    m,sig=mu0(F),sig0(F); L=loading(sig); Sigma=L@L.T
    Kcov=K.copy()
    for t in range(N): Kcov[:,3*t:3*t+3]=K[:,3*t:3*t+3]@CHANNEL_COV
    g=K@q.reshape(-1)
    V=(Sigma-Kcov@K.T); V=(V+V.T)/2
    min_v=float(np.min(np.linalg.eigvalsh(V)))
    Z=F[:,3:6]; ops=build_ops(F)
    Q=np.zeros((3,3)); b=np.zeros(3); c=0.0
    for o,(Wc,Wr) in enumerate(ops):
        s=o*STEP; te=o*STEP+TRAIN+GAP
        tr=slice(s,s+TRAIN); tt=slice(te,te+TEST)
        tri=np.arange(s,s+TRAIN); tti=np.arange(te,te+TEST)
        Hc=Wc@Z[tr]-Z[tt]; Hr=Wr@Z[tr]-Z[tt]
        dc=Wc@m[tr]-m[tt]+(Wc@g[tr]-g[tt]); dr=Wr@m[tr]-m[tt]+(Wr@g[tr]-g[tt])
        Q+=(Hc.T@Hc-Hr.T@Hr)/TEST
        b+=2*(Hc.T@dc-Hr.T@dr)/TEST
        c+=float((dc@dc-dr@dr)/TEST)
        Vtr=V[np.ix_(tri,tri)]; Vty=V[np.ix_(tri,tti)]
        vd=(np.trace(Wc@Vtr@Wc.T)-2*np.trace(Wc@Vty)-np.trace(Wr@Vtr@Wr.T)+2*np.trace(Wr@Vty))/TEST
        c+=float(vd)
    return (Q/ORIGINS,b/ORIGINS,c/ORIGINS,g,V,L,Sigma,ops,min_v,Kcov)

def metric_norm(G,L):
    Croot=np.linalg.cholesky(CHANNEL_COV); Gxs=G.copy()
    for tt in range(N): Gxs[:,3*tt:3*tt+3]=G[:,3*tt:3*tt+3]@Croot
    A=np.linalg.solve(L,Gxs)
    return float(np.sqrt(np.linalg.eigvalsh(A@A.T)[-1]))

def risk_surface(F,q,G,strength):
    L=loading(sig0(F))
    n=metric_norm(G,L); alpha=float(strength/max(n,1e-30))
    K=alpha*G
    Q,b,c,g,V,L,Sigma,ops,min_v,Kcov=surface_given_K(F,q,K)
    return Q,b,c,g,V,L,Sigma,ops,alpha,n,min_v,K

def delta(coeff,gamma): Q,b,c=coeff; return float(gamma@Q@gamma+b@gamma+c)

def roots(Q,b,c,u):
    A=float(u@Q@u); B=float(b@u); D=B*B-4*A*c
    if D<0: return []
    if abs(A)<1e-14: return [float(-c/B)] if abs(B)>1e-14 and -c/B>0 else []
    s=math.sqrt(max(D,0.0)); rr=[(-B-s)/(2*A),(-B+s)/(2*A)]
    return sorted(float(x) for x in rr if np.isfinite(x) and x>0)

def theorem(Q,b,c):
    Qs=(Q+Q.T)/2; ew=np.linalg.eigvalsh(Qs)
    if ew[0]<=1e-10: return "non-PD-Q",ew,None
    if c<-1e-10: return "all-directions-one-positive-root",ew,None
    qinv=float(b@np.linalg.solve(Qs,b)); margin=qinv-4*c
    if abs(c)<=1e-10: return "c-zero-boundary",ew,margin
    return ("cone-support" if margin>1e-10 else "no-discriminant-direction"),ew,margin

def spectrum_sig(Q,b,c):
    R=np.outer(b,b)-4*c*((Q+Q.T)/2); ev=np.linalg.eigvalsh((R+R.T)/2)
    sig=''.join('+' if x>1e-10 else '-' if x<-1e-10 else '0' for x in ev)
    return sig,ev

def loss_mc(Y,F,ops):
    vals=[]
    for o,(Wc,Wr) in enumerate(ops):
        s=o*STEP; te=o*STEP+TRAIN+GAP
        pc=Wc@Y[s:s+TRAIN]; pr=Wr@Y[s:s+TRAIN]; yt=Y[te:te+TEST]
        vals.append(np.mean((pc-yt)**2-(pr-yt)**2,axis=0))
    return np.mean(np.vstack(vals),axis=0)

def real_roots_in_unit_interval(coefs,upper=0.999999,tol=1e-9):
    nz=np.trim_zeros(np.asarray(coefs,float),'f')
    if len(nz)==0: return []
    rr=np.roots(nz); out=[]
    for x in rr:
        if abs(x.imag)<=tol and -tol<=x.real<=upper+tol:
            out.append(float(min(max(x.real,0.0),upper)))
    return sorted(set(round(x,12) for x in out))

def affine_strength_polynomial(F,q,G):
    L=loading(sig0(F)); n=metric_norm(G,L); Kbar=G/max(n,1e-30)
    Q0,b0,c0,*_=surface_given_K(F,q,np.zeros_like(G))
    hp=0.5
    Qp,bp,cp,*_=surface_given_K(F,q,hp*Kbar)
    Qm,bm,cm,*_=surface_given_K(F,q,-hp*Kbar)
    Q=(Q0+Q0.T)/2
    b1=(bp-bm)/(2*hp)
    c1=(cp-cm)/(2*hp)
    c2=(cp+cm-2*c0)/(2*hp*hp)
    qinv1=float(b0@np.linalg.solve(Q,b0))
    cross=float(2*b0@np.linalg.solve(Q,b1))
    qinv2=float(b1@np.linalg.solve(Q,b1))
    m0=qinv1-4*c0; m1=cross-4*c1; m2=qinv2-4*c2
    return {"b0":b0.tolist(),"b1":b1.tolist(),"c0":float(c0),"c1":float(c1),"c2":float(c2),"margin0":m0,"margin1":m1,"margin2":m2,"c_roots_0_1":real_roots_in_unit_interval([c2,c1,c0]),"margin_roots_0_1":real_roots_in_unit_interval([m2,m1,m0])}

def run(args):
    rng=SplitMix64(args.seed); summary={}; topo_rows=[]; mc_rows=[]; quad_err=[]; cov_err=[]; transitions=[]; dir_support=[]
    for pi in range(args.paths):
        F,q=feature_path(rng)
        for kind in FAMILIES:
            G=make_K_basis(kind,q)
            if pi < args.transition_paths:
                transitions.append({"path":pi,"family":kind,**affine_strength_polynomial(F,q,G)})
            for strength in STRENGTHS:
                Q,b,c,g,V,L,Sigma,ops,alpha,lmax,min_v,K=risk_surface(F,q,G,strength)
                case,ew,margin=theorem(Q,b,c); sig,rev=spectrum_sig(Q,b,c)
                expected=("+++" if case=="all-directions-one-positive-root" else "--+" if case=="cone-support" else "+00" if case=="c-zero-boundary" else "---" if case=="no-discriminant-direction" else None)
                topo_rows.append({"path":pi,"family":kind,"strength":strength,"case":case,"actual_signature":sig,"expected_signature":expected,"match":expected is None or expected==sig,"min_Q_eigen":float(ew[0]),"c":c,"qinv_minus_4c":margin,"K_scale":alpha,"K_explained_operator_norm":math.sqrt(lmax),"min_conditional_cov_eigen":min_v})
                dir_support.append({"path":pi,"family":kind,"strength":strength,**{d:bool(roots(Q,b,c,u)) for d,u in DIRECTIONS.items()}})
                coeff=(Q,b,c)
                u=np.array([0.2,-0.5,0.84261498]); u/=np.linalg.norm(u)
                v=np.array([-0.7,0.4,0.59160798]); v/=np.linalg.norm(v)
                dquad=delta(coeff,u+v)-delta(coeff,u)-delta(coeff,v)+delta(coeff,np.zeros(3))-2.0*float(u@Q@v)
                quad_err.append(abs(dquad))
                Kcov=K.copy()
                for tt in range(N): Kcov[:,3*tt:3*tt+3]=K[:,3*tt:3*tt+3]@CHANNEL_COV
                cov_err.append(float(np.max(np.abs(V-(Sigma-Kcov@K.T)))))
                if kind in ("single_rel","lag1"):
                    G2=make_K_basis("anti_rel" if kind=="single_rel" else "distributed_lag",q)
                    n1=metric_norm(G,L); n2=metric_norm(G2,L)
                    K1=(0.6/n1)*G; K2=(0.6/n2)*G2
                    vals=[]
                    gg=np.array([0.3,-0.6,0.74161985]); gg/=np.linalg.norm(gg)
                    for th in (0.,.25,.5,.75,1.):
                        Ki=(1-th)*K1+th*K2
                        cc=surface_given_K(F,q,Ki)[0:3]
                        vals.append(delta(cc,gg))
                    sec=[vals[i+2]-2*vals[i+1]+vals[i] for i in range(3)]
                    quad_err.append(max(abs(x-y) for x in sec for y in sec))
                if pi<args.mc_paths and strength in (0.60,):
                    for d,u0 in DIRECTIONS.items():
                        r0=roots(Q,b,c,u0)
                        if not r0: continue
                        gamma=r0[0]*u0
                        mean=mu0(F)+F[:,3:6]@gamma
                        chol=np.linalg.cholesky(V+1e-12*np.eye(N))
                        gr=SplitMix64(args.seed+700000+pi*1000+int(strength*1000)+(1 if d=="strong" else 0)+sum((j+1)*ord(c) for j,c in enumerate(kind))%997)
                        R=np.zeros((N,args.mc_outcomes))
                        for k in range(8): R += gr.r(args.mc_outcomes*N).reshape(args.mc_outcomes,N).T
                        R/=np.sqrt(8.0)
                        U=chol@R
                        Y=mean[:,None]+g[:,None]+U
                        emp=loss_mc(Y,F,ops)
                        ana=delta(coeff,gamma)
                        mc_rows.append({"path":pi,"family":kind,"strength":strength,"direction":d,"analytic":ana,"mc_mean":float(np.mean(emp)),"abs_error":abs(float(np.mean(emp))-ana)})
    counts=Counter((r["case"],r["actual_signature"]) for r in topo_rows)
    by_family={}
    for fam in FAMILIES:
        sub=[r for r in topo_rows if r["family"]==fam]
        by_family[fam]={"cells":len(sub),"case_counts":dict(Counter(r["case"] for r in sub)),"signature_counts":dict(Counter(r["actual_signature"] for r in sub)),"strength_case_map":{str(s):dict(Counter(r["case"] for r in sub if r["strength"]==s)) for s in STRENGTHS},"min_Q_eigen":min(r["min_Q_eigen"] for r in sub),"min_conditional_cov_eigen":min(r["min_conditional_cov_eigen"] for r in sub)}
    family_transitions={}
    for fam in FAMILIES:
        sub=[r for r in transitions if r["family"]==fam]
        roots_c=[x for r in sub for x in r["c_roots_0_1"]]
        roots_m=[x for r in sub for x in r["margin_roots_0_1"]]
        ds=[r for r in dir_support if r["family"]==fam]
        family_transitions[fam]={"paths":len(sub),"c_zero_strength_quantiles":np.quantile(roots_c,[0,.1,.5,.9,1]).tolist() if roots_c else [],"margin_zero_strength_quantiles":np.quantile(roots_m,[0,.1,.5,.9,1]).tolist() if roots_m else [],"directional_support_rate":{str(s):{d:sum(r[d] for r in ds if r["strength"]==s)/max(sum(1 for r in ds if r["strength"]==s),1) for d in DIRECTIONS} for s in STRENGTHS}}
    return {"schema":"rh006-general-k-joint-dgp/v2","status":"research-diagnostic-only","formal_inference":False,"applicability":"not-approved-for-execution","selection":"stop-assumption-failure","execution":{"paths":args.paths,"seed":args.seed,"families":list(FAMILIES),"strengths":list(STRENGTHS),"channels":3,"geometry":{"train":TRAIN,"gap":GAP,"test":TEST,"origins":ORIGINS,"step":STEP},"ridge":RIDGE,"mc_paths":args.mc_paths,"mc_outcomes":args.mc_outcomes,"transition_polynomial_probe":[-0.5,0.0,0.5]},"general_operator":{"q_definition":"stacked 3-channel innovation vector with block-diagonal Sigma_q using the declared channel covariance","sigma_q_channel":CHANNEL_COV.tolist(),"conditional_mean":"E[e|q]=Kq","conditional_covariance":"Var(e|q)=Sigma_e-K Sigma_q K'","marginal_covariance":"Sigma_e fixed from the declared AR(0.5)+heteroskedastic loading","operator_family":"explicit contemporaneous, cross-channel, and lagged convolution operators","matched_strength_definition":"strength = spectral norm of Sigma_e^{-1/2} K Sigma_q^{1/2}, so strength^2 is the maximum explained-covariance fraction under the fixed marginal Sigma_e","parameterization_result":"For affine K(theta)=K0+sum_j theta_j K_j, the conditional finite-sample loss surface is quadratic jointly in gamma and theta for fixed feature path and forecast operators."},"topology":{"cells":len(topo_rows),"theorem_spectrum_match_rate":sum(r["match"] for r in topo_rows)/len(topo_rows),"family_summary":by_family,"family_transition_summary":family_transitions,"signature_case_cross_tab":{"|".join(map(str,k)):v for k,v in counts.items()}},"mechanics":{"quadratic_identity_max_abs_error":max(quad_err),"conditional_covariance_identity_max_abs_error":max(cov_err),"mc_checks":len(mc_rows),"mc_max_abs_error":max((r["abs_error"] for r in mc_rows),default=0.0),"mc_mean_abs_error":float(np.mean([r["abs_error"] for r in mc_rows])) if mc_rows else 0.0},"sample_rows":topo_rows,"mc_rows":mc_rows,"nonclaims":["No formal p-value.","No confidence interval.","No uniform or least-favorable size guarantee.","No empirical validation.","No arbitrary-DGP validity claim.","No claim finite-sample inference follows from the mechanics identity.","No exact-Rust implementation qualification."]}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--paths',type=int,default=8); ap.add_argument('--seed',type=int,default=20261010); ap.add_argument('--mc-paths',type=int,default=2); ap.add_argument('--mc-outcomes',type=int,default=80); ap.add_argument('--transition-paths',type=int,default=4)
    a=ap.parse_args(); out=run(a); out['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(); raw=json.dumps(out,sort_keys=True,separators=(',',':')); out['payload_sha256']=hashlib.sha256(raw.encode()).hexdigest(); print(json.dumps(out,indent=2,sort_keys=True))
if __name__=='__main__': main()
