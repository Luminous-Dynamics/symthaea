#!/usr/bin/env python3
import itertools, json, math
from pathlib import Path
import numpy as np
import scipy.stats as st
import statsmodels.api as sm
from sklearn.isotonic import IsotonicRegression

out={}
# QR / OLS
Xraw=np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.],[2.,1.],[1.,2.],[3.,1.],[2.,3.]])
y=np.array([1.2,3.1,0.4,2.5,4.6,1.7,6.4,3.0])
X=np.column_stack([np.ones(len(Xraw)),Xraw])
beta=np.linalg.lstsq(X,y,rcond=None)[0]
cov_basis=np.linalg.inv(X.T@X)
fit=X@beta
res=y-fit
mse=np.sum(res**2)/(len(y)-X.shape[1])
h=np.diag(X@cov_basis@X.T)
cook=res**2*h/(X.shape[1]*mse*(1-h)**2)
press_res=res/(1-h)
out['ols']={'coefficients':beta.tolist(),'covariance_basis':cov_basis.tolist(),'mse':float(mse),'leverage':h.tolist(),'cooks_distance':cook.tolist(),'press':float(np.sum(press_res**2))}
# ridge standardized, intercept unpenalized
lam=2.5
means=Xraw.mean(axis=0); scales=np.sqrt(np.mean((Xraw-means)**2,axis=0)); Z=(Xraw-means)/scales
Za=np.column_stack([np.ones(len(Z)),Z]); P=np.diag([0.,1.,1.]); bstd=np.linalg.solve(Za.T@Za+lam*P,Za.T@y)
braw=np.r_[bstd[0]-np.sum(bstd[1:]*means/scales),bstd[1:]/scales]
inv=np.linalg.inv(Za.T@Za+lam*P); edf=len(bstd)-lam*np.trace(inv@P); sse=np.sum((y-X@braw)**2); gcv=len(y)*sse/(len(y)-edf)**2
out['ridge']={'coefficients':braw.tolist(),'edf':float(edf),'sse':float(sse),'gcv':float(gcv)}
# VIF and BP via statsmodels
vifs=[]
for j in range(Xraw.shape[1]):
    other=sm.add_constant(np.delete(Xraw,j,axis=1))
    r2=sm.OLS(Xraw[:,j],other).fit().rsquared
    vifs.append(1/(1-r2))
bp_aux=sm.OLS(res**2,X).fit(); bp=len(y)*bp_aux.rsquared
out['diagnostics']={'vif':vifs,'bp_statistic':float(bp),'bp_p':float(st.chi2.sf(bp,Xraw.shape[1]))}
# robust covariances
clusters=np.array([0,0,1,1,2,2,3,3])
ols=sm.OLS(y,X).fit()
cr=ols.get_robustcov_results(cov_type='cluster',groups=clusters,use_correction=True,df_correction=True)
hac=ols.get_robustcov_results(cov_type='HAC',maxlags=2,use_correction=False)
out['dependent_covariance']={'cluster_covariance':cr.cov_params().tolist(),'hac_covariance':hac.cov_params().tolist()}
# isotonic weighted
scores=np.array([.1,.2,.2,.4,.6,.8,.9]); labels=np.array([0,1,0,0,1,1,1]); weights=np.array([1.,2.,1.,1.,1.,2.,1.])
iso=IsotonicRegression(increasing=True,out_of_bounds='clip').fit(scores,labels,sample_weight=weights)
query=np.array([.1,.2,.3,.4,.7,.9])
out['isotonic']={'query':query.tolist(),'predictions':iso.predict(query).tolist()}
# conformal
obs=np.array([0.,1.,2.,3.,4.,5.]); pred=np.array([.2,.8,2.4,2.8,4.5,4.7]); confidence=.8
scores_abs=np.sort(np.abs(obs-pred)); k=min(len(scores_abs),max(1,math.ceil((len(scores_abs)+1)*confidence))); radius=scores_abs[k-1]
out['conformal']={'radius':float(radius),'interval_at_6':[float(6-radius),float(6+radius)]}
# AIPW
Y=np.array([4.,5.,1.,2.,6.,2.]); T=np.array([1,1,0,0,1,0],dtype=bool); e=np.array([.6,.7,.4,.3,.8,.2]); m0=np.array([1.5,2.,1.,2.,3.,2.]); m1=np.array([4.,5.,3.,4.,6.,4.])
psi=m1-m0+T*(Y-m1)/e-(~T)*(Y-m0)/(1-e); ate=psi.mean(); se=psi.std(ddof=1)/math.sqrt(len(psi))
out['aipw']={'scores':psi.tolist(),'ate':float(ate),'se':float(se)}
# DiD repeated cross section
Ydid=np.array([1.,1.2,2.,2.2,1.5,1.7,4.5,4.7]); Tdid=np.array([0,0,1,1,0,0,1,1]); Post=np.array([0,0,0,0,1,1,1,1]); Xdid=np.column_stack([np.ones(8),Tdid,Post,Tdid*Post]); did=sm.OLS(Ydid,Xdid).fit()
out['did']={'coefficient':float(did.params[3]),'standard_error':float(did.bse[3]),'t':float(did.tvalues[3]),'p':float(did.pvalues[3])}
# exact randomization
Yr=np.array([0.,1.,10.,11.]); obs_eff=10.; vals=[]
for comb in itertools.combinations(range(4),2):
    mask=np.zeros(4,dtype=bool); mask[list(comb)]=True; vals.append(float(Yr[mask].mean()-Yr[~mask].mean()))
out['randomization']={'effects':vals,'p_two_sided':sum(abs(v)>=abs(obs_eff) for v in vals)/len(vals)}
# KDE direct
vals=np.array([-2.,-1.,1.,2.]); h=.5; q=.75
pdf=np.mean(st.norm.pdf((q-vals)/h))/h; cdf=np.mean(st.norm.cdf((q-vals)/h))
out['density']={'pdf_at_0_75':float(pdf),'cdf_at_0_75':float(cdf),'dkw_epsilon_n100_95':float(math.sqrt(math.log(2/.05)/(2*100)))}
# residual diagnostics
r=np.array([.2,-.1,.3,-.2,.4,-.3,.1,-.4,.25,-.15])
lb=sm.stats.acorr_ljungbox(r,lags=[3],return_df=True,model_df=0)
jb=sm.stats.jarque_bera(r)
out['residual_diagnostics']={'ljung_box_statistic':float(lb['lb_stat'].iloc[0]),'ljung_box_p':float(lb['lb_pvalue'].iloc[0]),'jarque_bera_statistic':float(jb[0]),'jarque_bera_p':float(jb[1]),'skewness':float(jb[2]),'kurtosis':float(jb[3]-3)}
# SplitMix BCa exact mirror
MASK=(1<<64)-1
def next_u64(state):
    state=(state+0x9e3779b97f4a7c15)&MASK; v=state; v=((v^(v>>30))*0xbf58476d1ce4e5b9)&MASK; v=((v^(v>>27))*0x94d049bb133111eb)&MASK; return state,(v^(v>>31))&MASK
def idx(state,upper):
    threshold=(((-upper)&MASK)%upper)
    while True:
        state,c=next_u64(state)
        if c>=threshold:return state,c%upper
def type7(a,p):
    a=np.sort(np.array(a,float)); pos=p*(len(a)-1); lo=int(math.floor(pos)); hi=int(math.ceil(pos)); return float(a[lo]+(pos-lo)*(a[hi]-a[lo]))
xs=np.array([1.,2.,3.,10.]); B=256; state=42; reps=[]
for _ in range(B):
    draw=[]
    for _ in xs:
        state,j=idx(state,len(xs)); draw.append(xs[j])
    reps.append(float(np.mean(draw)))
observed=float(xs.mean()); less=sum(v<observed for v in reps); bp=(less+.5)/(B+1); z0=float(st.norm.ppf(bp)); jack=np.array([np.mean(np.delete(xs,i)) for i in range(len(xs))]); jm=jack.mean(); d=jm-jack; acc=float(np.sum(d**3)/(6*np.sum(d**2)**1.5)) if np.sum(d**2)>0 else 0.; alpha=.1
def adj(p):
    z=st.norm.ppf(p); shifted=z0+z; return float(st.norm.cdf(z0+shifted/(1-acc*shifted)))
lo,hi=sorted([adj(alpha/2),adj(1-alpha/2)]); interval=[type7(reps,lo),type7(reps,hi)]
out['bca']={'observed':observed,'bias_correction':z0,'acceleration':acc,'interval':interval,'replicate_mean':float(np.mean(reps)),'replicate_sd':float(np.std(reps,ddof=1))}
path=Path(__file__).with_name('v0_6_reference_results.json'); path.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); print(path)
