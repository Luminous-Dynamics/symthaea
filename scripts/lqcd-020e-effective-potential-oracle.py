#!/usr/bin/env python3
import hashlib,json,math,statistics as S
ID='wilson_effective_potential_declared_plateau_gls_v1'; V0=.43; B=8; DECL=(5,7); EARLY=(1,3); NEIGH=((4,6),(4,7))
def fixture(n=96):
 ts=list(range(1,9)); rows=[]
 for i in range(n):
  k=i+1; c=[.06*math.sin(2*math.pi*k/13),.04*math.cos(2*math.pi*k/11),.03*math.sin(2*math.pi*k/7),.025*math.cos(2*math.pi*k/5),.02*math.sin(2*math.pi*k/17),.018*math.cos(2*math.pi*k/19),.015*math.sin(2*math.pi*k/23)]; r=[]
  for t in ts:
   x=(t-4.5)/4; f=[1,x,x*x-.3,math.sin(.6*t),math.cos(.8*t),.4*((-1)**t),.5*math.sin(1.2*t)]; y=(.82*math.exp(-V0*t)+.22*math.exp(-(V0+1)*t))*math.exp(sum(a*b for a,b in zip(c,f)))
   if not math.isfinite(y) or y<=0: raise AssertionError('non-positive fixture')
   r.append(y)
  rows.append(r)
 return ts,rows
def means(rows):
 if not rows or len(rows[0])<2 or any(len(r)!=len(rows[0]) for r in rows): raise ValueError('shape')
 m=[S.mean(r[j] for r in rows) for j in range(len(rows[0]))]
 if any(not math.isfinite(x) or x<=0 for x in m): raise ValueError('mean loop')
 return m
def veff(rows):
 m=means(rows); return [math.log(m[i]/m[i+1]) for i in range(len(m)-1)]
def jack(rows,b):
 n=len(rows)
 if b<=0 or n<2*b or n%b: raise ValueError('blocks')
 full=veff(rows); reps=[veff(rows[:s]+rows[s+b:]) for s in range(0,n,b)]; q=len(reps); mr=[S.mean(r[j] for r in reps) for j in range(len(full))]; cov=[[0.0]*len(full) for _ in full]
 for i in range(len(full)):
  for j in range(len(full)): cov[i][j]=(q-1)/q*sum((r[i]-mr[i])*(r[j]-mr[j]) for r in reps)
 return full,reps,cov
def chol(a):
 n=len(a); l=[[0.0]*n for _ in range(n)]
 if n==0 or any(len(r)!=n for r in a): raise ValueError('cov')
 for i in range(n):
  for j in range(i+1):
   s=a[i][j]-sum(l[i][k]*l[j][k] for k in range(j))
   if i==j:
    if not math.isfinite(s) or s<=0: raise ValueError('spd')
    l[i][j]=math.sqrt(s)
   else:l[i][j]=s/l[j][j]
 return l
def solve(l,b):
 n=len(l); y=[0.0]*n; x=[0.0]*n
 for i in range(n): y[i]=(b[i]-sum(l[i][j]*y[j] for j in range(i)))/l[i][i]
 for i in range(n-1,-1,-1): x[i]=(y[i]-sum(l[j][i]*x[j] for j in range(i+1,n)))/l[i][i]
 return x
def fit(v,c,w):
 a,z=w
 if a<1 or z<a or z>len(v): raise ValueError('window')
 ix=list(range(a-1,z))
 if len(ix)<2: raise ValueError('short window')
 y=[v[i] for i in ix]; m=[[c[i][j] for j in ix] for i in ix]; l=chol(m); o=[1.0]*len(ix); iy=solve(l,y); io=solve(l,o); d=sum(io)
 if not math.isfinite(d) or d<=0: raise ValueError('gls')
 est=sum(iy)/d; se=math.sqrt(1/d); r=[x-est for x in y]; ir=solve(l,r); chi=sum(x*y for x,y in zip(r,ir)); dof=len(ix)-1
 return {'window':[a,z],'estimate':est,'standard_error':se,'chi2':chi,'dof':dof,'chi2_per_dof':chi/dof}
def main():
 ts,rows=fixture(); v,reps,c=jack(rows,B); d=fit(v,c,DECL); e=fit(v,c,EARLY); ns=[fit(v,c,w) for w in NEIGH]; mol=[S.mean(math.log(r[j]/r[j+1]) for r in rows) for j in range(len(ts)-1)]; diff=max(abs(a-b) for a,b in zip(v,mol))
 if abs(d['estimate']-V0)>.0025 or d['chi2_per_dof']>=.1 or e['chi2_per_dof']<=100 or diff<=1e-6: raise AssertionError('qualification fixture')
 out={'oracle_id':ID,'fixture':{'configurations':len(rows),'times':ts,'injected_ground_potential':V0,'block_size':B,'declared_window':list(DECL),'early_window':list(EARLY),'neighbor_windows':[list(w) for w in NEIGH]},'mean_wilson_loops':means(rows),'effective_potential':v,'effective_potential_standard_errors':[math.sqrt(c[i][i]) for i in range(len(v))],'declared_plateau_fit':d,'early_window_fit':e,'neighbor_window_fits':ns,'ratio_of_means_vs_mean_of_logs_max_abs_difference':diff,'jackknife_block_count':len(reps)}; s=json.dumps(out,sort_keys=True,separators=(',',':')); h=hashlib.sha256(s.encode()).hexdigest(); expected='1203f4edc0ae4bbf376a4e0dc20b15328a59e576c4ce51e6e624ab946b9fddb3'
 if h!=expected: raise AssertionError((h,expected))
 print('ok'); print('result_sha256='+h); print(s)
if __name__=='__main__':main()
