#!/usr/bin/env python3
import hashlib,json,math
ID='correlated_declared_range_lattice_cornell_gls_v1'
RV=((1,0,0),(2,0,0),(3,0,0),(1,1,0),(1,1,1),(2,1,0))
CL=(1.081520691702028,0.5389673179754996,0.3461467982875027,0.6935602595349944,0.5476259824378233,0.4515341044657458)
INJECT=(.7,.18,.25)
OFF=(.0010,-.0015,.0008,-.0007,.0012,-.0004)
def chol(a):
 n=len(a); l=[[0.0]*n for _ in range(n)]
 for i in range(n):
  for j in range(i+1):
   s=a[i][j]-sum(l[i][k]*l[j][k] for k in range(j))
   if i==j:
    if s<=0 or not math.isfinite(s): raise ValueError('spd')
    l[i][j]=math.sqrt(s)
   else:l[i][j]=s/l[j][j]
 return l
def solve(l,b):
 n=len(l); y=[0.0]*n; x=[0.0]*n
 for i in range(n): y[i]=(b[i]-sum(l[i][j]*y[j] for j in range(i)))/l[i][i]
 for i in range(n-1,-1,-1): x[i]=(y[i]-sum(l[j][i]*x[j] for j in range(i+1,n)))/l[i][i]
 return x
def inv(a):
 l=chol(a); n=len(a); cols=[]
 for k in range(n):
  e=[0.0]*n; e[k]=1.0; cols.append(solve(l,e))
 return [[cols[j][i] for j in range(n)] for i in range(n)]
def mm(a,b): return [[sum(a[i][k]*b[k][j] for k in range(len(b))) for j in range(len(b[0]))] for i in range(len(a))]
def mt(a): return [list(x) for x in zip(*a)]
def mv(a,v): return [sum(x*y for x,y in zip(row,v)) for row in a]
def gls(y,c,x):
 ci=inv(c); xt=mt(x); normal=mm(mm(xt,ci),x); rhs=mv(mm(xt,ci),y); cov=inv(normal); p=mv(cov,rhs); pred=mv(x,p); res=[a-b for a,b in zip(y,pred)]; wr=mv(ci,res); chi=sum(a*b for a,b in zip(res,wr)); dof=len(y)-len(p)
 return {'parameters':p,'parameter_covariance':cov,'chi2':chi,'dof':dof,'chi2_per_dof':chi/dof}
def main():
 r=[math.sqrt(sum(x*x for x in v)) for v in RV]; v0,sig,e=INJECT; y=[v0+sig*rr-e*cc+off for rr,cc,off in zip(r,CL,OFF)]; sd=.004; c=[[sd*sd*(.45**abs(i-j)) for j in range(len(RV))] for i in range(len(RV))]
 xl=[[1.0,rr,-cc] for rr,cc in zip(r,CL)]; xc=[[1.0,rr,-1.0/rr] for rr in r]; lat=gls(y,c,xl); cont=gls(y,c,xc)
 if lat['chi2_per_dof']>=1 or cont['chi2_per_dof']<=5: raise AssertionError('negative control')
 if abs(lat['parameters'][1]-sig)>.005: raise AssertionError('sigma recovery')
 out={'oracle_id':ID,'declared_separation_vectors':[list(v) for v in RV],'lattice_coulomb_values':list(CL),'observed_potential':y,'covariance':c,'injected_parameters':{'v0':v0,'sigma':sig,'e':e},'lattice_coulomb_fit':lat,'continuum_coulomb_negative_control':cont}; s=json.dumps(out,sort_keys=True,separators=(',',':')); h=hashlib.sha256(s.encode()).hexdigest(); expected='9e150c0a33e887c87e13ff7e263bfb76717242c48e9b93b9f1078f6352ffa4a5'
 if h!=expected: raise AssertionError((h,expected))
 print('ok'); print('result_sha256='+h); print(s)
if __name__=='__main__':main()
