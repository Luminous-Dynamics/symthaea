#!/usr/bin/env python3
import hashlib,json,math

ID="declared_lattice_potential_fit_family_gls_v1"
RV=((1,0,0),(2,0,0),(3,0,0),(1,1,0),(1,1,1),(2,1,0))
CL=(1.081520691702028,0.5389673179754996,0.3461467982875027,0.6935602595349944,0.5476259824378233,0.4515341044657458)
INJECT=(0.7,0.18,0.25,0.04)
OFF=(0.0010,-0.0015,0.0008,-0.0007,0.0012,-0.0004)
E_IR=math.pi/12.0

def chol(a):
 n=len(a); l=[[0.0]*n for _ in range(n)]
 for i in range(n):
  for j in range(i+1):
   s=a[i][j]-sum(l[i][k]*l[j][k] for k in range(j))
   if i==j:
    if s<=0 or not math.isfinite(s): raise ValueError("spd")
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
def mt(a): return [list(x) for x in zip(*a)]
def mm(a,b): return [[sum(a[i][k]*b[k][j] for k in range(len(b))) for j in range(len(b[0]))] for i in range(len(a))]
def mv(a,v): return [sum(x*y for x,y in zip(row,v)) for row in a]
def gls(y,c,x):
 ci=inv(c); xt=mt(x); normal=mm(mm(xt,ci),x); cov=inv(normal); rhs=mv(mm(xt,ci),y); p=mv(cov,rhs)
 pred=mv(x,p); res=[a-b for a,b in zip(y,pred)]; wr=mv(ci,res); chi=sum(a*b for a,b in zip(res,wr)); dof=len(y)-len(p)
 return {"parameters":p,"parameter_covariance":cov,"chi2":chi,"dof":dof,"chi2_per_dof":chi/dof}

def main():
 r=[math.sqrt(sum(x*x for x in v)) for v in RV]
 v0,sig,e,l=INJECT
 y=[v0+sig*rr-e*cc+l*(cc-1.0/rr)+off for rr,cc,off in zip(r,CL,OFF)]
 sd=.004
 c=[[sd*sd*(.45**abs(i-j)) for j in range(len(RV))] for i in range(len(RV))]
 x4=[[1.0,rr,-cc,cc-1.0/rr] for rr,cc in zip(r,CL)]
 free4=gls(y,c,x4)
 y3=[yy+E_IR*cc for yy,cc in zip(y,CL)]
 x3=[[1.0,rr,cc-1.0/rr] for rr,cc in zip(r,CL)]
 fixed_e3=gls(y3,c,x3)
 x2=[[1.0,rr] for rr in r]
 fixed_e_l0_2=gls(y3,c,x2)
 if free4["chi2_per_dof"]>=1.0: raise AssertionError("free4")
 if fixed_e3["chi2_per_dof"]>=1.0: raise AssertionError("fixed_e3")
 if fixed_e_l0_2["chi2_per_dof"]<=1.0: raise AssertionError("negative-control constraint")
 sigmas=[free4["parameters"][1],fixed_e3["parameters"][1],fixed_e_l0_2["parameters"][1]]
 if max(abs(s-INJECT[1]) for s in sigmas)>.006: raise AssertionError(("sigma stability",sigmas))
 out={
  "oracle_id":ID,
  "declared_separation_vectors":[list(v) for v in RV],
  "lattice_coulomb_values":list(CL),
  "observed_potential":y,
  "covariance":c,
  "injected_parameters":{"v0":v0,"sigma":sig,"e":e,"l":l},
  "fixed_ir_charge":E_IR,
  "free_4_parameter":free4,
  "fixed_e_3_parameter":fixed_e3,
  "fixed_e_l0_2_parameter":fixed_e_l0_2,
  "sigma_estimates":sigmas,
 }
 text=json.dumps(out,sort_keys=True,separators=(",",":"))
 digest=hashlib.sha256(text.encode()).hexdigest()
 print("ok"); print("result_sha256="+digest); print(text)

if __name__=="__main__": main()
