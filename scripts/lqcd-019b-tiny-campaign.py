#!/usr/bin/env python3
import math, statistics, struct, hashlib, json
from statistics import NormalDist
MASK32=0xffffffff; PAIRS=((0,1),(0,2),(1,2)); DIMS=(2,2,2,2); BETA=5.7
SEED=bytes([0x5A])*32; ENSEMBLE_SLOT=31; DOMAIN_TRANSITION=1; DOMAIN_INITIALIZATION=4
BURN_IN=60; STRIDE=2; MEASUREMENTS=24; FLOW_DT=1e-3; FLOW_STEPS=4

def rotl32(x,n): return ((x<<n)&MASK32)|(x>>(32-n))
def quarter_round(x,a,b,c,d):
    x[a]=(x[a]+x[b])&MASK32; x[d]^=x[a]; x[d]=rotl32(x[d],16); x[c]=(x[c]+x[d])&MASK32; x[b]^=x[c]; x[b]=rotl32(x[b],12); x[a]=(x[a]+x[b])&MASK32; x[d]^=x[a]; x[d]=rotl32(x[d],8); x[c]=(x[c]+x[d])&MASK32; x[b]^=x[c]; x[b]=rotl32(x[b],7)
def chacha8_block(seed,counter,stream):
    state=[0x61707865,0x3320646e,0x79622d32,0x6b206574]+list(struct.unpack('<8I',seed))+[counter&MASK32,(counter>>32)&MASK32,stream&MASK32,(stream>>32)&MASK32]; work=state.copy()
    for _ in range(4):
        quarter_round(work,0,4,8,12); quarter_round(work,1,5,9,13); quarter_round(work,2,6,10,14); quarter_round(work,3,7,11,15); quarter_round(work,0,5,10,15); quarter_round(work,1,6,11,12); quarter_round(work,2,7,8,13); quarter_round(work,3,4,9,14)
    return struct.pack('<16I',*[(work[i]+state[i])&MASK32 for i in range(16)])
def pack_stream_id(domain,slot,replica,rank=0): return (domain<<56)|(slot<<32)|(replica<<16)|rank
class ChaCha8Stream:
    def __init__(self,seed,stream): self.seed=seed; self.stream=stream; self.counter=0; self.buffer=b''; self.cursor=0; self.draws=0
    def _bytes(self,count):
        out=bytearray()
        while len(out)<count:
            if self.cursor>=len(self.buffer): self.buffer=chacha8_block(self.seed,self.counter,self.stream); self.counter+=1; self.cursor=0
            take=min(count-len(out),len(self.buffer)-self.cursor); out.extend(self.buffer[self.cursor:self.cursor+take]); self.cursor+=take
        return bytes(out)
    def open01(self):
        self.draws+=1; v=struct.unpack('<Q',self._bytes(8))[0]; return ((v>>12)+1)/((1<<52)+1)

def eye3(): return [[1+0j if i==j else 0j for j in range(3)] for i in range(3)]
def zero3(): return [[0j]*3 for _ in range(3)]
def add(a,b): return [[a[i][j]+b[i][j] for j in range(3)] for i in range(3)]
def scale(c,a): return [[c*a[i][j] for j in range(3)] for i in range(3)]
def mul(a,b): return [[sum(a[i][k]*b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
def dagger(a): return [[a[j][i].conjugate() for j in range(3)] for i in range(3)]
def trace(a): return a[0][0]+a[1][1]+a[2][2]
def field_copy(f): return [[row[:] for row in m] for m in f]
def frobenius(a): return math.sqrt(sum(abs(a[i][j])**2 for i in range(3) for j in range(3)))
def matrix_exp(a,terms=32):
    norm=frobenius(a); sq=max(0,math.ceil(math.log2(norm/0.5))) if norm>0.5 else 0; x=scale(1/(2**sq),a); out=eye3(); term=eye3()
    for k in range(1,terms+1): term=scale(1/k,mul(term,x)); out=add(out,term)
    for _ in range(sq): out=mul(out,out)
    return out
I=1j; SQRT3=math.sqrt(3)
G=[[[0,1,0],[1,0,0],[0,0,0]],[[0,-I,0],[I,0,0],[0,0,0]],[[1,0,0],[0,-1,0],[0,0,0]],[[0,0,1],[0,0,0],[1,0,0]],[[0,0,-I],[0,0,0],[I,0,0]],[[0,0,0],[0,0,1],[0,1,0]],[[0,0,0],[0,0,-I],[0,I,0]],[[1/SQRT3,0,0],[0,1/SQRT3,0],[0,0,-2/SQRT3]]]; G=[[[complex(v) for v in r] for r in m] for m in G]
def sites(d=DIMS):
    for x in range(d[0]):
      for y in range(d[1]):
       for z in range(d[2]):
        for t in range(d[3]): yield (x,y,z,t)
def site_index(s,d=DIMS): x,y,z,t=s; return (((x*d[1]+y)*d[2]+z)*d[3]+t)
def shift(s,mu,step,d=DIMS): o=list(s); o[mu]=(o[mu]+step)%d[mu]; return tuple(o)
def identity_field(d=DIMS): return [eye3() for _ in range(math.prod(d)*4)]
def link(f,s,mu,d=DIMS): return f[site_index(s,d)*4+mu]
def set_link(f,s,mu,v,d=DIMS): f[site_index(s,d)*4+mu]=v
def plaquette(f,s,mu,nu,d=DIMS):
    sm=shift(s,mu,1,d); sn=shift(s,nu,1,d); return mul(mul(mul(link(f,s,mu,d),link(f,sm,nu,d)),dagger(link(f,sn,mu,d))),dagger(link(f,s,nu,d)))
def avg_plaq(f,d=DIMS):
    v=[trace(plaquette(f,s,mu,nu,d)).real/3 for s in sites(d) for mu in range(4) for nu in range(mu+1,4)]; return sum(v)/len(v)
def embedded(pair,q):
    a0,a1,a2,a3=q; o=eye3(); i,j=pair; o[i][i]=a0+1j*a3; o[i][j]=a2+1j*a1; o[j][i]=-a2+1j*a1; o[j][j]=a0-1j*a3; return o
def q_from_emb(pair,m):
    i,j=pair; return ((m[i][i].real+m[j][j].real)/2,(m[i][j].imag+m[j][i].imag)/2,(m[i][j].real-m[j][i].real)/2,(m[i][i].imag-m[j][j].imag)/2)
def draw_rotation(src,pair,max_angle):
    uz,uphi,ua=src.open01(),src.open01(),src.open01(); z=2*uz-1; phi=2*math.pi*uphi; r=math.sqrt(max(0,1-z*z)); a=max_angle*(2*ua-1); c,s=math.cos(a),math.sin(a); return embedded(pair,(c,s*r*math.cos(phi),s*r*math.sin(phi),s*z))
def disordered(replica,rounds=2,d=DIMS):
    f=identity_field(d); src=ChaCha8Stream(SEED,pack_stream_id(DOMAIN_INITIALIZATION,ENSEMBLE_SLOT,replica))
    for _ in range(rounds):
      for s in sites(d):
       for mu in range(4):
        for p in PAIRS: set_link(f,s,mu,mul(draw_rotation(src,p,math.pi),link(f,s,mu,d)),d)
    return f

def affected(s,mu,d=DIMS):
    out=set()
    for nu in range(4):
        if nu==mu: continue
        a,b=sorted((mu,nu)); out.add((s,a,b)); out.add((shift(s,nu,-1,d),a,b))
    return sorted(out)
def touching_trace(f,s,mu,d=DIMS): return sum(trace(plaquette(f,b,a,c,d)).real for b,a,c in affected(s,mu,d))
def evaluated(f,s,mu,p,q,d=DIMS):
    c=field_copy(f); set_link(c,s,mu,mul(embedded(p,q),link(c,s,mu,d)),d); return touching_trace(c,s,mu,d)
def probe_force(f,s,mu,p,d=DIMS):
    plus=evaluated(f,s,mu,p,(1,0,0,0),d); minus=evaluated(f,s,mu,p,(-1,0,0,0),d); c=.5*(plus+minus); return c,(.5*(plus-minus),evaluated(f,s,mu,p,(0,1,0,0),d)-c,evaluated(f,s,mu,p,(0,0,1,0),d)-c,evaluated(f,s,mu,p,(0,0,0,1),d)-c)
def staple(f,s,mu,d=DIMS):
    st=zero3(); spm=shift(s,mu,1,d)
    for nu in range(4):
        if nu==mu: continue
        spn=shift(s,nu,1,d); st=add(st,mul(mul(link(f,spm,nu,d),dagger(link(f,spn,mu,d))),dagger(link(f,s,nu,d))))
        smn=shift(s,nu,-1,d); smnpm=shift(smn,mu,1,d); st=add(st,mul(mul(dagger(link(f,smnpm,nu,d)),dagger(link(f,smn,mu,d))),link(f,smn,nu,d)))
    return st
def spectator(p): return {(0,1):2,(0,2):1,(1,2):0}[p]
def staple_force(f,s,mu,p,d=DIMS):
    x=mul(link(f,s,mu,d),staple(f,s,mu,d)); i,j=p; k=spectator(p); return x[k][k].real,(x[i][i].real+x[j][j].real,-x[j][i].imag-x[i][j].imag,x[j][i].real-x[i][j].real,-x[i][i].imag+x[j][j].imag)
def kp(src,alpha,max_attempts=256):
    if alpha==0:
        for n in range(1,max_attempts+1):
            a0=2*src.open01()-1
            if src.open01()<math.sqrt(max(0,1-a0*a0)): return a0,n
        raise RuntimeError
    for n in range(1,max_attempts+1):
        r0,r1,r2,r3=(src.open01() for _ in range(4)); x1=-math.log(r1)/alpha; x2=-math.log(r2)/alpha; c=math.cos(2*math.pi*r3); dd=x2+x1*c*c; th=1-.5*dd
        if th>0 and r0*r0<th:
            a0=1-dd
            if -1<=a0<=1: return a0,n
    raise RuntimeError
def su2_hb(src,alpha):
    a0,n=kp(src,alpha); z=2*src.open01()-1; phi=2*math.pi*src.open01(); rr=math.sqrt(max(0,1-z*z)); rad=math.sqrt(max(0,1-a0*a0)); return (a0,rad*rr*math.cos(phi),rad*rr*math.sin(phi),rad*z),n
def hb_step(f,s,mu,p,beta,src,d=DIMS):
    _,q=staple_force(f,s,mu,p,d); rho=math.sqrt(sum(v*v for v in q)); can,n=su2_hb(src,beta*rho/3)
    if rho>1e-14:
        qh=tuple(v/rho for v in q); rot=mul(embedded(p,qh),embedded(p,can)); ori=q_from_emb(p,rot); proj=sum(q[i]*ori[i] for i in range(4))/rho
        if abs(proj-can[0])>1e-9: raise AssertionError('orientation')
    else: rot=embedded(p,can)
    set_link(f,s,mu,mul(rot,link(f,s,mu,d)),d); return n
def hb_sweep(f,beta,src,d=DIMS):
    n=0
    for s in sites(d):
      for mu in range(4):
       for p in PAIRS: n+=hb_step(f,s,mu,p,beta,src,d)
    return n
def or_step(f,s,mu,p,d=DIMS):
    before=touching_trace(f,s,mu,d); _,q=staple_force(f,s,mu,p,d); norm2=sum(v*v for v in q)
    if norm2<=1e-28: return
    r=[2*q[0]*v/norm2 for v in q]; r[0]-=1; set_link(f,s,mu,mul(embedded(p,r),link(f,s,mu,d)),d); after=touching_trace(f,s,mu,d)
    if abs(after-before)>2e-9: raise AssertionError(('micro',after-before))
def or_sweep(f,d=DIMS):
    for s in sites(d):
      for mu in range(4):
       for p in PAIRS: or_step(f,s,mu,p,d)
def hbor(f,beta,src,d=DIMS): n=hb_sweep(f,beta,src,d); or_sweep(f,d); return n

def polyakov(f,d=DIMS):
    vals=[]
    for x in range(d[0]):
      for y in range(d[1]):
       for z in range(d[2]):
        p=eye3()
        for t in range(d[3]): p=mul(p,link(f,(x,y,z,t),3,d))
        vals.append(trace(p)/3)
    return sum(vals)/len(vals)
def wilson_t1(f,d=DIMS): return sum(sum(trace(plaquette(f,s,sp,3,d)).real/3 for s in sites(d))/math.prod(d) for sp in (0,1,2))/3
def center_sector(p):
    if abs(p)<1e-15: return 0
    phase=math.atan2(p.imag,p.real); centers=[0.0,2*math.pi/3,-2*math.pi/3]
    def adiff(a,b): return abs(math.atan2(math.sin(a-b),math.cos(a-b)))
    return min(range(3),key=lambda k:adiff(phase,centers[k]))
def center_aligned(p):
    centers=[0.0,2*math.pi/3,-2*math.pi/3]; k=center_sector(p); rot=complex(math.cos(-centers[k]),math.sin(-centers[k])); return p*rot

def oriented_link(f,s,sd,d=DIMS):
    if sd>0: mu=sd-1; return link(f,s,mu,d),shift(s,mu,1,d)
    mu=-sd-1; prev=shift(s,mu,-1,d); return dagger(link(f,prev,mu,d)),prev
def transporter(f,s,dirs,d=DIMS):
    v=eye3()
    for sd in dirs: e,s=oriented_link(f,s,sd,d); v=mul(v,e)
    return v,s
def clover_sum(f,s,mu,nu,d=DIMS):
    a,b=mu+1,nu+1; loops=[[a,b,-a,-b],[b,-a,-b,a],[-a,-b,a,b],[-b,a,b,-a]]; out=zero3()
    for loop in loops: out=add(out,transporter(f,s,loop,d)[0])
    return out
def clover_f(f,s,mu,nu,d=DIMS):
    c=clover_sum(f,s,mu,nu,d); cd=dagger(c); ff=scale(1/(8j),[[c[i][j]-cd[i][j] for j in range(3)] for i in range(3)]); sing=trace(ff)/3
    for k in range(3): ff[k][k]-=sing
    return ff
def q_charge(f,d=DIMS):
    total=0
    for s in sites(d):
        f01=clover_f(f,s,0,1,d); f02=clover_f(f,s,0,2,d); f03=clover_f(f,s,0,3,d); f12=clover_f(f,s,1,2,d); f13=clover_f(f,s,1,3,d); f23=clover_f(f,s,2,3,d)
        total+=(trace(mul(f01,f23)).real-trace(mul(f02,f13)).real+trace(mul(f03,f12)).real)/(4*math.pi*math.pi)
    return total
def energy(f,d=DIMS):
    total=0
    for s in sites(d):
      for mu in range(4):
       for nu in range(mu+1,4):
        ff=clover_f(f,s,mu,nu,d); total+=trace(mul(ff,ff)).real
    return total/math.prod(d)
def analytic_grad(f,d=DIMS):
    out={}
    for s in sites(d):
      for mu in range(4):
        x=mul(link(f,s,mu,d),staple(f,s,mu,d)); out[(s,mu)]=[trace(mul(g,x)).imag/3 for g in G]
    return out
def flow_step(f,d=DIMS):
    grad=analytic_grad(f,d); out=field_copy(f)
    for (s,mu),comps in grad.items():
        h=zero3()
        for c,g in zip(comps,G): h=add(h,scale(c,g))
        set_link(out,s,mu,mul(matrix_exp(scale(-1j*FLOW_DT,h)),link(f,s,mu,d)),d)
    return out
def flow_history(f,d=DIMS):
    x=field_copy(f); out=[]
    for k in range(1,FLOW_STEPS+1): x=flow_step(x,d); out.append({'t':k*FLOW_DT,'energy':energy(x,d),'q':q_charge(x,d)})
    return out

def autocorr(xs,lag):
    m=statistics.mean(xs); den=sum((x-m)**2 for x in xs)
    if den==0:return None
    return sum((xs[i]-m)*(xs[i+lag]-m) for i in range(len(xs)-lag))/den
def tau(xs):
    maxlag=min(len(xs)//2,len(xs)-1); r=[autocorr(xs,k) for k in range(1,maxlag+1)]
    if not r or r[0] is None:return None
    tot=0; k=0
    while k<len(r):
        pair=r[k]+(r[k+1] if k+1<len(r) else 0)
        if pair<=0:break
        tot+=pair; k+=2
    return .5+tot
def ess(xs):
    t=tau(xs); return None if t is None else len(xs)/(2*t)
def split_rhat(chains):
    n0=len(chains[0]); n=n0//2; sp=[]
    for c in chains: sp.extend((c[:n],c[n:]))
    means=[statistics.mean(c) for c in sp]; vars=[statistics.variance(c) for c in sp]; W=statistics.mean(vars)
    if W==0:return None
    mm=statistics.mean(means); B=n*sum((m-mm)**2 for m in means)/(len(sp)-1); return math.sqrt((((n-1)/n)*W+B/n)/W)
def ranks(vals):
    order=sorted(range(len(vals)),key=lambda i:vals[i]); out=[0.]*len(vals); i=0
    while i<len(order):
        j=i+1
        while j<len(order) and vals[order[j]]==vals[order[i]]:j+=1
        rr=(i+1+j)/2
        for k in range(i,j):out[order[k]]=rr
        i=j
    return out
def ranknorm(chains):
    flat=[x for c in chains for x in c]; rs=ranks(flat); N=len(flat); nd=NormalDist(); z=[nd.inv_cdf((r-.375)/(N+.25)) for r in rs]; out=[]; p=0
    for c in chains: out.append(z[p:p+len(c)]); p+=len(c)
    return out
def rankfold(chains):
    bulk=split_rhat(ranknorm(chains)); med=statistics.median([x for c in chains for x in c]); fold=split_rhat(ranknorm([[abs(x-med) for x in c] for c in chains])); return bulk,fold,max(x for x in (bulk,fold) if x is not None)
def diag(series):
    b,f,m=rankfold(series); return {'means':[statistics.mean(s) for s in series],'classical_split_rhat':split_rhat(series),'rank_rhat':b,'folded_rhat':f,'max_rank_folded_rhat':m,'ess':[ess(s) for s in series],'tau_int':[tau(s) for s in series]}
def run_chain(start,replica):
    f=identity_field() if start=='cold' else disordered(replica); src=ChaCha8Stream(SEED,pack_stream_id(DOMAIN_TRANSITION,ENSEMBLE_SLOT,replica)); keep={BURN_IN+STRIDE*i for i in range(1,MEASUREMENTS+1)}; final=BURN_IN+STRIDE*MEASUREMENTS; obs=[]; attempts=0
    for cyc in range(1,final+1):
        attempts+=hbor(f,BETA,src)
        if cyc in keep:
            p=polyakov(f); pa=center_aligned(p); obs.append({'cycle':cyc,'plaquette':avg_plaq(f),'polyakov_re':p.real,'polyakov_im':p.imag,'polyakov_abs':abs(p),'polyakov_phase':math.atan2(p.imag,p.real),'polyakov_center_sector':center_sector(p),'polyakov_center_aligned_re':pa.real,'wilson_t1':wilson_t1(f),'flow':flow_history(f)})
    return {'start':start,'replica':replica,'uniform_draws':src.draws,'scalar_attempts':attempts,'observations':obs}
def parity():
    f=disordered(9,1); e=0
    for p in PAIRS:
        c1,q1=probe_force(f,(0,0,0,0),0,p); c2,q2=staple_force(f,(0,0,0,0),0,p); e=max(e,abs(c1-c2),*(abs(a-b) for a,b in zip(q1,q2)))
    return e
def main():
    zero=bytes.fromhex('3e00ef2f895f40d67f5bb8e81f09a5a12c840ec3ce9a7f3b181be188ef711a1e984ce172b9216f419f445367456d5619314a42a3da86b001387bfdb80e0cfe42'); assert chacha8_block(bytes(32),0,0)==zero
    pe=parity(); assert pe<3e-12
    chains=[run_chain('cold',0),run_chain('disordered',1)]
    keys=['plaquette','polyakov_re','polyakov_im','polyakov_abs','polyakov_center_aligned_re','wilson_t1']; diagnostics={k:diag([[o[k] for o in c['observations']] for c in chains]) for k in keys}
    diagnostics['polyakov_center_sector']=[]
    for c in chains:
        sec=[o['polyakov_center_sector'] for o in c['observations']]; counts={str(k):sec.count(k) for k in (0,1,2)}; transitions=sum(a!=b for a,b in zip(sec,sec[1:])); diagnostics['polyakov_center_sector'].append({'counts':counts,'transitions':transitions})
    for fld in ('q','energy'): diagnostics['flowed_'+fld+'_t004']=diag([[o['flow'][-1][fld] for o in c['observations']] for c in chains])
    result={'subject':{'dims':DIMS,'beta':BETA,'burn_in':BURN_IN,'stride':STRIDE,'measurements':MEASUREMENTS,'flow_dt':FLOW_DT,'flow_steps':FLOW_STEPS,'sampler':'hbor_direct_staple_1or','seed_hex':SEED.hex()},'force_parity_max_abs_error':pe,'chains':chains,'diagnostics':diagnostics}
    text=json.dumps(result,sort_keys=True,separators=(',',':')); result_hash=hashlib.sha256(text.encode()).hexdigest(); expected='d2c2f400678fc5961ceced3c03f71c5a4dc70afb05e63451961a93100ec8cf8b'
    if result_hash!=expected: raise AssertionError(('frozen result hash',result_hash,expected))
    print('ok'); print('result_sha256='+result_hash)
    for k,v in diagnostics.items(): print(k,json.dumps(v,sort_keys=True))
    print('draws',chains[0]['uniform_draws'],chains[1]['uniform_draws']); print('attempts',chains[0]['scalar_attempts'],chains[1]['scalar_attempts'])
if __name__=='__main__': main()
