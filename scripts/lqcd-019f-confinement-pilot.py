#!/usr/bin/env python3
import math, statistics, struct, json, hashlib, csv, argparse
from statistics import NormalDist
MASK32=0xffffffff
PAIRS=((0,1),(0,2),(1,2))
DIMS=(3,2,2,3)
BETA=5.7
SEED=bytes([0x6B])*32
ENSEMBLE_SLOT=37
DOMAIN_TRANSITION=1
DOMAIN_INITIALIZATION=4
BURN_IN=12
STRIDE=1
MEASUREMENTS=8

def rotl32(x,n): return ((x<<n)&MASK32)|(x>>(32-n))
def qr(x,a,b,c,d):
    x[a]=(x[a]+x[b])&MASK32; x[d]^=x[a]; x[d]=rotl32(x[d],16)
    x[c]=(x[c]+x[d])&MASK32; x[b]^=x[c]; x[b]=rotl32(x[b],12)
    x[a]=(x[a]+x[b])&MASK32; x[d]^=x[a]; x[d]=rotl32(x[d],8)
    x[c]=(x[c]+x[d])&MASK32; x[b]^=x[c]; x[b]=rotl32(x[b],7)
def chacha8(seed,counter,stream):
    st=[0x61707865,0x3320646e,0x79622d32,0x6b206574]+list(struct.unpack('<8I',seed))+[counter&MASK32,(counter>>32)&MASK32,stream&MASK32,(stream>>32)&MASK32]
    w=st.copy()
    for _ in range(4):
        qr(w,0,4,8,12); qr(w,1,5,9,13); qr(w,2,6,10,14); qr(w,3,7,11,15)
        qr(w,0,5,10,15); qr(w,1,6,11,12); qr(w,2,7,8,13); qr(w,3,4,9,14)
    return struct.pack('<16I',*[(w[i]+st[i])&MASK32 for i in range(16)])
def stream_id(domain,slot,replica,rank=0): return (domain<<56)|(slot<<32)|(replica<<16)|rank
class RNG:
    def __init__(self,seed,stream): self.seed=seed; self.stream=stream; self.counter=0; self.buf=b''; self.cur=0; self.draws=0
    def _bytes(self,n):
        out=bytearray()
        while len(out)<n:
            if self.cur>=len(self.buf): self.buf=chacha8(self.seed,self.counter,self.stream); self.counter+=1; self.cur=0
            take=min(n-len(out),len(self.buf)-self.cur); out.extend(self.buf[self.cur:self.cur+take]); self.cur+=take
        return bytes(out)
    def u(self):
        self.draws+=1; v=struct.unpack('<Q',self._bytes(8))[0]; return ((v>>12)+1)/((1<<52)+1)

def eye(): return [[1+0j if i==j else 0j for j in range(3)] for i in range(3)]
def zero(): return [[0j]*3 for _ in range(3)]
def add(a,b): return [[a[i][j]+b[i][j] for j in range(3)] for i in range(3)]
def mul(a,b): return [[sum(a[i][k]*b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
def dag(a): return [[a[j][i].conjugate() for j in range(3)] for i in range(3)]
def tr(a): return a[0][0]+a[1][1]+a[2][2]
def sites(d=DIMS):
    for x in range(d[0]):
      for y in range(d[1]):
       for z in range(d[2]):
        for t in range(d[3]): yield (x,y,z,t)
def idx(s,d=DIMS): x,y,z,t=s; return (((x*d[1]+y)*d[2]+z)*d[3]+t)
def shift(s,mu,n,d=DIMS): o=list(s); o[mu]=(o[mu]+n)%d[mu]; return tuple(o)
def ident_field(d=DIMS): return [eye() for _ in range(math.prod(d)*4)]
def get(f,s,mu,d=DIMS): return f[idx(s,d)*4+mu]
def put(f,s,mu,v,d=DIMS): f[idx(s,d)*4+mu]=v
def plaq(f,s,mu,nu,d=DIMS):
    sm=shift(s,mu,1,d); sn=shift(s,nu,1,d)
    return mul(mul(mul(get(f,s,mu,d),get(f,sm,nu,d)),dag(get(f,sn,mu,d))),dag(get(f,s,nu,d)))
def avg_plaq(f,d=DIMS):
    vals=[tr(plaq(f,s,mu,nu,d)).real/3 for s in sites(d) for mu in range(4) for nu in range(mu+1,4)]
    return sum(vals)/len(vals)
def emb(pair,q):
    a0,a1,a2,a3=q; o=eye(); i,j=pair
    o[i][i]=a0+1j*a3; o[i][j]=a2+1j*a1; o[j][i]=-a2+1j*a1; o[j][j]=a0-1j*a3
    return o
def q_from(pair,m):
    i,j=pair
    return ((m[i][i].real+m[j][j].real)/2,(m[i][j].imag+m[j][i].imag)/2,(m[i][j].real-m[j][i].real)/2,(m[i][i].imag-m[j][j].imag)/2)
def draw_rot(rng,pair,max_angle):
    uz,up,ua=rng.u(),rng.u(),rng.u(); z=2*uz-1; ph=2*math.pi*up; rr=math.sqrt(max(0,1-z*z)); a=max_angle*(2*ua-1); c,s=math.cos(a),math.sin(a)
    return emb(pair,(c,s*rr*math.cos(ph),s*rr*math.sin(ph),s*z))
def disordered(replica,rounds=2,d=DIMS):
    f=ident_field(d); rng=RNG(SEED,stream_id(DOMAIN_INITIALIZATION,ENSEMBLE_SLOT,replica))
    for _ in range(rounds):
      for s in sites(d):
       for mu in range(4):
        for p in PAIRS: put(f,s,mu,mul(draw_rot(rng,p,math.pi),get(f,s,mu,d)),d)
    return f
def affected(s,mu,d=DIMS):
    out=set()
    for nu in range(4):
        if nu==mu: continue
        a,b=sorted((mu,nu)); out.add((s,a,b)); out.add((shift(s,nu,-1,d),a,b))
    return sorted(out)
def touch(f,s,mu,d=DIMS): return sum(tr(plaq(f,b,a,c,d)).real for b,a,c in affected(s,mu,d))
def copy_field(f): return [[row[:] for row in m] for m in f]
def evaluated(f,s,mu,p,q,d=DIMS):
    c=copy_field(f); put(c,s,mu,mul(emb(p,q),get(c,s,mu,d)),d); return touch(c,s,mu,d)
def probe_force(f,s,mu,p,d=DIMS):
    plus=evaluated(f,s,mu,p,(1,0,0,0),d); minus=evaluated(f,s,mu,p,(-1,0,0,0),d); c=.5*(plus+minus)
    return c,(.5*(plus-minus),evaluated(f,s,mu,p,(0,1,0,0),d)-c,evaluated(f,s,mu,p,(0,0,1,0),d)-c,evaluated(f,s,mu,p,(0,0,0,1),d)-c)
def staple(f,s,mu,d=DIMS):
    st=zero(); spm=shift(s,mu,1,d)
    for nu in range(4):
        if nu==mu: continue
        spn=shift(s,nu,1,d)
        st=add(st,mul(mul(get(f,spm,nu,d),dag(get(f,spn,mu,d))),dag(get(f,s,nu,d))))
        smn=shift(s,nu,-1,d); smnpm=shift(smn,mu,1,d)
        st=add(st,mul(mul(dag(get(f,smnpm,nu,d)),dag(get(f,smn,mu,d))),get(f,smn,nu,d)))
    return st
def spectator(p): return {(0,1):2,(0,2):1,(1,2):0}[p]
def staple_force(f,s,mu,p,d=DIMS):
    x=mul(get(f,s,mu,d),staple(f,s,mu,d)); i,j=p; k=spectator(p)
    return x[k][k].real,(x[i][i].real+x[j][j].real,-x[j][i].imag-x[i][j].imag,x[j][i].real-x[i][j].real,-x[i][i].imag+x[j][j].imag)
def kp(rng,alpha,max_attempts=256):
    if alpha==0:
        for n in range(1,max_attempts+1):
            a0=2*rng.u()-1
            if rng.u()<math.sqrt(max(0,1-a0*a0)): return a0,n
        raise RuntimeError('haar rejection limit')
    for n in range(1,max_attempts+1):
        r0,r1,r2,r3=(rng.u() for _ in range(4)); x1=-math.log(r1)/alpha; x2=-math.log(r2)/alpha; c=math.cos(2*math.pi*r3); dd=x2+x1*c*c; th=1-.5*dd
        if th>0 and r0*r0<th:
            a0=1-dd
            if -1<=a0<=1:return a0,n
    raise RuntimeError('KP rejection limit')
def su2_hb(rng,alpha):
    a0,n=kp(rng,alpha); z=2*rng.u()-1; ph=2*math.pi*rng.u(); rr=math.sqrt(max(0,1-z*z)); rad=math.sqrt(max(0,1-a0*a0))
    return (a0,rad*rr*math.cos(ph),rad*rr*math.sin(ph),rad*z),n
def hb_step(f,s,mu,p,beta,rng,d=DIMS):
    _,q=staple_force(f,s,mu,p,d); rho=math.sqrt(sum(v*v for v in q)); can,n=su2_hb(rng,beta*rho/3)
    if rho>1e-14:
        qh=tuple(v/rho for v in q); rot=mul(emb(p,qh),emb(p,can)); ori=q_from(p,rot); proj=sum(q[i]*ori[i] for i in range(4))/rho
        if abs(proj-can[0])>1e-9: raise AssertionError('orientation')
    else: rot=emb(p,can)
    put(f,s,mu,mul(rot,get(f,s,mu,d)),d); return n
def hb_sweep(f,beta,rng,d=DIMS):
    n=0
    for s in sites(d):
      for mu in range(4):
       for p in PAIRS:n+=hb_step(f,s,mu,p,beta,rng,d)
    return n
def or_step(f,s,mu,p,d=DIMS):
    before=touch(f,s,mu,d); _,q=staple_force(f,s,mu,p,d); norm2=sum(v*v for v in q)
    if norm2<=1e-28:return
    r=[2*q[0]*v/norm2 for v in q]; r[0]-=1; put(f,s,mu,mul(emb(p,r),get(f,s,mu,d)),d)
    after=touch(f,s,mu,d)
    if abs(after-before)>3e-9: raise AssertionError(('micro',after-before))
def or_sweep(f,d=DIMS):
    for s in sites(d):
      for mu in range(4):
       for p in PAIRS:or_step(f,s,mu,p,d)
def hbor(f,beta,rng,d=DIMS):
    n=hb_sweep(f,beta,rng,d); or_sweep(f,d); return n
def polyakov(f,d=DIMS):
    vals=[]
    for x in range(d[0]):
      for y in range(d[1]):
       for z in range(d[2]):
        p=eye()
        for t in range(d[3]): p=mul(p,get(f,(x,y,z,t),3,d))
        vals.append(tr(p)/3)
    return sum(vals)/len(vals)
def oriented_link(f,s,sd,d=DIMS):
    if sd>0: mu=sd-1; return get(f,s,mu,d),shift(s,mu,1,d)
    mu=-sd-1; prev=shift(s,mu,-1,d); return dag(get(f,prev,mu,d)),prev
def transporter(f,s,dirs,d=DIMS):
    v=eye()
    for sd in dirs:
        e,s=oriented_link(f,s,sd,d); v=mul(v,e)
    return v,s
def rectangle(f,s,spatial_mu,R,T,d=DIMS):
    temporal_mu=3; dirs=[spatial_mu+1]*R+[temporal_mu+1]*T+[-(spatial_mu+1)]*R+[-(temporal_mu+1)]*T
    u,end=transporter(f,s,dirs,d)
    if end!=s: raise AssertionError('rectangle did not close')
    return tr(u).real/3
def mean_rectangle(f,R,T,d=DIMS):
    if R>=d[0] or T>=d[3]: raise ValueError('winding rectangle requested')
    vals=[rectangle(f,s,0,R,T,d) for s in sites(d)]
    return sum(vals)/len(vals)
def ac(xs,lag):
    m=statistics.mean(xs); den=sum((x-m)**2 for x in xs)
    if den==0:return None
    return sum((xs[i]-m)*(xs[i+lag]-m) for i in range(len(xs)-lag))/den
def tau(xs):
    maxlag=min(len(xs)//2,len(xs)-1); rr=[ac(xs,k) for k in range(1,maxlag+1)]
    if not rr or rr[0] is None:return None
    tot=0; k=0
    while k<len(rr):
        pair=rr[k]+(rr[k+1] if k+1<len(rr) else 0)
        if pair<=0:break
        tot+=pair; k+=2
    return .5+tot
def ess(xs):
    t=tau(xs); return None if t is None else len(xs)/(2*t)
def split_rhat(chains):
    n0=len(chains[0]); n=n0//2; sp=[]
    if n<2 or any(len(c)!=n0 for c in chains): return None
    for c in chains:sp.extend((c[:n],c[n:]))
    means=[statistics.mean(c) for c in sp]; vs=[statistics.variance(c) for c in sp]; W=statistics.mean(vs)
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
    flat=[x for c in chains for x in c]; rs=ranks(flat); N=len(flat); nd=NormalDist(); zz=[nd.inv_cdf((r-.375)/(N+.25)) for r in rs]; out=[]; p=0
    for c in chains:out.append(zz[p:p+len(c)]); p+=len(c)
    return out
def rankfold(chains):
    bulk=split_rhat(ranknorm(chains)); med=statistics.median([x for c in chains for x in c]); fold=split_rhat(ranknorm([[abs(x-med) for x in c] for c in chains])); vals=[x for x in (bulk,fold) if x is not None]
    return bulk,fold,max(vals) if vals else None
def diag(series):
    b,f,m=rankfold(series); return {'means':[statistics.mean(s) for s in series],'rank_rhat':b,'folded_rhat':f,'max_rank_folded_rhat':m,'ess':[ess(s) for s in series],'tau_int':[tau(s) for s in series]}
def run_chain(start,replica):
    f=ident_field() if start=='cold' else disordered(replica); rng=RNG(SEED,stream_id(DOMAIN_TRANSITION,ENSEMBLE_SLOT,replica)); keep={BURN_IN+STRIDE*i for i in range(1,MEASUREMENTS+1)}; final=BURN_IN+STRIDE*MEASUREMENTS; rows=[]; attempts=0
    for cyc in range(1,final+1):
        attempts += hbor(f,BETA,rng,DIMS)
        if cyc in keep:
            p=polyakov(f,DIMS); rows.append({'cycle':cyc,'plaquette':avg_plaq(f,DIMS),'polyakov_abs':abs(p),'w11':mean_rectangle(f,1,1,DIMS),'w21':mean_rectangle(f,2,1,DIMS),'w12':mean_rectangle(f,1,2,DIMS),'w22':mean_rectangle(f,2,2,DIMS)})
    return {'start':start,'replica':replica,'draws':rng.draws,'scalar_attempts':attempts,'rows':rows}
def force_parity():
    f=disordered(9,1,DIMS); e=0.0
    for p in PAIRS:
        c1,q1=probe_force(f,(0,0,0,0),0,p,DIMS); c2,q2=staple_force(f,(0,0,0,0),0,p,DIMS); e=max(e,abs(c1-c2),*(abs(a-b) for a,b in zip(q1,q2)))
    return e
def creutz_from_means(rows):
    m={k:statistics.mean([r[k] for r in rows]) for k in ('w11','w21','w12','w22')}; ratio=m['w22']*m['w11']/(m['w21']*m['w12'])
    return m,(-math.log(ratio) if ratio>0 else None),ratio
def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--history-out'); args=parser.parse_args()
    zero=bytes.fromhex('3e00ef2f895f40d67f5bb8e81f09a5a12c840ec3ce9a7f3b181be188ef711a1e984ce172b9216f419f445367456d5619314a42a3da86b001387bfdb80e0cfe42'); assert chacha8(bytes(32),0,0)==zero
    pe=force_parity(); assert pe<4e-12
    chains=[run_chain('cold',0),run_chain('disordered',1)]
    diagnostics={k:diag([[r[k] for r in c['rows']] for c in chains]) for k in ('plaquette','polyakov_abs','w11','w21','w12','w22')}
    pooled=[r for c in chains for r in c['rows']]; pooled_means,chi,ratio=creutz_from_means(pooled); per_chain=[]
    for c in chains:
        mm,cc,rr=creutz_from_means(c['rows']); per_chain.append({'means':mm,'creutz_22':cc,'ratio':rr})
    result={'subject':{'dims':DIMS,'beta':BETA,'burn_in':BURN_IN,'stride':STRIDE,'measurements_per_chain':MEASUREMENTS,'sampler':'hbor_direct_staple_1or','seed_hex':SEED.hex()},'force_parity_max_abs_error':pe,'chains':chains,'diagnostics':diagnostics,'pooled_loop_means':pooled_means,'pooled_creutz_22':chi,'pooled_creutz_ratio':ratio,'per_chain_creutz':per_chain}
    text=json.dumps(result,sort_keys=True,separators=(',',':')); rh=hashlib.sha256(text.encode()).hexdigest(); expected='89e1cb4f598b93518656f21969cac98e5ad715b96de629ee9692d72c24e94099'
    if rh!=expected: raise AssertionError(('frozen result hash',rh,expected))
    print('ok'); print('result_sha256='+rh); print('force_parity='+repr(pe)); print('pooled_loop_means='+json.dumps(pooled_means,sort_keys=True)); print('pooled_creutz_22='+repr(chi)); print('pooled_creutz_ratio='+repr(ratio)); print('per_chain_creutz='+json.dumps(per_chain,sort_keys=True))
    for k,v in diagnostics.items(): print(k,json.dumps(v,sort_keys=True))
    print('draws',chains[0]['draws'],chains[1]['draws']); print('attempts',chains[0]['scalar_attempts'],chains[1]['scalar_attempts'])
    if args.history_out:
        with open(args.history_out,'w',newline='') as fh:
            w=csv.writer(fh); w.writerow(['chain','start','cycle','plaquette','polyakov_abs','w11','w21','w12','w22'])
            for ci,c in enumerate(chains):
                for r in c['rows']: w.writerow([ci,c['start'],r['cycle'],r['plaquette'],r['polyakov_abs'],r['w11'],r['w21'],r['w12'],r['w22']])
        print('history_sha256='+hashlib.sha256(open(args.history_out,'rb').read()).hexdigest())
if __name__=='__main__': main()
