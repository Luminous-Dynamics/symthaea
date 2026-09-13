#!/usr/bin/env python3
import importlib.util, json, math, statistics, hashlib
from pathlib import Path
BASE=Path(__file__).with_name('lqcd-019b-tiny-campaign.py')
spec=importlib.util.spec_from_file_location('base019b', BASE)
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
m.ENSEMBLE_SLOT=32
BURN_IN=30
STRIDE=2
MEASUREMENTS=12
CENTERS=[complex(1,0), complex(math.cos(2*math.pi/3),math.sin(2*math.pi/3)), complex(math.cos(-2*math.pi/3),math.sin(-2*math.pi/3))]

def center_cold(sector):
    f=m.identity_field()
    z=CENTERS[sector]
    zmat=[[z if i==j else 0j for j in range(3)] for i in range(3)]
    for x in range(m.DIMS[0]):
      for y in range(m.DIMS[1]):
       for zz in range(m.DIMS[2]):
        s=(x,y,zz,0)
        m.set_link(f,s,3,m.mul(zmat,m.link(f,s,3)))
    if abs(m.avg_plaq(f)-1.0)>2e-14: raise AssertionError(('center action/plaquette',sector,m.avg_plaq(f)))
    p=m.polyakov(f)
    if abs(p-z)>2e-14: raise AssertionError(('center polyakov',sector,p,z))
    return f

def max_dwell(sec):
    best=cur=0; prev=None
    for x in sec:
        if x==prev: cur+=1
        else: cur=1; prev=x
        best=max(best,cur)
    return best

def run_chain(label, replica, field):
    src=m.ChaCha8Stream(m.SEED,m.pack_stream_id(m.DOMAIN_TRANSITION,m.ENSEMBLE_SLOT,replica))
    keep={BURN_IN+STRIDE*i for i in range(1,MEASUREMENTS+1)}; final=BURN_IN+STRIDE*MEASUREMENTS
    obs=[]; attempts=0
    for cyc in range(1,final+1):
        attempts+=m.hbor(field,m.BETA,src)
        if cyc in keep:
            p=m.polyakov(field); pa=m.center_aligned(p)
            obs.append({'cycle':cyc,'plaquette':m.avg_plaq(field),'polyakov_re':p.real,'polyakov_im':p.imag,'polyakov_abs':abs(p),'polyakov_sector':m.center_sector(p),'polyakov_aligned_re':pa.real,'wilson_t1':m.wilson_t1(field)})
    sec=[o['polyakov_sector'] for o in obs]
    return {'label':label,'replica':replica,'draws':src.draws,'scalar_attempts':attempts,'sector_counts':{str(k):sec.count(k) for k in range(3)},'sector_transitions':sum(a!=b for a,b in zip(sec,sec[1:])),'max_sector_dwell':max_dwell(sec),'observations':obs}

def main():
    zero=bytes.fromhex('3e00ef2f895f40d67f5bb8e81f09a5a12c840ec3ce9a7f3b181be188ef711a1e984ce172b9216f419f445367456d5619314a42a3da86b001387bfdb80e0cfe42')
    assert m.chacha8_block(bytes(32),0,0)==zero
    pe=m.parity(); assert pe<3e-12
    starts=[('center0',0,center_cold(0)),('center_plus',1,center_cold(1)),('center_minus',2,center_cold(2)),('disordered',3,m.disordered(3))]
    chains=[run_chain(label,replica,f) for label,replica,f in starts]
    keys=['plaquette','polyakov_re','polyakov_im','polyakov_abs','polyakov_aligned_re','wilson_t1']
    diagnostics={k:m.diag([[o[k] for o in c['observations']] for c in chains]) for k in keys}
    result={'subject':{'dims':m.DIMS,'beta':m.BETA,'burn_in':BURN_IN,'stride':STRIDE,'measurements':MEASUREMENTS,'sampler':'hbor_direct_staple_1or','ensemble_slot':m.ENSEMBLE_SLOT,'starts':[x[0] for x in starts]},'force_parity_max_abs_error':pe,'chains':chains,'diagnostics':diagnostics}
    text=json.dumps(result,sort_keys=True,separators=(',',':'))
    h=hashlib.sha256(text.encode()).hexdigest()
    print('ok'); print('result_sha256='+h); print('force_parity_max_abs_error='+repr(pe))
    for k,v in diagnostics.items(): print(k,json.dumps(v,sort_keys=True))
    for c in chains: print('sector',c['label'],c['sector_counts'],'transitions',c['sector_transitions'],'max_dwell',c['max_sector_dwell'],'draws',c['draws'])
if __name__=='__main__': main()
