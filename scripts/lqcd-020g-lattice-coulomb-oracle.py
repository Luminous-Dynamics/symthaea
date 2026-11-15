#!/usr/bin/env python3
"""Independent tree-level 3D lattice Coulomb kernel oracle."""
import hashlib,json,math
ID='tree_level_wilson_lattice_coulomb_midpoint_richardson_v1'
R_VECTORS=((1,0,0),(2,0,0),(3,0,0),(1,1,0),(1,1,1),(2,1,0))
def midpoint(r,n):
 if n<8 or n%2: raise ValueError('even n >= 8 required')
 rx,ry,rz=r; step=2*math.pi/n; total=0.0
 for ix in range(n):
  kx=-math.pi+(ix+.5)*step; sx=math.sin(kx/2)**2
  for iy in range(n):
   ky=-math.pi+(iy+.5)*step; sy=math.sin(ky/2)**2
   for iz in range(n):
    kz=-math.pi+(iz+.5)*step; den=sx+sy+math.sin(kz/2)**2
    total+=math.cos(kx*rx+ky*ry+kz*rz)/den
 return math.pi*total/(n**3)
def richardson(r,n):
 coarse=midpoint(r,n); fine=midpoint(r,2*n); return {'n':n,'coarse':coarse,'fine':fine,'extrapolated':2*fine-coarse}
def main():
 primary={str(r):richardson(r,64) for r in R_VECTORS}; check={str(r):richardson(r,80) for r in R_VECTORS}; disagreement={str(r):abs(primary[str(r)]['extrapolated']-check[str(r)]['extrapolated']) for r in R_VECTORS}; mx=max(disagreement.values())
 if mx>3e-6: raise AssertionError(('resolution disagreement',mx))
 if abs(primary[str((1,0,0))]['extrapolated']-1.0)<.05: raise AssertionError('missing lattice correction')
 out={'oracle_id':ID,'definition':'4pi integral_BZ d3k/(2pi)^3 cos(k.R)/(4 sum_j sin^2(k_j/2))','primary':primary,'independent_resolution_check':check,'max_extrapolated_pair_disagreement':mx}; s=json.dumps(out,sort_keys=True,separators=(',',':')); h=hashlib.sha256(s.encode()).hexdigest(); expected='429c13c0f308bd21659cc4c7bc5212a770b9f99e706e71557e339984994b3fef'
 if h!=expected: raise AssertionError((h,expected))
 print('ok'); print('result_sha256='+h); print(s)
if __name__=='__main__':main()
