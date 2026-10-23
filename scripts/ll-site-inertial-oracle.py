#!/usr/bin/env python3
"""Independent LL-004E lunar site -> inertial release-state oracle.

Standard library only. No Symthaea imports. Research/trade-study evidence only;
not launcher guidance or release authority.
"""
from __future__ import annotations
import argparse, json, math
from dataclasses import asdict, dataclass

Vec3 = tuple[float, float, float]
Mat3 = tuple[Vec3, Vec3, Vec3]
SECONDS_PER_DAY = 86400.0
ORTHO_TOL = 1e-9

def dot(a: Vec3, b: Vec3) -> float:
    return sum(x*y for x,y in zip(a,b))

def add(a: Vec3, b: Vec3) -> Vec3:
    return tuple(x+y for x,y in zip(a,b))  # type: ignore

def scale(v: Vec3, s: float) -> Vec3:
    return tuple(x*s for x in v)  # type: ignore

def cross(a: Vec3, b: Vec3) -> Vec3:
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])

def mat_vec(m: Mat3, v: Vec3) -> Vec3:
    return tuple(dot(row,v) for row in m)  # type: ignore

def det(m: Mat3) -> float:
    return (m[0][0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1])
            -m[0][1]*(m[1][0]*m[2][2]-m[1][2]*m[2][0])
            +m[0][2]*(m[1][0]*m[2][1]-m[1][1]*m[2][0]))

def is_rotation(m: Mat3) -> bool:
    vals=[x for row in m for x in row]
    if not all(math.isfinite(x) for x in vals): return False
    for i in range(3):
        if abs(dot(m[i],m[i])-1.0)>ORTHO_TOL: return False
        for j in range(i+1,3):
            if abs(dot(m[i],m[j]))>ORTHO_TOL: return False
    return abs(det(m)-1.0)<=10*ORTHO_TOL

def local_basis(lat_deg: float, lon_deg: float) -> tuple[Vec3,Vec3,Vec3]:
    if not (math.isfinite(lat_deg) and -90 <= lat_deg <= 90 and math.isfinite(lon_deg)):
        raise ValueError("invalid site coordinates")
    lat,lon=math.radians(lat_deg),math.radians(lon_deg)
    slat,clat=math.sin(lat),math.cos(lat)
    slon,clon=math.sin(lon),math.cos(lon)
    up=(clat*clon,clat*slon,slat)
    east=(-slon,clon,0.0)
    north=(-slat*clon,-slat*slon,clat)
    return north,east,up

@dataclass(frozen=True)
class Site:
    release_id: str
    site_id: str
    body_fixed_frame: str
    latitude_deg: float
    longitude_deg: float
    site_radius_km: float
    launch_azimuth_deg: float
    launch_elevation_deg: float
    release_speed_km_s: float
    epoch_jd: float

@dataclass(frozen=True)
class Orientation:
    epoch_jd: float
    body_fixed_frame: str
    inertial_frame: str
    rotation_body_fixed_to_inertial: Mat3
    angular_velocity_inertial_rad_s: Vec3
    max_epoch_delta_s: float
    source_ref: str

@dataclass(frozen=True)
class Result:
    position_km: Vec3
    velocity_km_s: Vec3
    epoch_jd: float
    inertial_frame: str
    body_fixed_frame: str
    source_ref: str

def transform(site: Site, ori: Orientation) -> Result:
    vals=[site.latitude_deg,site.longitude_deg,site.site_radius_km,site.launch_azimuth_deg,
          site.launch_elevation_deg,site.release_speed_km_s,site.epoch_jd,ori.epoch_jd,
          ori.max_epoch_delta_s,*ori.angular_velocity_inertial_rad_s]
    if not all(math.isfinite(x) for x in vals):
        raise ValueError("nonfinite input")
    if not site.release_id or not site.site_id or not site.body_fixed_frame or not ori.body_fixed_frame:
        raise ValueError("missing identifier")
    if site.body_fixed_frame != ori.body_fixed_frame:
        raise ValueError("frame mismatch")
    if site.site_radius_km <= 0 or site.release_speed_km_s < 0 or ori.max_epoch_delta_s < 0:
        raise ValueError("invalid magnitude")
    if not (-90 <= site.latitude_deg <= 90 and -90 <= site.launch_elevation_deg <= 90):
        raise ValueError("invalid angle")
    if abs(site.epoch_jd-ori.epoch_jd)*SECONDS_PER_DAY > ori.max_epoch_delta_s:
        raise ValueError("stale orientation")
    if not is_rotation(ori.rotation_body_fixed_to_inertial):
        raise ValueError("invalid rotation")

    north,east,up=local_basis(site.latitude_deg,site.longitude_deg)
    r_bf=scale(up,site.site_radius_km)
    az,el=math.radians(site.launch_azimuth_deg),math.radians(site.launch_elevation_deg)
    h=math.cos(el)
    direction=add(add(scale(north,h*math.cos(az)),scale(east,h*math.sin(az))),scale(up,math.sin(el)))
    v_rel_bf=scale(direction,site.release_speed_km_s)
    r_i=mat_vec(ori.rotation_body_fixed_to_inertial,r_bf)
    v_rel_i=mat_vec(ori.rotation_body_fixed_to_inertial,v_rel_bf)
    v_i=add(v_rel_i,cross(ori.angular_velocity_inertial_rad_s,r_i))
    return Result(r_i,v_i,site.epoch_jd,ori.inertial_frame,ori.body_fixed_frame,ori.source_ref)

def close(a: float,b: float,tol=1e-12): return abs(a-b)<=tol
def vclose(a: Vec3,b: Vec3,tol=1e-12): return all(close(x,y,tol) for x,y in zip(a,b))

def fixture_site(speed=0.1,az=0.0,el=0.0,lat=0.0,lon=0.0) -> Site:
    return Site("release-1","site-1","SYNTH_MOON_FIXED",lat,lon,1000.0,az,el,speed,2460000.5)

def fixture_ori(rotation=((1.,0.,0.),(0.,1.,0.),(0.,0.,1.)),omega=(0.,0.,0.)) -> Orientation:
    return Orientation(2460000.5,"SYNTH_MOON_FIXED","SYNTH_INERTIAL",rotation,omega,0.1,"fixture")

def self_test() -> None:
    r=transform(fixture_site(),fixture_ori())
    assert vclose(r.position_km,(1000.,0.,0.))
    assert vclose(r.velocity_km_s,(0.,0.,0.1))

    rz=((0.,-1.,0.),(1.,0.,0.),(0.,0.,1.))
    r=transform(fixture_site(speed=0.0),fixture_ori(rz))
    assert vclose(r.position_km,(0.,1000.,0.))

    r=transform(fixture_site(speed=0.0),fixture_ori(omega=(0.,0.,1e-3)))
    assert vclose(r.velocity_km_s,(0.,1.,0.))

    n,e,u=local_basis(-89.999999,37.0)
    for axis in (n,e,u):
        assert all(math.isfinite(x) for x in axis)
        assert close(dot(axis,axis),1.0,1e-12)
    assert abs(dot(n,e))<1e-12 and abs(dot(n,u))<1e-12 and abs(dot(e,u))<1e-12

    rinv=((0.,1.,0.),(-1.,0.,0.),(0.,0.,1.))
    p=mat_vec(rz,(321.,-17.,4.))
    assert vclose(mat_vec(rinv,p),(321.,-17.,4.))

    try:
        transform(fixture_site(),fixture_ori(((2.,0.,0.),(0.,1.,0.),(0.,0.,1.))))
        raise AssertionError("bad rotation accepted")
    except ValueError:
        pass
    print("LL-004E oracle self-test: PASS")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--self-test",action="store_true")
    ap.add_argument("--input")
    args=ap.parse_args()
    if args.self_test:
        self_test(); return
    if not args.input:
        ap.error("--input or --self-test required")
    data=json.load(open(args.input))
    site=Site(**data["site"])
    od=dict(data["orientation"])
    od["rotation_body_fixed_to_inertial"]=tuple(tuple(row) for row in od["rotation_body_fixed_to_inertial"])
    od["angular_velocity_inertial_rad_s"]=tuple(od["angular_velocity_inertial_rad_s"])
    result=transform(site,Orientation(**od))
    print(json.dumps(asdict(result),indent=2,sort_keys=True))

if __name__=="__main__":
    main()
