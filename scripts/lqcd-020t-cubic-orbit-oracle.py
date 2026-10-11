#!/usr/bin/env python3
"""Independent cubic-orbit enumeration oracle for static-potential displacements.

Qualifies the unique signed-permutation orbit used to average already-qualified
off-axis Wilson-loop measurements. No gauge algebra is implemented here.
"""
import hashlib
import itertools
import json
import math

ORACLE_ID = "cubic_signed_permutation_orbit_v1"

def canonical_orbit(v):
    if len(v) != 3 or all(x == 0 for x in v):
        raise ValueError("nonzero 3-vector required")
    if any(not isinstance(x, int) for x in v):
        raise ValueError("integer components required")
    out = set()
    for p in set(itertools.permutations(v)):
        for signs in itertools.product((-1, 1), repeat=3):
            out.add(tuple(signs[i] * p[i] for i in range(3)))
    return tuple(sorted(out))

def shortest_path_count(v):
    a,b,c = (abs(x) for x in v)
    n = a+b+c
    return math.factorial(n) // (math.factorial(a)*math.factorial(b)*math.factorial(c))

def main():
    fixtures = {
        "r100": ((1,0,0), 6, 1),
        "r200": ((2,0,0), 6, 1),
        "r300": ((3,0,0), 6, 1),
        "r110": ((1,1,0), 12, 2),
        "r111": ((1,1,1), 8, 6),
        "r210": ((2,1,0), 24, 3),
        "r220": ((2,2,0), 12, 6),
        "r330": ((3,3,0), 12, 20),
        "r222": ((2,2,2), 8, 90),
        "r333": ((3,3,3), 8, 1680),
        "r420": ((4,2,0), 24, 15),
        "r630": ((6,3,0), 24, 84),
    }
    records = {}
    for key,(v,want_orbit,want_paths) in fixtures.items():
        orbit = canonical_orbit(v)
        counts = {shortest_path_count(x) for x in orbit}
        if len(orbit) != want_orbit:
            raise AssertionError((key,len(orbit),want_orbit))
        if counts != {want_paths}:
            raise AssertionError((key,counts,want_paths))
        if orbit != tuple(sorted(orbit)):
            raise AssertionError("noncanonical ordering")
        if v not in orbit:
            raise AssertionError(("representative missing",v))
        records[key] = {
            "representative": v,
            "orbit_size": len(orbit),
            "shortest_paths_per_orientation": want_paths,
            "orbit": orbit,
        }

    # Sign/permutation-equivalent representatives must canonicalize identically.
    aliases = [
        ((2,1,0),(-1,0,2)),
        ((1,1,1),(-1,1,-1)),
        ((3,3,0),(0,-3,3)),
        ((6,3,0),(-3,6,0)),
    ]
    for a,b in aliases:
        if canonical_orbit(a) != canonical_orbit(b):
            raise AssertionError(("alias mismatch",a,b))

    # Zero displacement is not a static separation.
    try:
        canonical_orbit((0,0,0))
        raise AssertionError("zero displacement accepted")
    except ValueError:
        pass

    result = {"oracle_id": ORACLE_ID, "fixtures": records}
    text = json.dumps(result, sort_keys=True, separators=(",",":"))
    digest = hashlib.sha256(text.encode()).hexdigest()
    print("ok")
    print("result_sha256="+digest)
    print(text)

if __name__ == "__main__":
    main()
