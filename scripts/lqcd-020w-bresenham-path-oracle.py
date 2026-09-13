#!/usr/bin/env python3
"""Independent generalized-Bresenham path oracle for off-axis Wilson transport.

This subject qualifies geometry only. It implements the 3D generalization of
the Bresenham construction described for lattice static-potential Wilson paths:
one major-axis loop with independent error accumulators for the two remaining
axes. It does not import Symthaea/Rust or perform gauge-field algebra.
"""
import hashlib
import itertools
import json
import math

ORACLE_ID = "generalized_bresenham_3d_static_path_v1"

def cubic_orbit(v):
    if len(v) != 3 or any(not isinstance(x, int) for x in v) or all(x == 0 for x in v):
        raise ValueError("nonzero integer 3-vector required")
    out = set()
    for p in set(itertools.permutations(v)):
        for signs in itertools.product((-1, 1), repeat=3):
            out.add(tuple(signs[i] * p[i] for i in range(3)))
    return tuple(sorted(out))

def shortest_path_count(v):
    a, b, c = (abs(x) for x in v)
    n = a + b + c
    return math.factorial(n) // (math.factorial(a) * math.factorial(b) * math.factorial(c))

def bresenham_steps(v):
    if len(v) != 3 or any(not isinstance(x, int) for x in v) or all(x == 0 for x in v):
        raise ValueError("nonzero integer 3-vector required")

    magnitudes = [abs(x) for x in v]
    signs = [1 if x >= 0 else -1 for x in v]
    # Largest magnitude is the major axis; ties are broken by axis index.
    major, middle, minor = sorted(range(3), key=lambda axis: (-magnitudes[axis], axis))
    major_count = magnitudes[major]
    middle_count = magnitudes[middle]
    minor_count = magnitudes[minor]

    chi_middle = 2 * middle_count - major_count
    chi_minor = 2 * minor_count - major_count
    steps = []

    for _ in range(major_count):
        step = [0, 0, 0]
        step[major] = signs[major]
        steps.append(tuple(step))

        if middle_count and chi_middle >= 0:
            step = [0, 0, 0]
            step[middle] = signs[middle]
            steps.append(tuple(step))
            chi_middle -= 2 * major_count

        if minor_count and chi_minor >= 0:
            step = [0, 0, 0]
            step[minor] = signs[minor]
            steps.append(tuple(step))
            chi_minor -= 2 * major_count

        chi_middle += 2 * middle_count
        chi_minor += 2 * minor_count

    return tuple(steps)

def endpoint(steps):
    return tuple(sum(step[axis] for step in steps) for axis in range(3))

def reverse_steps(steps):
    return tuple(tuple(-component for component in step) for step in reversed(steps))

def validate_steps(v, steps):
    if endpoint(steps) != v:
        raise AssertionError(("endpoint", v, endpoint(steps)))
    if len(steps) != sum(abs(x) for x in v):
        raise AssertionError(("path length", v, len(steps)))
    for step in steps:
        if sum(abs(x) for x in step) != 1:
            raise AssertionError(("non-axial unit step", step))
        for axis, component in enumerate(step):
            if component and component != (1 if v[axis] > 0 else -1):
                raise AssertionError(("wrong direction", v, step))

    # The return transporter can be built by reversing and daggering exactly
    # this sequence; its geometric endpoint must be -v.
    if endpoint(reverse_steps(steps)) != tuple(-x for x in v):
        raise AssertionError(("reverse endpoint", v))

def main():
    fixtures = [
        (1, 0, 0),
        (2, 1, 0),
        (5, 3, 0),
        (5, 3, 2),
        (6, 3, 0),
        (7, 7, 0),
        (6, 6, 6),
        (-5, 3, -2),
    ]

    records = {}
    for v in fixtures:
        steps = bresenham_steps(v)
        validate_steps(v, steps)
        orbit = cubic_orbit(v)
        exhaustive = shortest_path_count(v)
        bresenham_links_per_orientation = len(steps)
        orbit_link_work = len(orbit) * bresenham_links_per_orientation
        exhaustive_orbit_link_work = len(orbit) * exhaustive * bresenham_links_per_orientation

        # Every cubic-orbit orientation has the same Manhattan length and the
        # same exhaustive shortest-path multiplicity.
        if {len(bresenham_steps(o)) for o in orbit} != {bresenham_links_per_orientation}:
            raise AssertionError(("orbit path length", v))
        if {shortest_path_count(o) for o in orbit} != {exhaustive}:
            raise AssertionError(("orbit exhaustive multiplicity", v))

        records[str(v)] = {
            "steps": steps,
            "step_count": bresenham_links_per_orientation,
            "orbit_size": len(orbit),
            "exhaustive_shortest_paths_per_orientation": exhaustive,
            "bresenham_orbit_link_work": orbit_link_work,
            "exhaustive_orbit_link_work": exhaustive_orbit_link_work,
        }

    # Freeze the characteristic 2D and 3D examples used to qualify local
    # decision ordering, not merely endpoint correctness.
    if bresenham_steps((5, 3, 0)) != (
        (1,0,0),(0,1,0),(1,0,0),(1,0,0),
        (0,1,0),(1,0,0),(1,0,0),(0,1,0),
    ):
        raise AssertionError("unexpected (5,3,0) path")
    if bresenham_steps((5, 3, 2)) != (
        (1,0,0),(0,1,0),(1,0,0),(0,0,1),(1,0,0),
        (0,1,0),(1,0,0),(0,0,1),(1,0,0),(0,1,0),
    ):
        raise AssertionError("unexpected (5,3,2) path")

    # Cost regression: the bounded construction remains tiny where exhaustive
    # shortest-path symmetrization is already prohibitive.
    if records[str((7, 7, 0))]["bresenham_orbit_link_work"] != 168:
        raise AssertionError("unexpected (7,7,0) bounded cost")
    if records[str((7, 7, 0))]["exhaustive_shortest_paths_per_orientation"] != 3432:
        raise AssertionError("unexpected (7,7,0) exhaustive count")
    if records[str((6, 6, 6))]["bresenham_orbit_link_work"] != 144:
        raise AssertionError("unexpected (6,6,6) bounded cost")
    if records[str((6, 6, 6))]["exhaustive_shortest_paths_per_orientation"] != 17153136:
        raise AssertionError("unexpected (6,6,6) exhaustive count")

    result = {"oracle_id": ORACLE_ID, "fixtures": records}
    text = json.dumps(result, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(text.encode()).hexdigest()
    print("ok")
    print("result_sha256=" + digest)
    print(text)

if __name__ == "__main__":
    main()
