#!/usr/bin/env python3
from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Tuple, FrozenSet, Dict

@dataclass(frozen=True)
class RestartNode:
    node_id: str
    recipes: Tuple[Tuple[str, ...], ...] = ()
    essential_target: bool = False
    productive_target: bool = False

@dataclass(frozen=True)
class SeedAnalysis:
    essential_minimal_seeds: Tuple[Tuple[str, ...], ...]
    productive_minimal_seeds: Tuple[Tuple[str, ...], ...]
    essential_minimal_cuts: Tuple[Tuple[str, ...], ...]
    productive_minimal_cuts: Tuple[Tuple[str, ...], ...]
    restart_wave: Tuple[Tuple[str, int], ...]


def validate(nodes: Iterable[RestartNode], seed_candidates: Iterable[str]) -> Tuple[Tuple[RestartNode, ...], Tuple[str, ...]]:
    nodes = tuple(nodes)
    seeds = tuple(seed_candidates)
    ids = set()
    for n in nodes:
        if not n.node_id or n.node_id in ids:
            raise ValueError("node ids must be unique and nonempty")
        ids.add(n.node_id)
        if len(set(n.recipes)) != len(n.recipes):
            raise ValueError("duplicate restart recipe")
        for recipe in n.recipes:
            if len(set(recipe)) != len(recipe):
                raise ValueError("duplicate prerequisite inside recipe")
    for n in nodes:
        for recipe in n.recipes:
            for dep in recipe:
                if dep not in ids:
                    raise ValueError("unknown restart dependency")
    if len(set(seeds)) != len(seeds) or any(s not in ids for s in seeds):
        raise ValueError("seed candidates must be unique known nodes")
    return nodes, seeds


def closure(nodes: Iterable[RestartNode], seeded: Iterable[str]) -> FrozenSet[str]:
    nodes = tuple(nodes)
    by_id = {n.node_id: n for n in nodes}
    active = set(seeded)
    if active - set(by_id):
        raise ValueError("unknown seeded node")
    changed = True
    while changed:
        changed = False
        for n in nodes:
            if n.node_id in active or not n.recipes:
                continue
            if any(set(recipe).issubset(active) for recipe in n.recipes):
                active.add(n.node_id)
                changed = True
    return frozenset(active)


def target_ok(nodes: Iterable[RestartNode], active: FrozenSet[str], productive: bool) -> bool:
    targets = {
        n.node_id for n in nodes
        if (n.productive_target if productive else n.essential_target)
    }
    if not targets:
        raise ValueError("target class must contain at least one node")
    return targets.issubset(active)


def inclusion_minimal_seed_sets(nodes, seed_candidates, productive: bool) -> Tuple[Tuple[str, ...], ...]:
    nodes, seeds = validate(nodes, seed_candidates)
    feasible = []
    for r in range(len(seeds) + 1):
        for combo in combinations(seeds, r):
            if target_ok(nodes, closure(nodes, combo), productive):
                s = frozenset(combo)
                if not any(prev.issubset(s) for prev in feasible):
                    feasible.append(s)
    return tuple(sorted((tuple(sorted(s)) for s in feasible), key=lambda x: (len(x), x)))


def inclusion_minimal_cut_sets(nodes, baseline_seeds, productive: bool) -> Tuple[Tuple[str, ...], ...]:
    nodes, seeds = validate(nodes, baseline_seeds)
    baseline = frozenset(seeds)
    if not target_ok(nodes, closure(nodes, baseline), productive):
        raise ValueError("baseline seed set does not satisfy target")
    cuts = []
    for r in range(1, len(seeds) + 1):
        for removed in combinations(seeds, r):
            remaining = baseline - set(removed)
            if not target_ok(nodes, closure(nodes, remaining), productive):
                c = frozenset(removed)
                if not any(prev.issubset(c) for prev in cuts):
                    cuts.append(c)
    return tuple(sorted((tuple(sorted(c)) for c in cuts), key=lambda x: (len(x), x)))


def restart_waves(nodes: Iterable[RestartNode], seeded: Iterable[str]) -> Tuple[Tuple[str, int], ...]:
    nodes = tuple(nodes)
    active = set(seeded)
    ids = {n.node_id for n in nodes}
    if active - ids:
        raise ValueError("unknown seeded node")
    wave: Dict[str, int] = {s: 0 for s in active}
    depth = 0
    while True:
        ready = []
        for n in nodes:
            if n.node_id in active or not n.recipes:
                continue
            if any(set(recipe).issubset(active) for recipe in n.recipes):
                ready.append(n.node_id)
        if not ready:
            break
        depth += 1
        for nid in ready:
            active.add(nid)
            wave[nid] = depth
    return tuple(sorted(wave.items()))


def analyze(nodes, seed_candidates, baseline_seeds) -> SeedAnalysis:
    nodes, _ = validate(nodes, seed_candidates)
    validate(nodes, baseline_seeds)
    return SeedAnalysis(
        inclusion_minimal_seed_sets(nodes, seed_candidates, False),
        inclusion_minimal_seed_sets(nodes, seed_candidates, True),
        inclusion_minimal_cut_sets(nodes, baseline_seeds, False),
        inclusion_minimal_cut_sets(nodes, baseline_seeds, True),
        restart_waves(nodes, baseline_seeds),
    )


def self_test() -> None:
    nodes = [
        RestartNode("battery"),
        RestartNode("rtg"),
        RestartNode("ind_seed"),
        RestartNode("control", (("battery",), ("rtg",))),
        RestartNode("hab_gen", (("control",),)),
        RestartNode("life", (("hab_gen",),), essential_target=True),
        RestartNode("water", (("hab_gen",),), essential_target=True),
        RestartNode("ind_gen", (("ind_seed",),)),
        RestartNode("shop", (("ind_gen",),), productive_target=True),
        RestartNode("metrology", (("shop",),), productive_target=True),
        RestartNode("cycle_a", (("cycle_b",),)),
        RestartNode("cycle_b", (("cycle_a",),)),
    ]
    candidate_seeds = ("battery", "rtg", "ind_seed")
    baseline = candidate_seeds
    a = analyze(nodes, candidate_seeds, baseline)
    assert a.essential_minimal_seeds == (("battery",), ("rtg",))
    assert a.productive_minimal_seeds == (("ind_seed",),)
    assert a.essential_minimal_cuts == (("battery", "rtg"),)
    assert a.productive_minimal_cuts == (("ind_seed",),)
    assert "cycle_a" not in dict(a.restart_wave)
    assert dict(a.restart_wave)["control"] == 1
    assert dict(a.restart_wave)["hab_gen"] == 2
    assert dict(a.restart_wave)["life"] == 3

    c1 = closure(nodes, ("battery",))
    c2 = closure(nodes, ("battery", "ind_seed"))
    assert c1.issubset(c2)
    assert ("battery", "rtg") not in a.essential_minimal_seeds

    try:
        inclusion_minimal_cut_sets(nodes, ("battery",), True)
        raise AssertionError("invalid productive baseline should fail")
    except ValueError as e:
        assert "does not satisfy target" in str(e)

    try:
        analyze([RestartNode("a", (("missing",),), essential_target=True, productive_target=True)], ("a",), ("a",))
        raise AssertionError("unknown dependency should fail")
    except ValueError as e:
        assert "unknown restart dependency" in str(e)

    print("ok")

if __name__ == "__main__":
    self_test()
