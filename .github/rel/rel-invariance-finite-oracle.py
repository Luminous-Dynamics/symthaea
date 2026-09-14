#!/usr/bin/env python3
"""Independent finite-dimensional oracle for REL symmetry classifications.

This script does not import Symthaea code. It exhaustively checks small exact
models corresponding to three analytic claims in the REL automorphism census:

1. GL(3, 2) has 168 invertible linear maps, and exactly 6 of them preserve
   Hamming distance; those 6 are exactly the coordinate permutations S_3.
2. All 48 signed coordinate permutations in 3D preserve Euclidean inner
   products, but only the 6 unsigned coordinate permutations preserve the
   unital coordinatewise/Hadamard algebra on an exhaustive {-1,0,1}^3 corpus.
3. The centralizer of a full 5-cycle inside S_5 has exactly 5 elements, and
   they are exactly the powers of that cycle.

The finite census supports the classification logic; it is not evidence that
production-dimension Symthaea implementations satisfy the corresponding laws.
"""

from __future__ import annotations

import itertools
import json
from typing import Iterable, Sequence


def apply_binary_matrix(rows: Sequence[int], vector: int, dimension: int) -> int:
    output = 0
    for row_index, row_mask in enumerate(rows):
        parity = (row_mask & vector).bit_count() & 1
        output |= parity << row_index
    assert output < (1 << dimension)
    return output


def enumerate_binary_matrices(dimension: int) -> Iterable[tuple[int, ...]]:
    for entries in itertools.product((0, 1), repeat=dimension * dimension):
        rows = []
        for row in range(dimension):
            row_mask = 0
            for column in range(dimension):
                if entries[row * dimension + column]:
                    row_mask |= 1 << column
            rows.append(row_mask)
        yield tuple(rows)


def is_invertible_binary(rows: Sequence[int], dimension: int) -> bool:
    images = {
        apply_binary_matrix(rows, vector, dimension)
        for vector in range(1 << dimension)
    }
    return len(images) == (1 << dimension)


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def is_hamming_isometry(rows: Sequence[int], dimension: int) -> bool:
    for left in range(1 << dimension):
        transformed_left = apply_binary_matrix(rows, left, dimension)
        for right in range(1 << dimension):
            transformed_right = apply_binary_matrix(rows, right, dimension)
            if hamming_distance(transformed_left, transformed_right) != hamming_distance(
                left, right
            ):
                return False
    return True


def permutation_matrix(permutation: Sequence[int]) -> tuple[int, ...]:
    rows = [0] * len(permutation)
    for source, target in enumerate(permutation):
        rows[target] |= 1 << source
    return tuple(rows)


def compose(first: Sequence[int], second: Sequence[int]) -> tuple[int, ...]:
    """Apply first, then second, using source->target permutation convention."""
    assert len(first) == len(second)
    return tuple(second[first[index]] for index in range(len(first)))


def permutation_power(permutation: Sequence[int], exponent: int) -> tuple[int, ...]:
    result = tuple(range(len(permutation)))
    for _ in range(exponent):
        result = compose(result, permutation)
    return result


def binary_gl3_hamming_census() -> dict[str, object]:
    dimension = 3
    invertible = [
        rows
        for rows in enumerate_binary_matrices(dimension)
        if is_invertible_binary(rows, dimension)
    ]
    hamming_isometries = [
        rows for rows in invertible if is_hamming_isometry(rows, dimension)
    ]
    coordinate_permutations = {
        permutation_matrix(permutation)
        for permutation in itertools.permutations(range(dimension))
    }

    assert len(invertible) == 168
    assert len(hamming_isometries) == 6
    assert set(hamming_isometries) == coordinate_permutations

    return {
        "dimension": dimension,
        "all_binary_matrices": 2 ** (dimension * dimension),
        "invertible_gl_count": len(invertible),
        "hamming_isometry_count": len(hamming_isometries),
        "coordinate_permutation_count": len(coordinate_permutations),
        "hamming_isometries_equal_coordinate_permutations": True,
    }


def apply_signed_permutation(
    vector: Sequence[int], permutation: Sequence[int], signs: Sequence[int]
) -> tuple[int, ...]:
    assert len(vector) == len(permutation) == len(signs)
    output = [0] * len(vector)
    for source, target in enumerate(permutation):
        output[target] = signs[source] * vector[source]
    return tuple(output)


def integer_dot(left: Sequence[int], right: Sequence[int]) -> int:
    return sum(a * b for a, b in zip(left, right))


def hadamard(left: Sequence[int], right: Sequence[int]) -> tuple[int, ...]:
    return tuple(a * b for a, b in zip(left, right))


def continuous_signed_permutation_census() -> dict[str, object]:
    dimension = 3
    corpus = list(itertools.product((-1, 0, 1), repeat=dimension))
    candidates = [
        (permutation, signs)
        for permutation in itertools.permutations(range(dimension))
        for signs in itertools.product((-1, 1), repeat=dimension)
    ]

    inner_product_isometries = []
    hadamard_automorphisms = []

    for permutation, signs in candidates:
        preserves_inner_product = all(
            integer_dot(
                apply_signed_permutation(left, permutation, signs),
                apply_signed_permutation(right, permutation, signs),
            )
            == integer_dot(left, right)
            for left in corpus
            for right in corpus
        )
        if preserves_inner_product:
            inner_product_isometries.append((permutation, signs))

        preserves_hadamard = all(
            apply_signed_permutation(hadamard(left, right), permutation, signs)
            == hadamard(
                apply_signed_permutation(left, permutation, signs),
                apply_signed_permutation(right, permutation, signs),
            )
            for left in corpus
            for right in corpus
        )
        if preserves_hadamard:
            hadamard_automorphisms.append((permutation, signs))

    unsigned_coordinate_permutations = {
        (permutation, (1,) * dimension)
        for permutation in itertools.permutations(range(dimension))
    }

    assert len(candidates) == 48
    assert len(inner_product_isometries) == 48
    assert len(hadamard_automorphisms) == 6
    assert set(hadamard_automorphisms) == unsigned_coordinate_permutations

    return {
        "dimension": dimension,
        "corpus_size": len(corpus),
        "signed_coordinate_permutation_count": len(candidates),
        "inner_product_isometry_count": len(inner_product_isometries),
        "hadamard_automorphism_count": len(hadamard_automorphisms),
        "hadamard_automorphisms_equal_unsigned_coordinate_permutations": True,
    }


def cycle_centralizer_census() -> dict[str, object]:
    dimension = 5
    cycle = tuple((index + 1) % dimension for index in range(dimension))
    symmetric_group = list(itertools.permutations(range(dimension)))
    centralizer = [
        permutation
        for permutation in symmetric_group
        if compose(permutation, cycle) == compose(cycle, permutation)
    ]
    cycle_powers = {permutation_power(cycle, exponent) for exponent in range(dimension)}

    assert len(symmetric_group) == 120
    assert len(centralizer) == 5
    assert set(centralizer) == cycle_powers

    return {
        "dimension": dimension,
        "symmetric_group_count": len(symmetric_group),
        "centralizer_count": len(centralizer),
        "cycle_power_count": len(cycle_powers),
        "centralizer_equals_cycle_powers": True,
    }


def main() -> None:
    report = {
        "schema": "symthaea.rel.finite-symmetry-oracle.v2",
        "authority": "MeasurementOnly",
        "binary_gl3_hamming": binary_gl3_hamming_census(),
        "continuous_signed_permutations": continuous_signed_permutation_census(),
        "sequence_cycle_centralizer": cycle_centralizer_census(),
        "claims": {
            "production_symthaea_execution": False,
            "production_dimension_exhaustively_enumerated": False,
            "all_real_linear_maps_exhaustively_enumerated": False,
            "field_theoretic_gauge_symmetry_established": False,
            "intelligence_improvement_established": False,
            "consciousness_claim_established": False,
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
