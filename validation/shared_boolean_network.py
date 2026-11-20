"""Canonical low-node Boolean-network models shared by Rust and PyPhi research lanes.

The TSV manifest is the source of truth for rule sets, noise, and state ensembles.
This module deliberately exposes transition TPM construction separately from
PyPhi analysis so model identity can be checked before comparing measurements.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import struct
from typing import Iterable

import numpy as np

MANIFEST_PATH = Path(__file__).resolve().parent / "fixtures" / "entropy_causality_models.tsv"
MAX_NODES = 8
MODEL_SCHEMA = "boolean-network-v1"
PRIOR_SCHEMA = "state-prior-v1"


def f64_from_hex(encoded: str) -> float:
    """Decode a canonical IEEE-754 binary64 hexadecimal bit pattern."""
    value = struct.unpack(">d", bytes.fromhex(encoded))[0]
    if f64_to_hex(value) != encoded.lower():
        raise ValueError(f"non-canonical binary64 hex encoding: {encoded}")
    return value


def f64_to_hex(value: float) -> str:
    """Encode a Python float using the same binary64 bits as Rust f64::to_bits()."""
    return struct.pack(">d", float(value)).hex()


@dataclass(frozen=True)
class BooleanRule:
    op: str
    sources: tuple[int, ...] = ()
    constant: bool = False

    def evaluate(self, state_index: int) -> bool:
        bits = lambda node: ((state_index >> node) & 1) != 0
        if self.op == "copy":
            return bits(self.sources[0])
        if self.op == "xor":
            return bits(self.sources[0]) ^ bits(self.sources[1])
        if self.op == "and":
            return bits(self.sources[0]) and bits(self.sources[1])
        if self.op == "constant":
            return self.constant
        raise ValueError(f"unsupported Boolean operation: {self.op}")

    def token(self) -> str:
        if self.op == "constant":
            return f"constant:{1 if self.constant else 0}"
        return ":".join((self.op, *(str(source) for source in self.sources)))


@dataclass(frozen=True)
class BooleanNetworkCase:
    name: str
    node_count: int
    rules: tuple[BooleanRule, ...]
    flip_probability: float
    prior: tuple[float, ...]

    def canonical_model_specification(self) -> str:
        rule_list = ",".join(rule.token() for rule in self.rules)
        return (
            f"schema={MODEL_SCHEMA};rules={rule_list};"
            f"flip={self.flip_probability:.6f};flip_bits={f64_to_hex(self.flip_probability)}"
        )

    def canonical_prior_specification(self) -> str:
        entries = ",".join(
            f"{state}:{probability:.6f}:{f64_to_hex(probability)}"
            for state, probability in enumerate(self.prior)
        )
        return f"schema={PRIOR_SCHEMA};prior={entries}"


def _parse_rule(token: str, node_count: int) -> BooleanRule:
    parts = token.split(":")
    op = parts[0]
    if op == "copy" and len(parts) == 2:
        sources = (int(parts[1]),)
        rule = BooleanRule(op=op, sources=sources)
    elif op in {"xor", "and"} and len(parts) == 3:
        sources = (int(parts[1]), int(parts[2]))
        rule = BooleanRule(op=op, sources=sources)
    elif op == "constant" and len(parts) == 2 and parts[1] in {"0", "1"}:
        rule = BooleanRule(op=op, constant=parts[1] == "1")
    else:
        raise ValueError(f"invalid Boolean rule token: {token}")

    if any(source < 0 or source >= node_count for source in rule.sources):
        raise ValueError(f"rule source is outside the {node_count}-node model: {token}")
    return rule


def _parse_prior(specification: str, node_count: int) -> tuple[float, ...]:
    state_count = 1 << node_count
    if specification == "uniform":
        return tuple(1.0 / state_count for _ in range(state_count))
    if not specification.startswith("sparse:"):
        raise ValueError(f"invalid prior specification: {specification}")

    probabilities = [0.0] * state_count
    seen: set[int] = set()
    for token in specification.removeprefix("sparse:").split(","):
        try:
            state_text, probability_hex = token.split(":", maxsplit=1)
            state = int(state_text)
            probability = f64_from_hex(probability_hex)
        except (ValueError, struct.error) as error:
            raise ValueError(f"invalid sparse-prior entry: {token}") from error
        if state < 0 or state >= state_count:
            raise ValueError(f"prior state index is outside [0, {state_count}): {state}")
        if state in seen:
            raise ValueError(f"duplicate prior state index: {state}")
        seen.add(state)
        if not np.isfinite(probability) or probability < 0.0:
            raise ValueError(f"invalid prior probability at state {state}: {probability}")
        probabilities[state] = probability

    if abs(sum(probabilities) - 1.0) > 1e-12:
        raise ValueError("prior probabilities must sum to 1 within 1e-12")
    return tuple(probabilities)


def load_model_manifest(path: Path = MANIFEST_PATH) -> list[BooleanNetworkCase]:
    """Load, validate, and preserve order of canonical model cases."""
    cases: list[BooleanNetworkCase] = []
    names: set[str] = set()
    with path.open("r", encoding="utf-8") as manifest:
        for line_number, raw_line in enumerate(manifest, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) != 5:
                raise ValueError(
                    f"{path}:{line_number}: expected five tab-separated columns, got {len(fields)}"
                )
            name, count_text, rule_list, noise_hex, prior_spec = fields
            if name in names:
                raise ValueError(f"{path}:{line_number}: duplicate case name {name}")
            names.add(name)

            try:
                node_count = int(count_text)
                if not 1 <= node_count <= MAX_NODES:
                    raise ValueError(f"node_count must be in [1, {MAX_NODES}]")
                rules = tuple(_parse_rule(token, node_count) for token in rule_list.split(","))
                if len(rules) != node_count:
                    raise ValueError(f"expected {node_count} rules, got {len(rules)}")
                flip_probability = f64_from_hex(noise_hex)
                if not 0.0 <= flip_probability <= 0.5:
                    raise ValueError("flip probability must be in [0, 0.5]")
                prior = _parse_prior(prior_spec, node_count)
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error

            cases.append(
                BooleanNetworkCase(
                    name=name,
                    node_count=node_count,
                    rules=rules,
                    flip_probability=flip_probability,
                    prior=prior,
                )
            )
    if not cases:
        raise ValueError(f"no model cases found in {path}")
    return cases


def transition_tpm(case: BooleanNetworkCase) -> np.ndarray:
    """Create PyPhi-compatible state-by-node TPM using node 0 as the LSB."""
    state_count = 1 << case.node_count
    tpm = np.zeros((state_count, case.node_count), dtype=np.float64)
    noise = case.flip_probability
    for previous_state in range(state_count):
        for target, rule in enumerate(case.rules):
            raw_probability = 1.0 if rule.evaluate(previous_state) else 0.0
            tpm[previous_state, target] = noise + (1.0 - 2.0 * noise) * raw_probability
    return tpm


def connectivity_matrix(case: BooleanNetworkCase) -> np.ndarray:
    """Return PyPhi adjacency where cm[source, target] encodes source -> target."""
    cm = np.zeros((case.node_count, case.node_count), dtype=np.uint8)
    for target, rule in enumerate(case.rules):
        for source in rule.sources:
            cm[source, target] = 1
    return cm


def model_digest(case: BooleanNetworkCase) -> str:
    """Hash the canonical rule/noise specification shared with the Rust fixture."""
    digest = hashlib.sha256()
    digest.update(case.canonical_model_specification().encode("ascii"))
    return digest.hexdigest()


def prior_digest(case: BooleanNetworkCase) -> str:
    """Hash the prior/state ensemble separately from the causal model identity."""
    digest = hashlib.sha256()
    digest.update(case.canonical_prior_specification().encode("ascii"))
    return digest.hexdigest()
