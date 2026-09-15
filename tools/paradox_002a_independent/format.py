"""PARADOX-002A-I frozen binary format parser/encoder.

This module owns framing only. It deliberately contains no manipulation theorem.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import struct
from typing import Iterable

DOMAIN = b"SYMT-PARADOX-002A-CONFIRMATORY-CENSUS-V1\0"
CONFIRMATORY_SEEDS = (11, 29, 47, 71, 101, 149, 197, 257, 331, 419, 521, 631, 751, 887, 1021, 1171)
CONDITION_COUNT = 7
TRIALS_PER_CONDITION = 64
RECORD_COUNT = len(CONFIRMATORY_SEEDS) * CONDITION_COUNT * TRIALS_PER_CONDITION
MATCHED_BUDGET = (4, 3, 1, 2, 4, 8, 4, 2)

POLARITIES = {1, 2}
EVENT_ROLES = {1, 2, 3}
EXPECTED_RESPONSES = {1, 2, 3, 4, 5}
RESOLUTION_CLASSES = {1, 2, 3, 4, 5}
INCONSISTENCY_KINDS = set(range(8))
RESOLUTION_STATES = set(range(6))
EVIDENCE_POLARITIES = set(range(4))


class FormatError(ValueError):
    pass


@dataclass
class F64:
    bits: int

    @classmethod
    def from_float(cls, value: float) -> "F64":
        return cls(struct.unpack("<Q", struct.pack("<d", value))[0])

    @property
    def value(self) -> float:
        return struct.unpack("<d", struct.pack("<Q", self.bits))[0]

    def require_finite(self, name: str) -> None:
        if not math.isfinite(self.value):
            raise FormatError(f"{name} must be finite")


@dataclass
class EvidenceEvent:
    slot: int
    source_id: int
    fault_domain_id: int
    polarity: int | None
    reliability: F64
    surface_cue: int
    role: int
    supersedes_slot: int | None
    caused_by_self_prediction: bool
    visible_context: int | None


@dataclass
class AgentView:
    prior_expectation: int
    query_cue: int
    budget: tuple[int, int, int, int, int, int, int, int]
    events: list[EvidenceEvent]


@dataclass
class OracleTruth:
    hidden_claim_contexts: list[int | None]
    latent_context_is_causal: bool
    context_dimension_available: bool
    target_context: int | None


@dataclass
class Fixture:
    condition: int
    seed: int
    trial_index: int
    agent_view: AgentView
    truth: OracleTruth


@dataclass
class ObservatoryReport:
    kind: int
    support_proposition: F64
    support_negation: F64
    resolution: int
    external_surprise: F64
    internal_disagreement: F64
    uncertainty: F64
    persistence: F64
    self_referential_relevance: F64
    evidence_polarity: int
    integration_coherence: F64
    conflict_load: F64
    candidate_recruitment_index: F64


@dataclass
class OracleReport:
    external_surprise: F64
    internal_disagreement: bool
    final_unresolved_disagreement: bool
    conflict_persistence: F64
    independent_conflict_sources: bool
    self_referential: bool
    explicit_context_resolution: bool
    ontology_failure: bool
    irreducible: bool
    conflict_slots: int
    expected_response: int
    resolution_class: int
    target_polarity: int | None
    observatory: ObservatoryReport


@dataclass
class Pair:
    fixture: Fixture
    report: OracleReport


@dataclass
class Census:
    pairs: list[Pair]


class Reader:
    def __init__(self, data: bytes, label: str = "stream") -> None:
        self.data = data
        self.pos = 0
        self.label = label

    @property
    def remaining(self) -> int:
        return len(self.data) - self.pos

    def take(self, size: int) -> bytes:
        if size < 0 or size > self.remaining:
            raise FormatError(f"{self.label}: truncated read of {size} bytes at offset {self.pos}")
        out = self.data[self.pos:self.pos + size]
        self.pos += size
        return out

    def u8(self) -> int:
        return self.take(1)[0]

    def u16(self) -> int:
        return struct.unpack("<H", self.take(2))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.take(8))[0]

    def f64(self, name: str) -> F64:
        out = F64(self.u64())
        out.require_finite(name)
        return out

    def boolean(self, name: str) -> bool:
        value = self.u8()
        if value not in (0, 1):
            raise FormatError(f"{self.label}: {name} has non-canonical bool byte {value}")
        return bool(value)

    def enum_u8(self, allowed: set[int], name: str) -> int:
        value = self.u8()
        if value not in allowed:
            raise FormatError(f"{self.label}: unknown {name} value {value}")
        return value

    def optional_polarity(self, name: str) -> int | None:
        tag = self.u8()
        if tag == 0:
            return None
        if tag != 1:
            raise FormatError(f"{self.label}: {name} has invalid option tag {tag}")
        return self.enum_u8(POLARITIES, name)

    def optional_context(self, name: str) -> int | None:
        tag = self.u8()
        if tag == 0:
            return None
        if tag != 1:
            raise FormatError(f"{self.label}: {name} has invalid option tag {tag}")
        return self.u16()

    def finish(self) -> None:
        if self.remaining != 0:
            raise FormatError(f"{self.label}: {self.remaining} trailing bytes")


def _parse_event(reader: Reader, index: int) -> EvidenceEvent:
    return EvidenceEvent(
        slot=reader.u8(),
        source_id=reader.u16(),
        fault_domain_id=reader.u16(),
        polarity=reader.optional_polarity(f"event[{index}].polarity"),
        reliability=reader.f64(f"event[{index}].reliability"),
        surface_cue=reader.u16(),
        role=reader.enum_u8(EVENT_ROLES, f"event[{index}].role"),
        supersedes_slot=_optional_u8(reader, f"event[{index}].supersedes_slot"),
        caused_by_self_prediction=reader.boolean(f"event[{index}].caused_by_self_prediction"),
        visible_context=reader.optional_context(f"event[{index}].visible_context"),
    )


def _optional_u8(reader: Reader, name: str) -> int | None:
    tag = reader.u8()
    if tag == 0:
        return None
    if tag != 1:
        raise FormatError(f"{reader.label}: {name} has invalid option tag {tag}")
    return reader.u8()


def parse_agent(data: bytes) -> AgentView:
    reader = Reader(data, "AgentView")
    prior = reader.enum_u8(POLARITIES, "prior_expectation")
    query = reader.u16()
    budget = tuple(reader.u8() for _ in range(8))
    events = [_parse_event(reader, i) for i in range(4)]
    reader.finish()
    return AgentView(prior, query, budget, events)  # type: ignore[arg-type]


def parse_fixture(data: bytes) -> Fixture:
    reader = Reader(data, "Fixture")
    condition = reader.enum_u8(set(range(CONDITION_COUNT)), "condition")
    seed = reader.u64()
    trial = reader.u64()
    agent_len = reader.u64()
    if agent_len > reader.remaining:
        raise FormatError("Fixture: declared AgentView length exceeds fixture bytes")
    agent = parse_agent(reader.take(agent_len))
    hidden = [reader.optional_context(f"hidden_claim_contexts[{i}]") for i in range(4)]
    latent = reader.boolean("latent_context_is_causal")
    available = reader.boolean("context_dimension_available")
    target = reader.optional_context("target_context")
    reader.finish()
    return Fixture(condition, seed, trial, agent, OracleTruth(hidden, latent, available, target))


def parse_report(data: bytes) -> OracleReport:
    reader = Reader(data, "OracleReport")
    external = reader.f64("external_surprise")
    internal = reader.boolean("internal_disagreement")
    final = reader.boolean("final_unresolved_disagreement")
    persistence = reader.f64("conflict_persistence")
    independent = reader.boolean("independent_conflict_sources")
    self_ref = reader.boolean("self_referential")
    explicit = reader.boolean("explicit_context_resolution")
    ontology = reader.boolean("ontology_failure")
    irreducible = reader.boolean("irreducible")
    slots = reader.u8()
    expected = reader.enum_u8(EXPECTED_RESPONSES, "expected_response")
    resolution_class = reader.enum_u8(RESOLUTION_CLASSES, "resolution_class")
    target = reader.optional_polarity("target_polarity")
    obs = ObservatoryReport(
        kind=reader.enum_u8(INCONSISTENCY_KINDS, "observatory.kind"),
        support_proposition=reader.f64("observatory.support_proposition"),
        support_negation=reader.f64("observatory.support_negation"),
        resolution=reader.enum_u8(RESOLUTION_STATES, "observatory.resolution"),
        external_surprise=reader.f64("observatory.external_surprise"),
        internal_disagreement=reader.f64("observatory.internal_disagreement"),
        uncertainty=reader.f64("observatory.uncertainty"),
        persistence=reader.f64("observatory.persistence"),
        self_referential_relevance=reader.f64("observatory.self_referential_relevance"),
        evidence_polarity=reader.enum_u8(EVIDENCE_POLARITIES, "observatory.evidence_polarity"),
        integration_coherence=reader.f64("observatory.integration_coherence"),
        conflict_load=reader.f64("observatory.conflict_load"),
        candidate_recruitment_index=reader.f64("observatory.candidate_recruitment_index"),
    )
    reader.finish()
    return OracleReport(
        external, internal, final, persistence, independent, self_ref, explicit,
        ontology, irreducible, slots, expected, resolution_class, target, obs,
    )


def parse_census(data: bytes, *, enforce_traversal: bool = True) -> Census:
    reader = Reader(data, "Census")
    if reader.take(len(DOMAIN)) != DOMAIN:
        raise FormatError("Census: domain/version prefix mismatch")
    seeds = reader.u64()
    conditions = reader.u64()
    trials = reader.u64()
    if (seeds, conditions, trials) != (len(CONFIRMATORY_SEEDS), CONDITION_COUNT, TRIALS_PER_CONDITION):
        raise FormatError(f"Census: cardinality mismatch {(seeds, conditions, trials)!r}")

    pairs: list[Pair] = []
    seen: set[tuple[int, int, int]] = set()
    ordinal = 0
    for expected_seed in CONFIRMATORY_SEEDS:
        for expected_condition in range(CONDITION_COUNT):
            for expected_trial in range(TRIALS_PER_CONDITION):
                fixture_len = reader.u64()
                if fixture_len > reader.remaining:
                    raise FormatError("Census: fixture record length exceeds remaining bytes")
                fixture = parse_fixture(reader.take(fixture_len))
                report_len = reader.u64()
                if report_len > reader.remaining:
                    raise FormatError("Census: report record length exceeds remaining bytes")
                report = parse_report(reader.take(report_len))
                identity = (fixture.seed, fixture.condition, fixture.trial_index)
                if identity in seen:
                    raise FormatError(f"Census: duplicate fixture identity {identity}")
                seen.add(identity)
                if enforce_traversal and identity != (expected_seed, expected_condition, expected_trial):
                    raise FormatError(
                        f"Census: traversal mismatch at ordinal {ordinal}: "
                        f"got {identity}, expected {(expected_seed, expected_condition, expected_trial)}"
                    )
                pairs.append(Pair(fixture, report))
                ordinal += 1
    reader.finish()
    if len(pairs) != RECORD_COUNT:
        raise FormatError(f"Census: expected {RECORD_COUNT} pairs, got {len(pairs)}")
    return Census(pairs)


def _u8(value: int) -> bytes:
    return bytes((value,))


def _u16(value: int) -> bytes:
    return struct.pack("<H", value)


def _u64(value: int) -> bytes:
    return struct.pack("<Q", value)


def _bool(value: bool) -> bytes:
    return _u8(1 if value else 0)


def _opt_polarity(value: int | None) -> bytes:
    return _u8(0) if value is None else _u8(1) + _u8(value)


def _opt_context(value: int | None) -> bytes:
    return _u8(0) if value is None else _u8(1) + _u16(value)


def _opt_u8(value: int | None) -> bytes:
    return _u8(0) if value is None else _u8(1) + _u8(value)


def encode_event(event: EvidenceEvent) -> bytes:
    return b"".join((
        _u8(event.slot), _u16(event.source_id), _u16(event.fault_domain_id),
        _opt_polarity(event.polarity), _u64(event.reliability.bits), _u16(event.surface_cue),
        _u8(event.role), _opt_u8(event.supersedes_slot), _bool(event.caused_by_self_prediction),
        _opt_context(event.visible_context),
    ))


def encode_agent(agent: AgentView) -> bytes:
    return b"".join((
        _u8(agent.prior_expectation), _u16(agent.query_cue), bytes(agent.budget),
        *(encode_event(event) for event in agent.events),
    ))


def encode_fixture(fixture: Fixture) -> bytes:
    agent = encode_agent(fixture.agent_view)
    return b"".join((
        _u8(fixture.condition), _u64(fixture.seed), _u64(fixture.trial_index),
        _u64(len(agent)), agent,
        *(_opt_context(value) for value in fixture.truth.hidden_claim_contexts),
        _bool(fixture.truth.latent_context_is_causal),
        _bool(fixture.truth.context_dimension_available),
        _opt_context(fixture.truth.target_context),
    ))


def encode_report(report: OracleReport) -> bytes:
    obs = report.observatory
    return b"".join((
        _u64(report.external_surprise.bits), _bool(report.internal_disagreement),
        _bool(report.final_unresolved_disagreement), _u64(report.conflict_persistence.bits),
        _bool(report.independent_conflict_sources), _bool(report.self_referential),
        _bool(report.explicit_context_resolution), _bool(report.ontology_failure),
        _bool(report.irreducible), _u8(report.conflict_slots), _u8(report.expected_response),
        _u8(report.resolution_class), _opt_polarity(report.target_polarity),
        _u8(obs.kind), _u64(obs.support_proposition.bits), _u64(obs.support_negation.bits),
        _u8(obs.resolution), _u64(obs.external_surprise.bits), _u64(obs.internal_disagreement.bits),
        _u64(obs.uncertainty.bits), _u64(obs.persistence.bits),
        _u64(obs.self_referential_relevance.bits), _u8(obs.evidence_polarity),
        _u64(obs.integration_coherence.bits), _u64(obs.conflict_load.bits),
        _u64(obs.candidate_recruitment_index.bits),
    ))


def encode_census(census: Census) -> bytes:
    chunks: list[bytes] = [
        DOMAIN,
        _u64(len(CONFIRMATORY_SEEDS)),
        _u64(CONDITION_COUNT),
        _u64(TRIALS_PER_CONDITION),
    ]
    for pair in census.pairs:
        fixture = encode_fixture(pair.fixture)
        report = encode_report(pair.report)
        chunks.extend((_u64(len(fixture)), fixture, _u64(len(report)), report))
    return b"".join(chunks)


def record_fixture_start(data: bytes, ordinal: int = 0) -> int:
    """Return absolute start of a fixture record's bytes (after its u64 length)."""
    if ordinal < 0 or ordinal >= RECORD_COUNT:
        raise ValueError("ordinal out of range")
    reader = Reader(data, "CensusSpan")
    reader.take(len(DOMAIN) + 24)
    for i in range(ordinal + 1):
        fixture_len = reader.u64()
        start = reader.pos
        if i == ordinal:
            return start
        reader.take(fixture_len)
        report_len = reader.u64()
        reader.take(report_len)
    raise AssertionError("unreachable")
