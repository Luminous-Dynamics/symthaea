#!/usr/bin/env python3
"""Independent LL-007A/B catcher momentum and energy oracle.

Research-only first-order accounting. No hardware control, capture qualification,
or real launcher/catcher sizing is implied.
"""
from __future__ import annotations
import argparse, json, math
from dataclasses import asdict, dataclass

@dataclass(frozen=True)
class CaptureInput:
    pod_mass_kg: float
    catcher_mass_kg: float
    incoming_relative_speed_m_s: float
    final_relative_speed_m_s: float
    capture_duration_s: float
    regeneration_efficiency: float
    storage_efficiency: float
    storage_acceptance_j: float
    momentum_transfer_fraction: float
    direction_sign: int = 1

@dataclass(frozen=True)
class CaptureResult:
    incoming_signed_momentum_kg_m_s: float
    required_impulse_n_s: float
    signed_catcher_momentum_change_kg_m_s: float
    external_momentum_sink_kg_m_s: float
    catcher_delta_v_m_s: float
    stationkeeping_impulse_to_restore_n_s: float
    kinetic_energy_removed_j: float
    average_capture_force_n: float
    constant_deceleration_distance_m: float
    average_mechanical_power_w: float
    regenerable_mechanical_energy_j: float
    stored_electrical_energy_j: float
    residual_energy_j: float

@dataclass(frozen=True)
class FleetLedger:
    captures: int
    gross_transferred_impulse_n_s: float
    net_catcher_momentum_kg_m_s: float
    minimum_net_stationkeeping_impulse_n_s: float
    stored_electrical_energy_j: float
    residual_energy_j: float

def finite(*values: float) -> bool:
    return all(math.isfinite(v) for v in values)

def validate(inp: CaptureInput) -> None:
    vals = (
        inp.pod_mass_kg, inp.catcher_mass_kg, inp.incoming_relative_speed_m_s,
        inp.final_relative_speed_m_s, inp.capture_duration_s,
        inp.regeneration_efficiency, inp.storage_efficiency,
        inp.storage_acceptance_j, inp.momentum_transfer_fraction,
    )
    if not finite(*vals):
        raise ValueError("all scalar inputs must be finite")
    if inp.pod_mass_kg <= 0 or inp.catcher_mass_kg <= 0 or inp.capture_duration_s <= 0:
        raise ValueError("mass and duration must be positive")
    if inp.incoming_relative_speed_m_s < 0 or inp.final_relative_speed_m_s < 0:
        raise ValueError("relative speeds must be nonnegative magnitudes")
    if inp.final_relative_speed_m_s > inp.incoming_relative_speed_m_s:
        raise ValueError("reference capture model cannot accelerate the pod")
    if not (0 <= inp.regeneration_efficiency <= 1):
        raise ValueError("invalid regeneration efficiency")
    if not (0 <= inp.storage_efficiency <= 1):
        raise ValueError("invalid storage efficiency")
    if inp.storage_acceptance_j < 0:
        raise ValueError("storage acceptance must be nonnegative")
    if not (0 <= inp.momentum_transfer_fraction <= 1):
        raise ValueError("invalid momentum transfer fraction")
    if inp.direction_sign not in (-1, 1):
        raise ValueError("direction_sign must be -1 or +1")

def capture(inp: CaptureInput) -> CaptureResult:
    validate(inp)
    dv = inp.incoming_relative_speed_m_s - inp.final_relative_speed_m_s
    impulse = inp.pod_mass_kg * dv
    signed_incoming_momentum = inp.direction_sign * inp.pod_mass_kg * inp.incoming_relative_speed_m_s
    signed_catcher_momentum = inp.direction_sign * impulse * inp.momentum_transfer_fraction
    external_momentum = impulse * (1 - inp.momentum_transfer_fraction)
    catcher_dv = signed_catcher_momentum / inp.catcher_mass_kg

    energy_removed = 0.5 * inp.pod_mass_kg * (
        inp.incoming_relative_speed_m_s**2 - inp.final_relative_speed_m_s**2
    )
    average_force = impulse / inp.capture_duration_s
    distance = 0.5 * (
        inp.incoming_relative_speed_m_s + inp.final_relative_speed_m_s
    ) * inp.capture_duration_s
    avg_power = energy_removed / inp.capture_duration_s
    regen_mech = energy_removed * inp.regeneration_efficiency
    converted = regen_mech * inp.storage_efficiency
    stored = min(converted, inp.storage_acceptance_j)
    residual = energy_removed - stored

    return CaptureResult(
        signed_incoming_momentum,
        impulse,
        signed_catcher_momentum,
        external_momentum,
        catcher_dv,
        abs(signed_catcher_momentum),
        energy_removed,
        average_force,
        distance,
        avg_power,
        regen_mech,
        stored,
        residual,
    )

def fleet(inputs: list[CaptureInput]) -> FleetLedger:
    results = [capture(i) for i in inputs]
    net = sum(r.signed_catcher_momentum_change_kg_m_s for r in results)
    return FleetLedger(
        captures=len(results),
        gross_transferred_impulse_n_s=sum(abs(r.signed_catcher_momentum_change_kg_m_s) for r in results),
        net_catcher_momentum_kg_m_s=net,
        minimum_net_stationkeeping_impulse_n_s=abs(net),
        stored_electrical_energy_j=sum(r.stored_electrical_energy_j for r in results),
        residual_energy_j=sum(r.residual_energy_j for r in results),
    )

def self_test() -> None:
    base = CaptureInput(
        pod_mass_kg=100.0,
        catcher_mass_kg=10_000.0,
        incoming_relative_speed_m_s=200.0,
        final_relative_speed_m_s=0.0,
        capture_duration_s=10.0,
        regeneration_efficiency=0.8,
        storage_efficiency=0.9,
        storage_acceptance_j=10_000_000.0,
        momentum_transfer_fraction=1.0,
        direction_sign=1,
    )
    r = capture(base)
    assert abs(r.required_impulse_n_s - 20_000.0) < 1e-12
    assert abs(r.kinetic_energy_removed_j - 2_000_000.0) < 1e-9
    assert abs(r.average_capture_force_n - 2_000.0) < 1e-12
    assert abs(r.constant_deceleration_distance_m - 1_000.0) < 1e-12
    assert abs(r.average_mechanical_power_w - 200_000.0) < 1e-9
    assert abs(r.regenerable_mechanical_energy_j - 1_600_000.0) < 1e-9
    assert abs(r.stored_electrical_energy_j - 1_440_000.0) < 1e-9
    assert abs(r.residual_energy_j - 560_000.0) < 1e-9
    assert abs(r.catcher_delta_v_m_s - 2.0) < 1e-12
    assert abs(r.stationkeeping_impulse_to_restore_n_s - 20_000.0) < 1e-12

    limited = CaptureInput(**{**base.__dict__, "storage_acceptance_j": 1_000_000.0})
    lr = capture(limited)
    assert lr.stored_electrical_energy_j == 1_000_000.0
    assert lr.residual_energy_j == 1_000_000.0

    lower_regen = CaptureInput(**{**base.__dict__, "regeneration_efficiency": 0.4})
    assert capture(lower_regen).residual_energy_j > r.residual_energy_j

    half_transfer = CaptureInput(**{**base.__dict__, "momentum_transfer_fraction": 0.5})
    hr = capture(half_transfer)
    assert abs(hr.catcher_delta_v_m_s - 1.0) < 1e-12
    assert abs(hr.external_momentum_sink_kg_m_s - 10_000.0) < 1e-12

    opposite = CaptureInput(**{**base.__dict__, "direction_sign": -1})
    ledger = fleet([base, opposite])
    assert abs(ledger.net_catcher_momentum_kg_m_s) < 1e-12
    assert abs(ledger.minimum_net_stationkeeping_impulse_n_s) < 1e-12
    assert ledger.gross_transferred_impulse_n_s == 40_000.0

    try:
        capture(CaptureInput(**{**base.__dict__, "final_relative_speed_m_s": 250.0}))
    except ValueError:
        pass
    else:
        raise AssertionError("accelerating capture case must fail")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--json")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("ok")
        return
    if not args.json:
        parser.error("--json or --self-test required")
    payload = json.loads(args.json)
    if "captures" in payload:
        inputs = [CaptureInput(**item) for item in payload["captures"]]
        print(json.dumps(asdict(fleet(inputs)), sort_keys=True))
    else:
        print(json.dumps(asdict(capture(CaptureInput(**payload))), sort_keys=True))

if __name__ == "__main__":
    main()
