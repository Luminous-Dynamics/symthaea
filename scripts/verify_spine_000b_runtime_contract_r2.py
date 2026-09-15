#!/usr/bin/env python3
"""Static consistency verifier for SPINE-000B runtime receipt contract R2.

Measurement-only. This does not execute cognition and does not qualify runtime
influence. It proves that the preregistered R2 contract names the current live
production seams accurately before instrumentation is implemented.
"""

from pathlib import Path

TRAIT_SOURCE = Path("src/cognitive_loop/subsystem_trait.rs")
DYNAMICS_SOURCE = Path("src/cognitive_loop/cycle_phase_dynamics/mod.rs")
CONTRACT = Path("docs/research/SPINE_000B_RUNTIME_RECEIPT_CONTRACT_R2.md")


def fail(msg: str) -> None:
    raise SystemExit(f"SPINE-000B-R2 CONTRACT FAIL: {msg}")


def require(text: str, needle: str, where: str) -> None:
    if needle not in text:
        fail(f"missing {needle!r} in {where}")


def main() -> int:
    for path in (TRAIT_SOURCE, DYNAMICS_SOURCE, CONTRACT):
        if not path.is_file():
            fail(f"missing subject {path}")

    trait = TRAIT_SOURCE.read_text(encoding="utf-8")
    dynamics = DYNAMICS_SOURCE.read_text(encoding="utf-8")
    contract = CONTRACT.read_text(encoding="utf-8")

    # Collector semantics relied upon by R2.
    require(trait, "if !output.is_neutral()", str(TRAIT_SOURCE))
    require(trait, "self.outputs.push((name, output));", str(TRAIT_SOURCE))
    require(trait, "pub fn integrate(&self) -> IntegratedOutput", str(TRAIT_SOURCE))

    # The live manager path is the Phase B macro, not safe_process().
    for needle in (
        "macro_rules! run_subsystem",
        "self.subsystem_health.is_faulted($name)",
        "std::panic::catch_unwind",
        "self.subsystem_health.record_success($name)",
        "self.subsystem_collector.record($name, output)",
        "self.subsystem_health.record_panic($name)",
        "should_run(cycle_num, urgency_u8)",
    ):
        require(dynamics, needle, str(DYNAMICS_SOURCE))

    # Contract must bind itself to that live path explicitly.
    for needle in (
        "live Phase B manager execution seam",
        "run_subsystem! macro",
        "eligible_to_run",
        "execution_attempted",
        "execution_completed",
        "EXECUTED_NEUTRAL",
        "emitted = true",
        "admitted = false",
        "PANICKED_CAUGHT",
        "NotApplicable(NotAdmitted)",
        "RuntimeTelemetryEnvelope",
        "StateApplicationReceipt",
        "CycleIntegrationReceipt",
        "SubsystemExecutionReceipt",
        "state_changed",
        "duration_ns",
        "measurement-only",
        "LOAD_BEARING",
    ):
        require(contract, needle, str(CONTRACT))

    # Prevent regression back to the helper-only seam.
    require(
        contract,
        "does **not** require routing live execution through `safe_process()`",
        str(CONTRACT),
    )

    # Prevent the two important attribution mistakes from returning silently.
    require(
        contract,
        "`n_contributors` is canonical metadata and MUST be compared",
        str(CONTRACT),
    )
    require(
        contract,
        "it is **not** itself an influence channel",
        str(CONTRACT),
    )
    require(contract, "SPINE-000B MUST NOT conclude", str(CONTRACT))
    require(contract, "subsystem S caused destination D to change", str(CONTRACT))

    # Empty cycles and nondeterministic timing both have explicit semantics.
    require(
        contract,
        "Emit exactly one cycle integration receipt for every qualified cognitive cycle",
        str(CONTRACT),
    )
    require(
        contract,
        "MUST NOT participate in canonical receipt hashes",
        str(CONTRACT),
    )

    print("SPINE-000B runtime contract R2 static consistency: PASS")
    print("live_execution_seam=cycle_phase_dynamics::run_subsystem")
    print("authority=measurement-only")
    print("runtime_evidence_claimed=false")
    print("causal_load_claimed=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
