"""Versioned provenance and content identity for persistent-platform solar resources.

Manifest integrity identifies the exact declared inputs. It does not certify
source truth, the physical model, or suitability for aircraft illumination.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import urlsplit

import reference_model as _reference_model
from reference_model import (
    HOURS_PER_DAY,
    SeasonalSolarScreen,
    SolarReserveSimulation,
    _finite,
    _fraction,
    _nonnegative,
    seasonal_solar_array_screen,
    simulate_solar_battery_reserve,
)


# Source-bound input identity -------------------------------------------------


_SOLAR_PROFILE_SCHEMA = "solar-resource-profile-v1"
_PROFILE_BASES = {
    "ground_fixed_plane",
    "airborne_model",
    "airborne_measured",
    "synthetic_fixture",
}
_SHA256_HEX = set("0123456789abcdef")
_TOP_LEVEL_FIELDS = {"schema_version", "profile_id", "resource_basis", "source", "geometry", "observations"}
_SOURCE_FIELDS = {"provider", "product", "dataset_version", "source_url", "retrieved_on", "period_start", "period_end", "request_manifest_sha256", "raw_artifact_sha256", "transform_sha256", "measurement_calibration_sha256"}
_GEOMETRY_FIELDS = {
    "latitude_deg", "longitude_deg", "elevation_m", "orientation_model",
    "plane_tilt_deg", "plane_azimuth_deg", "trajectory_or_flight_config_sha256",
}
_OBSERVATION_FIELDS = {
    "resolution", "time_basis", "units", "index", "plane_of_array_irradiation_kwh_m2_day", "daylight_hours",
}


def sha256_bytes(data: bytes) -> str:
    """Return the lowercase SHA-256 digest of exact artifact bytes."""
    if not isinstance(data, bytes):
        raise ValueError("data must be bytes")
    return hashlib.sha256(data).hexdigest()



def verify_solar_resource_artifact(manifest: dict, artifact_kind: str, artifact_bytes: bytes) -> str:
    """Verify exact bytes for a declared source, transformation, or calibration artifact.

    This proves byte identity only. It does not execute the transformation or
    prove that the declared observations were derived from the source artifact.
    """
    validate_solar_resource_manifest(manifest)
    fields = {
        "request_manifest": "request_manifest_sha256",
        "raw_source": "raw_artifact_sha256",
        "transformation": "transform_sha256",
        "measurement_calibration": "measurement_calibration_sha256",
    }
    if not isinstance(artifact_kind, str) or artifact_kind not in fields:
        raise ValueError(f"artifact_kind must be one of {sorted(fields)}")
    expected = manifest["source"][fields[artifact_kind]]
    if expected is None:
        raise ValueError(f"artifact_kind {artifact_kind} is not declared by this profile")
    actual = sha256_bytes(artifact_bytes)
    if actual != expected:
        raise ValueError(f"{artifact_kind} SHA-256 does not match manifest")
    return actual


def verify_solar_resource_raw_artifact(manifest: dict, raw_artifact: bytes) -> str:
    """Verify the exact source bytes declared by this profile."""
    return verify_solar_resource_artifact(manifest, "raw_source", raw_artifact)


@dataclass(frozen=True)
class SolarResourceArtifactVerification:
    """Byte-identity receipt for the exact profile and declared artifact set."""
    resource_profile_sha256: str
    request_manifest_sha256: str
    raw_artifact_sha256: str
    transform_sha256: str
    measurement_calibration_sha256: str | None


def verify_solar_resource_profile_artifacts(
    manifest: dict,
    *,
    request_manifest_bytes: bytes,
    raw_artifact_bytes: bytes,
    transformation_artifact_bytes: bytes,
    measurement_calibration_bytes: bytes | None = None,
) -> SolarResourceArtifactVerification:
    """Verify all declared bytes before running a profile-bound screening calculation.

    This receipt establishes exact artifact identity only. It does not execute
    the transformation, prove the observations were derived from the raw file,
    or establish the source's scientific validity.
    """
    validate_solar_resource_manifest(manifest)
    request_digest = verify_solar_resource_artifact(manifest, "request_manifest", request_manifest_bytes)
    request = _parse_json_object(request_manifest_bytes, "request_manifest")
    validate_solar_resource_request_manifest(manifest, request)
    raw_digest = verify_solar_resource_artifact(manifest, "raw_source", raw_artifact_bytes)
    transform_digest = verify_solar_resource_artifact(manifest, "transformation", transformation_artifact_bytes)
    calibration_digest = manifest["source"]["measurement_calibration_sha256"]
    if calibration_digest is not None:
        if measurement_calibration_bytes is None:
            raise ValueError("measurement_calibration_bytes required by this profile")
        calibration_digest = verify_solar_resource_artifact(
            manifest, "measurement_calibration", measurement_calibration_bytes
        )
    elif measurement_calibration_bytes is not None:
        raise ValueError("measurement_calibration_bytes supplied to a profile without a calibration digest")
    return SolarResourceArtifactVerification(
        resource_profile_sha256=solar_resource_profile_sha256(manifest),
        request_manifest_sha256=request_digest,
        raw_artifact_sha256=raw_digest,
        transform_sha256=transform_digest,
        measurement_calibration_sha256=calibration_digest,
    )


def _require_profile_verification(
    manifest: dict,
    verification: SolarResourceArtifactVerification,
) -> str:
    if not isinstance(verification, SolarResourceArtifactVerification):
        raise ValueError("artifact_verification must be a SolarResourceArtifactVerification receipt")
    profile_digest = solar_resource_profile_sha256(manifest)
    expected = manifest["source"]
    actuals = {
        "request_manifest_sha256": verification.request_manifest_sha256,
        "raw_artifact_sha256": verification.raw_artifact_sha256,
        "transform_sha256": verification.transform_sha256,
        "measurement_calibration_sha256": verification.measurement_calibration_sha256,
    }
    if verification.resource_profile_sha256 != profile_digest:
        raise ValueError("artifact-verification receipt belongs to a different resource profile")
    for field, actual in actuals.items():
        if actual != expected[field]:
            raise ValueError(f"artifact-verification receipt mismatch for {field}")
    return profile_digest


def canonical_solar_resource_json(manifest: dict) -> str:
    """Validate and serialize a profile with stable key order and separators."""
    validate_solar_resource_manifest(manifest)
    return json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def solar_resource_profile_sha256(manifest: dict) -> str:
    """Digest the exact validated resource manifest, including observations and lineage."""
    return hashlib.sha256(canonical_solar_resource_json(manifest).encode("utf-8")).hexdigest()


def _reject_duplicate_json_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str):
    raise ValueError(f"non-standard JSON numeric constant is forbidden: {value}")


def load_solar_resource_manifest(text: str) -> dict:
    """Decode JSON while rejecting duplicate keys, NaN, and infinity, then validate."""
    if not isinstance(text, str):
        raise ValueError("manifest JSON must be text")
    try:
        manifest = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError(f"invalid solar-resource JSON: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("solar-resource manifest root must be an object")
    validate_solar_resource_manifest(manifest)
    return manifest


def _exact_fields(value, expected: set[str], name: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    missing = expected - set(value)
    unknown = set(value) - expected
    if missing:
        raise ValueError(f"{name} missing required fields: {', '.join(sorted(missing))}")
    if unknown:
        raise ValueError(f"{name} has unknown fields: {', '.join(sorted(unknown))}")


def _nonempty_text(name: str, value) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    if any(ord(char) < 32 for char in value):
        raise ValueError(f"{name} must not contain control characters")
    return value


def _is_sha256(name: str, value) -> str:
    value = _nonempty_text(name, value)
    if len(value) != 64 or any(char not in _SHA256_HEX for char in value):
        raise ValueError(f"{name} must be 64 lowercase hexadecimal characters")
    return value


def _iso_date(name: str, value) -> date:
    value = _nonempty_text(name, value)
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO date (YYYY-MM-DD)") from exc
    if parsed.isoformat() != value:
        raise ValueError(f"{name} must use canonical YYYY-MM-DD form")
    return parsed


def _optional_angle(name: str, value, lower: float, upper: float) -> float | None:
    if value is None:
        return None
    value = _finite(name, value)
    if value < lower or value > upper:
        raise ValueError(f"{name} must be in [{lower}, {upper}]")
    return value



_REQUEST_FIELDS = {
    "schema_version", "provider", "product", "dataset_version", "source_url",
    "period_start", "period_end", "observation_resolution", "time_basis",
    "geometry", "request_parameters",
}


def _parse_json_object(raw: bytes, name: str) -> dict:
    if not isinstance(raw, bytes):
        raise ValueError(f"{name} must be bytes")
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
        raise ValueError(f"invalid {name} JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} root must be an object")
    return value


def validate_solar_resource_request_manifest(manifest: dict, request: dict) -> None:
    """Require exact request metadata to agree with the profile manifest."""
    _exact_fields(request, _REQUEST_FIELDS, "request_manifest")
    if request["schema_version"] != "solar-resource-request-v1":
        raise ValueError("request_manifest.schema_version must equal solar-resource-request-v1")
    source = manifest["source"]
    for name in ("provider", "product", "dataset_version", "source_url", "period_start", "period_end"):
        if request[name] != source[name]:
            raise ValueError(f"request manifest {name} does not match profile source metadata")
    request_geometry = json.dumps(request["geometry"], sort_keys=True, separators=(",", ":"), allow_nan=False)
    profile_geometry = json.dumps(manifest["geometry"], sort_keys=True, separators=(",", ":"), allow_nan=False)
    if request_geometry != profile_geometry:
        raise ValueError("request manifest geometry does not match resource profile geometry")
    observations = manifest["observations"]
    if request["observation_resolution"] != observations["resolution"]:
        raise ValueError("request manifest observation_resolution does not match resource profile")
    if request["time_basis"] != observations["time_basis"]:
        raise ValueError("request manifest time_basis does not match resource profile")
    if not isinstance(request["request_parameters"], dict):
        raise ValueError("request_manifest.request_parameters must be an object")
    try:
        json.dumps(request["request_parameters"], sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"request_manifest.request_parameters is not canonical JSON data: {exc}") from exc


def validate_solar_resource_manifest(manifest: dict) -> None:
    """Fail-closed validation for versioned solar-resource lineage and observations.

    A valid manifest gives input identity and provenance metadata; it does not
    validate the underlying physical resource model, database accuracy, or
    whether the declared source honestly supports the stated resource basis.
    """
    _exact_fields(manifest, _TOP_LEVEL_FIELDS, "manifest")
    if manifest["schema_version"] != _SOLAR_PROFILE_SCHEMA:
        raise ValueError(f"schema_version must equal {_SOLAR_PROFILE_SCHEMA}")
    _nonempty_text("profile_id", manifest["profile_id"])

    basis = manifest["resource_basis"]
    if not isinstance(basis, str) or basis not in _PROFILE_BASES:
        raise ValueError(f"resource_basis must be one of {sorted(_PROFILE_BASES)}")

    source = manifest["source"]
    _exact_fields(source, _SOURCE_FIELDS, "source")
    _nonempty_text("source.provider", source["provider"])
    _nonempty_text("source.product", source["product"])
    _nonempty_text("source.dataset_version", source["dataset_version"])
    source_url = _nonempty_text("source.source_url", source["source_url"])
    parsed_url = urlsplit(source_url)
    if basis == "synthetic_fixture":
        if parsed_url.scheme not in {"https", "urn"}:
            raise ValueError("synthetic source_url must be HTTPS or a URN")
    else:
        if parsed_url.scheme != "https" or not parsed_url.netloc:
            raise ValueError("non-synthetic source_url must be an absolute HTTPS URL")
    _iso_date("source.retrieved_on", source["retrieved_on"])
    period_start = _iso_date("source.period_start", source["period_start"])
    period_end = _iso_date("source.period_end", source["period_end"])
    if period_end < period_start:
        raise ValueError("source.period_end must not precede source.period_start")
    _is_sha256("source.request_manifest_sha256", source["request_manifest_sha256"])
    _is_sha256("source.raw_artifact_sha256", source["raw_artifact_sha256"])
    _is_sha256("source.transform_sha256", source["transform_sha256"])
    calibration_digest = source["measurement_calibration_sha256"]
    if calibration_digest is not None:
        _is_sha256("source.measurement_calibration_sha256", calibration_digest)
    if basis == "airborne_measured" and calibration_digest is None:
        raise ValueError("airborne_measured requires measurement_calibration_sha256")
    if basis != "airborne_measured" and calibration_digest is not None:
        raise ValueError("measurement_calibration_sha256 is only valid for airborne_measured profiles")

    geometry = manifest["geometry"]
    _exact_fields(geometry, _GEOMETRY_FIELDS, "geometry")
    orientation = _nonempty_text("geometry.orientation_model", geometry["orientation_model"])
    if basis == "synthetic_fixture":
        if any(geometry[field] is not None for field in ("latitude_deg", "longitude_deg", "elevation_m")):
            raise ValueError("synthetic_fixture coordinates/elevation must be null, not guessed")
    else:
        latitude = _finite("geometry.latitude_deg", geometry["latitude_deg"])
        longitude = _finite("geometry.longitude_deg", geometry["longitude_deg"])
        _finite("geometry.elevation_m", geometry["elevation_m"])
        if not -90.0 <= latitude <= 90.0:
            raise ValueError("geometry.latitude_deg must be in [-90, 90]")
        if not -180.0 <= longitude <= 180.0:
            raise ValueError("geometry.longitude_deg must be in [-180, 180]")
    tilt = _optional_angle("geometry.plane_tilt_deg", geometry["plane_tilt_deg"], 0.0, 180.0)
    azimuth = _optional_angle("geometry.plane_azimuth_deg", geometry["plane_azimuth_deg"], -180.0, 180.0)
    model_digest = geometry["trajectory_or_flight_config_sha256"]
    if model_digest is not None:
        _is_sha256("geometry.trajectory_or_flight_config_sha256", model_digest)

    if basis == "ground_fixed_plane":
        if orientation != "fixed_plane" or tilt is None or azimuth is None:
            raise ValueError("ground_fixed_plane requires fixed_plane orientation and explicit tilt/azimuth")
        if model_digest is not None:
            raise ValueError("ground_fixed_plane must not supply trajectory_or_flight_config_sha256")
    elif basis == "airborne_model":
        if orientation != "trajectory_modelled" or model_digest is None:
            raise ValueError("airborne_model requires trajectory_modelled orientation and a trajectory/configuration digest")
        if tilt is not None or azimuth is not None:
            raise ValueError("airborne_model must not encode a single fixed-plane tilt/azimuth")
    elif basis == "airborne_measured":
        if orientation != "flight_measured" or model_digest is None:
            raise ValueError("airborne_measured requires flight_measured orientation and a flight-configuration digest")
        if tilt is not None or azimuth is not None:
            raise ValueError("airborne_measured must not encode a single fixed-plane tilt/azimuth")
    elif basis == "synthetic_fixture":
        if orientation != "synthetic" or model_digest is not None or tilt is not None or azimuth is not None:
            raise ValueError("synthetic_fixture requires synthetic orientation and null flight/plane geometry")

    observations = manifest["observations"]
    _exact_fields(observations, _OBSERVATION_FIELDS, "observations")
    resolution = observations["resolution"]
    if not isinstance(resolution, str) or resolution not in {"monthly_average_daily", "daily_sequence"}:
        raise ValueError("observations.resolution must be monthly_average_daily or daily_sequence")
    if observations["units"] != "kWh/m2/day":
        raise ValueError("observations.units must equal kWh/m2/day")
    time_basis = observations["time_basis"]
    expected_time_basis = {
        "monthly_average_daily": "calendar_month_average",
        "daily_sequence": "UTC_calendar_day",
    }[resolution]
    if time_basis != expected_time_basis:
        raise ValueError(f"observations.time_basis must equal {expected_time_basis}")
    indexes = observations["index"]
    resource_values = observations["plane_of_array_irradiation_kwh_m2_day"]
    daylight_values = observations["daylight_hours"]
    if not isinstance(indexes, list) or not isinstance(resource_values, list) or not isinstance(daylight_values, list):
        raise ValueError("observations.index, resource values, and daylight hours must be arrays")
    if not indexes or len(indexes) != len(resource_values) or len(indexes) != len(daylight_values):
        raise ValueError("observation index, resource, and daylight arrays must have the same non-zero length")

    normalized_resources = tuple(
        _nonnegative(f"observations.resource[{idx}]", value)
        for idx, value in enumerate(resource_values)
    )
    normalized_daylight = tuple(
        _finite(f"observations.daylight_hours[{idx}]", value)
        for idx, value in enumerate(daylight_values)
    )
    if any(value < 0.0 or value > HOURS_PER_DAY for value in normalized_daylight):
        raise ValueError("observations.daylight_hours must be in [0, 24]")
    if any(hours == 0.0 and resource > 0.0 for hours, resource in zip(normalized_daylight, normalized_resources)):
        raise ValueError("positive irradiation is inconsistent with zero daylight hours")

    if resolution == "monthly_average_daily":
        expected = [f"{month:02d}" for month in range(1, 13)]
        if indexes != expected or len(resource_values) != 12:
            raise ValueError("monthly observations index must be exactly ['01', ..., '12'] in calendar order")
    else:
        parsed_dates = [_iso_date(f"observations.index[{idx}]", value) for idx, value in enumerate(indexes)]
        if any(parsed_dates[idx] >= parsed_dates[idx + 1] for idx in range(len(parsed_dates) - 1)):
            raise ValueError("daily observation dates must be strictly increasing")
        if any(parsed_dates[idx + 1] - parsed_dates[idx] != timedelta(days=1) for idx in range(len(parsed_dates) - 1)):
            raise ValueError("daily_sequence must not contain missing calendar days")
        if parsed_dates[0] != period_start or parsed_dates[-1] != period_end:
            raise ValueError("daily observation dates must match source.period_start and source.period_end")
        if len(set(parsed_dates)) != len(parsed_dates):
            raise ValueError("daily observation dates must not repeat")
        if basis == "ground_fixed_plane" and not (tilt is not None and azimuth is not None):
            raise ValueError("ground-fixed daily data requires an explicit plane orientation")


@dataclass(frozen=True)
class SourcedSeasonalSolarScreen:
    resource_profile_sha256: str
    scenario_sha256: str
    run_input_sha256: str
    implementation_sha256: str
    result_sha256: str
    screen: SeasonalSolarScreen


@dataclass(frozen=True)
class SourcedSolarReserveSimulation:
    resource_profile_sha256: str
    scenario_sha256: str
    run_input_sha256: str
    implementation_sha256: str
    result_sha256: str
    simulation: SolarReserveSimulation


def implementation_bundle_sha256() -> str:
    """Digest the exact source bytes of the equations and profile modules."""
    modules = [
        ("reference_model.py", Path(_reference_model.__file__)),
        ("solar_resource_profile.py", Path(__file__)),
    ]
    digest = hashlib.sha256()
    digest.update(b"solar-resource-implementation-bundle-v1\x00")
    for name, path in sorted(modules, key=lambda item: item[0]):
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise ValueError(f"cannot read implementation source {name}: {exc}") from exc
        name_bytes = name.encode("utf-8")
        digest.update(len(name_bytes).to_bytes(8, "big"))
        digest.update(name_bytes)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _stable_mapping_sha256(value: dict) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _run_input_id(resource_profile_sha256: str, scenario_sha256: str) -> str:
    return _stable_mapping_sha256({
        "schema_version": "solar-resource-run-input-v1",
        "resource_profile_sha256": resource_profile_sha256,
        "scenario_sha256": scenario_sha256,
    })


def _result_id(run_input_sha256: str, implementation_sha256: str, output_payload: dict) -> str:
    return _stable_mapping_sha256({
        "schema_version": "solar-resource-run-result-v1",
        "run_input_sha256": run_input_sha256,
        "implementation_sha256": implementation_sha256,
        "output": asdict(output_payload) if hasattr(output_payload, "__dataclass_fields__") else output_payload,
    })


def screen_sourced_monthly_profile(
    manifest: dict,
    *,
    artifact_verification: SolarResourceArtifactVerification,
    daylight_load_w: float,
    night_load_w: float,
    panel_efficiency: float,
    system_derate: float,
    battery_round_trip_efficiency: float,
) -> SourcedSeasonalSolarScreen:
    """Run seasonal screening directly from a validated, content-addressed profile."""
    validate_solar_resource_manifest(manifest)
    profile_digest = _require_profile_verification(manifest, artifact_verification)
    if manifest["observations"]["resolution"] != "monthly_average_daily":
        raise ValueError("screen_sourced_monthly_profile requires monthly_average_daily observations")
    observations = manifest["observations"]
    screen = seasonal_solar_array_screen(
        monthly_plane_of_array_irradiation_kwh_m2_day=tuple(observations["plane_of_array_irradiation_kwh_m2_day"]),
        monthly_daylight_hours=tuple(observations["daylight_hours"]),
        daylight_load_w=daylight_load_w,
        night_load_w=night_load_w,
        panel_efficiency=panel_efficiency,
        system_derate=system_derate,
        battery_round_trip_efficiency=battery_round_trip_efficiency,
    )
    scenario = {
        "schema_version": "seasonal-solar-screen-scenario-v1",
        "daylight_load_w": float(daylight_load_w),
        "night_load_w": float(night_load_w),
        "panel_efficiency": float(panel_efficiency),
        "system_derate": float(system_derate),
        "battery_round_trip_efficiency": float(battery_round_trip_efficiency),
    }
    scenario_digest = _stable_mapping_sha256(scenario)
    run_input_digest = _run_input_id(profile_digest, scenario_digest)
    implementation_digest = implementation_bundle_sha256()
    return SourcedSeasonalSolarScreen(
        resource_profile_sha256=profile_digest,
        scenario_sha256=scenario_digest,
        run_input_sha256=run_input_digest,
        implementation_sha256=implementation_digest,
        result_sha256=_result_id(run_input_digest, implementation_digest, screen),
        screen=screen,
    )


def simulate_sourced_daily_profile(
    manifest: dict,
    *,
    artifact_verification: SolarResourceArtifactVerification,
    solar_array_area_m2: float,
    daylight_load_w: float,
    night_load_w: float,
    panel_efficiency: float,
    array_system_derate: float,
    battery_capacity_kwh: float,
    initial_battery_energy_kwh: float,
    charge_efficiency: float,
    discharge_efficiency: float,
) -> SourcedSolarReserveSimulation:
    """Run reserve replay from a validated daily profile and bind its exact digest."""
    validate_solar_resource_manifest(manifest)
    profile_digest = _require_profile_verification(manifest, artifact_verification)
    if manifest["observations"]["resolution"] != "daily_sequence":
        raise ValueError("simulate_sourced_daily_profile requires daily_sequence observations")
    observations = manifest["observations"]
    simulation = simulate_solar_battery_reserve(
        daily_plane_of_array_irradiation_kwh_m2=tuple(observations["plane_of_array_irradiation_kwh_m2_day"]),
        daily_daylight_hours=tuple(observations["daylight_hours"]),
        solar_array_area_m2=solar_array_area_m2,
        daylight_load_w=daylight_load_w,
        night_load_w=night_load_w,
        panel_efficiency=panel_efficiency,
        array_system_derate=array_system_derate,
        battery_capacity_kwh=battery_capacity_kwh,
        initial_battery_energy_kwh=initial_battery_energy_kwh,
        charge_efficiency=charge_efficiency,
        discharge_efficiency=discharge_efficiency,
    )
    scenario = {
        "schema_version": "daily-solar-reserve-scenario-v1",
        "solar_array_area_m2": float(solar_array_area_m2),
        "daylight_load_w": float(daylight_load_w),
        "night_load_w": float(night_load_w),
        "panel_efficiency": float(panel_efficiency),
        "array_system_derate": float(array_system_derate),
        "battery_capacity_kwh": float(battery_capacity_kwh),
        "initial_battery_energy_kwh": float(initial_battery_energy_kwh),
        "charge_efficiency": float(charge_efficiency),
        "discharge_efficiency": float(discharge_efficiency),
    }
    scenario_digest = _stable_mapping_sha256(scenario)
    run_input_digest = _run_input_id(profile_digest, scenario_digest)
    implementation_digest = implementation_bundle_sha256()
    return SourcedSolarReserveSimulation(
        resource_profile_sha256=profile_digest,
        scenario_sha256=scenario_digest,
        run_input_sha256=run_input_digest,
        implementation_sha256=implementation_digest,
        result_sha256=_result_id(run_input_digest, implementation_digest, simulation),
        simulation=simulation,
    )
