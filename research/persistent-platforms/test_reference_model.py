"""Deterministic tests for persistent-platform screening equations."""
import math
import json
import unittest

from reference_model import (
    buoyancy_mass_screen,
    battery_nominal_capacity_kwh,
    buckling_reserve_ratio,
    hydrostatic_absolute_pressure_pa,
    solar_array_screen,
    seasonal_solar_array_screen,
    simulate_solar_battery_reserve,
)
from solar_resource_profile import (
    canonical_solar_resource_json,
    load_solar_resource_manifest,
    screen_sourced_monthly_profile,
    sha256_bytes,
    simulate_sourced_daily_profile,
    solar_resource_profile_sha256,
    validate_solar_resource_manifest,
    verify_solar_resource_raw_artifact,
    verify_solar_resource_artifact,
    verify_solar_resource_profile_artifacts,
)


class SolarArrayScreenTests(unittest.TestCase):
    def test_hand_calculated_daily_energy_and_area(self):
        result = solar_array_screen(
            daylight_load_w=200,
            night_load_w=100,
            daylight_hours=12,
            night_hours=12,
            average_daylight_irradiance_w_m2=550,
            panel_efficiency=0.25,
            system_derate=0.8,
            battery_round_trip_efficiency=0.9,
        )
        self.assertAlmostEqual(result.day_energy_kwh, 2.4)
        self.assertAlmostEqual(result.night_energy_kwh, 1.2)
        self.assertAlmostEqual(result.required_solar_generation_kwh, 2.4 + 1.2 / 0.9)
        self.assertAlmostEqual(result.required_array_area_m2, (2.4 + 1.2 / 0.9) / 1.32)

    def test_rejects_incoherent_day_and_night(self):
        with self.assertRaisesRegex(ValueError, "equal 24"):
            solar_array_screen(
                daylight_load_w=1, night_load_w=1, daylight_hours=10, night_hours=10,
                average_daylight_irradiance_w_m2=500, panel_efficiency=0.2,
                system_derate=0.8, battery_round_trip_efficiency=0.9,
            )

    def test_rejects_nonfinite_and_impossible_efficiency(self):
        args = dict(
            daylight_load_w=1, night_load_w=1, daylight_hours=12, night_hours=12,
            average_daylight_irradiance_w_m2=500, panel_efficiency=0.2,
            system_derate=0.8, battery_round_trip_efficiency=0.9,
        )
        with self.assertRaisesRegex(ValueError, "finite"):
            solar_array_screen(**{**args, "daylight_load_w": math.nan})
        with self.assertRaisesRegex(ValueError, "panel_efficiency"):
            solar_array_screen(**{**args, "panel_efficiency": 1.1})
        with self.assertRaisesRegex(ValueError, "daylight_load_w"):
            solar_array_screen(**{**args, "daylight_load_w": -1})


class BatteryTests(unittest.TestCase):
    def test_nominal_capacity_includes_declared_deratings(self):
        result = battery_nominal_capacity_kwh(
            night_load_w=100, night_hours=10, reserve_nights=2,
            usable_depth_of_discharge=0.8, discharge_efficiency=0.9,
            end_of_life_capacity_fraction=0.8, cold_capacity_fraction=0.75,
        )
        self.assertAlmostEqual(result, 2.0 / (0.8 * 0.9 * 0.8 * 0.75))

    def test_rejects_zero_efficiency(self):
        with self.assertRaisesRegex(ValueError, "discharge_efficiency"):
            battery_nominal_capacity_kwh(
                night_load_w=100, night_hours=10, reserve_nights=1,
                usable_depth_of_discharge=0.8, discharge_efficiency=0,
                end_of_life_capacity_fraction=0.8, cold_capacity_fraction=0.75,
            )


class BuoyancyTests(unittest.TestCase):
    def test_vacuum_is_only_a_gross_lift_and_mass_budget(self):
        result = buoyancy_mass_screen(
            volume_m3=1000, ambient_air_density_kg_m3=1.225,
            lifting_gas_density_kg_m3=0, envelope_mass_kg=700,
            frame_mass_kg=100, propulsion_mass_kg=50,
            energy_storage_mass_kg=0, payload_mass_kg=300,
        )
        self.assertAlmostEqual(result.gross_lift_equivalent_kg, 1225)
        self.assertAlmostEqual(result.fixed_non_payload_mass_kg, 850)
        self.assertAlmostEqual(result.payload_capacity_before_margin_kg, 375)
        self.assertAlmostEqual(result.payload_margin_kg, 75)

    def test_negative_payload_margin_is_retained_as_failure_signal(self):
        result = buoyancy_mass_screen(
            volume_m3=100, ambient_air_density_kg_m3=1.225,
            lifting_gas_density_kg_m3=0.085, envelope_mass_kg=80,
            frame_mass_kg=40, propulsion_mass_kg=20,
            energy_storage_mass_kg=10, payload_mass_kg=10,
        )
        self.assertLess(result.payload_margin_kg, 0)


class HydrostaticTests(unittest.TestCase):
    def test_pressure_at_zero_depth_equals_surface_pressure(self):
        self.assertEqual(hydrostatic_absolute_pressure_pa(depth_m=0), 101325.0)

    def test_pressure_increases_linearly_with_depth_in_reference_model(self):
        p0 = hydrostatic_absolute_pressure_pa(depth_m=0)
        p10 = hydrostatic_absolute_pressure_pa(depth_m=10)
        self.assertAlmostEqual(p10 - p0, 1025 * 9.80665 * 10)

    def test_buckling_ratio_does_not_infer_capacity(self):
        self.assertAlmostEqual(
            buckling_reserve_ratio(
                externally_validated_critical_pressure_pa=300_000,
                required_differential_pressure_pa=100_000,
                safety_factor=1.5,
            ),
            2.0,
        )
        with self.assertRaisesRegex(ValueError, "safety_factor"):
            buckling_reserve_ratio(
                externally_validated_critical_pressure_pa=300_000,
                required_differential_pressure_pa=100_000,
                safety_factor=0.9,
            )



class SeasonalSolarTests(unittest.TestCase):
    def _screen(self, profile):
        return seasonal_solar_array_screen(
            monthly_plane_of_array_irradiation_kwh_m2_day=tuple(profile),
            monthly_daylight_hours=(8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 15.0, 13.0, 10.0),
            daylight_load_w=200,
            night_load_w=100,
            panel_efficiency=0.25,
            system_derate=0.8,
            battery_round_trip_efficiency=0.9,
        )

    def test_worst_resource_month_sets_daily_energy_neutral_area(self):
        profile = [4.0] * 12
        profile[0] = 2.5
        result = self._screen(profile)
        self.assertEqual(len(result.months), 12)
        self.assertEqual(result.limiting_month, 1)
        self.assertEqual(result.months[0].daylight_hours, 8.0)
        self.assertEqual(result.months[0].night_hours, 16.0)
        self.assertAlmostEqual(result.minimum_area_for_all_monthly_averages_m2, (1.6 + 1.6 / 0.9) / (2.5 * 0.25 * 0.8))
        self.assertGreater(result.months[0].required_array_area_m2, result.months[1].required_array_area_m2)

    def test_zero_resource_with_zero_load_requires_zero_area_not_infeasible(self):
        result = seasonal_solar_array_screen(
            monthly_plane_of_array_irradiation_kwh_m2_day=(0.0,) * 12,
            monthly_daylight_hours=(0.0,) * 12,
            daylight_load_w=0.0,
            night_load_w=0.0,
            panel_efficiency=0.2,
            system_derate=0.8,
            battery_round_trip_efficiency=0.9,
        )
        self.assertEqual(result.minimum_area_for_all_monthly_averages_m2, 0.0)
        self.assertEqual(result.limiting_month, 1)

    def test_zero_resource_month_is_not_hidden_or_given_a_finite_area(self):
        profile = [3.0] * 12
        profile[5] = 0.0
        result = self._screen(profile)
        self.assertEqual(result.limiting_month, 6)
        self.assertIsNone(result.minimum_area_for_all_monthly_averages_m2)
        self.assertIsNone(result.months[5].required_array_area_m2)

    def test_rejects_wrong_month_count_and_negative_resource(self):
        with self.assertRaisesRegex(ValueError, "12 values"):
            self._screen([3.0] * 11)
        profile = [3.0] * 12
        profile[2] = -0.1
        with self.assertRaisesRegex(ValueError, "monthly_plane_of_array"):
            self._screen(profile)


class SolarReserveSimulationTests(unittest.TestCase):
    def _simulate(self, *, capacity):
        return simulate_solar_battery_reserve(
            daily_plane_of_array_irradiation_kwh_m2=(3.0, 0.3, 0.3, 3.0),
            daily_daylight_hours=(12.0, 12.0, 12.0, 12.0),
            solar_array_area_m2=10.0,
            daylight_load_w=100,
            night_load_w=100,
            panel_efficiency=0.2,
            array_system_derate=1.0,
            battery_capacity_kwh=capacity,
            initial_battery_energy_kwh=0.0,
            charge_efficiency=1.0,
            discharge_efficiency=1.0,
        )

    def test_declared_battery_survives_two_consecutive_low_solar_days(self):
        result = self._simulate(capacity=4.8)
        self.assertEqual(len(result.days), 4)
        self.assertAlmostEqual(result.total_unmet_load_kwh, 0.0)
        self.assertAlmostEqual(result.days[1].plane_of_array_irradiation_kwh_m2, 0.3)
        self.assertAlmostEqual(result.days[2].plane_of_array_irradiation_kwh_m2, 0.3)
        self.assertAlmostEqual(result.days[2].battery_end_kwh, 0.0)
        self.assertAlmostEqual(result.final_battery_energy_kwh, 3.6)

    def test_insufficient_reserve_preserves_unmet_load_as_a_failure_signal(self):
        result = self._simulate(capacity=3.0)
        self.assertAlmostEqual(result.total_unmet_load_kwh, 1.8)
        self.assertGreater(result.days[2].daytime_unmet_load_kwh, 0.0)
        self.assertGreater(result.days[2].nighttime_unmet_load_kwh, 0.0)

    def test_surplus_is_curtailed_when_storage_is_full(self):
        result = simulate_solar_battery_reserve(
            daily_plane_of_array_irradiation_kwh_m2=(3.0,),
            daily_daylight_hours=(12.0,),
            solar_array_area_m2=10.0,
            daylight_load_w=100,
            night_load_w=80,
            panel_efficiency=0.2,
            array_system_derate=1.0,
            battery_capacity_kwh=1.0,
            initial_battery_energy_kwh=1.0,
            charge_efficiency=1.0,
            discharge_efficiency=1.0,
        )
        self.assertAlmostEqual(result.total_curtailed_solar_kwh, 4.8)
        self.assertAlmostEqual(result.total_unmet_load_kwh, 0.0)

    def test_rejects_state_above_capacity_and_nonfinite_resource(self):
        with self.assertRaisesRegex(ValueError, "must not exceed"):
            simulate_solar_battery_reserve(
                daily_plane_of_array_irradiation_kwh_m2=(1.0,),
                daily_daylight_hours=(12.0,),
                solar_array_area_m2=1, daylight_load_w=0, night_load_w=0,
                panel_efficiency=0.2,
                array_system_derate=1, battery_capacity_kwh=1,
                initial_battery_energy_kwh=2, charge_efficiency=1,
                discharge_efficiency=1,
            )
        with self.assertRaisesRegex(ValueError, "length must match"):
            simulate_solar_battery_reserve(
                daily_plane_of_array_irradiation_kwh_m2=(1.0, 1.0),
                daily_daylight_hours=(12.0,),
                solar_array_area_m2=1, daylight_load_w=0, night_load_w=0,
                panel_efficiency=0.2, array_system_derate=1,
                battery_capacity_kwh=1, initial_battery_energy_kwh=0,
                charge_efficiency=1, discharge_efficiency=1,
            )
        with self.assertRaisesRegex(ValueError, "finite"):
            simulate_solar_battery_reserve(
                daily_plane_of_array_irradiation_kwh_m2=(1000.0,),
                daily_daylight_hours=(12.0,),
                solar_array_area_m2=1e308, daylight_load_w=0, night_load_w=0,
                panel_efficiency=0.2, array_system_derate=1,
                battery_capacity_kwh=1, initial_battery_energy_kwh=0,
                charge_efficiency=1, discharge_efficiency=1,
            )
        with self.assertRaisesRegex(ValueError, "finite"):
            simulate_solar_battery_reserve(
                daily_plane_of_array_irradiation_kwh_m2=(math.inf,),
                daily_daylight_hours=(12.0,),
                solar_array_area_m2=1, daylight_load_w=0, night_load_w=0,
                panel_efficiency=0.2,
                array_system_derate=1, battery_capacity_kwh=1,
                initial_battery_energy_kwh=0, charge_efficiency=1,
                discharge_efficiency=1,
            )


def _request_manifest_bytes(manifest):
    source = manifest["source"]
    request = {
        "schema_version": "solar-resource-request-v1",
        "provider": source["provider"],
        "product": source["product"],
        "dataset_version": source["dataset_version"],
        "source_url": source["source_url"],
        "period_start": source["period_start"],
        "period_end": source["period_end"],
        "observation_resolution": manifest["observations"]["resolution"],
        "time_basis": manifest["observations"]["time_basis"],
        "geometry": manifest["geometry"],
        "request_parameters": {
            "fixture_kind": manifest["observations"]["resolution"],
            "fixture_version": "synthetic-request-v1",
        },
    }
    return json.dumps(request, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _synthetic_manifest(*, daily=False):
    raw = b"synthetic solar resource fixture v1\\n"
    source = {
        "provider": "local-test-fixture",
        "product": "synthetic-irradiance-profile-v1",
        "dataset_version": "synthetic-profile-fixture-v1",
        "source_url": "urn:luminous-dynamics:synthetic-fixture:solar-resource-v1",
        "retrieved_on": "2026-10-10",
        "period_start": "2020-01-01" if daily else "2005-01-01",
        "period_end": "2020-01-03" if daily else "2020-12-31",
        "raw_artifact_sha256": sha256_bytes(raw),
        "transform_sha256": sha256_bytes(b"synthetic-transform-v1"),
        "measurement_calibration_sha256": None,
    }
    if daily:
        observations = {
            "resolution": "daily_sequence",
            "time_basis": "UTC_calendar_day",
            "units": "kWh/m2/day",
            "index": ["2020-01-01", "2020-01-02", "2020-01-03"],
            "plane_of_array_irradiation_kwh_m2_day": [3.0, 0.3, 3.0],
            "daylight_hours": [12.0, 12.0, 12.0],
        }
    else:
        observations = {
            "resolution": "monthly_average_daily",
            "time_basis": "calendar_month_average",
            "units": "kWh/m2/day",
            "index": [f"{month:02d}" for month in range(1, 13)],
            "plane_of_array_irradiation_kwh_m2_day": [3.0] * 12,
            "daylight_hours": [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 15.0, 13.0, 10.0],
        }
    manifest = {
        "schema_version": "solar-resource-profile-v1",
        "profile_id": "synthetic-profile-test-v1",
        "resource_basis": "synthetic_fixture",
        "source": source,
        "geometry": {
            "latitude_deg": None,
            "longitude_deg": None,
            "elevation_m": None,
            "orientation_model": "synthetic",
            "plane_tilt_deg": None,
            "plane_azimuth_deg": None,
            "trajectory_or_flight_config_sha256": None,
        },
        "observations": observations,
    }
    source["request_manifest_sha256"] = sha256_bytes(_request_manifest_bytes(manifest))
    return raw, manifest


def _verify_fixture_artifacts(raw, manifest):
    return verify_solar_resource_profile_artifacts(
        manifest,
        request_manifest_bytes=_request_manifest_bytes(manifest),
        raw_artifact_bytes=raw,
        transformation_artifact_bytes=b"synthetic-transform-v1",
    )


class SolarResourceManifestTests(unittest.TestCase):
    def test_canonical_digest_is_stable_under_object_key_order(self):
        _, manifest = _synthetic_manifest()
        reordered = dict(reversed(list(manifest.items())))
        self.assertEqual(canonical_solar_resource_json(manifest), canonical_solar_resource_json(reordered))
        self.assertEqual(solar_resource_profile_sha256(manifest), solar_resource_profile_sha256(reordered))

    def test_digest_changes_when_an_observation_changes(self):
        _, manifest = _synthetic_manifest()
        original = solar_resource_profile_sha256(manifest)
        changed = json.loads(json.dumps(manifest))
        changed["observations"]["plane_of_array_irradiation_kwh_m2_day"][0] += 0.01
        self.assertNotEqual(original, solar_resource_profile_sha256(changed))

    def test_json_loader_rejects_duplicate_keys_and_nonstandard_nan(self):
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            load_solar_resource_manifest('{"schema_version":"solar-resource-profile-v1","schema_version":"solar-resource-profile-v1"}')
        with self.assertRaisesRegex(ValueError, "NaN"):
            load_solar_resource_manifest('{"value":NaN}')

    def test_verifies_exact_raw_artifact_bytes_and_rejects_mismatch(self):
        raw, manifest = _synthetic_manifest()
        self.assertEqual(verify_solar_resource_raw_artifact(manifest, raw), manifest["source"]["raw_artifact_sha256"])
        with self.assertRaisesRegex(ValueError, "does not match"):
            verify_solar_resource_raw_artifact(manifest, raw + b"changed")

    def test_request_and_transform_artifact_identity_are_independently_verifiable(self):
        _, manifest = _synthetic_manifest()
        request = _request_manifest_bytes(manifest)
        transform = b"synthetic-transform-v1"
        self.assertEqual(
            verify_solar_resource_artifact(manifest, "request_manifest", request),
            manifest["source"]["request_manifest_sha256"],
        )
        self.assertEqual(
            verify_solar_resource_artifact(manifest, "transformation", transform),
            manifest["source"]["transform_sha256"],
        )
        with self.assertRaisesRegex(ValueError, "request_manifest SHA-256"):
            verify_solar_resource_artifact(manifest, "request_manifest", request + b"-edited")
        with self.assertRaisesRegex(ValueError, "transformation SHA-256"):
            verify_solar_resource_artifact(manifest, "transformation", transform + b"-edited")

    def test_request_metadata_splice_is_rejected_even_with_matching_request_hash(self):
        raw, manifest = _synthetic_manifest()
        request = json.loads(_request_manifest_bytes(manifest).decode("utf-8"))
        request["source_url"] = "https://example.invalid/spliced-source"
        altered_request = json.dumps(
            request, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        manifest["source"]["request_manifest_sha256"] = sha256_bytes(altered_request)
        with self.assertRaisesRegex(ValueError, "source_url does not match"):
            verify_solar_resource_profile_artifacts(
                manifest,
                request_manifest_bytes=altered_request,
                raw_artifact_bytes=raw,
                transformation_artifact_bytes=b"synthetic-transform-v1",
            )

    def test_ground_fixed_profile_cannot_claim_airborne_model_basis(self):
        _, manifest = _synthetic_manifest()
        manifest["resource_basis"] = "ground_fixed_plane"
        manifest["source"]["source_url"] = "https://example.invalid/ground-fixed-fixture"
        manifest["geometry"].update({
            "latitude_deg": -26.0,
            "longitude_deg": 28.0,
            "elevation_m": 1750.0,
            "orientation_model": "fixed_plane",
            "plane_tilt_deg": 25.0,
            "plane_azimuth_deg": 0.0,
        })
        validate_solar_resource_manifest(manifest)
        manifest["resource_basis"] = "airborne_model"
        with self.assertRaisesRegex(ValueError, "trajectory_modelled"):
            validate_solar_resource_manifest(manifest)

    def test_daily_sequence_rejects_a_missing_calendar_day(self):
        _, manifest = _synthetic_manifest(daily=True)
        manifest["observations"]["index"] = ["2020-01-01", "2020-01-03", "2020-01-04"]
        manifest["source"]["period_end"] = "2020-01-04"
        with self.assertRaisesRegex(ValueError, "missing calendar days"):
            validate_solar_resource_manifest(manifest)

    def test_sourced_monthly_result_carries_profile_digest(self):
        raw, manifest = _synthetic_manifest()
        verification = _verify_fixture_artifacts(raw, manifest)
        result = screen_sourced_monthly_profile(
            manifest,
            artifact_verification=verification,
            daylight_load_w=200,
            night_load_w=100,
            panel_efficiency=0.25,
            system_derate=0.8,
            battery_round_trip_efficiency=0.9,
        )
        self.assertEqual(result.resource_profile_sha256, solar_resource_profile_sha256(manifest))
        self.assertEqual(result.screen.limiting_month, 9)
        self.assertEqual(len(result.scenario_sha256), 64)
        self.assertEqual(len(result.run_input_sha256), 64)
        self.assertEqual(len(result.implementation_sha256), 64)
        self.assertEqual(len(result.result_sha256), 64)
        repeat = screen_sourced_monthly_profile(
            manifest,
            artifact_verification=verification,
            daylight_load_w=200,
            night_load_w=100,
            panel_efficiency=0.25,
            system_derate=0.8,
            battery_round_trip_efficiency=0.9,
        )
        self.assertEqual(repeat.run_input_sha256, result.run_input_sha256)
        self.assertEqual(repeat.implementation_sha256, result.implementation_sha256)
        self.assertEqual(repeat.result_sha256, result.result_sha256)
        changed_scenario = screen_sourced_monthly_profile(
            manifest,
            artifact_verification=verification,
            daylight_load_w=201,
            night_load_w=100,
            panel_efficiency=0.25,
            system_derate=0.8,
            battery_round_trip_efficiency=0.9,
        )
        self.assertEqual(changed_scenario.resource_profile_sha256, result.resource_profile_sha256)
        self.assertNotEqual(changed_scenario.scenario_sha256, result.scenario_sha256)
        self.assertNotEqual(changed_scenario.run_input_sha256, result.run_input_sha256)
        self.assertNotEqual(changed_scenario.result_sha256, result.result_sha256)

    def test_sourced_daily_simulation_carries_profile_digest(self):
        raw, manifest = _synthetic_manifest(daily=True)
        verification = _verify_fixture_artifacts(raw, manifest)
        result = simulate_sourced_daily_profile(
            manifest,
            artifact_verification=verification,
            solar_array_area_m2=10.0,
            daylight_load_w=100,
            night_load_w=100,
            panel_efficiency=0.2,
            array_system_derate=1.0,
            battery_capacity_kwh=4.8,
            initial_battery_energy_kwh=0.0,
            charge_efficiency=1.0,
            discharge_efficiency=1.0,
        )
        self.assertEqual(result.resource_profile_sha256, solar_resource_profile_sha256(manifest))
        self.assertEqual(len(result.scenario_sha256), 64)
        self.assertEqual(len(result.run_input_sha256), 64)
        self.assertEqual(len(result.implementation_sha256), 64)
        self.assertEqual(len(result.result_sha256), 64)
        self.assertAlmostEqual(result.simulation.total_unmet_load_kwh, 0.0)

    def test_daily_profile_rejects_wrong_time_basis(self):
        _, manifest = _synthetic_manifest(daily=True)
        manifest["observations"]["time_basis"] = "local_civil_day"
        with self.assertRaisesRegex(ValueError, "UTC_calendar_day"):
            validate_solar_resource_manifest(manifest)

    def test_airborne_measured_requires_calibration_identity(self):
        _, manifest = _synthetic_manifest(daily=True)
        manifest["resource_basis"] = "airborne_measured"
        manifest["source"]["source_url"] = "https://example.invalid/airborne-measurement-fixture"
        manifest["geometry"].update({
            "latitude_deg": -26.0,
            "longitude_deg": 28.0,
            "elevation_m": 1000.0,
            "orientation_model": "flight_measured",
            "trajectory_or_flight_config_sha256": "a" * 64,
        })
        with self.assertRaisesRegex(ValueError, "measurement_calibration_sha256"):
            validate_solar_resource_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
