"""Deterministic, dependency-free screening equations for persistent platforms.

This module is an arithmetic reference model only. It is not an aircraft,
airship, pressure-hull, airworthiness, structural-certification, or habitat
safety model. Inputs use SI units unless a name explicitly says kWh.
"""
from __future__ import annotations

from dataclasses import dataclass
import math


HOURS_PER_DAY = 24.0


def _finite(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value


def _nonnegative(name: str, value: float) -> float:
    value = _finite(name, value)
    if value < 0.0:
        raise ValueError(f"{name} must be >= 0")
    return value


def _positive(name: str, value: float) -> float:
    value = _finite(name, value)
    if value <= 0.0:
        raise ValueError(f"{name} must be > 0")
    return value


def _fraction(name: str, value: float, *, allow_zero: bool = False) -> float:
    value = _finite(name, value)
    lower_ok = value >= 0.0 if allow_zero else value > 0.0
    if not lower_ok or value > 1.0:
        left = "[0, 1]" if allow_zero else "(0, 1]"
        raise ValueError(f"{name} must be in {left}")
    return value


@dataclass(frozen=True)
class SolarArrayScreen:
    day_energy_kwh: float
    night_energy_kwh: float
    required_solar_generation_kwh: float
    required_array_area_m2: float


def solar_array_screen(
    *,
    daylight_load_w: float,
    night_load_w: float,
    daylight_hours: float,
    night_hours: float,
    average_daylight_irradiance_w_m2: float,
    panel_efficiency: float,
    system_derate: float,
    battery_round_trip_efficiency: float,
) -> SolarArrayScreen:
    """Screen daily solar generation and array area for a declared 24 h day.

    Irradiance is the average plane-of-array irradiance during daylight, not
    peak irradiance. system_derate covers only explicitly declared losses
    outside cell conversion and battery round-trip losses.
    """
    daylight_load_w = _nonnegative("daylight_load_w", daylight_load_w)
    night_load_w = _nonnegative("night_load_w", night_load_w)
    daylight_hours = _nonnegative("daylight_hours", daylight_hours)
    night_hours = _nonnegative("night_hours", night_hours)
    irradiance = _positive("average_daylight_irradiance_w_m2", average_daylight_irradiance_w_m2)
    panel_efficiency = _fraction("panel_efficiency", panel_efficiency)
    system_derate = _fraction("system_derate", system_derate)
    battery_efficiency = _fraction("battery_round_trip_efficiency", battery_round_trip_efficiency)

    if not math.isclose(daylight_hours + night_hours, HOURS_PER_DAY, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("daylight_hours + night_hours must equal 24")
    if daylight_hours <= 0.0:
        raise ValueError("daylight_hours must be > 0 for a solar-only daily-energy screen")

    day_kwh = _finite("computed_day_energy_kwh", daylight_load_w * daylight_hours / 1000.0)
    night_kwh = _finite("computed_night_energy_kwh", night_load_w * night_hours / 1000.0)
    required_generation_kwh = _finite("computed_required_generation_kwh", day_kwh + night_kwh / battery_efficiency)
    generation_kwh_per_m2 = _positive("computed_generation_kwh_per_m2", irradiance * daylight_hours * panel_efficiency * system_derate / 1000.0)
    required_area_m2 = _finite("computed_required_array_area_m2", required_generation_kwh / generation_kwh_per_m2)

    return SolarArrayScreen(
        day_energy_kwh=day_kwh,
        night_energy_kwh=night_kwh,
        required_solar_generation_kwh=required_generation_kwh,
        required_array_area_m2=required_area_m2,
    )


def battery_nominal_capacity_kwh(
    *,
    night_load_w: float,
    night_hours: float,
    reserve_nights: float,
    usable_depth_of_discharge: float,
    discharge_efficiency: float,
    end_of_life_capacity_fraction: float,
    cold_capacity_fraction: float,
) -> float:
    """Return nominal battery kWh for a declared night/reserve interval.

    Capacity retention and cold derating are independent multipliers supplied
    by the caller; do not count the same degradation factor twice.
    """
    night_load_w = _nonnegative("night_load_w", night_load_w)
    night_hours = _nonnegative("night_hours", night_hours)
    reserve_nights = _positive("reserve_nights", reserve_nights)
    dod = _fraction("usable_depth_of_discharge", usable_depth_of_discharge)
    discharge_eff = _fraction("discharge_efficiency", discharge_efficiency)
    eol = _fraction("end_of_life_capacity_fraction", end_of_life_capacity_fraction)
    cold = _fraction("cold_capacity_fraction", cold_capacity_fraction)
    night_energy_kwh = _finite("computed_night_energy_kwh", night_load_w * night_hours / 1000.0)
    result = night_energy_kwh * reserve_nights / (dod * discharge_eff * eol * cold)
    return _finite("computed_battery_capacity_kwh", result)


@dataclass(frozen=True)
class BuoyancyMassScreen:
    gross_lift_equivalent_kg: float
    fixed_non_payload_mass_kg: float
    payload_capacity_before_margin_kg: float
    payload_margin_kg: float


def buoyancy_mass_screen(
    *,
    volume_m3: float,
    ambient_air_density_kg_m3: float,
    lifting_gas_density_kg_m3: float,
    envelope_mass_kg: float,
    frame_mass_kg: float,
    propulsion_mass_kg: float,
    energy_storage_mass_kg: float,
    payload_mass_kg: float,
) -> BuoyancyMassScreen:
    """Compute a first-order buoyancy mass budget, not shell feasibility.

    Densities must be for matching temperature and pressure. Set gas density
    to 0 only for idealized vacuum buoyancy arithmetic. Structural mass still
    has to be supplied explicitly; no buckling or strength is inferred.
    """
    volume = _positive("volume_m3", volume_m3)
    air_density = _positive("ambient_air_density_kg_m3", ambient_air_density_kg_m3)
    gas_density = _nonnegative("lifting_gas_density_kg_m3", lifting_gas_density_kg_m3)
    masses = {
        "envelope_mass_kg": _nonnegative("envelope_mass_kg", envelope_mass_kg),
        "frame_mass_kg": _nonnegative("frame_mass_kg", frame_mass_kg),
        "propulsion_mass_kg": _nonnegative("propulsion_mass_kg", propulsion_mass_kg),
        "energy_storage_mass_kg": _nonnegative("energy_storage_mass_kg", energy_storage_mass_kg),
        "payload_mass_kg": _nonnegative("payload_mass_kg", payload_mass_kg),
    }
    gross_lift = _finite("computed_gross_lift_equivalent_kg", (air_density - gas_density) * volume)
    fixed_mass = _finite("computed_fixed_non_payload_mass_kg", sum(masses[name] for name in (
        "envelope_mass_kg", "frame_mass_kg", "propulsion_mass_kg", "energy_storage_mass_kg"
    )))
    payload_capacity = _finite("computed_payload_capacity_kg", gross_lift - fixed_mass)
    payload_margin = _finite("computed_payload_margin_kg", payload_capacity - masses["payload_mass_kg"])
    return BuoyancyMassScreen(
        gross_lift_equivalent_kg=gross_lift,
        fixed_non_payload_mass_kg=fixed_mass,
        payload_capacity_before_margin_kg=payload_capacity,
        payload_margin_kg=payload_margin,
    )


def hydrostatic_absolute_pressure_pa(
    *,
    depth_m: float,
    water_density_kg_m3: float = 1025.0,
    gravitational_acceleration_m_s2: float = 9.80665,
    surface_pressure_pa: float = 101325.0,
) -> float:
    """Compute absolute pressure from depth using a constant-density model."""
    depth = _nonnegative("depth_m", depth_m)
    density = _positive("water_density_kg_m3", water_density_kg_m3)
    gravity = _positive("gravitational_acceleration_m_s2", gravitational_acceleration_m_s2)
    surface = _positive("surface_pressure_pa", surface_pressure_pa)
    return _finite("computed_hydrostatic_absolute_pressure_pa", surface + density * gravity * depth)


def buckling_reserve_ratio(
    *,
    externally_validated_critical_pressure_pa: float,
    required_differential_pressure_pa: float,
    safety_factor: float,
) -> float:
    """Compare external pressure demand with an externally established capacity.

    This function does not calculate critical pressure. The caller must bind
    the supplied capacity to the exact geometry, material, imperfections,
    boundary conditions, failure mode, temperature, and validation method.
    A ratio >= 1 is arithmetic conformity to those supplied inputs only.
    """
    critical = _positive("externally_validated_critical_pressure_pa", externally_validated_critical_pressure_pa)
    required = _positive("required_differential_pressure_pa", required_differential_pressure_pa)
    factor = _positive("safety_factor", safety_factor)
    if factor < 1.0:
        raise ValueError("safety_factor must be >= 1")
    demand_with_factor = _positive("computed_factored_pressure_demand_pa", required * factor)
    return _finite("computed_buckling_reserve_ratio", critical / demand_with_factor)


@dataclass(frozen=True)
class MonthlySolarRequirement:
    """Daily-energy-neutral array estimate for one monthly resource average."""
    month: int
    daylight_hours: float
    night_hours: float
    plane_of_array_irradiation_kwh_m2_day: float
    required_array_area_m2: float | None


@dataclass(frozen=True)
class SeasonalSolarScreen:
    """Twelve monthly screening results, never a flight-resource qualification."""
    months: tuple[MonthlySolarRequirement, ...]
    limiting_month: int | None
    minimum_area_for_all_monthly_averages_m2: float | None


def seasonal_solar_array_screen(
    *,
    monthly_plane_of_array_irradiation_kwh_m2_day: tuple[float, ...],
    monthly_daylight_hours: tuple[float, ...],
    daylight_load_w: float,
    night_load_w: float,
    panel_efficiency: float,
    system_derate: float,
    battery_round_trip_efficiency: float,
) -> SeasonalSolarScreen:
    """Estimate an array against 12 monthly averages, including day-length change.

    Resource values are daily plane-of-array irradiation in kWh/m²/day, not
    instantaneous irradiance in W/m². Daylight hours vary by month; night hours
    are derived as 24 - daylight hours. The result is the area that balances
    the declared energy budget on every month's *average* day. A zero-resource
    month returns no finite daily-energy-neutral solar-only area.

    This average-day screen cannot size reserve for consecutive cloudy days;
    use simulate_solar_battery_reserve with a chronological daily series.
    For an aircraft, resources and day length must represent the flight
    trajectory/attitude, not be assumed from a ground-mounted PV dataset.
    """
    if not isinstance(monthly_plane_of_array_irradiation_kwh_m2_day, (tuple, list)):
        raise ValueError("monthly_plane_of_array_irradiation_kwh_m2_day must be a 12-item tuple or list")
    if not isinstance(monthly_daylight_hours, (tuple, list)):
        raise ValueError("monthly_daylight_hours must be a 12-item tuple or list")
    if len(monthly_plane_of_array_irradiation_kwh_m2_day) != 12:
        raise ValueError("monthly_plane_of_array_irradiation_kwh_m2_day must contain 12 values")
    if len(monthly_daylight_hours) != 12:
        raise ValueError("monthly_daylight_hours must contain 12 values")

    irradiation = tuple(
        _nonnegative(f"monthly_plane_of_array_irradiation_kwh_m2_day[{idx}]", value)
        for idx, value in enumerate(monthly_plane_of_array_irradiation_kwh_m2_day)
    )
    daylight = tuple(
        _finite(f"monthly_daylight_hours[{idx}]", value)
        for idx, value in enumerate(monthly_daylight_hours)
    )
    if any(hours < 0.0 or hours > HOURS_PER_DAY for hours in daylight):
        raise ValueError("monthly_daylight_hours values must be in [0, 24]")
    if any(hours == 0.0 and resource > 0.0 for hours, resource in zip(daylight, irradiation)):
        raise ValueError("positive solar irradiation is inconsistent with zero daylight hours")

    daylight_load_w = _nonnegative("daylight_load_w", daylight_load_w)
    night_load_w = _nonnegative("night_load_w", night_load_w)
    panel_efficiency = _fraction("panel_efficiency", panel_efficiency)
    system_derate = _fraction("system_derate", system_derate)
    battery_efficiency = _fraction("battery_round_trip_efficiency", battery_round_trip_efficiency)
    conversion = panel_efficiency * system_derate

    month_results = []
    for month, (resource, daylight_hours) in enumerate(zip(irradiation, daylight), start=1):
        night_hours = HOURS_PER_DAY - daylight_hours
        day_kwh = _finite("computed_monthly_day_energy_kwh", daylight_load_w * daylight_hours / 1000.0)
        night_kwh = _finite("computed_monthly_night_energy_kwh", night_load_w * night_hours / 1000.0)
        required_generation_kwh = _finite("computed_monthly_required_generation_kwh", day_kwh + night_kwh / battery_efficiency)
        resource_conversion = _positive("computed_monthly_resource_conversion", resource * conversion) if resource > 0.0 and daylight_hours > 0.0 else 0.0
        if required_generation_kwh == 0.0:
            area = 0.0
        elif resource == 0.0 or daylight_hours == 0.0:
            area = None
        else:
            area = _finite("computed_monthly_array_area_m2", required_generation_kwh / resource_conversion)
        month_results.append(MonthlySolarRequirement(
            month=month,
            daylight_hours=daylight_hours,
            night_hours=night_hours,
            plane_of_array_irradiation_kwh_m2_day=resource,
            required_array_area_m2=area,
        ))

    infeasible_months = [item.month for item in month_results if item.required_array_area_m2 is None]
    if infeasible_months:
        limiting_month = infeasible_months[0]
        minimum_area = None
    else:
        limiting = max(month_results, key=lambda item: item.required_array_area_m2 or 0.0)
        limiting_month = limiting.month
        minimum_area = limiting.required_array_area_m2

    return SeasonalSolarScreen(
        months=tuple(month_results),
        limiting_month=limiting_month,
        minimum_area_for_all_monthly_averages_m2=minimum_area,
    )


@dataclass(frozen=True)
class DailySolarBalance:
    """Energy ledger for one daily time bucket; quantities are modelled energy."""
    day_index: int
    daylight_hours: float
    night_hours: float
    plane_of_array_irradiation_kwh_m2: float
    solar_generation_kwh: float
    battery_start_kwh: float
    daytime_unmet_load_kwh: float
    nighttime_unmet_load_kwh: float
    battery_end_kwh: float
    curtailed_solar_kwh: float


@dataclass(frozen=True)
class SolarReserveSimulation:
    """Chronological daily energy ledger, not a validated persistence claim."""
    days: tuple[DailySolarBalance, ...]
    total_unmet_load_kwh: float
    total_curtailed_solar_kwh: float
    final_battery_energy_kwh: float


def simulate_solar_battery_reserve(
    *,
    daily_plane_of_array_irradiation_kwh_m2: tuple[float, ...],
    daily_daylight_hours: tuple[float, ...],
    solar_array_area_m2: float,
    daylight_load_w: float,
    night_load_w: float,
    panel_efficiency: float,
    array_system_derate: float,
    battery_capacity_kwh: float,
    initial_battery_energy_kwh: float,
    charge_efficiency: float,
    discharge_efficiency: float,
) -> SolarReserveSimulation:
    """Simulate day/night energy across a chronological, day-length-aware profile.

    The irradiation and daylight-hour sequences must have the same non-zero
    length and matching order. Battery capacity/state are internal stored-energy
    kWh after caller-declared capacity deratings. Solar serves each day's
    declared daytime load first; surplus charges storage, and any deficit is
    served from storage subject to discharge efficiency. The night load is
    served from remaining storage.

    This daily-bucket approximation cannot model intraday cloud transients,
    flight attitude, thermal behavior, cell voltage, C-rate limits, battery
    chemistry, or flight dynamics. Zero unmet energy means only that this
    supplied arithmetic profile was served under supplied assumptions.
    """
    if not isinstance(daily_plane_of_array_irradiation_kwh_m2, (tuple, list)):
        raise ValueError("daily_plane_of_array_irradiation_kwh_m2 must be a non-empty tuple or list")
    if not daily_plane_of_array_irradiation_kwh_m2:
        raise ValueError("daily_plane_of_array_irradiation_kwh_m2 must be non-empty")
    if not isinstance(daily_daylight_hours, (tuple, list)):
        raise ValueError("daily_daylight_hours must be a non-empty tuple or list")
    if len(daily_daylight_hours) != len(daily_plane_of_array_irradiation_kwh_m2):
        raise ValueError("daily_daylight_hours length must match the irradiation profile")

    irradiation = tuple(
        _nonnegative(f"daily_plane_of_array_irradiation_kwh_m2[{idx}]", value)
        for idx, value in enumerate(daily_plane_of_array_irradiation_kwh_m2)
    )
    daylight = tuple(
        _finite(f"daily_daylight_hours[{idx}]", value)
        for idx, value in enumerate(daily_daylight_hours)
    )
    if any(hours < 0.0 or hours > HOURS_PER_DAY for hours in daylight):
        raise ValueError("daily_daylight_hours values must be in [0, 24]")
    if any(hours == 0.0 and resource > 0.0 for hours, resource in zip(daylight, irradiation)):
        raise ValueError("positive solar irradiation is inconsistent with zero daylight hours")

    area = _nonnegative("solar_array_area_m2", solar_array_area_m2)
    day_load_w = _nonnegative("daylight_load_w", daylight_load_w)
    night_load_w = _nonnegative("night_load_w", night_load_w)
    panel_efficiency = _fraction("panel_efficiency", panel_efficiency)
    array_derate = _fraction("array_system_derate", array_system_derate)
    battery_capacity = _nonnegative("battery_capacity_kwh", battery_capacity_kwh)
    initial_energy = _nonnegative("initial_battery_energy_kwh", initial_battery_energy_kwh)
    charge_eff = _fraction("charge_efficiency", charge_efficiency)
    discharge_eff = _fraction("discharge_efficiency", discharge_efficiency)

    if initial_energy > battery_capacity:
        raise ValueError("initial_battery_energy_kwh must not exceed battery_capacity_kwh")

    state = initial_energy
    daily_results = []
    for day_index, (resource, daylight_hours) in enumerate(zip(irradiation, daylight), start=1):
        night_hours = HOURS_PER_DAY - daylight_hours
        day_load_kwh = _finite("computed_day_load_kwh", day_load_w * daylight_hours / 1000.0)
        night_load_kwh = _finite("computed_night_load_kwh", night_load_w * night_hours / 1000.0)
        start_state = state
        generated = _finite("computed_solar_generation_kwh", area * resource * panel_efficiency * array_derate)
        daytime_unmet = 0.0
        nighttime_unmet = 0.0
        curtailed = 0.0

        if generated >= day_load_kwh:
            surplus = generated - day_load_kwh
            requested_storage = surplus * charge_eff
            accepted_storage = min(battery_capacity - state, requested_storage)
            state += accepted_storage
            curtailed = max(0.0, surplus - accepted_storage / charge_eff)
        else:
            deficit = day_load_kwh - generated
            available_delivery = state * discharge_eff
            delivered = min(deficit, available_delivery)
            state -= delivered / discharge_eff
            daytime_unmet = max(0.0, deficit - delivered)

        night_available_delivery = state * discharge_eff
        night_delivered = min(night_load_kwh, night_available_delivery)
        state -= night_delivered / discharge_eff
        nighttime_unmet = max(0.0, night_load_kwh - night_delivered)

        if abs(state) < 1e-12:
            state = 0.0
        if abs(state - battery_capacity) < 1e-12:
            state = battery_capacity

        daily_results.append(DailySolarBalance(
            day_index=day_index,
            daylight_hours=daylight_hours,
            night_hours=night_hours,
            plane_of_array_irradiation_kwh_m2=resource,
            solar_generation_kwh=generated,
            battery_start_kwh=start_state,
            daytime_unmet_load_kwh=daytime_unmet,
            nighttime_unmet_load_kwh=nighttime_unmet,
            battery_end_kwh=state,
            curtailed_solar_kwh=curtailed,
        ))

    return SolarReserveSimulation(
        days=tuple(daily_results),
        total_unmet_load_kwh=_finite(
            "computed_total_unmet_load_kwh",
            sum(day.daytime_unmet_load_kwh + day.nighttime_unmet_load_kwh for day in daily_results),
        ),
        total_curtailed_solar_kwh=_finite(
            "computed_total_curtailed_solar_kwh",
            sum(day.curtailed_solar_kwh for day in daily_results),
        ),
        final_battery_energy_kwh=_finite("computed_final_battery_energy_kwh", state),
    )
