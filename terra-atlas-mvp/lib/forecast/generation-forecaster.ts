// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * Solar & Wind Generation Forecaster
 *
 * Physics-based generation models using real weather data from Open-Meteo.
 * No ML training required — uses established engineering models:
 *
 * Solar: PVWatts-style model (NREL)
 *   generation = capacity × (GHI / 1000) × efficiency × temp_derating × inverter_eff
 *
 * Wind: IEC power curve model
 *   generation = capacity × power_curve(wind_speed) × availability
 *
 * References:
 * - Dobos, A. (2014). PVWatts Version 5 Manual. NREL/TP-6A20-62641
 * - IEC 61400-12-1:2017. Power performance measurements of electricity producing wind turbines
 */

import { fetchWeatherForecast, type HourlyWeather } from './weather-client'

export interface GenerationForecast {
  latitude: number
  longitude: number
  project_type: string
  capacity_mw: number
  forecast_hours: number
  hourly: HourlyGeneration[]
  summary: {
    total_generation_mwh: number
    avg_capacity_factor: number
    peak_generation_mw: number
    min_generation_mw: number
    hours_at_zero: number
    avg_temperature_c: number
    avg_windspeed_ms: number
  }
  weather_source: string
}

export interface HourlyGeneration {
  timestamp: string
  generation_mw: number
  capacity_factor: number
  weather: {
    ghi_wm2: number
    temperature_c: number
    windspeed_ms: number
    cloud_cover_pct: number
  }
}

// Solar panel parameters (crystalline silicon, typical utility-scale)
const SOLAR_PANEL_EFFICIENCY = 0.20     // 20% panel efficiency
const SOLAR_INVERTER_EFFICIENCY = 0.96  // 96% inverter efficiency
const SOLAR_SYSTEM_LOSSES = 0.14        // 14% total system losses (soiling, wiring, mismatch)
const SOLAR_TEMP_COEFF = -0.004         // -0.4%/°C above 25°C (typical c-Si)
const SOLAR_NOCT = 45                   // Nominal Operating Cell Temperature (°C)

// Wind turbine parameters (generic 3 MW onshore turbine)
const WIND_CUT_IN = 3.0       // m/s — below this, no generation
const WIND_RATED = 12.0       // m/s — above this, full capacity
const WIND_CUT_OUT = 25.0     // m/s — above this, turbine shuts down
const WIND_AVAILABILITY = 0.97 // 97% availability (maintenance downtime)

/**
 * Forecast generation for a solar or wind project at a given location.
 */
export async function forecastGeneration(
  projectType: 'solar' | 'wind',
  capacityMw: number,
  latitude: number,
  longitude: number,
  horizonHours: number = 48
): Promise<GenerationForecast> {
  const forecastDays = Math.ceil(horizonHours / 24)
  const weather = await fetchWeatherForecast(latitude, longitude, forecastDays)

  const hourlyData = weather.hourly.slice(0, horizonHours)

  const hourly: HourlyGeneration[] = hourlyData.map(w => {
    const genMw = projectType === 'solar'
      ? solarGeneration(capacityMw, w)
      : windGeneration(capacityMw, w)

    return {
      timestamp: w.timestamp,
      generation_mw: Math.round(genMw * 1000) / 1000,
      capacity_factor: Math.round((genMw / capacityMw) * 1000) / 1000,
      weather: {
        ghi_wm2: w.shortwave_radiation_wm2,
        temperature_c: w.temperature_c,
        windspeed_ms: projectType === 'wind' ? w.windspeed_100m_ms : w.windspeed_ms,
        cloud_cover_pct: w.cloud_cover_pct,
      },
    }
  })

  const totalMwh = hourly.reduce((s, h) => s + h.generation_mw, 0) // each hour = 1 MWh per MW
  const avgCf = totalMwh / (hourly.length * capacityMw)
  const peakMw = Math.max(...hourly.map(h => h.generation_mw))
  const minMw = Math.min(...hourly.map(h => h.generation_mw))
  const hoursZero = hourly.filter(h => h.generation_mw < 0.001).length
  const avgTemp = hourly.reduce((s, h) => s + h.weather.temperature_c, 0) / hourly.length
  const avgWind = hourly.reduce((s, h) => s + h.weather.windspeed_ms, 0) / hourly.length

  return {
    latitude: weather.latitude,
    longitude: weather.longitude,
    project_type: projectType,
    capacity_mw: capacityMw,
    forecast_hours: hourly.length,
    hourly,
    summary: {
      total_generation_mwh: Math.round(totalMwh * 10) / 10,
      avg_capacity_factor: Math.round(avgCf * 1000) / 1000,
      peak_generation_mw: Math.round(peakMw * 10) / 10,
      min_generation_mw: Math.round(minMw * 10) / 10,
      hours_at_zero: hoursZero,
      avg_temperature_c: Math.round(avgTemp * 10) / 10,
      avg_windspeed_ms: Math.round(avgWind * 10) / 10,
    },
    weather_source: weather.source,
  }
}

/**
 * PVWatts-style solar generation model.
 *
 * Based on NREL PVWatts Version 5 methodology:
 * 1. Cell temperature from ambient + irradiance
 * 2. Temperature derating from cell temp vs STC (25°C)
 * 3. DC power from irradiance × area × efficiency
 * 4. AC power from DC × inverter efficiency × system losses
 */
function solarGeneration(capacityMw: number, weather: HourlyWeather): number {
  const ghi = weather.shortwave_radiation_wm2
  if (ghi <= 0) return 0

  // Cell temperature (simplified NOCT model)
  const cellTemp = weather.temperature_c + (SOLAR_NOCT - 20) * (ghi / 800)

  // Temperature derating (relative to 25°C STC)
  const tempDerate = 1 + SOLAR_TEMP_COEFF * (cellTemp - 25)

  // DC power (normalized to 1000 W/m² = 1 sun)
  const dcPowerFraction = (ghi / 1000) * tempDerate

  // AC power with inverter efficiency and system losses
  const acPowerMw = capacityMw * dcPowerFraction * SOLAR_INVERTER_EFFICIENCY * (1 - SOLAR_SYSTEM_LOSSES)

  return Math.max(0, Math.min(acPowerMw, capacityMw))
}

/**
 * IEC-style wind generation model.
 *
 * Uses a cubic power curve between cut-in and rated speed:
 *   P = P_rated × ((v - v_ci) / (v_rated - v_ci))³
 *
 * With cut-out shutdown above 25 m/s.
 * Uses 100m hub-height wind speed when available.
 */
function windGeneration(capacityMw: number, weather: HourlyWeather): number {
  // Use 100m wind speed (hub height) if available, else extrapolate from 10m
  const windSpeed = weather.windspeed_100m_ms > 0
    ? weather.windspeed_100m_ms
    : weather.windspeed_ms * Math.pow(100 / 10, 0.143) // power law extrapolation (α=0.143)

  if (windSpeed < WIND_CUT_IN) return 0
  if (windSpeed >= WIND_CUT_OUT) return 0 // safety shutdown

  let powerFraction: number
  if (windSpeed >= WIND_RATED) {
    powerFraction = 1.0 // rated power
  } else {
    // Cubic power curve between cut-in and rated
    powerFraction = Math.pow(
      (windSpeed - WIND_CUT_IN) / (WIND_RATED - WIND_CUT_IN),
      3
    )
  }

  return capacityMw * powerFraction * WIND_AVAILABILITY
}
