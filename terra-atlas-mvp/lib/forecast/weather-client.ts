// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * Open-Meteo Weather Client
 *
 * Fetches weather forecast data from Open-Meteo (free, no API key required).
 * Provides solar irradiance, temperature, wind speed, and cloud cover
 * at hourly resolution for up to 7 days ahead.
 *
 * API docs: https://open-meteo.com/en/docs
 */

export interface WeatherForecast {
  latitude: number
  longitude: number
  timezone: string
  hourly: HourlyWeather[]
  source: string
}

export interface HourlyWeather {
  timestamp: string           // ISO 8601
  shortwave_radiation_wm2: number  // Global Horizontal Irradiance (W/m²)
  temperature_c: number       // 2m temperature (°C)
  windspeed_ms: number        // 10m wind speed (m/s)
  windspeed_100m_ms: number   // 100m wind speed (m/s) — hub height
  wind_direction_deg: number  // Wind direction (degrees)
  cloud_cover_pct: number     // Cloud cover (%)
  precipitation_mm: number    // Precipitation (mm)
}

const OPEN_METEO_BASE = 'https://api.open-meteo.com/v1/forecast'

export async function fetchWeatherForecast(
  latitude: number,
  longitude: number,
  forecastDays: number = 7
): Promise<WeatherForecast> {
  const params = new URLSearchParams({
    latitude: latitude.toString(),
    longitude: longitude.toString(),
    hourly: [
      'shortwave_radiation',
      'temperature_2m',
      'windspeed_10m',
      'windspeed_100m',
      'winddirection_10m',
      'cloudcover',
      'precipitation',
    ].join(','),
    forecast_days: Math.min(forecastDays, 16).toString(),
    timezone: 'auto',
  })

  const response = await fetch(`${OPEN_METEO_BASE}?${params}`)

  if (!response.ok) {
    throw new Error(`Open-Meteo API error: ${response.status} ${response.statusText}`)
  }

  const data = await response.json()

  const hourly: HourlyWeather[] = []
  const times = data.hourly?.time || []

  for (let i = 0; i < times.length; i++) {
    hourly.push({
      timestamp: times[i],
      shortwave_radiation_wm2: data.hourly.shortwave_radiation?.[i] ?? 0,
      temperature_c: data.hourly.temperature_2m?.[i] ?? 25,
      windspeed_ms: data.hourly.windspeed_10m?.[i] ?? 0,
      windspeed_100m_ms: data.hourly.windspeed_100m?.[i] ?? 0,
      wind_direction_deg: data.hourly.winddirection_10m?.[i] ?? 0,
      cloud_cover_pct: data.hourly.cloudcover?.[i] ?? 0,
      precipitation_mm: data.hourly.precipitation?.[i] ?? 0,
    })
  }

  return {
    latitude: data.latitude,
    longitude: data.longitude,
    timezone: data.timezone,
    hourly,
    source: 'Open-Meteo (free, no API key)',
  }
}
