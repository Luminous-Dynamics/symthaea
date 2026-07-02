// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import { useEffect, useCallback } from 'react';
import { useContextStore, TimeOfDay, Weather } from '@/store/contextStore';

// Weather API response type (using Open-Meteo free API)
interface WeatherResponse {
  current: {
    temperature_2m: number;
    weather_code: number;
    wind_speed_10m: number;
    precipitation: number;
  };
}

// Map weather codes to our weather types
function mapWeatherCode(code: number): Weather {
  // WMO Weather interpretation codes
  // https://open-meteo.com/en/docs
  if (code === 0) return 'sunny';
  if (code >= 1 && code <= 3) return 'cloudy';
  if (code >= 45 && code <= 48) return 'foggy';
  if (code >= 51 && code <= 67) return 'rainy';
  if (code >= 71 && code <= 77) return 'snowy';
  if (code >= 80 && code <= 82) return 'rainy';
  if (code >= 85 && code <= 86) return 'snowy';
  if (code >= 95 && code <= 99) return 'stormy';
  return 'cloudy';
}

function getTimeOfDay(): TimeOfDay {
  const hour = new Date().getHours();
  if (hour >= 5 && hour < 7) return 'dawn';
  if (hour >= 7 && hour < 12) return 'morning';
  if (hour >= 12 && hour < 17) return 'afternoon';
  if (hour >= 17 && hour < 21) return 'evening';
  if (hour >= 21 && hour < 24) return 'night';
  return 'late_night';
}

export function useContextDetection() {
  const {
    autoDetectTime,
    autoDetectWeather,
    location,
    setTimeOfDay,
    setWeather,
    setLocation,
  } = useContextStore();

  // Update time of day
  const updateTime = useCallback(() => {
    if (autoDetectTime) {
      setTimeOfDay(getTimeOfDay());
    }
  }, [autoDetectTime, setTimeOfDay]);

  // Fetch weather from Open-Meteo
  const fetchWeather = useCallback(async (lat: number, lon: number) => {
    try {
      const response = await fetch(
        `https://api.open-meteo.com/v1/forecast?latitude=${lat}&longitude=${lon}&current=temperature_2m,weather_code,wind_speed_10m,precipitation`
      );

      if (!response.ok) throw new Error('Weather fetch failed');

      const data: WeatherResponse = await response.json();
      const weather = mapWeatherCode(data.current.weather_code);

      // Adjust for wind
      if (data.current.wind_speed_10m > 40) {
        setWeather('windy');
      } else {
        setWeather(weather);
      }
    } catch (error) {
      console.error('Failed to fetch weather:', error);
    }
  }, [setWeather]);

  // Get user location
  const requestLocation = useCallback(() => {
    if (!autoDetectWeather) return;

    if ('geolocation' in navigator) {
      navigator.geolocation.getCurrentPosition(
        async (position) => {
          const { latitude, longitude } = position.coords;
          setLocation(latitude, longitude);

          // Reverse geocode for location name
          try {
            const response = await fetch(
              `https://geocoding-api.open-meteo.com/v1/reverse?latitude=${latitude}&longitude=${longitude}&count=1`
            );
            const data = await response.json();
            if (data.results?.[0]) {
              setLocation(latitude, longitude, data.results[0].name);
            }
          } catch {
            // Ignore geocoding errors
          }

          // Fetch weather
          fetchWeather(latitude, longitude);
        },
        (error) => {
          console.warn('Geolocation error:', error.message);
        },
        { enableHighAccuracy: false, timeout: 10000, maximumAge: 300000 }
      );
    }
  }, [autoDetectWeather, setLocation, fetchWeather]);

  // Update time every minute
  useEffect(() => {
    updateTime();

    const interval = setInterval(updateTime, 60000);
    return () => clearInterval(interval);
  }, [updateTime]);

  // Get location and weather on mount
  useEffect(() => {
    if (autoDetectWeather && !location) {
      requestLocation();
    } else if (autoDetectWeather && location) {
      fetchWeather(location.lat, location.lon);

      // Refresh weather every 30 minutes
      const interval = setInterval(
        () => fetchWeather(location.lat, location.lon),
        30 * 60 * 1000
      );
      return () => clearInterval(interval);
    }
  }, [autoDetectWeather, location, fetchWeather, requestLocation]);

  return {
    updateTime,
    requestLocation,
    fetchWeather,
  };
}

// Hook to get contextual recommendations
export function useContextualRecommendations() {
  const {
    timeOfDay,
    weather,
    activity,
    energy,
    mood,
    contextualPlayEnabled,
    getCurrentMusicAttributes,
  } = useContextStore();

  const attributes = getCurrentMusicAttributes();

  // Generate search parameters for the recommendation API
  const getRecommendationParams = useCallback(() => {
    if (!contextualPlayEnabled) return null;

    return {
      target_tempo: (attributes.tempo.min + attributes.tempo.max) / 2,
      min_tempo: attributes.tempo.min,
      max_tempo: attributes.tempo.max,
      target_energy: attributes.energy,
      target_valence: attributes.valence,
      target_acousticness: attributes.acousticness,
      target_instrumentalness: attributes.instrumentalness,
      seed_genres: attributes.genres?.join(','),
      tags: attributes.tags?.join(','),
    };
  }, [contextualPlayEnabled, attributes]);

  return {
    context: {
      timeOfDay,
      weather,
      activity,
      energy,
      mood,
    },
    attributes,
    getRecommendationParams,
    isEnabled: contextualPlayEnabled,
  };
}
