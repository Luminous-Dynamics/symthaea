// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import { useState } from 'react';
import {
  useContextStore,
  DEFAULT_PROFILES,
  TimeOfDay,
  Weather,
  Activity,
  Energy,
  Mood,
} from '@/store/contextStore';
import { useContextDetection } from '@/hooks/useContextDetection';
import {
  Sun,
  Moon,
  Cloud,
  CloudRain,
  CloudSnow,
  CloudLightning,
  Wind,
  CloudFog,
  Sunrise,
  Sunset,
  Music2,
  Brain,
  Dumbbell,
  Car,
  Sofa,
  BedDouble,
  Palette,
  Users,
  Heart,
  Zap,
  Sparkles,
  Focus,
  Coffee,
  Flame,
  Settings,
  Check,
  RefreshCw,
  MapPin,
} from 'lucide-react';

const TIME_ICONS: Record<TimeOfDay, typeof Sun> = {
  dawn: Sunrise,
  morning: Sun,
  afternoon: Sun,
  evening: Sunset,
  night: Moon,
  late_night: Moon,
};

const WEATHER_ICONS: Record<Weather, typeof Sun> = {
  sunny: Sun,
  cloudy: Cloud,
  rainy: CloudRain,
  stormy: CloudLightning,
  snowy: CloudSnow,
  windy: Wind,
  foggy: CloudFog,
};

const ACTIVITY_OPTIONS: { value: Activity; label: string; icon: typeof Sun }[] = [
  { value: 'waking', label: 'Waking Up', icon: Coffee },
  { value: 'working', label: 'Working', icon: Brain },
  { value: 'exercising', label: 'Exercising', icon: Dumbbell },
  { value: 'commuting', label: 'Commuting', icon: Car },
  { value: 'relaxing', label: 'Relaxing', icon: Sofa },
  { value: 'sleeping', label: 'Sleeping', icon: BedDouble },
  { value: 'creating', label: 'Creating', icon: Palette },
  { value: 'socializing', label: 'Socializing', icon: Users },
];

const ENERGY_OPTIONS: { value: Energy; label: string; color: string }[] = [
  { value: 'low', label: 'Low', color: 'bg-blue-500' },
  { value: 'medium', label: 'Medium', color: 'bg-green-500' },
  { value: 'high', label: 'High', color: 'bg-yellow-500' },
  { value: 'peak', label: 'Peak', color: 'bg-red-500' },
];

const MOOD_OPTIONS: { value: Mood; label: string; emoji: string }[] = [
  { value: 'happy', label: 'Happy', emoji: '😊' },
  { value: 'calm', label: 'Calm', emoji: '😌' },
  { value: 'focused', label: 'Focused', emoji: '🎯' },
  { value: 'melancholic', label: 'Melancholic', emoji: '😢' },
  { value: 'energetic', label: 'Energetic', emoji: '⚡' },
  { value: 'romantic', label: 'Romantic', emoji: '💕' },
  { value: 'nostalgic', label: 'Nostalgic', emoji: '🌅' },
  { value: 'adventurous', label: 'Adventurous', emoji: '🚀' },
];

export function ContextualAudio() {
  const {
    timeOfDay,
    weather,
    activity,
    energy,
    mood,
    locationName,
    contextualPlayEnabled,
    activeProfileId,
    customProfiles,
    setActivity,
    setEnergy,
    setMood,
    setActiveProfile,
    toggleContextualPlay,
  } = useContextStore();

  const { requestLocation } = useContextDetection();
  const [showSettings, setShowSettings] = useState(false);

  const allProfiles = [...DEFAULT_PROFILES, ...customProfiles];
  const TimeIcon = TIME_ICONS[timeOfDay];
  const WeatherIcon = weather ? WEATHER_ICONS[weather] : Cloud;

  return (
    <div className="bg-white/5 rounded-2xl p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-full bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center">
            <Sparkles className="w-5 h-5" />
          </div>
          <div>
            <h2 className="font-semibold">Contextual Audio</h2>
            <p className="text-sm text-muted-foreground">
              Music adapts to your moment
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowSettings(!showSettings)}
            className="p-2 rounded-lg hover:bg-white/10"
          >
            <Settings className="w-5 h-5" />
          </button>
          <button
            onClick={toggleContextualPlay}
            className={`px-4 py-2 rounded-full text-sm font-medium transition-colors ${
              contextualPlayEnabled
                ? 'bg-purple-500 text-white'
                : 'bg-white/10 text-muted-foreground'
            }`}
          >
            {contextualPlayEnabled ? 'On' : 'Off'}
          </button>
        </div>
      </div>

      {/* Current Context */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        {/* Time */}
        <div className="bg-white/5 rounded-xl p-4 text-center">
          <TimeIcon className="w-8 h-8 mx-auto mb-2 text-yellow-400" />
          <p className="text-sm text-muted-foreground">Time</p>
          <p className="font-medium capitalize">{timeOfDay.replace('_', ' ')}</p>
        </div>

        {/* Weather */}
        <div className="bg-white/5 rounded-xl p-4 text-center">
          <WeatherIcon className="w-8 h-8 mx-auto mb-2 text-blue-400" />
          <p className="text-sm text-muted-foreground">Weather</p>
          <p className="font-medium capitalize">{weather || 'Unknown'}</p>
          {locationName && (
            <p className="text-xs text-muted-foreground flex items-center justify-center gap-1 mt-1">
              <MapPin className="w-3 h-3" />
              {locationName}
            </p>
          )}
          {!weather && (
            <button
              onClick={requestLocation}
              className="text-xs text-purple-400 hover:underline mt-1"
            >
              Enable location
            </button>
          )}
        </div>

        {/* Activity */}
        <div className="bg-white/5 rounded-xl p-4 text-center">
          {(() => {
            const activityOption = ACTIVITY_OPTIONS.find(a => a.value === activity);
            const ActivityIcon = activityOption?.icon || Sofa;
            return (
              <>
                <ActivityIcon className="w-8 h-8 mx-auto mb-2 text-green-400" />
                <p className="text-sm text-muted-foreground">Activity</p>
                <p className="font-medium">{activityOption?.label}</p>
              </>
            );
          })()}
        </div>

        {/* Mood */}
        <div className="bg-white/5 rounded-xl p-4 text-center">
          <div className="text-3xl mb-2">
            {MOOD_OPTIONS.find(m => m.value === mood)?.emoji || '😊'}
          </div>
          <p className="text-sm text-muted-foreground">Mood</p>
          <p className="font-medium capitalize">{mood}</p>
        </div>
      </div>

      {/* Activity Selection */}
      <div className="mb-6">
        <p className="text-sm font-medium mb-3">What are you doing?</p>
        <div className="flex flex-wrap gap-2">
          {ACTIVITY_OPTIONS.map((option) => {
            const Icon = option.icon;
            const isActive = activity === option.value;
            return (
              <button
                key={option.value}
                onClick={() => setActivity(option.value)}
                className={`flex items-center gap-2 px-3 py-2 rounded-full text-sm transition-colors ${
                  isActive
                    ? 'bg-purple-500 text-white'
                    : 'bg-white/5 hover:bg-white/10'
                }`}
              >
                <Icon className="w-4 h-4" />
                {option.label}
              </button>
            );
          })}
        </div>
      </div>

      {/* Energy Level */}
      <div className="mb-6">
        <p className="text-sm font-medium mb-3">Energy Level</p>
        <div className="flex gap-2">
          {ENERGY_OPTIONS.map((option) => {
            const isActive = energy === option.value;
            return (
              <button
                key={option.value}
                onClick={() => setEnergy(option.value)}
                className={`flex-1 py-2 rounded-lg text-sm font-medium transition-all ${
                  isActive
                    ? `${option.color} text-white`
                    : 'bg-white/5 hover:bg-white/10'
                }`}
              >
                {option.label}
              </button>
            );
          })}
        </div>
      </div>

      {/* Mood Selection */}
      <div className="mb-6">
        <p className="text-sm font-medium mb-3">How are you feeling?</p>
        <div className="grid grid-cols-4 gap-2">
          {MOOD_OPTIONS.map((option) => {
            const isActive = mood === option.value;
            return (
              <button
                key={option.value}
                onClick={() => setMood(option.value)}
                className={`flex flex-col items-center gap-1 py-3 rounded-xl text-sm transition-colors ${
                  isActive
                    ? 'bg-purple-500/20 border border-purple-500'
                    : 'bg-white/5 hover:bg-white/10 border border-transparent'
                }`}
              >
                <span className="text-xl">{option.emoji}</span>
                <span className="text-xs">{option.label}</span>
              </button>
            );
          })}
        </div>
      </div>

      {/* Quick Profiles */}
      <div>
        <p className="text-sm font-medium mb-3">Quick Vibes</p>
        <div className="grid grid-cols-2 md:grid-cols-5 gap-2">
          {allProfiles.slice(0, 10).map((profile) => {
            const isActive = activeProfileId === profile.id;
            return (
              <button
                key={profile.id}
                onClick={() => setActiveProfile(isActive ? null : profile.id)}
                className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm transition-colors ${
                  isActive
                    ? 'bg-purple-500 text-white'
                    : 'bg-white/5 hover:bg-white/10'
                }`}
              >
                <span>{profile.icon}</span>
                <span className="truncate">{profile.name}</span>
                {isActive && <Check className="w-4 h-4 ml-auto flex-shrink-0" />}
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}

// Compact version for sidebar/player
export function ContextIndicator() {
  const { timeOfDay, weather, mood, contextualPlayEnabled } = useContextStore();

  if (!contextualPlayEnabled) return null;

  const TimeIcon = TIME_ICONS[timeOfDay];
  const moodOption = MOOD_OPTIONS.find(m => m.value === mood);

  return (
    <div className="flex items-center gap-2 px-3 py-1.5 bg-white/5 rounded-full text-xs">
      <Sparkles className="w-3 h-3 text-purple-400" />
      <TimeIcon className="w-3 h-3 text-yellow-400" />
      {weather && (
        <>
          {(() => {
            const WeatherIcon = WEATHER_ICONS[weather];
            return <WeatherIcon className="w-3 h-3 text-blue-400" />;
          })()}
        </>
      )}
      <span>{moodOption?.emoji}</span>
    </div>
  );
}
