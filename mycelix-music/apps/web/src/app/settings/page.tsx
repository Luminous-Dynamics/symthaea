// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api } from '@/lib/api';
import { useAuth } from '@/hooks/useAuth';
import { Sidebar } from '@/components/layout/Sidebar';
import { Header } from '@/components/layout/Header';
import { Player } from '@/components/player/Player';
import { truncateAddress, cn } from '@/lib/utils';
import {
  User,
  Bell,
  Volume2,
  Palette,
  Shield,
  Wallet,
  LogOut,
  ChevronRight,
  Check,
  Moon,
  Sun,
  Smartphone,
  Mail,
  Music2,
  Users,
  DollarSign,
  ExternalLink,
} from 'lucide-react';
import { redirect } from 'next/navigation';

type SettingsSection = 'account' | 'notifications' | 'playback' | 'appearance' | 'privacy';

export default function SettingsPage() {
  const { authenticated, user, walletAddress, logout } = useAuth();
  const queryClient = useQueryClient();

  const [activeSection, setActiveSection] = useState<SettingsSection>('account');

  if (!authenticated) {
    redirect('/');
  }

  const { data: settings, isLoading } = useQuery({
    queryKey: ['settings'],
    queryFn: () => api.getSettings(),
    enabled: authenticated,
  });

  const updateSettings = useMutation({
    mutationFn: (data: Partial<typeof settings>) => api.updateSettings(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['settings'] });
    },
  });

  const sections = [
    { id: 'account' as const, label: 'Account', icon: User },
    { id: 'notifications' as const, label: 'Notifications', icon: Bell },
    { id: 'playback' as const, label: 'Playback', icon: Volume2 },
    { id: 'appearance' as const, label: 'Appearance', icon: Palette },
    { id: 'privacy' as const, label: 'Privacy', icon: Shield },
  ];

  const handleToggle = (key: string, value: boolean) => {
    updateSettings.mutate({ [key]: value });
  };

  if (isLoading) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="w-8 h-8 border-2 border-primary border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background">
      <Sidebar />

      <main className="ml-64 pb-24">
        <Header />

        <div className="px-6 py-4">
          <h1 className="text-3xl font-bold mb-8">Settings</h1>

          <div className="flex gap-8">
            {/* Sidebar Navigation */}
            <div className="w-64 flex-shrink-0">
              <nav className="space-y-1">
                {sections.map((section) => (
                  <button
                    key={section.id}
                    onClick={() => setActiveSection(section.id)}
                    className={cn(
                      'w-full flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition-colors',
                      activeSection === section.id
                        ? 'bg-white/10 text-white'
                        : 'text-muted-foreground hover:text-white hover:bg-white/5'
                    )}
                  >
                    <section.icon className="w-5 h-5" />
                    {section.label}
                  </button>
                ))}

                <div className="pt-4 mt-4 border-t border-white/10">
                  <button
                    onClick={logout}
                    className="w-full flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium text-red-400 hover:text-red-300 hover:bg-red-500/10 transition-colors"
                  >
                    <LogOut className="w-5 h-5" />
                    Log Out
                  </button>
                </div>
              </nav>
            </div>

            {/* Content */}
            <div className="flex-1 max-w-2xl">
              {activeSection === 'account' && (
                <div className="space-y-8">
                  <section>
                    <h2 className="text-xl font-semibold mb-4">Account</h2>

                    <div className="space-y-4">
                      {/* Wallet */}
                      <div className="p-4 bg-white/5 rounded-lg">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-3">
                            <Wallet className="w-5 h-5 text-muted-foreground" />
                            <div>
                              <p className="font-medium">Connected Wallet</p>
                              <p className="text-sm text-muted-foreground font-mono">
                                {truncateAddress(walletAddress || '')}
                              </p>
                            </div>
                          </div>
                          <span className="flex items-center gap-1 text-green-500 text-sm">
                            <div className="w-2 h-2 rounded-full bg-green-500" />
                            Connected
                          </span>
                        </div>
                      </div>

                      {/* Email */}
                      <div className="p-4 bg-white/5 rounded-lg">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-3">
                            <Mail className="w-5 h-5 text-muted-foreground" />
                            <div>
                              <p className="font-medium">Email</p>
                              <p className="text-sm text-muted-foreground">
                                {user?.email || 'Not connected'}
                              </p>
                            </div>
                          </div>
                          {!user?.email && (
                            <button className="text-sm text-primary hover:underline">
                              Add email
                            </button>
                          )}
                        </div>
                      </div>

                      {/* Artist Mode */}
                      <div className="p-4 bg-white/5 rounded-lg">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-3">
                            <Music2 className="w-5 h-5 text-muted-foreground" />
                            <div>
                              <p className="font-medium">Artist Mode</p>
                              <p className="text-sm text-muted-foreground">
                                {settings?.isArtist
                                  ? 'You are registered as an artist'
                                  : 'Become an artist to upload music'}
                              </p>
                            </div>
                          </div>
                          {settings?.isArtist ? (
                            <a
                              href="/dashboard"
                              className="text-sm text-primary hover:underline flex items-center gap-1"
                            >
                              Go to Dashboard
                              <ChevronRight className="w-4 h-4" />
                            </a>
                          ) : (
                            <button className="px-4 py-2 bg-primary text-black rounded-lg text-sm font-medium hover:bg-primary/90">
                              Become Artist
                            </button>
                          )}
                        </div>
                      </div>
                    </div>
                  </section>

                  {/* Danger Zone */}
                  <section>
                    <h2 className="text-xl font-semibold mb-4 text-red-400">Danger Zone</h2>
                    <div className="p-4 bg-red-500/10 border border-red-500/20 rounded-lg">
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="font-medium">Delete Account</p>
                          <p className="text-sm text-muted-foreground">
                            Permanently delete your account and all data
                          </p>
                        </div>
                        <button className="px-4 py-2 border border-red-500 text-red-400 rounded-lg text-sm font-medium hover:bg-red-500/10">
                          Delete Account
                        </button>
                      </div>
                    </div>
                  </section>
                </div>
              )}

              {activeSection === 'notifications' && (
                <div className="space-y-6">
                  <h2 className="text-xl font-semibold mb-4">Notifications</h2>

                  <div className="space-y-4">
                    <ToggleSetting
                      icon={<Users className="w-5 h-5" />}
                      title="New Followers"
                      description="Get notified when someone follows you"
                      enabled={settings?.notifyNewFollowers ?? true}
                      onToggle={(v) => handleToggle('notifyNewFollowers', v)}
                    />

                    <ToggleSetting
                      icon={<Music2 className="w-5 h-5" />}
                      title="New Releases"
                      description="Get notified when artists you follow release new music"
                      enabled={settings?.notifyNewReleases ?? true}
                      onToggle={(v) => handleToggle('notifyNewReleases', v)}
                    />

                    <ToggleSetting
                      icon={<DollarSign className="w-5 h-5" />}
                      title="Earnings Updates"
                      description="Get notified about your streaming earnings"
                      enabled={settings?.notifyEarnings ?? true}
                      onToggle={(v) => handleToggle('notifyEarnings', v)}
                    />

                    <ToggleSetting
                      icon={<Mail className="w-5 h-5" />}
                      title="Email Notifications"
                      description="Receive notifications via email"
                      enabled={settings?.emailNotifications ?? false}
                      onToggle={(v) => handleToggle('emailNotifications', v)}
                    />

                    <ToggleSetting
                      icon={<Smartphone className="w-5 h-5" />}
                      title="Push Notifications"
                      description="Receive push notifications on your device"
                      enabled={settings?.pushNotifications ?? true}
                      onToggle={(v) => handleToggle('pushNotifications', v)}
                    />
                  </div>
                </div>
              )}

              {activeSection === 'playback' && (
                <div className="space-y-6">
                  <h2 className="text-xl font-semibold mb-4">Playback</h2>

                  <div className="space-y-4">
                    <div className="p-4 bg-white/5 rounded-lg">
                      <div className="flex items-center justify-between mb-4">
                        <div>
                          <p className="font-medium">Audio Quality</p>
                          <p className="text-sm text-muted-foreground">
                            Higher quality uses more data
                          </p>
                        </div>
                      </div>
                      <div className="flex gap-2">
                        {['low', 'normal', 'high', 'lossless'].map((quality) => (
                          <button
                            key={quality}
                            onClick={() => handleToggle('audioQuality', quality as any)}
                            className={cn(
                              'flex-1 px-4 py-2 rounded-lg text-sm font-medium transition-colors capitalize',
                              settings?.audioQuality === quality
                                ? 'bg-primary text-black'
                                : 'bg-white/10 hover:bg-white/20'
                            )}
                          >
                            {quality}
                          </button>
                        ))}
                      </div>
                    </div>

                    <ToggleSetting
                      icon={<Volume2 className="w-5 h-5" />}
                      title="Normalize Volume"
                      description="Set the same volume level for all songs"
                      enabled={settings?.normalizeVolume ?? true}
                      onToggle={(v) => handleToggle('normalizeVolume', v)}
                    />

                    <ToggleSetting
                      icon={<Music2 className="w-5 h-5" />}
                      title="Crossfade"
                      description="Blend songs into each other"
                      enabled={settings?.crossfade ?? false}
                      onToggle={(v) => handleToggle('crossfade', v)}
                    />

                    <ToggleSetting
                      icon={<Music2 className="w-5 h-5" />}
                      title="Autoplay"
                      description="Continue playing similar songs when your queue ends"
                      enabled={settings?.autoplay ?? true}
                      onToggle={(v) => handleToggle('autoplay', v)}
                    />
                  </div>
                </div>
              )}

              {activeSection === 'appearance' && (
                <div className="space-y-6">
                  <h2 className="text-xl font-semibold mb-4">Appearance</h2>

                  <div className="p-4 bg-white/5 rounded-lg">
                    <div className="flex items-center justify-between mb-4">
                      <div>
                        <p className="font-medium">Theme</p>
                        <p className="text-sm text-muted-foreground">
                          Choose your preferred color scheme
                        </p>
                      </div>
                    </div>
                    <div className="flex gap-2">
                      {[
                        { value: 'dark', label: 'Dark', icon: Moon },
                        { value: 'light', label: 'Light', icon: Sun },
                        { value: 'system', label: 'System', icon: Smartphone },
                      ].map((theme) => (
                        <button
                          key={theme.value}
                          onClick={() => handleToggle('theme', theme.value as any)}
                          className={cn(
                            'flex-1 flex items-center justify-center gap-2 px-4 py-3 rounded-lg text-sm font-medium transition-colors',
                            settings?.theme === theme.value
                              ? 'bg-primary text-black'
                              : 'bg-white/10 hover:bg-white/20'
                          )}
                        >
                          <theme.icon className="w-4 h-4" />
                          {theme.label}
                        </button>
                      ))}
                    </div>
                  </div>

                  <div className="p-4 bg-white/5 rounded-lg">
                    <div className="flex items-center justify-between mb-4">
                      <div>
                        <p className="font-medium">Accent Color</p>
                        <p className="text-sm text-muted-foreground">
                          Choose your primary accent color
                        </p>
                      </div>
                    </div>
                    <div className="flex gap-2">
                      {[
                        { value: 'purple', color: '#8B5CF6' },
                        { value: 'pink', color: '#EC4899' },
                        { value: 'blue', color: '#3B82F6' },
                        { value: 'green', color: '#10B981' },
                        { value: 'orange', color: '#F59E0B' },
                      ].map((accent) => (
                        <button
                          key={accent.value}
                          onClick={() => handleToggle('accentColor', accent.value as any)}
                          className={cn(
                            'w-10 h-10 rounded-full transition-transform hover:scale-110',
                            settings?.accentColor === accent.value && 'ring-2 ring-white ring-offset-2 ring-offset-background'
                          )}
                          style={{ backgroundColor: accent.color }}
                        />
                      ))}
                    </div>
                  </div>
                </div>
              )}

              {activeSection === 'privacy' && (
                <div className="space-y-6">
                  <h2 className="text-xl font-semibold mb-4">Privacy</h2>

                  <div className="space-y-4">
                    <ToggleSetting
                      icon={<Users className="w-5 h-5" />}
                      title="Private Profile"
                      description="Only you can see your listening activity"
                      enabled={settings?.privateProfile ?? false}
                      onToggle={(v) => handleToggle('privateProfile', v)}
                    />

                    <ToggleSetting
                      icon={<Music2 className="w-5 h-5" />}
                      title="Hide Recently Played"
                      description="Don't show your recently played tracks"
                      enabled={settings?.hideRecentlyPlayed ?? false}
                      onToggle={(v) => handleToggle('hideRecentlyPlayed', v)}
                    />

                    <ToggleSetting
                      icon={<Shield className="w-5 h-5" />}
                      title="Explicit Content"
                      description="Allow explicit content to be played"
                      enabled={settings?.allowExplicit ?? true}
                      onToggle={(v) => handleToggle('allowExplicit', v)}
                    />
                  </div>

                  <div className="pt-6 border-t border-white/10">
                    <h3 className="font-medium mb-4">Data & Privacy</h3>
                    <div className="space-y-3">
                      <a
                        href="/privacy"
                        className="flex items-center justify-between p-4 bg-white/5 rounded-lg hover:bg-white/10 transition-colors"
                      >
                        <span>Privacy Policy</span>
                        <ExternalLink className="w-4 h-4 text-muted-foreground" />
                      </a>
                      <a
                        href="/terms"
                        className="flex items-center justify-between p-4 bg-white/5 rounded-lg hover:bg-white/10 transition-colors"
                      >
                        <span>Terms of Service</span>
                        <ExternalLink className="w-4 h-4 text-muted-foreground" />
                      </a>
                      <button className="w-full flex items-center justify-between p-4 bg-white/5 rounded-lg hover:bg-white/10 transition-colors text-left">
                        <span>Download My Data</span>
                        <ChevronRight className="w-4 h-4 text-muted-foreground" />
                      </button>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </main>

      <Player />
    </div>
  );
}

// Toggle Setting Component
function ToggleSetting({
  icon,
  title,
  description,
  enabled,
  onToggle,
}: {
  icon: React.ReactNode;
  title: string;
  description: string;
  enabled: boolean;
  onToggle: (value: boolean) => void;
}) {
  return (
    <div className="p-4 bg-white/5 rounded-lg">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="text-muted-foreground">{icon}</div>
          <div>
            <p className="font-medium">{title}</p>
            <p className="text-sm text-muted-foreground">{description}</p>
          </div>
        </div>
        <button
          onClick={() => onToggle(!enabled)}
          className={cn(
            'relative w-12 h-6 rounded-full transition-colors',
            enabled ? 'bg-primary' : 'bg-white/20'
          )}
        >
          <div
            className={cn(
              'absolute top-1 w-4 h-4 rounded-full bg-white transition-transform',
              enabled ? 'left-7' : 'left-1'
            )}
          />
        </button>
      </div>
    </div>
  );
}
