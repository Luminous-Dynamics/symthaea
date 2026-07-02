// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import { useState, useRef } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api } from '@/lib/api';
import { useAuth } from '@/hooks/useAuth';
import { Sidebar } from '@/components/layout/Sidebar';
import { Header } from '@/components/layout/Header';
import { Player } from '@/components/player/Player';
import { formatNumber, truncateAddress, cn } from '@/lib/utils';
import {
  User,
  Pencil,
  Camera,
  Music2,
  Users,
  Heart,
  ExternalLink,
  Save,
  X,
  Plus,
  Trash2,
} from 'lucide-react';
import Image from 'next/image';
import { redirect } from 'next/navigation';

type Tab = 'profile' | 'artist';

export default function ProfilePage() {
  const { authenticated, user, walletAddress } = useAuth();
  const queryClient = useQueryClient();
  const avatarInputRef = useRef<HTMLInputElement>(null);
  const bannerInputRef = useRef<HTMLInputElement>(null);

  const [activeTab, setActiveTab] = useState<Tab>('profile');
  const [isEditing, setIsEditing] = useState(false);
  const [editData, setEditData] = useState({
    displayName: '',
    bio: '',
  });
  const [socialLinks, setSocialLinks] = useState<{ platform: string; url: string }[]>([]);
  const [newSocialPlatform, setNewSocialPlatform] = useState('');
  const [newSocialUrl, setNewSocialUrl] = useState('');

  if (!authenticated) {
    redirect('/');
  }

  const { data: profile, isLoading } = useQuery({
    queryKey: ['profile'],
    queryFn: () => api.getProfile(),
    enabled: authenticated,
  });

  const { data: stats } = useQuery({
    queryKey: ['profileStats'],
    queryFn: () => api.getProfileStats(),
    enabled: authenticated,
  });

  const updateProfile = useMutation({
    mutationFn: (data: { displayName?: string; bio?: string; socialLinks?: Record<string, string> }) =>
      api.updateProfile(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['profile'] });
      setIsEditing(false);
    },
  });

  const uploadAvatar = useMutation({
    mutationFn: (file: File) => api.uploadProfileImage(file, 'avatar'),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['profile'] });
    },
  });

  const uploadBanner = useMutation({
    mutationFn: (file: File) => api.uploadProfileImage(file, 'banner'),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['profile'] });
    },
  });

  const handleStartEdit = () => {
    setEditData({
      displayName: profile?.displayName || '',
      bio: profile?.bio || '',
    });
    const links = profile?.socialLinks
      ? Object.entries(profile.socialLinks).map(([platform, url]) => ({
          platform,
          url: url as string,
        }))
      : [];
    setSocialLinks(links);
    setIsEditing(true);
  };

  const handleSave = () => {
    const socialLinksObj = socialLinks.reduce(
      (acc, { platform, url }) => {
        if (platform && url) acc[platform] = url;
        return acc;
      },
      {} as Record<string, string>
    );

    updateProfile.mutate({
      displayName: editData.displayName,
      bio: editData.bio,
      socialLinks: socialLinksObj,
    });
  };

  const handleAvatarChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      uploadAvatar.mutate(file);
    }
  };

  const handleBannerChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      uploadBanner.mutate(file);
    }
  };

  const addSocialLink = () => {
    if (newSocialPlatform && newSocialUrl) {
      setSocialLinks([...socialLinks, { platform: newSocialPlatform, url: newSocialUrl }]);
      setNewSocialPlatform('');
      setNewSocialUrl('');
    }
  };

  const removeSocialLink = (index: number) => {
    setSocialLinks(socialLinks.filter((_, i) => i !== index));
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

        {/* Profile Header */}
        <div className="relative">
          {/* Banner */}
          <div className="relative h-48 overflow-hidden">
            {profile?.bannerImage ? (
              <Image
                src={profile.bannerImage}
                alt=""
                fill
                className="object-cover"
              />
            ) : (
              <div className="w-full h-full bg-gradient-to-br from-purple-600/50 to-fuchsia-700/50" />
            )}
            <div className="absolute inset-0 bg-gradient-to-t from-background to-transparent" />

            {/* Edit Banner Button */}
            <button
              onClick={() => bannerInputRef.current?.click()}
              disabled={uploadBanner.isPending}
              className="absolute top-4 right-4 flex items-center gap-2 px-3 py-1.5 bg-black/60 rounded-full text-sm hover:bg-black/80 transition-colors"
            >
              <Camera className="w-4 h-4" />
              {uploadBanner.isPending ? 'Uploading...' : 'Edit Banner'}
            </button>
            <input
              ref={bannerInputRef}
              type="file"
              accept="image/*"
              onChange={handleBannerChange}
              className="hidden"
            />
          </div>

          {/* Avatar */}
          <div className="absolute bottom-0 left-6 translate-y-1/2">
            <div className="relative">
              <div className="w-36 h-36 rounded-full border-4 border-background overflow-hidden bg-gradient-to-br from-purple-500 to-fuchsia-600">
                {profile?.avatar ? (
                  <Image
                    src={profile.avatar}
                    alt={profile.displayName || ''}
                    fill
                    className="object-cover"
                  />
                ) : (
                  <div className="w-full h-full flex items-center justify-center">
                    <User className="w-16 h-16" />
                  </div>
                )}
              </div>
              <button
                onClick={() => avatarInputRef.current?.click()}
                disabled={uploadAvatar.isPending}
                className="absolute bottom-0 right-0 w-10 h-10 rounded-full bg-white/10 backdrop-blur flex items-center justify-center hover:bg-white/20 transition-colors"
              >
                <Camera className="w-5 h-5" />
              </button>
              <input
                ref={avatarInputRef}
                type="file"
                accept="image/*"
                onChange={handleAvatarChange}
                className="hidden"
              />
            </div>
          </div>
        </div>

        {/* Profile Info */}
        <div className="pt-24 px-6">
          <div className="flex items-start justify-between mb-8">
            <div>
              {isEditing ? (
                <input
                  type="text"
                  value={editData.displayName}
                  onChange={(e) => setEditData({ ...editData, displayName: e.target.value })}
                  placeholder="Display name"
                  className="text-3xl font-bold bg-white/10 rounded px-3 py-1 focus:outline-none focus:ring-2 focus:ring-primary mb-2"
                />
              ) : (
                <h1 className="text-3xl font-bold mb-2">
                  {profile?.displayName || 'Anonymous'}
                </h1>
              )}
              <p className="text-sm text-muted-foreground font-mono">
                {truncateAddress(walletAddress || '')}
              </p>
            </div>

            {isEditing ? (
              <div className="flex gap-2">
                <button
                  onClick={handleSave}
                  disabled={updateProfile.isPending}
                  className="flex items-center gap-2 px-4 py-2 bg-primary text-black rounded-lg text-sm font-medium hover:bg-primary/90 disabled:opacity-50"
                >
                  <Save className="w-4 h-4" />
                  {updateProfile.isPending ? 'Saving...' : 'Save'}
                </button>
                <button
                  onClick={() => setIsEditing(false)}
                  className="flex items-center gap-2 px-4 py-2 bg-white/10 rounded-lg text-sm font-medium hover:bg-white/20"
                >
                  <X className="w-4 h-4" />
                  Cancel
                </button>
              </div>
            ) : (
              <button
                onClick={handleStartEdit}
                className="flex items-center gap-2 px-4 py-2 bg-white/10 rounded-lg text-sm font-medium hover:bg-white/20 transition-colors"
              >
                <Pencil className="w-4 h-4" />
                Edit Profile
              </button>
            )}
          </div>

          {/* Stats */}
          <div className="grid grid-cols-4 gap-4 mb-8">
            <div className="p-4 bg-white/5 rounded-lg">
              <div className="flex items-center gap-2 text-muted-foreground mb-2">
                <Music2 className="w-4 h-4" />
                <span className="text-sm">Songs Played</span>
              </div>
              <p className="text-2xl font-bold">
                {formatNumber(stats?.songsPlayed || 0)}
              </p>
            </div>
            <div className="p-4 bg-white/5 rounded-lg">
              <div className="flex items-center gap-2 text-muted-foreground mb-2">
                <Heart className="w-4 h-4" />
                <span className="text-sm">Liked Songs</span>
              </div>
              <p className="text-2xl font-bold">
                {formatNumber(stats?.likedSongs || 0)}
              </p>
            </div>
            <div className="p-4 bg-white/5 rounded-lg">
              <div className="flex items-center gap-2 text-muted-foreground mb-2">
                <Users className="w-4 h-4" />
                <span className="text-sm">Following</span>
              </div>
              <p className="text-2xl font-bold">
                {formatNumber(stats?.following || 0)}
              </p>
            </div>
            <div className="p-4 bg-white/5 rounded-lg">
              <div className="flex items-center gap-2 text-muted-foreground mb-2">
                <Music2 className="w-4 h-4" />
                <span className="text-sm">Playlists</span>
              </div>
              <p className="text-2xl font-bold">
                {formatNumber(stats?.playlists || 0)}
              </p>
            </div>
          </div>

          {/* Bio */}
          <div className="mb-8">
            <h2 className="text-lg font-semibold mb-3">About</h2>
            {isEditing ? (
              <textarea
                value={editData.bio}
                onChange={(e) => setEditData({ ...editData, bio: e.target.value })}
                placeholder="Tell us about yourself..."
                className="w-full max-w-xl h-32 px-4 py-3 bg-white/10 rounded-lg text-sm resize-none focus:outline-none focus:ring-2 focus:ring-primary"
              />
            ) : (
              <p className="text-muted-foreground max-w-xl">
                {profile?.bio || 'No bio yet'}
              </p>
            )}
          </div>

          {/* Social Links */}
          <div className="mb-8">
            <h2 className="text-lg font-semibold mb-3">Social Links</h2>

            {isEditing ? (
              <div className="max-w-xl space-y-3">
                {socialLinks.map((link, index) => (
                  <div key={index} className="flex gap-2">
                    <input
                      type="text"
                      value={link.platform}
                      onChange={(e) => {
                        const updated = [...socialLinks];
                        updated[index].platform = e.target.value;
                        setSocialLinks(updated);
                      }}
                      placeholder="Platform"
                      className="flex-1 px-3 py-2 bg-white/10 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary"
                    />
                    <input
                      type="url"
                      value={link.url}
                      onChange={(e) => {
                        const updated = [...socialLinks];
                        updated[index].url = e.target.value;
                        setSocialLinks(updated);
                      }}
                      placeholder="URL"
                      className="flex-[2] px-3 py-2 bg-white/10 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary"
                    />
                    <button
                      onClick={() => removeSocialLink(index)}
                      className="p-2 text-red-400 hover:text-red-300"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                ))}

                <div className="flex gap-2">
                  <input
                    type="text"
                    value={newSocialPlatform}
                    onChange={(e) => setNewSocialPlatform(e.target.value)}
                    placeholder="Platform (e.g., Twitter)"
                    className="flex-1 px-3 py-2 bg-white/10 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary"
                  />
                  <input
                    type="url"
                    value={newSocialUrl}
                    onChange={(e) => setNewSocialUrl(e.target.value)}
                    placeholder="URL"
                    className="flex-[2] px-3 py-2 bg-white/10 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary"
                  />
                  <button
                    onClick={addSocialLink}
                    disabled={!newSocialPlatform || !newSocialUrl}
                    className="p-2 text-primary hover:text-primary/80 disabled:opacity-50"
                  >
                    <Plus className="w-5 h-5" />
                  </button>
                </div>
              </div>
            ) : profile?.socialLinks && Object.keys(profile.socialLinks).length > 0 ? (
              <div className="flex flex-wrap gap-2">
                {Object.entries(profile.socialLinks).map(([platform, url]) => (
                  <a
                    key={platform}
                    href={url as string}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center gap-2 px-4 py-2 bg-white/10 rounded-full text-sm hover:bg-white/20 transition-colors"
                  >
                    {platform}
                    <ExternalLink className="w-3 h-3" />
                  </a>
                ))}
              </div>
            ) : (
              <p className="text-muted-foreground text-sm">No social links added</p>
            )}
          </div>

          {/* Artist Mode */}
          {profile?.isArtist && (
            <div className="p-6 bg-gradient-to-r from-purple-500/20 to-fuchsia-600/20 rounded-xl">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-lg font-semibold mb-1">Artist Account</h3>
                  <p className="text-sm text-muted-foreground">
                    You're registered as an artist. Access your dashboard to manage releases.
                  </p>
                </div>
                <a
                  href="/dashboard"
                  className="px-6 py-2 bg-primary text-black rounded-lg font-medium hover:bg-primary/90 transition-colors"
                >
                  Go to Dashboard
                </a>
              </div>
            </div>
          )}
        </div>
      </main>

      <Player />
    </div>
  );
}
