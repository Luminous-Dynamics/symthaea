// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import {
  Home,
  Search,
  Library,
  PlusSquare,
  Heart,
  Music2,
  Radio,
  TrendingUp,
  Clock,
  BarChart3,
  Disc3,
  Mic,
  Sliders,
  Layers,
  Wifi,
  WifiOff,
  Users,
  Sparkles,
  Headphones,
  Wand2,
  GitBranch,
  Box,
  Gauge,
  BookOpen,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { useAuth } from '@/hooks/useAuth';

const mainNav = [
  { href: '/', label: 'Home', icon: Home },
  { href: '/search', label: 'Search', icon: Search },
  { href: '/library', label: 'Your Library', icon: Library },
];

const browseNav = [
  { href: '/trending', label: 'Trending', icon: TrendingUp },
  { href: '/new-releases', label: 'New Releases', icon: Music2 },
  { href: '/discover', label: 'Discover', icon: Sparkles },
  { href: '/radio', label: 'Radio', icon: Radio },
];

const studioNav = [
  { href: '/studio', label: 'Studio', icon: Headphones },
  { href: '/studio/dj', label: 'DJ Mixer', icon: Disc3 },
  { href: '/studio/stems', label: 'Stem Separation', icon: Layers },
  { href: '/studio/effects', label: 'Audio Effects', icon: Sliders },
  { href: '/studio/generate', label: 'AI Generation', icon: Wand2 },
  { href: '/studio/remix', label: 'Remix Studio', icon: GitBranch },
  { href: '/studio/mastering', label: 'AI Mastering', icon: Gauge },
  { href: '/studio/theory', label: 'Music Theory', icon: BookOpen },
  { href: '/studio/broadcast', label: 'Go Live', icon: Wifi },
];

const socialNav = [
  { href: '/listening-party', label: 'Listening Party', icon: Users },
  { href: '/live', label: 'Live Streams', icon: Radio },
  { href: '/immersive', label: 'Immersive Audio', icon: Box },
];

const libraryNav = [
  { href: '/liked', label: 'Liked Songs', icon: Heart },
  { href: '/history', label: 'Recently Played', icon: Clock },
];

export function Sidebar() {
  const pathname = usePathname();
  const { authenticated, user } = useAuth();

  return (
    <aside className="fixed left-0 top-0 bottom-20 w-64 bg-black p-4 flex flex-col gap-6 overflow-y-auto scrollbar-hide">
      {/* Logo */}
      <Link href="/" className="flex items-center gap-2 px-2">
        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-purple-500 to-fuchsia-600 flex items-center justify-center">
          <Music2 className="w-5 h-5 text-white" />
        </div>
        <span className="text-xl font-bold text-gradient">Mycelix</span>
      </Link>

      {/* Main Nav */}
      <nav className="flex flex-col gap-1">
        {mainNav.map((item) => {
          const Icon = item.icon;
          const isActive = pathname === item.href;

          return (
            <Link
              key={item.href}
              href={item.href}
              className={cn(
                'flex items-center gap-3 px-3 py-2 rounded-md text-sm font-medium transition-colors',
                isActive
                  ? 'bg-white/10 text-white'
                  : 'text-muted-foreground hover:text-white hover:bg-white/5'
              )}
            >
              <Icon className="w-5 h-5" />
              {item.label}
            </Link>
          );
        })}
      </nav>

      {/* Browse */}
      <div>
        <h3 className="px-3 text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">
          Browse
        </h3>
        <nav className="flex flex-col gap-1">
          {browseNav.map((item) => {
            const Icon = item.icon;
            const isActive = pathname === item.href;

            return (
              <Link
                key={item.href}
                href={item.href}
                className={cn(
                  'flex items-center gap-3 px-3 py-2 rounded-md text-sm font-medium transition-colors',
                  isActive
                    ? 'bg-white/10 text-white'
                    : 'text-muted-foreground hover:text-white hover:bg-white/5'
                )}
              >
                <Icon className="w-5 h-5" />
                {item.label}
              </Link>
            );
          })}
        </nav>
      </div>

      {/* Studio */}
      <div>
        <h3 className="px-3 text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">
          Studio
        </h3>
        <nav className="flex flex-col gap-1">
          {studioNav.map((item) => {
            const Icon = item.icon;
            const isActive = pathname === item.href || pathname.startsWith(item.href + '/');

            return (
              <Link
                key={item.href}
                href={item.href}
                className={cn(
                  'flex items-center gap-3 px-3 py-2 rounded-md text-sm font-medium transition-colors',
                  isActive
                    ? 'bg-purple-500/20 text-purple-400'
                    : 'text-muted-foreground hover:text-white hover:bg-white/5'
                )}
              >
                <Icon className="w-5 h-5" />
                {item.label}
              </Link>
            );
          })}
        </nav>
      </div>

      {/* Social */}
      <div>
        <h3 className="px-3 text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">
          Social
        </h3>
        <nav className="flex flex-col gap-1">
          {socialNav.map((item) => {
            const Icon = item.icon;
            const isActive = pathname === item.href;

            return (
              <Link
                key={item.href}
                href={item.href}
                className={cn(
                  'flex items-center gap-3 px-3 py-2 rounded-md text-sm font-medium transition-colors',
                  isActive
                    ? 'bg-white/10 text-white'
                    : 'text-muted-foreground hover:text-white hover:bg-white/5'
                )}
              >
                <Icon className="w-5 h-5" />
                {item.label}
              </Link>
            );
          })}
        </nav>
      </div>

      {/* Your Library */}
      {authenticated && (
        <div>
          <h3 className="px-3 text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">
            Your Library
          </h3>
          <nav className="flex flex-col gap-1">
            {libraryNav.map((item) => {
              const Icon = item.icon;
              const isActive = pathname === item.href;

              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={cn(
                    'flex items-center gap-3 px-3 py-2 rounded-md text-sm font-medium transition-colors',
                    isActive
                      ? 'bg-white/10 text-white'
                      : 'text-muted-foreground hover:text-white hover:bg-white/5'
                  )}
                >
                  <Icon className="w-5 h-5" />
                  {item.label}
                </Link>
              );
            })}

            <button className="flex items-center gap-3 px-3 py-2 rounded-md text-sm font-medium text-muted-foreground hover:text-white hover:bg-white/5 transition-colors">
              <PlusSquare className="w-5 h-5" />
              Create Playlist
            </button>
          </nav>
        </div>
      )}

      {/* Artist Dashboard Link */}
      {authenticated && user?.isArtist && (
        <div className="mt-auto">
          <Link
            href="/dashboard"
            className={cn(
              'flex items-center gap-3 px-3 py-2 rounded-md text-sm font-medium transition-colors',
              pathname.startsWith('/dashboard')
                ? 'bg-primary/20 text-primary'
                : 'text-muted-foreground hover:text-white hover:bg-white/5'
            )}
          >
            <BarChart3 className="w-5 h-5" />
            Artist Dashboard
          </Link>
        </div>
      )}

      {/* Playlists */}
      <div className="flex-1 border-t border-white/10 pt-4 mt-2">
        <div className="px-3 text-xs text-muted-foreground">
          Your playlists will appear here
        </div>
      </div>
    </aside>
  );
}

export default Sidebar;
