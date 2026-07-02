// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import { useState } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { cn } from '@/lib/utils';
import { useAuth } from '@/hooks/use-auth';
import { useLibrary } from '@/hooks/use-library';
import {
  Home,
  Search,
  Library,
  Radio,
  Heart,
  ListMusic,
  Users,
  Mic2,
  TrendingUp,
  Calendar,
  Plus,
  ChevronLeft,
  ChevronRight,
  Music2,
  Podcast,
  Sparkles,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import { CreatePlaylistDialog } from '@/components/dialogs/create-playlist';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';

const mainNavItems = [
  { href: '/', label: 'Home', icon: Home },
  { href: '/search', label: 'Search', icon: Search },
  { href: '/library', label: 'Your Library', icon: Library },
];

const discoverItems = [
  { href: '/discover', label: 'Discover', icon: Sparkles },
  { href: '/trending', label: 'Trending', icon: TrendingUp },
  { href: '/new-releases', label: 'New Releases', icon: Calendar },
  { href: '/genres', label: 'Genres', icon: Music2 },
];

const browseItems = [
  { href: '/artists', label: 'Artists', icon: Mic2 },
  { href: '/podcasts', label: 'Podcasts', icon: Podcast },
  { href: '/radio', label: 'Radio', icon: Radio },
  { href: '/live', label: 'Live Now', icon: Users },
];

export function Sidebar() {
  const pathname = usePathname();
  const { isAuthenticated, user } = useAuth();
  const { playlists, likedSongs } = useLibrary();
  const [collapsed, setCollapsed] = useState(false);
  const [showCreatePlaylist, setShowCreatePlaylist] = useState(false);

  return (
    <>
      <aside
        className={cn(
          'flex h-screen flex-col border-r bg-card transition-all duration-300',
          collapsed ? 'w-16' : 'w-64'
        )}
      >
        {/* Logo */}
        <div className="flex h-16 items-center justify-between px-4">
          {!collapsed && (
            <Link href="/" className="flex items-center gap-2">
              <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br from-violet-500 to-purple-600">
                <Music2 className="h-5 w-5 text-white" />
              </div>
              <span className="text-xl font-bold">Mycelix</span>
            </Link>
          )}
          <Button
            variant="ghost"
            size="icon"
            onClick={() => setCollapsed(!collapsed)}
            className="h-8 w-8"
          >
            {collapsed ? (
              <ChevronRight className="h-4 w-4" />
            ) : (
              <ChevronLeft className="h-4 w-4" />
            )}
          </Button>
        </div>

        <ScrollArea className="flex-1 px-2">
          {/* Main Navigation */}
          <nav className="space-y-1 py-2">
            {mainNavItems.map((item) => (
              <NavItem
                key={item.href}
                href={item.href}
                label={item.label}
                icon={item.icon}
                active={pathname === item.href}
                collapsed={collapsed}
              />
            ))}
          </nav>

          <Separator className="my-2" />

          {/* Discover */}
          {!collapsed && (
            <div className="py-2">
              <h3 className="mb-2 px-3 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                Discover
              </h3>
              <nav className="space-y-1">
                {discoverItems.map((item) => (
                  <NavItem
                    key={item.href}
                    href={item.href}
                    label={item.label}
                    icon={item.icon}
                    active={pathname === item.href}
                    collapsed={collapsed}
                  />
                ))}
              </nav>
            </div>
          )}

          {/* Browse */}
          {!collapsed && (
            <div className="py-2">
              <h3 className="mb-2 px-3 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                Browse
              </h3>
              <nav className="space-y-1">
                {browseItems.map((item) => (
                  <NavItem
                    key={item.href}
                    href={item.href}
                    label={item.label}
                    icon={item.icon}
                    active={pathname === item.href}
                    collapsed={collapsed}
                  />
                ))}
              </nav>
            </div>
          )}

          <Separator className="my-2" />

          {/* User Playlists */}
          {isAuthenticated && (
            <div className="py-2">
              {!collapsed && (
                <div className="mb-2 flex items-center justify-between px-3">
                  <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                    Your Playlists
                  </h3>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-6 w-6"
                    onClick={() => setShowCreatePlaylist(true)}
                  >
                    <Plus className="h-4 w-4" />
                  </Button>
                </div>
              )}

              <nav className="space-y-1">
                {/* Liked Songs */}
                <NavItem
                  href="/library/liked"
                  label="Liked Songs"
                  icon={Heart}
                  active={pathname === '/library/liked'}
                  collapsed={collapsed}
                  badge={likedSongs?.length}
                />

                {/* User Playlists */}
                {playlists?.slice(0, 10).map((playlist) => (
                  <Link
                    key={playlist.id}
                    href={`/playlist/${playlist.id}`}
                    className={cn(
                      'flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition-colors',
                      pathname === `/playlist/${playlist.id}`
                        ? 'bg-accent text-accent-foreground'
                        : 'text-muted-foreground hover:bg-accent/50 hover:text-foreground'
                    )}
                  >
                    {collapsed ? (
                      <ListMusic className="h-5 w-5" />
                    ) : (
                      <>
                        <Avatar className="h-8 w-8 rounded">
                          <AvatarImage src={playlist.coverUrl} />
                          <AvatarFallback className="rounded bg-gradient-to-br from-violet-500/20 to-purple-600/20">
                            <ListMusic className="h-4 w-4" />
                          </AvatarFallback>
                        </Avatar>
                        <span className="truncate">{playlist.name}</span>
                      </>
                    )}
                  </Link>
                ))}

                {!collapsed && playlists && playlists.length > 10 && (
                  <Link
                    href="/library/playlists"
                    className="block px-3 py-2 text-sm text-muted-foreground hover:text-foreground"
                  >
                    Show all ({playlists.length})
                  </Link>
                )}
              </nav>
            </div>
          )}
        </ScrollArea>

        {/* User Profile */}
        {isAuthenticated && user && (
          <div className="border-t p-2">
            <Link
              href="/profile"
              className={cn(
                'flex items-center gap-3 rounded-lg px-3 py-2 transition-colors hover:bg-accent',
                collapsed && 'justify-center px-0'
              )}
            >
              <Avatar className="h-8 w-8">
                <AvatarImage src={user.avatarUrl} />
                <AvatarFallback>
                  {user.displayName?.charAt(0) || 'U'}
                </AvatarFallback>
              </Avatar>
              {!collapsed && (
                <div className="flex-1 truncate">
                  <p className="truncate text-sm font-medium">{user.displayName}</p>
                  <p className="truncate text-xs text-muted-foreground">
                    {user.subscription?.tier || 'Free'}
                  </p>
                </div>
              )}
            </Link>
          </div>
        )}
      </aside>

      <CreatePlaylistDialog
        open={showCreatePlaylist}
        onOpenChange={setShowCreatePlaylist}
      />
    </>
  );
}

interface NavItemProps {
  href: string;
  label: string;
  icon: React.ComponentType<{ className?: string }>;
  active?: boolean;
  collapsed?: boolean;
  badge?: number;
}

function NavItem({ href, label, icon: Icon, active, collapsed, badge }: NavItemProps) {
  return (
    <Link
      href={href}
      className={cn(
        'flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition-colors',
        active
          ? 'bg-accent text-accent-foreground'
          : 'text-muted-foreground hover:bg-accent/50 hover:text-foreground',
        collapsed && 'justify-center px-0'
      )}
      title={collapsed ? label : undefined}
    >
      <Icon className="h-5 w-5 flex-shrink-0" />
      {!collapsed && (
        <>
          <span className="flex-1">{label}</span>
          {badge !== undefined && badge > 0 && (
            <span className="rounded-full bg-primary/10 px-2 py-0.5 text-xs font-medium text-primary">
              {badge > 99 ? '99+' : badge}
            </span>
          )}
        </>
      )}
    </Link>
  );
}
