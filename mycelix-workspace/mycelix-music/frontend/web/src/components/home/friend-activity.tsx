// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import Image from 'next/image';
import Link from 'next/link';
import { formatDistanceToNow } from 'date-fns';
import { Music, Radio, Users, MoreHorizontal } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import { ScrollArea } from '@/components/ui/scroll-area';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { useFriendActivity } from '@/hooks/use-friend-activity';
import { cn } from '@/lib/utils';

export function FriendActivity() {
  const { data: activities, isLoading } = useFriendActivity();

  if (isLoading) {
    return (
      <div className="space-y-4">
        <h2 className="text-lg font-semibold">Friend Activity</h2>
        {[...Array(5)].map((_, i) => (
          <div key={i} className="flex animate-pulse gap-3">
            <div className="h-10 w-10 rounded-full bg-muted" />
            <div className="flex-1 space-y-2">
              <div className="h-3 w-3/4 rounded bg-muted" />
              <div className="h-3 w-1/2 rounded bg-muted" />
            </div>
          </div>
        ))}
      </div>
    );
  }

  const friendActivities = activities || [];

  return (
    <div className="flex h-full flex-col">
      <div className="mb-4 flex items-center justify-between">
        <h2 className="text-lg font-semibold">Friend Activity</h2>
        <Button variant="ghost" size="sm" asChild>
          <Link href="/social/friends">
            <Users className="mr-2 h-4 w-4" />
            Friends
          </Link>
        </Button>
      </div>

      {friendActivities.length === 0 ? (
        <div className="flex flex-1 flex-col items-center justify-center text-center">
          <Users className="h-12 w-12 text-muted-foreground" />
          <p className="mt-4 text-sm text-muted-foreground">
            Connect with friends to see what they're listening to
          </p>
          <Button variant="outline" size="sm" className="mt-4" asChild>
            <Link href="/social/find-friends">Find Friends</Link>
          </Button>
        </div>
      ) : (
        <ScrollArea className="flex-1">
          <div className="space-y-4">
            {friendActivities.map((activity) => (
              <div
                key={activity.id}
                className="group flex gap-3"
              >
                {/* User Avatar */}
                <div className="relative">
                  <Avatar className="h-10 w-10">
                    <AvatarImage src={activity.user.avatarUrl} />
                    <AvatarFallback>
                      {activity.user.displayName.charAt(0)}
                    </AvatarFallback>
                  </Avatar>
                  {/* Online Status */}
                  <span
                    className={cn(
                      'absolute -bottom-0.5 -right-0.5 h-3 w-3 rounded-full border-2 border-card',
                      activity.user.isOnline ? 'bg-green-500' : 'bg-muted'
                    )}
                  />
                </div>

                {/* Activity Content */}
                <div className="min-w-0 flex-1">
                  <Link
                    href={`/user/${activity.user.id}`}
                    className="text-sm font-medium hover:underline"
                  >
                    {activity.user.displayName}
                  </Link>

                  <div className="mt-1 flex items-center gap-2">
                    {activity.type === 'listening' ? (
                      <Music className="h-3 w-3 text-green-500" />
                    ) : activity.type === 'live' ? (
                      <Radio className="h-3 w-3 text-red-500" />
                    ) : null}

                    <span className="text-xs text-muted-foreground">
                      {activity.type === 'listening' ? 'Listening to' :
                       activity.type === 'live' ? 'Watching live' :
                       'Was listening to'}
                    </span>
                  </div>

                  {activity.track && (
                    <div className="mt-2 flex gap-2 rounded-lg bg-muted/50 p-2 transition-colors group-hover:bg-muted">
                      <div className="relative h-10 w-10 flex-shrink-0 overflow-hidden rounded">
                        <Image
                          src={activity.track.coverUrl || '/images/default-cover.png'}
                          alt={activity.track.title}
                          fill
                          className="object-cover"
                        />
                      </div>
                      <div className="min-w-0 flex-1">
                        <Link
                          href={`/track/${activity.track.id}`}
                          className="block truncate text-sm font-medium hover:underline"
                        >
                          {activity.track.title}
                        </Link>
                        <Link
                          href={`/artist/${activity.track.artist.id}`}
                          className="block truncate text-xs text-muted-foreground hover:underline"
                        >
                          {activity.track.artist.name}
                        </Link>
                      </div>
                    </div>
                  )}

                  <p className="mt-1 text-xs text-muted-foreground">
                    {formatDistanceToNow(new Date(activity.timestamp), { addSuffix: true })}
                  </p>
                </div>

                {/* Actions */}
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-6 w-6 opacity-0 group-hover:opacity-100"
                    >
                      <MoreHorizontal className="h-4 w-4" />
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end">
                    <DropdownMenuItem>Play this track</DropdownMenuItem>
                    <DropdownMenuItem>Add to queue</DropdownMenuItem>
                    <DropdownMenuItem>View profile</DropdownMenuItem>
                    <DropdownMenuItem>Hide activity</DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              </div>
            ))}
          </div>
        </ScrollArea>
      )}
    </div>
  );
}
