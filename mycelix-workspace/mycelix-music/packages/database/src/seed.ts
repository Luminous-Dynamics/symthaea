// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Database Seed Script
 *
 * Populates the database with sample data for development.
 */

import { PrismaClient, UserRole, SongStatus, AlbumType } from '@prisma/client';

const prisma = new PrismaClient();

async function main() {
  console.log('Seeding database...');

  // Create sample artists
  const artists = await Promise.all([
    prisma.user.upsert({
      where: { address: '0x1111111111111111111111111111111111111111' },
      update: {},
      create: {
        address: '0x1111111111111111111111111111111111111111',
        name: 'Ethereal Waves',
        bio: 'Ambient electronic producer exploring the spaces between sound and silence.',
        role: UserRole.ARTIST,
        verified: true,
        avatar: 'https://api.dicebear.com/7.x/shapes/svg?seed=ethereal',
        twitter: '@etherealwaves',
      },
    }),
    prisma.user.upsert({
      where: { address: '0x2222222222222222222222222222222222222222' },
      update: {},
      create: {
        address: '0x2222222222222222222222222222222222222222',
        name: 'Neon Pulse',
        bio: 'Synthwave artist from the future. Retro vibes, modern production.',
        role: UserRole.ARTIST,
        verified: true,
        avatar: 'https://api.dicebear.com/7.x/shapes/svg?seed=neon',
      },
    }),
    prisma.user.upsert({
      where: { address: '0x3333333333333333333333333333333333333333' },
      update: {},
      create: {
        address: '0x3333333333333333333333333333333333333333',
        name: 'Forest Protocol',
        bio: 'Organic electronic music inspired by nature and technology.',
        role: UserRole.ARTIST,
        verified: false,
        avatar: 'https://api.dicebear.com/7.x/shapes/svg?seed=forest',
      },
    }),
  ]);

  console.log(`Created ${artists.length} artists`);

  // Create sample listeners
  const listeners = await Promise.all([
    prisma.user.upsert({
      where: { address: '0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa' },
      update: {},
      create: {
        address: '0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
        name: 'MusicLover42',
        role: UserRole.LISTENER,
        avatar: 'https://api.dicebear.com/7.x/avataaars/svg?seed=music42',
      },
    }),
    prisma.user.upsert({
      where: { address: '0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb' },
      update: {},
      create: {
        address: '0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb',
        name: 'ChillVibes',
        role: UserRole.LISTENER,
        avatar: 'https://api.dicebear.com/7.x/avataaars/svg?seed=chill',
      },
    }),
  ]);

  console.log(`Created ${listeners.length} listeners`);

  // Create songs for each artist
  const genres = ['Electronic', 'Ambient', 'Synthwave', 'Downtempo', 'House', 'Techno'];
  const songs = [];

  for (const artist of artists) {
    const artistSongs = await Promise.all(
      Array.from({ length: 5 }, (_, i) =>
        prisma.song.create({
          data: {
            title: `${['Digital', 'Cosmic', 'Neon', 'Crystal', 'Quantum'][i]} ${['Dreams', 'Waves', 'Pulse', 'Light', 'Flow'][i]}`,
            slug: `${artist.name?.toLowerCase().replace(/\s/g, '-')}-song-${i + 1}`,
            artistId: artist.id,
            duration: 180 + Math.floor(Math.random() * 180),
            genre: genres[Math.floor(Math.random() * genres.length)],
            tags: ['electronic', 'chill', 'atmospheric'].slice(0, Math.floor(Math.random() * 3) + 1),
            playCount: Math.floor(Math.random() * 10000),
            likeCount: Math.floor(Math.random() * 500),
            status: SongStatus.PUBLISHED,
            originalFile: `ipfs://Qm${Math.random().toString(36).substring(2, 15)}`,
            coverArt: `https://picsum.photos/seed/${artist.id}-${i}/400`,
            releaseDate: new Date(Date.now() - Math.random() * 365 * 24 * 60 * 60 * 1000),
          },
        })
      )
    );
    songs.push(...artistSongs);
  }

  console.log(`Created ${songs.length} songs`);

  // Create sample playlists
  const playlists = await Promise.all([
    prisma.playlist.create({
      data: {
        name: 'Late Night Coding',
        slug: 'late-night-coding',
        description: 'Perfect beats for those midnight programming sessions',
        creatorId: listeners[0].id,
        isPublic: true,
        tracks: {
          create: songs.slice(0, 5).map((song, i) => ({
            songId: song.id,
            position: i,
          })),
        },
      },
    }),
    prisma.playlist.create({
      data: {
        name: 'Morning Energy',
        slug: 'morning-energy',
        description: 'Start your day with positive vibes',
        creatorId: listeners[1].id,
        isPublic: true,
        tracks: {
          create: songs.slice(5, 10).map((song, i) => ({
            songId: song.id,
            position: i,
          })),
        },
      },
    }),
  ]);

  console.log(`Created ${playlists.length} playlists`);

  // Create some follows
  await prisma.follow.createMany({
    data: [
      { followerId: listeners[0].id, followingId: artists[0].id },
      { followerId: listeners[0].id, followingId: artists[1].id },
      { followerId: listeners[1].id, followingId: artists[0].id },
      { followerId: listeners[1].id, followingId: artists[2].id },
    ],
    skipDuplicates: true,
  });

  console.log('Created follows');

  // Create some likes
  await prisma.songLike.createMany({
    data: songs.slice(0, 8).map((song, i) => ({
      userId: listeners[i % 2].id,
      songId: song.id,
    })),
    skipDuplicates: true,
  });

  console.log('Created likes');

  // Create sample plays
  const plays = [];
  for (const song of songs) {
    const playCount = Math.floor(Math.random() * 20) + 5;
    for (let i = 0; i < playCount; i++) {
      plays.push({
        songId: song.id,
        userId: Math.random() > 0.3 ? listeners[Math.floor(Math.random() * listeners.length)].id : null,
        duration: Math.floor(song.duration * (0.3 + Math.random() * 0.7)),
        completed: Math.random() > 0.4,
        playedAt: new Date(Date.now() - Math.random() * 30 * 24 * 60 * 60 * 1000),
      });
    }
  }

  await prisma.play.createMany({ data: plays });
  console.log(`Created ${plays.length} plays`);

  console.log('Seeding complete!');
}

main()
  .catch((e) => {
    console.error(e);
    process.exit(1);
  })
  .finally(async () => {
    await prisma.$disconnect();
  });
