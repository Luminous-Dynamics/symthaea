// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import { writable, derived } from 'svelte/store';
import { browser } from '$app/environment';

const STORAGE_KEY = 'mycelix_favorites';

function loadFavorites(): string[] {
  if (!browser) return [];
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    return stored ? JSON.parse(stored) : [];
  } catch (e) {
    console.error('Failed to load favorites', e);
    return [];
  }
}

function saveFavorites(ids: string[]) {
  if (!browser) return;
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(ids));
  } catch (e) {
    console.error('Failed to save favorites', e);
  }
}

function createFavoritesStore() {
  const { subscribe, update, set } = writable<string[]>(loadFavorites());

  if (browser) {
    subscribe((ids) => saveFavorites(ids));
  }

  return {
    subscribe,
    toggle: (id: string) => {
      update((ids) => {
        const exists = ids.includes(id);
        return exists ? ids.filter((v) => v !== id) : [...ids, id];
      });
    },
    add: (id: string) => {
      update((ids) => (ids.includes(id) ? ids : [...ids, id]));
    },
    remove: (id: string) => {
      update((ids) => ids.filter((v) => v !== id));
    },
    clear: () => set([]),
  };
}

export const favorites = createFavoritesStore();
export const favoritesSet = derived(favorites, ($favorites) => new Set($favorites));
export const favoritesCount = derived(favorites, ($favorites) => $favorites.length);
