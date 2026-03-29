// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import { browser } from '$app/environment';

const STORAGE_KEY = 'mycelix_holochain_url';

export interface HolochainUrlOption {
  label: string;
  url: string;
}

export function getAvailableUrls(): HolochainUrlOption[] {
  const env = import.meta.env as Record<string, string | undefined>;
  const urls: HolochainUrlOption[] = [];

  if (env.VITE_HOLOCHAIN_URL_DEV) {
    urls.push({ label: 'Dev', url: env.VITE_HOLOCHAIN_URL_DEV });
  }
  if (env.VITE_HOLOCHAIN_URL_STAGE) {
    urls.push({ label: 'Stage', url: env.VITE_HOLOCHAIN_URL_STAGE });
  }
  if (env.VITE_HOLOCHAIN_URL_PROD) {
    urls.push({ label: 'Prod', url: env.VITE_HOLOCHAIN_URL_PROD });
  }
  if (env.VITE_HOLOCHAIN_URL) {
    urls.push({ label: 'Default', url: env.VITE_HOLOCHAIN_URL });
  }

  // Ensure at least one fallback
  if (urls.length === 0) {
    urls.push({ label: 'Localhost', url: 'ws://localhost:8888' });
  }

  // Deduplicate by URL
  const seen = new Set<string>();
  return urls.filter((opt) => {
    if (seen.has(opt.url)) return false;
    seen.add(opt.url);
    return true;
  });
}

export function loadSelectedUrl(defaultUrl: string): string {
  if (!browser) return defaultUrl;
  const stored = localStorage.getItem(STORAGE_KEY);
  return stored || defaultUrl;
}

export function saveSelectedUrl(url: string): void {
  if (!browser) return;
  localStorage.setItem(STORAGE_KEY, url);
}
