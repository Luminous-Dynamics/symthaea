// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

export default defineConfig({
	plugins: [sveltekit()],

	// Tauri expects a fixed port
	server: {
		port: 1420,
		strictPort: true,
	},

	// Build configuration for Tauri
	build: {
		target: process.env.TAURI_PLATFORM === 'windows' ? 'chrome105' : 'safari14',
		minify: !process.env.TAURI_DEBUG ? 'esbuild' : false,
		sourcemap: !!process.env.TAURI_DEBUG,
	},

	// Optimizations
	optimizeDeps: {
		exclude: ['@xenova/transformers'],
	},
});
