import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vitest/config';

export default defineConfig({
	plugins: [sveltekit()],
	test: {
		include: ['src/**/*.{test,spec}.{js,ts}'],
		environment: 'jsdom',
		globals: true
	},
	server: {
		port: 5173,
		strictPort: false,
		host: true
	},
	build: {
		target: 'es2020',
		sourcemap: true
	}
});
