import adapter from '@sveltejs/adapter-static';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

/** @type {import('@sveltejs/kit').Config} */
const config = {
	preprocess: vitePreprocess(),
	kit: {
		adapter: adapter({
			fallback: 'index.html' // SPA mode — all routes handled client-side
		}),
		serviceWorker: {
			register: false // We register manually in +layout.svelte for more control
		}
	}
};

export default config;
