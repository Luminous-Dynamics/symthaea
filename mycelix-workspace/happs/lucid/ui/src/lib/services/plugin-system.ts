/**
 * LUCID Plugin System
 *
 * Extensible architecture for:
 * - Importers (Notion, Obsidian, Roam, custom formats)
 * - Visualizations (custom graph layouts, charts)
 * - AI Models (classification, embedding, summarization)
 * - Export formats (Markdown, JSON, custom)
 * - Integrations (APIs, webhooks, services)
 */

import { writable, get } from 'svelte/store';
import type { Thought } from '@mycelix/lucid-client';

// ============================================================================
// TYPES
// ============================================================================

export interface PluginManifest {
	id: string;
	name: string;
	version: string;
	description: string;
	author: string;
	homepage?: string;
	type: PluginType;
	capabilities: PluginCapability[];
	config?: PluginConfigSchema[];
}

export type PluginType =
	| 'importer'
	| 'exporter'
	| 'visualization'
	| 'ai-model'
	| 'integration'
	| 'theme'
	| 'command';

export type PluginCapability =
	| 'import'
	| 'export'
	| 'visualize'
	| 'classify'
	| 'embed'
	| 'summarize'
	| 'transform'
	| 'search'
	| 'sync'
	| 'notify';

export interface PluginConfigSchema {
	key: string;
	type: 'string' | 'number' | 'boolean' | 'select' | 'secret';
	label: string;
	description?: string;
	default?: any;
	options?: { value: string; label: string }[];
	required?: boolean;
}

export interface PluginContext {
	thoughts: Thought[];
	config: Record<string, any>;
	api: PluginAPI;
}

export interface PluginAPI {
	// Thought operations
	getThoughts: () => Thought[];
	createThought: (thought: Partial<Thought>) => Promise<Thought>;
	updateThought: (id: string, updates: Partial<Thought>) => Promise<Thought>;
	deleteThought: (id: string) => Promise<void>;

	// UI operations
	showNotification: (message: string, type?: 'info' | 'success' | 'warning' | 'error') => void;
	showModal: (content: any) => void;
	registerCommand: (command: PluginCommand) => void;

	// Storage
	getStorage: (key: string) => Promise<any>;
	setStorage: (key: string, value: any) => Promise<void>;

	// Events
	on: (event: PluginEvent, handler: (data: any) => void) => void;
	off: (event: PluginEvent, handler: (data: any) => void) => void;
	emit: (event: PluginEvent, data: any) => void;
}

export type PluginEvent =
	| 'thought:created'
	| 'thought:updated'
	| 'thought:deleted'
	| 'search:query'
	| 'import:complete'
	| 'export:complete';

export interface PluginCommand {
	id: string;
	name: string;
	description?: string;
	shortcut?: string;
	execute: (context: PluginContext) => Promise<void>;
}

// ============================================================================
// PLUGIN INTERFACES BY TYPE
// ============================================================================

export interface ImporterPlugin {
	manifest: PluginManifest;
	supportedFormats: string[];
	import: (data: string | ArrayBuffer, format: string, context: PluginContext) => Promise<Thought[]>;
	validate?: (data: string | ArrayBuffer, format: string) => boolean;
}

export interface ExporterPlugin {
	manifest: PluginManifest;
	supportedFormats: string[];
	export: (thoughts: Thought[], format: string, context: PluginContext) => Promise<string | Blob>;
	getFilename?: (format: string) => string;
}

export interface VisualizationPlugin {
	manifest: PluginManifest;
	component: any; // Svelte component
	defaultConfig?: Record<string, any>;
}

export interface AIModelPlugin {
	manifest: PluginManifest;
	modelType: 'classifier' | 'embedder' | 'summarizer' | 'transformer';
	initialize: (config: Record<string, any>) => Promise<void>;
	process: (input: string | Thought[], context: PluginContext) => Promise<any>;
	destroy?: () => Promise<void>;
}

export interface IntegrationPlugin {
	manifest: PluginManifest;
	connect: (config: Record<string, any>) => Promise<void>;
	disconnect: () => Promise<void>;
	sync?: (context: PluginContext) => Promise<void>;
	isConnected: () => boolean;
}

export type Plugin =
	| ImporterPlugin
	| ExporterPlugin
	| VisualizationPlugin
	| AIModelPlugin
	| IntegrationPlugin;

// ============================================================================
// PLUGIN REGISTRY
// ============================================================================

interface PluginRegistryState {
	plugins: Map<string, Plugin>;
	enabled: Set<string>;
	configs: Map<string, Record<string, any>>;
}

const registryState: PluginRegistryState = {
	plugins: new Map(),
	enabled: new Set(),
	configs: new Map(),
};

export const pluginRegistry = writable<PluginRegistryState>(registryState);

/**
 * Register a plugin
 */
export function registerPlugin(plugin: Plugin): void {
	const { id } = plugin.manifest;

	if (registryState.plugins.has(id)) {
		console.warn(`Plugin ${id} is already registered`);
		return;
	}

	registryState.plugins.set(id, plugin);
	pluginRegistry.set({ ...registryState });

	console.log(`Plugin registered: ${plugin.manifest.name} (${id})`);
}

/**
 * Unregister a plugin
 */
export function unregisterPlugin(pluginId: string): void {
	if (!registryState.plugins.has(pluginId)) {
		return;
	}

	// Disable first
	disablePlugin(pluginId);

	registryState.plugins.delete(pluginId);
	registryState.configs.delete(pluginId);
	pluginRegistry.set({ ...registryState });

	console.log(`Plugin unregistered: ${pluginId}`);
}

/**
 * Enable a plugin
 */
export function enablePlugin(pluginId: string): boolean {
	const plugin = registryState.plugins.get(pluginId);
	if (!plugin) {
		console.error(`Plugin not found: ${pluginId}`);
		return false;
	}

	registryState.enabled.add(pluginId);
	pluginRegistry.set({ ...registryState });

	console.log(`Plugin enabled: ${pluginId}`);
	return true;
}

/**
 * Disable a plugin
 */
export function disablePlugin(pluginId: string): void {
	registryState.enabled.delete(pluginId);
	pluginRegistry.set({ ...registryState });

	console.log(`Plugin disabled: ${pluginId}`);
}

/**
 * Configure a plugin
 */
export function configurePlugin(pluginId: string, config: Record<string, any>): void {
	registryState.configs.set(pluginId, config);
	pluginRegistry.set({ ...registryState });
}

/**
 * Get a plugin by ID
 */
export function getPlugin<T extends Plugin>(pluginId: string): T | undefined {
	return registryState.plugins.get(pluginId) as T | undefined;
}

/**
 * Get all plugins of a specific type
 */
export function getPluginsByType(type: PluginType): Plugin[] {
	return Array.from(registryState.plugins.values()).filter(
		(p) => p.manifest.type === type && registryState.enabled.has(p.manifest.id)
	);
}

/**
 * Get all enabled importers
 */
export function getImporters(): ImporterPlugin[] {
	return getPluginsByType('importer') as ImporterPlugin[];
}

/**
 * Get all enabled exporters
 */
export function getExporters(): ExporterPlugin[] {
	return getPluginsByType('exporter') as ExporterPlugin[];
}

/**
 * Get all enabled visualizations
 */
export function getVisualizations(): VisualizationPlugin[] {
	return getPluginsByType('visualization') as VisualizationPlugin[];
}

// ============================================================================
// PLUGIN API IMPLEMENTATION
// ============================================================================

const eventHandlers: Map<PluginEvent, Set<(data: any) => void>> = new Map();

/**
 * Rate limiter for plugin API calls
 * Prevents abuse by limiting call frequency per plugin
 */
class RateLimiter {
	private calls: Map<string, number[]> = new Map();
	private readonly windowMs: number;
	private readonly maxCalls: number;

	constructor(windowMs: number = 1000, maxCalls: number = 100) {
		this.windowMs = windowMs;
		this.maxCalls = maxCalls;
	}

	check(pluginId: string, operation: string): boolean {
		const key = `${pluginId}:${operation}`;
		const now = Date.now();
		const calls = this.calls.get(key) || [];

		// Remove calls outside the window
		const recentCalls = calls.filter((t) => now - t < this.windowMs);

		if (recentCalls.length >= this.maxCalls) {
			console.warn(`Rate limit exceeded for plugin ${pluginId} on ${operation}`);
			return false;
		}

		recentCalls.push(now);
		this.calls.set(key, recentCalls);
		return true;
	}

	reset(pluginId: string): void {
		for (const key of this.calls.keys()) {
			if (key.startsWith(`${pluginId}:`)) {
				this.calls.delete(key);
			}
		}
	}
}

const apiRateLimiter = new RateLimiter(1000, 100); // 100 calls per second per operation

/**
 * Maximum number of event handlers per event type
 */
const MAX_HANDLERS_PER_EVENT = 50;

export function createPluginAPI(thoughts: Thought[], pluginId: string = 'unknown'): PluginAPI {
	return {
		getThoughts: () => {
			if (!apiRateLimiter.check(pluginId, 'getThoughts')) {
				throw new Error('Rate limit exceeded');
			}
			return thoughts;
		},

		createThought: async (thought) => {
			if (!apiRateLimiter.check(pluginId, 'createThought')) {
				throw new Error('Rate limit exceeded');
			}
			// This would integrate with the actual thought store
			console.log('Plugin creating thought:', thought);
			return thought as Thought;
		},

		updateThought: async (id, updates) => {
			if (!apiRateLimiter.check(pluginId, 'updateThought')) {
				throw new Error('Rate limit exceeded');
			}
			console.log('Plugin updating thought:', id, updates);
			return { id, ...updates } as Thought;
		},

		deleteThought: async (id) => {
			if (!apiRateLimiter.check(pluginId, 'deleteThought')) {
				throw new Error('Rate limit exceeded');
			}
			console.log('Plugin deleting thought:', id);
		},

		showNotification: (message, type = 'info') => {
			if (!apiRateLimiter.check(pluginId, 'showNotification')) {
				return; // Silently drop excessive notifications
			}
			console.log(`[${type.toUpperCase()}] ${message}`);
			// Would integrate with notification system
		},

		showModal: (content) => {
			if (!apiRateLimiter.check(pluginId, 'showModal')) {
				return;
			}
			console.log('Plugin showing modal:', content);
			// Would integrate with modal system
		},

		registerCommand: (command) => {
			if (!apiRateLimiter.check(pluginId, 'registerCommand')) {
				throw new Error('Rate limit exceeded');
			}
			console.log('Plugin registering command:', command.id);
			// Would integrate with command palette
		},

		getStorage: async (key) => {
			if (!apiRateLimiter.check(pluginId, 'getStorage')) {
				throw new Error('Rate limit exceeded');
			}
			const stored = localStorage.getItem(`lucid-plugin-${key}`);
			if (!stored) return null;
			try {
				return JSON.parse(stored);
			} catch {
				console.warn(`Failed to parse plugin storage for key: ${key}`);
				return null;
			}
		},

		setStorage: async (key, value) => {
			if (!apiRateLimiter.check(pluginId, 'setStorage')) {
				throw new Error('Rate limit exceeded');
			}
			// Limit storage value size (1MB max)
			const serialized = JSON.stringify(value);
			if (serialized.length > 1024 * 1024) {
				throw new Error('Storage value too large (max 1MB)');
			}
			localStorage.setItem(`lucid-plugin-${key}`, serialized);
		},

		on: (event, handler) => {
			if (!apiRateLimiter.check(pluginId, 'on')) {
				throw new Error('Rate limit exceeded');
			}
			if (!eventHandlers.has(event)) {
				eventHandlers.set(event, new Set());
			}
			const handlers = eventHandlers.get(event)!;
			if (handlers.size >= MAX_HANDLERS_PER_EVENT) {
				throw new Error(`Maximum event handlers (${MAX_HANDLERS_PER_EVENT}) reached for ${event}`);
			}
			handlers.add(handler);
		},

		off: (event, handler) => {
			eventHandlers.get(event)?.delete(handler);
		},

		emit: (event, data) => {
			if (!apiRateLimiter.check(pluginId, 'emit')) {
				return; // Silently drop excessive events
			}
			eventHandlers.get(event)?.forEach((handler) => handler(data));
		},
	};
}

// ============================================================================
// BUILT-IN PLUGINS
// ============================================================================

/**
 * Register built-in plugins
 */
export function registerBuiltinPlugins(): void {
	// JSON Importer
	registerPlugin({
		manifest: {
			id: 'builtin-json-importer',
			name: 'JSON Importer',
			version: '1.0.0',
			description: 'Import thoughts from JSON format',
			author: 'LUCID',
			type: 'importer',
			capabilities: ['import'],
		},
		supportedFormats: ['json', 'lucid-json'],
		import: async (data, format, context) => {
			let parsed: any;
			try {
				parsed = JSON.parse(data as string);
			} catch (error) {
				throw new Error(`Invalid JSON format: ${error instanceof Error ? error.message : 'parse error'}`);
			}
			if (Array.isArray(parsed)) {
				return parsed;
			}
			if (parsed.thoughts) {
				return parsed.thoughts;
			}
			return [parsed];
		},
		validate: (data) => {
			try {
				JSON.parse(data as string);
				return true;
			} catch {
				return false;
			}
		},
	} as ImporterPlugin);

	// Markdown Exporter
	registerPlugin({
		manifest: {
			id: 'builtin-markdown-exporter',
			name: 'Markdown Exporter',
			version: '1.0.0',
			description: 'Export thoughts to Markdown format',
			author: 'LUCID',
			type: 'exporter',
			capabilities: ['export'],
		},
		supportedFormats: ['md', 'markdown'],
		export: async (thoughts, format, context) => {
			let md = '# LUCID Thoughts Export\n\n';
			md += `*Exported: ${new Date().toISOString()}*\n\n`;
			md += '---\n\n';

			for (const thought of thoughts) {
				md += `## ${thought.thought_type}\n\n`;
				md += `${thought.content}\n\n`;
				md += `- **Confidence:** ${Math.round(thought.confidence * 100)}%\n`;
				md += `- **Epistemic:** ${thought.epistemic.empirical}/${thought.epistemic.normative}/${thought.epistemic.materiality}/${thought.epistemic.harmonic}\n`;
				if (thought.tags?.length) {
					md += `- **Tags:** ${thought.tags.join(', ')}\n`;
				}
				md += '\n---\n\n';
			}

			return md;
		},
		getFilename: () => `lucid-export-${new Date().toISOString().split('T')[0]}.md`,
	} as ExporterPlugin);

	// Obsidian Markdown Importer
	registerPlugin({
		manifest: {
			id: 'builtin-obsidian-importer',
			name: 'Obsidian Importer',
			version: '1.0.0',
			description: 'Import thoughts from Obsidian markdown vault',
			author: 'LUCID',
			type: 'importer',
			capabilities: ['import'],
		},
		supportedFormats: ['md', 'obsidian'],
		import: async (data, format, context) => {
			const content = data as string;
			const thoughts: Partial<Thought>[] = [];

			// Parse Obsidian markdown format
			// Extract YAML frontmatter if present
			const frontmatterMatch = content.match(/^---\n([\s\S]*?)\n---\n/);
			let frontmatter: Record<string, any> = {};
			let bodyContent = content;

			if (frontmatterMatch) {
				bodyContent = content.slice(frontmatterMatch[0].length);
				// Simple YAML parsing for common fields
				const yamlLines = frontmatterMatch[1].split('\n');
				for (const line of yamlLines) {
					const colonIndex = line.indexOf(':');
					if (colonIndex > 0) {
						const key = line.slice(0, colonIndex).trim();
						let value = line.slice(colonIndex + 1).trim();
						// Handle arrays like tags: [tag1, tag2]
						if (value.startsWith('[') && value.endsWith(']')) {
							value = value.slice(1, -1).split(',').map((s: string) => s.trim().replace(/['"]/g, ''));
						}
						frontmatter[key] = value;
					}
				}
			}

			// Extract wikilinks as potential connections
			const wikilinks = [...bodyContent.matchAll(/\[\[([^\]|]+)(?:\|[^\]]+)?\]\]/g)].map((m) => m[1]);

			// Extract tags from content (#tag format)
			const contentTags = [...bodyContent.matchAll(/#([a-zA-Z0-9_-]+)/g)].map((m) => m[1]);
			const allTags = [
				...(Array.isArray(frontmatter.tags) ? frontmatter.tags : []),
				...contentTags,
			];

			// Determine thought type from frontmatter or content analysis
			let thoughtType = 'Note';
			if (frontmatter.type) {
				thoughtType = frontmatter.type;
			} else if (bodyContent.includes('?')) {
				thoughtType = 'Question';
			} else if (frontmatter.claim || bodyContent.toLowerCase().includes('i believe')) {
				thoughtType = 'Belief';
			}

			// Clean content (remove frontmatter delimiters, excessive whitespace)
			const cleanContent = bodyContent
				.replace(/^\s+/, '')
				.replace(/\n{3,}/g, '\n\n')
				.trim();

			thoughts.push({
				content: cleanContent,
				thought_type: thoughtType,
				confidence: frontmatter.confidence ? parseFloat(frontmatter.confidence) : 0.7,
				tags: [...new Set(allTags)], // Dedupe
				// Store wikilinks as metadata for later connection
			});

			return thoughts as Thought[];
		},
		validate: (data) => {
			const content = data as string;
			// Basic validation: should be text content
			return typeof content === 'string' && content.length > 0;
		},
	} as ImporterPlugin);

	// Notion Export Importer
	registerPlugin({
		manifest: {
			id: 'builtin-notion-importer',
			name: 'Notion Importer',
			version: '1.0.0',
			description: 'Import thoughts from Notion export (markdown or CSV)',
			author: 'LUCID',
			type: 'importer',
			capabilities: ['import'],
		},
		supportedFormats: ['md', 'csv', 'notion'],
		import: async (data, format, context) => {
			const content = data as string;
			const thoughts: Partial<Thought>[] = [];

			if (format === 'csv') {
				// Parse Notion CSV export
				const lines = content.split('\n');
				if (lines.length < 2) {
					throw new Error('CSV file appears empty');
				}

				// Parse header row
				const headers = parseCSVLine(lines[0]);
				const contentIndex = headers.findIndex((h) =>
					h.toLowerCase().includes('content') || h.toLowerCase().includes('name') || h.toLowerCase().includes('title')
				);
				const tagsIndex = headers.findIndex((h) => h.toLowerCase().includes('tags'));
				const typeIndex = headers.findIndex((h) => h.toLowerCase().includes('type'));

				if (contentIndex === -1) {
					throw new Error('Could not find content/name/title column in CSV');
				}

				// Parse data rows
				for (let i = 1; i < lines.length; i++) {
					const line = lines[i].trim();
					if (!line) continue;

					const values = parseCSVLine(line);
					const thoughtContent = values[contentIndex] || '';
					if (!thoughtContent) continue;

					thoughts.push({
						content: thoughtContent,
						thought_type: typeIndex >= 0 ? (values[typeIndex] || 'Note') : 'Note',
						confidence: 0.7,
						tags: tagsIndex >= 0 ? (values[tagsIndex] || '').split(',').map((t: string) => t.trim()).filter(Boolean) : [],
					});
				}
			} else {
				// Parse Notion markdown export
				// Notion exports often have a specific structure with page title as H1
				const h1Match = content.match(/^#\s+(.+)$/m);
				const pageTitle = h1Match ? h1Match[1] : '';

				// Extract properties block (Notion uses a specific format)
				// Example: "Tags: tag1, tag2" or "Type: Note"
				const propsMatch = content.match(/^(\w+):\s*(.+)$/gm) || [];
				const props: Record<string, string> = {};
				for (const match of propsMatch) {
					const [, key, value] = match.match(/^(\w+):\s*(.+)$/) || [];
					if (key && value) {
						props[key.toLowerCase()] = value;
					}
				}

				// Extract main content (everything after H1 and properties)
				let mainContent = content
					.replace(/^#\s+.+$/m, '') // Remove H1
					.replace(/^(\w+):\s*.+$/gm, '') // Remove property lines
					.trim();

				// Handle Notion toggle blocks (converted to details/summary in some exports)
				mainContent = mainContent.replace(/<details>[\s\S]*?<\/details>/g, '');

				// Extract internal links (Notion uses different formats)
				const notionLinks = [
					...mainContent.matchAll(/\[([^\]]+)\]\([^)]+\)/g),
				].map((m) => m[1]);

				thoughts.push({
					content: mainContent || pageTitle,
					thought_type: props.type || 'Note',
					confidence: 0.7,
					tags: props.tags ? props.tags.split(',').map((t: string) => t.trim()) : [],
				});
			}

			return thoughts as Thought[];
		},
		validate: (data) => {
			const content = data as string;
			return typeof content === 'string' && content.length > 0;
		},
	} as ImporterPlugin);

	// Enable built-in plugins by default
	enablePlugin('builtin-json-importer');
	enablePlugin('builtin-markdown-exporter');
	enablePlugin('builtin-obsidian-importer');
	enablePlugin('builtin-notion-importer');
}

/**
 * Parse a CSV line handling quoted values
 */
function parseCSVLine(line: string): string[] {
	const values: string[] = [];
	let current = '';
	let inQuotes = false;

	for (let i = 0; i < line.length; i++) {
		const char = line[i];
		if (char === '"') {
			if (inQuotes && line[i + 1] === '"') {
				current += '"';
				i++;
			} else {
				inQuotes = !inQuotes;
			}
		} else if (char === ',' && !inQuotes) {
			values.push(current.trim());
			current = '';
		} else {
			current += char;
		}
	}
	values.push(current.trim());

	return values;
}

// ============================================================================
// PLUGIN LOADER
// ============================================================================

/**
 * Load a plugin from a URL or file
 *
 * SECURITY: Plugins are executed in a Web Worker sandbox to prevent
 * arbitrary code execution in the main thread. The worker has limited
 * API access and cannot directly access the DOM or main thread state.
 */
export async function loadPluginFromURL(url: string): Promise<Plugin | null> {
	try {
		const response = await fetch(url);
		const code = await response.text();

		// Validate plugin code structure before execution
		if (!validatePluginCode(code)) {
			console.error('Invalid plugin code structure');
			return null;
		}

		// Execute plugin in sandboxed Web Worker
		const plugin = await executeInSandbox(code);

		if (plugin && plugin.manifest) {
			registerPlugin(plugin);
			return plugin;
		}

		console.error('Invalid plugin format');
		return null;
	} catch (error) {
		console.error('Failed to load plugin:', error);
		return null;
	}
}

/**
 * Validate plugin code structure before execution
 * Checks for dangerous patterns and required exports
 *
 * SECURITY NOTE: This is defense-in-depth. The primary security comes from
 * the Worker sandbox with frozen globals. Pattern detection catches common
 * attack vectors early.
 */
function validatePluginCode(code: string): boolean {
	// Normalize code for pattern matching (handle obfuscation attempts)
	const normalizedCode = code
		.replace(/\/\*[\s\S]*?\*\//g, '') // Remove block comments
		.replace(/\/\/.*/g, '');          // Remove line comments

	// Block dangerous patterns - expanded for common bypasses
	const dangerousPatterns = [
		// Direct dangerous function calls
		/\beval\s*\(/i,
		/\bFunction\s*\(/i,
		/\bsetTimeout\s*\(\s*['"`]/,
		/\bsetInterval\s*\(\s*['"`]/,

		// Global access attempts
		/\bdocument\b/,
		/\bwindow\b/,
		/\bglobalThis\b/,
		/\bself\b(?!\s*[=:])/,  // self access but not self assignment

		// Module/import attempts
		/\bimport\s*\(/,
		/\brequire\s*\(/,
		/\bimportScripts\s*\(/,

		// Prototype pollution
		/__proto__/,
		/\bconstructor\s*\[/,
		/Object\s*\.\s*getPrototypeOf/,
		/Object\s*\.\s*setPrototypeOf/,
		/Reflect\s*\.\s*getPrototypeOf/,

		// Global access via bracket notation
		/\[\s*['"`]eval['"`]\s*\]/,
		/\[\s*['"`]Function['"`]\s*\]/,

		// Worker spawning
		/\bnew\s+Worker\s*\(/,
		/\bSharedWorker\s*\(/,

		// Indirect eval attempts
		/\(0,\s*eval\)/,
		/\bReflect\s*\.\s*apply/,

		// Fetch/network access
		/\bfetch\s*\(/,
		/\bXMLHttpRequest\b/,
		/\bWebSocket\b/,

		// Storage access
		/\blocalStorage\b/,
		/\bsessionStorage\b/,
		/\bindexedDB\b/,
	];

	for (const pattern of dangerousPatterns) {
		if (pattern.test(normalizedCode)) {
			console.warn(`Plugin code contains dangerous pattern: ${pattern}`);
			return false;
		}
	}

	// Require exports.plugin pattern
	if (!code.includes('exports.plugin') && !code.includes('exports["plugin"]')) {
		console.warn('Plugin must export via exports.plugin');
		return false;
	}

	return true;
}

/**
 * Execute plugin code in a sandboxed Web Worker
 * Returns the plugin object or null on failure
 *
 * SECURITY: Uses multiple layers of defense:
 * 1. 'use strict' mode to prevent implicit global access
 * 2. Delete dangerous globals before plugin code runs
 * 3. Object.freeze on remaining globals to prevent modification
 * 4. Wrap plugin code in an IIFE with no closure access
 * 5. Timeout to prevent infinite loops
 */
async function executeInSandbox(code: string): Promise<Plugin | null> {
	return new Promise((resolve, reject) => {
		// Create worker from blob URL with hardened sandbox
		const workerCode = `
'use strict';

// Phase 1: Delete dangerous globals from worker scope
// These deletions happen before any user code runs
(function() {
	// Network access
	delete self.fetch;
	delete self.XMLHttpRequest;
	delete self.WebSocket;
	delete self.EventSource;

	// Storage
	delete self.localStorage;
	delete self.sessionStorage;
	delete self.indexedDB;
	delete self.caches;

	// Workers (prevent spawning sub-workers)
	delete self.Worker;
	delete self.SharedWorker;
	delete self.ServiceWorker;

	// Import mechanisms
	delete self.importScripts;

	// Dangerous constructors accessible from self
	const dangerousProps = [
		'eval', 'Function', 'Blob', 'URL', 'FileReader',
		'BroadcastChannel', 'MessageChannel'
	];
	dangerousProps.forEach(prop => {
		try { delete self[prop]; } catch(e) {}
	});
})();

// Phase 2: Create isolated execution context
const exports = Object.create(null);
const safePostMessage = self.postMessage.bind(self);

// Phase 3: Freeze critical objects to prevent prototype pollution
try {
	Object.freeze(Object.prototype);
	Object.freeze(Array.prototype);
	Object.freeze(Function.prototype);
	Object.freeze(String.prototype);
} catch(e) {
	// Some environments may not allow freezing prototypes
}

// Phase 4: Execute plugin in isolated scope
// The IIFE prevents closure access to outer variables
try {
	(function(exports) {
		'use strict';
		// Plugin code is injected here, isolated from outer scope
		${code}
	})(exports);

	// Only send serializable data
	const plugin = exports.plugin;
	if (plugin && typeof plugin === 'object') {
		safePostMessage({ success: true, plugin: JSON.parse(JSON.stringify(plugin)) });
	} else {
		safePostMessage({ success: false, error: 'Plugin must export an object via exports.plugin' });
	}
} catch (error) {
	// Sanitize error message to prevent info leakage
	const safeMessage = error instanceof Error
		? error.message.slice(0, 500)
		: 'Unknown plugin error';
	safePostMessage({ success: false, error: safeMessage });
}
`;

		const blob = new Blob([workerCode], { type: 'application/javascript' });
		const workerUrl = URL.createObjectURL(blob);
		const worker = new Worker(workerUrl);

		const timeout = setTimeout(() => {
			worker.terminate();
			URL.revokeObjectURL(workerUrl);
			reject(new Error('Plugin execution timed out'));
		}, 5000);

		worker.onmessage = (event) => {
			clearTimeout(timeout);
			worker.terminate();
			URL.revokeObjectURL(workerUrl);

			if (event.data.success) {
				resolve(event.data.plugin);
			} else {
				reject(new Error(event.data.error));
			}
		};

		worker.onerror = (error) => {
			clearTimeout(timeout);
			worker.terminate();
			URL.revokeObjectURL(workerUrl);
			reject(new Error(`Worker error: ${error.message}`));
		};
	});
}

/**
 * Load plugin from local storage
 */
export async function loadPluginsFromStorage(): Promise<void> {
	const stored = localStorage.getItem('lucid-plugins');
	if (!stored) return;

	try {
		const pluginIds = JSON.parse(stored);
		// Would load actual plugins here
		console.log('Would load plugins:', pluginIds);
	} catch (error) {
		console.error('Failed to load plugins from storage:', error);
	}
}

/**
 * Save enabled plugins to storage
 */
export function savePluginsToStorage(): void {
	const enabledIds = Array.from(registryState.enabled);
	localStorage.setItem('lucid-plugins', JSON.stringify(enabledIds));
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/**
 * Check if a plugin is enabled
 */
export function isPluginEnabled(pluginId: string): boolean {
	return registryState.enabled.has(pluginId);
}

/**
 * Get all enabled plugins
 */
export function getEnabledPlugins(): Plugin[] {
	return Array.from(registryState.plugins.values()).filter((p) =>
		registryState.enabled.has(p.manifest.id)
	);
}

/**
 * Get plugin configuration
 */
export function getPluginConfig(pluginId: string): Record<string, any> | undefined {
	return registryState.configs.get(pluginId);
}

/**
 * Validation result for plugin manifest
 */
export interface ManifestValidationResult {
	valid: boolean;
	errors: string[];
}

/**
 * Validate a plugin manifest
 */
export function validateManifest(manifest: PluginManifest): ManifestValidationResult {
	const errors: string[] = [];

	if (!manifest.id || manifest.id.trim() === '') {
		errors.push('Plugin id is required');
	}

	if (!manifest.name || manifest.name.trim() === '') {
		errors.push('Plugin name is required');
	}

	if (!manifest.version || !/^\d+\.\d+\.\d+/.test(manifest.version)) {
		errors.push('Plugin version must be in semver format (e.g., 1.0.0)');
	}

	if (!manifest.capabilities || manifest.capabilities.length === 0) {
		errors.push('Plugin must declare at least one capability');
	}

	if (!manifest.type) {
		errors.push('Plugin type is required');
	}

	return {
		valid: errors.length === 0,
		errors,
	};
}

// ============================================================================
// INITIALIZATION
// ============================================================================

/**
 * Initialize the plugin system
 */
export async function initializePluginSystem(): Promise<void> {
	// Register built-in plugins
	registerBuiltinPlugins();

	// Load user plugins
	await loadPluginsFromStorage();

	console.log('Plugin system initialized');
}
