/**
 * Plugin System Tests
 *
 * Tests for the LUCID plugin architecture including:
 * - Plugin registration and lifecycle
 * - Manifest validation
 * - Plugin enabling/disabling
 * - Configuration management
 * - Plugin type-specific functionality
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import {
  registerPlugin,
  unregisterPlugin,
  enablePlugin,
  disablePlugin,
  configurePlugin,
  getPlugin,
  getPluginsByType,
  getEnabledPlugins,
  isPluginEnabled,
  getPluginConfig,
  validateManifest,
  type PluginManifest,
  type ImporterPlugin,
  type ExporterPlugin,
  type AIModelPlugin,
  pluginRegistry,
} from '../services/plugin-system';
import { get } from 'svelte/store';

// ============================================================================
// TEST FIXTURES
// ============================================================================

function createMockManifest(overrides: Partial<PluginManifest> = {}): PluginManifest {
  return {
    id: `test-plugin-${Math.random().toString(36).slice(2)}`,
    name: 'Test Plugin',
    version: '1.0.0',
    description: 'A test plugin',
    author: 'Test Author',
    type: 'importer',
    capabilities: ['import'],
    ...overrides,
  };
}

function createMockImporterPlugin(manifestOverrides: Partial<PluginManifest> = {}): ImporterPlugin {
  return {
    manifest: createMockManifest({ type: 'importer', capabilities: ['import'], ...manifestOverrides }),
    supportedFormats: ['json', 'csv'],
    import: vi.fn().mockResolvedValue([]),
    validate: vi.fn().mockReturnValue(true),
  };
}

function createMockExporterPlugin(manifestOverrides: Partial<PluginManifest> = {}): ExporterPlugin {
  return {
    manifest: createMockManifest({ type: 'exporter', capabilities: ['export'], ...manifestOverrides }),
    supportedFormats: ['json', 'markdown'],
    export: vi.fn().mockResolvedValue('exported content'),
    getFilename: vi.fn().mockReturnValue('export.json'),
  };
}

function createMockAIPlugin(manifestOverrides: Partial<PluginManifest> = {}): AIModelPlugin {
  return {
    manifest: createMockManifest({ type: 'ai-model', capabilities: ['classify'], ...manifestOverrides }),
    modelType: 'classifier',
    initialize: vi.fn().mockResolvedValue(undefined),
    process: vi.fn().mockResolvedValue({ classification: 'Claim' }),
    destroy: vi.fn().mockResolvedValue(undefined),
  };
}

// ============================================================================
// TESTS
// ============================================================================

describe('Plugin System', () => {
  beforeEach(() => {
    // Clear registry before each test
    const state = get(pluginRegistry);
    state.plugins.clear();
    state.enabled.clear();
    state.configs.clear();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  describe('registerPlugin', () => {
    it('should register a new plugin', () => {
      const plugin = createMockImporterPlugin({ id: 'test-importer' });

      registerPlugin(plugin);

      const registered = getPlugin('test-importer');
      expect(registered).toBeDefined();
      expect(registered?.manifest.id).toBe('test-importer');
    });

    it('should not register duplicate plugins', () => {
      const plugin = createMockImporterPlugin({ id: 'duplicate-test' });

      registerPlugin(plugin);
      registerPlugin(plugin); // Try to register again

      const state = get(pluginRegistry);
      const count = Array.from(state.plugins.values()).filter(
        (p) => p.manifest.id === 'duplicate-test'
      ).length;

      expect(count).toBe(1);
    });

    it('should register plugins of different types', () => {
      const importer = createMockImporterPlugin({ id: 'importer-1' });
      const exporter = createMockExporterPlugin({ id: 'exporter-1' });
      const aiModel = createMockAIPlugin({ id: 'ai-1' });

      registerPlugin(importer);
      registerPlugin(exporter);
      registerPlugin(aiModel);

      expect(getPlugin('importer-1')).toBeDefined();
      expect(getPlugin('exporter-1')).toBeDefined();
      expect(getPlugin('ai-1')).toBeDefined();
    });
  });

  describe('unregisterPlugin', () => {
    it('should unregister an existing plugin', () => {
      const plugin = createMockImporterPlugin({ id: 'to-unregister' });

      registerPlugin(plugin);
      expect(getPlugin('to-unregister')).toBeDefined();

      unregisterPlugin('to-unregister');
      expect(getPlugin('to-unregister')).toBeUndefined();
    });

    it('should disable plugin before unregistering', () => {
      const plugin = createMockImporterPlugin({ id: 'enabled-plugin' });

      registerPlugin(plugin);
      enablePlugin('enabled-plugin');
      expect(isPluginEnabled('enabled-plugin')).toBe(true);

      unregisterPlugin('enabled-plugin');
      expect(isPluginEnabled('enabled-plugin')).toBe(false);
    });

    it('should handle unregistering non-existent plugin gracefully', () => {
      expect(() => unregisterPlugin('non-existent')).not.toThrow();
    });
  });

  describe('enablePlugin / disablePlugin', () => {
    it('should enable a registered plugin', () => {
      const plugin = createMockImporterPlugin({ id: 'enable-test' });

      registerPlugin(plugin);
      const result = enablePlugin('enable-test');

      expect(result).toBe(true);
      expect(isPluginEnabled('enable-test')).toBe(true);
    });

    it('should return false when enabling non-existent plugin', () => {
      const result = enablePlugin('non-existent');

      expect(result).toBe(false);
    });

    it('should disable an enabled plugin', () => {
      const plugin = createMockImporterPlugin({ id: 'disable-test' });

      registerPlugin(plugin);
      enablePlugin('disable-test');
      disablePlugin('disable-test');

      expect(isPluginEnabled('disable-test')).toBe(false);
    });
  });

  describe('configurePlugin', () => {
    it('should store plugin configuration', () => {
      const plugin = createMockImporterPlugin({ id: 'config-test' });
      const config = { apiKey: 'test-key', endpoint: 'https://api.test.com' };

      registerPlugin(plugin);
      configurePlugin('config-test', config);

      const stored = getPluginConfig('config-test');
      expect(stored).toEqual(config);
    });

    it('should update existing configuration', () => {
      const plugin = createMockImporterPlugin({ id: 'config-update' });

      registerPlugin(plugin);
      configurePlugin('config-update', { key1: 'value1' });
      configurePlugin('config-update', { key1: 'updated', key2: 'new' });

      const stored = getPluginConfig('config-update');
      expect(stored).toEqual({ key1: 'updated', key2: 'new' });
    });
  });

  describe('getPluginsByType', () => {
    it('should return only enabled plugins of specified type', () => {
      registerPlugin(createMockImporterPlugin({ id: 'importer-a' }));
      registerPlugin(createMockImporterPlugin({ id: 'importer-b' }));
      registerPlugin(createMockExporterPlugin({ id: 'exporter-a' }));

      // Enable the importers
      enablePlugin('importer-a');
      enablePlugin('importer-b');

      const importers = getPluginsByType('importer');

      expect(importers.length).toBe(2);
      expect(importers.every((p) => p.manifest.type === 'importer')).toBe(true);
    });

    it('should not return disabled plugins', () => {
      registerPlugin(createMockImporterPlugin({ id: 'enabled-importer' }));
      registerPlugin(createMockImporterPlugin({ id: 'disabled-importer' }));

      enablePlugin('enabled-importer');
      // disabled-importer is NOT enabled

      const importers = getPluginsByType('importer');

      expect(importers.length).toBe(1);
      expect(importers[0].manifest.id).toBe('enabled-importer');
    });

    it('should return empty array for type with no enabled plugins', () => {
      registerPlugin(createMockImporterPlugin({ id: 'only-importer' }));
      // Not enabled

      const importers = getPluginsByType('importer');

      expect(importers).toEqual([]);
    });

    it('should return empty array for type with no plugins', () => {
      registerPlugin(createMockImporterPlugin({ id: 'only-importer' }));
      enablePlugin('only-importer');

      const themes = getPluginsByType('theme');

      expect(themes).toEqual([]);
    });
  });

  describe('getEnabledPlugins', () => {
    it('should return only enabled plugins', () => {
      registerPlugin(createMockImporterPlugin({ id: 'enabled-1' }));
      registerPlugin(createMockImporterPlugin({ id: 'disabled-1' }));
      registerPlugin(createMockExporterPlugin({ id: 'enabled-2' }));

      enablePlugin('enabled-1');
      enablePlugin('enabled-2');

      const enabled = getEnabledPlugins();

      expect(enabled.length).toBe(2);
      expect(enabled.map((p) => p.manifest.id)).toContain('enabled-1');
      expect(enabled.map((p) => p.manifest.id)).toContain('enabled-2');
      expect(enabled.map((p) => p.manifest.id)).not.toContain('disabled-1');
    });
  });

  describe('validateManifest', () => {
    it('should validate a correct manifest', () => {
      const manifest = createMockManifest();

      const result = validateManifest(manifest);

      expect(result.valid).toBe(true);
      expect(result.errors).toHaveLength(0);
    });

    it('should reject manifest without id', () => {
      const manifest = createMockManifest({ id: '' });

      const result = validateManifest(manifest);

      expect(result.valid).toBe(false);
      expect(result.errors.some((e) => e.includes('id'))).toBe(true);
    });

    it('should reject manifest without name', () => {
      const manifest = createMockManifest({ name: '' });

      const result = validateManifest(manifest);

      expect(result.valid).toBe(false);
      expect(result.errors.some((e) => e.includes('name'))).toBe(true);
    });

    it('should reject manifest with invalid version format', () => {
      const manifest = createMockManifest({ version: 'invalid' });

      const result = validateManifest(manifest);

      expect(result.valid).toBe(false);
      expect(result.errors.some((e) => e.includes('version'))).toBe(true);
    });

    it('should reject manifest with empty capabilities', () => {
      const manifest = createMockManifest({ capabilities: [] });

      const result = validateManifest(manifest);

      expect(result.valid).toBe(false);
      expect(result.errors.some((e) => e.includes('capability'))).toBe(true);
    });
  });

  describe('Plugin Lifecycle', () => {
    it('should support full lifecycle: register -> enable -> configure -> disable -> unregister', async () => {
      const plugin = createMockAIPlugin({ id: 'lifecycle-test' });

      // Register
      registerPlugin(plugin);
      expect(getPlugin('lifecycle-test')).toBeDefined();

      // Enable
      enablePlugin('lifecycle-test');
      expect(isPluginEnabled('lifecycle-test')).toBe(true);

      // Configure
      configurePlugin('lifecycle-test', { model: 'test-model' });
      expect(getPluginConfig('lifecycle-test')).toEqual({ model: 'test-model' });

      // Disable
      disablePlugin('lifecycle-test');
      expect(isPluginEnabled('lifecycle-test')).toBe(false);

      // Unregister
      unregisterPlugin('lifecycle-test');
      expect(getPlugin('lifecycle-test')).toBeUndefined();
    });
  });
});

describe('Importer Plugin', () => {
  it('should call import function with correct parameters', async () => {
    const plugin = createMockImporterPlugin({ id: 'import-test' });
    registerPlugin(plugin);

    const data = '{"thoughts": []}';
    const context = {
      thoughts: [],
      config: {},
      api: {} as any,
    };

    await plugin.import(data, 'json', context);

    expect(plugin.import).toHaveBeenCalledWith(data, 'json', context);
  });

  it('should validate data before import', () => {
    const plugin = createMockImporterPlugin({ id: 'validate-test' });
    registerPlugin(plugin);

    const validData = '{"valid": true}';
    const result = plugin.validate?.(validData, 'json');

    expect(result).toBe(true);
  });
});

describe('Exporter Plugin', () => {
  it('should export thoughts to specified format', async () => {
    const plugin = createMockExporterPlugin({ id: 'export-test' });
    registerPlugin(plugin);

    const thoughts = [{ id: '1', content: 'Test thought' }] as any;
    const context = {
      thoughts,
      config: {},
      api: {} as any,
    };

    const result = await plugin.export(thoughts, 'json', context);

    expect(result).toBe('exported content');
  });

  it('should generate appropriate filename', () => {
    const plugin = createMockExporterPlugin({ id: 'filename-test' });
    registerPlugin(plugin);

    const filename = plugin.getFilename?.('json');

    expect(filename).toBe('export.json');
  });
});

describe('AI Model Plugin', () => {
  it('should initialize with configuration', async () => {
    const plugin = createMockAIPlugin({ id: 'ai-init-test' });
    registerPlugin(plugin);

    await plugin.initialize({ modelPath: '/path/to/model' });

    expect(plugin.initialize).toHaveBeenCalledWith({ modelPath: '/path/to/model' });
  });

  it('should process input and return result', async () => {
    const plugin = createMockAIPlugin({ id: 'ai-process-test' });
    registerPlugin(plugin);

    const context = {
      thoughts: [],
      config: {},
      api: {} as any,
    };

    const result = await plugin.process('Test input', context);

    expect(result).toEqual({ classification: 'Claim' });
  });

  it('should cleanup on destroy', async () => {
    const plugin = createMockAIPlugin({ id: 'ai-destroy-test' });
    registerPlugin(plugin);

    await plugin.destroy?.();

    expect(plugin.destroy).toHaveBeenCalled();
  });
});
