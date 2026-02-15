/**
 * Tests for cross-backend migration tools
 */
import { describe, it, expect, beforeEach, vi } from 'vitest';
import {
  createMemoryBackend,
  createLocalBackend,
  migrate,
  exportBackend,
  importBundle,
  sync,
  type ExportBundle,
} from '../../src/storage/index.js';

describe('Migration Tools', () => {
  let source: ReturnType<typeof createMemoryBackend>;
  let target: ReturnType<typeof createMemoryBackend>;

  const testMetadata = {
    createdAt: Date.now(),
    updatedAt: Date.now(),
    createdBy: 'test-agent',
    classification: {
      empirical: 0,
      normative: 0,
      materiality: 0,
    },
    schemaId: { id: 'test', version: '1.0.0' },
    tags: [],
  };

  beforeEach(async () => {
    source = createMemoryBackend();
    target = createMemoryBackend();

    // Seed source with test data
    await source.set('profile:alice', { name: 'Alice' }, testMetadata);
    await source.set('profile:bob', { name: 'Bob' }, testMetadata);
    await source.set('session:123', { token: 'abc' }, testMetadata);
  });

  describe('migrate()', () => {
    it('should migrate all keys from source to target', async () => {
      const result = await migrate(source, target);

      expect(result.total).toBe(3);
      expect(result.migrated).toBe(3);
      expect(result.skipped).toBe(0);
      expect(result.failed).toBe(0);

      // Verify target has all data
      const alice = await target.get('profile:alice');
      expect(alice?.data).toEqual({ name: 'Alice' });
    });

    it('should filter by key pattern', async () => {
      const result = await migrate(source, target, {
        keyPattern: 'profile:*',
      });

      expect(result.total).toBe(2);
      expect(result.migrated).toBe(2);

      expect(await target.has('profile:alice')).toBe(true);
      expect(await target.has('session:123')).toBe(false);
    });

    it('should skip existing keys when skipExisting is true', async () => {
      await target.set('profile:alice', { name: 'Old Alice' }, testMetadata);

      const result = await migrate(source, target, { skipExisting: true });

      expect(result.skipped).toBe(1);
      expect(result.migrated).toBe(2);

      // Should NOT have overwritten
      const alice = await target.get('profile:alice');
      expect(alice?.data).toEqual({ name: 'Old Alice' });
    });

    it('should overwrite existing keys when skipExisting is false', async () => {
      await target.set('profile:alice', { name: 'Old Alice' }, testMetadata);

      const result = await migrate(source, target, { skipExisting: false });

      expect(result.migrated).toBe(3);

      const alice = await target.get('profile:alice');
      expect(alice?.data).toEqual({ name: 'Alice' });
    });

    it('should delete from source when deleteFromSource is true', async () => {
      await migrate(source, target, { deleteFromSource: true });

      expect(await source.has('profile:alice')).toBe(false);
      expect(await target.has('profile:alice')).toBe(true);
    });

    it('should report progress', async () => {
      const progress: Array<{ completed: number; total: number }> = [];

      await migrate(source, target, {
        concurrency: 1,
        onProgress: (p) => progress.push({ completed: p.completed, total: p.total }),
      });

      expect(progress.length).toBeGreaterThan(0);
      const last = progress[progress.length - 1];
      expect(last.completed).toBe(3);
      expect(last.total).toBe(3);
    });

    it('should respect concurrency limit', async () => {
      // Add more keys
      for (let i = 0; i < 10; i++) {
        await source.set(`bulk:${i}`, { i }, testMetadata);
      }

      const result = await migrate(source, target, { concurrency: 2 });

      expect(result.migrated).toBe(13); // 3 original + 10 bulk
    });

    it('should track elapsed time', async () => {
      const result = await migrate(source, target);
      expect(result.elapsedMs).toBeGreaterThanOrEqual(0);
    });
  });

  describe('exportBackend()', () => {
    it('should export all data to a bundle', async () => {
      const bundle = await exportBackend(source);

      expect(bundle.version).toBe('1.0.0');
      expect(bundle.itemCount).toBe(3);
      expect(bundle.items).toHaveLength(3);
      expect(bundle.exportedAt).toBeGreaterThan(0);
    });

    it('should filter by key pattern', async () => {
      const bundle = await exportBackend(source, { keyPattern: 'profile:*' });

      expect(bundle.itemCount).toBe(2);
      expect(bundle.items.every((i) => i.key.startsWith('profile:'))).toBe(true);
    });

    it('should include agent ID if provided', async () => {
      const bundle = await exportBackend(source, { agentId: 'agent:test' });
      expect(bundle.agentId).toBe('agent:test');
    });

    it('should produce JSON-serializable output', async () => {
      const bundle = await exportBackend(source);
      const json = JSON.stringify(bundle);
      const parsed = JSON.parse(json) as ExportBundle;

      expect(parsed.itemCount).toBe(3);
      expect(parsed.items[0].data).toBeDefined();
    });
  });

  describe('importBundle()', () => {
    it('should import all items from a bundle', async () => {
      const bundle = await exportBackend(source);

      const result = await importBundle(target, bundle);

      expect(result.migrated).toBe(3);
      expect(await target.has('profile:alice')).toBe(true);
    });

    it('should skip existing keys', async () => {
      await target.set('profile:alice', { name: 'Existing' }, testMetadata);
      const bundle = await exportBackend(source);

      const result = await importBundle(target, bundle, { skipExisting: true });

      expect(result.skipped).toBe(1);
      expect(result.migrated).toBe(2);
    });

    it('should roundtrip export → import correctly', async () => {
      const bundle = await exportBackend(source);
      await importBundle(target, bundle);

      const sourceKeys = await source.keys();
      const targetKeys = await target.keys();
      expect(targetKeys.sort()).toEqual(sourceKeys.sort());

      for (const key of sourceKeys) {
        const s = await source.get(key);
        const t = await target.get(key);
        expect(t?.data).toEqual(s?.data);
      }
    });
  });

  describe('sync()', () => {
    it('should detect differences in dry run', async () => {
      await target.set('target-only:1', { x: 1 }, testMetadata);

      const result = await sync(source, target, { dryRun: true });

      expect(result.sourceOnly).toHaveLength(3);
      expect(result.targetOnly).toHaveLength(1);
      expect(result.shared).toHaveLength(0);
      expect(result.migration).toBeUndefined();
    });

    it('should push source-only keys to target', async () => {
      const result = await sync(source, target, { direction: 'push' });

      expect(result.sourceOnly).toHaveLength(3);
      expect(result.migration?.migrated).toBe(3);
      expect(await target.has('profile:alice')).toBe(true);
    });

    it('should pull target-only keys to source', async () => {
      await target.set('target-only:1', { x: 1 }, testMetadata);

      const result = await sync(source, target, { direction: 'pull' });

      expect(result.targetOnly).toHaveLength(1);
      expect(result.migration?.migrated).toBe(1);
      expect(await source.has('target-only:1')).toBe(true);
    });

    it('should handle bidirectional sync', async () => {
      await target.set('target-only:1', { x: 1 }, testMetadata);

      const result = await sync(source, target, { direction: 'bidirectional' });

      // Source keys pushed to target
      expect(await target.has('profile:alice')).toBe(true);
      // Target keys pulled to source
      expect(await source.has('target-only:1')).toBe(true);
    });

    it('should detect shared keys', async () => {
      await target.set('profile:alice', { name: 'Target Alice' }, testMetadata);

      const result = await sync(source, target, { dryRun: true });

      expect(result.shared).toContain('profile:alice');
      expect(result.sourceOnly).not.toContain('profile:alice');
    });

    it('should filter by key pattern', async () => {
      const result = await sync(source, target, {
        direction: 'push',
        keyPattern: 'session:*',
      });

      expect(result.sourceOnly).toHaveLength(1);
      expect(result.migration?.migrated).toBe(1);
      expect(await target.has('session:123')).toBe(true);
      expect(await target.has('profile:alice')).toBe(false);
    });
  });

  describe('Cross-backend migration (memory → memory as proxy)', () => {
    it('should migrate between different backend instances', async () => {
      const otherTarget = createMemoryBackend();

      const result = await migrate(source, otherTarget, {
        keyPattern: 'profile:*',
      });

      expect(result.migrated).toBe(2);
      const alice = await otherTarget.get('profile:alice');
      expect(alice?.data).toEqual({ name: 'Alice' });
    });
  });
});
