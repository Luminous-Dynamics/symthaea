// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * UESS Persistence Utilities
 *
 * Environment-aware persistence for storage backends.
 * Automatically detects browser vs Node.js and uses appropriate storage.
 *
 * - Browser: IndexedDB (with localStorage fallback)
 * - Node.js: File system
 */

// =============================================================================
// Environment Detection
// =============================================================================

/**
 * Detect if running in browser environment
 */
export function isBrowser(): boolean {
  return typeof window !== 'undefined' && typeof window.indexedDB !== 'undefined';
}

/**
 * Detect if running in Node.js environment
 */
export function isNode(): boolean {
  return typeof process !== 'undefined' && process.versions?.node !== undefined;
}

// =============================================================================
// Persistence Interface
// =============================================================================

/**
 * Generic persistence adapter interface
 */
export interface PersistenceAdapter<T> {
  /** Save data to persistent storage */
  save(key: string, data: T): Promise<void>;

  /** Load data from persistent storage */
  load(key: string): Promise<T | null>;

  /** Delete data from persistent storage */
  delete(key: string): Promise<void>;

  /** Check if data exists */
  exists(key: string): Promise<boolean>;
}

// =============================================================================
// IndexedDB Persistence (Browser)
// =============================================================================

const IDB_DB_NAME = 'uess-storage';
const IDB_DB_VERSION = 1;
const IDB_STORE_NAME = 'data';

/**
 * IndexedDB persistence adapter for browser environments
 */
export class IndexedDBPersistence<T> implements PersistenceAdapter<T> {
  private dbPromise: Promise<IDBDatabase> | null = null;

  constructor(private readonly namespace: string = 'uess') {}

  private async getDB(): Promise<IDBDatabase> {
    if (this.dbPromise) {
      return this.dbPromise;
    }

    this.dbPromise = new Promise((resolve, reject) => {
      const request = indexedDB.open(`${IDB_DB_NAME}-${this.namespace}`, IDB_DB_VERSION);

      request.onerror = () => {
        reject(new Error(`Failed to open IndexedDB: ${request.error?.message}`));
      };

      request.onsuccess = () => {
        resolve(request.result);
      };

      request.onupgradeneeded = (event) => {
        const db = (event.target as IDBOpenDBRequest).result;
        if (!db.objectStoreNames.contains(IDB_STORE_NAME)) {
          db.createObjectStore(IDB_STORE_NAME);
        }
      };
    });

    return this.dbPromise;
  }

  async save(key: string, data: T): Promise<void> {
    const db = await this.getDB();

    return new Promise((resolve, reject) => {
      const transaction = db.transaction(IDB_STORE_NAME, 'readwrite');
      const store = transaction.objectStore(IDB_STORE_NAME);

      // Serialize Map objects specially
      const serializable = data instanceof Map
        ? { __type: 'Map', entries: Array.from(data.entries()) }
        : data;

      const request = store.put(JSON.stringify(serializable), key);

      request.onerror = () => {
        reject(new Error(`Failed to save to IndexedDB: ${request.error?.message}`));
      };

      request.onsuccess = () => {
        resolve();
      };
    });
  }

  async load(key: string): Promise<T | null> {
    const db = await this.getDB();

    return new Promise((resolve, reject) => {
      const transaction = db.transaction(IDB_STORE_NAME, 'readonly');
      const store = transaction.objectStore(IDB_STORE_NAME);
      const request = store.get(key);

      request.onerror = () => {
        reject(new Error(`Failed to load from IndexedDB: ${request.error?.message}`));
      };

      request.onsuccess = () => {
        if (request.result === undefined) {
          resolve(null);
          return;
        }

        try {
          const parsed = JSON.parse(request.result);

          // Deserialize Map objects
          if (parsed && typeof parsed === 'object' && parsed.__type === 'Map') {
            resolve(new Map(parsed.entries) as T);
          } else {
            resolve(parsed as T);
          }
        } catch {
          resolve(null);
        }
      };
    });
  }

  async delete(key: string): Promise<void> {
    const db = await this.getDB();

    return new Promise((resolve, reject) => {
      const transaction = db.transaction(IDB_STORE_NAME, 'readwrite');
      const store = transaction.objectStore(IDB_STORE_NAME);
      const request = store.delete(key);

      request.onerror = () => {
        reject(new Error(`Failed to delete from IndexedDB: ${request.error?.message}`));
      };

      request.onsuccess = () => {
        resolve();
      };
    });
  }

  async exists(key: string): Promise<boolean> {
    const db = await this.getDB();

    return new Promise((resolve, reject) => {
      const transaction = db.transaction(IDB_STORE_NAME, 'readonly');
      const store = transaction.objectStore(IDB_STORE_NAME);
      const request = store.getKey(key);

      request.onerror = () => {
        reject(new Error(`Failed to check existence in IndexedDB: ${request.error?.message}`));
      };

      request.onsuccess = () => {
        resolve(request.result !== undefined);
      };
    });
  }

  /**
   * Close the database connection
   */
  async close(): Promise<void> {
    if (this.dbPromise) {
      const db = await this.dbPromise;
      db.close();
      this.dbPromise = null;
    }
  }
}

// =============================================================================
// File System Persistence (Node.js)
// =============================================================================

/**
 * File system persistence adapter for Node.js environments
 */
export class FileSystemPersistence<T> implements PersistenceAdapter<T> {
  private readonly dataDir: string;
  private fsPromises: typeof import('fs/promises') | null = null;
  private path: typeof import('path') | null = null;

  constructor(
    namespace: string = 'uess',
    baseDir?: string
  ) {
    // Default to .uess-data in current directory
    this.dataDir = baseDir ?? `.uess-data/${namespace}`;
  }

  private async getFS(): Promise<typeof import('fs/promises')> {
    if (this.fsPromises) {
      return this.fsPromises;
    }

    // Dynamic import for Node.js modules
    this.fsPromises = await import('fs/promises');
    this.path = await import('path');

    // Ensure data directory exists
    await this.fsPromises.mkdir(this.dataDir, { recursive: true });

    return this.fsPromises;
  }

  private getFilePath(key: string): string {
    // Sanitize key for file system use
    const safeKey = key.replace(/[^a-zA-Z0-9_-]/g, '_');
    return this.path?.join(this.dataDir, `${safeKey}.json`) ?? `${this.dataDir}/${safeKey}.json`;
  }

  async save(key: string, data: T): Promise<void> {
    const fs = await this.getFS();
    const filePath = this.getFilePath(key);

    // Serialize Map objects specially
    const serializable = data instanceof Map
      ? { __type: 'Map', entries: Array.from(data.entries()) }
      : data;

    await fs.writeFile(filePath, JSON.stringify(serializable, null, 2), 'utf-8');
  }

  async load(key: string): Promise<T | null> {
    const fs = await this.getFS();
    const filePath = this.getFilePath(key);

    try {
      const content = await fs.readFile(filePath, 'utf-8');
      const parsed = JSON.parse(content);

      // Deserialize Map objects
      if (parsed && typeof parsed === 'object' && parsed.__type === 'Map') {
        return new Map(parsed.entries) as T;
      }

      return parsed as T;
    } catch (error) {
      // File doesn't exist or is invalid
      if ((error as NodeJS.ErrnoException).code === 'ENOENT') {
        return null;
      }
      throw error;
    }
  }

  async delete(key: string): Promise<void> {
    const fs = await this.getFS();
    const filePath = this.getFilePath(key);

    try {
      await fs.unlink(filePath);
    } catch (error) {
      // Ignore if file doesn't exist
      if ((error as NodeJS.ErrnoException).code !== 'ENOENT') {
        throw error;
      }
    }
  }

  async exists(key: string): Promise<boolean> {
    const fs = await this.getFS();
    const filePath = this.getFilePath(key);

    try {
      await fs.access(filePath);
      return true;
    } catch {
      return false;
    }
  }
}

// =============================================================================
// LocalStorage Fallback (Browser without IndexedDB)
// =============================================================================

/**
 * localStorage persistence adapter (fallback for older browsers)
 */
export class LocalStoragePersistence<T> implements PersistenceAdapter<T> {
  constructor(private readonly namespace: string = 'uess') {}

  private getKey(key: string): string {
    return `${this.namespace}:${key}`;
  }

  async save(key: string, data: T): Promise<void> {
    const serializable = data instanceof Map
      ? { __type: 'Map', entries: Array.from(data.entries()) }
      : data;

    localStorage.setItem(this.getKey(key), JSON.stringify(serializable));
  }

  async load(key: string): Promise<T | null> {
    const item = localStorage.getItem(this.getKey(key));
    if (item === null) {
      return null;
    }

    try {
      const parsed = JSON.parse(item);

      if (parsed && typeof parsed === 'object' && parsed.__type === 'Map') {
        return new Map(parsed.entries) as T;
      }

      return parsed as T;
    } catch {
      return null;
    }
  }

  async delete(key: string): Promise<void> {
    localStorage.removeItem(this.getKey(key));
  }

  async exists(key: string): Promise<boolean> {
    return localStorage.getItem(this.getKey(key)) !== null;
  }
}

// =============================================================================
// In-Memory Persistence (Fallback / Testing)
// =============================================================================

/**
 * In-memory persistence adapter (for testing or when no persistence is available)
 */
export class InMemoryPersistence<T> implements PersistenceAdapter<T> {
  private readonly store = new Map<string, string>();

  async save(key: string, data: T): Promise<void> {
    const serializable = data instanceof Map
      ? { __type: 'Map', entries: Array.from(data.entries()) }
      : data;

    this.store.set(key, JSON.stringify(serializable));
  }

  async load(key: string): Promise<T | null> {
    const item = this.store.get(key);
    if (item === undefined) {
      return null;
    }

    const parsed = JSON.parse(item);

    if (parsed && typeof parsed === 'object' && parsed.__type === 'Map') {
      return new Map(parsed.entries) as T;
    }

    return parsed as T;
  }

  async delete(key: string): Promise<void> {
    this.store.delete(key);
  }

  async exists(key: string): Promise<boolean> {
    return this.store.has(key);
  }

  clear(): void {
    this.store.clear();
  }
}

// =============================================================================
// Factory Function
// =============================================================================

/**
 * Create the appropriate persistence adapter for the current environment
 */
export function createPersistenceAdapter<T>(
  namespace: string = 'uess',
  options?: {
    /** Force a specific adapter type */
    forceAdapter?: 'indexeddb' | 'filesystem' | 'localstorage' | 'memory';
    /** Base directory for file system persistence (Node.js only) */
    baseDir?: string;
  }
): PersistenceAdapter<T> {
  // Handle forced adapter
  if (options?.forceAdapter) {
    switch (options.forceAdapter) {
      case 'indexeddb':
        return new IndexedDBPersistence<T>(namespace);
      case 'filesystem':
        return new FileSystemPersistence<T>(namespace, options.baseDir);
      case 'localstorage':
        return new LocalStoragePersistence<T>(namespace);
      case 'memory':
        return new InMemoryPersistence<T>();
    }
  }

  // Auto-detect environment
  if (isBrowser()) {
    // Prefer IndexedDB, fall back to localStorage
    if (typeof indexedDB !== 'undefined') {
      return new IndexedDBPersistence<T>(namespace);
    }
    if (typeof localStorage !== 'undefined') {
      return new LocalStoragePersistence<T>(namespace);
    }
  }

  if (isNode()) {
    return new FileSystemPersistence<T>(namespace, options?.baseDir);
  }

  // Fallback to in-memory
  console.warn('UESS: No persistent storage available, using in-memory storage');
  return new InMemoryPersistence<T>();
}
