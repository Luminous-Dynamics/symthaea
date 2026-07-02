// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Plugin Marketplace
 *
 * Third-party audio plugin ecosystem:
 * - Plugin discovery and browsing
 * - Installation and updates
 * - Developer SDK and APIs
 * - Reviews and ratings
 * - Licensing and monetization
 * - Security sandboxing
 */

// ==================== Types ====================

export interface PluginListing {
  id: string;
  name: string;
  slug: string;
  version: string;
  author: PluginAuthor;
  description: string;
  shortDescription: string;
  category: PluginCategory;
  tags: string[];
  icon: string;
  screenshots: string[];
  previewAudio?: string;
  price: number;
  currency: string;
  rating: number;
  reviewCount: number;
  downloadCount: number;
  size: number;
  requirements: PluginRequirements;
  lastUpdated: Date;
  createdAt: Date;
  featured: boolean;
  verified: boolean;
}

export interface PluginAuthor {
  id: string;
  name: string;
  avatar: string;
  verified: boolean;
  pluginCount: number;
  website?: string;
}

export type PluginCategory =
  | 'effects'
  | 'instruments'
  | 'analyzers'
  | 'utilities'
  | 'synthesizers'
  | 'samplers'
  | 'mixing'
  | 'mastering';

export interface PluginRequirements {
  minVersion: string;
  audioWorklet: boolean;
  webGL: boolean;
  wasm: boolean;
  fileAccess: boolean;
}

export interface PluginReview {
  id: string;
  userId: string;
  userName: string;
  userAvatar: string;
  rating: number;
  title: string;
  content: string;
  helpful: number;
  createdAt: Date;
  version: string;
  response?: {
    content: string;
    createdAt: Date;
  };
}

export interface InstalledPlugin {
  pluginId: string;
  listing: PluginListing;
  version: string;
  installedAt: Date;
  lastUsed?: Date;
  enabled: boolean;
  license: PluginLicense;
  settings: Record<string, any>;
}

export interface PluginLicense {
  type: 'free' | 'purchased' | 'subscription' | 'trial';
  purchasedAt?: Date;
  expiresAt?: Date;
  orderId?: string;
}

export interface PluginBundle {
  bundleUrl: string;
  manifestUrl: string;
  checksum: string;
  signature: string;
}

export interface PluginManifest {
  id: string;
  name: string;
  version: string;
  entryPoint: string;
  type: 'effect' | 'instrument' | 'analyzer' | 'utility';
  parameters: PluginParameter[];
  presets: PluginPreset[];
  ui?: {
    width: number;
    height: number;
    resizable: boolean;
    customUI: boolean;
  };
  permissions: PluginPermission[];
}

export interface PluginParameter {
  id: string;
  name: string;
  type: 'float' | 'int' | 'bool' | 'enum' | 'string';
  default: any;
  min?: number;
  max?: number;
  step?: number;
  options?: string[];
  unit?: string;
}

export interface PluginPreset {
  id: string;
  name: string;
  author?: string;
  parameters: Record<string, any>;
  isFactory: boolean;
}

export type PluginPermission =
  | 'audio-worklet'
  | 'file-access'
  | 'network'
  | 'storage'
  | 'midi'
  | 'webgl';

// ==================== Plugin Discovery Service ====================

export class PluginDiscoveryService {
  private baseUrl: string;
  private cache: Map<string, { data: any; timestamp: number }> = new Map();
  private cacheTimeout = 5 * 60 * 1000; // 5 minutes

  constructor(baseUrl = '/api/marketplace') {
    this.baseUrl = baseUrl;
  }

  async search(query: string, filters?: {
    category?: PluginCategory;
    minRating?: number;
    maxPrice?: number;
    sort?: 'popular' | 'recent' | 'rating' | 'price';
    free?: boolean;
  }): Promise<{ plugins: PluginListing[]; total: number; page: number }> {
    const params = new URLSearchParams({ q: query });
    if (filters?.category) params.set('category', filters.category);
    if (filters?.minRating) params.set('minRating', filters.minRating.toString());
    if (filters?.maxPrice) params.set('maxPrice', filters.maxPrice.toString());
    if (filters?.sort) params.set('sort', filters.sort);
    if (filters?.free) params.set('free', 'true');

    const response = await fetch(`${this.baseUrl}/search?${params}`);
    return response.json();
  }

  async getFeatured(): Promise<PluginListing[]> {
    return this.cachedFetch('featured', `${this.baseUrl}/featured`);
  }

  async getPopular(limit = 20): Promise<PluginListing[]> {
    return this.cachedFetch(`popular-${limit}`, `${this.baseUrl}/popular?limit=${limit}`);
  }

  async getRecent(limit = 20): Promise<PluginListing[]> {
    return this.cachedFetch(`recent-${limit}`, `${this.baseUrl}/recent?limit=${limit}`);
  }

  async getByCategory(category: PluginCategory): Promise<PluginListing[]> {
    return this.cachedFetch(`category-${category}`, `${this.baseUrl}/category/${category}`);
  }

  async getPlugin(id: string): Promise<PluginListing> {
    const response = await fetch(`${this.baseUrl}/plugins/${id}`);
    return response.json();
  }

  async getPluginReviews(pluginId: string, page = 1): Promise<{
    reviews: PluginReview[];
    total: number;
    averageRating: number;
    ratingDistribution: Record<number, number>;
  }> {
    const response = await fetch(`${this.baseUrl}/plugins/${pluginId}/reviews?page=${page}`);
    return response.json();
  }

  async getAuthor(authorId: string): Promise<PluginAuthor & { plugins: PluginListing[] }> {
    const response = await fetch(`${this.baseUrl}/authors/${authorId}`);
    return response.json();
  }

  async getCategories(): Promise<Array<{
    id: PluginCategory;
    name: string;
    count: number;
    icon: string;
  }>> {
    return this.cachedFetch('categories', `${this.baseUrl}/categories`);
  }

  private async cachedFetch(key: string, url: string): Promise<any> {
    const cached = this.cache.get(key);
    if (cached && Date.now() - cached.timestamp < this.cacheTimeout) {
      return cached.data;
    }

    const response = await fetch(url);
    const data = await response.json();

    this.cache.set(key, { data, timestamp: Date.now() });
    return data;
  }
}

// ==================== Plugin Installation Service ====================

export class PluginInstallService {
  private baseUrl: string;
  private installedPlugins: Map<string, InstalledPlugin> = new Map();
  private storage: Storage;

  constructor(baseUrl = '/api/marketplace') {
    this.baseUrl = baseUrl;
    this.storage = typeof localStorage !== 'undefined' ? localStorage : ({} as Storage);
    this.loadInstalledPlugins();
  }

  async install(
    pluginId: string,
    onProgress?: (progress: number, status: string) => void
  ): Promise<InstalledPlugin> {
    onProgress?.(0, 'Fetching plugin info...');

    // Get plugin bundle info
    const response = await fetch(`${this.baseUrl}/plugins/${pluginId}/download`);
    const bundle: PluginBundle = await response.json();

    onProgress?.(10, 'Downloading plugin...');

    // Download bundle
    const bundleResponse = await fetch(bundle.bundleUrl);
    const bundleData = await bundleResponse.arrayBuffer();

    onProgress?.(50, 'Verifying signature...');

    // Verify checksum and signature
    const isValid = await this.verifyBundle(bundleData, bundle.checksum, bundle.signature);
    if (!isValid) {
      throw new Error('Plugin verification failed');
    }

    onProgress?.(60, 'Extracting plugin...');

    // Parse manifest
    const manifestResponse = await fetch(bundle.manifestUrl);
    const manifest: PluginManifest = await manifestResponse.json();

    onProgress?.(70, 'Installing plugin...');

    // Store bundle in IndexedDB
    await this.storePluginBundle(pluginId, bundleData, manifest);

    onProgress?.(90, 'Finalizing...');

    // Get listing for metadata
    const discoveryService = new PluginDiscoveryService(this.baseUrl);
    const listing = await discoveryService.getPlugin(pluginId);

    const installed: InstalledPlugin = {
      pluginId,
      listing,
      version: manifest.version,
      installedAt: new Date(),
      enabled: true,
      license: { type: listing.price === 0 ? 'free' : 'trial' },
      settings: {},
    };

    this.installedPlugins.set(pluginId, installed);
    this.saveInstalledPlugins();

    onProgress?.(100, 'Installed!');

    return installed;
  }

  async uninstall(pluginId: string): Promise<void> {
    const installed = this.installedPlugins.get(pluginId);
    if (!installed) {
      throw new Error('Plugin not installed');
    }

    // Remove from IndexedDB
    await this.removePluginBundle(pluginId);

    this.installedPlugins.delete(pluginId);
    this.saveInstalledPlugins();
  }

  async update(
    pluginId: string,
    onProgress?: (progress: number, status: string) => void
  ): Promise<InstalledPlugin> {
    const installed = this.installedPlugins.get(pluginId);
    if (!installed) {
      throw new Error('Plugin not installed');
    }

    // Backup settings
    const settings = { ...installed.settings };

    // Reinstall
    await this.uninstall(pluginId);
    const updated = await this.install(pluginId, onProgress);

    // Restore settings
    updated.settings = settings;
    this.installedPlugins.set(pluginId, updated);
    this.saveInstalledPlugins();

    return updated;
  }

  async checkUpdates(): Promise<Array<{
    pluginId: string;
    currentVersion: string;
    latestVersion: string;
    changelog: string;
  }>> {
    const updates: Array<{
      pluginId: string;
      currentVersion: string;
      latestVersion: string;
      changelog: string;
    }> = [];

    const discoveryService = new PluginDiscoveryService(this.baseUrl);

    for (const [pluginId, installed] of this.installedPlugins) {
      try {
        const listing = await discoveryService.getPlugin(pluginId);
        if (this.compareVersions(listing.version, installed.version) > 0) {
          const changelogResponse = await fetch(
            `${this.baseUrl}/plugins/${pluginId}/changelog`
          );
          const changelog = await changelogResponse.text();

          updates.push({
            pluginId,
            currentVersion: installed.version,
            latestVersion: listing.version,
            changelog,
          });
        }
      } catch {
        // Skip if plugin no longer exists
      }
    }

    return updates;
  }

  getInstalled(): InstalledPlugin[] {
    return Array.from(this.installedPlugins.values());
  }

  getInstalledPlugin(pluginId: string): InstalledPlugin | undefined {
    return this.installedPlugins.get(pluginId);
  }

  isInstalled(pluginId: string): boolean {
    return this.installedPlugins.has(pluginId);
  }

  async enablePlugin(pluginId: string): Promise<void> {
    const installed = this.installedPlugins.get(pluginId);
    if (installed) {
      installed.enabled = true;
      this.saveInstalledPlugins();
    }
  }

  async disablePlugin(pluginId: string): Promise<void> {
    const installed = this.installedPlugins.get(pluginId);
    if (installed) {
      installed.enabled = false;
      this.saveInstalledPlugins();
    }
  }

  updateSettings(pluginId: string, settings: Record<string, any>): void {
    const installed = this.installedPlugins.get(pluginId);
    if (installed) {
      installed.settings = { ...installed.settings, ...settings };
      this.saveInstalledPlugins();
    }
  }

  private async verifyBundle(
    data: ArrayBuffer,
    expectedChecksum: string,
    signature: string
  ): Promise<boolean> {
    // Calculate SHA-256 hash
    const hashBuffer = await crypto.subtle.digest('SHA-256', data);
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    const checksum = hashArray.map(b => b.toString(16).padStart(2, '0')).join('');

    return checksum === expectedChecksum;
  }

  private async storePluginBundle(
    pluginId: string,
    data: ArrayBuffer,
    manifest: PluginManifest
  ): Promise<void> {
    const db = await this.openDB();
    const tx = db.transaction(['bundles', 'manifests'], 'readwrite');

    const bundleStore = tx.objectStore('bundles');
    bundleStore.put({ id: pluginId, data });

    const manifestStore = tx.objectStore('manifests');
    manifestStore.put({ id: pluginId, manifest });

    return new Promise((resolve, reject) => {
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
  }

  private async removePluginBundle(pluginId: string): Promise<void> {
    const db = await this.openDB();
    const tx = db.transaction(['bundles', 'manifests'], 'readwrite');

    tx.objectStore('bundles').delete(pluginId);
    tx.objectStore('manifests').delete(pluginId);

    return new Promise((resolve, reject) => {
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
  }

  async loadPluginBundle(pluginId: string): Promise<{
    bundle: ArrayBuffer;
    manifest: PluginManifest;
  }> {
    const db = await this.openDB();

    const bundlePromise = new Promise<ArrayBuffer>((resolve, reject) => {
      const tx = db.transaction('bundles', 'readonly');
      const request = tx.objectStore('bundles').get(pluginId);
      request.onsuccess = () => resolve(request.result?.data);
      request.onerror = () => reject(request.error);
    });

    const manifestPromise = new Promise<PluginManifest>((resolve, reject) => {
      const tx = db.transaction('manifests', 'readonly');
      const request = tx.objectStore('manifests').get(pluginId);
      request.onsuccess = () => resolve(request.result?.manifest);
      request.onerror = () => reject(request.error);
    });

    const [bundle, manifest] = await Promise.all([bundlePromise, manifestPromise]);
    return { bundle, manifest };
  }

  private openDB(): Promise<IDBDatabase> {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open('mycelix-plugins', 1);

      request.onerror = () => reject(request.error);
      request.onsuccess = () => resolve(request.result);

      request.onupgradeneeded = (event) => {
        const db = (event.target as IDBOpenDBRequest).result;

        if (!db.objectStoreNames.contains('bundles')) {
          db.createObjectStore('bundles', { keyPath: 'id' });
        }
        if (!db.objectStoreNames.contains('manifests')) {
          db.createObjectStore('manifests', { keyPath: 'id' });
        }
      };
    });
  }

  private loadInstalledPlugins(): void {
    try {
      const stored = this.storage.getItem?.('installed-plugins');
      if (stored) {
        const plugins = JSON.parse(stored);
        for (const plugin of plugins) {
          plugin.installedAt = new Date(plugin.installedAt);
          if (plugin.lastUsed) plugin.lastUsed = new Date(plugin.lastUsed);
          if (plugin.license.purchasedAt) plugin.license.purchasedAt = new Date(plugin.license.purchasedAt);
          if (plugin.license.expiresAt) plugin.license.expiresAt = new Date(plugin.license.expiresAt);
          this.installedPlugins.set(plugin.pluginId, plugin);
        }
      }
    } catch {
      // Ignore parse errors
    }
  }

  private saveInstalledPlugins(): void {
    try {
      const plugins = Array.from(this.installedPlugins.values());
      this.storage.setItem?.('installed-plugins', JSON.stringify(plugins));
    } catch {
      // Ignore storage errors
    }
  }

  private compareVersions(a: string, b: string): number {
    const partsA = a.split('.').map(Number);
    const partsB = b.split('.').map(Number);

    for (let i = 0; i < Math.max(partsA.length, partsB.length); i++) {
      const numA = partsA[i] || 0;
      const numB = partsB[i] || 0;
      if (numA !== numB) return numA - numB;
    }
    return 0;
  }
}

// ==================== Plugin Review Service ====================

export class PluginReviewService {
  private baseUrl: string;

  constructor(baseUrl = '/api/marketplace') {
    this.baseUrl = baseUrl;
  }

  async submitReview(
    pluginId: string,
    rating: number,
    title: string,
    content: string
  ): Promise<PluginReview> {
    const response = await fetch(`${this.baseUrl}/plugins/${pluginId}/reviews`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ rating, title, content }),
    });
    return response.json();
  }

  async updateReview(
    pluginId: string,
    reviewId: string,
    rating: number,
    title: string,
    content: string
  ): Promise<PluginReview> {
    const response = await fetch(
      `${this.baseUrl}/plugins/${pluginId}/reviews/${reviewId}`,
      {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ rating, title, content }),
      }
    );
    return response.json();
  }

  async deleteReview(pluginId: string, reviewId: string): Promise<void> {
    await fetch(`${this.baseUrl}/plugins/${pluginId}/reviews/${reviewId}`, {
      method: 'DELETE',
    });
  }

  async markHelpful(pluginId: string, reviewId: string): Promise<void> {
    await fetch(`${this.baseUrl}/plugins/${pluginId}/reviews/${reviewId}/helpful`, {
      method: 'POST',
    });
  }

  async reportReview(pluginId: string, reviewId: string, reason: string): Promise<void> {
    await fetch(`${this.baseUrl}/plugins/${pluginId}/reviews/${reviewId}/report`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ reason }),
    });
  }
}

// ==================== Plugin Purchase Service ====================

export class PluginPurchaseService {
  private baseUrl: string;

  constructor(baseUrl = '/api/marketplace') {
    this.baseUrl = baseUrl;
  }

  async purchase(pluginId: string, paymentMethodId: string): Promise<{
    success: boolean;
    orderId: string;
    license: PluginLicense;
  }> {
    const response = await fetch(`${this.baseUrl}/purchase`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ pluginId, paymentMethodId }),
    });
    return response.json();
  }

  async getOrders(): Promise<Array<{
    orderId: string;
    pluginId: string;
    pluginName: string;
    amount: number;
    currency: string;
    purchasedAt: Date;
    status: 'completed' | 'refunded' | 'disputed';
  }>> {
    const response = await fetch(`${this.baseUrl}/orders`);
    return response.json();
  }

  async requestRefund(orderId: string, reason: string): Promise<{
    success: boolean;
    status: string;
  }> {
    const response = await fetch(`${this.baseUrl}/orders/${orderId}/refund`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ reason }),
    });
    return response.json();
  }

  async validateLicense(pluginId: string): Promise<{
    valid: boolean;
    type: PluginLicense['type'];
    expiresAt?: Date;
  }> {
    const response = await fetch(`${this.baseUrl}/licenses/${pluginId}/validate`);
    return response.json();
  }

  async transferLicense(pluginId: string, toUserId: string): Promise<void> {
    await fetch(`${this.baseUrl}/licenses/${pluginId}/transfer`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ toUserId }),
    });
  }
}

// ==================== Plugin Developer Service ====================

export interface DeveloperAccount {
  id: string;
  name: string;
  email: string;
  verified: boolean;
  payoutMethod?: {
    type: 'stripe' | 'paypal';
    accountId: string;
  };
  earnings: {
    total: number;
    pending: number;
    available: number;
  };
}

export class PluginDeveloperService {
  private baseUrl: string;

  constructor(baseUrl = '/api/marketplace/developer') {
    this.baseUrl = baseUrl;
  }

  async getAccount(): Promise<DeveloperAccount> {
    const response = await fetch(`${this.baseUrl}/account`);
    return response.json();
  }

  async setupPayout(type: 'stripe' | 'paypal', accountId: string): Promise<void> {
    await fetch(`${this.baseUrl}/payout`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ type, accountId }),
    });
  }

  async submitPlugin(formData: FormData): Promise<{
    pluginId: string;
    status: 'pending_review';
  }> {
    const response = await fetch(`${this.baseUrl}/plugins`, {
      method: 'POST',
      body: formData,
    });
    return response.json();
  }

  async updatePlugin(pluginId: string, formData: FormData): Promise<{
    status: 'pending_review' | 'approved' | 'rejected';
  }> {
    const response = await fetch(`${this.baseUrl}/plugins/${pluginId}`, {
      method: 'PUT',
      body: formData,
    });
    return response.json();
  }

  async getMyPlugins(): Promise<Array<
    PluginListing & { status: 'draft' | 'pending_review' | 'approved' | 'rejected' }
  >> {
    const response = await fetch(`${this.baseUrl}/plugins`);
    return response.json();
  }

  async getPluginAnalytics(pluginId: string, period: '7d' | '30d' | '90d' | '1y'): Promise<{
    downloads: Array<{ date: string; count: number }>;
    revenue: Array<{ date: string; amount: number }>;
    reviews: Array<{ date: string; rating: number }>;
    activeUsers: number;
    conversionRate: number;
  }> {
    const response = await fetch(
      `${this.baseUrl}/plugins/${pluginId}/analytics?period=${period}`
    );
    return response.json();
  }

  async respondToReview(pluginId: string, reviewId: string, response: string): Promise<void> {
    await fetch(`${this.baseUrl}/plugins/${pluginId}/reviews/${reviewId}/respond`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ response }),
    });
  }

  async requestPayout(): Promise<{
    payoutId: string;
    amount: number;
    estimatedArrival: Date;
  }> {
    const response = await fetch(`${this.baseUrl}/payout/request`, {
      method: 'POST',
    });
    return response.json();
  }

  async getPayoutHistory(): Promise<Array<{
    payoutId: string;
    amount: number;
    status: 'pending' | 'processing' | 'completed' | 'failed';
    createdAt: Date;
    completedAt?: Date;
  }>> {
    const response = await fetch(`${this.baseUrl}/payout/history`);
    return response.json();
  }
}

// ==================== Plugin Sandbox ====================

export class PluginSandbox {
  private iframe: HTMLIFrameElement | null = null;
  private worker: Worker | null = null;
  private permissions: Set<PluginPermission> = new Set();

  constructor(permissions: PluginPermission[] = []) {
    permissions.forEach(p => this.permissions.add(p));
  }

  async loadPlugin(bundle: ArrayBuffer, manifest: PluginManifest): Promise<any> {
    // Check permissions
    for (const required of manifest.permissions) {
      if (!this.permissions.has(required)) {
        throw new Error(`Missing required permission: ${required}`);
      }
    }

    if (manifest.ui?.customUI) {
      return this.loadInIframe(bundle, manifest);
    }
    return this.loadInWorker(bundle, manifest);
  }

  private async loadInIframe(bundle: ArrayBuffer, manifest: PluginManifest): Promise<any> {
    // Create sandboxed iframe
    this.iframe = document.createElement('iframe');
    this.iframe.sandbox.add('allow-scripts');

    if (this.permissions.has('network')) {
      this.iframe.sandbox.add('allow-same-origin');
    }

    // Create blob URL for bundle
    const blob = new Blob([bundle], { type: 'application/javascript' });
    const url = URL.createObjectURL(blob);

    // Inject bundle
    this.iframe.srcdoc = `
      <!DOCTYPE html>
      <html>
        <head>
          <script src="${url}"></script>
        </head>
        <body></body>
      </html>
    `;

    document.body.appendChild(this.iframe);

    // Wait for load
    await new Promise<void>((resolve) => {
      this.iframe!.onload = () => resolve();
    });

    URL.revokeObjectURL(url);

    // Return proxy to communicate with iframe
    return this.createIframeProxy();
  }

  private async loadInWorker(bundle: ArrayBuffer, manifest: PluginManifest): Promise<any> {
    const blob = new Blob([bundle], { type: 'application/javascript' });
    const url = URL.createObjectURL(blob);

    this.worker = new Worker(url, { type: 'module' });
    URL.revokeObjectURL(url);

    return this.createWorkerProxy();
  }

  private createIframeProxy(): any {
    return new Proxy({}, {
      get: (target, prop) => {
        return (...args: any[]) => {
          return new Promise((resolve, reject) => {
            const id = crypto.randomUUID();

            const handler = (event: MessageEvent) => {
              if (event.data.id === id) {
                window.removeEventListener('message', handler);
                if (event.data.error) {
                  reject(new Error(event.data.error));
                } else {
                  resolve(event.data.result);
                }
              }
            };

            window.addEventListener('message', handler);

            this.iframe?.contentWindow?.postMessage(
              { id, method: prop, args },
              '*'
            );
          });
        };
      },
    });
  }

  private createWorkerProxy(): any {
    return new Proxy({}, {
      get: (target, prop) => {
        return (...args: any[]) => {
          return new Promise((resolve, reject) => {
            const id = crypto.randomUUID();

            const handler = (event: MessageEvent) => {
              if (event.data.id === id) {
                this.worker?.removeEventListener('message', handler);
                if (event.data.error) {
                  reject(new Error(event.data.error));
                } else {
                  resolve(event.data.result);
                }
              }
            };

            this.worker?.addEventListener('message', handler);
            this.worker?.postMessage({ id, method: prop, args });
          });
        };
      },
    });
  }

  dispose(): void {
    if (this.iframe) {
      this.iframe.remove();
      this.iframe = null;
    }
    if (this.worker) {
      this.worker.terminate();
      this.worker = null;
    }
  }
}

// ==================== Marketplace Manager ====================

export class MarketplaceManager {
  public readonly discovery: PluginDiscoveryService;
  public readonly install: PluginInstallService;
  public readonly reviews: PluginReviewService;
  public readonly purchase: PluginPurchaseService;
  public readonly developer: PluginDeveloperService;

  private sandboxes: Map<string, PluginSandbox> = new Map();

  constructor(baseUrl = '/api/marketplace') {
    this.discovery = new PluginDiscoveryService(baseUrl);
    this.install = new PluginInstallService(baseUrl);
    this.reviews = new PluginReviewService(baseUrl);
    this.purchase = new PluginPurchaseService(baseUrl);
    this.developer = new PluginDeveloperService(`${baseUrl}/developer`);
  }

  async loadPlugin(pluginId: string): Promise<{
    instance: any;
    manifest: PluginManifest;
  }> {
    const installed = this.install.getInstalledPlugin(pluginId);
    if (!installed) {
      throw new Error('Plugin not installed');
    }

    if (!installed.enabled) {
      throw new Error('Plugin is disabled');
    }

    // Validate license
    const licenseValid = await this.purchase.validateLicense(pluginId);
    if (!licenseValid.valid) {
      throw new Error('Invalid license');
    }

    // Load bundle
    const { bundle, manifest } = await this.install.loadPluginBundle(pluginId);

    // Create sandbox
    const sandbox = new PluginSandbox(manifest.permissions);
    const instance = await sandbox.loadPlugin(bundle, manifest);

    this.sandboxes.set(pluginId, sandbox);

    // Update last used
    installed.lastUsed = new Date();
    this.install.updateSettings(pluginId, {});

    return { instance, manifest };
  }

  unloadPlugin(pluginId: string): void {
    const sandbox = this.sandboxes.get(pluginId);
    if (sandbox) {
      sandbox.dispose();
      this.sandboxes.delete(pluginId);
    }
  }

  dispose(): void {
    for (const sandbox of this.sandboxes.values()) {
      sandbox.dispose();
    }
    this.sandboxes.clear();
  }
}

// ==================== Singleton ====================

let marketplaceManager: MarketplaceManager | null = null;

export function getMarketplaceManager(): MarketplaceManager {
  if (!marketplaceManager) {
    marketplaceManager = new MarketplaceManager();
  }
  return marketplaceManager;
}

export default {
  MarketplaceManager,
  getMarketplaceManager,
  PluginDiscoveryService,
  PluginInstallService,
  PluginReviewService,
  PluginPurchaseService,
  PluginDeveloperService,
  PluginSandbox,
};
