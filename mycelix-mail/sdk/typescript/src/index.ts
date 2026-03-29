// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mycelix Mail SDK for TypeScript/JavaScript
 *
 * Official SDK for integrating with Mycelix Mail API
 *
 * @example
 * ```typescript
 * import { MycelixClient } from '@mycelix/sdk';
 *
 * const client = new MycelixClient({
 *   apiKey: 'your-api-key',
 *   baseUrl: 'https://api.mycelix.example.com',
 * });
 *
 * const emails = await client.emails.list({ folder: 'inbox' });
 * ```
 */

export interface MycelixConfig {
  apiKey: string;
  baseUrl?: string;
  timeout?: number;
  retries?: number;
}

export interface Email {
  id: string;
  messageId: string;
  threadId?: string;
  from: string;
  fromName?: string;
  to: string[];
  cc?: string[];
  bcc?: string[];
  subject: string;
  bodyText: string;
  bodyHtml?: string;
  snippet: string;
  date: string;
  receivedAt: string;
  folder: string;
  labels: string[];
  isRead: boolean;
  isStarred: boolean;
  hasAttachments: boolean;
  attachments?: Attachment[];
  trustScore?: number;
  headers?: Record<string, string>;
}

export interface Attachment {
  id: string;
  filename: string;
  contentType: string;
  size: number;
  contentId?: string;
}

export interface Contact {
  id: string;
  email: string;
  name?: string;
  phone?: string;
  company?: string;
  notes?: string;
  trustScore?: number;
  isFavorite: boolean;
  createdAt: string;
  updatedAt: string;
}

export interface TrustAttestation {
  id: string;
  fromEmail: string;
  toEmail: string;
  level: number;
  context: string;
  expiresAt?: string;
  createdAt: string;
  revokedAt?: string;
}

export interface TrustScore {
  email: string;
  score: number;
  level: 'very_high' | 'high' | 'medium' | 'low' | 'very_low' | 'unknown';
  directAttestations: number;
  indirectPath?: string[];
}

export interface SendEmailOptions {
  to: string[];
  cc?: string[];
  bcc?: string[];
  subject: string;
  body: string;
  bodyHtml?: string;
  attachments?: File[] | Blob[];
  replyTo?: string;
  scheduledAt?: string;
}

export interface ListEmailsOptions {
  folder?: string;
  labels?: string[];
  isRead?: boolean;
  isStarred?: boolean;
  from?: string;
  to?: string;
  subject?: string;
  after?: string;
  before?: string;
  limit?: number;
  offset?: number;
  orderBy?: 'date' | 'subject' | 'from';
  orderDir?: 'asc' | 'desc';
}

export interface SearchOptions {
  query: string;
  folders?: string[];
  dateFrom?: string;
  dateTo?: string;
  hasAttachments?: boolean;
  limit?: number;
}

export interface WebhookConfig {
  url: string;
  events: WebhookEvent[];
  secret?: string;
  active?: boolean;
}

export type WebhookEvent =
  | 'email.received'
  | 'email.sent'
  | 'email.read'
  | 'email.deleted'
  | 'contact.created'
  | 'contact.updated'
  | 'trust.attestation_created'
  | 'trust.attestation_revoked';

// API Response types
interface ApiResponse<T> {
  data: T;
  meta?: {
    total?: number;
    limit?: number;
    offset?: number;
  };
}

interface ApiError {
  code: string;
  message: string;
  details?: Record<string, unknown>;
}

/**
 * Main Mycelix Mail API Client
 */
export class MycelixClient {
  private config: Required<MycelixConfig>;
  public readonly emails: EmailsApi;
  public readonly contacts: ContactsApi;
  public readonly trust: TrustApi;
  public readonly webhooks: WebhooksApi;
  public readonly folders: FoldersApi;
  public readonly labels: LabelsApi;

  constructor(config: MycelixConfig) {
    this.config = {
      apiKey: config.apiKey,
      baseUrl: config.baseUrl || 'https://api.mycelix.mail',
      timeout: config.timeout || 30000,
      retries: config.retries || 3,
    };

    this.emails = new EmailsApi(this);
    this.contacts = new ContactsApi(this);
    this.trust = new TrustApi(this);
    this.webhooks = new WebhooksApi(this);
    this.folders = new FoldersApi(this);
    this.labels = new LabelsApi(this);
  }

  async request<T>(
    method: string,
    path: string,
    options?: {
      body?: unknown;
      query?: Record<string, string | number | boolean | undefined>;
    }
  ): Promise<T> {
    const url = new URL(path, this.config.baseUrl);

    if (options?.query) {
      Object.entries(options.query).forEach(([key, value]) => {
        if (value !== undefined) {
          url.searchParams.set(key, String(value));
        }
      });
    }

    const headers: Record<string, string> = {
      'Authorization': `Bearer ${this.config.apiKey}`,
      'Content-Type': 'application/json',
      'User-Agent': 'mycelix-sdk-js/1.0.0',
    };

    let lastError: Error | null = null;

    for (let attempt = 0; attempt < this.config.retries; attempt++) {
      try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), this.config.timeout);

        const response = await fetch(url.toString(), {
          method,
          headers,
          body: options?.body ? JSON.stringify(options.body) : undefined,
          signal: controller.signal,
        });

        clearTimeout(timeoutId);

        if (!response.ok) {
          const error: ApiError = await response.json();
          throw new MycelixError(error.code, error.message, response.status);
        }

        const data: ApiResponse<T> = await response.json();
        return data.data;
      } catch (error) {
        lastError = error as Error;
        if (attempt < this.config.retries - 1) {
          await new Promise(resolve => setTimeout(resolve, Math.pow(2, attempt) * 1000));
        }
      }
    }

    throw lastError || new Error('Request failed');
  }
}

/**
 * Custom error class for API errors
 */
export class MycelixError extends Error {
  constructor(
    public code: string,
    message: string,
    public status: number
  ) {
    super(message);
    this.name = 'MycelixError';
  }
}

/**
 * Emails API
 */
class EmailsApi {
  constructor(private client: MycelixClient) {}

  async list(options?: ListEmailsOptions): Promise<Email[]> {
    return this.client.request<Email[]>('GET', '/v1/emails', {
      query: options as Record<string, string | number | boolean | undefined>,
    });
  }

  async get(id: string): Promise<Email> {
    return this.client.request<Email>('GET', `/v1/emails/${id}`);
  }

  async send(options: SendEmailOptions): Promise<Email> {
    return this.client.request<Email>('POST', '/v1/emails/send', {
      body: options,
    });
  }

  async reply(id: string, body: string, replyAll?: boolean): Promise<Email> {
    return this.client.request<Email>('POST', `/v1/emails/${id}/reply`, {
      body: { body, replyAll },
    });
  }

  async forward(id: string, to: string[], body?: string): Promise<Email> {
    return this.client.request<Email>('POST', `/v1/emails/${id}/forward`, {
      body: { to, body },
    });
  }

  async markRead(id: string, read: boolean = true): Promise<void> {
    await this.client.request<void>('PATCH', `/v1/emails/${id}`, {
      body: { isRead: read },
    });
  }

  async star(id: string, starred: boolean = true): Promise<void> {
    await this.client.request<void>('PATCH', `/v1/emails/${id}`, {
      body: { isStarred: starred },
    });
  }

  async move(id: string, folder: string): Promise<void> {
    await this.client.request<void>('POST', `/v1/emails/${id}/move`, {
      body: { folder },
    });
  }

  async addLabel(id: string, label: string): Promise<void> {
    await this.client.request<void>('POST', `/v1/emails/${id}/labels`, {
      body: { label },
    });
  }

  async removeLabel(id: string, label: string): Promise<void> {
    await this.client.request<void>('DELETE', `/v1/emails/${id}/labels/${label}`);
  }

  async delete(id: string, permanent?: boolean): Promise<void> {
    await this.client.request<void>('DELETE', `/v1/emails/${id}`, {
      query: { permanent },
    });
  }

  async search(options: SearchOptions): Promise<Email[]> {
    return this.client.request<Email[]>('GET', '/v1/emails/search', {
      query: options as Record<string, string | number | boolean | undefined>,
    });
  }

  async getThread(threadId: string): Promise<Email[]> {
    return this.client.request<Email[]>('GET', `/v1/threads/${threadId}`);
  }
}

/**
 * Contacts API
 */
class ContactsApi {
  constructor(private client: MycelixClient) {}

  async list(options?: { limit?: number; offset?: number; search?: string }): Promise<Contact[]> {
    return this.client.request<Contact[]>('GET', '/v1/contacts', { query: options });
  }

  async get(id: string): Promise<Contact> {
    return this.client.request<Contact>('GET', `/v1/contacts/${id}`);
  }

  async create(contact: Omit<Contact, 'id' | 'createdAt' | 'updatedAt'>): Promise<Contact> {
    return this.client.request<Contact>('POST', '/v1/contacts', { body: contact });
  }

  async update(id: string, contact: Partial<Contact>): Promise<Contact> {
    return this.client.request<Contact>('PATCH', `/v1/contacts/${id}`, { body: contact });
  }

  async delete(id: string): Promise<void> {
    await this.client.request<void>('DELETE', `/v1/contacts/${id}`);
  }

  async getByEmail(email: string): Promise<Contact | null> {
    const contacts = await this.list({ search: email });
    return contacts.find(c => c.email === email) || null;
  }
}

/**
 * Trust API
 */
class TrustApi {
  constructor(private client: MycelixClient) {}

  async getScore(email: string): Promise<TrustScore> {
    return this.client.request<TrustScore>('GET', `/v1/trust/score/${encodeURIComponent(email)}`);
  }

  async getScores(emails: string[]): Promise<Record<string, TrustScore>> {
    return this.client.request<Record<string, TrustScore>>('POST', '/v1/trust/scores', {
      body: { emails },
    });
  }

  async createAttestation(
    toEmail: string,
    level: number,
    context: string,
    expiresAt?: string
  ): Promise<TrustAttestation> {
    return this.client.request<TrustAttestation>('POST', '/v1/trust/attestations', {
      body: { toEmail, level, context, expiresAt },
    });
  }

  async revokeAttestation(id: string): Promise<void> {
    await this.client.request<void>('DELETE', `/v1/trust/attestations/${id}`);
  }

  async listAttestations(
    type: 'given' | 'received' = 'given'
  ): Promise<TrustAttestation[]> {
    return this.client.request<TrustAttestation[]>('GET', '/v1/trust/attestations', {
      query: { type },
    });
  }

  async getTrustPath(fromEmail: string, toEmail: string): Promise<string[]> {
    return this.client.request<string[]>('GET', '/v1/trust/path', {
      query: { from: fromEmail, to: toEmail },
    });
  }
}

/**
 * Webhooks API
 */
class WebhooksApi {
  constructor(private client: MycelixClient) {}

  async list(): Promise<WebhookConfig[]> {
    return this.client.request<WebhookConfig[]>('GET', '/v1/webhooks');
  }

  async create(config: WebhookConfig): Promise<WebhookConfig & { id: string }> {
    return this.client.request<WebhookConfig & { id: string }>('POST', '/v1/webhooks', {
      body: config,
    });
  }

  async update(id: string, config: Partial<WebhookConfig>): Promise<WebhookConfig> {
    return this.client.request<WebhookConfig>('PATCH', `/v1/webhooks/${id}`, {
      body: config,
    });
  }

  async delete(id: string): Promise<void> {
    await this.client.request<void>('DELETE', `/v1/webhooks/${id}`);
  }

  async test(id: string): Promise<{ success: boolean; statusCode: number }> {
    return this.client.request<{ success: boolean; statusCode: number }>(
      'POST',
      `/v1/webhooks/${id}/test`
    );
  }
}

/**
 * Folders API
 */
class FoldersApi {
  constructor(private client: MycelixClient) {}

  async list(): Promise<{ name: string; count: number; unreadCount: number }[]> {
    return this.client.request('GET', '/v1/folders');
  }

  async create(name: string, parent?: string): Promise<{ name: string }> {
    return this.client.request('POST', '/v1/folders', { body: { name, parent } });
  }

  async rename(name: string, newName: string): Promise<void> {
    await this.client.request('PATCH', `/v1/folders/${encodeURIComponent(name)}`, {
      body: { name: newName },
    });
  }

  async delete(name: string): Promise<void> {
    await this.client.request('DELETE', `/v1/folders/${encodeURIComponent(name)}`);
  }
}

/**
 * Labels API
 */
class LabelsApi {
  constructor(private client: MycelixClient) {}

  async list(): Promise<{ name: string; color: string; count: number }[]> {
    return this.client.request('GET', '/v1/labels');
  }

  async create(name: string, color: string): Promise<{ name: string; color: string }> {
    return this.client.request('POST', '/v1/labels', { body: { name, color } });
  }

  async update(name: string, updates: { name?: string; color?: string }): Promise<void> {
    await this.client.request('PATCH', `/v1/labels/${encodeURIComponent(name)}`, {
      body: updates,
    });
  }

  async delete(name: string): Promise<void> {
    await this.client.request('DELETE', `/v1/labels/${encodeURIComponent(name)}`);
  }
}

// Export default instance factory
export function createClient(config: MycelixConfig): MycelixClient {
  return new MycelixClient(config);
}

export default MycelixClient;
