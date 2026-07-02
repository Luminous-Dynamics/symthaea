// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mycelix ERP API Client
 *
 * Unified client for all ERP modules:
 * - SCM (Supply Chain Management)
 * - FIN (Finance & Accounting)
 * - Auth (Authentication & Authorization)
 */

import axios, { AxiosInstance } from 'axios';
import {
  SupplyEventVC,
  EventResponse,
  ClaimResponse,
  VerifyRequest,
  VerifyResponse,
  HealthResponse,
  BatchRequest,
  BatchResponse,
  LineageResponse,
  BatchClaimsResponse,
  ClaimFilters,
  SearchResponse,
} from './types';
import { FinanceClient } from './finance-client';
import { AuthClient } from './auth';

export interface ClientConfig {
  baseUrl: string;
  timeout?: number;
  headers?: Record<string, string>;
  apiKey?: string;
}

/**
 * Mycelix ERP Client
 *
 * @example
 * ```typescript
 * const client = new MycelixClient({ baseUrl: 'http://localhost:8080' });
 *
 * // Health check
 * const health = await client.health();
 *
 * // Supply Chain: Create and track events
 * const event = client.scm.createProducedEvent({...});
 * const result = await client.scm.ingestEvent(event);
 *
 * // Finance: Create invoice
 * const invoice = await client.fin.createInvoice({...});
 *
 * // Auth: Login
 * await client.auth.login({ email: 'user@example.com', password: '...' });
 * ```
 */
export class MycelixClient {
  private client: AxiosInstance;

  /** Supply Chain Management module */
  public readonly scm: SupplyChainModule;

  /** Finance & Accounting module */
  public readonly fin: FinanceClient;

  /** Authentication & Authorization */
  public readonly auth: AuthClient;

  constructor(config: ClientConfig) {
    this.client = axios.create({
      baseURL: config.baseUrl,
      timeout: config.timeout || 30000,
      headers: {
        'Content-Type': 'application/json',
        ...(config.apiKey ? { Authorization: `Bearer ${config.apiKey}` } : {}),
        ...config.headers,
      },
    });

    // Initialize modules
    this.scm = new SupplyChainModule(this.client);
    this.fin = new FinanceClient(this.client);
    this.auth = new AuthClient(this.client);
  }

  /**
   * Check service health
   */
  async health(): Promise<HealthResponse> {
    const response = await this.client.get<HealthResponse>('/health');
    return response.data;
  }

  /**
   * Check detailed health with components
   */
  async healthDetailed(): Promise<DetailedHealthResponse> {
    const response = await this.client.get<DetailedHealthResponse>('/health');
    return response.data;
  }

  /**
   * Get service metrics (Prometheus format)
   */
  async metrics(): Promise<string> {
    const response = await this.client.get<string>('/metrics');
    return response.data;
  }
}

export interface DetailedHealthResponse {
  status: 'healthy' | 'degraded' | 'unhealthy';
  version: string;
  uptime_seconds: number;
  timestamp: string;
  components: ComponentHealth[];
}

export interface ComponentHealth {
  name: string;
  status: 'healthy' | 'degraded' | 'unhealthy';
  message?: string;
  latency_ms?: number;
}

/**
 * Supply Chain Management Module
 */
export class SupplyChainModule {
  constructor(private client: AxiosInstance) {}

  /**
   * Ingest a supply chain event
   */
  async ingestEvent(event: SupplyEventVC): Promise<EventResponse> {
    const response = await this.client.post<EventResponse>('/v1/events', event);
    return response.data;
  }

  /**
   * Get a claim by ID
   */
  async getClaim(claimId: string, includeLineage = true): Promise<ClaimResponse> {
    const response = await this.client.get<ClaimResponse>(`/v1/claims/${claimId}`, {
      params: { include_lineage: includeLineage },
    });
    return response.data;
  }

  /**
   * Get lineage for a batch
   */
  async getBatchLineage(batchId: string): Promise<ClaimResponse> {
    const response = await this.client.get<ClaimResponse>(`/v1/batches/${batchId}/lineage`);
    return response.data;
  }

  /**
   * Verify a Verifiable Credential
   */
  async verify(request: VerifyRequest): Promise<VerifyResponse> {
    const response = await this.client.post<VerifyResponse>('/v1/verify', request);
    return response.data;
  }

  /**
   * Ingest multiple events in a batch
   */
  async ingestBatch(request: BatchRequest): Promise<BatchResponse> {
    const response = await this.client.post<BatchResponse>('/v1/events/batch', {
      events: request.events,
      mode: request.mode || 'best-effort',
    });
    return response.data;
  }

  /**
   * Get all claims for a specific batch
   */
  async getBatchClaims(batchId: string): Promise<BatchClaimsResponse> {
    const response = await this.client.get<BatchClaimsResponse>(
      `/v1/batches/${batchId}/claims`
    );
    return response.data;
  }

  /**
   * Get complete lineage graph for a batch
   */
  async getLineage(batchId: string): Promise<LineageResponse> {
    const response = await this.client.get<LineageResponse>(
      `/v1/lineage/${batchId}`
    );
    return response.data;
  }

  /**
   * Search and filter claims
   */
  async searchClaims(filters: ClaimFilters = {}): Promise<SearchResponse> {
    const response = await this.client.get<SearchResponse>('/v1/claims', {
      params: filters,
    });
    return response.data;
  }

  // ============ Event Builders ============

  /**
   * Helper: Create a PRODUCED event
   */
  createProducedEvent(params: {
    issuer: string;
    productId: string;
    batchId: string;
    quantity: number;
    unit: string;
    facility: { id: string; name: string };
    timestamp?: string;
    metadata?: Record<string, unknown>;
  }): SupplyEventVC {
    return {
      '@context': ['https://www.w3.org/2018/credentials/v1', 'https://mycelix.org/contexts/supply-chain/v1'],
      type: ['VerifiableCredential', 'SupplyChainEvent'],
      issuer: params.issuer,
      issuanceDate: new Date().toISOString(),
      credentialSubject: {
        eventType: 'PRODUCED',
        productId: params.productId,
        batchId: params.batchId,
        quantity: params.quantity,
        unit: params.unit,
        facility: params.facility,
        timestamp: params.timestamp || new Date().toISOString(),
        metadata: params.metadata,
      },
    };
  }

  /**
   * Helper: Create a SHIPPED event
   */
  createShippedEvent(params: {
    issuer: string;
    productId: string;
    batchId: string;
    quantity: number;
    unit: string;
    facility: { id: string; name: string };
    shipment: {
      shipmentId: string;
      carrier?: string;
      trackingNumber?: string;
      origin?: string;
      destination?: string;
    };
    timestamp?: string;
  }): SupplyEventVC {
    return {
      '@context': ['https://www.w3.org/2018/credentials/v1', 'https://mycelix.org/contexts/supply-chain/v1'],
      type: ['VerifiableCredential', 'SupplyChainEvent'],
      issuer: params.issuer,
      issuanceDate: new Date().toISOString(),
      credentialSubject: {
        eventType: 'SHIPPED',
        productId: params.productId,
        batchId: params.batchId,
        quantity: params.quantity,
        unit: params.unit,
        facility: params.facility,
        timestamp: params.timestamp || new Date().toISOString(),
        shipment: params.shipment,
      },
    };
  }

  /**
   * Helper: Create a RECEIVED event
   */
  createReceivedEvent(params: {
    issuer: string;
    productId: string;
    batchId: string;
    quantity: number;
    unit: string;
    facility: { id: string; name: string };
    shipment?: {
      shipmentId: string;
    };
    timestamp?: string;
    metadata?: Record<string, unknown>;
  }): SupplyEventVC {
    return {
      '@context': ['https://www.w3.org/2018/credentials/v1', 'https://mycelix.org/contexts/supply-chain/v1'],
      type: ['VerifiableCredential', 'SupplyChainEvent'],
      issuer: params.issuer,
      issuanceDate: new Date().toISOString(),
      credentialSubject: {
        eventType: 'RECEIVED',
        productId: params.productId,
        batchId: params.batchId,
        quantity: params.quantity,
        unit: params.unit,
        facility: params.facility,
        timestamp: params.timestamp || new Date().toISOString(),
        shipment: params.shipment,
        metadata: params.metadata,
      },
    };
  }

  /**
   * Helper: Create a TRANSFORMED event
   */
  createTransformedEvent(params: {
    issuer: string;
    productId: string;
    batchId: string;
    prevBatchIds: string[];
    quantity: number;
    unit: string;
    facility: { id: string; name: string };
    timestamp?: string;
    metadata?: Record<string, unknown>;
  }): SupplyEventVC {
    return {
      '@context': ['https://www.w3.org/2018/credentials/v1', 'https://mycelix.org/contexts/supply-chain/v1'],
      type: ['VerifiableCredential', 'SupplyChainEvent'],
      issuer: params.issuer,
      issuanceDate: new Date().toISOString(),
      credentialSubject: {
        eventType: 'TRANSFORMED',
        productId: params.productId,
        batchId: params.batchId,
        prevBatchIds: params.prevBatchIds,
        quantity: params.quantity,
        unit: params.unit,
        facility: params.facility,
        timestamp: params.timestamp || new Date().toISOString(),
        metadata: params.metadata,
      },
    };
  }

  /**
   * Helper: Create a CERTIFIED event
   */
  createCertifiedEvent(params: {
    issuer: string;
    productId: string;
    batchId: string;
    quantity: number;
    unit: string;
    facility: { id: string; name: string };
    certification: {
      certType: string;
      certBody: string;
      certId: string;
      validFrom: string;
      validUntil: string;
    };
    timestamp?: string;
  }): SupplyEventVC {
    return {
      '@context': ['https://www.w3.org/2018/credentials/v1', 'https://mycelix.org/contexts/supply-chain/v1'],
      type: ['VerifiableCredential', 'SupplyChainEvent'],
      issuer: params.issuer,
      issuanceDate: new Date().toISOString(),
      credentialSubject: {
        eventType: 'CERTIFIED',
        productId: params.productId,
        batchId: params.batchId,
        quantity: params.quantity,
        unit: params.unit,
        facility: params.facility,
        timestamp: params.timestamp || new Date().toISOString(),
        certification: params.certification,
      },
    };
  }

  /**
   * Helper: Create a batch request
   */
  createBatch(
    events: SupplyEventVC[],
    mode: 'best-effort' | 'atomic' = 'best-effort'
  ): BatchRequest {
    return { events, mode };
  }
}

// Legacy export for backwards compatibility
export { MycelixClient as SupplyChainClient };
