// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Authentication types and utilities for Mycelix ERP
 */

import { AxiosInstance } from 'axios';

// Auth types
export interface LoginRequest {
  email: string;
  password: string;
  tenant_id?: string;
}

export interface LoginResponse {
  access_token: string;
  refresh_token: string;
  token_type: 'Bearer';
  expires_in: number;
  user: User;
  tenant?: Tenant;
}

export interface RefreshRequest {
  refresh_token: string;
}

export interface User {
  id: string;
  email: string;
  name: string;
  role: UserRole;
  tenant_id?: string;
  is_active: boolean;
  mfa_enabled: boolean;
  created_at: string;
  last_login?: string;
}

export type UserRole =
  | 'ADMIN'
  | 'MANAGER'
  | 'ACCOUNTANT'
  | 'WAREHOUSE'
  | 'SALES'
  | 'VIEWER';

export interface Tenant {
  id: string;
  name: string;
  slug: string;
  plan: 'STARTER' | 'PROFESSIONAL' | 'ENTERPRISE';
  is_active: boolean;
  settings: TenantSettings;
  created_at: string;
}

export interface TenantSettings {
  currency: string;
  timezone: string;
  date_format: string;
  fiscal_year_start: string;
}

export interface RegisterRequest {
  email: string;
  password: string;
  name: string;
  company_name?: string;
}

export interface RegisterResponse {
  user: User;
  tenant?: Tenant;
  message: string;
}

export interface PasswordResetRequest {
  email: string;
}

export interface PasswordResetConfirm {
  token: string;
  new_password: string;
}

export interface ChangePasswordRequest {
  current_password: string;
  new_password: string;
}

export interface ApiKeyResponse {
  id: string;
  name: string;
  key_prefix: string;
  key?: string; // Only returned on creation
  scopes: string[];
  expires_at?: string;
  created_at: string;
  last_used_at?: string;
}

export interface CreateApiKeyRequest {
  name: string;
  scopes: string[];
  expires_in_days?: number;
}

/**
 * Authentication client for Mycelix ERP
 */
export class AuthClient {
  private accessToken?: string;
  private refreshToken?: string;
  private tokenExpiry?: Date;

  constructor(private client: AxiosInstance) {}

  /**
   * Login with email and password
   */
  async login(request: LoginRequest): Promise<LoginResponse> {
    const response = await this.client.post<LoginResponse>(
      '/v1/auth/login',
      request
    );
    this.setTokens(response.data);
    return response.data;
  }

  /**
   * Register a new user/tenant
   */
  async register(request: RegisterRequest): Promise<RegisterResponse> {
    const response = await this.client.post<RegisterResponse>(
      '/v1/auth/register',
      request
    );
    return response.data;
  }

  /**
   * Refresh the access token
   */
  async refresh(): Promise<LoginResponse> {
    if (!this.refreshToken) {
      throw new Error('No refresh token available');
    }
    const response = await this.client.post<LoginResponse>(
      '/v1/auth/refresh',
      { refresh_token: this.refreshToken }
    );
    this.setTokens(response.data);
    return response.data;
  }

  /**
   * Logout (invalidate tokens)
   */
  async logout(): Promise<void> {
    await this.client.post('/v1/auth/logout');
    this.clearTokens();
  }

  /**
   * Get current user profile
   */
  async getProfile(): Promise<User> {
    const response = await this.client.get<User>('/v1/auth/me');
    return response.data;
  }

  /**
   * Update user profile
   */
  async updateProfile(updates: Partial<Pick<User, 'name'>>): Promise<User> {
    const response = await this.client.patch<User>('/v1/auth/me', updates);
    return response.data;
  }

  /**
   * Request password reset
   */
  async requestPasswordReset(email: string): Promise<void> {
    await this.client.post('/v1/auth/password-reset', { email });
  }

  /**
   * Confirm password reset
   */
  async confirmPasswordReset(
    token: string,
    newPassword: string
  ): Promise<void> {
    await this.client.post('/v1/auth/password-reset/confirm', {
      token,
      new_password: newPassword,
    });
  }

  /**
   * Change password (when logged in)
   */
  async changePassword(
    currentPassword: string,
    newPassword: string
  ): Promise<void> {
    await this.client.post('/v1/auth/change-password', {
      current_password: currentPassword,
      new_password: newPassword,
    });
  }

  /**
   * List API keys
   */
  async listApiKeys(): Promise<ApiKeyResponse[]> {
    const response = await this.client.get<ApiKeyResponse[]>(
      '/v1/auth/api-keys'
    );
    return response.data;
  }

  /**
   * Create API key
   */
  async createApiKey(request: CreateApiKeyRequest): Promise<ApiKeyResponse> {
    const response = await this.client.post<ApiKeyResponse>(
      '/v1/auth/api-keys',
      request
    );
    return response.data;
  }

  /**
   * Revoke API key
   */
  async revokeApiKey(keyId: string): Promise<void> {
    await this.client.delete(`/v1/auth/api-keys/${keyId}`);
  }

  // Token management

  private setTokens(response: LoginResponse): void {
    this.accessToken = response.access_token;
    this.refreshToken = response.refresh_token;
    this.tokenExpiry = new Date(Date.now() + response.expires_in * 1000);

    // Set authorization header for future requests
    this.client.defaults.headers.common['Authorization'] =
      `Bearer ${this.accessToken}`;
  }

  private clearTokens(): void {
    this.accessToken = undefined;
    this.refreshToken = undefined;
    this.tokenExpiry = undefined;
    delete this.client.defaults.headers.common['Authorization'];
  }

  /**
   * Get current access token
   */
  getAccessToken(): string | undefined {
    return this.accessToken;
  }

  /**
   * Check if user is authenticated
   */
  isAuthenticated(): boolean {
    return !!this.accessToken && (!this.tokenExpiry || this.tokenExpiry > new Date());
  }

  /**
   * Set token manually (for API key auth)
   */
  setApiKey(apiKey: string): void {
    this.client.defaults.headers.common['Authorization'] = `Bearer ${apiKey}`;
  }

  /**
   * Set token manually (for external auth flows)
   */
  setToken(accessToken: string, refreshToken?: string, expiresIn?: number): void {
    this.accessToken = accessToken;
    this.refreshToken = refreshToken;
    if (expiresIn) {
      this.tokenExpiry = new Date(Date.now() + expiresIn * 1000);
    }
    this.client.defaults.headers.common['Authorization'] = `Bearer ${accessToken}`;
  }
}
