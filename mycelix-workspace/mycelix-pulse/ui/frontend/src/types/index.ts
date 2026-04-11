// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
export interface User {
  id: string;
  email: string;
  firstName?: string;
  lastName?: string;
  createdAt: string;
  updatedAt: string;
}

export interface EmailAccount {
  id: string;
  email: string;
  provider: string;
  imapHost: string;
  imapPort: number;
  imapSecure: boolean;
  smtpHost: string;
  smtpPort: number;
  smtpSecure: boolean;
  isDefault: boolean;
  lastSyncedAt?: string;
  createdAt: string;
  updatedAt: string;
}

export interface Folder {
  id: string;
  name: string;
  path: string;
  type: FolderType;
  unreadCount: number;
  totalCount: number;
  createdAt: string;
  updatedAt: string;
}

export type FolderType = 'INBOX' | 'SENT' | 'DRAFTS' | 'TRASH' | 'SPAM' | 'ARCHIVE' | 'CUSTOM';

export interface EmailAddress {
  name: string;
  address: string;
}

export interface Email {
  id: string;
  messageId: string;
  subject: string;
  from: EmailAddress;
  to: EmailAddress[];
  cc?: EmailAddress[];
  bcc?: EmailAddress[];
  bodyText?: string;
  bodyHtml?: string;
  isRead: boolean;
  isStarred: boolean;
  isDraft: boolean;
  date: string;
  size: number;
  folder?: {
    name: string;
    type: string;
  };
  attachments?: Attachment[];
  // Trust & safety metadata (optional; supplied by Holochain/MATL layer or heuristics)
  trustScore?: number;
  trustTier?: 'high' | 'medium' | 'low' | 'unknown';
  trustReasons?: string[];
  trustPathLength?: number;
  trustDecayAt?: string;
  isQuarantined?: boolean;
  quarantineReason?: string;
  createdAt: string;
  updatedAt: string;
}

export interface Attachment {
  id: string;
  filename: string;
  contentType: string;
  size: number;
  contentId?: string;
}

export interface LoginCredentials {
  email: string;
  password: string;
}

export interface RegisterData extends LoginCredentials {
  firstName?: string;
  lastName?: string;
}

export interface ApiResponse<T> {
  status: 'success' | 'error';
  data: T;
  message?: string;
}

export interface PaginatedResponse<T> {
  items: T[];
  pagination: {
    page: number;
    limit: number;
    total: number;
    pages: number;
  };
}
