// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * @mycelix/sdk Pagination Utilities
 *
 * Standardized pagination for all Mycelix APIs.
 * Supports cursor-based and offset-based pagination.
 *
 * @packageDocumentation
 * @module utils/pagination
 */

/** Default page size when not specified */
export const DEFAULT_PAGE_SIZE = 50;

/** Maximum allowed page size to prevent resource exhaustion */
export const MAX_PAGE_SIZE = 500;

/**
 * Pagination request parameters
 */
export interface PaginationRequest {
  /** Number of items per page (capped at MAX_PAGE_SIZE) */
  pageSize?: number;
  /** Cursor for cursor-based pagination (opaque string) */
  cursor?: string;
  /** Offset for offset-based pagination */
  offset?: number;
  /** Sort direction (true = descending/newest first) */
  sortDesc?: boolean;
}

/**
 * Paginated response wrapper
 */
export interface PaginatedResponse<T> {
  /** Items in this page */
  items: T[];
  /** Total count of items (if available) */
  totalCount?: number;
  /** Cursor for next page (if cursor-based) */
  nextCursor?: string;
  /** Whether there are more items */
  hasMore: boolean;
  /** Current page number (if offset-based) */
  page?: number;
  /** Total pages (if offset-based and total known) */
  totalPages?: number;
}

/**
 * Create a pagination request with defaults
 */
export function createPaginationRequest(
  pageSize: number = DEFAULT_PAGE_SIZE,
  cursor?: string
): PaginationRequest {
  return {
    pageSize: Math.min(pageSize, MAX_PAGE_SIZE),
    cursor,
    sortDesc: true,
  };
}

/**
 * Create a pagination request with offset
 */
export function createOffsetPaginationRequest(
  pageSize: number = DEFAULT_PAGE_SIZE,
  offset: number = 0
): PaginationRequest {
  return {
    pageSize: Math.min(pageSize, MAX_PAGE_SIZE),
    offset,
    sortDesc: true,
  };
}

/**
 * Get effective page size (capped at MAX_PAGE_SIZE)
 */
export function getEffectivePageSize(request: PaginationRequest): number {
  return Math.min(request.pageSize ?? DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE);
}

/**
 * Get effective offset
 */
export function getEffectiveOffset(request: PaginationRequest): number {
  return request.offset ?? 0;
}

/**
 * Paginate an array of items
 */
export function paginateArray<T>(
  items: T[],
  request: PaginationRequest
): PaginatedResponse<T> {
  const total = items.length;
  const offset = getEffectiveOffset(request);
  const pageSize = getEffectivePageSize(request);

  if (offset >= total) {
    return {
      items: [],
      totalCount: total,
      hasMore: false,
      page: 0,
      totalPages: Math.ceil(total / pageSize),
    };
  }

  const end = Math.min(offset + pageSize, total);
  const pageItems = items.slice(offset, end);
  const hasMore = end < total;

  return {
    items: pageItems,
    totalCount: total,
    nextCursor: hasMore ? end.toString() : undefined,
    hasMore,
    page: Math.floor(offset / pageSize),
    totalPages: Math.ceil(total / pageSize),
  };
}

/**
 * Create an empty paginated response
 */
export function emptyPaginatedResponse<T>(): PaginatedResponse<T> {
  return {
    items: [],
    totalCount: 0,
    hasMore: false,
    page: 0,
    totalPages: 0,
  };
}

/**
 * Cursor encoding/decoding utilities
 */
export const cursor = {
  /**
   * Encode a cursor from timestamp and optional ID
   */
  encode(timestampMicros: number, id?: string): string {
    return id ? `${timestampMicros}:${id}` : timestampMicros.toString();
  },

  /**
   * Decode a cursor to timestamp and optional ID
   */
  decode(cursor: string): { timestamp: number; id?: string } | null {
    if (cursor.includes(':')) {
      const [tsStr, ...idParts] = cursor.split(':');
      const timestamp = parseInt(tsStr, 10);
      if (isNaN(timestamp)) return null;
      return { timestamp, id: idParts.join(':') };
    }
    const timestamp = parseInt(cursor, 10);
    if (isNaN(timestamp)) return null;
    return { timestamp };
  },
};

/**
 * Pagination helper class for building paginated queries
 */
export class Paginator<T> {
  private items: T[] = [];
  private pageSize: number;
  private currentOffset: number = 0;

  constructor(pageSize: number = DEFAULT_PAGE_SIZE) {
    this.pageSize = Math.min(pageSize, MAX_PAGE_SIZE);
  }

  /**
   * Set items to paginate
   */
  setItems(items: T[]): this {
    this.items = items;
    return this;
  }

  /**
   * Get current page
   */
  getCurrentPage(): PaginatedResponse<T> {
    return paginateArray(this.items, {
      pageSize: this.pageSize,
      offset: this.currentOffset,
    });
  }

  /**
   * Go to next page
   */
  nextPage(): PaginatedResponse<T> | null {
    const current = this.getCurrentPage();
    if (!current.hasMore) return null;
    this.currentOffset += this.pageSize;
    return this.getCurrentPage();
  }

  /**
   * Go to previous page
   */
  prevPage(): PaginatedResponse<T> | null {
    if (this.currentOffset === 0) return null;
    this.currentOffset = Math.max(0, this.currentOffset - this.pageSize);
    return this.getCurrentPage();
  }

  /**
   * Go to specific page
   */
  goToPage(page: number): PaginatedResponse<T> {
    this.currentOffset = page * this.pageSize;
    return this.getCurrentPage();
  }

  /**
   * Reset to first page
   */
  reset(): this {
    this.currentOffset = 0;
    return this;
  }

  /**
   * Get all pages as an async iterator
   */
  async *pages(): AsyncGenerator<PaginatedResponse<T>> {
    this.reset();
    let page = this.getCurrentPage();
    yield page;

    while (page.hasMore) {
      const next = this.nextPage();
      if (!next) break;
      page = next;
      yield page;
    }
  }
}

/**
 * Create a paginator instance
 */
export function createPaginator<T>(pageSize?: number): Paginator<T> {
  return new Paginator<T>(pageSize);
}

/**
 * Pagination utilities for async data fetching
 */
export const asyncPagination = {
  /**
   * Fetch all pages and combine results
   */
  async fetchAll<T>(
    fetchFn: (request: PaginationRequest) => Promise<PaginatedResponse<T>>,
    initialRequest: PaginationRequest = { pageSize: DEFAULT_PAGE_SIZE }
  ): Promise<T[]> {
    const allItems: T[] = [];
    let request = { ...initialRequest };
    let hasMore = true;

    while (hasMore) {
      const response = await fetchFn(request);
      allItems.push(...response.items);
      hasMore = response.hasMore;

      if (hasMore && response.nextCursor) {
        request = { ...request, cursor: response.nextCursor };
      } else if (hasMore) {
        request = {
          ...request,
          offset: (request.offset ?? 0) + getEffectivePageSize(request),
        };
      }

      // Safety limit
      if (allItems.length > 10000) {
        console.warn('Pagination safety limit reached (10000 items)');
        break;
      }
    }

    return allItems;
  },

  /**
   * Create an async iterator for paginated results
   */
  async *iterate<T>(
    fetchFn: (request: PaginationRequest) => Promise<PaginatedResponse<T>>,
    initialRequest: PaginationRequest = { pageSize: DEFAULT_PAGE_SIZE }
  ): AsyncGenerator<T> {
    let request = { ...initialRequest };
    let hasMore = true;

    while (hasMore) {
      const response = await fetchFn(request);

      for (const item of response.items) {
        yield item;
      }

      hasMore = response.hasMore;

      if (hasMore && response.nextCursor) {
        request = { ...request, cursor: response.nextCursor };
      } else if (hasMore) {
        request = {
          ...request,
          offset: (request.offset ?? 0) + getEffectivePageSize(request),
        };
      }
    }
  },
};
