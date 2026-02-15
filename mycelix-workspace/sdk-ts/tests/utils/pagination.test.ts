/**
 * Pagination module tests
 * Tests for @mycelix/sdk pagination utilities
 */

import { describe, it, expect, vi } from 'vitest';
import {
  DEFAULT_PAGE_SIZE,
  MAX_PAGE_SIZE,
  createPaginationRequest,
  createOffsetPaginationRequest,
  getEffectivePageSize,
  getEffectiveOffset,
  paginateArray,
  emptyPaginatedResponse,
  cursor,
  Paginator,
  createPaginator,
  asyncPagination,
  type PaginationRequest,
  type PaginatedResponse,
} from '../../src/utils/pagination.js';

describe('Pagination Constants', () => {
  it('should have correct default page size', () => {
    expect(DEFAULT_PAGE_SIZE).toBe(50);
  });

  it('should have correct max page size', () => {
    expect(MAX_PAGE_SIZE).toBe(500);
  });
});

describe('createPaginationRequest', () => {
  it('should create request with default page size', () => {
    const request = createPaginationRequest();
    expect(request.pageSize).toBe(DEFAULT_PAGE_SIZE);
    expect(request.cursor).toBeUndefined();
    expect(request.sortDesc).toBe(true);
  });

  it('should create request with custom page size', () => {
    const request = createPaginationRequest(100);
    expect(request.pageSize).toBe(100);
  });

  it('should cap page size at MAX_PAGE_SIZE', () => {
    const request = createPaginationRequest(1000);
    expect(request.pageSize).toBe(MAX_PAGE_SIZE);
  });

  it('should include cursor when provided', () => {
    const request = createPaginationRequest(50, 'abc123');
    expect(request.cursor).toBe('abc123');
  });
});

describe('createOffsetPaginationRequest', () => {
  it('should create request with default values', () => {
    const request = createOffsetPaginationRequest();
    expect(request.pageSize).toBe(DEFAULT_PAGE_SIZE);
    expect(request.offset).toBe(0);
    expect(request.sortDesc).toBe(true);
  });

  it('should create request with custom page size and offset', () => {
    const request = createOffsetPaginationRequest(25, 100);
    expect(request.pageSize).toBe(25);
    expect(request.offset).toBe(100);
  });

  it('should cap page size at MAX_PAGE_SIZE', () => {
    const request = createOffsetPaginationRequest(1000, 0);
    expect(request.pageSize).toBe(MAX_PAGE_SIZE);
  });
});

describe('getEffectivePageSize', () => {
  it('should return page size from request', () => {
    expect(getEffectivePageSize({ pageSize: 100 })).toBe(100);
  });

  it('should return default when page size not specified', () => {
    expect(getEffectivePageSize({})).toBe(DEFAULT_PAGE_SIZE);
  });

  it('should cap at MAX_PAGE_SIZE', () => {
    expect(getEffectivePageSize({ pageSize: 1000 })).toBe(MAX_PAGE_SIZE);
  });
});

describe('getEffectiveOffset', () => {
  it('should return offset from request', () => {
    expect(getEffectiveOffset({ offset: 50 })).toBe(50);
  });

  it('should return 0 when offset not specified', () => {
    expect(getEffectiveOffset({})).toBe(0);
  });
});

describe('paginateArray', () => {
  const items = Array.from({ length: 100 }, (_, i) => ({ id: i, name: `Item ${i}` }));

  it('should paginate array with default settings', () => {
    const result = paginateArray(items, { pageSize: 10 });
    expect(result.items.length).toBe(10);
    expect(result.totalCount).toBe(100);
    expect(result.hasMore).toBe(true);
    expect(result.page).toBe(0);
    expect(result.totalPages).toBe(10);
    expect(result.nextCursor).toBe('10');
  });

  it('should handle offset', () => {
    const result = paginateArray(items, { pageSize: 10, offset: 20 });
    expect(result.items[0]).toEqual({ id: 20, name: 'Item 20' });
    expect(result.page).toBe(2);
    expect(result.hasMore).toBe(true);
  });

  it('should handle last page', () => {
    const result = paginateArray(items, { pageSize: 10, offset: 90 });
    expect(result.items.length).toBe(10);
    expect(result.hasMore).toBe(false);
    expect(result.nextCursor).toBeUndefined();
  });

  it('should handle offset beyond array bounds', () => {
    const result = paginateArray(items, { pageSize: 10, offset: 200 });
    expect(result.items.length).toBe(0);
    expect(result.hasMore).toBe(false);
    expect(result.page).toBe(0);
    expect(result.totalPages).toBe(10);
  });

  it('should handle empty array', () => {
    const result = paginateArray([], { pageSize: 10 });
    expect(result.items.length).toBe(0);
    expect(result.totalCount).toBe(0);
    expect(result.hasMore).toBe(false);
    expect(result.totalPages).toBe(0);
  });

  it('should handle page size larger than array', () => {
    const smallArray = [1, 2, 3];
    const result = paginateArray(smallArray, { pageSize: 10 });
    expect(result.items).toEqual([1, 2, 3]);
    expect(result.hasMore).toBe(false);
    expect(result.totalPages).toBe(1);
  });

  it('should correctly calculate partial last page', () => {
    const items = Array.from({ length: 25 }, (_, i) => i);
    const result = paginateArray(items, { pageSize: 10, offset: 20 });
    expect(result.items.length).toBe(5);
    expect(result.hasMore).toBe(false);
    expect(result.page).toBe(2);
  });
});

describe('emptyPaginatedResponse', () => {
  it('should return empty response with correct structure', () => {
    const response = emptyPaginatedResponse<string>();
    expect(response.items).toEqual([]);
    expect(response.totalCount).toBe(0);
    expect(response.hasMore).toBe(false);
    expect(response.page).toBe(0);
    expect(response.totalPages).toBe(0);
  });

  it('should be typed correctly', () => {
    const response = emptyPaginatedResponse<{ id: number }>();
    // TypeScript should infer the correct type
    const items: { id: number }[] = response.items;
    expect(items).toEqual([]);
  });
});

describe('cursor encoding/decoding', () => {
  describe('cursor.encode', () => {
    it('should encode timestamp only', () => {
      const encoded = cursor.encode(1234567890);
      expect(encoded).toBe('1234567890');
    });

    it('should encode timestamp with ID', () => {
      const encoded = cursor.encode(1234567890, 'abc123');
      expect(encoded).toBe('1234567890:abc123');
    });

    it('should handle ID with colons', () => {
      const encoded = cursor.encode(1234567890, 'id:with:colons');
      expect(encoded).toBe('1234567890:id:with:colons');
    });
  });

  describe('cursor.decode', () => {
    it('should decode timestamp only', () => {
      const decoded = cursor.decode('1234567890');
      expect(decoded).toEqual({ timestamp: 1234567890 });
      expect(decoded?.id).toBeUndefined();
    });

    it('should decode timestamp with ID', () => {
      const decoded = cursor.decode('1234567890:abc123');
      expect(decoded).toEqual({ timestamp: 1234567890, id: 'abc123' });
    });

    it('should handle ID with colons', () => {
      const decoded = cursor.decode('1234567890:id:with:colons');
      expect(decoded).toEqual({ timestamp: 1234567890, id: 'id:with:colons' });
    });

    it('should return null for invalid timestamp', () => {
      expect(cursor.decode('invalid')).toBeNull();
    });

    it('should return null for invalid timestamp with colon', () => {
      expect(cursor.decode('invalid:id')).toBeNull();
    });

    it('should handle empty string after colon', () => {
      const decoded = cursor.decode('1234567890:');
      expect(decoded).toEqual({ timestamp: 1234567890, id: '' });
    });
  });

  it('should round-trip encode/decode', () => {
    const original = { timestamp: 1704067200000, id: 'test-item-42' };
    const encoded = cursor.encode(original.timestamp, original.id);
    const decoded = cursor.decode(encoded);
    expect(decoded).toEqual(original);
  });
});

describe('Paginator class', () => {
  const createTestItems = (count: number) =>
    Array.from({ length: count }, (_, i) => ({ id: i, value: `value-${i}` }));

  describe('constructor', () => {
    it('should use default page size', () => {
      const paginator = new Paginator();
      paginator.setItems(createTestItems(100));
      const page = paginator.getCurrentPage();
      expect(page.items.length).toBe(DEFAULT_PAGE_SIZE);
    });

    it('should cap page size at MAX_PAGE_SIZE', () => {
      const paginator = new Paginator(1000);
      paginator.setItems(createTestItems(1000));
      const page = paginator.getCurrentPage();
      expect(page.items.length).toBe(MAX_PAGE_SIZE);
    });
  });

  describe('setItems', () => {
    it('should return this for chaining', () => {
      const paginator = new Paginator(10);
      const result = paginator.setItems([1, 2, 3]);
      expect(result).toBe(paginator);
    });
  });

  describe('getCurrentPage', () => {
    it('should return first page initially', () => {
      const paginator = new Paginator<number>(10);
      paginator.setItems(createTestItems(25));
      const page = paginator.getCurrentPage();
      expect(page.items.length).toBe(10);
      expect(page.page).toBe(0);
      expect(page.hasMore).toBe(true);
    });
  });

  describe('nextPage', () => {
    it('should advance to next page', () => {
      const paginator = new Paginator(10);
      paginator.setItems(createTestItems(25));

      const page1 = paginator.getCurrentPage();
      expect(page1.items[0]).toEqual({ id: 0, value: 'value-0' });

      const page2 = paginator.nextPage();
      expect(page2).not.toBeNull();
      expect(page2!.items[0]).toEqual({ id: 10, value: 'value-10' });
      expect(page2!.page).toBe(1);
    });

    it('should return null when no more pages', () => {
      const paginator = new Paginator(10);
      paginator.setItems(createTestItems(5));

      const result = paginator.nextPage();
      expect(result).toBeNull();
    });
  });

  describe('prevPage', () => {
    it('should go to previous page', () => {
      const paginator = new Paginator(10);
      paginator.setItems(createTestItems(30));

      paginator.goToPage(2);
      const page = paginator.prevPage();
      expect(page).not.toBeNull();
      expect(page!.page).toBe(1);
    });

    it('should return null on first page', () => {
      const paginator = new Paginator(10);
      paginator.setItems(createTestItems(30));

      const result = paginator.prevPage();
      expect(result).toBeNull();
    });
  });

  describe('goToPage', () => {
    it('should jump to specific page', () => {
      const paginator = new Paginator(10);
      paginator.setItems(createTestItems(100));

      const page = paginator.goToPage(5);
      expect(page.page).toBe(5);
      expect(page.items[0]).toEqual({ id: 50, value: 'value-50' });
    });

    it('should handle out of bounds page', () => {
      const paginator = new Paginator(10);
      paginator.setItems(createTestItems(25));

      const page = paginator.goToPage(100);
      expect(page.items.length).toBe(0);
    });
  });

  describe('reset', () => {
    it('should reset to first page', () => {
      const paginator = new Paginator(10);
      paginator.setItems(createTestItems(30));
      paginator.goToPage(2);

      const result = paginator.reset();
      expect(result).toBe(paginator); // Returns this
      expect(paginator.getCurrentPage().page).toBe(0);
    });
  });

  describe('pages async iterator', () => {
    it('should iterate through all pages', async () => {
      const paginator = new Paginator<{ id: number; value: string }>(10);
      paginator.setItems(createTestItems(25));

      const allPages: PaginatedResponse<{ id: number; value: string }>[] = [];
      for await (const page of paginator.pages()) {
        allPages.push(page);
      }

      expect(allPages.length).toBe(3);
      expect(allPages[0].page).toBe(0);
      expect(allPages[1].page).toBe(1);
      expect(allPages[2].page).toBe(2);
    });

    it('should handle empty items', async () => {
      const paginator = new Paginator<number>(10);
      paginator.setItems([]);

      const allPages = [];
      for await (const page of paginator.pages()) {
        allPages.push(page);
      }

      expect(allPages.length).toBe(1);
      expect(allPages[0].items.length).toBe(0);
    });
  });
});

describe('createPaginator', () => {
  it('should create paginator with default page size', () => {
    const paginator = createPaginator<string>();
    paginator.setItems(['a', 'b', 'c']);
    expect(paginator.getCurrentPage().items).toEqual(['a', 'b', 'c']);
  });

  it('should create paginator with custom page size', () => {
    const paginator = createPaginator<number>(2);
    paginator.setItems([1, 2, 3, 4, 5]);
    expect(paginator.getCurrentPage().items).toEqual([1, 2]);
  });
});

describe('asyncPagination', () => {
  describe('fetchAll', () => {
    it('should fetch all pages and combine results', async () => {
      const allItems = Array.from({ length: 25 }, (_, i) => i);

      const fetchFn = async (request: PaginationRequest): Promise<PaginatedResponse<number>> => {
        const pageSize = request.pageSize ?? 10;
        const offset = request.offset ?? 0;
        const end = Math.min(offset + pageSize, allItems.length);
        const items = allItems.slice(offset, end);

        return {
          items,
          totalCount: allItems.length,
          hasMore: end < allItems.length,
        };
      };

      const result = await asyncPagination.fetchAll(fetchFn, { pageSize: 10, offset: 0 });
      expect(result).toEqual(allItems);
    });

    it('should handle cursor-based pagination', async () => {
      const allItems = ['a', 'b', 'c', 'd', 'e'];
      let currentIndex = 0;

      const fetchFn = async (request: PaginationRequest): Promise<PaginatedResponse<string>> => {
        if (request.cursor) {
          currentIndex = parseInt(request.cursor, 10);
        }
        const pageSize = request.pageSize ?? 2;
        const end = Math.min(currentIndex + pageSize, allItems.length);
        const items = allItems.slice(currentIndex, end);

        return {
          items,
          hasMore: end < allItems.length,
          nextCursor: end < allItems.length ? end.toString() : undefined,
        };
      };

      const result = await asyncPagination.fetchAll(fetchFn, { pageSize: 2 });
      expect(result).toEqual(allItems);
    });

    it('should enforce safety limit', async () => {
      const consoleWarn = vi.spyOn(console, 'warn').mockImplementation(() => {});

      // Create a fetch function that always returns more
      const fetchFn = async (request: PaginationRequest): Promise<PaginatedResponse<number>> => {
        const pageSize = request.pageSize ?? DEFAULT_PAGE_SIZE;
        const offset = request.offset ?? 0;
        return {
          items: Array.from({ length: pageSize }, (_, i) => offset + i),
          hasMore: true,
          nextCursor: (offset + pageSize).toString(),
        };
      };

      const result = await asyncPagination.fetchAll(fetchFn, { pageSize: 500 });
      expect(result.length).toBeGreaterThanOrEqual(10000);
      expect(consoleWarn).toHaveBeenCalledWith('Pagination safety limit reached (10000 items)');

      consoleWarn.mockRestore();
    });
  });

  describe('iterate', () => {
    it('should yield individual items across pages', async () => {
      const allItems = Array.from({ length: 15 }, (_, i) => `item-${i}`);

      const fetchFn = async (request: PaginationRequest): Promise<PaginatedResponse<string>> => {
        const offset = request.offset ?? 0;
        const pageSize = request.pageSize ?? 5;
        const end = Math.min(offset + pageSize, allItems.length);
        return {
          items: allItems.slice(offset, end),
          hasMore: end < allItems.length,
        };
      };

      const results: string[] = [];
      for await (const item of asyncPagination.iterate(fetchFn, { pageSize: 5 })) {
        results.push(item);
      }

      expect(results).toEqual(allItems);
    });

    it('should handle empty results', async () => {
      const fetchFn = async (): Promise<PaginatedResponse<string>> => ({
        items: [],
        hasMore: false,
      });

      const results: string[] = [];
      for await (const item of asyncPagination.iterate(fetchFn)) {
        results.push(item);
      }

      expect(results).toEqual([]);
    });

    it('should work with cursor-based pagination', async () => {
      const pages = [
        { items: [1, 2], nextCursor: 'page2' },
        { items: [3, 4], nextCursor: 'page3' },
        { items: [5], nextCursor: undefined },
      ];
      let pageIndex = 0;

      const fetchFn = async (request: PaginationRequest): Promise<PaginatedResponse<number>> => {
        const page = pages[pageIndex];
        const hasMore = pageIndex < pages.length - 1;
        if (hasMore) pageIndex++;
        return {
          items: page.items,
          hasMore,
          nextCursor: page.nextCursor,
        };
      };

      const results: number[] = [];
      for await (const item of asyncPagination.iterate(fetchFn)) {
        results.push(item);
      }

      expect(results).toEqual([1, 2, 3, 4, 5]);
    });
  });
});
