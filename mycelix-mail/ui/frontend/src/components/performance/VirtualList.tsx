// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Virtual List Component
 *
 * High-performance virtualized list for rendering large datasets:
 * - Only renders visible items
 * - Smooth scrolling with overscan
 * - Dynamic item heights support
 * - Keyboard navigation
 * - Accessible with ARIA
 */

import {
  useRef,
  useState,
  useEffect,
  useCallback,
  useMemo,
  forwardRef,
  useImperativeHandle,
  CSSProperties,
  ReactNode,
} from 'react';

// ============================================
// Types
// ============================================

interface VirtualListProps<T> {
  items: T[];
  itemHeight: number | ((item: T, index: number) => number);
  renderItem: (item: T, index: number, style: CSSProperties) => ReactNode;
  overscan?: number;
  className?: string;
  onScroll?: (scrollTop: number) => void;
  onEndReached?: () => void;
  endReachedThreshold?: number;
  getItemKey?: (item: T, index: number) => string | number;
  estimatedItemHeight?: number;
  ariaLabel?: string;
}

interface VirtualListHandle {
  scrollToIndex: (index: number, align?: 'start' | 'center' | 'end') => void;
  scrollToTop: () => void;
  scrollToBottom: () => void;
  getScrollTop: () => number;
}

interface ItemMetadata {
  offset: number;
  height: number;
}

// ============================================
// Virtual List Implementation
// ============================================

function VirtualListInner<T>(
  {
    items,
    itemHeight,
    renderItem,
    overscan = 3,
    className = '',
    onScroll,
    onEndReached,
    endReachedThreshold = 200,
    getItemKey,
    estimatedItemHeight = 60,
    ariaLabel = 'Virtual list',
  }: VirtualListProps<T>,
  ref: React.ForwardedRef<VirtualListHandle>
) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [scrollTop, setScrollTop] = useState(0);
  const [containerHeight, setContainerHeight] = useState(0);

  // Calculate item metadata (offsets and heights)
  const itemMetadata = useMemo((): ItemMetadata[] => {
    const metadata: ItemMetadata[] = [];
    let offset = 0;

    for (let i = 0; i < items.length; i++) {
      const height =
        typeof itemHeight === 'function'
          ? itemHeight(items[i], i)
          : itemHeight;

      metadata.push({ offset, height });
      offset += height;
    }

    return metadata;
  }, [items, itemHeight]);

  // Total height of all items
  const totalHeight = useMemo(() => {
    if (itemMetadata.length === 0) return 0;
    const lastItem = itemMetadata[itemMetadata.length - 1];
    return lastItem.offset + lastItem.height;
  }, [itemMetadata]);

  // Find visible range using binary search
  const visibleRange = useMemo(() => {
    if (items.length === 0) {
      return { start: 0, end: 0 };
    }

    // Binary search for start index
    let start = 0;
    let end = items.length - 1;

    while (start < end) {
      const mid = Math.floor((start + end) / 2);
      if (itemMetadata[mid].offset + itemMetadata[mid].height < scrollTop) {
        start = mid + 1;
      } else {
        end = mid;
      }
    }

    const startIndex = Math.max(0, start - overscan);

    // Find end index
    const viewportEnd = scrollTop + containerHeight;
    end = start;

    while (end < items.length && itemMetadata[end].offset < viewportEnd) {
      end++;
    }

    const endIndex = Math.min(items.length, end + overscan);

    return { start: startIndex, end: endIndex };
  }, [scrollTop, containerHeight, items.length, itemMetadata, overscan]);

  // Handle scroll
  const handleScroll = useCallback(
    (e: React.UIEvent<HTMLDivElement>) => {
      const newScrollTop = e.currentTarget.scrollTop;
      setScrollTop(newScrollTop);
      onScroll?.(newScrollTop);

      // Check if end reached
      if (onEndReached) {
        const scrollBottom = newScrollTop + containerHeight;
        if (totalHeight - scrollBottom < endReachedThreshold) {
          onEndReached();
        }
      }
    },
    [containerHeight, totalHeight, endReachedThreshold, onScroll, onEndReached]
  );

  // Measure container height
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        setContainerHeight(entry.contentRect.height);
      }
    });

    resizeObserver.observe(container);
    setContainerHeight(container.clientHeight);

    return () => resizeObserver.disconnect();
  }, []);

  // Expose imperative methods
  useImperativeHandle(ref, () => ({
    scrollToIndex: (index: number, align: 'start' | 'center' | 'end' = 'start') => {
      const container = containerRef.current;
      if (!container || index < 0 || index >= items.length) return;

      const item = itemMetadata[index];
      let targetScroll: number;

      switch (align) {
        case 'start':
          targetScroll = item.offset;
          break;
        case 'center':
          targetScroll = item.offset - containerHeight / 2 + item.height / 2;
          break;
        case 'end':
          targetScroll = item.offset - containerHeight + item.height;
          break;
      }

      container.scrollTo({
        top: Math.max(0, targetScroll),
        behavior: 'smooth',
      });
    },
    scrollToTop: () => {
      containerRef.current?.scrollTo({ top: 0, behavior: 'smooth' });
    },
    scrollToBottom: () => {
      containerRef.current?.scrollTo({ top: totalHeight, behavior: 'smooth' });
    },
    getScrollTop: () => scrollTop,
  }));

  // Render visible items
  const visibleItems = useMemo(() => {
    const result: ReactNode[] = [];

    for (let i = visibleRange.start; i < visibleRange.end; i++) {
      const item = items[i];
      const metadata = itemMetadata[i];
      const key = getItemKey ? getItemKey(item, i) : i;

      const style: CSSProperties = {
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        height: metadata.height,
        transform: `translateY(${metadata.offset}px)`,
      };

      result.push(
        <div key={key} style={style} role="listitem" aria-posinset={i + 1} aria-setsize={items.length}>
          {renderItem(item, i, style)}
        </div>
      );
    }

    return result;
  }, [visibleRange, items, itemMetadata, getItemKey, renderItem]);

  return (
    <div
      ref={containerRef}
      className={`relative overflow-auto ${className}`}
      onScroll={handleScroll}
      role="list"
      aria-label={ariaLabel}
      tabIndex={0}
    >
      <div
        style={{
          height: totalHeight,
          position: 'relative',
        }}
      >
        {visibleItems}
      </div>
    </div>
  );
}

export const VirtualList = forwardRef(VirtualListInner) as <T>(
  props: VirtualListProps<T> & { ref?: React.ForwardedRef<VirtualListHandle> }
) => ReturnType<typeof VirtualListInner>;

// ============================================
// Virtualized Email List
// ============================================

interface Email {
  id: string;
  subject: string;
  from: { name: string; address: string };
  date: string;
  preview?: string;
  isRead: boolean;
  isStarred: boolean;
  trustScore?: number;
  epistemicTier?: number;
}

interface VirtualEmailListProps {
  emails: Email[];
  selectedId?: string;
  onSelect: (email: Email) => void;
  onLoadMore?: () => void;
  isLoading?: boolean;
  className?: string;
}

export function VirtualEmailList({
  emails,
  selectedId,
  onSelect,
  onLoadMore,
  isLoading,
  className = '',
}: VirtualEmailListProps) {
  const listRef = useRef<VirtualListHandle>(null);

  const renderEmail = useCallback(
    (email: Email, index: number) => {
      const isSelected = email.id === selectedId;

      return (
        <div
          className={`
            flex items-center gap-4 px-4 py-3 cursor-pointer transition-colors
            border-b border-gray-100 dark:border-gray-800
            ${isSelected ? 'bg-blue-50 dark:bg-blue-900/20' : 'hover:bg-gray-50 dark:hover:bg-gray-800/50'}
            ${!email.isRead ? 'font-semibold' : ''}
          `}
          onClick={() => onSelect(email)}
          data-testid="email-row"
        >
          {/* Avatar */}
          <div className="w-10 h-10 rounded-full bg-gray-200 dark:bg-gray-700 flex items-center justify-center text-gray-600 dark:text-gray-300 flex-shrink-0">
            {email.from.name.charAt(0).toUpperCase()}
          </div>

          {/* Content */}
          <div className="flex-1 min-w-0">
            <div className="flex items-center justify-between mb-1">
              <span className="text-sm text-gray-900 dark:text-gray-100 truncate">
                {email.from.name}
              </span>
              <span className="text-xs text-gray-500 flex-shrink-0 ml-2">
                {new Date(email.date).toLocaleDateString()}
              </span>
            </div>
            <div className="text-sm text-gray-900 dark:text-gray-100 truncate">
              {email.subject}
            </div>
            {email.preview && (
              <div className="text-xs text-gray-500 truncate mt-0.5">
                {email.preview}
              </div>
            )}
          </div>

          {/* Trust indicators */}
          <div className="flex items-center gap-2 flex-shrink-0">
            {email.trustScore !== undefined && (
              <div
                className={`
                  px-2 py-0.5 text-xs rounded-full
                  ${email.trustScore >= 0.7
                    ? 'bg-emerald-100 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-300'
                    : email.trustScore >= 0.4
                    ? 'bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-300'
                    : 'bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300'
                  }
                `}
                data-testid="trust-badge"
              >
                {Math.round(email.trustScore * 100)}%
              </div>
            )}
            {email.epistemicTier !== undefined && (
              <div className="w-5 h-5 rounded-full bg-gray-200 dark:bg-gray-700 flex items-center justify-center text-xs">
                T{email.epistemicTier}
              </div>
            )}
            {email.isStarred && (
              <span className="text-amber-500">★</span>
            )}
          </div>
        </div>
      );
    },
    [selectedId, onSelect]
  );

  return (
    <div className={`h-full ${className}`} data-testid="email-list">
      <VirtualList
        ref={listRef}
        items={emails}
        itemHeight={76}
        renderItem={renderEmail}
        getItemKey={(email) => email.id}
        onEndReached={onLoadMore}
        endReachedThreshold={300}
        overscan={5}
        ariaLabel="Email list"
        className="h-full"
      />
      {isLoading && (
        <div className="absolute bottom-0 left-0 right-0 p-4 bg-gradient-to-t from-white dark:from-gray-900">
          <div className="flex items-center justify-center gap-2 text-sm text-gray-500">
            <div className="w-4 h-4 border-2 border-gray-300 border-t-blue-500 rounded-full animate-spin" />
            Loading more...
          </div>
        </div>
      )}
    </div>
  );
}

// ============================================
// Hook for windowed data
// ============================================

interface UseWindowedDataOptions<T> {
  items: T[];
  pageSize?: number;
}

export function useWindowedData<T>({ items, pageSize = 50 }: UseWindowedDataOptions<T>) {
  const [visibleCount, setVisibleCount] = useState(pageSize);

  const visibleItems = useMemo(
    () => items.slice(0, visibleCount),
    [items, visibleCount]
  );

  const loadMore = useCallback(() => {
    setVisibleCount((prev) => Math.min(prev + pageSize, items.length));
  }, [items.length, pageSize]);

  const hasMore = visibleCount < items.length;

  const reset = useCallback(() => {
    setVisibleCount(pageSize);
  }, [pageSize]);

  return {
    visibleItems,
    loadMore,
    hasMore,
    reset,
    totalCount: items.length,
    loadedCount: visibleCount,
  };
}

export default VirtualList;
