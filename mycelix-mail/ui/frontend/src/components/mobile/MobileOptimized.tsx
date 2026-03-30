// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mobile-Optimized Components
 *
 * Touch-friendly, responsive components for mobile devices:
 * - Swipe gestures
 * - Bottom sheets
 * - Pull-to-refresh
 * - Touch-optimized controls
 */

import {
  useState,
  useRef,
  useEffect,
  useCallback,
  type ReactNode,
  type TouchEvent as ReactTouchEvent,
} from 'react';

// ============================================
// Types
// ============================================

export interface SwipeAction {
  id: string;
  label: string;
  icon: ReactNode;
  color: string;
  onAction: () => void;
}

export interface BottomSheetProps {
  isOpen: boolean;
  onClose: () => void;
  children: ReactNode;
  title?: string;
  snapPoints?: number[];
}

export interface PullToRefreshProps {
  onRefresh: () => Promise<void>;
  children: ReactNode;
  threshold?: number;
}

// ============================================
// Hooks
// ============================================

export function useSwipeGesture(options: {
  onSwipeLeft?: () => void;
  onSwipeRight?: () => void;
  onSwipeUp?: () => void;
  onSwipeDown?: () => void;
  threshold?: number;
}) {
  const { threshold = 50 } = options;
  const touchStart = useRef<{ x: number; y: number } | null>(null);
  const touchEnd = useRef<{ x: number; y: number } | null>(null);

  const onTouchStart = useCallback((e: ReactTouchEvent) => {
    touchEnd.current = null;
    touchStart.current = {
      x: e.targetTouches[0].clientX,
      y: e.targetTouches[0].clientY,
    };
  }, []);

  const onTouchMove = useCallback((e: ReactTouchEvent) => {
    touchEnd.current = {
      x: e.targetTouches[0].clientX,
      y: e.targetTouches[0].clientY,
    };
  }, []);

  const onTouchEnd = useCallback(() => {
    if (!touchStart.current || !touchEnd.current) return;

    const deltaX = touchStart.current.x - touchEnd.current.x;
    const deltaY = touchStart.current.y - touchEnd.current.y;
    const absDeltaX = Math.abs(deltaX);
    const absDeltaY = Math.abs(deltaY);

    // Determine if horizontal or vertical swipe
    if (absDeltaX > absDeltaY) {
      if (absDeltaX > threshold) {
        if (deltaX > 0) {
          options.onSwipeLeft?.();
        } else {
          options.onSwipeRight?.();
        }
      }
    } else {
      if (absDeltaY > threshold) {
        if (deltaY > 0) {
          options.onSwipeUp?.();
        } else {
          options.onSwipeDown?.();
        }
      }
    }

    touchStart.current = null;
    touchEnd.current = null;
  }, [options, threshold]);

  return { onTouchStart, onTouchMove, onTouchEnd };
}

export function useIsMobile() {
  const [isMobile, setIsMobile] = useState(false);

  useEffect(() => {
    const checkMobile = () => {
      setIsMobile(
        window.innerWidth < 768 ||
        /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent)
      );
    };

    checkMobile();
    window.addEventListener('resize', checkMobile);
    return () => window.removeEventListener('resize', checkMobile);
  }, []);

  return isMobile;
}

export function useHapticFeedback() {
  const vibrate = useCallback((pattern: number | number[] = 10) => {
    if ('vibrate' in navigator) {
      navigator.vibrate(pattern);
    }
  }, []);

  return {
    light: () => vibrate(10),
    medium: () => vibrate(20),
    heavy: () => vibrate([30, 10, 30]),
    success: () => vibrate([10, 50, 10]),
    error: () => vibrate([50, 50, 50]),
  };
}

// ============================================
// Swipeable List Item
// ============================================

export function SwipeableListItem({
  children,
  leftActions = [],
  rightActions = [],
  onSwipeComplete,
}: {
  children: ReactNode;
  leftActions?: SwipeAction[];
  rightActions?: SwipeAction[];
  onSwipeComplete?: (actionId: string) => void;
}) {
  const [swipeOffset, setSwipeOffset] = useState(0);
  const [isAnimating, setIsAnimating] = useState(false);
  const touchStartX = useRef(0);
  const containerRef = useRef<HTMLDivElement>(null);
  const haptic = useHapticFeedback();

  const maxSwipeLeft = rightActions.length * 80;
  const maxSwipeRight = leftActions.length * 80;

  const handleTouchStart = (e: ReactTouchEvent) => {
    if (isAnimating) return;
    touchStartX.current = e.touches[0].clientX - swipeOffset;
  };

  const handleTouchMove = (e: ReactTouchEvent) => {
    if (isAnimating) return;
    const currentX = e.touches[0].clientX;
    let newOffset = currentX - touchStartX.current;

    // Apply resistance at boundaries
    if (newOffset > maxSwipeRight) {
      newOffset = maxSwipeRight + (newOffset - maxSwipeRight) * 0.2;
    } else if (newOffset < -maxSwipeLeft) {
      newOffset = -maxSwipeLeft + (newOffset + maxSwipeLeft) * 0.2;
    }

    setSwipeOffset(newOffset);
  };

  const handleTouchEnd = () => {
    setIsAnimating(true);

    let finalOffset = 0;

    // Snap to action position or back to center
    if (swipeOffset > maxSwipeRight * 0.5 && leftActions.length > 0) {
      finalOffset = maxSwipeRight;
      haptic.medium();
    } else if (swipeOffset < -maxSwipeLeft * 0.5 && rightActions.length > 0) {
      finalOffset = -maxSwipeLeft;
      haptic.medium();
    }

    setSwipeOffset(finalOffset);
    setTimeout(() => setIsAnimating(false), 200);
  };

  const handleActionClick = (action: SwipeAction) => {
    haptic.light();
    action.onAction();
    onSwipeComplete?.(action.id);
    setIsAnimating(true);
    setSwipeOffset(0);
    setTimeout(() => setIsAnimating(false), 200);
  };

  const resetSwipe = () => {
    setIsAnimating(true);
    setSwipeOffset(0);
    setTimeout(() => setIsAnimating(false), 200);
  };

  return (
    <div className="relative overflow-hidden" ref={containerRef}>
      {/* Left Actions (revealed on swipe right) */}
      <div
        className="absolute inset-y-0 left-0 flex"
        style={{ width: maxSwipeRight }}
      >
        {leftActions.map((action) => (
          <button
            key={action.id}
            onClick={() => handleActionClick(action)}
            className={`flex-1 flex flex-col items-center justify-center ${action.color}`}
          >
            {action.icon}
            <span className="text-xs mt-1 text-white">{action.label}</span>
          </button>
        ))}
      </div>

      {/* Right Actions (revealed on swipe left) */}
      <div
        className="absolute inset-y-0 right-0 flex"
        style={{ width: maxSwipeLeft }}
      >
        {rightActions.map((action) => (
          <button
            key={action.id}
            onClick={() => handleActionClick(action)}
            className={`flex-1 flex flex-col items-center justify-center ${action.color}`}
          >
            {action.icon}
            <span className="text-xs mt-1 text-white">{action.label}</span>
          </button>
        ))}
      </div>

      {/* Main Content */}
      <div
        className={`relative bg-white dark:bg-gray-800 ${
          isAnimating ? 'transition-transform duration-200' : ''
        }`}
        style={{ transform: `translateX(${swipeOffset}px)` }}
        onTouchStart={handleTouchStart}
        onTouchMove={handleTouchMove}
        onTouchEnd={handleTouchEnd}
        onClick={swipeOffset !== 0 ? resetSwipe : undefined}
      >
        {children}
      </div>
    </div>
  );
}

// ============================================
// Bottom Sheet
// ============================================

export function BottomSheet({
  isOpen,
  onClose,
  children,
  title,
  snapPoints = [0.5, 0.9],
}: BottomSheetProps) {
  const [currentSnap, setCurrentSnap] = useState(0);
  const [dragOffset, setDragOffset] = useState(0);
  const [isDragging, setIsDragging] = useState(false);
  const sheetRef = useRef<HTMLDivElement>(null);
  const dragStartY = useRef(0);
  const haptic = useHapticFeedback();

  const currentHeight = `${snapPoints[currentSnap] * 100}vh`;

  useEffect(() => {
    if (isOpen) {
      setCurrentSnap(0);
      setDragOffset(0);
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = '';
    }

    return () => {
      document.body.style.overflow = '';
    };
  }, [isOpen]);

  const handleTouchStart = (e: ReactTouchEvent) => {
    dragStartY.current = e.touches[0].clientY;
    setIsDragging(true);
  };

  const handleTouchMove = (e: ReactTouchEvent) => {
    if (!isDragging) return;
    const deltaY = e.touches[0].clientY - dragStartY.current;
    setDragOffset(Math.max(0, deltaY));
  };

  const handleTouchEnd = () => {
    setIsDragging(false);

    if (dragOffset > 100) {
      // Check if we should snap to next point or close
      if (currentSnap > 0) {
        setCurrentSnap(currentSnap - 1);
        haptic.light();
      } else {
        onClose();
      }
    } else if (dragOffset < -100 && currentSnap < snapPoints.length - 1) {
      setCurrentSnap(currentSnap + 1);
      haptic.light();
    }

    setDragOffset(0);
  };

  if (!isOpen) return null;

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-black/50 z-40 transition-opacity"
        onClick={onClose}
      />

      {/* Sheet */}
      <div
        ref={sheetRef}
        className={`fixed bottom-0 left-0 right-0 bg-white dark:bg-gray-800 rounded-t-2xl z-50 ${
          isDragging ? '' : 'transition-all duration-300'
        }`}
        style={{
          height: currentHeight,
          transform: `translateY(${dragOffset}px)`,
        }}
      >
        {/* Handle */}
        <div
          className="flex justify-center py-3 cursor-grab active:cursor-grabbing"
          onTouchStart={handleTouchStart}
          onTouchMove={handleTouchMove}
          onTouchEnd={handleTouchEnd}
        >
          <div className="w-10 h-1 bg-gray-300 dark:bg-gray-600 rounded-full" />
        </div>

        {/* Header */}
        {title && (
          <div className="px-4 pb-3 border-b border-gray-200 dark:border-gray-700">
            <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
              {title}
            </h2>
          </div>
        )}

        {/* Content */}
        <div className="overflow-y-auto" style={{ height: `calc(100% - ${title ? '80px' : '40px'})` }}>
          {children}
        </div>
      </div>
    </>
  );
}

// ============================================
// Pull to Refresh
// ============================================

export function PullToRefresh({
  onRefresh,
  children,
  threshold = 80,
}: PullToRefreshProps) {
  const [pullDistance, setPullDistance] = useState(0);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [isPulling, setIsPulling] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);
  const touchStartY = useRef(0);
  const haptic = useHapticFeedback();

  const handleTouchStart = (e: ReactTouchEvent) => {
    if (containerRef.current?.scrollTop === 0) {
      touchStartY.current = e.touches[0].clientY;
      setIsPulling(true);
    }
  };

  const handleTouchMove = (e: ReactTouchEvent) => {
    if (!isPulling || isRefreshing) return;

    const deltaY = e.touches[0].clientY - touchStartY.current;
    if (deltaY > 0) {
      // Apply resistance
      const resistance = Math.min(deltaY * 0.5, threshold * 1.5);
      setPullDistance(resistance);

      if (resistance >= threshold && pullDistance < threshold) {
        haptic.medium();
      }
    }
  };

  const handleTouchEnd = async () => {
    setIsPulling(false);

    if (pullDistance >= threshold && !isRefreshing) {
      setIsRefreshing(true);
      setPullDistance(threshold * 0.6);

      try {
        await onRefresh();
        haptic.success();
      } catch {
        haptic.error();
      } finally {
        setIsRefreshing(false);
        setPullDistance(0);
      }
    } else {
      setPullDistance(0);
    }
  };

  return (
    <div className="relative overflow-hidden">
      {/* Pull Indicator */}
      <div
        className={`absolute top-0 left-0 right-0 flex justify-center items-center transition-transform ${
          isPulling ? '' : 'duration-200'
        }`}
        style={{
          height: Math.max(pullDistance, 0),
          transform: `translateY(${pullDistance > 0 ? 0 : -40}px)`,
        }}
      >
        {isRefreshing ? (
          <div className="w-6 h-6 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
        ) : (
          <div
            className={`transition-transform duration-200 ${
              pullDistance >= threshold ? 'rotate-180' : ''
            }`}
          >
            <ArrowDownIcon className="w-6 h-6 text-gray-400" />
          </div>
        )}
      </div>

      {/* Content */}
      <div
        ref={containerRef}
        className={`overflow-y-auto transition-transform ${isPulling ? '' : 'duration-200'}`}
        style={{ transform: `translateY(${pullDistance}px)` }}
        onTouchStart={handleTouchStart}
        onTouchMove={handleTouchMove}
        onTouchEnd={handleTouchEnd}
      >
        {children}
      </div>
    </div>
  );
}

// ============================================
// Touch-Friendly Button
// ============================================

export function TouchButton({
  children,
  onClick,
  variant = 'primary',
  size = 'md',
  disabled = false,
  className = '',
  hapticFeedback = true,
}: {
  children: ReactNode;
  onClick?: () => void;
  variant?: 'primary' | 'secondary' | 'ghost' | 'danger';
  size?: 'sm' | 'md' | 'lg';
  disabled?: boolean;
  className?: string;
  hapticFeedback?: boolean;
}) {
  const haptic = useHapticFeedback();

  const handleClick = () => {
    if (hapticFeedback) {
      haptic.light();
    }
    onClick?.();
  };

  const variantClasses = {
    primary: 'bg-blue-500 text-white active:bg-blue-600',
    secondary: 'bg-gray-100 dark:bg-gray-700 text-gray-900 dark:text-gray-100 active:bg-gray-200 dark:active:bg-gray-600',
    ghost: 'text-gray-600 dark:text-gray-300 active:bg-gray-100 dark:active:bg-gray-700',
    danger: 'bg-red-500 text-white active:bg-red-600',
  };

  const sizeClasses = {
    sm: 'px-3 py-2 text-sm min-h-[36px]',
    md: 'px-4 py-3 text-base min-h-[44px]',
    lg: 'px-6 py-4 text-lg min-h-[52px]',
  };

  return (
    <button
      onClick={handleClick}
      disabled={disabled}
      className={`
        rounded-xl font-medium transition-colors touch-manipulation select-none
        ${variantClasses[variant]}
        ${sizeClasses[size]}
        ${disabled ? 'opacity-50 cursor-not-allowed' : ''}
        ${className}
      `}
    >
      {children}
    </button>
  );
}

// ============================================
// Floating Action Button
// ============================================

export function FloatingActionButton({
  icon,
  onClick,
  label,
  position = 'bottom-right',
}: {
  icon: ReactNode;
  onClick: () => void;
  label?: string;
  position?: 'bottom-right' | 'bottom-center' | 'bottom-left';
}) {
  const haptic = useHapticFeedback();

  const handleClick = () => {
    haptic.medium();
    onClick();
  };

  const positionClasses = {
    'bottom-right': 'bottom-6 right-6',
    'bottom-center': 'bottom-6 left-1/2 -translate-x-1/2',
    'bottom-left': 'bottom-6 left-6',
  };

  return (
    <button
      onClick={handleClick}
      className={`
        fixed ${positionClasses[position]} z-30
        w-14 h-14 rounded-full bg-blue-500 text-white
        shadow-lg shadow-blue-500/30
        flex items-center justify-center
        active:scale-95 transition-transform touch-manipulation
      `}
      aria-label={label}
    >
      {icon}
    </button>
  );
}

// ============================================
// Mobile Email Row
// ============================================

export function MobileEmailRow({
  email,
  onOpen,
  onArchive,
  onDelete,
  onSnooze,
  onToggleRead,
}: {
  email: {
    id: string;
    from: string;
    subject: string;
    preview: string;
    date: string;
    isRead: boolean;
    isStarred: boolean;
    trustScore?: number;
  };
  onOpen: () => void;
  onArchive: () => void;
  onDelete: () => void;
  onSnooze: () => void;
  onToggleRead: () => void;
}) {
  const leftActions: SwipeAction[] = [
    {
      id: 'archive',
      label: 'Archive',
      icon: <ArchiveIcon className="w-6 h-6 text-white" />,
      color: 'bg-emerald-500',
      onAction: onArchive,
    },
  ];

  const rightActions: SwipeAction[] = [
    {
      id: 'snooze',
      label: 'Snooze',
      icon: <ClockIcon className="w-6 h-6 text-white" />,
      color: 'bg-amber-500',
      onAction: onSnooze,
    },
    {
      id: 'delete',
      label: 'Delete',
      icon: <TrashIcon className="w-6 h-6 text-white" />,
      color: 'bg-red-500',
      onAction: onDelete,
    },
  ];

  return (
    <SwipeableListItem leftActions={leftActions} rightActions={rightActions}>
      <div
        className={`p-4 border-b border-gray-100 dark:border-gray-700 ${
          !email.isRead ? 'bg-blue-50/50 dark:bg-blue-900/10' : ''
        }`}
        onClick={onOpen}
      >
        <div className="flex items-start space-x-3">
          {/* Avatar */}
          <div className="w-10 h-10 rounded-full bg-gray-200 dark:bg-gray-700 flex items-center justify-center flex-shrink-0">
            <span className="text-sm font-medium text-gray-600 dark:text-gray-300">
              {email.from.charAt(0).toUpperCase()}
            </span>
          </div>

          {/* Content */}
          <div className="flex-1 min-w-0">
            <div className="flex items-center justify-between mb-1">
              <span
                className={`font-medium truncate ${
                  !email.isRead
                    ? 'text-gray-900 dark:text-gray-100'
                    : 'text-gray-600 dark:text-gray-400'
                }`}
              >
                {email.from}
              </span>
              <span className="text-xs text-gray-500 dark:text-gray-400 flex-shrink-0 ml-2">
                {email.date}
              </span>
            </div>

            <div
              className={`text-sm truncate mb-1 ${
                !email.isRead
                  ? 'text-gray-800 dark:text-gray-200 font-medium'
                  : 'text-gray-600 dark:text-gray-400'
              }`}
            >
              {email.subject}
            </div>

            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-500 dark:text-gray-500 truncate">
                {email.preview}
              </span>

              {email.trustScore !== undefined && (
                <div
                  className={`ml-2 px-2 py-0.5 text-xs font-medium rounded-full flex-shrink-0 ${
                    email.trustScore >= 70
                      ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-300'
                      : email.trustScore >= 40
                      ? 'bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-300'
                      : 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-300'
                  }`}
                >
                  {email.trustScore}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </SwipeableListItem>
  );
}

// ============================================
// Mobile Navigation Bar
// ============================================

export function MobileNavBar({
  activeTab,
  onTabChange,
  unreadCount = 0,
}: {
  activeTab: 'inbox' | 'search' | 'compose' | 'trust' | 'settings';
  onTabChange: (tab: 'inbox' | 'search' | 'compose' | 'trust' | 'settings') => void;
  unreadCount?: number;
}) {
  const haptic = useHapticFeedback();

  const handleTabChange = (tab: typeof activeTab) => {
    haptic.light();
    onTabChange(tab);
  };

  const tabs = [
    { id: 'inbox', label: 'Inbox', icon: InboxIcon, badge: unreadCount },
    { id: 'search', label: 'Search', icon: SearchIcon },
    { id: 'compose', label: 'Compose', icon: PlusIcon, isAction: true },
    { id: 'trust', label: 'Trust', icon: ShieldIcon },
    { id: 'settings', label: 'Settings', icon: SettingsIcon },
  ] as const;

  return (
    <nav className="fixed bottom-0 left-0 right-0 bg-white dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700 z-40 pb-safe">
      <div className="flex items-center justify-around h-16">
        {tabs.map((tab) => {
          const Icon = tab.icon;
          const isActive = activeTab === tab.id;

          if (tab.isAction) {
            return (
              <button
                key={tab.id}
                onClick={() => handleTabChange(tab.id)}
                className="relative -mt-6 w-14 h-14 rounded-full bg-blue-500 text-white shadow-lg flex items-center justify-center active:scale-95 transition-transform"
              >
                <Icon className="w-6 h-6" />
              </button>
            );
          }

          return (
            <button
              key={tab.id}
              onClick={() => handleTabChange(tab.id)}
              className={`flex flex-col items-center justify-center w-16 h-full ${
                isActive
                  ? 'text-blue-500'
                  : 'text-gray-500 dark:text-gray-400'
              }`}
            >
              <div className="relative">
                <Icon className="w-6 h-6" />
                {tab.badge && tab.badge > 0 && (
                  <span className="absolute -top-1 -right-1 w-4 h-4 bg-red-500 text-white text-[10px] font-bold rounded-full flex items-center justify-center">
                    {tab.badge > 99 ? '99+' : tab.badge}
                  </span>
                )}
              </div>
              <span className="text-[10px] mt-1">{tab.label}</span>
            </button>
          );
        })}
      </div>
    </nav>
  );
}

// ============================================
// Icons
// ============================================

function ArrowDownIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
    </svg>
  );
}

function ArchiveIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 8h14M5 8a2 2 0 110-4h14a2 2 0 110 4M5 8v10a2 2 0 002 2h10a2 2 0 002-2V8m-9 4h4" />
    </svg>
  );
}

function ClockIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
    </svg>
  );
}

function TrashIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
    </svg>
  );
}

function InboxIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 13V6a2 2 0 00-2-2H6a2 2 0 00-2 2v7m16 0v5a2 2 0 01-2 2H6a2 2 0 01-2-2v-5m16 0h-2.586a1 1 0 00-.707.293l-2.414 2.414a1 1 0 01-.707.293h-3.172a1 1 0 01-.707-.293l-2.414-2.414A1 1 0 006.586 13H4" />
    </svg>
  );
}

function SearchIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
    </svg>
  );
}

function PlusIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
    </svg>
  );
}

function ShieldIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
    </svg>
  );
}

function SettingsIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
    </svg>
  );
}

export default {
  SwipeableListItem,
  BottomSheet,
  PullToRefresh,
  TouchButton,
  FloatingActionButton,
  MobileEmailRow,
  MobileNavBar,
  useSwipeGesture,
  useIsMobile,
  useHapticFeedback,
};
