// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mobile Optimizations Module
 *
 * Touch gestures, swipe actions, and mobile-optimized components
 */

// Core gesture management
export {
  GestureManager,
  PullToRefreshManager,
  SwipeableItem,
  type GestureConfig,
  type GestureHandlers,
  type TouchPoint,
  type SwipeEvent,
  type PinchEvent,
  type PullToRefreshConfig,
  type SwipeAction,
  type SwipeableItemConfig,
} from './GestureManager';

// React hooks for gestures
export {
  useGestures,
  useSwipe,
  useSwipeActions,
  usePullToRefresh,
  useLongPress,
  useDoubleTap,
  usePinchZoom,
  useIsTouchDevice,
  useBreakpoints,
  useSafeAreaInsets,
  useViewportHeight,
  useKeyboardVisibility,
  type UseSwipeOptions,
  type UsePinchZoomOptions,
  type Breakpoints,
  type SafeAreaInsets,
} from './useMobileGestures';

// Mobile-optimized components
export {
  MobileEmailList,
  type Email,
  type MobileEmailListProps,
} from './MobileEmailList';

export default {
  GestureManager,
  PullToRefreshManager,
  SwipeableItem,
};
