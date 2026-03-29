// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Gesture Manager - Mobile Touch Optimizations
 *
 * Features:
 * - Swipe gestures for email actions
 * - Pull-to-refresh
 * - Long press for context menus
 * - Pinch-to-zoom for attachments
 * - Haptic feedback integration
 * - Touch velocity tracking
 */

export interface GestureConfig {
  swipeThreshold?: number;
  swipeVelocityThreshold?: number;
  longPressDelay?: number;
  doubleTapDelay?: number;
  pinchThreshold?: number;
  hapticFeedback?: boolean;
}

export interface TouchPoint {
  x: number;
  y: number;
  timestamp: number;
}

export interface SwipeEvent {
  direction: 'left' | 'right' | 'up' | 'down';
  distance: number;
  velocity: number;
  startPoint: TouchPoint;
  endPoint: TouchPoint;
}

export interface PinchEvent {
  scale: number;
  center: { x: number; y: number };
  startDistance: number;
  currentDistance: number;
}

export interface GestureHandlers {
  onSwipeLeft?: (event: SwipeEvent) => void;
  onSwipeRight?: (event: SwipeEvent) => void;
  onSwipeUp?: (event: SwipeEvent) => void;
  onSwipeDown?: (event: SwipeEvent) => void;
  onLongPress?: (point: TouchPoint) => void;
  onDoubleTap?: (point: TouchPoint) => void;
  onPinch?: (event: PinchEvent) => void;
  onPinchEnd?: (event: PinchEvent) => void;
}

const DEFAULT_CONFIG: Required<GestureConfig> = {
  swipeThreshold: 50,
  swipeVelocityThreshold: 0.3,
  longPressDelay: 500,
  doubleTapDelay: 300,
  pinchThreshold: 0.1,
  hapticFeedback: true,
};

export class GestureManager {
  private element: HTMLElement;
  private config: Required<GestureConfig>;
  private handlers: GestureHandlers;
  private touchStartPoint: TouchPoint | null = null;
  private touchStartPoints: TouchPoint[] = [];
  private longPressTimer: ReturnType<typeof setTimeout> | null = null;
  private lastTapTime: number = 0;
  private lastTapPoint: TouchPoint | null = null;
  private initialPinchDistance: number = 0;
  private isPinching: boolean = false;

  constructor(
    element: HTMLElement,
    handlers: GestureHandlers,
    config: GestureConfig = {}
  ) {
    this.element = element;
    this.handlers = handlers;
    this.config = { ...DEFAULT_CONFIG, ...config };

    this.bindEvents();
  }

  private bindEvents(): void {
    this.element.addEventListener('touchstart', this.handleTouchStart, { passive: false });
    this.element.addEventListener('touchmove', this.handleTouchMove, { passive: false });
    this.element.addEventListener('touchend', this.handleTouchEnd, { passive: false });
    this.element.addEventListener('touchcancel', this.handleTouchCancel, { passive: false });
  }

  destroy(): void {
    this.element.removeEventListener('touchstart', this.handleTouchStart);
    this.element.removeEventListener('touchmove', this.handleTouchMove);
    this.element.removeEventListener('touchend', this.handleTouchEnd);
    this.element.removeEventListener('touchcancel', this.handleTouchCancel);
    this.clearLongPressTimer();
  }

  private handleTouchStart = (e: TouchEvent): void => {
    const touches = e.touches;

    if (touches.length === 1) {
      // Single touch - track for swipe, long press, double tap
      this.touchStartPoint = {
        x: touches[0].clientX,
        y: touches[0].clientY,
        timestamp: Date.now(),
      };

      // Start long press timer
      this.startLongPressTimer(this.touchStartPoint);

      // Check for double tap
      if (this.lastTapPoint && this.lastTapTime) {
        const timeDiff = Date.now() - this.lastTapTime;
        const distance = this.getDistance(
          { x: touches[0].clientX, y: touches[0].clientY },
          { x: this.lastTapPoint.x, y: this.lastTapPoint.y }
        );

        if (timeDiff < this.config.doubleTapDelay && distance < 30) {
          this.handlers.onDoubleTap?.(this.touchStartPoint);
          this.triggerHaptic('light');
          this.lastTapTime = 0;
          this.lastTapPoint = null;
        }
      }
    } else if (touches.length === 2) {
      // Two finger touch - pinch gesture
      this.isPinching = true;
      this.clearLongPressTimer();

      this.touchStartPoints = [
        { x: touches[0].clientX, y: touches[0].clientY, timestamp: Date.now() },
        { x: touches[1].clientX, y: touches[1].clientY, timestamp: Date.now() },
      ];

      this.initialPinchDistance = this.getDistance(
        this.touchStartPoints[0],
        this.touchStartPoints[1]
      );
    }
  };

  private handleTouchMove = (e: TouchEvent): void => {
    const touches = e.touches;

    // Cancel long press on movement
    if (this.touchStartPoint && touches.length === 1) {
      const dx = touches[0].clientX - this.touchStartPoint.x;
      const dy = touches[0].clientY - this.touchStartPoint.y;
      const distance = Math.sqrt(dx * dx + dy * dy);

      if (distance > 10) {
        this.clearLongPressTimer();
      }
    }

    // Handle pinch
    if (this.isPinching && touches.length === 2) {
      const currentDistance = this.getDistance(
        { x: touches[0].clientX, y: touches[0].clientY },
        { x: touches[1].clientX, y: touches[1].clientY }
      );

      const scale = currentDistance / this.initialPinchDistance;
      const center = {
        x: (touches[0].clientX + touches[1].clientX) / 2,
        y: (touches[0].clientY + touches[1].clientY) / 2,
      };

      if (Math.abs(scale - 1) > this.config.pinchThreshold) {
        this.handlers.onPinch?.({
          scale,
          center,
          startDistance: this.initialPinchDistance,
          currentDistance,
        });
      }
    }
  };

  private handleTouchEnd = (e: TouchEvent): void => {
    this.clearLongPressTimer();

    // Handle pinch end
    if (this.isPinching) {
      const changedTouches = e.changedTouches;
      if (changedTouches.length > 0) {
        const currentDistance = this.getDistance(
          { x: changedTouches[0].clientX, y: changedTouches[0].clientY },
          this.touchStartPoints[1] || { x: changedTouches[0].clientX, y: changedTouches[0].clientY }
        );

        this.handlers.onPinchEnd?.({
          scale: currentDistance / this.initialPinchDistance,
          center: { x: 0, y: 0 },
          startDistance: this.initialPinchDistance,
          currentDistance,
        });
      }
      this.isPinching = false;
      this.touchStartPoints = [];
      return;
    }

    // Handle swipe
    if (this.touchStartPoint && e.changedTouches.length === 1) {
      const touch = e.changedTouches[0];
      const endPoint: TouchPoint = {
        x: touch.clientX,
        y: touch.clientY,
        timestamp: Date.now(),
      };

      const swipeEvent = this.detectSwipe(this.touchStartPoint, endPoint);

      if (swipeEvent) {
        this.triggerHaptic('medium');

        switch (swipeEvent.direction) {
          case 'left':
            this.handlers.onSwipeLeft?.(swipeEvent);
            break;
          case 'right':
            this.handlers.onSwipeRight?.(swipeEvent);
            break;
          case 'up':
            this.handlers.onSwipeUp?.(swipeEvent);
            break;
          case 'down':
            this.handlers.onSwipeDown?.(swipeEvent);
            break;
        }
      } else {
        // Record tap for double tap detection
        this.lastTapTime = Date.now();
        this.lastTapPoint = endPoint;
      }

      this.touchStartPoint = null;
    }
  };

  private handleTouchCancel = (): void => {
    this.clearLongPressTimer();
    this.touchStartPoint = null;
    this.touchStartPoints = [];
    this.isPinching = false;
  };

  private detectSwipe(start: TouchPoint, end: TouchPoint): SwipeEvent | null {
    const dx = end.x - start.x;
    const dy = end.y - start.y;
    const distance = Math.sqrt(dx * dx + dy * dy);
    const duration = end.timestamp - start.timestamp;
    const velocity = distance / duration;

    if (
      distance < this.config.swipeThreshold ||
      velocity < this.config.swipeVelocityThreshold
    ) {
      return null;
    }

    const absDx = Math.abs(dx);
    const absDy = Math.abs(dy);

    let direction: 'left' | 'right' | 'up' | 'down';

    if (absDx > absDy) {
      direction = dx > 0 ? 'right' : 'left';
    } else {
      direction = dy > 0 ? 'down' : 'up';
    }

    return {
      direction,
      distance,
      velocity,
      startPoint: start,
      endPoint: end,
    };
  }

  private startLongPressTimer(point: TouchPoint): void {
    this.longPressTimer = setTimeout(() => {
      this.triggerHaptic('heavy');
      this.handlers.onLongPress?.(point);
    }, this.config.longPressDelay);
  }

  private clearLongPressTimer(): void {
    if (this.longPressTimer) {
      clearTimeout(this.longPressTimer);
      this.longPressTimer = null;
    }
  }

  private getDistance(
    point1: { x: number; y: number },
    point2: { x: number; y: number }
  ): number {
    const dx = point2.x - point1.x;
    const dy = point2.y - point1.y;
    return Math.sqrt(dx * dx + dy * dy);
  }

  private triggerHaptic(intensity: 'light' | 'medium' | 'heavy'): void {
    if (!this.config.hapticFeedback) return;

    if ('vibrate' in navigator) {
      switch (intensity) {
        case 'light':
          navigator.vibrate(10);
          break;
        case 'medium':
          navigator.vibrate(25);
          break;
        case 'heavy':
          navigator.vibrate(50);
          break;
      }
    }
  }
}

/**
 * Pull to Refresh Manager
 */
export interface PullToRefreshConfig {
  threshold?: number;
  maxPull?: number;
  resistance?: number;
}

export class PullToRefreshManager {
  private element: HTMLElement;
  private indicator: HTMLElement | null = null;
  private config: Required<PullToRefreshConfig>;
  private onRefresh: () => Promise<void>;
  private startY: number = 0;
  private currentY: number = 0;
  private isPulling: boolean = false;
  private isRefreshing: boolean = false;

  constructor(
    element: HTMLElement,
    onRefresh: () => Promise<void>,
    config: PullToRefreshConfig = {}
  ) {
    this.element = element;
    this.onRefresh = onRefresh;
    this.config = {
      threshold: config.threshold || 80,
      maxPull: config.maxPull || 150,
      resistance: config.resistance || 2.5,
    };

    this.createIndicator();
    this.bindEvents();
  }

  private createIndicator(): void {
    this.indicator = document.createElement('div');
    this.indicator.className = 'pull-to-refresh-indicator';
    this.indicator.innerHTML = `
      <div class="ptr-spinner">
        <svg viewBox="0 0 24 24" width="24" height="24">
          <path d="M12 4V1L8 5l4 4V6c3.31 0 6 2.69 6 6 0 1.01-.25 1.97-.7 2.8l1.46 1.46C19.54 15.03 20 13.57 20 12c0-4.42-3.58-8-8-8zm0 14c-3.31 0-6-2.69-6-6 0-1.01.25-1.97.7-2.8L5.24 7.74C4.46 8.97 4 10.43 4 12c0 4.42 3.58 8 8 8v3l4-4-4-4v3z" fill="currentColor"/>
        </svg>
      </div>
      <span class="ptr-text">Pull to refresh</span>
    `;
    this.indicator.style.cssText = `
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      padding: 16px;
      transform: translateY(-100%);
      transition: transform 0.2s ease;
      color: #64748b;
      font-size: 14px;
      z-index: 100;
    `;

    this.element.style.position = 'relative';
    this.element.insertBefore(this.indicator, this.element.firstChild);
  }

  private bindEvents(): void {
    this.element.addEventListener('touchstart', this.handleTouchStart, { passive: true });
    this.element.addEventListener('touchmove', this.handleTouchMove, { passive: false });
    this.element.addEventListener('touchend', this.handleTouchEnd, { passive: true });
  }

  destroy(): void {
    this.element.removeEventListener('touchstart', this.handleTouchStart);
    this.element.removeEventListener('touchmove', this.handleTouchMove);
    this.element.removeEventListener('touchend', this.handleTouchEnd);
    this.indicator?.remove();
  }

  private handleTouchStart = (e: TouchEvent): void => {
    if (this.isRefreshing || this.element.scrollTop > 0) return;
    this.startY = e.touches[0].clientY;
    this.isPulling = true;
  };

  private handleTouchMove = (e: TouchEvent): void => {
    if (!this.isPulling || this.isRefreshing) return;

    this.currentY = e.touches[0].clientY;
    const pullDistance = (this.currentY - this.startY) / this.config.resistance;

    if (pullDistance > 0) {
      e.preventDefault();
      const clampedDistance = Math.min(pullDistance, this.config.maxPull);
      this.updateIndicator(clampedDistance);
    }
  };

  private handleTouchEnd = async (): Promise<void> => {
    if (!this.isPulling) return;

    const pullDistance = (this.currentY - this.startY) / this.config.resistance;
    this.isPulling = false;

    if (pullDistance >= this.config.threshold) {
      await this.triggerRefresh();
    } else {
      this.resetIndicator();
    }
  };

  private updateIndicator(distance: number): void {
    if (!this.indicator) return;

    const progress = Math.min(distance / this.config.threshold, 1);
    const translateY = distance - this.indicator.offsetHeight;

    this.indicator.style.transform = `translateY(${Math.max(translateY, -100)}px)`;
    this.indicator.style.opacity = `${progress}`;

    const spinner = this.indicator.querySelector('.ptr-spinner') as HTMLElement;
    if (spinner) {
      spinner.style.transform = `rotate(${progress * 360}deg)`;
    }

    const text = this.indicator.querySelector('.ptr-text') as HTMLElement;
    if (text) {
      text.textContent = progress >= 1 ? 'Release to refresh' : 'Pull to refresh';
    }
  }

  private async triggerRefresh(): Promise<void> {
    this.isRefreshing = true;

    if (this.indicator) {
      const text = this.indicator.querySelector('.ptr-text') as HTMLElement;
      if (text) text.textContent = 'Refreshing...';

      const spinner = this.indicator.querySelector('.ptr-spinner') as HTMLElement;
      if (spinner) {
        spinner.style.animation = 'spin 1s linear infinite';
      }
    }

    try {
      await this.onRefresh();
    } finally {
      this.isRefreshing = false;
      this.resetIndicator();
    }
  }

  private resetIndicator(): void {
    if (!this.indicator) return;

    this.indicator.style.transform = 'translateY(-100%)';
    this.indicator.style.opacity = '0';

    const spinner = this.indicator.querySelector('.ptr-spinner') as HTMLElement;
    if (spinner) {
      spinner.style.animation = '';
      spinner.style.transform = '';
    }
  }
}

/**
 * Swipeable List Item for email actions
 */
export interface SwipeAction {
  id: string;
  label: string;
  icon?: string;
  color: string;
  backgroundColor: string;
}

export interface SwipeableItemConfig {
  leftActions?: SwipeAction[];
  rightActions?: SwipeAction[];
  actionThreshold?: number;
  maxSwipeDistance?: number;
}

export class SwipeableItem {
  private element: HTMLElement;
  private content: HTMLElement;
  private config: Required<SwipeableItemConfig>;
  private onAction: (actionId: string) => void;
  private startX: number = 0;
  private currentX: number = 0;
  private leftActionsEl: HTMLElement | null = null;
  private rightActionsEl: HTMLElement | null = null;

  constructor(
    element: HTMLElement,
    onAction: (actionId: string) => void,
    config: SwipeableItemConfig = {}
  ) {
    this.element = element;
    this.onAction = onAction;
    this.config = {
      leftActions: config.leftActions || [],
      rightActions: config.rightActions || [],
      actionThreshold: config.actionThreshold || 80,
      maxSwipeDistance: config.maxSwipeDistance || 150,
    };

    this.content = element.querySelector('.swipeable-content') || element;
    this.setupStructure();
    this.bindEvents();
  }

  private setupStructure(): void {
    this.element.style.position = 'relative';
    this.element.style.overflow = 'hidden';

    if (this.config.leftActions.length > 0) {
      this.leftActionsEl = this.createActionsContainer(this.config.leftActions, 'left');
    }

    if (this.config.rightActions.length > 0) {
      this.rightActionsEl = this.createActionsContainer(this.config.rightActions, 'right');
    }
  }

  private createActionsContainer(actions: SwipeAction[], side: 'left' | 'right'): HTMLElement {
    const container = document.createElement('div');
    container.className = `swipe-actions swipe-actions-${side}`;
    container.style.cssText = `
      position: absolute;
      top: 0;
      bottom: 0;
      ${side}: 0;
      display: flex;
      align-items: center;
      opacity: 0;
      transition: opacity 0.2s ease;
    `;

    actions.forEach((action) => {
      const actionEl = document.createElement('button');
      actionEl.className = 'swipe-action';
      actionEl.dataset.actionId = action.id;
      actionEl.innerHTML = `
        ${action.icon ? `<span class="action-icon">${action.icon}</span>` : ''}
        <span class="action-label">${action.label}</span>
      `;
      actionEl.style.cssText = `
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 12px 16px;
        border: none;
        background: ${action.backgroundColor};
        color: ${action.color};
        font-size: 12px;
        cursor: pointer;
        height: 100%;
      `;

      actionEl.addEventListener('click', () => {
        this.onAction(action.id);
        this.reset();
      });

      container.appendChild(actionEl);
    });

    this.element.insertBefore(container, this.element.firstChild);
    return container;
  }

  private bindEvents(): void {
    this.content.addEventListener('touchstart', this.handleTouchStart, { passive: true });
    this.content.addEventListener('touchmove', this.handleTouchMove, { passive: false });
    this.content.addEventListener('touchend', this.handleTouchEnd, { passive: true });
  }

  destroy(): void {
    this.content.removeEventListener('touchstart', this.handleTouchStart);
    this.content.removeEventListener('touchmove', this.handleTouchMove);
    this.content.removeEventListener('touchend', this.handleTouchEnd);
    this.leftActionsEl?.remove();
    this.rightActionsEl?.remove();
  }

  private handleTouchStart = (e: TouchEvent): void => {
    this.startX = e.touches[0].clientX;
  };

  private handleTouchMove = (e: TouchEvent): void => {
    this.currentX = e.touches[0].clientX;
    const deltaX = this.currentX - this.startX;

    // Clamp the swipe distance
    const clampedDelta = Math.max(
      -this.config.maxSwipeDistance,
      Math.min(this.config.maxSwipeDistance, deltaX)
    );

    this.content.style.transform = `translateX(${clampedDelta}px)`;

    // Show/hide action containers
    if (deltaX > 0 && this.leftActionsEl) {
      this.leftActionsEl.style.opacity = `${Math.min(deltaX / this.config.actionThreshold, 1)}`;
    } else if (deltaX < 0 && this.rightActionsEl) {
      this.rightActionsEl.style.opacity = `${Math.min(-deltaX / this.config.actionThreshold, 1)}`;
    }

    if (Math.abs(deltaX) > 10) {
      e.preventDefault();
    }
  };

  private handleTouchEnd = (): void => {
    const deltaX = this.currentX - this.startX;

    // Check if action threshold was met
    if (Math.abs(deltaX) >= this.config.actionThreshold) {
      const actions = deltaX > 0 ? this.config.leftActions : this.config.rightActions;
      if (actions.length > 0) {
        // Trigger first action
        this.onAction(actions[0].id);
      }
    }

    this.reset();
  };

  private reset(): void {
    this.content.style.transition = 'transform 0.2s ease';
    this.content.style.transform = 'translateX(0)';

    if (this.leftActionsEl) {
      this.leftActionsEl.style.opacity = '0';
    }
    if (this.rightActionsEl) {
      this.rightActionsEl.style.opacity = '0';
    }

    setTimeout(() => {
      this.content.style.transition = '';
    }, 200);
  }
}

export default GestureManager;
