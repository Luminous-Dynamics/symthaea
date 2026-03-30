// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Animation Components & Utilities
 *
 * Smooth, accessible animations for epistemic UI:
 * - Fade transitions
 * - Slide animations
 * - Scale effects
 * - Staggered lists
 * - Respects reduced motion preferences
 */

import { ReactNode, useEffect, useState, useRef, CSSProperties } from 'react';

// ============================================
// Motion Preference Hook
// ============================================

export function usePrefersReducedMotion(): boolean {
  const [prefersReducedMotion, setPrefersReducedMotion] = useState(false);

  useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
    setPrefersReducedMotion(mediaQuery.matches);

    const handler = (event: MediaQueryListEvent) => {
      setPrefersReducedMotion(event.matches);
    };

    mediaQuery.addEventListener('change', handler);
    return () => mediaQuery.removeEventListener('change', handler);
  }, []);

  return prefersReducedMotion;
}

// ============================================
// Fade Animation
// ============================================

interface FadeProps {
  children: ReactNode;
  show: boolean;
  duration?: number;
  delay?: number;
  className?: string;
  onExited?: () => void;
}

export function Fade({
  children,
  show,
  duration = 200,
  delay = 0,
  className = '',
  onExited,
}: FadeProps) {
  const [shouldRender, setShouldRender] = useState(show);
  const prefersReducedMotion = usePrefersReducedMotion();

  useEffect(() => {
    if (show) {
      setShouldRender(true);
    }
  }, [show]);

  const handleAnimationEnd = () => {
    if (!show) {
      setShouldRender(false);
      onExited?.();
    }
  };

  if (!shouldRender) return null;

  const effectiveDuration = prefersReducedMotion ? 0 : duration;

  return (
    <div
      className={className}
      style={{
        animation: `${show ? 'fadeIn' : 'fadeOut'} ${effectiveDuration}ms ease-out`,
        animationDelay: `${delay}ms`,
        animationFillMode: 'both',
      }}
      onAnimationEnd={handleAnimationEnd}
    >
      {children}
    </div>
  );
}

// ============================================
// Slide Animation
// ============================================

type SlideDirection = 'up' | 'down' | 'left' | 'right';

interface SlideProps {
  children: ReactNode;
  show: boolean;
  direction?: SlideDirection;
  duration?: number;
  distance?: number;
  className?: string;
}

export function Slide({
  children,
  show,
  direction = 'up',
  duration = 300,
  distance = 20,
  className = '',
}: SlideProps) {
  const [shouldRender, setShouldRender] = useState(show);
  const prefersReducedMotion = usePrefersReducedMotion();

  useEffect(() => {
    if (show) setShouldRender(true);
  }, [show]);

  if (!shouldRender) return null;

  const transforms: Record<SlideDirection, string> = {
    up: `translateY(${distance}px)`,
    down: `translateY(-${distance}px)`,
    left: `translateX(${distance}px)`,
    right: `translateX(-${distance}px)`,
  };

  const effectiveDuration = prefersReducedMotion ? 0 : duration;

  return (
    <div
      className={className}
      style={{
        opacity: show ? 1 : 0,
        transform: show ? 'translate(0)' : transforms[direction],
        transition: `opacity ${effectiveDuration}ms ease-out, transform ${effectiveDuration}ms ease-out`,
      }}
      onTransitionEnd={() => {
        if (!show) setShouldRender(false);
      }}
    >
      {children}
    </div>
  );
}

// ============================================
// Scale Animation
// ============================================

interface ScaleProps {
  children: ReactNode;
  show: boolean;
  duration?: number;
  origin?: string;
  className?: string;
}

export function Scale({
  children,
  show,
  duration = 200,
  origin = 'center',
  className = '',
}: ScaleProps) {
  const [shouldRender, setShouldRender] = useState(show);
  const prefersReducedMotion = usePrefersReducedMotion();

  useEffect(() => {
    if (show) setShouldRender(true);
  }, [show]);

  if (!shouldRender) return null;

  const effectiveDuration = prefersReducedMotion ? 0 : duration;

  return (
    <div
      className={className}
      style={{
        opacity: show ? 1 : 0,
        transform: show ? 'scale(1)' : 'scale(0.95)',
        transformOrigin: origin,
        transition: `opacity ${effectiveDuration}ms ease-out, transform ${effectiveDuration}ms ease-out`,
      }}
      onTransitionEnd={() => {
        if (!show) setShouldRender(false);
      }}
    >
      {children}
    </div>
  );
}

// ============================================
// Collapse Animation
// ============================================

interface CollapseProps {
  children: ReactNode;
  show: boolean;
  duration?: number;
  className?: string;
}

export function Collapse({
  children,
  show,
  duration = 300,
  className = '',
}: CollapseProps) {
  const contentRef = useRef<HTMLDivElement>(null);
  const [height, setHeight] = useState<number | 'auto'>(show ? 'auto' : 0);
  const prefersReducedMotion = usePrefersReducedMotion();

  useEffect(() => {
    if (contentRef.current) {
      if (show) {
        setHeight(contentRef.current.scrollHeight);
        const timer = setTimeout(() => setHeight('auto'), duration);
        return () => clearTimeout(timer);
      } else {
        // First set to actual height, then to 0
        setHeight(contentRef.current.scrollHeight);
        requestAnimationFrame(() => {
          setHeight(0);
        });
      }
    }
  }, [show, duration]);

  const effectiveDuration = prefersReducedMotion ? 0 : duration;

  return (
    <div
      className={`overflow-hidden ${className}`}
      style={{
        height: height === 'auto' ? 'auto' : height,
        transition: `height ${effectiveDuration}ms ease-out`,
      }}
    >
      <div ref={contentRef}>{children}</div>
    </div>
  );
}

// ============================================
// Staggered List Animation
// ============================================

interface StaggeredListProps {
  children: ReactNode[];
  show: boolean;
  staggerDelay?: number;
  duration?: number;
  direction?: 'up' | 'down';
  className?: string;
}

export function StaggeredList({
  children,
  show,
  staggerDelay = 50,
  duration = 300,
  direction = 'up',
  className = '',
}: StaggeredListProps) {
  const prefersReducedMotion = usePrefersReducedMotion();

  return (
    <div className={className}>
      {children.map((child, index) => (
        <div
          key={index}
          style={{
            opacity: show ? 1 : 0,
            transform: show
              ? 'translateY(0)'
              : `translateY(${direction === 'up' ? '10px' : '-10px'})`,
            transition: prefersReducedMotion
              ? 'none'
              : `opacity ${duration}ms ease-out, transform ${duration}ms ease-out`,
            transitionDelay: prefersReducedMotion ? '0ms' : `${index * staggerDelay}ms`,
          }}
        >
          {child}
        </div>
      ))}
    </div>
  );
}

// ============================================
// Animate Presence (simpler AnimatePresence)
// ============================================

interface AnimatePresenceProps {
  children: ReactNode;
  show: boolean;
  initial?: CSSProperties;
  animate?: CSSProperties;
  exit?: CSSProperties;
  duration?: number;
  className?: string;
}

export function AnimatePresence({
  children,
  show,
  initial = { opacity: 0 },
  animate = { opacity: 1 },
  exit = { opacity: 0 },
  duration = 200,
  className = '',
}: AnimatePresenceProps) {
  const [shouldRender, setShouldRender] = useState(show);
  const [currentStyle, setCurrentStyle] = useState<CSSProperties>(show ? animate : initial);
  const prefersReducedMotion = usePrefersReducedMotion();

  useEffect(() => {
    if (show) {
      setShouldRender(true);
      // Small delay to ensure initial state is applied
      requestAnimationFrame(() => {
        setCurrentStyle(animate);
      });
    } else {
      setCurrentStyle(exit);
    }
  }, [show, animate, exit]);

  if (!shouldRender) return null;

  const effectiveDuration = prefersReducedMotion ? 0 : duration;

  return (
    <div
      className={className}
      style={{
        ...currentStyle,
        transition: `all ${effectiveDuration}ms ease-out`,
      }}
      onTransitionEnd={() => {
        if (!show) setShouldRender(false);
      }}
    >
      {children}
    </div>
  );
}

// ============================================
// Pulse Animation (for notifications)
// ============================================

interface PulseProps {
  children: ReactNode;
  active?: boolean;
  color?: string;
  className?: string;
}

export function Pulse({
  children,
  active = true,
  color = 'rgba(59, 130, 246, 0.5)',
  className = '',
}: PulseProps) {
  const prefersReducedMotion = usePrefersReducedMotion();

  if (!active || prefersReducedMotion) {
    return <div className={className}>{children}</div>;
  }

  return (
    <div className={`relative ${className}`}>
      <div
        className="absolute inset-0 rounded-full animate-ping"
        style={{ backgroundColor: color, opacity: 0.75 }}
      />
      <div className="relative">{children}</div>
    </div>
  );
}

// ============================================
// Shimmer Effect (for loading)
// ============================================

export function Shimmer({ className = '' }: { className?: string }) {
  return (
    <div
      className={`relative overflow-hidden ${className}`}
      style={{ background: 'linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.2) 50%, transparent 100%)' }}
    >
      <div
        className="absolute inset-0"
        style={{
          animation: 'shimmer 1.5s infinite',
          background: 'linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.4) 50%, transparent 100%)',
        }}
      />
    </div>
  );
}

// ============================================
// Number Counter Animation
// ============================================

interface CounterProps {
  value: number;
  duration?: number;
  className?: string;
}

export function Counter({ value, duration = 1000, className = '' }: CounterProps) {
  const [displayValue, setDisplayValue] = useState(0);
  const prefersReducedMotion = usePrefersReducedMotion();

  useEffect(() => {
    if (prefersReducedMotion) {
      setDisplayValue(value);
      return;
    }

    const startValue = displayValue;
    const startTime = performance.now();

    const animate = (currentTime: number) => {
      const elapsed = currentTime - startTime;
      const progress = Math.min(elapsed / duration, 1);

      // Ease out cubic
      const eased = 1 - Math.pow(1 - progress, 3);
      const current = Math.round(startValue + (value - startValue) * eased);

      setDisplayValue(current);

      if (progress < 1) {
        requestAnimationFrame(animate);
      }
    };

    requestAnimationFrame(animate);
  }, [value, duration, prefersReducedMotion]);

  return <span className={className}>{displayValue}</span>;
}

// ============================================
// Progress Bar Animation
// ============================================

interface ProgressBarProps {
  value: number;
  max?: number;
  duration?: number;
  color?: string;
  className?: string;
}

export function ProgressBar({
  value,
  max = 100,
  duration = 500,
  color = 'bg-blue-500',
  className = '',
}: ProgressBarProps) {
  const [width, setWidth] = useState(0);
  const prefersReducedMotion = usePrefersReducedMotion();

  useEffect(() => {
    const percentage = Math.min((value / max) * 100, 100);
    if (prefersReducedMotion) {
      setWidth(percentage);
    } else {
      // Small delay for enter animation
      const timer = setTimeout(() => setWidth(percentage), 50);
      return () => clearTimeout(timer);
    }
  }, [value, max, prefersReducedMotion]);

  return (
    <div className={`h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden ${className}`}>
      <div
        className={`h-full ${color} rounded-full`}
        style={{
          width: `${width}%`,
          transition: prefersReducedMotion ? 'none' : `width ${duration}ms ease-out`,
        }}
      />
    </div>
  );
}

// ============================================
// CSS Keyframes (add to global styles)
// ============================================

export const animationKeyframes = `
@keyframes fadeIn {
  from { opacity: 0; }
  to { opacity: 1; }
}

@keyframes fadeOut {
  from { opacity: 1; }
  to { opacity: 0; }
}

@keyframes slideUp {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}

@keyframes slideDown {
  from { opacity: 0; transform: translateY(-10px); }
  to { opacity: 1; transform: translateY(0); }
}

@keyframes scaleIn {
  from { opacity: 0; transform: scale(0.95); }
  to { opacity: 1; transform: scale(1); }
}

@keyframes shimmer {
  0% { transform: translateX(-100%); }
  100% { transform: translateX(100%); }
}

@keyframes bounce {
  0%, 100% { transform: translateY(0); }
  50% { transform: translateY(-5px); }
}
`;

export default {
  Fade,
  Slide,
  Scale,
  Collapse,
  StaggeredList,
  AnimatePresence,
  Pulse,
  Shimmer,
  Counter,
  ProgressBar,
  usePrefersReducedMotion,
};
