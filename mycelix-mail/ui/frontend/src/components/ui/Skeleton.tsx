// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Skeleton Loading Components
 *
 * Provides smooth loading states for all epistemic components:
 * - Email list skeletons
 * - Trust graph skeleton
 * - Contact profile skeleton
 * - Insights panel skeleton
 */

import { memo } from 'react';

// Base skeleton with shimmer animation
export const Skeleton = memo(function Skeleton({
  className = '',
  animate = true,
}: {
  className?: string;
  animate?: boolean;
}) {
  return (
    <div
      className={`
        bg-gray-200 dark:bg-gray-700 rounded
        ${animate ? 'animate-pulse' : ''}
        ${className}
      `}
      aria-hidden="true"
    />
  );
});

// Circle skeleton for avatars
export const SkeletonCircle = memo(function SkeletonCircle({
  size = 'md',
}: {
  size?: 'sm' | 'md' | 'lg' | 'xl';
}) {
  const sizeClasses = {
    sm: 'w-6 h-6',
    md: 'w-10 h-10',
    lg: 'w-14 h-14',
    xl: 'w-20 h-20',
  };

  return <Skeleton className={`${sizeClasses[size]} rounded-full`} />;
});

// Text line skeleton
export const SkeletonText = memo(function SkeletonText({
  lines = 1,
  lastLineWidth = 'full',
}: {
  lines?: number;
  lastLineWidth?: 'full' | '3/4' | '1/2' | '1/4';
}) {
  const widthClasses = {
    full: 'w-full',
    '3/4': 'w-3/4',
    '1/2': 'w-1/2',
    '1/4': 'w-1/4',
  };

  return (
    <div className="space-y-2">
      {Array.from({ length: lines }).map((_, i) => (
        <Skeleton
          key={i}
          className={`h-4 ${
            i === lines - 1 ? widthClasses[lastLineWidth] : 'w-full'
          }`}
        />
      ))}
    </div>
  );
});

// Email row skeleton
export const EmailRowSkeleton = memo(function EmailRowSkeleton() {
  return (
    <div className="flex items-center gap-4 px-4 py-3 border-b border-gray-100 dark:border-gray-800">
      <SkeletonCircle size="md" />
      <div className="flex-1 min-w-0">
        <div className="flex items-center justify-between mb-1">
          <Skeleton className="h-4 w-32" />
          <Skeleton className="h-3 w-16" />
        </div>
        <Skeleton className="h-4 w-48 mb-1" />
        <Skeleton className="h-3 w-64" />
      </div>
      <div className="flex items-center gap-2">
        <Skeleton className="h-5 w-12 rounded-full" />
        <Skeleton className="h-5 w-8 rounded-full" />
      </div>
    </div>
  );
});

// Email list skeleton
export const EmailListSkeleton = memo(function EmailListSkeleton({
  count = 5,
}: {
  count?: number;
}) {
  return (
    <div role="status" aria-label="Loading emails">
      {Array.from({ length: count }).map((_, i) => (
        <EmailRowSkeleton key={i} />
      ))}
      <span className="sr-only">Loading emails...</span>
    </div>
  );
});

// Trust badge skeleton
export const TrustBadgeSkeleton = memo(function TrustBadgeSkeleton() {
  return (
    <div className="flex items-center gap-2">
      <Skeleton className="h-6 w-6 rounded-full" />
      <Skeleton className="h-4 w-16" />
    </div>
  );
});

// Contact profile skeleton
export const ContactProfileSkeleton = memo(function ContactProfileSkeleton() {
  return (
    <div className="p-6 space-y-6" role="status" aria-label="Loading contact profile">
      {/* Header */}
      <div className="flex items-start gap-4">
        <SkeletonCircle size="xl" />
        <div className="flex-1 space-y-2">
          <Skeleton className="h-6 w-48" />
          <Skeleton className="h-4 w-32" />
          <Skeleton className="h-4 w-40" />
        </div>
      </div>

      {/* Trust info */}
      <div className="grid grid-cols-3 gap-4">
        {Array.from({ length: 3 }).map((_, i) => (
          <div key={i} className="p-3 rounded-lg bg-gray-50 dark:bg-gray-800">
            <Skeleton className="h-3 w-16 mb-2" />
            <Skeleton className="h-6 w-12" />
          </div>
        ))}
      </div>

      {/* Attestations */}
      <div className="space-y-3">
        <Skeleton className="h-5 w-24" />
        {Array.from({ length: 3 }).map((_, i) => (
          <div key={i} className="flex items-center gap-3 p-3 rounded-lg bg-gray-50 dark:bg-gray-800">
            <SkeletonCircle size="sm" />
            <div className="flex-1">
              <Skeleton className="h-4 w-32 mb-1" />
              <Skeleton className="h-3 w-24" />
            </div>
          </div>
        ))}
      </div>

      <span className="sr-only">Loading contact profile...</span>
    </div>
  );
});

// AI insights skeleton
export const InsightsPanelSkeleton = memo(function InsightsPanelSkeleton() {
  return (
    <div className="p-4 space-y-4" role="status" aria-label="Loading AI insights">
      {/* Intent */}
      <div className="space-y-2">
        <Skeleton className="h-4 w-20" />
        <div className="flex items-center gap-2">
          <Skeleton className="h-6 w-24 rounded-full" />
          <Skeleton className="h-6 w-16 rounded-full" />
        </div>
      </div>

      {/* Summary */}
      <div className="space-y-2">
        <Skeleton className="h-4 w-16" />
        <SkeletonText lines={3} lastLineWidth="3/4" />
      </div>

      {/* Action items */}
      <div className="space-y-2">
        <Skeleton className="h-4 w-24" />
        {Array.from({ length: 2 }).map((_, i) => (
          <div key={i} className="flex items-start gap-2">
            <Skeleton className="h-4 w-4 mt-0.5" />
            <Skeleton className="h-4 flex-1" />
          </div>
        ))}
      </div>

      <span className="sr-only">Loading AI insights...</span>
    </div>
  );
});

// Trust graph skeleton
export const TrustGraphSkeleton = memo(function TrustGraphSkeleton() {
  return (
    <div
      className="relative w-full h-64 bg-gray-50 dark:bg-gray-900 rounded-lg overflow-hidden"
      role="status"
      aria-label="Loading trust graph"
    >
      {/* Simulated nodes */}
      <div className="absolute inset-0 flex items-center justify-center">
        <div className="relative">
          {/* Center node */}
          <Skeleton className="w-16 h-16 rounded-full" />

          {/* Orbiting nodes */}
          {[0, 60, 120, 180, 240, 300].map((angle, i) => {
            const x = Math.cos((angle * Math.PI) / 180) * 80;
            const y = Math.sin((angle * Math.PI) / 180) * 80;
            return (
              <Skeleton
                key={i}
                className="absolute w-10 h-10 rounded-full"
                style={{
                  left: `calc(50% + ${x}px - 20px)`,
                  top: `calc(50% + ${y}px - 20px)`,
                  animationDelay: `${i * 100}ms`,
                }}
              />
            );
          })}
        </div>
      </div>

      <span className="sr-only">Loading trust graph...</span>
    </div>
  );
});

// Thread summary skeleton
export const ThreadSummarySkeleton = memo(function ThreadSummarySkeleton() {
  return (
    <div className="p-4 space-y-4" role="status" aria-label="Loading thread summary">
      {/* Stats */}
      <div className="grid grid-cols-4 gap-2">
        {Array.from({ length: 4 }).map((_, i) => (
          <div key={i} className="text-center p-2">
            <Skeleton className="h-8 w-8 mx-auto mb-1 rounded-full" />
            <Skeleton className="h-3 w-12 mx-auto" />
          </div>
        ))}
      </div>

      {/* Summary */}
      <div className="space-y-2">
        <Skeleton className="h-4 w-24" />
        <SkeletonText lines={4} lastLineWidth="1/2" />
      </div>

      {/* Participants */}
      <div className="space-y-2">
        <Skeleton className="h-4 w-20" />
        <div className="flex -space-x-2">
          {Array.from({ length: 4 }).map((_, i) => (
            <SkeletonCircle key={i} size="sm" />
          ))}
        </div>
      </div>

      <span className="sr-only">Loading thread summary...</span>
    </div>
  );
});

// Dashboard stats skeleton
export const DashboardSkeleton = memo(function DashboardSkeleton() {
  return (
    <div className="space-y-6" role="status" aria-label="Loading dashboard">
      {/* Stats grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {Array.from({ length: 4 }).map((_, i) => (
          <div key={i} className="p-4 rounded-xl bg-white dark:bg-gray-800 shadow-sm">
            <Skeleton className="h-4 w-20 mb-2" />
            <Skeleton className="h-8 w-16 mb-1" />
            <Skeleton className="h-3 w-24" />
          </div>
        ))}
      </div>

      {/* Chart area */}
      <div className="p-4 rounded-xl bg-white dark:bg-gray-800 shadow-sm">
        <Skeleton className="h-5 w-32 mb-4" />
        <TrustGraphSkeleton />
      </div>

      {/* List */}
      <div className="p-4 rounded-xl bg-white dark:bg-gray-800 shadow-sm">
        <Skeleton className="h-5 w-40 mb-4" />
        {Array.from({ length: 5 }).map((_, i) => (
          <div key={i} className="flex items-center gap-3 py-3 border-b border-gray-100 dark:border-gray-700 last:border-0">
            <SkeletonCircle size="md" />
            <div className="flex-1">
              <Skeleton className="h-4 w-32 mb-1" />
              <Skeleton className="h-3 w-24" />
            </div>
            <Skeleton className="h-6 w-16 rounded-full" />
          </div>
        ))}
      </div>

      <span className="sr-only">Loading dashboard...</span>
    </div>
  );
});

// Settings skeleton
export const SettingsSkeleton = memo(function SettingsSkeleton() {
  return (
    <div className="space-y-6" role="status" aria-label="Loading settings">
      {/* Tabs */}
      <div className="flex gap-2 border-b border-gray-200 dark:border-gray-700 pb-2">
        {Array.from({ length: 5 }).map((_, i) => (
          <Skeleton key={i} className="h-8 w-20 rounded-lg" />
        ))}
      </div>

      {/* Settings items */}
      {Array.from({ length: 6 }).map((_, i) => (
        <div key={i} className="flex items-center justify-between py-4 border-b border-gray-100 dark:border-gray-800">
          <div className="space-y-1">
            <Skeleton className="h-4 w-32" />
            <Skeleton className="h-3 w-48" />
          </div>
          <Skeleton className="h-6 w-12 rounded-full" />
        </div>
      ))}

      <span className="sr-only">Loading settings...</span>
    </div>
  );
});

export default {
  Skeleton,
  SkeletonCircle,
  SkeletonText,
  EmailRowSkeleton,
  EmailListSkeleton,
  TrustBadgeSkeleton,
  ContactProfileSkeleton,
  InsightsPanelSkeleton,
  TrustGraphSkeleton,
  ThreadSummarySkeleton,
  DashboardSkeleton,
  SettingsSkeleton,
};
