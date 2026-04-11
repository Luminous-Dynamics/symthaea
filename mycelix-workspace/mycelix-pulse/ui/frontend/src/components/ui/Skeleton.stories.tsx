// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import type { Meta, StoryObj } from '@storybook/react';
import {
  Skeleton,
  SkeletonCircle,
  SkeletonText,
  EmailRowSkeleton,
  EmailListSkeleton,
  TrustBadgeSkeleton,
  ContactProfileSkeleton,
  InsightsPanelSkeleton,
  TrustGraphSkeleton,
  DashboardSkeleton,
} from './Skeleton';

const meta: Meta = {
  title: 'UI/Skeleton',
  tags: ['autodocs'],
  parameters: {
    layout: 'padded',
  },
};

export default meta;

export const Basic: StoryObj = {
  render: () => (
    <div className="space-y-4 w-64">
      <Skeleton className="h-4 w-full" />
      <Skeleton className="h-4 w-3/4" />
      <Skeleton className="h-4 w-1/2" />
    </div>
  ),
};

export const Circle: StoryObj = {
  render: () => (
    <div className="flex gap-4">
      <SkeletonCircle size="sm" />
      <SkeletonCircle size="md" />
      <SkeletonCircle size="lg" />
      <SkeletonCircle size="xl" />
    </div>
  ),
};

export const Text: StoryObj = {
  render: () => (
    <div className="w-64 space-y-8">
      <div>
        <h4 className="text-sm font-medium mb-2">Single line</h4>
        <SkeletonText lines={1} />
      </div>
      <div>
        <h4 className="text-sm font-medium mb-2">Paragraph</h4>
        <SkeletonText lines={3} lastLineWidth="3/4" />
      </div>
      <div>
        <h4 className="text-sm font-medium mb-2">Short paragraph</h4>
        <SkeletonText lines={2} lastLineWidth="1/2" />
      </div>
    </div>
  ),
};

export const EmailRow: StoryObj = {
  render: () => (
    <div className="w-full max-w-2xl border border-gray-200 rounded-lg overflow-hidden">
      <EmailRowSkeleton />
    </div>
  ),
};

export const EmailList: StoryObj = {
  render: () => (
    <div className="w-full max-w-2xl border border-gray-200 rounded-lg overflow-hidden">
      <EmailListSkeleton count={5} />
    </div>
  ),
};

export const TrustBadge: StoryObj = {
  render: () => <TrustBadgeSkeleton />,
};

export const ContactProfile: StoryObj = {
  render: () => (
    <div className="w-80 border border-gray-200 rounded-lg overflow-hidden">
      <ContactProfileSkeleton />
    </div>
  ),
};

export const InsightsPanel: StoryObj = {
  render: () => (
    <div className="w-80 border border-gray-200 rounded-lg overflow-hidden">
      <InsightsPanelSkeleton />
    </div>
  ),
};

export const TrustGraph: StoryObj = {
  render: () => (
    <div className="w-full max-w-lg">
      <TrustGraphSkeleton />
    </div>
  ),
};

export const Dashboard: StoryObj = {
  render: () => (
    <div className="w-full max-w-4xl">
      <DashboardSkeleton />
    </div>
  ),
};
