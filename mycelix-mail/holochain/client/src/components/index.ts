// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mycelix Mail - React Component Library
 *
 * Full-featured components for building email UIs.
 */

// Common components
export { Avatar } from './common/Avatar';
export type { AvatarProps } from './common/Avatar';

export { TrustBadge } from './common/TrustBadge';
export type { TrustBadgeProps } from './common/TrustBadge';

// Inbox components
export { EmailListItem } from './inbox/EmailListItem';
export type { EmailListItemProps, Email } from './inbox/EmailListItem';

export { InboxList } from './inbox/InboxList';
export type { InboxListProps, BulkAction } from './inbox/InboxList';

// Compose components
export { ComposeEmail } from './compose/ComposeEmail';
export type {
  ComposeEmailProps,
  Recipient,
  Attachment,
  ComposedEmail,
} from './compose/ComposeEmail';

// Thread components
export { ThreadView } from './thread/ThreadView';
export type { ThreadViewProps, ThreadMessage } from './thread/ThreadView';

// Contact components
export { ContactList } from './contacts/ContactList';
export type {
  ContactListProps,
  Contact,
  ContactGroup,
} from './contacts/ContactList';
