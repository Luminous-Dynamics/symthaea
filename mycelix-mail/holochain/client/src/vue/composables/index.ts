// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mycelix Mail - Vue Composables
 *
 * Reactive state management for Vue 3 applications.
 */

export { useMessages } from './useMessages';
export type {
  EmailMessage,
  UseMessagesOptions,
  UseMessagesReturn,
} from './useMessages';

export { useTrust } from './useTrust';
export type {
  TrustScore,
  Attestation,
  UseTrustReturn,
} from './useTrust';

export { useContacts } from './useContacts';
export type {
  Contact,
  ContactGroup,
  UseContactsReturn,
} from './useContacts';

export { useCompose } from './useCompose';
export type {
  Recipient,
  Attachment,
  Draft,
  UseComposeOptions,
  UseComposeReturn,
} from './useCompose';
