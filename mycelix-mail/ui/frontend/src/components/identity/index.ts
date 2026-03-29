// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Identity Components Index
 *
 * External identity provider integrations:
 * - ENS verification
 * - GitHub verification
 * - Domain verification
 * - Lens Protocol (coming soon)
 * - Farcaster (coming soon)
 */

export {
  default as ExternalIdentityProviders,
  ENSVerification,
  GitHubVerification,
  DomainVerification,
} from './ExternalProviders';
