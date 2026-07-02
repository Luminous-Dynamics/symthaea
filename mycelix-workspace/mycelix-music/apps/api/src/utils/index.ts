// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Utilities Index
 *
 * Central export for utility functions and helpers.
 */

export {
  response,
  success,
  paginated,
  created,
  noContent,
  error,
  errors,
  ErrorCodes,
  isErrorResponse,
  isPaginatedResponse,
  ResponseBuilder,
} from './response';

export type {
  ErrorCode,
  PaginationMeta,
  ResponseMeta,
  ErrorDetails,
  ApiResponse,
} from './response';
