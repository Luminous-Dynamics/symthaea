// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * OpenAPI Index
 *
 * Exports OpenAPI generator and specification.
 */

export { OpenAPIBuilder, zodToOpenAPI } from './generator';
export type {
  OpenAPIDocument,
  OpenAPIOperation,
  OpenAPIParameter,
  OpenAPIPathItem,
  OpenAPISchema,
  RouteDefinition,
} from './generator';

export { buildOpenAPISpec } from './spec';
