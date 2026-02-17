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
