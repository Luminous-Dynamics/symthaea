// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * OpenAPI Spec Generator
 *
 * Generates OpenAPI 3.0 specification from Zod schemas and route definitions.
 */

import { z, ZodSchema, ZodObject, ZodString, ZodNumber, ZodBoolean, ZodArray, ZodEnum, ZodOptional, ZodNullable, ZodDefault, ZodEffects } from 'zod';

/**
 * OpenAPI schema types
 */
export interface OpenAPISchema {
  type?: string;
  format?: string;
  properties?: Record<string, OpenAPISchema>;
  items?: OpenAPISchema;
  required?: string[];
  enum?: string[];
  description?: string;
  example?: unknown;
  default?: unknown;
  minimum?: number;
  maximum?: number;
  minLength?: number;
  maxLength?: number;
  pattern?: string;
  nullable?: boolean;
  oneOf?: OpenAPISchema[];
  allOf?: OpenAPISchema[];
}

/**
 * OpenAPI operation
 */
export interface OpenAPIOperation {
  summary?: string;
  description?: string;
  tags?: string[];
  operationId?: string;
  parameters?: OpenAPIParameter[];
  requestBody?: {
    required?: boolean;
    content: {
      'application/json': {
        schema: OpenAPISchema;
      };
    };
  };
  responses: Record<string, {
    description: string;
    content?: {
      'application/json': {
        schema: OpenAPISchema;
      };
    };
  }>;
  security?: Array<Record<string, string[]>>;
}

/**
 * OpenAPI parameter
 */
export interface OpenAPIParameter {
  name: string;
  in: 'query' | 'path' | 'header' | 'cookie';
  required?: boolean;
  schema: OpenAPISchema;
  description?: string;
}

/**
 * OpenAPI path item
 */
export interface OpenAPIPathItem {
  get?: OpenAPIOperation;
  post?: OpenAPIOperation;
  put?: OpenAPIOperation;
  patch?: OpenAPIOperation;
  delete?: OpenAPIOperation;
}

/**
 * OpenAPI document
 */
export interface OpenAPIDocument {
  openapi: string;
  info: {
    title: string;
    version: string;
    description?: string;
    contact?: {
      name?: string;
      url?: string;
      email?: string;
    };
    license?: {
      name: string;
      url?: string;
    };
  };
  servers?: Array<{
    url: string;
    description?: string;
  }>;
  paths: Record<string, OpenAPIPathItem>;
  components?: {
    schemas?: Record<string, OpenAPISchema>;
    securitySchemes?: Record<string, unknown>;
  };
  tags?: Array<{
    name: string;
    description?: string;
  }>;
}

/**
 * Convert Zod schema to OpenAPI schema
 */
export function zodToOpenAPI(schema: ZodSchema): OpenAPISchema {
  // Handle ZodEffects (transforms, refinements)
  if (schema instanceof ZodEffects) {
    return zodToOpenAPI(schema._def.schema);
  }

  // Handle ZodOptional
  if (schema instanceof ZodOptional) {
    return zodToOpenAPI(schema.unwrap());
  }

  // Handle ZodNullable
  if (schema instanceof ZodNullable) {
    return {
      ...zodToOpenAPI(schema.unwrap()),
      nullable: true,
    };
  }

  // Handle ZodDefault
  if (schema instanceof ZodDefault) {
    const inner = zodToOpenAPI(schema._def.innerType);
    return {
      ...inner,
      default: schema._def.defaultValue(),
    };
  }

  // Handle ZodString
  if (schema instanceof ZodString) {
    const result: OpenAPISchema = { type: 'string' };

    for (const check of schema._def.checks) {
      switch (check.kind) {
        case 'min':
          result.minLength = check.value;
          break;
        case 'max':
          result.maxLength = check.value;
          break;
        case 'email':
          result.format = 'email';
          break;
        case 'url':
          result.format = 'uri';
          break;
        case 'uuid':
          result.format = 'uuid';
          break;
        case 'datetime':
          result.format = 'date-time';
          break;
        case 'regex':
          result.pattern = check.regex.source;
          break;
      }
    }

    return result;
  }

  // Handle ZodNumber
  if (schema instanceof ZodNumber) {
    const result: OpenAPISchema = { type: 'number' };

    for (const check of schema._def.checks) {
      switch (check.kind) {
        case 'min':
          result.minimum = check.value;
          break;
        case 'max':
          result.maximum = check.value;
          break;
        case 'int':
          result.type = 'integer';
          break;
      }
    }

    return result;
  }

  // Handle ZodBoolean
  if (schema instanceof ZodBoolean) {
    return { type: 'boolean' };
  }

  // Handle ZodArray
  if (schema instanceof ZodArray) {
    return {
      type: 'array',
      items: zodToOpenAPI(schema.element),
    };
  }

  // Handle ZodEnum
  if (schema instanceof ZodEnum) {
    return {
      type: 'string',
      enum: schema._def.values,
    };
  }

  // Handle ZodObject
  if (schema instanceof ZodObject) {
    const shape = schema.shape;
    const properties: Record<string, OpenAPISchema> = {};
    const required: string[] = [];

    for (const [key, value] of Object.entries(shape)) {
      properties[key] = zodToOpenAPI(value as ZodSchema);

      // Check if required (not optional)
      if (!(value instanceof ZodOptional) && !(value instanceof ZodDefault)) {
        required.push(key);
      }
    }

    return {
      type: 'object',
      properties,
      required: required.length > 0 ? required : undefined,
    };
  }

  // Fallback
  return { type: 'object' };
}

/**
 * Route definition for spec generation
 */
export interface RouteDefinition {
  method: 'get' | 'post' | 'put' | 'patch' | 'delete';
  path: string;
  summary?: string;
  description?: string;
  tags?: string[];
  operationId?: string;
  body?: ZodSchema;
  query?: ZodSchema;
  params?: ZodSchema;
  response?: ZodSchema;
  responses?: Record<number, { description: string; schema?: ZodSchema }>;
  security?: string[];
}

/**
 * OpenAPI spec builder
 */
export class OpenAPIBuilder {
  private doc: OpenAPIDocument;
  private schemas: Map<string, { schema: ZodSchema; name: string }> = new Map();

  constructor(info: OpenAPIDocument['info']) {
    this.doc = {
      openapi: '3.0.3',
      info,
      paths: {},
      components: {
        schemas: {},
        securitySchemes: {},
      },
      tags: [],
    };
  }

  /**
   * Add server
   */
  addServer(url: string, description?: string): this {
    if (!this.doc.servers) {
      this.doc.servers = [];
    }
    this.doc.servers.push({ url, description });
    return this;
  }

  /**
   * Add tag
   */
  addTag(name: string, description?: string): this {
    if (!this.doc.tags) {
      this.doc.tags = [];
    }
    this.doc.tags.push({ name, description });
    return this;
  }

  /**
   * Register a schema for reuse
   */
  registerSchema(name: string, schema: ZodSchema): this {
    this.schemas.set(name, { schema, name });
    if (this.doc.components?.schemas) {
      this.doc.components.schemas[name] = zodToOpenAPI(schema);
    }
    return this;
  }

  /**
   * Add security scheme
   */
  addSecurityScheme(name: string, scheme: unknown): this {
    if (this.doc.components?.securitySchemes) {
      this.doc.components.securitySchemes[name] = scheme;
    }
    return this;
  }

  /**
   * Add route
   */
  addRoute(route: RouteDefinition): this {
    const { method, path, summary, description, tags, operationId, body, query, params, response, responses, security } = route;

    // Convert Express path params to OpenAPI format
    const openApiPath = path.replace(/:(\w+)/g, '{$1}');

    if (!this.doc.paths[openApiPath]) {
      this.doc.paths[openApiPath] = {};
    }

    const operation: OpenAPIOperation = {
      summary,
      description,
      tags,
      operationId,
      parameters: [],
      responses: {
        '200': {
          description: 'Successful response',
          content: response ? {
            'application/json': {
              schema: zodToOpenAPI(response),
            },
          } : undefined,
        },
      },
    };

    // Add path parameters
    if (params) {
      const paramsSchema = zodToOpenAPI(params);
      if (paramsSchema.properties) {
        for (const [name, schema] of Object.entries(paramsSchema.properties)) {
          operation.parameters!.push({
            name,
            in: 'path',
            required: true,
            schema,
          });
        }
      }
    }

    // Add query parameters
    if (query) {
      const querySchema = zodToOpenAPI(query);
      if (querySchema.properties) {
        const required = querySchema.required || [];
        for (const [name, schema] of Object.entries(querySchema.properties)) {
          operation.parameters!.push({
            name,
            in: 'query',
            required: required.includes(name),
            schema,
          });
        }
      }
    }

    // Add request body
    if (body) {
      operation.requestBody = {
        required: true,
        content: {
          'application/json': {
            schema: zodToOpenAPI(body),
          },
        },
      };
    }

    // Add custom responses
    if (responses) {
      for (const [code, { description, schema }] of Object.entries(responses)) {
        operation.responses[code] = {
          description,
          content: schema ? {
            'application/json': {
              schema: zodToOpenAPI(schema),
            },
          } : undefined,
        };
      }
    }

    // Add security
    if (security) {
      operation.security = security.map(s => ({ [s]: [] }));
    }

    // Remove empty parameters array
    if (operation.parameters?.length === 0) {
      delete operation.parameters;
    }

    this.doc.paths[openApiPath][method] = operation;

    return this;
  }

  /**
   * Build the OpenAPI document
   */
  build(): OpenAPIDocument {
    return this.doc;
  }

  /**
   * Build as JSON string
   */
  toJSON(): string {
    return JSON.stringify(this.doc, null, 2);
  }

  /**
   * Build as YAML string (basic implementation)
   */
  toYAML(): string {
    return jsonToYaml(this.doc);
  }
}

/**
 * Simple JSON to YAML converter
 */
function jsonToYaml(obj: unknown, indent = 0): string {
  const spaces = '  '.repeat(indent);

  if (obj === null || obj === undefined) {
    return 'null';
  }

  if (typeof obj === 'string') {
    // Quote strings with special characters
    if (obj.includes(':') || obj.includes('#') || obj.includes('\n')) {
      return `"${obj.replace(/"/g, '\\"')}"`;
    }
    return obj;
  }

  if (typeof obj === 'number' || typeof obj === 'boolean') {
    return String(obj);
  }

  if (Array.isArray(obj)) {
    if (obj.length === 0) return '[]';
    return obj.map(item => {
      const value = jsonToYaml(item, indent + 1);
      if (typeof item === 'object' && item !== null) {
        return `\n${spaces}- ${value.trim()}`;
      }
      return `\n${spaces}- ${value}`;
    }).join('');
  }

  if (typeof obj === 'object') {
    const entries = Object.entries(obj);
    if (entries.length === 0) return '{}';

    return entries.map(([key, value]) => {
      const valueStr = jsonToYaml(value, indent + 1);
      if (typeof value === 'object' && value !== null && !Array.isArray(value)) {
        return `${spaces}${key}:\n${valueStr}`;
      }
      if (Array.isArray(value)) {
        return `${spaces}${key}:${valueStr}`;
      }
      return `${spaces}${key}: ${valueStr}`;
    }).join('\n');
  }

  return String(obj);
}

export default OpenAPIBuilder;
