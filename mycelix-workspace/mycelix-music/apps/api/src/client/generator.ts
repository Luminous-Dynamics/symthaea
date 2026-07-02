// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * TypeScript API Client Generator
 *
 * Generates a fully-typed API client from OpenAPI specification.
 * Output can be used in frontend apps or SDK packages.
 */

import { OpenAPIDocument, OpenAPIOperation, OpenAPISchema } from '../openapi/generator';

/**
 * Convert OpenAPI type to TypeScript type
 */
function schemaToTypeScript(schema: OpenAPISchema, indent = 0): string {
  const spaces = '  '.repeat(indent);

  if (!schema.type && schema.properties) {
    schema.type = 'object';
  }

  if (schema.nullable) {
    const inner = schemaToTypeScript({ ...schema, nullable: undefined }, indent);
    return `${inner} | null`;
  }

  if (schema.enum) {
    return schema.enum.map(v => `'${v}'`).join(' | ');
  }

  switch (schema.type) {
    case 'string':
      if (schema.format === 'date-time') return 'string'; // ISO date string
      if (schema.format === 'uuid') return 'string';
      return 'string';

    case 'number':
    case 'integer':
      return 'number';

    case 'boolean':
      return 'boolean';

    case 'array':
      if (schema.items) {
        return `Array<${schemaToTypeScript(schema.items, indent)}>`;
      }
      return 'unknown[]';

    case 'object':
      if (!schema.properties) {
        return 'Record<string, unknown>';
      }

      const required = new Set(schema.required || []);
      const props = Object.entries(schema.properties)
        .map(([key, prop]) => {
          const optional = required.has(key) ? '' : '?';
          const type = schemaToTypeScript(prop as OpenAPISchema, indent + 1);
          return `${spaces}  ${key}${optional}: ${type};`;
        })
        .join('\n');

      return `{\n${props}\n${spaces}}`;

    default:
      return 'unknown';
  }
}

/**
 * Generate TypeScript interface from schema
 */
function generateInterface(name: string, schema: OpenAPISchema): string {
  const type = schemaToTypeScript(schema);
  if (type.startsWith('{')) {
    return `export interface ${name} ${type}`;
  }
  return `export type ${name} = ${type};`;
}

/**
 * Convert path to method name
 */
function pathToMethodName(method: string, path: string): string {
  // Remove /api/v2 prefix
  let cleanPath = path.replace(/^\/api\/v\d+/, '');

  // Convert path params to camelCase parts
  const parts = cleanPath
    .split('/')
    .filter(Boolean)
    .map((part, index) => {
      if (part.startsWith('{') && part.endsWith('}')) {
        const param = part.slice(1, -1);
        return 'By' + param.charAt(0).toUpperCase() + param.slice(1);
      }
      if (index === 0) {
        return part;
      }
      return part.charAt(0).toUpperCase() + part.slice(1);
    });

  // Add method prefix for non-GET
  const prefix = method === 'get' ? '' :
    method === 'post' ? 'create' :
    method === 'put' ? 'update' :
    method === 'patch' ? 'update' :
    method === 'delete' ? 'delete' : method;

  if (prefix && parts.length > 0) {
    parts[0] = parts[0].charAt(0).toUpperCase() + parts[0].slice(1);
  }

  return prefix + parts.join('');
}

/**
 * Generate method for an operation
 */
function generateMethod(
  method: string,
  path: string,
  operation: OpenAPIOperation
): { signature: string; implementation: string } {
  const methodName = operation.operationId || pathToMethodName(method, path);

  // Collect parameters
  const params: string[] = [];
  const pathParams: string[] = [];
  const queryParams: string[] = [];

  if (operation.parameters) {
    for (const param of operation.parameters) {
      const type = schemaToTypeScript(param.schema);
      const optional = param.required ? '' : '?';

      if (param.in === 'path') {
        params.push(`${param.name}: ${type}`);
        pathParams.push(param.name);
      } else if (param.in === 'query') {
        queryParams.push(`${param.name}${optional}: ${type}`);
      }
    }
  }

  // Add query params as options object
  if (queryParams.length > 0) {
    params.push(`options?: { ${queryParams.join('; ')} }`);
  }

  // Add body parameter
  if (operation.requestBody) {
    const bodySchema = operation.requestBody.content['application/json']?.schema;
    if (bodySchema) {
      const bodyType = schemaToTypeScript(bodySchema);
      params.push(`body: ${bodyType}`);
    }
  }

  // Determine return type
  let returnType = 'void';
  const successResponse = operation.responses['200'] || operation.responses['201'];
  if (successResponse?.content?.['application/json']?.schema) {
    returnType = schemaToTypeScript(successResponse.content['application/json'].schema);
  }

  // Build signature
  const signature = `${methodName}(${params.join(', ')}): Promise<${returnType}>`;

  // Build implementation
  let urlExpr = `\`${path.replace(/{(\w+)}/g, '${$1}')}\``;

  const fetchOptions: string[] = [`method: '${method.toUpperCase()}'`];

  if (queryParams.length > 0) {
    fetchOptions.push(`params: options`);
  }

  if (operation.requestBody) {
    fetchOptions.push(`body: JSON.stringify(body)`);
  }

  const implementation = `
  async ${methodName}(${params.join(', ')}): Promise<${returnType}> {
    return this.request<${returnType}>(${urlExpr}, {
      ${fetchOptions.join(',\n      ')}
    });
  }`.trim();

  return { signature, implementation };
}

/**
 * Generate the complete API client
 */
export function generateApiClient(spec: OpenAPIDocument): string {
  const interfaces: string[] = [];
  const methods: { signature: string; implementation: string }[] = [];

  // Generate interfaces from components
  if (spec.components?.schemas) {
    for (const [name, schema] of Object.entries(spec.components.schemas)) {
      interfaces.push(generateInterface(name, schema));
    }
  }

  // Generate methods from paths
  for (const [path, pathItem] of Object.entries(spec.paths)) {
    for (const method of ['get', 'post', 'put', 'patch', 'delete'] as const) {
      const operation = pathItem[method];
      if (operation) {
        methods.push(generateMethod(method, path, operation));
      }
    }
  }

  // Build the client
  return `/**
 * Mycelix Music API Client
 *
 * Auto-generated TypeScript client for the Mycelix Music API.
 * Generated from OpenAPI specification.
 *
 * @generated
 */

// ==================== Types ====================

${interfaces.join('\n\n')}

// ==================== API Response Types ====================

export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: {
    code: string;
    message: string;
    details?: unknown;
  };
  meta?: {
    pagination?: {
      total: number;
      limit: number;
      offset: number;
      hasMore: boolean;
    };
    timing?: {
      startedAt: string;
      duration: number;
    };
    requestId?: string;
  };
}

// ==================== Client Configuration ====================

export interface ClientConfig {
  baseUrl: string;
  headers?: Record<string, string>;
  onRequest?: (url: string, options: RequestInit) => RequestInit;
  onResponse?: (response: Response) => void;
  onError?: (error: Error) => void;
}

// ==================== API Client ====================

export class MycelixClient {
  private config: ClientConfig;

  constructor(config: ClientConfig) {
    this.config = {
      ...config,
      baseUrl: config.baseUrl.replace(/\\/$/, ''),
    };
  }

  /**
   * Set authentication header
   */
  setAuthHeader(signature: string): void {
    this.config.headers = {
      ...this.config.headers,
      'X-Wallet-Signature': signature,
    };
  }

  /**
   * Make an API request
   */
  private async request<T>(
    path: string,
    options: {
      method: string;
      params?: Record<string, unknown>;
      body?: string;
    }
  ): Promise<T> {
    let url = \`\${this.config.baseUrl}\${path}\`;

    // Add query parameters
    if (options.params) {
      const params = new URLSearchParams();
      for (const [key, value] of Object.entries(options.params)) {
        if (value !== undefined && value !== null) {
          params.set(key, String(value));
        }
      }
      const queryString = params.toString();
      if (queryString) {
        url += \`?\${queryString}\`;
      }
    }

    // Build request options
    let requestOptions: RequestInit = {
      method: options.method,
      headers: {
        'Content-Type': 'application/json',
        ...this.config.headers,
      },
      body: options.body,
    };

    // Apply request interceptor
    if (this.config.onRequest) {
      requestOptions = this.config.onRequest(url, requestOptions);
    }

    try {
      const response = await fetch(url, requestOptions);

      // Apply response interceptor
      if (this.config.onResponse) {
        this.config.onResponse(response);
      }

      if (!response.ok) {
        const error = await response.json().catch(() => ({
          error: { message: response.statusText }
        }));
        throw new ApiError(
          error.error?.message || 'Request failed',
          error.error?.code || 'UNKNOWN_ERROR',
          response.status,
          error.error?.details
        );
      }

      // Handle 204 No Content
      if (response.status === 204) {
        return undefined as T;
      }

      const data: ApiResponse<T> = await response.json();

      if (!data.success && data.error) {
        throw new ApiError(
          data.error.message,
          data.error.code,
          response.status,
          data.error.details
        );
      }

      return data.data as T;
    } catch (error) {
      if (this.config.onError) {
        this.config.onError(error as Error);
      }
      throw error;
    }
  }

  // ==================== API Methods ====================

${methods.map(m => '  ' + m.implementation.split('\n').join('\n  ')).join('\n\n')}
}

// ==================== Error Class ====================

export class ApiError extends Error {
  constructor(
    message: string,
    public readonly code: string,
    public readonly status: number,
    public readonly details?: unknown
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

// ==================== Factory Function ====================

export function createClient(baseUrl: string, options?: Partial<ClientConfig>): MycelixClient {
  return new MycelixClient({
    baseUrl,
    ...options,
  });
}

export default MycelixClient;
`;
}

/**
 * CLI entry point
 */
export async function generateClientCLI(): Promise<void> {
  const { buildOpenAPISpec } = await import('../openapi');
  const spec = buildOpenAPISpec();
  const client = generateApiClient(spec);
  console.log(client);
}

if (require.main === module) {
  generateClientCLI();
}
