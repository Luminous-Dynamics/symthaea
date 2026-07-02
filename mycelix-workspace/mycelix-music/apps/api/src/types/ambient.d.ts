// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// Minimal ambient module declarations to satisfy TypeScript in strict mode
// for runtime-only dependencies without @types packages installed.

declare module 'node-fetch' {
  const fetch: typeof globalThis.fetch;
  export default fetch;
}

declare module 'prom-client' {
  export class Registry {
    contentType: string;
    registerMetric(metric: any): void;
    metrics(): Promise<string>;
  }
  export class Counter<T extends string = string> {
    constructor(config: { name: string; help: string; labelNames?: T[] });
    inc(labels?: Record<T, string | number>, value?: number): void;
    labels(...values: (string | number)[]): { inc(value?: number): void };
  }
  export class Gauge<T extends string = string> {
    constructor(config: { name: string; help: string; labelNames?: T[] });
    set(value: number): void;
    labels(...values: (string | number)[]): { set(value: number): void };
  }
  export class Histogram<T extends string = string> {
    constructor(config: { name: string; help: string; labelNames?: T[]; buckets?: number[] });
    observe(value: number): void;
    labels(...values: (string | number)[]): { observe(value: number): void };
  }
  export function collectDefaultMetrics(config: { register: Registry }): void;
  const client: {
    Registry: typeof Registry;
    Counter: typeof Counter;
    Gauge: typeof Gauge;
    Histogram: typeof Histogram;
    collectDefaultMetrics: typeof collectDefaultMetrics;
  };
  export default client;
}

declare module 'multer' {
  const multer: any;
  export default multer;
}

declare module 'form-data' {
  const FormData: any;
  export default FormData;
}

declare module 'key-did-provider-ed25519' {
  export const Ed25519Provider: any;
}

declare module 'key-did-resolver' {
  export function getResolver(...args: any[]): any;
}

declare module 'helmet' {
  interface HelmetOptions {
    contentSecurityPolicy?: {
      directives?: Record<string, string[]>;
    } | boolean;
    crossOriginEmbedderPolicy?: boolean;
    crossOriginResourcePolicy?: { policy: string } | boolean;
    hsts?: {
      maxAge?: number;
      includeSubDomains?: boolean;
      preload?: boolean;
    } | boolean;
    noSniff?: boolean;
    xssFilter?: boolean;
    referrerPolicy?: { policy: string } | boolean;
  }
  function helmet(options?: HelmetOptions): any;
  export default helmet;
}

declare module 'express-rate-limit' {
  const rateLimit: any;
  export default rateLimit;
}

declare module 'file-type' {
  export function fileTypeFromBuffer(buffer: Buffer | Uint8Array): Promise<{ ext: string; mime: string } | undefined>;
}

// Augment Express Request for multer .file support used in upload endpoint
declare namespace Express {
  export interface Request {
    file?: {
      buffer: Buffer;
      originalname: string;
      mimetype: string;
      size: number;
    };
  }
}

