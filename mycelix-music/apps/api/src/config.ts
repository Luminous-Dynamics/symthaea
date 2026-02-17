import { z } from 'zod';

const schema = z.object({
  NODE_ENV: z.string().optional(),
  API_PORT: z.string().default('3100'),
  DATABASE_URL: z.string().optional(),
  REDIS_URL: z.string().optional(),
  RPC_URL: z.string().optional(),
  API_CHAIN_RPC_URL: z.string().optional(),
  ROUTER_ADDRESS: z.string().optional(),
  NEXT_PUBLIC_ROUTER_ADDRESS: z.string().optional(),
  API_ADMIN_KEY: z.string().optional(),
  ENABLE_UPLOADS: z.string().optional(),
  UPLOAD_AUTH_MODE: z.enum(['admin', 'open']).default('admin'),
  ENABLE_API_DOCS: z.string().optional(),
  ENABLE_SWAGGER_UI: z.string().optional(),
});

export type AppConfig = ReturnType<typeof loadConfig>;

export function loadConfig() {
  const raw = schema.parse(process.env);
  const routerAddress = raw.ROUTER_ADDRESS || raw.NEXT_PUBLIC_ROUTER_ADDRESS || '';
  const rpcUrl = raw.API_CHAIN_RPC_URL || raw.RPC_URL || 'http://localhost:8545';
  const port = parseInt(raw.API_PORT, 10);
  const isTest = raw.NODE_ENV === 'test';

  const missingCritical: string[] = [];
  if (!raw.DATABASE_URL && !isTest) missingCritical.push('DATABASE_URL');
  if (!raw.REDIS_URL && !isTest) missingCritical.push('REDIS_URL');

  return {
    raw,
    port,
    routerAddress,
    rpcUrl,
    uploadAuthMode: raw.UPLOAD_AUTH_MODE,
    missingCritical,
    isTest,
  };
}
