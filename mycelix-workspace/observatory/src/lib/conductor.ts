import { AppWebsocket } from '@holochain/client';
import { browser } from '$app/environment';

type ZomeCallParams = {
  role_name: string;
  zome_name: string;
  fn_name: string;
  payload: unknown;
  cap_secret?: Uint8Array | null;
};

const APP_WS_URL =
  import.meta.env.VITE_CONDUCTOR_URL ??
  import.meta.env.VITE_HOLOCHAIN_APP_WS_URL ??
  'ws://localhost:8888';

let client: AppWebsocket | null = null;
let pendingConnect: Promise<AppWebsocket | null> | null = null;

export async function connectToConductor(): Promise<boolean> {
  if (!browser) return false;
  if (client) return true;
  if (!pendingConnect) {
    pendingConnect = AppWebsocket.connect(APP_WS_URL)
      .then((connected) => {
        client = connected;
        return connected;
      })
      .catch((error) => {
        console.warn('[Observatory] Conductor connection failed:', error);
        client = null;
        return null;
      });
  }
  const result = await pendingConnect;
  if (result) return true;
  pendingConnect = null;
  return false;
}

export function isConnected(): boolean {
  return client !== null;
}

export function getClient(): AppWebsocket {
  if (!client) {
    throw new Error('Holochain conductor is not connected');
  }
  return client;
}

export async function callZome<T>(params: ZomeCallParams): Promise<T> {
  const connected = await connectToConductor();
  if (!connected || !client) {
    throw new Error('Holochain conductor is not connected');
  }
  return client.callZome(params);
}
