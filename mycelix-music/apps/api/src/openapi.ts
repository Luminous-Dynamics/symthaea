export const openapi = {
  openapi: '3.0.3',
  info: {
    title: 'Mycelix Music API',
    version: '0.1.0',
    description: 'REST API for songs, analytics, claims, and uploads',
  },
  servers: [{ url: 'http://localhost:3100' }],
  paths: {
    '/health': { get: { summary: 'Basic health', responses: { '200': { description: 'OK' } } } },
    '/health/details': { get: { summary: 'Detailed health', responses: { '200': { description: 'JSON details' } } } },
    '/health/ready': { get: { summary: 'Readiness', responses: { '200': { description: 'Ready' }, '503': { description: 'Not Ready' } } } },
    '/api/songs': {
      get: {
        summary: 'List songs',
        parameters: [
          { name: 'q', in: 'query', schema: { type: 'string' } },
          { name: 'genre', in: 'query', schema: { type: 'string' } },
          { name: 'model', in: 'query', schema: { type: 'string' } },
          { name: 'limit', in: 'query', schema: { type: 'integer', minimum: 1, maximum: 100 } },
          { name: 'offset', in: 'query', schema: { type: 'integer', minimum: 0 } },
          { name: 'sort', in: 'query', schema: { type: 'string', enum: ['created_at','plays','earnings'] } },
          { name: 'order', in: 'query', schema: { type: 'string', enum: ['asc','desc'] } },
        ],
        responses: { '200': { description: 'Song list' } },
      },
      post: {
        summary: 'Register a song in index',
        description: 'Auth: x-api-key OR signer/timestamp/signature over "mycelix-song|id|artistAddress|ipfsHash|paymentModel|timestamp"',
        requestBody: { required: true },
        responses: { '201': { description: 'Created' }, '400': { description: 'Validation error' }, '403': { description: 'Auth failed' } },
      },
    },
    '/api/songs/export': {
      get: { summary: 'Export songs as CSV (no pagination)', responses: { '200': { description: 'CSV stream' } } },
    },
    '/api/songs/{id}': { get: { summary: 'Get song', parameters: [{ name: 'id', in: 'path', required: true }], responses: { '200': { description: 'Song' }, '404': { description: 'Not found' } } } },
    '/api/songs/{id}/plays': {
      get: { summary: 'Get play history', parameters: [{ name: 'id', in: 'path', required: true }], responses: { '200': { description: 'List' } } },
      post: {
        summary: 'Record play (manual ingest)',
        description: 'Auth: x-api-key OR signer/timestamp/signature over "mycelix-play|songId|listener|amount|paymentType|timestamp". Disabled by default; in practice only enabled for tests.',
        parameters: [{ name: 'id', in: 'path', required: true }],
        responses: { '200': { description: 'OK' }, '403': { description: 'Auth failed' } },
      },
    },
    '/api/songs/{id}/claim': { get: { summary: 'Get song claim by id (Ceramic)', parameters: [{ name: 'id', in: 'path', required: true }], responses: { '200': { description: 'Claim or stub' }, '404': { description: 'No claim' } } } },
    '/api/claims/{streamId}': { get: { summary: 'Get claim by streamId (Ceramic)', parameters: [{ name: 'streamId', in: 'path', required: true }], responses: { '200': { description: 'Claim' }, '404': { description: 'Not found' } } } },
    '/api/artists/{address}/songs': { get: { summary: 'Artist songs', parameters: [{ name: 'address', in: 'path', required: true }], responses: { '200': { description: 'List' } } } },
    '/api/artists/{address}/stats': { get: { summary: 'Artist stats aggregate', parameters: [{ name: 'address', in: 'path', required: true }], responses: { '200': { description: 'Stats' } } } },
    '/api/analytics/artist/{address}': { get: { summary: 'Artist analytics', parameters: [{ name: 'address', in: 'path', required: true }], responses: { '200': { description: 'Timeseries' } } } },
    '/api/analytics/song/{id}': { get: { summary: 'Song analytics', parameters: [{ name: 'id', in: 'path', required: true }], responses: { '200': { description: 'Timeseries' } } } },
    '/api/analytics/top-songs': { get: { summary: 'Top songs', responses: { '200': { description: 'List' } } } },
    '/api/upload-to-ipfs': {
      post: {
        summary: 'Upload to IPFS (admin-key protected)',
        description: 'Requires x-api-key. Streams file to IPFS HTTP API; defaults to local gateway if configured. Uploads can be disabled via ENABLE_UPLOADS and are rate-limited by IP and wallet.',
        security: [{ ApiKey: [] }],
        responses: { '200': { description: 'Hash + gateway' }, '403': { description: 'Auth failed' } },
      },
    },
    '/api/create-dkg-claim': {
      post: {
        summary: 'Create DKG claim',
        description: 'Auth: x-api-key OR signer/timestamp/signature over "mycelix-claim|songId|artistAddress|ipfsHash|title|timestamp"',
        parameters: [{ name: 'x-api-key', in: 'header', required: false, schema: { type: 'string' } }],
        responses: { '200': { description: 'Stream id' }, '403': { description: 'Forbidden' } },
      },
    },
    '/api/strategy-configs': {
      post: {
        summary: 'Store strategy config (admin)',
        description: 'Admin-key protected endpoint to persist strategy configuration JSON for deployment. Hash required; adminSignature optional unless ADMIN_SIGNER_PUBLIC_KEY is set.',
        parameters: [{ name: 'x-api-key', in: 'header', required: true, schema: { type: 'string' } }],
        responses: { '201': { description: 'Stored' }, '403': { description: 'Forbidden' } },
      },
    },
    '/api/strategy-configs/latest': {
      get: {
        summary: 'Fetch latest strategy config',
        responses: { '200': { description: 'Latest config' }, '404': { description: 'Not found' } },
      },
    },
    '/api/strategy-configs/{id}': {
      get: {
        summary: 'Get strategy config by id',
        parameters: [{ name: 'id', in: 'path', required: true }],
        responses: { '200': { description: 'Config' }, '404': { description: 'Not found' } },
      },
    },
    '/api/strategy-configs/{id}/publish': {
      post: {
        summary: 'Publish strategy config (hash required; signature required if ADMIN_SIGNER_PUBLIC_KEY set)',
        parameters: [{ name: 'id', in: 'path', required: true }, { name: 'x-api-key', in: 'header', required: true }],
        responses: { '200': { description: 'Published' }, '404': { description: 'Not found' } },
      },
    },
    '/api/indexer/replay': {
      post: {
        summary: 'Replay block range (admin)',
        description: 'Scans PaymentRecorded logs for the router in the given block range and re-ingests them into Postgres.',
        responses: { '200': { description: 'Replay summary' }, '400': { description: 'Invalid range' }, '500': { description: 'Replay failed' } },
      },
    },
    '/api/indexer/poison': {
      get: {
        summary: 'List poison queue (admin)',
        description: 'Inspect indexer events that failed after retries.',
        responses: { '200': { description: 'List' }, '500': { description: 'Error' } },
      },
    },
    '/api/indexer/poison/retry': {
      post: {
        summary: 'Retry poison queue (admin)',
        description: 'Attempts to replay a small batch of poison events by fetching receipts/logs.',
        responses: { '200': { description: 'Retry summary' }, '500': { description: 'Error' } },
      },
    },
  },
  components: {
    securitySchemes: {
      ApiKey: { type: 'apiKey', in: 'header', name: 'x-api-key' },
    },
  },
};
