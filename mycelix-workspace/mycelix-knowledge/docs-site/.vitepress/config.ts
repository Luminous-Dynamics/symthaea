// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import { defineConfig } from 'vitepress';

export default defineConfig({
  title: 'Mycelix Knowledge',
  description: 'Decentralized Knowledge Graph with Epistemic Classification',

  head: [
    ['link', { rel: 'icon', href: '/favicon.ico' }],
    ['meta', { name: 'theme-color', content: '#8b5cf6' }],
  ],

  themeConfig: {
    logo: '/logo.svg',

    nav: [
      { text: 'Guide', link: '/guide/' },
      { text: 'API', link: '/api/' },
      { text: 'SDK', link: '/sdk/' },
      { text: 'Integrations', link: '/integrations/' },
      {
        text: 'Tools',
        items: [
          { text: 'Storybook', link: '/storybook/' },
          { text: 'Demo', link: 'https://demo.mycelix.net' },
          { text: 'GraphQL Playground', link: 'https://api.mycelix.net/graphql' },
        ],
      },
      { text: 'GitHub', link: 'https://github.com/Luminous-Dynamics/mycelix-knowledge' },
    ],

    sidebar: {
      '/guide/': [
        {
          text: 'Introduction',
          items: [
            { text: 'What is Mycelix Knowledge?', link: '/guide/' },
            { text: 'Quick Start', link: '/guide/quickstart' },
            { text: 'Core Concepts', link: '/guide/concepts' },
          ],
        },
        {
          text: 'Epistemic Framework',
          items: [
            { text: 'E-N-M Classification', link: '/guide/epistemic/classification' },
            { text: 'Credibility Scoring', link: '/guide/epistemic/credibility' },
            { text: 'Belief Propagation', link: '/guide/epistemic/belief-propagation' },
            { text: 'Information Value', link: '/guide/epistemic/information-value' },
          ],
        },
        {
          text: 'Knowledge Graph',
          items: [
            { text: 'Claims', link: '/guide/graph/claims' },
            { text: 'Relationships', link: '/guide/graph/relationships' },
            { text: 'Dependency Trees', link: '/guide/graph/dependencies' },
          ],
        },
        {
          text: 'Fact-Checking',
          items: [
            { text: 'How It Works', link: '/guide/factcheck/overview' },
            { text: 'Verdicts', link: '/guide/factcheck/verdicts' },
            { text: 'Evidence Assessment', link: '/guide/factcheck/evidence' },
          ],
        },
        {
          text: 'Markets Integration',
          items: [
            { text: 'Verification Markets', link: '/guide/markets/verification' },
            { text: 'Claim-Market Bidirectional Flow', link: '/guide/markets/bidirectional' },
            { text: 'Resolution & Updates', link: '/guide/markets/resolution' },
          ],
        },
      ],
      '/api/': [
        {
          text: 'REST API',
          items: [
            { text: 'Overview', link: '/api/' },
            { text: 'Authentication', link: '/api/auth' },
            { text: 'Claims', link: '/api/claims' },
            { text: 'Search', link: '/api/search' },
            { text: 'Fact-Check', link: '/api/factcheck' },
            { text: 'Graph', link: '/api/graph' },
          ],
        },
        {
          text: 'GraphQL',
          items: [
            { text: 'Schema', link: '/api/graphql/schema' },
            { text: 'Queries', link: '/api/graphql/queries' },
            { text: 'Mutations', link: '/api/graphql/mutations' },
            { text: 'Subscriptions', link: '/api/graphql/subscriptions' },
          ],
        },
        {
          text: 'WebSocket',
          items: [
            { text: 'Real-time Events', link: '/api/websocket/events' },
            { text: 'Subscription Topics', link: '/api/websocket/topics' },
          ],
        },
      ],
      '/sdk/': [
        {
          text: 'TypeScript SDK',
          items: [
            { text: 'Installation', link: '/sdk/' },
            { text: 'KnowledgeClient', link: '/sdk/client' },
            { text: 'Claims API', link: '/sdk/claims' },
            { text: 'Graph API', link: '/sdk/graph' },
            { text: 'Query API', link: '/sdk/query' },
            { text: 'Inference API', link: '/sdk/inference' },
            { text: 'Fact-Check API', link: '/sdk/factcheck' },
            { text: 'Markets API', link: '/sdk/markets' },
          ],
        },
        {
          text: 'React',
          items: [
            { text: 'Setup', link: '/sdk/react/' },
            { text: 'Hooks', link: '/sdk/react/hooks' },
            { text: 'Components', link: '/sdk/react/components' },
          ],
        },
        {
          text: 'Svelte',
          items: [
            { text: 'Setup', link: '/sdk/svelte/' },
            { text: 'Stores', link: '/sdk/svelte/stores' },
            { text: 'Components', link: '/sdk/svelte/components' },
          ],
        },
      ],
      '/integrations/': [
        {
          text: 'Extensions',
          items: [
            { text: 'VS Code Extension', link: '/integrations/vscode' },
            { text: 'Browser Extension', link: '/integrations/browser' },
          ],
        },
        {
          text: 'Embeddable',
          items: [
            { text: 'Widgets', link: '/integrations/widgets' },
            { text: 'Widget Types', link: '/integrations/widgets/types' },
            { text: 'Customization', link: '/integrations/widgets/customization' },
          ],
        },
        {
          text: 'AI/LLM',
          items: [
            { text: 'LLM Integration', link: '/integrations/llm/' },
            { text: 'Classification', link: '/integrations/llm/classification' },
            { text: 'Evidence Extraction', link: '/integrations/llm/extraction' },
            { text: 'Fact Analysis', link: '/integrations/llm/analysis' },
          ],
        },
        {
          text: 'Holochain',
          items: [
            { text: 'DNA Architecture', link: '/integrations/holochain/dna' },
            { text: 'Zome Functions', link: '/integrations/holochain/zomes' },
            { text: 'Entry Types', link: '/integrations/holochain/entries' },
          ],
        },
      ],
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/Luminous-Dynamics/mycelix-knowledge' },
      { icon: 'discord', link: 'https://discord.gg/mycelix' },
    ],

    footer: {
      message: 'Released under the MIT License.',
      copyright: 'Copyright © 2024 Luminous Dynamics',
    },

    search: {
      provider: 'local',
    },

    editLink: {
      pattern: 'https://github.com/Luminous-Dynamics/mycelix-knowledge/edit/main/docs-site/:path',
      text: 'Edit this page on GitHub',
    },
  },
});
