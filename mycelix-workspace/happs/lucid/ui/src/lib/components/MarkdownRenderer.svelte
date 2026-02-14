<script lang="ts">
  import { marked, type Renderer } from 'marked';
  import hljs from 'highlight.js';
  import katex from 'katex';

  export let content: string;
  export let inline: boolean = false;

  let renderedHtml = '';

  // Custom renderer with syntax highlighting
  const renderer: Partial<Renderer> = {
    code({ text, lang }: { text: string; lang?: string; escaped?: boolean }): string {
      if (lang && hljs.getLanguage(lang)) {
        try {
          const highlighted = hljs.highlight(text, { language: lang }).value;
          return `<pre><code class="hljs language-${lang}">${highlighted}</code></pre>`;
        } catch (err) {
          console.error('Highlight error:', err);
        }
      }
      const autoHighlighted = hljs.highlightAuto(text).value;
      return `<pre><code class="hljs">${autoHighlighted}</code></pre>`;
    }
  };

  // Configure marked
  marked.setOptions({
    breaks: true,
    gfm: true,
  });

  marked.use({ renderer });

  // Custom renderer to handle LaTeX
  function renderLatex(text: string): string {
    // Block LaTeX: $$...$$
    text = text.replace(/\$\$([\s\S]+?)\$\$/g, (match, latex) => {
      try {
        return katex.renderToString(latex.trim(), {
          displayMode: true,
          throwOnError: false,
        });
      } catch (e) {
        return `<span class="latex-error">${match}</span>`;
      }
    });

    // Inline LaTeX: $...$
    text = text.replace(/\$([^\$\n]+?)\$/g, (match, latex) => {
      try {
        return katex.renderToString(latex.trim(), {
          displayMode: false,
          throwOnError: false,
        });
      } catch (e) {
        return `<span class="latex-error">${match}</span>`;
      }
    });

    return text;
  }

  // Render content
  $: {
    if (content) {
      // First process LaTeX, then markdown
      const withLatex = renderLatex(content);
      if (inline) {
        renderedHtml = marked.parseInline(withLatex) as string;
      } else {
        renderedHtml = marked.parse(withLatex) as string;
      }
    } else {
      renderedHtml = '';
    }
  }
</script>

<svelte:head>
  <link
    rel="stylesheet"
    href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css"
  />
  <link
    rel="stylesheet"
    href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css"
  />
</svelte:head>

<div class="markdown-content" class:inline>
  {@html renderedHtml}
</div>

<style>
  .markdown-content {
    line-height: 1.6;
    color: #e5e5e5;
  }

  .markdown-content.inline {
    display: inline;
  }

  .markdown-content :global(h1),
  .markdown-content :global(h2),
  .markdown-content :global(h3),
  .markdown-content :global(h4) {
    color: #fff;
    margin-top: 1.5em;
    margin-bottom: 0.5em;
  }

  .markdown-content :global(h1) { font-size: 1.5rem; }
  .markdown-content :global(h2) { font-size: 1.3rem; }
  .markdown-content :global(h3) { font-size: 1.1rem; }

  .markdown-content :global(p) {
    margin: 0.75em 0;
  }

  .markdown-content :global(a) {
    color: #7c3aed;
    text-decoration: none;
  }

  .markdown-content :global(a:hover) {
    text-decoration: underline;
  }

  .markdown-content :global(code) {
    background: #252540;
    padding: 2px 6px;
    border-radius: 4px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 0.9em;
  }

  .markdown-content :global(pre) {
    background: #1a1a2e;
    border: 1px solid #2a2a4e;
    border-radius: 8px;
    padding: 16px;
    overflow-x: auto;
    margin: 1em 0;
  }

  .markdown-content :global(pre code) {
    background: none;
    padding: 0;
    font-size: 0.85rem;
    line-height: 1.5;
  }

  .markdown-content :global(blockquote) {
    border-left: 4px solid #7c3aed;
    margin: 1em 0;
    padding: 0.5em 1em;
    background: rgba(124, 58, 237, 0.1);
    border-radius: 0 8px 8px 0;
  }

  .markdown-content :global(blockquote p) {
    margin: 0;
  }

  .markdown-content :global(ul),
  .markdown-content :global(ol) {
    margin: 0.75em 0;
    padding-left: 1.5em;
  }

  .markdown-content :global(li) {
    margin: 0.25em 0;
  }

  .markdown-content :global(table) {
    width: 100%;
    border-collapse: collapse;
    margin: 1em 0;
  }

  .markdown-content :global(th),
  .markdown-content :global(td) {
    border: 1px solid #2a2a4e;
    padding: 8px 12px;
    text-align: left;
  }

  .markdown-content :global(th) {
    background: #1a1a2e;
    color: #fff;
  }

  .markdown-content :global(tr:nth-child(even)) {
    background: rgba(42, 42, 78, 0.3);
  }

  .markdown-content :global(hr) {
    border: none;
    border-top: 1px solid #2a2a4e;
    margin: 1.5em 0;
  }

  .markdown-content :global(img) {
    max-width: 100%;
    border-radius: 8px;
  }

  /* KaTeX styles */
  .markdown-content :global(.katex-display) {
    margin: 1em 0;
    overflow-x: auto;
    padding: 8px 0;
  }

  .markdown-content :global(.latex-error) {
    color: #ef4444;
    background: rgba(239, 68, 68, 0.1);
    padding: 2px 6px;
    border-radius: 4px;
    font-family: monospace;
  }

  /* Task lists */
  .markdown-content :global(input[type="checkbox"]) {
    margin-right: 8px;
    accent-color: #7c3aed;
  }
</style>
