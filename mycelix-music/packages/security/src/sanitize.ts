// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Input Sanitization
 *
 * Utilities for sanitizing user input.
 */

/**
 * Escape HTML special characters
 */
export function escapeHtml(str: string): string {
  const htmlEntities: Record<string, string> = {
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;',
    '/': '&#x2F;',
    '`': '&#x60;',
    '=': '&#x3D;',
  };

  return str.replace(/[&<>"'`=/]/g, (char) => htmlEntities[char]);
}

/**
 * Sanitize user input by removing potentially dangerous content
 */
export function sanitizeInput(input: string): string {
  return input
    // Remove null bytes
    .replace(/\0/g, '')
    // Remove control characters except newlines and tabs
    .replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g, '')
    // Normalize whitespace
    .replace(/\s+/g, ' ')
    .trim();
}

/**
 * Sanitize a filename to prevent path traversal
 */
export function sanitizeFilename(filename: string): string {
  return (
    filename
      // Remove path separators
      .replace(/[/\\]/g, '')
      // Remove null bytes
      .replace(/\0/g, '')
      // Remove leading dots (hidden files on Unix)
      .replace(/^\.+/, '')
      // Replace spaces with underscores
      .replace(/\s+/g, '_')
      // Remove special characters
      .replace(/[^a-zA-Z0-9_.-]/g, '')
      // Limit length
      .slice(0, 255)
  );
}

/**
 * Sanitize a URL to prevent open redirects
 */
export function sanitizeRedirectUrl(url: string, allowedHosts: string[]): string | null {
  try {
    const parsed = new URL(url);

    // Check if host is allowed
    if (!allowedHosts.includes(parsed.host)) {
      return null;
    }

    // Only allow http and https
    if (!['http:', 'https:'].includes(parsed.protocol)) {
      return null;
    }

    return parsed.href;
  } catch {
    // Invalid URL, check if it's a relative path
    if (url.startsWith('/') && !url.startsWith('//')) {
      return url;
    }
    return null;
  }
}

/**
 * Strip dangerous HTML tags (basic XSS prevention)
 */
export function stripDangerousTags(html: string): string {
  const dangerousTags = [
    'script',
    'iframe',
    'object',
    'embed',
    'form',
    'input',
    'button',
    'link',
    'meta',
    'style',
    'base',
  ];

  let result = html;

  for (const tag of dangerousTags) {
    // Remove opening tags
    result = result.replace(
      new RegExp(`<${tag}[^>]*>`, 'gi'),
      ''
    );
    // Remove closing tags
    result = result.replace(
      new RegExp(`</${tag}>`, 'gi'),
      ''
    );
  }

  // Remove event handlers
  result = result.replace(/\s*on\w+\s*=\s*["'][^"']*["']/gi, '');
  result = result.replace(/\s*on\w+\s*=\s*[^\s>]+/gi, '');

  // Remove javascript: URLs
  result = result.replace(/javascript:/gi, '');

  // Remove data: URLs in src attributes
  result = result.replace(/src\s*=\s*["']?data:/gi, 'src="');

  return result;
}

/**
 * Sanitize markdown content
 */
export function sanitizeMarkdown(markdown: string): string {
  return (
    markdown
      // Remove HTML tags
      .replace(/<[^>]+>/g, '')
      // Remove javascript: links
      .replace(/\[([^\]]+)\]\(javascript:[^)]*\)/gi, '$1')
      // Remove data: links
      .replace(/\[([^\]]+)\]\(data:[^)]*\)/gi, '$1')
      // Normalize whitespace
      .replace(/\s+/g, ' ')
      .trim()
  );
}

/**
 * Validate and sanitize an Ethereum address
 */
export function sanitizeEthereumAddress(address: string): string | null {
  // Remove whitespace
  const cleaned = address.trim();

  // Check format
  if (!/^0x[a-fA-F0-9]{40}$/.test(cleaned)) {
    return null;
  }

  // Return checksummed address (in production, use ethers.js or viem)
  return cleaned.toLowerCase();
}

/**
 * Sanitize SQL-like input (for additional protection alongside parameterized queries)
 */
export function sanitizeSqlInput(input: string): string {
  return input
    .replace(/'/g, "''")
    .replace(/;/g, '')
    .replace(/--/g, '')
    .replace(/\/\*/g, '')
    .replace(/\*\//g, '');
}
