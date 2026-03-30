// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { marked } from 'marked';
import logger from '@/lib/logger';

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ slug: string }> }
) {
  try {
    const { slug } = await params;
    
    // Build the path to the content file
    const filePath = path.join(process.cwd(), 'content', `${slug}.md`);
    
    // Check if file exists
    if (!fs.existsSync(filePath)) {
      return new NextResponse('Content not found', { status: 404 });
    }
    
    // Read the markdown file
    const markdown = fs.readFileSync(filePath, 'utf-8');
    
    // Convert markdown to HTML
    const html = await marked(markdown);
    
    // Return HTML response with proper styling
    const fullHtml = `
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Terra Atlas - 72% Renewable Failure Whitepaper</title>
  <style>
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
      line-height: 1.6;
      color: #333;
      max-width: 800px;
      margin: 0 auto;
      padding: 40px 20px;
      background: linear-gradient(to bottom, #f8f9fa, #ffffff);
    }
    h1 {
      color: #dc2626;
      font-size: 2.5em;
      margin-bottom: 0.5em;
      border-bottom: 3px solid #dc2626;
      padding-bottom: 10px;
    }
    h2 {
      color: #1e40af;
      font-size: 2em;
      margin-top: 1.5em;
    }
    h3 {
      color: #059669;
      font-size: 1.5em;
      margin-top: 1.2em;
    }
    strong {
      color: #dc2626;
      font-weight: 700;
    }
    ul, ol {
      margin-left: 20px;
    }
    li {
      margin: 10px 0;
    }
    pre {
      background: #f3f4f6;
      border: 1px solid #e5e7eb;
      border-radius: 8px;
      padding: 16px;
      overflow-x: auto;
    }
    code {
      background: #f3f4f6;
      padding: 2px 6px;
      border-radius: 3px;
      font-family: 'Monaco', 'Menlo', monospace;
      font-size: 0.9em;
    }
    pre code {
      background: none;
      padding: 0;
    }
    a {
      color: #2563eb;
      text-decoration: none;
    }
    a:hover {
      text-decoration: underline;
    }
    hr {
      border: none;
      border-top: 2px solid #e5e7eb;
      margin: 40px 0;
    }
    blockquote {
      border-left: 4px solid #2563eb;
      padding-left: 20px;
      margin: 20px 0;
      font-style: italic;
      color: #6b7280;
    }
    .cta-button {
      display: inline-block;
      background: linear-gradient(135deg, #dc2626, #f97316);
      color: white;
      padding: 12px 24px;
      border-radius: 8px;
      font-weight: bold;
      text-decoration: none;
      margin: 10px 5px;
      transition: transform 0.2s;
    }
    .cta-button:hover {
      transform: scale(1.05);
      text-decoration: none;
    }
  </style>
</head>
<body>
  ${html}
  <hr>
  <div style="text-align: center; margin-top: 40px;">
    <a href="/" class="cta-button">Back to Terra Atlas</a>
    <a href="/api/discovery" class="cta-button">Try the API</a>
    <a href="/west-texas-corridor" class="cta-button">See the Demo</a>
  </div>
</body>
</html>`;
    
    return new NextResponse(fullHtml, {
      status: 200,
      headers: { 'Content-Type': 'text/html' }
    });
    
  } catch (error) {
    logger.error('Error serving content', error);
    return NextResponse.json(
      { error: 'Failed to load content' },
      { status: 500 }
    );
  }
}