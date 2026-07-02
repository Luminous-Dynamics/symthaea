// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import { StemMixer } from '@/components/stems';
import Link from 'next/link';
import { ArrowLeft, Download, Share2 } from 'lucide-react';
import { useState } from 'react';

export default function StemSeparationPage() {
  const [exportedBlob, setExportedBlob] = useState<Blob | null>(null);

  const handleExport = (blob: Blob, filename: string) => {
    setExportedBlob(blob);
    // Trigger download
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-900 via-black to-black">
      {/* Header */}
      <div className="sticky top-0 z-50 bg-black/80 backdrop-blur-xl border-b border-gray-800">
        <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Link
              href="/studio"
              className="flex items-center gap-2 text-gray-400 hover:text-white transition-colors"
            >
              <ArrowLeft className="w-5 h-5" />
              <span>Studio</span>
            </Link>
            <div className="w-px h-6 bg-gray-800" />
            <h1 className="text-xl font-semibold bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">
              AI Stem Separation
            </h1>
          </div>
          <div className="flex items-center gap-2">
            <button className="flex items-center gap-2 px-4 py-2 rounded-lg bg-gray-800 text-gray-400 hover:text-white transition-colors">
              <Share2 className="w-4 h-4" />
              Share
            </button>
          </div>
        </div>
      </div>

      {/* Info Banner */}
      <div className="max-w-4xl mx-auto px-4 py-6">
        <div className="rounded-xl bg-gradient-to-r from-blue-500/10 to-cyan-500/10 border border-blue-500/20 p-4">
          <p className="text-sm text-gray-300">
            <span className="font-semibold text-blue-400">AI-Powered Separation:</span>{' '}
            Upload any audio file to isolate vocals, drums, bass, and other instruments.
            Use the individual stems for remixing, karaoke, or creative production.
          </p>
        </div>
      </div>

      {/* Stem Mixer */}
      <div className="max-w-4xl mx-auto px-4 pb-8">
        <StemMixer onExport={handleExport} />
      </div>

      {/* Tips Section */}
      <div className="max-w-4xl mx-auto px-4 pb-8">
        <div className="rounded-xl bg-gray-900/50 border border-gray-800 p-6">
          <h3 className="text-lg font-semibold text-white mb-4">Tips for Best Results</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <h4 className="text-sm font-medium text-purple-400">Input Quality</h4>
              <ul className="text-sm text-gray-400 space-y-1">
                <li>• Use lossless formats (WAV, FLAC) when possible</li>
                <li>• Higher bitrate MP3s (320kbps) work well too</li>
                <li>• Avoid heavily compressed or distorted audio</li>
              </ul>
            </div>
            <div className="space-y-2">
              <h4 className="text-sm font-medium text-cyan-400">Creative Uses</h4>
              <ul className="text-sm text-gray-400 space-y-1">
                <li>• Create acapellas for remixes</li>
                <li>• Isolate drums for sampling</li>
                <li>• Make karaoke versions of songs</li>
                <li>• Study individual instrument parts</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
