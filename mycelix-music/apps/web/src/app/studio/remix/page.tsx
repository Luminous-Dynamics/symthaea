// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import React from 'react';
import { RemixStudio } from '@/components/remix';

export default function RemixPage() {
  return (
    <div className="h-screen bg-gray-950 flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between px-6 py-4 border-b border-gray-800">
        <div>
          <h1 className="text-2xl font-bold text-white">Collaborative Remix Studio</h1>
          <p className="text-gray-400 text-sm">
            Create remixes together in real-time with version control
          </p>
        </div>
        <div className="flex items-center gap-4">
          <button className="px-4 py-2 bg-gray-800 text-gray-300 rounded-lg hover:bg-gray-700 transition-colors">
            New Project
          </button>
          <button className="px-4 py-2 bg-purple-500 text-white rounded-lg hover:bg-purple-600 transition-colors">
            Invite Collaborator
          </button>
        </div>
      </div>

      {/* Studio */}
      <div className="flex-1">
        <RemixStudio />
      </div>
    </div>
  );
}
