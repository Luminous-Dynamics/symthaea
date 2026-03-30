// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import { useEffect, useRef } from 'react';

interface GlobeProps {
  projects?: any[];
}

// Energy-type color mapping for visual storytelling
const getEnergyColor = (type: string): { fill: string; glow: string } => {
  const normalizedType = type?.toLowerCase() || '';

  if (normalizedType.includes('solar')) {
    return { fill: 'rgba(251, 191, 36, 0.9)', glow: 'rgba(245, 158, 11, 0.4)' }; // Amber/Gold
  } else if (normalizedType.includes('wind')) {
    return { fill: 'rgba(6, 182, 212, 0.9)', glow: 'rgba(8, 145, 178, 0.4)' }; // Cyan/Blue
  } else if (normalizedType.includes('hydro') || normalizedType.includes('dam')) {
    return { fill: 'rgba(59, 130, 246, 0.9)', glow: 'rgba(37, 99, 235, 0.4)' }; // Deep Blue
  } else if (normalizedType.includes('nuclear') || normalizedType.includes('smr')) {
    return { fill: 'rgba(168, 85, 247, 0.9)', glow: 'rgba(147, 51, 234, 0.4)' }; // Purple
  } else if (normalizedType.includes('geothermal')) {
    return { fill: 'rgba(239, 68, 68, 0.9)', glow: 'rgba(220, 38, 38, 0.4)' }; // Red/Orange
  } else if (normalizedType.includes('battery') || normalizedType.includes('storage')) {
    return { fill: 'rgba(168, 85, 247, 0.9)', glow: 'rgba(147, 51, 234, 0.4)' }; // Purple
  } else {
    return { fill: 'rgba(156, 163, 175, 0.8)', glow: 'rgba(107, 114, 128, 0.3)' }; // Gray default
  }
};

export default function Globe({ projects = [] }: GlobeProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    // Enhanced globe with energy-type color coding
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;

    // Create a simple animated gradient sphere
    let rotation = 0;

    const animate = () => {
      ctx.fillStyle = 'rgba(0, 0, 0, 0.02)';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      const centerX = canvas.width / 2;
      const centerY = canvas.height / 2;
      const radius = Math.min(canvas.width, canvas.height) * 0.3;

      // Draw sphere with gradient
      const gradient = ctx.createRadialGradient(
        centerX - radius * 0.3,
        centerY - radius * 0.3,
        0,
        centerX,
        centerY,
        radius
      );

      gradient.addColorStop(0, 'rgba(16, 185, 129, 0.3)');  // Emerald
      gradient.addColorStop(0.5, 'rgba(59, 130, 246, 0.2)'); // Blue
      gradient.addColorStop(1, 'rgba(139, 92, 246, 0.1)');   // Purple

      ctx.fillStyle = gradient;
      ctx.beginPath();
      ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
      ctx.fill();

      // Draw grid lines for globe effect
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
      ctx.lineWidth = 1;

      // Latitude lines
      for (let i = -80; i <= 80; i += 20) {
        ctx.beginPath();
        const y = centerY + (radius * i / 90);
        const lineRadius = Math.sqrt(Math.max(0, radius * radius - (radius * i / 90) * (radius * i / 90)));
        ctx.arc(centerX, y, lineRadius, 0, Math.PI * 2);
        ctx.stroke();
      }

      // Draw project points with energy-type color coding
      if (projects.length > 0) {
        projects.slice(0, 100).forEach((project, i) => {
          const angle = (i / 100) * Math.PI * 2 + rotation;
          const x = centerX + Math.cos(angle) * radius * 0.8;
          const y = centerY + Math.sin(angle) * radius * 0.8;

          const colors = getEnergyColor(project.type);

          // Draw glow effect
          const glowGradient = ctx.createRadialGradient(x, y, 0, x, y, 6);
          glowGradient.addColorStop(0, colors.fill);
          glowGradient.addColorStop(1, colors.glow);

          ctx.fillStyle = glowGradient;
          ctx.beginPath();
          ctx.arc(x, y, 6, 0, Math.PI * 2);
          ctx.fill();

          // Draw bright center dot
          ctx.fillStyle = colors.fill;
          ctx.beginPath();
          ctx.arc(x, y, 2.5, 0, Math.PI * 2);
          ctx.fill();
        });
      }

      rotation += 0.001;
      requestAnimationFrame(animate);
    };

    animate();

    // Handle resize
    const handleResize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [projects]);

  return (
    <canvas 
      ref={canvasRef}
      className="absolute inset-0 w-full h-full"
      style={{ mixBlendMode: 'screen' }}
    />
  );
}