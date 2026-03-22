'use client';

import { useEffect, useRef } from 'react';

// Lightweight, aesthetic-only globe with Seven Harmonies theme
// No interaction, just beautiful slow rotation with energy pulses
export default function LuminousGlobe() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>();

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Setup canvas size
    const resizeCanvas = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    // Animation variables
    let rotation = 0;
    let pulsePhase = 0;
    let harmonicPhase = 0;
    
    // Seven Harmonies colors
    const harmonies = [
      { color: 'rgba(16, 185, 129, 0.4)', phase: 0 },        // Emerald - Coherence
      { color: 'rgba(59, 130, 246, 0.4)', phase: Math.PI/3.5 }, // Blue - Reciprocity  
      { color: 'rgba(139, 92, 246, 0.4)', phase: Math.PI/2.8 }, // Purple - Regeneration
      { color: 'rgba(6, 182, 212, 0.3)', phase: Math.PI/2.3 },  // Cyan - Transparency
      { color: 'rgba(251, 191, 36, 0.3)', phase: Math.PI/2 },   // Amber - Resilience
      { color: 'rgba(236, 72, 153, 0.3)', phase: Math.PI/1.8 }, // Pink - Participation
      { color: 'rgba(132, 204, 22, 0.3)', phase: Math.PI/1.5 }  // Lime - Hope
    ];

    // Particle system for sparkles
    const particles: Array<{
      x: number;
      y: number;
      size: number;
      alpha: number;
      speed: number;
      angle: number;
    }> = [];
    
    // Initialize particles
    for (let i = 0; i < 50; i++) {
      particles.push({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        size: Math.random() * 2 + 0.5,
        alpha: Math.random() * 0.5 + 0.2,
        speed: Math.random() * 0.5 + 0.1,
        angle: Math.random() * Math.PI * 2
      });
    }

    const animate = () => {
      // Clear with fade effect for trails
      ctx.fillStyle = 'rgba(0, 0, 0, 0.05)';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      const centerX = canvas.width / 2;
      const centerY = canvas.height / 2;
      const baseRadius = Math.min(canvas.width, canvas.height) * 0.25;

      // Draw outer glow aura (Infinite Love)
      const auraGradient = ctx.createRadialGradient(centerX, centerY, baseRadius * 0.8, centerX, centerY, baseRadius * 2);
      auraGradient.addColorStop(0, 'rgba(251, 191, 36, 0.0)');
      auraGradient.addColorStop(0.5, 'rgba(236, 72, 153, 0.02)');
      auraGradient.addColorStop(1, 'rgba(139, 92, 246, 0.01)');
      ctx.fillStyle = auraGradient;
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      // Draw harmonic rings (Seven Harmonies)
      harmonies.forEach((harmony, i) => {
        const pulseRadius = baseRadius * (1 + Math.sin(harmonicPhase + harmony.phase) * 0.15);
        const ringOpacity = 0.1 + Math.sin(harmonicPhase + harmony.phase) * 0.05;
        
        ctx.strokeStyle = harmony.color.replace('0.4', String(ringOpacity));
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(centerX, centerY, pulseRadius + i * 15, 0, Math.PI * 2);
        ctx.stroke();
      });

      // Draw main globe with gradient
      const globeGradient = ctx.createRadialGradient(
        centerX - baseRadius * 0.3 * Math.cos(rotation),
        centerY - baseRadius * 0.3,
        baseRadius * 0.1,
        centerX,
        centerY,
        baseRadius
      );
      
      // Rotating gradient colors
      const gradientPhase = rotation * 0.5;
      globeGradient.addColorStop(0, `rgba(16, 185, 129, ${0.4 + Math.sin(gradientPhase) * 0.1})`);
      globeGradient.addColorStop(0.3, `rgba(59, 130, 246, ${0.3 + Math.cos(gradientPhase) * 0.1})`);
      globeGradient.addColorStop(0.6, `rgba(139, 92, 246, ${0.2 + Math.sin(gradientPhase + Math.PI) * 0.1})`);
      globeGradient.addColorStop(1, 'rgba(15, 23, 42, 0.3)');

      ctx.fillStyle = globeGradient;
      ctx.beginPath();
      ctx.arc(centerX, centerY, baseRadius, 0, Math.PI * 2);
      ctx.fill();

      // Draw latitude lines (curved for 3D effect)
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.03)';
      ctx.lineWidth = 1;
      
      for (let lat = -80; lat <= 80; lat += 20) {
        ctx.beginPath();
        const y = centerY + (baseRadius * lat / 90);
        const latRadius = Math.sqrt(Math.max(0, baseRadius * baseRadius - (baseRadius * lat / 90) ** 2));
        
        // Add rotation to latitude lines
        for (let angle = 0; angle <= Math.PI * 2; angle += 0.1) {
          const x = centerX + Math.cos(angle + rotation) * latRadius * Math.cos(lat * Math.PI / 180);
          const yOffset = Math.sin(angle + rotation) * latRadius * 0.3;
          if (angle === 0) {
            ctx.moveTo(x, y + yOffset);
          } else {
            ctx.lineTo(x, y + yOffset);
          }
        }
        ctx.stroke();
      }

      // Draw longitude lines
      for (let lon = 0; lon < 360; lon += 30) {
        ctx.beginPath();
        const angle = (lon * Math.PI / 180) + rotation;
        
        for (let lat = -90; lat <= 90; lat += 5) {
          const latRad = lat * Math.PI / 180;
          const x = centerX + baseRadius * Math.cos(latRad) * Math.sin(angle);
          const y = centerY + baseRadius * Math.sin(latRad);
          const z = Math.cos(latRad) * Math.cos(angle);
          
          // Only draw if on front side of globe
          if (z > -0.3) {
            if (lat === -90) {
              ctx.moveTo(x, y);
            } else {
              ctx.lineTo(x, y);
            }
          }
        }
        ctx.stroke();
      }

      // Draw energy flow lines (data connections)
      ctx.strokeStyle = 'rgba(251, 191, 36, 0.2)';
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 10]);
      
      for (let i = 0; i < 3; i++) {
        const angle1 = rotation + (i * Math.PI * 2 / 3);
        const angle2 = rotation + ((i + 1) * Math.PI * 2 / 3);
        
        ctx.beginPath();
        const x1 = centerX + Math.cos(angle1) * baseRadius * 0.8;
        const y1 = centerY + Math.sin(angle1) * baseRadius * 0.8;
        const x2 = centerX + Math.cos(angle2) * baseRadius * 0.8;
        const y2 = centerY + Math.sin(angle2) * baseRadius * 0.8;
        
        // Create arc between points
        const cpx = centerX + Math.cos((angle1 + angle2) / 2) * baseRadius * 1.2;
        const cpy = centerY + Math.sin((angle1 + angle2) / 2) * baseRadius * 1.2;
        
        ctx.moveTo(x1, y1);
        ctx.quadraticCurveTo(cpx, cpy, x2, y2);
        ctx.stroke();
      }
      ctx.setLineDash([]);

      // Draw sparkle particles (Sacred Dissonance)
      particles.forEach(particle => {
        particle.x += Math.cos(particle.angle) * particle.speed;
        particle.y += Math.sin(particle.angle) * particle.speed;
        
        // Wrap around screen
        if (particle.x < 0) particle.x = canvas.width;
        if (particle.x > canvas.width) particle.x = 0;
        if (particle.y < 0) particle.y = canvas.height;
        if (particle.y > canvas.height) particle.y = 0;
        
        // Draw particle
        ctx.fillStyle = `rgba(255, 255, 255, ${particle.alpha * (0.5 + Math.sin(pulsePhase) * 0.5)})`;
        ctx.beginPath();
        ctx.arc(particle.x, particle.y, particle.size, 0, Math.PI * 2);
        ctx.fill();
      });

      // Draw energy nodes (project locations)
      const nodeCount = 8;
      for (let i = 0; i < nodeCount; i++) {
        const nodeAngle = (i / nodeCount) * Math.PI * 2 + rotation * 2;
        const nodeRadius = baseRadius * (0.7 + Math.sin(nodeAngle * 3) * 0.2);
        const x = centerX + Math.cos(nodeAngle) * nodeRadius;
        const y = centerY + Math.sin(nodeAngle) * nodeRadius * 0.6;
        
        // Pulsing glow
        const glowSize = 3 + Math.sin(pulsePhase + i) * 2;
        const glowGradient = ctx.createRadialGradient(x, y, 0, x, y, glowSize * 3);
        glowGradient.addColorStop(0, 'rgba(251, 191, 36, 0.8)');
        glowGradient.addColorStop(0.5, 'rgba(251, 191, 36, 0.3)');
        glowGradient.addColorStop(1, 'rgba(251, 191, 36, 0)');
        
        ctx.fillStyle = glowGradient;
        ctx.beginPath();
        ctx.arc(x, y, glowSize * 3, 0, Math.PI * 2);
        ctx.fill();
        
        // Core node
        ctx.fillStyle = 'rgba(251, 191, 36, 1)';
        ctx.beginPath();
        ctx.arc(x, y, 2, 0, Math.PI * 2);
        ctx.fill();
      }

      // Update animation phases
      rotation += 0.001; // Very slow rotation
      pulsePhase += 0.02; // Breathing rhythm
      harmonicPhase += 0.005; // Seven harmonies cycle
      
      animationRef.current = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      window.removeEventListener('resize', resizeCanvas);
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, []);

  return (
    <canvas 
      ref={canvasRef}
      className="absolute inset-0 w-full h-full"
      style={{ 
        mixBlendMode: 'screen',
        opacity: 0.8
      }}
    />
  );
}