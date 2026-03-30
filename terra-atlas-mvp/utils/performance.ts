// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Performance Monitoring Utilities for Terra Atlas
 *
 * Tracks component load times, rendering performance, and user experience metrics.
 * Used to identify bottlenecks and optimize the globe visualization.
 */

export interface PerformanceMetrics {
  globeMountStart?: number
  globeMountEnd?: number
  threeJsInitStart?: number
  threeJsInitEnd?: number
  markersRenderStart?: number
  markersRenderEnd?: number
  totalLoadTime?: number
  fps?: number[]
  memoryUsage?: number
}

class PerformanceMonitor {
  private metrics: PerformanceMetrics = {}
  private fpsHistory: number[] = []
  private lastFrameTime: number = 0
  private frameCount: number = 0
  private telemetrySent = false
  private telemetryMeta: Record<string, unknown> = {
    component: 'TerraGlobeWithSites',
  }

  /**
   * Mark the start of globe component mounting
   */
  markGlobeMountStart(): void {
    this.metrics.globeMountStart = performance.now()
    console.log('⏱️  [Performance] Globe mount started')
  }

  /**
   * Mark the end of globe component mounting
   */
  markGlobeMountEnd(): void {
    this.metrics.globeMountEnd = performance.now()
    const duration = this.getMountDuration()
    console.log(`✅ [Performance] Globe mounted in ${duration.toFixed(2)}ms`)
  }

  /**
   * Mark the start of Three.js initialization
   */
  markThreeJsInitStart(): void {
    this.metrics.threeJsInitStart = performance.now()
    console.log('⏱️  [Performance] Three.js init started')
  }

  /**
   * Mark the end of Three.js initialization
   */
  markThreeJsInitEnd(): void {
    this.metrics.threeJsInitEnd = performance.now()
    const duration = this.getThreeJsInitDuration()
    console.log(`✅ [Performance] Three.js initialized in ${duration.toFixed(2)}ms`)
  }

  /**
   * Mark the start of markers rendering
   */
  markMarkersRenderStart(): void {
    this.metrics.markersRenderStart = performance.now()
    console.log('⏱️  [Performance] Markers rendering started')
  }

  /**
   * Mark the end of markers rendering
   */
  markMarkersRenderEnd(): void {
    this.metrics.markersRenderEnd = performance.now()
    const duration = this.getMarkersRenderDuration()
    console.log(`✅ [Performance] Markers rendered in ${duration.toFixed(2)}ms`)

    // Calculate total load time
    if (this.metrics.globeMountStart) {
      this.metrics.totalLoadTime = this.metrics.markersRenderEnd - this.metrics.globeMountStart
      console.log(`🎉 [Performance] TOTAL GLOBE LOAD TIME: ${this.metrics.totalLoadTime.toFixed(2)}ms`)
    }

    this.sendTelemetry()
  }

  /**
   * Track FPS during animation
   */
  trackFrame(): void {
    const now = performance.now()

    if (this.lastFrameTime > 0) {
      const delta = now - this.lastFrameTime
      const fps = 1000 / delta
      this.fpsHistory.push(fps)

      // Keep only last 60 frames
      if (this.fpsHistory.length > 60) {
        this.fpsHistory.shift()
      }
    }

    this.lastFrameTime = now
    this.frameCount++

    // Log FPS every 60 frames
    if (this.frameCount % 60 === 0) {
      const avgFps = this.getAverageFPS()
      console.log(`📊 [Performance] Average FPS: ${avgFps.toFixed(1)}`)
    }
  }

  /**
   * Get mount duration in milliseconds
   */
  getMountDuration(): number {
    if (!this.metrics.globeMountStart || !this.metrics.globeMountEnd) {
      return 0
    }
    return this.metrics.globeMountEnd - this.metrics.globeMountStart
  }

  /**
   * Get Three.js initialization duration
   */
  getThreeJsInitDuration(): number {
    if (!this.metrics.threeJsInitStart || !this.metrics.threeJsInitEnd) {
      return 0
    }
    return this.metrics.threeJsInitEnd - this.metrics.threeJsInitStart
  }

  /**
   * Get markers rendering duration
   */
  getMarkersRenderDuration(): number {
    if (!this.metrics.markersRenderStart || !this.metrics.markersRenderEnd) {
      return 0
    }
    return this.metrics.markersRenderEnd - this.metrics.markersRenderStart
  }

  /**
   * Get total load time
   */
  getTotalLoadTime(): number {
    return this.metrics.totalLoadTime || 0
  }

  /**
   * Get average FPS from recent frames
   */
  getAverageFPS(): number {
    if (this.fpsHistory.length === 0) return 0
    const sum = this.fpsHistory.reduce((a, b) => a + b, 0)
    return sum / this.fpsHistory.length
  }

  /**
   * Get memory usage (if available)
   */
  getMemoryUsage(): number | null {
    if ('memory' in performance) {
      const memory = (performance as any).memory
      return memory.usedJSHeapSize / 1024 / 1024 // MB
    }
    return null
  }

  /**
   * Generate a comprehensive performance report
   */
  generateReport(): {
    summary: string
    metrics: PerformanceMetrics
    analysis: string[]
  } {
    const mountDuration = this.getMountDuration()
    const threeJsInitDuration = this.getThreeJsInitDuration()
    const markersRenderDuration = this.getMarkersRenderDuration()
    const totalLoadTime = this.getTotalLoadTime()
    const avgFps = this.getAverageFPS()
    const memoryUsage = this.getMemoryUsage()

    const analysis: string[] = []

    // Analyze mount time
    if (mountDuration > 100) {
      analysis.push(`⚠️  Mount time is high (${mountDuration.toFixed(0)}ms) - consider lazy loading`)
    } else if (mountDuration > 0) {
      analysis.push(`✅ Mount time is good (${mountDuration.toFixed(0)}ms)`)
    }

    // Analyze Three.js init
    if (threeJsInitDuration > 500) {
      analysis.push(`⚠️  Three.js init is slow (${threeJsInitDuration.toFixed(0)}ms) - consider optimizing scene setup`)
    } else if (threeJsInitDuration > 0) {
      analysis.push(`✅ Three.js init is efficient (${threeJsInitDuration.toFixed(0)}ms)`)
    }

    // Analyze markers rendering
    if (markersRenderDuration > 2000) {
      analysis.push(`🔴 Markers rendering is very slow (${markersRenderDuration.toFixed(0)}ms) - CRITICAL`)
      analysis.push(`   Consider: instanced rendering, LOD, or reducing marker count`)
    } else if (markersRenderDuration > 1000) {
      analysis.push(`⚠️  Markers rendering is slow (${markersRenderDuration.toFixed(0)}ms)`)
      analysis.push(`   Consider: batching, geometry merging, or simpler materials`)
    } else if (markersRenderDuration > 0) {
      analysis.push(`✅ Markers rendering is good (${markersRenderDuration.toFixed(0)}ms)`)
    }

    // Analyze total load time
    if (totalLoadTime > 6000) {
      analysis.push(`🔴 TOTAL LOAD TIME IS CRITICAL: ${(totalLoadTime / 1000).toFixed(1)}s`)
      analysis.push(`   TARGET: <2s | CURRENT: ${(totalLoadTime / 1000).toFixed(1)}s`)
    } else if (totalLoadTime > 2000) {
      analysis.push(`⚠️  Total load time exceeds target: ${(totalLoadTime / 1000).toFixed(1)}s (target: 2s)`)
    } else if (totalLoadTime > 0) {
      analysis.push(`🎉 Total load time meets target: ${(totalLoadTime / 1000).toFixed(1)}s`)
    }

    // Analyze FPS
    if (avgFps > 0) {
      if (avgFps < 30) {
        analysis.push(`🔴 FPS is too low (${avgFps.toFixed(0)}) - users will notice lag`)
      } else if (avgFps < 50) {
        analysis.push(`⚠️  FPS could be better (${avgFps.toFixed(0)}) - aim for 60`)
      } else {
        analysis.push(`✅ FPS is excellent (${avgFps.toFixed(0)})`)
      }
    }

    // Analyze memory
    if (memoryUsage !== null) {
      if (memoryUsage > 200) {
        analysis.push(`⚠️  High memory usage (${memoryUsage.toFixed(0)}MB) - check for leaks`)
      } else {
        analysis.push(`✅ Memory usage is reasonable (${memoryUsage.toFixed(0)}MB)`)
      }
    }

    const summary = `
╔═══════════════════════════════════════════════════════════╗
║           TERRA ATLAS GLOBE PERFORMANCE REPORT            ║
╚═══════════════════════════════════════════════════════════╝

📊 TIMINGS:
   • Component Mount:     ${mountDuration.toFixed(2)}ms
   • Three.js Init:       ${threeJsInitDuration.toFixed(2)}ms
   • Markers Rendering:   ${markersRenderDuration.toFixed(2)}ms
   • ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   • TOTAL LOAD TIME:     ${totalLoadTime.toFixed(2)}ms (${(totalLoadTime / 1000).toFixed(2)}s)

📈 RUNTIME:
   • Average FPS:         ${avgFps > 0 ? avgFps.toFixed(1) : 'N/A'}
   • Memory Usage:        ${memoryUsage !== null ? memoryUsage.toFixed(1) + 'MB' : 'N/A'}

🎯 TARGETS:
   • Total Load Time:     <2000ms (currently ${totalLoadTime > 2000 ? 'OVER' : 'UNDER'} target)
   • Target FPS:          60 (currently ${avgFps >= 60 ? 'MET' : avgFps > 0 ? 'BELOW' : 'N/A'})
    `.trim()

    return {
      summary,
      metrics: {
        ...this.metrics,
        fps: this.fpsHistory,
        memoryUsage: memoryUsage || undefined
      },
      analysis
    }
  }

  /**
   * Update contextual metadata that will be sent with telemetry events
   */
  setContext(meta: Record<string, unknown>): void {
    this.telemetryMeta = {
      ...this.telemetryMeta,
      ...meta,
    }
  }

  private buildTelemetrySnapshot() {
    return {
      mountDurationMs: this.getMountDuration(),
      threeJsInitDurationMs: this.getThreeJsInitDuration(),
      markersRenderDurationMs: this.getMarkersRenderDuration(),
      totalLoadTimeMs: this.getTotalLoadTime(),
      averageFps: this.getAverageFPS(),
      memoryUsageMb: this.getMemoryUsage(),
      timestamp: new Date().toISOString(),
    }
  }

  private sendTelemetry(): void {
    if (this.telemetrySent || typeof window === 'undefined') {
      return
    }

    this.telemetrySent = true
    const payload = {
      type: 'globe_performance',
      metrics: this.buildTelemetrySnapshot(),
      meta: this.telemetryMeta,
    }
    const body = JSON.stringify(payload)

    try {
      if (typeof navigator !== 'undefined' && typeof navigator.sendBeacon === 'function') {
        const blob = new Blob([body], { type: 'application/json' })
        navigator.sendBeacon('/api/telemetry', blob)
      } else {
        fetch('/api/telemetry', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body,
          keepalive: true,
        }).catch((error) => console.error('Telemetry fetch error:', error))
      }
    } catch (error) {
      console.error('Failed to send telemetry:', error)
    }
  }

  /**
   * Log the performance report to console
   */
  logReport(): void {
    const report = this.generateReport()
    console.log(report.summary)
    console.log('\n💡 ANALYSIS:')
    report.analysis.forEach(line => console.log('   ' + line))
  }

  /**
   * Reset all metrics
   */
  reset(): void {
    this.metrics = {}
    this.fpsHistory = []
    this.lastFrameTime = 0
    this.frameCount = 0
  }
}

// Export singleton instance
export const performanceMonitor = new PerformanceMonitor()

// Export for type imports
export type { PerformanceMonitor }
