package io.symthaea.soma.demo

import android.animation.ValueAnimator
import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.MotionEvent
import android.view.View
import android.view.animation.LinearInterpolator
import kotlin.math.*
import kotlin.random.Random

/**
 * Ambient particle field with 5 distinct styles:
 * - STARS: scattered twinkling points (default)
 * - FIREFLIES: fewer, larger, warm, blink cycle
 * - MYCELIUM: connected lines between nearby particles, mild flocking
 * - AURORA: vertical flowing ribbons
 * - MINIMAL: very few, very dim, barely move
 *
 * Touch interaction scatters nearby particles; they slowly reconverge.
 */
class ParticleFieldView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    var consciousnessLevel: Float = 0.15f
    var harmonyColor: Int = Color.parseColor("#00E5CC")
    /** Neuromod levels [DA, NE, 5-HT, OT] for particle color tinting. */
    var neuromodLevels: FloatArray = floatArrayOf(0.5f, 0.5f, 0.5f, 0.5f)
    /** Particle style — set from avatar selection. */
    var particleStyle: ParticleStyle = ParticleStyle.STARS
    /** Secondary color for aurora ribbons and mycelium connections. */
    var secondaryColor: Int = Color.parseColor("#007A6D")
    /** Whether the system is in thinking mode — converge particles toward center. */
    var isThinking: Boolean = false
    /** Whether sleep mode is active — particles near-stop. */
    var isSleeping: Boolean = false

    private val particles = Array(80) { Particle() }
    private var touchX = -1f
    private var touchY = -1f
    private var touchActive = false
    private var longTouchActive = false

    private val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
    }
    private val linePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 1f
    }
    private val ribbonPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 8f
        strokeCap = Paint.Cap.ROUND
    }

    // Aurora ribbon state
    private val auroraRibbons = Array(7) { AuroraRibbon(it) }

    private val animator = ValueAnimator.ofFloat(0f, 1f).apply {
        duration = 16_000L
        repeatCount = ValueAnimator.INFINITE
        interpolator = LinearInterpolator()
        addUpdateListener {
            updateParticles()
            if (particleStyle == ParticleStyle.AURORA) updateAuroraRibbons()
            invalidate()
        }
    }

    override fun onAttachedToWindow() {
        super.onAttachedToWindow()
        animator.start()
    }

    override fun onDetachedFromWindow() {
        animator.cancel()
        super.onDetachedFromWindow()
    }

    override fun onSizeChanged(w: Int, h: Int, oldw: Int, oldh: Int) {
        super.onSizeChanged(w, h, oldw, oldh)
        for (p in particles) {
            p.x = Random.nextFloat() * w
            p.y = Random.nextFloat() * h
            p.vx = (Random.nextFloat() - 0.5f) * 0.5f
            p.vy = (Random.nextFloat() - 0.5f) * 0.5f
            p.size = 1.5f + Random.nextFloat() * 3f
            p.baseAlpha = 0.3f + Random.nextFloat() * 0.5f
            p.phase = Random.nextFloat() * PI.toFloat() * 2
            p.blinkTimer = Random.nextFloat() * 4f  // For FIREFLIES
        }
        // Initialize aurora ribbons
        for (ribbon in auroraRibbons) ribbon.init(w, h)
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        when (event.action) {
            MotionEvent.ACTION_DOWN, MotionEvent.ACTION_MOVE -> {
                touchX = event.x
                touchY = event.y
                touchActive = true
            }
            MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                touchActive = false
                longTouchActive = false
            }
        }
        return true
    }

    /** Set long-touch active for orbit behavior (Phase 4c). */
    fun setLongTouch(active: Boolean, x: Float = 0f, y: Float = 0f) {
        longTouchActive = active
        if (active) { touchX = x; touchY = y }
    }

    private fun updateParticles() {
        if (width == 0 || height == 0) return
        val cx = width / 2f
        val cy = height * 0.42f  // Match mandala center

        val activeCount = when (particleStyle) {
            ParticleStyle.MYCELIUM -> (20 + consciousnessLevel * 20).toInt().coerceIn(20, 40)
            ParticleStyle.FIREFLIES -> (10 + consciousnessLevel * 10).toInt().coerceIn(10, 20)
            ParticleStyle.MINIMAL -> (4 + consciousnessLevel * 4).toInt().coerceIn(4, 8)
            ParticleStyle.AURORA -> 0  // Aurora uses ribbons, not particles
            else -> (30 + consciousnessLevel * 50).toInt().coerceIn(30, particles.size)
        }

        val damping = when (particleStyle) {
            ParticleStyle.MYCELIUM -> 0.99f  // Slower, more viscous
            ParticleStyle.MINIMAL -> 0.995f
            else -> 0.98f
        }

        val sleepFactor = if (isSleeping) 0.1f else 1f

        for (i in particles.indices) {
            val p = particles[i]
            if (i >= activeCount) { p.alpha = 0f; continue }

            // Gravity toward center
            val dx = cx - p.x
            val dy = cy - p.y
            val dist = sqrt(dx * dx + dy * dy).coerceAtLeast(1f)
            // Thinking: stronger gravity = "concentrating"
            val gravity = if (isThinking) 0.06f * consciousnessLevel else 0.02f * consciousnessLevel
            p.vx += dx / dist * gravity * sleepFactor
            p.vy += dy / dist * gravity * sleepFactor

            // MYCELIUM: mild flocking toward nearest 3 neighbors
            if (particleStyle == ParticleStyle.MYCELIUM && i % 3 == 0) {
                var flockX = 0f; var flockY = 0f; var flockN = 0
                for (j in particles.indices) {
                    if (j == i || j >= activeCount) continue
                    val fdx = particles[j].x - p.x
                    val fdy = particles[j].y - p.y
                    val fd = sqrt(fdx * fdx + fdy * fdy)
                    if (fd < 150f && flockN < 3) {
                        flockX += fdx; flockY += fdy; flockN++
                    }
                }
                if (flockN > 0) {
                    p.vx += flockX / flockN * 0.002f
                    p.vy += flockY / flockN * 0.002f
                }
            }

            // Long touch: particles orbit touch point
            if (longTouchActive) {
                val odx = touchX - p.x
                val ody = touchY - p.y
                val odist = sqrt(odx * odx + ody * ody).coerceAtLeast(1f)
                if (odist < 300f) {
                    // Tangential force for orbit
                    val tangentX = -ody / odist
                    val tangentY = odx / odist
                    p.vx += tangentX * 0.3f + odx / odist * 0.05f
                    p.vy += tangentY * 0.3f + ody / odist * 0.05f
                }
            }
            // Touch repulsion (only if not long-touching)
            else if (touchActive) {
                val tdx = p.x - touchX
                val tdy = p.y - touchY
                val tdist = sqrt(tdx * tdx + tdy * tdy).coerceAtLeast(1f)
                if (tdist < 200f) {
                    val force = (200f - tdist) / 200f * 3f
                    p.vx += tdx / tdist * force
                    p.vy += tdy / tdist * force
                }
            }

            // Damping
            p.vx *= damping
            p.vy *= damping

            // Sleep: near-stop
            if (isSleeping) {
                p.vx *= 0.95f
                p.vy *= 0.95f
            }

            // Update position
            p.x += p.vx
            p.y += p.vy

            // Record trail position (ring buffer)
            p.trailX[p.trailHead] = p.x
            p.trailY[p.trailHead] = p.y
            p.trailHead = (p.trailHead + 1) % TRAIL_LENGTH
            if (p.trailHead == 0) p.trailFilled = true

            // Wrap around screen edges
            if (p.x < -20) p.x = width + 20f
            if (p.x > width + 20) p.x = -20f
            if (p.y < -20) p.y = height + 20f
            if (p.y > height + 20) p.y = -20f

            // Update alpha based on style
            when (particleStyle) {
                ParticleStyle.FIREFLIES -> {
                    // Blink cycle: on 0.5s, fade 1.5s, off 2-4s
                    p.blinkTimer += 0.016f  // ~60fps
                    val cycle = p.blinkTimer % (2f + p.baseAlpha * 2f)  // Varied cycle per particle
                    p.alpha = when {
                        cycle < 0.5f -> p.baseAlpha * (0.5f + consciousnessLevel * 0.5f)  // On
                        cycle < 2f -> p.baseAlpha * (2f - cycle) / 1.5f * (0.5f + consciousnessLevel * 0.5f)  // Fade
                        else -> 0f  // Off
                    }
                }
                ParticleStyle.MINIMAL -> {
                    p.phase += 0.005f  // Very slow
                    val twinkle = 0.5f + 0.5f * sin(p.phase)
                    p.alpha = p.baseAlpha * twinkle * 0.3f * (0.3f + consciousnessLevel * 0.3f)
                }
                else -> {
                    // STARS + MYCELIUM: standard twinkle
                    p.phase += 0.02f
                    val twinkle = 0.5f + 0.5f * sin(p.phase)
                    p.alpha = p.baseAlpha * twinkle * (0.7f + consciousnessLevel * 0.3f)
                }
            }
        }
    }

    private fun updateAuroraRibbons() {
        for (ribbon in auroraRibbons) ribbon.update(consciousnessLevel)
    }

    // Neuromod colors: DA=coral, NE=amber, 5-HT=teal, OT=indigo
    private val neuroColors = intArrayOf(
        Color.parseColor("#FF6B8A"), Color.parseColor("#FFBE4F"),
        Color.parseColor("#5EEAD4"), Color.parseColor("#818CF8"),
    )

    override fun onDraw(canvas: Canvas) {
        when (particleStyle) {
            ParticleStyle.AURORA -> drawAurora(canvas)
            else -> drawParticles(canvas)
        }
    }

    private fun drawParticles(canvas: Canvas) {
        val baseR = Color.red(harmonyColor)
        val baseG = Color.green(harmonyColor)
        val baseB = Color.blue(harmonyColor)

        // MYCELIUM: draw connecting lines first (behind particles)
        if (particleStyle == ParticleStyle.MYCELIUM) {
            val activeCount = (20 + consciousnessLevel * 20).toInt().coerceIn(20, 40)
            for (i in 0 until activeCount.coerceAtMost(particles.size)) {
                val p = particles[i]
                if (p.alpha < 0.01f) continue
                for (j in i + 1 until activeCount.coerceAtMost(particles.size)) {
                    val q = particles[j]
                    if (q.alpha < 0.01f) continue
                    val dx = p.x - q.x
                    val dy = p.y - q.y
                    val dist = sqrt(dx * dx + dy * dy)
                    if (dist < 120f) {
                        val lineAlpha = ((1f - dist / 120f) * p.alpha * 180).toInt().coerceIn(0, 120)
                        linePaint.color = Color.argb(lineAlpha, baseR, baseG, baseB)
                        canvas.drawLine(p.x, p.y, q.x, q.y, linePaint)
                    }
                }
            }
        }

        for ((i, p) in particles.withIndex()) {
            if (p.alpha < 0.01f) continue
            val a = (p.alpha * 255).toInt().coerceIn(0, 255)

            // Tint some particles with neuromod colors
            val neuroIdx = i % 4
            val neuroStrength = neuromodLevels.getOrElse(neuroIdx) { 0.5f }
            val nc = neuroColors[neuroIdx]
            val tint = (neuroStrength * 0.4f).coerceIn(0f, 0.4f)
            val r = ((1f - tint) * baseR + tint * Color.red(nc)).toInt().coerceIn(0, 255)
            val g = ((1f - tint) * baseG + tint * Color.green(nc)).toInt().coerceIn(0, 255)
            val b = ((1f - tint) * baseB + tint * Color.blue(nc)).toInt().coerceIn(0, 255)

            // FIREFLIES: larger particles, warmer colors
            val drawSize = when (particleStyle) {
                ParticleStyle.FIREFLIES -> p.size * 2f
                ParticleStyle.MYCELIUM -> p.size * 1.5f
                ParticleStyle.MINIMAL -> p.size * 0.8f
                else -> p.size
            }

            // Trail afterimages at consciousness > 0.5 (not MINIMAL or FIREFLIES)
            if (consciousnessLevel > 0.5f && particleStyle != ParticleStyle.MINIMAL
                && particleStyle != ParticleStyle.FIREFLIES && (p.trailFilled || p.trailHead > 1)
            ) {
                val trailCount = if (p.trailFilled) TRAIL_LENGTH else p.trailHead
                val trailFade = (consciousnessLevel - 0.5f) * 2f  // 0..1 at 0.5..1.0 consciousness
                for (ti in 0 until trailCount) {
                    val idx = ((p.trailHead - 1 - ti) % TRAIL_LENGTH + TRAIL_LENGTH) % TRAIL_LENGTH
                    val age = (ti + 1).toFloat() / TRAIL_LENGTH
                    val trailAlpha = (a * (1f - age) * 0.4f * trailFade).toInt().coerceIn(0, 60)
                    if (trailAlpha > 2) {
                        paint.color = Color.argb(trailAlpha, r, g, b)
                        canvas.drawCircle(p.trailX[idx], p.trailY[idx], drawSize * (1f - age * 0.3f), paint)
                    }
                }
            }

            // Soft glow halo
            val haloA = (a * 0.3f).toInt().coerceIn(0, 80)
            paint.color = Color.argb(haloA, r, g, b)
            canvas.drawCircle(p.x, p.y, drawSize * 3f, paint)
            // Bright core
            paint.color = Color.argb(a, r, g, b)
            canvas.drawCircle(p.x, p.y, drawSize, paint)
        }
    }

    /** Draw aurora-style vertical flowing ribbons using quadratic bezier curves. */
    private fun drawAurora(canvas: Canvas) {
        val primaryR = Color.red(harmonyColor)
        val primaryG = Color.green(harmonyColor)
        val primaryB = Color.blue(harmonyColor)
        val secR = Color.red(secondaryColor)
        val secG = Color.green(secondaryColor)
        val secB = Color.blue(secondaryColor)
        val baseAlpha = (60 + consciousnessLevel * 100).toInt().coerceIn(40, 160)

        for (ribbon in auroraRibbons) {
            val path = Path()
            val blend = ribbon.colorBlend
            val r = ((1f - blend) * primaryR + blend * secR).toInt().coerceIn(0, 255)
            val g = ((1f - blend) * primaryG + blend * secG).toInt().coerceIn(0, 255)
            val b = ((1f - blend) * primaryB + blend * secB).toInt().coerceIn(0, 255)

            ribbonPaint.color = Color.argb(
                (baseAlpha * ribbon.alphaFactor).toInt().coerceIn(10, 160), r, g, b
            )
            ribbonPaint.strokeWidth = (6f + consciousnessLevel * 6f) * ribbon.widthFactor

            val pts = ribbon.points
            if (pts.size < 2) continue

            // Smooth bezier path through control points for organic ribbon feel
            path.moveTo(pts[0].x, pts[0].y)
            if (pts.size == 2) {
                path.lineTo(pts[1].x, pts[1].y)
            } else {
                // First segment: quadratic from start to midpoint of first two
                path.quadTo(
                    pts[0].x, pts[0].y,
                    (pts[0].x + pts[1].x) / 2f, (pts[0].y + pts[1].y) / 2f
                )
                // Middle segments: smooth quadratic through midpoints
                for (i in 1 until pts.size - 1) {
                    val midX = (pts[i].x + pts[i + 1].x) / 2f
                    val midY = (pts[i].y + pts[i + 1].y) / 2f
                    path.quadTo(pts[i].x, pts[i].y, midX, midY)
                }
                // Last segment: end at final point
                val last = pts.size - 1
                path.quadTo(
                    pts[last].x, pts[last].y,
                    pts[last].x, pts[last].y
                )
            }
            canvas.drawPath(path, ribbonPaint)

            // Draw a slightly offset ghost ribbon for width/depth illusion
            val ghostPath = Path()
            val offset = ribbonPaint.strokeWidth * 0.4f
            ghostPath.moveTo(pts[0].x + offset, pts[0].y)
            if (pts.size > 2) {
                ghostPath.quadTo(
                    pts[0].x + offset, pts[0].y,
                    (pts[0].x + pts[1].x) / 2f + offset, (pts[0].y + pts[1].y) / 2f
                )
                for (i in 1 until pts.size - 1) {
                    val midX = (pts[i].x + pts[i + 1].x) / 2f + offset * sin(i.toFloat() * 0.5f)
                    val midY = (pts[i].y + pts[i + 1].y) / 2f
                    ghostPath.quadTo(pts[i].x + offset, pts[i].y, midX, midY)
                }
                val last = pts.size - 1
                ghostPath.quadTo(pts[last].x + offset, pts[last].y, pts[last].x + offset, pts[last].y)
            }
            val ghostAlpha = (baseAlpha * ribbon.alphaFactor * 0.3f).toInt().coerceIn(5, 60)
            ribbonPaint.color = Color.argb(ghostAlpha, r, g, b)
            canvas.drawPath(ghostPath, ribbonPaint)
        }
    }

    private class Particle {
        var x = 0f; var y = 0f
        var vx = 0f; var vy = 0f
        var size = 1.5f
        var alpha = 0f; var baseAlpha = 0.2f
        var phase = 0f
        var blinkTimer = 0f  // For FIREFLIES
        // Trail: ring buffer of previous positions for afterimage effect
        val trailX = FloatArray(TRAIL_LENGTH)
        val trailY = FloatArray(TRAIL_LENGTH)
        var trailHead = 0
        var trailFilled = false
    }

    companion object {
        const val TRAIL_LENGTH = 6  // Number of afterimage positions
    }

    /**
     * Aurora ribbon: a vertical flowing band of light with organic movement.
     *
     * Uses 8 control points with multiple harmonic oscillations for natural,
     * curtain-like aurora behavior. Bezier curves in drawAurora() smooth
     * between these control points.
     */
    private class AuroraRibbon(private val index: Int) {
        val points = Array(8) { PointF() }  // 8 control points for smooth bezier
        var baseX = 0f
        var phase = index * 0.9f
        val colorBlend = (index % 3) / 2f
        val alphaFactor = 0.5f + (index % 4) * 0.15f
        val widthFactor = 0.7f + (index % 3) * 0.3f
        private val speed = 0.008f + index * 0.002f
        private val amplitude = 40f + index * 12f
        // Each ribbon has unique harmonic ratios for non-repetitive motion
        private val harmonic1 = 1.0f + index * 0.13f
        private val harmonic2 = 0.5f + index * 0.07f

        fun init(w: Int, h: Int) {
            baseX = w * (index + 1f) / 8f
            for (i in points.indices) {
                points[i].x = baseX
                points[i].y = h * i.toFloat() / (points.size - 1)
            }
        }

        fun update(consciousness: Float) {
            phase += speed
            val amp = amplitude * (0.5f + consciousness * 0.5f)
            for (i in points.indices) {
                val t = i.toFloat() / (points.size - 1)
                // Multiple harmonics create organic, curtain-like sway
                val wave1 = sin(phase * harmonic1 + t * 2.5f) * amp
                val wave2 = sin(phase * harmonic2 + t * 4f) * amp * 0.3f
                val wave3 = cos(phase * 0.6f + t * 1.2f) * amp * 0.15f
                points[i].x = baseX + wave1 + wave2 + wave3
            }
        }
    }
}
