package io.symthaea.soma.demo

import android.animation.ValueAnimator
import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.MotionEvent
import android.view.View
import android.view.animation.LinearInterpolator
import kotlin.math.*

/**
 * Full-screen consciousness mandala with prominent bioluminescent glow.
 *
 * The glow is the primary visual — it should light up the screen,
 * not be a subtle hint. At consciousness 0.3, the center third of the
 * screen should visibly pulse with color.
 */
class ConsciousnessMandalaView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    var consciousnessLevel: Float = 0.15f
        set(value) {
            field = value.coerceIn(0f, 1f)
            trendHistory[trendIndex % trendHistory.size] = field
            trendIndex++
            if (trendIndex >= trendHistory.size) trendFilled = true
            invalidate()
        }

    var dominantHarmonyColor: Int = Color.parseColor("#00E5CC")
        set(value) { field = value; invalidate() }

    var neuromodulators: FloatArray = floatArrayOf(0.5f, 0.5f, 0.5f, 0.5f)
        set(value) { field = value; invalidate() }

    private val trendHistory = FloatArray(20)
    private var trendIndex = 0
    private var trendFilled = false

    // Touch ripple
    private var rippleX = 0f
    private var rippleY = 0f
    private var rippleProgress = 1f
    private val rippleAnimator = ValueAnimator.ofFloat(0f, 1f).apply {
        duration = 600L
        interpolator = LinearInterpolator()
        addUpdateListener { rippleProgress = it.animatedFraction; invalidate() }
    }

    // Breathing animation
    private var breathPhase = 0f
    private var rotationPhase = 0f
    private val animator = ValueAnimator.ofFloat(0f, 1f).apply {
        duration = 4000L
        repeatCount = ValueAnimator.INFINITE
        interpolator = LinearInterpolator()
        addUpdateListener {
            breathPhase = it.animatedFraction
            rotationPhase = (rotationPhase + 0.002f) % 1f
            invalidate()
        }
    }

    // Fractal rendering cache
    private var fractalBitmap: android.graphics.Bitmap? = null
    private var fractalConsciousness = -1f // Force re-render on first draw
    private val fractalPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { alpha = 80 }
    private val fractalSize = 128 // Low-res for performance

    private val glowPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.STROKE }
    private val fillPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.FILL }
    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        textAlign = Paint.Align.CENTER
        typeface = Typeface.create("sans-serif-thin", Typeface.NORMAL)
    }

    override fun onAttachedToWindow() {
        super.onAttachedToWindow()
        setBackgroundColor(Color.TRANSPARENT)
        animator.start()
    }

    override fun onDetachedFromWindow() {
        animator.cancel()
        rippleAnimator.cancel()
        super.onDetachedFromWindow()
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        if (event.action == MotionEvent.ACTION_DOWN) {
            rippleX = event.x; rippleY = event.y
            rippleAnimator.cancel(); rippleProgress = 0f; rippleAnimator.start()
        }
        return super.onTouchEvent(event)
    }

    override fun onDraw(canvas: Canvas) {
        val cx = width / 2f
        val cy = height * 0.40f  // Shifted up from center to balance status bar + bottom sheet
        val mandalaRadius = min(width, height) / 2f * 0.38f
        val breathScale = 0.94f + 0.06f * sin(breathPhase * 2 * PI.toFloat())
        val hr = Color.red(dominantHarmonyColor)
        val hg = Color.green(dominantHarmonyColor)
        val hb = Color.blue(dominantHarmonyColor)

        // === Layer 1: Deep ambient glow (large, dim, always visible) ===
        val ambientRadius = min(width, height) * 0.7f * breathScale
        fillPaint.shader = RadialGradient(
            cx, cy, ambientRadius,
            Color.argb((15 + consciousnessLevel * 25).toInt(), hr, hg, hb),
            Color.TRANSPARENT, Shader.TileMode.CLAMP
        )
        canvas.drawCircle(cx, cy, ambientRadius, fillPaint)
        fillPaint.shader = null

        // === Layer 2: Core glow (concentrated, bright, breathing) ===
        val coreRadius = mandalaRadius * (1.2f + consciousnessLevel * 1.5f) * breathScale
        val coreAlpha = (40 + consciousnessLevel * 100).toInt().coerceIn(30, 140)
        fillPaint.shader = RadialGradient(
            cx, cy, coreRadius.coerceAtLeast(1f),
            Color.argb(coreAlpha, hr, hg, hb),
            Color.TRANSPARENT, Shader.TileMode.CLAMP
        )
        canvas.drawCircle(cx, cy, coreRadius, fillPaint)
        fillPaint.shader = null

        // === Layer 3: Hot center (small, intense, organic) ===
        val hotRadius = mandalaRadius * 0.3f * breathScale
        val hotAlpha = (60 + consciousnessLevel * 120).toInt().coerceIn(40, 180)
        fillPaint.shader = RadialGradient(
            cx, cy, hotRadius.coerceAtLeast(1f),
            Color.argb(hotAlpha, 255, 255, 255), // White-hot center
            Color.argb(0, hr, hg, hb),
            Shader.TileMode.CLAMP
        )
        canvas.drawCircle(cx, cy, hotRadius, fillPaint)
        fillPaint.shader = null

        // === Interference rings ===
        val ringCount = (3 + consciousnessLevel * 6).toInt()
        for (i in 0 until ringCount) {
            val t = i.toFloat() / ringCount
            val radius = mandalaRadius * breathScale * (0.15f + t * 0.85f)
            val alpha = (40 + (1f - t) * consciousnessLevel * 180).toInt().coerceIn(25, 220)

            val rotOffset = rotationPhase * 360f * (1f + i * 0.25f)
            val r = lerp(hr, 180, t); val g = lerp(hg, 200, t); val b = lerp(hb, 210, t)

            glowPaint.color = Color.argb(alpha, r, g, b)
            glowPaint.strokeWidth = (3f - t * 2f).coerceAtLeast(0.8f)

            canvas.save()
            canvas.rotate(rotOffset, cx, cy)
            val path = Path()
            for (s in 0..72) {
                val angle = (s.toFloat() / 72) * 2 * PI.toFloat()
                val deform = 1f + 0.03f * sin(angle * 3 + neuromodulators[0] * 10) *
                    cos(angle * 2 + neuromodulators[1] * 8)
                val px = cx + radius * deform * cos(angle)
                val py = cy + radius * deform * sin(angle)
                if (s == 0) path.moveTo(px, py) else path.lineTo(px, py)
            }
            path.close()
            canvas.drawPath(path, glowPaint)
            canvas.restore()
        }

        // === Fractal overlay: Julia set modulated by consciousness ===
        if (abs(consciousnessLevel - fractalConsciousness) > 0.02f || fractalBitmap == null) {
            fractalBitmap = renderJuliaSet(consciousnessLevel, hr, hg, hb)
            fractalConsciousness = consciousnessLevel
        }
        fractalBitmap?.let { bmp ->
            fractalPaint.alpha = (80 + consciousnessLevel * 80).toInt().coerceIn(60, 160)
            val fractalRadius = mandalaRadius * 1.6f * breathScale
            val dst = RectF(
                cx - fractalRadius, cy - fractalRadius,
                cx + fractalRadius, cy + fractalRadius
            )
            canvas.save()
            canvas.rotate(rotationPhase * 60f, cx, cy) // Slow fractal rotation
            canvas.drawBitmap(bmp, null, dst, fractalPaint)
            canvas.restore()
        }

        // === Consciousness number ===
        val numSize = mandalaRadius * 0.5f
        textPaint.textSize = numSize
        textPaint.color = Color.argb(220, 255, 255, 255) // White, not harmony color
        val cText = "%.2f".format(consciousnessLevel)
        canvas.drawText(cText, cx, cy + numSize * 0.3f, textPaint)

        // Trend arrow
        if (trendFilled || trendIndex >= 5) {
            val count = if (trendFilled) trendHistory.size else trendIndex
            val recentAvg = (0 until (count / 2).coerceAtLeast(1)).map {
                trendHistory[((trendIndex - 1 - it) % trendHistory.size + trendHistory.size) % trendHistory.size]
            }.average().toFloat()
            val olderAvg = ((count / 2).coerceAtLeast(1) until count).map {
                trendHistory[((trendIndex - 1 - it) % trendHistory.size + trendHistory.size) % trendHistory.size]
            }.average().toFloat()
            val diff = recentAvg - olderAvg
            val arrow = when {
                diff > 0.005f -> "\u2197"; diff < -0.005f -> "\u2198"; else -> "\u2192"
            }
            val arrowColor = when {
                diff > 0.005f -> Color.parseColor("#5EEAD4")
                diff < -0.005f -> Color.parseColor("#FF6B8A")
                else -> Color.parseColor("#556677")
            }
            textPaint.textSize = numSize * 0.4f; textPaint.color = arrowColor
            val halfW = run { textPaint.textSize = numSize; textPaint.measureText(cText) / 2f }.also { textPaint.textSize = numSize * 0.3f }
            textPaint.textAlign = Paint.Align.LEFT
            canvas.drawText(arrow, cx + halfW + numSize * 0.08f, cy + numSize * 0.3f, textPaint)
            textPaint.textAlign = Paint.Align.CENTER
        }

        // Sparkline — wider, taller, brighter
        if (trendFilled || trendIndex >= 2) {
            val sparkW = mandalaRadius * 0.7f
            val sparkL = cx - sparkW / 2f
            val sparkY = cy + numSize * 0.55f
            val sparkH = numSize * 0.2f
            val count = if (trendFilled) trendHistory.size else trendIndex
            glowPaint.strokeWidth = 2f
            glowPaint.color = Color.argb(140, hr, hg, hb)
            val sp = Path()
            for (i in 0 until count) {
                val idx = ((trendIndex - count + i) % trendHistory.size + trendHistory.size) % trendHistory.size
                val px = sparkL + (i.toFloat() / (count - 1).coerceAtLeast(1)) * sparkW
                val py = sparkY - trendHistory[idx] * sparkH * 2
                if (i == 0) sp.moveTo(px, py) else sp.lineTo(px, py)
            }
            canvas.drawPath(sp, glowPaint)
        }

        // Touch ripple
        if (rippleProgress < 1f) {
            val rr = mandalaRadius * 0.8f * rippleProgress
            val ra = ((1f - rippleProgress) * 150).toInt()
            fillPaint.shader = RadialGradient(rippleX, rippleY, rr.coerceAtLeast(1f),
                Color.argb(ra, hr, hg, hb), Color.TRANSPARENT, Shader.TileMode.CLAMP)
            canvas.drawCircle(rippleX, rippleY, rr, fillPaint)
            fillPaint.shader = null
        }
    }

    /**
     * Render a Julia set at low resolution, colored by harmony.
     *
     * The `c` parameter of the Julia set is derived from consciousness level:
     * - Low consciousness (0.1): c near origin → simple circular boundary
     * - Mid consciousness (0.3-0.5): c at interesting boundary → branching tendrils
     * - High consciousness (0.7+): c deep in fractal → maximum complexity
     *
     * Neuromodulators modulate the imaginary component for organic variation.
     */
    private fun renderJuliaSet(consciousness: Float, hr: Int, hg: Int, hb: Int): android.graphics.Bitmap {
        val bmp = android.graphics.Bitmap.createBitmap(fractalSize, fractalSize, android.graphics.Bitmap.Config.ARGB_8888)

        // Known-beautiful Julia set c values, ordered by visual complexity.
        // Interpolate based on consciousness level for smooth transitions.
        val cValues = arrayOf(
            floatArrayOf(-0.4f,   0.6f),   // Simple spiral (low consciousness)
            floatArrayOf(-0.70f,  0.27f),   // Classic dendrite
            floatArrayOf(-0.8f,   0.156f),  // Branching tree
            floatArrayOf(-0.75f,  0.11f),   // Seahorse valley
            floatArrayOf(0.285f,  0.01f),   // Siegel disk (high consciousness)
            floatArrayOf(-0.12f, -0.77f),   // Lightning bolts (peak)
        )
        val idx = (consciousness * (cValues.size - 1)).coerceIn(0f, (cValues.size - 1).toFloat())
        val lo = idx.toInt().coerceIn(0, cValues.size - 2)
        val frac = idx - lo
        val cr = cValues[lo][0] * (1 - frac) + cValues[lo + 1][0] * frac + neuromodulators[0] * 0.02f
        val ci = cValues[lo][1] * (1 - frac) + cValues[lo + 1][1] * frac + neuromodulators[1] * 0.02f

        val maxIter = (30 + consciousness * 60).toInt()
        val zoom = 1.8f - consciousness * 0.3f

        for (py in 0 until fractalSize) {
            for (px in 0 until fractalSize) {
                // Map pixel to complex plane [-zoom, zoom]
                var zr = (px.toFloat() / fractalSize - 0.5f) * 2f * zoom
                var zi = (py.toFloat() / fractalSize - 0.5f) * 2f * zoom

                var iter = 0
                while (zr * zr + zi * zi < 4f && iter < maxIter) {
                    val tmp = zr * zr - zi * zi + cr
                    zi = 2f * zr * zi + ci
                    zr = tmp
                    iter++
                }

                // Radial mask: fade to transparent at edges
                val dx = (px.toFloat() / fractalSize - 0.5f) * 2f
                val dy = (py.toFloat() / fractalSize - 0.5f) * 2f
                val radialDist = dx * dx + dy * dy
                if (radialDist > 1f) {
                    bmp.setPixel(px, py, Color.TRANSPARENT)
                } else if (iter == maxIter) {
                    // Inside the set — very faint glow
                    val radial = 1f - radialDist
                    val a = (radial * 30).toInt().coerceIn(0, 30)
                    bmp.setPixel(px, py, Color.argb(a, hr, hg, hb))
                } else {
                    // Boundary glow: slow-escaping pixels are brightest (near boundary)
                    val t = iter.toFloat() / maxIter
                    // Inverted: slow escape (high t) = bright boundary
                    val brightness = (1f - t).coerceIn(0f, 1f)
                    val radial = (1f - radialDist).coerceIn(0f, 1f)
                    val a = (brightness * brightness * radial * 255).toInt().coerceIn(0, 255)
                    if (a > 3) {
                        // White-hot boundary fading to harmony color
                        val r = lerp(hr, 255, brightness)
                        val g = lerp(hg, 255, brightness)
                        val b = lerp(hb, 255, brightness)
                        bmp.setPixel(px, py, Color.argb(a, r, g, b))
                    } else {
                        bmp.setPixel(px, py, Color.TRANSPARENT)
                    }
                }
            }
        }
        return bmp
    }

    private fun lerp(a: Int, b: Int, t: Float): Int = (a + (b - a) * t).toInt().coerceIn(0, 255)
}
