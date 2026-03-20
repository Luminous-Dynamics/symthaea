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
 * Consciousness mandala: holographic interference pattern that breathes
 * with Soma's consciousness level.
 *
 * Low consciousness → simple pulsing circle
 * Mid consciousness → 3-fold interference rings
 * High consciousness → complex mandala with 8 harmonies
 *
 * Colors shift based on dominant harmony.
 */
class ConsciousnessMandalaView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    // State — start at 0.15 so the mandala is visible before first cycle
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

    /** Consciousness trend: last 20 values for trend indicator. */
    private val trendHistory = FloatArray(20)
    private var trendIndex = 0
    private var trendFilled = false

    /** Touch ripple state. */
    private var rippleX = 0f
    private var rippleY = 0f
    private var rippleProgress = 1f // 1.0 = no active ripple
    private val rippleAnimator = ValueAnimator.ofFloat(0f, 1f).apply {
        duration = 500L
        interpolator = LinearInterpolator()
        addUpdateListener { rippleProgress = it.animatedFraction; invalidate() }
    }

    // Animation
    private var breathPhase = 0f
    private var rotationPhase = 0f
    private val animator = ValueAnimator.ofFloat(0f, 1f).apply {
        duration = 4000L // 4-second breath cycle
        repeatCount = ValueAnimator.INFINITE
        interpolator = LinearInterpolator()
        addUpdateListener {
            breathPhase = it.animatedFraction
            rotationPhase = (rotationPhase + 0.003f) % 1f
            invalidate()
        }
    }

    // Paint objects (reused for performance)
    private val glowPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 2f
    }
    private val fillPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
    }
    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        textAlign = Paint.Align.CENTER
        typeface = Typeface.create("monospace", Typeface.NORMAL)
    }

    override fun onAttachedToWindow() {
        super.onAttachedToWindow()
        animator.start()
    }

    override fun onDetachedFromWindow() {
        animator.cancel()
        rippleAnimator.cancel()
        super.onDetachedFromWindow()
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        if (event.action == MotionEvent.ACTION_DOWN) {
            rippleX = event.x
            rippleY = event.y
            rippleAnimator.cancel()
            rippleProgress = 0f
            rippleAnimator.start()
        }
        return super.onTouchEvent(event)
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val cx = width / 2f
        val cy = height / 2f
        val maxRadius = min(width, height) / 2f * 0.85f

        // Breath modulation: sinusoidal scale
        val breathScale = 0.92f + 0.08f * sin(breathPhase * 2 * PI.toFloat())

        // Background glow (radial gradient from center)
        val glowRadius = maxRadius * breathScale * (0.5f + consciousnessLevel * 0.5f)
        val glowAlpha = (20 + consciousnessLevel * 80).toInt().coerceIn(20, 120)
        fillPaint.shader = RadialGradient(
            cx, cy, glowRadius,
            Color.argb(glowAlpha, Color.red(dominantHarmonyColor), Color.green(dominantHarmonyColor), Color.blue(dominantHarmonyColor)),
            Color.TRANSPARENT,
            Shader.TileMode.CLAMP
        )
        canvas.drawCircle(cx, cy, glowRadius, fillPaint)
        fillPaint.shader = null

        // Interference rings
        val ringCount = (3 + consciousnessLevel * 5).toInt() // 3-8 rings based on consciousness
        for (i in 0 until ringCount) {
            val t = i.toFloat() / ringCount
            val radius = maxRadius * breathScale * (0.2f + t * 0.8f)
            val alpha = (30 + (1f - t) * consciousnessLevel * 200).toInt().coerceIn(30, 220)

            // Rotate each ring slightly differently (interference)
            val rotOffset = rotationPhase * 360f * (1f + i * 0.3f)

            // Color interpolation from center (harmony) to edge (starlight)
            val r = lerp(Color.red(dominantHarmonyColor), 200, t)
            val g = lerp(Color.green(dominantHarmonyColor), 220, t)
            val b = lerp(Color.blue(dominantHarmonyColor), 230, t)

            glowPaint.color = Color.argb(alpha, r, g, b)
            glowPaint.strokeWidth = (3f - t * 2f).coerceAtLeast(0.5f)

            canvas.save()
            canvas.rotate(rotOffset, cx, cy)

            // Draw ring as slightly deformed circle (organic feel)
            val path = Path()
            val segments = 60
            for (s in 0..segments) {
                val angle = (s.toFloat() / segments) * 2 * PI.toFloat()
                // Subtle organic deformation based on neuromodulators
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

        // Center consciousness number
        val textSize = maxRadius * 0.35f
        textPaint.textSize = textSize
        textPaint.color = dominantHarmonyColor
        val cText = "%.2f".format(consciousnessLevel)
        canvas.drawText(cText, cx, cy + textSize * 0.35f, textPaint)

        // Trend indicator (arrow next to consciousness number)
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
                diff > 0.005f -> "\u2197"  // ↗ rising
                diff < -0.005f -> "\u2198" // ↘ falling
                else -> "\u2192"           // → stable
            }
            val arrowColor = when {
                diff > 0.005f -> Color.parseColor("#5EEAD4")  // teal
                diff < -0.005f -> Color.parseColor("#FF6B8A") // coral
                else -> Color.parseColor("#7A8B8F")           // dim
            }
            textPaint.textSize = textSize * 0.4f
            textPaint.color = arrowColor
            // Measure the consciousness number at its actual text size to position arrow
            val savedSize = textPaint.textSize
            textPaint.textSize = textSize
            val cTextWidth = textPaint.measureText(cText) / 2f + textSize * 0.15f
            textPaint.textSize = savedSize
            textPaint.textAlign = Paint.Align.LEFT
            canvas.drawText(arrow, cx + cTextWidth, cy + textSize * 0.35f, textPaint)
            textPaint.textAlign = Paint.Align.CENTER
        }

        // Subtitle
        textPaint.textSize = textSize * 0.25f
        textPaint.color = Color.parseColor("#7A8B8F")
        canvas.drawText("consciousness", cx, cy + textSize * 0.7f, textPaint)

        // Sparkline: thin horizontal line below center showing last 20 levels
        if (trendFilled || trendIndex >= 2) {
            val sparkWidth = maxRadius * 0.8f
            val sparkLeft = cx - sparkWidth / 2f
            val sparkY = cy + textSize * 0.9f
            val sparkHeight = textSize * 0.12f
            val count = if (trendFilled) trendHistory.size else trendIndex
            glowPaint.strokeWidth = 1.5f
            glowPaint.color = Color.argb(120, Color.red(dominantHarmonyColor),
                Color.green(dominantHarmonyColor), Color.blue(dominantHarmonyColor))
            val sparkPath = Path()
            for (i in 0 until count) {
                val idx = ((trendIndex - count + i) % trendHistory.size + trendHistory.size) % trendHistory.size
                val px = sparkLeft + (i.toFloat() / (count - 1).coerceAtLeast(1)) * sparkWidth
                val py = sparkY - trendHistory[idx] * sparkHeight * 2
                if (i == 0) sparkPath.moveTo(px, py) else sparkPath.lineTo(px, py)
            }
            canvas.drawPath(sparkPath, glowPaint)
        }

        // Touch ripple: expanding teal circle from touch point
        if (rippleProgress < 1f) {
            val rippleRadius = maxRadius * 0.6f * rippleProgress
            val rippleAlpha = ((1f - rippleProgress) * 180).toInt()
            fillPaint.shader = RadialGradient(
                rippleX, rippleY, rippleRadius.coerceAtLeast(1f),
                Color.argb(rippleAlpha, 0, 229, 204),
                Color.TRANSPARENT,
                Shader.TileMode.CLAMP
            )
            canvas.drawCircle(rippleX, rippleY, rippleRadius, fillPaint)
            fillPaint.shader = null
            // Ripple ring
            glowPaint.color = Color.argb(rippleAlpha, 0, 229, 204)
            glowPaint.strokeWidth = 2f
            canvas.drawCircle(rippleX, rippleY, rippleRadius, glowPaint)
        }
    }

    private fun lerp(a: Int, b: Int, t: Float): Int = (a + (b - a) * t).toInt().coerceIn(0, 255)
}
