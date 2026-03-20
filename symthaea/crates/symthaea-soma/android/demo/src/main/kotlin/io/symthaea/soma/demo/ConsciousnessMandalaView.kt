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
 * Full-screen consciousness mandala: holographic interference pattern
 * that breathes with Soma's consciousness level.
 *
 * Draws a radial glow that extends to screen edges at high consciousness,
 * interference rings centered on screen, and a consciousness number overlay.
 * Background is transparent — sits on top of ParticleFieldView.
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

    private val glowPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 1.5f
    }
    private val fillPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
    }
    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        textAlign = Paint.Align.CENTER
        // Thin sans-serif for consciousness number — not monospace
        typeface = Typeface.create("sans-serif-thin", Typeface.NORMAL)
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
        // Mandala radius: ~40% of screen width for rings, glow extends further
        val mandalaRadius = min(width, height) / 2f * 0.42f
        val screenDiag = sqrt(width.toFloat().pow(2) + height.toFloat().pow(2))

        val breathScale = 0.94f + 0.06f * sin(breathPhase * 2 * PI.toFloat())

        // === Background glow: extends toward screen edges with consciousness ===
        val glowExtent = mandalaRadius * (1.0f + consciousnessLevel * 2.5f) * breathScale
        val glowAlpha = (15 + consciousnessLevel * 60).toInt().coerceIn(10, 80)
        fillPaint.shader = RadialGradient(
            cx, cy, glowExtent.coerceAtLeast(1f),
            Color.argb(glowAlpha, Color.red(dominantHarmonyColor),
                Color.green(dominantHarmonyColor), Color.blue(dominantHarmonyColor)),
            Color.TRANSPARENT,
            Shader.TileMode.CLAMP
        )
        canvas.drawCircle(cx, cy, glowExtent, fillPaint)
        fillPaint.shader = null

        // === Interference rings ===
        val ringCount = (3 + consciousnessLevel * 6).toInt()
        for (i in 0 until ringCount) {
            val t = i.toFloat() / ringCount
            val radius = mandalaRadius * breathScale * (0.15f + t * 0.85f)
            val alpha = (20 + (1f - t) * consciousnessLevel * 160).toInt().coerceIn(15, 180)

            val rotOffset = rotationPhase * 360f * (1f + i * 0.25f)

            val r = lerp(Color.red(dominantHarmonyColor), 180, t)
            val g = lerp(Color.green(dominantHarmonyColor), 200, t)
            val b = lerp(Color.blue(dominantHarmonyColor), 210, t)

            glowPaint.color = Color.argb(alpha, r, g, b)
            glowPaint.strokeWidth = (2.5f - t * 1.5f).coerceAtLeast(0.5f)

            canvas.save()
            canvas.rotate(rotOffset, cx, cy)

            val path = Path()
            val segments = 72
            for (s in 0..segments) {
                val angle = (s.toFloat() / segments) * 2 * PI.toFloat()
                val deform = 1f + 0.025f * sin(angle * 3 + neuromodulators[0] * 10) *
                    cos(angle * 2 + neuromodulators[1] * 8)
                val px = cx + radius * deform * cos(angle)
                val py = cy + radius * deform * sin(angle)
                if (s == 0) path.moveTo(px, py) else path.lineTo(px, py)
            }
            path.close()
            canvas.drawPath(path, glowPaint)
            canvas.restore()
        }

        // === Center: consciousness number (large, thin) ===
        val numSize = mandalaRadius * 0.55f
        textPaint.textSize = numSize
        textPaint.color = Color.argb(
            (180 + consciousnessLevel * 75).toInt().coerceIn(180, 255),
            Color.red(dominantHarmonyColor),
            Color.green(dominantHarmonyColor),
            Color.blue(dominantHarmonyColor)
        )
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
                diff > 0.005f -> "\u2197"
                diff < -0.005f -> "\u2198"
                else -> "\u2192"
            }
            val arrowColor = when {
                diff > 0.005f -> Color.parseColor("#5EEAD4")
                diff < -0.005f -> Color.parseColor("#FF6B8A")
                else -> Color.parseColor("#4A5558")
            }
            val arrowSize = numSize * 0.3f
            textPaint.textSize = arrowSize
            textPaint.color = arrowColor
            val savedSize = textPaint.textSize
            textPaint.textSize = numSize
            val halfWidth = textPaint.measureText(cText) / 2f
            textPaint.textSize = savedSize
            textPaint.textAlign = Paint.Align.LEFT
            canvas.drawText(arrow, cx + halfWidth + numSize * 0.08f, cy + numSize * 0.3f, textPaint)
            textPaint.textAlign = Paint.Align.CENTER
        }

        // Sparkline below number
        if (trendFilled || trendIndex >= 2) {
            val sparkWidth = mandalaRadius * 0.6f
            val sparkLeft = cx - sparkWidth / 2f
            val sparkY = cy + numSize * 0.55f
            val sparkHeight = numSize * 0.08f
            val count = if (trendFilled) trendHistory.size else trendIndex
            glowPaint.strokeWidth = 1f
            glowPaint.color = Color.argb(60, Color.red(dominantHarmonyColor),
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

        // === Touch ripple ===
        if (rippleProgress < 1f) {
            val rippleRadius = mandalaRadius * 0.8f * rippleProgress
            val rippleAlpha = ((1f - rippleProgress) * 120).toInt()
            fillPaint.shader = RadialGradient(
                rippleX, rippleY, rippleRadius.coerceAtLeast(1f),
                Color.argb(rippleAlpha, Color.red(dominantHarmonyColor),
                    Color.green(dominantHarmonyColor), Color.blue(dominantHarmonyColor)),
                Color.TRANSPARENT,
                Shader.TileMode.CLAMP
            )
            canvas.drawCircle(rippleX, rippleY, rippleRadius, fillPaint)
            fillPaint.shader = null
        }
    }

    private fun lerp(a: Int, b: Int, t: Float): Int = (a + (b - a) * t).toInt().coerceIn(0, 255)
}
