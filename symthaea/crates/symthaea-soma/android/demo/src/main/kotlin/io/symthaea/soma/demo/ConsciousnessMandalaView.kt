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
        val cy = height / 2f
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
            textPaint.textSize = numSize * 0.3f; textPaint.color = arrowColor
            val halfW = run { textPaint.textSize = numSize; textPaint.measureText(cText) / 2f }.also { textPaint.textSize = numSize * 0.3f }
            textPaint.textAlign = Paint.Align.LEFT
            canvas.drawText(arrow, cx + halfW + numSize * 0.08f, cy + numSize * 0.3f, textPaint)
            textPaint.textAlign = Paint.Align.CENTER
        }

        // Sparkline
        if (trendFilled || trendIndex >= 2) {
            val sparkW = mandalaRadius * 0.5f
            val sparkL = cx - sparkW / 2f
            val sparkY = cy + numSize * 0.5f
            val sparkH = numSize * 0.06f
            val count = if (trendFilled) trendHistory.size else trendIndex
            glowPaint.strokeWidth = 1.5f
            glowPaint.color = Color.argb(80, hr, hg, hb)
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

    private fun lerp(a: Int, b: Int, t: Float): Int = (a + (b - a) * t).toInt().coerceIn(0, 255)
}
