package io.symthaea.soma.demo

import android.animation.ValueAnimator
import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View
import android.view.animation.LinearInterpolator
import kotlin.math.*

/**
 * Living consciousness garden: mycelium root network + harmony ring + dream nodes.
 *
 * Solarpunk identity: branching root networks grow with each dream,
 * harmony nodes pulse at compass points, session timeline marks milestones.
 * The visual language is organic growth, not abstract geometry.
 */
class ConsciousnessGardenView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    var consciousnessLevel: Float = 0f
    var dominantHarmonyIndex: Int = 0
    var harmonyStrengths: FloatArray = FloatArray(8) { 0.3f }
    var dreamCount: Int = 0
    var sessionProgress: Float = 0f
    var milestoneMarkers: MutableList<Float> = mutableListOf()
    var primaryColor: Int = Color.parseColor("#00E5CC")

    private var breathPhase: Float = 0f
    private var growthPhase: Float = 0f
    private val animator = ValueAnimator.ofFloat(0f, 1f).apply {
        duration = 5000L
        repeatCount = ValueAnimator.INFINITE
        interpolator = LinearInterpolator()
        addUpdateListener {
            breathPhase = it.animatedFraction
            growthPhase += 0.001f  // Very slow organic growth
            invalidate()
        }
    }

    override fun onAttachedToWindow() { super.onAttachedToWindow(); animator.start() }
    override fun onDetachedFromWindow() { animator.cancel(); super.onDetachedFromWindow() }

    private val linePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE; strokeCap = Paint.Cap.ROUND
    }
    private val dotPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.FILL }

    private val harmonyColors = intArrayOf(
        Color.parseColor("#00E5CC"), Color.parseColor("#47D4FF"),
        Color.parseColor("#9B7DFF"), Color.parseColor("#FF7EB3"),
        Color.parseColor("#FFD166"), Color.parseColor("#FF8C42"),
        Color.parseColor("#6BCB77"), Color.parseColor("#C4B5FD"),
    )

    // Root network: pre-seeded branching points for up to 20 dreams
    // Each dream adds a new root branch from the central stem
    private data class RootNode(val x: Float, val y: Float, val angle: Float, val depth: Int)

    override fun onDraw(canvas: Canvas) {
        if (width == 0 || height == 0) return
        val w = width.toFloat()
        val h = height.toFloat()
        val pulse = 0.5f + 0.5f * sin(breathPhase * 2 * PI.toFloat())
        val pr = Color.red(primaryColor)
        val pg = Color.green(primaryColor)
        val pb = Color.blue(primaryColor)

        // === Central root stem: horizontal line across the garden ===
        val stemY = h * 0.45f
        val stemLeft = w * 0.05f
        val stemRight = w * 0.95f
        val stemAlpha = (20 + consciousnessLevel * 25).toInt().coerceIn(15, 45)
        linePaint.strokeWidth = 1.5f
        linePaint.color = Color.argb(stemAlpha, pr, pg, pb)
        canvas.drawLine(stemLeft, stemY, stemRight, stemY, linePaint)

        // === Mycelium root branches: one per dream, growing from the stem ===
        val branchSpacing = (stemRight - stemLeft) / 21f
        for (i in 0 until dreamCount.coerceAtMost(20)) {
            val branchX = stemLeft + branchSpacing * (i + 1)
            val growUp = i % 2 == 0  // Alternate up/down
            val branchLen = h * 0.25f * (0.5f + consciousnessLevel * 0.5f)

            // Main branch
            val tipY = if (growUp) stemY - branchLen else stemY + branchLen
            val branchAlpha = (25 + consciousnessLevel * 30).toInt().coerceIn(15, 55)

            // Organic curve: quadratic bezier with wobble
            val ctrlX = branchX + sin(growthPhase + i * 1.3f) * 15f
            val ctrlY = (stemY + tipY) / 2f

            val path = Path()
            path.moveTo(branchX, stemY)
            path.quadTo(ctrlX, ctrlY, branchX + sin(growthPhase * 0.7f + i) * 8f, tipY)

            linePaint.strokeWidth = 1.2f
            linePaint.color = Color.argb(branchAlpha, pr, pg, pb)
            canvas.drawPath(path, linePaint)

            // Sub-branches (2-3 forks per main branch)
            val forks = (1 + consciousnessLevel * 2).toInt().coerceIn(1, 3)
            for (f in 0 until forks) {
                val forkT = (f + 1f) / (forks + 1f)
                val forkX = branchX + (ctrlX - branchX) * forkT * 0.5f
                val forkY = stemY + (tipY - stemY) * forkT
                val forkAngle = if (f % 2 == 0) 0.8f else -0.8f
                val forkLen = branchLen * 0.2f
                val forkEndX = forkX + forkLen * cos(forkAngle + growthPhase * 0.3f + i * 0.5f)
                val forkEndY = forkY + forkLen * sin(forkAngle) * (if (growUp) -1f else 1f)

                linePaint.strokeWidth = 0.8f
                linePaint.color = Color.argb((branchAlpha * 0.6f).toInt(), pr, pg, pb)
                canvas.drawLine(forkX, forkY, forkEndX, forkEndY, linePaint)
            }

            // Dream node at tip: glowing dot
            val twinkle = 0.6f + 0.4f * sin(breathPhase * 2 * PI.toFloat() + i * 1.3f)
            val nodeAlpha = ((40 + consciousnessLevel * 60) * twinkle).toInt().coerceIn(15, 120)
            val nodeSize = 2.5f + consciousnessLevel * 2f

            // Glow
            dotPaint.color = Color.argb((nodeAlpha * 0.3f).toInt(), pr, pg, pb)
            canvas.drawCircle(branchX + sin(growthPhase * 0.7f + i) * 8f, tipY, nodeSize * 2.5f, dotPaint)
            // Core
            dotPaint.color = Color.argb(nodeAlpha, 200, 230, 255)
            canvas.drawCircle(branchX + sin(growthPhase * 0.7f + i) * 8f, tipY, nodeSize, dotPaint)
        }

        // === Session timeline + milestones (bottom edge) ===
        val tlY = h * 0.88f
        val tlLeft = w * 0.08f
        val tlRight = w * 0.92f
        linePaint.strokeWidth = 0.8f
        linePaint.color = Color.argb(20, 255, 255, 255)
        canvas.drawLine(tlLeft, tlY, tlRight, tlY, linePaint)

        // Progress
        val progressX = tlLeft + sessionProgress * (tlRight - tlLeft)
        dotPaint.color = Color.argb(60, pr, pg, pb)
        canvas.drawCircle(progressX, tlY, 2.5f, dotPaint)

        // Milestone markers
        for (marker in milestoneMarkers) {
            val mx = tlLeft + marker * (tlRight - tlLeft)
            dotPaint.color = Color.argb(100, 255, 255, 255)
            canvas.drawCircle(mx, tlY, 2f, dotPaint)
        }
    }
}
