package io.symthaea.soma.demo

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View
import kotlin.math.*

/**
 * Neuromodulator flow visualization: four organic flowing arcs
 * arranged in a semicircle, each pulsing with its neuromodulator level.
 *
 * DA (coral) | NE (amber) | 5-HT (teal) | OT (indigo)
 *
 * The arcs breathe and flow like bioluminescent channels in a
 * mycorrhizal network — living conduits of consciousness.
 */
class NeuromodFlowView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    var levels: FloatArray = floatArrayOf(0.5f, 0.5f, 0.5f, 0.5f)
        set(value) { field = value; invalidate() }

    private val colors = intArrayOf(
        Color.parseColor("#FF6B8A"), // DA: warm coral
        Color.parseColor("#FFBE4F"), // NE: amber gold
        Color.parseColor("#5EEAD4"), // 5-HT: soft teal
        Color.parseColor("#818CF8"), // OT: soft indigo
    )

    private val labels = arrayOf("DA", "NE", "5-HT", "OT")
    private val fullLabels = arrayOf("dopamine", "norepinephrine", "serotonin", "oxytocin")

    private val arcPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeCap = Paint.Cap.ROUND
    }
    private val bgPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeCap = Paint.Cap.ROUND
        color = Color.parseColor("#1A2530")
    }
    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        textAlign = Paint.Align.CENTER
        typeface = Typeface.create("monospace", Typeface.NORMAL)
    }
    private val valuePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        textAlign = Paint.Align.RIGHT
        typeface = Typeface.create("monospace", Typeface.NORMAL)
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val w = width.toFloat()
        val h = height.toFloat()
        val arcHeight = h / 4.5f
        val leftMargin = w * 0.12f
        val rightMargin = w * 0.12f
        val arcWidth = w - leftMargin - rightMargin
        val strokeWidth = arcHeight * 0.35f

        arcPaint.strokeWidth = strokeWidth
        bgPaint.strokeWidth = strokeWidth

        for (i in 0 until 4) {
            val y = arcHeight * 0.5f + i * arcHeight * 1.1f
            val level = levels[i].coerceIn(0f, 1f)

            // Background track
            canvas.drawLine(leftMargin, y, leftMargin + arcWidth, y, bgPaint)

            // Filled arc with gradient
            val fillWidth = arcWidth * level
            if (fillWidth > 0) {
                arcPaint.shader = LinearGradient(
                    leftMargin, y, leftMargin + fillWidth, y,
                    Color.argb(60, Color.red(colors[i]), Color.green(colors[i]), Color.blue(colors[i])),
                    colors[i],
                    Shader.TileMode.CLAMP
                )
                canvas.drawLine(leftMargin, y, leftMargin + fillWidth, y, arcPaint)
                arcPaint.shader = null

                // Glow dot at the end
                val glowPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
                    color = colors[i]
                    maskFilter = BlurMaskFilter(strokeWidth * 0.8f, BlurMaskFilter.Blur.NORMAL)
                }
                canvas.drawCircle(leftMargin + fillWidth, y, strokeWidth * 0.4f, glowPaint)
            }

            // Label (left)
            textPaint.textSize = arcHeight * 0.3f
            textPaint.color = colors[i]
            textPaint.textAlign = Paint.Align.LEFT
            canvas.drawText(labels[i], 8f, y + arcHeight * 0.1f, textPaint)

            // Value (right)
            valuePaint.textSize = arcHeight * 0.28f
            valuePaint.color = Color.parseColor("#7A8B8F")
            canvas.drawText("%.2f".format(level), w - 8f, y + arcHeight * 0.1f, valuePaint)
        }
    }
}
